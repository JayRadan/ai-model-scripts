"""
CLEAN comparison — v6 (14 feats) vs v6b (21 feats) with a single global time cutoff.

All rules split on the SAME date (not per-rule 80/20). No rule can have a train
cutoff past the global holdout start.

Uses 2024-12-12 as the cutoff (same as existing "holdout quantile" for comparability).
"""
from __future__ import annotations
import glob, json, os, time as _time
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import sys
sys.path.insert(0, "/home/jay/Desktop/new-model-zigzag/model_pipeline")
import paths as P

GLOBAL_CUTOFF = pd.Timestamp("2024-12-12 00:00:00")

OLD_FEATS = [
    "hurst_rs", "ou_theta", "entropy_rate", "kramers_up", "wavelet_er",
    "vwap_dist", "hour_enc", "dow_enc",
    "quantum_flow", "quantum_flow_h4", "quantum_momentum", "quantum_vwap_conf",
    "quantum_divergence", "quantum_div_strength",
]
NEW_FEATS = ["permutation_entropy", "dfa_alpha", "higuchi_fd",
             "spectral_entropy", "hill_tail_index", "vol_of_vol", "log_drift"]
ALL_FEATS = OLD_FEATS + NEW_FEATS

MAX_HOLD = 60
MIN_HOLD = 2
SL_HARD = 4.0
EXIT_THRESHOLD = 0.55


def load_and_split():
    """Load all setups from v6b CSVs. Split by global cutoff date."""
    all_setups = []
    for f in sorted(glob.glob(P.data("setups_*_v6b.csv"))):
        cid = int(os.path.basename(f).split("_")[1])
        df = pd.read_csv(f, parse_dates=["time"])
        df["cid"] = cid
        all_setups.append(df)
    all_df = pd.concat(all_setups, ignore_index=True).sort_values("time").reset_index(drop=True)
    train = all_df[all_df["time"] < GLOBAL_CUTOFF].reset_index(drop=True)
    test = all_df[all_df["time"] >= GLOBAL_CUTOFF].reset_index(drop=True)
    print(f"  Global cutoff: {GLOBAL_CUTOFF}")
    print(f"  Train: {len(train):,} setups ({train['time'].iat[0]} to {train['time'].iat[-1]})")
    print(f"  Test:  {len(test):,} setups ({test['time'].iat[0]} to {test['time'].iat[-1]})")
    return train, test


def train_rule_models(train, features, tag):
    """For each (cid, rule), train one XGBoost model on the given feature set."""
    models = {}
    thresholds = {}
    disabled = set()
    for (cid, rule), grp in train.groupby(["cid", "rule"]):
        if len(grp) < 100:
            disabled.add((cid, rule))
            continue
        # Use last 20% of TRAIN as per-rule validation for threshold sweep
        grp = grp.sort_values("time").reset_index(drop=True)
        split_idx = int(len(grp) * 0.80)
        tr = grp.iloc[:split_idx]
        vd = grp.iloc[split_idx:]
        if len(vd) < 20:
            disabled.add((cid, rule))
            continue
        y_tr = tr["label"].astype(int).values
        y_vd = vd["label"].astype(int).values
        X_tr = tr[features].fillna(0).values
        X_vd = vd[features].fillna(0).values
        mdl = XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", verbosity=0,
        )
        mdl.fit(X_tr, y_tr)
        proba = mdl.predict_proba(X_vd)[:, 1]
        best_thr = 0.5; best_pf = 0
        for thr in np.arange(0.30, 0.70, 0.05):
            mask = proba >= thr
            if mask.sum() < 5: continue
            w = y_vd[mask].sum(); lo = mask.sum() - w
            if lo == 0: continue
            pf = (w * 2.0) / (lo * 1.0)
            if pf > best_pf:
                best_pf = pf; best_thr = float(thr)
        mask = proba >= best_thr
        if mask.sum() < 5 or best_pf < 0.8:
            disabled.add((cid, rule))
            continue
        models[(cid, rule)] = mdl
        thresholds[(cid, rule)] = best_thr
    print(f"  {tag}: trained {len(models)} active models, {len(disabled)} disabled", flush=True)
    return models, thresholds


def backtest(test, models, thresholds, features, tag):
    """Run forward-simulation backtest on the test period."""
    swing = pd.read_csv(P.data("swing_v5_xauusd.csv"), parse_dates=["time"])
    swing = swing.sort_values("time").reset_index(drop=True)
    c = swing["close"].values.astype(np.float64)
    h = swing["high"].values.astype(np.float64)
    l = swing["low"].values.astype(np.float64)
    tr = np.concatenate([[h[0]-l[0]],
          np.maximum.reduce([h[1:]-l[1:], np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])])])
    atr = pd.Series(tr).rolling(14, min_periods=14).mean().values
    time_to_idx = pd.Series(swing.index.values, index=swing["time"].values)

    # Confirm test setups
    confirmed_rows = []
    for (cid, rule), grp in test.groupby(["cid", "rule"]):
        if (cid, rule) not in models: continue
        mdl = models[(cid, rule)]; thr = thresholds[(cid, rule)]
        X = grp[features].fillna(0).values
        proba = mdl.predict_proba(X)[:, 1]
        mask = proba >= thr
        confirmed_rows.append(grp[mask].copy())
    if not confirmed_rows:
        print(f"  {tag}: NO TRADES")
        return None
    confirmed = pd.concat(confirmed_rows, ignore_index=True).sort_values("time").reset_index(drop=True)

    # Simulate with SIMPLE fixed TP/SL (no ML exit — removes one confounder)
    trades = []
    TP_R = 3.0; SL_R = 1.5  # fixed 2:1 R multiple, representative
    n = len(c)
    for _, setup in confirmed.iterrows():
        t = setup["time"]
        if t not in time_to_idx.index: continue
        entry_idx = int(time_to_idx[t])
        direction = int(setup["direction"])
        entry_price = c[entry_idx]; entry_atr = atr[entry_idx]
        if not np.isfinite(entry_atr) or entry_atr <= 0: continue
        tp_price = entry_price + direction * TP_R * entry_atr
        sl_price = entry_price - direction * SL_R * entry_atr
        exit_idx = None; pnl_R = None
        for k in range(1, MAX_HOLD + 1):
            bar = entry_idx + k
            if bar >= n: break
            if direction > 0:
                if l[bar] <= sl_price:
                    exit_idx = bar; pnl_R = -SL_R; break
                if h[bar] >= tp_price:
                    exit_idx = bar; pnl_R = TP_R; break
            else:
                if h[bar] >= sl_price:
                    exit_idx = bar; pnl_R = -SL_R; break
                if l[bar] <= tp_price:
                    exit_idx = bar; pnl_R = TP_R; break
        if exit_idx is None:
            # Close at MAX_HOLD by market price
            bar = min(entry_idx + MAX_HOLD, n - 1)
            pnl_R = direction * (c[bar] - entry_price) / entry_atr
        trades.append({"time": t, "cid": int(setup["cid"]), "rule": setup["rule"], "pnl_R": pnl_R})

    df = pd.DataFrame(trades)
    wins = df[df["pnl_R"] > 0]; losses = df[df["pnl_R"] <= 0]
    pf = wins["pnl_R"].sum() / max(-losses["pnl_R"].sum(), 1e-9)
    wr = len(wins) / max(len(df), 1)
    eq = df["pnl_R"].cumsum().values
    max_dd = (np.maximum.accumulate(eq) - eq).max() if len(eq) > 0 else 0
    print(f"\n  {tag}: n={len(df)}  WR={wr:.1%}  PF={pf:.2f}  Expectancy={df['pnl_R'].mean():+.3f}R  "
          f"Total={df['pnl_R'].sum():+.1f}R  MaxDD=-{max_dd:.1f}R")
    print(f"         Per-cluster:")
    for cid, grp in df.groupby("cid"):
        w = grp[grp["pnl_R"] > 0]; ll = grp[grp["pnl_R"] <= 0]
        ppf = w["pnl_R"].sum() / max(-ll["pnl_R"].sum(), 1e-9)
        print(f"           C{cid}: n={len(grp):,} WR={len(w)/len(grp):.1%} PF={ppf:.2f}")
    return {"n": len(df), "pf": pf, "wr": wr, "expectancy": df["pnl_R"].mean(),
            "total_R": df["pnl_R"].sum(), "max_dd": max_dd}


def main():
    print("CLEAN v6 vs v6b — single global time cutoff, no per-rule leakage")
    print("Fixed TP=3R / SL=1.5R exits (no ML exit — removes one confounder)")
    print("=" * 70)

    train, test = load_and_split()
    print()

    print("\nTraining v6 (14 existing features only)...")
    t0 = _time.time()
    v6_models, v6_thrs = train_rule_models(train, OLD_FEATS, "v6")
    print(f"  ({_time.time() - t0:.0f}s)")

    print("\nTraining v6b (21 features = 14 + 7 new)...")
    t0 = _time.time()
    v6b_models, v6b_thrs = train_rule_models(train, ALL_FEATS, "v6b")
    print(f"  ({_time.time() - t0:.0f}s)")

    print("\n" + "=" * 70)
    print("CLEAN HOLDOUT RESULTS")
    print("=" * 70)

    v6_res = backtest(test, v6_models, v6_thrs, OLD_FEATS, "v6")
    v6b_res = backtest(test, v6b_models, v6b_thrs, ALL_FEATS, "v6b")

    if v6_res and v6b_res:
        print(f"\n{'='*70}")
        print("COMPARISON (clean holdout, fixed TP/SL)")
        print(f"{'='*70}")
        print(f"{'Metric':<18} {'v6 (14 feats)':>15} {'v6b (21 feats)':>17} {'Delta':>10}")
        print(f"{'Trades':<18} {v6_res['n']:>15}   {v6b_res['n']:>15}   {v6b_res['n'] - v6_res['n']:>+10}")
        print(f"{'Win rate':<18} {v6_res['wr']*100:>14.1f}%   {v6b_res['wr']*100:>14.1f}%   {(v6b_res['wr']-v6_res['wr'])*100:>+9.1f}pt")
        print(f"{'Profit factor':<18} {v6_res['pf']:>15.2f}   {v6b_res['pf']:>15.2f}   {v6b_res['pf']-v6_res['pf']:>+10.2f}")
        print(f"{'Expectancy (R)':<18} {v6_res['expectancy']:>+15.3f}   {v6b_res['expectancy']:>+15.3f}   {v6b_res['expectancy']-v6_res['expectancy']:>+10.3f}")
        print(f"{'Total PnL (R)':<18} {v6_res['total_R']:>+15.1f}   {v6b_res['total_R']:>+15.1f}   {v6b_res['total_R']-v6_res['total_R']:>+10.1f}")
        print(f"{'Max DD (R)':<18} {-v6_res['max_dd']:>+15.1f}   {-v6b_res['max_dd']:>+15.1f}   {v6_res['max_dd']-v6b_res['max_dd']:>+10.1f}")


if __name__ == "__main__":
    main()
