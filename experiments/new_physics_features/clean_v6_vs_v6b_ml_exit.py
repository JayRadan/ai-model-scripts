"""
CLEAN v6 vs v6b — strict global time cutoff, WITH ML exit.

Single global cutoff (2024-12-12). Train both confirmation models AND the exit
model on data strictly before the cutoff. Backtest with ML exit on data after.

No per-rule leakage. Full architecture (not fixed TP/SL).
"""
from __future__ import annotations
import glob, os, time as _time
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

# Exit model always uses 11 features (same for v6 and v6b — exit is global, not per-rule)
EXIT_FEATS = [
    "unrealized_pnl_R", "bars_held", "pnl_velocity",
    "hurst_rs", "ou_theta", "entropy_rate", "kramers_up", "wavelet_er",
    "quantum_flow", "quantum_flow_h4", "vwap_dist",
]

MAX_HOLD = 60
MIN_HOLD = 2
SL_HARD = 4.0
EXIT_THRESHOLD = 0.55
MAX_FWD_EXIT = 60


def load_swing_with_physics():
    """Load swing + compute physics features (same as 04b) at every bar."""
    print("  Loading swing CSV + computing physics at every bar...", flush=True)
    swing = pd.read_csv(P.data("swing_v5_xauusd.csv"), parse_dates=["time"])
    swing = swing.sort_values("time").reset_index(drop=True)
    H = swing["high"].values.astype(np.float64)
    L = swing["low"].values.astype(np.float64)
    C = swing["close"].values.astype(np.float64)
    O = swing["open"].values.astype(np.float64)
    vol = np.maximum(swing["spread"].values.astype(np.float64), 1.0)

    tr = np.concatenate([[H[0]-L[0]],
          np.maximum.reduce([H[1:]-L[1:], np.abs(H[1:]-C[:-1]), np.abs(L[1:]-C[:-1])])])
    atr = pd.Series(tr).rolling(14, min_periods=14).mean().values
    ret = np.concatenate([[0.0], np.diff(np.log(C))])

    from importlib.machinery import SourceFileLoader
    mod = SourceFileLoader("p04b", "/home/jay/Desktop/new-model-zigzag/model_pipeline/04b_compute_physics_features.py").load_module()
    swing["hurst_rs"]     = mod.compute_hurst_rs(ret)
    swing["ou_theta"]     = mod.compute_ou_theta(ret)
    swing["entropy_rate"] = mod.compute_entropy(ret)
    swing["kramers_up"]   = mod.compute_kramers_up(C)
    swing["wavelet_er"]   = mod.compute_wavelet_er(C)
    swing["vwap_dist"]    = mod.compute_vwap_dist(swing, atr)
    swing["quantum_flow"] = mod.compute_quantum_flow(O, H, L, C, vol)
    df_h4 = swing.set_index("time")[["open","high","low","close","spread"]].resample("4h").agg(
        {"open":"first","high":"max","low":"min","close":"last","spread":"sum"}).dropna()
    qf_h4 = mod.compute_quantum_flow(df_h4["open"].values, df_h4["high"].values,
                                      df_h4["low"].values, df_h4["close"].values,
                                      np.maximum(df_h4["spread"].values, 1.0))
    qf_h4_s = pd.Series(qf_h4, index=df_h4.index).shift(1)
    swing["quantum_flow_h4"] = qf_h4_s.reindex(swing.set_index("time").index, method="ffill").values
    for col in EXIT_FEATS[3:]:
        swing[col] = swing[col].fillna(0)
    return swing, atr


def load_and_split():
    all_setups = []
    for f in sorted(glob.glob(P.data("setups_*_v6b.csv"))):
        cid = int(os.path.basename(f).split("_")[1])
        df = pd.read_csv(f, parse_dates=["time"])
        df["cid"] = cid
        all_setups.append(df)
    all_df = pd.concat(all_setups, ignore_index=True).sort_values("time").reset_index(drop=True)
    train = all_df[all_df["time"] < GLOBAL_CUTOFF].reset_index(drop=True)
    test = all_df[all_df["time"] >= GLOBAL_CUTOFF].reset_index(drop=True)
    print(f"  Cutoff: {GLOBAL_CUTOFF}")
    print(f"  Train: {len(train):,}  Test: {len(test):,}")
    return train, test


def train_confirmation_models(train, features, tag):
    """Per-rule XGBoost models with threshold swept on LAST 20% of train as validation."""
    models = {}
    thresholds = {}
    disabled = 0
    for (cid, rule), grp in train.groupby(["cid", "rule"]):
        if len(grp) < 100:
            disabled += 1; continue
        grp = grp.sort_values("time").reset_index(drop=True)
        split_idx = int(len(grp) * 0.80)
        tr = grp.iloc[:split_idx]
        vd = grp.iloc[split_idx:]
        if len(vd) < 20:
            disabled += 1; continue
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
            disabled += 1; continue
        models[(cid, rule)] = mdl
        thresholds[(cid, rule)] = best_thr
    print(f"  {tag}: {len(models)} active confirmation models, {disabled} disabled")
    return models, thresholds


def confirm_setups(setups, models, thresholds, features):
    """Apply confirmation models to a batch of setups."""
    rows = []
    for (cid, rule), grp in setups.groupby(["cid", "rule"]):
        if (cid, rule) not in models: continue
        mdl = models[(cid, rule)]; thr = thresholds[(cid, rule)]
        X = grp[features].fillna(0).values
        proba = mdl.predict_proba(X)[:, 1]
        rows.append(grp[proba >= thr].copy())
    return pd.concat(rows, ignore_index=True).sort_values("time").reset_index(drop=True) if rows else pd.DataFrame()


def train_exit_model(train_confirmed, swing, atr, tag):
    """Build exit-bar-by-bar rows from train_confirmed, train XGBoost exit classifier."""
    time_to_idx = pd.Series(swing.index.values, index=swing["time"].values)
    C = swing["close"].values.astype(np.float64)
    exit_rows = []
    for _, setup in train_confirmed.iterrows():
        t = setup["time"]
        if t not in time_to_idx.index: continue
        entry_idx = int(time_to_idx[t])
        direction = int(setup["direction"])
        entry_price = C[entry_idx]; entry_atr = atr[entry_idx]
        if not np.isfinite(entry_atr) or entry_atr <= 0: continue
        end_idx = min(entry_idx + MAX_FWD_EXIT + 1, len(C))
        if end_idx - entry_idx < 10: continue
        pnls_R = []
        for k in range(entry_idx+1, end_idx):
            pnls_R.append(direction * (C[k] - entry_price) / entry_atr)
        pnls_R = np.array(pnls_R)
        if len(pnls_R) < 5: continue
        for b in range(len(pnls_R)):
            bar_idx = entry_idx + 1 + b
            cur_pnl = pnls_R[b]
            if cur_pnl < -SL_HARD: break
            remaining = pnls_R[b+1:] if b+1 < len(pnls_R) else np.array([cur_pnl])
            best_rem = remaining.max() if len(remaining) > 0 else cur_pnl
            margin = 0.3
            if best_rem < cur_pnl - margin: label = 1
            elif best_rem > cur_pnl + margin: label = 0
            else: continue
            vel = cur_pnl - pnls_R[b-3] if b >= 3 else (cur_pnl - pnls_R[0] if b >= 1 else 0.0)
            row = {"unrealized_pnl_R": cur_pnl, "bars_held": float(b+1), "pnl_velocity": vel, "label": label}
            for feat in EXIT_FEATS[3:]:
                row[feat] = float(swing[feat].iat[bar_idx]) if bar_idx < len(swing) else 0.0
            exit_rows.append(row)
    exit_df = pd.DataFrame(exit_rows)
    print(f"  {tag} exit model: {len(exit_df):,} training rows  "
          f"(label=1: {exit_df['label'].sum():,}  label=0: {(1-exit_df['label']).sum():,})")
    X = exit_df[EXIT_FEATS].fillna(0).values
    y = exit_df["label"].values
    exit_mdl = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", verbosity=0,
    )
    exit_mdl.fit(X, y)
    return exit_mdl


def simulate_with_ml_exit(confirmed, swing, atr, exit_mdl, tag):
    time_to_idx = pd.Series(swing.index.values, index=swing["time"].values)
    C = swing["close"].values.astype(np.float64)
    n = len(C)
    trades = []
    for _, setup in confirmed.iterrows():
        t = setup["time"]
        if t not in time_to_idx.index: continue
        entry_idx = int(time_to_idx[t])
        direction = int(setup["direction"])
        entry_price = C[entry_idx]; entry_atr = atr[entry_idx]
        if not np.isfinite(entry_atr) or entry_atr <= 0: continue
        exit_idx = None; exit_reason = "max"
        for k in range(1, MAX_HOLD+1):
            bar = entry_idx + k
            if bar >= n: break
            cur_pnl = direction * (C[bar] - entry_price) / entry_atr
            if cur_pnl < -SL_HARD:
                exit_idx = bar; exit_reason = "hard_sl"; break
            if k >= MIN_HOLD:
                pnl_3ago = direction * (C[bar-3] - entry_price) / entry_atr if k >= 3 else cur_pnl
                row_d = {
                    "unrealized_pnl_R": cur_pnl, "bars_held": float(k),
                    "pnl_velocity": cur_pnl - pnl_3ago,
                    "hurst_rs": swing["hurst_rs"].iat[bar],
                    "ou_theta": swing["ou_theta"].iat[bar],
                    "entropy_rate": swing["entropy_rate"].iat[bar],
                    "kramers_up": swing["kramers_up"].iat[bar],
                    "wavelet_er": swing["wavelet_er"].iat[bar],
                    "quantum_flow": swing["quantum_flow"].iat[bar],
                    "quantum_flow_h4": swing["quantum_flow_h4"].iat[bar],
                    "vwap_dist": swing["vwap_dist"].iat[bar],
                }
                X_ = np.array([[row_d[ef] for ef in EXIT_FEATS]])
                p_exit = exit_mdl.predict_proba(X_)[0, 1]
                if p_exit >= EXIT_THRESHOLD:
                    exit_idx = bar; exit_reason = "ml_exit"; break
        if exit_idx is None:
            exit_idx = min(entry_idx + MAX_HOLD, n-1); exit_reason = "max"
        pnl_R = direction * (C[exit_idx] - entry_price) / entry_atr
        trades.append({"time": t, "cid": int(setup["cid"]), "rule": setup["rule"],
                       "bars": exit_idx - entry_idx, "pnl_R": pnl_R, "exit": exit_reason})
    df = pd.DataFrame(trades)
    if len(df) == 0:
        print(f"  {tag}: NO TRADES"); return None
    wins = df[df["pnl_R"] > 0]; losses = df[df["pnl_R"] <= 0]
    pf = wins["pnl_R"].sum() / max(-losses["pnl_R"].sum(), 1e-9)
    wr = len(wins) / len(df)
    eq = df["pnl_R"].cumsum().values
    max_dd = (np.maximum.accumulate(eq) - eq).max() if len(eq) > 0 else 0
    print(f"\n  {tag}: n={len(df)}  WR={wr:.1%}  PF={pf:.2f}  "
          f"Expect={df['pnl_R'].mean():+.3f}R  Total={df['pnl_R'].sum():+.1f}R  MaxDD=-{max_dd:.1f}R")
    print(f"         Exits: {df['exit'].value_counts().to_dict()}")
    print(f"         Per-cluster:")
    for cid, grp in df.groupby("cid"):
        w = grp[grp["pnl_R"]>0]; ll = grp[grp["pnl_R"]<=0]
        ppf = w["pnl_R"].sum() / max(-ll["pnl_R"].sum(), 1e-9)
        print(f"           C{cid}: n={len(grp):,} WR={len(w)/len(grp):.1%} PF={ppf:.2f} R={grp['pnl_R'].sum():+.1f}")
    return {"n": len(df), "pf": pf, "wr": wr, "expectancy": df["pnl_R"].mean(),
            "total_R": df["pnl_R"].sum(), "max_dd": max_dd}


def main():
    print("CLEAN v6 vs v6b — global time cutoff + ML exit (full architecture)")
    print("=" * 70)

    train, test = load_and_split()
    swing, atr = load_swing_with_physics()

    # ---------- v6 (14 feats) ----------
    print("\nTraining v6 (14 features)...")
    t0 = _time.time()
    v6_mdls, v6_thrs = train_confirmation_models(train, OLD_FEATS, "v6")
    print(f"  conf train {_time.time()-t0:.0f}s")
    # Exit model on v6-confirmed train trades
    t0 = _time.time()
    v6_train_conf = confirm_setups(train, v6_mdls, v6_thrs, OLD_FEATS)
    print(f"  v6: {len(v6_train_conf):,} confirmed train trades")
    v6_exit_mdl = train_exit_model(v6_train_conf, swing, atr, "v6")
    print(f"  exit train {_time.time()-t0:.0f}s")

    # ---------- v6b (21 feats) ----------
    print("\nTraining v6b (21 features)...")
    t0 = _time.time()
    v6b_mdls, v6b_thrs = train_confirmation_models(train, ALL_FEATS, "v6b")
    print(f"  conf train {_time.time()-t0:.0f}s")
    t0 = _time.time()
    v6b_train_conf = confirm_setups(train, v6b_mdls, v6b_thrs, ALL_FEATS)
    print(f"  v6b: {len(v6b_train_conf):,} confirmed train trades")
    v6b_exit_mdl = train_exit_model(v6b_train_conf, swing, atr, "v6b")
    print(f"  exit train {_time.time()-t0:.0f}s")

    # ---------- Backtest ----------
    print("\n" + "=" * 70)
    print("CLEAN HOLDOUT BACKTEST (ML exit)")
    print("=" * 70)

    v6_conf_test = confirm_setups(test, v6_mdls, v6_thrs, OLD_FEATS)
    v6b_conf_test = confirm_setups(test, v6b_mdls, v6b_thrs, ALL_FEATS)
    print(f"  v6 confirmed in holdout: {len(v6_conf_test):,}")
    print(f"  v6b confirmed in holdout: {len(v6b_conf_test):,}")

    v6_res = simulate_with_ml_exit(v6_conf_test, swing, atr, v6_exit_mdl, "v6")
    v6b_res = simulate_with_ml_exit(v6b_conf_test, swing, atr, v6b_exit_mdl, "v6b")

    if v6_res and v6b_res:
        print(f"\n{'='*70}")
        print("COMPARISON — clean holdout, ML exit (no per-rule leakage)")
        print(f"{'='*70}")
        print(f"{'Metric':<18} {'v6 (14)':>12} {'v6b (21)':>12} {'Delta':>10}")
        print(f"{'Trades':<18} {v6_res['n']:>12} {v6b_res['n']:>12} {v6b_res['n']-v6_res['n']:>+10}")
        print(f"{'Win rate':<18} {v6_res['wr']*100:>11.1f}% {v6b_res['wr']*100:>11.1f}% {(v6b_res['wr']-v6_res['wr'])*100:>+9.1f}pt")
        print(f"{'Profit factor':<18} {v6_res['pf']:>12.2f} {v6b_res['pf']:>12.2f} {v6b_res['pf']-v6_res['pf']:>+10.2f}")
        print(f"{'Expectancy R':<18} {v6_res['expectancy']:>+12.3f} {v6b_res['expectancy']:>+12.3f} {v6b_res['expectancy']-v6_res['expectancy']:>+10.3f}")
        print(f"{'Total PnL R':<18} {v6_res['total_R']:>+12.1f} {v6b_res['total_R']:>+12.1f} {v6b_res['total_R']-v6_res['total_R']:>+10.1f}")
        print(f"{'Max DD R':<18} {-v6_res['max_dd']:>+12.1f} {-v6b_res['max_dd']:>+12.1f} {v6_res['max_dd']-v6b_res['max_dd']:>+10.1f}")


if __name__ == "__main__":
    main()
