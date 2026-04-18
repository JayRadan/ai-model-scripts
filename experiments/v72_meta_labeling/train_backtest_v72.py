"""
v7.2 — Meta-labeling on top of v7.1 (López de Prado, Advances in Financial ML ch.3).

Architecture:
  Layer 1 — v7.1 per-rule confirmation model (unchanged).
  Layer 2 — SINGLE global XGBoost meta-model that takes v7.1-confirmed setups
            and predicts: "will this trade be a WINNER (pnl_R > 0)?"
  Trade only if P_meta(win) >= meta_threshold.

Critical guardrail — no leakage:
  • Meta model trains on TRAIN segment only (time < GLOBAL_CUTOFF).
  • Meta labels come from simulating the ML exit on train-confirmed setups
    using the exit model trained on those same train-confirmed setups.
    (Same exit model as used in evaluation — same pipeline.)
  • Meta threshold selected on the LAST 20% of train (chronological), NOT on
    holdout.  Holdout is never touched until the final eval.
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
V71_EXTRA = ["vpin", "sig_quad_var", "bocpd_recent_cp",
             "kyle_lambda", "har_rv_ratio", "hawkes_eta"]
V71_FEATS = OLD_FEATS + V71_EXTRA                                  # 20

# Meta features: v7.1 feature set + direction + cid (regime) + original rule proba surrogate
META_FEATS = V71_FEATS + ["direction", "cid"]

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
    print("  Loading swing + computing physics at every bar...", flush=True)
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
    for f in sorted(glob.glob(P.data("setups_*_v71.csv"))):
        cid = int(os.path.basename(f).split("_")[1])
        df = pd.read_csv(f, parse_dates=["time"])
        df["cid"] = cid
        all_setups.append(df)
    all_df = pd.concat(all_setups, ignore_index=True).sort_values("time").reset_index(drop=True)
    train = all_df[all_df["time"] < GLOBAL_CUTOFF].reset_index(drop=True)
    test = all_df[all_df["time"] >= GLOBAL_CUTOFF].reset_index(drop=True)
    print(f"  Cutoff: {GLOBAL_CUTOFF}   Train: {len(train):,}   Test: {len(test):,}")
    return train, test


def train_confirmation_models(train, features, tag):
    models, thresholds = {}, {}
    disabled = 0
    for (cid, rule), grp in train.groupby(["cid", "rule"]):
        if len(grp) < 100: disabled += 1; continue
        grp = grp.sort_values("time").reset_index(drop=True)
        split_idx = int(len(grp) * 0.80)
        tr, vd = grp.iloc[:split_idx], grp.iloc[split_idx:]
        if len(vd) < 20: disabled += 1; continue
        y_tr = tr["label"].astype(int).values
        y_vd = vd["label"].astype(int).values
        X_tr = tr[features].fillna(0).values
        X_vd = vd[features].fillna(0).values
        mdl = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8,
                            eval_metric="logloss", verbosity=0)
        mdl.fit(X_tr, y_tr)
        proba = mdl.predict_proba(X_vd)[:, 1]
        best_thr, best_pf = 0.5, 0
        for thr in np.arange(0.30, 0.70, 0.05):
            mask = proba >= thr
            if mask.sum() < 5: continue
            w = y_vd[mask].sum(); lo = mask.sum() - w
            if lo == 0: continue
            pf = (w * 2.0) / (lo * 1.0)
            if pf > best_pf: best_pf, best_thr = pf, float(thr)
        mask = proba >= best_thr
        if mask.sum() < 5 or best_pf < 0.8: disabled += 1; continue
        models[(cid, rule)] = mdl
        thresholds[(cid, rule)] = best_thr
    print(f"  {tag}: {len(models)} active, {disabled} disabled")
    return models, thresholds


def confirm_setups(setups, models, thresholds, features):
    rows = []
    for (cid, rule), grp in setups.groupby(["cid", "rule"]):
        if (cid, rule) not in models: continue
        mdl, thr = models[(cid, rule)], thresholds[(cid, rule)]
        X = grp[features].fillna(0).values
        proba = mdl.predict_proba(X)[:, 1]
        rows.append(grp[proba >= thr].copy())
    return (pd.concat(rows, ignore_index=True).sort_values("time").reset_index(drop=True)
            if rows else pd.DataFrame())


def train_exit_model(train_confirmed, swing, atr):
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
            row = {"unrealized_pnl_R": cur_pnl, "bars_held": float(b+1),
                   "pnl_velocity": vel, "label": label}
            for feat in EXIT_FEATS[3:]:
                row[feat] = float(swing[feat].iat[bar_idx]) if bar_idx < len(swing) else 0.0
            exit_rows.append(row)
    exit_df = pd.DataFrame(exit_rows)
    X = exit_df[EXIT_FEATS].fillna(0).values
    y = exit_df["label"].values
    exit_mdl = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             eval_metric="logloss", verbosity=0)
    exit_mdl.fit(X, y)
    return exit_mdl


def simulate_with_ml_exit(confirmed, swing, atr, exit_mdl, return_trades=True):
    """Simulate and return per-trade pnl_R — used both for eval AND meta training labels."""
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
        exit_idx, exit_reason = None, "max"
        for k in range(1, MAX_HOLD+1):
            bar = entry_idx + k
            if bar >= n: break
            cur_pnl = direction * (C[bar] - entry_price) / entry_atr
            if cur_pnl < -SL_HARD: exit_idx, exit_reason = bar, "hard_sl"; break
            if k >= MIN_HOLD:
                pnl_3ago = (direction * (C[bar-3] - entry_price) / entry_atr
                            if k >= 3 else cur_pnl)
                X_ = np.array([[cur_pnl, float(k), cur_pnl - pnl_3ago,
                                swing["hurst_rs"].iat[bar], swing["ou_theta"].iat[bar],
                                swing["entropy_rate"].iat[bar], swing["kramers_up"].iat[bar],
                                swing["wavelet_er"].iat[bar], swing["quantum_flow"].iat[bar],
                                swing["quantum_flow_h4"].iat[bar], swing["vwap_dist"].iat[bar]]])
                if exit_mdl.predict_proba(X_)[0, 1] >= EXIT_THRESHOLD:
                    exit_idx, exit_reason = bar, "ml_exit"; break
        if exit_idx is None:
            exit_idx = min(entry_idx + MAX_HOLD, n-1); exit_reason = "max"
        pnl_R = direction * (C[exit_idx] - entry_price) / entry_atr
        trades.append({"time": t, "cid": int(setup["cid"]), "rule": setup["rule"],
                       "direction": int(setup["direction"]),
                       "bars": exit_idx - entry_idx, "pnl_R": pnl_R, "exit": exit_reason})
    return pd.DataFrame(trades)


def report(df, tag):
    if len(df) == 0: print(f"  {tag}: NO TRADES"); return None
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
            "total_R": df["pnl_R"].sum(), "max_dd": max_dd, "trades": df}


def main():
    print("=" * 78)
    print("v7.2 — meta-labeling (López de Prado) on top of v7.1")
    print("Global cutoff 2024-12-12, ML exit, no per-rule leakage")
    print("=" * 78)

    train, test = load_and_split()
    swing, atr = load_swing_with_physics()

    # ---------- Train v7.1 confirmation + exit (base) ----------
    print("\n[1/3] Training v7.1 base models...")
    t0 = _time.time()
    mdls, thrs = train_confirmation_models(train, V71_FEATS, "v7.1-conf")
    print(f"  conf train {_time.time()-t0:.0f}s")
    train_conf = confirm_setups(train, mdls, thrs, V71_FEATS)
    print(f"  {len(train_conf):,} confirmed train setups")
    t0 = _time.time()
    exit_mdl = train_exit_model(train_conf, swing, atr)
    print(f"  exit train {_time.time()-t0:.0f}s")

    # ---------- Generate meta-training rows by SIMULATING train-confirmed trades ----------
    print("\n[2/3] Generating meta-training data (simulating train trades with ML exit)...")
    t0 = _time.time()
    train_trades = simulate_with_ml_exit(train_conf, swing, atr, exit_mdl)
    print(f"  simulated {len(train_trades):,} train trades in {_time.time()-t0:.0f}s")
    # Merge simulated trade outcome with its v7.1 features
    train_conf["direction"] = train_conf["direction"].astype(int)
    train_conf["cid"] = train_conf["cid"].astype(int)
    meta_df = train_trades.merge(
        train_conf[["time", "cid", "rule"] + V71_FEATS],
        on=["time", "cid", "rule"], how="left")
    meta_df["meta_label"] = (meta_df["pnl_R"] > 0).astype(int)
    print(f"  meta rows: {len(meta_df):,}  WR of raw train trades: {meta_df['meta_label'].mean():.1%}")

    # Train meta-model on first 80% of meta_df (chronological); hold last 20% for threshold sweep
    meta_df = meta_df.sort_values("time").reset_index(drop=True)
    split = int(len(meta_df) * 0.80)
    mtr = meta_df.iloc[:split]
    mvd = meta_df.iloc[split:]
    X_mtr = mtr[META_FEATS].fillna(0).values
    y_mtr = mtr["meta_label"].values
    X_mvd = mvd[META_FEATS].fillna(0).values
    y_mvd = mvd["meta_label"].values
    print(f"  meta train: {len(mtr):,}  meta validation: {len(mvd):,}")
    t0 = _time.time()
    meta_mdl = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             eval_metric="logloss", verbosity=0)
    meta_mdl.fit(X_mtr, y_mtr)
    print(f"  meta model trained in {_time.time()-t0:.0f}s")

    # ---------- Select meta threshold on the LAST 20% of TRAIN (never holdout) ----------
    proba_vd = meta_mdl.predict_proba(X_mvd)[:, 1]
    pnls_vd = mvd["pnl_R"].values
    baseline_n = len(y_mvd)
    baseline_wr = y_mvd.mean()
    baseline_pf = (pnls_vd[pnls_vd > 0].sum() / max(-pnls_vd[pnls_vd <= 0].sum(), 1e-9))
    print(f"\n  Meta-validation baseline (no filter): n={baseline_n}  WR={baseline_wr:.1%}  PF={baseline_pf:.2f}")
    print(f"  Threshold sweep (target: higher WR, keep PF, retain ≥50% of trades):")
    candidates = []
    for thr in np.arange(0.40, 0.80, 0.025):
        mask = proba_vd >= thr
        if mask.sum() < 50: continue
        wr_t = y_mvd[mask].mean()
        pnls_t = pnls_vd[mask]
        pf_t = (pnls_t[pnls_t > 0].sum() / max(-pnls_t[pnls_t <= 0].sum(), 1e-9))
        retain = mask.sum() / baseline_n
        print(f"    thr={thr:.3f}  n={mask.sum():>4}  WR={wr_t:.1%}  PF={pf_t:.2f}  retain={retain:.1%}")
        candidates.append((thr, wr_t, pf_t, mask.sum(), retain))
    # Pick threshold: max WR subject to PF ≥ baseline AND retain ≥ 0.50
    valid = [c for c in candidates if c[2] >= baseline_pf * 0.95 and c[4] >= 0.50]
    if not valid:
        valid = [c for c in candidates if c[2] >= baseline_pf * 0.90]
    if not valid:
        valid = candidates
    valid.sort(key=lambda c: (-c[1], -c[2]))
    best_thr = valid[0][0] if valid else 0.50
    print(f"  Selected meta threshold: {best_thr:.3f}  (WR={valid[0][1]:.1%}, PF={valid[0][2]:.2f}, retain={valid[0][4]:.1%})")

    # ---------- Backtest on holdout ----------
    print("\n[3/3] Holdout backtest")
    test_conf = confirm_setups(test, mdls, thrs, V71_FEATS)
    print(f"  v7.1 confirmed in holdout: {len(test_conf):,}")
    v71_trades = simulate_with_ml_exit(test_conf, swing, atr, exit_mdl)
    v71_res = report(v71_trades, "v7.1  (no meta)")

    # Apply meta filter
    test_conf["direction"] = test_conf["direction"].astype(int)
    test_conf["cid"] = test_conf["cid"].astype(int)
    X_test_meta = test_conf[META_FEATS].fillna(0).values
    meta_proba = meta_mdl.predict_proba(X_test_meta)[:, 1]
    test_conf_meta = test_conf[meta_proba >= best_thr].copy()
    print(f"\n  v7.2 confirmed after meta filter: {len(test_conf_meta):,}  "
          f"(dropped {len(test_conf)-len(test_conf_meta):,} = {(1-len(test_conf_meta)/max(len(test_conf),1)):.1%})")
    v72_trades = simulate_with_ml_exit(test_conf_meta, swing, atr, exit_mdl)
    v72_res = report(v72_trades, "v7.2  (meta-labeled)")

    if v71_res and v72_res:
        print(f"\n{'='*78}")
        print("COMPARISON — clean holdout, ML exit, meta-labeling ON/OFF")
        print(f"{'='*78}")
        print(f"{'Metric':<18} {'v7.1 (20)':>14} {'v7.2 (+meta)':>14} {'Delta':>11}")
        print(f"{'Trades':<18} {v71_res['n']:>14} {v72_res['n']:>14} {v72_res['n']-v71_res['n']:>+11}")
        print(f"{'Win rate':<18} {v71_res['wr']*100:>13.1f}% {v72_res['wr']*100:>13.1f}% {(v72_res['wr']-v71_res['wr'])*100:>+10.1f}pt")
        print(f"{'Profit factor':<18} {v71_res['pf']:>14.2f} {v72_res['pf']:>14.2f} {v72_res['pf']-v71_res['pf']:>+11.2f}")
        print(f"{'Expectancy R':<18} {v71_res['expectancy']:>+14.3f} {v72_res['expectancy']:>+14.3f} {v72_res['expectancy']-v71_res['expectancy']:>+11.3f}")
        print(f"{'Total PnL R':<18} {v71_res['total_R']:>+14.1f} {v72_res['total_R']:>+14.1f} {v72_res['total_R']-v71_res['total_R']:>+11.1f}")
        print(f"{'Max DD R':<18} {-v71_res['max_dd']:>+14.1f} {-v72_res['max_dd']:>+14.1f} {v71_res['max_dd']-v72_res['max_dd']:>+11.1f}")

        out_dir = "/home/jay/Desktop/new-model-zigzag/experiments/v72_meta_labeling"
        v71_trades.to_csv(f"{out_dir}/v71_trades_holdout.csv", index=False)
        v72_trades.to_csv(f"{out_dir}/v72_trades_holdout.csv", index=False)

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(11, 5))
            eq71 = v71_trades["pnl_R"].cumsum().values
            eq72 = v72_trades["pnl_R"].cumsum().values
            ax.plot(pd.to_datetime(v71_trades["time"]), eq71,
                    label=f"v7.1 (20)  PF={v71_res['pf']:.2f}  WR={v71_res['wr']:.1%}  n={v71_res['n']}", lw=1.8)
            ax.plot(pd.to_datetime(v72_trades["time"]), eq72,
                    label=f"v7.2 (meta)  PF={v72_res['pf']:.2f}  WR={v72_res['wr']:.1%}  n={v72_res['n']}", lw=1.8)
            ax.axhline(0, color="k", lw=0.5)
            ax.set_ylabel("Cumulative R")
            ax.set_title("v7.1 vs v7.2 — clean holdout, ML exit, meta-labeling")
            ax.legend(loc="best")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(f"{out_dir}/v71_vs_v72_holdout.png", dpi=110)
            print(f"  Saved: {out_dir}/v71_vs_v72_holdout.png")
        except Exception as e:
            print(f"  Could not save PNG: {e}")


if __name__ == "__main__":
    main()
