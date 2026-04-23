"""
v6 Midas XAU validation — 14 features, NO meta-label gate.

Analogous to experiments/v72_lite_btc_deploy/01_validate_v72_lite_btc.py, but:
  • 14-feature v6 confirmation stack (no vpin / sig_quad_var / har_rv_ratio / hawkes_eta)
  • XAU data (swing_v5_xauusd.csv)
  • NO meta-label XGB — v6 fires directly on per-rule confirmation threshold
  • Per-rule 80/20 chronological split (matches 05b_train_v6_all.py)

Purpose: produce the EXACT XGBClassifier objects (26 confirm heads + 1 exit head)
that reproduce the validated v6 holdout backtest. These objects are pickled by
02_pickle_v6_validated.py for the FastAPI decision engine.

Reference backtest (data/backtest_v6_summary.json): n=1650 PF=1.34 WR=48%.
"""
from __future__ import annotations
import glob, os, time as _time
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import sys
sys.path.insert(0, "/home/jay/Desktop/new-model-zigzag/model_pipeline")
import paths as P

V6_CONFIRM_FEATS = [
    "hurst_rs", "ou_theta", "entropy_rate", "kramers_up", "wavelet_er",
    "vwap_dist", "hour_enc", "dow_enc",
    "quantum_flow", "quantum_flow_h4", "quantum_momentum", "quantum_vwap_conf",
    "quantum_divergence", "quantum_div_strength",
]

EXIT_FEATS = [
    "unrealized_pnl_R", "bars_held", "pnl_velocity",
    "hurst_rs", "ou_theta", "entropy_rate", "kramers_up", "wavelet_er",
    "quantum_flow", "quantum_flow_h4", "vwap_dist",
]

MAX_HOLD = 60; MIN_HOLD = 2; SL_HARD = 4.0
EXIT_THRESHOLD = 0.55; MAX_FWD_EXIT = 60


def load_swing_with_physics():
    print("  Loading XAU swing + physics...", flush=True)
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
    for col in EXIT_FEATS[3:]: swing[col] = swing[col].fillna(0)
    return swing, atr


def load_and_split():
    """Per-rule chronological 80/20 split — matches 05b_train_v6_all.py."""
    all_train, all_test = [], []
    for f in sorted(glob.glob(P.data("setups_*_v6.csv"))):
        base = os.path.basename(f)
        if "_btc" in base: continue   # skip BTC setups; this is XAU
        cid = int(base.split("_")[1])
        df = pd.read_csv(f, parse_dates=["time"])
        df["cid"] = cid
        for rule, grp in df.groupby("rule"):
            rdf = grp.sort_values("time").reset_index(drop=True)
            if len(rdf) < 100: continue
            s = int(len(rdf) * 0.80)
            all_train.append(rdf.iloc[:s])
            all_test.append(rdf.iloc[s:])
    tr = pd.concat(all_train, ignore_index=True).sort_values("time").reset_index(drop=True)
    te = pd.concat(all_test, ignore_index=True).sort_values("time").reset_index(drop=True)
    print(f"  Per-rule 80/20 split  Train: {len(tr):,}  Test: {len(te):,}")
    return tr, te


def train_conf(train, features, tag):
    """Per-(cid, rule) XGB — matches 05b hyperparams exactly."""
    mdls, thrs = {}, {}
    disabled = 0
    for (cid, rule), grp in train.groupby(["cid", "rule"]):
        if len(grp) < 100: disabled += 1; continue
        grp = grp.sort_values("time").reset_index(drop=True)
        s = int(len(grp) * 0.80)
        tr, vd = grp.iloc[:s], grp.iloc[s:]
        if len(vd) < 20: disabled += 1; continue
        mdl = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8,
                            eval_metric="logloss", verbosity=0)
        mdl.fit(tr[features].fillna(0).values, tr["label"].astype(int).values)
        proba = mdl.predict_proba(vd[features].fillna(0).values)[:, 1]
        y_vd = vd["label"].astype(int).values
        best_thr, best_pf = 0.5, 0
        for thr in np.arange(0.30, 0.70, 0.05):
            m = proba >= thr
            if m.sum() < 5: continue
            w = y_vd[m].sum(); lo = m.sum() - w
            if lo == 0: continue
            pf = (w * 2.0) / (lo * 1.0)
            if pf > best_pf: best_pf, best_thr = pf, float(thr)
        m = proba >= best_thr
        if m.sum() < 5 or best_pf < 0.8: disabled += 1; continue
        mdls[(cid, rule)] = mdl; thrs[(cid, rule)] = best_thr
    print(f"  {tag}: {len(mdls)} active, {disabled} disabled")
    return mdls, thrs


def confirm(setups, mdls, thrs, features):
    rows = []
    for (cid, rule), grp in setups.groupby(["cid", "rule"]):
        if (cid, rule) not in mdls: continue
        X = grp[features].fillna(0).values
        p = mdls[(cid, rule)].predict_proba(X)[:, 1]
        rows.append(grp[p >= thrs[(cid, rule)]].copy())
    return (pd.concat(rows, ignore_index=True).sort_values("time").reset_index(drop=True)
            if rows else pd.DataFrame())


def train_exit(train_conf_df, swing, atr):
    time_to_idx = pd.Series(swing.index.values, index=swing["time"].values)
    C = swing["close"].values.astype(np.float64)
    rows = []
    for _, s in train_conf_df.iterrows():
        t = s["time"]
        if t not in time_to_idx.index: continue
        ei = int(time_to_idx[t])
        d = int(s["direction"])
        ep = C[ei]; ea = atr[ei]
        if not np.isfinite(ea) or ea <= 0: continue
        end = min(ei + MAX_FWD_EXIT + 1, len(C))
        if end - ei < 10: continue
        pnls = np.array([d * (C[k] - ep) / ea for k in range(ei+1, end)])
        if len(pnls) < 5: continue
        for b in range(len(pnls)):
            bi = ei + 1 + b
            cp = pnls[b]
            if cp < -SL_HARD: break
            rem = pnls[b+1:] if b+1 < len(pnls) else np.array([cp])
            br = rem.max() if len(rem) > 0 else cp
            if br < cp - 0.3: lbl = 1
            elif br > cp + 0.3: lbl = 0
            else: continue
            v = cp - pnls[b-3] if b >= 3 else (cp - pnls[0] if b >= 1 else 0.0)
            row = {"unrealized_pnl_R": cp, "bars_held": float(b+1), "pnl_velocity": v, "label": lbl}
            for f in EXIT_FEATS[3:]:
                row[f] = float(swing[f].iat[bi]) if bi < len(swing) else 0.0
            rows.append(row)
    ed = pd.DataFrame(rows)
    mdl = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        eval_metric="logloss", verbosity=0)
    mdl.fit(ed[EXIT_FEATS].fillna(0).values, ed["label"].values)
    return mdl


def simulate(confirmed, swing, atr, exit_mdl):
    """Batched forward sim with ML exit + 4R hard SL."""
    time_to_idx = pd.Series(swing.index.values, index=swing["time"].values)
    C = swing["close"].values.astype(np.float64); n = len(C)
    ctx_cols = ["hurst_rs","ou_theta","entropy_rate","kramers_up","wavelet_er",
                "quantum_flow","quantum_flow_h4","vwap_dist"]
    ctx_arr = swing[ctx_cols].fillna(0).values.astype(np.float64)

    entries = []
    for _, s in confirmed.iterrows():
        t = s["time"]
        if t not in time_to_idx.index: continue
        ei = int(time_to_idx[t])
        ea = atr[ei]
        if not np.isfinite(ea) or ea <= 0: continue
        entries.append((ei, int(s["direction"]), t, int(s["cid"]), s["rule"]))

    N = len(entries)
    if N == 0: return pd.DataFrame()

    n_feats = 3 + len(ctx_cols)
    X = np.zeros((N * MAX_HOLD, n_feats), dtype=np.float32)
    valid = np.zeros(N * MAX_HOLD, dtype=bool)
    cps = np.full((N, MAX_HOLD), np.nan, dtype=np.float64)

    for rank, (ei, d, _, _, _) in enumerate(entries):
        ep = C[ei]; ea = atr[ei]
        for k in range(1, MAX_HOLD + 1):
            bar = ei + k
            if bar >= n: break
            cp = d * (C[bar] - ep) / ea
            cps[rank, k-1] = cp
            if k < MIN_HOLD: continue
            p3 = d * (C[bar-3] - ep) / ea if k >= 3 else cp
            row = rank * MAX_HOLD + (k - 1)
            X[row, 0] = cp; X[row, 1] = float(k); X[row, 2] = cp - p3
            X[row, 3:] = ctx_arr[bar]
            valid[row] = True

    probs = np.zeros(N * MAX_HOLD, dtype=np.float32)
    if valid.any():
        probs[valid] = exit_mdl.predict_proba(X[valid])[:, 1]

    rows = []
    for rank, (ei, d, t, cid_v, rule_v) in enumerate(entries):
        ep = C[ei]
        xi, xr = None, "max"
        for k in range(1, MAX_HOLD + 1):
            bar = ei + k
            if bar >= n: break
            cp = cps[rank, k-1]
            if not np.isfinite(cp): break
            if cp < -SL_HARD: xi, xr = bar, "hard_sl"; break
            if k >= MIN_HOLD:
                if probs[rank * MAX_HOLD + (k-1)] >= EXIT_THRESHOLD:
                    xi, xr = bar, "ml_exit"; break
        if xi is None:
            xi = min(ei + MAX_HOLD, n-1); xr = "max"
        pnl = d * (C[xi] - ep) / atr[ei]
        rows.append({"time": t, "cid": cid_v, "rule": rule_v,
                     "direction": d, "bars": xi - ei, "pnl_R": pnl, "exit": xr})
    return pd.DataFrame(rows)


def report(df, tag):
    if len(df) == 0: print(f"  {tag}: NO TRADES"); return None
    w = df[df["pnl_R"] > 0]; l = df[df["pnl_R"] <= 0]
    pf = w["pnl_R"].sum() / max(-l["pnl_R"].sum(), 1e-9)
    wr = len(w) / len(df)
    eq = df["pnl_R"].cumsum().values
    dd = (np.maximum.accumulate(eq) - eq).max() if len(eq) > 0 else 0
    print(f"\n  {tag}: n={len(df)}  WR={wr:.1%}  PF={pf:.2f}  "
          f"Expect={df['pnl_R'].mean():+.3f}R  Total={df['pnl_R'].sum():+.1f}R  MaxDD=-{dd:.1f}R")
    for cid, grp in df.groupby("cid"):
        ww = grp[grp["pnl_R"]>0]; ll = grp[grp["pnl_R"]<=0]
        ppf = ww["pnl_R"].sum() / max(-ll["pnl_R"].sum(), 1e-9)
        print(f"    C{cid}: n={len(grp):,} WR={len(ww)/len(grp):.1%} PF={ppf:.2f} R={grp['pnl_R'].sum():+.1f}")
    return {"n": len(df), "pf": pf, "wr": wr, "dd": dd, "total": df["pnl_R"].sum()}


def main():
    print("="*78)
    print("v6 VALIDATION — XAU, 14 features, NO meta gate")
    print("="*78)
    train, test = load_and_split()
    swing, atr = load_swing_with_physics()

    print("\n[1/2] Training v6 confirmations + exit...")
    t0 = _time.time()
    mdls, thrs = train_conf(train, V6_CONFIRM_FEATS, "v6-conf")
    print(f"  conf train {_time.time()-t0:.0f}s")
    tc = confirm(train, mdls, thrs, V6_CONFIRM_FEATS)
    print(f"  {len(tc):,} confirmed train setups")
    t0 = _time.time()
    exit_mdl = train_exit(tc, swing, atr)
    print(f"  exit train {_time.time()-t0:.0f}s")

    print("\n[2/2] Holdout backtest")
    tec = confirm(test, mdls, thrs, V6_CONFIRM_FEATS)
    print(f"  v6 confirmed in holdout: {len(tec):,}")
    t_trades = simulate(tec, swing, atr, exit_mdl)
    r = report(t_trades, "v6 (no meta)")

    out_trades = "/home/jay/Desktop/new-model-zigzag/data/v6_trades_holdout_xau.csv"
    t_trades.to_csv(out_trades, index=False)
    print(f"  Dumped {len(t_trades):,} holdout trades to {out_trades}")

    if r:
        print(f"\n{'='*78}")
        print("v6 holdout vs recorded backtest_v6_summary.json")
        print(f"  recorded ref: n=1650  WR=48.2%  PF=1.34  DD=-109R")
        print(f"  this run   : n={r['n']}  WR={r['wr']:.1%}  PF={r['pf']:.2f}  DD=-{r['dd']:.1f}R")
        print(f"{'='*78}")

    # Stash the trained objects so step 02 can pickle them without retraining.
    import pickle
    out_pkl_tmp = "/home/jay/Desktop/new-model-zigzag/experiments/v6_xau_deploy/_v6_validated_raw.pkl"
    with open(out_pkl_tmp, "wb") as f:
        pickle.dump({
            "confirm_mdls": mdls,          # {(cid, rule): XGBClassifier}
            "confirm_thrs": thrs,          # {(cid, rule): float}
            "exit_mdl":     exit_mdl,      # XGBClassifier
            "confirm_feats": V6_CONFIRM_FEATS,
            "exit_feats":    EXIT_FEATS,
            "exit_threshold": EXIT_THRESHOLD,
            "hard_sl":        SL_HARD,
            "min_hold":       MIN_HOLD,
            "max_hold":       MAX_HOLD,
            "holdout":        r,
        }, f)
    print(f"\n  Saved raw validated objects to {out_pkl_tmp}")


if __name__ == "__main__":
    main()
