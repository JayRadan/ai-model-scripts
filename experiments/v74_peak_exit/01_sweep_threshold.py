"""
v7.4 Peak-Exit — sweep the EXIT_THRESHOLD on the new peak-exit head.

The peak-detector head was trained with a 2.8% positive rate, so it almost
never predicts > 0.55. Sweep the threshold from 0.05 -> 0.55 and find the
sweet spot where ml_exit fires often enough to MATTER but not so often it
exits early.

Reads peak_exit_v74.json from models/. Reads everything else from the
production pickle (read-only).
"""
from __future__ import annotations
import os, sys, pickle, json, copy
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

ZIGZAG = "/home/jay/Desktop/new-model-zigzag"
sys.path.insert(0, os.path.join(ZIGZAG, "model_pipeline"))
sys.path.insert(0, os.path.join(ZIGZAG, "experiments/v74_pivot_score"))
import paths as P
from importlib.machinery import SourceFileLoader
WF = SourceFileLoader("wf", os.path.join(ZIGZAG, "experiments/v74_pivot_score/05_walk_forward.py")).load_module()

OUT_DIR  = os.path.join(ZIGZAG, "experiments/v74_peak_exit")
PKL_PATH = os.path.join(ZIGZAG, "models/janus_xau_validated.pkl")
DATA     = os.path.join(ZIGZAG, "experiments/v74_pivot_score/data")
NEW_MDL_PATH = os.path.join(OUT_DIR, "models", "peak_exit_v74.json")

CUTOFF = pd.Timestamp("2024-12-12 00:00:00")

THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.55]


def simulate_with_threshold(confirmed, swing, atr, exit_mdl, exit_threshold):
    """Same as WF.simulate but with overridable exit_threshold."""
    H = swing["high"].values; L = swing["low"].values; C = swing["close"].values
    n = len(C)
    feats_arr = {col: swing[col].values for col in WF.EXIT_FEATS[3:]}
    entries = []
    for _, s in confirmed.iterrows():
        ei = int(s["idx"])
        if ei + WF.MAX_HOLD >= n: continue
        entries.append((ei, int(s["direction"]), s["time"], int(s["cid"]), s["rule"], float(s["atr"])))
    if not entries: return pd.DataFrame()

    X = np.zeros((len(entries) * WF.MAX_HOLD, len(WF.EXIT_FEATS)))
    valid = np.zeros(len(entries) * WF.MAX_HOLD, dtype=bool)
    for r, (ei, d, t, cid_v, rule_v, a) in enumerate(entries):
        if a <= 0: continue
        sl = WF.SL_HARD * a; entry_px = C[ei]
        end = min(ei + WF.MAX_HOLD + 1, n)
        for k in range(1, end - ei):
            bar = ei + k
            if d == 1: pnl = (C[bar] - entry_px) / sl
            else:      pnl = (entry_px - C[bar]) / sl
            row_idx = r * WF.MAX_HOLD + (k - 1)
            X[row_idx, 0] = pnl; X[row_idx, 1] = k; X[row_idx, 2] = pnl/k
            for fi, f in enumerate(WF.EXIT_FEATS[3:]):
                X[row_idx, 3 + fi] = feats_arr[f][bar]
            valid[row_idx] = True

    probs = np.zeros(len(entries) * WF.MAX_HOLD)
    if valid.any():
        probs[valid] = exit_mdl.predict_proba(X[valid])[:, 1]

    rows = []
    for rank, (ei, d, t, cid_v, rule_v, a) in enumerate(entries):
        if a <= 0: continue
        sl = WF.SL_HARD * a; entry_px = C[ei]
        end = min(ei + WF.MAX_HOLD + 1, n)
        xi, xr = end - 1, "max_hold"; pnl = 0.0
        for k in range(1, end - ei):
            bar = ei + k
            if d == 1:
                if (entry_px - L[bar]) >= sl: xi, xr, pnl = bar, "sl", -1.0; break
                pnl_now = (C[bar] - entry_px) / sl
            else:
                if (H[bar] - entry_px) >= sl: xi, xr, pnl = bar, "sl", -1.0; break
                pnl_now = (entry_px - C[bar]) / sl
            if k > 2 and probs[rank * WF.MAX_HOLD + (k - 1)] >= exit_threshold:
                xi, xr, pnl = bar, "ml_exit", pnl_now; break
            pnl = pnl_now
        rows.append({"time": t, "cid": cid_v, "rule": rule_v, "direction": d,
                      "bars": xi - ei, "pnl_R": pnl, "exit": xr})
    return pd.DataFrame(rows)


def report(df, name):
    if not len(df): return {}
    ww = df[df["pnl_R"] > 0]; ll = df[df["pnl_R"] <= 0]
    pf = (ww["pnl_R"].sum() / -ll["pnl_R"].sum()) if len(ll) and ll["pnl_R"].sum() < 0 else float("inf")
    cum = df["pnl_R"].cumsum().values
    dd = float((cum - np.maximum.accumulate(cum)).min())
    return {"name": name, "n": len(df), "wr": float(len(ww)/len(df)),
            "pf": float(pf), "expect": float(df["pnl_R"].mean()),
            "total_R": float(df["pnl_R"].sum()), "dd": float(dd),
            "avg_bars": float(df["bars"].mean()),
            "exit_dist": df["exit"].value_counts().to_dict()}


def main():
    print("Loading...", flush=True)
    feats = pd.read_csv(os.path.join(DATA, "features_v74.csv"), parse_dates=["time"])
    labs  = pd.read_csv(os.path.join(DATA, "labels_v74.csv"), parse_dates=["time"])
    df = feats.merge(labs, on="time", how="inner", suffixes=("", "_lab"))
    clusters = pd.read_csv(WF.CLUSTERS_PATH, parse_dates=["time"])[["time", "cid"]]
    swing, atr = WF.build_swing_with_exit_feats()
    payload = pickle.load(open(PKL_PATH, "rb"))
    new_mdl = XGBClassifier(); new_mdl.load_model(NEW_MDL_PATH)
    pivot_feats = list(payload["pivot_feats"])

    fold = df[df["time"] >= CUTOFF].reset_index(drop=True)
    p_fold    = payload["pivot_score_mdl"].predict_proba(fold[pivot_feats].fillna(0).values)[:, 1]
    pdir_fold = payload["pivot_dir_mdl"].predict_proba(fold[pivot_feats].fillna(0).values)[:, 1]
    m = p_fold >= payload["score_threshold"]
    fs = fold[m].copy()
    fs["direction"] = np.where(pdir_fold[m] >= 0.5, 1, -1)
    fs["rule"] = "RP_score"; fs["idx"] = fs["bar_idx"]
    fs["entry_price"] = fs["time"].map(swing.set_index("time")["close"])
    fs["label"] = (fs["best_R"] >= 1.0).astype(int)
    fs = fs.merge(clusters, on="time", how="left")

    fc = WF.filter_setups(fs, payload["mdls"], payload["thrs"], WF.V72L_FEATS)
    fc["direction"] = fc["direction"].astype(int); fc["cid"] = fc["cid"].astype(int)
    pm = payload["meta_mdl"].predict_proba(fc[WF.META_FEATS].fillna(0).values)[:, 1]
    fcm = fc[pm >= payload["meta_threshold"]].reset_index(drop=True)
    print(f"Holdout meta-filtered setups: {len(fcm):,}")

    # Baseline (current production exit head, 0.55)
    base = simulate_with_threshold(fcm, swing, atr, payload["exit_mdl"], 0.55)
    rb = report(base, "BASELINE (current exit, thr=0.55)")
    print(f"\n{'thr':>6s} {'n':>5s} {'WR':>6s} {'PF':>6s} {'R':>9s} {'DD':>7s} {'avg_bars':>9s}  exits")
    print(f"{'BASE':>6s} {rb['n']:>5d} {rb['wr']:>5.1%} {rb['pf']:>6.2f} {rb['total_R']:>+9.0f} {rb['dd']:>7.0f} {rb['avg_bars']:>9.1f}  {rb['exit_dist']}")

    rows = [{"thr": "BASE", **rb}]
    for thr in THRESHOLDS:
        t = simulate_with_threshold(fcm, swing, atr, new_mdl, thr)
        r = report(t, f"peak_exit thr={thr}")
        print(f"{thr:>6.2f} {r['n']:>5d} {r['wr']:>5.1%} {r['pf']:>6.2f} {r['total_R']:>+9.0f} {r['dd']:>7.0f} {r['avg_bars']:>9.1f}  {r['exit_dist']}")
        rows.append({"thr": thr, **r})

    with open(os.path.join(OUT_DIR, "reports", "threshold_sweep.json"), "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"\nSaved {OUT_DIR}/reports/threshold_sweep.json")


if __name__ == "__main__":
    main()
