"""
v7.4 Smart Exit — 2D sweep of P_HALF_THR × P_FULL_THR.

Reuses the trained 3-class model from 00_train_3class_exit.py. Sweeps the
inference thresholds to find the best PF / R / WR / DD trade-off.

Output: reports/2d_threshold_sweep.json + console table
"""
from __future__ import annotations
import os, sys, pickle, json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

ZIGZAG = "/home/jay/Desktop/new-model-zigzag"
sys.path.insert(0, os.path.join(ZIGZAG, "model_pipeline"))
sys.path.insert(0, os.path.join(ZIGZAG, "experiments/v74_pivot_score"))
import paths as P
from importlib.machinery import SourceFileLoader
WF = SourceFileLoader("wf", os.path.join(ZIGZAG, "experiments/v74_pivot_score/05_walk_forward.py")).load_module()

OUT_DIR  = os.path.join(ZIGZAG, "experiments/v74_smart_exit")
PKL_PATH = os.path.join(ZIGZAG, "models/janus_xau_validated.pkl")
DATA     = os.path.join(ZIGZAG, "experiments/v74_pivot_score/data")
NEW_MDL_PATH = os.path.join(OUT_DIR, "models", "exit_3class_v74.json")
CUTOFF = pd.Timestamp("2024-12-12 00:00:00")

P_HALF_GRID = [0.30, 0.40, 0.50, 0.60]
P_FULL_GRID = [0.30, 0.40, 0.50, 0.60, 0.70]


def simulate_3class(confirmed, swing, atr, exit_mdl, p_half_thr, p_full_thr, max_hold=60):
    H = swing["high"].values; L = swing["low"].values; C = swing["close"].values
    n = len(C)
    feats_arr = {col: swing[col].values for col in WF.EXIT_FEATS[3:]}
    entries = []
    for _, s in confirmed.iterrows():
        ei = int(s["idx"])
        if ei + max_hold >= n: continue
        entries.append((ei, int(s["direction"]), s["time"], int(s["cid"]), s["rule"], float(s["atr"])))
    if not entries: return pd.DataFrame()
    X = np.zeros((len(entries) * max_hold, len(WF.EXIT_FEATS)))
    valid = np.zeros(len(entries) * max_hold, dtype=bool)
    for r, (ei, d, _, _, _, a) in enumerate(entries):
        if a <= 0: continue
        sl = WF.SL_HARD * a; entry_px = C[ei]
        end = min(ei + max_hold + 1, n)
        for k in range(1, end - ei):
            bar = ei + k
            if d == 1: pnl = (C[bar] - entry_px) / sl
            else:      pnl = (entry_px - C[bar]) / sl
            row_idx = r * max_hold + (k - 1)
            X[row_idx, 0] = pnl; X[row_idx, 1] = k; X[row_idx, 2] = pnl/k
            for fi, f in enumerate(WF.EXIT_FEATS[3:]):
                X[row_idx, 3 + fi] = feats_arr[f][bar]
            valid[row_idx] = True
    probs = np.zeros((len(entries) * max_hold, 3))
    if valid.any(): probs[valid] = exit_mdl.predict_proba(X[valid])
    rows = []
    for rank, (ei, d, t, cid_v, rule_v, a) in enumerate(entries):
        if a <= 0: continue
        sl = WF.SL_HARD * a; entry_px = C[ei]
        end = min(ei + max_hold + 1, n)
        size = 1.0; realized_R = 0.0; half_at_bar = -1
        exit_kind = "max_hold"; last_bar = end - 1
        for k in range(1, end - ei):
            bar = ei + k
            if d == 1:
                if (entry_px - L[bar]) >= sl:
                    realized_R += size * (-1.0); last_bar = bar; size = 0.0; exit_kind = "sl"; break
                pnl_now = (C[bar] - entry_px) / sl
            else:
                if (H[bar] - entry_px) >= sl:
                    realized_R += size * (-1.0); last_bar = bar; size = 0.0; exit_kind = "sl"; break
                pnl_now = (entry_px - C[bar]) / sl
            if k <= 2: continue
            row_idx = rank * max_hold + (k - 1)
            p_hold, p_half, p_full = probs[row_idx]
            if size > 0 and p_full >= p_full_thr and p_full >= p_half:
                realized_R += size * pnl_now; last_bar = bar; size = 0.0
                exit_kind = "ml_full" if half_at_bar == -1 else "half_then_full"; break
            if size == 1.0 and p_half >= p_half_thr and half_at_bar == -1:
                realized_R += 0.5 * pnl_now; size = 0.5; half_at_bar = bar
                exit_kind = "half_active"
        if size > 0:
            bar = last_bar if exit_kind == "max_hold" else min(ei + max_hold, n - 1)
            if d == 1: pnl_at_end = (C[bar] - entry_px) / sl
            else:      pnl_at_end = (entry_px - C[bar]) / sl
            realized_R += size * pnl_at_end
            if exit_kind == "max_hold" or exit_kind == "half_active":
                exit_kind = "max_hold" if half_at_bar == -1 else "half_then_max"
            last_bar = bar
        rows.append({"time": t, "direction": d, "bars": last_bar - ei,
                      "pnl_R": realized_R, "exit": exit_kind})
    return pd.DataFrame(rows)


def report(df):
    if not len(df): return None
    ww = df[df["pnl_R"] > 0]; ll = df[df["pnl_R"] <= 0]
    pf = (ww["pnl_R"].sum() / -ll["pnl_R"].sum()) if len(ll) and ll["pnl_R"].sum() < 0 else float("inf")
    cum = df["pnl_R"].cumsum().values
    dd = float((cum - np.maximum.accumulate(cum)).min())
    return {"n": len(df), "wr": float(len(ww)/len(df)),
            "pf": float(pf), "total_R": float(df["pnl_R"].sum()),
            "dd": float(dd), "avg_bars": float(df["bars"].mean())}


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
    print(f"  meta-filtered fold: {len(fcm):,}\n")

    print(f"PROD baseline (binary head): WR 52.7%  PF 2.72  R +2794  DD -65  avg_bars 46.8\n")

    print(f"{'p_half':>7s} {'p_full':>7s}  {'WR':>6s} {'PF':>6s} {'+R':>9s} {'DD':>7s} {'bars':>6s}")
    print("-" * 60)
    rows = []
    for ph in P_HALF_GRID:
        for pf in P_FULL_GRID:
            t = simulate_3class(fcm, swing, atr, new_mdl, ph, pf)
            r = report(t)
            if not r: continue
            print(f"{ph:>7.2f} {pf:>7.2f}  {r['wr']:>5.1%} {r['pf']:>6.2f} "
                  f"{r['total_R']:>+9.0f} {r['dd']:>7.0f} {r['avg_bars']:>6.1f}")
            rows.append({"p_half": ph, "p_full": pf, **r})

    # Find best by various criteria
    by_pf = max(rows, key=lambda r: r["pf"])
    by_R = max(rows, key=lambda r: r["total_R"])
    by_wr = max(rows, key=lambda r: r["wr"])
    # Best PF with R retention >= 90% of prod (+2515)
    eligible = [r for r in rows if r["total_R"] >= 0.9 * 2794]
    best_pf_with_R = max(eligible, key=lambda r: r["pf"]) if eligible else None

    print(f"\nBest by PF      : p_half={by_pf['p_half']} p_full={by_pf['p_full']}  PF={by_pf['pf']:.2f}  R={by_pf['total_R']:+.0f}  WR={by_pf['wr']:.1%}")
    print(f"Best by total R : p_half={by_R['p_half']} p_full={by_R['p_full']}  PF={by_R['pf']:.2f}  R={by_R['total_R']:+.0f}  WR={by_R['wr']:.1%}")
    print(f"Best by WR      : p_half={by_wr['p_half']} p_full={by_wr['p_full']}  PF={by_wr['pf']:.2f}  R={by_wr['total_R']:+.0f}  WR={by_wr['wr']:.1%}")
    if best_pf_with_R:
        print(f"Best PF (R>=90% of prod): p_half={best_pf_with_R['p_half']} p_full={best_pf_with_R['p_full']}  "
              f"PF={best_pf_with_R['pf']:.2f}  R={best_pf_with_R['total_R']:+.0f}  WR={best_pf_with_R['wr']:.1%}")
    else:
        print("No combo retains >=90% of prod R.")

    with open(os.path.join(OUT_DIR, "reports", "2d_threshold_sweep.json"), "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"\nSaved {OUT_DIR}/reports/2d_threshold_sweep.json")


if __name__ == "__main__":
    main()
