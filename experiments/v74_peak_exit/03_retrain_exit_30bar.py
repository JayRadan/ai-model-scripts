"""
v7.4 Peak-Exit experiment — retrain the exit head for a 30-bar horizon.

Compares THREE configurations on the same Janus holdout:
  A. PROD     — current 60-bar exit head, max_hold = 60 (production)
  B. CAP30    — current 60-bar exit head, max_hold = 30 (just lower the cap)
  C. NEW30    — re-trained 30-bar exit head, max_hold = 30 (proper fix)

The 30-bar exit head is trained from the same pre-cutoff confirmed setups but
with MAX_FWD_EXIT = 30 (was 60), so the model learns to fire stronger signals
earlier in the trade rather than save them for late.

NO production artifacts touched. Outputs:
  models/exit_30bar_v74.json
  reports/maxhold30_compare.json
"""
from __future__ import annotations
import os, sys, pickle, json, time as _time
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

CUTOFF = pd.Timestamp("2024-12-12 00:00:00")
NEW_HORIZON = 30   # was 60


def train_exit_with_horizon(conf_df, swing, atr, max_fwd: int):
    """Re-implementation of WF.train_exit but with parametrized MAX_FWD_EXIT."""
    H = swing["high"].values; L = swing["low"].values; C = swing["close"].values
    feats_arr = {col: swing[col].values for col in WF.EXIT_FEATS[3:]}
    rows = []
    for _, s in conf_df.iterrows():
        ei = int(s["idx"]); d = int(s["direction"]); a = float(s["atr"])
        if a <= 0: continue
        end = min(ei + max_fwd + 1, len(C))
        sl = WF.SL_HARD * a; entry_px = C[ei]
        for k in range(1, end - ei):
            bar = ei + k
            if d == 1:
                if (entry_px - L[bar]) >= sl: break
                pnl = (C[bar] - entry_px) / sl
            else:
                if (H[bar] - entry_px) >= sl: break
                pnl = (entry_px - C[bar]) / sl
            future_max = pnl
            for k2 in range(k + 1, end - ei):
                b2 = ei + k2
                if d == 1: r2 = (C[b2] - entry_px) / sl
                else:      r2 = (entry_px - C[b2]) / sl
                if r2 > future_max: future_max = r2
            label = int(future_max <= pnl)
            row = {"unrealized_pnl_R": pnl, "bars_held": k, "pnl_velocity": pnl/k, "label": label}
            for f in WF.EXIT_FEATS[3:]: row[f] = feats_arr[f][bar]
            rows.append(row)
    if not rows: return None
    ed = pd.DataFrame(rows)
    mdl = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                         subsample=0.85, colsample_bytree=0.85,
                         objective="binary:logistic", eval_metric="auc",
                         tree_method="hist", n_jobs=4, verbosity=0)
    mdl.fit(ed[WF.EXIT_FEATS].fillna(0).values, ed["label"].values)
    return mdl, ed


def simulate(confirmed, swing, atr, exit_mdl, max_hold: int):
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
    for r, (ei, d, t, cid_v, rule_v, a) in enumerate(entries):
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
    probs = np.zeros(len(entries) * max_hold)
    if valid.any():
        probs[valid] = exit_mdl.predict_proba(X[valid])[:, 1]
    rows = []
    for rank, (ei, d, t, cid_v, rule_v, a) in enumerate(entries):
        if a <= 0: continue
        sl = WF.SL_HARD * a; entry_px = C[ei]
        end = min(ei + max_hold + 1, n)
        xi, xr = end - 1, "max_hold"; pnl = 0.0
        for k in range(1, end - ei):
            bar = ei + k
            if d == 1:
                if (entry_px - L[bar]) >= sl: xi, xr, pnl = bar, "sl", -1.0; break
                pnl_now = (C[bar] - entry_px) / sl
            else:
                if (H[bar] - entry_px) >= sl: xi, xr, pnl = bar, "sl", -1.0; break
                pnl_now = (entry_px - C[bar]) / sl
            if k > 2 and probs[rank * max_hold + (k - 1)] >= WF.EXIT_THRESHOLD:
                xi, xr, pnl = bar, "ml_exit", pnl_now; break
            pnl = pnl_now
        rows.append({"time": t, "cid": cid_v, "rule": rule_v, "direction": d,
                      "bars": xi - ei, "pnl_R": pnl, "exit": xr})
    return pd.DataFrame(rows)


def report(df, name):
    if not len(df): return {"name": name, "n": 0}
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
    t0 = _time.time()
    print("Loading...", flush=True)
    feats = pd.read_csv(os.path.join(DATA, "features_v74.csv"), parse_dates=["time"])
    labs  = pd.read_csv(os.path.join(DATA, "labels_v74.csv"), parse_dates=["time"])
    df = feats.merge(labs, on="time", how="inner", suffixes=("", "_lab"))
    clusters = pd.read_csv(WF.CLUSTERS_PATH, parse_dates=["time"])[["time", "cid"]]
    swing, atr = WF.build_swing_with_exit_feats()
    payload = pickle.load(open(PKL_PATH, "rb"))
    pivot_feats = list(payload["pivot_feats"])

    pre  = df[df["time"] < CUTOFF].reset_index(drop=True)
    fold = df[df["time"] >= CUTOFF].reset_index(drop=True)
    p_pre  = payload["pivot_score_mdl"].predict_proba(pre[pivot_feats].fillna(0).values)[:, 1]
    pdir_pre = payload["pivot_dir_mdl"].predict_proba(pre[pivot_feats].fillna(0).values)[:, 1]
    p_fold  = payload["pivot_score_mdl"].predict_proba(fold[pivot_feats].fillna(0).values)[:, 1]
    pdir_fold = payload["pivot_dir_mdl"].predict_proba(fold[pivot_feats].fillna(0).values)[:, 1]

    def _build(parent, p_p, p_d):
        m = p_p >= payload["score_threshold"]
        s = parent[m].copy()
        s["direction"] = np.where(p_d[m] >= 0.5, 1, -1)
        s["rule"] = "RP_score"; s["idx"] = s["bar_idx"]
        s["entry_price"] = s["time"].map(swing.set_index("time")["close"])
        s["label"] = (s["best_R"] >= 1.0).astype(int)
        return s.merge(clusters, on="time", how="left")

    pre_setups  = _build(pre, p_pre, pdir_pre)
    fold_setups = _build(fold, p_fold, pdir_fold)
    pre_conf  = WF.filter_setups(pre_setups,  payload["mdls"], payload["thrs"], WF.V72L_FEATS)
    fold_conf = WF.filter_setups(fold_setups, payload["mdls"], payload["thrs"], WF.V72L_FEATS)
    fold_conf["direction"] = fold_conf["direction"].astype(int); fold_conf["cid"] = fold_conf["cid"].astype(int)
    pm = payload["meta_mdl"].predict_proba(fold_conf[WF.META_FEATS].fillna(0).values)[:, 1]
    fcm = fold_conf[pm >= payload["meta_threshold"]].reset_index(drop=True)
    print(f"  meta-filtered fold: {len(fcm):,} setups", flush=True)

    print(f"\n[1/3] PROD baseline — current 60-bar exit, max_hold=60...", flush=True)
    A = report(simulate(fcm, swing, atr, payload["exit_mdl"], 60), "A_PROD_60")

    print(f"[2/3] CAP30 — current 60-bar exit, max_hold=30 (just lower cap)...", flush=True)
    B = report(simulate(fcm, swing, atr, payload["exit_mdl"], 30), "B_CAP30")

    print(f"[3/3] NEW30 — RETRAINING exit head with MAX_FWD_EXIT=30...", flush=True)
    res = train_exit_with_horizon(pre_conf, swing, atr, NEW_HORIZON)
    if res is None:
        print("  no training data!"); return
    new_mdl, ed = res
    print(f"  trained on {len(ed):,} rows  (positive_rate {ed['label'].mean():.1%})")
    new_mdl.save_model(os.path.join(OUT_DIR, "models", "exit_30bar_v74.json"))
    C = report(simulate(fcm, swing, atr, new_mdl, 30), "C_NEW30")

    print("\n" + "=" * 78)
    print(f"{'config':<10s} {'n':>5s} {'WR':>7s} {'PF':>6s} {'+R':>9s} {'DD':>7s} {'avg_bars':>9s}  exits")
    print("-" * 78)
    for r in [A, B, C]:
        print(f"{r['name']:<10s} {r['n']:>5d} {r['wr']:>6.1%} {r['pf']:>6.2f} "
              f"{r['total_R']:>+9.0f} {r['dd']:>7.0f} {r['avg_bars']:>9.1f}  {r['exit_dist']}")
    print("=" * 78)
    print(f"\nDelta vs PROD (A):")
    for r in [B, C]:
        print(f"  {r['name']:<10s}  WR {(r['wr']-A['wr'])*100:+.1f}pp  "
              f"PF {r['pf']-A['pf']:+.2f}  R {r['total_R']-A['total_R']:+.0f}  "
              f"DD {r['dd']-A['dd']:+.0f}  bars {r['avg_bars']-A['avg_bars']:+.1f}")

    with open(os.path.join(OUT_DIR, "reports", "maxhold30_compare.json"), "w") as f:
        json.dump({"A_PROD_60": A, "B_CAP30": B, "C_NEW30": C}, f, indent=2, default=str)
    print(f"\nSaved comparison to {OUT_DIR}/reports/maxhold30_compare.json")
    print(f"Total time: {_time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
