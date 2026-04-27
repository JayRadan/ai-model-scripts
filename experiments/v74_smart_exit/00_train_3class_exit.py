"""
v7.4 Smart Exit — 3-class exit head with scale-out.

Replaces the binary "exit-now / hold" exit head with a 3-class head that
outputs P(HOLD), P(HALF), P(FULL):

  HOLD  (0): trade still building, run it
  HALF  (1): trade reached >=50% of its eventual peak — lock 50% size
  FULL  (2): trade at/past peak — close remainder

Plus the existing 4*ATR hard SL safety + 60-bar max-hold.

Compares vs the production binary head on the same Janus holdout.

Outputs:
  models/exit_3class_v74.json          — trained XGB (objective=multi:softprob)
  reports/smart_exit_compare.json      — A vs B side-by-side
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

OUT_DIR  = os.path.join(ZIGZAG, "experiments/v74_smart_exit")
PKL_PATH = os.path.join(ZIGZAG, "models/janus_xau_validated.pkl")
DATA     = os.path.join(ZIGZAG, "experiments/v74_pivot_score/data")
CUTOFF = pd.Timestamp("2024-12-12 00:00:00")

# Class label thresholds (ratio of current_pnl to trade's eventual peak_pnl)
HALF_RATIO = 0.50    # lock 50% when within 50-85% of peak
FULL_RATIO = 0.85    # close all when within 85% of peak

# Inference-time class probability thresholds
P_HALF_THR = 0.40
P_FULL_THR = 0.40


def build_3class_training_set(conf_df, swing, atr):
    """For each confirmed train trade, label every in-position bar with
    HOLD/HALF/FULL based on its forward trajectory to the eventual peak."""
    H = swing["high"].values; L = swing["low"].values; C = swing["close"].values
    feats_arr = {col: swing[col].values for col in WF.EXIT_FEATS[3:]}
    rows = []
    for _, s in conf_df.iterrows():
        ei = int(s["idx"]); d = int(s["direction"]); a = float(s["atr"])
        if a <= 0: continue
        sl = WF.SL_HARD * a; entry_px = C[ei]
        end = min(ei + WF.MAX_HOLD + 1, len(C))
        # Build full pnl sequence; mark SL bar
        seq = []
        for k in range(1, end - ei):
            bar = ei + k
            if d == 1:
                if (entry_px - L[bar]) >= sl: break
                pnl = (C[bar] - entry_px) / sl
            else:
                if (H[bar] - entry_px) >= sl: break
                pnl = (entry_px - C[bar]) / sl
            seq.append((k, bar, pnl))
        if len(seq) < 3: continue
        pnls = np.array([p[2] for p in seq])
        peak_idx_within = int(np.argmax(pnls))
        peak_pnl = float(pnls[peak_idx_within])

        for j, (k, bar, pnl) in enumerate(seq):
            # Default: HOLD
            label = 0
            if peak_pnl > 0:
                ratio = pnl / peak_pnl
                # Find peak index FROM HERE (not global) — the model needs
                # to recognise "this is approaching THIS bar's local peak"
                fut = pnls[j + 1:] if j + 1 < len(pnls) else np.array([])
                fut_max = float(fut.max()) if fut.size else pnl
                # Once we're past the global peak (no future better), and
                # current pnl is at >= 85% of peak, that's FULL
                if pnl > 0 and pnl >= FULL_RATIO * peak_pnl and fut_max <= pnl + 0.05:
                    label = 2
                # If we've achieved at least HALF_RATIO of the peak AND price
                # is still climbing (peak ahead), suggest HALF lock
                elif ratio >= HALF_RATIO and ratio < FULL_RATIO and fut_max > pnl:
                    label = 1
                # Past the peak — definitely time to be out
                elif j >= peak_idx_within and pnl > 0:
                    label = 2
            row = {"unrealized_pnl_R": pnl, "bars_held": k, "pnl_velocity": pnl/k, "label": label}
            for f in WF.EXIT_FEATS[3:]: row[f] = feats_arr[f][bar]
            rows.append(row)
    return pd.DataFrame(rows)


def simulate_3class(confirmed, swing, atr, exit_mdl, max_hold=60):
    """Simulate trades with scale-out: HALF closes 50% size, FULL closes remainder.
    PnL = realized PnL across the partial closes."""
    H = swing["high"].values; L = swing["low"].values; C = swing["close"].values
    n = len(C)
    feats_arr = {col: swing[col].values for col in WF.EXIT_FEATS[3:]}
    entries = []
    for _, s in confirmed.iterrows():
        ei = int(s["idx"])
        if ei + max_hold >= n: continue
        entries.append((ei, int(s["direction"]), s["time"], int(s["cid"]), s["rule"], float(s["atr"])))
    if not entries: return pd.DataFrame()

    # Batch-predict all (trade, bar) pairs
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
    if valid.any():
        probs[valid] = exit_mdl.predict_proba(X[valid])

    rows = []
    for rank, (ei, d, t, cid_v, rule_v, a) in enumerate(entries):
        if a <= 0: continue
        sl = WF.SL_HARD * a; entry_px = C[ei]
        end = min(ei + max_hold + 1, n)
        size = 1.0
        realized_R = 0.0   # closed pieces
        half_at_bar = -1
        exit_kind = "max_hold"
        last_bar = end - 1
        for k in range(1, end - ei):
            bar = ei + k
            if d == 1:
                if (entry_px - L[bar]) >= sl:
                    realized_R += size * (-1.0)   # SL takes remainder at -1R
                    last_bar = bar; size = 0.0; exit_kind = "sl"; break
                pnl_now = (C[bar] - entry_px) / sl
            else:
                if (H[bar] - entry_px) >= sl:
                    realized_R += size * (-1.0)
                    last_bar = bar; size = 0.0; exit_kind = "sl"; break
                pnl_now = (entry_px - C[bar]) / sl

            if k <= 2: continue   # min hold
            row_idx = rank * max_hold + (k - 1)
            p_hold, p_half, p_full = probs[row_idx]

            # FULL exit fires whenever P(FULL) is clearly the dominant non-hold option
            if size > 0 and p_full >= P_FULL_THR and p_full >= p_half:
                realized_R += size * pnl_now
                last_bar = bar; size = 0.0
                exit_kind = "ml_full" if half_at_bar == -1 else "half_then_full"
                break
            # HALF exit only fires once per trade, and only if size is full
            if size == 1.0 and p_half >= P_HALF_THR and half_at_bar == -1:
                realized_R += 0.5 * pnl_now
                size = 0.5
                half_at_bar = bar
                exit_kind = "half_active"  # will be overwritten when full closes

        # If position still open at end of window, close at last close
        if size > 0:
            bar = last_bar if exit_kind == "max_hold" else min(ei + max_hold, n - 1)
            if d == 1: pnl_at_end = (C[bar] - entry_px) / sl
            else:      pnl_at_end = (entry_px - C[bar]) / sl
            realized_R += size * pnl_at_end
            if exit_kind == "max_hold" or exit_kind == "half_active":
                exit_kind = "max_hold" if half_at_bar == -1 else "half_then_max"
            last_bar = bar

        rows.append({"time": t, "cid": cid_v, "rule": rule_v, "direction": d,
                      "bars": last_bar - ei, "pnl_R": realized_R, "exit": exit_kind,
                      "half_at": half_at_bar - ei if half_at_bar > 0 else -1})
    return pd.DataFrame(rows)


def simulate_binary(confirmed, swing, atr, exit_mdl, max_hold=60):
    """Production binary simulator (copied from WF)."""
    return WF.simulate(confirmed, swing, atr, exit_mdl)


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
    print(f"  meta-filtered fold: {len(fcm):,}\n")

    print("[1/3] Building 3-class training set...", flush=True)
    ed = build_3class_training_set(pre_conf, swing, atr)
    dist = ed["label"].value_counts().sort_index().to_dict()
    print(f"  rows={len(ed):,}  class dist (HOLD/HALF/FULL): {dist}")

    print("\n[2/3] Training 3-class XGBoost...", flush=True)
    mdl = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                         subsample=0.85, colsample_bytree=0.85,
                         objective="multi:softprob", num_class=3,
                         eval_metric="mlogloss", tree_method="hist", n_jobs=4, verbosity=0)
    mdl.fit(ed[WF.EXIT_FEATS].fillna(0).values, ed["label"].values)
    mdl.save_model(os.path.join(OUT_DIR, "models", "exit_3class_v74.json"))

    print("\n[3/3] Holdout simulate — A binary (prod) vs B 3-class (new)...", flush=True)
    A = report(simulate_binary(fcm, swing, atr, payload["exit_mdl"]), "A_PROD_binary")
    B = report(simulate_3class(fcm, swing, atr, mdl, max_hold=60), "B_3class_smart")

    print("\n" + "=" * 80)
    print(f"{'config':<18s} {'n':>5s} {'WR':>7s} {'PF':>6s} {'+R':>9s} {'DD':>7s} {'avg_bars':>9s}")
    print("-" * 80)
    for r in [A, B]:
        print(f"{r['name']:<18s} {r['n']:>5d} {r['wr']:>6.1%} {r['pf']:>6.2f} "
              f"{r['total_R']:>+9.0f} {r['dd']:>7.0f} {r['avg_bars']:>9.1f}")
    print("=" * 80)
    print(f"\nDelta NEW vs PROD:  WR {(B['wr']-A['wr'])*100:+.1f}pp  "
          f"PF {B['pf']-A['pf']:+.2f}  R {B['total_R']-A['total_R']:+.0f}  "
          f"DD {B['dd']-A['dd']:+.0f}  bars {B['avg_bars']-A['avg_bars']:+.1f}")
    print(f"\nA exit_dist: {A['exit_dist']}")
    print(f"B exit_dist: {B['exit_dist']}")

    with open(os.path.join(OUT_DIR, "reports", "smart_exit_compare.json"), "w") as f:
        json.dump({"A_PROD_binary": A, "B_3class_smart": B,
                    "label_dist_train": dist, "n_train_rows": len(ed)}, f, indent=2, default=str)
    print(f"\nTotal time: {_time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
