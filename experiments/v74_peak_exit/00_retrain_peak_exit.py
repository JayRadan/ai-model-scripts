"""
v7.4 Peak-Exit — retrain exit head with strict peak-detector label.

For every confirmed train trade, simulate forward MAX_HOLD bars. For each
in-position bar j:
  peak_idx = argmax(pnl_R over [bar+1 .. bar+MAX_HOLD])
  label = 1 if j == peak_idx (the exact bar where pnl hits its peak)
  label = 1 if pnl[j] >= 0.95 * pnl[peak_idx] AND pnl[j+5_bars_max] < pnl[j]
          (softer near-peak with momentum-fading confirmation)

Train an XGBoost classifier on the same 11 EXIT_FEATS Oracle uses.
Then re-simulate the holdout fold with this new exit head and compare to
the current Janus pickle's exit head.

NO modifications to any v74 production artifact. Only this folder writes.
"""
from __future__ import annotations
import os, sys, pickle, time as _time
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

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

# Peak-label parameters
PEAK_TOLERANCE_R = 0.05       # within 5% of trade peak counts as "peak zone"
PEAK_FADE_BARS   = 5          # require pnl to fade in next K bars after the peak
USE_SOFT_LABEL   = True       # if True, use the soft "near-peak + fading" rule


def build_peak_exit_training_set(conf_df, swing, atr):
    """For each confirmed train trade, walk forward MAX_HOLD bars and emit
    one row per (trade, bar) with the EXIT_FEATS + peak-detector label."""
    H = swing["high"].values; L = swing["low"].values; C = swing["close"].values
    feats_arr = {col: swing[col].values for col in WF.EXIT_FEATS[3:]}
    rows = []
    n = len(C)

    for _, s in conf_df.iterrows():
        ei = int(s["idx"]); d = int(s["direction"]); a = float(s["atr"])
        if a <= 0: continue
        sl = WF.SL_HARD * a; entry_px = C[ei]
        end = min(ei + WF.MAX_HOLD + 1, n)
        if end - ei < 5: continue

        # Compute the full pnl trajectory; mark where SL would have closed.
        pnls = []
        sl_bar = None
        for k in range(1, end - ei):
            bar = ei + k
            if d == 1:
                if (entry_px - L[bar]) >= sl: sl_bar = k; break
                pnl = (C[bar] - entry_px) / sl
            else:
                if (H[bar] - entry_px) >= sl: sl_bar = k; break
                pnl = (entry_px - C[bar]) / sl
            pnls.append((k, bar, pnl))

        if not pnls: continue
        pnl_array = np.array([p[2] for p in pnls])
        peak_idx_within = int(np.argmax(pnl_array))
        peak_pnl = pnl_array[peak_idx_within]

        if peak_pnl <= 0:
            # No favorable excursion at all — every bar is a "hold" from peak-detector POV
            for (k, bar, pnl) in pnls:
                row = {"unrealized_pnl_R": pnl, "bars_held": k, "pnl_velocity": pnl/k, "label": 0}
                for f in WF.EXIT_FEATS[3:]: row[f] = feats_arr[f][bar]
                rows.append(row)
            continue

        for j, (k, bar, pnl) in enumerate(pnls):
            label = 0
            if USE_SOFT_LABEL:
                # near peak + about to fade
                if pnl >= 0.95 * peak_pnl and pnl > 0:
                    end_window = min(j + 1 + PEAK_FADE_BARS, len(pnl_array))
                    if end_window > j + 1:
                        future = pnl_array[j + 1: end_window]
                        if future.size and future.max() < pnl:
                            label = 1
                # also always label the exact peak bar
                if j == peak_idx_within:
                    label = 1
            else:
                if j == peak_idx_within:
                    label = 1
            row = {"unrealized_pnl_R": pnl, "bars_held": k, "pnl_velocity": pnl/k, "label": label}
            for f in WF.EXIT_FEATS[3:]: row[f] = feats_arr[f][bar]
            rows.append(row)
    return pd.DataFrame(rows)


def simulate_with_exit_mdl(conf_df, swing, atr, exit_mdl, exit_threshold=0.55):
    return WF.simulate(conf_df, swing, atr, exit_mdl)   # reuses Oracle's simulate; threshold baked into WF.EXIT_THRESHOLD


def report_trades(df, name):
    if not len(df): return {"name": name, "n": 0}
    ww = df[df["pnl_R"] > 0]; ll = df[df["pnl_R"] <= 0]
    pf = (ww["pnl_R"].sum() / -ll["pnl_R"].sum()) if len(ll) and ll["pnl_R"].sum() < 0 else float("inf")
    cum = df["pnl_R"].cumsum().values
    dd = float((cum - np.maximum.accumulate(cum)).min()) if len(cum) else 0.0
    exit_dist = df["exit"].value_counts().to_dict()
    return {"name": name, "n": int(len(df)), "wr": float(len(ww)/len(df)),
            "pf": float(pf), "expect": float(df["pnl_R"].mean()),
            "total_R": float(df["pnl_R"].sum()), "dd": float(dd),
            "avg_bars": float(df["bars"].mean()),
            "exit_dist": exit_dist}


def main():
    t0 = _time.time()
    print(f"Loading inputs (read-only)...", flush=True)
    feats = pd.read_csv(os.path.join(DATA, "features_v74.csv"), parse_dates=["time"])
    labs  = pd.read_csv(os.path.join(DATA, "labels_v74.csv"), parse_dates=["time"])
    df = feats.merge(labs, on="time", how="inner", suffixes=("", "_lab"))
    clusters = pd.read_csv(WF.CLUSTERS_PATH, parse_dates=["time"])[["time", "cid"]]
    swing, atr = WF.build_swing_with_exit_feats()
    payload = pickle.load(open(PKL_PATH, "rb"))
    print(f"  rows={len(df):,}  pivot_feats={len(payload['pivot_feats'])}  current_exit_mdl loaded")

    # Reuse the existing pivot+dir+confirm+meta from the production pickle.
    # We only retrain the exit head.
    pivot_feats = list(payload["pivot_feats"])
    score_mdl = payload["pivot_score_mdl"]
    dir_mdl   = payload["pivot_dir_mdl"]
    mdls = payload["mdls"]; thrs = payload["thrs"]
    meta_mdl = payload["meta_mdl"]; meta_thr = payload["meta_threshold"]
    score_thr = payload["score_threshold"]
    OLD_EXIT_MDL = payload["exit_mdl"]

    # Score and build setups for all bars
    pre  = df[df["time"] < CUTOFF].reset_index(drop=True)
    fold = df[df["time"] >= CUTOFF].reset_index(drop=True)
    p_pre  = score_mdl.predict_proba(pre[pivot_feats].fillna(0).values)[:, 1]
    pdir_pre = dir_mdl.predict_proba(pre[pivot_feats].fillna(0).values)[:, 1]
    p_fold  = score_mdl.predict_proba(fold[pivot_feats].fillna(0).values)[:, 1]
    pdir_fold = dir_mdl.predict_proba(fold[pivot_feats].fillna(0).values)[:, 1]

    def _build(parent, p_p, p_d):
        m = p_p >= score_thr
        s = parent[m].copy()
        s["direction"] = np.where(p_d[m] >= 0.5, 1, -1)
        s["rule"] = "RP_score"; s["idx"] = s["bar_idx"]
        s["entry_price"] = s["time"].map(swing.set_index("time")["close"])
        s["label"] = (s["best_R"] >= 1.0).astype(int)
        return s.merge(clusters, on="time", how="left")

    pre_setups  = _build(pre, p_pre, pdir_pre)
    fold_setups = _build(fold, p_fold, pdir_fold)
    pre_conf  = WF.filter_setups(pre_setups,  mdls, thrs, WF.V72L_FEATS)
    fold_conf = WF.filter_setups(fold_setups, mdls, thrs, WF.V72L_FEATS)
    print(f"  pre_conf={len(pre_conf):,}  fold_conf={len(fold_conf):,}")

    # 1. Train new peak-detector exit head
    print(f"\n[1/3] Building peak-exit training set...", flush=True)
    ed = build_peak_exit_training_set(pre_conf, swing, atr)
    pos_rate = ed["label"].mean()
    print(f"  rows={len(ed):,}  positive_rate={pos_rate:.2%}  "
          f"(should be much sparser than baseline ~30-50%)")

    print(f"[2/3] Training peak-exit XGBoost...", flush=True)
    new_mdl = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                             subsample=0.85, colsample_bytree=0.85,
                             objective="binary:logistic", eval_metric="auc",
                             tree_method="hist", n_jobs=4, verbosity=0)
    # split last 10% of training rows as a quick sanity-AUC slice
    split = int(len(ed) * 0.90)
    new_mdl.fit(ed.iloc[:split][WF.EXIT_FEATS].fillna(0).values, ed.iloc[:split]["label"].values)
    if ed.iloc[split:]["label"].nunique() > 1:
        auc = roc_auc_score(ed.iloc[split:]["label"].values,
                             new_mdl.predict_proba(ed.iloc[split:][WF.EXIT_FEATS].fillna(0).values)[:, 1])
        print(f"  AUC on val slice: {auc:.3f}")

    new_mdl.save_model(os.path.join(OUT_DIR, "models", "peak_exit_v74.json"))

    # 2. Re-run holdout sim with both exit heads
    print(f"\n[3/3] Running holdout sim — comparing OLD vs NEW exit head...", flush=True)
    fold_conf["direction"] = fold_conf["direction"].astype(int); fold_conf["cid"] = fold_conf["cid"].astype(int)
    pm = meta_mdl.predict_proba(fold_conf[WF.META_FEATS].fillna(0).values)[:, 1]
    fold_conf_m = fold_conf[pm >= meta_thr].reset_index(drop=True)
    print(f"  meta-filtered fold setups: {len(fold_conf_m):,}")

    trades_old = WF.simulate(fold_conf_m, swing, atr, OLD_EXIT_MDL)
    trades_new = WF.simulate(fold_conf_m, swing, atr, new_mdl)
    r_old = report_trades(trades_old, "current_exit (90% hold-to-max)")
    r_new = report_trades(trades_new, "peak_exit (NEW)")

    print(f"\n{'metric':<22s} {'OLD':>14s} {'NEW':>14s}  delta")
    for k in ["n", "wr", "pf", "expect", "total_R", "dd", "avg_bars"]:
        ov, nv = r_old[k], r_new[k]
        if isinstance(ov, float):
            d = nv - ov
            print(f"  {k:<20s} {ov:>14.3f} {nv:>14.3f}  {d:+.3f}")
        else:
            print(f"  {k:<20s} {ov:>14d} {nv:>14d}  {nv-ov:+d}")
    print(f"\nExit distribution (OLD): {r_old['exit_dist']}")
    print(f"Exit distribution (NEW): {r_new['exit_dist']}")

    import json
    with open(os.path.join(OUT_DIR, "reports", "comparison.json"), "w") as f:
        json.dump({"old": r_old, "new": r_new,
                    "peak_label": {"tolerance_R": PEAK_TOLERANCE_R,
                                    "fade_bars": PEAK_FADE_BARS,
                                    "use_soft": USE_SOFT_LABEL,
                                    "training_rows": len(ed),
                                    "positive_rate": float(pos_rate)}}, f, indent=2)
    print(f"\nSaved comparison to {OUT_DIR}/reports/comparison.json")
    print(f"Total time: {_time.time()-t0:.0f}s")

    # Verdict
    print("\n" + "="*70)
    if r_new["pf"] >= r_old["pf"] - 0.10 and r_new["avg_bars"] < r_old["avg_bars"] * 0.7:
        print("VERDICT: NEW exit head is competitive — meaningfully shorter holds with similar PF.")
        print("         Recommend swapping into janus_xau_validated.pkl.")
    elif r_new["pf"] > r_old["pf"]:
        print("VERDICT: NEW exit head wins on PF too. Recommend swap.")
    else:
        print("VERDICT: NEW exit head degrades PF below acceptable. Keep current.")
    print("="*70)


if __name__ == "__main__":
    main()
