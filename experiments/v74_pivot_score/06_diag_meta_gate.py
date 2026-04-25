"""
v7.4 Pivot Score — diagnose the meta gate.

Hypothesis: meta gate is inert in walk-forward folds because v72L features were
already squeezed by the pivot-score model upstream — meta has no fresh signal.

This script reproduces ONE fold (F4 2025-04-13 → 2026-04-13) and dumps:
  - Full meta-threshold sweep on val (0.05 -> 0.95, step 0.025)
  - Meta-prob histogram on val
  - Meta feature importances (V72L_FEATS + direction + cid)
  - AUC of meta on val
  - Compare three threshold-selection criteria:
      A. max total_R (current walk-forward criterion)
      B. max PF with min retention 30%
      C. max expectancy with min retention 50%
  - Then evaluate ALL THREE on the actual fold and print PF/WR/DD

Output: reports/v74_meta_diag_F4.json + console table.
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "/home/jay/Desktop/new-model-zigzag/model_pipeline")
import paths as P
sys.path.insert(0, "/home/jay/Desktop/new-model-zigzag/experiments/v74_pivot_score")
from importlib.machinery import SourceFileLoader
WF = SourceFileLoader("wf", "/home/jay/Desktop/new-model-zigzag/experiments/v74_pivot_score/05_walk_forward.py").load_module()

DATA = "/home/jay/Desktop/new-model-zigzag/experiments/v74_pivot_score/data"
RPT  = "/home/jay/Desktop/new-model-zigzag/experiments/v74_pivot_score/reports"

FOLD_NAME  = "F4_2025"
FOLD_START = pd.Timestamp("2025-04-13")
FOLD_END   = pd.Timestamp("2026-04-13")


def main():
    print("Loading...", flush=True)
    feats = pd.read_csv(os.path.join(DATA, "features_v74.csv"), parse_dates=["time"])
    labs = pd.read_csv(os.path.join(DATA, "labels_v74.csv"), parse_dates=["time"])
    df = feats.merge(labs, on="time", how="inner", suffixes=("", "_lab"))
    clusters = pd.read_csv(WF.CLUSTERS_PATH, parse_dates=["time"])[["time", "cid"]]
    swing, atr = WF.build_swing_with_exit_feats()

    pre  = df[df["time"] < FOLD_START].reset_index(drop=True)
    fold = df[(df["time"] >= FOLD_START) & (df["time"] < FOLD_END)].reset_index(drop=True)

    pivot_feats = [c for c in df.columns if c not in WF.NON_FEAT_PIVOT
                    and df[c].dtype != object and not c.endswith("_lab")]

    print(f"Training pivot-score on pre={len(pre):,}...", flush=True)
    score_mdl = XGBClassifier(**WF.XGB_S)
    score_mdl.fit(pre[pivot_feats].fillna(0).values, pre[WF.TARGET_COL].values)
    pos = pre[pre[WF.TARGET_COL] == 1]
    dir_mdl = XGBClassifier(**WF.XGB_S)
    dir_mdl.fit(pos[pivot_feats].fillna(0).values, (pos["best_dir"] == 1).astype(int).values)

    p_pre  = score_mdl.predict_proba(pre[pivot_feats].fillna(0).values)[:, 1]
    p_fold = score_mdl.predict_proba(fold[pivot_feats].fillna(0).values)[:, 1]
    pdir_pre  = dir_mdl.predict_proba(pre[pivot_feats].fillna(0).values)[:, 1]
    pdir_fold = dir_mdl.predict_proba(fold[pivot_feats].fillna(0).values)[:, 1]

    def build_setups(parent, p_p, p_d):
        m = p_p >= WF.SCORE_THR
        s = parent[m].copy()
        s["direction"] = np.where(p_d[m] >= 0.5, 1, -1)
        s["rule"] = "RP_score"; s["idx"] = s["bar_idx"]
        s["entry_price"] = s["time"].map(swing.set_index("time")["close"])
        s["label"] = (s["best_R"] >= 1.0).astype(int)
        s = s.merge(clusters, on="time", how="left")
        return s

    pre_setups  = build_setups(pre,  p_pre,  pdir_pre)
    fold_setups = build_setups(fold, p_fold, pdir_fold)

    mdls, thrs = WF.train_conf(pre_setups, WF.V72L_FEATS)
    pre_conf  = WF.filter_setups(pre_setups,  mdls, thrs, WF.V72L_FEATS)
    fold_conf = WF.filter_setups(fold_setups, mdls, thrs, WF.V72L_FEATS)
    print(f"  pre_conf={len(pre_conf):,}  fold_conf={len(fold_conf):,}")

    print("Training exit head...", flush=True)
    exit_mdl = WF.train_exit(pre_conf, swing, atr)

    print("Simulating pre-confirmed trades for meta labels...", flush=True)
    pre_trades = WF.simulate(pre_conf, swing, atr, exit_mdl)
    md = pre_trades.merge(pre_conf[["time","cid","rule"] + WF.V72L_FEATS],
                          on=["time","cid","rule"], how="left")
    md["meta_label"] = (md["pnl_R"] > 0).astype(int)
    md["direction"] = md["direction"].astype(int); md["cid"] = md["cid"].astype(int)
    val_n = max(500, int(len(md) * 0.10))
    mtr = md.iloc[:-val_n].reset_index(drop=True)
    mvd = md.iloc[-val_n:].reset_index(drop=True)
    print(f"  meta-train: {len(mtr):,}  val: {len(mvd):,}")

    meta_mdl = XGBClassifier(**WF.XGB_C)
    meta_mdl.fit(mtr[WF.META_FEATS].fillna(0).values, mtr["meta_label"].values)
    pv = meta_mdl.predict_proba(mvd[WF.META_FEATS].fillna(0).values)[:, 1]
    yv = mvd["meta_label"].values; pn = mvd["pnl_R"].values

    auc = roc_auc_score(yv, pv) if len(set(yv)) > 1 else float("nan")
    print(f"\nMETA AUC on val: {auc:.3f}")
    print(f"Val baseline:  WR={yv.mean():.1%}  total_R={pn.sum():+.1f}  expect={pn.mean():+.3f}")

    # --- prob histogram ---
    print("\nMeta-prob histogram on val:")
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0]
    h, _ = np.histogram(pv, bins=bins)
    for i in range(len(h)):
        print(f"  [{bins[i]:.2f}-{bins[i+1]:.2f})  {h[i]:>6,}  {'#' * int(h[i] / max(h) * 40)}")

    # --- full sweep ---
    print(f"\n{'thr':>6s} {'n':>6s} {'retain%':>8s} {'WR':>6s} {'PF':>6s} {'expect':>8s} {'total_R':>10s}")
    sweep = []
    for thr in np.linspace(0.05, 0.95, 37):
        m = pv >= thr
        if m.sum() == 0: continue
        sub_pn = pn[m]; sub_y = yv[m]
        wins = sub_pn[sub_pn > 0].sum(); losses = -sub_pn[sub_pn <= 0].sum()
        pf = (wins / losses) if losses > 0 else float("inf")
        wr = sub_y.mean(); exp = sub_pn.mean(); tot = sub_pn.sum()
        retain = m.sum() / len(pv)
        sweep.append({"thr": float(thr), "n": int(m.sum()), "retain": float(retain),
                      "wr": float(wr), "pf": float(pf), "expect": float(exp), "total_R": float(tot)})
        if int(thr * 100) % 5 == 0:
            print(f"  {thr:>5.3f} {m.sum():>6d} {retain*100:>7.1f}% {wr:>5.1%} {pf:>6.2f} {exp:>+7.3f} {tot:>+9.1f}")

    # --- 3 selection criteria ---
    def sel_max_R(s): return max(s, key=lambda r: r["total_R"])
    def sel_max_PF_min_30(s): return max([r for r in s if r["retain"] >= 0.30] or s, key=lambda r: r["pf"])
    def sel_max_exp_min_50(s): return max([r for r in s if r["retain"] >= 0.50] or s, key=lambda r: r["expect"])

    A = sel_max_R(sweep); B = sel_max_PF_min_30(sweep); C = sel_max_exp_min_50(sweep)
    print(f"\nCRITERION A — max total_R         : thr={A['thr']:.3f}  retain={A['retain']*100:.0f}%  PF={A['pf']:.2f}  WR={A['wr']:.1%}  exp={A['expect']:+.3f}")
    print(f"CRITERION B — max PF (retain≥30%) : thr={B['thr']:.3f}  retain={B['retain']*100:.0f}%  PF={B['pf']:.2f}  WR={B['wr']:.1%}  exp={B['expect']:+.3f}")
    print(f"CRITERION C — max exp (retain≥50%): thr={C['thr']:.3f}  retain={C['retain']*100:.0f}%  PF={C['pf']:.2f}  WR={C['wr']:.1%}  exp={C['expect']:+.3f}")

    # --- apply to ACTUAL fold ---
    fold_conf["direction"] = fold_conf["direction"].astype(int); fold_conf["cid"] = fold_conf["cid"].astype(int)
    pm = meta_mdl.predict_proba(fold_conf[WF.META_FEATS].fillna(0).values)[:, 1]
    print(f"\nFOLD evaluation under each criterion:")
    print(f"  {'crit':<5s} {'thr':>6s} {'n':>5s} {'WR':>6s} {'PF':>6s} {'R':>9s} {'DD':>7s}")
    for label, c in [("A", A), ("B", B), ("C", C)]:
        m = pm >= c["thr"]
        sub = fold_conf[m].reset_index(drop=True)
        if len(sub) == 0:
            print(f"  {label:<5s} {c['thr']:>5.3f}  empty"); continue
        trades = WF.simulate(sub, swing, atr, exit_mdl)
        r = WF.report(trades)
        print(f"  {label:<5s} {c['thr']:>5.3f} {r['n']:>5d} {r['wr']:>5.1%} {r['pf']:>6.2f} {r['total_R']:>+9.0f} {r['dd']:>7.0f}")

    # --- feature importances ---
    print("\nMeta feature importances:")
    imp = pd.Series(meta_mdl.feature_importances_, index=WF.META_FEATS).sort_values(ascending=False)
    for n, v in imp.items():
        print(f"  {v:.4f}  {n}")

    out = {"meta_auc": float(auc), "sweep": sweep,
            "selected_A": A, "selected_B": B, "selected_C": C,
            "meta_importances": {k: float(v) for k, v in imp.items()}}
    with open(os.path.join(RPT, "v74_meta_diag_F4.json"), "w") as f: json.dump(out, f, indent=2)
    print(f"\nSaved {RPT}/v74_meta_diag_F4.json")


if __name__ == "__main__":
    main()
