"""
v7.4 Pivot Score — train the per-bar pivot detector.

Single XGBoost classifier on every bar.
Target: is_pivot_25  (best_R >= 2.5R within 60 bars under oracle exit + 4*ATR SL)
Direction is predicted by a SECOND classifier on the same features.

Honest 80/20 temporal split (cutoff 2024-12-12, same as Oracle).

Saves:
  models/pivot_score_v74.json    (P(is_pivot))
  models/pivot_dir_v74.json      (P(direction = +1 | is_pivot))
  reports/v74_pivot_train.json   (AUC, threshold curves, top features)
"""
from __future__ import annotations
import os, json, sys
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

DATA = "/home/jay/Desktop/new-model-zigzag/experiments/v74_pivot_score/data"
MDL = "/home/jay/Desktop/new-model-zigzag/experiments/v74_pivot_score/models"
RPT = "/home/jay/Desktop/new-model-zigzag/experiments/v74_pivot_score/reports"
os.makedirs(MDL, exist_ok=True); os.makedirs(RPT, exist_ok=True)

CUTOFF = pd.Timestamp("2024-12-12 00:00:00")
TARGET_COL = "is_pivot_25"

XGB_PARAMS = dict(
    n_estimators=500, max_depth=5, learning_rate=0.05,
    subsample=0.85, colsample_bytree=0.85,
    objective="binary:logistic", eval_metric="auc",
    tree_method="hist", n_jobs=4, verbosity=0,
)

NON_FEAT = {"time", "bar_idx", "atr", "best_long_R", "best_short_R",
             "best_R", "best_dir", "is_pivot_15", "is_pivot_25", "is_pivot_4"}


def main():
    print("Loading features + labels...", flush=True)
    feats = pd.read_csv(os.path.join(DATA, "features_v74.csv"), parse_dates=["time"])
    labs = pd.read_csv(os.path.join(DATA, "labels_v74.csv"), parse_dates=["time"])
    df = feats.merge(labs, on="time", how="inner", suffixes=("", "_lab"))
    print(f"  merged {len(df):,} rows  {len(df.columns)} cols")

    feat_cols = [c for c in df.columns if c not in NON_FEAT and not c.endswith("_lab")
                  and df[c].dtype != object and c not in ("bar_idx_lab", "atr_lab")]
    print(f"  using {len(feat_cols)} features")

    train = df[df["time"] < CUTOFF].reset_index(drop=True)
    holdout = df[df["time"] >= CUTOFF].reset_index(drop=True)
    val_n = max(5000, int(len(train) * 0.10))
    train_fit = train.iloc[:-val_n].reset_index(drop=True)
    val = train.iloc[-val_n:].reset_index(drop=True)
    print(f"  split: fit={len(train_fit):,}  val={len(val):,}  holdout={len(holdout):,}")
    print(f"  base rate ({TARGET_COL}): train={train[TARGET_COL].mean():.1%}  hold={holdout[TARGET_COL].mean():.1%}")

    # --- pivot-score head ---
    print("\nTraining pivot-score head...", flush=True)
    X_tr = train_fit[feat_cols].fillna(0).values; y_tr = train_fit[TARGET_COL].values
    X_v = val[feat_cols].fillna(0).values; y_v = val[TARGET_COL].values
    X_h = holdout[feat_cols].fillna(0).values; y_h = holdout[TARGET_COL].values

    score_mdl = XGBClassifier(**XGB_PARAMS); score_mdl.fit(X_tr, y_tr)
    p_v = score_mdl.predict_proba(X_v)[:, 1]
    p_h = score_mdl.predict_proba(X_h)[:, 1]
    auc_v = roc_auc_score(y_v, p_v); auc_h = roc_auc_score(y_h, p_h)
    print(f"  AUC val={auc_v:.3f}  holdout={auc_h:.3f}")

    # threshold sweep
    print("\nThreshold sweep on holdout:")
    print(f"  {'thr':>5s}  {'fires':>8s}  {'fires/day':>10s}  {'pivot-rate':>10s}")
    span_days = max(1.0, (holdout['time'].iloc[-1] - holdout['time'].iloc[0]).total_seconds() / 86400)
    sweep = []
    for thr in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        mask = p_h >= thr
        if mask.sum() == 0: continue
        rate = y_h[mask].mean()
        fpd = mask.sum() / span_days
        print(f"  {thr:>5.2f}  {mask.sum():>8d}  {fpd:>10.1f}  {rate:>10.1%}")
        sweep.append({"thr": thr, "fires": int(mask.sum()), "fires_per_day": float(fpd), "pivot_rate": float(rate)})

    # --- direction head (only on actual pivots) ---
    print("\nTraining direction head (only on pivots)...", flush=True)
    pos = train_fit[train_fit[TARGET_COL] == 1].reset_index(drop=True)
    Xp_tr = pos[feat_cols].fillna(0).values
    yp_tr = (pos["best_dir"] == 1).astype(int).values
    print(f"  {len(pos):,} pivot-bars; long-rate {yp_tr.mean():.1%}")
    dir_mdl = XGBClassifier(**XGB_PARAMS); dir_mdl.fit(Xp_tr, yp_tr)
    pv_pos = val[val[TARGET_COL] == 1]
    if len(pv_pos):
        pv_dir = dir_mdl.predict_proba(pv_pos[feat_cols].fillna(0).values)[:, 1]
        pv_y = (pv_pos["best_dir"] == 1).astype(int).values
        try:
            auc_dir = roc_auc_score(pv_y, pv_dir)
            print(f"  direction AUC on val pivots: {auc_dir:.3f}")
        except Exception:
            auc_dir = float("nan")

    # --- top-importance features ---
    imp = pd.Series(score_mdl.feature_importances_, index=feat_cols).sort_values(ascending=False)
    print("\nTop 15 features (pivot-score):")
    for name, val_imp in imp.head(15).items():
        print(f"  {val_imp:.4f}  {name}")

    score_mdl.save_model(os.path.join(MDL, "pivot_score_v74.json"))
    dir_mdl.save_model(os.path.join(MDL, "pivot_dir_v74.json"))
    rpt = {
        "target": TARGET_COL,
        "n_train": int(len(train_fit)), "n_val": int(len(val)), "n_holdout": int(len(holdout)),
        "auc_val": float(auc_v), "auc_holdout": float(auc_h),
        "threshold_sweep_holdout": sweep,
        "top_features": [{"name": n, "imp": float(v)} for n, v in imp.head(20).items()],
        "feat_cols": feat_cols,
    }
    with open(os.path.join(RPT, "v74_pivot_train.json"), "w") as f:
        json.dump(rpt, f, indent=2)
    print(f"\nSaved models + report to {MDL} / {RPT}")


if __name__ == "__main__":
    main()
