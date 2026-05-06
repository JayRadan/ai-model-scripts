"""
v7.4 Pivot Score — turn the trained pivot-score model into Oracle-shape setups.

For every bar, score P(is_pivot) and P(direction=+1|is_pivot).
A bar becomes a setup if P(is_pivot) >= SCORE_THR.
Each setup gets:
  - direction = +1 if P(dir=+1) >= 0.5 else -1
  - rule = "RP_score"   (so Oracle's per-rule loop has something to group on)
  - cid from cluster_per_bar_v73.csv
  - all 55 features from features_v74.csv merged in

Outputs setups in Oracle schema, one CSV per cluster:
  data/setups_{0..K-1}_v74.csv

Tunable: SCORE_THR — start at 0.10 to get ~5-10 fires/day. Lower = more
trades, lower precision; higher = fewer, sharper.
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

sys.path.insert(0, "/home/jay/Desktop/new-model-zigzag/model_pipeline")
import paths as P

DATA = "/home/jay/Desktop/new-model-zigzag/experiments/v74_pivot_score/data"
MDL  = "/home/jay/Desktop/new-model-zigzag/experiments/v74_pivot_score/models"
CLUSTERS_PATH = "/home/jay/Desktop/new-model-zigzag/experiments/v73_pivot_oracle/data/cluster_per_bar_v73.csv"

SCORE_THR = 0.30           # tweakable. 0.30 ~ 13 fires/day @ 37.5% pivot-rate (2x base)

NON_FEAT = {"time", "bar_idx", "atr", "best_long_R", "best_short_R",
             "best_R", "best_dir", "is_pivot_15", "is_pivot_25", "is_pivot_4"}


def main():
    print("Loading features + labels + clusters...", flush=True)
    feats = pd.read_csv(os.path.join(DATA, "features_v74.csv"), parse_dates=["time"])
    labs = pd.read_csv(os.path.join(DATA, "labels_v74.csv"), parse_dates=["time"])
    clusters = pd.read_csv(CLUSTERS_PATH, parse_dates=["time"])

    df = feats.merge(labs[["time", "atr", "best_R", "best_dir"]], on="time", how="inner", suffixes=("", "_lab"))
    df = df.merge(clusters[["time", "cid"]], on="time", how="left")

    feat_cols = [c for c in df.columns if c not in NON_FEAT and c != "cid" and c != "atr_lab"
                  and df[c].dtype != object and not c.endswith("_lab")]
    print(f"  {len(df):,} bars  {len(feat_cols)} features", flush=True)

    print("Loading models...", flush=True)
    score_mdl = XGBClassifier(); score_mdl.load_model(os.path.join(MDL, "pivot_score_v74.json"))
    dir_mdl   = XGBClassifier(); dir_mdl.load_model(os.path.join(MDL, "pivot_dir_v74.json"))

    print("Scoring all bars...", flush=True)
    X = df[feat_cols].fillna(0).values
    p_pivot = score_mdl.predict_proba(X)[:, 1]
    p_dir   = dir_mdl.predict_proba(X)[:, 1]

    fires_mask = p_pivot >= SCORE_THR
    n_fires = int(fires_mask.sum())
    print(f"  thr={SCORE_THR} -> {n_fires:,} fires ({n_fires/len(df)*100:.2f}% of bars)")

    # Load swing for entry_price + Oracle-schema column generation
    swing = pd.read_csv(P.data("swing_v5_xauusd.csv"), parse_dates=["time"])
    swing = swing.sort_values("time").reset_index(drop=True)

    setups = df[fires_mask].copy()
    setups["direction"] = np.where(p_dir[fires_mask] >= 0.5, 1, -1)
    setups["rule"] = "RP_score"
    setups["entry_price"] = setups["time"].map(swing.set_index("time")["close"])
    setups["idx"] = setups["bar_idx"]
    setups["label"] = (setups["best_R"] >= 1.0).astype(int)   # placeholder; Oracle retrains
    setups["atr"] = setups["atr_lab"] if "atr_lab" in setups.columns else setups["atr"]

    # Drop columns Oracle doesn't expect
    drop = ["best_R", "best_dir", "atr_lab"]
    setups = setups.drop(columns=[c for c in drop if c in setups.columns])

    # Add the legacy aux columns Oracle's schema expects (fill 0)
    EXPECTED_AUX = ["rsi6","stoch_k","stoch_d","bb_pct","mom5","mom10","mom20",
                     "ll_dist10","hh_dist10","vol_accel","atr_ratio","spread_norm"]
    for c in EXPECTED_AUX:
        if c not in setups.columns: setups[c] = 0.0

    # write per-cluster
    K = clusters["cid"].nunique()
    for cid in range(K):
        sub = setups[setups["cid"] == cid].drop(columns=["cid"])
        out_path = os.path.join(DATA, f"setups_{cid}_v74.csv")
        sub.to_csv(out_path, index=False)
        if len(sub):
            n_long = (sub["direction"] == 1).sum(); n_short = (sub["direction"] == -1).sum()
            print(f"  C{cid}: {len(sub):>6,}  ({n_long:,}L/{n_short:,}S)  -> {out_path}")
        else:
            print(f"  C{cid}: 0 fires  -> (skipped)")


if __name__ == "__main__":
    main()
