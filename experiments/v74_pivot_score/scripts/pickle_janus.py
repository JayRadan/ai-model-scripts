"""
Janus (v7.4 Pivot Score) — production pickle.

Trains the FULL Janus stack on pre-2024-12-12 data (matches Oracle's cutoff so
the 16-month holdout PF 2.50 / WR 54.9% is what we ship), uses meta criterion
B (max PF, retain >= 30%) for threshold selection, and bundles everything
into a single pickle for the server.

Output: models/janus_xau_validated.pkl
Usage:  python3 pickle_janus.py
"""
from __future__ import annotations
import os, sys, pickle, subprocess, time, json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

ZIGZAG = "/home/jay/Desktop/new-model-zigzag"
sys.path.insert(0, os.path.join(ZIGZAG, "model_pipeline"))
sys.path.insert(0, os.path.join(ZIGZAG, "experiments/v74_pivot_score"))

import paths as P
from importlib.machinery import SourceFileLoader
WF = SourceFileLoader("wf", os.path.join(ZIGZAG, "experiments/v74_pivot_score/05_walk_forward.py")).load_module()

CUTOFF = pd.Timestamp("2024-12-12 00:00:00")
DATA = os.path.join(ZIGZAG, "experiments/v74_pivot_score/data")
OUT_PATH = os.path.join(ZIGZAG, "models/janus_xau_validated.pkl")


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "-C", ZIGZAG, "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def select_threshold_max_pf(sweep_rows, min_retain=0.30):
    eligible = [r for r in sweep_rows if r["retain"] >= min_retain]
    if not eligible: eligible = sweep_rows
    return max(eligible, key=lambda r: r["pf"])


def main():
    t0 = time.time()
    print(f"Janus pickle — cutoff {CUTOFF}", flush=True)

    print("[1/8] loading features + labels + clusters + swing...", flush=True)
    feats = pd.read_csv(os.path.join(DATA, "features_v74.csv"), parse_dates=["time"])
    labs = pd.read_csv(os.path.join(DATA, "labels_v74.csv"), parse_dates=["time"])
    df = feats.merge(labs, on="time", how="inner", suffixes=("", "_lab"))
    clusters = pd.read_csv(WF.CLUSTERS_PATH, parse_dates=["time"])[["time", "cid"]]
    swing, atr = WF.build_swing_with_exit_feats()

    pivot_feats = [c for c in df.columns if c not in WF.NON_FEAT_PIVOT
                    and df[c].dtype != object and not c.endswith("_lab")
                    and not WF._is_f_feat(c)]   # drop f01-f20 (not on server)
    print(f"  rows={len(df):,}  pivot_feats={len(pivot_feats)}")

    pre = df[df["time"] < CUTOFF].reset_index(drop=True)

    print(f"[2/8] training pivot-score on pre={len(pre):,}...", flush=True)
    score_mdl = XGBClassifier(**WF.XGB_S)
    score_mdl.fit(pre[pivot_feats].fillna(0).values, pre[WF.TARGET_COL].values)

    print(f"[3/8] training direction model on {(pre[WF.TARGET_COL]==1).sum():,} pivot rows...", flush=True)
    pos = pre[pre[WF.TARGET_COL] == 1]
    dir_mdl = XGBClassifier(**WF.XGB_S)
    dir_mdl.fit(pos[pivot_feats].fillna(0).values, (pos["best_dir"] == 1).astype(int).values)

    print(f"[4/8] scoring pre + building setups (thr={WF.SCORE_THR})...", flush=True)
    p_pre = score_mdl.predict_proba(pre[pivot_feats].fillna(0).values)[:, 1]
    pdir_pre = dir_mdl.predict_proba(pre[pivot_feats].fillna(0).values)[:, 1]
    m = p_pre >= WF.SCORE_THR
    pre_setups = pre[m].copy()
    pre_setups["direction"] = np.where(pdir_pre[m] >= 0.5, 1, -1)
    pre_setups["rule"] = "RP_score"
    pre_setups["idx"] = pre_setups["bar_idx"]
    pre_setups["entry_price"] = pre_setups["time"].map(swing.set_index("time")["close"])
    pre_setups["label"] = (pre_setups["best_R"] >= 1.0).astype(int)
    pre_setups = pre_setups.merge(clusters, on="time", how="left")
    print(f"  pre_setups={len(pre_setups):,}")

    print(f"[5/8] training per-(cid, rule) confirm head...", flush=True)
    mdls, thrs = WF.train_conf(pre_setups, WF.V72L_FEATS)
    pre_conf = WF.filter_setups(pre_setups, mdls, thrs, WF.V72L_FEATS)
    print(f"  pre_conf={len(pre_conf):,}  confirm_heads={len(mdls)}")

    print(f"[6/8] training exit head...", flush=True)
    exit_mdl = WF.train_exit(pre_conf, swing, atr)

    print(f"[7/8] simulating pre-confirmed trades for meta...", flush=True)
    pre_trades = WF.simulate(pre_conf, swing, atr, exit_mdl)
    md = pre_trades.merge(pre_conf[["time","cid","rule"] + WF.V72L_FEATS],
                           on=["time","cid","rule"], how="left")
    md["meta_label"] = (md["pnl_R"] > 0).astype(int)
    md["direction"] = md["direction"].astype(int); md["cid"] = md["cid"].astype(int)
    val_n = max(500, int(len(md) * 0.10))
    mtr = md.iloc[:-val_n].reset_index(drop=True)
    mvd = md.iloc[-val_n:].reset_index(drop=True)
    meta_mdl = XGBClassifier(**WF.XGB_C)
    meta_mdl.fit(mtr[WF.META_FEATS].fillna(0).values, mtr["meta_label"].values)

    pv = meta_mdl.predict_proba(mvd[WF.META_FEATS].fillna(0).values)[:, 1]
    pn = mvd["pnl_R"].values; yv = mvd["meta_label"].values
    sweep = []
    for thr in np.linspace(0.05, 0.95, 37):
        mm = pv >= thr
        if mm.sum() == 0: continue
        sub_pn = pn[mm]
        wins = sub_pn[sub_pn > 0].sum(); losses = -sub_pn[sub_pn <= 0].sum()
        pf = (wins / losses) if losses > 0 else float("inf")
        retain = mm.sum() / len(pv)
        sweep.append({"thr": float(thr), "n": int(mm.sum()), "retain": float(retain),
                       "wr": float(yv[mm].mean()), "pf": float(pf), "expect": float(sub_pn.mean()),
                       "total_R": float(sub_pn.sum())})
    chosen = select_threshold_max_pf(sweep, min_retain=0.30)
    print(f"  selected meta_threshold={chosen['thr']:.3f}  (PF={chosen['pf']:.2f}  retain={chosen['retain']*100:.0f}%)")

    print(f"[8/8] bundling pickle...", flush=True)
    payload = {
        "pivot_score_mdl": score_mdl,
        "pivot_dir_mdl":   dir_mdl,
        "score_threshold": float(WF.SCORE_THR),
        "mdls":            mdls,
        "thrs":            thrs,
        "exit_mdl":        exit_mdl,
        "meta_mdl":        meta_mdl,
        "meta_threshold":  float(chosen["thr"]),
        "pivot_feats":     list(pivot_feats),
        "v72l_feats":      list(WF.V72L_FEATS),
        "meta_feats":      list(WF.META_FEATS),
        "exit_feats":      list(WF.EXIT_FEATS),
        "exit_threshold":  float(WF.EXIT_THRESHOLD),
        "hard_sl":         float(WF.SL_HARD),
        "max_hold":        int(WF.MAX_HOLD),
        "min_hold":        2,
        "trained_on":      f"pre-{CUTOFF.date()} (walk-forward 4-fold validated; mean PF 2.07, F4 PF 2.61)",
        "meta_sweep":      sweep,
        "meta_select_criterion": "max PF, retain >= 30%",
        "git_rev":         git_rev(),
        "product":         "janus_xau",
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    sz = os.path.getsize(OUT_PATH) / 1e6
    print(f"\nWrote {OUT_PATH}  ({sz:.1f} MB)")
    print(f"  pivot_score_mdl + pivot_dir_mdl   (NEW for Janus)")
    print(f"  confirm heads: {len(mdls)}  exit_mdl  meta_mdl@thr={payload['meta_threshold']:.3f}")
    print(f"  pivot_feats={len(pivot_feats)}  v72l_feats={len(payload['v72l_feats'])}")
    print(f"  total: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
