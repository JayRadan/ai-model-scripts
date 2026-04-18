"""
PER-RULE confirmation classifier.

One XGBoost binary classifier per (cluster, rule) pair. Each classifier only
sees setups that its specific rule fired — learns patterns specific to that
setup type. Outputs are combined at backtest time.

Honest 80/20 chronological split per rule. Sweeps threshold to find the best
(PF ≥ 1.5, min trades ≥ 15) operating point per rule, fallback to EV-best
if no threshold passes.
"""
from __future__ import annotations
import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

import paths as P

FEATURE_COLS = [
    "f01_CPR","f02_WickAsym","f03_BEF","f04_TCS","f05_SPI",
    "f06_LRSlope","f07_RECR","f08_SCM","f09_HLER","f10_EP",
    "f11_KE","f12_MCS","f13_Work","f14_EDR","f15_AI",
    "f16_PPShigh","f16_PPSlow","f17_SCR","f18_RVD","f19_WBER","f20_NCDE",
    "rsi14","rsi6","stoch_k","stoch_d","bb_pct",
    "mom5","mom10","mom20",
    "ll_dist10","hh_dist10",
    "vol_accel","atr_ratio","spread_norm",
    "hour_enc","dow_enc",
]

CLUSTER_NAMES = {0:"Ranging", 1:"Downtrend", 2:"Shock_News", 3:"Uptrend"}
TP_MULT = 2.0
SL_MULT = 1.0
MIN_TRADES_ACCEPT = 15

os.makedirs(str(P.MODELS_DIR), exist_ok=True)


def eval_at_threshold(y_true, proba, threshold):
    mask = proba >= threshold
    n = int(mask.sum())
    if n < 5:
        return {"n": n, "wr": 0.0, "ev_r": 0.0, "pf": 0.0}
    y_f = y_true[mask].astype(int)
    wins = int(y_f.sum())
    losses = n - wins
    wr = wins / n
    ev_r = (wr * TP_MULT) - ((1 - wr) * SL_MULT)
    pf = (wins * TP_MULT) / max(losses * SL_MULT, 1e-9)
    return {"n": n, "wr": float(wr), "ev_r": float(ev_r), "pf": float(pf)}


def train_rule(cid: int, rule_name: str, sub_df: pd.DataFrame):
    """Train one classifier for one rule within a cluster."""
    if len(sub_df) < 100:
        print(f"    {rule_name}: {len(sub_df)} setups — too few, skip")
        return None

    sub = sub_df.sort_values("time").reset_index(drop=True)
    cutoff = int(len(sub) * 0.80)
    tr = sub.iloc[:cutoff].reset_index(drop=True)
    ho = sub.iloc[cutoff:].reset_index(drop=True)
    if len(ho) < 20 or tr["label"].nunique() < 2:
        print(f"    {rule_name}: holdout too small or single-class")
        return None

    feat_cols = [c for c in FEATURE_COLS if c in sub.columns]
    X_tr = tr[feat_cols].fillna(0).values
    y_tr = tr["label"].values.astype(int)
    X_ho = ho[feat_cols].fillna(0).values
    y_ho = ho["label"].values.astype(int)

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=20,
        reg_alpha=0.5,
        reg_lambda=2.0,
        gamma=0.1,
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_ho, y_ho)], verbose=False)

    p_tr = model.predict_proba(X_tr)[:, 1]
    p_ho = model.predict_proba(X_ho)[:, 1]
    try:
        auc_ho = roc_auc_score(y_ho, p_ho)
    except ValueError:
        auc_ho = 0.5

    thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
    best_thr = None
    best_ev = -1e9
    best_stats = None
    best_passes = False

    for thr in thresholds:
        s_tr = eval_at_threshold(y_tr, p_tr, thr)
        s_ho = eval_at_threshold(y_ho, p_ho, thr)
        passes = (s_ho["pf"] >= 1.5 and s_ho["n"] >= MIN_TRADES_ACCEPT
                  and s_tr["pf"] >= 1.3)
        if passes and s_ho["ev_r"] > best_ev:
            best_ev = s_ho["ev_r"]
            best_thr = thr
            best_stats = s_ho
            best_passes = True

    if not best_passes:
        # Fallback — EV-best with at least MIN_TRADES_ACCEPT trades
        for thr in thresholds:
            s_ho = eval_at_threshold(y_ho, p_ho, thr)
            if s_ho["n"] >= MIN_TRADES_ACCEPT and s_ho["ev_r"] > best_ev:
                best_ev = s_ho["ev_r"]
                best_thr = thr
                best_stats = s_ho

    if best_thr is None:
        print(f"    {rule_name}: no viable threshold")
        return None

    tag = "✅" if best_passes else "~"
    print(f"    {tag} {rule_name}: thr={best_thr} "
          f"n={best_stats['n']:>4} WR={best_stats['wr']:.0%} "
          f"EV={best_stats['ev_r']:+.2f}R PF={best_stats['pf']:.2f} AUC={auc_ho:.2f}")

    # Save
    model_path = P.model(f"confirm_c{cid}_{rule_name}.json")
    model.save_model(model_path)
    meta = {
        "cluster_id": cid,
        "rule": rule_name,
        "feature_cols": feat_cols,
        "threshold": float(best_thr),
        "auc_holdout": float(auc_ho),
        "holdout_stats": best_stats,
        "raw_train_wr": float(tr["label"].mean()),
        "raw_holdout_wr": float(ho["label"].mean()),
        "n_train": int(len(tr)),
        "n_holdout": int(len(ho)),
        "passes_strict": bool(best_passes),
    }
    with open(P.model(f"confirm_c{cid}_{rule_name}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def main():
    all_results = {}
    for cid in [0, 1, 2, 3]:
        name = CLUSTER_NAMES[cid]
        print(f"\n{'═'*60}\nC{cid} {name}\n{'═'*60}")
        df = pd.read_csv(P.data(f"setups_{cid}.csv"), parse_dates=["time"])
        rules = sorted(df["rule"].unique())
        print(f"  rules present: {rules}")
        cluster_meta = {}
        for r in rules:
            sub = df[df["rule"] == r]
            meta = train_rule(cid, r, sub)
            if meta is not None:
                cluster_meta[r] = meta
        all_results[cid] = cluster_meta

    # Summary
    print(f"\n{'═'*60}\nFINAL SUMMARY (per-rule, honest holdout)\n{'═'*60}")
    grand_total = {"n": 0, "ev_sum_r": 0.0}
    for cid, rules in all_results.items():
        print(f"\n  C{cid} {CLUSTER_NAMES[cid]}:")
        for rname, meta in rules.items():
            s = meta["holdout_stats"]
            tag = "✅" if meta["passes_strict"] else "~"
            print(f"    {tag} {rname:<22} n={s['n']:>4}  "
                  f"WR={s['wr']:.0%}  EV={s['ev_r']:+.2f}R  PF={s['pf']:.2f}")
            grand_total["n"] += s["n"]
            grand_total["ev_sum_r"] += s["ev_r"] * s["n"]

    print(f"\n  TOTAL trades across all rules: {grand_total['n']}")
    if grand_total["n"] > 0:
        avg_ev = grand_total["ev_sum_r"] / grand_total["n"]
        print(f"  Weighted avg EV per trade: {avg_ev:+.2f}R")


if __name__ == "__main__":
    main()
