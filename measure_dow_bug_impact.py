"""
Quantify impact of the DOW off-by-one bug.

For each instrument + rule, replay the holdout setups through the trained
classifier under two conditions:
  A) CORRECT features (as trained)
  B) SHIFTED dow features (simulating the live MQL5 bug: pandas Mon=0 shifted
     to match MQL5 Sun=0 convention)

Compare: prob distribution shift, fraction of decisions flipped at threshold,
feature importance of dow features.
"""
from __future__ import annotations
import glob, json, os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier


def replay(setups_path, models_dir, feat_cols, dow_cols, is_xau=False):
    df = pd.read_csv(setups_path, parse_dates=["time"])
    # time → pandas dayofweek (Mon=0..Sun=6)
    py_dow = df["time"].dt.dayofweek.astype(int).to_numpy()
    # Simulate MQL5 bug: live used (py_dow + 1) % 7, but trained on py_dow
    bug_dow = (py_dow + 1) % 7

    # Recompute dow features under both scenarios
    correct = {}
    buggy = {}
    for col in dow_cols:
        if col == "dow_enc":
            correct[col] = np.sin(2*np.pi*py_dow/5.0)
            buggy[col]   = np.sin(2*np.pi*bug_dow/5.0)
        elif col == "dow_sin":
            correct[col] = np.sin(2*np.pi*py_dow/5.0)
            buggy[col]   = np.sin(2*np.pi*bug_dow/5.0)
        elif col == "dow_cos":
            correct[col] = np.cos(2*np.pi*py_dow/5.0)
            buggy[col]   = np.cos(2*np.pi*bug_dow/5.0)

    results = []
    # group by rule / cluster to pick right model
    if is_xau:
        # For XAU, setups are pre-split by cluster; need cluster id from filename
        cid = int(setups_path.rsplit("_", 1)[-1].split(".")[0])
        groups = [(cid, r, g) for r, g in df.groupby("rule")]
    else:
        groups = [(int(r['cluster'].iloc[0]), rn, r)
                  for rn, r in df.groupby("rule")]

    per_rule = []
    for cid, rname, rdf in groups:
        mpath = f"{models_dir}/confirm_c{cid}_{rname}.json"
        mmeta = f"{models_dir}/confirm_c{cid}_{rname}_meta.json"
        if not (os.path.exists(mpath) and os.path.exists(mmeta)): continue
        meta = json.load(open(mmeta))
        thr = meta["threshold"]
        if meta.get("disabled", False): continue

        rdf_sorted = rdf.sort_values("time").reset_index(drop=True)
        split = int(len(rdf_sorted) * 0.8)
        test = rdf_sorted.iloc[split:].copy()
        if len(test) < 20: continue

        mdl = XGBClassifier(); mdl.load_model(mpath)
        use_cols = meta.get("feature_cols", feat_cols)

        # Build X_correct and X_buggy
        X_correct = test[use_cols].fillna(0).copy()
        X_buggy   = X_correct.copy()
        for col in dow_cols:
            if col in X_correct.columns:
                idx_in_test = rdf_sorted.index[split:]
                X_correct[col] = correct[col][idx_in_test.to_numpy()]
                X_buggy[col]   = buggy[col][idx_in_test.to_numpy()]

        p_correct = mdl.predict_proba(X_correct.values)[:,1]
        p_buggy   = mdl.predict_proba(X_buggy.values)[:,1]

        decided_correct = p_correct >= thr
        decided_buggy   = p_buggy   >= thr
        flips = decided_correct != decided_buggy

        # Feature importance of dow cols
        booster = mdl.get_booster()
        imp = booster.get_score(importance_type="gain")
        fnames = booster.feature_names or [f"f{i}" for i in range(len(use_cols))]
        # xgb names features "f0", "f1", ...
        dow_imp = 0.0; total_imp = sum(imp.values()) or 1.0
        for c in dow_cols:
            if c in use_cols:
                idx = use_cols.index(c)
                dow_imp += imp.get(f"f{idx}", 0.0)

        per_rule.append({
            "rule": rname, "cid": cid, "n": int(len(test)),
            "mean_prob_correct": float(p_correct.mean()),
            "mean_prob_buggy":   float(p_buggy.mean()),
            "mean_abs_prob_diff": float(np.abs(p_correct - p_buggy).mean()),
            "pct_flipped": float(flips.mean() * 100),
            "dow_gain_pct": float(dow_imp / total_imp * 100),
            "threshold": thr,
        })
    return per_rule


XAU_FEATS = [
    "f01_CPR","f02_WickAsym","f03_BEF","f04_TCS","f05_SPI","f06_LRSlope","f07_RECR","f08_SCM","f09_HLER","f10_EP",
    "f11_KE","f12_MCS","f13_Work","f14_EDR","f15_AI","f16_PPShigh","f16_PPSlow","f17_SCR","f18_RVD","f19_WBER","f20_NCDE",
    "rsi14","rsi6","stoch_k","stoch_d","bb_pct","mom5","mom10","mom20","ll_dist10","hh_dist10",
    "vol_accel","atr_ratio","spread_norm","hour_enc","dow_enc",
]
EUGJ_FEATS = [
    "f01_CPR","f02_WickAsym","f03_BEF","f04_TCS","f05_SPI","f06_LRSlope","f07_RECR","f08_SCM","f09_HLER","f10_EP",
    "f11_KE","f12_MCS","f13_Work","f14_EDR","f15_AI","f16_PPShigh","f16_PPSlow","f17_SCR","f18_RVD","f19_WBER","f20_NCDE",
    "stoch_k","rsi14","bb_pct","vol_ratio","range_atr","dist_sma20","dist_sma50","body_ratio","consec_dir",
    "hour_sin","hour_cos","dow_sin","dow_cos",
]


def summarize(name, rows):
    if not rows:
        print(f"{name}: no rules"); return
    n_tot = sum(r["n"] for r in rows)
    w_flip = sum(r["pct_flipped"] * r["n"] for r in rows) / n_tot
    w_prob = sum(r["mean_abs_prob_diff"] * r["n"] for r in rows) / n_tot
    w_imp  = sum(r["dow_gain_pct"] * r["n"] for r in rows) / n_tot
    print(f"\n{'='*72}\n{name}")
    print(f"  rules compared: {len(rows)}  total holdout setups: {n_tot:,}")
    print(f"  weighted-avg DOW feature importance (gain): {w_imp:.2f}%")
    print(f"  weighted-avg |prob shift| when DOW wrong:   {w_prob:.4f}")
    print(f"  weighted-avg % of decisions FLIPPED:        {w_flip:.2f}%")
    worst = sorted(rows, key=lambda r: -r["pct_flipped"])[:5]
    print("  worst-affected rules:")
    for w in worst:
        print(f"    c{w['cid']} {w['rule']:<25} n={w['n']:4d}  "
              f"flip={w['pct_flipped']:5.2f}%  Δprob={w['mean_abs_prob_diff']:.4f}  "
              f"dow_gain={w['dow_gain_pct']:4.1f}%")


def main():
    # XAU — per-cluster setups_{cid}.csv
    xau_rows = []
    for p in sorted(glob.glob("/home/jay/Desktop/new-model-zigzag/data/setups_*.csv")):
        xau_rows += replay(p, "/home/jay/Desktop/new-model-zigzag/models",
                           XAU_FEATS, ["dow_enc"], is_xau=True)
    summarize("XAUUSD (Midas)", xau_rows)

    # EURUSD
    eu_rows = replay("/home/jay/Desktop/new-model-zigzag/eurusd/data/setup_signals_eurusd.csv",
                     "/home/jay/Desktop/new-model-zigzag/eurusd/models",
                     EUGJ_FEATS, ["dow_sin","dow_cos"], is_xau=False)
    summarize("EURUSD (Meridian)", eu_rows)

    # GBPJPY
    gj_rows = replay("/home/jay/Desktop/new-model-zigzag/gbpjpy/data/setup_signals_gbpjpy.csv",
                     "/home/jay/Desktop/new-model-zigzag/gbpjpy/models",
                     EUGJ_FEATS, ["dow_sin","dow_cos"], is_xau=False)
    summarize("GBPJPY (Samurai)", gj_rows)


if __name__ == "__main__":
    main()
