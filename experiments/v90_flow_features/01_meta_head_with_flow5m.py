"""v9.0 Step 1 — Add flow_5m as a 19th feature to Oracle XAU meta head.

Cheap validation experiment before committing to a full retrain:
  - Compute flow_5m on full XAU history
  - Merge into setups_*_v72l.csv at trade times
  - Retrain meta head with 19 META_FEATS instead of 18
  - Evaluate on chronological holdout (2024-12-12+)
  - Compare PF/WR to current baseline (PF 3.48 / WR 65.3% pre-cohort-kill)

If meta head's feature_importance for flow_5m > some threshold (e.g. > 0.02)
AND PF/WR improves materially → flow_5m has residual signal → worth the
full retrain across confirm heads.
If not → abandon, save time.
"""
from __future__ import annotations
import os, sys, glob, pickle
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "/home/jay/Desktop/new-model-zigzag/experiments/v89_quantum_flow_tiebreaker")
from importlib.machinery import SourceFileLoader
qf = SourceFileLoader("qf01", "/home/jay/Desktop/new-model-zigzag/experiments/v89_quantum_flow_tiebreaker/01_port_and_test.py").load_module()

ROOT = "/home/jay/Desktop/new-model-zigzag"
HOLDOUT = pd.Timestamp("2024-12-12")

V72L_FEATS = [
    "hurst_rs","ou_theta","entropy_rate","kramers_up","wavelet_er",
    "vwap_dist","hour_enc","dow_enc",
    "quantum_flow","quantum_flow_h4","quantum_momentum","quantum_vwap_conf",
    "quantum_divergence","quantum_div_strength",
    "vpin","sig_quad_var","har_rv_ratio","hawkes_eta",
]
META_FEATS_BASE  = V72L_FEATS + ["direction", "cid"]
META_FEATS_NEW   = META_FEATS_BASE + ["flow_5m"]

def pf(rs):
    pos = rs[rs>0].sum(); neg = -rs[rs<=0].sum()
    return pos / max(neg, 1e-9)


def main():
    print("Step 1/4: load XAU swing + compute flow_5m...", flush=True)
    swing = pd.read_csv(os.path.join(ROOT, "data/swing_v5_xauusd.csv"),
                          parse_dates=["time"])
    swing = swing[["time","open","high","low","close","spread"]].sort_values("time").reset_index(drop=True)
    print(f"  swing: {len(swing)} bars", flush=True)
    flow_5m = qf.quantum_flow(swing)
    swing["flow_5m"] = flow_5m.values
    print(f"  flow_5m computed (median {np.median(flow_5m):.2f})", flush=True)

    print("\nStep 2/4: load setups + merge flow_5m + holdout trades...", flush=True)
    setups = []
    for f in sorted(glob.glob(os.path.join(ROOT, "data/setups_*_v72l.csv"))):
        cid = int(os.path.basename(f).split("_")[1])
        df = pd.read_csv(f, parse_dates=["time"])
        df["cid"] = cid
        setups.append(df)
    setups = pd.concat(setups, ignore_index=True).sort_values("time").reset_index(drop=True)
    setups["cid"] = setups["cid"].astype(int)
    setups["direction"] = setups["direction"].astype(int)
    setups = setups.loc[:, ~setups.columns.duplicated()]   # drop duplicates if any
    setups = setups.merge(swing[["time","flow_5m"]], on="time", how="left")
    print(f"  setups: {len(setups)} rows  ({setups.flow_5m.isna().sum()} NaN flow)", flush=True)

    # Holdout trades — these are the simulated ones with realized pnl_R
    trades = pd.read_csv(os.path.join(ROOT, "data/v72l_trades_holdout.csv"),
                          parse_dates=["time"])
    trades["cid"] = trades["cid"].astype(int)
    trades["direction"] = trades["direction"].astype(int)

    # We need to label META training data. The validation script uses train trades
    # (pre-2024-12-12) which were simulated by the validation pipeline. Since we
    # don't have those train trades on disk, we can't fully retrain meta. BUT we can
    # do this: use the v72l_trades_holdout as POST-2024-12-12, and ALSO fetch
    # /tmp cached train trades from the most recent validation run.
    # For now: chronological split of the 1367 holdout trades (50/50)
    # to get a quick "would adding flow_5m help" estimate.
    print("\nStep 3/4: prepare META training data (chrono-split of holdout)...", flush=True)
    # Attach features
    needed_cols = ["time","cid","rule","direction"] + V72L_FEATS + ["flow_5m"]
    needed_cols = list(dict.fromkeys(needed_cols))   # dedupe preserving order
    full = trades.merge(setups[needed_cols],
                          on=["time","cid","rule","direction"], how="left")
    # Filter
    full = full.dropna(subset=V72L_FEATS).reset_index(drop=True)
    full["meta_label"] = (full["pnl_R"] > 0).astype(int)
    full = full.sort_values("time").reset_index(drop=True)
    n_tr = len(full) // 2
    train, test = full.iloc[:n_tr], full.iloc[n_tr:]
    print(f"  train: {len(train)}  test: {len(test)}  win rate train={train.meta_label.mean():.3f} test={test.meta_label.mean():.3f}", flush=True)

    # ---- BASELINE: 20 META_FEATS ----
    print("\nStep 4/4: train BASELINE meta (20 feats)...", flush=True)
    Xtr = train[META_FEATS_BASE].fillna(0).values
    ytr = train["meta_label"].values
    Xte = test [META_FEATS_BASE].fillna(0).values
    yte = test ["meta_label"].values
    rs_te = test ["pnl_R"].values

    mdl_base = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                              eval_metric="logloss", verbosity=0, random_state=0)
    mdl_base.fit(Xtr, ytr)
    p_base = mdl_base.predict_proba(Xte)[:,1]
    auc_base = roc_auc_score(yte, p_base) if len(set(yte))>1 else 0.5

    # ---- WITH FLOW_5M: 21 META_FEATS ----
    print("Train WITH flow_5m (21 feats)...", flush=True)
    train_full = train.copy(); train_full["flow_5m"] = train_full["flow_5m"].fillna(0)
    test_full  = test.copy();  test_full ["flow_5m"] = test_full ["flow_5m"].fillna(0)
    Xtr_n = train_full[META_FEATS_NEW].values
    Xte_n = test_full [META_FEATS_NEW].values
    mdl_new = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                              eval_metric="logloss", verbosity=0, random_state=0)
    mdl_new.fit(Xtr_n, ytr)
    p_new = mdl_new.predict_proba(Xte_n)[:,1]
    auc_new = roc_auc_score(yte, p_new) if len(set(yte))>1 else 0.5

    # Feature importance for flow_5m
    fi = mdl_new.feature_importances_
    flow_imp = float(fi[-1])
    flow_rank = int(np.argsort(-fi).tolist().index(len(fi)-1)) + 1

    print(f"\n=== RESULTS on holdout test slice (n={len(test)}) ===")
    print(f"  Baseline meta AUC:        {auc_base:.3f}")
    print(f"  +flow_5m  meta AUC:       {auc_new:.3f}  (delta {auc_new-auc_base:+.3f})")
    print(f"  flow_5m feature importance: {flow_imp:.4f}  (rank {flow_rank}/{len(fi)})")

    # Apply each meta to threshold 0.675 (production setting) and compute trade outcomes
    THR = 0.675
    keep_base = p_base >= THR
    keep_new  = p_new  >= THR
    print(f"\n  At meta_threshold {THR}:")
    print(f"  Baseline:  kept {keep_base.sum()}/{len(test)}  WR {(rs_te[keep_base]>0).mean()*100:.1f}%  PF {pf(rs_te[keep_base]):.2f}  R {rs_te[keep_base].sum():+.0f}")
    print(f"  +flow_5m:  kept {keep_new.sum()}/{len(test)}   WR {(rs_te[keep_new]>0).mean()*100:.1f}%  PF {pf(rs_te[keep_new]):.2f}  R {rs_te[keep_new].sum():+.0f}")

    print(f"\nVERDICT:")
    if flow_imp > 0.02 and auc_new - auc_base > 0.005:
        print(f"  ✅ flow_5m carries residual signal (importance {flow_imp:.4f}, ΔAUC {auc_new-auc_base:+.3f})")
        print(f"     Worth proceeding to Step 2 — full retrain with flow_5m in confirm heads")
    elif flow_imp > 0.01:
        print(f"  ⚠ flow_5m has minor signal (importance {flow_imp:.4f}, ΔAUC {auc_new-auc_base:+.3f})")
        print(f"     Probably not worth the full retrain effort. Could ship as opt-in feature.")
    else:
        print(f"  ❌ flow_5m carries little incremental signal (importance {flow_imp:.4f}, ΔAUC {auc_new-auc_base:+.3f})")
        print(f"     The 18 v72L features already capture what's available. Don't retrain.")


if __name__ == "__main__":
    main()
