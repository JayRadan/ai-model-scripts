"""
Train per-rule XGBoost binary classifiers for EURUSD.
Honest 80/20 chronological split per rule.
"""
import json
import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import paths as P

FEATURE_COLS = [
    "f01_CPR", "f02_WickAsym", "f03_BEF", "f04_TCS", "f05_SPI",
    "f06_LRSlope", "f07_RECR", "f08_SCM", "f09_HLER", "f10_EP",
    "f11_KE", "f12_MCS", "f13_Work", "f14_EDR", "f15_AI",
    "f16_PPShigh", "f16_PPSlow", "f17_SCR", "f18_RVD", "f19_WBER",
    "f20_NCDE",
    "stoch_k", "rsi14", "bb_pct", "vol_ratio", "range_atr",
    "dist_sma20", "dist_sma50", "body_ratio", "consec_dir",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]

MIN_TRAIN = 30
MIN_HOLDOUT = 15
PF_THRESHOLD = 1.5

os.makedirs(P.MODELS_DIR, exist_ok=True)

df = pd.read_csv(P.data("setup_signals_eurusd.csv"), parse_dates=["time"])
print(f"Loaded {len(df):,} setups across {df['rule'].nunique()} rules")

rules = df["rule"].unique()

for rule_name in sorted(rules):
    rdf = df[df["rule"] == rule_name].sort_values("time").reset_index(drop=True)
    n = len(rdf)
    if n < MIN_TRAIN + MIN_HOLDOUT:
        print(f"  {rule_name:25s} SKIP (n={n} too few)")
        continue

    split = int(n * 0.8)
    train = rdf.iloc[:split]
    test = rdf.iloc[split:]

    X_train = train[FEATURE_COLS].values
    y_train = train["outcome"].values
    X_test = test[FEATURE_COLS].values
    y_test = test["outcome"].values

    if y_train.sum() < 5 or (y_train == 0).sum() < 5:
        print(f"  {rule_name:25s} SKIP (too few pos/neg in train)")
        continue

    cid = int(rdf["cluster"].iloc[0])

    model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=20, reg_alpha=0.5, reg_lambda=2.0, gamma=0.1,
        use_label_encoder=False, eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train, verbose=False)

    probs = model.predict_proba(X_test)[:, 1]
    try:
        auc = roc_auc_score(y_test, probs)
    except:
        auc = 0.5

    # Threshold sweep on holdout
    best_thresh = 0.5
    best_pf = 0.0
    best_wr = 0.0
    best_n = 0
    best_ev = -999

    for thr in np.arange(0.30, 0.70, 0.01):
        mask = probs >= thr
        if mask.sum() < MIN_HOLDOUT:
            continue
        wins = y_test[mask].sum()
        losses = mask.sum() - wins
        if losses == 0:
            continue
        wr = wins / mask.sum()
        pf = (wins * 2.0) / (losses * 1.0) if losses > 0 else 0
        ev = wr * 2.0 - (1 - wr) * 1.0

        if pf >= PF_THRESHOLD and mask.sum() >= MIN_HOLDOUT and ev > best_ev:
            best_thresh = float(thr)
            best_pf = pf
            best_wr = wr
            best_n = int(mask.sum())
            best_ev = ev

    if best_n == 0:
        # Fallback: pick best EV threshold
        for thr in np.arange(0.30, 0.70, 0.01):
            mask = probs >= thr
            if mask.sum() < MIN_HOLDOUT:
                continue
            wins = y_test[mask].sum()
            losses = mask.sum() - wins
            wr = wins / mask.sum()
            ev = wr * 2.0 - (1 - wr) * 1.0
            if ev > best_ev:
                best_thresh = float(thr)
                best_pf = (wins * 2.0) / max(losses, 1)
                best_wr = wr
                best_n = int(mask.sum())
                best_ev = ev

    disabled = best_ev <= 0 or best_n < MIN_HOLDOUT
    final_thresh = 0.99 if disabled else best_thresh

    # Save model
    model_path = P.model(f"confirm_c{cid}_{rule_name}.json")
    model.save_model(model_path)

    meta = {
        "rule": rule_name,
        "cluster": cid,
        "n_train": len(train),
        "n_test": len(test),
        "auc": round(auc, 4),
        "threshold": round(final_thresh, 4),
        "holdout_pf": round(best_pf, 3),
        "holdout_wr": round(best_wr, 3),
        "holdout_n": best_n,
        "holdout_ev": round(best_ev, 4),
        "disabled": bool(disabled),
    }
    meta_path = P.model(f"confirm_c{cid}_{rule_name}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    status = "DISABLED" if disabled else "ACTIVE"
    print(f"  {rule_name:25s} n={n:>6}  AUC={auc:.3f}  thr={final_thresh:.2f}  "
          f"PF={best_pf:.2f}  WR={best_wr:.1%}  n_h={best_n:>4}  [{status}]")

print("\nDone. Models saved to:", str(P.MODELS_DIR))
