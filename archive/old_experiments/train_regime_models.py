"""
PER-REGIME MODEL TRAINER
=========================
Trains one XGBoost classifier per cluster file.
Uses walk-forward validation to get honest performance.
Exports each model as ONNX for MT5 deployment.

Run: python train_regime_models.py
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import json, os, warnings
warnings.filterwarnings("ignore")

# ── Try ONNX export (optional) ────────────────────────────────────────────────
try:
    from onnxmltools import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("⚠ onnxmltools not installed — skipping ONNX export. Run: pip install onnxmltools")

BASE_FEATURES = [
    "f01_CPR","f02_WickAsym","f03_BEF","f04_TCS","f05_SPI",
    "f06_LRSlope","f07_RECR","f08_SCM","f09_HLER","f10_EP",
    "f11_KE","f12_MCS","f13_Work","f14_EDR","f15_AI",
    "f16_PPShigh","f16_PPSlow","f17_SCR","f18_RVD","f19_WBER","f20_NCDE",
]
TECH_FEATURES = [
    "rsi14","rsi6","stoch_k","stoch_d","bb_pct",
    "mom5","mom10","mom20",
    "ll_dist10","hh_dist10",
    "vol_accel","atr_ratio","spread_norm",
    "hour_enc","dow_enc",
]
FEATURE_COLS = BASE_FEATURES + TECH_FEATURES  # 36 total

LABEL_COL = "entry_class"  # v3 labels: {0=BUY, 1=FLAT, 2=SELL}

CLUSTER_NAMES = {
    0: "Ranging",
    1: "Downtrend",
    2: "Shock_News",
    3: "Uptrend",
}

# Clusters to skip — empty now. C2 gets a 3-class model like C0.
SKIP_CLUSTERS = set()

# XGBoost params — moderate regularization for v4 recent+short-horizon data
_BASE = dict(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=30,
    reg_alpha=0.5,
    reg_lambda=2.0,
    gamma=0.1,
)
XGB_PARAMS = {
    0: dict(_BASE),
    1: dict(_BASE),
    2: dict(_BASE, n_estimators=150, min_child_weight=15),  # less data in C2
    3: dict(_BASE),
}

# If honest_split=True, train_final() uses only the first 80% of each cluster
# and leaves the last 20% untouched → unbiased backtest on the holdout.
# Controlled by --honest-split CLI flag (set in main()).
HONEST_SPLIT = False
HONEST_TRAIN_FRAC = 0.80
MODEL_SUFFIX = ""  # "" for deployment, "_honest" when honest_split

os.makedirs("models", exist_ok=True)

# ── Load cluster file ─────────────────────────────────────────────────────────
def load_cluster(cluster_id):
    path = f"cluster_{cluster_id}_data.csv"
    df = pd.read_csv(path, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)

    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feat_cols].fillna(0).values

    # v3 labels already encoded as {0=BUY, 1=FLAT, 2=SELL}.
    # LabelEncoder remaps binary subsets contiguously (e.g. {1,2} → {0,1}).
    y_raw = df[LABEL_COL].values
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    print(f"\n{'═'*55}")
    print(f"CLUSTER {cluster_id} — {CLUSTER_NAMES[cluster_id]}")
    print(f"  Rows: {len(df):,}")
    print(f"  Features: {len(feat_cols)}")
    print(f"  Label dist: { {int(k): int(v) for k,v in zip(*np.unique(y_raw, return_counts=True))} }")
    print(f"{'═'*55}")

    return df, X, y, le, feat_cols

# ── Build XGB classifier (binary or multiclass based on n_classes) ───────────
def make_xgb(params, n_classes, y_train=None):
    if n_classes == 2:
        # NO scale_pos_weight — full rebalancing compresses probabilities into
        # a narrow band and kills the ability to rank by confidence. We train
        # on the natural class ratio and use a low probability threshold in
        # the backtest instead.
        return xgb.XGBClassifier(
            **params,
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
            verbosity=0,
        )
    return xgb.XGBClassifier(
        **params,
        objective="multi:softprob",  # softprob → ONNX export includes softmax
        num_class=n_classes,
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=42,
        verbosity=0,
    )

# ── Walk-forward validation ───────────────────────────────────────────────────
def walk_forward(X, y, params, n_classes, n_splits=5):
    """
    Time-series walk-forward: train on past, test on future.
    Never looks ahead.
    """
    n = len(X)
    fold_size = n // (n_splits + 1)
    scores = []

    for i in range(1, n_splits + 1):
        train_end = fold_size * i
        test_start = train_end
        test_end = min(train_end + fold_size, n)

        X_tr, y_tr = X[:train_end], y[:train_end]
        X_te, y_te = X[test_start:test_end], y[test_start:test_end]

        model = make_xgb(params, n_classes, y_train=y_tr)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_te, y_te)],
                  verbose=False)

        preds = model.predict(X_te)
        f1 = f1_score(y_te, preds, average="macro", zero_division=0)
        scores.append(f1)
        print(f"  Fold {i}/{n_splits} | train: {train_end:,} rows | F1: {f1:.4f}")

    print(f"  Walk-forward avg F1: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    return scores

# ── Train final model on all data ─────────────────────────────────────────────
def train_final(X, y, params, n_classes):
    model = make_xgb(params, n_classes, y_train=y)
    model.fit(X, y, verbose=False)
    return model

# ── Feature importance ────────────────────────────────────────────────────────
def print_top_features(model, feat_cols, top=10):
    imp = model.feature_importances_
    ranked = sorted(zip(feat_cols, imp), key=lambda x: -x[1])[:top]
    print(f"\n  Top {top} features:")
    for name, score in ranked:
        bar = "█" * int(score * 200)
        print(f"    {name:<20} {score:.4f} {bar}")

# ── Save model ────────────────────────────────────────────────────────────────
def save_model(model, cluster_id, feat_cols, le, wf_scores):
    name = CLUSTER_NAMES[cluster_id]
    suffix = MODEL_SUFFIX

    # XGBoost native format
    model_path = f"models/regime_{cluster_id}_{name}{suffix}.json"
    model.save_model(model_path)
    print(f"\n  Saved: {model_path}")

    # Metadata
    meta = {
        "cluster_id": cluster_id,
        "regime_name": name,
        "feature_cols": feat_cols,
        "label_classes": le.classes_.tolist(),
        "honest_split": HONEST_SPLIT,
        "train_frac": HONEST_TRAIN_FRAC if HONEST_SPLIT else 1.0,
        "walk_forward_f1_mean": float(np.mean(wf_scores)),
        "walk_forward_f1_std": float(np.std(wf_scores)),
        "walk_forward_scores": [float(s) for s in wf_scores],
    }
    meta_path = f"models/regime_{cluster_id}_{name}{suffix}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved: {meta_path}")

    # ONNX export — only for deployment models, not honest-split evaluation models
    if ONNX_AVAILABLE and not HONEST_SPLIT:
        try:
            n_features = len(feat_cols)
            onnx_model = convert_xgboost(
                model,
                initial_types=[("features", FloatTensorType([1, n_features]))],
                target_opset=15,
            )
            onnx_path = f"models/regime_{cluster_id}_{name}.onnx"
            with open(onnx_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            print(f"  Saved: {onnx_path}")
        except Exception as e:
            print(f"  ONNX export failed: {e}")

    return meta

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global HONEST_SPLIT, MODEL_SUFFIX
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--honest-split", action="store_true",
        help="Train on first 80% of each cluster only. Saves models with '_honest' suffix.")
    args = parser.parse_args()
    HONEST_SPLIT = args.honest_split
    MODEL_SUFFIX = "_honest" if HONEST_SPLIT else ""

    if HONEST_SPLIT:
        print(f"\n>>> HONEST MODE: training on first {HONEST_TRAIN_FRAC*100:.0f}% only, "
              f"last {(1-HONEST_TRAIN_FRAC)*100:.0f}% reserved for unbiased backtest <<<")

    all_meta = {}

    for cluster_id in [0, 1, 2, 3]:
        if cluster_id in SKIP_CLUSTERS:
            print(f"\n{'═'*55}")
            print(f"CLUSTER {cluster_id} — {CLUSTER_NAMES[cluster_id]} — SKIPPED")
            print(f"{'═'*55}")
            continue

        df, X, y, le, feat_cols = load_cluster(cluster_id)

        # Honest split — restrict all training (walk-forward + final) to first 80%
        if HONEST_SPLIT:
            cutoff = int(len(X) * HONEST_TRAIN_FRAC)
            X = X[:cutoff]
            y = y[:cutoff]
            print(f"  Honest split: using first {cutoff:,} rows (last {len(df)-cutoff:,} held out)")

        params = XGB_PARAMS[cluster_id]
        n_classes = len(le.classes_)
        print(f"  Classes: {le.classes_.tolist()} ({n_classes}-class)")

        # Walk-forward
        print("\nWalk-forward validation:")
        wf_scores = walk_forward(X, y, params, n_classes, n_splits=5)

        # Final model on all training data (= first 80% if honest, 100% if not)
        print("\nTraining final model...")
        model = train_final(X, y, params, n_classes)

        # Report
        preds = model.predict(X)
        print(f"\n  In-sample report:")
        names_map = {0: "BUY", 1: "FLAT", 2: "SELL"}
        target_names = [names_map[c] for c in le.classes_]
        print(classification_report(y, preds,
              target_names=target_names, zero_division=0))

        print_top_features(model, feat_cols)

        meta = save_model(model, cluster_id, feat_cols, le, wf_scores)
        all_meta[cluster_id] = meta

    # Summary
    print(f"\n{'═'*55}")
    print("TRAINING SUMMARY")
    print(f"{'═'*55}")
    for cid, meta in all_meta.items():
        f1 = meta["walk_forward_f1_mean"]
        std = meta["walk_forward_f1_std"]
        status = "✅" if f1 > 0.45 else "⚠ " if f1 > 0.35 else "❌"
        print(f"  {status} C{cid} {meta['regime_name']:<15} WF-F1: {f1:.4f} ± {std:.4f}")

    print(f"\nAll models saved to ./models/")
    print("Next: build the runtime selector that loads correct model per week")

if __name__ == "__main__":
    main()
