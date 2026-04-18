"""
RUNTIME INFERENCE MODULE
========================
Loads the full regime + model pipeline and runs end-to-end inference:

  1. load_pipeline()      — load selector, models, config once
  2. classify_week(bars)  — given M5 bars of a week → cluster_id
  3. predict_signal(bar)  — given a single bar's 36 features + cluster_id → (signal, prob)
  4. decide_trade(...)    — apply threshold + action rule → ("buy" | "sell" | "flat")

Usage example:
    pipe = load_pipeline()
    cluster = classify_week(last_week_bars, pipe)
    for bar in current_week_bars:
        signal, prob = predict_signal(bar, cluster, pipe)
        action = decide_trade(signal, prob, cluster, pipe)
        if action != "flat":
            place_order(action, bar)

NOTE: This is the Option A deployment version. Models were trained on 100%
of cluster data, so live performance will differ from the (optimistic)
in-sample backtest numbers.
"""
import json
import numpy as np
import pandas as pd
import xgboost as xgb

CONFIG_PATH   = "deployment_config.json"
SELECTOR_PATH = "regime_selector_K4.json"


# ── Loading ───────────────────────────────────────────────────────────────────
def load_pipeline(config_path=CONFIG_PATH):
    with open(config_path) as f:
        cfg = json.load(f)
    with open(cfg["regime_selector"]["file"]) as f:
        sel = json.load(f)

    # Load each tradeable cluster's model + meta
    models = {}
    for cid_str, cdef in cfg["clusters"].items():
        cid = int(cid_str)
        if cdef["action"] != "trade":
            continue
        model = xgb.XGBClassifier()
        model.load_model(cdef["model_file"])
        with open(cdef["meta_file"]) as f:
            meta = json.load(f)
        models[cid] = {"model": model, "meta": meta, "cdef": cdef}

    return {
        "config": cfg,
        "selector": {
            "feature_columns":   sel["feature_columns"],
            "scaler_mean":       np.asarray(sel["scaler_mean"]),
            "scaler_std":        np.asarray(sel["scaler_std"]),
            "pre_pca_mean":      np.asarray(sel["pre_pca_mean"]),
            "pre_pca_components":np.asarray(sel["pre_pca_components"]),
            "centroids":         np.asarray(sel["centroids_pre_pca"]),
        },
        "models": models,
    }


# ── Weekly regime classification ──────────────────────────────────────────────
def compute_week_fingerprint(bars):
    """
    Compute the 7 scale-invariant fingerprint features from a week of OHLC bars.
    `bars` must be a DataFrame with columns: close, high, low.
    """
    closes = bars["close"].to_numpy()
    highs  = bars["high"].to_numpy()
    lows   = bars["low"].to_numpy()

    if len(closes) < 2:
        return None

    returns    = np.diff(closes) / closes[:-1]
    bar_ranges = (highs - lows) / closes
    mean_ret   = returns.mean()

    fp = {
        "weekly_return_pct": float(returns.sum()),
        "volatility_pct":    float(returns.std()),
        "trend_consistency": float(np.mean(np.sign(returns) == np.sign(mean_ret))) if mean_ret != 0 else 0.5,
        "trend_strength":    float(returns.sum() / (returns.std() + 1e-9)),
        "volatility":        float(bar_ranges.mean()),
        "range_vs_atr":      float((highs.max() - lows.min()) / closes.mean() / (bar_ranges.mean() + 1e-9)),
    }
    if len(returns) > 2:
        r0, r1 = returns[:-1], returns[1:]
        denom  = r0.std() * r1.std()
        fp["return_autocorr"] = float(np.mean((r0 - r0.mean()) * (r1 - r1.mean())) / (denom + 1e-9))
    else:
        fp["return_autocorr"] = 0.0
    return fp


def classify_week(bars, pipe):
    """Return cluster_id (0-3) for a given week of M5 bars."""
    sel = pipe["selector"]
    fp = compute_week_fingerprint(bars)
    if fp is None:
        return None

    x = np.array([fp[c] for c in sel["feature_columns"]], dtype=np.float64)
    x_scaled = (x - sel["scaler_mean"]) / sel["scaler_std"]
    x_rot    = (x_scaled - sel["pre_pca_mean"]) @ sel["pre_pca_components"].T

    dists = np.linalg.norm(sel["centroids"] - x_rot, axis=1)
    return int(np.argmin(dists))


# ── Per-bar signal prediction ────────────────────────────────────────────────
def predict_signal(bar_features, cluster_id, pipe):
    """
    Given a dict of the 36 features and the active cluster, return
    (action_label, actionable_prob). action_label is one of {0, 1, 2}
    from the v3 encoding; 1=FLAT means no signal.
    """
    if cluster_id not in pipe["models"]:
        return 1, 0.0  # skipped cluster → always FLAT

    bundle = pipe["models"][cluster_id]
    model  = bundle["model"]
    meta   = bundle["meta"]

    feat_order = meta["feature_cols"]
    x = np.array([[bar_features[c] for c in feat_order]], dtype=np.float64)
    probs = model.predict_proba(x)[0]

    classes = meta["label_classes"]
    best_prob, best_label = 0.0, 1
    for col_idx, c in enumerate(classes):
        if c == 1:  # FLAT
            continue
        if probs[col_idx] > best_prob:
            best_prob  = float(probs[col_idx])
            best_label = int(c)
    return best_label, best_prob


def decide_trade(label, prob, cluster_id, pipe):
    """
    Apply the per-cluster probability threshold.
    Returns: "buy" | "sell" | "flat"
    """
    if cluster_id not in pipe["models"]:
        return "flat"
    cdef = pipe["models"][cluster_id]["cdef"]
    if prob < cdef["probability_threshold"]:
        return "flat"
    if label == 0:
        return "buy"
    if label == 2:
        return "sell"
    return "flat"


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading pipeline...")
    pipe = load_pipeline()
    print(f"  loaded {len(pipe['models'])} tradeable cluster models: {sorted(pipe['models'].keys())}")
    print(f"  selector: {len(pipe['selector']['feature_columns'])} fingerprint features, "
          f"{len(pipe['selector']['centroids'])} centroids")

    # Pull the last full week from labeled_v3 as a sanity check
    df = pd.read_csv("labeled_v3.csv", parse_dates=["time"]).sort_values("time")
    last_time = df["time"].max()
    week_bars = df[df["time"] > last_time - pd.Timedelta(days=7)]
    print(f"\nLast week: {week_bars['time'].min()} → {week_bars['time'].max()} "
          f"({len(week_bars):,} bars)")

    cluster = classify_week(week_bars, pipe)
    cluster_name = pipe["config"]["clusters"][str(cluster)]["name"]
    print(f"Predicted regime: C{cluster} {cluster_name}")

    # Predict on the last bar
    last_bar = week_bars.iloc[-1]
    feat_cols = pipe["config"]["feature_columns_36"]
    bar_feats = {c: float(last_bar[c]) for c in feat_cols if c in last_bar.index}
    label, prob = predict_signal(bar_feats, cluster, pipe)
    action = decide_trade(label, prob, cluster, pipe)
    names = {0: "BUY", 1: "FLAT", 2: "SELL"}
    print(f"Last bar prediction: label={names.get(label,'?')} prob={prob:.4f} → action={action}")
