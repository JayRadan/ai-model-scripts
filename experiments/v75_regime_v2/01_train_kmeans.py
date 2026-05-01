"""
v75 step 01: train K-means classifiers with K∈{5,6,7,8} on the rich
per-bar fingerprint matrix. Save selectors + cluster summaries.

Output:
  models/regime_selector_K{k}.json     — scaler + centroids
  reports/cluster_K{k}_summary.csv     — per-cluster mean of each feature
  reports/cluster_K{k}_size.json       — bar counts per cluster
"""
from __future__ import annotations
import os, sys, json, time as _time
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

ZIGZAG = "/home/jay/Desktop/new-model-zigzag"
EXP = os.path.join(ZIGZAG, "experiments/v75_regime_v2")
FP_PATH = os.path.join(EXP, "data/fingerprints_rich.parquet")

FEATS = [
    "ret_1h", "ret_3h", "ret_8h", "ret_24h",
    "slope_1h", "slope_3h", "slope_24h", "slope_accel",
    "vol_1h", "vol_24h", "vol_ratio",
    "pos_24h", "hh_ll_3h", "hh_ll_24h", "streak", "consistency_24h",
]

# To avoid extreme outliers dominating the centroids, clip each feature
# at its 99.5th percentile in absolute value before clustering.
CLIP_PCT = 99.5


def main():
    t0 = _time.time()
    print(f"Loading {FP_PATH}", flush=True)
    fp = pd.read_parquet(FP_PATH)
    X = fp[FEATS].values.astype(np.float64)
    print(f"  {len(fp):,} samples × {X.shape[1]} features")

    # Clip outliers
    for j in range(X.shape[1]):
        lo, hi = np.percentile(X[:, j], [100-CLIP_PCT, CLIP_PCT])
        X[:, j] = np.clip(X[:, j], lo, hi)

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    for K in [5, 6, 7, 8]:
        print(f"\n── K = {K} ──", flush=True)
        km = KMeans(n_clusters=K, random_state=42, n_init=10).fit(Xs)
        labels = km.labels_

        sizes = pd.Series(labels).value_counts().sort_index()
        print(f"  cluster sizes: {sizes.to_dict()}")

        # Per-cluster feature means (raw scale)
        summary = pd.DataFrame(X, columns=FEATS)
        summary["cluster"] = labels
        means = summary.groupby("cluster").mean().round(6)
        means["count"] = sizes.values
        means.to_csv(os.path.join(EXP, f"reports/cluster_K{K}_summary.csv"))
        print("  per-cluster means (top 6 features):")
        print(means[["ret_24h", "ret_3h", "slope_accel", "vol_ratio", "pos_24h", "streak"]].to_string())

        sel = {
            "K": K,
            "features": FEATS,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_std":  scaler.scale_.tolist(),
            "centroids":   km.cluster_centers_.tolist(),
            "inertia":     float(km.inertia_),
        }
        out = os.path.join(EXP, f"models/regime_selector_K{K}.json")
        with open(out, "w") as f:
            json.dump(sel, f)
        print(f"  saved {out}")

        # Persist labels for later analysis
        labels_df = pd.DataFrame({"time": fp["time"].values, "cid": labels})
        labels_df.to_parquet(os.path.join(EXP, f"data/labels_K{K}.parquet"), index=False)

    print(f"\nTotal: {_time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
