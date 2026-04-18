"""
EURUSD regime clustering — K-means on weekly scale-invariant fingerprints.
Same approach as gold: 7 features computed from % returns so clusters
represent market regimes, not price eras.
"""
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import paths as P

DATA_PATH = P.data("labeled_eurusd.csv")
K = 4
MIN_CANDLES = 200

print("Loading data...")
df = pd.read_csv(DATA_PATH, parse_dates=["time"])

# Weekly aggregation
df["week"] = df["time"].dt.isocalendar().year.astype(str) + "-W" + df["time"].dt.isocalendar().week.astype(str).str.zfill(2)

FINGERPRINT_COLS = [
    "weekly_return_pct",
    "volatility_pct",
    "trend_consistency",
    "trend_strength",
    "volatility",
    "range_vs_atr",
    "return_autocorr",
]

weeks = []
for wk, g in df.groupby("week"):
    if len(g) < MIN_CANDLES:
        continue
    c = g["close"].values
    h = g["high"].values
    l = g["low"].values
    o = g["open"].values

    ret = np.diff(c) / c[:-1]
    pct_ret = (c[-1] - c[0]) / c[0] * 100
    vol_pct = np.std(ret) * 100
    signs = np.sign(ret)
    trend_con = np.mean(signs) if len(signs) > 0 else 0
    trend_str = abs(pct_ret) / (vol_pct + 1e-10)

    ranges = h - l
    avg_range = np.mean(ranges)
    tr = np.maximum(ranges[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    avg_atr = np.mean(tr) if len(tr) > 0 else avg_range
    rva = avg_range / (avg_atr + 1e-10)

    ac = np.corrcoef(ret[:-1], ret[1:])[0, 1] if len(ret) > 2 else 0
    if np.isnan(ac):
        ac = 0

    weeks.append({
        "week": wk,
        "n_candles": len(g),
        "weekly_return_pct": pct_ret,
        "volatility_pct": vol_pct,
        "trend_consistency": trend_con,
        "trend_strength": trend_str,
        "volatility": avg_range,
        "range_vs_atr": rva,
        "return_autocorr": ac,
    })

wdf = pd.DataFrame(weeks)
print(f"  {len(wdf)} weeks (>= {MIN_CANDLES} candles each)")

X = wdf[FINGERPRINT_COLS].values
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

pca = PCA(n_components=min(7, X.shape[1]))
Xp = pca.fit_transform(Xs)
print(f"  PCA variance explained: {pca.explained_variance_ratio_.cumsum()[-1]:.2%}")

km = KMeans(n_clusters=K, n_init=20, random_state=42)
wdf["cluster"] = km.fit_predict(Xp)

# Print cluster summaries
print(f"\nCluster summaries (K={K}):")
for c in range(K):
    sub = wdf[wdf["cluster"] == c]
    print(f"  C{c}: {len(sub)} weeks  "
          f"ret={sub['weekly_return_pct'].mean():.3f}%  "
          f"vol={sub['volatility_pct'].mean():.3f}%  "
          f"trend_con={sub['trend_consistency'].mean():.3f}  "
          f"trend_str={sub['trend_strength'].mean():.3f}")

# Check temporal spread
for c in range(K):
    sub = wdf[wdf["cluster"] == c]
    years = sub["week"].str[:4].unique()
    print(f"  C{c} spans years: {sorted(years)}")

# Save fingerprints
wdf.to_csv(P.data(f"regime_fingerprints_K{K}.csv"), index=False)
print(f"\nSaved: regime_fingerprints_K{K}.csv")

# Save selector JSON
selector = {
    "K": K,
    "fingerprint_cols": FINGERPRINT_COLS,
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_std": scaler.scale_.tolist(),
    "pre_pca_components": pca.components_.tolist(),
    "pre_pca_mean": pca.mean_.tolist(),
    "centroids_pre_pca": km.cluster_centers_.tolist(),
}
with open(P.data(f"regime_selector_K{K}.json"), "w") as f:
    json.dump(selector, f, indent=2)
print(f"Saved: regime_selector_K{K}.json")

# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
colors = ["#f5c518", "#3b82f6", "#ef4444", "#10b981"]
for c in range(K):
    ax = axes[c // 2][c % 2]
    sub = wdf[wdf["cluster"] == c]
    ax.scatter(sub["weekly_return_pct"], sub["volatility_pct"],
               c=colors[c], alpha=0.6, s=20)
    ax.set_title(f"C{c} ({len(sub)} weeks)", color=colors[c])
    ax.set_xlabel("Weekly Return %")
    ax.set_ylabel("Volatility %")
    ax.grid(True, alpha=0.2)
plt.suptitle(f"EURUSD Regime Clusters (K={K})", fontsize=14)
plt.tight_layout()
plt.savefig(P.data(f"regime_clusters_K{K}.png"), dpi=150)
print(f"Saved: regime_clusters_K{K}.png")
