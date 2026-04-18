"""
EURUSD K=5 Rolling-Window Regime Selector
==========================================
Replaces the old K=4 weekly clustering with K=5 rolling-window KMeans.
Outputs the same files as the old 02_build_selector.py:
  - regime_fingerprints_K4.csv  (now K5, but same filename for compat)
  - regime_selector_K4.json     (now K5)
  - regime_clusters_K4.png

Cluster roles (auto-detected):
  C0: Uptrend        — buy-only rules
  C1: MeanRevert     — fade/reversal rules (both)
  C2: TrendRange     — breakout/momentum rules (both)
  C3: Downtrend      — sell-only rules
  C4: HighVol        — careful vol rules (both)
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import paths as P

K = 5
WINDOW = 288    # ~24h of M5 bars
STEP = 288      # non-overlapping
MIN_DATE = "2016-01-01"

CLUSTER_NAMES_CANONICAL = {0: "Uptrend", 1: "MeanRevert", 2: "TrendRange", 3: "Downtrend", 4: "HighVol"}

# ── Load data ──
DATA_PATH = P.data("swing_v5_eurusd.csv")
print(f"Loading {DATA_PATH}...")
df = pd.read_csv(DATA_PATH, parse_dates=["time"])
df = df.sort_values("time").reset_index(drop=True)
df = df[df["time"] >= MIN_DATE].reset_index(drop=True)
print(f"  {len(df):,} bars from {df['time'].iloc[0]} to {df['time'].iloc[-1]}")

closes = df["close"].values.astype(np.float64)
highs = df["high"].values.astype(np.float64)
lows = df["low"].values.astype(np.float64)
opens = df["open"].values.astype(np.float64)


def compute_fingerprint(c, h, l, o):
    n = len(c)
    if n < 10:
        return None
    returns = np.diff(c) / c[:-1]
    bar_ranges = (h - l) / c
    fp = {}
    fp["weekly_return_pct"] = float(returns.sum())
    fp["volatility_pct"] = float(returns.std())
    mean_ret = returns.mean()
    fp["trend_consistency"] = float(np.mean(np.sign(returns) == np.sign(mean_ret))) if abs(mean_ret) > 1e-12 else 0.5
    fp["trend_strength"] = float(returns.sum() / (returns.std() + 1e-9))
    fp["volatility"] = float(bar_ranges.mean())
    total_range = (h.max() - l.min()) / c.mean()
    fp["range_vs_atr"] = float(total_range / (bar_ranges.mean() + 1e-9))
    if len(returns) > 2:
        r1, r2 = returns[:-1], returns[1:]
        denom = r1.std() * r2.std()
        fp["return_autocorr"] = float(np.corrcoef(r1, r2)[0, 1]) if denom > 1e-12 else 0.0
    else:
        fp["return_autocorr"] = 0.0
    return fp


# ── Compute rolling fingerprints ──
print(f"Computing rolling fingerprints (window={WINDOW}, step={STEP})...")
fingerprints = []
for start in range(0, len(df) - WINDOW, STEP):
    end = start + WINDOW
    fp = compute_fingerprint(closes[start:end], highs[start:end], lows[start:end], opens[start:end])
    if fp is not None:
        fp["start_idx"] = start
        fp["end_idx"] = end
        fp["center_time"] = str(df["time"].iloc[(start + end) // 2])
        fingerprints.append(fp)

fp_df = pd.DataFrame(fingerprints)
feat_cols = ["weekly_return_pct", "volatility_pct", "trend_consistency",
             "trend_strength", "volatility", "range_vs_atr", "return_autocorr"]
X_raw = fp_df[feat_cols].values

# Scale, remove outliers
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
mask = np.all(np.abs(X_scaled) < 4, axis=1)
fp_df = fp_df[mask].reset_index(drop=True)
X_raw = fp_df[feat_cols].values
print(f"  {len(fp_df)} windows after outlier removal")

# Refit scaler on clean data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# PCA
pca = PCA(n_components=len(feat_cols))
X_pca = pca.fit_transform(X_scaled)
print(f"  PCA variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%")

# KMeans K=5
kmeans = KMeans(n_clusters=K, n_init=20, random_state=42)
raw_labels = kmeans.fit_predict(X_pca)

# ── Auto-detect cluster roles ──
cluster_stats = {}
for cid in range(K):
    cmask = raw_labels == cid
    cluster_stats[cid] = {
        "ret": fp_df.loc[cmask, "weekly_return_pct"].mean(),
        "vol": fp_df.loc[cmask, "volatility_pct"].mean(),
        "autocorr": fp_df.loc[cmask, "return_autocorr"].mean(),
        "n": int(cmask.sum()),
    }

sorted_by_ret = sorted(cluster_stats.items(), key=lambda x: x[1]["ret"])
downtrend_raw = sorted_by_ret[0][0]
uptrend_raw = sorted_by_ret[-1][0]
remaining = [c for c in range(K) if c not in (downtrend_raw, uptrend_raw)]
remaining_by_vol = sorted(remaining, key=lambda c: cluster_stats[c]["vol"], reverse=True)
highvol_raw = remaining_by_vol[0]
remaining2 = [c for c in remaining if c != highvol_raw]
remaining_by_autocorr = sorted(remaining2, key=lambda c: cluster_stats[c]["autocorr"])
meanrevert_raw = remaining_by_autocorr[0]
trendrange_raw = remaining_by_autocorr[1]

raw_to_canonical = {
    uptrend_raw: 0, meanrevert_raw: 1, trendrange_raw: 2,
    downtrend_raw: 3, highvol_raw: 4,
}

print(f"\nCluster mapping:")
for raw_cid, canon_cid in sorted(raw_to_canonical.items()):
    s = cluster_stats[raw_cid]
    print(f"  raw C{raw_cid} → C{canon_cid} {CLUSTER_NAMES_CANONICAL[canon_cid]:>12}  "
          f"n={s['n']:>4}  ret={s['ret']*100:+.2f}%  vol={s['vol']*100:.3f}%  autocorr={s['autocorr']:+.3f}")

# ── Remap centroids to canonical order ──
canonical_centroids = np.zeros_like(kmeans.cluster_centers_)
for raw_cid, canon_cid in raw_to_canonical.items():
    canonical_centroids[canon_cid] = kmeans.cluster_centers_[raw_cid]

# ── Assign bars to clusters ──
canonical_labels = np.array([raw_to_canonical[r] for r in raw_labels])
bar_clusters = np.full(len(df), -1, dtype=int)
for i in range(len(fp_df)):
    start = int(fp_df.iloc[i]["start_idx"])
    end = int(fp_df.iloc[i]["end_idx"])
    bar_clusters[start:end] = canonical_labels[i]

# Forward-fill
last = -1
for i in range(len(bar_clusters)):
    if bar_clusters[i] >= 0: last = bar_clusters[i]
    elif last >= 0: bar_clusters[i] = last

df["cluster"] = bar_clusters

# ── Save fingerprints ──
fp_df["cluster"] = canonical_labels
fp_df.to_csv(P.data("regime_fingerprints_K4.csv"), index=False)
print(f"\nSaved: regime_fingerprints_K4.csv")

# ── Save selector JSON (for MQL5 code generation) ──
selector = {
    "K": K,
    "window": WINDOW,
    "step": STEP,
    "n_feats": len(feat_cols),
    "feat_names": feat_cols,
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_std": scaler.scale_.tolist(),
    "pca_mean": pca.mean_.tolist(),
    "pca_components": pca.components_.tolist(),
    "centroids": canonical_centroids.tolist(),
    "cluster_names": CLUSTER_NAMES_CANONICAL,
    "tradeable": [1, 1, 1, 1, 1],
    "thresholds": [0.4, 0.4, 0.4, 0.4, 0.4],
}
with open(P.data("regime_selector_K4.json"), "w") as f:
    json.dump(selector, f, indent=2)
print(f"Saved: regime_selector_K4.json")

# ── Print cluster distribution ──
for cid in range(K):
    n = (bar_clusters == cid).sum()
    print(f"  C{cid} {CLUSTER_NAMES_CANONICAL[cid]:>12}: {n:>7,} bars ({100*n/len(df):.1f}%)")

# ── Plot ──
fig, ax = plt.subplots(figsize=(10, 8), facecolor="#080c12")
ax.set_facecolor("#0d1117")
colors = ["#f5c518", "#3b82f6", "#00E5FF", "#ef4444", "#10b981"]
for cid in range(K):
    cmask = canonical_labels == cid
    ax.scatter(X_pca[cmask, 0], X_pca[cmask, 1], c=colors[cid], s=10, alpha=0.5,
              label=f"C{cid} {CLUSTER_NAMES_CANONICAL[cid]}")
ax.legend(fontsize=10, loc="upper right")
ax.set_title("EURUSD K=5 Rolling-Window Clustering", color="#FFD700", fontsize=14)
ax.tick_params(colors="#5a7080")
for sp in ax.spines.values(): sp.set_edgecolor("#1e2a3a")
plt.savefig(P.data("regime_clusters_K4.png"), dpi=140, bbox_inches="tight", facecolor="#080c12")
print(f"Saved: regime_clusters_K4.png")
