"""v9.3 — Add Quantum Flow as an 8th regime fingerprint feature.

Modified version of model_pipeline/02_build_selector_k5.py that includes
the volume-weighted Heikin-Ashi momentum (flow) as a fingerprint feature
for the K=5 cluster assignment. Hypothesis: regime classifier becomes
more stable on ambiguous days because flow adds volume-info that the
existing 7 features lack.
"""
import sys, os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import json
sys.path.insert(0, "/home/jay/Desktop/new-model-zigzag/model_pipeline")
from importlib.machinery import SourceFileLoader
qf = SourceFileLoader("qf01",
    "/home/jay/Desktop/new-model-zigzag/experiments/v89_quantum_flow_tiebreaker/01_port_and_test.py"
).load_module()

ROOT = "/home/jay/Desktop/new-model-zigzag"
DATA = ROOT + "/data"
WINDOW = 288
STEP = 288
K = 5
MIN_DATE = "2020-01-01"


print(f"Loading swing data...", flush=True)
df = pd.read_csv(DATA + "/swing_v5_xauusd.csv", parse_dates=["time"])
df = df.rename(columns={"tick_volume": "volume"}) if "tick_volume" in df.columns else df
df = df.sort_values("time").reset_index(drop=True)
df = df[df["time"] >= MIN_DATE].reset_index(drop=True)
print(f"  {len(df):,} bars from {df['time'].iloc[0]} to {df['time'].iloc[-1]}", flush=True)

print("Computing flow_4h on full bars (one-time)...", flush=True)
flow_4h = qf.quantum_flow_mtf(df[["time","open","high","low","close","volume"]])
print(f"  flow_4h med={float(np.nanmedian(flow_4h.values)):.2f}", flush=True)

closes = df["close"].values.astype(np.float64)
highs = df["high"].values.astype(np.float64)
lows = df["low"].values.astype(np.float64)
opens = df["open"].values.astype(np.float64)
flow4h_arr = flow_4h.values.astype(np.float64)


def compute_fingerprint(c, h, l, o, flow):
    n = len(c)
    if n < 10: return None
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
    # NEW: flow_4h aggregate (mean over the window)
    flow_clean = flow[~np.isnan(flow)]
    fp["flow_4h_mean"] = float(flow_clean.mean()) if len(flow_clean) else 0.0
    return fp


print(f"Computing rolling fingerprints (window={WINDOW}, step={STEP})...", flush=True)
fingerprints = []
for start in range(0, len(df) - WINDOW, STEP):
    end = start + WINDOW
    fp = compute_fingerprint(closes[start:end], highs[start:end], lows[start:end],
                              opens[start:end], flow4h_arr[start:end])
    if fp is not None:
        fp["start_idx"] = start; fp["end_idx"] = end
        fp["center_time"] = str(df["time"].iloc[(start + end) // 2])
        fingerprints.append(fp)

fp_df = pd.DataFrame(fingerprints)
feat_cols = ["weekly_return_pct", "volatility_pct", "trend_consistency",
              "trend_strength", "volatility", "range_vs_atr", "return_autocorr",
              "flow_4h_mean"]   # ← 8th feature
X_raw = fp_df[feat_cols].values
print(f"  {len(fp_df)} fingerprints  feats={len(feat_cols)}", flush=True)

scaler = StandardScaler(); X_scaled = scaler.fit_transform(X_raw)
mask = np.all(np.abs(X_scaled) < 4, axis=1)
fp_df = fp_df[mask].reset_index(drop=True)
X_raw = fp_df[feat_cols].values
print(f"  {len(fp_df)} windows after outlier removal", flush=True)

scaler = StandardScaler(); X_scaled = scaler.fit_transform(X_raw)
pca = PCA(n_components=len(feat_cols))
X_pca = pca.fit_transform(X_scaled)
print(f"  PCA variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%", flush=True)

kmeans = KMeans(n_clusters=K, n_init=20, random_state=42)
raw_labels = kmeans.fit_predict(X_pca)

# Auto-detect cluster roles (same as original)
cluster_stats = {}
for cid in range(K):
    cmask = raw_labels == cid
    cluster_stats[cid] = {
        "ret": fp_df.loc[cmask, "weekly_return_pct"].mean(),
        "vol": fp_df.loc[cmask, "volatility_pct"].mean(),
        "autocorr": fp_df.loc[cmask, "return_autocorr"].mean(),
        "flow": fp_df.loc[cmask, "flow_4h_mean"].mean(),
        "n": int(cmask.sum()),
    }
sorted_by_ret = sorted(cluster_stats.items(), key=lambda x: x[1]["ret"])
downtrend_raw = sorted_by_ret[0][0]; uptrend_raw = sorted_by_ret[-1][0]
remaining = [c for c in range(K) if c not in (downtrend_raw, uptrend_raw)]
remaining_by_vol = sorted(remaining, key=lambda c: cluster_stats[c]["vol"], reverse=True)
highvol_raw = remaining_by_vol[0]
remaining2 = [c for c in remaining if c != highvol_raw]
remaining_by_autocorr = sorted(remaining2, key=lambda c: cluster_stats[c]["autocorr"])
meanrev_raw = remaining_by_autocorr[0]
trendrange_raw = [c for c in remaining2 if c != meanrev_raw][0]

label_map = {downtrend_raw: 3, uptrend_raw: 0, highvol_raw: 4,
              meanrev_raw: 1, trendrange_raw: 2}
new_labels = np.array([label_map[r] for r in raw_labels])

# Reorder centroids
new_centroids = np.empty_like(kmeans.cluster_centers_)
for raw, new in label_map.items():
    new_centroids[new] = kmeans.cluster_centers_[raw]

cluster_names = {0:"Uptrend", 1:"MeanRevert", 2:"TrendRange", 3:"Downtrend", 4:"HighVol"}
print("\nCluster mapping:")
for raw, new in label_map.items():
    s = cluster_stats[raw]
    print(f"  raw C{raw} → C{new} {cluster_names[new]:>11s}  n={s['n']:>4}  "
          f"ret={s['ret']:+.2%}  vol={s['vol']:.3%}  autocorr={s['autocorr']:+.3f}  "
          f"flow={s['flow']:+.2f}")

# Save selector JSON
out = {
    "K": K, "window": WINDOW, "step": STEP,
    "n_feats": len(feat_cols), "feat_names": feat_cols,
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_std": scaler.scale_.tolist(),
    "pca_mean": pca.mean_.tolist(),
    "pca_components": pca.components_.tolist(),
    "centroids": new_centroids.tolist(),
    "cluster_names": {str(k): v for k, v in cluster_names.items()},
    "tradeable": {"0": True, "1": True, "2": True, "3": True, "4": True},
    "thresholds": {"weekly_return_pct": 0.0},
}
sel_path = DATA + "/regime_selector_K4.json"
with open(sel_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {sel_path}")

# Save fingerprints csv too (for debugging)
fp_df["new_label"] = new_labels
fp_df.to_csv(DATA + "/regime_fingerprints_K4.csv", index=False)
print(f"Saved: regime_fingerprints_K4.csv")

print(f"\nFingerprint feature added: flow_4h_mean")
print(f"PCA components shape: {len(pca.components_)} × {len(pca.components_[0])}")

# Now check stability on the LIVE WEEK ambiguous days
print("\n" + "="*60)
print("STABILITY CHECK on recent ambiguous days:")
print("="*60)
import datetime as _dt
test_dates = [_dt.date(2026, 4, 27), _dt.date(2026, 4, 28),
               _dt.date(2026, 4, 29), _dt.date(2026, 4, 30),
               _dt.date(2026, 5, 1)]
for d in test_dates:
    # Find first bar of day d
    matches = df[df.time.dt.date == d]
    if not len(matches): continue
    end_idx = matches.index[0]
    if end_idx < WINDOW: continue
    start_idx = end_idx - WINDOW
    fp = compute_fingerprint(closes[start_idx:end_idx], highs[start_idx:end_idx],
                              lows[start_idx:end_idx], opens[start_idx:end_idx],
                              flow4h_arr[start_idx:end_idx])
    fp_vec = np.array([fp[c] for c in feat_cols])
    fp_scaled = (fp_vec - scaler.mean_) / scaler.scale_
    fp_pca = (fp_scaled - pca.mean_) @ pca.components_.T
    dists = np.linalg.norm(new_centroids - fp_pca, axis=1)
    order = np.argsort(dists)
    closest = cluster_names[order[0]]; second = cluster_names[order[1]]
    gap = dists[order[1]] - dists[order[0]]
    print(f"  {d}  closest={closest:>11s} (d={dists[order[0]]:.2f})  "
          f"2nd={second:>11s} (d={dists[order[1]]:.2f})  "
          f"gap={gap:.2f}  {'✓ stable' if gap > 1.0 else '⚠ ambiguous'}")
