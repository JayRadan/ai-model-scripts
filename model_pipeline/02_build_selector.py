"""
REGIME SELECTOR BUILDER
=======================
Step 1: Load your 1M row M5 data
Step 2: Split into weekly windows
Step 3: Compute one fingerprint per week
Step 4: KMeans cluster the fingerprints
Step 5: Visualize — scatter + cluster profiles
Step 6: Test any week against the clusters
Step 7: Save centroids for MQL5/ONNX later

Run: python build_selector.py --file your_data.csv --k 6
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import argparse
import json

# ── Config ────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "f01_CPR","f02_WickAsym","f03_BEF","f04_TCS","f05_SPI",
    "f06_LRSlope","f07_RECR","f08_SCM","f09_HLER","f10_EP",
    "f11_KE","f12_MCS","f13_Work","f14_EDR","f15_AI",
    "f16_PPShigh","f16_PPSlow","f17_SCR","f18_RVD","f19_WBER","f20_NCDE"
]

COLORS = ["#FFD700","#00E5FF","#FF6B6B","#69FF94","#FF9F43","#A29BFE","#FD79A8","#FDCB6E"]

# ── Load data ─────────────────────────────────────────────────────────────────
def load_data(filepath):
    print(f"Loading {filepath}...")
    df = pd.read_csv(filepath, parse_dates=["time"], encoding="utf-16")
    df = df.sort_values("time").reset_index(drop=True)
    print(f"  Rows: {len(df):,}")
    print(f"  Range: {df['time'].min()} → {df['time'].max()}")

    # Keep only feature cols that exist in file
    available = [c for c in FEATURE_COLS if c in df.columns]
    print(f"  Features found: {len(available)}/{len(FEATURE_COLS)}")
    return df, available

# ── Compute weekly fingerprint ────────────────────────────────────────────────
def compute_fingerprint(week_df, feat_cols):
    """
    One week of M5 candles → one vector describing its character.
    We compute per-feature: mean, std
    Plus price-based: trend_strength, volatility, choppiness
    """
    feats = {}

    # Per-feature stats
    for col in feat_cols:
        vals = week_df[col].dropna().values
        if len(vals) == 0:
            feats[f"{col}_mean"] = 0.0
            feats[f"{col}_std"]  = 0.0
        else:
            feats[f"{col}_mean"] = float(np.mean(vals))
            feats[f"{col}_std"]  = float(np.std(vals))

    # Price-based features — ALL scale-invariant (% returns, not raw prices)
    closes = week_df["close"].values
    highs  = week_df["high"].values
    lows   = week_df["low"].values

    feats["candle_count"] = len(week_df)

    if len(closes) > 1:
        returns = np.diff(closes) / closes[:-1]              # bar-to-bar % returns
        bar_ranges = (highs - lows) / closes                 # bar range as % of close

        # Directional features (scale-invariant)
        feats["weekly_return_pct"]  = float(returns.sum())
        feats["volatility_pct"]     = float(returns.std())
        mean_ret = returns.mean()
        feats["trend_consistency"]  = float(np.mean(np.sign(returns) == np.sign(mean_ret))) if mean_ret != 0 else 0.5

        # Trend strength: net move relative to volatility (Sharpe-like)
        feats["trend_strength"]     = float(returns.sum() / (returns.std() + 1e-9))

        # Volatility / range features
        feats["volatility"]         = float(bar_ranges.mean())
        feats["range_vs_atr"]       = float((highs.max() - lows.min()) / closes.mean() / (bar_ranges.mean() + 1e-9))

        # Autocorrelation of returns — positive = trending, negative = mean-reverting
        if len(returns) > 2:
            r_shift = returns[:-1]
            r_next  = returns[1:]
            denom = r_shift.std() * r_next.std()
            feats["return_autocorr"] = float(np.mean((r_shift - r_shift.mean()) * (r_next - r_next.mean())) / (denom + 1e-9))
        else:
            feats["return_autocorr"] = 0.0
    else:
        feats["weekly_return_pct"] = 0.0
        feats["volatility_pct"]    = 0.0
        feats["trend_consistency"] = 0.5
        feats["trend_strength"]    = 0.0
        feats["volatility"]        = 0.0
        feats["range_vs_atr"]      = 0.0
        feats["return_autocorr"]   = 0.0

    # Label distribution (useful to check later)
    if "label" in week_df.columns:
        labels = week_df["label"].values
        feats["label_buy_pct"]  = float(np.mean(labels == 1))
        feats["label_sell_pct"] = float(np.mean(labels == -1))

    return feats

# ── Build fingerprint matrix ──────────────────────────────────────────────────
def build_fingerprints(df, feat_cols):
    print("Building weekly fingerprints...")
    df["week"] = df["time"].dt.to_period("W")
    weeks = df.groupby("week")

    records = []
    for week_label, group in weeks:
        if len(group) < 50:   # skip tiny weeks (gaps, holidays)
            continue
        fp = compute_fingerprint(group, feat_cols)
        fp["week"] = str(week_label)
        fp["week_start"] = str(group["time"].min())
        fp["week_end"]   = str(group["time"].max())
        fp["n_candles"]  = len(group)
        records.append(fp)

    fp_df = pd.DataFrame(records)
    print(f"  Total weeks (raw): {len(fp_df)}")

    # Remove outlier weeks (less than 200 candles or extreme values)
    fp_df = fp_df[fp_df["n_candles"] >= 200].reset_index(drop=True)
    print(f"  Total weeks (after outlier filter): {len(fp_df)}")
    return fp_df

# Scale-invariant fingerprint features — price structure only, no raw f-feature stats
# (the f01-f20 _mean/_std columns drift heavily with time and cause era-clustering)
FINGERPRINT_COLS = [
    "weekly_return_pct",
    "volatility_pct",
    "trend_consistency",
    "trend_strength",
    "volatility",
    "range_vs_atr",
    "return_autocorr",
]

# ── Cluster ───────────────────────────────────────────────────────────────────
def cluster_weeks(fp_df, K, feature_prefix_exclude=["week","week_start","week_end","n_candles","label"]):
    # Use the curated scale-invariant feature set (not all numeric columns)
    num_cols = [c for c in FINGERPRINT_COLS if c in fp_df.columns]
    print(f"  Clustering on {len(num_cols)} scale-invariant features: {num_cols}")

    X = fp_df[num_cols].fillna(0).values

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # BEFORE KMeans, reduce dims (cap at min(features, 10))
    n_comp = min(10, X_scaled.shape[1])
    pre_pca = PCA(n_components=n_comp, random_state=42)
    X_reduced = pre_pca.fit_transform(X_scaled)
    print(f"  Pre-cluster PCA: {X_scaled.shape[1]} → {X_reduced.shape[1]} dims "
          f"({pre_pca.explained_variance_ratio_.sum()*100:.1f}% variance)")

    # KMeans on reduced space
    print(f"Running KMeans K={K}...")
    km = KMeans(n_clusters=K, n_init=20, random_state=42)
    labels = km.fit_predict(X_reduced)
    fp_df["cluster"] = labels

    # PCA for visualization
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_scaled)
    fp_df["pca_x"] = X_2d[:, 0]
    fp_df["pca_y"] = X_2d[:, 1]

    print(f"  Variance explained by PC1+PC2: {pca.explained_variance_ratio_.sum()*100:.1f}%")

    return fp_df, km, scaler, pca, pre_pca, num_cols, X_scaled

# ── Visualize ─────────────────────────────────────────────────────────────────
def visualize(fp_df, K, num_cols, output_prefix="regime"):
    fig = plt.figure(figsize=(18, 12), facecolor="#080c12")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    ax_scatter = fig.add_subplot(gs[0, :])  # full width top
    ax_profile = fig.add_subplot(gs[1, 0])
    ax_sizes   = fig.add_subplot(gs[1, 1])

    # ── Scatter ──────────────────────────────────────────────────────────────
    ax_scatter.set_facecolor("#0d1117")
    ax_scatter.set_title("WEEKLY REGIME CLUSTERS (PCA 2D)", color="#FFD700",
                          fontsize=14, fontweight="bold", pad=12, fontfamily="monospace")

    for ci in range(K):
        mask = fp_df["cluster"] == ci
        ax_scatter.scatter(
            fp_df.loc[mask, "pca_x"],
            fp_df.loc[mask, "pca_y"],
            c=COLORS[ci % len(COLORS)], s=60, alpha=0.75,
            label=f"Cluster {ci+1} ({mask.sum()} weeks)", edgecolors="none"
        )

    ax_scatter.legend(facecolor="#0d1117", edgecolor="#333", labelcolor="white",
                      fontsize=9, loc="upper right")
    ax_scatter.set_xlabel("PC1", color="#5a7080")
    ax_scatter.set_ylabel("PC2", color="#5a7080")
    ax_scatter.tick_params(colors="#5a7080")
    for spine in ax_scatter.spines.values():
        spine.set_edgecolor("#1e2a3a")

    # ── Cluster sizes ─────────────────────────────────────────────────────────
    ax_sizes.set_facecolor("#0d1117")
    ax_sizes.set_title("WEEKS PER CLUSTER", color="#FFD700", fontsize=11,
                        fontfamily="monospace", pad=8)
    sizes = [int((fp_df["cluster"] == ci).sum()) for ci in range(K)]
    bars = ax_sizes.bar(range(K), sizes,
                        color=[COLORS[i % len(COLORS)] for i in range(K)], alpha=0.8)
    for bar, val in zip(bars, sizes):
        ax_sizes.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                      str(val), ha="center", va="bottom", color="white", fontsize=9)
    ax_sizes.set_xticks(range(K))
    ax_sizes.set_xticklabels([f"C{i+1}" for i in range(K)], color="#8899aa")
    ax_sizes.tick_params(colors="#5a7080")
    ax_sizes.set_facecolor("#0d1117")
    for spine in ax_sizes.spines.values():
        spine.set_edgecolor("#1e2a3a")

    # ── Feature profile heatmap ───────────────────────────────────────────────
    ax_profile.set_facecolor("#0d1117")
    ax_profile.set_title("CLUSTER FEATURE PROFILES (mean per cluster)", color="#FFD700",
                          fontsize=11, fontfamily="monospace", pad=8)

    # Pick top 12 most discriminative features
    mean_cols = [c for c in num_cols if c.endswith("_mean") or c in ["trend_strength","volatility"]][:12]
    profile_data = np.array([
        fp_df[fp_df["cluster"] == ci][mean_cols].mean().values
        for ci in range(K)
    ])
    im = ax_profile.imshow(profile_data, aspect="auto", cmap="RdYlGn", vmin=-1, vmax=1)
    ax_profile.set_xticks(range(len(mean_cols)))
    ax_profile.set_xticklabels([c.replace("_mean","").replace("f0","").replace("f1","") for c in mean_cols],
                                rotation=45, ha="right", color="#8899aa", fontsize=8)
    ax_profile.set_yticks(range(K))
    ax_profile.set_yticklabels([f"Cluster {i+1}" for i in range(K)], color="#8899aa")
    plt.colorbar(im, ax=ax_profile, label="Mean value")

    plt.suptitle("EDGEPREDICTOR — REGIME SELECTOR", color="#FFD700",
                 fontsize=16, fontweight="bold", fontfamily="monospace", y=1.01)

    import paths as P
    out = P.data(f"{output_prefix}_clusters_K{K}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#080c12")
    print(f"  Saved: {out}")
    plt.show()

# ── Test a single week against clusters ───────────────────────────────────────
def test_week(df, feat_cols, km, scaler, pca, pre_pca, num_cols,
              week_start="2023-01-02", week_end="2023-01-06"):
    """
    Given a date range, fingerprint that week and find its cluster.
    This is exactly what MQL5 will do at runtime.
    """
    mask = (df["time"] >= week_start) & (df["time"] <= week_end)
    week_df = df[mask]
    if len(week_df) == 0:
        print(f"  No data for {week_start} → {week_end}")
        return

    fp = compute_fingerprint(week_df, feat_cols)
    row = pd.DataFrame([fp]).reindex(columns=num_cols).fillna(0)
    row_scaled = scaler.transform(row)
    row_reduced = pre_pca.transform(row_scaled)
    cluster = km.predict(row_reduced)[0]
    coords = pca.transform(row_scaled)[0]

    print(f"\n{'─'*50}")
    print(f"TEST WEEK: {week_start} → {week_end}")
    print(f"  Candles: {len(week_df)}")
    print(f"  → CLUSTER {cluster + 1}  (color: {COLORS[cluster % len(COLORS)]})")
    print(f"  PCA position: ({coords[0]:.3f}, {coords[1]:.3f})")
    print(f"{'─'*50}\n")
    return cluster

# ── Save for MQL5 ─────────────────────────────────────────────────────────────
def save_centroids(km, scaler, pca, pre_pca, num_cols, K, output_prefix="regime"):
    """Save everything needed for runtime deployment.

    Runtime inference pipeline:
      1. compute fingerprint → num_cols vector
      2. standardize: (x - scaler_mean) / scaler_std
      3. rotate: pre_pca.transform(scaled)
      4. nearest centroid in pre_pca space = cluster id
    """
    export = {
        "K": K,
        "feature_columns": num_cols,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_std": scaler.scale_.tolist(),
        "pre_pca_components": pre_pca.components_.tolist(),
        "pre_pca_mean": pre_pca.mean_.tolist(),
        "centroids_pre_pca": km.cluster_centers_.tolist(),
        # 2D PCA for visualization only
        "viz_pca_components": pca.components_.tolist(),
        "viz_pca_mean": pca.mean_.tolist(),
    }
    import paths as P
    out = P.data(f"{output_prefix}_selector_K{K}.json")
    with open(out, "w") as f:
        json.dump(export, f, indent=2)
    print(f"Saved selector config: {out}")
    print("→ Load this JSON in Python at runtime to classify any new week")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    import os
    import paths as P
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="swing_v4.csv",
                        help="CSV file (relative to data/ or absolute)")
    parser.add_argument("--k", type=int, default=4, help="Number of clusters")
    parser.add_argument("--test_start", default=None, help="Test week start YYYY-MM-DD")
    parser.add_argument("--test_end",   default=None, help="Test week end   YYYY-MM-DD")
    args = parser.parse_args()

    fp = args.file if os.path.isabs(args.file) else P.data(args.file)
    df, feat_cols = load_data(fp)
    fp_df = build_fingerprints(df, feat_cols)
    fp_df, km, scaler, pca, pre_pca, num_cols, X_scaled = cluster_weeks(fp_df, args.k)

    # Print cluster summary
    print("\nCLUSTER SUMMARY:")
    for ci in range(args.k):
        mask = fp_df["cluster"] == ci
        weeks_in = fp_df[mask]["week_start"].tolist()
        print(f"  Cluster {ci+1}: {mask.sum()} weeks | first: {weeks_in[0] if weeks_in else 'n/a'}")

    visualize(fp_df, args.k, num_cols)
    save_centroids(km, scaler, pca, pre_pca, num_cols, args.k)

    # Optional: test a specific week
    if args.test_start and args.test_end:
        test_week(df, feat_cols, km, scaler, pca, pre_pca, num_cols, args.test_start, args.test_end)
    else:
        # Auto-test the last week in dataset
        last = df["time"].max()
        week_end = str(last.date())
        week_start = str((last - pd.Timedelta(days=7)).date())
        test_week(df, feat_cols, km, scaler, pca, pre_pca, num_cols, week_start, week_end)

    # Save fingerprint table
    import paths as P
    fp_path = P.data(f"regime_fingerprints_K{args.k}.csv")
    fp_df.to_csv(fp_path, index=False)
    print(f"Saved fingerprints: {fp_path}")

if __name__ == "__main__":
    main()