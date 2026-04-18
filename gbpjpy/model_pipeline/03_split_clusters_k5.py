"""
Split labeled GBPJPY data into per-cluster CSVs using K=5 rolling-window assignments.
The new 02_build_selector_k5.py already saved bar-level cluster assignments.
We reload the v5 CSV, assign clusters via the same rolling-window approach,
and split into per-cluster CSVs.

K=5 clusters:
  C0: Uptrend     — buy-only
  C1: MeanRevert  — both (fade rules)
  C2: TrendRange  — both (breakout/momentum rules)
  C3: Downtrend   — sell-only
  C4: HighVol     — both (careful reversal rules)
"""
import pandas as pd
import numpy as np
import json
import paths as P

CLUSTER_NAMES = {0: "Uptrend", 1: "MeanRevert", 2: "TrendRange", 3: "Downtrend", 4: "HighVol"}
K = 5
WINDOW = 288
STEP = 288
MIN_DATE = "2016-01-01"

# Load labeled data (has entry_class: 0=BUY, 1=FLAT, 2=SELL)
df = pd.read_csv(P.data("labeled_gbpjpy.csv"), parse_dates=["time"])
print(f"Labeled data: {len(df):,} rows")

# Load fingerprints with cluster assignments
fp = pd.read_csv(P.data("regime_fingerprints_K4.csv"))
print(f"Fingerprints: {len(fp)} windows")

# Load the v5 raw CSV to get cluster assignments by time alignment
raw = pd.read_csv(P.data("swing_v5_gbpjpy.csv"), parse_dates=["time"])
raw = raw[raw["time"] >= MIN_DATE].reset_index(drop=True)

# Assign clusters to raw bars using rolling windows (same as selector)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

sel_json = json.load(open(P.data("regime_selector_K4.json")))
scaler_mean = np.array(sel_json["scaler_mean"])
scaler_std = np.array(sel_json["scaler_std"])
pca_mean = np.array(sel_json["pca_mean"])
pca_comp = np.array(sel_json["pca_components"])
centroids = np.array(sel_json["centroids"])

feat_names = sel_json["feat_names"]

def compute_fp(c, h, l, o):
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
    return fp

def classify_fp(fp_dict):
    vec = np.array([fp_dict[f] for f in feat_names])
    scaled = (vec - scaler_mean) / scaler_std
    rotated = (scaled - pca_mean) @ pca_comp.T
    dists = np.sum((rotated - centroids) ** 2, axis=1)
    return int(np.argmin(dists))

# Assign clusters to raw bars
closes = raw["close"].values.astype(np.float64)
highs = raw["high"].values.astype(np.float64)
lows = raw["low"].values.astype(np.float64)
opens = raw["open"].values.astype(np.float64)

bar_clusters = np.full(len(raw), -1, dtype=int)
for start in range(0, len(raw) - WINDOW, STEP):
    end = start + WINDOW
    fp = compute_fp(closes[start:end], highs[start:end], lows[start:end], opens[start:end])
    if fp is not None:
        cid = classify_fp(fp)
        bar_clusters[start:end] = cid

# Forward-fill
last = -1
for i in range(len(bar_clusters)):
    if bar_clusters[i] >= 0: last = bar_clusters[i]
    elif last >= 0: bar_clusters[i] = last

raw["cluster"] = bar_clusters

# Merge cluster assignments into labeled data by time
time_to_cluster = dict(zip(raw["time"], raw["cluster"]))
df["cluster"] = df["time"].map(time_to_cluster)
df = df.dropna(subset=["cluster"])
df["cluster"] = df["cluster"].astype(int)
df = df[df["cluster"] >= 0].reset_index(drop=True)
print(f"After cluster assignment: {len(df):,} rows")

# Split and apply direction filters
for cid in range(K):
    sub = df[df["cluster"] == cid].copy()
    if cid == 0:  # Uptrend: drop sells
        sub.loc[sub["entry_class"] == 2, "entry_class"] = 1
    elif cid == 3:  # Downtrend: drop buys
        sub.loc[sub["entry_class"] == 0, "entry_class"] = 1

    out = P.data(f"cluster_{cid}_{CLUSTER_NAMES[cid]}.csv")
    sub.to_csv(out, index=False)

    buys = (sub["entry_class"] == 0).sum()
    sells = (sub["entry_class"] == 2).sum()
    flats = (sub["entry_class"] == 1).sum()
    print(f"C{cid} {CLUSTER_NAMES[cid]:>12}: {len(sub):>7,} rows  BUY={buys:>6,}  FLAT={flats:>6,}  SELL={sells:>6,}")
