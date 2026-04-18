"""
Split labeled GBPJPY data into per-cluster CSVs.
Auto-detects cluster roles from fingerprint statistics:
  - Uptrend: highest mean weekly return
  - Downtrend: lowest mean weekly return
  - HighVol: highest volatility among remaining
  - Ranging: the remaining cluster

Uptrend: drop sells (SELL->FLAT) — only buy signals
Downtrend: drop buys (BUY->FLAT) — only sell signals
Ranging/HighVol: keep all directions
"""
import pandas as pd
import numpy as np
import paths as P

df = pd.read_csv(P.data("labeled_gbpjpy.csv"), parse_dates=["time"])
fp = pd.read_csv(P.data("regime_fingerprints_K4.csv"))

# Auto-detect cluster roles
cluster_stats = fp.groupby("cluster").agg({
    "weekly_return_pct": "mean",
    "volatility_pct": "mean",
}).reset_index()

# Uptrend = highest return, Downtrend = lowest return
uptrend_cid = int(cluster_stats.loc[cluster_stats["weekly_return_pct"].idxmax(), "cluster"])
downtrend_cid = int(cluster_stats.loc[cluster_stats["weekly_return_pct"].idxmin(), "cluster"])

# HighVol = highest volatility among remaining two
remaining = cluster_stats[~cluster_stats["cluster"].isin([uptrend_cid, downtrend_cid])]
highvol_cid = int(remaining.loc[remaining["volatility_pct"].idxmax(), "cluster"])

# Ranging = the remaining one
ranging_cid = int(remaining.loc[remaining["volatility_pct"].idxmin(), "cluster"])

CLUSTER_MAP = {
    uptrend_cid: "Uptrend",
    ranging_cid: "Ranging",
    downtrend_cid: "Downtrend",
    highvol_cid: "HighVol",
}

print("Auto-detected cluster mapping:")
for cid, name in sorted(CLUSTER_MAP.items()):
    row = cluster_stats[cluster_stats["cluster"] == cid].iloc[0]
    n_weeks = len(fp[fp["cluster"] == cid])
    print(f"  C{cid} -> {name:>10}  ({n_weeks} weeks, ret={row['weekly_return_pct']:.3f}%, vol={row['volatility_pct']:.3f}%)")

# Remap clusters to canonical IDs: 0=Uptrend, 1=Ranging, 2=Downtrend, 3=HighVol
CANONICAL = {"Uptrend": 0, "Ranging": 1, "Downtrend": 2, "HighVol": 3}
remap = {cid: CANONICAL[name] for cid, name in CLUSTER_MAP.items()}

week_to_cluster = {}
for _, row in fp.iterrows():
    old_cid = int(row["cluster"])
    week_to_cluster[row["week"]] = remap[old_cid]

# Also remap in fingerprints file for downstream consistency
fp["cluster"] = fp["cluster"].map(remap)
fp.to_csv(P.data("regime_fingerprints_K4.csv"), index=False)
print("\nRemapped regime_fingerprints_K4.csv to canonical IDs")

# Also update regime_selector_K4.json centroids order
import json
with open(P.data("regime_selector_K4.json")) as f:
    sel = json.load(f)

old_centroids = sel["centroids_pre_pca"]
new_centroids = [None] * 4
for old_cid, new_cid in remap.items():
    new_centroids[new_cid] = old_centroids[old_cid]
sel["centroids_pre_pca"] = new_centroids

with open(P.data("regime_selector_K4.json"), "w") as f:
    json.dump(sel, f, indent=2)
print("Remapped regime_selector_K4.json centroids")

# Now split
CLUSTER_NAMES = {0: "Uptrend", 1: "Ranging", 2: "Downtrend", 3: "HighVol"}

df["week"] = df["time"].dt.isocalendar().year.astype(str) + "-W" + df["time"].dt.isocalendar().week.astype(str).str.zfill(2)
df["cluster"] = df["week"].map(week_to_cluster)
df = df.dropna(subset=["cluster"])
df["cluster"] = df["cluster"].astype(int)

for cid in range(4):
    sub = df[df["cluster"] == cid].copy()
    if cid == 0:  # Uptrend: drop sells
        sub.loc[sub["entry_class"] == 2, "entry_class"] = 1
    elif cid == 2:  # Downtrend: drop buys
        sub.loc[sub["entry_class"] == 0, "entry_class"] = 1

    out = P.data(f"cluster_{cid}_{CLUSTER_NAMES[cid]}.csv")
    sub.to_csv(out, index=False)

    buys = (sub["entry_class"] == 0).sum()
    sells = (sub["entry_class"] == 2).sum()
    flats = (sub["entry_class"] == 1).sum()
    print(f"C{cid} {CLUSTER_NAMES[cid]:>10}: {len(sub):>7,} rows  BUY={buys:>6,}  FLAT={flats:>6,}  SELL={sells:>6,}")
