"""
v75 step 02: side-by-side comparison of v1 (production K=5 K-means on 7
weekly-snapshot features) vs v2 (this experiment, K=5/6/7/8 on 16 rich
per-bar features).

For each candidate K:
  1. Print cluster assignments per bar over the most recent 60 days.
  2. Identify "shift days" — bars where v1 said C0/C4 (long-bias) but the
     subsequent 12 bars (1h forward) closed lower by >0.3%, or vice versa.
  3. Tabulate how many of those shift-bars v2 had ALREADY flagged with a
     down-bias cluster (C3-equivalent).
  4. Plot v1 vs v2 cluster timelines for visual inspection.
"""
from __future__ import annotations
import os, sys, json, time as _time
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ZIGZAG = "/home/jay/Desktop/new-model-zigzag"
EXP = os.path.join(ZIGZAG, "experiments/v75_regime_v2")
sys.path.insert(0, os.path.join(ZIGZAG, "model_pipeline"))
import paths as P
sys.path.insert(0, "/home/jay/Desktop/my-agents-and-website/commercial/server")
from decision_engine import regime as v1_regime

V1_SEL = json.load(open(P.data("regime_selector_K4.json")))
WINDOW_DAYS = 60   # last 60 days for the analysis window


def assign_v2(fp_df, K):
    """Assign v2 cluster IDs given the trained K-means selector."""
    sel_path = os.path.join(EXP, f"models/regime_selector_K{K}.json")
    sel = json.load(open(sel_path))
    feats = sel["features"]
    mean = np.array(sel["scaler_mean"])
    std  = np.array(sel["scaler_std"])
    centroids = np.array(sel["centroids"])
    X = fp_df[feats].values.astype(np.float64)
    Xs = (X - mean) / std
    # nearest-centroid assignment
    d2 = ((Xs[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return d2.argmin(axis=1)


def label_v2_clusters_by_means(K):
    """Map v2 cluster IDs to short names based on their feature means
    (read from cluster_K{k}_summary.csv)."""
    summ = pd.read_csv(os.path.join(EXP, f"reports/cluster_K{K}_summary.csv"))
    names = {}
    for _, row in summ.iterrows():
        cid = int(row["cluster"])
        r24 = row["ret_24h"]; r3 = row["ret_3h"]; sa = row["slope_accel"]
        vr = row["vol_ratio"]; pos = row["pos_24h"]; streak = row["streak"]
        if r24 > 0.005 and sa > 0.0001 and vr > 1.2:
            names[cid] = "BullBreakout"
        elif r24 < -0.005 and sa < -0.0001 and vr > 1.2:
            names[cid] = "BearBreakdown"
        elif r24 > 0.005:
            names[cid] = "UptrendMature"
        elif r24 < -0.005:
            names[cid] = "DowntrendMature"
        elif vr > 1.2:
            names[cid] = "VolBurst"
        else:
            names[cid] = "Quiet"
    return names


def assign_v1(swing):
    """Run the production v1 classifier per bar (using last 288 bars)."""
    n = len(swing); cid = np.full(n, -1, dtype=np.int32)
    for i in range(288, n):
        try: cid[i] = int(v1_regime.classify(swing.iloc[:i+1], V1_SEL))
        except Exception: pass
    return cid


def main():
    fp = pd.read_parquet(os.path.join(EXP, "data/fingerprints_rich.parquet"))
    fp = fp.sort_values("time").reset_index(drop=True)

    # Restrict to recent N days for the comparison window
    cutoff = fp["time"].max() - pd.Timedelta(days=WINDOW_DAYS)
    recent = fp[fp["time"] >= cutoff].reset_index(drop=True)
    print(f"Recent window: {recent['time'].iloc[0]} → {recent['time'].iloc[-1]}  ({len(recent):,} bars)")

    swing = pd.read_csv(P.data("swing_v5_xauusd.csv"), parse_dates=["time"])
    swing = swing.sort_values("time").reset_index(drop=True)
    swing_recent = swing[swing["time"] >= cutoff].reset_index(drop=True)
    # Keep some warmup before the window for v1 classify (288 bars)
    pad_start = swing[swing["time"] >= cutoff - pd.Timedelta(hours=48)].reset_index(drop=True)

    print(f"\nAssigning v1 (production K=5 K-means)...", flush=True)
    t0 = _time.time()
    v1_cids = assign_v1(pad_start)
    print(f"  done in {_time.time()-t0:.0f}s")
    v1_aligned = pd.DataFrame({"time": pad_start["time"].values, "v1_cid": v1_cids})
    v1_aligned = v1_aligned[v1_aligned["time"] >= cutoff].reset_index(drop=True)

    V1_NAMES = {0:"Uptrend", 1:"MeanRevert", 2:"TrendRange", 3:"Downtrend", 4:"HighVol"}

    forward_ret_1h = pd.Series(recent["close"].values).pct_change(12).shift(-12).values

    for K in [5, 6, 7, 8]:
        print(f"\n========== K = {K} ==========")
        v2_cids = assign_v2(recent, K)
        v2_names = label_v2_clusters_by_means(K)
        recent[f"v2_K{K}"] = v2_cids

        # Merge v1 + v2
        df = recent[["time", "close"]].copy()
        df["v2"] = v2_cids
        df["v2_name"] = [v2_names[c] for c in v2_cids]
        df = df.merge(v1_aligned, on="time", how="left")
        df["v1_name"] = df["v1_cid"].map(V1_NAMES).fillna("?")
        df["fwd_ret_1h"] = forward_ret_1h

        # Distribution
        print(f"  v2 cluster distribution: {pd.Series(v2_cids).value_counts().to_dict()}")
        print(f"  v2 cluster names: {v2_names}")

        # Shift-day diagnostic: bars where v1 said long-bias regime (C0/C4)
        # but next 1h closed >0.3% lower → v1 was wrong-side.
        long_bias_mask = df["v1_cid"].isin([0, 4])
        downward_shift = df["fwd_ret_1h"] < -0.003
        v1_blind_shift = df[long_bias_mask & downward_shift]
        print(f"  v1 long-bias (C0/C4) bars w/ next-1h drop >0.3%: {len(v1_blind_shift)}")
        if len(v1_blind_shift) > 0:
            v2_caught = v1_blind_shift[v1_blind_shift["v2_name"].isin(
                ["BearBreakdown", "DowntrendMature"])]
            print(f"    v2 (K={K}) flagged as Down/BearBreakdown: {len(v2_caught)} "
                  f"({100*len(v2_caught)/len(v1_blind_shift):.1f}%)")

        # Plot timeline for last 7 days
        last7 = df[df["time"] >= df["time"].max() - pd.Timedelta(days=7)].reset_index(drop=True)
        if len(last7) > 0:
            fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
            axes[0].plot(last7["time"], last7["close"], color="#888", lw=0.8)
            axes[0].set_title(f"XAU close (last 7d) — v1 vs v2 K={K}")
            axes[0].grid(alpha=0.3)

            # v1 cluster
            v1_palette = {0:"#10B981", 1:"#3B82F6", 2:"#8B5CF6", 3:"#EF4444", 4:"#F59E0B"}
            for c, color in v1_palette.items():
                m = last7["v1_cid"] == c
                if m.any():
                    axes[1].scatter(last7.loc[m,"time"], [c]*int(m.sum()), c=color, s=4, label=V1_NAMES[c])
            axes[1].set_ylabel("v1 cluster"); axes[1].legend(loc="upper right", fontsize=8)
            axes[1].grid(alpha=0.3)

            # v2 cluster
            cmap = plt.cm.tab10
            for c in range(K):
                m = last7["v2"] == c
                if m.any():
                    axes[2].scatter(last7.loc[m,"time"], [c]*int(m.sum()),
                                    c=[cmap(c)], s=4, label=f"C{c} {v2_names[c]}")
            axes[2].set_ylabel(f"v2 K={K} cluster"); axes[2].legend(loc="upper right", fontsize=8)
            axes[2].grid(alpha=0.3)

            plt.tight_layout()
            out_png = os.path.join(EXP, f"reports/timeline_K{K}_last7d.png")
            plt.savefig(out_png, dpi=100); plt.close()
            print(f"  plot: {out_png}")

    print("\nDone.")


if __name__ == "__main__":
    main()
