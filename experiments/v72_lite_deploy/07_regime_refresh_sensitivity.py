"""
Regime refresh cadence sensitivity test.

Measures how much of the v7.2-lite holdout edge survives if we refresh the
active cluster more frequently (step=1 bar or step=12 bars) instead of the
deployed default (step=288 bars = ~1 day).

Approach: for every M5 bar, use the saved KMeans+PCA selector to compute the
'live cluster' at that moment using the trailing 288 bars. Then for each
holdout trade, check if its rule's home cluster still matches the live
cluster at that bar. Keep only matching trades, report WR/PF.

Caveat: this only measures the COST (trades that would have been filtered out
by a more frequent refresh). It cannot measure the GAIN from new trades that
would have fired in clusters the slow refresh missed — that needs retraining.
"""
from __future__ import annotations
import json, time as _time
from pathlib import Path
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "/home/jay/Desktop/new-model-zigzag/model_pipeline")
import paths as P

SELECTOR_JSON = P.data("regime_selector_K4.json")
SWING_CSV     = P.data("swing_v5_xauusd.csv")
TRADES_CSV    = P.data("v72l_trades_holdout.csv")  # 1,367 meta-filtered trades

WINDOW = 288
HOLDOUT_CUTOFF = pd.Timestamp("2024-12-12 00:00:00")


def compute_fp(c, h, l, o):
    returns = np.diff(c) / c[:-1]
    bar_ranges = (h - l) / c
    fp = np.zeros(7)
    fp[0] = returns.sum()
    fp[1] = returns.std()
    mean_ret = returns.mean()
    fp[2] = np.mean(np.sign(returns) == np.sign(mean_ret)) if abs(mean_ret) > 1e-12 else 0.5
    fp[3] = returns.sum() / (returns.std() + 1e-9)
    fp[4] = bar_ranges.mean()
    total_range = (h.max() - l.min()) / c.mean()
    fp[5] = total_range / (bar_ranges.mean() + 1e-9)
    if len(returns) > 2:
        r1, r2 = returns[:-1], returns[1:]
        denom = r1.std() * r2.std()
        fp[6] = np.corrcoef(r1, r2)[0, 1] if denom > 1e-12 else 0.0
    return fp


def classify(fp, sel):
    scaled = (fp - np.array(sel["scaler_mean"])) / np.array(sel["scaler_std"])
    centered = scaled - np.array(sel["pca_mean"])
    pca_proj = centered @ np.array(sel["pca_components"]).T
    centroids = np.array(sel["centroids"])
    dists = np.linalg.norm(centroids - pca_proj, axis=1)
    return int(np.argmin(dists))


def main():
    t0 = _time.time()
    with open(SELECTOR_JSON) as f:
        sel = json.load(f)

    print("Loading swing data ...")
    swing = pd.read_csv(SWING_CSV, parse_dates=["time"])
    swing = swing.sort_values("time").reset_index(drop=True)
    C = swing["close"].values.astype(np.float64)
    H = swing["high"].values.astype(np.float64)
    L = swing["low"].values.astype(np.float64)
    O = swing["open"].values.astype(np.float64)
    n = len(swing)
    print(f"  {n:,} bars  {swing['time'].iat[0]} → {swing['time'].iat[-1]}")

    # Classify every bar from WINDOW onward
    print(f"Classifying cluster at every bar (trailing {WINDOW}-bar fp) ...")
    live_cid = np.full(n, -1, dtype=int)
    t_cl = _time.time()
    for i in range(WINDOW, n):
        fp = compute_fp(C[i-WINDOW:i], H[i-WINDOW:i], L[i-WINDOW:i], O[i-WINDOW:i])
        live_cid[i] = classify(fp, sel)
    print(f"  done in {_time.time()-t_cl:.0f}s")

    time_to_idx = pd.Series(swing.index.values, index=swing["time"].values)

    print("\nLoading holdout trades ...")
    trades = pd.read_csv(TRADES_CSV, parse_dates=["time"])
    print(f"  {len(trades):,} trades in holdout set  (cid,rule) pairs")

    # The trade's 'cid' is the rule's home cluster — each trade only fires
    # if its rule belongs to the active cluster, so cid == rule_home_cluster.
    # Simulate different refresh cadences by decimating live_cid.
    print("\nSimulating refresh cadences:")
    print(f"{'step':>8} {'refresh_hz':>12} {'kept':>8} {'retain%':>8} "
          f"{'WR%':>6} {'PF':>6} {'totR':>8} {'avgR':>7} {'DD':>7}")

    # Build held cluster-id series for each step (fwd-fill)
    for step in [1, 3, 6, 12, 72, 288]:
        # At every bar, the 'held' cluster is live_cid at the most recent refresh tick
        held = np.full(n, -1, dtype=int)
        last = -1
        for i in range(n):
            if i % step == 0 and live_cid[i] >= 0:
                last = live_cid[i]
            held[i] = last

        # For each trade, compare its home cid to held cid at its bar
        keep = []
        for _, tr in trades.iterrows():
            t = tr["time"]
            if t not in time_to_idx.index: continue
            bi = int(time_to_idx[t])
            if held[bi] == int(tr["cid"]):
                keep.append(tr)
        kept_df = pd.DataFrame(keep)
        if len(kept_df) == 0:
            print(f"{step:>8} 0 trades survive")
            continue
        wr = (kept_df["pnl_R"] > 0).mean() * 100
        pf = (kept_df.loc[kept_df["pnl_R"]>0,"pnl_R"].sum()
              / max(-kept_df.loc[kept_df["pnl_R"]<0,"pnl_R"].sum(), 1e-9))
        eq = kept_df["pnl_R"].cumsum().values
        dd = float((np.maximum.accumulate(eq) - eq).max()) if len(eq) else 0.0
        retain = 100 * len(kept_df) / len(trades)
        hz = f"{24*60/(5*step):.1f}/day" if step < 288 else "1/day"
        print(f"{step:>8} {hz:>12} {len(kept_df):>8,} {retain:>7.1f}% "
              f"{wr:>6.1f} {pf:>6.2f} {kept_df['pnl_R'].sum():>+8.0f} "
              f"{kept_df['pnl_R'].mean():>+7.3f} {dd:>7.1f}")

    print(f"\nWallclock: {_time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
