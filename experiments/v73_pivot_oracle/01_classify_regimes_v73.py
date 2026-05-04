"""
v7.3 Pivot Oracle — per-bar regime classification.

Uses Oracle's existing regime selector (data/regime_selector_K4.json — actually
K=5 by inspection; the filename is legacy) byte-for-byte: same fingerprint
formula as model_pipeline/02_build_selector_k5.py:compute_fingerprint, same
scaler, same PCA, same centroids.

For each bar i, computes the rolling fingerprint over the trailing WINDOW
bars, scales + PCA-projects, then assigns the nearest centroid.

Output: experiments/v73_pivot_oracle/data/cluster_per_bar_v73.csv
Schema: time, bar_idx, cid
"""
from __future__ import annotations
import os, json, sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/jay/Desktop/new-model-zigzag/model_pipeline")
import paths as P

OUT_DIR = "/home/jay/Desktop/new-model-zigzag/experiments/v73_pivot_oracle/data"
os.makedirs(OUT_DIR, exist_ok=True)

SELECTOR_PATH = "/home/jay/Desktop/new-model-zigzag/data/regime_selector_K4.json"


def compute_fingerprint(c, h, l, o):
    """Byte-for-byte copy of model_pipeline/02_build_selector_k5.py."""
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


def main():
    print(f"Loading regime selector: {SELECTOR_PATH}", flush=True)
    sel = json.load(open(SELECTOR_PATH))
    K = sel["K"]
    WINDOW = sel["window"]
    STEP = sel["step"]
    feat_names = sel["feat_names"]
    sc_mean = np.array(sel["scaler_mean"])
    sc_std = np.array(sel["scaler_std"])
    pca_mean = np.array(sel["pca_mean"])
    pca_comp = np.array(sel["pca_components"])
    centroids = np.array(sel["centroids"])
    print(f"  K={K}  window={WINDOW}  step={STEP}  feats={feat_names}", flush=True)

    print("Loading swing CSV...", flush=True)
    swing = pd.read_csv(P.data("swing_v5_xauusd.csv"), parse_dates=["time"])
    swing = swing.sort_values("time").reset_index(drop=True)
    n = len(swing)
    O = swing["open"].values; H = swing["high"].values; L = swing["low"].values; C = swing["close"].values

    print(f"Classifying {n:,} bars (re-classify every {STEP} bars)...", flush=True)
    cid_per_bar = np.full(n, -1, dtype=np.int32)
    last_cid = -1
    for i in range(WINDOW, n):
        if i % STEP != 0 and last_cid != -1:
            cid_per_bar[i] = last_cid; continue
        c_w = C[i - WINDOW:i]; h_w = H[i - WINDOW:i]; l_w = L[i - WINDOW:i]; o_w = O[i - WINDOW:i]
        fp = compute_fingerprint(c_w, h_w, l_w, o_w)
        if fp is None:
            cid_per_bar[i] = last_cid; continue
        x = np.array([fp[k] for k in feat_names])
        xs = (x - sc_mean) / np.where(sc_std > 0, sc_std, 1.0)
        xp = pca_comp @ (xs - pca_mean)
        d = np.linalg.norm(centroids - xp, axis=1)
        cid = int(np.argmin(d))
        cid_per_bar[i] = cid
        last_cid = cid
        if i % 100000 == 0:
            print(f"  {i:>8,}/{n:,}  ({i/n*100:.1f}%)", flush=True)

    # bars before WINDOW: assign cluster from first valid bar
    first_valid = next((c for c in cid_per_bar if c != -1), 0)
    cid_per_bar[cid_per_bar == -1] = first_valid

    out = pd.DataFrame({
        "time": swing["time"].values,
        "bar_idx": np.arange(n),
        "cid": cid_per_bar,
    })

    print("\nCluster distribution:")
    for cid in range(K):
        cnt = (cid_per_bar == cid).sum()
        name = (sel["cluster_names"].get(str(cid)) if isinstance(sel.get("cluster_names"), dict)
                else (sel["cluster_names"][cid] if "cluster_names" in sel else f"C{cid}"))
        print(f"  C{cid} {name:<14}  {cnt:>8,} bars  ({cnt/n*100:.1f}%)")

    out_path = os.path.join(OUT_DIR, "cluster_per_bar_v73.csv")
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}  ({os.path.getsize(out_path)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
