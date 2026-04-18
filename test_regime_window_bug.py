"""
Empirically verify the regime fingerprint window-size bug.

For each instrument:
  1. Load recent OHLC (swing_v5_*.csv)
  2. At many end-points, compute the 7-dim fingerprint with:
       (a) WINDOW = 1440 bars  (the buggy live EA value)
       (b) WINDOW =  288 bars  (the correct training value)
  3. Apply the *exact* MQL5 pipeline: scaler → PCA → nearest centroid
  4. Report: cluster distribution under each window

If (a) is stuck on C0 and (b) is balanced across clusters, the window
mismatch is the bug.
"""
from __future__ import annotations
import json, re, os
import numpy as np
import pandas as pd

ROOT = "/home/jay/Desktop/new-model-zigzag"
CLUSTER_NAMES = {0:"Uptrend", 1:"MeanRevert", 2:"TrendRange", 3:"Downtrend", 4:"HighVol"}


def parse_mqh_constants(path: str) -> dict:
    """Extract SCALER_MEAN, SCALER_STD, PCA_MEAN, PCA_COMP, CENTROIDS arrays."""
    src = open(path).read()
    def vec(name):
        m = re.search(rf"{name}\s*\[[^\]]*\]\s*=\s*\{{([^}}]+)\}}", src)
        if not m: raise ValueError(f"missing {name}")
        return np.array([float(x) for x in m.group(1).split(",") if x.strip()])
    def mat(name, rows, cols):
        m = re.search(rf"{name}\s*\[[^\]]*\]\s*\[[^\]]*\]\s*=\s*\{{(.*?)\}};", src, re.DOTALL)
        if not m: raise ValueError(f"missing {name}")
        body = m.group(1)
        rows_txt = re.findall(r"\{([^{}]+)\}", body)
        out = np.array([[float(x) for x in r.split(",") if x.strip()] for r in rows_txt])
        assert out.shape == (rows, cols), f"{name} shape {out.shape}"
        return out
    K = 5; N = 7
    return {
        "scaler_mean": vec("REGIME_SCALER_MEAN"),
        "scaler_std":  vec("REGIME_SCALER_STD"),
        "pca_mean":    vec("REGIME_PCA_MEAN"),
        "pca_comp":    mat("REGIME_PCA_COMP", N, N),
        "centroids":   mat("REGIME_CENTROIDS", K, N),
    }


def compute_fingerprint(c, h, l):
    """Reproduces ComputeWeekFingerprint (MQL5) and compute_fingerprint (py)."""
    returns = np.diff(c) / c[:-1]
    if len(returns) < 3:
        return None
    bar_ranges = (h - l) / c
    mean_ret = returns.mean()
    std_ret = returns.std(ddof=0)  # MQL5 uses ddof=0; Python std()->ddof=1 (tiny diff)
    pos = (returns > 0).sum()
    neg = (returns < 0).sum()
    if abs(mean_ret) > 1e-12:
        trend_consistency = pos / len(returns) if mean_ret > 0 else neg / len(returns)
    else:
        trend_consistency = 0.5
    r1, r2 = returns[:-1], returns[1:]
    denom = r1.std() * r2.std()
    autocorr = float(np.corrcoef(r1, r2)[0, 1]) if denom > 1e-12 else 0.0
    return np.array([
        returns.sum(),                                 # weekly_return_pct
        std_ret,                                       # volatility_pct
        trend_consistency,
        returns.sum() / (std_ret + 1e-9),              # trend_strength
        bar_ranges.mean(),                             # volatility
        (h.max() - l.min()) / c.mean() / (bar_ranges.mean() + 1e-9),  # range_vs_atr
        autocorr,
    ])


def classify(fp: np.ndarray, cfg: dict) -> int:
    """MQL5 ClassifyRegime reproduction."""
    scaled = (fp - cfg["scaler_mean"]) / cfg["scaler_std"]
    rotated = cfg["pca_comp"] @ (scaled - cfg["pca_mean"])
    # nearest centroid
    dists = np.linalg.norm(cfg["centroids"] - rotated, axis=1)
    return int(np.argmin(dists))


def run(name, swing_path, mqh_path, n_samples=200):
    print(f"\n{'='*70}\n{name}\n{'='*70}")
    df = pd.read_csv(swing_path, usecols=["time","open","high","low","close"],
                     parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    cfg = parse_mqh_constants(mqh_path)
    print(f"  loaded {len(df):,} bars from {df['time'].iloc[0]} to {df['time'].iloc[-1]}")

    # take n_samples evenly spaced end-points from the last ~year of data
    n = len(df)
    end_idxs = np.linspace(n - 500_000 if n > 500_000 else 10_000,
                           n - 1, n_samples).astype(int)

    dist_buggy = np.zeros(5, dtype=int)
    dist_fix   = np.zeros(5, dtype=int)
    for end in end_idxs:
        if end < 1500: continue
        # window = 1440 (buggy live value)
        bar_1440 = df.iloc[end - 1440:end]
        fp_b = compute_fingerprint(bar_1440["close"].values, bar_1440["high"].values, bar_1440["low"].values)
        if fp_b is not None:
            dist_buggy[classify(fp_b, cfg)] += 1
        # window = 288 (correct trained value)
        bar_288 = df.iloc[end - 288:end]
        fp_c = compute_fingerprint(bar_288["close"].values, bar_288["high"].values, bar_288["low"].values)
        if fp_c is not None:
            dist_fix[classify(fp_c, cfg)] += 1

    total_b = dist_buggy.sum(); total_f = dist_fix.sum()
    print(f"\n  {'cluster':<14} {'BUGGY (1440)':<18} {'FIXED (288)':<18}")
    for cid in range(5):
        pb = dist_buggy[cid] / total_b * 100
        pf = dist_fix[cid]   / total_f * 100
        print(f"  C{cid} {CLUSTER_NAMES[cid]:<11} "
              f"{dist_buggy[cid]:3d} ({pb:5.1f}%)     "
              f"{dist_fix[cid]:3d} ({pf:5.1f}%)")
    c0b = dist_buggy[0] / total_b * 100
    c0f = dist_fix[0]   / total_f * 100
    print(f"\n  Verdict: Buggy routes {c0b:.0f}% to C0. Fixed routes {c0f:.0f}%.")
    print(f"           Fixed is {'balanced' if c0f < 60 else 'still dominated'} across regimes.")


if __name__ == "__main__":
    run("XAUUSD (Midas)",
        f"{ROOT}/data/swing_v5_xauusd.csv",
        "/home/jay/.mt5/drive_c/Program Files/MetaTrader 5/MQL5/Include/regime_selector.mqh")
    run("EURUSD (Meridian)",
        f"{ROOT}/eurusd/data/swing_v5_eurusd.csv",
        "/home/jay/.mt5/drive_c/Program Files/MetaTrader 5/MQL5/Include/regime_selector_eurusd.mqh")
    run("GBPJPY (Samurai)",
        f"{ROOT}/gbpjpy/data/swing_v5_gbpjpy.csv",
        "/home/jay/.mt5/drive_c/Program Files/MetaTrader 5/MQL5/Include/regime_selector_gbpjpy.mqh")
