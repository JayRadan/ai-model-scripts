"""
v75 step 00: build a 15-feature per-bar fingerprint matrix on the full XAU
swing. Rolling = every bar gets its own fingerprint computed from the
trailing window. STEP=1, no skipping.

Output: data/fingerprints_rich.parquet (~1M rows × 15 feats + time)
"""
from __future__ import annotations
import os, sys, time as _time
import numpy as np, pandas as pd

ZIGZAG = "/home/jay/Desktop/new-model-zigzag"
sys.path.insert(0, os.path.join(ZIGZAG, "model_pipeline"))
import paths as P

OUT = os.path.join(ZIGZAG, "experiments/v75_regime_v2/data/fingerprints_rich.parquet")
SWING = P.data("swing_v5_xauusd.csv")
MIN_DATE = "2018-01-01"

# Multi-scale windows (bars on M5 = 1h/3h/8h/24h)
W_1H, W_3H, W_8H, W_24H = 12, 36, 96, 288


def rolling_return(c, w):
    """Trailing w-bar simple return, NaN before warmup."""
    out = np.full_like(c, np.nan, dtype=np.float64)
    out[w:] = (c[w:] - c[:-w]) / c[:-w]
    return out


def rolling_slope(c, w):
    """Linear-regression slope of last w bars (ATR-normalised) — captures
    direction + steepness in one number."""
    n = len(c)
    out = np.full(n, np.nan, dtype=np.float64)
    x = np.arange(w, dtype=np.float64); xm = x.mean(); xv = ((x-xm)**2).sum()
    for i in range(w-1, n):
        seg = c[i-w+1:i+1]
        ym = seg.mean()
        slope = ((x-xm) * (seg-ym)).sum() / xv
        out[i] = slope / c[i]   # ATR-free normalisation: slope per unit price
    return out


def rolling_std(returns, w):
    """Trailing volatility = std of returns over w bars."""
    return pd.Series(returns).rolling(w, min_periods=w).std().values


def rolling_pos_in_range(c, h, l, w):
    """(close - rolling_low_w) / (rolling_high_w - rolling_low_w) ∈ [0,1].
    0 = at floor, 1 = at ceiling. Captures overbought/oversold."""
    s_high = pd.Series(h).rolling(w, min_periods=w).max().values
    s_low  = pd.Series(l).rolling(w, min_periods=w).min().values
    rng = s_high - s_low
    out = np.where(rng > 1e-9, (c - s_low) / rng, 0.5)
    return np.where(np.isfinite(out), out, 0.5)


def rolling_hh_ll_balance(h, l, w):
    """Count higher-highs minus lower-lows in last w bars (normalised by w)."""
    n = len(h)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(w, n):
        hh = sum(1 for k in range(i-w+1, i+1) if h[k] > h[k-1])
        ll = sum(1 for k in range(i-w+1, i+1) if l[k] < l[k-1])
        out[i] = (hh - ll) / w
    return out


def rolling_streak(c, max_look=10):
    """Length of current up/down streak (signed): +N for N consecutive up
    closes, -N for N consecutive down closes. Capped at ±max_look."""
    n = len(c); out = np.zeros(n)
    for i in range(1, n):
        if c[i] > c[i-1]:   out[i] = max(1.0, out[i-1] + 1) if out[i-1] >= 0 else 1.0
        elif c[i] < c[i-1]: out[i] = min(-1.0, out[i-1] - 1) if out[i-1] <= 0 else -1.0
        else:               out[i] = out[i-1]
    return np.clip(out, -max_look, max_look)


def main():
    t0 = _time.time()
    print(f"Loading {SWING}", flush=True)
    df = pd.read_csv(SWING, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)
    df = df[df["time"] >= MIN_DATE].reset_index(drop=True)
    print(f"  {len(df):,} bars  {df['time'].iloc[0]} → {df['time'].iloc[-1]}")

    c = df["close"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    o = df["open"].values.astype(np.float64)

    print("Computing trailing returns...", flush=True)
    ret_1h  = rolling_return(c, W_1H)
    ret_3h  = rolling_return(c, W_3H)
    ret_8h  = rolling_return(c, W_8H)
    ret_24h = rolling_return(c, W_24H)

    print("Computing slopes...", flush=True)
    slope_1h  = rolling_slope(c, W_1H)
    slope_3h  = rolling_slope(c, W_3H)
    slope_24h = rolling_slope(c, W_24H)
    slope_accel = slope_1h - slope_24h   # >0 = recent steeper than long-term

    print("Computing volatility...", flush=True)
    bar_returns = np.concatenate([[0.0], (c[1:] - c[:-1]) / c[:-1]])
    vol_1h  = rolling_std(bar_returns, W_1H)
    vol_24h = rolling_std(bar_returns, W_24H)
    vol_ratio = vol_1h / np.where(vol_24h > 1e-12, vol_24h, 1e-12)
    vol_ratio = np.clip(vol_ratio, 0.0, 10.0)

    print("Computing position-in-range...", flush=True)
    pos_24h = rolling_pos_in_range(c, h, l, W_24H)

    print("Computing HH/LL balance...", flush=True)
    hh_ll_3h  = rolling_hh_ll_balance(h, l, W_3H)
    hh_ll_24h = rolling_hh_ll_balance(h, l, W_24H)

    print("Computing streak...", flush=True)
    streak = rolling_streak(c, max_look=10) / 10.0   # in [-1, +1]

    print("Computing trend-consistency 24h...", flush=True)
    # fraction of last 24h bars in same direction as the net 24h return
    sign24 = np.sign(ret_24h)
    bar_sign = np.sign(bar_returns)
    consistency_24h = np.full(len(c), np.nan, dtype=np.float64)
    for i in range(W_24H, len(c)):
        seg_sign = bar_sign[i-W_24H+1:i+1]
        s24 = sign24[i]
        consistency_24h[i] = float(np.mean(seg_sign == s24)) if abs(s24) > 0 else 0.5

    print("Assembling matrix...", flush=True)
    out = pd.DataFrame({
        "time": df["time"].values,
        "close": c,
        "ret_1h": ret_1h,
        "ret_3h": ret_3h,
        "ret_8h": ret_8h,
        "ret_24h": ret_24h,
        "slope_1h": slope_1h,
        "slope_3h": slope_3h,
        "slope_24h": slope_24h,
        "slope_accel": slope_accel,
        "vol_1h": vol_1h,
        "vol_24h": vol_24h,
        "vol_ratio": vol_ratio,
        "pos_24h": pos_24h,
        "hh_ll_3h": hh_ll_3h,
        "hh_ll_24h": hh_ll_24h,
        "streak": streak,
        "consistency_24h": consistency_24h,
    })
    out = out.dropna().reset_index(drop=True)
    out.to_parquet(OUT, index=False)
    print(f"\nWrote {OUT}  ({len(out):,} rows × {len(out.columns)} cols, "
          f"{os.path.getsize(OUT)/1e6:.1f} MB)")
    print(f"Time: {_time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
