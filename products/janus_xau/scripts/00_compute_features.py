"""
v7.4 Pivot Score — feature computation.

Builds on v73 features with new pivot-confluence features:
  - dist_h1_swing_high   : ATR-normalized distance to last H1 swing high
  - dist_h1_swing_low    : same for swing low
  - dist_h4_swing_high   : same on H4
  - dist_h4_swing_low    : same on H4
  - dist_session_hod     : ATR-normalized distance to session high (since 00:00 GMT)
  - dist_session_lod     : same for low
  - dist_round_50        : ATR-normalized distance to nearest 50-USD round level
  - dist_round_10        : same for 10-USD level
  - rsi_h1               : RSI(14) on H1 close (ffill onto M5)
  - streak_count         : signed count of consecutive same-direction closes

Reuses v73's already-computed features file by extending it.

Output: experiments/v74_pivot_score/data/features_v74.csv
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import sys

sys.path.insert(0, "/home/jay/Desktop/new-model-zigzag/model_pipeline")
import paths as P

V73_FEATURES = "/home/jay/Desktop/new-model-zigzag/experiments/v73_pivot_oracle/data/features_v73.csv"
OUT_DIR = "/home/jay/Desktop/new-model-zigzag/experiments/v74_pivot_score/data"
os.makedirs(OUT_DIR, exist_ok=True)


def compute_atr(H, L, C, n=14):
    tr = np.concatenate([
        [H[0] - L[0]],
        np.maximum.reduce([H[1:] - L[1:], np.abs(H[1:] - C[:-1]), np.abs(L[1:] - C[:-1])]),
    ])
    return pd.Series(tr).rolling(n, min_periods=n).mean().values


def compute_rsi(C, n=14):
    diff = np.diff(C, prepend=C[0])
    up = np.where(diff > 0, diff, 0.0); dn = np.where(diff < 0, -diff, 0.0)
    roll_up = pd.Series(up).rolling(n, min_periods=n).mean().values
    roll_dn = pd.Series(dn).rolling(n, min_periods=n).mean().values
    rs = np.divide(roll_up, roll_dn, out=np.zeros_like(roll_up), where=roll_dn > 0)
    return 100.0 - (100.0 / (1.0 + rs))


def fractal_pivots(H, L, n_left=2, n_right=2):
    """Return arrays of pivot_high indices and pivot_low indices using N-bar fractals."""
    nbars = len(H); ph = np.zeros(nbars, dtype=bool); pl = np.zeros(nbars, dtype=bool)
    for i in range(n_left, nbars - n_right):
        if H[i] == max(H[i - n_left:i + n_right + 1]) and H[i] > H[i - 1]:
            ph[i] = True
        if L[i] == min(L[i - n_left:i + n_right + 1]) and L[i] < L[i - 1]:
            pl[i] = True
    return ph, pl


def dist_to_recent_pivots(prices, pivot_mask, atr, max_lookback=2000):
    """For each bar, ATR-normalized distance to most recent pivot (up to max_lookback bars back).
    Returns +inf if no pivot within lookback."""
    nbars = len(prices)
    out = np.full(nbars, 0.0, dtype=np.float64)
    last_idx = -1
    for i in range(nbars):
        if pivot_mask[i]: last_idx = i
        if last_idx == -1 or i - last_idx > max_lookback or atr[i] <= 0:
            out[i] = 0.0; continue
        out[i] = (prices[i] - prices[last_idx]) / atr[i]   # signed
    return out


def main():
    print(f"Loading v73 base features: {V73_FEATURES}", flush=True)
    base = pd.read_csv(V73_FEATURES, parse_dates=["time"])
    print(f"  {len(base):,} rows  {len(base.columns)} cols", flush=True)

    print("Loading swing CSV for H1/H4 resampling...", flush=True)
    swing = pd.read_csv(P.data("swing_v5_xauusd.csv"), parse_dates=["time"])
    swing = swing.sort_values("time").reset_index(drop=True)
    H = swing["high"].values.astype(np.float64); L = swing["low"].values.astype(np.float64)
    C = swing["close"].values.astype(np.float64); O = swing["open"].values.astype(np.float64)
    atr = compute_atr(H, L, C, 14)

    # ---- H1 / H4 swing pivots ----
    print("Resampling to H1, H4...", flush=True)
    s_idx = swing.set_index("time")
    h1 = s_idx[["open","high","low","close"]].resample("1h").agg(
        {"open":"first","high":"max","low":"min","close":"last"}).dropna()
    h4 = s_idx[["open","high","low","close"]].resample("4h").agg(
        {"open":"first","high":"max","low":"min","close":"last"}).dropna()

    h1_atr = compute_atr(h1["high"].values, h1["low"].values, h1["close"].values, 14)
    h4_atr = compute_atr(h4["high"].values, h4["low"].values, h4["close"].values, 14)

    h1_ph, h1_pl = fractal_pivots(h1["high"].values, h1["low"].values, 2, 2)
    h4_ph, h4_pl = fractal_pivots(h4["high"].values, h4["low"].values, 2, 2)

    h1_dist_high = dist_to_recent_pivots(h1["high"].values, h1_ph, h1_atr)
    h1_dist_low  = dist_to_recent_pivots(h1["low"].values,  h1_pl, h1_atr)
    h4_dist_high = dist_to_recent_pivots(h4["high"].values, h4_ph, h4_atr)
    h4_dist_low  = dist_to_recent_pivots(h4["low"].values,  h4_pl, h4_atr)
    h1_rsi = compute_rsi(h1["close"].values, 14)

    # ffill back onto M5
    def reindex_ffill(arr_index, arr_values, target_index):
        s = pd.Series(arr_values, index=arr_index).shift(1)   # shift to avoid look-ahead
        return s.reindex(target_index, method="ffill").values

    base_idx = swing.set_index("time").index
    base["dist_h1_swing_high"] = reindex_ffill(h1.index, h1_dist_high, base_idx)
    base["dist_h1_swing_low"]  = reindex_ffill(h1.index, h1_dist_low,  base_idx)
    base["dist_h4_swing_high"] = reindex_ffill(h4.index, h4_dist_high, base_idx)
    base["dist_h4_swing_low"]  = reindex_ffill(h4.index, h4_dist_low,  base_idx)
    base["rsi_h1"]             = reindex_ffill(h1.index, h1_rsi,       base_idx)

    # ---- Session HOD / LOD (since midnight GMT) ----
    print("Computing session HOD/LOD distances...", flush=True)
    swing["date"] = swing["time"].dt.date
    cum_hi = swing.groupby("date")["high"].cummax().values
    cum_lo = swing.groupby("date")["low"].cummin().values
    base["dist_session_hod"] = np.where(atr > 0, (C - cum_hi) / atr, 0.0)   # negative if below HOD
    base["dist_session_lod"] = np.where(atr > 0, (C - cum_lo) / atr, 0.0)   # positive if above LOD

    # ---- Round-number distance ----
    print("Computing round-number distances...", flush=True)
    nearest_50 = np.round(C / 50.0) * 50.0
    nearest_10 = np.round(C / 10.0) * 10.0
    base["dist_round_50"] = np.where(atr > 0, (C - nearest_50) / atr, 0.0)
    base["dist_round_10"] = np.where(atr > 0, (C - nearest_10) / atr, 0.0)

    # ---- Streak count ----
    print("Computing close-direction streak...", flush=True)
    diff = np.sign(np.diff(C, prepend=C[0]))
    streak = np.zeros(len(diff))
    s = 0
    for i in range(1, len(diff)):
        s = s + diff[i] if np.sign(diff[i]) == np.sign(s) else diff[i]
        streak[i] = s
    base["streak_count"] = streak

    # NaN -> 0
    for c in base.columns:
        if c in ("time", "bar_idx"): continue
        if base[c].isna().any():
            base[c] = base[c].fillna(0)

    out_path = os.path.join(OUT_DIR, "features_v74.csv")
    base.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}  ({os.path.getsize(out_path)/1e6:.1f} MB)")
    print(f"Total feature columns: {len(base.columns)}")


if __name__ == "__main__":
    main()
