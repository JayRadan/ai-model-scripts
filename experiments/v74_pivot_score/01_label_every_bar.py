"""
v7.4 Pivot Score — per-bar oracle-exit labelling.

For every bar i, simulate the best-possible long AND short trade entered at C[i]:
  - SL hard at 4 * ATR (Oracle's safety floor)
  - Within MAX_HOLD = 60 bars
  - Exit = oracle exit (peak unrealized R if SL didn't fire; else -1R)

Outputs per bar:
  best_long_R    : max favorable R for long, or -1.0 if SL hit
  best_short_R   : same for short
  best_R         : max(best_long_R, best_short_R)
  best_dir       : +1 if long won, -1 if short won, 0 if both flat
  is_pivot_15    : 1 if best_R >= 1.5
  is_pivot_25    : 1 if best_R >= 2.5
  is_pivot_4     : 1 if best_R >= 4.0

Output: experiments/v74_pivot_score/data/labels_v74.csv
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/jay/Desktop/new-model-zigzag/model_pipeline")
import paths as P

OUT_DIR = "/home/jay/Desktop/new-model-zigzag/experiments/v74_pivot_score/data"
os.makedirs(OUT_DIR, exist_ok=True)

MAX_HOLD = 60
SL_HARD = 4.0


def compute_atr(H, L, C, n=14):
    tr = np.concatenate([
        [H[0] - L[0]],
        np.maximum.reduce([H[1:] - L[1:], np.abs(H[1:] - C[:-1]), np.abs(L[1:] - C[:-1])]),
    ])
    return pd.Series(tr).rolling(n, min_periods=n).mean().values


def oracle_exit_R(direction: int, entry_idx: int, atr_e: float,
                   H: np.ndarray, L: np.ndarray, C: np.ndarray) -> float:
    """Best possible R for a trade entered at C[entry_idx] with hard SL=4*ATR."""
    if not np.isfinite(atr_e) or atr_e <= 0: return 0.0
    sl_dist = SL_HARD * atr_e
    entry_px = C[entry_idx]
    end = min(entry_idx + 1 + MAX_HOLD, len(H))
    max_r = 0.0
    for j in range(entry_idx + 1, end):
        if direction == +1:
            if (entry_px - L[j]) >= sl_dist: return -1.0
            r = (H[j] - entry_px) / sl_dist
        else:
            if (H[j] - entry_px) >= sl_dist: return -1.0
            r = (entry_px - L[j]) / sl_dist
        if r > max_r: max_r = r
    return max_r


def main():
    print("Loading swing CSV...", flush=True)
    swing = pd.read_csv(P.data("swing_v5_xauusd.csv"), parse_dates=["time"])
    swing = swing.sort_values("time").reset_index(drop=True)
    H = swing["high"].values.astype(np.float64); L = swing["low"].values.astype(np.float64)
    C = swing["close"].values.astype(np.float64)
    n = len(swing)
    atr = compute_atr(H, L, C, 14)
    print(f"  {n:,} bars; oracle-simulating long+short for each...", flush=True)

    long_R = np.zeros(n); short_R = np.zeros(n)
    for i in range(14, n - 1):
        a = atr[i]
        if not np.isfinite(a) or a <= 0: continue
        long_R[i] = oracle_exit_R(+1, i, a, H, L, C)
        short_R[i] = oracle_exit_R(-1, i, a, H, L, C)
        if i % 100000 == 0 and i > 0:
            print(f"  {i:>8,}/{n:,}  ({i/n*100:.1f}%)", flush=True)

    best_R = np.maximum(long_R, short_R)
    best_dir = np.where(long_R > short_R, 1, np.where(short_R > long_R, -1, 0))

    out = pd.DataFrame({
        "time": swing["time"].values,
        "bar_idx": np.arange(n),
        "atr": atr,
        "best_long_R": long_R,
        "best_short_R": short_R,
        "best_R": best_R,
        "best_dir": best_dir,
        "is_pivot_15": (best_R >= 1.5).astype(int),
        "is_pivot_25": (best_R >= 2.5).astype(int),
        "is_pivot_4":  (best_R >= 4.0).astype(int),
    })
    print()
    print(f"  best_R distribution:  median {np.median(best_R):.2f}  mean {best_R.mean():.2f}  p90 {np.percentile(best_R, 90):.2f}  p99 {np.percentile(best_R, 99):.2f}")
    print(f"  is_pivot_15: {out['is_pivot_15'].sum():,} / {n:,}  ({out['is_pivot_15'].mean()*100:.1f}%)")
    print(f"  is_pivot_25: {out['is_pivot_25'].sum():,} / {n:,}  ({out['is_pivot_25'].mean()*100:.1f}%)")
    print(f"  is_pivot_4 : {out['is_pivot_4'].sum():,} / {n:,}  ({out['is_pivot_4'].mean()*100:.1f}%)")

    out_path = os.path.join(OUT_DIR, "labels_v74.csv")
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}  ({os.path.getsize(out_path)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
