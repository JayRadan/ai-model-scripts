#!/usr/bin/env python3
"""
V3 Causal Swing Labeler — XAUUSD M5
=====================================
ZERO look-ahead bias. Features at bar i use ONLY data from bars <= i.
Labels use forward price movement, but ONLY to determine training targets.

Label rules (BUY=0 / FLAT=1 / SELL=2):

  BUY at bar i:
    (1) low[i] == min(low[i-LOOKBACK .. i])          — causal local low
    (2) max(high[i-PRIOR_WINDOW .. i-1]) >= low[i] + MIN_SWING_ATR * atr[i]
                                                      — real prior downswing
    (3) within next MAX_FWD bars: high >= close[i] + TP_MULT*atr[i]
        BEFORE low <= close[i] - SL_MULT*atr[i]      — forward profitable

  SELL at bar i: symmetric (local high, prior upswing, short profitable)

  FLAT: everything else

Target signal density: ~3-5% BUY + ~3-5% SELL ≈ 10-15 signals/day on M5.

Output: labeled_v3.csv  (36 features + entry_class)
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

import paths as P

# ─── paths ─────────────────────────────────────────────────────────────────
DATA_PATH = Path(P.data("swing_v5_xauusd.csv"))
OUT_PATH  = Path(P.data("labeled_v4.csv"))

# ─── label parameters (v4: SHORT HORIZON + RECENT DATA ONLY) ──────────────
LOOKBACK      = 5    # tighter swing detection
PRIOR_WINDOW  = 15   # shorter prior swing window
MIN_SWING_ATR = 1.0  # easier to qualify (shorter targets need fewer ATRs)
TP_MULT       = 1.2  # take-profit at 1.2 × ATR  (v3 used 2.0)
SL_MULT       = 0.8  # stop-loss  at 0.8 × ATR  (v3 used 1.0)
MAX_FWD       = 10   # max forward bars (v3 used 40) — 10 × 5 min = 50 minutes
MIN_DATE      = "2020-01-01"   # drop pre-2020 data — gold regime shift around COVID

# ─── feature columns ───────────────────────────────────────────────────────
BASE_FEATURES = [
    "f01_CPR", "f02_WickAsym", "f03_BEF", "f04_TCS", "f05_SPI",
    "f06_LRSlope", "f07_RECR", "f08_SCM", "f09_HLER", "f10_EP",
    "f11_KE", "f12_MCS", "f13_Work", "f14_EDR", "f15_AI",
    "f16_PPShigh", "f16_PPSlow", "f17_SCR", "f18_RVD", "f19_WBER",
    "f20_NCDE",
]
TECH_FEATURES = [
    "rsi14", "rsi6", "stoch_k", "stoch_d", "bb_pct",
    "mom5", "mom10", "mom20",
    "ll_dist10", "hh_dist10",
    "vol_accel", "atr_ratio", "spread_norm",
    "hour_enc", "dow_enc",
]
HTF_FEATURES = [
    "h1_trend_sma20", "h1_trend_sma50", "h1_slope5", "h1_rsi14",
    "h1_atr_ratio", "h1_dist_sma20", "h1_dist_sma50",
    "h4_trend_sma20", "h4_trend_sma50", "h4_slope5", "h4_rsi14",
    "h4_atr_ratio", "h4_dist_sma20", "h4_dist_sma50",
]
ALL_FEATURES = BASE_FEATURES + TECH_FEATURES + HTF_FEATURES
RAW_COLS     = ["time", "open", "high", "low", "close", "spread"]

BUY  = 0
FLAT = 1
SELL = 2


# ─── helpers ───────────────────────────────────────────────────────────────

def read_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be"):
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeError, UnicodeDecodeError):
            continue
    raise RuntimeError(f"Cannot decode {path}")


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Simple-mean ATR matching MQL5 RawATR (no EMA)."""
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean().clip(lower=1e-10)


def simple_rsi(close: pd.Series, period: int) -> pd.Series:
    """
    Simple-mean RSI matching MQL5 manual computation.
    Returns values in [0, 100].
    """
    delta  = close.diff()
    gains  = delta.clip(lower=0).rolling(period, min_periods=period).mean()
    losses = (-delta).clip(lower=0).rolling(period, min_periods=period).mean()
    rs     = gains / (losses + 1e-10)
    return 100.0 - 100.0 / (1.0 + rs)


def compute_tech_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all 15 technical features from OHLCV. All fully causal."""
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["spread"].copy()   # use spread as proxy when volume unavailable
    # Use tick_volume if present; fall back to rolling high-low proxy
    if "tick_volume" in df.columns:
        volume = df["tick_volume"].astype(float)
    elif "volume" in df.columns:
        volume = df["volume"].astype(float)
    else:
        # Proxy: normalised range as activity measure
        volume = ((high - low) / (high - low).rolling(20, min_periods=1).mean()).clip(lower=1e-3)

    atr14 = compute_atr(high, low, close, 14)
    atr6  = compute_atr(high, low, close, 6)

    # RSI (normalised to [-0.5, 0.5])
    rsi14 = simple_rsi(close, 14) / 100.0 - 0.5
    rsi6  = simple_rsi(close, 6)  / 100.0 - 0.5

    # Stochastic %K(5), %D(3-period SMA of K)
    lo5      = low.rolling(5, min_periods=1).min()
    hi5      = high.rolling(5, min_periods=1).max()
    stoch_k  = (close - lo5) / (hi5 - lo5 + 1e-10)
    stoch_d  = stoch_k.rolling(3, min_periods=1).mean()

    # Bollinger Band %B (20, 2 stdev)
    sma20   = close.rolling(20, min_periods=1).mean()
    std20   = close.rolling(20, min_periods=1).std(ddof=0).fillna(1e-10)
    bb_low  = sma20 - 2.0 * std20
    bb_high = sma20 + 2.0 * std20
    bb_pct  = (close - bb_low) / (bb_high - bb_low + 1e-10)

    # Momentum (ATR-normalised price change)
    mom5  = (close - close.shift(5))  / (atr14 + 1e-10)
    mom10 = (close - close.shift(10)) / (atr14 + 1e-10)
    mom20 = (close - close.shift(20)) / (atr14 + 1e-10)

    # Distance from 10-bar low/high (past bars only via shift(1))
    lo10       = low.shift(1).rolling(10, min_periods=1).min()
    hi10       = high.shift(1).rolling(10, min_periods=1).max()
    ll_dist10  = (close - lo10)   / (atr14 + 1e-10)
    hh_dist10  = (hi10  - close)  / (atr14 + 1e-10)

    # Volume acceleration: SMA3 / SMA20 − 1
    vol_s3    = volume.rolling(3,  min_periods=1).mean()
    vol_s20   = volume.rolling(20, min_periods=1).mean()
    vol_accel = vol_s3 / (vol_s20 + 1e-10) - 1.0

    # ATR ratio: current ATR vs 50-bar SMA of ATR − 1
    atr_sma50  = atr14.rolling(50, min_periods=1).mean()
    atr_ratio  = atr14 / (atr_sma50 + 1e-10) - 1.0

    # Spread normalised by ATR
    spread_norm = df["spread"].astype(float) / (atr14 + 1e-10)

    # Cyclical time encoding
    hour      = df["time"].dt.hour.astype(float)
    dow       = df["time"].dt.dayofweek.astype(float)  # 0=Mon .. 4=Fri
    hour_enc  = np.sin(2.0 * np.pi * hour / 24.0)
    dow_enc   = np.sin(2.0 * np.pi * dow  /  5.0)

    df = df.copy()
    df["rsi14"]       = rsi14
    df["rsi6"]        = rsi6
    df["stoch_k"]     = stoch_k
    df["stoch_d"]     = stoch_d
    df["bb_pct"]      = bb_pct
    df["mom5"]        = mom5
    df["mom10"]       = mom10
    df["mom20"]       = mom20
    df["ll_dist10"]   = ll_dist10
    df["hh_dist10"]   = hh_dist10
    df["vol_accel"]   = vol_accel
    df["atr_ratio"]   = atr_ratio
    df["spread_norm"] = spread_norm
    df["hour_enc"]    = hour_enc
    df["dow_enc"]     = dow_enc
    df["_atr14"]      = atr14   # keep for labeling, removed before save
    return df


def compute_forward_outcomes(
    close: np.ndarray, high: np.ndarray, low: np.ndarray, atr: np.ndarray,
    tp_mult: float, sl_mult: float, max_fwd: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorised (chunk-based) forward TP/SL hit detection.
    Returns:
      buy_outcome[i]  = 1 if long TP hit before SL within max_fwd bars
      sell_outcome[i] = 1 if short TP hit before SL within max_fwd bars
    """
    n = len(close)
    buy_out  = np.zeros(n, dtype=np.int8)
    sell_out = np.zeros(n, dtype=np.int8)

    CHUNK = 50_000
    for start in range(0, n - max_fwd, CHUNK):
        end = min(start + CHUNK, n - max_fwd)
        sz  = end - start

        # Build future high/low windows for this chunk: shape (sz, max_fwd)
        fut_hi = np.empty((sz, max_fwd), dtype=np.float64)
        fut_lo = np.empty((sz, max_fwd), dtype=np.float64)
        for k in range(max_fwd):
            fut_hi[:, k] = high[start + k + 1 : end + k + 1]
            fut_lo[:, k] = low[ start + k + 1 : end + k + 1]

        tp_long  = (close[start:end] + tp_mult * atr[start:end])[:, None]
        sl_long  = (close[start:end] - sl_mult * atr[start:end])[:, None]
        tp_short = (close[start:end] - tp_mult * atr[start:end])[:, None]
        sl_short = (close[start:end] + sl_mult * atr[start:end])[:, None]

        # Long: find first bar where high >= tp or low <= sl
        tp_hit_l = fut_hi >= tp_long
        sl_hit_l = fut_lo <= sl_long
        f_tp_l = np.where(tp_hit_l.any(1), np.argmax(tp_hit_l, 1), max_fwd)
        f_sl_l = np.where(sl_hit_l.any(1), np.argmax(sl_hit_l, 1), max_fwd)
        buy_out[start:end] = (f_tp_l < f_sl_l).astype(np.int8)

        # Short: find first bar where low <= tp or high >= sl
        tp_hit_s = fut_lo <= tp_short
        sl_hit_s = fut_hi >= sl_short
        f_tp_s = np.where(tp_hit_s.any(1), np.argmax(tp_hit_s, 1), max_fwd)
        f_sl_s = np.where(sl_hit_s.any(1), np.argmax(sl_hit_s, 1), max_fwd)
        sell_out[start:end] = (f_tp_s < f_sl_s).astype(np.int8)

    return buy_out, sell_out


def assign_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign entry_class (BUY=0, FLAT=1, SELL=2) using causal local extrema
    + prior swing size filter + forward profitability.
    """
    n     = len(df)
    close = df["close"].to_numpy(np.float64)
    high  = df["high"].to_numpy(np.float64)
    low   = df["low"].to_numpy(np.float64)
    atr   = df["_atr14"].to_numpy(np.float64)

    print(f"  [label] computing forward outcomes (n={n:,}) …", flush=True)
    buy_out, sell_out = compute_forward_outcomes(
        close, high, low, atr, TP_MULT, SL_MULT, MAX_FWD
    )
    print("  [label] forward outcomes done", flush=True)

    # Causal local extrema:
    # bar i is local low  if low[i]  == min(low[i-LOOKBACK..i])
    # bar i is local high if high[i] == max(high[i-LOOKBACK..i])
    win = LOOKBACK + 1
    roll_min = df["low"].rolling(win, min_periods=win).min().to_numpy()
    roll_max = df["high"].rolling(win, min_periods=win).max().to_numpy()
    is_loc_low  = (np.abs(low  - roll_min) < 1e-8)
    is_loc_high = (np.abs(high - roll_max) < 1e-8)

    # Prior swing size filter using pandas for efficiency
    prior_high = df["high"].rolling(PRIOR_WINDOW, min_periods=1).max().shift(1).to_numpy()
    prior_low  = df["low"].rolling(PRIOR_WINDOW, min_periods=1).min().shift(1).to_numpy()
    has_prior_down = (prior_high - low)  >= (MIN_SWING_ATR * atr)   # came from above
    has_prior_up   = (high - prior_low)  >= (MIN_SWING_ATR * atr)   # came from below

    buy_mask  = is_loc_low  & has_prior_down & buy_out.astype(bool)
    sell_mask = is_loc_high & has_prior_up   & sell_out.astype(bool)
    overlap   = buy_mask & sell_mask

    entry = np.full(n, FLAT, dtype=np.int8)
    entry[buy_mask  & ~overlap] = BUY
    entry[sell_mask & ~overlap] = SELL

    df = df.copy()
    df["entry_class"] = entry
    return df


def main() -> None:
    print("=" * 60)
    print("  V3 Causal Swing Labeler")
    print("=" * 60)

    # ── load ────────────────────────────────────────────────────────────────
    print(f"\n[1] Loading {DATA_PATH} …", flush=True)
    df = read_csv(DATA_PATH)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    req_cols = set(RAW_COLS + BASE_FEATURES + HTF_FEATURES)
    miss = sorted(req_cols - set(df.columns))
    if miss:
        sys.exit(f"Missing columns: {miss}")
    for c in RAW_COLS[1:] + BASE_FEATURES + HTF_FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = (df.replace([np.inf, -np.inf], np.nan)
            .dropna(subset=RAW_COLS[1:] + BASE_FEATURES + HTF_FEATURES)
            .sort_values("time")
            .reset_index(drop=True))
    print(f"  rows (all): {len(df):,}  range: {df['time'].min()} → {df['time'].max()}", flush=True)

    # v4: drop pre-2020 data — gold regime changed with COVID
    before = len(df)
    df = df[df["time"] >= pd.Timestamp(MIN_DATE)].reset_index(drop=True)
    print(f"  rows (>= {MIN_DATE}): {len(df):,}  dropped {before - len(df):,} pre-filter", flush=True)
    print(f"  new range: {df['time'].min()} → {df['time'].max()}", flush=True)

    # ── technical features ──────────────────────────────────────────────────
    print("\n[2] Computing technical features …", flush=True)
    df = compute_tech_features(df)
    df = df.dropna(subset=ALL_FEATURES + ["_atr14"]).reset_index(drop=True)
    print(f"  rows after dropna: {len(df):,}", flush=True)

    # ── labels ──────────────────────────────────────────────────────────────
    print("\n[3] Assigning labels …", flush=True)
    df = assign_labels(df)

    counts = dict(zip(*np.unique(df["entry_class"], return_counts=True)))
    n      = len(df)
    buy_n  = counts.get(BUY,  0)
    flat_n = counts.get(FLAT, 0)
    sell_n = counts.get(SELL, 0)
    print(f"  BUY  = {buy_n:>8,}  ({100*buy_n/n:.2f}%)")
    print(f"  FLAT = {flat_n:>8,}  ({100*flat_n/n:.2f}%)")
    print(f"  SELL = {sell_n:>8,}  ({100*sell_n/n:.2f}%)")

    # Estimate trades/day
    trading_days = max((df["time"].max() - df["time"].min()).days * 5 / 7, 1)
    tpd = (buy_n + sell_n) / trading_days
    print(f"  Estimated signals/day (raw): {tpd:.1f}")

    # ── save ────────────────────────────────────────────────────────────────
    save_cols = RAW_COLS + ALL_FEATURES + ["entry_class"]
    df = df.drop(columns=["_atr14"], errors="ignore")
    df[save_cols].to_csv(OUT_PATH, index=False)
    print(f"\n[4] Saved → {OUT_PATH}  ({len(df):,} rows, {len(ALL_FEATURES)} features)")
    print("Done.\n")


if __name__ == "__main__":
    main()
