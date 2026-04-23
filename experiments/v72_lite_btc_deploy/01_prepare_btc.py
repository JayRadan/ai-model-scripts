"""
BTC pipeline stage 1: compute tech features + assign clusters + split per-cluster.

Input:  data/swing_v5_btc.csv  (42 cols: OHLC + f01-f20 + H1/H4 context + label)
Output: data/cluster_{cid}_data_btc.csv  (per-cluster CSVs with full features)

This replaces model_pipeline/01_labeler_v4 + 03_split_clusters_k5 for BTC:
  - we already have label from the exporter; map it to entry_class {0=BUY,1=FLAT,2=SELL}
  - compute 15 tech features that the rule scanner (04) consumes
  - assign cluster per-bar using the BTC selector (02_build_selector_k5_btc output)
  - write one CSV per cluster
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "/home/jay/Desktop/new-model-zigzag/model_pipeline")
import paths as P

SWING_CSV   = P.data("swing_v5_btc.csv")
SELECTOR_JS = P.data("regime_selector_btc_K5.json")
MIN_DATE    = "2016-01-01"

K = 5
WINDOW = 288
STEP = 288
CLUSTER_NAMES = {0:"Uptrend", 1:"MeanRevert", 2:"TrendRange", 3:"Downtrend", 4:"HighVol"}


def compute_atr(high, low, close, period=14):
    prev_c = np.concatenate([[close.iloc[0]], close.values[:-1]])
    tr = np.maximum.reduce([high.values - low.values,
                            np.abs(high.values - prev_c),
                            np.abs(low.values - prev_c)])
    return pd.Series(tr, index=close.index).rolling(period, min_periods=1).mean()


def simple_rsi(series, period):
    delta = series.diff()
    up = delta.clip(lower=0).rolling(period, min_periods=1).mean()
    dn = (-delta.clip(upper=0)).rolling(period, min_periods=1).mean()
    rs = up / (dn + 1e-10)
    return 100.0 - (100.0 / (1.0 + rs))


def add_tech_features(df):
    close, high, low = df["close"], df["high"], df["low"]
    atr14 = compute_atr(high, low, close, 14)
    df["rsi14"]     = simple_rsi(close, 14) / 100.0 - 0.5
    df["rsi6"]      = simple_rsi(close, 6)  / 100.0 - 0.5
    lo5, hi5        = low.rolling(5, min_periods=1).min(), high.rolling(5, min_periods=1).max()
    df["stoch_k"]   = (close - lo5) / (hi5 - lo5 + 1e-10)
    df["stoch_d"]   = df["stoch_k"].rolling(3, min_periods=1).mean()
    sma20           = close.rolling(20, min_periods=1).mean()
    std20           = close.rolling(20, min_periods=1).std(ddof=0).fillna(1e-10)
    df["bb_pct"]    = (close - (sma20 - 2.0*std20)) / ((sma20 + 2.0*std20) - (sma20 - 2.0*std20) + 1e-10)
    df["mom5"]      = (close - close.shift(5))  / (atr14 + 1e-10)
    df["mom10"]     = (close - close.shift(10)) / (atr14 + 1e-10)
    df["mom20"]     = (close - close.shift(20)) / (atr14 + 1e-10)
    lo10            = low.shift(1).rolling(10, min_periods=1).min()
    hi10            = high.shift(1).rolling(10, min_periods=1).max()
    df["ll_dist10"] = (close - lo10) / (atr14 + 1e-10)
    df["hh_dist10"] = (hi10 - close) / (atr14 + 1e-10)
    # volume proxy = normalized range
    rng_sma20       = (high - low).rolling(20, min_periods=1).mean()
    vol             = ((high - low) / (rng_sma20 + 1e-10)).clip(lower=1e-3)
    df["vol_accel"] = vol.rolling(3, min_periods=1).mean() / (vol.rolling(20, min_periods=1).mean() + 1e-10) - 1.0
    atr_sma50       = atr14.rolling(50, min_periods=1).mean()
    df["atr_ratio"] = atr14 / (atr_sma50 + 1e-10) - 1.0
    df["spread_norm"] = df["spread"].astype(float) / (atr14 + 1e-10)
    hour            = df["time"].dt.hour.astype(float)
    dow             = df["time"].dt.dayofweek.astype(float)
    df["hour_enc"]  = np.sin(2.0 * np.pi * hour / 24.0)
    df["dow_enc"]   = np.sin(2.0 * np.pi * dow  /  7.0)   # BTC trades 7 days/week
    df["_atr14"]    = atr14
    return df


def compute_fp(c, h, l, o):
    if len(c) < 10: return None
    returns = np.diff(c) / c[:-1]; bar_ranges = (h - l) / c
    fp = {}
    fp["weekly_return_pct"] = float(returns.sum())
    fp["volatility_pct"]    = float(returns.std())
    mean_ret = returns.mean()
    fp["trend_consistency"] = float(np.mean(np.sign(returns) == np.sign(mean_ret))) if abs(mean_ret) > 1e-12 else 0.5
    fp["trend_strength"]    = float(returns.sum() / (returns.std() + 1e-9))
    fp["volatility"]        = float(bar_ranges.mean())
    total_range             = (h.max() - l.min()) / c.mean()
    fp["range_vs_atr"]      = float(total_range / (bar_ranges.mean() + 1e-9))
    if len(returns) > 2:
        r1, r2 = returns[:-1], returns[1:]
        denom = r1.std() * r2.std()
        fp["return_autocorr"] = float(np.corrcoef(r1, r2)[0, 1]) if denom > 1e-12 else 0.0
    else:
        fp["return_autocorr"] = 0.0
    return fp


def main():
    print(f"Loading {SWING_CSV} ...")
    df = pd.read_csv(SWING_CSV, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)
    df = df[df["time"] >= MIN_DATE].reset_index(drop=True)
    print(f"  {len(df):,} rows  {df['time'].iat[0]} → {df['time'].iat[-1]}")

    # Map label → entry_class: label 1→BUY(0), label -1→SELL(2), label 0→FLAT(1)
    df["entry_class"] = df["label"].map({1: 0, 0: 1, -1: 2}).astype(int)

    print("Computing tech features ...")
    df = add_tech_features(df)

    print("Assigning regime cluster per bar ...")
    with open(SELECTOR_JS) as f: sel = json.load(f)
    scaler_mean = np.array(sel["scaler_mean"]); scaler_std = np.array(sel["scaler_std"])
    pca_mean    = np.array(sel["pca_mean"]);    pca_comp   = np.array(sel["pca_components"])
    centroids   = np.array(sel["centroids"]);   feat_names = sel["feat_names"]

    closes, highs, lows, opens = (df["close"].values.astype(np.float64),
                                   df["high"].values.astype(np.float64),
                                   df["low"].values.astype(np.float64),
                                   df["open"].values.astype(np.float64))
    bar_cluster = np.full(len(df), -1, dtype=int)
    for start in range(0, len(df) - WINDOW, STEP):
        end = start + WINDOW
        fp = compute_fp(closes[start:end], highs[start:end], lows[start:end], opens[start:end])
        if fp is None: continue
        vec = np.array([fp[k] for k in feat_names])
        scaled  = (vec - scaler_mean) / scaler_std
        rotated = (scaled - pca_mean) @ pca_comp.T
        cid     = int(np.argmin(np.sum((centroids - rotated) ** 2, axis=1)))
        bar_cluster[start:end] = cid
    # Forward-fill tail
    last = -1
    for i in range(len(bar_cluster)):
        if bar_cluster[i] >= 0: last = bar_cluster[i]
        elif last >= 0: bar_cluster[i] = last
    df["cluster"] = bar_cluster
    df = df[df["cluster"] >= 0].reset_index(drop=True)

    # Drop _atr14 helper
    if "_atr14" in df.columns: df = df.drop(columns=["_atr14"])

    BUY, FLAT, SELL = 0, 1, 2
    for cid in range(K):
        sub = df[df["cluster"] == cid].copy()
        if cid == 0: sub.loc[sub["entry_class"] == SELL, "entry_class"] = FLAT
        elif cid == 3: sub.loc[sub["entry_class"] == BUY, "entry_class"] = FLAT
        out = P.data(f"cluster_{cid}_data_btc.csv")
        sub.to_csv(out, index=False)
        buys  = (sub["entry_class"] == BUY).sum()
        sells = (sub["entry_class"] == SELL).sum()
        flats = (sub["entry_class"] == FLAT).sum()
        print(f"  C{cid} {CLUSTER_NAMES[cid]:>12}: {len(sub):>7,} rows  BUY={buys:>6,}  FLAT={flats:>6,}  SELL={sells:>6,}")


if __name__ == "__main__":
    main()
