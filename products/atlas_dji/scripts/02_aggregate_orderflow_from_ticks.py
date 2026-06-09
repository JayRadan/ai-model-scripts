"""Aggregate Dukascopy Dow ticks → M1 with orderflow features.
Saves to data/m1_dji_orderflow.parquet. Same schema as XAU.
"""
from __future__ import annotations

import time
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/jay/Desktop/new-model-zigzag")
OUT = ROOT / "data" / "m1_dji_orderflow.parquet"


def aggregate_one_day(parquet_path: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    if len(df) == 0:
        return pd.DataFrame()
    mid = (df["bidPrice"] + df["askPrice"]).to_numpy() / 2.0
    bv = df["bidVolume"].to_numpy()
    av = df["askVolume"].to_numpy()
    tot_v = bv + av
    mid_prev = np.concatenate([[mid[0]], mid[:-1]])
    sign = np.where(mid > mid_prev, 1.0, np.where(mid < mid_prev, -1.0, 0.0))
    signed_vol = sign * tot_v
    spread = (df["askPrice"] - df["bidPrice"]).to_numpy()

    df2 = pd.DataFrame({
        "mid": mid, "tot_v": tot_v, "bid_v": bv, "ask_v": av,
        "signed_vol": signed_vol, "spread": spread,
    }, index=df.index)
    g = df2.resample("1min")
    m1 = pd.DataFrame({
        "open": g["mid"].first(),
        "high": g["mid"].max(),
        "low": g["mid"].min(),
        "close": g["mid"].last(),
        "tick_volume": g["tot_v"].sum(),
        "n_ticks": g["mid"].count(),
        "signed_flow": g["signed_vol"].sum(),
        "abs_volume": g["tot_v"].sum(),
        "bid_v_sum": g["bid_v"].sum(),
        "ask_v_sum": g["ask_v"].sum(),
        "median_spread": g["spread"].median(),
    }).dropna(subset=["open"])
    return m1


def main():
    tick_dir = ROOT / "data" / "ticks" / "dji"
    parquets = sorted(glob(str(tick_dir / "*.parquet")))
    print(f"  aggregating {len(parquets)} tick parquets → M1 orderflow ...", flush=True)
    t0 = time.time()
    frames = []
    for i, p in enumerate(parquets):
        try:
            m1 = aggregate_one_day(p)
            if len(m1): frames.append(m1)
        except Exception as e:
            print(f"    skip {Path(p).name}: {e}")
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(parquets)}  ({time.time()-t0:.0f}s)", flush=True)
    m1 = pd.concat(frames).sort_index()
    m1 = m1[~m1.index.duplicated(keep="first")]

    m1["imbalance_ratio"] = m1["signed_flow"] / m1["abs_volume"].replace(0, np.nan)
    m1["bid_ask_vol_ratio"] = m1["bid_v_sum"] / (m1["bid_v_sum"] + m1["ask_v_sum"]).replace(0, np.nan)
    m1["vpin_proxy"] = (m1["signed_flow"].abs() / m1["abs_volume"].replace(0, np.nan))
    for w in (5, 15, 60):
        m1[f"cum_signed_{w}"] = m1["signed_flow"].rolling(w, min_periods=1).sum()
        m1[f"cum_signed_abs_{w}"] = m1["signed_flow"].abs().rolling(w, min_periods=1).sum()
        m1[f"flow_persistence_{w}"] = m1[f"cum_signed_{w}"] / m1[f"cum_signed_abs_{w}"].replace(0, np.nan)
    m1["spread_vol_50"] = m1["median_spread"].rolling(50, min_periods=10).std()
    m1["tick_intensity_50"] = m1["n_ticks"] / m1["n_ticks"].rolling(50, min_periods=10).mean()

    m1 = m1.reset_index().rename(columns={"timestamp": "time"})
    m1["time"] = pd.to_datetime(m1["time"]).dt.tz_localize(None)
    m1.to_parquet(OUT)
    print(f"  saved {OUT.name} → {len(m1):,} bars  ({time.time()-t0:.0f}s)", flush=True)
    print(f"  cols: {list(m1.columns)}")


if __name__ == "__main__":
    main()
