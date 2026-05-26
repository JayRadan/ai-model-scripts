"""Download BTC M1 OHLCV from Dukascopy.

Matches the BTC tick coverage already on disk (data/ticks/btc/ spans
2024-12-01 → 2026-05-02). Starts 1 month earlier so TFK warmup is clean.

Chunks ≈ 20 days to stay under 30k-row API limit on M1.
Saves to data/m1_btc_full.parquet (resumable — appends new chunks).
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUT = ROOT / "data" / "m1_btc_full.parquet"

START = datetime(2024, 11, 1)
END = datetime(2026, 5, 3)
CHUNK_DAYS = 20

INSTRUMENT = "BTC/USD"


def main():
    import dukascopy_python

    # Load existing if present
    if OUT.exists():
        existing = pd.read_parquet(OUT)
        existing["time"] = pd.to_datetime(existing["time"]).dt.tz_localize(None)
        last_time = existing["time"].max()
        print(f"  resuming — existing has {len(existing):,} bars, last={last_time}", flush=True)
        cur = last_time.to_pydatetime() + timedelta(minutes=1)
        all_parts = [existing]
    else:
        all_parts = []
        cur = START

    print(f"  fetching {INSTRUMENT} M1 from {cur} → {END} in {CHUNK_DAYS}-day chunks", flush=True)
    t0 = time.time()
    fetched = 0
    chunks = 0
    while cur < END:
        end_chunk = min(cur + timedelta(days=CHUNK_DAYS), END)
        try:
            df = dukascopy_python.fetch(
                instrument=INSTRUMENT,
                interval=dukascopy_python.INTERVAL_MIN_1,
                offer_side=dukascopy_python.OFFER_SIDE_BID,
                start=cur, end=end_chunk,
            )
        except Exception as e:
            print(f"  ERR {cur.date()} → {end_chunk.date()}: {e}", flush=True)
            cur = end_chunk
            continue
        if df is not None and len(df):
            d2 = df.reset_index().rename(columns={"timestamp": "time"})
            d2["time"] = pd.to_datetime(d2["time"]).dt.tz_localize(None)
            all_parts.append(d2)
            fetched += len(df)
            print(f"  {cur.date()} → {end_chunk.date()}: {len(df):,} bars  (cum {fetched:,}, {time.time()-t0:.0f}s)", flush=True)
        else:
            print(f"  {cur.date()} → {end_chunk.date()}: 0 bars", flush=True)
        cur = end_chunk
        chunks += 1
        # save every 20 chunks (~1 year)
        if chunks % 20 == 0 and all_parts:
            tmp = pd.concat(all_parts).drop_duplicates("time").sort_values("time").reset_index(drop=True)
            tmp["time"] = pd.to_datetime(tmp["time"]).dt.tz_localize(None)
            tmp.to_parquet(OUT)
            all_parts = [tmp]
            print(f"  → checkpointed {OUT.name} ({len(tmp):,} bars)", flush=True)

    if not all_parts:
        print("  no data fetched"); return

    df = pd.concat(all_parts).drop_duplicates("time").sort_values("time").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
    # Standard schema
    if "volume" in df.columns:
        df["tick_volume"] = df["volume"]
    df["spread"] = 0
    keep = ["time", "open", "high", "low", "close", "spread", "tick_volume"]
    df = df[keep]
    df.to_parquet(OUT)
    print(f"\n  TOTAL {len(df):,} bars  range: {df.time.min()} → {df.time.max()}", flush=True)
    print(f"  saved {OUT}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
