"""Download XAU/USD M1 OHLCV from Dukascopy → data/m1_xau_full.parquet.

Mirror of products/hermes_xau/scripts/01_download_m1_dukascopy.py so this
experiment is self-contained. Resumable, ~20-day chunks, tz-aware UTC.

NOTE: Dukascopy's freeserv endpoint must be reachable. In some sandboxed
execution environments it is blocked (HTTP 403); run this where the network
policy allows api/freeserv.dukascopy.com.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "m1_xau_full.parquet"
OUT.parent.mkdir(parents=True, exist_ok=True)

START = datetime(2018, 1, 1, tzinfo=timezone.utc)
END = datetime(2026, 5, 1, tzinfo=timezone.utc)
CHUNK_DAYS = 20


def main():
    import dukascopy_python
    from dukascopy_python.instruments import INSTRUMENT_FX_METALS_XAU_USD

    if OUT.exists():
        existing = pd.read_parquet(OUT)
        existing["time"] = pd.to_datetime(existing["time"]).dt.tz_localize(None)
        last_time = existing["time"].max()
        print(f"  resuming — existing {len(existing):,} bars, last={last_time}", flush=True)
        cur = last_time.to_pydatetime().replace(tzinfo=timezone.utc) + timedelta(minutes=1)
        all_parts = [existing]
    else:
        all_parts = []
        cur = START

    print(f"  fetching XAU M1 from {cur} → {END} in {CHUNK_DAYS}-day chunks", flush=True)
    t0 = time.time()
    fetched = 0
    chunks = 0
    while cur < END:
        end_chunk = min(cur + timedelta(days=CHUNK_DAYS), END)
        try:
            df = dukascopy_python.fetch(
                instrument=INSTRUMENT_FX_METALS_XAU_USD,
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
            print(f"  {cur.date()} → {end_chunk.date()}: {len(df):,}  (cum {fetched:,}, {time.time()-t0:.0f}s)", flush=True)
        else:
            print(f"  {cur.date()} → {end_chunk.date()}: 0", flush=True)
        cur = end_chunk
        chunks += 1
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
    if "volume" in df.columns:
        df["tick_volume"] = df["volume"]
    df["spread"] = 0
    keep = ["time", "open", "high", "low", "close", "spread", "tick_volume"]
    df = df[[c for c in keep if c in df.columns]]
    df.to_parquet(OUT)
    print(f"\n  TOTAL {len(df):,}  range: {df.time.min()} → {df.time.max()}", flush=True)
    print(f"  saved {OUT}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
