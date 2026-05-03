"""Download XAU + BTC M5 with real volume from Dukascopy.
Chunks in 90-day batches to stay under the 30000-row limit."""
import dukascopy_python
from datetime import datetime, timedelta
import pandas as pd
import os
import sys

OUT = "/tmp/duk"
os.makedirs(OUT, exist_ok=True)

START = datetime(2024, 11, 1)   # ~1 month before holdout for warmup
END   = datetime(2026, 5, 3)
CHUNK = 90  # days per fetch

def download(symbol_duk: str, out_name: str):
    print(f"\n=== {symbol_duk} → {out_name} ===", flush=True)
    parts = []
    cur = START
    while cur < END:
        end = min(cur + timedelta(days=CHUNK), END)
        try:
            df = dukascopy_python.fetch(
                instrument=symbol_duk,
                interval=dukascopy_python.INTERVAL_MIN_5,
                offer_side=dukascopy_python.OFFER_SIDE_BID,
                start=cur, end=end,
            )
            if df is not None and len(df):
                parts.append(df)
                print(f"  {cur.date()} → {end.date()}: {len(df)} bars", flush=True)
            else:
                print(f"  {cur.date()} → {end.date()}: 0 bars", flush=True)
        except Exception as e:
            print(f"  {cur.date()} → {end.date()}: ERROR {e}", flush=True)
        cur = end
    if not parts:
        print(f"  no data for {symbol_duk}"); return
    df = pd.concat(parts).reset_index()
    df = df.rename(columns={"timestamp": "time"})
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
    # Standard schema: time, open, high, low, close, spread, tick_volume
    df["spread"] = 0          # Dukascopy bid-only doesn't give spread
    df["tick_volume"] = df["volume"]
    df = df[["time","open","high","low","close","spread","tick_volume"]]
    out = os.path.join(OUT, out_name)
    df.to_csv(out, index=False)
    print(f"  → {out}  ({len(df)} bars, {df.time.min()} → {df.time.max()})", flush=True)


if __name__ == "__main__":
    download("XAU/USD", "xau_m5.csv")
    download("BTC/USD", "btc_m5.csv")
