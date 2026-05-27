"""Download Dow Jones tick data from Dukascopy 2018-01-01 → 2026-05-27.

Per-day parquet, parallel fetching, resumable. Uses tz-aware UTC.
Output: data/ticks/dji/YYYY-MM-DD.parquet
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_IDX_AMERICA_E_D_J_IND

PROJECT = "/home/jay/Desktop/new-model-zigzag"
TICK_DIR = f"{PROJECT}/data/ticks/dji"
START = datetime(2018, 1, 1, tzinfo=timezone.utc)
END   = datetime(2026, 5, 27, tzinfo=timezone.utc)
WORKERS = 6


def fetch_day(day):
    fn = os.path.join(TICK_DIR, day.strftime("%Y-%m-%d") + ".parquet")
    if os.path.exists(fn):
        return ("skip", day, os.path.getsize(fn))
    try:
        df = dukascopy_python.fetch(
            instrument=INSTRUMENT_IDX_AMERICA_E_D_J_IND,
            interval=dukascopy_python.INTERVAL_TICK,
            offer_side=dukascopy_python.OFFER_SIDE_BID,
            start=day, end=day + timedelta(days=1),
        )
        if df is None or len(df) == 0:
            return ("empty", day, 0)
        df.to_parquet(fn, compression="snappy")
        return ("ok", day, os.path.getsize(fn))
    except Exception as e:
        return ("err", day, str(e))


def main():
    os.makedirs(TICK_DIR, exist_ok=True)
    days = []
    cur = START
    while cur < END:
        # Skip weekends (no Dow ticks)
        if cur.weekday() < 5:
            days.append(cur)
        cur += timedelta(days=1)
    print(f"=== DJI ticks → {TICK_DIR}  ({len(days)} weekdays) ===", flush=True)
    t0 = time.time(); ok = sk = em = er = 0; total_bytes = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(fetch_day, d): d for d in days}
        for i, fut in enumerate(as_completed(futures), 1):
            status, day, info = fut.result()
            if status == "ok": ok += 1; total_bytes += info
            elif status == "skip": sk += 1; total_bytes += info
            elif status == "empty": em += 1
            else: er += 1
            if i % 50 == 0 or i == len(days):
                pct = i * 100 / len(days)
                print(f"  [{i}/{len(days)}] {pct:5.1f}%  ok={ok} skip={sk} empty={em} err={er}  "
                      f"size={total_bytes/1e6:.0f}MB  ({time.time()-t0:.0f}s)", flush=True)
    print(f"  DONE: {ok} fetched, {sk} skipped, {em} empty, {er} errors. "
          f"{total_bytes/1e9:.2f} GB. {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
