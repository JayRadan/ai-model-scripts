"""Download Dukascopy tick data (XAU + BTC) for the v84 RL trade window.

Strategy:
  - 1-day chunks, parquet output, parallel ThreadPoolExecutor
  - Skip existing files (resumable)
  - Range: 2024-12-01 → 2026-05-03 (518 days)
  - Output: data/ticks/{xau,btc}/YYYY-MM-DD.parquet

After this, 08_tick_features.py builds entry-time tick aggregates for
each v84 RL trade and feeds them into the adversarial entry filter.
"""
import dukascopy_python
from datetime import datetime, timedelta
import pandas as pd, os, time
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT="/home/jay/Desktop/new-model-zigzag"
TICK_DIR=f"{PROJECT}/data/ticks"
START=datetime(2024,12,1)
END  =datetime(2026,5,3)
WORKERS=4

def fetch_day(sym,out_dir,day):
    fn=os.path.join(out_dir,day.strftime("%Y-%m-%d")+".parquet")
    if os.path.exists(fn):
        return ("skip",day,os.path.getsize(fn))
    try:
        df=dukascopy_python.fetch(
            instrument=sym,
            interval=dukascopy_python.INTERVAL_TICK,
            offer_side=dukascopy_python.OFFER_SIDE_BID,
            start=day, end=day+timedelta(days=1))
        if df is None or len(df)==0:
            return ("empty",day,0)
        df.to_parquet(fn,compression='snappy')
        return ("ok",day,os.path.getsize(fn))
    except Exception as e:
        return ("err",day,str(e))

def download_all(sym,out_name):
    out_dir=os.path.join(TICK_DIR,out_name); os.makedirs(out_dir,exist_ok=True)
    days=[]
    cur=START
    while cur<END:
        days.append(cur); cur+=timedelta(days=1)
    print(f"\n=== {sym} → {out_dir}  ({len(days)} days) ===",flush=True)
    t0=time.time(); ok=0; sk=0; em=0; er=0; total_bytes=0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures={ex.submit(fetch_day,sym,out_dir,d):d for d in days}
        for i,fut in enumerate(as_completed(futures),1):
            status,day,info = fut.result()
            if status=="ok":   ok+=1; total_bytes+=info
            elif status=="skip": sk+=1; total_bytes+=info
            elif status=="empty": em+=1
            else: er+=1
            if i%25==0 or i==len(days):
                pct=i*100/len(days)
                print(f"  [{i}/{len(days)}] {pct:5.1f}%  ok={ok} skip={sk} empty={em} err={er}  size={total_bytes/1e6:.0f}MB  ({time.time()-t0:.0f}s)",flush=True)
    print(f"  DONE: {ok} fetched, {sk} skipped, {em} empty, {er} errors. {total_bytes/1e9:.2f} GB. {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    download_all("XAU/USD","xau")
    download_all("BTC/USD","btc")
