"""Backfill 2018 -> 2024-12 Dukascopy ticks for XAU + BTC (resumable, parallel).
Writes daily parquets to data/ticks/{xau,btc}/YYYY-MM-DD.parquet (skips existing).
Gives XAU/BTC the same 8-year tick depth DOW already has."""
import os, time, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import dukascopy_python
ROOT="/home/jay/Desktop/new-model-zigzag"
START=datetime(2018,1,1,tzinfo=timezone.utc)
END=datetime(2024,12,1,tzinfo=timezone.utc)   # existing data covers 2024-12 onward
WORKERS=8
JOBS={"xau":("XAU/USD",True),"btc":("BTC/USD",False)}  # (symbol, skip_weekends)

def fetch_day(sym, tick_dir, day):
    fn=os.path.join(tick_dir, day.strftime("%Y-%m-%d")+".parquet")
    if os.path.exists(fn): return ("skip",day,0)
    try:
        df=dukascopy_python.fetch(instrument=sym,interval=dukascopy_python.INTERVAL_TICK,
            offer_side=dukascopy_python.OFFER_SIDE_BID,start=day,end=day+timedelta(days=1))
        if df is None or len(df)==0: return ("empty",day,0)
        df.to_parquet(fn,compression="snappy"); return ("ok",day,os.path.getsize(fn))
    except Exception as e: return ("err",day,str(e)[:60])

def run(inst):
    sym,skipwe=JOBS[inst]; tick_dir=f"{ROOT}/data/ticks/{inst}"; os.makedirs(tick_dir,exist_ok=True)
    days=[]; cur=START
    while cur<END:
        if not (skipwe and cur.weekday()>=5): days.append(cur)
        cur+=timedelta(days=1)
    print(f"[{inst}] {len(days)} days to consider 2018->2024-12 ({sym})",flush=True)
    t0=time.time(); ok=skip=empty=err=0; tot=0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs=[ex.submit(fetch_day,sym,tick_dir,d) for d in days]
        for i,f in enumerate(as_completed(futs)):
            s,day,sz=f.result()
            if s=="ok": ok+=1; tot+=sz
            elif s=="skip": skip+=1
            elif s=="empty": empty+=1
            else: err+=1
            if (i+1)%200==0:
                print(f"[{inst}] {i+1}/{len(days)}  ok={ok} skip={skip} empty={empty} err={err}  {tot/1e6:.0f}MB  {time.time()-t0:.0f}s",flush=True)
    print(f"[{inst}] DONE ok={ok} skip={skip} empty={empty} err={err}  {tot/1e6:.0f}MB  {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    for inst in (sys.argv[1:] or ["xau","btc"]): run(inst)
    print("ALL BACKFILL DONE",flush=True)
