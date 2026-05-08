"""Download M1 OHLCV from Dukascopy for XAU and BTC.

Range matches the M5 download in v89_quantum_flow_tiebreaker but uses
14-day chunks to stay under the 30000-row limit (M1 = 5x M5 density).

Output:
  /tmp/duk/xau_m1.csv
  /tmp/duk/btc_m1.csv
"""
import dukascopy_python
from datetime import datetime, timedelta
import pandas as pd, os, time

OUT="/tmp/duk"; os.makedirs(OUT,exist_ok=True)
START=datetime(2024,11,1)
END  =datetime(2026,5,3)
CHUNK=14  # days

def download(sym,out_name):
    print(f"\n=== {sym} → {out_name} ===",flush=True)
    parts=[]; cur=START; t0=time.time()
    while cur<END:
        end=min(cur+timedelta(days=CHUNK),END)
        try:
            df=dukascopy_python.fetch(
                instrument=sym,
                interval=dukascopy_python.INTERVAL_MIN_1,
                offer_side=dukascopy_python.OFFER_SIDE_BID,
                start=cur, end=end)
            if df is not None and len(df):
                parts.append(df)
                if len(parts)%5==0:
                    print(f"  {cur.date()}: cumulative {sum(len(p) for p in parts):,} bars ({time.time()-t0:.0f}s)",flush=True)
        except Exception as e:
            print(f"  {cur.date()}→{end.date()}: ERROR {e}",flush=True)
        cur=end
    if not parts: print("  no data"); return
    df=pd.concat(parts).reset_index().rename(columns={"timestamp":"time"})
    df["time"]=pd.to_datetime(df["time"]).dt.tz_localize(None)
    df["spread"]=0; df["tick_volume"]=df["volume"]
    df=df[["time","open","high","low","close","spread","tick_volume"]]
    df=df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    out=os.path.join(OUT,out_name)
    df.to_csv(out,index=False)
    print(f"  → {out}  {len(df):,} bars  {df.time.min()} → {df.time.max()}  ({time.time()-t0:.0f}s)",flush=True)

if __name__=="__main__":
    download("XAU/USD","xau_m1.csv")
    download("BTC/USD","btc_m1.csv")
