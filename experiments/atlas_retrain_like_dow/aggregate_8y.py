"""Rebuild FULL 8-year M1 orderflow parquets for XAU + BTC from all ticks.
Waits for the backfill to finish first. Writes to data/m1_{inst}_orderflow_8y.parquet
(does NOT overwrite the deployed-training parquets). Exact same feature recipe as
products/atlas_xau/scripts/02_aggregate_orderflow_from_ticks.py."""
import time, sys
from glob import glob
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path("/home/jay/Desktop/new-model-zigzag")

def aggregate_one_day(path):
    df = pd.read_parquet(path)
    if len(df) == 0: return pd.DataFrame()
    mid = (df["bidPrice"] + df["askPrice"]).to_numpy()/2.0
    bv = df["bidVolume"].to_numpy(); av = df["askVolume"].to_numpy(); tot = bv+av
    mp = np.concatenate([[mid[0]], mid[:-1]])
    sign = np.where(mid>mp, 1.0, np.where(mid<mp, -1.0, 0.0)); sv = sign*tot
    spread = (df["askPrice"]-df["bidPrice"]).to_numpy()
    df2 = pd.DataFrame({"mid":mid,"tot_v":tot,"bid_v":bv,"ask_v":av,"signed_vol":sv,"spread":spread}, index=df.index)
    g = df2.resample("1min")
    return pd.DataFrame({"open":g["mid"].first(),"high":g["mid"].max(),"low":g["mid"].min(),"close":g["mid"].last(),
        "tick_volume":g["tot_v"].sum(),"n_ticks":g["mid"].count(),"signed_flow":g["signed_vol"].sum(),
        "abs_volume":g["tot_v"].sum(),"bid_v_sum":g["bid_v"].sum(),"ask_v_sum":g["ask_v"].sum(),
        "median_spread":g["spread"].median()}).dropna(subset=["open"])

def rebuild(inst):
    tick_dir = ROOT/"data"/"ticks"/inst; OUT = ROOT/f"data/m1_{inst}_orderflow_8y.parquet"
    files = sorted(glob(str(tick_dir/"*.parquet")))
    print(f"[{inst}] aggregating {len(files)} tick days ...", flush=True); t0=time.time()
    frames=[]
    for i,p in enumerate(files):
        try:
            m=aggregate_one_day(p)
            if len(m): frames.append(m)
        except Exception as e: print(f"  skip {Path(p).name}: {e}")
        if (i+1)%300==0: print(f"  [{inst}] {i+1}/{len(files)} ({time.time()-t0:.0f}s)", flush=True)
    m1=pd.concat(frames).sort_index(); m1=m1[~m1.index.duplicated(keep="first")]
    m1["imbalance_ratio"]=m1["signed_flow"]/m1["abs_volume"].replace(0,np.nan)
    m1["bid_ask_vol_ratio"]=m1["bid_v_sum"]/(m1["bid_v_sum"]+m1["ask_v_sum"]).replace(0,np.nan)
    m1["vpin_proxy"]=(m1["signed_flow"].abs()/m1["abs_volume"].replace(0,np.nan))
    for w in (5,15,60):
        m1[f"cum_signed_{w}"]=m1["signed_flow"].rolling(w,min_periods=1).sum()
        m1[f"cum_signed_abs_{w}"]=m1["signed_flow"].abs().rolling(w,min_periods=1).sum()
        m1[f"flow_persistence_{w}"]=m1[f"cum_signed_{w}"]/m1[f"cum_signed_abs_{w}"].replace(0,np.nan)
    m1["spread_vol_50"]=m1["median_spread"].rolling(50,min_periods=10).std()
    m1["tick_intensity_50"]=m1["n_ticks"]/m1["n_ticks"].rolling(50,min_periods=10).mean()
    m1=m1.reset_index().rename(columns={"timestamp":"time"}); m1["time"]=pd.to_datetime(m1["time"]).dt.tz_localize(None)
    m1.to_parquet(OUT)
    print(f"[{inst}] DONE → {OUT.name}  {len(m1):,} bars  {m1.time.iloc[0]} -> {m1.time.iloc[-1]}  ({time.time()-t0:.0f}s)", flush=True)

import subprocess
print("waiting for backfill to finish ...", flush=True)
while subprocess.run(["pgrep","-f","backfill_ticks"],capture_output=True).returncode==0:
    time.sleep(15)
print("backfill done — aggregating", flush=True)
for inst in (sys.argv[1:] or ["xau","btc"]): rebuild(inst)
print("ALL AGGREGATION DONE", flush=True)
