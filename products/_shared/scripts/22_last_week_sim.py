"""Run all 6 products over Jun 1-7 2026, filter to that window, summarise."""
import sys, subprocess, importlib.util, pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np, pandas as pd
from numba import njit

ROOT = Path("/home/jay/Desktop/new-model-zigzag")
SERVER = Path("/home/jay/Desktop/my-agents-and-website/commercial/server")
sys.path.insert(0, str(ROOT / "experiments/kalman_color_flip"))
sys.path.insert(0, str(ROOT / "products/hermes_xau"))
from kalman import compute_kalman, bars_in_regime_array
from tfk import compute_tfk
_spec = importlib.util.spec_from_file_location("ofm1", ROOT / "products/_shared/m1_with_orderflow.py")
_of = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_of)
add_standard_features = _of.add_standard_features
FLOW = list(_of.FLOW_FEATS)
import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_IDX_AMERICA_E_D_J_IND

SPREAD=0.30; SL=6.0; TRAIL=2.0; MAXH=300; BE_R=0.5
MAX_CONC=4; SWITCH=0.5; COOLDOWN=5
STRONG=0.8; RMULT=1.0
NEAR=0.50; COUNTER=1.5

# Window: Jun 1 00:00 → Jun 7 23:59 UTC
WINDOW_START = pd.Timestamp("2026-06-01 00:00")
WINDOW_END   = pd.Timestamp("2026-06-08 00:00")

def fetch(symbol, days=10, dji=False):
    end = datetime.now(timezone.utc); start = end - timedelta(days=days)
    if dji:
        df = dukascopy_python.fetch(instrument=INSTRUMENT_IDX_AMERICA_E_D_J_IND,
                                     interval=dukascopy_python.INTERVAL_MIN_1,
                                     offer_side=dukascopy_python.OFFER_SIDE_BID,
                                     start=start, end=end)
    else:
        df = dukascopy_python.fetch(instrument=symbol, interval=dukascopy_python.INTERVAL_MIN_1,
                                     offer_side=dukascopy_python.OFFER_SIDE_BID,
                                     start=start, end=end)
    df = df.reset_index().rename(columns={"timestamp":"time"})
    df["time"] = pd.to_datetime(df["time"]).dt.tz_convert("UTC").dt.tz_localize(None)
    df = df.sort_values("time").reset_index(drop=True)
    if "tick_volume" not in df.columns:
        vc = "volume" if "volume" in df.columns else [c for c in df.columns if "vol" in c.lower()][0]
        df["tick_volume"] = df[vc]
    return df

def streak_up(v):
    n=len(v); o=np.zeros(n,np.int32); c=0
    for i in range(1,n):
        c=c+1 if v[i]>v[i-1] else 0; o[i]=c
    return o

def add_lookahead_htf(df):
    df=df.copy()
    c=df["close"]; h=df["high"]; l=df["low"]
    prev=c.shift(1).fillna(c.iloc[0])
    tr=pd.concat([(h-l),(h-prev).abs(),(l-prev).abs()],axis=1).max(axis=1)
    df["atr14"]=tr.rolling(14,min_periods=14).mean()
    delta=c.diff()
    up=delta.clip(lower=0).rolling(14,min_periods=14).mean()
    dn=(-delta.clip(upper=0)).rolling(14,min_periods=14).mean()
    rs=up/dn.replace(0,np.nan); df["rsi14"]=100-100/(1+rs)
    for nm in (20,50,100,200):
        ema=c.ewm(span=nm,adjust=False).mean(); df[f"dist_ema{nm}"]=(c-ema)/df["atr14"]
    for nm in (5,10,20): df[f"slope{nm}"]=(c-c.shift(nm))/df["atr14"]
    df["atr_ratio"]=df["atr14"]/df["atr14"].rolling(50,min_periods=50).mean()
    df_ts=df.set_index("time")
    for tf_name,tf in [("m5","5min"),("m15","15min"),("h1","60min")]:
        g=df_ts[["high","low","close"]].resample(tf).agg({"high":"max","low":"min","close":"last"}).dropna()
        ch=g["close"]; dlt=ch.diff()
        up=dlt.clip(lower=0).rolling(14,min_periods=14).mean()
        dn=(-dlt.clip(upper=0)).rolling(14,min_periods=14).mean()
        rs=up/dn.replace(0,np.nan)
        g[f"{tf_name}_rsi14"]=100-100/(1+rs)
        g[f"{tf_name}_slope5"]=ch-ch.shift(5)
        g[f"{tf_name}_ema50_dist"]=ch-ch.ewm(span=50,adjust=False).mean()
        keep=[c_ for c_ in g.columns if c_ not in ("high","low","close")]
        out=g[keep].reindex(df_ts.index,method="ffill")
        for col in out.columns: df[col]=out[col].to_numpy()
    return df


def hermes_candidates(df, bundle_pkl, qthr, dist_cap=None):
    tfk_df=compute_tfk(df)
    fdf=add_lookahead_htf(df)
    for c in ["force","velocity","x_est","regime_w","trend_raw","trend","committed_dir","confirmed_dir","tfk_line"]:
        fdf[c]=tfk_df[c].to_numpy()
    cdir=tfk_df["committed_dir"].to_numpy(np.int64); tline=tfk_df["tfk_line"].to_numpy(float)
    O=df["open"].to_numpy(float); H=df["high"].to_numpy(float); L=df["low"].to_numpy(float); C=df["close"].to_numpy(float)
    atr=fdf["atr14"].to_numpy(float); n=len(fdf); sp=SPREAD/np.nanmedian(atr)
    dist_signed=np.where(atr>0,(C-tline)/atr,0.0)
    fdf["dist_at_signal"]=dist_signed; fdf["dist_abs"]=np.abs(dist_signed)
    fdf["regime_age"]=bars_in_regime_array(cdir)
    fdf["bar_range_atr"]=(H-L)/np.maximum(atr,1e-9)
    for f in FLOW:
        if f not in fdf.columns: fdf[f]=0.0
    counter=dist_signed*cdir
    is_pb=fdf["dist_abs"].to_numpy()<=NEAR
    is_ct=counter<=-COUNTER
    ok=np.isfinite(atr)&(atr>0)&(cdir!=0); ok[:250]=False; ok[-(MAXH+1):]=False
    mask=ok&(is_pb|is_ct)
    if dist_cap is not None:
        stretched=is_ct & ~is_pb & (fdf["dist_abs"].to_numpy()>dist_cap)
        mask=mask & ~stretched
    idxs=np.where(mask)[0]; dirs=cdir[idxs].copy()
    bundle=pickle.load(open(bundle_pkl,"rb"))
    M=bundle["q_model"]; feat=bundle["feat_cols"]
    Xall=fdf[feat].fillna(0).to_numpy(np.float32)
    q=M.predict(Xall[idxs])
    return {"idxs":idxs,"dirs":dirs,"q":q,"n":n,"sp":sp,"O":O,"H":H,"L":L,"C":C,"atr":atr,"df":df}


def atlas_candidates(df, bundle_pkl, qthr):
    tfk_df=compute_tfk(df); kf=compute_kalman(df,r_mult=RMULT)
    fdf=add_standard_features(kf)
    for c in ["force","velocity","x_est","regime_w","trend_raw","trend","committed_dir"]: fdf[c]=tfk_df[c].to_numpy()
    fdf["tfk_line"]=tfk_df["tfk_line"].to_numpy()
    O=df["open"].to_numpy(float); H=df["high"].to_numpy(float); L=df["low"].to_numpy(float); C=df["close"].to_numpy(float)
    atr=fdf["atr14"].to_numpy(float)
    kdir=fdf["kf_dir"].to_numpy(np.int64); kline=fdf["kf_p"].to_numpy(float)
    cdir=tfk_df["committed_dir"].to_numpy(np.int64); tline=tfk_df["tfk_line"].to_numpy(float)
    vel=fdf["f_velRaw"].to_numpy(float)
    kage=bars_in_regime_array(kdir); fdf["kf_regime_age"]=kage
    fdf["bar_range_atr"]=(H-L)/np.maximum(atr,1e-9)
    fdf["dist_kf"]=np.where(atr>0,(C-kline)/atr,0.0); fdf["dist_tfk"]=np.where(atr>0,(C-tline)/atr,0.0)
    fdf["vel_up_streak"]=streak_up(vel)
    body=C-O; body_atr=np.where(atr>0,np.abs(body)/atr,0.0); fdf["body_atr"]=body_atr
    sb=(body<0)&(body_atr>=STRONG); su=(body>0)&(body_atr>=STRONG)
    fdf["strong_bear_prev"]=np.concatenate([[False],sb[:-1]])
    fdf["strong_bull_prev"]=np.concatenate([[False],su[:-1]])
    n=len(fdf); sp=SPREAD/np.nanmedian(atr)
    pbk=np.concatenate([[False],C[:-1]<kline[:-1]]); pak=np.concatenate([[False],C[:-1]>kline[:-1]])
    pbt=np.concatenate([[False],C[:-1]<tline[:-1]]); pat=np.concatenate([[False],C[:-1]>tline[:-1]])
    g=C>O; r=C<O
    ok=np.isfinite(atr)&(atr>0); ok[:250]=False; ok[-(MAXH+1):]=False
    buy=ok&(cdir==1)&(kdir==-1)&fdf["strong_bear_prev"].to_numpy()&pbk&pbt&g&(kage>=3)
    sell=ok&(cdir==-1)&(kdir==1)&fdf["strong_bull_prev"].to_numpy()&pak&pat&r&(kage>=3)
    mask=buy|sell
    idxs=np.where(mask)[0]; dirs=np.where(cdir[idxs]==1,1,-1).astype(np.int64)
    bundle=pickle.load(open(bundle_pkl,"rb"))
    M=bundle["q_model"]; feat=bundle["feat_cols"]
    for f in FLOW:
        if f not in fdf.columns: fdf[f]=0.0
    Xall=fdf[feat].fillna(0).to_numpy(np.float32)
    q=M.predict(Xall[idxs])
    return {"idxs":idxs,"dirs":dirs,"q":q,"n":n,"sp":sp,"O":O,"H":H,"L":L,"C":C,"atr":atr,"df":df}


def multipos_sim(data, q_thr):
    idxs=data["idxs"]; dirs=data["dirs"]; q=data["q"]
    n=data["n"]; sp=data["sp"]; O=data["O"]; H=data["H"]; L=data["L"]; C=data["C"]; atr=data["atr"]; df=data["df"]
    info={int(idxs[k]):(int(dirs[k]),float(q[k])) for k in range(len(idxs)) if q[k]>=q_thr}
    if not info: return []
    active=[]; ex=[]; last={-1:-10**9,1:-10**9}
    b0=min(info.keys()); b1=min(max(info.keys())+MAXH+1,n)
    times=df["time"].to_numpy()
    for i in range(b0,b1):
        still=[]
        for t in active:
            if i<=t["entry_idx"]: still.append(t); continue
            if i>min(t["entry_idx"]+MAXH,n-1):
                cp=C[min(t["entry_idx"]+MAXH,n-1)]
                t["pnl_R"]=float(t["dir"]*(cp-t["ep"])/t["a"]-sp); ex.append(t); continue
            d=t["dir"]; ep=t["ep"]; a=t["a"]; fav=d*(C[i]-ep)
            if fav>t["mf"]: t["mf"]=fav
            hit=False
            if t["sl_r"]==0:
                if d==1 and L[i]<=ep: t["pnl_R"]=-sp; hit=True
                elif d==-1 and H[i]>=ep: t["pnl_R"]=-sp; hit=True
            else:
                dist=abs(t["sl_r"])*a
                if d==1 and (ep-L[i])>=dist: t["pnl_R"]=float(t["sl_r"]-sp); hit=True
                elif d==-1 and (H[i]-ep)>=dist: t["pnl_R"]=float(t["sl_r"]-sp); hit=True
            if hit: ex.append(t); continue
            td=TRAIL*a
            if t["mf"]>=td and (t["mf"]-fav)>=td:
                t["pnl_R"]=float((t["mf"]-td)/a-sp); ex.append(t); continue
            still.append(t)
        active=still
        if i not in info: continue
        d_,q_=info[i]
        if i-last[d_]<COOLDOWN: continue
        for t in active:
            if t["sl_r"]==0: continue
            cur=t["dir"]*(C[i]-t["ep"])/t["a"]
            if cur>=BE_R: t["sl_r"]=0
        ei=i+1
        if ei>=n or not(np.isfinite(atr[i]) and atr[i]>0): continue
        if len(active)>=MAX_CONC:
            worst=min(active,key=lambda x:x["q"])
            if q_>=worst["q"]+SWITCH:
                worst["pnl_R"]=float(worst["dir"]*(C[i]-worst["ep"])/worst["a"]-sp); ex.append(worst); active.remove(worst)
            else: continue
        active.append({"entry_idx":ei,"entry_time":pd.Timestamp(times[ei]),"dir":d_,"ep":float(O[ei]),
                       "a":float(atr[i]),"sl_r":float(-SL),"mf":0.0,"q":float(q_),"pnl_R":None})
        last[d_]=i
    for t in active:
        eb=min(t["entry_idx"]+MAXH,n-1)
        t["pnl_R"]=float(t["dir"]*(C[eb]-t["ep"])/t["a"]-sp); ex.append(t)
    return ex


# Fetch data once
print("Fetching XAU/USD 10d...")
xau = fetch("XAU/USD", days=10)
print("Fetching BTC/USD 10d...")
btc = fetch("BTC/USD", days=10)
print("Fetching DJI 10d...")
dji = fetch("XAU/USD", days=10, dji=True)
print(f"  XAU bars: {len(xau)}  BTC: {len(btc)}  DJI: {len(dji)}")

CONFIGS = [
    ("hermes_xau", hermes_candidates, xau, SERVER/"decision_engine/models/hermes_xau_validated.pkl", 4.0, 3.0, 1.5),
    ("hermes_btc", hermes_candidates, btc, SERVER/"decision_engine/models/hermes_btc_validated.pkl", 4.0, None, 2.3),
    ("hermes_dji", hermes_candidates, dji, SERVER/"decision_engine/models/hermes_dji_validated.pkl", 3.0, None, 0.75),
    ("atlas_xau",  atlas_candidates,  xau, SERVER/"decision_engine/models/atlas_xau_validated.pkl",  2.0, None, 1.5),
    ("atlas_btc",  atlas_candidates,  btc, SERVER/"decision_engine/models/atlas_btc_validated.pkl",  3.0, None, 2.3),
    ("atlas_dji",  atlas_candidates,  dji, SERVER/"decision_engine/models/atlas_dji_validated.pkl",  3.0, None, 0.75),
]

print("\n" + "="*90)
print(f"  LAST WEEK: {WINDOW_START.date()} → {WINDOW_END.date() - pd.Timedelta('1D')}")
print("="*90)
print(f"{'Product':<13} {'trades':>7} {'WR%':>6} {'PF':>7} {'sumR':>9} {'DD':>7} {'$@0.01':>10} {'$@0.10':>10}")
print("-"*90)
tot_R=0; tot_d01=0; tot_d10=0; tot_t=0
for name, fn, df, pkl, qthr, dist_cap, rate in CONFIGS:
    if name=="hermes_xau":
        data = fn(df, pkl, qthr, dist_cap=dist_cap)
    else:
        data = fn(df, pkl, qthr) if fn is not hermes_candidates else fn(df, pkl, qthr, dist_cap=None)
    ex = multipos_sim(data, qthr)
    week = [t for t in ex if WINDOW_START <= t["entry_time"] < WINDOW_END]
    if not week:
        print(f"{name:<13} {'0':>7}"); continue
    rs = np.array([t["pnl_R"] for t in week]); w = int((rs>0).sum())
    pf = float(rs[rs>0].sum()/max(-rs[rs<=0].sum(),1e-9)) if (rs<=0).any() else float("inf")
    eq=np.cumsum(rs); dd=float((np.maximum.accumulate(eq)-eq).max())
    d01=rs.sum()*rate; d10=d01*10
    print(f"{name:<13} {len(week):>7} {w/len(rs)*100:>5.1f}% {pf:>7.2f} {rs.sum():>+9.2f} {dd:>7.2f} {d01:>+10.2f} {d10:>+10.2f}")
    tot_R+=rs.sum(); tot_d01+=d01; tot_d10+=d10; tot_t+=len(week)
print("-"*90)
print(f"{'TOTAL':<13} {tot_t:>7} {'':>6} {'':>7} {tot_R:>+9.2f} {'':>7} {tot_d01:>+10.2f} {tot_d10:>+10.2f}")
