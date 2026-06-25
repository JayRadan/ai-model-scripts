"""
Oracle EXIT LAB — test profit-locking exits on real Oracle trade R-paths.
Centerpiece: the RATCHET ("goes in profit, never comes back").
Entries = M15-TFK PRO-gated setups (deployed entry universe). For each trade we
build the intrabar R-path (FAV/ADV/CLOSE in R) and compare exit RULES on the
SAME trades. Conservative intrabar order: adverse extreme assumed hit first.
"""
from __future__ import annotations
import sys, glob, time
from pathlib import Path
import numpy as np, pandas as pd
from numba import njit
ROOT = Path("/home/jay/Desktop/new-model-zigzag")
sys.path.insert(0, str(ROOT / "products/hermes_xau"))
from tfk import compute_tfk

MAXH = 60; MIN_HOLD = 3; SL_HARD = 6.0
SWING = ROOT / "data/swing_v5_xauusd.csv"
SETUPS = sorted(glob.glob(str(ROOT / "data/setups_*_v72l.csv")))

def m15_dir_at_bars(df):
    s = df.set_index("time")
    m15 = pd.DataFrame({"open":s["open"].resample("15min").first(),"high":s["high"].resample("15min").max(),
        "low":s["low"].resample("15min").min(),"close":s["close"].resample("15min").last(),
        "tick_volume":s["tick_volume"].resample("15min").sum()}).dropna(subset=["close"]).reset_index()
    m15["d"]=compute_tfk(m15,flip_bars=5,color_confirm=8)["committed_dir"].to_numpy()
    al=pd.merge_asof(df[["time"]],m15[["time","d"]].assign(time=m15["time"]+pd.Timedelta("15min")),on="time")
    return al["d"].fillna(0).to_numpy(np.int64)

def metrics(r):
    r=np.asarray(r); w=r[r>0]; l=r[r<=0]; eq=np.cumsum(r)
    return (len(r),(r>0).mean()*100, w.sum()/max(-l.sum(),1e-9), r.sum(),
            float((np.maximum.accumulate(eq)-eq).max()))

# ---- exit rules (numba). Each takes FAV/ADV/CLO (N,MAXH), returns pnl per trade ----
@njit(cache=True)
def ex_baseline(FAV,ADV,CLO,sp):   # 6R SL + maxhold, no profit lock
    N=FAV.shape[0]; out=np.empty(N)
    for i in range(N):
        r=CLO[i,MAXH-1]
        for k in range(MAXH):
            if k>=1 and ADV[i,k]<=-SL_HARD: r=-SL_HARD; break
            if not np.isfinite(CLO[i,k]): r=CLO[i,k-1] if k>0 else 0.0; break
        out[i]=r-sp
    return out

@njit(cache=True)
def ex_ratchet(FAV,ADV,CLO,sp,step):  # CLOSE-triggered floor, ratchets up, 6R intrabar broker SL
    N=FAV.shape[0]; out=np.empty(N)
    for i in range(N):
        peak=0.0; floor=-SL_HARD; r=CLO[i,MAXH-1]; done=False
        for k in range(MAXH):
            if not np.isfinite(CLO[i,k]):
                if not done: r=CLO[i,k-1] if k>0 else 0.0
                break
            if k>=1 and ADV[i,k]<=-SL_HARD: r=-SL_HARD; done=True; break  # broker stop (intrabar)
            if CLO[i,k]>peak: peak=CLO[i,k]
            lvl=(np.floor(peak/step)-1.0)*step
            if lvl>floor: floor=lvl
            if k>=1 and CLO[i,k]<=floor: r=floor; done=True; break        # lock (close-triggered)
        out[i]=r-sp
    return out

@njit(cache=True)
def ex_trail(FAV,ADV,CLO,sp,act,give):  # close-triggered give-back trail
    N=FAV.shape[0]; out=np.empty(N)
    for i in range(N):
        peak=0.0; r=CLO[i,MAXH-1]; done=False
        for k in range(MAXH):
            if not np.isfinite(CLO[i,k]):
                if not done: r=CLO[i,k-1] if k>0 else 0.0
                break
            if k>=1 and ADV[i,k]<=-SL_HARD: r=-SL_HARD; done=True; break
            if CLO[i,k]>peak: peak=CLO[i,k]
            if peak>=act and CLO[i,k]<=peak-give: r=CLO[i,k]; done=True; break
        out[i]=r-sp
    return out

@njit(cache=True)
def ex_scaleout(FAV,ADV,CLO,sp,tp,act,give):  # half at +tp (limit), trail rest on close
    N=FAV.shape[0]; out=np.empty(N)
    for i in range(N):
        peak=0.0; locked=0.0; half=False; r=CLO[i,MAXH-1]; done=False
        for k in range(MAXH):
            if not np.isfinite(CLO[i,k]):
                if not done: r=CLO[i,k-1] if k>0 else 0.0
                break
            if k>=1 and ADV[i,k]<=-SL_HARD: r=-SL_HARD; done=True; break
            if (not half) and FAV[i,k]>=tp: locked=0.5*tp; half=True
            if CLO[i,k]>peak: peak=CLO[i,k]
            if half and peak>=act and CLO[i,k]<=peak-give: r=CLO[i,k]; done=True; break
        out[i]= (locked + 0.5*r) - sp if half else (r-sp)
    return out

@njit(cache=True)
def ex_hardtp(FAV,ADV,CLO,sp,tp):  # full exit at +tp
    N=FAV.shape[0]; out=np.empty(N)
    for i in range(N):
        r=CLO[i,MAXH-1]; done=False
        for k in range(MAXH):
            if not np.isfinite(CLO[i,k]):
                if not done: r=CLO[i,k-1] if k>0 else 0.0
                break
            if k>=1 and ADV[i,k]<=-SL_HARD: r=-SL_HARD; done=True; break
            if FAV[i,k]>=tp: r=tp; done=True; break
        out[i]=r-sp
    return out

def main():
    t0=time.time()
    sw=pd.read_csv(SWING,parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    C=sw["close"].to_numpy(float);H=sw["high"].to_numpy(float);L=sw["low"].to_numpy(float)
    pc=np.concatenate([[C[0]],C[:-1]]); tr=np.maximum(H-L,np.maximum(np.abs(H-pc),np.abs(L-pc)))
    atr=pd.Series(tr).rolling(14,min_periods=14).mean().to_numpy()
    md=m15_dir_at_bars(sw); n=len(sw); sp=0.30/np.nanmedian(atr)
    setups=pd.concat([pd.read_csv(f,parse_dates=["time"]) for f in SETUPS],ignore_index=True)
    setups=setups.dropna(subset=["time","direction"]).sort_values("time")
    st=setups["time"].values.astype("datetime64[ns]"); tw=sw["time"].values.astype("datetime64[ns]")
    idx=np.searchsorted(tw,st); idx=np.minimum(idx,n-1)
    exact=tw[idx]==st; d=setups["direction"].to_numpy()
    keep=exact&(atr[idx]>0)&np.isfinite(atr[idx])&(d==md[idx])&(idx<n-MAXH-1)  # PRO gate: dir==m15
    ei=idx[keep]; dd=d[keep].astype(np.int64); N=len(ei)
    print(f"swing {n:,} | PRO-gated Oracle trades: {N:,}")
    # build R-path matrices
    FAV=np.full((N,MAXH),np.nan);ADV=np.full((N,MAXH),np.nan);CLO=np.full((N,MAXH),np.nan)
    for r in range(N):
        e=ei[r];dr=dd[r];ep=C[e];ea=atr[e]
        for k in range(1,MAXH+1):
            b=e+k
            if b>=n: break
            FAV[r,k-1]=dr*((H[b] if dr==1 else L[b])-ep)/ea
            ADV[r,k-1]=dr*((L[b] if dr==1 else H[b])-ep)/ea
            CLO[r,k-1]=dr*(C[b]-ep)/ea
    mfe=np.nanmax(FAV,axis=1)
    base=ex_baseline(FAV,ADV,CLO,sp)
    win=mfe>=2  # trades that became real winners
    print(f"  MFE: reach +2R={100*(mfe>=2).mean():.0f}%  +4R={100*(mfe>=4).mean():.0f}%  +6R={100*(mfe>=6).mean():.0f}%  median peak={np.nanmedian(mfe):.2f}R")
    print(f"  of +2R winners: peak avg={mfe[win].mean():.2f}R  baseline EXIT avg={base[win].mean():+.2f}R  → gives back {mfe[win].mean()-base[win].mean():.2f}R/trade\n")
    runs=[("BASELINE 6RSL+maxhold",base),
          ("RATCHET step=1",ex_ratchet(FAV,ADV,CLO,sp,1.0)),
          ("RATCHET step=2",ex_ratchet(FAV,ADV,CLO,sp,2.0)),
          ("TRAIL act2 give2",ex_trail(FAV,ADV,CLO,sp,2.0,2.0)),
          ("TRAIL act3 give2",ex_trail(FAV,ADV,CLO,sp,3.0,2.0)),
          ("TRAIL act3 give3",ex_trail(FAV,ADV,CLO,sp,3.0,3.0)),
          ("TRAIL act4 give2",ex_trail(FAV,ADV,CLO,sp,4.0,2.0)),
          ("SCALEOUT tp4 act4 give2",ex_scaleout(FAV,ADV,CLO,sp,4.0,4.0,2.0)),
          ("HARD TP 6R",ex_hardtp(FAV,ADV,CLO,sp,6.0)),
          ("HARD TP 8R",ex_hardtp(FAV,ADV,CLO,sp,8.0)),
          ("HARD TP 4R",ex_hardtp(FAV,ADV,CLO,sp,4.0))]
    # time-ordered holdout: first 70% train / last 30% unseen (by entry order)
    order=np.argsort(ei); cut=int(0.70*N); hold=order[cut:]
    htime=sw["time"].values[ei[hold[0]]]
    print(f"  {'strategy':<24}|{'  ALL HISTORY (in-sample)':>26}|{'  HOLDOUT last 30% (OOS)':>26}")
    print(f"  {'':<24}|{'PF':>7}{'sumR':>10}{'DD':>8}|{'PF':>7}{'sumR':>10}{'DD':>8}")
    print("  "+"-"*76)
    for name,r in runs:
        _,_,pf,sr,dd=metrics(r)
        _,_,pfh,srh,ddh=metrics(r[hold])
        print(f"  {name:<24}|{pf:>7.2f}{sr:>+10.0f}{dd:>8.0f}|{pfh:>7.2f}{srh:>+10.0f}{ddh:>8.0f}")
    print(f"\n  holdout starts {str(htime)[:10]}  ({len(hold):,} trades)\n  TOTAL {time.time()-t0:.0f}s")

if __name__=="__main__": main()
