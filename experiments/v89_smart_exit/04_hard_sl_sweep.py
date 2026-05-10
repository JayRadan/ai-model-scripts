"""Test hard-SL sweep: does tighter SL improve PF/DD with current prod policy?

Production: hard SL @ -4R, trail (act=3R, gb=60%), v88 reverse-setup, 60-bar max.
Sweep SL_HARD ∈ {-1.5, -2.0, -2.5, -3.0, -3.5, -4.0} keeping everything else fixed.

Tighter SL trade-off:
  + smaller loss per losing trade
  - more whipsaw exits on trades that would have recovered
  - alters the trail's MTM-from-SL math (we keep trail in absolute R,
    so trail behavior at peak=3R unchanged)
"""
import os, time, glob as _glob, pickle
import numpy as np, pandas as pd
from importlib.util import spec_from_file_location, module_from_spec

PROJECT="/home/jay/Desktop/new-model-zigzag"
DATA=f"{PROJECT}/data"
spec=spec_from_file_location("v89", os.path.join(os.path.dirname(os.path.abspath(__file__)),"01_optimal_stopping.py"))
v89=module_from_spec(spec); spec.loader.exec_module(v89)
V72L=v89.V72L

def load_setups(setups_glob):
    parts=[]
    for f in sorted(_glob.glob(setups_glob)):
        cid_str=os.path.basename(f).split('_')[1]
        try: old_cid=int(cid_str)
        except: continue
        df=pd.read_csv(f,parse_dates=["time"]); df['old_cid']=old_cid
        parts.append(df)
    s=pd.concat(parts,ignore_index=True)
    return s.sort_values(['time','direction']).drop_duplicates(['time','direction'],keep='first').reset_index(drop=True)

def simulate(trades,t2i,n,C,atr,sl_hard,
             trail_act=3.0,trail_gb=0.60,max_hold=60,
             setup_lkp=None,setup_Q=None,sw_times=None,v88_thr=0.05):
    pnls=[]; reasons={}
    for ti,(_,trade) in enumerate(trades.iterrows()):
        tm=trade["time"]
        if tm not in t2i: continue
        ei=t2i[tm]; d=int(trade["direction"]); ep=C[ei]; ea=atr[ei]
        if not np.isfinite(ea) or ea<=0: continue
        peak=0.0; max_k=min(max_hold,n-ei-1); fired=False
        for k in range(1,max_k+1):
            bar=ei+k
            R=d*(C[bar]-ep)/ea
            if R<=sl_hard:
                pnls.append(sl_hard)   # exit AT the SL price (clamp to SL)
                reasons['hard_sl']=reasons.get('hard_sl',0)+1; fired=True; break
            if R>peak: peak=R

            # v88 reverse-setup
            if setup_lkp is not None:
                bt=pd.Timestamp(sw_times[bar])
                if bt in setup_lkp:
                    opp=-d
                    if opp in setup_lkp[bt]:
                        idx=setup_lkp[bt][opp]
                        if setup_Q[idx]>v88_thr:
                            pnls.append(R); reasons['v88']=reasons.get('v88',0)+1; fired=True; break

            # Trail
            if peak>=trail_act and R<=peak*(1.0-trail_gb):
                pnls.append(R); reasons['trail']=reasons.get('trail',0)+1; fired=True; break
        if not fired:
            last=min(ei+max_k,n-1)
            pnls.append(d*(C[last]-ep)/ea); reasons['max_hold']=reasons.get('max_hold',0)+1
    return pnls, reasons

def pf(p):
    s=sum(p); w=[x for x in p if x>0]; l=[x for x in p if x<=0]
    pf=sum(w)/max(-sum(l),1e-9) if l else 99.0
    return pf,s,len(w)/max(len(p),1)

def maxdd(p):
    eq=np.cumsum(p); peak=np.maximum.accumulate(eq)
    return float((peak-eq).max())

def evaluate(name,swing_csv,setups_glob,trades_csv,bundle_path):
    print("\n"+"="*72); print(f"  {name}"); print("="*72)
    t0=time.time()
    n,C,atr,t2i,ctx=v89.load_market(swing_csv,setups_glob)
    sw_times=pd.read_csv(swing_csv,parse_dates=["time"]).sort_values("time").drop_duplicates('time',keep='last')["time"].values
    trades=pd.read_csv(trades_csv,parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    split=trades["time"].quantile(0.70)
    tst=trades[trades["time"]>=split].reset_index(drop=True)

    # Setup lookup for v88 reverse-setup
    setups=load_setups(setups_glob)
    bundle=pickle.load(open(bundle_path,"rb"))
    setup_Q=np.full(len(setups),-9.0,dtype=np.float32)
    for cid,m in bundle['q_entry'].items():
        mask=(setups['old_cid']==cid).values
        if mask.sum()<1: continue
        Xc=setups.loc[mask,V72L].fillna(0).values
        setup_Q[mask]=m.predict(Xc)
    setup_lkp={}
    times_arr=setups.time.values; dirs_arr=setups.direction.values
    for i in range(len(setups)):
        tt=pd.Timestamp(times_arr[i]); d=int(dirs_arr[i])
        if tt not in setup_lkp: setup_lkp[tt]={}
        setup_lkp[tt][d]=i

    print(f"  Test: {len(tst)} trades on unseen 30%")
    print(f"\n  HARD-SL SWEEP (trail act=3R/gb=60%, v88 reverse-setup, 60-bar max)")
    print(f"  {'SL':>5s} {'PF':>5s} {'TotalR':>8s} {'WR%':>5s} {'DD':>5s} {'sl':>4s} {'v88':>4s} {'tr':>3s} {'mx':>3s}")
    print(f"  {'-'*5} {'-'*5} {'-'*8} {'-'*5} {'-'*5} {'-'*4} {'-'*4} {'-'*3} {'-'*3}")
    rows=[]
    for sl in [-1.5,-2.0,-2.5,-3.0,-3.5,-4.0]:
        pn,r=simulate(tst,t2i,n,C,atr,sl,
                      setup_lkp=setup_lkp,setup_Q=setup_Q,sw_times=sw_times)
        pf_,sR,wr_=pf(pn); dd_=maxdd(pn)
        sl_n=r.get('hard_sl',0); v88_n=r.get('v88',0); tr_n=r.get('trail',0); mx_n=r.get('max_hold',0)
        marker = " ← prod" if abs(sl - -4.0) < 1e-6 else ""
        rows.append((sl,pf_,sR,wr_,dd_))
        print(f"  {sl:5.1f} {pf_:5.2f} {sR:+8.0f} {wr_*100:5.1f} {dd_:4.0f}R {sl_n:4d} {v88_n:4d} {tr_n:3d} {mx_n:3d}{marker}")

    print(f"\n  TOP 3 BY TOTAL R:")
    for r in sorted(rows,key=lambda x:-x[2])[:3]:
        sl,pf_,sR,wr_,dd_=r
        print(f"    SL={sl:.1f}R: PF={pf_:.2f}  Total={sR:+.0f}R  DD={dd_:.0f}R")
    print(f"\n  TOP 3 BY LOWEST DD:")
    for r in sorted(rows,key=lambda x:x[4])[:3]:
        sl,pf_,sR,wr_,dd_=r
        print(f"    SL={sl:.1f}R: PF={pf_:.2f}  Total={sR:+.0f}R  DD={dd_:.0f}R")
    print(f"\n  Done in {time.time()-t0:.0f}s")

if __name__=="__main__":
    evaluate("Oracle XAU — hard-SL sweep",
             f"{DATA}/swing_v5_xauusd.csv",
             f"{DATA}/setups_*_v72l.csv",
             f"{PROJECT}/experiments/v84_rl_entry/v84_rl_trades.csv",
             f"{PROJECT}/products/models/oracle_xau_validated.pkl")
    evaluate("Oracle BTC — hard-SL sweep",
             f"{DATA}/swing_v5_btc.csv",
             f"{DATA}/setups_*_v72l_btc.csv",
             f"{PROJECT}/experiments/v84_rl_entry/btc_rl_trades.csv",
             f"{PROJECT}/products/models/oracle_btc_validated.pkl")
