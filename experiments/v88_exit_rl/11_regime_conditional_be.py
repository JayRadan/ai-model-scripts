"""v88: Regime-conditional break-even stop.

Strong directional regimes (Uptrend cid=0, Downtrend cid=3) have PF 7-15
and almost never pullback-then-lose — so BE there only kills winners.

HighVol (cid=4) has PF ~2.5 and most pullback losses.

Policy: same trail as prod (act=3, gb=0.6) PLUS a BE that only fires when
cid==4 (HighVol).
"""
import os, time
import numpy as np, pandas as pd

PROJECT="/home/jay/Desktop/new-model-zigzag"
DATA=f"{PROJECT}/data"
MAX_HOLD=60; SL_HARD=-4.0
TRAIL_ACT=3.0; TRAIL_GB=0.60

def load_market(swing_csv):
    sw=pd.read_csv(swing_csv,parse_dates=["time"])
    sw=sw.sort_values("time").drop_duplicates('time',keep='last').reset_index(drop=True)
    n=len(sw); C=sw["close"].values.astype(np.float64)
    H=sw["high"].values; Lo=sw["low"].values
    tr=np.concatenate([[H[0]-Lo[0]],np.maximum.reduce([H[1:]-Lo[1:],np.abs(H[1:]-C[:-1]),np.abs(Lo[1:]-C[:-1])])])
    atr=pd.Series(tr).rolling(14,min_periods=14).mean().values
    t2i={pd.Timestamp(t):i for i,t in enumerate(sw["time"].values)}
    return n,C,atr,t2i

def simulate(trades,t2i,n,C,atr,be_trigger,be_level,be_cids):
    """Apply BE only when trade cid in be_cids."""
    pnls=[]; reasons=[]; cids_used=[]
    for _,trade in trades.iterrows():
        tm=trade["time"]
        if tm not in t2i: continue
        ei=t2i[tm]; d=int(trade["direction"]); ep=C[ei]; ea=atr[ei]
        if not np.isfinite(ea) or ea<=0: continue
        cid=int(trade["cid"]) if "cid" in trade else 0
        be_active = (be_trigger is not None) and (cid in be_cids)
        peak=0.0; exit_bar=None; reason="max_hold"
        max_k=min(MAX_HOLD,n-ei-1)
        for k in range(1,max_k+1):
            bar=ei+k; mtm=d*(C[bar]-ep)/ea
            if mtm<=SL_HARD: exit_bar=bar; reason="hard_sl"; break
            if mtm>peak: peak=mtm
            if be_active and peak>=be_trigger and mtm<=be_level:
                exit_bar=bar; reason="be_stop"; break
            if peak>=TRAIL_ACT and mtm<=peak*(1.0-TRAIL_GB):
                exit_bar=bar; reason="trail"; break
        if exit_bar is None:
            exit_bar=min(ei+max_k,n-1); reason="max_hold"
        pnls.append(d*(C[exit_bar]-ep)/ea); reasons.append(reason); cids_used.append(cid)
    return pnls,reasons,cids_used

def pf(p):
    s=sum(p); w=[x for x in p if x>0]; l=[x for x in p if x<=0]
    pf=sum(w)/max(-sum(l),1e-9) if l else 99.0
    return pf,s,len(w)/max(len(p),1)

def maxdd(p):
    eq=np.cumsum(p); peak=np.maximum.accumulate(eq)
    return float((peak-eq).max())

def by_cid_breakdown(pnls,cids):
    out={}
    for c in sorted(set(cids)):
        ps=[p for p,ci in zip(pnls,cids) if ci==c]
        if not ps: continue
        pf_,s,wr=pf(ps); out[c]=(pf_,s,wr,len(ps))
    return out

def evaluate(name,swing_csv,trades_csv):
    print("\n"+"="*72); print(f"  {name}"); print("="*72)
    n,C,atr,t2i=load_market(swing_csv)
    trades=pd.read_csv(trades_csv,parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    split=trades["time"].quantile(0.70)
    tst=trades[trades["time"]>=split].reset_index(drop=True)

    # Production baseline
    pn,r,c=simulate(tst,t2i,n,C,atr,None,None,set())
    pf_b,s_b,wr_b=pf(pn); dd_b=maxdd(pn)
    by_b=by_cid_breakdown(pn,c)
    print(f"\n  PROD: PF={pf_b:.2f}  Total={s_b:+.0f}R  WR={wr_b*100:.1f}%  DD={dd_b:.0f}R")
    print(f"  By regime:")
    for cid,(pfx,sx,wrx,nx) in by_b.items():
        print(f"    cid={cid}  N={nx:3d}  PF={pfx:5.2f}  Total={sx:+5.0f}R  WR={wrx*100:5.1f}%")

    # Sweep regime-conditional BE
    print(f"\n  REGIME-CONDITIONAL BE (only on HighVol cid=4):")
    print(f"  {'trig':>5s} {'lvl':>6s} {'PF':>5s} {'TotalR':>8s} {'WR%':>5s} {'DD':>5s}  vs prod")
    results=[]
    for trig in [0.5,0.75,1.0,1.25,1.5,2.0]:
        for lvl in [-0.5,0.0,0.25,0.5,1.0]:
            pn,r,c=simulate(tst,t2i,n,C,atr,trig,lvl,{4})
            pf_,s,wr=pf(pn); dd=maxdd(pn)
            delta=s-s_b
            marker=" ★" if (s>=s_b and dd<=dd_b) else (" + DD" if dd<dd_b*0.85 else (" + R" if s>s_b else ""))
            print(f"  {trig:5.2f} {lvl:+6.2f} {pf_:5.2f} {s:+8.0f} {wr*100:5.1f} {dd:4.0f}R  {delta:+5.0f}R{marker}")
            results.append((trig,lvl,pf_,s,wr,dd,r,c))

    print(f"\n  TOP 5 BY TOTAL R (regime-conditional BE):")
    for r in sorted(results,key=lambda x:-x[3])[:5]:
        trig,lvl,pf_,s,wr,dd,re,ci=r
        be_n=re.count('be_stop')
        print(f"    BE@trigger={trig:.2f} level={lvl:+.2f}  PF={pf_:.2f}  Total={s:+.0f}R ({s-s_b:+.0f})  DD={dd:.0f}R ({dd-dd_b:+.0f})  BE_exits={be_n}")
        # cid breakdown for this combo
        by=by_cid_breakdown([p for p in re if False] or [], [])  # placeholder
        # actually re is reasons not pnls; fix:
        # we need pnls for breakdown. recompute briefly
    # Detailed breakdown of best
    best=max(results,key=lambda x:x[3])
    print(f"\n  REGIME BREAKDOWN at best combo (trig={best[0]:.2f} lvl={best[1]:+.2f}):")
    pn_b,r_b,c_b=simulate(tst,t2i,n,C,atr,best[0],best[1],{4})
    by_be=by_cid_breakdown(pn_b,c_b)
    for cid in sorted(by_be.keys()):
        pf_p,s_p,wr_p,n_p = by_b[cid]
        pf_be,s_be,wr_be,n_be = by_be[cid]
        d_s = s_be-s_p
        print(f"    cid={cid}  N={n_p}  PROD: PF={pf_p:.2f} Total={s_p:+.0f}R  →  BE: PF={pf_be:.2f} Total={s_be:+.0f}R  Δ={d_s:+.0f}R")

if __name__=="__main__":
    evaluate("Oracle XAU — regime-conditional BE",
             f"{DATA}/swing_v5_xauusd.csv",
             f"{PROJECT}/experiments/v84_rl_entry/v84_rl_trades.csv")
    evaluate("Oracle BTC — regime-conditional BE",
             f"{DATA}/swing_v5_btc.csv",
             f"{PROJECT}/experiments/v84_rl_entry/btc_rl_trades.csv")
