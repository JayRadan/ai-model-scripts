"""v88: Break-even stop layered on top of current trail.

Targets the 30 XAU / 29 BTC trades that peak >= 1R then reverse to a loss.

Policy: at every bar, evaluate in priority:
  1. Hard SL @ -4R                        (always)
  2. Break-even stop: if peak >= BE_TRIGGER and mtm <= BE_LEVEL → exit  (new)
  3. Trail: if peak >= 3.0 and mtm <= peak*0.4 → exit  (current prod)
  4. 60-bar max hold

Sweep BE_TRIGGER ∈ {0.75, 1.0, 1.25, 1.5, 2.0}
      BE_LEVEL   ∈ {-0.5, 0.0, +0.25, +0.5}
"""
import os, time
import numpy as np, pandas as pd

PROJECT="/home/jay/Desktop/new-model-zigzag"
DATA=f"{PROJECT}/data"
MAX_HOLD=60; SL_HARD=-4.0
TRAIL_ACT=3.0; TRAIL_GB=0.60   # current production

def load_market(swing_csv):
    sw=pd.read_csv(swing_csv,parse_dates=["time"])
    sw=sw.sort_values("time").drop_duplicates('time',keep='last').reset_index(drop=True)
    n=len(sw); C=sw["close"].values.astype(np.float64)
    H=sw["high"].values; Lo=sw["low"].values
    tr=np.concatenate([[H[0]-Lo[0]],np.maximum.reduce([H[1:]-Lo[1:],np.abs(H[1:]-C[:-1]),np.abs(Lo[1:]-C[:-1])])])
    atr=pd.Series(tr).rolling(14,min_periods=14).mean().values
    t2i={pd.Timestamp(t):i for i,t in enumerate(sw["time"].values)}
    return n,C,atr,t2i

def simulate(trades,t2i,n,C,atr,be_trigger,be_level):
    """be_trigger=None disables BE; be_level is mtm at which BE exits."""
    pnls=[]; reasons=[]
    for _,trade in trades.iterrows():
        tm=trade["time"]
        if tm not in t2i: continue
        ei=t2i[tm]; d=int(trade["direction"]); ep=C[ei]; ea=atr[ei]
        if not np.isfinite(ea) or ea<=0: continue
        peak=0.0; exit_bar=None; reason="max_hold"
        max_k=min(MAX_HOLD,n-ei-1)
        for k in range(1,max_k+1):
            bar=ei+k; mtm=d*(C[bar]-ep)/ea
            if mtm<=SL_HARD: exit_bar=bar; reason="hard_sl"; break
            if mtm>peak: peak=mtm
            # Break-even check (new)
            if be_trigger is not None and peak>=be_trigger and mtm<=be_level:
                exit_bar=bar; reason="be_stop"; break
            # Original trail
            if peak>=TRAIL_ACT and mtm<=peak*(1.0-TRAIL_GB):
                exit_bar=bar; reason="trail"; break
        if exit_bar is None:
            exit_bar=min(ei+max_k,n-1); reason="max_hold"
        pnls.append(d*(C[exit_bar]-ep)/ea); reasons.append(reason)
    return pnls,reasons

def pf(p):
    s=sum(p); w=[x for x in p if x>0]; l=[x for x in p if x<=0]
    pf=sum(w)/max(-sum(l),1e-9) if l else 99.0
    return pf,s,len(w)/max(len(p),1)

def maxdd(p):
    eq=np.cumsum(p); peak=np.maximum.accumulate(eq)
    return float((peak-eq).max())

def evaluate(name,swing_csv,trades_csv):
    print("\n"+"="*72); print(f"  {name}"); print("="*72)
    n,C,atr,t2i=load_market(swing_csv)
    trades=pd.read_csv(trades_csv,parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    split=trades["time"].quantile(0.70)
    tst=trades[trades["time"]>=split].reset_index(drop=True)
    print(f"  Unseen: {len(tst)} trades")

    # Baseline = production
    pn,r=simulate(tst,t2i,n,C,atr,None,None)
    pf_b,s_b,wr_b=pf(pn); dd_b=maxdd(pn)
    n_be_b=r.count('be_stop'); n_tr_b=r.count('trail'); n_sl_b=r.count('hard_sl'); n_mx_b=r.count('max_hold')
    print(f"\n  PROD (no BE):  PF={pf_b:.2f}  Total={s_b:+.0f}R  WR={wr_b*100:.1f}%  DD={dd_b:.0f}R")
    print(f"                 exits: hard_sl={n_sl_b} trail={n_tr_b} max={n_mx_b}")

    print(f"\n  BREAK-EVEN STOP SWEEP (layered on top of current trail):")
    print(f"  {'trig':>5s} {'lvl':>6s} {'PF':>5s} {'TotalR':>8s} {'WR%':>5s} {'DD':>5s}  {'sl':>3s} {'be':>3s} {'tr':>3s} {'mx':>3s}  vs prod")
    print(f"  {'-'*5} {'-'*6} {'-'*5} {'-'*8} {'-'*5} {'-'*5}  {'-'*3} {'-'*3} {'-'*3} {'-'*3}  -------")

    results=[]
    for trig in [0.75,1.0,1.25,1.5,2.0]:
        for lvl in [-0.5,0.0,0.25,0.5]:
            pn,r=simulate(tst,t2i,n,C,atr,trig,lvl)
            pf_,s,wr=pf(pn); dd=maxdd(pn)
            sl=r.count('hard_sl'); be=r.count('be_stop'); tr=r.count('trail'); mx=r.count('max_hold')
            delta=s-s_b
            marker = " ★" if (s>=s_b and dd<=dd_b) else (" + DD" if dd<dd_b*0.9 else (" + R" if s>s_b else ""))
            print(f"  {trig:5.2f} {lvl:+6.2f} {pf_:5.2f} {s:+8.0f} {wr*100:5.1f} {dd:4.0f}R  {sl:>3d} {be:>3d} {tr:>3d} {mx:>3d}  {delta:+5.0f}R{marker}")
            results.append((trig,lvl,pf_,s,wr,dd))

    print(f"\n  TOP 5 BY TOTAL R:")
    for r in sorted(results,key=lambda x:-x[3])[:5]:
        trig,lvl,pf_,s,wr,dd=r
        print(f"    BE: trigger={trig:.2f} level={lvl:+.2f}  PF={pf_:.2f}  Total={s:+.0f}R ({s-s_b:+.0f})  DD={dd:.0f}R ({dd-dd_b:+.0f})")

if __name__=="__main__":
    evaluate("Oracle XAU — break-even stop sweep",
             f"{DATA}/swing_v5_xauusd.csv",
             f"{PROJECT}/experiments/v84_rl_entry/v84_rl_trades.csv")
    evaluate("Oracle BTC — break-even stop sweep",
             f"{DATA}/swing_v5_btc.csv",
             f"{PROJECT}/experiments/v84_rl_entry/btc_rl_trades.csv")
