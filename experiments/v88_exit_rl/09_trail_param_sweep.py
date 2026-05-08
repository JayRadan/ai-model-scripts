"""v88: Trail parameter sweep — fix the "profit wiped out by pullback" issue.

Production XAU trail: activation=3.0R, giveback=60% of peak.
A trade peaking at +2.5R has NO protection — it can reverse all the way
to -4R SL with zero trail intervention.

Step 1: DIAGNOSE — how many unseen losers had peak >= 1R, 1.5R, 2R, 2.5R?
        These are the trades currently leaking profit on pullbacks.
Step 2: SWEEP — try tighter (activation, giveback) combos, simulating each
        on the unseen 30%. Compare PF / TotalR / WR / DD vs current.
Step 3: REPORT — best (activation, giveback) for XAU and BTC.

The simulator applies (in priority order) at each in-trade bar:
  1. hard SL @ -4R
  2. trail exit if peak >= activation AND mtm <= peak*(1-giveback)
  3. 60-bar max hold
"""
import os, time, glob
import numpy as np, pandas as pd

PROJECT="/home/jay/Desktop/new-model-zigzag"
DATA=f"{PROJECT}/data"
MAX_HOLD=60; SL_HARD=-4.0

def load_market(swing_csv):
    sw=pd.read_csv(swing_csv,parse_dates=["time"])
    sw=sw.sort_values("time").drop_duplicates('time',keep='last').reset_index(drop=True)
    n=len(sw); C=sw["close"].values.astype(np.float64)
    H=sw["high"].values; Lo=sw["low"].values
    tr=np.concatenate([[H[0]-Lo[0]],np.maximum.reduce([H[1:]-Lo[1:],np.abs(H[1:]-C[:-1]),np.abs(Lo[1:]-C[:-1])])])
    atr=pd.Series(tr).rolling(14,min_periods=14).mean().values
    t2i={pd.Timestamp(t):i for i,t in enumerate(sw["time"].values)}
    return n,C,atr,t2i

def simulate(trades,t2i,n,C,atr,trail_activation_r,trail_giveback_pct):
    """Walk each trade bar-by-bar applying trail policy."""
    pnls=[]; exit_reasons=[]
    for _,trade in trades.iterrows():
        tm=trade["time"]
        if tm not in t2i: continue
        ei=t2i[tm]; d=int(trade["direction"]); ep=C[ei]; ea=atr[ei]
        if not np.isfinite(ea) or ea<=0: continue
        peak=0.0; exit_bar=None; reason="max_hold"
        max_k=min(MAX_HOLD,n-ei-1)
        for k in range(1,max_k+1):
            bar=ei+k
            mtm=d*(C[bar]-ep)/ea
            if mtm<=SL_HARD:
                exit_bar=bar; reason="hard_sl"; break
            if mtm>peak: peak=mtm
            # Trail check (only after activation)
            if peak>=trail_activation_r:
                if mtm <= peak*(1.0-trail_giveback_pct):
                    exit_bar=bar; reason="trail"; break
        if exit_bar is None:
            exit_bar=min(ei+max_k,n-1); reason="max_hold"
        pnl=d*(C[exit_bar]-ep)/ea
        pnls.append(pnl); exit_reasons.append(reason)
    return pnls,exit_reasons

def diagnose_pullbacks(trades,t2i,n,C,atr):
    """How many trades hit peak >= X but ended in a loss?"""
    buckets={1.0:0,1.5:0,2.0:0,2.5:0,3.0:0}
    bucket_loss={k:0.0 for k in buckets}
    n_total=0; n_loser=0
    for _,trade in trades.iterrows():
        tm=trade["time"]
        if tm not in t2i: continue
        ei=t2i[tm]; d=int(trade["direction"]); ep=C[ei]; ea=atr[ei]
        if not np.isfinite(ea) or ea<=0: continue
        n_total+=1
        peak=0.0; exit_bar=None
        max_k=min(MAX_HOLD,n-ei-1)
        # Walk to determine baseline final pnl using current prod logic (act=3.0, gb=0.6)
        for k in range(1,max_k+1):
            bar=ei+k
            mtm=d*(C[bar]-ep)/ea
            if mtm<=SL_HARD: exit_bar=bar; break
            if mtm>peak: peak=mtm
            if peak>=3.0 and mtm<=peak*0.4:
                exit_bar=bar; break
        if exit_bar is None: exit_bar=min(ei+max_k,n-1)
        final=d*(C[exit_bar]-ep)/ea
        if final<=-1.0:
            n_loser+=1
            for thr in buckets:
                if peak>=thr:
                    buckets[thr]+=1
                    bucket_loss[thr]+=final
    return n_total,n_loser,buckets,bucket_loss

def pf_stats(p):
    s=sum(p); w=[x for x in p if x>0]; l=[x for x in p if x<=0]
    pf=sum(w)/max(-sum(l),1e-9) if l else 99.0
    return pf,s,len(w)/max(len(p),1),len(p)

def maxdd_R(p):
    eq=np.cumsum(p)
    peak=np.maximum.accumulate(eq)
    return float((peak-eq).max())

def evaluate(name,swing_csv,trades_csv,prod_act,prod_gb):
    print("\n"+"="*72); print(f"  {name}"); print("="*72)
    t0=time.time()
    n,C,atr,t2i=load_market(swing_csv)
    trades=pd.read_csv(trades_csv,parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    split=trades["time"].quantile(0.70)
    tst=trades[trades["time"]>=split].reset_index(drop=True)
    print(f"  Unseen: {len(tst)} trades  (split @ {split})")

    # ── Diagnose: how many losers had meaningful peaks? ──
    print(f"\n  DIAGNOSIS — losers with peak >= X (using current prod policy):")
    n_total,n_loser,buckets,bucket_loss=diagnose_pullbacks(tst,t2i,n,C,atr)
    print(f"  Total trades: {n_total}, losers (final<=-1R): {n_loser} ({n_loser*100/max(n_total,1):.1f}%)")
    for thr in sorted(buckets.keys()):
        cnt=buckets[thr]; loss=bucket_loss[thr]
        print(f"    Peaked >= {thr:.1f}R then lost: {cnt:3d} trades  ({loss:+.0f}R total)")

    # ── Sweep ──
    print(f"\n  TRAIL POLICY SWEEP:")
    print(f"  {'act':>4s} {'gb':>5s} {'PF':>5s} {'TotalR':>8s} {'WR%':>5s} {'MaxDD':>7s} {'%trail':>7s} {'%maxh':>7s}  notes")
    print(f"  {'-'*4} {'-'*5} {'-'*5} {'-'*8} {'-'*5} {'-'*7} {'-'*7} {'-'*7}  -----")
    pn,reasons=simulate(tst,t2i,n,C,atr,prod_act,prod_gb)
    pf_b,s_b,wr_b,N_b=pf_stats(pn); dd_b=maxdd_R(pn)
    pct_trail_b=100*reasons.count('trail')/max(len(reasons),1)
    pct_max_b=100*reasons.count('max_hold')/max(len(reasons),1)
    print(f"  {prod_act:4.1f} {prod_gb:5.2f} {pf_b:5.2f} {s_b:+8.0f} {wr_b*100:5.1f} {dd_b:6.0f}R {pct_trail_b:6.1f}% {pct_max_b:6.1f}%  ← CURRENT PROD")

    results=[]
    for act in [0.5,0.75,1.0,1.25,1.5,2.0,2.5]:
        for gb in [0.30,0.40,0.50,0.60]:
            pn,reasons=simulate(tst,t2i,n,C,atr,act,gb)
            pf,s,wr,N=pf_stats(pn); dd=maxdd_R(pn)
            pct_trail=100*reasons.count('trail')/max(len(reasons),1)
            pct_max  =100*reasons.count('max_hold')/max(len(reasons),1)
            results.append((act,gb,pf,s,wr,dd,pct_trail,pct_max))
            marker=""
            if s>s_b and dd<dd_b: marker=" ★ better PF+DD"
            elif s>s_b: marker=" + more R"
            elif dd<dd_b*0.8: marker=" + lower DD"
            print(f"  {act:4.1f} {gb:5.2f} {pf:5.2f} {s:+8.0f} {wr*100:5.1f} {dd:6.0f}R {pct_trail:6.1f}% {pct_max:6.1f}%{marker}")

    # Top-3 by Total R, top-3 by lowest DD with TotalR within 10% of baseline
    print(f"\n  TOP 5 BY TOTAL R:")
    for r in sorted(results,key=lambda x:-x[3])[:5]:
        act,gb,pf,s,wr,dd,pt,pm=r
        print(f"    act={act:.2f} gb={gb:.2f}  PF={pf:.2f}  Total={s:+.0f}R ({s-s_b:+.0f})  DD={dd:.0f}R ({dd-dd_b:+.0f})")
    print(f"\n  TOP 5 BY LOWEST DRAWDOWN (TotalR within 10% of prod):")
    flt=[r for r in results if r[3]>=s_b*0.9]
    for r in sorted(flt,key=lambda x:x[5])[:5]:
        act,gb,pf,s,wr,dd,pt,pm=r
        print(f"    act={act:.2f} gb={gb:.2f}  PF={pf:.2f}  Total={s:+.0f}R ({s-s_b:+.0f})  DD={dd:.0f}R ({dd-dd_b:+.0f})")

    print(f"\n  Done in {time.time()-t0:.0f}s")

if __name__=="__main__":
    evaluate("Oracle XAU — trail param sweep on unseen 30%",
             f"{DATA}/swing_v5_xauusd.csv",
             f"{PROJECT}/experiments/v84_rl_entry/v84_rl_trades.csv",
             prod_act=3.0,prod_gb=0.60)
    evaluate("Oracle BTC — trail param sweep on unseen 30%",
             f"{DATA}/swing_v5_btc.csv",
             f"{PROJECT}/experiments/v84_rl_entry/btc_rl_trades.csv",
             prod_act=3.0,prod_gb=0.60)
