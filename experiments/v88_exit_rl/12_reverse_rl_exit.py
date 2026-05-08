"""v88: Reverse-RL exit — use the same q_entry as an exit signal.

Jay's idea: at each in-trade bar, ask the OPPOSITE-direction RL entry head:
"Would you open a SHORT here?" If yes (Q high), our LONG should exit
(and vice versa for shorts).

Mapping:
  - Long trade  → check q_entry[3] (Downtrend's short-side Q)
  - Short trade → check q_entry[0] (Uptrend's long-side Q)

Policy: same prod trail (act=3, gb=0.6) PLUS reverse-RL exit when
opposite-Q > threshold for K consecutive bars (to filter flickers).

Sweep threshold ∈ {0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40}
      consecutive K ∈ {1, 2, 3}
"""
import os, time, pickle, glob as _glob
import numpy as np, pandas as pd

PROJECT="/home/jay/Desktop/new-model-zigzag"
DATA=f"{PROJECT}/data"
MAX_HOLD=60; SL_HARD=-4.0
TRAIL_ACT=3.0; TRAIL_GB=0.60
V72L=['hurst_rs','ou_theta','entropy_rate','kramers_up','wavelet_er',
      'vwap_dist','hour_enc','dow_enc','quantum_flow','quantum_flow_h4',
      'quantum_momentum','quantum_vwap_conf','quantum_divergence','quantum_div_strength',
      'vpin','sig_quad_var','har_rv_ratio','hawkes_eta']

def load_market(swing_csv,setups_glob):
    sw=pd.read_csv(swing_csv,parse_dates=["time"])
    sw=sw.sort_values("time").drop_duplicates('time',keep='last').reset_index(drop=True)
    n=len(sw); C=sw["close"].values.astype(np.float64)
    H=sw["high"].values; Lo=sw["low"].values
    tr=np.concatenate([[H[0]-Lo[0]],np.maximum.reduce([H[1:]-Lo[1:],np.abs(H[1:]-C[:-1]),np.abs(Lo[1:]-C[:-1])])])
    atr=pd.Series(tr).rolling(14,min_periods=14).mean().values
    t2i={pd.Timestamp(t):i for i,t in enumerate(sw["time"].values)}
    setups=[pd.read_csv(f,parse_dates=["time"]) for f in sorted(_glob.glob(setups_glob))]
    all_df=pd.concat(setups,ignore_index=True)
    phys=all_df[['time']+V72L].drop_duplicates('time',keep='last').sort_values('time')
    sw=pd.merge_asof(sw.sort_values('time'),phys,on='time',direction='nearest')
    for c in V72L: sw[c]=sw[c].fillna(0)
    ctx=sw[V72L].fillna(0).values.astype(np.float32)
    return n,C,atr,t2i,ctx

def precompute_opposite_Q(ctx,q_entry):
    """For every bar, compute Q for cid=0 (long) and cid=3 (short) so we
    can do per-bar lookups in O(1)."""
    print("  Pre-computing per-bar Q values...",end='',flush=True); t=time.time()
    q_long  = q_entry[0].predict(ctx)  # Uptrend's long-Q for every bar
    q_short = q_entry[3].predict(ctx)  # Downtrend's short-Q for every bar
    print(f" {time.time()-t:.1f}s",flush=True)
    return q_long.astype(np.float32), q_short.astype(np.float32)

def simulate(trades,t2i,n,C,atr,q_long,q_short,thr,k_consec):
    pnls=[]; reasons=[]
    for _,trade in trades.iterrows():
        tm=trade["time"]
        if tm not in t2i: continue
        ei=t2i[tm]; d=int(trade["direction"]); ep=C[ei]; ea=atr[ei]
        if not np.isfinite(ea) or ea<=0: continue
        opp_Q = q_short if d==1 else q_long  # opposite-direction Q
        peak=0.0; exit_bar=None; reason="max_hold"
        consec=0
        max_k=min(MAX_HOLD,n-ei-1)
        for k in range(1,max_k+1):
            bar=ei+k; mtm=d*(C[bar]-ep)/ea
            if mtm<=SL_HARD: exit_bar=bar; reason="hard_sl"; break
            if mtm>peak: peak=mtm
            # Reverse-RL exit
            if thr is not None and opp_Q[bar] > thr:
                consec += 1
                if consec >= k_consec:
                    exit_bar=bar; reason="rev_rl"; break
            else:
                consec = 0
            # Trail
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

def evaluate(name,swing_csv,setups_glob,trades_csv,bundle_path):
    print("\n"+"="*72); print(f"  {name}"); print("="*72)
    n,C,atr,t2i,ctx=load_market(swing_csv,setups_glob)
    bundle=pickle.load(open(bundle_path,"rb"))
    q_entry=bundle['q_entry']
    q_long,q_short=precompute_opposite_Q(ctx,q_entry)
    print(f"  q_long  [bar-level]: mean={q_long.mean():.3f} p90={np.percentile(q_long,90):.3f} max={q_long.max():.3f}")
    print(f"  q_short [bar-level]: mean={q_short.mean():.3f} p90={np.percentile(q_short,90):.3f} max={q_short.max():.3f}")

    trades=pd.read_csv(trades_csv,parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    split=trades["time"].quantile(0.70)
    tst=trades[trades["time"]>=split].reset_index(drop=True)
    print(f"  Unseen test: {len(tst)} trades")

    # Baseline
    pn,r=simulate(tst,t2i,n,C,atr,q_long,q_short,None,1)
    pf_b,s_b,wr_b=pf(pn); dd_b=maxdd(pn)
    print(f"\n  PROD: PF={pf_b:.2f}  Total={s_b:+.0f}R  WR={wr_b*100:.1f}%  DD={dd_b:.0f}R")

    print(f"\n  REVERSE-RL EXIT SWEEP (opp-direction q_entry > thr for K consec bars):")
    print(f"  {'thr':>5s} {'K':>2s} {'PF':>5s} {'TotalR':>8s} {'WR%':>5s} {'DD':>5s} {'rev':>4s} {'sl':>3s} {'tr':>3s} {'mx':>3s}  vs prod")
    print(f"  {'-'*5} {'-'*2} {'-'*5} {'-'*8} {'-'*5} {'-'*5} {'-'*4} {'-'*3} {'-'*3} {'-'*3}  -------")
    results=[]
    for thr in [0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50]:
        for k in [1,2,3]:
            pn,r=simulate(tst,t2i,n,C,atr,q_long,q_short,thr,k)
            pf_,s,wr=pf(pn); dd=maxdd(pn)
            rev=r.count('rev_rl'); sl=r.count('hard_sl'); tr=r.count('trail'); mx=r.count('max_hold')
            delta=s-s_b
            marker = " ★" if (s>=s_b and dd<=dd_b) else (" + DD" if dd<dd_b*0.85 else (" + R" if s>s_b else ""))
            print(f"  {thr:5.2f} {k:2d} {pf_:5.2f} {s:+8.0f} {wr*100:5.1f} {dd:4.0f}R {rev:>4d} {sl:>3d} {tr:>3d} {mx:>3d}  {delta:+5.0f}R{marker}")
            results.append((thr,k,pf_,s,wr,dd,rev))

    print(f"\n  TOP 5 BY TOTAL R:")
    for r in sorted(results,key=lambda x:-x[3])[:5]:
        thr,k,pf_,s,wr,dd,rev=r
        print(f"    thr={thr:.2f} K={k}  PF={pf_:.2f}  Total={s:+.0f}R ({s-s_b:+.0f})  DD={dd:.0f}R ({dd-dd_b:+.0f})  rev_exits={rev}")

if __name__=="__main__":
    evaluate("Oracle XAU — Reverse-RL exit",
             f"{DATA}/swing_v5_xauusd.csv",
             f"{DATA}/setups_*_v72l.csv",
             f"{PROJECT}/experiments/v84_rl_entry/v84_rl_trades.csv",
             f"{PROJECT}/products/models/oracle_xau_validated.pkl")
    evaluate("Oracle BTC — Reverse-RL exit",
             f"{DATA}/swing_v5_btc.csv",
             f"{DATA}/setups_*_v72l_btc.csv",
             f"{PROJECT}/experiments/v84_rl_entry/btc_rl_trades.csv",
             f"{PROJECT}/products/models/oracle_btc_validated.pkl")
