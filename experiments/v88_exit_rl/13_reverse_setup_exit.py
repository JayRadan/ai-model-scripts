"""v88: Reverse-RL exit — properly gated by an actual opposite-direction setup.

Jay's idea, properly implemented:
  At each in-trade bar t, check if a setup pattern in the OPPOSITE direction
  was detected at exactly time t. If yes, score that setup with q_entry[cid].
  If Q > threshold, exit our trade.

This avoids the v12 failure mode (q_entry firing on every bar regardless of
whether a setup exists). q_entry is now used the same way it was trained:
on detected setups, with their direction.
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

def load_market(swing_csv):
    sw=pd.read_csv(swing_csv,parse_dates=["time"])
    sw=sw.sort_values("time").drop_duplicates('time',keep='last').reset_index(drop=True)
    n=len(sw); C=sw["close"].values.astype(np.float64)
    H=sw["high"].values; Lo=sw["low"].values
    tr=np.concatenate([[H[0]-Lo[0]],np.maximum.reduce([H[1:]-Lo[1:],np.abs(H[1:]-C[:-1]),np.abs(Lo[1:]-C[:-1])])])
    atr=pd.Series(tr).rolling(14,min_periods=14).mean().values
    t2i={pd.Timestamp(t):i for i,t in enumerate(sw["time"].values)}
    return n,C,atr,t2i

def load_setups(setups_glob):
    """Load all setups, keep one row per (time, direction)."""
    print("  Loading setups...",end='',flush=True); t=time.time()
    parts=[]
    for f in sorted(_glob.glob(setups_glob)):
        # Filename like setups_<cid>_..._v72l.csv — old_cid encoded in path
        cid_str=os.path.basename(f).split('_')[1]
        try: old_cid=int(cid_str)
        except: old_cid=-1
        df=pd.read_csv(f,parse_dates=["time"])
        df['old_cid']=old_cid
        parts.append(df)
    s=pd.concat(parts,ignore_index=True)
    # Need direction column
    if 'direction' not in s.columns:
        print(" — no 'direction' col found; cols:",s.columns.tolist())
        return None
    # Keep first occurrence per (time, direction)
    s=s.sort_values(['time','direction']).drop_duplicates(['time','direction'],keep='first').reset_index(drop=True)
    print(f" {time.time()-t:.0f}s — {len(s):,} setup-rows ({len(s.time.unique()):,} unique times)",flush=True)
    return s

def build_setup_lookup(setups):
    """Build dict: time -> {direction: row_index_in_setups_df}."""
    print("  Indexing setups by time/direction...",end='',flush=True); t=time.time()
    lkp={}
    times=setups.time.values
    dirs=setups.direction.values
    for i in range(len(setups)):
        tm=pd.Timestamp(times[i])
        d=int(dirs[i])
        if tm not in lkp: lkp[tm]={}
        lkp[tm][d]=i
    print(f" {time.time()-t:.0f}s",flush=True)
    return lkp

def precompute_setup_Q(setups,q_entry):
    """For each setup row, compute Q from q_entry using its old_cid."""
    print("  Pre-computing Q for every setup...",end='',flush=True); t=time.time()
    Q=np.full(len(setups),-9.0,dtype=np.float32)
    for cid,m in q_entry.items():
        mask=(setups['old_cid']==cid).values
        if mask.sum()<1: continue
        Xc=setups.loc[mask,V72L].fillna(0).values
        Q[mask]=m.predict(Xc)
    print(f" {time.time()-t:.0f}s — Q range [{Q.min():.2f}, {Q.max():.2f}]",flush=True)
    return Q

def simulate(trades,t2i,n,C,atr,sw_times,setup_lkp,setup_Q,thr):
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
            # Reverse-setup exit: opposite-direction setup detected at this bar?
            bar_time=pd.Timestamp(sw_times[bar])
            if thr is not None and bar_time in setup_lkp:
                opp = -d
                if opp in setup_lkp[bar_time]:
                    idx=setup_lkp[bar_time][opp]
                    if setup_Q[idx] > thr:
                        exit_bar=bar; reason="rev_setup"; break
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
    n,C,atr,t2i=load_market(swing_csv)
    sw_times=pd.read_csv(swing_csv,parse_dates=["time"]).sort_values("time").drop_duplicates('time',keep='last')["time"].values
    setups=load_setups(setups_glob)
    if setups is None: return
    bundle=pickle.load(open(bundle_path,"rb"))
    setup_Q=precompute_setup_Q(setups,bundle['q_entry'])
    setup_lkp=build_setup_lookup(setups)

    trades=pd.read_csv(trades_csv,parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    split=trades["time"].quantile(0.70)
    tst=trades[trades["time"]>=split].reset_index(drop=True)

    pn,r=simulate(tst,t2i,n,C,atr,sw_times,setup_lkp,setup_Q,None)
    pf_b,s_b,wr_b=pf(pn); dd_b=maxdd(pn)
    print(f"\n  PROD: PF={pf_b:.2f}  Total={s_b:+.0f}R  WR={wr_b*100:.1f}%  DD={dd_b:.0f}R")

    print(f"\n  REVERSE-SETUP EXIT SWEEP (opp-direction setup detected with Q > thr):")
    print(f"  {'thr':>5s} {'PF':>5s} {'TotalR':>8s} {'WR%':>5s} {'DD':>5s} {'rev':>4s} {'sl':>3s} {'tr':>3s} {'mx':>3s}  vs prod")
    print(f"  {'-'*5} {'-'*5} {'-'*8} {'-'*5} {'-'*5} {'-'*4} {'-'*3} {'-'*3} {'-'*3}  -------")
    results=[]
    for thr in [0.10,0.20,0.30,0.40,0.50,0.60,0.70]:
        pn,r=simulate(tst,t2i,n,C,atr,sw_times,setup_lkp,setup_Q,thr)
        pf_,s,wr=pf(pn); dd=maxdd(pn)
        rev=r.count('rev_setup'); sl=r.count('hard_sl'); tr=r.count('trail'); mx=r.count('max_hold')
        delta=s-s_b
        marker = " ★" if (s>s_b and dd<=dd_b) else (" + R" if s>s_b else "")
        print(f"  {thr:5.2f} {pf_:5.2f} {s:+8.0f} {wr*100:5.1f} {dd:4.0f}R {rev:>4d} {sl:>3d} {tr:>3d} {mx:>3d}  {delta:+5.0f}R{marker}")
        results.append((thr,pf_,s,wr,dd,rev))

    print(f"\n  TOP 3 BY TOTAL R:")
    for r in sorted(results,key=lambda x:-x[2])[:3]:
        thr,pf_,s,wr,dd,rev=r
        print(f"    thr={thr:.2f}  PF={pf_:.2f}  Total={s:+.0f}R ({s-s_b:+.0f})  DD={dd:.0f}R ({dd-dd_b:+.0f})  rev_exits={rev}")

if __name__=="__main__":
    evaluate("Oracle XAU — Reverse-SETUP exit",
             f"{DATA}/swing_v5_xauusd.csv",
             f"{DATA}/setups_*_v72l.csv",
             f"{PROJECT}/experiments/v84_rl_entry/v84_rl_trades.csv",
             f"{PROJECT}/products/models/oracle_xau_validated.pkl")
    evaluate("Oracle BTC — Reverse-SETUP exit",
             f"{DATA}/swing_v5_btc.csv",
             f"{DATA}/setups_*_v72l_btc.csv",
             f"{PROJECT}/experiments/v84_rl_entry/btc_rl_trades.csv",
             f"{PROJECT}/products/models/oracle_btc_validated.pkl")
