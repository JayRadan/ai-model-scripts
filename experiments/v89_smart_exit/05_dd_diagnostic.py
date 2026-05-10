"""Diagnose: where does the MaxDD come from?

Walk the test trades chronologically with current prod logic. At each
trade, track:
  - rolling equity curve
  - drawdown contributors (which clusters / hours / sequences cause DD?)
  - longest losing streak
  - worst single trade
  - cluster-by-cluster breakdown

This tells us WHAT to target if we want lower DD.
"""
import os, time, glob as _glob, pickle
import numpy as np, pandas as pd
from importlib.util import spec_from_file_location, module_from_spec

PROJECT="/home/jay/Desktop/new-model-zigzag"
DATA=f"{PROJECT}/data"
spec=spec_from_file_location("v89", os.path.join(os.path.dirname(os.path.abspath(__file__)),"01_optimal_stopping.py"))
v89=module_from_spec(spec); spec.loader.exec_module(v89)
V72L=v89.V72L

NAMES={0:"Uptrend",1:"MeanRevert",2:"TrendRange",3:"Downtrend",4:"HighVol"}

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

def simulate_with_meta(trades,t2i,n,C,atr,setup_lkp,setup_Q,sw_times,
                       sl=-4.0,trail_act=3.0,trail_gb=0.60,max_hold=60,v88_thr=0.05):
    """Run full prod sim, return list of dicts: time, cid, dir, pnl_R, exit_reason."""
    out=[]
    for ti,(_,trade) in enumerate(trades.iterrows()):
        tm=trade["time"]
        if tm not in t2i: continue
        ei=t2i[tm]; d=int(trade["direction"]); ep=C[ei]; ea=atr[ei]
        cid=int(trade["cid"]) if "cid" in trade else -1
        if not np.isfinite(ea) or ea<=0: continue
        peak=0.0; max_k=min(max_hold,n-ei-1); fired=False; bars=0
        for k in range(1,max_k+1):
            bar=ei+k; bars=k
            R=d*(C[bar]-ep)/ea
            if R<=sl: out.append({'time':tm,'cid':cid,'dir':d,'pnl_R':sl,'reason':'hard_sl','bars':bars,'peak':peak}); fired=True; break
            if R>peak: peak=R
            bt=pd.Timestamp(sw_times[bar])
            if bt in setup_lkp:
                opp=-d
                if opp in setup_lkp[bt]:
                    idx=setup_lkp[bt][opp]
                    if setup_Q[idx]>v88_thr:
                        out.append({'time':tm,'cid':cid,'dir':d,'pnl_R':R,'reason':'v88','bars':bars,'peak':peak}); fired=True; break
            if peak>=trail_act and R<=peak*(1.0-trail_gb):
                out.append({'time':tm,'cid':cid,'dir':d,'pnl_R':R,'reason':'trail','bars':bars,'peak':peak}); fired=True; break
        if not fired:
            last=min(ei+max_k,n-1)
            out.append({'time':tm,'cid':cid,'dir':d,'pnl_R':d*(C[last]-ep)/ea,'reason':'max_hold','bars':bars+1,'peak':peak})
    return out

def evaluate(name,swing_csv,setups_glob,trades_csv,bundle_path):
    print("\n"+"="*72); print(f"  {name}"); print("="*72)
    n,C,atr,t2i,_=v89.load_market(swing_csv,setups_glob)
    sw_times=pd.read_csv(swing_csv,parse_dates=["time"]).sort_values("time").drop_duplicates('time',keep='last')["time"].values
    trades=pd.read_csv(trades_csv,parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    split=trades["time"].quantile(0.70)
    tst=trades[trades["time"]>=split].reset_index(drop=True)

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
        tt=pd.Timestamp(times_arr[i]); dd=int(dirs_arr[i])
        if tt not in setup_lkp: setup_lkp[tt]={}
        setup_lkp[tt][dd]=i

    trades_log=simulate_with_meta(tst,t2i,n,C,atr,setup_lkp,setup_Q,sw_times)
    df=pd.DataFrame(trades_log)
    pnls=df['pnl_R'].values
    eq=np.cumsum(pnls); peak=np.maximum.accumulate(eq); ddc=peak-eq
    df['eq']=eq; df['dd_at_close']=ddc

    # ── Aggregate stats ──
    total=pnls.sum(); pos=pnls>0; wr=pos.mean()*100
    pf=pnls[pos].sum()/max(-pnls[~pos].sum(),1e-9)
    print(f"\n  Aggregate: PF={pf:.2f}  Total={total:+.0f}R  WR={wr:.1f}%  N={len(pnls)}  MaxDD={ddc.max():.1f}R")

    # ── Per-cluster breakdown ──
    print(f"\n  Per-cluster contribution to PnL and DD:")
    print(f"  {'cid':>3s} {'name':<11s} {'N':>4s} {'WR%':>5s} {'mean':>5s} {'TotalR':>7s} {'losses':>7s} {'%loss':>5s}")
    for cid in sorted(df['cid'].unique()):
        g=df[df['cid']==cid]
        N=len(g); wrc=(g['pnl_R']>0).mean()*100
        mean=g['pnl_R'].mean(); tot=g['pnl_R'].sum()
        losses=g[g['pnl_R']<=0]['pnl_R'].sum()
        loss_share=losses*100/df[df['pnl_R']<=0]['pnl_R'].sum() if df[df['pnl_R']<=0]['pnl_R'].sum()<0 else 0
        print(f"  {cid:>3d} {NAMES.get(int(cid),'?'):<11s} {N:>4d} {wrc:5.1f} {mean:+5.2f} {tot:+7.0f} {losses:+7.0f} {loss_share:5.1f}")

    # ── Worst losing streaks ──
    print(f"\n  Top-5 losing trades:")
    worst=df.nsmallest(5,'pnl_R')[['time','cid','dir','pnl_R','reason','bars','peak']]
    for _,r in worst.iterrows():
        print(f"    {r['time']} cid={int(r['cid'])} ({NAMES.get(int(r['cid']),'?')}) dir={int(r['dir']):+d} pnl={r['pnl_R']:+.2f} reason={r['reason']} bars={int(r['bars'])} peak={r['peak']:.2f}")

    # Find longest losing streak
    losses=(pnls<=0).astype(int)
    streak=0; max_streak=0; current=0; max_streak_R=0
    cum=0; current_R=0
    for v,p in zip(losses,pnls):
        if v: current+=1; current_R+=p
        else: current=0; current_R=0
        if current>max_streak: max_streak=current; max_streak_R=current_R
    print(f"\n  Longest losing streak: {max_streak} trades  ({max_streak_R:+.1f}R cumulative)")

    # ── DD-contributing window ──
    dd_idx=np.argmax(ddc)
    peak_idx=np.argmax(eq[:dd_idx+1])
    dd_window=df.iloc[peak_idx:dd_idx+1]
    print(f"\n  MaxDD window: trade #{peak_idx} → #{dd_idx} ({len(dd_window)} trades, {dd_window['time'].iloc[0]} → {dd_window['time'].iloc[-1]})")
    print(f"  Peak equity: {eq[peak_idx]:.0f}R   Trough: {eq[dd_idx]:.0f}R   Drawdown: -{ddc.max():.1f}R")
    print(f"  Cluster mix in DD window:")
    for cid in sorted(dd_window['cid'].unique()):
        g=dd_window[dd_window['cid']==cid]
        wrc=(g['pnl_R']>0).mean()*100; tot=g['pnl_R'].sum()
        print(f"    cid={int(cid)} ({NAMES.get(int(cid),'?')}): N={len(g)} WR={wrc:.1f}% Total={tot:+.0f}R")

    # ── Reason breakdown for losers ──
    print(f"\n  Loser exit-reason breakdown ({(pnls<=0).sum()} losers):")
    losers=df[df['pnl_R']<=0]
    for r,g in losers.groupby('reason'):
        print(f"    {r:>15s}: {len(g):3d} trades, mean {g['pnl_R'].mean():+.2f}R, total {g['pnl_R'].sum():+.0f}R")

if __name__=="__main__":
    evaluate("Oracle XAU — DD diagnostic",
             f"{DATA}/swing_v5_xauusd.csv",
             f"{DATA}/setups_*_v72l.csv",
             f"{PROJECT}/experiments/v84_rl_entry/v84_rl_trades.csv",
             f"{PROJECT}/products/models/oracle_xau_validated.pkl")
    evaluate("Oracle BTC — DD diagnostic",
             f"{DATA}/swing_v5_btc.csv",
             f"{DATA}/setups_*_v72l_btc.csv",
             f"{PROJECT}/experiments/v84_rl_entry/btc_rl_trades.csv",
             f"{PROJECT}/products/models/oracle_btc_validated.pkl")
