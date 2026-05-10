"""Backtest two trend-maturity policies vs current production:

  A) Skip the worst tail: drop trades where stretch_100 > 15
  B) (extra reference) Skip stretch_100 > 10
  C) Full 4-tier sizing by maturity-risk score
        score = stretch_100 + 5*pct_to_extreme_50
        Q1 (lowest score): size = 1.00
        Q2:                 size = 0.75
        Q3:                 size = 0.50
        Q4 (highest score): size = 0.25

Equity curve is computed on size-weighted contributions: each trade's
contribution is pnl_R * size_mult. PF/DD/Total reported in R-equivalents
where 1.0R = full size base. All policies layer the existing prod logic
(hard SL + v88 reverse-setup at q>0.05 + trail act=3R/gb=60% + 60-bar max).
"""
import os, glob as _glob, pickle
import numpy as np, pandas as pd
from importlib.util import spec_from_file_location, module_from_spec

PROJECT="/home/jay/Desktop/new-model-zigzag"
DATA=f"{PROJECT}/data"
spec=spec_from_file_location("v89", os.path.join(os.path.dirname(os.path.abspath(__file__)),"01_optimal_stopping.py"))
v89=module_from_spec(spec); spec.loader.exec_module(v89)

V72L=v89.V72L
SL_HARD=-4.0; MAX_HOLD=60; TRAIL_ACT=3.0; TRAIL_GB=0.60

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

def maturity_features(t_idx, d, C, atr):
    ea=atr[t_idx]
    if not np.isfinite(ea) or ea<=0: return None
    out={}
    L=100
    if t_idx>=L:
        win=C[t_idx-L:t_idx+1]
        out['stretch_100']=float((C[t_idx]-win.min())/ea) if d==1 else float((win.max()-C[t_idx])/ea)
    else:
        out['stretch_100']=0.0
    L=50
    if t_idx>=L:
        win=C[t_idx-L:t_idx+1]
        rng=win.max()-win.min()
        if d==1:
            out['pct_to_extreme_50']=float((C[t_idx]-win.min())/rng) if rng>0 else 0.5
        else:
            out['pct_to_extreme_50']=float((win.max()-C[t_idx])/rng) if rng>0 else 0.5
    else:
        out['pct_to_extreme_50']=0.5
    return out

def simulate_trade(time_,d,t2i,n,C,atr,setup_lkp,setup_Q,sw_times):
    if time_ not in t2i: return None
    ei=t2i[time_]; ep=C[ei]; ea=atr[ei]
    if not np.isfinite(ea) or ea<=0: return None
    peak=0.0; max_k=min(MAX_HOLD,n-ei-1)
    for k in range(1,max_k+1):
        bar=ei+k; R=d*(C[bar]-ep)/ea
        if R<=SL_HARD: return R
        if R>peak: peak=R
        bt=pd.Timestamp(sw_times[bar])
        if bt in setup_lkp:
            opp=-d
            if opp in setup_lkp[bt]:
                idx=setup_lkp[bt][opp]
                if setup_Q[idx]>0.05: return R
        if peak>=TRAIL_ACT and R<=peak*(1.0-TRAIL_GB): return R
    last=min(ei+max_k,n-1)
    return d*(C[last]-ep)/ea

def metrics(weighted_pnls):
    wp=np.asarray(weighted_pnls)
    s=wp.sum()
    pos=wp[wp>0]; neg=wp[wp<=0]
    pf=pos.sum()/max(-neg.sum(),1e-9) if len(neg)>0 else 99.0
    wr=(wp>0).mean()*100
    eq=np.cumsum(wp); peak=np.maximum.accumulate(eq); dd=float((peak-eq).max())
    return pf,float(s),wr,dd

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

    # Compute features + per-trade PnL
    rows=[]
    for _,trade in tst.iterrows():
        tm=trade["time"]
        if tm not in t2i: continue
        ei=t2i[tm]; d=int(trade["direction"])
        f=maturity_features(ei,d,C,atr)
        if f is None: continue
        pnl=simulate_trade(tm,d,t2i,n,C,atr,setup_lkp,setup_Q,sw_times)
        if pnl is None: continue
        rows.append({'time':tm,'cid':int(trade['cid']),'d':d,'pnl_R':pnl,
                     'stretch_100':f['stretch_100'],'pct_to_extreme_50':f['pct_to_extreme_50'],
                     'risk_score':f['stretch_100']+5*f['pct_to_extreme_50']})
    df=pd.DataFrame(rows).sort_values('time').reset_index(drop=True)
    print(f"  Test: {len(df)} trades")

    # ── Baseline (current prod, full size on every trade) ──
    base=df.pnl_R.values
    pf_b,s_b,wr_b,dd_b=metrics(base)
    print(f"\n  BASELINE (current prod, full size):")
    print(f"    PF={pf_b:.2f}  Total={s_b:+.0f}R  WR={wr_b:.1f}%  DD={dd_b:.0f}R  N={len(df)}")

    # ── Policy A: skip stretch_100 > 15 ──
    print(f"\n  POLICY A — skip trades with stretch_100 > 15 (very stretched tail)")
    mA=df.stretch_100<=15
    pA=df.loc[mA,'pnl_R'].values
    pf_A,s_A,wr_A,dd_A=metrics(pA)
    dropped=(~mA).sum()
    print(f"    Dropped: {dropped} trades   ({df.loc[~mA,'pnl_R'].sum():+.0f}R lost)")
    print(f"    PF={pf_A:.2f}  Total={s_A:+.0f}R  WR={wr_A:.1f}%  DD={dd_A:.0f}R  N={mA.sum()}")
    print(f"    vs prod: ΔPF={pf_A-pf_b:+.2f}  ΔTotal={s_A-s_b:+.0f}R  ΔDD={dd_A-dd_b:+.0f}R")

    # ── Policy B: skip stretch_100 > 10 (more aggressive) ──
    print(f"\n  POLICY B — skip trades with stretch_100 > 10 (extended too)")
    mB=df.stretch_100<=10
    pB=df.loc[mB,'pnl_R'].values
    pf_B,s_B,wr_B,dd_B=metrics(pB)
    dropped=(~mB).sum()
    print(f"    Dropped: {dropped} trades   ({df.loc[~mB,'pnl_R'].sum():+.0f}R lost)")
    print(f"    PF={pf_B:.2f}  Total={s_B:+.0f}R  WR={wr_B:.1f}%  DD={dd_B:.0f}R  N={mB.sum()}")
    print(f"    vs prod: ΔPF={pf_B-pf_b:+.2f}  ΔTotal={s_B-s_b:+.0f}R  ΔDD={dd_B-dd_b:+.0f}R")

    # ── Policy C: 4-tier sizing by risk_score ──
    print(f"\n  POLICY C — 4-tier sizing by risk_score = stretch_100 + 5×pct_to_extreme_50")
    # Quartiles
    q1,q2,q3=df.risk_score.quantile([0.25,0.50,0.75])
    print(f"    Quartile cutoffs: Q1≤{q1:.2f}  Q2≤{q2:.2f}  Q3≤{q3:.2f}  Q4>{q3:.2f}")
    for sizing_label,sizes in [
        ('Conservative (1, 0.75, 0.5, 0.25)', [1.0, 0.75, 0.5, 0.25]),
        ('Moderate     (1, 0.75, 0.5, 0.5)',  [1.0, 0.75, 0.5, 0.5]),
        ('Mild         (1, 1, 0.75, 0.5)',    [1.0, 1.0, 0.75, 0.5]),
        ('Half-tail    (1, 1, 1, 0.5)',       [1.0, 1.0, 1.0, 0.5]),
        ('Quarter-tail (1, 1, 1, 0.25)',      [1.0, 1.0, 1.0, 0.25]),
    ]:
        sm=np.where(df.risk_score<=q1, sizes[0],
            np.where(df.risk_score<=q2, sizes[1],
                np.where(df.risk_score<=q3, sizes[2], sizes[3])))
        weighted=df.pnl_R.values*sm
        pf_,s_,wr_,dd_=metrics(weighted)
        avg_size=sm.mean()
        marker = " ★" if (s_>=s_b*0.95 and dd_<dd_b*0.85) else (" + DD" if dd_<dd_b*0.85 else "")
        print(f"    {sizing_label:<35s} avg_size={avg_size:.2f}  PF={pf_:.2f}  Total={s_:+.0f}R  DD={dd_:.0f}R  ΔPF={pf_-pf_b:+.2f}  ΔDD={dd_-dd_b:+.0f}R{marker}")

    # ── Policy C variant: skip Q4 entirely ──
    print(f"\n  POLICY C-skip — drop Q4 (worst quartile by risk_score) entirely")
    mskip=df.risk_score<=q3
    pSk=df.loc[mskip,'pnl_R'].values
    pf_,s_,wr_,dd_=metrics(pSk)
    print(f"    Dropped: {(~mskip).sum()} trades  ({df.loc[~mskip,'pnl_R'].sum():+.0f}R)")
    print(f"    PF={pf_:.2f}  Total={s_:+.0f}R  WR={wr_:.1f}%  DD={dd_:.0f}R  ΔPF={pf_-pf_b:+.2f}  ΔDD={dd_-dd_b:+.0f}R")

if __name__=="__main__":
    evaluate("Oracle XAU — maturity-filter & sizing backtest",
             f"{DATA}/swing_v5_xauusd.csv",
             f"{DATA}/setups_*_v72l.csv",
             f"{PROJECT}/experiments/v84_rl_entry/v84_rl_trades.csv",
             f"{PROJECT}/products/models/oracle_xau_validated.pkl")
    evaluate("Oracle BTC — maturity-filter & sizing backtest",
             f"{DATA}/swing_v5_btc.csv",
             f"{DATA}/setups_*_v72l_btc.csv",
             f"{PROJECT}/experiments/v84_rl_entry/btc_rl_trades.csv",
             f"{PROJECT}/products/models/oracle_btc_validated.pkl")
