"""v87: Holdout evaluation of XAU + BTC multi-head exit bundles.

Runs the saved multi-head exit policies on the last-30% (truly unseen)
portion of v84 RL trades — for both XAUUSD and BTCUSD.

Compares:
  * baseline    — hard -4R SL only, MAX_HOLD=60
  * binary ML   — current Oracle exit_mdl (XAU only)
  * multi-head  — saved deploy thresholds in the bundle
  * multi-head sweep — a few alt threshold combos
"""
import os, time, pickle, glob as _glob
import numpy as np, pandas as pd

PROJECT="/home/jay/Desktop/new-model-zigzag"
DATA=f"{PROJECT}/data"
EXP=f"{PROJECT}/experiments/v87_multi_head_exit"

V72L=['hurst_rs','ou_theta','entropy_rate','kramers_up','wavelet_er',
      'vwap_dist','hour_enc','dow_enc','quantum_flow','quantum_flow_h4',
      'quantum_momentum','quantum_vwap_conf','quantum_divergence','quantum_div_strength',
      'vpin','sig_quad_var','har_rv_ratio','hawkes_eta']
ENRICHED=['current_R','max_R_seen','drawdown_from_peak',
          'bars_in_trade','bars_remaining','dist_to_SL','dist_to_TP',
          'vol_10bar','mom_3bar','mom_10bar']+V72L
MAX_HOLD=60; MIN_HOLD=2; SL_HARD=-4.0

OLD_CTX=['hurst_rs','ou_theta','entropy_rate','kramers_up','wavelet_er',
         'quantum_flow','quantum_flow_h4','vwap_dist']

def pf_stats(pnls):
    s=sum(pnls); w=[p for p in pnls if p>0]; l=[p for p in pnls if p<=0]
    pf=sum(w)/max(-sum(l),1e-9) if l else 99.0
    wr=len(w)/max(len(pnls),1)
    return pf,s,wr,len(pnls)

def load_market(swing_csv,setups_glob):
    swing=pd.read_csv(swing_csv,parse_dates=["time"])
    swing=swing.sort_values("time").drop_duplicates('time',keep='last').reset_index(drop=True)
    n=len(swing)
    C=swing["close"].values.astype(np.float64); H=swing["high"].values; Lo=swing["low"].values
    tr=np.concatenate([[H[0]-Lo[0]],np.maximum.reduce([H[1:]-Lo[1:],np.abs(H[1:]-C[:-1]),np.abs(Lo[1:]-C[:-1])])])
    atr=pd.Series(tr).rolling(14,min_periods=14).mean().values
    t2i={pd.Timestamp(t):i for i,t in enumerate(swing["time"].values)}
    setups=[pd.read_csv(f,parse_dates=["time"]) for f in sorted(_glob.glob(setups_glob))]
    all_df=pd.concat(setups,ignore_index=True)
    phys=all_df[['time']+V72L].drop_duplicates('time',keep='last').sort_values('time')
    swing=pd.merge_asof(swing.sort_values('time'),phys,on='time',direction='nearest')
    for c in V72L: swing[c]=swing[c].fillna(0)
    ctx=swing[V72L].fillna(0).values.astype(np.float64)
    return n,C,Lo,H,atr,t2i,ctx

def feats_at(C,atr,ctx,ei,k,d,ep,ea,n,max_seen):
    bar=ei+k
    mtm=d*(C[bar]-ep)/ea
    dist_sl=mtm-SL_HARD; dist_tp=2.0-mtm
    vol10=np.std([d*(C[max(0,bar-j)]-C[max(0,bar-j-1)])/ea for j in range(min(10,bar))]) if bar>5 else 0
    mom3=d*(C[bar]-C[max(0,bar-3)])/ea if bar>=3 else 0
    mom10=d*(C[bar]-C[max(0,bar-10)])/ea if bar>=10 else 0
    feats=[mtm,max_seen,max_seen-mtm,float(k),float(MAX_HOLD-k),
           dist_sl,dist_tp,vol10,mom3,mom10]+[float(ctx[bar,j]) for j in range(len(V72L))]
    return mtm,np.asarray(feats,dtype=np.float32)

def simulate(test_trades,t2i,n,C,atr,ctx,bundle,thresholds=None,ml_exit=None,policy="multi"):
    """policy in {'baseline','ml','multi'}; thresholds=(th_up,th_gb,th_sl,th_nh)."""
    if policy=='multi':
        m_up=bundle['models']['label_up']
        m_gb=bundle['models']['label_gb']
        m_sl=bundle['models'].get('label_sl')
        m_nh=bundle['models'].get('label_nh')
        th_up,th_gb,th_sl,th_nh=thresholds
    pnls=[]
    for _,trade in test_trades.iterrows():
        tm=trade["time"]
        if tm not in t2i: continue
        ei=t2i[tm]; d=int(trade["direction"])
        ep=C[ei]; ea=atr[ei]
        if not np.isfinite(ea) or ea<=0: continue
        # running peak so far
        peak=0.0
        exit_bar=None
        for k in range(1,MAX_HOLD+1):
            bar=ei+k
            if bar>=n: break
            mtm=d*(C[bar]-ep)/ea
            if mtm<=SL_HARD: exit_bar=bar; break
            if mtm>peak: peak=mtm
            if k<MIN_HOLD: continue
            if policy=='baseline':
                continue
            if policy=='ml' and ml_exit is not None:
                p3=d*(C[bar-3]-ep)/ea if k>=3 else mtm
                f=np.asarray([mtm,float(k),mtm-p3]+
                             [ctx[bar,V72L.index(c)] for c in OLD_CTX],dtype=np.float32)
                if float(ml_exit.predict_proba(f.reshape(1,-1))[0,1])>=0.55:
                    exit_bar=bar; break
                continue
            # multi-head
            _,f=feats_at(C,atr,ctx,ei,k,d,ep,ea,n,peak)
            f=f.reshape(1,-1)
            p_up=float(m_up.predict_proba(f)[0,1])
            p_gb=float(m_gb.predict_proba(f)[0,1])
            p_sl=float(m_sl.predict_proba(f)[0,1]) if m_sl is not None else 0.0
            p_nh=float(m_nh.predict_proba(f)[0,1]) if m_nh is not None else 0.0
            sig=False
            if p_gb>th_gb and p_up<th_up: sig=True
            if m_sl is not None and p_sl>th_sl: sig=True
            if m_nh is not None and p_nh>th_nh and p_gb<th_gb: sig=False
            if sig: exit_bar=bar; break
        if exit_bar is None: exit_bar=min(ei+MAX_HOLD,n-1)
        pnls.append(d*(C[exit_bar]-ep)/ea)
    return pnls

def evaluate(name,swing_csv,setups_glob,trades_csv,bundle_path,oracle_pkl=None):
    print("\n"+"="*64); print(f"  {name}"); print("="*64)
    t0=time.time()
    n,C,Lo,H,atr,t2i,ctx=load_market(swing_csv,setups_glob)
    trades=pd.read_csv(trades_csv,parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    split=trades["time"].quantile(0.70)
    test=trades[trades["time"]>=split].reset_index(drop=True)
    print(f"  Total RL trades: {len(trades)}  | Unseen test (>= {split}): {len(test)}")

    with open(bundle_path,"rb") as f: bundle=pickle.load(f)
    th_dep=(bundle.get('upside_threshold',0.4),
            bundle.get('giveback_threshold',0.6),
            bundle.get('stop_threshold',0.6),
            bundle.get('new_high_threshold',0.7))
    print(f"  Bundle thresholds (up,gb,sl,nh): {th_dep}")
    if 'aucs' in bundle:
        a=bundle['aucs']
        print(f"  AUCs: " + ", ".join(f"{k}={v:.3f}" for k,v in a.items()))

    ml_exit=None
    if oracle_pkl and os.path.exists(oracle_pkl):
        with open(oracle_pkl,"rb") as f: rl=pickle.load(f)
        ml_exit=rl.get("exit_mdl")

    print("\n  Running policies on unseen 30% test trades...")
    print(f"  {'Policy':<32s} {'PF':>6s} {'TotalR':>9s} {'WR%':>6s} {'N':>5s}")
    print(f"  {'-'*32} {'-'*6} {'-'*9} {'-'*6} {'-'*5}")
    base=simulate(test,t2i,n,C,atr,ctx,bundle,policy='baseline')
    pf,s,wr,N=pf_stats(base); print(f"  {'baseline (hard SL @60bars)':<32s} {pf:6.2f} {s:+9.0f} {wr*100:6.1f} {N:5d}")
    base_total=s
    if ml_exit is not None:
        ml=simulate(test,t2i,n,C,atr,ctx,bundle,ml_exit=ml_exit,policy='ml')
        pf,s,wr,N=pf_stats(ml); print(f"  {'binary ML exit (Oracle)':<32s} {pf:6.2f} {s:+9.0f} {wr*100:6.1f} {N:5d}  ({s-base_total:+.0f}R)")
    sweeps=[("deploy thresholds",th_dep),
            ("gb>0.6 up<0.4",(0.4,0.6,0.6,0.7)),
            ("gb>0.5 up<0.3",(0.3,0.5,0.6,0.7)),
            ("gb>0.7 up<0.5",(0.5,0.7,0.6,0.7)),
            ("gb>0.5 up<0.4 sl>0.5",(0.4,0.5,0.5,0.7))]
    for label,th in sweeps:
        pn=simulate(test,t2i,n,C,atr,ctx,bundle,thresholds=th,policy='multi')
        pf,s,wr,N=pf_stats(pn); print(f"  {label:<32s} {pf:6.2f} {s:+9.0f} {wr*100:6.1f} {N:5d}  ({s-base_total:+.0f}R)")
    print(f"\n  Done in {time.time()-t0:.0f}s")

if __name__=="__main__":
    evaluate("Oracle XAU — multi-head exit holdout",
             f"{DATA}/swing_v5_xauusd.csv",
             f"{DATA}/setups_*_v72l.csv",
             f"{PROJECT}/experiments/v84_rl_entry/v84_rl_trades.csv",
             f"{EXP}/multi_head_exit_bundle.pkl",
             oracle_pkl=f"{PROJECT}/products/models/oracle_xau_validated.pkl")
    evaluate("Oracle BTC — multi-head exit holdout",
             f"{DATA}/swing_v5_btc.csv",
             f"{DATA}/setups_*_v72l_btc.csv",
             f"{PROJECT}/experiments/v84_rl_entry/btc_rl_trades.csv",
             f"{EXP}/multi_head_exit_oracle_btc.pkl",
             oracle_pkl=f"{PROJECT}/products/models/oracle_btc_validated.pkl")
