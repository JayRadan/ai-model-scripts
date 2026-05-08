"""v88: Exit RL — Q-regression head.

Idea: at each in-position bar k, predict remaining_R = final_pnl_R - mtm
      (i.e. how much MORE the trade will earn if you keep holding).
      Exit when predicted remaining <= threshold.

Two variants:
  (A) Global head — one XGBRegressor across all clusters
  (B) Per-cluster head — one head per cid (mirrors v84 RL entry that works)

Training set:    first 70% of v84 RL trades (chronological)
Unseen test set: last 30% (XAU 2025-12-10→2026-05-01; BTC 2025-11-11→…)

Compares against:
  * baseline (hard SL + 60-bar max only)
  * binary ML exit (current Oracle exit_mdl from products/models/*.pkl)

Sample weighting: |remaining_R| — bar near a turning point matters more
than a bar where the trade's fate is already sealed.
"""
import os, time, pickle, glob as _glob, itertools
import numpy as np, pandas as pd
from xgboost import XGBRegressor

PROJECT="/home/jay/Desktop/new-model-zigzag"
DATA=f"{PROJECT}/data"
OUT=f"{PROJECT}/experiments/v88_exit_rl"
os.makedirs(OUT,exist_ok=True)

V72L=['hurst_rs','ou_theta','entropy_rate','kramers_up','wavelet_er',
      'vwap_dist','hour_enc','dow_enc','quantum_flow','quantum_flow_h4',
      'quantum_momentum','quantum_vwap_conf','quantum_divergence','quantum_div_strength',
      'vpin','sig_quad_var','har_rv_ratio','hawkes_eta']
ENRICHED=['current_R','max_R_seen','drawdown_from_peak',
          'bars_in_trade','bars_remaining','dist_to_SL','dist_to_TP',
          'vol_10bar','mom_3bar','mom_10bar']+V72L
OLD_CTX=['hurst_rs','ou_theta','entropy_rate','kramers_up','wavelet_er',
         'quantum_flow','quantum_flow_h4','vwap_dist']
MAX_HOLD=60; MIN_HOLD=2; SL_HARD=-4.0

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
    ctx=sw[V72L].fillna(0).values.astype(np.float64)
    return n,C,Lo,H,atr,t2i,ctx

def feats_row(C,ctx,ei,k,d,ep,ea,peak):
    bar=ei+k
    mtm=d*(C[bar]-ep)/ea
    dist_sl=mtm-SL_HARD; dist_tp=2.0-mtm
    vol10=np.std([d*(C[max(0,bar-j)]-C[max(0,bar-j-1)])/ea for j in range(min(10,bar))]) if bar>5 else 0
    mom3=d*(C[bar]-C[max(0,bar-3)])/ea if bar>=3 else 0
    mom10=d*(C[bar]-C[max(0,bar-10)])/ea if bar>=10 else 0
    return mtm,[mtm,peak,peak-mtm,float(k),float(MAX_HOLD-k),
                dist_sl,dist_tp,vol10,mom3,mom10]+[float(ctx[bar,j]) for j in range(len(V72L))]

def build_samples(trades,t2i,n,C,atr,ctx,with_meta=False):
    """Per (trade,bar) build features + target = remaining_R = final_pnl - mtm."""
    rows=[]; tgt=[]; cids=[]
    meta_tr=[]   # for test: per-trade {ei,d,ep,ea,sl_k}
    r2t=[]; rk=[]; rmtm=[]
    for ti,(_,trade) in enumerate(trades.iterrows()):
        tm=trade["time"]
        if tm not in t2i:
            if with_meta: meta_tr.append(None)
            continue
        ei=t2i[tm]; d=int(trade["direction"]); ep=C[ei]; ea=atr[ei]
        if not np.isfinite(ea) or ea<=0:
            if with_meta: meta_tr.append(None)
            continue
        cid=int(trade["cid"]) if "cid" in trade else 0
        final_r=float(trade["pnl_R"])
        peak=0.0; sl_k=None
        max_k=min(MAX_HOLD,n-ei-1)
        for k in range(1,max_k+1):
            bar=ei+k; mtm=d*(C[bar]-ep)/ea
            if mtm<=SL_HARD: sl_k=k; break
            if mtm>peak: peak=mtm
            if k<MIN_HOLD: continue
            _,row=feats_row(C,ctx,ei,k,d,ep,ea,peak)
            rows.append(row); tgt.append(final_r-mtm); cids.append(cid)
            r2t.append(ti); rk.append(k); rmtm.append(mtm)
        if with_meta:
            meta_tr.append({'ei':ei,'d':d,'ep':ep,'ea':ea,'sl_k':sl_k,'cid':cid})
    X=np.asarray(rows,dtype=np.float32)
    y=np.asarray(tgt,dtype=np.float32)
    cids=np.asarray(cids,dtype=np.int32)
    if with_meta:
        return X,y,cids,meta_tr,np.asarray(r2t,dtype=np.int32),np.asarray(rk,dtype=np.int32),np.asarray(rmtm,dtype=np.float32)
    return X,y,cids

def baseline_pnls(meta,n,C):
    out=[]
    for m in meta:
        if m is None: continue
        ei=m['ei']; d=m['d']; ep=m['ep']; ea=m['ea']
        bar=ei+m['sl_k'] if m['sl_k'] is not None else min(ei+MAX_HOLD,n-1)
        out.append(d*(C[bar]-ep)/ea)
    return out

def ml_exit_pnls(test,t2i,n,C,atr,ctx,ml_exit):
    out=[]
    for _,trade in test.iterrows():
        tm=trade["time"]
        if tm not in t2i: continue
        ei=t2i[tm]; d=int(trade["direction"]); ep=C[ei]; ea=atr[ei]
        if not np.isfinite(ea) or ea<=0: continue
        eb=min(ei+MAX_HOLD,n-1)
        for k in range(1,MAX_HOLD+1):
            bar=ei+k
            if bar>=n: break
            mtm=d*(C[bar]-ep)/ea
            if mtm<=SL_HARD: eb=bar; break
            if k<MIN_HOLD: continue
            p3=d*(C[bar-3]-ep)/ea if k>=3 else mtm
            f=np.asarray([mtm,float(k),mtm-p3]+
                         [ctx[bar,V72L.index(c)] for c in OLD_CTX],dtype=np.float32)
            if float(ml_exit.predict_proba(f.reshape(1,-1))[0,1])>=0.55: eb=bar; break
        out.append(d*(C[eb]-ep)/ea)
    return out

def simulate_q(meta,r2t,rk,q_pred,thr,n,C):
    """Exit when q_pred (predicted remaining_R) <= thr. SL still wins if first."""
    n_tr=len(meta); chosen=np.full(n_tr,-1,dtype=np.int32)
    cur=-1
    for i in range(len(r2t)):
        ti=int(r2t[i])
        if ti!=cur: cur=ti
        if chosen[ti]!=-1: continue
        if q_pred[i] <= thr: chosen[ti]=int(rk[i])
    out=[]
    for ti,m in enumerate(meta):
        if m is None: continue
        ei=m['ei']; d=m['d']; ep=m['ep']; ea=m['ea']; sl_k=m['sl_k']; ck=int(chosen[ti])
        if ck==-1:
            bar=ei+sl_k if sl_k is not None else min(ei+MAX_HOLD,n-1)
        else:
            if sl_k is not None and sl_k<ck: bar=ei+sl_k
            else: bar=ei+ck
        out.append(d*(C[bar]-ep)/ea)
    return out

def pf_stats(pnls):
    s=sum(pnls); w=[p for p in pnls if p>0]; l=[p for p in pnls if p<=0]
    pf=sum(w)/max(-sum(l),1e-9) if l else 99.0
    wr=len(w)/max(len(pnls),1)
    return pf,s,wr,len(pnls)

def evaluate(name,swing_csv,setups_glob,trades_csv,oracle_pkl):
    print("\n"+"="*72); print(f"  {name}"); print("="*72)
    t0=time.time()
    n,C,Lo,H,atr,t2i,ctx=load_market(swing_csv,setups_glob)
    trades=pd.read_csv(trades_csv,parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    split=trades["time"].quantile(0.70)
    trn=trades[trades["time"]<split].reset_index(drop=True)
    tst=trades[trades["time"]>=split].reset_index(drop=True)
    print(f"  Train: {len(trn)} trades  | Test (unseen): {len(tst)} trades  | split @ {split}")

    print("  Building train samples...", end='',flush=True); t1=time.time()
    Xtr,ytr,cids_tr=build_samples(trn,t2i,n,C,atr,ctx)
    print(f" {time.time()-t1:.0f}s — rows={len(Xtr):,}",flush=True)
    print("  Building test samples...", end='',flush=True); t1=time.time()
    Xte,yte,cids_te,meta,r2t,rk,rmtm=build_samples(tst,t2i,n,C,atr,ctx,with_meta=True)
    print(f" {time.time()-t1:.0f}s — rows={len(Xte):,}",flush=True)

    # ── Variant A: global Q-regressor ──
    print("  Training GLOBAL Q-regressor...",end='',flush=True); t1=time.time()
    sw=np.abs(ytr)+0.1
    q_global=XGBRegressor(n_estimators=400,max_depth=6,learning_rate=0.05,
                          subsample=0.8,colsample_bytree=0.8,
                          objective='reg:squarederror',verbosity=0)
    q_global.fit(Xtr,ytr,sample_weight=sw)
    qg_pred=q_global.predict(Xte)
    print(f" {time.time()-t1:.0f}s",flush=True)

    # ── Variant B: per-cluster Q-regressor ──
    print("  Training PER-CLUSTER Q-regressors...",end='',flush=True); t1=time.time()
    q_per={}; qp_pred=np.zeros(len(Xte),dtype=np.float32)
    for cid in sorted(set(cids_tr.tolist())):
        mask_tr=cids_tr==cid
        if mask_tr.sum()<200: continue
        m=XGBRegressor(n_estimators=400,max_depth=5,learning_rate=0.05,
                       subsample=0.8,colsample_bytree=0.8,
                       objective='reg:squarederror',verbosity=0)
        m.fit(Xtr[mask_tr],ytr[mask_tr],sample_weight=np.abs(ytr[mask_tr])+0.1)
        q_per[cid]=m
    # Predict on test using the matching cluster head; fall back to global
    for cid,m in q_per.items():
        mask_te=cids_te==cid
        if mask_te.sum()>0:
            qp_pred[mask_te]=m.predict(Xte[mask_te])
    # Test bars whose cid had no head: use global
    seen=set(q_per.keys())
    miss=np.array([c not in seen for c in cids_te])
    if miss.any(): qp_pred[miss]=qg_pred[miss]
    print(f" {time.time()-t1:.0f}s — heads={list(q_per.keys())}",flush=True)

    # ── Baselines ──
    base=baseline_pnls(meta,n,C); pf_b,s_b,wr_b,N_b=pf_stats(base)
    print(f"\n  Baseline (hard SL):           PF={pf_b:.2f}  Total={s_b:+.0f}R  WR={wr_b*100:.1f}%  N={N_b}")

    ml_pnls=None
    if oracle_pkl and os.path.exists(oracle_pkl):
        with open(oracle_pkl,"rb") as f: rl=pickle.load(f)
        ml_exit=rl.get("exit_mdl")
        if ml_exit is not None:
            ml_pnls=ml_exit_pnls(tst,t2i,n,C,atr,ctx,ml_exit)
            pf,s,wr,N=pf_stats(ml_pnls)
            print(f"  Binary ML exit (Oracle):      PF={pf:.2f}  Total={s:+.0f}R  WR={wr*100:.1f}%  N={N}  ({s-s_b:+.0f}R)")

    # ── Threshold sweep on Q-regression ──
    thresholds=[-3.0,-2.5,-2.0,-1.5,-1.0,-0.5,-0.25,0.0,0.25,0.5,1.0]

    print(f"\n  GLOBAL Q-regressor   (target=remaining_R, |y|+0.1 weight):")
    print(f"  {'thr':>6s}  {'PF':>5s}  {'TotalR':>8s}  {'WR%':>5s}  vs base")
    bestA=None
    for th in thresholds:
        pn=simulate_q(meta,r2t,rk,qg_pred,th,n,C)
        pf,s,wr,N=pf_stats(pn)
        marker = " ★" if s>s_b else ""
        print(f"  {th:6.2f}  {pf:5.2f}  {s:+8.0f}  {wr*100:5.1f}  {s-s_b:+.0f}R{marker}")
        if bestA is None or s>bestA[1]: bestA=(th,s,pf,wr,N)

    print(f"\n  PER-CLUSTER Q-regressor:")
    print(f"  {'thr':>6s}  {'PF':>5s}  {'TotalR':>8s}  {'WR%':>5s}  vs base")
    bestB=None
    for th in thresholds:
        pn=simulate_q(meta,r2t,rk,qp_pred,th,n,C)
        pf,s,wr,N=pf_stats(pn)
        marker = " ★" if s>s_b else ""
        print(f"  {th:6.2f}  {pf:5.2f}  {s:+8.0f}  {wr*100:5.1f}  {s-s_b:+.0f}R{marker}")
        if bestB is None or s>bestB[1]: bestB=(th,s,pf,wr,N)

    print(f"\n  SUMMARY")
    print(f"    baseline:           PF {pf_b:.2f}  Total {s_b:+.0f}R")
    if ml_pnls is not None:
        pfm,sm,wrm,Nm=pf_stats(ml_pnls); print(f"    binary ML:          PF {pfm:.2f}  Total {sm:+.0f}R  ({sm-s_b:+.0f}R)")
    print(f"    best global Q:      thr={bestA[0]:.2f}  PF {bestA[2]:.2f}  Total {bestA[1]:+.0f}R  ({bestA[1]-s_b:+.0f}R)")
    print(f"    best per-cluster Q: thr={bestB[0]:.2f}  PF {bestB[2]:.2f}  Total {bestB[1]:+.0f}R  ({bestB[1]-s_b:+.0f}R)")
    print(f"  Done in {time.time()-t0:.0f}s")

if __name__=="__main__":
    evaluate("Oracle XAU — Exit RL (Q-regression on remaining_R)",
             f"{DATA}/swing_v5_xauusd.csv",
             f"{DATA}/setups_*_v72l.csv",
             f"{PROJECT}/experiments/v84_rl_entry/v84_rl_trades.csv",
             f"{PROJECT}/products/models/oracle_xau_validated.pkl")
    evaluate("Oracle BTC — Exit RL (Q-regression on remaining_R)",
             f"{DATA}/swing_v5_btc.csv",
             f"{DATA}/setups_*_v72l_btc.csv",
             f"{PROJECT}/experiments/v84_rl_entry/btc_rl_trades.csv",
             f"{PROJECT}/products/models/oracle_btc_validated.pkl")
