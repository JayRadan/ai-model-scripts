"""v88: Exit RL with M1 micro-structure features.

For each in-trade M5 bar, look back at M1 data to extract pullback-warning
signals that show up faster than the 4h regime classifier:

  m1_rv_5m / m1_rv_30m         realized vol on M1 close-to-close
  m1_rv_ratio                  5m/30m vol (>>1 = sudden vol expansion)
  m1_skew_15m                  return skew (negative = downward bias)
  m1_eff_15m                   |net move| / sum |moves|  (low = chop/turn)
  m1_against_pos_5m            count of M1 bars moving against trade dir
  m1_mae_15m                   max adverse excursion vs trade dir, in atr
  m1_velocity_z                |last 5m return| / |last 30m std|
  m1_jerk                      change in 1m velocity (acceleration)

Targets and split: identical to 01_train_test.py (Q-regression on
remaining_R, |y|+0.1 sample weight, 70/30 chrono on v84 RL trades).

Compares:
  baseline (hard SL) | binary ML | global Q (M5 only) | global Q + M1
"""
import os, time, pickle, glob as _glob
import numpy as np, pandas as pd
from xgboost import XGBRegressor

PROJECT="/home/jay/Desktop/new-model-zigzag"
DATA=f"{PROJECT}/data"
DUK="/tmp/duk"
OUT=f"{PROJECT}/experiments/v88_exit_rl"

V72L=['hurst_rs','ou_theta','entropy_rate','kramers_up','wavelet_er',
      'vwap_dist','hour_enc','dow_enc','quantum_flow','quantum_flow_h4',
      'quantum_momentum','quantum_vwap_conf','quantum_divergence','quantum_div_strength',
      'vpin','sig_quad_var','har_rv_ratio','hawkes_eta']
M1_FEATS=['m1_rv_5m','m1_rv_30m','m1_rv_ratio','m1_skew_15m','m1_eff_15m',
          'm1_against_pos_5m','m1_mae_15m','m1_velocity_z','m1_jerk']
ENRICHED=['current_R','max_R_seen','drawdown_from_peak',
          'bars_in_trade','bars_remaining','dist_to_SL','dist_to_TP',
          'vol_10bar','mom_3bar','mom_10bar']+V72L
ENRICHED_M1=ENRICHED+M1_FEATS
OLD_CTX=['hurst_rs','ou_theta','entropy_rate','kramers_up','wavelet_er',
         'quantum_flow','quantum_flow_h4','vwap_dist']
MAX_HOLD=60; MIN_HOLD=2; SL_HARD=-4.0

def load_m1(path):
    """Load M1 OHLC CSV, return (m1_time np.int64 ns, m1_close, m1_high, m1_low)."""
    m=pd.read_csv(path,parse_dates=["time"])
    m=m.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    return (m["time"].values.astype("datetime64[ns]").astype(np.int64),
            m["close"].values.astype(np.float64),
            m["high"].values.astype(np.float64),
            m["low"].values.astype(np.float64))

def m1_feats_at(t_ns, d, ea, m1_t, m1_c, m1_h, m1_l):
    """Compute M1 features in [t-30min, t] for a trade with direction d, ATR ea."""
    win30=30*60_000_000_000
    win15=15*60_000_000_000
    win5 =5 *60_000_000_000
    win1 =1 *60_000_000_000
    j=np.searchsorted(m1_t,t_ns,side='right')   # idx of first bar > t
    if j<2: return [0]*len(M1_FEATS)
    # 30m slice
    a30=np.searchsorted(m1_t,t_ns-win30,side='left')
    if a30>=j-1: return [0]*len(M1_FEATS)
    closes=m1_c[a30:j]
    if len(closes)<5: return [0]*len(M1_FEATS)
    rets=np.diff(closes)
    rv_30=float(np.std(rets))/ea if ea>0 else 0.0
    # 5m slice
    a5=np.searchsorted(m1_t,t_ns-win5,side='left')
    closes5=m1_c[a5:j]
    rets5=np.diff(closes5) if len(closes5)>1 else np.array([0.0])
    rv_5=float(np.std(rets5))/ea if ea>0 and len(rets5)>1 else 0.0
    rv_ratio=rv_5/(rv_30+1e-9)
    # 15m
    a15=np.searchsorted(m1_t,t_ns-win15,side='left')
    closes15=m1_c[a15:j]
    rets15=np.diff(closes15) if len(closes15)>1 else np.array([0.0])
    if len(rets15)>3 and rets15.std()>0:
        skew=float(((rets15-rets15.mean())**3).mean()/(rets15.std()**3))
    else:
        skew=0.0
    if len(closes15)>1:
        net=closes15[-1]-closes15[0]
        gross=np.sum(np.abs(rets15))+1e-9
        eff=float(np.abs(net)/gross)
    else:
        eff=0.0
    # against-position count over last 5m
    against=float(np.sum((d*rets5)<0))/max(len(rets5),1) if len(rets5)>0 else 0.0
    # MAE over 15m (vs trade direction)
    if d==1:
        worst=closes15[0]-m1_l[a15:j].min() if a15<j else 0.0
    else:
        worst=m1_h[a15:j].max()-closes15[0] if a15<j else 0.0
    mae=float(worst)/ea if ea>0 else 0.0
    # velocity z
    a1=np.searchsorted(m1_t,t_ns-win1,side='left')
    last_ret=closes[-1]-m1_c[max(a1-1,0)] if a1<j else 0.0
    vel_z=float(last_ret/(rets.std()+1e-9))*d  # signed by direction
    # jerk: velocity 1m vs velocity 5m-ago
    if len(closes)>=6:
        v_now=closes[-1]-closes[-2]
        v_5ago=closes[-6]-closes[-7] if len(closes)>=7 else 0.0
        jerk=float((v_now-v_5ago)/(rets.std()+1e-9))*d
    else:
        jerk=0.0
    return [rv_5,rv_30,rv_ratio,skew,eff,against,mae,vel_z,jerk]

def load_market(swing_csv,setups_glob):
    sw=pd.read_csv(swing_csv,parse_dates=["time"])
    sw=sw.sort_values("time").drop_duplicates('time',keep='last').reset_index(drop=True)
    n=len(sw); C=sw["close"].values.astype(np.float64)
    H=sw["high"].values; Lo=sw["low"].values
    tr=np.concatenate([[H[0]-Lo[0]],np.maximum.reduce([H[1:]-Lo[1:],np.abs(H[1:]-C[:-1]),np.abs(Lo[1:]-C[:-1])])])
    atr=pd.Series(tr).rolling(14,min_periods=14).mean().values
    t2i={pd.Timestamp(t):i for i,t in enumerate(sw["time"].values)}
    swing_ns=sw["time"].values.astype("datetime64[ns]").astype(np.int64)
    setups=[pd.read_csv(f,parse_dates=["time"]) for f in sorted(_glob.glob(setups_glob))]
    all_df=pd.concat(setups,ignore_index=True)
    phys=all_df[['time']+V72L].drop_duplicates('time',keep='last').sort_values('time')
    sw=pd.merge_asof(sw.sort_values('time'),phys,on='time',direction='nearest')
    for c in V72L: sw[c]=sw[c].fillna(0)
    ctx=sw[V72L].fillna(0).values.astype(np.float64)
    return n,C,Lo,H,atr,t2i,ctx,swing_ns

def m5_feats(C,ctx,ei,k,d,ep,ea,peak):
    bar=ei+k
    mtm=d*(C[bar]-ep)/ea
    dist_sl=mtm-SL_HARD; dist_tp=2.0-mtm
    vol10=np.std([d*(C[max(0,bar-j)]-C[max(0,bar-j-1)])/ea for j in range(min(10,bar))]) if bar>5 else 0
    mom3=d*(C[bar]-C[max(0,bar-3)])/ea if bar>=3 else 0
    mom10=d*(C[bar]-C[max(0,bar-10)])/ea if bar>=10 else 0
    return mtm,[mtm,peak,peak-mtm,float(k),float(MAX_HOLD-k),
                dist_sl,dist_tp,vol10,mom3,mom10]+[float(ctx[bar,j]) for j in range(len(V72L))]

def build_samples(trades,t2i,n,C,atr,ctx,swing_ns,m1_t,m1_c,m1_h,m1_l,with_meta=False):
    rows=[]; tgt=[]; cids=[]
    meta=[]; r2t=[]; rk=[]
    for ti,(_,trade) in enumerate(trades.iterrows()):
        tm=trade["time"]
        if tm not in t2i:
            if with_meta: meta.append(None)
            continue
        ei=t2i[tm]; d=int(trade["direction"]); ep=C[ei]; ea=atr[ei]
        if not np.isfinite(ea) or ea<=0:
            if with_meta: meta.append(None)
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
            _,row=m5_feats(C,ctx,ei,k,d,ep,ea,peak)
            row=row+m1_feats_at(swing_ns[bar],d,ea,m1_t,m1_c,m1_h,m1_l)
            rows.append(row); tgt.append(final_r-mtm); cids.append(cid)
            r2t.append(ti); rk.append(k)
        if with_meta:
            meta.append({'ei':ei,'d':d,'ep':ep,'ea':ea,'sl_k':sl_k,'cid':cid})
    X=np.asarray(rows,dtype=np.float32)
    y=np.asarray(tgt,dtype=np.float32)
    return (X,y,np.asarray(cids,dtype=np.int32),
            meta,np.asarray(r2t,dtype=np.int32),np.asarray(rk,dtype=np.int32))

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
    n_tr=len(meta); chosen=np.full(n_tr,-1,dtype=np.int32)
    for i in range(len(r2t)):
        ti=int(r2t[i])
        if chosen[ti]!=-1: continue
        if q_pred[i]<=thr: chosen[ti]=int(rk[i])
    out=[]
    for ti,m in enumerate(meta):
        if m is None: continue
        ei=m['ei']; d=m['d']; ep=m['ep']; ea=m['ea']; sl_k=m['sl_k']; ck=int(chosen[ti])
        if ck==-1: bar=ei+sl_k if sl_k is not None else min(ei+MAX_HOLD,n-1)
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

def evaluate(name,swing_csv,setups_glob,trades_csv,oracle_pkl,m1_csv):
    print("\n"+"="*72); print(f"  {name}"); print("="*72)
    t0=time.time()
    n,C,Lo,H,atr,t2i,ctx,swing_ns=load_market(swing_csv,setups_glob)
    m1_t,m1_c,m1_h,m1_l=load_m1(m1_csv)
    print(f"  M1 bars: {len(m1_t):,}")
    trades=pd.read_csv(trades_csv,parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    split=trades["time"].quantile(0.70)
    trn=trades[trades["time"]<split].reset_index(drop=True)
    tst=trades[trades["time"]>=split].reset_index(drop=True)
    print(f"  Train: {len(trn)} | Unseen test: {len(tst)} | split @ {split}")

    print("  Building train (M5+M1) samples...",end='',flush=True); t1=time.time()
    Xtr,ytr,cids_tr,_,_,_=build_samples(trn,t2i,n,C,atr,ctx,swing_ns,m1_t,m1_c,m1_h,m1_l)
    print(f" {time.time()-t1:.0f}s — rows={len(Xtr):,}",flush=True)

    print("  Building test (M5+M1) samples...",end='',flush=True); t1=time.time()
    Xte,yte,cids_te,meta,r2t,rk=build_samples(tst,t2i,n,C,atr,ctx,swing_ns,m1_t,m1_c,m1_h,m1_l,with_meta=True)
    print(f" {time.time()-t1:.0f}s — rows={len(Xte):,}",flush=True)

    n_m5=len(ENRICHED)
    print("  Train Q-regressor (M5 only) ...",end='',flush=True); t1=time.time()
    sw=np.abs(ytr)+0.1
    q_m5=XGBRegressor(n_estimators=400,max_depth=6,learning_rate=0.05,
                      subsample=0.8,colsample_bytree=0.8,verbosity=0,objective='reg:squarederror')
    q_m5.fit(Xtr[:,:n_m5],ytr,sample_weight=sw)
    pred_m5=q_m5.predict(Xte[:,:n_m5])
    print(f" {time.time()-t1:.0f}s",flush=True)

    print("  Train Q-regressor (M5+M1) ...",end='',flush=True); t1=time.time()
    q_m1=XGBRegressor(n_estimators=400,max_depth=6,learning_rate=0.05,
                      subsample=0.8,colsample_bytree=0.8,verbosity=0,objective='reg:squarederror')
    q_m1.fit(Xtr,ytr,sample_weight=sw)
    pred_m1=q_m1.predict(Xte)
    print(f" {time.time()-t1:.0f}s",flush=True)

    base=baseline_pnls(meta,n,C); pf_b,s_b,wr_b,N_b=pf_stats(base)
    print(f"\n  Baseline (hard SL):     PF={pf_b:.2f}  Total={s_b:+.0f}R  WR={wr_b*100:.1f}%  N={N_b}")
    if oracle_pkl and os.path.exists(oracle_pkl):
        with open(oracle_pkl,"rb") as f: rl=pickle.load(f)
        ml_exit=rl.get("exit_mdl")
        if ml_exit is not None:
            ml_pnls=ml_exit_pnls(tst,t2i,n,C,atr,ctx,ml_exit)
            pf,s,wr,N=pf_stats(ml_pnls)
            print(f"  Binary ML exit:         PF={pf:.2f}  Total={s:+.0f}R  WR={wr*100:.1f}%  N={N}  ({s-s_b:+.0f}R)")

    thresholds=[-3.0,-2.5,-2.0,-1.5,-1.0,-0.5,-0.25,0.0,0.25,0.5,1.0]

    print(f"\n  Q-regression (M5 only) — for reference:")
    print(f"  {'thr':>6s}  {'PF':>5s}  {'TotalR':>8s}  {'WR%':>5s}  vs base")
    bestA=None
    for th in thresholds:
        pn=simulate_q(meta,r2t,rk,pred_m5,th,n,C)
        pf,s,wr,N=pf_stats(pn)
        marker=" ★" if s>s_b else ""
        print(f"  {th:6.2f}  {pf:5.2f}  {s:+8.0f}  {wr*100:5.1f}  {s-s_b:+.0f}R{marker}")
        if bestA is None or s>bestA[1]: bestA=(th,s,pf,wr,N)

    print(f"\n  Q-regression (M5+M1):")
    print(f"  {'thr':>6s}  {'PF':>5s}  {'TotalR':>8s}  {'WR%':>5s}  vs base")
    bestB=None
    for th in thresholds:
        pn=simulate_q(meta,r2t,rk,pred_m1,th,n,C)
        pf,s,wr,N=pf_stats(pn)
        marker=" ★" if s>s_b else ""
        print(f"  {th:6.2f}  {pf:5.2f}  {s:+8.0f}  {wr*100:5.1f}  {s-s_b:+.0f}R{marker}")
        if bestB is None or s>bestB[1]: bestB=(th,s,pf,wr,N)

    # Feature importance for the M1 head
    fi=q_m1.feature_importances_
    fnames=ENRICHED+M1_FEATS
    top=sorted(enumerate(fi),key=lambda x:-x[1])[:12]
    print(f"\n  Top-12 features (M5+M1 head):")
    for i,v in top:
        tag="M1" if fnames[i] in M1_FEATS else "M5"
        print(f"    {tag} {fnames[i]:<28s} {v:.4f}")

    print(f"\n  SUMMARY")
    print(f"    baseline:     PF {pf_b:.2f}  Total {s_b:+.0f}R")
    print(f"    best Q M5:    thr={bestA[0]:.2f}  PF {bestA[2]:.2f}  Total {bestA[1]:+.0f}R  ({bestA[1]-s_b:+.0f}R)")
    print(f"    best Q M5+M1: thr={bestB[0]:.2f}  PF {bestB[2]:.2f}  Total {bestB[1]:+.0f}R  ({bestB[1]-s_b:+.0f}R)")
    print(f"  Done in {time.time()-t0:.0f}s")

if __name__=="__main__":
    evaluate("Oracle XAU — Exit RL with M1 features",
             f"{DATA}/swing_v5_xauusd.csv",
             f"{DATA}/setups_*_v72l.csv",
             f"{PROJECT}/experiments/v84_rl_entry/v84_rl_trades.csv",
             f"{PROJECT}/products/models/oracle_xau_validated.pkl",
             f"{DUK}/xau_m1.csv")
    evaluate("Oracle BTC — Exit RL with M1 features",
             f"{DATA}/swing_v5_btc.csv",
             f"{DATA}/setups_*_v72l_btc.csv",
             f"{PROJECT}/experiments/v84_rl_entry/btc_rl_trades.csv",
             f"{PROJECT}/products/models/oracle_btc_validated.pkl",
             f"{DUK}/btc_m1.csv")
