"""v88: Adversarial entry filter with TICK-LEVEL features (memory-safe).

Memory strategy: never concat all tick parquets. Instead group trades by
date and load only that date + previous date (covers the 30-min lookback
across day boundaries). Caps RAM at ~10 MB regardless of total tick volume.

Tick features (at entry time t, 30-min lookback):
  spread_mean         mean (ask - bid) / atr
  spread_recent_z     current spread vs 30m mean
  bid_vol_share_30m   bidVol / (bidVol + askVol)
  vol_imbalance_5m    same over last 5m
  signed_flow_5m      sum(bidVol - askVol) * direction over 5m
  microprice_drift_5m delta microprice over 5m, signed
  microprice_skew_30m mean(microprice - mid) over 30m
  tick_rate_ratio     n_ticks_5m * 6 / n_ticks_30m
  mean_intertick_ms   avg ms between ticks (last 5m)
  quote_vol_30m       std of mid prices
"""
import os, time, glob
import numpy as np, pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

PROJECT="/home/jay/Desktop/new-model-zigzag"
DATA=f"{PROJECT}/data"
TICK_DIR=f"{PROJECT}/data/ticks"

V72L=['hurst_rs','ou_theta','entropy_rate','kramers_up','wavelet_er',
      'vwap_dist','hour_enc','dow_enc','quantum_flow','quantum_flow_h4',
      'quantum_momentum','quantum_vwap_conf','quantum_divergence','quantum_div_strength',
      'vpin','sig_quad_var','har_rv_ratio','hawkes_eta']
TICK_FEATS=['spread_mean','spread_recent_z','bid_vol_share_30m',
            'vol_imbalance_5m','signed_flow_5m','microprice_drift_5m',
            'microprice_skew_30m','tick_rate_ratio','mean_intertick_ms',
            'quote_vol_30m']
ENTRY_FEATS=V72L+TICK_FEATS+['atr_norm','hour','dow']
MAX_HOLD=60; SL_HARD=-4.0
PEAK_THR=1.5; LOSE_THR=-2.0

def load_market(swing_csv,setups_glob):
    sw=pd.read_csv(swing_csv,parse_dates=["time"])
    sw=sw.sort_values("time").drop_duplicates('time',keep='last').reset_index(drop=True)
    n=len(sw); C=sw["close"].values.astype(np.float64)
    H=sw["high"].values; Lo=sw["low"].values
    tr=np.concatenate([[H[0]-Lo[0]],np.maximum.reduce([H[1:]-Lo[1:],np.abs(H[1:]-C[:-1]),np.abs(Lo[1:]-C[:-1])])])
    atr=pd.Series(tr).rolling(14,min_periods=14).mean().values
    t2i={pd.Timestamp(t):i for i,t in enumerate(sw["time"].values)}
    setups=[pd.read_csv(f,parse_dates=["time"]) for f in sorted(glob.glob(setups_glob))]
    all_df=pd.concat(setups,ignore_index=True)
    phys=all_df[['time']+V72L].drop_duplicates('time',keep='last').sort_values('time')
    sw=pd.merge_asof(sw.sort_values('time'),phys,on='time',direction='nearest')
    for c in V72L: sw[c]=sw[c].fillna(0)
    ctx=sw[V72L].fillna(0).values.astype(np.float64)
    return n,C,Lo,H,atr,t2i,ctx

class TickCache:
    """Caches at most the current and previous day's tick parquet."""
    def __init__(self,tick_dir):
        self.dir=tick_dir; self.cache={}
    def _load_one(self,date):
        if date in self.cache: return self.cache[date]
        fn=os.path.join(self.dir,date.strftime("%Y-%m-%d")+".parquet")
        if not os.path.exists(fn):
            self.cache[date]=None; return None
        df=pd.read_parquet(fn)
        if df.index.tz is not None: df.index=df.index.tz_localize(None)
        ts=df.index.values.astype('datetime64[ns]').astype(np.int64)
        bid=df['bidPrice'].values.astype(np.float64)
        ask=df['askPrice'].values.astype(np.float64)
        bv =df['bidVolume'].values.astype(np.float64)
        av =df['askVolume'].values.astype(np.float64)
        self.cache[date]=(ts,bid,ask,bv,av)
        return self.cache[date]
    def get_window(self,t_ns,lookback_ns):
        """Return concatenated arrays for [t-lookback, t], pulling from
        today + yesterday if needed."""
        t=pd.Timestamp(t_ns,unit='ns')
        today=pd.Timestamp(t.date())
        yest=today-pd.Timedelta(days=1)
        # Prune cache to only today/yest
        for k in list(self.cache.keys()):
            if k!=today and k!=yest: del self.cache[k]
        parts=[]
        for d in (yest,today):
            r=self._load_one(d)
            if r is not None: parts.append(r)
        if not parts: return None
        ts=np.concatenate([p[0] for p in parts])
        bid=np.concatenate([p[1] for p in parts])
        ask=np.concatenate([p[2] for p in parts])
        bv=np.concatenate([p[3] for p in parts])
        av=np.concatenate([p[4] for p in parts])
        # Slice to window
        a=np.searchsorted(ts,t_ns-lookback_ns,side='left')
        b=np.searchsorted(ts,t_ns,side='right')
        return ts[a:b],bid[a:b],ask[a:b],bv[a:b],av[a:b]

def tick_feats(t_ns,d,ea,cache):
    win30=30*60_000_000_000
    win5 =5 *60_000_000_000
    r=cache.get_window(t_ns,win30)
    if r is None: return [0.0]*len(TICK_FEATS)
    ts,bid,ask,bv,av=r
    n30=len(ts)
    if n30<5 or ea<=0: return [0.0]*len(TICK_FEATS)
    spread=ask-bid
    spread_mean=float(spread.mean())/ea
    sp_std=float(spread.std())+1e-9
    spread_z=float((spread[-1]-spread.mean())/sp_std)
    tot_bv=bv.sum(); tot_av=av.sum()
    bid_share=float(tot_bv/(tot_bv+tot_av+1e-9))
    a5=np.searchsorted(ts,t_ns-win5,side='left')
    n5=n30-a5
    if n5>=2:
        bv5=bv[a5:].sum(); av5=av[a5:].sum()
        vol_imb_5=float(bv5/(bv5+av5+1e-9))
        signed_flow_5=float(bv5-av5)*d
    else:
        vol_imb_5=bid_share; signed_flow_5=0.0
    mid=(bid+ask)*0.5
    mp=(ask*bv+bid*av)/(bv+av+1e-9)
    mp_skew=float((mp-mid).mean())/ea
    if n5>=2:
        mp_drift=float(mp[-1]-mp[a5])*d/ea
    else:
        mp_drift=0.0
    rate=float(n5*6)/max(n30,1)
    if n5>=2:
        intertick=float(np.diff(ts[a5:]).mean())/1_000_000
    else:
        intertick=60_000.0
    qvol=float(np.std(mid))/ea
    return [spread_mean,spread_z,bid_share,vol_imb_5,signed_flow_5,
            mp_drift,mp_skew,rate,intertick,qvol]

def label_trade(trade,t2i,n,C,atr,d):
    tm=trade["time"]
    if tm not in t2i: return None
    ei=t2i[tm]; ep=C[ei]; ea=atr[ei]
    if not np.isfinite(ea) or ea<=0: return None
    final=trade["pnl_R"]; peak=0.0
    for k in range(1,min(MAX_HOLD,n-ei-1)+1):
        bar=ei+k; mtm=d*(C[bar]-ep)/ea
        if mtm<=SL_HARD: break
        if mtm>peak: peak=mtm
    return float(final),float(peak),ei,ea

def build(trades,t2i,n,C,atr,ctx,swing_ns,cache):
    rows=[]; lbl_any=[]; lbl_pull=[]; meta=[]
    # Sort trades by date so cache stays warm
    order=trades.sort_values("time").index.tolist()
    for idx in order:
        trade=trades.loc[idx]
        tm=trade["time"]
        if tm not in t2i: meta.append(None); continue
        d=int(trade["direction"])
        info=label_trade(trade,t2i,n,C,atr,d)
        if info is None: meta.append(None); continue
        final,peak,ei,ea=info
        v72=[float(ctx[ei,j]) for j in range(len(V72L))]
        tk=tick_feats(swing_ns[ei],d,ea,cache)
        atr_norm=float(ea/C[ei]) if C[ei]>0 else 0.0
        ts_p=pd.Timestamp(tm)
        rows.append((idx,v72+tk+[atr_norm,float(ts_p.hour),float(ts_p.dayofweek)]))
        lbl_any.append((idx,1 if final<=LOSE_THR else 0))
        lbl_pull.append((idx,1 if (peak>=PEAK_THR and final<=LOSE_THR) else 0))
        meta.append((idx,{'final':final,'peak':peak,'ei':ei}))
    # Re-order back to original trades order
    rows.sort(); lbl_any.sort(); lbl_pull.sort(); meta.sort()
    X=np.asarray([r[1] for r in rows],dtype=np.float32)
    return (X,
            np.asarray([r[1] for r in lbl_any],dtype=np.int32),
            np.asarray([r[1] for r in lbl_pull],dtype=np.int32),
            [m[1] for m in meta])

def pf_stats(p):
    s=sum(p); w=[x for x in p if x>0]; l=[x for x in p if x<=0]
    pf=sum(w)/max(-sum(l),1e-9) if l else 99.0
    return pf,s,len(w)/max(len(p),1),len(p)

def evaluate(name,swing_csv,setups_glob,trades_csv,tick_dir):
    print("\n"+"="*72); print(f"  {name}"); print("="*72)
    t0=time.time()
    n,C,Lo,H,atr,t2i,ctx=load_market(swing_csv,setups_glob)
    sw_pd=pd.read_csv(swing_csv,parse_dates=["time"])
    sw_pd=sw_pd.sort_values("time").drop_duplicates('time',keep='last').reset_index(drop=True)
    swing_ns=sw_pd["time"].values.astype("datetime64[ns]").astype(np.int64)
    trades=pd.read_csv(trades_csv,parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    print(f"  Tick dir: {tick_dir}  files: {len(glob.glob(os.path.join(tick_dir,'*.parquet')))}")
    cache=TickCache(tick_dir)

    split=trades["time"].quantile(0.70)
    trn=trades[trades["time"]<split].reset_index(drop=True)
    tst=trades[trades["time"]>=split].reset_index(drop=True)

    print("  Building entry features (memory-safe day-cache)...",end='',flush=True); t1=time.time()
    Xtr,ytr_any,ytr_pull,meta_tr=build(trn,t2i,n,C,atr,ctx,swing_ns,cache)
    Xte,yte_any,yte_pull,meta_te=build(tst,t2i,n,C,atr,ctx,swing_ns,cache)
    print(f" {time.time()-t1:.0f}s",flush=True)
    print(f"  Train: N={len(Xtr)} | losers={int(ytr_any.sum())} | pullback_sl={int(ytr_pull.sum())}")
    print(f"  Test:  N={len(Xte)} | losers={int(yte_any.sum())} | pullback_sl={int(yte_pull.sum())}")

    # Sanity: did tick features actually populate?
    tick_cols=Xtr[:,len(V72L):len(V72L)+len(TICK_FEATS)]
    nz=(np.abs(tick_cols).sum(axis=0)>0).sum()
    print(f"  Tick features populated: {nz}/{len(TICK_FEATS)} (col-sum >0)")

    base=[m['final'] for m in meta_te if m is not None]
    pf_b,s_b,wr_b,N_b=pf_stats(base)
    print(f"\n  Baseline (take all):       PF={pf_b:.2f}  Total={s_b:+.0f}R  WR={wr_b*100:.1f}%  N={N_b}")

    for label_name,ytr,yte in [("any-loser",ytr_any,yte_any),
                                ("pullback-SL",ytr_pull,yte_pull)]:
        pos=int(ytr.sum())
        if pos<10: print(f"  [{label_name}] too few positives, skipping"); continue
        sw=(len(ytr)-pos)/pos
        sample_w=np.where(ytr==1,sw,1.0).astype(np.float32)
        mdl=XGBClassifier(n_estimators=400,max_depth=4,learning_rate=0.05,
                          subsample=0.8,colsample_bytree=0.8,
                          eval_metric='logloss',verbosity=0)
        mdl.fit(Xtr,ytr,sample_weight=sample_w)
        p=mdl.predict_proba(Xte)[:,1]
        auc=roc_auc_score(yte,p) if yte.sum()>0 else 0
        pr_auc=average_precision_score(yte,p) if yte.sum()>0 else 0
        print(f"\n  [{label_name}]  ROC AUC={auc:.4f}  PR AUC={pr_auc:.4f} (rand={yte.mean():.4f}, lift={pr_auc/max(yte.mean(),1e-9):.2f}x)")

        finals=np.array([m['final'] for m in meta_te if m is not None])
        print(f"  {'thr':>6s}  {'kept':>5s}  {'PF':>5s}  {'TotalR':>8s}  {'WR%':>5s}  vs base")
        for thr in [0.30,0.40,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90]:
            keep=p<=thr
            kept=finals[keep].tolist()
            if len(kept)<20: print(f"  {thr:6.2f}  {len(kept):5d}  too few"); continue
            pf,s,wr,N=pf_stats(kept)
            marker=" ★" if (s>s_b and N>=int(0.5*N_b)) else ""
            print(f"  {thr:6.2f}  {N:5d}  {pf:5.2f}  {s:+8.0f}  {wr*100:5.1f}  {s-s_b:+.0f}R{marker}")

        fi=mdl.feature_importances_
        top=sorted(enumerate(fi),key=lambda x:-x[1])[:10]
        print(f"  Top-10 features:")
        for i,v in top:
            tag="TICK" if ENTRY_FEATS[i] in TICK_FEATS else "M5"
            print(f"    {tag:5s} {ENTRY_FEATS[i]:<28s} {v:.4f}")

    print(f"\n  Done in {time.time()-t0:.0f}s")

if __name__=="__main__":
    evaluate("Oracle XAU — Adversarial entry filter + TICKS",
             f"{DATA}/swing_v5_xauusd.csv",
             f"{DATA}/setups_*_v72l.csv",
             f"{PROJECT}/experiments/v84_rl_entry/v84_rl_trades.csv",
             f"{TICK_DIR}/xau")
    evaluate("Oracle BTC — Adversarial entry filter + TICKS",
             f"{DATA}/swing_v5_btc.csv",
             f"{DATA}/setups_*_v72l_btc.csv",
             f"{PROJECT}/experiments/v84_rl_entry/btc_rl_trades.csv",
             f"{TICK_DIR}/btc")
