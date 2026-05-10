"""v88: Train a fresh q_entry with V72L + tick features.

Hypothesis: tick aggregates (spread, signed flow, microprice drift, etc.)
at setup-formation time carry information about the setup's near-term
PnL that V72L doesn't capture. If a tick-aware q_entry filters trades
better than v84's V72L-only q_entry on the same unseen window, deploy it
— and reuse the same model inside the v88 reverse-setup exit logic
(replacing the existing q_entry call in decide_exit).

Procedure:
  1. Load all setups in the tick window (2024-12-01+)
  2. For each setup, compute pnl_R via simple simulator (hard SL @ -4R,
     trail act=3.0/gb=0.6, 60-bar max).
  3. For each setup, compute 10 tick aggregates at the setup time.
  4. Chrono 70/30 split.
  5. Train two q_entry variants per cluster:
       (A) V72L-only      (18 features) — baseline matching v84 architecture
       (B) V72L + ticks   (28 features) — proposed
  6. Sweep Q-thresholds, simulate filtered trades on the unseen 30%.
  7. Compare PF / Total R / N for both variants at matched thresholds.

Cluster sparsity guard: skip any cluster with <300 train setups.
"""
import os, time, glob as _glob
import numpy as np, pandas as pd
from xgboost import XGBRegressor

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
MAX_HOLD=60; SL_HARD=-4.0
TRAIL_ACT=3.0; TRAIL_GB=0.60
MIN_CLUSTER_TRAIN=300

# --- TickCache (memory-safe, only 2 days resident) -----------------
class TickCache:
    def __init__(self,tick_dir):
        self.dir=tick_dir; self.cache={}
    def _load(self,date):
        if date in self.cache: return self.cache[date]
        fn=os.path.join(self.dir,date.strftime("%Y-%m-%d")+".parquet")
        if not os.path.exists(fn):
            self.cache[date]=None; return None
        df=pd.read_parquet(fn)
        if df.index.tz is not None: df.index=df.index.tz_localize(None)
        ts=df.index.values.astype('datetime64[ns]').astype(np.int64)
        self.cache[date]=(ts,df['bidPrice'].values.astype(np.float64),
                          df['askPrice'].values.astype(np.float64),
                          df['bidVolume'].values.astype(np.float64),
                          df['askVolume'].values.astype(np.float64))
        return self.cache[date]
    def window(self,t_ns,lookback_ns):
        t=pd.Timestamp(t_ns,unit='ns')
        today=pd.Timestamp(t.date()); yest=today-pd.Timedelta(days=1)
        for k in list(self.cache.keys()):
            if k!=today and k!=yest: del self.cache[k]
        parts=[]
        for d in (yest,today):
            r=self._load(d)
            if r is not None: parts.append(r)
        if not parts: return None
        ts=np.concatenate([p[0] for p in parts])
        bid=np.concatenate([p[1] for p in parts])
        ask=np.concatenate([p[2] for p in parts])
        bv=np.concatenate([p[3] for p in parts])
        av=np.concatenate([p[4] for p in parts])
        a=np.searchsorted(ts,t_ns-lookback_ns,side='left')
        b=np.searchsorted(ts,t_ns,side='right')
        return ts[a:b],bid[a:b],ask[a:b],bv[a:b],av[a:b]

def tick_feats(t_ns,d,ea,cache):
    win30=30*60_000_000_000; win5=5*60_000_000_000
    r=cache.window(t_ns,win30)
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
    if n5>=2: mp_drift=float(mp[-1]-mp[a5])*d/ea
    else: mp_drift=0.0
    rate=float(n5*6)/max(n30,1)
    if n5>=2: intertick=float(np.diff(ts[a5:]).mean())/1_000_000
    else: intertick=60_000.0
    qvol=float(np.std(mid))/ea
    return [spread_mean,spread_z,bid_share,vol_imb_5,signed_flow_5,
            mp_drift,mp_skew,rate,intertick,qvol]

# --- Setup loading + simulation -----------------------------------
def load_market(swing_csv):
    sw=pd.read_csv(swing_csv,parse_dates=["time"])
    sw=sw.sort_values("time").drop_duplicates('time',keep='last').reset_index(drop=True)
    n=len(sw); C=sw["close"].values.astype(np.float64)
    H=sw["high"].values; Lo=sw["low"].values
    tr=np.concatenate([[H[0]-Lo[0]],np.maximum.reduce([H[1:]-Lo[1:],np.abs(H[1:]-C[:-1]),np.abs(Lo[1:]-C[:-1])])])
    atr=pd.Series(tr).rolling(14,min_periods=14).mean().values
    t2i={pd.Timestamp(t):i for i,t in enumerate(sw["time"].values)}
    swing_ns=sw["time"].values.astype("datetime64[ns]").astype(np.int64)
    return n,C,atr,t2i,swing_ns

def load_setups(setups_glob,tick_window_start):
    parts=[]
    for f in sorted(_glob.glob(setups_glob)):
        cid_str=os.path.basename(f).split('_')[1]
        try: old_cid=int(cid_str)
        except: continue
        df=pd.read_csv(f,parse_dates=["time"])
        df['old_cid']=old_cid
        parts.append(df)
    s=pd.concat(parts,ignore_index=True)
    s=s[s['time']>=tick_window_start].reset_index(drop=True)
    return s

def simulate_trade(t_idx,d,n,C,atr):
    """Walk forward applying hard SL + trail + 60-bar max. Returns pnl_R."""
    ep=C[t_idx]; ea=atr[t_idx]
    if not np.isfinite(ea) or ea<=0: return None
    peak=0.0; max_k=min(MAX_HOLD,n-t_idx-1)
    for k in range(1,max_k+1):
        bar=t_idx+k; mtm=d*(C[bar]-ep)/ea
        if mtm<=SL_HARD: return mtm
        if mtm>peak: peak=mtm
        if peak>=TRAIL_ACT and mtm<=peak*(1.0-TRAIL_GB):
            return mtm
    last=min(t_idx+max_k,n-1)
    return d*(C[last]-ep)/ea

def pf_stats(p):
    s=sum(p); w=[x for x in p if x>0]; l=[x for x in p if x<=0]
    pf=sum(w)/max(-sum(l),1e-9) if l else 99.0
    return pf,s,len(w)/max(len(p),1)

def evaluate(name,swing_csv,setups_glob,tick_dir,tick_window_start):
    print("\n"+"="*72); print(f"  {name}"); print("="*72)
    t0=time.time()
    n,C,atr,t2i,swing_ns=load_market(swing_csv)
    setups=load_setups(setups_glob,tick_window_start)
    print(f"  Setups in tick window: {len(setups):,}")

    # Map each setup to its swing index + ATR
    setup_idx=np.full(len(setups),-1,dtype=np.int64)
    for i,t in enumerate(setups['time']):
        ti=pd.Timestamp(t)
        if ti in t2i: setup_idx[i]=t2i[ti]
    keep=setup_idx>=0
    setups=setups[keep].reset_index(drop=True)
    setup_idx=setup_idx[keep]
    print(f"  Setups with valid swing idx: {len(setups):,}")

    # Compute pnl_R label per setup (simulate baseline policy)
    print("  Computing pnl_R labels via simulator...",end='',flush=True); t=time.time()
    pnls=np.full(len(setups),np.nan,dtype=np.float32)
    dirs=setups['direction'].values
    for i in range(len(setups)):
        p=simulate_trade(int(setup_idx[i]),int(dirs[i]),n,C,atr)
        if p is not None: pnls[i]=p
    valid=~np.isnan(pnls)
    setups=setups[valid].reset_index(drop=True)
    setup_idx=setup_idx[valid]; pnls=pnls[valid]
    print(f" {time.time()-t:.0f}s — {len(pnls):,} trades, mean={pnls.mean():+.2f}R",flush=True)

    # Compute tick features at each setup time
    print("  Computing tick features per setup...",end='',flush=True); t=time.time()
    cache=TickCache(tick_dir)
    tick_X=np.zeros((len(setups),len(TICK_FEATS)),dtype=np.float32)
    # Sort by date so cache stays warm
    order=np.argsort(setups['time'].values)
    for o in order:
        ei=int(setup_idx[o])
        ea=float(atr[ei]) if np.isfinite(atr[ei]) else 1.0
        d=int(dirs[o])
        tick_X[o]=tick_feats(int(swing_ns[ei]),d,ea,cache)
    print(f" {time.time()-t:.0f}s",flush=True)

    # Chrono split 70/30 on setup time
    split=setups['time'].quantile(0.70)
    is_train=setups['time'].values<split
    is_test =~is_train
    print(f"  Train: {is_train.sum():,} setups before {split}")
    print(f"  Test:  {is_test.sum():,} setups (unseen)")

    # Per-cluster training
    cids=setups['old_cid'].values
    is_train_arr=is_train.values if hasattr(is_train,'values') else is_train
    cluster_counts={c:int(((cids==c)&is_train_arr).sum()) for c in sorted(set(cids))}
    is_test_arr=is_test.values if hasattr(is_test,'values') else is_test
    print(f"\n  Per-cluster train counts: {cluster_counts}")

    V72L_X=setups[V72L].fillna(0).values.astype(np.float32)
    full_X=np.concatenate([V72L_X,tick_X],axis=1)
    print(f"  Feature shapes: V72L={V72L_X.shape} | full={full_X.shape}")

    # Train per-cluster
    print(f"\n  Training per-cluster q_entry models...")
    models_v72={}; models_full={}
    for cid,n_tr in cluster_counts.items():
        if n_tr<MIN_CLUSTER_TRAIN:
            print(f"    cid={cid}: only {n_tr} train, skipping (will use ZERO Q)")
            continue
        mask_tr=(cids==cid)&is_train_arr
        m1=XGBRegressor(n_estimators=300,max_depth=4,learning_rate=0.05,
                        subsample=0.8,colsample_bytree=0.8,verbosity=0,objective='reg:squarederror')
        m1.fit(V72L_X[mask_tr],pnls[mask_tr])
        m2=XGBRegressor(n_estimators=300,max_depth=4,learning_rate=0.05,
                        subsample=0.8,colsample_bytree=0.8,verbosity=0,objective='reg:squarederror')
        m2.fit(full_X[mask_tr],pnls[mask_tr])
        models_v72[cid]=m1; models_full[cid]=m2
        print(f"    cid={cid}: trained both models on {n_tr:,} setups")

    # Predict Q on test
    Q_v72=np.full(len(setups),-9.0,dtype=np.float32)
    Q_full=np.full(len(setups),-9.0,dtype=np.float32)
    for cid in models_v72:
        mask_te=(cids==cid)&is_test_arr
        if mask_te.sum()<1: continue
        Q_v72[mask_te]=models_v72[cid].predict(V72L_X[mask_te])
        Q_full[mask_te]=models_full[cid].predict(full_X[mask_te])

    test_pnls=pnls[is_test_arr]
    test_Q_v72=Q_v72[is_test_arr]
    test_Q_full=Q_full[is_test_arr]
    n_test=len(test_pnls)

    # Sweep Q-thresholds for each variant
    print(f"\n  TEST RESULTS — chronological 30% holdout ({n_test:,} setups)")
    print(f"  {'thr':>5s}  {'V72L-only':>32s}    {'V72L + ticks':>32s}")
    print(f"  {'-'*5}  {'-'*32}    {'-'*32}")
    print(f"  {'':>5s}  {'N':>5s} {'PF':>5s} {'Total':>7s} {'WR%':>5s}    {'N':>5s} {'PF':>5s} {'Total':>7s} {'WR%':>5s}")

    for thr in [0.0,0.1,0.2,0.3,0.5,0.7,1.0,1.5,2.0]:
        rows=[]
        for variant,Q in [("v72l",test_Q_v72),("full",test_Q_full)]:
            keep=Q>thr
            if keep.sum()<10:
                rows.append((0,0.0,0.0,0.0))
                continue
            kept_pnls=test_pnls[keep].tolist()
            pf,s,wr=pf_stats(kept_pnls)
            rows.append((int(keep.sum()),pf,s,wr))
        n1,p1,s1,w1=rows[0]; n2,p2,s2,w2=rows[1]
        marker = " ★" if (p2>p1 and s2>=s1*0.95) else ""
        print(f"  {thr:5.2f}  {n1:5d} {p1:5.2f} {s1:+7.0f} {w1*100:5.1f}    {n2:5d} {p2:5.2f} {s2:+7.0f} {w2*100:5.1f}{marker}")

    # Feature importance for the tick model — see what got used
    print(f"\n  Top features in V72L+ticks model (cid=0 if available):")
    if 0 in models_full:
        fi=models_full[0].feature_importances_
        names=V72L+TICK_FEATS
        top=sorted(enumerate(fi),key=lambda x:-x[1])[:12]
        for i,v in top:
            tag="TICK" if names[i] in TICK_FEATS else "M5"
            print(f"    {tag:5s} {names[i]:<28s} {v:.4f}")

    print(f"\n  Done in {time.time()-t0:.0f}s")

if __name__=="__main__":
    TICK_START=pd.Timestamp("2024-12-01")
    evaluate("Oracle XAU — fresh q_entry: V72L vs V72L+ticks",
             f"{DATA}/swing_v5_xauusd.csv",
             f"{DATA}/setups_*_v72l.csv",
             f"{TICK_DIR}/xau",
             TICK_START)
    evaluate("Oracle BTC — fresh q_entry: V72L vs V72L+ticks",
             f"{DATA}/swing_v5_btc.csv",
             f"{DATA}/setups_*_v72l_btc.csv",
             f"{TICK_DIR}/btc",
             TICK_START)
