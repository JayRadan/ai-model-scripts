"""v88: Ablation — does the +5-7% lift come from RECENCY or from TICKS?

Three q_entry variants tested on the same test setups + same simulator:
  (1) PROD       — load v84 q_entry from oracle_*_validated.pkl
                   (trained on full historical data, V72L-only)
  (2) RECENT_V72L — train fresh on tick-window setups (12k XAU / 17k BTC),
                   V72L-only — answers "is recent data alone better?"
  (3) RECENT_FULL — train fresh on tick-window setups, V72L + 10 ticks
                   (this was script 15's winner)

If (2) ≈ (1)  → recency doesn't matter; ticks are doing all the lift → (A) deploy
If (2) > (1)  → recency helps too; cheap option = retrain (B)
If (2) ≈ (3)  → ticks don't actually help → no point in tick infra at all
"""
import os, time, pickle, glob as _glob
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

# Reuse helpers from script 15 (TickCache, tick_feats, etc.)
from importlib.util import spec_from_file_location, module_from_spec
spec=spec_from_file_location("s15", os.path.join(os.path.dirname(os.path.abspath(__file__)), "15_tick_aware_q_entry.py"))
s15=module_from_spec(spec); spec.loader.exec_module(s15)

def evaluate(name,swing_csv,setups_glob,tick_dir,bundle_path,tick_window_start):
    print("\n"+"="*72); print(f"  {name}"); print("="*72)
    t0=time.time()
    n,C,atr,t2i,swing_ns=s15.load_market(swing_csv)
    setups=s15.load_setups(setups_glob,tick_window_start)

    # Map setups → swing idx, drop misses
    setup_idx=np.full(len(setups),-1,dtype=np.int64)
    for i,t in enumerate(setups['time']):
        ti=pd.Timestamp(t)
        if ti in t2i: setup_idx[i]=t2i[ti]
    keep=setup_idx>=0
    setups=setups[keep].reset_index(drop=True)
    setup_idx=setup_idx[keep]

    # Compute pnl_R labels
    print("  Simulating pnl_R for each setup...",end='',flush=True); t=time.time()
    pnls=np.full(len(setups),np.nan,dtype=np.float32)
    dirs=setups['direction'].values
    for i in range(len(setups)):
        p=s15.simulate_trade(int(setup_idx[i]),int(dirs[i]),n,C,atr)
        if p is not None: pnls[i]=p
    valid=~np.isnan(pnls)
    setups=setups[valid].reset_index(drop=True)
    setup_idx=setup_idx[valid]; pnls=pnls[valid]; dirs=dirs[valid]
    print(f" {time.time()-t:.0f}s — {len(pnls):,} setups",flush=True)

    # Tick features
    print("  Computing tick features...",end='',flush=True); t=time.time()
    cache=s15.TickCache(tick_dir)
    tick_X=np.zeros((len(setups),len(TICK_FEATS)),dtype=np.float32)
    order=np.argsort(setups['time'].values)
    for o in order:
        ei=int(setup_idx[o])
        ea=float(atr[ei]) if np.isfinite(atr[ei]) else 1.0
        d=int(dirs[o])
        tick_X[o]=s15.tick_feats(int(swing_ns[ei]),d,ea,cache)
    print(f" {time.time()-t:.0f}s",flush=True)

    V72L_X=setups[V72L].fillna(0).values.astype(np.float32)
    full_X=np.concatenate([V72L_X,tick_X],axis=1)
    cids=setups['old_cid'].values

    # Chrono split
    split=setups['time'].quantile(0.70)
    is_train=(setups['time'].values<split)
    is_test =~is_train

    # ── Variant 1: PROD q_entry ──
    print("\n  Loading PROD q_entry...",end='',flush=True)
    bundle=pickle.load(open(bundle_path,"rb"))
    q_prod=bundle['q_entry']
    print(f" cids={list(q_prod.keys())}",flush=True)
    Q_prod=np.full(len(setups),-9.0,dtype=np.float32)
    for cid in q_prod:
        mask=(cids==cid)&is_test
        if mask.sum()<1: continue
        Q_prod[mask]=q_prod[cid].predict(V72L_X[mask])

    # ── Variant 2 + 3: train fresh on recent ──
    Q_v72=np.full(len(setups),-9.0,dtype=np.float32)
    Q_full=np.full(len(setups),-9.0,dtype=np.float32)
    print(f"  Training fresh per-cluster q_entry models...")
    for cid in sorted(set(cids)):
        mask_tr=(cids==cid)&is_train
        n_tr=int(mask_tr.sum())
        if n_tr<MIN_CLUSTER_TRAIN:
            print(f"    cid={cid}: only {n_tr} train, skipping (Q stays -9)")
            continue
        m1=XGBRegressor(n_estimators=300,max_depth=4,learning_rate=0.05,
                        subsample=0.8,colsample_bytree=0.8,verbosity=0,objective='reg:squarederror')
        m1.fit(V72L_X[mask_tr],pnls[mask_tr])
        m2=XGBRegressor(n_estimators=300,max_depth=4,learning_rate=0.05,
                        subsample=0.8,colsample_bytree=0.8,verbosity=0,objective='reg:squarederror')
        m2.fit(full_X[mask_tr],pnls[mask_tr])
        mask_te=(cids==cid)&is_test
        if mask_te.sum()>0:
            Q_v72[mask_te]=m1.predict(V72L_X[mask_te])
            Q_full[mask_te]=m2.predict(full_X[mask_te])
        print(f"    cid={cid}: trained on {n_tr:,}")

    test_pnls=pnls[is_test]
    test_Q_prod=Q_prod[is_test]
    test_Q_v72 =Q_v72[is_test]
    test_Q_full=Q_full[is_test]

    print(f"\n  THREE-WAY COMPARISON  (test={len(test_pnls):,} setups)")
    print(f"  {'thr':>5s}  {'PROD (v84 historical)':>26s}  {'RECENT V72L-only':>26s}  {'RECENT V72L+ticks':>26s}")
    print(f"  {'':>5s}  {'N':>5s} {'PF':>5s} {'TotalR':>7s}    {'N':>5s} {'PF':>5s} {'TotalR':>7s}    {'N':>5s} {'PF':>5s} {'TotalR':>7s}")

    for thr in [0.0,0.10,0.20,0.30,0.50,0.70,1.0,1.5]:
        cells=[]
        for Q in (test_Q_prod,test_Q_v72,test_Q_full):
            keep=Q>thr
            if keep.sum()<10:
                cells.append((0,0.0,0.0)); continue
            kept=test_pnls[keep].tolist()
            pf,s,_=s15.pf_stats(kept)
            cells.append((int(keep.sum()),pf,s))
        marks=[]
        # Mark PROD column with ←, mark best-by-R among the three
        best=max(range(3),key=lambda i:cells[i][2])
        for i,c in enumerate(cells):
            n_,pf_,s_=c
            tag=" ★" if i==best else ""
            marks.append(f"{n_:5d} {pf_:5.2f} {s_:+7.0f}{tag}")
        print(f"  {thr:5.2f}  {marks[0]:>27s}  {marks[1]:>27s}  {marks[2]:>27s}")

    # Summarize at the threshold most relevant to deploy decision
    summary_thr=0.30  # production currently uses min_q=0.3 for v84
    print(f"\n  AT MIN_Q=0.30 (production gate):")
    for label,Q in [("PROD v84",test_Q_prod),("RECENT V72L-only",test_Q_v72),("RECENT V72L+ticks",test_Q_full)]:
        keep=Q>summary_thr
        if keep.sum()<10:
            print(f"    {label}: too few"); continue
        kept=test_pnls[keep].tolist()
        pf,s,wr=s15.pf_stats(kept)
        print(f"    {label:>22s}: N={int(keep.sum()):4d}  PF={pf:.3f}  Total={s:+.0f}R  WR={wr*100:.1f}%")

    print(f"\n  Done in {time.time()-t0:.0f}s")

if __name__=="__main__":
    TICK_START=pd.Timestamp("2024-12-01")
    evaluate("Oracle XAU — recency vs ticks ablation",
             f"{DATA}/swing_v5_xauusd.csv",
             f"{DATA}/setups_*_v72l.csv",
             f"{TICK_DIR}/xau",
             f"{PROJECT}/products/models/oracle_xau_validated.pkl",
             TICK_START)
    evaluate("Oracle BTC — recency vs ticks ablation",
             f"{DATA}/swing_v5_btc.csv",
             f"{DATA}/setups_*_v72l_btc.csv",
             f"{TICK_DIR}/btc",
             f"{PROJECT}/products/models/oracle_btc_validated.pkl",
             TICK_START)
