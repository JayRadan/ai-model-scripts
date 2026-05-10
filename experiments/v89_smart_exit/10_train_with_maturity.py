"""Test: does adding maturity features to q_entry training help?

Three q_entry variants compared head-to-head on identical test setups:
  (1) V72L-only         (18 features)  — baseline, recent-trained
  (2) V72L + maturity   (21 features)  — adds stretch_100, stretch_200, pct_to_extreme_50
  (3) Hard rule on top  (variant 1 +   skip stretch_100 > 15 post-hoc)

Same harness as script 15: train on first 70% chrono, test on last 30%.
Setups limited to tick-window (2024-12-01+) for consistency w/ recency-ablation.
"""
import os, time, glob as _glob, pickle
import numpy as np, pandas as pd
from xgboost import XGBRegressor
from importlib.util import spec_from_file_location, module_from_spec

PROJECT="/home/jay/Desktop/new-model-zigzag"
DATA=f"{PROJECT}/data"
spec=spec_from_file_location("v89", os.path.join(os.path.dirname(os.path.abspath(__file__)),"01_optimal_stopping.py"))
v89=module_from_spec(spec); spec.loader.exec_module(v89)
spec15=spec_from_file_location("s15", os.path.join(os.path.dirname(os.path.abspath(__file__)),"15_tick_aware_q_entry.py").replace('v89_smart_exit','v88_exit_rl'))
s15=module_from_spec(spec15); spec15.loader.exec_module(s15)

V72L=v89.V72L
MATURITY=['stretch_100','stretch_200','pct_to_extreme_50']
MIN_CLUSTER_TRAIN=300

def maturity_feats(t_idx,d,C,atr):
    ea=atr[t_idx]
    if not np.isfinite(ea) or ea<=0: return [0.0,0.0,0.5]
    out={}
    for L in [100,200]:
        if t_idx>=L:
            win=C[t_idx-L:t_idx+1]
            v=(C[t_idx]-win.min())/ea if d==1 else (win.max()-C[t_idx])/ea
        else:
            v=0.0
        out[f'stretch_{L}']=float(v)
    L=50
    if t_idx>=L:
        win=C[t_idx-L:t_idx+1]
        rng=win.max()-win.min()
        if d==1:
            pct=(C[t_idx]-win.min())/rng if rng>0 else 0.5
        else:
            pct=(win.max()-C[t_idx])/rng if rng>0 else 0.5
    else:
        pct=0.5
    return [out['stretch_100'],out['stretch_200'],float(pct)]

def evaluate(name,swing_csv,setups_glob,bundle_path,tick_window_start):
    print("\n"+"="*72); print(f"  {name}"); print("="*72)
    t0=time.time()
    n,C,atr,t2i,_=v89.load_market(swing_csv,setups_glob)
    setups=s15.load_setups(setups_glob,tick_window_start)
    setup_idx=np.full(len(setups),-1,dtype=np.int64)
    for i,t in enumerate(setups['time']):
        ti=pd.Timestamp(t)
        if ti in t2i: setup_idx[i]=t2i[ti]
    keep=setup_idx>=0
    setups=setups[keep].reset_index(drop=True); setup_idx=setup_idx[keep]

    print("  Computing pnl_R labels...",end='',flush=True); t=time.time()
    pnls=np.full(len(setups),np.nan,dtype=np.float32)
    dirs=setups['direction'].values
    for i in range(len(setups)):
        p=s15.simulate_trade(int(setup_idx[i]),int(dirs[i]),n,C,atr)
        if p is not None: pnls[i]=p
    valid=~np.isnan(pnls)
    setups=setups[valid].reset_index(drop=True); setup_idx=setup_idx[valid]
    pnls=pnls[valid]; dirs=dirs[valid]
    print(f" {time.time()-t:.0f}s — {len(pnls):,} setups",flush=True)

    print("  Computing maturity features...",end='',flush=True); t=time.time()
    mat_X=np.zeros((len(setups),len(MATURITY)),dtype=np.float32)
    for i in range(len(setups)):
        mat_X[i]=maturity_feats(int(setup_idx[i]),int(dirs[i]),C,atr)
    print(f" {time.time()-t:.0f}s",flush=True)

    V72L_X=setups[V72L].fillna(0).values.astype(np.float32)
    full_X=np.concatenate([V72L_X,mat_X],axis=1)
    cids=setups['old_cid'].values
    split=setups['time'].quantile(0.70)
    is_train=(setups['time'].values<split)
    is_test =~is_train

    # Train both variants per cluster
    print("  Training per-cluster q_entry (V72L vs V72L+maturity)...",end='',flush=True); t=time.time()
    Q_v72=np.full(len(setups),-9.0,dtype=np.float32)
    Q_full=np.full(len(setups),-9.0,dtype=np.float32)
    for cid in sorted(set(cids)):
        mask_tr=(cids==cid)&is_train
        if mask_tr.sum()<MIN_CLUSTER_TRAIN: continue
        m1=XGBRegressor(n_estimators=300,max_depth=4,learning_rate=0.05,subsample=0.8,
                        colsample_bytree=0.8,verbosity=0,objective='reg:squarederror')
        m1.fit(V72L_X[mask_tr],pnls[mask_tr])
        m2=XGBRegressor(n_estimators=300,max_depth=4,learning_rate=0.05,subsample=0.8,
                        colsample_bytree=0.8,verbosity=0,objective='reg:squarederror')
        m2.fit(full_X[mask_tr],pnls[mask_tr])
        mask_te=(cids==cid)&is_test
        if mask_te.sum()>0:
            Q_v72[mask_te]=m1.predict(V72L_X[mask_te])
            Q_full[mask_te]=m2.predict(full_X[mask_te])
    print(f" {time.time()-t:.0f}s",flush=True)

    # Test
    test_pnls=pnls[is_test]; test_Q_v72=Q_v72[is_test]; test_Q_full=Q_full[is_test]
    test_stretch=mat_X[is_test,0]   # stretch_100 column

    print(f"\n  COMPARISON  (test={len(test_pnls):,} setups)")
    print(f"  {'thr':>5s}  {'V72L-only':>30s}    {'V72L + maturity':>30s}    {'V72L + hardRule(s>15)':>30s}")
    print(f"  {'':>5s}  {'N':>5s} {'PF':>5s} {'Total':>7s} {'WR%':>5s}    {'N':>5s} {'PF':>5s} {'Total':>7s} {'WR%':>5s}    {'N':>5s} {'PF':>5s} {'Total':>7s} {'WR%':>5s}")

    def stats(keep_mask):
        kept=test_pnls[keep_mask]
        if len(kept)<10: return (0,0.0,0.0,0.0)
        s=kept.sum(); pos=kept[kept>0]; neg=kept[kept<=0]
        pf=pos.sum()/max(-neg.sum(),1e-9); wr=(kept>0).mean()*100
        return (int(keep_mask.sum()),pf,float(s),wr)

    for thr in [0.0,0.10,0.20,0.30,0.50,0.70,1.0,1.5]:
        v72_keep=test_Q_v72>thr
        full_keep=test_Q_full>thr
        rule_keep=(test_Q_v72>thr)&(test_stretch<=15)
        a=stats(v72_keep); b=stats(full_keep); c=stats(rule_keep)
        # Mark best-by-Total-R
        sums=[a[2],b[2],c[2]]
        best=int(np.argmax(sums))
        marks=['','','']
        marks[best]=' ★'
        print(f"  {thr:5.2f}  {a[0]:5d} {a[1]:5.2f} {a[2]:+7.0f} {a[3]:5.1f}{marks[0]:>2s}    "
              f"{b[0]:5d} {b[1]:5.2f} {b[2]:+7.0f} {b[3]:5.1f}{marks[1]:>2s}    "
              f"{c[0]:5d} {c[1]:5.2f} {c[2]:+7.0f} {c[3]:5.1f}{marks[2]:>2s}")

    # Top features in V72L+maturity model (any cid)
    # Just show importance for cid=0 (Uptrend) which has largest sample
    cid_show=0
    mask_tr=(cids==cid_show)&is_train
    if mask_tr.sum()>=MIN_CLUSTER_TRAIN:
        m=XGBRegressor(n_estimators=300,max_depth=4,learning_rate=0.05,subsample=0.8,
                       colsample_bytree=0.8,verbosity=0,objective='reg:squarederror')
        m.fit(full_X[mask_tr],pnls[mask_tr])
        fi=m.feature_importances_
        names=V72L+MATURITY
        top=sorted(enumerate(fi),key=lambda x:-x[1])[:10]
        print(f"\n  Top-10 features (cid={cid_show}, V72L+maturity model):")
        for i,v in top:
            tag="MAT" if names[i] in MATURITY else "V72L"
            print(f"    {tag:>4s} {names[i]:<22s} {v:.4f}")

    print(f"\n  Done in {time.time()-t0:.0f}s")

if __name__=="__main__":
    TICK_START=pd.Timestamp("2024-12-01")
    evaluate("Oracle XAU — train with maturity vs hard rule",
             f"{DATA}/swing_v5_xauusd.csv",
             f"{DATA}/setups_*_v72l.csv",
             f"{PROJECT}/products/models/oracle_xau_validated.pkl",
             TICK_START)
    evaluate("Oracle BTC — train with maturity vs hard rule",
             f"{DATA}/swing_v5_btc.csv",
             f"{DATA}/setups_*_v72l_btc.csv",
             f"{PROJECT}/products/models/oracle_btc_validated.pkl",
             TICK_START)
