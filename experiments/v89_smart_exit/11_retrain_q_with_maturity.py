"""Phase 1: retrain q_entry with V72L + 3 maturity features, full pipeline validation.

Training:
  - Same train window as v84 (all setups before 2024-12-12 holdout)
  - Features: V72L (18) + stretch_100 + stretch_200 + pct_to_extreme_50 (21 total)
  - Per-cluster XGBRegressor matching v84's hyperparams (300 est, depth 4, lr 0.05)
  - Label = pnl_R from baseline simulator (hard SL + trail + 60-bar max + v88)

Validation (apples-to-apples vs production):
  - Take the SAME post-2024-12-12 holdout setups
  - Apply NEW q_entry > 0.3 filter (same threshold as prod)
  - Simulate selected trades through full prod exit (hard SL + v88 + trail + 60-bar)
  - Compare metrics to OLD q_entry > 0.3 filter on the same setups

Note: this validates q_entry filtering quality. Downstream gates (confirm,
meta, range filter, kill-switch) are NOT applied here, so the absolute trade
counts are higher than production. The RELATIVE comparison (new vs old at
matched threshold) tells us if maturity features improve filtering.
"""
import os, time, glob as _glob, pickle
import numpy as np, pandas as pd
from xgboost import XGBRegressor
from importlib.util import spec_from_file_location, module_from_spec

PROJECT="/home/jay/Desktop/new-model-zigzag"
DATA=f"{PROJECT}/data"
spec=spec_from_file_location("v89", os.path.join(os.path.dirname(os.path.abspath(__file__)),"01_optimal_stopping.py"))
v89=module_from_spec(spec); spec.loader.exec_module(v89)

V72L=v89.V72L
MATURITY=['stretch_100','stretch_200','pct_to_extreme_50']
SL_HARD=-4.0; MAX_HOLD=60; TRAIL_ACT=3.0; TRAIL_GB=0.60
HOLDOUT_START=pd.Timestamp("2024-12-12")
MIN_CLUSTER=200

def maturity_at(t_idx,d,C,atr):
    ea=atr[t_idx]
    if not np.isfinite(ea) or ea<=0: return [0.0,0.0,0.5]
    out=[]
    for L in [100,200]:
        if t_idx>=L:
            win=C[t_idx-L:t_idx+1]
            v=(C[t_idx]-win.min())/ea if d==1 else (win.max()-C[t_idx])/ea
        else: v=0.0
        out.append(float(v))
    L=50
    if t_idx>=L:
        win=C[t_idx-L:t_idx+1]
        rng=win.max()-win.min()
        if d==1: pct=(C[t_idx]-win.min())/rng if rng>0 else 0.5
        else:    pct=(win.max()-C[t_idx])/rng if rng>0 else 0.5
    else: pct=0.5
    out.append(float(pct))
    return out

def load_setups_all(setups_glob):
    parts=[]
    for f in sorted(_glob.glob(setups_glob)):
        cid_str=os.path.basename(f).split('_')[1]
        try: old_cid=int(cid_str)
        except: continue
        df=pd.read_csv(f,parse_dates=["time"]); df['old_cid']=old_cid
        parts.append(df)
    s=pd.concat(parts,ignore_index=True)
    return s.sort_values(['time','direction']).drop_duplicates(['time','direction'],keep='first').reset_index(drop=True)

def simulate_trade(t_idx,d,n,C,atr,setup_lkp=None,setup_Q_old=None,sw_times=None):
    """Simulate a trade through full prod exit logic. Returns pnl_R."""
    ep=C[t_idx]; ea=atr[t_idx]
    if not np.isfinite(ea) or ea<=0: return None
    peak=0.0; max_k=min(MAX_HOLD,n-t_idx-1)
    for k in range(1,max_k+1):
        bar=t_idx+k; R=d*(C[bar]-ep)/ea
        if R<=SL_HARD: return R
        if R>peak: peak=R
        # v88 reverse-setup if lookup provided
        if setup_lkp is not None:
            bt=pd.Timestamp(sw_times[bar])
            if bt in setup_lkp:
                opp=-d
                if opp in setup_lkp[bt]:
                    idx=setup_lkp[bt][opp]
                    if setup_Q_old[idx]>0.05: return R
        if peak>=TRAIL_ACT and R<=peak*(1.0-TRAIL_GB): return R
    last=min(t_idx+max_k,n-1)
    return d*(C[last]-ep)/ea

def metrics(pnls):
    p=np.asarray(pnls)
    if len(p)==0: return (0,0.0,0.0,0.0,0.0)
    s=p.sum()
    pos=p[p>0]; neg=p[p<=0]
    pf=pos.sum()/max(-neg.sum(),1e-9) if len(neg)>0 else 99.0
    wr=(p>0).mean()*100
    eq=np.cumsum(p); peak=np.maximum.accumulate(eq); dd=float((peak-eq).max())
    return (len(p),pf,float(s),wr,dd)

def evaluate(name, swing_csv, setups_glob, bundle_path):
    print("\n"+"="*72); print(f"  {name}"); print("="*72)
    t0=time.time()
    n,C,atr,t2i,_=v89.load_market(swing_csv,setups_glob)
    sw_times=pd.read_csv(swing_csv,parse_dates=["time"]).sort_values("time").drop_duplicates('time',keep='last')["time"].values
    setups=load_setups_all(setups_glob)
    print(f"  Total setups: {len(setups):,}  ({setups.time.min()} → {setups.time.max()})")

    # Map setups to swing index
    setup_idx=np.full(len(setups),-1,dtype=np.int64)
    for i,t in enumerate(setups['time']):
        ti=pd.Timestamp(t)
        if ti in t2i: setup_idx[i]=t2i[ti]
    keep=setup_idx>=0
    setups=setups[keep].reset_index(drop=True); setup_idx=setup_idx[keep]
    print(f"  Setups w/ valid swing idx: {len(setups):,}")

    # Build setup_lkp (for v88 simulation) using OLD q_entry
    bundle=pickle.load(open(bundle_path,"rb"))
    q_entry_old=bundle['q_entry']
    print(f"  Loaded current v84 q_entry: cids={list(q_entry_old.keys())}")

    setup_Q_old=np.full(len(setups),-9.0,dtype=np.float32)
    for cid,m in q_entry_old.items():
        mask=(setups['old_cid']==cid).values
        if mask.sum()<1: continue
        Xc=setups.loc[mask,V72L].fillna(0).values
        setup_Q_old[mask]=m.predict(Xc)
    setup_lkp={}
    times_arr=setups.time.values; dirs_arr=setups.direction.values
    for i in range(len(setups)):
        tt=pd.Timestamp(times_arr[i]); dd=int(dirs_arr[i])
        if tt not in setup_lkp: setup_lkp[tt]={}
        setup_lkp[tt][dd]=i

    # Simulate pnl_R for each setup (used as label AND as ground truth)
    print("  Simulating pnl_R for every setup (full prod exit logic)...",end='',flush=True); t=time.time()
    pnls=np.full(len(setups),np.nan,dtype=np.float32)
    for i in range(len(setups)):
        p=simulate_trade(int(setup_idx[i]),int(dirs_arr[i]),n,C,atr,
                         setup_lkp=setup_lkp,setup_Q_old=setup_Q_old,sw_times=sw_times)
        if p is not None: pnls[i]=p
    valid=~np.isnan(pnls)
    setups=setups[valid].reset_index(drop=True); setup_idx=setup_idx[valid]
    pnls=pnls[valid]; dirs_arr=dirs_arr[valid]
    setup_Q_old=setup_Q_old[valid]
    print(f" {time.time()-t:.0f}s — {len(pnls):,} usable setups",flush=True)

    # Compute maturity features per setup
    print("  Computing maturity features per setup...",end='',flush=True); t=time.time()
    mat_X=np.zeros((len(setups),len(MATURITY)),dtype=np.float32)
    for i in range(len(setups)):
        mat_X[i]=maturity_at(int(setup_idx[i]),int(dirs_arr[i]),C,atr)
    V72L_X=setups[V72L].fillna(0).values.astype(np.float32)
    full_X=np.concatenate([V72L_X,mat_X],axis=1)
    print(f" {time.time()-t:.0f}s",flush=True)

    cids=setups['old_cid'].values
    is_train=setups['time'].values<HOLDOUT_START.to_datetime64()
    is_test =~is_train
    print(f"  Train (pre-2024-12-12): {is_train.sum():,}  | Holdout: {is_test.sum():,}")
    print(f"  Per-cluster train counts: {dict((c,int(((cids==c)&is_train).sum())) for c in sorted(set(cids)))}")

    # Train new q_entry per cluster
    print("  Training new q_entry (V72L + maturity) per cluster...",end='',flush=True); t=time.time()
    q_entry_new={}
    for cid in sorted(set(cids)):
        mask_tr=(cids==cid)&is_train
        if mask_tr.sum()<MIN_CLUSTER: continue
        m=XGBRegressor(n_estimators=300,max_depth=4,learning_rate=0.05,
                       subsample=0.8,colsample_bytree=0.8,
                       random_state=42,verbosity=0,objective='reg:squarederror')
        m.fit(full_X[mask_tr],pnls[mask_tr])
        q_entry_new[cid]=m
    print(f" {time.time()-t:.0f}s — trained for cids={list(q_entry_new.keys())}",flush=True)

    # Score holdout setups with both old and new q_entry
    Q_old_test=setup_Q_old[is_test]
    Q_new_test=np.full(int(is_test.sum()),-9.0,dtype=np.float32)
    test_idx=np.where(is_test)[0]
    test_cids=cids[is_test]
    for cid,m in q_entry_new.items():
        mt=test_cids==cid
        if mt.sum()<1: continue
        # full_X but only test rows of this cid
        full_X_test=full_X[is_test]
        Q_new_test[mt]=m.predict(full_X_test[mt])

    test_pnls=pnls[is_test]

    # Compare at production threshold (q > 0.3) and a sweep
    print(f"\n  HOLDOUT COMPARISON  (post-2024-12-12, {len(test_pnls):,} setups)")
    print(f"  Filter: setups where q_entry > thr; pnls already from full prod exit policy")
    print(f"  {'thr':>5s}  {'OLD q_entry':>30s}    {'NEW q_entry (V72L+maturity)':>32s}    Δ")
    print(f"  {'':>5s}  {'N':>5s} {'PF':>5s} {'Total':>7s} {'WR%':>5s} {'DD':>4s}    {'N':>5s} {'PF':>5s} {'Total':>7s} {'WR%':>5s} {'DD':>4s}")
    for thr in [0.0,0.10,0.20,0.30,0.40,0.50,0.70]:
        Lo=Q_old_test>thr; Ln=Q_new_test>thr
        # OLD
        a=metrics(test_pnls[Lo])
        b=metrics(test_pnls[Ln])
        marker = " ★" if (b[2]>a[2] and b[1]>=a[1]) else (" + R" if b[2]>a[2] else "")
        print(f"  {thr:5.2f}  {a[0]:5d} {a[1]:5.2f} {a[2]:+7.0f} {a[3]:5.1f} {a[4]:3.0f}R    {b[0]:5d} {b[1]:5.2f} {b[2]:+7.0f} {b[3]:5.1f} {b[4]:3.0f}R    ΔR={b[2]-a[2]:+5.0f}{marker}")

    # Save the new q_entry models for Phase 2 deployment
    out_dir=os.path.dirname(bundle_path)
    out_path=os.path.join(out_dir, os.path.basename(bundle_path).replace('.pkl','_v89mat.pkl'))
    new_bundle=dict(bundle)  # shallow copy
    new_bundle['q_entry']=q_entry_new
    new_bundle['q_entry_features']=V72L+MATURITY
    new_bundle['v89_maturity']=True
    with open(out_path,'wb') as f: pickle.dump(new_bundle,f)
    print(f"\n  Saved retrained bundle: {out_path}")
    print(f"  q_entry features: {V72L+MATURITY}")
    print(f"\n  Done in {time.time()-t0:.0f}s")

if __name__=="__main__":
    evaluate("Oracle XAU — retrain q_entry with maturity features",
             f"{DATA}/swing_v5_xauusd.csv",
             f"{DATA}/setups_*_v72l.csv",
             f"{PROJECT}/products/models/oracle_xau_validated.pkl")
    evaluate("Oracle BTC — retrain q_entry with maturity features",
             f"{DATA}/swing_v5_btc.csv",
             f"{DATA}/setups_*_v72l_btc.csv",
             f"{PROJECT}/products/models/oracle_btc_validated.pkl")
