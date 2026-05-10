"""Slot the new q_entry (V72L+maturity) into the existing pipeline — FAST version.

Skip the ML exit head (per-bar predict_proba was bottleneck). Use the
deployed exit policy: hard SL + v88 reverse-setup (q>0.05) + trail
(act=3R/gb=0.6) + 60-bar max. This is what's actually live in production
since v87 multi-head was removed and v88 reverse-setup was added.

Both OLD and NEW q_entry candidates go through the same exit logic.
Only difference is which trades they let through.
"""
import os, time, glob as _glob, pickle
import numpy as np, pandas as pd
from importlib.util import spec_from_file_location, module_from_spec

PROJECT="/home/jay/Desktop/new-model-zigzag"
DATA=f"{PROJECT}/data"
spec=spec_from_file_location("v89", os.path.join(os.path.dirname(os.path.abspath(__file__)),"01_optimal_stopping.py"))
v89=module_from_spec(spec); spec.loader.exec_module(v89)

V72L=v89.V72L
MATURITY=['stretch_100','stretch_200','pct_to_extreme_50']
META_FEATS=V72L+["direction","cid"]
SL_HARD=-4.0; MAX_HOLD=60; TRAIL_ACT=3.0; TRAIL_GB=0.60
HOLDOUT_START=pd.Timestamp("2024-12-12")
MIN_Q=0.3

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

def simulate_trades(meta_passed_df, t2i, n, C, atr,
                    setup_lkp=None, setup_Q=None, sw_times=None, v88_thr=0.05):
    """Hard SL + v88 reverse-setup + trail + max-hold. NO ML exit (skipped for speed)."""
    pnls=[]
    for _,s in meta_passed_df.iterrows():
        tm=s["time"]
        if tm not in t2i: continue
        ei=t2i[tm]; d=int(s["direction"]); ep=C[ei]; ea=atr[ei]
        if not np.isfinite(ea) or ea<=0: continue
        peak=0.0; max_k=min(MAX_HOLD,n-ei-1); fired=False
        for k in range(1,max_k+1):
            bar=ei+k; R=d*(C[bar]-ep)/ea
            if R<=SL_HARD: pnls.append(R); fired=True; break
            if R>peak: peak=R
            if setup_lkp is not None:
                bt=pd.Timestamp(sw_times[bar])
                if bt in setup_lkp:
                    opp=-d
                    if opp in setup_lkp[bt]:
                        idx=setup_lkp[bt][opp]
                        if setup_Q[idx]>v88_thr:
                            pnls.append(R); fired=True; break
            if peak>=TRAIL_ACT and R<=peak*(1.0-TRAIL_GB):
                pnls.append(R); fired=True; break
        if not fired:
            last=min(ei+max_k,n-1); pnls.append(d*(C[last]-ep)/ea)
    return pnls

def metrics(pnls):
    if not pnls: return None
    p=np.asarray(pnls); s=p.sum()
    pos=p[p>0]; neg=p[p<=0]
    pf=pos.sum()/max(-neg.sum(),1e-9) if len(neg)>0 else 99.0
    wr=(p>0).mean()*100
    eq=np.cumsum(p); peak=np.maximum.accumulate(eq); dd=float((peak-eq).max())
    return dict(n=len(pnls),pf=pf,total=float(s),wr=wr,dd=dd)

def evaluate(name,swing_csv,setups_glob,bundle_old_path,bundle_new_path):
    print("\n"+"="*72); print(f"  {name}"); print("="*72,flush=True)
    t0=time.time()
    n,C,atr,t2i,_=v89.load_market(swing_csv,setups_glob)
    sw_times=pd.read_csv(swing_csv,parse_dates=["time"]).sort_values("time").drop_duplicates('time',keep='last')["time"].values
    setups=load_setups_all(setups_glob)

    setup_idx=np.full(len(setups),-1,dtype=np.int64)
    for i,t in enumerate(setups['time']):
        ti=pd.Timestamp(t)
        if ti in t2i: setup_idx[i]=t2i[ti]
    keep=setup_idx>=0
    setups=setups[keep].reset_index(drop=True); setup_idx=setup_idx[keep]
    is_holdout=setups['time'].values>=HOLDOUT_START.to_datetime64()
    holdout=setups[is_holdout].reset_index(drop=True)
    holdout_idx=setup_idx[is_holdout]
    print(f"  Holdout setups: {len(holdout):,}",flush=True)

    bundle_old=pickle.load(open(bundle_old_path,"rb"))
    bundle_new=pickle.load(open(bundle_new_path,"rb"))
    q_old=bundle_old['q_entry']; q_new=bundle_new['q_entry']
    confirm_mdls=bundle_old['mdls']; confirm_thrs=bundle_old['thrs']
    meta_mdl=bundle_old['meta_mdl']; meta_thr=bundle_old.get('meta_threshold',0.775)

    # Setup_Q (for v88 reverse-setup) computed using OLD q_entry — matches live behavior
    print("  Computing setup_Q for v88 lookup...",end='',flush=True); t=time.time()
    setup_Q_full=np.full(len(setups),-9.0,dtype=np.float32)
    for cid,m in q_old.items():
        mk=(setups['old_cid']==cid).values
        if mk.sum()<1: continue
        Xc=setups.loc[mk,V72L].fillna(0).values
        setup_Q_full[mk]=m.predict(Xc)
    setup_lkp={}
    times_arr=setups.time.values; dirs_arr=setups.direction.values
    for i in range(len(setups)):
        tt=pd.Timestamp(times_arr[i]); ddv=int(dirs_arr[i])
        if tt not in setup_lkp: setup_lkp[tt]={}
        setup_lkp[tt][ddv]=i
    print(f" {time.time()-t:.0f}s",flush=True)

    # Compute V72L + maturity for holdout setups
    print("  Computing maturity features...",end='',flush=True); t=time.time()
    V72L_X=holdout[V72L].fillna(0).values.astype(np.float32)
    mat_X=np.zeros((len(holdout),len(MATURITY)),dtype=np.float32)
    for i in range(len(holdout)):
        mat_X[i]=maturity_at(int(holdout_idx[i]),int(holdout['direction'].iat[i]),C,atr)
    full_X=np.concatenate([V72L_X,mat_X],axis=1)
    cids=holdout['old_cid'].values
    print(f" {time.time()-t:.0f}s",flush=True)

    print("  Scoring holdout with both Q models...",end='',flush=True); t=time.time()
    Q_old=np.full(len(holdout),-9.0,dtype=np.float32)
    Q_new=np.full(len(holdout),-9.0,dtype=np.float32)
    for cid in q_old: Q_old[cids==cid]=q_old[cid].predict(V72L_X[cids==cid])
    for cid in q_new: Q_new[cids==cid]=q_new[cid].predict(full_X[cids==cid])
    print(f" {time.time()-t:.0f}s",flush=True)

    def run_pipeline(Q, mq_thr, label):
        passed_q = Q > mq_thr
        sub = holdout[passed_q].copy()
        sub['rule']='RL'
        rows=[]
        for cid_v in sub['cid'].unique() if 'cid' in sub.columns else sub['old_cid'].unique():
            cid_int=int(cid_v)
            mask=(sub['cid']==cid_v) if 'cid' in sub.columns else (sub['old_cid']==cid_v)
            if mask.sum()<1: continue
            key=(cid_int,'RL')
            if key not in confirm_mdls: continue
            X=sub.loc[mask,V72L].fillna(0).values
            p=confirm_mdls[key].predict_proba(X)[:,1]
            sub_pass=sub.loc[mask].copy()
            sub_pass=sub_pass[p>=confirm_thrs[key]]
            rows.append(sub_pass)
        if not rows:
            print(f"  {label} q>{mq_thr}: 0 confirmed.",flush=True); return None
        confirmed=pd.concat(rows,ignore_index=True)
        if 'cid' not in confirmed.columns: confirmed['cid']=confirmed['old_cid']
        confirmed['direction']=confirmed['direction'].astype(int)
        confirmed['cid']=confirmed['cid'].astype(int)
        meta_in=confirmed[META_FEATS].fillna(0).values
        p_meta=meta_mdl.predict_proba(meta_in)[:,1]
        meta_passed=confirmed[p_meta>=meta_thr].copy()
        # Simulate
        pnls=simulate_trades(meta_passed,t2i,n,C,atr,
                             setup_lkp=setup_lkp,setup_Q=setup_Q_full,sw_times=sw_times)
        m=metrics(pnls)
        if m:
            print(f"  {label} q>{mq_thr}: q-pass={passed_q.sum():>5d} → confirm={len(confirmed):>5d} → meta={len(meta_passed):>5d} → trades={m['n']:>4d}  PF={m['pf']:.2f}  Total={m['total']:+.0f}R  DD={m['dd']:.0f}R  WR={m['wr']:.1f}%",flush=True)
        return m

    print(f"\n  FULL-PIPELINE COMPARISON  (q_entry → confirm → meta → simulate)",flush=True)
    print(f"  Exit policy: hard SL -4R + v88 reverse-setup q>0.05 + trail act=3R gb=60% + 60-bar max",flush=True)
    print()

    print(f"  ── OLD q_entry (V72L only — current production) ──",flush=True)
    m_old=run_pipeline(Q_old, MIN_Q, "OLD")

    print(f"\n  ── NEW q_entry (V72L + maturity) — threshold sweep ──",flush=True)
    for q_thr in [0.30, 0.50, 0.70, 1.00, 1.50]:
        run_pipeline(Q_new, q_thr, f"NEW")

    print(f"\n  Done in {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__":
    evaluate("Oracle XAU — full pipeline w/ new q_entry",
             f"{DATA}/swing_v5_xauusd.csv",
             f"{DATA}/setups_*_v72l.csv",
             f"{PROJECT}/products/models/oracle_xau_validated.pkl",
             f"{PROJECT}/products/models/oracle_xau_validated_v89mat.pkl")
    evaluate("Oracle BTC — full pipeline w/ new q_entry",
             f"{DATA}/swing_v5_btc.csv",
             f"{DATA}/setups_*_v72l_btc.csv",
             f"{PROJECT}/products/models/oracle_btc_validated.pkl",
             f"{PROJECT}/products/models/oracle_btc_validated_v89mat.pkl")
