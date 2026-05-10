"""High-Q sweep + check: does NEW q_entry actually filter out stretched entries?

Two questions:
  1. What happens at very high Q thresholds (q>2.5, 3.0, 4.0, 5.0)?
  2. What's the stretch_100 distribution of trades that NEW lets through
     vs OLD? If maturity features did their job, NEW should avoid the
     high-stretch (mature/extended) entries.
"""
import os, glob as _glob, pickle
import numpy as np, pandas as pd
from importlib.util import spec_from_file_location, module_from_spec

PROJECT="/home/jay/Desktop/new-model-zigzag"
DATA=f"{PROJECT}/data"
spec_main=spec_from_file_location("v89", os.path.join(os.path.dirname(os.path.abspath(__file__)),"01_optimal_stopping.py"))
v89=module_from_spec(spec_main); spec_main.loader.exec_module(v89)
spec_p13=spec_from_file_location("p13", os.path.join(os.path.dirname(os.path.abspath(__file__)),"13_proper_full_pipeline_test.py"))
p13=module_from_spec(spec_p13); spec_p13.loader.exec_module(p13)

V72L=v89.V72L
MATURITY=p13.MATURITY
META_FEATS=p13.META_FEATS
HOLDOUT_START=p13.HOLDOUT_START

def evaluate(name,swing_csv,setups_glob,bundle_old_path,bundle_new_path,v84_trades_csv):
    print("\n"+"="*72); print(f"  {name}"); print("="*72,flush=True)
    n,C,atr,t2i,_=v89.load_market(swing_csv,setups_glob)
    sw_times=pd.read_csv(swing_csv,parse_dates=["time"]).sort_values("time").drop_duplicates('time',keep='last')["time"].values
    setups=p13.load_setups_all(setups_glob)
    setup_idx=np.full(len(setups),-1,dtype=np.int64)
    for i,t in enumerate(setups['time']):
        ti=pd.Timestamp(t)
        if ti in t2i: setup_idx[i]=t2i[ti]
    keep=setup_idx>=0
    setups=setups[keep].reset_index(drop=True); setup_idx=setup_idx[keep]
    is_holdout=setups['time'].values>=HOLDOUT_START.to_datetime64()
    holdout=setups[is_holdout].reset_index(drop=True)
    holdout_idx=setup_idx[is_holdout]

    bundle_old=pickle.load(open(bundle_old_path,"rb"))
    bundle_new=pickle.load(open(bundle_new_path,"rb"))
    q_old=bundle_old['q_entry']; q_new=bundle_new['q_entry']
    confirm_mdls=bundle_old['mdls']; confirm_thrs=bundle_old['thrs']
    meta_mdl=bundle_old['meta_mdl']; meta_thr=bundle_old.get('meta_threshold',0.775)

    # v88 setup lookup
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

    # Maturity features for holdout
    V72L_X=holdout[V72L].fillna(0).values.astype(np.float32)
    mat_X=np.zeros((len(holdout),len(MATURITY)),dtype=np.float32)
    for i in range(len(holdout)):
        mat_X[i]=p13.maturity_at(int(holdout_idx[i]),int(holdout['direction'].iat[i]),C,atr)
    full_X=np.concatenate([V72L_X,mat_X],axis=1)
    cids=holdout['old_cid'].values

    Q_old=np.full(len(holdout),-9.0,dtype=np.float32)
    Q_new=np.full(len(holdout),-9.0,dtype=np.float32)
    for cid in q_old: Q_old[cids==cid]=q_old[cid].predict(V72L_X[cids==cid])
    for cid in q_new: Q_new[cids==cid]=q_new[cid].predict(full_X[cids==cid])
    stretch_100=mat_X[:,0]
    pct_extreme=mat_X[:,2]

    # Reference baseline
    v84=pd.read_csv(v84_trades_csv,parse_dates=["time"])
    v84=v84[v84['time']>=HOLDOUT_START].reset_index(drop=True)
    v84_sim=p13.simulate_with_v88(v84,t2i,n,C,atr,setup_lkp,setup_Q_full,sw_times,v88_thr=0.05)
    v84_final=p13.apply_kill_switch(v84_sim)
    m_v84=p13.metrics(v84_final)
    print(f"\n  REFERENCE (current production): n={m_v84['n']}  PF={m_v84['pf']:.2f}  Total={m_v84['total']:+.0f}R  DD={m_v84['dd']:.0f}R  WR={m_v84['wr']:.1f}%")

    def run(Q, thr, label):
        passed_q = Q > thr
        sub = holdout[passed_q].copy()
        sub['rule']='RL'; sub['cid']=sub['old_cid']
        rows=[]
        for cid_v in sub['cid'].unique():
            cid_int=int(cid_v); mask=sub['cid']==cid_v
            if mask.sum()<1: continue
            key=(cid_int,'RL')
            if key not in confirm_mdls: continue
            X=sub.loc[mask,V72L].fillna(0).values
            p=confirm_mdls[key].predict_proba(X)[:,1]
            sub_pass=sub.loc[mask].copy()
            sub_pass=sub_pass[p>=confirm_thrs[key]]
            rows.append(sub_pass)
        if not rows: return None
        confirmed=pd.concat(rows,ignore_index=True)
        confirmed['direction']=confirmed['direction'].astype(int)
        confirmed['cid']=confirmed['cid'].astype(int)
        meta_in=confirmed[META_FEATS].fillna(0).values
        p_meta=meta_mdl.predict_proba(meta_in)[:,1]
        meta_passed=confirmed[p_meta>=meta_thr].copy()
        trades=p13.simulate_with_v88(meta_passed,t2i,n,C,atr,setup_lkp,setup_Q_full,sw_times,v88_thr=0.05)
        trades_final=p13.apply_kill_switch(trades)
        return p13.metrics(trades_final), trades_final, meta_passed

    # ── HIGH-Q SWEEP ──
    print(f"\n  ── NEW q_entry SWEEP at higher thresholds ──")
    print(f"  {'q_thr':>5s}  {'N':>5s} {'PF':>5s} {'TotalR':>7s} {'DD':>5s} {'WR%':>5s}")
    for q_thr in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0]:
        result=run(Q_new, q_thr, f"NEW q>{q_thr}")
        if result is None or result[0] is None:
            print(f"  {q_thr:5.2f}: too few trades"); continue
        m=result[0]
        marker = " ★" if (m['pf']>m_v84['pf'] and m['dd']<=m_v84['dd']) else ""
        print(f"  {q_thr:5.2f}  {m['n']:5d} {m['pf']:5.2f} {m['total']:+7.0f} {m['dd']:4.0f}R {m['wr']:5.1f}{marker}")

    # ── STRETCH DISTRIBUTION CHECK ──
    print(f"\n  ── KEY QUESTION: are stretched/top-of-leg entries filtered out? ──")
    print(f"  Compute the stretch_100 distribution of META-PASSED trades for each policy.")
    print(f"  If maturity features did their job: NEW should pass FEWER high-stretch entries.")

    # Build "trades passed each filter" sets
    def passed_trades(Q, thr):
        passed_q = Q > thr
        sub = holdout[passed_q].copy()
        sub['rule']='RL'; sub['cid']=sub['old_cid']
        rows=[]; idx_track=[]
        sub_idx=np.where(passed_q)[0]
        for j,(_,r) in enumerate(sub.iterrows()):
            idx_track.append(sub_idx[j])
        # Apply confirm
        for cid_v in sub['cid'].unique():
            cid_int=int(cid_v); mask=(sub['cid']==cid_v).values
            if mask.sum()<1: continue
            key=(cid_int,'RL')
            if key not in confirm_mdls: continue
            X=sub[V72L].fillna(0).values[mask]
            p=confirm_mdls[key].predict_proba(X)[:,1]
            keep_local=p>=confirm_thrs[key]
            local_idx=np.where(mask)[0][keep_local]
            for li in local_idx: rows.append(li)
        if not rows: return np.array([]), np.array([])
        confirmed_local=np.array(sorted(set(rows)))
        confirmed=sub.iloc[confirmed_local].copy()
        confirmed['direction']=confirmed['direction'].astype(int)
        confirmed['cid']=confirmed['cid'].astype(int)
        meta_in=confirmed[META_FEATS].fillna(0).values
        p_meta=meta_mdl.predict_proba(meta_in)[:,1]
        meta_pass_local=p_meta>=meta_thr
        sub_idx_arr=np.array(idx_track)
        final_holdout_idx=sub_idx_arr[confirmed_local[meta_pass_local]] if meta_pass_local.any() else np.array([])
        return final_holdout_idx, confirmed[meta_pass_local]['cid'].values

    print(f"\n  {'policy':<22s} {'N':>5s}  {'mean stretch_100':>16s}  {'p25':>5s} {'p50':>5s} {'p75':>5s}  {'%>10':>5s}  {'%>15':>5s}")
    for label,Q,thr in [('OLD q>0.30',Q_old,0.30),
                         ('NEW q>0.30',Q_new,0.30),
                         ('NEW q>1.00',Q_new,1.00),
                         ('NEW q>2.00',Q_new,2.00),
                         ('NEW q>3.00',Q_new,3.00)]:
        idx,_=passed_trades(Q, thr)
        if len(idx)<5: continue
        s=stretch_100[idx]
        n_pass=len(idx)
        mean=s.mean(); p25=np.percentile(s,25); p50=np.median(s); p75=np.percentile(s,75)
        pct_10=(s>10).mean()*100; pct_15=(s>15).mean()*100
        print(f"  {label:<22s} {n_pass:5d}  {mean:>16.2f}  {p25:5.2f} {p50:5.2f} {p75:5.2f}  {pct_10:5.1f} {pct_15:5.1f}")

    # Same but for pct_to_extreme_50
    print(f"\n  {'policy':<22s} {'N':>5s}  {'mean pct_to_top':>16s}  {'p25':>5s} {'p50':>5s} {'p75':>5s}  {'%>0.6':>5s} {'%>0.8':>5s}")
    for label,Q,thr in [('OLD q>0.30',Q_old,0.30),
                         ('NEW q>0.30',Q_new,0.30),
                         ('NEW q>1.00',Q_new,1.00),
                         ('NEW q>2.00',Q_new,2.00),
                         ('NEW q>3.00',Q_new,3.00)]:
        idx,_=passed_trades(Q, thr)
        if len(idx)<5: continue
        s=pct_extreme[idx]
        n_pass=len(idx)
        mean=s.mean(); p25=np.percentile(s,25); p50=np.median(s); p75=np.percentile(s,75)
        pct_06=(s>0.6).mean()*100; pct_08=(s>0.8).mean()*100
        print(f"  {label:<22s} {n_pass:5d}  {mean:>16.3f}  {p25:5.3f} {p50:5.3f} {p75:5.3f}  {pct_06:5.1f} {pct_08:5.1f}")

if __name__=="__main__":
    evaluate("Oracle XAU — high-Q + stretch distribution check",
             f"{DATA}/swing_v5_xauusd.csv",
             f"{DATA}/setups_*_v72l.csv",
             f"{PROJECT}/products/models/oracle_xau_validated.pkl",
             f"{PROJECT}/products/models/oracle_xau_validated_v89mat.pkl",
             f"{PROJECT}/experiments/v84_rl_entry/v84_rl_trades.csv")
    evaluate("Oracle BTC — high-Q + stretch distribution check",
             f"{DATA}/swing_v5_btc.csv",
             f"{DATA}/setups_*_v72l_btc.csv",
             f"{PROJECT}/products/models/oracle_btc_validated.pkl",
             f"{PROJECT}/products/models/oracle_btc_validated_v89mat.pkl",
             f"{PROJECT}/experiments/v84_rl_entry/btc_rl_trades.csv")
