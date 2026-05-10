"""Predict ONLY the hard-SL losers (the -4R big losses) at entry time.

Different from v88 #06 (which tried to predict any-loser, AUC 0.48):
  - Target = (final_pnl <= -4R)  — only the costly losses (~9% base rate)
  - Features = V72L + cross-cluster q_entry + p_conf + p_meta + regime margin
              + time features + interactions

Hypothesis: while v84's meta gate sees (V72L + cid + direction), it doesn't
see CROSS-CLUSTER Q values. A trade where cid=4 is "best" but Q[cid=0] is
also high might be safer than one where cid=4 is the only high-Q cluster.
Similarly, regime margin to runner-up is a confidence proxy v84 doesn't
explicitly use.

Goal: AUC ≥ 0.65 on unseen → kill the predicted hard-SL trades, evaluate.
"""
import os, time, glob as _glob, pickle
import numpy as np, pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from importlib.util import spec_from_file_location, module_from_spec

PROJECT="/home/jay/Desktop/new-model-zigzag"
DATA=f"{PROJECT}/data"
spec=spec_from_file_location("v89", os.path.join(os.path.dirname(os.path.abspath(__file__)),"01_optimal_stopping.py"))
v89=module_from_spec(spec); spec.loader.exec_module(v89)
V72L=v89.V72L
SL_HARD=-4.0; MAX_HOLD=60; TRAIL_ACT=3.0; TRAIL_GB=0.60

def simulate_trade(t_idx,d,n,C,atr):
    ep=C[t_idx]; ea=atr[t_idx]
    if not np.isfinite(ea) or ea<=0: return None
    peak=0.0
    for k in range(1,min(MAX_HOLD,n-t_idx-1)+1):
        bar=t_idx+k; mtm=d*(C[bar]-ep)/ea
        if mtm<=SL_HARD: return mtm
        if mtm>peak: peak=mtm
        if peak>=TRAIL_ACT and mtm<=peak*(1.0-TRAIL_GB): return mtm
    return d*(C[t_idx+min(MAX_HOLD,n-t_idx-1)]-ep)/ea

def build_meta_features(trades,t2i,n,C,atr,ctx,bundle):
    """Compute rich features per trade including cross-cluster Q's."""
    q_entry=bundle['q_entry']
    mdls=bundle.get('mdls',{})
    meta_mdl=bundle.get('meta_mdl')

    rows=[]; labels=[]
    for _,trade in trades.iterrows():
        tm=trade["time"]
        if tm not in t2i: continue
        ei=t2i[tm]; d=int(trade["direction"]); cid=int(trade["cid"])
        ea=atr[ei]
        if not np.isfinite(ea) or ea<=0: continue

        v72_vec=ctx[ei].astype(np.float32)
        v72=v72_vec.reshape(1,-1)

        # Cross-cluster Q
        q_per_cid=np.zeros(5,dtype=np.float32)
        for c in range(5):
            if c in q_entry: q_per_cid[c]=float(q_entry[c].predict(v72)[0])
        q_max=float(q_per_cid.max())
        q_chosen=float(q_per_cid[cid])
        q_runnerup=float(np.partition(q_per_cid,-2)[-2])
        q_margin=q_chosen-q_runnerup

        # Confirm head probability
        p_conf=0.5
        if (cid,'RL') in mdls:
            try: p_conf=float(mdls[(cid,'RL')].predict_proba(v72)[0,1])
            except: pass

        # Meta head (same input v84 uses: v72 + direction + cid)
        p_meta=0.5
        if meta_mdl is not None:
            meta_in=np.concatenate([v72_vec,np.array([float(d),float(cid)],dtype=np.float32)]).reshape(1,-1)
            try: p_meta=float(meta_mdl.predict_proba(meta_in)[0,1])
            except: pass

        # Time features
        ts=pd.Timestamp(tm)
        hour=float(ts.hour); dow=float(ts.dayofweek)

        # Direction-aware q (signed by direction)
        q_signed=q_chosen if d==1 else -q_chosen

        feats=list(v72_vec)+list(q_per_cid)+[q_max,q_chosen,q_runnerup,q_margin,
                                              q_signed,p_conf,p_meta,
                                              float(d),float(cid),hour,dow,
                                              float(ea/C[ei])]
        rows.append(feats)

        # Label: hard-SL (the costly -4R losses)
        pnl=simulate_trade(ei,d,n,C,atr)
        labels.append(1 if (pnl is not None and pnl<=SL_HARD) else 0)

    F_NAMES=(V72L+[f'q_cid{c}' for c in range(5)]+
             ['q_max','q_chosen','q_runnerup','q_margin','q_signed',
              'p_conf','p_meta','direction','cid','hour','dow','atr_norm'])
    return np.asarray(rows,dtype=np.float32), np.asarray(labels,dtype=np.int32), F_NAMES

def evaluate(name,swing_csv,setups_glob,trades_csv,bundle_path):
    print("\n"+"="*72); print(f"  {name}"); print("="*72)
    n,C,atr,t2i,ctx=v89.load_market(swing_csv,setups_glob)
    trades=pd.read_csv(trades_csv,parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    split=trades["time"].quantile(0.70)
    trn=trades[trades["time"]<split].reset_index(drop=True)
    tst=trades[trades["time"]>=split].reset_index(drop=True)
    print(f"  Train: {len(trn)} trades  | Test (unseen): {len(tst)}")

    bundle=pickle.load(open(bundle_path,"rb"))

    print("  Building rich entry-time features (incl. cross-cluster Q)...",end='',flush=True); t=time.time()
    Xtr,ytr,fnames=build_meta_features(trn,t2i,n,C,atr,ctx,bundle)
    Xte,yte,_     =build_meta_features(tst,t2i,n,C,atr,ctx,bundle)
    print(f" {time.time()-t:.0f}s — {Xtr.shape[1]} features",flush=True)
    print(f"  Train: hard_sl={int(ytr.sum())} ({ytr.mean()*100:.1f}%) | Test: hard_sl={int(yte.sum())} ({yte.mean()*100:.1f}%)")

    if ytr.sum()<10:
        print("  Too few hard_sl positives in train. Aborting."); return

    # Train classifier
    pos=int(ytr.sum()); neg=int((ytr==0).sum())
    sw=np.where(ytr==1,neg/max(pos,1),1.0).astype(np.float32)
    print("  Training hard_sl predictor...",end='',flush=True); t=time.time()
    mdl=XGBClassifier(n_estimators=400,max_depth=4,learning_rate=0.05,
                      subsample=0.8,colsample_bytree=0.8,
                      eval_metric='logloss',verbosity=0,random_state=42)
    mdl.fit(Xtr,ytr,sample_weight=sw)
    print(f" {time.time()-t:.0f}s",flush=True)

    p_test=mdl.predict_proba(Xte)[:,1]
    if yte.sum()>0:
        auc=roc_auc_score(yte,p_test); pr_auc=average_precision_score(yte,p_test)
        rate=yte.mean()
        print(f"  TEST: ROC AUC={auc:.4f}  PR AUC={pr_auc:.4f} (rand={rate:.4f}, lift={pr_auc/max(rate,1e-9):.2f}x)")
    else:
        print("  No positives in test."); return

    # Train AUC for sanity (overfitting check)
    p_train=mdl.predict_proba(Xtr)[:,1]
    auc_tr=roc_auc_score(ytr,p_train); pr_tr=average_precision_score(ytr,p_train)
    print(f"  TRAIN sanity: ROC AUC={auc_tr:.3f}  PR AUC={pr_tr:.3f}")

    # Top features
    fi=mdl.feature_importances_
    top=sorted(enumerate(fi),key=lambda x:-x[1])[:12]
    print(f"\n  Top-12 features:")
    for i,v in top:
        print(f"    {fnames[i]:<22s} {v:.4f}")

    # Skip-trade policy: drop trades where p_hardsl > thr; keep rest
    print(f"\n  SKIP-TRADE POLICY (drop trades with high P(hard_sl)):")
    print(f"  {'thr':>6s} {'kept':>5s} {'dropped':>7s} {'PF':>5s} {'TotalR':>7s} {'WR%':>5s} {'sl_caught':>10s}")

    # Re-simulate full prod logic with v88 reverse-setup
    setups=load_setups(setups_glob)
    setup_Q=np.full(len(setups),-9.0,dtype=np.float32)
    for cid,m in bundle['q_entry'].items():
        mask=(setups['old_cid']==cid).values
        if mask.sum()<1: continue
        Xc=setups.loc[mask,V72L].fillna(0).values
        setup_Q[mask]=m.predict(Xc)
    setup_lkp={}
    times_arr=setups.time.values; dirs_arr=setups.direction.values
    for i in range(len(setups)):
        tt=pd.Timestamp(times_arr[i]); dd=int(dirs_arr[i])
        if tt not in setup_lkp: setup_lkp[tt]={}
        setup_lkp[tt][dd]=i
    sw_times=pd.read_csv(swing_csv,parse_dates=["time"]).sort_values("time").drop_duplicates('time',keep='last')["time"].values

    def sim_kept(mask_keep):
        pnls=[]; sl_count=0
        kept_trades=tst[mask_keep].reset_index(drop=True)
        for ti,(_,trade) in enumerate(kept_trades.iterrows()):
            tm=trade["time"]
            if tm not in t2i: continue
            ei=t2i[tm]; d=int(trade["direction"]); ep=C[ei]; ea=atr[ei]
            if not np.isfinite(ea) or ea<=0: continue
            peak=0.0; max_k=min(MAX_HOLD,n-ei-1); fired=False
            for k in range(1,max_k+1):
                bar=ei+k; R=d*(C[bar]-ep)/ea
                if R<=SL_HARD: pnls.append(R); sl_count+=1; fired=True; break
                if R>peak: peak=R
                bt=pd.Timestamp(sw_times[bar])
                if bt in setup_lkp:
                    opp=-d
                    if opp in setup_lkp[bt]:
                        idx=setup_lkp[bt][opp]
                        if setup_Q[idx]>0.05:
                            pnls.append(R); fired=True; break
                if peak>=TRAIL_ACT and R<=peak*(1.0-TRAIL_GB):
                    pnls.append(R); fired=True; break
            if not fired:
                last=min(ei+max_k,n-1); pnls.append(d*(C[last]-ep)/ea)
        return pnls, sl_count

    # Baseline = take all trades (current prod)
    base_pnls,base_sl=sim_kept(np.ones(len(tst),dtype=bool))
    s_b=sum(base_pnls); pos=[x for x in base_pnls if x>0]; neg=[x for x in base_pnls if x<=0]
    pf_b=sum(pos)/max(-sum(neg),1e-9); wr_b=len(pos)/len(base_pnls)
    eq=np.cumsum(base_pnls); peak_eq=np.maximum.accumulate(eq); dd_b=float((peak_eq-eq).max())

    print(f"  ALL  (baseline)         {len(base_pnls):5d} {0:7d}  {pf_b:5.2f} {s_b:+7.0f} {wr_b*100:5.1f} {base_sl:10d}")

    # We need test-trade index → p_test ordering.
    # Filter on p_test to skip predicted-hardsl trades
    for thr in [0.30,0.40,0.50,0.60,0.70,0.80,0.90]:
        mask_keep=p_test<=thr
        if mask_keep.sum()<20:
            print(f"  thr={thr:.2f}: too few kept ({mask_keep.sum()})"); continue
        kept_pnls,sl_count=sim_kept(mask_keep)
        s=sum(kept_pnls); pos=[x for x in kept_pnls if x>0]; neg=[x for x in kept_pnls if x<=0]
        pf_=sum(pos)/max(-sum(neg),1e-9); wr=len(pos)/len(kept_pnls)
        eq=np.cumsum(kept_pnls); peak_eq=np.maximum.accumulate(eq); dd_=float((peak_eq-eq).max())
        # How many ACTUAL hard_sl trades did we successfully drop?
        sl_dropped=int(((p_test>thr) & (yte==1)).sum())
        sl_total=int(yte.sum())
        marker = " ★" if (s>=s_b*0.95 and dd_<dd_b) else ""
        print(f"  thr={thr:.2f} {len(kept_pnls):5d} {len(tst)-len(kept_pnls):7d} {pf_:5.2f} {s:+7.0f} {wr*100:5.1f}  caught {sl_dropped}/{sl_total} SL  DD={dd_:.0f}R{marker}")

def load_setups(setups_glob):
    parts=[]
    for f in sorted(_glob.glob(setups_glob)):
        cid_str=os.path.basename(f).split('_')[1]
        try: old_cid=int(cid_str)
        except: continue
        df=pd.read_csv(f,parse_dates=["time"]); df['old_cid']=old_cid
        parts.append(df)
    s=pd.concat(parts,ignore_index=True)
    return s.sort_values(['time','direction']).drop_duplicates(['time','direction'],keep='first').reset_index(drop=True)

if __name__=="__main__":
    evaluate("Oracle XAU — hard-SL predictor",
             f"{DATA}/swing_v5_xauusd.csv",
             f"{DATA}/setups_*_v72l.csv",
             f"{PROJECT}/experiments/v84_rl_entry/v84_rl_trades.csv",
             f"{PROJECT}/products/models/oracle_xau_validated.pkl")
    evaluate("Oracle BTC — hard-SL predictor",
             f"{DATA}/swing_v5_btc.csv",
             f"{DATA}/setups_*_v72l_btc.csv",
             f"{PROJECT}/experiments/v84_rl_entry/btc_rl_trades.csv",
             f"{PROJECT}/products/models/oracle_btc_validated.pkl")
