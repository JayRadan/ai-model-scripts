"""v89 final — best-gate combo (Q-dominates + deep-loser) vs current production.

Test policies:
  A) BASELINE-40           hard SL + trail + 40-bar max     (the v89 ablation baseline)
  B) PROD-60               hard SL + trail + 60-bar max     (closer to live prod, no v88)
  C) PROD-60 + v88         hard SL + trail + 60-bar + reverse-setup exit (current live)
  D) v89-Q                 hard SL + trail + Q_dominates(eps=1.0)
  E) v89-Q + deep-loser    Q_dominates + deep-loser at R<=-3.5,rec<=0.05
  F) v89-Q + v88           Q_dominates layered ON TOP of v88 reverse-setup

Goal: figure out if v89 Q-stopping
  - replaces v88 (D vs C),
  - or adds to v88 (F vs C).
"""
import os, time, glob as _glob, pickle
import numpy as np, pandas as pd
from importlib.util import spec_from_file_location, module_from_spec

PROJECT="/home/jay/Desktop/new-model-zigzag"
DATA=f"{PROJECT}/data"
spec=spec_from_file_location("v89", os.path.join(os.path.dirname(os.path.abspath(__file__)),"01_optimal_stopping.py"))
v89=module_from_spec(spec); spec.loader.exec_module(v89)
V72L=v89.V72L; STATE_FEATS=v89.STATE_FEATS; SL_HARD=-4.0

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

def state_at(bar,d,ep,ea,C,ctx,k,peak,max_hold):
    R=d*(C[bar]-ep)/ea
    c3=C[max(0,bar-3)]; c10=C[max(0,bar-10)]
    mom3=d*(C[bar]-c3)/ea if bar>=3 else 0.0
    mom10=d*(C[bar]-c10)/ea if bar>=10 else 0.0
    vol10=float(np.std([d*(C[max(0,bar-j)]-C[max(0,bar-j-1)])/ea for j in range(min(10,bar))])) if bar>5 else 0.0
    v72=[float(ctx[bar,j]) for j in range(len(V72L))]
    s={'current_R':float(R),'peak_R':float(peak),'drawdown_from_peak':float(peak-R),
       'bars_in_trade':float(k),'bars_remaining':float(max_hold-k),
       'mom_3bar':float(mom3),'mom_10bar':float(mom10),'vol_10bar':vol10,
       'dist_to_SL':float(R-SL_HARD),'dist_to_TP':float(2.0-R)}
    for fname,fv in zip(V72L,v72): s[fname]=fv
    s['_features']=np.asarray([s[f] for f in STATE_FEATS],dtype=np.float32)
    return s,R

def simulate(trades,t2i,n,C,atr,ctx,policy_id,
             max_hold=40,
             trail_act=3.0,trail_gb=0.60,
             q_ens=None, rec_clf=None,
             setup_lkp=None, setup_Q=None, sw_times=None,
             v88_thr=0.05, q_eps=1.0, deep_R=-3.5, deep_rec=0.05,
             use_q=False, use_deep=False, use_v88=False):
    pnls=[]; reasons={}
    for ti,(_,trade) in enumerate(trades.iterrows()):
        tm=trade["time"]
        if tm not in t2i: continue
        ei=t2i[tm]; d=int(trade["direction"]); ep=C[ei]; ea=atr[ei]
        if not np.isfinite(ea) or ea<=0: continue
        peak=0.0; max_k=min(max_hold,n-ei-1); fired=False
        for k in range(1,max_k+1):
            bar=ei+k
            R=d*(C[bar]-ep)/ea
            if R<=SL_HARD: pnls.append(R); reasons['hard_sl']=reasons.get('hard_sl',0)+1; fired=True; break
            if R>peak: peak=R

            # v88 reverse-setup exit
            if use_v88 and setup_lkp is not None:
                bt=pd.Timestamp(sw_times[bar])
                if bt in setup_lkp:
                    opp=-d
                    if opp in setup_lkp[bt]:
                        idx=setup_lkp[bt][opp]
                        if setup_Q[idx]>v88_thr:
                            pnls.append(R); reasons['v88_rev_setup']=reasons.get('v88_rev_setup',0)+1; fired=True; break
            # Q-dominates
            if use_q and q_ens is not None:
                s,_=state_at(bar,d,ep,ea,C,ctx,k,peak,max_hold)
                preds=np.array([m.predict(s['_features'].reshape(1,-1))[0] for m in q_ens])
                q_hold=float(np.median(preds))   # p50 (matches best-of-V1 finding)
                if R>=q_hold+q_eps:
                    pnls.append(R); reasons['q_dominates']=reasons.get('q_dominates',0)+1; fired=True; break
                if use_deep and R<=deep_R and rec_clf is not None:
                    p_rec=float(rec_clf.predict_proba(s['_features'].reshape(1,-1))[0,1])
                    if p_rec<=deep_rec:
                        pnls.append(R); reasons['deep_loser']=reasons.get('deep_loser',0)+1; fired=True; break

            # Trail
            if peak>=trail_act and R<=peak*(1.0-trail_gb):
                pnls.append(R); reasons['trail']=reasons.get('trail',0)+1; fired=True; break
        if not fired:
            last=min(ei+max_k,n-1)
            pnls.append(d*(C[last]-ep)/ea); reasons['max_hold']=reasons.get('max_hold',0)+1
    return pnls, reasons

def pf(p):
    s=sum(p); w=[x for x in p if x>0]; l=[x for x in p if x<=0]
    pf=sum(w)/max(-sum(l),1e-9) if l else 99.0
    return pf,s,len(w)/max(len(p),1)

def maxdd(p):
    eq=np.cumsum(p); peak=np.maximum.accumulate(eq)
    return float((peak-eq).max())

def evaluate(name,swing_csv,setups_glob,trades_csv,bundle_path):
    print("\n"+"="*72); print(f"  {name}"); print("="*72)
    t0=time.time()
    n,C,atr,t2i,ctx=v89.load_market(swing_csv,setups_glob)
    sw_times=pd.read_csv(swing_csv,parse_dates=["time"]).sort_values("time").drop_duplicates('time',keep='last')["time"].values

    trades=pd.read_csv(trades_csv,parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    split=trades["time"].quantile(0.70)
    trn=trades[trades["time"]<split].reset_index(drop=True)
    tst=trades[trades["time"]>=split].reset_index(drop=True)

    # Train v89 components on TRAIN
    print("  Training v89 components (Q-ensemble, recovery clf)...",end='',flush=True); t=time.time()
    train_seqs=[]
    for _,tr in trn.iterrows():
        seq,_=v89.build_trade_state_seq(tr,t2i,n,C,atr,ctx)
        if seq: train_seqs.append(seq)
    q_ens=v89.fit_q_hold(train_seqs)
    Xa,y_rec,_,_=v89.build_aux_labels(train_seqs)
    rec_clf=v89.train_classifier(Xa,y_rec,mask=(y_rec>=0))
    print(f" {time.time()-t:.0f}s",flush=True)

    # Build setup lookup for v88 reverse-setup
    print("  Building setup lookup + setup_Q for v88 reverse-setup...",end='',flush=True); t=time.time()
    setups=load_setups(setups_glob)
    bundle=pickle.load(open(bundle_path,"rb"))
    setup_Q=np.full(len(setups),-9.0,dtype=np.float32)
    for cid,m in bundle['q_entry'].items():
        mask=(setups['old_cid']==cid).values
        if mask.sum()<1: continue
        Xc=setups.loc[mask,V72L].fillna(0).values
        setup_Q[mask]=m.predict(Xc)
    setup_lkp={}
    times_arr=setups.time.values; dirs_arr=setups.direction.values
    for i in range(len(setups)):
        tt=pd.Timestamp(times_arr[i]); d=int(dirs_arr[i])
        if tt not in setup_lkp: setup_lkp[tt]={}
        setup_lkp[tt][d]=i
    print(f" {time.time()-t:.0f}s",flush=True)

    # Run all 6 policies
    common=dict(q_ens=q_ens, rec_clf=rec_clf,
                setup_lkp=setup_lkp, setup_Q=setup_Q, sw_times=sw_times)
    policies=[
        ("A) BASELINE-40 (no v88, trail only, max=40)",
            dict(max_hold=40,use_q=False,use_deep=False,use_v88=False)),
        ("B) PROD-60 (no v88, trail only, max=60)",
            dict(max_hold=60,use_q=False,use_deep=False,use_v88=False)),
        ("C) PROD-60 + v88 reverse-setup (current live)",
            dict(max_hold=60,use_q=False,use_deep=False,use_v88=True)),
        ("D) v89 Q-dominates only (eps=1.0, max=40)",
            dict(max_hold=40,use_q=True,use_deep=False,use_v88=False,q_eps=1.0)),
        ("E) v89 Q-dominates + deep-loser",
            dict(max_hold=40,use_q=True,use_deep=True,use_v88=False,q_eps=1.0)),
        ("F) v89 Q-dominates + v88 reverse-setup (max=60)",
            dict(max_hold=60,use_q=True,use_deep=False,use_v88=True,q_eps=1.0)),
    ]
    print(f"\n  POLICY COMPARISON  (test={len(tst)} trades)")
    print(f"  {'policy':<60s}  {'PF':>5s} {'Total':>7s} {'WR%':>5s} {'DD':>4s}")
    print(f"  {'-'*60}  {'-'*5} {'-'*7} {'-'*5} {'-'*4}")
    rows=[]
    for label,params in policies:
        ps,rs=simulate(tst,t2i,n,C,atr,ctx,label,**common,**params)
        pf_,sR,wr_=pf(ps); dd_=maxdd(ps)
        rows.append((label,pf_,sR,wr_,dd_,rs))
        print(f"  {label:<60s}  {pf_:5.2f} {sR:+7.0f} {wr_*100:5.1f} {dd_:3.0f}R")

    # Reason breakdown for D, E, F (interesting ones)
    for label,pf_,sR,wr_,dd_,rs in rows[3:]:
        print(f"\n  {label}")
        for k,v in sorted(rs.items(),key=lambda x:-x[1]):
            print(f"    {k:>20s}: {v:4d}")

    print(f"\n  Done in {time.time()-t0:.0f}s")

if __name__ == "__main__":
    evaluate("Oracle XAU — v89 vs v88 vs combined",
             f"{DATA}/swing_v5_xauusd.csv",
             f"{DATA}/setups_*_v72l.csv",
             f"{PROJECT}/experiments/v84_rl_entry/v84_rl_trades.csv",
             f"{PROJECT}/products/models/oracle_xau_validated.pkl")
    evaluate("Oracle BTC — v89 vs v88 vs combined",
             f"{DATA}/swing_v5_btc.csv",
             f"{DATA}/setups_*_v72l_btc.csv",
             f"{PROJECT}/experiments/v84_rl_entry/btc_rl_trades.csv",
             f"{PROJECT}/products/models/oracle_btc_validated.pkl")
