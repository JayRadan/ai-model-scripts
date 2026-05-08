"""v88: Adversarial entry filter.

Pipeline so far: setups → q_entry≥0.3 → meta gate → range-position → kill-switch
                          → regime relabel → final RL trade

Hypothesis: among the trades that pass ALL current filters, a subset still
loses systematically. Train a binary head on (entry-time features) →
P(loser). Skip trades when P(loser) > threshold. If even this clean,
post-all-filters subset isn't separable, we're at the noise floor.

Two labels tried:
  (A) any_loser:    final_pnl_R <= -2.0
  (B) pullback_sl:  peak_R >= 1.5 AND final_pnl_R <= -2.0  (the pattern Jay flagged)

Features at entry: V72L (18) + atr_z + bar_idx_hour/dow already encoded.
Plus the same M1 micro-structure features computed at the entry bar.

70/30 chrono split on v84 RL trades.
"""
import os, time, pickle, glob as _glob
import numpy as np, pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib.util import spec_from_file_location, module_from_spec
spec=spec_from_file_location("m1mod", os.path.join(os.path.dirname(os.path.abspath(__file__)), "03_m1_features_train.py"))
m1mod=module_from_spec(spec); spec.loader.exec_module(m1mod)
load_market=m1mod.load_market; load_m1=m1mod.load_m1; m1_feats_at=m1mod.m1_feats_at
V72L=m1mod.V72L; M1_FEATS=m1mod.M1_FEATS
MAX_HOLD=60; SL_HARD=-4.0
PEAK_THR=1.5; LOSE_THR=-2.0

PROJECT="/home/jay/Desktop/new-model-zigzag"
DATA=f"{PROJECT}/data"
DUK="/tmp/duk"

ENTRY_FEATS=V72L+M1_FEATS+['atr_norm','hour','dow']

def label_trade(trade,t2i,n,C,atr,d):
    """Walk the trade forward to compute (final_pnl, peak_R)."""
    tm=trade["time"]
    if tm not in t2i: return None
    ei=t2i[tm]; ep=C[ei]; ea=atr[ei]
    if not np.isfinite(ea) or ea<=0: return None
    final=trade["pnl_R"]
    peak=0.0
    for k in range(1,min(MAX_HOLD,n-ei-1)+1):
        bar=ei+k
        mtm=d*(C[bar]-ep)/ea
        if mtm<=SL_HARD: break
        if mtm>peak: peak=mtm
    return float(final),float(peak),ei,ea

def build(trades,t2i,n,C,atr,ctx,swing_ns,m1_t,m1_c,m1_h,m1_l):
    rows=[]; lbl_any=[]; lbl_pull=[]; meta=[]
    for _,trade in trades.iterrows():
        tm=trade["time"]
        if tm not in t2i:
            meta.append(None); continue
        d=int(trade["direction"])
        info=label_trade(trade,t2i,n,C,atr,d)
        if info is None:
            meta.append(None); continue
        final,peak,ei,ea=info
        # Entry-time features
        v72 = [float(ctx[ei,j]) for j in range(len(V72L))]
        m1f = m1_feats_at(swing_ns[ei],d,ea,m1_t,m1_c,m1_h,m1_l)
        atr_norm = float(ea/C[ei]) if C[ei]>0 else 0.0
        ts=pd.Timestamp(tm)
        hour=float(ts.hour); dow=float(ts.dayofweek)
        rows.append(v72+m1f+[atr_norm,hour,dow])
        lbl_any.append(1 if final<=LOSE_THR else 0)
        lbl_pull.append(1 if (peak>=PEAK_THR and final<=LOSE_THR) else 0)
        meta.append({'final':final,'peak':peak,'ei':ei,'ea':ea,'d':d})
    return (np.asarray(rows,dtype=np.float32),
            np.asarray(lbl_any,dtype=np.int32),
            np.asarray(lbl_pull,dtype=np.int32),
            meta)

def pf_stats(p):
    s=sum(p); w=[x for x in p if x>0]; l=[x for x in p if x<=0]
    pf=sum(w)/max(-sum(l),1e-9) if l else 99.0
    return pf,s,len(w)/max(len(p),1),len(p)

def evaluate(name,swing_csv,setups_glob,trades_csv,m1_csv):
    print("\n"+"="*72); print(f"  {name}"); print("="*72)
    t0=time.time()
    n,C,Lo,H,atr,t2i,ctx,swing_ns=load_market(swing_csv,setups_glob)
    m1_t,m1_c,m1_h,m1_l=load_m1(m1_csv)
    trades=pd.read_csv(trades_csv,parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    split=trades["time"].quantile(0.70)
    trn=trades[trades["time"]<split].reset_index(drop=True)
    tst=trades[trades["time"]>=split].reset_index(drop=True)

    print("  Building features at entry-time...",end='',flush=True); t1=time.time()
    Xtr,ytr_any,ytr_pull,meta_tr=build(trn,t2i,n,C,atr,ctx,swing_ns,m1_t,m1_c,m1_h,m1_l)
    Xte,yte_any,yte_pull,meta_te=build(tst,t2i,n,C,atr,ctx,swing_ns,m1_t,m1_c,m1_h,m1_l)
    print(f" {time.time()-t1:.0f}s",flush=True)
    print(f"  Train: N={len(Xtr)} | losers={int(ytr_any.sum())} ({ytr_any.mean()*100:.1f}%) | pullback_sl={int(ytr_pull.sum())} ({ytr_pull.mean()*100:.1f}%)")
    print(f"  Test:  N={len(Xte)} | losers={int(yte_any.sum())} ({yte_any.mean()*100:.1f}%) | pullback_sl={int(yte_pull.sum())} ({yte_pull.mean()*100:.1f}%)")

    # Baseline = take all test trades
    base=[m['final'] for m in meta_te if m is not None]
    pf_b,s_b,wr_b,N_b=pf_stats(base)
    print(f"\n  Baseline (take all):       PF={pf_b:.2f}  Total={s_b:+.0f}R  WR={wr_b*100:.1f}%  N={N_b}")

    for label_name, ytr, yte in [("any-loser",ytr_any,yte_any),
                                  ("pullback-SL",ytr_pull,yte_pull)]:
        pos=int(ytr.sum())
        if pos<10:
            print(f"\n  [{label_name}] too few positives, skipping"); continue
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

        # Skip-trade policy: drop trades where p > threshold
        print(f"  {'thr':>6s}  {'kept':>5s}  {'PF':>5s}  {'TotalR':>8s}  {'WR%':>5s}  vs base")
        finals=np.array([m['final'] for m in meta_te if m is not None])
        # mask: keep if p <= thr
        order_thr=[0.30,0.40,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90]
        for thr in order_thr:
            keep=p<=thr
            kept_pnls=finals[keep].tolist()
            if len(kept_pnls)<20:
                print(f"  {thr:6.2f}  {len(kept_pnls):5d}  too few"); continue
            pf,s,wr,N=pf_stats(kept_pnls)
            marker=" ★" if (s>s_b and N>=int(0.5*N_b)) else ""
            print(f"  {thr:6.2f}  {N:5d}  {pf:5.2f}  {s:+8.0f}  {wr*100:5.1f}  {s-s_b:+.0f}R{marker}")

        fi=mdl.feature_importances_
        top=sorted(enumerate(fi),key=lambda x:-x[1])[:8]
        print(f"  Top-8 features:")
        for i,v in top:
            tag="M1" if ENTRY_FEATS[i] in M1_FEATS else "M5/ent"
            print(f"    {tag} {ENTRY_FEATS[i]:<28s} {v:.4f}")

    print(f"\n  Done in {time.time()-t0:.0f}s")

if __name__=="__main__":
    evaluate("Oracle XAU — Adversarial entry filter",
             f"{DATA}/swing_v5_xauusd.csv",
             f"{DATA}/setups_*_v72l.csv",
             f"{PROJECT}/experiments/v84_rl_entry/v84_rl_trades.csv",
             f"{DUK}/xau_m1.csv")
    evaluate("Oracle BTC — Adversarial entry filter",
             f"{DATA}/swing_v5_btc.csv",
             f"{DATA}/setups_*_v72l_btc.csv",
             f"{PROJECT}/experiments/v84_rl_entry/btc_rl_trades.csv",
             f"{DUK}/btc_m1.csv")
