"""v88: Pullback-specific binary head — definitive test.

LABEL (per in-position bar k where mtm > 0):
  1 if peak_R_so_far >= 1.5  AND  final_pnl_R <= -2.0
  0 otherwise

This isolates the "trade peaked decently, then gave it all back to SL"
pattern. If even a model with M5 + M1 features can't reach AUC ≥ 0.65 on
this label on the unseen window, the failure mode is not predictable from
the available features.

Train/test: same 70/30 chrono split on v84 RL trades.
Eval: AUC, PR-AUC, precision @ recall = 0.20, plus a policy backtest
(exit at current MTM if p_pullback > threshold AND mtm > 0).
"""
import os, time, pickle, glob as _glob
import numpy as np, pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Reuse helpers from 03_m1_features_train
from importlib.util import spec_from_file_location, module_from_spec
spec=spec_from_file_location("m1mod", os.path.join(os.path.dirname(os.path.abspath(__file__)), "03_m1_features_train.py"))
m1mod=module_from_spec(spec); spec.loader.exec_module(m1mod)
load_market=m1mod.load_market; load_m1=m1mod.load_m1
m1_feats_at=m1mod.m1_feats_at; m5_feats=m1mod.m5_feats
V72L=m1mod.V72L; M1_FEATS=m1mod.M1_FEATS; ENRICHED=m1mod.ENRICHED
ENRICHED_M1=m1mod.ENRICHED_M1
MAX_HOLD=m1mod.MAX_HOLD; MIN_HOLD=m1mod.MIN_HOLD; SL_HARD=m1mod.SL_HARD

PROJECT="/home/jay/Desktop/new-model-zigzag"
DATA=f"{PROJECT}/data"
DUK="/tmp/duk"

PEAK_THR=1.5      # qualified as "had a peak"
DROP_THR=-2.0     # qualified as "gave it back"
PROFIT_GATE=0.0   # only label/predict at bars where mtm > this

def build_samples(trades,t2i,n,C,atr,ctx,swing_ns,m1_t,m1_c,m1_h,m1_l,with_meta=False):
    rows=[]; lbl=[]; mtm_arr=[]
    meta=[]; r2t=[]; rk=[]
    for ti,(_,trade) in enumerate(trades.iterrows()):
        tm=trade["time"]
        if tm not in t2i:
            if with_meta: meta.append(None); continue
            continue
        ei=t2i[tm]; d=int(trade["direction"]); ep=C[ei]; ea=atr[ei]
        if not np.isfinite(ea) or ea<=0:
            if with_meta: meta.append(None); continue
            continue
        final_r=float(trade["pnl_R"])
        is_pullback_sl = (final_r<=DROP_THR)  # only meaningful if we also reach a peak
        peak=0.0; sl_k=None
        max_k=min(MAX_HOLD,n-ei-1)
        for k in range(1,max_k+1):
            bar=ei+k; mtm=d*(C[bar]-ep)/ea
            if mtm<=SL_HARD: sl_k=k; break
            if mtm>peak: peak=mtm
            if k<MIN_HOLD: continue
            # Only sample bars where we're still in profit (decision-relevant)
            if mtm<=PROFIT_GATE: continue
            _,row=m5_feats(C,ctx,ei,k,d,ep,ea,peak)
            row=row+m1_feats_at(swing_ns[bar],d,ea,m1_t,m1_c,m1_h,m1_l)
            label = 1 if (peak>=PEAK_THR and is_pullback_sl) else 0
            rows.append(row); lbl.append(label); mtm_arr.append(mtm)
            r2t.append(ti); rk.append(k)
        if with_meta:
            meta.append({'ei':ei,'d':d,'ep':ep,'ea':ea,'sl_k':sl_k})
    return (np.asarray(rows,dtype=np.float32),
            np.asarray(lbl,dtype=np.int32),
            np.asarray(mtm_arr,dtype=np.float32),
            meta, np.asarray(r2t,dtype=np.int32), np.asarray(rk,dtype=np.int32))

def baseline_pnls(meta,n,C):
    out=[]
    for m in meta:
        if m is None: continue
        ei=m['ei']; d=m['d']; ep=m['ep']; ea=m['ea']
        bar=ei+m['sl_k'] if m['sl_k'] is not None else min(ei+MAX_HOLD,n-1)
        out.append(d*(C[bar]-ep)/ea)
    return out

def simulate_pullback_exit(meta,r2t,rk,probs,mtms,thr,n,C):
    """Exit at first bar where p>thr (mtm already filtered to >0 in build)."""
    n_tr=len(meta); chosen=np.full(n_tr,-1,dtype=np.int32)
    for i in range(len(r2t)):
        ti=int(r2t[i])
        if chosen[ti]!=-1: continue
        if probs[i]>thr: chosen[ti]=int(rk[i])
    out=[]
    for ti,m in enumerate(meta):
        if m is None: continue
        ei=m['ei']; d=m['d']; ep=m['ep']; ea=m['ea']; sl_k=m['sl_k']; ck=int(chosen[ti])
        if ck==-1: bar=ei+sl_k if sl_k is not None else min(ei+MAX_HOLD,n-1)
        else:
            if sl_k is not None and sl_k<ck: bar=ei+sl_k
            else: bar=ei+ck
        out.append(d*(C[bar]-ep)/ea)
    return out

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
    print(f"  Train: {len(trn)} | Unseen test: {len(tst)} | split @ {split}")

    print("  Building train samples (mtm>0 only)...",end='',flush=True); t1=time.time()
    Xtr,ytr,mtm_tr,_,_,_=build_samples(trn,t2i,n,C,atr,ctx,swing_ns,m1_t,m1_c,m1_h,m1_l)
    pos_tr=int(ytr.sum())
    print(f" {time.time()-t1:.0f}s — rows={len(Xtr):,}, pullback bars={pos_tr} ({pos_tr/max(len(ytr),1)*100:.1f}%)",flush=True)
    print("  Building test samples...",end='',flush=True); t1=time.time()
    Xte,yte,mtm_te,meta,r2t,rk=build_samples(tst,t2i,n,C,atr,ctx,swing_ns,m1_t,m1_c,m1_h,m1_l,with_meta=True)
    pos_te=int(yte.sum())
    print(f" {time.time()-t1:.0f}s — rows={len(Xte):,}, pullback bars={pos_te} ({pos_te/max(len(yte),1)*100:.1f}%)",flush=True)

    if pos_tr<20 or pos_te<10:
        print("  Too few positive examples — pattern is too rare to model reliably."); return

    sw_pos=(len(ytr)-pos_tr)/max(pos_tr,1)
    sample_w=np.where(ytr==1,sw_pos,1.0).astype(np.float32)
    print(f"  Training pullback head (pos weight={sw_pos:.1f})...",end='',flush=True); t1=time.time()
    mdl=XGBClassifier(n_estimators=500,max_depth=6,learning_rate=0.04,
                      subsample=0.8,colsample_bytree=0.8,
                      eval_metric='logloss',verbosity=0)
    mdl.fit(Xtr,ytr,sample_weight=sample_w)
    print(f" {time.time()-t1:.0f}s",flush=True)

    p=mdl.predict_proba(Xte)[:,1]
    auc=roc_auc_score(yte,p) if pos_te>0 else 0
    pr_auc=average_precision_score(yte,p) if pos_te>0 else 0
    base_rate=yte.mean()
    print(f"\n  ROC AUC : {auc:.4f}")
    print(f"  PR  AUC : {pr_auc:.4f}  (random = {base_rate:.4f})  → lift = {pr_auc/max(base_rate,1e-9):.2f}x")

    # Precision @ recall = 0.20
    order=np.argsort(-p)
    cum_pos=np.cumsum(yte[order])
    target=int(0.20*pos_te)
    if target>0:
        idx=np.searchsorted(cum_pos,target)
        prec_20=cum_pos[idx]/(idx+1) if idx<len(cum_pos) else 0
        thr_20=p[order[idx]] if idx<len(order) else 1.0
        print(f"  Precision @ recall=0.20 : {prec_20:.3f}  (threshold={thr_20:.3f}, captures {target} of {pos_te} pullback bars)")

    base=baseline_pnls(meta,n,C); pf_b,s_b,wr_b,N_b=pf_stats(base)
    print(f"\n  Baseline (hard SL):       PF={pf_b:.2f}  Total={s_b:+.0f}R  WR={wr_b*100:.1f}%  N={N_b}")
    print(f"\n  Policy backtest — exit at current mtm if p_pullback > thr (mtm>0 only):")
    print(f"  {'thr':>6s}  {'PF':>5s}  {'TotalR':>8s}  {'WR%':>5s}  vs base")
    best=None
    for thr in [0.5,0.6,0.7,0.75,0.8,0.85,0.9,0.95]:
        pn=simulate_pullback_exit(meta,r2t,rk,p,mtm_te,thr,n,C)
        pf,s,wr,N=pf_stats(pn)
        marker=" ★" if s>s_b else ""
        print(f"  {thr:6.2f}  {pf:5.2f}  {s:+8.0f}  {wr*100:5.1f}  {s-s_b:+.0f}R{marker}")
        if best is None or s>best[1]: best=(thr,s,pf,wr,N)

    fi=mdl.feature_importances_
    fnames=ENRICHED_M1
    top=sorted(enumerate(fi),key=lambda x:-x[1])[:12]
    print(f"\n  Top-12 features:")
    for i,v in top:
        tag="M1" if fnames[i] in M1_FEATS else "M5"
        print(f"    {tag} {fnames[i]:<28s} {v:.4f}")

    print(f"\n  VERDICT")
    if auc>=0.70:
        print(f"  AUC {auc:.3f} ≥ 0.70 — pullback pattern IS detectable; pursue policy tuning.")
    elif auc>=0.60:
        print(f"  AUC {auc:.3f} ∈ [0.60,0.70) — weak signal; unlikely to beat baseline robustly.")
    else:
        print(f"  AUC {auc:.3f} < 0.60 — pullback pattern not predictable from M5+M1 features.")
    print(f"  Best policy: thr={best[0]}  PF={best[2]:.2f}  Total={best[1]:+.0f}R  vs base {s_b:+.0f}R ({best[1]-s_b:+.0f}R)")
    print(f"  Done in {time.time()-t0:.0f}s")

if __name__=="__main__":
    evaluate("Oracle XAU — Pullback-specific binary head",
             f"{DATA}/swing_v5_xauusd.csv",
             f"{DATA}/setups_*_v72l.csv",
             f"{PROJECT}/experiments/v84_rl_entry/v84_rl_trades.csv",
             f"{DUK}/xau_m1.csv")
    evaluate("Oracle BTC — Pullback-specific binary head",
             f"{DATA}/swing_v5_btc.csv",
             f"{DATA}/setups_*_v72l_btc.csv",
             f"{PROJECT}/experiments/v84_rl_entry/btc_rl_trades.csv",
             f"{DUK}/btc_m1.csv")
