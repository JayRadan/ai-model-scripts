"""v89 ablation — find which rule(s) actually help on the unseen window.

Test each gate in isolation + combinations + parameter sweeps:
  V1: q_dominates only       (epsilon ∈ {0, 0.25, 0.5, 1.0}, percentile ∈ {25, 10, 5})
  V2: winner_giveback only   (peak_thr ∈ {1.5, 2.0, 2.5}, giveback_pct ∈ {0.5, 0.6, 0.7})
  V3: deep_loser only        (R_thr ∈ {-2, -2.5, -3}, recovery_thr ∈ {0.1, 0.2, 0.3})
  V4: best-of-each combined
"""
import os, time, glob as _glob
import numpy as np, pandas as pd
from xgboost import XGBRegressor, XGBClassifier
from importlib.util import spec_from_file_location, module_from_spec

PROJECT="/home/jay/Desktop/new-model-zigzag"
DATA=f"{PROJECT}/data"

# Reuse from script 01
spec=spec_from_file_location("v89", os.path.join(os.path.dirname(os.path.abspath(__file__)),"01_optimal_stopping.py"))
v89=module_from_spec(spec); spec.loader.exec_module(v89)
V72L=v89.V72L; STATE_FEATS=v89.STATE_FEATS
MAX_HOLD=v89.MAX_HOLD; SL_HARD=v89.SL_HARD

def state_at(bar,d,ep,ea,C,ctx,k,peak):
    R=d*(C[bar]-ep)/ea
    c3=C[max(0,bar-3)]; c10=C[max(0,bar-10)]
    mom3=d*(C[bar]-c3)/ea if bar>=3 else 0.0
    mom10=d*(C[bar]-c10)/ea if bar>=10 else 0.0
    vol10=float(np.std([d*(C[max(0,bar-j)]-C[max(0,bar-j-1)])/ea for j in range(min(10,bar))])) if bar>5 else 0.0
    v72=[float(ctx[bar,j]) for j in range(len(V72L))]
    s={'current_R':float(R),'peak_R':float(peak),'drawdown_from_peak':float(peak-R),
       'bars_in_trade':float(k),'bars_remaining':float(MAX_HOLD-k),
       'mom_3bar':float(mom3),'mom_10bar':float(mom10),'vol_10bar':vol10,
       'dist_to_SL':float(R-SL_HARD),'dist_to_TP':float(2.0-R)}
    for fname,fv in zip(V72L,v72): s[fname]=fv
    s['_features']=np.asarray([s[f] for f in STATE_FEATS],dtype=np.float32)
    return s,R

def simulate(trades,t2i,n,C,atr,ctx,policy_fn):
    pnls=[]; reasons=[]
    for _,trade in trades.iterrows():
        tm=trade["time"]
        if tm not in t2i: continue
        ei=t2i[tm]; d=int(trade["direction"]); ep=C[ei]; ea=atr[ei]
        if not np.isfinite(ea) or ea<=0: continue
        peak=0.0; max_k=min(MAX_HOLD,n-ei-1); fired=False
        for k in range(1,max_k+1):
            bar=ei+k
            R=d*(C[bar]-ep)/ea
            if R<=SL_HARD: pnls.append(R); reasons.append("hard_sl"); fired=True; break
            if R>peak: peak=R
            s,_=state_at(bar,d,ep,ea,C,ctx,k,peak)
            ex,why=policy_fn(s)
            if ex: pnls.append(R); reasons.append(why); fired=True; break
        if not fired:
            last=min(ei+max_k,n-1)
            pnls.append(d*(C[last]-ep)/ea); reasons.append("max_hold")
    return pnls,reasons

def pf(p):
    s=sum(p); w=[x for x in p if x>0]; l=[x for x in p if x<=0]
    pf=sum(w)/max(-sum(l),1e-9) if l else 99.0
    return pf,s,len(w)/max(len(p),1)

def maxdd(p):
    eq=np.cumsum(p); peak=np.maximum.accumulate(eq)
    return float((peak-eq).max())

def baseline_policy(s):
    """Pure trail @ act=3, gb=0.6 (no smart logic)."""
    if s['peak_R']>=3.0 and s['current_R']<=s['peak_R']*0.4:
        return True,"trail"
    return False,"hold"

def evaluate(name,swing_csv,setups_glob,trades_csv):
    print("\n"+"="*72); print(f"  {name}"); print("="*72)
    t0=time.time()
    n,C,atr,t2i,ctx=v89.load_market(swing_csv,setups_glob)
    trades=pd.read_csv(trades_csv,parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    split=trades["time"].quantile(0.70)
    trn=trades[trades["time"]<split].reset_index(drop=True)
    tst=trades[trades["time"]>=split].reset_index(drop=True)
    print(f"  Train: {len(trn)} | Test: {len(tst)}")

    print("  Building train state seqs + Q ensemble...",end='',flush=True); t=time.time()
    train_seqs=[]
    for _,tr in trn.iterrows():
        seq,_=v89.build_trade_state_seq(tr,t2i,n,C,atr,ctx)
        if seq: train_seqs.append(seq)
    q_ens=v89.fit_q_hold(train_seqs)
    print(f" {time.time()-t:.0f}s",flush=True)

    print("  Training auxiliary classifiers (with proper holdout AUC)...",end='',flush=True); t=time.time()
    Xa,y_rec,y_brk,y_sl=v89.build_aux_labels(train_seqs)
    rec_clf=v89.train_classifier(Xa,y_rec,mask=(y_rec>=0))
    brk_clf=v89.train_classifier(Xa,y_brk)
    print(f" {time.time()-t:.0f}s",flush=True)

    # Build TEST aux labels for honest AUC (not on train data)
    test_seqs=[]
    for _,tr in tst.iterrows():
        seq,_=v89.build_trade_state_seq(tr,t2i,n,C,atr,ctx)
        if seq: test_seqs.append(seq)
    Xt,yt_rec,yt_brk,yt_sl=v89.build_aux_labels(test_seqs)
    from sklearn.metrics import roc_auc_score
    auc_rec_te=roc_auc_score(yt_rec[yt_rec>=0],rec_clf.predict_proba(Xt[yt_rec>=0])[:,1]) if rec_clf else float('nan')
    auc_brk_te=roc_auc_score(yt_brk,brk_clf.predict_proba(Xt)[:,1]) if brk_clf else float('nan')
    print(f"  TEST AUCs: recovery={auc_rec_te:.3f}  breakdown={auc_brk_te:.3f}")

    # Cache q_hold predictions on test bars (for fast policy sweeping)
    print("  Pre-caching q_hold predictions on test bars...",end='',flush=True); t=time.time()
    test_q_cache={}  # (trade_idx, k) -> p25 q_hold
    for ti,(_,trade) in enumerate(tst.iterrows()):
        tm=trade["time"]
        if tm not in t2i: continue
        ei=t2i[tm]; d=int(trade["direction"]); ep=C[ei]; ea=atr[ei]
        if not np.isfinite(ea) or ea<=0: continue
        peak=0.0; max_k=min(MAX_HOLD,n-ei-1)
        for k in range(1,max_k+1):
            bar=ei+k
            R=d*(C[bar]-ep)/ea
            if R<=SL_HARD: break
            if R>peak: peak=R
            s,_=state_at(bar,d,ep,ea,C,ctx,k,peak)
            preds=np.array([m.predict(s['_features'].reshape(1,-1))[0] for m in q_ens])
            test_q_cache[(ti,k)]={
                'q_p25':float(np.percentile(preds,25)),
                'q_p10':float(np.percentile(preds,10)),
                'q_p50':float(np.median(preds)),
                'p_rec':float(rec_clf.predict_proba(s['_features'].reshape(1,-1))[0,1]) if rec_clf else 0.5,
                'p_brk':float(brk_clf.predict_proba(s['_features'].reshape(1,-1))[0,1]) if brk_clf else 0.5,
            }
    print(f" {time.time()-t:.0f}s",flush=True)

    # Baseline
    bp,br=simulate(tst,t2i,n,C,atr,ctx,baseline_policy)
    pf_b,s_b,wr_b=pf(bp); dd_b=maxdd(bp)
    print(f"\n  BASELINE: PF={pf_b:.2f}  Total={s_b:+.0f}R  WR={wr_b*100:.1f}%  DD={dd_b:.0f}R  N={len(bp)}")

    # Sweep policies
    def pol_q(state, ti, k, eps, percentile_key):
        if state['current_R']<=SL_HARD: return True,"hard_sl"
        if state['bars_in_trade']>=MAX_HOLD: return True,"max_hold"
        c=test_q_cache.get((ti,k))
        if c is None: return False,"hold"
        q_hold=c[percentile_key]
        if state['current_R']>=q_hold+eps: return True,"q_dominates"
        return False,"hold"

    def pol_winner(state, peak_thr, frac):
        if state['current_R']<=SL_HARD: return True,"hard_sl"
        if state['bars_in_trade']>=MAX_HOLD: return True,"max_hold"
        if state['peak_R']>=peak_thr and state['current_R']<state['peak_R']*frac:
            return True,"winner_giveback"
        return False,"hold"

    def pol_loser(state, ti, k, R_thr, rec_thr):
        if state['current_R']<=SL_HARD: return True,"hard_sl"
        if state['bars_in_trade']>=MAX_HOLD: return True,"max_hold"
        c=test_q_cache.get((ti,k))
        if c is None: return False,"hold"
        if state['current_R']<=R_thr and c['p_rec']<rec_thr:
            return True,"deep_loser"
        return False,"hold"

    # We need to simulate with policies that need (ti,k). Rewrite simulate inline.
    def sim_idx(policy_fn_idx):
        pnls=[]; reasons=[]
        for ti,(_,trade) in enumerate(tst.iterrows()):
            tm=trade["time"]
            if tm not in t2i: continue
            ei=t2i[tm]; d=int(trade["direction"]); ep=C[ei]; ea=atr[ei]
            if not np.isfinite(ea) or ea<=0: continue
            peak=0.0; max_k=min(MAX_HOLD,n-ei-1); fired=False
            for k in range(1,max_k+1):
                bar=ei+k
                R=d*(C[bar]-ep)/ea
                if R<=SL_HARD: pnls.append(R); reasons.append("hard_sl"); fired=True; break
                if R>peak: peak=R
                s,_=state_at(bar,d,ep,ea,C,ctx,k,peak)
                ex,why=policy_fn_idx(s,ti,k)
                if ex: pnls.append(R); reasons.append(why); fired=True; break
            if not fired:
                last=min(ei+max_k,n-1)
                pnls.append(d*(C[last]-ep)/ea); reasons.append("max_hold")
        return pnls,reasons

    print(f"\n  V1: Q-DOMINATES ONLY (sweep epsilon, percentile)")
    print(f"  {'eps':>5s} {'pct':>5s} {'PF':>5s} {'Total':>7s} {'DD':>4s} {'fires':>5s}  vs base")
    for eps in [0.0,0.25,0.5,1.0,1.5,2.0]:
        for pct in ['q_p10','q_p25','q_p50']:
            ps,rs=sim_idx(lambda s,ti,k: pol_q(s,ti,k,eps,pct))
            pf_,sR,wr_=pf(ps); dd_=maxdd(ps)
            fires=sum(1 for r in rs if r=='q_dominates')
            mark=" ★" if (sR>s_b and dd_<=dd_b) else (" + R" if sR>s_b else "")
            print(f"  {eps:5.2f} {pct:>5s} {pf_:5.2f} {sR:+7.0f} {dd_:3.0f}R {fires:5d}  {sR-s_b:+5.0f}R{mark}")

    print(f"\n  V2: WINNER-GIVEBACK ONLY")
    print(f"  {'peak':>5s} {'frac':>5s} {'PF':>5s} {'Total':>7s} {'DD':>4s} {'fires':>5s}  vs base")
    for peak_thr in [1.5,2.0,2.5,3.0]:
        for frac in [0.30,0.40,0.50,0.60]:
            ps,rs=simulate(tst,t2i,n,C,atr,ctx,lambda s: pol_winner(s,peak_thr,frac))
            pf_,sR,wr_=pf(ps); dd_=maxdd(ps)
            fires=sum(1 for r in rs if r=='winner_giveback')
            mark=" ★" if (sR>s_b and dd_<=dd_b) else (" + R" if sR>s_b else "")
            print(f"  {peak_thr:5.2f} {frac:5.2f} {pf_:5.2f} {sR:+7.0f} {dd_:3.0f}R {fires:5d}  {sR-s_b:+5.0f}R{mark}")

    print(f"\n  V3: DEEP-LOSER ONLY")
    print(f"  {'R_thr':>5s} {'rec':>5s} {'PF':>5s} {'Total':>7s} {'DD':>4s} {'fires':>5s}  vs base")
    for R_thr in [-2.0,-2.5,-3.0,-3.5]:
        for rec_thr in [0.05,0.10,0.20,0.30]:
            ps,rs=sim_idx(lambda s,ti,k: pol_loser(s,ti,k,R_thr,rec_thr))
            pf_,sR,wr_=pf(ps); dd_=maxdd(ps)
            fires=sum(1 for r in rs if r=='deep_loser')
            mark=" ★" if (sR>s_b and dd_<=dd_b) else (" + R" if sR>s_b else "")
            print(f"  {R_thr:5.2f} {rec_thr:5.2f} {pf_:5.2f} {sR:+7.0f} {dd_:3.0f}R {fires:5d}  {sR-s_b:+5.0f}R{mark}")

    print(f"\n  Done in {time.time()-t0:.0f}s")

if __name__ == "__main__":
    evaluate("Oracle XAU — v89 ablation",
             f"{DATA}/swing_v5_xauusd.csv",
             f"{DATA}/setups_*_v72l.csv",
             f"{PROJECT}/experiments/v84_rl_entry/v84_rl_trades.csv")
    evaluate("Oracle BTC — v89 ablation",
             f"{DATA}/swing_v5_btc.csv",
             f"{DATA}/setups_*_v72l_btc.csv",
             f"{PROJECT}/experiments/v84_rl_entry/btc_rl_trades.csv")
