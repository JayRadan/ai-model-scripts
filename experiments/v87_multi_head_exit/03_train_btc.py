"""v87: Multi-Head Exit Model — BTC (Oracle BTC)

Trains 4 XGBClassifier heads on BTCUSD RL trade data:
  1. P(meaningful upside)     — more profit remaining
  2. P(large giveback)        — current profits about to vanish
  3. P(stop loss hit)         — trade will hit -4R SL
  4. P(new high before drawdown) — will reach new peak before 1R drop

Mirrors 01_train_and_test.py but for BTC data.
Saves to multi_head_exit_oracle_btc.pkl for loader.py auto-detection.
"""
import pandas as pd, numpy as np, pickle, glob as _glob, time as _time, os, shutil
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

PROJECT = "/home/jay/Desktop/new-model-zigzag"
DATA = f"{PROJECT}/data"
OUT = f"{PROJECT}/experiments/v87_multi_head_exit"
os.makedirs(OUT, exist_ok=True)

# Target deploy path (loader.py looks for multi_head_exit_{config.name}.pkl)
DEPLOY_MODEL_DIR = "/home/jay/Desktop/my-agents-and-website/commercial/server/decision_engine/models"
DEPLOY_PATH = os.path.join(DEPLOY_MODEL_DIR, "multi_head_exit_oracle_btc.pkl")

V72L_FEATS = ['hurst_rs','ou_theta','entropy_rate','kramers_up','wavelet_er',
    'vwap_dist','hour_enc','dow_enc','quantum_flow','quantum_flow_h4',
    'quantum_momentum','quantum_vwap_conf','quantum_divergence','quantum_div_strength',
    'vpin','sig_quad_var','har_rv_ratio','hawkes_eta']
ENRICHED_FEATS = ['current_R','max_R_seen','drawdown_from_peak',
    'bars_in_trade','bars_remaining','dist_to_SL','dist_to_TP',
    'vol_10bar','mom_3bar','mom_10bar'] + V72L_FEATS  # 28 features
MAX_HOLD=60; MIN_HOLD=2; SL_HARD=-4.0

print("="*60)
print("  v87 Multi-Head Exit Model — BTC")
print("="*60)

# ═══ 1. Load BTC data ═══
print("[1/5] Loading BTC data...", end='', flush=True); t0=_time.time()
swing=pd.read_csv(f"{DATA}/swing_v5_btc.csv",parse_dates=["time"])
swing=swing.sort_values("time").drop_duplicates('time',keep='last').reset_index(drop=True)
n=len(swing); C=swing["close"].values.astype(np.float64)
H=swing["high"].values; L=swing["low"].values
tr=np.concatenate([[H[0]-L[0]],np.maximum.reduce([H[1:]-L[1:],np.abs(H[1:]-C[:-1]),np.abs(L[1:]-C[:-1])])])
atr=pd.Series(tr).rolling(14,min_periods=14).mean().values
t2i={t:i for i,t in enumerate(swing["time"].values)}

all_setups=[]
for f in sorted(_glob.glob(f"{DATA}/setups_*_v72l_btc.csv")):
    df=pd.read_csv(f,parse_dates=["time"]); all_setups.append(df)
print(f" {len(all_setups)} setup files", end='', flush=True)
all_df=pd.concat(all_setups,ignore_index=True)
phys=all_df[['time']+V72L_FEATS].drop_duplicates('time',keep='last').sort_values('time')
swing=pd.merge_asof(swing.sort_values('time'),phys,on='time',direction='nearest')
for col in V72L_FEATS: swing[col]=swing[col].fillna(0)
ctx_arr=swing[V72L_FEATS].fillna(0).values.astype(np.float64)

trades=pd.read_csv(f"{PROJECT}/experiments/v84_rl_entry/btc_rl_trades.csv",parse_dates=["time"])
trades=trades.sort_values("time").reset_index(drop=True)

# Chronological split: first 70% train, last 30% test
split_time = trades["time"].quantile(0.70)
train_trades = trades[trades["time"] < split_time].reset_index(drop=True)
test_trades  = trades[trades["time"] >= split_time].reset_index(drop=True)
print(f" — {_time.time()-t0:.0f}s — train={len(train_trades):,} test={len(test_trades):,}",flush=True)

# ═══ 2. Build training samples with 4 targets ═══
print("[2/5] Building samples...", end='', flush=True); t1=_time.time()

def build_samples(trades_df, max_samples=500_000):
    rows=[]
    n_bars=len(C)
    for _, trade in trades_df.iterrows():
        tm=trade["time"]
        if tm not in t2i: continue
        ei=t2i[tm]; d=int(trade["direction"])
        ep=C[ei]; ea=atr[ei]
        if not np.isfinite(ea) or ea<=0: continue
        actual_pnl=trade["pnl_R"]
        # Find peak MTM
        max_seen=0.0
        for k in range(1,min(MAX_HOLD,n_bars-ei-1)):
            bar=ei+k; mtm=d*(C[bar]-ep)/ea
            if mtm<=SL_HARD: break
            if mtm>max_seen: max_seen=mtm
        # Build bar-level samples
        for k in range(MIN_HOLD,min(MAX_HOLD,n_bars-ei-1)):
            bar=ei+k; mtm=d*(C[bar]-ep)/ea
            if mtm<=SL_HARD: break
            
            # TARGET 1: P(meaningful upside) — remaining > 0.5R
            remaining=actual_pnl-mtm
            label_up = 1 if remaining>0.5 else 0
            
            # TARGET 2: P(large giveback) — mtm>1R and loses>0.5R from here
            min_future=mtm
            for k2 in range(k+1,min(MAX_HOLD,n_bars-ei-1)):
                bar2=ei+k2; mtm2=d*(C[bar2]-ep)/ea
                if mtm2<=SL_HARD: min_future=SL_HARD; break
                min_future=min(min_future,mtm2)
            label_gb = 1 if (mtm>1.0 and (mtm-min_future)>0.5) else 0
            
            # TARGET 3: P(stop) — trade eventually hits SL
            label_sl = 1 if actual_pnl<=SL_HARD else 0
            
            # TARGET 4: P(new high before 1R drawdown)
            peak_after=mtm; found_hi=False; found_dd=False
            for k2 in range(k+1,min(MAX_HOLD,n_bars-ei-1)):
                bar2=ei+k2; mtm2=d*(C[bar2]-ep)/ea
                if mtm2<=SL_HARD: break
                if mtm2>peak_after: peak_after=mtm2; found_hi=True
                if peak_after-mtm2>1.0: found_dd=True; break
            label_nh = 1 if (found_hi and not found_dd) else 0
            
            # Features
            dist_sl=mtm-SL_HARD; dist_tp=2.0-mtm
            vol10=np.std([d*(C[max(0,bar-j)]-C[max(0,bar-j-1)])/ea
                          for j in range(min(10,bar))]) if bar>5 else 0
            mom3=d*(C[bar]-C[max(0,bar-3)])/ea if bar>=3 else 0
            mom10=d*(C[bar]-C[max(0,bar-10)])/ea if bar>=10 else 0
            
            row=[mtm,max_seen,max_seen-mtm,float(k),float(MAX_HOLD-k),
                 dist_sl,dist_tp,vol10,mom3,mom10]
            for j in range(len(V72L_FEATS)):
                row.append(float(ctx_arr[bar,j]))
            row.extend([label_up,label_gb,label_sl,label_nh])
            rows.append(row)
            if len(rows)>=max_samples: break
        if len(rows)>=max_samples: break
    return pd.DataFrame(rows,columns=ENRICHED_FEATS+
                        ['label_up','label_gb','label_sl','label_nh'])

trn_ed=build_samples(train_trades)
tst_ed=build_samples(test_trades)
print(f" {_time.time()-t1:.0f}s — train={len(trn_ed):,} test={len(tst_ed):,}",flush=True)

# ═══ 3. Train 4 heads ═══
print("[3/5] Training 4 heads...", flush=True)
models={}; aucs={}
for label,desc in [('label_up','upside'),('label_gb','giveback'),
                    ('label_sl','stop'),('label_nh','new_high')]:
    pos=(trn_ed[label]==1).sum()
    neg=len(trn_ed)-pos
    sw=neg/max(pos,1)  # balance weight
    print(f"  {desc}: {pos:,}/{len(trn_ed):,} positive ({pos/len(trn_ed)*100:.1f}%)",flush=True)
    
    mdl=XGBClassifier(n_estimators=300,max_depth=5,learning_rate=0.05,
                       subsample=0.8,colsample_bytree=0.8,
                       eval_metric='logloss',verbosity=0)
    mdl.fit(trn_ed[ENRICHED_FEATS].values,trn_ed[label].values,
            sample_weight=np.where(trn_ed[label]==1,sw,1.0))
    models[label]=mdl
    
    # AUC on test
    proba=mdl.predict_proba(tst_ed[ENRICHED_FEATS].values)[:,1]
    auc=roc_auc_score(tst_ed[label].values,proba)
    aucs[label]=auc
    print(f"    AUC={auc:.4f}",flush=True)

# ═══ 4. Policy sweep ═══
print("[4/5] Policy sweep on test trades...", flush=True)

# Batch-predict all test samples
probas={}
for label in ['label_up','label_gb','label_sl','label_nh']:
    probas[label]=models[label].predict_proba(tst_ed[ENRICHED_FEATS].values)[:,1]

# Group predictions by trade index in test_trades
trade_ids=[]
for ti, (_, trade) in enumerate(test_trades.iterrows()):
    tm=trade["time"]
    if tm not in t2i: continue
    ei=t2i[tm]; d=int(trade["direction"])
    ep=C[ei]; ea=atr[ei]
    if not np.isfinite(ea) or ea<=0: continue
    for k in range(MIN_HOLD,min(MAX_HOLD,n-ei-1)):
        bar=ei+k; mtm=d*(C[bar]-ep)/ea
        if mtm<=SL_HARD: break
        trade_ids.append(ti)

# Trim to match
n_samples=min(len(trade_ids),len(tst_ed))
trade_ids=trade_ids[:n_samples]
for k in probas: probas[k]=probas[k][:n_samples]

def apply_policy(probas_dict, trade_ids_arr, thresholds, test_trades_df):
    """Simulate exit for each trade using multi-head policy."""
    results=[]
    n_trades=len(test_trades_df)
    
    for ti in range(n_trades):
        trade=test_trades_df.iloc[ti]
        tm=trade["time"]
        if tm not in t2i: continue
        ei=t2i[tm]; d=int(trade["direction"])
        ep=C[ei]; ea=atr[ei]
        if not np.isfinite(ea) or ea<=0: continue
        
        exit_bar=None
        # Fall back to bar-level simulation
        for k in range(1,MAX_HOLD+1):
            bar=ei+k
            if bar>=n: break
            mtm=d*(C[bar]-ep)/ea
            if mtm<=SL_HARD: exit_bar=bar; break
            if k<MIN_HOLD: continue
            
            # Features at this bar
            dist_sl=mtm-SL_HARD; dist_tp=2.0-mtm
            vol10=np.std([d*(C[max(0,bar-jj)]-C[max(0,bar-jj-1)])/ea
                          for jj in range(min(10,bar))]) if bar>5 else 0
            mom3=d*(C[bar]-C[max(0,bar-3)])/ea if bar>=3 else 0
            mom10=d*(C[bar]-C[max(0,bar-10)])/ea if bar>=10 else 0
            
            # Find peak so far
            peak_sofar=0.0
            for kk in range(1,k+1):
                bb=ei+kk; mtm_kk=d*(C[bb]-ep)/ea
                if mtm_kk>peak_sofar: peak_sofar=mtm_kk
            
            feats=np.array([mtm,peak_sofar,peak_sofar-mtm,float(k),float(MAX_HOLD-k),
                  dist_sl,dist_tp,vol10,mom3,mom10]+
                 [ctx_arr[bar,jj] for jj in range(len(V72L_FEATS))],dtype=np.float32)
            
            p_up=float(models['label_up'].predict_proba(feats.reshape(1,-1))[0,1])
            p_gb=float(models['label_gb'].predict_proba(feats.reshape(1,-1))[0,1])
            p_sl=float(models['label_sl'].predict_proba(feats.reshape(1,-1))[0,1])
            p_nh=float(models['label_nh'].predict_proba(feats.reshape(1,-1))[0,1])
            
            # Apply policy
            th_up,th_gb,th_sl,th_nh=thresholds
            exit_signal=False
            if p_gb>th_gb and p_up<th_up: exit_signal=True
            if p_sl>th_sl: exit_signal=True
            if p_nh>th_nh and p_gb<th_gb: exit_signal=False  # override: hope!
            
            if exit_signal:
                exit_bar=bar; break
        
        if exit_bar is None: exit_bar=min(ei+MAX_HOLD,n-1)
        results.append(d*(C[exit_bar]-ep)/ea)
    
    return results

# Test a few policies (small subset for speed)
print("  Testing policies on first 300 test trades...")
test_subset=test_trades.head(300).reset_index(drop=True)

# Baseline: hard SL only
base_pnls=[]
for _,trade in test_subset.iterrows():
    tm=trade["time"]
    if tm not in t2i: continue
    ei=t2i[tm]; d=int(trade["direction"])
    ep=C[ei]; ea=atr[ei]
    if not np.isfinite(ea) or ea<=0: continue
    exit_bar=min(ei+MAX_HOLD,n-1)
    for k in range(1,MAX_HOLD+1):
        bar=ei+k
        if bar>=n: break
        if d*(C[bar]-ep)/ea<=SL_HARD: exit_bar=bar; break
    base_pnls.append(d*(C[exit_bar]-ep)/ea)

base_total=sum(base_pnls)

policies=[
    ("baseline (hard SL)",None),
    ("giveback>0.6 + up<0.4",(0.4,0.6,0.6,0.7)),
    ("giveback>0.7",(0.5,0.7,0.6,0.7)),
    ("stop>0.5",(0.5,0.6,0.5,0.7)),
    ("gb>0.5 + up<0.3",(0.3,0.5,0.6,0.7)),
]

print(f"\n  {'Policy':<30s} {'PF':>6s} {'Total':>8s}  vs baseline")
print(f"  {'-'*30} {'-'*6} {'-'*8}  {'-'*10}")
for name,thresh in policies:
    if thresh is None:
        total=base_total
        w=[p for p in base_pnls if p>0]; l=[p for p in base_pnls if p<=0]
        pf=sum(w)/max(-sum(l),1e-9) if l else 99
        print(f"  {name:<30s} {pf:6.2f} {total:+8.0f}")
    else:
        pnls=apply_policy(probas,trade_ids,thresh,test_subset)
        total=sum(pnls); w=[p for p in pnls if p>0]; l=[p for p in pnls if p<=0]
        pf=sum(w)/max(-sum(l),1e-9) if l else 99
        print(f"  {name:<30s} {pf:6.2f} {total:+8.0f}  {total-base_total:+8.0f}R")

# ═══ 5. Save ═══
print(f"\n[5/5] Saving...", end='', flush=True)
bundle={
    'models':models,
    'aucs':aucs,
    'enriched_feats':ENRICHED_FEATS,
    'v72l_feats':V72L_FEATS,
    'version':'v87-multi-head-exit-btc',
    # BTC-optimal thresholds from 04_sweep_btc_thresholds.py:
    # Multi-head is net harmful vs hard-SL for BTC RL trades.
    # Use extreme conservative values so multi-head rarely triggers,
    # preserving diagnostic output while letting binary ML exit work.
    # Test-set baseline: PF=4.06 +924R. Best multi-head: PF=2.12 +260R.
    'giveback_threshold': 0.99,
    'upside_threshold': 0.05,
    'stop_threshold': 0.95,
    'new_high_threshold': 0.80,
}

# Save to experiments
exp_path = f"{OUT}/multi_head_exit_oracle_btc.pkl"
with open(exp_path,'wb') as f:
    pickle.dump(bundle,f)
print(f" ✓ → {exp_path}",flush=True)

# Copy to deploy models directory
os.makedirs(DEPLOY_MODEL_DIR, exist_ok=True)
shutil.copy2(exp_path, DEPLOY_PATH)
print(f" ✓ → {DEPLOY_PATH} (deploy)",flush=True)

print(f"\n{'='*60}")
print(f"  Multi-Head Exit Model — BTC Summary")
print(f"  {'Head':<20s} {'AUC':>7s}")
for label,desc in [('label_up','upside'),('label_gb','giveback'),
                    ('label_sl','stop'),('label_nh','new_high')]:
    print(f"  {desc:<20s} {aucs[label]:7.4f}")
print(f"\n  Trained on {len(trn_ed):,} samples, {len(train_trades):,} trades")
print(f"  Tested on  {len(tst_ed):,} samples, {len(test_trades):,} trades")
print(f"  Total: {_time.time()-t0:.0f}s")
print(f"\n  Deployed to: {DEPLOY_PATH}")
print(f"  loader.py will auto-detect on next restart.")
