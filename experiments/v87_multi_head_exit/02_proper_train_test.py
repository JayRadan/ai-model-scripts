"""v87: Proper Train/Test Split — train on pre-2024-12-12, test on holdout"""
import pandas as pd, numpy as np, pickle, glob as _glob, time as _time, os
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import roc_auc_score

PROJECT = "/home/jay/Desktop/new-model-zigzag"
DATA = f"{PROJECT}/data"
OUT = f"{PROJECT}/experiments/v87_multi_head_exit"
os.makedirs(OUT, exist_ok=True)

CUTOFF = pd.Timestamp("2024-12-12 00:00:00")
V72L_FEATS = ['hurst_rs','ou_theta','entropy_rate','kramers_up','wavelet_er',
    'vwap_dist','hour_enc','dow_enc','quantum_flow','quantum_flow_h4',
    'quantum_momentum','quantum_vwap_conf','quantum_divergence','quantum_div_strength',
    'vpin','sig_quad_var','har_rv_ratio','hawkes_eta']
ENRICHED = ['current_R','max_R_seen','drawdown_from_peak',
    'bars_in_trade','bars_remaining','dist_to_SL','dist_to_TP',
    'vol_10bar','mom_3bar','mom_10bar'] + V72L_FEATS
MAX_HOLD=60; MIN_HOLD=2; SL_HARD=-4.0; MIN_Q=0.3

print("="*60)
print("  v87 Multi-Head Exit — Proper Train/Test")
print("="*60)

# ═══ 1. Load data ═══
print("[1/4] Loading...", end='', flush=True); t0=_time.time()
swing=pd.read_csv(f"{DATA}/swing_v5_xauusd.csv",parse_dates=["time"])
swing=swing.sort_values("time").drop_duplicates('time',keep='last').reset_index(drop=True)
n=len(swing); C=swing["close"].values.astype(np.float64)
H=swing["high"].values; L=swing["low"].values
tr=np.concatenate([[H[0]-L[0]],np.maximum.reduce([H[1:]-L[1:],np.abs(H[1:]-C[:-1]),np.abs(L[1:]-C[:-1])])])
atr=pd.Series(tr).rolling(14,min_periods=14).mean().values
t2i={t:i for i,t in enumerate(swing["time"].values)}

# Load RL bundle for Q-models
with open(f"{PROJECT}/products/models/oracle_xau_validated.pkl","rb") as f:
    rl_bundle=pickle.load(f)
q_entry=rl_bundle['q_entry']

# Load setups and split by cutoff
all_setups=[]
for f in sorted(_glob.glob(f"{DATA}/setups_*_v72l.csv")):
    cid=int(os.path.basename(f).split('_')[1])
    df=pd.read_csv(f,parse_dates=["time"]); df['old_cid']=cid
    all_setups.append(df)
all_df=pd.concat(all_setups,ignore_index=True).sort_values("time").reset_index(drop=True)

# Compute PnL labels
pnl_labels=[]
for _,row in all_df.iterrows():
    tm=row["time"]
    if tm not in t2i: pnl_labels.append(0); continue
    idx=t2i[tm]; d=int(row["direction"]); ep=C[idx]; ea=atr[idx]
    if not np.isfinite(ea) or ea<=0: pnl_labels.append(0); continue
    tp=ep+d*2.0*ea-0.4; sl=ep-d*1.0*ea+0.4; end_idx=min(idx+41,len(C))
    outcome=0.0
    for k in range(idx+1,end_idx):
        if d==1:
            if L[k]<=sl: outcome=-1.0; break
            if H[k]>=tp: outcome=+2.0; break
        else:
            if H[k]>=sl: outcome=-1.0; break
            if L[k]<=tp: outcome=+2.0; break
    pnl_labels.append(outcome)
all_df['pnl_r']=pnl_labels

# Forward-fill features into swing
phys=all_df[['time']+V72L_FEATS].drop_duplicates('time',keep='last').sort_values('time')
swing=pd.merge_asof(swing.sort_values('time'),phys,on='time',direction='nearest')
for col in V72L_FEATS: swing[col]=swing[col].fillna(0)
ctx_arr=swing[V72L_FEATS].fillna(0).values.astype(np.float64)

# Split
train_setups=all_df[all_df["time"]<CUTOFF].reset_index(drop=True)
test_setups=all_df[all_df["time"]>=CUTOFF].reset_index(drop=True)
print(f" {_time.time()-t0:.0f}s — train={len(train_setups):,} test={len(test_setups):,}",flush=True)

# ═══ 2. Generate RL trades (train period only) ═══
print("[2/4] Generating RL training trades...",flush=True)
def gen_rl(setups_df):
    rows=[]
    for cid in [0,1,2,3,4]:
        if cid not in q_entry: continue
        g=setups_df[setups_df['old_cid']==cid]
        if len(g)<100: continue
        X=g[V72L_FEATS].fillna(0).values
        q_pred=q_entry[cid].predict(X)
        s=g[q_pred>=MIN_Q].copy(); s['rule']='RL'; s['cid']=cid
        rows.append(s)
    if not rows: return pd.DataFrame()
    return pd.concat(rows,ignore_index=True).sort_values("time").reset_index(drop=True)

train_trades=gen_rl(train_setups)
print(f"  Train RL trades: {len(train_trades):,}",flush=True)

# Load holdout RL trades for testing
test_trades=pd.read_csv(f"{PROJECT}/experiments/v84_rl_entry/v84_rl_trades.csv",parse_dates=["time"])
test_trades=test_trades.sort_values("time").reset_index(drop=True)
print(f"  Test RL trades: {len(test_trades):,}",flush=True)

# ═══ 3. Build samples + train ═══
print("[3/4] Building samples + training...",flush=True)

def build_samples(trades_df, max_samples=200_000):
    rows=[]
    for _,trade in trades_df.iterrows():
        tm=trade["time"]
        if tm not in t2i: continue
        ei=t2i[tm]; d=int(trade["direction"])
        ep=C[ei]; ea=atr[ei]
        if not np.isfinite(ea) or ea<=0: continue
        actual_pnl=trade["pnl_r"] if "pnl_r" in trade else trade.get("pnl_R",0)
        # Find max MTM
        max_seen=0.0
        for k in range(1,min(MAX_HOLD,n-ei-1)):
            bar=ei+k; mtm=d*(C[bar]-ep)/ea
            if mtm<=SL_HARD: break
            if mtm>max_seen: max_seen=mtm
        for k in range(MIN_HOLD,min(MAX_HOLD,n-ei-1)):
            bar=ei+k; mtm=d*(C[bar]-ep)/ea
            if mtm<=SL_HARD: break
            remaining=actual_pnl-mtm
            # target: giveback
            min_future=mtm
            for k2 in range(k+1,min(MAX_HOLD,n-ei-1)):
                bar2=ei+k2; mtm2=d*(C[bar2]-ep)/ea
                if mtm2<=SL_HARD: min_future=SL_HARD; break
                min_future=min(min_future,mtm2)
            label_gb=1 if (mtm>1.0 and (mtm-min_future)>0.5) else 0
            label_up=1 if remaining>0.5 else 0
            dist_sl=mtm-SL_HARD; dist_tp=2.0-mtm
            vol10=np.std([d*(C[max(0,bar-j)]-C[max(0,bar-j-1)])/ea for j in range(min(10,bar))]) if bar>5 else 0
            mom3=d*(C[bar]-C[max(0,bar-3)])/ea if bar>=3 else 0
            mom10=d*(C[bar]-C[max(0,bar-10)])/ea if bar>=10 else 0
            row=[mtm,max_seen,max_seen-mtm,float(k),float(MAX_HOLD-k),dist_sl,dist_tp,vol10,mom3,mom10]
            for j in range(len(V72L_FEATS)): row.append(float(ctx_arr[bar,j]))
            row.extend([label_up,label_gb])
            rows.append(row)
            if len(rows)>=max_samples: break
        if len(rows)>=max_samples: break
    return pd.DataFrame(rows,columns=ENRICHED+['label_up','label_gb'])

print("  Building TRAIN samples...",end='',flush=True); t1=_time.time()
trn_ed=build_samples(train_trades)
print(f" {_time.time()-t1:.0f}s — {len(trn_ed):,}",flush=True)

print("  Building TEST samples...",end='',flush=True); t1=_time.time()
tst_ed=build_samples(test_trades)
print(f" {_time.time()-t1:.0f}s — {len(tst_ed):,}",flush=True)

# Train 2 key heads
models={}
for label,desc in [('label_up','upside'),('label_gb','giveback')]:
    pos=(trn_ed[label]==1).sum(); sw=len(trn_ed)/max(pos,1)/2
    print(f"  Training {desc} ({pos:,} pos)...",end='',flush=True); t1=_time.time()
    mdl=XGBClassifier(n_estimators=300,max_depth=5,learning_rate=0.05,
                       subsample=0.8,colsample_bytree=0.8,eval_metric='logloss',verbosity=0)
    mdl.fit(trn_ed[ENRICHED].values,trn_ed[label].values,
            sample_weight=np.where(trn_ed[label]==1,sw,1.0))
    models[label]=mdl
    proba=mdl.predict_proba(tst_ed[ENRICHED].values)[:,1]
    auc=roc_auc_score(tst_ed[label].values,proba)
    print(f" AUC={auc:.4f}",flush=True)

# ═══ 4. Test on holdout ═══
print("[4/4] Testing on holdout...",flush=True)

# Batch features for test trades
all_X=[]; trade_map=[]
for ti,(_,trade) in enumerate(test_trades.iterrows()):
    tm=trade["time"]
    if tm not in t2i: continue
    ei=t2i[tm]; d=int(trade["direction"])
    ep=C[ei]; ea=atr[ei]
    if not np.isfinite(ea) or ea<=0: continue
    max_seen=0.0
    for k in range(1,min(MAX_HOLD,n-ei-1)):
        bar=ei+k; mtm=d*(C[bar]-ep)/ea
        if mtm<=SL_HARD: break
        if mtm>max_seen: max_seen=mtm
    for k in range(MIN_HOLD,min(MAX_HOLD,n-ei-1)):
        bar=ei+k; mtm=d*(C[bar]-ep)/ea
        if mtm<=SL_HARD: break
        dist_sl=mtm-SL_HARD; dist_tp=2.0-mtm
        vol10=np.std([d*(C[max(0,bar-j)]-C[max(0,bar-j-1)])/ea for j in range(min(10,bar))]) if bar>5 else 0
        mom3=d*(C[bar]-C[max(0,bar-3)])/ea if bar>=3 else 0
        mom10=d*(C[bar]-C[max(0,bar-10)])/ea if bar>=10 else 0
        feats=[mtm,max_seen,max_seen-mtm,float(k),float(MAX_HOLD-k),dist_sl,dist_tp,vol10,mom3,mom10]
        for j in range(len(V72L_FEATS)): feats.append(float(ctx_arr[bar,j]))
        all_X.append(feats); trade_map.append((ti,k,mtm,d,ep,ea,ei))
    if len(all_X)>300000: break

all_X=np.array(all_X,dtype=np.float32)
p_up=models['label_up'].predict_proba(all_X)[:,1]
p_gb=models['label_gb'].predict_proba(all_X)[:,1]

def sim(pu,pg):
    results=[]; ti_curr=-1; exit_k=None; ei_p=ep_p=ea_p=d_p=0
    for idx,(ti,k,mtm,d,ep,ea,ei) in enumerate(trade_map):
        if ti!=ti_curr:
            if ti_curr>=0 and exit_k is None:
                results.append(d_p*(C[min(ei_p+MAX_HOLD,n-1)]-ep_p)/ea_p)
            ti_curr=ti; exit_k=None; ei_p=ei; d_p=d; ep_p=ep; ea_p=ea
        if exit_k is not None: continue
        if p_gb[idx]>pg and p_up[idx]<pu:
            exit_k=k; results.append(d*(C[ei+k]-ep)/ea)
    if exit_k is None and ti_curr>=0:
        results.append(d_p*(C[min(ei_p+MAX_HOLD,n-1)]-ep_p)/ea_p)
    return results

# Baselines
base=[]; ml_base=[]
old_ctx=['hurst_rs','ou_theta','entropy_rate','kramers_up','wavelet_er',
         'quantum_flow','quantum_flow_h4','vwap_dist']
ml_exit=rl_bundle.get("exit_mdl")

for _,trade in test_trades.iterrows():
    tm=trade["time"]
    if tm not in t2i: continue
    ei=t2i[tm]; d=int(trade["direction"])
    ep=C[ei]; ea=atr[ei]
    if not np.isfinite(ea) or ea<=0: continue
    eb=min(ei+MAX_HOLD,n-1)
    for k in range(1,MAX_HOLD+1):
        bar=ei+k; mtm=d*(C[bar]-ep)/ea
        if bar>=n: break
        if mtm<=SL_HARD: eb=bar; break
    base.append(d*(C[eb]-ep)/ea)
    
    # ML exit
    eb_ml=min(ei+MAX_HOLD,n-1)
    for k in range(1,MAX_HOLD+1):
        bar=ei+k
        if bar>=n: break
        mtm=d*(C[bar]-ep)/ea
        if mtm<=SL_HARD: eb_ml=bar; break
        if k<MIN_HOLD: continue
        p3=d*(C[bar-3]-ep)/ea if k>=3 else mtm
        old_f=np.array([mtm,float(k),mtm-p3]+[ctx_arr[bar,V72L_FEATS.index(c)] for c in old_ctx],dtype=np.float32)
        if ml_exit is not None:
            pe=float(ml_exit.predict_proba(old_f.reshape(1,-1))[0,1])
            if pe>=0.55: eb_ml=bar; break
    ml_base.append(d*(C[eb_ml]-ep)/ea)

base_total=sum(base); base_w=[p for p in base if p>0]; base_l=[p for p in base if p<=0]
base_pf=sum(base_w)/max(-sum(base_l),1e-9) if base_l else 99
ml_total=sum(ml_base); ml_w=[p for p in ml_base if p>0]; ml_l=[p for p in ml_base if p<=0]
ml_pf=sum(ml_w)/max(-sum(ml_l),1e-9) if ml_l else 99

print(f"\n  PROPER TRAIN/TEST SPLIT (pre-2024-12-12 train, post test)")
print(f"  Train samples: {len(trn_ed):,} | Test samples: {len(tst_ed):,}")
print(f"  AUC giveback: {roc_auc_score(tst_ed['label_gb'],models['label_gb'].predict_proba(tst_ed[ENRICHED].values)[:,1]):.4f}")
print(f"  AUC upside:   {roc_auc_score(tst_ed['label_up'],models['label_up'].predict_proba(tst_ed[ENRICHED].values)[:,1]):.4f}")
print(f"\n  Baseline (hard SL):  PF={base_pf:.2f}  Total={base_total:+.0f}R")
print(f"  Current ML exit:     PF={ml_pf:.2f}  Total={ml_total:+.0f}R")
print(f"\n  {'Policy':<25s} {'PF':>6s} {'Total':>8s}  vs ML exit")
print(f"  {'-'*25} {'-'*6} {'-'*8}  {'-'*10}")

for pu,pg in [(0.4,0.6),(0.3,0.5),(0.5,0.7),(0.3,0.6),(0.4,0.5),(0.3,0.4)]:
    pnls=sim(pu,pg); total=sum(pnls)
    w=[p for p in pnls if p>0]; l=[p for p in pnls if p<=0]
    pf=sum(w)/max(-sum(l),1e-9) if l else 99
    print(f"  gb>{pg} up<{pu}             {pf:6.2f} {total:+8.0f}  {total-ml_total:+8.0f}R")

print(f"\nDone. {_time.time()-t0:.0f}s")
