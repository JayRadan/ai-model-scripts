"""
v84 — Full RL: Entry + Exit with Q-Learning
============================================
RL Entry: 5 XGBoost regressors predict expected PnL from entering
RL Exit: 1 XGBoost regressor predicts optimal remaining PnL from current state
         Exit when current_PnL > predicted_remaining * 0.9 (near-optimal)

No hard SL/TP — the RL exit model learns when to close.
"""
from __future__ import annotations
import glob, os, json, time as _time
import numpy as np, pandas as pd
from xgboost import XGBRegressor, XGBClassifier

PROJECT = "/home/jay/Desktop/new-model-zigzag"
DATA = f"{PROJECT}/data"
OUT = f"{PROJECT}/experiments/v84_rl_entry"
os.makedirs(OUT, exist_ok=True)

GLOBAL_CUTOFF = pd.Timestamp("2024-12-12 00:00:00")
V72L_FEATS = ["hurst_rs","ou_theta","entropy_rate","kramers_up","wavelet_er",
    "vwap_dist","hour_enc","dow_enc","quantum_flow","quantum_flow_h4",
    "quantum_momentum","quantum_vwap_conf","quantum_divergence","quantum_div_strength",
    "vpin","sig_quad_var","har_rv_ratio","hawkes_eta"]
META_FEATS = V72L_FEATS + ["direction","cid"]
EXIT_FEATS = ["unrealized_pnl_R","bars_held","pnl_velocity",
    "hurst_rs","ou_theta","entropy_rate","kramers_up","wavelet_er",
    "quantum_flow","quantum_flow_h4","vwap_dist"]
MAX_HOLD=60; MIN_HOLD=2; MAX_FWD=40
TP_MULT=2.0; SL_MULT=1.0; MIN_Q = 0.3

print("="*60)
print("  v84 — FULL RL: Entry + Exit with Q-Learning")
print("="*60)

# ═══ Load ═══
print("\n[1/7] Loading data...", flush=True); t0=_time.time()
swing=pd.read_csv(f"{DATA}/swing_v5_xauusd.csv",parse_dates=["time"]); swing=swing.sort_values("time").reset_index(drop=True)
C=swing["close"].values.astype(np.float64); H=swing["high"].values.astype(np.float64); L=swing["low"].values.astype(np.float64)
tr=np.concatenate([[H[0]-L[0]],np.maximum.reduce([H[1:]-L[1:],np.abs(H[1:]-C[:-1]),np.abs(L[1:]-C[:-1])])])
atr=pd.Series(tr).rolling(14,min_periods=14).mean().values
time_to_idx={}
for i,t in enumerate(swing["time"].values): time_to_idx[t]=i

fp_new=pd.read_csv(f"{PROJECT}/experiments/v83_range_position_filter/regime_fingerprints_4h.csv")
fp_new["center_time"]=pd.to_datetime(fp_new["center_time"]); fp_new=fp_new.sort_values("center_time")
new_regime=np.full(len(swing),-1,dtype=int)
for _,row in fp_new.iterrows():
    s=int(row["start_idx"]); e=int(row["end_idx"])
    if 0<=s<e<=len(swing): new_regime[s:e]=int(row["new_label"])

all_setups=[]
for f in sorted(glob.glob(f"{DATA}/setups_*_v72l.csv")):
    cid=int(os.path.basename(f).split("_")[1])
    df=pd.read_csv(f,parse_dates=["time"]); df["old_cid"]=cid; all_setups.append(df)
all_df=pd.concat(all_setups,ignore_index=True).sort_values("time").reset_index(drop=True)
new_cids=[]
for _,row in all_df.iterrows():
    tm=row["time"]
    if tm in time_to_idx:
        idx=time_to_idx[tm]
        new_cids.append(new_regime[idx] if 0<=idx<len(new_regime) else -1)
    else: new_cids.append(-1)
all_df["cid"]=new_cids; all_df=all_df[all_df["cid"]>=0].reset_index(drop=True)

pnl_labels=[]
for _,row in all_df.iterrows():
    tm=row["time"]
    if tm not in time_to_idx: pnl_labels.append(0); continue
    idx=time_to_idx[tm]; d=int(row["direction"]); ep=C[idx]; ea=atr[idx]
    if not np.isfinite(ea) or ea<=0: pnl_labels.append(0); continue
    tp=ep+d*TP_MULT*ea-0.4; sl=ep-d*SL_MULT*ea+0.4; end_idx=min(idx+MAX_FWD+1,len(C))
    outcome=0
    for k in range(idx+1,end_idx):
        if d==+1:
            if L[k]<=sl: outcome=-1.0; break
            if H[k]>=tp: outcome=+2.0; break
        else:
            if H[k]>=sl: outcome=-1.0; break
            if L[k]<=tp: outcome=+2.0; break
    pnl_labels.append(outcome)
all_df["pnl_r"]=pnl_labels

phys_lookup=all_df[["time"]+V72L_FEATS].drop_duplicates("time",keep="last").set_index("time")
for col in V72L_FEATS: swing[col]=phys_lookup[col].reindex(swing.set_index("time").index,method="ffill").values; swing[col]=swing[col].fillna(0)

train_setups=all_df[all_df["time"]<GLOBAL_CUTOFF].reset_index(drop=True)
test_setups=all_df[all_df["time"]>=GLOBAL_CUTOFF].reset_index(drop=True)
print(f"  {_time.time()-t0:.0f}s — train={len(train_setups):,} test={len(test_setups):,}", flush=True)

# ═══ Step 2: Train RL Entry Q-functions ═══
print("\n[2/7] Training RL Entry Q-models...", flush=True); t0=_time.time()
q_entry={}
for cid in range(5):
    g=train_setups[train_setups["cid"]==cid]
    if len(g)<500: continue
    X=g[V72L_FEATS].fillna(0).values; y=g["pnl_r"].values
    mdl=XGBRegressor(n_estimators=300,max_depth=4,learning_rate=0.05,
                     subsample=0.8,colsample_bytree=0.8,verbosity=0)
    mdl.fit(X,y); q_entry[cid]=mdl
    pred=mdl.predict(X); pos=(pred>MIN_Q).sum()
    pos_wr=(y[pred>MIN_Q]>0).mean() if pos>0 else 0
    print(f"  C{cid}: {pos:,} with Q>{MIN_Q}R, true WR={pos_wr:.1%}", flush=True)
print(f"  {len(q_entry)} entry models, {_time.time()-t0:.0f}s", flush=True)

# ═══ Step 3: Train RL Exit Q-function ═══
print("\n[3/7] Training RL Exit Q-model (optimal remaining PnL)...", flush=True); t0=_time.time()

# Generate exit training data: for every bar of every long simulation, 
# compute "optimal PnL from here"
ctx_np=swing[EXIT_FEATS[3:]].fillna(0).values.astype(np.float64)
exit_rows=[]
for _,s in train_setups.iterrows():
    tm=s["time"]
    if tm not in time_to_idx: continue
    ei=time_to_idx[tm]; d=int(s["direction"]); ep=C[ei]; ea=atr[ei]
    if not np.isfinite(ea) or ea<=0: continue
    end=min(ei+MAX_HOLD+1,len(C))
    if end-ei<10: continue
    pnls=np.array([d*(C[k]-ep)/ea for k in range(ei+1,end)])
    if len(pnls)<5: continue
    # For each bar in the trade, what's the optimal future PnL?
    for b in range(len(pnls)):
        bi=ei+1+b; cp=pnls[b]
        # Optimal from here = max PnL achievable in remaining bars
        remaining=pnls[b:]
        optimal_remaining=remaining.max() if len(remaining)>0 else cp
        v=cp-pnls[b-3] if b>=3 else (cp-pnls[0] if b>=1 else 0.0)
        row={"current_pnl_R":cp,"bars_held":float(b+1),"pnl_velocity":v,
             "optimal_remaining_R":optimal_remaining}
        if bi<len(swing):
            for j,f_ in enumerate(EXIT_FEATS[3:]): row[f_]=float(ctx_np[bi,j])
        else:
            for f_ in EXIT_FEATS[3:]: row[f_]=0.0
        exit_rows.append(row)

ed=pd.DataFrame(exit_rows).sample(frac=0.5, random_state=42)  # Sample for speed
print(f"  {len(ed):,} exit training rows (sampled 50%)", flush=True)

RL_EXIT_FEATS = ["current_pnl_R","bars_held","pnl_velocity",
    "hurst_rs","ou_theta","entropy_rate","kramers_up","wavelet_er",
    "quantum_flow","quantum_flow_h4","vwap_dist"]

q_exit=XGBRegressor(n_estimators=300,max_depth=5,learning_rate=0.05,
                    subsample=0.8,colsample_bytree=0.8,verbosity=0)
q_exit.fit(ed[RL_EXIT_FEATS].fillna(0).values, ed["optimal_remaining_R"].values)
print(f"  Exit Q-model trained, {_time.time()-t0:.0f}s", flush=True)

# ═══ Step 4: Generate RL entries ═══
print("\n[4/7] Generating RL entries...", flush=True); t0=_time.time()

def gen_entries(setups_df, q_entry):
    rows=[]
    for cid in sorted(setups_df["cid"].unique()):
        if cid not in q_entry: continue
        g=setups_df[setups_df["cid"]==cid]; X=g[V72L_FEATS].fillna(0).values
        q_pred=q_entry[cid].predict(X)
        s=g[q_pred>=MIN_Q].copy(); s["q_value"]=q_pred[q_pred>=MIN_Q]; s["rule"]="RL"
        rows.append(s)
    return pd.concat(rows,ignore_index=True).sort_values("time").reset_index(drop=True) if rows else pd.DataFrame()

rl_train=gen_entries(train_setups,q_entry); rl_test=gen_entries(test_setups,q_entry)
print(f"  Train={len(rl_train):,} Test={len(rl_test):,} entries", flush=True)

# ═══ Step 5: Confirm + Meta ═══
print("\n[5/7] Training confirm + meta...", flush=True); t0=_time.time()

def train_conf(train,features,tag):
    mdls,thrs={},{}
    for (cid,rule),grp in train.groupby(["cid","rule"]):
        if len(grp)<100: continue
        grp=grp.sort_values("time").reset_index(drop=True)
        s=int(len(grp)*0.80); trn,vd=grp.iloc[:s],grp.iloc[s:]
        if len(vd)<20: continue
        mdl=XGBClassifier(n_estimators=200,max_depth=3,learning_rate=0.05,
                          subsample=0.8,colsample_bytree=0.8,eval_metric="logloss",verbosity=0)
        mdl.fit(trn[features].fillna(0).values,trn["label"].astype(int).values)
        proba=mdl.predict_proba(vd[features].fillna(0).values)[:,1]
        y_vd=vd["label"].astype(int).values
        best_thr,best_pf=0.5,0
        for thr in np.arange(0.30,0.70,0.05):
            m=proba>=thr
            if m.sum()<5: continue
            w=y_vd[m].sum(); l=m.sum()-w
            if l==0: continue
            pf=(w*2.0)/(l*1.0)
            if pf>best_pf: best_pf,best_thr=pf,float(thr)
        m=proba>=best_thr
        if m.sum()<5 or best_pf<0.8: continue
        mdls[(cid,rule)]=mdl; thrs[(cid,rule)]=best_thr
    print(f"  {tag}: {len(mdls)} models", flush=True)
    return mdls,thrs

def confirm(setups,mdls,thrs,features):
    rows=[]
    for (cid,rule),grp in setups.groupby(["cid","rule"]):
        if (cid,rule) not in mdls: continue
        X=grp[features].fillna(0).values; p=mdls[(cid,rule)].predict_proba(X)[:,1]
        rows.append(grp[p>=thrs[(cid,rule)]].copy())
    return pd.concat(rows,ignore_index=True).sort_values("time").reset_index(drop=True) if rows else pd.DataFrame()

c_mdls,c_thrs=train_conf(rl_train,V72L_FEATS,"rl-confirm")
tc=confirm(rl_train,c_mdls,c_thrs,V72L_FEATS)
print(f"  {len(tc):,} confirmed", flush=True)

# Meta (same as v83c)
ctx_cols=["hurst_rs","ou_theta","entropy_rate","kramers_up","wavelet_er",
          "quantum_flow","quantum_flow_h4","vwap_dist"]
ctx_arr=swing[ctx_cols].fillna(0).values.astype(np.float64); n=len(C)

def simulate_rl(confirmed, swing, atr, q_exit):
    """RL exit: exit when current_pnl > predicted_optimal * 0.9"""
    entries=[]
    for _,s in confirmed.iterrows():
        tm=s["time"]
        if tm not in time_to_idx: continue
        ei=time_to_idx[tm]; ea=atr[ei]
        if not np.isfinite(ea) or ea<=0: continue
        entries.append((ei,int(s["direction"]),tm,int(s["cid"]),s.get("rule","RL")))
    N=len(entries)
    if N==0: return pd.DataFrame()
    rows_s=[]
    for (ei,d,tm,cid_v,rule_v) in entries:
        ep=C[ei]; ea=atr[ei]
        xi=None; xr="max"
        for k in range(1,MAX_HOLD+1):
            bar=ei+k
            if bar>=n: break
            cp=d*(C[bar]-ep)/ea
            if k<MIN_HOLD: continue
            
            # RL Exit: predict optimal remaining, exit if near it
            p3=d*(C[bar-3]-ep)/ea if k>=3 and bar>=3 else cp
            vec=np.array([cp,float(k),cp-p3]+
                        [float(ctx_arr[bar,j]) for j in range(len(ctx_cols))], dtype=np.float32)
            pred_optimal=q_exit.predict(vec.reshape(1,-1)[:, :len(RL_EXIT_FEATS)])[0]
            
            if cp>0 and cp>pred_optimal*0.9:
                xi=bar; xr="rl_exit"; break
            
            # Hard safety: don't let it run to -100R
            if cp<-4.0: xi=bar; xr="rl_safety"; break
        
        if xi is None: xi=min(ei+MAX_HOLD,n-1); xr="max"
        pnl=d*(C[xi]-ep)/ea
        rows_s.append({"time":tm,"cid":cid_v,"rule":rule_v,"direction":d,"bars":xi-ei,"pnl_R":pnl,"exit":xr})
    return pd.DataFrame(rows_s)

# Simulate train for meta
tt=simulate_rl(tc,swing,atr,q_exit)
tc["direction"]=tc["direction"].astype(int); tc["cid"]=tc["cid"].astype(int)
md_=tt.merge(tc[["time","cid","rule"]+V72L_FEATS],on=["time","cid","rule"],how="left")
md_["meta_label"]=(md_["pnl_R"]>0).astype(int); md_=md_.sort_values("time").reset_index(drop=True)
s_=int(len(md_)*0.80); mtr=md_.iloc[:s_]; mvd=md_.iloc[s_:]
meta_mdl=XGBClassifier(n_estimators=300,max_depth=4,learning_rate=0.05,
                       subsample=0.8,colsample_bytree=0.8,eval_metric="logloss",verbosity=0)
meta_mdl.fit(mtr[META_FEATS].fillna(0).values,mtr["meta_label"].values)
pv=meta_mdl.predict_proba(mvd[META_FEATS].fillna(0).values)[:,1]; pn=mvd["pnl_R"].values
cands=[]
for thr in np.arange(0.40,0.80,0.025):
    mx=pv>=thr
    if mx.sum()<20: continue
    pt=pn[mx]; pf_=pt[pt>0].sum()/max(-pt[pt<=0].sum(),1e-9); cands.append((thr,mx.sum(),pf_))
vc=[c for c in cands if c[2]>=pn[pn>0].sum()/max(-pn[pn<=0].sum(),1e-9)*0.95] or cands
vc.sort(key=lambda c:(-c[2],c[1])); best_thr=vc[0][0]
print(f"  Meta thr={best_thr:.3f}, {_time.time()-t0:.0f}s", flush=True)

# ═══ Step 6: Holdout + Kill-switch ═══
print("\n[6/7] Holdout simulation...", flush=True); t0=_time.time()
tec=confirm(rl_test,c_mdls,c_thrs,V72L_FEATS)
tec["direction"]=tec["direction"].astype(int); tec["cid"]=tec["cid"].astype(int)
pm_=meta_mdl.predict_proba(tec[META_FEATS].fillna(0).values)[:,1]
tec_m=tec[pm_>=best_thr].copy()
trades_out=simulate_rl(tec_m,swing,atr,q_exit)

# Kill-switch
streak={}; last_kill={}; keep_idx=[]
for i,(_,t) in enumerate(trades_out.sort_values("time").iterrows()):
    key=(int(t["cid"]),int(t["direction"]))
    if key in last_kill:
        if (t["time"]-last_kill[key]).total_seconds()/3600<12: continue
        else: del last_kill[key]; streak[key]=0
    keep_idx.append(i)
    if t["pnl_R"]<=0: streak[key]=streak.get(key,0)+1
    else: streak[key]=0
    if streak.get(key,0)>=3: last_kill[key]=t["time"]
trades_final=trades_out.iloc[keep_idx]

def report(df,tag):
    if len(df)==0: print(f"  {tag}: NO TRADES"); return None
    w=df[df["pnl_R"]>0]; l=df[df["pnl_R"]<=0]
    pf=w["pnl_R"].sum()/max(-l["pnl_R"].sum(),1e-9)
    wr=len(w)/len(df); total=df["pnl_R"].sum()
    print(f"\n  {tag}: n={len(df)} WR={wr:.1%} PF={pf:.2f} Total={total:+.1f}R")
    names={0:"Up",1:"MR",2:"TR",3:"Down",4:"HV"}
    for cv in sorted(df["cid"].unique()):
        g=df[df["cid"]==cv]; ww=g[g["pnl_R"]>0]; ll=g[g["pnl_R"]<=0]
        ppf=ww["pnl_R"].sum()/max(-ll["pnl_R"].sum(),1e-9)
        print(f"    C{cv} {names[cv]}: n={len(g):,} WR={len(ww)/len(g):.1%} PF={ppf:.2f} R={g['pnl_R'].sum():+.1f}")
    return {"n":len(df),"pf":pf,"wr":wr,"total":total}

r_rl=report(trades_final,"RL ENTRY + RL EXIT")

# ═══ Step 7: Compare ═══
print(f"\n[7/7] Comparison...")
print(f"  {'Model':<25} {'PF':>6} {'WR':>7} {'Trades':>8} {'TotalR':>10}")
print(f"  {'─'*25} {'─'*6} {'─'*7} {'─'*8} {'─'*10}")

# v83c
v83c_path=f"{PROJECT}/experiments/v83_range_position_filter/v83c_final_trades.csv"
if os.path.exists(v83c_path):
    v83c=pd.read_csv(v83c_path,parse_dates=["time"])
    w=v83c[v83c["pnl_R"]>0]; l=v83c[v83c["pnl_R"]<=0]
    print(f"  {'Rule-based v83c':<25} {w['pnl_R'].sum()/max(-l['pnl_R'].sum(),1e-9):>6.2f} {len(w)/len(v83c):>6.1%} {len(v83c):>8,} {v83c['pnl_R'].sum():>+10.1f}")

# RL entry only
rl_entry_path=f"{OUT}/v84_rl_trades.csv"
if os.path.exists(rl_entry_path):
    rle=pd.read_csv(rl_entry_path,parse_dates=["time"])
    w=rle[rle["pnl_R"]>0]; l=rle[rle["pnl_R"]<=0]
    print(f"  {'RL Entry only':<25} {w['pnl_R'].sum()/max(-l['pnl_R'].sum(),1e-9):>6.2f} {len(w)/len(rle):>6.1%} {len(rle):>8,} {rle['pnl_R'].sum():>+10.1f}")

if r_rl:
    expt=r_rl['total']/r_rl['n'] if r_rl['n']>0 else 0
    print(f"  {'RL Entry + RL Exit':<25} {r_rl['pf']:>6.2f} {r_rl['wr']:>6.1%} {r_rl['n']:>8,} {r_rl['total']:>+10.1f}")

trades_final.to_csv(f"{OUT}/v84_full_rl_trades.csv",index=False)
print(f"\nSaved: {OUT}/v84_full_rl_trades.csv")
print(f"Done.")
