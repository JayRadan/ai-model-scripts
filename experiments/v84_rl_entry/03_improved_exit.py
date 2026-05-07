"""
v84b — Improved ML Exit: Better than hard SL + time-decay
===========================================================
Trains an ML exit model that predicts "maximum PnL achievable from here."
Exit when current_PnL >= predicted_max * 0.95 (captured 95% of potential).
No hard SL — the model decides when to cut losses.
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
MAX_HOLD=60; MIN_HOLD=2; MAX_FWD=40
TP_MULT=2.0; SL_MULT=1.0; MIN_Q=0.25  # Slightly lower to get more entries

# ── Improved exit features: more context ──
BETTER_EXIT_FEATS = [
    "current_pnl_R",      # Current PnL in R-units
    "bars_held",           # How long in trade
    "pnl_velocity",        # Recent PnL change rate
    "drawdown_from_peak",  # Current PnL - peak PnL (negative = underwater)
    "peak_pnl",            # Best PnL achieved so far
    "hurst_rs","ou_theta","entropy_rate","kramers_up","wavelet_er",
    "quantum_flow","quantum_flow_h4","vwap_dist",
]

print("="*60)
print("  v84b — IMPROVED ML EXIT (no hard SL)")
print("="*60)

# ═══ Load ═══
print("\n[1/5] Loading data...", flush=True); t0=_time.time()
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
print(f"  {_time.time()-t0:.0f}s", flush=True)

# ═══ Train RL Entry (same as before) ═══
print("\n[2/5] Training RL Entry...", flush=True); t0=_time.time()
q_entry={}
for cid in range(5):
    g=train_setups[train_setups["cid"]==cid]
    if len(g)<500: continue
    X=g[V72L_FEATS].fillna(0).values; y=g["pnl_r"].values
    mdl=XGBRegressor(n_estimators=300,max_depth=4,learning_rate=0.05,
                     subsample=0.8,colsample_bytree=0.8,verbosity=0)
    mdl.fit(X,y); q_entry[cid]=mdl
    pred=mdl.predict(X); pos=(pred>MIN_Q).sum()
    print(f"  C{cid}: {pos:,} with Q>{MIN_Q}R ({(pred>MIN_Q).mean()*100:.1f}%)", flush=True)

def gen(setups_df,q_entry):
    rows=[]
    for cid in sorted(setups_df["cid"].unique()):
        if cid not in q_entry: continue
        g=setups_df[setups_df["cid"]==cid]; X=g[V72L_FEATS].fillna(0).values
        q_pred=q_entry[cid].predict(X)
        s=g[q_pred>=MIN_Q].copy(); s["q_value"]=q_pred[q_pred>=MIN_Q]; s["rule"]="RL"
        rows.append(s)
    return pd.concat(rows,ignore_index=True).sort_values("time").reset_index(drop=True) if rows else pd.DataFrame()

print(f"  {len(q_entry)} entry models, {_time.time()-t0:.0f}s", flush=True)

# ═══ Train BETTER exit model ═══
print("\n[3/5] Training improved exit model...", flush=True); t0=_time.time()

ctx_np=swing[BETTER_EXIT_FEATS[5:]].fillna(0).values.astype(np.float64)
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
    
    peak_so_far=pnls[0]
    for b in range(len(pnls)):
        bi=ei+1+b; cp=pnls[b]
        peak_so_far=max(peak_so_far,cp)
        # Target: max PnL achievable from here (next 60 bars)
        remaining=pnls[b:]
        max_remaining=remaining.max() if len(remaining)>0 else cp
        
        p3=pnls[b-3] if b>=3 else pnls[0]
        v=cp-p3
        
        row={
            "current_pnl_R": cp,
            "bars_held": float(b+1),
            "pnl_velocity": v,
            "drawdown_from_peak": cp - peak_so_far,
            "peak_pnl": peak_so_far,
            "target_max_remaining": max_remaining,
        }
        if bi<len(swing):
            for j,f_ in enumerate(BETTER_EXIT_FEATS[5:]): row[f_]=float(ctx_np[bi,j])
        else:
            for f_ in BETTER_EXIT_FEATS[5:]: row[f_]=0.0
        exit_rows.append(row)

# Sample for speed (full dataset is too large)
ed=pd.DataFrame(exit_rows)
n_samples=min(300000,len(ed))
ed_sample=ed.sample(n=n_samples,random_state=42)
print(f"  {len(ed):,} rows → sampled {n_samples:,} for training", flush=True)

exit_mdl=XGBRegressor(n_estimators=200,max_depth=5,learning_rate=0.05,
                      subsample=0.8,colsample_bytree=0.8,verbosity=0)
exit_mdl.fit(ed_sample[BETTER_EXIT_FEATS].fillna(0).values, ed_sample["target_max_remaining"].values)
print(f"  Exit model trained, {_time.time()-t0:.0f}s", flush=True)

# ═══ Simulate with new exit ═══
print("\n[4/5] Simulating (RL entry + improved exit + meta + kill-switch)...", flush=True); t0=_time.time()

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

rl_train=gen(train_setups,q_entry); rl_test=gen(test_setups,q_entry)
c_mdls,c_thrs=train_conf(rl_train,V72L_FEATS,"confirm")
tc=confirm(rl_train,c_mdls,c_thrs,V72L_FEATS)
print(f"  {len(tc):,} confirmed", flush=True)

ctx_cols=["hurst_rs","ou_theta","entropy_rate","kramers_up","wavelet_er",
          "quantum_flow","quantum_flow_h4","vwap_dist"]
ctx_arr=swing[ctx_cols].fillna(0).values.astype(np.float64); n=len(C)

def simulate_improved(confirmed, swing, atr, exit_mdl):
    """Improved exit: predict max_remaining, exit when captured 90% of potential.
    For losing trades (negative PnL): exit if predicted recovery < 0.5R."""
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
        xi=None; xr="max"; peak_so_far=0
        for k in range(1,MAX_HOLD+1):
            bar=ei+k
            if bar>=n: break
            cp=d*(C[bar]-ep)/ea
            peak_so_far=max(peak_so_far,cp)
            if k<MIN_HOLD: continue
            
            p3=d*(C[bar-3]-ep)/ea if k>=3 and bar>=3 else cp
            dd=cp-peak_so_far
            
            vec=np.array([cp,float(k),cp-p3,dd,peak_so_far]+
                        [float(ctx_arr[bar,j]) for j in range(len(ctx_cols))], dtype=np.float32)
            pred_max=exit_mdl.predict(vec.reshape(1,-1)[:,:len(BETTER_EXIT_FEATS)])[0]
            
            # Exit logic:
            # Winner: exit when we've captured 90% of predicted max
            if cp>0.5 and cp>=pred_max*0.9:
                xi=bar; xr="ml_peak"; break
            # Loser: exit when model says no recovery expected
            if cp<-0.5 and pred_max<0.3:
                xi=bar; xr="ml_cut"; break
            # Deep loss: model says don't wait for -50R
            if cp<-3.0 and pred_max<cp*0.5:
                xi=bar; xr="ml_deep"; break
        
        if xi is None: xi=min(ei+MAX_HOLD,n-1); xr="max"
        pnl=d*(C[xi]-ep)/ea
        rows_s.append({"time":tm,"cid":cid_v,"rule":rule_v,"direction":d,"bars":xi-ei,"pnl_R":pnl,"exit":xr})
    return pd.DataFrame(rows_s)

# Meta
tt=simulate_improved(tc,swing,atr,exit_mdl)
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
print(f"  Meta={best_thr:.3f}, {_time.time()-t0:.0f}s", flush=True)

# ═══ Holdout + kill-switch ═══
print("\n[5/5] Holdout...", flush=True)
tec=confirm(rl_test,c_mdls,c_thrs,V72L_FEATS)
tec["direction"]=tec["direction"].astype(int); tec["cid"]=tec["cid"].astype(int)
pm_=meta_mdl.predict_proba(tec[META_FEATS].fillna(0).values)[:,1]
tec_m=tec[pm_>=best_thr].copy()
trades_out=simulate_improved(tec_m,swing,atr,exit_mdl)

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
    avg_win=w["pnl_R"].mean() if len(w)>0 else 0
    avg_loss=l["pnl_R"].mean() if len(l)>0 else 0
    print(f"\n  {tag}: n={len(df)} WR={wr:.1%} PF={pf:.2f} Total={total:+.1f}R "
          f"WinAvg={avg_win:+.2f}R LossAvg={avg_loss:+.2f}R")
    names={0:"Up",1:"MR",2:"TR",3:"Down",4:"HV"}
    for cv in sorted(df["cid"].unique()):
        g=df[df["cid"]==cv]; ww=g[g["pnl_R"]>0]; ll=g[g["pnl_R"]<=0]
        ppf=ww["pnl_R"].sum()/max(-ll["pnl_R"].sum(),1e-9)
        print(f"    C{cv} {names[cv]}: n={len(g):,} WR={len(ww)/len(g):.1%} PF={ppf:.2f} WinAvg={ww['pnl_R'].mean():+.2f}R LossAvg={ll['pnl_R'].mean():+.2f}R" if len(ll)>0 else "")
    return {"n":len(df),"pf":pf,"wr":wr,"total":total}

r=report(trades_final,"RL Entry + IMPROVED Exit")

# Compare
print(f"\n  Comparison:")
v83c_path=f"{PROJECT}/experiments/v83_range_position_filter/v83c_final_trades.csv"
if os.path.exists(v83c_path):
    v83c=pd.read_csv(v83c_path,parse_dates=["time"])
    w=v83c[v83c["pnl_R"]>0]; l=v83c[v83c["pnl_R"]<=0]
    print(f"    Rule-based v83c:  PF={w['pnl_R'].sum()/max(-l['pnl_R'].sum(),1e-9):.2f}  WR={len(w)/len(v83c):.1%}  n={len(v83c):,}")
    print(f"    RL Entry only:    PF=4.21  WR=70.4%  n=1,401  (from previous run)")

if r:
    print(f"    RL Entry + NewExit: PF={r['pf']:.2f}  WR={r['wr']:.1%}  n={r['n']:,}")

trades_final.to_csv(f"{OUT}/v84_improved_exit.csv",index=False)
print(f"\nSaved: {OUT}/v84_improved_exit.csv")
print(f"Done.")
