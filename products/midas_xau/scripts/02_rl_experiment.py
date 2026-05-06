"""Deploy RL Entry for Midas XAU (v6, 14 features, no meta gate)"""
import glob, os, pickle, time as _time
import numpy as np, pandas as pd
from xgboost import XGBRegressor, XGBClassifier

PROJECT = "/home/jay/Desktop/new-model-zigzag"
DATA = f"{PROJECT}/data"
OUT = f"{PROJECT}/experiments/v84_rl_entry"
SERVER = "/home/jay/Desktop/my-agents-and-website/commercial/server/decision_engine"
os.makedirs(OUT, exist_ok=True)

GLOBAL_CUTOFF = pd.Timestamp("2024-12-12 00:00:00")
V6_FEATS = ["hurst_rs","ou_theta","entropy_rate","kramers_up","wavelet_er",
    "vwap_dist","hour_enc","dow_enc","quantum_flow","quantum_flow_h4",
    "quantum_momentum","quantum_vwap_conf","quantum_divergence","quantum_div_strength"]
META_FEATS = V6_FEATS + ["direction","cid"]
EXIT_FEATS = ["unrealized_pnl_R","bars_held","pnl_velocity",
    "hurst_rs","ou_theta","entropy_rate","kramers_up","wavelet_er",
    "quantum_flow","quantum_flow_h4","vwap_dist"]
MAX_HOLD=60; MIN_HOLD=2; SL_HARD=4.0; EXIT_THRESHOLD=0.55; MIN_Q=0.3

print("="*60)
print("  RL Entry for MIDAS XAU")
print("="*60)

print("[1/4] Loading...", flush=True); t0=_time.time()
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
for f in sorted(glob.glob(f"{DATA}/setups_*_v6.csv")):
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
    tp=ep+d*2.0*ea-0.4; sl=ep-d*1.0*ea+0.4; end_idx=min(idx+41,len(C))
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

phys_lookup=all_df[["time"]+V6_FEATS].drop_duplicates("time",keep="last").set_index("time")
for col in V6_FEATS: swing[col]=phys_lookup[col].reindex(swing.set_index("time").index,method="ffill").values; swing[col]=swing[col].fillna(0)

train=all_df[all_df["time"]<GLOBAL_CUTOFF].reset_index(drop=True)
print(f"  {_time.time()-t0:.0f}s — train={len(train):,}", flush=True)

print("[2/4] Training RL Entry...", flush=True); t0=_time.time()
q_entry={}
for cid in range(5):
    g=train[train["cid"]==cid]
    if len(g)<500: continue
    X=g[V6_FEATS].fillna(0).values; y=g["pnl_r"].values
    mdl=XGBRegressor(n_estimators=300,max_depth=4,learning_rate=0.05,
                     subsample=0.8,colsample_bytree=0.8,verbosity=0)
    mdl.fit(X,y); q_entry[cid]=mdl
    pred=mdl.predict(X); pos=(pred>MIN_Q).sum()
    print(f"  C{cid}: {pos:,} with Q>{MIN_Q}R ({pos/len(g):.1%})", flush=True)

def gen(setups_df):
    rows=[]
    for cid in sorted(setups_df["cid"].unique()):
        if cid not in q_entry: continue
        g=setups_df[setups_df["cid"]==cid]; X=g[V6_FEATS].fillna(0).values
        q_pred=q_entry[cid].predict(X)
        s=g[q_pred>=MIN_Q].copy(); s["rule"]="RL"
        rows.append(s)
    return pd.concat(rows,ignore_index=True).sort_values("time").reset_index(drop=True) if rows else pd.DataFrame()

rl_train=gen(train)
print(f"  {len(rl_train):,} entries, {_time.time()-t0:.0f}s", flush=True)

print("[3/4] Training confirm + exit + meta...", flush=True); t0=_time.time()

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

c_mdls,c_thrs=train_conf(rl_train,V6_FEATS,"confirm"); tc=confirm(rl_train,c_mdls,c_thrs,V6_FEATS)
print(f"  {len(tc):,} confirmed", flush=True)

# Exit model (numpy-optimized)
ctx_np=swing[EXIT_FEATS[3:]].fillna(0).values.astype(np.float64); rows_e=[]
for _,s in tc.iterrows():
    tm=s["time"]
    if tm not in time_to_idx: continue
    ei=time_to_idx[tm]; d=int(s["direction"]); ep=C[ei]; ea=atr[ei]
    if not np.isfinite(ea) or ea<=0: continue
    end=min(ei+MAX_HOLD+1,len(C))
    if end-ei<10: continue
    pnls=np.array([d*(C[k]-ep)/ea for k in range(ei+1,end)])
    if len(pnls)<5: continue
    for b in range(len(pnls)):
        bi=ei+1+b; cp=pnls[b]
        if cp<-SL_HARD: break
        rem=pnls[b+1:] if b+1<len(pnls) else np.array([cp])
        br=rem.max() if len(rem)>0 else cp
        if br<cp-0.3: lbl=1
        elif br>cp+0.3: lbl=0
        else: continue
        v=cp-pnls[b-3] if b>=3 else (cp-pnls[0] if b>=1 else 0.0)
        row={"unrealized_pnl_R":cp,"bars_held":float(b+1),"pnl_velocity":v,"label":lbl}
        if bi<len(swing):
            for j,f_ in enumerate(EXIT_FEATS[3:]): row[f_]=float(ctx_np[bi,j])
        else:
            for f_ in EXIT_FEATS[3:]: row[f_]=0.0
        rows_e.append(row)
ed=pd.DataFrame(rows_e)
exit_mdl=XGBClassifier(n_estimators=300,max_depth=5,learning_rate=0.05,
                      subsample=0.8,colsample_bytree=0.8,eval_metric="logloss",verbosity=0)
if len(ed)>100: exit_mdl.fit(ed[EXIT_FEATS].fillna(0).values,ed["label"].values)

# Meta (Midas v6 has meta gate now with v83c)
ctx_cols=["hurst_rs","ou_theta","entropy_rate","kramers_up","wavelet_er",
          "quantum_flow","quantum_flow_h4","vwap_dist"]
ctx_arr=swing[ctx_cols].fillna(0).values.astype(np.float64); n=len(C)

def simulate(confirmed,swing,atr,exit_mdl):
    entries=[]
    for _,s in confirmed.iterrows():
        tm=s["time"]
        if tm not in time_to_idx: continue
        ei=time_to_idx[tm]; ea=atr[ei]
        if not np.isfinite(ea) or ea<=0: continue
        entries.append((ei,int(s["direction"]),tm,int(s["cid"]),s.get("rule","RL")))
    N=len(entries)
    if N==0: return pd.DataFrame()
    nf=3+len(ctx_cols); Xs=np.zeros((N*MAX_HOLD,nf),dtype=np.float32)
    valid=np.zeros(N*MAX_HOLD,dtype=bool); cps=np.full((N,MAX_HOLD),np.nan,dtype=np.float64)
    for rank,(ei,d,_,_,_) in enumerate(entries):
        ep=C[ei]; ea2=atr[ei]
        for k in range(1,MAX_HOLD+1):
            bar=ei+k
            if bar>=n: break
            cp=d*(C[bar]-ep)/ea2; cps[rank,k-1]=cp
            if k<MIN_HOLD: continue
            p3=d*(C[bar-3]-ep)/ea2 if k>=3 else cp
            row=rank*MAX_HOLD+(k-1); Xs[row,0]=cp; Xs[row,1]=float(k); Xs[row,2]=cp-p3
            Xs[row,3:]=ctx_arr[bar]; valid[row]=True
    probs=np.zeros(N*MAX_HOLD,dtype=np.float32)
    if exit_mdl is not None and valid.any(): probs[valid]=exit_mdl.predict_proba(Xs[valid])[:,1]
    rows_s=[]
    for rank,(ei,d,tm,cid_v,rule_v) in enumerate(entries):
        ep=C[ei]; xi,xr=None,"max"
        for k in range(1,MAX_HOLD+1):
            bar=ei+k
            if bar>=n: break
            cp=cps[rank,k-1]
            if not np.isfinite(cp): break
            if cp<-SL_HARD: xi,xr=bar,"hard_sl"; break
            if k>=MIN_HOLD and exit_mdl is not None:
                if probs[rank*MAX_HOLD+(k-1)]>=EXIT_THRESHOLD: xi,xr=bar,"ml_exit"; break
        if xi is None: xi=min(ei+MAX_HOLD,n-1); xr="max"
        pnl=d*(C[xi]-ep)/ea
        rows_s.append({"time":tm,"cid":cid_v,"rule":rule_v,"direction":d,"bars":xi-ei,"pnl_R":pnl,"exit":xr})
    return pd.DataFrame(rows_s)

tt=simulate(tc,swing,atr,exit_mdl)
tc["direction"]=tc["direction"].astype(int); tc["cid"]=tc["cid"].astype(int)
md_=tt.merge(tc[["time","cid","rule"]+V6_FEATS],on=["time","cid","rule"],how="left")
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

# Save + Deploy
print("[4/4] Deploying Midas RL bundle...", flush=True)
bundle = {
    "q_entry": q_entry, "mdls": c_mdls, "thrs": c_thrs,
    "exit_mdl": exit_mdl, "meta_mdl": meta_mdl, "meta_threshold": float(best_thr),
    "v72l_feats": V6_FEATS, "meta_feats": META_FEATS, "exit_feats": EXIT_FEATS,
    "min_q": MIN_Q, "rl_rule_name": "RL",
    "trained_on": "v84 RL Entry — Midas XAU (PF 5.25 base, RL PF TBD)",
    "version": "v84-rl",
}

with open(f"{SERVER}/models/midas_xau_validated.pkl", "wb") as f:
    pickle.dump(bundle, f)
with open(f"{OUT}/midas_rl_bundle.pkl", "wb") as f:
    pickle.dump(bundle, f)

print(f"  Deployed: {SERVER}/models/midas_xau_validated.pkl")
print(f"  Entry: {len(q_entry)} Q-models")
print(f"  Confirm: {len(c_mdls)}")
print(f"Done.")
