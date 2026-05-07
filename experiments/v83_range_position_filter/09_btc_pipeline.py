"""
v83c BTC — Full Pipeline: 4h Regime + Range Filter + Kill-Switch
=================================================================
1. Retrain regime with 4h step
2. Relabel setups with new regime
3. Train confirm + exit + meta
4. Holdout validation
"""
from __future__ import annotations
import glob, os, json, pickle, time as _time
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
import sys

PROJECT = "/home/jay/Desktop/new-model-zigzag"
DATA = f"{PROJECT}/data"
OUT = f"{PROJECT}/experiments/v83_range_position_filter"
MODELS_DIR = f"{OUT}/models_btc"
os.makedirs(MODELS_DIR, exist_ok=True)

sys.path.insert(0, f"{PROJECT}/products/_shared")
from importlib.machinery import SourceFileLoader
qf = SourceFileLoader("qf01", f"{PROJECT}/products/_shared/quantum_flow.py").load_module()

GLOBAL_CUTOFF = pd.Timestamp("2024-12-12 00:00:00")
WINDOW = 288; STEP = 48; K = 5; MIN_DATE = "2018-01-01"

V72L_FEATS = ["hurst_rs","ou_theta","entropy_rate","kramers_up","wavelet_er",
    "vwap_dist","hour_enc","dow_enc","quantum_flow","quantum_flow_h4",
    "quantum_momentum","quantum_vwap_conf","quantum_divergence","quantum_div_strength",
    "vpin","sig_quad_var","har_rv_ratio","hawkes_eta"]
META_FEATS = V72L_FEATS + ["direction","cid"]
EXIT_FEATS = ["unrealized_pnl_R","bars_held","pnl_velocity",
    "hurst_rs","ou_theta","entropy_rate","kramers_up","wavelet_er",
    "quantum_flow","quantum_flow_h4","vwap_dist"]
MAX_HOLD=60; MIN_HOLD=2; SL_HARD=4.0; EXIT_THRESHOLD=0.55; MAX_FWD_EXIT=60

feat_names = ["weekly_return_pct","volatility_pct","trend_consistency",
              "trend_strength","volatility","range_vs_atr","return_autocorr","flow_4h_mean"]

def compute_fingerprint(c, h, l, o, flow):
    n=len(c)
    if n<10: return None
    returns=np.diff(c)/c[:-1]; bar_ranges=(h-l)/c
    fp={"weekly_return_pct":float(returns.sum()),
        "volatility_pct":float(returns.std())}
    mean_ret=returns.mean()
    fp["trend_consistency"]=float(np.mean(np.sign(returns)==np.sign(mean_ret))) if abs(mean_ret)>1e-12 else 0.5
    fp["trend_strength"]=float(returns.sum()/(returns.std()+1e-9))
    fp["volatility"]=float(bar_ranges.mean())
    fp["range_vs_atr"]=float(((h.max()-l.min())/c.mean())/(bar_ranges.mean()+1e-9))
    if len(returns)>2:
        r1,r2=returns[:-1],returns[1:]
        d=r1.std()*r2.std()
        fp["return_autocorr"]=float(np.corrcoef(r1,r2)[0,1]) if d>1e-12 else 0.0
    else: fp["return_autocorr"]=0.0
    fc=flow[~np.isnan(flow)]
    fp["flow_4h_mean"]=float(fc.mean()) if len(fc) else 0.0
    return fp

print("="*60)
print("  v83c BTC — FULL PIPELINE")
print("="*60)

# ═══ STEP 1: Retrain regime (4h step) ═══
print("\n[1/6] Retraining BTC regime (4h step)...", flush=True); t0=_time.time()
df = pd.read_csv(f"{DATA}/swing_v5_btc.csv", parse_dates=["time"])
df = df.sort_values("time").reset_index(drop=True)
df = df[df["time"]>=MIN_DATE].reset_index(drop=True)
if "tick_volume" in df.columns and "volume" not in df.columns: df=df.rename(columns={"tick_volume":"volume"})
elif "volume" not in df.columns: df["volume"]=df.get("spread",df["close"]*0.001)
print(f"  {len(df):,} bars from {df['time'].iloc[0]} to {df['time'].iloc[-1]}", flush=True)

flow_4h = qf.quantum_flow_mtf(df[["time","open","high","low","close","volume"]])
closes=df["close"].values.astype(np.float64); highs=df["high"].values.astype(np.float64)
lows=df["low"].values.astype(np.float64); opens=df["open"].values.astype(np.float64)
flow4h_arr=flow_4h.values.astype(np.float64)

fingerprints=[]
for start in range(0,len(df)-WINDOW,STEP):
    end=start+WINDOW
    fp=compute_fingerprint(closes[start:end],highs[start:end],lows[start:end],opens[start:end],flow4h_arr[start:end])
    if fp is not None:
        fp["start_idx"]=start; fp["end_idx"]=end
        fp["center_time"]=str(df["time"].iloc[(start+end)//2])
        fingerprints.append(fp)

fp_df=pd.DataFrame(fingerprints); X_raw=fp_df[feat_names].values
scaler=StandardScaler(); X_scaled=scaler.fit_transform(X_raw)
mask=np.all(np.abs(X_scaled)<4,axis=1); fp_df=fp_df[mask].reset_index(drop=True)
X_raw=fp_df[feat_names].values
scaler=StandardScaler(); X_scaled=scaler.fit_transform(X_raw)
pca=PCA(n_components=len(feat_names)); X_pca=pca.fit_transform(X_scaled)
kmeans=KMeans(n_clusters=K,n_init=20,random_state=42); raw_labels=kmeans.fit_predict(X_pca)

# Auto-label clusters (same as Oracle)
cs={}
for cid in range(K):
    cmask=raw_labels==cid
    cs[cid]={"ret":fp_df.loc[cmask,"weekly_return_pct"].mean(),"vol":fp_df.loc[cmask,"volatility_pct"].mean(),
              "autocorr":fp_df.loc[cmask,"return_autocorr"].mean(),"flow":fp_df.loc[cmask,"flow_4h_mean"].mean(),"n":int(cmask.sum())}
sr=sorted(cs.items(),key=lambda x:x[1]["ret"])
down_raw=sr[0][0]; up_raw=sr[-1][0]
rem=[c for c in range(K) if c not in(down_raw,up_raw)]
rem_v=sorted(rem,key=lambda c:cs[c]["vol"],reverse=True); hv_raw=rem_v[0]
rem2=[c for c in rem if c!=hv_raw]
rem_a=sorted(rem2,key=lambda c:cs[c]["autocorr"]); mr_raw=rem_a[0]; tr_raw=[c for c in rem2 if c!=mr_raw][0]
label_map={down_raw:3,up_raw:0,hv_raw:4,mr_raw:1,tr_raw:2}
new_labels=np.array([label_map[r] for r in raw_labels])
new_cents=np.empty_like(kmeans.cluster_centers_)
for raw,new in label_map.items(): new_cents[new]=kmeans.cluster_centers_[raw]

cn={0:"Uptrend",1:"MeanRevert",2:"TrendRange",3:"Downtrend",4:"HighVol"}
for raw,new in label_map.items():
    s=cs[raw]
    print(f"  C{raw}→C{new} {cn[new]:>11s} n={s['n']:>5} ret={s['ret']:+.2%} vol={s['vol']:.3%}", flush=True)

regime_sel = {
    "K":K,"window":WINDOW,"step":STEP,"n_feats":len(feat_names),"feat_names":feat_names,
    "scaler_mean":scaler.mean_.tolist(),"scaler_std":scaler.scale_.tolist(),
    "pca_mean":pca.mean_.tolist(),"pca_components":pca.components_.tolist(),
    "centroids":new_cents.tolist(),"cluster_names":{str(k):v for k,v in cn.items()},
    "relabel":{"mode":"full_directional","threshold":0.01,"up_cid":0,"down_cid":3},
}
sel_path=f"{OUT}/regime_selector_btc_4h.json"
with open(sel_path,"w") as f: json.dump(regime_sel,f,indent=2)
fp_df["new_label"]=new_labels; fp_df.to_csv(f"{OUT}/regime_fingerprints_btc_4h.csv",index=False)
print(f"  Saved: {sel_path} ({len(fp_df)} windows, {_time.time()-t0:.0f}s)", flush=True)

# ═══ STEP 2: Load swing + assign new regime per bar ═══
print("\n[2/6] Assigning new regime to all bars...", flush=True); t0=_time.time()
new_regime_per_bar=np.full(len(df),-1,dtype=int)
for _,row in fp_df.iterrows():
    s=int(row["start_idx"]); e=int(row["end_idx"])
    if 0<=s<e<=len(df): new_regime_per_bar[s:e]=int(row["new_label"])
time_to_idx = {}
for i, t in enumerate(df["time"].values):
    time_to_idx[t] = i
print(f"  {_time.time()-t0:.0f}s", flush=True)

# ═══ STEP 3: Load setups + relabel with new regime ═══
print("\n[3/6] Loading BTC setups + relabeling...", flush=True); t0=_time.time()
all_setups=[]
for f in sorted(glob.glob(f"{DATA}/setups_*_v72l_btc.csv")):
    old_cid=int(os.path.basename(f).split("_")[1])
    sdf=pd.read_csv(f,parse_dates=["time"]); sdf["old_cid"]=old_cid; all_setups.append(sdf)
all_df=pd.concat(all_setups,ignore_index=True).sort_values("time").reset_index(drop=True)
new_cids=[]
for _,row in all_df.iterrows():
    tm=row["time"]
    if tm in time_to_idx:
        idx=time_to_idx[tm]
        new_cids.append(new_regime_per_bar[idx] if 0<=idx<len(new_regime_per_bar) else -1)
    else: new_cids.append(-1)
all_df["cid"]=new_cids; all_df=all_df[all_df["cid"]>=0].reset_index(drop=True)

# Physics for exit context
phys_lookup=all_df[["time"]+V72L_FEATS].drop_duplicates("time",keep="last").set_index("time")
for col in V72L_FEATS:
    df[col]=phys_lookup[col].reindex(df.set_index("time").index,method="ffill").values; df[col]=df[col].fillna(0)

# ATR on BTC
Cb=df["close"].values.astype(np.float64); Hb=df["high"].values.astype(np.float64); Lb=df["low"].values.astype(np.float64)
trb=np.concatenate([[Hb[0]-Lb[0]],np.maximum.reduce([Hb[1:]-Lb[1:],np.abs(Hb[1:]-Cb[:-1]),np.abs(Lb[1:]-Cb[:-1])])])
atr_b=pd.Series(trb).rolling(14,min_periods=14).mean().values

train_raw=all_df[all_df["time"]<GLOBAL_CUTOFF].reset_index(drop=True)
test_raw=all_df[all_df["time"]>=GLOBAL_CUTOFF].reset_index(drop=True)
print(f"  train={len(train_raw):,} test={len(test_raw):,} ({_time.time()-t0:.0f}s)", flush=True)

# ═══ STEP 4: Train + validate ═══
print("\n[4/6] Training...", flush=True); t0=_time.time()

def train_conf(train,features,tag):
    mdls,thrs={},{}
    for (cid,rule),grp in train.groupby(["cid","rule"]):
        if len(grp)<100: continue
        grp=grp.sort_values("time").reset_index(drop=True)
        s=int(len(grp)*0.80); trn,vd=grp.iloc[:s],grp.iloc[s:]
        if len(vd)<20: continue
        mdl=XGBClassifier(n_estimators=200,max_depth=3,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,eval_metric="logloss",verbosity=0)
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
        X=grp[features].fillna(0).values
        p=mdls[(cid,rule)].predict_proba(X)[:,1]
        rows.append(grp[p>=thrs[(cid,rule)]].copy())
    return pd.concat(rows,ignore_index=True).sort_values("time").reset_index(drop=True) if rows else pd.DataFrame()

mdls,thrs=train_conf(train_raw,V72L_FEATS,"btc-conf")
tc=confirm(train_raw,mdls,thrs,V72L_FEATS)
print(f"  {len(tc):,} confirmed", flush=True)

# Exit model (numpy-optimized)
ctx_np=df[EXIT_FEATS[3:]].fillna(0).values.astype(np.float64)
rows_e=[]
for _,s in tc.iterrows():
    tm=s["time"]
    if tm not in time_to_idx: continue
    ei=time_to_idx[tm]; d=int(s["direction"]); ep=Cb[ei]; ea=atr_b[ei]
    if not np.isfinite(ea) or ea<=0: continue
    end=min(ei+MAX_FWD_EXIT+1,len(Cb))
    if end-ei<10: continue
    pnls=np.array([d*(Cb[k]-ep)/ea for k in range(ei+1,end)])
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
        if bi<len(df):
            for j,f_ in enumerate(EXIT_FEATS[3:]): row[f_]=float(ctx_np[bi,j])
        else:
            for f_ in EXIT_FEATS[3:]: row[f_]=0.0
        rows_e.append(row)
ed=pd.DataFrame(rows_e)
exit_mdl=XGBClassifier(n_estimators=300,max_depth=5,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,eval_metric="logloss",verbosity=0)
exit_mdl.fit(ed[EXIT_FEATS].fillna(0).values,ed["label"].values)
print(f"  Exit: {len(ed):,} rows", flush=True)

# Simulate + Meta
ctx_cols=EXIT_FEATS[3:]; ctx_arr=df[ctx_cols].fillna(0).values.astype(np.float64); n2=len(Cb)
def simulate(confirmed,swing,atr,exit_mdl):
    entries=[]
    for _,s in confirmed.iterrows():
        tm=s["time"]
        if tm not in time_to_idx: continue
        ei=time_to_idx[tm]; ea=atr[ei]
        if not np.isfinite(ea) or ea<=0: continue
        entries.append((ei,int(s["direction"]),tm,int(s["cid"]),s["rule"]))
    N=len(entries)
    if N==0: return pd.DataFrame()
    nf=3+len(ctx_cols); Xs=np.zeros((N*MAX_HOLD,nf),dtype=np.float32)
    valid=np.zeros(N*MAX_HOLD,dtype=bool); cps=np.full((N,MAX_HOLD),np.nan,dtype=np.float64)
    for rank,(ei,d,_,_,_) in enumerate(entries):
        ep=Cb[ei]; ea2=atr[ei]
        for k in range(1,MAX_HOLD+1):
            bar=ei+k
            if bar>=n2: break
            cp=d*(Cb[bar]-ep)/ea2; cps[rank,k-1]=cp
            if k<MIN_HOLD: continue
            p3=d*(Cb[bar-3]-ep)/ea2 if k>=3 else cp
            row=rank*MAX_HOLD+(k-1); Xs[row,0]=cp; Xs[row,1]=float(k); Xs[row,2]=cp-p3
            Xs[row,3:]=ctx_arr[bar]; valid[row]=True
    probs=np.zeros(N*MAX_HOLD,dtype=np.float32)
    if exit_mdl is not None and valid.any(): probs[valid]=exit_mdl.predict_proba(Xs[valid])[:,1]
    rows_s=[]
    for rank,(ei,d,tm,cid_v,rule_v) in enumerate(entries):
        ep=Cb[ei]; xi,xr=None,"max"
        for k in range(1,MAX_HOLD+1):
            bar=ei+k
            if bar>=n2: break
            cp=cps[rank,k-1]
            if not np.isfinite(cp): break
            if cp<-SL_HARD: xi,xr=bar,"hard_sl"; break
            if k>=MIN_HOLD and exit_mdl is not None:
                if probs[rank*MAX_HOLD+(k-1)]>=EXIT_THRESHOLD: xi,xr=bar,"ml_exit"; break
        if xi is None: xi=min(ei+MAX_HOLD,n2-1); xr="max"
        pnl=d*(Cb[xi]-ep)/atr[ei]
        rows_s.append({"time":tm,"cid":cid_v,"rule":rule_v,"direction":d,"bars":xi-ei,"pnl_R":pnl,"exit":xr})
    return pd.DataFrame(rows_s)

tt=simulate(tc,df,atr_b,exit_mdl)
tc["direction"]=tc["direction"].astype(int); tc["cid"]=tc["cid"].astype(int)
md_=tt.merge(tc[["time","cid","rule"]+V72L_FEATS],on=["time","cid","rule"],how="left")
md_["meta_label"]=(md_["pnl_R"]>0).astype(int); md_=md_.sort_values("time").reset_index(drop=True)
s_=int(len(md_)*0.80); mtr=md_.iloc[:s_]; mvd=md_.iloc[s_:]
meta_mdl=XGBClassifier(n_estimators=300,max_depth=4,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,eval_metric="logloss",verbosity=0)
meta_mdl.fit(mtr[META_FEATS].fillna(0).values,mtr["meta_label"].values)
pv=meta_mdl.predict_proba(mvd[META_FEATS].fillna(0).values)[:,1]
pn=mvd["pnl_R"].values; cands=[]
for thr in np.arange(0.40,0.80,0.025):
    mx=pv>=thr
    if mx.sum()<20: continue
    pt=pn[mx]; pf_=pt[pt>0].sum()/max(-pt[pt<=0].sum(),1e-9)
    cands.append((thr,mx.sum(),pf_))
vc=[c for c in cands if c[2]>=pn[pn>0].sum()/max(-pn[pn<=0].sum(),1e-9)*0.95] or cands
vc.sort(key=lambda c:(-c[2],c[1])); best_thr=vc[0][0]
print(f"  Meta threshold: {best_thr:.3f}, {_time.time()-t0:.0f}s", flush=True)

# ═══ STEP 5: Save models ═══
print("\n[5/6] Saving models...", flush=True); t0=_time.time()
for (cid,rule),mdl in mdls.items():
    with open(f"{MODELS_DIR}/confirm_btc_c{cid}_{rule}.pkl","wb") as f: pickle.dump(mdl,f)
with open(f"{MODELS_DIR}/exit_btc_v83c.pkl","wb") as f: pickle.dump(exit_mdl,f)
with open(f"{MODELS_DIR}/meta_btc_v83c.pkl","wb") as f: pickle.dump(meta_mdl,f)
print(f"  {len(mdls)} confirm + exit + meta saved, {_time.time()-t0:.0f}s", flush=True)

# ═══ STEP 6: Holdout ═══
print("\n[6/6] Holdout...", flush=True); t0=_time.time()
tec=confirm(test_raw,mdls,thrs,V72L_FEATS)
tec["direction"]=tec["direction"].astype(int); tec["cid"]=tec["cid"].astype(int)
pm_=meta_mdl.predict_proba(tec[META_FEATS].fillna(0).values)[:,1]
tec_m=tec[pm_>=best_thr].copy()
trades_out=simulate(tec_m,df,atr_b,exit_mdl)

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
    if len(df)==0: print(f"  {tag}: NO TRADES"); return
    w=df[df["pnl_R"]>0]; l=df[df["pnl_R"]<=0]
    pf=w["pnl_R"].sum()/max(-l["pnl_R"].sum(),1e-9)
    wr=len(w)/len(df); total=df["pnl_R"].sum()
    print(f"\n  {tag}: n={len(df)} WR={wr:.1%} PF={pf:.2f} Total={total:+.1f}R")
    names={0:"Uptrend",1:"MeanRevert",2:"TrendRange",3:"Downtrend",4:"HighVol"}
    for cv in sorted(df["cid"].unique()):
        g=df[df["cid"]==cv]; ww=g[g["pnl_R"]>0]; ll=g[g["pnl_R"]<=0]
        ppf=ww["pnl_R"].sum()/max(-ll["pnl_R"].sum(),1e-9)
        print(f"    C{cv} {names[cv]}: n={len(g):,} WR={len(ww)/len(g):.1%} PF={ppf:.2f} R={g['pnl_R'].sum():+.1f}")

orig=pd.read_csv(f"{DATA}/v72l_trades_holdout_btc.csv",parse_dates=["time"])
w=orig[orig["pnl_R"]>0]; l=orig[orig["pnl_R"]<=0]
print(f"\n  ORIGINAL: n={len(orig):,} WR={len(w)/len(orig):.1%} PF={w['pnl_R'].sum()/max(-l['pnl_R'].sum(),1e-9):.2f} Total={orig['pnl_R'].sum():+.1f}R")
report(trades_final,"BTC v83c FINAL")

trades_final.to_csv(f"{OUT}/btc_v83c_final_trades.csv",index=False)
print(f"\nDone. Total: {_time.time()-t0:.0f}s")
