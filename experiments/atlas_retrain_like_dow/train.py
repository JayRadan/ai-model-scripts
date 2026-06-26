"""Train XAU/BTC with the EXACT Atlas DOW recipe (verbatim from
products/atlas_dji/scripts/03_train_q_production.py). Output to TEMP (does NOT
touch deployed bundles). Prints the built-in OOS holdout Q-sweep so we can see,
apples-to-apples, whether the identical recipe produces an edge on each market."""
import importlib.util, sys, time, pickle
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd
from numba import njit
ROOT = Path("/home/jay/Desktop/new-model-zigzag")
sys.path.insert(0, str(ROOT/"experiments/kalman_color_flip")); sys.path.insert(0, str(ROOT/"products/hermes_xau"))
from kalman import compute_kalman, bars_in_regime_array
from tfk import compute_tfk
_spec=importlib.util.spec_from_file_location("ofm1", ROOT/"products/_shared/m1_with_orderflow.py")
_ofm1=importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_ofm1)
add_standard_features=_ofm1.add_standard_features; FLOW_FEATS=list(_ofm1.FLOW_FEATS)

SPREAD=0.30; SL,TRAIL,MAXH=6.0,2.0,300; STRONG_ATR=0.8; RMULT=1.0; MFE_FILTER=2.0
HOLDOUT_CUTOFF=pd.Timestamp("2025-09-01 00:00:00")
KF_FEATS=["f_velPct","f_velSignif","f_innovZ","f_volState","f_accel","f_velRaw"]
TFK_FEATS=["force","velocity","x_est","regime_w","trend_raw","trend","committed_dir"]
STD=["rsi14","dist_ema20","dist_ema50","dist_ema100","dist_ema200","slope5","slope10","slope20","atr_ratio",
     "m5_rsi14","m5_slope5","m5_ema50_dist","m15_rsi14","m15_slope5","m15_ema50_dist","h1_rsi14","h1_slope5","h1_ema50_dist"]
EXTRA=["dist_kf","dist_tfk","kf_regime_age","vel_up_streak","bar_range_atr","kf_dir","body_atr","strong_bear_prev","strong_bull_prev"]

@njit(cache=True)
def labels_with_mfe(idxs,dirs,O,H,L,C,atr,sp,SL,TRAIL,MAXH,n):
    m=len(idxs); pnl=np.empty(m); xit=np.empty(m,np.int64); mfe=np.empty(m)
    for k in range(m):
        i=idxs[k]; d=dirs[k]; a=atr[i]; ei=i+1
        if ei>=n or not(a>0): pnl[k]=0.0; xit[k]=ei; mfe[k]=0.0; continue
        ep=O[ei]; hard=SL*a; trd=TRAIL*a; mf=0.0; end=min(ei+MAXH,n-1); done=False
        for j in range(ei,end+1):
            fav=d*(C[j]-ep)
            if fav>mf: mf=fav
            if d==1 and (ep-L[j])>=hard: pnl[k]=-SL-sp; xit[k]=j; done=True; break
            if d==-1 and (H[j]-ep)>=hard: pnl[k]=-SL-sp; xit[k]=j; done=True; break
            if mf>=trd and (mf-fav)>=trd: pnl[k]=(mf-trd)/a-sp; xit[k]=j; done=True; break
        if not done: pnl[k]=d*(C[end]-ep)/a-sp; xit[k]=end
        mfe[k]=mf/a
    return pnl,xit,mfe

def streak_up(v):
    n=len(v); o=np.zeros(n,np.int32); c=0
    for i in range(1,n): c=c+1 if v[i]>v[i-1] else 0; o[i]=c
    return o

def run(inst):
    DATA=ROOT/f"data/m1_{inst}_orderflow_8y.parquet"
    m1=pd.read_parquet(DATA).sort_values("time").reset_index(drop=True)
    print(f"\n===== {inst.upper()} =====  {len(m1):,} bars  {m1.time.iloc[0]} -> {m1.time.iloc[-1]}", flush=True)
    tfk_df=compute_tfk(m1); kf=compute_kalman(m1,r_mult=RMULT); df=add_standard_features(kf)
    for c in TFK_FEATS: df[c]=tfk_df[c].to_numpy()
    df["tfk_line"]=tfk_df["tfk_line"].to_numpy()
    O=m1.open.to_numpy(float);H=m1.high.to_numpy(float);L=m1.low.to_numpy(float);C=m1.close.to_numpy(float)
    atr=df["atr14"].to_numpy(float); kdir=df["kf_dir"].to_numpy(np.int64); kline=df["kf_p"].to_numpy(float)
    cdir=tfk_df["committed_dir"].to_numpy(np.int64); tline=tfk_df["tfk_line"].to_numpy(float); vel=df["f_velRaw"].to_numpy(float)
    kage=bars_in_regime_array(kdir); df["kf_regime_age"]=kage
    df["bar_range_atr"]=(H-L)/np.maximum(atr,1e-9)
    df["dist_kf"]=np.where(atr>0,(C-kline)/atr,0.0); df["dist_tfk"]=np.where(atr>0,(C-tline)/atr,0.0)
    df["vel_up_streak"]=streak_up(vel)
    body=C-O; body_atr=np.where(atr>0,np.abs(body)/atr,0.0); df["body_atr"]=body_atr
    strong_bear=(body<0)&(body_atr>=STRONG_ATR); strong_bull=(body>0)&(body_atr>=STRONG_ATR)
    df["strong_bear_prev"]=np.concatenate([[False],strong_bear[:-1]]); df["strong_bull_prev"]=np.concatenate([[False],strong_bull[:-1]])
    n=len(df); sp=SPREAD/np.nanmedian(atr)
    pbk=np.concatenate([[False],C[:-1]<kline[:-1]]); pak=np.concatenate([[False],C[:-1]>kline[:-1]])
    pbt=np.concatenate([[False],C[:-1]<tline[:-1]]); pat=np.concatenate([[False],C[:-1]>tline[:-1]])
    g=C>O; r=C<O; ok=np.isfinite(atr)&(atr>0); ok[:250]=False; ok[-(MAXH+1):]=False
    buy=ok&(cdir==1)&(kdir==-1)&df["strong_bear_prev"].to_numpy()&pbk&pbt&g&(kage>=3)
    sell=ok&(cdir==-1)&(kdir==1)&df["strong_bull_prev"].to_numpy()&pak&pat&r&(kage>=3)
    mask=buy|sell; idxs=np.where(mask)[0]; dirs=np.where(cdir[idxs]==1,1,-1).astype(np.int64)
    pnl,xit,mfe=labels_with_mfe(idxs,dirs,O,H,L,C,atr,sp,SL,TRAIL,MAXH,n)
    mfe_2r=mfe>=MFE_FILTER
    feat_cols=[c for c in dict.fromkeys(EXTRA+KF_FEATS+TFK_FEATS+STD+FLOW_FEATS) if c in df.columns]
    print(f"  candidates: {len(idxs):,}  MFE>=2R train: {mfe_2r.sum():,}  feats: {len(feat_cols)}", flush=True)
    from xgboost import XGBRegressor
    times=m1["time"].to_numpy(); tmask=times<np.datetime64(HOLDOUT_CUTOFF)
    tr_m=tmask[idxs]; te_m=~tr_m; hold_train=tr_m&mfe_2r
    if te_m.sum()<50 or hold_train.sum()<50: print("  too few holdout/train rows"); return
    Mh=XGBRegressor(n_estimators=500,max_depth=5,learning_rate=0.04,subsample=0.85,colsample_bytree=0.85,
                    min_child_weight=8,reg_lambda=1.0,objective="reg:squarederror",tree_method="hist",random_state=42,verbosity=0)
    Mh.fit(df.iloc[idxs[hold_train]][feat_cols].fillna(0).to_numpy(np.float32), pnl[hold_train].astype(np.float32))
    q_te=Mh.predict(df.iloc[idxs[te_m]][feat_cols].fillna(0).to_numpy(np.float32)); test_pnl=pnl[te_m]
    print(f"  ── OOS holdout (post {HOLDOUT_CUTOFF.date()}, n_test={te_m.sum():,}) ──")
    print(f"  {'Q':>5} {'n':>6} {'WR%':>6} {'PF':>6} {'sumR':>9}")
    for q in [1.0,2.0,3.0,4.0]:
        qm=q_te>=q; rs=test_pnl[qm]; rs=rs[np.isfinite(rs)]
        if len(rs)==0: print(f"  {q:>5.1f} {0:>6}"); continue
        w,l=rs[rs>0],rs[rs<=0]; pf=w.sum()/max(-l.sum(),1e-9)
        print(f"  {q:>5.1f} {len(rs):>6,} {(rs>0).mean()*100:>5.1f}% {pf:>6.2f} {rs.sum():>+9.1f}")

for inst in ["xau","btc"]:
    run(inst)
