"""v88: Sweep v84 RL entry Q-threshold on the unseen 30% window.

If tightening Q (currently 0.3) to e.g. 0.5 lifts PF without losing too
many trades, that's a free win — no exit-head needed.

For each Q threshold:
  1. Re-generate trades from setups using q_entry[cid].predict(feats) >= Q
  2. Apply the same hard-SL + 60-bar-max simulation
  3. Report PF, Total R, WR%, N

Train/test split: same chrono 70/30 by setup time, so the sweep happens
on the unseen window only.
"""
import os, time, pickle, glob as _glob
import numpy as np, pandas as pd

PROJECT="/home/jay/Desktop/new-model-zigzag"
DATA=f"{PROJECT}/data"

V72L=['hurst_rs','ou_theta','entropy_rate','kramers_up','wavelet_er',
      'vwap_dist','hour_enc','dow_enc','quantum_flow','quantum_flow_h4',
      'quantum_momentum','quantum_vwap_conf','quantum_divergence','quantum_div_strength',
      'vpin','sig_quad_var','har_rv_ratio','hawkes_eta']
MAX_HOLD=60; SL_HARD=-4.0

def load_market(swing_csv):
    sw=pd.read_csv(swing_csv,parse_dates=["time"])
    sw=sw.sort_values("time").drop_duplicates('time',keep='last').reset_index(drop=True)
    n=len(sw); C=sw["close"].values.astype(np.float64)
    H=sw["high"].values; Lo=sw["low"].values
    tr=np.concatenate([[H[0]-Lo[0]],np.maximum.reduce([H[1:]-Lo[1:],np.abs(H[1:]-C[:-1]),np.abs(Lo[1:]-C[:-1])])])
    atr=pd.Series(tr).rolling(14,min_periods=14).mean().values
    t2i={pd.Timestamp(t):i for i,t in enumerate(sw["time"].values)}
    return n,C,Lo,H,atr,t2i

def simulate(setups,t2i,n,C,atr):
    out=[]
    for _,row in setups.iterrows():
        tm=row["time"]
        if tm not in t2i: continue
        ei=t2i[tm]; d=int(row["direction"]); ep=C[ei]; ea=atr[ei]
        if not np.isfinite(ea) or ea<=0: continue
        bar=min(ei+MAX_HOLD,n-1)
        for k in range(1,MAX_HOLD+1):
            b=ei+k
            if b>=n: break
            if d*(C[b]-ep)/ea<=SL_HARD: bar=b; break
        out.append(d*(C[bar]-ep)/ea)
    return out

def pf_stats(p):
    s=sum(p); w=[x for x in p if x>0]; l=[x for x in p if x<=0]
    pf=sum(w)/max(-sum(l),1e-9) if l else 99.0
    return pf,s,len(w)/max(len(p),1),len(p)

def evaluate(name,swing_csv,setups_glob,bundle_path,split_quantile=0.70):
    print("\n"+"="*72); print(f"  {name}"); print("="*72)
    t0=time.time()
    n,C,Lo,H,atr,t2i=load_market(swing_csv)
    setups=[]
    for f in sorted(_glob.glob(setups_glob)):
        cid=int(os.path.basename(f).split('_')[1])
        df=pd.read_csv(f,parse_dates=["time"]); df['old_cid']=cid
        setups.append(df)
    all_df=pd.concat(setups,ignore_index=True).sort_values("time").reset_index(drop=True)

    with open(bundle_path,"rb") as f: bundle=pickle.load(f)
    q_entry=bundle['q_entry']

    # Score every setup
    scores=np.full(len(all_df),-9.0)
    for cid,m in q_entry.items():
        mask=(all_df['old_cid']==cid).values
        if mask.sum()<1: continue
        Xc=all_df.loc[mask,V72L].fillna(0).values
        scores[mask]=m.predict(Xc)
    all_df['q_score']=scores

    # v5 holdout cutoff is canonical (q_entry models trained on pre-2024-12-12)
    HOLDOUT=pd.Timestamp("2024-12-12")
    test=all_df[all_df["time"]>=HOLDOUT].copy().reset_index(drop=True)
    print(f"  Test setups (post-2024-12-12 holdout): {len(test):,}")

    print(f"\n  {'Q_thr':>6s}  {'N':>5s}  {'PF':>5s}  {'TotalR':>8s}  {'WR%':>5s}  R/trade")
    print(f"  {'-'*6}  {'-'*5}  {'-'*5}  {'-'*8}  {'-'*5}  {'-'*7}")
    base=None
    for q_thr in [0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.70]:
        sel=test[test['q_score']>=q_thr]
        if len(sel)<10:
            print(f"  {q_thr:6.2f}  {len(sel):5d}  ---   too few trades"); continue
        pn=simulate(sel,t2i,n,C,atr)
        pf,s,wr,N=pf_stats(pn)
        rpt=s/max(N,1)
        marker=""
        if base is None and abs(q_thr-0.30)<1e-6: base=(s,N); marker=" ← current"
        if base is not None and s>base[0]: marker+=" ★"
        print(f"  {q_thr:6.2f}  {N:5d}  {pf:5.2f}  {s:+8.0f}  {wr*100:5.1f}  {rpt:+6.2f}{marker}")
    print(f"  Done in {time.time()-t0:.0f}s")

if __name__=="__main__":
    evaluate("Oracle XAU — Entry Q-threshold sweep on unseen 30%",
             f"{DATA}/swing_v5_xauusd.csv",
             f"{DATA}/setups_*_v72l.csv",
             f"{PROJECT}/products/models/oracle_xau_validated.pkl")
    evaluate("Oracle BTC — Entry Q-threshold sweep on unseen 30%",
             f"{DATA}/swing_v5_btc.csv",
             f"{DATA}/setups_*_v72l_btc.csv",
             f"{PROJECT}/products/models/oracle_btc_validated.pkl")
