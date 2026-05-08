"""v87: Wide threshold sweep on unseen (last-30%) RL trades.

Optimization:
  - Build all bar-level feature rows once for the test trades.
  - Run each of the 4 heads with a single batched predict_proba.
  - Sweep thresholds by indexing the cached probabilities — cheap.

Reports the top combos for XAU and BTC. If none beat baseline, multi-head
is genuinely net-negative on unseen data.
"""
import os, time, pickle, glob as _glob, itertools
import numpy as np, pandas as pd

PROJECT="/home/jay/Desktop/new-model-zigzag"
DATA=f"{PROJECT}/data"
EXP=f"{PROJECT}/experiments/v87_multi_head_exit"

V72L=['hurst_rs','ou_theta','entropy_rate','kramers_up','wavelet_er',
      'vwap_dist','hour_enc','dow_enc','quantum_flow','quantum_flow_h4',
      'quantum_momentum','quantum_vwap_conf','quantum_divergence','quantum_div_strength',
      'vpin','sig_quad_var','har_rv_ratio','hawkes_eta']
ENRICHED=['current_R','max_R_seen','drawdown_from_peak',
          'bars_in_trade','bars_remaining','dist_to_SL','dist_to_TP',
          'vol_10bar','mom_3bar','mom_10bar']+V72L
MAX_HOLD=60; MIN_HOLD=2; SL_HARD=-4.0

def load_market(swing_csv,setups_glob):
    swing=pd.read_csv(swing_csv,parse_dates=["time"])
    swing=swing.sort_values("time").drop_duplicates('time',keep='last').reset_index(drop=True)
    n=len(swing)
    C=swing["close"].values.astype(np.float64); H=swing["high"].values; Lo=swing["low"].values
    tr=np.concatenate([[H[0]-Lo[0]],np.maximum.reduce([H[1:]-Lo[1:],np.abs(H[1:]-C[:-1]),np.abs(Lo[1:]-C[:-1])])])
    atr=pd.Series(tr).rolling(14,min_periods=14).mean().values
    t2i={pd.Timestamp(t):i for i,t in enumerate(swing["time"].values)}
    setups=[pd.read_csv(f,parse_dates=["time"]) for f in sorted(_glob.glob(setups_glob))]
    all_df=pd.concat(setups,ignore_index=True)
    phys=all_df[['time']+V72L].drop_duplicates('time',keep='last').sort_values('time')
    swing=pd.merge_asof(swing.sort_values('time'),phys,on='time',direction='nearest')
    for c in V72L: swing[c]=swing[c].fillna(0)
    ctx=swing[V72L].fillna(0).values.astype(np.float64)
    return n,C,Lo,H,atr,t2i,ctx

def build_test_arrays(test_trades,t2i,n,C,atr,ctx):
    """Per trade: list of (k, mtm) plus bar-level feature rows.
       Returns:
         tr_meta: list of dict {ei,d,ep,ea,end_k_sl,max_hold_k}
         all_X:   (Nrows, 28) feature matrix
         row_to_trade: (Nrows,) trade index
         row_k:        (Nrows,) bar offset k
         row_mtm:      (Nrows,) running MTM at bar k
       SL bars are NOT in all_X (we exit immediately on SL).
    """
    tr_meta=[]; rows=[]; r2t=[]; rk=[]; rmtm=[]
    for ti,(_,trade) in enumerate(test_trades.iterrows()):
        tm=trade["time"]
        if tm not in t2i:
            tr_meta.append(None); continue
        ei=t2i[tm]; d=int(trade["direction"]); ep=C[ei]; ea=atr[ei]
        if not np.isfinite(ea) or ea<=0:
            tr_meta.append(None); continue
        # walk forward: track peak, detect SL, collect feature rows for k>=MIN_HOLD
        peak=0.0; sl_k=None; max_k=min(MAX_HOLD,n-ei-1)
        for k in range(1,max_k+1):
            bar=ei+k
            mtm=d*(C[bar]-ep)/ea
            if mtm<=SL_HARD: sl_k=k; break
            if mtm>peak: peak=mtm
            if k<MIN_HOLD: continue
            dist_sl=mtm-SL_HARD; dist_tp=2.0-mtm
            vol10=np.std([d*(C[max(0,bar-j)]-C[max(0,bar-j-1)])/ea for j in range(min(10,bar))]) if bar>5 else 0
            mom3=d*(C[bar]-C[max(0,bar-3)])/ea if bar>=3 else 0
            mom10=d*(C[bar]-C[max(0,bar-10)])/ea if bar>=10 else 0
            row=[mtm,peak,peak-mtm,float(k),float(MAX_HOLD-k),
                 dist_sl,dist_tp,vol10,mom3,mom10]+[float(ctx[bar,j]) for j in range(len(V72L))]
            rows.append(row); r2t.append(ti); rk.append(k); rmtm.append(mtm)
        tr_meta.append({'ei':ei,'d':d,'ep':ep,'ea':ea,'sl_k':sl_k,'max_k':max_k})
    return tr_meta, np.asarray(rows,dtype=np.float32), np.asarray(r2t,dtype=np.int32), np.asarray(rk,dtype=np.int32), np.asarray(rmtm,dtype=np.float32)

def baseline_pnls(tr_meta,n,C):
    out=[]
    for m in tr_meta:
        if m is None: continue
        ei=m['ei']; d=m['d']; ep=m['ep']; ea=m['ea']
        if m['sl_k'] is not None:
            bar=ei+m['sl_k']
        else:
            bar=min(ei+MAX_HOLD,n-1)
        out.append(d*(C[bar]-ep)/ea)
    return out

def simulate_thresholds(tr_meta,r2t,rk,rmtm,p_up,p_gb,p_sl,p_nh,
                        th_up,th_gb,th_sl,th_nh,n,C):
    """Apply policy. Find earliest exit signal per trade among cached rows."""
    sig_mask=((p_gb>th_gb)&(p_up<th_up))|(p_sl>th_sl)
    overrider=(p_nh>th_nh)&(p_gb<th_gb)
    sig_mask=sig_mask & ~overrider
    # Per trade: earliest k where sig_mask is True
    out=[]
    # Build per-trade list of (idx,k) — rows already sorted by trade then k
    n_tr=len(tr_meta)
    # sweep
    cur_ti=-1; chosen_k=-1
    chosen=np.full(n_tr,-1,dtype=np.int32)
    for i in range(len(r2t)):
        ti=int(r2t[i])
        if ti!=cur_ti:
            cur_ti=ti; chosen_k=-1
        if chosen[ti]!=-1: continue
        if sig_mask[i]:
            chosen[ti]=int(rk[i])
    # Compute pnls
    for ti,m in enumerate(tr_meta):
        if m is None: continue
        ei=m['ei']; d=m['d']; ep=m['ep']; ea=m['ea']
        sl_k=m['sl_k']; ck=int(chosen[ti])
        # SL always wins if it would happen first
        if ck==-1:
            bar=ei+sl_k if sl_k is not None else min(ei+MAX_HOLD,n-1)
        else:
            if sl_k is not None and sl_k<ck:
                bar=ei+sl_k
            else:
                bar=ei+ck
        out.append(d*(C[bar]-ep)/ea)
    return out

def pf_stats(pnls):
    s=sum(pnls); w=[p for p in pnls if p>0]; l=[p for p in pnls if p<=0]
    pf=sum(w)/max(-sum(l),1e-9) if l else 99.0
    wr=len(w)/max(len(pnls),1)
    return pf,s,wr,len(pnls)

def evaluate(name,swing_csv,setups_glob,trades_csv,bundle_path):
    print("\n"+"="*70); print(f"  {name}"); print("="*70)
    t0=time.time()
    n,C,Lo,H,atr,t2i,ctx=load_market(swing_csv,setups_glob)
    trades=pd.read_csv(trades_csv,parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    split=trades["time"].quantile(0.70)
    test=trades[trades["time"]>=split].reset_index(drop=True)
    print(f"  Unseen test (>= {split}): {len(test)} trades")
    with open(bundle_path,"rb") as f: bundle=pickle.load(f)
    m_up=bundle['models']['label_up']
    m_gb=bundle['models']['label_gb']
    m_sl=bundle['models'].get('label_sl')
    m_nh=bundle['models'].get('label_nh')

    print("  Building cached features...", end='', flush=True); t1=time.time()
    tr_meta,X,r2t,rk,rmtm=build_test_arrays(test,t2i,n,C,atr,ctx)
    print(f" {time.time()-t1:.0f}s — rows={len(X):,}", flush=True)

    print("  Batch predicting heads...", end='', flush=True); t1=time.time()
    p_up=m_up.predict_proba(X)[:,1]
    p_gb=m_gb.predict_proba(X)[:,1]
    p_sl=m_sl.predict_proba(X)[:,1] if m_sl is not None else np.zeros(len(X))
    p_nh=m_nh.predict_proba(X)[:,1] if m_nh is not None else np.zeros(len(X))
    print(f" {time.time()-t1:.0f}s", flush=True)

    base=baseline_pnls(tr_meta,n,C)
    pf_b,s_b,wr_b,N_b=pf_stats(base)
    print(f"\n  Baseline: PF={pf_b:.2f}  Total={s_b:+.0f}R  WR={wr_b*100:.1f}%  N={N_b}")

    # Wide grid
    grid_up=[0.20,0.30,0.40,0.50,0.60,0.70,0.80]
    grid_gb=[0.30,0.40,0.50,0.60,0.70,0.80,0.90]
    grid_sl=[0.50,0.60,0.70,0.80,0.90,0.99]   # 0.99 = effectively off
    grid_nh=[0.50,0.60,0.70,0.80,0.90]
    print(f"  Sweeping {len(grid_up)*len(grid_gb)*len(grid_sl)*len(grid_nh)} combos...", end='', flush=True)
    t1=time.time()

    results=[]
    for th_up,th_gb,th_sl,th_nh in itertools.product(grid_up,grid_gb,grid_sl,grid_nh):
        pn=simulate_thresholds(tr_meta,r2t,rk,rmtm,p_up,p_gb,p_sl,p_nh,
                               th_up,th_gb,th_sl,th_nh,n,C)
        pf,s,wr,N=pf_stats(pn)
        results.append((s,pf,wr,N,th_up,th_gb,th_sl,th_nh))
    print(f" {time.time()-t1:.0f}s", flush=True)

    # Sort by total R desc
    results.sort(key=lambda r:r[0], reverse=True)
    print(f"\n  TOP 10 by Total R:")
    print(f"  {'PF':>5s} {'TotalR':>8s} {'WR%':>5s} {'N':>4s}  th_up th_gb th_sl th_nh   vs base")
    for s,pf,wr,N,tu,tg,tsl,tnh in results[:10]:
        print(f"  {pf:5.2f} {s:+8.0f} {wr*100:5.1f} {N:4d}   {tu:.2f}  {tg:.2f}  {tsl:.2f}  {tnh:.2f}   {s-s_b:+.0f}R")

    # Best PF (with min N, total>0)
    fr=[r for r in results if r[3]>=N_b*0.8 and r[0]>0]
    fr.sort(key=lambda r:r[1],reverse=True)
    if fr:
        print(f"\n  TOP 10 by PF (Total>0, N>=80% of baseline):")
        for s,pf,wr,N,tu,tg,tsl,tnh in fr[:10]:
            print(f"  {pf:5.2f} {s:+8.0f} {wr*100:5.1f} {N:4d}   {tu:.2f}  {tg:.2f}  {tsl:.2f}  {tnh:.2f}   {s-s_b:+.0f}R")

    # How many beat baseline at all?
    beats=[r for r in results if r[0]>s_b]
    print(f"\n  Combos beating baseline TotalR ({s_b:+.0f}): {len(beats)}/{len(results)}")
    print(f"  Done in {time.time()-t0:.0f}s")
    return s_b, results

if __name__=="__main__":
    evaluate("Oracle XAU — wide threshold sweep on unseen 30%",
             f"{DATA}/swing_v5_xauusd.csv",
             f"{DATA}/setups_*_v72l.csv",
             f"{PROJECT}/experiments/v84_rl_entry/v84_rl_trades.csv",
             f"{EXP}/multi_head_exit_bundle.pkl")
    evaluate("Oracle BTC — wide threshold sweep on unseen 30%",
             f"{DATA}/swing_v5_btc.csv",
             f"{DATA}/setups_*_v72l_btc.csv",
             f"{PROJECT}/experiments/v84_rl_entry/btc_rl_trades.csv",
             f"{EXP}/multi_head_exit_oracle_btc.pkl")
