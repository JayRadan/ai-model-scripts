"""Test the hierarchical-regime hypothesis: are losers concentrated in
counter-trend micro-states at entry time?

Micro-regime features computed AT ENTRY:
  mom_5_signed   = direction × (C[t] - C[t-5]) / atr   (5-bar recent move)
  mom_10_signed  = direction × (C[t] - C[t-10]) / atr  (10-bar)
  mom_20_signed  = direction × (C[t] - C[t-20]) / atr  (20-bar)
  against_5      = count of last 5 M5 bars going against direction
  max_excursion_against_10 = deepest counter-direction excursion in last 10 bars

Diagnostic: bucket all v84 RL trades by mom_signed at entry, then report
per-bucket WR / PF / Total R. If counter-trend micro-state predicts losers,
trades with negative mom_signed should have meaningfully worse stats.

Test on UNSEEN window only (no train/test leakage; we're not training a
model — just measuring conditional outcomes).
"""
import os, glob as _glob
import numpy as np, pandas as pd
from importlib.util import spec_from_file_location, module_from_spec

PROJECT="/home/jay/Desktop/new-model-zigzag"
DATA=f"{PROJECT}/data"
spec=spec_from_file_location("v89", os.path.join(os.path.dirname(os.path.abspath(__file__)),"01_optimal_stopping.py"))
v89=module_from_spec(spec); spec.loader.exec_module(v89)
NAMES={0:"Uptrend",1:"MeanRevert",2:"TrendRange",3:"Downtrend",4:"HighVol"}

def micro_feats_at_entry(t_idx,d,C,atr):
    ea=atr[t_idx]
    if not np.isfinite(ea) or ea<=0: return None
    feats={}
    for L in [5,10,20]:
        if t_idx-L<0: feats[f'mom_{L}']=0.0; continue
        feats[f'mom_{L}']=float(d*(C[t_idx]-C[t_idx-L])/ea)
    # against-bar count over last 5 M5 bars (each bar's signed return vs direction)
    against=0; max_against=0.0; cur_against=0.0
    for j in range(1,11):
        if t_idx-j<0: break
        bar_ret=d*(C[t_idx-j+1]-C[t_idx-j])/ea
        if bar_ret<0:
            cur_against+=abs(bar_ret); against+=(1 if j<=5 else 0)
            if cur_against>max_against: max_against=cur_against
        else:
            cur_against=0.0
    feats['against_5']=float(against)
    feats['max_excursion_against_10']=float(max_against)
    return feats

def evaluate(name,swing_csv,trades_csv):
    print("\n"+"="*72); print(f"  {name}"); print("="*72)
    n,C,atr,t2i,_=v89.load_market(swing_csv,f"{DATA}/setups_*_v72l*.csv" if 'btc' not in swing_csv else f"{DATA}/setups_*_v72l_btc.csv")
    trades=pd.read_csv(trades_csv,parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    split=trades["time"].quantile(0.70)
    tst=trades[trades["time"]>=split].reset_index(drop=True)
    print(f"  Test (unseen): {len(tst)} trades")

    rows=[]
    for _,trade in tst.iterrows():
        tm=trade["time"]
        if tm not in t2i: continue
        ei=t2i[tm]; d=int(trade["direction"])
        f=micro_feats_at_entry(ei,d,C,atr)
        if f is None: continue
        f['cid']=int(trade["cid"]); f['direction']=d; f['pnl_R']=float(trade["pnl_R"])
        rows.append(f)
    df=pd.DataFrame(rows)

    pos_total=(df['pnl_R']>0).sum(); neg_total=(df['pnl_R']<=0).sum()
    print(f"  Overall: WR={pos_total*100/len(df):.1f}%  PF={df[df.pnl_R>0].pnl_R.sum()/max(-df[df.pnl_R<=0].pnl_R.sum(),1e-9):.2f}  Total={df.pnl_R.sum():+.0f}R")

    # ── Bucket by mom_10 ──
    def stats(g):
        if len(g)==0: return (0,0.0,0.0,0.0,0)
        wr=(g.pnl_R>0).mean()*100
        pf_=g[g.pnl_R>0].pnl_R.sum()/max(-g[g.pnl_R<=0].pnl_R.sum(),1e-9)
        total=g.pnl_R.sum()
        sl=(g.pnl_R<=-3.9).sum()
        return len(g),pf_,total,wr,sl

    print(f"\n  ── Bucket by mom_10_signed (10-bar entry-time momentum, signed by direction) ──")
    print(f"  {'bucket':<25s} {'N':>4s} {'PF':>5s} {'TotalR':>7s} {'WR%':>5s} {'SL':>3s} {'mean':>6s}")
    buckets=[
        ('mom_10 < -1.0  (deep counter)',  df[df.mom_10<-1.0]),
        ('mom_10 ∈ [-1, -0.3]  (counter)',  df[(df.mom_10>=-1.0)&(df.mom_10<-0.3)]),
        ('mom_10 ∈ [-0.3, +0.3]  (stall)',  df[(df.mom_10>=-0.3)&(df.mom_10<=0.3)]),
        ('mom_10 ∈ [+0.3, +1.0]  (mild w/)', df[(df.mom_10>0.3)&(df.mom_10<=1.0)]),
        ('mom_10 > +1.0  (strong w/)',      df[df.mom_10>1.0]),
    ]
    for label,g in buckets:
        N,pf_,tot,wr,sl=stats(g)
        mean=g.pnl_R.mean() if len(g) else 0
        marker = " ⚠ poor" if (pf_<2.0 and N>=20) else (" ★ best" if pf_>5.5 else "")
        print(f"  {label:<25s} {N:>4d} {pf_:5.2f} {tot:+7.0f} {wr:5.1f} {sl:3d} {mean:+6.2f}{marker}")

    # By cluster + bucket
    print(f"\n  ── Per-cluster × micro-momentum bucket (most interesting interactions) ──")
    print(f"  {'regime':<11s} {'micro':<22s} {'N':>3s} {'PF':>5s} {'TotalR':>7s} {'WR%':>5s} {'SL':>3s}")
    for cid in sorted(df['cid'].unique()):
        for label,bucket_df in buckets:
            g=df[(df.cid==cid)&bucket_df.index.isin(bucket_df.index) if False else (df.cid==cid)]
            # Re-filter
            if 'deep counter' in label: g=g[g.mom_10<-1.0]
            elif 'counter' in label and 'deep' not in label: g=g[(g.mom_10>=-1.0)&(g.mom_10<-0.3)]
            elif 'stall' in label: g=g[(g.mom_10>=-0.3)&(g.mom_10<=0.3)]
            elif 'mild w/' in label: g=g[(g.mom_10>0.3)&(g.mom_10<=1.0)]
            elif 'strong w/' in label: g=g[g.mom_10>1.0]
            if len(g)<5: continue
            N,pf_,tot,wr,sl=stats(g)
            print(f"  {NAMES.get(cid,'?'):<11s} {label[:22]:<22s} {N:>3d} {pf_:5.2f} {tot:+7.0f} {wr:5.1f} {sl:3d}")

    # ── Bucket by max_excursion_against ──
    print(f"\n  ── Bucket by max_excursion_against_10 (deepest counter pullback in last 10 bars) ──")
    print(f"  {'bucket':<25s} {'N':>4s} {'PF':>5s} {'TotalR':>7s} {'WR%':>5s} {'SL':>3s}")
    me=df['max_excursion_against_10']
    for thr_lo,thr_hi,label in [(0,0.3,'no/tiny pullback'),(0.3,0.6,'small'),(0.6,1.0,'medium'),(1.0,1.5,'large'),(1.5,99,'deep')]:
        g=df[(me>=thr_lo)&(me<thr_hi)]
        if len(g)<5: continue
        N,pf_,tot,wr,sl=stats(g)
        print(f"  {label:<25s} {N:>4d} {pf_:5.2f} {tot:+7.0f} {wr:5.1f} {sl:3d}")

    # ── against_5 buckets ──
    print(f"\n  ── Bucket by against_5 (number of last 5 M5 bars going AGAINST direction) ──")
    print(f"  {'bucket':<25s} {'N':>4s} {'PF':>5s} {'TotalR':>7s} {'WR%':>5s} {'SL':>3s}")
    for k in range(0,6):
        g=df[df.against_5==k]
        if len(g)<5: continue
        N,pf_,tot,wr,sl=stats(g)
        print(f"  against_5 = {k}                {N:>4d} {pf_:5.2f} {tot:+7.0f} {wr:5.1f} {sl:3d}")

if __name__=="__main__":
    evaluate("Oracle XAU — micro-regime diagnosis",
             f"{DATA}/swing_v5_xauusd.csv",
             f"{PROJECT}/experiments/v84_rl_entry/v84_rl_trades.csv")
    evaluate("Oracle BTC — micro-regime diagnosis",
             f"{DATA}/swing_v5_btc.csv",
             f"{PROJECT}/experiments/v84_rl_entry/btc_rl_trades.csv")
