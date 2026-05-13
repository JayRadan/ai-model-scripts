"""v91 alt — use trend_consistency as the override signal.

The user's complaint: HighVol when 'actually trending'. The K-means
fingerprint has a `trend_consistency` feature (-1 to +1, smoothness of
direction). When trend_consistency is HIGH, the market IS directional
even if volatility is also high.

Proposed override:
  if K-means → HighVol AND trend_consistency_raw > 0.55:
    if weekly_return > 0: relabel to Uptrend
    if weekly_return < 0: relabel to Downtrend

If trend_consistency_raw is high enough, the market has clear direction
regardless of volatility level — that's the v91 hypothesis.
"""
import os, glob as _glob
import numpy as np, pandas as pd
PROJECT = "/home/jay/Desktop/new-model-zigzag"
DATA = f"{PROJECT}/data"
HOLDOUT_START = pd.Timestamp("2024-12-12")
NAMES = {0:"Uptrend",1:"MeanRevert",2:"TrendRange",3:"Downtrend",4:"HighVol"}
HIGHVOL_CID=4; UP_CID=0; DOWN_CID=3
WEEKLY_THR = 0.003

def analyze(name, fp_csv, consistency_thr):
    print(f"\n--- {name}: consistency_thr = {consistency_thr} ---")
    fp = pd.read_csv(fp_csv, parse_dates=["center_time"]).sort_values("end_idx").reset_index(drop=True)
    h = fp[fp['center_time'] >= HOLDOUT_START].reset_index(drop=True)

    def old(c, w):
        if c != HIGHVOL_CID: return c
        if w > WEEKLY_THR: return UP_CID
        if w < -WEEKLY_THR: return DOWN_CID
        return c

    def v91(c, w, t):
        if c != HIGHVOL_CID: return c
        # Old weekly rule
        if w > WEEKLY_THR: return UP_CID
        if w < -WEEKLY_THR: return DOWN_CID
        # NEW: high consistency → directional even for small weekly return
        if t > consistency_thr:
            return UP_CID if w > 0 else DOWN_CID
        return c

    h['old'] = [old(int(c), float(w)) for c,w in zip(h['cluster'], h['weekly_return_pct'])]
    h['v91'] = [v91(int(c), float(w), float(t)) for c,w,t in zip(h['cluster'], h['weekly_return_pct'], h['trend_consistency'])]
    diff = (h['old'] != h['v91']).sum()
    hv_old = (h['old'] == HIGHVOL_CID).sum()
    hv_v91 = (h['v91'] == HIGHVOL_CID).sum()
    print(f"  HighVol count: {hv_old} → {hv_v91} ({hv_old - hv_v91} relabeled)")
    if diff > 0:
        print(f"  {diff} overrides")
        d = h[h['old'] != h['v91']][['center_time','cluster','old','v91','weekly_return_pct','trend_consistency']]
        for _, r in d.head(10).iterrows():
            print(f"    {str(r['center_time'])[:19]}  cluster={int(r['cluster'])}  "
                  f"{NAMES[int(r['old'])]:>10} → {NAMES[int(r['v91'])]:<10}  "
                  f"weekly={r['weekly_return_pct']*100:+.2f}%  consistency={r['trend_consistency']:+.2f}")

for t in [0.50, 0.55, 0.60, 0.65, 0.70]:
    analyze("XAU", f"{DATA}/regime_fingerprints_K4.csv", t)
for t in [0.50, 0.55, 0.60, 0.65, 0.70]:
    analyze("BTC", f"{DATA}/regime_fingerprints_btc_K5.csv", t)
