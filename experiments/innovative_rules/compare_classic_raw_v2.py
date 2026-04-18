"""
Fair comparison: score existing XAU rules at TP=6×ATR/SL=2×ATR
(same geometry as innovative Phase 1), using forward OHLC.
"""
import glob, os
import numpy as np
import pandas as pd

ROOT = "/home/jay/Desktop/new-model-zigzag"
TP_MULT, SL_MULT, MAX_FWD = 6.0, 2.0, 40

# Load raw OHLC
swing = pd.read_csv(f"{ROOT}/data/swing_v5_xauusd.csv",
                    usecols=["time","high","low","close"],
                    parse_dates=["time"]).sort_values("time").reset_index(drop=True)
h = swing["high"].values; l = swing["low"].values; c = swing["close"].values
# ATR14
tr = np.concatenate([[h[0]-l[0]],
      np.maximum.reduce([h[1:]-l[1:], np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])])])
atr = pd.Series(tr).rolling(14, min_periods=14).mean().values
time_to_idx = pd.Series(swing.index.values, index=swing["time"].values)

def fwd(i, direction):
    a = atr[i]
    if not np.isfinite(a) or a <= 0: return -1
    e = c[i]
    if direction == 1:
        tp = e + TP_MULT*a; sl = e - SL_MULT*a
        for k in range(i+1, min(i+1+MAX_FWD, len(c))):
            if l[k] <= sl: return 0
            if h[k] >= tp: return 1
    else:
        tp = e - TP_MULT*a; sl = e + SL_MULT*a
        for k in range(i+1, min(i+1+MAX_FWD, len(c))):
            if h[k] >= sl: return 0
            if l[k] <= tp: return 1
    return 2

# Scan all classic setup rows, use same last 150k bar window as innovative
start = len(swing) - 150_000
rows = []
for f in sorted(glob.glob(f"{ROOT}/data/setups_*.csv")):
    cid = os.path.basename(f).replace("setups_","").replace(".csv","")
    df = pd.read_csv(f, parse_dates=["time"])
    for rule, grp in df.groupby("rule"):
        wins=tot=0
        for _, r in grp.iterrows():
            t = r["time"]
            if t not in time_to_idx.index: continue
            i = int(time_to_idx[t])
            if i < start or i >= len(c) - MAX_FWD - 1: continue
            direc = int(r["direction"])
            out = fwd(i, direc)
            if out in (0, 1):
                tot += 1
                if out == 1: wins += 1
        if tot >= 30:
            wr = wins/tot; pf = (wr*TP_MULT)/((1-wr)*SL_MULT+1e-9)
            rows.append((cid, rule, tot, wr, pf))

rows.sort(key=lambda x: -x[4])
print(f"{'cluster/rule':<30} {'n':<6} {'WR':<6}  {'PF at 6:2':<10}")
print("-" * 60)
for cid, rule, n, wr, pf in rows:
    flag = "✓" if pf >= 1.0 else "✗"
    print(f"  C{cid} {rule:<25} {n:<6} {wr:.1%}  {pf:<6.2f}  {flag}")
