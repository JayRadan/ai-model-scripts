"""v91 — smarter regime relabel for XAU/BTC.

Problem (user complaint, May 11): regime said HighVol all day but market
was clearly uptrending intraday. The current relabel rule only fires if
WEEKLY return crosses ±0.3% — which misses strong intraday moves.

Proposed fix: extend the relabel to consider BOTH weekly AND daily
(24h) returns:

  if K-means → HighVol:
    weekly_return > +0.3%  → Uptrend
    daily_return  > +0.8%  → Uptrend       (NEW)
    weekly_return < -0.3%  → Downtrend
    daily_return  < -0.8%  → Downtrend     (NEW)
    else stay HighVol

This catches both slow-week trends AND clear single-day breakouts.

This experiment:
  1. Compute current K-means + old relabel for every historical bar
  2. Compute K-means + NEW dual-window relabel
  3. Compare label distributions — how often does the new rule override?
  4. Where they DIFFER, what was the actual market doing?
  5. Verify the new labels are more "intuitive" using ground truth (next-24h return)
"""
import os, glob as _glob, time, pickle, json
import numpy as np, pandas as pd

PROJECT = "/home/jay/Desktop/new-model-zigzag"
DATA = f"{PROJECT}/data"
HOLDOUT_START = pd.Timestamp("2024-12-12")
NAMES = {0:"Uptrend",1:"MeanRevert",2:"TrendRange",3:"Downtrend",4:"HighVol"}

# Current XAU relabel thresholds
WEEKLY_THR = 0.003          # 0.3% weekly (current)
# NEW daily thresholds for v91
DAILY_THR_XAU = 0.008       # 0.8% daily (~2.5σ of XAU daily moves)
DAILY_THR_BTC = 0.020       # 2.0% daily (BTC moves bigger)

HIGHVOL_CID = 4; UP_CID = 0; DOWN_CID = 3

def apply_old_relabel(km_cid, weekly_return_raw):
    if km_cid != HIGHVOL_CID: return km_cid
    if weekly_return_raw > +WEEKLY_THR: return UP_CID
    if weekly_return_raw < -WEEKLY_THR: return DOWN_CID
    return km_cid

def apply_v91_relabel(km_cid, weekly_return_raw, daily_return_raw, daily_thr):
    if km_cid != HIGHVOL_CID: return km_cid
    # Either weekly OR daily can trigger the directional relabel
    if weekly_return_raw > +WEEKLY_THR: return UP_CID
    if daily_return_raw  > +daily_thr:  return UP_CID
    if weekly_return_raw < -WEEKLY_THR: return DOWN_CID
    if daily_return_raw  < -daily_thr:  return DOWN_CID
    return km_cid

def load_market(swing_csv):
    sw = pd.read_csv(swing_csv, parse_dates=["time"]).sort_values("time").drop_duplicates('time',keep='last').reset_index(drop=True)
    return sw

def analyze(name, swing_csv, fp_csv, daily_thr):
    print(f"\n{'='*72}\n  {name}\n{'='*72}")
    sw = load_market(swing_csv)
    n = len(sw); C = sw["close"].values
    fp = pd.read_csv(fp_csv, parse_dates=["center_time"]).sort_values("end_idx").reset_index(drop=True)

    # For each fp block, compute daily_return = close at end_idx vs close 288 bars before
    daily_returns = np.zeros(len(fp), dtype=np.float32)
    for i in range(len(fp)):
        end = int(fp["end_idx"].iat[i])
        if end - 288 < 0 or end >= n: continue
        c0 = float(C[end-288]); ct = float(C[end])
        if c0 > 0: daily_returns[i] = (ct - c0) / c0
    fp['daily_return_raw'] = daily_returns

    # Build labels
    fp['old_label'] = [apply_old_relabel(int(c), float(w))
                       for c,w in zip(fp['cluster'], fp['weekly_return_pct'])]
    fp['v91_label'] = [apply_v91_relabel(int(c), float(w), float(d), daily_thr)
                       for c,w,d in zip(fp['cluster'], fp['weekly_return_pct'], fp['daily_return_raw'])]

    # Holdout slice
    holdout = fp[fp['center_time'] >= HOLDOUT_START].reset_index(drop=True)
    print(f"  Holdout blocks: {len(holdout):,}")
    print(f"  Threshold:  weekly > {WEEKLY_THR*100:.1f}%  OR  daily > {daily_thr*100:.1f}%")

    # Distribution comparison
    def dist(col):
        out = {}
        for c in range(5):
            out[NAMES[c]] = int((holdout[col] == c).sum())
        return out
    print(f"\n  Label distribution on holdout:")
    print(f"  {'regime':<12} {'OLD':>6} {'v91':>6} {'Δ':>6}")
    old_d = dist('old_label'); new_d = dist('v91_label')
    for r in ['Uptrend','MeanRevert','TrendRange','Downtrend','HighVol']:
        print(f"  {r:<12} {old_d[r]:>6d} {new_d[r]:>6d} {new_d[r]-old_d[r]:>+6d}")

    # Where labels DIFFER, what was the market doing?
    diff_mask = holdout['old_label'] != holdout['v91_label']
    n_diff = int(diff_mask.sum())
    print(f"\n  Blocks where v91 differs from OLD: {n_diff} ({n_diff*100/len(holdout):.1f}%)")

    if n_diff > 0:
        diff_df = holdout[diff_mask].copy()
        # For each diff block, what was the daily_return and the actual next-24h return?
        diff_df['old_name'] = diff_df['old_label'].map(NAMES)
        diff_df['v91_name'] = diff_df['v91_label'].map(NAMES)

        # Ground truth: next-block daily_return (forward), to check accuracy
        for_returns = np.zeros(len(diff_df), dtype=np.float32)
        for j, (_, row) in enumerate(diff_df.iterrows()):
            end = int(row['end_idx'])
            fwd_end = min(end + 288, n - 1)
            if end + 1 >= n: continue
            c0 = float(C[end]); cf = float(C[fwd_end])
            if c0 > 0: for_returns[j] = (cf - c0) / c0
        diff_df['next_24h_return'] = for_returns

        # Show a few examples
        print(f"\n  Sample v91 overrides (first 12):")
        print(f"  {'center_time':<20} {'old → v91':<28} {'24h ret (now)':>14} {'24h ret (next)':>14}")
        for _, row in diff_df.head(12).iterrows():
            print(f"  {str(row['center_time'])[:19]:<20} "
                  f"{row['old_name']:>10s} → {row['v91_name']:<14s} "
                  f"{row['daily_return_raw']*100:>+12.2f}%  "
                  f"{row['next_24h_return']*100:>+12.2f}%")

        # Accuracy check: when v91 says Uptrend, what did the market do next 24h?
        for label_to_check, name_to_check in [(UP_CID, "Uptrend"), (DOWN_CID, "Downtrend")]:
            v91_mask = diff_df['v91_label'] == label_to_check
            if v91_mask.sum() == 0: continue
            fwd_pct = diff_df.loc[v91_mask, 'next_24h_return'] * 100
            agree = ((fwd_pct > 0) if label_to_check == UP_CID else (fwd_pct < 0)).mean()
            mean_fwd = fwd_pct.mean()
            print(f"\n  v91 → {name_to_check}: N={v91_mask.sum()}, "
                  f"next-24h mean return = {mean_fwd:+.2f}%, "
                  f"directional accuracy = {agree*100:.0f}%")

if __name__ == "__main__":
    # Sweep daily-threshold for XAU
    print("\n████ XAU threshold sweep ████")
    for d in [0.002, 0.003, 0.004, 0.005, 0.008]:
        print(f"\n----- daily_thr = {d*100:.2f}% -----")
        analyze("Oracle XAU",
                f"{DATA}/swing_v5_xauusd.csv",
                f"{DATA}/regime_fingerprints_K4.csv",
                d)
    # Sweep daily-threshold for BTC
    print("\n████ BTC threshold sweep ████")
    for d in [0.005, 0.010, 0.015, 0.020]:
        print(f"\n----- daily_thr = {d*100:.2f}% -----")
        analyze("Oracle BTC",
                f"{DATA}/swing_v5_btc.csv",
                f"{DATA}/regime_fingerprints_btc_K5.csv",
                d)
