"""SL-hunt diagnostic.

For every holdout trade that hit hard SL (R <= -4.0 ATR), measure what
price did over the NEXT N bars in the direction the trade had wanted.

If a large % of SL hits are followed by favorable moves within 10-30
bars, the 4R SL is being hunted (broker liquidity grab / classic stop
sweep) and a wider SL (or different SL logic) would recover money.
"""
import os, sys, glob as _glob
import numpy as np, pandas as pd

PROJECT = "/home/jay/Desktop/new-model-zigzag"
DATA = f"{PROJECT}/data"
HOLDOUT_START = pd.Timestamp("2024-12-12")
SL_HARD = -4.0; MAX_HOLD = 60; TRAIL_ACT = 3.0; TRAIL_GB = 0.60
LOOK_AHEAD = [10, 30, 60, 120]   # bars to look ahead after SL hit

def load_market(swing_csv):
    sw = pd.read_csv(swing_csv, parse_dates=["time"]).sort_values("time")\
           .drop_duplicates('time', keep='last').reset_index(drop=True)
    n = len(sw); C = sw["close"].values.astype(np.float64)
    H = sw["high"].values; Lo = sw["low"].values
    tr = np.concatenate([[H[0]-Lo[0]],
        np.maximum.reduce([H[1:]-Lo[1:], np.abs(H[1:]-C[:-1]), np.abs(Lo[1:]-C[:-1])])])
    atr = pd.Series(tr).rolling(14, min_periods=14).mean().values
    t2i = {pd.Timestamp(t):i for i,t in enumerate(sw["time"].values)}
    return sw, n, C, atr, t2i

def load_setups(setups_glob):
    parts = []
    for f in sorted(_glob.glob(setups_glob)):
        df = pd.read_csv(f, parse_dates=["time"]); parts.append(df)
    return pd.concat(parts, ignore_index=True).sort_values(['time','direction'])\
             .drop_duplicates(['time','direction'], keep='first').reset_index(drop=True)

def sim_with_exit_info(t_idx, d, n, C, atr):
    """Return (R, exit_bar_offset_k, exit_reason). Reasons:
       'sl' = hard SL hit, 'trail' = trailing stop, 'maxhold' = bar 60.
    """
    ep = C[t_idx]; ea = atr[t_idx]
    if not np.isfinite(ea) or ea <= 0: return None
    peak = 0.0
    max_k = min(MAX_HOLD, n-t_idx-1)
    for k in range(1, max_k+1):
        bar = t_idx + k
        R = d * (C[bar] - ep) / ea
        if R <= SL_HARD: return (R, k, 'sl')
        if R > peak: peak = R
        if peak >= TRAIL_ACT and R <= peak * (1.0 - TRAIL_GB): return (R, k, 'trail')
    last = min(t_idx + max_k, n-1)
    return (d * (C[last] - ep) / ea, max_k, 'maxhold')

def run(name, swing_csv, setups_glob):
    print(f"\n{'='*72}\n  {name}\n{'='*72}")
    sw, n, C, atr, t2i = load_market(swing_csv)
    setups = load_setups(setups_glob)
    idx = np.full(len(setups), -1, dtype=np.int64)
    for i, t in enumerate(setups['time']):
        ti = pd.Timestamp(t)
        if ti in t2i: idx[i] = t2i[ti]
    keep = idx >= 0
    setups = setups[keep].reset_index(drop=True); idx = idx[keep]
    is_test = setups['time'].values >= HOLDOUT_START.to_datetime64()
    setups = setups[is_test].reset_index(drop=True); idx = idx[is_test]
    dirs = setups['direction'].values
    print(f"  holdout setups: {len(setups):,}")

    sl_hits = []     # rows: (setup_idx, exit_k, direction, entry_idx, entry_price, atr_at_entry)
    all_exits = {'sl':0, 'trail':0, 'maxhold':0}
    for i in range(len(setups)):
        info = sim_with_exit_info(int(idx[i]), int(dirs[i]), n, C, atr)
        if info is None: continue
        R, k, reason = info
        all_exits[reason] = all_exits.get(reason, 0) + 1
        if reason == 'sl':
            t_idx = int(idx[i])
            sl_hits.append((t_idx + k, int(dirs[i]), t_idx, C[t_idx], atr[t_idx]))

    print(f"  exits: sl={all_exits['sl']:,}  trail={all_exits['trail']:,}  "
          f"maxhold={all_exits['maxhold']:,}")
    print(f"  SL-hit fraction: {all_exits['sl']*100/max(sum(all_exits.values()),1):.1f}%")
    if not sl_hits:
        print("  no SL hits, nothing to analyse"); return

    print(f"\n  After SL hit, what does price do in the favored direction over next N bars?")
    print(f"  (favored direction = the direction the original trade wanted)")
    print(f"  positive value = price moved favorable AFTER stop hit (= stop-hunt pattern)")
    print(f"  {'bars':>5s}  {'mean_R':>8s}  {'median_R':>9s}  {'pct_recovered':>14s}  "
          f"{'pct_full_4R':>12s}")
    for lh in LOOK_AHEAD:
        post_R = []
        recovered = 0; full = 0
        for (exit_idx, d, _, _, ea) in sl_hits:
            end_idx = min(exit_idx + lh, n - 1)
            if end_idx <= exit_idx or ea <= 0: continue
            # measure peak favorable move from SL exit price (= C[exit_idx])
            ep = C[exit_idx]
            win = C[exit_idx:end_idx+1]
            if d == 1:
                peak_fav = (win.max() - ep) / ea     # how high price went after SL
            else:
                peak_fav = (ep - win.min()) / ea     # how low price went after SL
            post_R.append(peak_fav)
            if peak_fav >= 1.0: recovered += 1
            if peak_fav >= 4.0: full += 1
        post_R = np.array(post_R)
        n_eval = len(post_R)
        print(f"  {lh:>5d}  {post_R.mean():>+8.2f}  {np.median(post_R):>+9.2f}  "
              f"{recovered*100/n_eval:>13.1f}%  {full*100/n_eval:>11.1f}%")

    print(f"\n  what fraction of SL hits would have been WINNERS with a wider SL?")
    print(f"  (run the same simulator with SL_HARD = -X instead of -4)")
    for wider in [-5.0, -6.0, -8.0, -10.0]:
        saved = 0; new_wins = 0
        for (_, d, t_idx, ep_unused, ea) in sl_hits:
            if not np.isfinite(ea) or ea <= 0: continue
            ep = C[t_idx]
            peak = 0.0
            max_k = min(MAX_HOLD, n-t_idx-1)
            R_final = None; k_final = None
            for k in range(1, max_k+1):
                bar = t_idx + k
                R = d * (C[bar] - ep) / ea
                if R <= wider:
                    R_final = R; k_final = k; break
                if R > peak: peak = R
                if peak >= TRAIL_ACT and R <= peak * (1.0 - TRAIL_GB):
                    R_final = R; k_final = k; break
            if R_final is None:
                R_final = d * (C[min(t_idx+max_k, n-1)] - ep) / ea
            if R_final > -4.0: saved += 1
            if R_final > 0:    new_wins += 1
        pct_saved = saved * 100 / len(sl_hits)
        pct_win   = new_wins * 100 / len(sl_hits)
        print(f"    SL={wider:>5.1f}: {saved}/{len(sl_hits)} saved ({pct_saved:.1f}%), "
              f"{new_wins} turned positive ({pct_win:.1f}%)")

if __name__ == "__main__":
    run("XAU", f"{DATA}/swing_v5_xauusd.csv", f"{DATA}/setups_*_v72l.csv")
    run("BTC", f"{DATA}/swing_v5_btc.csv", f"{DATA}/setups_*_v72l_btc.csv")
