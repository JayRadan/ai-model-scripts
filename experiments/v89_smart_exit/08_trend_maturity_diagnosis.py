"""Test the refined hypothesis: trades opened on top-of-rally / bottom-of-selloff
(extended trends near exhaustion) are the losers. Trades opened in young /
mid trends are the winners.

Features at entry time:
  stretch_from_low_100   = (close - min(close[t-100:t])) / atr,  signed by direction
                           High value = price already extended in trade direction
                           Low value  = trade direction is fresh
  stretch_high_water     = (close - min(close[t-50:t])) / atr   (50-bar version)
  drawup_pct_50          = % move from recent low, signed by direction (50-bar window)
  in_regime_bars         = number of consecutive M5 bars in the current regime cluster
                           (proxy for "regime maturity" / how long the trend has lasted)
  pct_to_recent_extreme  = how close current price is to the highest/lowest of last 50 bars

If the user's hypothesis is correct, trades with HIGH stretch / HIGH in_regime_bars
should have measurably lower WR / PF / Total R.
"""
import os
import numpy as np, pandas as pd
from importlib.util import spec_from_file_location, module_from_spec

PROJECT="/home/jay/Desktop/new-model-zigzag"
DATA=f"{PROJECT}/data"
spec=spec_from_file_location("v89", os.path.join(os.path.dirname(os.path.abspath(__file__)),"01_optimal_stopping.py"))
v89=module_from_spec(spec); spec.loader.exec_module(v89)
NAMES={0:"Uptrend",1:"MeanRevert",2:"TrendRange",3:"Downtrend",4:"HighVol"}

def maturity_feats_at_entry(t_idx, d, C, atr, cid_series=None):
    ea = atr[t_idx]
    if not np.isfinite(ea) or ea <= 0: return None

    feats = {}
    # Stretch from rolling low (long) / high (short) over 100 bars
    L = 100
    if t_idx >= L:
        win = C[t_idx-L:t_idx+1]
        if d == 1:   # long → measure how far we are above recent low
            feats['stretch_100'] = float((C[t_idx] - win.min()) / ea)
        else:        # short → how far below recent high
            feats['stretch_100'] = float((win.max() - C[t_idx]) / ea)
    else:
        feats['stretch_100'] = 0.0

    # 50-bar version
    L = 50
    if t_idx >= L:
        win = C[t_idx-L:t_idx+1]
        if d == 1:
            feats['stretch_50'] = float((C[t_idx] - win.min()) / ea)
        else:
            feats['stretch_50'] = float((win.max() - C[t_idx]) / ea)
    else:
        feats['stretch_50'] = 0.0

    # 200-bar version (longer-term trend extension)
    L = 200
    if t_idx >= L:
        win = C[t_idx-L:t_idx+1]
        if d == 1:
            feats['stretch_200'] = float((C[t_idx] - win.min()) / ea)
        else:
            feats['stretch_200'] = float((win.max() - C[t_idx]) / ea)
    else:
        feats['stretch_200'] = 0.0

    # % to recent extreme (high if long, low if short) — proximity to top/bottom
    L = 50
    if t_idx >= L:
        win = C[t_idx-L:t_idx+1]
        if d == 1:    # long → how close to the recent high (1 = at top)
            rng = win.max() - win.min()
            feats['pct_to_extreme_50'] = float((C[t_idx] - win.min()) / rng) if rng > 0 else 0.5
        else:         # short → how close to the recent low
            rng = win.max() - win.min()
            feats['pct_to_extreme_50'] = float((win.max() - C[t_idx]) / rng) if rng > 0 else 0.5
    else:
        feats['pct_to_extreme_50'] = 0.5

    # Bars in same regime (rough — count consecutive bars with same cid)
    if cid_series is not None and t_idx > 0:
        cur_cid = cid_series[t_idx]
        bars = 0
        j = t_idx - 1
        while j >= 0 and cid_series[j] == cur_cid:
            bars += 1; j -= 1
        feats['bars_in_regime'] = float(min(bars, 500))
    else:
        feats['bars_in_regime'] = 0.0

    return feats

def stats(g):
    if len(g) == 0: return (0, 0.0, 0.0, 0.0, 0)
    wr = (g.pnl_R > 0).mean() * 100
    pf_ = g[g.pnl_R > 0].pnl_R.sum() / max(-g[g.pnl_R <= 0].pnl_R.sum(), 1e-9)
    total = g.pnl_R.sum()
    sl = (g.pnl_R <= -3.9).sum()
    return len(g), pf_, total, wr, sl

def evaluate(name, swing_csv, trades_csv, setups_glob):
    print("\n"+"="*72); print(f"  {name}"); print("="*72)
    n, C, atr, t2i, ctx = v89.load_market(swing_csv, setups_glob)
    trades = pd.read_csv(trades_csv, parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    split = trades["time"].quantile(0.70)
    tst = trades[trades["time"] >= split].reset_index(drop=True)
    print(f"  Test (unseen): {len(tst)} trades")

    # We don't have a per-bar regime cid feed handy here, so bars_in_regime
    # uses the trade-level cid as a coarse proxy (skip the calculation).
    rows = []
    for _, trade in tst.iterrows():
        tm = trade["time"]
        if tm not in t2i: continue
        ei = t2i[tm]; d = int(trade["direction"])
        f = maturity_feats_at_entry(ei, d, C, atr, cid_series=None)
        if f is None: continue
        f['cid'] = int(trade["cid"]); f['direction'] = d; f['pnl_R'] = float(trade["pnl_R"])
        rows.append(f)
    df = pd.DataFrame(rows)

    print(f"  Overall: WR={(df.pnl_R>0).mean()*100:.1f}%  Total={df.pnl_R.sum():+.0f}R  "
          f"PF={df[df.pnl_R>0].pnl_R.sum()/max(-df[df.pnl_R<=0].pnl_R.sum(),1e-9):.2f}")

    # ── Stretch from 100-bar extreme ──
    print(f"\n  ── stretch_100 (price already extended in direction, in ATR units) ──")
    print(f"  Higher = trade direction is already a mature move. User hypothesis: high → bad")
    print(f"  {'bucket':<25s} {'N':>4s} {'PF':>5s} {'Total':>7s} {'WR%':>5s} {'SL':>3s}")
    for lo, hi, label in [(0, 1, '0-1 ATR (fresh)'), (1, 3, '1-3 (young)'),
                          (3, 6, '3-6 (mid)'), (6, 10, '6-10 (mature)'),
                          (10, 15, '10-15 (extended)'), (15, 999, '15+ (very stretched)')]:
        g = df[(df.stretch_100 >= lo) & (df.stretch_100 < hi)]
        if len(g) < 5: continue
        N, pf_, tot, wr, sl = stats(g)
        marker = " ⚠ POOR" if (pf_ < 2.0 and N >= 15) else (" ★" if pf_ > 6 else "")
        print(f"  {label:<25s} {N:>4d} {pf_:5.2f} {tot:+7.0f} {wr:5.1f} {sl:3d}{marker}")

    # ── stretch_200 ──
    print(f"\n  ── stretch_200 (longer-term extension) ──")
    print(f"  {'bucket':<25s} {'N':>4s} {'PF':>5s} {'Total':>7s} {'WR%':>5s} {'SL':>3s}")
    for lo, hi, label in [(0, 2, '0-2 ATR'), (2, 5, '2-5'), (5, 10, '5-10'),
                          (10, 15, '10-15 (mature)'), (15, 25, '15-25 (extended)'),
                          (25, 999, '25+ (very stretched)')]:
        g = df[(df.stretch_200 >= lo) & (df.stretch_200 < hi)]
        if len(g) < 5: continue
        N, pf_, tot, wr, sl = stats(g)
        marker = " ⚠ POOR" if (pf_ < 2.0 and N >= 15) else (" ★" if pf_ > 6 else "")
        print(f"  {label:<25s} {N:>4d} {pf_:5.2f} {tot:+7.0f} {wr:5.1f} {sl:3d}{marker}")

    # ── pct_to_extreme_50 ──
    print(f"\n  ── pct_to_extreme_50 (0=at low/start, 1=at top of recent move) ──")
    print(f"  Higher = price closer to recent peak in trade direction. User hypothesis: high → bad")
    print(f"  {'bucket':<25s} {'N':>4s} {'PF':>5s} {'Total':>7s} {'WR%':>5s} {'SL':>3s}")
    for lo, hi, label in [(0.0, 0.2, '0-20% (near opposite)'),
                          (0.2, 0.4, '20-40%'),
                          (0.4, 0.6, '40-60% (middle)'),
                          (0.6, 0.8, '60-80%'),
                          (0.8, 1.01, '80-100% (near top)')]:
        g = df[(df.pct_to_extreme_50 >= lo) & (df.pct_to_extreme_50 < hi)]
        if len(g) < 5: continue
        N, pf_, tot, wr, sl = stats(g)
        marker = " ⚠ POOR" if (pf_ < 2.0 and N >= 15) else (" ★" if pf_ > 6 else "")
        print(f"  {label:<25s} {N:>4d} {pf_:5.2f} {tot:+7.0f} {wr:5.1f} {sl:3d}{marker}")

    # ── Per-cluster: trending regimes only (cid 0 and 3) ──
    print(f"\n  ── stretch_100 BY trending regime ──")
    for cid in [0, 3]:
        sub = df[df.cid == cid]
        if len(sub) < 30: continue
        print(f"\n  {NAMES[cid]} (cid={cid}, N={len(sub)}):")
        print(f"  {'bucket':<25s} {'N':>4s} {'PF':>5s} {'Total':>7s} {'WR%':>5s}")
        for lo, hi, label in [(0, 3, '0-3 ATR (young)'),
                              (3, 8, '3-8 (mid)'),
                              (8, 15, '8-15 (mature)'),
                              (15, 999, '15+ (extended)')]:
            g = sub[(sub.stretch_100 >= lo) & (sub.stretch_100 < hi)]
            if len(g) < 5: continue
            N, pf_, tot, wr, sl = stats(g)
            marker = " ⚠" if (pf_ < 2.0 and N >= 10) else ""
            print(f"  {label:<25s} {N:>4d} {pf_:5.2f} {tot:+7.0f} {wr:5.1f}{marker}")

if __name__ == "__main__":
    evaluate("Oracle XAU — trend-maturity diagnosis",
             f"{DATA}/swing_v5_xauusd.csv",
             f"{PROJECT}/experiments/v84_rl_entry/v84_rl_trades.csv",
             f"{DATA}/setups_*_v72l.csv")
    evaluate("Oracle BTC — trend-maturity diagnosis",
             f"{DATA}/swing_v5_btc.csv",
             f"{PROJECT}/experiments/v84_rl_entry/btc_rl_trades.csv",
             f"{DATA}/setups_*_v72l_btc.csv")
