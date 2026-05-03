"""
v8.4b — Sweep guard exit thresholds [0.50, 1.0, 1.5, 2.0, 2.5, 3.0]R
WITHOUT the broken pivot pre-filter. Pure exit modification.

Goal: find the highest WR achievable with PF preserved (or close to it).

Logic per trade:
  Walk bar-by-bar from entry to original exit. If pnl_R hits +G first,
  replace exit with pnl_R = G. Otherwise keep original.

The guard never makes losers worse — it only caps winners. So WR
monotonically increases as G decreases (more wins captured before they
turn into losers). PF and R have a non-monotonic relationship — small G
crushes PF (we proved this in v8.4 with G=0.50). Larger G preserves
more upside.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

ROOT = "/home/jay/Desktop/new-model-zigzag"

GUARD_LEVELS = [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00]


def pf(rs):
    pos = rs[rs > 0].sum(); neg = -rs[rs <= 0].sum()
    return pos / max(neg, 1e-9)


def max_dd(rs):
    eq = np.cumsum(rs); peak = np.maximum.accumulate(eq)
    return (eq - peak).min() if len(eq) else 0.0


def load_swing_with_atr(path):
    swing = pd.read_csv(path, parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    H = swing.high.values; L = swing.low.values; C = swing.close.values
    tr = np.concatenate([[H[0]-L[0]],
          np.maximum.reduce([H[1:]-L[1:],
                              np.abs(H[1:]-C[:-1]),
                              np.abs(L[1:]-C[:-1])])])
    swing["atr"] = pd.Series(tr).rolling(14, min_periods=14).mean().values
    return swing


def simulate_guard(trade, swing, guard_R):
    idx = swing.index[swing["time"] == trade["time"]]
    if len(idx) == 0: return float(trade["pnl_R"])
    ei = int(idx[0])
    ep = float(swing.close.iat[ei]); ea = float(swing.atr.iat[ei])
    if not np.isfinite(ea) or ea <= 0: return float(trade["pnl_R"])
    bars_held = int(trade["bars"]); direction = int(trade["direction"])
    for k in range(1, bars_held + 1):
        bar = ei + k
        if bar >= len(swing): break
        cp = direction * (float(swing.close.iat[bar]) - ep) / ea
        if cp >= guard_R:
            return guard_R
    return float(trade["pnl_R"])


def evaluate(name, trades, swing):
    rs0 = trades["pnl_R"].values
    base = {"n": len(rs0), "wr": (rs0 > 0).mean(), "pf": pf(rs0),
             "r": rs0.sum(), "dd": max_dd(rs0)}
    print(f"\n=== {name} ===")
    print(f"baseline (no guard):  n={base['n']}  WR={base['wr']:.1%}  "
          f"PF={base['pf']:.2f}  R={base['r']:+.0f}  DD={base['dd']:+.0f}")
    print(f"\n{'guard':>6}  {'n':>5}  {'WR':>6}  {'PF':>5}  {'R':>8}  {'DD':>7}  "
          f"{'ΔWR_pp':>8}  {'ΔPF':>6}  {'ΔR':>8}  {'fired%':>6}")
    rows = []
    for G in GUARD_LEVELS:
        new_pnl = np.array([simulate_guard(trades.iloc[i], swing, G)
                              for i in range(len(trades))])
        # Approximate "guard fired" — count of trades whose pnl exactly
        # equals G (within float tolerance) AND wasn't already that value
        # at original exit.
        fired = (np.abs(new_pnl - G) < 1e-9) & (np.abs(rs0 - G) > 1e-6)
        wr = (new_pnl > 0).mean()
        pf_ = pf(new_pnl); r_ = new_pnl.sum(); dd_ = max_dd(new_pnl)
        rows.append({
            "guard_R": G, "n": len(new_pnl),
            "WR": round(wr, 4), "PF": round(pf_, 3), "R": round(r_, 1),
            "DD": round(dd_, 1),
            "ΔWR_pp": round((wr - base["wr"]) * 100, 1),
            "ΔPF":    round(pf_ - base["pf"], 2),
            "ΔR":     round(r_ - base["r"], 1),
            "fired_pct": round(fired.mean(), 3),
        })
        print(f"{G:>6.2f}  {len(new_pnl):>5}  {wr*100:>5.1f}%  {pf_:>5.2f}  "
              f"{r_:>+8.0f}  {dd_:>+7.0f}  {(wr-base['wr'])*100:>+7.1f}  "
              f"{pf_-base['pf']:>+6.2f}  {r_-base['r']:>+8.0f}  {fired.mean()*100:>5.1f}%")
    return pd.DataFrame(rows), base


def main():
    print("Loading XAU swing + ATR…")
    swing_x = load_swing_with_atr(os.path.join(ROOT, "data/swing_v5_xauusd.csv"))
    print("Loading BTC swing + ATR…")
    swing_b = load_swing_with_atr(os.path.join(ROOT, "data/swing_v5_btc.csv"))

    out = {}
    for name, path, swing in [
        ("Oracle XAU", "data/v72l_trades_holdout.csv", swing_x),
        ("Midas XAU",  "data/v6_trades_holdout_xau.csv", swing_x),
        ("Oracle BTC", "data/v72l_trades_holdout_btc.csv", swing_b),
    ]:
        trades = pd.read_csv(os.path.join(ROOT, path), parse_dates=["time"])
        df, base = evaluate(name, trades, swing)
        df.to_csv(os.path.join(os.path.dirname(__file__),
                                f"sweep_{name.lower().replace(' ','_')}.csv"), index=False)
        out[name] = (df, base)

    # Summary: best guard per product (highest WR with PF >= 0.9 × baseline AND R > 0)
    print("\n" + "=" * 78)
    print("BEST GUARD PER PRODUCT (highest WR with PF >= 0.9×baseline AND R > 0):")
    for name, (df, base) in out.items():
        ok = df[(df["PF"] >= base["pf"] * 0.9) & (df["R"] > 0)]
        if len(ok):
            best = ok.sort_values("WR", ascending=False).iloc[0]
            print(f"  {name:12s} guard={best['guard_R']:.2f}R  "
                  f"WR={best['WR']*100:.1f}%  PF={best['PF']:.2f}  "
                  f"R={best['R']:+.0f}  ΔWR={best['ΔWR_pp']:+.1f}pp  "
                  f"ΔPF={best['ΔPF']:+.2f}  ΔR={best['ΔR']:+.0f}")
        else:
            print(f"  {name:12s} NO guard level meets bar (PF≥0.9×base AND R>0)")


if __name__ == "__main__":
    main()
