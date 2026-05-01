"""
v8.1 — Diagnose "late entry on extended legs" hypothesis.

Hypothesis (from Jay): losers cluster at points where price has
already moved a lot in the trade's direction — we buy at the top of
an exhausted up-leg or sell at the bottom of an exhausted down-leg.

Method:
  For each holdout trade, look up swing_v5 at fire time and compute:
    dist_from_30bar_extreme_atr  — for BUY: (close - low_of_30) / ATR
                                    for SELL: (high_of_30 - close) / ATR
                                    (always positive, larger = more extended)
    leg_run_30bar_atr            — for BUY: (close - close[-30]) / ATR
                                    for SELL: (close[-30] - close) / ATR
                                    (signed in trade direction)
    dist_from_sma50_atr          — for BUY: (close - sma50) / ATR
                                    for SELL: (sma50 - close) / ATR
    rsi14                        — already computed in swing_v5

Compare WIN vs LOSE distributions on each metric. If the gap is real
(losers' median materially higher AND p<0.05), we have a filter signal.
Then walk-forward H1→H2 to confirm.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

ROOT = "/home/jay/Desktop/new-model-zigzag"
HOLDOUT_START = pd.Timestamp("2024-12-12")


def pf(rs):
    pos = rs[rs > 0].sum(); neg = -rs[rs <= 0].sum()
    return pos / max(neg, 1e-9)


def attach_extension_features(trades, swing_path):
    swing = pd.read_csv(swing_path, parse_dates=["time"])
    swing = swing.sort_values("time").reset_index(drop=True)
    H = swing["high"].values; L = swing["low"].values; C = swing["close"].values
    # ATR(14) — same definition as in training
    tr = np.concatenate([[H[0]-L[0]],
          np.maximum.reduce([H[1:]-L[1:], np.abs(H[1:]-C[:-1]), np.abs(L[1:]-C[:-1])])])
    atr = pd.Series(tr).rolling(14, min_periods=14).mean().values
    sma50 = pd.Series(C).rolling(50, min_periods=50).mean().values
    # 30-bar high/low and 30-bar prior close
    high30 = pd.Series(H).rolling(30, min_periods=30).max().values
    low30  = pd.Series(L).rolling(30, min_periods=30).min().values
    prev30 = pd.Series(C).shift(30).values

    swing["atr"]    = atr
    swing["sma50"]  = sma50
    swing["high30"] = high30
    swing["low30"]  = low30
    swing["prev30"] = prev30

    out = trades.sort_values("time").reset_index(drop=True).copy()
    out = pd.merge_asof(out, swing[["time", "close", "atr", "sma50",
                                      "high30", "low30", "prev30",
                                      "h1_rsi14", "h4_rsi14"]
                                    ].rename(columns={"close": "entry_close"}),
                          on="time", direction="backward",
                          tolerance=pd.Timedelta("10min"))
    out = out.dropna(subset=["entry_close", "atr", "sma50", "high30",
                              "low30", "prev30"]).reset_index(drop=True)

    d = out["direction"].values
    c = out["entry_close"].values; a = out["atr"].values
    out["dist_extreme_atr"] = np.where(d == 1,
                                         (c - out["low30"].values) / a,
                                         (out["high30"].values - c) / a)
    out["leg_run_atr"]      = np.where(d == 1,
                                         (c - out["prev30"].values) / a,
                                         (out["prev30"].values - c) / a)
    out["dist_sma50_atr"]   = np.where(d == 1,
                                         (c - out["sma50"].values) / a,
                                         (out["sma50"].values - c) / a)
    return out


def compare(out, name):
    w = out[out["pnl_R"] > 0]
    l = out[out["pnl_R"] <= 0]
    print(f"\n[{name}] WIN n={len(w)}  LOSE n={len(l)}")
    print(f"{'metric':<25}  WIN_med   LOSE_med   gap   MWU_p")
    for m in ["dist_extreme_atr", "leg_run_atr", "dist_sma50_atr",
               "h1_rsi14", "h4_rsi14"]:
        wv = w[m].dropna().values; lv = l[m].dropna().values
        if len(wv) < 20 or len(lv) < 20: continue
        wm = np.median(wv); lm = np.median(lv)
        try:
            _, p = mannwhitneyu(wv, lv, alternative="two-sided")
        except Exception:
            p = float("nan")
        flag = " ⚠" if p < 0.05 else ""
        print(f"{m:<25}  {wm:>+7.3f}  {lm:>+7.3f}   {lm-wm:+.3f}  {p:.4f}{flag}")


def kill_sweep(out, name, metric):
    """For each percentile threshold on `metric`, skip trades above it.
    Report (kept_n, kept_WR, kept_PF, kept_R, ΔWR, ΔR)."""
    print(f"\n  Kill sweep on {metric} (above each percentile cutoff):")
    rs0 = out["pnl_R"].values; base_wr = (rs0 > 0).mean()
    base_pf = pf(rs0); base_r = rs0.sum()
    print(f"  baseline n={len(rs0)}  WR={base_wr:.1%}  PF={base_pf:.2f}  R={base_r:+.1f}")
    pcts = [99, 97, 95, 90, 85, 80, 75, 70]
    for pct in pcts:
        thr = np.percentile(out[metric].dropna(), pct)
        keep = out[metric].values <= thr
        if keep.sum() < 50: continue
        rs = out.loc[keep, "pnl_R"].values
        wr = (rs > 0).mean(); rR = rs.sum()
        print(f"   p{pct:>2} ({metric}≤{thr:+.2f})  kept={keep.sum():>4}"
              f"  WR={wr:.1%}  PF={pf(rs):.2f}  R={rR:+.1f}"
              f"  ΔWR={(wr-base_wr)*100:+.1f}pp"
              f"  ΔPF={pf(rs)-base_pf:+.2f}"
              f"  ΔR={rR-base_r:+.1f}"
              f"  Δn={keep.sum()-len(rs0):+d}")


def walk_forward(out, name, metric, pct):
    out = out.sort_values("time").reset_index(drop=True)
    h1 = out.iloc[:len(out)//2]; h2 = out.iloc[len(out)//2:]
    thr = np.percentile(h1[metric].dropna(), pct)
    keep = h2[metric].values <= thr
    rs0 = h2["pnl_R"].values; rs1 = h2.loc[keep, "pnl_R"].values
    base_wr = (rs0 > 0).mean(); base_pf = pf(rs0); base_r = rs0.sum()
    new_wr = (rs1 > 0).mean(); new_pf = pf(rs1); new_r = rs1.sum()
    print(f"\n  WALK-FORWARD ({metric} > p{pct} of H1 = {thr:+.2f}):")
    print(f"    H2 base: n={len(rs0)} WR={base_wr:.1%} PF={base_pf:.2f} R={base_r:+.1f}")
    print(f"    H2 kill: n={len(rs1)} WR={new_wr:.1%} PF={new_pf:.2f} R={new_r:+.1f}"
          f"  ΔWR={(new_wr-base_wr)*100:+.1f}pp ΔPF={new_pf-base_pf:+.2f} ΔR={new_r-base_r:+.1f}")


def main():
    for name, trades_csv, swing_csv in [
        ("Oracle XAU", "data/v72l_trades_holdout.csv", "data/swing_v5_xauusd.csv"),
        ("Midas XAU",  "data/v6_trades_holdout_xau.csv", "data/swing_v5_xauusd.csv"),
        ("Oracle BTC", "data/v72l_trades_holdout_btc.csv", "data/swing_v5_btc.csv"),
    ]:
        print("="*78)
        print(f"=== {name} ===")
        trades = pd.read_csv(os.path.join(ROOT, trades_csv), parse_dates=["time"])
        out = attach_extension_features(trades, os.path.join(ROOT, swing_csv))
        compare(out, name)
        # Sweep on the most promising single metric
        kill_sweep(out, name, "dist_extreme_atr")
        kill_sweep(out, name, "leg_run_atr")
        kill_sweep(out, name, "dist_sma50_atr")
        # Walk-forward on the metric that looked best (we'll rerun by hand
        # if needed; default to dist_extreme_atr at p90).
        walk_forward(out, name, "dist_extreme_atr", 90)
        walk_forward(out, name, "leg_run_atr", 90)


if __name__ == "__main__":
    main()
