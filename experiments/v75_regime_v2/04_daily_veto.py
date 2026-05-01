"""
v75 step 04: daily-timeframe veto experiment.

Hypothesis: filter every M5 trade by the higher-timeframe (daily) regime.
Block longs on Down days, block shorts on Up days, allow both on Neutral.

Daily regime rule (simple, no training):
  - Aggregate M5 → daily OHLC
  - Compute 5-day SMA and 5-day return
  - As of EVERY M5 bar's prior day's close:
      Up      = close > 5d_SMA AND 5d_return > +UP_THR
      Down    = close < 5d_SMA AND 5d_return < -DOWN_THR
      Neutral = otherwise
  - "Prior day's close" is strictly past data — no look-ahead.

Apply to:
  • Oracle XAU original holdout trade dump (1,367 trades, post-2024-12-12)
  • Midas XAU original holdout trade dump (2,636 trades)

Compare:
  Baseline (no veto)        : the validated PF
  + veto (block conflict)   : drop trades where daily disagrees with M5
"""
from __future__ import annotations
import os, sys
import numpy as np, pandas as pd

ZIGZAG = "/home/jay/Desktop/new-model-zigzag"
sys.path.insert(0, os.path.join(ZIGZAG, "model_pipeline"))
import paths as P

ORACLE_HOLDOUT = P.data("v72l_trades_holdout.csv")
MIDAS_HOLDOUT  = P.data("v6_trades_holdout_xau.csv")
SWING          = P.data("swing_v5_xauusd.csv")

# Daily-regime thresholds — picked conservative so most days are "Neutral"
# and only clear directional days override the M5 system.
SMA_LEN     = 5            # 5-day SMA
RET_LOOK    = 5            # last 5 daily bars
# Two threshold tiers — sweep both to find any setting where the veto helps.
THRESHOLD_GRID = [
    ("loose ±0.5%",   +0.005, -0.005),
    ("normal ±1.0%",  +0.010, -0.010),
    ("strict ±2.0%",  +0.020, -0.020),
    ("very-strict ±3%",+0.030, -0.030),
]


def build_daily_regime(up_thr, down_thr):
    df = pd.read_csv(SWING, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)
    df["date"] = df["time"].dt.normalize()
    daily = df.groupby("date").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"),     close=("close", "last")).reset_index()
    daily["sma"] = daily["close"].rolling(SMA_LEN, min_periods=SMA_LEN).mean()
    daily["ret_5d"] = daily["close"].pct_change(RET_LOOK)
    def cls(r):
        if pd.isna(r["sma"]) or pd.isna(r["ret_5d"]): return "Warmup"
        if r["close"] > r["sma"] and r["ret_5d"] >  up_thr:   return "Up"
        if r["close"] < r["sma"] and r["ret_5d"] < down_thr:  return "Down"
        return "Neutral"
    daily["regime"] = daily.apply(cls, axis=1)
    daily["regime_lagged"] = daily["regime"].shift(1)
    return daily[["date", "regime_lagged"]].rename(columns={"regime_lagged": "daily_regime"})


def apply_veto(trades, daily, name):
    trades = trades.copy()
    if "pnl_R" in trades.columns and trades["pnl_R"].abs().mean() > 2:
        trades["pnl_R"] = trades["pnl_R"] / 4.0   # rescale R-per-ATR → R-per-SL
    trades["date"] = pd.to_datetime(trades["time"]).dt.normalize()
    trades = trades.merge(daily, on="date", how="left")
    trades["daily_regime"] = trades["daily_regime"].fillna("Warmup")
    trades["dir_str"] = np.where(trades["direction"] > 0, "BUY", "SELL")

    # Veto rules
    veto_long_in_down  = (trades["dir_str"] == "BUY")  & (trades["daily_regime"] == "Down")
    veto_short_in_up   = (trades["dir_str"] == "SELL") & (trades["daily_regime"] == "Up")
    trades["vetoed"] = veto_long_in_down | veto_short_in_up

    survivors = trades[~trades["vetoed"]].reset_index(drop=True)
    blocked   = trades[ trades["vetoed"]].reset_index(drop=True)

    def stats(df, label):
        if not len(df): return f"{label:<28s} n=0"
        ww = df[df.pnl_R > 0]; ll = df[df.pnl_R <= 0]
        pf = (ww.pnl_R.sum() / -ll.pnl_R.sum()) if len(ll) and ll.pnl_R.sum() < 0 else float("inf")
        cum = df.pnl_R.cumsum().values
        dd = float((cum - np.maximum.accumulate(cum)).min())
        return (f"{label:<32s} n={len(df):>4d}  WR={len(ww)/len(df)*100:>5.1f}%  "
                f"PF={pf:>5.2f}  R={df.pnl_R.sum():+8.1f}  DD={dd:+7.1f}")

    print(f"\n=== {name} ===")
    print(f"  {stats(trades,    '(A) baseline (all trades)')}")
    print(f"  {stats(survivors, '(B) + daily veto')}")
    print(f"  {stats(blocked,   '    blocked by veto')}")
    n_blocked = len(blocked)
    blocked_pnl = blocked.pnl_R.sum() if n_blocked else 0
    blocked_wr = (blocked.pnl_R > 0).mean() * 100 if n_blocked else 0
    print(f"  veto blocked {n_blocked} trades  net_R={blocked_pnl:+.1f}  WR={blocked_wr:.1f}%  → "
          f"{('GOOD: removed losers' if blocked_pnl < 0 else 'BAD: removed winners')}")

    # Spot-check today's window
    recent = trades[trades["date"] >= pd.Timestamp("2026-04-29")]
    print(f"\n  --- 2026-04-29..present ({len(recent)} trades) ---")
    if len(recent):
        print(f"  daily regimes seen: {recent['daily_regime'].value_counts().to_dict()}")
        print(f"  vetoed: {recent['vetoed'].sum()} of {len(recent)}")
        for _, t in recent.head(15).iterrows():
            mark = "VETO" if t["vetoed"] else "    "
            print(f"  {t['time']}  {t['dir_str']:<4s} {t['rule']:<22s}  daily={t['daily_regime']:<8s}  "
                  f"pnl={t['pnl_R']:+.2f}R  {mark}")

    return survivors, blocked


def main():
    o = pd.read_csv(ORACLE_HOLDOUT, parse_dates=["time"])
    m = pd.read_csv(MIDAS_HOLDOUT,  parse_dates=["time"])
    for label, up, down in THRESHOLD_GRID:
        print(f"\n############ {label}  (UP_THR={up:+.3f}  DOWN_THR={down:+.3f}) ############")
        daily = build_daily_regime(up, down)
        print(f"  daily regime distribution: {daily['daily_regime'].value_counts().to_dict()}")
        apply_veto(o, daily, "Oracle XAU")
        apply_veto(m, daily, "Midas XAU")


if __name__ == "__main__":
    main()
