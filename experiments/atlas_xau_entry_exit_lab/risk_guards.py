"""
PORTFOLIO RISK GUARDS — calibrated on DEV (<2025), confirmed on HOLDOUT (2025+).
Consumes streams_all.json (8 streams, per-trade entry/exit ts, netR@headline-spread,
ATR, direction). $ at live lot sizes (XAU 0.05 -> 5u, DJI 0.05, BTC 0.1).

A. DAILY PORTFOLIO STOP: once realized day P&L <= -X$, no NEW entries that day.
B. XAU SAME-DIRECTION CAP: max 2 concurrent same-direction XAU positions (of 3 streams).
C. TIME-OF-DAY: per-stream dev entry-hour P&L profile (candidate session blocks).
Metrics: total$, %positive days, max losing-day streak, maxDD$, worst day.
"""
import json, itertools
from pathlib import Path
import numpy as np, pandas as pd
OUT = Path(__file__).parent
DEV_END = pd.Timestamp("2025-01-01")
UNITS = {"xau": 5.0, "dji": 0.05, "btc": 0.1}

raw = json.load(open(OUT / "streams_all.json"))
trades = []
for name, blob in raw.items():
    sym = name.split("_")[0]
    for var, rows in blob["streams"].items():
        for et, xt, r, a, d in rows:
            trades.append(dict(stream=name, sym=sym, et=pd.Timestamp(et), xt=pd.Timestamp(xt),
                               usd=r * a * UNITS[sym], dir=int(d)))
T = pd.DataFrame(trades).sort_values("et").reset_index(drop=True)
print(f"{len(T):,} trades, {T.et.min().date()} -> {T.et.max().date()}")
print(T.groupby("stream")["usd"].agg(["count", "sum"]).round(0).to_string())

def metrics(df):
    if not len(df): return {}
    daily = df.set_index("xt")["usd"].resample("1D").sum()
    daily = daily[daily != 0]
    eq = daily.cumsum(); dd = float((eq.cummax() - eq).max())
    neg = (daily < 0).astype(int)
    streak = int(max((len(list(g)) for k, g in itertools.groupby(neg) if k == 1), default=0))
    return dict(total=float(daily.sum()), pos_days=float((daily > 0).mean() * 100),
                max_streak=streak, maxDD=dd, worst=float(daily.min()), ndays=len(daily))

def show(tag, m):
    print(f"  {tag:<28} total ${m['total']:>+9.0f} | +days {m['pos_days']:4.1f}% | "
          f"streak {m['max_streak']:>2} | maxDD ${m['maxDD']:>7.0f} | worst ${m['worst']:>+7.0f}")

def guard_daily_stop(df, X):
    """block NEW entries once realized day P&L <= -X. Realized at exit time."""
    keep = np.ones(len(df), bool)
    df = df.sort_values("et").reset_index(drop=True)
    exits = df[["xt", "usd"]].copy()
    for day, day_df in df.groupby(df.et.dt.date):
        idx = day_df.index
        # realized P&L (this day) before each entry
        dts = exits[(exits.xt.dt.date == day)]
        for i in idx:
            realized = float(dts[dts.xt <= df.at[i, "et"]]["usd"].sum())
            if realized <= -X: keep[i] = False
    return df[keep[df.index]]

def guard_xau_cap(df, cap=2):
    df = df.sort_values("et").reset_index(drop=True)
    keep = np.ones(len(df), bool); open_x = []   # (xt, dir)
    for i, row in df.iterrows():
        if row.sym != "xau": continue
        open_x = [(x, d) for x, d in open_x if x > row.et]
        same = sum(1 for x, d in open_x if d == row.dir)
        if same >= cap: keep[i] = False
        else: open_x.append((row.xt, row.dir))
    return df[keep]

for dev in (True, False):
    lab = "DEV (calibration)" if dev else "HOLDOUT 2025+ (confirmation)"
    D = T[(T.et < DEV_END) == dev]
    print(f"\n{'='*90}\n{lab}\n{'='*90}")
    show("baseline (8 streams)", metrics(D))
    for X in (75, 100, 150, 200, 300):
        show(f"A: daily stop -${X}", metrics(guard_daily_stop(D, X)))
    show("B: XAU same-dir cap 2", metrics(guard_xau_cap(D)))
    both = guard_xau_cap(guard_daily_stop(D, 150))
    show("A(-$150) + B", metrics(both))

print(f"\n{'='*90}\nC. TIME-OF-DAY — dev entry-hour P&L ($, per stream; negative blocks = candidates)\n{'='*90}")
D = T[T.et < DEV_END]
for name in sorted(T.stream.unique()):
    s = D[D.stream == name]
    if len(s) < 300: continue
    byh = s.groupby(s.et.dt.hour)["usd"].sum().round(0)
    worst = byh.nsmallest(4)
    print(f"  {name:<9} worst hours (UTC): " + "  ".join(f"{h:02d}h:{v:+.0f}" for h, v in worst.items())
          + f"   | total {byh.sum():+.0f}")
