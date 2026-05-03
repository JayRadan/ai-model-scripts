"""v8.9 — Port Quantum Volume-Flow Quantizer to Python + tie-breaker test.

Pine logic per bar:
  ha_close  = (O+H+L+C) / 4
  ha_open   = (O[1] + C[1]) / 2          # uses prior bar
  trend     = ha_close - ha_open
  vol_ratio = volume / SMA(volume, 50)   # avg-vol denominator
  raw       = trend × vol_ratio × 1000
  flow      = EMA(raw, 21)
  flow_q    = round(flow / (ATR×0.5)) × (ATR×0.5)   # quantized
  signal    = sign(flow_q) — buy when crosses above 0, sell below

MTF: same calc on 4H resampled bars, forward-filled to M5 grid.

Tie-breaker test on Jay's live trade window (April 27 → May 2):
  Earlier we found 5/5 days had K=5 cluster gap < 1.0 (ambiguous).
  For each of those days, classify the dominant direction the indicator
  was pointing, and check how many broker trades agreed.
"""
from __future__ import annotations
import os, sys
import pandas as pd
import numpy as np

ROOT = "/home/jay/Desktop/new-model-zigzag"


def quantum_flow(df: pd.DataFrame, lookback: int = 21,
                  vol_lookback: int = 50, use_quant: bool = True) -> pd.Series:
    """Port of Pine 'Quantum Volume-Flow Quantizer' to Python.
    Input df: time, open, high, low, close, volume (or spread as proxy)
    Returns: pd.Series of flow values (signed, EMA-smoothed, optionally
    ATR-quantized). Sign indicates regime direction."""
    o = df["open"].astype(float).values
    h = df["high"].astype(float).values
    l = df["low"].astype(float).values
    c = df["close"].astype(float).values
    # Use spread as volume proxy if no real volume column
    v_col = "volume" if "volume" in df.columns else "spread"
    v = df[v_col].astype(float).values

    ha_close = (o + h + l + c) / 4.0
    # ha_open uses prior bar's open + close
    ha_open = np.empty_like(ha_close)
    ha_open[0] = (o[0] + c[0]) / 2.0
    ha_open[1:] = (o[:-1] + c[:-1]) / 2.0

    trend = ha_close - ha_open
    avg_vol = pd.Series(v).rolling(vol_lookback, min_periods=1).mean().values
    vol_ratio = np.where(avg_vol != 0, v / avg_vol, 0.0)

    raw = trend * vol_ratio * 1000.0
    flow = pd.Series(raw).ewm(span=lookback, adjust=False).mean().values

    if use_quant:
        # ATR(14)
        tr = np.empty_like(c)
        tr[0] = h[0] - l[0]
        tr[1:] = np.maximum.reduce([h[1:] - l[1:],
                                       np.abs(h[1:] - c[:-1]),
                                       np.abs(l[1:] - c[:-1])])
        atr = pd.Series(tr).rolling(14, min_periods=1).mean().values
        step = atr * 0.5
        flow_q = np.where(step > 0, np.round(flow / step) * step, flow)
        return pd.Series(flow_q, index=df.index)
    return pd.Series(flow, index=df.index)


def quantum_flow_mtf(df_m5: pd.DataFrame, htf: str = "4h") -> pd.Series:
    """Compute flow on htf-resampled bars, forward-fill to M5 grid."""
    df = df_m5.set_index("time")
    df_htf = df[["open", "high", "low", "close", "spread"]].resample(htf).agg(
        {"open":"first","high":"max","low":"min","close":"last","spread":"sum"}
    ).dropna()
    df_htf = df_htf.reset_index()
    flow_htf = quantum_flow(df_htf)
    # Forward-fill to M5 grid using shift(1) to avoid lookahead
    s_htf = pd.Series(flow_htf.values, index=df_htf["time"]).shift(1)
    return s_htf.reindex(df_m5["time"], method="ffill").reset_index(drop=True)


def main():
    print("Loading XAU bars (live window)...", flush=True)
    xau = pd.read_csv("/tmp/swing_xau_live.csv", parse_dates=["time"],
                        date_format="%Y.%m.%d %H:%M")
    xau = xau[["time","open","high","low","close","spread"]].sort_values("time").reset_index(drop=True)
    print(f"  {len(xau)} bars  {xau.time.min()} → {xau.time.max()}", flush=True)

    # Compute 5m and 4h flows
    print("Computing 5m flow...", flush=True)
    flow_5m = quantum_flow(xau)
    print("Computing 4h MTF flow...", flush=True)
    flow_4h = quantum_flow_mtf(xau)
    xau["flow_5m"] = flow_5m.values
    xau["flow_4h"] = flow_4h.values

    # Distribution sanity
    print(f"\n5m flow:  min={flow_5m.min():.2f} med={flow_5m.median():.2f} max={flow_5m.max():.2f}")
    print(f"4h flow:  min={flow_4h.min():.2f} med={flow_4h.median():.2f} max={flow_4h.max():.2f}")

    # Live broker trades
    TRADES = [
        # (open_time, side, profit$)
        ("2026-04-27 01:10:03", "sell", -99.45),
        ("2026-04-27 05:05:01", "sell", -134.45),
        ("2026-04-27 07:35:01", "sell", 117.20),
        ("2026-04-27 13:10:02", "sell", 122.65),
        ("2026-04-27 14:05:06", "sell", 134.15),
        ("2026-04-27 18:35:07", "sell", -6.00),
        ("2026-04-27 18:55:08", "sell", -36.35),
        ("2026-04-27 19:05:06", "sell", -59.45),
        ("2026-04-27 19:25:06", "sell", -59.40),
        ("2026-04-28 01:55:07", "buy",  -51.15),
        ("2026-04-28 06:25:03", "sell", 226.50),
        ("2026-04-28 06:55:03", "sell", 210.70),
        ("2026-04-28 13:25:11", "sell", 206.00),
        ("2026-04-28 13:25:12", "sell", 206.10),
        ("2026-04-28 16:45:12", "sell", -67.50),
        ("2026-04-28 18:45:07", "sell", -80.35),
        ("2026-04-28 18:45:08", "sell", -88.40),
        ("2026-04-28 21:55:04", "buy",  15.20),
        ("2026-04-29 01:05:06", "sell", -50.00),
        ("2026-04-29 04:05:03", "sell", -115.45),
        ("2026-04-29 04:05:04", "sell", -114.90),
        ("2026-04-29 11:50:07", "buy",  -107.60),
        ("2026-04-29 14:10:07", "buy",  -95.75),
        ("2026-04-29 17:45:07", "sell", -9.45),
        ("2026-04-29 20:15:05", "sell", -18.95),
        ("2026-04-29 20:15:06", "sell", -12.40),
        ("2026-04-30 02:45:06", "sell", -47.95),
        ("2026-04-30 04:35:06", "sell", -6.80),
        ("2026-04-30 05:45:06", "sell", -103.80),
        ("2026-04-30 14:45:06", "buy",  -117.70),
        ("2026-04-30 16:55:06", "sell", 15.65),
        ("2026-04-30 19:20:03", "buy",  22.41),
        ("2026-04-30 19:25:06", "buy",  19.63),
        ("2026-04-30 20:20:06", "buy",  11.68),
        ("2026-04-30 20:20:06", "buy",  11.68),
        ("2026-04-30 21:00:06", "buy",  8.08),
        ("2026-04-30 21:00:06", "buy",  8.09),
        ("2026-04-30 21:20:09", "buy",  2.38),
        ("2026-04-30 21:40:06", "buy",  -0.25),
        ("2026-04-30 21:40:07", "buy",  -0.25),
        ("2026-04-30 21:45:06", "buy",  -0.21),
        ("2026-04-30 22:35:06", "buy",  0.59),
        ("2026-04-30 22:35:06", "buy",  0.60),
        ("2026-04-30 22:55:07", "buy",  3.45),
        ("2026-04-30 23:35:06", "buy",  10.58),
        ("2026-05-01 01:10:06", "buy",  -12.28),
        ("2026-05-01 01:10:06", "buy",  -12.28),
        ("2026-05-01 01:40:07", "buy",  -12.65),
        ("2026-05-01 01:40:07", "buy",  -12.65),
        ("2026-05-01 02:40:07", "buy",  -10.73),
        ("2026-05-01 02:55:06", "buy",  -9.81),
        ("2026-05-01 04:10:07", "buy",  -15.80),
        ("2026-05-01 05:05:05", "buy",  -13.93),
        ("2026-05-01 05:05:06", "buy",  -13.98),
        ("2026-05-01 05:10:06", "buy",  -13.41),
        ("2026-05-01 05:15:06", "buy",  -14.56),
        ("2026-05-01 06:10:06", "buy",  -16.39),
        ("2026-05-01 06:30:06", "buy",  -12.92),
        ("2026-05-01 06:30:07", "buy",  -12.92),
        ("2026-05-01 06:35:06", "buy",  -12.71),
        ("2026-05-01 07:45:06", "buy",  -14.44),
        ("2026-05-01 12:50:06", "sell", -14.68),
        ("2026-05-01 14:00:06", "sell", -15.06),
        ("2026-05-01 14:05:06", "sell", -13.77),
        ("2026-05-01 14:10:25", "sell", -13.63),
        ("2026-05-01 14:30:07", "sell", -15.56),
        ("2026-05-01 18:20:06", "buy",  -29.13),
        ("2026-05-01 18:45:07", "buy",  -27.39),
        ("2026-05-01 18:55:08", "buy",  -27.81),
        ("2026-05-01 19:40:07", "buy",  -25.38),
    ]

    # For each trade, look up flow_5m and flow_4h at the prior bar close
    # (the bar the EA would have decided on)
    print(f"\nReplaying {len(TRADES)} live trades against indicator...")
    rows = []
    xau_idx = pd.Series(xau.index.values, index=xau.time.values)
    for tt, side, profit in TRADES:
        t = pd.Timestamp(tt)
        # last bar with time < trade open
        prior_bars = xau[xau.time < t]
        if len(prior_bars) == 0: continue
        bar_idx = prior_bars.index[-1]
        f5 = xau.flow_5m.iloc[bar_idx]
        f4 = xau.flow_4h.iloc[bar_idx]
        # Direction agreement
        side_int = 1 if side == "buy" else -1
        agree_5m = (np.sign(f5) == side_int)
        agree_4h = (np.sign(f4) == side_int) if not pd.isna(f4) else None
        rows.append({
            "time": t, "side": side, "profit": profit,
            "flow_5m": float(f5), "flow_4h": float(f4) if not pd.isna(f4) else np.nan,
            "agree_5m": agree_5m,
            "agree_4h": agree_4h,
            "both_agree": (agree_5m and agree_4h is True),
            "any_agree":  (agree_5m or agree_4h is True),
        })

    df = pd.DataFrame(rows)
    print(f"\nDistribution of agreement vs broker side:")
    print(f"  5m agrees:        {df.agree_5m.sum()}/{len(df)} ({df.agree_5m.mean():.0%})")
    print(f"  4h agrees:        {df.agree_4h.sum()}/{len(df)} ({df.agree_4h.mean():.0%})")
    print(f"  both agree:       {df.both_agree.sum()}/{len(df)} ({df.both_agree.mean():.0%})")
    print(f"  either agrees:    {df.any_agree.sum()}/{len(df)} ({df.any_agree.mean():.0%})")

    # Filter test: keep only trades where both timeframes agree with direction
    print(f"\n=== FILTER TEST: keep only trades where indicator direction agrees ===")
    print(f"\n{'filter':<25s} {'n_kept':>7} {'n_won':>6} {'WR':>6} {'total_$':>9} {'avg_$':>7}")
    base = df.copy()
    for name, mask in [
        ("baseline (no filter)", np.ones(len(df), dtype=bool)),
        ("5m agrees",              df.agree_5m.values),
        ("4h agrees",              (df.agree_4h == True).values),
        ("BOTH agree",             df.both_agree.values),
        ("EITHER agrees",          df.any_agree.values),
    ]:
        sub = df[mask]
        if len(sub) == 0:
            print(f"{name:<25s} {0:>7} - - - -")
            continue
        n = len(sub); wins = (sub.profit > 0).sum(); wr = wins / n
        tot = sub.profit.sum(); avg = sub.profit.mean()
        print(f"{name:<25s} {n:>7} {wins:>6} {wr*100:>5.1f}% {tot:>+9.2f} {avg:>+7.2f}")

    # Save detailed
    out_dir = os.path.dirname(__file__)
    df.to_csv(os.path.join(out_dir, "trades_with_flow.csv"), index=False)
    print(f"\nDetailed: trades_with_flow.csv")


if __name__ == "__main__":
    main()
