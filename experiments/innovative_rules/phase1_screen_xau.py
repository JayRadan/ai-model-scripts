"""
Phase 1: Raw base-rate screen of 14 innovative XAU rules.

Goal: find which innovative rule ideas have inherent positive edge BEFORE
investing in per-rule ML confirmation training. No classifier, no features
beyond OHLC+tickvol — pure signal detection + forward-outcome labeling.

Rules tested:
   1. Absorption
   2. Iceberg trap
   3. Delta divergence
   4. Liquidity sweep reversal
   5. Fair value gap fill
   6. Order block retest
   7. Return z-score tail (mean revert)
   8. Vol-of-vol compression → expansion
   9. Autocorrelation regime flip
  10. Asian range break
  11. Hourly seasonality (skipped — needs cluster context, revisit in phase 2)
  12. Friday close mean-revert
  13. M5/H1 divergence
  14. H4-trend-gated breakout

Outputs:
  - phase1_results_xau.csv   (per-rule stats)
  - phase1_results_xau.png   (bar chart)
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/jay/Desktop/new-model-zigzag"
OUT_DIR = f"{ROOT}/experiments/innovative_rules"
os.makedirs(OUT_DIR, exist_ok=True)

SWING = f"{ROOT}/data/swing_v5_xauusd.csv"
TP_MULT = 6.0
SL_MULT = 2.0
MAX_FWD = 40


# ───────────────────────── Helpers ─────────────────────────
def load_data():
    df = pd.read_csv(SWING, parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    c, h, l, o = df.close.values, df.high.values, df.low.values, df.open.values
    # ATR14 using rolling mean TR
    tr = np.concatenate([[h[0]-l[0]],
          np.maximum.reduce([h[1:]-l[1:], np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])])])
    atr14 = pd.Series(tr).rolling(14, min_periods=14).mean().values
    # SMAs / EMAs
    sma20 = pd.Series(c).rolling(20, min_periods=20).mean().values
    sma50 = pd.Series(c).rolling(50, min_periods=50).mean().values
    ema12 = pd.Series(c).ewm(span=12, adjust=False).mean().values
    # RSI(14)
    delta = np.diff(c, prepend=c[0])
    gain = np.where(delta>0, delta, 0.0); loss = np.where(delta<0, -delta, 0.0)
    rsi = 100 - 100/(1 + pd.Series(gain).ewm(span=14,adjust=False).mean().values /
                     (pd.Series(loss).ewm(span=14,adjust=False).mean().values + 1e-12))
    # Returns
    ret = np.concatenate([[0.0], np.diff(c)/c[:-1]])
    # Tick volume (stored in swing CSV? check)
    tv = df["spread"].values if "tick_volume" not in df.columns else df["tick_volume"].values
    # time components
    hr = df["time"].dt.hour.values
    dow = df["time"].dt.dayofweek.values
    # HTF H1 / H4 resample
    df_h1 = df.set_index("time")[["open","high","low","close"]].resample("1h").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
    df_h4 = df.set_index("time")[["open","high","low","close"]].resample("4h").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
    h1_sma20 = df_h1["close"].rolling(20, min_periods=20).mean()
    h1_rsi = 100 - 100/(1 + df_h1["close"].diff().clip(lower=0).ewm(span=14, adjust=False).mean() /
                           (-df_h1["close"].diff().clip(upper=0).ewm(span=14, adjust=False).mean() + 1e-12))
    h4_sma20 = df_h4["close"].rolling(20, min_periods=20).mean()
    h4_slope = (df_h4["close"] - df_h4["close"].shift(5))
    # CRITICAL: shift HTF by 1 bar so M5 bar at time t sees only COMPLETED H1/H4 bars.
    # Without this shift, H1 value at M5 time 10:15 would include future M5 data 10:15-10:55.
    h1_sma20 = h1_sma20.shift(1)
    h1_rsi   = h1_rsi.shift(1)
    h4_sma20 = h4_sma20.shift(1)
    h4_slope = h4_slope.shift(1)
    m5_idx = df["time"]
    h1_sma20_m5 = h1_sma20.reindex(m5_idx, method="ffill").values
    h1_rsi_m5 = h1_rsi.reindex(m5_idx, method="ffill").values
    h4_sma20_m5 = h4_sma20.reindex(m5_idx, method="ffill").values
    h4_slope_m5 = h4_slope.reindex(m5_idx, method="ffill").values
    # Vectorized rank features (50-bar rolling percentile of bar range, tick vol)
    ranges = h - l
    rng_rank = pd.Series(ranges).rolling(50, min_periods=50).rank(pct=True).values
    tv_rank = pd.Series(tv).rolling(50, min_periods=50).rank(pct=True).values
    # Vectorized 500-bar return percentiles
    ret_p2  = pd.Series(ret).rolling(500, min_periods=500).quantile(0.02).values
    ret_p98 = pd.Series(ret).rolling(500, min_periods=500).quantile(0.98).values
    # Vol-of-vol stack
    rng5 = pd.Series(ranges).rolling(5, min_periods=5).mean().values
    vov = pd.Series(rng5).rolling(20, min_periods=20).std().values
    vov_p10 = pd.Series(vov).rolling(100, min_periods=60).quantile(0.10).values
    # Weekly return (1440 bars back) + its rolling std
    week_ret = pd.Series(c).pct_change(1440).values
    week_sigma = pd.Series(week_ret).rolling(2000, min_periods=500).std().values
    # 20-bar rolling highs/lows (shifted by 1 so current bar excluded)
    hi20 = pd.Series(h).rolling(20, min_periods=20).max().shift(1).values
    lo20 = pd.Series(l).rolling(20, min_periods=20).min().shift(1).values
    # Per-day asian session hi/lo forward-filled to every M5 bar
    date = df["time"].dt.date
    asian_mask = (df["time"].dt.hour < 7)
    a_hi = df[["high"]].assign(date=date, _m=asian_mask)
    a_hi = a_hi.where(a_hi["_m"]).groupby(a_hi["date"])["high"].cummax().ffill().values
    a_lo = df[["low"]].assign(date=date, _m=asian_mask)
    a_lo = a_lo.where(a_lo["_m"]).groupby(a_lo["date"])["low"].cummin().ffill().values
    tv_med50 = pd.Series(tv).rolling(50, min_periods=50).median().values
    # Rolling lag-1 return autocorrelation (50-bar)
    r_s = pd.Series(ret)
    ac_now = r_s.rolling(50, min_periods=50).corr(r_s.shift(1)).values
    ac_prev = pd.Series(ac_now).shift(5).values
    return dict(
        df=df, c=c, h=h, l=l, o=o, atr=atr14, sma20=sma20, sma50=sma50, ema12=ema12,
        rsi=rsi, ret=ret, tv=tv, hr=hr, dow=dow,
        h1_sma20=h1_sma20_m5, h1_rsi=h1_rsi_m5,
        h4_sma20=h4_sma20_m5, h4_slope=h4_slope_m5,
        ranges=ranges, rng_rank=rng_rank, tv_rank=tv_rank,
        ret_p2=ret_p2, ret_p98=ret_p98,
        rng5=rng5, vov=vov, vov_p10=vov_p10,
        week_ret=week_ret, week_sigma=week_sigma,
        hi20=hi20, lo20=lo20,
        asian_hi=a_hi, asian_lo=a_lo, tv_med50=tv_med50,
        ac_now=ac_now, ac_prev=ac_prev,
    )


def forward_outcome(d, i, direction):
    """Return +1 if TP first, 0 if SL first, 2 if neither in MAX_FWD bars."""
    atr = d["atr"][i]
    if not np.isfinite(atr) or atr <= 0: return -1
    entry = d["c"][i]
    if direction == 1:
        tp = entry + TP_MULT * atr
        sl = entry - SL_MULT * atr
        for k in range(i+1, min(i+1+MAX_FWD, len(d["c"]))):
            if d["l"][k] <= sl: return 0
            if d["h"][k] >= tp: return 1
        return 2
    else:
        tp = entry - TP_MULT * atr
        sl = entry + SL_MULT * atr
        for k in range(i+1, min(i+1+MAX_FWD, len(d["c"]))):
            if d["h"][k] >= sl: return 0
            if d["l"][k] <= tp: return 1
        return 2


# ───────────────────────── Rules ─────────────────────────

def rule_1_absorption(d, i):
    """Large range, small body, high tick volume → fade in wick direction."""
    if i < 50: return 0
    rng = d["ranges"][i]
    if rng <= 0: return 0
    body = abs(d["c"][i] - d["o"][i])
    rng_rank = d["rng_rank"][i]
    tv_rank  = d["tv_rank"][i]
    if not (np.isfinite(rng_rank) and np.isfinite(tv_rank)): return 0
    body_pct = body / rng
    if rng_rank > 0.80 and body_pct < 0.30 and tv_rank > 0.75:
        lower_wick = min(d["o"][i], d["c"][i]) - d["l"][i]
        upper_wick = d["h"][i] - max(d["o"][i], d["c"][i])
        return 1 if lower_wick > upper_wick else -1
    return 0

def rule_2_iceberg_trap(d, i):
    """Breakout of 20-bar high/low on thin vol, closes back within 2 bars."""
    if i < 52: return 0
    hi20, lo20 = d["hi20"][i-1], d["lo20"][i-1]
    tv_med = d["tv_med50"][i-2]
    if not all(np.isfinite(x) for x in (hi20, lo20, tv_med)): return 0
    broke_up = d["h"][i-2] > hi20 and d["tv"][i-2] < 0.7 * tv_med
    broke_dn = d["l"][i-2] < lo20 and d["tv"][i-2] < 0.7 * tv_med
    if broke_up and d["c"][i] < hi20: return -1
    if broke_dn and d["c"][i] > lo20: return +1
    return 0

def rule_3_delta_divergence(d, i):
    """5-bar delta cum vs price direction mismatch."""
    if i < 10: return 0
    delta = (d["c"][i-4:i+1] - d["o"][i-4:i+1]) * d["tv"][i-4:i+1]
    delta_sum = delta.sum()
    price_move = d["c"][i] - d["c"][i-5]
    if delta_sum > 0 and price_move < 0: return +1
    if delta_sum < 0 and price_move > 0: return -1
    return 0

def rule_4_liquidity_sweep(d, i):
    """Take out 20-bar swing by >=0.3 ATR, close back inside within 2 bars."""
    atr = d["atr"][i]
    if not np.isfinite(atr) or atr <= 0: return 0
    hi20, lo20 = d["hi20"][i-1], d["lo20"][i-1]  # 20-bar window ending at i-2
    if not (np.isfinite(hi20) and np.isfinite(lo20)): return 0
    swept_up = d["h"][i-1] > hi20 + 0.3*atr
    swept_dn = d["l"][i-1] < lo20 - 0.3*atr
    if swept_up and d["c"][i] < hi20: return -1
    if swept_dn and d["c"][i] > lo20: return +1
    return 0

def rule_5_fvg_fill(d, i):
    """3-bar FVG created 5-15 bars ago, price revisits midpoint now."""
    if i < 20: return 0
    # Look back 5-15 bars for bullish FVG: high[k-2] < low[k]
    for k in range(i-15, i-5):
        if k < 2: continue
        # Bullish FVG: high at k-2 < low at k, and gap not yet filled (check prior)
        if d["h"][k-2] < d["l"][k]:
            gap_mid = (d["h"][k-2] + d["l"][k]) / 2
            # Not filled since
            filled = False
            for j in range(k+1, i):
                if d["l"][j] <= gap_mid:
                    filled = True; break
            if filled: continue
            # Current bar touching gap mid from above?
            if d["l"][i] <= gap_mid and d["c"][i] > gap_mid:
                return +1
        # Bearish FVG
        if d["l"][k-2] > d["h"][k]:
            gap_mid = (d["l"][k-2] + d["h"][k]) / 2
            filled = False
            for j in range(k+1, i):
                if d["h"][j] >= gap_mid:
                    filled = True; break
            if filled: continue
            if d["h"][i] >= gap_mid and d["c"][i] < gap_mid:
                return -1
    return 0

def rule_6_order_block(d, i):
    """Last down-bar before +2 ATR impulse up acts as support on retest."""
    if i < 30: return 0
    atr = d["atr"][i]
    if not np.isfinite(atr) or atr <= 0: return 0
    # Look for an impulsive move in last 5–20 bars
    for k in range(i-20, i-5):
        if k < 2: continue
        impulse = d["c"][k] - d["o"][k]
        if impulse > 2.0 * atr and d["c"][k-1] < d["o"][k-1]:  # bearish bar before bullish impulse
            ob_hi = d["h"][k-1]; ob_lo = d["l"][k-1]
            # Not revisited since
            revisited = any(d["l"][j] < ob_hi for j in range(k+1, i))
            if revisited: continue
            # Current bar retests OB zone from above
            if d["l"][i] <= ob_hi and d["c"][i] > ob_lo and d["c"][i] > d["o"][i]:
                return +1
        if impulse < -2.0 * atr and d["c"][k-1] > d["o"][k-1]:
            ob_hi = d["h"][k-1]; ob_lo = d["l"][k-1]
            revisited = any(d["h"][j] > ob_lo for j in range(k+1, i))
            if revisited: continue
            if d["h"][i] >= ob_lo and d["c"][i] < ob_hi and d["c"][i] < d["o"][i]:
                return -1
    return 0

def rule_7_return_tail(d, i):
    """1-bar return below 2nd / above 98th percentile of 500-bar window → fade."""
    p2, p98 = d["ret_p2"][i], d["ret_p98"][i]
    if not (np.isfinite(p2) and np.isfinite(p98)): return 0
    r = d["ret"][i]
    if r <= p2:  return +1
    if r >= p98: return -1
    return 0

def rule_8_vov_compression(d, i):
    """5-bar range's 20-bar std drops below 10th percentile, current bar expands."""
    p10 = d["vov_p10"][i]; vov_prev = d["vov"][i-1]; rng5_prev = d["rng5"][i-1]
    if not all(np.isfinite(x) for x in (p10, vov_prev, rng5_prev)): return 0
    if vov_prev < p10 and d["ranges"][i] > 1.5 * rng5_prev:
        return +1 if d["c"][i] > d["o"][i] else -1
    return 0

def rule_9_autocorr_flip(d, i):
    """Rolling 50-bar lag-1 autocorr of returns changed sign from + to −."""
    if i < 100: return 0
    ac_now = d["ac_now"][i]; ac_prev = d["ac_prev"][i]
    if not (np.isfinite(ac_now) and np.isfinite(ac_prev)): return 0
    if ac_prev > 0.15 and ac_now < -0.05:
        if d["c"][i] > d["c"][i-10]: return -1
        if d["c"][i] < d["c"][i-10]: return +1
    return 0

def rule_10_asian_break(d, i):
    """First London-hour (7-9 UTC) M5 close beyond Asian (0-6 UTC) range."""
    if d["hr"][i] not in (7, 8, 9): return 0
    a_hi = d["asian_hi"][i]; a_lo = d["asian_lo"][i]
    if not (np.isfinite(a_hi) and np.isfinite(a_lo)): return 0
    if d["c"][i] > a_hi and d["c"][i-1] <= a_hi: return +1
    if d["c"][i] < a_lo and d["c"][i-1] >= a_lo: return -1
    return 0

def rule_12_friday_close(d, i):
    """Fri 17-19 UTC: if weekly return is >1.5σ extreme, fade toward mean."""
    if d["dow"][i] != 4 or d["hr"][i] not in (17, 18, 19): return 0
    wr = d["week_ret"][i]; sigma = d["week_sigma"][i]
    if not (np.isfinite(wr) and np.isfinite(sigma)) or sigma < 1e-10: return 0
    z = wr / sigma
    if z > 1.5:  return -1
    if z < -1.5: return +1
    return 0

def rule_13_mh1_divergence(d, i):
    """M5 makes higher high vs 20-bar but H1 RSI14 fails to make higher high."""
    if i < 40: return 0
    m5_hh_now = d["h"][i] > d["h"][i-20:i].max()
    m5_ll_now = d["l"][i] < d["l"][i-20:i].min()
    if not np.isfinite(d["h1_rsi"][i]): return 0
    h1_rsi_now = d["h1_rsi"][i]
    h1_rsi_prev = d["h1_rsi"][max(0, i-24)]  # 2 H1 bars back on M5 scale
    if m5_hh_now and h1_rsi_now < h1_rsi_prev - 5: return -1
    if m5_ll_now and h1_rsi_now > h1_rsi_prev + 5: return +1
    return 0

def rule_14_h4_gated_breakout(d, i):
    """M5 break of 20-bar high/low, taken only if H4 SMA20 slope agrees."""
    if i < 22: return 0
    if not np.isfinite(d["h4_slope"][i]): return 0
    hi20 = d["h"][i-21:i-1].max(); lo20 = d["l"][i-21:i-1].min()
    broke_up = d["c"][i] > hi20 and d["c"][i-1] <= hi20
    broke_dn = d["c"][i] < lo20 and d["c"][i-1] >= lo20
    if broke_up and d["h4_slope"][i] > 0: return +1
    if broke_dn and d["h4_slope"][i] < 0: return -1
    return 0


RULES = [
    ("01_absorption",          rule_1_absorption),
    ("02_iceberg_trap",        rule_2_iceberg_trap),
    ("03_delta_divergence",    rule_3_delta_divergence),
    ("04_liquidity_sweep",     rule_4_liquidity_sweep),
    ("05_fvg_fill",            rule_5_fvg_fill),
    ("06_order_block",         rule_6_order_block),
    ("07_return_z_tail",       rule_7_return_tail),
    ("08_vov_compression",     rule_8_vov_compression),
    ("09_autocorr_flip",       rule_9_autocorr_flip),
    ("10_asian_break",         rule_10_asian_break),
    ("12_friday_close_fade",   rule_12_friday_close),
    ("13_m5_h1_divergence",    rule_13_mh1_divergence),
    ("14_h4_gated_breakout",   rule_14_h4_gated_breakout),
]


def main():
    print("Loading XAU data…", flush=True)
    d = load_data()
    n = len(d["c"])
    # Last ~1.5 years of M5 for phase 1 screen (keep runtime reasonable)
    start = max(600, n - 150_000)
    print(f"Scanning bars {start:,} to {n:,} ({(n-start):,} bars)", flush=True)

    results = []
    for rname, rfn in RULES:
        import time as _t
        t0 = _t.time()
        print(f"  {rname}: scanning...", flush=True)
        signals = []
        step_report = max(1, (n - start) // 10)
        last_report = start
        for i in range(start, n - MAX_FWD - 1):
            if i - last_report >= step_report:
                last_report = i
                sys.stdout.write("."); sys.stdout.flush()
            s = rfn(d, i)
            if s != 0:
                out = forward_outcome(d, i, s)
                if out in (0, 1):
                    signals.append((i, s, out))
        dt = _t.time() - t0
        sys.stdout.write(f" done in {dt:.1f}s  ")
        if not signals:
            print("0 signals, skipping")
            continue
        sig_arr = np.array(signals)
        wins = (sig_arr[:,2] == 1).sum()
        total = len(sig_arr)
        wr = wins / total
        # PF at 6:2 geometry = (wr*6) / ((1-wr)*2)
        pf = (wr * TP_MULT) / ((1-wr) * SL_MULT + 1e-9)
        # Expectancy in R units
        ev_R = wr * TP_MULT - (1-wr) * SL_MULT
        # Directional breakdown
        longs = sig_arr[sig_arr[:,1]==1]
        shorts = sig_arr[sig_arr[:,1]==-1]
        long_wr = (longs[:,2]==1).mean() if len(longs)>0 else 0.0
        short_wr = (shorts[:,2]==1).mean() if len(shorts)>0 else 0.0
        print(f"n={total:5d}  WR={wr:.1%}  PF={pf:.2f}  EV={ev_R:+.2f}R  "
              f"long={len(longs)}/{long_wr:.0%}  short={len(shorts)}/{short_wr:.0%}")
        results.append(dict(rule=rname, n=total, wr=wr, pf=pf, ev_R=ev_R,
                           n_long=int(len(longs)), long_wr=long_wr,
                           n_short=int(len(shorts)), short_wr=short_wr))

    # Save CSV
    rdf = pd.DataFrame(results).sort_values("pf", ascending=False)
    out_csv = f"{OUT_DIR}/phase1_results_xau.csv"
    rdf.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # Plot
    fig, ax = plt.subplots(figsize=(13, 6), facecolor="#0b0e14")
    ax.set_facecolor("#0d1117")
    colors = ["#00e887" if pf >= 1.0 else "#ff5a5a" for pf in rdf["pf"]]
    bars = ax.bar(rdf["rule"], rdf["pf"], color=colors, edgecolor="#1a2332")
    ax.axhline(1.0, color="#888", linewidth=0.6, linestyle="--")
    for b, n in zip(bars, rdf["n"]):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.03, f"n={n}",
                ha="center", color="#aaa", fontsize=8)
    ax.set_ylabel("Profit Factor (TP=6×ATR, SL=2×ATR)", color="#ddd")
    ax.set_title("XAU — Innovative Rules Phase 1 (Raw, No ML Filter)",
                 color="#ffd700", fontsize=13)
    ax.tick_params(colors="#aaa", labelsize=9)
    plt.xticks(rotation=35, ha="right")
    for sp in ax.spines.values(): sp.set_edgecolor("#1a2332")
    plt.tight_layout()
    out_png = f"{OUT_DIR}/phase1_results_xau.png"
    plt.savefig(out_png, dpi=140, facecolor="#0b0e14")
    print(f"Saved: {out_png}")

    print(f"\n{'='*60}\nTop survivors (PF >= 1.10):")
    for _, r in rdf[rdf["pf"] >= 1.10].iterrows():
        print(f"  {r['rule']:<25} PF={r['pf']:.2f}  WR={r['wr']:.1%}  n={r['n']}")


if __name__ == "__main__":
    main()
