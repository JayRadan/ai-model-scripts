"""
Rule-based setup detector per cluster.

For each cluster, scan the full CSV and emit every bar where a high-quality
setup rule fires. Each emitted row carries:
  - the 36 features at the setup bar
  - direction (+1 long / -1 short)
  - ATR at the setup bar
  - forward outcome label: 1 if TP (+2×ATR) hits before SL (-1×ATR) within
    MAX_FWD bars; 0 otherwise.

Per-cluster rules:
  C0 Ranging    — Bollinger-Band mean reversion
  C1 Downtrend  — Rejection of prior swing high (short only)
  C2 Shock      — SKIP (no rule)
  C3 Uptrend    — Pullback-to-support bounce (long only)

Output: setups_{cid}.csv per cluster.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

FEATURE_COLS = [
    "f01_CPR","f02_WickAsym","f03_BEF","f04_TCS","f05_SPI",
    "f06_LRSlope","f07_RECR","f08_SCM","f09_HLER","f10_EP",
    "f11_KE","f12_MCS","f13_Work","f14_EDR","f15_AI",
    "f16_PPShigh","f16_PPSlow","f17_SCR","f18_RVD","f19_WBER","f20_NCDE",
    "rsi14","rsi6","stoch_k","stoch_d","bb_pct",
    "mom5","mom10","mom20",
    "ll_dist10","hh_dist10",
    "vol_accel","atr_ratio","spread_norm",
    "hour_enc","dow_enc",
]
CLUSTER_NAMES = {0:"Ranging", 1:"Downtrend", 2:"Shock_News", 3:"Uptrend"}

TP_MULT      = 2.0
SL_MULT      = 1.0
MAX_FWD      = 40     # bars to wait for TP/SL
SPREAD_USD   = 0.40


def compute_atr(high, low, close, period=14):
    prev_close = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum.reduce([high - low,
                            np.abs(high - prev_close),
                            np.abs(low  - prev_close)])
    atr = pd.Series(tr).rolling(period, min_periods=1).mean().to_numpy()
    return np.clip(atr, 1e-10, None)


def forward_outcome(highs, lows, closes, atr, i, direction, tp_mult, sl_mult,
                    max_fwd, spread):
    """
    Simulate the trade: TP (+2ATR), SL (-1ATR). Return 1 if TP first, 0 if SL
    first, 0 on timeout. spread is subtracted from TP and added to SL to make
    the label spread-aware.
    """
    entry = closes[i]
    a = max(atr[i], 1e-6)
    if direction == +1:   # LONG
        tp = entry + tp_mult * a - spread
        sl = entry - sl_mult * a - spread
    else:                 # SHORT
        tp = entry - tp_mult * a + spread
        sl = entry + sl_mult * a + spread

    n = len(closes)
    end = min(n, i + max_fwd + 1)
    for k in range(i + 1, end):
        hi, lo = highs[k], lows[k]
        if direction == +1:
            if lo <= sl:
                return 0
            if hi >= tp:
                return 1
        else:
            if hi >= sl:
                return 0
            if lo <= tp:
                return 1
    return 0   # timeout = miss


# ── RULE 0a — C0 Ranging: Bollinger mean reversion ───────────────────────
def rule_ranging_bb(df: pd.DataFrame) -> list[dict]:
    events = []
    closes = df["close"].to_numpy()
    opens  = df["open"].to_numpy()
    rsi6   = df["rsi6"].to_numpy()
    bb     = df["bb_pct"].to_numpy()
    last_fire = -1000
    for i in range(30, len(df) - MAX_FWD - 1):
        if i - last_fire < 10: continue
        if bb[i] <= 0.05 and rsi6[i] < -0.25 and closes[i] > opens[i]:
            events.append({"idx": i, "direction": +1, "rule": "R0a_bb"})
            last_fire = i
        elif bb[i] >= 0.95 and rsi6[i] > 0.25 and closes[i] < opens[i]:
            events.append({"idx": i, "direction": -1, "rule": "R0a_bb"})
            last_fire = i
    return events


def rule_ranging_stoch(df: pd.DataFrame) -> list[dict]:
    """Stochastic extreme reversal: stoch_k in deep zone, bar reverses."""
    events = []
    closes = df["close"].to_numpy()
    opens  = df["open"].to_numpy()
    sk     = df["stoch_k"].to_numpy()
    sd     = df["stoch_d"].to_numpy()
    last_fire = -1000
    for i in range(30, len(df) - MAX_FWD - 1):
        if i - last_fire < 8: continue
        # Oversold → long reversal
        if sk[i] <= 0.10 and sk[i] > sd[i] and closes[i] > opens[i]:
            events.append({"idx": i, "direction": +1, "rule": "R0b_stoch"})
            last_fire = i
        # Overbought → short reversal
        elif sk[i] >= 0.90 and sk[i] < sd[i] and closes[i] < opens[i]:
            events.append({"idx": i, "direction": -1, "rule": "R0b_stoch"})
            last_fire = i
    return events


def rule_ranging_double_touch(df: pd.DataFrame) -> list[dict]:
    """Second test of a recent extreme that was rejected the first time."""
    events = []
    highs = df["high"].to_numpy()
    lows  = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    opens  = df["open"].to_numpy()
    last_fire = -1000
    for i in range(40, len(df) - MAX_FWD - 1):
        if i - last_fire < 10: continue
        # Look back 10-30 bars for a prior low near current low → double bottom
        prior_low = lows[i-30:i-5].min()
        if abs(lows[i] - prior_low) / max(lows[i], 1e-6) < 0.002 and closes[i] > opens[i]:
            events.append({"idx": i, "direction": +1, "rule": "R0c_doubletouch"})
            last_fire = i
            continue
        prior_high = highs[i-30:i-5].max()
        if abs(highs[i] - prior_high) / max(highs[i], 1e-6) < 0.002 and closes[i] < opens[i]:
            events.append({"idx": i, "direction": -1, "rule": "R0c_doubletouch"})
            last_fire = i
    return events


def rule_ranging_squeeze(df: pd.DataFrame) -> list[dict]:
    """Volatility contraction (low atr_ratio) then a directional breakout bar."""
    events = []
    closes = df["close"].to_numpy()
    opens  = df["open"].to_numpy()
    atr_ratio = df["atr_ratio"].to_numpy()
    highs = df["high"].to_numpy()
    lows  = df["low"].to_numpy()
    last_fire = -1000
    for i in range(30, len(df) - MAX_FWD - 1):
        if i - last_fire < 12: continue
        # Require prior 3 bars of low vol, then expansion this bar
        if atr_ratio[i-3:i].max() > -0.15: continue
        rng = highs[i] - lows[i]
        if rng <= 0: continue
        prev_rng = highs[i-1] - lows[i-1]
        if rng < prev_rng * 1.5: continue
        if closes[i] > opens[i]:
            events.append({"idx": i, "direction": +1, "rule": "R0d_squeeze"})
        else:
            events.append({"idx": i, "direction": -1, "rule": "R0d_squeeze"})
        last_fire = i
    return events


def rule_ranging(df: pd.DataFrame) -> list[dict]:
    return (rule_ranging_bb(df) + rule_ranging_stoch(df)
            + rule_ranging_double_touch(df) + rule_ranging_squeeze(df))


# ── RULE 1 — C1 Downtrend: rejection of swing high ──────────────────────
def rule_downtrend_swing_high(df: pd.DataFrame) -> list[dict]:
    """Rejection of a local swing high."""
    events = []
    highs = df["high"].to_numpy()
    lows  = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    hhdist = df["hh_dist10"].to_numpy()
    last_fire = -1000
    for i in range(30, len(df) - MAX_FWD - 1):
        if i - last_fire < 8: continue
        window_hi = highs[i-5:i+1].max()
        if highs[i] != window_hi: continue
        rng = highs[i] - lows[i]
        if rng < 1e-6: continue
        close_pos = (closes[i] - lows[i]) / rng
        if close_pos < 0.5 and hhdist[i] < 0.5:
            events.append({"idx": i, "direction": -1, "rule": "R1a_swinghigh"})
            last_fire = i
    return events


def rule_downtrend_lower_high(df: pd.DataFrame) -> list[dict]:
    """Lower-high continuation: price made a local high that's below the prior one."""
    events = []
    highs = df["high"].to_numpy()
    closes = df["close"].to_numpy()
    opens  = df["open"].to_numpy()
    last_fire = -1000
    for i in range(40, len(df) - MAX_FWD - 1):
        if i - last_fire < 8: continue
        local_hi = highs[i-5:i+1].max()
        if highs[i] != local_hi: continue
        prior_hi = highs[i-30:i-5].max()
        if highs[i] >= prior_hi: continue   # must be LOWER than prior
        if closes[i] >= opens[i]: continue  # red candle
        events.append({"idx": i, "direction": -1, "rule": "R1b_lowerhigh"})
        last_fire = i
    return events


def rule_downtrend_bounce_fade(df: pd.DataFrame) -> list[dict]:
    """Bounce into prior resistance fades (closes red after tagging)."""
    events = []
    highs = df["high"].to_numpy()
    closes = df["close"].to_numpy()
    opens  = df["open"].to_numpy()
    mom5   = df["mom5"].to_numpy()
    last_fire = -1000
    for i in range(40, len(df) - MAX_FWD - 1):
        if i - last_fire < 10: continue
        # Last 3 bars were green (a bounce) then this bar is red
        if not (closes[i-3] > opens[i-3] and closes[i-2] > opens[i-2] and closes[i-1] > opens[i-1]):
            continue
        if closes[i] >= opens[i]: continue
        # Must have tagged a recent high
        prior_hi = highs[i-20:i-3].max()
        if highs[i] < prior_hi * 0.998: continue
        if mom5[i] <= 0: continue   # upward momentum into the failure
        events.append({"idx": i, "direction": -1, "rule": "R1c_bouncefade"})
        last_fire = i
    return events


def rule_downtrend_overbought(df: pd.DataFrame) -> list[dict]:
    """Momentum-exhaustion short: stoch extreme + bearish reversal bar."""
    events = []
    sk = df["stoch_k"].to_numpy()
    closes = df["close"].to_numpy()
    opens  = df["open"].to_numpy()
    last_fire = -1000
    for i in range(30, len(df) - MAX_FWD - 1):
        if i - last_fire < 8: continue
        if sk[i-1] < 0.80: continue     # prior bar overbought
        if sk[i] >= sk[i-1]: continue   # stoch turning down
        if closes[i] >= opens[i]: continue
        events.append({"idx": i, "direction": -1, "rule": "R1d_overbought"})
        last_fire = i
    return events


def rule_downtrend(df: pd.DataFrame) -> list[dict]:
    return (rule_downtrend_swing_high(df) + rule_downtrend_lower_high(df)
            + rule_downtrend_bounce_fade(df) + rule_downtrend_overbought(df))


def rule_uptrend_pullback(df: pd.DataFrame) -> list[dict]:
    events = []
    highs = df["high"].to_numpy()
    lows  = df["low"].to_numpy()
    opens = df["open"].to_numpy()
    closes = df["close"].to_numpy()
    rsi6  = df["rsi6"].to_numpy()
    lldist = df["ll_dist10"].to_numpy()
    last_fire = -1000
    for i in range(30, len(df) - MAX_FWD - 1):
        if i - last_fire < 8: continue
        if closes[i] <= opens[i]: continue
        rng = highs[i] - lows[i]
        if rng < 1e-6: continue
        if (closes[i] - lows[i]) / rng < 0.5: continue
        if closes[i] >= closes[i-5] or closes[i-5] >= closes[i-10]:
            continue
        if lldist[i] >= 0.8 or rsi6[i] >= 0.15: continue
        events.append({"idx": i, "direction": +1, "rule": "R3a_pullback"})
        last_fire = i
    return events


def rule_uptrend_higher_low(df: pd.DataFrame) -> list[dict]:
    """Higher-low continuation: local low above prior local low + bullish bar."""
    events = []
    lows  = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    opens  = df["open"].to_numpy()
    last_fire = -1000
    for i in range(40, len(df) - MAX_FWD - 1):
        if i - last_fire < 8: continue
        local_lo = lows[i-5:i+1].min()
        if lows[i] != local_lo: continue
        prior_lo = lows[i-30:i-5].min()
        if lows[i] <= prior_lo: continue
        if closes[i] <= opens[i]: continue
        events.append({"idx": i, "direction": +1, "rule": "R3b_higherlow"})
        last_fire = i
    return events


def rule_uptrend_breakout_pullback(df: pd.DataFrame) -> list[dict]:
    """Breakout of prior 20-bar high, then pullback that holds → continuation."""
    events = []
    highs = df["high"].to_numpy()
    lows  = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    opens  = df["open"].to_numpy()
    last_fire = -1000
    for i in range(40, len(df) - MAX_FWD - 1):
        if i - last_fire < 10: continue
        # In last 10 bars there was a breakout (bar j had close > prior 20-bar high)
        broke = False
        break_level = None
        for j in range(i - 10, i - 2):
            prior_hi = highs[j-20:j].max()
            if closes[j] > prior_hi:
                broke = True
                break_level = prior_hi
                break
        if not broke: continue
        # Current bar pulled back near break_level and closed bullish
        if lows[i] > break_level * 1.003: continue   # didn't pull back enough
        if lows[i] < break_level * 0.997: continue   # broke back down
        if closes[i] <= opens[i]: continue
        events.append({"idx": i, "direction": +1, "rule": "R3c_breakpullback"})
        last_fire = i
    return events


def rule_uptrend_oversold(df: pd.DataFrame) -> list[dict]:
    """Oversold bounce in an uptrend: stoch low + bullish reversal bar."""
    events = []
    sk = df["stoch_k"].to_numpy()
    closes = df["close"].to_numpy()
    opens  = df["open"].to_numpy()
    mom20  = df["mom20"].to_numpy()
    last_fire = -1000
    for i in range(30, len(df) - MAX_FWD - 1):
        if i - last_fire < 8: continue
        if sk[i-1] > 0.20: continue     # prior bar oversold
        if sk[i] <= sk[i-1]: continue   # stoch turning up
        if closes[i] <= opens[i]: continue
        if mom20[i] <= 0: continue      # must be within an uptrend
        events.append({"idx": i, "direction": +1, "rule": "R3d_oversold"})
        last_fire = i
    return events


def rule_uptrend(df: pd.DataFrame) -> list[dict]:
    return (rule_uptrend_pullback(df) + rule_uptrend_higher_low(df)
            + rule_uptrend_breakout_pullback(df) + rule_uptrend_oversold(df))


RULES = {
    0: ("Ranging",   rule_ranging),
    1: ("Downtrend", rule_downtrend),
    3: ("Uptrend",   rule_uptrend),
}


def process_cluster(cid: int):
    name, rule_fn = RULES[cid]
    print(f"\n── C{cid} {name} ──")
    df = (pd.read_csv(f"cluster_{cid}_data.csv", parse_dates=["time"])
            .sort_values("time").reset_index(drop=True))
    print(f"  rows: {len(df):,}")

    highs  = df["high"].to_numpy(np.float64)
    lows   = df["low"].to_numpy(np.float64)
    closes = df["close"].to_numpy(np.float64)
    atr    = compute_atr(highs, lows, closes, 14)

    events = rule_fn(df)
    print(f"  raw events (pre-dedup): {len(events):,}")
    if not events:
        print("  no events — rule too strict")
        return

    # De-duplicate by (idx, direction) — if multiple rules fire same bar same
    # direction, keep the first one (stable order).
    seen = set()
    unique = []
    for ev in sorted(events, key=lambda e: e["idx"]):
        key = (ev["idx"], ev["direction"])
        if key in seen: continue
        seen.add(key)
        unique.append(ev)
    events = unique
    print(f"  unique setups (post-dedup): {len(events):,}")

    # Per-rule breakdown
    rule_counts = {}
    for ev in events:
        rule_counts[ev["rule"]] = rule_counts.get(ev["rule"], 0) + 1
    for r, c in sorted(rule_counts.items()):
        print(f"    {r}: {c:,}")

    # Attach label and feature vector
    rows = []
    for ev in events:
        i = ev["idx"]
        lbl = forward_outcome(highs, lows, closes, atr, i, ev["direction"],
                              TP_MULT, SL_MULT, MAX_FWD, SPREAD_USD)
        feat = df.iloc[i][FEATURE_COLS].to_dict()
        feat["time"]        = df.iloc[i]["time"]
        feat["idx"]         = i
        feat["direction"]   = ev["direction"]
        feat["rule"]        = ev["rule"]
        feat["atr"]         = float(atr[i])
        feat["entry_price"] = float(closes[i])
        feat["label"]       = int(lbl)
        rows.append(feat)

    out = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
    path = f"setups_{cid}.csv"
    out.to_csv(path, index=False)
    win_rate_raw = out["label"].mean()
    print(f"  labeled setups: {len(out):,}  raw WR: {win_rate_raw*100:.1f}%")
    print(f"  long/short: {(out['direction']==1).sum()} / {(out['direction']==-1).sum()}")
    # Per-rule raw WR
    for r in sorted(rule_counts):
        sub = out[out["rule"] == r]
        if len(sub) > 0:
            print(f"    {r}: n={len(sub):,}  raw WR={sub['label'].mean()*100:.1f}%")
    print(f"  saved: {path}")


if __name__ == "__main__":
    for cid in [0, 1, 3]:
        process_cluster(cid)
    print("\nSkipped C2 Shock_News — no rule (never trade).")
