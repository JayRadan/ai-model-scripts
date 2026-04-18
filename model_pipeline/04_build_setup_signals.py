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
import paths as P

BASE_FEAT_COLS = [
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
HTF_FEAT_COLS = [
    "h1_trend_sma20", "h1_trend_sma50", "h1_slope5", "h1_rsi14",
    "h1_atr_ratio", "h1_dist_sma20", "h1_dist_sma50",
    "h4_trend_sma20", "h4_trend_sma50", "h4_slope5", "h4_rsi14",
    "h4_atr_ratio", "h4_dist_sma20", "h4_dist_sma50",
]
FEATURE_COLS = BASE_FEAT_COLS + HTF_FEAT_COLS
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


def rule_uptrend_pullback_sma(df: pd.DataFrame) -> list[dict]:
    """R3f: bar tags 20-SMA from above and closes back above (trend support bounce)."""
    events = []
    closes = df["close"].to_numpy()
    opens  = df["open"].to_numpy()
    lows   = df["low"].to_numpy()
    mom20  = df["mom20"].to_numpy()
    sma20  = pd.Series(closes).rolling(20, min_periods=20).mean().to_numpy()
    last_fire = -1000
    for i in range(30, len(df) - MAX_FWD - 1):
        if i - last_fire < 10: continue
        if np.isnan(sma20[i]): continue
        if mom20[i] <= 0: continue                         # must be in uptrend context
        if lows[i] > sma20[i]: continue                    # didn't tag
        if closes[i] <= sma20[i]: continue                 # didn't reclaim
        if closes[i] <= opens[i]: continue                 # must be bullish bar
        events.append({"idx": i, "direction": +1, "rule": "R3f_sma_bounce"})
        last_fire = i
    return events


def rule_uptrend_false_breakdown(df: pd.DataFrame) -> list[dict]:
    """R3e: price broke below 20-bar low but closed back inside → failed breakdown."""
    events = []
    lows   = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    last_fire = -1000
    for i in range(30, len(df) - MAX_FWD - 1):
        if i - last_fire < 10: continue
        prior_lo = lows[i-20:i].min()
        if lows[i] >= prior_lo: continue                   # must dip below
        if closes[i] <= prior_lo: continue                 # must close back above
        if closes[i] <= lows[i] + (closes[i] - lows[i]) * 0.5: continue  # closed in top half
        events.append({"idx": i, "direction": +1, "rule": "R3e_false_breakdown"})
        last_fire = i
    return events


def rule_uptrend_three_green(df: pd.DataFrame) -> list[dict]:
    """R3g: 3 consecutive green bars in an uptrend, current bar in upper 60% of range."""
    events = []
    opens  = df["open"].to_numpy()
    closes = df["close"].to_numpy()
    highs  = df["high"].to_numpy()
    lows   = df["low"].to_numpy()
    mom20  = df["mom20"].to_numpy()
    last_fire = -1000
    for i in range(30, len(df) - MAX_FWD - 1):
        if i - last_fire < 8: continue
        if not (closes[i]>opens[i] and closes[i-1]>opens[i-1] and closes[i-2]>opens[i-2]):
            continue
        if mom20[i] <= 0: continue
        rng = highs[i] - lows[i]
        if rng < 1e-6: continue
        if (closes[i] - lows[i]) / rng < 0.6: continue
        events.append({"idx": i, "direction": +1, "rule": "R3g_three_green"})
        last_fire = i
    return events


def rule_uptrend_higher_close_streak(df: pd.DataFrame) -> list[dict]:
    """R3h: 4+ higher closes in a row + bullish bar + uptrend context."""
    events = []
    closes = df["close"].to_numpy()
    opens  = df["open"].to_numpy()
    mom10  = df["mom10"].to_numpy()
    last_fire = -1000
    for i in range(30, len(df) - MAX_FWD - 1):
        if i - last_fire < 10: continue
        if not (closes[i]>closes[i-1]>closes[i-2]>closes[i-3]>closes[i-4]): continue
        if mom10[i] <= 0: continue
        if closes[i] <= opens[i]: continue
        events.append({"idx": i, "direction": +1, "rule": "R3h_close_streak"})
        last_fire = i
    return events


def rule_uptrend_inside_bar_break(df: pd.DataFrame) -> list[dict]:
    """R3i: prior bar contained inside the bar before it; current bar breaks above."""
    events = []
    highs  = df["high"].to_numpy()
    lows   = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    opens  = df["open"].to_numpy()
    mom20  = df["mom20"].to_numpy()
    last_fire = -1000
    for i in range(30, len(df) - MAX_FWD - 1):
        if i - last_fire < 10: continue
        # bar i-1 is inside bar i-2
        if not (highs[i-1] <= highs[i-2] and lows[i-1] >= lows[i-2]): continue
        # bar i breaks above bar i-1's high
        if closes[i] <= highs[i-1]: continue
        if closes[i] <= opens[i]: continue
        if mom20[i] <= 0: continue
        events.append({"idx": i, "direction": +1, "rule": "R3i_inside_break"})
        last_fire = i
    return events


def rule_uptrend(df: pd.DataFrame) -> list[dict]:
    return (rule_uptrend_pullback(df) + rule_uptrend_higher_low(df)
            + rule_uptrend_breakout_pullback(df) + rule_uptrend_oversold(df)
            + rule_uptrend_pullback_sma(df) + rule_uptrend_false_breakdown(df)
            + rule_uptrend_three_green(df) + rule_uptrend_higher_close_streak(df)
            + rule_uptrend_inside_bar_break(df))


# ── NEW C0 RULES ────────────────────────────────────────────────────────
def rule_ranging_nr4_breakout(df: pd.DataFrame) -> list[dict]:
    """R0e: prior bar had smallest range of last 4 (NR4), current bar breaks out."""
    events = []
    highs  = df["high"].to_numpy()
    lows   = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    opens  = df["open"].to_numpy()
    last_fire = -1000
    for i in range(30, len(df) - MAX_FWD - 1):
        if i - last_fire < 10: continue
        ranges = highs[i-4:i] - lows[i-4:i]
        if len(ranges) < 4: continue
        nr4_range = ranges[-1]                             # bar i-1 range
        if nr4_range != ranges.min(): continue             # i-1 must be the narrowest
        cur_rng = highs[i] - lows[i]
        if cur_rng <= 0 or cur_rng < nr4_range * 1.8: continue
        direction = +1 if closes[i] > opens[i] else -1
        events.append({"idx": i, "direction": direction, "rule": "R0e_nr4_break"})
        last_fire = i
    return events


def rule_ranging_mean_revert(df: pd.DataFrame) -> list[dict]:
    """R0f: close >2 ATR away from 20-bar SMA, bar reverses toward mean."""
    events = []
    closes = df["close"].to_numpy()
    opens  = df["open"].to_numpy()
    highs  = df["high"].to_numpy()
    lows   = df["low"].to_numpy()
    atr    = compute_atr(highs, lows, closes, 14)
    sma20  = pd.Series(closes).rolling(20, min_periods=20).mean().to_numpy()
    last_fire = -1000
    for i in range(30, len(df) - MAX_FWD - 1):
        if i - last_fire < 10: continue
        if np.isnan(sma20[i]) or atr[i] <= 0: continue
        dev = closes[i] - sma20[i]
        rng = highs[i] - lows[i]
        if rng <= 0: continue
        if dev >= 2.0 * atr[i]:                            # stretched above → short
            close_pos = (closes[i] - lows[i]) / rng
            if close_pos < 0.45 and closes[i] < opens[i]:
                events.append({"idx": i, "direction": -1, "rule": "R0f_mean_revert"})
                last_fire = i
        elif dev <= -2.0 * atr[i]:                         # stretched below → long
            close_pos = (closes[i] - lows[i]) / rng
            if close_pos > 0.55 and closes[i] > opens[i]:
                events.append({"idx": i, "direction": +1, "rule": "R0f_mean_revert"})
                last_fire = i
    return events


def rule_ranging_inside_break(df: pd.DataFrame) -> list[dict]:
    """R0g: inside bar followed by directional break."""
    events = []
    highs  = df["high"].to_numpy()
    lows   = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    opens  = df["open"].to_numpy()
    last_fire = -1000
    for i in range(30, len(df) - MAX_FWD - 1):
        if i - last_fire < 8: continue
        # bar i-1 inside bar i-2
        if not (highs[i-1] <= highs[i-2] and lows[i-1] >= lows[i-2]): continue
        # current bar breaks the inside bar's high or low
        if closes[i] > highs[i-1] and closes[i] > opens[i]:
            events.append({"idx": i, "direction": +1, "rule": "R0g_inside_break"})
            last_fire = i
        elif closes[i] < lows[i-1] and closes[i] < opens[i]:
            events.append({"idx": i, "direction": -1, "rule": "R0g_inside_break"})
            last_fire = i
    return events


def rule_ranging_three_bar_reversal(df: pd.DataFrame) -> list[dict]:
    """R0h: 2 bars in one direction then a strong reversal bar."""
    events = []
    closes = df["close"].to_numpy()
    opens  = df["open"].to_numpy()
    highs  = df["high"].to_numpy()
    lows   = df["low"].to_numpy()
    last_fire = -1000
    for i in range(30, len(df) - MAX_FWD - 1):
        if i - last_fire < 8: continue
        rng = highs[i] - lows[i]
        if rng < 1e-6: continue
        prv1_red = closes[i-1] < opens[i-1]
        prv2_red = closes[i-2] < opens[i-2]
        prv1_grn = closes[i-1] > opens[i-1]
        prv2_grn = closes[i-2] > opens[i-2]
        # 2 reds → green reversal
        if prv1_red and prv2_red and closes[i] > opens[i]:
            if (closes[i] - lows[i]) / rng > 0.6:
                events.append({"idx": i, "direction": +1, "rule": "R0h_3bar_reversal"})
                last_fire = i
                continue
        # 2 greens → red reversal
        if prv1_grn and prv2_grn and closes[i] < opens[i]:
            if (closes[i] - lows[i]) / rng < 0.4:
                events.append({"idx": i, "direction": -1, "rule": "R0h_3bar_reversal"})
                last_fire = i
    return events


def rule_ranging_close_extreme(df: pd.DataFrame) -> list[dict]:
    """R0i: close in extreme top/bottom 10% of bar's range AND 10-bar extreme touched."""
    events = []
    highs  = df["high"].to_numpy()
    lows   = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    last_fire = -1000
    for i in range(30, len(df) - MAX_FWD - 1):
        if i - last_fire < 8: continue
        rng = highs[i] - lows[i]
        if rng < 1e-6: continue
        close_pos = (closes[i] - lows[i]) / rng
        prior_lo = lows[i-10:i].min()
        prior_hi = highs[i-10:i].max()
        # closed at the bottom AND tagged a recent low → reversal long
        if close_pos > 0.90 and lows[i] <= prior_lo:
            events.append({"idx": i, "direction": +1, "rule": "R0i_close_extreme"})
            last_fire = i
        elif close_pos < 0.10 and highs[i] >= prior_hi:
            events.append({"idx": i, "direction": -1, "rule": "R0i_close_extreme"})
            last_fire = i
    return events


def rule_ranging(df: pd.DataFrame) -> list[dict]:
    return (rule_ranging_bb(df) + rule_ranging_stoch(df)
            + rule_ranging_double_touch(df) + rule_ranging_squeeze(df)
            + rule_ranging_nr4_breakout(df) + rule_ranging_mean_revert(df)
            + rule_ranging_inside_break(df) + rule_ranging_three_bar_reversal(df)
            + rule_ranging_close_extreme(df))


# ── NEW C1 RULES ────────────────────────────────────────────────────────
def rule_downtrend_false_breakout(df: pd.DataFrame) -> list[dict]:
    """R1e: price broke above 20-bar high but closed back inside → failed breakout."""
    events = []
    highs  = df["high"].to_numpy()
    closes = df["close"].to_numpy()
    last_fire = -1000
    for i in range(30, len(df) - MAX_FWD - 1):
        if i - last_fire < 10: continue
        prior_hi = highs[i-20:i].max()
        if highs[i] <= prior_hi: continue                  # must poke above
        if closes[i] >= prior_hi: continue                 # must close back inside
        rng = highs[i] - (highs[i] - (highs[i] - closes[i]))
        events.append({"idx": i, "direction": -1, "rule": "R1e_false_breakout"})
        last_fire = i
    return events


def rule_downtrend_sma_rejection(df: pd.DataFrame) -> list[dict]:
    """R1f: bar pokes above 20-SMA from below and closes back below (rejection)."""
    events = []
    highs  = df["high"].to_numpy()
    closes = df["close"].to_numpy()
    opens  = df["open"].to_numpy()
    mom20  = df["mom20"].to_numpy()
    sma20  = pd.Series(closes).rolling(20, min_periods=20).mean().to_numpy()
    last_fire = -1000
    for i in range(30, len(df) - MAX_FWD - 1):
        if i - last_fire < 10: continue
        if np.isnan(sma20[i]): continue
        if mom20[i] >= 0: continue                         # must be in downtrend
        if highs[i] < sma20[i]: continue                   # didn't tag
        if closes[i] >= sma20[i]: continue                 # didn't reject
        if closes[i] >= opens[i]: continue                 # must be bearish bar
        events.append({"idx": i, "direction": -1, "rule": "R1f_sma_reject"})
        last_fire = i
    return events


def rule_downtrend_three_red(df: pd.DataFrame) -> list[dict]:
    """R1g: 3 consecutive red bars + downtrend context."""
    events = []
    opens  = df["open"].to_numpy()
    closes = df["close"].to_numpy()
    highs  = df["high"].to_numpy()
    lows   = df["low"].to_numpy()
    mom20  = df["mom20"].to_numpy()
    last_fire = -1000
    for i in range(30, len(df) - MAX_FWD - 1):
        if i - last_fire < 8: continue
        if not (closes[i]<opens[i] and closes[i-1]<opens[i-1] and closes[i-2]<opens[i-2]):
            continue
        if mom20[i] >= 0: continue
        rng = highs[i] - lows[i]
        if rng < 1e-6: continue
        if (closes[i] - lows[i]) / rng > 0.4: continue          # closed in lower 40%
        events.append({"idx": i, "direction": -1, "rule": "R1g_three_red"})
        last_fire = i
    return events


def rule_downtrend_lower_close_streak(df: pd.DataFrame) -> list[dict]:
    """R1h: 4+ lower closes in a row + bearish bar + downtrend."""
    events = []
    closes = df["close"].to_numpy()
    opens  = df["open"].to_numpy()
    mom10  = df["mom10"].to_numpy()
    last_fire = -1000
    for i in range(30, len(df) - MAX_FWD - 1):
        if i - last_fire < 10: continue
        if not (closes[i]<closes[i-1]<closes[i-2]<closes[i-3]<closes[i-4]): continue
        if mom10[i] >= 0: continue
        if closes[i] >= opens[i]: continue
        events.append({"idx": i, "direction": -1, "rule": "R1h_close_streak"})
        last_fire = i
    return events


def rule_downtrend(df: pd.DataFrame) -> list[dict]:
    return (rule_downtrend_swing_high(df) + rule_downtrend_lower_high(df)
            + rule_downtrend_bounce_fade(df) + rule_downtrend_overbought(df)
            + rule_downtrend_false_breakout(df) + rule_downtrend_sma_rejection(df)
            + rule_downtrend_three_red(df) + rule_downtrend_lower_close_streak(df))


# ── C2 SHOCK RULES ──────────────────────────────────────────────────────
def rule_shock_vol_breakout(df: pd.DataFrame) -> list[dict]:
    """R2a: ATR spike (>1.5x recent mean) + strong directional bar.
    Direction = sign(close - open). Captures the continuation of a volatility burst.
    """
    events = []
    highs  = df["high"].to_numpy()
    lows   = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    opens  = df["open"].to_numpy()
    atr    = compute_atr(highs, lows, closes, 14)
    last_fire = -1000
    for i in range(30, len(df) - MAX_FWD - 1):
        if i - last_fire < 8: continue
        recent_atr = atr[i-10:i].mean()
        if recent_atr <= 0 or atr[i] < 1.5 * recent_atr: continue
        rng = highs[i] - lows[i]
        if rng <= 0: continue
        close_pos = (closes[i] - lows[i]) / rng
        if close_pos > 0.75 and closes[i] > opens[i]:
            events.append({"idx": i, "direction": +1, "rule": "R2a_vol_breakout"})
            last_fire = i
        elif close_pos < 0.25 and closes[i] < opens[i]:
            events.append({"idx": i, "direction": -1, "rule": "R2a_vol_breakout"})
            last_fire = i
    return events


def rule_shock_v_reversal(df: pd.DataFrame) -> list[dict]:
    """R2b: bar made a new 10-bar extreme but closed more than 50% back → V-reversal."""
    events = []
    highs  = df["high"].to_numpy()
    lows   = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    last_fire = -1000
    for i in range(30, len(df) - MAX_FWD - 1):
        if i - last_fire < 10: continue
        rng = highs[i] - lows[i]
        if rng <= 0: continue
        prior_lo = lows[i-10:i].min()
        prior_hi = highs[i-10:i].max()
        # V-bottom: new low but closed in upper half → long
        if lows[i] < prior_lo and (closes[i] - lows[i]) / rng > 0.60:
            events.append({"idx": i, "direction": +1, "rule": "R2b_v_reversal"})
            last_fire = i
        # V-top: new high but closed in lower half → short
        elif highs[i] > prior_hi and (closes[i] - lows[i]) / rng < 0.40:
            events.append({"idx": i, "direction": -1, "rule": "R2b_v_reversal"})
            last_fire = i
    return events


def rule_shock_gap_fade(df: pd.DataFrame) -> list[dict]:
    """R2c: large open gap from prior close (>1.2 ATR), bar closes direction of fade."""
    events = []
    opens  = df["open"].to_numpy()
    closes = df["close"].to_numpy()
    highs  = df["high"].to_numpy()
    lows   = df["low"].to_numpy()
    atr    = compute_atr(highs, lows, closes, 14)
    last_fire = -1000
    for i in range(30, len(df) - MAX_FWD - 1):
        if i - last_fire < 10: continue
        if atr[i] <= 0: continue
        gap = opens[i] - closes[i-1]
        if abs(gap) < 1.2 * atr[i]: continue
        # Gap up → fade down. Bar must close below open to confirm fade.
        if gap > 0 and closes[i] < opens[i]:
            events.append({"idx": i, "direction": -1, "rule": "R2c_gap_fade"})
            last_fire = i
        elif gap < 0 and closes[i] > opens[i]:
            events.append({"idx": i, "direction": +1, "rule": "R2c_gap_fade"})
            last_fire = i
    return events


def rule_shock_continuation(df: pd.DataFrame) -> list[dict]:
    """R2d: large bar (>1.3x recent ATR) followed by next bar in same direction.
    Captures 'momentum sticks' inside shock weeks.
    """
    events = []
    highs  = df["high"].to_numpy()
    lows   = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    opens  = df["open"].to_numpy()
    atr    = compute_atr(highs, lows, closes, 14)
    last_fire = -1000
    for i in range(30, len(df) - MAX_FWD - 1):
        if i - last_fire < 6: continue
        if atr[i] <= 0: continue
        prev_rng = highs[i-1] - lows[i-1]
        prev_atr = atr[i-1]
        if prev_atr <= 0: continue
        if prev_rng < 1.3 * prev_atr: continue                 # prior bar must be big
        prev_dir = +1 if closes[i-1] > opens[i-1] else -1
        cur_dir = +1 if closes[i] > opens[i] else -1
        if cur_dir != prev_dir: continue                       # must continue same direction
        rng = highs[i] - lows[i]
        if rng < 1e-6: continue
        close_pos = (closes[i] - lows[i]) / rng
        if cur_dir == +1 and close_pos > 0.55:
            events.append({"idx": i, "direction": +1, "rule": "R2d_continuation"})
            last_fire = i
        elif cur_dir == -1 and close_pos < 0.45:
            events.append({"idx": i, "direction": -1, "rule": "R2d_continuation"})
            last_fire = i
    return events


def rule_shock(df: pd.DataFrame) -> list[dict]:
    return (rule_shock_vol_breakout(df) + rule_shock_v_reversal(df)
            + rule_shock_gap_fade(df) + rule_shock_continuation(df))


# Split old Ranging into MeanRevert (fade rules) and TrendRange (breakout rules)
def rule_meanrevert(df):
    """Fade/reversal rules for mean-reverting range markets."""
    return (rule_ranging_bb(df) + rule_ranging_stoch(df) +
            rule_ranging_double_touch(df) + rule_ranging_mean_revert(df) +
            rule_ranging_close_extreme(df))

def rule_trendrange(df):
    """Breakout/momentum rules for trending range markets."""
    return (rule_ranging_squeeze(df) + rule_ranging_nr4_breakout(df) +
            rule_ranging_inside_break(df) + rule_ranging_three_bar_reversal(df))

RULES = {
    0: ("Uptrend",    rule_uptrend),
    1: ("MeanRevert", rule_meanrevert),
    2: ("TrendRange", rule_trendrange),
    3: ("Downtrend",  rule_downtrend),
    4: ("HighVol",    rule_shock),
}


def process_cluster(cid: int):
    name, rule_fn = RULES[cid]
    print(f"\n── C{cid} {name} ──")
    df = (pd.read_csv(P.data(f"cluster_{cid}_data.csv"), parse_dates=["time"])
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
        feat = df.iloc[i][[c for c in BASE_FEAT_COLS if c in df.columns]].to_dict()
        feat["time"]        = df.iloc[i]["time"]
        feat["idx"]         = i
        feat["direction"]   = ev["direction"]
        feat["rule"]        = ev["rule"]
        feat["atr"]         = float(atr[i])
        feat["entry_price"] = float(closes[i])
        feat["label"]       = int(lbl)
        rows.append(feat)

    out = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
    path = P.data(f"setups_{cid}.csv")
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
    for cid in [0, 1, 2, 3, 4]:
        process_cluster(cid)
