"""
Rule-parity audit — verbatim Python port of every MQL5 rule in
setup_rules.mqh, run against the same bars as the Python training rules.
Prints per-rule MATCH / DIFFER status and the exact timing diff.

MQL5 convention: rb[] is AS-SERIES (rb[0]=current bar, rb[1]=prev, …).
When we port to Python operating on arrays indexed with i=current,
rb[k] → arr[i-k].
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

ROOT = "/home/jay/Desktop/new-model-zigzag"
sys.path.insert(0, ROOT + "/model_pipeline")
sys.path.insert(0, ROOT + "/experiments/v72_lite_btc_deploy")

from importlib.machinery import SourceFileLoader
tech = SourceFileLoader("labeler_v4", ROOT + "/model_pipeline/01_labeler_v4.py").load_module()
setup_py = SourceFileLoader("setup_btc", ROOT + "/experiments/v72_lite_btc_deploy/04_build_setup_signals_btc.py").load_module()

CSV = os.path.expanduser(
    "~/.mt5/drive_c/Program Files/MetaTrader 5/MQL5/Files/swing_v5_btc_today.csv")

MAX_FWD = 40

# ─── MQL5 RULE PORTS ─────────────────────────────────────────────────────
# rb[k] in MQL5 maps to arr[i-k] in Python.
# feat[FI_X] is bar-local feature; we look it up by name in the df.
# Direction convention: +1 long, -1 short, 0 no-signal.

def mql_R0a_bb(df, i):
    bb = df["bb_pct"].iat[i]; rsi6 = df["rsi6"].iat[i]
    o = df["open"].iat[i]; c = df["close"].iat[i]
    if bb <= 0.05 and rsi6 < -0.25 and c > o: return +1
    if bb >= 0.95 and rsi6 >  0.25 and c < o: return -1
    return 0

def mql_R0b_stoch(df, i):
    sk = df["stoch_k"].iat[i]; sd = df["stoch_d"].iat[i]
    o = df["open"].iat[i]; c = df["close"].iat[i]
    if sk <= 0.10 and sk > sd and c > o: return +1
    if sk >= 0.90 and sk < sd and c < o: return -1
    return 0

def mql_R0c_doubletouch(df, i):
    if i < 35: return 0
    cur_lo = df["low"].iat[i]; cur_hi = df["high"].iat[i]
    prior_lo = df["low"].iat[i-5]; prior_hi = df["high"].iat[i-5]
    for k in range(6, 31):
        if df["low"].iat[i-k] < prior_lo: prior_lo = df["low"].iat[i-k]
        if df["high"].iat[i-k] > prior_hi: prior_hi = df["high"].iat[i-k]
    o = df["open"].iat[i]; c = df["close"].iat[i]
    if abs(cur_lo - prior_lo) / max(cur_lo, 1e-6) < 0.002 and c > o: return +1
    if abs(cur_hi - prior_hi) / max(cur_hi, 1e-6) < 0.002 and c < o: return -1
    return 0

def mql_R0d_squeeze(df, i):
    """Post-patch behavior: mirrors the updated setup_rules.mqh — require
    the prior 3 bars' atr_ratio all <= -0.15 (3-bar compression window)."""
    if i < 5: return 0
    for k in range(1, 4):
        if df["atr_ratio"].iat[i-k] > -0.15: return 0
    cur_rng = df["high"].iat[i] - df["low"].iat[i]
    prv_rng = df["high"].iat[i-1] - df["low"].iat[i-1]
    if cur_rng <= 0 or cur_rng < prv_rng * 1.5: return 0
    return +1 if df["close"].iat[i] > df["open"].iat[i] else -1

def mql_R0e_nr4_break(df, i):
    if i < 5: return 0
    r1 = df["high"].iat[i-1] - df["low"].iat[i-1]
    r2 = df["high"].iat[i-2] - df["low"].iat[i-2]
    r3 = df["high"].iat[i-3] - df["low"].iat[i-3]
    r4 = df["high"].iat[i-4] - df["low"].iat[i-4]
    mn = min(r1, r2, r3, r4)
    if r1 != mn: return 0
    cur_rng = df["high"].iat[i] - df["low"].iat[i]
    if cur_rng <= 0 or cur_rng < r1 * 1.8: return 0
    return +1 if df["close"].iat[i] > df["open"].iat[i] else -1

def _inline_atr14(df, i):
    # Matches InlineATR14(rb, 0) — average of TR over bars [i .. i-13]
    # (MQL5 loops k=0..13, using rb[k] and rb[k+1] for prev-close).
    p = 14
    if i - p < 0: p = i
    if p < 1: return df["high"].iat[i] - df["low"].iat[i]
    s = 0.0
    for k in range(p):
        ki = i - k
        h = df["high"].iat[ki]; l = df["low"].iat[ki]
        pc = df["close"].iat[ki-1]
        tr = max(h - l, abs(h - pc), abs(l - pc))
        s += tr
    return s / p

def _sma_close(df, i, period):
    if i + 1 < period: return df["close"].iat[i]
    s = 0.0
    for k in range(period): s += df["close"].iat[i-k]
    return s / period

def mql_R0f_mean_revert(df, i):
    if i < 21: return 0
    sma20 = _sma_close(df, i, 20)
    atr = _inline_atr14(df, i)
    if atr <= 0: return 0
    dev = df["close"].iat[i] - sma20
    rng = df["high"].iat[i] - df["low"].iat[i]
    if rng <= 0: return 0
    close_pos = (df["close"].iat[i] - df["low"].iat[i]) / rng
    if dev >= 2.0 * atr:
        if close_pos < 0.45 and df["close"].iat[i] < df["open"].iat[i]: return -1
    elif dev <= -2.0 * atr:
        if close_pos > 0.55 and df["close"].iat[i] > df["open"].iat[i]: return +1
    return 0

def mql_R0g_inside_break(df, i):
    if i < 3: return 0
    h1,l1 = df["high"].iat[i-1], df["low"].iat[i-1]
    h2,l2 = df["high"].iat[i-2], df["low"].iat[i-2]
    if not (h1 <= h2 and l1 >= l2): return 0
    c0,o0 = df["close"].iat[i], df["open"].iat[i]
    if c0 > h1 and c0 > o0: return +1
    if c0 < l1 and c0 < o0: return -1
    return 0

def mql_R0h_3bar_reversal(df, i):
    if i < 3: return 0
    rng = df["high"].iat[i] - df["low"].iat[i]
    if rng < 1e-6: return 0
    p1r = df["close"].iat[i-1] < df["open"].iat[i-1]
    p2r = df["close"].iat[i-2] < df["open"].iat[i-2]
    p1g = df["close"].iat[i-1] > df["open"].iat[i-1]
    p2g = df["close"].iat[i-2] > df["open"].iat[i-2]
    c0,o0,l0 = df["close"].iat[i], df["open"].iat[i], df["low"].iat[i]
    if p1r and p2r and c0 > o0 and (c0-l0)/rng > 0.6: return +1
    if p1g and p2g and c0 < o0 and (c0-l0)/rng < 0.4: return -1
    return 0

def mql_R0i_close_extreme(df, i):
    if i < 11: return 0
    rng = df["high"].iat[i] - df["low"].iat[i]
    if rng < 1e-6: return 0
    close_pos = (df["close"].iat[i] - df["low"].iat[i]) / rng
    prior_lo = df["low"].iat[i-1]; prior_hi = df["high"].iat[i-1]
    for k in range(2, 11):
        if df["low"].iat[i-k] < prior_lo: prior_lo = df["low"].iat[i-k]
        if df["high"].iat[i-k] > prior_hi: prior_hi = df["high"].iat[i-k]
    if close_pos > 0.90 and df["low"].iat[i] <= prior_lo: return +1
    if close_pos < 0.10 and df["high"].iat[i] >= prior_hi: return -1
    return 0

def mql_R1a_swinghigh(df, i):
    if i < 6: return 0
    hi_window = df["high"].iat[i]
    for k in range(1, 6):
        if df["high"].iat[i-k] > hi_window: hi_window = df["high"].iat[i-k]
    if df["high"].iat[i] != hi_window: return 0
    rng = df["high"].iat[i] - df["low"].iat[i]
    if rng < 1e-6: return 0
    close_pos = (df["close"].iat[i] - df["low"].iat[i]) / rng
    hh_dist10 = df["hh_dist10"].iat[i]
    if close_pos < 0.5 and hh_dist10 < 0.5: return -1
    return 0

def mql_R1b_lowerhigh(df, i):
    if i < 35: return 0
    local_hi = df["high"].iat[i]
    for k in range(1, 6):
        if df["high"].iat[i-k] > local_hi: local_hi = df["high"].iat[i-k]
    if df["high"].iat[i] != local_hi: return 0
    prior_hi = df["high"].iat[i-5]
    for k in range(6, 31):
        if df["high"].iat[i-k] > prior_hi: prior_hi = df["high"].iat[i-k]
    if df["high"].iat[i] >= prior_hi: return 0
    if df["close"].iat[i] >= df["open"].iat[i]: return 0
    return -1

def mql_R1c_bouncefade(df, i):
    if i < 25: return 0
    bounce = (df["close"].iat[i-3] > df["open"].iat[i-3] and
              df["close"].iat[i-2] > df["open"].iat[i-2] and
              df["close"].iat[i-1] > df["open"].iat[i-1])
    if not bounce: return 0
    if df["close"].iat[i] >= df["open"].iat[i]: return 0
    prior_hi = df["high"].iat[i-3]
    for k in range(4, 21):
        if df["high"].iat[i-k] > prior_hi: prior_hi = df["high"].iat[i-k]
    if df["high"].iat[i] < prior_hi * 0.998: return 0
    mom5 = df["mom5"].iat[i]
    if mom5 <= 0: return 0
    return -1

def mql_R1d_overbought(df, i):
    sk = df["stoch_k"].iat[i]
    prev_sk = df["stoch_k"].iat[i-1] if i >= 1 else 0
    if prev_sk < 0.80: return 0
    if sk >= prev_sk: return 0
    if df["close"].iat[i] >= df["open"].iat[i]: return 0
    return -1

def mql_R1e_false_breakout(df, i):
    if i < 21: return 0
    prior_hi = df["high"].iat[i-1]
    for k in range(2, 21):
        if df["high"].iat[i-k] > prior_hi: prior_hi = df["high"].iat[i-k]
    if df["high"].iat[i] <= prior_hi: return 0
    if df["close"].iat[i] >= prior_hi: return 0
    return -1

def mql_R1f_sma_reject(df, i):
    if i < 21: return 0
    mom20 = df["mom20"].iat[i]
    if mom20 >= 0: return 0
    sma20 = _sma_close(df, i, 20)
    if df["high"].iat[i] < sma20: return 0
    if df["close"].iat[i] >= sma20: return 0
    if df["close"].iat[i] >= df["open"].iat[i]: return 0
    return -1

def mql_R1g_three_red(df, i):
    if i < 3: return 0
    if not (df["close"].iat[i] < df["open"].iat[i] and
            df["close"].iat[i-1] < df["open"].iat[i-1] and
            df["close"].iat[i-2] < df["open"].iat[i-2]): return 0
    mom20 = df["mom20"].iat[i]
    if mom20 >= 0: return 0
    rng = df["high"].iat[i] - df["low"].iat[i]
    if rng < 1e-6: return 0
    if (df["close"].iat[i] - df["low"].iat[i]) / rng > 0.4: return 0
    return -1

def mql_R1h_close_streak(df, i):
    if i < 5: return 0
    if not (df["close"].iat[i] < df["close"].iat[i-1] <
            df["close"].iat[i-2] < df["close"].iat[i-3] <
            df["close"].iat[i-4]): return 0
    mom10 = df["mom10"].iat[i]
    if mom10 >= 0: return 0
    if df["close"].iat[i] >= df["open"].iat[i]: return 0
    return -1

def mql_R3a_pullback(df, i):
    if i < 11: return 0
    if df["close"].iat[i] <= df["open"].iat[i]: return 0
    rng = df["high"].iat[i] - df["low"].iat[i]
    if rng < 1e-6: return 0
    if (df["close"].iat[i] - df["low"].iat[i]) / rng < 0.5: return 0
    if df["close"].iat[i] >= df["close"].iat[i-5] or df["close"].iat[i-5] >= df["close"].iat[i-10]: return 0
    lldist = df["ll_dist10"].iat[i]
    rsi6 = df["rsi6"].iat[i]
    if lldist >= 0.8 or rsi6 >= 0.15: return 0
    return +1

def mql_R3b_higherlow(df, i):
    if i < 35: return 0
    local_lo = df["low"].iat[i]
    for k in range(1, 6):
        if df["low"].iat[i-k] < local_lo: local_lo = df["low"].iat[i-k]
    if df["low"].iat[i] != local_lo: return 0
    prior_lo = df["low"].iat[i-5]
    for k in range(6, 31):
        if df["low"].iat[i-k] < prior_lo: prior_lo = df["low"].iat[i-k]
    if df["low"].iat[i] <= prior_lo: return 0
    if df["close"].iat[i] <= df["open"].iat[i]: return 0
    return +1

def mql_R3c_breakpullback(df, i):
    if i < 35: return 0
    broke = False
    break_level = 0.0
    for j in range(10, 1, -1):
        if i - (j+1) < 0 or i - (j+21) < 0: continue
        prior_hi = df["high"].iat[i-(j+1)]
        for k in range(j+2, j+21):
            if df["high"].iat[i-k] > prior_hi: prior_hi = df["high"].iat[i-k]
        if df["close"].iat[i-j] > prior_hi:
            broke = True
            break_level = prior_hi
            break
    if not broke: return 0
    if df["low"].iat[i] > break_level * 1.003: return 0
    if df["low"].iat[i] < break_level * 0.997: return 0
    if df["close"].iat[i] <= df["open"].iat[i]: return 0
    return +1

def mql_R3d_oversold(df, i):
    sk = df["stoch_k"].iat[i]
    prev_sk = df["stoch_k"].iat[i-1] if i >= 1 else 0
    if prev_sk > 0.20: return 0
    if sk <= prev_sk: return 0
    if df["close"].iat[i] <= df["open"].iat[i]: return 0
    mom20 = df["mom20"].iat[i]
    if mom20 <= 0: return 0
    return +1

def mql_R3e_false_breakdown(df, i):
    if i < 21: return 0
    prior_lo = df["low"].iat[i-1]
    for k in range(2, 21):
        if df["low"].iat[i-k] < prior_lo: prior_lo = df["low"].iat[i-k]
    if df["low"].iat[i] >= prior_lo: return 0
    if df["close"].iat[i] <= prior_lo: return 0
    rng = df["high"].iat[i] - df["low"].iat[i]
    if rng < 1e-6: return 0
    midpoint = df["low"].iat[i] + (df["close"].iat[i] - df["low"].iat[i]) * 0.5
    if df["close"].iat[i] <= midpoint: return 0
    return +1

def mql_R3f_sma_bounce(df, i):
    if i < 21: return 0
    mom20 = df["mom20"].iat[i]
    if mom20 <= 0: return 0
    sma20 = _sma_close(df, i, 20)
    if df["low"].iat[i] > sma20: return 0
    if df["close"].iat[i] <= sma20: return 0
    if df["close"].iat[i] <= df["open"].iat[i]: return 0
    return +1

def mql_R3g_three_green(df, i):
    if i < 3: return 0
    if not (df["close"].iat[i] > df["open"].iat[i] and
            df["close"].iat[i-1] > df["open"].iat[i-1] and
            df["close"].iat[i-2] > df["open"].iat[i-2]): return 0
    mom20 = df["mom20"].iat[i]
    if mom20 <= 0: return 0
    rng = df["high"].iat[i] - df["low"].iat[i]
    if rng < 1e-6: return 0
    if (df["close"].iat[i] - df["low"].iat[i]) / rng < 0.6: return 0
    return +1

def mql_R3h_close_streak(df, i):
    if i < 5: return 0
    if not (df["close"].iat[i] > df["close"].iat[i-1] >
            df["close"].iat[i-2] > df["close"].iat[i-3] >
            df["close"].iat[i-4]): return 0
    mom10 = df["mom10"].iat[i]
    if mom10 <= 0: return 0
    if df["close"].iat[i] <= df["open"].iat[i]: return 0
    return +1

def mql_R3i_inside_break(df, i):
    if i < 3: return 0
    h1,l1 = df["high"].iat[i-1], df["low"].iat[i-1]
    h2,l2 = df["high"].iat[i-2], df["low"].iat[i-2]
    if not (h1 <= h2 and l1 >= l2): return 0
    if df["close"].iat[i] <= h1: return 0
    if df["close"].iat[i] <= df["open"].iat[i]: return 0
    mom20 = df["mom20"].iat[i]
    if mom20 <= 0: return 0
    return +1


# ─── Driver: scan every bar with each MQL5 rule, no cooldown ────────────
MQL_RULES = [
    ("R0a_bb", mql_R0a_bb), ("R0b_stoch", mql_R0b_stoch),
    ("R0c_doubletouch", mql_R0c_doubletouch), ("R0d_squeeze", mql_R0d_squeeze),
    ("R0e_nr4_break", mql_R0e_nr4_break), ("R0f_mean_revert", mql_R0f_mean_revert),
    ("R0g_inside_break", mql_R0g_inside_break), ("R0h_3bar_reversal", mql_R0h_3bar_reversal),
    ("R0i_close_extreme", mql_R0i_close_extreme),
    ("R1a_swinghigh", mql_R1a_swinghigh), ("R1b_lowerhigh", mql_R1b_lowerhigh),
    ("R1c_bouncefade", mql_R1c_bouncefade), ("R1d_overbought", mql_R1d_overbought),
    ("R1e_false_breakout", mql_R1e_false_breakout), ("R1f_sma_reject", mql_R1f_sma_reject),
    ("R1g_three_red", mql_R1g_three_red), ("R1h_close_streak", mql_R1h_close_streak),
    ("R3a_pullback", mql_R3a_pullback), ("R3b_higherlow", mql_R3b_higherlow),
    ("R3c_breakpullback", mql_R3c_breakpullback), ("R3d_oversold", mql_R3d_oversold),
    ("R3e_false_breakdown", mql_R3e_false_breakdown), ("R3f_sma_bounce", mql_R3f_sma_bounce),
    ("R3g_three_green", mql_R3g_three_green), ("R3h_close_streak", mql_R3h_close_streak),
    ("R3i_inside_break", mql_R3i_inside_break),
]

PY_RULES = [
    ("R0a_bb",              setup_py.rule_ranging_bb),
    ("R0b_stoch",           setup_py.rule_ranging_stoch),
    ("R0c_doubletouch",     setup_py.rule_ranging_double_touch),
    ("R0d_squeeze",         setup_py.rule_ranging_squeeze),
    ("R0e_nr4_break",       setup_py.rule_ranging_nr4_breakout),
    ("R0f_mean_revert",     setup_py.rule_ranging_mean_revert),
    ("R0g_inside_break",    setup_py.rule_ranging_inside_break),
    ("R0h_3bar_reversal",   setup_py.rule_ranging_three_bar_reversal),
    ("R0i_close_extreme",   setup_py.rule_ranging_close_extreme),
    ("R1a_swinghigh",       setup_py.rule_downtrend_swing_high),
    ("R1b_lowerhigh",       setup_py.rule_downtrend_lower_high),
    ("R1c_bouncefade",      setup_py.rule_downtrend_bounce_fade),
    ("R1d_overbought",      setup_py.rule_downtrend_overbought),
    ("R1e_false_breakout",  setup_py.rule_downtrend_false_breakout),
    ("R1f_sma_reject",      setup_py.rule_downtrend_sma_rejection),
    ("R1g_three_red",       setup_py.rule_downtrend_three_red),
    ("R1h_close_streak",    setup_py.rule_downtrend_lower_close_streak),
    ("R3a_pullback",        setup_py.rule_uptrend_pullback),
    ("R3b_higherlow",       setup_py.rule_uptrend_higher_low),
    ("R3c_breakpullback",   setup_py.rule_uptrend_breakout_pullback),
    ("R3d_oversold",        setup_py.rule_uptrend_oversold),
    ("R3e_false_breakdown", setup_py.rule_uptrend_false_breakdown),
    ("R3f_sma_bounce",      setup_py.rule_uptrend_pullback_sma),
    ("R3g_three_green",     setup_py.rule_uptrend_three_green),
    ("R3h_close_streak",    setup_py.rule_uptrend_higher_close_streak),
    ("R3i_inside_break",    setup_py.rule_uptrend_inside_bar_break),
]


def load_df():
    raw = open(CSV, "rb").read()
    txt = raw.decode("utf-16-le").lstrip("\ufeff")
    import io
    df = pd.read_csv(io.StringIO(txt))
    df["time"] = pd.to_datetime(df["time"], format="%Y.%m.%d %H:%M")
    df = df.sort_values("time").reset_index(drop=True)
    # MQL5 f32_ATR_RATIO maps to our "atr_ratio" — the exporter column.
    # The exporter columns are f01..f20 (uppercase); Python uses lowercase
    # derived names (rsi6, stoch_k, bb_pct, mom5, mom20, ll_dist10, hh_dist10,
    # atr_ratio). We compute these from OHLC using the labeler's helper.
    df = tech.compute_tech_features(df)
    return df


# Python training rules use per-rule cooldowns; the EA enforces the same at
# runtime via RULE_COOLDOWN[] (all 8 bars). Apply the SAME cooldown here so
# both scanners are comparable. The Python training script uses 8 for most,
# 10-12 for a few rules — we match those specific values.
RULE_COOLDOWN_BARS = {
    "R0a_bb": 10, "R0b_stoch": 8, "R0c_doubletouch": 10, "R0d_squeeze": 12,
    "R0e_nr4_break": 10, "R0f_mean_revert": 10, "R0g_inside_break": 8,
    "R0h_3bar_reversal": 8, "R0i_close_extreme": 8,
    "R1a_swinghigh": 8, "R1b_lowerhigh": 8, "R1c_bouncefade": 10,
    "R1d_overbought": 8, "R1e_false_breakout": 10, "R1f_sma_reject": 10,
    "R1g_three_red": 8, "R1h_close_streak": 10,
    "R3a_pullback": 8, "R3b_higherlow": 8, "R3c_breakpullback": 10,
    "R3d_oversold": 8, "R3e_false_breakdown": 10, "R3f_sma_bounce": 10,
    "R3g_three_green": 8, "R3h_close_streak": 10, "R3i_inside_break": 10,
}

def scan_mql(df, rule_name, fn):
    out = []
    last_fire = -1000
    cool = RULE_COOLDOWN_BARS.get(rule_name, 8)
    for i in range(30, len(df) - MAX_FWD - 1):
        if i - last_fire < cool: continue
        d = fn(df, i)
        if d != 0:
            out.append((df["time"].iat[i], d))
            last_fire = i
    return out


def scan_py(df, rule_name, fn):
    out = []
    for ev in fn(df):
        out.append((df["time"].iat[ev["idx"]], ev["direction"]))
    return out


def diff_sets(a, b):
    """a, b are sets of (time, dir). Return (only_a, only_b, common)."""
    sa, sb = set(a), set(b)
    return (sa - sb, sb - sa, sa & sb)


def main():
    print("="*95)
    print("MQL5 ↔ Python rule parity audit (no cooldown, no meta — raw rule firings)")
    print("="*95)
    df = load_df()
    today = df["time"].dt.date.max()
    print(f"  Loaded {len(df):,} bars  range {df['time'].iat[0]} → {df['time'].iat[-1]}")

    # Map python rule-name → events list; mql5 the same
    mql_events = {name: scan_mql(df, name, fn) for name, fn in MQL_RULES}
    py_events  = {name: scan_py(df, name, fn) for name, fn in PY_RULES}

    print(f"\n{'Rule':<22} {'MQL5':>8} {'Python':>8} {'only MQL':>9} {'only Py':>9} {'common':>7}  status")
    print("-"*95)
    divergent = []
    for name, _ in MQL_RULES:
        mql = mql_events.get(name, [])
        py  = py_events.get(name, [])
        if not py and name in ("R0h_3bar_reversal",):
            # some rules exist only in MQL5 (disabled detectors)
            status = "MQL-only"
            print(f"  {name:<22} {len(mql):>6}   {'-':>8}  {'-':>9} {'-':>9} {'-':>7}  {status}")
            continue
        if not mql:
            status = "Py-only"
            print(f"  {name:<22} {'-':>8} {len(py):>6}   {'-':>9} {'-':>9} {'-':>7}  {status}")
            continue
        only_mql, only_py, common = diff_sets(mql, py)
        status = "MATCH" if not only_mql and not only_py else "DIFFER"
        if status == "DIFFER":
            divergent.append((name, only_mql, only_py, common))
        print(f"  {name:<22} {len(mql):>8} {len(py):>8} {len(only_mql):>9} {len(only_py):>9} {len(common):>7}  {status}")

    print("-"*95)
    print(f"\nDivergent rules: {len(divergent)} / {len(MQL_RULES)}")

    # Show first 3 example divergences per rule
    for name, only_mql, only_py, common in divergent[:10]:
        print(f"\n  {name}:")
        if only_mql:
            sample = sorted(only_mql)[:3]
            print(f"    MQL-only ({len(only_mql)} bars): e.g. {[(t.strftime('%m-%d %H:%M'), d) for t,d in sample]}")
        if only_py:
            sample = sorted(only_py)[:3]
            print(f"    Py-only  ({len(only_py)} bars): e.g. {[(t.strftime('%m-%d %H:%M'), d) for t,d in sample]}")

    # Focus: which of today's bars had an MQL5 candidate that Python missed?
    # (This is exactly what caused zero BTC trades today.)
    print(f"\n\nToday's ({today}) MQL5-only firings per rule:")
    for name, only_mql, _, _ in divergent:
        todays = [(t, d) for t, d in only_mql if t.date() == today]
        if todays:
            for t, d in sorted(todays):
                print(f"  {t.strftime('%H:%M')}  {name:<22} dir={d:+d}")


if __name__ == "__main__":
    main()
