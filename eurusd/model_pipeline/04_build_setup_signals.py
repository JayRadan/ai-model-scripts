"""
EURUSD M5 — Build setup signals v2 (expanded rules, lower cooldowns).
Target: 5-10 trades/day on holdout.
"""
import numpy as np
import pandas as pd
import paths as P

CLUSTER_NAMES = {0: "Uptrend", 1: "MeanRevert", 2: "TrendRange", 3: "Downtrend", 4: "HighVol"}
TP_MULT = 2.0
SL_MULT = 1.0
MAX_FWD = 40
SPREAD_PIPS = 1.5
POINT = 0.00001
spread_price = SPREAD_PIPS * POINT

FEATURE_COLS = [
    "f01_CPR", "f02_WickAsym", "f03_BEF", "f04_TCS", "f05_SPI",
    "f06_LRSlope", "f07_RECR", "f08_SCM", "f09_HLER", "f10_EP",
    "f11_KE", "f12_MCS", "f13_Work", "f14_EDR", "f15_AI",
    "f16_PPShigh", "f16_PPSlow", "f17_SCR", "f18_RVD", "f19_WBER",
    "f20_NCDE",
]

def add_tech(df):
    c = df["close"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    o = df["open"].values.astype(np.float64)
    n = len(c)

    tr = np.empty(n); tr[0] = h[0] - l[0]
    tr[1:] = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    atr14 = pd.Series(tr).rolling(14, min_periods=1).mean().values

    sma20 = pd.Series(c).rolling(20, min_periods=1).mean().values
    sma50 = pd.Series(c).rolling(50, min_periods=1).mean().values
    std20 = np.nan_to_num(pd.Series(c).rolling(20, min_periods=1).std().values, nan=1e-10)
    bb_up = sma20 + 2 * std20; bb_lo = sma20 - 2 * std20
    bb_w = bb_up - bb_lo
    bb_pct = np.where(bb_w > 1e-10, (c - bb_lo) / bb_w, 0.5)

    hi14 = pd.Series(h).rolling(14, min_periods=1).max().values
    lo14 = pd.Series(l).rolling(14, min_periods=1).min().values
    raw_k = np.where(hi14 - lo14 > 1e-10, (c - lo14) / (hi14 - lo14), 0.5)
    stoch_k = pd.Series(raw_k).rolling(3, min_periods=1).mean().values

    delta = np.diff(c, prepend=c[0])
    gain = np.where(delta > 0, delta, 0.0); loss_v = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=14, adjust=False).mean().values
    avg_l = pd.Series(loss_v).ewm(span=14, adjust=False).mean().values
    rsi = 100 - 100 / (1 + avg_g / np.maximum(avg_l, 1e-10))

    bar_range = h - l
    range_atr = np.where(atr14 > 1e-10, bar_range / atr14, 1.0)
    dist_sma20 = np.where(atr14 > 1e-10, (c - sma20) / atr14, 0.0)
    dist_sma50 = np.where(atr14 > 1e-10, (c - sma50) / atr14, 0.0)
    body = np.abs(c - o)
    body_ratio = np.where(bar_range > 1e-10, body / bar_range, 0.0)
    bullish = c > o
    upper_wick = h - np.maximum(c, o)
    lower_wick = np.minimum(c, o) - l
    wick_ratio_up = np.where(bar_range > 1e-10, upper_wick / bar_range, 0.0)
    wick_ratio_lo = np.where(bar_range > 1e-10, lower_wick / bar_range, 0.0)

    direction = np.sign(c - o)
    consec = np.zeros(n)
    for i in range(1, n):
        if direction[i] == direction[i-1] and direction[i] != 0:
            consec[i] = consec[i-1] + direction[i]
        else:
            consec[i] = direction[i]

    hours = df["time"].dt.hour.values
    dow = df["time"].dt.dayofweek.values
    bb_w_ma5 = pd.Series(bb_w).rolling(5, min_periods=1).mean().values
    hi10 = pd.Series(h).rolling(10, min_periods=1).max().values
    lo10 = pd.Series(l).rolling(10, min_periods=1).min().values
    hi20 = pd.Series(h).rolling(20, min_periods=5).max().values
    lo20 = pd.Series(l).rolling(20, min_periods=5).min().values

    # EMA-12 for faster trend
    ema12 = pd.Series(c).ewm(span=12, adjust=False).mean().values

    return {
        "c": c, "h": h, "l": l, "o": o,
        "atr14": atr14, "sma20": sma20, "sma50": sma50, "ema12": ema12,
        "bb_up": bb_up, "bb_lo": bb_lo, "bb_pct": bb_pct, "bb_w": bb_w, "bb_w_ma5": bb_w_ma5,
        "stoch_k": stoch_k, "rsi": rsi,
        "range_atr": range_atr, "dist_sma20": dist_sma20, "dist_sma50": dist_sma50,
        "body_ratio": body_ratio, "bullish": bullish, "bar_range": bar_range,
        "body": body, "upper_wick": upper_wick, "lower_wick": lower_wick,
        "wick_ratio_up": wick_ratio_up, "wick_ratio_lo": wick_ratio_lo,
        "consec": consec, "hours": hours, "dow": dow, "direction": direction,
        "hi10": hi10, "lo10": lo10, "hi20": hi20, "lo20": lo20,
    }

EXTRA_FEAT_COLS = [
    "stoch_k", "rsi14", "bb_pct", "vol_ratio", "range_atr",
    "dist_sma20", "dist_sma50", "body_ratio", "consec_dir",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]
HTF_FEAT_COLS = [
    "h1_trend_sma20", "h1_trend_sma50", "h1_slope5", "h1_rsi14",
    "h1_atr_ratio", "h1_dist_sma20", "h1_dist_sma50",
    "h4_trend_sma20", "h4_trend_sma50", "h4_slope5", "h4_rsi14",
    "h4_atr_ratio", "h4_dist_sma20", "h4_dist_sma50",
]
ALL_FEAT_COLS = FEATURE_COLS + EXTRA_FEAT_COLS

def compute_extra_feats(df, t):
    vol = np.abs(np.diff(t["c"], prepend=t["c"][0]))
    vol_ma = pd.Series(np.maximum(vol, 1e-10)).rolling(20, min_periods=1).mean().values
    return {
        "stoch_k": t["stoch_k"], "rsi14": t["rsi"], "bb_pct": t["bb_pct"],
        "vol_ratio": np.where(vol_ma > 1e-10, vol / vol_ma, 1.0),
        "range_atr": t["range_atr"],
        "dist_sma20": t["dist_sma20"], "dist_sma50": t["dist_sma50"],
        "body_ratio": t["body_ratio"], "consec_dir": t["consec"],
        "hour_sin": np.sin(2 * np.pi * t["hours"] / 24),
        "hour_cos": np.cos(2 * np.pi * t["hours"] / 24),
        "dow_sin": np.sin(2 * np.pi * t["dow"] / 5),
        "dow_cos": np.cos(2 * np.pi * t["dow"] / 5),
    }

def forward_outcome(t, i, direction):
    tp_d = TP_MULT * t["atr14"][i]; sl_d = SL_MULT * t["atr14"][i]; n = len(t["c"])
    if direction == "buy":
        tp_p = t["c"][i] + tp_d + spread_price; sl_p = t["c"][i] - sl_d
        for k in range(1, min(MAX_FWD + 1, n - i)):
            if t["h"][i+k] >= tp_p: return 1
            if t["l"][i+k] <= sl_p: return 0
    else:
        tp_p = t["c"][i] - tp_d - spread_price; sl_p = t["c"][i] + sl_d
        for k in range(1, min(MAX_FWD + 1, n - i)):
            if t["l"][i+k] <= tp_p: return 1
            if t["h"][i+k] >= sl_p: return 0
    return 0

# ═══════════════════════════════════════════════════════════════
# SESSION HELPERS
# ═══════════════════════════════════════════════════════════════
def is_london(t, i): return 7 <= t["hours"][i] <= 16
def is_ny(t, i): return 13 <= t["hours"][i] <= 21
def is_london_open(t, i): return 7 <= t["hours"][i] <= 10
def is_ny_open(t, i): return 13 <= t["hours"][i] <= 16

# ═══════════════════════════════════════════════════════════════
# C0 UPTREND RULES (buy only) — 12 rules
# ═══════════════════════════════════════════════════════════════
def r0a_pullback(t, i):
    return (t["dist_sma20"][i] > -0.5 and t["dist_sma20"][i] < 0.5 and
            t["dist_sma50"][i] > 0 and t["bullish"][i])

def r0b_higher_low(t, i):
    if i < 10: return False
    return min(t["l"][i-k] for k in range(3, 10)) < t["l"][i] and t["bullish"][i] and t["stoch_k"][i] < 0.5

def r0c_breakout_pb(t, i):
    if i < 20: return False
    return t["h"][i-1] > t["hi20"][i-2] and t["c"][i] > t["sma20"][i] and t["bullish"][i]

def r0d_oversold(t, i):
    return t["rsi"][i] < 35 and t["dist_sma50"][i] > 0 and t["bullish"][i]

def r0e_sma_bounce(t, i):
    a = t["atr14"][i]
    return a > 1e-10 and abs(t["l"][i] - t["sma50"][i]) < 0.5 * a and t["c"][i] > t["sma50"][i] and t["bullish"][i]

def r0f_false_breakdown(t, i):
    if i < 10: return False
    support = min(t["l"][i-k] for k in range(2, 10))
    return t["l"][i] < support and t["c"][i] > support and t["bullish"][i]

def r0g_close_streak(t, i):
    return t["consec"][i] >= 3

def r0h_pin_bar(t, i):
    return t["wick_ratio_lo"][i] > 0.6 and t["body_ratio"][i] < 0.25 and t["bullish"][i]

def r0i_engulfing(t, i):
    if i < 1: return False
    return (not t["bullish"][i-1] and t["bullish"][i] and
            t["o"][i] <= t["c"][i-1] and t["c"][i] >= t["o"][i-1] and
            t["body_ratio"][i] > 0.5)

def r0j_ema_pullback(t, i):
    a = t["atr14"][i]
    return (a > 1e-10 and abs(t["l"][i] - t["ema12"][i]) < 0.3 * a and
            t["c"][i] > t["ema12"][i] and t["bullish"][i] and t["dist_sma50"][i] > 0)

def r0k_london_buy(t, i):
    return is_london_open(t, i) and t["bullish"][i] and t["dist_sma20"][i] > 0 and t["stoch_k"][i] < 0.4

def r0l_doji_reversal(t, i):
    if i < 1: return False
    return (t["body_ratio"][i-1] < 0.15 and not t["bullish"][i-1] and
            t["bullish"][i] and t["body_ratio"][i] > 0.4 and t["stoch_k"][i] < 0.35)

# ═══════════════════════════════════════════════════════════════
# C1 RANGING RULES (both) — 14 rules
# ═══════════════════════════════════════════════════════════════
def r1a_bb_buy(t, i): return t["bb_pct"][i] < 0.05 and t["stoch_k"][i] < 0.25
def r1a_bb_sell(t, i): return t["bb_pct"][i] > 0.95 and t["stoch_k"][i] > 0.75

def r1b_stoch_buy(t, i): return t["stoch_k"][i] < 0.15 and t["bullish"][i] and t["body_ratio"][i] > 0.4
def r1b_stoch_sell(t, i): return t["stoch_k"][i] > 0.85 and not t["bullish"][i] and t["body_ratio"][i] > 0.4

def r1c_inside_buy(t, i):
    if i < 2: return False
    return t["h"][i-1] < t["h"][i-2] and t["l"][i-1] > t["l"][i-2] and t["c"][i] > t["h"][i-1]
def r1c_inside_sell(t, i):
    if i < 2: return False
    return t["h"][i-1] < t["h"][i-2] and t["l"][i-1] > t["l"][i-2] and t["c"][i] < t["l"][i-1]

def r1d_squeeze_buy(t, i):
    if i < 5: return False
    return t["bb_w"][i] < t["bb_w_ma5"][i] * 0.7 and t["c"][i] > t["sma20"][i]
def r1d_squeeze_sell(t, i):
    if i < 5: return False
    return t["bb_w"][i] < t["bb_w_ma5"][i] * 0.7 and t["c"][i] < t["sma20"][i]

def r1e_double_buy(t, i):
    if i < 20: return False
    return t["bb_pct"][i] < 0.10 and any(t["bb_pct"][i-k] < 0.10 for k in range(5, min(20, i))) and t["rsi"][i] > t["rsi"][i-1]
def r1e_double_sell(t, i):
    if i < 20: return False
    return t["bb_pct"][i] > 0.90 and any(t["bb_pct"][i-k] > 0.90 for k in range(5, min(20, i))) and t["rsi"][i] < t["rsi"][i-1]

def r1f_mean_buy(t, i): return t["dist_sma20"][i] < -2.0 and t["bullish"][i]
def r1f_mean_sell(t, i): return t["dist_sma20"][i] > 2.0 and not t["bullish"][i]

def r1g_extreme_buy(t, i): return t["bb_pct"][i] < 0.08 and t["bullish"][i] and t["body_ratio"][i] > 0.5
def r1g_extreme_sell(t, i): return t["bb_pct"][i] > 0.92 and not t["bullish"][i] and t["body_ratio"][i] > 0.5

def r1h_nr4_buy(t, i):
    if i < 4: return False
    return all(t["bar_range"][i] < t["bar_range"][i-k] for k in range(1, 4)) and t["c"][i] > t["h"][i-1]
def r1h_nr4_sell(t, i):
    if i < 4: return False
    return all(t["bar_range"][i] < t["bar_range"][i-k] for k in range(1, 4)) and t["c"][i] < t["l"][i-1]

def r1i_pin_buy(t, i): return t["wick_ratio_lo"][i] > 0.6 and t["body_ratio"][i] < 0.25 and t["bb_pct"][i] < 0.3
def r1i_pin_sell(t, i): return t["wick_ratio_up"][i] > 0.6 and t["body_ratio"][i] < 0.25 and t["bb_pct"][i] > 0.7

def r1j_engulf_buy(t, i):
    if i < 1: return False
    return not t["bullish"][i-1] and t["bullish"][i] and t["o"][i] <= t["c"][i-1] and t["c"][i] >= t["o"][i-1] and t["bb_pct"][i] < 0.3
def r1j_engulf_sell(t, i):
    if i < 1: return False
    return t["bullish"][i-1] and not t["bullish"][i] and t["o"][i] >= t["c"][i-1] and t["c"][i] <= t["o"][i-1] and t["bb_pct"][i] > 0.7

def r1k_rsi_div_buy(t, i):
    if i < 10: return False
    return t["rsi"][i] < 35 and t["rsi"][i] > t["rsi"][i-5] and t["l"][i] < t["l"][i-5] and t["bullish"][i]
def r1k_rsi_div_sell(t, i):
    if i < 10: return False
    return t["rsi"][i] > 65 and t["rsi"][i] < t["rsi"][i-5] and t["h"][i] > t["h"][i-5] and not t["bullish"][i]

def r1l_session_break_buy(t, i):
    if i < 12 or not is_london_open(t, i): return False
    asian_hi = max(t["h"][i-k] for k in range(1, min(12, i)))
    return t["c"][i] > asian_hi and t["bullish"][i]
def r1l_session_break_sell(t, i):
    if i < 12 or not is_london_open(t, i): return False
    asian_lo = min(t["l"][i-k] for k in range(1, min(12, i)))
    return t["c"][i] < asian_lo and not t["bullish"][i]

def r1m_3bar_rev_buy(t, i):
    if i < 3: return False
    return (not t["bullish"][i-2] and not t["bullish"][i-1] and t["bullish"][i] and
            t["c"][i] > t["h"][i-1] and t["stoch_k"][i] < 0.4)
def r1m_3bar_rev_sell(t, i):
    if i < 3: return False
    return (t["bullish"][i-2] and t["bullish"][i-1] and not t["bullish"][i] and
            t["c"][i] < t["l"][i-1] and t["stoch_k"][i] > 0.6)

def r1n_range_fade_buy(t, i):
    if i < 20: return False
    return t["c"][i] <= t["lo20"][i] * 1.001 and t["bullish"][i] and t["rsi"][i] < 40
def r1n_range_fade_sell(t, i):
    if i < 20: return False
    return t["c"][i] >= t["hi20"][i] * 0.999 and not t["bullish"][i] and t["rsi"][i] > 60

# ═══════════════════════════════════════════════════════════════
# C2 DOWNTREND RULES (sell only) — 12 rules
# ═══════════════════════════════════════════════════════════════
def r2a_swing_high(t, i):
    if i < 10: return False
    return t["h"][i] >= max(t["h"][i-k] for k in range(3, 10)) * 0.999 and not t["bullish"][i]

def r2b_lower_high(t, i):
    if i < 10: return False
    return t["h"][i] < max(t["h"][i-k] for k in range(3, 10)) and not t["bullish"][i] and t["stoch_k"][i] > 0.5

def r2c_bounce_fade(t, i):
    return t["dist_sma20"][i] > 0 and t["dist_sma20"][i] < 1.0 and t["dist_sma50"][i] < 0 and not t["bullish"][i]

def r2d_overbought(t, i):
    return t["rsi"][i] > 65 and t["dist_sma50"][i] < 0 and not t["bullish"][i]

def r2e_sma_reject(t, i):
    a = t["atr14"][i]
    return a > 1e-10 and abs(t["h"][i] - t["sma50"][i]) < 0.5 * a and t["c"][i] < t["sma50"][i] and not t["bullish"][i]

def r2f_three_red(t, i): return t["consec"][i] <= -3

def r2g_false_breakout(t, i):
    if i < 10: return False
    resistance = max(t["h"][i-k] for k in range(2, 10))
    return t["h"][i] > resistance and t["c"][i] < resistance and not t["bullish"][i]

def r2h_pin_bar(t, i):
    return t["wick_ratio_up"][i] > 0.6 and t["body_ratio"][i] < 0.25 and not t["bullish"][i]

def r2i_engulfing(t, i):
    if i < 1: return False
    return (t["bullish"][i-1] and not t["bullish"][i] and
            t["o"][i] >= t["c"][i-1] and t["c"][i] <= t["o"][i-1] and t["body_ratio"][i] > 0.5)

def r2j_ema_reject(t, i):
    a = t["atr14"][i]
    return (a > 1e-10 and abs(t["h"][i] - t["ema12"][i]) < 0.3 * a and
            t["c"][i] < t["ema12"][i] and not t["bullish"][i] and t["dist_sma50"][i] < 0)

def r2k_london_sell(t, i):
    return is_london_open(t, i) and not t["bullish"][i] and t["dist_sma20"][i] < 0 and t["stoch_k"][i] > 0.6

def r2l_doji_cont(t, i):
    if i < 1: return False
    return (t["body_ratio"][i-1] < 0.15 and t["bullish"][i-1] and
            not t["bullish"][i] and t["body_ratio"][i] > 0.4 and t["stoch_k"][i] > 0.65)

# ═══════════════════════════════════════════════════════════════
# C3 HIGHVOL RULES (both) — 10 rules
# ═══════════════════════════════════════════════════════════════
def r3a_vrev_buy(t, i):
    if i < 3: return False
    return t["range_atr"][i-1] > 1.5 and not t["bullish"][i-1] and t["bullish"][i] and t["body_ratio"][i] > 0.5
def r3a_vrev_sell(t, i):
    if i < 3: return False
    return t["range_atr"][i-1] > 1.5 and t["bullish"][i-1] and not t["bullish"][i] and t["body_ratio"][i] > 0.5

def r3b_mom_buy(t, i): return t["range_atr"][i] > 1.3 and t["bullish"][i] and t["bb_pct"][i] > 0.75
def r3b_mom_sell(t, i): return t["range_atr"][i] > 1.3 and not t["bullish"][i] and t["bb_pct"][i] < 0.25

def r3c_bb_buy(t, i): return r1a_bb_buy(t, i)
def r3c_bb_sell(t, i): return r1a_bb_sell(t, i)
def r3d_stoch_buy(t, i): return r1b_stoch_buy(t, i)
def r3d_stoch_sell(t, i): return r1b_stoch_sell(t, i)
def r3e_inside_buy(t, i): return r1c_inside_buy(t, i)
def r3e_inside_sell(t, i): return r1c_inside_sell(t, i)
def r3f_squeeze_buy(t, i): return r1d_squeeze_buy(t, i)
def r3f_squeeze_sell(t, i): return r1d_squeeze_sell(t, i)

def r3g_pin_buy(t, i): return t["wick_ratio_lo"][i] > 0.6 and t["body_ratio"][i] < 0.25
def r3g_pin_sell(t, i): return t["wick_ratio_up"][i] > 0.6 and t["body_ratio"][i] < 0.25

def r3h_engulf_buy(t, i):
    if i < 1: return False
    return not t["bullish"][i-1] and t["bullish"][i] and t["o"][i] <= t["c"][i-1] and t["c"][i] >= t["o"][i-1]
def r3h_engulf_sell(t, i):
    if i < 1: return False
    return t["bullish"][i-1] and not t["bullish"][i] and t["o"][i] >= t["c"][i-1] and t["c"][i] <= t["o"][i-1]

def r3i_spike_fade_buy(t, i):
    if i < 2: return False
    return t["range_atr"][i-1] > 2.0 and not t["bullish"][i-1] and t["bullish"][i] and t["rsi"][i] < 30
def r3i_spike_fade_sell(t, i):
    if i < 2: return False
    return t["range_atr"][i-1] > 2.0 and t["bullish"][i-1] and not t["bullish"][i] and t["rsi"][i] > 70

def r3j_session_vol_buy(t, i):
    return is_ny_open(t, i) and t["range_atr"][i] > 1.2 and t["bullish"][i] and t["stoch_k"][i] < 0.4
def r3j_session_vol_sell(t, i):
    return is_ny_open(t, i) and t["range_atr"][i] > 1.2 and not t["bullish"][i] and t["stoch_k"][i] > 0.6


# ═══════════════════════════════════════════════════════════════
# RULE REGISTRY — 48 rules total, reduced cooldowns
# ═══════════════════════════════════════════════════════════════
RULES = [
    # C0 Uptrend (buy only) — 12 rules
    ("R0a_pullback",       0, "buy", r0a_pullback, None, 3),
    ("R0b_higher_low",     0, "buy", r0b_higher_low, None, 3),
    ("R0c_breakout_pb",    0, "buy", r0c_breakout_pb, None, 3),
    ("R0d_oversold",       0, "buy", r0d_oversold, None, 3),
    ("R0e_sma_bounce",     0, "buy", r0e_sma_bounce, None, 3),
    ("R0f_false_breakdown",0, "buy", r0f_false_breakdown, None, 3),
    ("R0g_close_streak",   0, "buy", r0g_close_streak, None, 3),
    ("R0h_pin_bar",        0, "buy", r0h_pin_bar, None, 3),
    ("R0i_engulfing",      0, "buy", r0i_engulfing, None, 3),
    ("R0j_ema_pullback",   0, "buy", r0j_ema_pullback, None, 3),
    ("R0k_london_buy",     0, "buy", r0k_london_buy, None, 3),
    ("R0l_doji_reversal",  0, "buy", r0l_doji_reversal, None, 3),
    # C1 MeanRevert (both) — 8 fade/reversal rules
    ("R1a_bb",             1, "both", r1a_bb_buy, r1a_bb_sell, 3),
    ("R1b_stoch",          1, "both", r1b_stoch_buy, r1b_stoch_sell, 3),
    ("R1e_double_touch",   1, "both", r1e_double_buy, r1e_double_sell, 3),
    ("R1f_mean_revert",    1, "both", r1f_mean_buy, r1f_mean_sell, 3),
    ("R1g_close_extreme",  1, "both", r1g_extreme_buy, r1g_extreme_sell, 3),
    ("R1i_pin_bar",        1, "both", r1i_pin_buy, r1i_pin_sell, 3),
    ("R1j_engulfing",      1, "both", r1j_engulf_buy, r1j_engulf_sell, 3),
    ("R1n_range_fade",     1, "both", r1n_range_fade_buy, r1n_range_fade_sell, 3),
    # C2 TrendRange (both) — 6 breakout/momentum rules
    ("R2a_inside_break",   2, "both", r1c_inside_buy, r1c_inside_sell, 3),
    ("R2b_squeeze",        2, "both", r1d_squeeze_buy, r1d_squeeze_sell, 3),
    ("R2c_nr4_break",      2, "both", r1h_nr4_buy, r1h_nr4_sell, 3),
    ("R2d_3bar_reversal",  2, "both", r1m_3bar_rev_buy, r1m_3bar_rev_sell, 3),
    ("R2e_session_break",  2, "both", r1l_session_break_buy, r1l_session_break_sell, 3),
    ("R2f_rsi_divergence", 2, "both", r1k_rsi_div_buy, r1k_rsi_div_sell, 3),
    # C3 Downtrend (sell only) — 12 rules
    ("R3a_swing_high",     3, "sell", None, r2a_swing_high, 3),
    ("R3b_lower_high",     3, "sell", None, r2b_lower_high, 3),
    ("R3c_bounce_fade",    3, "sell", None, r2c_bounce_fade, 3),
    ("R3d_overbought",     3, "sell", None, r2d_overbought, 3),
    ("R3e_sma_reject",     3, "sell", None, r2e_sma_reject, 3),
    ("R3f_three_red",      3, "sell", None, r2f_three_red, 3),
    ("R3g_false_breakout", 3, "sell", None, r2g_false_breakout, 3),
    ("R3h_pin_bar",        3, "sell", None, r2h_pin_bar, 3),
    ("R3i_engulfing",      3, "sell", None, r2i_engulfing, 3),
    ("R3j_ema_reject",     3, "sell", None, r2j_ema_reject, 3),
    ("R3k_london_sell",    3, "sell", None, r2k_london_sell, 3),
    ("R3l_doji_cont",      3, "sell", None, r2l_doji_cont, 3),
    # C4 HighVol (both) — 10 rules
    ("R4a_v_reversal",     4, "both", r3a_vrev_buy, r3a_vrev_sell, 3),
    ("R4b_momentum",       4, "both", r3b_mom_buy, r3b_mom_sell, 3),
    ("R4c_bb_vol",         4, "both", r3c_bb_buy, r3c_bb_sell, 3),
    ("R4d_stoch_vol",      4, "both", r3d_stoch_buy, r3d_stoch_sell, 3),
    ("R4e_inside_vol",     4, "both", r3e_inside_buy, r3e_inside_sell, 3),
    ("R4f_squeeze_vol",    4, "both", r3f_squeeze_buy, r3f_squeeze_sell, 3),
    ("R4g_pin_vol",        4, "both", r3g_pin_buy, r3g_pin_sell, 3),
    ("R4h_engulf_vol",     4, "both", r3h_engulf_buy, r3h_engulf_sell, 3),
    ("R4i_spike_fade",     4, "both", r3i_spike_fade_buy, r3i_spike_fade_sell, 3),
    ("R4j_session_vol",    4, "both", r3j_session_vol_buy, r3j_session_vol_sell, 3),
]

print(f"Registered {len(RULES)} rules")

# ═══════════════════════════════════════════════════════════════
# SCAN
# ═══════════════════════════════════════════════════════════════
all_setups = []

for cid in range(5):
    csv_path = P.data(f"cluster_{cid}_{CLUSTER_NAMES[cid]}.csv")
    df = pd.read_csv(csv_path, parse_dates=["time"])
    t = add_tech(df)
    ef = compute_extra_feats(df, t)
    n = len(df)
    start_idx = 60; end_idx = n - MAX_FWD - 1
    if end_idx <= start_idx:
        print(f"C{cid}: not enough data"); continue

    feat_base = df[FEATURE_COLS].values
    feat_extra = np.column_stack([ef[col] for col in EXTRA_FEAT_COLS])
    cluster_rules = [(nm, ci, dr, bf, sf, cd) for nm, ci, dr, bf, sf, cd in RULES if ci == cid]
    cooldowns = {nm: 0 for nm, _, _, _, _, _ in cluster_rules}
    rule_counts = {nm: 0 for nm, _, _, _, _, _ in cluster_rules}
    times = df["time"].values

    for i in range(start_idx, end_idx):
        a = t["atr14"][i]
        if np.isnan(a) or a < 1e-10: continue
        for rname, _, direc, buy_fn, sell_fn, cd in cluster_rules:
            if cooldowns[rname] > 0: cooldowns[rname] -= 1; continue
            fired_dir = None
            try:
                if direc == "buy":
                    if buy_fn(t, i): fired_dir = "buy"
                elif direc == "sell":
                    if sell_fn(t, i): fired_dir = "sell"
                else:
                    if buy_fn(t, i): fired_dir = "buy"
                    elif sell_fn(t, i): fired_dir = "sell"
            except: continue
            if fired_dir is None: continue
            cooldowns[rname] = cd
            outcome = forward_outcome(t, i, fired_dir)
            row = {"time": times[i], "cluster": cid, "rule": rname, "direction": fired_dir, "outcome": outcome}
            for j, col in enumerate(FEATURE_COLS): row[col] = feat_base[i, j]
            for j, col in enumerate(EXTRA_FEAT_COLS): row[col] = feat_extra[i, j]
            all_setups.append(row)
            rule_counts[rname] += 1

    print(f"\nC{cid} {CLUSTER_NAMES[cid]}:")
    for rn, cnt in rule_counts.items():
        print(f"  {rn:25s} {cnt:>6}")

setups_df = pd.DataFrame(all_setups)
out_path = P.data("setup_signals_eurusd.csv")
setups_df.to_csv(out_path, index=False)
print(f"\nTotal setups: {len(setups_df):,}")
if len(setups_df) > 0:
    print(f"Win rate: {setups_df['outcome'].mean():.3f}")
print(f"Saved: {out_path}")
