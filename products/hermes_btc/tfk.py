"""
Tick-Flow Kinematics (TFK) — Python port of the Pine v6 indicator.

Pine's `request.security_lower_tf` (real tick aggregation) is not available on
M5 CSV bars, so the order-flow proxy is signed bar volume:
    flow_t = sign(close_t - open_t) * tick_volume_t

Everything else (Kalman, kinematics, regime gate, hysteresis commitment) is a
faithful port. The output column we care about for relabeling is `committed_dir`
∈ {-1, +1}.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Pine helpers ────────────────────────────────────────────────────────────
def ema(x: np.ndarray, n: int) -> np.ndarray:
    a = 2.0 / (n + 1.0)
    out = np.empty_like(x, dtype=np.float64)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def rolling_stdev(x: np.ndarray, n: int) -> np.ndarray:
    # Pine ta.stdev default is biased=true (ddof=0)
    s = pd.Series(x).rolling(n, min_periods=n).std(ddof=0).to_numpy()
    return s


def atr(high, low, close, n: int) -> np.ndarray:
    prev_close = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)),
    )
    return ema(tr, n)  # Pine ta.atr uses RMA but EMA is close enough for the line scale


def session_vwap(high, low, close, vol, session_idx) -> np.ndarray:
    """VWAP that resets when session_idx changes. session_idx: integer per day."""
    hlc3 = (high + low + close) / 3.0
    out = np.empty_like(close, dtype=np.float64)
    cum_pv = 0.0
    cum_v = 0.0
    cur = session_idx[0]
    for i in range(len(close)):
        if session_idx[i] != cur:
            cur = session_idx[i]
            cum_pv = 0.0
            cum_v = 0.0
        cum_pv += hlc3[i] * vol[i]
        cum_v += vol[i]
        out[i] = cum_pv / cum_v if cum_v > 0 else hlc3[i]
    return out


# ── TFK core ────────────────────────────────────────────────────────────────
def compute_tfk(
    df: pd.DataFrame,
    *,
    flow_len: int = 20,
    damping: float = 0.93,
    mass: float = 1.0,
    q_noise: float = 0.001,
    r_noise: float = 0.10,
    hurst_len: int = 50,
    trend_thresh: float = 0.30,
    atr_len: int = 50,
    slope_k: float = 0.15,
    anchor_k: float = 0.02,
    flip_band: float = 0.30,
    flip_bars: int = 3,
    color_confirm: int = 5,
    tie_break: str = "vwap",  # "vwap" | "htf" | "prior"
    htf_ema_n: int = 50,
    htf_bars: int = 12,  # M5 → H1 = 12 bars
) -> pd.DataFrame:
    """
    df must have columns: time, open, high, low, close, tick_volume.
    Returns df with extra columns: force, velocity, x_est, eff_ratio, regime_w,
    trend_raw, in_range, vwap, htf_ema, bias, committed_dir, trend, tfk_line.
    """
    o = df["open"].to_numpy(dtype=np.float64)
    h = df["high"].to_numpy(dtype=np.float64)
    l = df["low"].to_numpy(dtype=np.float64)
    c = df["close"].to_numpy(dtype=np.float64)
    v = df["tick_volume"].to_numpy(dtype=np.float64)
    n = len(df)

    # 1. Signed bar flow (proxy for tick rule on LTF)
    sgn = np.where(c > o, 1.0, np.where(c < o, -1.0, 0.0))
    raw_flow = sgn * v
    flow_mu = ema(raw_flow, flow_len)
    flow_sig = rolling_stdev(raw_flow, flow_len)
    with np.errstate(invalid="ignore", divide="ignore"):
        force = np.where(flow_sig > 0, (raw_flow - flow_mu) / flow_sig, 0.0)
    force = np.nan_to_num(force, nan=0.0)

    # 2. Kinematics — single-pass IIR
    velocity = np.zeros(n)
    accel = force / mass
    vel = 0.0
    for i in range(n):
        vel = damping * vel + accel[i]
        velocity[i] = vel

    # 3. Kalman
    x_est = np.zeros(n)
    x = 0.0
    p = 1.0
    for i in range(n):
        x_pred = x
        p_pred = p + q_noise
        k = p_pred / (p_pred + r_noise)
        x = x_pred + k * (velocity[i] - x_pred)
        p = (1.0 - k) * p_pred
        x_est[i] = x

    # 4. Regime gate (efficiency ratio over hurst_len)
    c_shift = np.concatenate([np.full(hurst_len, np.nan), c[:-hurst_len]])
    net_move = np.abs(c - c_shift)
    abs_change = np.abs(np.diff(c, prepend=c[0]))
    path_sum = pd.Series(abs_change).rolling(hurst_len, min_periods=hurst_len).sum().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        eff_ratio = np.where(path_sum > 0, net_move / path_sum, 0.0)
    eff_ratio = np.nan_to_num(eff_ratio, nan=0.0)
    regime_w = np.maximum(0.0, eff_ratio - trend_thresh) / (1.0 - trend_thresh)
    trend_raw = x_est * regime_w
    in_range = regime_w == 0.0

    # 5. Tie-breaker biases
    t = pd.to_datetime(df["time"], format="%Y.%m.%d %H:%M", errors="coerce")
    if t.isna().all():
        t = pd.to_datetime(df["time"], errors="coerce")
    session_idx = t.dt.normalize().astype("int64").to_numpy()
    vwap_val = session_vwap(h, l, c, np.maximum(v, 1.0), session_idx)
    vwap_bias = np.where(c > vwap_val, 1.0, np.where(c < vwap_val, -1.0, 0.0))

    # HTF EMA approximated by EMA over htf_bars * htf_ema_n on the same series
    htf_ema = ema(c, htf_bars * htf_ema_n)
    htf_ema_prev = np.concatenate([[htf_ema[0]], htf_ema[:-1]])
    htf_bias = np.where(htf_ema > htf_ema_prev, 1.0, np.where(htf_ema < htf_ema_prev, -1.0, 0.0))

    if tie_break == "vwap":
        bias = vwap_bias
    elif tie_break == "htf":
        bias = htf_bias
    else:
        bias = np.zeros(n)

    # 6. Committed direction with hysteresis
    committed = np.zeros(n, dtype=np.int8)
    cur = 0
    opp = 0
    for i in range(n):
        if cur == 0:
            if trend_raw[i] > 0:
                cur = 1
            elif trend_raw[i] < 0:
                cur = -1
            elif bias[i] > 0:
                cur = 1
            elif bias[i] < 0:
                cur = -1
            else:
                cur = 1

        opposing = (cur > 0 and trend_raw[i] < -flip_band) or (
            cur < 0 and trend_raw[i] > flip_band
        )
        opp = opp + 1 if opposing else 0

        if opp >= flip_bars:
            cur = -cur
            opp = 0
        elif in_range[i] and tie_break != "prior" and bias[i] != 0:
            cur = 1 if bias[i] > 0 else -1

        committed[i] = cur

    # 6b. Confirmed (color) direction — only changes after committed has held
    #     for `color_confirm` consecutive bars. The committed direction itself
    #     flips immediately (per hysteresis above); this delays only the color.
    confirmed = np.zeros(n, dtype=np.int8)
    dir_streak = 1
    cur_confirmed = int(committed[0])
    confirmed[0] = cur_confirmed
    for i in range(1, n):
        if committed[i] == committed[i - 1]:
            dir_streak += 1
        else:
            dir_streak = 1
        if dir_streak >= color_confirm:
            cur_confirmed = int(committed[i])
        confirmed[i] = cur_confirmed

    # 7. Trend magnitude + price-space line — uses IMMEDIATE committed
    #    (so the line itself bends in real time; only color is delayed).
    aligned_mag = np.where(
        committed * trend_raw > 0, np.abs(trend_raw), np.abs(trend_raw) * 0.3
    )
    trend = committed * np.maximum(aligned_mag, 0.15)

    atr_v = atr(h, l, c, atr_len)
    slope = trend * slope_k * atr_v

    tfk_line = np.empty(n)
    tfk_line[0] = c[0]
    for i in range(1, n):
        tfk_line[i] = tfk_line[i - 1] + slope[i] + anchor_k * (c[i] - tfk_line[i - 1])

    out = df.copy()
    out["force"] = force
    out["velocity"] = velocity
    out["x_est"] = x_est
    out["eff_ratio"] = eff_ratio
    out["regime_w"] = regime_w
    out["trend_raw"] = trend_raw
    out["in_range"] = in_range
    out["vwap"] = vwap_val
    out["bias"] = bias
    out["committed_dir"] = committed
    out["confirmed_dir"] = confirmed
    out["trend"] = trend
    out["tfk_line"] = tfk_line
    return out
