"""
Verify Python training features match the updated MQL5 EA implementation.

Conventions both sides should now share:
  - Windows EXCLUDE current bar (matching Python's seg = ret[i-window:i] / c[i-window:i])
  - std() uses ddof=0
  - Symmetric Ricker kernel, np.convolve 'same' ≡ MQL5 K//2 loop (proven equivalent)
  - Quantum Flow uses forward EMA (MQL5 warmup=100 bars ≈ pandas ewm after burn-in)
  - VWAP uses spread, not tick_volume
"""
from __future__ import annotations
import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/jay/Desktop/new-model-zigzag/model_pipeline")
import importlib.machinery
p04b = importlib.machinery.SourceFileLoader(
    "p04b",
    "/home/jay/Desktop/new-model-zigzag/model_pipeline/04b_compute_physics_features.py",
).load_module()

DATA = "/home/jay/Desktop/new-model-zigzag/data/swing_v5_xauusd.csv"

FEATS = [
    "hurst_rs", "ou_theta", "entropy_rate", "kramers_up", "wavelet_er",
    "vwap_dist", "hour_enc", "dow_enc",
    "quantum_flow", "quantum_flow_h4", "quantum_momentum", "quantum_vwap_conf",
    "quantum_divergence", "quantum_div_strength",
]


# ---- MQL5-style reimplementations matching the current EA ----

def mql5_hurst_rs(rb, idx, window=120):
    if idx + window + 1 >= len(rb): return 0.5
    rets = []
    for k in range(window):
        ri_old = window - k + 1
        ri_new = window - k
        c_old = rb[idx + ri_old]["close"]
        c_new = rb[idx + ri_new]["close"]
        rets.append(np.log(c_new / c_old) if c_old > 1e-10 else 0.0)
    rets = np.array(rets)
    mean_ret = rets.mean()
    dev = rets - mean_ret
    cum = np.cumsum(dev)
    R = cum.max() - cum.min()
    S = np.sqrt(np.sum(dev**2) / window)  # ddof=0
    if S < 1e-15 or R <= 0: return 0.5
    return np.log(R/S) / np.log(window)


def mql5_ou_theta(rb, idx, window=60):
    if idx + window + 1 >= len(rb): return 0.0
    x = np.zeros(window)
    for k in range(1, window):
        ri_old = window - k + 1
        ri_new = window - k
        c_old = rb[idx + ri_old]["close"]; c_new = rb[idx + ri_new]["close"]
        lr = np.log(c_new/c_old) if c_old > 1e-10 else 0.0
        x[k] = x[k-1] + lr
    mu = x.mean()
    num, den = 0.0, 0.0
    for k in range(window-1):
        dx = x[k+1] - x[k]
        xm = x[k] - mu
        num += dx*xm; den += xm*xm
    return -num/den if den > 1e-15 else 0.0


def mql5_entropy(rb, idx, window=100):
    if idx + window + 1 >= len(rb): return 1.0
    rets = []
    for k in range(window):
        ri_old = window - k + 1
        ri_new = window - k
        c_old = rb[idx + ri_old]["close"]; c_new = rb[idx + ri_new]["close"]
        rets.append(np.log(c_new/c_old) if c_old > 1e-10 else 0.0)
    rets = np.array(rets)
    mn, mx = rets.min(), rets.max()
    if mx - mn < 1e-15: return 1.0
    nbins = 10
    counts, _ = np.histogram(rets, bins=nbins)
    total = window
    ent = 0.0
    for c in counts:
        if c > 0:
            p = c/total
            ent -= p*np.log2(p)
    return ent / np.log2(nbins)


def mql5_kramers(rb, idx, window=100):
    if idx + window >= len(rb): return 0.5
    # seg = bars i-window..i-1 = rb[1..window]
    hi = max(rb[idx + k]["close"] for k in range(1, window+1))
    # sigma = std of log returns within seg
    rets = []
    for k in range(window-1):
        ri_old = window - k
        ri_new = window - k - 1
        c_old = rb[idx + ri_old]["close"]; c_new = rb[idx + ri_new]["close"]
        rets.append(np.log(c_new/c_old) if c_old > 1e-10 else 0.0)
    rets = np.array(rets)
    # Python: np.std(diff(log)) = ddof=0
    sigma = rets.std()  # ddof=0
    c_cur = rb[idx]["close"]
    if sigma < 1e-12 or c_cur < 1e-10: return 0.5
    return np.exp(-(hi - c_cur) / (sigma * c_cur + 1e-10))


def mql5_wavelet_er(rb, idx, window=120):
    if idx + window >= len(rb): return 1.0
    # seg = rb[window] ... rb[1] chronological
    seg = np.array([rb[idx + window - k]["close"] for k in range(window)])
    m = seg.mean(); s = seg.std()  # ddof=0
    # Python: sn = (seg - m)/(s + 1e-10)
    sn = (seg - m) / (s + 1e-10)

    def ricker(points, a):
        A = 2.0/(np.sqrt(3*a)*np.pi**0.25)
        v = np.arange(points) - (points-1)/2
        return A*(1 - v**2/(a**2))*np.exp(-v**2/(2*a**2))

    pt = min(200, window); kt = ricker(pt, 20.0)
    pn = min(30, window);  kn = ricker(pn, 3.0)

    # Match MQL5 loop offset (pt-1)//2
    ct_off = (pt - 1) // 2
    cn_off = (pn - 1) // 2

    e_trend = 0.0; e_noise = 0.0
    for i in range(window):
        ct = 0.0
        for j in range(pt):
            si = i - ct_off + j
            if 0 <= si < window: ct += sn[si]*kt[j]
        e_trend += ct*ct
        cn = 0.0
        for j in range(pn):
            si = i - cn_off + j
            if 0 <= si < window: cn += sn[si]*kn[j]
        e_noise += cn*cn
    return (e_trend/window) / (e_noise/window + 1e-15) if e_noise > 1e-15 else 1.0


class VWAPState:
    def __init__(self):
        self.cum_tv = 0.0; self.cum_v = 0.0; self.day = -1

def mql5_vwap_dist(rb, idx, atr, state):
    dt = rb[idx]["time"]
    day = dt.day
    if day != state.day:
        state.cum_tv = 0.0; state.cum_v = 0.0; state.day = day
    typical = (rb[idx]["high"] + rb[idx]["low"] + rb[idx]["close"]) / 3.0
    vol = max(rb[idx]["spread"], 1.0)
    state.cum_tv += typical*vol; state.cum_v += vol
    vwap = state.cum_tv/state.cum_v if state.cum_v > 0 else rb[idx]["close"]
    return (rb[idx]["close"] - vwap)/atr if atr > 1e-10 else 0.0


def mql5_quantum_flow(rb, idx, lookback=21, vol_lb=50, warmup=100):
    """Forward EMA walking from oldest (idx+warmup) to newest (idx)."""
    N = len(rb)
    if idx + warmup + vol_lb >= N: return 0.0
    alpha = 2.0 / (lookback + 1.0)

    ema = 0.0
    ema_init = False
    for k in range(warmup, -1, -1):
        i = idx + k
        ha_c = (rb[i]["open"] + rb[i]["high"] + rb[i]["low"] + rb[i]["close"])/4.0
        ha_o = (rb[i+1]["open"] + rb[i+1]["close"])/2.0 if i+1 < N else rb[i]["open"]
        tf = ha_c - ha_o
        avg_v = 0.0; cnt = 0
        for j in range(i, min(i + vol_lb, N)):
            avg_v += max(rb[j]["spread"], 1.0); cnt += 1
        if cnt > 0: avg_v /= cnt
        else: avg_v = 1.0
        vf = max(rb[i]["spread"],1.0)/avg_v if avg_v > 1e-10 else 0.0
        raw_i = tf * vf * 1000.0
        if not ema_init: ema = raw_i; ema_init = True
        else: ema = alpha * raw_i + (1 - alpha) * ema

    # ATR14 at idx: last 14 TRs from rb[idx]..rb[idx+13]
    tr_sum = 0.0
    for k in range(idx, idx + 14):
        if k + 1 < N:
            hk, lk, ck1 = rb[k]["high"], rb[k]["low"], rb[k+1]["close"]
            tr = max(hk - lk, abs(hk - ck1), abs(lk - ck1))
        else:
            tr = rb[k]["high"] - rb[k]["low"]
        tr_sum += tr
    atr_v = tr_sum / 14.0
    step = atr_v * 0.5
    if step > 1e-10: ema = round(ema/step)*step
    return ema


def main():
    print("Loading swing CSV...", flush=True)
    df = pd.read_csv(DATA, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)
    c = df["close"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    o = df["open"].values.astype(np.float64)
    vol = np.maximum(df["spread"].values.astype(np.float64), 1.0)
    n = len(c)
    print(f"  Loaded {n:,} bars, range {df['time'].iat[0]} to {df['time'].iat[-1]}", flush=True)

    tr = np.concatenate([[h[0]-l[0]],
          np.maximum.reduce([h[1:]-l[1:], np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])])])
    atr = pd.Series(tr).rolling(14, min_periods=14).mean().values
    ret = np.concatenate([[0.0], np.diff(np.log(c))])

    print("Computing Python training features (exact 04b pipeline)...", flush=True)
    py = {}
    py["hurst_rs"]     = p04b.compute_hurst_rs(ret)
    py["ou_theta"]     = p04b.compute_ou_theta(ret)
    py["entropy_rate"] = p04b.compute_entropy(ret)
    py["kramers_up"]   = p04b.compute_kramers_up(c)
    py["wavelet_er"]   = p04b.compute_wavelet_er(c)
    py["vwap_dist"]    = p04b.compute_vwap_dist(df, atr)
    hour = df["time"].dt.hour.astype(float).values
    dow = df["time"].dt.dayofweek.astype(float).values
    py["hour_enc"] = np.sin(2*np.pi*hour/24)
    py["dow_enc"]  = np.sin(2*np.pi*dow/5)
    py["quantum_flow"] = p04b.compute_quantum_flow(o, h, l, c, vol)

    df_h4 = df.set_index("time")[["open","high","low","close","spread"]].resample("4h").agg(
        {"open":"first","high":"max","low":"min","close":"last","spread":"sum"}).dropna()
    qf_h4 = p04b.compute_quantum_flow(
        df_h4["open"].values, df_h4["high"].values, df_h4["low"].values,
        df_h4["close"].values, np.maximum(df_h4["spread"].values, 1.0))
    qf_h4_series = pd.Series(qf_h4, index=df_h4.index).shift(1)
    py["quantum_flow_h4"] = qf_h4_series.reindex(df.set_index("time").index, method="ffill").values

    py["quantum_momentum"] = np.concatenate([[0.0], np.diff(py["quantum_flow"])])
    vwap_dir = np.sign(c - np.where(atr > 1e-10, c - py["vwap_dist"]*atr, c))
    qf_sign  = np.sign(py["quantum_flow"])
    py["quantum_vwap_conf"] = qf_sign * vwap_dir
    div = np.zeros(n)
    div[(py["quantum_flow_h4"] > 0) & (py["quantum_flow"] < 0)] = +1.0
    div[(py["quantum_flow_h4"] < 0) & (py["quantum_flow"] > 0)] = -1.0
    py["quantum_divergence"] = div
    py["quantum_div_strength"] = np.where(div != 0, np.abs(py["quantum_flow_h4"]), 0.0)

    # Sample bars — use indices that are multiples of step=6 or 12 so Python's ffill
    # doesn't introduce lag (pick bars where Python actually computed)
    base = 6 * 12  # LCM of step=6 and step=12
    sample_idxs = [int(i) for i in np.arange(n - 5000, n - 300, 200) if i > 500 and i < n-300]
    sample_idxs = [i - (i % base) for i in sample_idxs]
    sample_idxs = sorted(set(sample_idxs))[:15]
    print(f"Testing on {len(sample_idxs)} sample bars (aligned to step grid)", flush=True)

    def make_rb(up_to_i):
        start = max(0, up_to_i - 400)
        sub = df.iloc[start:up_to_i+1].iloc[::-1].reset_index(drop=True)
        return sub.to_dict("records")

    diffs = {f: [] for f in FEATS}

    for i in sample_idxs:
        rb = make_rb(i)
        atr_i = atr[i]
        if not np.isfinite(atr_i) or atr_i <= 0: continue

        # VWAP replay from day start
        this_day = df["time"].iat[i].date()
        day_start = i
        while day_start > 0 and df["time"].iat[day_start-1].date() == this_day:
            day_start -= 1
        state = VWAPState()
        mq_vwap_final = 0.0
        for j in range(day_start, i+1):
            rb_j = make_rb(j)
            atr_j = atr[j]
            if np.isfinite(atr_j) and atr_j > 0:
                mq_vwap_final = mql5_vwap_dist(rb_j, 0, atr_j, state)

        mq = {}
        mq["hurst_rs"]     = mql5_hurst_rs(rb, 0, 120)
        mq["ou_theta"]     = mql5_ou_theta(rb, 0, 60)
        mq["entropy_rate"] = mql5_entropy(rb, 0, 100)
        mq["kramers_up"]   = mql5_kramers(rb, 0, 100)
        mq["wavelet_er"]   = mql5_wavelet_er(rb, 0, 120)
        mq["vwap_dist"]    = mq_vwap_final
        dt = df["time"].iat[i]
        mq["hour_enc"] = np.sin(2*np.pi*dt.hour/24)
        mq["dow_enc"]  = np.sin(2*np.pi*dt.dayofweek/5)
        mq["quantum_flow"] = mql5_quantum_flow(rb, 0, 21, 50, warmup=100)

        h4_end_time = pd.Timestamp(dt).floor("4h")
        h4_bars = df_h4[df_h4.index < h4_end_time].tail(150)
        if len(h4_bars) >= 125:
            h4_rb = []
            for t, row in h4_bars.iloc[::-1].iterrows():
                h4_rb.append({
                    "open": row["open"], "high": row["high"], "low": row["low"],
                    "close": row["close"], "spread": row["spread"], "time": t,
                })
            mq["quantum_flow_h4"] = mql5_quantum_flow(h4_rb, 0, 21, 20, warmup=100)
        else:
            mq["quantum_flow_h4"] = 0.0

        qf_prev = mql5_quantum_flow(rb, 1, 21, 50, warmup=100)
        mq["quantum_momentum"] = mq["quantum_flow"] - qf_prev

        vwap_dir_sign = 1.0 if mq["vwap_dist"]>0 else (-1.0 if mq["vwap_dist"]<0 else 0.0)
        qf_sign_ = 1.0 if mq["quantum_flow"]>0 else (-1.0 if mq["quantum_flow"]<0 else 0.0)
        mq["quantum_vwap_conf"] = qf_sign_ * vwap_dir_sign

        div_v = 0.0
        if mq["quantum_flow_h4"]>0 and mq["quantum_flow"]<0: div_v = 1.0
        if mq["quantum_flow_h4"]<0 and mq["quantum_flow"]>0: div_v = -1.0
        mq["quantum_divergence"] = div_v
        mq["quantum_div_strength"] = abs(mq["quantum_flow_h4"]) if div_v != 0 else 0.0

        for f in FEATS:
            py_v = py[f][i]
            mq_v = mq[f]
            if not np.isfinite(py_v): py_v = 0.0
            if not np.isfinite(mq_v): mq_v = 0.0
            diffs[f].append((py_v, mq_v, py_v - mq_v))

    print(f"\n{'='*80}")
    print(f"{'FEATURE':<25} {'PY range':<22} {'MQL5 range':<22} {'MAX |diff|':<12} {'STATUS'}")
    print("="*80)
    n_bad = 0
    for f in FEATS:
        arr = diffs[f]
        if not arr:
            print(f"{f:<25} (no samples)"); continue
        py_vals = np.array([a[0] for a in arr])
        mq_vals = np.array([a[1] for a in arr])
        d = np.array([a[2] for a in arr])
        max_abs = np.max(np.abs(d))
        if f in ("hour_enc", "dow_enc", "quantum_divergence", "quantum_vwap_conf"):
            tol = 1e-9
        elif f in ("hurst_rs", "ou_theta", "entropy_rate", "kramers_up"):
            tol = 1e-6
        elif f == "wavelet_er":
            tol = 1e-4
        elif f in ("vwap_dist",):
            tol = 1e-6
        elif f in ("quantum_flow", "quantum_flow_h4"):
            tol = 0.5    # ATR-quantized + EMA warmup differs from pandas all-history
        elif f in ("quantum_momentum", "quantum_div_strength"):
            tol = 1.0
        else:
            tol = 1e-3
        status = "OK " if max_abs <= tol else "MISMATCH"
        if max_abs > tol: n_bad += 1
        py_s = f"[{py_vals.min():+.3f},{py_vals.max():+.3f}]"
        mq_s = f"[{mq_vals.min():+.3f},{mq_vals.max():+.3f}]"
        print(f"{f:<25} {py_s:<22} {mq_s:<22} {max_abs:<12.4g} {status}")

    print("="*80)
    print(f"{'ALL FEATURES MATCH' if n_bad==0 else f'{n_bad} feature(s) still mismatched'}")


if __name__ == "__main__":
    main()
