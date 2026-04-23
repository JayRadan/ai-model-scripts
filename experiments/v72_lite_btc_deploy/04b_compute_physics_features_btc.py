"""
04b: Compute 14 physics+quantum features on the full XAU swing CSV,
then merge into per-cluster setup CSVs for v6 confirmation training.

Output: data/setups_{cid}_v6.csv  (same rows as setups_{cid}.csv + 14 new feature cols)
"""
from __future__ import annotations
import glob, os, time as _time
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "/home/jay/Desktop/new-model-zigzag/model_pipeline")
import paths as P

V6_FEAT_COLS = [
    "hurst_rs", "ou_theta", "entropy_rate", "kramers_up", "wavelet_er",
    "vwap_dist", "hour_enc", "dow_enc",
    "quantum_flow", "quantum_flow_h4", "quantum_momentum", "quantum_vwap_conf",
    "quantum_divergence", "quantum_div_strength",
]


def _ffill(arr):
    last = np.nan
    for i in range(len(arr)):
        if np.isfinite(arr[i]): last = arr[i]
        else: arr[i] = last


def compute_hurst_rs(ret, window=120, step=6):
    n = len(ret); h = np.full(n, np.nan)
    for i in range(window, n, step):
        seg = ret[i-window:i]; m = seg.mean()
        y = np.cumsum(seg - m); R = y.max() - y.min(); S = seg.std()
        if S > 1e-15 and R > 0: h[i] = np.log(R/S) / np.log(window)
    _ffill(h); return h


def compute_ou_theta(ret, window=60, step=6):
    n = len(ret); theta = np.full(n, np.nan); x = np.cumsum(ret)
    for i in range(window, n, step):
        seg = x[i-window:i]; mu = seg.mean()
        dx = np.diff(seg); xm = seg[:-1] - mu; d = np.sum(xm**2)
        if d > 1e-15: theta[i] = -np.sum(dx*xm)/d
    _ffill(theta); return theta


def compute_entropy(ret, window=100, n_bins=10, step=6):
    n = len(ret); ent = np.full(n, np.nan)
    for i in range(window, n, step):
        seg = ret[i-window:i]
        counts, _ = np.histogram(seg, bins=n_bins)
        p = counts/counts.sum(); p = p[p>0]
        ent[i] = -np.sum(p*np.log2(p)) / np.log2(n_bins)
    _ffill(ent); return ent


def compute_kramers_up(c, window=100, step=6):
    n = len(c); esc = np.full(n, np.nan)
    for i in range(window, n, step):
        seg = c[i-window:i]; hi = seg.max()
        sigma = np.std(np.diff(np.log(seg+1e-10)))
        if sigma < 1e-12: continue
        esc[i] = np.exp(-(hi-c[i])/(sigma*c[i]+1e-10))
    _ffill(esc); return esc


def _ricker(points, a):
    A = 2/(np.sqrt(3*a)*(np.pi**0.25)); wsq = a**2
    vec = np.arange(0,points)-(points-1.0)/2
    return A*(1-vec**2/wsq)*np.exp(-vec**2/(2*wsq))


def compute_wavelet_er(c, window=120, step=12):
    n = len(c); er = np.full(n, np.nan)
    kt = _ricker(min(200,window),20); kn = _ricker(min(30,window),3)
    for i in range(window, n, step):
        seg = c[i-window:i]; sn = (seg-seg.mean())/(seg.std()+1e-10)
        ct = np.convolve(sn,kt,mode='same'); cn = np.convolve(sn,kn,mode='same')
        er[i] = np.mean(ct**2)/(np.mean(cn**2)+1e-15)
    _ffill(er); return er


def compute_vwap_dist(df, atr):
    c = df["close"].values.astype(np.float64)
    v = df["spread"].values.astype(np.float64)
    v = np.maximum(v, 1.0)
    typical = (df["high"].values + df["low"].values + c) / 3.0
    dates = df["time"].dt.date.values
    vwap = np.zeros(len(c))
    cum_tv = 0.0; cum_v = 0.0; prev_date = dates[0]
    for i in range(len(c)):
        if dates[i] != prev_date:
            cum_tv = 0.0; cum_v = 0.0; prev_date = dates[i]
        cum_tv += typical[i]*v[i]; cum_v += v[i]
        vwap[i] = cum_tv/cum_v if cum_v > 0 else c[i]
    return np.where(atr > 1e-10, (c - vwap)/atr, 0.0)


def compute_quantum_flow(o, h, l, c, vol, lookback=21, vol_lookback=50):
    n = len(c)
    ha_close = (o+h+l+c)/4.0
    ha_open = np.empty(n); ha_open[0] = (o[0]+c[0])/2
    for i in range(1,n): ha_open[i] = (o[i-1]+c[i-1])/2
    trend_force = ha_close - ha_open
    avg_vol = pd.Series(vol).rolling(vol_lookback, min_periods=1).mean().values
    vol_factor = np.where(avg_vol > 1e-10, vol/avg_vol, 0.0)
    raw = trend_force * vol_factor * 1000.0
    smooth = pd.Series(raw).ewm(span=lookback, adjust=False).mean().values
    tr = np.empty(n); tr[0] = h[0]-l[0]
    tr[1:] = np.maximum.reduce([h[1:]-l[1:], np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])])
    atr14 = pd.Series(tr).rolling(14, min_periods=1).mean().values
    step = atr14 * 0.5
    return np.where(step > 1e-10, np.round(smooth/step)*step, smooth)


def main():
    print("04b: Computing physics features on full XAU swing CSV")
    df = pd.read_csv(P.data("swing_v5_btc.csv"), parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)
    c = df["close"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    o = df["open"].values.astype(np.float64)
    vol = df["spread"].values.astype(np.float64)
    vol = np.maximum(vol, 1.0)
    n = len(c)

    # ATR
    tr = np.concatenate([[h[0]-l[0]],
          np.maximum.reduce([h[1:]-l[1:], np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])])])
    atr = pd.Series(tr).rolling(14, min_periods=14).mean().values
    ret = np.concatenate([[0.0], np.diff(np.log(c))])

    print("  Hurst R/S...", end="", flush=True); t0=_time.time()
    df["hurst_rs"] = compute_hurst_rs(ret)
    print(f" {_time.time()-t0:.0f}s", flush=True)

    print("  OU theta...", end="", flush=True); t0=_time.time()
    df["ou_theta"] = compute_ou_theta(ret)
    print(f" {_time.time()-t0:.0f}s", flush=True)

    print("  Entropy...", end="", flush=True); t0=_time.time()
    df["entropy_rate"] = compute_entropy(ret)
    print(f" {_time.time()-t0:.0f}s", flush=True)

    print("  Kramers...", end="", flush=True); t0=_time.time()
    df["kramers_up"] = compute_kramers_up(c)
    print(f" {_time.time()-t0:.0f}s", flush=True)

    print("  Wavelet...", end="", flush=True); t0=_time.time()
    df["wavelet_er"] = compute_wavelet_er(c)
    print(f" {_time.time()-t0:.0f}s", flush=True)

    print("  VWAP...", end="", flush=True); t0=_time.time()
    df["vwap_dist"] = compute_vwap_dist(df, atr)
    print(f" {_time.time()-t0:.0f}s", flush=True)

    # Time encoding
    hour = df["time"].dt.hour.astype(float)
    dow = df["time"].dt.dayofweek.astype(float)
    df["hour_enc"] = np.sin(2*np.pi*hour/24)
    df["dow_enc"] = np.sin(2*np.pi*dow/5)

    # Quantum Flow M5
    print("  Quantum Flow M5...", end="", flush=True); t0=_time.time()
    df["quantum_flow"] = compute_quantum_flow(o, h, l, c, vol)
    print(f" {_time.time()-t0:.0f}s", flush=True)

    # Quantum Flow H4 (shift by 1 to avoid look-ahead)
    print("  Quantum Flow H4...", end="", flush=True); t0=_time.time()
    df_h4 = df.set_index("time")[["open","high","low","close","spread"]].resample("4h").agg(
        {"open":"first","high":"max","low":"min","close":"last","spread":"sum"}).dropna()
    qf_h4 = compute_quantum_flow(
        df_h4["open"].values, df_h4["high"].values, df_h4["low"].values,
        df_h4["close"].values, np.maximum(df_h4["spread"].values, 1.0))
    qf_h4_series = pd.Series(qf_h4, index=df_h4.index).shift(1)
    df["quantum_flow_h4"] = qf_h4_series.reindex(df.set_index("time").index, method="ffill").values
    print(f" {_time.time()-t0:.0f}s", flush=True)

    # Derived quantum features
    df["quantum_momentum"] = np.concatenate([[0.0], np.diff(df["quantum_flow"].values)])
    vwap_dir = np.sign(c - np.where(atr > 1e-10, c - df["vwap_dist"].values * atr, c))
    qf_sign = np.sign(df["quantum_flow"].values)
    df["quantum_vwap_conf"] = qf_sign * vwap_dir
    qf_m5 = df["quantum_flow"].values; qf_h4_vals = df["quantum_flow_h4"].values
    div = np.zeros(n)
    div[(qf_h4_vals > 0) & (qf_m5 < 0)] = +1.0
    div[(qf_h4_vals < 0) & (qf_m5 > 0)] = -1.0
    df["quantum_divergence"] = div
    df["quantum_div_strength"] = np.where(div != 0, np.abs(qf_h4_vals), 0.0)

    # Build physics feature lookup by time
    physics = df[["time"] + V6_FEAT_COLS].copy()
    physics = physics.fillna(0)

    # Merge into each setup CSV — skip already-processed _v6 / _v72l files
    for f in sorted(glob.glob(P.data("setups_*_btc.csv"))):
        name = os.path.basename(f)
        if "_v6_" in name or "_v72l_" in name: continue
        cid = name.replace("setups_", "").replace("_btc.csv", "")
        setup = pd.read_csv(f, parse_dates=["time"])
        merged = setup.merge(physics, on="time", how="left", suffixes=("","_v6"))
        for col in V6_FEAT_COLS:
            if col+"_v6" in merged.columns:
                merged[col] = merged[col+"_v6"].fillna(merged.get(col, 0))
                merged.drop(columns=[col+"_v6"], inplace=True)
            elif col not in merged.columns:
                merged[col] = 0
            merged[col] = merged[col].fillna(0)
        out = P.data(f"setups_{cid}_v6_btc.csv")
        merged.to_csv(out, index=False)
        print(f"  Saved: {out} ({len(merged):,} rows)")

    print("\nDone. Physics features merged into setup CSVs.")


if __name__ == "__main__":
    main()
