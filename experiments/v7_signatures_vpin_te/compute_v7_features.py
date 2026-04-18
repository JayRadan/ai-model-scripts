"""
v7 experimental features — 3 orthogonal groups added to v6's 14.

Groups:
  A. VPIN            — range-based volume-synchronized toxicity (Easley/LdP 2012)
  B. Path signatures — level-2 iterated integrals of the (t, log-price) path
                       (Lévy area, quadratic variation, time-weighted drift)
  C. Transfer entropy H1->M5 — binned TE (Schreiber 2000), strictly past-only

All features computed at bar i use ONLY bars with index < i.
Uses same step+ffill convention as v6b.
Outputs setups_{cid}_v7.csv by merging onto existing v6 CSVs.
"""
from __future__ import annotations
import glob, math, os, time as _time
import numpy as np
import pandas as pd
from scipy.special import erf
import sys
sys.path.insert(0, "/home/jay/Desktop/new-model-zigzag/model_pipeline")
import paths as P


NEW_COLS = [
    "vpin",
    "sig_levy_area", "sig_quad_var", "sig_time_weighted_drift",
    "te_h1_m5",
]


def _ffill(arr):
    last = np.nan
    for i in range(len(arr)):
        if np.isfinite(arr[i]): last = arr[i]
        else: arr[i] = last


# ---------------------------------------------------------------------
# A. VPIN  (range-based activity proxy for FX bars without tick volume)
# ---------------------------------------------------------------------
def compute_vpin(high, low, close, n_buckets=50, step=6):
    """Bulk-volume-classified VPIN on rolling n_buckets window.

    Activity proxy (since no tick volume in FX CSV): V_t = high_t - low_t.
    Bulk-volume classification (Easley, Lopez de Prado, O'Hara 2012):
        frac_buy_t = Φ((r_t - μ) / σ)      r_t = close_t - close_{t-1}
        B_t = V_t · frac_buy,   S_t = V_t · (1 - frac_buy)
    VPIN_t = Σ_{last n_buckets} |B - S|  /  Σ_{last n_buckets} V

    Strict past-only: at bar i, uses indices [i - n_buckets, i).
    """
    n = len(close)
    vpin = np.full(n, np.nan)
    V = high - low                                      # activity proxy
    r = np.concatenate([[0.0], np.diff(close)])
    # Rolling 200-bar μ/σ for BVC normalization (past only, min_periods enforced)
    s = pd.Series(r)
    mu_roll    = s.rolling(200, min_periods=200).mean().values
    sigma_roll = s.rolling(200, min_periods=200).std(ddof=0).values
    sqrt2 = math.sqrt(2.0)
    warmup = n_buckets + 200                            # need past bars for both
    for i in range(warmup, n, step):
        seg_V = V[i - n_buckets:i]
        seg_r = r[i - n_buckets:i]
        mu, sig = mu_roll[i - 1], sigma_roll[i - 1]     # latest past bar
        if not np.isfinite(sig) or sig <= 1e-12: continue
        z = (seg_r - mu) / (sig * sqrt2)
        frac_buy = 0.5 * (1.0 + erf(z))                 # Gaussian CDF
        B = seg_V * frac_buy
        S = seg_V * (1.0 - frac_buy)
        tot = seg_V.sum()
        if tot <= 1e-12: continue
        vpin[i] = np.abs(B - S).sum() / tot
    _ffill(vpin); return vpin


# ---------------------------------------------------------------------
# B. Path signatures — iterated integrals of (t, log-price) path
# ---------------------------------------------------------------------
def compute_path_signatures(close, window=60, step=6):
    """Three level-2 signature coordinates of the normalized 2D path
        X_k = (k/window,  100 · (log c_{i-window+k} - log c_{i-window}))
    over the trailing `window` past bars.  Strict past-only: uses close[i-window:i].

      sig_levy_area           = 0.5 · (S_{t,Y} − S_{Y,t})
                                iterated integral — path asymmetry / coiling
      sig_quad_var            = Σ (ΔY_k)^2
                                realized variance, scaled
      sig_time_weighted_drift = Σ (k/window) · ΔY_k
                                did price move EARLY or LATE in window?
    """
    n = len(close)
    la  = np.full(n, np.nan)
    qv  = np.full(n, np.nan)
    twd = np.full(n, np.nan)
    logc = np.log(close)
    # Use window past bars → W-1 log-returns. Normalised time axis length = W-1.
    W = window
    t_axis = np.arange(W, dtype=np.float64) / (W - 1)     # 0..1 over W points
    dt = np.diff(t_axis)                                  # length W-1, constant = 1/(W-1)
    for i in range(W, n, step):
        seg_logc = logc[i - W:i]                          # W past points
        Y = 100.0 * (seg_logc - seg_logc[0])              # percent-log cum path
        dY = np.diff(Y)                                   # length W-1
        # Quadratic variation
        qv[i] = float(np.sum(dY * dY))
        # Time-weighted drift: Σ t_k · dY_k (k=0..W-2)
        t_k = t_axis[:-1]
        twd[i] = float(np.sum(t_k * dY))
        # Lévy area (discrete iterated integral):
        #   S_{t,Y}  = Σ_{l>=1} dY_l · (Σ_{k<l} dt_k)
        #   S_{Y,t}  = Σ_{l>=1} dt_l · (Σ_{k<l} dY_k)
        cum_dt = np.cumsum(dt)
        cum_dY = np.cumsum(dY)
        s12 = float(np.sum(dY[1:] * cum_dt[:-1]))
        s21 = float(np.sum(dt[1:] * cum_dY[:-1]))
        la[i] = 0.5 * (s12 - s21)
    _ffill(la); _ffill(qv); _ffill(twd)
    return la, qv, twd


# ---------------------------------------------------------------------
# C. Transfer entropy H1->M5  (binned, strictly past-only)
# ---------------------------------------------------------------------
def _te_binned_2x2x2(x, y):
    """One-lag TE(Y→X) on two aligned 1-D arrays, binarized at window median.
       TE = Σ p(x_{t+1}, x_t, y_t) · log2[ p(x_{t+1}|x_t,y_t) / p(x_{t+1}|x_t) ]
    Length >= 30 preferred."""
    n_pairs = len(x) - 1
    if n_pairs < 20: return np.nan
    bx = (x > np.median(x)).astype(np.int64)
    by = (y > np.median(y)).astype(np.int64)
    x_next = bx[1:]
    x_prev = bx[:-1]
    y_prev = by[:-1]
    # 2x2x2 joint counts  key = x_next*4 + x_prev*2 + y_prev
    keys = x_next * 4 + x_prev * 2 + y_prev
    counts = np.bincount(keys, minlength=8).astype(np.float64)
    total = counts.sum()
    if total <= 0: return np.nan
    p_xxy = counts / total
    pxxy_3d = p_xxy.reshape(2, 2, 2)
    p_xy   = pxxy_3d.sum(axis=0)                         # p(x_t, y_t)
    p_xnx  = pxxy_3d.sum(axis=2)                         # p(x_{t+1}, x_t)
    p_x    = p_xnx.sum(axis=0)                           # p(x_t)
    te = 0.0
    for a in range(2):
        for b in range(2):
            for c in range(2):
                pxxy = pxxy_3d[a, b, c]
                if pxxy <= 0: continue
                num = pxxy * p_x[b]
                den = p_xnx[a, b] * p_xy[b, c]
                if den <= 0 or num <= 0: continue
                te += pxxy * math.log2(num / den)
    return float(te)


def compute_transfer_entropy_h1_to_m5(m5_time, m5_close, window=200, step=6):
    """TE(H1_ret → M5_ret) using last `window` aligned (H1, M5) return pairs.

    Strict past-only construction:
      • H1 bins are [h, h+1h) with label = h (left-label, left-closed).
        Each H1 bar h contains M5 bars timestamped [h, h+1h).
      • At M5 bar i (timestamp t_i, the OPEN of bar i), the latest FULLY CLOSED
        H1 bar is the one labeled h ≤ t_i - 1h. We only use h1 bars with
        label + 1h <= t_i.
      • For alignment, we pair each H1 bar's return with the M5 return at the
        same H1 label time (which is the M5 bar at the start of that hour —
        strictly in the past of our current bar).
    """
    n = len(m5_close)
    te = np.full(n, np.nan)
    m5 = pd.DataFrame({"close": m5_close}, index=pd.DatetimeIndex(m5_time))
    # Left-labeled, left-closed hourly bars: label h represents bin [h, h+1h).
    h1 = m5["close"].resample("1h", label="left", closed="left").last().dropna()
    h1_ret = h1.pct_change().dropna()                      # return indexed at H1 label h

    # M5 returns: r_i = (close_i - close_{i-1}) / close_{i-1}
    prev = np.concatenate([[m5_close[0]], m5_close[:-1]])
    m5_ret_vals = (m5_close - prev) / np.where(prev > 0, prev, 1.0)
    m5_ret_series = pd.Series(m5_ret_vals, index=m5.index)
    # Align an M5 return to each H1 label h — take the M5 bar whose timestamp equals h
    m5_at_h1 = m5_ret_series.reindex(h1_ret.index, method="ffill")

    h1_vals    = h1_ret.values
    m5_at_vals = m5_at_h1.values
    h1_labels  = h1_ret.index.values.astype("datetime64[ns]")
    m5_np      = m5.index.values.astype("datetime64[ns]")

    one_hour_ns = np.timedelta64(1, 'h')
    warmup_bars = window * 12 + 500       # ~window H1 bars = window·12 M5 bars
    for i in range(warmup_bars, n, step):
        t_now = m5_np[i]
        # Latest H1 label h satisfying h + 1h <= t_now  (H1 bar fully closed)
        cutoff = t_now - one_hour_ns
        h1_idx = np.searchsorted(h1_labels, cutoff, side="right") - 1
        if h1_idx < window: continue
        # Use the `window` H1 returns strictly before & including cutoff label
        h1_slice = h1_vals[h1_idx - window + 1:h1_idx + 1]
        m5_slice = m5_at_vals[h1_idx - window + 1:h1_idx + 1]
        if not (np.isfinite(h1_slice).all() and np.isfinite(m5_slice).all()): continue
        te[i] = _te_binned_2x2x2(m5_slice, h1_slice)
    _ffill(te); return te


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    t_total = _time.time()
    print("v7 — computing VPIN + path signatures + transfer entropy", flush=True)
    swing = pd.read_csv(P.data("swing_v5_xauusd.csv"), parse_dates=["time"])
    swing = swing.sort_values("time").reset_index(drop=True)
    c = swing["close"].values.astype(np.float64)
    h = swing["high"].values.astype(np.float64)
    l = swing["low"].values.astype(np.float64)
    n = len(c)
    print(f"  Bars: {n:,}", flush=True)

    print("  [1/3] VPIN...", end="", flush=True); t0 = _time.time()
    vpin = compute_vpin(h, l, c)
    print(f" {_time.time() - t0:.0f}s", flush=True)

    print("  [2/3] Path signatures...", end="", flush=True); t0 = _time.time()
    la, qv, twd = compute_path_signatures(c)
    print(f" {_time.time() - t0:.0f}s", flush=True)

    print("  [3/3] Transfer entropy H1->M5...", end="", flush=True); t0 = _time.time()
    te = compute_transfer_entropy_h1_to_m5(swing["time"].values, c)
    print(f" {_time.time() - t0:.0f}s", flush=True)

    feat_df = pd.DataFrame({
        "time": swing["time"].values,
        "vpin": vpin,
        "sig_levy_area": la,
        "sig_quad_var": qv,
        "sig_time_weighted_drift": twd,
        "te_h1_m5": te,
    })

    # Sanity print BEFORE saving — catches dead or pathological features
    print("\n  Feature sanity (global):", flush=True)
    for col in NEW_COLS:
        s = feat_df[col]
        nan_pct = 100 * s.isna().mean()
        nonzero_pct = 100 * (s.fillna(0) != 0).mean()
        print(f"    {col:<26} mean={s.mean():+.6f}  std={s.std():.6f}  "
              f"min={s.min():+.6f}  max={s.max():+.6f}  NaN%={nan_pct:.2f}  "
              f"nonzero%={nonzero_pct:.1f}",
              flush=True)

    feat_df = feat_df.fillna(0)

    # Merge onto v6 setup CSVs
    for f in sorted(glob.glob(P.data("setups_*_v6.csv"))):
        cid = os.path.basename(f).split("_")[1]
        setup = pd.read_csv(f, parse_dates=["time"])
        merged = setup.merge(feat_df, on="time", how="left")
        for col in NEW_COLS:
            merged[col] = merged[col].fillna(0)
        out = P.data(f"setups_{cid}_v7.csv")
        merged.to_csv(out, index=False)
        print(f"  Saved: {out} ({len(merged):,} rows)", flush=True)

    print(f"\nTotal: {_time.time() - t_total:.0f}s", flush=True)


if __name__ == "__main__":
    main()
