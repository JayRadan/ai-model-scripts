"""
v7.2-lite feature compute — step=1 (every bar) for bit-parity with MQL5 live.

Computes 4 features on the XAU swing CSV:
  1. vpin            — range-based VPIN (BVC, Gaussian CDF, 50 buckets, 200-bar baseline)
  2. sig_quad_var    — Σ(ΔY)² over last 60 log-returns (scaled to pct)
  3. har_rv_ratio    — RV(last 288 bars) / RV(last 8640 bars) — short/long vol regime
  4. hawkes_eta      — event-rate(last 60) / event-rate(last 600)  event = |r| > 2·σ_500

All computations are STRICTLY PAST-ONLY: feature at bar i uses ONLY bars indexed < i.
Outputs setups_{cid}_v72l.csv, ready for 01_validate_v72_lite.py (with setup path adjusted)
and 02_train_and_export_v72l.py.
"""
from __future__ import annotations
import glob, math, os, time as _time
import numpy as np
import pandas as pd
from scipy.special import erf
import sys
sys.path.insert(0, "/home/jay/Desktop/new-model-zigzag/model_pipeline")
import paths as P

NEW_COLS = ["vpin", "sig_quad_var", "har_rv_ratio", "hawkes_eta"]


# ---------------------------------------------------------------------
# 1. VPIN — same formula as v7, now step=1 (every bar)
# ---------------------------------------------------------------------
def compute_vpin(high, low, close, n_buckets=50, mu_sigma_window=200):
    n = len(close)
    vpin = np.full(n, np.nan)
    V = high - low
    r = np.concatenate([[0.0], np.diff(close)])
    s = pd.Series(r)
    mu_roll    = s.rolling(mu_sigma_window, min_periods=mu_sigma_window).mean().values
    sigma_roll = s.rolling(mu_sigma_window, min_periods=mu_sigma_window).std(ddof=0).values
    sqrt2 = math.sqrt(2.0)
    warmup = n_buckets + mu_sigma_window
    for i in range(warmup, n):          # step=1
        seg_V = V[i - n_buckets:i]
        seg_r = r[i - n_buckets:i]
        mu, sig = mu_roll[i - 1], sigma_roll[i - 1]
        if not np.isfinite(sig) or sig <= 1e-12: continue
        z = (seg_r - mu) / (sig * sqrt2)
        frac_buy = 0.5 * (1.0 + erf(z))
        B = seg_V * frac_buy
        S = seg_V * (1.0 - frac_buy)
        tot = seg_V.sum()
        if tot <= 1e-12: continue
        vpin[i] = np.abs(B - S).sum() / tot
    return vpin


# ---------------------------------------------------------------------
# 2. sig_quad_var — realized variance of last W log-returns (scaled to pct)
# ---------------------------------------------------------------------
def compute_sig_quad_var(close, window=60):
    n = len(close)
    qv = np.full(n, np.nan)
    logc = np.log(close)
    for i in range(window, n):          # step=1
        seg_logc = logc[i - window:i]                          # W past points
        Y = 100.0 * (seg_logc - seg_logc[0])                   # percent-log cum path
        dY = np.diff(Y)                                        # length W-1
        qv[i] = float(np.sum(dY * dY))
    return qv


# ---------------------------------------------------------------------
# 3. HAR-RV ratio — RV_short / RV_long
# ---------------------------------------------------------------------
def compute_har_rv_ratio(close, short_bars=288, long_bars=8640):
    r = np.concatenate([[0.0], np.diff(np.log(close))])
    r2 = r * r
    # Past-only: use [i - W, i) via rolling().shift(1)
    rv_s = pd.Series(r2).rolling(short_bars, min_periods=short_bars).mean().shift(1).values
    rv_l = pd.Series(r2).rolling(long_bars,  min_periods=long_bars ).mean().shift(1).values
    ratio = rv_s / np.where(rv_l > 1e-18, rv_l, 1e-18)
    ratio = np.clip(ratio, 0.0, 20.0)
    return ratio


# ---------------------------------------------------------------------
# 4. Hawkes η — event-rate(short)/event-rate(long)
# ---------------------------------------------------------------------
def compute_hawkes_eta(close, sigma_window=500, event_k=2.0,
                       short_window=60, long_window=600):
    r = np.concatenate([[0.0], np.diff(np.log(close))])
    sig = pd.Series(r).rolling(sigma_window, min_periods=sigma_window).std(ddof=0).values
    event = np.where((np.isfinite(sig)) & (sig > 0) & (np.abs(r) > event_k * sig), 1.0, 0.0)
    rate_s = pd.Series(event).rolling(short_window, min_periods=short_window).mean().shift(1).values
    rate_l = pd.Series(event).rolling(long_window,  min_periods=long_window ).mean().shift(1).values
    eta = rate_s / np.where(rate_l > 1e-9, rate_l, 1e-9)
    eta = np.where(np.isfinite(eta), np.clip(eta, 0.0, 20.0), 1.0)
    return eta


def main():
    t_total = _time.time()
    print("v7.2-lite features (step=1, every bar) — compute on XAU swing", flush=True)
    swing = pd.read_csv(P.data("swing_v5_xauusd.csv"), parse_dates=["time"])
    swing = swing.sort_values("time").reset_index(drop=True)
    c = swing["close"].values.astype(np.float64)
    h = swing["high"].values.astype(np.float64)
    l = swing["low"].values.astype(np.float64)
    n = len(c)
    print(f"  Bars: {n:,}", flush=True)

    print("  [1/4] VPIN (step=1)...", end="", flush=True); t0 = _time.time()
    vpin = compute_vpin(h, l, c)
    print(f" {_time.time() - t0:.0f}s", flush=True)

    print("  [2/4] sig_quad_var (step=1)...", end="", flush=True); t0 = _time.time()
    qv = compute_sig_quad_var(c)
    print(f" {_time.time() - t0:.0f}s", flush=True)

    print("  [3/4] HAR-RV ratio...", end="", flush=True); t0 = _time.time()
    har = compute_har_rv_ratio(c)
    print(f" {_time.time() - t0:.0f}s", flush=True)

    print("  [4/4] Hawkes η...", end="", flush=True); t0 = _time.time()
    eta = compute_hawkes_eta(c)
    print(f" {_time.time() - t0:.0f}s", flush=True)

    # 6-bar trailing rolling mean — recovers most of the smoothing the old
    # step=6 + ffill provided, while staying strictly past-only and reproducible
    # identically in MQL5 (just average the last 6 computed values).
    def _smooth(arr, w=6):
        return pd.Series(arr).rolling(w, min_periods=1).mean().values

    feat_df = pd.DataFrame({
        "time":         swing["time"].values,
        "vpin":         _smooth(vpin),
        "sig_quad_var": _smooth(qv),
        "har_rv_ratio": _smooth(har),
        "hawkes_eta":   _smooth(eta),
    })

    print("\n  Sanity:", flush=True)
    for col in NEW_COLS:
        s = feat_df[col]
        nan_pct = 100 * s.isna().mean()
        print(f"    {col:<14} mean={s.mean():+.4f}  std={s.std():.4f}  "
              f"min={s.min():+.4f}  max={s.max():+.4f}  NaN%={nan_pct:.2f}", flush=True)

    feat_df = feat_df.fillna(0)

    # Merge onto v6 setup CSVs → setups_{cid}_v72l.csv
    # Filter to canonical setups_{N}_v6.csv only (skip junk variants like
    # setups_0_v6_v6.csv, setups_0_btc_v6.csv, setups_0_v72l_v6.csv).
    import re
    for f in sorted(glob.glob(P.data("setups_*_v6.csv"))):
        base = os.path.basename(f).replace(".csv","")
        if not re.fullmatch(r"setups_\d+_v6", base):
            continue
        cid = os.path.basename(f).split("_")[1]
        setup = pd.read_csv(f, parse_dates=["time"])
        merged = setup.merge(feat_df, on="time", how="left")
        for col in NEW_COLS:
            merged[col] = merged[col].fillna(0)
        out = P.data(f"setups_{cid}_v72l.csv")
        merged.to_csv(out, index=False)
        print(f"  Saved: {out} ({len(merged):,} rows)", flush=True)

    print(f"\nTotal: {_time.time() - t_total:.0f}s", flush=True)


if __name__ == "__main__":
    main()
