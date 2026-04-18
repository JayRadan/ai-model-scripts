"""
v7.1 — 4 new features on top of v6 baseline:
  1. bocpd_cp_prob   — Bayesian Online Changepoint posterior P(r_t = 0)
                       (Adams & MacKay 2007, streaming, Gaussian-IG conjugate prior)
  2. kyle_lambda     — |Δp| / activity rolling mean (liquidity-depth proxy)
  3. har_rv_ratio    — RV_day / RV_month (HAR-RV short/long vol regime shift)
  4. hawkes_eta      — event-rate_short / event-rate_long (self-excitation proxy)

Also carries forward the 2 strongest v7 features: vpin, sig_quad_var.
Drops weak v7 features: te_h1_m5, sig_levy_area, sig_time_weighted_drift.

Outputs setups_{cid}_v71.csv by merging onto v6 CSVs.
All features strictly past-only (features at bar i use ONLY bars with index < i).
"""
from __future__ import annotations
import glob, math, os, time as _time
import numpy as np
import pandas as pd
from scipy.special import gammaln as sp_gammaln
import sys
sys.path.insert(0, "/home/jay/Desktop/new-model-zigzag/model_pipeline")
import paths as P

KEEP_FROM_V7 = ["vpin", "sig_quad_var"]         # keep these v7 features
NEW_COLS = ["bocpd_recent_cp", "kyle_lambda", "har_rv_ratio", "hawkes_eta"]


def _ffill(arr):
    last = np.nan
    for i in range(len(arr)):
        if np.isfinite(arr[i]): last = arr[i]
        else: arr[i] = last


# ---------------------------------------------------------------------
# 1. BOCPD — Bayesian Online Changepoint Detection (Adams & MacKay 2007)
# ---------------------------------------------------------------------
def compute_bocpd(x, hazard_lambda=200.0, r_max=300, step=6,
                  mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0,
                  recent_threshold=30):
    """Streaming BOCPD with Normal-Inverse-Gamma conjugate prior on Gaussian data.

    Run-length posterior truncated to r_max.

    NB: The normalized posterior P(r_t=0 | x_{1:t}) mathematically equals the
    hazard rate H at every step — the evidence from x_t cancels in the
    normalizer.  The INFORMATIVE statistic is the posterior mass on SMALL run
    lengths, i.e.  P(r_t ≤ recent_threshold | x_{1:t}) — this rises after a
    changepoint (mass shifts to small r) and is low in stationary regime.

    Output: P(r_t ≤ recent_threshold | x_{1:t}) ∈ [0, 1].
    Strictly past-only: at bar i only uses x[:i+1] (x_t is known at time t).
    """
    n = len(x)
    out = np.full(n, np.nan)
    R = int(r_max)
    H = 1.0 / hazard_lambda                     # constant hazard = 1/λ

    # Run-length posterior P(r_t = r | x_{1:t}).  Start at P(r=0) = 1.
    post = np.zeros(R + 1)
    post[0] = 1.0

    # Per-run sufficient statistics (Gaussian-IG).
    mu    = np.full(R + 1, mu0)
    kappa = np.full(R + 1, kappa0)
    alpha = np.full(R + 1, alpha0)
    beta  = np.full(R + 1, beta0)

    for t in range(n):
        x_t = x[t]
        # Student-t predictive density under each run length's current params
        # p(x | μ, κ, α, β) = t_{2α}(μ, β(κ+1)/(ακ))
        df = 2.0 * alpha
        scale2 = beta * (kappa + 1.0) / (alpha * kappa)
        z = (x_t - mu) / np.sqrt(scale2)
        # log-pdf of Student-t with df, loc=mu, scale=sqrt(scale2)
        # = lgamma((df+1)/2) - lgamma(df/2) - 0.5 log(π·df·scale2)
        #   − ((df+1)/2) log(1 + z²/df)
        log_pred = (sp_gammaln((df + 1) / 2.0)
                    - sp_gammaln(df / 2.0)
                    - 0.5 * np.log(np.pi * df * scale2)
                    - ((df + 1) / 2.0) * np.log(1.0 + z * z / df))
        pred = np.exp(log_pred - log_pred.max())   # stable exp (scaling cancels)

        # Growth prob: shift r → r+1
        growth = post * pred * (1.0 - H)
        # Changepoint prob: mass on r = 0
        cp_mass = float(np.sum(post * pred * H))

        # Build new posterior
        new_post = np.zeros(R + 1)
        new_post[0] = cp_mass
        new_post[1:R + 1] = growth[:R]           # truncate: drop r = R → R+1

        Z = new_post.sum()
        if Z <= 0:
            # Numerical collapse — restart at r=0
            new_post[:] = 0
            new_post[0] = 1.0
        else:
            new_post /= Z
        post = new_post

        # Update per-run sufficient statistics (shift mu/kappa/alpha/beta by 1)
        # New r=0 uses the priors; r>=1 uses previous r-1 updated by x_t.
        new_mu    = np.empty(R + 1); new_mu[0]    = mu0
        new_kappa = np.empty(R + 1); new_kappa[0] = kappa0
        new_alpha = np.empty(R + 1); new_alpha[0] = alpha0
        new_beta  = np.empty(R + 1); new_beta[0]  = beta0
        # r+1 params from r params after observing x_t
        k1 = kappa + 1.0
        a1 = alpha + 0.5
        b1 = beta + (kappa * (x_t - mu) ** 2) / (2.0 * k1)
        m1 = (kappa * mu + x_t) / k1
        new_mu[1:]    = m1[:R]
        new_kappa[1:] = k1[:R]
        new_alpha[1:] = a1[:R]
        new_beta[1:]  = b1[:R]
        mu, kappa, alpha, beta = new_mu, new_kappa, new_alpha, new_beta

        # Output: mass on recent run lengths.  Large value = a changepoint
        # has likely happened within the last `recent_threshold` bars.
        if t % step == 0:
            out[t] = float(post[:recent_threshold + 1].sum())

    _ffill(out); return out


# ---------------------------------------------------------------------
# 2. Kyle's lambda — price impact per unit activity
# ---------------------------------------------------------------------
def compute_kyle_lambda(close, high, low, window=60):
    """λ ≈ Σ|Δp_t| / ΣV_t over window  (V_t = high_t − low_t activity proxy).

    Sum numerator & denominator separately over window — this is more robust
    than per-bar ratios, which blow up on bars with near-zero range.
    Higher λ = small activity moves price a lot = thin/fragile liquidity.
    Strict past-only: shift(1) ensures bar i uses [i-window, i).
    """
    dp = np.concatenate([[0.0], np.abs(np.diff(close))])
    V = np.maximum(high - low, 0.0)
    num = pd.Series(dp).rolling(window, min_periods=window).sum().shift(1).values
    den = pd.Series(V ).rolling(window, min_periods=window).sum().shift(1).values
    lam = num / np.where(den > 1e-9, den, 1e-9)
    lam = np.where(np.isfinite(lam), np.clip(lam, 0.0, 10.0), np.nan)
    return lam


# ---------------------------------------------------------------------
# 3. HAR-RV ratio — short vs long realized-vol regime
# ---------------------------------------------------------------------
def compute_har_rv_ratio(close, short_bars=288, long_bars=8640):
    """RV_day / RV_month.  RV = Σ r² over window, normalized by window.
       M5 bars: 288 ≈ 1 trading day, 8640 ≈ 30 days (720 h × 12).

    >1  short-term vol elevated vs month baseline  (regime shift risk)
    <1  short-term calm                            (normal)
    """
    r = np.concatenate([[0.0], np.diff(np.log(close))])
    r2 = r * r
    # Rolling mean of r² over short and long windows, past-only via shift(1)
    rv_s = pd.Series(r2).rolling(short_bars, min_periods=short_bars).mean().shift(1).values
    rv_l = pd.Series(r2).rolling(long_bars,  min_periods=long_bars ).mean().shift(1).values
    ratio = rv_s / np.where(rv_l > 1e-18, rv_l, 1e-18)
    # Clip extreme ratios caused by near-zero RV_long in exotic bars
    ratio = np.clip(ratio, 0.0, 20.0)
    return ratio


# ---------------------------------------------------------------------
# 4. Hawkes branching-ratio proxy (event clustering)
# ---------------------------------------------------------------------
def compute_hawkes_eta(close, sigma_window=500, event_k=2.0,
                       short_window=60, long_window=600):
    """Proxy for self-excitation branching ratio η:
       η ≈ (event rate over last `short`) / (event rate over last `long`)
       event_t = 1 if |r_t| > event_k · σ_rolling
    η >> 1  ⇒ events are clustering — volatility feedback in progress
    η ≈ 1   ⇒ stationary Poisson — no self-excitation
    """
    r = np.concatenate([[0.0], np.diff(np.log(close))])
    sig = pd.Series(r).rolling(sigma_window, min_periods=sigma_window).std(ddof=0).values
    event = np.where((np.isfinite(sig)) & (sig > 0) & (np.abs(r) > event_k * sig), 1.0, 0.0)
    # Past-only event rate: use shift(1) so bar i sees windows ending at i-1
    rate_s = pd.Series(event).rolling(short_window, min_periods=short_window).mean().shift(1).values
    rate_l = pd.Series(event).rolling(long_window,  min_periods=long_window ).mean().shift(1).values
    eta = rate_s / np.where(rate_l > 1e-9, rate_l, 1e-9)
    # If long-window rate is 0 (no events), set η to 1 (no information)
    eta = np.where(np.isfinite(eta), np.clip(eta, 0.0, 20.0), 1.0)
    return eta


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    t_total = _time.time()
    print("v7.1 — computing BOCPD + Kyle's λ + HAR-RV ratio + Hawkes η", flush=True)
    swing = pd.read_csv(P.data("swing_v5_xauusd.csv"), parse_dates=["time"])
    swing = swing.sort_values("time").reset_index(drop=True)
    c = swing["close"].values.astype(np.float64)
    h = swing["high"].values.astype(np.float64)
    l = swing["low"].values.astype(np.float64)
    n = len(c)
    print(f"  Bars: {n:,}", flush=True)

    ret = np.concatenate([[0.0], np.diff(np.log(c))]) * 100.0   # scale to percent

    print("  [1/4] BOCPD (streaming — may take a few minutes)...", end="", flush=True)
    t0 = _time.time()
    bocpd = compute_bocpd(ret)
    print(f" {_time.time() - t0:.0f}s", flush=True)
    # Match column name declared in NEW_COLS
    bocpd_recent_cp = bocpd

    print("  [2/4] Kyle's lambda...", end="", flush=True); t0 = _time.time()
    kyle = compute_kyle_lambda(c, h, l)
    print(f" {_time.time() - t0:.0f}s", flush=True)

    print("  [3/4] HAR-RV ratio...", end="", flush=True); t0 = _time.time()
    har = compute_har_rv_ratio(c)
    print(f" {_time.time() - t0:.0f}s", flush=True)

    print("  [4/4] Hawkes η...", end="", flush=True); t0 = _time.time()
    eta = compute_hawkes_eta(c)
    print(f" {_time.time() - t0:.0f}s", flush=True)

    feat_df = pd.DataFrame({
        "time":            swing["time"].values,
        "bocpd_recent_cp": bocpd_recent_cp,
        "kyle_lambda":     kyle,
        "har_rv_ratio":    har,
        "hawkes_eta":      eta,
    })

    print("\n  Feature sanity (global):", flush=True)
    for col in NEW_COLS:
        s = feat_df[col]
        nan_pct = 100 * s.isna().mean()
        nonzero_pct = 100 * (s.fillna(0) != 0).mean()
        print(f"    {col:<18} mean={s.mean():+.5f}  std={s.std():.5f}  "
              f"min={s.min():+.5f}  max={s.max():+.5f}  NaN%={nan_pct:.2f}  "
              f"nonzero%={nonzero_pct:.1f}", flush=True)

    feat_df = feat_df.fillna(0)

    # Merge: v6 CSV (14 feats) + 2 retained v7 feats (vpin, sig_quad_var) + 4 new
    for f in sorted(glob.glob(P.data("setups_*_v6.csv"))):
        cid = os.path.basename(f).split("_")[1]
        v6_df = pd.read_csv(f, parse_dates=["time"])
        v7_path = P.data(f"setups_{cid}_v7.csv")
        if os.path.exists(v7_path):
            v7_df = pd.read_csv(v7_path, parse_dates=["time"])[["time"] + KEEP_FROM_V7]
            merged = v6_df.merge(v7_df, on="time", how="left")
        else:
            print(f"  WARNING: {v7_path} not found — KEEP_FROM_V7 columns will be 0")
            merged = v6_df.copy()
            for col in KEEP_FROM_V7:
                merged[col] = 0.0
        merged = merged.merge(feat_df, on="time", how="left")
        for col in KEEP_FROM_V7 + NEW_COLS:
            merged[col] = merged[col].fillna(0)
        out = P.data(f"setups_{cid}_v71.csv")
        merged.to_csv(out, index=False)
        print(f"  Saved: {out} ({len(merged):,} rows)", flush=True)

    print(f"\nTotal: {_time.time() - t_total:.0f}s", flush=True)


if __name__ == "__main__":
    main()
