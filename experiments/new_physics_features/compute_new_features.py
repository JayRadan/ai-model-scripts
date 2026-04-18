"""
Compute 7 new candidate physics features at every bar of the XAU swing CSV,
then merge into per-cluster setup CSVs as setups_{cid}_v6b.csv.

New features:
  1. permutation_entropy  — Bandt-Pompe ordinal entropy (order m=3)
  2. dfa_alpha            — Detrended Fluctuation Analysis scaling exponent
  3. higuchi_fd           — Higuchi Fractal Dimension
  4. spectral_entropy     — Shannon entropy of FFT power spectrum
  5. hill_tail_index      — Hill estimator of tail index (fatness)
  6. vol_of_vol           — std of rolling-std of returns (regime stability)
  7. log_drift            — slope of linear regression on cumulative log returns
"""
from __future__ import annotations
import glob, math, os, time as _time
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
import sys
sys.path.insert(0, "/home/jay/Desktop/new-model-zigzag/model_pipeline")
import paths as P

V6B_FEAT_COLS = [
    # Existing 14
    "hurst_rs", "ou_theta", "entropy_rate", "kramers_up", "wavelet_er",
    "vwap_dist", "hour_enc", "dow_enc",
    "quantum_flow", "quantum_flow_h4", "quantum_momentum", "quantum_vwap_conf",
    "quantum_divergence", "quantum_div_strength",
    # New 7
    "permutation_entropy", "dfa_alpha", "higuchi_fd",
    "spectral_entropy", "hill_tail_index", "vol_of_vol", "log_drift",
]


def _ffill(arr):
    last = np.nan
    for i in range(len(arr)):
        if np.isfinite(arr[i]): last = arr[i]
        else: arr[i] = last


# ---------------------------------------------------------------------
# 1. Permutation Entropy (Bandt & Pompe 2002) — m=3, lag=1
# ---------------------------------------------------------------------
def compute_permutation_entropy(x, window=100, m=3, step=6):
    """For each window, compute entropy of ordinal patterns of length m.
    With m=3 there are 6 possible orderings (3! = 6). max entropy = log2(6)."""
    n = len(x)
    pe = np.full(n, np.nan)
    max_ent = np.log2(math.factorial(m))
    for i in range(window, n, step):
        seg = x[i - window:i]
        # Get ordinal patterns via argsort
        patterns = np.zeros(len(seg) - m + 1, dtype=np.int32)
        for j in range(len(seg) - m + 1):
            triplet = seg[j:j + m]
            # Map each permutation to an integer 0..m!-1 via lexicographic order
            order = np.argsort(triplet, kind='stable')
            # Encode as base-m integer
            patterns[j] = order[0] * m * m + order[1] * m + order[2] if m == 3 else hash(tuple(order)) % 1000
        counts = np.bincount(patterns, minlength=m * m * m)
        p = counts[counts > 0] / counts.sum()
        pe[i] = -np.sum(p * np.log2(p)) / max_ent
    _ffill(pe); return pe


# ---------------------------------------------------------------------
# 2. DFA — Detrended Fluctuation Analysis (Peng 1994)
# ---------------------------------------------------------------------
def compute_dfa_alpha(x, window=120, step=6, scales=(4, 8, 16, 32, 64)):
    """α from log-log fit of fluctuation F(s) vs scale s.
    α = 0.5 random walk, α > 0.5 persistent, α < 0.5 anti-persistent."""
    n = len(x)
    alpha = np.full(n, np.nan)
    scales = [s for s in scales if s <= window // 2]
    for i in range(window, n, step):
        seg = x[i - window:i]
        y = np.cumsum(seg - seg.mean())  # integrated series
        Fs = []
        for s in scales:
            n_segs = len(y) // s
            if n_segs < 2: continue
            y_trimmed = y[:n_segs * s].reshape(n_segs, s)
            t = np.arange(s)
            # Detrend each segment with linear fit, compute RMS
            rms_sum = 0
            for row in y_trimmed:
                p = np.polyfit(t, row, 1)
                detrended = row - np.polyval(p, t)
                rms_sum += np.sum(detrended ** 2)
            F = np.sqrt(rms_sum / (n_segs * s))
            Fs.append((s, F))
        if len(Fs) < 2: continue
        ls = np.log([s for s, _ in Fs])
        lF = np.log([max(f, 1e-15) for _, f in Fs])
        slope = np.polyfit(ls, lF, 1)[0]
        alpha[i] = slope
    _ffill(alpha); return alpha


# ---------------------------------------------------------------------
# 3. Higuchi Fractal Dimension (Higuchi 1988)
# ---------------------------------------------------------------------
def compute_higuchi_fd(x, window=100, kmax=8, step=6):
    """HFD from log-log fit of curve length L(k) vs 1/k.
    Range [1, 2]: 1 = smooth, 2 = very rough (near-random walk)."""
    n = len(x)
    fd = np.full(n, np.nan)
    for i in range(window, n, step):
        seg = x[i - window:i]
        L_over_k = []
        for k in range(1, kmax + 1):
            Lm = 0
            for m in range(k):
                idx = np.arange(m, len(seg), k)
                if len(idx) < 2: continue
                sub = seg[idx]
                norm = (len(seg) - 1) / (k * (len(idx) - 1))
                Lk = np.sum(np.abs(np.diff(sub))) * norm / k
                Lm += Lk
            if Lm > 0:
                L_over_k.append((1.0 / k, Lm / k))
        if len(L_over_k) < 3: continue
        log_x = np.log([p[0] for p in L_over_k])
        log_y = np.log([p[1] for p in L_over_k])
        fd[i] = -np.polyfit(log_x, log_y, 1)[0]
    _ffill(fd); return fd


# ---------------------------------------------------------------------
# 4. Spectral Entropy
# ---------------------------------------------------------------------
def compute_spectral_entropy(x, window=128, step=6):
    """Shannon entropy of normalized power spectrum, 0..1 (1 = flat spectrum / noise)."""
    n = len(x)
    se = np.full(n, np.nan)
    max_ent = np.log2(window // 2)
    for i in range(window, n, step):
        seg = x[i - window:i]
        seg = seg - seg.mean()
        spectrum = np.abs(np.fft.rfft(seg)) ** 2
        spectrum = spectrum[1:]  # drop DC
        total = spectrum.sum()
        if total < 1e-15: continue
        p = spectrum / total
        p = p[p > 0]
        se[i] = -np.sum(p * np.log2(p)) / max_ent
    _ffill(se); return se


# ---------------------------------------------------------------------
# 5. Hill Tail Index (Hill 1975)
# ---------------------------------------------------------------------
def compute_hill_tail_index(x, window=200, step=6, k_frac=0.1):
    """Fatness of right tail via Hill estimator. Larger = fatter tails.
    k = top 10% of |returns| used to estimate tail."""
    n = len(x)
    hti = np.full(n, np.nan)
    k_fixed = int(window * k_frac)
    for i in range(window, n, step):
        abs_seg = np.abs(x[i - window:i])
        sorted_desc = np.sort(abs_seg)[::-1]
        if k_fixed < 2 or sorted_desc[k_fixed] < 1e-15: continue
        ratios = sorted_desc[:k_fixed] / sorted_desc[k_fixed]
        ratios = ratios[ratios > 1e-15]
        if len(ratios) == 0: continue
        xi = np.mean(np.log(ratios))
        hti[i] = xi
    _ffill(hti); return hti


# ---------------------------------------------------------------------
# 6. Vol of Vol
# ---------------------------------------------------------------------
def compute_vol_of_vol(x, inner=20, outer=60):
    """std of (rolling-std of returns). Regime-stability indicator.
    Vectorized via pandas."""
    s = pd.Series(x)
    inner_std = s.rolling(inner, min_periods=inner).std()
    outer_vov = inner_std.rolling(outer, min_periods=outer).std()
    return outer_vov.fillna(method='ffill').fillna(0).values


# ---------------------------------------------------------------------
# 7. Log-return drift µ
# ---------------------------------------------------------------------
def compute_log_drift(x, window=100, step=6):
    """Slope of linear regression on cumulative log returns over window.
    Positive = up-trend drift, negative = down-trend drift."""
    n = len(x)
    drift = np.full(n, np.nan)
    t = np.arange(window, dtype=np.float64)
    for i in range(window, n, step):
        y = np.cumsum(x[i - window:i])
        drift[i] = np.polyfit(t, y, 1)[0]
    _ffill(drift); return drift


def main():
    print("Computing 7 new physics features on XAU swing CSV", flush=True)
    t_total = _time.time()
    df = pd.read_csv(P.data("swing_v5_xauusd.csv"), parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)
    c = df["close"].values.astype(np.float64)
    n = len(c)
    print(f"  Bars: {n:,}", flush=True)
    ret = np.concatenate([[0.0], np.diff(np.log(c))])

    print("  [1/7] Permutation entropy...", end="", flush=True); t = _time.time()
    df["permutation_entropy"] = compute_permutation_entropy(ret)
    print(f" {_time.time() - t:.0f}s", flush=True)

    print("  [2/7] DFA alpha...", end="", flush=True); t = _time.time()
    df["dfa_alpha"] = compute_dfa_alpha(ret)
    print(f" {_time.time() - t:.0f}s", flush=True)

    print("  [3/7] Higuchi FD...", end="", flush=True); t = _time.time()
    df["higuchi_fd"] = compute_higuchi_fd(c)
    print(f" {_time.time() - t:.0f}s", flush=True)

    print("  [4/7] Spectral entropy...", end="", flush=True); t = _time.time()
    df["spectral_entropy"] = compute_spectral_entropy(ret)
    print(f" {_time.time() - t:.0f}s", flush=True)

    print("  [5/7] Hill tail index...", end="", flush=True); t = _time.time()
    df["hill_tail_index"] = compute_hill_tail_index(ret)
    print(f" {_time.time() - t:.0f}s", flush=True)

    print("  [6/7] Vol of vol...", end="", flush=True); t = _time.time()
    df["vol_of_vol"] = compute_vol_of_vol(ret)
    print(f" {_time.time() - t:.0f}s", flush=True)

    print("  [7/7] Log drift...", end="", flush=True); t = _time.time()
    df["log_drift"] = compute_log_drift(ret)
    print(f" {_time.time() - t:.0f}s", flush=True)

    new_cols = ["permutation_entropy", "dfa_alpha", "higuchi_fd",
                "spectral_entropy", "hill_tail_index", "vol_of_vol", "log_drift"]
    new_phys = df[["time"] + new_cols].copy().fillna(0)

    # Merge into existing v6 setup CSVs — same rows, added columns
    for f in sorted(glob.glob(P.data("setups_*_v6.csv"))):
        cid = os.path.basename(f).split("_")[1]
        setup = pd.read_csv(f, parse_dates=["time"])
        merged = setup.merge(new_phys, on="time", how="left", suffixes=("", "_new"))
        for col in new_cols:
            if col + "_new" in merged.columns:
                merged[col] = merged[col + "_new"].fillna(merged.get(col, 0))
                merged.drop(columns=[col + "_new"], inplace=True)
            merged[col] = merged[col].fillna(0)
        out = P.data(f"setups_{cid}_v6b.csv")
        merged.to_csv(out, index=False)
        print(f"  Saved: {out} ({len(merged):,} rows)", flush=True)

    print(f"\nTotal: {_time.time() - t_total:.0f}s", flush=True)


if __name__ == "__main__":
    main()
