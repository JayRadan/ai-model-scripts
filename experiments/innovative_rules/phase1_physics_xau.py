"""
Phase 1: Physics / information-theory / stochastic-process rules for XAU.

Rules derived from:
  1. Local Hurst exponent (DFA) — persistence transitions
  2. Ornstein-Uhlenbeck θ estimation — mean-reversion speed
  3. Shannon entropy rate — predictability regime
  4. Kramers escape rate — potential well breakout
  5. Lévy α-stable tail index — jump regime detection
  6. Transfer entropy H1→M5 — causal information flow
  7. Fokker-Planck local drift/diffusion ratio
  8. Wavelet energy ratio (CWT trend vs noise scale)

All vectorized where possible. Forward-outcome at TP=6×ATR, SL=2×ATR, MAX_FWD=40.
"""
from __future__ import annotations
import os, sys, time as _time
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy.stats import levy_stable
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/jay/Desktop/new-model-zigzag"
OUT_DIR = f"{ROOT}/experiments/innovative_rules"
os.makedirs(OUT_DIR, exist_ok=True)

TP_MULT, SL_MULT, MAX_FWD = 6.0, 2.0, 40


# ═══════════════════════ Data loader ═══════════════════════
def load_data():
    df = pd.read_csv(f"{ROOT}/data/swing_v5_xauusd.csv",
                     parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    c = df["close"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    o = df["open"].values.astype(np.float64)
    n = len(c)
    tr = np.empty(n); tr[0] = h[0]-l[0]
    tr[1:] = np.maximum.reduce([h[1:]-l[1:], np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])])
    atr = pd.Series(tr).rolling(14, min_periods=14).mean().values
    ret = np.concatenate([[0.0], np.diff(np.log(c))])  # log returns
    return dict(df=df, c=c, h=h, l=l, o=o, atr=atr, ret=ret, n=n)


def forward_outcome(d, i, direction):
    a = d["atr"][i]
    if not np.isfinite(a) or a <= 0: return -1
    e = d["c"][i]
    H, L = d["h"], d["l"]
    end = min(i + 1 + MAX_FWD, d["n"])
    if direction == 1:
        tp, sl = e + TP_MULT*a, e - SL_MULT*a
        for k in range(i+1, end):
            if L[k] <= sl: return 0
            if H[k] >= tp: return 1
    else:
        tp, sl = e - TP_MULT*a, e + SL_MULT*a
        for k in range(i+1, end):
            if H[k] >= sl: return 0
            if L[k] <= tp: return 1
    return 2  # expired


def _ffill(arr):
    """In-place forward-fill NaN values."""
    last = np.nan
    for i in range(len(arr)):
        if np.isfinite(arr[i]): last = arr[i]
        else: arr[i] = last

# ═══════════════════════ Vectorized precompute ═══════════════════════

def rolling_hurst_rs(ret, window=120, step=6):
    """Rescaled Range (R/S) Hurst estimator — fast, computed every `step` bars."""
    n = len(ret)
    hurst = np.full(n, np.nan)
    for i in range(window, n, step):
        seg = ret[i-window:i]
        m = seg.mean()
        y = np.cumsum(seg - m)
        R = y.max() - y.min()
        S = seg.std()
        if S > 1e-15 and R > 0:
            hurst[i] = np.log(R / S) / np.log(window)
    # forward-fill gaps from stepping
    last = np.nan
    for i in range(n):
        if np.isfinite(hurst[i]): last = hurst[i]
        else: hurst[i] = last
    return hurst


def rolling_ou_theta(ret, window=60, step=6):
    n = len(ret); theta = np.full(n, np.nan)
    x = np.cumsum(ret)
    for i in range(window, n, step):
        seg = x[i-window:i]; mu = seg.mean()
        dx = np.diff(seg); xm = seg[:-1] - mu
        denom = np.sum(xm**2)
        if denom > 1e-15: theta[i] = -np.sum(dx * xm) / denom
    _ffill(theta)
    return theta


def rolling_entropy(ret, window=100, n_bins=10, step=6):
    n = len(ret); ent = np.full(n, np.nan)
    for i in range(window, n, step):
        seg = ret[i-window:i]
        counts, _ = np.histogram(seg, bins=n_bins)
        p = counts / counts.sum(); p = p[p > 0]
        ent[i] = -np.sum(p * np.log2(p))
    _ffill(ent)
    return ent, np.log2(n_bins)


def rolling_kramers(c, window=100, step=6):
    n = len(c)
    escape_up = np.full(n, np.nan); escape_dn = np.full(n, np.nan)
    for i in range(window, n, step):
        seg = c[i-window:i]; hi, lo = seg.max(), seg.min()
        sigma = np.std(np.diff(np.log(seg + 1e-10)))
        if sigma < 1e-12: continue
        barrier_up = (hi - c[i]) / (sigma * c[i] + 1e-10)
        barrier_dn = (c[i] - lo) / (sigma * c[i] + 1e-10)
        escape_up[i] = np.exp(-barrier_up); escape_dn[i] = np.exp(-barrier_dn)
    _ffill(escape_up); _ffill(escape_dn)
    return escape_up, escape_dn


def rolling_levy_alpha(ret, window=200, step=12):
    n = len(ret); alpha = np.full(n, np.nan)
    for i in range(window, n, step):
        seg = np.abs(ret[i-window:i]); seg = seg[seg > 1e-10]
        if len(seg) < 50: continue
        seg_sorted = np.sort(seg)[::-1]
        k = max(10, len(seg) // 10); top_k = seg_sorted[:k]
        threshold = seg_sorted[k]
        if threshold < 1e-12: continue
        alpha[i] = k / np.sum(np.log(top_k / threshold))
    _ffill(alpha)
    return alpha


def rolling_transfer_entropy(ret_m5, window=120, h1_bars=12, step=12):
    """Transfer entropy H1→M5, computed every `step` bars."""
    n = len(ret_m5); te = np.full(n, np.nan); nb = 5
    for i in range(window + h1_bars, n, step):
        try:
            x_cur = ret_m5[i-window:i]; x_prev = ret_m5[i-window-1:i-1]
            x_h1 = np.array([ret_m5[j-h1_bars:j].sum() for j in range(i-window, i)])
            bc = np.linspace(x_cur.min()-1e-10, x_cur.max()+1e-10, nb+1)
            bp = np.linspace(x_prev.min()-1e-10, x_prev.max()+1e-10, nb+1)
            bh = np.linspace(x_h1.min()-1e-10, x_h1.max()+1e-10, nb+1)
            h2, _, _ = np.histogram2d(x_cur, x_prev, bins=[bc, bp])
            p_cp = h2/h2.sum(); p_p = p_cp.sum(axis=0)
            h_cp = -np.nansum(np.where((p_cp>0)&(p_p[None,:]>0), p_cp*np.log2(p_cp/(p_p[None,:]+1e-15)), 0))
            dc = np.clip(np.digitize(x_cur,bc)-1, 0, nb-1)
            dp = np.clip(np.digitize(x_prev,bp)-1, 0, nb-1)
            dh = np.clip(np.digitize(x_h1,bh)-1, 0, nb-1)
            h3 = np.zeros((nb,nb,nb))
            for k in range(len(dc)): h3[dc[k],dp[k],dh[k]] += 1
            p3 = h3/h3.sum(); p_ph = p3.sum(axis=0)
            h_cph = 0
            for a in range(nb):
                for b in range(nb):
                    for g in range(nb):
                        if p3[a,b,g]>0 and p_ph[b,g]>0:
                            h_cph -= p3[a,b,g]*np.log2(p3[a,b,g]/p_ph[b,g])
            te[i] = h_cp - h_cph
        except: pass
    _ffill(te)
    return te


def rolling_fp_drift_diffusion(ret, window=60):
    """Fokker-Planck drift/diffusion — vectorized via rolling mean/var."""
    s = pd.Series(ret)
    mu = s.rolling(window, min_periods=window).mean().values
    D  = s.rolling(window, min_periods=window).var().values
    ratio = np.where(D > 1e-15, mu / np.sqrt(D), np.nan)
    return ratio, mu


def _ricker(points, a):
    """Ricker (Mexican hat) wavelet."""
    A = 2 / (np.sqrt(3 * a) * (np.pi**0.25))
    wsq = a**2
    vec = np.arange(0, points) - (points - 1.0) / 2
    tsq = vec**2
    mod = (1 - tsq / wsq)
    gauss = np.exp(-tsq / (2 * wsq))
    return A * mod * gauss

def rolling_wavelet_energy(c, window=120, step=12):
    """CWT energy ratio: trend scale (20) vs noise scale (3). Every `step` bars."""
    n = len(c); eratio = np.full(n, np.nan)
    kernel_t = _ricker(min(10*20, window), 20)
    kernel_n = _ricker(min(10*3, window), 3)
    for i in range(window, n, step):
        seg = c[i-window:i]
        seg_norm = (seg - seg.mean()) / (seg.std() + 1e-10)
        ct = np.convolve(seg_norm, kernel_t, mode='same')
        cn = np.convolve(seg_norm, kernel_n, mode='same')
        eratio[i] = np.mean(ct**2) / (np.mean(cn**2) + 1e-15)
    _ffill(eratio)
    return eratio


# ═══════════════════════ Rule definitions ═══════════════════════

def rule_1_hurst_transition(d, i):
    """Hurst crosses 0.5 boundary — persistence↔antipersistence transition."""
    if i < 2: return 0
    h_now, h_prev = d["hurst"][i], d["hurst"][i-1]
    if not (np.isfinite(h_now) and np.isfinite(h_prev)): return 0
    # crossed from persistent (>0.55) to anti-persistent (<0.45) → fade
    if h_prev > 0.55 and h_now < 0.45:
        return -1 if d["ret"][i-1] > 0 else +1
    # crossed from anti-persistent to persistent → follow momentum
    if h_prev < 0.45 and h_now > 0.55:
        return +1 if d["ret"][i-1] > 0 else -1
    return 0


def rule_2_ou_reversion(d, i):
    """OU θ spikes above 2σ of its own distribution → strong mean reversion expected."""
    if i < 200: return 0
    th = d["ou_theta"][i]
    if not np.isfinite(th): return 0
    window = d["ou_theta"][i-199:i+1]
    window = window[np.isfinite(window)]
    if len(window) < 50: return 0
    mu, sigma = window.mean(), window.std()
    if sigma < 1e-12: return 0
    z = (th - mu) / sigma
    if z > 2.0:
        # strong reversion → fade recent move
        cum = d["ret"][i-9:i+1].sum()
        if cum > 0: return -1
        if cum < 0: return +1
    return 0


def rule_3_entropy_drop(d, i):
    """Entropy drops below 60% of max → market becomes predictable → follow structure."""
    ent = d["entropy"][i]
    if not np.isfinite(ent): return 0
    if ent < 0.60 * d["max_entropy"]:
        # low entropy = strong pattern. Follow the dominant direction.
        cum = d["ret"][i-4:i+1].sum()
        if cum > 0: return +1
        if cum < 0: return -1
    return 0


def rule_4_kramers_escape(d, i):
    """Kramers escape rate high → breakout imminent. Trade the closer edge."""
    eu, ed = d["escape_up"][i], d["escape_dn"][i]
    if not (np.isfinite(eu) and np.isfinite(ed)): return 0
    # high escape probability AND directional bias
    if eu > 0.7 and eu > 1.5 * ed: return +1  # upside breakout likely
    if ed > 0.7 and ed > 1.5 * eu: return -1
    return 0


def rule_5_levy_jump(d, i):
    """Lévy α < 1.5 → fat-tail jump regime. Trade momentum of the jump."""
    if i < 3: return 0
    a = d["levy_alpha"][i]
    if not np.isfinite(a): return 0
    if a < 1.5:
        # in jump regime — ride the last jump's direction
        if d["ret"][i] > 0 and abs(d["ret"][i]) > 1.5 * d["atr"][i] / d["c"][i]: return +1
        if d["ret"][i] < 0 and abs(d["ret"][i]) > 1.5 * d["atr"][i] / d["c"][i]: return -1
    return 0


def rule_6_transfer_entropy(d, i):
    """High TE from H1→M5 → M5 is predictable from H1. Trade H1 direction."""
    te_val = d["te"][i]
    if not np.isfinite(te_val): return 0
    if i < 200: return 0
    window = d["te"][i-199:i+1]
    window = window[np.isfinite(window)]
    if len(window) < 50: return 0
    p90 = np.percentile(window, 90)
    if te_val > p90 and te_val > 0.05:
        h1_dir = d["ret"][i-12:i].sum()
        if h1_dir > 0: return +1
        if h1_dir < 0: return -1
    return 0


def rule_7_fp_drift(d, i):
    """Fokker-Planck drift/diffusion ratio exceeds ±2 → strong directional force."""
    ratio = d["fp_ratio"][i]
    if not np.isfinite(ratio): return 0
    if ratio > 2.0:  return +1
    if ratio < -2.0: return -1
    return 0


def rule_8_wavelet_energy(d, i):
    """Wavelet trend/noise energy ratio > 5 → clean trend signal. Follow it."""
    er = d["wavelet_eratio"][i]
    if not np.isfinite(er): return 0
    if er > 5.0:
        cum = d["ret"][i-11:i+1].sum()
        if cum > 0: return +1
        if cum < 0: return -1
    return 0


RULES = [
    ("01_hurst_transition",    rule_1_hurst_transition),
    ("02_ou_reversion",        rule_2_ou_reversion),
    ("03_entropy_drop",        rule_3_entropy_drop),
    ("04_kramers_escape",      rule_4_kramers_escape),
    ("05_levy_jump",           rule_5_levy_jump),
    ("06_transfer_entropy",    rule_6_transfer_entropy),
    ("07_fp_drift_diffusion",  rule_7_fp_drift),
    ("08_wavelet_energy",      rule_8_wavelet_energy),
]


def main():
    print("Loading XAU data…", flush=True)
    d = load_data()
    n = d["n"]
    start = max(600, n - 150_000)

    # Only precompute over the scan region + buffer (not full 1M bars)
    buf = 500  # warm-up buffer
    lo = max(0, start - buf)
    # Slice arrays for precompute range
    ret_s = d["ret"][lo:n].copy()
    c_s   = d["c"][lo:n].copy()
    offset = lo  # to map back to full-array indices

    def store(arr_short, name):
        """Map short array back to full-length array."""
        full = np.full(n, np.nan)
        full[lo:lo+len(arr_short)] = arr_short
        d[name] = full

    print("Precomputing Hurst (DFA)…", flush=True); t0=_time.time()
    store(rolling_hurst_rs(ret_s, window=120, step=6), "hurst")
    print(f"  done {_time.time()-t0:.1f}s", flush=True)

    print("Precomputing OU θ…", flush=True); t0=_time.time()
    store(rolling_ou_theta(ret_s, window=60), "ou_theta")
    print(f"  done {_time.time()-t0:.1f}s", flush=True)

    print("Precomputing entropy…", flush=True); t0=_time.time()
    ent, max_ent = rolling_entropy(ret_s, window=100)
    store(ent, "entropy"); d["max_entropy"] = max_ent
    print(f"  done {_time.time()-t0:.1f}s", flush=True)

    print("Precomputing Kramers escape…", flush=True); t0=_time.time()
    eu, ed = rolling_kramers(c_s, window=100)
    store(eu, "escape_up"); store(ed, "escape_dn")
    print(f"  done {_time.time()-t0:.1f}s", flush=True)

    print("Precomputing Lévy α…", flush=True); t0=_time.time()
    store(rolling_levy_alpha(ret_s, window=200), "levy_alpha")
    print(f"  done {_time.time()-t0:.1f}s", flush=True)

    print("Precomputing transfer entropy…", flush=True); t0=_time.time()
    store(rolling_transfer_entropy(ret_s, window=120), "te")
    print(f"  done {_time.time()-t0:.1f}s", flush=True)

    print("Precomputing Fokker-Planck drift…", flush=True); t0=_time.time()
    ratio, drift = rolling_fp_drift_diffusion(ret_s, window=60)
    store(ratio, "fp_ratio"); store(drift, "fp_drift")
    print(f"  done {_time.time()-t0:.1f}s", flush=True)

    print("Precomputing wavelet energy…", flush=True); t0=_time.time()
    store(rolling_wavelet_energy(c_s, window=120), "wavelet_eratio")
    print(f"  done {_time.time()-t0:.1f}s", flush=True)

    print(f"\nScanning bars {start:,} to {n:,} ({n-start:,} bars)", flush=True)
    results = []
    for rname, rfn in RULES:
        t0 = _time.time()
        print(f"  {rname}: ", end="", flush=True)
        signals = []
        for i in range(start, n - MAX_FWD - 1):
            s = rfn(d, i)
            if s != 0:
                out = forward_outcome(d, i, s)
                if out in (0, 1):
                    signals.append((i, s, out))
        dt = _time.time() - t0
        if not signals:
            print(f"0 signals ({dt:.1f}s)", flush=True); continue
        sig = np.array(signals)
        wins = (sig[:,2]==1).sum(); total = len(sig)
        wr = wins/total
        pf = (wr*TP_MULT)/((1-wr)*SL_MULT+1e-9)
        ev = wr*TP_MULT - (1-wr)*SL_MULT
        longs = sig[sig[:,1]==1]; shorts = sig[sig[:,1]==-1]
        lwr = (longs[:,2]==1).mean() if len(longs)>0 else 0
        swr = (shorts[:,2]==1).mean() if len(shorts)>0 else 0
        print(f"n={total:5d}  WR={wr:.1%}  PF={pf:.2f}  EV={ev:+.2f}R  "
              f"L={len(longs)}/{lwr:.0%}  S={len(shorts)}/{swr:.0%}  ({dt:.1f}s)", flush=True)
        results.append(dict(rule=rname, n=total, wr=wr, pf=pf, ev_R=ev,
                           n_long=len(longs), long_wr=lwr, n_short=len(shorts), short_wr=swr))

    rdf = pd.DataFrame(results).sort_values("pf", ascending=False)
    out_csv = f"{OUT_DIR}/phase1_physics_xau.csv"
    rdf.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}", flush=True)

    # Plot
    fig, ax = plt.subplots(figsize=(13, 6), facecolor="#0b0e14")
    ax.set_facecolor("#0d1117")
    colors = ["#00e887" if pf >= 1.0 else "#ff5a5a" for pf in rdf["pf"]]
    bars = ax.bar(rdf["rule"], rdf["pf"], color=colors, edgecolor="#1a2332")
    ax.axhline(1.0, color="#888", linewidth=0.6, linestyle="--")
    for b, nn in zip(bars, rdf["n"]):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.03, f"n={nn}",
                ha="center", color="#aaa", fontsize=8)
    ax.set_ylabel("Profit Factor (TP=6×ATR, SL=2×ATR)", color="#ddd")
    ax.set_title("XAU — Physics Rules Phase 1 (Raw, No ML Filter)",
                 color="#ffd700", fontsize=13)
    ax.tick_params(colors="#aaa", labelsize=9)
    plt.xticks(rotation=35, ha="right")
    for sp in ax.spines.values(): sp.set_edgecolor("#1a2332")
    plt.tight_layout()
    out_png = f"{OUT_DIR}/phase1_physics_xau.png"
    plt.savefig(out_png, dpi=140, facecolor="#0b0e14")
    print(f"Saved: {out_png}", flush=True)

    print(f"\n{'='*60}")
    print("RESULTS RANKED BY PF:")
    for _, r in rdf.iterrows():
        flag = "✓" if r["pf"] >= 1.0 else "✗"
        print(f"  {flag} {r['rule']:<28} PF={r['pf']:.2f}  WR={r['wr']:.1%}  n={r['n']:5d}  EV={r['ev_R']:+.2f}R")


if __name__ == "__main__":
    main()
