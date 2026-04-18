"""
Phase 2: Add physics indicators as FEATURES to the existing XAU confirmation
classifier and measure PF improvement on honest holdout.

Adds 5 new features:
  - hurst_rs     (R/S Hurst exponent, 120-bar)
  - ou_theta     (OU mean-reversion speed, 60-bar)
  - entropy      (Shannon entropy of returns, 100-bar)
  - kramers_up   (Kramers escape rate upward, 100-bar)
  - wavelet_er   (CWT trend/noise energy ratio, 120-bar)

Comparison:
  A) Baseline: existing 36 features → per-rule XGBoost → holdout PF
  B) Enhanced: existing 36 + 5 physics features → same pipeline → holdout PF
"""
from __future__ import annotations
import glob, json, os, sys, time as _time
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

ROOT = "/home/jay/Desktop/new-model-zigzag"
TP_MULT, SL_MULT, MAX_FWD = 6.0, 2.0, 40
SPREAD = 0.40

PHYSICS_COLS = ["hurst_rs", "ou_theta", "entropy_rate", "kramers_up", "wavelet_er"]


# ═══════════════ Physics feature computation ═══════════════

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
    n = len(ret); theta = np.full(n, np.nan)
    x = np.cumsum(ret)
    for i in range(window, n, step):
        seg = x[i-window:i]; mu = seg.mean()
        dx = np.diff(seg); xm = seg[:-1] - mu
        denom = np.sum(xm**2)
        if denom > 1e-15: theta[i] = -np.sum(dx * xm) / denom
    _ffill(theta); return theta

def compute_entropy(ret, window=100, n_bins=10, step=6):
    n = len(ret); ent = np.full(n, np.nan)
    for i in range(window, n, step):
        seg = ret[i-window:i]
        counts, _ = np.histogram(seg, bins=n_bins)
        p = counts / counts.sum(); p = p[p > 0]
        ent[i] = -np.sum(p * np.log2(p)) / np.log2(n_bins)  # normalised 0-1
    _ffill(ent); return ent

def compute_kramers_up(c, window=100, step=6):
    n = len(c); esc = np.full(n, np.nan)
    for i in range(window, n, step):
        seg = c[i-window:i]; hi = seg.max()
        sigma = np.std(np.diff(np.log(seg + 1e-10)))
        if sigma < 1e-12: continue
        barrier = (hi - c[i]) / (sigma * c[i] + 1e-10)
        esc[i] = np.exp(-barrier)
    _ffill(esc); return esc

def _ricker(points, a):
    A = 2 / (np.sqrt(3*a) * (np.pi**0.25))
    wsq = a**2; vec = np.arange(0, points) - (points-1.0)/2
    return A * (1 - vec**2/wsq) * np.exp(-vec**2/(2*wsq))

def compute_wavelet_er(c, window=120, step=12):
    n = len(c); er = np.full(n, np.nan)
    kt = _ricker(min(200, window), 20)
    kn = _ricker(min(30, window), 3)
    for i in range(window, n, step):
        seg = c[i-window:i]
        seg_n = (seg - seg.mean()) / (seg.std() + 1e-10)
        ct = np.convolve(seg_n, kt, mode='same')
        cn = np.convolve(seg_n, kn, mode='same')
        er[i] = np.mean(ct**2) / (np.mean(cn**2) + 1e-15)
    _ffill(er); return er


def add_physics_to_swing(swing_path):
    """Compute physics features and return as DataFrame with 'time' column."""
    print("  Loading swing data...", flush=True)
    df = pd.read_csv(swing_path, usecols=["time", "close"], parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)
    c = df["close"].values.astype(np.float64)
    ret = np.concatenate([[0.0], np.diff(np.log(c))])

    print("  Computing Hurst R/S...", end="", flush=True); t0=_time.time()
    hurst = compute_hurst_rs(ret); print(f" {_time.time()-t0:.1f}s", flush=True)
    print("  Computing OU θ...", end="", flush=True); t0=_time.time()
    ou = compute_ou_theta(ret); print(f" {_time.time()-t0:.1f}s", flush=True)
    print("  Computing entropy...", end="", flush=True); t0=_time.time()
    ent = compute_entropy(ret); print(f" {_time.time()-t0:.1f}s", flush=True)
    print("  Computing Kramers...", end="", flush=True); t0=_time.time()
    kr = compute_kramers_up(c); print(f" {_time.time()-t0:.1f}s", flush=True)
    print("  Computing wavelet...", end="", flush=True); t0=_time.time()
    wv = compute_wavelet_er(c); print(f" {_time.time()-t0:.1f}s", flush=True)

    physics = pd.DataFrame({
        "time": df["time"],
        "hurst_rs": hurst,
        "ou_theta": ou,
        "entropy_rate": ent,
        "kramers_up": kr,
        "wavelet_er": wv,
    })
    return physics


def forward_outcome(H, L, C, atr, i, direction):
    a = atr[i]
    if not np.isfinite(a) or a <= 0: return -1
    e = C[i]; n = len(C); end = min(i+1+MAX_FWD, n)
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
    return 2


def run_comparison():
    print("="*70)
    print("PHASE 2: Physics features as ML inputs — XAU holdout comparison")
    print("="*70, flush=True)

    # Load swing for OHLC + ATR (for forward outcome relabeling at 6:2)
    swing = pd.read_csv(f"{ROOT}/data/swing_v5_xauusd.csv",
                        parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    H = swing["high"].values; L = swing["low"].values; C = swing["close"].values
    tr = np.concatenate([[H[0]-L[0]],
          np.maximum.reduce([H[1:]-L[1:], np.abs(H[1:]-C[:-1]), np.abs(L[1:]-C[:-1])])])
    atr = pd.Series(tr).rolling(14, min_periods=14).mean().values
    time_to_idx = pd.Series(swing.index.values, index=swing["time"].values)

    # Physics features
    print("\nComputing physics features on full swing data:", flush=True)
    physics = add_physics_to_swing(f"{ROOT}/data/swing_v5_xauusd.csv")

    # Load existing models' meta for baseline feature cols
    baseline_results = []
    enhanced_results = []

    for f in sorted(glob.glob(f"{ROOT}/data/setups_*.csv")):
        cid = int(os.path.basename(f).replace("setups_","").replace(".csv",""))
        df = pd.read_csv(f, parse_dates=["time"])

        # Merge physics features
        df = df.merge(physics, on="time", how="left")
        for col in PHYSICS_COLS:
            df[col] = df[col].fillna(0)

        for rule_name, grp in df.groupby("rule"):
            meta_path = f"{ROOT}/models/confirm_c{cid}_{rule_name}_meta.json"
            model_path = f"{ROOT}/models/confirm_c{cid}_{rule_name}.json"
            if not (os.path.exists(meta_path) and os.path.exists(model_path)): continue
            meta = json.load(open(meta_path))
            if meta.get("disabled", False): continue
            base_feats = meta.get("feature_cols")
            if not base_feats: continue

            rdf = grp.sort_values("time").reset_index(drop=True)
            split = int(len(rdf) * 0.8)
            train, test = rdf.iloc[:split], rdf.iloc[split:]
            if len(test) < 20 or len(train) < 50: continue

            # Relabel at 6:2 geometry using forward OHLC
            def relabel(subset):
                labels = []
                for _, row in subset.iterrows():
                    t = row["time"]
                    if t not in time_to_idx.index: labels.append(-1); continue
                    idx = int(time_to_idx[t])
                    d = int(row["direction"])
                    labels.append(forward_outcome(H, L, C, atr, idx, d))
                return np.array(labels)

            train_lbl = relabel(train)
            test_lbl  = relabel(test)
            # Keep only 0/1 outcomes
            tr_mask = np.isin(train_lbl, [0,1]); te_mask = np.isin(test_lbl, [0,1])
            if tr_mask.sum() < 30 or te_mask.sum() < 10: continue
            y_tr = train_lbl[tr_mask]; y_te = test_lbl[te_mask]

            # A) BASELINE: original features
            missing = [c for c in base_feats if c not in rdf.columns]
            if missing: continue
            X_tr_base = train[base_feats].fillna(0).values[tr_mask]
            X_te_base = test[base_feats].fillna(0).values[te_mask]

            mdl_base = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                     subsample=0.8, colsample_bytree=0.8,
                                     eval_metric="logloss", verbosity=0)
            mdl_base.fit(X_tr_base, y_tr)
            p_base = mdl_base.predict_proba(X_te_base)[:,1]
            thr_base = meta["threshold"]
            mask_base = p_base >= thr_base
            if mask_base.sum() == 0: continue
            y_sel_base = y_te[mask_base]
            wr_base = y_sel_base.mean()
            pf_base = (wr_base * TP_MULT) / ((1-wr_base)*SL_MULT + 1e-9)

            # B) ENHANCED: base + physics
            enh_feats = base_feats + PHYSICS_COLS
            X_tr_enh = train[enh_feats].fillna(0).values[tr_mask]
            X_te_enh = test[enh_feats].fillna(0).values[te_mask]

            mdl_enh = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                    subsample=0.8, colsample_bytree=0.8,
                                    eval_metric="logloss", verbosity=0)
            mdl_enh.fit(X_tr_enh, y_tr)
            p_enh = mdl_enh.predict_proba(X_te_enh)[:,1]
            mask_enh = p_enh >= thr_base
            if mask_enh.sum() == 0: continue
            y_sel_enh = y_te[mask_enh]
            wr_enh = y_sel_enh.mean()
            pf_enh = (wr_enh * TP_MULT) / ((1-wr_enh)*SL_MULT + 1e-9)

            n_base = int(mask_base.sum())
            n_enh = int(mask_enh.sum())

            baseline_results.append(dict(cid=cid, rule=rule_name, n=n_base, wr=wr_base, pf=pf_base))
            enhanced_results.append(dict(cid=cid, rule=rule_name, n=n_enh, wr=wr_enh, pf=pf_enh))

    # Summary
    bdf = pd.DataFrame(baseline_results)
    edf = pd.DataFrame(enhanced_results)

    print(f"\n{'='*70}")
    print(f"{'rule':<28} {'base_PF':<8} {'enh_PF':<8} {'Δ PF':<8} {'base_n':<7} {'enh_n':<7}")
    print("-"*70)
    merged = bdf.merge(edf, on=["cid","rule"], suffixes=("_b","_e"))
    for _, r in merged.sort_values("pf_e", ascending=False).iterrows():
        d = r["pf_e"] - r["pf_b"]
        flag = "↑" if d > 0.05 else ("↓" if d < -0.05 else "=")
        print(f"  C{r['cid']} {r['rule']:<24} {r['pf_b']:.2f}    {r['pf_e']:.2f}    {d:+.2f} {flag}  "
              f"{r['n_b']:<7} {r['n_e']:<7}")

    # Weighted averages
    if len(merged) > 0:
        w = merged["n_b"]
        wpf_b = (merged["pf_b"] * w).sum() / w.sum()
        w2 = merged["n_e"]
        wpf_e = (merged["pf_e"] * w2).sum() / w2.sum()
        print(f"\n  WEIGHTED AVG PF:  baseline={wpf_b:.3f}   enhanced={wpf_e:.3f}   Δ={wpf_e-wpf_b:+.3f}")
        print(f"  Rules compared: {len(merged)}")

    # Save
    merged.to_csv(f"{ROOT}/experiments/innovative_rules/phase2_physics_comparison.csv", index=False)
    print(f"\nSaved: phase2_physics_comparison.csv", flush=True)


if __name__ == "__main__":
    run_comparison()
