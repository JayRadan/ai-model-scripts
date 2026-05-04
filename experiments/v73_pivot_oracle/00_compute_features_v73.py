"""
v7.3 Pivot Oracle — feature computation (full v7.2-lite parity + pivot context).

Produces ONE row per bar with:
  - 21 base chart features (f01_CPR ... f20_NCDE) from swing CSV (already there)
  - 14 v7.2-lite OLD_FEATS  (hurst, ou_theta, entropy, kramers, wavelet_er,
                              vwap_dist, hour_enc, dow_enc, quantum_flow,
                              quantum_flow_h4, quantum_momentum,
                              quantum_vwap_conf, quantum_divergence,
                              quantum_div_strength)
  - 4 v7.2-lite EXTRA      (vpin, sig_quad_var, har_rv_ratio, hawkes_eta)
  - 4 pivot-context        (rsi14, leg_age_bars, wick_signed, range_atr_ratio)

Total = 21 + 14 + 4 + 4 = 43 features.

Compute time budget on 1M bars: ~15-30 min (most cost in step=6 physics rollups).

Output: experiments/v73_pivot_oracle/data/features_v73.csv
"""
from __future__ import annotations
import os, time as _time
import numpy as np
import pandas as pd
import sys
from importlib.machinery import SourceFileLoader

sys.path.insert(0, "/home/jay/Desktop/new-model-zigzag/model_pipeline")
import paths as P

OUT_DIR = "/home/jay/Desktop/new-model-zigzag/experiments/v73_pivot_oracle/data"
os.makedirs(OUT_DIR, exist_ok=True)

# Reuse existing implementations
PHYS = SourceFileLoader(
    "phys", "/home/jay/Desktop/new-model-zigzag/model_pipeline/04b_compute_physics_features.py"
).load_module()
V72L = SourceFileLoader(
    "v72l_step1",
    "/home/jay/Desktop/new-model-zigzag/experiments/v72_lite_deploy/00_compute_v72l_features_step1.py"
).load_module()


def compute_atr(H, L, C, n=14):
    tr = np.concatenate([
        [H[0] - L[0]],
        np.maximum.reduce([H[1:] - L[1:], np.abs(H[1:] - C[:-1]), np.abs(L[1:] - C[:-1])]),
    ])
    return pd.Series(tr).rolling(n, min_periods=n).mean().values


def compute_rsi(C, n=14):
    diff = np.diff(C, prepend=C[0])
    up = np.where(diff > 0, diff, 0.0)
    dn = np.where(diff < 0, -diff, 0.0)
    roll_up = pd.Series(up).rolling(n, min_periods=n).mean().values
    roll_dn = pd.Series(dn).rolling(n, min_periods=n).mean().values
    rs = np.divide(roll_up, roll_dn, out=np.zeros_like(roll_up), where=roll_dn > 0)
    return 100.0 - (100.0 / (1.0 + rs))


def compute_leg_age(C, atr, thr_atr=0.5):
    n = len(C); age = np.zeros(n); leg_dir = 0
    last_pivot = C[0]; last_pivot_idx = 0
    for i in range(1, n):
        a = atr[i] if np.isfinite(atr[i]) and atr[i] > 0 else None
        if a is None: age[i] = age[i-1] + 1; continue
        thr = thr_atr * a; new_pivot = False
        if leg_dir >= 0:
            if C[i] > last_pivot: last_pivot = C[i]; last_pivot_idx = i; leg_dir = +1
            elif (last_pivot - C[i]) >= thr: new_pivot = True; last_pivot = C[i]; last_pivot_idx = i; leg_dir = -1
        elif leg_dir <= 0:
            if C[i] < last_pivot: last_pivot = C[i]; last_pivot_idx = i; leg_dir = -1
            elif (C[i] - last_pivot) >= thr: new_pivot = True; last_pivot = C[i]; last_pivot_idx = i; leg_dir = +1
        age[i] = 0 if new_pivot else (i - last_pivot_idx)
    return age


def main():
    t0 = _time.time()
    print("Loading swing CSV...", flush=True)
    swing = pd.read_csv(P.data("swing_v5_xauusd.csv"), parse_dates=["time"])
    swing = swing.sort_values("time").reset_index(drop=True)
    n = len(swing)
    print(f"  {n:,} bars", flush=True)

    O = swing["open"].values.astype(np.float64)
    H = swing["high"].values.astype(np.float64)
    L = swing["low"].values.astype(np.float64)
    C = swing["close"].values.astype(np.float64)
    spread = np.maximum(swing["spread"].values.astype(np.float64), 1.0)
    atr = compute_atr(H, L, C, 14)
    ret = np.concatenate([[0.0], np.diff(np.log(C))])

    # ----- PHYSICS (14 OLD_FEATS) -----
    print("\n[physics] hurst_rs (~slow)...", flush=True); t = _time.time()
    hurst = PHYS.compute_hurst_rs(ret); print(f"  {_time.time()-t:.0f}s", flush=True)

    print("[physics] ou_theta...", flush=True); t = _time.time()
    ou = PHYS.compute_ou_theta(ret); print(f"  {_time.time()-t:.0f}s", flush=True)

    print("[physics] entropy_rate...", flush=True); t = _time.time()
    ent = PHYS.compute_entropy(ret); print(f"  {_time.time()-t:.0f}s", flush=True)

    print("[physics] kramers_up...", flush=True); t = _time.time()
    kram = PHYS.compute_kramers_up(C); print(f"  {_time.time()-t:.0f}s", flush=True)

    print("[physics] wavelet_er...", flush=True); t = _time.time()
    wav = PHYS.compute_wavelet_er(C); print(f"  {_time.time()-t:.0f}s", flush=True)

    print("[physics] vwap_dist...", flush=True); t = _time.time()
    vwap = PHYS.compute_vwap_dist(swing, atr); print(f"  {_time.time()-t:.0f}s", flush=True)

    print("[physics] quantum_flow...", flush=True); t = _time.time()
    qflow = PHYS.compute_quantum_flow(O, H, L, C, spread); print(f"  {_time.time()-t:.0f}s", flush=True)

    print("[physics] quantum_flow_h4 (resample)...", flush=True); t = _time.time()
    df_h4 = swing.set_index("time")[["open","high","low","close","spread"]].resample("4h").agg(
        {"open":"first","high":"max","low":"min","close":"last","spread":"sum"}).dropna()
    qf_h4 = PHYS.compute_quantum_flow(df_h4["open"].values, df_h4["high"].values,
                                       df_h4["low"].values, df_h4["close"].values,
                                       np.maximum(df_h4["spread"].values, 1.0))
    qf_h4_s = pd.Series(qf_h4, index=df_h4.index).shift(1)
    qflow_h4 = qf_h4_s.reindex(swing.set_index("time").index, method="ffill").values
    print(f"  {_time.time()-t:.0f}s", flush=True)

    # quantum_* derivatives — match v7.2-lite formulas
    qmom = pd.Series(qflow).diff(5).fillna(0).values
    qvwap_conf = np.where(np.sign(qflow) == np.sign(vwap), 1.0, -1.0)
    qdiv = qflow - qflow_h4
    qdiv_strength = np.abs(qdiv) / (np.abs(qflow) + np.abs(qflow_h4) + 1e-9)

    # time encodings
    hour = swing["time"].dt.hour.values
    dow = swing["time"].dt.dayofweek.values
    hour_enc = np.cos(2 * np.pi * hour / 24.0)
    dow_enc = np.cos(2 * np.pi * dow / 7.0)

    # ----- V72L EXTRA (4) -----
    print("\n[v72L] vpin...", flush=True); t = _time.time()
    vpin = V72L.compute_vpin(H, L, C); print(f"  {_time.time()-t:.0f}s", flush=True)

    print("[v72L] sig_quad_var...", flush=True); t = _time.time()
    sqv = V72L.compute_sig_quad_var(C); print(f"  {_time.time()-t:.0f}s", flush=True)

    print("[v72L] har_rv_ratio...", flush=True); t = _time.time()
    har = V72L.compute_har_rv_ratio(C); print(f"  {_time.time()-t:.0f}s", flush=True)

    print("[v72L] hawkes_eta...", flush=True); t = _time.time()
    hawk = V72L.compute_hawkes_eta(C); print(f"  {_time.time()-t:.0f}s", flush=True)

    # 6-bar smoothing per v72L convention
    def _sm(a): return pd.Series(a).rolling(6, min_periods=1).mean().values
    vpin = _sm(vpin); sqv = _sm(sqv); har = _sm(har); hawk = _sm(hawk)

    # ----- PIVOT CONTEXT (4) -----
    print("\n[pivot] rsi/leg/wick/range...", flush=True)
    rsi = compute_rsi(C, 14)
    leg_age = compute_leg_age(C, atr, 0.5)
    rng = H - L; body = np.abs(C - O)
    upper = H - np.maximum(O, C); lower = np.minimum(O, C) - L
    wick_signed = np.where(rng > 0, (lower - upper) / rng, 0.0)
    range_atr_ratio = np.where(atr > 0, rng / atr, 0.0)

    # ----- ASSEMBLE -----
    base_feats = [c for c in swing.columns if any(c.startswith(f"f{i:02d}_") for i in range(1, 21))]
    print(f"\nAssembling {len(base_feats)} base + 14 phys + 4 v72L + 4 pivot = {len(base_feats)+22} features", flush=True)

    out = swing[["time"] + base_feats].copy()
    out.insert(1, "bar_idx", np.arange(n))

    out["hurst_rs"] = hurst
    out["ou_theta"] = ou
    out["entropy_rate"] = ent
    out["kramers_up"] = kram
    out["wavelet_er"] = wav
    out["vwap_dist"] = vwap
    out["hour_enc"] = hour_enc
    out["dow_enc"] = dow_enc
    out["quantum_flow"] = qflow
    out["quantum_flow_h4"] = qflow_h4
    out["quantum_momentum"] = qmom
    out["quantum_vwap_conf"] = qvwap_conf
    out["quantum_divergence"] = qdiv
    out["quantum_div_strength"] = qdiv_strength

    out["vpin"] = vpin
    out["sig_quad_var"] = sqv
    out["har_rv_ratio"] = har
    out["hawkes_eta"] = hawk

    out["rsi14"] = rsi
    out["leg_age_bars"] = leg_age
    out["wick_signed"] = wick_signed
    out["range_atr_ratio"] = range_atr_ratio

    # NaN -> 0 (same as v72L convention)
    for c in out.columns:
        if c in ("time", "bar_idx"): continue
        if out[c].isna().any():
            out[c] = out[c].fillna(0)

    out_path = os.path.join(OUT_DIR, "features_v73.csv")
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}  ({os.path.getsize(out_path)/1e6:.1f} MB)")
    print(f"Total time: {_time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
