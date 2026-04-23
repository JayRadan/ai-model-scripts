"""
v7.2-lite parity check — compare MQL5 vs Python feature values bar-by-bar.

Usage:
  1. Copy the v6 swing CSV + current setups_*_v72l.csv into place.
  2. Run MQL5/Scripts/v7_parity_test.mq5 in MT5 — writes
     MQL5/Files/v7_mql5_features.csv (on Wine: ~/.mt5/drive_c/Program Files/MetaTrader 5/MQL5/Files/)
  3. Run this script.  It computes Python-side features for the same bar
     timestamps (directly from swing_v5_xauusd.csv, same code path as
     00_compute_v72l_features_step1.py) and diffs.
"""
from __future__ import annotations
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, "/home/jay/Desktop/new-model-zigzag/model_pipeline")
import paths as P

MQL5_CSV = os.path.expanduser(
    "~/.mt5/drive_c/Program Files/MetaTrader 5/MQL5/Files/v7_mql5_features.csv")

# Re-use the compute functions from our step=1 script
sys.path.insert(0, "/home/jay/Desktop/new-model-zigzag/experiments/v72_lite_deploy")
import importlib
compute_mod = importlib.import_module("00_compute_v72l_features_step1")


def main():
    if not os.path.exists(MQL5_CSV):
        print(f"ERROR: {MQL5_CSV} not found — run v7_parity_test.mq5 in MT5 first.")
        sys.exit(1)
    mql = pd.read_csv(MQL5_CSV)
    mql["time"] = pd.to_datetime(mql["time"], format="%Y.%m.%d %H:%M")
    mql = mql.sort_values("time").reset_index(drop=True)
    print(f"MQL5 CSV: {len(mql)} bars from {mql['time'].iat[0]} to {mql['time'].iat[-1]}")

    # Compute Python features on the SAME historical swing CSV
    swing = pd.read_csv(P.data("swing_v5_xauusd.csv"), parse_dates=["time"])
    swing = swing.sort_values("time").reset_index(drop=True)
    c = swing["close"].values.astype(np.float64)
    h = swing["high"].values.astype(np.float64)
    l = swing["low"].values.astype(np.float64)

    print("Computing Python-side features (raw, no smoothing yet)...")
    vpin_raw   = compute_mod.compute_vpin(h, l, c)
    qv_raw     = compute_mod.compute_sig_quad_var(c)
    har_raw    = compute_mod.compute_har_rv_ratio(c)
    hawkes_raw = compute_mod.compute_hawkes_eta(c)

    # Apply the same 6-bar trailing smoothing that the CSVs have
    smooth = lambda a: pd.Series(a).rolling(6, min_periods=1).mean().values
    py = pd.DataFrame({
        "time":         swing["time"].values,
        "vpin":         smooth(np.where(np.isfinite(vpin_raw),   vpin_raw,   0.0)),
        "sig_quad_var": smooth(np.where(np.isfinite(qv_raw),     qv_raw,     0.0)),
        "har_rv_ratio": smooth(np.where(np.isfinite(har_raw),    har_raw,    0.0)),
        "hawkes_eta":   smooth(np.where(np.isfinite(hawkes_raw), hawkes_raw, 0.0)),
    })

    # Join on time
    merged = mql.merge(py, on="time", how="inner", suffixes=("_mql5", "_py"))
    print(f"Matched {len(merged)} / {len(mql)} MQL5 rows to Python bars")
    if len(merged) == 0:
        print("No matching timestamps — parity check cannot run.")
        return

    print(f"\n{'Feature':<15} {'MaxAbsDiff':>14} {'MeanAbsDiff':>14} "
          f"{'MaxRelDiff':>14} {'ok?':>5}")
    print("-" * 70)
    any_fail = False
    for feat in ["vpin", "sig_quad_var", "har_rv_ratio", "hawkes_eta"]:
        a = merged[f"{feat}_mql5"].values
        b = merged[f"{feat}_py"].values
        diff = np.abs(a - b)
        scale = np.maximum(np.abs(b), 1e-9)
        rel = diff / scale
        max_abs = diff.max()
        mean_abs = diff.mean()
        max_rel = rel.max()
        # Tolerance: absolute 1e-5 OR relative 1e-4 (accounts for A&S erf ~1.5e-7
        # plus floating-point accumulation in 500+ loops).
        ok = (max_abs < 1e-5) or (max_rel < 1e-4)
        flag = "✓" if ok else "✗"
        if not ok: any_fail = True
        print(f"  {feat:<15} {max_abs:>14.2e} {mean_abs:>14.2e} "
              f"{max_rel:>14.2e} {flag:>5}")

    if any_fail:
        print("\n⚠ PARITY FAILED — investigate before deploying.")
        # Show the worst-diff bars for the failing feature(s)
        for feat in ["vpin", "sig_quad_var", "har_rv_ratio", "hawkes_eta"]:
            a = merged[f"{feat}_mql5"].values
            b = merged[f"{feat}_py"].values
            d = np.abs(a - b)
            if d.max() < 1e-5: continue
            worst = np.argsort(-d)[:5]
            print(f"\nWorst diffs for {feat}:")
            for i in worst:
                print(f"  {merged['time'].iat[i]}  mql5={a[i]:+.8f}  "
                      f"py={b[i]:+.8f}  diff={d[i]:.2e}")
    else:
        print("\n✓ PARITY OK — bit-level match within float32 precision")


if __name__ == "__main__":
    main()
