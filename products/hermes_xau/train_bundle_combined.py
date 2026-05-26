"""
Hermes XAU — train + freeze the production Q-regressor bundle.

Trains XGBRegressor on the v103 spec:
  - Source bars: data/m1_xau_full.parquet (8.5 yr Dukascopy M1) JOINED with
                 data/m1_xau_orderflow.parquet for the tick-derived features
  - Train cutoff: 2025-09-01 (research holdout split)
  - Label: forward pnl_R with SL=4×ATR, trail=3×ATR, max_hold=300, spread-adjusted
  - Features: 29 standard + 14 order-flow = 43 total
  - Output: commercial/server/decision_engine/models/hermes_xau_combined.pkl

Holdout validation (research, Sep 2025 → May 2026):
  PF 2.09 | sumR +6,372 | DD 117 | WR 59.1% | 4,828 trades | 20 trd/day
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent          # /home/jay/Desktop/new-model-zigzag
sys.path.insert(0, str(ROOT / "experiments" / "v103_tfk_regime"))

# Reuse the proven research feature engineering pipeline.
from tfk import compute_tfk
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "ofm1", ROOT / "experiments/v103_tfk_regime/43_m1_with_orderflow.py")
_ofm1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ofm1)
add_standard_features = _ofm1.add_standard_features
STD_FEATS = list(_ofm1.STD_FEATS)
FLOW_FEATS = list(_ofm1.FLOW_FEATS)


# ── Config (must match commercial/server/.../configs/hermes_xau.py) ─────
CUTOFF = pd.Timestamp("2025-09-01 00:00:00")
SL = 4.0
TRAIL = 3.0
MAX_HOLD = 300
NEAR = 0.50         # pullback gate
COUNTER_THR = 1.5   # counter-pullback gate (dist_signed × cdir ≤ -1.5)
XAU_SPREAD_USD = 0.30

OUT_DIR = Path("/home/jay/Desktop/my-agents-and-website/commercial/server/"
               "decision_engine/models")
OUT_PKL = OUT_DIR / "hermes_xau_combined.pkl"


def simulate_label(entry_idx, direction, C, H, L, O, atr_at_entry, sl_atr=SL,
                   spread_R=0.0, trail_atr=TRAIL, max_hold=MAX_HOLD):
    n = len(C)
    if entry_idx >= n - 1 or not (np.isfinite(atr_at_entry) and atr_at_entry > 0):
        return np.nan
    ep = O[entry_idx]; a = atr_at_entry
    hard = sl_atr * a; trail = trail_atr * a; max_favor = 0.0
    end = min(entry_idx + max_hold, n - 1)
    for k in range(entry_idx, end + 1):
        favor_now = direction * (C[k] - ep)
        if favor_now > max_favor: max_favor = favor_now
        if direction == 1:
            if (ep - L[k]) >= hard: return -sl_atr - spread_R
        else:
            if (H[k] - ep) >= hard: return -sl_atr - spread_R
        if max_favor >= trail:
            if (max_favor - favor_now) >= trail:
                return (max_favor - trail) / a - spread_R
    return direction * (C[end] - ep) / a - spread_R


def bars_in_regime_array(cdir):
    n = len(cdir); out = np.zeros(n, dtype=np.int64); cur = 1
    for i in range(1, n):
        if cdir[i] == cdir[i - 1]: cur += 1
        else: cur = 1
        out[i] = cur
    return out


def metrics(r):
    r = r[np.isfinite(r)]
    if len(r) == 0: return None
    w, l = r[r > 0], r[r <= 0]
    pf = float(w.sum() / max(-l.sum(), 1e-9))
    eq = np.cumsum(r)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    return {"n": int(len(r)), "wr": float((r > 0).mean()), "pf": pf,
            "sum_r": float(r.sum()), "max_dd_r": dd, "avg_r": float(r.mean())}


def main():
    print("="*72)
    print("  Hermes XAU — bundle training pipeline (M1 + order-flow)")
    print("="*72)
    t0 = time.time()

    # 1. Load bars + orderflow
    print("\n[1/5] loading M1 + orderflow ...", flush=True)
    m1 = pd.read_parquet(ROOT / "data" / "m1_xau_orderflow.parquet")
    m1 = m1.sort_values("time").reset_index(drop=True)
    print(f"  bars: {len(m1):,}  range: {m1.time.iloc[0]} → {m1.time.iloc[-1]}", flush=True)

    # 2. Compute TFK + standard features
    print("\n[2/5] TFK + standard features ...", flush=True)
    tfk_out = compute_tfk(m1)
    df = add_standard_features(tfk_out)
    O = df["open"].to_numpy(np.float64); H = df["high"].to_numpy(np.float64)
    L = df["low"].to_numpy(np.float64); C = df["close"].to_numpy(np.float64)
    line = df["tfk_line"].to_numpy(np.float64)
    cdir = df["committed_dir"].to_numpy(np.int64)
    times = df["time"].to_numpy()
    atr = df["atr14"].to_numpy(np.float64)
    bir = bars_in_regime_array(cdir)
    n = len(df)
    spread_R = XAU_SPREAD_USD / np.nanmedian(atr)
    print(f"  bars={n:,}  median_atr={np.nanmedian(atr):.3f}  spread={spread_R:.3f}R", flush=True)

    # 3. Build candidate set + labels
    print("\n[3/5] building candidates + labels ...", flush=True)
    dist_signed = np.where(atr > 0, (C - line) / atr, 0.0)
    dist_abs = np.abs(dist_signed)
    valid = (np.isfinite(atr) & (atr > 0))
    valid[:200] = False; valid[-(MAX_HOLD + 1):] = False
    # Combined candidate set: pullback OR counter
    pullback_mask = dist_abs <= NEAR
    counter_score = dist_signed * cdir
    counter_mask  = counter_score <= -COUNTER_THR
    valid &= (cdir != 0) & (pullback_mask | counter_mask)
    idxs = np.where(valid)[0]
    dirs = cdir[idxs].copy()
    print(f"  candidates: {len(idxs):,}  "
          f"(pullback {int(pullback_mask[idxs].sum()):,}, "
          f"counter {int(counter_mask[idxs].sum()):,})", flush=True)

    pnl_s = np.zeros(len(idxs), dtype=np.float32)
    for k, i in enumerate(idxs):
        d = int(dirs[k])
        pnl_s[k] = simulate_label(i + 1, d, C, H, L, O, atr[i], spread_R=spread_R)
        if k % 50000 == 0 and k > 0: print(f"    {k:,}/{len(idxs):,}", flush=True)
    pnl_s = np.where(np.isfinite(pnl_s), pnl_s, 0.0)

    extra = pd.DataFrame({
        "dist_at_signal": dist_signed[idxs], "dist_abs": dist_abs[idxs],
        "regime_age": bir[idxs],
        "bar_range_atr": (H[idxs] - L[idxs]) / np.maximum(atr[idxs], 1e-9),
    })
    feats_std = df.iloc[idxs][[f for f in STD_FEATS if f not in extra.columns and f in df.columns]].reset_index(drop=True)
    feats_flow = df.iloc[idxs][[f for f in FLOW_FEATS if f in df.columns]].reset_index(drop=True)
    X_all = pd.concat([extra.reset_index(drop=True), feats_std, feats_flow], axis=1)
    feat_cols = list(X_all.columns)
    print(f"  features: {len(feat_cols)}", flush=True)
    print(f"  feat_cols: {feat_cols}")

    times_at_idx = times[idxs]
    train_m = times_at_idx < np.datetime64(CUTOFF)
    test_m = ~train_m
    print(f"  train={train_m.sum():,}  test={test_m.sum():,}  cutoff={CUTOFF}", flush=True)

    # 4. Fit + evaluate
    print("\n[4/5] fitting XGBRegressor ...", flush=True)
    from xgboost import XGBRegressor
    mdl = XGBRegressor(n_estimators=600, max_depth=5, learning_rate=0.04,
                       subsample=0.85, colsample_bytree=0.85,
                       min_child_weight=10, reg_lambda=1.0,
                       objective="reg:squarederror", tree_method="hist",
                       random_state=42, verbosity=0)
    mdl.fit(X_all.loc[train_m].fillna(0).to_numpy(np.float32), pnl_s[train_m])
    q_te = mdl.predict(X_all.loc[test_m].fillna(0).to_numpy(np.float32))
    # Holdout sanity across q thresholds — production target is Q≥4.0
    print(f"  holdout sweep:")
    holdout_by_q = {}
    for qt in [1.0, 2.0, 3.0, 4.0, 5.0]:
        m = metrics(pnl_s[test_m][q_te >= qt])
        holdout_by_q[qt] = m
        print(f"    Q≥{qt}: {m}", flush=True)
    bare = holdout_by_q[4.0]

    # 5. Freeze bundle
    print("\n[5/5] freezing bundle ...", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "q_model": mdl,
        "feat_cols": feat_cols,
        "train_meta": {
            "trained_on": datetime.now(timezone.utc).isoformat(),
            "n_train": int(train_m.sum()),
            "n_test": int(test_m.sum()),
            "cutoff": str(CUTOFF),
            "spread_R_train": float(spread_R),
            "holdout_q4": bare,
            "holdout_sweep": holdout_by_q,
            "config_snapshot": {
                "NEAR": NEAR, "COUNTER_THR": COUNTER_THR,
                "SL": SL, "TRAIL": TRAIL, "MAX_HOLD": MAX_HOLD,
                "SPREAD_USD": XAU_SPREAD_USD, "q_thr_recommended": 4.0,
            },
        },
    }
    with open(OUT_PKL, "wb") as f:
        pickle.dump(payload, f)
    print(f"  wrote {OUT_PKL}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"\n  done.")


if __name__ == "__main__":
    main()
