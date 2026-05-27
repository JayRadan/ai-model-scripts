"""Hermes DJI — combined-Q with 43 features, FAIR comparison.

Base = native Dukascopy M1 (same as M1-only run).
Flow features are LEFT-JOINED from the tick-aggregated parquet by minute.
This isolates whether orderflow features carry information on top of the
identical base candidate set + labels.
"""
from __future__ import annotations
import pickle, sys, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
from tfk import compute_tfk

sys.path.insert(0, str(ROOT / "experiments" / "v103_tfk_regime"))
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "ofm1", ROOT / "experiments/v103_tfk_regime/43_m1_with_orderflow.py")
_ofm1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ofm1)
add_standard_features = _ofm1.add_standard_features
STD_FEATS = list(_ofm1.STD_FEATS)
FLOW_FEATS = list(_ofm1.FLOW_FEATS)

CUTOFF = pd.Timestamp("2025-09-01 00:00:00")
SL, TRAIL, MAX_HOLD = 4.0, 3.0, 300
NEAR, COUNTER_THR = 0.50, 1.5
DJI_SPREAD_USD = 2.0
OUT_PKL = HERE / "hermes_dji_validated_native_plus_flow.pkl"


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
        if max_favor >= trail and (max_favor - favor_now) >= trail:
            return (max_favor - trail) / a - spread_R
    return direction * (C[end] - ep) / a - spread_R


def bars_in_regime_array(cdir):
    n = len(cdir); out = np.zeros(n, dtype=np.int64); cur = 1
    for i in range(1, n):
        cur = cur + 1 if cdir[i] == cdir[i - 1] else 1
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
    print("="*72); print("  Hermes DJI — combined-Q (native M1 base + joined flow)"); print("="*72)
    t0 = time.time()

    print("\n[1/5] loading native M1 + flow ...", flush=True)
    m1 = pd.read_parquet(ROOT / "data" / "m1_dji_full.parquet")
    m1 = m1.sort_values("time").reset_index(drop=True)
    print(f"  native M1: {len(m1):,}", flush=True)
    flow = pd.read_parquet(ROOT / "data" / "m1_dji_orderflow.parquet")
    flow["time"] = pd.to_datetime(flow["time"]).dt.tz_localize(None)
    keep_flow = ["time"] + [c for c in FLOW_FEATS if c in flow.columns]
    flow = flow[keep_flow]
    print(f"  flow: {len(flow):,} rows, cols={keep_flow[1:]}")

    print("\n[2/5] TFK + standard features on native M1 ...", flush=True)
    tfk_out = compute_tfk(m1)
    df = add_standard_features(tfk_out)
    # Left-join flow by floor-to-minute
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
    df["_t_floor"] = df["time"].dt.floor("1min")
    flow["_t_floor"] = flow["time"].dt.floor("1min")
    flow_only = flow.drop(columns=["time"]).drop_duplicates("_t_floor", keep="last")
    df = df.merge(flow_only, on="_t_floor", how="left")
    df = df.drop(columns=["_t_floor"])
    flow_coverage = df[FLOW_FEATS[0]].notna().mean()
    print(f"  flow join coverage: {flow_coverage*100:.1f}%  ({df[FLOW_FEATS[0]].notna().sum():,}/{len(df):,})")

    O = df["open"].to_numpy(np.float64); H = df["high"].to_numpy(np.float64)
    L = df["low"].to_numpy(np.float64); C = df["close"].to_numpy(np.float64)
    line = df["tfk_line"].to_numpy(np.float64)
    cdir = df["committed_dir"].to_numpy(np.int64)
    times = df["time"].to_numpy()
    atr = df["atr14"].to_numpy(np.float64)
    bir = bars_in_regime_array(cdir)
    n = len(df)
    median_atr = float(np.nanmedian(atr))
    spread_R = DJI_SPREAD_USD / median_atr
    print(f"  bars={n:,}  median_atr={median_atr:.3f}  spread={spread_R:.3f}R", flush=True)

    print("\n[3/5] building candidates + labels (SAME as M1-only run) ...", flush=True)
    dist_signed = np.where(atr > 0, (C - line) / atr, 0.0)
    dist_abs = np.abs(dist_signed)
    valid = (np.isfinite(atr) & (atr > 0))
    valid[:200] = False; valid[-(MAX_HOLD + 1):] = False
    pullback_mask = dist_abs <= NEAR
    counter_score = dist_signed * cdir
    counter_mask = counter_score <= -COUNTER_THR
    valid &= (cdir != 0) & (pullback_mask | counter_mask)
    idxs = np.where(valid)[0]
    dirs = cdir[idxs].copy()
    print(f"  candidates: {len(idxs):,}  (pullback {int(pullback_mask[idxs].sum()):,}, counter {int(counter_mask[idxs].sum()):,})")

    pnl_s = np.zeros(len(idxs), dtype=np.float32)
    for k, i in enumerate(idxs):
        d = int(dirs[k])
        pnl_s[k] = simulate_label(i + 1, d, C, H, L, O, atr[i], spread_R=spread_R)
        if k % 100000 == 0 and k > 0:
            print(f"    {k:,}/{len(idxs):,}", flush=True)
    pnl_s = np.where(np.isfinite(pnl_s), pnl_s, 0.0)

    extra = pd.DataFrame({
        "dist_at_signal": dist_signed[idxs], "dist_abs": dist_abs[idxs],
        "regime_age": bir[idxs],
        "bar_range_atr": (H[idxs] - L[idxs]) / np.maximum(atr[idxs], 1e-9),
    })
    feats_std = df.iloc[idxs][[f for f in STD_FEATS if f not in extra.columns and f in df.columns]].reset_index(drop=True)
    feats_flow = df.iloc[idxs][[f for f in FLOW_FEATS if f in df.columns]].reset_index(drop=True)
    X_std_only = pd.concat([extra.reset_index(drop=True), feats_std], axis=1)
    X_with_flow = pd.concat([extra.reset_index(drop=True), feats_std, feats_flow], axis=1)
    print(f"  features std={len(X_std_only.columns)}, std+flow={len(X_with_flow.columns)}", flush=True)

    times_at_idx = times[idxs]
    train_m = times_at_idx < np.datetime64(CUTOFF)
    test_m = ~train_m
    print(f"  train={train_m.sum():,}  test={test_m.sum():,}", flush=True)

    print("\n[4/5] fitting both models on IDENTICAL candidates ...", flush=True)
    from xgboost import XGBRegressor
    def fit(X):
        m = XGBRegressor(n_estimators=600, max_depth=5, learning_rate=0.04,
                         subsample=0.85, colsample_bytree=0.85,
                         min_child_weight=10, reg_lambda=1.0,
                         objective="reg:squarederror", tree_method="hist",
                         random_state=42, verbosity=0)
        m.fit(X.loc[train_m].fillna(0).to_numpy(np.float32), pnl_s[train_m])
        return m, m.predict(X.loc[test_m].fillna(0).to_numpy(np.float32))

    print("  fitting M1-only ...", flush=True)
    mdl_a, q_a = fit(X_std_only)
    print("  fitting M1+flow ...", flush=True)
    mdl_b, q_b = fit(X_with_flow)

    test_pnl = pnl_s[test_m]
    print(f"\n  HOLDOUT SWEEP — same candidates, two models:")
    print(f"  {'Q':>4}  {'A: M1-only':>30}    {'B: M1+flow':>30}")
    for qt in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        ma = metrics(test_pnl[q_a >= qt])
        mb = metrics(test_pnl[q_b >= qt])
        def fmt(m): return f"n={m['n']:>5} WR={m['wr']*100:4.1f}% PF={m['pf']:5.2f} sumR={m['sum_r']:+7.0f}" if m else "  -"
        print(f"  {qt:>4}  {fmt(ma)}    {fmt(mb)}")

    print("\n[5/5] freezing both bundles ...", flush=True)
    for label, mdl, X in [("std", mdl_a, X_std_only), ("std_flow", mdl_b, X_with_flow)]:
        out = HERE / f"hermes_dji_native_{label}.pkl"
        payload = {"q_model": mdl, "feat_cols": list(X.columns),
                   "train_meta": {"trained_on": datetime.now(timezone.utc).isoformat(),
                                  "n_train": int(train_m.sum()), "n_test": int(test_m.sum())}}
        with open(out, "wb") as f:
            pickle.dump(payload, f)
        print(f"  wrote {out.name}")
    print(f"\n  done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
