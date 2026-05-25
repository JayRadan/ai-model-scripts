"""
SL sweep on M1 with order-flow features.
Test SL_ATR ∈ {2, 3, 4, 5, 6, 7, 8} keeping TRAIL=3, Q≥1.0, NEAR=0.25.
Re-label, re-train Q, re-execute for each SL.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

from tfk import compute_tfk

# Reuse setup from script 43
import importlib.util
_spec = importlib.util.spec_from_file_location("ofm1", HERE / "43_m1_with_orderflow.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
add_standard_features = _mod.add_standard_features
STD_FEATS = _mod.STD_FEATS
FLOW_FEATS = _mod.FLOW_FEATS
bars_in_regime_array = _mod.bars_in_regime_array
metrics = _mod.metrics

CUTOFF = pd.Timestamp("2025-09-01 00:00:00")
TRAIL = 3.0
MAX_HOLD = 300
NEAR = 0.25
Q_THR = 1.0
PROFIT_TO_BE_R = 1.0
MAX_CONCURRENT = 4
SWITCH_DELTA = 0.5
COOLDOWN_BARS = 5
XAU_SPREAD_USD = 0.30

SL_GRID = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]


def simulate_label(entry_idx, direction, C, H, L, O, atr_at_entry, sl_atr,
                   spread_R=0.0, trail_atr=TRAIL, max_hold=MAX_HOLD):
    n = len(C)
    if entry_idx >= n - 1 or not (np.isfinite(atr_at_entry) and atr_at_entry > 0):
        return np.nan, "skip", entry_idx
    ep = O[entry_idx]; a = atr_at_entry
    hard = sl_atr * a; trail = trail_atr * a; max_favor = 0.0
    end = min(entry_idx + max_hold, n - 1)
    for k in range(entry_idx, end + 1):
        favor_now = direction * (C[k] - ep)
        if favor_now > max_favor: max_favor = favor_now
        if direction == 1:
            if (ep - L[k]) >= hard: return -sl_atr - spread_R, "hard_sl", k
        else:
            if (H[k] - ep) >= hard: return -sl_atr - spread_R, "hard_sl", k
        if max_favor >= trail:
            if (max_favor - favor_now) >= trail:
                return (max_favor - trail) / a - spread_R, "trail", k
    return direction * (C[end] - ep) / a - spread_R, "max_hold", end


def main():
    t0 = time.time()
    print(f"\n{'='*72}\n  XAU M1 SL sweep (TRAIL={TRAIL}R, Q≥{Q_THR}, NEAR={NEAR}, with orderflow)\n{'='*72}", flush=True)

    print("\n[1/3] loading M1 + computing features (once) ...", flush=True)
    m1 = pd.read_parquet(ROOT / "data" / "m1_xau_orderflow.parquet")
    m1 = m1.sort_values("time").reset_index(drop=True)
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
    dist_signed = np.where(atr > 0, (C - line) / atr, 0.0)
    dist_abs = np.abs(dist_signed)

    valid = (np.isfinite(atr) & (atr > 0))
    valid[:200] = False; valid[-(MAX_HOLD + 1):] = False
    valid &= (dist_abs <= NEAR) & (cdir != 0)
    idxs = np.where(valid)[0]
    dirs = cdir[idxs].copy()
    print(f"  candidates: {len(idxs):,}  spread={spread_R:.3f}R", flush=True)

    extra = pd.DataFrame({
        "dist_at_signal": dist_signed[idxs], "dist_abs": dist_abs[idxs],
        "regime_age": bir[idxs],
        "bar_range_atr": (H[idxs] - L[idxs]) / np.maximum(atr[idxs], 1e-9),
    })
    feats_std = df.iloc[idxs][[f for f in STD_FEATS if f not in extra.columns and f in df.columns]].reset_index(drop=True)
    feats_flow = df.iloc[idxs][[f for f in FLOW_FEATS if f in df.columns]].reset_index(drop=True)
    X_flow = pd.concat([extra.reset_index(drop=True), feats_std, feats_flow], axis=1)
    flow_cols = list(X_flow.columns)
    times_at_idx = times[idxs]
    train_m = times_at_idx < np.datetime64(CUTOFF)
    test_m = ~train_m
    print(f"  train={train_m.sum():,}  test={test_m.sum():,}", flush=True)

    print("\n[2/3] SL sweep — re-label, re-train, re-execute per SL ...\n", flush=True)
    from xgboost import XGBRegressor

    def run_for_sl(SL):
        # label
        pnl_s = np.zeros(len(idxs), dtype=np.float32)
        exit_idxs = np.zeros(len(idxs), dtype=np.int32)
        for k, i in enumerate(idxs):
            d = int(dirs[k])
            r, _, xi = simulate_label(i + 1, d, C, H, L, O, atr[i], sl_atr=SL, spread_R=spread_R)
            pnl_s[k] = r if np.isfinite(r) else 0.0
            exit_idxs[k] = xi
        # train Q
        mdl = XGBRegressor(n_estimators=600, max_depth=5, learning_rate=0.04,
                           subsample=0.85, colsample_bytree=0.85,
                           min_child_weight=10, reg_lambda=1.0,
                           objective="reg:squarederror", tree_method="hist",
                           random_state=42, verbosity=0)
        mdl.fit(X_flow.loc[train_m].fillna(0).to_numpy(np.float32), pnl_s[train_m])
        q_te = mdl.predict(X_flow.loc[test_m].fillna(0).to_numpy(np.float32))

        # execute
        test_idxs = idxs[test_m]
        test_dirs = dirs[test_m]
        test_pnl = pnl_s[test_m]
        test_exits = exit_idxs[test_m]
        test_times = times_at_idx[test_m]
        span_days = max((test_times[-1] - test_times[0]).astype("timedelta64[D]").astype(int), 1)
        q_by_idx = {int(test_idxs[k]): float(q_te[k]) for k in range(len(test_idxs))}
        info = {int(test_idxs[k]): (int(test_dirs[k]), float(test_pnl[k]), int(test_exits[k]))
                for k in range(len(test_idxs))}

        active = []; executed = []
        last_open_per_dir = {-1: -10**9, +1: -10**9}
        bar_start = int(test_idxs[0]); bar_end = min(int(test_idxs[-1]) + MAX_HOLD + 1, n)
        for i in range(bar_start, bar_end):
            still = []
            for t in active:
                if i < t["entry_idx"]: still.append(t); continue
                if i > min(t["entry_idx"] + MAX_HOLD, n - 1):
                    cp = C[min(t["entry_idx"] + MAX_HOLD, n - 1)]
                    t["pnl_R"] = float(t["direction"] * (cp - t["ep"]) / t["a"] - spread_R)
                    t["exit_idx"] = min(t["entry_idx"] + MAX_HOLD, n - 1)
                    t["exit_reason"] = "max_hold"; executed.append(t); continue
                d = t["direction"]; ep = t["ep"]; a = t["a"]
                favor_now = d * (C[i] - ep)
                if favor_now > t["max_favor"]: t["max_favor"] = favor_now
                sl_r = t["sl_r"]; hit = False
                if sl_r == 0:
                    if d == 1 and L[i] <= ep:
                        t["pnl_R"] = -spread_R; t["exit_idx"] = i; t["exit_reason"] = "be_stop"; hit = True
                    elif d == -1 and H[i] >= ep:
                        t["pnl_R"] = -spread_R; t["exit_idx"] = i; t["exit_reason"] = "be_stop"; hit = True
                else:
                    dist_r = abs(sl_r) * a
                    if d == 1 and (ep - L[i]) >= dist_r:
                        t["pnl_R"] = float(sl_r - spread_R); t["exit_idx"] = i; t["exit_reason"] = "hard_sl"; hit = True
                    elif d == -1 and (H[i] - ep) >= dist_r:
                        t["pnl_R"] = float(sl_r - spread_R); t["exit_idx"] = i; t["exit_reason"] = "hard_sl"; hit = True
                if hit: executed.append(t); continue
                trail_d = TRAIL * a
                if t["max_favor"] >= trail_d and (t["max_favor"] - favor_now) >= trail_d:
                    t["pnl_R"] = float((t["max_favor"] - trail_d) / a - spread_R)
                    t["exit_idx"] = i; t["exit_reason"] = "trail"
                    executed.append(t); continue
                still.append(t)
            active = still
            if i not in info: continue
            if q_by_idx[i] < Q_THR: continue
            d_, _, _ = info[i]
            if i - last_open_per_dir[d_] < COOLDOWN_BARS: continue
            for t in active:
                if t["sl_r"] == 0: continue
                a_ = t["a"]; ep_ = t["ep"]; d2 = t["direction"]
                cur_R = d2 * (C[i] - ep_) / a_
                if cur_R >= PROFIT_TO_BE_R: t["sl_r"] = 0
            entry_idx = i + 1
            if entry_idx >= n: continue
            a_new = atr[i]
            if not (np.isfinite(a_new) and a_new > 0): continue
            ep_new = O[entry_idx]
            if len(active) >= MAX_CONCURRENT:
                worst = min(active, key=lambda x: x["q"])
                if q_by_idx[i] >= worst["q"] + SWITCH_DELTA:
                    cp = C[i]; ep_ = worst["ep"]; d2 = worst["direction"]; a_ = worst["a"]
                    worst["pnl_R"] = float(d2 * (cp - ep_) / a_ - spread_R)
                    worst["exit_idx"] = i; worst["exit_reason"] = "switch_close"
                    executed.append(worst); active.remove(worst)
                else: continue
            new_trade = {"signal_idx": int(i), "entry_idx": int(entry_idx),
                         "direction": int(d_), "ep": float(ep_new), "a": float(a_new),
                         "sl_r": float(-SL), "max_favor": 0.0, "q": float(q_by_idx[i]),
                         "time": pd.Timestamp(times[i]),
                         "exit_idx": int(entry_idx), "exit_reason": None, "pnl_R": None}
            active.append(new_trade)
            last_open_per_dir[d_] = i
        for t in active:
            end_bar = min(t["entry_idx"] + MAX_HOLD, n - 1)
            cp = C[end_bar]
            t["pnl_R"] = float(t["direction"] * (cp - t["ep"]) / t["a"] - spread_R)
            t["exit_idx"] = end_bar; t["exit_reason"] = "end_of_data"
            executed.append(t)
        ex_df = pd.DataFrame(executed)
        m = metrics(ex_df["pnl_R"].to_numpy())
        em = ex_df["exit_reason"].value_counts().to_dict()
        return m, em, span_days

    print(f"  {'SL':>4}  {'n':>5} {'WR%':>5} {'PF':>5} {'sumR':>8} {'DD':>6} {'avgR':>6} {'trd/d':>5}  exit-mix", flush=True)
    rows = []
    for SL in SL_GRID:
        m, em, sd = run_for_sl(SL)
        rows.append({"sl": SL, **m, "trd_per_day": m["n"]/sd, "exits": em})
        em_str = ", ".join(f"{k}={v}" for k, v in em.items())
        print(f"  {SL:>4.1f}  {m['n']:>5,} {m['wr']*100:>5.1f} {m['pf']:>5.2f} {m['sum_r']:>+8.1f} "
              f"{m['max_dd_r']:>6.1f} {m['avg_r']:>+6.3f} {m['n']/sd:>5.2f}  {em_str}", flush=True)

    (HERE / "sl_sweep_summary.json").write_text(json.dumps(rows, indent=2, default=str))
    print(f"\n  done in {time.time()-t0:.0f}s  wrote sl_sweep_summary.json", flush=True)


if __name__ == "__main__":
    main()
