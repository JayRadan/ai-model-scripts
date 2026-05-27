"""Hermes DJI — multi-position portfolio simulator on holdout.

Loads the saved M1-only model, replays the holdout window with:
  - max 4 concurrent positions
  - 5-bar cooldown per direction
  - switch-delta 0.5 (close worst Q if better Q candidate)
  - BE-on-new-entry (move SL to entry when new high-Q signal arrives)
  - $2 spread per trade
  - SL = 4×ATR, trail = 3×ATR, max_hold = 300 bars

Reports per-Q sweep and USD pnl at 1 contract = $1/point on DJ30 CFD.
"""
from __future__ import annotations

import pickle, sys, time
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

CUTOFF = pd.Timestamp("2025-09-01 00:00:00")
SL, TRAIL, MAX_HOLD = 4.0, 3.0, 300
NEAR, COUNTER_THR = 0.50, 1.5
MAX_CONCURRENT = 4
SWITCH_DELTA = 0.5
COOLDOWN_BARS = 5
PROFIT_TO_BE_R = 1.0
DJI_SPREAD_USD = 2.0

MODEL_PKL = HERE / "hermes_dji_native_std.pkl"  # from fair-comparison run


def bars_in_regime_array(cdir):
    n = len(cdir); out = np.zeros(n, dtype=np.int64); cur = 1
    for i in range(1, n):
        cur = cur + 1 if cdir[i] == cdir[i - 1] else 1
        out[i] = cur
    return out


def main():
    t0 = time.time()
    print("="*72); print("  Hermes DJI — portfolio sim (multi-pos + BE + switch)"); print("="*72)

    print("\n[1/4] loading native M1 + features ...", flush=True)
    m1 = pd.read_parquet(ROOT / "data" / "m1_dji_full.parquet").sort_values("time").reset_index(drop=True)
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
    median_atr = float(np.nanmedian(atr))
    spread_R = DJI_SPREAD_USD / median_atr
    print(f"  bars={n:,}  median_atr={median_atr:.3f}  spread_R={spread_R:.3f}", flush=True)

    print("\n[2/4] building holdout candidates ...", flush=True)
    dist_signed = np.where(atr > 0, (C - line) / atr, 0.0)
    dist_abs = np.abs(dist_signed)
    valid = (np.isfinite(atr) & (atr > 0))
    valid[:200] = False; valid[-(MAX_HOLD + 1):] = False
    pullback_mask = dist_abs <= NEAR
    counter_mask = (dist_signed * cdir) <= -COUNTER_THR
    valid &= (cdir != 0) & (pullback_mask | counter_mask)
    test_valid = valid & (times >= np.datetime64(CUTOFF))
    test_idxs = np.where(test_valid)[0]
    print(f"  test candidates: {len(test_idxs):,}", flush=True)

    print("\n[3/4] loading model + predicting Q ...", flush=True)
    with open(MODEL_PKL, "rb") as f:
        bundle = pickle.load(f)
    mdl = bundle["q_model"]; feat_cols = bundle["feat_cols"]
    extra = pd.DataFrame({
        "dist_at_signal": dist_signed[test_idxs], "dist_abs": dist_abs[test_idxs],
        "regime_age": bir[test_idxs],
        "bar_range_atr": (H[test_idxs] - L[test_idxs]) / np.maximum(atr[test_idxs], 1e-9),
    }).reset_index(drop=True)
    feats_std = df.iloc[test_idxs][[f for f in STD_FEATS if f not in extra.columns and f in df.columns]].reset_index(drop=True)
    X = pd.concat([extra, feats_std], axis=1)[feat_cols].fillna(0).to_numpy(np.float32)
    q_te = mdl.predict(X)
    print(f"  features: {len(feat_cols)}, Q distribution: min={q_te.min():.2f} median={np.median(q_te):.2f} max={q_te.max():.2f}")

    print("\n[4/4] running portfolio simulator across Q thresholds ...", flush=True)
    q_by_idx = {int(test_idxs[k]): float(q_te[k]) for k in range(len(test_idxs))}
    dir_by_idx = {int(test_idxs[k]): int(cdir[test_idxs[k]]) for k in range(len(test_idxs))}

    def run_exec(thr):
        active = []; executed = []
        last_open = {-1: -10**9, +1: -10**9}
        bar_start = int(test_idxs[0]); bar_end = min(int(test_idxs[-1]) + MAX_HOLD + 1, n)
        for i in range(bar_start, bar_end):
            # Update active trades
            still = []
            for t in active:
                if i < t["entry_idx"]: still.append(t); continue
                if i > min(t["entry_idx"] + MAX_HOLD, n - 1):
                    cp = C[min(t["entry_idx"] + MAX_HOLD, n - 1)]
                    t["pnl_R"] = float(t["direction"] * (cp - t["ep"]) / t["a"] - spread_R)
                    t["exit_reason"] = "max_hold"; executed.append(t); continue
                d = t["direction"]; ep = t["ep"]; a = t["a"]
                favor_now = d * (C[i] - ep)
                if favor_now > t["max_favor"]: t["max_favor"] = favor_now
                sl_r = t["sl_r"]; hit = False
                if sl_r == 0:  # BE stop
                    if d == 1 and L[i] <= ep:
                        t["pnl_R"] = -spread_R; t["exit_reason"] = "be_stop"; hit = True
                    elif d == -1 and H[i] >= ep:
                        t["pnl_R"] = -spread_R; t["exit_reason"] = "be_stop"; hit = True
                else:
                    dist_r = abs(sl_r) * a
                    if d == 1 and (ep - L[i]) >= dist_r:
                        t["pnl_R"] = float(sl_r - spread_R); t["exit_reason"] = "hard_sl"; hit = True
                    elif d == -1 and (H[i] - ep) >= dist_r:
                        t["pnl_R"] = float(sl_r - spread_R); t["exit_reason"] = "hard_sl"; hit = True
                if hit: executed.append(t); continue
                trail_d = TRAIL * a
                if t["max_favor"] >= trail_d and (t["max_favor"] - favor_now) >= trail_d:
                    t["pnl_R"] = float((t["max_favor"] - trail_d) / a - spread_R)
                    t["exit_reason"] = "trail"; executed.append(t); continue
                still.append(t)
            active = still

            if i not in q_by_idx: continue
            if q_by_idx[i] < thr: continue
            direction = dir_by_idx[i]
            if i - last_open[direction] < COOLDOWN_BARS: continue

            # BE-on-new-entry: move SL to entry for trades already 1R+ in favor
            for t in active:
                if t["sl_r"] == 0: continue
                cur_R = t["direction"] * (C[i] - t["ep"]) / t["a"]
                if cur_R >= PROFIT_TO_BE_R: t["sl_r"] = 0

            entry_idx = i + 1
            if entry_idx >= n: continue
            a_new = atr[i]
            if not (np.isfinite(a_new) and a_new > 0): continue

            if len(active) >= MAX_CONCURRENT:
                worst = min(active, key=lambda x: x["q"])
                if q_by_idx[i] >= worst["q"] + SWITCH_DELTA:
                    cp = C[i]
                    worst["pnl_R"] = float(worst["direction"] * (cp - worst["ep"]) / worst["a"] - spread_R)
                    worst["exit_reason"] = "switch_close"; executed.append(worst); active.remove(worst)
                else: continue

            active.append({
                "signal_idx": i, "entry_idx": entry_idx, "direction": direction,
                "ep": float(O[entry_idx]), "a": float(a_new),
                "sl_r": float(-SL), "max_favor": 0.0, "q": float(q_by_idx[i]),
                "time": pd.Timestamp(times[i]),
            })
            last_open[direction] = i
        return executed

    test_times = times[test_idxs]
    span_days = max(int((test_times[-1] - test_times[0]) / np.timedelta64(1, "D")), 1)
    span_tdays = span_days * 5 // 7  # rough trading days

    print(f"\n  holdout span: {span_days} cal days (~{span_tdays} trading days)")
    print(f"  {'Q':>4}  {'n':>5}  {'WR':>5}  {'PF':>6}  {'sumR':>8}  {'DD R':>6}  {'trd/day':>7}  {'USD@$1/pt':>10}")
    for qt in [2.0, 2.5, 3.0, 4.0, 5.0]:
        trades = run_exec(qt)
        if not trades: continue
        R = np.array([t["pnl_R"] for t in trades if t.get("pnl_R") is not None])
        # USD pnl: each R = ATR points. ATR(median)=7.4 pt → at $1/pt avg pt-pnl per trade = R*ATR_at_trade
        usd = sum((t["pnl_R"] * t["a"]) for t in trades if t.get("pnl_R") is not None)
        w, l = R[R > 0], R[R <= 0]
        pf = float(w.sum() / max(-l.sum(), 1e-9))
        eq = np.cumsum(R)
        dd = float((np.maximum.accumulate(eq) - eq).max())
        print(f"  {qt:>4}  {len(trades):>5}  {(R>0).mean()*100:>4.1f}%  {pf:>6.2f}  "
              f"{R.sum():>+8.0f}  {dd:>6.0f}  {len(trades)/max(span_tdays,1):>7.1f}  ${usd:>+9.0f}")

    print(f"\n  done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
