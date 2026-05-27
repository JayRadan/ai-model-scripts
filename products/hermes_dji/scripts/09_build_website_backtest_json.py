"""Generate Hermes DJI backtest curve from the FROZEN bundle on holdout data.
Adds a "hermes_dji" entry to public/backtest_data.json so the website's
BacktestChart renders real trade-by-trade USD equity for Dow.
"""
from __future__ import annotations
import json, pickle, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/jay/Desktop/new-model-zigzag")
SERVER = Path("/home/jay/Desktop/my-agents-and-website/commercial/server")
sys.path.insert(0, str(ROOT / "experiments/v103_tfk_regime"))
sys.path.insert(0, str(SERVER))
sys.path.insert(0, str(ROOT / "products/hermes_dji"))

from tfk import compute_tfk
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "ofm1", ROOT / "experiments/v103_tfk_regime/43_m1_with_orderflow.py")
_ofm1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ofm1)
add_standard_features = _ofm1.add_standard_features
STD_FEATS = list(_ofm1.STD_FEATS)

# Match hermes_dji production config
CUTOFF = pd.Timestamp("2025-09-01 00:00:00")
SL, TRAIL, MAX_HOLD = 4.0, 3.0, 300
NEAR, COUNTER_THR = 0.50, 1.5
Q_THR = 3.0
MAX_CONCURRENT, SWITCH_DELTA, COOLDOWN_BARS = 4, 0.5, 5
PROFIT_TO_BE_R = 1.0
SPREAD_USD = 2.0
LOT = 0.01
USD_PER_DOLLAR_MOVE_AT_BASE_LOT = 0.10   # MT5 Dow CFD: ~$0.10/pt at 0.01 lot (Vantage/FBS scaling)
STARTING_USD = 1000.0


def bars_in_regime_array(cdir):
    n = len(cdir); out = np.zeros(n, dtype=np.int64); cur = 1
    for i in range(1, n):
        cur = cur + 1 if cdir[i] == cdir[i - 1] else 1
        out[i] = cur
    return out


def main():
    t0 = time.time()
    print("Loading bars + bundle ...", flush=True)
    m1 = pd.read_parquet(ROOT / "data/m1_dji_full.parquet").sort_values("time").reset_index(drop=True)
    with open(SERVER / "decision_engine/models/hermes_dji_validated.pkl", "rb") as f:
        bundle = pickle.load(f)
    q_mdl = bundle["q_model"]; feat_cols = bundle["feat_cols"]
    print(f"  bundle feat_cols: {len(feat_cols)}", flush=True)

    print("TFK + standard features ...", flush=True)
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
    spread_R = SPREAD_USD / float(np.nanmedian(atr))

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

    extra = pd.DataFrame({
        "dist_at_signal": dist_signed[test_idxs], "dist_abs": dist_abs[test_idxs],
        "regime_age": bir[test_idxs],
        "bar_range_atr": (H[test_idxs] - L[test_idxs]) / np.maximum(atr[test_idxs], 1e-9),
    }).reset_index(drop=True)
    feats_std = df.iloc[test_idxs][[f for f in STD_FEATS if f not in extra.columns and f in df.columns]].reset_index(drop=True)
    X = pd.concat([extra, feats_std], axis=1)[feat_cols].fillna(0).to_numpy(np.float32)
    q_te = q_mdl.predict(X)

    q_by_idx = {int(test_idxs[k]): float(q_te[k]) for k in range(len(test_idxs))}
    dir_by_idx = {int(test_idxs[k]): int(cdir[test_idxs[k]]) for k in range(len(test_idxs))}

    # Multi-position portfolio sim
    active = []; executed = []
    last_open = {-1: -10**9, +1: -10**9}
    bar_start = int(test_idxs[0]); bar_end = min(int(test_idxs[-1]) + MAX_HOLD + 1, n)
    for i in range(bar_start, bar_end):
        still = []
        for t in active:
            if i < t["entry_idx"]: still.append(t); continue
            if i > min(t["entry_idx"] + MAX_HOLD, n - 1):
                cp = C[min(t["entry_idx"] + MAX_HOLD, n - 1)]
                t["pnl_R"] = float(t["direction"] * (cp - t["ep"]) / t["a"] - spread_R)
                t["exit_idx"] = min(t["entry_idx"] + MAX_HOLD, n - 1)
                executed.append(t); continue
            d = t["direction"]; ep = t["ep"]; a = t["a"]
            favor_now = d * (C[i] - ep)
            if favor_now > t["max_favor"]: t["max_favor"] = favor_now
            sl_r = t["sl_r"]; hit = False
            if sl_r == 0:
                if d == 1 and L[i] <= ep:
                    t["pnl_R"] = -spread_R; t["exit_idx"] = i; hit = True
                elif d == -1 and H[i] >= ep:
                    t["pnl_R"] = -spread_R; t["exit_idx"] = i; hit = True
            else:
                dist_r = abs(sl_r) * a
                if d == 1 and (ep - L[i]) >= dist_r:
                    t["pnl_R"] = float(sl_r - spread_R); t["exit_idx"] = i; hit = True
                elif d == -1 and (H[i] - ep) >= dist_r:
                    t["pnl_R"] = float(sl_r - spread_R); t["exit_idx"] = i; hit = True
            if hit: executed.append(t); continue
            trail_d = TRAIL * a
            if t["max_favor"] >= trail_d and (t["max_favor"] - favor_now) >= trail_d:
                t["pnl_R"] = float((t["max_favor"] - trail_d) / a - spread_R)
                t["exit_idx"] = i; executed.append(t); continue
            still.append(t)
        active = still

        if i not in q_by_idx: continue
        if q_by_idx[i] < Q_THR: continue
        d_ = dir_by_idx[i]
        if i - last_open[d_] < COOLDOWN_BARS: continue
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
                worst["exit_idx"] = i; executed.append(worst); active.remove(worst)
            else: continue
        active.append({"entry_idx": int(entry_idx), "direction": int(d_),
                       "ep": float(O[entry_idx]), "a": float(a_new),
                       "sl_r": float(-SL), "max_favor": 0.0, "q": float(q_by_idx[i]),
                       "exit_idx": int(entry_idx), "pnl_R": None})
        last_open[d_] = i
    for t in active:
        end_bar = min(t["entry_idx"] + MAX_HOLD, n - 1)
        t["pnl_R"] = float(t["direction"] * (C[end_bar] - t["ep"]) / t["a"] - spread_R)
        t["exit_idx"] = end_bar; executed.append(t)

    trades = pd.DataFrame(executed).sort_values("entry_idx").reset_index(drop=True)
    trades = trades[np.isfinite(trades["pnl_R"])]
    r = trades["pnl_R"].to_numpy()
    pf = float(r[r > 0].sum() / max(-r[r <= 0].sum(), 1e-9))
    print(f"  executed: {len(trades):,}  WR={(r>0).mean()*100:.1f}%  PF={pf:.2f}  sumR={r.sum():+.0f}")

    # USD scaling
    usd_pnl = []
    for t in executed:
        if not np.isfinite(t["pnl_R"]): continue
        usd_pnl.append(t["pnl_R"] * t["a"] * USD_PER_DOLLAR_MOVE_AT_BASE_LOT * (LOT / 0.01))
    usd_pnl = np.array(usd_pnl)
    equity = STARTING_USD + np.cumsum(usd_pnl)
    equity = np.concatenate([[STARTING_USD], equity])
    sorted_trades = sorted(executed, key=lambda x: x["entry_idx"])
    dates = [str(pd.Timestamp(times[test_idxs[0]]))[:10]] + \
            [str(pd.Timestamp(times[t["entry_idx"]]))[:10] for t in sorted_trades]
    if len(equity) > 200:
        idxs_s = np.linspace(0, len(equity) - 1, 200).astype(int)
        eq_sample = equity[idxs_s].tolist()
        d_sample = [dates[i] for i in idxs_s]
    else:
        eq_sample = equity.tolist(); d_sample = dates

    test_times = times[test_idxs]
    total_pnl_usd = float(usd_pnl.sum())
    max_dd_usd = float((np.maximum.accumulate(equity) - equity).max())
    avg_win = float(np.mean([p for p in usd_pnl if p > 0])) if any(p > 0 for p in usd_pnl) else 0
    avg_loss = float(np.mean([p for p in usd_pnl if p <= 0])) if any(p <= 0 for p in usd_pnl) else 0
    calendar_days = (pd.Timestamp(test_times[-1]) - pd.Timestamp(test_times[0])).days

    P = Path("/home/jay/Desktop/my-agents-and-website/commercial/website/public/backtest_data.json")
    bdata = json.load(P.open())
    bdata["hermes_dji"] = {
        "name": "EdgePredictor Hermes DJI",
        "asset": "US30",
        "color": "#ef4444",
        "timeframe": "M1",
        "regimes": [],
        "top_rules": [],
        "image": "",
        "period": f"{str(pd.Timestamp(test_times[0]))[:10]} – {str(pd.Timestamp(test_times[-1]))[:10]} (clean holdout)",
        "calendar_days": int(calendar_days),
        "base_lot": LOT,
        "pip_value_per_lot": USD_PER_DOLLAR_MOVE_AT_BASE_LOT * 100,  # display only
        "total_trades": int(len(trades)),
        "trades_per_day": round(len(trades) / max(calendar_days, 1), 2),
        "win_rate": round((r > 0).mean() * 100, 1),
        "profit_factor": round(pf, 2),
        "total_pnl": round(total_pnl_usd, 1),
        "max_dd": round(-max_dd_usd, 1),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "rr_ratio": round(abs(avg_win / avg_loss), 2) if avg_loss < 0 else 0,
        "expectancy": round(total_pnl_usd / max(len(trades), 1), 2),
        "long_trades": int((trades.direction == 1).sum()),
        "short_trades": int((trades.direction == -1).sum()),
        "active_rules": 0, "total_rules": 0,
        "max_win_streak": 0, "max_loss_streak": 0,
        "equity_curve": eq_sample,
        "equity_dates": d_sample,
    }
    json.dump(bdata, P.open("w"), indent=2)
    print(f"\n  wrote {P.name}", flush=True)
    print(f"  total_pnl=${total_pnl_usd:.1f}  max_dd=${-max_dd_usd:.1f}  "
          f"final=${equity[-1]:.1f}  trades={len(trades)}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
