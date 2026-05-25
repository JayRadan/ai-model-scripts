"""Generate Hermes backtest curve from the FROZEN bundle on actual holdout data.

Writes the equity_curve / equity_dates arrays into backtest_data.json so the
website's BacktestChart renders real trade-by-trade USD equity (same as Oracle).
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/jay/Desktop/new-model-zigzag")
SERVER = Path("/home/jay/Desktop/my-agents-and-website/commercial/server")
sys.path.insert(0, str(ROOT / "experiments/v103_tfk_regime"))
sys.path.insert(0, str(SERVER))

from tfk import compute_tfk
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "ofm1", ROOT / "experiments/v103_tfk_regime/43_m1_with_orderflow.py")
_ofm1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ofm1)
add_standard_features = _ofm1.add_standard_features
STD_FEATS = list(_ofm1.STD_FEATS)
FLOW_FEATS = list(_ofm1.FLOW_FEATS)

CUTOFF = pd.Timestamp("2025-09-01 00:00:00")
SL = 4.0; TRAIL = 3.0; MAX_HOLD = 300; NEAR = 0.50; Q_THR = 1.0
MAX_CONCURRENT = 4; SWITCH_DELTA = 0.5; COOLDOWN_BARS = 5; PROFIT_TO_BE_R = 1.0
SPREAD_USD = 0.30
LOT = 0.01
PIP_VALUE_PER_LOT_PER_DOLLAR = 100.0  # 1 USD move on XAU at 0.01 lot = $1 per pip... actually
# For XAU on standard MT5 broker: 1 dollar price move @ 0.01 lot = $1 P&L (cent units)
USD_PER_DOLLAR_MOVE_AT_BASE_LOT = 1.0
STARTING_USD = 1000.0


def simulate_label(entry_idx, direction, C, H, L, O, atr_at_entry, spread_R=0.0):
    n = len(C)
    if entry_idx >= n - 1 or not (np.isfinite(atr_at_entry) and atr_at_entry > 0):
        return np.nan, "skip", entry_idx
    ep = O[entry_idx]; a = atr_at_entry
    hard = SL * a; trail = TRAIL * a; max_favor = 0.0
    end = min(entry_idx + MAX_HOLD, n - 1)
    for k in range(entry_idx, end + 1):
        favor_now = direction * (C[k] - ep)
        if favor_now > max_favor: max_favor = favor_now
        if direction == 1:
            if (ep - L[k]) >= hard: return -SL - spread_R, "hard_sl", k
        else:
            if (H[k] - ep) >= hard: return -SL - spread_R, "hard_sl", k
        if max_favor >= trail:
            if (max_favor - favor_now) >= trail:
                return (max_favor - trail) / a - spread_R, "trail", k
    return direction * (C[end] - ep) / a - spread_R, "max_hold", end


def bars_in_regime_array(cdir):
    n = len(cdir); out = np.zeros(n, dtype=np.int64); cur = 1
    for i in range(1, n):
        if cdir[i] == cdir[i - 1]: cur += 1
        else: cur = 1
        out[i] = cur
    return out


def main():
    t0 = time.time()
    print("Loading data + bundle ...", flush=True)
    m1 = pd.read_parquet(ROOT / "data/m1_xau_orderflow.parquet").sort_values("time").reset_index(drop=True)

    # Load the FROZEN bundle
    import pickle
    with open(SERVER / "decision_engine/models/hermes_xau_validated.pkl", "rb") as f:
        bundle = pickle.load(f)
    q_mdl = bundle["q_model"]
    feat_cols = bundle["feat_cols"]
    print(f"  bundle feat_cols: {len(feat_cols)}", flush=True)

    print("Computing TFK + features ...", flush=True)
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
    spread_R = SPREAD_USD / np.nanmedian(atr)
    median_atr = float(np.nanmedian(atr))
    print(f"  median ATR = ${median_atr:.3f}", flush=True)

    dist_signed = np.where(atr > 0, (C - line) / atr, 0.0)
    dist_abs = np.abs(dist_signed)
    valid = (np.isfinite(atr) & (atr > 0))
    valid[:200] = False; valid[-(MAX_HOLD + 1):] = False
    valid &= (dist_abs <= NEAR) & (cdir != 0)
    idxs = np.where(valid)[0]
    dirs = cdir[idxs].copy()
    print(f"  candidates: {len(idxs):,}", flush=True)

    # Label all candidates
    pnl_s = np.zeros(len(idxs), dtype=np.float32)
    exits = np.zeros(len(idxs), dtype=np.int32)
    for k, i in enumerate(idxs):
        d = int(dirs[k])
        r, _, xi = simulate_label(i + 1, d, C, H, L, O, atr[i], spread_R=spread_R)
        pnl_s[k] = r if np.isfinite(r) else 0.0
        exits[k] = xi

    # Build feature matrix matching bundle order
    extra = pd.DataFrame({
        "dist_at_signal": dist_signed[idxs], "dist_abs": dist_abs[idxs],
        "regime_age": bir[idxs],
        "bar_range_atr": (H[idxs] - L[idxs]) / np.maximum(atr[idxs], 1e-9),
    })
    feats_std = df.iloc[idxs][[f for f in STD_FEATS if f not in extra.columns and f in df.columns]].reset_index(drop=True)
    feats_flow = df.iloc[idxs][[f for f in FLOW_FEATS if f in df.columns]].reset_index(drop=True)
    X = pd.concat([extra.reset_index(drop=True), feats_std, feats_flow], axis=1)
    # Reorder to bundle's exact order
    X = X[feat_cols]

    times_at_idx = times[idxs]
    test_m = times_at_idx >= np.datetime64(CUTOFF)

    print("Predicting Q on holdout ...", flush=True)
    q_te = q_mdl.predict(X.loc[test_m].fillna(0).to_numpy(np.float32))
    test_idxs = idxs[test_m]
    test_dirs = dirs[test_m]
    test_pnl = pnl_s[test_m]
    test_exits = exits[test_m]
    test_times = times_at_idx[test_m]
    print(f"  test candidates: {len(test_idxs):,}  range: {test_times[0]} → {test_times[-1]}", flush=True)

    # Execute with cooldown + max_concurrent + switch + BE-on-new-entry
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
                worst["exit_idx"] = i
                executed.append(worst); active.remove(worst)
            else: continue
        active.append({"signal_idx": int(i), "entry_idx": int(entry_idx),
                       "direction": int(d_), "ep": float(ep_new), "a": float(a_new),
                       "sl_r": float(-SL), "max_favor": 0.0, "q": float(q_by_idx[i]),
                       "time": pd.Timestamp(times[i]),
                       "exit_idx": int(entry_idx), "pnl_R": None})
        last_open_per_dir[d_] = i
    for t in active:
        end_bar = min(t["entry_idx"] + MAX_HOLD, n - 1)
        cp = C[end_bar]
        t["pnl_R"] = float(t["direction"] * (cp - t["ep"]) / t["a"] - spread_R)
        t["exit_idx"] = end_bar
        executed.append(t)

    trades = pd.DataFrame(executed).sort_values("entry_idx").reset_index(drop=True)
    trades = trades[np.isfinite(trades["pnl_R"])]
    print(f"\n  executed: {len(trades):,} trades", flush=True)

    r = trades["pnl_R"].to_numpy()
    pf = r[r > 0].sum() / max(-r[r <= 0].sum(), 1e-9)
    wins = trades[trades.pnl_R > 0]; losses = trades[trades.pnl_R <= 0]
    print(f"  WR={(r>0).mean()*100:.1f}%  PF={pf:.2f}  sumR={r.sum():+.0f}", flush=True)

    # Convert R-units to USD at base_lot=0.01
    # Each R = `atr` price units; @ 0.01 lot on XAU, 1$ price move = $1 P&L (varies by broker)
    # Use per-bar ATR (varies). Compute realized $ per trade.
    usd_pnl = []
    for t in executed:
        if not np.isfinite(t["pnl_R"]): continue
        # USD = R × atr_at_entry × dollars_per_dollar_at_base_lot
        usd_pnl.append(t["pnl_R"] * t["a"] * USD_PER_DOLLAR_MOVE_AT_BASE_LOT * (LOT / 0.01))
    usd_pnl = np.array(usd_pnl)
    # Equity curve
    equity = STARTING_USD + np.cumsum(usd_pnl)
    equity = np.concatenate([[STARTING_USD], equity])
    # Build dates per-trade
    sorted_trades = sorted(executed, key=lambda x: x["entry_idx"])
    dates = ["2025-09-01"] + [str(pd.Timestamp(times[t["entry_idx"]]))[:10] for t in sorted_trades]

    # Sample to 200 points
    if len(equity) > 200:
        idxs_sample = np.linspace(0, len(equity) - 1, 200).astype(int)
        eq_sample = equity[idxs_sample].tolist()
        d_sample = [dates[i] for i in idxs_sample]
    else:
        eq_sample = equity.tolist()
        d_sample = dates

    # Stats
    total_pnl_usd = float(usd_pnl.sum())
    max_dd_usd = float((np.maximum.accumulate(equity) - equity).max())
    avg_win = float(np.mean([p for p in usd_pnl if p > 0])) if any(p > 0 for p in usd_pnl) else 0
    avg_loss = float(np.mean([p for p in usd_pnl if p <= 0])) if any(p <= 0 for p in usd_pnl) else 0
    calendar_days = (pd.Timestamp(test_times[-1]) - pd.Timestamp(test_times[0])).days

    # Update backtest_data.json
    P = Path("/home/jay/Desktop/my-agents-and-website/commercial/website/public/backtest_data.json")
    bdata = json.load(P.open())
    bdata["hermes"].update({
        "image": "",
        "period": f"{str(pd.Timestamp(test_times[0]))[:10]} – {str(pd.Timestamp(test_times[-1]))[:10]} (clean holdout)",
        "calendar_days": int(calendar_days),
        "base_lot": LOT,
        "pip_value_per_lot": 1.0,
        "total_trades": int(len(trades)),
        "trades_per_day": round(len(trades) / max(calendar_days, 1), 2),
        "win_rate": round((r > 0).mean() * 100, 1),
        "profit_factor": round(float(pf), 2),
        "total_pnl": round(total_pnl_usd, 1),
        "max_dd": round(-max_dd_usd, 1),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "rr_ratio": round(abs(avg_win / avg_loss), 2) if avg_loss < 0 else 0,
        "expectancy": round(total_pnl_usd / max(len(trades), 1), 2),
        "long_trades": int((trades.direction == 1).sum()),
        "short_trades": int((trades.direction == -1).sum()),
        "equity_curve": eq_sample,
        "equity_dates": d_sample,
    })
    json.dump(bdata, P.open("w"), indent=2)
    print(f"\n  wrote {P}", flush=True)
    print(f"  Hermes USD stats: total_pnl=${total_pnl_usd:.0f}  max_dd=${-max_dd_usd:.0f}  "
          f"final=${equity[-1]:.0f}  n_trades={len(trades)}  {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
