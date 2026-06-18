"""
10_sim_today.py — Atlas XAU live-equivalent simulation for the current day.

2026-06-17: REWRITTEN for the new ushape_m15 architecture (M15 macro Kalman
+ M1 Kalman U-shape edge entry). Replaces the prior STRICT-2-bar-reversal sim.

Pipeline:
  1. load deployed bundle (atlas_xau_validated.pkl, M15 U-shape Q)
  2. fetch last ~3 days of M1 XAU from Dukascopy
  3. compute features (M1 TFK+std+Kalman + M15 Kalman causal forward-fill)
  4. build candidates (edge-detected U-shape, M15 macro filter)
  5. predict Q, multi-pos sim with same rules as decide_atlas
  6. filter to today's entries, print PnL summary
"""
from __future__ import annotations
import sys, time, importlib.util, pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np, pandas as pd
import dukascopy_python

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
SERVER = Path("/home/jay/Desktop/my-agents-and-website/commercial/server")
sys.path.insert(0, str(ROOT / "products/hermes_xau"))
sys.path.insert(0, str(ROOT / "experiments/kalman_color_flip"))
from tfk import compute_tfk
from kalman import compute_kalman
_spec = importlib.util.spec_from_file_location("ofm1", ROOT / "products/_shared/m1_with_orderflow.py")
_ofm1 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_ofm1)
add_standard_features = _ofm1.add_standard_features

BUNDLE_PATH = SERVER / "decision_engine/models/atlas_xau_validated.pkl"
SPREAD       = 0.30
SL           = 6.0
TRAIL        = 1.0
MAX_HOLD     = 300
MAX_CONCURRENT = 4
SWITCH_DELTA = 0.5
COOLDOWN_BARS = 5
PROFIT_TO_BE_R = 0.5
Q_THR        = 2.0          # 2026-06-18 dynamic Q: strict chop default (raised 1.5→2.0)
Q_THR_TREND  = 0.5          # 2026-06-18 dynamic Q: looser when trend_strong
TREND_AGE_MIN     = 30
TREND_SLOPE_MIN   = 1.0
TREND_DEMA50_MIN  = 1.0
R_to_USD     = 1.50         # 0.01 lot XAUUSD
KAL = dict(q=0.05, r_mult=1.0, r_len=50, dt=1.0, mintick=0.01)


def simulate_label_R(entry_idx, direction, C, H, L, O, a, spread_R):
    n = len(C)
    if entry_idx >= n-1 or not (np.isfinite(a) and a > 0): return np.nan, None, "warmup"
    ep = O[entry_idx]
    hard = SL*a; trail_d = TRAIL*a; max_favor = 0.0
    end = min(entry_idx + MAX_HOLD, n-1)
    for k in range(entry_idx, end+1):
        favor_now = direction*(C[k] - ep)
        if favor_now > max_favor: max_favor = favor_now
        if direction == 1:
            if (ep - L[k]) >= hard: return -SL - spread_R, k, "hard_sl"
        else:
            if (H[k] - ep) >= hard: return -SL - spread_R, k, "hard_sl"
        if max_favor >= trail_d:
            if (max_favor - favor_now) >= trail_d:
                return (max_favor - trail_d)/a - spread_R, k, "trail"
    return direction*(C[end]-ep)/a - spread_R, end, "max_hold"


def main():
    t0 = time.time()
    print(f"[atlas_xau] loading bundle ...")
    bundle = pickle.load(open(BUNDLE_PATH, "rb"))
    feat_cols = bundle["feat_cols"]
    macro_tf = bundle["atlas_params"].get("macro_tf_min", 15)
    q_thr = bundle["atlas_params"].get("q_thr", Q_THR)

    end = datetime.now(timezone.utc).replace(microsecond=0)
    start = end - timedelta(days=3)
    print(f"[atlas_xau] Fetching XAU/USD M1 {start.isoformat()} → {end.isoformat()}")
    df = dukascopy_python.fetch(instrument="XAU/USD", interval=dukascopy_python.INTERVAL_MIN_1,
                                 offer_side=dukascopy_python.OFFER_SIDE_BID, start=start, end=end)
    df = df.reset_index().rename(columns={"timestamp": "time"})
    df["time"] = pd.to_datetime(df["time"]).dt.tz_convert("UTC").dt.tz_localize(None)
    df = df.sort_values("time").reset_index(drop=True)
    if "tick_volume" not in df.columns:
        vc = "volume" if "volume" in df.columns else [c for c in df.columns if "vol" in c.lower()][0]
        df["tick_volume"] = df[vc]
    print(f"  fetched {len(df):,} bars  last bar: {df.time.iloc[-1]}")

    print(f"  computing features ...")
    fdf = compute_tfk(df)
    fdf = add_standard_features(fdf)
    fdf = compute_kalman(fdf, **KAL)
    # M15 Kalman causal
    g = fdf.set_index("time")[["open","high","low","close","tick_volume"]].resample(f"{macro_tf}min").agg(
        {"open":"first","high":"max","low":"min","close":"last","tick_volume":"sum"}).dropna().reset_index()
    g = compute_kalman(g, **KAL)
    g["end"] = g["time"] + pd.Timedelta(minutes=macro_tf)
    j = np.searchsorted(g["end"].values, fdf["time"].values, side="right") - 1
    valid_j = j >= 0
    jj = np.clip(j, 0, len(g)-1)
    fdf[f"kf_p_m{macro_tf}"]     = np.where(valid_j, g["kf_p"].to_numpy()[jj], np.nan)
    fdf[f"kf_dir_m{macro_tf}"]   = np.where(valid_j, g["kf_dir"].to_numpy()[jj], 0).astype(np.int64)
    fdf[f"kf_v_m{macro_tf}"]     = np.where(valid_j, g["kf_v"].to_numpy()[jj], 0.0)
    fdf[f"f_accel_m{macro_tf}"]  = np.where(valid_j, g["f_accel"].to_numpy()[jj], 0.0)
    fdf[f"f_velPct_m{macro_tf}"] = np.where(valid_j, g["f_velPct"].to_numpy()[jj], 0.0)
    atr_arr = fdf["atr14"].to_numpy(np.float64)
    C_arr = fdf["close"].to_numpy(np.float64)
    fdf[f"dist_m{macro_tf}kf"] = np.where(atr_arr > 0,
        (C_arr - fdf[f"kf_p_m{macro_tf}"].to_numpy(np.float64)) / atr_arr, 0.0)
    # kv_pos_50
    kv_s = pd.Series(fdf["kf_v"].to_numpy(np.float64))
    kv_min50 = kv_s.rolling(50, min_periods=20).min().fillna(0).to_numpy()
    kv_max50 = kv_s.rolling(50, min_periods=20).max().fillna(0).to_numpy()
    fdf["kv_pos_50"] = (kv_s.to_numpy() - kv_min50) / np.maximum(kv_max50 - kv_min50, 1e-9)
    # bar_range_atr
    H_arr = fdf["high"].to_numpy(np.float64); L_arr = fdf["low"].to_numpy(np.float64)
    fdf["bar_range_atr"] = (H_arr - L_arr) / np.maximum(atr_arr, 1e-9)

    # build edge-detected candidates
    kd15 = fdf[f"kf_dir_m{macro_tf}"].to_numpy(np.int64)
    kd1  = fdf["kf_dir"].to_numpy(np.int64)
    kv1  = fdf["kf_v"].to_numpy(np.float64)
    fa1  = fdf["f_accel"].to_numpy(np.float64)
    buy_raw  = (kd15 == +1) & (kd1 == -1) & (kv1 < 0) & (fa1 > 0)
    sell_raw = (kd15 == -1) & (kd1 == +1) & (kv1 > 0) & (fa1 < 0)
    buy_edge  = buy_raw  & ~np.concatenate([[False], buy_raw[:-1]])
    sell_edge = sell_raw & ~np.concatenate([[False], sell_raw[:-1]])
    valid = np.isfinite(atr_arr) & (atr_arr > 0)
    valid[:500] = False; valid[-(MAX_HOLD+1):] = False
    mask = (buy_edge | sell_edge) & valid
    cand_idxs = np.where(mask)[0]
    cand_dirs = np.where(buy_edge[cand_idxs], +1, -1).astype(np.int64)
    print(f"  U-shape edge candidates over 3-day window: {len(cand_idxs)}")
    if len(cand_idxs) == 0:
        print("  no candidates"); return

    # predict Q in bundle's feat_cols order
    X_rows = []
    for i in cand_idxs:
        row = []
        for f in feat_cols:
            v = fdf.iloc[i].get(f, 0.0) if f in fdf.columns else 0.0
            row.append(float(v) if (v is not None and not pd.isna(v)) else 0.0)
        X_rows.append(row)
    X = np.array(X_rows, dtype=np.float32)
    # Prefer the holdout-trained Q for live calibration (matches the model
    # whose Q distribution q_thr was tuned against in backtest).
    q_mdl_live = bundle.get("q_model_holdout") or bundle["q_model"]
    q_arr = q_mdl_live.predict(X)
    q_by_idx = {int(cand_idxs[k]): float(q_arr[k]) for k in range(len(cand_idxs))}
    dir_by_idx = {int(cand_idxs[k]): int(cand_dirs[k]) for k in range(len(cand_idxs))}
    # Dynamic Q threshold per candidate based on trend_strong
    # regime_age: bars in TFK committed_dir streak
    from kalman import bars_in_regime_array as _bira
    cdir_arr = fdf["committed_dir"].to_numpy(np.int64) if "committed_dir" in fdf.columns else np.zeros(len(fdf), dtype=np.int64)
    bir = _bira(cdir_arr)
    sl20 = fdf["slope20"].to_numpy(float); de50 = fdf["dist_ema50"].to_numpy(float)
    trend_strong = (bir[cand_idxs] >= TREND_AGE_MIN) & (np.abs(sl20[cand_idxs]) >= TREND_SLOPE_MIN) & (np.abs(de50[cand_idxs]) >= TREND_DEMA50_MIN)
    q_thr_by_idx = {int(cand_idxs[k]): (Q_THR_TREND if trend_strong[k] else Q_THR) for k in range(len(cand_idxs))}
    pass_q = q_arr >= np.array([Q_THR_TREND if t else Q_THR for t in trend_strong])
    print(f"  Q dist: median={np.median(q_arr):.2f}  p75={np.percentile(q_arr,75):.2f}  max={np.max(q_arr):.2f}")
    print(f"  trend_strong: {int(trend_strong.sum())} ({100*trend_strong.mean():.0f}%)")
    print(f"  candidates passing dynamic Q (≥{Q_THR}/chop or ≥{Q_THR_TREND}/trend): {int(pass_q.sum())}")

    # multi-pos sim
    O_arr = fdf["open"].to_numpy(np.float64)
    times = fdf["time"].to_numpy()
    spread_R = SPREAD / np.nanmedian(atr_arr)
    n = len(fdf)
    active = []; executed = []
    last_open = {-1: -10**9, +1: -10**9}
    cand_set = set(int(i) for i in cand_idxs)
    bar_start = int(cand_idxs[0]); bar_end = min(int(cand_idxs[-1]) + MAX_HOLD + 1, n)
    for i in range(bar_start, bar_end):
        still = []
        for t in active:
            if i < t["entry_idx"]: still.append(t); continue
            if i > min(t["entry_idx"] + MAX_HOLD, n-1):
                cp = C_arr[min(t["entry_idx"] + MAX_HOLD, n-1)]
                t["pnl_R"] = float(t["direction"]*(cp-t["ep"])/t["a"] - spread_R)
                t["exit_reason"] = "max_hold"; t["exit_time"] = pd.Timestamp(times[i]); t["exit_px"] = cp
                executed.append(t); continue
            d = t["direction"]; ep = t["ep"]; a = t["a"]
            favor_now = d*(C_arr[i] - ep)
            if favor_now > t["max_favor"]: t["max_favor"] = favor_now
            sl_r = t["sl_r"]; hit = False
            if sl_r == 0:
                if d == 1 and L_arr[i] <= ep:  t["pnl_R"] = -spread_R; t["exit_reason"] = "BE"; hit = True
                elif d == -1 and H_arr[i] >= ep: t["pnl_R"] = -spread_R; t["exit_reason"] = "BE"; hit = True
            else:
                dist_r = abs(sl_r)*a
                if d == 1 and (ep - L_arr[i]) >= dist_r:
                    t["pnl_R"] = float(sl_r - spread_R); t["exit_reason"] = "SL"; hit = True
                elif d == -1 and (H_arr[i] - ep) >= dist_r:
                    t["pnl_R"] = float(sl_r - spread_R); t["exit_reason"] = "SL"; hit = True
            if hit:
                t["exit_time"] = pd.Timestamp(times[i]); t["exit_px"] = float(C_arr[i])
                executed.append(t); continue
            trail_d = TRAIL*a
            if t["max_favor"] >= trail_d and (t["max_favor"] - favor_now) >= trail_d:
                t["pnl_R"] = float((t["max_favor"] - trail_d)/a - spread_R)
                t["exit_reason"] = "trail"
                t["exit_time"] = pd.Timestamp(times[i]); t["exit_px"] = float(C_arr[i])
                executed.append(t); continue
            still.append(t)
        active = still
        if i not in cand_set: continue
        if q_by_idx.get(i, -1e9) < q_thr_by_idx.get(i, q_thr): continue
        direction = dir_by_idx[i]
        if direction == 0: continue
        if i - last_open[direction] < COOLDOWN_BARS: continue
        for t in active:
            if t["sl_r"] == 0: continue
            cur_R = t["direction"]*(C_arr[i] - t["ep"])/t["a"]
            if cur_R >= PROFIT_TO_BE_R: t["sl_r"] = 0
        entry_idx = i + 1
        if entry_idx >= n: continue
        a_new = atr_arr[i]
        if not (np.isfinite(a_new) and a_new > 0): continue
        if len(active) >= MAX_CONCURRENT:
            worst = min(active, key=lambda x: x["q"])
            if q_by_idx[i] >= worst["q"] + SWITCH_DELTA:
                cp = C_arr[i]
                worst["pnl_R"] = float(worst["direction"]*(cp-worst["ep"])/worst["a"] - spread_R)
                worst["exit_reason"] = "switch_closed"
                worst["exit_time"] = pd.Timestamp(times[i]); worst["exit_px"] = float(cp)
                executed.append(worst); active.remove(worst)
            else: continue
        active.append({
            "signal_idx": i, "entry_idx": entry_idx, "direction": direction,
            "ep": float(O_arr[entry_idx]), "a": float(a_new),
            "sl_r": float(-SL), "max_favor": 0.0, "q": float(q_by_idx[i]),
            "entry_time": pd.Timestamp(times[entry_idx]),
        })
        last_open[direction] = i

    # filter to today
    today_date = pd.Timestamp(times[-1]).date()
    today = [t for t in executed if t.get("entry_time") and pd.Timestamp(t["entry_time"]).date() == today_date]
    today.sort(key=lambda t: t["entry_time"])
    print(f"\n=== Atlas XAU sim — {today_date} ===")
    print(f"({len(today)} trades opened today)")
    if not today:
        print("(no Atlas XAU trades fired today)")
        return
    R_today = np.array([t["pnl_R"] for t in today if t.get("pnl_R") is not None])
    if len(R_today) > 0:
        w, l = R_today[R_today>0], R_today[R_today<=0]
        pf = float(w.sum()/max(-l.sum(), 1e-9))
        eq = np.cumsum(R_today); dd = float((np.maximum.accumulate(eq) - eq).max())
        usd01 = sum(t["pnl_R"]*t["a"]*R_to_USD for t in today)
        usd10 = usd01*10
        dd_usd = dd * float(np.mean([t["a"] for t in today])) * R_to_USD
        print(f"sumR {R_today.sum():+.2f} | WR {(R_today>0).mean()*100:.1f}% | PF {pf:.2f} | DD {dd:.2f}R")
        print(f"Estimated USD @ 0.01 lot: {usd01:+.2f} (DD ${dd_usd:.2f})")
        print(f"Estimated USD @ 0.10 lot: {usd10:+.2f} (DD ${dd_usd*10:.2f})")
    print()
    print(f"  {'#':>2} {'entry_time':>16} {'dir':>4} {'entry_px':>9} {'exit_time':>17} {'exit_px':>9} {'atr':>5} {'Q':>5} {'R':>7} {'reason':>10}")
    for k, t in enumerate(today):
        et = t["entry_time"].strftime("%Y-%m-%d %H:%M")
        xt = t["exit_time"].strftime("%Y-%m-%d %H:%M") if t.get("exit_time") else "(open)"
        print(f"  {k:>2} {et:>16} {('BUY' if t['direction']==1 else 'SELL'):>4} "
              f"{t['ep']:>9.3f} {xt:>17} {t.get('exit_px', 0):>9.3f} "
              f"{t['a']:>5.2f} {t['q']:>5.2f} {t['pnl_R']:>+7.2f} {t.get('exit_reason', ''):>10}")
    print(f"\n[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
