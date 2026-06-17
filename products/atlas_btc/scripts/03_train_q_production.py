"""
Atlas BTC — PRODUCTION training script (M15-macro + M1 U-shape, 2026-06-17 deploy).

ARCHITECTURE CHANGE: replaces the prior STRICT-2-bar-reversal-candle architecture
(same change applied to Atlas XAU at commit 7bf5a37, deployed 2026-06-17).

Rule (mirrors deployed Atlas XAU ushape_m15):
  BUY  iff M15 Kalman GREEN AND M1 Kalman RED AND M1 kf_v<0 AND M1 f_accel>0
                              AND edge-bar (one signal per U episode).
  SELL: mirror.

Label: forward R with SL=6×ATR / TRAIL=1.0×ATR / MAX_HOLD=300 bars.

Holdout (2025-09-01 → 2026-05-01, 8mo Dukascopy):
  Multi-pos Q≥1.5: ~6 trd/day | PF 1.58 | DD 57R
  vs deployed strict_candle baseline: PF 1.31 | DD ~95R
  → +21% PF, −40% DD.

3-day live sim (06-14..06-17):
  OLD strict_candle: 47 trades, -$2,710
  NEW ushape_m15:    37 trades, +$2,044  (NEW wins by $4,754)

14-day live sim (06-04..06-17):
  75 trades · 8 green days vs 5 red days · positive expected value

Backups:
  - products/atlas_btc/scripts/03_train_q_production.py.bak_pre_m15_ushape_2026-06-17
  - server/decision_engine/models/atlas_btc_validated.pkl.bak_pre_m15_ushape_2026-06-17
"""
from __future__ import annotations
import importlib.util, sys, time, pickle
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "products/hermes_btc"))
sys.path.insert(0, str(ROOT / "experiments/kalman_color_flip"))
from tfk import compute_tfk
from kalman import compute_kalman
_spec = importlib.util.spec_from_file_location("ofm1", ROOT / "products/_shared/m1_with_orderflow.py")
_ofm1 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_ofm1)
add_standard_features = _ofm1.add_standard_features

OUT = Path("/home/jay/Desktop/my-agents-and-website/commercial/server/decision_engine/models/atlas_btc_validated.pkl")
DATA_PARQUET = ROOT / "data" / "m1_btc_orderflow.parquet"
CUTOFF = pd.Timestamp("2025-09-01 00:00:00")

SL = 6.0
TRAIL = 1.0
MAX_HOLD = 300
SPREAD_USD = 5.0
Q_THR = 1.5

KAL = dict(q=0.05, r_mult=1.0, r_len=50, dt=1.0, mintick=0.01)

FEATS = [
    "atr14","atr_ratio","rsi14","dist_ema20","dist_ema50","dist_ema100","dist_ema200",
    "slope5","slope10","slope20",
    "m5_rsi14","m5_slope5","m5_ema50_dist",
    "m15_rsi14","m15_slope5","m15_ema50_dist",
    "h1_rsi14","h1_slope5","h1_ema50_dist",
    "kf_p","kf_v","kf_dir","kf_innov","kf_S","kf_P11","kf_R",
    "f_velPct","f_velSignif","f_innovZ","f_volState","f_accel","f_velRaw",
    "kf_p_m15","kf_dir_m15","kf_v_m15","f_accel_m15","f_velPct_m15",
]


def simulate_label(entry_idx, direction, C, H, L, O, a, spread_R):
    n = len(C)
    if entry_idx >= n-1 or not (np.isfinite(a) and a > 0): return np.nan
    ep = O[entry_idx]
    hard = SL*a; trail_d = TRAIL*a; max_favor = 0.0
    end = min(entry_idx + MAX_HOLD, n-1)
    for k in range(entry_idx, end+1):
        favor_now = direction*(C[k] - ep)
        if favor_now > max_favor: max_favor = favor_now
        if direction == 1:
            if (ep - L[k]) >= hard: return -SL - spread_R
        else:
            if (H[k] - ep) >= hard: return -SL - spread_R
        if max_favor >= trail_d:
            if (max_favor - favor_now) >= trail_d:
                return (max_favor - trail_d)/a - spread_R
    return direction*(C[end]-ep)/a - spread_R


def main():
    t0 = time.time()
    print("="*72); print("  Atlas BTC PRODUCTION — M15 U-shape (2026-06-17 deploy)"); print("="*72)

    print("\n[1/5] loading + features ...", flush=True)
    m1 = pd.read_parquet(DATA_PARQUET).sort_values("time").reset_index(drop=True)
    if "tick_volume" not in m1.columns: m1["tick_volume"] = 1.0
    df = compute_tfk(m1)
    df = add_standard_features(df)
    df = compute_kalman(df, **KAL)
    print(f"  bars: {len(df):,}  ({time.time()-t0:.0f}s)")

    print("\n[2/5] M15 Kalman (causal forward-fill to M1) ...", flush=True)
    g15 = df.set_index("time")[["open","high","low","close","tick_volume"]].resample("15min").agg(
        {"open":"first","high":"max","low":"min","close":"last","tick_volume":"sum"}).dropna().reset_index()
    g15 = compute_kalman(g15, **KAL)
    g15["end"] = g15["time"] + pd.Timedelta(minutes=15)
    j = np.searchsorted(g15["end"].values, df["time"].values, side="right") - 1
    valid_j = j >= 0
    jj = np.clip(j, 0, len(g15)-1)
    df["kf_p_m15"]      = np.where(valid_j, g15["kf_p"].to_numpy()[jj], np.nan)
    df["kf_dir_m15"]    = np.where(valid_j, g15["kf_dir"].to_numpy()[jj], 0).astype(np.int64)
    df["kf_v_m15"]      = np.where(valid_j, g15["kf_v"].to_numpy()[jj], 0.0)
    df["f_accel_m15"]   = np.where(valid_j, g15["f_accel"].to_numpy()[jj], 0.0)
    df["f_velPct_m15"]  = np.where(valid_j, g15["f_velPct"].to_numpy()[jj], 0.0)
    print(f"  M15 bars: {len(g15):,}  ({time.time()-t0:.0f}s)")

    O = df["open"].to_numpy(np.float64); H = df["high"].to_numpy(np.float64)
    L = df["low"].to_numpy(np.float64); C = df["close"].to_numpy(np.float64)
    times = df["time"].to_numpy()
    atr = df["atr14"].to_numpy(np.float64)
    spread_R = SPREAD_USD / np.nanmedian(atr)

    kd15 = df["kf_dir_m15"].to_numpy(np.int64)
    kd1  = df["kf_dir"].to_numpy(np.int64)
    kv1  = df["kf_v"].to_numpy(np.float64)
    fa1  = df["f_accel"].to_numpy(np.float64)
    kp15 = df["kf_p_m15"].to_numpy(np.float64)
    dist_m15 = np.where(atr > 0, (C - kp15)/atr, np.nan)

    print("\n[3/5] U-shape edge-detected candidate set ...", flush=True)
    buy_raw  = (kd15 == +1) & (kd1 == -1) & (kv1 < 0) & (fa1 > 0)
    sell_raw = (kd15 == -1) & (kd1 == +1) & (kv1 > 0) & (fa1 < 0)
    buy_edge  = buy_raw  & ~np.concatenate([[False], buy_raw[:-1]])
    sell_edge = sell_raw & ~np.concatenate([[False], sell_raw[:-1]])
    kv_s = pd.Series(kv1)
    kv_min50 = kv_s.rolling(50, min_periods=20).min().fillna(0).to_numpy()
    kv_max50 = kv_s.rolling(50, min_periods=20).max().fillna(0).to_numpy()
    kv_pos = (kv1 - kv_min50) / np.maximum(kv_max50 - kv_min50, 1e-9)
    valid = np.isfinite(atr) & (atr > 0)
    valid[:500] = False; valid[-(MAX_HOLD+1):] = False
    mask = (buy_edge | sell_edge) & valid
    idxs = np.where(mask)[0]
    dirs = np.where(buy_edge[idxs], +1, -1).astype(np.int64)
    print(f"  candidates: {len(idxs):,}  (buys={int((dirs==+1).sum()):,}  sells={int((dirs==-1).sum()):,})")

    print(f"\n[4/5] labels + train (cutoff {CUTOFF.date()}) ...", flush=True)
    pnl = np.zeros(len(idxs), dtype=np.float32)
    for k, i in enumerate(idxs):
        pnl[k] = simulate_label(i+1, int(dirs[k]), C, H, L, O, atr[i], spread_R)
    pnl = np.where(np.isfinite(pnl), pnl, 0.0)
    extras = pd.DataFrame({
        "dist_m15kf": dist_m15[idxs],
        "kv_pos_50": kv_pos[idxs],
        "bar_range_atr": (H[idxs]-L[idxs])/np.maximum(atr[idxs], 1e-9),
    })
    feats = df.iloc[idxs][[c for c in FEATS if c in df.columns]].fillna(0).reset_index(drop=True)
    X = pd.concat([extras, feats], axis=1)
    feat_cols = list(X.columns)
    X_np = X.to_numpy(np.float32)
    train_m = times[idxs] < np.datetime64(CUTOFF)
    test_m = ~train_m
    print(f"  feats: {len(feat_cols)}  train={int(train_m.sum()):,}  test={int(test_m.sum()):,}")

    from xgboost import XGBRegressor
    common = dict(n_estimators=600, max_depth=5, learning_rate=0.04,
                  subsample=0.85, colsample_bytree=0.85, min_child_weight=10,
                  reg_lambda=1.0, objective="reg:squarederror", tree_method="hist",
                  random_state=42, verbosity=0)
    M = XGBRegressor(**common); M.fit(X_np, pnl)
    M_hold = XGBRegressor(**common); M_hold.fit(X_np[train_m], pnl[train_m])
    q_te = M_hold.predict(X_np[test_m]); pnl_te = pnl[test_m]
    qm = q_te >= Q_THR
    if qm.sum() > 0:
        rs = pnl_te[qm]; rs = rs[np.isfinite(rs)]
        w, l = rs[rs>0], rs[rs<=0]
        pf = float(w.sum()/max(-l.sum(), 1e-9))
        wr = float((rs>0).mean()); sumR = float(rs.sum())
        print(f"  HOLDOUT @ Q≥{Q_THR}: n={len(rs):,} WR={wr*100:.1f}% PF={pf:.2f} sumR={sumR:+.0f}")

    print("\n[5/5] write bundle ...", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "q_model": M,
        "q_model_holdout": M_hold,
        "feat_cols": feat_cols,
        "kalman_params": KAL,
        "m15_kalman_params": KAL,
        "atlas_params": {
            "entry_mode": "ushape_m15",
            "macro_tf_min": 15,
            "q_thr": Q_THR,
            "sl_atr": SL, "trail_atr": TRAIL, "max_hold": MAX_HOLD,
            "max_conc": 4, "switch_delta": 0.5, "cooldown_bars": 5, "be_trigger_r": 0.5,
        },
        "train_meta": {
            "trained_on": datetime.now(timezone.utc).isoformat(),
            "architecture": "m15_macro_kalman + m1_ushape_edge",
            "n_candidates_total": int(len(idxs)),
            "n_train_production": int(len(idxs)),
            "holdout_cutoff": str(CUTOFF),
        },
    }
    with open(OUT, "wb") as f:
        pickle.dump(payload, f)
    print(f"  wrote {OUT}  ({OUT.stat().st_size//1024} KB)")
    print(f"  done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
