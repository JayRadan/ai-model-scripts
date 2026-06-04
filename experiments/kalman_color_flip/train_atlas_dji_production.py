"""
Atlas DJI — production bundle training.

Produces:
  /home/jay/Desktop/my-agents-and-website/commercial/server/decision_engine/models/atlas_dji_validated.pkl

Same frozen architecture as Atlas XAU but trained on M1 BTC/USD data:
  - Indicator stack: M1 Kalman regime + M1 Hermes TFK
  - Entry rule: STRICT candle confirmation + both-lines + MFE>=2R Q training
      BUY:  TFK GREEN + Kalman RED + prior strong-bear (body>=0.8xATR)
            closing BELOW both lines + current green + kf_age>=3
      SELL: mirror
  - Exit policy: SL=6xATR, TRAIL=2xATR, MAXH=300, BE @ +0.5R on new-signal
  - Multi-pos: 4 slots, switch_delta=0.5, cooldown=5
  - Q model: XGBRegressor trained ONLY on candidates whose forward MFE reached
             >= 2R, on the FULL parquet (production model, no holdout cutoff)
"""
from __future__ import annotations
import importlib.util, sys, time, pickle
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd
from numba import njit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "products/hermes_xau"))
from kalman import compute_kalman, bars_in_regime_array
from tfk import compute_tfk
_spec = importlib.util.spec_from_file_location("ofm1", ROOT / "products/_shared/m1_with_orderflow.py")
_ofm1 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_ofm1)
add_standard_features = _ofm1.add_standard_features
FLOW_FEATS = list(_ofm1.FLOW_FEATS)

SPREAD = 0.30   # ATR units — same as XAU/Hermes; 0.30 ATR ≈ tight ECN BTC spread
SL, TRAIL, MAXH = 6.0, 2.0, 300
BE_R = 0.5; MAX_CONC, SWITCH, COOLDOWN = 4, 0.5, 5
STRONG_ATR = 0.8; RMULT = 1.0; MFE_FILTER = 2.0
Q_THR = 1.0
HOLDOUT_CUTOFF = pd.Timestamp("2025-09-01 00:00:00")
KF_FEATS = ["f_velPct","f_velSignif","f_innovZ","f_volState","f_accel","f_velRaw"]
TFK_FEATS = ["force","velocity","x_est","regime_w","trend_raw","trend","committed_dir"]
STD = ["rsi14","dist_ema20","dist_ema50","dist_ema100","dist_ema200","slope5","slope10","slope20","atr_ratio",
       "m5_rsi14","m5_slope5","m5_ema50_dist","m15_rsi14","m15_slope5","m15_ema50_dist","h1_rsi14","h1_slope5","h1_ema50_dist"]
EXTRA = ["dist_kf","dist_tfk","kf_regime_age","vel_up_streak","bar_range_atr","kf_dir","body_atr","strong_bear_prev","strong_bull_prev"]

DATA = ROOT / "data/m1_dji_orderflow.parquet"
OUT = Path("/home/jay/Desktop/my-agents-and-website/commercial/server/decision_engine/models/atlas_dji_validated.pkl")


@njit(cache=True)
def labels_with_mfe(idxs, dirs, O, H, L, C, atr, sp, SL, TRAIL, MAXH, n):
    m = len(idxs); pnl = np.empty(m); xit = np.empty(m, np.int64); mfe = np.empty(m)
    for k in range(m):
        i = idxs[k]; d = dirs[k]; a = atr[i]; ei = i + 1
        if ei >= n or not (a > 0):
            pnl[k] = 0.0; xit[k] = ei; mfe[k] = 0.0; continue
        ep = O[ei]; hard = SL * a; trd = TRAIL * a; mf = 0.0
        end = min(ei + MAXH, n - 1); done = False
        for j in range(ei, end + 1):
            fav = d * (C[j] - ep)
            if fav > mf: mf = fav
            if d == 1 and (ep - L[j]) >= hard: pnl[k] = -SL - sp; xit[k] = j; done = True; break
            if d == -1 and (H[j] - ep) >= hard: pnl[k] = -SL - sp; xit[k] = j; done = True; break
            if mf >= trd and (mf - fav) >= trd: pnl[k] = (mf - trd) / a - sp; xit[k] = j; done = True; break
        if not done: pnl[k] = d * (C[end] - ep) / a - sp; xit[k] = end
        mfe[k] = mf / a
    return pnl, xit, mfe


def streak_up(v):
    n = len(v); o = np.zeros(n, np.int32); c = 0
    for i in range(1, n):
        c = c + 1 if v[i] > v[i - 1] else 0; o[i] = c
    return o


def main():
    print("="*72); print("  Atlas DJI production bundle"); print("="*72)
    t0 = time.time()
    m1 = pd.read_parquet(DATA).sort_values("time").reset_index(drop=True)
    print(f"  loaded {len(m1):,} bars from {m1.time.iloc[0]} -> {m1.time.iloc[-1]}", flush=True)

    print("  computing Kalman + Hermes TFK + standard features ...", flush=True)
    tfk_df = compute_tfk(m1)
    kf = compute_kalman(m1, r_mult=RMULT)
    df = add_standard_features(kf)
    for c in TFK_FEATS: df[c] = tfk_df[c].to_numpy()
    df["tfk_line"] = tfk_df["tfk_line"].to_numpy()
    O = m1["open"].to_numpy(float); H = m1["high"].to_numpy(float)
    L = m1["low"].to_numpy(float); C = m1["close"].to_numpy(float)
    atr = df["atr14"].to_numpy(float)
    kdir = df["kf_dir"].to_numpy(np.int64); kline = df["kf_p"].to_numpy(float)
    cdir = tfk_df["committed_dir"].to_numpy(np.int64); tline = tfk_df["tfk_line"].to_numpy(float)
    vel = df["f_velRaw"].to_numpy(float)
    kage = bars_in_regime_array(kdir); df["kf_regime_age"] = kage
    df["bar_range_atr"] = (H - L) / np.maximum(atr, 1e-9)
    df["dist_kf"] = np.where(atr > 0, (C - kline) / atr, 0.0)
    df["dist_tfk"] = np.where(atr > 0, (C - tline) / atr, 0.0)
    df["vel_up_streak"] = streak_up(vel)
    body = C - O; body_atr = np.where(atr > 0, np.abs(body) / atr, 0.0); df["body_atr"] = body_atr
    strong_bear = (body < 0) & (body_atr >= STRONG_ATR)
    strong_bull = (body > 0) & (body_atr >= STRONG_ATR)
    df["strong_bear_prev"] = np.concatenate([[False], strong_bear[:-1]])
    df["strong_bull_prev"] = np.concatenate([[False], strong_bull[:-1]])
    n = len(df); sp = SPREAD / np.nanmedian(atr)

    pbk = np.concatenate([[False], C[:-1] < kline[:-1]])
    pak = np.concatenate([[False], C[:-1] > kline[:-1]])
    pbt = np.concatenate([[False], C[:-1] < tline[:-1]])
    pat = np.concatenate([[False], C[:-1] > tline[:-1]])
    g = C > O; r = C < O
    ok = np.isfinite(atr) & (atr > 0); ok[:250] = False; ok[-(MAXH + 1):] = False
    buy  = ok & (cdir == 1)  & (kdir == -1) & df["strong_bear_prev"].to_numpy() & pbk & pbt & g & (kage >= 3)
    sell = ok & (cdir == -1) & (kdir == 1)  & df["strong_bull_prev"].to_numpy() & pak & pat & r & (kage >= 3)
    mask = buy | sell
    idxs = np.where(mask)[0]; dirs = np.where(cdir[idxs] == 1, 1, -1).astype(np.int64)
    print(f"  candidates: {len(idxs):,}  (buys {(dirs == 1).sum():,}  sells {(dirs == -1).sum():,})", flush=True)

    print("  computing labels (forward pnl_R + MFE) ...", flush=True)
    pnl, xit, mfe = labels_with_mfe(idxs, dirs, O, H, L, C, atr, sp, SL, TRAIL, MAXH, n)
    mfe_2r_mask = mfe >= MFE_FILTER
    print(f"  MFE>=2R candidates (production training set): {mfe_2r_mask.sum():,} ({mfe_2r_mask.mean()*100:.1f}%)", flush=True)

    feat_cols = [c for c in dict.fromkeys(EXTRA + KF_FEATS + TFK_FEATS + STD + FLOW_FEATS) if c in df.columns]
    print(f"  features: {len(feat_cols)}", flush=True)
    X_train = df.iloc[idxs[mfe_2r_mask]][feat_cols].fillna(0).to_numpy(np.float32)
    y_train = pnl[mfe_2r_mask].astype(np.float32)
    print("  training XGBRegressor (production, full data, no cutoff) ...", flush=True)
    from xgboost import XGBRegressor
    M = XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.04, subsample=0.85,
                     colsample_bytree=0.85, min_child_weight=8, reg_lambda=1.0,
                     objective="reg:squarederror", tree_method="hist", random_state=42, verbosity=0)
    M.fit(X_train, y_train)

    # Holdout model for validation
    times = m1["time"].to_numpy(); tmask = times < np.datetime64(HOLDOUT_CUTOFF)
    tr_m = tmask[idxs]; te_m = ~tr_m
    hold_train_mask = tr_m & mfe_2r_mask
    M_hold = XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.04, subsample=0.85,
                          colsample_bytree=0.85, min_child_weight=8, reg_lambda=1.0,
                          objective="reg:squarederror", tree_method="hist", random_state=42, verbosity=0)
    M_hold.fit(df.iloc[idxs[hold_train_mask]][feat_cols].fillna(0).to_numpy(np.float32),
               pnl[hold_train_mask].astype(np.float32))
    q_te = M_hold.predict(df.iloc[idxs[te_m]][feat_cols].fillna(0).to_numpy(np.float32))

    test_pnl = pnl[te_m]
    print("\n  ── Holdout (post 2025-09-01) Q sweep ──", flush=True)
    holdout_results = {}
    print(f"  {'Q':>5} {'n':>6} {'WR%':>6} {'PF':>6} {'sumR':>8}")
    for q in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
        qm = q_te >= q
        if qm.sum() == 0:
            print(f"  {q:>5.1f} {0:>6}"); continue
        rs = test_pnl[qm]; rs = rs[np.isfinite(rs)]
        if len(rs) == 0: continue
        w, l = rs[rs > 0], rs[rs <= 0]
        pf = float(w.sum() / max(-l.sum(), 1e-9))
        wr = float((rs > 0).mean())
        sumR = float(rs.sum())
        print(f"  {q:>5.1f} {len(rs):>6,} {wr*100:>5.1f}% {pf:>6.2f} {sumR:>+8.1f}")
        holdout_results[str(q)] = {"n": int(len(rs)), "wr": wr, "pf": pf, "sum_r": sumR}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "q_model": M,
        "q_model_holdout": M_hold,
        "feat_cols": feat_cols,
        "kalman_params": {"q": 0.05, "r_mult": RMULT, "r_len": 50, "dt": 1.0, "mintick": 0.01},
        "tfk_params": {"flow_len": 20, "damping": 0.93, "mass": 1.0, "q_noise": 0.001, "r_noise": 0.10,
                       "hurst_len": 50, "trend_thresh": 0.30, "atr_len": 50, "slope_k": 0.15,
                       "anchor_k": 0.02, "flip_band": 0.30, "flip_bars": 3, "color_confirm": 5,
                       "tie_break": "vwap", "htf_ema_n": 50, "htf_bars": 12},
        "atlas_params": {"strong_atr": STRONG_ATR, "mfe_filter": MFE_FILTER,
                         "kf_age_min": 3, "q_thr": Q_THR, "sl_atr": SL, "trail_atr": TRAIL,
                         "max_hold": MAXH, "be_trigger_r": BE_R, "max_conc": MAX_CONC,
                         "switch_delta": SWITCH, "cooldown_bars": COOLDOWN},
        "train_meta": {
            "trained_on": datetime.now(timezone.utc).isoformat(),
            "instrument": "DJIUSD",
            "data_file": str(DATA),
            "n_candidates_total": int(len(idxs)),
            "n_train_production": int(mfe_2r_mask.sum()),
            "n_train_holdout": int(hold_train_mask.sum()),
            "n_test_holdout": int(te_m.sum()),
            "holdout_cutoff": str(HOLDOUT_CUTOFF),
            "spread_R_train": float(sp),
            "holdout_q_sweep": holdout_results,
            "strategy": "Atlas STRICT candle confirm + both-lines + MFE>=2R + multi-pos + BE@+0.5R (DJI)",
            "config_snapshot": {
                "STRONG_ATR": STRONG_ATR, "MFE_FILTER": MFE_FILTER,
                "SL": SL, "TRAIL": TRAIL, "MAX_HOLD": MAXH, "BE_R": BE_R,
                "MAX_CONC": MAX_CONC, "SWITCH": SWITCH, "COOLDOWN": COOLDOWN,
                "Q_THR": Q_THR, "SPREAD_R": SPREAD,
                "RMULT": RMULT,
            },
        },
    }
    with open(OUT, "wb") as f: pickle.dump(payload, f)
    print(f"\n  wrote {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")
    print(f"  done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
