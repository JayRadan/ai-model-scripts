TT_AFTER = 15
"""
Build production bundles for the hermes_btc + hermes_xau fix (2026-07-07).
  hermes_btc: edge_pullback tt30/0.75 on BTC M1 (validated run_lab_btc_edge.py,
              holdout 3/3 +3,493R @0.2R spread).
  hermes_xau: edge_pullback tt30/0.75 on XAU **M5** (validated run_lab_xau_m5.py,
              holdout 3/3 +1,371R @20c; bar_minutes=5 — engine resamples live M1).
Trains on ALL data + fresh Dukascopy tail with the LIVE feature pipeline
(training == serving). Thresholds calibrated on the tt-exit 1-slot portfolio.
Backs up existing bundles.
"""
import sys, pickle, time, shutil
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd
from numba import njit
from xgboost import XGBRegressor

SRV = Path("/home/jay/Desktop/my-agents-and-website/commercial/server")
sys.path.insert(0, str(SRV))
import decision_engine.edge_pullback as ep
MODELS = SRV / "decision_engine" / "models"
FC = pickle.load(open(MODELS / "hermes_dji_validated.pkl", "rb"))["feat_cols"]
t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)

import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_FX_METALS_XAU_USD, INSTRUMENT_VCCY_BTC_USD

@njit(cache=True)
def sim_tt(idxs, dirs, O, H, L, C, atr, n, SL, TRAIL, MAXH, ta, tt):
    m = len(idxs); pnl = np.zeros(m); ebar = np.full(m, -1, np.int64); xit = np.full(m, -1, np.int64)
    for k in range(m):
        i = idxs[k]; d = dirs[k]; a = atr[i]
        if i + 1 >= n or not (a > 0): continue
        st = i + 1; epr = O[st]; hard = SL * a; mf = 0.0
        end = min(st + MAXH, n - 1); done = False
        for jx in range(st, end + 1):
            adv = (epr - L[jx]) if d == 1 else (H[jx] - epr)
            if adv >= hard: pnl[k] = -SL; xit[k] = jx; done = True; break
            fav = d * (C[jx] - epr)
            if fav > mf: mf = fav
            trd = TRAIL * a
            if ta > 0 and (jx - st) >= ta:
                w = tt * a
                if w < trd: trd = w
            if mf >= trd and (mf - fav) >= trd: pnl[k] = (mf - trd) / a; xit[k] = jx; done = True; break
        if not done: pnl[k] = d * (C[end] - epr) / a; xit[k] = end
        ebar[k] = st
    return pnl, ebar, xit

@njit(cache=True)
def take(order_idx, ebar, xit, cd):
    busy = -1; m = len(order_idx); c = 0
    for t in range(m):
        k = order_idx[t]
        if ebar[k] <= busy: continue
        busy = xit[k] + cd; c += 1
    return c

def fetch_tail(instr, start):
    end = datetime.now(timezone.utc); out = []; cur = pd.Timestamp(start)
    while cur < pd.Timestamp(end).tz_localize(None):
        nxt = min(cur + pd.Timedelta(days=90), pd.Timestamp(end).tz_localize(None))
        try:
            r = dukascopy_python.fetch(instrument=instr, interval=dukascopy_python.INTERVAL_MIN_1,
                    offer_side=dukascopy_python.OFFER_SIDE_BID,
                    start=cur.to_pydatetime().replace(tzinfo=timezone.utc),
                    end=nxt.to_pydatetime().replace(tzinfo=timezone.utc))
            if r is not None and len(r): out.append(r)
        except Exception as e: log(f"  fetch {cur.date()} err {e}")
        cur = nxt
    if not out: return None
    d = pd.concat(out).reset_index().rename(columns={"timestamp": "time"})
    d["time"] = pd.to_datetime(d["time"]).dt.tz_convert("UTC").dt.tz_localize(None)
    return d

def build(name, parquet, instr, bm, maxh, target_per_day, cd, trading_days_frac, cols=None):
    log(f"=== {name} (bar_minutes={bm}) ===")
    df = pd.read_parquet(parquet, columns=cols).rename(columns={"timestamp": "time"})
    if "time" not in df.columns:
        df = df.rename(columns={[c for c in df.columns if "time" in c.lower()][0]: "time"})
    df["time"] = pd.to_datetime(df["time"]); df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    tail = fetch_tail(instr, df["time"].iloc[-1] - pd.Timedelta(days=2))
    if tail is not None:
        df = pd.concat([df, tail], ignore_index=True).sort_values("time").drop_duplicates("time").reset_index(drop=True)
    if "tick_volume" not in df.columns: df["tick_volume"] = df.get("volume", 0)
    log(f"  {len(df):,} M1 bars -> {df.time.iloc[-1]}")
    if bm > 1:
        df = ep._resample(df, bm)
        log(f"  resampled to {len(df):,} M{bm} bars")
    feat = ep.compute_edge_features(df)
    atr = feat["atr14"].to_numpy(float); cdir = feat["committed_dir"].to_numpy(np.int64)
    da = np.abs(feat["dist_at_signal"].to_numpy(float))
    O = df["open"].to_numpy(float); H = df["high"].to_numpy(float); L = df["low"].to_numpy(float); C = df["close"].to_numpy(float)
    n = len(df)
    ok = np.isfinite(atr) & (atr > 0) & (cdir != 0); ok[:300] = False; ok[-(maxh + 2):] = False
    idx = np.where((da <= 1.0) & ok)[0]; dirs = cdir[idx].astype(np.int64)
    y, ebar, xit_tt = sim_tt(idx, dirs, O, H, L, C, atr, n, 7.0, 2.0, maxh, 0, 0.0)      # base labels
    _, _, xit_live = sim_tt(idx, dirs, O, H, L, C, atr, n, 7.0, 2.0, maxh, TT_AFTER, 0.75)     # live exits for calibration
    X = np.nan_to_num(feat[FC].to_numpy(np.float32)[idx], nan=0.0, posinf=0.0, neginf=0.0)
    log(f"  candidates {len(idx):,}, mean gross R {y.mean():+.3f}")
    m = XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.85,
                     colsample_bytree=0.85, min_child_weight=10, n_jobs=-1, random_state=0)
    m.fit(X, y)
    pall = m.predict(X)
    days = max((df.time.iloc[-1] - df.time.iloc[0]).days * trading_days_frac, 1)
    thr = pall.max(); gap = 1e9
    for th in np.quantile(pall, np.linspace(0.3, 0.95, 30)):
        kk = np.where(pall >= th)[0]; order = kk[np.argsort(ebar[kk])]
        taken = take(order.astype(np.int64), ebar, xit_live, cd)
        g = abs(taken / days - target_per_day)
        if g < gap: gap = g; thr = th
    payload = {"version": f"edge_pullback_v3_tt{TT_AFTER}m{bm}_{name}",
               "q_model": m, "feat_cols": FC, "threshold": float(thr), "near_thr": 1.0,
               "sl_R": 7.0, "trail_R": 2.0, "be_r": 0.0, "maxh": maxh,
               "tight_after": TT_AFTER, "tight_trail_R": 0.75, "bar_minutes": bm,
               "trained_through": str(df.time.iloc[-1]), "n_candidates": int(len(idx)),
               "recipe": f"pullback |dist_tfk|<=1.0 dir=committed_dir on M{bm}, XGB gross R "
                         f"(SL7/trail2/maxh{maxh}), take pred>=thr, 1-slot; "
                         f"tt-trail 2->0.75*ATR after {TT_AFTER} M{bm} bars"}
    out = MODELS / f"{name}_validated.pkl"
    bak = MODELS / f"{name}_validated.pkl.bak_pre_edge_m15_2026-07-07"
    if out.exists() and not bak.exists(): shutil.copy(out, bak); log(f"  backed up -> {bak.name}")
    pickle.dump(payload, open(out, "wb"))
    log(f"  WROTE {out.name}  version={payload['version']} thr={thr:.3f}")


build("oracle_xau", "/home/jay/Desktop/new-model-zigzag/data/m1_xau_full.parquet",
      INSTRUMENT_FX_METALS_XAU_USD, bm=15, maxh=20, target_per_day=3.0, cd=3, trading_days_frac=5/7)
log("oracle_xau bundle built")
