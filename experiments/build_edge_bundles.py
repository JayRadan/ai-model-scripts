"""
Build production bundles for the edge_pullback engine (hermes_dji + atlas_xau).
Trains on ALL local data + fresh Dukascopy tail, using the SAME feature pipeline
the live module uses (decision_engine.edge_pullback.compute_edge_features), so
training == serving. Calibrates threshold to ~11/day. Writes <name>_validated.pkl
(after backing up the existing one).
"""
import sys, pickle, time, shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np, pandas as pd
from numba import njit
from xgboost import XGBRegressor

SRV = Path("/home/jay/Desktop/my-agents-and-website/commercial/server")
sys.path.insert(0, str(SRV))
import decision_engine.edge_pullback as ep   # the LIVE feature pipeline
MODELS = SRV / "decision_engine" / "models"
FC = pickle.load(open(MODELS / "hermes_dji_validated.pkl", "rb"))["feat_cols"]  # 29 std feats
t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)

import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_IDX_AMERICA_E_D_J_IND, INSTRUMENT_FX_METALS_XAU_USD

@njit
def sim_trail(idxs, dirs, O, H, L, C, atr, SL, TRAIL, MAXH, n):
    m = len(idxs); pnl = np.empty(m); xit = np.empty(m, np.int64)
    for k in range(m):
        i = idxs[k]; d = dirs[k]; a = atr[i]; ei = i + 1
        if ei >= n or not (a > 0): pnl[k] = 0.0; xit[k] = min(i + 1, n - 1); continue
        ep_ = O[ei]; hard = SL * a; trd = TRAIL * a; mf = 0.0; end = min(ei + MAXH, n - 1); done = False
        for jx in range(ei, end + 1):
            fav = d * (C[jx] - ep_)
            if fav > mf: mf = fav
            if d == 1 and (ep_ - L[jx]) >= hard: pnl[k] = -SL; xit[k] = jx; done = True; break
            if d == -1 and (H[jx] - ep_) >= hard: pnl[k] = -SL; xit[k] = jx; done = True; break
            if mf >= trd and (mf - fav) >= trd: pnl[k] = (mf - trd) / a; xit[k] = jx; done = True; break
        if not done: pnl[k] = d * (C[end] - ep_) / a; xit[k] = end
    return pnl, xit

@njit
def take(order_idx, entry_bar, exit_bar, pnl, cd):
    busy = -1; m = len(order_idx); c = 0
    for t in range(m):
        k = order_idx[t]
        if entry_bar[k] <= busy: continue
        busy = exit_bar[k] + cd; c += 1
    return c

def fetch_tail(instr, start):
    end = datetime.now(timezone.utc)
    out = []; cur = pd.Timestamp(start)
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

def build(name, parquet, instr):
    log(f"=== {name} ===")
    df = pd.read_parquet(parquet).rename(columns={"timestamp": "time"})
    if "time" not in df.columns: df = df.rename(columns={[c for c in df.columns if "time" in c.lower()][0]: "time"})
    df["time"] = pd.to_datetime(df["time"]); df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    tail = fetch_tail(instr, df["time"].iloc[-1] - pd.Timedelta(days=2))
    if tail is not None:
        df = pd.concat([df, tail], ignore_index=True).sort_values("time").drop_duplicates("time").reset_index(drop=True)
    if "tick_volume" not in df.columns: df["tick_volume"] = df.get("volume", 0)
    log(f"  {len(df):,} bars, {df.time.iloc[0].date()} -> {df.time.iloc[-1].date()}")
    feat = ep.compute_edge_features(df)              # IDENTICAL to live serving
    atr = feat["atr14"].to_numpy(float); cdir = feat["committed_dir"].to_numpy(np.int64)
    da = np.abs(feat["dist_at_signal"].to_numpy(float))
    O = df["open"].to_numpy(float); H = df["high"].to_numpy(float); L = df["low"].to_numpy(float); C = df["close"].to_numpy(float)
    n = len(df)
    ok = np.isfinite(atr) & (atr > 0) & (cdir != 0); ok[:300] = False; ok[-301:] = False
    idx = np.where((da <= 1.0) & ok)[0]; dirs = cdir[idx].astype(np.int64); eb = idx + 1
    y, xit = sim_trail(idx, dirs, O, H, L, C, atr, 6.0, 2.0, 300, n)
    X = feat[FC].to_numpy(np.float32)[idx]
    log(f"  candidates {len(idx):,}, mean gross R {y.mean():+.3f}")
    m = XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.85,
                     colsample_bytree=0.85, min_child_weight=10, n_jobs=-1, random_state=0)
    m.fit(X, y)
    pall = m.predict(X)
    days = max((df.time.iloc[-1] - df.time.iloc[0]).days * 5 / 7, 1)
    thr = pall.max(); gap = 1e9
    for th in np.quantile(pall, np.linspace(0.3, 0.95, 30)):
        kk = np.where(pall >= th)[0]; order = kk[np.argsort(eb[kk])]
        taken = take(order.astype(np.int64), eb, xit, y, 5)
        g = abs(taken / days - 11.0)
        if g < gap: gap = g; thr = th
    payload = {"version": f"edge_pullback_v1_{name}", "q_model": m, "feat_cols": FC,
               "threshold": float(thr), "near_thr": 1.0, "sl_R": 6.0, "trail_R": 2.0,
               "be_r": 0.0, "maxh": 300, "trained_through": str(df.time.iloc[-1]),
               "n_candidates": int(len(idx)),
               "recipe": "pullback |dist_tfk|<=1.0, dir=committed_dir, XGB predict gross R "
                         "(SL6/trail2/maxhold300), take if pred>=threshold, 1-slot cooldown5"}
    out = MODELS / f"{name}_validated.pkl"
    bak = MODELS / f"{name}_validated.pkl.bak_pre_edge_pullback_2026-06-30"
    if out.exists() and not bak.exists(): shutil.copy(out, bak); log(f"  backed up -> {bak.name}")
    pickle.dump(payload, open(out, "wb"))
    log(f"  WROTE {out.name}  version={payload['version']} thr={thr:.3f}")
    return thr

build("hermes_dji", "/home/jay/Desktop/new-model-zigzag/data/m1_dji_full.parquet", INSTRUMENT_IDX_AMERICA_E_D_J_IND)
build("atlas_xau",  "/home/jay/Desktop/new-model-zigzag/data/m1_xau_full.parquet",  INSTRUMENT_FX_METALS_XAU_USD)
log("bundles built")
