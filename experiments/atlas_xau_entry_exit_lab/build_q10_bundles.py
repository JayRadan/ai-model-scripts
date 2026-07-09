"""
Build q10 production bundles (2026-07-09) — downside-quantile entry gate.
XGB objective reg:quantileerror alpha=0.10: rank entries by predicted 10th-pct
R of the LIVE tt-exit distribution instead of mean R. Validated per stream
(run_lab_slreduce.py + run_lab_q10_streams.py), dev-select + holdout-confirm:
  atlas_xau  XAU M1  dev +2170->+5287 (9/9)  holdout +2530->+3561
  hermes_dji DJI M1  dev +6853->+7094 (9/9)  holdout +2953->+3053
  hermes_btc BTC M1  dev +15835->+17861      holdout +3631->+6146
  hermes_xau XAU M5  dev tie (-1.7%)         holdout +1226->+1374, SL% down
  oracle_btc BTC M5  dev +5733->+5883        holdout +1372->+1566
oracle_xau (XAU M15) NOT swapped: q10 dev-worse (-12%) -> keeps mean gate.
Labels = tt-exit pnl (matches validation). Threshold calibrated on the same
tt 1-slot portfolio to the deployed trades/day target. Backs up existing pkls.
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
BAK = ".bak_pre_q10_2026-07-09"

import dukascopy_python
from dukascopy_python.instruments import (INSTRUMENT_FX_METALS_XAU_USD,
    INSTRUMENT_VCCY_BTC_USD, INSTRUMENT_IDX_AMERICA_E_D_J_IND)

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

def build(name, parquet, instr, bm, maxh, ta, tpd, cd, dfrac, cols=None):
    log(f"=== {name} (M{bm}, tt{ta}/0.75, target {tpd}/d) ===")
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
        df = ep._resample(df, bm); log(f"  resampled to {len(df):,} M{bm} bars")
    feat = ep.compute_edge_features(df)
    atr = feat["atr14"].to_numpy(float); cdir = feat["committed_dir"].to_numpy(np.int64)
    da = np.abs(feat["dist_at_signal"].to_numpy(float))
    O = df["open"].to_numpy(float); H = df["high"].to_numpy(float); L = df["low"].to_numpy(float); C = df["close"].to_numpy(float)
    n = len(df)
    ok = np.isfinite(atr) & (atr > 0) & (cdir != 0); ok[:300] = False; ok[-(maxh + 2):] = False
    idx = np.where((da <= 1.0) & ok)[0]; dirs = cdir[idx].astype(np.int64)
    y, ebar, xit = sim_tt(idx, dirs, O, H, L, C, atr, n, 7.0, 2.0, maxh, ta, 0.75)  # LIVE tt exits = labels
    X = np.nan_to_num(feat[FC].to_numpy(np.float32)[idx], nan=0.0, posinf=0.0, neginf=0.0)
    ct = pd.to_datetime(df["time"].values[idx])
    # EXACT validated recipe (run_lab_q10_streams.py): fit on the trailing 3y
    # subsampled to <=150k rows. A full-8y quantile fit degenerates (most leaves
    # contain >=10% SL-hitters -> predictions collapse to -7, gate loses all
    # selectivity — observed on first build attempt, thresholds -6.999).
    tr_s = ct.max() - pd.DateOffset(years=3)
    tix = np.where(ct >= tr_s)[0]
    rng = np.random.RandomState(0)
    tix_f = tix if len(tix) <= 150_000 else rng.choice(tix, 150_000, replace=False)
    log(f"  candidates {len(idx):,} (train window {len(tix):,} -> fit {len(tix_f):,}), mean tt R {y.mean():+.3f}")
    m = XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.85,
                     colsample_bytree=0.85, min_child_weight=10, n_jobs=-1, random_state=0,
                     objective="reg:quantileerror", quantile_alpha=0.10)
    m.fit(X[tix_f], y[tix_f])
    pall = m.predict(X).astype(np.float64)
    days = max((ct[tix].max() - ct[tix].min()).days * dfrac, 1)
    thr = pall.max(); gap = 1e9
    for th in np.quantile(pall[tix], np.linspace(0.30, 0.97, 24)):
        kk = tix[pall[tix] >= th]
        if len(kk) < 5: continue
        order = kk[np.argsort(ebar[kk])]
        taken = take(order.astype(np.int64), ebar, xit, cd)
        g = abs(taken / days - tpd)
        if g < gap: gap = g; thr = th
    pass_frac = float((pall[tix] >= thr).mean())
    kk = tix[pall[tix] >= thr]
    taken = take(kk[np.argsort(ebar[kk])].astype(np.int64), ebar, xit, cd)
    log(f"  thr {thr:+.4f}: pass_frac {pass_frac:.3f}, ~{taken / days:.1f} trades/day on train window")
    # M1 gates must be clearly selective; on M5 the 1-slot occupancy + cooldown
    # does most of the selection (lab q10 n was only ~8% above base there), so a
    # looser gate is the VALIDATED behaviour, not degeneracy.
    lim = 0.5 if bm == 1 else 0.85
    assert pass_frac < lim, f"{name}: degenerate gate (pass_frac {pass_frac:.2f} >= {lim}) — NOT shipping"
    ver = f"edge_pullback_v4_q10_tt{ta}{f'm{bm}' if bm > 1 else ''}_{name}"
    payload = {"version": ver, "q_model": m, "feat_cols": FC, "threshold": float(thr),
               "near_thr": 1.0, "sl_R": 7.0, "trail_R": 2.0, "be_r": 0.0, "maxh": maxh,
               "tight_after": ta, "tight_trail_R": 0.75, "bar_minutes": bm,
               "trained_through": str(df.time.iloc[-1]), "n_candidates": int(len(idx)),
               "recipe": f"pullback |dist_tfk|<=1.0 dir=committed_dir on M{bm}, XGB QUANTILE "
                         f"alpha=0.10 predicts 10th-pct tt-exit R (SL7/trail2/tt{ta}/0.75/"
                         f"maxh{maxh}), take pred>=thr, 1-slot"}
    out = MODELS / f"{name}_validated.pkl"
    bak = MODELS / f"{name}_validated.pkl{BAK}"
    if out.exists() and not bak.exists(): shutil.copy(out, bak); log(f"  backed up -> {bak.name}")
    pickle.dump(payload, open(out, "wb"))
    log(f"  WROTE {out.name}  version={ver} thr={thr:.4f}")

BTC_PQ = "/home/jay/Desktop/new-model-zigzag/data/m1_btc_orderflow_8y.parquet"
BTC_COLS = ["time", "open", "high", "low", "close", "tick_volume"]
XAU_PQ = "/home/jay/Desktop/new-model-zigzag/data/m1_xau_full.parquet"
DJI_PQ = "/home/jay/Desktop/new-model-zigzag/data/m1_dji_full.parquet"

build("atlas_xau",  XAU_PQ, INSTRUMENT_FX_METALS_XAU_USD,      bm=1, maxh=300, ta=30, tpd=11.0, cd=5, dfrac=5/7)
build("hermes_dji", DJI_PQ, INSTRUMENT_IDX_AMERICA_E_D_J_IND,  bm=1, maxh=300, ta=30, tpd=11.0, cd=5, dfrac=5/7)
build("hermes_btc", BTC_PQ, INSTRUMENT_VCCY_BTC_USD,           bm=1, maxh=300, ta=30, tpd=11.0, cd=5, dfrac=1.0, cols=BTC_COLS)
build("hermes_xau", XAU_PQ, INSTRUMENT_FX_METALS_XAU_USD,      bm=5, maxh=60,  ta=30, tpd=5.0,  cd=3, dfrac=5/7)
build("oracle_btc", BTC_PQ, INSTRUMENT_VCCY_BTC_USD,           bm=5, maxh=60,  ta=30, tpd=5.0,  cd=3, dfrac=1.0, cols=BTC_COLS)
log("q10 bundles built (oracle_xau untouched — q10 dev-worse there)")
