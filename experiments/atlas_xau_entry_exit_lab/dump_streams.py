"""
NEW-STREAM VALIDATIONS (portfolio widening): edge_pullback + tt-trail on
  DJI M5, DJI M15  (new timeframe streams for the DOW products)
  BTC M5           (oracle_btc migration candidate)
Same honest harness: live feature pipeline on resampled bars, 6mo test windows,
3y rolling train, train-only thresholds, 1-slot, DEV 2020-07..2024-12 / HOLDOUT 2025+.
DJI spread charged in index points/ATR (1/1.5/2pt); BTC flat R (0.1/0.2/0.3).
Saves per-trade streams (entry/exit ts, netR, atr) for the risk-guard sims.
"""
import sys, pickle, time, json
from pathlib import Path
import numpy as np, pandas as pd
from numba import njit
from xgboost import XGBRegressor

SRV = Path("/home/jay/Desktop/my-agents-and-website/commercial/server")
sys.path.insert(0, str(SRV))
import decision_engine.edge_pullback as ep
OUT = Path(__file__).parent
FC = pickle.load(open(SRV / "decision_engine/models/hermes_dji_validated.pkl", "rb"))["feat_cols"]
t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
DEV_END = pd.Timestamp("2025-01-01"); COOLDOWN = 3

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
    busy = -1; m = len(order_idx); out = np.empty(m, np.int64); c = 0
    for t in range(m):
        k = order_idx[t]
        if ebar[k] <= busy: continue
        out[c] = k; busy = xit[k] + cd; c += 1
    return out[:c]

XAUPQ = "/home/jay/Desktop/new-model-zigzag/data/m1_xau_full.parquet"
DJIPQ = "/home/jay/Desktop/new-model-zigzag/data/m1_dji_full.parquet"
BTCPQ = "/home/jay/Desktop/new-model-zigzag/data/m1_btc_orderflow_8y.parquet"
BCOLS = ["time", "open", "high", "low", "close", "tick_volume"]
CFGS = [
    ("xau_m1",  XAUPQ, None, 1, 300, [(30, 0.75)], 11.0, [0.2], "pts", 5/7),
    ("xau_m5",  XAUPQ, None, 5, 60,  [(30, 0.75)], 5.0,  [0.2], "pts", 5/7),
    ("xau_m15", XAUPQ, None, 15, 20, [(15, 0.75)], 3.0,  [0.2], "pts", 5/7),
    ("dji_m1",  DJIPQ, None, 1, 300, [(30, 0.75)], 11.0, [1.5], "pts", 5/7),
    ("dji_m5",  DJIPQ, None, 5, 60,  [(15, 0.75)], 5.0,  [1.5], "pts", 5/7),
    ("dji_m15", DJIPQ, None, 15, 20, [(15, 0.75)], 3.0,  [1.5], "pts", 5/7),
    ("btc_m1",  BTCPQ, BCOLS, 1, 300, [(30, 0.75)], 11.0, [0.2], "R", 1.0),
    ("btc_m5",  BTCPQ, BCOLS, 5, 60,  [(30, 0.75)], 5.0,  [0.2], "R", 1.0),
]
XGB = dict(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.85,
           colsample_bytree=0.85, min_child_weight=10, n_jobs=-1, random_state=0)
rng = np.random.RandomState(0)
ALL = {}
for name, pq, cols, bm, MAXH, GRID, TPD, SPREADS, smode, dfrac in CFGS:
    log(f"=== {name} ===")
    m1 = pd.read_parquet(pq, columns=cols)
    m1 = m1.rename(columns={[c for c in m1.columns if "time" in c.lower()][0]: "time"})
    m1["time"] = pd.to_datetime(m1["time"]); m1 = m1.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    if "tick_volume" not in m1.columns: m1["tick_volume"] = m1.get("volume", 0)
    df = ep._resample(m1, bm); del m1
    feat = ep.compute_edge_features(df)
    atr = feat["atr14"].to_numpy(float); cdir = feat["committed_dir"].to_numpy(np.int64)
    da = np.abs(feat["dist_at_signal"].to_numpy(float))
    O = df["open"].to_numpy(float); H = df["high"].to_numpy(float); L = df["low"].to_numpy(float); C = df["close"].to_numpy(float)
    times = df["time"].values; n = len(df)
    ok = np.isfinite(atr) & (atr > 0) & (cdir != 0); ok[:300] = False; ok[-(MAXH + 2):] = False
    idx = np.where((da <= 1.0) & ok)[0]; dirs = cdir[idx].astype(np.int64)
    ct = pd.to_datetime(times[idx]); sig_atr = atr[idx]
    Xc = np.nan_to_num(feat[FC].to_numpy(np.float32)[idx], nan=0.0, posinf=0.0, neginf=0.0)
    del feat
    log(f"  {n:,} M{bm} bars, {len(idx):,} candidates, median ATR {np.median(sig_atr):.2f}")
    SIMS = {g: sim_tt(idx, dirs, O, H, L, C, atr, n, 7.0, 2.0, MAXH, *g) for g in GRID}
    base_pnl = SIMS[GRID[0]][0]
    WINS = []; tsw = pd.Timestamp("2020-07-01"); lastd = ct.max()
    while tsw < lastd:
        WINS.append((tsw - pd.DateOffset(years=3), tsw, tsw + pd.DateOffset(months=6))); tsw += pd.DateOffset(months=6)
    res = {g: [] for g in GRID}; streams = {g: [] for g in GRID}
    for tr_s, te_s, te_e in WINS:
        trm = (ct >= tr_s) & (ct < te_s); tem = (ct >= te_s) & (ct < te_e)
        if trm.sum() < 2000 or tem.sum() < 20: continue
        tix = np.where(trm)[0]; tix_f = tix if len(tix) <= 150_000 else rng.choice(tix, 150_000, replace=False)
        tr_days = max((ct[tix].max() - ct[tix].min()).days * dfrac, 1)
        m = XGBRegressor(**XGB); m.fit(Xc[tix_f], base_pnl[tix_f])
        p = m.predict(Xc).astype(np.float64)
        for g in GRID:
            pnl_, eb_, xit_ = SIMS[g]
            cand = np.quantile(p[tix], np.linspace(0.30, 0.97, 24)); thr = cand[-1]; best = 1e18
            for th in cand:
                kk = tix[p[tix] >= th]
                if len(kk) < 5: continue
                tk = take(kk[np.argsort(eb_[kk])].astype(np.int64), eb_, xit_, COOLDOWN)
                gap = abs(len(tk) / tr_days - TPD)
                if gap < best: best = gap; thr = th
            kk = np.where(tem & (p >= thr))[0]
            if len(kk) == 0: res[g].append(dict(win=str(te_s.date()), n=0)); continue
            tk = take(kk[np.argsort(eb_[kk])].astype(np.int64), eb_, xit_, COOLDOWN)
            R = pnl_[tk]; cost = (1.0 / sig_atr[tk]) if smode == "pts" else np.ones(len(tk))
            res[g].append(dict(win=str(te_s.date()), n=len(tk),
                               **{f"net{sp}": float((R - sp * cost).sum()) for sp in SPREADS}))
            hs = SPREADS[0]
            dd_ = dirs[tk]
            for et_, xt_, r_, a_, d_ in zip(times[eb_[tk]], times[xit_[tk]], R - hs * cost, sig_atr[tk], dd_):
                streams[g].append((str(et_), str(xt_), float(r_), float(a_), int(d_)))
        log(f"  window {te_s.date()} done")
    def agg(rows, dev):
        rr = [r for r in rows if r.get("n", 0) > 0 and ((pd.Timestamp(r["win"]) < DEV_END) == dev)]
        if not rr: return None
        d = dict(nwin=len(rr), n=sum(r["n"] for r in rr))
        for sp in SPREADS:
            d[f"net{sp}"] = sum(r[f"net{sp}"] for r in rr)
            d[f"w{sp}"] = sum(1 for r in rr if r[f"net{sp}"] > 0)
        return d
    print(f"\n===== {name} (spreads {SPREADS} {smode}) =====")
    for dev, lab_ in [(True, "DEV"), (False, "HOLDOUT")]:
        print(f"--- {lab_} ---")
        for g in GRID:
            a = agg(res[g], dev)
            if a is None: continue
            gs = "base" if g[0] == 0 else f"tt{g[0]}/{g[1]}"
            print(f"  {gs:<10} n={a['n']:>6} " + " ".join(
                f"net@{sp}:{a[f'net{sp}']:>+8.0f}({a[f'w{sp}']}/{a['nwin']})" for sp in SPREADS))
    ALL[name] = {"res": {str(g): res[g] for g in GRID},
                 "streams": {str(g): streams[g] for g in GRID}}
json.dump(ALL, open(OUT / "streams_all.json", "w"), default=str)
log("stream dump done")
