"""
Q10 STREAM VALIDATION — the downside-quantile gate (XGB reg:quantileerror
alpha=0.10, rank entries by predicted 10th-pct R instead of mean R) beat the
deployed mean gate on XAU M1 (dev +2170->+5287 9/9, holdout +2530->+3561) and
DJI M1 (dev +6853->+7094, holdout +2953->+3053, SL% down) in run_lab_slreduce.py.
Before swapping the other four production bundles, validate q10 vs base on each
deployed stream with its EXACT live config:
  hermes_btc  BTC M1  maxh300 tt30/0.75 target 11/d cd5  spread flat 0.2R
  hermes_xau  XAU M5  maxh60  tt30/0.75 target  5/d cd3  spread $0.20/ATR
  oracle_xau  XAU M15 maxh20  tt15/0.75 target  3/d cd3  spread $0.20/ATR
  oracle_btc  BTC M5  maxh60  tt30/0.75 target  5/d cd3  spread flat 0.2R
Same honest harness: 3y train / 6mo test, train-only thresholds, DEV<2025 / HOLDOUT.
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
DEV_END = pd.Timestamp("2025-01-01")

@njit(cache=True)
def sim_tt(idxs, dirs, O, H, L, C, atr, n, SL, TRAIL, MAXH, ta, tt):
    m = len(idxs); pnl = np.zeros(m); ebar = np.full(m, -1, np.int64)
    xit = np.full(m, -1, np.int64); code = np.full(m, -1, np.int64)
    for k in range(m):
        i = idxs[k]; d = dirs[k]; a = atr[i]
        if i + 1 >= n or not (a > 0): continue
        st = i + 1; epr = O[st]; hard = SL * a; mf = 0.0
        end = min(st + MAXH, n - 1); done = False
        for jx in range(st, end + 1):
            adv = (epr - L[jx]) if d == 1 else (H[jx] - epr)
            if adv >= hard: pnl[k] = -SL; xit[k] = jx; code[k] = 0; done = True; break
            fav = d * (C[jx] - epr)
            if fav > mf: mf = fav
            trd = TRAIL * a
            if ta > 0 and (jx - st) >= ta:
                w = tt * a
                if w < trd: trd = w
            if mf >= trd and (mf - fav) >= trd: pnl[k] = (mf - trd) / a; xit[k] = jx; code[k] = 1; done = True; break
        if not done: pnl[k] = d * (C[end] - epr) / a; xit[k] = end; code[k] = 2
        ebar[k] = st
    return pnl, ebar, xit, code

@njit(cache=True)
def take(order_idx, ebar, xit, cd):
    busy = -1; m = len(order_idx); out = np.empty(m, np.int64); c = 0
    for t in range(m):
        k = order_idx[t]
        if ebar[k] <= busy: continue
        out[c] = k; busy = xit[k] + cd; c += 1
    return out[:c]

XGBM = dict(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.85,
            colsample_bytree=0.85, min_child_weight=10, n_jobs=-1, random_state=0)
XGBQ = dict(XGBM, objective="reg:quantileerror", quantile_alpha=0.10)
rng = np.random.RandomState(0)
ALL = {}

CFGS = [
    ("hermes_btc", "/home/jay/Desktop/new-model-zigzag/data/m1_btc_orderflow_8y.parquet",
     ["time", "open", "high", "low", "close", "tick_volume"], 1, 300, (30, 0.75), 11.0, 5, 1.0, ("R", 0.2)),
    ("hermes_xau", "/home/jay/Desktop/new-model-zigzag/data/m1_xau_full.parquet",
     None, 5, 60, (30, 0.75), 5.0, 3, 5 / 7, ("usd", 0.20)),
    ("oracle_xau", "/home/jay/Desktop/new-model-zigzag/data/m1_xau_full.parquet",
     None, 15, 20, (15, 0.75), 3.0, 3, 5 / 7, ("usd", 0.20)),
    ("oracle_btc", "/home/jay/Desktop/new-model-zigzag/data/m1_btc_orderflow_8y.parquet",
     ["time", "open", "high", "low", "close", "tick_volume"], 5, 60, (30, 0.75), 5.0, 3, 1.0, ("R", 0.2)),
]

for name, pq, cols, bm, MAXH, (ta, tt), TPD, CD, dfrac, (smode, sp) in CFGS:
    log(f"=== {name} (M{bm}) ===")
    m1 = pd.read_parquet(pq, columns=cols)
    m1 = m1.rename(columns={[c for c in m1.columns if "time" in c.lower()][0]: "time"})
    m1["time"] = pd.to_datetime(m1["time"]); m1 = m1.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    if "tick_volume" not in m1.columns: m1["tick_volume"] = m1.get("volume", 0)
    df = ep._resample(m1, bm) if bm > 1 else m1
    if bm > 1: del m1
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
    pnl, eb, xt, code = sim_tt(idx, dirs, O, H, L, C, atr, n, 7.0, 2.0, MAXH, ta, tt)
    log(f"  {n:,} M{bm} bars, {len(idx):,} candidates")

    WINS = []; tsw = pd.Timestamp("2020-07-01"); lastd = ct.max()
    while tsw < lastd:
        WINS.append((tsw - pd.DateOffset(years=3), tsw, tsw + pd.DateOffset(months=6))); tsw += pd.DateOffset(months=6)
    res = {"base": [], "q10": []}
    for tr_s, te_s, te_e in WINS:
        trm = (ct >= tr_s) & (ct < te_s); tem = (ct >= te_s) & (ct < te_e)
        if trm.sum() < 2000 or tem.sum() < 20: continue
        tix = np.where(trm)[0]; tix_f = tix if len(tix) <= 150_000 else rng.choice(tix, 150_000, replace=False)
        tr_days = max((ct[tix].max() - ct[tix].min()).days * dfrac, 1)
        for vn, kw in [("base", XGBM), ("q10", XGBQ)]:
            m = XGBRegressor(**kw); m.fit(Xc[tix_f], pnl[tix_f])
            p = m.predict(Xc).astype(np.float64)
            cand = np.quantile(p[tix], np.linspace(0.30, 0.97, 24)); thr = cand[-1]; best = 1e18
            for th in cand:
                kk = tix[p[tix] >= th]
                if len(kk) < 5: continue
                tk = take(kk[np.argsort(eb[kk])].astype(np.int64), eb, xt, CD)
                gap = abs(len(tk) / tr_days - TPD)
                if gap < best: best = gap; thr = th
            kk = np.where(tem & (p >= thr))[0]
            tk = take(kk[np.argsort(eb[kk])].astype(np.int64), eb, xt, CD)
            cost = (sp / sig_atr[tk]) if smode == "usd" else np.full(len(tk), sp)
            netv = pnl[tk] - cost
            res[vn].append(dict(win=str(te_s.date()), n=len(tk), net=float(netv.sum()),
                                sl=int((code[tk] == 0).sum()),
                                wr=float((netv > 0).mean()) if len(tk) else 0.0))
        log(f"  window {te_s.date()} done")

    def agg(rows, dev):
        rr = [r for r in rows if (pd.Timestamp(r["win"]) < DEV_END) == dev and r["n"] > 0]
        if not rr: return None
        nn = sum(r["n"] for r in rr)
        return dict(nwin=len(rr), n=nn, net=sum(r["net"] for r in rr),
                    w=sum(1 for r in rr if r["net"] > 0),
                    slr=100.0 * sum(r["sl"] for r in rr) / nn,
                    wr=100.0 * np.average([r["wr"] for r in rr], weights=[r["n"] for r in rr]))
    print(f"\n===== {name} — q10 vs base (net @ {sp}{smode}) =====")
    for dev, lab in [(True, "DEV"), (False, "HOLDOUT")]:
        print(f"--- {lab} ---")
        for vn in ("base", "q10"):
            a = agg(res[vn], dev)
            if a is None: continue
            print(f"  {vn:<6} n={a['n']:>6}  net {a['net']:>+8.0f}  w+ {a['w']}/{a['nwin']}"
                  f"  SLhit {a['slr']:.2f}%  WR {a['wr']:.1f}%")
    ALL[name] = res

json.dump(ALL, open(OUT / "q10_streams_results.json", "w"), default=str, indent=1)
log("q10 stream validation done")
