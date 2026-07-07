"""
HERMES XAU FIX — edge_pullback + tt-trail on XAU **M5** bars.
Context: deployed hermes_xau bundle (2026-05-25) predates the causal-HTF fix and
honest 8y revalidation said the old recipe loses (PF 0.82-0.85). The validated
edge_pullback recipe IS live on XAU M1 (atlas_xau) — hermes_xau cloning it would
double exposure on identical signals. This lab tests the SAME recipe on M5:
different timeframe -> different trades (true diversification) + bigger ATR ->
smaller spread-in-R (XAU's weak point).

Harness: identical protocol — live feature pipeline on M5 resampled bars, 6mo
test windows, 3y rolling train, train-only thr (~5/day target on M5), 1-slot cd3,
DEV 2020-07..2024-12 / HOLDOUT 2025-01+. Net at $0.10/0.20/0.30 per-trade /ATR.
Exit grid (dev-selected): base SL7/T2 maxh60 | tt after {6,15,30} M5 bars -> 0.75.
"""
import sys, pickle, time, json
from pathlib import Path
import numpy as np, pandas as pd
from numba import njit
from xgboost import XGBRegressor
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

SRV = Path("/home/jay/Desktop/my-agents-and-website/commercial/server")
sys.path.insert(0, str(SRV))
import decision_engine.edge_pullback as ep
OUT = Path(__file__).parent
FC = pickle.load(open(SRV / "decision_engine/models/hermes_dji_validated.pkl", "rb"))["feat_cols"]
t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
SPREADS = [0.10, 0.20, 0.30]; HEAD_SP = 0.20
TARGET_PER_DAY = 5.0; COOLDOWN = 3; DEV_END = pd.Timestamp("2025-01-01")
MAXH = 60   # 5h on M5 (same wall-clock as M1 maxh300)

m1 = pd.read_parquet("/home/jay/Desktop/new-model-zigzag/data/m1_xau_full.parquet")
m1["time"] = pd.to_datetime(m1["time"]); m1 = m1.sort_values("time").drop_duplicates("time").reset_index(drop=True)
s = m1.set_index("time")
df = pd.DataFrame({"open": s.open.resample("5min").first(), "high": s.high.resample("5min").max(),
                   "low": s.low.resample("5min").min(), "close": s.close.resample("5min").last(),
                   "tick_volume": s.tick_volume.resample("5min").sum()}).dropna(subset=["close"]).reset_index()
log(f"M5 bars: {len(df):,} ({df.time.iloc[0].date()} -> {df.time.iloc[-1].date()})")
feat = ep.compute_edge_features(df); log("features done")
atr = feat["atr14"].to_numpy(float); cdir = feat["committed_dir"].to_numpy(np.int64)
da = np.abs(feat["dist_at_signal"].to_numpy(float))
O = df["open"].to_numpy(float); H = df["high"].to_numpy(float); L = df["low"].to_numpy(float); C = df["close"].to_numpy(float)
times = df["time"].values; n = len(df)
ok = np.isfinite(atr) & (atr > 0) & (cdir != 0); ok[:300] = False; ok[-(MAXH + 2):] = False
idx = np.where((da <= 1.0) & ok)[0]; dirs = cdir[idx].astype(np.int64)
ct = pd.to_datetime(times[idx])
Xc = np.nan_to_num(feat[FC].to_numpy(np.float32)[idx], nan=0.0, posinf=0.0, neginf=0.0)
sig_atr = atr[idx]
log(f"candidates {len(idx):,} | median M5 ATR ${np.median(sig_atr):.3f} "
    f"(2025+: ${np.median(sig_atr[ct >= '2025-01-01']):.3f}) — spread-in-R at $0.20: "
    f"{0.20/np.median(sig_atr):.3f}R (vs ~0.43R on M1 pre-2025)")

@njit(cache=True)
def sim_exit(idxs, dirs, O, H, L, C, atr, n, SL, TRAIL, MAXH, ta, tt):
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

VAR = {"base SL7/T2 mh60": (0, 2.0), "tt6/0.75": (6, 0.75), "tt15/0.75": (15, 0.75), "tt30/0.75": (30, 0.75)}
SIMS = {k: sim_exit(idx, dirs, O, H, L, C, atr, n, 7.0, 2.0, MAXH, *v) for k, v in VAR.items()}
base_pnl = SIMS["base SL7/T2 mh60"][0]
log("sims done")

WINS = []; tsw = pd.Timestamp("2020-07-01"); lastd = ct.max()
while tsw < lastd:
    WINS.append((tsw - pd.DateOffset(years=3), tsw, tsw + pd.DateOffset(months=6))); tsw += pd.DateOffset(months=6)
rng = np.random.RandomState(0)
XGB = dict(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.85,
           colsample_bytree=0.85, min_child_weight=10, n_jobs=-1, random_state=0)

def calibrate(preds, trk, tr_days, eb_, xit_):
    cand = np.quantile(preds[trk], np.linspace(0.30, 0.97, 24)); thr = cand[-1]; best = 1e18
    for th in cand:
        kk = trk[preds[trk] >= th]
        if len(kk) < 5: continue
        tk = take(kk[np.argsort(eb_[kk])].astype(np.int64), eb_, xit_, COOLDOWN)
        gap = abs(len(tk) / tr_days - TARGET_PER_DAY)
        if gap < best: best = gap; thr = th
    return thr

results = {v: [] for v in VAR}; trades = {v: [] for v in VAR}
for tr_s, te_s, te_e in WINS:
    trm = (ct >= tr_s) & (ct < te_s); tem = (ct >= te_s) & (ct < te_e)
    if trm.sum() < 2000 or tem.sum() < 20: continue
    tix = np.where(trm)[0]
    tr_days = max((ct[tix].max() - ct[tix].min()).days * 5 / 7, 1)
    te_days = max((ct[tem].max() - ct[tem].min()).days * 5 / 7, 1)
    m = XGBRegressor(**XGB); m.fit(Xc[tix], base_pnl[tix])
    p = m.predict(Xc).astype(np.float64)
    for vname, (pnl_, eb_, xit_) in SIMS.items():
        thr = calibrate(p, tix, tr_days, eb_, xit_)
        kk = np.where(tem & (p >= thr))[0]
        if len(kk) == 0: results[vname].append(dict(win=str(te_s.date()), n=0)); continue
        tk = take(kk[np.argsort(eb_[kk])].astype(np.int64), eb_, xit_, COOLDOWN)
        R = pnl_[tk]; cost = 1.0 / sig_atr[tk]; nh = R - HEAD_SP * cost
        results[vname].append(dict(
            win=str(te_s.date()), n=len(tk), perday=len(tk) / te_days,
            **{f"net{int(sp*100)}": float((R - sp * cost).sum()) for sp in SPREADS},
            wr=float((nh > 0).mean() * 100)))
        for tt_, r_ in zip(times[xit_[tk]], nh): trades[vname].append((tt_, float(r_)))
    log(f"window {te_s.date()} done")

def agg(rows, dev):
    rr = [r for r in rows if r.get("n", 0) > 0 and ((pd.Timestamp(r["win"]) < DEV_END) == dev)]
    if not rr: return None
    d = dict(nwin=len(rr), n=sum(r["n"] for r in rr), perday=float(np.mean([r["perday"] for r in rr])))
    for sp in SPREADS:
        key = f"net{int(sp*100)}"
        d[key] = sum(r[key] for r in rr); d[key + "_w"] = sum(1 for r in rr if r[key] > 0)
    d["wr"] = float(np.average([r["wr"] for r in rr], weights=[r["n"] for r in rr]))
    return d

def maxdd(v, dev):
    tr = sorted([z for z in trades[v] if (pd.Timestamp(z[0]) < DEV_END) == dev], key=lambda z: z[0])
    if not tr: return 0.0
    eq = np.cumsum([r for _, r in tr]); return float((np.maximum.accumulate(eq) - eq).max())

hdr = f"{'variant':<18}{'n':>7}{'/day':>6}{'net@10c':>9}{'net@20c':>9}{'net@30c':>9}{'w+@20':>7}{'WR%':>6}{'DD@20':>8}"
for dev, label in [(True, "DEV (selection)"), (False, "HOLDOUT (untouched)")]:
    print(f"\n{'='*92}\nXAU M5 edge_pullback lab — {label}\n{'='*92}\n{hdr}")
    for vname in VAR:
        a = agg(results[vname], dev)
        if a is None: continue
        print(f"{vname:<18}{a['n']:>7}{a['perday']:>6.1f}{a['net10']:>+9.0f}{a['net20']:>+9.0f}"
              f"{a['net30']:>+9.0f}{a['net20_w']:>4}/{a['nwin']:<2}{a['wr']:>6.0f}{maxdd(vname, dev):>8.0f}")

# save daily PnL of best variant for the diversification-correlation check vs atlas_xau M1
for vname in VAR:
    tr = sorted(trades[vname], key=lambda z: z[0])
    if tr:
        ser = pd.Series([r for _, r in tr], index=pd.to_datetime([t for t, _ in tr])).resample("1D").sum()
        ser.to_csv(OUT / f"xau_m5_daily_{vname.split()[0].replace('/','_')}.csv")
json.dump(results, open(OUT / "xau_m5_results.json", "w"), default=str, indent=1)
log("xau m5 lab done")
