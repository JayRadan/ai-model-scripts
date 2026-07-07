"""
HERMES BTC FIX — validate the edge_pullback + tt-trail recipe on 8y BTC M1.
Context: deployed hermes_btc bundle (2026-05-26) predates the causal-HTF fix
(trained on look-ahead features, honest revalidation said 'BTC no edge') and is
losing live. This lab tests the ONE recipe that survives honest validation
(edge_pullback, live on 4 products) on BTC — never validated here before.

Harness identical to the XAU/DJI labs: live feature pipeline, 6mo test windows,
3y rolling train, train-only thr ~11/day, 1-slot cd5,
DEV = 2020-07..2024-12 (selection) / HOLDOUT = 2025-01+ (untouched).
Spread charged per trade as flat R (BTC spread scales with price): 0.10/0.20/0.30 R.
Variants: base SL7/T2 | tt30/0.75 (deployed pick) | tt30/0.5 | uniform-tight control.
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
SPREADS_R = [0.10, 0.20, 0.30]; HEAD = 0.20
TARGET_PER_DAY = 11.0; COOLDOWN = 5; DEV_END = pd.Timestamp("2025-01-01")

df = pd.read_parquet("/home/jay/Desktop/new-model-zigzag/data/m1_btc_orderflow_8y.parquet",
                     columns=["time", "open", "high", "low", "close", "tick_volume"])
df["time"] = pd.to_datetime(df["time"]); df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
log(f"{len(df):,} bars {df.time.iloc[0].date()} -> {df.time.iloc[-1].date()}")
feat = ep.compute_edge_features(df); log("features done")
atr = feat["atr14"].to_numpy(float); cdir = feat["committed_dir"].to_numpy(np.int64)
da = np.abs(feat["dist_at_signal"].to_numpy(float))
O = df["open"].to_numpy(float); H = df["high"].to_numpy(float); L = df["low"].to_numpy(float); C = df["close"].to_numpy(float)
times = df["time"].values; n = len(df)
ok = np.isfinite(atr) & (atr > 0) & (cdir != 0); ok[:300] = False; ok[-301:] = False
idx = np.where((da <= 1.0) & ok)[0]; dirs = cdir[idx].astype(np.int64)
ct = pd.to_datetime(times[idx])
Xc = np.nan_to_num(feat[FC].to_numpy(np.float32)[idx], nan=0.0, posinf=0.0, neginf=0.0)
sig_atr = atr[idx]
log(f"candidates {len(idx):,} | median ATR ${np.median(sig_atr):,.0f} "
    f"(2025+: ${np.median(sig_atr[ct >= '2025-01-01']):,.0f}) | spread grid = flat R units")

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

VAR = {"base SL7/T2": (0, 2.0), "tt30/0.75": (30, 0.75), "tt30/0.5": (30, 0.5), "uni1/0.5 ctrl": (1, 0.5)}
log("sims...")
SIMS = {k: sim_exit(idx, dirs, O, H, L, C, atr, n, 7.0, 2.0, 300, *v) for k, v in VAR.items()}
base_pnl = SIMS["base SL7/T2"][0]

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
    if trm.sum() < 4000 or tem.sum() < 20: continue
    tix = np.where(trm)[0]; tix_f = tix if len(tix) <= 150_000 else rng.choice(tix, 150_000, replace=False)
    tr_days = max((ct[tix].max() - ct[tix].min()).days * 7 / 7, 1)   # BTC trades 7d/wk
    te_days = max((ct[tem].max() - ct[tem].min()).days, 1)
    m = XGBRegressor(**XGB); m.fit(Xc[tix_f], base_pnl[tix_f])
    p = m.predict(Xc).astype(np.float64)
    for vname, (pnl_, eb_, xit_) in SIMS.items():
        thr = calibrate(p, tix, tr_days, eb_, xit_)
        kk = np.where(tem & (p >= thr))[0]
        if len(kk) == 0: results[vname].append(dict(win=str(te_s.date()), n=0)); continue
        tk = take(kk[np.argsort(eb_[kk])].astype(np.int64), eb_, xit_, COOLDOWN)
        R = pnl_[tk]; nh = R - HEAD
        results[vname].append(dict(
            win=str(te_s.date()), n=len(tk), perday=len(tk) / te_days,
            **{f"net{int(sp*100)}": float((R - sp).sum()) for sp in SPREADS_R},
            wr=float((nh > 0).mean() * 100)))
        for tt_, r_ in zip(times[xit_[tk]], nh): trades[vname].append((tt_, float(r_)))
    log(f"window {te_s.date()} done")

def agg(rows, dev):
    rr = [r for r in rows if r.get("n", 0) > 0 and ((pd.Timestamp(r["win"]) < DEV_END) == dev)]
    if not rr: return None
    d = dict(nwin=len(rr), n=sum(r["n"] for r in rr), perday=float(np.mean([r["perday"] for r in rr])))
    for sp in SPREADS_R:
        key = f"net{int(sp*100)}"
        d[key] = sum(r[key] for r in rr); d[key + "_w"] = sum(1 for r in rr if r[key] > 0)
    d["wr"] = float(np.average([r["wr"] for r in rr], weights=[r["n"] for r in rr]))
    return d

def maxdd(v, dev):
    tr = sorted([z for z in trades[v] if (pd.Timestamp(z[0]) < DEV_END) == dev], key=lambda z: z[0])
    if not tr: return 0.0
    eq = np.cumsum([r for _, r in tr]); return float((np.maximum.accumulate(eq) - eq).max())

hdr = f"{'variant':<16}{'n':>7}{'/day':>6}{'net@.1R':>9}{'net@.2R':>9}{'net@.3R':>9}{'w+@.2':>7}{'WR%':>6}{'DD@.2':>8}"
for dev, label in [(True, "DEV (2020-07..2024-12, selection)"), (False, "HOLDOUT (2025+, untouched)")]:
    print(f"\n{'='*92}\nBTC M1 edge_pullback lab — {label}\n{'='*92}\n{hdr}")
    for vname in VAR:
        a = agg(results[vname], dev)
        if a is None: continue
        print(f"{vname:<16}{a['n']:>7}{a['perday']:>6.1f}{a['net10']:>+9.0f}{a['net20']:>+9.0f}"
              f"{a['net30']:>+9.0f}{a['net20_w']:>4}/{a['nwin']:<2}{a['wr']:>6.0f}{maxdd(vname, dev):>8.0f}")

fig, ax = plt.subplots(figsize=(13, 5))
for vname, col in [("base SL7/T2", "#64748b"), ("tt30/0.75", "#16a34a")]:
    tr = sorted(trades[vname], key=lambda z: z[0])
    tt_ = pd.to_datetime([t for t, _ in tr]); eq = np.cumsum([r for _, r in tr])
    ax.plot(tt_, eq, lw=1.2, color=col, label=f"{vname} ({eq[-1]:+.0f}R)")
ax.axvline(DEV_END, color="#dc2626", lw=1, ls="--")
ax.set_title("BTC M1 edge_pullback — WF equity net @ 0.2R spread (holdout right of red line)")
ax.axhline(0, color="k", lw=0.6); ax.grid(alpha=0.3); ax.legend()
plt.tight_layout(); plt.savefig(OUT / "btc_edge_equity.png", dpi=110)
json.dump(results, open(OUT / "btc_edge_results.json", "w"), default=str, indent=1)
log("btc edge lab done")
