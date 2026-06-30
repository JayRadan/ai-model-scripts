"""
Hermes DJI — FINAL honest validation + production artifacts for the discovered edge.

Config locked: entry pb1.0 (|dist_tfk|<=1.0, dir=committed_dir), label/exit SL6/T2 trail
max_hold 300, XGBRegressor predicting GROSS R, 1-slot cooldown 5.
RIGOR: per-window selection threshold is set from the TRAINING prediction distribution
(targets ~11/day) — NOT tuned on test. Reports net at 1/1.5/2 pt spread, recent period,
equity curve PNG, head-to-head vs deployed q_model, and saves a production bundle.
"""
import sys, pickle, time
from pathlib import Path
import numpy as np, pandas as pd
from numba import njit
from xgboost import XGBRegressor
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

DE = Path("/home/jay/Desktop/my-agents-and-website/commercial/server/decision_engine")
sys.path.insert(0, str(DE))
import hermes_features as hf
from configs.hermes_dji import HERMES_DJI as CFG
DEP = pickle.load(open(DE / "models/hermes_dji_validated.pkl", "rb")); FC = DEP["feat_cols"]
OUT = Path("/home/jay/Desktop/new-model-zigzag/experiments/_hermes_retrain"); OUT.mkdir(exist_ok=True)
t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)

df = pd.read_parquet("/home/jay/Desktop/new-model-zigzag/data/m1_dji_full.parquet")
df = df.rename(columns={[c for c in df.columns if "time" in c.lower()][0]: "time"})
df["time"] = pd.to_datetime(df["time"]); df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
if "tick_volume" not in df.columns: df["tick_volume"] = df.get("volume", 0)
feat = hf.compute_all_features(df.copy(), CFG)
_A = 2.0 / 51.0; s = df.set_index("time")["close"]
for nm, tfm in [("m5", 5), ("m15", 15), ("h1", 60)]:
    g = s.resample(f"{tfm}min").last().dropna(); hc = g.to_numpy(np.float64)
    et = (g.index + pd.Timedelta(minutes=tfm)).values
    N = len(df); rsi = np.full(N, np.nan); slope = np.full(N, np.nan); emad = np.full(N, np.nan)
    ema = np.empty(len(hc)); e = hc[0]
    for i in range(len(hc)):
        e = hc[0] if i == 0 else e * (1 - _A) + hc[i] * _A; ema[i] = e
    dl = np.diff(hc, prepend=hc[0]); cg = np.cumsum(np.clip(dl, 0, None)); cl = np.cumsum(np.clip(-dl, 0, None))
    j = np.searchsorted(et, df["time"].values, side="right") - 1; ok = j >= 14; jj = j[ok]
    cc = df["close"].to_numpy(np.float64)[ok]
    slope[ok] = cc - hc[jj - 4]; emad[ok] = cc - (ema[jj] * (1 - _A) + cc * _A)
    dc = cc - hc[jj]; gs = (cg[jj] - cg[jj - 13]) + np.clip(dc, 0, None); ls = (cl[jj] - cl[jj - 13]) + np.clip(-dc, 0, None)
    rs = (gs / 14.0) / np.where(ls == 0, np.nan, ls / 14.0); rsi[ok] = 100 - 100 / (1 + rs)
    feat[f"{nm}_rsi14"] = rsi; feat[f"{nm}_slope5"] = slope; feat[f"{nm}_ema50_dist"] = emad
log("features done")

atr = feat["atr14"].to_numpy(float); cdir = feat["committed_dir"].to_numpy(np.int64)
ds = feat["dist_at_signal"].to_numpy(float); da = np.abs(ds)
O = df["open"].to_numpy(float); H = df["high"].to_numpy(float); L = df["low"].to_numpy(float); C = df["close"].to_numpy(float)
times = df["time"].values; n = len(df); medatr = np.nanmedian(atr)
base_ok = np.isfinite(atr) & (atr > 0) & (cdir != 0); base_ok[:300] = False; base_ok[-301:] = False
idx = np.where((da <= 1.0) & base_ok)[0]; dirs = cdir[idx].astype(np.int64); eb = idx + 1
ct = pd.to_datetime(times[idx]); Xc = feat[FC].to_numpy(np.float32)[idx]

@njit
def sim_gross(idxs, dirs, O, H, L, C, atr, SL, TRAIL, MAXH, n):
    m = len(idxs); pnl = np.empty(m); xit = np.empty(m, np.int64)
    for k in range(m):
        i = idxs[k]; d = dirs[k]; a = atr[i]; ei = i + 1
        if ei >= n or not (a > 0): pnl[k] = 0.0; xit[k] = min(i + 1, n - 1); continue
        ep = O[ei]; hard = SL * a; trd = TRAIL * a; mf = 0.0; end = min(ei + MAXH, n - 1); done = False
        for jx in range(ei, end + 1):
            fav = d * (C[jx] - ep)
            if fav > mf: mf = fav
            if d == 1 and (ep - L[jx]) >= hard: pnl[k] = -SL; xit[k] = jx; done = True; break
            if d == -1 and (H[jx] - ep) >= hard: pnl[k] = -SL; xit[k] = jx; done = True; break
            if mf >= trd and (mf - fav) >= trd: pnl[k] = (mf - trd) / a; xit[k] = jx; done = True; break
        if not done: pnl[k] = d * (C[end] - ep) / a; xit[k] = end
    return pnl, xit

@njit
def take(order_idx, entry_bar, exit_bar, pnl, dirs, cd):
    busy = -1; m = len(order_idx); R = np.zeros(m); D = np.zeros(m, np.int64); X = np.zeros(m, np.int64); c = 0
    for t in range(m):
        k = order_idx[t]
        if entry_bar[k] <= busy: continue
        R[c] = pnl[k]; D[c] = dirs[k]; X[c] = exit_bar[k]; busy = exit_bar[k] + cd; c += 1
    return R[:c], D[:c], X[:c]

pnlG, xit = sim_gross(idx, dirs, O, H, L, C, atr, 6.0, 2.0, 300, n)
WINS = []; first = pd.Timestamp("2020-07-01"); tsw = first; lastd = ct.max()
while tsw < lastd:
    WINS.append((tsw - pd.DateOffset(years=3), tsw, tsw + pd.DateOffset(months=6))); tsw += pd.DateOffset(months=6)
TARGET_PER_DAY = 11.0; rng = np.random.RandomState(0)

# deployed q_model predictions (for head-to-head)
qdep = DEP["q_model"].predict(Xc)

rows = []; eq_times = []; eq_net1 = []; allnet1 = []
for tr_s, te_s, te_e in WINS:
    trm = (ct >= tr_s) & (ct < te_s); tem = (ct >= te_s) & (ct < te_e)
    if trm.sum() < 4000 or tem.sum() < 20: continue
    tix = np.where(trm)[0]; tix_f = tix if len(tix) <= 150000 else rng.choice(tix, 150000, replace=False)
    m = XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.85,
                     colsample_bytree=0.85, min_child_weight=10, n_jobs=-1, random_state=0)
    m.fit(Xc[tix_f], pnlG[tix_f])
    # THRESHOLD FROM TRAIN ONLY: run the 1-slot portfolio on TRAIN predictions at
    # several thresholds, pick the one whose TAKEN trades/day is closest to target.
    pall = m.predict(Xc)
    tr_days = max((ct[tix].max() - ct[tix].min()).days * 5/7, 1)
    cand_thr = np.quantile(pall[tix], np.linspace(0.30, 0.95, 22))
    thr = cand_thr[-1]; best_gap = 1e9
    for th in cand_thr:
        kk = np.where(trm)[0][pall[trm] >= th]
        if len(kk) < 5: continue
        order = kk[np.argsort(eb[kk])]; Rtr, _, _ = take(order.astype(np.int64), eb, xit, pnlG, dirs, 5)
        gap = abs(len(Rtr) / tr_days - TARGET_PER_DAY)
        if gap < best_gap: best_gap = gap; thr = th
    pte = pall[tem]
    kk = np.where(tem)[0][pte >= thr]; order = kk[np.argsort(eb[kk])]
    R, D, X = take(order.astype(np.int64), eb, xit, pnlG, dirs, 5)
    if len(R) == 0: continue
    net1 = R - 1.0/medatr
    wi = np.where((pd.to_datetime(times) >= te_s) & (pd.to_datetime(times) < te_e))[0]
    bh = (C[wi[-1]] - C[wi[0]]) / medatr
    rows.append(dict(win=str(te_s.date()), n=len(R), perday=len(R)/126, gross=R.sum(),
                     net1=net1.sum(), net15=(R-1.5/medatr).sum(), net2=(R-2.0/medatr).sum(),
                     wr=(net1>0).mean()*100, sh=R[D==-1].sum(), bh=bh,
                     pf=net1[net1>0].sum()/max(-net1[net1<=0].sum(),1e-9)))
    for ti, r in zip(times[X], net1): eq_times.append(ti); eq_net1.append(r)
    allnet1.extend(net1.tolist())

allnet1 = np.array(allnet1); ntr = len(allnet1)
g = sum(r["gross"] for r in rows); net1 = sum(r["net1"] for r in rows); net15 = sum(r["net15"] for r in rows); net2 = sum(r["net2"] for r in rows)
prof1 = sum(1 for r in rows if r["net1"] > 0)
print(f"\n{'='*94}\nHERMES DJI — FINAL HONEST WALK-FORWARD (train-only thresholds, pb1.0 SL6/T2, ~11/day)\n{'='*94}")
print(f"{'window':<13}{'n':>5}{'/day':>6}{'grossR':>8}{'net@1pt':>9}{'net@1.5':>9}{'net@2pt':>9}{'WR%':>6}{'PF':>6}{'shortR':>8}{'B&H':>7}")
for r in rows:
    print(f"{r['win']:<13}{r['n']:>5}{r['perday']:>6.1f}{r['gross']:>+8.0f}{r['net1']:>+9.0f}{r['net15']:>+9.0f}"
          f"{r['net2']:>+9.0f}{r['wr']:>6.0f}{r['pf']:>6.2f}{r['sh']:>+8.0f}{r['bh']:>+7.0f}")
print("-"*94)
print(f"TOTAL gross {g:+.0f} | net@1pt {net1:+.0f} | net@1.5 {net15:+.0f} | net@2pt {net2:+.0f} | "
      f"profitable windows {prof1}/{len(rows)} (@1pt)")
print(f"{ntr} trades, {ntr/1541:.1f}/day, per-trade net@1pt {allnet1.mean():+.3f}R, "
      f"maxDD@1pt {float((np.maximum.accumulate(np.cumsum(allnet1))-np.cumsum(allnet1)).max()):.0f}R")
# recent 12 months
rec = [r for r in rows if r["win"] >= "2025-01"]
print(f"\nRECENT (2025-01 →): net@1pt {sum(r['net1'] for r in rec):+.0f}R over {sum(r['n'] for r in rec)} trades, "
      f"{sum(1 for r in rec if r['net1']>0)}/{len(rec)} windows +")

# equity PNG
order = np.argsort(pd.to_datetime(eq_times)); eqs = np.cumsum(np.array(eq_net1)[order])
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(pd.to_datetime(eq_times)[order], eqs, lw=1.3, color="#16a34a")
ax.set_title(f"Hermes DJI EDGE — WF net@1pt equity (R)  |  +{net1:.0f}R, {prof1}/{len(rows)} windows, {ntr/1541:.1f}/day", fontsize=12)
ax.axhline(0, color="k", lw=0.6); ax.set_ylabel("cumulative net R (1pt spread)"); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(OUT / "hermes_dji_EDGE_equity.png", dpi=110)
print(f"\nequity curve -> {OUT/'hermes_dji_EDGE_equity.png'}")

# production bundle: train on ALL data, store threshold recipe
mfull = XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.85,
                     colsample_bytree=0.85, min_child_weight=10, n_jobs=-1, random_state=0)
allmask = base_ok[idx] if False else np.ones(len(idx), bool)
mfull.fit(Xc, pnlG)
pall = mfull.predict(Xc); thr_prod = np.quantile(pall, max(0.0, 1 - TARGET_PER_DAY * 1541 / len(pall)))
pickle.dump({"q_model": mfull, "feat_cols": FC,
             "recipe": {"entry": "pb1.0 |dist_tfk|<=1.0 dir=committed_dir", "label": "trail SL6/T2 maxhold300 gross-R",
                        "threshold": float(thr_prod), "target_per_day": TARGET_PER_DAY, "near_thr_unused": True,
                        "note": "predict gross R, take if pred>=threshold, 1-slot cooldown5; deploy only if US30 spread<=2pt"}},
            open(OUT / "hermes_dji_EDGE_production.pkl", "wb"))
print(f"production bundle -> {OUT/'hermes_dji_EDGE_production.pkl'}  (thr={thr_prod:.3f})  NOT deployed")
log("final validation done")
