"""Final equity chart: deployed exit (SL7/trail2) vs time-boxed-patience trail
(after 30 bars, trail 2 -> 0.75 conservative / -> 0.5 aggressive). Honest WF
harness (train-only thr ~11/day, 1-slot cd5), net @ $0.20 spread. Holdout shaded."""
import sys, pickle, time
from pathlib import Path
import numpy as np, pandas as pd
from numba import njit
from xgboost import XGBRegressor
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

SRV = Path("/home/jay/Desktop/my-agents-and-website/commercial/server")
sys.path.insert(0, str(SRV))
import decision_engine.edge_pullback as ep
OUT = Path(__file__).parent
FC = pickle.load(open(SRV / "decision_engine/models/atlas_xau_validated.pkl", "rb"))["feat_cols"]
t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
TARGET = 11.0; CD = 5; DEV_END = pd.Timestamp("2025-01-01"); SP = 0.20

df = pd.read_parquet("/home/jay/Desktop/new-model-zigzag/data/m1_xau_full.parquet")
df["time"] = pd.to_datetime(df["time"]); df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
feat = ep.compute_edge_features(df); log("features done")
atr = feat["atr14"].to_numpy(float); cdir = feat["committed_dir"].to_numpy(np.int64)
da = np.abs(feat["dist_at_signal"].to_numpy(float))
O = df["open"].to_numpy(float); H = df["high"].to_numpy(float); L = df["low"].to_numpy(float); C = df["close"].to_numpy(float)
times = df["time"].values; n = len(df)
ok = np.isfinite(atr) & (atr > 0) & (cdir != 0); ok[:300] = False; ok[-301:] = False
idx = np.where((da <= 1.0) & ok)[0]; dirs = cdir[idx].astype(np.int64)
ct = pd.to_datetime(times[idx]); Xc = feat[FC].to_numpy(np.float32)[idx]; sig_atr = atr[idx]

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

CFGS = {"deployed SL7/trail2": (0, 2.0), "tt30/0.75 (conservative)": (30, 0.75), "tt30/0.5 (aggressive)": (30, 0.5)}
SIMS = {k: sim_tt(idx, dirs, O, H, L, C, atr, n, 7.0, 2.0, 300, *v) for k, v in CFGS.items()}
base_pnl = SIMS["deployed SL7/trail2"][0]; log("sims done")

WINS = []; tsw = pd.Timestamp("2020-07-01"); lastd = ct.max()
while tsw < lastd:
    WINS.append((tsw - pd.DateOffset(years=3), tsw, tsw + pd.DateOffset(months=6))); tsw += pd.DateOffset(months=6)
rng = np.random.RandomState(0)
trades = {k: [] for k in CFGS}
for tr_s, te_s, te_e in WINS:
    trm = (ct >= tr_s) & (ct < te_s); tem = (ct >= te_s) & (ct < te_e)
    if trm.sum() < 4000 or tem.sum() < 20: continue
    tix = np.where(trm)[0]; tix_f = tix if len(tix) <= 150_000 else rng.choice(tix, 150_000, replace=False)
    tr_days = max((ct[tix].max() - ct[tix].min()).days * 5 / 7, 1)
    m = XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.85,
                     colsample_bytree=0.85, min_child_weight=10, n_jobs=-1, random_state=0)
    m.fit(Xc[tix_f], base_pnl[tix_f]); p = m.predict(Xc).astype(np.float64)
    for name, (pnl_, eb_, xit_) in SIMS.items():
        cand = np.quantile(p[tix], np.linspace(0.30, 0.97, 24)); thr = cand[-1]; best = 1e18
        for th in cand:
            kk = tix[p[tix] >= th]
            if len(kk) < 5: continue
            tk = take(kk[np.argsort(eb_[kk])].astype(np.int64), eb_, xit_, CD)
            gap = abs(len(tk) / tr_days - TARGET)
            if gap < best: best = gap; thr = th
        kk = np.where(tem & (p >= thr))[0]
        if len(kk) == 0: continue
        tk = take(kk[np.argsort(eb_[kk])].astype(np.int64), eb_, xit_, CD)
        net = pnl_[tk] - SP / sig_atr[tk]
        for tt_, r_ in zip(times[xit_[tk]], net): trades[name].append((tt_, r_))
    log(f"window {te_s.date()} done")

fig, ax = plt.subplots(figsize=(13, 5.5))
for (name, _), col in zip(CFGS.items(), ["#64748b", "#16a34a", "#2563eb"]):
    tr = sorted(trades[name], key=lambda z: z[0])
    tt_ = pd.to_datetime([t for t, _ in tr]); eq = np.cumsum([r for _, r in tr])
    dd = float((np.maximum.accumulate(eq) - eq).max())
    ax.plot(tt_, eq, lw=1.3, color=col, label=f"{name}  ({eq[-1]:+.0f}R, maxDD {dd:.0f}R)")
ax.axvline(DEV_END, color="#dc2626", lw=1, ls="--")
ax.axvspan(DEV_END, pd.Timestamp("2026-06-01"), alpha=0.06, color="#dc2626")
ax.text(DEV_END, ax.get_ylim()[0], " HOLDOUT →", color="#dc2626", fontsize=9, va="bottom")
ax.set_title("Atlas XAU — deployed exit vs time-boxed-patience trail | WF net @ $0.20 spread, ~11/day, 1-slot")
ax.axhline(0, color="k", lw=0.6); ax.grid(alpha=0.3); ax.legend(loc="upper left")
plt.tight_layout(); plt.savefig(OUT / "final_equity.png", dpi=110)
print(f"-> {OUT/'final_equity.png'}")
log("done")
