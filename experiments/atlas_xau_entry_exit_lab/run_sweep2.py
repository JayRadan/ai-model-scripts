"""
ATLAS XAU entry/exit lab — follow-up: robustness sweep of the DEV winner (tt60 =
time-tightening trail: after N bars, trail 2 -> tighter). Checks:
  1. parameter PLATEAU: tight_after x tight_trail grid — a robust edge shows a broad
     positive region on DEV, not a single lucky point
  2. LABEL-MATCHED retrain: XGB trained on tt-exit labels (instead of base-label
     model + exit overlay) — does matching the label lift it further?
  3. holding-time stats (EA implementability sanity)
Same honest harness as run_lab.py: train-only thr ~11/day, 1-slot cd5, dev/holdout.
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
FC = pickle.load(open(SRV / "decision_engine/models/atlas_xau_validated.pkl", "rb"))["feat_cols"]
t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)

SPREADS = [0.20, 0.30]; TARGET_PER_DAY = 11.0; COOLDOWN = 5
DEV_END = pd.Timestamp("2025-01-01")

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
def sim_tt(idxs, dirs, O, H, L, C, atr, n, SL, TRAIL, MAXH, tight_after, tight_trail):
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
            if tight_after > 0 and (jx - st) >= tight_after:
                tt = tight_trail * a
                if tt < trd: trd = tt
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

GRID = [(0, 2.0)] + [(ta, tt) for ta in (10, 20, 30) for tt in (0.5, 0.75)] + [(1, 1.5), (1, 1.0), (1, 0.75), (1, 0.5)]
log("global sims...")
SIMS = {g: sim_tt(idx, dirs, O, H, L, C, atr, n, 7.0, 2.0, 300, g[0], g[1]) for g in GRID}
base_pnl, base_eb, base_xit = SIMS[(0, 2.0)]
tt60_pnl = SIMS[(30, 0.75)][0]
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

res = {g: [] for g in GRID}; res["label_matched"] = []; hold_stats = {"base": [], "tt60": []}
for tr_s, te_s, te_e in WINS:
    trm = (ct >= tr_s) & (ct < te_s); tem = (ct >= te_s) & (ct < te_e)
    if trm.sum() < 4000 or tem.sum() < 20: continue
    tix = np.where(trm)[0]; tix_f = tix if len(tix) <= 150_000 else rng.choice(tix, 150_000, replace=False)
    tr_days = max((ct[tix].max() - ct[tix].min()).days * 5 / 7, 1)
    m_mean = XGBRegressor(**XGB); m_mean.fit(Xc[tix_f], base_pnl[tix_f])
    p_mean = m_mean.predict(Xc).astype(np.float64)
    m_lm = XGBRegressor(**XGB); m_lm.fit(Xc[tix_f], tt60_pnl[tix_f])
    p_lm = m_lm.predict(Xc).astype(np.float64)

    def ev(preds, pnl_, eb_, xit_, dest):
        thr = calibrate(preds, tix, tr_days, eb_, xit_)
        kk = np.where(tem & (preds >= thr))[0]
        if len(kk) == 0: dest.append(dict(win=str(te_s.date()), n=0)); return None
        tk = take(kk[np.argsort(eb_[kk])].astype(np.int64), eb_, xit_, COOLDOWN)
        R = pnl_[tk]; cost = 1.0 / sig_atr[tk]
        dest.append(dict(win=str(te_s.date()), n=len(tk),
                         net20=float((R - 0.20 * cost).sum()), net30=float((R - 0.30 * cost).sum())))
        return tk

    for g in GRID:
        pnl_, eb_, xit_ = SIMS[g]
        tk = ev(p_mean, pnl_, eb_, xit_, res[g])
        if g == (0, 2.0) and tk is not None: hold_stats["base"].append((xit_[tk] - eb_[tk]))
        if g == (30, 0.75) and tk is not None: hold_stats["tt60"].append((xit_[tk] - eb_[tk]))
    ev(p_lm, *SIMS[(30, 0.75)], res["label_matched"])
    log(f"window {te_s.date()} done")

def agg(rows, dev):
    rr = [r for r in rows if r.get("n", 0) > 0 and ((pd.Timestamp(r["win"]) < DEV_END) == dev)]
    if not rr: return dict(net20=0, net30=0, w=0, nw=0, n=0)
    return dict(net20=sum(r["net20"] for r in rr), net30=sum(r["net30"] for r in rr),
                w=sum(1 for r in rr if r["net20"] > 0), nw=len(rr), n=sum(r["n"] for r in rr))

print(f"\n{'='*84}\nPLATEAU — trail 2 -> tight_trail after tight_after bars (net R, ~11/day)\n{'='*84}")
print(f"{'config':<16}{'DEV net@20':>11}{'DEV net@30':>11}{'DEV w+':>8}{'HOLD net@20':>12}{'HOLD net@30':>12}{'HOLD w+':>9}")
for g in GRID:
    d = agg(res[g], True); h = agg(res[g], False)
    name = "base (no tt)" if g == (0, 2.0) else f"ta={g[0]:>3} tt={g[1]}"
    print(f"{name:<16}{d['net20']:>+11.0f}{d['net30']:>+11.0f}{d['w']:>5}/{d['nw']:<2}"
          f"{h['net20']:>+12.0f}{h['net30']:>+12.0f}{h['w']:>6}/{h['nw']:<2}")
d = agg(res["label_matched"], True); h = agg(res["label_matched"], False)
print(f"{'lm ta60 tt1.0':<16}{d['net20']:>+11.0f}{d['net30']:>+11.0f}{d['w']:>5}/{d['nw']:<2}"
      f"{h['net20']:>+12.0f}{h['net30']:>+12.0f}{h['w']:>6}/{h['nw']:<2}   (model trained on tt60 labels)")

for k in ("base", "tt60"):
    hb = np.concatenate(hold_stats[k])
    print(f"\nholding bars [{k}]: mean {hb.mean():.0f}  median {np.median(hb):.0f}  p90 {np.percentile(hb,90):.0f}  max {hb.max()}")
json.dump({str(k): v for k, v in res.items()}, open(OUT / "sweep2_results.json", "w"), default=str, indent=1)
log("sweep done")
