"""
ATLAS XAU — loser-side exit lab: can trades that are going to hit the -7R hard SL
be cut early? On the deployed tt30/0.75 baseline, tests rule-based loss-cuts:
  A. SL time-tighten: after K bars held, hard SL tightens 7 -> S*ATR
  B. underwater time-cut: at bar >= K, if close-based favor <= -X*ATR -> exit at close
  C. regime-flip cut: committed_dir flips opposite the trade while underwater -> exit
Plus a loss-composition diagnostic (how much loss comes from SL-hits vs trail vs timeout).
Same honest harness as run_lab.py: live features, train-only thr ~11/day, 1-slot cd5,
DEV 2020-07..2024-12 (selection) / HOLDOUT 2025-01..2026-05 (untouched), net @ $0.10/0.20/0.30.
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

SPREADS = [0.10, 0.20, 0.30]; HEAD_SP = 0.20
TARGET_PER_DAY = 11.0; COOLDOWN = 5; DEV_END = pd.Timestamp("2025-01-01")
TA, TT = 30, 0.75   # deployed tt-trail baseline

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

# exit codes: 0=hardSL 1=trail 2=timeout 3=uw_cut 4=regime_cut
@njit(cache=True)
def sim_loss(idxs, dirs, O, H, L, C, atr, cdir_arr, n,
             SL, TRAIL, MAXH, ta, tt,                  # deployed tt-trail base
             sl_ta, sl_tight,                          # A: <=0 off
             uw_ta, uw_thr,                            # B: <=0 off
             rf_on):                                   # C: 0/1
    m = len(idxs); pnl = np.zeros(m); ebar = np.full(m, -1, np.int64)
    xit = np.full(m, -1, np.int64); code = np.full(m, -1, np.int64)
    for k in range(m):
        i = idxs[k]; d = dirs[k]; a = atr[i]
        if i + 1 >= n or not (a > 0): continue
        st = i + 1; epr = O[st]; mf = 0.0
        end = min(st + MAXH, n - 1); done = False; tight_prev = False
        for jx in range(st, end + 1):
            held = jx - st
            hard = SL * a
            tightening = sl_ta > 0 and held >= sl_ta
            if tightening:
                hs = sl_tight * a
                if hs < hard: hard = hs
            adv = (epr - L[jx]) if d == 1 else (H[jx] - epr)
            if adv >= hard:
                # realistic fill: on the bar the stop MOVES, price may already be
                # beyond the new level -> EA market-closes ~at the open, not at the stop
                if tightening and not tight_prev:
                    adv_open = (epr - O[jx]) if d == 1 else (O[jx] - epr)
                    pnl[k] = -max(adv_open, hard) / a
                else:
                    pnl[k] = -hard / a
                xit[k] = jx; code[k] = 0; done = True; break
            tight_prev = tightening
            fav = d * (C[jx] - epr)
            if fav > mf: mf = fav
            trd = TRAIL * a
            if ta > 0 and held >= ta:
                w = tt * a
                if w < trd: trd = w
            if mf >= trd and (mf - fav) >= trd:
                pnl[k] = (mf - trd) / a; xit[k] = jx; code[k] = 1; done = True; break
            if uw_ta > 0 and held >= uw_ta and fav <= -uw_thr * a:
                pnl[k] = fav / a; xit[k] = jx; code[k] = 3; done = True; break
            if rf_on == 1 and cdir_arr[jx] == -d and fav < 0.0:
                pnl[k] = fav / a; xit[k] = jx; code[k] = 4; done = True; break
        if not done:
            pnl[k] = d * (C[end] - epr) / a; xit[k] = end; code[k] = 2
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

def SIM(sl_ta=0, sl_tight=0.0, uw_ta=0, uw_thr=0.0, rf=0):
    return sim_loss(idx, dirs, O, H, L, C, atr, cdir, n, 7.0, 2.0, 300, TA, TT,
                    sl_ta, sl_tight, uw_ta, uw_thr, rf)

log("global sims...")
SIMS = {"base_tt (deployed)": SIM()}
for K in (30, 60):
    for S in (2.0, 3.0, 4.0):
        SIMS[f"A_sl{K}/{S:g}"] = SIM(sl_ta=K, sl_tight=S)
for K in (30, 60):
    for X in (1.0, 2.0):
        SIMS[f"B_uw{K}/-{X:g}"] = SIM(uw_ta=K, uw_thr=X)
SIMS["C_regflip"] = SIM(rf=1)
SIMS["AC_sl60/3+rf"] = SIM(sl_ta=60, sl_tight=3.0, rf=1)
log(f"{len(SIMS)} sims done")

WINS = []; tsw = pd.Timestamp("2020-07-01"); lastd = ct.max()
while tsw < lastd:
    WINS.append((tsw - pd.DateOffset(years=3), tsw, tsw + pd.DateOffset(months=6))); tsw += pd.DateOffset(months=6)
rng = np.random.RandomState(0)
XGB = dict(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.85,
           colsample_bytree=0.85, min_child_weight=10, n_jobs=-1, random_state=0)
base_pnl = SIMS["base_tt (deployed)"][0]

def calibrate(preds, trk, tr_days, eb_, xit_):
    cand = np.quantile(preds[trk], np.linspace(0.30, 0.97, 24)); thr = cand[-1]; best = 1e18
    for th in cand:
        kk = trk[preds[trk] >= th]
        if len(kk) < 5: continue
        tk = take(kk[np.argsort(eb_[kk])].astype(np.int64), eb_, xit_, COOLDOWN)
        gap = abs(len(tk) / tr_days - TARGET_PER_DAY)
        if gap < best: best = gap; thr = th
    return thr

results = {v: [] for v in SIMS}; comp = {v: np.zeros(5) for v in SIMS}; compR = {v: np.zeros(5) for v in SIMS}
for tr_s, te_s, te_e in WINS:
    trm = (ct >= tr_s) & (ct < te_s); tem = (ct >= te_s) & (ct < te_e)
    if trm.sum() < 4000 or tem.sum() < 20: continue
    tix = np.where(trm)[0]; tix_f = tix if len(tix) <= 150_000 else rng.choice(tix, 150_000, replace=False)
    tr_days = max((ct[tix].max() - ct[tix].min()).days * 5 / 7, 1)
    te_days = max((ct[tem].max() - ct[tem].min()).days * 5 / 7, 1)
    m = XGBRegressor(**XGB); m.fit(Xc[tix_f], base_pnl[tix_f])
    p = m.predict(Xc).astype(np.float64)
    for vname, (pnl_, eb_, xit_, code_) in SIMS.items():
        thr = calibrate(p, tix, tr_days, eb_, xit_)
        kk = np.where(tem & (p >= thr))[0]
        if len(kk) == 0: results[vname].append(dict(win=str(te_s.date()), n=0)); continue
        tk = take(kk[np.argsort(eb_[kk])].astype(np.int64), eb_, xit_, COOLDOWN)
        R = pnl_[tk]; cost = 1.0 / sig_atr[tk]; nh = R - HEAD_SP * cost
        results[vname].append(dict(
            win=str(te_s.date()), n=len(tk), perday=len(tk) / te_days,
            **{f"net{int(sp*100)}": float((R - sp * cost).sum()) for sp in SPREADS},
            wr=float((nh > 0).mean() * 100)))
        for cc in range(5):
            mm = code_[tk] == cc
            comp[vname][cc] += mm.sum(); compR[vname][cc] += nh[mm].sum()
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

CN = ["hardSL", "trail", "timeout", "uw_cut", "regime_cut"]
print(f"\nLOSS COMPOSITION — base_tt, all taken trades (net@20c):")
v = "base_tt (deployed)"
for cc in range(5):
    if comp[v][cc]: print(f"  {CN[cc]:<10} n={int(comp[v][cc]):>6}  sum {compR[v][cc]:>+9.0f}R")

hdr = f"{'variant':<20}{'n':>7}{'/day':>6}{'net@10c':>9}{'net@20c':>9}{'net@30c':>9}{'w+@20':>7}{'WR%':>6}"
for dev, label in [(True, "DEV (selection)"), (False, "HOLDOUT (untouched)")]:
    print(f"\n{'='*80}\nATLAS XAU loser-exit lab — {label}\n{'='*80}\n{hdr}")
    for vname in SIMS:
        a = agg(results[vname], dev)
        if a is None: continue
        print(f"{vname:<20}{a['n']:>7}{a['perday']:>6.1f}{a['net10']:>+9.0f}{a['net20']:>+9.0f}"
              f"{a['net30']:>+9.0f}{a['net20_w']:>4}/{a['nwin']:<2}{a['wr']:>6.0f}")
json.dump({v: results[v] for v in SIMS}, open(OUT / "losers_results.json", "w"), default=str, indent=1)
log("losers lab done")
