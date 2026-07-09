"""
HEDGE SPREAD GRID — DJI only. The hedge lab winner (d2r trigger + own-trail exits)
was DEV 6/9 +208R / HOLDOUT 3/3 +411R at 1.5pt. Dev per-trade is thin (+0.03R),
so before any deploy: charge 1.5 / 2.0 / 2.5pt on the hedge stream. Naive triggers
only (no ML) — same harness as run_lab_hedge.py.
"""
import sys, pickle, time, json
from pathlib import Path
import numpy as np, pandas as pd
from numba import njit
from xgboost import XGBRegressor

SRV = Path("/home/jay/Desktop/my-agents-and-website/commercial/server")
sys.path.insert(0, str(SRV))
import decision_engine.edge_pullback as ep
# NOTE: run_lab_hedge has no __main__ guard — do NOT import it (would re-run the
# whole lab). Kernels are copied verbatim below.
OUT = Path(__file__).parent
FC = pickle.load(open(SRV / "decision_engine/models/atlas_xau_validated.pkl", "rb"))["feat_cols"]
t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
COOLDOWN = 5; DEV_END = pd.Timestamp("2025-01-01"); TA, TT = 30, 0.75
SPREADS = (1.5, 2.0, 2.5)

@njit(cache=True)
def sim_base(idxs, dirs, O, H, L, C, atr, n, SL, TRAIL, MAXH, ta, tt):
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
def snapshots(idxs, dirs, O, H, L, C, atr, ebar, xit, n, mode, fixed_k, uw_thr):
    m = len(idxs)
    snap = np.full(m, -1, np.int64); fnow = np.zeros(m); mfs = np.zeros(m)
    mins = np.zeros(m); uwf = np.zeros(m); slp = np.zeros(m)
    for k in range(m):
        st = ebar[k]; xb = xit[k]
        if st < 0 or xb < 0: continue
        i = idxs[k]; d = dirs[k]; a = atr[i]; epr = O[st]
        sb = -1
        if mode == 0:
            if xb > st + fixed_k: sb = st + fixed_k
        else:
            end2 = min(xb - 1, st + 299)
            for jx in range(st, end2 + 1):
                if d * (C[jx] - epr) <= -uw_thr * a: sb = jx; break
        if sb < 0 or sb >= xb: continue
        f_now = d * (C[sb] - epr)
        snap[k] = sb; fnow[k] = f_now / a
    return snap, fnow, mfs, mins, uwf, slp

@njit(cache=True)
def take(order_idx, ebar, xit, cd):
    busy = -1; m = len(order_idx); out = np.empty(m, np.int64); c = 0
    for t in range(m):
        k = order_idx[t]
        if ebar[k] <= busy: continue
        out[c] = k; busy = xit[k] + cd; c += 1
    return out[:c]

@njit(cache=True)
def sim_hedge(sb_arr, dirs, O, H, L, C, atr, n, SL, TRAIL, MAXH, ta, tt):
    m = len(sb_arr); pnl = np.zeros(m); ha = np.zeros(m); valid = np.zeros(m, np.int8)
    for k in range(m):
        sbb = sb_arr[k]
        if sbb < 0: continue
        d = -dirs[k]; a = atr[sbb]; st = sbb + 1
        if st >= n or not (a > 0): continue
        epr = O[st]; hard = SL * a; mf = 0.0
        end = min(st + MAXH, n - 1); done = False
        for jx in range(st, end + 1):
            adv = (epr - L[jx]) if d == 1 else (H[jx] - epr)
            if adv >= hard: pnl[k] = -SL; done = True; break
            fav = d * (C[jx] - epr)
            if fav > mf: mf = fav
            trd = TRAIL * a
            if ta > 0 and (jx - st) >= ta:
                w = tt * a
                if w < trd: trd = w
            if mf >= trd and (mf - fav) >= trd: pnl[k] = (mf - trd) / a; done = True; break
        if not done: pnl[k] = d * (C[end] - epr) / a
        ha[k] = a; valid[k] = 1
    return pnl, ha, valid

df = pd.read_parquet("/home/jay/Desktop/new-model-zigzag/data/m1_dji_full.parquet")
df = df.rename(columns={[c for c in df.columns if "time" in c.lower()][0]: "time"})
df["time"] = pd.to_datetime(df["time"]); df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
if "tick_volume" not in df.columns: df["tick_volume"] = df.get("volume", 0)
feat = ep.compute_edge_features(df); log("features done")
atr = feat["atr14"].to_numpy(float); cdir = feat["committed_dir"].to_numpy(np.int64)
da = np.abs(feat["dist_at_signal"].to_numpy(float))
O = df["open"].to_numpy(float); H = df["high"].to_numpy(float); L = df["low"].to_numpy(float); C = df["close"].to_numpy(float)
times = df["time"].values; n = len(df)
ok = np.isfinite(atr) & (atr > 0) & (cdir != 0); ok[:300] = False; ok[-301:] = False
idx = np.where((da <= 1.0) & ok)[0]; dirs = cdir[idx].astype(np.int64)
ct = pd.to_datetime(times[idx]); sig_atr = atr[idx]
Xc = np.nan_to_num(feat[FC].to_numpy(np.float32)[idx], nan=0.0, posinf=0.0, neginf=0.0)
del feat

pnlB, ebB, xitB, codeB = sim_base(idx, dirs, O, H, L, C, atr, n, 7.0, 2.0, 300, TA, TT)
SNAP = {"d2r": snapshots(idx, dirs, O, H, L, C, atr, ebB, xitB, n, 1, 0, 2.0),
        "d30": snapshots(idx, dirs, O, H, L, C, atr, ebB, xitB, n, 0, 30, 0.0)}
EXITS = {"t2": (2.0, 2.0, 300, 0, 0.0), "tt2": (2.0, 2.0, 300, 30, 0.75), "tt15": (1.5, 1.5, 300, 30, 0.75)}
HG = {(s, e): sim_hedge(SNAP[s][0], dirs, O, H, L, C, atr, n, *cfg)
      for s in SNAP for e, cfg in EXITS.items()}
log("sims done")

XGBR = dict(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.85,
            colsample_bytree=0.85, min_child_weight=10, n_jobs=-1, random_state=0)
rng = np.random.RandomState(0)
WINS = []; tsw = pd.Timestamp("2020-07-01"); lastd = ct.max()
while tsw < lastd:
    WINS.append((tsw - pd.DateOffset(years=3), tsw, tsw + pd.DateOffset(months=6))); tsw += pd.DateOffset(months=6)

rows = []
for tr_s, te_s, te_e in WINS:
    trm = (ct >= tr_s) & (ct < te_s); tem = (ct >= te_s) & (ct < te_e)
    if trm.sum() < 4000 or tem.sum() < 20: continue
    tix = np.where(trm)[0]; tix_f = tix if len(tix) <= 150_000 else rng.choice(tix, 150_000, replace=False)
    tr_days = max((ct[tix].max() - ct[tix].min()).days * 5 / 7, 1)
    mg = XGBRegressor(**XGBR); mg.fit(Xc[tix_f], pnlB[tix_f])
    p = mg.predict(Xc).astype(np.float64)
    cand = np.quantile(p[tix], np.linspace(0.30, 0.97, 24)); thr = cand[-1]; best = 1e18
    for th in cand:
        kk = tix[p[tix] >= th]
        if len(kk) < 5: continue
        tk = take(kk[np.argsort(ebB[kk])].astype(np.int64), ebB, xitB, COOLDOWN)
        gap = abs(len(tk) / tr_days - 11.0)
        if gap < best: best = gap; thr = th
    kkte = np.where(tem & (p >= thr))[0]
    tk_te = take(kkte[np.argsort(ebB[kkte])].astype(np.int64), ebB, xitB, COOLDOWN)
    sb2, f30 = SNAP["d2r"][0], SNAP["d30"][1]
    sb30 = SNAP["d30"][0]
    sels = {"d2r": tk_te[sb2[tk_te] >= 0], "d30uw": tk_te[(sb30[tk_te] >= 0) & (f30[tk_te] < 0)]}
    for tnm, sel in sels.items():
        snm = "d2r" if tnm == "d2r" else "d30"
        for enm in EXITS:
            p_, a_, v_ = HG[(snm, enm)]
            s = sel[v_[sel] == 1]
            for sp in SPREADS:
                r = p_[s] - sp / a_[s]
                rows.append(dict(win=str(te_s.date()), trig=tnm, exit=enm, sp=sp,
                                 n=int(len(s)), net=float(r.sum())))
    log(f"window {te_s.date()} done")

R = pd.DataFrame(rows); R["dev"] = pd.to_datetime(R["win"]) < DEV_END
for dev, lab in [(True, "DEV"), (False, "HOLDOUT")]:
    print(f"\n===== DJI hedge spread grid — {lab} =====")
    print(f"{'trig':<8}{'exit':<7}" + "".join(f"{f'net@{s}pt':>12}" for s in SPREADS) + f"{'w+@2.0':>8}")
    for (tnm, enm), g in R[R.dev == dev].groupby(["trig", "exit"]):
        nets = [g[g.sp == sp].net.sum() for sp in SPREADS]
        w2 = g[(g.sp == 2.0) & (g.n > 0)]
        print(f"{tnm:<8}{enm:<7}" + "".join(f"{x:>+12.0f}" for x in nets) +
              f"{(w2.net > 0).sum():>5}/{len(w2)}")
R.to_json(OUT / "hedge_spread_results.json")
log("spread grid done")
