"""
LEG-POSITION LAB — Jay's hypothesis: losses come from entering LATE in the leg
(buys at the top of the up-move / sells at the bottom). Testable: for every trade
the DEPLOYED gate takes (XAU M1 + DJI M1, tt exits), measure causal entry-stretch:
  ext_atr  = (close - min(low, 240 bars)) / ATR   for buys   (mirror for sells)
             -> how much of the recent leg has already run, in ATR
  rng_pos  = position of close in the 240-bar high-low range (0=bottom, 1=top)
  leg_age  = bars since the 240-bar extreme the trade is riding from
Then: net-R by ext_atr quintile (DEV), monotonicity check, and a stretch-cut gate
(skip entries above the dev-chosen cutoff) confirmed on the untouched HOLDOUT.
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
DEV_END = pd.Timestamp("2025-01-01"); COOLDOWN = 5; LOOK = 240

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

@njit(cache=True)
def leg_feats(idxs, dirs, H, L, C, atr, look):
    m = len(idxs)
    ext = np.zeros(m); rng = np.zeros(m); age = np.zeros(m)
    for k in range(m):
        i = idxs[k]; d = dirs[k]; a = atr[i]
        s = max(0, i - look)
        lo = 1e18; hi = -1e18; loi = i; hii = i
        for j in range(s, i + 1):
            if L[j] < lo: lo = L[j]; loi = j
            if H[j] > hi: hi = H[j]; hii = j
        if d == 1:
            ext[k] = (C[i] - lo) / a; age[k] = i - loi
        else:
            ext[k] = (hi - C[i]) / a; age[k] = i - hii
        rr = hi - lo
        p = (C[i] - lo) / rr if rr > 0 else 0.5
        rng[k] = p if d == 1 else 1.0 - p
    return ext, rng, age

XGB = dict(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.85,
           colsample_bytree=0.85, min_child_weight=10, n_jobs=-1, random_state=0)
rng_ = np.random.RandomState(0)

for name, pq, sp_usd in [("xau_m1", "/home/jay/Desktop/new-model-zigzag/data/m1_xau_full.parquet", 0.20),
                         ("dji_m1", "/home/jay/Desktop/new-model-zigzag/data/m1_dji_full.parquet", 1.5)]:
    log(f"=== {name} ===")
    df = pd.read_parquet(pq)
    df = df.rename(columns={[c for c in df.columns if "time" in c.lower()][0]: "time"})
    df["time"] = pd.to_datetime(df["time"]); df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    if "tick_volume" not in df.columns: df["tick_volume"] = df.get("volume", 0)
    feat = ep.compute_edge_features(df)
    atr = feat["atr14"].to_numpy(float); cdir = feat["committed_dir"].to_numpy(np.int64)
    da = np.abs(feat["dist_at_signal"].to_numpy(float))
    O = df["open"].to_numpy(float); H = df["high"].to_numpy(float); L = df["low"].to_numpy(float); C = df["close"].to_numpy(float)
    n = len(df)
    ok = np.isfinite(atr) & (atr > 0) & (cdir != 0); ok[:300] = False; ok[-301:] = False
    idx = np.where((da <= 1.0) & ok)[0]; dirs = cdir[idx].astype(np.int64)
    ct = pd.to_datetime(df["time"].values[idx]); sig_atr = atr[idx]
    Xc = np.nan_to_num(feat[FC].to_numpy(np.float32)[idx], nan=0.0, posinf=0.0, neginf=0.0)
    del feat
    pnl, ebar, xit = sim_tt(idx, dirs, O, H, L, C, atr, n, 7.0, 2.0, 300, 30, 0.75)
    EXT, RNG, AGE = leg_feats(idx, dirs, H, L, C, atr, LOOK)
    WINS = []; tsw = pd.Timestamp("2020-07-01"); lastd = ct.max()
    while tsw < lastd:
        WINS.append((tsw - pd.DateOffset(years=3), tsw, tsw + pd.DateOffset(months=6))); tsw += pd.DateOffset(months=6)
    taken = []
    for tr_s, te_s, te_e in WINS:
        trm = (ct >= tr_s) & (ct < te_s); tem = (ct >= te_s) & (ct < te_e)
        if trm.sum() < 4000 or tem.sum() < 20: continue
        tix = np.where(trm)[0]; tix_f = tix if len(tix) <= 150_000 else rng_.choice(tix, 150_000, replace=False)
        tr_days = max((ct[tix].max() - ct[tix].min()).days * 5 / 7, 1)
        m = XGBRegressor(**XGB); m.fit(Xc[tix_f], pnl[tix_f])
        p = m.predict(Xc).astype(np.float64)
        cand = np.quantile(p[tix], np.linspace(0.30, 0.97, 24)); thr = cand[-1]; best = 1e18
        for th in cand:
            kk = tix[p[tix] >= th]
            if len(kk) < 5: continue
            tk = take(kk[np.argsort(ebar[kk])].astype(np.int64), ebar, xit, COOLDOWN)
            gap = abs(len(tk) / tr_days - 11.0)
            if gap < best: best = gap; thr = th
        kk = np.where(tem & (p >= thr))[0]
        tk = take(kk[np.argsort(ebar[kk])].astype(np.int64), ebar, xit, COOLDOWN)
        for k in tk: taken.append(k)
        log(f"  window {te_s.date()} done")
    taken = np.array(taken)
    net = pnl[taken] - sp_usd / sig_atr[taken]
    dev = ct[taken] < DEV_END
    res = {"name": name}
    print(f"\n----- {name}: net R by ENTRY-STRETCH quintile (DEV, n={int(dev.sum())}) -----")
    q = np.quantile(EXT[taken][dev], [0.2, 0.4, 0.6, 0.8])
    print(f"  ext_atr quintile edges: {np.round(q, 1)}")
    for lo, hi, tag in [(-1, q[0], "Q1 early-leg"), (q[0], q[1], "Q2"), (q[1], q[2], "Q3"),
                        (q[2], q[3], "Q4"), (q[3], 1e9, "Q5 stretched")]:
        for lab_, mask in [("dev", dev), ("hold", ~dev)]:
            mm = mask & (EXT[taken] > lo) & (EXT[taken] <= hi)
            if mm.sum() < 30: continue
            print(f"  {tag:<14}{lab_:<5} n={int(mm.sum()):>5}  mean {net[mm].mean():+.3f}R  WR {(net[mm]>0).mean()*100:4.1f}%")
    # gate test: cut the worst dev quintile(s) if negative
    for cut_q, cutlab in [(q[3], "cut Q5"), (q[2], "cut Q4+Q5")]:
        for lab_, mask in [("dev", dev), ("hold", ~dev)]:
            keep = mask & (EXT[taken] <= cut_q)
            print(f"  GATE {cutlab:<10}{lab_:<5} n={int(keep.sum()):>5}  sum {net[keep].sum():+8.0f}R "
                  f"(vs all {net[mask].sum():+8.0f}R)  per-trade {net[keep].mean():+.3f} vs {net[mask].mean():+.3f}")
    # same by range-position
    print(f"  --- rng_pos (0=early side, 1=entering at extreme) ---")
    for lo, hi in [(0.0, 0.5), (0.5, 0.8), (0.8, 1.01)]:
        for lab_, mask in [("dev", dev), ("hold", ~dev)]:
            mm = mask & (RNG[taken] >= lo) & (RNG[taken] < hi)
            if mm.sum() < 30: continue
            print(f"  rng {lo:.1f}-{hi:.1f}  {lab_:<5} n={int(mm.sum()):>5}  mean {net[mm].mean():+.3f}R  WR {(net[mm]>0).mean()*100:4.1f}%")
log("legpos lab done")
