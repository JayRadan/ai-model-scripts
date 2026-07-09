"""
BOTH-SIDES LAB — Jay: "remove the regime, allow always both sides".
Deployed edge_pullback only ever trades WITH the TFK committed direction
(dir = committed_dir). Test: at the SAME pullback bars (|dist_tfk|<=1.0),
also allow the OPPOSITE side and let the XGB gross-R gate pick.

Variants (each its own walk-forward model + train-only ~11/day threshold):
  base — dir = committed_dir (deployed behaviour, reference)
  anti — dir = -committed_dir only (pure fade — does counter-regime have edge?)
  both — union of the two; X gains 2 cols: trade_dir, pro(=trade_dir*cdir);
         1-slot take() with ties broken by higher predicted R.
Exits identical everywhere: SL7 / trail2 / tt30/0.75 / maxh300. Net at live
spread (XAU $0.20, DJI 1.5pt). DEV(<2025) select / HOLDOUT confirm.
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
COOLDOWN = 5; DEV_END = pd.Timestamp("2025-01-01"); TA, TT = 30, 0.75

@njit(cache=True)
def sim_base(idxs, dirs, O, H, L, C, atr, n, SL, TRAIL, MAXH, ta, tt):
    m = len(idxs); pnl = np.zeros(m); ebar = np.full(m, -1, np.int64)
    xit = np.full(m, -1, np.int64)
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

XGBR = dict(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.85,
            colsample_bytree=0.85, min_child_weight=10, n_jobs=-1, random_state=0)
rng = np.random.RandomState(0)
ALL = {}

for name, pq, sp_usd in [("xau_m1", "/home/jay/Desktop/new-model-zigzag/data/m1_xau_full.parquet", 0.20),
                         ("dji_m1", "/home/jay/Desktop/new-model-zigzag/data/m1_dji_full.parquet", 1.5)]:
    log(f"=== {name} ===")
    df = pd.read_parquet(pq)
    df = df.rename(columns={[c for c in df.columns if "time" in c.lower()][0]: "time"})
    df["time"] = pd.to_datetime(df["time"]); df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    if "tick_volume" not in df.columns: df["tick_volume"] = df.get("volume", 0)
    feat = ep.compute_edge_features(df); log("features done")
    atr = feat["atr14"].to_numpy(float); cdir = feat["committed_dir"].to_numpy(np.int64)
    da = np.abs(feat["dist_at_signal"].to_numpy(float))
    O = df["open"].to_numpy(float); H = df["high"].to_numpy(float); L = df["low"].to_numpy(float); C = df["close"].to_numpy(float)
    times = df["time"].values; n = len(df)
    ok = np.isfinite(atr) & (atr > 0) & (cdir != 0); ok[:300] = False; ok[-301:] = False
    idx0 = np.where((da <= 1.0) & ok)[0]; d0 = cdir[idx0].astype(np.int64)
    Xc0 = np.nan_to_num(feat[FC].to_numpy(np.float32)[idx0], nan=0.0, posinf=0.0, neginf=0.0)
    del feat
    m0 = len(idx0)
    # combined candidate set: [0:m0)=pro (deployed), [m0:2m0)=anti
    idx2 = np.concatenate([idx0, idx0]); dirs2 = np.concatenate([d0, -d0])
    pro2 = np.concatenate([np.ones(m0, np.float32), -np.ones(m0, np.float32)])
    X2 = np.concatenate(
        [np.vstack([Xc0, Xc0]), dirs2.astype(np.float32)[:, None], pro2[:, None]], axis=1)
    del Xc0
    ct2 = pd.to_datetime(np.concatenate([times[idx0], times[idx0]]))
    sa2 = np.concatenate([atr[idx0], atr[idx0]])
    pnl2, eb2, xt2 = sim_base(idx2, dirs2, O, H, L, C, atr, n, 7.0, 2.0, 300, TA, TT)
    log(f"sims done: {m0} pullback bars -> {2*m0} candidates")

    SETS = {"base": np.arange(m0), "anti": np.arange(m0, 2 * m0), "both": np.arange(2 * m0)}
    WINS = []; tsw = pd.Timestamp("2020-07-01"); lastd = ct2.max()
    while tsw < lastd:
        WINS.append((tsw - pd.DateOffset(years=3), tsw, tsw + pd.DateOffset(months=6))); tsw += pd.DateOffset(months=6)
    res = {v: [] for v in SETS}
    for tr_s, te_s, te_e in WINS:
        trm = (ct2 >= tr_s) & (ct2 < te_s); tem = (ct2 >= te_s) & (ct2 < te_e)
        for vn, sub in SETS.items():
            trs = sub[trm[sub]]; tes = sub[tem[sub]]
            if len(trs) < 4000 or len(tes) < 20: continue
            trf = trs if len(trs) <= 150_000 else rng.choice(trs, 150_000, replace=False)
            tr_days = max((ct2[trs].max() - ct2[trs].min()).days * 5 / 7, 1)
            mg = XGBRegressor(**XGBR); mg.fit(X2[trf], pnl2[trf])
            pv = np.full(2 * m0, -1e9)
            pv[sub] = mg.predict(X2[sub]).astype(np.float64)
            cand = np.quantile(pv[trs], np.linspace(0.30, 0.97, 24)); thr = cand[-1]; best = 1e18
            for th in cand:
                kk = trs[pv[trs] >= th]
                if len(kk) < 5: continue
                tk = take(kk[np.lexsort((-pv[kk], eb2[kk]))].astype(np.int64), eb2, xt2, COOLDOWN)
                gap = abs(len(tk) / tr_days - 11.0)
                if gap < best: best = gap; thr = th
            kk = tes[pv[tes] >= thr]
            tk = take(kk[np.lexsort((-pv[kk], eb2[kk]))].astype(np.int64), eb2, xt2, COOLDOWN)
            net = pnl2[tk] - sp_usd / sa2[tk]
            res[vn].append(dict(win=str(te_s.date()), n=len(tk), net=float(net.sum()),
                                wr=float((net > 0).mean()) if len(tk) else 0.0,
                                n_anti=int((tk >= m0).sum()),
                                net_anti=float(net[tk >= m0].sum()),
                                net_pro=float(net[tk < m0].sum())))
        log(f"window {te_s.date()} done")

    def agg(rows, dev):
        rr = [r for r in rows if (pd.Timestamp(r["win"]) < DEV_END) == dev and r["n"] > 0]
        if not rr: return None
        return dict(nwin=len(rr), n=sum(r["n"] for r in rr), net=sum(r["net"] for r in rr),
                    w=sum(1 for r in rr if r["net"] > 0),
                    n_anti=sum(r["n_anti"] for r in rr),
                    net_anti=sum(r["net_anti"] for r in rr), net_pro=sum(r["net_pro"] for r in rr))
    for dev, lab in [(True, "DEV"), (False, "HOLDOUT")]:
        print(f"\n{'='*72}\nBOTH-SIDES LAB {name} — {lab}  (net @ live spread)\n{'='*72}")
        print(f"{'variant':<8}{'n':>7}{'netR':>9}{'w+':>7}{'n_anti':>8}{'net_anti':>10}{'net_pro':>9}")
        for vn in SETS:
            a = agg(res[vn], dev)
            if a is None: continue
            print(f"{vn:<8}{a['n']:>7}{a['net']:>+9.0f}{a['w']:>4}/{a['nwin']:<3}"
                  f"{a['n_anti']:>8}{a['net_anti']:>+10.0f}{a['net_pro']:>+9.0f}")
    ALL[name] = res
    del X2

json.dump(ALL, open(OUT / "bothsides_results.json", "w"), default=str, indent=1)
log("bothsides lab done")
