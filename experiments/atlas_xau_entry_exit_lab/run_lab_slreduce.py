"""
SL-REDUCE LAB — Jay: "more accurate, fewer trades that hit SL, find a way".
Entry-time doom is noise in the CURRENT feature space (AUC 0.53 x3 replications),
so the only legitimate levers are (1) NEW INFORMATION and (2) BETTER SELECTION MATH:

  base      — deployed 29 feats, gate on predicted MEAN gross R (reference)
  ens       — 29 feats, 4 bootstrap XGBs, gate on mean - std (skip trades the
              ensemble disagrees on: epistemic-uncertainty veto of the fat left tail)
  q10       — 29 feats, XGB quantile regression alpha=0.1, gate on predicted
              10th-percentile R (downside-robust selection instead of mean)
  macro     — 29 + 14 cross-asset feats: DXY/SPX/BND/(DJI or XAU) 30m+240m
              z-returns + 240m rolling correlation, + own variance-ratio and
              return-autocorr. Information the model has NEVER seen.
  macro_ens — macro features + ensemble mean-std gate

All variants calibrated on TRAIN only to ~11 trades/day, identical tt30/0.75
exits, 1-slot cd5. Cross series aligned merge_asof backward + shift(1) bar (no
look-ahead). Reports net R AND SL-hit rate. DEV(<2025) select / HOLDOUT confirm.
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
DATA = Path("/home/jay/Desktop/new-model-zigzag/data")
FC = pickle.load(open(SRV / "decision_engine/models/atlas_xau_validated.pkl", "rb"))["feat_cols"]
t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
COOLDOWN = 5; DEV_END = pd.Timestamp("2025-01-01"); TA, TT = 30, 0.75; NBAG = 4

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
def take(order_idx, ebar, xit, cd):
    busy = -1; m = len(order_idx); out = np.empty(m, np.int64); c = 0
    for t in range(m):
        k = order_idx[t]
        if ebar[k] <= busy: continue
        out[c] = k; busy = xit[k] + cd; c += 1
    return out[:c]

def load_m1(fname):
    df = pd.read_parquet(DATA / fname)
    tc = [c for c in df.columns if "time" in c.lower()][0]
    df = df.rename(columns={tc: "time"})
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time").drop_duplicates("time").reset_index(drop=True)

def cross_feats(own_time, own_close, drivers):
    """14 cross/stat features aligned to own bar times, causal (shift 1 bar)."""
    base = pd.DataFrame({"time": own_time})
    ret_own = pd.Series(own_close).pct_change()
    cols = {}
    for nm, ddf in drivers.items():
        s = ddf[["time", "close"]].copy()
        s["close"] = s["close"].shift(1)                      # driver close known strictly before own bar close
        al = pd.merge_asof(base, s, on="time", tolerance=pd.Timedelta("30min"), direction="backward")["close"]
        r1 = al.pct_change()
        sd = r1.rolling(1440, min_periods=300).std()
        cols[f"{nm}_ret30z"] = (al / al.shift(30) - 1) / (sd * np.sqrt(30))
        cols[f"{nm}_ret240z"] = (al / al.shift(240) - 1) / (sd * np.sqrt(240))
        cols[f"{nm}_corr240"] = r1.rolling(240, min_periods=60).corr(ret_own)
    # own distribution-shape stats
    v5 = ret_own.rolling(5).std(); v30 = ret_own.rolling(30).std()
    cols["vr_30_5"] = v30 / (v5 * np.sqrt(6))
    cols["ac_120"] = ret_own.rolling(120, min_periods=60).corr(ret_own.shift(1))
    X = pd.DataFrame(cols)
    return np.nan_to_num(X.to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0), list(X.columns)

XGBR = dict(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.85,
            colsample_bytree=0.85, min_child_weight=10, n_jobs=-1, random_state=0)
XGBQ = dict(XGBR, objective="reg:quantileerror", quantile_alpha=0.10)
rng = np.random.RandomState(0)
ALL = {}

for name, pq, sp_usd, driver_files in [
        ("xau_m1", "m1_xau_full.parquet", 0.20,
         {"dxy": "m1_dxy_full.parquet", "spx": "m1_spx_full.parquet",
          "bnd": "m1_bnd_full.parquet", "dji": "m1_dji_full.parquet"}),
        ("dji_m1", "m1_dji_full.parquet", 1.5,
         {"dxy": "m1_dxy_full.parquet", "spx": "m1_spx_full.parquet",
          "bnd": "m1_bnd_full.parquet", "xau": "m1_xau_full.parquet"})]:
    log(f"=== {name} ===")
    df = load_m1(pq)
    if "tick_volume" not in df.columns: df["tick_volume"] = df.get("volume", 0)
    feat = ep.compute_edge_features(df); log("features done")
    atr = feat["atr14"].to_numpy(float); cdir = feat["committed_dir"].to_numpy(np.int64)
    da = np.abs(feat["dist_at_signal"].to_numpy(float))
    O = df["open"].to_numpy(float); H = df["high"].to_numpy(float); L = df["low"].to_numpy(float); C = df["close"].to_numpy(float)
    times = df["time"].values; n = len(df)
    drivers = {nm: load_m1(f) for nm, f in driver_files.items()}
    XM_all, mcols = cross_feats(df["time"], df["close"], drivers)
    del drivers
    log(f"cross feats done: {mcols}")
    ok = np.isfinite(atr) & (atr > 0) & (cdir != 0); ok[:300] = False; ok[-301:] = False
    idx = np.where((da <= 1.0) & ok)[0]; dirs = cdir[idx].astype(np.int64)
    ct = pd.to_datetime(times[idx]); sig_atr = atr[idx]
    X29 = np.nan_to_num(feat[FC].to_numpy(np.float32)[idx], nan=0.0, posinf=0.0, neginf=0.0)
    XMA = np.concatenate([X29, XM_all[idx]], axis=1)
    del feat, XM_all
    pnl, eb, xt, code = sim_base(idx, dirs, O, H, L, C, atr, n, 7.0, 2.0, 300, TA, TT)
    log(f"sim done: {len(idx)} candidates")

    WINS = []; tsw = pd.Timestamp("2020-07-01"); lastd = ct.max()
    while tsw < lastd:
        WINS.append((tsw - pd.DateOffset(years=3), tsw, tsw + pd.DateOffset(months=6))); tsw += pd.DateOffset(months=6)

    VAR = ("base", "ens", "q10", "macro", "macro_ens")
    res = {v: [] for v in VAR}
    for tr_s, te_s, te_e in WINS:
        trm = (ct >= tr_s) & (ct < te_s); tem = (ct >= te_s) & (ct < te_e)
        if trm.sum() < 4000 or tem.sum() < 20: continue
        tix = np.where(trm)[0]; tix_f = tix if len(tix) <= 150_000 else rng.choice(tix, 150_000, replace=False)
        tr_days = max((ct[tix].max() - ct[tix].min()).days * 5 / 7, 1)

        scores = {}
        m0 = XGBRegressor(**XGBR); m0.fit(X29[tix_f], pnl[tix_f])
        scores["base"] = m0.predict(X29).astype(np.float64)
        mq = XGBRegressor(**XGBQ); mq.fit(X29[tix_f], pnl[tix_f])
        scores["q10"] = mq.predict(X29).astype(np.float64)
        mm = XGBRegressor(**XGBR); mm.fit(XMA[tix_f], pnl[tix_f])
        scores["macro"] = mm.predict(XMA).astype(np.float64)
        for vn, XX in [("ens", X29), ("macro_ens", XMA)]:
            preds = []
            for b in range(NBAG):
                bi = rng.choice(len(tix_f), len(tix_f), replace=True)
                mb = XGBRegressor(**dict(XGBR, random_state=100 + b))
                mb.fit(XX[tix_f][bi], pnl[tix_f][bi])
                preds.append(mb.predict(XX).astype(np.float64))
            P = np.stack(preds)
            scores[vn] = P.mean(0) - P.std(0)
        for vn in VAR:
            p = scores[vn]
            cand = np.quantile(p[tix], np.linspace(0.30, 0.97, 24)); thr = cand[-1]; best = 1e18
            for th in cand:
                kk = tix[p[tix] >= th]
                if len(kk) < 5: continue
                tk = take(kk[np.argsort(eb[kk])].astype(np.int64), eb, xt, COOLDOWN)
                gap = abs(len(tk) / tr_days - 11.0)
                if gap < best: best = gap; thr = th
            kk = np.where(tem & (p >= thr))[0]
            tk = take(kk[np.argsort(eb[kk])].astype(np.int64), eb, xt, COOLDOWN)
            netv = pnl[tk] - sp_usd / sig_atr[tk]
            res[vn].append(dict(win=str(te_s.date()), n=len(tk), net=float(netv.sum()),
                                sl=int((code[tk] == 0).sum()),
                                wr=float((netv > 0).mean()) if len(tk) else 0.0))
        log(f"window {te_s.date()} done")

    def agg(rows, dev):
        rr = [r for r in rows if (pd.Timestamp(r["win"]) < DEV_END) == dev and r["n"] > 0]
        if not rr: return None
        n = sum(r["n"] for r in rr)
        return dict(nwin=len(rr), n=n, net=sum(r["net"] for r in rr),
                    w=sum(1 for r in rr if r["net"] > 0),
                    slr=100.0 * sum(r["sl"] for r in rr) / n,
                    wr=100.0 * np.average([r["wr"] for r in rr], weights=[r["n"] for r in rr]))
    for dev, lab in [(True, "DEV"), (False, "HOLDOUT")]:
        print(f"\n{'='*70}\nSL-REDUCE LAB {name} — {lab}  (net @ live spread)\n{'='*70}")
        print(f"{'variant':<11}{'n':>7}{'netR':>9}{'w+':>7}{'SLhit%':>8}{'WR%':>7}")
        for vn in VAR:
            a = agg(res[vn], dev)
            if a is None: continue
            print(f"{vn:<11}{a['n']:>7}{a['net']:>+9.0f}{a['w']:>4}/{a['nwin']:<3}{a['slr']:>8.2f}{a['wr']:>7.1f}")
    ALL[name] = res
    del X29, XMA

json.dump(ALL, open(OUT / "slreduce_results.json", "w"), default=str, indent=1)
log("slreduce lab done")
