"""
REGIME-TIMEFRAME LAB — Jay: "change regime timeframe to lower, to 5m".
Deployed M1 products compute TFK regime (committed_dir + dist_tfk pullback) on
the SAME M1 bars they execute on. Test: regime from M5 bars (faster commitment
flips in wall-clock terms? actually SLOWER per-bar — this measures which info
set is right), execution still M1 with M1 ATR/exits.

Variants (all q10 gate = the go-forward objective, tt30/0.75 exits, ~11/day):
  base    — deployed: committed_dir + |dist_tfk|<=1.0 from M1 features
  reg5    — candidates where M5 committed_dir!=0 AND M5 |dist_tfk|<=1.0
            (mapped causally: only COMPLETED M5 buckets, close-time asof)
  reg5dir — M1 pullback bars (deployed candidate def) but direction taken from
            the M5 committed_dir (skip when M5 uncommitted / disagrees)
Model input = 29 live M1 feats at the candidate bar (unchanged).
DEV(<2025) select / HOLDOUT confirm, live spreads. XAU M1 + DJI M1.
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
COOLDOWN = 5; DEV_END = pd.Timestamp("2025-01-01"); TA, TT = 30, 0.75

@njit(cache=True)
def sim_tt(idxs, dirs, O, H, L, C, atr, n, SL, TRAIL, MAXH, ta, tt):
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

XGBQ = dict(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.85,
            colsample_bytree=0.85, min_child_weight=10, n_jobs=-1, random_state=0,
            objective="reg:quantileerror", quantile_alpha=0.10)
rng = np.random.RandomState(0)
ALL = {}

for name, pq, sp_usd in [("xau_m1", "/home/jay/Desktop/new-model-zigzag/data/m1_xau_full.parquet", 0.20),
                         ("dji_m1", "/home/jay/Desktop/new-model-zigzag/data/m1_dji_full.parquet", 1.5)]:
    log(f"=== {name} ===")
    df = pd.read_parquet(pq)
    df = df.rename(columns={[c for c in df.columns if "time" in c.lower()][0]: "time"})
    df["time"] = pd.to_datetime(df["time"]); df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    if "tick_volume" not in df.columns: df["tick_volume"] = df.get("volume", 0)
    feat = ep.compute_edge_features(df); log("M1 features done")
    atr = feat["atr14"].to_numpy(float); cdir1 = feat["committed_dir"].to_numpy(np.int64)
    da1 = np.abs(feat["dist_at_signal"].to_numpy(float))
    O = df["open"].to_numpy(float); H = df["high"].to_numpy(float); L = df["low"].to_numpy(float); C = df["close"].to_numpy(float)
    times = df["time"].values; n = len(df)
    FEAT1 = np.nan_to_num(feat[FC].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    del feat

    # M5 regime, causal mapping: bucket labeled b covers [b, b+5), closes at the
    # M1 bar labeled b+4. Usable from that bar's close onward.
    d5 = ep._resample(df, 5)
    f5 = ep.compute_edge_features(d5)
    cdir5_ = f5["committed_dir"].to_numpy(np.int64)
    da5_ = np.abs(f5["dist_at_signal"].to_numpy(float))
    close5 = (pd.to_datetime(d5["time"]) + pd.Timedelta(minutes=4)).values
    pos = np.searchsorted(close5, times, side="right") - 1   # last COMPLETED bucket
    valid5 = pos >= 0
    cdir5 = np.where(valid5, cdir5_[np.clip(pos, 0, None)], 0)
    da5 = np.where(valid5, da5_[np.clip(pos, 0, None)], 9e9)
    del d5, f5
    log("M5 regime mapped")

    ok = np.isfinite(atr) & (atr > 0); ok[:300] = False; ok[-301:] = False
    CANDS = {
        "base":    (np.where(ok & (cdir1 != 0) & (da1 <= 1.0))[0], cdir1),
        "reg5":    (np.where(ok & (cdir5 != 0) & (da5 <= 1.0) & np.isfinite(da5))[0], cdir5),
        "reg5dir": (np.where(ok & (cdir1 != 0) & (da1 <= 1.0) & (cdir5 != 0))[0], cdir5),
    }
    res = {v: [] for v in CANDS}
    for vn, (idx, cs) in CANDS.items():
        dirs = cs[idx].astype(np.int64)
        ct = pd.to_datetime(times[idx]); sig_atr = atr[idx]
        Xc = FEAT1[idx]
        pnl, eb, xt, code = sim_tt(idx, dirs, O, H, L, C, atr, n, 7.0, 2.0, 300, TA, TT)
        log(f"  {vn}: {len(idx):,} candidates")
        WINS = []; tsw = pd.Timestamp("2020-07-01"); lastd = ct.max()
        while tsw < lastd:
            WINS.append((tsw - pd.DateOffset(years=3), tsw, tsw + pd.DateOffset(months=6))); tsw += pd.DateOffset(months=6)
        for tr_s, te_s, te_e in WINS:
            trm = (ct >= tr_s) & (ct < te_s); tem = (ct >= te_s) & (ct < te_e)
            if trm.sum() < 4000 or tem.sum() < 20: continue
            tix = np.where(trm)[0]; tix_f = tix if len(tix) <= 150_000 else rng.choice(tix, 150_000, replace=False)
            tr_days = max((ct[tix].max() - ct[tix].min()).days * 5 / 7, 1)
            m = XGBRegressor(**XGBQ); m.fit(Xc[tix_f], pnl[tix_f])
            p = m.predict(Xc).astype(np.float64)
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
        log(f"  {vn} windows done")

    def agg(rows, dev):
        rr = [r for r in rows if (pd.Timestamp(r["win"]) < DEV_END) == dev and r["n"] > 0]
        if not rr: return None
        nn = sum(r["n"] for r in rr)
        return dict(nwin=len(rr), n=nn, net=sum(r["net"] for r in rr),
                    w=sum(1 for r in rr if r["net"] > 0),
                    slr=100.0 * sum(r["sl"] for r in rr) / nn,
                    wr=100.0 * np.average([r["wr"] for r in rr], weights=[r["n"] for r in rr]))
    for dev, lab in [(True, "DEV"), (False, "HOLDOUT")]:
        print(f"\n{'='*70}\nREGIME-TF LAB {name} — {lab}  (q10 gate, net @ live spread)\n{'='*70}")
        print(f"{'variant':<9}{'n':>7}{'netR':>9}{'w+':>7}{'SLhit%':>8}{'WR%':>7}")
        for vn in CANDS:
            a = agg(res[vn], dev)
            if a is None: continue
            print(f"{vn:<9}{a['n']:>7}{a['net']:>+9.0f}{a['w']:>4}/{a['nwin']:<3}{a['slr']:>8.2f}{a['wr']:>7.1f}")
    ALL[name] = res
    del FEAT1

json.dump(ALL, open(OUT / "regime_tf_results.json", "w"), default=str, indent=1)
log("regime-tf lab done")
