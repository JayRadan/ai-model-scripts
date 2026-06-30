"""
Hermes DJI — 8-YEAR WALK-FORWARD of the CAUSAL retrain.

Uses local data/m1_dji_full.parquet (2018-2026, no re-fetch). Computes causal
features ONCE, then rolls: train 3y / test 6m / step 6m. Each window trains a
fresh XGB on causal features and runs the 1-slot live exit engine OOS.

Pass = most windows PF>1.3 and few losing windows → robust enough to deploy.
"""
import sys, pickle, time
from pathlib import Path
import numpy as np, pandas as pd
from numba import njit

DE = Path("/home/jay/Desktop/my-agents-and-website/commercial/server/decision_engine")
sys.path.insert(0, str(DE))
import hermes_features as hf
from configs.hermes_dji import HERMES_DJI as CFG
FC = pickle.load(open(DE / "models/hermes_dji_validated.pkl", "rb"))["feat_cols"]
from xgboost import XGBRegressor

t0 = time.time()
df = pd.read_parquet("/home/jay/Desktop/new-model-zigzag/data/m1_dji_full.parquet")
df = df.rename(columns={[c for c in df.columns if "time" in c.lower()][0]: "time"})
df["time"] = pd.to_datetime(df["time"]); df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
if "tick_volume" not in df.columns: df["tick_volume"] = df.get("volume", 0)
print(f"loaded {len(df):,} bars {df.time.iloc[0]} → {df.time.iloc[-1]}  ({time.time()-t0:.0f}s)", flush=True)

print("computing causal features over full history ...", flush=True)
feat = hf.compute_all_features(df.copy(), CFG)
_A = 2.0 / 51.0
s = df.set_index("time")["close"]
for nm, tfm in [("m5", 5), ("m15", 15), ("h1", 60)]:
    g = s.resample(f"{tfm}min").last().dropna(); hc = g.to_numpy(np.float64)
    et = (g.index + pd.Timedelta(minutes=tfm)).values
    n = len(df); rsi = np.full(n, np.nan); slope = np.full(n, np.nan); emad = np.full(n, np.nan)
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
print(f"  features done ({time.time()-t0:.0f}s)", flush=True)

atr = feat["atr14"].to_numpy(float); cdir = feat["committed_dir"].to_numpy(np.int64)
ds = feat["dist_at_signal"].to_numpy(float); da = np.abs(ds)
O = df["open"].to_numpy(float); H = df["high"].to_numpy(float); L = df["low"].to_numpy(float); C = df["close"].to_numpy(float)
times = df["time"].values; n = len(df); sp = CFG.spread_usd / np.nanmedian(atr)

is_pb = da <= CFG.near_thr; is_ct = (ds * cdir) <= -CFG.counter_thr
ok = np.isfinite(atr) & (atr > 0) & (cdir != 0); ok[:300] = False; ok[-301:] = False
cand = np.where(ok & (is_pb | is_ct))[0]; cdr = cdir[cand].astype(np.int64)

@njit
def lbl(idxs, dirs, O, H, L, C, atr, sp, SL, TRAIL, MAXH, n):
    m = len(idxs); pnl = np.empty(m)
    for k in range(m):
        i = idxs[k]; d = dirs[k]; a = atr[i]; ei = i + 1
        if ei >= n or not (a > 0): pnl[k] = 0.0; continue
        ep = O[ei]; hard = SL * a; trd = TRAIL * a; mf = 0.0; end = min(ei + MAXH, n - 1); done = False
        for jx in range(ei, end + 1):
            fav = d * (C[jx] - ep)
            if fav > mf: mf = fav
            if d == 1 and (ep - L[jx]) >= hard: pnl[k] = -SL - sp; done = True; break
            if d == -1 and (H[jx] - ep) >= hard: pnl[k] = -SL - sp; done = True; break
            if mf >= trd and (mf - fav) >= trd: pnl[k] = (mf - trd) / a - sp; done = True; break
        if not done: pnl[k] = d * (C[end] - ep) / a - sp
    return pnl

y = lbl(cand, cdr, O, H, L, C, atr, float(sp), 4.0, 3.0, 300, n)
X = feat[FC].to_numpy(np.float32)[cand]
ct = pd.to_datetime(times[cand])
print(f"candidates {len(cand):,}  ({time.time()-t0:.0f}s)", flush=True)

def portsim(sel, seld, selq):
    info = {int(sel[k]): (int(seld[k]), float(selq[k])) for k in range(len(sel))}
    if not info: return None
    SL, TRAIL, BE, MAXC, COOL = 4.0, 3.0, 1.0, 1, 5
    active, ex = [], []; last = {-1: -10**9, 1: -10**9}
    for i in range(min(info), min(max(info) + 301, n)):
        still = []
        for t in active:
            if i <= t["ei"]: still.append(t); continue
            if i > min(t["ei"] + 300, n - 1):
                cp = C[min(t["ei"] + 300, n - 1)]; t["R"] = float(t["d"] * (cp - t["ep"]) / t["a"] - sp); ex.append(t); continue
            d = t["d"]; ep = t["ep"]; a = t["a"]; fav = d * (C[i] - ep)
            if fav > t["mf"]: t["mf"] = fav
            hit = False
            if t["slr"] == 0:
                if d == 1 and L[i] <= ep: t["R"] = -sp; hit = True
                elif d == -1 and H[i] >= ep: t["R"] = -sp; hit = True
            else:
                dd = abs(t["slr"]) * a
                if d == 1 and (ep - L[i]) >= dd: t["R"] = float(t["slr"] - sp); hit = True
                elif d == -1 and (H[i] - ep) >= dd: t["R"] = float(t["slr"] - sp); hit = True
            if hit: ex.append(t); continue
            td = TRAIL * a
            if t["mf"] >= td and (t["mf"] - fav) >= td: t["R"] = float((t["mf"] - td) / a - sp); ex.append(t); continue
            still.append(t)
        active = still
        if i not in info: continue
        d_, q_ = info[i]
        if i - last[d_] < COOL: continue
        for t in active:
            if t["slr"] != 0 and t["d"] * (C[i] - t["ep"]) / t["a"] >= BE: t["slr"] = 0
        ei = i + 1
        if ei >= n or not (atr[i] > 0) or len(active) >= MAXC: continue
        active.append({"ei": ei, "d": d_, "ep": float(O[ei]), "a": float(atr[i]), "slr": -SL, "mf": 0.0, "R": None})
        last[d_] = i
    for t in active:
        eb = min(t["ei"] + 300, n - 1); t["R"] = float(t["d"] * (C[eb] - t["ep"]) / t["a"] - sp); ex.append(t)
    r = np.array([t["R"] for t in ex])
    if len(r) == 0: return None
    w = int((r > 0).sum()); pf = r[r > 0].sum() / max(-r[r <= 0].sum(), 1e-9)
    eq = np.cumsum(r); dd = float((np.maximum.accumulate(eq) - eq).max())
    return dict(n=len(r), sumR=float(r.sum()), wr=w / len(r) * 100, pf=float(pf), dd=dd)

# ── walk-forward: train 3y / test 6m / step 6m ───────────────────────────────
THR = 3.0
first_test = pd.Timestamp("2020-07-01"); last = ct.max()
windows = []
ts = first_test
while ts < last:
    te_end = ts + pd.DateOffset(months=6)
    tr_start = ts - pd.DateOffset(years=3)
    windows.append((tr_start, ts, te_end)); ts = te_end

print(f"\n{'='*86}\nHERMES DJI 8Y WALK-FORWARD (causal retrain, train3y/test6m, q>={THR}, 1-slot live exit)\n{'='*86}")
print(f"{'test window':<20}{'n_tr':>9}{'trades':>8}{'sumR':>9}{'WR':>7}{'PF':>7}{'DD':>7}")
agg = []
for tr_start, te_s, te_e in windows:
    trm = (ct >= tr_start) & (ct < te_s); tem = (ct >= te_s) & (ct < te_e)
    if trm.sum() < 5000 or tem.sum() < 20: continue
    m = XGBRegressor(n_estimators=600, max_depth=5, learning_rate=0.04, subsample=0.85,
                     colsample_bytree=0.85, min_child_weight=10, n_jobs=-1, random_state=0)
    m.fit(X[trm], y[trm])
    qp = m.predict(X[tem]); smask = qp >= THR
    res = portsim(cand[tem][smask], cdr[tem][smask], qp[smask])
    lbl_ = f"{te_s.date()}→{te_e.date()}"
    if not res:
        print(f"{lbl_:<20}{trm.sum():>9,}{0:>8}"); continue
    print(f"{lbl_:<20}{trm.sum():>9,}{res['n']:>8}{res['sumR']:>+9.1f}{res['wr']:>6.0f}%{res['pf']:>7.2f}{res['dd']:>7.0f}", flush=True)
    agg.append(res)

if agg:
    pfs = [a["pf"] for a in agg]; sr = sum(a["sumR"] for a in agg); nt = sum(a["n"] for a in agg)
    win = sum(1 for a in agg if a["sumR"] > 0)
    print(f"\n  {len(agg)} windows | total {nt} trades | sumR {sr:+.1f}")
    print(f"  windows PF>1: {sum(p>1 for p in pfs)}/{len(agg)}   PF>1.3: {sum(p>1.3 for p in pfs)}/{len(agg)}   profitable: {win}/{len(agg)}")
    print(f"  median PF {np.median(pfs):.2f}  | worst window sumR {min(a['sumR'] for a in agg):+.1f}  | best {max(a['sumR'] for a in agg):+.1f}")
    robust = (sum(p>1.3 for p in pfs)/len(agg) >= 0.6) and (win/len(agg) >= 0.7)
    print(f"\n  VERDICT: {'DEPLOY-WORTHY — robust across regimes' if robust else 'NOT ROBUST — do not deploy'}")
print(f"\n(total {time.time()-t0:.0f}s)")
