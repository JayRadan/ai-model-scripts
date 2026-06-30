"""
Hermes DJI — focused 8Y walk-forward of the CORRECT shipped recipe.
Label + exit = SL6 / TRAIL2 / BE0.5 (train_bundle_combined.py:43-44 + README),
candidates NEAR<=0.5 OR counter>=1.5, causal features, 1-slot live engine.
Sweeps Q in [1.5..5], prints per-window for each Q + aggregate. Local data.
"""
import sys, pickle, time
from pathlib import Path
import numpy as np, pandas as pd
from numba import njit
from xgboost import XGBRegressor

DE = Path("/home/jay/Desktop/my-agents-and-website/commercial/server/decision_engine")
sys.path.insert(0, str(DE))
import hermes_features as hf
from configs.hermes_dji import HERMES_DJI as CFG
FC = pickle.load(open(DE / "models/hermes_dji_validated.pkl", "rb"))["feat_cols"]
SL, TRAIL, BE = 6.0, 2.0, 0.5

t0 = time.time()
df = pd.read_parquet("/home/jay/Desktop/new-model-zigzag/data/m1_dji_full.parquet")
df = df.rename(columns={[c for c in df.columns if "time" in c.lower()][0]: "time"})
df["time"] = pd.to_datetime(df["time"]); df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
if "tick_volume" not in df.columns: df["tick_volume"] = df.get("volume", 0)
feat = hf.compute_all_features(df.copy(), CFG)
_A = 2.0 / 51.0; s = df.set_index("time")["close"]
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
print(f"features done ({time.time()-t0:.0f}s)", flush=True)

atr = feat["atr14"].to_numpy(float); cdir = feat["committed_dir"].to_numpy(np.int64)
ds = feat["dist_at_signal"].to_numpy(float); da = np.abs(ds)
O = df["open"].to_numpy(float); H = df["high"].to_numpy(float); L = df["low"].to_numpy(float); C = df["close"].to_numpy(float)
times = df["time"].values; n = len(df); sp = CFG.spread_usd / np.nanmedian(atr)
is_pb = da <= CFG.near_thr; is_ct = (ds * cdir) <= -CFG.counter_thr
ok = np.isfinite(atr) & (atr > 0) & (cdir != 0); ok[:300] = False; ok[-301:] = False
cand = np.where(ok & (is_pb | is_ct))[0]; cdr = cdir[cand].astype(np.int64)
X = feat[FC].to_numpy(np.float32)[cand]; ct = pd.to_datetime(times[cand])

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

y = lbl(cand, cdr, O, H, L, C, atr, float(sp), SL, TRAIL, 300, n)
print(f"candidates {len(cand):,}  label mean {y.mean():+.3f}R", flush=True)

def portsim(sel, seld, selq):
    info = {int(sel[k]): (int(seld[k]), float(selq[k])) for k in range(len(sel))}
    if not info: return None
    active, ex = [], []; lastd = {-1: -10**9, 1: -10**9}
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
        if i - lastd[d_] < 5: continue
        for t in active:
            if t["slr"] != 0 and t["d"] * (C[i] - t["ep"]) / t["a"] >= BE: t["slr"] = 0
        ei = i + 1
        if ei >= n or not (atr[i] > 0) or len(active) >= 1: continue
        active.append({"ei": ei, "d": d_, "ep": float(O[ei]), "a": float(atr[i]), "slr": -SL, "mf": 0.0, "R": None})
        lastd[d_] = i
    for t in active:
        eb = min(t["ei"] + 300, n - 1); t["R"] = float(t["d"] * (C[eb] - t["ep"]) / t["a"] - sp); ex.append(t)
    r = np.array([t["R"] for t in ex])
    if len(r) == 0: return None
    pf = r[r > 0].sum() / max(-r[r <= 0].sum(), 1e-9); eq = np.cumsum(r)
    return dict(n=len(r), sumR=float(r.sum()), wr=float((r > 0).mean() * 100), pf=float(pf),
                dd=float((np.maximum.accumulate(eq) - eq).max()))

first = pd.Timestamp("2020-07-01"); last = ct.max(); wins = []
tsw = first
while tsw < last:
    wins.append((tsw - pd.DateOffset(years=3), tsw, tsw + pd.DateOffset(months=6))); tsw += pd.DateOffset(months=6)

# train once per window, sweep Q on predictions
preds = []
for tr_s, te_s, te_e in wins:
    trm = (ct >= tr_s) & (ct < te_s); tem = (ct >= te_s) & (ct < te_e)
    if trm.sum() < 5000 or tem.sum() < 20: preds.append(None); continue
    m = XGBRegressor(n_estimators=600, max_depth=5, learning_rate=0.04, subsample=0.85,
                     colsample_bytree=0.85, min_child_weight=10, n_jobs=-1, random_state=0)
    m.fit(X[trm], y[trm]); preds.append((tem, m.predict(X[tem]), (te_s, te_e)))
print(f"models fit ({time.time()-t0:.0f}s)\n", flush=True)

print(f"{'='*78}\nHERMES DJI — SHIPPED RECIPE SL6/T2/BE.5, 8Y walk-forward, 1-slot\n{'='*78}")
print(f"{'Qthr':>5}{'wins':>6}{'profWk':>8}{'totTrd':>8}{'/day':>6}{'totSumR':>9}{'medPF':>7}{'worstWk':>9}")
for thr in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
    ws = []
    for p in preds:
        if p is None: continue
        tem, qp, _ = p; sm = qp >= thr
        r = portsim(cand[tem][sm], cdr[tem][sm], qp[sm])
        ws.append(r or {"n": 0, "sumR": 0.0, "pf": 0.0})
    nw = len(ws); prof = sum(1 for w in ws if w["sumR"] > 0); tot = sum(w["sumR"] for w in ws); trd = sum(w["n"] for w in ws)
    pfs = [w["pf"] for w in ws if w["n"] > 0]; medpf = float(np.median(pfs)) if pfs else 0
    worst = min((w["sumR"] for w in ws), default=0)
    ndays = sum(1 for _ in range(1))  # placeholder
    flag = "  ★" if (prof / max(nw, 1) >= 0.65 and tot > 0 and medpf >= 1.2) else ""
    print(f"{thr:>5}{nw:>6}{prof:>6}/{nw}{trd:>8}{trd/1500:>6.1f}{tot:>+9.1f}{medpf:>7.2f}{worst:>+9.1f}{flag}")

# detail for q=3.0
print(f"\nper-window @ Q>=3.0:")
for p in preds:
    if p is None: continue
    tem, qp, (a, b) = p; sm = qp >= 3.0
    r = portsim(cand[tem][sm], cdr[tem][sm], qp[sm]) or {"n": 0, "sumR": 0, "wr": 0, "pf": 0, "dd": 0}
    print(f"  {a.date()}→{b.date()}: {r['n']:>3} trd  sumR {r['sumR']:>+7.1f}  WR {r['wr']:>4.0f}%  PF {r['pf']:.2f}")
print(f"\n(total {time.time()-t0:.0f}s)")
