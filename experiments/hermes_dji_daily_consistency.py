"""
Hermes DJI — search for DAILY CONSISTENCY (most days positive, no long losing
streaks). Hypothesis: higher trades/day smooths daily P&L (law of large numbers).
For pb1.0 & near_ema20, sweep threshold to ~10/20/30/40 trades/day, run 8y WF,
and report DAILY net@1pt stats: % positive days, worst day, max consecutive losing
days, daily Sharpe. Train-only thresholds (no look-ahead).
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
t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
_A = 2.0 / 51.0

df = pd.read_parquet("/home/jay/Desktop/new-model-zigzag/data/m1_dji_full.parquet")
df = df.rename(columns={[c for c in df.columns if "time" in c.lower()][0]: "time"})
df["time"] = pd.to_datetime(df["time"]); df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
if "tick_volume" not in df.columns: df["tick_volume"] = df.get("volume", 0)
feat = hf.compute_all_features(df.copy(), CFG); s = df.set_index("time")["close"]
for nm, tfm in [("m5", 5), ("m15", 15), ("h1", 60)]:
    g = s.resample(f"{tfm}min").last().dropna(); hc = g.to_numpy(np.float64)
    et = (g.index + pd.Timedelta(minutes=tfm)).values
    N = len(df); rsi = np.full(N, np.nan); slope = np.full(N, np.nan); emad = np.full(N, np.nan)
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
log("features done")

atr = feat["atr14"].to_numpy(float); cdir = feat["committed_dir"].to_numpy(np.int64)
da = np.abs(feat["dist_at_signal"].to_numpy(float)); dema = np.abs(feat["dist_ema20"].to_numpy(float))
O = df["open"].to_numpy(float); H = df["high"].to_numpy(float); L = df["low"].to_numpy(float); C = df["close"].to_numpy(float)
times = df["time"].values; n = len(df); medatr = np.nanmedian(atr); spR = 1.0 / medatr
base_ok = np.isfinite(atr) & (atr > 0) & (cdir != 0); base_ok[:300] = False; base_ok[-301:] = False
BASES = {"pb1.0": (da <= 1.0), "pb1.5": (da <= 1.5), "near_ema20": (dema <= 0.5)}

@njit
def sim_gross(idxs, dirs, O, H, L, C, atr, SL, TRAIL, MAXH, n):
    m = len(idxs); pnl = np.empty(m); xit = np.empty(m, np.int64)
    for k in range(m):
        i = idxs[k]; d = dirs[k]; a = atr[i]; ei = i + 1
        if ei >= n or not (a > 0): pnl[k] = 0.0; xit[k] = min(i + 1, n - 1); continue
        ep = O[ei]; hard = SL * a; trd = TRAIL * a; mf = 0.0; end = min(ei + MAXH, n - 1); done = False
        for jx in range(ei, end + 1):
            fav = d * (C[jx] - ep)
            if fav > mf: mf = fav
            if d == 1 and (ep - L[jx]) >= hard: pnl[k] = -SL; xit[k] = jx; done = True; break
            if d == -1 and (H[jx] - ep) >= hard: pnl[k] = -SL; xit[k] = jx; done = True; break
            if mf >= trd and (mf - fav) >= trd: pnl[k] = (mf - trd) / a; xit[k] = jx; done = True; break
        if not done: pnl[k] = d * (C[end] - ep) / a; xit[k] = end
    return pnl, xit

@njit
def take(order_idx, entry_bar, exit_bar, pnl, cd):
    busy = -1; m = len(order_idx); R = np.zeros(m); EN = np.zeros(m, np.int64); c = 0
    for t in range(m):
        k = order_idx[t]
        if entry_bar[k] <= busy: continue
        R[c] = pnl[k]; EN[c] = entry_bar[k]; busy = exit_bar[k] + cd; c += 1
    return R[:c], EN[:c]

WINS = []; first = pd.Timestamp("2020-07-01"); tsw = first; lastd = pd.to_datetime(times).max()
while tsw < lastd:
    WINS.append((tsw - pd.DateOffset(years=3), tsw, tsw + pd.DateOffset(months=6))); tsw += pd.DateOffset(months=6)

def thr_for_rate(pall, trm, eb, xit, pnlG, target, days):
    best = pall.max(); gap = 1e9
    for th in np.quantile(pall[trm], np.linspace(0.05, 0.95, 30)):
        kk = np.where(trm)[0][pall[trm] >= th]
        if len(kk) < 5: continue
        order = kk[np.argsort(eb[kk])]; R, _ = take(order.astype(np.int64), eb, xit, pnlG, 5)
        g = abs(len(R) / days - target)
        if g < gap: gap = g; best = th
    return best

print(f"\n{'='*104}\nHERMES DJI — DAILY CONSISTENCY vs FREQUENCY (8y WF, net@1pt, train-only thr). 1R≈{medatr:.1f}pts\n{'='*104}")
print(f"{'base':<11}{'target/d':>9}{'act/day':>8}{'net@1pt':>9}{'/trade':>8}{'%posDays':>9}{'worstDay':>9}{'maxLoseStreak':>14}{'dailySharpe':>12}")
for bname, bmask in BASES.items():
    idx = np.where(bmask & base_ok)[0]; dirs = cdir[idx].astype(np.int64)
    pnlG, xit = sim_gross(idx, dirs, O, H, L, C, atr, 6.0, 2.0, 300, n)
    ct = pd.to_datetime(times[idx]); Xc = feat[FC].to_numpy(np.float32)[idx]; eb = idx + 1
    rng = np.random.RandomState(0)
    models = []
    for tr_s, te_s, te_e in WINS:
        trm = (ct >= tr_s) & (ct < te_s); tem = (ct >= te_s) & (ct < te_e)
        if trm.sum() < 4000 or tem.sum() < 20: models.append(None); continue
        tix = np.where(trm)[0]; tix_f = tix if len(tix) <= 150000 else rng.choice(tix, 150000, replace=False)
        m = XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.85,
                         colsample_bytree=0.85, min_child_weight=10, n_jobs=-1, random_state=0)
        m.fit(Xc[tix_f], pnlG[tix_f]); models.append((m.predict(Xc), trm, tem, ct[tix]))
    for target in [10, 20, 30, 45]:
        days_R = {}; ntot = 0
        for mo in models:
            if mo is None: continue
            pall, trm, tem, ctr = mo
            trd_days = max((ctr.max() - ctr.min()).days * 5 / 7, 1)
            thr = thr_for_rate(pall, trm, eb, xit, pnlG, target, trd_days)
            kk = np.where(tem)[0][pall[tem] >= thr]; order = kk[np.argsort(eb[kk])]
            R, EN = take(order.astype(np.int64), eb, xit, pnlG, 5)
            if len(R) == 0: continue
            net = R - spR; ntot += len(R)
            for d, r in zip(pd.to_datetime(times[EN]).date, net):
                days_R[d] = days_R.get(d, 0.0) + r
        if not days_R: continue
        dser = pd.Series(days_R).sort_index(); dv = dser.to_numpy()
        posd = (dv > 0).mean() * 100; worst = dv.min()
        # max consecutive losing days
        streak = mx = 0
        for x in dv:
            streak = streak + 1 if x <= 0 else 0; mx = max(mx, streak)
        sharpe = dv.mean() / (dv.std() + 1e-9) * np.sqrt(252)
        print(f"{bname:<11}{target:>9}{ntot/1541:>8.1f}{dv.sum():>+9.0f}{dv.sum()/max(ntot,1):>+8.3f}"
              f"{posd:>8.0f}%{worst:>+9.1f}{mx:>14}{sharpe:>12.2f}")
log("daily consistency done")
