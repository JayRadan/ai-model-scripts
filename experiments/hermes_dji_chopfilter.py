"""
Hermes DJI EDGE — chop-filter test. Does gating entries to TRENDING conditions
trim the -6R stop clusters and improve the 8y walk-forward (not just June)?

Filters (entry-gate on the pb1.0 base): Kaufman Efficiency Ratio (ER20/ER50),
regime stability (regime_age), slope strength, + a portfolio loss-streak cooldown.
Each variant: retrain XGB on gross-R of filtered candidates, WF with TRAIN-ONLY
threshold calibrated to ~10/day, report 8y net@1pt + windows + per-trade + maxDD,
AND June-2026 net@1pt. Baseline = no filter.
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
da = np.abs(feat["dist_at_signal"].to_numpy(float)); rage = feat["regime_age"].to_numpy(float)
sl20 = np.abs(feat["slope20"].to_numpy(float))
O = df["open"].to_numpy(float); H = df["high"].to_numpy(float); L = df["low"].to_numpy(float); C = df["close"].to_numpy(float)
times = df["time"].values; n = len(df); medatr = np.nanmedian(atr); spR = 1.0 / medatr
# Kaufman efficiency ratio
absd = np.abs(np.diff(C, prepend=C[0]))
def ER(N):
    num = np.abs(C - np.concatenate([np.full(N, np.nan), C[:-N]]))
    den = pd.Series(absd).rolling(N).sum().to_numpy()
    return np.where(den > 0, num / den, 0.0)
er20 = ER(20); er50 = ER(50)
base_ok = np.isfinite(atr) & (atr > 0) & (cdir != 0); base_ok[:300] = False; base_ok[-301:] = False
pb = (da <= 1.0) & base_ok

FILTERS = {
    "baseline":      np.ones(n, bool),
    "ER50>0.30":     er50 > 0.30,
    "ER50>0.40":     er50 > 0.40,
    "ER20>0.35":     er20 > 0.35,
    "regime_age>=8": rage >= 8,
    "slope20>0.6":   sl20 > 0.6,
    "ER50>0.3&rage>=5": (er50 > 0.30) & (rage >= 5),
}

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
jun_s = pd.Timestamp("2026-06-01")

print(f"\n{'='*100}\nHERMES DJI — CHOP-FILTER test (pb1.0, SL6/T2, train-only thr ~10/day). 1R≈{medatr:.1f}pts.\n{'='*100}")
print(f"{'filter':<18}{'/day':>6}{'8y net@1pt':>12}{'profWk':>8}{'per-trd':>9}{'maxDD':>8}{'JUNE net@1pt':>14}")
for fname, fmask in FILTERS.items():
    idx = np.where(pb & fmask)[0]; dirs = cdir[idx].astype(np.int64)
    if len(idx) < 20000: print(f"{fname:<18} too few candidates"); continue
    pnlG, xit = sim_gross(idx, dirs, O, H, L, C, atr, 6.0, 2.0, 300, n)
    ct = pd.to_datetime(times[idx]); Xc = feat[FC].to_numpy(np.float32)[idx]; eb = idx + 1
    rng = np.random.RandomState(0); allnet = []; jun_net = 0.0; nwin = 0; profwin = 0; ntot = 0
    for tr_s, te_s, te_e in WINS:
        trm = (ct >= tr_s) & (ct < te_s); tem = (ct >= te_s) & (ct < te_e)
        if trm.sum() < 4000 or tem.sum() < 20: continue
        tix = np.where(trm)[0]; tix_f = tix if len(tix) <= 150000 else rng.choice(tix, 150000, replace=False)
        m = XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.85,
                         colsample_bytree=0.85, min_child_weight=10, n_jobs=-1, random_state=0)
        m.fit(Xc[tix_f], pnlG[tix_f]); pall = m.predict(Xc)
        trd_days = max((ct[tix].max() - ct[tix].min()).days * 5 / 7, 1); thr = pall.max(); gap = 1e9
        for th in np.quantile(pall[tix], np.linspace(0.3, 0.92, 20)):
            kk = np.where(trm)[0][pall[trm] >= th]
            if len(kk) < 5: continue
            order = kk[np.argsort(eb[kk])]; Rtr, _ = take(order.astype(np.int64), eb, xit, pnlG, 5)
            g = abs(len(Rtr) / trd_days - 10.0)
            if g < gap: gap = g; thr = th
        kk = np.where(tem)[0][pall[tem] >= thr]; order = kk[np.argsort(eb[kk])]
        R, EN = take(order.astype(np.int64), eb, xit, pnlG, 5)
        if len(R) == 0: continue
        net = R - spR; allnet.append(net); ntot += len(R); nwin += 1; profwin += int(net.sum() > 0)
        ent_t = pd.to_datetime(times[EN]); jun_net += net[ent_t >= jun_s].sum()
    flat = np.concatenate(allnet) if allnet else np.array([0.0])
    dd = float((np.maximum.accumulate(np.cumsum(flat)) - np.cumsum(flat)).max())
    print(f"{fname:<18}{ntot/1541:>6.1f}{flat.sum():>+12.0f}{profwin:>6}/{nwin:<1}{flat.mean():>+9.3f}{dd:>8.0f}{jun_net:>+14.1f}")
log("chop-filter test done")
