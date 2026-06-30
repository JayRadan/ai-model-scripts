"""
Campaign 4 — SESSION filter to make the edge cost-robust.

The edge is +0.40R gross but only ~2.8-3pt breakeven spread. US30 spreads are
tightest in the US cash session. Test whether restricting entries to liquid hours
(a) keeps 10-20/day, (b) raises gross per-trade, (c) stays robust on WF.
Compare ALL-hours vs US-cash (13:30-20:00 UTC) vs CORE (14:30-19:30 UTC).
Net edge reported at 1.0 and 1.5 pt spread (realistic in-session).
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
t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

df = pd.read_parquet("/home/jay/Desktop/new-model-zigzag/data/m1_dji_full.parquet")
df = df.rename(columns={[c for c in df.columns if "time" in c.lower()][0]: "time"})
df["time"] = pd.to_datetime(df["time"]); df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
if "tick_volume" not in df.columns: df["tick_volume"] = df.get("volume", 0)
feat = hf.compute_all_features(df.copy(), CFG)
_A = 2.0 / 51.0; s = df.set_index("time")["close"]
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
ds = feat["dist_at_signal"].to_numpy(float); da = np.abs(ds); demaA = feat["dist_ema20"].to_numpy(float)
O = df["open"].to_numpy(float); H = df["high"].to_numpy(float); L = df["low"].to_numpy(float); C = df["close"].to_numpy(float)
times = df["time"].values; n = len(df); medatr = np.nanmedian(atr)
hour = pd.DatetimeIndex(times).hour.values + pd.DatetimeIndex(times).minute.values / 60.0
base_ok = np.isfinite(atr) & (atr > 0) & (cdir != 0); base_ok[:300] = False; base_ok[-301:] = False

SESS = {
    "ALL":  np.ones(n, bool),
    "CASH13.5-20": (hour >= 13.5) & (hour < 20.0),
    "CORE14.5-19.5": (hour >= 14.5) & (hour < 19.5),
}
ENTRIES = {"pb1.0": da <= 1.0, "near_ema20": np.abs(demaA) <= 0.5}

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
def take(order_idx, entry_bar, exit_bar, pnl, dirs, cd):
    busy = -1; m = len(order_idx); R = np.zeros(m); D = np.zeros(m, np.int64); c = 0
    for t in range(m):
        k = order_idx[t]
        if entry_bar[k] <= busy: continue
        R[c] = pnl[k]; D[c] = dirs[k]; busy = exit_bar[k] + cd; c += 1
    return R[:c], D[:c]

WINS = []; first = pd.Timestamp("2020-07-01"); tsw = first; lastd = pd.to_datetime(times).max()
while tsw < lastd:
    WINS.append((tsw - pd.DateOffset(years=3), tsw, tsw + pd.DateOffset(months=6))); tsw += pd.DateOffset(months=6)

print(f"\n{'='*100}\nCAMPAIGN 4 — SESSION filter (1R≈{medatr:.1f}pts). Want 10-20/day, higher gross/trade, robust.\n{'='*100}")
print(f"{'entry':<11}{'session':<15}{'thr':>6}{'/day':>6}{'grossR/t':>9}{'shortR':>8}{'profWk':>8}{'net@1pt':>9}{'net@1.5pt':>10}")
for ename, emask in ENTRIES.items():
    for sname, smask in SESS.items():
        idx = np.where(emask & smask & base_ok)[0]; dirs = cdir[idx].astype(np.int64)
        if len(idx) < 20000: print(f"{ename:<11}{sname:<15} too few"); continue
        pnlG, xit = sim_gross(idx, dirs, O, H, L, C, atr, 6.0, 2.0, 300, n)
        ct = pd.to_datetime(times[idx]); Xc = feat[FC].to_numpy(np.float32)[idx]; eb = idx + 1
        rng = np.random.RandomState(0); preds = []
        for tr_s, te_s, te_e in WINS:
            trm = (ct >= tr_s) & (ct < te_s); tem = (ct >= te_s) & (ct < te_e)
            if trm.sum() < 3000 or tem.sum() < 20: preds.append((te_s, te_e, np.full(len(idx), -9e9))); continue
            tix = np.where(trm)[0]
            if len(tix) > 150000: tix = rng.choice(tix, 150000, replace=False)
            m = XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.85,
                             colsample_bytree=0.85, min_child_weight=10, n_jobs=-1, random_state=0)
            m.fit(Xc[tix], pnlG[tix]); full = np.full(len(idx), -9e9); full[tem] = m.predict(Xc[tem]); preds.append((te_s, te_e, full))
        allp = np.concatenate([p[2][p[2] > -9e8] for p in preds])
        # pick threshold to land ~14/day TAKEN: search
        best = None
        for q in np.linspace(0.60, 0.985, 18):
            thr = np.quantile(allp, q); per = []; tot = 0.0; totS = 0.0; ntr = 0; prof = 0; nwk = 0
            for (te_s, te_e, full) in preds:
                kk = np.where((ct >= te_s) & (ct < te_e) & (full >= thr))[0]
                if len(kk) < 3: continue
                order = kk[np.argsort(eb[kk])]; R, D = take(order.astype(np.int64), eb, xit, pnlG, dirs, 5)
                if len(R) == 0: continue
                tot += R.sum(); totS += R[D == -1].sum(); ntr += len(R); nwk += 1; prof += int(R.sum() - len(R)*(1.0/medatr) > 0)
            perday = ntr / 1541
            if 10 <= perday <= 20:
                gpt = tot / max(ntr, 1)
                score = gpt  # maximise gross per-trade within the freq band
                if best is None or score > best[0]: best = (score, thr, perday, gpt, totS, prof, nwk, ntr, tot)
        if best is None:
            print(f"{ename:<11}{sname:<15} no thr in 10-20/day band"); continue
        _, thr, perday, gpt, totS, prof, nwk, ntr, tot = best
        net1 = (gpt - 1.0/medatr); net15 = (gpt - 1.5/medatr)
        print(f"{ename:<11}{sname:<15}{thr:>6.2f}{perday:>6.1f}{gpt:>+9.3f}{totS:>+8.0f}{prof:>6}/{nwk:<1}"
              f"{net1*ntr:>+9.0f}{net15*ntr:>+10.0f}  (net/t @1pt {net1:+.3f})")
log("campaign 4 done")
