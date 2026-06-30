"""
PORTFOLIO daily-consistency test — the real path to smooth daily equity.

Run the SAME edge recipe (pb1.0 pullback, SL6/T2 trail, XGBRegressor on gross R,
1-slot cooldown5, train-only threshold ~11/day, causal features) independently on
DJI, BTC, XAU. Get each instrument's DAILY net P&L (in R, 0.15R spread each), then
combine equal-risk. Compare standalone vs combined: % positive days, max losing
streak, daily Sharpe. Diversification should lift %pos-days and shrink streaks.
Bonus: running the identical recipe on 3 instruments tests if the edge GENERALIZES.
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
_A = 2.0 / 51.0; SPREAD_R = 0.15

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

def add_feats(df):
    f = hf.compute_all_features(df.copy(), CFG); s = df.set_index("time")["close"]
    for nm, tfm in [("m5", 5), ("m15", 15), ("h1", 60)]:
        g = s.resample(f"{tfm}min").last().dropna(); hc = g.to_numpy(np.float64)
        et = (g.index + pd.Timedelta(minutes=tfm)).values
        N = len(df); rsi = np.full(N, np.nan); slope = np.full(N, np.nan); emad = np.full(N, np.nan)
        if len(hc) >= 16:
            ema = np.empty(len(hc)); e = hc[0]
            for i in range(len(hc)):
                e = hc[0] if i == 0 else e * (1 - _A) + hc[i] * _A; ema[i] = e
            dl = np.diff(hc, prepend=hc[0]); cg = np.cumsum(np.clip(dl, 0, None)); cl = np.cumsum(np.clip(-dl, 0, None))
            j = np.searchsorted(et, df["time"].values, side="right") - 1; ok = j >= 14; jj = j[ok]
            cc = df["close"].to_numpy(np.float64)[ok]
            slope[ok] = cc - hc[jj - 4]; emad[ok] = cc - (ema[jj] * (1 - _A) + cc * _A)
            dc = cc - hc[jj]; gs = (cg[jj] - cg[jj - 13]) + np.clip(dc, 0, None); ls = (cl[jj] - cl[jj - 13]) + np.clip(-dc, 0, None)
            rs = (gs / 14.0) / np.where(ls == 0, np.nan, ls / 14.0); rsi[ok] = 100 - 100 / (1 + rs)
        f[f"{nm}_rsi14"] = rsi; f[f"{nm}_slope5"] = slope; f[f"{nm}_ema50_dist"] = emad
    return f

def run_edge(path, name, train_years=2):
    df = pd.read_parquet(path).rename(columns={"timestamp": "time"})
    if "time" not in df.columns: df = df.rename(columns={[c for c in df.columns if "time" in c.lower()][0]: "time"})
    df["time"] = pd.to_datetime(df["time"]); df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    if "tick_volume" not in df.columns: df["tick_volume"] = df.get("volume", 0)
    feat = add_feats(df)
    atr = feat["atr14"].to_numpy(float); cdir = feat["committed_dir"].to_numpy(np.int64)
    da = np.abs(feat["dist_at_signal"].to_numpy(float))
    O = df["open"].to_numpy(float); H = df["high"].to_numpy(float); L = df["low"].to_numpy(float); C = df["close"].to_numpy(float)
    times = df["time"].values; n = len(df)
    ok = np.isfinite(atr) & (atr > 0) & (cdir != 0); ok[:300] = False; ok[-301:] = False
    idx = np.where((da <= 1.0) & ok)[0]; dirs = cdir[idx].astype(np.int64); eb = idx + 1
    pnlG, xit = sim_gross(idx, dirs, O, H, L, C, atr, 6.0, 2.0, 300, n)
    ct = pd.to_datetime(times[idx]); Xc = feat[FC].to_numpy(np.float32)[idx]; rng = np.random.RandomState(0)
    wins = []; first = pd.Timestamp(df.time.iloc[0]) + pd.DateOffset(years=train_years)
    tsw = first; lastd = ct.max(); step = pd.DateOffset(months=3 if train_years < 2 else 6)
    while tsw < lastd:
        wins.append((tsw - pd.DateOffset(years=train_years), tsw, tsw + step)); tsw += step
    daily = {}; ntot = 0
    for tr_s, te_s, te_e in wins:
        trm = (ct >= tr_s) & (ct < te_s); tem = (ct >= te_s) & (ct < te_e)
        if trm.sum() < 3000 or tem.sum() < 20: continue
        tix = np.where(trm)[0]; tix_f = tix if len(tix) <= 150000 else rng.choice(tix, 150000, replace=False)
        m = XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.85,
                         colsample_bytree=0.85, min_child_weight=10, n_jobs=-1, random_state=0)
        m.fit(Xc[tix_f], pnlG[tix_f]); pall = m.predict(Xc)
        days = max((ct[tix].max() - ct[tix].min()).days * 5 / 7, 1); thr = pall.max(); gap = 1e9
        for th in np.quantile(pall[tix], np.linspace(0.3, 0.92, 18)):
            kk = np.where(trm)[0][pall[trm] >= th]
            if len(kk) < 5: continue
            order = kk[np.argsort(eb[kk])]; Rtr, _ = take(order.astype(np.int64), eb, xit, pnlG, 5)
            g = abs(len(Rtr) / days - 11.0)
            if g < gap: gap = g; thr = th
        kk = np.where(tem)[0][pall[tem] >= thr]; order = kk[np.argsort(eb[kk])]
        R, EN = take(order.astype(np.int64), eb, xit, pnlG, 5)
        if len(R) == 0: continue
        net = R - SPREAD_R; ntot += len(R)
        for d, r in zip(pd.to_datetime(times[EN]).date, net): daily[d] = daily.get(d, 0.0) + r
    log(f"  {name}: {ntot} trades, {ntot/max(len(daily),1):.1f}/active-day, {len(daily)} days, {df.time.iloc[0].date()}→{df.time.iloc[-1].date()}")
    return pd.Series(daily).sort_index()

def stats(dser):
    dv = dser.to_numpy(); pos = (dv > 0).mean() * 100; streak = mx = 0
    for x in dv:
        streak = streak + 1 if x <= 0 else 0; mx = max(mx, streak)
    sh = dv.mean() / (dv.std() + 1e-9) * np.sqrt(252)
    return dict(days=len(dv), total=dv.sum(), pos=pos, worst=dv.min(), streak=mx, sharpe=sh)

DJI = run_edge("/home/jay/Desktop/new-model-zigzag/data/m1_dji_full.parquet", "DJI", train_years=2)
XAU = run_edge("/home/jay/Desktop/new-model-zigzag/data/m1_xau_full.parquet", "XAU", train_years=2)
BTC = run_edge("/home/jay/Desktop/new-model-zigzag/data/m1_btc_full.parquet", "BTC", train_years=1)

def report(title, members):
    idxs = [m[1].index for m in members]
    common = sorted(set.intersection(*[set(i) for i in idxs])) if idxs else []
    if len(common) < 20:
        print(f"\n{title}: too few shared days ({len(common)})"); return
    print(f"\n{'='*92}\n{title}\ncommon period: {common[0]} → {common[-1]}  ({len(common)} shared days)\n{'='*92}")
    print(f"{'strategy':<24}{'totalR':>9}{'%posDays':>9}{'worstDay':>9}{'maxLoseStreak':>15}{'Sharpe':>8}")
    port = pd.Series(0.0, index=common)
    for nm, ser in members:
        sc = ser.reindex(common).fillna(0.0); st = stats(sc); port = port + sc
        print(f"{nm+' standalone':<24}{st['total']:>+9.0f}{st['pos']:>8.0f}%{st['worst']:>+9.1f}{st['streak']:>15}{st['sharpe']:>8.2f}")
    stp = stats(port); names = "+".join(m[0] for m in members)
    print("-"*92)
    print(f"{'PORTFOLIO '+names:<24}{stp['total']:>+9.0f}{stp['pos']:>8.0f}%{stp['worst']:>+9.1f}{stp['streak']:>15}{stp['sharpe']:>8.2f}")
    mat = pd.DataFrame({nm: ser.reindex(common).fillna(0) for nm, ser in members})
    print(f"\ndaily-return correlations:\n{mat.corr().round(2).to_string()}")

# Primary: DJI+XAU over the long 2018-2026 history (robust)
report("PORTFOLIO — DJI + XAU (long history, spread 0.15R each)", [("DJI", DJI), ("XAU", XAU)])
# Bonus: all 3 over BTC's shorter overlap
if len(BTC) > 20:
    report("PORTFOLIO — DJI + BTC + XAU (BTC overlap window)", [("DJI", DJI), ("BTC", BTC), ("XAU", XAU)])
log("portfolio test done")
