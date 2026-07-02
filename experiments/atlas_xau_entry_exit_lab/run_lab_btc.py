"""
ATLAS BTC — time-boxed-patience trail test on the strict-candle engine.
Harness = exact copy of experiments/atlas_retrain_like_dow/wf_btc.py (the protocol
that justified the 2026-06-26 8y deploy): expanding-window quarterly WF, train on
MFE>=2R candidates, per-candidate scoring at Q>=3.0, flat 0.30R spread.
NEW: exit variants (trail 2 -> tighter after N bars) evaluated as overlays on the
same model/gate. DEV = 2022..2024 windows (selection), HOLDOUT = 2025+ (untouched).
"""
import sys, time, json
from pathlib import Path
import numpy as np, pandas as pd
from numba import njit
ROOT = Path("/home/jay/Desktop/new-model-zigzag")
sys.path.insert(0, str(ROOT / "experiments/kalman_color_flip")); sys.path.insert(0, str(ROOT / "products/hermes_xau"))
import importlib.util
from kalman import compute_kalman, bars_in_regime_array
from tfk import compute_tfk
_spec = importlib.util.spec_from_file_location("ofm1", ROOT / "products/_shared/m1_with_orderflow.py")
_ofm1 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_ofm1)
add_standard_features = _ofm1.add_standard_features; FLOW_FEATS = list(_ofm1.FLOW_FEATS)
SPREAD = 0.30; SL, TRAIL, MAXH = 6.0, 2.0, 300; STRONG_ATR = 0.8; RMULT = 1.0; MFE_FILTER = 2.0; QTEST = 3.0
KF_FEATS = ["f_velPct", "f_velSignif", "f_innovZ", "f_volState", "f_accel", "f_velRaw"]
TFK_FEATS = ["force", "velocity", "x_est", "regime_w", "trend_raw", "trend", "committed_dir"]
STD = ["rsi14", "dist_ema20", "dist_ema50", "dist_ema100", "dist_ema200", "slope5", "slope10", "slope20", "atr_ratio",
       "m5_rsi14", "m5_slope5", "m5_ema50_dist", "m15_rsi14", "m15_slope5", "m15_ema50_dist", "h1_rsi14", "h1_slope5", "h1_ema50_dist"]
EXTRA = ["dist_kf", "dist_tfk", "kf_regime_age", "vel_up_streak", "bar_range_atr", "kf_dir", "body_atr", "strong_bear_prev", "strong_bull_prev"]
OUT = Path(__file__).parent
DEV_END = pd.Timestamp("2025-01-01")

@njit(cache=True)
def labels_tt(idxs, dirs, O, H, L, C, atr, sp, SL, TRAIL, MAXH, n, ta, tt):
    m = len(idxs); pnl = np.empty(m); mfe = np.empty(m); hold = np.zeros(m, np.int64)
    for k in range(m):
        i = idxs[k]; d = dirs[k]; a = atr[i]; ei = i + 1
        if ei >= n or not (a > 0): pnl[k] = 0.0; mfe[k] = 0.0; continue
        ep = O[ei]; hard = SL * a; mf = 0.0; end = min(ei + MAXH, n - 1); done = False
        for j in range(ei, end + 1):
            fav = d * (C[j] - ep)
            if fav > mf: mf = fav
            if d == 1 and (ep - L[j]) >= hard: pnl[k] = -SL - sp; hold[k] = j - ei; done = True; break
            if d == -1 and (H[j] - ep) >= hard: pnl[k] = -SL - sp; hold[k] = j - ei; done = True; break
            trd = TRAIL * a
            if ta > 0 and (j - ei) >= ta:
                w = tt * a
                if w < trd: trd = w
            if mf >= trd and (mf - fav) >= trd: pnl[k] = (mf - trd) / a - sp; hold[k] = j - ei; done = True; break
        if not done: pnl[k] = d * (C[end] - ep) / a - sp; hold[k] = end - ei
        mfe[k] = mf / a
    return pnl, mfe, hold

def streak_up(v):
    n = len(v); o = np.zeros(n, np.int32); c = 0
    for i in range(1, n): c = c + 1 if v[i] > v[i - 1] else 0; o[i] = c
    return o

t0 = time.time()
m1 = pd.read_parquet(ROOT / "data/m1_btc_orderflow_8y.parquet").sort_values("time").reset_index(drop=True)
print(f"BTC {len(m1):,} bars  {m1.time.iloc[0]} -> {m1.time.iloc[-1]}", flush=True)
tfk_df = compute_tfk(m1); kf = compute_kalman(m1, r_mult=RMULT); df = add_standard_features(kf)
for c in TFK_FEATS: df[c] = tfk_df[c].to_numpy()
df["tfk_line"] = tfk_df["tfk_line"].to_numpy()
O = m1.open.to_numpy(float); H = m1.high.to_numpy(float); L = m1.low.to_numpy(float); C = m1.close.to_numpy(float)
atr = df["atr14"].to_numpy(float); kdir = df["kf_dir"].to_numpy(np.int64); kline = df["kf_p"].to_numpy(float)
cdir = tfk_df["committed_dir"].to_numpy(np.int64); tline = tfk_df["tfk_line"].to_numpy(float); vel = df["f_velRaw"].to_numpy(float)
kage = bars_in_regime_array(kdir); df["kf_regime_age"] = kage
df["bar_range_atr"] = (H - L) / np.maximum(atr, 1e-9)
df["dist_kf"] = np.where(atr > 0, (C - kline) / atr, 0.0); df["dist_tfk"] = np.where(atr > 0, (C - tline) / atr, 0.0)
df["vel_up_streak"] = streak_up(vel); body = C - O; body_atr = np.where(atr > 0, np.abs(body) / atr, 0.0); df["body_atr"] = body_atr
sb = (body < 0) & (body_atr >= STRONG_ATR); su = (body > 0) & (body_atr >= STRONG_ATR)
df["strong_bear_prev"] = np.concatenate([[False], sb[:-1]]); df["strong_bull_prev"] = np.concatenate([[False], su[:-1]])
n = len(df); sp = SPREAD / np.nanmedian(atr)
pbk = np.concatenate([[False], C[:-1] < kline[:-1]]); pak = np.concatenate([[False], C[:-1] > kline[:-1]])
pbt = np.concatenate([[False], C[:-1] < tline[:-1]]); pat = np.concatenate([[False], C[:-1] > tline[:-1]])
g = C > O; r = C < O; ok = np.isfinite(atr) & (atr > 0); ok[:250] = False; ok[-(MAXH + 1):] = False
buy = ok & (cdir == 1) & (kdir == -1) & df["strong_bear_prev"].to_numpy() & pbk & pbt & g & (kage >= 3)
sell = ok & (cdir == -1) & (kdir == 1) & df["strong_bull_prev"].to_numpy() & pak & pat & r & (kage >= 3)
mask = buy | sell; idxs = np.where(mask)[0]; dirs = np.where(cdir[idxs] == 1, 1, -1).astype(np.int64)

VAR = {"base (deployed)": (0, 2.0), "tt30/0.75": (30, 0.75), "tt30/0.5": (30, 0.5),
       "tt60/0.75": (60, 0.75), "tt60/1.0": (60, 1.0), "tt90/1.0": (90, 1.0),
       "uni1/0.5 (ctrl)": (1, 0.5)}
LAB = {}
for name, (ta, tt) in VAR.items():
    LAB[name] = labels_tt(idxs, dirs, O, H, L, C, atr, sp, SL, TRAIL, MAXH, n, ta, tt)
pnl0, mfe0, hold0 = LAB["base (deployed)"]
print(f"  candidates {len(idxs):,}  median hold {np.median(hold0):.0f} bars  p90 {np.percentile(hold0,90):.0f}  ({time.time()-t0:.0f}s)", flush=True)

feat_cols = [c for c in dict.fromkeys(EXTRA + KF_FEATS + TFK_FEATS + STD + FLOW_FEATS) if c in df.columns]
X = df.iloc[idxs][feat_cols].fillna(0).to_numpy(np.float32)
ctime = m1["time"].to_numpy()[idxs]
del df, m1, tfk_df, kf
from xgboost import XGBRegressor
def fitpred(trX, trY, teX):
    M = XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.04, subsample=0.85, colsample_bytree=0.85,
                     min_child_weight=8, reg_lambda=1.0, objective="reg:squarederror", tree_method="hist", random_state=42, verbosity=0)
    M.fit(trX, trY); return M.predict(teX)

wins = pd.date_range("2022-01-01", "2026-04-01", freq="3MS")
res = {name: [] for name in VAR}
for ws in wins:
    we = ws + pd.DateOffset(months=3)
    tr = ctime < np.datetime64(ws); te = (ctime >= np.datetime64(ws)) & (ctime < np.datetime64(we))
    trsel = tr & (mfe0 >= MFE_FILTER)
    if trsel.sum() < 500 or te.sum() < 30: continue
    q = fitpred(X[trsel], pnl0[trsel].astype(np.float32), X[te])   # model = deployed labels, all variants share it
    qm = q >= QTEST
    for name in VAR:
        rs = LAB[name][0][te][qm]; rs = rs[np.isfinite(rs)]
        if len(rs) < 10: res[name].append(dict(win=str(ws.date()), n=0)); continue
        w, l = rs[rs > 0], rs[rs <= 0]
        res[name].append(dict(win=str(ws.date()), n=len(rs), wr=(rs > 0).mean() * 100,
                              pf=w.sum() / max(-l.sum(), 1e-9), sumR=float(rs.sum())))
    print(f"  window {ws.date()} done ({time.time()-t0:.0f}s)", flush=True)

def agg(rows, dev):
    rr = [r for r in rows if r.get("n", 0) > 0 and ((pd.Timestamp(r["win"]) < DEV_END) == dev)]
    if not rr: return None
    allR = sum(r["sumR"] for r in rr); nn = sum(r["n"] for r in rr)
    return dict(nwin=len(rr), n=nn, sumR=allR, wpos=sum(1 for r in rr if r["sumR"] > 0),
                medpf=float(np.median([r["pf"] for r in rr])),
                wr=float(np.average([r["wr"] for r in rr], weights=[r["n"] for r in rr])))

for dev, label in [(True, "DEV 2022-2024 (selection)"), (False, "HOLDOUT 2025+ (untouched)")]:
    print(f"\n{'='*74}\nATLAS BTC strict-candle, Q>=3.0, spread 0.30R — {label}\n{'='*74}")
    print(f"{'variant':<18}{'n':>6}{'sumR':>9}{'medPF':>7}{'WR%':>6}{'w+':>6}")
    for name in VAR:
        a = agg(res[name], dev)
        if a is None: continue
        print(f"{name:<18}{a['n']:>6}{a['sumR']:>+9.0f}{a['medpf']:>7.2f}{a['wr']:>6.0f}{a['wpos']:>4}/{a['nwin']:<2}")
json.dump(res, open(OUT / "btc_lab_results.json", "w"), default=str, indent=1)
print(f"\n({time.time()-t0:.0f}s) btc lab done")
