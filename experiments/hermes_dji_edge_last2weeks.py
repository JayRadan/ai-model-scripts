"""
Hermes DJI EDGE — honest out-of-sample test on the LAST 2 WEEKS.
Train model on local parquet (through 2026-05-27), calibrate threshold to ~11/day
taken on the TRAIN portfolio, then run on FRESH Dukascopy bars for the last 2 weeks
(unseen). Report every trade + gross/net@1/1.5/2pt, per-day, WR, PF.
"""
import sys, pickle, time
from datetime import datetime, timedelta, timezone
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
    busy = -1; m = len(order_idx); R = np.zeros(m); D = np.zeros(m, np.int64); EN = np.zeros(m, np.int64); EX = np.zeros(m, np.int64); c = 0
    for t in range(m):
        k = order_idx[t]
        if entry_bar[k] <= busy: continue
        R[c] = pnl[k]; D[c] = dirs[k]; EN[c] = entry_bar[k]; EX[c] = exit_bar[k]; busy = exit_bar[k] + cd; c += 1
    return R[:c], D[:c], EN[:c], EX[:c]

def candset(feat, df):
    atr = feat["atr14"].to_numpy(float); cdir = feat["committed_dir"].to_numpy(np.int64)
    da = np.abs(feat["dist_at_signal"].to_numpy(float)); n = len(df)
    ok = np.isfinite(atr) & (atr > 0) & (cdir != 0); ok[:300] = False; ok[-301:] = False
    idx = np.where((da <= 1.0) & ok)[0]; return idx, cdir[idx].astype(np.int64), atr

# ── 1. TRAIN on local parquet (through 2026-05-27) ───────────────────────────
dtr = pd.read_parquet("/home/jay/Desktop/new-model-zigzag/data/m1_dji_full.parquet")
dtr = dtr.rename(columns={[c for c in dtr.columns if "time" in c.lower()][0]: "time"})
dtr["time"] = pd.to_datetime(dtr["time"]); dtr = dtr.sort_values("time").drop_duplicates("time").reset_index(drop=True)
if "tick_volume" not in dtr.columns: dtr["tick_volume"] = dtr.get("volume", 0)
ftr = add_feats(dtr); log(f"train feats ({len(dtr):,} bars to {dtr.time.iloc[-1]})")
Otr=dtr["open"].to_numpy(float);Htr=dtr["high"].to_numpy(float);Ltr=dtr["low"].to_numpy(float);Ctr=dtr["close"].to_numpy(float)
itr, dtrs, atr_tr = candset(ftr, dtr); ntr = len(dtr)
ytr, xtr = sim_gross(itr, dtrs, Otr, Htr, Ltr, Ctr, atr_tr, 6.0, 2.0, 300, ntr)
Xtr = ftr[FC].to_numpy(np.float32)[itr]
medatr = np.nanmedian(atr_tr)
m = XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.85,
                 colsample_bytree=0.85, min_child_weight=10, n_jobs=-1, random_state=0)
m.fit(Xtr, ytr); log("model trained")
# calibrate threshold on TRAIN portfolio -> ~11/day taken
ptr = m.predict(Xtr); ebtr = itr + 1
tr_days = max((dtr.time.iloc[-1] - dtr.time.iloc[0]).days * 5/7, 1)
thr = ptr.max(); gap = 1e9
for th in np.quantile(ptr, np.linspace(0.3, 0.9, 25)):
    kk = np.where(ptr >= th)[0]; order = kk[np.argsort(ebtr[kk])]
    R, _, _, _ = take(order.astype(np.int64), ebtr, xtr, ytr, dtrs, 5)
    g = abs(len(R) / tr_days - 11.0)
    if g < gap: gap = g; thr = th
log(f"calibrated threshold = {thr:.3f}  (target 11/day)")

# ── 2. FRESH Dukascopy for last ~7 weeks (warmup + 2wk test) ─────────────────
import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_IDX_AMERICA_E_D_J_IND
end = datetime.now(timezone.utc); start = end - timedelta(days=50)
raw = dukascopy_python.fetch(instrument=INSTRUMENT_IDX_AMERICA_E_D_J_IND, interval=dukascopy_python.INTERVAL_MIN_1,
                             offer_side=dukascopy_python.OFFER_SIDE_BID, start=start, end=end)
dt = raw.reset_index().rename(columns={"timestamp": "time"})
dt["time"] = pd.to_datetime(dt["time"]).dt.tz_convert("UTC").dt.tz_localize(None)
dt = dt.sort_values("time").drop_duplicates("time").reset_index(drop=True)
if "tick_volume" not in dt.columns: dt["tick_volume"] = dt[[c for c in dt.columns if "vol" in c.lower()][0]]
ft = add_feats(dt); log(f"OOS feats ({len(dt):,} bars, {dt.time.iloc[0]} -> {dt.time.iloc[-1]})")
Ot=dt["open"].to_numpy(float);Ht=dt["high"].to_numpy(float);Lt=dt["low"].to_numpy(float);Ct=dt["close"].to_numpy(float)
it, dts, atr_t = candset(ft, dt); nt = len(dt)
yt, xt = sim_gross(it, dts, Ot, Ht, Lt, Ct, atr_t, 6.0, 2.0, 300, nt)
Xt = ft[FC].to_numpy(np.float32)[it]; pt = m.predict(Xt); ebt = it + 1
ctt = pd.to_datetime(dt["time"].to_numpy()[it])

# ── 3. test on LAST 2 WEEKS ──────────────────────────────────────────────────
test_start = pd.Timestamp("2026-06-01")
sel = (pt >= thr) & (ctt >= test_start)
kk = np.where(sel)[0]; order = kk[np.argsort(ebt[kk])]
R, D, EN, EX = take(order.astype(np.int64), ebt, xt, yt, dts, 5)
spR = 1.0/medatr
tdays = len(pd.bdate_range(test_start, dt.time.iloc[-1]))
print(f"\n{'='*86}\nHERMES DJI EDGE — LAST 2 WEEKS OOS  ({test_start.date()} → {dt.time.iloc[-1].date()}, {tdays} trading days)\n{'='*86}")
print(f"model trained through {dtr.time.iloc[-1].date()} | thr {thr:.3f} | 1R ≈ {medatr:.1f}pts | spread tested 1/1.5/2 pt")
if len(R) == 0:
    print("no trades in window"); sys.exit(0)
net1 = R - spR
tdf = pd.to_datetime(dt["time"].to_numpy())
if len(R) <= 40:
    print(f"\n{'#':>3} {'entry':>16} {'dir':>4} {'exit':>16} {'grossR':>7} {'net@1pt':>8}")
    for i in range(len(R)):
        print(f"{i+1:>3} {str(tdf[EN[i]])[:16]:>16} {'BUY' if D[i]==1 else 'SELL':>4} {str(tdf[EX[i]])[:16]:>16} {R[i]:>+7.2f} {net1[i]:>+8.2f}")
w = int((net1 > 0).sum()); pf = net1[net1>0].sum()/max(-net1[net1<=0].sum(),1e-9)
# per-day split
dd = pd.to_datetime([tdf[EN[i]] for i in range(len(R))]).date
for day in sorted(set(dd)):
    msk = dd == day; rday = R[msk]; nday = (rday - spR)
    print(f"  {day}:  {int(msk.sum())} trades  gross {rday.sum():+.1f}R  net@1pt {nday.sum():+.1f}R  WR {int((nday>0).sum())}/{int(msk.sum())}")
print(f"\n  trades {len(R)}  ({len(R)/max(tdays,1):.1f}/day)   LONG {int((D==1).sum())} / SHORT {int((D==-1).sum())}")
print(f"  GROSS sumR {R.sum():+.1f}")
print(f"  NET sumR  @1pt {(R-1.0/medatr).sum():+.1f} | @1.5pt {(R-1.5/medatr).sum():+.1f} | @2pt {(R-2.0/medatr).sum():+.1f}")
print(f"  WR {w/len(R)*100:.0f}%  PF@1pt {pf:.2f}  per-trade@1pt {net1.mean():+.3f}R")
print(f"  est USD @0.10 lot (net@1pt): {net1.sum()*7.5*10:+.0f}   (R≈$7.5/0.1lot)")
log("done")
