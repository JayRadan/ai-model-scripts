"""
Hermes DJI — CAUSAL retrain + honest OOS test.

Why: the deployed q_model was trained on LOOK-AHEAD HTF features (resample+ffill),
but the live server serves causal/partial-bucket features at decide-time → train/serve
skew. This retrains the SAME recipe on CAUSAL HTF features and tests out-of-sample,
head-to-head vs the deployed model, on identical candidates.

Recipe (from bundle train_meta):
  XGBRegressor(n_estimators=600, max_depth=5, lr=0.04, subsample=.85,
               colsample_bytree=.85, min_child_weight=10)
  29 STD feats | label = single-pos trail-exit R (SL4/trail3/maxhold300)
  candidates = pullback(|dist|<=0.5) OR counter(dist*cdir<=-1.5)
  split cutoff 2025-09-01 (train before / test after)
Eval: gate q>=thr, 1-slot live exit engine (SL4/trail3/BE1, cooldown5), sweep thr.
"""
import sys, pickle, json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np, pandas as pd
from numba import njit

DE = Path("/home/jay/Desktop/my-agents-and-website/commercial/server/decision_engine")
sys.path.insert(0, str(DE))
import hermes_features as hf
from configs.hermes_dji import HERMES_DJI as CFG
DEP = pickle.load(open(DE / "models/hermes_dji_validated.pkl", "rb"))
FC = DEP["feat_cols"]
OUT = Path("/home/jay/Desktop/new-model-zigzag/experiments/_hermes_retrain"); OUT.mkdir(exist_ok=True)

import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_IDX_AMERICA_E_D_J_IND
CUTOFF = pd.Timestamp("2025-09-01")
START  = pd.Timestamp("2022-09-01")          # 3y, ~2y train + ~9-10mo OOS
END    = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=3)

# ── chunked fetch (quarterly) to keep each request small ─────────────────────
def fetch_range(a, b):
    out = []
    cur = a
    while cur < b:
        nxt = min(cur + pd.Timedelta(days=90), b)
        try:
            r = dukascopy_python.fetch(instrument=INSTRUMENT_IDX_AMERICA_E_D_J_IND,
                    interval=dukascopy_python.INTERVAL_MIN_1, offer_side=dukascopy_python.OFFER_SIDE_BID,
                    start=cur.to_pydatetime().replace(tzinfo=timezone.utc),
                    end=nxt.to_pydatetime().replace(tzinfo=timezone.utc))
            if r is not None and len(r): out.append(r)
            print(f"  {cur.date()}→{nxt.date()}: {0 if r is None else len(r):,}", flush=True)
        except Exception as e:
            print(f"  {cur.date()}→{nxt.date()}: ERR {e}", flush=True)
        cur = nxt
    return pd.concat(out) if out else None

print(f"fetching DJI M1 {START.date()} → {END.date()} (chunked) ...", flush=True)
raw = fetch_range(START, pd.Timestamp(END))
df = raw.reset_index().rename(columns={"timestamp": "time"})
df["time"] = pd.to_datetime(df["time"]).dt.tz_convert("UTC").dt.tz_localize(None)
df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
if "tick_volume" not in df.columns:
    df["tick_volume"] = df[[c for c in df.columns if "vol" in c.lower()][0]]
print(f"TOTAL {len(df):,} bars {df.time.iloc[0]} → {df.time.iloc[-1]}", flush=True)

# ── features: TFK + standard, then overwrite 9 HTF cols with CAUSAL ──────────
print("computing features (TFK + causal HTF) ...", flush=True)
feat = hf.compute_all_features(df.copy(), CFG)
_A = 2.0 / 51.0
def causal_htf(d):
    d = d.copy(); s = d.set_index("time")["close"]; out = {}
    for nm, tfm in [("m5", 5), ("m15", 15), ("h1", 60)]:
        g = s.resample(f"{tfm}min").last().dropna(); hc = g.to_numpy(np.float64)
        et = (g.index + pd.Timedelta(minutes=tfm)).values
        n = len(d); rsi = np.full(n, np.nan); slope = np.full(n, np.nan); emad = np.full(n, np.nan)
        if len(hc) >= 16:
            ema = np.empty(len(hc)); e = hc[0]
            for i in range(len(hc)):
                e = hc[0] if i == 0 else e * (1 - _A) + hc[i] * _A; ema[i] = e
            dl = np.diff(hc, prepend=hc[0]); cg = np.cumsum(np.clip(dl, 0, None)); cl = np.cumsum(np.clip(-dl, 0, None))
            j = np.searchsorted(et, d["time"].values, side="right") - 1; ok = j >= 14; jj = j[ok]
            cc = d["close"].to_numpy(np.float64)[ok]
            slope[ok] = cc - hc[jj - 4]; emad[ok] = cc - (ema[jj] * (1 - _A) + cc * _A)
            dc = cc - hc[jj]; gs = (cg[jj] - cg[jj - 13]) + np.clip(dc, 0, None); ls = (cl[jj] - cl[jj - 13]) + np.clip(-dc, 0, None)
            rs = (gs / 14.0) / np.where(ls == 0, np.nan, ls / 14.0); rsi[ok] = 100 - 100 / (1 + rs)
        out[f"{nm}_rsi14"] = rsi; out[f"{nm}_slope5"] = slope; out[f"{nm}_ema50_dist"] = emad
    return out
for col, arr in causal_htf(df).items():
    feat[col] = arr

atr = feat["atr14"].to_numpy(float); cdir = feat["committed_dir"].to_numpy(np.int64)
dist_signed = feat["dist_at_signal"].to_numpy(float); dist_abs = np.abs(dist_signed)
O = df["open"].to_numpy(float); H = df["high"].to_numpy(float); L = df["low"].to_numpy(float); C = df["close"].to_numpy(float)
times = df["time"].values; n = len(df); sp = CFG.spread_usd / np.nanmedian(atr)

# ── candidates + single-position trail-exit label ────────────────────────────
is_pb = dist_abs <= CFG.near_thr; is_ct = (dist_signed * cdir) <= -CFG.counter_thr
ok = np.isfinite(atr) & (atr > 0) & (cdir != 0); ok[:300] = False; ok[-(CFG.max_hold_bars + 1):] = False
cand = np.where(ok & (is_pb | is_ct))[0]; cdr = cdir[cand]
print(f"candidates: {len(cand):,}", flush=True)

@njit
def label_trail(idxs, dirs, O, H, L, C, atr, sp, SL, TRAIL, MAXH, n):
    m = len(idxs); pnl = np.empty(m)
    for k in range(m):
        i = idxs[k]; d = dirs[k]; a = atr[i]; ei = i + 1
        if ei >= n or not (a > 0): pnl[k] = 0.0; continue
        ep = O[ei]; hard = SL * a; trd = TRAIL * a; mf = 0.0; end = min(ei + MAXH, n - 1); done = False
        for j in range(ei, end + 1):
            fav = d * (C[j] - ep)
            if fav > mf: mf = fav
            if d == 1 and (ep - L[j]) >= hard: pnl[k] = -SL - sp; done = True; break
            if d == -1 and (H[j] - ep) >= hard: pnl[k] = -SL - sp; done = True; break
            if mf >= trd and (mf - fav) >= trd: pnl[k] = (mf - trd) / a - sp; done = True; break
        if not done: pnl[k] = d * (C[end] - ep) / a - sp
    return pnl

y = label_trail(cand, cdr.astype(np.int64), O, H, L, C, atr, float(sp), 4.0, 3.0, 300, n)
X = feat[FC].to_numpy(np.float32)[cand]
ct = pd.to_datetime(times[cand])
tr = ct < CUTOFF; te = ct >= CUTOFF
print(f"train {tr.sum():,}  |  test {te.sum():,}  (cutoff {CUTOFF.date()})", flush=True)

# ── train causal model (deployed recipe) ─────────────────────────────────────
from xgboost import XGBRegressor
print("training XGB (causal) ...", flush=True)
m = XGBRegressor(n_estimators=600, max_depth=5, learning_rate=0.04, subsample=0.85,
                 colsample_bytree=0.85, min_child_weight=10, objective="reg:squarederror",
                 n_jobs=-1, random_state=0)
m.fit(X[tr], y[tr])

# ── portfolio sim (1-slot live exit) on OOS for a given model+thr ────────────
def portsim(qpred, thr):
    selmask = qpred >= thr
    sel = cand[te][selmask]; seld = cdr[te][selmask]; selq = qpred[selmask]
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
            if hit: t["xt"] = times[i]; ex.append(t); continue
            td = TRAIL * a
            if t["mf"] >= td and (t["mf"] - fav) >= td: t["R"] = float((t["mf"] - td) / a - sp); t["xt"] = times[i]; ex.append(t); continue
            still.append(t)
        active = still
        if i not in info: continue
        d_, q_ = info[i]
        if i - last[d_] < COOL: continue
        for t in active:
            if t["slr"] != 0 and t["d"] * (C[i] - t["ep"]) / t["a"] >= BE: t["slr"] = 0
        ei = i + 1
        if ei >= n or not (atr[i] > 0) or len(active) >= MAXC: continue
        active.append({"ei": ei, "et": times[ei], "d": d_, "ep": float(O[ei]), "a": float(atr[i]), "slr": -SL, "mf": 0.0, "R": None})
        last[d_] = i
    for t in active:
        eb = min(t["ei"] + 300, n - 1); t["R"] = float(t["d"] * (C[eb] - t["ep"]) / t["a"] - sp); ex.append(t)
    r = np.array([t["R"] for t in ex]); ets = pd.to_datetime([t["et"] for t in ex])
    w = int((r > 0).sum()); pf = r[r > 0].sum() / max(-r[r <= 0].sum(), 1e-9)
    eq = np.cumsum(r); dd = float((np.maximum.accumulate(eq) - eq).max())
    return dict(n=len(r), sumR=float(r.sum()), wr=w / len(r) * 100, pf=float(pf), dd=dd, r=r, ets=ets)

q_new = m.predict(X[te]); q_dep = DEP["q_model"].predict(X[te])

print(f"\n{'='*78}\nOOS ({CUTOFF.date()} → {ct.max().date()})  —  DEPLOYED (lookahead-trained) vs RETRAIN (causal)\n{'='*78}")
print(f"{'q_thr':>6} | {'DEPLOYED                       ':<32} | {'RETRAIN-CAUSAL':<32}")
print(f"{'':>6} | {'n   /d   sumR    WR    PF   DD':<32} | {'n   /d   sumR    WR    PF   DD':<32}")
ndays = len(set(ct[te].date))
for thr in [1.0, 2.0, 3.0, 4.0, 5.0]:
    a = portsim(q_dep, thr); b = portsim(q_new, thr)
    def fmt(s):
        if not s: return f"{'0':>3}"
        return f"{s['n']:>3} {s['n']/max(ndays,1):>4.1f} {s['sumR']:>+7.1f} {s['wr']:>5.1f}% {s['pf']:>5.2f} {s['dd']:>5.0f}"
    print(f"{thr:>6} | {fmt(a):<32} | {fmt(b):<32}")

# best-thr by-month for retrain
best = max([3.0, 4.0], key=lambda t: (portsim(q_new, t) or {'sumR': -9e9})['sumR'])
bb = portsim(q_new, best)
print(f"\nRETRAIN-CAUSAL @ q>={best} — by month:")
md = pd.DataFrame({"m": bb["ets"].to_period("M").astype(str), "R": bb["r"]})
for mo, g in md.groupby("m"):
    gpf = g.R[g.R > 0].sum() / max(-g.R[g.R <= 0].sum(), 1e-9)
    print(f"    {mo}: {len(g):>3} trd  sumR {g.R.sum():>+7.1f}  WR {(g.R>0).mean()*100:>4.0f}%  PF {gpf:.2f}")

# save retrained model + meta (NOT deployed)
pickle.dump({"q_model": m, "feat_cols": FC,
             "train_meta": {"causal_retrain": True, "cutoff": str(CUTOFF), "n_train": int(tr.sum()),
                            "n_test": int(te.sum()), "trained_at": "offline_experiment"}},
            open(OUT / "hermes_dji_causal_retrain.pkl", "wb"))
print(f"\nsaved retrained bundle → {OUT/'hermes_dji_causal_retrain.pkl'}  (NOT deployed)")
