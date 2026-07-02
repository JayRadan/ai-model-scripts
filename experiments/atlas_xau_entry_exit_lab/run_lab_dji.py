"""
HERMES DJI — same entry/exit lab as atlas_xau (identical edge_pullback engine).
Tests whether the time-boxed-patience trail (XAU winner, deployed 2026-07-02)
transfers to DJI, plus the same entry variants. Honest harness: live feature
pipeline, 6mo test windows, 3y rolling train, train-only thr ~11/day, 1-slot cd5,
DEV = 2020-07..2024-12 (selection) / HOLDOUT = 2025-01..2026-05 (untouched).
Spread charged per trade in INDEX POINTS / ATR(signal): grid 1.0 / 1.5 / 2.0 pt
(campaign convention; DJI breakeven was ~3pt). Headline = 1.5pt.
"""
import sys, pickle, time, json
from pathlib import Path
import numpy as np, pandas as pd
from numba import njit
from xgboost import XGBRegressor
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

SRV = Path("/home/jay/Desktop/my-agents-and-website/commercial/server")
sys.path.insert(0, str(SRV))
import decision_engine.edge_pullback as ep
OUT = Path(__file__).parent
FC = pickle.load(open(SRV / "decision_engine/models/hermes_dji_validated.pkl", "rb"))["feat_cols"]
t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)

SPREADS = [1.0, 1.5, 2.0]      # index points
HEAD_SP = 1.5
TARGET_PER_DAY = 11.0; COOLDOWN = 5
DEV_END = pd.Timestamp("2025-01-01")

df = pd.read_parquet("/home/jay/Desktop/new-model-zigzag/data/m1_dji_full.parquet")
df = df.rename(columns={[c for c in df.columns if "time" in c.lower()][0]: "time"})
df["time"] = pd.to_datetime(df["time"]); df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
if "tick_volume" not in df.columns: df["tick_volume"] = df.get("volume", 0)
log(f"{len(df):,} bars {df.time.iloc[0].date()} -> {df.time.iloc[-1].date()}")
feat = ep.compute_edge_features(df)
log("features done")

atr = feat["atr14"].to_numpy(float); cdir = feat["committed_dir"].to_numpy(np.int64)
da = np.abs(feat["dist_at_signal"].to_numpy(float))
O = df["open"].to_numpy(float); H = df["high"].to_numpy(float); L = df["low"].to_numpy(float); C = df["close"].to_numpy(float)
times = df["time"].values; n = len(df)
ok = np.isfinite(atr) & (atr > 0) & (cdir != 0); ok[:300] = False; ok[-301:] = False
idx = np.where((da <= 1.0) & ok)[0]; dirs = cdir[idx].astype(np.int64)
ct = pd.to_datetime(times[idx]); Xc = feat[FC].to_numpy(np.float32)[idx]
sig_atr = atr[idx]
log(f"candidates {len(idx):,} | median ATR {np.median(sig_atr):.2f}pt "
    f"(2025+: {np.median(sig_atr[ct >= '2025-01-01']):.2f}pt)")

@njit(cache=True)
def sim_u(idxs, dirs, O, H, L, C, atr, n,
          entry_mode, delta, ttl, SL, TRAIL, MAXH,
          be_peak, be_floor, tight_after, tight_trail):
    m = len(idxs)
    filled = np.zeros(m, np.uint8); pnl = np.zeros(m); ebar = np.full(m, -1, np.int64); xit = np.full(m, -1, np.int64)
    for k in range(m):
        i = idxs[k]; d = dirs[k]; a = atr[i]
        if i + 1 >= n or not (a > 0): continue
        if entry_mode == 0:
            st = i + 1; epr = O[st]
        elif entry_mode == 1:
            base = O[i + 1]; lp = base - d * delta * a; st = -1
            lim_end = min(i + ttl, n - 1)
            for jx in range(i + 1, lim_end + 1):
                if (d == 1 and L[jx] <= lp) or (d == -1 and H[jx] >= lp):
                    st = jx; break
            if st < 0: continue
            epr = lp
        else:
            b1 = i + 1
            if b1 + 1 >= n: continue
            if d * (C[b1] - O[b1]) <= 0: continue
            st = b1 + 1; epr = O[st]
        hard = SL * a; mf = 0.0; locked = False
        end = min(st + MAXH, n - 1); done = False
        for jx in range(st, end + 1):
            if locked:
                fp = epr + d * be_floor * a
                if (d == 1 and L[jx] <= fp) or (d == -1 and H[jx] >= fp):
                    pnl[k] = be_floor; xit[k] = jx; done = True; break
            else:
                adv = (epr - L[jx]) if d == 1 else (H[jx] - epr)
                if adv >= hard:
                    pnl[k] = -SL; xit[k] = jx; done = True; break
            fav = d * (C[jx] - epr)
            if fav > mf: mf = fav
            trd = TRAIL * a
            if tight_after > 0 and (jx - st) >= tight_after:
                tt = tight_trail * a
                if tt < trd: trd = tt
            if mf >= trd and (mf - fav) >= trd:
                pnl[k] = (mf - trd) / a; xit[k] = jx; done = True; break
            if be_peak > 0.0 and mf >= be_peak * a: locked = True
        if not done:
            pnl[k] = d * (C[end] - epr) / a; xit[k] = end
        filled[k] = 1; ebar[k] = st
    return filled, pnl, ebar, xit

@njit(cache=True)
def take(order_idx, ebar, xit, cd):
    busy = -1; m = len(order_idx); out = np.empty(m, np.int64); c = 0
    for t in range(m):
        k = order_idx[t]
        if ebar[k] <= busy: continue
        out[c] = k; busy = xit[k] + cd; c += 1
    return out[:c]

def SIM(em=0, delta=0.0, ttl=5, SL=7.0, TR=2.0, MH=300, bp=0.0, bf=0.0, ta=0, tt=0.0):
    return sim_u(idx, dirs, O, H, L, C, atr, n, em, delta, ttl, SL, TR, MH, bp, bf, ta, tt)

log("global sims...")
SIMS = {}
SIMS["base"]     = SIM()                       # deployed (SL7/trail2)
SIMS["tt30_075"] = SIM(ta=30, tt=0.75)         # XAU deployed pick
SIMS["tt30_05"]  = SIM(ta=30, tt=0.5)
SIMS["tt20_05"]  = SIM(ta=20, tt=0.5)
SIMS["tt45_075"] = SIM(ta=45, tt=0.75)
SIMS["tt60_10"]  = SIM(ta=60, tt=1.0)
SIMS["uni1_05"]  = SIM(ta=1, tt=0.5)           # mechanism check: ~uniform tight
SIMS["belock"]   = SIM(bp=2.0, bf=0.2)
SIMS["lim030"]   = SIM(em=1, delta=0.30)
SIMS["confirm"]  = SIM(em=2)
for nm, (f, p, e, x) in SIMS.items():
    log(f"  {nm:<10} fill {f.mean()*100:5.1f}%  mean gross R {p[f==1].mean():+.3f}")

VARIANTS = [
    ("V0_deployed", "base",     "mean"),
    ("tt30_075",    "tt30_075", "mean"),
    ("tt30_05",     "tt30_05",  "mean"),
    ("tt20_05",     "tt20_05",  "mean"),
    ("tt45_075",    "tt45_075", "mean"),
    ("tt60_10",     "tt60_10",  "mean"),
    ("uni1_05",     "uni1_05",  "mean"),
    ("X_belock",    "belock",   "mean"),
    ("E_lim030",    "lim030",   "mean"),
    ("E_confirm",   "confirm",  "mean"),
]

WINS = []; tsw = pd.Timestamp("2020-07-01"); lastd = ct.max()
while tsw < lastd:
    WINS.append((tsw - pd.DateOffset(years=3), tsw, tsw + pd.DateOffset(months=6))); tsw += pd.DateOffset(months=6)
rng = np.random.RandomState(0)
XGB = dict(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.85,
           colsample_bytree=0.85, min_child_weight=10, n_jobs=-1, random_state=0)

def calibrate(preds, trm, tr_days, ff, eb_, xit_):
    trk = np.where(trm & (ff == 1))[0]
    if len(trk) < 50: return np.inf
    cand = np.quantile(preds[trk], np.linspace(0.30, 0.97, 24)); thr = cand[-1]; best = 1e18
    for th in cand:
        kk = trk[preds[trk] >= th]
        if len(kk) < 5: continue
        tk = take(kk[np.argsort(eb_[kk])].astype(np.int64), eb_, xit_, COOLDOWN)
        gap = abs(len(tk) / tr_days - TARGET_PER_DAY)
        if gap < best: best = gap; thr = th
    return thr

results = {v[0]: [] for v in VARIANTS}; trades = {v[0]: [] for v in VARIANTS}
base_pnl = SIMS["base"][1]
for tr_s, te_s, te_e in WINS:
    trm = (ct >= tr_s) & (ct < te_s); tem = (ct >= te_s) & (ct < te_e)
    if trm.sum() < 4000 or tem.sum() < 20: continue
    tix = np.where(trm)[0]; tix_f = tix if len(tix) <= 150_000 else rng.choice(tix, 150_000, replace=False)
    tr_days = max((ct[tix].max() - ct[tix].min()).days * 5 / 7, 1)
    te_days = max((ct[tem].max() - ct[tem].min()).days * 5 / 7, 1)
    m = XGBRegressor(**XGB); m.fit(Xc[tix_f], base_pnl[tix_f])
    p_mean = m.predict(Xc).astype(np.float64)
    for vname, skey, gate in VARIANTS:
        ff, pnl_, eb_, xit_ = SIMS[skey]
        thr = calibrate(p_mean, trm, tr_days, ff, eb_, xit_)
        kk = np.where(tem & (ff == 1) & (p_mean >= thr))[0]
        if len(kk) == 0:
            results[vname].append(dict(win=str(te_s.date()), n=0)); continue
        tk = take(kk[np.argsort(eb_[kk])].astype(np.int64), eb_, xit_, COOLDOWN)
        R = pnl_[tk]; cost = 1.0 / sig_atr[tk]
        nets = {sp: R - sp * cost for sp in SPREADS}
        nh = nets[HEAD_SP]
        results[vname].append(dict(
            win=str(te_s.date()), n=len(tk), perday=len(tk) / te_days, gross=R.sum(),
            **{f"net{sp}": nets[sp].sum() for sp in SPREADS},
            wr=(nh > 0).mean() * 100,
            pf=nh[nh > 0].sum() / max(-nh[nh <= 0].sum(), 1e-9)))
        for tt_, r_ in zip(times[xit_[tk]], nh): trades[vname].append((tt_, r_))
    log(f"window {te_s.date()} done")

def agg(rows, dev):
    rr = [r for r in rows if r.get("n", 0) > 0 and ((pd.Timestamp(r["win"]) < DEV_END) == dev)]
    if not rr: return None
    d = dict(nwin=len(rr), n=sum(r["n"] for r in rr), perday=np.mean([r["perday"] for r in rr]),
             gross=sum(r["gross"] for r in rr))
    for sp in SPREADS:
        d[f"net{sp}"] = sum(r[f"net{sp}"] for r in rr)
        d[f"net{sp}_w"] = sum(1 for r in rr if r[f"net{sp}"] > 0)
    d["wr"] = np.average([r["wr"] for r in rr], weights=[r["n"] for r in rr])
    return d

def maxdd(vname, dev):
    tr = [(t, r) for t, r in trades[vname] if (pd.Timestamp(t) < DEV_END) == dev]
    if not tr: return 0.0
    tr.sort(key=lambda z: z[0]); eq = np.cumsum([r for _, r in tr])
    return float((np.maximum.accumulate(eq) - eq).max())

for dev, label in [(True, "DEV windows (2020-07 .. 2024-12) — selection"),
                   (False, "HOLDOUT windows (2025-01 .. 2026-05) — untouched")]:
    print(f"\n{'='*96}\nHERMES DJI lab — {label}\n{'='*96}")
    print(f"{'variant':<14}{'n':>7}{'/day':>6}{'gross':>8}{'net@1pt':>9}{'net@1.5':>9}{'net@2pt':>9}{'w+@1.5':>7}{'WR%':>6}{'DD@1.5':>8}")
    for vname, _, _ in VARIANTS:
        a = agg(results[vname], dev)
        if a is None: continue
        print(f"{vname:<14}{a['n']:>7}{a['perday']:>6.1f}{a['gross']:>+8.0f}{a['net1.0']:>+9.0f}{a['net1.5']:>+9.0f}"
              f"{a['net2.0']:>+9.0f}{a['net1.5_w']:>4}/{a['nwin']:<2}{a['wr']:>6.0f}{maxdd(vname, dev):>8.0f}")

# equity PNG: deployed vs tt30/0.75 (the XAU-deployed config — pre-registered pick)
fig, ax = plt.subplots(figsize=(13, 5.5))
for vname, col in [("V0_deployed", "#64748b"), ("tt30_075", "#16a34a"), ("tt30_05", "#2563eb")]:
    tr = sorted(trades[vname], key=lambda z: z[0])
    tt_ = pd.to_datetime([t for t, _ in tr]); eq = np.cumsum([r for _, r in tr])
    dd = float((np.maximum.accumulate(eq) - eq).max())
    ax.plot(tt_, eq, lw=1.2, color=col, label=f"{vname} ({eq[-1]:+.0f}R, maxDD {dd:.0f}R)")
ax.axvline(DEV_END, color="#dc2626", lw=1, ls="--"); ax.axvspan(DEV_END, pd.Timestamp("2026-06-01"), alpha=0.06, color="#dc2626")
ax.text(DEV_END, ax.get_ylim()[0], " HOLDOUT →", color="#dc2626", fontsize=9, va="bottom")
ax.set_title("Hermes DJI — deployed exit vs time-boxed-patience trail | WF net @ 1.5pt spread, ~11/day, 1-slot")
ax.axhline(0, color="k", lw=0.6); ax.grid(alpha=0.3); ax.legend(loc="upper left")
plt.tight_layout(); plt.savefig(OUT / "dji_lab_equity.png", dpi=110)
print(f"\nequity -> {OUT/'dji_lab_equity.png'}")
json.dump({v: results[v] for v, _, _ in VARIANTS}, open(OUT / "dji_lab_results.json", "w"), default=str, indent=1)
log("dji lab done")
