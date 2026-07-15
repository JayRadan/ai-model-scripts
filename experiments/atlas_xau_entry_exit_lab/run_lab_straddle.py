"""
STRADDLE LAB — Jay 2026-07-15: XAU HFT-style. Buy-stop ABOVE + sell-stop BELOW
at the same time (close +/- k*ATR); first trigger wins (OCO), trail t*ATR until
closed, then re-arm. In-and-out fast, both sides always covered.

Round mechanics (per candidate bar i, ATR a = atr[i]):
  levels: up = C[i]+k*a, dn = C[i]-k*a; armed for W bars, else expires
  trigger bar: if BOTH levels hit in the same bar -> worst case: filled and
  stopped for -t (conservative); else fill at the level, hard stop t*a intrabar,
  close-based trail t*a giveback, max maxh bars from trigger
Variants per (k, t):
  mech       — every bar re-arms (pure mechanical, 1-slot sequential)
  mean30/q10_30 — XGB (mean / 10th-pct objective) predicts round R at the ARM
  bar from the 29 live feats; arm only when pred >= thr (train-calibrated to
  ~30/day); q10_10 = q10 at ~10/day
Honest harness: 3y train / 6mo test, train-only thr, 1-slot cd1,
DEV(<2025)/HOLDOUT, spread grid $0.10/0.20/0.30 per round (charged /ATR).
"""
import sys, pickle, time, json
from pathlib import Path
import numpy as np, pandas as pd
from numba import njit
from xgboost import XGBRegressor

SRV = Path("/home/jay/Desktop/my-agents-and-website/commercial/server")
sys.path.insert(0, str(SRV))
import decision_engine.edge_pullback as ep
OUT = Path(__file__).parent
FC = pickle.load(open(SRV / "decision_engine/models/hermes_dji_validated.pkl", "rb"))["feat_cols"]
t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
DEV_END = pd.Timestamp("2025-01-01"); CD = 1; W = 15; MAXH = 120
SPREADS = (0.10, 0.20, 0.30)
GRID = [(0.5, 0.5), (0.5, 1.0), (1.0, 1.0), (1.0, 2.0), (1.5, 1.0)]

@njit(cache=True)
def sim_straddle(O, H, L, C, atr, n, k, t, W, MAXH):
    """One round per bar i. Returns pnl (R in arm-bar ATR), trig(+1/-1/0),
    ebar (trigger bar, -1 none), xbar (exit bar)."""
    pnl = np.zeros(n); trig = np.zeros(n, np.int64)
    ebar = np.full(n, -1, np.int64); xbar = np.full(n, -1, np.int64)
    for i in range(300, n - MAXH - W - 2):
        a = atr[i]
        if not (a > 0): continue
        up = C[i] + k * a; dn = C[i] - k * a
        tb = -1; d = 0
        for j in range(i + 1, i + W + 1):
            hu = H[j] >= up; hd = L[j] <= dn
            if hu and hd:
                # both levels swept in one bar: worst case = filled then stopped
                pnl[i] = -t; trig[i] = 2; ebar[i] = j; xbar[i] = j
                tb = -2; break
            if hu: tb = j; d = 1; break
            if hd: tb = j; d = -1; break
        if tb < 0: continue
        epr = up if d == 1 else dn
        hard = t * a; mf = 0.0
        end = min(tb + MAXH, n - 1); done = False
        for j in range(tb, end + 1):
            adv = (epr - L[j]) if d == 1 else (H[j] - epr)
            if adv >= hard:
                pnl[i] = -t; xbar[i] = j; done = True; break
            fav = d * (C[j] - epr)
            if fav > mf: mf = fav
            if mf >= hard and (mf - fav) >= hard:
                pnl[i] = (mf - hard) / a; xbar[i] = j; done = True; break
        if not done:
            pnl[i] = d * (C[end] - epr) / a; xbar[i] = end
        trig[i] = d; ebar[i] = tb
    return pnl, trig, ebar, xbar

@njit(cache=True)
def take(order_idx, ebar, xbar, cd):
    busy = -1; m = len(order_idx); out = np.empty(m, np.int64); c = 0
    for q in range(m):
        i = order_idx[q]
        if ebar[i] <= busy: continue
        out[c] = i; busy = xbar[i] + cd; c += 1
    return out[:c]

XGBM = dict(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.85,
            colsample_bytree=0.85, min_child_weight=10, n_jobs=-1, random_state=0)
XGBQ = dict(XGBM, objective="reg:quantileerror", quantile_alpha=0.10)
rng = np.random.RandomState(0)

df = pd.read_parquet("/home/jay/Desktop/new-model-zigzag/data/m1_xau_full.parquet")
df = df.rename(columns={[c for c in df.columns if "time" in c.lower()][0]: "time"})
df["time"] = pd.to_datetime(df["time"]); df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
if "tick_volume" not in df.columns: df["tick_volume"] = df.get("volume", 0)
feat = ep.compute_edge_features(df); log("features done")
atr = feat["atr14"].to_numpy(float)
O = df["open"].to_numpy(float); H = df["high"].to_numpy(float); L = df["low"].to_numpy(float); C = df["close"].to_numpy(float)
times = df["time"].values; n = len(df)
FEAT1 = np.nan_to_num(feat[FC].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
del feat
ct_all = pd.to_datetime(times)

RES = {}
for (k, t) in GRID:
    log(f"=== straddle k={k} t={t} ===")
    pnl, trig, eb, xb = sim_straddle(O, H, L, C, atr, n, k, t, W, MAXH)
    cand = np.where(trig != 0)[0]          # rounds that actually triggered
    log(f"  {len(cand):,} triggered rounds ({100*len(cand)/n:.0f}% of bars), "
        f"mean gross {pnl[cand].mean():+.4f}R, both-swept {100*(trig[cand]==2).mean():.1f}%")
    ct = ct_all[cand]; sa = atr[cand]
    Xc = FEAT1[cand]; pv = pnl[cand]
    WINS = []; tsw = pd.Timestamp("2020-07-01"); lastd = ct.max()
    while tsw < lastd:
        WINS.append((tsw - pd.DateOffset(years=3), tsw, tsw + pd.DateOffset(months=6))); tsw += pd.DateOffset(months=6)
    res = {v: [] for v in ("mech", "mean30", "q10_30", "q10_10")}
    for tr_s, te_s, te_e in WINS:
        trm = (ct >= tr_s) & (ct < te_s); tem = (ct >= te_s) & (ct < te_e)
        if trm.sum() < 4000 or tem.sum() < 20: continue
        tix = np.where(trm)[0]; tix_f = tix if len(tix) <= 150_000 else rng.choice(tix, 150_000, replace=False)
        tr_days = max((ct[tix].max() - ct[tix].min()).days * 5 / 7, 1)
        preds = {}
        mm = XGBRegressor(**XGBM); mm.fit(Xc[tix_f], pv[tix_f]); preds["mean"] = mm.predict(Xc).astype(np.float64)
        mq = XGBRegressor(**XGBQ); mq.fit(Xc[tix_f], pv[tix_f]); preds["q10"] = mq.predict(Xc).astype(np.float64)
        for vn, pk, tpd in (("mech", None, None), ("mean30", "mean", 30.0),
                            ("q10_30", "q10", 30.0), ("q10_10", "q10", 10.0)):
            if pk is None:
                sel_te = np.where(tem)[0]
            else:
                p = preds[pk]
                candq = np.quantile(p[tix], np.linspace(0.30, 0.97, 24)); thr = candq[-1]; best = 1e18
                for th in candq:
                    kk = tix[p[tix] >= th]
                    if len(kk) < 5: continue
                    tk = take(cand[kk][np.argsort(eb[cand[kk]])], eb, xb, CD)
                    gap = abs(len(tk) / tr_days - tpd)
                    if gap < best: best = gap; thr = th
                sel_te = np.where(tem & (p >= thr))[0]
            tk = take(cand[sel_te][np.argsort(eb[cand[sel_te]])], eb, xb, CD)
            gross = pnl[tk]
            row = dict(win=str(te_s.date()), n=len(tk))
            for sp in SPREADS:
                row[f"net{sp}"] = float((gross - sp / atr[tk]).sum())
            row["wr"] = float((gross > 0).mean()) if len(tk) else 0.0
            res[vn].append(row)
        log(f"  window {te_s.date()} done")
    def agg(rows, dev):
        rr = [r for r in rows if (pd.Timestamp(r["win"]) < DEV_END) == dev and r["n"] > 0]
        if not rr: return None
        d = dict(nwin=len(rr), n=sum(r["n"] for r in rr),
                 wr=100 * np.average([r["wr"] for r in rr], weights=[r["n"] for r in rr]))
        for sp in SPREADS:
            d[f"net{sp}"] = sum(r[f"net{sp}"] for r in rr)
            d[f"w{sp}"] = sum(1 for r in rr if r[f"net{sp}"] > 0)
        return d
    print(f"\n===== straddle k={k} t={t} (XAU M1) =====")
    for dev, lab in [(True, "DEV"), (False, "HOLDOUT")]:
        print(f"--- {lab} ---")
        for vn in res:
            a = agg(res[vn], dev)
            if a is None: continue
            print(f"  {vn:<8} n={a['n']:>7} WR {a['wr']:4.1f}% " + " ".join(
                f"net@{sp}:{a[f'net{sp}']:>+9.0f}({a[f'w{sp}']}/{a['nwin']})" for sp in SPREADS))
    RES[f"{k}|{t}"] = res

json.dump(RES, open(OUT / "straddle_results.json", "w"), default=str, indent=1)
log("straddle lab done")
