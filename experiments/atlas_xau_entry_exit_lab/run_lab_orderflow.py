"""
ATLAS XAU — order-flow entry upgrade for edge_pullback: does adding the 14
tick-orderflow features (imbalance, VPIN, signed flow, persistence...) to the
29-feature entry model improve ranking — more net R and FEWER hard-SL hitters
among taken trades? This is the 'new feature class' lever: loss control lives at
the entry gate (mid-trade cuts proven unprofitable even with AUC 0.89 foresight).

Model A = 29 live feats (deployed) | Model B = 29 + 14 flow feats.
Exit fixed at deployed tt30/0.75. Same honest harness: train-only thr ~11/day,
1-slot cd5, DEV 2020-07..2024-12 / HOLDOUT 2025-01+, net @ $0.10/0.20/0.30.
Also reports per-model doom-rate (SL-hits per taken trade) + doom AUC with flow.
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
FC = pickle.load(open(SRV / "decision_engine/models/atlas_xau_validated.pkl", "rb"))["feat_cols"]
FLOW = ["imbalance_ratio", "bid_ask_vol_ratio", "vpin_proxy", "median_spread",
        "cum_signed_5", "flow_persistence_5", "cum_signed_15", "flow_persistence_15",
        "cum_signed_60", "flow_persistence_60", "spread_vol_50", "tick_intensity_50",
        "signed_flow", "n_ticks"]
t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
SPREADS = [0.10, 0.20, 0.30]; HEAD_SP = 0.20
TARGET_PER_DAY = 11.0; COOLDOWN = 5; DEV_END = pd.Timestamp("2025-01-01")
TA, TT = 30, 0.75

df = pd.read_parquet("/home/jay/Desktop/new-model-zigzag/data/m1_xau_orderflow_8y.parquet")
df["time"] = pd.to_datetime(df["time"]); df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
log(f"{len(df):,} bars {df.time.iloc[0].date()} -> {df.time.iloc[-1].date()}")
flow_df = df[FLOW].copy()
feat = ep.compute_edge_features(df[["time", "open", "high", "low", "close", "tick_volume"]].copy())
log("features done")
atr = feat["atr14"].to_numpy(float); cdir = feat["committed_dir"].to_numpy(np.int64)
da = np.abs(feat["dist_at_signal"].to_numpy(float))
O = df["open"].to_numpy(float); H = df["high"].to_numpy(float); L = df["low"].to_numpy(float); C = df["close"].to_numpy(float)
times = df["time"].values; n = len(df)
ok = np.isfinite(atr) & (atr > 0) & (cdir != 0); ok[:300] = False; ok[-301:] = False
idx = np.where((da <= 1.0) & ok)[0]; dirs = cdir[idx].astype(np.int64)
ct = pd.to_datetime(times[idx]); sig_atr = atr[idx]
XA = np.nan_to_num(feat[FC].to_numpy(np.float32)[idx], nan=0.0, posinf=0.0, neginf=0.0)
XB = np.concatenate([XA, np.nan_to_num(flow_df.to_numpy(np.float32)[idx], nan=0.0, posinf=0.0, neginf=0.0)], axis=1)
del feat, flow_df
log(f"candidates {len(idx):,}  XA {XA.shape[1]} feats, XB {XB.shape[1]} feats")

@njit(cache=True)
def sim_base(idxs, dirs, O, H, L, C, atr, n, SL, TRAIL, MAXH, ta, tt):
    m = len(idxs); pnl = np.zeros(m); ebar = np.full(m, -1, np.int64)
    xit = np.full(m, -1, np.int64); code = np.full(m, -1, np.int64)
    for k in range(m):
        i = idxs[k]; d = dirs[k]; a = atr[i]
        if i + 1 >= n or not (a > 0): continue
        st = i + 1; epr = O[st]; hard = SL * a; mf = 0.0
        end = min(st + MAXH, n - 1); done = False
        for jx in range(st, end + 1):
            adv = (epr - L[jx]) if d == 1 else (H[jx] - epr)
            if adv >= hard: pnl[k] = -SL; xit[k] = jx; code[k] = 0; done = True; break
            fav = d * (C[jx] - epr)
            if fav > mf: mf = fav
            trd = TRAIL * a
            if ta > 0 and (jx - st) >= ta:
                w = tt * a
                if w < trd: trd = w
            if mf >= trd and (mf - fav) >= trd: pnl[k] = (mf - trd) / a; xit[k] = jx; code[k] = 1; done = True; break
        if not done: pnl[k] = d * (C[end] - epr) / a; xit[k] = end; code[k] = 2
        ebar[k] = st
    return pnl, ebar, xit, code

@njit(cache=True)
def take(order_idx, ebar, xit, cd):
    busy = -1; m = len(order_idx); out = np.empty(m, np.int64); c = 0
    for t in range(m):
        k = order_idx[t]
        if ebar[k] <= busy: continue
        out[c] = k; busy = xit[k] + cd; c += 1
    return out[:c]

pnlB, ebB, xitB, codeB = sim_base(idx, dirs, O, H, L, C, atr, n, 7.0, 2.0, 300, TA, TT)
doom = (codeB == 0)
log(f"labels done, overall doom rate {doom.mean()*100:.1f}%")

try:
    from sklearn.metrics import roc_auc_score
except Exception:
    roc_auc_score = None

WINS = []; tsw = pd.Timestamp("2020-07-01"); lastd = ct.max()
while tsw < lastd:
    WINS.append((tsw - pd.DateOffset(years=3), tsw, tsw + pd.DateOffset(months=6))); tsw += pd.DateOffset(months=6)
rng = np.random.RandomState(0)
XGB = dict(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.85,
           colsample_bytree=0.85, min_child_weight=10, n_jobs=-1, random_state=0)

def calibrate(preds, trk, tr_days):
    cand = np.quantile(preds[trk], np.linspace(0.30, 0.97, 24)); thr = cand[-1]; best = 1e18
    for th in cand:
        kk = trk[preds[trk] >= th]
        if len(kk) < 5: continue
        tk = take(kk[np.argsort(ebB[kk])].astype(np.int64), ebB, xitB, COOLDOWN)
        gap = abs(len(tk) / tr_days - TARGET_PER_DAY)
        if gap < best: best = gap; thr = th
    return thr

results = {"A_29feats (deployed)": [], "B_+orderflow": []}
aucs = []
for tr_s, te_s, te_e in WINS:
    trm = (ct >= tr_s) & (ct < te_s); tem = (ct >= te_s) & (ct < te_e)
    if trm.sum() < 4000 or tem.sum() < 20: continue
    tix = np.where(trm)[0]; tix_f = tix if len(tix) <= 150_000 else rng.choice(tix, 150_000, replace=False)
    tr_days = max((ct[tix].max() - ct[tix].min()).days * 5 / 7, 1)
    te_days = max((ct[tem].max() - ct[tem].min()).days * 5 / 7, 1)
    for vname, X in [("A_29feats (deployed)", XA), ("B_+orderflow", XB)]:
        m = XGBRegressor(**XGB); m.fit(X[tix_f], pnlB[tix_f])
        p = m.predict(X).astype(np.float64)
        thr = calibrate(p, tix, tr_days)
        kk = np.where(tem & (p >= thr))[0]
        if len(kk) == 0: results[vname].append(dict(win=str(te_s.date()), n=0)); continue
        tk = take(kk[np.argsort(ebB[kk])].astype(np.int64), ebB, xitB, COOLDOWN)
        R = pnlB[tk]; cost = 1.0 / sig_atr[tk]; nh = R - HEAD_SP * cost
        results[vname].append(dict(
            win=str(te_s.date()), n=len(tk), perday=len(tk) / te_days,
            **{f"net{int(sp*100)}": float((R - sp * cost).sum()) for sp in SPREADS},
            wr=float((nh > 0).mean() * 100), doomr=float(doom[tk].mean() * 100)))
        # rank-corr of test preds with doom (does flow rank SL-hitters lower?)
        if vname == "B_+orderflow" and roc_auc_score is not None and doom[kk].any():
            aucs.append(roc_auc_score(doom[kk], -p[kk]))
    log(f"window {te_s.date()} done")

def agg(rows, dev):
    rr = [r for r in rows if r.get("n", 0) > 0 and ((pd.Timestamp(r["win"]) < DEV_END) == dev)]
    if not rr: return None
    d = dict(nwin=len(rr), n=sum(r["n"] for r in rr), perday=float(np.mean([r["perday"] for r in rr])))
    for sp in SPREADS:
        key = f"net{int(sp*100)}"
        d[key] = sum(r[key] for r in rr); d[key + "_w"] = sum(1 for r in rr if r[key] > 0)
    d["wr"] = float(np.average([r["wr"] for r in rr], weights=[r["n"] for r in rr]))
    d["doomr"] = float(np.average([r["doomr"] for r in rr], weights=[r["n"] for r in rr]))
    return d

hdr = f"{'model':<22}{'n':>7}{'/day':>6}{'net@10c':>9}{'net@20c':>9}{'net@30c':>9}{'w+@20':>7}{'WR%':>6}{'SLhit%':>8}"
for dev, label in [(True, "DEV (selection)"), (False, "HOLDOUT (untouched)")]:
    print(f"\n{'='*86}\nORDER-FLOW entry lab, XAU — {label}\n{'='*86}\n{hdr}")
    for vname in results:
        a = agg(results[vname], dev)
        if a is None: continue
        print(f"{vname:<22}{a['n']:>7}{a['perday']:>6.1f}{a['net10']:>+9.0f}{a['net20']:>+9.0f}"
              f"{a['net30']:>+9.0f}{a['net20_w']:>4}/{a['nwin']:<2}{a['wr']:>6.0f}{a['doomr']:>8.1f}")
if aucs: print(f"\nflow-model anti-doom ranking AUC on test candidates (median): {np.median(aucs):.3f}")
json.dump(results, open(OUT / "orderflow_results.json", "w"), default=str, indent=1)
log("orderflow lab done")
