"""
ATLAS XAU — the definitive 'is there a way' test for the hard-SL tail.

D. MID-TRADE DOOM HEAD: XGBClassifier scored mid-trade with PATH features
   (fav now, max adverse, MFE so far, % bars underwater, 10-bar slope) + the full
   29 live features AT the snapshot bar + entry pred_R. Two trigger designs:
     D30  — evaluate once at bar 30 held (aligned with the tt-trail timing)
     D2R  — evaluate at the FIRST bar the trade is <= -2R (any time)
   Cut at snapshot close if P(doom) >= tau (tau calibrated on TRAIN portfolio only).
   Doom label = trade ends at the hard SL. Reports train/test AUC per window
   (transparency: is there ANY separable signal mid-trade?).

S. ENTRY RISK-SIZING: XGBClassifier at ENTRY predicts P(hits hard SL); trade risk
   is scaled w = 1 - 0.5*phat_pctl (never exits early, just risks less on fat-tail
   entries), normalized so mean weight = 1 on train (same total risk budget).
   Judged on maxDD + Sharpe of the weighted R stream, not just sum.

Harness: identical to run_lab_losers.py — deployed tt30/0.75 base, live features,
train-only thresholds ~11/day, 1-slot cd5, DEV/HOLDOUT split, net @ $0.20.
"""
import sys, pickle, time, json
from pathlib import Path
import numpy as np, pandas as pd
from numba import njit
from xgboost import XGBRegressor, XGBClassifier

SRV = Path("/home/jay/Desktop/my-agents-and-website/commercial/server")
sys.path.insert(0, str(SRV))
import decision_engine.edge_pullback as ep
OUT = Path(__file__).parent
FC = pickle.load(open(SRV / "decision_engine/models/atlas_xau_validated.pkl", "rb"))["feat_cols"]
t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
HEAD_SP = 0.20; TARGET_PER_DAY = 11.0; COOLDOWN = 5; DEV_END = pd.Timestamp("2025-01-01")
TA, TT = 30, 0.75

df = pd.read_parquet("/home/jay/Desktop/new-model-zigzag/data/m1_xau_full.parquet")
df["time"] = pd.to_datetime(df["time"]); df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
feat = ep.compute_edge_features(df); log("features done")
atr = feat["atr14"].to_numpy(float); cdir = feat["committed_dir"].to_numpy(np.int64)
da = np.abs(feat["dist_at_signal"].to_numpy(float))
O = df["open"].to_numpy(float); H = df["high"].to_numpy(float); L = df["low"].to_numpy(float); C = df["close"].to_numpy(float)
times = df["time"].values; n = len(df)
ok = np.isfinite(atr) & (atr > 0) & (cdir != 0); ok[:300] = False; ok[-301:] = False
idx = np.where((da <= 1.0) & ok)[0]; dirs = cdir[idx].astype(np.int64)
ct = pd.to_datetime(times[idx]); Xc = feat[FC].to_numpy(np.float32)[idx]; sig_atr = atr[idx]
FEAT_ALL = feat[FC].to_numpy(np.float32)   # per-bar live features (for snapshot rows)
del feat

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
def snapshots(idxs, dirs, O, H, L, C, atr, ebar, xit, n, mode, fixed_k, uw_thr):
    """mode 0: snapshot at bar st+fixed_k if trade still open after it.
       mode 1: snapshot at FIRST bar with close-fav <= -uw_thr*a, if before exit.
       Returns snap bar (-1 none) + path stats computed over [st, snap]."""
    m = len(idxs)
    snap = np.full(m, -1, np.int64); fnow = np.zeros(m); mfs = np.zeros(m)
    mins = np.zeros(m); uwf = np.zeros(m); slp = np.zeros(m)
    for k in range(m):
        st = ebar[k]; xb = xit[k]
        if st < 0 or xb < 0: continue
        i = idxs[k]; d = dirs[k]; a = atr[i]; epr = O[st]
        sb = -1
        if mode == 0:
            if xb > st + fixed_k: sb = st + fixed_k
        else:
            end2 = min(xb - 1, st + 299)
            for jx in range(st, end2 + 1):
                if d * (C[jx] - epr) <= -uw_thr * a: sb = jx; break
        if sb < 0 or sb >= xb: continue
        mf = -1e18; mn = 1e18; uw = 0; cnt = 0
        for jx in range(st, sb + 1):
            fv = d * (C[jx] - epr)
            if fv > mf: mf = fv
            if fv < mn: mn = fv
            if fv < 0: uw += 1
            cnt += 1
        f_now = d * (C[sb] - epr)
        pb = max(st, sb - 10)
        slope = (f_now - d * (C[pb] - epr)) / a
        snap[k] = sb; fnow[k] = f_now / a; mfs[k] = mf / a
        mins[k] = mn / a; uwf[k] = uw / max(cnt, 1); slp[k] = slope
    return snap, fnow, mfs, mins, uwf, slp

@njit(cache=True)
def take(order_idx, ebar, xit, cd):
    busy = -1; m = len(order_idx); out = np.empty(m, np.int64); c = 0
    for t in range(m):
        k = order_idx[t]
        if ebar[k] <= busy: continue
        out[c] = k; busy = xit[k] + cd; c += 1
    return out[:c]

log("base sim + snapshots...")
pnlB, ebB, xitB, codeB = sim_base(idx, dirs, O, H, L, C, atr, n, 7.0, 2.0, 300, TA, TT)
SNAP = {}
SNAP["D30"] = snapshots(idx, dirs, O, H, L, C, atr, ebB, xitB, n, 0, 30, 0.0)
SNAP["D2R"] = snapshots(idx, dirs, O, H, L, C, atr, ebB, xitB, n, 1, 0, 2.0)
doom = (codeB == 0).astype(np.int8)
for nm, s in SNAP.items():
    has = s[0] >= 0
    log(f"  {nm}: snapshots on {has.mean()*100:.1f}% of candidates; "
        f"doom rate among snapped {doom[has].mean()*100:.1f}% (overall {doom.mean()*100:.1f}%)")

def snap_X(name):
    sb, fnow, mfs, mins, uwf, slp = SNAP[name]
    has = sb >= 0
    rows = FEAT_ALL[np.clip(sb, 0, n - 1)]
    path = np.stack([fnow, mfs, mins, uwf, slp,
                     (sb - ebB).astype(np.float64)], axis=1).astype(np.float32)
    X = np.concatenate([rows, path], axis=1)
    return has, np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

HASX = {nm: snap_X(nm) for nm in SNAP}
try:
    from sklearn.metrics import roc_auc_score
except Exception:
    roc_auc_score = None

WINS = []; tsw = pd.Timestamp("2020-07-01"); lastd = ct.max()
while tsw < lastd:
    WINS.append((tsw - pd.DateOffset(years=3), tsw, tsw + pd.DateOffset(months=6))); tsw += pd.DateOffset(months=6)
rng = np.random.RandomState(0)
XGBR = dict(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.85,
            colsample_bytree=0.85, min_child_weight=10, n_jobs=-1, random_state=0)
XGBC = dict(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.85,
            colsample_bytree=0.85, min_child_weight=20, n_jobs=-1, random_state=0)

def calibrate(preds, trk, tr_days):
    cand = np.quantile(preds[trk], np.linspace(0.30, 0.97, 24)); thr = cand[-1]; best = 1e18
    for th in cand:
        kk = trk[preds[trk] >= th]
        if len(kk) < 5: continue
        tk = take(kk[np.argsort(ebB[kk])].astype(np.int64), ebB, xitB, COOLDOWN)
        gap = abs(len(tk) / tr_days - TARGET_PER_DAY)
        if gap < best: best = gap; thr = th
    return thr

def port_net(kk, pnl_, xit_):
    tk = take(kk[np.argsort(ebB[kk])].astype(np.int64), ebB, xit_, COOLDOWN)
    return float((pnl_[tk] - HEAD_SP / sig_atr[tk]).sum()), tk

results = {v: [] for v in ("base_tt", "D30", "D2R")}
size_stream = {"unweighted": [], "weighted": []}
aucs = {"D30": [], "D2R": [], "ENTRY": []}
for tr_s, te_s, te_e in WINS:
    trm = (ct >= tr_s) & (ct < te_s); tem = (ct >= te_s) & (ct < te_e)
    if trm.sum() < 4000 or tem.sum() < 20: continue
    tix = np.where(trm)[0]; tix_f = tix if len(tix) <= 150_000 else rng.choice(tix, 150_000, replace=False)
    tr_days = max((ct[tix].max() - ct[tix].min()).days * 5 / 7, 1)
    te_days = max((ct[tem].max() - ct[tem].min()).days * 5 / 7, 1)
    mg = XGBRegressor(**XGBR); mg.fit(Xc[tix_f], pnlB[tix_f])
    p = mg.predict(Xc).astype(np.float64)
    thr = calibrate(p, tix, tr_days)

    kkte = np.where(tem & (p >= thr))[0]
    netB, tkB = port_net(kkte, pnlB, xitB)
    results["base_tt"].append(dict(win=str(te_s.date()), n=len(tkB), net20=netB))

    # ---- D heads
    for nm in ("D30", "D2R"):
        has, Xs = HASX[nm]
        sb, fnow = SNAP[nm][0], SNAP[nm][1]
        trs = np.where(trm & has)[0]
        if len(trs) > 120_000: trs = rng.choice(trs, 120_000, replace=False)
        mc = XGBClassifier(**XGBC); mc.fit(Xs[trs], doom[trs])
        ph = np.zeros(len(idx)); hs = np.where(has)[0]
        ph[hs] = mc.predict_proba(Xs[hs])[:, 1]
        if roc_auc_score is not None:
            tes = np.where(tem & has)[0]
            if doom[trs].any() and doom[tes].any():
                aucs[nm].append((roc_auc_score(doom[trs], ph[trs]), roc_auc_score(doom[tes], ph[tes])))
        # tau from TRAIN portfolio net: sweep cut-thresholds, keep best train net
        trg = np.where(trm & (p >= thr))[0]
        best_tau, best_net = None, None
        for q in (1.01, 0.95, 0.90, 0.80, 0.70, 0.50):   # 1.01 = never cut
            tau = np.quantile(ph[trg][ph[trg] > 0], q) if q <= 1 and (ph[trg] > 0).any() else 9e9
            cut = has & (ph >= tau)
            pnl_v = np.where(cut, fnow, pnlB); xit_v = np.where(cut, sb, xitB)
            nt, _ = port_net(trg, pnl_v, xit_v)
            if best_net is None or nt > best_net: best_net, best_tau = nt, tau
        cut = has & (ph >= best_tau)
        pnl_v = np.where(cut, fnow, pnlB); xit_v = np.where(cut, sb, xitB)
        netV, tkV = port_net(kkte, pnl_v, xit_v)
        ncut = int(cut[tkV].sum())
        results[nm].append(dict(win=str(te_s.date()), n=len(tkV), net20=netV, ncut=ncut))

    # ---- S entry risk-sizing
    me = XGBClassifier(**XGBC); me.fit(np.nan_to_num(Xc[tix_f], nan=0.0, posinf=0.0, neginf=0.0), doom[tix_f])
    pe = me.predict_proba(np.nan_to_num(Xc, nan=0.0, posinf=0.0, neginf=0.0))[:, 1]
    if roc_auc_score is not None and doom[tix_f].any() and doom[kkte].any():
        aucs["ENTRY"].append((roc_auc_score(doom[tix_f], pe[tix_f]), roc_auc_score(doom[kkte], pe[kkte])))
    trg = np.where(trm & (p >= thr))[0]
    lo, hi = np.quantile(pe[trg], 0.1), np.quantile(pe[trg], 0.9)
    def wgt(x):
        z = np.clip((x - lo) / max(hi - lo, 1e-9), 0, 1)
        return 1.25 - 0.75 * z          # 1.25x low-risk .. 0.5x high-risk
    wtr = wgt(pe[trg]); norm = wtr.mean()
    netU, tkU = port_net(kkte, pnlB, xitB)
    w = wgt(pe[tkU]) / norm
    nhU = pnlB[tkU] - HEAD_SP / sig_atr[tkU]
    for t_, r_, w_ in zip(times[xitB[tkU]], nhU, w):
        size_stream["unweighted"].append((t_, float(r_))); size_stream["weighted"].append((t_, float(r_ * w_)))
    log(f"window {te_s.date()} done")

def agg(rows, dev):
    rr = [r for r in rows if r.get("n", 0) > 0 and ((pd.Timestamp(r["win"]) < DEV_END) == dev)]
    if not rr: return None
    return dict(nwin=len(rr), n=sum(r["n"] for r in rr), net20=sum(r["net20"] for r in rr),
                w=sum(1 for r in rr if r["net20"] > 0), ncut=sum(r.get("ncut", 0) for r in rr))

print(f"\nAUC (train -> test, median across windows):")
for nm, aa in aucs.items():
    if aa: print(f"  {nm:<6} {np.median([a[0] for a in aa]):.3f} -> {np.median([a[1] for a in aa]):.3f}   ({len(aa)} windows)")

for dev, label in [(True, "DEV (selection)"), (False, "HOLDOUT (untouched)")]:
    print(f"\n{'='*66}\nDOOM-HEAD lab — {label}  (net @ $0.20)\n{'='*66}")
    print(f"{'variant':<12}{'n':>7}{'net@20c':>10}{'w+':>6}{'cuts':>7}")
    for v in ("base_tt", "D30", "D2R"):
        a = agg(results[v], dev)
        if a is None: continue
        print(f"{v:<12}{a['n']:>7}{a['net20']:>+10.0f}{a['w']:>4}/{a['nwin']:<2}{a['ncut']:>7}")

print(f"\nENTRY RISK-SIZING (same trades, weighted risk, mean weight normalized to 1):")
for nm in ("unweighted", "weighted"):
    for dev, lab in [(True, "dev"), (False, "holdout")]:
        tr = sorted([z for z in size_stream[nm] if (pd.Timestamp(z[0]) < DEV_END) == dev], key=lambda z: z[0])
        if not tr: continue
        r = np.array([x for _, x in tr]); eq = np.cumsum(r)
        dd = float((np.maximum.accumulate(eq) - eq).max())
        daily = pd.Series(r, index=pd.to_datetime([t for t, _ in tr])).resample("1D").sum()
        daily = daily[daily != 0]
        sh = float(daily.mean() / max(daily.std(), 1e-9) * np.sqrt(252))
        print(f"  {nm:<11} {lab:<8} sum {eq[-1]:>+8.0f}R   maxDD {dd:>6.0f}R   dailySharpe {sh:>5.2f}")
json.dump({k: v for k, v in results.items()}, open(OUT / "doomhead_results.json", "w"), default=str, indent=1)
log("doomhead lab done")
