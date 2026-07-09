"""
HEDGE LAB — Jay: "start hedging bad trades". Instead of cutting a doomed trade
(proven negative: run_lab_losers/doomhead), OPEN AN OPPOSITE trade with its own
exits when doom is detected. P&L is additive, so hedging helps iff the hedge
stream itself is net positive after spread. Hedge != cut because the hedge can
ride the adverse move BEYOND the original's hard SL and exit early (small loss)
if the original recovers.

Triggers (one hedge max per original trade, at first trigger):
  d2r    — first bar the trade floats <= -2R (naive, no ML)
  d3r    — first bar <= -3R (naive)
  d30uw  — bar 30 held and still underwater (naive)
  d2r_ml — d2r snapshot + walk-forward doom-head P(doom) >= tau (tau train-picked)
  d30_ml — bar-30 snapshot + P(doom) >= tau
Hedge exits (independent of the original trade):
  tie    — close when the original closes (== early-cut, sanity: must be ~negative)
  t2     — SL 2xATR, trail 2xATR, maxh 300
  tt2    — SL 2, trail 2 -> tighten 0.75 after 30 bars (deployed exit shape)
  tt15   — SL 1.5, trail 1.5 -> 0.75 after 30
Harness identical to doomhead lab: deployed tt30/0.75 base, live features,
train-only gate thr ~target/day, 1-slot cd5, DEV(<2025)/HOLDOUT split,
hedge net charged spread/ATR(hedge-open). XAU M1 + DJI M1.
"""
import sys, pickle, time, json
from pathlib import Path
import numpy as np, pandas as pd
from numba import njit
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import roc_auc_score

SRV = Path("/home/jay/Desktop/my-agents-and-website/commercial/server")
sys.path.insert(0, str(SRV))
import decision_engine.edge_pullback as ep
OUT = Path(__file__).parent
FC = pickle.load(open(SRV / "decision_engine/models/atlas_xau_validated.pkl", "rb"))["feat_cols"]
t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
COOLDOWN = 5; DEV_END = pd.Timestamp("2025-01-01"); TA, TT = 30, 0.75

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
    """mode 0: bar st+fixed_k if still open (fnow<0 filter applied by caller).
       mode 1: FIRST bar with close-fav <= -uw_thr*ATR, before exit."""
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
        snap[k] = sb; fnow[k] = f_now / a; mfs[k] = mf / a
        mins[k] = mn / a; uwf[k] = uw / max(cnt, 1)
        slp[k] = (f_now - d * (C[pb] - epr)) / a
    return snap, fnow, mfs, mins, uwf, slp

@njit(cache=True)
def take(order_idx, ebar, xit, cd):
    busy = -1; m = len(order_idx); out = np.empty(m, np.int64); c = 0
    for t in range(m):
        k = order_idx[t]
        if ebar[k] <= busy: continue
        out[c] = k; busy = xit[k] + cd; c += 1
    return out[:c]

@njit(cache=True)
def sim_hedge(sb_arr, dirs, O, H, L, C, atr, n, SL, TRAIL, MAXH, ta, tt):
    """Hedge = -dir, opens O[sb+1], own SL/trail/tt. Returns R in hedge-ATR units."""
    m = len(sb_arr); pnl = np.zeros(m); ha = np.zeros(m); valid = np.zeros(m, np.int8)
    for k in range(m):
        sbb = sb_arr[k]
        if sbb < 0: continue
        d = -dirs[k]; a = atr[sbb]; st = sbb + 1
        if st >= n or not (a > 0): continue
        epr = O[st]; hard = SL * a; mf = 0.0
        end = min(st + MAXH, n - 1); done = False
        for jx in range(st, end + 1):
            adv = (epr - L[jx]) if d == 1 else (H[jx] - epr)
            if adv >= hard: pnl[k] = -SL; done = True; break
            fav = d * (C[jx] - epr)
            if fav > mf: mf = fav
            trd = TRAIL * a
            if ta > 0 and (jx - st) >= ta:
                w = tt * a
                if w < trd: trd = w
            if mf >= trd and (mf - fav) >= trd: pnl[k] = (mf - trd) / a; done = True; break
        if not done: pnl[k] = d * (C[end] - epr) / a
        ha[k] = a; valid[k] = 1
    return pnl, ha, valid

@njit(cache=True)
def sim_hedge_tie(sb_arr, xit, dirs, O, C, atr, n):
    """Hedge closes at the ORIGINAL trade's exit close (== early-cut economics)."""
    m = len(sb_arr); pnl = np.zeros(m); ha = np.zeros(m); valid = np.zeros(m, np.int8)
    for k in range(m):
        sbb = sb_arr[k]; xb = xit[k]
        if sbb < 0 or xb <= sbb: continue
        d = -dirs[k]; a = atr[sbb]; st = sbb + 1
        if st >= n or not (a > 0): continue
        pnl[k] = d * (C[min(xb, n - 1)] - O[st]) / a
        ha[k] = a; valid[k] = 1
    return pnl, ha, valid

XGBR = dict(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.85,
            colsample_bytree=0.85, min_child_weight=10, n_jobs=-1, random_state=0)
XGBC = dict(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.85,
            colsample_bytree=0.85, min_child_weight=20, n_jobs=-1, random_state=0)
EXITS = {"tie": None, "t2": (2.0, 2.0, 300, 0, 0.0),
         "tt2": (2.0, 2.0, 300, 30, 0.75), "tt15": (1.5, 1.5, 300, 30, 0.75)}
TAU_QS = (0.5, 0.7, 0.8, 0.9, 0.95)
rng = np.random.RandomState(0)
ALL = {}

for name, pq, sp_usd, tgt in [
        ("xau_m1", "/home/jay/Desktop/new-model-zigzag/data/m1_xau_full.parquet", 0.20, 11.0),
        ("dji_m1", "/home/jay/Desktop/new-model-zigzag/data/m1_dji_full.parquet", 1.5, 11.0)]:
    log(f"=== {name} ===")
    df = pd.read_parquet(pq)
    df = df.rename(columns={[c for c in df.columns if "time" in c.lower()][0]: "time"})
    df["time"] = pd.to_datetime(df["time"]); df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    if "tick_volume" not in df.columns: df["tick_volume"] = df.get("volume", 0)
    feat = ep.compute_edge_features(df); log("features done")
    atr = feat["atr14"].to_numpy(float); cdir = feat["committed_dir"].to_numpy(np.int64)
    da = np.abs(feat["dist_at_signal"].to_numpy(float))
    O = df["open"].to_numpy(float); H = df["high"].to_numpy(float); L = df["low"].to_numpy(float); C = df["close"].to_numpy(float)
    times = df["time"].values; n = len(df)
    ok = np.isfinite(atr) & (atr > 0) & (cdir != 0); ok[:300] = False; ok[-301:] = False
    idx = np.where((da <= 1.0) & ok)[0]; dirs = cdir[idx].astype(np.int64)
    ct = pd.to_datetime(times[idx]); sig_atr = atr[idx]
    Xc = np.nan_to_num(feat[FC].to_numpy(np.float32)[idx], nan=0.0, posinf=0.0, neginf=0.0)
    FEAT_ALL = feat[FC].to_numpy(np.float32); del feat

    pnlB, ebB, xitB, codeB = sim_base(idx, dirs, O, H, L, C, atr, n, 7.0, 2.0, 300, TA, TT)
    doom = (codeB == 0).astype(np.int8)
    SNAP = {"d2r": snapshots(idx, dirs, O, H, L, C, atr, ebB, xitB, n, 1, 0, 2.0),
            "d3r": snapshots(idx, dirs, O, H, L, C, atr, ebB, xitB, n, 1, 0, 3.0),
            "d30": snapshots(idx, dirs, O, H, L, C, atr, ebB, xitB, n, 0, 30, 0.0)}
    log("base sim + snapshots done")

    # precompute hedge pnl per (snapshot, exit) once — window-independent
    HG = {}
    for snm, s in SNAP.items():
        sb = s[0]
        for enm, cfgh in EXITS.items():
            if enm == "tie":
                HG[(snm, enm)] = sim_hedge_tie(sb, xitB, dirs, O, C, atr, n)
            else:
                SLh, TRh, MHh, tah, tth = cfgh
                HG[(snm, enm)] = sim_hedge(sb, dirs, O, H, L, C, atr, n, SLh, TRh, MHh, tah, tth)
    log("hedge sims done")

    def hedge_net(sel, snm, enm):
        p_, a_, v_ = HG[(snm, enm)]
        s = sel[v_[sel] == 1]
        if len(s) == 0: return 0.0, 0, 0.0
        r = p_[s] - sp_usd / a_[s]
        return float(r.sum()), int(len(s)), float((r > 0).mean())

    # ML snapshot design matrices
    def snap_X(snm):
        sb, fnow, mfs, mins, uwf, slp = SNAP[snm]
        rows = FEAT_ALL[np.clip(sb, 0, n - 1)]
        path = np.stack([fnow, mfs, mins, uwf, slp,
                         (sb - ebB).astype(np.float64)], axis=1).astype(np.float32)
        return np.nan_to_num(np.concatenate([rows, path], axis=1), nan=0.0, posinf=0.0, neginf=0.0)
    XS = {snm: snap_X(snm) for snm in SNAP}

    WINS = []; tsw = pd.Timestamp("2020-07-01"); lastd = ct.max()
    while tsw < lastd:
        WINS.append((tsw - pd.DateOffset(years=3), tsw, tsw + pd.DateOffset(months=6))); tsw += pd.DateOffset(months=6)

    VARIANTS = [(t, e) for t in ("d2r", "d3r", "d30uw", "d2r_ml", "d30_ml") for e in EXITS]
    res = {v: [] for v in VARIANTS}; base_rows = []; aucs = []
    for tr_s, te_s, te_e in WINS:
        trm = (ct >= tr_s) & (ct < te_s); tem = (ct >= te_s) & (ct < te_e)
        if trm.sum() < 4000 or tem.sum() < 20: continue
        tix = np.where(trm)[0]; tix_f = tix if len(tix) <= 150_000 else rng.choice(tix, 150_000, replace=False)
        tr_days = max((ct[tix].max() - ct[tix].min()).days * 5 / 7, 1)
        mg = XGBRegressor(**XGBR); mg.fit(Xc[tix_f], pnlB[tix_f])
        p = mg.predict(Xc).astype(np.float64)
        cand = np.quantile(p[tix], np.linspace(0.30, 0.97, 24)); thr = cand[-1]; best = 1e18
        for th in cand:
            kk = tix[p[tix] >= th]
            if len(kk) < 5: continue
            tk = take(kk[np.argsort(ebB[kk])].astype(np.int64), ebB, xitB, COOLDOWN)
            gap = abs(len(tk) / tr_days - tgt)
            if gap < best: best = gap; thr = th
        tk_tr = take(np.where(trm & (p >= thr))[0][np.argsort(ebB[np.where(trm & (p >= thr))[0]])].astype(np.int64), ebB, xitB, COOLDOWN)
        kkte = np.where(tem & (p >= thr))[0]
        tk_te = take(kkte[np.argsort(ebB[kkte])].astype(np.int64), ebB, xitB, COOLDOWN)
        netB = float((pnlB[tk_te] - sp_usd / sig_atr[tk_te]).sum())
        base_rows.append(dict(win=str(te_s.date()), n=len(tk_te), net=netB))

        # naive triggers
        for snm, tnm in [("d2r", "d2r"), ("d3r", "d3r")]:
            sb = SNAP[snm][0]
            sel = tk_te[sb[tk_te] >= 0]
            for enm in EXITS:
                s_, n_, wr = hedge_net(sel, snm, enm)
                res[(tnm, enm)].append(dict(win=str(te_s.date()), n=n_, net=s_, wr=wr))
        sb30, f30 = SNAP["d30"][0], SNAP["d30"][1]
        sel = tk_te[(sb30[tk_te] >= 0) & (f30[tk_te] < 0)]
        for enm in EXITS:
            s_, n_, wr = hedge_net(sel, "d30", enm)
            res[("d30uw", enm)].append(dict(win=str(te_s.date()), n=n_, net=s_, wr=wr))

        # ML triggers: doom head per snapshot type; tau picked on TRAIN hedge net
        for snm, tnm in [("d2r", "d2r_ml"), ("d30", "d30_ml")]:
            sb = SNAP[snm][0]; has = sb >= 0
            trs = np.where(trm & has)[0]
            if len(trs) > 120_000: trs = rng.choice(trs, 120_000, replace=False)
            if doom[trs].sum() < 50: continue
            mc = XGBClassifier(**XGBC); mc.fit(XS[snm][trs], doom[trs])
            ph = np.zeros(len(idx)); hs = np.where(has)[0]
            ph[hs] = mc.predict_proba(XS[snm][hs])[:, 1]
            tes = np.where(tem & has)[0]
            if doom[tes].any():
                aucs.append((snm, roc_auc_score(doom[trs], ph[trs]), roc_auc_score(doom[tes], ph[tes])))
            tr_sel_all = tk_tr[has[tk_tr]]
            te_sel_all = tk_te[has[tk_te]]
            for enm in EXITS:
                best_tau, best_net = 0.0, -1e18
                for q in TAU_QS:
                    if len(tr_sel_all) < 30: break
                    tau = np.quantile(ph[tr_sel_all], q)
                    s_, n_, _ = hedge_net(tr_sel_all[ph[tr_sel_all] >= tau], snm, enm)
                    if n_ >= 10 and s_ > best_net: best_net, best_tau = s_, tau
                if best_net <= -1e17:
                    res[(tnm, enm)].append(dict(win=str(te_s.date()), n=0, net=0.0, wr=0.0)); continue
                s_, n_, wr = hedge_net(te_sel_all[ph[te_sel_all] >= best_tau], snm, enm)
                res[(tnm, enm)].append(dict(win=str(te_s.date()), n=n_, net=s_, wr=wr))
        log(f"window {te_s.date()} done")

    def agg(rows, dev):
        rr = [r for r in rows if (pd.Timestamp(r["win"]) < DEV_END) == dev]
        act = [r for r in rr if r["n"] > 0]
        if not rr: return None
        return dict(nwin=len(act), n=sum(r["n"] for r in rr), net=sum(r["net"] for r in rr),
                    w=sum(1 for r in act if r["net"] > 0))
    if aucs:
        for snm in ("d2r", "d30"):
            aa = [(a, b) for s_, a, b in aucs if s_ == snm]
            if aa: print(f"  doom AUC {snm}: train {np.median([x[0] for x in aa]):.3f} -> test {np.median([x[1] for x in aa]):.3f}")
    for dev, lab in [(True, "DEV"), (False, "HOLDOUT")]:
        b = agg(base_rows, dev)
        print(f"\n{'='*70}\nHEDGE LAB {name} — {lab}   [base portfolio: n={b['n']} net {b['net']:+.0f}R]\n{'='*70}")
        print(f"{'trigger':<9}{'exit':<7}{'nhedge':>7}{'netR':>9}{'w+':>7}")
        for (tnm, enm) in VARIANTS:
            a = agg(res[(tnm, enm)], dev)
            if a is None or a["n"] == 0: continue
            print(f"{tnm:<9}{enm:<7}{a['n']:>7}{a['net']:>+9.0f}{a['w']:>4}/{a['nwin']:<3}")
    ALL[name] = {f"{t}|{e}": res[(t, e)] for (t, e) in VARIANTS}
    ALL[name]["base"] = base_rows
    del FEAT_ALL, Xc, HG, XS

json.dump(ALL, open(OUT / "hedge_results.json", "w"), default=str, indent=1)
log("hedge lab done")
