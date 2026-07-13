"""
HEDGE LAB v2 — "train the model to hedge bad trades" (Jay, 2026-07-14).
v1 (run_lab_hedge.py) used doom-CLASSIFIER triggers -> rejected. v2 trains the
strongest remaining version: an XGB regressor that directly predicts the HEDGE
TRADE'S OWN net R (opposite dir, own SL2/trail2/tt30/0.75 exits) from mid-trade
state, and hedges only when predicted hedge R >= tau (tau train-calibrated,
grid includes 0). Two snapshot triggers per original trade (first -2R bar;
bar-30-underwater). Base portfolio = the NEW q10 gate (deployed 2026-07-13).
Hedge stream is additive overlay P&L -> deployable iff net-positive standalone.
XAU M1 (spread $0.20/ATR) + DJI M1 (1.5pt AND 2.0pt). DEV(<2025)/HOLDOUT.
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
COOLDOWN = 5; DEV_END = pd.Timestamp("2025-01-01"); TA, TT = 30, 0.75

@njit(cache=True)
def sim_tt(idxs, dirs, O, H, L, C, atr, n, SL, TRAIL, MAXH, ta, tt):
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

XGBQ = dict(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.85,
            colsample_bytree=0.85, min_child_weight=10, n_jobs=-1, random_state=0,
            objective="reg:quantileerror", quantile_alpha=0.10)   # base gate (deployed)
XGBH = dict(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.85,
            colsample_bytree=0.85, min_child_weight=20, n_jobs=-1, random_state=0)  # hedge head
TAUS = (0.0, 0.1, 0.25, 0.5)
rng = np.random.RandomState(0)
ALL = {}

for name, pq, spreads in [
        ("xau_m1", "/home/jay/Desktop/new-model-zigzag/data/m1_xau_full.parquet", [0.20]),
        ("dji_m1", "/home/jay/Desktop/new-model-zigzag/data/m1_dji_full.parquet", [1.5, 2.0])]:
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
    FEAT_ALL = np.nan_to_num(feat[FC].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    del feat
    pnlB, ebB, xitB, codeB = sim_tt(idx, dirs, O, H, L, C, atr, n, 7.0, 2.0, 300, TA, TT)
    SNAP = {"d2r": snapshots(idx, dirs, O, H, L, C, atr, ebB, xitB, n, 1, 0, 2.0),
            "d30": snapshots(idx, dirs, O, H, L, C, atr, ebB, xitB, n, 0, 30, 0.0)}
    HED = {snm: sim_hedge(SNAP[snm][0], dirs, O, H, L, C, atr, n, 2.0, 2.0, 300, 30, 0.75)
           for snm in SNAP}
    log("base + hedge sims done")

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
    VAR = [(snm, sp) for snm in SNAP for sp in spreads]
    res = {v: [] for v in VAR}; base_rows = []
    for tr_s, te_s, te_e in WINS:
        trm = (ct >= tr_s) & (ct < te_s); tem = (ct >= te_s) & (ct < te_e)
        if trm.sum() < 4000 or tem.sum() < 20: continue
        tix = np.where(trm)[0]; tix_f = tix if len(tix) <= 150_000 else rng.choice(tix, 150_000, replace=False)
        tr_days = max((ct[tix].max() - ct[tix].min()).days * 5 / 7, 1)
        mg = XGBRegressor(**XGBQ); mg.fit(Xc[tix_f], pnlB[tix_f])   # deployed q10 gate
        p = mg.predict(Xc).astype(np.float64)
        cand = np.quantile(p[tix], np.linspace(0.30, 0.97, 24)); thr = cand[-1]; best = 1e18
        for th in cand:
            kk = tix[p[tix] >= th]
            if len(kk) < 5: continue
            tk = take(kk[np.argsort(ebB[kk])].astype(np.int64), ebB, xitB, COOLDOWN)
            gap = abs(len(tk) / tr_days - 11.0)
            if gap < best: best = gap; thr = th
        tkk = np.where(trm & (p >= thr))[0]
        tk_tr = take(tkk[np.argsort(ebB[tkk])].astype(np.int64), ebB, xitB, COOLDOWN)
        kkte = np.where(tem & (p >= thr))[0]
        tk_te = take(kkte[np.argsort(ebB[kkte])].astype(np.int64), ebB, xitB, COOLDOWN)
        base_rows.append(dict(win=str(te_s.date()), n=len(tk_te),
                              net=float((pnlB[tk_te] - spreads[0] / sig_atr[tk_te]).sum())))

        for snm in SNAP:
            sb = SNAP[snm][0]; hp, ha, hv = HED[snm]
            has = (sb >= 0) & (hv == 1)
            for sp in spreads:
                hnet = hp - sp / np.maximum(ha, 1e-9)   # hedge net R (own ATR)
                trs = tk_tr[has[tk_tr]]
                if len(trs) < 200:
                    res[(snm, sp)].append(dict(win=str(te_s.date()), n=0, net=0.0)); continue
                mh = XGBRegressor(**XGBH); mh.fit(XS[snm][trs], hnet[trs])   # TRAIN: predict hedge R
                ph = np.zeros(len(idx))
                tes = tk_te[has[tk_te]]
                allsnap = np.concatenate([trs, tes])
                ph[allsnap] = mh.predict(XS[snm][allsnap]).astype(np.float64)
                best_tau, best_net = None, -1e18
                for tau in TAUS:
                    s = trs[ph[trs] >= tau]
                    if len(s) < 10: continue
                    nt = float(hnet[s].sum())
                    if nt > best_net: best_net, best_tau = nt, tau
                if best_tau is None or best_net <= 0:
                    # train says hedging loses -> hedge nothing this window
                    res[(snm, sp)].append(dict(win=str(te_s.date()), n=0, net=0.0)); continue
                s = tes[ph[tes] >= best_tau]
                res[(snm, sp)].append(dict(win=str(te_s.date()), n=int(len(s)),
                                           net=float(hnet[s].sum()),
                                           wr=float((hnet[s] > 0).mean()) if len(s) else 0.0))
        log(f"window {te_s.date()} done")

    def agg(rows, dev):
        rr = [r for r in rows if (pd.Timestamp(r["win"]) < DEV_END) == dev]
        act = [r for r in rr if r["n"] > 0]
        if not rr: return None
        return dict(nwin=len(rr), nact=len(act), n=sum(r["n"] for r in rr),
                    net=sum(r["net"] for r in rr), w=sum(1 for r in act if r["net"] > 0))
    for dev, lab in [(True, "DEV"), (False, "HOLDOUT")]:
        b = agg(base_rows, dev)
        print(f"\n{'='*70}\nHEDGE v2 (learned hedge head) {name} — {lab}  [base q10: n={b['n']} net {b['net']:+.0f}R]\n{'='*70}")
        print(f"{'trigger':<9}{'spread':<8}{'nhedge':>7}{'netR':>9}{'w+/active':>11}")
        for (snm, sp) in VAR:
            a = agg(res[(snm, sp)], dev)
            if a is None: continue
            print(f"{snm:<9}{sp:<8}{a['n']:>7}{a['net']:>+9.0f}{a['w']:>6}/{a['nact']:<4}")
    ALL[name] = {f"{s}|{sp}": res[(s, sp)] for (s, sp) in VAR}
    ALL[name]["base"] = base_rows
    del FEAT_ALL, Xc, XS

json.dump(ALL, open(OUT / "hedge_v2_results.json", "w"), default=str, indent=1)
log("hedge v2 lab done")
