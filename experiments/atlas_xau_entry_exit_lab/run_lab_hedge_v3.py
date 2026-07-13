"""
HEDGE LAB v3 — Jay's exact dynamic-hedge spec (2026-07-14), DJI only (XAU
hedging failed 5x, closed):
  While a main trade is open, EVERY completed bar the learned hedge head
  predicts the reverse trade's own net R. Hedge ON when pred >= tau_on and no
  hedge open; hedge OFF when pred <= tau_off ("danger passed") — exit modes:
    mkt — close hedge at bar close (pay whatever it costs)
    be  — after signal-off, wait for price to touch the hedge entry -> close at
          breakeven (Jay's version); forced market close when the main exits
  Re-hedge unlimited (spread charged per episode). Main trade untouched.
  Head trained on d30+d2r snapshot union (labels = hedge tt-exit net R), same
  walk-forward as v2; tau_on/tau_off grid calibrated on TRAIN hedge net.
Base = deployed q10 gate + tt30/0.75. DEV(<2025)/HOLDOUT, spreads 1.5/2.0pt.
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
MIN_HELD = 5   # don't evaluate hedge before this many bars held

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

@njit(cache=True)
def perbar_stats(tk, ebar, xit, idxs, dirs, O, C, atr, min_held):
    """Flat per-bar rows for taken trades: (trade_k, bar, fnow, mfs, mins, uwf, slp, held)."""
    total = 0
    for q in range(len(tk)):
        k = tk[q]
        lo = ebar[k] + min_held; hi = xit[k] - 1
        if hi >= lo: total += hi - lo + 1
    TK = np.empty(total, np.int64); BR = np.empty(total, np.int64)
    ST = np.empty((total, 6), np.float64)
    c = 0
    for q in range(len(tk)):
        k = tk[q]
        st = ebar[k]; xb = xit[k]; i = idxs[k]; d = dirs[k]; a = atr[i]; epr = O[st]
        mf = -1e18; mn = 1e18; uw = 0; cnt = 0
        for jx in range(st, xb):
            fv = d * (C[jx] - epr)
            if fv > mf: mf = fv
            if fv < mn: mn = fv
            if fv < 0: uw += 1
            cnt += 1
            if jx - st >= min_held and jx <= xb - 1:
                pb = st if jx - 10 < st else jx - 10
                TK[c] = k; BR[c] = jx
                ST[c, 0] = fv / a; ST[c, 1] = mf / a; ST[c, 2] = mn / a
                ST[c, 3] = uw / cnt; ST[c, 4] = (fv - d * (C[pb] - epr)) / a
                ST[c, 5] = jx - st
                c += 1
    return TK[:c], BR[:c], ST[:c]

@njit(cache=True)
def hedge_machine(tk, row_start, row_end, BR, PH, ebar, xit, idxs, dirs,
                  O, H, L, C, atr, n, tau_on, tau_off, be_mode, sp_pts):
    """Jay's state machine per taken trade. Returns total hedge net R (own-ATR
    units) and episode count. Hedge opens next bar open after signal; closes:
      be_mode 0: market at close of first bar with PH<=tau_off (or main exit)
      be_mode 1: after PH<=tau_off, at touch of hedge entry (BE) else main exit
    """
    tot = 0.0; eps = 0
    for q in range(len(tk)):
        k = tk[q]; xb = xit[k]; d = dirs[k]
        open_h = False; epr = 0.0; a = 1.0; armed_be = False
        r0 = row_start[q]; r1 = row_end[q]
        for r in range(r0, r1):
            jx = BR[r]
            if jx >= xb: break
            if not open_h:
                if PH[r] >= tau_on and jx + 1 < xb:   # open at next bar's open
                    a = atr[jx]
                    if a > 0:
                        epr = O[jx + 1]; open_h = True; armed_be = False
                        eps += 1
                        tot -= sp_pts / a
            else:
                hd = -d
                if be_mode == 1 and armed_be:
                    # BE exit: touch of hedge entry price
                    if (L[jx] <= epr) and (H[jx] >= epr):
                        open_h = False; armed_be = False
                        continue
                if PH[r] <= tau_off:
                    if be_mode == 0:
                        tot += hd * (C[jx] - epr) / a
                        open_h = False
                    else:
                        armed_be = True
                        if (L[jx] <= epr) and (H[jx] >= epr):
                            open_h = False; armed_be = False
        if open_h:
            e = xb if xb < n else n - 1
            tot += (-d) * (C[e] - epr) / a
    return tot, eps

XGBQ = dict(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.85,
            colsample_bytree=0.85, min_child_weight=10, n_jobs=-1, random_state=0,
            objective="reg:quantileerror", quantile_alpha=0.10)
XGBH = dict(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.85,
            colsample_bytree=0.85, min_child_weight=20, n_jobs=-1, random_state=0)
GRID = [(ton, toff) for ton in (0.1, 0.25, 0.5) for toff in (-0.25, 0.0)]
rng = np.random.RandomState(0)

log("=== dji_m1 (v3 dynamic hedge) ===")
df = pd.read_parquet("/home/jay/Desktop/new-model-zigzag/data/m1_dji_full.parquet")
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
SN2 = snapshots(idx, dirs, O, H, L, C, atr, ebB, xitB, n, 1, 0, 2.0)
SN30 = snapshots(idx, dirs, O, H, L, C, atr, ebB, xitB, n, 0, 30, 0.0)
H2 = sim_hedge(SN2[0], dirs, O, H, L, C, atr, n, 2.0, 2.0, 300, 30, 0.75)
H30 = sim_hedge(SN30[0], dirs, O, H, L, C, atr, n, 2.0, 2.0, 300, 30, 0.75)
log("sims done")

def snap_X(SN):
    sb, fnow, mfs, mins, uwf, slp = SN
    rows = FEAT_ALL[np.clip(sb, 0, n - 1)]
    path = np.stack([fnow, mfs, mins, uwf, slp, (sb - ebB).astype(np.float64)], axis=1).astype(np.float32)
    return np.nan_to_num(np.concatenate([rows, path], axis=1), nan=0.0, posinf=0.0, neginf=0.0)
XS2, XS30 = snap_X(SN2), snap_X(SN30)

WINS = []; tsw = pd.Timestamp("2020-07-01"); lastd = ct.max()
while tsw < lastd:
    WINS.append((tsw - pd.DateOffset(years=3), tsw, tsw + pd.DateOffset(months=6))); tsw += pd.DateOffset(months=6)
SPREADS = (1.5, 2.0)
res = {(bm, sp): [] for bm in (0, 1) for sp in SPREADS}; base_rows = []
for tr_s, te_s, te_e in WINS:
    trm = (ct >= tr_s) & (ct < te_s); tem = (ct >= te_s) & (ct < te_e)
    if trm.sum() < 4000 or tem.sum() < 20: continue
    tix = np.where(trm)[0]; tix_f = tix if len(tix) <= 150_000 else rng.choice(tix, 150_000, replace=False)
    tr_days = max((ct[tix].max() - ct[tix].min()).days * 5 / 7, 1)
    mg = XGBRegressor(**XGBQ); mg.fit(Xc[tix_f], pnlB[tix_f])
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
                          net=float((pnlB[tk_te] - SPREADS[0] / sig_atr[tk_te]).sum())))

    # hedge head: train on union of d2r + d30 snapshots of TRAIN taken trades
    Xtr, ytr = [], []
    for SN, HH, XSs in ((SN2, H2, XS2), (SN30, H30, XS30)):
        sb = SN[0]; hp, ha, hv = HH
        s = tk_tr[(sb[tk_tr] >= 0) & (hv[tk_tr] == 1)]
        if len(s):
            Xtr.append(XSs[s]); ytr.append(hp[s] - 2.0 / np.maximum(ha[s], 1e-9))
    if not Xtr:
        continue
    mh = XGBRegressor(**XGBH); mh.fit(np.concatenate(Xtr), np.concatenate(ytr))

    def perbar_ph(tk_set):
        TK, BR, ST = perbar_stats(tk_set.astype(np.int64), ebB, xitB, idx, dirs, O, C, atr, MIN_HELD)
        if len(TK) == 0: return TK, BR, np.zeros(0), np.zeros(0, np.int64), np.zeros(0, np.int64)
        Xpb = np.concatenate([FEAT_ALL[BR], ST.astype(np.float32)], axis=1)
        PH = mh.predict(np.nan_to_num(Xpb, nan=0.0, posinf=0.0, neginf=0.0)).astype(np.float64)
        rs = np.zeros(len(tk_set), np.int64); re = np.zeros(len(tk_set), np.int64)
        pos = 0
        for q, k in enumerate(tk_set):
            rs[q] = pos
            while pos < len(TK) and TK[pos] == k: pos += 1
            re[q] = pos
        return TK, BR, PH, rs, re
    TKr, BRr, PHr, rs_tr, re_tr = perbar_ph(tk_tr)
    TKe, BRe, PHe, rs_te, re_te = perbar_ph(tk_te)

    for bm in (0, 1):
        for sp in SPREADS:
            best_g, best_net = None, -1e18
            for ton, toff in GRID:
                nt, _ = hedge_machine(tk_tr.astype(np.int64), rs_tr, re_tr, BRr, PHr, ebB, xitB,
                                      idx, dirs, O, H, L, C, atr, n, ton, toff, bm, sp)
                if nt > best_net: best_net, best_g = nt, (ton, toff)
            if best_net <= 0:
                res[(bm, sp)].append(dict(win=str(te_s.date()), n=0, net=0.0)); continue
            nt, ne = hedge_machine(tk_te.astype(np.int64), rs_te, re_te, BRe, PHe, ebB, xitB,
                                   idx, dirs, O, H, L, C, atr, n, best_g[0], best_g[1], bm, sp)
            res[(bm, sp)].append(dict(win=str(te_s.date()), n=int(ne), net=float(nt)))
    log(f"window {te_s.date()} done")

def agg(rows, dev):
    rr = [r for r in rows if (pd.Timestamp(r["win"]) < DEV_END) == dev]
    act = [r for r in rr if r["n"] > 0]
    if not rr: return None
    return dict(nwin=len(rr), nact=len(act), n=sum(r["n"] for r in rr),
                net=sum(r["net"] for r in rr), w=sum(1 for r in act if r["net"] > 0))
for dev, lab in [(True, "DEV"), (False, "HOLDOUT")]:
    b = agg(base_rows, dev)
    print(f"\n{'='*70}\nHEDGE v3 dynamic (DJI) — {lab}  [base q10: n={b['n']} net {b['net']:+.0f}R]\n{'='*70}")
    print(f"{'exit':<6}{'spread':<8}{'nepisodes':>10}{'netR':>9}{'w+/active':>11}")
    for bm, bn in ((0, "mkt"), (1, "be")):
        for sp in SPREADS:
            a = agg(res[(bm, sp)], dev)
            if a is None: continue
            print(f"{bn:<6}{sp:<8}{a['n']:>10}{a['net']:>+9.0f}{a['w']:>6}/{a['nact']:<4}")
json.dump({f"{bm}|{sp}": res[(bm, sp)] for (bm, sp) in res}, open(OUT / "hedge_v3_results.json", "w"), default=str, indent=1)
log("hedge v3 lab done")
