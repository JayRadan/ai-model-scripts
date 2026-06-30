"""
Hermes DJI EDGE CAMPAIGN — autonomous overnight search for a robust walk-forward
edge at 10-20 trades/day. Tries many ENTRY bases x LABEL schemes x MODELS, sweeps
the selection threshold to hit the target frequency, judges on 8y walk-forward.

Design:
 - causal features computed once
 - each candidate gets (exit_bar, R) precomputed by its label sim -> any label
   plugs into ONE exit-agnostic 1-slot portfolio sim (sequential, cooldown 5)
 - regression (predict R) OR classification (predict P(win)) per config
 - threshold chosen to land trades/day in [10,20]; judged by robust profit
 - results appended to campaign_results.csv incrementally + live leaderboard

Run: nohup python3 experiments/hermes_dji_campaign.py &  (runs for hours)
"""
import sys, pickle, time, csv, traceback
from pathlib import Path
import numpy as np, pandas as pd
from numba import njit
from xgboost import XGBRegressor, XGBClassifier

DE = Path("/home/jay/Desktop/my-agents-and-website/commercial/server/decision_engine")
sys.path.insert(0, str(DE))
import hermes_features as hf
from configs.hermes_dji import HERMES_DJI as CFG
FC = pickle.load(open(DE / "models/hermes_dji_validated.pkl", "rb"))["feat_cols"]
OUT = Path("/home/jay/Desktop/new-model-zigzag/experiments/_hermes_retrain"); OUT.mkdir(exist_ok=True)
RESCSV = OUT / "campaign_results.csv"
t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

# ── data + causal features (once) ────────────────────────────────────────────
df = pd.read_parquet("/home/jay/Desktop/new-model-zigzag/data/m1_dji_full.parquet")
df = df.rename(columns={[c for c in df.columns if "time" in c.lower()][0]: "time"})
df["time"] = pd.to_datetime(df["time"]); df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
if "tick_volume" not in df.columns: df["tick_volume"] = df.get("volume", 0)
feat = hf.compute_all_features(df.copy(), CFG)
_A = 2.0 / 51.0; s = df.set_index("time")["close"]
for nm, tfm in [("m5", 5), ("m15", 15), ("h1", 60)]:
    g = s.resample(f"{tfm}min").last().dropna(); hc = g.to_numpy(np.float64)
    et = (g.index + pd.Timedelta(minutes=tfm)).values
    N = len(df); rsi = np.full(N, np.nan); slope = np.full(N, np.nan); emad = np.full(N, np.nan)
    ema = np.empty(len(hc)); e = hc[0]
    for i in range(len(hc)):
        e = hc[0] if i == 0 else e * (1 - _A) + hc[i] * _A; ema[i] = e
    dl = np.diff(hc, prepend=hc[0]); cg = np.cumsum(np.clip(dl, 0, None)); cl = np.cumsum(np.clip(-dl, 0, None))
    j = np.searchsorted(et, df["time"].values, side="right") - 1; ok = j >= 14; jj = j[ok]
    cc = df["close"].to_numpy(np.float64)[ok]
    slope[ok] = cc - hc[jj - 4]; emad[ok] = cc - (ema[jj] * (1 - _A) + cc * _A)
    dc = cc - hc[jj]; gs = (cg[jj] - cg[jj - 13]) + np.clip(dc, 0, None); ls = (cl[jj] - cl[jj - 13]) + np.clip(-dc, 0, None)
    rs = (gs / 14.0) / np.where(ls == 0, np.nan, ls / 14.0); rsi[ok] = 100 - 100 / (1 + rs)
    feat[f"{nm}_rsi14"] = rsi; feat[f"{nm}_slope5"] = slope; feat[f"{nm}_ema50_dist"] = emad
log(f"features done; {len(df):,} bars")

atr = feat["atr14"].to_numpy(float); cdir = feat["committed_dir"].to_numpy(np.int64)
ds = feat["dist_at_signal"].to_numpy(float); da = np.abs(ds)
rsi14 = feat["rsi14"].to_numpy(float); demaA = feat["dist_ema20"].to_numpy(float)
O = df["open"].to_numpy(float); H = df["high"].to_numpy(float); L = df["low"].to_numpy(float); C = df["close"].to_numpy(float)
times = df["time"].values; n = len(df); sp = CFG.spread_usd / np.nanmedian(atr)
ct_all = pd.to_datetime(times)
hh = pd.DatetimeIndex(times).hour.values
roll_hi = pd.Series(H).rolling(20).max().to_numpy(); roll_lo = pd.Series(L).rolling(20).min().to_numpy()
base_ok = np.isfinite(atr) & (atr > 0) & (cdir != 0); base_ok[:300] = False; base_ok[-301:] = False

# ── ENTRY BASES: dict name -> (cand_idx array, direction array) ──────────────
def mk(mask, direction):
    idx = np.where(mask & base_ok)[0]; return idx, direction[idx].astype(np.int64)
BASES = {}
BASES["pb0.5|ct1.5"] = mk((da <= 0.5) | ((ds * cdir) <= -1.5), cdir)        # shipped
BASES["pb1.0"]       = mk(da <= 1.0, cdir)
BASES["pb1.5"]       = mk(da <= 1.5, cdir)
BASES["in_regime"]   = mk(np.ones(n, bool), cdir)                            # very dense
BASES["rsi_revert"]  = mk(((rsi14 < 38) & (cdir == 1)) | ((rsi14 > 62) & (cdir == -1)), cdir)
BASES["near_ema20"]  = mk(np.abs(demaA) <= 0.5, cdir)
BASES["counter1.0"]  = mk((ds * cdir) <= -1.0, cdir)
BASES["breakout"]    = mk(((C > roll_hi) & (cdir == 1)) | ((C < roll_lo) & (cdir == -1)), cdir)
for k, (ix, _) in BASES.items():
    log(f"  base {k:<14} {len(ix):>7,} candidates  ({len(ix)/1500:.1f}/day raw)")

# ── label sims: return (pnl_R, exit_idx) per candidate ───────────────────────
@njit
def sim_trail(idxs, dirs, O, H, L, C, atr, sp, SL, TRAIL, MAXH, n):
    m = len(idxs); pnl = np.empty(m); xit = np.empty(m, np.int64)
    for k in range(m):
        i = idxs[k]; d = dirs[k]; a = atr[i]; ei = i + 1
        if ei >= n or not (a > 0): pnl[k] = 0.0; xit[k] = min(i + 1, n - 1); continue
        ep = O[ei]; hard = SL * a; trd = TRAIL * a; mf = 0.0; end = min(ei + MAXH, n - 1); done = False
        for jx in range(ei, end + 1):
            fav = d * (C[jx] - ep)
            if fav > mf: mf = fav
            if d == 1 and (ep - L[jx]) >= hard: pnl[k] = -SL - sp; xit[k] = jx; done = True; break
            if d == -1 and (H[jx] - ep) >= hard: pnl[k] = -SL - sp; xit[k] = jx; done = True; break
            if mf >= trd and (mf - fav) >= trd: pnl[k] = (mf - trd) / a - sp; xit[k] = jx; done = True; break
        if not done: pnl[k] = d * (C[end] - ep) / a - sp; xit[k] = end
    return pnl, xit

@njit
def sim_barrier(idxs, dirs, O, H, L, C, atr, sp, TP, SL, MAXH, n):
    m = len(idxs); pnl = np.empty(m); xit = np.empty(m, np.int64)
    for k in range(m):
        i = idxs[k]; d = dirs[k]; a = atr[i]; ei = i + 1
        if ei >= n or not (a > 0): pnl[k] = 0.0; xit[k] = min(i + 1, n - 1); continue
        ep = O[ei]; up = TP * a; dn = SL * a; end = min(ei + MAXH, n - 1); done = False
        for jx in range(ei, end + 1):
            fav_hi = d * (H[jx] - ep); fav_lo = d * (L[jx] - ep)
            # check SL first (conservative): adverse excursion
            if d == 1:
                if (ep - L[jx]) >= dn: pnl[k] = -SL - sp; xit[k] = jx; done = True; break
                if (H[jx] - ep) >= up: pnl[k] = TP - sp; xit[k] = jx; done = True; break
            else:
                if (H[jx] - ep) >= dn: pnl[k] = -SL - sp; xit[k] = jx; done = True; break
                if (ep - L[jx]) >= up: pnl[k] = TP - sp; xit[k] = jx; done = True; break
        if not done: pnl[k] = d * (C[end] - ep) / a - sp; xit[k] = end
    return pnl, xit

@njit
def portfolio_1slot(order_idx, entry_bar, exit_bar, pnl, cooldown):
    # order_idx: indices (into the candidate arrays) sorted by entry_bar ascending
    busy_until = -1; m = len(order_idx); out = np.zeros(m); cnt = 0
    for t in range(m):
        k = order_idx[t]; eb = entry_bar[k]
        if eb <= busy_until: continue
        out[cnt] = pnl[k]; busy_until = exit_bar[k] + cooldown; cnt += 1
    return out[:cnt]

def wf_eval(idx, dirs, pnl, xit, score, thr, train_mask_fn, wins):
    """For threshold thr on score, run 1-slot portfolio per window; return per-window list."""
    entry_bar = idx + 1
    res = []
    for (tr_s, te_s, te_e, sel_pred) in wins:
        sel = sel_pred >= thr
        wm = (ct_cand >= te_s) & (ct_cand < te_e) & sel
        kk = np.where(wm)[0]
        if len(kk) < 5: res.append((0, 0.0, 0.0)); continue
        order = kk[np.argsort(entry_bar[kk])]
        r = portfolio_1slot(order.astype(np.int64), entry_bar, xit, pnl, 5)
        if len(r) == 0: res.append((0, 0.0, 0.0)); continue
        pf = r[r > 0].sum() / max(-r[r <= 0].sum(), 1e-9)
        res.append((len(r), float(r.sum()), float(pf)))
    return res

# ── walk-forward windows ─────────────────────────────────────────────────────
WINS = []
first = pd.Timestamp("2020-07-01"); tsw = first; lastd = ct_all.max()
while tsw < lastd:
    WINS.append((tsw - pd.DateOffset(years=3), tsw, tsw + pd.DateOffset(months=6))); tsw += pd.DateOffset(months=6)
test_days = len(pd.bdate_range(first, lastd))  # approx trading days across all test windows

# ── config grid ──────────────────────────────────────────────────────────────
LABELS = {
    "trailSL6T2":  ("trail", dict(SL=6.0, TRAIL=2.0, MAXH=300)),
    "trailSL4T2":  ("trail", dict(SL=4.0, TRAIL=2.0, MAXH=300)),
    "barT2S1H120": ("barrier", dict(TP=2.0, SL=1.0, MAXH=120)),
    "barT3S2H240": ("barrier", dict(TP=3.0, SL=2.0, MAXH=240)),
    "barT1.5S1H60":("barrier", dict(TP=1.5, SL=1.0, MAXH=60)),
}
# curated (base,label,model) combos — dense bases get barrier+classifier emphasis
COMBOS = []
for b in BASES:
    COMBOS.append((b, "trailSL6T2", "reg"))
for b in ["in_regime", "pb1.5", "pb1.0", "rsi_revert", "breakout", "near_ema20", "counter1.0"]:
    COMBOS.append((b, "barT2S1H120", "clf"))
    COMBOS.append((b, "barT3S2H240", "clf"))
for b in ["in_regime", "pb1.5", "rsi_revert", "breakout"]:
    COMBOS.append((b, "barT1.5S1H60", "clf"))
    COMBOS.append((b, "trailSL4T2", "reg"))
log(f"{len(COMBOS)} configs queued; {len(WINS)} WF windows; ~{test_days} test bdays")

# CSV header
if not RESCSV.exists():
    with open(RESCSV, "w", newline="") as f:
        csv.writer(f).writerow(["base","label","model","thr","trades","per_day","profWk","nWk","totSumR","medPF","worstWk","meets"])

leaderboard = []
def label_arrays(base, label):
    idx, dirs = BASES[base]; kind, p = LABELS[label]
    if kind == "trail":
        pnl, xit = sim_trail(idx, dirs, O, H, L, C, atr, float(sp), p["SL"], p["TRAIL"], p["MAXH"], n)
    else:
        pnl, xit = sim_barrier(idx, dirs, O, H, L, C, atr, float(sp), p["TP"], p["SL"], p["MAXH"], n)
    return idx, dirs, pnl, xit

# ── run campaign ─────────────────────────────────────────────────────────────
for ci, (base, label, model) in enumerate(COMBOS):
    try:
        idx, dirs, pnl, xit = label_arrays(base, label)
        global ct_cand; ct_cand = pd.to_datetime(times[idx])
        Xc = feat[FC].to_numpy(np.float32)[idx]
        ybin = (pnl > 0).astype(int)
        # walk-forward: train per window, store predictions
        preds_per_win = []
        TRAIN_CAP = 150000
        rng = np.random.RandomState(0)
        for tr_s, te_s, te_e in WINS:
            trm = (ct_cand >= tr_s) & (ct_cand < te_s); tem = (ct_cand >= te_s) & (ct_cand < te_e)
            if trm.sum() < 4000 or tem.sum() < 20:
                preds_per_win.append((tr_s, te_s, te_e, np.full(len(idx), -9e9))); continue
            tr_ix = np.where(trm)[0]
            if len(tr_ix) > TRAIN_CAP:               # cap so dense bases stay fast
                tr_ix = rng.choice(tr_ix, TRAIN_CAP, replace=False)
            if model == "reg":
                mdl = XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.85,
                                   colsample_bytree=0.85, min_child_weight=10, n_jobs=-1, random_state=0)
                mdl.fit(Xc[tr_ix], pnl[tr_ix])
            else:
                pos = ybin[tr_ix].mean(); spw = (1 - pos) / max(pos, 1e-3)
                mdl = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.85,
                                    colsample_bytree=0.85, min_child_weight=10, n_jobs=-1, random_state=0,
                                    scale_pos_weight=spw, eval_metric="logloss")
                mdl.fit(Xc[tr_ix], ybin[tr_ix])
            full = np.full(len(idx), -9e9)
            if model == "reg": full[tem] = mdl.predict(Xc[tem])
            else: full[tem] = mdl.predict_proba(Xc[tem])[:, 1]
            preds_per_win.append((tr_s, te_s, te_e, full))
        # threshold sweep -> find one giving ~10-20 trades/day with best robust profit
        if model == "reg":
            grid = np.quantile(np.concatenate([p[3][p[3] > -9e8] for p in preds_per_win]) if any((p[3] > -9e8).any() for p in preds_per_win) else np.array([0.0]), np.linspace(0.3, 0.97, 14))
        else:
            grid = np.linspace(0.40, 0.80, 14)
        entry_bar = idx + 1
        best = None
        for thr in grid:
            per = []
            for (tr_s, te_s, te_e, full) in preds_per_win:
                sel = full >= thr
                wm = (ct_cand >= te_s) & (ct_cand < te_e) & sel; kk = np.where(wm)[0]
                if len(kk) < 3: per.append((0, 0.0, 0.0)); continue
                order = kk[np.argsort(entry_bar[kk])]
                r = portfolio_1slot(order.astype(np.int64), entry_bar, xit, pnl, 5)
                if len(r) == 0: per.append((0, 0.0, 0.0)); continue
                pf = r[r > 0].sum() / max(-r[r <= 0].sum(), 1e-9)
                per.append((len(r), float(r.sum()), float(pf)))
            tottrd = sum(p[0] for p in per); perday = tottrd / max(test_days, 1)
            prof = sum(1 for p in per if p[1] > 0); nw = len(per)
            tot = sum(p[1] for p in per); pfs = [p[2] for p in per if p[0] > 0]
            medpf = float(np.median(pfs)) if pfs else 0; worst = min((p[1] for p in per), default=0)
            meets = 10 <= perday <= 20
            rowkey = (base, label, model, round(float(thr), 4), tottrd, round(perday, 2), prof, nw, round(tot, 1), round(medpf, 3), round(worst, 1), int(meets))
            with open(RESCSV, "a", newline="") as f:
                csv.writer(f).writerow(rowkey)
            # score: prefer meets-frequency, then profitable-window-fraction, then total
            score = (1 if meets else 0, prof / max(nw, 1), tot)
            if best is None or score > best[0]:
                best = (score, rowkey, perday, prof, nw, tot, medpf, worst)
        if best:
            _, rk, pd_, prof, nw, tot, medpf, worst = best
            tag = "★MEETS" if (10 <= pd_ <= 20 and prof/max(nw,1) >= 0.6 and tot > 0) else ""
            log(f"[{ci+1}/{len(COMBOS)}] {base:<12} {label:<11} {model}  best thr={rk[3]:<7} "
                f"{pd_:>5.1f}/day  prof {prof}/{nw}  sumR {tot:>+7.1f}  medPF {medpf:.2f} {tag}")
            leaderboard.append((base, label, model, pd_, prof, nw, tot, medpf, worst))
    except Exception as ex:
        log(f"[{ci+1}/{len(COMBOS)}] {base}/{label}/{model} ERROR: {ex}")
        traceback.print_exc()

# ── leaderboard ──────────────────────────────────────────────────────────────
log("="*90); log("CAMPAIGN 1 LEADERBOARD (configs hitting 10-20/day, ranked by robust profit)")
lead = [r for r in leaderboard if 10 <= r[3] <= 20]
lead.sort(key=lambda r: (r[4] / max(r[5], 1), r[6]), reverse=True)
if not lead:
    log("  NONE hit 10-20/day with the swept thresholds — see CSV for what each base can do.")
    leaderboard.sort(key=lambda r: (r[4]/max(r[5],1), r[6]), reverse=True)
    for r in leaderboard[:8]:
        log(f"  {r[0]:<12} {r[1]:<11} {r[2]}  {r[3]:>5.1f}/day  prof {r[4]}/{r[5]}  sumR {r[6]:+.1f}  medPF {r[7]:.2f}")
else:
    for r in lead[:12]:
        log(f"  {r[0]:<12} {r[1]:<11} {r[2]}  {r[3]:>5.1f}/day  prof {r[4]}/{r[5]}  sumR {r[6]:+.1f}  medPF {r[7]:.2f}  worst {r[8]:+.1f}")
log(f"campaign 1 done ({time.time()-t0:.0f}s). results -> {RESCSV}")
