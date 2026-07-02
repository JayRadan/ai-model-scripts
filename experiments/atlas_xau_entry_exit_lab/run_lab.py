"""
ATLAS XAU — entry/exit lab on the deployed edge_pullback engine.

Base (deployed 2026-06-30, SL7 2026-07-01): pullback candidates |dist_tfk|<=1.0,
dir=committed_dir, XGBRegressor on gross R (SL7/trail2/maxhold300 labels),
take if pred>=thr (~11/day), 1-slot cooldown5, market entry at next open.

This lab tests NEW entry + exit mechanics on that base, XAU-specific:
  ENTRY: limit entry at next_open -/+ delta*ATR (TTL 5 bars)  [price improvement
         vs adverse selection — attacks the spread-sensitivity caveat],
         confirm-bar entry (1 bar must close in trade direction),
         quantile-gate (XGB q30 ranks by downside, not mean).
  EXIT:  BE-lock (peak>=2R -> floor +0.2R), confidence-tiered trail
         (pred high -> trail 3, low -> trail 1.5), time-tighten (trail 2->1
         after 60 bars), SL5 sanity, plus pre-registered entry x exit combos.

HONESTY PROTOCOL:
  - exact live feature pipeline (decision_engine.edge_pullback.compute_edge_features)
  - walk-forward: 6-month test windows, 3y rolling train, thresholds calibrated
    on TRAIN ONLY to ~11 trades/day per variant (comparable frequency)
  - DEV windows = test starts 2020-07 .. 2024-07 (selection happens here)
  - HOLDOUT windows = 2025-01, 2025-07, 2026-01 (NEVER used for selection)
  - net PnL at per-trade spread cost spread_$/ATR(signal), grid $0.10/0.20/0.30
    (XAU raw spread ~0.10-0.15, standard ~0.25-0.35)
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
FC = pickle.load(open(SRV / "decision_engine/models/atlas_xau_validated.pkl", "rb"))["feat_cols"]
t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)

SPREADS = [0.10, 0.20, 0.30]   # USD, applied per trade as spread/atr_at_signal
HEAD_SP = 0.20                 # headline spread for selection
TARGET_PER_DAY = 11.0; COOLDOWN = 5
DEV_END = pd.Timestamp("2025-01-01")

# ---------------------------------------------------------------- data + features
df = pd.read_parquet("/home/jay/Desktop/new-model-zigzag/data/m1_xau_full.parquet")
df["time"] = pd.to_datetime(df["time"]); df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
log(f"{len(df):,} bars {df.time.iloc[0].date()} -> {df.time.iloc[-1].date()}")
feat = ep.compute_edge_features(df)   # IDENTICAL to live serving (causal HTF inside)
log("features done")

atr = feat["atr14"].to_numpy(float); cdir = feat["committed_dir"].to_numpy(np.int64)
da = np.abs(feat["dist_at_signal"].to_numpy(float))
O = df["open"].to_numpy(float); H = df["high"].to_numpy(float); L = df["low"].to_numpy(float); C = df["close"].to_numpy(float)
times = df["time"].values; n = len(df)
ok = np.isfinite(atr) & (atr > 0) & (cdir != 0); ok[:300] = False; ok[-301:] = False
idx = np.where((da <= 1.0) & ok)[0]; dirs = cdir[idx].astype(np.int64)
ct = pd.to_datetime(times[idx]); Xc = feat[FC].to_numpy(np.float32)[idx]
sig_atr = atr[idx]
log(f"candidates {len(idx):,} | median ATR ${np.median(sig_atr):.3f} "
    f"(2025+: ${np.median(sig_atr[ct >= '2025-01-01']):.3f})")

# ---------------------------------------------------------------- universal simulator
@njit(cache=True)
def sim_u(idxs, dirs, O, H, L, C, atr, n,
          entry_mode, delta, ttl,          # 0 market @ next open | 1 limit | 2 confirm-bar
          SL, TRAIL, MAXH,
          be_peak, be_floor,               # <=0 -> off ; lock: mf>=be_peak*a -> stop at ep+d*be_floor*a
          tight_after, tight_trail):       # <=0 -> off ; after N bars trail = tight_trail
    m = len(idxs)
    filled = np.zeros(m, np.uint8); pnl = np.zeros(m); ebar = np.full(m, -1, np.int64); xit = np.full(m, -1, np.int64)
    for k in range(m):
        i = idxs[k]; d = dirs[k]; a = atr[i]
        if i + 1 >= n or not (a > 0): continue
        # ---- entry
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
        # ---- exit path
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

# ---------------------------------------------------------------- global variant sims (model-independent)
def SIM(em=0, delta=0.0, ttl=5, SL=7.0, TR=2.0, MH=300, bp=0.0, bf=0.0, ta=0, tt=0.0):
    return sim_u(idx, dirs, O, H, L, C, atr, n, em, delta, ttl, SL, TR, MH, bp, bf, ta, tt)

log("running global sims...")
SIMS = {}
SIMS["base"]        = SIM()                                        # deployed
SIMS["sl5"]         = SIM(SL=5.0)
SIMS["belock"]      = SIM(bp=2.0, bf=0.2)
SIMS["tt60"]        = SIM(ta=60, tt=1.0)
SIMS["trail15"]     = SIM(TR=1.5)                                  # tier component
SIMS["trail3"]      = SIM(TR=3.0)                                  # tier component
SIMS["lim030"]      = SIM(em=1, delta=0.30)
SIMS["lim050"]      = SIM(em=1, delta=0.50)
SIMS["confirm"]     = SIM(em=2)
# pre-registered combos
SIMS["lim030_belock"]  = SIM(em=1, delta=0.30, bp=2.0, bf=0.2)
SIMS["lim050_belock"]  = SIM(em=1, delta=0.50, bp=2.0, bf=0.2)
SIMS["confirm_belock"] = SIM(em=2, bp=2.0, bf=0.2)
SIMS["lim030_tt60"]    = SIM(em=1, delta=0.30, ta=60, tt=1.0)
for nm, (f, p, e, x) in SIMS.items():
    fr = f.mean() * 100
    log(f"  {nm:<16} fill {fr:5.1f}%  mean gross R {p[f == 1].mean():+.3f}")

# variant spec: (sim_key, gate) — gate 'mean' | 'q30' | 'tier'
VARIANTS = [
    ("V0_deployed",      "base",           "mean"),
    ("X_sl5",            "sl5",            "mean"),
    ("X_belock",         "belock",         "mean"),
    ("X_tt60",           "tt60",           "mean"),
    ("X_tier_trail",     "tier",           "mean"),
    ("E_lim030",         "lim030",         "mean"),
    ("E_lim050",         "lim050",         "mean"),
    ("E_confirm",        "confirm",        "mean"),
    ("E_q30gate",        "base",           "q30"),
    ("C_lim030_belock",  "lim030_belock",  "mean"),
    ("C_lim050_belock",  "lim050_belock",  "mean"),
    ("C_confirm_belock", "confirm_belock", "mean"),
    ("C_lim030_tt60",    "lim030_tt60",    "mean"),
    ("C_q30_belock",     "belock",         "q30"),
]

# ---------------------------------------------------------------- walk-forward
WINS = []; tsw = pd.Timestamp("2020-07-01"); lastd = ct.max()
while tsw < lastd:
    WINS.append((tsw - pd.DateOffset(years=3), tsw, tsw + pd.DateOffset(months=6))); tsw += pd.DateOffset(months=6)
rng = np.random.RandomState(0)
XGB = dict(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.85,
           colsample_bytree=0.85, min_child_weight=10, n_jobs=-1, random_state=0)

def calibrate(preds, trm, tr_days, ff, eb_, xit_):
    """train-only threshold targeting TARGET_PER_DAY taken/day for THIS variant"""
    trk = np.where(trm & (ff == 1))[0]
    if len(trk) < 50: return np.inf
    cand = np.quantile(preds[trk], np.linspace(0.30, 0.97, 24))
    thr = cand[-1]; best = 1e18
    for th in cand:
        kk = trk[preds[trk] >= th]
        if len(kk) < 5: continue
        order = kk[np.argsort(eb_[kk])]
        tk = take(order.astype(np.int64), eb_, xit_, COOLDOWN)
        gap = abs(len(tk) / tr_days - TARGET_PER_DAY)
        if gap < best: best = gap; thr = th
    return thr

results = {v[0]: [] for v in VARIANTS}
trades  = {v[0]: [] for v in VARIANTS}     # (exit_time, net_headline) for equity/DD
base_pnl = SIMS["base"][1]

for wi_, (tr_s, te_s, te_e) in enumerate(WINS):
    trm = (ct >= tr_s) & (ct < te_s); tem = (ct >= te_s) & (ct < te_e)
    if trm.sum() < 4000 or tem.sum() < 20: continue
    tix = np.where(trm)[0]; tix_f = tix if len(tix) <= 150_000 else rng.choice(tix, 150_000, replace=False)
    tr_days = max((ct[tix].max() - ct[tix].min()).days * 5 / 7, 1)
    te_days = max((ct[tem].max() - ct[tem].min()).days * 5 / 7, 1)

    m_mean = XGBRegressor(**XGB); m_mean.fit(Xc[tix_f], base_pnl[tix_f])
    m_q30 = XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.30, **XGB)
    m_q30.fit(Xc[tix_f], base_pnl[tix_f])
    p_mean = m_mean.predict(Xc).astype(np.float64); p_q30 = m_q30.predict(Xc).astype(np.float64)

    # tier arrays: split at median train pred among gated candidates (base gate)
    ff_b, _, eb_b, xit_b = SIMS["base"]
    thr_base = calibrate(p_mean, trm, tr_days, ff_b, eb_b, xit_b)
    gated_tr = np.where(trm & (p_mean >= thr_base))[0]
    tau = np.median(p_mean[gated_tr]) if len(gated_tr) else np.inf
    hi = p_mean >= tau
    f15, p15, e15, x15 = SIMS["trail15"]; f30, p30, e30, x30 = SIMS["trail3"]
    tier_sim = (np.where(hi, f30, f15), np.where(hi, p30, p15),
                np.where(hi, e30, e15), np.where(hi, x30, x15))

    for vname, skey, gate in VARIANTS:
        ff, pnl_, eb_, xit_ = tier_sim if skey == "tier" else SIMS[skey]
        preds = p_q30 if gate == "q30" else p_mean
        thr = thr_base if (skey == "base" and gate == "mean") else calibrate(preds, trm, tr_days, ff, eb_, xit_)
        kk = np.where(tem & (ff == 1) & (preds >= thr))[0]
        if len(kk) == 0:
            results[vname].append(dict(win=str(te_s.date()), n=0)); continue
        order = kk[np.argsort(eb_[kk])]
        tk = take(order.astype(np.int64), eb_, xit_, COOLDOWN)
        R = pnl_[tk]; cost = 1.0 / sig_atr[tk]
        nets = {sp: R - sp * cost for sp in SPREADS}
        nh = nets[HEAD_SP]
        results[vname].append(dict(
            win=str(te_s.date()), n=len(tk), perday=len(tk) / te_days, gross=R.sum(),
            **{f"net{int(sp*100)}": nets[sp].sum() for sp in SPREADS},
            wr=(nh > 0).mean() * 100,
            pf=nh[nh > 0].sum() / max(-nh[nh <= 0].sum(), 1e-9)))
        for tt_, r_ in zip(times[xit_[tk]], nh): trades[vname].append((tt_, r_))
    log(f"window {te_s.date()} done (thr_base={thr_base:.3f})")

# ---------------------------------------------------------------- report
def agg(rows, dev):
    rr = [r for r in rows if r.get("n", 0) > 0 and ((pd.Timestamp(r["win"]) < DEV_END) == dev)]
    if not rr: return None
    d = dict(nwin=len(rr), n=sum(r["n"] for r in rr),
             perday=np.mean([r["perday"] for r in rr]), gross=sum(r["gross"] for r in rr))
    for sp in SPREADS:
        key = f"net{int(sp*100)}"
        d[key] = sum(r[key] for r in rr); d[key + "_w"] = sum(1 for r in rr if r[key] > 0)
    d["wr"] = np.average([r["wr"] for r in rr], weights=[r["n"] for r in rr])
    return d

def maxdd(vname, dev):
    tr = [(t, r) for t, r in trades[vname] if (pd.Timestamp(t) < DEV_END) == dev]
    if not tr: return 0.0
    tr.sort(key=lambda z: z[0]); eq = np.cumsum([r for _, r in tr])
    return float((np.maximum.accumulate(eq) - eq).max())

hdr = f"{'variant':<18}{'n':>7}{'/day':>6}{'gross':>8}{'net@10c':>9}{'net@20c':>9}{'net@30c':>9}{'w+@20':>7}{'WR%':>6}{'DD@20':>7}"
print(f"\n{'='*96}\nATLAS XAU entry/exit lab — DEV windows (2020-07 .. 2024-12), train-only thr ~11/day, 1-slot\n{'='*96}\n{hdr}")
summary = {}
for vname, _, _ in VARIANTS:
    a = agg(results[vname], dev=True)
    if a is None: continue
    summary[vname] = a
    print(f"{vname:<18}{a['n']:>7}{a['perday']:>6.1f}{a['gross']:>+8.0f}{a['net10']:>+9.0f}{a['net20']:>+9.0f}"
          f"{a['net30']:>+9.0f}{a['net20_w']:>4}/{a['nwin']:<2}{a['wr']:>6.0f}{maxdd(vname, True):>7.0f}")

best = max(summary, key=lambda v: summary[v]["net20"])
print(f"\nDEV WINNER by net@20c: {best}")

print(f"\n{'='*96}\nHOLDOUT windows (2025-01 .. 2026-05) — untouched by selection\n{'='*96}\n{hdr}")
for vname, _, _ in VARIANTS:
    a = agg(results[vname], dev=False)
    if a is None: continue
    tag = " <== DEV WINNER" if vname == best else (" <== deployed" if vname == "V0_deployed" else "")
    print(f"{vname:<18}{a['n']:>7}{a['perday']:>6.1f}{a['gross']:>+8.0f}{a['net10']:>+9.0f}{a['net20']:>+9.0f}"
          f"{a['net30']:>+9.0f}{a['net20_w']:>4}/{a['nwin']:<2}{a['wr']:>6.0f}{maxdd(vname, False):>7.0f}{tag}")

# per-window detail for deployed + winner
for vname in dict.fromkeys(["V0_deployed", best]):
    print(f"\n--- {vname} per window ---")
    for r in results[vname]:
        if r.get("n", 0) == 0: print(f"{r['win']}  (no trades)"); continue
        print(f"{r['win']}  n={r['n']:>4} {r['perday']:>5.1f}/d gross{r['gross']:>+7.0f} "
              f"net20{r['net20']:>+7.0f} net30{r['net30']:>+7.0f} WR{r['wr']:>4.0f}% PF{r['pf']:>5.2f}")

# equity PNG: deployed vs winner (net@20c), holdout shaded
fig, ax = plt.subplots(figsize=(13, 5.5))
for vname, col in [("V0_deployed", "#64748b"), (best, "#16a34a")]:
    tr = sorted(trades[vname], key=lambda z: z[0])
    tt_ = pd.to_datetime([t for t, _ in tr]); eq = np.cumsum([r for _, r in tr])
    ax.plot(tt_, eq, lw=1.2, color=col, label=f"{vname} ({eq[-1]:+.0f}R)")
ax.axvline(DEV_END, color="#dc2626", lw=1, ls="--"); ax.axvspan(DEV_END, pd.Timestamp("2026-06-01"), alpha=0.06, color="#dc2626")
ax.text(DEV_END, ax.get_ylim()[0], " HOLDOUT →", color="#dc2626", fontsize=9, va="bottom")
ax.set_title("Atlas XAU entry/exit lab — WF equity, net @ $0.20 spread (R units)")
ax.axhline(0, color="k", lw=0.6); ax.grid(alpha=0.3); ax.legend()
plt.tight_layout(); plt.savefig(OUT / "lab_equity.png", dpi=110)
print(f"\nequity -> {OUT/'lab_equity.png'}")

json.dump({v: results[v] for v, _, _ in VARIANTS}, open(OUT / "lab_results.json", "w"), default=str, indent=1)
log("lab done")
