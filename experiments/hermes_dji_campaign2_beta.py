"""
Campaign 2 — ADVERSARIAL beta test on Campaign 1's winners.

Campaign 1 found 'in_regime / near_ema20 / pb1.0 SL6/T2 reg' giving 12/12 positive
WF windows at ~16 trades/day. DJI doubled 2020-2026 and entries are mostly LONG, so
this could be pure long beta, not skill. This test decomposes each winner:
  - long-only sumR vs short-only sumR  (skill works both ways; beta is long-only)
  - per-window strategy sumR vs BUY-AND-HOLD-long over the same window
  - % of profit from shorts
A real edge: shorts not bleeding AND strategy beats buy-and-hold. Else = beta.
"""
import sys, pickle, time
from pathlib import Path
import numpy as np, pandas as pd
from numba import njit
from xgboost import XGBRegressor

DE = Path("/home/jay/Desktop/my-agents-and-website/commercial/server/decision_engine")
sys.path.insert(0, str(DE))
import hermes_features as hf
from configs.hermes_dji import HERMES_DJI as CFG
FC = pickle.load(open(DE / "models/hermes_dji_validated.pkl", "rb"))["feat_cols"]
t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

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
log("features done")

atr = feat["atr14"].to_numpy(float); cdir = feat["committed_dir"].to_numpy(np.int64)
ds = feat["dist_at_signal"].to_numpy(float); da = np.abs(ds); demaA = feat["dist_ema20"].to_numpy(float)
O = df["open"].to_numpy(float); H = df["high"].to_numpy(float); L = df["low"].to_numpy(float); C = df["close"].to_numpy(float)
times = df["time"].values; n = len(df); sp = CFG.spread_usd / np.nanmedian(atr); medatr = np.nanmedian(atr)
base_ok = np.isfinite(atr) & (atr > 0) & (cdir != 0); base_ok[:300] = False; base_ok[-301:] = False

def mk(mask): idx = np.where(mask & base_ok)[0]; return idx, cdir[idx].astype(np.int64)
# base -> (mask, Campaign-1 best threshold giving ~10-20/day taken)
SPECS = {
    "near_ema20": (np.abs(demaA) <= 0.5, 0.1099),
    "pb1.0":      (da <= 1.0,            0.3602),
    "pb1.5":      (da <= 1.5,            0.1492),
    "in_regime":  (np.ones(n, bool),     0.4095),
}
BASES = {k: (mk(m)[0], mk(m)[1], thr) for k, (m, thr) in SPECS.items()}

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
def port_dir(order_idx, entry_bar, exit_bar, pnl, dirs, cooldown):
    busy = -1; m = len(order_idx); R = np.zeros(m); D = np.zeros(m, np.int64); cnt = 0
    for t in range(m):
        k = order_idx[t]
        if entry_bar[k] <= busy: continue
        R[cnt] = pnl[k]; D[cnt] = dirs[k]; busy = exit_bar[k] + cooldown; cnt += 1
    return R[:cnt], D[:cnt]

WINS = []; first = pd.Timestamp("2020-07-01"); tsw = first; lastd = pd.to_datetime(times).max()
while tsw < lastd:
    WINS.append((tsw - pd.DateOffset(years=3), tsw, tsw + pd.DateOffset(months=6))); tsw += pd.DateOffset(months=6)

def maxdd(equity):
    return float((np.maximum.accumulate(equity) - equity).max()) if len(equity) else 0.0

print(f"\n{'='*96}\nCAMPAIGN 2b — BETA TEST at Campaign-1 thresholds (SL6/T2 reg, ~16/day). Risk-adjusted vs B&H.\n{'='*96}")
for bname, (idx, dirs, thr) in BASES.items():
    pnl, xit = sim_trail(idx, dirs, O, H, L, C, atr, float(sp), 6.0, 2.0, 300, n)
    ct = pd.to_datetime(times[idx]); Xc = feat[FC].to_numpy(np.float32)[idx]; entry_bar = idx + 1
    rng = np.random.RandomState(0); preds = []
    for tr_s, te_s, te_e in WINS:
        trm = (ct >= tr_s) & (ct < te_s); tem = (ct >= te_s) & (ct < te_e)
        if trm.sum() < 4000 or tem.sum() < 20: preds.append((te_s, te_e, np.full(len(idx), -9e9))); continue
        tix = np.where(trm)[0]
        if len(tix) > 150000: tix = rng.choice(tix, 150000, replace=False)
        m = XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.85,
                         colsample_bytree=0.85, min_child_weight=10, n_jobs=-1, random_state=0)
        m.fit(Xc[tix], pnl[tix]); full = np.full(len(idx), -9e9); full[tem] = m.predict(Xc[tem]); preds.append((te_s, te_e, full))
    tot_long = tot_short = 0.0; nL = nS = 0; tot_bh = 0.0; strat_tot = 0.0; nwk = 0
    all_R = []; bh_series = []; rows = []
    for (te_s, te_e, full) in preds:
        sel = full >= thr; wm = (ct >= te_s) & (ct < te_e) & sel; kk = np.where(wm)[0]
        if len(kk) < 3: continue
        order = kk[np.argsort(entry_bar[kk])]
        R, D = port_dir(order.astype(np.int64), entry_bar, xit, pnl, dirs, 5)
        if len(R) == 0: continue
        lR = R[D == 1].sum(); sR = R[D == -1].sum()
        wi = np.where((pd.to_datetime(times) >= te_s) & (pd.to_datetime(times) < te_e))[0]
        bh = (C[wi[-1]] - C[wi[0]]) / medatr if len(wi) > 1 else 0.0
        tot_long += lR; tot_short += sR; nL += int((D == 1).sum()); nS += int((D == -1).sum())
        tot_bh += bh; strat_tot += R.sum(); nwk += 1; all_R.extend(R.tolist()); bh_series.append(bh)
        rows.append((te_s.date(), R.sum(), lR, sR, int((D==1).sum()), int((D==-1).sum()), bh))
    all_R = np.array(all_R); perday = len(all_R) / 1541
    s_dd = maxdd(np.cumsum(all_R)); bh_dd = maxdd(np.cumsum(bh_series))
    sharpe = all_R.mean() / (all_R.std() + 1e-9) * np.sqrt(len(all_R))  # crude total-period sharpe
    s_ret_risk = strat_tot / max(s_dd, 1e-9); bh_ret_risk = tot_bh / max(bh_dd * 1, 1e-9)
    print(f"\n--- {bname}  thr={thr}  ({perday:.1f}/day, {len(all_R)} trd) ---")
    print(f"  strat sumR {strat_tot:+.0f}  (LONG {tot_long:+.0f}/{nL}trd  SHORT {tot_short:+.0f}/{nS}trd)  maxDD {s_dd:.0f}R")
    print(f"  shorts profitable? {'YES +'+str(int(tot_short))+'R' if tot_short>0 else 'NO '+str(int(tot_short))+'R'}   "
          f"short profit share {100*tot_short/max(strat_tot,1e-9):+.0f}%   per-trade {all_R.mean():+.3f}R")
    print(f"  RISK-ADJ:  strat ret/DD {s_ret_risk:.2f} (sumR {strat_tot:+.0f}/DD {s_dd:.0f})   "
          f"vs  B&H ret/DD {bh_ret_risk:.2f} (sumR {tot_bh:+.0f}/DD {bh_dd:.0f})")
    real = tot_short > 0 and s_ret_risk > bh_ret_risk
    print(f"  >>> {'REAL EDGE — shorts +ve AND better risk-adjusted than B&H' if real else 'LIKELY BETA — shorts weak or worse risk-adj than B&H'}")
    for r in rows: print(f"     {r[0]}  {r[1]:>+7.0f}  L{r[2]:>+6.0f} S{r[3]:>+6.0f}  {r[4]:>3}/{r[5]:<3}  B&H {r[6]:>+6.0f}")
log("campaign 2b done")
