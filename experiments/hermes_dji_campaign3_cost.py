"""
Campaign 3 — TRANSACTION-COST gate on the real edge (pb1.0, near_ema20).

The edge is only +0.17R/trade, so spread/commission decides deployability.
Re-runs the WF, collects every taken trade's gross R (label computed at ZERO spread),
then nets various realistic US30 spreads. Reports net sumR, per-trade, % profitable
windows, and the BREAKEVEN spread (in R and in index points).
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
times = df["time"].values; n = len(df); medatr = np.nanmedian(atr)
base_ok = np.isfinite(atr) & (atr > 0) & (cdir != 0); base_ok[:300] = False; base_ok[-301:] = False
print(f"median M1 ATR = {medatr:.2f} index points  (so 1R ≈ {medatr:.1f} pts; SL6 risk ≈ {6*medatr:.0f} pts)")

def mk(mask): idx = np.where(mask & base_ok)[0]; return idx, cdir[idx].astype(np.int64)
SPECS = {"pb1.0": (da <= 1.0, 0.3602), "near_ema20": (np.abs(demaA) <= 0.5, 0.1099)}

@njit
def sim_trail_gross(idxs, dirs, O, H, L, C, atr, SL, TRAIL, MAXH, n):
    # gross R (NO spread) + exit_bar
    m = len(idxs); pnl = np.empty(m); xit = np.empty(m, np.int64)
    for k in range(m):
        i = idxs[k]; d = dirs[k]; a = atr[i]; ei = i + 1
        if ei >= n or not (a > 0): pnl[k] = 0.0; xit[k] = min(i + 1, n - 1); continue
        ep = O[ei]; hard = SL * a; trd = TRAIL * a; mf = 0.0; end = min(ei + MAXH, n - 1); done = False
        for jx in range(ei, end + 1):
            fav = d * (C[jx] - ep)
            if fav > mf: mf = fav
            if d == 1 and (ep - L[jx]) >= hard: pnl[k] = -SL; xit[k] = jx; done = True; break
            if d == -1 and (H[jx] - ep) >= hard: pnl[k] = -SL; xit[k] = jx; done = True; break
            if mf >= trd and (mf - fav) >= trd: pnl[k] = (mf - trd) / a; xit[k] = jx; done = True; break
        if not done: pnl[k] = d * (C[end] - ep) / a; xit[k] = end
    return pnl, xit

@njit
def port_take(order_idx, entry_bar, exit_bar, pnl, cooldown):
    busy = -1; m = len(order_idx); R = np.zeros(m); cnt = 0
    for t in range(m):
        k = order_idx[t]
        if entry_bar[k] <= busy: continue
        R[cnt] = pnl[k]; busy = exit_bar[k] + cooldown; cnt += 1
    return R[:cnt]

WINS = []; first = pd.Timestamp("2020-07-01"); tsw = first; lastd = pd.to_datetime(times).max()
while tsw < lastd:
    WINS.append((tsw - pd.DateOffset(years=3), tsw, tsw + pd.DateOffset(months=6))); tsw += pd.DateOffset(months=6)
SPREADS_PTS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]   # index points round-trip-ish (one spread per trade)

print(f"\n{'='*92}\nCAMPAIGN 3 — COST GATE (gross edge net of US30 spread). 1R ≈ {medatr:.1f} pts.\n{'='*92}")
for bname, (mask, thr) in SPECS.items():
    idx, dirs = mk(mask)
    pnlG, xit = sim_trail_gross(idx, dirs, O, H, L, C, atr, 6.0, 2.0, 300, n)
    ct = pd.to_datetime(times[idx]); Xc = feat[FC].to_numpy(np.float32)[idx]; entry_bar = idx + 1
    rng = np.random.RandomState(0)
    # collect taken-trade gross R per window (model trained on gross R)
    win_R = []
    for tr_s, te_s, te_e in WINS:
        trm = (ct >= tr_s) & (ct < te_s); tem = (ct >= te_s) & (ct < te_e)
        if trm.sum() < 4000 or tem.sum() < 20: win_R.append(np.array([])); continue
        tix = np.where(trm)[0]
        if len(tix) > 150000: tix = rng.choice(tix, 150000, replace=False)
        m = XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.85,
                         colsample_bytree=0.85, min_child_weight=10, n_jobs=-1, random_state=0)
        m.fit(Xc[tix], pnlG[tix]); pred = np.full(len(idx), -9e9); pred[tem] = m.predict(Xc[tem])
        sel = pred >= thr; kk = np.where((ct >= te_s) & (ct < te_e) & sel)[0]
        if len(kk) < 3: win_R.append(np.array([])); continue
        order = kk[np.argsort(entry_bar[kk])]
        R = port_take(order.astype(np.int64), entry_bar, xit, pnlG, 5)
        win_R.append(R)
    allR = np.concatenate(win_R) if win_R else np.array([]); ntr = len(allR); perday = ntr / 1541
    print(f"\n--- {bname}  (thr={thr}, {perday:.1f}/day, {ntr} trd) ---")
    print(f"  gross per-trade {allR.mean():+.3f}R | gross total {allR.sum():+.0f}R")
    print(f"  {'spread_pts':>10}{'=R':>7}{'net/trade':>11}{'net sumR':>10}{'profWk':>8}{'medPF':>7}")
    breakeven = None
    for spt in SPREADS_PTS:
        spR = spt / medatr
        net_win = [w - spR for w in win_R if len(w)]
        nettot = sum(w.sum() for w in net_win); ntpt = allR.mean() - spR
        prof = sum(1 for w in net_win if w.sum() > 0); nwk = len(net_win)
        pfs = []
        for w in net_win:
            pos = w[w > 0].sum(); neg = -w[w <= 0].sum(); pfs.append(pos / max(neg, 1e-9))
        medpf = np.median(pfs) if pfs else 0
        mark = "  <= breakeven" if (breakeven is None and ntpt <= 0) else ""
        if breakeven is None and ntpt <= 0: breakeven = spt
        print(f"  {spt:>10.1f}{spR:>7.2f}{ntpt:>+11.3f}{nettot:>+10.0f}{prof:>6}/{nwk:<1}{medpf:>7.2f}{mark}")
    be_pts = (allR.mean()) * medatr
    print(f"  >>> BREAKEVEN spread ≈ {be_pts:.1f} index points  ({allR.mean():.3f}R).  "
          f"Deployable if real US30 spread < {be_pts:.1f} pts.")
log("campaign 3 done")
