"""
Build the DJI hedge head (2026-07-14) and append it to the LIVE hermes_dji
bundle. Validated recipe (run_lab_hedge_v2.py, d30 trigger):
  - portfolio = trades taken by the DEPLOYED q10 gate (its model + threshold)
  - snapshot at bar 30 held (trade still open), features = 29 live feats at the
    snapshot bar + path stats (fnow, mfe, mae, uw_frac, slope10, held)
  - label = the reverse trade's net R (SL2/trail2/tt30/0.75/maxh300, own ATR)
    charged 2.0pt spread
  - head = XGBRegressor (same hyperparams as v2); hedge_tau calibrated on the
    trailing-3y train hedge stream (grid 0/.1/.25/.5, max net; must be > 0)
Appends: hedge_model, hedge_tau, hedge_after=30, hedge_sl_R=2.0.
Gate model/threshold/exits UNTOUCHED. Backup .bak_pre_hedge_2026-07-14.
"""
import sys, pickle, time, shutil
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd
from numba import njit
from xgboost import XGBRegressor

SRV = Path("/home/jay/Desktop/my-agents-and-website/commercial/server")
sys.path.insert(0, str(SRV))
import decision_engine.edge_pullback as ep
MODELS = SRV / "decision_engine" / "models"
t0 = time.time(); log = lambda m: print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)
COOLDOWN = 5; TA, TT = 30, 0.75; SP = 2.0; TAUS = (0.0, 0.1, 0.25, 0.5)

import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_IDX_AMERICA_E_D_J_IND

@njit(cache=True)
def sim_tt(idxs, dirs, O, H, L, C, atr, n, SL, TRAIL, MAXH, ta, tt):
    m = len(idxs); pnl = np.zeros(m); ebar = np.full(m, -1, np.int64); xit = np.full(m, -1, np.int64)
    for k in range(m):
        i = idxs[k]; d = dirs[k]; a = atr[i]
        if i + 1 >= n or not (a > 0): continue
        st = i + 1; epr = O[st]; hard = SL * a; mf = 0.0
        end = min(st + MAXH, n - 1); done = False
        for jx in range(st, end + 1):
            adv = (epr - L[jx]) if d == 1 else (H[jx] - epr)
            if adv >= hard: pnl[k] = -SL; xit[k] = jx; done = True; break
            fav = d * (C[jx] - epr)
            if fav > mf: mf = fav
            trd = TRAIL * a
            if ta > 0 and (jx - st) >= ta:
                w = tt * a
                if w < trd: trd = w
            if mf >= trd and (mf - fav) >= trd: pnl[k] = (mf - trd) / a; xit[k] = jx; done = True; break
        if not done: pnl[k] = d * (C[end] - epr) / a; xit[k] = end
        ebar[k] = st
    return pnl, ebar, xit

@njit(cache=True)
def snapshots30(idxs, dirs, O, C, atr, ebar, xit):
    m = len(idxs)
    snap = np.full(m, -1, np.int64); stats = np.zeros((m, 6))
    for k in range(m):
        st = ebar[k]; xb = xit[k]
        if st < 0 or xb <= st + 30: continue
        sb = st + 30
        i = idxs[k]; d = dirs[k]; a = atr[i]; epr = O[st]
        mf = -1e18; mn = 1e18; uw = 0; cnt = 0
        for jx in range(st, sb + 1):
            fv = d * (C[jx] - epr)
            if fv > mf: mf = fv
            if fv < mn: mn = fv
            if fv < 0: uw += 1
            cnt += 1
        f_now = d * (C[sb] - epr)
        snap[k] = sb
        stats[k, 0] = f_now / a; stats[k, 1] = mf / a; stats[k, 2] = mn / a
        stats[k, 3] = uw / cnt; stats[k, 4] = (f_now - d * (C[max(st, sb - 10)] - epr)) / a
        stats[k, 5] = sb - st
    return snap, stats

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

def fetch_tail(instr, start):
    end = datetime.now(timezone.utc); out = []; cur = pd.Timestamp(start)
    while cur < pd.Timestamp(end).tz_localize(None):
        nxt = min(cur + pd.Timedelta(days=90), pd.Timestamp(end).tz_localize(None))
        try:
            r = dukascopy_python.fetch(instrument=instr, interval=dukascopy_python.INTERVAL_MIN_1,
                    offer_side=dukascopy_python.OFFER_SIDE_BID,
                    start=cur.to_pydatetime().replace(tzinfo=timezone.utc),
                    end=nxt.to_pydatetime().replace(tzinfo=timezone.utc))
            if r is not None and len(r): out.append(r)
        except Exception as e: log(f"  fetch err {e}")
        cur = nxt
    if not out: return None
    d = pd.concat(out).reset_index().rename(columns={"timestamp": "time"})
    d["time"] = pd.to_datetime(d["time"]).dt.tz_convert("UTC").dt.tz_localize(None)
    return d

bundle = pickle.load(open(MODELS / "hermes_dji_validated.pkl", "rb"))
assert bundle["version"].startswith("edge_pullback_v4_q10"), bundle["version"]
FC = bundle["feat_cols"]
df = pd.read_parquet("/home/jay/Desktop/new-model-zigzag/data/m1_dji_full.parquet")
df = df.rename(columns={[c for c in df.columns if "time" in c.lower()][0]: "time"})
df["time"] = pd.to_datetime(df["time"]); df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
tail = fetch_tail(INSTRUMENT_IDX_AMERICA_E_D_J_IND, df["time"].iloc[-1] - pd.Timedelta(days=2))
if tail is not None:
    df = pd.concat([df, tail], ignore_index=True).sort_values("time").drop_duplicates("time").reset_index(drop=True)
if "tick_volume" not in df.columns: df["tick_volume"] = df.get("volume", 0)
log(f"{len(df):,} M1 bars -> {df.time.iloc[-1]}")
feat = ep.compute_edge_features(df)
atr = feat["atr14"].to_numpy(float); cdir = feat["committed_dir"].to_numpy(np.int64)
da = np.abs(feat["dist_at_signal"].to_numpy(float))
O = df["open"].to_numpy(float); H = df["high"].to_numpy(float); L = df["low"].to_numpy(float); C = df["close"].to_numpy(float)
n = len(df)
ok = np.isfinite(atr) & (atr > 0) & (cdir != 0); ok[:300] = False; ok[-301:] = False
idx = np.where((da <= 1.0) & ok)[0]; dirs = cdir[idx].astype(np.int64)
ct = pd.to_datetime(df["time"].values[idx])
X29 = np.nan_to_num(feat[FC].to_numpy(np.float32)[idx], nan=0.0, posinf=0.0, neginf=0.0)
FEAT_ALL = np.nan_to_num(feat[FC].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
del feat
pnlB, ebB, xitB = sim_tt(idx, dirs, O, H, L, C, atr, n, 7.0, 2.0, 300, TA, TT)

# taken trades under the DEPLOYED gate (its model + threshold), trailing 3y
p = bundle["q_model"].predict(X29).astype(np.float64)
thr = float(bundle["threshold"])
tr_s = ct.max() - pd.DateOffset(years=3)
kk = np.where((ct >= tr_s) & (p >= thr))[0]
tk = take(kk[np.argsort(ebB[kk])].astype(np.int64), ebB, xitB, COOLDOWN)
log(f"deployed-gate trades in trailing 3y: {len(tk):,}")

snap, stats = snapshots30(idx, dirs, O, C, atr, ebB, xitB)
hp, ha, hv = sim_hedge(snap, dirs, O, H, L, C, atr, n, 2.0, 2.0, 300, 30, 0.75)
s = tk[(snap[tk] >= 0) & (hv[tk] == 1)]
y = hp[s] - SP / np.maximum(ha[s], 1e-9)
X = np.nan_to_num(np.concatenate([FEAT_ALL[np.clip(snap[s], 0, n - 1)],
                                  stats[s].astype(np.float32)], axis=1),
                  nan=0.0, posinf=0.0, neginf=0.0)
log(f"snapshots: {len(s):,} (of {len(tk):,} trades), label mean {y.mean():+.3f}R")
mh = XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.85,
                  colsample_bytree=0.85, min_child_weight=20, n_jobs=-1, random_state=0)
mh.fit(X, y)
ph = mh.predict(X).astype(np.float64)
best_tau, best_net = None, -1e18
for tau in TAUS:
    sel = ph >= tau
    if sel.sum() < 10: continue
    nt = float(y[sel].sum())
    log(f"  tau {tau}: n={int(sel.sum())} trainNet {nt:+.0f}R  per-hedge {y[sel].mean():+.3f}R")
    if nt > best_net: best_net, best_tau = nt, tau
assert best_tau is not None and best_net > 0, f"hedge head not train-positive ({best_net}) — NOT shipping"

bundle["hedge_model"] = mh
bundle["hedge_tau"] = float(best_tau)
bundle["hedge_after"] = 30
bundle["hedge_sl_R"] = 2.0
bundle["hedge_recipe"] = ("one-shot hedge at bar-30 held: head predicts reverse-trade net R "
                          "(29 feats @ snapshot + path stats), open -dir if pred>=tau; hedge "
                          "exits SL2/trail2/tt30/0.75/maxh300. run_lab_hedge_v2.py "
                          "dev +171R 7/9 @2pt, holdout +282R 3/3 @2pt. v3 dynamic REJECTED.")
out = MODELS / "hermes_dji_validated.pkl"
bak = MODELS / "hermes_dji_validated.pkl.bak_pre_hedge_2026-07-14"
if not bak.exists(): shutil.copy(out, bak); log(f"backed up -> {bak.name}")
pickle.dump(bundle, open(out, "wb"))
log(f"WROTE {out.name}: hedge_tau={best_tau} (trainNet {best_net:+.0f}R), gate untouched")
