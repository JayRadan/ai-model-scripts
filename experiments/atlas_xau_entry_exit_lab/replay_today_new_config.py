"""
WHAT WOULD TODAY HAVE BEEN — replay 07-06/07-07 through the config deployed NOW:
  atlas_xau  M1  edge tt30/0.75 (live since 07-02)
  hermes_xau M5  edge tt30/0.75 (deployed this morning)
  oracle_xau M15 edge tt15/0.75 (deployed tonight — replaces the -$377 V7 stack)
  hermes_btc M1  edge tt30/0.75 (deployed this morning)
Exact deployed bundles + thresholds, 1-slot per product, tt exits, spread charged
($0.20/oz XAU -> $1/trade at 0.05 lots; 0.2R BTC). $ at live lot sizes.
"""
import sys, pickle, warnings
from datetime import datetime, timezone
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from numba import njit
SRV = "/home/jay/Desktop/my-agents-and-website/commercial/server"
sys.path.insert(0, SRV)
import decision_engine.edge_pullback as ep
import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_FX_METALS_XAU_USD, INSTRUMENT_VCCY_BTC_USD

def fetch(instr, days=12):
    end = datetime(2026, 7, 8, 0, 0, tzinfo=timezone.utc)
    start = datetime(2026, 6, 25, 0, 0, tzinfo=timezone.utc)
    r = dukascopy_python.fetch(instrument=instr, interval=dukascopy_python.INTERVAL_MIN_1,
                               offer_side=dukascopy_python.OFFER_SIDE_BID, start=start, end=end)
    d = r.reset_index().rename(columns={"timestamp": "time"})
    d["time"] = pd.to_datetime(d["time"]).dt.tz_convert("UTC").dt.tz_localize(None)
    d["tick_volume"] = 1
    return d.sort_values("time").drop_duplicates("time").reset_index(drop=True)

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
def take(order_idx, ebar, xit, cd):
    busy = -1; m = len(order_idx); out = np.empty(m, np.int64); c = 0
    for t in range(m):
        k = order_idx[t]
        if ebar[k] <= busy: continue
        out[c] = k; busy = xit[k] + cd; c += 1
    return out[:c]

XAU = fetch(INSTRUMENT_FX_METALS_XAU_USD); BTC = fetch(INSTRUMENT_VCCY_BTC_USD)
print(f"XAU {len(XAU)} bars -> {XAU.time.iloc[-1]} | BTC {len(BTC)} bars -> {BTC.time.iloc[-1]}", flush=True)

PRODUCTS = [
    ("atlas_xau  M1 ", "atlas_xau",  XAU, 1,  5, 0.05 * 100, 0.20),
    ("hermes_xau M5 ", "hermes_xau", XAU, 5,  3, 0.05 * 100, 0.20),
    ("oracle_xau M15", "oracle_xau", XAU, 15, 3, 0.05 * 100, 0.20),
    ("hermes_btc M1 ", "hermes_btc", BTC, 1,  5, 0.10 * 1,   None),   # None -> 0.2R
]
grand = {}
for label, name, m1, bm, cd, usd_per_unit, spread_usd in PRODUCTS:
    pl = pickle.load(open(f"{SRV}/decision_engine/models/{name}_validated.pkl", "rb"))
    df = ep._resample(m1, bm) if bm > 1 else m1.copy()
    feat = ep.compute_edge_features(df)
    atr = feat["atr14"].to_numpy(float); cdir = feat["committed_dir"].to_numpy(np.int64)
    da = np.abs(feat["dist_at_signal"].to_numpy(float))
    O = df["open"].to_numpy(float); H = df["high"].to_numpy(float); L = df["low"].to_numpy(float); C = df["close"].to_numpy(float)
    n = len(df); times = df["time"].values
    ok = np.isfinite(atr) & (atr > 0) & (cdir != 0); ok[:300] = False
    idx = np.where((da <= 1.0) & ok)[0]; dirs = cdir[idx].astype(np.int64)
    X = np.nan_to_num(feat[pl["feat_cols"]].to_numpy(np.float32)[idx], nan=0.0, posinf=0.0, neginf=0.0)
    maxh = pl["maxh"]; ta = pl.get("tight_after", 0); tt = pl.get("tight_trail_R", 0.0)
    pnl, ebar, xit = sim_tt(idx, dirs, O, H, L, C, atr, n, pl["sl_R"], pl["trail_R"], maxh, ta, tt)
    p = pl["q_model"].predict(X)
    kk = np.where(p >= pl["threshold"])[0]
    tk = take(kk[np.argsort(ebar[kk])].astype(np.int64), ebar, xit, cd)
    # keep trades whose ENTRY is on 07-06 or 07-07
    et = pd.to_datetime(times[ebar[tk]])
    sel = tk[(et >= "2026-07-06") & (et < "2026-07-08")]
    R = pnl[sel]; a_ = atr[idx[sel]]
    usd = R * a_ * usd_per_unit
    if spread_usd is not None: usd = usd - spread_usd * usd_per_unit
    else: usd = (R - 0.2) * a_ * usd_per_unit
    days = pd.to_datetime(times[ebar[sel]]).date
    out = pd.DataFrame({"day": days, "usd": usd})
    daily = out.groupby("day")["usd"].agg(["count", "sum"]).round(2)
    print(f"\n[{label}] thr={pl['threshold']:.3f}  trades 07-06/07: {len(sel)}")
    print(daily.to_string())
    grand[label] = out

tot = pd.concat(grand.values())
print("\n===== NEW-CONFIG TOTAL (07-06 + 07-07, XAU+BTC streams) =====")
print(tot.groupby("day")["usd"].agg(["count", "sum"]).round(2).to_string())
print(f"\nTOTAL: {tot['usd'].sum():+.2f} USD  ({len(tot)} trades)")
print("(actual live 07-07: -$740 — of which V7/oracle -$377, old-bundle products included)")
