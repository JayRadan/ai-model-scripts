"""
Train a smarter Oracle exit: future-upside regressor.

Why the current XGBClassifier exit fails:
  - Binary "exit/hold" target — sharp decision boundary, fires only 20% of time
  - Result: 58% of trades hit max_hold time-cap

New approach:
  - For each (trade, bar k) state, label = max future favorable R from k → end-of-trade
    minus current unrealized R. This is "how much more upside is left if I hold."
  - Train XGBRegressor to predict that scalar.
  - Exit rule: HOLD if predicted_upside >= θ_R (e.g., 0.3 R). Else EXIT.
  - Plus hard SL at -5R, plus trail-stop floor for sanity, plus MAX_HOLD fallback.

Features (richer than current bundle):
  - unrealized_R, bars_held, pnl_velocity_3, pnl_velocity_5
  - MFE_so_far_R, MAE_so_far_R, drawdown_from_peak_R
  - ATR_normalized progress: cp / sl_dist, cp / trail_dist
  - hour, dow, m15_dir alignment to trade
  - cluster ID (one-hot)
  - context: hurst, ou_theta, entropy_rate, kramers_up, wavelet_er, quantum_flow, vwap_dist

Train: pre-2024.  Test: 2024+ holdout.
Compare to existing RL exit head-to-head.
"""
from __future__ import annotations
import sys, time, glob, pickle
from pathlib import Path
import numpy as np, pandas as pd
from numba import njit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "products/hermes_xau"))
from tfk import compute_tfk

SWING = ROOT / "data" / "swing_v5_xauusd.csv"
SETUP_GLOB = str(ROOT / "data" / "setups_*_v72l.csv")
ORACLE_PKL = ROOT / "products" / "models" / "oracle_xau_validated.pkl"
FINGERPRINTS = ROOT / "products" / "_shared" / "data" / "regime_fingerprints_4h.csv"

SPREAD_USD = 0.30
MIN_HOLD = 3
MAX_HOLD = 60
SL_HARD = 5.0
HOLDOUT = pd.Timestamp("2024-01-01")
TRAIL_FLOOR = 2.0  # safety: also trail-stop after MFE>=2R, 2R retrace


def metrics(r):
    r = np.asarray(r); r = r[np.isfinite(r)]
    if len(r) == 0: return None
    w, l = r[r>0], r[r<=0]; eq = np.cumsum(r)
    return dict(n=int(len(r)), wr=float((r>0).mean()),
                pf=float(w.sum()/max(-l.sum(),1e-9)),
                sum_r=float(r.sum()),
                max_dd_r=float((np.maximum.accumulate(eq)-eq).max()),
                avg=float(r.mean()))


def m15_dir(df_bars):
    s = df_bars.set_index("time")
    m15 = pd.DataFrame({"open":s["open"].resample("15min").first(),
                        "high":s["high"].resample("15min").max(),
                        "low": s["low"].resample("15min").min(),
                        "close":s["close"].resample("15min").last(),
                        "tick_volume":s["tick_volume"].resample("15min").sum()
                        }).dropna(subset=["close"]).reset_index()
    tfk_m15 = compute_tfk(m15, flip_bars=5, color_confirm=8)
    m15["m15_dir"] = tfk_m15["committed_dir"].to_numpy()
    aligned = pd.merge_asof(df_bars[["time"]].sort_values("time"),
                            m15[["time","m15_dir"]].assign(
                              time=m15["time"]+pd.Timedelta("15min")).sort_values("time"),
                            on="time", direction="backward")
    return aligned["m15_dir"].fillna(0).to_numpy(np.int64)


@njit(cache=True)
def build_state_table(sw_idxs, dirs, cids, hours, dows, m15_dirs,
                      ctx_arr,
                      C, H, L, atr, max_hold):
    """For each (trade, k) generate features + label = future_upside_R."""
    N = len(sw_idxs); nf = ctx_arr.shape[1]
    # 17 trade-state features + nf context features + 1 cid
    total_feats = 18 + nf
    rows_per_trade = max_hold
    Xs = np.zeros((N*rows_per_trade, total_feats), dtype=np.float32)
    ys = np.zeros(N*rows_per_trade, dtype=np.float32)
    valid = np.zeros(N*rows_per_trade, dtype=np.uint8)
    for rank in range(N):
        ei = sw_idxs[rank]; d = dirs[rank]; a = atr[ei]; ep = C[ei]
        if not (a > 0): continue
        # Pre-compute cp trajectory for this trade
        cps = np.zeros(max_hold)
        valid_k = 0
        mfe = 0.0; mae = 0.0
        for k in range(1, max_hold+1):
            bar = ei + k
            if bar >= len(C): break
            fav = d*(C[bar]-ep)
            cps[k-1] = fav / a
            if fav > mfe*a: mfe = fav/a
            if -fav/a > mae: mae = -fav/a
            valid_k = k
        # Build features for each k in [MIN_HOLD, valid_k]
        for k in range(3, valid_k+1):
            bar = ei + k
            cp = cps[k-1]
            p3 = cps[k-4] if k >= 4 else cp
            p5 = cps[k-6] if k >= 6 else cp
            # Compute MFE and MAE up to bar k (running)
            mfe_k = 0.0; mae_k = 0.0
            for kk in range(k):
                v = cps[kk]
                if v > mfe_k: mfe_k = v
                if -v > mae_k: mae_k = -v
            dd_from_peak = mfe_k - cp  # how far we've retraced from MFE
            # Future upside: max future favorable R from k+1..valid_k minus current cp
            future_max = cp
            for kk in range(k, valid_k):
                if cps[kk] > future_max: future_max = cps[kk]
            future_upside = future_max - cp
            row = rank*rows_per_trade + (k-1)
            # Features
            Xs[row, 0] = cp                                    # unrealized R
            Xs[row, 1] = float(k)                              # bars held
            Xs[row, 2] = cp - p3                               # vel 3-bar
            Xs[row, 3] = cp - p5                               # vel 5-bar
            Xs[row, 4] = mfe_k                                 # MFE so far
            Xs[row, 5] = mae_k                                 # MAE so far
            Xs[row, 6] = dd_from_peak                          # retrace from peak
            Xs[row, 7] = cp / 5.0                              # progress vs hard SL
            Xs[row, 8] = cp / 2.0                              # progress vs trail
            Xs[row, 9] = float(hours[bar])                     # hour at current bar
            Xs[row, 10] = float(dows[bar])                     # dow
            Xs[row, 11] = float(m15_dirs[bar] * d)             # M15 aligned with trade dir? (-1/0/+1)
            Xs[row, 12] = float(cids[rank])                    # cluster id as numeric
            Xs[row, 13] = float(d)                             # trade direction
            Xs[row, 14] = float(max_hold - k)                  # bars remaining
            Xs[row, 15] = float(k) / float(max_hold)           # fractional time
            Xs[row, 16] = mfe_k - mae_k                        # range
            Xs[row, 17] = float(valid_k)                       # available horizon
            for j in range(nf):
                Xs[row, 18+j] = float(ctx_arr[bar, j])
            ys[row] = future_upside
            valid[row] = 1
    return Xs, ys, valid


@njit(cache=True)
def sim_smart_exit(sw_idxs, dirs, cids, hours, dows, m15_dirs, ctx_arr,
                   O, H, L, C, atr,
                   sl_hard_R, mfe_trigger_R, trail_retrace_R, upside_thr_R,
                   use_smart, use_trail,
                   pred_per_state,
                   max_hold, sp_R, ratchet_step, be_lock_R, be_floor_R):
    N = len(sw_idxs)
    pnl = np.zeros(N); bars = np.zeros(N, dtype=np.int64); reason = np.zeros(N, dtype=np.int64)
    # 0=hard_sl, 1=trail/ratchet, 2=smart_exit (upside below thr), 3=max_hold
    for rank in range(N):
        ei = sw_idxs[rank]; d = dirs[rank]; a = atr[ei]; ep = C[ei]
        if not (a > 0): bars[rank]=0; reason[rank]=3; continue
        mf = 0.0; xi = -1; xr = 3; peak_cp = 0.0; floor = -sl_hard_R
        for k in range(1, max_hold+1):
            bar = ei + k
            if bar >= len(C): break
            fav = d*(C[bar]-ep)
            if fav > mf: mf = fav
            cp = fav / a
            # Hard SL (intrabar broker stop)
            if d == 1 and (ep - L[bar]) >= sl_hard_R*a: xi=bar; xr=0; break
            if d == -1 and (H[bar] - ep) >= sl_hard_R*a: xi=bar; xr=0; break
            # RATCHET: close-based floor that only moves up; once +N*step, locks +(N-1)*step
            if ratchet_step > 0.0:
                if cp > peak_cp: peak_cp = cp
                lvl = (np.floor(peak_cp / ratchet_step) - 1.0) * ratchet_step
                if lvl > floor: floor = lvl
                if cp <= floor: xi=bar; xr=1; break
            # BE-LOCK: once peak >= be_lock_R, floor jumps to be_floor_R and stays (upside uncapped)
            if be_lock_R > 0.0:
                if cp > peak_cp: peak_cp = cp
                if peak_cp >= be_lock_R and floor < be_floor_R: floor = be_floor_R
                if floor > -sl_hard_R and cp <= floor: xi=bar; xr=1; break
            # Trail: arm only after MFE >= mfe_trigger_R, fire on trail_retrace_R retrace
            if use_trail and mf >= mfe_trigger_R*a and (mf - fav) >= trail_retrace_R*a:
                xi = bar; xr = 1; break
            # Smart exit
            if use_smart and k >= 3 and cp >= 0.3:
                if pred_per_state[rank, k-1] < upside_thr_R:
                    xi = bar; xr = 2; break
        if xi < 0:
            xi = min(ei + max_hold, len(C)-1); xr = 3
        pnl[rank] = d*(C[xi]-ep)/a - sp_R
        bars[rank] = xi - ei
        reason[rank] = xr
    return pnl, bars, reason


def main():
    t0 = time.time()
    print("="*80); print("  Training smarter Oracle exit: future-upside regressor"); print("="*80)
    bundle = pickle.load(open(ORACLE_PKL, "rb"))
    mdls = bundle["mdls"]; thrs = bundle["thrs"]
    exit_mdl_orig = bundle["exit_mdl"]; exit_feats_orig = bundle["exit_feats"]
    v72l_feats = bundle["v72l_feats"]

    sw = pd.read_csv(SWING, parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    setup_files = sorted(glob.glob(SETUP_GLOB))
    setups = pd.concat([pd.read_csv(f, parse_dates=["time"]) for f in setup_files],
                       ignore_index=True).sort_values("time").reset_index(drop=True)
    fp = pd.read_csv(FINGERPRINTS, parse_dates=["center_time"])
    available_ctx = [c for c in exit_feats_orig[3:] if c in setups.columns]
    if available_ctx:
        ctx_df = setups[["time"]+available_ctx].sort_values("time").drop_duplicates("time")
        merged = pd.merge_asof(sw[["time"]].sort_values("time"),
                               ctx_df.sort_values("time"), on="time", direction="backward")
        for c in available_ctx:
            sw[c] = merged[c].fillna(0).to_numpy()
    for c in [c for c in exit_feats_orig[3:] if c not in available_ctx]:
        sw[c] = 0.0

    times_sw = sw["time"].values.astype("datetime64[ns]")
    setup_t = setups["time"].values.astype("datetime64[ns]")
    sw_idx = np.searchsorted(times_sw, setup_t)
    sw_idx_safe = np.minimum(sw_idx, len(times_sw)-1)
    exact = times_sw[sw_idx_safe] == setup_t
    setups["sw_idx"] = np.where(exact, sw_idx_safe, -1)
    cid_per = np.full(len(sw), -1, dtype=np.int64)
    for _, row in fp.iterrows():
        s, e = int(row["start_idx"]), int(row["end_idx"])
        if 0 <= s < e <= len(sw): cid_per[s:e] = int(row["new_label"])
    setups["cid"] = np.where(setups["sw_idx"] >= 0, cid_per[np.maximum(setups["sw_idx"],0)], -1)
    setups = setups[setups["cid"] >= 0].reset_index(drop=True)

    print("  computing M15 TFK ...", flush=True)
    m15_arr = m15_dir(sw)
    sw_idx_safe = np.minimum(np.searchsorted(times_sw, setups["time"].values.astype("datetime64[ns]")), len(times_sw)-1)
    exact = times_sw[sw_idx_safe] == setups["time"].values.astype("datetime64[ns]")
    setups["m15_dir"] = np.where(exact, m15_arr[sw_idx_safe], 0)

    # per-cluster confirm
    setups["confirm_ok"] = False
    for cid in sorted(setups["cid"].unique()):
        key = (int(cid), "RL")
        if key not in mdls: continue
        sub = setups[setups["cid"] == cid]
        X = sub[v72l_feats].fillna(0).to_numpy(np.float32)
        p = mdls[key].predict_proba(X)[:, 1]
        setups.loc[sub.index, "confirm_ok"] = p >= thrs[key]
    confirmed = setups[setups["confirm_ok"] & (setups["sw_idx"] >= 0)].reset_index(drop=True)

    # M15 ANTI gate
    gated = confirmed[(confirmed["m15_dir"] == -confirmed["direction"]) &
                      (confirmed["m15_dir"] != 0)].reset_index(drop=True)
    print(f"  gated trades: {len(gated):,}", flush=True)

    O = sw["open"].to_numpy(float); H = sw["high"].to_numpy(float)
    L = sw["low"].to_numpy(float);  C = sw["close"].to_numpy(float)
    prev_c = np.concatenate([[C[0]], C[:-1]])
    tr = np.maximum(H-L, np.maximum(np.abs(H-prev_c), np.abs(L-prev_c)))
    atr = pd.Series(tr).rolling(14, min_periods=14).mean().to_numpy()
    sp_R = SPREAD_USD / np.nanmedian(atr)

    hours = pd.to_datetime(sw["time"]).dt.hour.to_numpy()
    dows = pd.to_datetime(sw["time"]).dt.dayofweek.to_numpy()

    # Context features per swing bar (forward-fill from setups)
    ctx_cols = ["hurst_rs","ou_theta","entropy_rate","kramers_up","wavelet_er",
                "quantum_flow","quantum_flow_h4","vwap_dist"]
    ctx_arr = np.zeros((len(sw), len(ctx_cols)), dtype=np.float64)
    for j, c in enumerate(ctx_cols):
        if c in sw.columns: ctx_arr[:, j] = sw[c].fillna(0).to_numpy(float)

    sw_idxs = gated["sw_idx"].to_numpy(np.int64)
    dirs = gated["direction"].to_numpy(np.int64)
    cids = gated["cid"].to_numpy(np.int64)

    # Build state table for ALL gated trades
    print(f"  building state table for {len(sw_idxs):,} trades × {MAX_HOLD} bars ...", flush=True)
    ts = time.time()
    Xs, ys, valid = build_state_table(sw_idxs, dirs, cids, hours, dows, m15_arr,
                                       ctx_arr, C, H, L, atr, MAX_HOLD)
    print(f"    built in {time.time()-ts:.0f}s  valid rows={int(valid.sum()):,}", flush=True)

    # Train/test split BY TRADE (so all bars of a trade go together)
    trade_times = gated["time"].values
    trade_is_train = trade_times < np.datetime64(HOLDOUT)
    # Row-wise mask
    row_trade_idx = np.repeat(np.arange(len(sw_idxs)), MAX_HOLD)
    row_is_train = trade_is_train[row_trade_idx]
    valid_bool = valid.astype(bool)
    train_rows = valid_bool & row_is_train
    test_rows  = valid_bool & (~row_is_train)
    print(f"    train rows: {int(train_rows.sum()):,}  test rows: {int(test_rows.sum()):,}", flush=True)

    # Fit XGBRegressor on future_upside
    from xgboost import XGBRegressor
    print("  training XGBRegressor (predict future_upside_R) ...", flush=True)
    ts = time.time()
    M = XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.04,
                     subsample=0.85, colsample_bytree=0.85, min_child_weight=10,
                     reg_lambda=1.0, objective="reg:squarederror",
                     tree_method="hist", random_state=42, verbosity=0, n_jobs=-1)
    M.fit(Xs[train_rows], ys[train_rows])
    print(f"    fit in {time.time()-ts:.0f}s", flush=True)

    # Predict for ALL (trade, k) — both train and test rows.
    # At sim time we use these predictions to decide exits.
    print("  predicting ...", flush=True)
    preds = np.zeros(len(Xs), dtype=np.float32)
    preds[valid_bool] = M.predict(Xs[valid_bool])
    pred_per_state = preds.reshape(len(sw_idxs), MAX_HOLD)

    # Sweep variants
    print(f"\n  ============ EXIT VARIANT SWEEP (2.4y holdout) ============")
    print(f"   {'variant':>40} {'n':>5} {'WR%':>5} {'PF':>5} {'sumR':>7} {'DD':>5} {'avg_bars':>9} {'avg_min':>7}   {'mix':>30}")
    REASON_NAMES = {0:"hard_sl", 1:"trail", 2:"smart", 3:"max_hold"}
    hd_mask = gated["time"].values >= np.datetime64(HOLDOUT)
    def run(name, mfe_trig, trail_retr, upside_thr, use_smart, use_trail, ratchet_step=0.0,
            be_lock_R=0.0, be_floor_R=0.0):
        pnl, bars, reasons = sim_smart_exit(sw_idxs, dirs, cids, hours, dows, m15_arr, ctx_arr,
                                             O, H, L, C, atr,
                                             SL_HARD, float(mfe_trig), float(trail_retr), float(upside_thr),
                                             use_smart, use_trail,
                                             pred_per_state, MAX_HOLD, sp_R, float(ratchet_step),
                                             float(be_lock_R), float(be_floor_R))
        pnl_h = pnl[hd_mask]; bars_h = bars[hd_mask]; reasons_h = reasons[hd_mask]
        m = metrics(pnl_h)
        if m is None: return
        rcounts = {REASON_NAMES[k]: int((reasons_h==k).sum()) for k in REASON_NAMES}
        mix = " ".join([f"{k[:4]}={v/m['n']*100:.0f}%" for k,v in rcounts.items()])
        avg_b = float(bars_h.mean())
        print(f"   {name:>40} {m['n']:>5,} {m['wr']*100:>5.1f} {m['pf']:>5.2f} "
              f"{m['sum_r']:>+7.0f} {m['max_dd_r']:>5.0f} {avg_b:>9.1f} {avg_b*5:>7.0f}   {mix:>30}")
    # 0) original smart (trail=2 trigger, retrace=2)
    run("BASELINE smart trail(MFE2,r2)+up0.3",  2.0, 2.0, 0.3, True,  True)
    # 1) WIDER trail retrace (3)
    run("VAR1 wider trail (MFE2, retrace3)",    2.0, 3.0, 0.3, True,  True)
    # 2) CONDITIONAL trail (arm only after MFE>=3)
    run("VAR2 conditional trail (MFE3, r2)",    3.0, 2.0, 0.3, True,  True)
    # 2b) CONDITIONAL trail later (arm only after MFE>=4)
    run("VAR2b conditional trail (MFE4, r2)",   4.0, 2.0, 0.3, True,  True)
    # === RATCHET variants (profit-lock; "goes in profit, never comes back") ===
    run("RATCHET step2 ALONE (no smart)",       0.0, 0.0, 0.3, False, False, 2.0)
    run("RATCHET step1 ALONE (no smart)",       0.0, 0.0, 0.3, False, False, 1.0)
    run("RATCHET step2 + smart up0.3",          0.0, 0.0, 0.3, True,  False, 2.0)
    run("RATCHET step1 + smart up0.3",          0.0, 0.0, 0.3, True,  False, 1.0)
    run("RATCHET step3 + smart up0.3",          0.0, 0.0, 0.3, True,  False, 3.0)
    # === BE-LOCK variants (smart-exit + 'winner can never become a loss') ===
    run("BE-LOCK @3R->BE + smart",              0.0, 0.0, 0.3, True,  False, 0.0, 3.0, 0.0)
    run("BE-LOCK @4R->BE + smart",              0.0, 0.0, 0.3, True,  False, 0.0, 4.0, 0.0)
    run("BE-LOCK @3R->+1R + smart",             0.0, 0.0, 0.3, True,  False, 0.0, 3.0, 1.0)
    run("BE-LOCK @4R->+2R + smart",             0.0, 0.0, 0.3, True,  False, 0.0, 4.0, 2.0)
    run("BE-LOCK @5R->+2R + smart",             0.0, 0.0, 0.3, True,  False, 0.0, 5.0, 2.0)
    # 3) PURE smart-only (no trail at all)
    run("VAR3 PURE smart, no trail (up0.3)",    0.0, 0.0, 0.3, True,  False)
    run("VAR3 PURE smart, no trail (up0.2)",    0.0, 0.0, 0.2, True,  False)
    run("VAR3 PURE smart, no trail (up0.1)",    0.0, 0.0, 0.1, True,  False)
    run("VAR3 PURE smart, no trail (up0.5)",    0.0, 0.0, 0.5, True,  False)
    # 4) combos
    run("VAR4 wider trail + smart up0.2",       2.0, 3.0, 0.2, True,  True)
    run("VAR4 wider trail + smart up0.5",       2.0, 3.0, 0.5, True,  True)
    run("VAR4 conditional + smart up0.5",       3.0, 2.0, 0.5, True,  True)

    # Baseline (current RL exit) for comparison
    print(f"\n  ============ BASELINE: current RL exit (for comparison) ============")
    # Replicate from earlier: hard_sl @ -5R, RL prob >= 0.55, max_hold=60
    # Use the same probs precomputed via the bundle's exit_mdl
    # For brevity, sim with old logic inline:
    @njit(cache=True)
    def sim_baseline(sw_idxs, dirs, O, H, L, C, atr, probs2d, sp,
                     sl_R, thr, min_hold, max_hold):
        N=len(sw_idxs); pnl=np.zeros(N); bars=np.zeros(N,np.int64); reasons=np.zeros(N,np.int64)
        for rank in range(N):
            ei=sw_idxs[rank]; d=dirs[rank]; a=atr[ei]; ep=C[ei]
            if not (a>0): reasons[rank]=3; continue
            xi=-1; xr=3
            for k in range(1, max_hold+1):
                bar=ei+k
                if bar>=len(C): break
                if d==1 and (ep-L[bar])>=sl_R*a: xi=bar; xr=0; break
                if d==-1 and (H[bar]-ep)>=sl_R*a: xi=bar; xr=0; break
                if k>=min_hold and probs2d[rank,k-1]>=thr: xi=bar; xr=2; break
            if xi < 0: xi=min(ei+max_hold,len(C)-1); xr=3
            pnl[rank]=d*(C[xi]-ep)/a-sp; bars[rank]=xi-ei; reasons[rank]=xr
        return pnl, bars, reasons

    # Need to build RL probs for baseline
    ctx_target = list(exit_feats_orig[3:])
    ctx_orig = np.zeros((len(sw), len(ctx_target)), dtype=np.float64)
    for j, c in enumerate(ctx_target):
        if c in sw.columns: ctx_orig[:, j] = sw[c].fillna(0).to_numpy(float)
    nf = 3 + ctx_orig.shape[1]
    N = len(sw_idxs)
    Xb = np.zeros((N*MAX_HOLD, nf), dtype=np.float32)
    valid_b = np.zeros(N*MAX_HOLD, dtype=np.bool_)
    for rank in range(N):
        ei=sw_idxs[rank]; d=dirs[rank]; ep=C[ei]; ea=atr[ei]
        if not (ea>0): continue
        for k in range(1, MAX_HOLD+1):
            bar=ei+k
            if bar>=len(C): break
            cp=d*(C[bar]-ep)/ea
            if k < MIN_HOLD: continue
            p3=d*(C[bar-3]-ep)/ea if k>=3 else cp
            row=rank*MAX_HOLD+(k-1)
            Xb[row,0]=cp; Xb[row,1]=float(k); Xb[row,2]=cp-p3
            if ctx_orig.shape[1]>0: Xb[row,3:]=ctx_orig[bar]
            valid_b[row]=True
    probs_b = np.zeros(N*MAX_HOLD, dtype=np.float32)
    if valid_b.any():
        probs_b[valid_b] = exit_mdl_orig.predict_proba(Xb[valid_b])[:,1]
    probs_b = probs_b.reshape(N, MAX_HOLD)

    pnl_b, bars_b, reas_b = sim_baseline(sw_idxs, dirs, O, H, L, C, atr, probs_b, sp_R,
                                          SL_HARD, 0.55, MIN_HOLD, MAX_HOLD)
    hd_mask = gated["time"].values >= np.datetime64(HOLDOUT)
    pnl_bh = pnl_b[hd_mask]; bars_bh = bars_b[hd_mask]; reas_bh = reas_b[hd_mask]
    m_b = metrics(pnl_bh)
    rcounts = {REASON_NAMES[k]: int((reas_bh==k).sum()) for k in REASON_NAMES}
    mix = " ".join([f"{k[:4]}={v/m_b['n']*100:.0f}%" for k,v in rcounts.items()])
    print(f"   {'BASELINE RL':>14} {m_b['n']:>5,} {m_b['wr']*100:>5.1f} {m_b['pf']:>5.2f} "
          f"{m_b['sum_r']:>+7.0f} {m_b['max_dd_r']:>5.0f} {float(bars_bh.mean()):>9.1f} "
          f"{float(bars_bh.mean())*5:>7.0f}  {mix:>30}")

    print(f"\n  TOTAL: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
