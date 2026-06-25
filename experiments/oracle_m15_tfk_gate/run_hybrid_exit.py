"""
Oracle Hybrid Exit sweep — fix the max_hold time-out problem.

Each variant uses the same Oracle setups + per-cluster confirm + M15 TFK ANTI
gate. Only the EXIT differs. Reports duration distribution + PF/WR/DD.

Variants:
  A) RL-only          (current Oracle: hard_sl @ -5R, RL prob >= 0.55, max_hold=60)
  B) Hermes-trail     (hard_sl @ -SL=6, trail=2, max_hold=60)
  C) Hybrid (whichever-first)
        - hard SL (-5R) immediate
        - trail (TRAIL=2 ATR after MFE >= 2 ATR)
        - RL signal (prob >= EXIT_THR, only after profit)
        - max_hold fallback
  D) Hybrid + lower RL threshold (0.40 instead of 0.55)
  E) Hybrid + tighter trail (TRAIL=1.2)
  F) Hybrid + shorter MAX_HOLD (30 bars = 2.5h)
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
MIN_HOLD_DEF = 3
SL_HARD = 5.0
HOLDOUT = pd.Timestamp("2024-01-01")


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
    m15 = pd.DataFrame({"open": s["open"].resample("15min").first(),
                        "high": s["high"].resample("15min").max(),
                        "low":  s["low"].resample("15min").min(),
                        "close":s["close"].resample("15min").last(),
                        "tick_volume":s["tick_volume"].resample("15min").sum(),
                        }).dropna(subset=["close"]).reset_index()
    tfk_m15 = compute_tfk(m15, flip_bars=5, color_confirm=8)
    m15["m15_dir"] = tfk_m15["committed_dir"].to_numpy()
    aligned = pd.merge_asof(df_bars[["time"]].sort_values("time"),
                            m15[["time","m15_dir"]].assign(
                              time=m15["time"]+pd.Timedelta("15min")).sort_values("time"),
                            on="time", direction="backward")
    return aligned["m15_dir"].fillna(0).to_numpy(np.int64)


def precompute_rl_probs(sw_idxs, dirs, swing_df, exit_mdl, exit_feats, MAX_HOLD, MIN_HOLD):
    """Pre-compute RL exit probabilities for all (trade, k) pairs."""
    C = swing_df["close"].to_numpy(float)
    n = len(swing_df)
    ctx_target = list(exit_feats[3:])
    ctx_arr = np.zeros((n, len(ctx_target)), dtype=np.float64)
    for j, c in enumerate(ctx_target):
        if c in swing_df.columns: ctx_arr[:, j] = swing_df[c].fillna(0).to_numpy(float)
    N = len(sw_idxs)
    nf = 3 + ctx_arr.shape[1]
    Xs = np.zeros((N*MAX_HOLD, nf), dtype=np.float32)
    valid = np.zeros(N*MAX_HOLD, dtype=bool)
    for rank in range(N):
        ei = sw_idxs[rank]; d = dirs[rank]; ep = C[ei]
        prev_c = swing_df["close"].to_numpy(float)[ei]
        for k in range(1, MAX_HOLD+1):
            bar = ei + k
            if bar >= n: break
            cp = d*(C[bar]-ep) / max(1e-9, _atr_at(swing_df, ei))
            if k < MIN_HOLD: continue
            p3 = d*(C[bar-3]-ep)/max(1e-9, _atr_at(swing_df, ei)) if k >= 3 else cp
            row = rank*MAX_HOLD + (k-1)
            Xs[row, 0] = cp; Xs[row, 1] = float(k); Xs[row, 2] = cp - p3
            if ctx_arr.shape[1] > 0: Xs[row, 3:] = ctx_arr[bar]
            valid[row] = True
    probs = np.zeros(N*MAX_HOLD, dtype=np.float32)
    if valid.any():
        probs[valid] = exit_mdl.predict_proba(Xs[valid])[:, 1]
    return probs.reshape(N, MAX_HOLD)


_atr_cache = {}
def _atr_at(swing_df, idx):
    # Inefficient — compute once
    if "atr14_cached" not in _atr_cache:
        H=swing_df["high"].to_numpy(float); L=swing_df["low"].to_numpy(float); C=swing_df["close"].to_numpy(float)
        prev_c = np.concatenate([[C[0]], C[:-1]])
        tr = np.maximum(H-L, np.maximum(np.abs(H-prev_c), np.abs(L-prev_c)))
        atr = pd.Series(tr).rolling(14, min_periods=14).mean().to_numpy()
        _atr_cache["atr14_cached"] = atr
    return _atr_cache["atr14_cached"][idx]


@njit(cache=True)
def sim_hybrid(sw_idxs, dirs, O, H, L, C, atr, probs_2d, sp,
               use_hard_sl, hard_sl_R,
               use_trail, trail_atr,
               use_rl, rl_thr, rl_min_pnl_R,
               max_hold):
    N = len(sw_idxs)
    pnl = np.zeros(N); bars = np.zeros(N, dtype=np.int64); reason = np.zeros(N, dtype=np.int64)
    # reason codes: 0=hard_sl, 1=trail, 2=rl_exit, 3=max_hold
    for rank in range(N):
        ei = sw_idxs[rank]; d = dirs[rank]; a = atr[ei]; ep = C[ei]
        if not (a > 0): bars[rank]=0; reason[rank]=3; continue
        mf = 0.0; xi = None; xr = 3
        for k in range(1, max_hold+1):
            bar = ei + k
            if bar >= len(C): break
            fav = d*(C[bar]-ep)
            if fav > mf: mf = fav
            # Hard SL
            if use_hard_sl:
                if d == 1 and (ep - L[bar]) >= hard_sl_R*a: xi=bar; xr=0; break
                if d == -1 and (H[bar] - ep) >= hard_sl_R*a: xi=bar; xr=0; break
            # Trail (Hermes-style): once MFE >= trail_atr*ATR, exit on trail_atr retrace
            if use_trail and mf >= trail_atr*a and (mf - fav) >= trail_atr*a:
                xi = bar; xr = 1; break
            # RL exit (only after MIN_HOLD and only if already profitable enough)
            if use_rl and k >= 3:
                cp_R = fav / a
                if cp_R >= rl_min_pnl_R and probs_2d[rank, k-1] >= rl_thr:
                    xi = bar; xr = 2; break
        if xi is None:
            xi = min(ei + max_hold, len(C)-1); xr = 3
        pnl[rank] = d*(C[xi]-ep)/a - sp/a
        bars[rank] = xi - ei
        reason[rank] = xr
    return pnl, bars, reason


def main():
    t0 = time.time()
    print("="*80); print("  Oracle HYBRID EXIT sweep — fix the max_hold time-out problem"); print("="*80)
    bundle = pickle.load(open(ORACLE_PKL, "rb"))
    mdls = bundle["mdls"]; thrs = bundle["thrs"]
    exit_mdl = bundle["exit_mdl"]; exit_feats = bundle["exit_feats"]
    v72l_feats = bundle["v72l_feats"]

    sw = pd.read_csv(SWING, parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    setup_files = sorted(glob.glob(SETUP_GLOB))
    setups = pd.concat([pd.read_csv(f, parse_dates=["time"]) for f in setup_files],
                       ignore_index=True).sort_values("time").reset_index(drop=True)
    fp = pd.read_csv(FINGERPRINTS, parse_dates=["center_time"])
    available_ctx = [c for c in exit_feats[3:] if c in setups.columns]
    if available_ctx:
        ctx_df = setups[["time"]+available_ctx].sort_values("time").drop_duplicates("time")
        merged = pd.merge_asof(sw[["time"]].sort_values("time"),
                               ctx_df.sort_values("time"), on="time", direction="backward")
        for c in available_ctx:
            sw[c] = merged[c].fillna(0).to_numpy()
    for c in [c for c in exit_feats[3:] if c not in available_ctx]:
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

    # confirm
    setups["confirm_ok"] = False
    for cid in sorted(setups["cid"].unique()):
        key = (int(cid), "RL")
        if key not in mdls: continue
        sub = setups[setups["cid"] == cid]
        X = sub[v72l_feats].fillna(0).to_numpy(np.float32)
        p = mdls[key].predict_proba(X)[:, 1]
        setups.loc[sub.index, "confirm_ok"] = p >= thrs[key]
    confirmed = setups[setups["confirm_ok"] & (setups["sw_idx"] >= 0)].reset_index(drop=True)

    gated = confirmed[(confirmed["m15_dir"] == -confirmed["direction"]) &
                      (confirmed["m15_dir"] != 0)].reset_index(drop=True)
    print(f"  gated trades (PURE M15 ANTI): {len(gated):,}", flush=True)

    # Pre-compute ATR
    H=sw["high"].to_numpy(float); L=sw["low"].to_numpy(float); C=sw["close"].to_numpy(float); O=sw["open"].to_numpy(float)
    prev_c = np.concatenate([[C[0]], C[:-1]])
    tr = np.maximum(H-L, np.maximum(np.abs(H-prev_c), np.abs(L-prev_c)))
    atr = pd.Series(tr).rolling(14, min_periods=14).mean().to_numpy()
    sp = SPREAD_USD / np.nanmedian(atr)

    sw_idxs = gated["sw_idx"].to_numpy(np.int64)
    dirs = gated["direction"].to_numpy(np.int64)

    # Pre-compute RL probs for max horizon (60 bars; smaller variants will just ignore tail)
    MAX_HOLD_MAX = 60
    print(f"  pre-computing RL exit probabilities for {len(sw_idxs):,} trades × {MAX_HOLD_MAX} bars ...", flush=True)
    probs = precompute_rl_probs(sw_idxs, dirs, sw, exit_mdl, exit_feats, MAX_HOLD_MAX, MIN_HOLD_DEF)
    print(f"    probs shape: {probs.shape}  mean={probs.mean():.3f}  frac>=0.55={float((probs>=0.55).mean()):.3f}  frac>=0.40={float((probs>=0.40).mean()):.3f}", flush=True)

    REASON_NAMES = {0:"hard_sl", 1:"trail", 2:"rl_exit", 3:"max_hold"}

    def run_variant(name, use_hard_sl, hard_sl_R, use_trail, trail_atr,
                    use_rl, rl_thr, rl_min_pnl_R, max_hold):
        pnl, bars, reasons = sim_hybrid(sw_idxs, dirs, O, H, L, C, atr, probs, sp,
                                         use_hard_sl, hard_sl_R,
                                         use_trail, trail_atr,
                                         use_rl, rl_thr, rl_min_pnl_R,
                                         max_hold)
        # Filter to holdout
        hd_mask = gated["time"].values >= np.datetime64(HOLDOUT)
        pnl_h = pnl[hd_mask]; bars_h = bars[hd_mask]; reasons_h = reasons[hd_mask]
        m = metrics(pnl_h)
        if m is None:
            print(f"  {name:<32}: no trades"); return
        # exit-reason mix
        rcounts = {REASON_NAMES[k]: int((reasons_h==k).sum()) for k in REASON_NAMES}
        total = m['n']
        print(f"\n  {name}")
        print(f"    HOLDOUT: n={m['n']:>5,}  WR={m['wr']*100:>5.1f}%  PF={m['pf']:>5.2f}  "
              f"sumR={m['sum_r']:>+7.0f}  DD={m['max_dd_r']:>5.0f}  avgR={m['avg']:>+5.2f}  "
              f"avg_bars={float(bars_h.mean()):>4.1f} ({float(bars_h.mean())*5:>3.0f} min)")
        print(f"    exit-mix: " + "  ".join([f"{k}={v}({v/total*100:.0f}%)" for k,v in rcounts.items()]))

    print("\n  ============ EXIT VARIANT SWEEP ============")
    # A) RL only (current Oracle baseline)
    run_variant("A) RL-only (current Oracle baseline)",
                use_hard_sl=True,  hard_sl_R=SL_HARD,
                use_trail=False,   trail_atr=0,
                use_rl=True,       rl_thr=0.55, rl_min_pnl_R=-1e9,
                max_hold=60)
    # B) Hermes trail only
    run_variant("B) Hermes trail only (SL=6, TRAIL=2)",
                use_hard_sl=True,  hard_sl_R=6.0,
                use_trail=True,    trail_atr=2.0,
                use_rl=False,      rl_thr=1.0, rl_min_pnl_R=1e9,
                max_hold=60)
    # C) Hybrid: trail + RL (whichever first)
    run_variant("C) HYBRID: trail(2)+RL(0.55) after +0.5R",
                use_hard_sl=True,  hard_sl_R=SL_HARD,
                use_trail=True,    trail_atr=2.0,
                use_rl=True,       rl_thr=0.55, rl_min_pnl_R=0.5,
                max_hold=60)
    # D) Hybrid with lower RL threshold
    run_variant("D) HYBRID: trail(2)+RL(0.40) after +0.5R",
                use_hard_sl=True,  hard_sl_R=SL_HARD,
                use_trail=True,    trail_atr=2.0,
                use_rl=True,       rl_thr=0.40, rl_min_pnl_R=0.5,
                max_hold=60)
    # E) Tighter trail
    run_variant("E) HYBRID: trail(1.2)+RL(0.55) after +0.5R",
                use_hard_sl=True,  hard_sl_R=SL_HARD,
                use_trail=True,    trail_atr=1.2,
                use_rl=True,       rl_thr=0.55, rl_min_pnl_R=0.5,
                max_hold=60)
    # F) Shorter MAX_HOLD
    run_variant("F) HYBRID: trail(2)+RL(0.55), MAX_HOLD=30 (2.5h)",
                use_hard_sl=True,  hard_sl_R=SL_HARD,
                use_trail=True,    trail_atr=2.0,
                use_rl=True,       rl_thr=0.55, rl_min_pnl_R=0.5,
                max_hold=30)
    # G) Trail-only with shorter hold
    run_variant("G) Trail(2) only, MAX_HOLD=30",
                use_hard_sl=True,  hard_sl_R=6.0,
                use_trail=True,    trail_atr=2.0,
                use_rl=False,      rl_thr=1.0, rl_min_pnl_R=1e9,
                max_hold=30)
    # H) Hybrid + lower RL after +0.3R
    run_variant("H) HYBRID: trail(2)+RL(0.40) after +0.3R, MAX_HOLD=30",
                use_hard_sl=True,  hard_sl_R=SL_HARD,
                use_trail=True,    trail_atr=2.0,
                use_rl=True,       rl_thr=0.40, rl_min_pnl_R=0.3,
                max_hold=30)
    print(f"\n  TOTAL: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
