"""
Oracle XAU with REAL deployed RL exit + M15 TFK gate sweep.

Runs Oracle setups through the production exit policy (the deployed
XGBClassifier from oracle_xau_validated.pkl) and tests the M15 TFK gate
on three timeframes:

  - M5 (Oracle native, swing data)
  - M1 (Hermes-style candidates on m1_xau_full.parquet)
  - M15 (resample M5 → M15 entries)

For each, sweep: NO gate / M15-ALIGN gate / M15-ANTI gate.
Report last week, last 30/90d, full holdout.
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
M1_PARQUET = ROOT / "data" / "m1_xau_full.parquet"
SETUP_GLOB = str(ROOT / "data" / "setups_*_v72l.csv")
ORACLE_PKL = ROOT / "products" / "models" / "oracle_xau_validated.pkl"
SPREAD_USD = 0.30

# Oracle exit hyperparams (from deploy_bundle.py)
MIN_HOLD = 3
MAX_HOLD = 60
SL_HARD = 5.0
EXIT_THRESHOLD = 0.55     # default from deploy bundle


def metrics(r):
    r=np.asarray(r); r=r[np.isfinite(r)]
    if len(r)==0: return None
    w,l=r[r>0],r[r<=0]; eq=np.cumsum(r)
    return dict(n=int(len(r)),wr=float((r>0).mean()),
                pf=float(w.sum()/max(-l.sum(),1e-9)),
                sum_r=float(r.sum()),
                max_dd_r=float((np.maximum.accumulate(eq)-eq).max()))


def m15_dir_at_bars(df_bars):
    """Causal M15 TFK direction for each bar in df_bars.
    df_bars must have time, open, high, low, close, tick_volume."""
    s = df_bars.set_index("time")
    m15 = pd.DataFrame({
        "open":  s["open"].resample("15min").first(),
        "high":  s["high"].resample("15min").max(),
        "low":   s["low"].resample("15min").min(),
        "close": s["close"].resample("15min").last(),
        "tick_volume": s["tick_volume"].resample("15min").sum(),
    }).dropna(subset=["close"]).reset_index()
    tfk_m15 = compute_tfk(m15, flip_bars=5, color_confirm=8)
    m15["m15_dir"] = tfk_m15["committed_dir"].to_numpy()
    # shift to "AFTER close" so it's causal at the next bar
    aligned = pd.merge_asof(
        df_bars[["time"]].sort_values("time"),
        m15[["time","m15_dir"]].assign(time=m15["time"] + pd.Timedelta("15min")).sort_values("time"),
        on="time", direction="backward"
    )
    return aligned["m15_dir"].fillna(0).to_numpy(np.int64)


def sim_oracle_rl_exit(setups, swing_df, exit_mdl, exit_feats):
    """Replicate Oracle's exit logic from deploy_bundle.simulate."""
    times_sw = swing_df["time"].values.astype("datetime64[ns]")
    C = swing_df["close"].to_numpy(float)
    H = swing_df["high"].to_numpy(float)
    L = swing_df["low"].to_numpy(float)
    n = len(swing_df)
    # ATR (M5)
    prev_c = np.concatenate([[C[0]], C[:-1]])
    tr = np.maximum(H-L, np.maximum(np.abs(H-prev_c), np.abs(L-prev_c)))
    atr = pd.Series(tr).rolling(14, min_periods=14).mean().to_numpy()
    # Pad with zeros for any missing context feature, in the exact order the
    # model expects (exit_feats[3:]). These are normally live-computed by the
    # server; here we approximate with zeros (gives a context-free RL exit).
    ctx_cols_target = list(exit_feats[3:])
    ctx_arr = np.zeros((n, len(ctx_cols_target)), dtype=np.float64)
    for j, c in enumerate(ctx_cols_target):
        if c in swing_df.columns:
            ctx_arr[:, j] = swing_df[c].fillna(0).to_numpy(float)
    # Align setups to swing index
    setup_t = setups["time"].values.astype("datetime64[ns]")
    sw_idx = np.searchsorted(times_sw, setup_t)
    sw_idx_safe = np.minimum(sw_idx, n-1)
    exact = times_sw[sw_idx_safe] == setup_t
    valid = exact & (atr[sw_idx_safe] > 0) & np.isfinite(atr[sw_idx_safe])
    sub = setups[valid].reset_index(drop=True)
    sw_idx_sub = sw_idx_safe[valid]
    dirs = sub["direction"].to_numpy(np.int64)
    N = len(sub)
    if N == 0:
        return pd.DataFrame()
    # Build feature matrix for exit predictions at each (entry, k)
    nf = 3 + ctx_arr.shape[1]
    Xs = np.zeros((N*MAX_HOLD, nf), dtype=np.float32)
    cps = np.full((N, MAX_HOLD), np.nan, dtype=np.float64)
    valid_rows = np.zeros(N*MAX_HOLD, dtype=bool)
    for rank in range(N):
        ei = sw_idx_sub[rank]; d = dirs[rank]; ep = C[ei]; ea = atr[ei]
        for k in range(1, MAX_HOLD+1):
            bar = ei + k
            if bar >= n: break
            cp = d*(C[bar]-ep)/ea
            cps[rank, k-1] = cp
            if k < MIN_HOLD: continue
            p3 = d*(C[bar-3]-ep)/ea if k >= 3 else cp
            row = rank*MAX_HOLD + (k-1)
            Xs[row, 0] = cp; Xs[row, 1] = float(k); Xs[row, 2] = cp - p3
            if ctx_arr.shape[1] > 0:
                Xs[row, 3:] = ctx_arr[bar]
            valid_rows[row] = True
    probs = np.zeros(N*MAX_HOLD, dtype=np.float32)
    if valid_rows.any():
        probs[valid_rows] = exit_mdl.predict_proba(Xs[valid_rows])[:, 1]
    pnls = np.zeros(N, dtype=np.float64)
    exits = ["max"] * N
    for rank in range(N):
        ei = sw_idx_sub[rank]; d = dirs[rank]; ep = C[ei]; ea = atr[ei]
        xi = None; xr = "max"
        for k in range(1, MAX_HOLD+1):
            bar = ei + k
            if bar >= n: break
            cp = cps[rank, k-1]
            if not np.isfinite(cp): break
            if cp < -SL_HARD: xi, xr = bar, "hard_sl"; break
            if k >= MIN_HOLD and probs[rank*MAX_HOLD + (k-1)] >= EXIT_THRESHOLD:
                xi, xr = bar, "ml_exit"; break
        if xi is None: xi = min(ei + MAX_HOLD, n-1); xr = "max"
        pnls[rank] = d*(C[xi]-ep)/ea - SPREAD_USD/ea
        exits[rank] = xr
    sub = sub.copy()
    sub["pnl_R"] = pnls
    sub["exit"] = exits
    return sub


def report_block(label, df, end_t):
    slices = [
        ("LAST WEEK",       end_t - pd.Timedelta(days=7)),
        ("LAST 30 DAYS",    end_t - pd.Timedelta(days=30)),
        ("LAST 90 DAYS",    end_t - pd.Timedelta(days=90)),
        ("HOLDOUT (2024+)", pd.Timestamp("2024-01-01")),
    ]
    print(f"\n  ============ {label} ============")
    for name, start_t in slices:
        sub = df[(df["time"] >= start_t) & (df["time"] <= end_t)].reset_index(drop=True)
        if len(sub) < 5: continue
        dirs = sub["direction"].to_numpy(np.int64)
        m15  = sub["m15_dir"].to_numpy(np.int64)
        pnl  = sub["pnl_R"].to_numpy(float)
        gate_align = (m15 == dirs) & (m15 != 0)
        gate_anti  = (m15 == -dirs) & (m15 != 0)
        m_all  = metrics(pnl)
        m_aln  = metrics(pnl[gate_align])
        m_ant  = metrics(pnl[gate_anti])
        print(f"\n  --- {name}  ({str(start_t)[:10]} → {str(end_t)[:10]})  setups: {len(sub):,} ---")
        def fmt(name, m):
            if m is None: return f"      {name:>18}: (no trades)"
            return (f"      {name:>18}: n={m['n']:>5}  WR={m['wr']*100:>5.1f}%  "
                    f"PF={m['pf']:>5.2f}  sumR={m['sum_r']:>+7.1f}  DD={m['max_dd_r']:>5.1f}")
        print(fmt("NO GATE", m_all))
        print(fmt("M15 ALIGN", m_aln))
        print(fmt("M15 ANTI", m_ant))


def main():
    t0 = time.time()
    print("="*78); print("  Oracle RL exit × M15 TFK gate × M5+M1+M15 timeframes"); print("="*78)

    # 1. Load Oracle deployed bundle
    print("\n  loading Oracle bundle ...", flush=True)
    bundle = pickle.load(open(ORACLE_PKL, "rb"))
    exit_mdl = bundle["exit_mdl"]; exit_feats = bundle["exit_feats"]
    print(f"    exit features: {exit_feats}", flush=True)

    # 2. Load M5 swing data + setups
    print("\n  loading M5 swing + setups ...", flush=True)
    sw = pd.read_csv(SWING, parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    setup_files = sorted(glob.glob(SETUP_GLOB))
    needed_cols = ["time","direction","rule","atr","entry_price","label"] + list(exit_feats[3:])
    all_setups = []
    for f in setup_files:
        df = pd.read_csv(f, parse_dates=["time"])
        keep = [c for c in needed_cols if c in df.columns]
        all_setups.append(df[keep])
    setups = pd.concat(all_setups, ignore_index=True).sort_values("time").reset_index(drop=True)
    print(f"    swing rows: {len(sw):,}  setups: {len(setups):,}", flush=True)

    # 2b. Forward-fill per-setup context features onto the swing grid
    ctx_target = list(exit_feats[3:])
    available = [c for c in ctx_target if c in setups.columns]
    if available:
        ctx_df = setups[["time"] + available].sort_values("time").drop_duplicates("time")
        merged = pd.merge_asof(
            sw[["time"]].sort_values("time"),
            ctx_df.sort_values("time"),
            on="time", direction="backward")
        for c in available:
            sw[c] = merged[c].fillna(0).to_numpy()
        missing = [c for c in ctx_target if c not in available]
        for c in missing:
            sw[c] = 0.0
        print(f"    context cols filled from setups: {len(available)} / {len(ctx_target)}", flush=True)
    else:
        for c in ctx_target: sw[c] = 0.0
        print(f"    context cols all-zero (none found)", flush=True)

    # 3. Compute M15 TFK direction on M5 swing
    print("\n  computing M15 TFK on swing ...", flush=True)
    m15_per_m5 = m15_dir_at_bars(sw)
    times_sw = sw["time"].values.astype("datetime64[ns]")
    setup_t = setups["time"].values.astype("datetime64[ns]")
    sw_idx = np.searchsorted(times_sw, setup_t)
    sw_idx_safe = np.minimum(sw_idx, len(times_sw)-1)
    exact = times_sw[sw_idx_safe] == setup_t
    setups["m15_dir"] = np.where(exact, m15_per_m5[sw_idx_safe], 0)

    # 4. M5 NATIVE: sim with Oracle RL exit
    print("\n  simulating M5 native with Oracle RL exit ...", flush=True)
    setups_m5 = sim_oracle_rl_exit(setups, sw, exit_mdl, exit_feats)
    # attach m15_dir via the original setups time
    m_dir_map = dict(zip(setups["time"].astype("datetime64[ns]").astype(str), setups["m15_dir"]))
    setups_m5["m15_dir"] = setups_m5["time"].astype("datetime64[ns]").astype(str).map(m_dir_map).fillna(0).astype(int)
    print(f"    M5 simulated: {len(setups_m5):,}", flush=True)
    end_t = setups_m5["time"].max()
    report_block("M5 NATIVE (Oracle setups + RL exit)", setups_m5, end_t)

    # 5. M1 TEST: Hermes-style entries on m1_xau_full.parquet
    print("\n  building Hermes-style M1 entries ...", flush=True)
    from numba import njit as _njit

    m1 = pd.read_parquet(M1_PARQUET).sort_values("time").reset_index(drop=True)
    print(f"    M1 bars: {len(m1):,}", flush=True)
    tfk_m1 = compute_tfk(m1, flip_bars=5, color_confirm=8)
    O1=m1["open"].to_numpy(float); H1=m1["high"].to_numpy(float); L1=m1["low"].to_numpy(float); C1=m1["close"].to_numpy(float)
    cdir1=tfk_m1["committed_dir"].to_numpy(np.int64); tline1=tfk_m1["tfk_line"].to_numpy(float)
    # ATR M1
    prev_c=np.concatenate([[C1[0]],C1[:-1]])
    tr=np.maximum(H1-L1,np.maximum(np.abs(H1-prev_c),np.abs(L1-prev_c)))
    atr1=pd.Series(tr).rolling(14,min_periods=14).mean().to_numpy()
    dist_signed=np.where(atr1>0,(C1-tline1)/atr1,0.0); dist_abs=np.abs(dist_signed)
    valid=np.isfinite(atr1)&(atr1>0); valid[:200]=False; valid[-301:]=False
    valid&=(cdir1!=0)&((dist_abs<=0.5)|((dist_signed*cdir1)<=-1.5))
    idxs=np.where(valid)[0]; dirs=cdir1[idxs].copy()
    print(f"    M1 candidates: {len(idxs):,}", flush=True)
    # Simple trail exit on M1 (SL=6 ATR, TRAIL=2 ATR, MAXH=300)
    SL,TRAIL,MAXH_M1=6.0,2.0,300
    n1=len(m1); sp_m1=SPREAD_USD/np.nanmedian(atr1)
    @_njit(cache=True)
    def _sim_m1(idxs,dirs,O,H,L,C,atr,sp,SL,TRAIL,MAXH,n):
        m=len(idxs); pnl=np.empty(m)
        for k in range(m):
            i=idxs[k]; d=dirs[k]; a=atr[i]; ei=i+1
            if ei>=n or not(a>0): pnl[k]=0.0; continue
            ep=O[ei]; hard=SL*a; trd=TRAIL*a; mf=0.0
            end=min(ei+MAXH,n-1); done=False; out_r=0.0
            for j in range(ei,end+1):
                fav=d*(C[j]-ep)
                if fav>mf: mf=fav
                if d==1 and (ep-L[j])>=hard: out_r=-SL-sp; done=True; break
                if d==-1 and (H[j]-ep)>=hard: out_r=-SL-sp; done=True; break
                if mf>=trd and (mf-fav)>=trd: out_r=(mf-trd)/a-sp; done=True; break
            if not done: out_r=d*(C[end]-ep)/a-sp
            pnl[k]=out_r
        return pnl
    pnl_m1 = _sim_m1(idxs, dirs, O1,H1,L1,C1, atr1, sp_m1, SL,TRAIL,MAXH_M1, n1)
    # M15 dir at each M1 candidate (from M15 TFK)
    print("  computing M15 TFK on M1 ...", flush=True)
    m1_for_resample = m1[["time","open","high","low","close","tick_volume"]]
    m15_per_m1 = m15_dir_at_bars(m1_for_resample)
    m1_df = pd.DataFrame({
        "time": m1["time"].iloc[idxs].values,
        "direction": dirs,
        "m15_dir": m15_per_m1[idxs],
        "pnl_R": pnl_m1,
    })
    end_m1 = m1_df["time"].max()
    report_block("M1 (Hermes-style + trail exit)", m1_df, end_m1)

    # 6. M15 timeframe: aggregate M1 → M15 entries
    print("\n  building M15 entries ...", flush=True)
    s = m1.set_index("time")
    m15_bars = pd.DataFrame({
        "open":  s["open"].resample("15min").first(),
        "high":  s["high"].resample("15min").max(),
        "low":   s["low"].resample("15min").min(),
        "close": s["close"].resample("15min").last(),
        "tick_volume": s["tick_volume"].resample("15min").sum(),
    }).dropna(subset=["close"]).reset_index()
    tfk_m15 = compute_tfk(m15_bars, flip_bars=5, color_confirm=8)
    O15=m15_bars["open"].to_numpy(float); H15=m15_bars["high"].to_numpy(float)
    L15=m15_bars["low"].to_numpy(float); C15=m15_bars["close"].to_numpy(float)
    cdir15=tfk_m15["committed_dir"].to_numpy(np.int64); tline15=tfk_m15["tfk_line"].to_numpy(float)
    prev_c=np.concatenate([[C15[0]],C15[:-1]])
    tr=np.maximum(H15-L15,np.maximum(np.abs(H15-prev_c),np.abs(L15-prev_c)))
    atr15=pd.Series(tr).rolling(14,min_periods=14).mean().to_numpy()
    dist_signed=np.where(atr15>0,(C15-tline15)/atr15,0.0); dist_abs=np.abs(dist_signed)
    valid=np.isfinite(atr15)&(atr15>0); valid[:50]=False; valid[-51:]=False
    valid&=(cdir15!=0)&((dist_abs<=0.5)|((dist_signed*cdir15)<=-1.5))
    idxs15=np.where(valid)[0]; dirs15=cdir15[idxs15].copy()
    print(f"    M15 candidates: {len(idxs15):,}", flush=True)
    SL15,TRAIL15,MAXH15=6.0,2.0,30   # 30 M15 bars = 7.5h
    n15=len(m15_bars); sp_m15=SPREAD_USD/np.nanmedian(atr15)
    pnl_m15 = _sim_m1(idxs15, dirs15, O15,H15,L15,C15, atr15, sp_m15, SL15,TRAIL15,MAXH15, n15)
    m15_dir_for_m15 = cdir15  # same direction value (we're already on M15)
    m15_df = pd.DataFrame({
        "time": m15_bars["time"].iloc[idxs15].values,
        "direction": dirs15,
        "m15_dir": m15_dir_for_m15[idxs15],   # ALIGN trivially (direction = m15_dir)
        "pnl_R": pnl_m15,
    })
    end_m15 = m15_df["time"].max()
    # For M15: gate is whether the entry's direction aligns with M15 TFK at entry time.
    # Since cdir15 IS the entry's M15 TFK direction at entry, ALIGN gate is by definition
    # always-true. So gate the M15 entries by the *prior* M15 bar's TFK as a proxy:
    cdir15_prev = np.concatenate([[0], cdir15[:-1]])
    m15_df["m15_dir"] = cdir15_prev[idxs15]
    report_block("M15 (TFK pullback entries + trail exit)", m15_df, end_m15)

    print(f"\n  TOTAL: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
