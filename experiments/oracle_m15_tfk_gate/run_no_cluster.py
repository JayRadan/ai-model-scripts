"""
Oracle XAU — CLUSTER-FREE rebuild.

Replaces the entire cluster machinery (regime detection, C1/C2 block,
per-cluster q_entry + confirm + meta models) with:

  1. M15 TFK ANTI gate (only setups where M15 TFK opposes trade direction)
  2. A SINGLE cluster-agnostic Q-entry XGB regressor trained on union of
     all v72L features
  3. The deployed RL exit policy (unchanged)

Honest cutoff: trains on pre-2024-01-01 setups, tests on 2024-2026 holdout.
Reports last week / 30d / 90d / full holdout.
"""
from __future__ import annotations
import sys, time, glob, pickle
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "products/hermes_xau"))
from tfk import compute_tfk

SWING = ROOT / "data" / "swing_v5_xauusd.csv"
SETUP_GLOB = str(ROOT / "data" / "setups_*_v72l.csv")
ORACLE_PKL = ROOT / "products" / "models" / "oracle_xau_validated.pkl"
TRAIN_CUTOFF = pd.Timestamp("2024-01-01 00:00:00")
SPREAD_USD = 0.30

# RL exit params (same as deploy_bundle)
MIN_HOLD = 3; MAX_HOLD = 60
SL_HARD = 5.0; EXIT_THRESHOLD = 0.55


def metrics(r):
    r=np.asarray(r); r=r[np.isfinite(r)]
    if len(r)==0: return None
    w,l=r[r>0],r[r<=0]; eq=np.cumsum(r)
    return dict(n=int(len(r)),wr=float((r>0).mean()),
                pf=float(w.sum()/max(-l.sum(),1e-9)),
                sum_r=float(r.sum()),
                max_dd_r=float((np.maximum.accumulate(eq)-eq).max()))


def m15_dir_at_bars(df_bars):
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
    aligned = pd.merge_asof(
        df_bars[["time"]].sort_values("time"),
        m15[["time","m15_dir"]].assign(time=m15["time"] + pd.Timedelta("15min")).sort_values("time"),
        on="time", direction="backward")
    return aligned["m15_dir"].fillna(0).to_numpy(np.int64)


def sim_rl_exit(setups, swing_df, exit_mdl, exit_feats):
    times_sw = swing_df["time"].values.astype("datetime64[ns]")
    C = swing_df["close"].to_numpy(float)
    H = swing_df["high"].to_numpy(float)
    L = swing_df["low"].to_numpy(float)
    n = len(swing_df)
    prev_c = np.concatenate([[C[0]], C[:-1]])
    tr = np.maximum(H-L, np.maximum(np.abs(H-prev_c), np.abs(L-prev_c)))
    atr = pd.Series(tr).rolling(14, min_periods=14).mean().to_numpy()
    ctx_target = list(exit_feats[3:])
    ctx_arr = np.zeros((n, len(ctx_target)), dtype=np.float64)
    for j, c in enumerate(ctx_target):
        if c in swing_df.columns:
            ctx_arr[:, j] = swing_df[c].fillna(0).to_numpy(float)
    setup_t = setups["time"].values.astype("datetime64[ns]")
    sw_idx = np.searchsorted(times_sw, setup_t)
    sw_idx_safe = np.minimum(sw_idx, n-1)
    exact = times_sw[sw_idx_safe] == setup_t
    valid = exact & (atr[sw_idx_safe] > 0) & np.isfinite(atr[sw_idx_safe])
    sub = setups[valid].reset_index(drop=True)
    sw_idx_sub = sw_idx_safe[valid]
    dirs = sub["direction"].to_numpy(np.int64)
    N = len(sub)
    if N == 0: return pd.DataFrame()
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
            if ctx_arr.shape[1] > 0: Xs[row, 3:] = ctx_arr[bar]
            valid_rows[row] = True
    probs = np.zeros(N*MAX_HOLD, dtype=np.float32)
    if valid_rows.any():
        probs[valid_rows] = exit_mdl.predict_proba(Xs[valid_rows])[:, 1]
    pnls = np.zeros(N, dtype=np.float64)
    for rank in range(N):
        ei = sw_idx_sub[rank]; d = dirs[rank]; ep = C[ei]; ea = atr[ei]
        xi = None
        for k in range(1, MAX_HOLD+1):
            bar = ei + k
            if bar >= n: break
            cp = cps[rank, k-1]
            if not np.isfinite(cp): break
            if cp < -SL_HARD: xi = bar; break
            if k >= MIN_HOLD and probs[rank*MAX_HOLD + (k-1)] >= EXIT_THRESHOLD:
                xi = bar; break
        if xi is None: xi = min(ei + MAX_HOLD, n-1)
        pnls[rank] = d*(C[xi]-ep)/ea - SPREAD_USD/ea
    sub = sub.copy(); sub["pnl_R"] = pnls
    return sub


def main():
    t0 = time.time()
    print("="*78); print("  Oracle CLUSTER-FREE — M15 TFK ANTI gate + single Q + RL exit"); print("="*78)
    print("\n  loading Oracle bundle (for exit model + feature list) ...", flush=True)
    bundle = pickle.load(open(ORACLE_PKL, "rb"))
    exit_mdl = bundle["exit_mdl"]; exit_feats = bundle["exit_feats"]
    v72l_feats = bundle["v72l_feats"]

    print("\n  loading swing + setups ...", flush=True)
    sw = pd.read_csv(SWING, parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    setup_files = sorted(glob.glob(SETUP_GLOB))
    setups = pd.concat(
        [pd.read_csv(f, parse_dates=["time"]) for f in setup_files],
        ignore_index=True).sort_values("time").reset_index(drop=True)
    print(f"    swing: {len(sw):,}  setups: {len(setups):,}", flush=True)

    # Forward-fill context features from setups onto swing grid (for exit mdl)
    ctx_target = list(exit_feats[3:])
    available_ctx = [c for c in ctx_target if c in setups.columns]
    if available_ctx:
        ctx_df = setups[["time"]+available_ctx].sort_values("time").drop_duplicates("time")
        merged = pd.merge_asof(sw[["time"]].sort_values("time"),
                               ctx_df.sort_values("time"), on="time", direction="backward")
        for c in available_ctx:
            sw[c] = merged[c].fillna(0).to_numpy()
    for c in [c for c in ctx_target if c not in available_ctx]:
        sw[c] = 0.0

    # M15 TFK direction per swing bar
    print("\n  computing M15 TFK on swing ...", flush=True)
    m15_per_m5 = m15_dir_at_bars(sw)
    times_sw = sw["time"].values.astype("datetime64[ns]")
    setup_t = setups["time"].values.astype("datetime64[ns]")
    sw_idx = np.searchsorted(times_sw, setup_t)
    sw_idx_safe = np.minimum(sw_idx, len(times_sw)-1)
    exact = times_sw[sw_idx_safe] == setup_t
    setups["m15_dir"] = np.where(exact, m15_per_m5[sw_idx_safe], 0)

    # Apply M15 TFK ANTI gate
    dirs = setups["direction"].to_numpy(np.int64)
    m15 = setups["m15_dir"].to_numpy(np.int64)
    gate_anti = (m15 == -dirs) & (m15 != 0)
    print(f"    setups passing M15 ANTI gate: {int(gate_anti.sum()):,} / {len(setups):,}", flush=True)
    setups_gated = setups[gate_anti].reset_index(drop=True)

    # Train one cluster-agnostic Q-entry XGB regressor on pre-2024 gated setups
    # Target: simulate each setup once (with RL exit) → pnl_R as label
    print("\n  simulating ALL gated setups with RL exit (for labels + final eval) ...", flush=True)
    sim_all = sim_rl_exit(setups_gated, sw, exit_mdl, exit_feats)
    print(f"    simulated: {len(sim_all):,}", flush=True)

    train_mask = sim_all["time"] < TRAIN_CUTOFF
    test_mask = ~train_mask
    feat_cols = [c for c in v72l_feats if c in sim_all.columns]
    print(f"    Q features: {len(feat_cols)}  (using v72l_feats)", flush=True)
    print(f"    train: {int(train_mask.sum()):,}  test: {int(test_mask.sum()):,}", flush=True)

    from xgboost import XGBRegressor
    Q = XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.04, subsample=0.85,
                     colsample_bytree=0.85, min_child_weight=10, reg_lambda=1.0,
                     objective="reg:squarederror", tree_method="hist",
                     random_state=42, verbosity=0, n_jobs=-1)
    Xtr = sim_all.loc[train_mask, feat_cols].fillna(0).to_numpy(np.float32)
    ytr = sim_all.loc[train_mask, "pnl_R"].astype(np.float32).to_numpy()
    print(f"\n  fitting cluster-agnostic Q regressor on {len(Xtr):,} samples ...", flush=True)
    Q.fit(Xtr, ytr)
    q_te = Q.predict(sim_all.loc[test_mask, feat_cols].fillna(0).to_numpy(np.float32))
    sim_all.loc[test_mask, "q_score"] = q_te

    test_df = sim_all[test_mask].reset_index(drop=True)
    end_t = test_df["time"].max()
    print(f"\n  test span: {test_df['time'].min()} → {end_t}", flush=True)

    # Sweep Q thresholds across multiple time windows
    print(f"\n  ======================= RESULTS =======================")
    slices = [
        ("LAST WEEK",        end_t - pd.Timedelta(days=7)),
        ("LAST 30 DAYS",     end_t - pd.Timedelta(days=30)),
        ("LAST 90 DAYS",     end_t - pd.Timedelta(days=90)),
        ("HOLDOUT (2024+)",  pd.Timestamp("2024-01-01")),
    ]
    for label, start_t in slices:
        sub = test_df[(test_df["time"] >= start_t) & (test_df["time"] <= end_t)].reset_index(drop=True)
        if len(sub) < 5: continue
        print(f"\n  --- {label}  ({str(start_t)[:10]} → {str(end_t)[:10]})  candidates: {len(sub):,} ---")
        print(f"    {'Q≥':>5} {'n':>5} {'WR%':>5} {'PF':>5} {'sumR':>+8} {'DD':>6} {'t/d':>5}")
        span = max((end_t-start_t).days, 1)
        for qt in [-5, 0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5]:
            sel = sub["q_score"] >= qt
            if sel.sum() < 5: continue
            m = metrics(sub.loc[sel, "pnl_R"].to_numpy())
            if m:
                tag = ""
                if m["pf"] >= 2.0 and m["sum_r"] > 0: tag = " <-- PF>=2"
                elif m["pf"] >= 1.5 and m["sum_r"] > 0: tag = " <-- PF>=1.5"
                print(f"    {qt:>5.1f} {m['n']:>5,} {m['wr']*100:>5.1f} {m['pf']:>5.2f} {m['sum_r']:>+8.1f} {m['max_dd_r']:>6.1f} {m['n']/span:>5.2f}{tag}")

    # Equity plot at best q threshold over the full holdout
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    best=None
    for qt in [-5, 0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5]:
        sel = test_df["q_score"] >= qt
        if sel.sum() < 100: continue
        m = metrics(test_df.loc[sel, "pnl_R"].to_numpy())
        if m and m["sum_r"] > 0:
            score = m["pf"]*1000 + m["n"]
            if best is None or score > best[0]: best = (score, qt, m, sel)
    if best:
        _, qt, m, sel = best
        sub = test_df[sel].sort_values("time").reset_index(drop=True)
        eq = np.cumsum(sub["pnl_R"].to_numpy())
        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(eq, lw=1.0, color="darkgreen")
        ax.set_title(f"Cluster-free Oracle (M15 ANTI + Q + RL exit) @ Q≥{qt:.1f}\n"
                     f"n={m['n']:,}  WR={m['wr']*100:.1f}%  PF={m['pf']:.2f}  "
                     f"sumR={m['sum_r']:+.0f}  DD={m['max_dd_r']:.0f}")
        ax.set_xlabel("trade #"); ax.set_ylabel("cum R"); ax.grid(alpha=0.3)
        fig.tight_layout(); out = HERE / "equity_no_cluster.png"; fig.savefig(out, dpi=110); plt.close(fig)
        print(f"\n  saved {out}")
        print(f"  BEST HOLDOUT: Q≥{qt:.1f}  PF={m['pf']:.2f}  sumR={m['sum_r']:+.0f}  DD={m['max_dd_r']:.0f}")

    print(f"\n  TOTAL: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
