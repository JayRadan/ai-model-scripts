"""
Oracle XAU — REPLACE the cluster-based gate with M15 TFK ANTI gate.

EVERYTHING ELSE STAYS THE SAME:
  - Per-cluster confirm models   (bundle.mdls[(cid,'RL')])
  - Per-cluster confirm thresholds (bundle.thrs[(cid,'RL')])
  - Deployed RL exit policy       (bundle.exit_mdl)
  - Meta filter                   (bundle.meta_mdl @ 0.775)

THE ONE CHANGE: instead of blocking clusters {1,2} (current live), we
keep ONLY setups where M15 TFK direction is ANTI to trade direction.
Cluster IDs are still used to ROUTE setups to their per-cluster models —
we don't drop the trained machinery, we just drop the cluster-based block.

Honest cutoff 2024-01-01.
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
FINGERPRINTS = ROOT / "products" / "_shared" / "data" / "regime_fingerprints_4h.csv"
SPREAD_USD = 0.30
MIN_HOLD = 3; MAX_HOLD = 60; SL_HARD = 5.0; EXIT_THRESHOLD = 0.55


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


def attach_cluster_id(setups, swing_df, fingerprints_df):
    """Each setup's cluster = the new_label of the 4h regime block whose
    [start_idx, end_idx) covers the setup's swing index."""
    times_sw = swing_df["time"].values.astype("datetime64[ns]")
    setup_t = setups["time"].values.astype("datetime64[ns]")
    sw_idx = np.searchsorted(times_sw, setup_t)
    sw_idx_safe = np.minimum(sw_idx, len(times_sw)-1)
    exact = times_sw[sw_idx_safe] == setup_t
    sw_idx_for_setup = np.where(exact, sw_idx_safe, -1)
    setups = setups.copy(); setups["sw_idx"] = sw_idx_for_setup

    # Build cluster lookup: for each swing index, which cluster does it fall in?
    cid_per_sw_idx = np.full(len(swing_df), -1, dtype=np.int64)
    for _, row in fingerprints_df.iterrows():
        s, e = int(row["start_idx"]), int(row["end_idx"])
        if 0 <= s < e <= len(swing_df):
            cid_per_sw_idx[s:e] = int(row["new_label"])
    setups["cid"] = np.where(setups["sw_idx"] >= 0,
                             cid_per_sw_idx[np.maximum(setups["sw_idx"], 0)],
                             -1)
    return setups


def sim_rl_exit(setups, swing_df, exit_mdl, exit_feats):
    times_sw = swing_df["time"].values.astype("datetime64[ns]")
    C = swing_df["close"].to_numpy(float)
    H = swing_df["high"].to_numpy(float); L = swing_df["low"].to_numpy(float)
    n = len(swing_df)
    prev_c = np.concatenate([[C[0]], C[:-1]])
    tr = np.maximum(H-L, np.maximum(np.abs(H-prev_c), np.abs(L-prev_c)))
    atr = pd.Series(tr).rolling(14, min_periods=14).mean().to_numpy()
    ctx_target = list(exit_feats[3:])
    ctx_arr = np.zeros((n, len(ctx_target)), dtype=np.float64)
    for j, c in enumerate(ctx_target):
        if c in swing_df.columns: ctx_arr[:, j] = swing_df[c].fillna(0).to_numpy(float)
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


def report(label, setups, end_t):
    slices = [
        ("LAST WEEK",      end_t - pd.Timedelta(days=7)),
        ("LAST 30 DAYS",   end_t - pd.Timedelta(days=30)),
        ("LAST 90 DAYS",   end_t - pd.Timedelta(days=90)),
        ("HOLDOUT 2024+",  pd.Timestamp("2024-01-01")),
    ]
    print(f"\n  =============== {label} ===============")
    for name, start_t in slices:
        sub = setups[(setups["time"] >= start_t) & (setups["time"] <= end_t)]
        if len(sub) < 5: continue
        m = metrics(sub["pnl_R"].to_numpy())
        if m:
            tag = ""
            if m["pf"] >= 2.0 and m["sum_r"] > 0: tag = " <-- PF>=2"
            elif m["pf"] >= 1.5 and m["sum_r"] > 0: tag = " <-- PF>=1.5"
            print(f"    {name:>15}: n={m['n']:>5,}  WR={m['wr']*100:>5.1f}%  "
                  f"PF={m['pf']:>5.2f}  sumR={m['sum_r']:>+8.1f}  DD={m['max_dd_r']:>6.1f}{tag}")


def main():
    t0 = time.time()
    print("="*78); print("  Oracle XAU — REPLACE cluster gate with M15 TFK ANTI"); print("="*78)

    bundle = pickle.load(open(ORACLE_PKL, "rb"))
    mdls = bundle["mdls"]; thrs = bundle["thrs"]
    exit_mdl = bundle["exit_mdl"]; exit_feats = bundle["exit_feats"]
    meta_mdl = bundle["meta_mdl"]; meta_feats = bundle["meta_feats"]
    meta_threshold = bundle["meta_threshold"]
    v72l_feats = bundle["v72l_feats"]
    BLOCKED_CLUSTERS = {1, 2}  # current live block (per memory)
    print(f"  bundle: clusters {sorted(bundle['q_entry'].keys())}  "
          f"meta_thr={meta_threshold:.3f}  live-blocked={sorted(BLOCKED_CLUSTERS)}", flush=True)

    print("\n  loading swing + setups + fingerprints ...", flush=True)
    sw = pd.read_csv(SWING, parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    setup_files = sorted(glob.glob(SETUP_GLOB))
    setups = pd.concat(
        [pd.read_csv(f, parse_dates=["time"]) for f in setup_files],
        ignore_index=True).sort_values("time").reset_index(drop=True)
    fp = pd.read_csv(FINGERPRINTS, parse_dates=["center_time"])
    print(f"    swing: {len(sw):,}  setups: {len(setups):,}  fingerprints: {len(fp):,}", flush=True)

    # Forward-fill context features from setups onto swing (for exit_mdl)
    available_ctx = [c for c in exit_feats[3:] if c in setups.columns]
    if available_ctx:
        ctx_df = setups[["time"]+available_ctx].sort_values("time").drop_duplicates("time")
        merged = pd.merge_asof(sw[["time"]].sort_values("time"),
                               ctx_df.sort_values("time"), on="time", direction="backward")
        for c in available_ctx:
            sw[c] = merged[c].fillna(0).to_numpy()
    for c in [c for c in exit_feats[3:] if c not in available_ctx]:
        sw[c] = 0.0

    # Attach cluster IDs to setups
    setups = attach_cluster_id(setups, sw, fp)
    print(f"  cluster distribution: {setups['cid'].value_counts().to_dict()}", flush=True)
    # Drop setups without a cluster assignment
    setups = setups[(setups["cid"] >= 0)].reset_index(drop=True)

    # Compute M15 TFK direction at each setup time
    print("\n  computing M15 TFK on swing ...", flush=True)
    m15_per_m5 = m15_dir_at_bars(sw)
    times_sw = sw["time"].values.astype("datetime64[ns]")
    setup_t = setups["time"].values.astype("datetime64[ns]")
    sw_idx = np.searchsorted(times_sw, setup_t)
    sw_idx_safe = np.minimum(sw_idx, len(times_sw)-1)
    exact = times_sw[sw_idx_safe] == setup_t
    setups["m15_dir"] = np.where(exact, m15_per_m5[sw_idx_safe], 0)

    # Per-cluster confirm filter: P(winner) from mdls[(cid,'RL')] >= thrs[(cid,'RL')]
    print("\n  applying per-cluster confirm filter ...", flush=True)
    setups["confirm_ok"] = False
    for cid in sorted(setups["cid"].unique()):
        key = (int(cid), "RL")
        if key not in mdls or key not in thrs:
            print(f"    cluster {cid}: NO MODEL — skipping", flush=True); continue
        mdl = mdls[key]; thr = thrs[key]
        sub = setups[setups["cid"] == cid]
        X = sub[v72l_feats].fillna(0).to_numpy(np.float32)
        p = mdl.predict_proba(X)[:, 1]
        keep = p >= thr
        setups.loc[sub.index, "confirm_ok"] = keep
        print(f"    cluster {cid}: {int(keep.sum()):>6,} / {len(sub):>6,} confirmed (thr={thr:.3f})", flush=True)
    confirmed = setups[setups["confirm_ok"]].reset_index(drop=True)
    print(f"\n  CONFIRMED total: {len(confirmed):,}", flush=True)

    # Define the two pipelines:
    # CURRENT LIVE: drop confirmed setups whose cluster ∈ {1,2}
    pipe_live = confirmed[~confirmed["cid"].isin(BLOCKED_CLUSTERS)].reset_index(drop=True)
    # NEW (proposed): keep confirmed setups whose M15 TFK is ANTI to direction
    pipe_new = confirmed[(confirmed["m15_dir"] == -confirmed["direction"]) &
                         (confirmed["m15_dir"] != 0)].reset_index(drop=True)
    print(f"\n  CURRENT LIVE gate (block clusters 1,2): {len(pipe_live):,} pass", flush=True)
    print(f"  NEW gate (M15 TFK ANTI):                {len(pipe_new):,} pass", flush=True)

    # RL exit on both
    print("\n  simulating with RL exit (CURRENT LIVE) ...", flush=True)
    sim_live = sim_rl_exit(pipe_live, sw, exit_mdl, exit_feats)
    print("  simulating with RL exit (NEW M15 ANTI) ...", flush=True)
    sim_new = sim_rl_exit(pipe_new, sw, exit_mdl, exit_feats)

    # Meta filter on both
    print("\n  applying meta filter (thr=", meta_threshold, ") ...", flush=True)
    def apply_meta(sim_df):
        if len(sim_df) == 0: return sim_df
        feat_cols = [f for f in meta_feats if f in sim_df.columns or f in ("direction","cid")]
        for f in ("direction","cid"):
            if f in feat_cols and f not in sim_df.columns: sim_df[f] = 0
        X = sim_df[feat_cols].fillna(0).to_numpy(np.float32)
        p = meta_mdl.predict_proba(X)[:, 1]
        keep = p >= meta_threshold
        out = sim_df[keep].reset_index(drop=True)
        print(f"    meta-keep: {int(keep.sum()):,} / {len(sim_df):,}", flush=True)
        return out
    sim_live = apply_meta(sim_live)
    sim_new  = apply_meta(sim_new)

    end_t = max(sim_live["time"].max() if len(sim_live) else pd.Timestamp("2020"),
                sim_new["time"].max() if len(sim_new) else pd.Timestamp("2020"))
    print(f"\n  end_t = {end_t}")

    report("CURRENT LIVE  (cluster gate: block C1+C2)", sim_live, end_t)
    report("NEW PROPOSED  (M15 TFK ANTI gate)",         sim_new,  end_t)

    # Plot equity comparison
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 5))
    for label, df, color in [("CURRENT (C1+C2 block)", sim_live, "steelblue"),
                              ("NEW (M15 ANTI)",        sim_new,  "darkgreen")]:
        sub = df[df["time"] >= pd.Timestamp("2024-01-01")].sort_values("time")
        if len(sub) == 0: continue
        eq = np.cumsum(sub["pnl_R"].to_numpy())
        m = metrics(sub["pnl_R"].to_numpy())
        ax.plot(eq, lw=1.0, color=color,
                label=f"{label}  n={m['n']}  PF={m['pf']:.2f}  sumR={m['sum_r']:+.0f}  DD={m['max_dd_r']:.0f}")
    ax.set_title("Oracle XAU — replace cluster gate with M15 TFK ANTI (honest 2024+ holdout)")
    ax.set_xlabel("trade #"); ax.set_ylabel("cum R"); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); out = HERE / "equity_replace_gate.png"; fig.savefig(out, dpi=110); plt.close(fig)
    print(f"\n  saved {out}")
    print(f"\n  TOTAL: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
