"""
Per-cluster breakdown of how Oracle performs under each gate variant.

For each cluster c0..c4, show holdout PF/WR/DD/sumR under:
  - NO GATE          (all confirmed setups of that cluster, raw)
  - CLUSTER BLOCK    (current live: block if cid ∈ {1,2})
  - M15 TFK ANTI     (the proposed replacement)
  - BOTH (cluster allowed AND M15 ANTI)

Same per-cluster confirm + RL exit + meta pipeline as the head-to-head test.
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
BLOCKED_CLUSTERS = {1, 2}
HOLDOUT_START = pd.Timestamp("2024-01-01")


def metrics(r):
    r = np.asarray(r); r = r[np.isfinite(r)]
    if len(r) == 0: return None
    w, l = r[r>0], r[r<=0]; eq = np.cumsum(r)
    return dict(n=int(len(r)), wr=float((r>0).mean()),
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


def attach_cid(setups, swing_df, fingerprints_df):
    times_sw = swing_df["time"].values.astype("datetime64[ns]")
    setup_t = setups["time"].values.astype("datetime64[ns]")
    sw_idx = np.searchsorted(times_sw, setup_t)
    sw_idx_safe = np.minimum(sw_idx, len(times_sw)-1)
    exact = times_sw[sw_idx_safe] == setup_t
    sw_idx_for_setup = np.where(exact, sw_idx_safe, -1)
    setups = setups.copy(); setups["sw_idx"] = sw_idx_for_setup
    cid_per = np.full(len(swing_df), -1, dtype=np.int64)
    for _, row in fingerprints_df.iterrows():
        s, e = int(row["start_idx"]), int(row["end_idx"])
        if 0 <= s < e <= len(swing_df):
            cid_per[s:e] = int(row["new_label"])
    setups["cid"] = np.where(setups["sw_idx"] >= 0,
                             cid_per[np.maximum(setups["sw_idx"], 0)], -1)
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


def main():
    t0 = time.time()
    print("="*80); print("  Per-cluster breakdown — c0..c4 under each gate variant"); print("="*80)
    bundle = pickle.load(open(ORACLE_PKL, "rb"))
    mdls = bundle["mdls"]; thrs = bundle["thrs"]
    exit_mdl = bundle["exit_mdl"]; exit_feats = bundle["exit_feats"]
    meta_mdl = bundle["meta_mdl"]; meta_feats = bundle["meta_feats"]
    meta_threshold = bundle["meta_threshold"]
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

    setups = attach_cid(setups, sw, fp)
    setups = setups[setups["cid"] >= 0].reset_index(drop=True)
    m15_per_m5 = m15_dir_at_bars(sw)
    times_sw = sw["time"].values.astype("datetime64[ns]")
    setup_t = setups["time"].values.astype("datetime64[ns]")
    sw_idx = np.searchsorted(times_sw, setup_t)
    sw_idx_safe = np.minimum(sw_idx, len(times_sw)-1)
    exact = times_sw[sw_idx_safe] == setup_t
    setups["m15_dir"] = np.where(exact, m15_per_m5[sw_idx_safe], 0)

    # Per-cluster confirm
    setups["confirm_ok"] = False
    for cid in sorted(setups["cid"].unique()):
        key = (int(cid), "RL")
        if key not in mdls: continue
        sub = setups[setups["cid"] == cid]
        X = sub[v72l_feats].fillna(0).to_numpy(np.float32)
        p = mdls[key].predict_proba(X)[:, 1]
        setups.loc[sub.index, "confirm_ok"] = p >= thrs[key]
    confirmed = setups[setups["confirm_ok"]].reset_index(drop=True)

    # RL exit on ALL confirmed (we'll filter by gate later)
    print(f"\n  simulating RL exit on all {len(confirmed):,} confirmed setups ...", flush=True)
    sim_all = sim_rl_exit(confirmed, sw, exit_mdl, exit_feats)

    # Meta filter
    feat_cols = [f for f in meta_feats if f in sim_all.columns or f in ("direction","cid")]
    for f in ("direction","cid"):
        if f in feat_cols and f not in sim_all.columns: sim_all[f] = 0
    X = sim_all[feat_cols].fillna(0).to_numpy(np.float32)
    p = meta_mdl.predict_proba(X)[:, 1]
    sim_all["meta_keep"] = p >= meta_threshold
    print(f"  meta-keep: {int(sim_all['meta_keep'].sum()):,} / {len(sim_all):,}", flush=True)

    # Per-cluster, holdout-only breakdown
    hd = sim_all[(sim_all["time"] >= HOLDOUT_START) & (sim_all["meta_keep"])].reset_index(drop=True)
    print(f"\n  HOLDOUT 2024+ confirmed+meta-kept: {len(hd):,}")

    print(f"\n  ============ Per-cluster, 2.4y honest holdout (after confirm + meta) ============")
    print(f"  {'cluster':>7} | {'NO GATE (raw)':<30} | {'BLOCKED' if True else '':<10} | {'M15 ANTI gate':<30} | {'M15 ALIGN gate':<30}")
    print(f"  {'':>7} | {'n  WR%   PF  sumR  DD':>30} | {'live?':<10} | {'n  WR%   PF  sumR  DD':>30} | {'n  WR%   PF  sumR  DD':>30}")
    def fmt(m):
        if not m: return "       —"
        return f"{m['n']:>4} {m['wr']*100:>4.1f}% {m['pf']:>5.2f} {m['sum_r']:>+6.0f} {m['max_dd_r']:>4.0f}"
    for cid in [0, 1, 2, 3, 4]:
        cdf = hd[hd["cid"] == cid].reset_index(drop=True)
        if len(cdf) < 5: continue
        m_all = metrics(cdf["pnl_R"].to_numpy())
        anti = (cdf["m15_dir"] == -cdf["direction"]) & (cdf["m15_dir"] != 0)
        align = (cdf["m15_dir"] == cdf["direction"]) & (cdf["m15_dir"] != 0)
        m_anti = metrics(cdf.loc[anti, "pnl_R"].to_numpy())
        m_align = metrics(cdf.loc[align, "pnl_R"].to_numpy())
        blocked = "BLOCKED" if cid in BLOCKED_CLUSTERS else "allowed"
        print(f"  c{cid:>6} | {fmt(m_all):<30} | {blocked:<10} | {fmt(m_anti):<30} | {fmt(m_align):<30}")

    # Aggregated comparisons
    def pf_block(df, name):
        m = metrics(df["pnl_R"].to_numpy()) if len(df) > 0 else None
        if m:
            print(f"    {name:>35}: n={m['n']:>5,}  WR={m['wr']*100:>5.1f}%  PF={m['pf']:>5.2f}  sumR={m['sum_r']:>+7.0f}  DD={m['max_dd_r']:>5.0f}")
        else:
            print(f"    {name:>35}: (empty)")
    print(f"\n  ============ Aggregated holdout (all 5 clusters combined) ============")
    pf_block(hd, "NO GATE (all clusters allowed)")
    pf_block(hd[~hd["cid"].isin(BLOCKED_CLUSTERS)], "CURRENT LIVE (block C1+C2)")
    anti_all = (hd["m15_dir"] == -hd["direction"]) & (hd["m15_dir"] != 0)
    pf_block(hd[anti_all], "M15 TFK ANTI (replace cluster gate)")
    align_all = (hd["m15_dir"] == hd["direction"]) & (hd["m15_dir"] != 0)
    pf_block(hd[align_all], "M15 TFK ALIGN (curiosity)")
    pf_block(hd[(~hd["cid"].isin(BLOCKED_CLUSTERS)) & anti_all], "BOTH: cluster-allowed AND M15 ANTI")

    print(f"\n  TOTAL: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
