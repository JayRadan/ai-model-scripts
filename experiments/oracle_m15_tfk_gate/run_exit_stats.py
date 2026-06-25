"""
Trade-duration + exit-reason stats for Oracle RL exit pipeline.

For each closed trade in the PURE M15 ANTI gate pipeline (2.4y holdout),
record:
  - bars_held  (M5 bars = minutes/5)
  - exit reason: {hard_sl, ml_exit, max_hold}

Report distributions and PF/sumR sliced by exit reason and hold duration.
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
HOLDOUT = pd.Timestamp("2024-01-01")


def metrics(r):
    r = np.asarray(r); r = r[np.isfinite(r)]
    if len(r) == 0: return None
    w, l = r[r>0], r[r<=0]
    return dict(n=int(len(r)), wr=float((r>0).mean()),
                pf=float(w.sum()/max(-l.sum(),1e-9)),
                sum_r=float(r.sum()), avg=float(r.mean()))


def m15_dir(df_bars):
    s = df_bars.set_index("time")
    m15 = pd.DataFrame({
        "open": s["open"].resample("15min").first(),
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


def sim_rl_exit_with_reasons(setups, swing_df, exit_mdl, exit_feats):
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
    bars_held = np.zeros(N, dtype=np.int64)
    reasons = []
    for rank in range(N):
        ei = sw_idx_sub[rank]; d = dirs[rank]; ep = C[ei]; ea = atr[ei]
        xi = None; xr = "max_hold"
        for k in range(1, MAX_HOLD+1):
            bar = ei + k
            if bar >= n: break
            cp = cps[rank, k-1]
            if not np.isfinite(cp): break
            if cp < -SL_HARD: xi = bar; xr = "hard_sl"; break
            if k >= MIN_HOLD and probs[rank*MAX_HOLD + (k-1)] >= EXIT_THRESHOLD:
                xi = bar; xr = "ml_exit"; break
        if xi is None: xi = min(ei + MAX_HOLD, n-1); xr = "max_hold"
        pnls[rank] = d*(C[xi]-ep)/ea - SPREAD_USD/ea
        bars_held[rank] = xi - ei
        reasons.append(xr)
    sub = sub.copy()
    sub["pnl_R"] = pnls
    sub["bars_held"] = bars_held
    sub["exit_reason"] = reasons
    return sub


def main():
    t0 = time.time()
    print("="*78); print("  Oracle RL exit — trade duration + exit reason stats"); print("="*78)
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

    # cluster
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

    m15_per_m5 = m15_dir(sw)
    sw_idx_safe = np.minimum(np.searchsorted(times_sw, setups["time"].values.astype("datetime64[ns]")), len(times_sw)-1)
    exact = times_sw[sw_idx_safe] == setups["time"].values.astype("datetime64[ns]")
    setups["m15_dir"] = np.where(exact, m15_per_m5[sw_idx_safe], 0)

    # confirm
    setups["confirm_ok"] = False
    for cid in sorted(setups["cid"].unique()):
        key = (int(cid), "RL")
        if key not in mdls: continue
        sub = setups[setups["cid"] == cid]
        X = sub[v72l_feats].fillna(0).to_numpy(np.float32)
        p = mdls[key].predict_proba(X)[:, 1]
        setups.loc[sub.index, "confirm_ok"] = p >= thrs[key]
    confirmed = setups[setups["confirm_ok"]].reset_index(drop=True)

    # PURE M15 ANTI gate
    gated = confirmed[(confirmed["m15_dir"] == -confirmed["direction"]) &
                      (confirmed["m15_dir"] != 0)].reset_index(drop=True)
    print(f"  gated setups: {len(gated):,}", flush=True)

    print("  simulating RL exit (with reasons + bars_held) ...", flush=True)
    sim = sim_rl_exit_with_reasons(gated, sw, exit_mdl, exit_feats)
    hd = sim[sim["time"] >= HOLDOUT].reset_index(drop=True)
    print(f"  holdout trades: {len(hd):,}", flush=True)

    # 1) Exit reason distribution
    print(f"\n  ============ EXIT REASON DISTRIBUTION (2.4y holdout) ============")
    print(f"   {'reason':>10}   {'n':>6}  {'pct':>5}   {'WR%':>5}   {'PF':>5}   {'sumR':>7}   {'avg_R':>6}   {'avg bars':>8}   {'avg min':>7}")
    total = len(hd)
    for reason in ["ml_exit", "hard_sl", "max_hold"]:
        sub = hd[hd["exit_reason"] == reason]
        if len(sub) == 0: continue
        m = metrics(sub["pnl_R"].to_numpy())
        avg_bars = sub["bars_held"].mean()
        avg_min = avg_bars * 5   # M5 bars → minutes
        print(f"   {reason:>10}   {m['n']:>6,}  {m['n']/total*100:>4.1f}%   {m['wr']*100:>5.1f}   {m['pf']:>5.2f}   {m['sum_r']:>+7.0f}   {m['avg']:>+6.2f}   {avg_bars:>8.1f}   {avg_min:>7.0f}")

    print(f"\n  ============ BARS-HELD DISTRIBUTION (all 2.4y holdout trades) ============")
    bh = hd["bars_held"]
    print(f"    min:    {bh.min():>3} bars ({bh.min()*5:>4} min)")
    print(f"    p25:    {bh.quantile(0.25):>3.0f} bars ({bh.quantile(0.25)*5:>4.0f} min)")
    print(f"    median: {bh.median():>3.0f} bars ({bh.median()*5:>4.0f} min)")
    print(f"    mean:   {bh.mean():>5.1f} bars ({bh.mean()*5:>4.0f} min)")
    print(f"    p75:    {bh.quantile(0.75):>3.0f} bars ({bh.quantile(0.75)*5:>4.0f} min)")
    print(f"    p90:    {bh.quantile(0.90):>3.0f} bars ({bh.quantile(0.90)*5:>4.0f} min)")
    print(f"    max:    {bh.max():>3} bars ({bh.max()*5:>4} min)")
    print(f"    MAX_HOLD cap: {MAX_HOLD} bars ({MAX_HOLD*5} min = {MAX_HOLD*5//60}h{(MAX_HOLD*5)%60}m)")

    print(f"\n  ============ HOLD-DURATION BUCKETS (vs PF) ============")
    print(f"   {'bucket':>15}   {'n':>5}   {'pct':>5}   {'WR%':>5}   {'PF':>5}   {'sumR':>7}")
    buckets = [(0, 5, "≤25 min"), (5, 15, "25-75 min"),
               (15, 30, "75-150 min"), (30, 60, "150-300 min"),
               (60, 60, "=MAX_HOLD (5h)")]
    for lo, hi, name in buckets:
        if name == "=MAX_HOLD (5h)":
            sub = hd[hd["bars_held"] == 60]
        else:
            sub = hd[(hd["bars_held"] > lo) & (hd["bars_held"] <= hi)]
        if len(sub) == 0: continue
        m = metrics(sub["pnl_R"].to_numpy())
        print(f"   {name:>15}   {m['n']:>5,}   {m['n']/total*100:>4.1f}%   {m['wr']*100:>5.1f}   {m['pf']:>5.2f}   {m['sum_r']:>+7.0f}")

    # 2) Histogram plot
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].hist(hd["bars_held"], bins=range(1, 62), color="steelblue", edgecolor="black")
    axes[0].axvline(MAX_HOLD, color="red", linestyle="--", label=f"MAX_HOLD={MAX_HOLD}")
    axes[0].set_title("Bars held (M5 bars; 1 bar = 5 min)")
    axes[0].set_xlabel("bars"); axes[0].set_ylabel("count"); axes[0].legend(); axes[0].grid(alpha=0.3)
    counts = hd["exit_reason"].value_counts()
    axes[1].bar(counts.index, counts.values, color=["seagreen","crimson","goldenrod"])
    for i,(k,v) in enumerate(counts.items()):
        axes[1].text(i, v, f"{v:,}\n{v/total*100:.1f}%", ha="center", va="bottom")
    axes[1].set_title("Exit reason"); axes[1].set_ylabel("count"); axes[1].grid(alpha=0.3, axis="y")
    fig.suptitle(f"Oracle RL exit — duration + reason stats ({len(hd):,} holdout trades)")
    fig.tight_layout(); out = HERE / "exit_stats.png"; fig.savefig(out, dpi=110); plt.close(fig)
    print(f"\n  saved {out}")
    print(f"\n  TOTAL: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
