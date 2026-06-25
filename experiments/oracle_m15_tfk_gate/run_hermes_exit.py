"""
Oracle XAU setups + per-cluster confirm + M15 TFK ANTI gate
+ HERMES-STYLE EXIT (replacing Oracle's RL exit).

Hermes exit: SL=6 ATR (hard stop), TRAIL=2 ATR (trailing stop after MFE>=2ATR),
MAXH=60 M5 bars (= 5 hours, time-scaled from Hermes M1's MAXH=300 minutes).

Compares per-cluster + aggregated:
  - CURRENT LIVE   (cluster block C1+C2)
  - M15 TFK ANTI   (replace cluster block)
  - BOTH (cluster-allowed AND M15 ANTI)
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
SL, TRAIL, MAXH = 6.0, 2.0, 60   # Hermes-style exit (M5-scaled: 60 bars = 5h)
BLOCKED = {1, 2}
HOLDOUT = pd.Timestamp("2024-01-01")


@njit(cache=True)
def hermes_trail_sim(sw_idxs, dirs, O, H, L, C, atr, sp, SL, TRAIL, MAXH, n):
    m = len(sw_idxs); pnl = np.empty(m)
    for k in range(m):
        i = sw_idxs[k]; d = dirs[k]; a = atr[i]; ei = i + 1
        if ei >= n or not (a > 0): pnl[k] = 0.0; continue
        ep = O[ei]; hard = SL * a; trd = TRAIL * a; mf = 0.0
        end = min(ei + MAXH, n - 1); done = False; out_r = 0.0
        for j in range(ei, end + 1):
            fav = d * (C[j] - ep)
            if fav > mf: mf = fav
            if d == 1 and (ep - L[j]) >= hard: out_r = -SL - sp; done = True; break
            if d == -1 and (H[j] - ep) >= hard: out_r = -SL - sp; done = True; break
            if mf >= trd and (mf - fav) >= trd: out_r = (mf - trd) / a - sp; done = True; break
        if not done: out_r = d * (C[end] - ep) / a - sp
        pnl[k] = out_r
    return pnl


def metrics(r):
    r = np.asarray(r); r = r[np.isfinite(r)]
    if len(r) == 0: return None
    w, l = r[r > 0], r[r <= 0]; eq = np.cumsum(r)
    return dict(n=int(len(r)), wr=float((r > 0).mean()),
                pf=float(w.sum() / max(-l.sum(), 1e-9)),
                sum_r=float(r.sum()),
                max_dd_r=float((np.maximum.accumulate(eq) - eq).max()))


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
        m15[["time", "m15_dir"]].assign(time=m15["time"] + pd.Timedelta("15min")).sort_values("time"),
        on="time", direction="backward")
    return aligned["m15_dir"].fillna(0).to_numpy(np.int64)


def attach_cid(setups, swing_df, fp):
    times_sw = swing_df["time"].values.astype("datetime64[ns]")
    setup_t = setups["time"].values.astype("datetime64[ns]")
    sw_idx = np.searchsorted(times_sw, setup_t)
    sw_idx_safe = np.minimum(sw_idx, len(times_sw) - 1)
    exact = times_sw[sw_idx_safe] == setup_t
    setups = setups.copy()
    setups["sw_idx"] = np.where(exact, sw_idx_safe, -1)
    cid_per = np.full(len(swing_df), -1, dtype=np.int64)
    for _, row in fp.iterrows():
        s, e = int(row["start_idx"]), int(row["end_idx"])
        if 0 <= s < e <= len(swing_df): cid_per[s:e] = int(row["new_label"])
    setups["cid"] = np.where(setups["sw_idx"] >= 0,
                             cid_per[np.maximum(setups["sw_idx"], 0)], -1)
    return setups


def report_window(name, df, end_t):
    slices = [
        ("LAST WEEK",     end_t - pd.Timedelta(days=7)),
        ("LAST 30 DAYS",  end_t - pd.Timedelta(days=30)),
        ("LAST 90 DAYS",  end_t - pd.Timedelta(days=90)),
        ("HOLDOUT 2024+", HOLDOUT),
    ]
    print(f"\n  ============ {name} ============")
    for s_name, start in slices:
        sub = df[(df["time"] >= start) & (df["time"] <= end_t)]
        if len(sub) < 5: continue
        m = metrics(sub["pnl_R"].to_numpy())
        if m:
            tag = ""
            if m["pf"] >= 2.0 and m["sum_r"] > 0: tag = " <-- PF>=2"
            elif m["pf"] >= 1.5 and m["sum_r"] > 0: tag = " <-- PF>=1.5"
            print(f"    {s_name:>15}: n={m['n']:>5,}  WR={m['wr']*100:>5.1f}%  "
                  f"PF={m['pf']:>5.2f}  sumR={m['sum_r']:>+8.1f}  DD={m['max_dd_r']:>6.1f}{tag}")


def main():
    t0 = time.time()
    print("="*80); print("  Oracle setups + per-cluster confirm + M15 TFK ANTI gate + HERMES EXIT"); print("="*80)
    bundle = pickle.load(open(ORACLE_PKL, "rb"))
    mdls = bundle["mdls"]; thrs = bundle["thrs"]
    v72l_feats = bundle["v72l_feats"]

    sw = pd.read_csv(SWING, parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    setup_files = sorted(glob.glob(SETUP_GLOB))
    setups = pd.concat([pd.read_csv(f, parse_dates=["time"]) for f in setup_files],
                       ignore_index=True).sort_values("time").reset_index(drop=True)
    fp = pd.read_csv(FINGERPRINTS, parse_dates=["center_time"])

    # ATR (M5)
    O = sw["open"].to_numpy(float); H = sw["high"].to_numpy(float)
    L = sw["low"].to_numpy(float);  C = sw["close"].to_numpy(float)
    prev_c = np.concatenate([[C[0]], C[:-1]])
    tr = np.maximum(H-L, np.maximum(np.abs(H-prev_c), np.abs(L-prev_c)))
    atr = pd.Series(tr).rolling(14, min_periods=14).mean().to_numpy()
    n = len(sw); sp = SPREAD_USD / np.nanmedian(atr)

    setups = attach_cid(setups, sw, fp)
    setups = setups[setups["cid"] >= 0].reset_index(drop=True)

    # M15 TFK direction at each setup
    print("  computing M15 TFK ...", flush=True)
    m15 = m15_dir_at_bars(sw)
    times_sw = sw["time"].values.astype("datetime64[ns]")
    setup_t = setups["time"].values.astype("datetime64[ns]")
    sw_idx = np.searchsorted(times_sw, setup_t)
    sw_idx_safe = np.minimum(sw_idx, len(times_sw) - 1)
    exact = times_sw[sw_idx_safe] == setup_t
    setups["sw_idx"] = np.where(exact, sw_idx_safe, -1)
    setups["m15_dir"] = np.where(exact, m15[sw_idx_safe], 0)

    # Per-cluster confirm
    setups["confirm_ok"] = False
    for cid in sorted(setups["cid"].unique()):
        key = (int(cid), "RL")
        if key not in mdls: continue
        sub = setups[setups["cid"] == cid]
        X = sub[v72l_feats].fillna(0).to_numpy(np.float32)
        p = mdls[key].predict_proba(X)[:, 1]
        setups.loc[sub.index, "confirm_ok"] = p >= thrs[key]
    confirmed = setups[setups["confirm_ok"] & (setups["sw_idx"] >= 0)].reset_index(drop=True)
    print(f"  confirmed setups: {len(confirmed):,}", flush=True)

    # Hermes-style trail exit on ALL confirmed
    print(f"  simulating Hermes exit (SL={SL} TRAIL={TRAIL} MAXH={MAXH} M5 bars) ...", flush=True)
    sw_idxs = confirmed["sw_idx"].to_numpy(np.int64)
    dirs = confirmed["direction"].to_numpy(np.int64)
    pnls = hermes_trail_sim(sw_idxs, dirs, O, H, L, C, atr, sp, SL, TRAIL, MAXH, n)
    confirmed["pnl_R"] = pnls

    # Filter to holdout
    hd = confirmed[confirmed["time"] >= HOLDOUT].reset_index(drop=True)
    end_t = hd["time"].max()

    # Per-cluster breakdown
    print(f"\n  ============ Per-cluster 2.4y holdout (Hermes exit) ============")
    print(f"  {'cid':>4} | {'NO GATE':<35} | {'live?':<8} | {'M15 ANTI':<35}")
    def fmt(m):
        if not m: return "       —"
        return f"n={m['n']:>4} WR={m['wr']*100:>4.1f}% PF={m['pf']:>5.2f} sumR={m['sum_r']:>+6.0f} DD={m['max_dd_r']:>4.0f}"
    for cid in [0, 1, 2, 3, 4]:
        cdf = hd[hd["cid"] == cid].reset_index(drop=True)
        if len(cdf) < 5: continue
        m_all = metrics(cdf["pnl_R"].to_numpy())
        anti = (cdf["m15_dir"] == -cdf["direction"]) & (cdf["m15_dir"] != 0)
        m_anti = metrics(cdf.loc[anti, "pnl_R"].to_numpy())
        blocked = "BLOCKED" if cid in BLOCKED else "allowed"
        print(f"  c{cid:>3} | {fmt(m_all):<35} | {blocked:<8} | {fmt(m_anti):<35}")

    # Aggregate variants
    def pf_block(df, name):
        m = metrics(df["pnl_R"].to_numpy()) if len(df) > 0 else None
        if m:
            tag = ""
            if m["pf"] >= 2.0 and m["sum_r"] > 0: tag = " <-- PF>=2"
            elif m["pf"] >= 1.5 and m["sum_r"] > 0: tag = " <-- PF>=1.5"
            print(f"    {name:>40}: n={m['n']:>5,}  WR={m['wr']*100:>5.1f}%  "
                  f"PF={m['pf']:>5.2f}  sumR={m['sum_r']:>+7.0f}  DD={m['max_dd_r']:>5.0f}{tag}")
        else:
            print(f"    {name:>40}: (empty)")
    anti_all = (hd["m15_dir"] == -hd["direction"]) & (hd["m15_dir"] != 0)
    print(f"\n  ============ Aggregated holdout (Hermes exit) ============")
    pf_block(hd,                                             "NO GATE (all 5 clusters)")
    pf_block(hd[~hd["cid"].isin(BLOCKED)],                   "CURRENT LIVE (block C1+C2)")
    pf_block(hd[anti_all],                                   "PURE M15 ANTI gate")
    pf_block(hd[~hd["cid"].isin(BLOCKED) & anti_all],        "BOTH: cluster-allowed AND M15 ANTI")

    # Windowed views for the PURE M15 ANTI version
    new_df = hd[anti_all].reset_index(drop=True)
    report_window("PURE M15 ANTI gate + Hermes exit", new_df, end_t)
    live_df = hd[~hd["cid"].isin(BLOCKED)].reset_index(drop=True)
    report_window("CURRENT LIVE (cluster block C1+C2) + Hermes exit", live_df, end_t)

    # Equity plot
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 5))
    for label, df, color in [
        ("CURRENT (C1+C2 block)", live_df, "steelblue"),
        ("PURE M15 ANTI",         new_df,  "darkgreen"),
        ("BOTH",                  hd[~hd["cid"].isin(BLOCKED) & anti_all], "purple"),
    ]:
        if len(df) == 0: continue
        sub = df.sort_values("time")
        eq = np.cumsum(sub["pnl_R"].to_numpy())
        m = metrics(sub["pnl_R"].to_numpy())
        ax.plot(eq, lw=1.0, color=color,
                label=f"{label}  n={m['n']}  PF={m['pf']:.2f}  sumR={m['sum_r']:+.0f}  DD={m['max_dd_r']:.0f}")
    ax.set_title("Oracle setups + Hermes exit — gate comparison (honest 2024+ holdout)")
    ax.set_xlabel("trade #"); ax.set_ylabel("cum R"); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); out = HERE / "equity_hermes_exit.png"; fig.savefig(out, dpi=110); plt.close(fig)
    print(f"\n  saved {out}")
    print(f"\n  TOTAL: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
