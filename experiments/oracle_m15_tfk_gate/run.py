"""
Oracle XAU — replace 4h regime-cluster gate with M15 TFK color gate.

For each Oracle setup candidate:
  - Compute M15 TFK committed direction at setup time (resample M5→M15, run
    TFK on M15, forward-fill back to M5 grid).
  - Gate: keep only setups where setup direction matches M15 TFK direction
    (long-only when M15 TFK is green / +1; short-only when red / -1).
  - Simulate trade forward on M5 swing bars with fixed bracket: TP=2*ATR,
    SL=1*ATR, max_hold=60 M5 bars (5h).

Run on:
  - LAST WEEK (2026-04-24 → 2026-05-01)
  - FULL HOLDOUT (2024-01-01 → 2026-05-01) for context

Compare:
  A) NO gate (raw setups)
  B) M15 TFK gate (the user's idea)
"""
from __future__ import annotations
import sys, time, glob
from pathlib import Path
import numpy as np, pandas as pd
from numba import njit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "products/hermes_xau"))
from tfk import compute_tfk

SWING = ROOT / "data" / "swing_v5_xauusd.csv"
SETUP_GLOB = str(ROOT / "data" / "setups_*_v72l.csv")
TP_ATR = 2.0; SL_ATR = 1.0; MAX_HOLD_BARS = 60

# Spread in USD (M5 XAU)
SPREAD_USD = 0.30


@njit(cache=True)
def simulate(entries_idx, dirs, entry_price, atr_at, O, H, L, C, sp, TP, SL, MH, n):
    m = len(entries_idx); pnl = np.empty(m); xit = np.empty(m, np.int64)
    for k in range(m):
        i = entries_idx[k]; d = dirs[k]; a = atr_at[k]; ep = entry_price[k]
        if i+1 >= n or not(a > 0):
            pnl[k] = 0.0; xit[k] = i+1; continue
        end = min(i+1+MH, n-1); done = False; out_r = 0.0
        for j in range(i+1, end+1):
            if d == 1:
                if (H[j] - ep) >= TP*a: out_r = TP - sp/a; xit[k]=j; done=True; break
                if (ep - L[j]) >= SL*a: out_r = -SL - sp/a; xit[k]=j; done=True; break
            else:
                if (ep - L[j]) >= TP*a: out_r = TP - sp/a; xit[k]=j; done=True; break
                if (H[j] - ep) >= SL*a: out_r = -SL - sp/a; xit[k]=j; done=True; break
        if not done:
            out_r = d * (C[end] - ep) / a - sp/a
            xit[k] = end
        pnl[k] = out_r
    return pnl, xit


def metrics(r):
    r = np.asarray(r); r = r[np.isfinite(r)]
    if len(r) == 0: return None
    w, l = r[r > 0], r[r <= 0]; eq = np.cumsum(r)
    return dict(n=int(len(r)), wr=float((r>0).mean()),
                pf=float(w.sum()/max(-l.sum(), 1e-9)),
                sum_r=float(r.sum()),
                max_dd_r=float((np.maximum.accumulate(eq)-eq).max()))


def m15_tfk_direction(swing_df):
    """Compute M15 TFK from M5 swing data, return per-M5-bar M15 direction
    using causal forward-fill (each M5 bar sees the *completed* M15 bar)."""
    s = swing_df.set_index("time")
    # Resample to M15 (each M15 bar's close = last M5 close in the window)
    m15 = s["close"].resample("15min").last().dropna().to_frame(name="close")
    m15["open"] = s["open"].resample("15min").first().reindex(m15.index)
    m15["high"] = s["high"].resample("15min").max().reindex(m15.index)
    m15["low"] = s["low"].resample("15min").min().reindex(m15.index)
    m15["tick_volume"] = s["tick_volume"].resample("15min").sum().reindex(m15.index).fillna(0)
    m15 = m15.reset_index()
    m15.rename(columns={"index":"time"}, inplace=True)
    print(f"    M15 bars: {len(m15):,}", flush=True)
    tfk_m15 = compute_tfk(m15, flip_bars=5, color_confirm=8)
    m15["m15_dir"] = tfk_m15["committed_dir"].to_numpy()
    # M15 bar close time = bar start + 15min. Each M15 dir is valid AFTER its
    # close. Forward-fill onto M5 grid with causal alignment.
    m15_dir_at = pd.merge_asof(
        swing_df[["time"]].sort_values("time"),
        m15[["time","m15_dir"]].assign(time=m15["time"] + pd.Timedelta("15min")).sort_values("time"),
        on="time", direction="backward"
    )
    return m15_dir_at["m15_dir"].fillna(0).to_numpy(np.int64)


def main():
    t0 = time.time()
    print("="*78); print("  Oracle XAU — M15 TFK gate sim (replace regime cluster)"); print("="*78)

    # 1. Load swing M5 bars
    print("\n  loading swing M5 ...", flush=True)
    sw = pd.read_csv(SWING, parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    print(f"    swing rows: {len(sw):,}  {sw.time.iloc[0]} → {sw.time.iloc[-1]}", flush=True)
    times_sw = sw["time"].to_numpy()
    O = sw["open"].to_numpy(float); H = sw["high"].to_numpy(float)
    L = sw["low"].to_numpy(float);  C = sw["close"].to_numpy(float)
    n = len(sw)

    # 2. Compute M15 TFK direction at each M5 bar
    print("\n  computing M15 TFK ...", flush=True)
    m15_dir_per_m5 = m15_tfk_direction(sw)
    pos = (m15_dir_per_m5 ==  1).sum()
    neg = (m15_dir_per_m5 == -1).sum()
    print(f"    M15 dir at M5 bars: +1 {pos:,} | -1 {neg:,} | 0 {len(m15_dir_per_m5)-pos-neg:,}", flush=True)

    # 3. Load Oracle setups
    print("\n  loading Oracle setups ...", flush=True)
    setup_files = sorted(glob.glob(SETUP_GLOB))
    all_setups = []
    for f in setup_files:
        df = pd.read_csv(f, parse_dates=["time"],
                         usecols=["time","direction","rule","atr","entry_price","label"])
        all_setups.append(df)
    setups = pd.concat(all_setups, ignore_index=True).sort_values("time").reset_index(drop=True)
    print(f"    setups: {len(setups):,}  {setups.time.iloc[0]} → {setups.time.iloc[-1]}", flush=True)

    # 4. Align setups to swing index via searchsorted, attach M15 dir
    setup_t = setups["time"].values.astype("datetime64[ns]")
    times_sw_ns = times_sw.astype("datetime64[ns]")
    sw_idx = np.searchsorted(times_sw_ns, setup_t)
    in_range = sw_idx < len(times_sw_ns)
    sw_idx_safe = np.minimum(sw_idx, len(times_sw_ns)-1)
    exact_match = (times_sw_ns[sw_idx_safe] == setup_t) & in_range
    setups["sw_idx"] = np.where(exact_match, sw_idx_safe, -1)
    valid = (setups["sw_idx"] >= 0) & (setups["atr"] > 0)
    print(f"    exact_match: {int(exact_match.sum()):,}", flush=True)
    setups = setups[valid].reset_index(drop=True)
    setups["m15_dir"] = m15_dir_per_m5[setups["sw_idx"].to_numpy()]
    print(f"    after alignment: {len(setups):,}", flush=True)

    # 5. Define test slices: last week, 30 days, 90 days, full holdout
    end_t = setups["time"].max()
    slices = [
        ("LAST WEEK",       end_t - pd.Timedelta(days=7)),
        ("LAST 30 DAYS",    end_t - pd.Timedelta(days=30)),
        ("LAST 90 DAYS",    end_t - pd.Timedelta(days=90)),
        ("HOLDOUT (2024+)", pd.Timestamp("2024-01-01")),
    ]
    sp = SPREAD_USD  # absolute USD; converted to /R per setup using its own atr

    for label, start_t in slices:
        sel_window = setups[(setups["time"] >= start_t) & (setups["time"] <= end_t)].reset_index(drop=True)
        if len(sel_window) < 5: continue
        ei = sel_window["sw_idx"].to_numpy(np.int64)
        dr = sel_window["direction"].to_numpy(np.int64)
        ep = sel_window["entry_price"].to_numpy(float)
        at = sel_window["atr"].to_numpy(float)
        pnl, xit = simulate(ei, dr, ep, at, O, H, L, C,
                            sp, TP_ATR, SL_ATR, MAX_HOLD_BARS, n)
        m15 = sel_window["m15_dir"].to_numpy(np.int64)
        gate_pass = (m15 == dr) & (m15 != 0)

        m_all   = metrics(pnl)
        m_gate  = metrics(pnl[gate_pass])
        m_anti  = metrics(pnl[(m15 == -dr) & (m15 != 0)])
        m_zero  = metrics(pnl[m15 == 0])

        print(f"\n  ===== {label}  ({str(start_t)[:10]} → {str(end_t)[:10]})  total setups: {len(sel_window):,} =====")
        def fmt(name, m, pct_of_total=None):
            if m is None: return f"    {name:>18}:  (no trades)"
            pct = f"  [{m['n']/max(len(sel_window),1)*100:>4.1f}% of total]" if pct_of_total is None else ""
            return (f"    {name:>18}:  n={m['n']:>5}  WR={m['wr']*100:>5.1f}%  "
                    f"PF={m['pf']:>5.2f}  sumR={m['sum_r']:>+7.1f}  DD={m['max_dd_r']:>5.1f}{pct}")
        print(fmt("A) NO GATE", m_all))
        print(fmt("B) M15 TFK ALIGN", m_gate))
        print(fmt("  M15 TFK ANTI", m_anti))
        print(fmt("  M15 TFK ZERO", m_zero))

    # 6. Plot equity for HOLDOUT, with vs without gate
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    sel = setups[setups["time"] >= pd.Timestamp("2024-01-01")].reset_index(drop=True)
    ei = sel["sw_idx"].to_numpy(np.int64); dr = sel["direction"].to_numpy(np.int64)
    ep = sel["entry_price"].to_numpy(float); at = sel["atr"].to_numpy(float)
    pnl, xit = simulate(ei, dr, ep, at, O, H, L, C, sp, TP_ATR, SL_ATR, MAX_HOLD_BARS, n)
    gate = (sel["m15_dir"].to_numpy(np.int64) == dr) & (sel["m15_dir"].to_numpy() != 0)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    eq_all = np.cumsum(pnl)
    eq_gate = np.cumsum(pnl[gate])
    axes[0].plot(eq_all, lw=1.0, color="gray", label=f"NO GATE n={len(pnl):,}")
    axes[0].plot(eq_gate, lw=1.0, color="darkgreen", label=f"M15 TFK ALIGN n={gate.sum():,}")
    axes[0].set_title(f"Holdout 2024-01-01 → end\n"
                      f"NO GATE PF={metrics(pnl)['pf']:.2f}  sumR={metrics(pnl)['sum_r']:+.0f} | "
                      f"M15 GATE PF={metrics(pnl[gate])['pf']:.2f}  sumR={metrics(pnl[gate])['sum_r']:+.0f}")
    axes[0].set_xlabel("trade #"); axes[0].set_ylabel("cum R"); axes[0].grid(alpha=0.3); axes[0].legend()
    # Last week only
    lw = sel[sel["time"] >= end_t - pd.Timedelta(days=7)].reset_index(drop=True)
    if len(lw) >= 5:
        ei2 = lw["sw_idx"].to_numpy(np.int64); dr2 = lw["direction"].to_numpy(np.int64)
        ep2 = lw["entry_price"].to_numpy(float); at2 = lw["atr"].to_numpy(float)
        pnl2, _ = simulate(ei2, dr2, ep2, at2, O, H, L, C, sp, TP_ATR, SL_ATR, MAX_HOLD_BARS, n)
        gate2 = (lw["m15_dir"].to_numpy(np.int64) == dr2) & (lw["m15_dir"].to_numpy() != 0)
        axes[1].plot(np.cumsum(pnl2), lw=1.2, color="gray", label=f"NO GATE n={len(pnl2)}")
        axes[1].plot(np.cumsum(pnl2[gate2]), lw=1.2, color="darkgreen", label=f"M15 TFK ALIGN n={gate2.sum()}")
        m_lw = metrics(pnl2); m_lwg = metrics(pnl2[gate2])
        title = "Last week\n"
        if m_lw: title += f"NO GATE PF={m_lw['pf']:.2f} sumR={m_lw['sum_r']:+.1f} | "
        if m_lwg: title += f"M15 GATE PF={m_lwg['pf']:.2f} sumR={m_lwg['sum_r']:+.1f}"
        axes[1].set_title(title); axes[1].set_xlabel("trade #"); axes[1].set_ylabel("cum R")
        axes[1].grid(alpha=0.3); axes[1].legend()
    fig.suptitle("Oracle XAU — M15 TFK gate vs no gate (TP=2R SL=1R bracket)")
    fig.tight_layout(); out = HERE / "equity.png"; fig.savefig(out, dpi=110); plt.close(fig)
    print(f"\n  saved {out}")
    print(f"\n  TOTAL: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
