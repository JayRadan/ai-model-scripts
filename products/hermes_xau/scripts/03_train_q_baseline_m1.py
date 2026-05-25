"""
XAU M1 pipeline on FULL Dukascopy M1 data (2018-2026).
Train: 2018-2024-12-12 (~6.9 years, ~2.4M bars)
Test:  2024-12-12 → 2026-05-01 (~17 months)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

from tfk import compute_tfk

CUTOFF = pd.Timestamp("2024-12-12 00:00:00")
SL_ATR = 6.0
TRAIL = 3.0
MAX_HOLD = 300
NEAR = 0.25
PROFIT_TO_BE_R = 1.0
MAX_CONCURRENT = 4
SWITCH_DELTA = 0.5
COOLDOWN_BARS = 5
XAU_SPREAD_USD = 0.30


def add_features(m1: pd.DataFrame) -> pd.DataFrame:
    df = m1.copy()
    c = df["close"]; h = df["high"]; l = df["low"]
    prev = c.shift(1).fillna(c.iloc[0])
    tr = pd.concat([(h - l), (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14, min_periods=14).mean()
    delta = c.diff()
    up = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    dn = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
    rs = up / dn.replace(0, np.nan)
    df["rsi14"] = 100 - 100 / (1 + rs)
    for n in (20, 50, 100, 200):
        ema = c.ewm(span=n, adjust=False).mean()
        df[f"dist_ema{n}"] = (c - ema) / df["atr14"]
    for n in (5, 10, 20):
        df[f"slope{n}"] = (c - c.shift(n)) / df["atr14"]
    df["atr_ratio"] = df["atr14"] / df["atr14"].rolling(50, min_periods=50).mean()
    for tf_name, tf in [("m5", "5min"), ("m15", "15min"), ("h1", "60min")]:
        g = df.set_index("time")[["high", "low", "close"]].resample(tf).agg({
            "high": "max", "low": "min", "close": "last"
        }).dropna()
        c_htf = g["close"]
        delta = c_htf.diff()
        up = delta.clip(lower=0).rolling(14, min_periods=14).mean()
        dn = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
        rs = up / dn.replace(0, np.nan)
        g[f"{tf_name}_rsi14"] = 100 - 100 / (1 + rs)
        g[f"{tf_name}_slope5"] = (c_htf - c_htf.shift(5))
        g[f"{tf_name}_ema50_dist"] = c_htf - c_htf.ewm(span=50, adjust=False).mean()
        out = g[[c_ for c_ in g.columns if c_ not in ("high", "low", "close")]]
        out = out.reindex(df["time"], method="ffill")
        for col in out.columns:
            df[col] = out[col].to_numpy()
    return df


Q_FEATS_M1 = [
    "dist_at_signal", "dist_abs", "regime_age", "bar_range_atr",
    "force", "velocity", "x_est", "regime_w", "trend_raw", "trend",
    "rsi14", "dist_ema20", "dist_ema50", "dist_ema100", "dist_ema200",
    "slope5", "slope10", "slope20", "atr_ratio",
    "m5_rsi14", "m5_slope5", "m5_ema50_dist",
    "m15_rsi14", "m15_slope5", "m15_ema50_dist",
    "h1_rsi14", "h1_slope5", "h1_ema50_dist",
    "committed_dir",
]


def simulate_label(entry_idx, direction, C, H, L, O, atr_at_entry,
                   spread_R=0.0, sl_atr=SL_ATR, trail_atr=TRAIL, max_hold=MAX_HOLD):
    n = len(C)
    if entry_idx >= n - 1 or not (np.isfinite(atr_at_entry) and atr_at_entry > 0):
        return np.nan, "skip", entry_idx
    ep = O[entry_idx]; a = atr_at_entry
    hard = sl_atr * a; trail = trail_atr * a; max_favor = 0.0
    end = min(entry_idx + max_hold, n - 1)
    for k in range(entry_idx, end + 1):
        favor_now = direction * (C[k] - ep)
        if favor_now > max_favor: max_favor = favor_now
        if direction == 1:
            if (ep - L[k]) >= hard: return -sl_atr - spread_R, "hard_sl", k
        else:
            if (H[k] - ep) >= hard: return -sl_atr - spread_R, "hard_sl", k
        if max_favor >= trail:
            if (max_favor - favor_now) >= trail:
                return (max_favor - trail) / a - spread_R, "trail", k
    return direction * (C[end] - ep) / a - spread_R, "max_hold", end


def bars_in_regime_array(cdir):
    n = len(cdir); out = np.zeros(n, dtype=np.int64); cur = 1
    for i in range(1, n):
        if cdir[i] == cdir[i - 1]: cur += 1
        else: cur = 1
        out[i] = cur
    return out


def metrics(r):
    r = r[np.isfinite(r)]
    if len(r) == 0: return None
    w, l = r[r > 0], r[r <= 0]
    pf = float(w.sum() / max(-l.sum(), 1e-9))
    eq = np.cumsum(r)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    return {"n": int(len(r)), "wr": float((r > 0).mean()), "pf": pf,
            "sum_r": float(r.sum()), "max_dd_r": dd, "avg_r": float(r.mean())}


def main():
    t0 = time.time()
    print(f"\n{'='*72}\n  XAU M1 FULL — 2018-2026 Dukascopy\n{'='*72}", flush=True)

    print("\n[1/4] loading M1 full ...", flush=True)
    m1 = pd.read_parquet(ROOT / "data" / "m1_xau_full.parquet")
    m1 = m1.sort_values("time").reset_index(drop=True)
    print(f"  bars: {len(m1):,}  range: {m1.time.iloc[0]} → {m1.time.iloc[-1]}", flush=True)

    print("\n[2/4] computing TFK + features ...", flush=True)
    tfk_out = compute_tfk(m1)
    df = add_features(tfk_out)

    O = df["open"].to_numpy(np.float64); H = df["high"].to_numpy(np.float64)
    L = df["low"].to_numpy(np.float64); C = df["close"].to_numpy(np.float64)
    line = df["tfk_line"].to_numpy(np.float64)
    cdir = df["committed_dir"].to_numpy(np.int64)
    times = df["time"].to_numpy()
    atr = df["atr14"].to_numpy(np.float64)
    bir = bars_in_regime_array(cdir)
    n = len(df)
    median_atr = np.nanmedian(atr)
    spread_R = XAU_SPREAD_USD / median_atr
    print(f"  bars={n:,}  median ATR={median_atr:.3f}  spread={XAU_SPREAD_USD}$ ≈ {spread_R:.3f}R", flush=True)

    print("\n[3/4] building candidates + labels ...", flush=True)
    dist_signed = np.where(atr > 0, (C - line) / atr, 0.0)
    dist_abs = np.abs(dist_signed)
    valid = (np.isfinite(atr) & (atr > 0))
    valid[:200] = False; valid[-(MAX_HOLD + 1):] = False
    valid &= (dist_abs <= NEAR) & (cdir != 0)
    idxs = np.where(valid)[0]
    dirs = cdir[idxs].copy()
    print(f"  candidates: {len(idxs):,}", flush=True)

    print(f"  labelling (TRAIL={TRAIL}R, no spread for training) ...", flush=True)
    pnl = np.zeros(len(idxs), dtype=np.float32)
    pnl_spread = np.zeros(len(idxs), dtype=np.float32)
    exit_idxs = np.zeros(len(idxs), dtype=np.int32)
    for k, i in enumerate(idxs):
        d = int(dirs[k])
        r, _, xi = simulate_label(i + 1, d, C, H, L, O, atr[i], spread_R=0.0)
        rs, _, _ = simulate_label(i + 1, d, C, H, L, O, atr[i], spread_R=spread_R)
        pnl[k] = r if np.isfinite(r) else 0.0
        pnl_spread[k] = rs if np.isfinite(rs) else 0.0
        exit_idxs[k] = xi
        if k % 200000 == 0 and k > 0: print(f"    {k:,}/{len(idxs):,}", flush=True)

    bare = metrics(pnl)
    bare_s = metrics(pnl_spread)
    print(f"  BARE no-spread: PF={bare['pf']:.2f} sumR={bare['sum_r']:+.0f} WR={bare['wr']*100:.1f}% avgR={bare['avg_r']:+.3f}", flush=True)
    print(f"  BARE +spread  : PF={bare_s['pf']:.2f} sumR={bare_s['sum_r']:+.0f} WR={bare_s['wr']*100:.1f}% avgR={bare_s['avg_r']:+.3f}", flush=True)

    extra = pd.DataFrame({
        "dist_at_signal": dist_signed[idxs], "dist_abs": dist_abs[idxs],
        "regime_age": bir[idxs],
        "bar_range_atr": (H[idxs] - L[idxs]) / np.maximum(atr[idxs], 1e-9),
    })
    avail_feats = [f for f in Q_FEATS_M1 if f in df.columns]
    swing_feats = df.iloc[idxs][[f for f in avail_feats if f not in extra.columns]].reset_index(drop=True)
    X_all = pd.concat([extra.reset_index(drop=True), swing_feats], axis=1)
    feat_cols = [c for c in Q_FEATS_M1 if c in X_all.columns]
    print(f"  features: {len(feat_cols)}", flush=True)

    times_at_idx = times[idxs]
    train_m = times_at_idx < np.datetime64(CUTOFF)
    test_m = ~train_m
    print(f"  train={train_m.sum():,}  test={test_m.sum():,}", flush=True)

    print("\n[4/4] fitting Q + executing ...", flush=True)
    from xgboost import XGBRegressor
    mdl = XGBRegressor(n_estimators=600, max_depth=5, learning_rate=0.04,
                       subsample=0.85, colsample_bytree=0.85,
                       min_child_weight=10, reg_lambda=1.0,
                       objective="reg:squarederror", tree_method="hist",
                       random_state=42, verbosity=0)
    mdl.fit(X_all.loc[train_m, feat_cols].fillna(0).to_numpy(np.float32), pnl[train_m])
    q_te = mdl.predict(X_all.loc[test_m, feat_cols].fillna(0).to_numpy(np.float32))

    test_idxs = idxs[test_m]
    test_dirs = dirs[test_m]
    test_pnl = pnl[test_m]; test_pnl_s = pnl_spread[test_m]
    test_exits = exit_idxs[test_m]
    test_times = times_at_idx[test_m]
    span_days = max((test_times[-1] - test_times[0]).astype("timedelta64[D]").astype(int), 1)
    q_by_idx = {int(test_idxs[k]): float(q_te[k]) for k in range(len(test_idxs))}
    info = {int(test_idxs[k]): (int(test_dirs[k]), float(test_pnl[k]), float(test_pnl_s[k]), int(test_exits[k]))
            for k in range(len(test_idxs))}

    def run_exec(thr, use_spread=False):
        active = []; executed = []
        last_open_per_dir = {-1: -10**9, +1: -10**9}
        bar_start = int(test_idxs[0]); bar_end = min(int(test_idxs[-1]) + MAX_HOLD + 1, n)
        sp_R = spread_R if use_spread else 0.0
        for i in range(bar_start, bar_end):
            still = []
            for t in active:
                if i < t["entry_idx"]: still.append(t); continue
                if i > min(t["entry_idx"] + MAX_HOLD, n - 1):
                    cp = C[min(t["entry_idx"] + MAX_HOLD, n - 1)]
                    t["pnl_R"] = float(t["direction"] * (cp - t["ep"]) / t["a"] - sp_R)
                    t["exit_idx"] = min(t["entry_idx"] + MAX_HOLD, n - 1)
                    t["exit_reason"] = "max_hold"; executed.append(t); continue
                d = t["direction"]; ep = t["ep"]; a = t["a"]
                favor_now = d * (C[i] - ep)
                if favor_now > t["max_favor"]: t["max_favor"] = favor_now
                sl_r = t["sl_r"]; hit = False
                if sl_r == 0:
                    if d == 1 and L[i] <= ep:
                        t["pnl_R"] = -sp_R; t["exit_idx"] = i; t["exit_reason"] = "be_stop"; hit = True
                    elif d == -1 and H[i] >= ep:
                        t["pnl_R"] = -sp_R; t["exit_idx"] = i; t["exit_reason"] = "be_stop"; hit = True
                else:
                    dist_r = abs(sl_r) * a
                    if d == 1 and (ep - L[i]) >= dist_r:
                        t["pnl_R"] = float(sl_r - sp_R); t["exit_idx"] = i; t["exit_reason"] = "hard_sl"; hit = True
                    elif d == -1 and (H[i] - ep) >= dist_r:
                        t["pnl_R"] = float(sl_r - sp_R); t["exit_idx"] = i; t["exit_reason"] = "hard_sl"; hit = True
                if hit: executed.append(t); continue
                trail_d = TRAIL * a
                if t["max_favor"] >= trail_d and (t["max_favor"] - favor_now) >= trail_d:
                    t["pnl_R"] = float((t["max_favor"] - trail_d) / a - sp_R)
                    t["exit_idx"] = i; t["exit_reason"] = "trail"
                    executed.append(t); continue
                still.append(t)
            active = still
            if i not in info: continue
            if q_by_idx[i] < thr: continue
            direction, _, _, _ = info[i]
            if i - last_open_per_dir[direction] < COOLDOWN_BARS: continue
            for t in active:
                if t["sl_r"] == 0: continue
                a_ = t["a"]; ep_ = t["ep"]; d_ = t["direction"]
                cur_R = d_ * (C[i] - ep_) / a_
                if cur_R >= PROFIT_TO_BE_R: t["sl_r"] = 0
            entry_idx = i + 1
            if entry_idx >= n: continue
            a_new = atr[i]
            if not (np.isfinite(a_new) and a_new > 0): continue
            ep_new = O[entry_idx]
            if len(active) >= MAX_CONCURRENT:
                worst = min(active, key=lambda x: x["q"])
                if q_by_idx[i] >= worst["q"] + SWITCH_DELTA:
                    cp = C[i]; ep_ = worst["ep"]; d_ = worst["direction"]; a_ = worst["a"]
                    worst["pnl_R"] = float(d_ * (cp - ep_) / a_ - sp_R)
                    worst["exit_idx"] = i; worst["exit_reason"] = "switch_close"
                    executed.append(worst); active.remove(worst)
                else: continue
            new_trade = {
                "signal_idx": int(i), "entry_idx": int(entry_idx),
                "direction": int(direction), "ep": float(ep_new), "a": float(a_new),
                "sl_r": float(-SL_ATR), "max_favor": 0.0, "q": float(q_by_idx[i]),
                "time": pd.Timestamp(times[i]),
                "exit_idx": int(entry_idx), "exit_reason": None, "pnl_R": None,
            }
            active.append(new_trade)
            last_open_per_dir[direction] = i
        for t in active:
            end_bar = min(t["entry_idx"] + MAX_HOLD, n - 1)
            cp = C[end_bar]
            t["pnl_R"] = float(t["direction"] * (cp - t["ep"]) / t["a"] - sp_R)
            t["exit_idx"] = end_bar; t["exit_reason"] = "end_of_data"
            executed.append(t)
        return executed

    print(f"\n  Q sweep — NO SPREAD vs WITH SPREAD ({XAU_SPREAD_USD}$ ≈ {spread_R:.3f}R/trade):")
    print(f"  {'thr':>5}  {'n':>6} {'WR%':>5} {'PF(0)':>6} {'sumR(0)':>9} | {'PF(s)':>6} {'sumR(s)':>9} {'DD(s)':>6} {'avgR(s)':>7} {'trd/d':>5}", flush=True)
    rows = []
    for thr in [-1.0, 0.0, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]:
        ex0 = run_exec(thr, use_spread=False)
        exs = run_exec(thr, use_spread=True)
        if not ex0: continue
        m0 = metrics(pd.DataFrame(ex0)["pnl_R"].to_numpy())
        ms = metrics(pd.DataFrame(exs)["pnl_R"].to_numpy())
        tpd = m0["n"] / span_days
        rows.append({"thr": thr, "no_spread": m0, "with_spread": ms, "trd_per_day": tpd})
        print(f"  {thr:>5.2f}  {m0['n']:>6,} {m0['wr']*100:>5.1f} "
              f"{m0['pf']:>6.2f} {m0['sum_r']:>+9.1f} | "
              f"{ms['pf']:>6.2f} {ms['sum_r']:>+9.1f} {ms['max_dd_r']:>6.1f} {ms['avg_r']:>+7.3f} {tpd:>5.1f}", flush=True)

    # Save trades at best Q≥1.0 with spread for charting
    best = run_exec(1.0, use_spread=True)
    if best:
        ex_df = pd.DataFrame(best)
        ex_df["entry_idx"] = ex_df["entry_idx"].astype(int)
        ex_df["exit_idx"] = ex_df["exit_idx"].astype(int)
        ex_df["direction"] = ex_df["direction"].astype(int)
        out_dir = HERE / "trade_charts"; out_dir.mkdir(exist_ok=True)
        ex_df.to_csv(out_dir / "xau_m1_full_q1.0_trades.csv", index=False)
        # Equity plot
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (a0, a1) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={"height_ratios": [3, 2]})
        s = ex_df.sort_values("time").reset_index(drop=True)
        eq = np.cumsum(s["pnl_R"].to_numpy())
        a0.plot(s["time"], eq, lw=1.0, color="#00C896")
        a0.fill_between(s["time"], 0, eq, alpha=0.10, color="#00C896")
        a0.axhline(0, color="black", lw=0.5)
        m = metrics(ex_df["pnl_R"].to_numpy())
        a0.set_title(f"XAU M1 FULL — Q≥1.0 with spread ({XAU_SPREAD_USD}$) — n={m['n']} PF={m['pf']:.2f} sumR={m['sum_r']:+.0f}R DD={m['max_dd_r']:.0f}R", fontsize=12)
        a0.set_ylabel("cum R"); a0.grid(alpha=0.2)
        wins = ex_df[ex_df.pnl_R > 0]; losses = ex_df[ex_df.pnl_R <= 0]
        bins = np.arange(-7, 12, 0.5)
        a1.hist(wins["pnl_R"], bins=bins, alpha=0.7, color="#00C896", label=f"wins n={len(wins)}")
        a1.hist(losses["pnl_R"], bins=bins, alpha=0.7, color="#FF3B69", label=f"losses n={len(losses)}")
        a1.axvline(0, color="black", lw=0.5); a1.legend(); a1.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(out_dir / "xau_m1_full_q1.0_equity.png", dpi=110)
        plt.close()
        print(f"\n  wrote xau_m1_full_q1.0_trades.csv + equity.png", flush=True)

    (HERE / "m1_full_summary.json").write_text(json.dumps(rows, indent=2, default=str))
    print(f"\n  done in {time.time()-t0:.0f}s — wrote m1_full_summary.json", flush=True)


if __name__ == "__main__":
    main()
