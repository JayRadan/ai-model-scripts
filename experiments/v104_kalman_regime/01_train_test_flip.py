"""
v104 — Kalman regime, COLOR-FLIP entry logic. Full train + test pipeline.

Entry logic (NEW — replaces the v103 pullback/counter-to-line gate):
    red  -> green  (committed_dir -1 -> +1)  => BUY
    green -> red   (committed_dir +1 -> -1)  => SELL
Entry fills at the open of the bar AFTER the flip. Exit policy (SL / trail /
max_hold) and the XGBRegressor Q-filter are unchanged from v103, so the only
thing under test is the regime indicator + flip entry.

Head-to-head: runs the IDENTICAL flip pipeline with two indicators on the SAME
bars — the new Kalman filter (kalman.py) and the old TFK (tfk.py) — so the
comparison isolates the indicator.

Usage:
    python 01_train_test_flip.py                 # uses data/m1_xau_full.parquet
    python 01_train_test_flip.py --synthetic     # smoke test on generated bars
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from kalman import compute_kalman, KALMAN_FEATS
from tfk import compute_tfk

# ── Config (kept identical to the v103 research baseline for a fair compare) ──
CUTOFF = pd.Timestamp("2025-09-01 00:00:00")
SL_ATR = 4.0
TRAIL = 3.0
MAX_HOLD = 300
XAU_SPREAD_USD = 0.30

TFK_FEATS = ["force", "velocity", "x_est", "regime_w", "trend_raw", "trend"]
GENERIC_FEATS = [
    "rsi14", "dist_ema20", "dist_ema50", "dist_ema100", "dist_ema200",
    "slope5", "slope10", "slope20", "atr_ratio",
    "m5_rsi14", "m5_slope5", "m5_ema50_dist",
    "m15_rsi14", "m15_slope5", "m15_ema50_dist",
    "h1_rsi14", "h1_slope5", "h1_ema50_dist",
]
# Per-candidate extras computed at the flip bar.
EXTRA_FEATS = ["dist_at_flip", "bar_range_atr"]


def add_features(m1: pd.DataFrame) -> pd.DataFrame:
    """Standard generic + HTF features (identical recipe to v103 script 03)."""
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
        g = df.set_index("time")[["high", "low", "close"]].resample(tf).agg(
            {"high": "max", "low": "min", "close": "last"}).dropna()
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


def simulate_label(entry_idx, direction, C, H, L, O, atr_at_entry,
                   spread_R=0.0, sl_atr=SL_ATR, trail_atr=TRAIL, max_hold=MAX_HOLD):
    n = len(C)
    if entry_idx >= n - 1 or not (np.isfinite(atr_at_entry) and atr_at_entry > 0):
        return np.nan
    ep = O[entry_idx]; a = atr_at_entry
    hard = sl_atr * a; trail = trail_atr * a; max_favor = 0.0
    end = min(entry_idx + max_hold, n - 1)
    for k in range(entry_idx, end + 1):
        favor_now = direction * (C[k] - ep)
        if favor_now > max_favor: max_favor = favor_now
        if direction == 1:
            if (ep - L[k]) >= hard: return -sl_atr - spread_R
        else:
            if (H[k] - ep) >= hard: return -sl_atr - spread_R
        if max_favor >= trail and (max_favor - favor_now) >= trail:
            return (max_favor - trail) / a - spread_R
    return direction * (C[end] - ep) / a - spread_R


def metrics(r):
    r = np.asarray(r, dtype=np.float64)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return None
    w, l = r[r > 0], r[r <= 0]
    pf = float(w.sum() / max(-l.sum(), 1e-9))
    eq = np.cumsum(r)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    return {"n": int(len(r)), "wr": float((r > 0).mean()), "pf": pf,
            "sum_r": float(r.sum()), "max_dd_r": dd, "avg_r": float(r.mean())}


def build_flip_candidates(cdir: np.ndarray) -> np.ndarray:
    """Indices where the regime color flips (committed_dir != previous)."""
    flip = np.zeros(len(cdir), dtype=bool)
    flip[1:] = cdir[1:] != cdir[:-1]
    return flip


def run_indicator(name: str, base: pd.DataFrame, cutoff: pd.Timestamp) -> dict:
    """Compute indicator, build flip candidates, label, train Q, sweep on holdout."""
    print(f"\n{'='*72}\n  {name.upper()} — color-flip pipeline\n{'='*72}", flush=True)

    if name == "kalman":
        ind = compute_kalman(base)
        native = KALMAN_FEATS
    elif name == "tfk":
        ind = compute_tfk(base)
        native = TFK_FEATS
    else:
        raise ValueError(name)

    df = add_features(ind)
    O = df["open"].to_numpy(np.float64); H = df["high"].to_numpy(np.float64)
    L = df["low"].to_numpy(np.float64); C = df["close"].to_numpy(np.float64)
    cdir = df["committed_dir"].to_numpy(np.int64)
    line_col = "kf_line" if name == "kalman" else "tfk_line"
    line = df[line_col].to_numpy(np.float64)
    times = df["time"].to_numpy()
    atr = df["atr14"].to_numpy(np.float64)
    n = len(df)
    median_atr = float(np.nanmedian(atr))
    spread_R = XAU_SPREAD_USD / median_atr

    # ── Flip candidates ──
    flip = build_flip_candidates(cdir)
    valid = np.isfinite(atr) & (atr > 0)
    valid[:250] = False
    valid[-(MAX_HOLD + 1):] = False
    valid &= flip
    idxs = np.where(valid)[0]
    dirs = cdir[idxs].copy()  # direction = the NEW color at the flip
    n_buy = int((dirs == 1).sum()); n_sell = int((dirs == -1).sum())
    print(f"  bars={n:,}  median_atr={median_atr:.3f}  flips={len(idxs):,} "
          f"(buy {n_buy:,} / sell {n_sell:,})", flush=True)

    # ── Labels ──
    pnl = np.zeros(len(idxs), dtype=np.float64)
    pnl_s = np.zeros(len(idxs), dtype=np.float64)
    dist_signed = np.where(atr > 0, (C - line) / atr, 0.0)
    for k, i in enumerate(idxs):
        d = int(dirs[k])
        pnl[k] = simulate_label(i + 1, d, C, H, L, O, atr[i], spread_R=0.0)
        pnl_s[k] = simulate_label(i + 1, d, C, H, L, O, atr[i], spread_R=spread_R)
    pnl = np.where(np.isfinite(pnl), pnl, 0.0)
    pnl_s = np.where(np.isfinite(pnl_s), pnl_s, 0.0)
    print(f"  RAW flips  no-spread: {metrics(pnl)}", flush=True)
    print(f"  RAW flips  +spread  : {metrics(pnl_s)}", flush=True)

    # ── Feature matrix ──
    extra = pd.DataFrame({
        "dist_at_flip": dist_signed[idxs],
        "bar_range_atr": (H[idxs] - L[idxs]) / np.maximum(atr[idxs], 1e-9),
    })
    feat_cols = EXTRA_FEATS + [f for f in GENERIC_FEATS if f in df.columns] \
        + [f for f in native if f in df.columns]
    block = df.iloc[idxs][[f for f in (GENERIC_FEATS + native) if f in df.columns]].reset_index(drop=True)
    X = pd.concat([extra.reset_index(drop=True), block], axis=1)[feat_cols]

    times_at_idx = times[idxs]
    train_m = times_at_idx < np.datetime64(cutoff)
    test_m = ~train_m
    print(f"  features={len(feat_cols)}  train={train_m.sum():,}  test={test_m.sum():,}", flush=True)
    if train_m.sum() < 50 or test_m.sum() < 20:
        print("  !! not enough flips on one side of the split — results unreliable", flush=True)

    # ── Fit Q + holdout sweep ──
    from xgboost import XGBRegressor
    mdl = XGBRegressor(n_estimators=600, max_depth=5, learning_rate=0.04,
                       subsample=0.85, colsample_bytree=0.85, min_child_weight=10,
                       reg_lambda=1.0, objective="reg:squarederror",
                       tree_method="hist", random_state=42, verbosity=0)
    mdl.fit(X.loc[train_m].fillna(0).to_numpy(np.float32), pnl[train_m])
    q_te = mdl.predict(X.loc[test_m].fillna(0).to_numpy(np.float32))

    test_pnl_s = pnl_s[test_m]
    test_times = times_at_idx[test_m]
    if len(test_times) == 0:
        print("  !! empty holdout — skipping sweep", flush=True)
        return {"indicator": name, "bars": n, "median_atr": median_atr,
                "n_flips": len(idxs), "n_buy": n_buy, "n_sell": n_sell,
                "raw_no_spread": metrics(pnl), "raw_spread": metrics(pnl_s),
                "n_train": int(train_m.sum()), "n_test": 0,
                "holdout_sweep": {}, "span_days": 0}
    span_days = max((test_times.max() - test_times.min()).astype("timedelta64[D]").astype(int), 1)

    print(f"\n  HOLDOUT Q-sweep (+spread {XAU_SPREAD_USD}$ ≈ {spread_R:.3f}R):")
    print(f"  {'Q':>5} {'n':>6} {'WR%':>6} {'PF':>6} {'sumR':>9} {'DD':>7} {'avgR':>7} {'trd/d':>6}", flush=True)
    sweep = {}
    for qt in [-99, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
        sel = q_te >= qt
        m = metrics(test_pnl_s[sel])
        if not m:
            continue
        sweep[qt] = m
        print(f"  {qt:>5.1f} {m['n']:>6,} {m['wr']*100:>6.1f} {m['pf']:>6.2f} "
              f"{m['sum_r']:>+9.1f} {m['max_dd_r']:>7.1f} {m['avg_r']:>+7.3f} "
              f"{m['n']/span_days:>6.1f}", flush=True)

    return {
        "indicator": name, "bars": n, "median_atr": median_atr,
        "n_flips": len(idxs), "n_buy": n_buy, "n_sell": n_sell,
        "raw_no_spread": metrics(pnl), "raw_spread": metrics(pnl_s),
        "n_train": int(train_m.sum()), "n_test": int(test_m.sum()),
        "holdout_sweep": sweep, "span_days": int(span_days),
    }


def make_synthetic(n_bars=80_000, seed=7) -> pd.DataFrame:
    """Generate a plausible XAU-like M1 OHLCV series for a code smoke test.
    NOT real data — only validates that the pipeline runs end to end."""
    rng = np.random.default_rng(seed)
    times = pd.date_range("2023-01-02 00:00", periods=n_bars, freq="1min")
    # Regime-switching drift + GBM noise so flips actually occur.
    drift = np.zeros(n_bars)
    d = 0.0
    for i in range(n_bars):
        if rng.random() < 0.001:
            d = rng.normal(0, 0.02)
        drift[i] = d
    ret = drift + rng.normal(0, 0.15, n_bars)
    close = 1800 + np.cumsum(ret)
    open_ = np.concatenate([[close[0]], close[:-1]])
    hi = np.maximum(open_, close) + np.abs(rng.normal(0, 0.1, n_bars))
    lo = np.minimum(open_, close) - np.abs(rng.normal(0, 0.1, n_bars))
    vol = rng.integers(50, 500, n_bars).astype(float)
    return pd.DataFrame({"time": times, "open": open_, "high": hi,
                         "low": lo, "close": close, "tick_volume": vol})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true",
                    help="smoke test on generated bars (no real data)")
    ap.add_argument("--data", default=str(HERE / "data" / "m1_xau_full.parquet"))
    args = ap.parse_args()

    t0 = time.time()
    if args.synthetic:
        print("  [SYNTHETIC SMOKE TEST — results are meaningless, code-path only]")
        base = make_synthetic()
    else:
        p = Path(args.data)
        if not p.exists():
            sys.exit(f"  data not found: {p}\n  run 00_download_m1_dukascopy.py first "
                     f"(needs Dukascopy network access).")
        base = pd.read_parquet(p).sort_values("time").reset_index(drop=True)
        base["time"] = pd.to_datetime(base["time"])
    print(f"  bars: {len(base):,}  range: {base.time.iloc[0]} → {base.time.iloc[-1]}", flush=True)

    # Real run uses the production research cutoff; synthetic uses an 80% time split.
    if args.synthetic:
        cutoff = pd.Timestamp(base["time"].quantile(0.8))
        print(f"  [synthetic cutoff = {cutoff}]")
    else:
        cutoff = CUTOFF

    results = {}
    for name in ("kalman", "tfk"):
        results[name] = run_indicator(name, base, cutoff)

    # ── Head-to-head summary at Q=0 and the best Q ──
    print(f"\n{'='*72}\n  HEAD-TO-HEAD (holdout, +spread)\n{'='*72}")
    print(f"  {'indicator':>9} {'flips':>7} {'bestQ':>6} {'n':>6} {'WR%':>6} {'PF':>6} {'sumR':>9} {'DD':>7}")
    for name, r in results.items():
        sw = r["holdout_sweep"]
        if not sw:
            print(f"  {name:>9}  (no holdout trades)"); continue
        bestq = max(sw, key=lambda q: sw[q]["pf"])
        m = sw[bestq]
        print(f"  {name:>9} {r['n_flips']:>7,} {bestq:>6.1f} {m['n']:>6,} "
              f"{m['wr']*100:>6.1f} {m['pf']:>6.2f} {m['sum_r']:>+9.1f} {m['max_dd_r']:>7.1f}")

    out = HERE / ("results_synthetic.json" if args.synthetic else "results.json")
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n  done in {time.time()-t0:.0f}s — wrote {out.name}", flush=True)


if __name__ == "__main__":
    main()
