"""
M1 v10 meta-classifier — train second-stage filter on Q≥1.0 trades.

Stage 1: Q-regressor (already trained, predicts pnl_R from features)
Stage 2: This script — XGBClassifier on Q≥1.0 candidates predicting
         outcome bucket. Two heads:
           A) P(win)  — predict pnl_R > 0
           B) P(bad)  — predict pnl_R ≤ -3 (near hard SL)

Filter at inference: keep trade only if P(win) > thr_win AND P(bad) < thr_bad.
Sweep thresholds and report PF/sumR/DD.

Run on top of M1 v10 (NEAR=0.25, TRAIL=3, Q≥1.0, with $0.30 spread).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(HERE))

from tfk import compute_tfk

CUTOFF = pd.Timestamp("2024-12-12 00:00:00")
SL_ATR = 6.0
TRAIL = 3.0
MAX_HOLD = 300
NEAR = 0.25
Q_THR = 1.0
PROFIT_TO_BE_R = 1.0
MAX_CONCURRENT = 4
SWITCH_DELTA = 0.5
COOLDOWN_BARS = 5
BTC_SPREAD_USD = 5.0
BAD_R = -3.0


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
    print(f"\n{'='*72}\n  BTC M1 v10 + META-CLASSIFIER\n{'='*72}", flush=True)

    print("\n[1/5] loading M1 + computing TFK + features ...", flush=True)
    m1 = pd.read_parquet(ROOT / "data" / "m1_btc_full.parquet")
    m1 = m1.sort_values("time").reset_index(drop=True)
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
    spread_R = BTC_SPREAD_USD / median_atr
    print(f"  bars={n:,}  spread={spread_R:.3f}R", flush=True)

    print("\n[2/5] building candidates + labels ...", flush=True)
    dist_signed = np.where(atr > 0, (C - line) / atr, 0.0)
    dist_abs = np.abs(dist_signed)
    valid = (np.isfinite(atr) & (atr > 0))
    valid[:200] = False; valid[-(MAX_HOLD + 1):] = False
    valid &= (dist_abs <= NEAR) & (cdir != 0)
    idxs = np.where(valid)[0]
    dirs = cdir[idxs].copy()
    print(f"  candidates: {len(idxs):,}", flush=True)

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

    extra = pd.DataFrame({
        "dist_at_signal": dist_signed[idxs], "dist_abs": dist_abs[idxs],
        "regime_age": bir[idxs],
        "bar_range_atr": (H[idxs] - L[idxs]) / np.maximum(atr[idxs], 1e-9),
    })
    avail_feats = [f for f in Q_FEATS_M1 if f in df.columns]
    swing_feats = df.iloc[idxs][[f for f in avail_feats if f not in extra.columns]].reset_index(drop=True)
    X_all = pd.concat([extra.reset_index(drop=True), swing_feats], axis=1)
    feat_cols = [c for c in Q_FEATS_M1 if c in X_all.columns]

    times_at_idx = times[idxs]
    train_m = times_at_idx < np.datetime64(CUTOFF)
    test_m = ~train_m
    print(f"  train={train_m.sum():,}  test={test_m.sum():,}", flush=True)

    print("\n[3/5] Stage-1: fitting Q-regressor (same as v10) ...", flush=True)
    from xgboost import XGBRegressor, XGBClassifier
    q_mdl = XGBRegressor(n_estimators=600, max_depth=5, learning_rate=0.04,
                         subsample=0.85, colsample_bytree=0.85,
                         min_child_weight=10, reg_lambda=1.0,
                         objective="reg:squarederror", tree_method="hist",
                         random_state=42, verbosity=0)
    q_mdl.fit(X_all.loc[train_m, feat_cols].fillna(0).to_numpy(np.float32), pnl[train_m])
    q_train = q_mdl.predict(X_all.loc[train_m, feat_cols].fillna(0).to_numpy(np.float32))
    q_test = q_mdl.predict(X_all.loc[test_m, feat_cols].fillna(0).to_numpy(np.float32))

    print("\n[4/5] Stage-2: filtering to Q≥1.0 and training meta-classifiers ...", flush=True)
    # Subset Q≥1.0 from train (use spread-adjusted pnl for label since execution will use it)
    train_sel = q_train >= Q_THR
    test_sel = q_test >= Q_THR
    n_train_qsel = int(train_sel.sum())
    n_test_qsel = int(test_sel.sum())
    print(f"  Q≥{Q_THR} train: {n_train_qsel:,}  test: {n_test_qsel:,}", flush=True)

    train_pnl_s = pnl_spread[train_m][train_sel]
    train_feat = X_all.loc[train_m].iloc[train_sel.nonzero()[0]][feat_cols].fillna(0).to_numpy(np.float32)
    train_q = q_train[train_sel]
    # add Q as a feature
    train_X_meta = np.column_stack([train_feat, train_q])
    test_pnl_s = pnl_spread[test_m][test_sel]
    test_feat = X_all.loc[test_m].iloc[test_sel.nonzero()[0]][feat_cols].fillna(0).to_numpy(np.float32)
    test_q = q_test[test_sel]
    test_X_meta = np.column_stack([test_feat, test_q])

    # Head A: P(win) = P(pnl_spread > 0)
    y_win = (train_pnl_s > 0).astype(np.int64)
    print(f"  train P(win) base rate: {y_win.mean()*100:.1f}%", flush=True)
    win_mdl = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.04,
                            subsample=0.85, colsample_bytree=0.85,
                            min_child_weight=10, reg_lambda=1.0,
                            objective="binary:logistic", tree_method="hist",
                            random_state=42, verbosity=0, eval_metric="logloss")
    win_mdl.fit(train_X_meta, y_win)
    p_win_test = win_mdl.predict_proba(test_X_meta)[:, 1]
    print(f"  P(win) test mean: {p_win_test.mean()*100:.1f}%  median: {np.median(p_win_test)*100:.1f}%", flush=True)

    # Head B: P(bad) = P(pnl_spread ≤ BAD_R)
    y_bad = (train_pnl_s <= BAD_R).astype(np.int64)
    print(f"  train P(bad ≤{BAD_R}R) base rate: {y_bad.mean()*100:.1f}%", flush=True)
    bad_mdl = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.04,
                            subsample=0.85, colsample_bytree=0.85,
                            min_child_weight=10, reg_lambda=1.0,
                            objective="binary:logistic", tree_method="hist",
                            random_state=42, verbosity=0, eval_metric="logloss")
    bad_mdl.fit(train_X_meta, y_bad)
    p_bad_test = bad_mdl.predict_proba(test_X_meta)[:, 1]
    print(f"  P(bad) test mean: {p_bad_test.mean()*100:.1f}%  median: {np.median(p_bad_test)*100:.1f}%", flush=True)

    print("\n[5/5] applying meta filters + simulating ...", flush=True)
    # Map back to test indices
    test_idxs_qsel = idxs[test_m][test_sel]
    test_dirs_qsel = dirs[test_m][test_sel]
    test_exits_qsel = exit_idxs[test_m][test_sel]
    test_pnl_qsel = pnl_spread[test_m][test_sel]
    test_times_qsel = times_at_idx[test_m][test_sel]
    span_days = max((test_times_qsel[-1] - test_times_qsel[0]).astype("timedelta64[D]").astype(int), 1)

    def run_exec(p_win_thr=0.0, p_bad_thr=1.0):
        active = []; executed = []
        last_open_per_dir = {-1: -10**9, +1: -10**9}
        bar_start = int(test_idxs_qsel[0]); bar_end = min(int(test_idxs_qsel[-1]) + MAX_HOLD + 1, n)
        info = {}
        for k in range(len(test_idxs_qsel)):
            i = int(test_idxs_qsel[k])
            info[i] = (int(test_dirs_qsel[k]), float(test_pnl_qsel[k]), int(test_exits_qsel[k]),
                       float(test_q[k]), float(p_win_test[k]), float(p_bad_test[k]))
        for i in range(bar_start, bar_end):
            still = []
            for t in active:
                if i < t["entry_idx"]: still.append(t); continue
                if i > min(t["entry_idx"] + MAX_HOLD, n - 1):
                    cp = C[min(t["entry_idx"] + MAX_HOLD, n - 1)]
                    t["pnl_R"] = float(t["direction"] * (cp - t["ep"]) / t["a"] - spread_R)
                    t["exit_idx"] = min(t["entry_idx"] + MAX_HOLD, n - 1)
                    t["exit_reason"] = "max_hold"; executed.append(t); continue
                d = t["direction"]; ep = t["ep"]; a = t["a"]
                favor_now = d * (C[i] - ep)
                if favor_now > t["max_favor"]: t["max_favor"] = favor_now
                sl_r = t["sl_r"]; hit = False
                if sl_r == 0:
                    if d == 1 and L[i] <= ep:
                        t["pnl_R"] = -spread_R; t["exit_idx"] = i; t["exit_reason"] = "be_stop"; hit = True
                    elif d == -1 and H[i] >= ep:
                        t["pnl_R"] = -spread_R; t["exit_idx"] = i; t["exit_reason"] = "be_stop"; hit = True
                else:
                    dist_r = abs(sl_r) * a
                    if d == 1 and (ep - L[i]) >= dist_r:
                        t["pnl_R"] = float(sl_r - spread_R); t["exit_idx"] = i; t["exit_reason"] = "hard_sl"; hit = True
                    elif d == -1 and (H[i] - ep) >= dist_r:
                        t["pnl_R"] = float(sl_r - spread_R); t["exit_idx"] = i; t["exit_reason"] = "hard_sl"; hit = True
                if hit: executed.append(t); continue
                trail_d = TRAIL * a
                if t["max_favor"] >= trail_d and (t["max_favor"] - favor_now) >= trail_d:
                    t["pnl_R"] = float((t["max_favor"] - trail_d) / a - spread_R)
                    t["exit_idx"] = i; t["exit_reason"] = "trail"
                    executed.append(t); continue
                still.append(t)
            active = still
            if i not in info: continue
            d_, _, _, q_, pw, pb = info[i]
            if pw < p_win_thr: continue
            if pb > p_bad_thr: continue
            if i - last_open_per_dir[d_] < COOLDOWN_BARS: continue
            for t in active:
                if t["sl_r"] == 0: continue
                a_ = t["a"]; ep_ = t["ep"]; d2 = t["direction"]
                cur_R = d2 * (C[i] - ep_) / a_
                if cur_R >= PROFIT_TO_BE_R: t["sl_r"] = 0
            entry_idx = i + 1
            if entry_idx >= n: continue
            a_new = atr[i]
            if not (np.isfinite(a_new) and a_new > 0): continue
            ep_new = O[entry_idx]
            if len(active) >= MAX_CONCURRENT:
                worst = min(active, key=lambda x: x["q"])
                if q_ >= worst["q"] + SWITCH_DELTA:
                    cp = C[i]; ep_ = worst["ep"]; d2 = worst["direction"]; a_ = worst["a"]
                    worst["pnl_R"] = float(d2 * (cp - ep_) / a_ - spread_R)
                    worst["exit_idx"] = i; worst["exit_reason"] = "switch_close"
                    executed.append(worst); active.remove(worst)
                else: continue
            new_trade = {
                "signal_idx": int(i), "entry_idx": int(entry_idx),
                "direction": int(d_), "ep": float(ep_new), "a": float(a_new),
                "sl_r": float(-SL_ATR), "max_favor": 0.0, "q": float(q_),
                "time": pd.Timestamp(times[i]),
                "exit_idx": int(entry_idx), "exit_reason": None, "pnl_R": None,
                "p_win": float(pw), "p_bad": float(pb),
            }
            active.append(new_trade)
            last_open_per_dir[d_] = i
        for t in active:
            end_bar = min(t["entry_idx"] + MAX_HOLD, n - 1)
            cp = C[end_bar]
            t["pnl_R"] = float(t["direction"] * (cp - t["ep"]) / t["a"] - spread_R)
            t["exit_idx"] = end_bar; t["exit_reason"] = "end_of_data"
            executed.append(t)
        return executed

    # Baseline (no meta filter)
    print(f"\n  Baseline (Q≥{Q_THR}, no meta):")
    ex_base = run_exec()
    m_base = metrics(pd.DataFrame(ex_base)["pnl_R"].to_numpy())
    print(f"    n={m_base['n']:,}  WR={m_base['wr']*100:.1f}%  PF={m_base['pf']:.2f}  sumR={m_base['sum_r']:+.0f}  DD={m_base['max_dd_r']:.0f}  trd/d={m_base['n']/span_days:.1f}", flush=True)

    print(f"\n  P(win) filter sweep:")
    print(f"  {'p_win':>6}  {'n':>5} {'WR%':>5} {'PF':>5} {'sumR':>8} {'DD':>6} {'avgR':>6} {'trd/d':>5}", flush=True)
    for pw_thr in [0.0, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        ex = run_exec(p_win_thr=pw_thr)
        if not ex: continue
        m = metrics(pd.DataFrame(ex)["pnl_R"].to_numpy())
        print(f"  {pw_thr:>6.2f}  {m['n']:>5,} {m['wr']*100:>5.1f} {m['pf']:>5.2f} {m['sum_r']:>+8.1f} {m['max_dd_r']:>6.1f} {m['avg_r']:>+6.3f} {m['n']/span_days:>5.1f}")

    print(f"\n  P(bad ≤{BAD_R}R) filter sweep (lower P(bad) better):")
    print(f"  {'p_bad':>6}  {'n':>5} {'WR%':>5} {'PF':>5} {'sumR':>8} {'DD':>6} {'avgR':>6} {'trd/d':>5}", flush=True)
    for pb_thr in [1.0, 0.50, 0.40, 0.30, 0.25, 0.20, 0.15, 0.10]:
        ex = run_exec(p_bad_thr=pb_thr)
        if not ex: continue
        m = metrics(pd.DataFrame(ex)["pnl_R"].to_numpy())
        print(f"  {pb_thr:>6.2f}  {m['n']:>5,} {m['wr']*100:>5.1f} {m['pf']:>5.2f} {m['sum_r']:>+8.1f} {m['max_dd_r']:>6.1f} {m['avg_r']:>+6.3f} {m['n']/span_days:>5.1f}")

    print(f"\n  Combined P(win) ≥ X AND P(bad) ≤ Y sweep:")
    print(f"  {'pw':>5} {'pb':>5}  {'n':>5} {'WR%':>5} {'PF':>5} {'sumR':>8} {'DD':>6} {'avgR':>6} {'trd/d':>5}", flush=True)
    best = m_base
    best_cfg = (0.0, 1.0)
    for pw in [0.50, 0.55, 0.60, 0.65]:
        for pb in [0.40, 0.30, 0.20]:
            ex = run_exec(p_win_thr=pw, p_bad_thr=pb)
            if not ex: continue
            m = metrics(pd.DataFrame(ex)["pnl_R"].to_numpy())
            if m["n"] < 100: continue
            print(f"  {pw:>5.2f} {pb:>5.2f}  {m['n']:>5,} {m['wr']*100:>5.1f} {m['pf']:>5.2f} {m['sum_r']:>+8.1f} {m['max_dd_r']:>6.1f} {m['avg_r']:>+6.3f} {m['n']/span_days:>5.1f}")
            if m["sum_r"] > best["sum_r"]:
                best = m; best_cfg = (pw, pb)
    print(f"\n  BEST by sumR: P(win)≥{best_cfg[0]} AND P(bad)≤{best_cfg[1]} → "
          f"n={best['n']:,}  PF={best['pf']:.2f}  sumR={best['sum_r']:+.0f}  DD={best['max_dd_r']:.0f}", flush=True)

    # Feature importance
    try:
        imp_w = win_mdl.feature_importances_
        imp_b = bad_mdl.feature_importances_
        labels = feat_cols + ["q_value"]
        ow = np.argsort(imp_w)[::-1][:8]
        ob = np.argsort(imp_b)[::-1][:8]
        print(f"\n  P(win) top features: " + ", ".join(f"{labels[i]}({imp_w[i]:.3f})" for i in ow))
        print(f"  P(bad) top features: " + ", ".join(f"{labels[i]}({imp_b[i]:.3f})" for i in ob))
    except Exception:
        pass

    print(f"\n  done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
