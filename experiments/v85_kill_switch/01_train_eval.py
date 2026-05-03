"""
v8.5 — Kill-switch model (vectorized).

Per-bar features for each trade (bars MIN_HOLD..exit-1):
  bars_held, current pnl_R, velocity, max/min so far, bars_since_max/min,
  direction, cluster_id

Label: 1 if trade ends in hard_sl, else 0.

Train on H1 (chronological first half of holdout).
Test on H2. Sweep kill thresholds {0.5, 0.6, 0.7, 0.8, 0.9}.
For each threshold, simulate: walk bars in order, exit at first bar where
killswitch_p >= threshold. Otherwise keep original exit.

Strict pass bar:
  - ΔR ≥ +50 R on H2 (kill saves > damages)
  - Test AUC ≥ 0.55
  - Doesn't catastrophically harm winners
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import time

ROOT = "/home/jay/Desktop/new-model-zigzag"
MIN_HOLD = 5
SL_HARD_R = -4.0
MAX_BARS_PER_TRADE = 60


def pf(rs):
    pos = rs[rs > 0].sum(); neg = -rs[rs <= 0].sum()
    return pos / max(neg, 1e-9)


def max_dd(rs):
    eq = np.cumsum(rs); peak = np.maximum.accumulate(eq)
    return (eq - peak).min() if len(eq) else 0.0


def load_swing(path):
    df = pd.read_csv(path, parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    H = df.high.values; L = df.low.values; C = df.close.values
    tr = np.concatenate([[H[0]-L[0]],
          np.maximum.reduce([H[1:]-L[1:],
                              np.abs(H[1:]-C[:-1]),
                              np.abs(L[1:]-C[:-1])])])
    df["atr"] = pd.Series(tr).rolling(14, min_periods=14).mean().values
    return df


def build_trajectory_matrix(trades, swing):
    """Vectorized: for each trade, return (N, MAX_BARS+1) pnl_R matrix.
    Row i is trade i's per-bar pnl. NaN past trade exit. close[entry]=0.
    Returns: pnl (N x MAX_BARS+1), bars_held (N), is_hardsl (N), direction (N), cid (N)
    """
    swing = swing.reset_index(drop=True)
    swing["__idx"] = np.arange(len(swing))
    time_to_idx = pd.Series(swing["__idx"].values, index=swing["time"].values)
    n = len(trades)
    pnl = np.full((n, MAX_BARS_PER_TRADE + 1), np.nan, dtype=np.float32)
    bars_held = np.zeros(n, dtype=np.int32)
    is_hardsl = np.zeros(n, dtype=np.int32)
    direction = np.zeros(n, dtype=np.int32)
    cid = np.zeros(n, dtype=np.int32)
    valid_mask = np.zeros(n, dtype=bool)

    closes = swing["close"].values.astype(np.float64)
    atrs   = swing["atr"].values.astype(np.float64)
    nbars = len(closes)

    for i, t in enumerate(trades.itertuples(index=False)):
        if t.time not in time_to_idx.index: continue
        ei = int(time_to_idx.loc[t.time])
        ep = closes[ei]; ea = atrs[ei]
        if not np.isfinite(ea) or ea <= 0: continue
        bh = min(int(t.bars), MAX_BARS_PER_TRADE)
        end = min(ei + bh, nbars - 1)
        if end <= ei: continue
        seg = closes[ei:end+1]   # bar 0 = entry close
        d = int(t.direction)
        pnl[i, :len(seg)] = (d * (seg - ep) / ea).astype(np.float32)
        bars_held[i] = bh
        is_hardsl[i] = 1 if t.exit == "hard_sl" else 0
        direction[i] = d
        cid[i] = int(t.cid)
        valid_mask[i] = True

    return pnl, bars_held, is_hardsl, direction, cid, valid_mask


def build_per_bar_features(pnl, bars_held, is_hardsl, direction, cid):
    """For each (trade, bar) where bar in [MIN_HOLD, bars_held-1] AND
    pnl > SL_HARD+0.1, build feature row + label.
    Returns: X (M x F), y (M,), trade_idx (M,), bar (M,)
    Vectorized rolling max/min via cummax/cummin."""
    n, T = pnl.shape  # T = MAX_BARS+1
    # Per-trade rolling cummax / cummin (ignoring NaN)
    # Replace NaN with very small / large for max/min computations
    safe_max = np.where(np.isnan(pnl), -np.inf, pnl)
    safe_min = np.where(np.isnan(pnl),  np.inf, pnl)
    cummax = np.maximum.accumulate(safe_max, axis=1)
    cummin = np.minimum.accumulate(safe_min, axis=1)

    # bars_since_max[i,k] = k - argmax of cummax up to k.
    # Compute via running last-update index of max.
    bars_since_max = np.zeros_like(pnl, dtype=np.float32)
    bars_since_min = np.zeros_like(pnl, dtype=np.float32)
    for i in range(n):
        last_max_k = 0; last_min_k = 0
        for k in range(T):
            if np.isnan(pnl[i, k]): break
            if k == 0 or pnl[i, k] > cummax[i, k-1]:
                last_max_k = k
            if k == 0 or pnl[i, k] < cummin[i, k-1]:
                last_min_k = k
            bars_since_max[i, k] = k - last_max_k
            bars_since_min[i, k] = k - last_min_k

    # Velocity: pnl[k] - pnl[k-3]
    velocity = np.zeros_like(pnl, dtype=np.float32)
    velocity[:, 3:] = pnl[:, 3:] - pnl[:, :-3]

    # Build sample rows for valid (trade, bar) pairs
    rows_X = []; rows_y = []; trade_idx = []; bar_idx = []
    for i in range(n):
        bh = int(bars_held[i])
        if bh < MIN_HOLD + 1: continue
        end = min(bh - 1, T - 1)
        for k in range(MIN_HOLD, end + 1):
            cp = pnl[i, k]
            if not np.isfinite(cp) or cp <= SL_HARD_R + 0.1: break
            rows_X.append([float(k), cp, velocity[i, k],
                            cummax[i, k], cummin[i, k],
                            bars_since_max[i, k], bars_since_min[i, k],
                            float(direction[i]), float(cid[i])])
            rows_y.append(int(is_hardsl[i]))
            trade_idx.append(i); bar_idx.append(k)
    X = np.array(rows_X, dtype=np.float32)
    y = np.array(rows_y, dtype=np.int32)
    return X, y, np.array(trade_idx), np.array(bar_idx), velocity, cummax, cummin, bars_since_max, bars_since_min


def evaluate_one(name, trades_path, swing):
    print("\n" + "="*72); print(f"=== {name} ===", flush=True)
    t0 = time.time()
    trades = pd.read_csv(trades_path, parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    print(f"  loaded {len(trades)} trades  hard_sl rate={(trades.exit=='hard_sl').mean():.1%}", flush=True)

    pnl, bars_held, is_hardsl, direction, cid, valid = build_trajectory_matrix(trades, swing)
    print(f"  trajectory matrix built  ({valid.sum()}/{len(trades)} valid)  {time.time()-t0:.1f}s", flush=True)

    X, y, t_idx, b_idx, vel, cmax, cmin, bsm, bsmn = build_per_bar_features(
        pnl, bars_held, is_hardsl, direction, cid)
    print(f"  built {len(X):,} per-bar samples  hard_sl rate={y.mean():.3f}  {time.time()-t0:.1f}s", flush=True)

    # Chrono split: first half of trades = train, second half = test
    n_tr = len(trades) // 2
    train_mask = t_idx < n_tr
    test_mask  = t_idx >= n_tr
    print(f"  H1 samples: {train_mask.sum()}  H2 samples: {test_mask.sum()}", flush=True)

    Xtr = X[train_mask]; ytr = y[train_mask]
    Xte = X[test_mask];  yte = y[test_mask]

    mdl = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                         eval_metric="logloss", verbosity=0, random_state=0,
                         tree_method="hist")
    mdl.fit(Xtr, ytr)
    auc_tr = roc_auc_score(ytr, mdl.predict_proba(Xtr)[:,1]) if len(set(ytr))>1 else 0.5
    auc_te = roc_auc_score(yte, mdl.predict_proba(Xte)[:,1]) if len(set(yte))>1 else 0.5
    print(f"  AUC train={auc_tr:.3f}  test={auc_te:.3f}  {time.time()-t0:.1f}s", flush=True)
    if auc_te < 0.55:
        print(f"  ⚠ AUC < 0.55 — not predictive enough, abort", flush=True); return None

    # ---- Vectorized simulation on H2 ----
    # Score every (test, bar) sample at once
    p_all = mdl.predict_proba(Xte)[:, 1]
    # For each test trade, find first bar where p >= threshold
    test_t_idx = t_idx[test_mask]
    test_b_idx = b_idx[test_mask]
    test_cp = X[test_mask, 1]   # cp column

    # H2 trade subset
    h2_trades = trades.iloc[n_tr:].reset_index(drop=True)
    rs0 = h2_trades.pnl_R.values
    base_R = rs0.sum(); base_wr = (rs0 > 0).mean(); base_pf = pf(rs0); base_dd = max_dd(rs0)
    base_hardsl_R = h2_trades[h2_trades.exit=="hard_sl"].pnl_R.sum()
    print(f"\n  H2 baseline: n={len(rs0)} WR={base_wr:.1%} PF={base_pf:.2f} R={base_R:+.0f} DD={base_dd:+.0f} hardsl_R={base_hardsl_R:+.0f}", flush=True)
    print(f"\n  {'thr':>4} {'killed':>6} {'killed%':>7} {'WR':>6} {'PF':>5} {'R':>8} {'DD':>7} {'ΔR':>8} {'sl_save':>8} {'win_dmg':>8}", flush=True)

    rows = []
    for thr in [0.50, 0.60, 0.70, 0.80, 0.90]:
        # For each test trade, find earliest bar where p >= thr
        new_pnl = rs0.copy()
        killed = np.zeros(len(rs0), dtype=bool)
        # Group samples by trade
        df_p = pd.DataFrame({"trade_local": test_t_idx - n_tr,
                              "bar": test_b_idx, "cp": test_cp, "p": p_all})
        fired = df_p[df_p.p >= thr]
        if len(fired):
            # Earliest bar per trade
            first = fired.sort_values(["trade_local", "bar"]).drop_duplicates("trade_local", keep="first")
            for _, row in first.iterrows():
                idx = int(row.trade_local)
                new_pnl[idx] = float(row.cp)
                killed[idx] = True
        n_killed = killed.sum()
        wr = (new_pnl > 0).mean(); pf_ = pf(new_pnl); R_ = new_pnl.sum(); dd_ = max_dd(new_pnl)
        was_hardsl = (h2_trades.exit == "hard_sl").values
        was_winner = (h2_trades.pnl_R > 0).values
        sl_save = (new_pnl[killed & was_hardsl] - rs0[killed & was_hardsl]).sum()
        win_dmg = (new_pnl[killed & was_winner] - rs0[killed & was_winner]).sum()
        rows.append({
            "thr": thr, "killed": int(n_killed),
            "killed_pct": round(n_killed/len(rs0), 3),
            "WR": round(wr, 4), "PF": round(pf_, 2),
            "R": round(R_, 1), "DD": round(dd_, 1),
            "ΔR": round(R_ - base_R, 1),
            "sl_save": round(sl_save, 1),
            "win_dmg": round(win_dmg, 1),
        })
        print(f"  {thr:>4.2f} {n_killed:>6} {n_killed/len(rs0):>6.1%} "
              f"{wr*100:>5.1f}% {pf_:>5.2f} {R_:>+8.1f} {dd_:>+7.1f} "
              f"{R_-base_R:>+8.1f} {sl_save:>+8.1f} {win_dmg:>+8.1f}", flush=True)
    print(f"  total time: {time.time()-t0:.1f}s", flush=True)
    return pd.DataFrame(rows)


def main():
    print("Loading XAU swing…", flush=True)
    sx = load_swing(os.path.join(ROOT, "data/swing_v5_xauusd.csv"))
    print("Loading BTC swing…", flush=True)
    sb = load_swing(os.path.join(ROOT, "data/swing_v5_btc.csv"))

    out_dir = os.path.dirname(__file__)
    for name, path, swing in [
        ("Oracle XAU", "data/v72l_trades_holdout.csv", sx),
        ("Midas XAU",  "data/v6_trades_holdout_xau.csv", sx),
        ("Oracle BTC", "data/v72l_trades_holdout_btc.csv", sb),
    ]:
        df = evaluate_one(name, os.path.join(ROOT, path), swing)
        if df is not None:
            df.to_csv(os.path.join(out_dir,
                                    f"sweep_{name.lower().replace(' ','_')}.csv"),
                       index=False)


if __name__ == "__main__":
    main()
