"""
Phase 5: ML-driven exit — remove fixed TP/SL, let a classifier decide
when to close each trade bar-by-bar.

Architecture:
  1. ENTRY: same as before (rule fires → confirmation classifier)
  2. EXIT:  at each bar after entry, an exit classifier predicts:
            "is the remaining forward return likely negative?"
            If yes → close now. If no → hold.

Exit classifier features (per-bar while in trade):
  - unrealized_pnl_R:  (current_close - entry) / ATR at entry (signed by direction)
  - bars_held:          how many bars since entry
  - pnl_velocity:       change in unrealized PnL over last 3 bars
  - hurst_rs, ou_theta, entropy_rate, kramers_up, wavelet_er (current bar physics)
  - quantum_flow, quantum_flow_h4 (current bar quantum)
  - vwap_dist (current bar)

Label:
  For bar k after entry, look at remaining bars k+1..k+MAX_REMAINING.
  best_remaining = max unrealized PnL from k+1 to end of window
  If best_remaining < unrealized_pnl_at_k → exit now (label=1, "close")
  If best_remaining > unrealized_pnl_at_k + 0.5*ATR → hold (label=0)
  (0.5 ATR threshold avoids labeling marginal holds)

Test: 60/20/20 split. Compare vs fixed TP=6/SL=2.
"""
from __future__ import annotations
import glob, json, os, time as _time
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/jay/Desktop/new-model-zigzag"
MAX_HOLD = 60  # max bars to hold (wider than fixed 40)
MIN_HOLD = 2   # always hold at least 2 bars (avoid noise exits)
SL_HARD  = 4.0 # hard stop-loss in ATR units (safety net, wider than before)

from phase3_physics_only import compute_all_features, PHYSICS_COLS, CURRENT_36, ROOT
from phase4_quantum_flow import compute_quantum_features, QUANTUM_COLS

EXIT_FEATURES = [
    "unrealized_pnl_R", "bars_held", "pnl_velocity",
    "hurst_rs", "ou_theta", "entropy_rate", "kramers_up", "wavelet_er",
    "quantum_flow", "quantum_flow_h4", "vwap_dist",
]


def build_exit_training_data(confirmed_setups, swing, atr, time_to_idx):
    """For each confirmed setup, generate per-bar exit training rows."""
    H, L, C = swing["high"].values, swing["low"].values, swing["close"].values
    n_bars = len(C)
    rows = []

    for _, setup in confirmed_setups.iterrows():
        t = setup["time"]
        if t not in time_to_idx.index: continue
        entry_idx = int(time_to_idx[t])
        direction = int(setup["direction"])
        entry_price = C[entry_idx]
        entry_atr = atr[entry_idx]
        if not np.isfinite(entry_atr) or entry_atr <= 0: continue

        end_idx = min(entry_idx + MAX_HOLD + 1, n_bars)
        if end_idx - entry_idx < 10: continue

        # Compute unrealized PnL at each bar
        pnls_R = []
        for k in range(entry_idx + 1, end_idx):
            upnl = direction * (C[k] - entry_price) / entry_atr
            pnls_R.append(upnl)

        if len(pnls_R) < 5: continue
        pnls_R = np.array(pnls_R)

        # For each bar after entry, build exit training row
        for b in range(len(pnls_R)):
            bar_idx = entry_idx + 1 + b
            current_pnl = pnls_R[b]
            bars_held = b + 1

            # Check hard stop
            if current_pnl < -SL_HARD:
                break  # would have been stopped out

            # Remaining best PnL
            remaining = pnls_R[b+1:] if b+1 < len(pnls_R) else np.array([current_pnl])
            if len(remaining) == 0:
                best_remaining = current_pnl
            else:
                best_remaining = remaining.max()

            # Label: should we exit now?
            # Exit if best remaining is worse than current (diminishing returns)
            margin = 0.3  # ATR units of margin
            if best_remaining < current_pnl - margin:
                label = 1  # exit now
            elif best_remaining > current_pnl + margin:
                label = 0  # hold
            else:
                continue  # ambiguous, skip

            # PnL velocity (change over last 3 bars)
            if b >= 3:
                vel = current_pnl - pnls_R[b-3]
            elif b >= 1:
                vel = current_pnl - pnls_R[0]
            else:
                vel = 0.0

            # Physics features at current bar
            row = {
                "unrealized_pnl_R": current_pnl,
                "bars_held": bars_held,
                "pnl_velocity": vel,
            }
            for feat in ["hurst_rs","ou_theta","entropy_rate","kramers_up",
                         "wavelet_er","quantum_flow","quantum_flow_h4","vwap_dist"]:
                if bar_idx < n_bars and feat in swing.columns:
                    row[feat] = swing[feat].iat[bar_idx]
                else:
                    row[feat] = 0.0
            row["label"] = label
            row["setup_time"] = t
            rows.append(row)

    return pd.DataFrame(rows)


def simulate_ml_exit(confirmed_test, exit_model, swing, atr, time_to_idx):
    """Simulate trades using ML exit on test set. Return list of PnLs in R."""
    H, L, C = swing["high"].values, swing["low"].values, swing["close"].values
    n_bars = len(C)
    pnls = []

    for _, setup in confirmed_test.iterrows():
        t = setup["time"]
        if t not in time_to_idx.index: continue
        entry_idx = int(time_to_idx[t])
        direction = int(setup["direction"])
        entry_price = C[entry_idx]
        entry_atr = atr[entry_idx]
        if not np.isfinite(entry_atr) or entry_atr <= 0: continue

        end_idx = min(entry_idx + MAX_HOLD + 1, n_bars)
        exited = False

        for b in range(1, end_idx - entry_idx):
            bar_idx = entry_idx + b
            current_pnl = direction * (C[bar_idx] - entry_price) / entry_atr

            # Hard stop
            if current_pnl < -SL_HARD:
                pnls.append(-SL_HARD)
                exited = True; break

            # Skip exit model for first MIN_HOLD bars
            if b < MIN_HOLD: continue

            # PnL velocity
            if b >= 4:
                prev_pnl = direction * (C[bar_idx-3] - entry_price) / entry_atr
                vel = current_pnl - prev_pnl
            else:
                vel = current_pnl

            feat_dict = {
                "unrealized_pnl_R": current_pnl,
                "bars_held": float(b),
                "pnl_velocity": vel,
            }
            for feat in ["hurst_rs","ou_theta","entropy_rate","kramers_up",
                         "wavelet_er","quantum_flow","quantum_flow_h4","vwap_dist"]:
                feat_dict[feat] = float(swing[feat].iat[bar_idx]) if bar_idx < n_bars and feat in swing.columns else 0.0

            X = np.array([[feat_dict[f] for f in EXIT_FEATURES]])
            prob_exit = exit_model.predict_proba(X)[0, 1]

            if prob_exit > 0.55:  # exit threshold
                pnls.append(current_pnl)
                exited = True; break

        if not exited:
            # Expired — close at last bar
            final_pnl = direction * (C[min(end_idx-1, n_bars-1)] - entry_price) / entry_atr
            pnls.append(final_pnl)

    return np.array(pnls)


def simulate_fixed_tp_sl(confirmed_test, swing, atr, time_to_idx, tp_mult=6.0, sl_mult=2.0, max_fwd=40):
    """Simulate with fixed TP/SL for comparison."""
    H, L, C = swing["high"].values, swing["low"].values, swing["close"].values
    n_bars = len(C)
    pnls = []
    for _, setup in confirmed_test.iterrows():
        t = setup["time"]
        if t not in time_to_idx.index: continue
        entry_idx = int(time_to_idx[t])
        direction = int(setup["direction"])
        entry_price = C[entry_idx]; a = atr[entry_idx]
        if not np.isfinite(a) or a <= 0: continue
        end = min(entry_idx + 1 + max_fwd, n_bars)
        found = False
        if direction == 1:
            tp, sl = entry_price + tp_mult*a, entry_price - sl_mult*a
            for k in range(entry_idx+1, end):
                if L[k] <= sl: pnls.append(-sl_mult); found=True; break
                if H[k] >= tp: pnls.append(+tp_mult); found=True; break
        else:
            tp, sl = entry_price - tp_mult*a, entry_price + sl_mult*a
            for k in range(entry_idx+1, end):
                if H[k] >= sl: pnls.append(-sl_mult); found=True; break
                if L[k] <= tp: pnls.append(+tp_mult); found=True; break
        if not found:
            final = direction*(C[end-1]-entry_price)/a
            pnls.append(final)
    return np.array(pnls)


def main():
    print("="*70)
    print("PHASE 5: ML-driven exit vs fixed TP/SL — 60/20/20")
    print("="*70, flush=True)

    swing, atr_arr = compute_all_features(f"{ROOT}/data/swing_v5_xauusd.csv")
    H, L, C = swing["high"].values, swing["low"].values, swing["close"].values
    time_to_idx = pd.Series(swing.index.values, index=swing["time"].values)

    print("Computing Quantum Flow...", flush=True)
    qf = compute_quantum_features(swing)
    for col, vals in qf.items(): swing[col] = vals

    # Load setups + merge physics
    all_setups = []
    for f in sorted(glob.glob(f"{ROOT}/data/setups_*.csv")):
        cid = int(os.path.basename(f).replace("setups_","").replace(".csv",""))
        df = pd.read_csv(f, parse_dates=["time"]); df["cid"] = cid
        all_setups.append(df)
    all_df = pd.concat(all_setups, ignore_index=True).sort_values("time").reset_index(drop=True)
    print(f"Total setups: {len(all_df):,}", flush=True)

    # Get CONFIRMED setups using existing models (entry confirmation)
    confirmed = []
    for (cid, rule), grp in all_df.groupby(["cid","rule"]):
        meta_path = f"{ROOT}/models/confirm_c{cid}_{rule}_meta.json"
        model_path = f"{ROOT}/models/confirm_c{cid}_{rule}.json"
        if not (os.path.exists(meta_path) and os.path.exists(model_path)): continue
        meta = json.load(open(meta_path))
        if meta.get("disabled", False): continue
        thr = meta["threshold"]; feat_cols = meta.get("feature_cols")
        if not feat_cols: continue
        miss = [c for c in feat_cols if c not in grp.columns]
        if miss: continue
        rdf = grp.sort_values("time").reset_index(drop=True)
        mdl = XGBClassifier(); mdl.load_model(model_path)
        X = rdf[feat_cols].fillna(0).values
        proba = mdl.predict_proba(X)[:,1]
        mask = proba >= thr
        confirmed.append(rdf[mask].copy())

    confirmed_df = pd.concat(confirmed, ignore_index=True).sort_values("time").reset_index(drop=True)
    print(f"Confirmed setups: {len(confirmed_df):,}", flush=True)

    # 60/20/20 split on confirmed setups
    n = len(confirmed_df)
    s1, s2 = int(n*0.60), int(n*0.80)
    train_setups = confirmed_df.iloc[:s1]
    val_setups   = confirmed_df.iloc[s1:s2]
    test_setups  = confirmed_df.iloc[s2:]
    print(f"Train: {len(train_setups)}, Val: {len(val_setups)}, Test: {len(test_setups)}", flush=True)

    # Build exit training data from train setups
    print("\nBuilding exit training data...", flush=True)
    t0 = _time.time()
    exit_train = build_exit_training_data(train_setups, swing, atr_arr, time_to_idx)
    print(f"  {len(exit_train):,} exit training rows ({_time.time()-t0:.1f}s)", flush=True)
    print(f"  label distribution: exit={exit_train['label'].sum():,} / hold={len(exit_train)-exit_train['label'].sum():,}", flush=True)

    # Train exit model
    print("Training exit classifier...", flush=True)
    y = exit_train["label"].values
    X = exit_train[EXIT_FEATURES].fillna(0).values
    exit_model = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8,
                                eval_metric="logloss", verbosity=0)
    exit_model.fit(X, y)

    # Simulate on TEST set
    print("\nSimulating on unseen test set...", flush=True)
    pnls_ml = simulate_ml_exit(test_setups, exit_model, swing, atr_arr, time_to_idx)
    pnls_fixed = simulate_fixed_tp_sl(test_setups, swing, atr_arr, time_to_idx, tp_mult=6.0, sl_mult=2.0)

    # Results
    for label, pnls in [("FIXED TP=6/SL=2", pnls_fixed), ("ML EXIT", pnls_ml)]:
        if len(pnls) == 0: print(f"{label}: no trades"); continue
        wins = pnls[pnls > 0]; losses = pnls[pnls < 0]
        wr = (pnls > 0).mean()
        pf = wins.sum() / (-losses.sum() + 1e-9) if len(losses) > 0 else float("inf")
        eq = np.cumsum(pnls); dd = (eq - np.maximum.accumulate(eq)).min()
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        print(f"\n{label}:")
        print(f"  Trades: {len(pnls)}  Wins: {len(wins)}  Losses: {len(losses)}")
        print(f"  Win Rate: {wr:.1%}")
        print(f"  PF: {pf:.3f}")
        print(f"  Avg Win: {avg_win:+.2f}R   Avg Loss: {avg_loss:+.2f}R")
        print(f"  Total PnL: {pnls.sum():+.1f}R")
        print(f"  Max DD: {dd:.1f}R")

    # Equity curves
    fig, ax = plt.subplots(figsize=(14, 5), facecolor="#0b0e14")
    ax.set_facecolor("#0d1117")
    if len(pnls_fixed) > 0:
        ax.plot(np.cumsum(pnls_fixed), color="#3b82f6", linewidth=1.5, label="Fixed TP=6/SL=2")
    if len(pnls_ml) > 0:
        ax.plot(np.cumsum(pnls_ml), color="#f5c518", linewidth=1.5, label="ML Exit")
    ax.axhline(0, color="#444", linewidth=0.5)
    ax.legend(facecolor="#111", edgecolor="#333", fontsize=11)
    ax.set_title("XAU — Fixed TP/SL vs ML-Driven Exit — Unseen Test",
                 color="#ffd700", fontsize=13)
    ax.set_xlabel("trade #", color="#888"); ax.set_ylabel("cumulative R", color="#888")
    ax.tick_params(colors="#888")
    for sp in ax.spines.values(): sp.set_edgecolor("#1a2332")
    plt.tight_layout()
    out = f"{ROOT}/experiments/innovative_rules/phase5_ml_exit.png"
    plt.savefig(out, dpi=140, facecolor="#0b0e14")
    print(f"\nSaved: {out}", flush=True)

    # Feature importance of exit model
    imp = exit_model.get_booster().get_score(importance_type="gain")
    print("\nExit model feature importance:")
    for i, feat in enumerate(EXIT_FEATURES):
        g = imp.get(f"f{i}", 0)
        print(f"  {feat:<22} gain={g:.1f}")


if __name__ == "__main__":
    main()
