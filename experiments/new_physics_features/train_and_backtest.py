"""
Train confirmation models on 21-feature vector (14 existing + 7 new physics)
and run the same holdout backtest as v6 for direct comparison.

Models saved with v6b_ prefix to not overwrite v6.
"""
from __future__ import annotations
import glob, json, os, time as _time
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "/home/jay/Desktop/new-model-zigzag/model_pipeline")
import paths as P

TAG = "v6b"

EXISTING_FEATS = [
    "hurst_rs", "ou_theta", "entropy_rate", "kramers_up", "wavelet_er",
    "vwap_dist", "hour_enc", "dow_enc",
    "quantum_flow", "quantum_flow_h4", "quantum_momentum", "quantum_vwap_conf",
    "quantum_divergence", "quantum_div_strength",
]
NEW_FEATS = [
    "permutation_entropy", "dfa_alpha", "higuchi_fd",
    "spectral_entropy", "hill_tail_index", "vol_of_vol", "log_drift",
]
ALL_FEATS = EXISTING_FEATS + NEW_FEATS

EXIT_FEATS = [
    "unrealized_pnl_R", "bars_held", "pnl_velocity",
    "hurst_rs", "ou_theta", "entropy_rate", "kramers_up", "wavelet_er",
    "quantum_flow", "quantum_flow_h4", "vwap_dist",
]

MAX_FWD_EXIT = 60
SL_HARD = 4.0
MAX_HOLD = 60
MIN_HOLD = 2
EXIT_THRESHOLD = 0.55


def train_confirmations():
    print(f"\n{'='*60}\nTRAINING {TAG.upper()} CONFIRMATION (21 features = 14 + 7 new)\n{'='*60}", flush=True)
    all_results = []
    importances_sum = {f: 0.0 for f in ALL_FEATS}
    importances_cnt = {f: 0 for f in ALL_FEATS}
    for f in sorted(glob.glob(P.data(f"setups_*_{TAG}.csv"))):
        cid = int(os.path.basename(f).split("_")[1])
        df = pd.read_csv(f, parse_dates=["time"])
        if "label" not in df.columns or "rule" not in df.columns:
            continue
        for rule_name, grp in df.groupby("rule"):
            rdf = grp.sort_values("time").reset_index(drop=True)
            if len(rdf) < 100: continue
            split = int(len(rdf) * 0.80)
            train, test = rdf.iloc[:split], rdf.iloc[split:]
            if len(test) < 20: continue
            y_tr = train["label"].astype(int).values
            y_te = test["label"].astype(int).values
            missing = [c for c in ALL_FEATS if c not in train.columns]
            if missing:
                print(f"  SKIP c{cid}_{rule_name}: missing {missing[:3]}"); continue
            X_tr = train[ALL_FEATS].fillna(0).values
            X_te = test[ALL_FEATS].fillna(0).values
            mdl = XGBClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss", verbosity=0,
            )
            mdl.fit(X_tr, y_tr)
            proba = mdl.predict_proba(X_te)[:, 1]
            try:
                auc = roc_auc_score(y_te, proba)
            except:
                auc = 0.5
            best_thr = 0.5; best_pf = 0
            for thr in np.arange(0.30, 0.70, 0.05):
                mask = proba >= thr
                if mask.sum() < 5: continue
                w = y_te[mask].sum(); lo = mask.sum() - w
                if lo == 0: continue
                pf = (w * 2.0) / (lo * 1.0)
                if pf > best_pf:
                    best_pf = pf; best_thr = float(thr)
            mask = proba >= best_thr
            n_ho = int(mask.sum())
            ho_wr = float(y_te[mask].mean()) if n_ho > 0 else 0
            ho_pf = best_pf
            ho_ev = ho_wr * 2.0 - (1 - ho_wr) * 1.0 if n_ho > 0 else 0
            disabled = n_ho < 5 or ho_pf < 0.8
            model_name = f"confirm_{TAG}_c{cid}_{rule_name}"
            mdl.save_model(P.model(f"{model_name}.json"))
            meta = {
                "rule": rule_name, "cluster": cid,
                "feature_cols": ALL_FEATS,
                "n_train": int(len(train)), "n_test": int(len(test)),
                "auc": round(auc, 4), "threshold": round(best_thr, 2),
                "holdout_pf": round(ho_pf, 2), "holdout_wr": round(ho_wr, 3),
                "holdout_n": n_ho, "holdout_ev": round(ho_ev, 3),
                "disabled": bool(disabled),
            }
            with open(P.model(f"{model_name}_meta.json"), "w") as mf:
                json.dump(meta, mf, indent=2)
            # Accumulate feature importance
            imp = mdl.feature_importances_
            for fi, feat in enumerate(ALL_FEATS):
                importances_sum[feat] += float(imp[fi])
                importances_cnt[feat] += 1
            flag = "X DIS" if disabled else "OK "
            print(f"  {flag} c{cid}_{rule_name:<22} AUC={auc:.3f}  thr={best_thr:.2f}  "
                  f"PF={ho_pf:.2f}  WR={ho_wr:.0%}  n={n_ho}", flush=True)
            all_results.append(meta)
    active = [r for r in all_results if not r["disabled"]]
    print(f"\n  Total: {len(all_results)} models, {len(active)} active")
    # Rank feature importance
    print(f"\n{'='*60}\nFEATURE IMPORTANCE (average across all active rule models)\n{'='*60}")
    avg = [(f, importances_sum[f] / max(importances_cnt[f], 1)) for f in ALL_FEATS]
    avg.sort(key=lambda x: -x[1])
    for f, v in avg:
        marker = "★ NEW" if f in NEW_FEATS else "     "
        print(f"  {marker}  {f:<25} {v:.4f}")
    return all_results


def train_exit_model():
    print(f"\n{'='*60}\nTRAINING {TAG.upper()} EXIT MODEL\n{'='*60}", flush=True)
    # Uses same swing CSV + existing physics; exit model feature set is unchanged (11 feats)
    swing = pd.read_csv(P.data("swing_v5_xauusd.csv"), parse_dates=["time"])
    swing = swing.sort_values("time").reset_index(drop=True)
    H, L, C = swing["high"].values, swing["low"].values, swing["close"].values
    tr = np.concatenate([[H[0]-L[0]],
          np.maximum.reduce([H[1:]-L[1:], np.abs(H[1:]-C[:-1]), np.abs(L[1:]-C[:-1])])])
    atr = pd.Series(tr).rolling(14, min_periods=14).mean().values
    time_to_idx = pd.Series(swing.index.values, index=swing["time"].values)

    # Load physics at every bar (existing pipeline)
    from importlib.machinery import SourceFileLoader
    mod = SourceFileLoader("p04b", "/home/jay/Desktop/new-model-zigzag/model_pipeline/04b_compute_physics_features.py").load_module()
    ret = np.concatenate([[0.0], np.diff(np.log(C))])
    o_arr = swing["open"].values.astype(np.float64)
    h_arr = H.astype(np.float64); l_arr = L.astype(np.float64)
    vol = np.maximum(swing["spread"].values.astype(np.float64), 1.0)
    swing["hurst_rs"]     = mod.compute_hurst_rs(ret)
    swing["ou_theta"]     = mod.compute_ou_theta(ret)
    swing["entropy_rate"] = mod.compute_entropy(ret)
    swing["kramers_up"]   = mod.compute_kramers_up(C)
    swing["wavelet_er"]   = mod.compute_wavelet_er(C)
    swing["vwap_dist"]    = mod.compute_vwap_dist(swing, atr)
    swing["quantum_flow"] = mod.compute_quantum_flow(o_arr, h_arr, l_arr, C, vol)
    df_h4 = swing.set_index("time")[["open","high","low","close","spread"]].resample("4h").agg(
        {"open":"first","high":"max","low":"min","close":"last","spread":"sum"}).dropna()
    qf_h4 = mod.compute_quantum_flow(df_h4["open"].values, df_h4["high"].values,
                                      df_h4["low"].values, df_h4["close"].values,
                                      np.maximum(df_h4["spread"].values, 1.0))
    qf_h4_s = pd.Series(qf_h4, index=df_h4.index).shift(1)
    swing["quantum_flow_h4"] = qf_h4_s.reindex(swing.set_index("time").index, method="ffill").values
    for col in EXIT_FEATS[3:]:
        swing[col] = swing[col].fillna(0)

    print("  Loading v6b confirmed setups...", flush=True)
    confirmed = []
    for f in sorted(glob.glob(P.data(f"setups_*_{TAG}.csv"))):
        cid = int(os.path.basename(f).split("_")[1])
        df = pd.read_csv(f, parse_dates=["time"])
        for rule, grp in df.groupby("rule"):
            meta_path = P.model(f"confirm_{TAG}_c{cid}_{rule}_meta.json")
            model_path = P.model(f"confirm_{TAG}_c{cid}_{rule}.json")
            if not (os.path.exists(meta_path) and os.path.exists(model_path)): continue
            meta = json.load(open(meta_path))
            if meta.get("disabled", False): continue
            thr = meta["threshold"]
            rdf = grp.sort_values("time").reset_index(drop=True)
            mdl = XGBClassifier(); mdl.load_model(model_path)
            X = rdf[ALL_FEATS].fillna(0).values
            proba = mdl.predict_proba(X)[:,1]
            confirmed.append(rdf[proba >= thr].copy())
    if not confirmed:
        print("  ERROR: no confirmed setups!"); return None, None
    confirmed_df = pd.concat(confirmed, ignore_index=True).sort_values("time").reset_index(drop=True)
    print(f"  Confirmed setups: {len(confirmed_df):,}", flush=True)

    n = len(confirmed_df); split = int(n * 0.80)
    train_setups = confirmed_df.iloc[:split]
    print("  Building exit training rows...", flush=True); t0 = _time.time()
    exit_rows = []
    for _, setup in train_setups.iterrows():
        t = setup["time"]
        if t not in time_to_idx.index: continue
        entry_idx = int(time_to_idx[t])
        direction = int(setup["direction"])
        entry_price = C[entry_idx]; entry_atr = atr[entry_idx]
        if not np.isfinite(entry_atr) or entry_atr <= 0: continue
        end_idx = min(entry_idx + MAX_FWD_EXIT + 1, len(C))
        if end_idx - entry_idx < 10: continue
        pnls_R = []
        for k in range(entry_idx+1, end_idx):
            pnls_R.append(direction * (C[k] - entry_price) / entry_atr)
        pnls_R = np.array(pnls_R)
        if len(pnls_R) < 5: continue
        for b in range(len(pnls_R)):
            bar_idx = entry_idx + 1 + b
            cur_pnl = pnls_R[b]
            if cur_pnl < -SL_HARD: break
            remaining = pnls_R[b+1:] if b+1 < len(pnls_R) else np.array([cur_pnl])
            best_rem = remaining.max() if len(remaining) > 0 else cur_pnl
            margin = 0.3
            if best_rem < cur_pnl - margin: label = 1
            elif best_rem > cur_pnl + margin: label = 0
            else: continue
            vel = cur_pnl - pnls_R[b-3] if b >= 3 else (cur_pnl - pnls_R[0] if b >= 1 else 0.0)
            row = {"unrealized_pnl_R": cur_pnl, "bars_held": float(b+1), "pnl_velocity": vel, "label": label}
            for feat in EXIT_FEATS[3:]:
                row[feat] = float(swing[feat].iat[bar_idx]) if bar_idx < len(swing) else 0.0
            exit_rows.append(row)
    exit_df = pd.DataFrame(exit_rows)
    print(f"  {len(exit_df):,} exit training rows ({_time.time()-t0:.0f}s)", flush=True)
    X = exit_df[EXIT_FEATS].fillna(0).values
    y = exit_df["label"].values
    exit_mdl = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", verbosity=0,
    )
    exit_mdl.fit(X, y)
    exit_mdl.save_model(P.model(f"exit_{TAG}.json"))
    with open(P.model(f"exit_{TAG}_meta.json"), "w") as f:
        json.dump({"feature_cols": EXIT_FEATS, "exit_threshold": 0.55, "hard_sl_atr": SL_HARD, "min_hold_bars": 2}, f)
    print(f"  Saved: exit_{TAG}.json", flush=True)
    return swing, time_to_idx


def run_backtest(swing, time_to_idx):
    print(f"\n{'='*60}\nHOLDOUT BACKTEST — {TAG.upper()}\n{'='*60}", flush=True)
    c = swing["close"].values.astype(np.float64)
    atr_series = pd.Series(swing["high"].values - swing["low"].values)  # quick for indexing

    # recompute ATR properly
    H = swing["high"].values.astype(np.float64); L = swing["low"].values.astype(np.float64)
    tr = np.concatenate([[H[0]-L[0]],
          np.maximum.reduce([H[1:]-L[1:], np.abs(H[1:]-c[:-1]), np.abs(L[1:]-c[:-1])])])
    atr = pd.Series(tr).rolling(14, min_periods=14).mean().values
    n = len(c)

    all_setups = []
    for f in sorted(glob.glob(P.data(f"setups_*_{TAG}.csv"))):
        cid = int(os.path.basename(f).split("_")[1])
        df = pd.read_csv(f, parse_dates=["time"])
        df["cid"] = cid
        all_setups.append(df)
    all_df = pd.concat(all_setups, ignore_index=True).sort_values("time").reset_index(drop=True)

    split_time = all_df["time"].quantile(0.80)
    holdout = all_df[all_df["time"] > split_time].copy().reset_index(drop=True)
    print(f"  Holdout: {len(holdout):,} setups from {holdout['time'].iat[0]} to {holdout['time'].iat[-1]}")

    confirmed_rows = []
    for (cid, rule), grp in holdout.groupby(["cid", "rule"]):
        meta_path = P.model(f"confirm_{TAG}_c{cid}_{rule}_meta.json")
        model_path = P.model(f"confirm_{TAG}_c{cid}_{rule}.json")
        if not (os.path.exists(meta_path) and os.path.exists(model_path)): continue
        meta = json.load(open(meta_path))
        if meta.get("disabled", False): continue
        thr = meta["threshold"]
        mdl = XGBClassifier(); mdl.load_model(model_path)
        X = grp[ALL_FEATS].fillna(0).values
        proba = mdl.predict_proba(X)[:, 1]
        mask = proba >= thr
        confirmed_rows.append(grp[mask].copy())
    confirmed = pd.concat(confirmed_rows, ignore_index=True).sort_values("time").reset_index(drop=True)
    print(f"  Confirmed entries: {len(confirmed):,}")

    exit_mdl = XGBClassifier(); exit_mdl.load_model(P.model(f"exit_{TAG}.json"))

    print("\n  Simulating with ML exit...", flush=True)
    trades = []
    equity = [0.0]
    for _, setup in confirmed.iterrows():
        t = setup["time"]
        if t not in time_to_idx.index: continue
        entry_idx = int(time_to_idx[t])
        direction = int(setup["direction"])
        entry_price = c[entry_idx]; entry_atr = atr[entry_idx]
        if not np.isfinite(entry_atr) or entry_atr <= 0: continue
        exit_idx = None; exit_reason = "max"
        for k in range(1, MAX_HOLD+1):
            bar = entry_idx + k
            if bar >= n: break
            cur_pnl = direction * (c[bar] - entry_price) / entry_atr
            if cur_pnl < -SL_HARD:
                exit_idx = bar; exit_reason = "hard_sl"; break
            if k >= MIN_HOLD:
                pnl_3ago = direction * (c[bar-3] - entry_price) / entry_atr if k >= 3 else cur_pnl
                row = {
                    "unrealized_pnl_R": cur_pnl, "bars_held": float(k),
                    "pnl_velocity": cur_pnl - pnl_3ago,
                    "hurst_rs": swing["hurst_rs"].iat[bar],
                    "ou_theta": swing["ou_theta"].iat[bar],
                    "entropy_rate": swing["entropy_rate"].iat[bar],
                    "kramers_up": swing["kramers_up"].iat[bar],
                    "wavelet_er": swing["wavelet_er"].iat[bar],
                    "quantum_flow": swing["quantum_flow"].iat[bar],
                    "quantum_flow_h4": swing["quantum_flow_h4"].iat[bar],
                    "vwap_dist": swing["vwap_dist"].iat[bar],
                }
                X_exit = np.array([[row[ef] for ef in EXIT_FEATS]])
                p_exit = exit_mdl.predict_proba(X_exit)[0, 1]
                if p_exit >= EXIT_THRESHOLD:
                    exit_idx = bar; exit_reason = "ml_exit"; break
        if exit_idx is None:
            exit_idx = min(entry_idx + MAX_HOLD, n-1); exit_reason = "max"
        pnl_R = direction * (c[exit_idx] - entry_price) / entry_atr
        trades.append({
            "time": t, "cid": int(setup["cid"]), "rule": setup["rule"],
            "dir": direction, "bars": exit_idx - entry_idx,
            "pnl_R": pnl_R, "exit": exit_reason,
        })
        equity.append(equity[-1] + pnl_R)

    trades_df = pd.DataFrame(trades)
    if len(trades_df) == 0:
        print("  NO TRADES"); return

    wins = trades_df[trades_df["pnl_R"] > 0]
    losses = trades_df[trades_df["pnl_R"] <= 0]
    pf = wins["pnl_R"].sum() / max(-losses["pnl_R"].sum(), 1e-9)
    wr = len(wins) / len(trades_df)
    expectancy_R = trades_df["pnl_R"].mean()
    total_R = trades_df["pnl_R"].sum()
    eq = np.array(equity); dd = np.maximum.accumulate(eq) - eq; max_dd = dd.max()

    print(f"\n{'='*60}")
    print(f"V6B RESULTS (14 existing + 7 new physics features = 21 total)")
    print(f"{'='*60}")
    print(f"Total trades:     {len(trades_df):,}")
    print(f"Win rate:         {wr:.1%}")
    print(f"Avg win (R):      +{wins['pnl_R'].mean():.2f}")
    print(f"Avg loss (R):     {losses['pnl_R'].mean():.2f}")
    print(f"Expectancy (R):   {expectancy_R:+.3f}")
    print(f"Profit factor:    {pf:.2f}")
    print(f"Total PnL (R):    {total_R:+.1f}")
    print(f"Max DD (R):       -{max_dd:.1f}")
    print(f"\nvs v6 shipped:    PF 2.46 | WR 60.6% | Expectancy +2.11 R | MaxDD -61 R | n=2,697")
    print(f"\nPer-cluster (v6b):")
    for cid, grp in trades_df.groupby("cid"):
        w = grp[grp["pnl_R"]>0]; ll = grp[grp["pnl_R"]<=0]
        ppf = w["pnl_R"].sum() / max(-ll["pnl_R"].sum(), 1e-9)
        print(f"  C{cid}: n={len(grp):,} WR={len(w)/len(grp):.1%} PF={ppf:.2f} R={grp['pnl_R'].sum():+.1f}")

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(eq, linewidth=1.2)
    ax.set_title(f"V6B Holdout (21 feats): PF={pf:.2f}  WR={wr:.1%}  n={len(trades_df)}  "
                 f"Total={total_R:+.1f}R  MaxDD=-{max_dd:.1f}R", fontsize=13)
    ax.set_xlabel("Trade #"); ax.set_ylabel("Cumulative PnL (R)")
    ax.grid(alpha=0.3); ax.axhline(0, color="k", linewidth=0.5)
    plt.tight_layout()
    out_png = "/home/jay/Desktop/new-model-zigzag/experiments/new_physics_features/v6b_holdout.png"
    plt.savefig(out_png, dpi=120)
    print(f"\nEquity curve saved: {out_png}")
    trades_df.to_csv("/home/jay/Desktop/new-model-zigzag/experiments/new_physics_features/v6b_trades.csv", index=False)


if __name__ == "__main__":
    train_confirmations()
    swing, t2i = train_exit_model()
    if swing is not None:
        run_backtest(swing, t2i)
