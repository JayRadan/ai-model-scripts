"""
05b: Train ALL v6 models in one shot:
  (A) 26 per-rule confirmation classifiers (14 physics features)
  (B) 1 global exit classifier (11 features)
  (C) Export all to ONNX

Outputs:
  models/confirm_v6_c{cid}_{rule}.json + _meta.json  (26 models)
  models/exit_v6.json + exit_v6_meta.json             (1 model)
  models/confirm_v6_c{cid}_{rule}.onnx                (26 ONNX)
  models/exit_v6.onnx                                  (1 ONNX)
"""
from __future__ import annotations
import glob, json, os, time as _time
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import onnxmltools
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
import paths as P

CLUSTER_NAMES = {0:"Uptrend", 1:"MeanRevert", 2:"TrendRange", 3:"Downtrend", 4:"HighVol"}

V6_CONFIRM_FEATS = [
    "hurst_rs", "ou_theta", "entropy_rate", "kramers_up", "wavelet_er",
    "vwap_dist", "hour_enc", "dow_enc",
    "quantum_flow", "quantum_flow_h4", "quantum_momentum", "quantum_vwap_conf",
    "quantum_divergence", "quantum_div_strength",
]

EXIT_FEATS = [
    "unrealized_pnl_R", "bars_held", "pnl_velocity",
    "hurst_rs", "ou_theta", "entropy_rate", "kramers_up", "wavelet_er",
    "quantum_flow", "quantum_flow_h4", "vwap_dist",
]

TP_MULT, SL_MULT = 2.0, 1.0   # for confirmation labels (original geometry)
MAX_FWD_EXIT = 60
SL_HARD = 4.0


def export_onnx(model, n_features, out_path):
    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = convert_xgboost(model, initial_types=initial_type)
    with open(out_path, "wb") as f:
        f.write(onnx_model.SerializeToString())


def train_confirmations():
    """Train 26 per-rule confirmation models on v6 features."""
    print(f"\n{'='*60}\nTRAINING V6 CONFIRMATION MODELS (14 features)\n{'='*60}", flush=True)

    all_results = []
    for f in sorted(glob.glob(P.data("setups_*_v6.csv"))):
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

            missing = [c for c in V6_CONFIRM_FEATS if c not in train.columns]
            if missing:
                print(f"  SKIP c{cid}_{rule_name}: missing {missing[:3]}")
                continue

            X_tr = train[V6_CONFIRM_FEATS].fillna(0).values
            X_te = test[V6_CONFIRM_FEATS].fillna(0).values

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

            # Threshold sweep on holdout
            best_thr = 0.5; best_pf = 0
            for thr in np.arange(0.30, 0.70, 0.05):
                mask = proba >= thr
                if mask.sum() < 5: continue
                w = y_te[mask].sum(); lo = mask.sum() - w
                if lo == 0: continue
                pf = (w * 2.0) / (lo * 1.0)
                if pf > best_pf:
                    best_pf = pf; best_thr = float(thr)

            # Holdout stats at best threshold
            mask = proba >= best_thr
            n_ho = int(mask.sum())
            ho_wr = float(y_te[mask].mean()) if n_ho > 0 else 0
            ho_pf = best_pf
            ho_ev = ho_wr * 2.0 - (1-ho_wr) * 1.0 if n_ho > 0 else 0

            disabled = n_ho < 5 or ho_pf < 0.8

            # Save model
            model_name = f"confirm_v6_c{cid}_{rule_name}"
            mdl.save_model(P.model(f"{model_name}.json"))
            meta = {
                "rule": rule_name, "cluster": cid,
                "feature_cols": V6_CONFIRM_FEATS,
                "n_train": int(len(train)), "n_test": int(len(test)),
                "auc": round(auc, 4), "threshold": round(best_thr, 2),
                "holdout_pf": round(ho_pf, 2), "holdout_wr": round(ho_wr, 3),
                "holdout_n": n_ho, "holdout_ev": round(ho_ev, 3),
                "disabled": bool(disabled),
            }
            with open(P.model(f"{model_name}_meta.json"), "w") as mf:
                json.dump(meta, mf, indent=2)

            # Export ONNX
            export_onnx(mdl, len(V6_CONFIRM_FEATS), P.model(f"{model_name}.onnx"))

            flag = "✗ DISABLED" if disabled else "✓"
            print(f"  {flag} c{cid}_{rule_name:<22} AUC={auc:.3f}  thr={best_thr:.2f}  "
                  f"PF={ho_pf:.2f}  WR={ho_wr:.0%}  n={n_ho}", flush=True)
            all_results.append(meta)

    active = [r for r in all_results if not r["disabled"]]
    print(f"\n  Total: {len(all_results)} models, {len(active)} active")
    return all_results


def train_exit_model():
    """Train global exit classifier on confirmed setups."""
    print(f"\n{'='*60}\nTRAINING V6 EXIT MODEL (11 features)\n{'='*60}", flush=True)

    # Load swing for forward simulation
    swing = pd.read_csv(P.data("swing_v5_xauusd.csv"), parse_dates=["time"])
    swing = swing.sort_values("time").reset_index(drop=True)
    H, L, C = swing["high"].values, swing["low"].values, swing["close"].values
    tr = np.concatenate([[H[0]-L[0]],
          np.maximum.reduce([H[1:]-L[1:], np.abs(H[1:]-C[:-1]), np.abs(L[1:]-C[:-1])])])
    atr = pd.Series(tr).rolling(14, min_periods=14).mean().values
    time_to_idx = pd.Series(swing.index.values, index=swing["time"].values)

    # Load physics features from swing
    physics_df = pd.read_csv(P.data("setups_0_v6.csv"), nrows=0)  # just for column check
    # Actually load full physics from swing — recompute needed cols
    # For efficiency, load from the v6 CSV merge we already did
    # But for exit model we need physics at EVERY bar (not just setup bars)
    # So we need to use the swing_v5 + the 04b output

    # Reload physics from 04b's computation: they're in the merged setups,
    # but for exit we need them at arbitrary bars. Recompute from swing.
    print("  Loading physics features at every bar...", flush=True)
    from importlib.machinery import SourceFileLoader
    mod = SourceFileLoader("p04b", os.path.join(os.path.dirname(__file__), "04b_compute_physics_features.py")).load_module()
    ret = np.concatenate([[0.0], np.diff(np.log(C))])
    o, h_arr, l_arr = swing["open"].values, H, L
    vol = np.maximum(swing["spread"].values.astype(np.float64), 1.0)

    swing["hurst_rs"]     = mod.compute_hurst_rs(ret)
    swing["ou_theta"]     = mod.compute_ou_theta(ret)
    swing["entropy_rate"] = mod.compute_entropy(ret)
    swing["kramers_up"]   = mod.compute_kramers_up(C)
    swing["wavelet_er"]   = mod.compute_wavelet_er(C)
    swing["vwap_dist"]    = mod.compute_vwap_dist(swing, atr)
    swing["quantum_flow"] = mod.compute_quantum_flow(o.astype(np.float64), h_arr.astype(np.float64),
                                                      l_arr.astype(np.float64), C.astype(np.float64), vol)
    # H4 quantum
    df_h4 = swing.set_index("time")[["open","high","low","close","spread"]].resample("4h").agg(
        {"open":"first","high":"max","low":"min","close":"last","spread":"sum"}).dropna()
    qf_h4 = mod.compute_quantum_flow(df_h4["open"].values, df_h4["high"].values,
                                      df_h4["low"].values, df_h4["close"].values,
                                      np.maximum(df_h4["spread"].values, 1.0))
    qf_h4_s = pd.Series(qf_h4, index=df_h4.index).shift(1)
    swing["quantum_flow_h4"] = qf_h4_s.reindex(swing.set_index("time").index, method="ffill").values

    for col in EXIT_FEATS[3:]:  # fill NaN in physics cols
        swing[col] = swing[col].fillna(0)

    # Get confirmed setups using v6 models
    print("  Loading v6 confirmed setups...", flush=True)
    confirmed = []
    for f in sorted(glob.glob(P.data("setups_*_v6.csv"))):
        cid = int(os.path.basename(f).split("_")[1])
        df = pd.read_csv(f, parse_dates=["time"])
        for rule, grp in df.groupby("rule"):
            meta_path = P.model(f"confirm_v6_c{cid}_{rule}_meta.json")
            model_path = P.model(f"confirm_v6_c{cid}_{rule}.json")
            if not (os.path.exists(meta_path) and os.path.exists(model_path)): continue
            meta = json.load(open(meta_path))
            if meta.get("disabled", False): continue
            thr = meta["threshold"]
            rdf = grp.sort_values("time").reset_index(drop=True)
            mdl = XGBClassifier(); mdl.load_model(model_path)
            X = rdf[V6_CONFIRM_FEATS].fillna(0).values
            proba = mdl.predict_proba(X)[:,1]
            confirmed.append(rdf[proba >= thr].copy())

    if not confirmed:
        print("  ERROR: no confirmed setups!"); return
    confirmed_df = pd.concat(confirmed, ignore_index=True).sort_values("time").reset_index(drop=True)
    print(f"  Confirmed setups: {len(confirmed_df):,}", flush=True)

    # 80/20 split for exit model
    n = len(confirmed_df)
    split = int(n * 0.80)
    train_setups = confirmed_df.iloc[:split]

    # Build exit training data
    print("  Building exit training rows...", flush=True)
    t0 = _time.time()
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
    print(f"  Labels: exit={exit_df['label'].sum():,} hold={len(exit_df)-exit_df['label'].sum():,}", flush=True)

    # Train
    X = exit_df[EXIT_FEATS].fillna(0).values
    y = exit_df["label"].values
    exit_mdl = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", verbosity=0,
    )
    exit_mdl.fit(X, y)
    print(f"  Exit model trained", flush=True)

    # Save
    exit_mdl.save_model(P.model("exit_v6.json"))
    meta = {"feature_cols": EXIT_FEATS, "exit_threshold": 0.55, "hard_sl_atr": SL_HARD, "min_hold_bars": 2}
    with open(P.model("exit_v6_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Export ONNX
    export_onnx(exit_mdl, len(EXIT_FEATS), P.model("exit_v6.onnx"))
    print(f"  Saved: exit_v6.json + exit_v6.onnx", flush=True)


def summary():
    """Print summary of all v6 models."""
    print(f"\n{'='*60}\nV6 MODEL SUMMARY\n{'='*60}")
    onnx_files = sorted(glob.glob(P.model("confirm_v6_*.onnx")))
    print(f"  Confirmation ONNX files: {len(onnx_files)}")
    for f in onnx_files[:5]:
        print(f"    {os.path.basename(f)} ({os.path.getsize(f):,} bytes)")
    if len(onnx_files) > 5:
        print(f"    ... and {len(onnx_files)-5} more")
    exit_onnx = P.model("exit_v6.onnx")
    if os.path.exists(exit_onnx):
        print(f"  Exit ONNX: exit_v6.onnx ({os.path.getsize(exit_onnx):,} bytes)")
    print()


if __name__ == "__main__":
    train_confirmations()
    train_exit_model()
    summary()
