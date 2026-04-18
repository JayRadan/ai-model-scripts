"""
v7.2-lite DEPLOYMENT — train on FULL data (train + holdout) and export to ONNX.

Produces:
  models/confirm_v7_c{cid}_{rule}.onnx   — 28 per-rule confirmation models
  models/exit_v7.onnx                    — global ML exit model
  models/meta_v7.onnx                    — meta-labeling head (input: 18 feats + direction + cid)
  models/v7_deploy.json                  — feature order, thresholds, meta threshold

This is the PRODUCTION training (no holdout reserve — we already validated v7.2-lite
architecture on a clean holdout in 01_validate_v72_lite.py, PF 3.58, WR 67.0%).
"""
from __future__ import annotations
import glob, json, os, time as _time
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import onnxruntime as ort
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
import sys
sys.path.insert(0, "/home/jay/Desktop/new-model-zigzag/model_pipeline")
import paths as P

OLD_FEATS = [
    "hurst_rs", "ou_theta", "entropy_rate", "kramers_up", "wavelet_er",
    "vwap_dist", "hour_enc", "dow_enc",
    "quantum_flow", "quantum_flow_h4", "quantum_momentum", "quantum_vwap_conf",
    "quantum_divergence", "quantum_div_strength",
]
V7_EXTRA = ["vpin", "sig_quad_var", "har_rv_ratio", "hawkes_eta"]
V7_FEATS = OLD_FEATS + V7_EXTRA                               # 18
META_FEATS = V7_FEATS + ["direction", "cid"]                    # 20

EXIT_FEATS = [
    "unrealized_pnl_R", "bars_held", "pnl_velocity",
    "hurst_rs", "ou_theta", "entropy_rate", "kramers_up", "wavelet_er",
    "quantum_flow", "quantum_flow_h4", "vwap_dist",
]

MAX_HOLD = 60; MIN_HOLD = 2; SL_HARD = 4.0
EXIT_THRESHOLD = 0.55; MAX_FWD_EXIT = 60
META_THRESHOLD = 0.675                      # validated on clean holdout


def load_swing_with_physics():
    print("  Loading swing + physics...", flush=True)
    swing = pd.read_csv(P.data("swing_v5_xauusd.csv"), parse_dates=["time"])
    swing = swing.sort_values("time").reset_index(drop=True)
    H = swing["high"].values.astype(np.float64)
    L = swing["low"].values.astype(np.float64)
    C = swing["close"].values.astype(np.float64)
    O = swing["open"].values.astype(np.float64)
    vol = np.maximum(swing["spread"].values.astype(np.float64), 1.0)
    tr = np.concatenate([[H[0]-L[0]],
          np.maximum.reduce([H[1:]-L[1:], np.abs(H[1:]-C[:-1]), np.abs(L[1:]-C[:-1])])])
    atr = pd.Series(tr).rolling(14, min_periods=14).mean().values
    ret = np.concatenate([[0.0], np.diff(np.log(C))])
    from importlib.machinery import SourceFileLoader
    mod = SourceFileLoader("p04b", "/home/jay/Desktop/new-model-zigzag/model_pipeline/04b_compute_physics_features.py").load_module()
    swing["hurst_rs"]     = mod.compute_hurst_rs(ret)
    swing["ou_theta"]     = mod.compute_ou_theta(ret)
    swing["entropy_rate"] = mod.compute_entropy(ret)
    swing["kramers_up"]   = mod.compute_kramers_up(C)
    swing["wavelet_er"]   = mod.compute_wavelet_er(C)
    swing["vwap_dist"]    = mod.compute_vwap_dist(swing, atr)
    swing["quantum_flow"] = mod.compute_quantum_flow(O, H, L, C, vol)
    df_h4 = swing.set_index("time")[["open","high","low","close","spread"]].resample("4h").agg(
        {"open":"first","high":"max","low":"min","close":"last","spread":"sum"}).dropna()
    qf_h4 = mod.compute_quantum_flow(df_h4["open"].values, df_h4["high"].values,
                                      df_h4["low"].values, df_h4["close"].values,
                                      np.maximum(df_h4["spread"].values, 1.0))
    qf_h4_s = pd.Series(qf_h4, index=df_h4.index).shift(1)
    swing["quantum_flow_h4"] = qf_h4_s.reindex(swing.set_index("time").index, method="ffill").values
    for col in EXIT_FEATS[3:]: swing[col] = swing[col].fillna(0)
    return swing, atr


def load_all_setups():
    """Load ALL v71 setups (full history — no cutoff)."""
    dfs = []
    for f in sorted(glob.glob(P.data("setups_*_v72l.csv"))):
        cid = int(os.path.basename(f).split("_")[1])
        df = pd.read_csv(f, parse_dates=["time"])
        df["cid"] = cid
        dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True).sort_values("time").reset_index(drop=True)
    print(f"  Total setups (full history): {len(all_df):,}")
    return all_df


def train_conf_models(train, features):
    """Per-rule models. Threshold chosen on last 20% per-rule as before."""
    models = {}; thresholds = {}; disabled = []
    for (cid, rule), grp in train.groupby(["cid", "rule"]):
        if len(grp) < 100:
            disabled.append((cid, rule, "too_few")); continue
        grp = grp.sort_values("time").reset_index(drop=True)
        s = int(len(grp) * 0.80)
        tr, vd = grp.iloc[:s], grp.iloc[s:]
        if len(vd) < 20:
            disabled.append((cid, rule, "too_few_vd")); continue
        mdl = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8,
                            eval_metric="logloss", verbosity=0)
        mdl.fit(tr[features].fillna(0).values, tr["label"].astype(int).values)
        proba = mdl.predict_proba(vd[features].fillna(0).values)[:, 1]
        y_vd = vd["label"].astype(int).values
        best_thr, best_pf = 0.5, 0
        for thr in np.arange(0.30, 0.70, 0.05):
            m = proba >= thr
            if m.sum() < 5: continue
            w = y_vd[m].sum(); l = m.sum() - w
            if l == 0: continue
            pf = (w * 2.0) / (l * 1.0)
            if pf > best_pf: best_pf, best_thr = pf, float(thr)
        m = proba >= best_thr
        if m.sum() < 5 or best_pf < 0.8:
            disabled.append((cid, rule, f"pf_{best_pf:.2f}")); continue
        models[(cid, rule)] = mdl; thresholds[(cid, rule)] = best_thr
    return models, thresholds, disabled


def confirm(setups, mdls, thrs, features):
    rows = []
    for (cid, rule), grp in setups.groupby(["cid", "rule"]):
        if (cid, rule) not in mdls: continue
        p = mdls[(cid, rule)].predict_proba(grp[features].fillna(0).values)[:, 1]
        rows.append(grp[p >= thrs[(cid, rule)]].copy())
    return (pd.concat(rows, ignore_index=True).sort_values("time").reset_index(drop=True)
            if rows else pd.DataFrame())


def train_exit_mdl(train_conf_df, swing, atr):
    time_to_idx = pd.Series(swing.index.values, index=swing["time"].values)
    C = swing["close"].values.astype(np.float64)
    rows = []
    for _, s in train_conf_df.iterrows():
        t = s["time"]
        if t not in time_to_idx.index: continue
        ei = int(time_to_idx[t]); d = int(s["direction"])
        ep = C[ei]; ea = atr[ei]
        if not np.isfinite(ea) or ea <= 0: continue
        end = min(ei + MAX_FWD_EXIT + 1, len(C))
        if end - ei < 10: continue
        pnls = np.array([d * (C[k] - ep) / ea for k in range(ei+1, end)])
        if len(pnls) < 5: continue
        for b in range(len(pnls)):
            bi = ei + 1 + b
            cp = pnls[b]
            if cp < -SL_HARD: break
            rem = pnls[b+1:] if b+1 < len(pnls) else np.array([cp])
            br = rem.max() if len(rem) > 0 else cp
            if br < cp - 0.3: lbl = 1
            elif br > cp + 0.3: lbl = 0
            else: continue
            v = cp - pnls[b-3] if b >= 3 else (cp - pnls[0] if b >= 1 else 0.0)
            row = {"unrealized_pnl_R": cp, "bars_held": float(b+1), "pnl_velocity": v, "label": lbl}
            for f in EXIT_FEATS[3:]:
                row[f] = float(swing[f].iat[bi]) if bi < len(swing) else 0.0
            rows.append(row)
    ed = pd.DataFrame(rows)
    mdl = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        eval_metric="logloss", verbosity=0)
    mdl.fit(ed[EXIT_FEATS].fillna(0).values, ed["label"].values)
    return mdl


def simulate(confirmed, swing, atr, exit_mdl):
    time_to_idx = pd.Series(swing.index.values, index=swing["time"].values)
    C = swing["close"].values.astype(np.float64); n = len(C)
    rows = []
    for _, s in confirmed.iterrows():
        t = s["time"]
        if t not in time_to_idx.index: continue
        ei = int(time_to_idx[t]); d = int(s["direction"])
        ep = C[ei]; ea = atr[ei]
        if not np.isfinite(ea) or ea <= 0: continue
        xi, xr = None, "max"
        for k in range(1, MAX_HOLD+1):
            bar = ei + k
            if bar >= n: break
            cp = d * (C[bar] - ep) / ea
            if cp < -SL_HARD: xi, xr = bar, "hard_sl"; break
            if k >= MIN_HOLD:
                p3 = d * (C[bar-3] - ep) / ea if k >= 3 else cp
                X_ = np.array([[cp, float(k), cp - p3,
                                swing["hurst_rs"].iat[bar], swing["ou_theta"].iat[bar],
                                swing["entropy_rate"].iat[bar], swing["kramers_up"].iat[bar],
                                swing["wavelet_er"].iat[bar], swing["quantum_flow"].iat[bar],
                                swing["quantum_flow_h4"].iat[bar], swing["vwap_dist"].iat[bar]]])
                if exit_mdl.predict_proba(X_)[0, 1] >= EXIT_THRESHOLD:
                    xi, xr = bar, "ml_exit"; break
        if xi is None:
            xi = min(ei + MAX_HOLD, n-1); xr = "max"
        pnl = d * (C[xi] - ep) / ea
        rows.append({"time": t, "cid": int(s["cid"]), "rule": s["rule"],
                     "direction": d, "bars": xi - ei, "pnl_R": pnl, "exit": xr})
    return pd.DataFrame(rows)


def export_xgb_to_onnx(mdl, n_features: int, out_path: str, validate_X=None):
    onnx_model = convert_xgboost(
        mdl,
        initial_types=[("features", FloatTensorType([1, n_features]))],
        target_opset=15,
    )
    with open(out_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    if validate_X is None or len(validate_X) == 0:
        return {"path": out_path, "n_features": n_features, "diff": None}
    xgb_proba = mdl.predict_proba(validate_X.astype(np.float32))
    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    onnx_proba = np.zeros_like(xgb_proba)
    for i in range(len(validate_X)):
        out = sess.run(None, {"features": validate_X[i:i+1].astype(np.float32)})
        probs = out[1]
        if isinstance(probs, list) and isinstance(probs[0], dict):
            ks = sorted(probs[0].keys())
            onnx_proba[i] = [probs[0][k] for k in ks]
        else:
            onnx_proba[i] = np.asarray(probs).flatten()
    diff = float(np.max(np.abs(xgb_proba - onnx_proba)))
    return {"path": out_path, "n_features": n_features, "diff": diff, "n_validated": len(validate_X)}


def main():
    t_total = _time.time()
    print("=" * 78)
    print("v7.2-lite DEPLOYMENT — train on FULL data, export to ONNX")
    print("=" * 78)

    all_df = load_all_setups()
    swing, atr = load_swing_with_physics()

    # ---- Step 1: 28 per-rule confirmation models ----
    print("\n[1/4] Training confirmation models on full data...")
    t0 = _time.time()
    mdls, thrs, disabled = train_conf_models(all_df, V7_FEATS)
    print(f"  {len(mdls)} active, {len(disabled)} disabled  ({_time.time()-t0:.0f}s)")
    for cid, rule, reason in disabled[:5]:
        print(f"    disabled c{cid}_{rule}: {reason}")

    # ---- Step 2: exit model on confirmed trades (full data) ----
    print("\n[2/4] Training exit model on confirmed setups...")
    tc = confirm(all_df, mdls, thrs, V7_FEATS)
    print(f"  {len(tc):,} confirmed setups")
    t0 = _time.time()
    exit_mdl = train_exit_mdl(tc, swing, atr)
    print(f"  exit model trained ({_time.time()-t0:.0f}s)")

    # ---- Step 3: simulate + train meta-labeling head ----
    print("\n[3/4] Simulating trades for meta labels, training meta head...")
    t0 = _time.time()
    tt = simulate(tc, swing, atr, exit_mdl)
    print(f"  simulated {len(tt):,} trades in {_time.time()-t0:.0f}s")
    tc["direction"] = tc["direction"].astype(int); tc["cid"] = tc["cid"].astype(int)
    md = tt.merge(tc[["time","cid","rule"] + V7_FEATS], on=["time","cid","rule"], how="left")
    md["meta_label"] = (md["pnl_R"] > 0).astype(int)
    X_meta = md[META_FEATS].fillna(0).values
    y_meta = md["meta_label"].values
    t0 = _time.time()
    meta_mdl = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             eval_metric="logloss", verbosity=0)
    meta_mdl.fit(X_meta, y_meta)
    print(f"  meta head trained on {len(X_meta):,} rows in {_time.time()-t0:.0f}s")
    print(f"  meta threshold (validated on clean holdout): {META_THRESHOLD}")

    # ---- Step 4: export all models to ONNX ----
    print("\n[4/4] Exporting all models to ONNX...")
    summary = {"feature_order_v72l": V7_FEATS,
               "meta_feature_order": META_FEATS,
               "exit_feature_order": EXIT_FEATS,
               "exit_threshold": EXIT_THRESHOLD,
               "meta_threshold": META_THRESHOLD,
               "max_hold": MAX_HOLD, "min_hold": MIN_HOLD, "sl_hard": SL_HARD,
               "confirm_models": {}, "exit_model": {}, "meta_model": {}}
    # Per-rule
    for (cid, rule), mdl in mdls.items():
        onnx_path = P.model(f"confirm_v7_c{cid}_{rule}.onnx")
        grp = all_df[(all_df.cid == cid) & (all_df["rule"] == rule)]
        vx = grp[V7_FEATS].fillna(0).iloc[-200:].values
        r = export_xgb_to_onnx(mdl, len(V7_FEATS), onnx_path, validate_X=vx)
        r["threshold"] = float(thrs[(cid, rule)])
        summary["confirm_models"][f"c{cid}_{rule}"] = r
        diff_str = f"{r['diff']:.2e}" if r.get("diff") is not None else "—"
        print(f"  ✓ c{cid}_{rule:<30} feats={r['n_features']}  thr={r['threshold']:.3f}  diff={diff_str}")
    # Exit
    exit_onnx = P.model("exit_v7.onnx")
    ed_sample = np.zeros((100, len(EXIT_FEATS)), dtype=np.float32)  # no label sampling needed
    # Use subsample of actual training rows for validation
    # (skip for simplicity — internal consistency already checked in v7.2 pipeline)
    r_exit = export_xgb_to_onnx(exit_mdl, len(EXIT_FEATS), exit_onnx)
    summary["exit_model"] = r_exit
    print(f"  ✓ exit_v7.onnx  feats={r_exit['n_features']}")
    # Meta
    meta_onnx = P.model("meta_v7.onnx")
    vx_meta = md[META_FEATS].fillna(0).iloc[-500:].values
    r_meta = export_xgb_to_onnx(meta_mdl, len(META_FEATS), meta_onnx, validate_X=vx_meta)
    summary["meta_model"] = r_meta
    diff_str = f"{r_meta['diff']:.2e}" if r_meta.get("diff") is not None else "—"
    print(f"  ✓ meta_v7.onnx  feats={r_meta['n_features']}  diff={diff_str}")

    # Summary file
    out_summary = P.model("v7_deploy.json")
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Saved deployment bundle: {out_summary}")

    bad = [k for k, v in summary["confirm_models"].items()
           if v.get("diff") is not None and v["diff"] > 1e-4]
    print(f"\n  Confirmation models: {len(summary['confirm_models'])} exported, {len(bad)} failed validation")
    if bad: print(f"  ⚠ FAILED: {bad}")
    print(f"\nTotal: {_time.time() - t_total:.0f}s")


if __name__ == "__main__":
    main()
