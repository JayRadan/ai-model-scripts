"""
v7.4 Pivot Score — walk-forward validation.

Five sequential 12-month holdout windows. Each fold:
  1. Train pivot-score + direction models on bars BEFORE fold start
  2. Score every bar (model never saw fold bars)
  3. Build setups (filter to fold window only)
  4. Run Oracle pipeline (confirm + exit + meta) on fold setups; train on
     pre-fold portion, evaluate on fold window
  5. Record fold metrics

Folds (anchor = fold START):
  F1: 2022-04-13 → 2023-04-13
  F2: 2023-04-13 → 2024-04-13
  F3: 2024-04-13 → 2025-04-13
  F4: 2025-04-13 → 2026-04-13   (this is the closest to our original holdout)
  F5: skip if data ends earlier

Reuses already-computed:
  data/features_v74.csv
  data/labels_v74.csv
  ../v73_pivot_oracle/data/cluster_per_bar_v73.csv

NO models are written to disk between folds — everything in memory.
"""
from __future__ import annotations
import os, json, glob, sys, time as _time
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "/home/jay/Desktop/new-model-zigzag/model_pipeline")
import paths as P

DATA = "/home/jay/Desktop/new-model-zigzag/experiments/v74_pivot_score/data"
RPT = "/home/jay/Desktop/new-model-zigzag/experiments/v74_pivot_score/reports"
CLUSTERS_PATH = "/home/jay/Desktop/new-model-zigzag/experiments/v73_pivot_oracle/data/cluster_per_bar_v73.csv"

FOLDS = [
    ("F1_2022", pd.Timestamp("2022-04-13"), pd.Timestamp("2023-04-13")),
    ("F2_2023", pd.Timestamp("2023-04-13"), pd.Timestamp("2024-04-13")),
    ("F3_2024", pd.Timestamp("2024-04-13"), pd.Timestamp("2025-04-13")),
    ("F4_2025", pd.Timestamp("2025-04-13"), pd.Timestamp("2026-04-13")),
]

SCORE_THR = 0.30
TARGET_COL = "is_pivot_25"

# v72L feature list — for the per-cluster confirm heads + meta + exit (must match Oracle)
V72L_FEATS = [
    "hurst_rs", "ou_theta", "entropy_rate", "kramers_up", "wavelet_er",
    "vwap_dist", "hour_enc", "dow_enc",
    "quantum_flow", "quantum_flow_h4", "quantum_momentum", "quantum_vwap_conf",
    "quantum_divergence", "quantum_div_strength",
    "vpin", "sig_quad_var", "har_rv_ratio", "hawkes_eta",
]
META_FEATS = V72L_FEATS + ["direction", "cid"]
EXIT_FEATS = [
    "unrealized_pnl_R", "bars_held", "pnl_velocity",
    "hurst_rs", "ou_theta", "entropy_rate", "kramers_up", "wavelet_er",
    "quantum_flow", "quantum_flow_h4", "vwap_dist",
]
NON_FEAT_PIVOT = {"time", "bar_idx", "atr", "best_long_R", "best_short_R",
                   "best_R", "best_dir", "is_pivot_15", "is_pivot_25", "is_pivot_4"}
# Drop the 21 f01-f20 base chart features — they're pre-computed in the swing
# CSV and not available on the server (EA only sends OHLCV). They contributed
# zero importance in the top-20 anyway. Pivot model uses 32 features instead
# of 53 after this drop.
SKIP_F_FEATS = True
def _is_f_feat(c: str) -> bool:
    if not SKIP_F_FEATS: return False
    if not c.startswith("f"): return False
    if len(c) < 4: return False
    return c[1:3].isdigit() and c[3] == "_"

XGB_S = dict(n_estimators=400, max_depth=5, learning_rate=0.05,
              subsample=0.85, colsample_bytree=0.85, objective="binary:logistic",
              eval_metric="auc", tree_method="hist", n_jobs=4, verbosity=0)
XGB_C = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
              subsample=0.8, colsample_bytree=0.8, objective="binary:logistic",
              eval_metric="auc", tree_method="hist", n_jobs=4, verbosity=0)

MAX_HOLD = 60; SL_HARD = 4.0
EXIT_THRESHOLD = 0.55; MAX_FWD_EXIT = 60


def compute_atr(H, L, C, n=14):
    tr = np.concatenate([
        [H[0] - L[0]],
        np.maximum.reduce([H[1:] - L[1:], np.abs(H[1:] - C[:-1]), np.abs(L[1:] - C[:-1])]),
    ])
    return pd.Series(tr).rolling(n, min_periods=n).mean().values


def build_swing_with_exit_feats():
    """Replicate Oracle's load_swing_with_physics — needed for the exit head sim."""
    swing = pd.read_csv(P.data("swing_v5_xauusd.csv"), parse_dates=["time"])
    swing = swing.sort_values("time").reset_index(drop=True)
    H = swing["high"].values.astype(np.float64); L = swing["low"].values.astype(np.float64)
    C = swing["close"].values.astype(np.float64)
    atr = compute_atr(H, L, C, 14)
    # Pull EXIT_FEATS context (hurst, ou_theta, etc.) from features file
    feats = pd.read_csv(os.path.join(DATA, "features_v74.csv"), parse_dates=["time"])
    keep = ["time"] + [c for c in EXIT_FEATS[3:] if c in feats.columns]
    swing = swing.merge(feats[keep], on="time", how="left")
    for col in EXIT_FEATS[3:]:
        if col in swing.columns:
            swing[col] = swing[col].fillna(0)
    return swing, atr


def train_conf(train_df, feat_list):
    """Per-(cid, rule) confirm head. Returns (mdls, thrs)."""
    mdls = {}; thrs = {}
    for (cid, rule), grp in train_df.groupby(["cid", "rule"]):
        if len(grp) < 100 or grp["label"].nunique() < 2: continue
        X = grp[feat_list].fillna(0).values; y = grp["label"].values
        mdl = XGBClassifier(**XGB_C); mdl.fit(X, y)
        p = mdl.predict_proba(X)[:, 1]
        # threshold = pick by max F1
        best_f1, best_thr = 0, 0.5
        for thr in np.linspace(0.3, 0.85, 56):
            yhat = (p >= thr).astype(int)
            tp = ((yhat == 1) & (y == 1)).sum(); fp = ((yhat == 1) & (y == 0)).sum()
            fn = ((yhat == 0) & (y == 1)).sum()
            if tp == 0: continue
            prec = tp / (tp + fp); rec = tp / (tp + fn)
            f1 = 2 * prec * rec / (prec + rec)
            if f1 > best_f1: best_f1, best_thr = f1, thr
        mdls[(cid, rule)] = mdl; thrs[(cid, rule)] = best_thr
    return mdls, thrs


def filter_setups(setups, mdls, thrs, feat_list):
    rows = []
    for (cid, rule), grp in setups.groupby(["cid", "rule"]):
        if (cid, rule) not in mdls: continue
        X = grp[feat_list].fillna(0).values
        p = mdls[(cid, rule)].predict_proba(X)[:, 1]
        rows.append(grp[p >= thrs[(cid, rule)]].copy())
    if not rows: return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values("time").reset_index(drop=True)


def train_exit(conf_df, swing, atr):
    """Generate exit training data by simulating each confirmed trade forward."""
    H = swing["high"].values; L = swing["low"].values; C = swing["close"].values
    feats_arr = {col: swing[col].values for col in EXIT_FEATS[3:]}
    rows = []
    for _, s in conf_df.iterrows():
        ei = int(s["idx"]); d = int(s["direction"]); a = float(s["atr"])
        if a <= 0: continue
        end = min(ei + MAX_FWD_EXIT + 1, len(C))
        sl = SL_HARD * a; entry_px = C[ei]
        for k in range(1, end - ei):
            bar = ei + k
            if d == 1:
                if (entry_px - L[bar]) >= sl: break
                pnl = (C[bar] - entry_px) / sl
            else:
                if (H[bar] - entry_px) >= sl: break
                pnl = (entry_px - C[bar]) / sl
            future_max = pnl
            for k2 in range(k + 1, end - ei):
                b2 = ei + k2
                if d == 1: r2 = (C[b2] - entry_px) / sl
                else:      r2 = (entry_px - C[b2]) / sl
                if r2 > future_max: future_max = r2
            label = int(future_max <= pnl)   # exit now is at least as good
            row = {"unrealized_pnl_R": pnl, "bars_held": k,
                   "pnl_velocity": pnl / k, "label": label}
            for f in EXIT_FEATS[3:]:
                row[f] = feats_arr[f][bar] if f in feats_arr else 0.0
            rows.append(row)
    if not rows: return None
    ed = pd.DataFrame(rows)
    mdl = XGBClassifier(**XGB_C); mdl.fit(ed[EXIT_FEATS].fillna(0).values, ed["label"].values)
    return mdl


def simulate(confirmed, swing, atr, exit_mdl):
    H = swing["high"].values; L = swing["low"].values; C = swing["close"].values
    n = len(C)
    feats_arr = {col: swing[col].values for col in EXIT_FEATS[3:]}
    entries = []
    for _, s in confirmed.iterrows():
        ei = int(s["idx"])
        if ei + MAX_HOLD >= n: continue
        entries.append((ei, int(s["direction"]), s["time"], int(s["cid"]), s["rule"], float(s["atr"])))

    rows = []
    # Build batch exit-feature tensor: [trade*MAX_HOLD, len(EXIT_FEATS)]
    if not entries: return pd.DataFrame()
    X = np.zeros((len(entries) * MAX_HOLD, len(EXIT_FEATS)), dtype=np.float64)
    valid = np.zeros(len(entries) * MAX_HOLD, dtype=bool)
    for r, (ei, d, t, cid_v, rule_v, a) in enumerate(entries):
        if a <= 0: continue
        sl = SL_HARD * a; entry_px = C[ei]
        end = min(ei + MAX_HOLD + 1, n)
        for k in range(1, end - ei):
            bar = ei + k
            if d == 1: pnl = (C[bar] - entry_px) / sl
            else:      pnl = (entry_px - C[bar]) / sl
            row_idx = r * MAX_HOLD + (k - 1)
            X[row_idx, 0] = pnl; X[row_idx, 1] = k; X[row_idx, 2] = pnl / k
            for fi, f in enumerate(EXIT_FEATS[3:]):
                X[row_idx, 3 + fi] = feats_arr[f][bar] if f in feats_arr else 0.0
            valid[row_idx] = True

    probs = np.zeros(len(entries) * MAX_HOLD)
    if valid.any():
        probs[valid] = exit_mdl.predict_proba(X[valid])[:, 1]

    for rank, (ei, d, t, cid_v, rule_v, a) in enumerate(entries):
        if a <= 0: continue
        sl = SL_HARD * a; entry_px = C[ei]
        end = min(ei + MAX_HOLD + 1, n)
        xi, xr = end - 1, "max_hold"; pnl = 0.0
        for k in range(1, end - ei):
            bar = ei + k
            if d == 1:
                if (entry_px - L[bar]) >= sl: xi, xr, pnl = bar, "sl", -1.0; break
                pnl_now = (C[bar] - entry_px) / sl
            else:
                if (H[bar] - entry_px) >= sl: xi, xr, pnl = bar, "sl", -1.0; break
                pnl_now = (entry_px - C[bar]) / sl
            if k > 2 and probs[rank * MAX_HOLD + (k - 1)] >= EXIT_THRESHOLD:
                xi, xr, pnl = bar, "ml_exit", pnl_now; break
            pnl = pnl_now
        rows.append({"time": t, "cid": cid_v, "rule": rule_v, "direction": d,
                      "bars": xi - ei, "pnl_R": pnl, "exit": xr})
    return pd.DataFrame(rows)


def report(df, name=""):
    if not len(df): return {"name": name, "n": 0}
    ww = df[df["pnl_R"] > 0]; ll = df[df["pnl_R"] <= 0]
    pf = (ww["pnl_R"].sum() / -ll["pnl_R"].sum()) if len(ll) and ll["pnl_R"].sum() < 0 else float("inf")
    cum = df["pnl_R"].cumsum().values
    dd = float((cum - np.maximum.accumulate(cum)).min()) if len(cum) else 0.0
    return {"name": name, "n": int(len(df)), "wr": float(len(ww) / len(df)),
            "pf": float(pf), "expect": float(df["pnl_R"].mean()),
            "total_R": float(df["pnl_R"].sum()), "dd": float(dd)}


def run_one_fold(name, fold_start, fold_end, df, swing, atr, clusters_map):
    print(f"\n========== {name}  ({fold_start.date()} -> {fold_end.date()}) ==========", flush=True)
    t0 = _time.time()

    pre = df[df["time"] < fold_start].reset_index(drop=True)
    fold = df[(df["time"] >= fold_start) & (df["time"] < fold_end)].reset_index(drop=True)
    print(f"  pre={len(pre):,}  fold={len(fold):,}  pre_pivot_rate={pre[TARGET_COL].mean():.1%}", flush=True)

    pivot_feats = [c for c in df.columns if c not in NON_FEAT_PIVOT
                    and df[c].dtype != object and not c.endswith("_lab")
                    and not _is_f_feat(c)]

    # 1. Train pivot-score + direction on pre-fold
    print("  [1] training pivot-score model...", flush=True)
    Xtr = pre[pivot_feats].fillna(0).values; ytr = pre[TARGET_COL].values
    score_mdl = XGBClassifier(**XGB_S); score_mdl.fit(Xtr, ytr)
    pos = pre[pre[TARGET_COL] == 1]
    Xp = pos[pivot_feats].fillna(0).values; yp = (pos["best_dir"] == 1).astype(int).values
    dir_mdl = XGBClassifier(**XGB_S); dir_mdl.fit(Xp, yp)

    p_fold = score_mdl.predict_proba(fold[pivot_feats].fillna(0).values)[:, 1]
    pdir_fold = dir_mdl.predict_proba(fold[pivot_feats].fillna(0).values)[:, 1]
    p_pre = score_mdl.predict_proba(pre[pivot_feats].fillna(0).values)[:, 1]
    pdir_pre = dir_mdl.predict_proba(pre[pivot_feats].fillna(0).values)[:, 1]

    auc_fold = roc_auc_score(fold[TARGET_COL].values, p_fold)
    print(f"  pivot-score AUC on fold: {auc_fold:.3f}", flush=True)

    # 2. Build setups (filtered by SCORE_THR), partition by cluster
    def build_setups(parent, p_p, p_d):
        m = p_p >= SCORE_THR
        s = parent[m].copy()
        s["direction"] = np.where(p_d[m] >= 0.5, 1, -1)
        s["rule"] = "RP_score"; s["idx"] = s["bar_idx"]
        s["entry_price"] = s["time"].map(swing.set_index("time")["close"])
        s["label"] = (s["best_R"] >= 1.0).astype(int)
        s = s.merge(clusters_map, on="time", how="left")
        return s

    pre_setups = build_setups(pre, p_pre, pdir_pre)
    fold_setups = build_setups(fold, p_fold, pdir_fold)
    print(f"  setups: pre={len(pre_setups):,}  fold={len(fold_setups):,}", flush=True)
    if not len(fold_setups):
        return {"fold": name, "skipped": True}

    # 3. Train confirm head on pre-setups, filter both
    mdls, thrs = train_conf(pre_setups, V72L_FEATS)
    pre_conf = filter_setups(pre_setups, mdls, thrs, V72L_FEATS)
    fold_conf = filter_setups(fold_setups, mdls, thrs, V72L_FEATS)
    print(f"  confirmed: pre={len(pre_conf):,}  fold={len(fold_conf):,}", flush=True)
    if not len(fold_conf): return {"fold": name, "skipped": True}

    # 4. Train exit head on pre-confirmed
    print(f"  [2] training exit head...", flush=True)
    exit_mdl = train_exit(pre_conf, swing, atr)
    if exit_mdl is None: return {"fold": name, "skipped": True}

    # 5. Simulate trades; train meta on pre, apply to fold
    pre_trades = simulate(pre_conf, swing, atr, exit_mdl)
    if not len(pre_trades): return {"fold": name, "skipped": True}
    md = pre_trades.merge(pre_conf[["time","cid","rule"] + V72L_FEATS], on=["time","cid","rule"], how="left")
    md["meta_label"] = (md["pnl_R"] > 0).astype(int)
    md["direction"] = md["direction"].astype(int); md["cid"] = md["cid"].astype(int)
    val_n = max(500, int(len(md) * 0.10))
    mtr = md.iloc[:-val_n]; mvd = md.iloc[-val_n:]
    meta_mdl = XGBClassifier(**XGB_C)
    meta_mdl.fit(mtr[META_FEATS].fillna(0).values, mtr["meta_label"].values)
    pv = meta_mdl.predict_proba(mvd[META_FEATS].fillna(0).values)[:, 1]
    pn = mvd["pnl_R"].values
    best_thr, best_total = 0.5, -np.inf
    for thr in np.linspace(0.40, 0.80, 17):
        m = pv >= thr
        if m.sum() < 50: continue
        if pn[m].sum() > best_total: best_total = pn[m].sum(); best_thr = thr

    # 6. Holdout sim
    fold_conf["direction"] = fold_conf["direction"].astype(int); fold_conf["cid"] = fold_conf["cid"].astype(int)
    pm = meta_mdl.predict_proba(fold_conf[META_FEATS].fillna(0).values)[:, 1]
    fold_conf_m = fold_conf[pm >= best_thr].reset_index(drop=True)
    fold_trades_nm = simulate(fold_conf, swing, atr, exit_mdl)
    fold_trades_m  = simulate(fold_conf_m, swing, atr, exit_mdl)
    r_nm = report(fold_trades_nm, "no_meta")
    r_m  = report(fold_trades_m, "with_meta")
    days = max(1.0, (fold_end - fold_start).days)
    r_m["tpd"] = r_m["n"] / days; r_nm["tpd"] = r_nm["n"] / days
    print(f"  NO META : n={r_nm['n']:,} WR={r_nm['wr']:.1%} PF={r_nm['pf']:.2f} R={r_nm['total_R']:+.0f} DD={r_nm['dd']:.0f} tpd={r_nm['tpd']:.1f}")
    print(f"  + META  : n={r_m['n']:,} WR={r_m['wr']:.1%} PF={r_m['pf']:.2f} R={r_m['total_R']:+.0f} DD={r_m['dd']:.0f} tpd={r_m['tpd']:.1f}  thr={best_thr:.3f}")
    print(f"  fold time: {_time.time() - t0:.0f}s", flush=True)

    return {"fold": name, "fold_start": str(fold_start.date()), "fold_end": str(fold_end.date()),
            "pivot_auc": float(auc_fold), "meta_thr": float(best_thr),
            "no_meta": r_nm, "with_meta": r_m}


def main():
    print("Loading features + labels + clusters + swing...", flush=True)
    feats = pd.read_csv(os.path.join(DATA, "features_v74.csv"), parse_dates=["time"])
    labs = pd.read_csv(os.path.join(DATA, "labels_v74.csv"), parse_dates=["time"])
    df = feats.merge(labs, on="time", how="inner", suffixes=("", "_lab"))
    clusters = pd.read_csv(CLUSTERS_PATH, parse_dates=["time"])[["time", "cid"]]
    swing, atr = build_swing_with_exit_feats()
    print(f"  rows={len(df):,}  pivot_features={len([c for c in df.columns if c not in NON_FEAT_PIVOT])}", flush=True)

    results = []
    for (name, s, e) in FOLDS:
        try:
            r = run_one_fold(name, s, e, df, swing, atr, clusters)
            results.append(r)
        except Exception as ex:
            import traceback; traceback.print_exc()
            results.append({"fold": name, "error": str(ex)})

    print("\n" + "=" * 80)
    print(f"WALK-FORWARD SUMMARY — v7.4 Pivot Score (thr={SCORE_THR}, target={TARGET_COL})")
    print("=" * 80)
    print(f"  {'fold':<10s} {'AUC':>5s} {'n':>6s} {'WR':>6s} {'PF':>6s} {'R':>9s} {'DD':>8s} {'tpd':>5s}")
    for r in results:
        if r.get("skipped") or "error" in r:
            print(f"  {r['fold']:<10s} SKIPPED/ERROR")
            continue
        m = r["with_meta"]
        print(f"  {r['fold']:<10s} {r['pivot_auc']:>5.2f} {m['n']:>6d} {m['wr']:>6.1%} {m['pf']:>6.2f} {m['total_R']:>+9.0f} {m['dd']:>8.0f} {m['tpd']:>5.1f}")

    out_path = os.path.join(RPT, "v74_walk_forward.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
