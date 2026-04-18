"""
verify_live_trace.py — compare live MQL5 rule evaluations against the
Python training pipeline bar-by-bar.

Input:  trace_XAUUSD.csv  (from trace_logger.mqh in the EA)
Lookup: setups_{cid}.csv   (Python-computed training features per setup)
        confirm_c{cid}_{rule}.json (+ _meta.json) — trained XGB model

For each trace row, we find the matching Python-pipeline row at the same
(time, rule) and report:
  1. Per-feature absolute and relative diffs.
  2. XGB probability for live-feats vs python-feats — if classifier agrees
     on both but features differ, features are a problem; if feats match
     but probs differ, ONNX conversion is a problem; if both differ, check
     feature order and scaling.

Usage:
    python3 verify_live_trace.py --trace trace_XAUUSD.csv \\
        --setups-dir data --models-dir models
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np
import pandas as pd
from xgboost import XGBClassifier


def load_trace(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    # features column is comma-separated
    feat_mat = df["features"].str.split(",", expand=True).astype(float).values
    df["_feat"] = list(feat_mat)
    df["bar_time"] = pd.to_datetime(df["bar_time"])
    return df


def find_setup_row(setups_by_cid: dict, cid: int, rule: str, t: pd.Timestamp):
    df = setups_by_cid.get(cid)
    if df is None: return None
    m = (df["rule"] == rule) & (df["time"] == t)
    hits = df[m]
    if len(hits) == 0: return None
    return hits.iloc[0]


def compare_row(trace_row, setup_row, feat_cols, model, thr):
    live = np.array(trace_row["_feat"], dtype=float)
    py   = setup_row[feat_cols].astype(float).fillna(0).values

    if len(live) != len(py):
        return {"ok": False, "reason": f"length mismatch live={len(live)} py={len(py)}"}

    adiff = np.abs(live - py)
    denom = np.maximum(np.abs(py), 1e-9)
    rdiff = adiff / denom

    # classifier probabilities
    p_live = float(model.predict_proba(live.reshape(1, -1))[:, 1][0])
    p_py   = float(model.predict_proba(py.reshape(1, -1))[:, 1][0])

    # top-5 worst features by absolute diff
    worst = np.argsort(-adiff)[:5]
    worst_rep = [(feat_cols[i], float(live[i]), float(py[i]),
                  float(adiff[i]), float(rdiff[i])) for i in worst]

    return {
        "ok": True,
        "n_features": len(live),
        "max_adiff": float(adiff.max()),
        "max_rdiff": float(rdiff.max()),
        "n_sig_diff": int((adiff > 1e-4).sum()),
        "prob_live": p_live,
        "prob_py": p_py,
        "prob_diff": p_live - p_py,
        "trace_prob_reported": float(trace_row["prob"]),
        "onnx_vs_xgb_diff": float(trace_row["prob"]) - p_live,
        "confirmed_live": bool(trace_row["confirmed"]),
        "would_confirm_py": p_py >= thr,
        "threshold": thr,
        "worst_features": worst_rep,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--setups-dir", required=True,
                    help="directory containing setups_{cid}.csv")
    ap.add_argument("--models-dir", required=True,
                    help="directory containing confirm_c{cid}_{rule}.json")
    ap.add_argument("--setup-pattern", default="setups_{cid}.csv",
                    help="e.g. setups_{cid}.csv or setup_signals_eurusd.csv")
    ap.add_argument("--max-rows", type=int, default=200)
    args = ap.parse_args()

    trace = load_trace(args.trace)
    print(f"Loaded {len(trace)} trace rows from {args.trace}")
    print(f"  rules seen: {sorted(trace['rule_name'].unique())[:8]}"
          f" ({trace['rule_name'].nunique()} unique)")

    # load setup CSVs (one per cluster for XAU; single file for EU/GJ)
    setups_by_cid: dict[int, pd.DataFrame] = {}
    if "{cid}" in args.setup_pattern:
        for p in glob.glob(os.path.join(args.setups_dir,
                                        args.setup_pattern.replace("{cid}", "*"))):
            cid = int(p.rsplit("_", 1)[-1].split(".")[0])
            df = pd.read_csv(p, parse_dates=["time"])
            setups_by_cid[cid] = df
    else:
        df = pd.read_csv(os.path.join(args.setups_dir, args.setup_pattern),
                         parse_dates=["time"])
        for cid, g in df.groupby("cluster"):
            setups_by_cid[int(cid)] = g.reset_index(drop=True)
    print(f"  setup pools loaded: clusters {sorted(setups_by_cid.keys())}")

    # diff each trace row
    model_cache: dict[tuple[int, str], tuple[XGBClassifier, list[str], float]] = {}
    results = []
    not_found = 0
    for _, tr in trace.head(args.max_rows).iterrows():
        cid = int(tr["cluster"])
        rule = tr["rule_name"]
        key = (cid, rule)
        if key not in model_cache:
            mp = os.path.join(args.models_dir, f"confirm_c{cid}_{rule}.json")
            mm = os.path.join(args.models_dir, f"confirm_c{cid}_{rule}_meta.json")
            if not (os.path.exists(mp) and os.path.exists(mm)):
                model_cache[key] = (None, None, None)
            else:
                meta = json.load(open(mm))
                mdl = XGBClassifier(); mdl.load_model(mp)
                model_cache[key] = (mdl, meta["feature_cols"], meta["threshold"])
        model, feat_cols, thr = model_cache[key]
        if model is None:
            print(f"  [skip] no model for c{cid}_{rule}"); continue
        sr = find_setup_row(setups_by_cid, cid, rule, tr["bar_time"])
        if sr is None:
            not_found += 1
            results.append({"bar_time": tr["bar_time"], "cid": cid, "rule": rule,
                            "status": "NO_PYTHON_SETUP_AT_TIME"})
            continue
        r = compare_row(tr, sr, feat_cols, model, thr)
        r.update({"bar_time": tr["bar_time"], "cid": cid, "rule": rule})
        results.append(r)

    # summary
    print(f"\n{'='*72}\nSUMMARY over {len(results)} trace rows\n{'='*72}")
    ok = [r for r in results if r.get("ok")]
    print(f"  python setup not found: {not_found}")
    print(f"  compared with python:   {len(ok)}")

    if ok:
        max_adiff = max(r["max_adiff"] for r in ok)
        max_pdiff = max(abs(r["prob_diff"]) for r in ok)
        max_onnx  = max(abs(r["onnx_vs_xgb_diff"]) for r in ok)
        print(f"  max feature abs-diff: {max_adiff:.6f}")
        print(f"  max (prob_live - prob_py): {max_pdiff:.4f}")
        print(f"  max (trace_onnx_prob - xgb_on_live_feats): {max_onnx:.4f}")

        worst = sorted(ok, key=lambda r: -r["max_adiff"])[:3]
        for w in worst:
            print(f"\n  WORST  c{w['cid']} {w['rule']} @ {w['bar_time']}")
            print(f"    max_adiff={w['max_adiff']:.6f}  prob live={w['prob_live']:.3f}"
                  f"  py={w['prob_py']:.3f}  reported_onnx={w['trace_prob_reported']:.3f}")
            for name, lv, pv, ad, rd in w["worst_features"]:
                print(f"    {name:<20} live={lv:+.6f}  py={pv:+.6f}  "
                      f"Δ={ad:.6f} ({rd*100:.2f}%)")

    # write full report
    out = args.trace.replace(".csv", "_diff.json")
    with open(out, "w") as f:
        json.dump(results, f, default=str, indent=2)
    print(f"\nFull per-row report: {out}")


if __name__ == "__main__":
    main()
