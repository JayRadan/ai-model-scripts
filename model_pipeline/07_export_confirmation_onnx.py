"""
Export every per-rule XGBoost binary classifier to ONNX, validated against
the native XGBoost output.
"""
import glob
import json
import os

import numpy as np
import xgboost as xgb
import pandas as pd
import onnxruntime as ort
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType

import paths as P

CLUSTER_NAMES = {0:"Ranging", 1:"Downtrend", 2:"Shock_News", 3:"Uptrend"}


def export_one(cid: int, rule_name: str):
    model_path = P.model(f"confirm_c{cid}_{rule_name}.json")
    meta_path  = P.model(f"confirm_c{cid}_{rule_name}_meta.json")
    onnx_path  = P.model(f"confirm_c{cid}_{rule_name}.onnx")

    if not (os.path.exists(model_path) and os.path.exists(meta_path)):
        return None

    with open(meta_path) as f:
        meta = json.load(f)
    feat_cols = meta["feature_cols"]
    n_features = len(feat_cols)

    model = xgb.XGBClassifier()
    model.load_model(model_path)

    onnx_model = convert_xgboost(
        model,
        initial_types=[("features", FloatTensorType([1, n_features]))],
        target_opset=15,
    )
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    # Validate vs native XGBoost
    setup_csv = P.data(f"setups_{cid}.csv")
    df = pd.read_csv(setup_csv, parse_dates=["time"])
    df = df[df["rule"] == rule_name].sort_values("time").iloc[-300:]
    if len(df) == 0:
        print(f"  C{cid} {rule_name}: no rows to validate")
        return None
    X = df[feat_cols].fillna(0).values.astype(np.float32)
    xgb_proba = model.predict_proba(X)

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_proba = np.zeros_like(xgb_proba)
    for i in range(len(X)):
        out = sess.run(None, {"features": X[i:i+1]})
        probs = out[1]
        if isinstance(probs, list) and isinstance(probs[0], dict):
            ks = sorted(probs[0].keys())
            onnx_proba[i] = [probs[0][k] for k in ks]
        else:
            onnx_proba[i] = np.asarray(probs).flatten()

    diff = float(np.max(np.abs(xgb_proba - onnx_proba)))
    print(f"  C{cid} {rule_name}: feats={n_features}  diff={diff:.2e}  "
          f"({len(X)} validation rows)")
    return {"path": onnx_path, "n_features": n_features,
            "diff": diff, "threshold": meta["threshold"]}


def main():
    print("Exporting per-rule confirmation models to ONNX...\n")
    summary = {}
    for cid in [0, 1, 2, 3]:
        print(f"── C{cid} {CLUSTER_NAMES[cid]} ──")
        rules = sorted(set(
            os.path.basename(p)[len(f"confirm_c{cid}_"):-len(".json")]
            for p in glob.glob(P.model(f"confirm_c{cid}_*.json"))
            if not p.endswith("_meta.json")
        ))
        for r in rules:
            res = export_one(cid, r)
            if res is not None:
                summary[f"c{cid}_{r}"] = res
        print()

    print("── Summary ──")
    bad = [k for k, v in summary.items() if v["diff"] > 1e-4]
    print(f"  exported: {len(summary)} models")
    print(f"  validation OK (diff < 1e-4): {len(summary) - len(bad)}")
    if bad:
        print(f"  ⚠ failed validation: {bad}")
    with open(P.model("onnx_export_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
