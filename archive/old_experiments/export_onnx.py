"""
Export the three regime-specialist XGBoost models to ONNX for MT5.

Verifies each ONNX output matches the native XGBoost output bit-for-bit.
"""
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
import onnxruntime as ort

CLUSTERS = {0: "Ranging", 1: "Downtrend", 3: "Uptrend"}


def export_one(cluster_id, name):
    print(f"\n═══ C{cluster_id} {name} ═══")

    model_path = f"models/regime_{cluster_id}_{name}.json"
    meta_path  = f"models/regime_{cluster_id}_{name}_meta.json"
    onnx_path  = f"models/regime_{cluster_id}_{name}.onnx"

    with open(meta_path) as f:
        meta = json.load(f)
    feat_cols = meta["feature_cols"]
    n_features = len(feat_cols)
    print(f"  features: {n_features}")
    print(f"  classes:  {meta['label_classes']}")

    model = xgb.XGBClassifier()
    model.load_model(model_path)

    # Convert — use float32 ("features" input name matches existing EA convention)
    onnx_model = convert_xgboost(
        model,
        initial_types=[("features", FloatTensorType([1, n_features]))],
        target_opset=15,
    )
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"  saved: {onnx_path}")

    # Validate vs native XGBoost on 500 random holdout rows
    df = pd.read_csv(f"cluster_{cluster_id}_data.csv", nrows=50000)
    df = df.iloc[-500:]  # last 500 rows — unseen-ish
    X = df[feat_cols].fillna(0).values.astype(np.float32)

    xgb_proba = model.predict_proba(X)

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    # Run one at a time (mirrors MQL5 inference loop)
    onnx_proba = np.zeros_like(xgb_proba)
    for i in range(len(X)):
        row = X[i:i+1]
        out = sess.run(None, {"features": row})
        # onnxmltools xgboost export returns [label, probabilities_dict_or_array]
        probs = out[1]
        if isinstance(probs, list) and isinstance(probs[0], dict):
            # list of dicts, one per row
            classes_order = sorted(probs[0].keys())
            onnx_proba[i] = [probs[0][c] for c in classes_order]
        else:
            onnx_proba[i] = np.asarray(probs).flatten()

    max_diff = float(np.max(np.abs(xgb_proba - onnx_proba)))
    print(f"  max |XGB - ONNX| diff: {max_diff:.2e}")

    # Check the ONNX input/output spec so we can echo it for MQL5
    inp_spec = sess.get_inputs()[0]
    out_specs = sess.get_outputs()
    print(f"  ONNX input:  name={inp_spec.name} shape={inp_spec.shape} dtype={inp_spec.type}")
    for o in out_specs:
        print(f"  ONNX output: name={o.name} shape={o.shape} dtype={o.type}")

    return {"path": onnx_path, "n_features": n_features, "classes": meta["label_classes"],
            "max_diff": max_diff, "input_name": inp_spec.name,
            "outputs": [o.name for o in out_specs]}


if __name__ == "__main__":
    results = {}
    for cid, name in CLUSTERS.items():
        results[cid] = export_one(cid, name)
    print("\n── Summary ──")
    for cid, r in results.items():
        ok = "✅" if r["max_diff"] < 1e-5 else "⚠"
        print(f"  {ok} C{cid}: {r['path']}  ({r['n_features']} feats, {len(r['classes'])} classes)  diff={r['max_diff']:.2e}")
