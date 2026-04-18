"""Export all 48 EURUSD XGBoost models to ONNX (including disabled ones for index alignment)."""
import json
import glob
import numpy as np
import xgboost as xgb
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
import onnxruntime as ort
import os
import paths as P

FEATURE_DIM = 34
os.makedirs(P.LIVE_DIR / "MQL5_Files", exist_ok=True)

metas = sorted(glob.glob(str(P.MODELS_DIR / "*_meta.json")))
count = 0
max_diff = 0

for meta_path in metas:
    with open(meta_path) as f:
        meta = json.load(f)

    rule = meta["rule"]
    cid = meta["cluster"]
    model_path = P.model(f"confirm_c{cid}_{rule}.json")

    model = xgb.XGBClassifier()
    model.load_model(model_path)

    onnx_model = convert_xgboost(
        model.get_booster(),
        initial_types=[("input", FloatTensorType([1, FEATURE_DIM]))],
        target_opset=15,
    )

    onnx_path = str(P.LIVE_DIR / "MQL5_Files" / f"confirm_c{cid}_{rule}.onnx")
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    # Validate
    sess = ort.InferenceSession(onnx_path)
    test_input = np.random.randn(1, FEATURE_DIM).astype(np.float32)
    xgb_prob = model.predict_proba(test_input)[0][1]
    onnx_out = sess.run(None, {"input": test_input})
    onnx_prob = onnx_out[1][0][1]
    diff = abs(float(xgb_prob) - float(onnx_prob))
    max_diff = max(max_diff, diff)

    count += 1
    status = "DISABLED" if meta.get("disabled") else "ACTIVE"
    print(f"  {rule:28s} → {os.path.basename(onnx_path):45s} diff={diff:.2e}  [{status}]")

print(f"\nExported {count} ONNX models (max numerical diff: {max_diff:.2e})")
print(f"Output: {P.LIVE_DIR / 'MQL5_Files'}")
