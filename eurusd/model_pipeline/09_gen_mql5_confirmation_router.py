"""
Generate confirmation_router_eurusd.mqh — embeds the per-rule probability
thresholds as constants and exposes a single ConfirmRule() function.
EURUSD version — 48 rules. FEATURE_DIM = 34 (21 base + 13 tech).
"""
import json
import os

import paths as P

CLUSTER_NAMES = {0: "Uptrend", 1: "MeanRevert", 2: "TrendRange", 3: "Downtrend", 4: "HighVol"}

# Auto-generate RULES from model meta files
import glob
RULES = []
for meta_path in sorted(glob.glob(str(P.MODELS_DIR / "confirm_c*_meta.json"))):
    with open(meta_path) as mf:
        m = json.load(mf)
    RULES.append((m["rule"], m["cluster"]))

FEATURE_DIM = 34

thresholds = []
onnx_files = []
for rule_name, cid in RULES:
    meta_path = P.model(f"confirm_c{cid}_{rule_name}_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    thresholds.append(meta["threshold"])
    onnx_files.append(f"eu_confirm_c{cid}_{rule_name}.onnx")

lines = []
lines.append("//+------------------------------------------------------------------+")
lines.append("//| confirmation_router_eurusd.mqh                                   |")
lines.append(f"//| Auto-generated. Loads {len(RULES)} per-rule ONNX classifiers.              |")
lines.append("//+------------------------------------------------------------------+")
lines.append("#ifndef __CONFIRMATION_ROUTER_EURUSD_MQH__")
lines.append("#define __CONFIRMATION_ROUTER_EURUSD_MQH__")
lines.append("")
lines.append('#include "setup_rules_eurusd.mqh"')
lines.append("")
lines.append(f"#define CONFIRM_FEATURE_DIM {FEATURE_DIM}")
lines.append("")
thrs = ", ".join(f"{t:.4f}" for t in thresholds)
lines.append(f"const double RULE_THRESHOLD[RULE_COUNT] = {{{thrs}}};")
lines.append("")
fns = ", ".join(f'"{f}"' for f in onnx_files)
lines.append(f"const string RULE_ONNX_FILE[RULE_COUNT] = {{{fns}}};")
lines.append("")
lines.append("long g_rule_handles[RULE_COUNT];")
lines.append("")
lines.append("bool ConfirmRouter_Load()")
lines.append("{")
lines.append("   ulong in_shape[]  = {1, CONFIRM_FEATURE_DIM};")
lines.append("   ulong lbl_shape[] = {1};")
lines.append("   ulong prob_shape[]= {1, 2};")
lines.append("   for(int r = 0; r < RULE_COUNT; r++)")
lines.append("   {")
lines.append("      g_rule_handles[r] = OnnxCreate(RULE_ONNX_FILE[r], ONNX_DEFAULT);")
lines.append("      if(g_rule_handles[r] == INVALID_HANDLE)")
lines.append("      {")
lines.append('         Print("ConfirmRouter: failed to load ", RULE_ONNX_FILE[r], " err=", GetLastError());')
lines.append("         return false;")
lines.append("      }")
lines.append("      if(!OnnxSetInputShape (g_rule_handles[r], 0, in_shape) ||")
lines.append("         !OnnxSetOutputShape(g_rule_handles[r], 0, lbl_shape) ||")
lines.append("         !OnnxSetOutputShape(g_rule_handles[r], 1, prob_shape))")
lines.append("      {")
lines.append('         Print("ConfirmRouter: shape setup failed for ", RULE_ONNX_FILE[r], " err=", GetLastError());')
lines.append("         return false;")
lines.append("      }")
lines.append("   }")
lines.append(f'   Print("ConfirmRouter: all {len(RULES)} confirmation models loaded");')
lines.append("   return true;")
lines.append("}")
lines.append("")
lines.append("void ConfirmRouter_Release()")
lines.append("{")
lines.append("   for(int r = 0; r < RULE_COUNT; r++)")
lines.append("   {")
lines.append("      if(g_rule_handles[r] != INVALID_HANDLE)")
lines.append("      {")
lines.append("         OnnxRelease(g_rule_handles[r]);")
lines.append("         g_rule_handles[r] = INVALID_HANDLE;")
lines.append("      }")
lines.append("   }")
lines.append("}")
lines.append("")
lines.append("bool ConfirmRule(int rule_id, const float &feat[], double &out_prob)")
lines.append("{")
lines.append("   out_prob = 0.0;")
lines.append("   if(rule_id < 0 || rule_id >= RULE_COUNT) return false;")
lines.append("   long h = g_rule_handles[rule_id];")
lines.append("   if(h == INVALID_HANDLE) return false;")
lines.append("")
lines.append("   float in[CONFIRM_FEATURE_DIM];")
lines.append("   for(int i = 0; i < CONFIRM_FEATURE_DIM; i++) in[i] = feat[i];")
lines.append("")
lines.append("   long out_label[1];")
lines.append("   float probs[2];")
lines.append("   if(!OnnxRun(h, ONNX_DEFAULT, in, out_label, probs))")
lines.append("   {")
lines.append('      Print("ConfirmRule: OnnxRun failed for ", RULE_NAMES[rule_id], " err=", GetLastError());')
lines.append("      return false;")
lines.append("   }")
lines.append("   out_prob = (double)probs[1];")
lines.append("   return out_prob >= RULE_THRESHOLD[rule_id];")
lines.append("}")
lines.append("")
lines.append("#endif  // __CONFIRMATION_ROUTER_EURUSD_MQH__")

text = "\n".join(lines) + "\n"
out = str(P.LIVE_DIR / "MQL5_Include" / "confirmation_router_eurusd.mqh")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    f.write(text)
print(f"Wrote {out}")
print(f"  thresholds: {thresholds}")
