"""
Generate confirmation_router.mqh — embeds the per-rule probability thresholds
as constants and exposes a single ConfirmRule() function that takes a rule_id
+ feature vector and returns true/false.
"""
import json
import os

import paths as P

CLUSTER_NAMES = {0:"Ranging", 1:"Downtrend", 2:"Shock_News", 3:"Uptrend"}
# Order MUST match the RULE_ID enum in setup_rules.mqh
# All 28 rules listed — disabled ones still need ONNX handles allocated so
# indices stay aligned. Their threshold = 0.99 prevents any trade.
RULES = [
    ("R0a_bb",            0),
    ("R0b_stoch",         0),
    ("R0c_doubletouch",   0),
    ("R0d_squeeze",       0),
    ("R0e_nr4_break",     0),
    ("R0f_mean_revert",   0),
    ("R0g_inside_break",  0),
    ("R0h_3bar_reversal", 0),   # disabled
    ("R0i_close_extreme", 0),
    ("R1a_swinghigh",     1),
    ("R1b_lowerhigh",     1),
    ("R1c_bouncefade",    1),
    ("R1d_overbought",    1),
    ("R1e_false_breakout",1),   # disabled
    ("R1f_sma_reject",    1),
    ("R1g_three_red",     1),
    ("R1h_close_streak",  1),   # disabled
    ("R2b_v_reversal",    2),
    ("R2d_continuation",  2),   # disabled
    ("R3a_pullback",      3),
    ("R3b_higherlow",     3),
    ("R3c_breakpullback", 3),
    ("R3d_oversold",      3),
    ("R3e_false_breakdown", 3),
    ("R3f_sma_bounce",    3),
    ("R3g_three_green",   3),   # disabled
    ("R3h_close_streak",  3),
    ("R3i_inside_break",  3),
]

thresholds = []
onnx_files = []
for rule_name, cid in RULES:
    meta_path = P.model(f"confirm_c{cid}_{rule_name}_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    thresholds.append(meta["threshold"])
    onnx_files.append(f"confirm_c{cid}_{rule_name}.onnx")

lines = []
lines.append("//+------------------------------------------------------------------+")
lines.append("//| confirmation_router.mqh                                           |")
lines.append(f"//| Auto-generated. Loads {len(RULES)} per-rule ONNX classifiers and provides   |")
lines.append("//| ConfirmRule(rule_id, feat[]) → returns probability + pass flag.   |")
lines.append("//+------------------------------------------------------------------+")
lines.append("#ifndef __CONFIRMATION_ROUTER_MQH__")
lines.append("#define __CONFIRMATION_ROUTER_MQH__")
lines.append("")
lines.append('#include "setup_rules.mqh"')
lines.append("")
lines.append(f"#define CONFIRM_FEATURE_DIM 36")
lines.append("")
lines.append("// Per-rule probability threshold (auto-generated from meta files)")
thrs = ", ".join(f"{t:.4f}" for t in thresholds)
lines.append(f"const double RULE_THRESHOLD[RULE_COUNT] = {{{thrs}}};")
lines.append("")
lines.append("// Per-rule ONNX filenames")
fns = ", ".join(f'"{f}"' for f in onnx_files)
lines.append(f"const string RULE_ONNX_FILE[RULE_COUNT] = {{{fns}}};")
lines.append("")
lines.append("// ONNX handles, one per rule")
lines.append("long g_rule_handles[RULE_COUNT];")
lines.append("")
lines.append("//+------------------------------------------------------------------+")
lines.append(f"//| Load all {len(RULES)} confirmation models                                   |")
lines.append("//+------------------------------------------------------------------+")
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
lines.append("         Print(\"ConfirmRouter: failed to load \", RULE_ONNX_FILE[r], \" err=\", GetLastError());")
lines.append("         return false;")
lines.append("      }")
lines.append("      if(!OnnxSetInputShape (g_rule_handles[r], 0, in_shape) ||")
lines.append("         !OnnxSetOutputShape(g_rule_handles[r], 0, lbl_shape) ||")
lines.append("         !OnnxSetOutputShape(g_rule_handles[r], 1, prob_shape))")
lines.append("      {")
lines.append("         Print(\"ConfirmRouter: shape setup failed for \", RULE_ONNX_FILE[r], \" err=\", GetLastError());")
lines.append("         return false;")
lines.append("      }")
lines.append("   }")
lines.append(f"   Print(\"ConfirmRouter: all {len(RULES)} confirmation models loaded\");")
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
lines.append("//+------------------------------------------------------------------+")
lines.append("//| Run confirmation for one rule. Returns true if probability       |")
lines.append("//| crosses the rule's threshold. out_prob is filled with P(label=1).|")
lines.append("//+------------------------------------------------------------------+")
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
lines.append("      Print(\"ConfirmRule: OnnxRun failed for \", RULE_NAMES[rule_id], \" err=\", GetLastError());")
lines.append("      return false;")
lines.append("   }")
lines.append("   // Binary classifier: probs[0]=P(label=0)  probs[1]=P(label=1)")
lines.append("   out_prob = (double)probs[1];")
lines.append("   return out_prob >= RULE_THRESHOLD[rule_id];")
lines.append("}")
lines.append("")
lines.append("#endif  // __CONFIRMATION_ROUTER_MQH__")

text = "\n".join(lines) + "\n"
out = str(P.LIVE_DIR / "MQL5_Include" / "confirmation_router.mqh")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    f.write(text)
print(f"Wrote {out}")
print(f"  thresholds: {thresholds}")
