"""
Generate MQL5 include confirmation_router_v7.mqh from v7_deploy.json,
matching the existing setup_rules.mqh rule order (RULE_NAMES) exactly.
"""
import json, os, sys
sys.path.insert(0, "/home/jay/Desktop/new-model-zigzag/model_pipeline")
import paths as P

# Exact ordering from /home/jay/.mt5/drive_c/Program Files/MetaTrader 5/MQL5/Include/setup_rules.mqh
RULE_NAMES_MQL5 = [
    "C0_R3a_pullback", "C0_R3b_higherlow", "C0_R3c_breakpullback", "C0_R3d_oversold",
    "C0_R3e_false_breakdown", "C0_R3f_sma_bounce", "C0_R3g_three_green",
    "C0_R3h_close_streak", "C0_R3i_inside_break",
    "C1_R0a_bb", "C1_R0b_stoch", "C1_R0c_doubletouch", "C1_R0f_mean_revert",
    "C1_R0i_close_extreme",
    "C2_R0d_squeeze", "C2_R0e_nr4_break", "C2_R0g_inside_break", "C2_R0h_3bar_reversal",
    "C3_R1a_swinghigh", "C3_R1b_lowerhigh", "C3_R1c_bouncefade", "C3_R1d_overbought",
    "C3_R1e_false_breakout", "C3_R1f_sma_reject", "C3_R1g_three_red", "C3_R1h_close_streak",
]

def main():
    with open(P.model("v7_deploy.json")) as f:
        bundle = json.load(f)

    # Build maps: MQL5 rule_name "C0_R3a_pullback" → python key "c0_R3a_pullback"
    confirm = bundle["confirm_models"]                # keys like "c0_R3a_pullback"

    # Produce entries in MQL5 RULE_NAMES order
    onnx_filenames = []
    thresholds     = []
    disabled_flags = []
    for mql5_name in RULE_NAMES_MQL5:
        # Normalize: first letter lowercase
        py_key = mql5_name[:1].lower() + mql5_name[1:]         # C0_... → c0_...
        if py_key in confirm:
            onnx_filenames.append(f"confirm_v7_{py_key}.onnx")
            thresholds.append(confirm[py_key]["threshold"])
            disabled_flags.append(0)
        else:
            # Python disabled this rule — mark disabled, use dummy onnx name + threshold 1.0 (never passes)
            onnx_filenames.append(f"confirm_v7_{py_key}.onnx")    # file won't exist; router skips it
            thresholds.append(1.01)                                 # impossible threshold → always rejects
            disabled_flags.append(1)

    n = len(RULE_NAMES_MQL5)
    out_path = "/home/jay/.mt5/drive_c/Program Files/MetaTrader 5/MQL5/Include/confirmation_router_v7.mqh"

    lines = []
    lines.append("//+------------------------------------------------------------------+")
    lines.append("//| confirmation_router_v7.mqh — 18 features (v7.2-lite)          |")
    lines.append("//| Auto-generated from models/v7_deploy.json                      |")
    lines.append("//| Disabled rules have threshold 1.01 (always rejects)              |")
    lines.append("//+------------------------------------------------------------------+")
    lines.append("#ifndef __CONFIRMATION_ROUTER_V7_MQH__")
    lines.append("#define __CONFIRMATION_ROUTER_V7_MQH__")
    lines.append("#include \"setup_rules.mqh\"")
    lines.append("#define CONFIRM_V7_FEATURE_DIM 18")
    lines.append("")
    lines.append(f"const double RULE_THRESHOLD_V7[RULE_COUNT] = {{{', '.join(f'{t:.4f}' for t in thresholds)}}};")
    lines.append("")
    lines.append(f"const int RULE_DISABLED_V7[RULE_COUNT] = {{{', '.join(str(d) for d in disabled_flags)}}};")
    lines.append("")
    lines.append(f"const string RULE_ONNX_V7[RULE_COUNT] = {{{', '.join(f'\"{f}\"' for f in onnx_filenames)}}};")
    lines.append("")
    lines.append("long g_v7_handles[RULE_COUNT];")
    lines.append("")
    lines.append("bool ConfirmV7_Load()")
    lines.append("{")
    lines.append("   for(int r = 0; r < RULE_COUNT; r++) g_v7_handles[r] = INVALID_HANDLE;")
    lines.append("   ulong in_shape[]   = {1, CONFIRM_V7_FEATURE_DIM};")
    lines.append("   ulong lbl_shape[]  = {1};")
    lines.append("   ulong prob_shape[] = {1, 2};")
    lines.append("   for(int r = 0; r < RULE_COUNT; r++)")
    lines.append("   {")
    lines.append("      if(RULE_DISABLED_V7[r]) continue;                      // skip rules Python disabled")
    lines.append("      g_v7_handles[r] = OnnxCreate(RULE_ONNX_V7[r], ONNX_DEFAULT);")
    lines.append("      if(g_v7_handles[r] == INVALID_HANDLE)")
    lines.append("      { Print(\"V7 Router: failed to load \", RULE_ONNX_V7[r]); return false; }")
    lines.append("      if(!OnnxSetInputShape(g_v7_handles[r], 0, in_shape) ||")
    lines.append("         !OnnxSetOutputShape(g_v7_handles[r], 0, lbl_shape) ||")
    lines.append("         !OnnxSetOutputShape(g_v7_handles[r], 1, prob_shape))")
    lines.append("      { Print(\"V7 Router: shape fail \", RULE_ONNX_V7[r]); return false; }")
    lines.append("   }")
    lines.append("   return true;")
    lines.append("}")
    lines.append("")
    lines.append("void ConfirmV7_Release()")
    lines.append("{")
    lines.append("   for(int r = 0; r < RULE_COUNT; r++)")
    lines.append("   {")
    lines.append("      if(g_v7_handles[r] != INVALID_HANDLE && g_v7_handles[r] > 0)")
    lines.append("      { OnnxRelease(g_v7_handles[r]); g_v7_handles[r] = INVALID_HANDLE; }")
    lines.append("   }")
    lines.append("}")
    lines.append("")
    lines.append("bool ConfirmV7_Rule(int rule_id, const float &feat[], double &out_prob)")
    lines.append("{")
    lines.append("   out_prob = 0.0;")
    lines.append("   if(RULE_DISABLED_V7[rule_id]) return false;                // disabled → never passes")
    lines.append("   long h = g_v7_handles[rule_id];")
    lines.append("   if(h == INVALID_HANDLE) return false;")
    lines.append("   float in_data[];")
    lines.append("   ArrayResize(in_data, CONFIRM_V7_FEATURE_DIM);")
    lines.append("   ArrayCopy(in_data, feat, 0, 0, CONFIRM_V7_FEATURE_DIM);")
    lines.append("   long out_label[1]; float probs[2];")
    lines.append("   if(!OnnxRun(h, ONNX_DEFAULT, in_data, out_label, probs))")
    lines.append("   { Print(\"V7: OnnxRun failed \", RULE_NAMES[rule_id]); return false; }")
    lines.append("   out_prob = (double)probs[1];")
    lines.append("   return out_prob >= RULE_THRESHOLD_V7[rule_id];")
    lines.append("}")
    lines.append("")
    lines.append("#endif // __CONFIRMATION_ROUTER_V7_MQH__")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    n_disabled = sum(disabled_flags)
    n_active = n - n_disabled
    print(f"Generated: {out_path}")
    print(f"  Total rules (matching setup_rules.mqh RULE_COUNT): {n}")
    print(f"  Active: {n_active}, Disabled (threshold=1.01): {n_disabled}")
    print(f"  Meta threshold: {bundle['meta_threshold']}")


if __name__ == "__main__":
    main()
