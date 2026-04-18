//+------------------------------------------------------------------+
//| confirmation_router.mqh — K=5, 36 features                      |
//+------------------------------------------------------------------+
#ifndef __CONFIRMATION_ROUTER_MQH__
#define __CONFIRMATION_ROUTER_MQH__
#include "setup_rules.mqh"
#define CONFIRM_FEATURE_DIM 36

const double RULE_THRESHOLD[RULE_COUNT] = {0.5000, 0.5500, 0.6500, 0.6500, 0.6500, 0.6500, 0.4000, 0.4500, 0.6500, 0.4500, 0.4000, 0.6500, 0.6000, 0.5500, 0.6500, 0.6000, 0.6500, 0.5500, 0.4500, 0.6000, 0.6000, 0.5500, 0.5000, 0.6000, 0.6500, 0.6000};

const string RULE_ONNX_FILE[RULE_COUNT] = {"confirm_c0_R3a_pullback.onnx", "confirm_c0_R3b_higherlow.onnx", "confirm_c0_R3c_breakpullback.onnx", "confirm_c0_R3d_oversold.onnx", "confirm_c0_R3e_false_breakdown.onnx", "confirm_c0_R3f_sma_bounce.onnx", "confirm_c0_R3g_three_green.onnx", "confirm_c0_R3h_close_streak.onnx", "confirm_c0_R3i_inside_break.onnx", "confirm_c1_R0a_bb.onnx", "confirm_c1_R0b_stoch.onnx", "confirm_c1_R0c_doubletouch.onnx", "confirm_c1_R0f_mean_revert.onnx", "confirm_c1_R0i_close_extreme.onnx", "confirm_c2_R0d_squeeze.onnx", "confirm_c2_R0e_nr4_break.onnx", "confirm_c2_R0g_inside_break.onnx", "confirm_c2_R0h_3bar_reversal.onnx", "confirm_c3_R1a_swinghigh.onnx", "confirm_c3_R1b_lowerhigh.onnx", "confirm_c3_R1c_bouncefade.onnx", "confirm_c3_R1d_overbought.onnx", "confirm_c3_R1e_false_breakout.onnx", "confirm_c3_R1f_sma_reject.onnx", "confirm_c3_R1g_three_red.onnx", "confirm_c3_R1h_close_streak.onnx"};

long g_rule_handles[RULE_COUNT];
bool ConfirmRouter_Load()
{
   ulong in_shape[]  = {1, CONFIRM_FEATURE_DIM};
   ulong lbl_shape[] = {1};
   ulong prob_shape[]= {1, 2};
   for(int r = 0; r < RULE_COUNT; r++)
   {
      g_rule_handles[r] = OnnxCreate(RULE_ONNX_FILE[r], ONNX_DEFAULT);
      if(g_rule_handles[r] == INVALID_HANDLE)
      { Print("ConfirmRouter: failed to load ", RULE_ONNX_FILE[r], " err=", GetLastError()); return false; }
      if(!OnnxSetInputShape(g_rule_handles[r], 0, in_shape) ||
         !OnnxSetOutputShape(g_rule_handles[r], 0, lbl_shape) ||
         !OnnxSetOutputShape(g_rule_handles[r], 1, prob_shape))
      { Print("ConfirmRouter: shape failed ", RULE_ONNX_FILE[r]); return false; }
   }
   Print("ConfirmRouter: all ", RULE_COUNT, " models loaded");
   return true;
}
void ConfirmRouter_Release()
{
   for(int r = 0; r < RULE_COUNT; r++)
      if(g_rule_handles[r] != INVALID_HANDLE) { OnnxRelease(g_rule_handles[r]); g_rule_handles[r] = INVALID_HANDLE; }
}
bool ConfirmRule(int rule_id, const float &feat[], double &out_prob)
{
   out_prob = 0.0;
   if(rule_id < 0 || rule_id >= RULE_COUNT) return false;
   long h = g_rule_handles[rule_id];
   if(h == INVALID_HANDLE) return false;
   float in[CONFIRM_FEATURE_DIM];
   for(int i = 0; i < CONFIRM_FEATURE_DIM; i++) in[i] = feat[i];
   long out_label[1]; float probs[2];
   if(!OnnxRun(h, ONNX_DEFAULT, in, out_label, probs))
   { Print("ConfirmRule: OnnxRun failed ", RULE_NAMES[rule_id]); return false; }
   out_prob = (double)probs[1];
   return out_prob >= RULE_THRESHOLD[rule_id];
}
#endif