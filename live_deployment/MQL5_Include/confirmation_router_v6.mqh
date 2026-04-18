//+------------------------------------------------------------------+
//| confirmation_router_v6.mqh — 14 physics features, auto-generated |
//+------------------------------------------------------------------+
#ifndef __CONFIRMATION_ROUTER_V6_MQH__
#define __CONFIRMATION_ROUTER_V6_MQH__
#include "setup_rules.mqh"
#define CONFIRM_V6_FEATURE_DIM 14

const double RULE_THRESHOLD_V6[RULE_COUNT] = {0.5000, 0.6500, 0.6500, 0.6500, 0.3500, 0.6000, 0.6500, 0.3500, 0.6000, 0.6500, 0.5000, 0.5000, 0.6500, 0.5500, 0.6500, 0.6000, 0.5000, 0.5000, 0.6000, 0.6500, 0.4000, 0.3000, 0.6500, 0.6000, 0.6500, 0.6500};

const string RULE_ONNX_V6[RULE_COUNT] = {"confirm_v6_c0_R3a_pullback.onnx", "confirm_v6_c0_R3b_higherlow.onnx", "confirm_v6_c0_R3c_breakpullback.onnx", "confirm_v6_c0_R3d_oversold.onnx", "confirm_v6_c0_R3e_false_breakdown.onnx", "confirm_v6_c0_R3f_sma_bounce.onnx", "confirm_v6_c0_R3g_three_green.onnx", "confirm_v6_c0_R3h_close_streak.onnx", "confirm_v6_c0_R3i_inside_break.onnx", "confirm_v6_c1_R0a_bb.onnx", "confirm_v6_c1_R0b_stoch.onnx", "confirm_v6_c1_R0c_doubletouch.onnx", "confirm_v6_c1_R0f_mean_revert.onnx", "confirm_v6_c1_R0i_close_extreme.onnx", "confirm_v6_c2_R0d_squeeze.onnx", "confirm_v6_c2_R0e_nr4_break.onnx", "confirm_v6_c2_R0g_inside_break.onnx", "confirm_v6_c2_R0h_3bar_reversal.onnx", "confirm_v6_c3_R1a_swinghigh.onnx", "confirm_v6_c3_R1b_lowerhigh.onnx", "confirm_v6_c3_R1c_bouncefade.onnx", "confirm_v6_c3_R1d_overbought.onnx", "confirm_v6_c3_R1e_false_breakout.onnx", "confirm_v6_c3_R1f_sma_reject.onnx", "confirm_v6_c3_R1g_three_red.onnx", "confirm_v6_c3_R1h_close_streak.onnx"};

long g_v6_handles[RULE_COUNT];

bool ConfirmV6_Load()
{
   for(int r=0;r<RULE_COUNT;r++) g_v6_handles[r]=INVALID_HANDLE;
   ulong in_shape[]  = {1, CONFIRM_V6_FEATURE_DIM};
   ulong lbl_shape[] = {1};
   ulong prob_shape[]= {1, 2};
   for(int r = 0; r < RULE_COUNT; r++)
   {
      g_v6_handles[r] = OnnxCreate(RULE_ONNX_V6[r], ONNX_DEFAULT);
      if(g_v6_handles[r] == INVALID_HANDLE)
      { Print("V6 Router: failed to load ", RULE_ONNX_V6[r]); return false; }
      if(!OnnxSetInputShape(g_v6_handles[r], 0, in_shape) ||
         !OnnxSetOutputShape(g_v6_handles[r], 0, lbl_shape) ||
         !OnnxSetOutputShape(g_v6_handles[r], 1, prob_shape))
      { Print("V6 Router: shape fail ", RULE_ONNX_V6[r]); return false; }
   }
   return true;
}

void ConfirmV6_Release()
{
   for(int r=0;r<RULE_COUNT;r++)
   {
      if(g_v6_handles[r] != INVALID_HANDLE && g_v6_handles[r] > 0)
      {
         OnnxRelease(g_v6_handles[r]);
         g_v6_handles[r] = INVALID_HANDLE;
      }
   }
}

bool ConfirmV6_Rule(int rule_id, const float &feat[], double &out_prob)
{
   out_prob = 0.0;
   long h = g_v6_handles[rule_id];
   if(h == INVALID_HANDLE) return false;
   float in_data[];
   ArrayResize(in_data, CONFIRM_V6_FEATURE_DIM);
   ArrayCopy(in_data, feat, 0, 0, CONFIRM_V6_FEATURE_DIM);
   long out_label[1]; float probs[2];
   if(!OnnxRun(h, ONNX_DEFAULT, in_data, out_label, probs))
   { Print("V6: OnnxRun failed ", RULE_NAMES[rule_id]); return false; }
   out_prob = (double)probs[1];
   return out_prob >= RULE_THRESHOLD_V6[rule_id];
}

#endif