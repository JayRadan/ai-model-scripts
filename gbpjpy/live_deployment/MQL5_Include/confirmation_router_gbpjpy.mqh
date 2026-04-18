//+------------------------------------------------------------------+
//| confirmation_router_gbpjpy.mqh                                   |
//| Auto-generated. Loads 48 per-rule ONNX classifiers.              |
//+------------------------------------------------------------------+
#ifndef __CONFIRMATION_ROUTER_GBPJPY_MQH__
#define __CONFIRMATION_ROUTER_GBPJPY_MQH__

#include "setup_rules_gbpjpy.mqh"

#define CONFIRM_FEATURE_DIM 34

const double RULE_THRESHOLD[RULE_COUNT] = {0.5900, 0.6000, 0.4100, 0.3000, 0.6200, 0.6400, 0.5700, 0.5100, 0.6000, 0.4100, 0.4400, 0.5700, 0.6000, 0.5100, 0.4000, 0.6100, 0.5200, 0.5400, 0.5700, 0.5600, 0.4600, 0.9900, 0.3700, 0.4800, 0.5000, 0.5300, 0.5500, 0.5300, 0.5800, 0.9900, 0.6300, 0.5000, 0.5900, 0.4800, 0.5700, 0.5000, 0.4500, 0.3000, 0.5200, 0.4900, 0.3800, 0.3200, 0.4400, 0.4200, 0.3900, 0.4900, 0.4100, 0.4700};

const string RULE_ONNX_FILE[RULE_COUNT] = {"gj_confirm_c0_R0a_pullback.onnx", "gj_confirm_c0_R0b_higher_low.onnx", "gj_confirm_c0_R0c_breakout_pb.onnx", "gj_confirm_c0_R0d_oversold.onnx", "gj_confirm_c0_R0e_sma_bounce.onnx", "gj_confirm_c0_R0f_false_breakdown.onnx", "gj_confirm_c0_R0g_close_streak.onnx", "gj_confirm_c0_R0h_pin_bar.onnx", "gj_confirm_c0_R0i_engulfing.onnx", "gj_confirm_c0_R0j_ema_pullback.onnx", "gj_confirm_c0_R0k_tokyo_buy.onnx", "gj_confirm_c0_R0l_london_open_buy.onnx", "gj_confirm_c1_R1a_bb.onnx", "gj_confirm_c1_R1b_stoch.onnx", "gj_confirm_c1_R1e_double_touch.onnx", "gj_confirm_c1_R1f_mean_revert.onnx", "gj_confirm_c1_R1g_close_extreme.onnx", "gj_confirm_c1_R1i_pin_bar.onnx", "gj_confirm_c1_R1j_engulfing.onnx", "gj_confirm_c1_R1n_range_fade.onnx", "gj_confirm_c2_R2a_inside_break.onnx", "gj_confirm_c2_R2b_squeeze.onnx", "gj_confirm_c2_R2c_nr4_break.onnx", "gj_confirm_c2_R2d_3bar_reversal.onnx", "gj_confirm_c2_R2e_session_break.onnx", "gj_confirm_c2_R2f_rsi_divergence.onnx", "gj_confirm_c3_R3a_swing_high.onnx", "gj_confirm_c3_R3b_lower_high.onnx", "gj_confirm_c3_R3c_bounce_fade.onnx", "gj_confirm_c3_R3d_overbought.onnx", "gj_confirm_c3_R3e_sma_reject.onnx", "gj_confirm_c3_R3f_three_red.onnx", "gj_confirm_c3_R3g_false_breakout.onnx", "gj_confirm_c3_R3h_pin_bar.onnx", "gj_confirm_c3_R3i_engulfing.onnx", "gj_confirm_c3_R3j_ema_reject.onnx", "gj_confirm_c3_R3k_london_sell.onnx", "gj_confirm_c3_R3l_tokyo_fade.onnx", "gj_confirm_c4_R4a_v_reversal.onnx", "gj_confirm_c4_R4b_momentum.onnx", "gj_confirm_c4_R4c_bb_vol.onnx", "gj_confirm_c4_R4d_stoch_vol.onnx", "gj_confirm_c4_R4e_inside_vol.onnx", "gj_confirm_c4_R4f_squeeze_vol.onnx", "gj_confirm_c4_R4g_pin_vol.onnx", "gj_confirm_c4_R4h_engulf_vol.onnx", "gj_confirm_c4_R4i_spike_fade.onnx", "gj_confirm_c4_R4j_session_vol.onnx"};

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
      {
         Print("ConfirmRouter: failed to load ", RULE_ONNX_FILE[r], " err=", GetLastError());
         return false;
      }
      if(!OnnxSetInputShape (g_rule_handles[r], 0, in_shape) ||
         !OnnxSetOutputShape(g_rule_handles[r], 0, lbl_shape) ||
         !OnnxSetOutputShape(g_rule_handles[r], 1, prob_shape))
      {
         Print("ConfirmRouter: shape setup failed for ", RULE_ONNX_FILE[r], " err=", GetLastError());
         return false;
      }
   }
   Print("ConfirmRouter: all 48 confirmation models loaded");
   return true;
}

void ConfirmRouter_Release()
{
   for(int r = 0; r < RULE_COUNT; r++)
   {
      if(g_rule_handles[r] != INVALID_HANDLE)
      {
         OnnxRelease(g_rule_handles[r]);
         g_rule_handles[r] = INVALID_HANDLE;
      }
   }
}

bool ConfirmRule(int rule_id, const float &feat[], double &out_prob)
{
   out_prob = 0.0;
   if(rule_id < 0 || rule_id >= RULE_COUNT) return false;
   long h = g_rule_handles[rule_id];
   if(h == INVALID_HANDLE) return false;

   float in[CONFIRM_FEATURE_DIM];
   for(int i = 0; i < CONFIRM_FEATURE_DIM; i++) in[i] = feat[i];

   long out_label[1];
   float probs[2];
   if(!OnnxRun(h, ONNX_DEFAULT, in, out_label, probs))
   {
      Print("ConfirmRule: OnnxRun failed for ", RULE_NAMES[rule_id], " err=", GetLastError());
      return false;
   }
   out_prob = (double)probs[1];
   return out_prob >= RULE_THRESHOLD[rule_id];
}

#endif  // __CONFIRMATION_ROUTER_GBPJPY_MQH__
