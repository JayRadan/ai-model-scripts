//+------------------------------------------------------------------+
//| regime_router.mqh                                                 |
//| Multi-model ONNX router for regime-specialist inference.          |
//|                                                                    |
//| Loads 3 ONNX models (C0 Ranging, C1 Downtrend, C3 Uptrend)        |
//| and routes predictions through the active cluster.                 |
//|                                                                    |
//| Depends on regime_selector.mqh for the weekly regime detection.    |
//+------------------------------------------------------------------+
#ifndef __REGIME_ROUTER_MQH__
#define __REGIME_ROUTER_MQH__

#include "regime_selector.mqh"

// Cluster IDs (0-indexed — match deployment_config.json)
#define RR_C_RANGING    0
#define RR_C_DOWNTREND  1
#define RR_C_SHOCK      2   // never trade
#define RR_C_UPTREND    3

// Signal enum
enum RR_SIGNAL { RR_FLAT = 0, RR_BUY = 1, RR_SELL = -1 };

// ── State ────────────────────────────────────────────────────────────────────
long     g_rr_model[REGIME_K];   // one ONNX handle per cluster (INVALID_HANDLE = skip)
int      g_rr_active_cluster = RR_C_SHOCK;
datetime g_rr_last_week_check = 0;

// ── Load all cluster models ──────────────────────────────────────────────────
// Returns true if at least C0+C1+C3 loaded successfully.
bool RR_LoadModels(string path_c0, string path_c1, string path_c3)
{
   for(int i = 0; i < REGIME_K; ++i) g_rr_model[i] = INVALID_HANDLE;

   // Per-terminal MQL5/Files/ (same as v3 EA). Use ONNX_COMMON_FOLDER if you
   // prefer the shared Common/Files folder instead.
   g_rr_model[RR_C_RANGING]   = OnnxCreate(path_c0, ONNX_DEFAULT);
   g_rr_model[RR_C_DOWNTREND] = OnnxCreate(path_c1, ONNX_DEFAULT);
   g_rr_model[RR_C_UPTREND]   = OnnxCreate(path_c3, ONNX_DEFAULT);

   if(g_rr_model[RR_C_RANGING]   == INVALID_HANDLE) { Print("RR: failed to load ", path_c0); return false; }
   if(g_rr_model[RR_C_DOWNTREND] == INVALID_HANDLE) { Print("RR: failed to load ", path_c1); return false; }
   if(g_rr_model[RR_C_UPTREND]   == INVALID_HANDLE) { Print("RR: failed to load ", path_c3); return false; }

   // Fixed input shape: [1, 36]
   ulong input_shape[] = {1, 36};
   ulong lbl_shape[]   = {1};
   ulong prob3_shape[] = {1, 3};
   ulong prob2_shape[] = {1, 2};

   if(!OnnxSetInputShape (g_rr_model[RR_C_RANGING],   0, input_shape) ||
      !OnnxSetOutputShape(g_rr_model[RR_C_RANGING],   0, lbl_shape)   ||
      !OnnxSetOutputShape(g_rr_model[RR_C_RANGING],   1, prob3_shape) ||
      !OnnxSetInputShape (g_rr_model[RR_C_DOWNTREND], 0, input_shape) ||
      !OnnxSetOutputShape(g_rr_model[RR_C_DOWNTREND], 0, lbl_shape)   ||
      !OnnxSetOutputShape(g_rr_model[RR_C_DOWNTREND], 1, prob2_shape) ||
      !OnnxSetInputShape (g_rr_model[RR_C_UPTREND],   0, input_shape) ||
      !OnnxSetOutputShape(g_rr_model[RR_C_UPTREND],   0, lbl_shape)   ||
      !OnnxSetOutputShape(g_rr_model[RR_C_UPTREND],   1, prob2_shape))
   {
      Print("RR: OnnxSetShape failed, err=", GetLastError());
      return false;
   }

   Print("RR: all 3 cluster models loaded");
   return true;
}

void RR_ReleaseModels()
{
   for(int i = 0; i < REGIME_K; ++i)
   {
      if(g_rr_model[i] != INVALID_HANDLE)
      {
         OnnxRelease(g_rr_model[i]);
         g_rr_model[i] = INVALID_HANDLE;
      }
   }
}

// ── Weekly regime refresh ────────────────────────────────────────────────────
// Call once per new week (or once per day — cheap enough). Returns active cluster.
int RR_RefreshRegime(ENUM_TIMEFRAMES tf, string symbol, int lookback_bars = 2000)
{
   MqlRates rb[];
   ArraySetAsSeries(rb, true);
   int copied = CopyRates(symbol, tf, 0, lookback_bars, rb);
   if(copied < 200) { Print("RR: insufficient bars for regime detection"); return g_rr_active_cluster; }

   // Find the start of the last completed week: scan back until day-of-week resets
   // (simpler: just use last 288*5 = 1440 M5 bars = 5 trading days)
   int week_bars = MathMin(copied - 1, 1440);
   MqlRates week[];
   ArrayResize(week, week_bars);
   for(int i = 0; i < week_bars; ++i) week[i] = rb[i];

   double fp[REGIME_N_FEATS];
   ComputeWeekFingerprint(week, fp);
   g_rr_active_cluster = ClassifyRegime(fp);
   Print("RR: active regime = C", g_rr_active_cluster, " ", REGIME_NAMES[g_rr_active_cluster],
         " (tradeable=", REGIME_TRADEABLE[g_rr_active_cluster], ")");
   return g_rr_active_cluster;
}

// ── Per-bar inference ────────────────────────────────────────────────────────
// Given a 36-feature vector, return the routing decision.
// out_prob fills with the winning class probability (0 if FLAT/skip).
RR_SIGNAL RR_Predict(const float &features[], double &out_prob)
{
   out_prob = 0.0;
   int cid = g_rr_active_cluster;
   if(cid < 0 || cid >= REGIME_K) return RR_FLAT;
   if(REGIME_TRADEABLE[cid] == 0) return RR_FLAT;  // C2 Shock
   long h = g_rr_model[cid];
   if(h == INVALID_HANDLE) return RR_FLAT;

   long   out_label[1];
   double thr = REGIME_THRESHOLD[cid];

   // Input buffer
   float in[36];
   for(int i = 0; i < 36; ++i) in[i] = features[i];

   if(cid == RR_C_RANGING)
   {
      // 3-class model: classes = [BUY=0, FLAT=1, SELL=2]
      float probs[3];
      if(!OnnxRun(h, ONNX_DEFAULT, in, out_label, probs))
      {
         Print("RR: OnnxRun C0 failed: ", GetLastError());
         return RR_FLAT;
      }
      double p_buy = probs[0], p_sell = probs[2];
      if(p_buy >= thr && p_buy >= p_sell) { out_prob = p_buy;  return RR_BUY;  }
      if(p_sell >= thr)                   { out_prob = p_sell; return RR_SELL; }
      return RR_FLAT;
   }

   if(cid == RR_C_DOWNTREND)
   {
      // Binary classifier: classes = [FLAT=1, SELL=2]
      // probs[0] = FLAT prob, probs[1] = SELL prob
      float probs[2];
      if(!OnnxRun(h, ONNX_DEFAULT, in, out_label, probs))
      {
         Print("RR: OnnxRun C1 failed: ", GetLastError());
         return RR_FLAT;
      }
      double p_sell = probs[1];
      if(p_sell >= thr) { out_prob = p_sell; return RR_SELL; }
      return RR_FLAT;
   }

   if(cid == RR_C_UPTREND)
   {
      // Binary classifier: classes = [BUY=0, FLAT=1]
      // probs[0] = BUY prob, probs[1] = FLAT prob
      float probs[2];
      if(!OnnxRun(h, ONNX_DEFAULT, in, out_label, probs))
      {
         Print("RR: OnnxRun C3 failed: ", GetLastError());
         return RR_FLAT;
      }
      double p_buy = probs[0];
      if(p_buy >= thr) { out_prob = p_buy; return RR_BUY; }
      return RR_FLAT;
   }

   return RR_FLAT;
}

#endif  // __REGIME_ROUTER_MQH__
