//+------------------------------------------------------------------+
//| regime_v4_sizer.mqh — Vol-only regime classifier (3 classes)    |
//|                                                                  |
//| Predicts next-288-bar vol regime from past-288-bar fingerprint. |
//| Output: Quiet (0), Normal (1), HighVol (2)                      |
//| Lot multiplier: 0.5x / 1.0x / 1.3x — based on experiment showing|
//| HighVol trades have PF 1.98 vs Normal 1.62 vs Quiet 1.28        |
//|                                                                  |
//| Features (same as Python compute, must match exactly):          |
//|   fp0 = sum of returns                                          |
//|   fp1 = std of returns                                          |
//|   fp2 = trend consistency (% same-sign as mean)                 |
//|   fp3 = sum/std                                                 |
//|   fp4 = mean bar range (h-l)/close                              |
//|   fp5 = total range / (mean bar range * mean close)             |
//|   fp6 = lag-1 autocorrelation of returns                        |
//+------------------------------------------------------------------+
#ifndef __REGIME_V4_SIZER_MQH__
#define __REGIME_V4_SIZER_MQH__

#define REGIME_V4_WINDOW 288
#define REGIME_V4_FEAT_DIM 7

// Lot multipliers per predicted vol regime (indexed 0..2)
static const double REGIME_V4_LOT_MULT[3] = { 0.5, 1.0, 1.3 };
static const string REGIME_V4_NAMES[3]    = { "Quiet", "Normal", "HighVol" };

long g_regime_v4_handle = INVALID_HANDLE;

//--- Load ONNX model (call in OnInit)
bool RegimeV4_Load(const string onnx_filename = "regime_v4_xau.onnx")
{
   g_regime_v4_handle = OnnxCreate(onnx_filename, ONNX_DEFAULT);
   if(g_regime_v4_handle == INVALID_HANDLE)
   {
      PrintFormat("RegimeV4: failed to load %s (err=%d)", onnx_filename, GetLastError());
      return false;
   }
   ulong in_shape[]  = {1, REGIME_V4_FEAT_DIM};
   ulong lbl_shape[] = {1};
   ulong prob_shape[]= {1, 3};
   if(!OnnxSetInputShape(g_regime_v4_handle, 0, in_shape) ||
      !OnnxSetOutputShape(g_regime_v4_handle, 0, lbl_shape) ||
      !OnnxSetOutputShape(g_regime_v4_handle, 1, prob_shape))
   {
      Print("RegimeV4: shape config failed");
      OnnxRelease(g_regime_v4_handle);
      g_regime_v4_handle = INVALID_HANDLE;
      return false;
   }
   Print("RegimeV4: vol classifier loaded");
   return true;
}

void RegimeV4_Release()
{
   if(g_regime_v4_handle != INVALID_HANDLE)
   {
      OnnxRelease(g_regime_v4_handle);
      g_regime_v4_handle = INVALID_HANDLE;
   }
}

//--- Compute fingerprint from last REGIME_V4_WINDOW M5 bars
bool RegimeV4_Fingerprint(float &fp[])
{
   ArrayResize(fp, REGIME_V4_FEAT_DIM);
   MqlRates rb[];
   ArraySetAsSeries(rb, true);
   int n = CopyRates(_Symbol, PERIOD_M5, 1, REGIME_V4_WINDOW, rb);
   if(n < REGIME_V4_WINDOW) return false;

   // Returns: (close[i] - close[i+1]) / close[i+1] — i ranges old-to-new
   // In series order, bar 0 = newest. Walk oldest→newest for stable computation.
   double sum_r = 0.0, sum_r2 = 0.0;
   int nr = 0;
   double prev = rb[REGIME_V4_WINDOW - 1].close; // oldest
   double sum_br = 0.0;
   double max_h = rb[REGIME_V4_WINDOW - 1].high, min_l = rb[REGIME_V4_WINDOW - 1].low;
   double sum_c = 0.0;

   // First bar: only contributes to range + close sum
   sum_br += (rb[REGIME_V4_WINDOW - 1].high - rb[REGIME_V4_WINDOW - 1].low) / prev;
   sum_c += prev;

   double ret_arr[];
   ArrayResize(ret_arr, REGIME_V4_WINDOW - 1);

   for(int k = REGIME_V4_WINDOW - 2; k >= 0; k--)
   {
      double cur = rb[k].close;
      double r = (cur - prev) / prev;
      ret_arr[nr] = r;
      sum_r += r;
      sum_r2 += r * r;
      nr++;
      double br = (rb[k].high - rb[k].low) / cur;
      sum_br += br;
      if(rb[k].high > max_h) max_h = rb[k].high;
      if(rb[k].low < min_l) min_l = rb[k].low;
      sum_c += cur;
      prev = cur;
   }

   double mean_r = sum_r / (double)nr;
   double var_r = sum_r2 / (double)nr - mean_r * mean_r;
   if(var_r < 0) var_r = 0;
   double std_r = MathSqrt(var_r);
   double mean_br = sum_br / (double)REGIME_V4_WINDOW;
   double mean_c = sum_c / (double)REGIME_V4_WINDOW;
   double total_range = (max_h - min_l) / mean_c;

   // Trend consistency: % of returns with same sign as mean
   double trend_cons = 0.5;
   if(MathAbs(mean_r) > 1e-12)
   {
      int same = 0;
      double sign_m = (mean_r > 0) ? 1.0 : -1.0;
      for(int i = 0; i < nr; i++)
      {
         double s = (ret_arr[i] > 0) ? 1.0 : ((ret_arr[i] < 0) ? -1.0 : 0.0);
         if(s == sign_m) same++;
      }
      trend_cons = (double)same / (double)nr;
   }

   // Lag-1 autocorrelation
   double ac = 0.0;
   if(nr > 2)
   {
      double s1 = 0, s2 = 0, s12 = 0, s11 = 0, s22 = 0;
      int n1 = nr - 1;
      for(int i = 0; i < n1; i++)
      {
         double a = ret_arr[i], b = ret_arr[i + 1];
         s1 += a; s2 += b;
         s11 += a * a; s22 += b * b; s12 += a * b;
      }
      double m1 = s1 / n1, m2 = s2 / n1;
      double v1 = s11 / n1 - m1 * m1;
      double v2 = s22 / n1 - m2 * m2;
      if(v1 > 0 && v2 > 0)
      {
         double cov = s12 / n1 - m1 * m2;
         double denom = MathSqrt(v1 * v2);
         if(denom > 1e-12) ac = cov / denom;
      }
   }

   fp[0] = (float)sum_r;
   fp[1] = (float)std_r;
   fp[2] = (float)trend_cons;
   fp[3] = (float)(sum_r / (std_r + 1e-9));
   fp[4] = (float)mean_br;
   fp[5] = (float)(total_range / (mean_br + 1e-9));
   fp[6] = (float)ac;
   return true;
}

//--- Predict regime: 0=Quiet, 1=Normal, 2=HighVol. -1 on failure.
int RegimeV4_Predict(double &out_max_prob)
{
   out_max_prob = 0.0;
   if(g_regime_v4_handle == INVALID_HANDLE) return -1;
   float fp[];
   if(!RegimeV4_Fingerprint(fp)) return -1;
   long out_label[1]; float probs[3];
   if(!OnnxRun(g_regime_v4_handle, ONNX_DEFAULT, fp, out_label, probs))
   { Print("RegimeV4: OnnxRun failed"); return -1; }
   int cls = (int)out_label[0];
   if(cls < 0 || cls > 2) cls = 1; // safety
   double mp = probs[0];
   if(probs[1] > mp) mp = probs[1];
   if(probs[2] > mp) mp = probs[2];
   out_max_prob = mp;
   return cls;
}

//--- Convenience: return the lot-size multiplier given current regime
double RegimeV4_LotMultiplier()
{
   double p;
   int cls = RegimeV4_Predict(p);
   if(cls < 0) return 1.0; // fallback to baseline if model unavailable
   return REGIME_V4_LOT_MULT[cls];
}

#endif  // __REGIME_V4_SIZER_MQH__
