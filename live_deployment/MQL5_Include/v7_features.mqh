//+------------------------------------------------------------------+
//| v7_features.mqh — 4 new features for v7.2-lite                 |
//|                                                                  |
//| PARITY CONVENTION (must match Python compute exactly):           |
//|   • rb[] is ArraySetAsSeries(true): rb[0]=newest, rb[idx+k]=old  |
//|   • At bar idx, features use ONLY bars rb[idx+1..idx+W]          |
//|     (strictly past-only, does NOT include rb[idx])               |
//|   • Returns, stds, sums all match numpy ddof=0 / step=1 behavior |
//|   • 6-bar trailing rolling mean applied as final smoothing step  |
//|     (same as Python _smooth() in 00_compute_v7_features_step1) |
//|                                                                  |
//| The 4 features (in ONNX feature-order positions 14..17):         |
//|   feat[14] = vpin          (smoothed, 50-bucket BVC)             |
//|   feat[15] = sig_quad_var  (smoothed, Σ(ΔY)² over 60 bars)       |
//|   feat[16] = har_rv_ratio  (smoothed, RV_short/RV_long)          |
//|   feat[17] = hawkes_eta    (smoothed, event rate short/long)     |
//+------------------------------------------------------------------+
#ifndef __V7_FEATURES_MQH__
#define __V7_FEATURES_MQH__

//--- state for 6-bar trailing smoothing (one ring buffer per feature)
double g_smooth_vpin[6];       int g_smooth_vpin_n    = 0;
double g_smooth_qv[6];         int g_smooth_qv_n      = 0;
double g_smooth_har[6];        int g_smooth_har_n     = 0;
double g_smooth_hawkes[6];     int g_smooth_hawkes_n  = 0;

// Last "smoothing key" (bar time) we pushed — prevents double-push per bar.
datetime g_smooth_last_time = 0;

void V7_ResetSmoothingState()
{
   for(int i=0;i<6;i++){ g_smooth_vpin[i]=0; g_smooth_qv[i]=0; g_smooth_har[i]=0; g_smooth_hawkes[i]=0; }
   g_smooth_vpin_n=0; g_smooth_qv_n=0; g_smooth_har_n=0; g_smooth_hawkes_n=0;
   g_smooth_last_time=0;
}

double _smooth_push_and_mean(double raw, double &buf[], int &n)
{
   // Shift right, insert at [0] — keep newest at [0], oldest at [min(n,5)]
   for(int i=5;i>=1;i--) buf[i]=buf[i-1];
   buf[0]=raw;
   if(n<6) n++;
   double s=0.0; for(int i=0;i<n;i++) s+=buf[i];
   return s/n;
}

//+------------------------------------------------------------------+
//| erf approximation — Abramowitz & Stegun 7.1.26                   |
//|   max |error| < 1.5e-7  (matches float32 XGBoost precision)      |
//+------------------------------------------------------------------+
double _erf(double x)
{
   const double a1= 0.254829592;
   const double a2=-0.284496736;
   const double a3= 1.421413741;
   const double a4=-1.453152027;
   const double a5= 1.061405429;
   const double p = 0.3275911;
   double sign = (x < 0.0) ? -1.0 : 1.0;
   double ax = MathAbs(x);
   double t = 1.0 / (1.0 + p*ax);
   double y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t * MathExp(-ax*ax);
   return sign * y;
}

//+------------------------------------------------------------------+
//| 1. VPIN — range-based BVC, 50 buckets, 200-bar μ/σ baseline      |
//|                                                                  |
//| Python (step=1, past-only at bar i using r[:i], V[:i]):          |
//|   r_j  = close[j] - close[j-1]                                    |
//|   V_j  = high[j] - low[j]                                         |
//|   μ_i  = mean(r[i-200:i])                                         |
//|   σ_i  = std (r[i-200:i], ddof=0)                                 |
//|   VPIN_i = Σ_{j=i-50..i-1} V_j·|erf((r_j-μ_i)/(σ_i·√2))| /        |
//|            Σ_{j=i-50..i-1} V_j                                    |
//|                                                                  |
//| MQL5 (at bar idx, use rb[idx+1..idx+200] for baseline, rb[idx+1] |
//| ..rb[idx+50] for BVC sum):                                        |
//+------------------------------------------------------------------+
double ComputeVPIN_Raw(const MqlRates &rb[], int idx, int n_buckets=50, int mu_sigma_window=200)
{
   int need = idx + n_buckets + mu_sigma_window + 1;   // +1 for close[i-200-1] to make r[i-200]
   if(need >= ArraySize(rb)) return 0.5;

   // μ/σ over the 200 returns ending at bar i-1 (series indices idx+1..idx+200).
   // Return r[j] = close[j] - close[j-1]; in series terms r at series index s uses rb[s], rb[s+1].
   double sum_r = 0.0, sum_r2 = 0.0;
   for(int k = 1; k <= mu_sigma_window; k++)    // k=1..200: series indices idx+k
   {
      double r = rb[idx+k].close - rb[idx+k+1].close;
      sum_r  += r;
      sum_r2 += r*r;
   }
   double mu = sum_r / mu_sigma_window;
   double var = sum_r2 / mu_sigma_window - mu*mu;
   if(var < 0.0) var = 0.0;
   double sig = MathSqrt(var);                  // ddof=0 matches Python .std()
   if(sig < 1e-12) return 0.5;

   const double sqrt2 = 1.41421356237309504880;
   double num = 0.0, den = 0.0;
   for(int k = 1; k <= n_buckets; k++)          // k=1..50: series indices idx+k
   {
      double r_j = rb[idx+k].close - rb[idx+k+1].close;
      double V_j = rb[idx+k].high  - rb[idx+k].low;
      double z = (r_j - mu) / (sig * sqrt2);
      double erf_z = _erf(z);
      num += V_j * MathAbs(erf_z);
      den += V_j;
   }
   if(den < 1e-12) return 0.5;
   return num / den;
}

//+------------------------------------------------------------------+
//| 2. sig_quad_var — Σ(ΔY)² where Y_k = 100·(log c - log c_first)   |
//| Python: Y = 100·(logc[i-60:i] - logc[i-60]); dY = diff(Y);        |
//|         qv = Σ dY²                                                 |
//+------------------------------------------------------------------+
double ComputeSigQuadVar_Raw(const MqlRates &rb[], int idx, int window=60)
{
   if(idx + window + 1 >= ArraySize(rb)) return 0.0;
   // Build Y in chronological order from rb[idx+window] (oldest) → rb[idx+1] (newest).
   // Y_0 = 0; Y_k = 100 · (log(close_k) - log(close_0))
   // ΔY_k = Y_{k+1} - Y_k = 100 · log(close_{k+1} / close_k)
   // So we can compute dY directly without materializing Y.
   double qv = 0.0;
   for(int k = 0; k < window - 1; k++)
   {
      // chronological step: older=window-k, newer=window-k-1
      int s_old = idx + window - k;
      int s_new = idx + window - k - 1;
      double c_old = rb[s_old].close;
      double c_new = rb[s_new].close;
      if(c_old < 1e-10) continue;
      double dY = 100.0 * MathLog(c_new / c_old);
      qv += dY * dY;
   }
   return qv;
}

//+------------------------------------------------------------------+
//| 3. HAR-RV ratio — (mean r² over last 288 bars) /                 |
//|                   (mean r² over last 8640 bars)                  |
//| Python: past-only via rolling().shift(1); we mimic with idx+1..  |
//+------------------------------------------------------------------+
double ComputeHARRatio_Raw(const MqlRates &rb[], int idx, int short_bars=288, int long_bars=8640)
{
   if(idx + long_bars + 1 >= ArraySize(rb)) return 1.0;   // not enough history

   // r[i] = log(close[i] / close[i-1])
   double sum_short = 0.0;
   for(int k = 1; k <= short_bars; k++)
   {
      double c_new = rb[idx+k].close;
      double c_old = rb[idx+k+1].close;
      if(c_old < 1e-10) continue;
      double r = MathLog(c_new / c_old);
      sum_short += r * r;
   }
   double sum_long = 0.0;
   for(int k = 1; k <= long_bars; k++)
   {
      double c_new = rb[idx+k].close;
      double c_old = rb[idx+k+1].close;
      if(c_old < 1e-10) continue;
      double r = MathLog(c_new / c_old);
      sum_long += r * r;
   }
   double rv_s = sum_short / short_bars;
   double rv_l = sum_long  / long_bars;
   if(rv_l < 1e-18) rv_l = 1e-18;
   double ratio = rv_s / rv_l;
   if(ratio < 0.0)   ratio = 0.0;
   if(ratio > 20.0)  ratio = 20.0;
   return ratio;
}

//+------------------------------------------------------------------+
//| 4. Hawkes η — event-rate(last 60) / event-rate(last 600)         |
//| event_j = 1 if |r_j| > 2·σ500  where σ500 = std of last 500 rets |
//|                                                                  |
//| Python: at bar i, sigma[j] uses r[j-500:j] for each j∈[i-600..i].|
//| Past-only: r[j] uses close[j-1], close[j]; sigma[j] uses rets    |
//|           up through r[j-1] inclusive (rolling with shift(1) =   |
//|           "latest included return is r[j-1]").                   |
//| Our MQL5 computes per-bar sigma inline, matching Python's        |
//| rolling(500).std(ddof=0).shift(0) — same alignment.              |
//+------------------------------------------------------------------+
double ComputeHawkesEta_Raw(const MqlRates &rb[], int idx,
                            int short_window=60, int long_window=600,
                            int sigma_window=500, double event_k=2.0)
{
   int need = idx + long_window + sigma_window + 2;
   if(need >= ArraySize(rb)) return 1.0;

   // For each k in [1..600] (Python bar j = i-k), compute event[j].
   // Python sigma[j] = std(r[j-499], r[j-498], ..., r[j])  — 500 returns,
   // INCLUSIVE of r[j] (pandas rolling default).  In MQL5 series indexing,
   // r at series s = log(rb[s].close/rb[s+1].close).  Python r[j-m] maps to
   // series s = idx+k+m.  So sigma[j]'s 500 returns live at series indices
   // idx+k, idx+k+1, ..., idx+k+499 — note idx+k (not idx+k+1) is included.
   int event_short = 0, event_long = 0;
   for(int k = 1; k <= long_window; k++)
   {
      // r_j  (Python r[i-k])
      double c_new = rb[idx+k].close;
      double c_old = rb[idx+k+1].close;
      if(c_old < 1e-10) continue;
      double r_j = MathLog(c_new / c_old);

      // sigma_j uses returns at series idx+k..idx+k+499 (includes r_j itself)
      double s_sum = 0.0, s_sum2 = 0.0;
      int cnt = 0;
      for(int m = 0; m < sigma_window; m++)
      {
         int s = idx + k + m;
         double cn = rb[s].close;
         double co = rb[s+1].close;
         if(co < 1e-10) continue;
         double rr = MathLog(cn / co);
         s_sum  += rr;
         s_sum2 += rr*rr;
         cnt++;
      }
      if(cnt < sigma_window) continue;
      double s_mu  = s_sum / sigma_window;
      double s_var = s_sum2 / sigma_window - s_mu*s_mu;
      if(s_var < 0.0) s_var = 0.0;
      double sigma = MathSqrt(s_var);
      if(sigma <= 0.0) continue;

      if(MathAbs(r_j) > event_k * sigma)
      {
         if(k <= short_window) event_short++;
         event_long++;
      }
   }
   double rate_s = (double)event_short / short_window;
   double rate_l = (double)event_long  / long_window;
   if(rate_l < 1e-9) return 1.0;
   double eta = rate_s / rate_l;
   if(eta < 0.0)  eta = 0.0;
   if(eta > 20.0) eta = 20.0;
   return eta;
}

//+------------------------------------------------------------------+
//| Build all 4 v7 features WITH 6-bar trailing smoothing.         |
//| Call this once per new-bar event; pass the just-closed bar time  |
//| so the smoothing ring-buffer doesn't double-push within a tick.  |
//+------------------------------------------------------------------+
void ComputeV7Features(const MqlRates &rb[],
                        datetime bar_time,
                        double &out_vpin,
                        double &out_qv,
                        double &out_har,
                        double &out_hawkes)
{
   double raw_vpin   = ComputeVPIN_Raw(rb, 0);
   double raw_qv     = ComputeSigQuadVar_Raw(rb, 0);
   double raw_har    = ComputeHARRatio_Raw(rb, 0);
   double raw_hawkes = ComputeHawkesEta_Raw(rb, 0);

   // Push into smoothing buffer exactly once per new bar
   if(bar_time != g_smooth_last_time)
   {
      out_vpin   = _smooth_push_and_mean(raw_vpin,   g_smooth_vpin,   g_smooth_vpin_n);
      out_qv     = _smooth_push_and_mean(raw_qv,     g_smooth_qv,     g_smooth_qv_n);
      out_har    = _smooth_push_and_mean(raw_har,    g_smooth_har,    g_smooth_har_n);
      out_hawkes = _smooth_push_and_mean(raw_hawkes, g_smooth_hawkes, g_smooth_hawkes_n);
      g_smooth_last_time = bar_time;
   }
   else
   {
      // Same bar as last call — return current smoothed values without re-pushing
      // Recompute the mean from current buffer (no new push).
      double s;
      s=0; for(int i=0;i<g_smooth_vpin_n;i++)   s+=g_smooth_vpin[i];   out_vpin   = (g_smooth_vpin_n>0)   ? s/g_smooth_vpin_n   : raw_vpin;
      s=0; for(int i=0;i<g_smooth_qv_n;i++)     s+=g_smooth_qv[i];     out_qv     = (g_smooth_qv_n>0)     ? s/g_smooth_qv_n     : raw_qv;
      s=0; for(int i=0;i<g_smooth_har_n;i++)    s+=g_smooth_har[i];    out_har    = (g_smooth_har_n>0)    ? s/g_smooth_har_n    : raw_har;
      s=0; for(int i=0;i<g_smooth_hawkes_n;i++) s+=g_smooth_hawkes[i]; out_hawkes = (g_smooth_hawkes_n>0) ? s/g_smooth_hawkes_n : raw_hawkes;
   }
}

#endif // __V7_FEATURES_MQH__
