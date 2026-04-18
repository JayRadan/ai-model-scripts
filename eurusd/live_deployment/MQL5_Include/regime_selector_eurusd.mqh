//+------------------------------------------------------------------+
//| regime_selector_eurusd.mqh                                       |
//| Auto-generated from regime_selector_K4.json. DO NOT EDIT BY HAND.|
//+------------------------------------------------------------------+
#ifndef __REGIME_SELECTOR_EURUSD_MQH__
#define __REGIME_SELECTOR_EURUSD_MQH__

#define REGIME_K              5
#define REGIME_N_FEATS        7

// Fingerprint feature order (compute in this exact order)
// [0] weekly_return_pct
// [1] volatility_pct
// [2] trend_consistency
// [3] trend_strength
// [4] volatility
// [5] range_vs_atr
// [6] return_autocorr

// Cluster meta — 0:Uptrend 1:MeanRevert 2:TrendRange 3:Downtrend 4:HighVol
// C0 Uptrend: threshold=0.4, classes=[0, 1, 2]
// C1 MeanRevert: threshold=0.4, classes=[0, 1, 2]
// C2 TrendRange: threshold=0.4, classes=[0, 1, 2]
// C3 Downtrend: threshold=0.4, classes=[0, 1, 2]
// C4 HighVol: threshold=0.4, classes=[0, 1, 2]

const double REGIME_SCALER_MEAN[REGIME_N_FEATS] = {+1.78670793164e-05, +0.000262262531224096, +0.497600894608556, -0.0237725357856194, +0.000342506296700551, +19.7459854047706, -0.0356363612231206};
const double REGIME_SCALER_STD [REGIME_N_FEATS] = {+0.00433624317904175, +9.23217689500178e-05, +0.0234684435707294, +15.5371146709403, +0.000110949925294987, +5.79571818072906, +0.0859238514867224};

const double REGIME_PCA_MEAN[REGIME_N_FEATS] = {-2.02165042420362e-17, +6.25363864553654e-16, -2.21033779712929e-16, -2.4933688565178e-17, +4.42067559425859e-16, +1.75209703430981e-16, -1.07821355957527e-17};

const double REGIME_PCA_COMP[REGIME_N_FEATS][REGIME_N_FEATS] = {
    {+0.273761261552906, +0.593700043852863, +0.287173232785714, +0.270716133614486, +0.581078868209557, +0.249063595649497, +0.130896956483866},
    {+0.648326254506805, -0.27136476364367, -0.0340056104718337, +0.65060623280945, -0.270474429735018, -0.0865467368448918, -0.030702219084999},
    {-0.0495526527357584, -0.232959701084821, +0.35270138238845, -0.0405408274949452, -0.295232615030038, +0.636221050434377, +0.570344603245551},
    {-0.0396975740539116, -0.114058450690164, +0.707166363750936, -0.0286103669267286, -0.10062224575805, +0.122236857319838, -0.677823920863794},
    {+0.0361966450532492, +0.10439070508592, -0.537247822927001, +0.0227079478289411, -0.0710255286962938, +0.704926535478912, -0.443481074337179},
    {-0.66907340593496, -0.217936008835529, -0.0292529434307705, +0.672348095667757, +0.224262741201323, +0.0394754088122112, -0.00921369003097667},
    {-0.227435830441695, +0.669483642236954, +0.0490067722792394, +0.219987682377905, -0.660730955566052, -0.110806971970172, +0.0206099733328364}
};

const double REGIME_CENTROIDS[REGIME_K][REGIME_N_FEATS] = {
    {+1.1172509523629, +1.63270156577915, +0.789930842276396, +0.390519923878365, +0.278375093545741, +0.0782451102657083, -0.0128162815108504},
    {-1.01729099150519, +0.297926721883317, -0.692554092893817, +0.342555128139514, -0.0469293206589753, -0.0218594539479372, +0.0231777775625191},
    {-0.572551841782792, +0.169919175501347, +0.266980387663869, -0.966363714567652, -0.258744843675572, -0.00642334101228834, +0.0201763141683391},
    {-0.129762795543451, -2.04158755365898, +0.871049500528289, +0.452092674544959, +0.254402792139292, +0.00628800204306605, -0.0436039489354518},
    {+2.19158836051211, -0.615635520522364, -1.06264249187983, -0.28092617623375, -0.156691531838454, -0.0539816810947649, -0.0174011442024581}
};

const string REGIME_NAMES[REGIME_K] = {"Uptrend", "MeanRevert", "TrendRange", "Downtrend", "HighVol"};

const int REGIME_TRADEABLE[REGIME_K] = {1, 1, 1, 1, 1};

const double REGIME_THRESHOLD[REGIME_K] = {0.4000, 0.4000, 0.4000, 0.4000, 0.4000};

void ComputeWeekFingerprint(const MqlRates &bars[], double &fp[])
{
   int n = ArraySize(bars);
   if(n < 2) { ArrayInitialize(fp, 0.0); return; }

   double sum_ret = 0.0, sum_ret2 = 0.0;
   double sum_range_pct = 0.0;
   double max_h = bars[n-1].high;
   double min_l = bars[n-1].low;
   double mean_close = 0.0;
   int n_ret = n - 1;
   int pos_ret = 0;

   double prev_close = bars[n-1].close;
   for(int i = n - 2; i >= 0; --i)
   {
      double c = bars[i].close;
      double r = (c - prev_close) / prev_close;
      sum_ret  += r;
      sum_ret2 += r * r;
      sum_range_pct += (bars[i].high - bars[i].low) / c;
      if(bars[i].high > max_h) max_h = bars[i].high;
      if(bars[i].low  < min_l) min_l = bars[i].low;
      mean_close += c;
      if(r > 0) pos_ret++;
      prev_close = c;
   }
   mean_close = (mean_close + bars[n-1].close) / n;

   double mean_ret = sum_ret / n_ret;
   double var_ret  = sum_ret2 / n_ret - mean_ret * mean_ret;
   if(var_ret < 0) var_ret = 0;
   double std_ret  = MathSqrt(var_ret);
   double neg_ret  = n_ret - pos_ret;
   double mean_bar_range = sum_range_pct / n_ret;

   double trend_consistency = 0.5;
   if(MathAbs(mean_ret) > 1e-12)
      trend_consistency = (mean_ret > 0) ? (double)pos_ret / n_ret : (double)neg_ret / n_ret;

   double r_autocorr = 0.0;
   if(n_ret > 2)
   {
      double s0 = 0, s1 = 0, s00 = 0, s11 = 0, s01 = 0;
      double pc = bars[n-1].close;
      double prev_r = 0;
      bool have_prev = false;
      int cnt = 0;
      for(int i = n - 2; i >= 0; --i)
      {
         double r = (bars[i].close - pc) / pc;
         if(have_prev)
         {
            s0  += prev_r; s1 += r;
            s00 += prev_r * prev_r; s11 += r * r;
            s01 += prev_r * r;
            cnt++;
         }
         prev_r = r;
         have_prev = true;
         pc = bars[i].close;
      }
      if(cnt > 0)
      {
         double m0 = s0 / cnt, m1 = s1 / cnt;
         double v0 = s00 / cnt - m0 * m0;
         double v1 = s11 / cnt - m1 * m1;
         if(v0 > 0 && v1 > 0)
            r_autocorr = (s01 / cnt - m0 * m1) / MathSqrt(v0 * v1);
      }
   }

   fp[0] = sum_ret;
   fp[1] = std_ret;
   fp[2] = trend_consistency;
   fp[3] = sum_ret / (std_ret + 1e-9);
   fp[4] = mean_bar_range;
   fp[5] = (max_h - min_l) / mean_close / (mean_bar_range + 1e-9);
   fp[6] = r_autocorr;
}

int ClassifyRegime(const double &fp[])
{
   double scaled[REGIME_N_FEATS];
   for(int i = 0; i < REGIME_N_FEATS; ++i)
      scaled[i] = (fp[i] - REGIME_SCALER_MEAN[i]) / REGIME_SCALER_STD[i];

   double rotated[REGIME_N_FEATS];
   for(int i = 0; i < REGIME_N_FEATS; ++i)
   {
      double s = 0.0;
      for(int j = 0; j < REGIME_N_FEATS; ++j)
         s += (scaled[j] - REGIME_PCA_MEAN[j]) * REGIME_PCA_COMP[i][j];
      rotated[i] = s;
   }

   int    best_k = 0;
   double best_d = DBL_MAX;
   for(int k = 0; k < REGIME_K; ++k)
   {
      double d = 0.0;
      for(int i = 0; i < REGIME_N_FEATS; ++i)
      {
         double v = rotated[i] - REGIME_CENTROIDS[k][i];
         d += v * v;
      }
      if(d < best_d) { best_d = d; best_k = k; }
   }
   return best_k;
}

#endif  // __REGIME_SELECTOR_EURUSD_MQH__