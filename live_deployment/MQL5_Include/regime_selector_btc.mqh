//+------------------------------------------------------------------+
//| regime_selector_btc.mqh                                               |
//| Auto-generated from regime_selector_btc_K5.json. DO NOT EDIT BY HAND. |
//|                                                                   |
//| Usage:                                                            |
//|   double fp[7];                                                   |
//|   ComputeWeekFingerprint(last_week_bars, fp);                     |
//|   int cluster = ClassifyRegime(fp);                               |
//+------------------------------------------------------------------+
#ifndef __REGIME_SELECTOR_MQH__
#define __REGIME_SELECTOR_MQH__

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

// Cluster meta — 0:Ranging 1:Downtrend 2:Shock(SKIP) 3:Uptrend
// C0 Uptrend: threshold=0.4, classes=[0, 1, 2]
// C1 MeanRevert: threshold=0.4, classes=[0, 1, 2]
// C2 TrendRange: threshold=0.4, classes=[0, 1, 2]
// C3 Downtrend: threshold=0.4, classes=[0, 1, 2]
// C4 HighVol: threshold=0.4, classes=[0, 1, 2]

const double REGIME_SCALER_MEAN[REGIME_N_FEATS] = {+0.00251588439026234, +0.00187909178306638, +0.506775182088164, +1.78554587622058, +0.00243303499758662, +20.0785320614939, -0.0558639280652208};
const double REGIME_SCALER_STD [REGIME_N_FEATS] = {+0.0316368650776448, +0.00107179771294956, +0.0291106891236045, +14.3785643816174, +0.0013860292432575, +6.54537940880749, +0.114448832696698};

const double REGIME_PCA_MEAN[REGIME_N_FEATS] = {-1.25908340654477e-17, +6.71511150157212e-17, -1.76271676916268e-15, -8.39388937696515e-18, +2.68604460062885e-16, +4.94190237068823e-16, +1.67877787539303e-17};

// Pre-PCA rotation matrix: rows = output dims, cols = input dims.
// To rotate: for(i in 0..NF) out[i] = sum_j (scaled[j] - mean[j]) * comp[i][j]
const double REGIME_PCA_COMP[REGIME_N_FEATS][REGIME_N_FEATS] = {
    {-0.20050120381869, +0.652070863580704, +0.196622877990755, -0.201108352225861, +0.649397699977519, +0.09734563814722, +0.155897994373313},
    {+0.641925873707177, +0.123341856167656, +0.266285389911334, +0.644331657807988, +0.125297854091078, +0.224953864347628, +0.142628789954249},
    {-0.198938649156968, -0.213018525961819, +0.363473610709645, -0.189157557862894, -0.248210867633594, +0.532049289401099, +0.634403493784561},
    {-0.0121696176203602, -0.159901765193168, +0.734887823282263, -0.0461257666143023, +0.00413416339133139, -0.654726906033713, +0.0584055986131399},
    {+0.101444140143627, -0.0116474606128845, -0.465376328027719, +0.0713791809365698, +0.0974107715269548, -0.459834546103995, +0.739571930735549},
    {-0.705464576961652, -0.0177626097316032, +0.00398363631947719, +0.707650452143972, +0.0192178581234732, -0.026783969111878, +0.0115100607168316},
    {+0.00587973587381472, -0.698708619403102, -0.0408333250258143, -0.0248963346348984, +0.700784677647408, +0.126533106418965, -0.0487310426887753}
};

// Cluster centroids in post-PCA space
const double REGIME_CENTROIDS[REGIME_K][REGIME_N_FEATS] = {
    {-0.023694345529498, +2.26962984867184, +0.284172712649382, -0.226694509497553, -0.243288225726459, +0.0271025055346858, +0.0241740604028145},
    {-1.17721315697615, -0.907853922010263, -1.81653755325484, -0.736722117443405, -0.737876718612587, -0.0270200772457588, -0.0188300516042877},
    {-0.657568874070157, -0.217296488194964, -0.0168271651501998, +0.324617229050469, +0.338043840788835, +0.0243297532876927, -0.0246335111643546},
    {+0.634799441558512, -1.29283059274462, +1.25429668883458, -0.202699241964593, -0.423694034920829, -0.19262428229921, +0.0492886934657287},
    {+2.86688257340768, -0.185302658588356, -0.779213964455402, -0.0837552035893756, +0.220274130338202, +0.163063649890006, +0.0058938460511358}
};

// Cluster names (for logging)
const string REGIME_NAMES[REGIME_K] = {"Uptrend", "MeanRevert", "TrendRange", "Downtrend", "HighVol"};

// Is this cluster tradeable (has a model)?
const int REGIME_TRADEABLE[REGIME_K] = {1, 1, 1, 1, 1};

// Per-cluster probability threshold (0 = skip)
const double REGIME_THRESHOLD[REGIME_K] = {0.4000, 0.4000, 0.4000, 0.4000, 0.4000};

//+------------------------------------------------------------------+
//| ComputeWeekFingerprint — fill 7-dim fp[] from week of OHLC bars  |
//| `bars` is AS-SERIES, [0] = most recent bar of the week.          |
//+------------------------------------------------------------------+
void ComputeWeekFingerprint(const MqlRates &bars[], double &fp[])
{
   int n = ArraySize(bars);
   if(n < 2) { ArrayInitialize(fp, 0.0); return; }

   // Re-index oldest-first for clarity
   double sum_ret = 0.0, sum_ret2 = 0.0;
   double sum_range_pct = 0.0;
   double max_h = bars[n-1].high;
   double min_l = bars[n-1].low;
   double mean_close = 0.0;
   int n_ret = n - 1;
   int pos_ret = 0;

   // First pass: returns, range pct, max/min
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

   // Trend consistency: fraction of returns with sign(r) == sign(mean_ret)
   double trend_consistency = 0.5;
   if(MathAbs(mean_ret) > 1e-12)
      trend_consistency = (mean_ret > 0) ? (double)pos_ret / n_ret : (double)neg_ret / n_ret;

   // Return autocorrelation (lag 1)
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

   fp[0] = sum_ret;                                       // weekly_return_pct
   fp[1] = std_ret;                                       // volatility_pct
   fp[2] = trend_consistency;
   fp[3] = sum_ret / (std_ret + 1e-9);                    // trend_strength
   fp[4] = mean_bar_range;                                // volatility
   fp[5] = (max_h - min_l) / mean_close / (mean_bar_range + 1e-9);  // range_vs_atr
   fp[6] = r_autocorr;
}

//+------------------------------------------------------------------+
//| ClassifyRegime — fingerprint → cluster id (0..K-1)                |
//+------------------------------------------------------------------+
int ClassifyRegime(const double &fp[])
{
   // 1. Standardize
   double scaled[REGIME_N_FEATS];
   for(int i = 0; i < REGIME_N_FEATS; ++i)
      scaled[i] = (fp[i] - REGIME_SCALER_MEAN[i]) / REGIME_SCALER_STD[i];

   // 2. Centre by PCA mean then rotate
   double rotated[REGIME_N_FEATS];
   for(int i = 0; i < REGIME_N_FEATS; ++i)
   {
      double s = 0.0;
      for(int j = 0; j < REGIME_N_FEATS; ++j)
         s += (scaled[j] - REGIME_PCA_MEAN[j]) * REGIME_PCA_COMP[i][j];
      rotated[i] = s;
   }

   // 3. Nearest centroid (squared L2)
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

#endif  // __REGIME_SELECTOR_MQH__