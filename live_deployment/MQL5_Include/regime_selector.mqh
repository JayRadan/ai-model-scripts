//+------------------------------------------------------------------+
//| regime_selector.mqh                                               |
//| Auto-generated from regime_selector_K4.json. DO NOT EDIT BY HAND. |
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

const double REGIME_SCALER_MEAN[REGIME_N_FEATS] = {+0.00080177760142026, +0.000507811619576694, +0.50788733656744, +1.70583143104578, +0.000687932936754639, +19.7965706807434, -0.0272889500671677};
const double REGIME_SCALER_STD [REGIME_N_FEATS] = {+0.00889267675297552, +0.000199423787204588, +0.0246191472668004, +15.8487154434288, +0.000265778315482896, +5.95697708554118, +0.0875362551913955};

const double REGIME_PCA_MEAN[REGIME_N_FEATS] = {-1.71352749781375e-17, -2.17046816389741e-16, -1.27943386503426e-15, -1.71352749781375e-17, -1.48505716477191e-16, +2.22044604925031e-16, +2.28470333041833e-17};

// Pre-PCA rotation matrix: rows = output dims, cols = input dims.
// To rotate: for(i in 0..NF) out[i] = sum_j (scaled[j] - mean[j]) * comp[i][j]
const double REGIME_PCA_COMP[REGIME_N_FEATS][REGIME_N_FEATS] = {
    {+0.31546986631514, +0.582382243065122, +0.282284984955384, +0.294239439059702, +0.570159322120484, +0.237184393908224, +0.117089439599021},
    {+0.615425479477612, -0.333952742879381, +0.0394881780866264, +0.627702205320655, -0.329948676483321, +0.00472644669330253, -0.0725892660703737},
    {-0.105307001165912, -0.182608684578857, +0.322095891747392, -0.0900392481546175, -0.262626708921965, +0.661431491679186, +0.580729943871409},
    {-0.103015246798704, -0.0990161078719976, +0.755390269934565, -0.108395392054174, -0.0700814874177078, +0.0900419284760649, -0.619839240945685},
    {-0.00297093409608602, +0.103933958867443, -0.492047646179603, +0.000597409147491013, -0.0464380215963704, +0.696726958767328, -0.509404277380162},
    {+0.705373311106388, +0.0391137110198668, -0.00100148506052218, -0.704656527549089, -0.0658460358161713, -0.00330217580234631, +0.00549363281115167},
    {-0.0496810086321884, +0.70272627458238, +0.0477886867842233, +0.0549116380684591, -0.696904458936563, -0.112617483017522, +0.00707187712495297}
};

// Cluster centroids in post-PCA space
const double REGIME_CENTROIDS[REGIME_K][REGIME_N_FEATS] = {
    {+1.40160760669528, +1.62193787542403, +0.716651319763422, +0.324175136471367, +0.162350654591112, -0.0574283984331771, -0.0171299350613223},
    {-0.814718753520525, +0.252708900035879, -0.798296396535431, +0.376754930950975, +0.0194951870478249, +0.00548035088681341, +0.0301404852649703},
    {-0.599317563634894, +0.158383228267126, +0.234228397595826, -0.914509871465781, -0.2516100409639, +0.00148830064586277, +0.00409096629932675},
    {-0.410219995467957, -1.85315358131997, +1.19961701271084, +0.593786189934924, +0.267929045338428, +0.0739000681665042, -0.0431009492914486},
    {+2.48882538328402, -1.52758678060485, -1.09053006932806, -0.343079879661601, -0.102027985258738, -0.0293355422213926, -0.0170928918241606}
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