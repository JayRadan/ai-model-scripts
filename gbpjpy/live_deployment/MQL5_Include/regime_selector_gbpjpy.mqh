//+------------------------------------------------------------------+
//| regime_selector_gbpjpy.mqh                                       |
//| Auto-generated from regime_selector_K4.json. DO NOT EDIT BY HAND.|
//+------------------------------------------------------------------+
#ifndef __REGIME_SELECTOR_GBPJPY_MQH__
#define __REGIME_SELECTOR_GBPJPY_MQH__

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

const double REGIME_SCALER_MEAN[REGIME_N_FEATS] = {+0.000185182687327554, +0.00036274610978996, +0.507871590039942, +1.04695381151984, +0.000500329896180754, +18.6807307212445, -0.044114442114176};
const double REGIME_SCALER_STD [REGIME_N_FEATS] = {+0.00630976131707114, +0.000139518318790788, +0.0237760339767559, +15.93667366052, +0.000171739268902812, +5.83463273631027, +0.0825534787437399};

const double REGIME_PCA_MEAN[REGIME_N_FEATS] = {-1.48932356961911e-17, +0, -4.26758899221767e-15, -8.12358310701334e-18, -1.73303106282951e-16, +5.00954291599156e-17, +5.95729427847645e-17};

const double REGIME_PCA_COMP[REGIME_N_FEATS][REGIME_N_FEATS] = {
    {-0.399011549205985, +0.556089546552826, +0.00127259134672664, -0.406395173127222, +0.546942885669323, +0.184628215188859, +0.182102843352221},
    {+0.541014213737111, +0.331179273253215, +0.301263677223356, +0.544240774370945, +0.320585202700414, +0.309552765411821, +0.109855810202701},
    {-0.184029951483717, -0.276254732620827, +0.481487998616714, -0.149937609033601, -0.311915268499656, +0.57238081823363, +0.458904548290646},
    {+0.135831839747667, -0.0263567321069341, -0.615216790813625, +0.112016661738817, -0.0272059631267937, -0.0510001655191204, +0.765816179291512},
    {+0.0154354830205951, +0.0309982317985125, -0.544097737211683, +0.00777084097441783, -0.127083116533716, +0.727854130538298, -0.395949956473517},
    {-0.702686254203337, +0.0495122261852178, -0.0228440941294182, +0.708509209478157, -0.0317141594936914, -0.016217971070722, +0.00214582428627633},
    {+0.04206978036458, +0.707574024143717, +0.0484972629198164, -0.0396262258659109, -0.694903879721552, -0.0991666863920593, +0.0303558545919212}
};

const double REGIME_CENTROIDS[REGIME_K][REGIME_N_FEATS] = {
    {-0.76076071623679, +2.43526564778526, +0.666726136199519, +0.0413576488454042, +0.441758750510452, -1.83828779971967e-05, -0.0331747057294269},
    {-1.04640683888312, +0.162926178236306, +0.0626468423058237, -0.382315301325186, -0.387696465139592, +0.0762475491207328, +0.0199618953105208},
    {-0.284678148636967, -0.856320238249258, -0.560021770462783, +0.395916194719109, +0.116978068948652, -0.0333344356307055, +0.00612587263128484},
    {+1.5392240183863, -0.939687541707871, +1.28873348879468, -0.320052325737134, +0.178963236013025, -0.0787595997723339, -0.0315785522417163},
    {+2.59870367337183, +1.30496481384116, -1.19120808673926, +0.0363631206957392, -0.281410291755644, +0.053813361504026, +0.0201266417196571}
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

#endif  // __REGIME_SELECTOR_GBPJPY_MQH__