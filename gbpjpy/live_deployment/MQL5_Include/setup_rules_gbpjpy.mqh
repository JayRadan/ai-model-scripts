//+------------------------------------------------------------------+
//| setup_rules_gbpjpy.mqh                                           |
//| Hand-coded setup rules — exact mirror of                          |
//| 04_build_setup_signals.py (GBPJPY M5, 48 rules, 4 clusters).     |
//|                                                                    |
//| Each detector takes the M5 rates buffer (AS-SERIES, [0]=newest   |
//| fully-closed bar) and decides whether the CURRENT bar fires that  |
//| rule. Returns +1 (buy), -1 (sell), or 0 (no signal).             |
//+------------------------------------------------------------------+
#ifndef __SETUP_RULES_GBPJPY_MQH__
#define __SETUP_RULES_GBPJPY_MQH__

// Per-rule enum index — order MUST match confirmation_router_gbpjpy.mqh
enum RULE_ID
{
   // C0 Uptrend (buy only) — 12 rules
   RULE_R0a_pullback = 0,
   RULE_R0b_higher_low,
   RULE_R0c_breakout_pb,
   RULE_R0d_oversold,
   RULE_R0e_sma_bounce,
   RULE_R0f_false_breakdown,
   RULE_R0g_close_streak,
   RULE_R0h_pin_bar,
   RULE_R0i_engulfing,
   RULE_R0j_ema_pullback,
   RULE_R0k_tokyo_buy,
   RULE_R0l_london_open_buy,
   // C1 MeanRevert (both) — 8 rules
   RULE_R1a_bb,
   RULE_R1b_stoch,
   RULE_R1e_double_touch,
   RULE_R1f_mean_revert,
   RULE_R1g_close_extreme,
   RULE_R1i_pin_bar,
   RULE_R1j_engulfing,
   RULE_R1n_range_fade,
   // C2 TrendRange (both) — 6 rules
   RULE_R2a_inside_break,
   RULE_R2b_squeeze,
   RULE_R2c_nr4_break,
   RULE_R2d_3bar_reversal,
   RULE_R2e_session_break,
   RULE_R2f_rsi_divergence,
   // C3 Downtrend (sell only) — 12 rules
   RULE_R3a_swing_high,
   RULE_R3b_lower_high,
   RULE_R3c_bounce_fade,
   RULE_R3d_overbought,
   RULE_R3e_sma_reject,
   RULE_R3f_three_red,
   RULE_R3g_false_breakout,
   RULE_R3h_pin_bar,
   RULE_R3i_engulfing,
   RULE_R3j_ema_reject,
   RULE_R3k_london_sell,
   RULE_R3l_tokyo_fade,
   // C4 HighVol (both) — 10 rules
   RULE_R4a_v_reversal,
   RULE_R4b_momentum,
   RULE_R4c_bb_vol,
   RULE_R4d_stoch_vol,
   RULE_R4e_inside_vol,
   RULE_R4f_squeeze_vol,
   RULE_R4g_pin_vol,
   RULE_R4h_engulf_vol,
   RULE_R4i_spike_fade,
   RULE_R4j_session_vol,
   RULE_COUNT           // 48
};

const string RULE_NAMES[RULE_COUNT] =
{
   "R0a_pullback","R0b_higher_low","R0c_breakout_pb","R0d_oversold",
   "R0e_sma_bounce","R0f_false_breakdown","R0g_close_streak","R0h_pin_bar",
   "R0i_engulfing","R0j_ema_pullback","R0k_tokyo_buy","R0l_london_open_buy",
   "R1a_bb","R1b_stoch","R1e_double_touch","R1f_mean_revert",
   "R1g_close_extreme","R1i_pin_bar","R1j_engulfing","R1n_range_fade",
   "R2a_inside_break","R2b_squeeze","R2c_nr4_break","R2d_3bar_reversal",
   "R2e_session_break","R2f_rsi_divergence",
   "R3a_swing_high","R3b_lower_high","R3c_bounce_fade","R3d_overbought",
   "R3e_sma_reject","R3f_three_red","R3g_false_breakout","R3h_pin_bar",
   "R3i_engulfing","R3j_ema_reject","R3k_london_sell","R3l_tokyo_fade",
   "R4a_v_reversal","R4b_momentum","R4c_bb_vol","R4d_stoch_vol",
   "R4e_inside_vol","R4f_squeeze_vol","R4g_pin_vol","R4h_engulf_vol",
   "R4i_spike_fade","R4j_session_vol"
};

const int RULE_CLUSTER[RULE_COUNT] =
{
   0,0,0,0,0,0,0,0,0,0,0,0,          // C0 Uptrend x12
   1,1,1,1,1,1,1,1,                    // C1 MeanRevert x8
   2,2,2,2,2,2,                        // C2 TrendRange x6
   3,3,3,3,3,3,3,3,3,3,3,3,          // C3 Downtrend x12
   4,4,4,4,4,4,4,4,4,4               // C4 HighVol x10
};

const int RULE_COOLDOWN[RULE_COUNT] =
{
   3,3,3,3,3,3,3,3,3,3,3,3,          // C0 Uptrend x12
   3,3,3,3,3,3,3,3,                    // C1 MeanRevert x8
   3,3,3,3,3,3,                        // C2 TrendRange x6
   3,3,3,3,3,3,3,3,3,3,3,3,          // C3 Downtrend x12
   3,3,3,3,3,3,3,3,3,3               // C4 HighVol x10
};

//+------------------------------------------------------------------+
//| Feature indices — 34 features total (21 base + 13 tech)           |
//| Matches ALL_FEAT_COLS in Python pipeline.                         |
//+------------------------------------------------------------------+
#define FI_F01_CPR        0
#define FI_F02_WickAsym   1
#define FI_F03_BEF        2
#define FI_F04_TCS        3
#define FI_F05_SPI        4
#define FI_F06_LRSlope    5
#define FI_F07_RECR       6
#define FI_F08_SCM        7
#define FI_F09_HLER       8
#define FI_F10_EP         9
#define FI_F11_KE        10
#define FI_F12_MCS       11
#define FI_F13_Work      12
#define FI_F14_EDR       13
#define FI_F15_AI        14
#define FI_F16_PPShigh   15
#define FI_F16_PPSlow    16
#define FI_F17_SCR       17
#define FI_F18_RVD       18
#define FI_F19_WBER      19
#define FI_F20_NCDE      20
// 13 tech features
#define FI_STOCH_K       21
#define FI_RSI14         22
#define FI_BB_PCT        23
#define FI_VOL_RATIO     24
#define FI_RANGE_ATR     25
#define FI_DIST_SMA20    26
#define FI_DIST_SMA50    27
#define FI_BODY_RATIO    28
#define FI_CONSEC_DIR    29
#define FI_HOUR_SIN      30
#define FI_HOUR_COS      31
#define FI_DOW_SIN       32
#define FI_DOW_COS       33

//+------------------------------------------------------------------+
//| Helper: simple SMA over last N closes from rb[] (AS-SERIES).     |
//+------------------------------------------------------------------+
double SMAClose(const MqlRates &rb[], int idx, int period)
{
   if(idx + period > ArraySize(rb)) return rb[idx].close;
   double s = 0;
   for(int k = 0; k < period; k++) s += rb[idx + k].close;
   return s / period;
}

//+------------------------------------------------------------------+
//| Helper: inline ATR(14)                                            |
//+------------------------------------------------------------------+
double InlineATR14(const MqlRates &rb[], int idx)
{
   int p = 14;
   if(idx + p >= ArraySize(rb)) p = ArraySize(rb) - idx - 1;
   if(p < 1) return rb[idx].high - rb[idx].low;
   double s = 0;
   for(int k = 0; k < p; k++)
   {
      int ki = idx + k;
      double tr = MathMax(rb[ki].high - rb[ki].low,
                  MathMax(MathAbs(rb[ki].high - rb[ki+1].close),
                          MathAbs(rb[ki].low  - rb[ki+1].close)));
      s += tr;
   }
   return s / p;
}

//+------------------------------------------------------------------+
//| Helper: Bollinger %B over period closes                           |
//+------------------------------------------------------------------+
double InlineBollingerPctB(const MqlRates &rb[], int idx, int period)
{
   if(idx + period >= ArraySize(rb)) return 0.5;
   double sum = 0;
   for(int k = idx; k < idx + period; k++) sum += rb[k].close;
   double sma = sum / period;
   double var = 0;
   for(int k = idx; k < idx + period; k++)
   {
      double d = rb[k].close - sma;
      var += d * d;
   }
   double std_ = MathSqrt(var / (period - 1)); // ddof=1 to match pandas rolling.std()
   double upper = sma + 2.0 * std_, lower = sma - 2.0 * std_;
   double w = upper - lower;
   if(w < 1e-10) return 0.5;
   return (rb[idx].close - lower) / w;
}

//+------------------------------------------------------------------+
//| Helper: Bollinger bandwidth (upper - lower)                       |
//+------------------------------------------------------------------+
double InlineBollingerWidth(const MqlRates &rb[], int idx, int period)
{
   if(idx + period >= ArraySize(rb)) return 0.0;
   double sum = 0;
   for(int k = idx; k < idx + period; k++) sum += rb[k].close;
   double sma = sum / period;
   double var = 0;
   for(int k = idx; k < idx + period; k++)
   {
      double d = rb[k].close - sma;
      var += d * d;
   }
   double std_ = MathSqrt(var / (period - 1)); // ddof=1 to match pandas rolling.std()
   return 4.0 * std_;   // upper - lower = 4 * std
}

//+------------------------------------------------------------------+
//| Helper: StochK (smoothed %K over 14-period, 3-period SMA)        |
//+------------------------------------------------------------------+
double InlineStochK(const MqlRates &rb[], int idx)
{
   // Raw %K over 14 bars
   if(idx + 14 >= ArraySize(rb)) return 0.5;
   double lo = rb[idx].low, hi = rb[idx].high;
   for(int k = idx; k < idx + 14; k++)
   {
      lo = MathMin(lo, rb[k].low);
      hi = MathMax(hi, rb[k].high);
   }
   double rk0 = (hi - lo > 1e-10) ? (rb[idx].close - lo) / (hi - lo) : 0.5;

   // bar idx+1
   lo = rb[idx+1].low; hi = rb[idx+1].high;
   for(int k = idx+1; k < idx + 15 && k < ArraySize(rb); k++)
   {
      lo = MathMin(lo, rb[k].low);
      hi = MathMax(hi, rb[k].high);
   }
   double rk1 = (hi - lo > 1e-10) ? (rb[idx+1].close - lo) / (hi - lo) : 0.5;

   // bar idx+2
   lo = rb[idx+2].low; hi = rb[idx+2].high;
   for(int k = idx+2; k < idx + 16 && k < ArraySize(rb); k++)
   {
      lo = MathMin(lo, rb[k].low);
      hi = MathMax(hi, rb[k].high);
   }
   double rk2 = (hi - lo > 1e-10) ? (rb[idx+2].close - lo) / (hi - lo) : 0.5;

   return (rk0 + rk1 + rk2) / 3.0;
}

//+------------------------------------------------------------------+
//| Helper: RSI(14) using EWM (matching Python ewm(span=14))          |
//+------------------------------------------------------------------+
double InlineRSI14(const MqlRates &rb[], int idx)
{
   // Need enough history; we'll use up to 100 bars
   int maxbars = ArraySize(rb);
   int start = MathMin(idx + 100, maxbars - 1);
   double alpha = 2.0 / (14.0 + 1.0);  // ewm span=14
   double avg_g = 0.0, avg_l = 0.0;

   // Walk forward from oldest to newest (remember AS-SERIES: higher idx = older)
   for(int k = start; k > idx; k--)
   {
      if(k >= maxbars) continue;
      double delta = rb[k-1].close - rb[k].close;
      double g = (delta > 0) ? delta : 0.0;
      double lv = (delta < 0) ? -delta : 0.0;
      avg_g = alpha * g + (1.0 - alpha) * avg_g;
      avg_l = alpha * lv + (1.0 - alpha) * avg_l;
   }
   if(avg_l < 1e-10) return 50.0;
   return 100.0 - 100.0 / (1.0 + avg_g / avg_l);
}

//+------------------------------------------------------------------+
//| Helper: EMA(12) of close                                          |
//+------------------------------------------------------------------+
double InlineEMA12(const MqlRates &rb[], int idx)
{
   int maxbars = ArraySize(rb);
   int start = MathMin(idx + 100, maxbars - 1);
   double alpha = 2.0 / (12.0 + 1.0);
   double ema = rb[start].close;
   for(int k = start - 1; k >= idx; k--)
      ema = alpha * rb[k].close + (1.0 - alpha) * ema;
   return ema;
}

//+------------------------------------------------------------------+
//| Helper: SMA(50) of close                                          |
//+------------------------------------------------------------------+
double InlineSMA50(const MqlRates &rb[], int idx)
{
   if(idx + 50 > ArraySize(rb)) return rb[idx].close;
   double s = 0;
   for(int k = 0; k < 50; k++) s += rb[idx + k].close;
   return s / 50.0;
}

//+------------------------------------------------------------------+
//| Helper: bar body / range ratio                                    |
//+------------------------------------------------------------------+
double BarBodyRatio(const MqlRates &rb[], int idx)
{
   double rng = rb[idx].high - rb[idx].low;
   if(rng < 1e-10) return 0.0;
   return MathAbs(rb[idx].close - rb[idx].open) / rng;
}

//+------------------------------------------------------------------+
//| Helper: upper wick / range ratio                                  |
//+------------------------------------------------------------------+
double WickRatioUp(const MqlRates &rb[], int idx)
{
   double rng = rb[idx].high - rb[idx].low;
   if(rng < 1e-10) return 0.0;
   return (rb[idx].high - MathMax(rb[idx].close, rb[idx].open)) / rng;
}

//+------------------------------------------------------------------+
//| Helper: lower wick / range ratio                                  |
//+------------------------------------------------------------------+
double WickRatioLo(const MqlRates &rb[], int idx)
{
   double rng = rb[idx].high - rb[idx].low;
   if(rng < 1e-10) return 0.0;
   return (MathMin(rb[idx].close, rb[idx].open) - rb[idx].low) / rng;
}

//+------------------------------------------------------------------+
//| Helper: is bar bullish (close > open)                             |
//+------------------------------------------------------------------+
bool IsBullish(const MqlRates &rb[], int idx)
{
   return rb[idx].close > rb[idx].open;
}

//+------------------------------------------------------------------+
//| Session helpers                                                    |
//+------------------------------------------------------------------+
bool IsTokyoOpen(const MqlRates &rb[], int idx)
{
   MqlDateTime dt;
   TimeToStruct(rb[idx].time, dt);
   return (dt.hour >= 0 && dt.hour <= 3);
}

bool IsTokyo(const MqlRates &rb[], int idx)
{
   MqlDateTime dt;
   TimeToStruct(rb[idx].time, dt);
   return (dt.hour >= 0 && dt.hour <= 8);
}

bool IsLondonOpen(const MqlRates &rb[], int idx)
{
   MqlDateTime dt;
   TimeToStruct(rb[idx].time, dt);
   return (dt.hour >= 7 && dt.hour <= 10);
}

bool IsNYOpen(const MqlRates &rb[], int idx)
{
   MqlDateTime dt;
   TimeToStruct(rb[idx].time, dt);
   return (dt.hour >= 13 && dt.hour <= 16);
}

//+------------------------------------------------------------------+
//| Helper: range / ATR ratio for a bar                               |
//+------------------------------------------------------------------+
double BarRangeATR(const MqlRates &rb[], int idx)
{
   double atr = InlineATR14(rb, idx);
   if(atr < 1e-10) return 1.0;
   return (rb[idx].high - rb[idx].low) / atr;
}

//+------------------------------------------------------------------+
//| Helper: dist_sma20 = (close - sma20) / atr                       |
//+------------------------------------------------------------------+
double DistSMA20(const MqlRates &rb[], int idx)
{
   double sma20 = SMAClose(rb, idx, 20);
   double atr = InlineATR14(rb, idx);
   if(atr < 1e-10) return 0.0;
   return (rb[idx].close - sma20) / atr;
}

//+------------------------------------------------------------------+
//| Helper: dist_sma50 = (close - sma50) / atr                       |
//+------------------------------------------------------------------+
double DistSMA50(const MqlRates &rb[], int idx)
{
   double sma50 = InlineSMA50(rb, idx);
   double atr = InlineATR14(rb, idx);
   if(atr < 1e-10) return 0.0;
   return (rb[idx].close - sma50) / atr;
}

//+------------------------------------------------------------------+
//| Helper: BB bandwidth rolling 5-bar average                        |
//+------------------------------------------------------------------+
double BBWidthMA5(const MqlRates &rb[], int idx)
{
   double s = 0;
   int cnt = 0;
   for(int k = 0; k < 5 && idx + k < ArraySize(rb); k++)
   {
      s += InlineBollingerWidth(rb, idx + k, 20);
      cnt++;
   }
   return (cnt > 0) ? s / cnt : 0.0;
}

//+------------------------------------------------------------------+
//| Helper: highest high over [idx-start .. idx-end] (inclusive, but  |
//| using AS-SERIES offsets from idx)                                  |
//+------------------------------------------------------------------+
double HiN(const MqlRates &rb[], int from, int to)
{
   // from, to are AS-SERIES indices
   double mx = rb[from].high;
   for(int k = from + 1; k <= to && k < ArraySize(rb); k++)
      if(rb[k].high > mx) mx = rb[k].high;
   return mx;
}

double LoN(const MqlRates &rb[], int from, int to)
{
   double mn = rb[from].low;
   for(int k = from + 1; k <= to && k < ArraySize(rb); k++)
      if(rb[k].low < mn) mn = rb[k].low;
   return mn;
}

//+------------------------------------------------------------------+
//| Helper: rolling high/low over N bars ending at idx                |
//+------------------------------------------------------------------+
double Hi20(const MqlRates &rb[], int idx)
{
   double mx = rb[idx].high;
   for(int k = 1; k < 20 && idx + k < ArraySize(rb); k++)
      if(rb[idx + k].high > mx) mx = rb[idx + k].high;
   return mx;
}

double Lo20(const MqlRates &rb[], int idx)
{
   double mn = rb[idx].low;
   for(int k = 1; k < 20 && idx + k < ArraySize(rb); k++)
      if(rb[idx + k].low < mn) mn = rb[idx + k].low;
   return mn;
}

// ===================================================================
// C0 UPTREND RULES (buy only) — 12 rules
// ===================================================================

//+------------------------------------------------------------------+
//| R0a: Pullback to SMA20 in uptrend (buy)                          |
//| Python: dist_sma20 > -0.5 and < 0.5 and dist_sma50 > 0 and bull |
//+------------------------------------------------------------------+
int Rule_R0a_pullback(const float &feat[], const MqlRates &rb[])
{
   double dsma20 = (double)feat[FI_DIST_SMA20];
   double dsma50 = (double)feat[FI_DIST_SMA50];
   if(dsma20 > -0.5 && dsma20 < 0.5 && dsma50 > 0 && IsBullish(rb, 0))
      return +1;
   return 0;
}

//+------------------------------------------------------------------+
//| R0b: Higher low formation (buy)                                   |
//| Python: min(l[i-3..i-9]) < l[i] and bullish and stoch_k < 0.5   |
//+------------------------------------------------------------------+
int Rule_R0b_higher_low(const float &feat[], const MqlRates &rb[])
{
   if(ArraySize(rb) < 10) return 0;
   double minLo = rb[3].low;
   for(int k = 4; k <= 9; k++)
      if(rb[k].low < minLo) minLo = rb[k].low;
   if(minLo < rb[0].low && IsBullish(rb, 0) && (double)feat[FI_STOCH_K] < 0.5)
      return +1;
   return 0;
}

//+------------------------------------------------------------------+
//| R0c: Breakout pullback (buy)                                      |
//| Python: h[i-1] > hi20[i-2] and c[i] > sma20[i] and bullish      |
//+------------------------------------------------------------------+
int Rule_R0c_breakout_pb(const float &feat[], const MqlRates &rb[])
{
   if(ArraySize(rb) < 22) return 0;
   double hi20_at2 = Hi20(rb, 2);
   if(rb[1].high > hi20_at2 && rb[0].close > SMAClose(rb, 0, 20) && IsBullish(rb, 0))
      return +1;
   return 0;
}

//+------------------------------------------------------------------+
//| R0d: Oversold in uptrend (buy)                                    |
//| Python: rsi < 35 and dist_sma50 > 0 and bullish                  |
//+------------------------------------------------------------------+
int Rule_R0d_oversold(const float &feat[], const MqlRates &rb[])
{
   double rsi = (double)feat[FI_RSI14];
   double dsma50 = (double)feat[FI_DIST_SMA50];
   if(rsi < 35.0 && dsma50 > 0 && IsBullish(rb, 0))
      return +1;
   return 0;
}

//+------------------------------------------------------------------+
//| R0e: SMA50 bounce (buy)                                           |
//| Python: abs(l[i]-sma50) < 0.5*atr and c > sma50 and bullish     |
//+------------------------------------------------------------------+
int Rule_R0e_sma_bounce(const float &feat[], const MqlRates &rb[])
{
   double atr = InlineATR14(rb, 0);
   if(atr < 1e-10) return 0;
   double sma50 = InlineSMA50(rb, 0);
   if(MathAbs(rb[0].low - sma50) < 0.5 * atr && rb[0].close > sma50 && IsBullish(rb, 0))
      return +1;
   return 0;
}

//+------------------------------------------------------------------+
//| R0f: False breakdown (buy)                                        |
//| Python: l[i] < min(l[i-2..i-9]) and c[i] > support and bullish  |
//+------------------------------------------------------------------+
int Rule_R0f_false_breakdown(const float &feat[], const MqlRates &rb[])
{
   if(ArraySize(rb) < 10) return 0;
   double support = rb[2].low;
   for(int k = 3; k <= 9; k++)
      if(rb[k].low < support) support = rb[k].low;
   if(rb[0].low < support && rb[0].close > support && IsBullish(rb, 0))
      return +1;
   return 0;
}

//+------------------------------------------------------------------+
//| R0g: Close streak >= 3 (buy)                                      |
//| Python: consec[i] >= 3                                            |
//+------------------------------------------------------------------+
int Rule_R0g_close_streak(const float &feat[], const MqlRates &rb[])
{
   double consec = (double)feat[FI_CONSEC_DIR];
   if(consec >= 3.0) return +1;
   return 0;
}

//+------------------------------------------------------------------+
//| R0h: Pin bar (buy) — long lower wick, small body, bullish        |
//| Python: wick_ratio_lo > 0.6 and body_ratio < 0.25 and bullish   |
//+------------------------------------------------------------------+
int Rule_R0h_pin_bar(const float &feat[], const MqlRates &rb[])
{
   if(WickRatioLo(rb, 0) > 0.6 && BarBodyRatio(rb, 0) < 0.25 && IsBullish(rb, 0))
      return +1;
   return 0;
}

//+------------------------------------------------------------------+
//| R0i: Bullish engulfing (buy)                                      |
//| Python: !bull[i-1] and bull[i] and o[i]<=c[i-1] and c[i]>=o[i-1]|
//|         and body_ratio > 0.5                                      |
//+------------------------------------------------------------------+
int Rule_R0i_engulfing(const float &feat[], const MqlRates &rb[])
{
   if(ArraySize(rb) < 2) return 0;
   if(!IsBullish(rb, 1) && IsBullish(rb, 0) &&
      rb[0].open <= rb[1].close && rb[0].close >= rb[1].open &&
      BarBodyRatio(rb, 0) > 0.5)
      return +1;
   return 0;
}

//+------------------------------------------------------------------+
//| R0j: EMA12 pullback (buy)                                        |
//| Python: abs(l-ema12) < 0.3*atr and c > ema12 and bull            |
//|         and dist_sma50 > 0                                        |
//+------------------------------------------------------------------+
int Rule_R0j_ema_pullback(const float &feat[], const MqlRates &rb[])
{
   double atr = InlineATR14(rb, 0);
   if(atr < 1e-10) return 0;
   double ema12 = InlineEMA12(rb, 0);
   double dsma50 = (double)feat[FI_DIST_SMA50];
   if(MathAbs(rb[0].low - ema12) < 0.3 * atr && rb[0].close > ema12 &&
      IsBullish(rb, 0) && dsma50 > 0)
      return +1;
   return 0;
}

//+------------------------------------------------------------------+
//| R0k: Tokyo open buy (GBPJPY-specific)                             |
//| tokyo_open(0-3h) AND bullish AND dist_sma20 > -0.3 AND < 0.3    |
//|   AND stoch_k < 0.4                                               |
//+------------------------------------------------------------------+
int Rule_R0k_tokyo_buy(const float &feat[], const MqlRates &rb[])
{
   if(!IsTokyoOpen(rb, 0)) return 0;
   double dsma20 = (double)feat[FI_DIST_SMA20];
   double sk = (double)feat[FI_STOCH_K];
   if(IsBullish(rb, 0) && dsma20 > -0.3 && dsma20 < 0.3 && sk < 0.4)
      return +1;
   return 0;
}

//+------------------------------------------------------------------+
//| R0l: London open buy (GBPJPY-specific)                            |
//| london_open(7-10h) AND bullish AND dist_sma20 > 0 AND            |
//|   range_atr > 0.8                                                 |
//+------------------------------------------------------------------+
int Rule_R0l_london_open_buy(const float &feat[], const MqlRates &rb[])
{
   if(!IsLondonOpen(rb, 0)) return 0;
   double dsma20 = (double)feat[FI_DIST_SMA20];
   double ra = (double)feat[FI_RANGE_ATR];
   if(IsBullish(rb, 0) && dsma20 > 0 && ra > 0.8)
      return +1;
   return 0;
}

// ===================================================================
// C1 MEANREVERT RULES (both) — 14 rules
// ===================================================================

//+------------------------------------------------------------------+
//| R1a: BB extreme (buy/sell)                                        |
//| Buy: bb_pct < 0.05 and stoch_k < 0.25                            |
//| Sell: bb_pct > 0.95 and stoch_k > 0.75                           |
//+------------------------------------------------------------------+
int Rule_R1a_bb(const float &feat[], const MqlRates &rb[])
{
   double bb = (double)feat[FI_BB_PCT];
   double sk = (double)feat[FI_STOCH_K];
   if(bb < 0.05 && sk < 0.25) return +1;
   if(bb > 0.95 && sk > 0.75) return -1;
   return 0;
}

//+------------------------------------------------------------------+
//| R1b: Stochastic extreme (buy/sell)                                |
//| Buy: sk < 0.15 and bullish and body_ratio > 0.4                  |
//| Sell: sk > 0.85 and !bullish and body_ratio > 0.4                |
//+------------------------------------------------------------------+
int Rule_R1b_stoch(const float &feat[], const MqlRates &rb[])
{
   double sk = (double)feat[FI_STOCH_K];
   double br = BarBodyRatio(rb, 0);
   if(sk < 0.15 && IsBullish(rb, 0) && br > 0.4) return +1;
   if(sk > 0.85 && !IsBullish(rb, 0) && br > 0.4) return -1;
   return 0;
}

//+------------------------------------------------------------------+
//| R1c: Inside bar breakout (buy/sell)                               |
//| Buy: h[i-1]<h[i-2] and l[i-1]>l[i-2] and c[i]>h[i-1]           |
//| Sell: same inside bar and c[i]<l[i-1]                             |
//+------------------------------------------------------------------+
int Rule_R2a_inside_break(const float &feat[], const MqlRates &rb[])
{
   if(ArraySize(rb) < 3) return 0;
   bool inside = (rb[1].high < rb[2].high && rb[1].low > rb[2].low);
   if(!inside) return 0;
   if(rb[0].close > rb[1].high) return +1;
   if(rb[0].close < rb[1].low)  return -1;
   return 0;
}

//+------------------------------------------------------------------+
//| R1d: BB squeeze breakout (buy/sell)                               |
//| Python: bb_w < bb_w_ma5 * 0.7 and c > sma20 (buy) / c < (sell)  |
//+------------------------------------------------------------------+
int Rule_R2b_squeeze(const float &feat[], const MqlRates &rb[])
{
   if(ArraySize(rb) < 25) return 0;
   double bbw = InlineBollingerWidth(rb, 0, 20);
   double bbw_ma5 = BBWidthMA5(rb, 0);
   if(bbw >= bbw_ma5 * 0.7) return 0;
   double sma20 = SMAClose(rb, 0, 20);
   if(rb[0].close > sma20) return +1;
   if(rb[0].close < sma20) return -1;
   return 0;
}

//+------------------------------------------------------------------+
//| R1e: Double touch at BB extremes (buy/sell)                       |
//| Buy: bb_pct < 0.10 and any bb_pct[i-5..i-19] < 0.10             |
//|      and rsi > rsi[i-1]                                           |
//| Sell: bb_pct > 0.90 and any bb_pct[i-5..i-19] > 0.90            |
//|      and rsi < rsi[i-1]                                           |
//+------------------------------------------------------------------+
int Rule_R1e_double_touch(const float &feat[], const MqlRates &rb[])
{
   if(ArraySize(rb) < 20) return 0;
   double bb0 = InlineBollingerPctB(rb, 0, 20);
   double rsi0 = InlineRSI14(rb, 0);
   double rsi1 = InlineRSI14(rb, 1);

   if(bb0 < 0.10)
   {
      bool prior = false;
      for(int k = 5; k < 20 && k < ArraySize(rb); k++)
      {
         if(InlineBollingerPctB(rb, k, 20) < 0.10) { prior = true; break; }
      }
      if(prior && rsi0 > rsi1) return +1;
   }
   if(bb0 > 0.90)
   {
      bool prior = false;
      for(int k = 5; k < 20 && k < ArraySize(rb); k++)
      {
         if(InlineBollingerPctB(rb, k, 20) > 0.90) { prior = true; break; }
      }
      if(prior && rsi0 < rsi1) return -1;
   }
   return 0;
}

//+------------------------------------------------------------------+
//| R1f: Mean reversion (buy/sell)                                    |
//| Buy: dist_sma20 < -2.0 and bullish                               |
//| Sell: dist_sma20 > 2.0 and !bullish                              |
//+------------------------------------------------------------------+
int Rule_R1f_mean_revert(const float &feat[], const MqlRates &rb[])
{
   double d = (double)feat[FI_DIST_SMA20];
   if(d < -2.0 && IsBullish(rb, 0)) return +1;
   if(d >  2.0 && !IsBullish(rb, 0)) return -1;
   return 0;
}

//+------------------------------------------------------------------+
//| R1g: Close at BB extreme (buy/sell)                               |
//| Buy: bb_pct < 0.08 and bullish and body_ratio > 0.5              |
//| Sell: bb_pct > 0.92 and !bullish and body_ratio > 0.5            |
//+------------------------------------------------------------------+
int Rule_R1g_close_extreme(const float &feat[], const MqlRates &rb[])
{
   double bb = (double)feat[FI_BB_PCT];
   double br = BarBodyRatio(rb, 0);
   if(bb < 0.08 && IsBullish(rb, 0) && br > 0.5) return +1;
   if(bb > 0.92 && !IsBullish(rb, 0) && br > 0.5) return -1;
   return 0;
}

//+------------------------------------------------------------------+
//| R1h: NR4 breakout (buy/sell)                                      |
//| Python: all(range[i] < range[i-k] for k in 1..3)                 |
//|   and c > h[i-1] (buy) / c < l[i-1] (sell)                       |
//+------------------------------------------------------------------+
int Rule_R2c_nr4_break(const float &feat[], const MqlRates &rb[])
{
   if(ArraySize(rb) < 4) return 0;
   double r0 = rb[0].high - rb[0].low;
   for(int k = 1; k <= 3; k++)
   {
      double rk = rb[k].high - rb[k].low;
      if(r0 >= rk) return 0;
   }
   if(rb[0].close > rb[1].high) return +1;
   if(rb[0].close < rb[1].low)  return -1;
   return 0;
}

//+------------------------------------------------------------------+
//| R1i: Pin bar at BB extreme (buy/sell)                             |
//| Buy: wick_lo > 0.6, body < 0.25, bb_pct < 0.3                   |
//| Sell: wick_up > 0.6, body < 0.25, bb_pct > 0.7                  |
//+------------------------------------------------------------------+
int Rule_R1i_pin_bar(const float &feat[], const MqlRates &rb[])
{
   double bb = (double)feat[FI_BB_PCT];
   if(WickRatioLo(rb, 0) > 0.6 && BarBodyRatio(rb, 0) < 0.25 && bb < 0.3) return +1;
   if(WickRatioUp(rb, 0) > 0.6 && BarBodyRatio(rb, 0) < 0.25 && bb > 0.7) return -1;
   return 0;
}

//+------------------------------------------------------------------+
//| R1j: Engulfing at BB extreme (buy/sell)                           |
//| Buy: bear[i-1] bull[i] engulf and bb_pct < 0.3                   |
//| Sell: bull[i-1] bear[i] engulf and bb_pct > 0.7                  |
//+------------------------------------------------------------------+
int Rule_R1j_engulfing(const float &feat[], const MqlRates &rb[])
{
   if(ArraySize(rb) < 2) return 0;
   double bb = (double)feat[FI_BB_PCT];
   if(!IsBullish(rb, 1) && IsBullish(rb, 0) &&
      rb[0].open <= rb[1].close && rb[0].close >= rb[1].open && bb < 0.3)
      return +1;
   if(IsBullish(rb, 1) && !IsBullish(rb, 0) &&
      rb[0].open >= rb[1].close && rb[0].close <= rb[1].open && bb > 0.7)
      return -1;
   return 0;
}

//+------------------------------------------------------------------+
//| R1k: RSI divergence (buy/sell)                                    |
//| Buy: rsi < 35, rsi > rsi[i-5], l < l[i-5], bullish              |
//| Sell: rsi > 65, rsi < rsi[i-5], h > h[i-5], !bullish            |
//+------------------------------------------------------------------+
int Rule_R2f_rsi_divergence(const float &feat[], const MqlRates &rb[])
{
   if(ArraySize(rb) < 10) return 0;
   double rsi0 = InlineRSI14(rb, 0);
   double rsi5 = InlineRSI14(rb, 5);
   if(rsi0 < 35.0 && rsi0 > rsi5 && rb[0].low < rb[5].low && IsBullish(rb, 0))
      return +1;
   if(rsi0 > 65.0 && rsi0 < rsi5 && rb[0].high > rb[5].high && !IsBullish(rb, 0))
      return -1;
   return 0;
}

//+------------------------------------------------------------------+
//| R1l: Session break (buy/sell) — London open breaks Asian range    |
//| Buy: c > asian_hi, bullish                                        |
//| Sell: c < asian_lo, !bullish                                      |
//+------------------------------------------------------------------+
int Rule_R2e_session_break(const float &feat[], const MqlRates &rb[])
{
   if(ArraySize(rb) < 12) return 0;
   if(!IsLondonOpen(rb, 0)) return 0;
   double asian_hi = rb[1].high, asian_lo = rb[1].low;
   for(int k = 2; k <= 11; k++)
   {
      if(rb[k].high > asian_hi) asian_hi = rb[k].high;
      if(rb[k].low  < asian_lo) asian_lo = rb[k].low;
   }
   if(rb[0].close > asian_hi && IsBullish(rb, 0)) return +1;
   if(rb[0].close < asian_lo && !IsBullish(rb, 0)) return -1;
   return 0;
}

//+------------------------------------------------------------------+
//| R1m: 3-bar reversal (buy/sell)                                    |
//| Buy: bear[i-2], bear[i-1], bull[i], c > h[i-1], sk < 0.4        |
//| Sell: bull[i-2], bull[i-1], bear[i], c < l[i-1], sk > 0.6       |
//+------------------------------------------------------------------+
int Rule_R2d_3bar_reversal(const float &feat[], const MqlRates &rb[])
{
   if(ArraySize(rb) < 3) return 0;
   double sk = (double)feat[FI_STOCH_K];
   if(!IsBullish(rb, 2) && !IsBullish(rb, 1) && IsBullish(rb, 0) &&
      rb[0].close > rb[1].high && sk < 0.4)
      return +1;
   if(IsBullish(rb, 2) && IsBullish(rb, 1) && !IsBullish(rb, 0) &&
      rb[0].close < rb[1].low && sk > 0.6)
      return -1;
   return 0;
}

//+------------------------------------------------------------------+
//| R1n: Range fade at 20-bar extremes (buy/sell)                     |
//| Buy: c <= lo20 * 1.001 and bullish and rsi < 40                  |
//| Sell: c >= hi20 * 0.999 and !bullish and rsi > 60                |
//+------------------------------------------------------------------+
int Rule_R1n_range_fade(const float &feat[], const MqlRates &rb[])
{
   if(ArraySize(rb) < 20) return 0;
   double lo20 = Lo20(rb, 0);
   double hi20 = Hi20(rb, 0);
   double rsi = (double)feat[FI_RSI14];
   if(rb[0].close <= lo20 * 1.001 && IsBullish(rb, 0) && rsi < 40.0) return +1;
   if(rb[0].close >= hi20 * 0.999 && !IsBullish(rb, 0) && rsi > 60.0) return -1;
   return 0;
}

// ===================================================================
// C2 TRENDRANGE + C3 DOWNTREND RULES (sell only) — 12 rules
// ===================================================================

//+------------------------------------------------------------------+
//| R2a: Swing high rejection (sell)                                  |
//| Python: h[i] >= max(h[i-3..i-9]) * 0.999 and !bullish            |
//+------------------------------------------------------------------+
int Rule_R3a_swing_high(const float &feat[], const MqlRates &rb[])
{
   if(ArraySize(rb) < 10) return 0;
   double hi_prior = rb[3].high;
   for(int k = 4; k <= 9; k++)
      if(rb[k].high > hi_prior) hi_prior = rb[k].high;
   if(rb[0].high >= hi_prior * 0.999 && !IsBullish(rb, 0))
      return -1;
   return 0;
}

//+------------------------------------------------------------------+
//| R2b: Lower high (sell)                                            |
//| Python: h[i] < max(h[i-3..i-9]) and !bullish and stoch_k > 0.5  |
//+------------------------------------------------------------------+
int Rule_R3b_lower_high(const float &feat[], const MqlRates &rb[])
{
   if(ArraySize(rb) < 10) return 0;
   double hi_prior = rb[3].high;
   for(int k = 4; k <= 9; k++)
      if(rb[k].high > hi_prior) hi_prior = rb[k].high;
   if(rb[0].high < hi_prior && !IsBullish(rb, 0) && (double)feat[FI_STOCH_K] > 0.5)
      return -1;
   return 0;
}

//+------------------------------------------------------------------+
//| R2c: Bounce fade (sell)                                           |
//| Python: dist_sma20 > 0 and < 1.0 and dist_sma50 < 0 and !bull   |
//+------------------------------------------------------------------+
int Rule_R3c_bounce_fade(const float &feat[], const MqlRates &rb[])
{
   double dsma20 = (double)feat[FI_DIST_SMA20];
   double dsma50 = (double)feat[FI_DIST_SMA50];
   if(dsma20 > 0 && dsma20 < 1.0 && dsma50 < 0 && !IsBullish(rb, 0))
      return -1;
   return 0;
}

//+------------------------------------------------------------------+
//| R2d: Overbought in downtrend (sell)                               |
//| Python: rsi > 65 and dist_sma50 < 0 and !bullish                 |
//+------------------------------------------------------------------+
int Rule_R3d_overbought(const float &feat[], const MqlRates &rb[])
{
   double rsi = (double)feat[FI_RSI14];
   double dsma50 = (double)feat[FI_DIST_SMA50];
   if(rsi > 65.0 && dsma50 < 0 && !IsBullish(rb, 0))
      return -1;
   return 0;
}

//+------------------------------------------------------------------+
//| R2e: SMA50 rejection (sell)                                       |
//| Python: abs(h-sma50) < 0.5*atr and c < sma50 and !bullish       |
//+------------------------------------------------------------------+
int Rule_R3e_sma_reject(const float &feat[], const MqlRates &rb[])
{
   double atr = InlineATR14(rb, 0);
   if(atr < 1e-10) return 0;
   double sma50 = InlineSMA50(rb, 0);
   if(MathAbs(rb[0].high - sma50) < 0.5 * atr && rb[0].close < sma50 && !IsBullish(rb, 0))
      return -1;
   return 0;
}

//+------------------------------------------------------------------+
//| R2f: Three red bars (sell)                                        |
//| Python: consec <= -3                                              |
//+------------------------------------------------------------------+
int Rule_R3f_three_red(const float &feat[], const MqlRates &rb[])
{
   double consec = (double)feat[FI_CONSEC_DIR];
   if(consec <= -3.0) return -1;
   return 0;
}

//+------------------------------------------------------------------+
//| R2g: False breakout (sell)                                        |
//| Python: h > max(h[i-2..i-9]) and c < resistance and !bull        |
//+------------------------------------------------------------------+
int Rule_R3g_false_breakout(const float &feat[], const MqlRates &rb[])
{
   if(ArraySize(rb) < 10) return 0;
   double resistance = rb[2].high;
   for(int k = 3; k <= 9; k++)
      if(rb[k].high > resistance) resistance = rb[k].high;
   if(rb[0].high > resistance && rb[0].close < resistance && !IsBullish(rb, 0))
      return -1;
   return 0;
}

//+------------------------------------------------------------------+
//| R2h: Pin bar (sell) — long upper wick, small body, bearish       |
//| Python: wick_ratio_up > 0.6 and body_ratio < 0.25 and !bullish  |
//+------------------------------------------------------------------+
int Rule_R3h_pin_bar(const float &feat[], const MqlRates &rb[])
{
   if(WickRatioUp(rb, 0) > 0.6 && BarBodyRatio(rb, 0) < 0.25 && !IsBullish(rb, 0))
      return -1;
   return 0;
}

//+------------------------------------------------------------------+
//| R2i: Bearish engulfing (sell)                                     |
//| Python: bull[i-1] and !bull[i] and o[i]>=c[i-1] and c[i]<=o[i-1]|
//|         and body_ratio > 0.5                                      |
//+------------------------------------------------------------------+
int Rule_R3i_engulfing(const float &feat[], const MqlRates &rb[])
{
   if(ArraySize(rb) < 2) return 0;
   if(IsBullish(rb, 1) && !IsBullish(rb, 0) &&
      rb[0].open >= rb[1].close && rb[0].close <= rb[1].open &&
      BarBodyRatio(rb, 0) > 0.5)
      return -1;
   return 0;
}

//+------------------------------------------------------------------+
//| R2j: EMA12 rejection (sell)                                       |
//| Python: abs(h-ema12) < 0.3*atr and c < ema12 and !bull           |
//|         and dist_sma50 < 0                                        |
//+------------------------------------------------------------------+
int Rule_R3j_ema_reject(const float &feat[], const MqlRates &rb[])
{
   double atr = InlineATR14(rb, 0);
   if(atr < 1e-10) return 0;
   double ema12 = InlineEMA12(rb, 0);
   double dsma50 = (double)feat[FI_DIST_SMA50];
   if(MathAbs(rb[0].high - ema12) < 0.3 * atr && rb[0].close < ema12 &&
      !IsBullish(rb, 0) && dsma50 < 0)
      return -1;
   return 0;
}

//+------------------------------------------------------------------+
//| R2k: London open sell                                             |
//| Python: london_open and !bullish and dist_sma20 < 0 and sk > 0.6 |
//+------------------------------------------------------------------+
int Rule_R3k_london_sell(const float &feat[], const MqlRates &rb[])
{
   if(!IsLondonOpen(rb, 0)) return 0;
   double dsma20 = (double)feat[FI_DIST_SMA20];
   double sk = (double)feat[FI_STOCH_K];
   if(!IsBullish(rb, 0) && dsma20 < 0 && sk > 0.6)
      return -1;
   return 0;
}

//+------------------------------------------------------------------+
//| R2l: Tokyo fade (GBPJPY-specific, sell)                           |
//| is_tokyo(0-8h) AND !bullish AND rsi > 60 AND dist_sma50 < 0      |
//+------------------------------------------------------------------+
int Rule_R3l_tokyo_fade(const float &feat[], const MqlRates &rb[])
{
   if(!IsTokyo(rb, 0)) return 0;
   double rsi = (double)feat[FI_RSI14];
   double dsma50 = (double)feat[FI_DIST_SMA50];
   if(!IsBullish(rb, 0) && rsi > 60.0 && dsma50 < 0)
      return -1;
   return 0;
}

// ===================================================================
// C4 HIGHVOL RULES (both) — 10 rules
// ===================================================================

//+------------------------------------------------------------------+
//| R3a: V-reversal (buy/sell)                                        |
//| Buy: range_atr[i-1] > 1.5, !bull[i-1], bull[i], body > 0.5      |
//| Sell: range_atr[i-1] > 1.5, bull[i-1], !bull[i], body > 0.5     |
//+------------------------------------------------------------------+
int Rule_R4a_v_reversal(const float &feat[], const MqlRates &rb[])
{
   if(ArraySize(rb) < 3) return 0;
   double rng_atr1 = BarRangeATR(rb, 1);
   if(rng_atr1 <= 1.5) return 0;
   double br = BarBodyRatio(rb, 0);
   if(br <= 0.5) return 0;
   if(!IsBullish(rb, 1) && IsBullish(rb, 0)) return +1;
   if(IsBullish(rb, 1) && !IsBullish(rb, 0)) return -1;
   return 0;
}

//+------------------------------------------------------------------+
//| R3b: Momentum (buy/sell)                                          |
//| Buy: range_atr > 1.3, bullish, bb_pct > 0.75                     |
//| Sell: range_atr > 1.3, !bullish, bb_pct < 0.25                   |
//+------------------------------------------------------------------+
int Rule_R4b_momentum(const float &feat[], const MqlRates &rb[])
{
   double ra = (double)feat[FI_RANGE_ATR];
   double bb = (double)feat[FI_BB_PCT];
   if(ra > 1.3 && IsBullish(rb, 0) && bb > 0.75) return +1;
   if(ra > 1.3 && !IsBullish(rb, 0) && bb < 0.25) return -1;
   return 0;
}

//+------------------------------------------------------------------+
//| R3c: BB vol (buy/sell) — same as R1a_bb                           |
//+------------------------------------------------------------------+
int Rule_R4c_bb_vol(const float &feat[], const MqlRates &rb[])
{
   return Rule_R1a_bb(feat, rb);
}

//+------------------------------------------------------------------+
//| R3d: Stoch vol (buy/sell) — same as R1b_stoch                    |
//+------------------------------------------------------------------+
int Rule_R4d_stoch_vol(const float &feat[], const MqlRates &rb[])
{
   return Rule_R1b_stoch(feat, rb);
}

//+------------------------------------------------------------------+
//| R3e: Inside vol (buy/sell) — same as R2a_inside_break             |
//+------------------------------------------------------------------+
int Rule_R4e_inside_vol(const float &feat[], const MqlRates &rb[])
{
   return Rule_R2a_inside_break(feat, rb);
}

//+------------------------------------------------------------------+
//| R3f: Squeeze vol (buy/sell) — same as R2b_squeeze                 |
//+------------------------------------------------------------------+
int Rule_R4f_squeeze_vol(const float &feat[], const MqlRates &rb[])
{
   return Rule_R2b_squeeze(feat, rb);
}

//+------------------------------------------------------------------+
//| R3g: Pin bar vol (buy/sell) — no BB filter                        |
//| Buy: wick_lo > 0.6, body < 0.25                                  |
//| Sell: wick_up > 0.6, body < 0.25                                 |
//+------------------------------------------------------------------+
int Rule_R4g_pin_vol(const float &feat[], const MqlRates &rb[])
{
   if(WickRatioLo(rb, 0) > 0.6 && BarBodyRatio(rb, 0) < 0.25) return +1;
   if(WickRatioUp(rb, 0) > 0.6 && BarBodyRatio(rb, 0) < 0.25) return -1;
   return 0;
}

//+------------------------------------------------------------------+
//| R3h: Engulfing vol (buy/sell) — no BB filter                      |
//| Buy: bear[i-1], bull[i], engulf pattern                           |
//| Sell: bull[i-1], bear[i], engulf pattern                          |
//+------------------------------------------------------------------+
int Rule_R4h_engulf_vol(const float &feat[], const MqlRates &rb[])
{
   if(ArraySize(rb) < 2) return 0;
   if(!IsBullish(rb, 1) && IsBullish(rb, 0) &&
      rb[0].open <= rb[1].close && rb[0].close >= rb[1].open)
      return +1;
   if(IsBullish(rb, 1) && !IsBullish(rb, 0) &&
      rb[0].open >= rb[1].close && rb[0].close <= rb[1].open)
      return -1;
   return 0;
}

//+------------------------------------------------------------------+
//| R3i: Spike fade (buy/sell)                                        |
//| Buy: range_atr[i-1] > 2.0, !bull[i-1], bull[i], rsi < 30        |
//| Sell: range_atr[i-1] > 2.0, bull[i-1], !bull[i], rsi > 70       |
//+------------------------------------------------------------------+
int Rule_R4i_spike_fade(const float &feat[], const MqlRates &rb[])
{
   if(ArraySize(rb) < 2) return 0;
   double ra1 = BarRangeATR(rb, 1);
   double rsi = (double)feat[FI_RSI14];
   if(ra1 > 2.0 && !IsBullish(rb, 1) && IsBullish(rb, 0) && rsi < 30.0) return +1;
   if(ra1 > 2.0 && IsBullish(rb, 1) && !IsBullish(rb, 0) && rsi > 70.0) return -1;
   return 0;
}

//+------------------------------------------------------------------+
//| R3j: Session vol (buy/sell) — London open with vol (GBPJPY)       |
//| Buy: London open, range_atr > 1.2, bullish, stoch_k < 0.4       |
//| Sell: London open, range_atr > 1.2, !bullish, stoch_k > 0.6     |
//+------------------------------------------------------------------+
int Rule_R4j_session_vol(const float &feat[], const MqlRates &rb[])
{
   if(!IsLondonOpen(rb, 0)) return 0;
   double ra = (double)feat[FI_RANGE_ATR];
   double sk = (double)feat[FI_STOCH_K];
   if(ra > 1.2 && IsBullish(rb, 0) && sk < 0.4) return +1;
   if(ra > 1.2 && !IsBullish(rb, 0) && sk > 0.6) return -1;
   return 0;
}

//+------------------------------------------------------------------+
//| Master dispatch — all 48 rules                                    |
//+------------------------------------------------------------------+
int CheckRule(int rule_id, const float &feat[], const MqlRates &rb[])
{
   switch(rule_id)
   {
      case RULE_R0a_pullback:       return Rule_R0a_pullback(feat, rb);
      case RULE_R0b_higher_low:     return Rule_R0b_higher_low(feat, rb);
      case RULE_R0c_breakout_pb:    return Rule_R0c_breakout_pb(feat, rb);
      case RULE_R0d_oversold:       return Rule_R0d_oversold(feat, rb);
      case RULE_R0e_sma_bounce:     return Rule_R0e_sma_bounce(feat, rb);
      case RULE_R0f_false_breakdown:return Rule_R0f_false_breakdown(feat, rb);
      case RULE_R0g_close_streak:   return Rule_R0g_close_streak(feat, rb);
      case RULE_R0h_pin_bar:        return Rule_R0h_pin_bar(feat, rb);
      case RULE_R0i_engulfing:      return Rule_R0i_engulfing(feat, rb);
      case RULE_R0j_ema_pullback:   return Rule_R0j_ema_pullback(feat, rb);
      case RULE_R0k_tokyo_buy:      return Rule_R0k_tokyo_buy(feat, rb);
      case RULE_R0l_london_open_buy:return Rule_R0l_london_open_buy(feat, rb);
      case RULE_R1a_bb:             return Rule_R1a_bb(feat, rb);
      case RULE_R1b_stoch:          return Rule_R1b_stoch(feat, rb);
      case RULE_R2a_inside_break:   return Rule_R2a_inside_break(feat, rb);
      case RULE_R2b_squeeze:        return Rule_R2b_squeeze(feat, rb);
      case RULE_R1e_double_touch:   return Rule_R1e_double_touch(feat, rb);
      case RULE_R1f_mean_revert:    return Rule_R1f_mean_revert(feat, rb);
      case RULE_R1g_close_extreme:  return Rule_R1g_close_extreme(feat, rb);
      case RULE_R2c_nr4_break:      return Rule_R2c_nr4_break(feat, rb);
      case RULE_R1i_pin_bar:        return Rule_R1i_pin_bar(feat, rb);
      case RULE_R1j_engulfing:      return Rule_R1j_engulfing(feat, rb);
      case RULE_R2f_rsi_divergence: return Rule_R2f_rsi_divergence(feat, rb);
      case RULE_R2e_session_break:  return Rule_R2e_session_break(feat, rb);
      case RULE_R2d_3bar_reversal:  return Rule_R2d_3bar_reversal(feat, rb);
      case RULE_R1n_range_fade:     return Rule_R1n_range_fade(feat, rb);
      case RULE_R3a_swing_high:     return Rule_R3a_swing_high(feat, rb);
      case RULE_R3b_lower_high:     return Rule_R3b_lower_high(feat, rb);
      case RULE_R3c_bounce_fade:    return Rule_R3c_bounce_fade(feat, rb);
      case RULE_R3d_overbought:     return Rule_R3d_overbought(feat, rb);
      case RULE_R3e_sma_reject:     return Rule_R3e_sma_reject(feat, rb);
      case RULE_R3f_three_red:      return Rule_R3f_three_red(feat, rb);
      case RULE_R3g_false_breakout: return Rule_R3g_false_breakout(feat, rb);
      case RULE_R3h_pin_bar:        return Rule_R3h_pin_bar(feat, rb);
      case RULE_R3i_engulfing:      return Rule_R3i_engulfing(feat, rb);
      case RULE_R3j_ema_reject:     return Rule_R3j_ema_reject(feat, rb);
      case RULE_R3k_london_sell:    return Rule_R3k_london_sell(feat, rb);
      case RULE_R3l_tokyo_fade:     return Rule_R3l_tokyo_fade(feat, rb);
      case RULE_R4a_v_reversal:     return Rule_R4a_v_reversal(feat, rb);
      case RULE_R4b_momentum:       return Rule_R4b_momentum(feat, rb);
      case RULE_R4c_bb_vol:         return Rule_R4c_bb_vol(feat, rb);
      case RULE_R4d_stoch_vol:      return Rule_R4d_stoch_vol(feat, rb);
      case RULE_R4e_inside_vol:     return Rule_R4e_inside_vol(feat, rb);
      case RULE_R4f_squeeze_vol:    return Rule_R4f_squeeze_vol(feat, rb);
      case RULE_R4g_pin_vol:        return Rule_R4g_pin_vol(feat, rb);
      case RULE_R4h_engulf_vol:     return Rule_R4h_engulf_vol(feat, rb);
      case RULE_R4i_spike_fade:     return Rule_R4i_spike_fade(feat, rb);
      case RULE_R4j_session_vol:    return Rule_R4j_session_vol(feat, rb);
   }
   return 0;
}

#endif  // __SETUP_RULES_GBPJPY_MQH__
