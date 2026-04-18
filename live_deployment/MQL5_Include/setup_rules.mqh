//+------------------------------------------------------------------+
//| setup_rules.mqh  — K=5 rolling-window, 36 features              |
//| Auto-generated for XAUUSD.                                       |
//+------------------------------------------------------------------+
#ifndef __SETUP_RULES_MQH__
#define __SETUP_RULES_MQH__

enum RULE_ID
{
   RULE_C0_R3a_pullback = 0,  // C0 Uptrend
   RULE_C0_R3b_higherlow,  // C0 Uptrend
   RULE_C0_R3c_breakpullback,  // C0 Uptrend
   RULE_C0_R3d_oversold,  // C0 Uptrend
   RULE_C0_R3e_false_breakdown,  // C0 Uptrend
   RULE_C0_R3f_sma_bounce,  // C0 Uptrend
   RULE_C0_R3g_three_green,  // C0 Uptrend
   RULE_C0_R3h_close_streak,  // C0 Uptrend
   RULE_C0_R3i_inside_break,  // C0 Uptrend
   RULE_C1_R0a_bb,  // C1 MeanRevert
   RULE_C1_R0b_stoch,  // C1 MeanRevert [DIS]
   RULE_C1_R0c_doubletouch,  // C1 MeanRevert
   RULE_C1_R0f_mean_revert,  // C1 MeanRevert
   RULE_C1_R0i_close_extreme,  // C1 MeanRevert
   RULE_C2_R0d_squeeze,  // C2 TrendRange
   RULE_C2_R0e_nr4_break,  // C2 TrendRange
   RULE_C2_R0g_inside_break,  // C2 TrendRange
   RULE_C2_R0h_3bar_reversal,  // C2 TrendRange
   RULE_C3_R1a_swinghigh,  // C3 Downtrend
   RULE_C3_R1b_lowerhigh,  // C3 Downtrend
   RULE_C3_R1c_bouncefade,  // C3 Downtrend
   RULE_C3_R1d_overbought,  // C3 Downtrend
   RULE_C3_R1e_false_breakout,  // C3 Downtrend
   RULE_C3_R1f_sma_reject,  // C3 Downtrend
   RULE_C3_R1g_three_red,  // C3 Downtrend
   RULE_C3_R1h_close_streak,  // C3 Downtrend
   RULE_COUNT  // 26
};

const string RULE_NAMES[RULE_COUNT] = {"C0_R3a_pullback", "C0_R3b_higherlow", "C0_R3c_breakpullback", "C0_R3d_oversold", "C0_R3e_false_breakdown", "C0_R3f_sma_bounce", "C0_R3g_three_green", "C0_R3h_close_streak", "C0_R3i_inside_break", "C1_R0a_bb", "C1_R0b_stoch", "C1_R0c_doubletouch", "C1_R0f_mean_revert", "C1_R0i_close_extreme", "C2_R0d_squeeze", "C2_R0e_nr4_break", "C2_R0g_inside_break", "C2_R0h_3bar_reversal", "C3_R1a_swinghigh", "C3_R1b_lowerhigh", "C3_R1c_bouncefade", "C3_R1d_overbought", "C3_R1e_false_breakout", "C3_R1f_sma_reject", "C3_R1g_three_red", "C3_R1h_close_streak"};

const int RULE_CLUSTER[RULE_COUNT] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3};

const int RULE_COOLDOWN[RULE_COUNT] = {8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8};

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
#define FI_RSI14         21
#define FI_RSI6          22
#define FI_STOCH_K       23
#define FI_STOCH_D       24
#define FI_BB_PCT        25
#define FI_MOM5          26
#define FI_MOM10         27
#define FI_MOM20         28
#define FI_LL_DIST10     29
#define FI_HH_DIST10     30
#define FI_VOL_ACCEL     31
#define FI_ATR_RATIO     32
#define FI_SPREAD_NORM   33
#define FI_HOUR_ENC      34
#define FI_DOW_ENC       35



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
//| C0 RANGING — Bollinger mean reversion (R0a)                       |
//+------------------------------------------------------------------+
int Rule_R0a_bb(const float &feat[], const MqlRates &rb[])
{
   double bb   = feat[FI_BB_PCT];
   double rsi6 = feat[FI_RSI6];
   double o = rb[0].open, c = rb[0].close;
   if(bb <= 0.05 && rsi6 < -0.25 && c > o) return +1;
   if(bb >= 0.95 && rsi6 >  0.25 && c < o) return -1;
   return 0;
}

//+------------------------------------------------------------------+
//| C0 RANGING — Stochastic extreme reversal (R0b)                    |
//+------------------------------------------------------------------+
int Rule_R0b_stoch(const float &feat[], const MqlRates &rb[])
{
   double sk = feat[FI_STOCH_K];
   double sd = feat[FI_STOCH_D];
   double o = rb[0].open, c = rb[0].close;
   if(sk <= 0.10 && sk > sd && c > o) return +1;
   if(sk >= 0.90 && sk < sd && c < o) return -1;
   return 0;
}

//+------------------------------------------------------------------+
//| C0 RANGING — Double touch (R0c)                                   |
//+------------------------------------------------------------------+
int Rule_R0c_doubletouch(const float &feat[], const MqlRates &rb[])
{
   if(ArraySize(rb) < 35) return 0;
   double cur_lo = rb[0].low, cur_hi = rb[0].high;
   double prior_lo = rb[5].low, prior_hi = rb[5].high;
   for(int k = 6; k <= 30; k++)
   {
      if(rb[k].low  < prior_lo) prior_lo = rb[k].low;
      if(rb[k].high > prior_hi) prior_hi = rb[k].high;
   }
   double o = rb[0].open, c = rb[0].close;
   if(MathAbs(cur_lo - prior_lo) / MathMax(cur_lo, 1e-6) < 0.002 && c > o) return +1;
   if(MathAbs(cur_hi - prior_hi) / MathMax(cur_hi, 1e-6) < 0.002 && c < o) return -1;
   return 0;
}

//+------------------------------------------------------------------+
//| C0 RANGING — Volatility squeeze breakout (R0d)                    |
//+------------------------------------------------------------------+
int Rule_R0d_squeeze(const float &feat[], const MqlRates &rb[])
{
   if(ArraySize(rb) < 5) return 0;
   double atr_ratio = feat[FI_ATR_RATIO];
   if(atr_ratio > -0.15) return 0;
   double cur_rng = rb[0].high - rb[0].low;
   double prv_rng = rb[1].high - rb[1].low;
   if(cur_rng <= 0 || cur_rng < prv_rng * 1.5) return 0;
   double o = rb[0].open, c = rb[0].close;
   return (c > o) ? +1 : -1;
}

//+------------------------------------------------------------------+
//| C0 RANGING — NR4 breakout (R0e)                                   |
//+------------------------------------------------------------------+
int Rule_R0e_nr4_break(const float &feat[], const MqlRates &rb[])
{
   if(ArraySize(rb) < 5) return 0;
   // Python: ranges = highs[i-4:i] - lows[i-4:i]  → bars i-4..i-1
   //         nr4_range = ranges[-1] = bar i-1     → must equal min
   double r1 = rb[1].high - rb[1].low;   // i-1
   double r2 = rb[2].high - rb[2].low;   // i-2
   double r3 = rb[3].high - rb[3].low;   // i-3
   double r4 = rb[4].high - rb[4].low;   // i-4
   double mn = MathMin(MathMin(r1, r2), MathMin(r3, r4));
   if(r1 != mn) return 0;
   double cur_rng = rb[0].high - rb[0].low;
   if(cur_rng <= 0 || cur_rng < r1 * 1.8) return 0;
   return (rb[0].close > rb[0].open) ? +1 : -1;
}

//+------------------------------------------------------------------+
//| C0 RANGING — Mean reversion off 2-ATR stretch (R0f)               |
//| Needs ATR(14) — caller passes it via the feat[] context (we use   |
//| spread_norm × spread_pts → ATR back-out, OR re-compute here).     |
//| Simpler: re-compute ATR(14) inline.                               |
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

int Rule_R0f_mean_revert(const float &feat[], const MqlRates &rb[])
{
   if(ArraySize(rb) < 21) return 0;
   double sma20 = SMAClose(rb, 0, 20);
   double atr   = InlineATR14(rb, 0);
   if(atr <= 0) return 0;
   double dev = rb[0].close - sma20;
   double rng = rb[0].high - rb[0].low;
   if(rng <= 0) return 0;
   double close_pos = (rb[0].close - rb[0].low) / rng;
   if(dev >= 2.0 * atr)
   {
      if(close_pos < 0.45 && rb[0].close < rb[0].open) return -1;
   }
   else if(dev <= -2.0 * atr)
   {
      if(close_pos > 0.55 && rb[0].close > rb[0].open) return +1;
   }
   return 0;
}

//+------------------------------------------------------------------+
//| C0 RANGING — Inside bar breakout (R0g)                            |
//+------------------------------------------------------------------+
int Rule_R0g_inside_break(const float &feat[], const MqlRates &rb[])
{
   if(ArraySize(rb) < 3) return 0;
   // bar i-1 inside bar i-2
   if(!(rb[1].high <= rb[2].high && rb[1].low >= rb[2].low)) return 0;
   // current bar breaks the inside bar's high or low
   if(rb[0].close > rb[1].high && rb[0].close > rb[0].open) return +1;
   if(rb[0].close < rb[1].low  && rb[0].close < rb[0].open) return -1;
   return 0;
}

//+------------------------------------------------------------------+
//| C0 RANGING — Close at extreme + 10-bar extreme tagged (R0i)       |
//+------------------------------------------------------------------+
int Rule_R0i_close_extreme(const float &feat[], const MqlRates &rb[])
{
   if(ArraySize(rb) < 11) return 0;
   double rng = rb[0].high - rb[0].low;
   if(rng < 1e-6) return 0;
   double close_pos = (rb[0].close - rb[0].low) / rng;
   double prior_lo = rb[1].low, prior_hi = rb[1].high;
   for(int k = 2; k <= 10; k++)
   {
      if(rb[k].low  < prior_lo) prior_lo = rb[k].low;
      if(rb[k].high > prior_hi) prior_hi = rb[k].high;
   }
   if(close_pos > 0.90 && rb[0].low <= prior_lo) return +1;
   if(close_pos < 0.10 && rb[0].high >= prior_hi) return -1;
   return 0;
}

//+------------------------------------------------------------------+
//| C1 DOWNTREND — Swing-high rejection (R1a)                         |
//+------------------------------------------------------------------+
int Rule_R1a_swinghigh(const float &feat[], const MqlRates &rb[])
{
   if(ArraySize(rb) < 6) return 0;
   double hi_window = rb[0].high;
   for(int k = 1; k <= 5; k++) if(rb[k].high > hi_window) hi_window = rb[k].high;
   if(rb[0].high != hi_window) return 0;
   double rng = rb[0].high - rb[0].low;
   if(rng < 1e-6) return 0;
   double close_pos = (rb[0].close - rb[0].low) / rng;
   double hh_dist10 = feat[FI_HH_DIST10];
   if(close_pos < 0.5 && hh_dist10 < 0.5) return -1;
   return 0;
}

//+------------------------------------------------------------------+
//| C1 DOWNTREND — Lower-high continuation (R1b)                       |
//+------------------------------------------------------------------+
int Rule_R1b_lowerhigh(const float &feat[], const MqlRates &rb[])
{
   if(ArraySize(rb) < 35) return 0;
   double local_hi = rb[0].high;
   for(int k = 1; k <= 5; k++) if(rb[k].high > local_hi) local_hi = rb[k].high;
   if(rb[0].high != local_hi) return 0;
   double prior_hi = rb[5].high;
   for(int k = 6; k <= 30; k++) if(rb[k].high > prior_hi) prior_hi = rb[k].high;
   if(rb[0].high >= prior_hi) return 0;
   if(rb[0].close >= rb[0].open) return 0;
   return -1;
}

//+------------------------------------------------------------------+
//| C1 DOWNTREND — Bounce-into-resistance fade (R1c)                  |
//+------------------------------------------------------------------+
int Rule_R1c_bouncefade(const float &feat[], const MqlRates &rb[])
{
   if(ArraySize(rb) < 25) return 0;
   bool bounce = (rb[3].close > rb[3].open) &&
                 (rb[2].close > rb[2].open) &&
                 (rb[1].close > rb[1].open);
   if(!bounce) return 0;
   if(rb[0].close >= rb[0].open) return 0;
   double prior_hi = rb[3].high;
   for(int k = 4; k <= 20; k++) if(rb[k].high > prior_hi) prior_hi = rb[k].high;
   if(rb[0].high < prior_hi * 0.998) return 0;
   double mom5 = feat[FI_MOM5];
   if(mom5 <= 0) return 0;
   return -1;
}

//+------------------------------------------------------------------+
//| C1 DOWNTREND — Overbought stochastic short (R1d)                  |
//+------------------------------------------------------------------+
int Rule_R1d_overbought(const float &feat[], const MqlRates &rb[],
                        double prev_stoch_k)
{
   double sk = feat[FI_STOCH_K];
   if(prev_stoch_k < 0.80) return 0;
   if(sk >= prev_stoch_k) return 0;
   if(rb[0].close >= rb[0].open) return 0;
   return -1;
}

//+------------------------------------------------------------------+
//| C1 DOWNTREND — Failed breakout fade (R1e) — DISABLED at runtime   |
//| via threshold = 0.99 in meta. Detector kept for completeness.     |
//+------------------------------------------------------------------+
int Rule_R1e_false_breakout(const float &feat[], const MqlRates &rb[])
{
   if(ArraySize(rb) < 21) return 0;
   double prior_hi = rb[1].high;
   for(int k = 2; k <= 20; k++) if(rb[k].high > prior_hi) prior_hi = rb[k].high;
   if(rb[0].high <= prior_hi) return 0;
   if(rb[0].close >= prior_hi) return 0;
   return -1;
}

//+------------------------------------------------------------------+
//| C1 DOWNTREND — SMA20 rejection from below (R1f)                   |
//+------------------------------------------------------------------+
int Rule_R1f_sma_reject(const float &feat[], const MqlRates &rb[])
{
   if(ArraySize(rb) < 21) return 0;
   double mom20 = feat[FI_MOM20];
   if(mom20 >= 0) return 0;
   double sma20 = SMAClose(rb, 0, 20);
   if(rb[0].high < sma20) return 0;
   if(rb[0].close >= sma20) return 0;
   if(rb[0].close >= rb[0].open) return 0;
   return -1;
}

//+------------------------------------------------------------------+
//| C2 SHOCK — V-reversal (R2b)                                       |
//+------------------------------------------------------------------+
int Rule_R2b_v_reversal(const float &feat[], const MqlRates &rb[])
{
   if(ArraySize(rb) < 11) return 0;
   double rng = rb[0].high - rb[0].low;
   if(rng <= 0) return 0;
   double prior_lo = rb[1].low, prior_hi = rb[1].high;
   for(int k = 2; k <= 10; k++)
   {
      if(rb[k].low  < prior_lo) prior_lo = rb[k].low;
      if(rb[k].high > prior_hi) prior_hi = rb[k].high;
   }
   double close_pos = (rb[0].close - rb[0].low) / rng;
   if(rb[0].low  < prior_lo && close_pos > 0.60) return +1;
   if(rb[0].high > prior_hi && close_pos < 0.40) return -1;
   return 0;
}

//+------------------------------------------------------------------+
//| C3 UPTREND — Pullback bounce (R3a)                                |
//+------------------------------------------------------------------+
int Rule_R3a_pullback(const float &feat[], const MqlRates &rb[])
{
   if(ArraySize(rb) < 11) return 0;
   if(rb[0].close <= rb[0].open) return 0;
   double rng = rb[0].high - rb[0].low;
   if(rng < 1e-6) return 0;
   if((rb[0].close - rb[0].low) / rng < 0.5) return 0;
   if(rb[0].close >= rb[5].close || rb[5].close >= rb[10].close) return 0;
   double lldist = feat[FI_LL_DIST10];
   double rsi6   = feat[FI_RSI6];
   if(lldist >= 0.8 || rsi6 >= 0.15) return 0;
   return +1;
}

//+------------------------------------------------------------------+
//| C3 UPTREND — Higher-low continuation (R3b)                        |
//+------------------------------------------------------------------+
int Rule_R3b_higherlow(const float &feat[], const MqlRates &rb[])
{
   if(ArraySize(rb) < 35) return 0;
   double local_lo = rb[0].low;
   for(int k = 1; k <= 5; k++) if(rb[k].low < local_lo) local_lo = rb[k].low;
   if(rb[0].low != local_lo) return 0;
   double prior_lo = rb[5].low;
   for(int k = 6; k <= 30; k++) if(rb[k].low < prior_lo) prior_lo = rb[k].low;
   if(rb[0].low <= prior_lo) return 0;
   if(rb[0].close <= rb[0].open) return 0;
   return +1;
}

//+------------------------------------------------------------------+
//| C3 UPTREND — Breakout pullback (R3c)                              |
//+------------------------------------------------------------------+
int Rule_R3c_breakpullback(const float &feat[], const MqlRates &rb[])
{
   if(ArraySize(rb) < 35) return 0;
   bool broke = false;
   double break_level = 0.0;
   for(int j = 10; j >= 2; j--)
   {
      double prior_hi = rb[j+1].high;
      for(int k = j+2; k <= j+20; k++) if(rb[k].high > prior_hi) prior_hi = rb[k].high;
      if(rb[j].close > prior_hi)
      {
         broke = true;
         break_level = prior_hi;
         break;
      }
   }
   if(!broke) return 0;
   if(rb[0].low > break_level * 1.003) return 0;
   if(rb[0].low < break_level * 0.997) return 0;
   if(rb[0].close <= rb[0].open) return 0;
   return +1;
}

//+------------------------------------------------------------------+
//| C3 UPTREND — Oversold bounce (R3d)                                |
//+------------------------------------------------------------------+
int Rule_R3d_oversold(const float &feat[], const MqlRates &rb[],
                      double prev_stoch_k)
{
   double sk = feat[FI_STOCH_K];
   if(prev_stoch_k > 0.20) return 0;
   if(sk <= prev_stoch_k) return 0;
   if(rb[0].close <= rb[0].open) return 0;
   double mom20 = feat[FI_MOM20];
   if(mom20 <= 0) return 0;
   return +1;
}

//+------------------------------------------------------------------+
//| C3 UPTREND — False breakdown reversal (R3e)                       |
//+------------------------------------------------------------------+
int Rule_R3e_false_breakdown(const float &feat[], const MqlRates &rb[])
{
   if(ArraySize(rb) < 21) return 0;
   double prior_lo = rb[1].low;
   for(int k = 2; k <= 20; k++) if(rb[k].low < prior_lo) prior_lo = rb[k].low;
   if(rb[0].low >= prior_lo) return 0;        // must dip below
   if(rb[0].close <= prior_lo) return 0;       // must close back above
   double rng = rb[0].high - rb[0].low;
   if(rng < 1e-6) return 0;
   double midpoint = rb[0].low + (rb[0].close - rb[0].low) * 0.5;
   if(rb[0].close <= midpoint) return 0;       // closed in top half (Python check)
   return +1;
}

//+------------------------------------------------------------------+
//| C3 UPTREND — SMA20 bounce from above (R3f)                         |
//+------------------------------------------------------------------+
int Rule_R3f_sma_bounce(const float &feat[], const MqlRates &rb[])
{
   if(ArraySize(rb) < 21) return 0;
   double mom20 = feat[FI_MOM20];
   if(mom20 <= 0) return 0;
   double sma20 = SMAClose(rb, 0, 20);
   if(rb[0].low > sma20) return 0;            // didn't tag from above
   if(rb[0].close <= sma20) return 0;         // didn't reclaim
   if(rb[0].close <= rb[0].open) return 0;    // bullish bar required
   return +1;
}

//+------------------------------------------------------------------+
//| C0 — 3-bar reversal (R0h) — DISABLED but detector kept           |
//+------------------------------------------------------------------+
int Rule_R0h_3bar_reversal(const float &feat[], const MqlRates &rb[])
{
   if(ArraySize(rb) < 3) return 0;
   double rng = rb[0].high - rb[0].low;
   if(rng < 1e-6) return 0;
   bool p1r = rb[1].close < rb[1].open;
   bool p2r = rb[2].close < rb[2].open;
   bool p1g = rb[1].close > rb[1].open;
   bool p2g = rb[2].close > rb[2].open;
   if(p1r && p2r && rb[0].close > rb[0].open && (rb[0].close-rb[0].low)/rng > 0.6) return +1;
   if(p1g && p2g && rb[0].close < rb[0].open && (rb[0].close-rb[0].low)/rng < 0.4) return -1;
   return 0;
}

//+------------------------------------------------------------------+
//| C1 — 3 red bars continuation (R1g)                                |
//+------------------------------------------------------------------+
int Rule_R1g_three_red(const float &feat[], const MqlRates &rb[])
{
   if(ArraySize(rb) < 3) return 0;
   if(!(rb[0].close<rb[0].open && rb[1].close<rb[1].open && rb[2].close<rb[2].open)) return 0;
   double mom20 = feat[FI_MOM20];
   if(mom20 >= 0) return 0;
   double rng = rb[0].high - rb[0].low;
   if(rng < 1e-6) return 0;
   if((rb[0].close - rb[0].low) / rng > 0.4) return 0;
   return -1;
}

//+------------------------------------------------------------------+
//| C1 — Lower close streak (R1h) — DISABLED                         |
//+------------------------------------------------------------------+
int Rule_R1h_close_streak(const float &feat[], const MqlRates &rb[])
{
   if(ArraySize(rb) < 5) return 0;
   if(!(rb[0].close<rb[1].close && rb[1].close<rb[2].close &&
        rb[2].close<rb[3].close && rb[3].close<rb[4].close)) return 0;
   double mom10 = feat[FI_MOM10];
   if(mom10 >= 0) return 0;
   if(rb[0].close >= rb[0].open) return 0;
   return -1;
}

//+------------------------------------------------------------------+
//| C2 — Momentum continuation (R2d) — DISABLED                      |
//+------------------------------------------------------------------+
int Rule_R2d_continuation(const float &feat[], const MqlRates &rb[])
{
   if(ArraySize(rb) < 2) return 0;
   double atr = InlineATR14(rb, 0);
   if(atr <= 0) return 0;
   double prev_rng = rb[1].high - rb[1].low;
   double prev_atr = InlineATR14(rb, 1);
   if(prev_atr <= 0 || prev_rng < 1.3 * prev_atr) return 0;
   int prev_dir = (rb[1].close > rb[1].open) ? +1 : -1;
   int cur_dir  = (rb[0].close > rb[0].open) ? +1 : -1;
   if(cur_dir != prev_dir) return 0;
   double rng = rb[0].high - rb[0].low;
   if(rng < 1e-6) return 0;
   double cp = (rb[0].close - rb[0].low) / rng;
   if(cur_dir == +1 && cp > 0.55) return +1;
   if(cur_dir == -1 && cp < 0.45) return -1;
   return 0;
}

//+------------------------------------------------------------------+
//| C3 — 3 green bars continuation (R3g) — DISABLED                   |
//+------------------------------------------------------------------+
int Rule_R3g_three_green(const float &feat[], const MqlRates &rb[])
{
   if(ArraySize(rb) < 3) return 0;
   if(!(rb[0].close>rb[0].open && rb[1].close>rb[1].open && rb[2].close>rb[2].open)) return 0;
   double mom20 = feat[FI_MOM20];
   if(mom20 <= 0) return 0;
   double rng = rb[0].high - rb[0].low;
   if(rng < 1e-6) return 0;
   if((rb[0].close - rb[0].low) / rng < 0.6) return 0;
   return +1;
}

//+------------------------------------------------------------------+
//| C3 — Higher close streak (R3h)                                    |
//+------------------------------------------------------------------+
int Rule_R3h_close_streak(const float &feat[], const MqlRates &rb[])
{
   if(ArraySize(rb) < 5) return 0;
   if(!(rb[0].close>rb[1].close && rb[1].close>rb[2].close &&
        rb[2].close>rb[3].close && rb[3].close>rb[4].close)) return 0;
   double mom10 = feat[FI_MOM10];
   if(mom10 <= 0) return 0;
   if(rb[0].close <= rb[0].open) return 0;
   return +1;
}

//+------------------------------------------------------------------+
//| C3 — Inside bar breakout (R3i)                                    |
//+------------------------------------------------------------------+
int Rule_R3i_inside_break(const float &feat[], const MqlRates &rb[])
{
   if(ArraySize(rb) < 3) return 0;
   if(!(rb[1].high <= rb[2].high && rb[1].low >= rb[2].low)) return 0;
   if(rb[0].close <= rb[1].high) return 0;
   if(rb[0].close <= rb[0].open) return 0;
   double mom20 = feat[FI_MOM20];
   if(mom20 <= 0) return 0;
   return +1;
}

//+------------------------------------------------------------------+
//| Master dispatch — all 28 rules                                    |
//+------------------------------------------------------------------+


int CheckRule(int rule_id, const float &feat[], const MqlRates &rb[], double prev_stoch_k=0.0)
{
   switch(rule_id)
   {
      case RULE_C0_R3a_pullback: return Rule_R3a_pullback(feat, rb);
      case RULE_C0_R3b_higherlow: return Rule_R3b_higherlow(feat, rb);
      case RULE_C0_R3c_breakpullback: return Rule_R3c_breakpullback(feat, rb);
      case RULE_C0_R3d_oversold: return Rule_R3d_oversold(feat, rb, prev_stoch_k);
      case RULE_C0_R3e_false_breakdown: return Rule_R3e_false_breakdown(feat, rb);
      case RULE_C0_R3f_sma_bounce: return Rule_R3f_sma_bounce(feat, rb);
      case RULE_C0_R3g_three_green: return Rule_R3g_three_green(feat, rb);
      case RULE_C0_R3h_close_streak: return Rule_R3h_close_streak(feat, rb);
      case RULE_C0_R3i_inside_break: return Rule_R3i_inside_break(feat, rb);
      case RULE_C1_R0a_bb: return Rule_R0a_bb(feat, rb);
      case RULE_C1_R0b_stoch: return Rule_R0b_stoch(feat, rb);
      case RULE_C1_R0c_doubletouch: return Rule_R0c_doubletouch(feat, rb);
      case RULE_C1_R0f_mean_revert: return Rule_R0f_mean_revert(feat, rb);
      case RULE_C1_R0i_close_extreme: return Rule_R0i_close_extreme(feat, rb);
      case RULE_C2_R0d_squeeze: return Rule_R0d_squeeze(feat, rb);
      case RULE_C2_R0e_nr4_break: return Rule_R0e_nr4_break(feat, rb);
      case RULE_C2_R0g_inside_break: return Rule_R0g_inside_break(feat, rb);
      case RULE_C2_R0h_3bar_reversal: return Rule_R0h_3bar_reversal(feat, rb);
      case RULE_C3_R1a_swinghigh: return Rule_R1a_swinghigh(feat, rb);
      case RULE_C3_R1b_lowerhigh: return Rule_R1b_lowerhigh(feat, rb);
      case RULE_C3_R1c_bouncefade: return Rule_R1c_bouncefade(feat, rb);
      case RULE_C3_R1d_overbought: return Rule_R1d_overbought(feat, rb, prev_stoch_k);
      case RULE_C3_R1e_false_breakout: return Rule_R1e_false_breakout(feat, rb);
      case RULE_C3_R1f_sma_reject: return Rule_R1f_sma_reject(feat, rb);
      case RULE_C3_R1g_three_red: return Rule_R1g_three_red(feat, rb);
      case RULE_C3_R1h_close_streak: return Rule_R1h_close_streak(feat, rb);
   }
   return 0;
}

#endif  // __SETUP_RULES_MQH__