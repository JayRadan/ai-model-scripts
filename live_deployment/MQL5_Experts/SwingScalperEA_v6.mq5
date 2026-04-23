//+------------------------------------------------------------------+
//|  SwingScalperEA_v6.mq5                                            |
//|  XAUUSD M5 — Physics + Quantum Flow + ML Exit                        |
//|                                                                    |
//|  Architecture:                                                     |
//|   1. Detect weekly regime → cluster (C0/C1/C2/C3)                 |
//|   2. C2 Shock weeks → never trade                                  |
//|   3. For other clusters: scan all rules belonging to that cluster |
//|      on the just-closed bar                                        |
//|   4. For each fired rule, run its ONNX confirmation classifier;   |
//|      take the trade only if probability ≥ rule's threshold        |
//|                                                                    |
//|   12 hand-coded setup rules (4 per tradeable cluster)             |
//|   12 ONNX classifiers (1 per rule)                                |
//|   1  regime selector (constants in regime_selector.mqh)            |
//|                                                                    |
//|   SL = SL_ATR_MULT × ATR(14)                                      |
//|   TP = TP_ATR_MULT × ATR(14)                                      |
//+------------------------------------------------------------------+
#property copyright "EdgePredictor"
#property version   "6.00"
#property description "AI-powered XAUUSD rule + ML confirmation trading system."
#property description "Requires license key. Allow WebRequest for your server URL in MT5 settings."
#property strict

#include <Trade/Trade.mqh>
#include <regime_selector.mqh>
#include <confirmation_router_v6.mqh>
#include <regime_v4_sizer.mqh>


//--- inputs
input group "=== License ==="
input string InpLicenseKey  = "";                                        // License Key (EP-account-signature)
input string InpServerURL   = "https://edge-predictor.onrender.com";     // License Server URL

input group "=== Trading ==="
input int    InpRegimeRefreshBars  = 288;   // refresh regime every N M5 bars (288 = 1 day)
input double InpLots               = 0.01;
input bool   InpAllowLong          = true;
input bool   InpAllowShort         = true;
input int    InpMaxSpread          = 80;    // skip entry if spread > N points
input long   InpMagic              = 420305;
input int    InpMaxDevPoints       = 30;

input group "=== Debug ==="
input bool   InpVerbose            = true;

//--- constants
static const int    FEATURE_DIM    = 14;     // v6 physics features
static const int    EXIT_FEAT_DIM  = 11;
static const int    RULE_FEAT_DIM  = 36;     // old features for rule detection
static const int    LOOKBACK       = 220;
static const int    MIN_HOLD_BARS  = 2;
static const double EXIT_THRESHOLD = 0.55;
static const double SL_HARD_ATR    = 4.0;    // safety stop only
static const ENUM_TIMEFRAMES MODEL_TF = PERIOD_M5;

long g_exit_handle = INVALID_HANDLE;
double g_entry_price = 0, g_entry_atr = 0;
int g_bars_held = 0, g_entry_dir = 0;
double g_vwap_cum_tv = 0, g_vwap_cum_v = 0;
int g_vwap_day = -1;

#define LICENSE_STATUS_INVALID -1
#define LICENSE_STATUS_ACTIVE   1
#define LICENSE_STATUS_GRACE    2
#define LICENSE_GRACE_SECONDS   3600
#define LICENSE_RECHECK_BARS    288

CTrade   g_trade;
datetime g_last_bar    = 0;
int      g_bars_since_refresh = 0;
int      g_last_regime_day   = -1;
int      g_rule_cooldown[RULE_COUNT];
double   g_prev_stoch_k = 0.5;
bool     g_licensed         = false;
datetime g_last_license_ok  = 0;
int      g_bars_since_lic   = 0;
int      g_bars_since_report = 0;
int      g_uptime_bars      = 0;
int      g_total_trades     = 0;
int      g_wins             = 0;
double   g_total_pnl        = 0;
double   g_max_dd           = 0;
string   g_last_signal      = "—";

#define REPORT_INTERVAL_BARS 12   // report stats every ~1h (12 x M5)

//+------------------------------------------------------------------+
//| Chart styling, dashboard, watermark, stats reporting              |
//+------------------------------------------------------------------+
void SetupChart()
{
   long chart = ChartID();
   if(_Period != PERIOD_M5)
      ChartSetSymbolPeriod(chart, _Symbol, PERIOD_M5);
   ChartSetInteger(chart, CHART_COLOR_BACKGROUND,    clrBlack);
   ChartSetInteger(chart, CHART_COLOR_FOREGROUND,     0x5A5A5A);
   ChartSetInteger(chart, CHART_COLOR_GRID,           0x1A1A1A);
   ChartSetInteger(chart, CHART_COLOR_CHART_UP,       0x10B981);
   ChartSetInteger(chart, CHART_COLOR_CHART_DOWN,     0xEF4444);
   ChartSetInteger(chart, CHART_COLOR_CANDLE_BULL,    0x10B981);
   ChartSetInteger(chart, CHART_COLOR_CANDLE_BEAR,    0xEF4444);
   ChartSetInteger(chart, CHART_COLOR_CHART_LINE,     0x5A7080);
   ChartSetInteger(chart, CHART_COLOR_VOLUME,         0x333333);
   ChartSetInteger(chart, CHART_COLOR_ASK,            0xF5C518);
   ChartSetInteger(chart, CHART_COLOR_BID,            0x3B82F6);
   ChartSetInteger(chart, CHART_COLOR_STOP_LEVEL,     0xEF4444);
   ChartSetInteger(chart, CHART_SHOW_GRID,            false);
   ChartSetInteger(chart, CHART_SHOW_VOLUMES,         false);
   ChartSetInteger(chart, CHART_MODE,                 CHART_CANDLES);
   ChartSetInteger(chart, CHART_AUTOSCROLL,           true);
   ChartSetInteger(chart, CHART_SHIFT,                true);
   ChartSetInteger(chart, CHART_SHOW_TRADE_LEVELS,    true);
   ChartRedraw(chart);
}

void DrawWatermark()
{
   long chart = ChartID();
   string name = "EP_Watermark";
   if(ObjectFind(chart, name) < 0)
      ObjectCreate(chart, name, OBJ_LABEL, 0, 0, 0);
   ObjectSetInteger(chart, name, OBJPROP_CORNER,    CORNER_RIGHT_LOWER);
   ObjectSetInteger(chart, name, OBJPROP_XDISTANCE, 15);
   ObjectSetInteger(chart, name, OBJPROP_YDISTANCE, 15);
   ObjectSetInteger(chart, name, OBJPROP_ANCHOR,    ANCHOR_RIGHT_LOWER);
   ObjectSetString (chart, name, OBJPROP_TEXT,       "EdgePredictor Midas  |  Jay  |  edgepredictor.pro");
   ObjectSetString (chart, name, OBJPROP_FONT,       "Consolas");
   ObjectSetInteger(chart, name, OBJPROP_FONTSIZE,   8);
   ObjectSetInteger(chart, name, OBJPROP_COLOR,      0x444444);
   ObjectSetInteger(chart, name, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(chart, name, OBJPROP_HIDDEN,     true);
}

void UpdateDashboard()
{
   double wr = (g_total_trades > 0) ? 100.0 * g_wins / g_total_trades : 0;
   string dash = "";
   dash += "═══════════════════════════════════════\n";
   dash += "   EdgePredictor Midas  v6.0\n";
   dash += "   XAUUSD · M5 · Rule + ML Confirmation\n";
   dash += "═══════════════════════════════════════\n";
   dash += "\n";
   dash += StringFormat("   Regime:     C%d %s\n", g_active_cluster, REGIME_NAMES[g_active_cluster]);
   dash += StringFormat("   License:    %s\n", g_licensed ? "ACTIVE" : "INACTIVE");
   dash += StringFormat("   Models:     %d loaded\n", RULE_COUNT);
   dash += "\n";
   dash += StringFormat("   Trades:     %d\n", g_total_trades);
   dash += StringFormat("   Win Rate:   %.1f%%\n", wr);
   dash += StringFormat("   Net P&L:    $%.2f\n", g_total_pnl);
   dash += StringFormat("   Last Signal: %s\n", g_last_signal);
   dash += "\n";
   dash += StringFormat("   Spread:     %d pts\n", (int)SymbolInfoInteger(_Symbol, SYMBOL_SPREAD));
   dash += "═══════════════════════════════════════\n";
   Comment(dash);
}

void RefreshTradeStats()
{
   g_total_trades = 0; g_wins = 0; g_total_pnl = 0; g_max_dd = 0;
   if(!HistorySelect(0, TimeCurrent())) return;
   int total = HistoryDealsTotal();
   double peak = 0, running = 0;
   for(int i = 0; i < total; i++)
   {
      ulong ticket = HistoryDealGetTicket(i);
      if(ticket == 0) continue;
      if(HistoryDealGetInteger(ticket, DEAL_MAGIC) != InpMagic) continue;
      if(HistoryDealGetString(ticket, DEAL_SYMBOL) != _Symbol) continue;
      if(HistoryDealGetInteger(ticket, DEAL_ENTRY) != DEAL_ENTRY_OUT) continue;
      double profit = HistoryDealGetDouble(ticket, DEAL_PROFIT)
                    + HistoryDealGetDouble(ticket, DEAL_COMMISSION)
                    + HistoryDealGetDouble(ticket, DEAL_SWAP);
      g_total_trades++;
      g_total_pnl += profit;
      if(profit > 0) g_wins++;
      running += profit;
      if(running > peak) peak = running;
      double dd = running - peak;
      if(dd < g_max_dd) g_max_dd = dd;
   }
}

void ReportStats()
{
   double wr = (g_total_trades > 0) ? 100.0 * g_wins / g_total_trades : 0;
   double bal = AccountInfoDouble(ACCOUNT_BALANCE);
   double eq  = AccountInfoDouble(ACCOUNT_EQUITY);
   string json = StringFormat(
      "{\"ea_name\":\"Midas\",\"account\":\"%s\",\"symbol\":\"%s\","
      "\"total_trades\":%d,\"wins\":%d,\"losses\":%d,"
      "\"total_pnl\":%.2f,\"win_rate\":%.2f,\"max_dd\":%.2f,"
      "\"balance\":%.2f,\"equity\":%.2f,"
      "\"regime\":\"C%d %s\",\"active_rules\":%d,"
      "\"lot_size\":%.2f,\"uptime_bars\":%d}",
      IntegerToString(AccountInfoInteger(ACCOUNT_LOGIN)), _Symbol,
      g_total_trades, g_wins, g_total_trades - g_wins,
      g_total_pnl, wr, g_max_dd, bal, eq,
      g_active_cluster, REGIME_NAMES[g_active_cluster],
      RULE_COUNT, InpLots, g_uptime_bars
   );
   string url = InpServerURL + "/report-stats?key=" + InpLicenseKey;
   char post[];
   StringToCharArray(json, post, 0, WHOLE_ARRAY, CP_UTF8);
   ArrayResize(post, ArraySize(post) - 1);
   char result[];
   string headers = "Content-Type: application/json\r\n";
   string resp_headers;
   ResetLastError();
   int res = WebRequest("POST", url, headers, 10000, post, result, resp_headers);
   if(res == 200) Print("Stats reported to server");
   else if(InpVerbose) PrintFormat("Stats report HTTP %d (err=%d)", res, GetLastError());
}

//+------------------------------------------------------------------+
// Utility
//+------------------------------------------------------------------+
double _sd(double n, double d) { return (MathAbs(d) < 1e-12) ? 0.0 : n / d; }
double _sg(double x)           { return x > 0.0 ? 1.0 : x < 0.0 ? -1.0 : 0.0; }
double _sq(double x)           { return x * x; }

//+------------------------------------------------------------------+
double RawATR14(const MqlRates &rb[], int idx)
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

double LRSlope5(const MqlRates &rb[], int idx)
{
   const int N = 5;
   if(idx + N >= ArraySize(rb)) return 0.0;
   double sX=0,sY=0,sXY=0,sX2=0;
   for(int k=0;k<N;k++)
   {
      double x=(double)k, y=rb[idx+N-1-k].close;
      sX+=x; sY+=y; sXY+=x*y; sX2+=x*x;
   }
   return _sd(N*sXY-sX*sY, N*sX2-sX*sX);
}

double VolAvg5(const MqlRates &rb[], int idx)
{
   double s=0;
   for(int k=idx;k<idx+5&&k<ArraySize(rb);k++) s+=(double)rb[k].tick_volume;
   return s/5.0;
}

double AvgAbsWork5(const MqlRates &rb[], int idx)
{
   double s=0;
   for(int k=idx;k<idx+5&&k+1<ArraySize(rb);k++)
      s+=MathAbs((rb[k].close-rb[k+1].close)*((double)rb[k].tick_volume+(double)rb[k+1].tick_volume)*0.5);
   return s/5.0;
}

double SimpleRSI(const MqlRates &rb[], int idx, int period)
{
   if(idx + period >= ArraySize(rb)) return 0.0;
   double gains=0, losses=0;
   for(int k=idx;k<idx+period;k++)
   {
      double d = rb[k].close - rb[k+1].close;
      if(d > 0) gains  += d;
      else      losses -= d;
   }
   double ag = gains  / period;
   double al = losses / period;
   if(al < 1e-10) return 1.0 - 0.5;
   return (100.0 - 100.0/(1.0 + ag/al)) / 100.0 - 0.5;
}

double StochK(const MqlRates &rb[], int idx, int period)
{
   if(idx + period >= ArraySize(rb)) return 0.5;
   double lo=rb[idx].low, hi=rb[idx].high;
   for(int k=idx;k<idx+period;k++)
   {
      lo = MathMin(lo, rb[k].low);
      hi = MathMax(hi, rb[k].high);
   }
   return _sd(rb[idx].close - lo, hi - lo);
}

double BollingerPctB(const MqlRates &rb[], int idx, int period=20)
{
   if(idx + period >= ArraySize(rb)) return 0.5;
   double sum=0;
   for(int k=idx;k<idx+period;k++) sum+=rb[k].close;
   double sma=sum/period;
   double var=0;
   for(int k=idx;k<idx+period;k++) var+=_sq(rb[k].close-sma);
   double std_=MathSqrt(var/period);
   double upper=sma+2.0*std_, lower=sma-2.0*std_;
   return _sd(rb[idx].close - lower, upper - lower);
}

double ATRRatio(const MqlRates &rb[], int idx)
{
   if(idx + 64 >= ArraySize(rb)) return 0.0;
   double cur = RawATR14(rb, idx);
   double sum = 0;
   for(int k=idx;k<idx+50;k++) sum += RawATR14(rb,k);
   double sma = sum / 50.0;
   return _sd(cur - sma, sma);
}

double VolAccel(const MqlRates &rb[], int idx)
{
   if(idx + 22 >= ArraySize(rb)) return 0.0;
   double s3=0,s20=0;
   for(int k=idx;k<idx+3;k++)  s3  += (double)rb[k].tick_volume;
   for(int k=idx;k<idx+20;k++) s20 += (double)rb[k].tick_volume;
   return _sd(s3/3.0 - s20/20.0, s20/20.0);
}

//+------------------------------------------------------------------+
//| Build 36-feature vector  (identical to v4)                        |
//+------------------------------------------------------------------+
bool BuildRuleFeatures(const MqlRates &rb[], float &feat[])
{
   if(ArraySize(rb) < LOOKBACK) return false;
   ArrayResize(feat, RULE_FEAT_DIM);

   int i=0;
   double o=rb[i].open, h=rb[i].high, l=rb[i].low, c=rb[i].close;
   double v=(double)rb[i].tick_volume;
   double range=h-l, body=MathAbs(c-o);
   double uwick=h-MathMax(o,c), lwick=MathMin(o,c)-l;
   double atr=RawATR14(rb,i);
   if(atr<1e-10) atr=1e-10;

   double f01=_sd(c-l,range);
   double f02=_sd(uwick-lwick,range);
   double f03=_sd(body,range);
   double f04=(c>=o) ? f03*f01 : -(f03*(1.0-f01));
   double f05=_sd(_sq(uwick)-_sq(lwick),_sq(range));
   double f06=_sd(LRSlope5(rb,i),atr);
   double pr4=0; for(int k=1;k<=4;k++) pr4+=rb[i+k].high-rb[i+k].low; pr4/=4.0;
   double f07=_sd(range,pr4);
   double f08=0; for(int k=1;k<=4;k++) f08+=_sg(rb[i+k-1].close-rb[i+k].close)/(double)k;
   double hi4=h,lo4=l; for(int k=1;k<=4;k++){hi4=MathMax(hi4,rb[i+k].high);lo4=MathMin(lo4,rb[i+k].low);}
   double f09=_sd(c-lo4,hi4-lo4);
   double so=_sg(c-o), sp=_sg(rb[i+1].close-rb[i+1].open);
   double oppDir=(so!=0&&sp!=0&&so!=sp)?so:0.0;
   double f10=_sd(range,rb[i+1].high-rb[i+1].low)*oppDir;
   double f11=_sg(c-o)*0.5*_sd(_sq(range),_sq(atr));
   double f12=_sd(c*v,rb[i+1].close*(double)rb[i+1].tick_volume)-1.0;
   double rawW=(c-rb[i+1].close)*(v+(double)rb[i+1].tick_volume)*0.5;
   double avgW=AvgAbsWork5(rb,i);
   double f13=_sd(rawW,avgW);
   double prevW=(rb[i+1].close-rb[i+2].close)*((double)rb[i+1].tick_volume+(double)rb[i+2].tick_volume)*0.5;
   double f14=_sd(MathAbs(rawW-prevW),avgW);
   double f15=_sd(2.0*c-h-l,range);
   double maxH3=MathMax(rb[i+1].high,MathMax(rb[i+2].high,rb[i+3].high));
   double minL3=MathMin(rb[i+1].low, MathMin(rb[i+2].low, rb[i+3].low));
   double f16h=_sd(h-maxH3,atr);
   double f16l=_sd(minL3-l,atr);
   double hi5=h,lo5=l; for(int k=1;k<=4;k++){hi5=MathMax(hi5,rb[i+k].high);lo5=MathMin(lo5,rb[i+k].low);}
   double f17=_sd(range,hi5-lo5);
   double f18=_sd(MathAbs(c-rb[i+1].close),MathAbs(rb[i+1].close-rb[i+2].close));
   double f19=_sd(uwick+lwick,body);
   double f20=_sd(c-o,range)*_sd(v,VolAvg5(rb,i));

   feat[0]=(float)f01; feat[1]=(float)f02; feat[2]=(float)f03;
   feat[3]=(float)f04; feat[4]=(float)f05; feat[5]=(float)f06;
   feat[6]=(float)f07; feat[7]=(float)f08; feat[8]=(float)f09;
   feat[9]=(float)f10; feat[10]=(float)f11;feat[11]=(float)f12;
   feat[12]=(float)f13;feat[13]=(float)f14;feat[14]=(float)f15;
   feat[15]=(float)f16h;feat[16]=(float)f16l;feat[17]=(float)f17;
   feat[18]=(float)f18;feat[19]=(float)f19;feat[20]=(float)f20;

   feat[21]=(float)SimpleRSI(rb,i,14);
   feat[22]=(float)SimpleRSI(rb,i,6);

   double sk=StochK(rb,i,5);
   feat[23]=(float)sk;
   double sk1=StochK(rb,i+1,5), sk2=StochK(rb,i+2,5);
   feat[24]=(float)((sk+sk1+sk2)/3.0);

   feat[25]=(float)BollingerPctB(rb,i,20);

   feat[26]=(float)(_sd(c - rb[i+5].close,  atr));
   feat[27]=(float)(_sd(c - rb[i+10].close, atr));
   feat[28]=(float)(_sd(c - rb[i+20].close, atr));

   double lo10=rb[i+1].low;
   for(int k=2;k<=10;k++) lo10=MathMin(lo10,rb[i+k].low);
   feat[29]=(float)(_sd(c-lo10,atr));

   double hi10=rb[i+1].high;
   for(int k=2;k<=10;k++) hi10=MathMax(hi10,rb[i+k].high);
   feat[30]=(float)(_sd(hi10-c,atr));

   feat[31]=(float)VolAccel(rb,i);
   feat[32]=(float)ATRRatio(rb,i);

   // spread_norm: Python training uses RAW POINTS / atr (see 01_labeler_v4.py:165),
   // so we must NOT convert to price here — pass spr_pts as-is.
   long spr_pts = SymbolInfoInteger(_Symbol,SYMBOL_SPREAD);
   feat[33]=(float)_sd((double)spr_pts, atr);

   MqlDateTime dt; TimeToStruct(rb[i].time, dt);
   // pandas dayofweek: Mon=0..Sun=6. MQL5 day_of_week: Sun=0..Sat=6. Convert.
   int py_dow = (dt.day_of_week + 6) % 7;
   feat[34]=(float)MathSin(2.0*M_PI*(double)dt.hour/24.0);
   feat[35]=(float)MathSin(2.0*M_PI*(double)py_dow/5.0);

   return true;
}

//+------------------------------------------------------------------+
//| V6 Physics feature helpers                                        |
//+------------------------------------------------------------------+
double HurstRS(const MqlRates &rb[], int idx, int window=120)
{
   if(idx + window >= ArraySize(rb)) return 0.5;
   double sum_ret = 0;
   for(int k = idx; k < idx + window - 1; k++)
      if(rb[k+1].close > 1e-10) sum_ret += (rb[k].close - rb[k+1].close) / rb[k+1].close;
   double mean_ret = sum_ret / (window - 1);
   double cum = 0, maxC = -1e30, minC = 1e30, ss = 0;
   for(int k = idx; k < idx + window - 1; k++)
   {
      double r = (rb[k+1].close > 1e-10) ? (rb[k].close - rb[k+1].close) / rb[k+1].close : 0;
      cum += r - mean_ret;
      if(cum > maxC) maxC = cum; if(cum < minC) minC = cum;
      ss += (r - mean_ret) * (r - mean_ret);
   }
   double R = maxC - minC, S = MathSqrt(ss / (window - 1));
   if(S < 1e-15 || R <= 0) return 0.5;
   return MathLog(R / S) / MathLog((double)window);
}

double OUTheta(const MqlRates &rb[], int idx, int window=60)
{
   if(idx + window >= ArraySize(rb)) return 0.0;
   double x[]; ArrayResize(x, window); x[window-1] = 0;
   for(int k = window-2; k >= 0; k--)
      x[k] = x[k+1] + ((rb[idx+k+1].close>1e-10) ? MathLog(rb[idx+k].close/rb[idx+k+1].close) : 0);
   double mu = 0; for(int k=0;k<window;k++) mu+=x[k]; mu/=window;
   double num=0, den=0;
   for(int k=0;k<window-1;k++) { double dx=x[k]-x[k+1], xm=x[k+1]-mu; num+=dx*xm; den+=xm*xm; }
   return (den > 1e-15) ? -num/den : 0.0;
}

double EntropyRate(const MqlRates &rb[], int idx, int window=100)
{
   if(idx + window >= ArraySize(rb)) return 1.0;
   double rets[]; ArrayResize(rets, window-1);
   for(int k=0;k<window-1;k++)
      rets[k] = (rb[idx+k+1].close>1e-10) ? (rb[idx+k].close-rb[idx+k+1].close)/rb[idx+k+1].close : 0;
   double mn=rets[0], mx=rets[0];
   for(int k=1;k<window-1;k++) { if(rets[k]<mn) mn=rets[k]; if(rets[k]>mx) mx=rets[k]; }
   if(mx-mn < 1e-15) return 1.0;
   int nbins=10; int counts[]; ArrayResize(counts, nbins); ArrayInitialize(counts, 0);
   for(int k=0;k<window-1;k++)
   { int b=(int)((rets[k]-mn)/(mx-mn+1e-15)*nbins); if(b>=nbins)b=nbins-1; if(b<0)b=0; counts[b]++; }
   double ent=0, total=(double)(window-1);
   for(int b=0;b<nbins;b++) if(counts[b]>0) { double p=counts[b]/total; ent-=p*MathLog(p)/MathLog(2.0); }
   return ent / (MathLog((double)nbins)/MathLog(2.0));
}

double KramersEscape(const MqlRates &rb[], int idx, int window=100)
{
   if(idx + window >= ArraySize(rb)) return 0.5;
   double hi = rb[idx].close;
   for(int k=idx;k<idx+window;k++) if(rb[k].close>hi) hi=rb[k].close;
   double ss=0;
   for(int k=idx;k<idx+window-1;k++)
   { double lr=(rb[k+1].close>1e-10)?MathLog(rb[k].close/rb[k+1].close):0; ss+=lr*lr; }
   double sigma = MathSqrt(ss/(window-1));
   if(sigma<1e-12 || rb[idx].close<1e-10) return 0.5;
   return MathExp(-(hi-rb[idx].close)/(sigma*rb[idx].close+1e-10));
}

double WaveletER(const MqlRates &rb[], int idx, int window=120)
{
   if(idx + window >= ArraySize(rb)) return 1.0;
   double e_trend=0, e_noise=0;
   for(int k=idx;k<idx+window;k++)
   {
      double sm20=0; int c20=0;
      for(int j=k;j<k+20 && j<ArraySize(rb);j++){sm20+=rb[j].close;c20++;} if(c20>0)sm20/=c20; else sm20=rb[k].close;
      double sm3=0; int c3=0;
      for(int j=k;j<k+3 && j<ArraySize(rb);j++){sm3+=rb[j].close;c3++;} if(c3>0)sm3/=c3; else sm3=rb[k].close;
      e_trend += (rb[k].close-sm20)*(rb[k].close-sm20);
      e_noise += (rb[k].close-sm3)*(rb[k].close-sm3);
   }
   return (e_noise>1e-15) ? e_trend/e_noise : 1.0;
}

double ComputeVWAPDist(const MqlRates &rb[], int idx, double atr)
{
   MqlDateTime dt; TimeToStruct(rb[idx].time, dt);
   if(dt.day != g_vwap_day) { g_vwap_cum_tv=0; g_vwap_cum_v=0; g_vwap_day=dt.day; }
   double typical = (rb[idx].high+rb[idx].low+rb[idx].close)/3.0;
   double vol = MathMax((double)rb[idx].tick_volume, 1.0);
   g_vwap_cum_tv += typical*vol; g_vwap_cum_v += vol;
   double vwap = (g_vwap_cum_v>0) ? g_vwap_cum_tv/g_vwap_cum_v : rb[idx].close;
   return (atr>1e-10) ? (rb[idx].close-vwap)/atr : 0.0;
}

double QuantumFlowCalc(const MqlRates &rb[], int idx, int lookback=21, int vol_lb=50)
{
   if(idx+vol_lb >= ArraySize(rb)) return 0.0;
   double ha_c = (rb[idx].open+rb[idx].high+rb[idx].low+rb[idx].close)/4.0;
   double ha_o = (idx+1<ArraySize(rb)) ? (rb[idx+1].open+rb[idx+1].close)/2.0 : rb[idx].open;
   double tf = ha_c - ha_o;
   double avg_v=0; for(int k=idx;k<idx+vol_lb&&k<ArraySize(rb);k++) avg_v+=MathMax((double)rb[k].tick_volume,1.0); avg_v/=vol_lb;
   double vf = (avg_v>1e-10) ? MathMax((double)rb[idx].tick_volume,1.0)/avg_v : 0.0;
   double raw = tf * vf * 1000.0;
   double alpha = 2.0/(lookback+1.0), ema = raw;
   for(int k=1; k<MathMin(lookback*3, ArraySize(rb)-idx); k++)
   {
      double hc2=(rb[idx+k].open+rb[idx+k].high+rb[idx+k].low+rb[idx+k].close)/4.0;
      double ho2=(idx+k+1<ArraySize(rb))?(rb[idx+k+1].open+rb[idx+k+1].close)/2.0:rb[idx+k].open;
      double tf2=hc2-ho2;
      double av2=0; int cnt=0;
      for(int j=idx+k;j<idx+k+vol_lb&&j<ArraySize(rb);j++){av2+=MathMax((double)rb[j].tick_volume,1.0);cnt++;}
      if(cnt>0)av2/=cnt; double vf2=(av2>1e-10)?MathMax((double)rb[idx+k].tick_volume,1.0)/av2:0;
      ema = alpha*raw + (1.0-alpha)*(alpha*(tf2*vf2*1000.0)+(1.0-alpha)*ema);
   }
   double atr_v = RawATR14(rb, idx); double step = atr_v*0.5;
   if(step>1e-10) ema = MathRound(ema/step)*step;
   return ema;
}

//+------------------------------------------------------------------+
//| V6 BuildFeatures: 14 physics+quantum features                     |
//+------------------------------------------------------------------+
bool BuildFeatures(const MqlRates &rb[], float &feat[])
{
   if(ArraySize(rb) < LOOKBACK) return false;
   ArrayResize(feat, FEATURE_DIM);
   double atr = RawATR14(rb, 0); if(atr<1e-10) atr=1e-10;

   feat[0]  = (float)HurstRS(rb, 0, 120);
   feat[1]  = (float)OUTheta(rb, 0, 60);
   feat[2]  = (float)EntropyRate(rb, 0, 100);
   feat[3]  = (float)KramersEscape(rb, 0, 100);
   feat[4]  = (float)WaveletER(rb, 0, 120);
   feat[5]  = (float)ComputeVWAPDist(rb, 0, atr);

   MqlDateTime dt; TimeToStruct(rb[0].time, dt);
   int py_dow = (dt.day_of_week + 6) % 7;
   feat[6]  = (float)MathSin(2.0*M_PI*(double)dt.hour/24.0);
   feat[7]  = (float)MathSin(2.0*M_PI*(double)py_dow/5.0);

   double qf_m5 = QuantumFlowCalc(rb, 0);
   feat[8]  = (float)qf_m5;

   MqlRates h4[]; ArraySetAsSeries(h4, true);
   int h4n = CopyRates(_Symbol, PERIOD_H4, 1, 60, h4);
   double qf_h4 = (h4n >= 55) ? QuantumFlowCalc(h4, 0, 21, 20) : 0.0;
   feat[9]  = (float)qf_h4;

   double qf_m5_prev = QuantumFlowCalc(rb, 1);
   feat[10] = (float)(qf_m5 - qf_m5_prev);

   double vwap_dir = (feat[5]>0)?1.0:(feat[5]<0?-1.0:0.0);
   double qf_sign  = (qf_m5>0)?1.0:(qf_m5<0?-1.0:0.0);
   feat[11] = (float)(qf_sign * vwap_dir);

   double div = 0;
   if(qf_h4>0 && qf_m5<0) div = +1.0;
   if(qf_h4<0 && qf_m5>0) div = -1.0;
   feat[12] = (float)div;
   feat[13] = (float)((div!=0) ? MathAbs(qf_h4) : 0.0);
   return true;
}

//+------------------------------------------------------------------+
//| Exit model: load, run, release                                     |
//+------------------------------------------------------------------+
bool ExitModel_Load()
{
   g_exit_handle = OnnxCreate("exit_v6.onnx", ONNX_DEFAULT);
   if(g_exit_handle == INVALID_HANDLE)
   { Print("ExitModel: failed to load exit_v6.onnx err=", GetLastError()); return false; }
   ulong in_shape[]={1,EXIT_FEAT_DIM}; ulong lbl_shape[]={1}; ulong prob_shape[]={1,2};
   if(!OnnxSetInputShape(g_exit_handle,0,in_shape) ||
      !OnnxSetOutputShape(g_exit_handle,0,lbl_shape) ||
      !OnnxSetOutputShape(g_exit_handle,1,prob_shape))
   { Print("ExitModel: shape fail"); return false; }
   Print("ExitModel: exit_v6.onnx loaded"); return true;
}
void ExitModel_Release() { if(g_exit_handle!=INVALID_HANDLE && g_exit_handle>0) { OnnxRelease(g_exit_handle); g_exit_handle=INVALID_HANDLE; } }

bool CheckExitModel(const MqlRates &rb[], double &exit_prob)
{
   exit_prob = 0.0;
   if(g_exit_handle==INVALID_HANDLE || g_entry_atr<1e-10) return false;
   double unrealized = g_entry_dir * (rb[0].close - g_entry_price) / g_entry_atr;
   double pnl_3ago = (ArraySize(rb)>3) ? g_entry_dir*(rb[3].close-g_entry_price)/g_entry_atr : unrealized;
   float in_data[]; ArrayResize(in_data, EXIT_FEAT_DIM);
   in_data[0]=(float)unrealized; in_data[1]=(float)g_bars_held; in_data[2]=(float)(unrealized-pnl_3ago);
   in_data[3]=(float)HurstRS(rb,0,120); in_data[4]=(float)OUTheta(rb,0,60);
   in_data[5]=(float)EntropyRate(rb,0,100); in_data[6]=(float)KramersEscape(rb,0,100);
   in_data[7]=(float)WaveletER(rb,0,120); in_data[8]=(float)QuantumFlowCalc(rb,0);
   MqlRates h4[]; ArraySetAsSeries(h4,true);
   int h4n=CopyRates(_Symbol,PERIOD_H4,1,60,h4);
   in_data[9]=(float)((h4n>=55)?QuantumFlowCalc(h4,0,21,20):0.0);
   in_data[10]=(float)ComputeVWAPDist(rb,0,RawATR14(rb,0));
   long out_label[1]; float probs[2];
   if(!OnnxRun(g_exit_handle,ONNX_DEFAULT,in_data,out_label,probs)) { Print("ExitModel: OnnxRun failed"); return false; }
   exit_prob = (double)probs[1];
   return exit_prob >= EXIT_THRESHOLD;
}


//+------------------------------------------------------------------+
//| Position helpers (single position guard, magic-filtered)         |
//+------------------------------------------------------------------+
int CurrentDirection()
{
   for(int i=PositionsTotal()-1;i>=0;i--)
   {
      ulong t=PositionGetTicket(i);
      if(!PositionSelectByTicket(t)) continue;
      if(PositionGetString(POSITION_SYMBOL)!=_Symbol) continue;
      if((long)PositionGetInteger(POSITION_MAGIC)!=InpMagic) continue;
      return (PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY)?1:-1;
   }
   return 0;
}

double NormLots(double lots)
{
   double vmin=SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_MIN);
   double vmax=SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_MAX);
   double step=SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_STEP);
   if(step<=0) step=vmin;
   lots=MathMax(vmin,MathMin(vmax,MathRound(lots/step)*step));
   int dig=0; double sc=step;
   while(dig<8&&MathAbs(sc-MathRound(sc))>1e-12){sc*=10;dig++;}
   return NormalizeDouble(lots,dig);
}

//+------------------------------------------------------------------+
//| Weekly regime detection — same as v4                             |
//+------------------------------------------------------------------+
int g_active_cluster = REGIME_K - 1;   // start defensive (HighVol)

void RefreshActiveRegime()
{
   MqlRates rb[];
   ArraySetAsSeries(rb, true);
   int copied = CopyRates(_Symbol, MODEL_TF, 0, 1500, rb);
   if(copied < 200) { Print("regime refresh: insufficient bars"); return; }
   int week_bars = MathMin(copied - 1, 288);  // match Python WINDOW=288 in 02_build_selector_k5.py
   MqlRates week[];
   ArrayResize(week, week_bars);
   for(int i = 0; i < week_bars; i++) week[i] = rb[i];
   double fp[REGIME_N_FEATS];
   ComputeWeekFingerprint(week, fp);
   g_active_cluster = ClassifyRegime(fp);
   Print("v6: active regime = C", g_active_cluster, " ", REGIME_NAMES[g_active_cluster],
         " (tradeable=", REGIME_TRADEABLE[g_active_cluster], ")");
}

//+------------------------------------------------------------------+
//| Run all rules belonging to the active cluster, return best       |
//| confirmed direction (+1 buy / -1 sell / 0 nothing).              |
//| out_rule is filled with the rule_id of the chosen trade.         |
//+------------------------------------------------------------------+
int ScanAndConfirm(const float &feat_rules[], const float &feat_v6[], const MqlRates &rb[], int &out_rule)
{
   out_rule = -1;
   int best_dir = 0;
   double best_prob = 0.0;

   for(int r = 0; r < RULE_COUNT; r++)
   {
      if(RULE_CLUSTER[r] != g_active_cluster) continue;
      if(g_rule_cooldown[r] > 0) continue;

      int dir = CheckRule(r, feat_rules, rb, g_prev_stoch_k);
      if(dir == 0) continue;

      double prob = 0.0;
      bool pass = ConfirmV6_Rule(r, feat_v6, prob);
      if(!pass) continue;

      if(prob > best_prob)
      {
         best_prob = prob;
         best_dir  = dir;
         out_rule  = r;
      }
   }
   return best_dir;
}

//+------------------------------------------------------------------+
// License helpers
//+------------------------------------------------------------------+
string JsonString(const string &json, const string key)
{
   string search = "\"" + key + "\"";
   int pos = StringFind(json, search);
   if(pos < 0) return "";
   int colon = StringFind(json, ":", pos + StringLen(search));
   if(colon < 0) return "";
   int start = colon + 1;
   while(start < StringLen(json))
   {
      ushort ch = StringGetCharacter(json, start);
      if(ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r' || ch == '"') start++;
      else break;
   }
   int end = start;
   while(end < StringLen(json))
   {
      ushort ch = StringGetCharacter(json, end);
      if(ch == '"' || ch == ',' || ch == '}') break;
      end++;
   }
   return StringSubstr(json, start, end - start);
}

string LicenseQuery()
{
   return "key=" + InpLicenseKey + "&account=" + StringFormat("%I64d", AccountInfoInteger(ACCOUNT_LOGIN));
}

int RefreshLicenseStatus(bool startup_check)
{
   if(StringLen(InpLicenseKey) < 10)
   {
      Print("EdgePredictor ZigZag: No license key provided. Paste your key in Inputs tab.");
      if(startup_check)
         Alert("EdgePredictor ZigZag\nLicense key missing\nPaste your license key in the Inputs tab.");
      return LICENSE_STATUS_INVALID;
   }

   string url = InpServerURL + "/license-status?" + LicenseQuery();
   char post[]; char result[];
   string headers = "Content-Type: application/json\r\n";
   string resp_headers;

   ResetLastError();
   int res = WebRequest("GET", url, headers, 15000, post, result, resp_headers);
   string json = CharArrayToString(result, 0, WHOLE_ARRAY, CP_UTF8);

   if(res == 200)
   {
      g_licensed = true;
      g_last_license_ok = TimeCurrent();
      if(startup_check)
         Print("EdgePredictor ZigZag: License validated successfully.");
      return LICENSE_STATUS_ACTIVE;
   }

   if(res == 403)
   {
      string reason = JsonString(json, "reason");
      string message = JsonString(json, "message");
      if(message == "") message = "Your license is not active.";
      g_licensed = false;
      Print("EdgePredictor ZigZag: license inactive — ", message);
      if(startup_check)
         Alert("EdgePredictor ZigZag\nLicense: ", reason, "\n", message);
      return LICENSE_STATUS_INVALID;
   }

   if(g_last_license_ok > 0 && (TimeCurrent() - g_last_license_ok) <= LICENSE_GRACE_SECONDS)
   {
      g_licensed = true;
      return LICENSE_STATUS_GRACE;
   }

   g_licensed = false;
   PrintFormat("EdgePredictor ZigZag: license check failed (HTTP %d, err=%d)", res, GetLastError());
   if(startup_check)
      Alert("EdgePredictor ZigZag\nServer connection failed\nCheck MT5 WebRequest settings for: ", InpServerURL);
   return startup_check ? LICENSE_STATUS_INVALID : LICENSE_STATUS_ACTIVE;
}

bool ValidateLicense()
{
   int status = RefreshLicenseStatus(true);
   if(status == LICENSE_STATUS_ACTIVE)
   {
      Comment("EdgePredictor ZigZag v6.0\n\n  License: ACTIVE\n  Status: Ready to trade");
      return true;
   }
   return false;
}

bool DownloadModel(string filename)
{
   string url = InpServerURL + "/model?" + LicenseQuery() + "&name=" + filename;
   char post[]; char result[];
   string headers = "Content-Type: application/json\r\n";
   string resp_headers;
   ResetLastError();
   int res = WebRequest("GET", url, headers, 60000, post, result, resp_headers);
   if(res != 200)
   {
      PrintFormat("EdgePredictor ZigZag: model download HTTP %d for %s (err=%d)", res, filename, GetLastError());
      return false;
   }
   int fh = FileOpen(filename, FILE_WRITE | FILE_BIN);
   if(fh == INVALID_HANDLE)
   {
      PrintFormat("EdgePredictor ZigZag: cannot write '%s' (err=%d)", filename, GetLastError());
      return false;
   }
   FileWriteArray(fh, result, 0, ArraySize(result));
   FileClose(fh);
   PrintFormat("EdgePredictor ZigZag: '%s' downloaded (%d bytes)", filename, ArraySize(result));
   return true;
}

bool DownloadAllModels()
{
   Comment("EdgePredictor ZigZag v6.0\n\n  License: ACTIVE\n  Downloading AI models...");
   int ok = 0;
   int total = RULE_COUNT + 2; // + exit_v6 + regime_v4
   for(int r = 0; r < RULE_COUNT; r++)
   {
      string fname = RULE_ONNX_V6[r];
      if(!DownloadModel(fname))
      {
         PrintFormat("Midas v6: FAILED to download %s — aborting", fname);
         return false;
      }
      ok++;
      Comment(StringFormat("Midas v6.0\n\n  Downloading models... %d/%d", ok, total));
   }
   // Exit model
   if(!DownloadModel("exit_v6.onnx"))
   { PrintFormat("Midas v6: FAILED exit_v6.onnx — aborting"); return false; }
   ok++;
   // Vol-regime sizer — non-fatal: if server is older, EA falls back to 1.0x lot mult
   if(DownloadModel("regime_v4_xau.onnx"))
   {
      ok++;
      Comment(StringFormat("EdgePredictor ZigZag v6.0\n\n  License: ACTIVE\n  Downloading models... %d/%d", ok, total));
   }
   else
   {
      Print("EdgePredictor ZigZag: regime_v4_xau.onnx unavailable on server — continuing without vol sizing");
   }
   PrintFormat("EdgePredictor ZigZag: %d of %d models downloaded from server", ok, total);
   return true;
}

//+------------------------------------------------------------------+
int OnInit()
{
   SetupChart();
   DrawWatermark();

   Comment("Midas v6.0\n\n  Validating license...");

   if(!ValidateLicense())
   {
      Comment("Midas v6.0\n\n  LICENSE FAILED — check your key");
      return INIT_FAILED;
   }

   Comment("Midas v6.0\n\n  License: ACTIVE\n  Downloading AI models...");

   if(!DownloadAllModels())
   {
      Comment("Midas v6.0\n\n  MODEL DOWNLOAD FAILED");
      return INIT_FAILED;
   }

   g_trade.SetExpertMagicNumber(InpMagic);
   g_trade.SetDeviationInPoints(InpMaxDevPoints);

   if(!ConfirmV6_Load())
   { Print("Midas v6: failed to load v6 confirmation models"); return INIT_FAILED; }
   if(!ExitModel_Load())
   { Print("Midas v6: failed to load exit model"); return INIT_FAILED; }
   // Vol-regime sizer (regime_v4) — non-fatal if it fails; falls back to 1.0×
   if(!RegimeV4_Load("regime_v4_xau.onnx"))
      Print("EdgePredictor Midas: regime_v4 sizer unavailable — using flat 1.0× lots");
   ArrayInitialize(g_rule_cooldown, 0);
   g_bars_since_lic = 0;

   RefreshActiveRegime();
   g_last_regime_day = (int)(TimeCurrent() / 86400);

   // Recover any open position after recompile / reload / MT5 restart.
   for(int p = PositionsTotal() - 1; p >= 0; p--)
   {
      ulong ticket = PositionGetTicket(p); if(ticket == 0) continue;
      if(PositionGetInteger(POSITION_MAGIC) != InpMagic) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;

      g_entry_price = PositionGetDouble(POSITION_PRICE_OPEN);
      g_entry_dir   = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) ? +1 : -1;

      MqlRates rb[]; ArraySetAsSeries(rb, true);
      if(CopyRates(_Symbol, MODEL_TF, 0, 30, rb) >= 20)
         g_entry_atr = RawATR14(rb, 0);
      if(g_entry_atr < 1e-10) g_entry_atr = 1e-10;

      datetime t_open = (datetime)PositionGetInteger(POSITION_TIME);
      int secs_per_bar = PeriodSeconds(MODEL_TF);
      g_bars_held = (secs_per_bar > 0) ? (int)((TimeCurrent() - t_open) / secs_per_bar) : 0;
      if(g_bars_held < 0) g_bars_held = 0;

      PrintFormat("v6 RECOVERED open position: ticket=%I64u dir=%+d entry=%.2f atr=%.2f bars_held=%d",
                  ticket, g_entry_dir, g_entry_price, g_entry_atr, g_bars_held);
      break;
   }

   RefreshTradeStats();
   UpdateDashboard();

   PrintFormat("EdgePredictor Midas ready | regime=C%d %s | refresh=daily (00:00 UTC) long=%s short=%s",
               g_active_cluster, REGIME_NAMES[g_active_cluster],
               InpAllowLong?"Y":"N", InpAllowShort?"Y":"N");
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   ConfirmV6_Release();
   ExitModel_Release();
   RegimeV4_Release();
   ObjectDelete(ChartID(), "EP_Watermark");
   Comment("");
}

//+------------------------------------------------------------------+
void OnTick()
{
   datetime bt = iTime(_Symbol, MODEL_TF, 1);
   if(bt == 0 || bt == g_last_bar) return;
   g_last_bar = bt;

   g_uptime_bars++;

   // Periodic license recheck
   if(++g_bars_since_lic >= LICENSE_RECHECK_BARS)
   {
      RefreshLicenseStatus(false);
      g_bars_since_lic = 0;
   }
   if(!g_licensed) return;

   // Periodic stats report
   if(++g_bars_since_report >= REPORT_INTERVAL_BARS)
   {
      RefreshTradeStats();
      ReportStats();
      g_bars_since_report = 0;
   }

   // Calendar-anchored regime refresh — fires on first tick of each new UTC day
   // so every user's EA converges on the same cluster regardless of start time.
   int day_today = (int)(TimeCurrent() / 86400);
   if(day_today > g_last_regime_day)
   {
      RefreshActiveRegime();
      g_last_regime_day = day_today;
   }

   // Decrement per-rule cooldowns
   for(int r = 0; r < RULE_COUNT; r++)
      if(g_rule_cooldown[r] > 0) g_rule_cooldown[r]--;

   // Skip non-tradeable cluster (e.g. C2 Shock)
   if(REGIME_TRADEABLE[g_active_cluster] == 0)
   {
      if(InpVerbose)
         PrintFormat("v6 %s | regime=C%d %s | SKIP (non-tradeable)",
                     TimeToString(bt, TIME_DATE|TIME_MINUTES),
                     g_active_cluster, REGIME_NAMES[g_active_cluster]);
      return;
   }

   // Pull bars
   MqlRates rb[];
   ArraySetAsSeries(rb, true);
   if(CopyRates(_Symbol, MODEL_TF, 1, LOOKBACK, rb) < LOOKBACK)
   {
      if(InpVerbose) Print("v6: CopyRates insufficient");
      return;
   }

   // Spread gate
   long spread_pts = (long)SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
   if(spread_pts > InpMaxSpread)
   {
      if(InpVerbose)
         PrintFormat("v6: spread %d > %d, skipping", (int)spread_pts, InpMaxSpread);
      return;
   }

   // Build TWO feature arrays
   float feat_rules[];
   if(!BuildRuleFeatures(rb, feat_rules)) return;
   float feat_v6[];
   if(!BuildFeatures(rb, feat_v6)) return;

   double cur_sk = (double)feat_rules[FI_STOCH_K];

   // PATH A: In position → check ML exit
   int cur_dir = CurrentDirection();
   if(cur_dir != 0)
   {
      g_bars_held++;
      if(g_bars_held >= MIN_HOLD_BARS)
      {
         double exit_prob = 0;
         if(CheckExitModel(rb, exit_prob))
         {
            for(int p=PositionsTotal()-1;p>=0;p--)
            {
               ulong ticket=PositionGetTicket(p); if(ticket==0) continue;
               if(PositionGetInteger(POSITION_MAGIC)!=InpMagic) continue;
               if(PositionGetString(POSITION_SYMBOL)!=_Symbol) continue;
               g_trade.PositionClose(ticket);
               PrintFormat("v6 ML EXIT: prob=%.3f bars=%d", exit_prob, g_bars_held);
            }
            g_entry_dir=0; g_bars_held=0;
         }
      }
      RefreshTradeStats(); UpdateDashboard();
      return;
   }

   // PATH B: Flat → scan rules + confirm entry
   int rule_id = -1;
   int dir = ScanAndConfirm(feat_rules, feat_v6, rb, rule_id);

   RefreshTradeStats();
   UpdateDashboard();

   if(dir != 0)
   {
      g_last_signal = StringFormat("%s %s @ %s",
         dir > 0 ? "BUY" : "SELL",
         RULE_NAMES[rule_id],
         TimeToString(bt, TIME_MINUTES));
   }

   if(InpVerbose)
   {
      if(dir == 0)
         PrintFormat("v6 %s | regime=C%d %s | no signal",
                     TimeToString(bt, TIME_DATE|TIME_MINUTES),
                     g_active_cluster, REGIME_NAMES[g_active_cluster]);
      else
         PrintFormat("v6 %s | regime=C%d %s | rule=%s dir=%+d",
                     TimeToString(bt, TIME_DATE|TIME_MINUTES),
                     g_active_cluster, REGIME_NAMES[g_active_cluster],
                     RULE_NAMES[rule_id], dir);
   }

   g_prev_stoch_k = cur_sk;

   if(dir == 0) return;
   if(dir == +1 && !InpAllowLong) return;
   if(dir == -1 && !InpAllowShort) return;
   if(CurrentDirection() != 0) return;   // single position guard

   // Apply cooldown for the firing rule
   g_rule_cooldown[rule_id] = RULE_COOLDOWN[rule_id];

   // Compute SL/TP from current ATR
   double atr = RawATR14(rb, 0);
   if(atr < 1e-10) atr = 1e-10;
   double sl_dist = SL_HARD_ATR * atr;

   MqlTick tick;
   if(!SymbolInfoTick(_Symbol, tick)) return;

   // Vol-regime sizing: scale base lot by predicted regime's multiplier
   double regime_prob = 0.0;
   int vol_regime = RegimeV4_Predict(regime_prob);  // 0=Quiet 1=Normal 2=HighVol, -1 fail
   double lot_mult = (vol_regime >= 0) ? REGIME_V4_LOT_MULT[vol_regime] : 1.0;
   double lots = NormLots(InpLots * lot_mult);
   if(InpVerbose && vol_regime >= 0)
      PrintFormat("v6: vol=%s (p=%.2f) lot_mult=%.2fx base=%.2f final=%.2f",
                  REGIME_V4_NAMES[vol_regime], regime_prob, lot_mult, InpLots, lots);

   bool ok = false;
   if(dir == +1)
   {
      double sl = NormalizeDouble(tick.ask - sl_dist, _Digits);
      ok = g_trade.Buy(lots, _Symbol, tick.ask, sl, 0,
                       StringFormat("V6-L %s", RULE_NAMES[rule_id]));
      if(ok) { g_entry_price=tick.ask; g_entry_atr=atr; g_entry_dir=+1; g_bars_held=0; }
   }
   else
   {
      double sl = NormalizeDouble(tick.bid + sl_dist, _Digits);
      ok = g_trade.Sell(lots, _Symbol, tick.bid, sl, 0,
                        StringFormat("V6-S %s", RULE_NAMES[rule_id]));
      if(ok) { g_entry_price=tick.bid; g_entry_atr=atr; g_entry_dir=-1; g_bars_held=0; }
   }
   if(!ok && InpVerbose)
      PrintFormat("v6: order failed (%d)", GetLastError());
}
//+------------------------------------------------------------------+
