//+------------------------------------------------------------------+
//|  SwingScalperEA_v4.mq5                                            |
//|  XAUUSD M5 — Regime-specialist multi-model scalper                |
//|                                                                    |
//|  Changes from v3:                                                  |
//|   • Loads 3 regime-specialist ONNX models (C0 C1 C3)              |
//|   • C2 Shock weeks are auto-skipped (never trade)                 |
//|   • Weekly regime detection via regime_selector.mqh constants     |
//|   • Per-cluster probability thresholds baked in                   |
//|                                                                    |
//|  Feature computation (36 features) is unchanged from v3 —         |
//|  same base-f01..f20 + tech (rsi, stoch, mom, bb, etc.)           |
//|                                                                    |
//|  SL = SL_ATR_MULT × ATR(14)                                       |
//|  TP = TP_ATR_MULT × ATR(14)                                       |
//+------------------------------------------------------------------+
#property copyright "SwingScalper V4 (regime)"
#property version   "4.00"
#property strict

#include <Trade/Trade.mqh>
#include <regime_router.mqh>   // pulls in regime_selector.mqh

//--- inputs
input string InpModelC0            = "regime_0_Ranging.onnx";
input string InpModelC1            = "regime_1_Downtrend.onnx";
input string InpModelC3            = "regime_3_Uptrend.onnx";
input int    InpRegimeRefreshBars  = 288;   // refresh regime every N M5 bars (288 = 1 day)
input double InpLots               = 0.01;
input bool   InpAllowShort         = true;
input int    InpMaxSpread          = 80;    // skip entry if spread > N points
input long   InpMagic              = 420304;
input int    InpMaxDevPoints       = 30;
input bool   InpVerbose            = true;

//--- constants  (MUST match labeler.py)
static const double TP_ATR_MULT    = 2.0;
static const double SL_ATR_MULT    = 1.0;
static const int    FEATURE_DIM    = 36;
static const int    LOOKBACK       = 90;     // bars to copy for feature compute
static const ENUM_TIMEFRAMES MODEL_TF = PERIOD_M5;

CTrade   g_trade;
datetime g_last_bar    = 0;
int      g_bars_since_refresh = 0;

//+------------------------------------------------------------------+
// Utility
//+------------------------------------------------------------------+
double _sd(double n, double d) { return (MathAbs(d) < 1e-12) ? 0.0 : n / d; }
double _sg(double x)           { return x > 0.0 ? 1.0 : x < 0.0 ? -1.0 : 0.0; }
double _sq(double x)           { return x * x; }

//+------------------------------------------------------------------+
// Simple-mean ATR(14) — matches RawATR in labeler / MQL5 exporter
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

//+------------------------------------------------------------------+
// LR slope over last 5 bars — matches f06 in exporter
//+------------------------------------------------------------------+
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

//+------------------------------------------------------------------+
// Simple RSI (plain mean — matches labeler.py simple_rsi)
// Returns raw RSI [0..100] / 100 - 0.5 → [-0.5, 0.5]
//+------------------------------------------------------------------+
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

//+------------------------------------------------------------------+
// Stochastic %K(period)
//+------------------------------------------------------------------+
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

//+------------------------------------------------------------------+
// Bollinger %B (period=20, dev=2.0)
//+------------------------------------------------------------------+
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

//+------------------------------------------------------------------+
// ATR SMA-50 ratio  (current ATR / sma50(ATR) - 1)
//+------------------------------------------------------------------+
double ATRRatio(const MqlRates &rb[], int idx)
{
   if(idx + 64 >= ArraySize(rb)) return 0.0;
   double cur = RawATR14(rb, idx);
   double sum = 0;
   for(int k=idx;k<idx+50;k++) sum += RawATR14(rb,k);
   double sma = sum / 50.0;
   return _sd(cur - sma, sma);
}

//+------------------------------------------------------------------+
// Volume acceleration: SMA3/SMA20 - 1
//+------------------------------------------------------------------+
double VolAccel(const MqlRates &rb[], int idx)
{
   if(idx + 22 >= ArraySize(rb)) return 0.0;
   double s3=0,s20=0;
   for(int k=idx;k<idx+3;k++)  s3  += (double)rb[k].tick_volume;
   for(int k=idx;k<idx+20;k++) s20 += (double)rb[k].tick_volume;
   return _sd(s3/3.0 - s20/20.0, s20/20.0);
}

//+------------------------------------------------------------------+
//| Build 36-feature vector                                           |
//| rb[] is AS-SERIES: rb[0] = current fully-closed bar               |
//+------------------------------------------------------------------+
bool BuildFeatures(const MqlRates &rb[], float &feat[])
{
   if(ArraySize(rb) < LOOKBACK) return false;
   ArrayResize(feat, FEATURE_DIM);

   int i=0;
   double o=rb[i].open, h=rb[i].high, l=rb[i].low, c=rb[i].close;
   double v=(double)rb[i].tick_volume;
   double range=h-l, body=MathAbs(c-o);
   double uwick=h-MathMax(o,c), lwick=MathMin(o,c)-l;
   double atr=RawATR14(rb,i);
   if(atr<1e-10) atr=1e-10;

   //--- f01..f20 ---------------------------------------------------------
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

   //--- f21..f35 technical indicators -----------------------------------
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

   long spr_pts = SymbolInfoInteger(_Symbol,SYMBOL_SPREAD);
   double spr_price = spr_pts * SymbolInfoDouble(_Symbol,SYMBOL_POINT);
   feat[33]=(float)_sd(spr_price,atr);

   MqlDateTime dt; TimeToStruct(rb[i].time, dt);
   feat[34]=(float)MathSin(2.0*M_PI*(double)dt.hour/24.0);
   feat[35]=(float)MathSin(2.0*M_PI*(double)dt.day_of_week/5.0);

   return true;
}

//+------------------------------------------------------------------+
// Position management
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

bool CloseAllManaged()
{
   bool ok=true;
   for(int i=PositionsTotal()-1;i>=0;i--)
   {
      ulong t=PositionGetTicket(i);
      if(!PositionSelectByTicket(t)) continue;
      if(PositionGetString(POSITION_SYMBOL)!=_Symbol) continue;
      if((long)PositionGetInteger(POSITION_MAGIC)!=InpMagic) continue;
      if(!g_trade.PositionClose(t)) ok=false;
   }
   return ok;
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
int OnInit()
{
   if(_Period!=MODEL_TF)
   {
      Print("SwingScalperV4: attach to M5 chart only.");
      return INIT_FAILED;
   }

   g_trade.SetExpertMagicNumber(InpMagic);
   g_trade.SetDeviationInPoints(InpMaxDevPoints);

   if(!RR_LoadModels(InpModelC0, InpModelC1, InpModelC3))
   {
      Print("SwingScalperV4: failed to load one or more regime models. "
            "Verify files exist in MQL5/Files/Common/");
      return INIT_FAILED;
   }

   // Pick the starting regime so we're not flat until the first refresh
   RR_RefreshRegime(MODEL_TF, _Symbol, 2000);
   g_bars_since_refresh = 0;

   PrintFormat("SwingScalperV4 ready | C0=%s C1=%s C3=%s refresh=%d bars short=%s",
               InpModelC0, InpModelC1, InpModelC3,
               InpRegimeRefreshBars, InpAllowShort?"Y":"N");
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   RR_ReleaseModels();
}

//+------------------------------------------------------------------+
void OnTick()
{
   // Run once per fully-closed M5 bar
   datetime bt=iTime(_Symbol,MODEL_TF,1);
   if(bt==0||bt==g_last_bar) return;
   g_last_bar=bt;

   // Periodic regime refresh (once per day by default)
   if(++g_bars_since_refresh >= InpRegimeRefreshBars)
   {
      RR_RefreshRegime(MODEL_TF, _Symbol, 2000);
      g_bars_since_refresh = 0;
   }

   // Copy LOOKBACK bars (start from bar 1 = most recently closed)
   MqlRates rb[];
   ArraySetAsSeries(rb,true);
   if(CopyRates(_Symbol,MODEL_TF,1,LOOKBACK,rb)<LOOKBACK)
   {
      if(InpVerbose) Print("SwingScalperV4: CopyRates insufficient");
      return;
   }

   // Spread gate
   long spread_pts=(long)SymbolInfoInteger(_Symbol,SYMBOL_SPREAD);
   bool spread_ok=(spread_pts<=InpMaxSpread);

   int cur_dir=CurrentDirection();

   // Build features
   float feat[];
   if(!BuildFeatures(rb,feat)) return;

   // Regime-routed inference
   double win_prob = 0.0;
   RR_SIGNAL signal = RR_Predict(feat, win_prob);
   int sig = (int)signal;  // RR_BUY=+1, RR_SELL=-1, RR_FLAT=0

   if(InpVerbose)
      PrintFormat("SwingV4 %s | regime=C%d %s | sig=%d prob=%.4f pos=%d spread=%d",
                  TimeToString(bt,TIME_DATE|TIME_MINUTES),
                  g_rr_active_cluster,
                  REGIME_NAMES[g_rr_active_cluster],
                  sig, win_prob, cur_dir, spread_pts);

   if(sig==0)          return;  // flat
   if(sig==cur_dir)    return;  // already in that direction
   if(sig==-1 && !InpAllowShort) return;

   if(!spread_ok)
   {
      if(InpVerbose) Print("SwingV4: spread too wide, skipping");
      return;
   }

   // Close opposite position first
   if(cur_dir!=0)
   {
      if(!CloseAllManaged())
      {
         if(InpVerbose) PrintFormat("SwingV4: close failed (%d)",GetLastError());
         return;
      }
   }

   // Compute ATR for SL/TP (matches the labeler / backtest exactly)
   double atr=RawATR14(rb,0);
   if(atr<1e-10) atr=1e-10;
   double sl_dist=SL_ATR_MULT*atr;
   double tp_dist=TP_ATR_MULT*atr;

   MqlTick tick;
   if(!SymbolInfoTick(_Symbol,tick)) return;

   double lots=NormLots(InpLots);
   bool ok=false;

   if(sig==1)   // BUY
   {
      double sl=NormalizeDouble(tick.ask-sl_dist,_Digits);
      double tp=NormalizeDouble(tick.ask+tp_dist,_Digits);
      ok=g_trade.Buy(lots,_Symbol,tick.ask,sl,tp,"SwingV4-L");
   }
   else         // SELL
   {
      double sl=NormalizeDouble(tick.bid+sl_dist,_Digits);
      double tp=NormalizeDouble(tick.bid-tp_dist,_Digits);
      ok=g_trade.Sell(lots,_Symbol,tick.bid,sl,tp,"SwingV4-S");
   }

   if(!ok && InpVerbose)
      PrintFormat("SwingV4: order failed (%d)",GetLastError());
}
//+------------------------------------------------------------------+
