//+------------------------------------------------------------------+
//|                                EdgePredictor_LiveTickReporter.mq5 |
//|   POSTs the current XAUUSD M5 candle + last tick to the backend   |
//|   every N seconds so the Oracle Coach live page has a real-time   |
//|   broker feed (not Yahoo futures) to drive its chart and analyst. |
//|                                                                   |
//|   Deployment:                                                     |
//|     1. Attach to XAUUSD M5 chart (any broker symbol — XAUUSD,     |
//|        XAUUSD-ECN, XAUUSDm, etc. all work; we use the chart's     |
//|        own symbol for data pulls and normalise the reported       |
//|        symbol to "XAUUSD" for the website).                       |
//|     2. Whitelist the POST_URL in Tools → Options → Expert         |
//|        Advisors → Allow WebRequest.                               |
//|     3. The EA does NOT trade — it's read-only (no OrderSend).     |
//+------------------------------------------------------------------+
#property copyright "EdgePredictor"
#property version   "1.10"
#property strict

input string   PostUrl           = "https://edgepredictor.pro/api/tick";
input string   AuthSecret        = "";          // optional: set same value as TICK_SECRET env var on server
input int      PostEverySec      = 5;           // how often to POST (seconds)
input int      HistoryBars       = 288;         // how many past bars to send each POST (288 = 24h of M5)
input string   ReportedSymbol    = "XAUUSD";    // symbol name sent to the website (normalised across brokers)
input bool     VerboseLog        = false;       // print every POST result

datetime g_last_post = 0;

int OnInit()
{
   if(StringLen(PostUrl) < 8)
     {
      Print("LiveTickReporter: PostUrl is empty — disabling.");
      return INIT_FAILED;
     }
   EventSetTimer(1);
   PrintFormat("LiveTickReporter: chart=%s report_as=%s tf=%s url=%s every=%ds",
               _Symbol, ReportedSymbol, EnumToString(_Period), PostUrl, PostEverySec);
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason) { EventKillTimer(); }

void OnTimer()
{
   if(TimeCurrent() - g_last_post < PostEverySec) return;
   g_last_post = TimeCurrent();
   ReportTick();
}

void ReportTick()
{
   // Always pull from the chart's own symbol + timeframe so the EA works on
   // brokers that use suffixes (XAUUSD-ECN, XAUUSDm, XAUUSD.i, etc.).
   // Pull HistoryBars+1 so we have the current forming bar [0] + N past bars.
   int need = MathMax(1, HistoryBars) + 1;
   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   int got = CopyRates(_Symbol, _Period, 0, need, rates);
   if(got <= 0)
     {
      if(VerboseLog) PrintFormat("LiveTickReporter: CopyRates failed for %s %s (err=%d)",
                                 _Symbol, EnumToString(_Period), GetLastError());
      return;
     }

   MqlTick tick;
   if(!SymbolInfoTick(_Symbol, tick))
     {
      if(VerboseLog) PrintFormat("LiveTickReporter: SymbolInfoTick failed for %s (err=%d)",
                                 _Symbol, GetLastError());
      return;
     }

   string s_bid = DoubleToString(tick.bid, 2);
   string s_ask = DoubleToString(tick.ask, 2);

   // Build history array — indexed from 1..got-1 (skip [0] which is the
   // forming bar, sent separately as `bar`). Oldest first so the client
   // can append directly without re-sorting.
   string hist = "[";
   for(int i = got - 1; i >= 1; --i)
     {
      if(i < got - 1) hist += ",";
      hist += StringFormat(
         "{\"t\":%I64d,\"o\":%s,\"h\":%s,\"l\":%s,\"c\":%s,\"v\":%I64d}",
         (long)rates[i].time,
         DoubleToString(rates[i].open,  2),
         DoubleToString(rates[i].high,  2),
         DoubleToString(rates[i].low,   2),
         DoubleToString(rates[i].close, 2),
         rates[i].tick_volume
      );
     }
   hist += "]";

   // Current (forming) bar
   string s_o = DoubleToString(rates[0].open,  2);
   string s_h = DoubleToString(rates[0].high,  2);
   string s_l = DoubleToString(rates[0].low,   2);
   string s_c = DoubleToString(rates[0].close, 2);

   // We report normalised "XAUUSD" + "PERIOD_M5" (or whatever the chart is on)
   // so the server's keyed lookup stays stable across brokers.
   string body = StringFormat(
      "{\"symbol\":\"%s\",\"tf\":\"%s\",\"bar\":{\"t\":%I64d,\"o\":%s,\"h\":%s,\"l\":%s,\"c\":%s,\"v\":%I64d},\"tick\":{\"bid\":%s,\"ask\":%s,\"t\":%I64d},\"history\":%s,\"ts\":%I64d}",
      ReportedSymbol,
      EnumToString(_Period),
      (long)rates[0].time,
      s_o, s_h, s_l, s_c,
      rates[0].tick_volume,
      s_bid, s_ask, (long)tick.time,
      hist,
      (long)TimeCurrent()
   );

   // Convert body to UTF-8 byte array using WHOLE_ARRAY so we get a reliable
   // count back; then trim exactly one byte for the trailing null terminator.
   // (count=StringLen(...) inconsistently appends the null which caused the
   //  closing '}' to get chopped → server saw malformed JSON → HTTP 400.)
   char post_data[], result[];
   int converted = StringToCharArray(body, post_data, 0, WHOLE_ARRAY, CP_UTF8);
   if(converted <= 1)
     {
      Print("LiveTickReporter: StringToCharArray failed");
      return;
     }
   ArrayResize(post_data, converted - 1);

   string headers = "Content-Type: application/json\r\n";
   if(StringLen(AuthSecret) > 0)
      headers += "X-Tick-Secret: " + AuthSecret + "\r\n";

   string result_headers;
   ResetLastError();
   int status = WebRequest("POST", PostUrl, headers, 5000, post_data, result, result_headers);
   if(status == -1)
     {
      PrintFormat("LiveTickReporter WebRequest error %d (whitelist %s?)", GetLastError(), PostUrl);
     }
   else if(status != 200)
     {
      // On error, dump the body we sent + server's response so the cause is
      // visible instead of silent-failing.
      string resp = CharArrayToString(result, 0, -1, CP_UTF8);
      PrintFormat("LiveTickReporter POST %s -> HTTP %d", ReportedSymbol, status);
      PrintFormat("  body: %s", body);
      PrintFormat("  resp: %s", resp);
     }
   else if(VerboseLog)
     {
      PrintFormat("LiveTickReporter POST %s -> HTTP %d", ReportedSymbol, status);
     }
}
//+------------------------------------------------------------------+
