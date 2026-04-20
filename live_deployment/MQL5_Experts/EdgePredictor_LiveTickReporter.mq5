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
   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   int got = CopyRates(_Symbol, _Period, 0, 1, rates);
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

   // We report normalised "XAUUSD" + "PERIOD_M5" (or whatever the chart is on)
   // so the server's keyed lookup stays stable across brokers.
   string body = StringFormat(
      "{\"symbol\":\"%s\",\"tf\":\"%s\",\"bar\":{\"t\":%I64d,\"o\":%.2f,\"h\":%.2f,\"l\":%.2f,\"c\":%.2f,\"v\":%I64d},\"tick\":{\"bid\":%.2f,\"ask\":%.2f,\"t\":%I64d},\"ts\":%I64d}",
      ReportedSymbol,
      EnumToString(_Period),
      (long)rates[0].time,
      rates[0].open, rates[0].high, rates[0].low, rates[0].close,
      rates[0].tick_volume,
      tick.bid, tick.ask, (long)tick.time,
      (long)TimeCurrent()
   );

   char post_data[], result[];
   StringToCharArray(body, post_data, 0, StringLen(body), CP_UTF8);
   ArrayResize(post_data, ArraySize(post_data) - 1);

   string headers = "Content-Type: application/json\r\n";
   if(StringLen(AuthSecret) > 0)
      headers += "X-Tick-Secret: " + AuthSecret + "\r\n";

   string result_headers;
   ResetLastError();
   int status = WebRequest("POST", PostUrl, headers, 5000, post_data, result, result_headers);
   if(status == -1)
      PrintFormat("LiveTickReporter WebRequest error %d (whitelist %s?)", GetLastError(), PostUrl);
   else if(VerboseLog)
      PrintFormat("LiveTickReporter POST %s -> HTTP %d", ReportedSymbol, status);
}
//+------------------------------------------------------------------+
