//+------------------------------------------------------------------+
//| v7_parity_test.mq5                                              |
//|                                                                   |
//| Computes the 4 v7 features (VPIN, sig_quad_var, HAR-RV ratio,  |
//| Hawkes η) at the last N bars of XAUUSD M5, with the same 6-bar   |
//| smoothing convention as the Python side.  Writes to              |
//| MQL5/Files/v7_mql5_features.csv so a Python diff script can    |
//| compare row-by-row vs the values stored in setups_*_v7.csv.    |
//+------------------------------------------------------------------+
#property script_show_inputs
#property strict

#include <v7_features.mqh>

input string InpSymbol    = "";      // leave blank → use current chart symbol (handles XAUUSD-ECN etc.)
input int    InpWarmupBars= 30;      // prime smoothing state before sampling
input int    InpTestBars  = 50;      // number of bars to sample feature values at
input int    InpSkipRecent= 5000;    // skip the N most recent bars (~17 days on M5) so the sample
                                     //   window sits inside the Python swing CSV training range

void OnStart()
{
   string sym = (StringLen(InpSymbol) > 0) ? InpSymbol : _Symbol;

   // Pull enough history to satisfy:  deepest lookback (HAR-RV long 8640) +
   //   smoothing warmup + test bars + skip-recent buffer.
   int NEED = 12000 + InpSkipRecent;
   MqlRates rb[];
   ArraySetAsSeries(rb, true);
   int got = CopyRates(sym, PERIOD_M5, 0, NEED, rb);
   int min_required = 10000 + InpSkipRecent;
   if(got < min_required)
   {
      PrintFormat("Only got %d bars for %s — need at least %d.  Aborting.", got, sym, min_required);
      return;
   }
   PrintFormat("Loaded %d bars of %s M5 (skipping most recent %d to stay inside training range)",
               got, sym, InpSkipRecent);

   // For each test point, we need rb[] such that rb[0] is the bar BEING
   // evaluated.  We'll iterate from oldest → newest so the smoothing ring
   // buffer fills in chronological order (matching Python).
   V7_ResetSmoothingState();

   // Build a secondary array where rb[0] is the oldest we want to feed,
   // rb[k] is older.  We want to simulate running the EA forward through
   // time.  For feature at bar i, rb[0]=bar_i, rb[1]=bar_{i-1}, ...
   //
   // Easiest: for each bar i from (oldest_test - warmup) forward, build a
   // slice view with rb[0]=bar_i and call ComputeV7Features().
   //
   // To do that without copying, we use a helper: sub_rb[k] = rb[base - i + k].
   // But MQL5 doesn't offer array views.  We make copies.

   // Sample window, shifted into the past by InpSkipRecent so all timestamps
   // land inside the Python swing CSV's training range.
   int test_start_idx = InpSkipRecent + InpTestBars + InpWarmupBars - 1;   // oldest sampled bar
   int test_end_idx   = InpSkipRecent;                                       // newest sampled bar
   if(test_start_idx >= got)
   {
      Print("Not enough bars for requested warmup + test span");
      return;
   }

   int fh = FileOpen("v7_mql5_features.csv", FILE_WRITE|FILE_CSV|FILE_ANSI, ',');
   if(fh == INVALID_HANDLE)
   {
      PrintFormat("FileOpen failed err=%d", GetLastError());
      return;
   }
   FileWriteString(fh, "time,vpin,sig_quad_var,har_rv_ratio,hawkes_eta\n");

   int warmup_count = 0, sampled = 0;
   // Iterate oldest → newest.  series index `s` where s=0 is newest bar.
   for(int s = test_start_idx; s >= test_end_idx; s--)
   {
      // Build a sliced view where local_rb[0] = rb[s], local_rb[1] = rb[s+1], ...
      int local_depth = got - s;
      if(local_depth < 10000) continue;   // not enough history for this evaluation
      MqlRates local_rb[];
      ArraySetAsSeries(local_rb, true);
      ArrayResize(local_rb, local_depth);
      for(int k = 0; k < local_depth; k++)
      {
         local_rb[k] = rb[s + k];
      }

      datetime bar_time = rb[s].time;
      double vpin, qv, har, hawkes;
      ComputeV7Features(local_rb, bar_time, vpin, qv, har, hawkes);

      if(warmup_count < InpWarmupBars)
      {
         warmup_count++;
         continue;   // don't record warmup bars — they exist only to prime smoothing
      }

      string row = StringFormat("%s,%.10f,%.10f,%.10f,%.10f\n",
                                TimeToString(bar_time, TIME_DATE|TIME_MINUTES),
                                vpin, qv, har, hawkes);
      FileWriteString(fh, row);
      sampled++;
   }
   FileClose(fh);
   PrintFormat("Wrote %d sampled bars (after %d warmup) to v7_mql5_features.csv",
               sampled, warmup_count);
   Print("Copy the file from MQL5/Files/ to the repo for comparison.");
}
