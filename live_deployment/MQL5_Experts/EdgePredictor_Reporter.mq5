//+------------------------------------------------------------------+
//| EdgePredictor_Reporter.mq5 — pushes forward-test stats to server |
//| Run on your MT5 only (not for customers). Uses WebRequest POST.  |
//+------------------------------------------------------------------+
#property copyright "EdgePredictor"
#property version   "1.02"

input string InpServerURL    = "https://edge-predictor.onrender.com";
input string InpReportSecret  = "6a46f7b2bf49458366a07838b25df7578a5633b0b735288e9ba99d3f59f59ec9";                    // Same as REPORT_SECRET on server (.env)
input int    InpIntervalSec  = 3600;                   // Push every N seconds
input bool   InpTodayOnly    = false;                  // Full history mode for website charts; set true only if you intentionally merge a daily segment
input int    InpLookbackDays = 45;                     // When today-only: load this many days of deals so overnight positions count correctly
input int    InpHistoryDays  = 60;                     // When InpTodayOnly=false: how far back to scan deals (2 months)
input bool   InpVerbose      = true;

struct PosBucket
  {
   ulong    pos_id;
   double   net;
   string   comment;
   datetime t_first;
   datetime t_last;
   bool     is_buy;        // direction captured from the entry-in deal
  };

struct DealSnap
  {
   datetime tm;
   double   net;
  };

//+------------------------------------------------------------------+
bool IsOurEA(const string cmt)
  {
   if(StringLen(cmt) < 2)
      return false;
   // Currently-supported products: Midas v6 + Phantom + Oracle v7
   // (V5- Midas is deprecated; EU-/GJ- products discontinued)
   if(StringFind(cmt, "V6-") == 0)       return true;   // Midas v6
   if(StringFind(cmt, "V7-") == 0)       return true;   // Oracle v7
   if(StringFind(cmt, "EP-DQN-B") >= 0)  return true;   // Phantom
   return false;
  }

//+------------------------------------------------------------------+
bool IsGoldSymbol(const string sym)
  {
   return (StringFind(sym, "XAU") >= 0 || StringFind(sym, "GOLD") >= 0);
  }

bool IsOurSymbol(const string sym)
  {
   return IsGoldSymbol(sym);
  }

//+------------------------------------------------------------------+
bool IsCurrentEP(const string cmt)
  {
   return (StringFind(cmt, "V6-") == 0 ||
           StringFind(cmt, "V7-") == 0 ||
           StringFind(cmt, "EP-DQN-B") >= 0);
  }

//+------------------------------------------------------------------+
string ModelCanonical(const string cmt)
  {
   if(StringFind(cmt, "V6-") == 0)
      return "EdgePredictor Midas";
   if(StringFind(cmt, "V7-") == 0)
      return "EdgePredictor Oracle";
   if(StringFind(cmt, "EP-DQN-B") >= 0)
      return "EdgePredictor Phantom";
   return "";
  }

//+------------------------------------------------------------------+
string ModelColor(const string name)
  {
   if(name == "EdgePredictor Midas")     return "#10b981";
   if(name == "EdgePredictor Phantom")   return "#a855f7";
   if(name == "EdgePredictor Oracle")    return "#6366f1";
   return "#888888";
  }

//+------------------------------------------------------------------+
void AddBucket(PosBucket &buckets[], const ulong pos_id, const double net,
               const string cmt, const datetime tm, const long entry,
               const long deal_type)
  {
   int n = ArraySize(buckets);
   for(int i = 0; i < n; i++)
     {
      if(buckets[i].pos_id != pos_id)
         continue;
      buckets[i].net += net;
      if(tm < buckets[i].t_first)
         buckets[i].t_first = tm;
      if(tm > buckets[i].t_last)
         buckets[i].t_last = tm;
      if(StringLen(cmt) > 0 && (entry == DEAL_ENTRY_IN || StringLen(buckets[i].comment) == 0))
         buckets[i].comment = cmt;
      if(entry == DEAL_ENTRY_IN)
         buckets[i].is_buy = (deal_type == DEAL_TYPE_BUY);
      return;
     }
   ArrayResize(buckets, n + 1);
   buckets[n].pos_id = pos_id;
   buckets[n].net = net;
   buckets[n].comment = cmt;
   buckets[n].t_first = tm;
   buckets[n].t_last = tm;
   buckets[n].is_buy = (deal_type == DEAL_TYPE_BUY);   // best-effort default
  }

//+------------------------------------------------------------------+
void SortDealsByTime(DealSnap &deals[])
  {
   int n = ArraySize(deals);
   for(int i = 0; i < n - 1; i++)
      for(int j = 0; j < n - i - 1; j++)
         if(deals[j].tm > deals[j + 1].tm)
           {
            DealSnap t = deals[j];
            deals[j] = deals[j + 1];
            deals[j + 1] = t;
           }
  }

//+------------------------------------------------------------------+
string JsonEscape(const string s)
  {
   string o = s;
   StringReplace(o, "\\", "\\\\");
   StringReplace(o, "\"", "\\\"");
   StringReplace(o, "\r", " ");
   StringReplace(o, "\n", " ");
   return o;
  }

//+------------------------------------------------------------------+
string FmtDateShort(const datetime t)
  {
   MqlDateTime dt;
   TimeToStruct(t, dt);
   string mon = "???";
   if(dt.mon == 1)
      mon = "Jan";
   if(dt.mon == 2)
      mon = "Feb";
   if(dt.mon == 3)
      mon = "Mar";
   if(dt.mon == 4)
      mon = "Apr";
   if(dt.mon == 5)
      mon = "May";
   if(dt.mon == 6)
      mon = "Jun";
   if(dt.mon == 7)
      mon = "Jul";
   if(dt.mon == 8)
      mon = "Aug";
   if(dt.mon == 9)
      mon = "Sep";
   if(dt.mon == 10)
      mon = "Oct";
   if(dt.mon == 11)
      mon = "Nov";
   if(dt.mon == 12)
      mon = "Dec";
   return StringFormat("%s %d, %d", mon, dt.day, dt.year);
  }

//+------------------------------------------------------------------+
string FmtIsoDate(const datetime t)
  {
   MqlDateTime dt;
   TimeToStruct(t, dt);
   return StringFormat("%04d-%02d-%02d", dt.year, dt.mon, dt.day);
  }

//+------------------------------------------------------------------+
string FmtIsoTimestamp(const datetime t)
  {
   MqlDateTime dt;
   TimeToStruct(t, dt);
   // Broker-local time — Date.parse() will treat unsuffixed ISO as local,
   // which is what we want for the ticker's "X hours ago" display.
   return StringFormat("%04d-%02d-%02dT%02d:%02d:%02d",
                       dt.year, dt.mon, dt.day, dt.hour, dt.min, dt.sec);
  }

//+------------------------------------------------------------------+
datetime BrokerDayStart()
  {
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   dt.hour = 0;
   dt.min = 0;
   dt.sec = 0;
   return StructToTime(dt);
  }

//+------------------------------------------------------------------+
bool PositionClosedToday(const ulong pos_id, const PosBucket &use_buckets[])
  {
   for(int j = 0; j < ArraySize(use_buckets); j++)
      if(use_buckets[j].pos_id == pos_id)
         return true;
   return false;
  }

//+------------------------------------------------------------------+
void SendReport()
  {
   if(StringLen(InpReportSecret) < 8)
     {
      Print("EdgePredictor_Reporter: set InpReportSecret (must match server REPORT_SECRET)");
      return;
     }

   datetime today_start = BrokerDayStart();
   datetime from_select;
   if(InpTodayOnly)
      from_select = today_start - (datetime)InpLookbackDays * 86400;
   else
      from_select = TimeCurrent() - (datetime)InpHistoryDays * 86400;

   if(!HistorySelect(from_select, TimeCurrent()))
     {
      Print("EdgePredictor_Reporter: HistorySelect failed");
      return;
     }

   // First pass: identify our position IDs from any deal whose comment matches our EAs
   // (brokers may rewrite OUT-deal comments to "[sl ...]" / "[tp ...]", so we must
   // anchor on pos_id, not on each deal's own comment).
   ulong our_pos_ids[];
   int total_deals = HistoryDealsTotal();
   for(int i = 0; i < total_deals; i++)
     {
      ulong ticket = HistoryDealGetTicket(i);
      if(ticket == 0)
         continue;
      string sym = HistoryDealGetString(ticket, DEAL_SYMBOL);
      if(!IsOurSymbol(sym))
         continue;
      string cmt = HistoryDealGetString(ticket, DEAL_COMMENT);
      if(!IsOurEA(cmt))
         continue;
      ulong pid = (ulong)HistoryDealGetInteger(ticket, DEAL_POSITION_ID);
      bool seen = false;
      for(int k = 0; k < ArraySize(our_pos_ids); k++)
         if(our_pos_ids[k] == pid) { seen = true; break; }
      if(!seen)
        {
         int nn = ArraySize(our_pos_ids);
         ArrayResize(our_pos_ids, nn + 1);
         our_pos_ids[nn] = pid;
        }
     }

   // Second pass: aggregate ALL deals for our position IDs (including SL/TP OUT deals
   // whose comment was rewritten by the broker).
   PosBucket buckets[];
   for(int i = 0; i < total_deals; i++)
     {
      ulong ticket = HistoryDealGetTicket(i);
      if(ticket == 0)
         continue;

      string sym = HistoryDealGetString(ticket, DEAL_SYMBOL);
      if(!IsOurSymbol(sym))
         continue;

      ulong pos_id = (ulong)HistoryDealGetInteger(ticket, DEAL_POSITION_ID);
      bool is_ours = false;
      for(int k = 0; k < ArraySize(our_pos_ids); k++)
         if(our_pos_ids[k] == pos_id) { is_ours = true; break; }
      if(!is_ours)
         continue;

      string cmt = HistoryDealGetString(ticket, DEAL_COMMENT);
      double profit = HistoryDealGetDouble(ticket, DEAL_PROFIT);
      double comm = HistoryDealGetDouble(ticket, DEAL_COMMISSION);
      double swap = HistoryDealGetDouble(ticket, DEAL_SWAP);
      double fee = HistoryDealGetDouble(ticket, DEAL_FEE);
      double net = profit + comm + swap + fee;
      datetime tm = (datetime)HistoryDealGetInteger(ticket, DEAL_TIME);
      long entry = HistoryDealGetInteger(ticket, DEAL_ENTRY);
      long dtype = HistoryDealGetInteger(ticket, DEAL_TYPE);

      AddBucket(buckets, pos_id, net, cmt, tm, entry, dtype);
     }

   PosBucket use_buckets[];
   if(InpTodayOnly)
     {
      for(int b = 0; b < ArraySize(buckets); b++)
        {
         if(buckets[b].t_last < today_start)
            continue;
         int n = ArraySize(use_buckets);
         ArrayResize(use_buckets, n + 1);
         use_buckets[n] = buckets[b];
        }
     }
   else
     {
      ArrayResize(use_buckets, ArraySize(buckets));
      for(int b = 0; b < ArraySize(buckets); b++)
         use_buckets[b] = buckets[b];
     }

   DealSnap deals[];
   datetime t_min = 0;
   datetime t_max = 0;
   for(int i = 0; i < total_deals; i++)
     {
      ulong ticket = HistoryDealGetTicket(i);
      if(ticket == 0)
         continue;

      string sym = HistoryDealGetString(ticket, DEAL_SYMBOL);
      if(!IsOurSymbol(sym))
         continue;

      ulong pos_id = (ulong)HistoryDealGetInteger(ticket, DEAL_POSITION_ID);
      bool is_ours = false;
      for(int k = 0; k < ArraySize(our_pos_ids); k++)
         if(our_pos_ids[k] == pos_id) { is_ours = true; break; }
      if(!is_ours)
         continue;

      if(InpTodayOnly && !PositionClosedToday(pos_id, use_buckets))
         continue;

      double profit = HistoryDealGetDouble(ticket, DEAL_PROFIT);
      double comm = HistoryDealGetDouble(ticket, DEAL_COMMISSION);
      double swap = HistoryDealGetDouble(ticket, DEAL_SWAP);
      double fee = HistoryDealGetDouble(ticket, DEAL_FEE);
      double net = profit + comm + swap + fee;
      datetime tm = (datetime)HistoryDealGetInteger(ticket, DEAL_TIME);

      int nd = ArraySize(deals);
      ArrayResize(deals, nd + 1);
      deals[nd].tm = tm;
      deals[nd].net = net;

      if(t_min == 0 || tm < t_min)
         t_min = tm;
      if(tm > t_max)
         t_max = tm;
     }

   SortDealsByTime(deals);

   if(t_min == 0)
     {
      t_min = InpTodayOnly ? today_start : TimeCurrent();
      t_max = TimeCurrent();
     }

   double gross_profit = 0;
   double gross_loss = 0;

   int nb = ArraySize(use_buckets);
   int wins = 0;
   int losses = 0;
   double sum_net = 0;
   double largest_win = 0;
   double largest_loss = 0;

   for(int b = 0; b < nb; b++)
     {
      double n = use_buckets[b].net;
      sum_net += n;
      if(n > 0.0001)
        {
         wins++;
         gross_profit += n;
         if(n > largest_win)
            largest_win = n;
        }
      else
         if(n < -0.0001)
           {
            losses++;
            gross_loss += -n;
            if(n < largest_loss)
               largest_loss = n;
           }
     }

   int total_trades = wins + losses;
   double win_rate = (total_trades > 0 ? 100.0 * wins / total_trades : 0);
   double profit_factor = (gross_loss > 0.0001 ? gross_profit / gross_loss : 0);
   double expected_payoff = (total_trades > 0 ? sum_net / total_trades : 0);

   double balance_now = AccountInfoDouble(ACCOUNT_BALANCE);
   double floating    = AccountInfoDouble(ACCOUNT_PROFIT);

   // Compute the true window-start balance by subtracting *every* trade deal
   // in the window (ours + manual) from current balance. This gives the real
   // account balance at the beginning of the window, unpolluted by manual trades.
   double all_trade_net_window = 0.0;
   double manual_floating = 0.0;
   for(int i = 0; i < total_deals; i++)
     {
      ulong ticket = HistoryDealGetTicket(i);
      if(ticket == 0)
         continue;
      long dtype = HistoryDealGetInteger(ticket, DEAL_TYPE);
      // Only trade deals (BUY/SELL), skip balance ops / credits / etc.
      if(dtype != DEAL_TYPE_BUY && dtype != DEAL_TYPE_SELL)
         continue;
      double dp = HistoryDealGetDouble(ticket, DEAL_PROFIT);
      double dc = HistoryDealGetDouble(ticket, DEAL_COMMISSION);
      double ds = HistoryDealGetDouble(ticket, DEAL_SWAP);
      double df = HistoryDealGetDouble(ticket, DEAL_FEE);
      all_trade_net_window += dp + dc + ds + df;
     }
   // Also exclude floating P&L of any currently-open non-EA (manual) positions
   for(int pi = 0; pi < PositionsTotal(); pi++)
     {
      ulong pt = PositionGetTicket(pi);
      if(pt == 0 || !PositionSelectByTicket(pt))
         continue;
      string pcmt = PositionGetString(POSITION_COMMENT);
      if(IsOurEA(pcmt))
         continue;
      manual_floating += PositionGetDouble(POSITION_PROFIT) + PositionGetDouble(POSITION_SWAP);
     }

   double clean_start_bal = balance_now - all_trade_net_window;
   double ea_floating     = floating - manual_floating;
   // Displayed equity: EA-only trajectory projected from the real window-start balance.
   double equity_now  = clean_start_bal + sum_net + ea_floating;
   double start_bal   = clean_start_bal;

   double eq[];
   datetime eq_times[];
   ArrayResize(eq, ArraySize(deals));
   ArrayResize(eq_times, ArraySize(deals));
   double run = start_bal;
   for(int j = 0; j < ArraySize(deals); j++)
     {
      run += deals[j].net;
      eq[j] = run;
      eq_times[j] = deals[j].tm;
     }

   double max_dd_pct = 0;
   double max_dd_abs = 0;
   double peak_eq = start_bal;
   for(int j = 0; j < ArraySize(eq); j++)
     {
      if(eq[j] > peak_eq)
         peak_eq = eq[j];
      double dda = peak_eq - eq[j];
      if(dda > max_dd_abs)
         max_dd_abs = dda;
      if(peak_eq > 0.0001)
        {
         double dd = 100.0 * dda / peak_eq;
         if(dd > max_dd_pct)
            max_dd_pct = dd;
        }
     }

   double recovery_factor = (max_dd_abs > 0.0001 ? sum_net / max_dd_abs : 0);

   int max_cw = 0;
   int max_cl = 0;
   int run_w = 0;
   int run_l = 0;
   for(int a = 0; a < nb - 1; a++)
      for(int b = 0; b < nb - a - 1; b++)
         if(use_buckets[b].t_last > use_buckets[b + 1].t_last)
           {
            PosBucket tmp = use_buckets[b];
            use_buckets[b] = use_buckets[b + 1];
            use_buckets[b + 1] = tmp;
           }
   for(int b = 0; b < nb; b++)
     {
      double n = use_buckets[b].net;
      if(n > 0.0001)
        {
         run_w++;
         if(run_l > max_cl)
            max_cl = run_l;
         run_l = 0;
        }
      else
         if(n < -0.0001)
           {
            run_l++;
            if(run_w > max_cw)
               max_cw = run_w;
            run_w = 0;
           }
     }
   if(run_w > max_cw)
      max_cw = run_w;
   if(run_l > max_cl)
      max_cl = run_l;

   double avg_win = (wins > 0 ? gross_profit / wins : 0);
   double avg_loss = (losses > 0 ? -gross_loss / losses : 0);

   int cur_trades = 0;
   double cur_net = 0;
   int cur_wins = 0;
   int old_trades = 0;

   for(int b = 0; b < nb; b++)
     {
      if(IsCurrentEP(use_buckets[b].comment))
        {
         cur_trades++;
         cur_net += use_buckets[b].net;
         if(use_buckets[b].net > 0.0001)
            cur_wins++;
        }
      else
         old_trades++;
     }

   double cur_wr = (cur_trades > 0 ? 100.0 * cur_wins / cur_trades : 0);

   string models_json = "[";
   bool first_m = true;
   string mnames[] = {"EdgePredictor Midas", "EdgePredictor Phantom", "EdgePredictor Oracle"};
   int n_models = ArraySize(mnames);
   for(int mi = 0; mi < n_models; mi++)
     {
      string mn = mnames[mi];
      int mt = 0, mw = 0;
      double mn_net = 0, mbest = -1e100, mworst = 1e100;
      for(int b = 0; b < nb; b++)
        {
         string bcmt = use_buckets[b].comment;
         string bcanon = ModelCanonical(bcmt);
         bool match = (bcanon == mn);
         if(!match)
            continue;
         mt++;
         double n = use_buckets[b].net;
         mn_net += n;
         if(n > 0.0001)
            mw++;
         if(n > mbest)
            mbest = n;
         if(n < mworst)
            mworst = n;
        }
      if(mt == 0)
         continue;
      if(!first_m)
         models_json += ",";
      first_m = false;
      double wr = 100.0 * mw / mt;
      if(mbest < -1e50)
         mbest = 0;
      if(mworst > 1e50)
         mworst = 0;
      models_json += StringFormat(
         "{\"name\":\"%s\",\"color\":\"%s\",\"trades\":%d,\"wins\":%d,\"losses\":%d,\"win_rate\":%.1f,\"gross_pnl\":%.2f,\"net_pnl\":%.2f,\"best\":%.2f,\"worst\":%.2f}",
         mn, ModelColor(mn), mt, mw, mt - mw, wr, mn_net, mn_net, mbest, mworst);
     }
   models_json += "]";

   // Always append current equity as final anchor point so chart endpoint = header value.
   int eq_size = ArraySize(eq);
   ArrayResize(eq, eq_size + 1);
   ArrayResize(eq_times, eq_size + 1);
   eq[eq_size]       = equity_now;
   eq_times[eq_size] = TimeCurrent();
   eq_size++;

   string eq_json = "[";
   string dt_json = "[";
   for(int j = 0; j < eq_size; j++)
     {
      if(j > 0)
        {
         eq_json += ",";
         dt_json += ",";
        }
      eq_json += DoubleToString(eq[j], 2);
      string ds = TimeToString(eq_times[j], TIME_DATE | TIME_MINUTES);
      StringReplace(ds, ".", "-");
      dt_json += "\"" + ds + "\"";
     }
   eq_json += "]";
   dt_json += "]";

   // --- Per-model equity curves (all EA versions mapped to canonical names) ---
   string mc_json = "{";
   bool first_mc = true;
   for(int mi = 0; mi < n_models; mi++)
     {
      string mn = mnames[mi];
      PosBucket mbk[];
      int mbk_n = 0;
      for(int b = 0; b < nb; b++)
        {
         string canon = ModelCanonical(use_buckets[b].comment);
         if(canon == mn)
           {
            ArrayResize(mbk, mbk_n + 1);
            mbk[mbk_n] = use_buckets[b];
            mbk_n++;
           }
        }
      if(mbk_n == 0)
         continue;

      // Sort model buckets by close time
      for(int a = 0; a < mbk_n - 1; a++)
         for(int bb = 0; bb < mbk_n - a - 1; bb++)
            if(mbk[bb].t_last > mbk[bb + 1].t_last)
              {
               PosBucket tmp2 = mbk[bb];
               mbk[bb] = mbk[bb + 1];
               mbk[bb + 1] = tmp2;
              }

      // Build cumulative P&L curve
      string mc_eq = "[";
      string mc_dt = "[";
      double mc_cum = 0;
      int mc_wins = 0;
      int mc_losses = 0;
      double mc_best = -1e100;
      double mc_worst = 1e100;
      for(int k = 0; k < mbk_n; k++)
        {
         mc_cum += mbk[k].net;
         if(mbk[k].net > 0.0001)
            mc_wins++;
         else
            if(mbk[k].net < -0.0001)
               mc_losses++;
         if(mbk[k].net > mc_best)
            mc_best = mbk[k].net;
         if(mbk[k].net < mc_worst)
            mc_worst = mbk[k].net;
         if(k > 0)
           {
            mc_eq += ",";
            mc_dt += ",";
           }
         mc_eq += DoubleToString(mc_cum, 2);
         string mds = TimeToString(mbk[k].t_last, TIME_DATE | TIME_MINUTES);
         StringReplace(mds, ".", "-");
         mc_dt += "\"" + mds + "\"";
        }

      double model_floating = 0.0;
      for(int pi = 0; pi < PositionsTotal(); pi++)
        {
         ulong pt = PositionGetTicket(pi);
         if(pt == 0 || !PositionSelectByTicket(pt))
            continue;

         string psym = PositionGetString(POSITION_SYMBOL);
         if(!IsOurSymbol(psym))
            continue;

         string pcmt = PositionGetString(POSITION_COMMENT);
         if(!IsOurEA(pcmt))
            continue;

         string live_model = ModelCanonical(pcmt);
         if(StringLen(live_model) == 0)
            continue;  // skip unrecognized trades

         if(live_model != mn)
            continue;

         model_floating += PositionGetDouble(POSITION_PROFIT) + PositionGetDouble(POSITION_SWAP);
        }

      double mc_live = mc_cum + model_floating;
      if(mbk_n > 0)
        {
         mc_eq += ",";
         mc_dt += ",";
        }
      mc_eq += DoubleToString(mc_live, 2);
      string mnow = TimeToString(TimeCurrent(), TIME_DATE | TIME_MINUTES);
      StringReplace(mnow, ".", "-");
      mc_dt += "\"" + mnow + "\"";

      mc_eq += "]";
      mc_dt += "]";

      double mc_wr = (mbk_n > 0 ? 100.0 * mc_wins / mbk_n : 0);
      if(mc_best < -1e50) mc_best = 0;
      if(mc_worst > 1e50) mc_worst = 0;

      if(!first_mc)
         mc_json += ",";
      first_mc = false;

      mc_json += "\"" + JsonEscape(mn) + "\":{";
      mc_json += "\"dates\":" + mc_dt + ",";
      mc_json += "\"equity_curve\":" + mc_eq + ",";
      mc_json += StringFormat("\"trades\":%d,", mbk_n);
      mc_json += StringFormat("\"wins\":%d,", mc_wins);
      mc_json += StringFormat("\"losses\":%d,", mbk_n - mc_wins);
      mc_json += StringFormat("\"win_rate\":%.1f,", mc_wr);
      mc_json += StringFormat("\"net_pnl\":%.2f,", mc_live);
      mc_json += StringFormat("\"best\":%.2f,", mc_best);
      mc_json += StringFormat("\"worst\":%.2f,", mc_worst);
      mc_json += "\"color\":\"" + ModelColor(mn) + "\"";
      mc_json += "}";
     }
   mc_json += "}";

   string open_json = "[";
   int opn = 0;
   for(int pi = 0; pi < PositionsTotal(); pi++)
     {
      ulong pt = PositionGetTicket(pi);
      if(pt == 0 || !PositionSelectByTicket(pt))
         continue;
      string psym = PositionGetString(POSITION_SYMBOL);
      if(!IsOurSymbol(psym))
         continue;
      string pcmt = PositionGetString(POSITION_COMMENT);
      if(!IsOurEA(pcmt))
         continue;

      string mdl = ModelCanonical(pcmt);
      if(StringLen(mdl) == 0)
         continue;

      long ptype = PositionGetInteger(POSITION_TYPE);
      string dir = (ptype == POSITION_TYPE_BUY ? "Buy" : "Sell");
      double entry = PositionGetDouble(POSITION_PRICE_OPEN);
      double bid = SymbolInfoDouble(psym, SYMBOL_BID);
      double pnl = PositionGetDouble(POSITION_PROFIT) + PositionGetDouble(POSITION_SWAP);

      if(opn > 0)
         open_json += ",";
      open_json += StringFormat(
         "{\"model\":\"%s\",\"direction\":\"%s\",\"entry\":%.5f,\"current\":%.5f,\"pnl\":%.2f,\"color\":\"%s\"}",
         mdl, dir, entry, bid, pnl, ModelColor(mdl));
      opn++;
     }
   open_json += "]";

   // ── Recent closed trades (for live-ticker on the website) ──
   // Take the N most-recently-closed buckets, emit one JSON object each
   // with real model/direction/pnl/close_time. Uses `buckets` (full history
   // when InpTodayOnly=false, else use_buckets which is today-only).
   const int RECENT_LIMIT = 30;
   PosBucket src[];
   int src_n = ArraySize(buckets);
   ArrayResize(src, src_n);
   for(int b = 0; b < src_n; b++)
      src[b] = buckets[b];
   // Sort descending by t_last (simple O(n^2) — n is small)
   for(int i = 0; i < src_n - 1; i++)
      for(int j = 0; j < src_n - i - 1; j++)
         if(src[j].t_last < src[j + 1].t_last)
           {
            PosBucket tmp = src[j];
            src[j] = src[j + 1];
            src[j + 1] = tmp;
           }
   string recent_json = "[";
   int rc_n = 0;
   for(int b = 0; b < src_n && rc_n < RECENT_LIMIT; b++)
     {
      string mdl = ModelCanonical(src[b].comment);
      if(StringLen(mdl) == 0)
         continue;
      if(rc_n > 0)
         recent_json += ",";
      recent_json += StringFormat(
         "{\"model\":\"%s\",\"direction\":\"%s\",\"pnl\":%.2f,\"time_close\":\"%s\",\"color\":\"%s\"}",
         mdl,
         src[b].is_buy ? "Buy" : "Sell",
         src[b].net,
         FmtIsoTimestamp(src[b].t_last),
         ModelColor(mdl));
      rc_n++;
     }
   recent_json += "]";

   long acct = AccountInfoInteger(ACCOUNT_LOGIN);
   string period_str = FmtDateShort(t_min) + " – " + FmtDateShort(t_max);
   string report_date = FmtIsoDate(TimeGMT());

   string json = "{";
   json += "\"account\":\"" + IntegerToString(acct) + "\",";
   json += "\"report_date\":\"" + report_date + "\",";
   if(InpTodayOnly)
      json += "\"merge_history\":true,";
   json += "\"period\":\"" + JsonEscape(period_str) + "\",";
   json += "\"period_start\":\"" + FmtIsoDate(t_min) + "\",";
   json += "\"period_end\":\"" + FmtIsoDate(t_max) + "\",";
   json += StringFormat("\"total_trades\":%d,", total_trades);
   json += StringFormat("\"win_rate\":%.2f,", win_rate);
   json += StringFormat("\"profit_factor\":%.2f,", profit_factor);
   json += StringFormat("\"net_profit\":%.2f,", sum_net);
   json += StringFormat("\"max_drawdown_pct\":%.2f,", max_dd_pct);
   json += StringFormat("\"recovery_factor\":%.2f,", recovery_factor);
   json += StringFormat("\"profit_trades\":%d,", wins);
   json += StringFormat("\"loss_trades\":%d,", losses);
   json += StringFormat("\"avg_win\":%.2f,", avg_win);
   json += StringFormat("\"avg_loss\":%.2f,", avg_loss);
   json += StringFormat("\"expected_payoff\":%.2f,", expected_payoff);
   json += StringFormat("\"largest_win\":%.2f,", largest_win);
   json += StringFormat("\"largest_loss\":%.2f,", largest_loss);
   json += StringFormat("\"max_consec_wins\":%d,", max_cw);
   json += StringFormat("\"max_consec_losses\":%d,", max_cl);
   json += "\"models\":" + models_json + ",";
   json += StringFormat("\"current_version_trades\":%d,", cur_trades);
   json += StringFormat("\"current_version_win_rate\":%.1f,", cur_wr);
   json += StringFormat("\"current_version_net\":%.2f,", cur_net);
   json += StringFormat("\"older_version_trades\":%d,", old_trades);
   json += "\"open_positions\":" + open_json + ",";
   json += "\"recent_closed\":" + recent_json + ",";
   json += StringFormat("\"floating_pnl\":%.2f,", floating);
   json += "\"equity_curve\":" + eq_json + ",";
   json += "\"dates\":" + dt_json + ",";
   json += StringFormat("\"starting_balance\":%.2f,", start_bal);
   json += StringFormat("\"final_balance\":%.2f,", equity_now);
   json += StringFormat("\"balance\":%.2f,", balance_now);
   json += StringFormat("\"floating_equity\":%.2f,", floating);
   json += "\"model_curves\":" + mc_json;
   json += "}";

   string url = InpServerURL + "/admin/forward-test";
   int _sec_len = StringLen(InpReportSecret);
   PrintFormat("EdgePredictor_Reporter: secret_len=%d first6=%s last6=%s",
               _sec_len,
               (_sec_len >= 6 ? StringSubstr(InpReportSecret, 0, 6) : InpReportSecret),
               (_sec_len >= 6 ? StringSubstr(InpReportSecret, _sec_len - 6, 6) : ""));
   string headers = "Content-Type: application/json; charset=utf-8\r\nX-Report-Secret: " + InpReportSecret + "\r\n";

   uchar post[];
   uchar result[];
   string resp_headers;
   int converted = StringToCharArray(json, post, 0, WHOLE_ARRAY, CP_UTF8);
   if(converted < 1)
     {
      Print("EdgePredictor_Reporter: StringToCharArray failed");
      return;
     }
   ArrayResize(post, converted - 1);

   ResetLastError();
   int res = WebRequest("POST", url, headers, 30000, post, result, resp_headers);
   if(res == 200)
     {
      if(InpVerbose)
         PrintFormat("EdgePredictor_Reporter: OK (%s) %d closed positions, %d equity points",
                     InpTodayOnly ? "today broker day" : "full range",
                     total_trades, ArraySize(eq));
     }
   else
     {
      Print("EdgePredictor_Reporter: HTTP ", res, " err=", GetLastError(), " body=", CharArrayToString(result));
     }
  }

//+------------------------------------------------------------------+
int OnInit()
  {
   EventSetTimer(InpIntervalSec);
   SendReport();
   return INIT_SUCCEEDED;
  }

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   EventKillTimer();
  }

//+------------------------------------------------------------------+
void OnTimer()
  {
   SendReport();
  }

//+------------------------------------------------------------------+
void OnTick()
  {
  }

//+------------------------------------------------------------------+
