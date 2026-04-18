//+------------------------------------------------------------------+
//| trace_logger.mqh — logs every rule evaluation to CSV so we can   |
//| diff live feature computation against the Python pipeline.       |
//| Writes to MQL5/Files/trace_<symbol>.csv (semicolon separated).   |
//+------------------------------------------------------------------+
#ifndef __TRACE_LOGGER_MQH__
#define __TRACE_LOGGER_MQH__

int g_trace_handle = INVALID_HANDLE;

bool TraceLogger_Open(string filename)
{
   g_trace_handle = FileOpen(filename, FILE_WRITE|FILE_READ|FILE_CSV|FILE_ANSI, ';');
   if(g_trace_handle == INVALID_HANDLE)
   { Print("TraceLogger: failed to open ", filename, " err=", GetLastError()); return false; }
   FileSeek(g_trace_handle, 0, SEEK_END);
   if(FileTell(g_trace_handle) == 0)
      FileWrite(g_trace_handle,
                "bar_time","symbol","cluster","rule_id","rule_name",
                "onnx_file","direction","prob","threshold","confirmed","features");
   return true;
}

void TraceLogger_Log(datetime bar_time, int cluster, int rule_id, string rule_name,
                     string onnx_file, int direction, double prob, double threshold,
                     bool confirmed, const float &feat[])
{
   if(g_trace_handle == INVALID_HANDLE) return;
   string fs = "";
   for(int i = 0; i < ArraySize(feat); i++)
   {
      if(i > 0) fs += ",";
      fs += DoubleToString((double)feat[i], 10);
   }
   FileWrite(g_trace_handle,
             TimeToString(bar_time, TIME_DATE|TIME_SECONDS),
             _Symbol,
             IntegerToString(cluster),
             IntegerToString(rule_id),
             rule_name,
             onnx_file,
             IntegerToString(direction),
             DoubleToString(prob, 6),
             DoubleToString(threshold, 4),
             confirmed ? "1" : "0",
             fs);
   FileFlush(g_trace_handle);
}

void TraceLogger_Close()
{
   if(g_trace_handle != INVALID_HANDLE)
   { FileClose(g_trace_handle); g_trace_handle = INVALID_HANDLE; }
}

#endif
