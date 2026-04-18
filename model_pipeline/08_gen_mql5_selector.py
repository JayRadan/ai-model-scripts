"""
Generate a self-contained MQL5 header with the regime selector embedded as
constants. The EA can then classify a week into a cluster using only C-style
math (no second ONNX model needed for regime detection).
"""
import json
import numpy as np

import paths as P

with open(P.data("regime_selector_K4.json")) as f:
    sel = json.load(f)

# Cluster table — name + tradeable flag. The 4 clusters are fixed by build-selector.
# All 4 clusters are tradeable in v5+: C2 now has R2b_v_reversal.
CLUSTER_TABLE = [
    {"id": 0, "name": "Uptrend",    "tradeable": True,  "threshold": 0.40},
    {"id": 1, "name": "MeanRevert", "tradeable": True,  "threshold": 0.40},
    {"id": 2, "name": "TrendRange", "tradeable": True,  "threshold": 0.40},
    {"id": 3, "name": "Downtrend",  "tradeable": True,  "threshold": 0.40},
    {"id": 4, "name": "HighVol",    "tradeable": True,  "threshold": 0.40},
]

K = sel["K"]
FEATS = sel.get("feat_names", sel.get("feature_columns", sel.get("fingerprint_cols")))
scaler_mean = np.asarray(sel["scaler_mean"])
scaler_std  = np.asarray(sel["scaler_std"])
pre_mean    = np.asarray(sel.get("pca_mean", sel.get("pre_pca_mean")))
pre_comp    = np.asarray(sel.get("pca_components", sel.get("pre_pca_components")))
centroids   = np.asarray(sel.get("centroids", sel.get("centroids_pre_pca")))
NF          = len(FEATS)

def fmt(v): return f"{float(v):+.15g}"
def arr1(v): return "{" + ", ".join(fmt(x) for x in v) + "}"
def arr2(m, indent="    "):
    rows = [",\n".join(indent + arr1(r) for r in m)]
    return "{\n" + rows[0] + "\n}"

cluster_info = [(c["id"], c["name"],
                 c["threshold"] if c["tradeable"] else 0.0,
                 [] if not c["tradeable"] else [0,1,2])
                for c in CLUSTER_TABLE]

lines = []
lines.append("//+------------------------------------------------------------------+")
lines.append("//| regime_selector.mqh                                               |")
lines.append("//| Auto-generated from regime_selector_K4.json. DO NOT EDIT BY HAND. |")
lines.append("//|                                                                   |")
lines.append("//| Usage:                                                            |")
lines.append("//|   double fp[7];                                                   |")
lines.append("//|   ComputeWeekFingerprint(last_week_bars, fp);                     |")
lines.append("//|   int cluster = ClassifyRegime(fp);                               |")
lines.append("//+------------------------------------------------------------------+")
lines.append("#ifndef __REGIME_SELECTOR_MQH__")
lines.append("#define __REGIME_SELECTOR_MQH__")
lines.append("")
lines.append(f"#define REGIME_K              {K}")
lines.append(f"#define REGIME_N_FEATS        {NF}")
lines.append("")
lines.append("// Fingerprint feature order (compute in this exact order)")
for i, name in enumerate(FEATS):
    lines.append(f"// [{i}] {name}")
lines.append("")
lines.append("// Cluster meta — 0:Ranging 1:Downtrend 2:Shock(SKIP) 3:Uptrend")
for cid, name, thr, classes in cluster_info:
    lines.append(f"// C{cid} {name}: threshold={thr}, classes={classes}")
lines.append("")
lines.append(f"const double REGIME_SCALER_MEAN[REGIME_N_FEATS] = {arr1(scaler_mean)};")
lines.append(f"const double REGIME_SCALER_STD [REGIME_N_FEATS] = {arr1(scaler_std)};")
lines.append("")
lines.append(f"const double REGIME_PCA_MEAN[REGIME_N_FEATS] = {arr1(pre_mean)};")
lines.append("")
lines.append("// Pre-PCA rotation matrix: rows = output dims, cols = input dims.")
lines.append("// To rotate: for(i in 0..NF) out[i] = sum_j (scaled[j] - mean[j]) * comp[i][j]")
lines.append(f"const double REGIME_PCA_COMP[REGIME_N_FEATS][REGIME_N_FEATS] = {arr2(pre_comp)};")
lines.append("")
lines.append("// Cluster centroids in post-PCA space")
lines.append(f"const double REGIME_CENTROIDS[REGIME_K][REGIME_N_FEATS] = {arr2(centroids)};")
lines.append("")
lines.append("// Cluster names (for logging)")
names_literal = "{" + ", ".join(f'"{c[1]}"' for c in cluster_info) + "}"
lines.append(f"const string REGIME_NAMES[REGIME_K] = {names_literal};")
lines.append("")
lines.append("// Is this cluster tradeable (has a model)?")
tradeable = [1 if c[2] > 0 else 0 for c in cluster_info]
lines.append(f"const int REGIME_TRADEABLE[REGIME_K] = {{{', '.join(str(x) for x in tradeable)}}};")
lines.append("")
lines.append("// Per-cluster probability threshold (0 = skip)")
thrs = [f"{c[2]:.4f}" for c in cluster_info]
lines.append(f"const double REGIME_THRESHOLD[REGIME_K] = {{{', '.join(thrs)}}};")
lines.append("")
lines.append("//+------------------------------------------------------------------+")
lines.append("//| ComputeWeekFingerprint — fill 7-dim fp[] from week of OHLC bars  |")
lines.append("//| `bars` is AS-SERIES, [0] = most recent bar of the week.          |")
lines.append("//+------------------------------------------------------------------+")
lines.append("""void ComputeWeekFingerprint(const MqlRates &bars[], double &fp[])
{
   int n = ArraySize(bars);
   if(n < 2) { ArrayInitialize(fp, 0.0); return; }

   // Re-index oldest-first for clarity
   double sum_ret = 0.0, sum_ret2 = 0.0;
   double sum_range_pct = 0.0;
   double max_h = bars[n-1].high;
   double min_l = bars[n-1].low;
   double mean_close = 0.0;
   int n_ret = n - 1;
   int pos_ret = 0;

   // First pass: returns, range pct, max/min
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

   // Trend consistency: fraction of returns with sign(r) == sign(mean_ret)
   double trend_consistency = 0.5;
   if(MathAbs(mean_ret) > 1e-12)
      trend_consistency = (mean_ret > 0) ? (double)pos_ret / n_ret : (double)neg_ret / n_ret;

   // Return autocorrelation (lag 1)
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

   fp[0] = sum_ret;                                       // weekly_return_pct
   fp[1] = std_ret;                                       // volatility_pct
   fp[2] = trend_consistency;
   fp[3] = sum_ret / (std_ret + 1e-9);                    // trend_strength
   fp[4] = mean_bar_range;                                // volatility
   fp[5] = (max_h - min_l) / mean_close / (mean_bar_range + 1e-9);  // range_vs_atr
   fp[6] = r_autocorr;
}

//+------------------------------------------------------------------+
//| ClassifyRegime — fingerprint → cluster id (0..K-1)                |
//+------------------------------------------------------------------+
int ClassifyRegime(const double &fp[])
{
   // 1. Standardize
   double scaled[REGIME_N_FEATS];
   for(int i = 0; i < REGIME_N_FEATS; ++i)
      scaled[i] = (fp[i] - REGIME_SCALER_MEAN[i]) / REGIME_SCALER_STD[i];

   // 2. Centre by PCA mean then rotate
   double rotated[REGIME_N_FEATS];
   for(int i = 0; i < REGIME_N_FEATS; ++i)
   {
      double s = 0.0;
      for(int j = 0; j < REGIME_N_FEATS; ++j)
         s += (scaled[j] - REGIME_PCA_MEAN[j]) * REGIME_PCA_COMP[i][j];
      rotated[i] = s;
   }

   // 3. Nearest centroid (squared L2)
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

#endif  // __REGIME_SELECTOR_MQH__""")

out = str(P.LIVE_DIR / "MQL5_Include" / "regime_selector.mqh")
import os
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    f.write("\n".join(lines))
print(f"Wrote {out}")
