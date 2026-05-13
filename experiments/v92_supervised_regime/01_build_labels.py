"""v92 — supervised "current regime" classifier.

Idea: instead of K-means on a backward window (which lags and repaints),
predict the regime from current-bar features against a forward-looking
ground truth label. Pointwise -> zero repainting.

Label definition (forward 12h = 144 bars on M5):
  - forward_ret = (C[t+144] - C[t]) / C[t]
  - forward_atr_norm = std of 5-min returns over t..t+144  (vol intensity)
  - thresholds (XAU):
      |forward_ret| < 0.2% AND vol_low      -> 2 TrendRange
      forward_ret > +0.3%                    -> 0 Uptrend
      forward_ret < -0.3%                    -> 3 Downtrend
      vol_high AND |forward_ret| small       -> 1 MeanRevert
      vol_high AND |forward_ret| big         -> 4 HighVol (rare combo: directional but choppy)

Features (current-bar only, NO forward leak):
  - V72L per-bar physics features
  - 24h-momentum: ret_24h_signed, ret_24h_abs
  - maturity: stretch_100, stretch_200, pct_to_extreme_50 (direction-agnostic abs)
  - short returns: ret_1h, ret_4h
  - realized vol: std of 5-min ret over last 1h/4h/24h
  - hour_enc, dow_enc (already in V72L)

This script just BUILDS the labeled dataset and saves to parquet.
Training is script 02.
"""
import os, numpy as np, pandas as pd
PROJECT = "/home/jay/Desktop/new-model-zigzag"
DATA = f"{PROJECT}/data"
OUT = f"{PROJECT}/experiments/v92_supervised_regime"

FWD_BARS = 144            # 12h forward window for label
RET_UP_THR = 0.003        # 0.3%
RET_DOWN_THR = -0.003
RET_RANGE_THR = 0.002     # 0.2% = "range"
VOL_HIGH_QUANTILE = 0.75  # top quartile of forward 5-min ret std

V72L_COLS = ['hurst_rs','ou_theta','entropy_rate','kramers_up','wavelet_er',
             'vwap_dist','hour_enc','dow_enc','quantum_flow','quantum_flow_h4',
             'quantum_momentum','quantum_vwap_conf','quantum_divergence','quantum_div_strength',
             'vpin','sig_quad_var','har_rv_ratio','hawkes_eta']

def build(name, swing_csv, v72l_glob_template):
    print(f"\n=== {name} ===")
    sw = pd.read_csv(swing_csv, parse_dates=["time"]).sort_values("time")\
           .drop_duplicates('time', keep='last').reset_index(drop=True)
    n = len(sw)
    C = sw['close'].values.astype(np.float64)
    print(f"  bars: {n:,}  range {sw.time.min()} -> {sw.time.max()}")

    # --- Forward-looking labels ---
    fwd_ret = np.full(n, np.nan)
    fwd_vol = np.full(n, np.nan)
    for t in range(n - FWD_BARS):
        c0 = C[t]; c1 = C[t + FWD_BARS]
        if c0 <= 0: continue
        fwd_ret[t] = (c1 - c0) / c0
        seg = C[t:t+FWD_BARS+1]
        r = np.diff(seg) / seg[:-1]
        fwd_vol[t] = float(np.std(r))

    valid = ~np.isnan(fwd_ret)
    vol_thr = np.nanquantile(fwd_vol[valid], VOL_HIGH_QUANTILE)
    print(f"  forward 12h: vol_high threshold (top 25%) = {vol_thr:.5f}")

    label = np.full(n, -1, dtype=np.int8)
    for t in range(n):
        if not valid[t]: continue
        r = fwd_ret[t]; v = fwd_vol[t]; ar = abs(r)
        vol_high = v >= vol_thr
        if ar < RET_RANGE_THR and not vol_high:
            label[t] = 2     # TrendRange
        elif r >= RET_UP_THR and not vol_high:
            label[t] = 0     # Uptrend
        elif r <= RET_DOWN_THR and not vol_high:
            label[t] = 3     # Downtrend
        elif vol_high and ar >= RET_UP_THR:
            label[t] = 4     # HighVol directional
        else:
            label[t] = 1     # MeanRevert (choppy / weak directional)

    # --- Current-bar features (NO forward leak) ---
    rets1 = np.zeros(n); rets1[1:] = np.diff(C) / C[:-1]
    def rolling_std(arr, w):
        s = pd.Series(arr).rolling(w, min_periods=w).std().values
        return np.nan_to_num(s, nan=0.0)
    feat = pd.DataFrame({'time': sw['time']})
    feat['ret_1h']   = pd.Series(C).pct_change(12).fillna(0).values
    feat['ret_4h']   = pd.Series(C).pct_change(48).fillna(0).values
    feat['ret_24h']  = pd.Series(C).pct_change(288).fillna(0).values
    feat['vol_1h']   = rolling_std(rets1, 12)
    feat['vol_4h']   = rolling_std(rets1, 48)
    feat['vol_24h']  = rolling_std(rets1, 288)
    feat['ret_24h_abs'] = feat['ret_24h'].abs()
    # stretch features
    for L in [100, 200]:
        feat[f'stretch_{L}_long']  = 0.0
        feat[f'stretch_{L}_short'] = 0.0
    # vectorized stretches
    high_roll = pd.Series(C).rolling(200, min_periods=200).max().values
    low_roll  = pd.Series(C).rolling(200, min_periods=200).min().values
    high100 = pd.Series(C).rolling(100, min_periods=100).max().values
    low100  = pd.Series(C).rolling(100, min_periods=100).min().values
    # ATR(14) on close as scale
    diffs = np.abs(np.diff(C)); atr = np.concatenate([[0], pd.Series(diffs).rolling(14, min_periods=14).mean().fillna(0).values])
    atr_safe = np.where(atr > 0, atr, np.nan)
    feat['stretch_100_long']  = np.nan_to_num((C - low100) / atr_safe, nan=0.0)
    feat['stretch_100_short'] = np.nan_to_num((high100 - C) / atr_safe, nan=0.0)
    feat['stretch_200_long']  = np.nan_to_num((C - low_roll) / atr_safe, nan=0.0)
    feat['stretch_200_short'] = np.nan_to_num((high_roll - C) / atr_safe, nan=0.0)

    # Merge V72L features if present in swing csv (some have them)
    have_v72l = [c for c in V72L_COLS if c in sw.columns]
    print(f"  V72L cols found in bars: {len(have_v72l)}/{len(V72L_COLS)}")
    for c in have_v72l:
        feat[c] = sw[c].fillna(0).values

    feat['label'] = label
    feat['fwd_ret'] = fwd_ret
    feat['fwd_vol'] = fwd_vol
    feat = feat[feat['label'] >= 0].reset_index(drop=True)

    print(f"  labeled rows: {len(feat):,}")
    print(f"  label distribution:")
    NAMES = {0:'Uptrend',1:'MeanRevert',2:'TrendRange',3:'Downtrend',4:'HighVol'}
    for k in range(5):
        c = int((feat['label']==k).sum())
        print(f"    {k}={NAMES[k]:<10}  {c:>6}  ({c*100/len(feat):.1f}%)")

    out_path = f"{OUT}/labeled_{name.lower()}.parquet"
    feat.to_parquet(out_path, index=False)
    print(f"  wrote {out_path}")

if __name__ == "__main__":
    build("XAU", f"{DATA}/swing_v5_xauusd.csv", None)
    build("BTC", f"{DATA}/swing_v5_btc.csv", None)
