"""Sweep multi-head exit thresholds for BTC — find optimal policy.

Loads the trained BTC multi-head bundle, runs bar-level simulation
on the full test set, and sweeps threshold combinations.
"""
import pandas as pd, numpy as np, pickle, glob as _glob, time as _time, os, itertools
from xgboost import XGBClassifier

PROJECT = "/home/jay/Desktop/new-model-zigzag"
DATA = f"{PROJECT}/data"
MODEL_DIR = "/home/jay/Desktop/my-agents-and-website/commercial/server/decision_engine/models"
BUNDLE_PATH = os.path.join(MODEL_DIR, "multi_head_exit_oracle_btc.pkl")

V72L_FEATS = ['hurst_rs','ou_theta','entropy_rate','kramers_up','wavelet_er',
    'vwap_dist','hour_enc','dow_enc','quantum_flow','quantum_flow_h4',
    'quantum_momentum','quantum_vwap_conf','quantum_divergence','quantum_div_strength',
    'vpin','sig_quad_var','har_rv_ratio','hawkes_eta']
MAX_HOLD=60; MIN_HOLD=2; SL_HARD=-4.0

print("="*60)
print("  BTC Multi-Head Threshold Sweep")
print("="*60)

# ═══ 1. Load data ═══
print("[1/4] Loading BTC data + model...", end='', flush=True); t0=_time.time()

with open(BUNDLE_PATH,'rb') as f:
    bundle = pickle.load(f)
models = bundle['models']

swing=pd.read_csv(f"{DATA}/swing_v5_btc.csv",parse_dates=["time"])
swing=swing.sort_values("time").drop_duplicates('time',keep='last').reset_index(drop=True)
n=len(swing); C=swing["close"].values.astype(np.float64)
H=swing["high"].values; L=swing["low"].values
tr=np.concatenate([[H[0]-L[0]],np.maximum.reduce([H[1:]-L[1:],np.abs(H[1:]-C[:-1]),np.abs(L[1:]-C[:-1])])])
atr=pd.Series(tr).rolling(14,min_periods=14).mean().values
t2i={t:i for i,t in enumerate(swing["time"].values)}

all_setups=[]
for f in sorted(_glob.glob(f"{DATA}/setups_*_v72l_btc.csv")):
    df=pd.read_csv(f,parse_dates=["time"]); all_setups.append(df)
all_df=pd.concat(all_setups,ignore_index=True)
phys=all_df[['time']+V72L_FEATS].drop_duplicates('time',keep='last').sort_values('time')
swing=pd.merge_asof(swing.sort_values('time'),phys,on='time',direction='nearest')
for col in V72L_FEATS: swing[col]=swing[col].fillna(0)
ctx_arr=swing[V72L_FEATS].fillna(0).values.astype(np.float64)

trades=pd.read_csv(f"{PROJECT}/experiments/v84_rl_entry/btc_rl_trades.csv",parse_dates=["time"])
trades=trades.sort_values("time").reset_index(drop=True)

# Use FULL trade set for sweep (not just test split)
print(f" {_time.time()-t0:.0f}s — {len(trades):,} trades",flush=True)

# ═══ 2. Pre-compute per-bar features for all trades ═══
print("[2/4] Pre-computing bar features...", end='', flush=True); t1=_time.time()

# Structure: per trade, list of (bar_idx, mtm, peak_sofar, features_vector)
trade_bars = []  # list of lists
trade_info = []  # (time, direction, ep, ea, actual_pnl, cid)

for ti, (_, trade) in enumerate(trades.iterrows()):
    tm=trade["time"]
    if tm not in t2i: continue
    ei=t2i[tm]; d=int(trade["direction"])
    ep=C[ei]; ea=atr[ei]
    if not np.isfinite(ea) or ea<=0: continue
    
    bars = []
    peak_sofar = 0.0
    for k in range(1, min(MAX_HOLD, n-ei-1)):
        bar = ei+k
        mtm = d*(C[bar]-ep)/ea
        if mtm > peak_sofar:
            peak_sofar = mtm
        if mtm <= SL_HARD:
            break
        if k < MIN_HOLD:
            continue
        
        # Features at this bar
        dist_sl = mtm - SL_HARD
        dist_tp = 2.0 - mtm
        vol10 = np.std([d*(C[max(0,bar-j)]-C[max(0,bar-j-1)])/ea
                        for j in range(min(10,bar))]) if bar>5 else 0.0
        mom3 = d*(C[bar]-C[max(0,bar-3)])/ea if bar>=3 else 0.0
        mom10 = d*(C[bar]-C[max(0,bar-10)])/ea if bar>=10 else 0.0
        
        feats = np.array([mtm, peak_sofar, peak_sofar-mtm, float(k), float(MAX_HOLD-k),
                          dist_sl, dist_tp, vol10, mom3, mom10] +
                         [float(ctx_arr[bar,j]) for j in range(len(V72L_FEATS))],
                         dtype=np.float32)
        bars.append((bar, mtm, peak_sofar, feats))
    
    if bars:
        trade_bars.append(bars)
        trade_info.append((tm, d, ep, ea, trade["pnl_R"], trade.get("cid",-1)))

print(f" {_time.time()-t1:.0f}s — {len(trade_bars):,} trades, {sum(len(b) for b in trade_bars):,} bar-samples",flush=True)

# ═══ 3. Batch-predict all bars (model inference once) ═══
print("[3/4] Batch-predicting all bars...", end='', flush=True); t1=_time.time()

all_feats = np.vstack([np.vstack([b[3] for b in tb]) for tb in trade_bars])
print(f" {len(all_feats):,} samples", end='', flush=True)

p_up  = models['label_up'].predict_proba(all_feats)[:,1]
p_gb  = models['label_gb'].predict_proba(all_feats)[:,1]
p_sl  = models['label_sl'].predict_proba(all_feats)[:,1]
p_nh  = models['label_nh'].predict_proba(all_feats)[:,1]

# Map back to per-trade
offset = 0
trade_probas = []
for tb in trade_bars:
    nbars = len(tb)
    trade_probas.append({
        'up': p_up[offset:offset+nbars],
        'gb': p_gb[offset:offset+nbars],
        'sl': p_sl[offset:offset+nbars],
        'nh': p_nh[offset:offset+nbars],
    })
    offset += nbars

print(f" — {_time.time()-t1:.0f}s",flush=True)

# ═══ 4. Sweep thresholds ═══
print("[4/4] Sweeping thresholds...", flush=True)

def simulate(th_gb, th_up, th_sl, th_nh):
    """Simulate all trades with given thresholds, return (pf, total_r, n_trades, winrate)."""
    results = []
    for ti, tb in enumerate(trade_bars):
        probs = trade_probas[ti]
        tm, d, ep, ea, actual_pnl, cid = trade_info[ti]
        
        exit_bar = None
        for bi, (bar, mtm, peak, feats) in enumerate(tb):
            # Hard SL check
            if mtm <= SL_HARD:
                exit_bar = bar
                break
            
            p_gb_i = probs['gb'][bi]
            p_up_i = probs['up'][bi]
            p_sl_i = probs['sl'][bi]
            p_nh_i = probs['nh'][bi]
            
            exit_now = False
            if p_gb_i > th_gb and p_up_i < th_up:
                exit_now = True
            if p_sl_i > th_sl:
                exit_now = True
            # Hope override
            if p_nh_i > th_nh and p_gb_i < th_gb:
                exit_now = False
            
            if exit_now:
                exit_bar = bar
                break
        
        if exit_bar is None:
            exit_bar = min(t2i[tm] + MAX_HOLD, n-1)
        
        pnl = d * (C[exit_bar] - ep) / ea
        results.append(pnl)
    
    if not results:
        return 0, 0, 0, 0
    
    total = sum(results)
    wins = [r for r in results if r > 0]
    losses = [r for r in results if r <= 0]
    pf = sum(wins) / max(-sum(losses), 1e-9) if losses else 99.0
    wr = len(wins) / len(results) * 100
    return pf, total, len(results), wr

# Baseline: hard SL only
base_pf, base_total, base_n, base_wr = simulate(0.99, 0.01, 0.99, 0.99)
print(f"\n  Baseline (hard SL only):  PF={base_pf:.2f}  Total={base_total:+.0f}R  WR={base_wr:.1f}%  N={base_n}")

# Disable multi-head by setting extreme thresholds
print(f"  (extreme thresholds = effectively disabled)")

# Coarse grid sweep
print(f"\n  Sweeping thresholds...")

# Define search ranges
gb_range = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
up_range = [0.2, 0.3, 0.4, 0.5, 0.6]
sl_range = [0.4, 0.5, 0.6, 0.7, 0.8]
nh_range = [0.6, 0.7, 0.8, 0.9]

total_combos = len(gb_range) * len(up_range) * len(sl_range) * len(nh_range)
print(f"  Grid: {len(gb_range)}×{len(up_range)}×{len(sl_range)}×{len(nh_range)} = {total_combos} combos")

results_list = []
count = 0
t_start = _time.time()

for th_gb in gb_range:
    for th_up in up_range:
        for th_sl in sl_range:
            for th_nh in nh_range:
                pf, total, nt, wr = simulate(th_gb, th_up, th_sl, th_nh)
                results_list.append({
                    'th_gb': th_gb, 'th_up': th_up,
                    'th_sl': th_sl, 'th_nh': th_nh,
                    'pf': pf, 'total_r': total, 'n': nt, 'wr': wr,
                })
                count += 1
                if count % 100 == 0:
                    elapsed = _time.time() - t_start
                    rate = count / elapsed
                    remaining = (total_combos - count) / rate
                    print(f"    {count}/{total_combos} ({count/total_combos*100:.0f}%) — "
                          f"{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining", flush=True)

print(f"  Done. {_time.time()-t_start:.0f}s total", flush=True)

# Sort and display
results_df = pd.DataFrame(results_list)

# Top by PF
print(f"\n{'='*80}")
print(f"  TOP 20 BY PROFIT FACTOR")
print(f"  {'th_gb':>6s} {'th_up':>6s} {'th_sl':>6s} {'th_nh':>6s}  {'PF':>7s} {'Total':>8s} {'WR%':>6s} {'N':>5s}  vs base")
print(f"  {'-'*6} {'-'*6} {'-'*6} {'-'*6}  {'-'*7} {'-'*8} {'-'*6} {'-'*5}  {'-'*8}")
top_pf = results_df.nlargest(20, 'pf')
for _, r in top_pf.iterrows():
    delta = r['total_r'] - base_total
    print(f"  {r['th_gb']:6.2f} {r['th_up']:6.2f} {r['th_sl']:6.2f} {r['th_nh']:6.2f}  "
          f"{r['pf']:7.2f} {r['total_r']:+8.0f} {r['wr']:6.1f} {int(r['n']):5d}  {delta:+8.0f}R")

# Top by total R
print(f"\n{'='*80}")
print(f"  TOP 20 BY TOTAL R")
print(f"  {'th_gb':>6s} {'th_up':>6s} {'th_sl':>6s} {'th_nh':>6s}  {'PF':>7s} {'Total':>8s} {'WR%':>6s} {'N':>5s}  vs base")
print(f"  {'-'*6} {'-'*6} {'-'*6} {'-'*6}  {'-'*7} {'-'*8} {'-'*6} {'-'*5}  {'-'*8}")
top_r = results_df.nlargest(20, 'total_r')
for _, r in top_r.iterrows():
    delta = r['total_r'] - base_total
    print(f"  {r['th_gb']:6.2f} {r['th_up']:6.2f} {r['th_sl']:6.2f} {r['th_nh']:6.2f}  "
          f"{r['pf']:7.2f} {r['total_r']:+8.0f} {r['wr']:6.1f} {int(r['n']):5d}  {delta:+8.0f}R")

# Show baseline for reference
print(f"\n  BASELINE (hard SL)         {base_pf:7.2f} {base_total:+8.0f} {base_wr:6.1f} {base_n:5d}")

# Best overall (PF × total_R product heuristic)
results_df['score'] = results_df['pf'] * np.maximum(results_df['total_r'], 1.0)
best = results_df.nlargest(1, 'score').iloc[0]
print(f"\n{'='*80}")
print(f"  RECOMMENDED (best PF×R):")
print(f"  th_gb={best['th_gb']:.2f}  th_up={best['th_up']:.2f}  "
      f"th_sl={best['th_sl']:.2f}  th_nh={best['th_nh']:.2f}")
print(f"  PF={best['pf']:.2f}  Total={best['total_r']:+.0f}R  WR={best['wr']:.1f}%  N={int(best['n'])}")
print(f"  vs baseline: ΔPF={best['pf']-base_pf:+.2f}  ΔR={best['total_r']-base_total:+.0f}R")

print(f"\n  Total sweep time: {_time.time()-t0:.0f}s")
