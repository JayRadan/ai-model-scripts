"""Full pipeline rebuild: retrain q_entry on TRAILING cluster labels.

If we believe the chart staleness is the real bug, the proper fix is:
  - relabel every training setup with trailing-cid (not block-cid)
  - refit q_entry XGB per trailing-cid
  - route inference via trailing-cid (no model mismatch)

Question this answers: does the end-to-end (relabel + retrain) system
beat current production (block-cid + current q_entry)?

Comparison on holdout:
  PROD = block-cid routed through current q_entry (oracle_*_validated.pkl)
  NEW  = trailing-cid routed through freshly-trained q_entry

Same min_q gates, same simulator. Total R / PF / WR.

Speed: classify_trailing is the bottleneck. Subsample setups (every 5th
train, every 10th test) to keep runtime under ~2 hours per product.
"""
import os, sys, glob as _glob, pickle, json, time
import numpy as np, pandas as pd
from xgboost import XGBRegressor

PROJECT = "/home/jay/Desktop/new-model-zigzag"
DATA = f"{PROJECT}/data"
SERVER = "/home/jay/Desktop/my-agents-and-website/commercial/server"
OUT = f"{PROJECT}/experiments/v91_smart_regime"
sys.path.insert(0, SERVER)
from decision_engine import regime as _regime

HOLDOUT_START = pd.Timestamp("2024-12-12")
NAMES = {0:"Uptrend",1:"MeanRevert",2:"TrendRange",3:"Downtrend",4:"HighVol"}

V72L = ['hurst_rs','ou_theta','entropy_rate','kramers_up','wavelet_er',
        'vwap_dist','hour_enc','dow_enc','quantum_flow','quantum_flow_h4',
        'quantum_momentum','quantum_vwap_conf','quantum_divergence','quantum_div_strength',
        'vpin','sig_quad_var','har_rv_ratio','hawkes_eta']
SL_HARD=-4.0; MAX_HOLD=60; TRAIL_ACT=3.0; TRAIL_GB=0.60

# Subsampling
TRAIN_STRIDE = 5
TEST_STRIDE  = 10

def load_market(swing_csv):
    sw = pd.read_csv(swing_csv, parse_dates=["time"]).sort_values("time")\
           .drop_duplicates('time',keep='last').reset_index(drop=True)
    n = len(sw); C = sw["close"].values.astype(np.float64)
    H = sw["high"].values; Lo = sw["low"].values
    tr = np.concatenate([[H[0]-Lo[0]],
        np.maximum.reduce([H[1:]-Lo[1:], np.abs(H[1:]-C[:-1]), np.abs(Lo[1:]-C[:-1])])])
    atr = pd.Series(tr).rolling(14, min_periods=14).mean().values
    t2i = {pd.Timestamp(t):i for i,t in enumerate(sw["time"].values)}
    return sw, n, C, atr, t2i

def load_setups(setups_glob):
    parts = []
    for f in sorted(_glob.glob(setups_glob)):
        cid_str = os.path.basename(f).split('_')[1]
        try: old_cid = int(cid_str)
        except: continue
        df = pd.read_csv(f, parse_dates=["time"]); df['old_cid'] = old_cid
        parts.append(df)
    return pd.concat(parts, ignore_index=True).sort_values(['time','direction'])\
             .drop_duplicates(['time','direction'], keep='first').reset_index(drop=True)

def maturity_at(t_idx, d, C, atr):
    ea = atr[t_idx]
    if not np.isfinite(ea) or ea<=0: return [0.0, 0.0, 0.5]
    out = []
    for L in [100, 200]:
        if t_idx >= L:
            win = C[t_idx-L:t_idx+1]
            v = (C[t_idx]-win.min())/ea if d==1 else (win.max()-C[t_idx])/ea
        else: v = 0.0
        out.append(float(v))
    L = 50
    if t_idx >= L:
        win = C[t_idx-L:t_idx+1]
        rng = win.max()-win.min()
        if d==1: pct = (C[t_idx]-win.min())/rng if rng>0 else 0.5
        else:    pct = (win.max()-C[t_idx])/rng if rng>0 else 0.5
    else: pct = 0.5
    out.append(float(pct))
    return out

def mom24h_at(t_idx, d, C):
    if t_idx < 288: return [0.0, 0.0]
    c0 = float(C[t_idx-288]); ct = float(C[t_idx])
    if c0 <= 0: return [0.0, 0.0]
    ret = (ct - c0) / c0
    return [float(d * ret), float(abs(ret))]

def sim(t_idx, d, n, C, atr):
    ep = C[t_idx]; ea = atr[t_idx]
    if not np.isfinite(ea) or ea<=0: return None
    peak = 0.0; max_k = min(MAX_HOLD, n-t_idx-1)
    for k in range(1, max_k+1):
        bar = t_idx+k; R = d*(C[bar]-ep)/ea
        if R <= SL_HARD: return R
        if R > peak: peak = R
        if peak >= TRAIL_ACT and R <= peak*(1.0-TRAIL_GB): return R
    last = min(t_idx+max_k, n-1)
    return d*(C[last]-ep)/ea

def metrics(p):
    p = np.asarray(p)
    if len(p) == 0: return None
    s = p.sum(); pos = p[p>0]; neg = p[p<=0]
    pf = pos.sum()/max(-neg.sum(), 1e-9) if len(neg)>0 else 99
    return {"n":len(p), "pf":pf, "total":float(s), "wr": (p>0).mean()}

def run(name, swing_csv, setups_glob, selector_json, bundle_path):
    print(f"\n{'='*78}\n  {name}\n{'='*78}", flush=True)
    t0 = time.time()
    sw, n, C, atr, t2i = load_market(swing_csv)
    selector = json.loads(open(selector_json).read())
    setups = load_setups(setups_glob)

    setup_idx = np.full(len(setups), -1, dtype=np.int64)
    for i, t in enumerate(setups['time']):
        ti = pd.Timestamp(t)
        if ti in t2i: setup_idx[i] = t2i[ti]
    keep = setup_idx >= 0
    setups = setups[keep].reset_index(drop=True); setup_idx = setup_idx[keep]

    is_test = setups['time'].values >= HOLDOUT_START.to_datetime64()
    train_setups = setups[~is_test].reset_index(drop=True)
    train_idx    = setup_idx[~is_test]
    test_setups  = setups[is_test].reset_index(drop=True)
    test_idx     = setup_idx[is_test]

    # Subsample
    train_setups = train_setups.iloc[::TRAIN_STRIDE].reset_index(drop=True)
    train_idx    = train_idx[::TRAIN_STRIDE]
    test_setups  = test_setups.iloc[::TEST_STRIDE].reset_index(drop=True)
    test_idx     = test_idx[::TEST_STRIDE]
    print(f"  train setups (1/{TRAIN_STRIDE}): {len(train_setups):,}", flush=True)
    print(f"  test  setups (1/{TEST_STRIDE}): {len(test_setups):,}",  flush=True)

    # Simulate PnL for both sets
    def sim_pnls(setups_df, idx_arr):
        pnls = np.full(len(setups_df), np.nan, dtype=np.float32)
        for i in range(len(setups_df)):
            p = sim(int(idx_arr[i]), int(setups_df['direction'].iat[i]), n, C, atr)
            if p is not None: pnls[i] = p
        return pnls
    train_pnls = sim_pnls(train_setups, train_idx)
    test_pnls  = sim_pnls(test_setups,  test_idx)
    tm = ~np.isnan(train_pnls); train_setups = train_setups[tm].reset_index(drop=True)
    train_idx = train_idx[tm]; train_pnls = train_pnls[tm]
    om = ~np.isnan(test_pnls);  test_setups  = test_setups[om].reset_index(drop=True)
    test_idx  = test_idx[om];   test_pnls  = test_pnls[om]
    print(f"  valid train: {len(train_pnls):,}   valid test: {len(test_pnls):,}", flush=True)

    # Features (23-dim) for both sets
    def feat23(setups_df, idx_arr):
        V72L_X = setups_df[V72L].fillna(0).values.astype(np.float32)
        mat = np.zeros((len(setups_df), 3), dtype=np.float32)
        mom = np.zeros((len(setups_df), 2), dtype=np.float32)
        for i in range(len(setups_df)):
            d = int(setups_df['direction'].iat[i])
            mat[i] = maturity_at(int(idx_arr[i]), d, C, atr)
            mom[i] = mom24h_at(int(idx_arr[i]), d, C)
        return np.concatenate([V72L_X, mat, mom], axis=1)
    train_X = feat23(train_setups, train_idx)
    test_X  = feat23(test_setups,  test_idx)
    print(f"  features built. elapsed {(time.time()-t0)/60:.1f}m", flush=True)

    # Compute trailing cid (cached per unique bar)
    def trail_lookup(idx_arr):
        unique_idx = sorted(set(int(i) for i in idx_arr))
        print(f"    trailing-classify at {len(unique_idx):,} unique bars...", flush=True)
        cache = {}
        for k, i in enumerate(unique_idx):
            sub = sw.iloc[:i+1]
            try: cache[i] = int(_regime.classify_trailing(sub, selector))
            except ValueError: cache[i] = -1
            if (k+1) % 500 == 0:
                rate = (k+1) / (time.time() - t0 + 1e-9)
                print(f"      {k+1}/{len(unique_idx)}  ({rate:.1f}/s)", flush=True)
        return np.array([cache[int(i)] for i in idx_arr], dtype=np.int32)

    print("  TRAIN trailing-classify...", flush=True)
    train_trail = trail_lookup(train_idx)
    print(f"  done. elapsed {(time.time()-t0)/60:.1f}m", flush=True)

    print("  TEST trailing-classify...", flush=True)
    test_trail = trail_lookup(test_idx)
    print(f"  done. elapsed {(time.time()-t0)/60:.1f}m", flush=True)

    # Filter to valid trailing
    tm = train_trail >= 0
    train_setups, train_X, train_pnls, train_trail = (train_setups[tm].reset_index(drop=True),
                                                       train_X[tm], train_pnls[tm], train_trail[tm])
    om = test_trail >= 0
    test_setups,  test_X,  test_pnls,  test_trail  = (test_setups[om].reset_index(drop=True),
                                                       test_X[om],  test_pnls[om],  test_trail[om])
    print(f"  trailing-cid label distribution (train):  "
          f"{dict(zip(*np.unique(train_trail, return_counts=True)))}", flush=True)
    print(f"  trailing-cid label distribution (test):   "
          f"{dict(zip(*np.unique(test_trail,  return_counts=True)))}", flush=True)

    # Train q_entry per trailing-cid using train_pnls as target
    new_q = {}
    for cid in sorted(set(train_trail)):
        m = train_trail == cid
        if m.sum() < 100:
            print(f"    cid {cid}: {m.sum()} train rows — skipping (insufficient)", flush=True)
            continue
        mdl = XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05,
                           tree_method='hist', n_jobs=-1)
        mdl.fit(train_X[m], train_pnls[m])
        new_q[cid] = mdl
        print(f"    cid {cid}: trained on {m.sum():,} setups", flush=True)

    # Evaluate
    bundle = pickle.load(open(bundle_path, "rb"))
    prod_q = bundle['q_entry']
    block_cid_test = test_setups['old_cid'].values.astype(np.int32)

    Q_prod = np.full(len(test_pnls), -9.0, dtype=np.float32)
    for cid in prod_q:
        m = block_cid_test == cid
        if m.sum() > 0: Q_prod[m] = prod_q[cid].predict(test_X[m])

    Q_new = np.full(len(test_pnls), -9.0, dtype=np.float32)
    for cid in new_q:
        m = test_trail == cid
        if m.sum() > 0: Q_new[m] = new_q[cid].predict(test_X[m])

    print(f"\n  {'thr':>5s}  {'PROD (block + current q_entry)':>34s}  "
          f"{'NEW (trail + retrained q_entry)':>34s}  {'Δ R':>7s}", flush=True)
    for thr in [0.5, 1.0, 2.0, 3.0, 5.0]:
        mp = metrics(test_pnls[Q_prod > thr])
        mn = metrics(test_pnls[Q_new  > thr])
        if mp is None or mn is None: continue
        print(f"  {thr:>5.2f}  "
              f"N={mp['n']:>4d} PF={mp['pf']:>5.2f} R={mp['total']:>+8.0f} WR={mp['wr']*100:>4.1f}%  "
              f"N={mn['n']:>4d} PF={mn['pf']:>5.2f} R={mn['total']:>+8.0f} WR={mn['wr']*100:>4.1f}%  "
              f"{mn['total']-mp['total']:>+7.0f}", flush=True)

    out_pkl = f"{OUT}/q_entry_trailing_{name.split()[0].lower()}.pkl"
    pickle.dump({'q_entry_trailing': new_q, 'features': V72L + ['mat1','mat2','mat3','mom1','mom2']},
                open(out_pkl, "wb"))
    print(f"\n  saved {out_pkl}   total runtime {(time.time()-t0)/60:.1f}m", flush=True)

if __name__ == "__main__":
    run("XAU",
        f"{DATA}/swing_v5_xauusd.csv",
        f"{DATA}/setups_*_v72l.csv",
        f"{SERVER}/decision_engine/data/regime_selector_K4.json",
        f"{PROJECT}/products/models/oracle_xau_validated.pkl")
    run("BTC",
        f"{DATA}/swing_v5_btc.csv",
        f"{DATA}/setups_*_v72l_btc.csv",
        f"{SERVER}/decision_engine/data/regime_selector_btc_K5.json",
        f"{PROJECT}/products/models/oracle_btc_validated.pkl")
