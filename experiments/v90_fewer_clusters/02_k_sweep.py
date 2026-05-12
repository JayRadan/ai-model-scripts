"""v90 experiment 2 — sweep K from 5 to 10 to see if more granular
regimes give better trade quality.

For each K:
  1. Refit K-means with that K on the training fingerprints
  2. Assign new cluster to each setup
  3. Train V72L+maturity q_entry per cluster on training data
  4. Score the post-2024-12-12 holdout setups
  5. Apply hard SL + trail + 60-bar max simulator
  6. Report PF / Total R / n_trades at fixed q-thresholds

Includes K=5 with the SAME PIPELINE as a fair baseline (vs the
current production bundle which uses a slightly different training
recipe — that comparison is reported separately).
"""
import os, glob as _glob, time, pickle
import numpy as np, pandas as pd
from xgboost import XGBRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

PROJECT = "/home/jay/Desktop/new-model-zigzag"
DATA = f"{PROJECT}/data"
HOLDOUT_START = pd.Timestamp("2024-12-12")

V72L = ['hurst_rs','ou_theta','entropy_rate','kramers_up','wavelet_er',
        'vwap_dist','hour_enc','dow_enc','quantum_flow','quantum_flow_h4',
        'quantum_momentum','quantum_vwap_conf','quantum_divergence','quantum_div_strength',
        'vpin','sig_quad_var','har_rv_ratio','hawkes_eta']
MATURITY = ['stretch_100','stretch_200','pct_to_extreme_50']
FP_FEATS = ["weekly_return_pct","volatility_pct","trend_consistency",
            "trend_strength","volatility","range_vs_atr","return_autocorr"]
SL_HARD=-4.0; MAX_HOLD=60; TRAIL_ACT=3.0; TRAIL_GB=0.60
MIN_CLUSTER_TRAIN = 200

def load_market(swing_csv):
    sw = pd.read_csv(swing_csv, parse_dates=["time"]).sort_values("time").drop_duplicates('time',keep='last').reset_index(drop=True)
    n = len(sw); C = sw["close"].values.astype(np.float64)
    H = sw["high"].values; Lo = sw["low"].values
    tr = np.concatenate([[H[0]-Lo[0]], np.maximum.reduce([H[1:]-Lo[1:], np.abs(H[1:]-C[:-1]), np.abs(Lo[1:]-C[:-1])])])
    atr = pd.Series(tr).rolling(14, min_periods=14).mean().values
    t2i = {pd.Timestamp(t):i for i,t in enumerate(sw["time"].values)}
    return n, C, atr, t2i

def load_setups(setups_glob):
    parts = []
    for f in sorted(_glob.glob(setups_glob)):
        cid_str = os.path.basename(f).split('_')[1]
        try: old_cid = int(cid_str)
        except: continue
        df = pd.read_csv(f, parse_dates=["time"]); df['old_cid'] = old_cid
        parts.append(df)
    s = pd.concat(parts, ignore_index=True)
    return s.sort_values(['time','direction']).drop_duplicates(['time','direction'], keep='first').reset_index(drop=True)

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

def simulate_trade(t_idx, d, n, C, atr):
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

def metrics(pnls):
    p = np.asarray(pnls)
    if len(p) == 0: return None
    s = p.sum(); pos = p[p>0]; neg = p[p<=0]
    pf = pos.sum()/max(-neg.sum(), 1e-9) if len(neg)>0 else 99
    wr = (p>0).mean()*100
    eq = np.cumsum(p); peak = np.maximum.accumulate(eq); dd = float((peak-eq).max())
    return {"n":len(p), "pf":pf, "total":float(s), "wr":wr, "dd":dd}

def train_and_test_at_K(K, setups, setup_idx, dirs, pnls, full_X, fp,
                        is_train, is_test, scaler):
    """Refit K-means with K clusters; train per-cluster q_entry; predict on test."""
    fp_train = fp[fp['center_time'] < HOLDOUT_START]
    Xfp_train = scaler.transform(fp_train[FP_FEATS].values)
    Xfp_all = scaler.transform(fp[FP_FEATS].values)
    km = KMeans(n_clusters=K, random_state=42, n_init=10).fit(Xfp_train)
    fp_new_cid = km.predict(Xfp_all)

    block_ends = fp.sort_values('end_idx')['end_idx'].values
    block_cids = fp.sort_values('end_idx').reset_index(drop=True)
    block_cids_arr = fp_new_cid[block_cids.index] if False else None
    # Sort fp by end_idx and pull new_cid in that order
    fp_sorted = fp.assign(_new_cid=fp_new_cid).sort_values('end_idx').reset_index(drop=True)
    block_ends = fp_sorted['end_idx'].values
    block_cids = fp_sorted['_new_cid'].values

    new_cid_per_setup = np.zeros(len(setups), dtype=np.int32)
    for i in range(len(setups)):
        j = np.searchsorted(block_ends, setup_idx[i], side='right') - 1
        if j < 0: j = 0
        new_cid_per_setup[i] = int(block_cids[j])

    # Train q_entry per cluster
    q_models = {}
    unique = np.unique(new_cid_per_setup)
    for cid in unique:
        mask_tr = (new_cid_per_setup==cid) & is_train
        if mask_tr.sum() < MIN_CLUSTER_TRAIN: continue
        m = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         random_state=42, verbosity=0, objective='reg:squarederror')
        m.fit(full_X[mask_tr], pnls[mask_tr])
        q_models[int(cid)] = m

    # Predict on test
    Q_test = np.full(is_test.sum(), -9.0, dtype=np.float32)
    test_cids = new_cid_per_setup[is_test]
    full_X_test = full_X[is_test]
    for cid in q_models:
        mt = test_cids == cid
        if mt.sum() > 0:
            Q_test[mt] = q_models[cid].predict(full_X_test[mt])

    return Q_test, dict(zip(*np.unique(new_cid_per_setup[is_train], return_counts=True))), q_models

def evaluate(name, swing_csv, setups_glob, fp_csv):
    print(f"\n{'='*72}\n  {name}\n{'='*72}")
    t0 = time.time()
    n, C, atr, t2i = load_market(swing_csv)
    setups = load_setups(setups_glob)
    setup_idx = np.full(len(setups), -1, dtype=np.int64)
    for i, t in enumerate(setups['time']):
        ti = pd.Timestamp(t)
        if ti in t2i: setup_idx[i] = t2i[ti]
    keep = setup_idx >= 0
    setups = setups[keep].reset_index(drop=True); setup_idx = setup_idx[keep]

    print("  Simulating pnl_R labels...", end='', flush=True); t=time.time()
    pnls = np.full(len(setups), np.nan, dtype=np.float32)
    dirs = setups['direction'].values
    for i in range(len(setups)):
        p = simulate_trade(int(setup_idx[i]), int(dirs[i]), n, C, atr)
        if p is not None: pnls[i] = p
    valid = ~np.isnan(pnls)
    setups = setups[valid].reset_index(drop=True); setup_idx = setup_idx[valid]
    pnls = pnls[valid]; dirs = dirs[valid]
    print(f" {time.time()-t:.0f}s — {len(pnls):,} setups", flush=True)

    print("  Computing features...", end='', flush=True); t=time.time()
    V72L_X = setups[V72L].fillna(0).values.astype(np.float32)
    mat_X = np.zeros((len(setups), len(MATURITY)), dtype=np.float32)
    for i in range(len(setups)):
        mat_X[i] = maturity_at(int(setup_idx[i]), int(dirs[i]), C, atr)
    full_X = np.concatenate([V72L_X, mat_X], axis=1)
    print(f" {time.time()-t:.0f}s", flush=True)

    is_train = setups['time'].values < HOLDOUT_START.to_datetime64()
    is_test = ~is_train
    test_pnls = pnls[is_test]

    fp = pd.read_csv(fp_csv, parse_dates=["center_time"])
    scaler = StandardScaler().fit(fp[fp['center_time']<HOLDOUT_START][FP_FEATS].values)

    # Run for each K
    results_by_k = {}
    for K in [5, 6, 7, 8, 9, 10]:
        print(f"\n  ── K={K} ──")
        t = time.time()
        Q_test, train_counts, q_models = train_and_test_at_K(
            K, setups, setup_idx, dirs, pnls, full_X, fp,
            is_train, is_test, scaler)
        print(f"    Trained {len(q_models)} per-cluster models in {time.time()-t:.0f}s")
        print(f"    Cluster sizes (train): " +
              ", ".join(f"c{c}={n_}" for c,n_ in sorted(train_counts.items())))
        results_by_k[K] = Q_test

    # Comparison table
    print(f"\n  HOLDOUT — q-only filter at various thresholds:")
    headers = " ".join(f"K={K:>2d}".rjust(8) for K in [5,6,7,8,9,10])
    print(f"  thr   " + " ".join(f"{'N':>4s} {'PF':>5s} {'Total':>7s}" for _ in range(6)))
    for thr in [0.50, 1.00, 2.00, 3.00, 5.00]:
        row = [f"  {thr:>3.2f}  "]
        for K in [5, 6, 7, 8, 9, 10]:
            Q = results_by_k[K]
            m = metrics(test_pnls[Q > thr])
            if m is None or m['n'] < 5:
                row.append(f"{0:>4d} {0:>5.2f} {0:>+7.0f}")
            else:
                row.append(f"{m['n']:>4d} {m['pf']:>5.2f} {m['total']:>+7.0f}")
        print("".join(row[:1]) + " | ".join(row[1:]))

    # Best K by Total R per threshold
    print(f"\n  BEST K BY TOTAL R per threshold:")
    print(f"  {'thr':>5s}  {'best K':>6s}  {'N':>5s}  {'PF':>5s}  {'TotalR':>7s}")
    for thr in [0.50, 1.00, 2.00, 3.00, 5.00]:
        best_K = None; best_total = -1e9; best_m = None
        for K in [5, 6, 7, 8, 9, 10]:
            m = metrics(test_pnls[results_by_k[K] > thr])
            if m is None or m['n']<5: continue
            if m['total'] > best_total:
                best_total = m['total']; best_K = K; best_m = m
        if best_m:
            print(f"  {thr:>5.2f}  K={best_K:>3d}  {best_m['n']:>5d}  {best_m['pf']:>5.2f}  {best_m['total']:>+7.0f}")

    print(f"\n  Done in {time.time()-t0:.0f}s")

if __name__ == "__main__":
    evaluate("Oracle XAU — K sweep",
             f"{DATA}/swing_v5_xauusd.csv",
             f"{DATA}/setups_*_v72l.csv",
             f"{DATA}/regime_fingerprints_K4.csv")
    evaluate("Oracle BTC — K sweep",
             f"{DATA}/swing_v5_btc.csv",
             f"{DATA}/setups_*_v72l_btc.csv",
             f"{DATA}/regime_fingerprints_btc_K5.csv")
