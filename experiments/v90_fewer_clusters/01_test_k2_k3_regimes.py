"""v90 experiment — retrain regime detector with fewer clusters.

Hypothesis: by reducing K-means from K=5 → K=2 (XAU) / K=3 (BTC), every
bar gets classified into a strong-trading regime. Previously-MeanRevert /
HighVol bars get forced into Uptrend/Downtrend (or TrendRange for BTC).

Steps:
  1. Re-fit K-means with new K on the existing regime fingerprints
  2. Map every setup to its new cluster via block lookup
  3. Train fresh per-cluster q_entry (V72L + maturity) on the new labels
  4. On the 2024-12-12+ holdout: classify, score, simulate (hard SL + v88
     + trail + 60-bar max). Confirm + meta are SKIPPED for this directional
     test — they'd need retraining to handle new cluster semantics.
  5. Compare to v89 baseline (q-only filter at matching trade counts)

The "baseline" for v90 is also q-only filter on the current v89 q_entry
(same simulator, same exit policy) — apples-to-apples q_entry comparison.
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
SL_HARD=-4.0; MAX_HOLD=60; TRAIL_ACT=3.0; TRAIL_GB=0.60

FP_FEATS = ["weekly_return_pct","volatility_pct","trend_consistency",
            "trend_strength","volatility","range_vs_atr","return_autocorr"]

def load_market(swing_csv):
    sw = pd.read_csv(swing_csv, parse_dates=["time"]).sort_values("time").drop_duplicates('time',keep='last').reset_index(drop=True)
    n = len(sw); C = sw["close"].values.astype(np.float64)
    H = sw["high"].values; Lo = sw["low"].values
    tr = np.concatenate([[H[0]-Lo[0]], np.maximum.reduce([H[1:]-Lo[1:], np.abs(H[1:]-C[:-1]), np.abs(Lo[1:]-C[:-1])])])
    atr = pd.Series(tr).rolling(14, min_periods=14).mean().values
    t2i = {pd.Timestamp(t):i for i,t in enumerate(sw["time"].values)}
    return n, C, atr, t2i, sw

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
    if len(p) == 0: return {"n":0, "pf":0, "total":0, "wr":0, "dd":0}
    s = p.sum(); pos = p[p>0]; neg = p[p<=0]
    pf = pos.sum()/max(-neg.sum(), 1e-9) if len(neg)>0 else 99
    wr = (p>0).mean()*100
    eq = np.cumsum(p); peak = np.maximum.accumulate(eq); dd = float((peak-eq).max())
    return {"n":len(p), "pf":pf, "total":float(s), "wr":wr, "dd":dd}

def evaluate(name, swing_csv, setups_glob, fp_csv, bundle_v89, new_k):
    print(f"\n{'='*70}\n  {name}\n{'='*70}")
    t0 = time.time()
    n, C, atr, t2i, sw = load_market(swing_csv)
    setups = load_setups(setups_glob)
    print(f"  Setups: {len(setups):,}")

    # Map setups to swing idx
    setup_idx = np.full(len(setups), -1, dtype=np.int64)
    for i, t in enumerate(setups['time']):
        ti = pd.Timestamp(t)
        if ti in t2i: setup_idx[i] = t2i[ti]
    keep = setup_idx >= 0
    setups = setups[keep].reset_index(drop=True); setup_idx = setup_idx[keep]

    # Simulate pnl_R label
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

    # Compute V72L + maturity features
    V72L_X = setups[V72L].fillna(0).values.astype(np.float32)
    mat_X = np.zeros((len(setups), len(MATURITY)), dtype=np.float32)
    for i in range(len(setups)):
        mat_X[i] = maturity_at(int(setup_idx[i]), int(dirs[i]), C, atr)
    full_X = np.concatenate([V72L_X, mat_X], axis=1)

    # ─────────── Re-cluster with new K ───────────
    fp = pd.read_csv(fp_csv, parse_dates=["center_time"])
    fp_train_mask = fp['center_time'] < HOLDOUT_START
    fp_train = fp[fp_train_mask]

    # Re-fit K-means with new K
    scaler = StandardScaler().fit(fp_train[FP_FEATS].values)
    Xfp_train = scaler.transform(fp_train[FP_FEATS].values)
    Xfp_all = scaler.transform(fp[FP_FEATS].values)
    km = KMeans(n_clusters=new_k, random_state=42, n_init=10).fit(Xfp_train)
    fp['new_cid'] = km.predict(Xfp_all)

    # Map setup time → fingerprint block → new cluster
    # Each fingerprint row has start_idx, end_idx (bar indices in the swing df).
    # A setup at bar B falls into block where start_idx <= B < end_idx, but
    # since blocks overlap (288-bar window, 48-bar step), use most recent
    # block whose end_idx <= B.
    fp_sorted = fp.sort_values('end_idx').reset_index(drop=True)
    block_ends = fp_sorted['end_idx'].values
    block_cids = fp_sorted['new_cid'].values
    new_cid_per_setup = np.zeros(len(setups), dtype=np.int32)
    for i in range(len(setups)):
        j = np.searchsorted(block_ends, setup_idx[i], side='right') - 1
        if j < 0: j = 0
        new_cid_per_setup[i] = int(block_cids[j])

    # Cluster sizes
    unique, counts = np.unique(new_cid_per_setup, return_counts=True)
    print(f"\n  NEW K={new_k} cluster assignments (setups):")
    for u, c in zip(unique, counts):
        print(f"    cid={u}: {c:,} setups ({c*100/len(setups):.1f}%)")

    # Split chrono
    is_train = setups['time'].values < HOLDOUT_START.to_datetime64()
    is_test = ~is_train

    # Train per-new-cluster q_entry on V72L+maturity
    print("\n  Training new q_entry per cluster (V72L + maturity)...")
    q_new = {}
    for cid in unique:
        mask_tr = (new_cid_per_setup==cid) & is_train
        if mask_tr.sum() < 200:
            print(f"    cid={cid}: only {mask_tr.sum()} train, skipping"); continue
        m = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         random_state=42, verbosity=0, objective='reg:squarederror')
        m.fit(full_X[mask_tr], pnls[mask_tr])
        q_new[int(cid)] = m
        print(f"    cid={cid}: trained on {mask_tr.sum():,} setups")

    # Predict Q for holdout
    test_idx = np.where(is_test)[0]
    Q_test = np.full(len(test_idx), -9.0, dtype=np.float32)
    test_cids = new_cid_per_setup[is_test]
    full_X_test = full_X[is_test]
    for cid in q_new:
        m = test_cids == cid
        if m.sum() > 0:
            Q_test[m] = q_new[cid].predict(full_X_test[m])
    test_pnls = pnls[is_test]

    # ─────────── BASELINE: current v89 q_entry on same setups ───────────
    bundle_v89_obj = pickle.load(open(bundle_v89, "rb"))
    q_v89 = bundle_v89_obj['q_entry']
    Q_v89_test = np.full(len(test_idx), -9.0, dtype=np.float32)
    setup_cids = setups['old_cid'].values[is_test]
    for cid, m in q_v89.items():
        msk = setup_cids == cid
        if msk.sum() > 0:
            Q_v89_test[msk] = m.predict(full_X_test[msk])

    # ─────────── Compare at multiple thresholds ───────────
    print(f"\n  HOLDOUT COMPARISON (q-only filter, same simulator)")
    print(f"  {'thr':>5s}  {'v89 (K=5)':>25s}    {'v90 (K='+str(new_k)+')':>25s}    Δ Total R")
    print(f"  {'':>5s}  {'N':>5s} {'PF':>5s} {'Total':>7s}    {'N':>5s} {'PF':>5s} {'Total':>7s}")
    for thr in [0.30, 0.50, 1.0, 2.0, 3.0, 4.0, 5.0]:
        m_v89 = metrics(test_pnls[Q_v89_test>thr])
        m_v90 = metrics(test_pnls[Q_test>thr])
        marker = ""
        if m_v90['total'] > m_v89['total'] and m_v90['pf'] > m_v89['pf']: marker = " ★"
        print(f"  {thr:>5.2f}  {m_v89['n']:>5d} {m_v89['pf']:>5.2f} {m_v89['total']:>+7.0f}    "
              f"{m_v90['n']:>5d} {m_v90['pf']:>5.2f} {m_v90['total']:>+7.0f}    "
              f"{m_v90['total']-m_v89['total']:>+7.0f}{marker}")

    print(f"\n  Done in {time.time()-t0:.0f}s")

if __name__ == "__main__":
    evaluate("Oracle XAU — K=2 (Uptrend + Downtrend forced)",
             f"{DATA}/swing_v5_xauusd.csv",
             f"{DATA}/setups_*_v72l.csv",
             f"{DATA}/regime_fingerprints_K4.csv",
             f"{PROJECT}/products/models/oracle_xau_validated.pkl",
             new_k=2)
    evaluate("Oracle BTC — K=3 (Uptrend + Downtrend + TrendRange forced)",
             f"{DATA}/swing_v5_btc.csv",
             f"{DATA}/setups_*_v72l_btc.csv",
             f"{DATA}/regime_fingerprints_btc_K5.csv",
             f"{PROJECT}/products/models/oracle_btc_validated.pkl",
             new_k=3)
