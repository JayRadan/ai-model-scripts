"""v90 experiment 3 — add 24h-momentum-signed features to q_entry.

Hypothesis: when HighVol regime fires shorts during periods where the
24h return is meaningfully positive, those shorts often lose (the user
observed this on May 11). By passing direction-signed 24h return as an
input feature, q_entry should learn to score shorts LOW when 24h is up,
and longs LOW when 24h is down — even within the HighVol cluster.

New features (in addition to V72L + maturity):
  ret_24h_signed = direction × (close - close[t-288]) / close[t-288]
                   (positive = trend-following; negative = counter-trend)
  ret_24h_abs    = |close - close[t-288]| / close[t-288]
                   (magnitude — how directional was the last 24h)

Compares apples-to-apples with current v89 (V72L + maturity) on the
post-2024-12-12 holdout, with special attention to per-cluster behavior
in HighVol (cid=4) and the long/short balance within each cluster.
"""
import os, glob as _glob, time, pickle
import numpy as np, pandas as pd
from xgboost import XGBRegressor

PROJECT = "/home/jay/Desktop/new-model-zigzag"
DATA = f"{PROJECT}/data"
HOLDOUT_START = pd.Timestamp("2024-12-12")

V72L = ['hurst_rs','ou_theta','entropy_rate','kramers_up','wavelet_er',
        'vwap_dist','hour_enc','dow_enc','quantum_flow','quantum_flow_h4',
        'quantum_momentum','quantum_vwap_conf','quantum_divergence','quantum_div_strength',
        'vpin','sig_quad_var','har_rv_ratio','hawkes_eta']
MATURITY = ['stretch_100','stretch_200','pct_to_extreme_50']
MOM24H = ['ret_24h_signed','ret_24h_abs']
SL_HARD=-4.0; MAX_HOLD=60; TRAIL_ACT=3.0; TRAIL_GB=0.60
MIN_CLUSTER = 200
NAMES = {0:"Uptrend",1:"MeanRevert",2:"TrendRange",3:"Downtrend",4:"HighVol"}

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

def mom24h_at(t_idx, d, C):
    """24h-momentum features. 288 M5 bars = 24 hours.
    ret_24h_signed: positive if trade direction matches recent 24h trend.
    ret_24h_abs:    magnitude of 24h return (regime-stress proxy).
    """
    if t_idx < 288: return [0.0, 0.0]
    c0 = float(C[t_idx-288])
    ct = float(C[t_idx])
    if c0 <= 0: return [0.0, 0.0]
    ret = (ct - c0) / c0
    return [float(d * ret), float(abs(ret))]

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

def metrics(p):
    p = np.asarray(p)
    if len(p) == 0: return None
    s = p.sum(); pos = p[p>0]; neg = p[p<=0]
    pf = pos.sum()/max(-neg.sum(), 1e-9) if len(neg)>0 else 99
    wr = (p>0).mean()*100
    eq = np.cumsum(p); peak = np.maximum.accumulate(eq); dd = float((peak-eq).max())
    return {"n":len(p), "pf":pf, "total":float(s), "wr":wr, "dd":dd}

def evaluate(name, swing_csv, setups_glob, bundle_v89_path):
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

    print("  Computing labels...", end='', flush=True); t=time.time()
    pnls = np.full(len(setups), np.nan, dtype=np.float32)
    dirs = setups['direction'].values
    for i in range(len(setups)):
        p = simulate_trade(int(setup_idx[i]), int(dirs[i]), n, C, atr)
        if p is not None: pnls[i] = p
    valid = ~np.isnan(pnls)
    setups = setups[valid].reset_index(drop=True); setup_idx = setup_idx[valid]
    pnls = pnls[valid]; dirs = dirs[valid]
    print(f" {time.time()-t:.0f}s — {len(pnls):,} setups", flush=True)

    print("  Computing features (V72L + maturity + 24h)...", end='', flush=True); t=time.time()
    V72L_X = setups[V72L].fillna(0).values.astype(np.float32)
    mat_X = np.zeros((len(setups), len(MATURITY)), dtype=np.float32)
    mom_X = np.zeros((len(setups), len(MOM24H)), dtype=np.float32)
    for i in range(len(setups)):
        mat_X[i] = maturity_at(int(setup_idx[i]), int(dirs[i]), C, atr)
        mom_X[i] = mom24h_at(int(setup_idx[i]), int(dirs[i]), C)
    feat21 = np.concatenate([V72L_X, mat_X], axis=1)              # v89 features
    feat23 = np.concatenate([V72L_X, mat_X, mom_X], axis=1)       # +24h
    print(f" {time.time()-t:.0f}s", flush=True)

    is_train = setups['time'].values < HOLDOUT_START.to_datetime64()
    is_test = ~is_train
    cids = setups['old_cid'].values

    # Train new q_entry per cluster with 23 features
    print("  Training v90 q_entry (V72L + maturity + 24h)...")
    q_v90 = {}
    for cid in sorted(set(cids)):
        mask_tr = (cids==cid) & is_train
        if mask_tr.sum() < MIN_CLUSTER:
            print(f"    cid={cid}: only {mask_tr.sum()} train, skipping"); continue
        m = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         random_state=42, verbosity=0, objective='reg:squarederror')
        m.fit(feat23[mask_tr], pnls[mask_tr])
        q_v90[int(cid)] = m
        print(f"    cid={cid} {NAMES[cid]:<11}: trained on {mask_tr.sum():,} setups")

    # Predict v90 Q on test
    Q_v90 = np.full(int(is_test.sum()), -9.0, dtype=np.float32)
    test_cids = cids[is_test]
    feat23_test = feat23[is_test]
    for cid in q_v90:
        m = test_cids == cid
        if m.sum() > 0:
            Q_v90[m] = q_v90[cid].predict(feat23_test[m])

    # Load v89 baseline q_entry and predict on same test setups
    bundle = pickle.load(open(bundle_v89_path, "rb"))
    q_v89 = bundle['q_entry']
    Q_v89 = np.full(int(is_test.sum()), -9.0, dtype=np.float32)
    feat21_test = feat21[is_test]
    for cid, m in q_v89.items():
        msk = test_cids == cid
        if msk.sum() > 0:
            Q_v89[msk] = m.predict(feat21_test[msk])

    test_pnls = pnls[is_test]
    test_dirs = dirs[is_test]
    test_mom24h_signed = mom_X[is_test, 0]

    # ── Threshold sweep comparison ──
    print(f"\n  HOLDOUT COMPARISON  (q-only filter, same simulator)")
    print(f"  {'thr':>5s}  {'v89 (V72L+mat)':>27s}    {'v90 (+24h)':>27s}    {'Δ Total R':>10s}")
    print(f"  {'':>5s}  {'N':>5s} {'PF':>5s} {'TotalR':>7s} {'WR%':>5s}    {'N':>5s} {'PF':>5s} {'TotalR':>7s} {'WR%':>5s}    {'':>10s}")
    for thr in [0.50, 1.00, 2.00, 3.00, 5.00]:
        m_v89 = metrics(test_pnls[Q_v89>thr])
        m_v90 = metrics(test_pnls[Q_v90>thr])
        if m_v89 is None or m_v90 is None:
            print(f"  {thr:>5.2f}  (too few trades at this threshold)"); continue
        delta = m_v90['total']-m_v89['total']
        marker = " ★" if (m_v90['total']>m_v89['total'] and m_v90['pf']>m_v89['pf']) else ""
        print(f"  {thr:>5.2f}  {m_v89['n']:>5d} {m_v89['pf']:>5.2f} {m_v89['total']:>+7.0f} {m_v89['wr']:>5.1f}    "
              f"{m_v90['n']:>5d} {m_v90['pf']:>5.2f} {m_v90['total']:>+7.0f} {m_v90['wr']:>5.1f}    "
              f"{delta:>+9.0f}{marker}")

    # ── Per-cluster + per-direction at q>3.0 (production threshold) ──
    print(f"\n  AT q>3.0 — per-cluster x direction breakdown:")
    print(f"  {'cid':>3s} {'regime':<11s} {'dir':>4s}   {'v89 N':>5s} {'PF':>5s} {'R':>6s}    {'v90 N':>5s} {'PF':>5s} {'R':>6s}    {'Δ R':>6s}")
    for cid in [0, 1, 2, 3, 4]:
        for d_val in [+1, -1]:
            mask = (test_cids==cid) & (test_dirs==d_val)
            if mask.sum() < 3: continue
            v89_pass = (Q_v89>3.0) & mask
            v90_pass = (Q_v90>3.0) & mask
            m89 = metrics(test_pnls[v89_pass])
            m90 = metrics(test_pnls[v90_pass])
            n89, n90 = m89['n'] if m89 else 0, m90['n'] if m90 else 0
            pf89 = m89['pf'] if m89 and m89['n']>2 else 0
            pf90 = m90['pf'] if m90 and m90['n']>2 else 0
            r89 = m89['total'] if m89 else 0
            r90 = m90['total'] if m90 else 0
            d_label = "LONG" if d_val==+1 else "SHORT"
            print(f"  {cid:>3d} {NAMES[cid]:<11s} {d_label:>5s}  {n89:>5d} {pf89:>5.2f} {r89:>+6.0f}    "
                  f"{n90:>5d} {pf90:>5.2f} {r90:>+6.0f}    {r90-r89:>+6.0f}")

    # ── Focused HighVol analysis: when 24h_return > +0.5%, did v90 reject shorts? ──
    print(f"\n  HIGHVOL (cid=4) — split by 24h directionality:")
    cid4 = test_cids == 4
    # Convert signed back to unsigned 24h return: signed = d * ret → ret = signed * d
    test_ret_24h_unsigned = test_mom24h_signed * test_dirs  # back to signed-by-time
    pos_24h = cid4 & (test_ret_24h_unsigned > 0.005)   # +0.5% 24h up
    neg_24h = cid4 & (test_ret_24h_unsigned < -0.005)  # -0.5% 24h down
    print(f"  HighVol + 24h > +0.5% (uptrending intraday):")
    for d_val in [+1, -1]:
        msk = pos_24h & (test_dirs==d_val)
        v89_pass = (Q_v89>3.0) & msk
        v90_pass = (Q_v90>3.0) & msk
        m89 = metrics(test_pnls[v89_pass]); m90 = metrics(test_pnls[v90_pass])
        d_label = "LONG" if d_val==+1 else "SHORT"
        n89 = m89['n'] if m89 else 0; n90 = m90['n'] if m90 else 0
        r89 = m89['total'] if m89 else 0; r90 = m90['total'] if m90 else 0
        print(f"    {d_label:>5s}: v89 {n89:>3d} trades ({r89:>+5.0f}R) → v90 {n90:>3d} trades ({r90:>+5.0f}R)  Δn={n90-n89:+d}")
    print(f"  HighVol + 24h < -0.5% (downtrending intraday):")
    for d_val in [+1, -1]:
        msk = neg_24h & (test_dirs==d_val)
        v89_pass = (Q_v89>3.0) & msk
        v90_pass = (Q_v90>3.0) & msk
        m89 = metrics(test_pnls[v89_pass]); m90 = metrics(test_pnls[v90_pass])
        d_label = "LONG" if d_val==+1 else "SHORT"
        n89 = m89['n'] if m89 else 0; n90 = m90['n'] if m90 else 0
        r89 = m89['total'] if m89 else 0; r90 = m90['total'] if m90 else 0
        print(f"    {d_label:>5s}: v89 {n89:>3d} trades ({r89:>+5.0f}R) → v90 {n90:>3d} trades ({r90:>+5.0f}R)  Δn={n90-n89:+d}")

    # ── Feature importance for HighVol q_entry ──
    if 4 in q_v90:
        fi = q_v90[4].feature_importances_
        names = V72L + MATURITY + MOM24H
        top = sorted(enumerate(fi), key=lambda x:-x[1])[:10]
        print(f"\n  HighVol (cid=4) v90 q_entry — top-10 feature importance:")
        for i, v in top:
            tag = "★ NEW" if names[i] in MOM24H else ("MAT" if names[i] in MATURITY else "V72L")
            print(f"    {tag:>6s}  {names[i]:<22s}  {v:.4f}")

    print(f"\n  Done in {time.time()-t0:.0f}s")

if __name__ == "__main__":
    evaluate("Oracle XAU — 24h-aware q_entry",
             f"{DATA}/swing_v5_xauusd.csv",
             f"{DATA}/setups_*_v72l.csv",
             f"{PROJECT}/products/models/oracle_xau_validated.pkl")
    evaluate("Oracle BTC — 24h-aware q_entry",
             f"{DATA}/swing_v5_btc.csv",
             f"{DATA}/setups_*_v72l_btc.csv",
             f"{PROJECT}/products/models/oracle_btc_validated.pkl")
