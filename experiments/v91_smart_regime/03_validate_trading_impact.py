"""v91 — validate trading impact of consistency-based relabel.

If we relabel HighVol → Uptrend/Downtrend when consistency > 0.50,
the affected setups go through q_entry[Uptrend] or q_entry[Downtrend]
instead of q_entry[HighVol]. Do we make MORE money, or do we route
them through a mismatched model?

This script:
  1. Identify which bar timestamps fall in the v91-relabeled blocks
  2. For setups at those timestamps:
     - Score with v90 q_entry[OLD label] (current production)
     - Score with v90 q_entry[v91 label] (proposed)
  3. Run both filtered sets through the same simulator
  4. Compare total R / PF
"""
import os, glob as _glob, pickle
import numpy as np, pandas as pd
PROJECT = "/home/jay/Desktop/new-model-zigzag"
DATA = f"{PROJECT}/data"
HOLDOUT_START = pd.Timestamp("2024-12-12")
NAMES = {0:"Uptrend",1:"MeanRevert",2:"TrendRange",3:"Downtrend",4:"HighVol"}
WEEKLY_THR = 0.003; CONSISTENCY_THR = 0.50
HIGHVOL_CID=4; UP_CID=0; DOWN_CID=3

V72L = ['hurst_rs','ou_theta','entropy_rate','kramers_up','wavelet_er',
        'vwap_dist','hour_enc','dow_enc','quantum_flow','quantum_flow_h4',
        'quantum_momentum','quantum_vwap_conf','quantum_divergence','quantum_div_strength',
        'vpin','sig_quad_var','har_rv_ratio','hawkes_eta']
MATURITY = ['stretch_100','stretch_200','pct_to_extreme_50']
MOM24H = ['ret_24h_signed','ret_24h_abs']
SL_HARD=-4.0; MAX_HOLD=60; TRAIL_ACT=3.0; TRAIL_GB=0.60

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
    return pd.concat(parts, ignore_index=True).sort_values(['time','direction']).drop_duplicates(['time','direction'], keep='first').reset_index(drop=True)

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
    return {"n":len(p), "pf":pf, "total":float(s)}

def evaluate(name, swing_csv, setups_glob, fp_csv, bundle_path):
    print(f"\n{'='*72}\n  {name}\n{'='*72}")
    n, C, atr, t2i = load_market(swing_csv)
    setups = load_setups(setups_glob)
    setup_idx = np.full(len(setups), -1, dtype=np.int64)
    for i, t in enumerate(setups['time']):
        ti = pd.Timestamp(t)
        if ti in t2i: setup_idx[i] = t2i[ti]
    keep = setup_idx >= 0
    setups = setups[keep].reset_index(drop=True); setup_idx = setup_idx[keep]

    pnls = np.full(len(setups), np.nan, dtype=np.float32)
    dirs = setups['direction'].values
    for i in range(len(setups)):
        p = sim(int(setup_idx[i]), int(dirs[i]), n, C, atr)
        if p is not None: pnls[i] = p
    valid = ~np.isnan(pnls)
    setups = setups[valid].reset_index(drop=True); setup_idx = setup_idx[valid]
    pnls = pnls[valid]; dirs = dirs[valid]

    V72L_X = setups[V72L].fillna(0).values.astype(np.float32)
    mat_X = np.zeros((len(setups), 3), dtype=np.float32)
    mom_X = np.zeros((len(setups), 2), dtype=np.float32)
    for i in range(len(setups)):
        mat_X[i] = maturity_at(int(setup_idx[i]), int(dirs[i]), C, atr)
        mom_X[i] = mom24h_at(int(setup_idx[i]), int(dirs[i]), C)
    feat23 = np.concatenate([V72L_X, mat_X, mom_X], axis=1)

    # Load fp + apply v91 relabel rule per block
    fp = pd.read_csv(fp_csv, parse_dates=["center_time"]).sort_values("end_idx").reset_index(drop=True)
    def v91_relabel(c, w, t):
        if c != HIGHVOL_CID: return c
        if w > WEEKLY_THR: return UP_CID
        if w < -WEEKLY_THR: return DOWN_CID
        if t > CONSISTENCY_THR:
            return UP_CID if w > 0 else DOWN_CID
        return c
    fp['v91_label'] = [v91_relabel(int(c), float(w), float(t))
                        for c,w,t in zip(fp['cluster'], fp['weekly_return_pct'], fp['trend_consistency'])]

    # Build a per-bar v91_cid lookup using block_end_idx
    block_ends = fp['end_idx'].values
    block_labels = fp['v91_label'].values
    new_cid_per_setup = np.zeros(len(setups), dtype=np.int32)
    for i in range(len(setups)):
        j = np.searchsorted(block_ends, setup_idx[i], side='right') - 1
        if j < 0: j = 0
        new_cid_per_setup[i] = int(block_labels[j])

    is_test = setups['time'].values >= HOLDOUT_START.to_datetime64()
    test_pnls = pnls[is_test]
    test_old_cid = setups['old_cid'].values[is_test]
    test_new_cid = new_cid_per_setup[is_test]
    test_feat = feat23[is_test]

    # Load v90 q_entry (current production)
    bundle = pickle.load(open(bundle_path, "rb"))
    q_entry = bundle['q_entry']

    # Score with OLD cid routing
    Q_old = np.full(len(test_pnls), -9.0, dtype=np.float32)
    for cid in q_entry:
        m = test_old_cid == cid
        if m.sum() > 0: Q_old[m] = q_entry[cid].predict(test_feat[m])

    # Score with v91 cid routing
    Q_v91 = np.full(len(test_pnls), -9.0, dtype=np.float32)
    for cid in q_entry:
        m = test_new_cid == cid
        if m.sum() > 0: Q_v91[m] = q_entry[cid].predict(test_feat[m])

    relabeled = (test_old_cid != test_new_cid)
    print(f"  Holdout: {len(test_pnls):,} setups, {relabeled.sum()} would be relabeled by v91 ({relabeled.sum()*100/max(len(test_pnls),1):.1f}%)")

    print(f"\n  {'thr':>5s}  {'OLD routing (current prod)':>32s}  {'v91 routing':>32s}  {'Δ R':>6s}")
    for thr in [0.5, 1.0, 2.0, 3.0, 5.0]:
        m_old = metrics(test_pnls[Q_old > thr])
        m_v91 = metrics(test_pnls[Q_v91 > thr])
        if m_old is None or m_v91 is None: continue
        print(f"  {thr:>5.2f}  N={m_old['n']:>5d} PF={m_old['pf']:>5.2f} R={m_old['total']:>+7.0f}  "
              f"N={m_v91['n']:>5d} PF={m_v91['pf']:>5.2f} R={m_v91['total']:>+7.0f}  "
              f"{m_v91['total']-m_old['total']:>+6.0f}")

    # Focused look: only the relabeled subset
    if relabeled.sum() > 0:
        print(f"\n  RELABELED SUBSET ({relabeled.sum()} setups):")
        for thr in [0.5, 1.0, 2.0, 3.0]:
            m_old = metrics(test_pnls[relabeled & (Q_old > thr)])
            m_v91 = metrics(test_pnls[relabeled & (Q_v91 > thr)])
            if m_old is None or m_v91 is None: continue
            print(f"    thr={thr:>4.2f}  OLD: N={m_old['n']:>3d} PF={m_old['pf']:>5.2f} R={m_old['total']:>+6.0f}    "
                  f"v91: N={m_v91['n']:>3d} PF={m_v91['pf']:>5.2f} R={m_v91['total']:>+6.0f}")

evaluate("Oracle XAU — v91 consistency-relabel impact",
         f"{DATA}/swing_v5_xauusd.csv",
         f"{DATA}/setups_*_v72l.csv",
         f"{DATA}/regime_fingerprints_K4.csv",
         f"{PROJECT}/products/models/oracle_xau_validated.pkl")
evaluate("Oracle BTC — v91 consistency-relabel impact",
         f"{DATA}/swing_v5_btc.csv",
         f"{DATA}/setups_*_v72l_btc.csv",
         f"{DATA}/regime_fingerprints_btc_K5.csv",
         f"{PROJECT}/products/models/oracle_btc_validated.pkl")
