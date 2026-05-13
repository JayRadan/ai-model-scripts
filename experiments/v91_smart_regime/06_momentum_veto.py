"""Momentum-disagreement veto — no model retraining required.

Rule:
  if regime == Uptrend   AND 24h-return < -THR  AND direction = +1  -> VETO
  if regime == Downtrend AND 24h-return > +THR  AND direction = -1  -> VETO

Validated on holdout. Compare baseline (current prod, no veto) to
veto-applied for several THR values.

This is orthogonal to regime classification — it's a guard at /decide
time. Easy to wire, easy to remove.
"""
import os, sys, glob as _glob, pickle, json
import numpy as np, pandas as pd

PROJECT = "/home/jay/Desktop/new-model-zigzag"
DATA = f"{PROJECT}/data"
SERVER = "/home/jay/Desktop/my-agents-and-website/commercial/server"

HOLDOUT_START = pd.Timestamp("2024-12-12")
UP_CID = 0; DOWN_CID = 3

V72L = ['hurst_rs','ou_theta','entropy_rate','kramers_up','wavelet_er',
        'vwap_dist','hour_enc','dow_enc','quantum_flow','quantum_flow_h4',
        'quantum_momentum','quantum_vwap_conf','quantum_divergence','quantum_div_strength',
        'vpin','sig_quad_var','har_rv_ratio','hawkes_eta']
SL_HARD=-4.0; MAX_HOLD=60; TRAIL_ACT=3.0; TRAIL_GB=0.60

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
        try: cid = int(cid_str)
        except: continue
        df = pd.read_csv(f, parse_dates=["time"]); df['old_cid'] = cid
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

def raw_ret24h(t_idx, C):
    if t_idx < 288: return 0.0
    c0 = float(C[t_idx-288]); ct = float(C[t_idx])
    if c0 <= 0: return 0.0
    return (ct - c0) / c0

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

def run(name, swing_csv, setups_glob, bundle_path):
    print(f"\n{'='*78}\n  {name}\n{'='*78}", flush=True)
    sw, n, C, atr, t2i = load_market(swing_csv)
    setups = load_setups(setups_glob)

    setup_idx = np.full(len(setups), -1, dtype=np.int64)
    for i, t in enumerate(setups['time']):
        ti = pd.Timestamp(t)
        if ti in t2i: setup_idx[i] = t2i[ti]
    keep = setup_idx >= 0
    setups = setups[keep].reset_index(drop=True); setup_idx = setup_idx[keep]

    is_test = setups['time'].values >= HOLDOUT_START.to_datetime64()
    setups = setups[is_test].reset_index(drop=True); setup_idx = setup_idx[is_test]
    print(f"  holdout setups: {len(setups):,}", flush=True)

    pnls = np.full(len(setups), np.nan, dtype=np.float32)
    dirs = setups['direction'].values
    for i in range(len(setups)):
        p = sim(int(setup_idx[i]), int(dirs[i]), n, C, atr)
        if p is not None: pnls[i] = p
    valid = ~np.isnan(pnls)
    setups = setups[valid].reset_index(drop=True); setup_idx = setup_idx[valid]
    pnls = pnls[valid]; dirs = dirs[valid]

    # Build 23-feature input
    V72L_X = setups[V72L].fillna(0).values.astype(np.float32)
    mat_X = np.zeros((len(setups), 3), dtype=np.float32)
    mom_X = np.zeros((len(setups), 2), dtype=np.float32)
    raw_24h = np.zeros(len(setups), dtype=np.float32)
    for i in range(len(setups)):
        mat_X[i] = maturity_at(int(setup_idx[i]), int(dirs[i]), C, atr)
        mom_X[i] = mom24h_at(int(setup_idx[i]), int(dirs[i]), C)
        raw_24h[i] = raw_ret24h(int(setup_idx[i]), C)
    feat23 = np.concatenate([V72L_X, mat_X, mom_X], axis=1)
    block_cid = setups['old_cid'].values.astype(np.int32)

    # Score with current production q_entry
    bundle = pickle.load(open(bundle_path, "rb"))
    q = bundle['q_entry']
    Q = np.full(len(pnls), -9.0, dtype=np.float32)
    for cid in q:
        m = block_cid == cid
        if m.sum() > 0: Q[m] = q[cid].predict(feat23[m])

    # Veto mask: regime says Up but 24h is sharply down (long), or
    # regime says Down but 24h is sharply up (short)
    def veto_mask(thr):
        m_up_long_bad   = (block_cid == UP_CID)   & (dirs ==  1) & (raw_24h < -thr)
        m_down_short_bad = (block_cid == DOWN_CID) & (dirs == -1) & (raw_24h >  thr)
        return m_up_long_bad | m_down_short_bad

    print(f"\n  baseline (no veto) at min_q thresholds:")
    print(f"  {'min_q':>6s}  {'baseline':>34s}")
    base_at = {}
    for thr_q in [0.5, 1.0, 2.0, 3.0, 5.0]:
        m = Q > thr_q
        mm = metrics(pnls[m])
        base_at[thr_q] = mm
        print(f"  {thr_q:>6.2f}  N={mm['n']:>4d} PF={mm['pf']:>5.2f} R={mm['total']:>+8.0f} WR={mm['wr']*100:>4.1f}%")

    print(f"\n  veto sweep (apply veto + min_q gate):")
    print(f"  {'veto%':>5s}  ", end="")
    for thr_q in [0.5, 1.0, 2.0, 3.0, 5.0]:
        print(f"{'q>'+str(thr_q):>30s}  ", end="")
    print()
    for vthr in [0.003, 0.005, 0.008, 0.010, 0.015, 0.020]:
        v_mask = veto_mask(vthr)
        n_vetoed = int(v_mask.sum())
        print(f"  {vthr*100:>5.2f}  ", end="")
        for thr_q in [0.5, 1.0, 2.0, 3.0, 5.0]:
            keep = ~v_mask & (Q > thr_q)
            mm = metrics(pnls[keep])
            if mm is None:
                print(f"{'--':>30s}  ", end="")
            else:
                delta = mm['total'] - base_at[thr_q]['total']
                print(f"N={mm['n']:>4d} PF={mm['pf']:>4.2f} R={mm['total']:>+5.0f}({delta:+5.0f})  ", end="")
        print(f"   [{n_vetoed} vetoed]")

    print(f"\n  veto-only subset (the trades that would be killed):")
    for vthr in [0.005, 0.010, 0.015]:
        v_mask = veto_mask(vthr)
        for thr_q in [1.0, 3.0]:
            both = v_mask & (Q > thr_q)
            mm = metrics(pnls[both])
            if mm is None: continue
            print(f"    veto={vthr*100:.2f}% min_q={thr_q}: N={mm['n']:>3d} PF={mm['pf']:>4.2f} "
                  f"R={mm['total']:>+6.0f} WR={mm['wr']*100:.1f}%")

if __name__ == "__main__":
    run("XAU",
        f"{DATA}/swing_v5_xauusd.csv",
        f"{DATA}/setups_*_v72l.csv",
        f"{PROJECT}/products/models/oracle_xau_validated.pkl")
    run("BTC",
        f"{DATA}/swing_v5_btc.csv",
        f"{DATA}/setups_*_v72l_btc.csv",
        f"{PROJECT}/products/models/oracle_btc_validated.pkl")
