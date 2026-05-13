"""Wider SL — full holdout PnL impact.

The SL-hunt diagnostic showed 56% of SL hits would recover within 60
bars. But wider SL means each genuine loser costs more. So this script
re-runs the FULL holdout with SL = -4, -5, -6, -8, -10 and reports:
  - Total R / PF / WR at each min_q gate (matches production gates)
  - Δ R vs baseline (-4)

Same q_entry routing as production. No retraining. Pure exit policy.
"""
import os, sys, glob as _glob, pickle
import numpy as np, pandas as pd

PROJECT = "/home/jay/Desktop/new-model-zigzag"
DATA = f"{PROJECT}/data"
HOLDOUT_START = pd.Timestamp("2024-12-12")
MAX_HOLD = 60; TRAIL_ACT = 3.0; TRAIL_GB = 0.60

V72L = ['hurst_rs','ou_theta','entropy_rate','kramers_up','wavelet_er',
        'vwap_dist','hour_enc','dow_enc','quantum_flow','quantum_flow_h4',
        'quantum_momentum','quantum_vwap_conf','quantum_divergence','quantum_div_strength',
        'vpin','sig_quad_var','har_rv_ratio','hawkes_eta']

def load_market(swing_csv):
    sw = pd.read_csv(swing_csv, parse_dates=["time"]).sort_values("time")\
           .drop_duplicates('time', keep='last').reset_index(drop=True)
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
    if not np.isfinite(ea) or ea <= 0: return [0.0, 0.0, 0.5]
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

def sim_sl(t_idx, d, n, C, atr, sl_hard):
    ep = C[t_idx]; ea = atr[t_idx]
    if not np.isfinite(ea) or ea <= 0: return None
    peak = 0.0; max_k = min(MAX_HOLD, n - t_idx - 1)
    for k in range(1, max_k + 1):
        bar = t_idx + k
        R = d * (C[bar] - ep) / ea
        if R <= sl_hard: return R
        if R > peak: peak = R
        if peak >= TRAIL_ACT and R <= peak * (1.0 - TRAIL_GB): return R
    last = min(t_idx + max_k, n - 1)
    return d * (C[last] - ep) / ea

def metrics(p):
    p = np.asarray(p)
    if len(p) == 0: return None
    pos = p[p > 0]; neg = p[p <= 0]
    pf = pos.sum() / max(-neg.sum(), 1e-9) if len(neg) > 0 else 99
    return {"n": len(p), "pf": pf, "total": float(p.sum()), "wr": (p > 0).mean()}

def run(name, swing_csv, setups_glob, bundle_path):
    print(f"\n{'='*78}\n  {name}\n{'='*78}", flush=True)
    sw, n, C, atr, t2i = load_market(swing_csv)
    setups = load_setups(setups_glob)
    idx = np.full(len(setups), -1, dtype=np.int64)
    for i, t in enumerate(setups['time']):
        ti = pd.Timestamp(t)
        if ti in t2i: idx[i] = t2i[ti]
    keep = idx >= 0
    setups = setups[keep].reset_index(drop=True); idx = idx[keep]
    is_test = setups['time'].values >= HOLDOUT_START.to_datetime64()
    setups = setups[is_test].reset_index(drop=True); idx = idx[is_test]
    dirs = setups['direction'].values
    print(f"  holdout setups: {len(setups):,}", flush=True)

    # 23-feat
    V72L_X = setups[V72L].fillna(0).values.astype(np.float32)
    mat_X = np.zeros((len(setups), 3), dtype=np.float32)
    mom_X = np.zeros((len(setups), 2), dtype=np.float32)
    for i in range(len(setups)):
        mat_X[i] = maturity_at(int(idx[i]), int(dirs[i]), C, atr)
        mom_X[i] = mom24h_at(int(idx[i]), int(dirs[i]), C)
    feat23 = np.concatenate([V72L_X, mat_X, mom_X], axis=1)
    block_cid = setups['old_cid'].values.astype(np.int32)

    bundle = pickle.load(open(bundle_path, "rb"))
    q = bundle['q_entry']
    Q = np.full(len(setups), -9.0, dtype=np.float32)
    for cid in q:
        m = block_cid == cid
        if m.sum() > 0: Q[m] = q[cid].predict(feat23[m])

    # Simulate with multiple SL levels
    SL_LEVELS = [-4.0, -5.0, -6.0, -8.0, -10.0]
    pnls_by_sl = {}
    for sl in SL_LEVELS:
        pnls = np.full(len(setups), np.nan, dtype=np.float32)
        for i in range(len(setups)):
            p = sim_sl(int(idx[i]), int(dirs[i]), n, C, atr, sl)
            if p is not None: pnls[i] = p
        pnls_by_sl[sl] = pnls
        print(f"  simulated SL={sl}", flush=True)

    valid = ~np.isnan(pnls_by_sl[-4.0])
    print(f"  valid setups: {valid.sum():,}\n")

    print(f"  {'SL':>5s}  ", end="")
    for thr_q in [0.5, 1.0, 2.0, 3.0, 5.0]:
        print(f"{'q>'+str(thr_q):>28s}  ", end="")
    print()

    base_at = {}
    for thr_q in [0.5, 1.0, 2.0, 3.0, 5.0]:
        m = valid & (Q > thr_q)
        base_at[thr_q] = metrics(pnls_by_sl[-4.0][m])

    for sl in SL_LEVELS:
        print(f"  {sl:>5.1f}  ", end="")
        for thr_q in [0.5, 1.0, 2.0, 3.0, 5.0]:
            m = valid & (Q > thr_q)
            mm = metrics(pnls_by_sl[sl][m])
            if mm is None:
                print(f"{'--':>28s}  ", end="")
            else:
                d = mm['total'] - base_at[thr_q]['total']
                print(f"N={mm['n']:>4d} PF={mm['pf']:>4.2f} R={mm['total']:>+6.0f}({d:+5.0f})  ", end="")
        print()

if __name__ == "__main__":
    run("XAU", f"{DATA}/swing_v5_xauusd.csv", f"{DATA}/setups_*_v72l.csv",
        f"{PROJECT}/products/models/oracle_xau_validated.pkl")
    run("BTC", f"{DATA}/swing_v5_btc.csv", f"{DATA}/setups_*_v72l_btc.csv",
        f"{PROJECT}/products/models/oracle_btc_validated.pkl")
