"""v90 deploy step — train new q_entry bundles with V72L + maturity + 24h features.

Saves bundle copies as oracle_*_validated_v90mom24h.pkl that include the
existing confirm/meta/exit heads but with a new 23-input q_entry.
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
    if t_idx < 288: return [0.0, 0.0]
    c0 = float(C[t_idx-288]); ct = float(C[t_idx])
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

def train_for(name, swing_csv, setups_glob, bundle_old_path):
    print(f"\n[{name}]")
    t0 = time.time()
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
        p = simulate_trade(int(setup_idx[i]), int(dirs[i]), n, C, atr)
        if p is not None: pnls[i] = p
    valid = ~np.isnan(pnls)
    setups = setups[valid].reset_index(drop=True); setup_idx = setup_idx[valid]
    pnls = pnls[valid]; dirs = dirs[valid]
    print(f"  pnls: {len(pnls):,} setups")

    V72L_X = setups[V72L].fillna(0).values.astype(np.float32)
    mat_X = np.zeros((len(setups), 3), dtype=np.float32)
    mom_X = np.zeros((len(setups), 2), dtype=np.float32)
    for i in range(len(setups)):
        mat_X[i] = maturity_at(int(setup_idx[i]), int(dirs[i]), C, atr)
        mom_X[i] = mom24h_at(int(setup_idx[i]), int(dirs[i]), C)
    full_X = np.concatenate([V72L_X, mat_X, mom_X], axis=1)
    print(f"  feature matrix: {full_X.shape}")

    is_train = setups['time'].values < HOLDOUT_START.to_datetime64()
    cids = setups['old_cid'].values

    q_new = {}
    for cid in sorted(set(cids)):
        mask_tr = (cids==cid) & is_train
        if mask_tr.sum() < MIN_CLUSTER: continue
        m = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         random_state=42, verbosity=0, objective='reg:squarederror')
        m.fit(full_X[mask_tr], pnls[mask_tr])
        q_new[int(cid)] = m
        print(f"    cid={cid}: trained on {mask_tr.sum():,}")

    # Save new bundle (replace q_entry, keep everything else)
    bundle = pickle.load(open(bundle_old_path, "rb"))
    bundle['q_entry'] = q_new
    bundle['q_entry_features'] = V72L + MATURITY + MOM24H
    bundle['v90_mom24h'] = True
    out_path = bundle_old_path.replace('.pkl', '_v90mom24h.pkl')
    with open(out_path, 'wb') as f: pickle.dump(bundle, f)
    print(f"  Saved: {out_path}  ({len(q_new)} cluster heads, 23 inputs each)  in {time.time()-t0:.0f}s")

if __name__ == "__main__":
    train_for("XAU",
              f"{DATA}/swing_v5_xauusd.csv",
              f"{DATA}/setups_*_v72l.csv",
              f"{PROJECT}/products/models/oracle_xau_validated.pkl")
    train_for("BTC",
              f"{DATA}/swing_v5_btc.csv",
              f"{DATA}/setups_*_v72l_btc.csv",
              f"{PROJECT}/products/models/oracle_btc_validated.pkl")
