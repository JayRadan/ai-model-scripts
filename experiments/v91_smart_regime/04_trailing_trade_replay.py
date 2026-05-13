"""Trade-replay: block-aligned vs trailing-window regime routing.

For each holdout setup:
  1. Compute the block-aligned cid (current production)
  2. Compute the trailing-window cid (proposed)
  3. Score the setup with q_entry[block_cid] AND q_entry[trailing_cid]
  4. Apply min_q gate (3.0) to each
  5. Simulate the same exit rules
  6. Compare total R / PF on the kept setups

Same harness shape as v91/03 but the relabel rule is trailing classification
instead of weekly+consistency override.
"""
import os, sys, glob as _glob, pickle, json
import numpy as np, pandas as pd

PROJECT = "/home/jay/Desktop/new-model-zigzag"
DATA = f"{PROJECT}/data"
SERVER = "/home/jay/Desktop/my-agents-and-website/commercial/server"
sys.path.insert(0, SERVER)
from decision_engine import regime as _regime

HOLDOUT_START = pd.Timestamp("2024-12-12")
NAMES = {0:"Uptrend",1:"MeanRevert",2:"TrendRange",3:"Downtrend",4:"HighVol"}

V72L = ['hurst_rs','ou_theta','entropy_rate','kramers_up','wavelet_er',
        'vwap_dist','hour_enc','dow_enc','quantum_flow','quantum_flow_h4',
        'quantum_momentum','quantum_vwap_conf','quantum_divergence','quantum_div_strength',
        'vpin','sig_quad_var','har_rv_ratio','hawkes_eta']
MATURITY = ['stretch_100','stretch_200','pct_to_extreme_50']
MOM24H = ['ret_24h_signed','ret_24h_abs']
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

def evaluate(name, swing_csv, setups_glob, selector_json, bundle_path):
    print(f"\n{'='*78}\n  {name}\n{'='*78}")
    sw, n, C, atr, t2i = load_market(swing_csv)
    selector = json.loads(open(selector_json).read())
    setups = load_setups(setups_glob)

    setup_idx = np.full(len(setups), -1, dtype=np.int64)
    for i, t in enumerate(setups['time']):
        ti = pd.Timestamp(t)
        if ti in t2i: setup_idx[i] = t2i[ti]
    keep = setup_idx >= 0
    setups = setups[keep].reset_index(drop=True); setup_idx = setup_idx[keep]

    # holdout only
    is_test = setups['time'].values >= HOLDOUT_START.to_datetime64()
    setups = setups[is_test].reset_index(drop=True); setup_idx = setup_idx[is_test]
    print(f"  holdout setups: {len(setups):,}")

    # SUBSAMPLE for speed: every 10th setup (statistically representative)
    SAMPLE = 10
    setups = setups.iloc[::SAMPLE].reset_index(drop=True)
    setup_idx = setup_idx[::SAMPLE]
    print(f"  subsampled 1/{SAMPLE}: {len(setups):,} setups")

    # simulate PnL for all setups
    pnls = np.full(len(setups), np.nan, dtype=np.float32)
    dirs = setups['direction'].values
    for i in range(len(setups)):
        p = sim(int(setup_idx[i]), int(dirs[i]), n, C, atr)
        if p is not None: pnls[i] = p
    valid = ~np.isnan(pnls)
    setups = setups[valid].reset_index(drop=True); setup_idx = setup_idx[valid]
    pnls = pnls[valid]; dirs = dirs[valid]
    print(f"  valid setups (PnL simulable): {len(setups):,}")

    # features for q_entry
    V72L_X = setups[V72L].fillna(0).values.astype(np.float32)
    mat_X = np.zeros((len(setups), 3), dtype=np.float32)
    mom_X = np.zeros((len(setups), 2), dtype=np.float32)
    for i in range(len(setups)):
        mat_X[i] = maturity_at(int(setup_idx[i]), int(dirs[i]), C, atr)
        mom_X[i] = mom24h_at(int(setup_idx[i]), int(dirs[i]), C)
    feat23 = np.concatenate([V72L_X, mat_X, mom_X], axis=1)

    # Compute trailing cid at each setup time. CACHED by unique bar index.
    unique_idx = sorted(set(int(i) for i in setup_idx))
    print(f"  computing trailing cid at {len(unique_idx):,} unique bars...")
    trail_at_idx = {}
    for k, i in enumerate(unique_idx):
        sub = sw.iloc[:i+1]
        try:
            trail_at_idx[i] = int(_regime.classify_trailing(sub, selector))
        except ValueError:
            trail_at_idx[i] = -1
        if (k+1) % 200 == 0:
            print(f"    {k+1}/{len(unique_idx)}", flush=True)

    trail_cid = np.array([trail_at_idx[int(i)] for i in setup_idx], dtype=np.int32)
    block_cid = setups['old_cid'].values.astype(np.int32)
    relabeled = (trail_cid != block_cid) & (trail_cid >= 0)
    print(f"  setups where trailing != block: {relabeled.sum()} ({relabeled.mean()*100:.1f}%)")

    # Load v90 q_entry
    bundle = pickle.load(open(bundle_path, "rb"))
    q_entry = bundle['q_entry']

    # Score with BLOCK cid routing (current production)
    Q_block = np.full(len(pnls), -9.0, dtype=np.float32)
    for cid in q_entry:
        m = block_cid == cid
        if m.sum() > 0: Q_block[m] = q_entry[cid].predict(feat23[m])

    # Score with TRAILING cid routing (proposed)
    Q_trail = np.full(len(pnls), -9.0, dtype=np.float32)
    for cid in q_entry:
        m = trail_cid == cid
        if m.sum() > 0: Q_trail[m] = q_entry[cid].predict(feat23[m])

    print(f"\n  {'thr':>5s}  {'BLOCK routing (current prod)':>34s}  "
          f"{'TRAIL routing':>34s}  {'Δ R':>7s}")
    for thr in [0.5, 1.0, 2.0, 3.0, 5.0]:
        mb = metrics(pnls[Q_block > thr])
        mt = metrics(pnls[Q_trail > thr])
        if mb is None or mt is None: continue
        print(f"  {thr:>5.2f}  "
              f"N={mb['n']:>4d} PF={mb['pf']:>5.2f} R={mb['total']:>+8.0f} WR={mb['wr']*100:>4.1f}%  "
              f"N={mt['n']:>4d} PF={mt['pf']:>5.2f} R={mt['total']:>+8.0f} WR={mt['wr']*100:>4.1f}%  "
              f"{mt['total']-mb['total']:>+7.0f}")

    # Focused: only the relabeled subset (the staleness-affected trades)
    if relabeled.sum() > 0:
        print(f"\n  RELABELED-ONLY SUBSET ({relabeled.sum()} setups):")
        print(f"  {'thr':>5s}  {'BLOCK':>32s}  {'TRAIL':>32s}  {'Δ R':>7s}")
        for thr in [0.5, 1.0, 2.0, 3.0]:
            mb = metrics(pnls[relabeled & (Q_block > thr)])
            mt = metrics(pnls[relabeled & (Q_trail > thr)])
            if mb is None or mt is None: continue
            print(f"  {thr:>5.2f}  "
                  f"N={mb['n']:>3d} PF={mb['pf']:>5.2f} R={mb['total']:>+7.0f}  "
                  f"N={mt['n']:>3d} PF={mt['pf']:>5.2f} R={mt['total']:>+7.0f}  "
                  f"{mt['total']-mb['total']:>+7.0f}")

if __name__ == "__main__":
    evaluate("Oracle XAU — block vs trailing trade replay",
             f"{DATA}/swing_v5_xauusd.csv",
             f"{DATA}/setups_*_v72l.csv",
             f"{SERVER}/decision_engine/data/regime_selector_K4.json",
             f"{PROJECT}/products/models/oracle_xau_validated.pkl")
    evaluate("Oracle BTC — block vs trailing trade replay",
             f"{DATA}/swing_v5_btc.csv",
             f"{DATA}/setups_*_v72l_btc.csv",
             f"{SERVER}/decision_engine/data/regime_selector_btc_K5.json",
             f"{PROJECT}/products/models/oracle_btc_validated.pkl")
