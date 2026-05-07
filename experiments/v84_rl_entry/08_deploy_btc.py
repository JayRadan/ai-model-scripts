"""Minimal BTC RL deploy — train & save bundle only."""
import pickle, glob, os, numpy as np, pandas as pd
from xgboost import XGBRegressor, XGBClassifier
import time as _time

PROJECT = '/home/jay/Desktop/new-model-zigzag'
DATA = f'{PROJECT}/data'
SERVER = '/home/jay/Desktop/my-agents-and-website/commercial/server/decision_engine'

GLOBAL_CUTOFF = pd.Timestamp('2024-12-12 00:00:00')
V72L_FEATS = ['hurst_rs','ou_theta','entropy_rate','kramers_up','wavelet_er',
    'vwap_dist','hour_enc','dow_enc','quantum_flow','quantum_flow_h4',
    'quantum_momentum','quantum_vwap_conf','quantum_divergence','quantum_div_strength',
    'vpin','sig_quad_var','har_rv_ratio','hawkes_eta']
EXIT_FEATS = ['unrealized_pnl_R','bars_held','pnl_velocity',
    'hurst_rs','ou_theta','entropy_rate','kramers_up','wavelet_er',
    'quantum_flow','quantum_flow_h4','vwap_dist']
META_FEATS = V72L_FEATS + ['direction','cid']
MIN_Q = 0.3; MAX_HOLD = 60; MIN_HOLD = 2

t_total = _time.time()

# ── 1. Load ──
print("[1/4] Loading...", end='', flush=True); t0 = _time.time()
swing = pd.read_csv(f'{DATA}/swing_v5_btc.csv', parse_dates=['time'])
swing = swing.sort_values('time').reset_index(drop=True)
n = len(swing)
C = swing['close'].values.astype(np.float64)
H = swing['high'].values.astype(np.float64)
L = swing['low'].values.astype(np.float64)
tr = np.concatenate([[H[0]-L[0]], np.maximum.reduce([
    H[1:]-L[1:], np.abs(H[1:]-C[:-1]), np.abs(L[1:]-C[:-1])
])])
atr = pd.Series(tr).rolling(14, min_periods=14).mean().values
t2i = {t: i for i, t in enumerate(swing['time'].values)}

fp_new = pd.read_csv(f'{PROJECT}/experiments/v83_range_position_filter/regime_fingerprints_btc_4h.csv')
fp_new['center_time'] = pd.to_datetime(fp_new['center_time'])
fp_new = fp_new.sort_values('center_time')
new_regime = np.full(len(swing), -1, dtype=int)
for _, row in fp_new.iterrows():
    s, e = int(row['start_idx']), int(row['end_idx'])
    if 0 <= s < e <= len(swing):
        new_regime[s:e] = int(row['new_label'])

all_setups = []
for f in sorted(glob.glob(f'{DATA}/setups_*_v72l_btc.csv')):
    cid = int(os.path.basename(f).split('_')[1])
    df = pd.read_csv(f, parse_dates=['time'])
    df['old_cid'] = cid
    all_setups.append(df)
all_df = pd.concat(all_setups, ignore_index=True).sort_values('time').reset_index(drop=True)
# Vectorized cid assignment — use t2i dict (same as 07_btc_rl.py)
new_cids = np.full(len(all_df), -1, dtype=int)
times_arr = all_df['time'].values
for i in range(len(all_df)):
    idx = t2i.get(times_arr[i], -1)
    if idx >= 0 and idx < len(new_regime):
        new_cids[i] = new_regime[idx]
all_df['cid'] = new_cids
all_df = all_df[all_df['cid'] >= 0].reset_index(drop=True)
print(f" {_time.time()-t0:.0f}s — {len(all_df):,} setups", flush=True)

# ── 2. PnL labels ──
print("[2/4] Computing PnL labels...", end='', flush=True); t0 = _time.time()
times_arr = all_df['time'].values
indices = np.array([t2i.get(t, -1) for t in times_arr])
dirs = all_df['direction'].values.astype(np.int32)
pnl_r = np.zeros(len(all_df), dtype=np.float64)

for i in range(len(all_df)):
    idx = indices[i]
    if idx < 0: continue
    d = dirs[i]
    ep = C[idx]; ea = atr[idx]
    if not np.isfinite(ea) or ea <= 0: continue
    tp = ep + d * 2.0 * ea - 0.4; sl_val = ep - d * 1.0 * ea + 0.4
    end = min(idx + 41, n)
    outcome = 0.0
    for k in range(idx + 1, end):
        if d == 1:
            if L[k] <= sl_val: outcome = -1.0; break
            if H[k] >= tp: outcome = 2.0; break
        else:
            if H[k] >= sl_val: outcome = -1.0; break
            if L[k] <= tp: outcome = 2.0; break
    pnl_r[i] = outcome

all_df['pnl_r'] = pnl_r

# Forward-fill features
phys_lookup = all_df[['time'] + V72L_FEATS].drop_duplicates('time', keep='last').set_index('time')
for col in V72L_FEATS:
    swing[col] = phys_lookup[col].reindex(swing.set_index('time').index, method='ffill').values
    swing[col] = swing[col].fillna(0)

train = all_df[all_df['time'] < GLOBAL_CUTOFF].reset_index(drop=True)
test = all_df[all_df['time'] >= GLOBAL_CUTOFF].reset_index(drop=True)
print(f" {_time.time()-t0:.0f}s — train={len(train):,} test={len(test):,}", flush=True)

# ── 3. Train RL Q-models ──
print("[3/4] Training RL Q-entry...", flush=True)
q_entry = {}
for cid in range(5):
    g = train[train['cid'] == cid]
    if len(g) < 500: continue
    X = g[V72L_FEATS].fillna(0).values; y = g['pnl_r'].values
    mdl = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                       subsample=0.8, colsample_bytree=0.8, verbosity=0)
    mdl.fit(X, y); q_entry[cid] = mdl
    pred = mdl.predict(X); pos = (pred > MIN_Q).sum()
    wr = (y[pred > MIN_Q] > 0).mean() if pos > 0 else 0
    print(f"  C{cid}: {pos:,} Q>0.3R, WR={wr:.1%}", flush=True)

def gen_rl(df):
    rows = []
    for cid in sorted(df['cid'].unique()):
        if cid not in q_entry: continue
        g = df[df['cid'] == cid]; X = g[V72L_FEATS].fillna(0).values
        q_pred = q_entry[cid].predict(X)
        s = g[q_pred >= MIN_Q].copy(); s['rule'] = 'RL'; rows.append(s)
    return pd.concat(rows, ignore_index=True).sort_values('time').reset_index(drop=True) if rows else pd.DataFrame()

# ── 4. Confirm + Exit + Meta ──
print("[4/4] Training confirm + exit + meta...", flush=True)
rl_train = gen_rl(train)

def train_conf(train_df, features):
    mdls, thrs = {}, {}
    for (cid, rule), grp in train_df.groupby(['cid', 'rule']):
        if len(grp) < 100: continue
        grp = grp.sort_values('time').reset_index(drop=True)
        spl = int(len(grp) * 0.80); trn, vd = grp.iloc[:spl], grp.iloc[spl:]
        if len(vd) < 20: continue
        mdl = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', verbosity=0)
        mdl.fit(trn[features].fillna(0).values, trn['label'].astype(int).values)
        proba = mdl.predict_proba(vd[features].fillna(0).values)[:, 1]
        y_vd = vd['label'].astype(int).values
        best_thr, best_pf = 0.5, 0.0
        for thr in np.arange(0.30, 0.70, 0.05):
            m = proba >= thr
            if m.sum() < 5: continue
            w = y_vd[m].sum(); l_ = m.sum() - w
            if l_ == 0: continue
            pf = (w * 2.0) / (l_ * 1.0)
            if pf > best_pf: best_pf, best_thr = pf, float(thr)
        m = proba >= best_thr
        if m.sum() < 5 or best_pf < 0.8: continue
        mdls[(cid, rule)] = mdl; thrs[(cid, rule)] = best_thr
    return mdls, thrs

def confirm_setups(setups, mdls, thrs, features):
    rows = []
    for (cid, rule), grp in setups.groupby(['cid', 'rule']):
        if (cid, rule) not in mdls: continue
        X = grp[features].fillna(0).values; p = mdls[(cid, rule)].predict_proba(X)[:, 1]
        rows.append(grp[p >= thrs[(cid, rule)]].copy())
    return pd.concat(rows, ignore_index=True).sort_values('time').reset_index(drop=True) if rows else pd.DataFrame()

c_mdls, c_thrs = train_conf(rl_train, V72L_FEATS)
tc = confirm_setups(rl_train, c_mdls, c_thrs, V72L_FEATS)
print(f"  Confirm: {len(c_mdls)} models, {len(tc):,} confirmed", flush=True)

# Exit
ctx_np = swing[EXIT_FEATS[3:]].fillna(0).values.astype(np.float64); rows_e = []
for _, s in tc.iterrows():
    tm = s['time']
    if tm not in t2i: continue
    ei = t2i[tm]; d = int(s['direction']); ep = C[ei]; ea = atr[ei]
    if not np.isfinite(ea) or ea <= 0: continue
    end = min(ei + MAX_HOLD + 1, n)
    if end - ei < 10: continue
    pnls = np.array([d * (C[k] - ep) / ea for k in range(ei + 1, end)])
    if len(pnls) < 5: continue
    for b in range(len(pnls)):
        bi = ei + 1 + b; cp = pnls[b]
        if cp < -4.0: break
        rem = pnls[b+1:] if b + 1 < len(pnls) else np.array([cp])
        br = rem.max() if len(rem) > 0 else cp
        if br < cp - 0.3: lbl = 1
        elif br > cp + 0.3: lbl = 0
        else: continue
        v = cp - pnls[b-3] if b >= 3 else (cp - pnls[0] if b >= 1 else 0.0)
        row = {'unrealized_pnl_R': cp, 'bars_held': float(b+1), 'pnl_velocity': v, 'label': lbl}
        if bi < n:
            for j, f_ in enumerate(EXIT_FEATS[3:]): row[f_] = float(ctx_np[bi, j])
        else:
            for f_ in EXIT_FEATS[3:]: row[f_] = 0.0
        rows_e.append(row)
ed = pd.DataFrame(rows_e)
exit_mdl = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', verbosity=0)
if len(ed) > 100: exit_mdl.fit(ed[EXIT_FEATS].fillna(0).values, ed['label'].values)
print(f"  Exit: {len(ed):,} samples", flush=True)

# Meta
ctx_cols = ['hurst_rs','ou_theta','entropy_rate','kramers_up','wavelet_er',
            'quantum_flow','quantum_flow_h4','vwap_dist']
ctx_arr = swing[ctx_cols].fillna(0).values.astype(np.float64)

# Simulate for meta labels
entries_meta = []
for _, s in tc.iterrows():
    tm = s['time']
    if tm not in t2i: continue
    ei = t2i[tm]; ea = atr[ei]
    if not np.isfinite(ea) or ea <= 0: continue
    entries_meta.append((ei, int(s['direction']), tm, int(s['cid'])))
Nm = len(entries_meta)
if Nm > 0:
    nf = 3 + len(ctx_cols)
    Xm = np.zeros((Nm * MAX_HOLD, nf), dtype=np.float32)
    vm = np.zeros(Nm * MAX_HOLD, dtype=bool)
    cm = np.full((Nm, MAX_HOLD), np.nan, dtype=np.float64)
    for rank, (ei, d, _, _) in enumerate(entries_meta):
        ep = C[ei]; ea2 = atr[ei]
        for k in range(1, MAX_HOLD + 1):
            bar = ei + k
            if bar >= n: break
            cp = d * (C[bar] - ep) / ea2; cm[rank, k-1] = cp
            if k < MIN_HOLD: continue
            p3 = d * (C[bar-3] - ep) / ea2 if k >= 3 else cp
            row_idx = rank * MAX_HOLD + (k-1)
            Xm[row_idx, 0] = cp; Xm[row_idx, 1] = float(k); Xm[row_idx, 2] = cp - p3
            Xm[row_idx, 3:] = ctx_arr[bar]; vm[row_idx] = True
    pm_ = np.zeros(Nm * MAX_HOLD, dtype=np.float32)
    if exit_mdl is not None and vm.any(): pm_[vm] = exit_mdl.predict_proba(Xm[vm])[:, 1]
    rows_m = []
    for rank, (ei, d, tm, cid_v) in enumerate(entries_meta):
        ep = C[ei]; xi, xr = None, 'max'
        for k in range(1, MAX_HOLD + 1):
            bar = ei + k
            if bar >= n: break
            cp = cm[rank, k-1]
            if not np.isfinite(cp): break
            if cp < -4.0: xi, xr = bar, 'hard_sl'; break
            if k >= MIN_HOLD and exit_mdl is not None:
                if pm_[rank * MAX_HOLD + (k-1)] >= 0.55:
                    xi, xr = bar, 'ml_exit'; break
        if xi is None: xi = min(ei + MAX_HOLD, n - 1); xr = 'max'
        pnl = d * (C[xi] - ep) / ea
        rows_m.append({'time': tm, 'cid': cid_v, 'direction': d,
                       'bars': xi - ei, 'pnl_R': pnl, 'exit': xr})
    tt = pd.DataFrame(rows_m)
    tc['direction'] = tc['direction'].astype(int); tc['cid'] = tc['cid'].astype(int)
    md_ = tt.merge(tc[['time','cid','rule'] + V72L_FEATS], on=['time','cid','rule'], how='left')
    md_['meta_label'] = (md_['pnl_R'] > 0).astype(int)
    md_ = md_.sort_values('time').reset_index(drop=True)
    s_ = int(len(md_) * 0.80); mtr, mvd = md_.iloc[:s_], md_.iloc[s_:]
    meta_mdl = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', verbosity=0)
    meta_mdl.fit(mtr[META_FEATS].fillna(0).values, mtr['meta_label'].values)
    pv = meta_mdl.predict_proba(mvd[META_FEATS].fillna(0).values)[:, 1]
    pn = mvd['pnl_R'].values
    cands = []
    for thr in np.arange(0.40, 0.80, 0.025):
        mx = pv >= thr
        if mx.sum() < 20: continue
        pt = pn[mx]; pf_ = pt[pt > 0].sum() / max(-pt[pt <= 0].sum(), 1e-9)
        cands.append((thr, mx.sum(), pf_))
    base_pf = pn[pn > 0].sum() / max(-pn[pn <= 0].sum(), 1e-9)
    vc = [c for c in cands if c[2] >= base_pf * 0.95] or cands
    vc.sort(key=lambda c: (-c[2], c[1])); best_thr = vc[0][0]
else:
    meta_mdl = None; best_thr = 0.5

# ── Save ──
print("Saving...", end='', flush=True)
bundle = {
    'q_entry': q_entry, 'mdls': c_mdls, 'thrs': c_thrs,
    'exit_mdl': exit_mdl, 'meta_mdl': meta_mdl, 'meta_threshold': float(best_thr),
    'v72l_feats': V72L_FEATS, 'meta_feats': META_FEATS, 'exit_feats': EXIT_FEATS,
    'min_q': MIN_Q, 'rl_rule_name': 'RL',
    'version': 'v84-rl-btc', 'trained_on': 'BTC RL Entry (PF 3.82 vs 3.03 rule-based)',
}
os.makedirs(f'{SERVER}/models', exist_ok=True)
with open(f'{SERVER}/models/oracle_btc_validated.pkl', 'wb') as f:
    pickle.dump(bundle, f)
sz = os.path.getsize(f'{SERVER}/models/oracle_btc_validated.pkl') / 1024 / 1024
print(f" ✓ ({sz:.1f} MB)")
print(f"  Bundle: {len(q_entry)} Q-models + {len(c_mdls)} confirms")
print(f"\n✓ Done in {_time.time()-t_total:.0f}s")
