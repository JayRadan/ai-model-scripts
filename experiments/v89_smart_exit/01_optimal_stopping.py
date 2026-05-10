"""v89 Smart Exit — offline optimal-stopping Q-learning.

Per spec:
  - Q_exit(s) = current_R (deterministic, no learning needed)
  - Learn Q_hold(s) via backward fitted-Q iteration on logged v84 RL trades
  - Conservative Q_hold = 25th percentile of 5-model bootstrapped ensemble
  - Asymmetric reward: time-cost + loss-zone penalty + worsening + giveback
  - 3 auxiliary classifiers: recovery / breakdown / SL probabilities
  - Strict no-leakage: split by trade_id (not by row), chrono walk-forward
  - Trained only on trades v84 RL entry would actually take

Decision policy:
  if current_R <= -4R:                 EXIT (hard SL)
  elif bars_in_trade >= 40:            EXIT (max hold)
  elif current_R < 0 AND p_recover<.30 AND p_breakdown>.50:
                                       EXIT (loser breakdown)
  elif peak_R >= 1.5 AND current_R < 0.4*peak_R:
                                       EXIT (winner giveback)
  elif q_exit >= q_hold_p25:           EXIT (Q-stopping condition)
  else:                                HOLD
"""
import os, time, pickle, glob as _glob
import numpy as np, pandas as pd
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import roc_auc_score

PROJECT="/home/jay/Desktop/new-model-zigzag"
DATA=f"{PROJECT}/data"

V72L=['hurst_rs','ou_theta','entropy_rate','kramers_up','wavelet_er',
      'vwap_dist','hour_enc','dow_enc','quantum_flow','quantum_flow_h4',
      'quantum_momentum','quantum_vwap_conf','quantum_divergence','quantum_div_strength',
      'vpin','sig_quad_var','har_rv_ratio','hawkes_eta']
STATE_FEATS = (
    ['current_R','peak_R','drawdown_from_peak','bars_in_trade','bars_remaining',
     'mom_3bar','mom_10bar','vol_10bar','dist_to_SL','dist_to_TP'] + V72L
)
MAX_HOLD=40   # per user spec
SL_HARD=-4.0
GAMMA=0.99
N_FQI_ITER=4
N_ENSEMBLE=5

# ─────────────────────────────────────────────────────────────────────
# 1) Reward / hold-cost function
# ─────────────────────────────────────────────────────────────────────
def hold_cost(s_curr, s_next):
    """Asymmetric per-bar cost of holding from s_curr → s_next.
    Returned as POSITIVE; the FQI step uses reward = -hold_cost."""
    c = 0.005    # base time cost
    R = s_curr['current_R']

    # Loss zone
    if R < 0:
        c += 0.02
        if R < -2.0:
            c += 0.10        # deep-loser aggressive penalty

        # Worsening: if next bar gets worse, extra cost
        dR = s_next['current_R'] - R
        if dR < 0:
            c += 0.05 * abs(dR)

    # Giveback from peak (winner protection cost)
    if s_curr['peak_R'] >= 1.0:
        gb = max(0.0, s_curr['peak_R'] - R)
        c += 0.03 * gb

    return c

# ─────────────────────────────────────────────────────────────────────
# 2) Build per-bar state sequences for each trade
# ─────────────────────────────────────────────────────────────────────
def load_market(swing_csv, setups_glob):
    sw=pd.read_csv(swing_csv,parse_dates=["time"])
    sw=sw.sort_values("time").drop_duplicates('time',keep='last').reset_index(drop=True)
    n=len(sw); C=sw["close"].values.astype(np.float64)
    H=sw["high"].values; Lo=sw["low"].values
    tr=np.concatenate([[H[0]-Lo[0]],np.maximum.reduce([H[1:]-Lo[1:],np.abs(H[1:]-C[:-1]),np.abs(Lo[1:]-C[:-1])])])
    atr=pd.Series(tr).rolling(14,min_periods=14).mean().values
    t2i={pd.Timestamp(t):i for i,t in enumerate(sw["time"].values)}
    setups=[pd.read_csv(f,parse_dates=["time"]) for f in sorted(_glob.glob(setups_glob))]
    all_df=pd.concat(setups,ignore_index=True)
    phys=all_df[['time']+V72L].drop_duplicates('time',keep='last').sort_values('time')
    sw=pd.merge_asof(sw.sort_values('time'),phys,on='time',direction='nearest')
    for c in V72L: sw[c]=sw[c].fillna(0)
    ctx=sw[V72L].fillna(0).values.astype(np.float64)
    return n,C,atr,t2i,ctx

def build_trade_state_seq(trade, t2i, n, C, atr, ctx):
    """Walk a trade forward bar-by-bar. Stop at hard SL or MAX_HOLD.
    Returns list of state dicts (one per in-trade bar) + terminal pnl."""
    tm = trade["time"]
    if tm not in t2i: return None, None
    ei = t2i[tm]; d = int(trade["direction"]); ep = C[ei]; ea = atr[ei]
    if not np.isfinite(ea) or ea <= 0: return None, None

    seq = []
    peak = 0.0
    max_k = min(MAX_HOLD, n - ei - 1)
    terminal_R = None
    for k in range(1, max_k+1):
        bar = ei + k
        R = d * (C[bar] - ep) / ea
        if R <= SL_HARD:
            terminal_R = R    # hard SL terminal
            break
        if R > peak: peak = R

        # State features at this bar
        c1 = C[max(0, bar-1)]; c3 = C[max(0, bar-3)]; c10 = C[max(0, bar-10)]
        mom3 = d*(C[bar]-c3)/ea if bar >= 3 else 0.0
        mom10 = d*(C[bar]-c10)/ea if bar >= 10 else 0.0
        vol10 = float(np.std([d*(C[max(0,bar-j)]-C[max(0,bar-j-1)])/ea
                              for j in range(min(10,bar))])) if bar > 5 else 0.0
        v72 = [float(ctx[bar, j]) for j in range(len(V72L))]

        s = {
            'current_R': float(R),
            'peak_R': float(peak),
            'drawdown_from_peak': float(peak - R),
            'bars_in_trade': float(k),
            'bars_remaining': float(MAX_HOLD - k),
            'mom_3bar': float(mom3),
            'mom_10bar': float(mom10),
            'vol_10bar': vol10,
            'dist_to_SL': float(R - SL_HARD),
            'dist_to_TP': float(2.0 - R),
        }
        for fname, fv in zip(V72L, v72): s[fname] = fv
        s['_bar'] = bar; s['_d'] = d; s['_ep'] = ep; s['_ea'] = ea
        s['_features'] = np.asarray([s[f] for f in STATE_FEATS], dtype=np.float32)
        seq.append(s)

    if terminal_R is None and seq:
        # Closed at max-hold
        last_bar = seq[-1]['_bar']
        terminal_R = seq[-1]['current_R']
    return seq, terminal_R

# ─────────────────────────────────────────────────────────────────────
# 3) Backward fitted-Q iteration
# ─────────────────────────────────────────────────────────────────────
def fit_q_hold(trade_seqs, n_iter=N_FQI_ITER, ensemble_last=N_ENSEMBLE, seed=42):
    """Run n_iter rounds of backward FQI. In the final round, train an
    ensemble of `ensemble_last` bootstrapped models for conservative Q_hold.
    Returns the ensemble (list of XGBRegressor)."""
    model = None
    for it in range(n_iter):
        Xs, ys = [], []
        for seq in trade_seqs:
            T = len(seq)
            if T < 1: continue
            V = np.zeros(T+1, dtype=np.float64)
            # Terminal: V(after last bar) = current_R at last bar (we exit there)
            V[T] = seq[T-1]['current_R']

            for k in range(T-1, -1, -1):
                if k == T-1:
                    Q_target = seq[k]['current_R']  # absorbing terminal
                else:
                    c = hold_cost(seq[k], seq[k+1])
                    Q_target = -c + GAMMA * V[k+1]

                # Update V for use at this k (looking back from k-1 next iter)
                if model is None:
                    V[k] = max(seq[k]['current_R'], Q_target)
                else:
                    pred = float(model.predict(seq[k]['_features'].reshape(1,-1))[0])
                    V[k] = max(seq[k]['current_R'], pred)

                Xs.append(seq[k]['_features']); ys.append(Q_target)

        X = np.vstack(Xs); y = np.asarray(ys, dtype=np.float32)
        if it < n_iter - 1:
            model = XGBRegressor(
                n_estimators=400, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=seed,
                objective='reg:squarederror', verbosity=0)
            model.fit(X, y)
        else:
            # Final iter: ensemble of bootstrapped models for conservative Q_hold
            ensemble = []
            rng = np.random.default_rng(seed)
            for k in range(ensemble_last):
                idx = rng.integers(0, len(X), size=len(X))
                m = XGBRegressor(
                    n_estimators=400, max_depth=5, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, random_state=seed+k,
                    objective='reg:squarederror', verbosity=0)
                m.fit(X[idx], y[idx])
                ensemble.append(m)
            return ensemble
    return [model] if model is not None else []

# ─────────────────────────────────────────────────────────────────────
# 4) Auxiliary classifiers (loser-specific)
# ─────────────────────────────────────────────────────────────────────
def build_aux_labels(trade_seqs):
    """For every (state, k) in every trade, compute three binary labels by
    looking ahead in the SAME trade only:
      label_recover     1 if (R<0 at k) AND R reaches >=0 within next 6 bars before hitting -2
      label_breakdown   1 if R reaches current_R - 1 within next 6 bars before reaching current_R + 0.5
      label_sl          1 if R reaches -4 within next 12 bars
    """
    Xs, y_rec, y_brk, y_sl = [], [], [], []
    LOOKAHEAD_REC = 6; LOOKAHEAD_SL = 12
    for seq in trade_seqs:
        T = len(seq)
        for k in range(T):
            R0 = seq[k]['current_R']
            # ── Recovery ──
            if R0 < 0:
                rec = 0
                for j in range(k+1, min(k+LOOKAHEAD_REC+1, T)):
                    if seq[j]['current_R'] <= -2.0: break
                    if seq[j]['current_R'] >= 0.0:  rec = 1; break
            else:
                rec = -1   # not applicable; we'll mask out

            # ── Breakdown ──
            brk = 0
            for j in range(k+1, min(k+LOOKAHEAD_REC+1, T)):
                if seq[j]['current_R'] >= R0 + 0.5: break
                if seq[j]['current_R'] <= R0 - 1.0: brk = 1; break

            # ── SL within 12 bars ──
            sl = 0
            for j in range(k+1, min(k+LOOKAHEAD_SL+1, T)):
                if seq[j]['current_R'] <= SL_HARD: sl = 1; break

            Xs.append(seq[k]['_features']); y_rec.append(rec); y_brk.append(brk); y_sl.append(sl)
    return (np.vstack(Xs),
            np.asarray(y_rec, dtype=np.int32),
            np.asarray(y_brk, dtype=np.int32),
            np.asarray(y_sl,  dtype=np.int32))

def train_classifier(X, y, mask=None):
    if mask is not None:
        X = X[mask]; y = y[mask]
    pos = int(y.sum()); neg = int((y==0).sum())
    if pos < 20 or neg < 20:
        return None
    sw = np.where(y==1, neg/max(pos,1), 1.0).astype(np.float32)
    m = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
                      random_state=42, verbosity=0)
    m.fit(X, y, sample_weight=sw)
    return m

# ─────────────────────────────────────────────────────────────────────
# 5) should_exit policy
# ─────────────────────────────────────────────────────────────────────
def should_exit(state, q_ensemble, recovery_clf, breakdown_clf, sl_clf,
                exit_threshold_q=0.0, recovery_thr=0.30, breakdown_thr=0.50,
                deep_loser_pct=20):
    R = state['current_R']; bars = state['bars_in_trade']; peak = state['peak_R']

    # 1. Hard SL
    if R <= SL_HARD:
        return True, "hard_sl"
    # 2. Max hold
    if bars >= MAX_HOLD:
        return True, "max_hold"

    feats = state['_features'].reshape(1, -1)

    # 3. Loser-zone breakdown
    if R < 0:
        p_rec = float(recovery_clf.predict_proba(feats)[0,1]) if recovery_clf else 0.5
        p_brk = float(breakdown_clf.predict_proba(feats)[0,1]) if breakdown_clf else 0.5
        if p_rec < recovery_thr and p_brk > breakdown_thr:
            return True, f"loser_breakdown(rec={p_rec:.2f},brk={p_brk:.2f})"
        # Aggressive deep-loser
        if R < -2.5 and p_rec < 0.20:
            return True, f"deep_loser(rec={p_rec:.2f})"

    # 4. Winner-giveback
    if peak >= 1.5 and R < peak * 0.4:
        return True, f"winner_giveback(peak={peak:.2f}R,now={R:.2f}R)"

    # 5. Q-stopping condition
    q_exit = R
    q_preds = np.array([m.predict(feats)[0] for m in q_ensemble])
    q_hold = float(np.percentile(q_preds, deep_loser_pct))
    if q_exit >= q_hold + exit_threshold_q:
        return True, f"q_dominates(exit={q_exit:.2f},hold_p25={q_hold:.2f})"

    return False, "hold"

# ─────────────────────────────────────────────────────────────────────
# 6) Simulator
# ─────────────────────────────────────────────────────────────────────
def simulate_trade_with_policy(trade, t2i, n, C, atr, ctx,
                               q_ensemble=None, recovery_clf=None,
                               breakdown_clf=None, sl_clf=None,
                               policy="baseline"):
    """policy in {'baseline', 'smart'}.
    baseline = hard SL + trail (act=3, gb=0.6) + MAX_HOLD bars max.
    smart    = should_exit() function above.
    """
    tm = trade["time"]
    if tm not in t2i: return None, "no_idx"
    ei = t2i[tm]; d = int(trade["direction"]); ep = C[ei]; ea = atr[ei]
    if not np.isfinite(ea) or ea <= 0: return None, "no_atr"

    peak = 0.0
    max_k = min(MAX_HOLD, n - ei - 1)
    for k in range(1, max_k+1):
        bar = ei + k
        R = d * (C[bar] - ep) / ea
        if R <= SL_HARD: return float(R), "hard_sl"
        if R > peak: peak = R

        if policy == "baseline":
            if peak >= 3.0 and R <= peak * 0.4:
                return float(R), "trail"
        elif policy == "smart":
            # Build state
            c3 = C[max(0,bar-3)]; c10 = C[max(0,bar-10)]
            mom3 = d*(C[bar]-c3)/ea if bar>=3 else 0.0
            mom10 = d*(C[bar]-c10)/ea if bar>=10 else 0.0
            vol10 = float(np.std([d*(C[max(0,bar-j)]-C[max(0,bar-j-1)])/ea for j in range(min(10,bar))])) if bar>5 else 0.0
            v72 = [float(ctx[bar,j]) for j in range(len(V72L))]
            s = {'current_R':float(R),'peak_R':float(peak),'drawdown_from_peak':float(peak-R),
                 'bars_in_trade':float(k),'bars_remaining':float(MAX_HOLD-k),
                 'mom_3bar':float(mom3),'mom_10bar':float(mom10),'vol_10bar':vol10,
                 'dist_to_SL':float(R-SL_HARD),'dist_to_TP':float(2.0-R)}
            for fname,fv in zip(V72L,v72): s[fname]=fv
            s['_features']=np.asarray([s[f] for f in STATE_FEATS],dtype=np.float32)
            ex, why = should_exit(s, q_ensemble, recovery_clf, breakdown_clf, sl_clf)
            if ex: return float(R), why
    # Reached max-hold without explicit exit
    last_bar = ei + max_k
    R = d*(C[last_bar]-ep)/ea
    return float(R), "max_hold"

def pf(p):
    s=sum(p); w=[x for x in p if x>0]; l=[x for x in p if x<=0]
    pf=sum(w)/max(-sum(l),1e-9) if l else 99.0
    return pf,s,len(w)/max(len(p),1)

def maxdd(p):
    eq=np.cumsum(p); peak=np.maximum.accumulate(eq)
    return float((peak-eq).max())

# ─────────────────────────────────────────────────────────────────────
# 7) Walk-forward evaluation
# ─────────────────────────────────────────────────────────────────────
def evaluate(name, swing_csv, setups_glob, trades_csv):
    print("\n"+"="*72); print(f"  {name}"); print("="*72)
    t0 = time.time()
    n,C,atr,t2i,ctx = load_market(swing_csv, setups_glob)
    trades = pd.read_csv(trades_csv, parse_dates=["time"]).sort_values("time").reset_index(drop=True)
    split = trades["time"].quantile(0.70)
    trn = trades[trades["time"] < split].reset_index(drop=True)
    tst = trades[trades["time"] >= split].reset_index(drop=True)
    print(f"  Train: {len(trn)} trades  | Test (unseen): {len(tst)}")

    print("  Building state sequences (TRAIN)...", end='', flush=True); t=time.time()
    train_seqs = []
    for _, tr in trn.iterrows():
        seq, _ = build_trade_state_seq(tr, t2i, n, C, atr, ctx)
        if seq: train_seqs.append(seq)
    print(f" {time.time()-t:.0f}s — {len(train_seqs)} seqs, {sum(len(s) for s in train_seqs):,} bars", flush=True)

    print("  Backward FQI (Q_hold ensemble)...", end='', flush=True); t=time.time()
    q_ensemble = fit_q_hold(train_seqs, n_iter=N_FQI_ITER, ensemble_last=N_ENSEMBLE)
    print(f" {time.time()-t:.0f}s — {len(q_ensemble)} bootstrapped models", flush=True)

    print("  Auxiliary classifiers (recovery / breakdown / SL)...", end='', flush=True); t=time.time()
    Xa, y_rec, y_brk, y_sl = build_aux_labels(train_seqs)
    rec_clf = train_classifier(Xa, y_rec, mask=(y_rec >= 0))   # only loser bars
    brk_clf = train_classifier(Xa, y_brk)
    sl_clf  = train_classifier(Xa, y_sl)
    print(f" {time.time()-t:.0f}s — sizes rec={int((y_rec>=0).sum())} brk={len(y_brk)} sl={len(y_sl)}", flush=True)

    # ── Quick sanity: AUC on train (cheap)
    if rec_clf:
        m = (y_rec >= 0)
        auc_rec = roc_auc_score(y_rec[m], rec_clf.predict_proba(Xa[m])[:,1])
    else: auc_rec = float('nan')
    auc_brk = roc_auc_score(y_brk, brk_clf.predict_proba(Xa)[:,1]) if brk_clf else float('nan')
    auc_sl  = roc_auc_score(y_sl,  sl_clf.predict_proba(Xa)[:,1])  if sl_clf  else float('nan')
    print(f"  Train AUCs: recovery={auc_rec:.3f}  breakdown={auc_brk:.3f}  sl={auc_sl:.3f}")

    # ── Test simulation: baseline vs smart-exit
    print("\n  Simulating TEST trades...")
    base_pnls, smart_pnls = [], []
    smart_reasons = {}
    for _, tr in tst.iterrows():
        rb, why_b = simulate_trade_with_policy(tr, t2i, n, C, atr, ctx, policy="baseline")
        rs, why_s = simulate_trade_with_policy(tr, t2i, n, C, atr, ctx,
                                               q_ensemble=q_ensemble,
                                               recovery_clf=rec_clf,
                                               breakdown_clf=brk_clf,
                                               sl_clf=sl_clf,
                                               policy="smart")
        if rb is not None: base_pnls.append(rb)
        if rs is not None:
            smart_pnls.append(rs)
            tag = why_s.split('(')[0]
            smart_reasons[tag] = smart_reasons.get(tag, 0) + 1

    pf_b, s_b, wr_b = pf(base_pnls);   dd_b = maxdd(base_pnls)
    pf_s, s_s, wr_s = pf(smart_pnls);  dd_s = maxdd(smart_pnls)

    print(f"\n  BASELINE (hard SL + trail act=3/gb=0.6 + max={MAX_HOLD}):")
    print(f"    PF={pf_b:.2f}  Total={s_b:+.0f}R  WR={wr_b*100:.1f}%  DD={dd_b:.0f}R  N={len(base_pnls)}")
    print(f"\n  SMART EXIT (Q_hold p25 + aux classifiers):")
    print(f"    PF={pf_s:.2f}  Total={s_s:+.0f}R  WR={wr_s*100:.1f}%  DD={dd_s:.0f}R  N={len(smart_pnls)}")
    delta = s_s - s_b
    print(f"\n  Δ Total R: {delta:+.0f}R   Δ PF: {pf_s-pf_b:+.2f}   Δ DD: {dd_s-dd_b:+.0f}R")

    print(f"\n  Smart-exit reason breakdown:")
    for k,v in sorted(smart_reasons.items(), key=lambda x:-x[1]):
        print(f"    {k:>30s}: {v:4d}  ({v*100/len(smart_pnls):.1f}%)")

    print(f"\n  Done in {time.time()-t0:.0f}s")
    return {'pf_base':pf_b,'pf_smart':pf_s,'delta':delta,'dd_b':dd_b,'dd_s':dd_s}

if __name__ == "__main__":
    evaluate("Oracle XAU — v89 smart exit",
             f"{DATA}/swing_v5_xauusd.csv",
             f"{DATA}/setups_*_v72l.csv",
             f"{PROJECT}/experiments/v84_rl_entry/v84_rl_trades.csv")
    evaluate("Oracle BTC — v89 smart exit",
             f"{DATA}/swing_v5_btc.csv",
             f"{DATA}/setups_*_v72l_btc.csv",
             f"{PROJECT}/experiments/v84_rl_entry/btc_rl_trades.csv")
