"""
v8.4 — Validate v8.3 strategy implemented CORRECTLY.

What we're testing (per product, per pivot threshold):
  ENTRY:  existing rule cascade + meta gate + cohort kill (unchanged)
          PLUS: pivot pre-filter (skip if pivot_score < threshold)
  EXIT:   existing ML exit + hard SL + max hold (unchanged)
          PLUS: 0.50R guard (close immediately when PnL >= 0.50R)

For each holdout trade we:
  1. Look up pivot features at fire time (v72L + 14 Janus extras)
  2. Score with Janus pivot model (XAU products)
  3. If pivot_score < threshold → DROP the trade
  4. If kept → walk bar-by-bar from entry to original exit, find first bar
     where PnL_R >= 0.50; if it exists, replace exit with the guard
     (pnl_R = 0.50, exit_reason = "guard")

Sweep thresholds: 0.20, 0.25, 0.30, 0.35, 0.40
Report per (product, threshold): n / WR / PF / R / DD / R_per_trade

NOTE: BTC pivot v2 has a label-leak issue (label was in training feature
list). NOT validating BTC here — needs retrain first. Sticking to XAU.
"""
from __future__ import annotations
import os, sys, pickle
import numpy as np
import pandas as pd

ROOT = "/home/jay/Desktop/new-model-zigzag"
JANUS_PKL = "/home/jay/Desktop/my-agents-and-website/commercial/server/decision_engine/models/janus_xau_validated.pkl"

GUARD_R = 0.50           # exit when pnl_R >= this
SL_HARD = 4.0            # mirror config: hard SL at 4×ATR (already in trade lifetime)


def pf(rs):
    pos = rs[rs > 0].sum(); neg = -rs[rs <= 0].sum()
    return pos / max(neg, 1e-9)


def max_dd(rs):
    eq = np.cumsum(rs); peak = np.maximum.accumulate(eq)
    return (eq - peak).min() if len(eq) else 0.0


def load_swing_with_features(swing_path):
    """Load swing + compute all features needed for Janus pivot scoring +
    the bar-level close trajectory needed to simulate guard exit."""
    sys.path.insert(0, os.path.join(ROOT, "model_pipeline"))
    sys.path.insert(0, os.path.join(ROOT, "experiments/v72_lite_deploy"))
    # Borrow the production feature-compute path
    from importlib.machinery import SourceFileLoader
    v72_mod = SourceFileLoader(
        "v72v", os.path.join(ROOT, "experiments/v72_lite_deploy/01_validate_v72_lite.py")
    ).load_module()
    # load_swing_with_physics returns (swing_df_with_physics, atr_array)
    # but it uses paths.py P.data() — we want a specific path. Patch by
    # temporarily overriding sys.path and calling the function with a
    # custom swing path. Simpler: replicate just what we need.
    print(f"  loading {swing_path}…")
    swing = pd.read_csv(swing_path, parse_dates=["time"])
    swing = swing.sort_values("time").reset_index(drop=True)
    H = swing["high"].values.astype(np.float64)
    L = swing["low"].values.astype(np.float64)
    C = swing["close"].values.astype(np.float64)
    O = swing["open"].values.astype(np.float64)
    vol = np.maximum(swing["spread"].values.astype(np.float64), 1.0)
    tr = np.concatenate([[H[0]-L[0]],
          np.maximum.reduce([H[1:]-L[1:], np.abs(H[1:]-C[:-1]),
                              np.abs(L[1:]-C[:-1])])])
    atr = pd.Series(tr).rolling(14, min_periods=14).mean().values
    ret = np.concatenate([[0.0], np.diff(np.log(C))])
    physics = SourceFileLoader(
        "p04b", os.path.join(ROOT, "model_pipeline/04b_compute_physics_features.py")
    ).load_module()
    swing["hurst_rs"]     = physics.compute_hurst_rs(ret)
    swing["ou_theta"]     = physics.compute_ou_theta(ret)
    swing["entropy_rate"] = physics.compute_entropy(ret)
    swing["kramers_up"]   = physics.compute_kramers_up(C)
    swing["wavelet_er"]   = physics.compute_wavelet_er(C)
    swing["vwap_dist"]    = physics.compute_vwap_dist(swing, atr)
    swing["quantum_flow"] = physics.compute_quantum_flow(O, H, L, C, vol)
    df_h4 = swing.set_index("time")[["open","high","low","close","spread"]]\
                  .resample("4h").agg({"open":"first","high":"max","low":"min",
                                         "close":"last","spread":"sum"}).dropna()
    qf_h4 = physics.compute_quantum_flow(df_h4["open"].values, df_h4["high"].values,
                                          df_h4["low"].values, df_h4["close"].values,
                                          np.maximum(df_h4["spread"].values, 1.0))
    qf_h4_s = pd.Series(qf_h4, index=df_h4.index).shift(1)
    swing["quantum_flow_h4"] = qf_h4_s.reindex(swing.set_index("time").index,
                                                 method="ffill").values
    swing["atr"] = atr
    return swing


def load_v72l_features_per_trade(trades, swing):
    """Merge v72L + Janus features onto each trade. Uses setups files
    (which already have v72L extras) for fast lookup, then computes Janus
    features from swing."""
    import glob
    # Setups files have the v72L extras (vpin, sig_quad_var, har_rv_ratio, hawkes_eta)
    # plus quantum_momentum, quantum_vwap_conf, etc. These are pre-computed.
    pattern = "setups_*_v72l.csv"
    cluster_files = sorted(glob.glob(os.path.join(ROOT, "data", pattern)))
    if not cluster_files:
        raise FileNotFoundError(f"No {pattern} files found in data/")
    setups = []
    for f in cluster_files:
        cid = int(os.path.basename(f).split("_")[1])
        df = pd.read_csv(f, parse_dates=["time"])
        df["cid"] = cid
        setups.append(df)
    setups = pd.concat(setups, ignore_index=True).sort_values("time").reset_index(drop=True)
    # The setups files have all v72L extras + base candle + h1/h4 features
    # Drop dups (same trade fired multiple rules at same time) — take first
    setups_unique = setups.drop_duplicates(subset=["time"], keep="first")

    # Compute Janus features on swing
    # Use the production module
    sys.path.insert(0, "/home/jay/Desktop/my-agents-and-website")
    from commercial.server.decision_engine import features_janus
    # The compute_janus_features expects v72L feats already on the df.
    # Merge setups extras onto swing by time first.
    extras = ["vpin", "sig_quad_var", "har_rv_ratio", "hawkes_eta",
                "quantum_momentum", "quantum_vwap_conf", "quantum_divergence",
                "quantum_div_strength", "hour_enc", "dow_enc"]
    extras_present = [c for c in extras if c in setups_unique.columns]
    print(f"  v72L extras from setups: {len(extras_present)}/{len(extras)} found")
    swing_full = swing.merge(setups_unique[["time"] + extras_present],
                              on="time", how="left")
    # Forward-fill the extras (they're computed on bar close — should not
    # leave gaps within trading hours)
    swing_full[extras_present] = swing_full[extras_present].ffill()

    # Now compute Janus features
    swing_full = features_janus.compute_janus_features(swing_full)

    # Merge per-trade features (lookup at fire time)
    out = trades.sort_values("time").reset_index(drop=True).copy()
    pivot_cols = pickle.load(open(JANUS_PKL, "rb"))["pivot_feats"]
    swing_full_subset = swing_full[["time"] + pivot_cols + ["close", "atr"]]
    merged = pd.merge_asof(out, swing_full_subset, on="time",
                            direction="backward",
                            tolerance=pd.Timedelta("10min"))
    merged = merged.dropna(subset=pivot_cols + ["atr"]).reset_index(drop=True)
    return merged, swing_full, pivot_cols


def simulate_guard_per_trade(trade, swing):
    """Walk bar-by-bar from trade entry. If pnl_R hits +GUARD_R first,
    return (GUARD_R, 'guard'). Otherwise return (original_pnl_R, original_exit).
    Hard SL handled by original simulator already — we don't alter losers."""
    entry_time = trade["time"]
    direction = int(trade["direction"])
    bars_held = int(trade["bars"])
    orig_pnl_R = float(trade["pnl_R"])
    orig_exit = trade["exit"]

    # Find entry index in swing
    idx = swing.index[swing["time"] == entry_time]
    if len(idx) == 0:
        return orig_pnl_R, orig_exit  # can't simulate, keep original
    ei = int(idx[0])
    ep = float(swing["close"].iat[ei])
    ea = float(swing["atr"].iat[ei])
    if not np.isfinite(ea) or ea <= 0:
        return orig_pnl_R, orig_exit

    # Walk bars from ei+1 to ei+bars_held (the original holding window)
    # If pnl_R >= GUARD_R at any bar BEFORE original exit, guard fires first.
    for k in range(1, bars_held + 1):
        bar = ei + k
        if bar >= len(swing): break
        cp = direction * (float(swing["close"].iat[bar]) - ep) / ea
        if cp >= GUARD_R:
            return GUARD_R, "guard"
        # If hard SL would have hit, the original exit was hard_sl —
        # leave it alone (guard is a profit-taking rule, not a loss-cutting rule)
    return orig_pnl_R, orig_exit


def evaluate(name, trades, pivot_mdl, pivot_cols, swing_full, thresholds):
    print(f"\n=== {name} ===")
    print(f"  trades after feature merge: {len(trades)}")

    rs0 = trades["pnl_R"].values
    base = {
        "n": len(rs0), "wr": (rs0 > 0).mean(), "pf": pf(rs0),
        "r":  rs0.sum(), "dd": max_dd(rs0),
    }
    print(f"  baseline (no pivot, no guard): n={base['n']} "
          f"WR={base['wr']:.1%} PF={base['pf']:.2f} R={base['r']:+.0f} DD={base['dd']:+.0f}")

    # Score every trade with the pivot model (one-time)
    Xp = trades[pivot_cols].fillna(0).values.astype(np.float32)
    p_pivot = pivot_mdl.predict_proba(Xp)[:, 1]
    print(f"  pivot scores: min={p_pivot.min():.3f} med={np.median(p_pivot):.3f} "
          f"p25={np.percentile(p_pivot,25):.3f} p75={np.percentile(p_pivot,75):.3f} max={p_pivot.max():.3f}")

    # Pre-simulate guard for ALL trades (cache, threshold doesn't change this)
    print("  simulating guard exit per trade…")
    new_pnl_R = np.empty(len(trades))
    new_exit = np.empty(len(trades), dtype=object)
    for i in range(len(trades)):
        pnl, ex = simulate_guard_per_trade(trades.iloc[i], swing_full)
        new_pnl_R[i] = pnl; new_exit[i] = ex
    n_guard = int((new_exit == "guard").sum())
    print(f"  guard fired on {n_guard} trades ({n_guard/len(trades):.1%})")

    rows = []
    for thr in thresholds:
        keep = p_pivot >= thr
        if keep.sum() < 30: continue
        rs = new_pnl_R[keep]
        rows.append({
            "thr":      thr,
            "n":        int(keep.sum()),
            "kept_pct": round(keep.sum() / base["n"], 3),
            "WR":       round((rs > 0).mean(), 3),
            "PF":       round(pf(rs), 2),
            "R":        round(rs.sum(), 1),
            "DD":       round(max_dd(rs), 1),
            "R/trade":  round(rs.sum() / len(rs), 3),
            "ΔWR_pp":   round(((rs > 0).mean() - base["wr"]) * 100, 1),
            "ΔPF":      round(pf(rs) - base["pf"], 2),
            "ΔR":       round(rs.sum() - base["r"], 1),
        })
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    return out


def main():
    print("Loading Janus pivot model…")
    janus = pickle.load(open(JANUS_PKL, "rb"))
    pivot_mdl = janus["pivot_score_mdl"]
    print(f"  expects {len(janus['pivot_feats'])} features")

    print("\nLoading XAU swing + computing physics features…")
    swing = load_swing_with_features(os.path.join(ROOT, "data/swing_v5_xauusd.csv"))

    print("\n--- Oracle XAU ---")
    oracle = pd.read_csv(os.path.join(ROOT, "data/v72l_trades_holdout.csv"),
                          parse_dates=["time"])
    oracle, swing_full, pivot_cols = load_v72l_features_per_trade(oracle, swing)
    out_o = evaluate("Oracle XAU", oracle, pivot_mdl, pivot_cols, swing_full,
                      [0.20, 0.25, 0.30, 0.35, 0.40])
    out_o.to_csv(os.path.join(os.path.dirname(__file__),
                                "sweep_oracle_xau.csv"), index=False)

    print("\n--- Midas XAU ---")
    midas = pd.read_csv(os.path.join(ROOT, "data/v6_trades_holdout_xau.csv"),
                          parse_dates=["time"])
    midas, _, _ = load_v72l_features_per_trade(midas, swing)
    out_m = evaluate("Midas XAU", midas, pivot_mdl, pivot_cols, swing_full,
                      [0.20, 0.25, 0.30, 0.35, 0.40])
    out_m.to_csv(os.path.join(os.path.dirname(__file__),
                                "sweep_midas_xau.csv"), index=False)

    print("\n" + "="*72)
    print("BTC: SKIPPED — btc_pivot_v2.pkl has 'label' in feature list (training")
    print("leak). Need to retrain BTC pivot before adding it to v8.4.")


if __name__ == "__main__":
    main()
