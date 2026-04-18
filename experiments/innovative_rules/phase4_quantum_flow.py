"""
Phase 4: Add Quantum Volume-Flow Quantizer features to confirmation classifier.

Converts the Pine Script indicator to Python:
  - quantum_flow:      M5 smoothed HA-trend × volume-pressure, ATR-quantized
  - quantum_flow_h4:   same calc on H4, mapped to M5
  - quantum_momentum:  derivative of quantum_flow (acceleration)
  - quantum_vwap_conf: quantum_flow sign agrees with VWAP direction (confluence)

Test configs on big holdout:
  A) Current 36 features (baseline)
  D) Pure physics + VWAP (8 features) — previous winner
  E) Physics + VWAP + Quantum Flow (12 features)
  F) ALL: 36 + physics + VWAP + Quantum (48 features)
"""
from __future__ import annotations
import glob, json, os, time as _time
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/jay/Desktop/new-model-zigzag"
TP_MULT, SL_MULT, MAX_FWD = 6.0, 2.0, 40

# ═══════ Reuse physics from phase3 ═══════
from phase3_physics_only import (compute_all_features, forward_outcome,
                                  PHYSICS_COLS, CURRENT_36, BASE_F01_F20)

QUANTUM_COLS = ["quantum_flow", "quantum_flow_h4", "quantum_momentum", "quantum_vwap_conf"]


def compute_quantum_flow_series(o, h, l, c, vol, lookback=21, vol_lookback=50):
    """Replicate the Pine Script Quantum Volume-Flow Quantizer."""
    n = len(c)

    # Heikin-Ashi synthetic
    ha_close = (o + h + l + c) / 4.0
    ha_open = np.empty(n)
    ha_open[0] = (o[0] + c[0]) / 2.0
    for i in range(1, n):
        ha_open[i] = (o[i-1] + c[i-1]) / 2.0

    trend_force = ha_close - ha_open

    # Volume pressure
    avg_vol = pd.Series(vol).rolling(vol_lookback, min_periods=1).mean().values
    vol_factor = np.where(avg_vol > 1e-10, vol / avg_vol, 0.0)

    # Raw energy
    raw_energy = trend_force * vol_factor * 1000.0

    # EMA smoothing
    smooth = pd.Series(raw_energy).ewm(span=lookback, adjust=False).mean().values

    # ATR quantization
    tr = np.empty(n); tr[0] = h[0] - l[0]
    tr[1:] = np.maximum.reduce([h[1:]-l[1:], np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])])
    atr14 = pd.Series(tr).rolling(14, min_periods=1).mean().values
    step = atr14 * 0.5
    quantized = np.where(step > 1e-10, np.round(smooth / step) * step, smooth)

    return quantized


def compute_quantum_features(swing_df):
    """Compute Quantum Flow on M5 and H4, plus derived features."""
    o = swing_df["open"].values.astype(np.float64)
    h = swing_df["high"].values.astype(np.float64)
    l = swing_df["low"].values.astype(np.float64)
    c = swing_df["close"].values.astype(np.float64)
    vol = swing_df["spread"].values.astype(np.float64)  # tick_volume proxy
    vol = np.maximum(vol, 1.0)
    n = len(c)

    # M5 quantum flow
    qf_m5 = compute_quantum_flow_series(o, h, l, c, vol)

    # H4 quantum flow
    df_h4 = swing_df.set_index("time")[["open","high","low","close","spread"]].resample("4h").agg(
        {"open":"first","high":"max","low":"min","close":"last","spread":"sum"}).dropna()
    qf_h4_raw = compute_quantum_flow_series(
        df_h4["open"].values, df_h4["high"].values, df_h4["low"].values,
        df_h4["close"].values, np.maximum(df_h4["spread"].values, 1.0))
    # Shift H4 by 1 bar to avoid look-ahead, then map to M5
    qf_h4_series = pd.Series(qf_h4_raw, index=df_h4.index).shift(1)
    qf_h4_m5 = qf_h4_series.reindex(swing_df["time"], method="ffill").values

    # Momentum (1st derivative of quantum flow)
    qf_mom = np.concatenate([[0.0], np.diff(qf_m5)])

    # VWAP confluence: does quantum flow sign agree with VWAP direction?
    # Compute session VWAP
    typical = (h + l + c) / 3.0
    dates = swing_df["time"].dt.date.values
    vwap = np.zeros(n)
    cum_tv = 0.0; cum_v = 0.0; prev_date = dates[0]
    for i in range(n):
        if dates[i] != prev_date:
            cum_tv = 0.0; cum_v = 0.0; prev_date = dates[i]
        cum_tv += typical[i] * vol[i]; cum_v += vol[i]
        vwap[i] = cum_tv / cum_v if cum_v > 0 else c[i]
    vwap_dir = np.sign(c - vwap)
    qf_sign = np.sign(qf_m5)
    # confluence = 1 if both agree, -1 if disagree, 0 if either neutral
    confluence = qf_sign * vwap_dir

    return {
        "quantum_flow": qf_m5,
        "quantum_flow_h4": qf_h4_m5,
        "quantum_momentum": qf_mom,
        "quantum_vwap_conf": confluence,
    }


def run_config(all_df, feat_cols, config_name, time_to_idx, H, L, C, atr):
    """Train per-rule XGB on feat_cols, return aggregated holdout results."""
    combined = {"pnls": [], "wins": 0, "losses": 0}
    rules_done = 0
    for (cid, rule), grp in all_df.groupby(["cid","rule"]):
        meta_path = f"{ROOT}/models/confirm_c{cid}_{rule}_meta.json"
        if not os.path.exists(meta_path): continue
        meta = json.load(open(meta_path))
        if meta.get("disabled", False): continue
        thr = meta["threshold"]
        rdf = grp.sort_values("time").reset_index(drop=True)
        split = int(len(rdf)*0.8)
        train, test = rdf.iloc[:split], rdf.iloc[split:]
        if len(train)<50 or len(test)<20: continue
        y_tr, y_te = train["label"].values, test["label"].values
        miss = [c for c in feat_cols if c not in train.columns]
        if miss: continue
        X_tr = train[feat_cols].fillna(0).values
        X_te = test[feat_cols].fillna(0).values
        mdl = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8,
                            eval_metric="logloss", verbosity=0)
        mdl.fit(X_tr, y_tr)
        proba = mdl.predict_proba(X_te)[:,1]
        mask = proba >= thr
        if mask.sum() == 0: continue
        for s in y_te[mask]:
            combined["pnls"].append(TP_MULT if s==1 else -SL_MULT)
            if s==1: combined["wins"]+=1
            else: combined["losses"]+=1
        rules_done += 1
    pnls = np.array(combined["pnls"])
    if len(pnls)==0: return None
    n=len(pnls); w=combined["wins"]; lo=combined["losses"]
    wr=w/n; pf=(w*TP_MULT)/(lo*SL_MULT+1e-9)
    eq=np.cumsum(pnls); dd=(eq-np.maximum.accumulate(eq)).min()
    print(f"  {config_name:40s} n={n:3d}  WR={wr:.1%}  PF={pf:.3f}  PnL={pnls.sum():+.1f}R  DD={dd:.1f}R  rules={rules_done}", flush=True)
    return {"name": config_name, "pnls": pnls, "n":n, "wr":wr, "pf":pf, "total":pnls.sum(), "dd":dd}


def main():
    print("="*70)
    print("PHASE 4: Quantum Flow + Physics — big holdout comparison")
    print("="*70, flush=True)

    # Physics + VWAP + time features (from phase3)
    swing, atr_arr = compute_all_features(f"{ROOT}/data/swing_v5_xauusd.csv")
    H, L, C = swing["high"].values, swing["low"].values, swing["close"].values
    time_to_idx = pd.Series(swing.index.values, index=swing["time"].values)

    # Quantum Flow features
    print("Computing Quantum Flow...", flush=True)
    t0 = _time.time()
    qf = compute_quantum_features(swing)
    for col, vals in qf.items():
        swing[col] = vals
    print(f"  done in {_time.time()-t0:.1f}s", flush=True)

    # Load setups
    all_setups = []
    for f in sorted(glob.glob(f"{ROOT}/data/setups_*.csv")):
        cid = int(os.path.basename(f).replace("setups_","").replace(".csv",""))
        df = pd.read_csv(f, parse_dates=["time"]); df["cid"] = cid
        all_setups.append(df)
    all_df = pd.concat(all_setups, ignore_index=True).sort_values("time").reset_index(drop=True)

    # Merge ALL new features from swing
    merge_cols = PHYSICS_COLS + ["vwap_dist","hour_enc","dow_enc"] + QUANTUM_COLS
    swing_feats = swing[["time"] + merge_cols].copy()
    all_df = all_df.merge(swing_feats, on="time", how="left", suffixes=("","_new"))
    for col in ["hour_enc","dow_enc"]:
        if col+"_new" in all_df.columns:
            all_df[col] = all_df[col+"_new"].fillna(all_df[col])
            all_df.drop(columns=[col+"_new"], inplace=True)
    for col in PHYSICS_COLS + ["vwap_dist"] + QUANTUM_COLS:
        all_df[col] = all_df[col].fillna(0)

    # Relabel at 6:2
    print("Relabeling...", flush=True)
    labels = []
    for _, row in all_df.iterrows():
        t = row["time"]
        if t not in time_to_idx.index: labels.append(-1); continue
        idx = int(time_to_idx[t]); d = int(row["direction"])
        labels.append(forward_outcome(H, L, C, atr_arr, idx, d))
    all_df["label"] = labels
    all_df = all_df[all_df["label"].isin([0,1])].reset_index(drop=True)
    print(f"Valid: {len(all_df):,}\n", flush=True)

    # Feature configs
    PHYS_VWAP = PHYSICS_COLS + ["vwap_dist", "hour_enc", "dow_enc"]
    configs = [
        ("A) Current 36",                      CURRENT_36),
        ("D) Pure physics+VWAP (8)",            PHYS_VWAP),
        ("E) Physics+VWAP+Quantum (12)",        PHYS_VWAP + QUANTUM_COLS),
        ("F) f01-f20+Physics+VWAP+Quantum (32)",BASE_F01_F20 + PHYS_VWAP + QUANTUM_COLS),
        ("G) ALL 48",                           CURRENT_36 + PHYSICS_COLS + ["vwap_dist"] + QUANTUM_COLS),
    ]

    results = []
    for name, feats in configs:
        r = run_config(all_df, feats, name, time_to_idx, H, L, C, atr_arr)
        if r: results.append(r)

    # Equity curves
    fig, ax = plt.subplots(figsize=(15, 6), facecolor="#0b0e14")
    ax.set_facecolor("#0d1117")
    palette = ["#3b82f6", "#888", "#f5c518", "#10b981", "#a855f7"]
    for i, r in enumerate(results):
        eq = np.cumsum(r["pnls"])
        ax.plot(eq, color=palette[i % len(palette)], linewidth=1.5,
                label=f"{r['name']} PF={r['pf']:.2f}")
    ax.axhline(0, color="#444", linewidth=0.5)
    ax.legend(facecolor="#111", edgecolor="#333", fontsize=9, loc="upper left")
    ax.set_title("XAU — Feature Set Comparison with Quantum Flow — Full Holdout",
                 color="#ffd700", fontsize=13)
    ax.set_xlabel("trade #", color="#888"); ax.set_ylabel("cumulative R", color="#888")
    ax.tick_params(colors="#888")
    for sp in ax.spines.values(): sp.set_edgecolor("#1a2332")
    plt.tight_layout()
    out = f"{ROOT}/experiments/innovative_rules/phase4_quantum_comparison.png"
    plt.savefig(out, dpi=140, facecolor="#0b0e14")
    print(f"\nSaved: {out}", flush=True)


if __name__ == "__main__":
    main()
