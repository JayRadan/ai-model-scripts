"""
Phase 4b: Quantum Flow MTF divergence pattern.

Pattern (from user's insight):
  BUY setup:  H4 quantum > 0 (bullish bias) AND M5 quantum < 0 (temporary dip)
              → existing rules decide exact entry timing within this window
  SELL setup: H4 quantum < 0 (bearish bias) AND M5 quantum > 0 (temporary rally)
              → existing rules decide exact entry timing

This is used as a FILTER on existing rules (only take trades that align),
AND as a standalone feature (quantum_divergence) for the classifier.

Test:
  E)  Physics+VWAP+Quantum 12 features (previous best)
  E2) Same 12 + divergence filter (only take trades when pattern active)
  E3) Same 12 + divergence as additional feature (let classifier decide)
  E4) Physics+VWAP+Quantum+divergence feature (13 features, no classic)
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

from phase3_physics_only import (compute_all_features, forward_outcome,
                                  PHYSICS_COLS, CURRENT_36, BASE_F01_F20)
from phase4_quantum_flow import compute_quantum_features, QUANTUM_COLS


def main():
    print("="*70)
    print("PHASE 4b: Quantum Flow MTF Divergence Pattern")
    print("="*70, flush=True)

    swing, atr_arr = compute_all_features(f"{ROOT}/data/swing_v5_xauusd.csv")
    H, L, C = swing["high"].values, swing["low"].values, swing["close"].values
    time_to_idx = pd.Series(swing.index.values, index=swing["time"].values)

    print("Computing Quantum Flow...", flush=True)
    qf = compute_quantum_features(swing)
    for col, vals in qf.items():
        swing[col] = vals

    # Divergence feature:
    #   +1 = H4 > 0 AND M5 < 0 (bullish divergence → buy opportunity)
    #   -1 = H4 < 0 AND M5 > 0 (bearish divergence → sell opportunity)
    #    0 = aligned (no divergence)
    qf_m5 = swing["quantum_flow"].values
    qf_h4 = swing["quantum_flow_h4"].values
    divergence = np.zeros(len(qf_m5))
    divergence[(qf_h4 > 0) & (qf_m5 < 0)] = +1.0  # bullish divergence
    divergence[(qf_h4 < 0) & (qf_m5 > 0)] = -1.0  # bearish divergence
    swing["quantum_divergence"] = divergence

    # Also: divergence strength = |H4| when divergence active, 0 otherwise
    swing["quantum_div_strength"] = np.where(
        divergence != 0,
        np.abs(qf_h4),
        0.0
    )

    # Load setups
    all_setups = []
    for f in sorted(glob.glob(f"{ROOT}/data/setups_*.csv")):
        cid = int(os.path.basename(f).replace("setups_","").replace(".csv",""))
        df = pd.read_csv(f, parse_dates=["time"]); df["cid"] = cid
        all_setups.append(df)
    all_df = pd.concat(all_setups, ignore_index=True).sort_values("time").reset_index(drop=True)

    # Merge features
    merge_cols = PHYSICS_COLS + ["vwap_dist","hour_enc","dow_enc"] + QUANTUM_COLS + ["quantum_divergence","quantum_div_strength"]
    swing_feats = swing[["time"] + merge_cols].copy()
    all_df = all_df.merge(swing_feats, on="time", how="left", suffixes=("","_new"))
    for col in ["hour_enc","dow_enc"]:
        if col+"_new" in all_df.columns:
            all_df[col] = all_df[col+"_new"].fillna(all_df[col])
            all_df.drop(columns=[col+"_new"], inplace=True)
    for col in merge_cols:
        if col in ["hour_enc","dow_enc"]: continue
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

    # Feature sets
    PHYS_VWAP = PHYSICS_COLS + ["vwap_dist","hour_enc","dow_enc"]
    E_12 = PHYS_VWAP + QUANTUM_COLS
    E3_14 = E_12 + ["quantum_divergence", "quantum_div_strength"]

    def run_config(df_in, feat_cols, name, use_div_filter=False):
        combined = {"pnls": [], "wins": 0, "losses": 0}
        rules_done = 0
        for (cid, rule), grp in df_in.groupby(["cid","rule"]):
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

            if use_div_filter:
                # Only take trades where divergence aligns with direction
                dirs = test["direction"].values
                divs = test["quantum_divergence"].values
                # Buy (dir=1) only when divergence=+1 (H4 bull, M5 dip)
                # Sell (dir=-1) only when divergence=-1 (H4 bear, M5 rally)
                div_ok = (dirs * divs) > 0  # same sign = divergence supports direction
                mask = mask & div_ok

            if mask.sum() == 0: continue
            sel = y_te[mask]
            for s in sel:
                combined["pnls"].append(TP_MULT if s==1 else -SL_MULT)
                if s==1: combined["wins"]+=1
                else: combined["losses"]+=1
            rules_done += 1

        pnls = np.array(combined["pnls"])
        if len(pnls)==0: print(f"  {name:45s} no trades"); return None
        n=len(pnls); w=combined["wins"]; lo=combined["losses"]
        wr=w/n; pf=(w*TP_MULT)/(lo*SL_MULT+1e-9)
        eq=np.cumsum(pnls); dd=(eq-np.maximum.accumulate(eq)).min()
        print(f"  {name:45s} n={n:3d}  WR={wr:.1%}  PF={pf:.3f}  PnL={pnls.sum():+.1f}R  DD={dd:.1f}R", flush=True)
        return {"name":name, "pnls":pnls, "n":n, "wr":wr, "pf":pf, "total":pnls.sum(), "dd":dd}

    results = []
    r = run_config(all_df, CURRENT_36, "A) Current 36 (baseline)");
    if r: results.append(r)
    r = run_config(all_df, E_12, "E) Phys+VWAP+Quantum (12)");
    if r: results.append(r)
    r = run_config(all_df, E_12, "E2) E + divergence FILTER", use_div_filter=True);
    if r: results.append(r)
    r = run_config(all_df, E3_14, "E3) E + divergence as FEATURE (14)");
    if r: results.append(r)
    r = run_config(all_df, E3_14, "E4) E3 + divergence FILTER (14)", use_div_filter=True);
    if r: results.append(r)

    # Equity curves
    fig, ax = plt.subplots(figsize=(15, 6), facecolor="#0b0e14")
    ax.set_facecolor("#0d1117")
    palette = ["#3b82f6", "#f5c518", "#ff6b6b", "#10b981", "#a855f7"]
    for i, r in enumerate(results):
        eq = np.cumsum(r["pnls"])
        ax.plot(eq, color=palette[i%len(palette)], linewidth=1.5,
                label=f"{r['name']} PF={r['pf']:.2f}")
    ax.axhline(0, color="#444", linewidth=0.5)
    ax.legend(facecolor="#111", edgecolor="#333", fontsize=8, loc="upper left")
    ax.set_title("XAU — Quantum Flow Divergence Pattern Test — Full Holdout",
                 color="#ffd700", fontsize=13)
    ax.set_xlabel("trade #", color="#888"); ax.set_ylabel("cumulative R", color="#888")
    ax.tick_params(colors="#888")
    for sp in ax.spines.values(): sp.set_edgecolor("#1a2332")
    plt.tight_layout()
    out = f"{ROOT}/experiments/innovative_rules/phase4b_divergence.png"
    plt.savefig(out, dpi=140, facecolor="#0b0e14")
    print(f"\nSaved: {out}", flush=True)


if __name__ == "__main__":
    main()
