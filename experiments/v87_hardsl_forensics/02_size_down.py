"""
Follow-up to v8.7 forensics: predictive AUC was 0.57-0.58 — too low for
a binary skip filter (winners-killed > losers-saved), but maybe useful
as a NEGATIVE size multiplier (don't skip, just risk less).

Sizing variants tested:
  A) lot_mult = 1.0 - α × p_hsl, with α ∈ {0.3, 0.5, 0.8, 1.0}
     (clipped to [0.2, 1.0])
  B) Step: lot_mult = 1.0 if p_hsl < t1, 0.5 if t1 ≤ p_hsl < t2, 0.25 else
"""
from __future__ import annotations
import os, sys, glob
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

ROOT = "/home/jay/Desktop/new-model-zigzag"
V72L_FEATS = [
    "hurst_rs","ou_theta","entropy_rate","kramers_up","wavelet_er",
    "vwap_dist","hour_enc","dow_enc",
    "quantum_flow","quantum_flow_h4","quantum_momentum","quantum_vwap_conf",
    "quantum_divergence","quantum_div_strength",
    "vpin","sig_quad_var","har_rv_ratio","hawkes_eta",
]
V6_FEATS = V72L_FEATS[:14]


def pf(rs):
    pos = rs[rs > 0].sum(); neg = -rs[rs <= 0].sum()
    return pos / max(neg, 1e-9)


def load_setups(g):
    parts = []
    for f in sorted(glob.glob(os.path.join(ROOT, "data", g))):
        cid = int(os.path.basename(f).split("_")[1])
        df = pd.read_csv(f, parse_dates=["time"])
        df["cid"] = cid; parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def attach(trades, setups, feats):
    out = trades.copy()
    out["cid"] = out["cid"].astype(int); out["direction"] = out["direction"].astype(int)
    return out.merge(setups[["time","cid","rule","direction"] + feats],
                       on=["time","cid","rule","direction"], how="left")\
               .dropna(subset=feats).reset_index(drop=True)


def evaluate(name, trades_path, feats, glob_pat):
    print("\n" + "="*78); print(f"=== {name} ===")
    setups = load_setups(glob_pat)
    trades = pd.read_csv(os.path.join(ROOT, trades_path), parse_dates=["time"])
    df = attach(trades, setups, feats).sort_values("time").reset_index(drop=True)

    # Train on H1 winner-vs-hardsl
    win = df[df.pnl_R > 0].copy(); win["y"] = 0
    hsl = df[df.exit == "hard_sl"].copy(); hsl["y"] = 1
    pool = pd.concat([win, hsl]).sort_values("time").reset_index(drop=True)
    n_tr = len(pool) // 2
    Xtr = pool.iloc[:n_tr][feats + ["direction","cid"]].fillna(0).values
    ytr = pool.iloc[:n_tr]["y"].values
    mdl = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                         eval_metric="logloss", verbosity=0, random_state=0,
                         tree_method="hist")
    mdl.fit(Xtr, ytr)

    # Score H2 of the FULL holdout
    H2_start = len(df) // 2
    h2 = df.iloc[H2_start:].copy()
    p_hsl = mdl.predict_proba(h2[feats + ["direction","cid"]].fillna(0).values)[:, 1]
    h2["p_hsl"] = p_hsl
    rs0 = h2.pnl_R.values
    base_R = rs0.sum(); base_pf = pf(rs0); base_wr = (rs0 > 0).mean()
    print(f"  H2 baseline: n={len(rs0)} WR={base_wr:.1%} PF={base_pf:.2f} R={base_R:+.0f}")

    rows = []
    for alpha in [0.3, 0.5, 0.8, 1.0]:
        lots = np.clip(1.0 - alpha * p_hsl, 0.2, 1.0)
        wr = rs0 * lots
        rows.append({
            "scheme": f"linear_α={alpha}",
            "avg_lot": round(lots.mean(), 3),
            "WR": round((wr > 0).mean(), 4),
            "PF": round(pf(wr), 2),
            "R":  round(wr.sum(), 1),
            "ΔR": round(wr.sum() - base_R, 1),
            "R_per_lot": round(wr.sum() / lots.sum(), 3),
        })
    # Step schemes
    for (t1, t2, l1, l2, l3) in [
        (0.30, 0.50, 1.0, 0.5, 0.25),
        (0.40, 0.60, 1.0, 0.6, 0.30),
    ]:
        lots = np.where(p_hsl < t1, l1,
                          np.where(p_hsl < t2, l2, l3))
        wr = rs0 * lots
        rows.append({
            "scheme": f"step({t1}/{t2}→{l1}/{l2}/{l3})",
            "avg_lot": round(lots.mean(), 3),
            "WR": round((wr > 0).mean(), 4),
            "PF": round(pf(wr), 2),
            "R":  round(wr.sum(), 1),
            "ΔR": round(wr.sum() - base_R, 1),
            "R_per_lot": round(wr.sum() / lots.sum(), 3),
        })
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    return out


def main():
    products = [
        ("Oracle XAU", "data/v72l_trades_holdout.csv",   V72L_FEATS, "setups_*_v72l.csv"),
        ("Midas XAU",  "data/v6_trades_holdout_xau.csv", V6_FEATS,   "setups_*_v6.csv"),
        ("Oracle BTC", "data/v72l_trades_holdout_btc.csv", V72L_FEATS, "setups_*_v72l_btc.csv"),
    ]
    for name, path, feats, g in products:
        evaluate(name, path, feats, g)


if __name__ == "__main__":
    main()
