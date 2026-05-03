"""
v8.6 — Train per-product variance heads (predict |pnl_R|) for production
sizing.

Feature set: 18 v72L features (already computed by decide.py) + direction + cid.
This is exactly META_FEATS — no new runtime feature compute required.

Label: |pnl_R| of the trade (magnitude regardless of sign).

Per product:
  Train on pre-2024-12-12 trades (these are NOT in our holdout — they're
  validation-window trades, but we're predicting magnitude not direction
  so the leak risk is lower than for a sign-prediction model).

Wait — for Oracle the holdout STARTS at 2024-12-12. So we don't have
"pre-2024-12-12 Oracle trades" available. We have Midas pre-12-12 trades
(holdout Midas starts 2024-01-02).

Strategy:
  - For Oracle XAU: train variance head on MIDAS pre-12-12 trades (using
    same v72L features). v78 showed this transfers cross-product.
  - For Midas XAU: same Midas pre-12-12 trades.
  - For Oracle BTC: train on Midas pre-12-12 trades too (cross-instrument
    transfer is uncertain — may underperform; we'll measure).

Verify holdout spearman per product. If ≥ +0.05, accept. If not, the
sizing won't help and we abort that product.

Saves variance heads to:
  /home/jay/Desktop/my-agents-and-website/commercial/server/decision_engine/models/variance_<product>.pkl

Each pickle contains:
  {
    "variance_mdl":     XGBRegressor,
    "variance_feats":   tuple of feat names (= META_FEATS),
    "variance_median":  float (median predicted value, used as the size baseline),
    "variance_min_lot": 0.5,
    "variance_max_lot": 2.0,
    "trained_on":       "midas pre-2024-12-12",
    "holdout_spearman": float (sanity-check value),
  }
"""
from __future__ import annotations
import os, glob, pickle
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from scipy.stats import spearmanr, pearsonr

ROOT = "/home/jay/Desktop/new-model-zigzag"
OUT  = "/home/jay/Desktop/my-agents-and-website/commercial/server/decision_engine/models"
HOLDOUT_START = pd.Timestamp("2024-12-12")

# Match Oracle/Janus META_FEATS: 18 v72L + direction + cid.
V72L_FEATS = [
    "hurst_rs", "ou_theta", "entropy_rate", "kramers_up", "wavelet_er",
    "vwap_dist", "hour_enc", "dow_enc",
    "quantum_flow", "quantum_flow_h4", "quantum_momentum", "quantum_vwap_conf",
    "quantum_divergence", "quantum_div_strength",
    "vpin", "sig_quad_var", "har_rv_ratio", "hawkes_eta",
]

# Midas v6 has 14 features (no v72L extras). For Midas variance head we use
# only the 14 it has at runtime. For Oracle/BTC we use the 18.
V6_FEATS = [
    "hurst_rs", "ou_theta", "entropy_rate", "kramers_up", "wavelet_er",
    "vwap_dist", "hour_enc", "dow_enc",
    "quantum_flow", "quantum_flow_h4", "quantum_momentum", "quantum_vwap_conf",
    "quantum_divergence", "quantum_div_strength",
]


def load_setups_with_features(feat_glob, feat_cols):
    """Load all setups for a feature pack (e.g. setups_*_v72l.csv)."""
    setups = []
    for f in sorted(glob.glob(os.path.join(ROOT, "data", feat_glob))):
        cid = int(os.path.basename(f).split("_")[1])
        df = pd.read_csv(f, parse_dates=["time"])
        df["cid"] = cid
        setups.append(df)
    setups = pd.concat(setups, ignore_index=True)
    setups["cid"] = setups["cid"].astype(int)
    setups["direction"] = setups["direction"].astype(int)
    return setups[["time", "cid", "rule", "direction"] + feat_cols]


def attach_features(trades, setups, feat_cols):
    trades = trades.copy()
    trades["cid"] = trades["cid"].astype(int)
    trades["direction"] = trades["direction"].astype(int)
    return trades.merge(setups, on=["time","cid","rule","direction"], how="left")\
                  .dropna(subset=feat_cols)\
                  .reset_index(drop=True)


def train_one(name, training_trades_path, holdout_trades_path,
                feat_cols, feat_glob, out_filename,
                trained_on_label):
    print(f"\n=== {name} ===", flush=True)
    setups = load_setups_with_features(feat_glob, feat_cols)
    print(f"  loaded {len(setups):,} setup rows ({feat_glob})", flush=True)

    train_trades = pd.read_csv(os.path.join(ROOT, training_trades_path),
                                 parse_dates=["time"])
    train_trades = train_trades[train_trades["time"] < HOLDOUT_START].copy()
    train = attach_features(train_trades, setups, feat_cols)
    META_FEATS = feat_cols + ["direction", "cid"]
    print(f"  training set: {len(train)} trades pre-{HOLDOUT_START.date()}", flush=True)
    print(f"  feature count: {len(META_FEATS)}", flush=True)

    Xtr = train[META_FEATS].fillna(0).values
    ytr = train["pnl_R"].abs().values

    mdl = XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=0, n_jobs=4, verbosity=0)
    mdl.fit(Xtr, ytr)
    pred_tr = mdl.predict(Xtr)
    median_pred = float(np.median(pred_tr))
    in_corr, _ = spearmanr(pred_tr, ytr)
    print(f"  in-sample spearman(pred, |R|) = {in_corr:+.4f}", flush=True)
    print(f"  median predicted |R|: {median_pred:.3f}", flush=True)

    # Holdout sanity
    holdout = pd.read_csv(os.path.join(ROOT, holdout_trades_path),
                            parse_dates=["time"])
    holdout = holdout[holdout["time"] >= HOLDOUT_START].copy()
    h = attach_features(holdout, setups, feat_cols)
    Xh = h[META_FEATS].fillna(0).values
    yh = h["pnl_R"].abs().values
    pred_h = mdl.predict(Xh)
    h_pearson, _ = pearsonr(pred_h, yh)
    h_spearman, _ = spearmanr(pred_h, yh)
    print(f"  HOLDOUT pearson(pred, |R|)  = {h_pearson:+.4f}", flush=True)
    print(f"  HOLDOUT spearman(pred, |R|) = {h_spearman:+.4f}", flush=True)

    # Sizing simulation: B_sqrt
    rs = h["pnl_R"].values
    lots = np.clip(np.sqrt(np.maximum(pred_h, 1e-6) / median_pred), 0.5, 2.0)
    weighted = rs * lots
    base_R = rs.sum(); base_PF = rs[rs>0].sum() / max(-rs[rs<=0].sum(), 1e-9)
    new_R = weighted.sum(); new_PF = weighted[weighted>0].sum() / max(-weighted[weighted<=0].sum(), 1e-9)
    print(f"  baseline:  R={base_R:+.0f}  PF={base_PF:.3f}", flush=True)
    print(f"  B_sqrt:    R={new_R:+.0f}  PF={new_PF:.3f}  avg_lot={lots.mean():.3f}  "
          f"ΔR={new_R-base_R:+.0f} ({(new_R/base_R - 1)*100:+.1f}%)", flush=True)

    if h_spearman < 0.05:
        print(f"  ⚠ HOLDOUT spearman {h_spearman:.4f} < +0.05 — variance head not predictive enough, NOT shipping {name}", flush=True)
        return None

    payload = {
        "variance_mdl":     mdl,
        "variance_feats":   tuple(META_FEATS),
        "variance_median":  median_pred,
        "variance_min_lot": 0.5,
        "variance_max_lot": 2.0,
        "trained_on":       trained_on_label,
        "holdout_spearman": float(h_spearman),
        "holdout_R_lift":   float(new_R - base_R),
    }
    out_path = os.path.join(OUT, out_filename)
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"  → wrote {out_path}", flush=True)
    return payload


def main():
    os.makedirs(OUT, exist_ok=True)
    # Oracle XAU + Janus XAU + Oracle BTC use 18 v72L features
    # Midas XAU uses 14 v6 features (no v72L extras)
    train_one(
        name="Oracle XAU variance head",
        training_trades_path="data/v6_trades_holdout_xau.csv",   # using Midas trades pre-12-12 as training
        holdout_trades_path="data/v72l_trades_holdout.csv",       # Oracle holdout for sanity
        feat_cols=V72L_FEATS,
        feat_glob="setups_*_v72l.csv",
        out_filename="variance_oracle_xau.pkl",
        trained_on_label="midas v6 trades pre-2024-12-12",
    )
    train_one(
        name="Midas XAU variance head",
        training_trades_path="data/v6_trades_holdout_xau.csv",
        holdout_trades_path="data/v6_trades_holdout_xau.csv",
        feat_cols=V6_FEATS,
        feat_glob="setups_*_v6.csv",
        out_filename="variance_midas_xau.pkl",
        trained_on_label="midas v6 trades pre-2024-12-12",
    )
    train_one(
        name="Oracle BTC variance head",
        training_trades_path="data/v6_trades_holdout_xau.csv",   # XAU trades — cross-instrument
        holdout_trades_path="data/v72l_trades_holdout_btc.csv",
        feat_cols=V72L_FEATS,
        feat_glob="setups_*_v72l.csv",                            # XAU setups
        out_filename="variance_oracle_btc.pkl",
        trained_on_label="midas v6 XAU trades pre-2024-12-12 (cross-instrument)",
    )


if __name__ == "__main__":
    main()
