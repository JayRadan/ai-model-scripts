"""
Train cross-instrument trade-quality predictor at per-trade level
(not per-day — gives us more samples), then apply as a gate on the
holdout for both Midas and Oracle.

Approach:
  1. Build feature matrix at TRADE level: for each trade in v6 Midas
     holdout, look up that date's cross-instrument features.
  2. Train (Midas trades, 2024-01-02 → 2024-12-11) → predict pnl_R.
     We use a logistic classifier on (won = pnl_R > 0).
  3. Score every holdout trade (both v6 Midas and v72l Oracle) with the
     classifier → p_good.
  4. Sweep gate thresholds: keep trades only when p_good > thr.
     Report ΔPF, ΔR, blocked count for each threshold.
  5. Sanity test: pearson(p_good, pnl_R). Needs ≥ +0.05 to matter
     (same bar v76 used).
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

ROOT = "/home/jay/Desktop/new-model-zigzag"
DATA = os.path.join(os.path.dirname(__file__), "data")
REPS = os.path.join(os.path.dirname(__file__), "reports")
os.makedirs(REPS, exist_ok=True)

HOLDOUT_START = pd.Timestamp("2024-12-12")


def load_features() -> pd.DataFrame:
    df = pd.read_parquet(os.path.join(DATA, "features_labels.parquet"))
    feature_cols = [c for c in df.columns
                    if c not in {"date", "n_trades", "sum_R",
                                  "label", "split"}]
    return df[["date"] + feature_cols], feature_cols


def attach_features(trades: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    trades = trades.copy()
    trades["date"] = pd.to_datetime(trades["time"]).dt.normalize()
    return trades.merge(feats, on="date", how="inner")


def pf(rs: np.ndarray) -> float:
    pos = rs[rs > 0].sum()
    neg = -rs[rs <= 0].sum()
    return pos / max(neg, 1e-9)


def evaluate_gate(name: str, trades: pd.DataFrame, p: np.ndarray) -> None:
    rs = trades["pnl_R"].values
    base_pf = pf(rs); base_r = rs.sum(); n = len(rs)
    print(f"\n[{name}] baseline: n={n}, PF={base_pf:.3f}, R={base_r:+.1f}")

    corr, _ = pearsonr(p, rs)
    print(f"  pearson(p_good, pnl_R) = {corr:+.4f}  "
          f"({'>= +0.05 useful' if abs(corr) >= 0.05 else '~ noise'})")

    rows = []
    for thr in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        keep = p > thr
        if keep.sum() == 0:
            continue
        kept_pf = pf(rs[keep]); kept_r = rs[keep].sum()
        blocked = (~keep).sum(); blocked_r = rs[~keep].sum()
        rows.append({
            "threshold":   thr,
            "kept":        int(keep.sum()),
            "blocked":     int(blocked),
            "kept_PF":     round(kept_pf, 3),
            "kept_R":      round(kept_r, 1),
            "ΔPF":         round(kept_pf - base_pf, 3),
            "ΔR":          round(kept_r - base_r, 1),
            "blocked_R":   round(blocked_r, 1),
        })
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    out.to_csv(os.path.join(REPS, f"gate_sweep_{name}.csv"), index=False)


def main() -> None:
    feats, fcols = load_features()

    # ---- TRAINING: Midas trades pre-2024-12-12 ----
    midas = pd.read_csv(os.path.join(ROOT, "data/v6_trades_holdout_xau.csv"),
                         parse_dates=["time"])
    midas_full = attach_features(midas, feats)
    train = midas_full[midas_full["date"] < HOLDOUT_START].copy()
    print(f"train: {len(train)} Midas trades 2024-01 → 2024-12-11")
    print(f"  win rate: {(train['pnl_R'] > 0).mean():.3f}")

    Xtr = train[fcols].values
    ytr = (train["pnl_R"] > 0).astype(int).values
    sc  = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=0)
    clf.fit(sc.transform(Xtr), ytr)
    in_prob = clf.predict_proba(sc.transform(Xtr))[:, 1]
    print(f"  in-sample AUC proxy "
          f"(corr(p, pnl_R)) = {pearsonr(in_prob, train['pnl_R'].values)[0]:+.4f}")

    # Coefficients — which features carry weight?
    coef_df = pd.DataFrame({"feature": fcols,
                             "coef":    clf.coef_[0]}).sort_values(
        "coef", key=abs, ascending=False)
    print("\nTop coefficients:")
    print(coef_df.head(8).to_string(index=False))
    coef_df.to_csv(os.path.join(REPS, "coefficients.csv"), index=False)

    # ---- HOLDOUT: Midas + Oracle 2024-12-12 onwards ----
    midas_hold = midas_full[midas_full["date"] >= HOLDOUT_START].copy()
    if len(midas_hold) > 0:
        p = clf.predict_proba(sc.transform(midas_hold[fcols].values))[:, 1]
        evaluate_gate("midas_holdout", midas_hold, p)

    oracle = pd.read_csv(os.path.join(ROOT, "data/v72l_trades_holdout.csv"),
                          parse_dates=["time"])
    oracle_hold = attach_features(oracle, feats)
    if len(oracle_hold) > 0:
        p = clf.predict_proba(sc.transform(oracle_hold[fcols].values))[:, 1]
        evaluate_gate("oracle_holdout", oracle_hold, p)


if __name__ == "__main__":
    main()
