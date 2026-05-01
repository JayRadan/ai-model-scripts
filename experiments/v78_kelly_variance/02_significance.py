"""
Validate v78 sizing wins:
  1) Permutation test — is B_sqrt's R/|DD| lift on Midas significant?
     Null: shuffle predictions, recompute lots, recompute R/|DD|.
     Real lift must beat 95th percentile of null.
  2) Walk-forward halves — does the lift hold in BOTH halves of the
     holdout independently? Catches "signal works in one regime, not
     other" failure mode.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

ROOT = "/home/jay/Desktop/new-model-zigzag"
HOLDOUT_START = pd.Timestamp("2024-12-12")
RNG = np.random.default_rng(0)

XAU_PREFIXES = [f"f{i:02d}_" for i in range(1, 21)]
H_COLS = ["h1_trend_sma20","h1_trend_sma50","h1_slope5","h1_rsi14",
          "h1_atr_ratio","h1_dist_sma20","h1_dist_sma50",
          "h4_trend_sma20","h4_trend_sma50","h4_slope5","h4_rsi14",
          "h4_atr_ratio","h4_dist_sma20","h4_dist_sma50"]


def pf(rs):
    pos = rs[rs > 0].sum(); neg = -rs[rs <= 0].sum()
    return pos / max(neg, 1e-9)


def max_dd(rs):
    eq = np.cumsum(rs); peak = np.maximum.accumulate(eq)
    return (eq - peak).min()


def b_sqrt_lots(p, median):
    return np.clip(np.sqrt(np.maximum(p, 1e-6) / median), 0.5, 2.0)


def attach(trades):
    swing = pd.read_csv(os.path.join(ROOT, "data/swing_v5_xauusd.csv"),
                         parse_dates=["time"])
    f_cols = ([c for c in swing.columns
                if any(c.startswith(p) for p in XAU_PREFIXES)] +
              [c for c in H_COLS if c in swing.columns])
    swing = swing[["time"] + f_cols].sort_values("time").reset_index(drop=True)
    out = trades.sort_values("time").reset_index(drop=True).copy()
    out = pd.merge_asof(out, swing, on="time", direction="backward",
                         tolerance=pd.Timedelta("10min"))
    cross = pd.read_parquet(os.path.join(
        ROOT, "experiments/v77_cross_instrument/data/features_labels.parquet"))
    cross_cols = [c for c in cross.columns
                  if c not in {"date","n_trades","sum_R","label","split"}]
    out["date"] = pd.to_datetime(out["time"]).dt.normalize()
    out = out.merge(cross[["date"] + cross_cols], on="date", how="left")
    return out, f_cols + cross_cols


def main():
    midas = pd.read_csv(os.path.join(ROOT, "data/v6_trades_holdout_xau.csv"),
                          parse_dates=["time"])
    df, fcols = attach(midas)
    df = df.dropna(subset=fcols).reset_index(drop=True)
    df["abs_R"] = df["pnl_R"].abs()
    train = df[df["time"] < HOLDOUT_START]
    test  = df[df["time"] >= HOLDOUT_START].sort_values("time").reset_index(drop=True)

    model = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                          random_state=0, n_jobs=4)
    model.fit(train[fcols].values, train["abs_R"].values)
    pred = model.predict(test[fcols].values)
    rs = test["pnl_R"].values
    median = np.median(pred)

    # -------- 1) Permutation test --------
    real_lots = b_sqrt_lots(pred, median)
    real_wr = rs * real_lots
    real_R   = real_wr.sum()
    real_DD  = max_dd(real_wr)
    real_RDD = real_R / abs(real_DD)
    base_R   = rs.sum(); base_DD = max_dd(rs)
    base_RDD = base_R / abs(base_DD)

    print("=== Permutation: shuffle predictions, recompute B_sqrt sizing ===")
    print(f"baseline:  R={base_R:+.1f}  DD={base_DD:+.1f}  R/|DD|={base_RDD:.2f}")
    print(f"real:      R={real_R:+.1f}  DD={real_DD:+.1f}  R/|DD|={real_RDD:.2f}")

    n_iter = 5000
    null_R = np.empty(n_iter); null_DD = np.empty(n_iter); null_RDD = np.empty(n_iter)
    for i in range(n_iter):
        shuf = RNG.permutation(pred)
        lots = b_sqrt_lots(shuf, np.median(shuf))
        wr = rs * lots
        null_R[i] = wr.sum(); null_DD[i] = max_dd(wr)
        null_RDD[i] = wr.sum() / abs(max_dd(wr))
    print(f"null R    : mean={null_R.mean():+.1f}  95th={np.percentile(null_R,95):+.1f}"
          f"  p(real ≤ null)={(null_R >= real_R).mean():.4f}")
    print(f"null DD   : mean={null_DD.mean():+.1f}  5th ={np.percentile(null_DD,5):+.1f}"
          f"  p(real worse)={(null_DD <= real_DD).mean():.4f}")
    print(f"null R/DD : mean={null_RDD.mean():.2f}  95th={np.percentile(null_RDD,95):.2f}"
          f"  p(real ≤ null)={(null_RDD >= real_RDD).mean():.4f}")

    # -------- 2) Walk-forward halves --------
    print("\n=== Walk-forward: split holdout into halves ===")
    mid = len(test) // 2
    for label, sl in [("H1 (early)", slice(0, mid)),
                       ("H2 (late)",  slice(mid, None))]:
        seg_rs   = rs[sl]; seg_p = pred[sl]
        seg_med  = np.median(seg_p)
        seg_lots = b_sqrt_lots(seg_p, seg_med)
        seg_wr   = seg_rs * seg_lots
        b_R = seg_rs.sum(); b_DD = max_dd(seg_rs)
        s_R = seg_wr.sum(); s_DD = max_dd(seg_wr)
        print(f"  {label}  n={len(seg_rs)}")
        print(f"    baseline: R={b_R:+.1f}  PF={pf(seg_rs):.3f}  DD={b_DD:+.1f}"
              f"  R/|DD|={b_R/abs(b_DD):.2f}")
        print(f"    B_sqrt:   R={s_R:+.1f}  PF={pf(seg_wr):.3f}  DD={s_DD:+.1f}"
              f"  R/|DD|={s_R/abs(s_DD):.2f}  ΔR={s_R-b_R:+.1f}")


if __name__ == "__main__":
    main()
