"""
v7.8 — Predict trade magnitude |pnl_R| (not direction), size by Kelly-ish proxy.

Hypothesis: |pnl_R| is more predictable than sign(pnl_R) because |R|
correlates with realized volatility & trend persistence, both of which
have memory. Sign is closer to a coin flip after the confirm heads
already extracted the directional signal.

Method:
  1. Per-trade features: swing_v5 (XAU 40 features at fire time) + v77
     daily cross-instrument (already-shifted, no lookahead).
  2. Train XGB regressor on Midas pre-2024-12-12, target = |pnl_R|.
  3. Holdout test: corr(predicted, actual |pnl_R|).
  4. Sizing schemes:
     A) Kelly-ish:  lot = clip(predicted / median_predicted, 0.4, 2.5)
     B) Sqrt:       lot = clip(sqrt(predicted / median), 0.5, 2.0)
     C) Cap-only:   lot = 1.0 if predicted < p25 else 1.5 if > p75 else 1.0
  5. Compare: R, PF, maxDD, R/|maxDD| (sortino-ish).

Pass bar: ANY scheme delivers ≥+5% R AND ≥+0.10 PF AND DD not worse,
   OR R/|maxDD| improves ≥10% with R within 5% of baseline.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from scipy.stats import pearsonr, spearmanr

ROOT = "/home/jay/Desktop/new-model-zigzag"
OUT  = os.path.join(os.path.dirname(__file__), "reports")
os.makedirs(OUT, exist_ok=True)

HOLDOUT_START = pd.Timestamp("2024-12-12")

XAU_FEAT_COLS = [f"f{i:02d}_" for i in range(1, 21)]   # prefix match
H_FEAT_COLS   = ["h1_trend_sma20","h1_trend_sma50","h1_slope5","h1_rsi14",
                 "h1_atr_ratio","h1_dist_sma20","h1_dist_sma50",
                 "h4_trend_sma20","h4_trend_sma50","h4_slope5","h4_rsi14",
                 "h4_atr_ratio","h4_dist_sma20","h4_dist_sma50"]


def pf(rs):
    pos = rs[rs > 0].sum(); neg = -rs[rs <= 0].sum()
    return pos / max(neg, 1e-9)


def max_dd(rs):
    eq = np.cumsum(rs); peak = np.maximum.accumulate(eq)
    return (eq - peak).min()


def attach_features(trades: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    print("loading swing_v5 features…")
    swing = pd.read_csv(os.path.join(ROOT, "data/swing_v5_xauusd.csv"),
                         parse_dates=["time"])
    f_cols = [c for c in swing.columns
              if any(c.startswith(p) for p in XAU_FEAT_COLS)] + H_FEAT_COLS
    f_cols = [c for c in f_cols if c in swing.columns]
    swing = swing[["time"] + f_cols].sort_values("time").reset_index(drop=True)

    out = trades.sort_values("time").reset_index(drop=True).copy()
    out = pd.merge_asof(out, swing, on="time", direction="backward",
                         tolerance=pd.Timedelta("10min"))

    # Cross-instrument daily features from v77 — already shifted to no-lookahead
    cross = pd.read_parquet(os.path.join(
        ROOT, "experiments/v77_cross_instrument/data/features_labels.parquet"))
    cross_cols = [c for c in cross.columns
                  if c not in {"date","n_trades","sum_R","label","split"}]
    out["date"] = pd.to_datetime(out["time"]).dt.normalize()
    out = out.merge(cross[["date"] + cross_cols], on="date", how="left")

    feat_cols = f_cols + cross_cols
    return out, feat_cols


def main():
    trades = pd.read_csv(os.path.join(ROOT, "data/v6_trades_holdout_xau.csv"),
                          parse_dates=["time"])
    print(f"loaded {len(trades)} Midas trades")
    df, feat_cols = attach_features(trades)
    df = df.dropna(subset=feat_cols).reset_index(drop=True)
    df["abs_R"] = df["pnl_R"].abs()
    print(f"after feature merge: {len(df)} trades, {len(feat_cols)} features")

    train = df[df["time"] < HOLDOUT_START].copy()
    test  = df[df["time"] >= HOLDOUT_START].copy()
    print(f"train: {len(train)}  holdout: {len(test)}")
    print(f"  train |R| stats: mean={train.abs_R.mean():.3f} "
          f"med={train.abs_R.median():.3f} max={train.abs_R.max():.2f}")
    print(f"  test  |R| stats: mean={test.abs_R.mean():.3f} "
          f"med={test.abs_R.median():.3f} max={test.abs_R.max():.2f}")

    Xtr = train[feat_cols].values; ytr = train["abs_R"].values
    Xte = test [feat_cols].values; yte = test ["abs_R"].values

    model = XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=0, n_jobs=4)
    model.fit(Xtr, ytr)

    pred_tr = model.predict(Xtr)
    pred_te = model.predict(Xte)

    print(f"\nin-sample  pearson(pred, |R|) = {pearsonr(pred_tr, ytr)[0]:+.4f}"
          f"  spearman = {spearmanr(pred_tr, ytr)[0]:+.4f}")
    print(f"holdout    pearson(pred, |R|) = {pearsonr(pred_te, yte)[0]:+.4f}"
          f"  spearman = {spearmanr(pred_te, yte)[0]:+.4f}")

    # Predicted magnitude vs actual signed pnl_R (the thing that matters)
    print(f"\nholdout    pearson(pred_|R|, pnl_R)     = "
          f"{pearsonr(pred_te, test['pnl_R'].values)[0]:+.4f}  "
          f"(should be near 0 — magnitude isn't direction)")
    print(f"holdout    pearson(pred_|R|, sign(pnl_R)) = "
          f"{pearsonr(pred_te, np.sign(test['pnl_R'].values))[0]:+.4f}")

    # ---- Sizing simulations on holdout ----
    print("\n=== Sizing schemes (holdout Midas) ===")
    rs = test["pnl_R"].values
    base_r = rs.sum(); base_pf = pf(rs); base_dd = max_dd(rs)
    print(f"  baseline lot=1.0: R={base_r:+.1f}  PF={base_pf:.3f}  "
          f"maxDD={base_dd:+.1f}  R/|DD|={base_r/abs(base_dd):.2f}")

    median_pred = np.median(pred_te)
    p25, p75 = np.percentile(pred_te, [25, 75])

    schemes = {
        "A_kelly":   np.clip(pred_te / median_pred, 0.4, 2.5),
        "B_sqrt":    np.clip(np.sqrt(np.maximum(pred_te, 1e-6) / median_pred), 0.5, 2.0),
        "C_3step":   np.where(pred_te < p25, 1.0,
                       np.where(pred_te > p75, 1.5, 1.0)),
        "D_5step":   np.clip(0.5 + 2.0 * (pred_te - p25) / (p75 - p25), 0.4, 2.5),
    }
    rows = [{"scheme":"baseline","R":round(base_r,1),"PF":round(base_pf,3),
             "maxDD":round(base_dd,1),"avg_lot":1.0,
             "R/|DD|":round(base_r/abs(base_dd),2)}]
    for name, lots in schemes.items():
        wr = rs * lots
        rows.append({
            "scheme":   name,
            "R":        round(wr.sum(), 1),
            "PF":       round(pf(wr), 3),
            "maxDD":    round(max_dd(wr), 1),
            "avg_lot":  round(lots.mean(), 3),
            "R/|DD|":   round(wr.sum()/abs(max_dd(wr)), 2),
        })
    out_df = pd.DataFrame(rows)
    print(out_df.to_string(index=False))
    out_df.to_csv(os.path.join(OUT, "sizing_midas.csv"), index=False)

    # ---- Now test if model TRANSFERS to Oracle ----
    print("\n=== Transfer test on Oracle holdout ===")
    oracle = pd.read_csv(os.path.join(ROOT, "data/v72l_trades_holdout.csv"),
                          parse_dates=["time"])
    odf, _ = attach_features(oracle)
    odf = odf.dropna(subset=feat_cols).reset_index(drop=True)
    odf = odf[odf["time"] >= HOLDOUT_START]
    if len(odf) > 0:
        op = model.predict(odf[feat_cols].values)
        oy = odf["pnl_R"].abs().values
        print(f"  oracle pearson(pred, |R|)  = {pearsonr(op, oy)[0]:+.4f}")
        print(f"  oracle spearman(pred, |R|) = {spearmanr(op, oy)[0]:+.4f}")
        ors = odf["pnl_R"].values
        ob_r = ors.sum(); ob_pf = pf(ors); ob_dd = max_dd(ors)
        print(f"  baseline: R={ob_r:+.1f} PF={ob_pf:.3f} maxDD={ob_dd:+.1f}")
        med = np.median(op); p25, p75 = np.percentile(op, [25, 75])
        rows2 = [{"scheme":"baseline","R":round(ob_r,1),"PF":round(ob_pf,3),
                  "maxDD":round(ob_dd,1),"avg_lot":1.0,
                  "R/|DD|":round(ob_r/abs(ob_dd),2)}]
        oschemes = {
            "A_kelly":   np.clip(op / med, 0.4, 2.5),
            "B_sqrt":    np.clip(np.sqrt(np.maximum(op, 1e-6)/med), 0.5, 2.0),
            "D_5step":   np.clip(0.5 + 2.0 * (op - p25) / (p75 - p25), 0.4, 2.5),
        }
        for name, lots in oschemes.items():
            wr = ors * lots
            rows2.append({
                "scheme": name, "R": round(wr.sum(),1),
                "PF": round(pf(wr),3), "maxDD": round(max_dd(wr),1),
                "avg_lot": round(lots.mean(),3),
                "R/|DD|": round(wr.sum()/abs(max_dd(wr)),2),
            })
        out_df = pd.DataFrame(rows2)
        print(out_df.to_string(index=False))
        out_df.to_csv(os.path.join(OUT, "sizing_oracle.csv"), index=False)


if __name__ == "__main__":
    main()
