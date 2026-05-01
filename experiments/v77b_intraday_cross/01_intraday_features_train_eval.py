"""
Build per-trade intraday cross-instrument features, train classifier on
Midas pre-2024-12-12 trades, evaluate on Midas + Oracle holdouts.

For each XAU trade time t:
  for each instrument X (EURUSD, SPX, TNX, VIX, UUP):
    find most recent X close ≤ t (asof merge — strictly no lookahead)
    features:
      X_ret_1h        last 1h return (or change for TNX/VIX)
      X_ret_4h        last 4h return
      X_ret_24h       last 24h return
      X_ret_24h_z     24h return / std(24h returns over last 30 days)

This gives 4 features × 5 instruments = 20 intraday features.

Compare against v77 daily baseline:
  - Trade-level corr(p, pnl_R)
  - Top-quintile gate PF + permutation p-value
  - Sizing scheme A_linear vs baseline
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

ROOT  = "/home/jay/Desktop/new-model-zigzag"
DATA  = os.path.join(os.path.dirname(__file__), "data")
REPS  = os.path.join(os.path.dirname(__file__), "reports")
os.makedirs(REPS, exist_ok=True)

HOLDOUT_START = pd.Timestamp("2024-12-12")
RNG = np.random.default_rng(0)

# (file, prefix, mode) — mode='ret' for prices, 'diff' for levels (TNX yield, VIX index)
INSTRUMENTS = [
    ("eurusd_1h.parquet", "eur", "ret"),
    ("spx_1h.parquet",    "spx", "ret"),
    ("tnx_1h.parquet",    "tnx", "diff"),
    ("vix_1h.parquet",    "vix", "diff"),
    ("uup_1h.parquet",    "uup", "ret"),
]


def build_inst_features(path: str, prefix: str, mode: str) -> pd.DataFrame:
    df = pd.read_parquet(os.path.join(DATA, path)).sort_values("time").reset_index(drop=True)
    if mode == "ret":
        df[f"{prefix}_ret_1h"]  = df["close"].pct_change(1)
        df[f"{prefix}_ret_4h"]  = df["close"].pct_change(4)
        df[f"{prefix}_ret_24h"] = df["close"].pct_change(24)
    else:
        df[f"{prefix}_ret_1h"]  = df["close"].diff(1)
        df[f"{prefix}_ret_4h"]  = df["close"].diff(4)
        df[f"{prefix}_ret_24h"] = df["close"].diff(24)
    # z-score 24h return vs trailing 30 days (~720 1h bars) of 24h returns
    roll_std = df[f"{prefix}_ret_24h"].rolling(720, min_periods=120).std()
    df[f"{prefix}_ret_24h_z"] = df[f"{prefix}_ret_24h"] / roll_std
    cols = ["time", f"{prefix}_ret_1h", f"{prefix}_ret_4h",
            f"{prefix}_ret_24h", f"{prefix}_ret_24h_z"]
    return df[cols].dropna().reset_index(drop=True)


def attach_intraday(trades: pd.DataFrame) -> pd.DataFrame:
    out = trades.sort_values("time").reset_index(drop=True).copy()
    feature_cols = []
    for path, prefix, mode in INSTRUMENTS:
        feats = build_inst_features(path, prefix, mode)
        # Align: for each trade, take the most recent instrument bar
        # whose time ≤ trade time (asof merge backward — no lookahead).
        out = pd.merge_asof(out, feats, on="time", direction="backward")
        feature_cols.extend([c for c in feats.columns if c != "time"])
    return out, feature_cols


def pf(rs):
    pos = rs[rs > 0].sum(); neg = -rs[rs <= 0].sum()
    return pos / max(neg, 1e-9)


def perm_pf(rs, n_keep, n_iter=5000):
    out = np.empty(n_iter)
    idx = np.arange(len(rs))
    for i in range(n_iter):
        c = RNG.choice(idx, n_keep, replace=False)
        out[i] = pf(rs[c])
    return out


def evaluate(name, trades, p, base_rs):
    rs = trades["pnl_R"].values
    base_pf = pf(rs); base_r = rs.sum()
    corr, _ = pearsonr(p, rs)
    print(f"\n[{name}] n={len(rs)} baseline PF={base_pf:.3f} R={base_r:+.1f}")
    print(f"  pearson(p, pnl_R) = {corr:+.4f}")
    rows = []
    for thr in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        keep = p > thr
        if keep.sum() < 30: continue
        gate_pf = pf(rs[keep])
        null = perm_pf(rs, int(keep.sum()))
        rows.append({
            "thr": thr, "kept": int(keep.sum()),
            "gate_PF": round(gate_pf, 3),
            "null_mean": round(null.mean(), 3),
            "null_95": round(np.percentile(null, 95), 3),
            "p_value": round((null >= gate_pf).mean(), 4),
            "kept_R": round(rs[keep].sum(), 1),
            "ΔR": round(rs[keep].sum() - base_r, 1),
        })
    df_out = pd.DataFrame(rows)
    print(df_out.to_string(index=False))
    df_out.to_csv(os.path.join(REPS, f"gate_{name}.csv"), index=False)

    # Sizing: A_linear  lot = 0.5 + 1.0 * p
    lots = 0.5 + 1.0 * p
    weighted = rs * lots
    sizing = {
        "scheme":      ["baseline_1.0", "A_linear"],
        "R":           [round(base_r, 1), round(weighted.sum(), 1)],
        "PF":          [round(base_pf, 3), round(pf(weighted), 3)],
        "avg_lot":     [1.0, round(lots.mean(), 3)],
        "R_per_lot":   [round(base_r, 3),
                         round(weighted.sum() / lots.sum(), 3)],
    }
    print(pd.DataFrame(sizing).to_string(index=False))


def main():
    midas = pd.read_csv(os.path.join(ROOT, "data/v6_trades_holdout_xau.csv"),
                         parse_dates=["time"])
    midas, fcols = attach_intraday(midas)
    midas = midas.dropna(subset=fcols).reset_index(drop=True)
    print(f"Midas after intraday merge: {len(midas)} trades, {len(fcols)} features")

    train = midas[midas["time"] < HOLDOUT_START]
    test  = midas[midas["time"] >= HOLDOUT_START]
    print(f"  train: {len(train)}  holdout: {len(test)}  "
          f"(train win rate {(train.pnl_R>0).mean():.3f})")

    sc = StandardScaler().fit(train[fcols].values)
    clf = LogisticRegression(C=1.0, max_iter=2000, random_state=0)
    clf.fit(sc.transform(train[fcols].values),
            (train["pnl_R"] > 0).astype(int).values)

    coef_df = pd.DataFrame({"feature": fcols,
                             "coef":    clf.coef_[0]}).sort_values(
        "coef", key=abs, ascending=False)
    print("\nTop 10 coefficients:")
    print(coef_df.head(10).to_string(index=False))
    coef_df.to_csv(os.path.join(REPS, "coefficients.csv"), index=False)

    # In-sample sanity
    in_p = clf.predict_proba(sc.transform(train[fcols].values))[:, 1]
    print(f"\n  in-sample corr(p, pnl_R) = {pearsonr(in_p, train['pnl_R'].values)[0]:+.4f}")

    # Holdout: Midas
    p_m = clf.predict_proba(sc.transform(test[fcols].values))[:, 1]
    evaluate("midas", test, p_m, None)

    # Holdout: Oracle
    oracle = pd.read_csv(os.path.join(ROOT, "data/v72l_trades_holdout.csv"),
                          parse_dates=["time"])
    oracle, _ = attach_intraday(oracle)
    oracle = oracle.dropna(subset=fcols).reset_index(drop=True)
    oracle_h = oracle[oracle["time"] >= HOLDOUT_START]
    p_o = clf.predict_proba(sc.transform(oracle_h[fcols].values))[:, 1]
    evaluate("oracle", oracle_h, p_o, None)


if __name__ == "__main__":
    main()
