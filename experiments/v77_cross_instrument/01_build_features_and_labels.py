"""
Build per-date cross-instrument features + per-date label
("good day for trading?") from v6 Midas holdout trades.

Why Midas trades for the label:
  Midas holdout starts 2024-01-02 (vs Oracle 2024-12-12), so we get a
  chunk of 2024 (Jan→Dec 11) to TRAIN the regime classifier, then
  hold out 2024-12-12 → 2026-04-13 to evaluate the gate on both
  Midas AND Oracle (Oracle's full holdout sits inside that window).

Features (all from prior day's daily close — strictly no lookahead):
  DXY: ret_1d, ret_5d, ret_20d, dist_from_sma50_z
  SPX: ret_1d, ret_5d, ret_20d, dist_from_sma50_z
  TNX: level, chg_1d, chg_5d, level_z60
  VIX: level, chg_1d, level_z60
  20-day rolling corr(XAU_daily_ret, {DXY,SPX,VIX}_ret)

Label: good_day = (sum of pnl_R on date > 0)

Output: data/features_labels.parquet — one row per trading date with
features + label + train/holdout split flag.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

ROOT = "/home/jay/Desktop/new-model-zigzag"
DATA = os.path.join(os.path.dirname(__file__), "data")

HOLDOUT_START = pd.Timestamp("2024-12-12")


def load_daily(symbol_file: str) -> pd.DataFrame:
    df = pd.read_parquet(os.path.join(DATA, symbol_file))
    df["date"] = pd.to_datetime(df["time"]).dt.normalize()
    return df.sort_values("date").reset_index(drop=True)


def add_returns_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df = df.copy()
    df[f"{prefix}_ret_1d"]  = df["close"].pct_change(1)
    df[f"{prefix}_ret_5d"]  = df["close"].pct_change(5)
    df[f"{prefix}_ret_20d"] = df["close"].pct_change(20)
    sma50 = df["close"].rolling(50).mean()
    sd50  = df["close"].rolling(50).std()
    df[f"{prefix}_dist_sma50_z"] = (df["close"] - sma50) / sd50
    return df[["date",
               f"{prefix}_ret_1d", f"{prefix}_ret_5d", f"{prefix}_ret_20d",
               f"{prefix}_dist_sma50_z"]]


def add_level_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df = df.copy()
    df[f"{prefix}_level"]    = df["close"]
    df[f"{prefix}_chg_1d"]   = df["close"].diff(1)
    df[f"{prefix}_chg_5d"]   = df["close"].diff(5)
    m60 = df["close"].rolling(60).mean(); s60 = df["close"].rolling(60).std()
    df[f"{prefix}_level_z60"] = (df["close"] - m60) / s60
    return df[["date",
               f"{prefix}_level", f"{prefix}_chg_1d",
               f"{prefix}_chg_5d", f"{prefix}_level_z60"]]


def main() -> None:
    print("loading cross-instrument daily…")
    dxy = add_returns_features(load_daily("dxy_1d.parquet"), "dxy")
    spx = add_returns_features(load_daily("spx_1d.parquet"), "spx")
    tnx = add_level_features(load_daily("tnx_1d.parquet"),   "tnx")
    vix = add_level_features(load_daily("vix_1d.parquet"),   "vix")
    # vix only needs level + 1d change + z (drop chg_5d to keep features tight)
    vix = vix.drop(columns=["vix_chg_5d"])

    feats = dxy.merge(spx, on="date", how="outer") \
               .merge(tnx, on="date", how="outer") \
               .merge(vix, on="date", how="outer") \
               .sort_values("date").reset_index(drop=True)

    # XAU daily for correlation features. Resample swing to daily.
    print("loading XAU M5, resampling to daily…")
    xau = pd.read_csv(os.path.join(ROOT, "data/swing_v5_xauusd.csv"),
                       parse_dates=["time"], usecols=["time", "close"])
    xau["date"] = xau["time"].dt.normalize()
    xau_daily = xau.groupby("date")["close"].last().reset_index()
    xau_daily["xau_ret_1d"] = xau_daily["close"].pct_change(1)

    # Roll 20d correlations of xau daily ret vs cross-instrument 1d returns.
    merged_for_corr = xau_daily.merge(feats[["date", "dxy_ret_1d",
                                              "spx_ret_1d", "vix_chg_1d"]],
                                       on="date", how="left")
    for col, name in [("dxy_ret_1d", "corr_xau_dxy_20d"),
                       ("spx_ret_1d", "corr_xau_spx_20d"),
                       ("vix_chg_1d", "corr_xau_vix_20d")]:
        merged_for_corr[name] = (merged_for_corr["xau_ret_1d"]
                                  .rolling(20).corr(merged_for_corr[col]))
    corr_df = merged_for_corr[["date", "corr_xau_dxy_20d",
                                "corr_xau_spx_20d", "corr_xau_vix_20d"]]

    feats = feats.merge(corr_df, on="date", how="left")

    # CRITICAL: use prior day's close to avoid lookahead. Shift all features
    # by 1 — the row for date D contains features known at D's open
    # (i.e. features through D-1 close).
    feature_cols = [c for c in feats.columns if c != "date"]
    feats[feature_cols] = feats[feature_cols].shift(1)

    # Reindex to ALL calendar days and forward-fill. XAU trades 24/5
    # including Sundays, but DXY/SPX/etc only Mon-Fri. Without this we'd
    # drop every Sunday (and any holiday) from the merge.
    full_idx = pd.date_range(feats["date"].min(), feats["date"].max(),
                              freq="D")
    feats = (feats.set_index("date").reindex(full_idx).ffill()
                  .rename_axis("date").reset_index())

    # ---- LABELS from Midas v6 trades ----
    print("loading Midas v6 holdout trades for labels…")
    trades = pd.read_csv(os.path.join(ROOT, "data/v6_trades_holdout_xau.csv"),
                          parse_dates=["time"])
    trades["date"] = trades["time"].dt.normalize()
    daily = trades.groupby("date").agg(
        n_trades=("pnl_R", "size"),
        sum_R=("pnl_R", "sum"),
    ).reset_index()
    daily["label"] = (daily["sum_R"] > 0).astype(int)

    # Merge labels onto features
    df = feats.merge(daily, on="date", how="inner")
    df["split"] = np.where(df["date"] < HOLDOUT_START, "train", "holdout")

    # Drop rows with any NaN in feature columns (early window)
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    out = os.path.join(DATA, "features_labels.parquet")
    df.to_parquet(out, index=False)

    n_tr = (df["split"] == "train").sum()
    n_ho = (df["split"] == "holdout").sum()
    print(f"\n→ {out}")
    print(f"  rows: {len(df):,}  ({n_tr} train, {n_ho} holdout)")
    print(f"  train label balance: "
          f"{df[df.split=='train']['label'].mean():.3f}")
    print(f"  holdout label balance: "
          f"{df[df.split=='holdout']['label'].mean():.3f}")
    print(f"  features: {len(feature_cols)} → {feature_cols}")


if __name__ == "__main__":
    main()
