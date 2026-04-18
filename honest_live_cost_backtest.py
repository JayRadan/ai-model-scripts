"""
Honest re-simulation of the holdout under LIVE TP/SL/spread geometry.

For each instrument we keep the already-trained per-rule classifiers and
their thresholds intact. For every holdout setup that passes confirmation,
we scan the forward OHLC bars and ask: which hits first, TP or SL, under
the ACTUAL geometry the live EA is using?

Reports, per instrument:
    - "backtest" config (TP=2, SL=1, matching training labels)
    - "live"     config (TP=6, SL=2 gold/gj, SL=1.5 eu) with realistic spread

All metrics are in ATR-R units so instruments are comparable.
"""
from __future__ import annotations
import json, os, glob, sys
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

ROOT = "/home/jay/Desktop/new-model-zigzag"
MAX_FWD = 80  # bars forward to wait for TP/SL (M5 ≈ 6.7h)

XAU_FEATS = [
    "f01_CPR","f02_WickAsym","f03_BEF","f04_TCS","f05_SPI",
    "f06_LRSlope","f07_RECR","f08_SCM","f09_HLER","f10_EP",
    "f11_KE","f12_MCS","f13_Work","f14_EDR","f15_AI",
    "f16_PPShigh","f16_PPSlow","f17_SCR","f18_RVD","f19_WBER","f20_NCDE",
    "rsi14","rsi6","stoch_k","stoch_d","bb_pct",
    "mom5","mom10","mom20","ll_dist10","hh_dist10",
    "vol_accel","atr_ratio","spread_norm","hour_enc","dow_enc",
    "h1_trend_sma20","h1_trend_sma50","h1_slope5","h1_rsi14",
    "h1_atr_ratio","h1_dist_sma20","h1_dist_sma50",
    "h4_trend_sma20","h4_trend_sma50","h4_slope5","h4_rsi14",
    "h4_atr_ratio","h4_dist_sma20","h4_dist_sma50",
]
EUGJ_FEATS = [
    "f01_CPR","f02_WickAsym","f03_BEF","f04_TCS","f05_SPI",
    "f06_LRSlope","f07_RECR","f08_SCM","f09_HLER","f10_EP",
    "f11_KE","f12_MCS","f13_Work","f14_EDR","f15_AI",
    "f16_PPShigh","f16_PPSlow","f17_SCR","f18_RVD","f19_WBER","f20_NCDE",
    "stoch_k","rsi14","bb_pct","vol_ratio","range_atr",
    "dist_sma20","dist_sma50","body_ratio","consec_dir",
    "hour_sin","hour_cos","dow_sin","dow_cos",
]

INSTRUMENTS = {
    "XAUUSD": {
        "swing":    f"{ROOT}/data/swing_v5_xauusd.csv",
        "models":   f"{ROOT}/models",
        "setup_per_cluster": f"{ROOT}/data/setups_{{cid}}.csv",
        "clusters": [0,1,2,3,4],
        "feats":    XAU_FEATS,
        "live_TP": 6.0, "live_SL": 2.0,
        "live_spread_R": 0.15,  # ~0.45 USD on ~3 USD ATR
    },
    "EURUSD": {
        "swing":    f"{ROOT}/eurusd/data/swing_v5_eurusd.csv",
        "models":   f"{ROOT}/eurusd/models",
        "setup_combined": f"{ROOT}/eurusd/data/setup_signals_eurusd.csv",
        "feats":    EUGJ_FEATS,
        "live_TP": 6.0, "live_SL": 1.5,
        "live_spread_R": 0.25,  # ~1.5pip on ~6pip ATR(14) M5
    },
    "GBPJPY": {
        "swing":    f"{ROOT}/gbpjpy/data/swing_v5_gbpjpy.csv",
        "models":   f"{ROOT}/gbpjpy/models",
        "setup_combined": f"{ROOT}/gbpjpy/data/setup_signals_gbpjpy.csv",
        "feats":    EUGJ_FEATS,
        "live_TP": 6.0, "live_SL": 2.0,
        "live_spread_R": 0.30,  # ~2.5pip on ~8pip ATR(14) M5
    },
}


def compute_atr(df: pd.DataFrame, n: int = 14) -> np.ndarray:
    """True-range ATR over n bars. df has open/high/low/close."""
    h, l, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    # simple rolling mean
    s = pd.Series(tr).rolling(n, min_periods=1).mean().to_numpy()
    return s


def simulate_trade(high, low, close, entry_idx, entry_price, direction,
                   atr, tp_mult, sl_mult, max_fwd):
    """Forward-scan: returns pnl in R-units (tp_mult on win, -sl_mult on loss,
    or realized close/atr on expiry). Sign applied for short."""
    tp = entry_price + direction * tp_mult * atr
    sl = entry_price - direction * sl_mult * atr
    end = min(entry_idx + 1 + max_fwd, len(close))
    for i in range(entry_idx + 1, end):
        hi, lo = high[i], low[i]
        if direction == 1:
            hit_sl = lo <= sl
            hit_tp = hi >= tp
        else:
            hit_sl = hi >= sl
            hit_tp = lo <= tp
        if hit_sl and hit_tp:
            # ambiguous within same bar → assume SL hit first (conservative)
            return -sl_mult, "sl"
        if hit_sl:
            return -sl_mult, "sl"
        if hit_tp:
            return +tp_mult, "tp"
    # expiry
    realized = direction * (close[end - 1] - entry_price) / atr
    return realized, "exp"


def get_confirmed_holdout(rule_df, feats, model_path, meta_path):
    """Replay last 20% through classifier, return (test_subset, mask)."""
    if not (os.path.exists(model_path) and os.path.exists(meta_path)):
        return None, None
    with open(meta_path) as f:
        meta = json.load(f)
    if meta.get("disabled", False):
        return None, None
    thr = meta["threshold"]
    feat_cols = meta.get("feature_cols", feats)
    split = int(len(rule_df) * 0.8)
    test = rule_df.iloc[split:].copy()
    if len(test) < 5:
        return None, None
    missing = [c for c in feat_cols if c not in test.columns]
    if missing:
        return None, None
    model = XGBClassifier()
    model.load_model(model_path)
    X = test[feat_cols].fillna(0).values
    proba = model.predict_proba(X)[:, 1]
    test = test[proba >= thr].copy()
    return test, len(test)


def run_instrument(name: str, cfg: dict):
    print(f"\n{'='*70}\n{name}\n{'='*70}")
    swing = pd.read_csv(cfg["swing"], parse_dates=["time"])
    swing = swing.sort_values("time").reset_index(drop=True)
    swing["atr14"] = compute_atr(swing, 14)
    H = swing["high"].to_numpy()
    L = swing["low"].to_numpy()
    C = swing["close"].to_numpy()
    time_to_idx = pd.Series(swing.index.values, index=swing["time"].values)

    # collect rule/cluster→dataframe pairs
    rule_pools = []  # each: (cid, rule, df_slice, feat_cols_used)
    if "setup_per_cluster" in cfg:
        for cid in cfg["clusters"]:
            p = cfg["setup_per_cluster"].format(cid=cid)
            if not os.path.exists(p): continue
            df = pd.read_csv(p, parse_dates=["time"])
            if "rule" not in df.columns: continue
            for rname, grp in df.groupby("rule"):
                rule_pools.append((cid, rname, grp.sort_values("time").reset_index(drop=True)))
    else:
        df = pd.read_csv(cfg["setup_combined"], parse_dates=["time"])
        for (cid, rname), grp in df.groupby(["cluster","rule"]):
            rule_pools.append((int(cid), rname,
                               grp.sort_values("time").reset_index(drop=True)))

    configs = {
        "backtest(TP2,SL1)": dict(TP=2.0, SL=1.0, spread=0.0),
        "live(TP6,SL_live)": dict(TP=cfg["live_TP"], SL=cfg["live_SL"],
                                   spread=cfg["live_spread_R"]),
    }
    results = {k: [] for k in configs}

    for cid, rname, rdf in rule_pools:
        mpath = f"{cfg['models']}/confirm_c{cid}_{rname}.json"
        mmeta = f"{cfg['models']}/confirm_c{cid}_{rname}_meta.json"
        sub, n = get_confirmed_holdout(rdf, cfg["feats"], mpath, mmeta)
        if sub is None or n == 0:
            continue
        for _, row in sub.iterrows():
            t = row["time"]
            if t not in time_to_idx.index: continue
            idx = int(time_to_idx[t])
            if idx+1 >= len(C): continue
            atr = swing["atr14"].iat[idx]
            if not np.isfinite(atr) or atr <= 0: continue
            entry = C[idx]
            d = row["direction"]
            if isinstance(d, str):
                direction = 1 if d.lower().startswith("b") or d == "1" else -1
            else:
                direction = 1 if int(d) > 0 else -1
            for label, c in configs.items():
                pnl, how = simulate_trade(H, L, C, idx, entry, direction,
                                           atr, c["TP"], c["SL"], MAX_FWD)
                results[label].append(pnl - c["spread"])

    print(f"  confirmed holdout trades: {len(results['live(TP6,SL_live)'])}")
    for label, pnls in results.items():
        a = np.array(pnls, dtype=float)
        if len(a) == 0:
            print(f"  {label}: no trades"); continue
        wins = a[a > 0]; losses = a[a < 0]
        wr = float((a > 0).mean())
        pf = (wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf")
        exp_R = a.mean()
        print(f"  {label:22s}  n={len(a):4d}  WR={wr:.1%}  "
              f"PF={pf:.2f}  E[R]={exp_R:+.3f}  "
              f"sum_R={a.sum():+.1f}")


if __name__ == "__main__":
    for name, cfg in INSTRUMENTS.items():
        try:
            run_instrument(name, cfg)
        except Exception as e:
            print(f"{name} failed: {e}")
            import traceback; traceback.print_exc()
