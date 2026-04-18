"""
REGIME MODEL BACKTEST
=====================
Simulates real trading on UNSEEN data.
Uses the last 20% of each cluster as holdout — model never saw this data.
Calculates: PnL, winrate, profit factor, max drawdown, sharpe.

Run: python backtest_regimes.py
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import json
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

BASE_FEATURES = [
    "f01_CPR","f02_WickAsym","f03_BEF","f04_TCS","f05_SPI",
    "f06_LRSlope","f07_RECR","f08_SCM","f09_HLER","f10_EP",
    "f11_KE","f12_MCS","f13_Work","f14_EDR","f15_AI",
    "f16_PPShigh","f16_PPSlow","f17_SCR","f18_RVD","f19_WBER","f20_NCDE",
]
TECH_FEATURES = [
    "rsi14","rsi6","stoch_k","stoch_d","bb_pct",
    "mom5","mom10","mom20",
    "ll_dist10","hh_dist10",
    "vol_accel","atr_ratio","spread_norm",
    "hour_enc","dow_enc",
]
FEATURE_COLS = BASE_FEATURES + TECH_FEATURES  # 36 total

# v3 label encoding
BUY_LABEL  = 0
FLAT_LABEL = 1
SELL_LABEL = 2

CLUSTER_NAMES = {0:"Ranging", 1:"Downtrend", 2:"Shock_News", 3:"Uptrend"}
CLUSTER_IDS   = [0, 1, 2, 3]
SKIP_CLUSTERS = set()  # C2 now has a real model too
COLORS = ["#FFD700","#00E5FF","#FF6B6B","#69FF94"]

# Empty string = deployment models. "_honest" = models trained on first 80% only.
MODEL_SUFFIX = os.environ.get("BT_SUFFIX", "")

SPREAD_POINTS = 40       # your spread from the data
POINT_VALUE   = 0.01     # gold: 1 point = $0.01 per 0.01 lot
LOT_SIZE      = 0.01     # micro lot

# Exit rule MUST match the labeler that trained the models being backtested.
# v4 labeler: TP=1.2×ATR, SL=0.8×ATR, MAX_FWD=10  (shorter-horizon retry)
# v3 labeler: TP=2.0×ATR, SL=1.0×ATR, MAX_FWD=40  (original)
TP_MULT = 1.2
SL_MULT = 0.8
MAX_FWD = 10

# Per-cluster backtest params — found via sweep_thresholds.py on calibrated
# (no scale_pos_weight) models. C2 uses C0's threshold as a starting point
# since both are 3-class; tune after first backtest.
CLUSTER_PARAMS = {
    0: {"threshold": 0.40},
    1: {"threshold": 0.40},
    2: {"threshold": 0.40},
    3: {"threshold": 0.35},
}

# ── Load model ────────────────────────────────────────────────────────────────
def load_model(cluster_id):
    name = CLUSTER_NAMES[cluster_id]
    path = f"models/regime_{cluster_id}_{name}{MODEL_SUFFIX}.json"
    model = xgb.XGBClassifier()
    model.load_model(path)

    meta_path = f"models/regime_{cluster_id}_{name}{MODEL_SUFFIX}_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)

    return model, meta

def compute_atr(high, low, close, period=14):
    """Simple-mean ATR matching the v3 labeler."""
    prev_close = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low  - prev_close),
    ])
    atr = pd.Series(tr).rolling(period, min_periods=1).mean().to_numpy()
    return np.clip(atr, 1e-10, None)


# ── Backtest engine — probability threshold + ATR TP/SL exit ─────────────────
def backtest(df, model, feat_cols, label_classes, spread=40, threshold=0.60):
    """
    For each bar:
      - Run predict_proba
      - If actionable-class prob > threshold, enter trade
      - Exit on first of: TP hit (2×ATR), SL hit (1×ATR), or timeout (40 bars)
      - Block overlapping entries (no position stacking)
    """
    feat_cols_avail = [c for c in feat_cols if c in df.columns]
    X = df[feat_cols_avail].fillna(0).values
    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    times  = df["time"].values

    atr = compute_atr(highs, lows, closes, period=14)
    spread_pts = spread * 0.01

    probs = model.predict_proba(X)
    classes = list(label_classes)
    actionable_idx = [(col, c) for col, c in enumerate(classes) if c != FLAT_LABEL]

    trades = []
    i = 0
    n = len(X)
    while i < n - MAX_FWD:
        best_prob = 0.0
        best_label = None
        for col_idx, label in actionable_idx:
            p = probs[i, col_idx]
            if p > best_prob:
                best_prob = p
                best_label = label

        if best_label is None or best_prob < threshold:
            i += 1
            continue

        direction = +1 if best_label == BUY_LABEL else -1  # +1 long, -1 short
        entry_price = closes[i]
        entry_atr   = atr[i]

        if direction == +1:
            tp = entry_price + TP_MULT * entry_atr
            sl = entry_price - SL_MULT * entry_atr
        else:
            tp = entry_price - TP_MULT * entry_atr
            sl = entry_price + SL_MULT * entry_atr

        # Walk forward up to MAX_FWD bars, find first TP/SL hit
        exit_i = i + MAX_FWD  # default: timeout
        exit_price = closes[exit_i]
        exit_reason = "timeout"

        for k in range(1, MAX_FWD + 1):
            bar = i + k
            if direction == +1:
                hit_sl = lows[bar]  <= sl
                hit_tp = highs[bar] >= tp
            else:
                hit_sl = highs[bar] >= sl
                hit_tp = lows[bar]  <= tp

            if hit_sl and hit_tp:
                # Same-bar ambiguity — assume SL first (conservative, matches labeler)
                exit_i = bar
                exit_price = sl
                exit_reason = "sl"
                break
            if hit_tp:
                exit_i = bar
                exit_price = tp
                exit_reason = "tp"
                break
            if hit_sl:
                exit_i = bar
                exit_price = sl
                exit_reason = "sl"
                break

        if direction == +1:
            pnl_points = (exit_price - entry_price) - spread_pts
        else:
            pnl_points = (entry_price - exit_price) - spread_pts

        trades.append({
            "entry_time":  times[i],
            "exit_time":   times[exit_i],
            "signal":      direction,
            "prob":        float(best_prob),
            "entry_price": entry_price,
            "exit_price":  exit_price,
            "pnl_points":  pnl_points,
            "pnl_usd":     pnl_points * POINT_VALUE / LOT_SIZE,
            "candles_held": exit_i - i,
            "exit_reason": exit_reason,
        })

        # Block overlapping entries
        i = exit_i + 1

    return pd.DataFrame(trades)

# ── Performance metrics ───────────────────────────────────────────────────────
def calc_metrics(trades_df):
    if len(trades_df) == 0:
        return {}

    pnl = trades_df["pnl_points"].values
    cum = np.cumsum(pnl)

    wins  = pnl[pnl > 0]
    loses = pnl[pnl < 0]

    # Drawdown
    peak = np.maximum.accumulate(cum)
    dd   = cum - peak
    max_dd = float(dd.min())

    # Sharpe (annualized, assume 288 M5 candles/day)
    if pnl.std() > 0:
        sharpe = (pnl.mean() / pnl.std()) * np.sqrt(288 * 252)
    else:
        sharpe = 0.0

    return {
        "n_trades":       len(trades_df),
        "winrate":        float(len(wins) / len(pnl)) if len(pnl) > 0 else 0,
        "total_pnl_pts":  float(pnl.sum()),
        "avg_win_pts":    float(wins.mean()) if len(wins) > 0 else 0,
        "avg_loss_pts":   float(loses.mean()) if len(loses) > 0 else 0,
        "profit_factor":  float(wins.sum() / abs(loses.sum())) if len(loses) > 0 and loses.sum() != 0 else 999,
        "max_drawdown_pts": max_dd,
        "sharpe":         sharpe,
        "expectancy_pts": float(pnl.mean()),
    }

# ── Plot equity curves ────────────────────────────────────────────────────────
def plot_results(all_trades, all_metrics):
    fig = plt.figure(figsize=(18, 10), facecolor="#080c12")
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)

    for panel_idx, cluster_id in enumerate(CLUSTER_IDS):
        trades_df = all_trades[panel_idx]
        metrics   = all_metrics[panel_idx]
        color = COLORS[panel_idx]
        name  = CLUSTER_NAMES[cluster_id]

        ax = fig.add_subplot(gs[0, panel_idx])
        ax.set_facecolor("#0d1117")

        if cluster_id in SKIP_CLUSTERS:
            ax.text(0.5, 0.5, "SKIPPED\n(never trade)", ha="center", va="center",
                    color="#666", fontsize=10, fontfamily="monospace", transform=ax.transAxes)
            ax.set_title(f"— C{cluster_id} {name}", color="#666",
                         fontsize=9, fontfamily="monospace", pad=6)
            ax.tick_params(colors="#5a7080", labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor("#1e2a3a")
            continue

        if len(trades_df) > 0:
            cum_pnl = trades_df["pnl_points"].cumsum().values
            ax.plot(cum_pnl, color=color, linewidth=1.5)
            ax.fill_between(range(len(cum_pnl)), cum_pnl, 0,
                           alpha=0.15, color=color)
            ax.axhline(0, color="#333", linewidth=0.5)

        pf  = metrics.get("profit_factor", 0)
        wr  = metrics.get("winrate", 0)
        pnl = metrics.get("total_pnl_pts", 0)
        status = "✅" if pnl > 0 and pf > 1.2 else "❌"

        ax.set_title(f"{status} C{cluster_id} {name}\nPF:{pf:.2f} WR:{wr:.0%} PnL:{pnl:+.0f}pts",
                    color=color, fontsize=9, fontfamily="monospace", pad=6)
        ax.tick_params(colors="#5a7080", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1e2a3a")

    # Combined equity
    ax_all = fig.add_subplot(gs[1, :])
    ax_all.set_facecolor("#0d1117")
    ax_all.set_title("COMBINED EQUITY (all 4 regime models)", color="#FFD700",
                     fontsize=12, fontfamily="monospace", pad=8)

    all_pnl = pd.concat([t[["entry_time","pnl_points"]] for t in all_trades if len(t) > 0])
    all_pnl = all_pnl.sort_values("entry_time")
    cum_all  = all_pnl["pnl_points"].cumsum().values

    ax_all.plot(cum_all, color="#FFD700", linewidth=1.5)
    ax_all.fill_between(range(len(cum_all)), cum_all, 0, alpha=0.1, color="#FFD700")
    ax_all.axhline(0, color="#333", linewidth=0.5)
    ax_all.tick_params(colors="#5a7080", labelsize=8)
    for spine in ax_all.spines.values():
        spine.set_edgecolor("#1e2a3a")

    plt.suptitle("EDGEPREDICTOR — REGIME MODEL BACKTEST (20% holdout)",
                color="#FFD700", fontsize=14, fontfamily="monospace")
    plt.savefig("backtest_results.png", dpi=150, bbox_inches="tight", facecolor="#080c12")
    print("Saved: backtest_results.png")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    all_trades  = []
    all_metrics = []

    print(f"\n{'═'*55}")
    print("REGIME MODEL BACKTEST — 20% holdout per cluster")
    print(f"{'═'*55}")

    for cluster_id in CLUSTER_IDS:
        name = CLUSTER_NAMES[cluster_id]

        if cluster_id in SKIP_CLUSTERS:
            print(f"\nC{cluster_id} {name} — SKIPPED (never trade)")
            all_trades.append(pd.DataFrame())
            all_metrics.append({})
            continue

        df   = pd.read_csv(f"cluster_{cluster_id}_data.csv", parse_dates=["time"])
        df   = df.sort_values("time").reset_index(drop=True)

        # STRICT holdout — last 20% only, never seen during training
        holdout_start = int(len(df) * 0.80)
        holdout = df.iloc[holdout_start:].copy()

        model, meta = load_model(cluster_id)
        feat_cols    = meta["feature_cols"]
        label_classes = meta["label_classes"]

        cp = CLUSTER_PARAMS[cluster_id]
        print(f"\nC{cluster_id} {name}")
        print(f"  Holdout rows: {len(holdout):,}")
        print(f"  Params: threshold={cp['threshold']}  TP={TP_MULT}xATR SL={SL_MULT}xATR fwd={MAX_FWD}")

        trades  = backtest(holdout, model, feat_cols, label_classes,
                           threshold=cp["threshold"])
        metrics = calc_metrics(trades)

        all_trades.append(trades)
        all_metrics.append(metrics)

        if metrics:
            print(f"  Trades:        {metrics['n_trades']}")
            print(f"  Win rate:      {metrics['winrate']:.1%}")
            print(f"  Profit factor: {metrics['profit_factor']:.2f}")
            print(f"  Total PnL:     {metrics['total_pnl_pts']:+.1f} pts")
            print(f"  Max drawdown:  {metrics['max_drawdown_pts']:.1f} pts")
            print(f"  Sharpe:        {metrics['sharpe']:.2f}")
            print(f"  Expectancy:    {metrics['expectancy_pts']:+.3f} pts/trade")

    # Summary
    print(f"\n{'═'*55}")
    print("SUMMARY — is it worth going live?")
    print(f"{'═'*55}")
    for panel_idx, cluster_id in enumerate(CLUSTER_IDS):
        if cluster_id in SKIP_CLUSTERS:
            print(f"  — C{cluster_id} {CLUSTER_NAMES[cluster_id]:<15} SKIPPED (never trade)")
            continue
        metrics = all_metrics[panel_idx]
        pf   = metrics.get("profit_factor", 0)
        pnl  = metrics.get("total_pnl_pts", 0)
        good = pf > 1.2 and pnl > 0
        print(f"  {'✅' if good else '❌'} C{cluster_id} {CLUSTER_NAMES[cluster_id]:<15} PF:{pf:.2f}  PnL:{pnl:+.0f}pts  {'→ USE' if good else '→ SKIP or retrain'}")

    print(f"\n  Minimum bar to go live:")
    print(f"  - Profit factor > 1.2")
    print(f"  - Positive total PnL on holdout")
    print(f"  - Max drawdown < 200 pts (~$2 per micro lot)")

    plot_results(all_trades, all_metrics)

if __name__ == "__main__":
    main()
