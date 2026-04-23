"""
Post-process v72l_trades_holdout_btc.csv into website backtest assets:
  - public/btc_backtest.png       (equity curve image)
  - public/backtest_data.json     (add/update 'btc' entry)

Mirrors the schema of midas/oracle so the /backtest page renders identically.

No retraining — just reads the already-validated holdout trades and the swing
CSV (for ATR lookup to convert pnl_R → pnl_usd).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "/home/jay/Desktop/new-model-zigzag/model_pipeline")
import paths as P

TRADES_CSV = P.data("v72l_trades_holdout_btc.csv")
SWING_CSV  = P.data("swing_v5_btc.csv")
WEB_DIR    = Path("/home/jay/Desktop/my-agents-and-website/commercial/website/public")
OUT_PNG    = WEB_DIR / "btc_backtest.png"
OUT_JSON   = WEB_DIR / "backtest_data.json"

# BTC at 0.01 lot: $1 price move = $0.01 per position-lot.
LOT_FACTOR = 0.01
SL_HARD    = 4.0      # matches deployed EA (hard SL)
ATR_PERIOD = 14

CLUSTER_NAMES = {0:"Uptrend", 1:"MeanRevert", 2:"TrendRange", 3:"Downtrend", 4:"HighVol"}


def compute_atr(df, period=14):
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    prev_c = np.roll(c, 1); prev_c[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    return pd.Series(tr).rolling(period, min_periods=period).mean().values


def main():
    print(f"Loading trades: {TRADES_CSV}")
    trades = pd.read_csv(TRADES_CSV, parse_dates=["time"])
    trades = trades.sort_values("time").reset_index(drop=True)
    print(f"  {len(trades):,} trades  {trades['time'].iat[0]} → {trades['time'].iat[-1]}")

    print(f"Loading swing: {SWING_CSV}")
    swing = pd.read_csv(SWING_CSV, parse_dates=["time"])
    swing = swing.sort_values("time").reset_index(drop=True)
    atr = compute_atr(swing, ATR_PERIOD)

    # Join ATR to each trade
    time_to_idx = pd.Series(swing.index.values, index=swing["time"].values)
    trades["atr"] = trades["time"].map(lambda t: atr[time_to_idx[t]] if t in time_to_idx.index else np.nan)
    trades = trades.dropna(subset=["atr"]).reset_index(drop=True)

    # pnl_R → pnl_usd  (each 1R = sl_dist dollars per whole coin; 0.01 lot = $0.01/$1 move)
    trades["sl_dist_usd"] = SL_HARD * trades["atr"]
    trades["pnl_usd"]     = trades["pnl_R"] * trades["sl_dist_usd"] * LOT_FACTOR

    # ── Aggregate stats ──
    wins  = trades[trades["pnl_usd"] > 0]
    losses = trades[trades["pnl_usd"] <= 0]
    wr = 100.0 * len(wins) / max(len(trades), 1)
    total_pnl = float(trades["pnl_usd"].sum())
    gross_win = float(wins["pnl_usd"].sum())
    gross_loss = float(-losses["pnl_usd"].sum())
    pf = gross_win / max(gross_loss, 1e-9)
    avg_win = float(wins["pnl_usd"].mean()) if len(wins) else 0.0
    avg_loss = float(losses["pnl_usd"].mean()) if len(losses) else 0.0
    expectancy = float(trades["pnl_usd"].mean())

    eq = trades["pnl_usd"].cumsum().values
    max_dd = float((np.maximum.accumulate(eq) - eq).max()) if len(eq) else 0.0

    # Win/loss streaks
    signs = np.sign(trades["pnl_usd"].values)
    w_streak = l_streak = max_w = max_l = 0
    for s in signs:
        if s > 0: w_streak += 1; l_streak = 0; max_w = max(max_w, w_streak)
        elif s < 0: l_streak += 1; w_streak = 0; max_l = max(max_l, l_streak)
        else: w_streak = l_streak = 0

    days = (trades["time"].iat[-1] - trades["time"].iat[0]).days
    trades_per_day = len(trades) / max(days, 1)

    # Per-cluster summaries — schema matches Midas/Oracle entries
    REGIME_COLORS = {0:"#10b981", 1:"#f5c518", 2:"#6366f1", 3:"#ef4444", 4:"#f59e0b"}
    regimes = []
    for cid in sorted(trades["cid"].unique()):
        sub = trades[trades["cid"] == cid]
        w = (sub["pnl_usd"] > 0).sum()
        _wr = 100.0 * w / max(len(sub), 1)
        _pf = (sub.loc[sub["pnl_usd"]>0, "pnl_usd"].sum()
               / max(-sub.loc[sub["pnl_usd"]<0, "pnl_usd"].sum(), 1e-9))
        regimes.append({
            "name": f"C{int(cid)} {CLUSTER_NAMES.get(int(cid), f'C{cid}')}",
            "color": REGIME_COLORS.get(int(cid), "#888"),
            "trades": int(len(sub)),
            "wr": round(_wr, 1),
            "pf": round(float(_pf), 2),
            "pnl": round(float(sub["pnl_usd"].sum()), 1),
        })

    # Top rules by PF — schema matches Midas/Oracle entries
    rule_stats = []
    for rule, sub in trades.groupby("rule"):
        if len(sub) < 20: continue
        _pf = (sub.loc[sub["pnl_usd"]>0, "pnl_usd"].sum()
               / max(-sub.loc[sub["pnl_usd"]<0, "pnl_usd"].sum(), 1e-9))
        rule_stats.append({"name": rule, "pf": round(float(_pf), 2),
                          "trades": int(len(sub))})
    top_rules = sorted(rule_stats, key=lambda r: r["pf"], reverse=True)[:5]

    # Equity curve samples (~200 pts)
    sample_stride = max(1, len(trades) // 200)
    eq_curve = [float(x) for x in eq[::sample_stride]]
    eq_dates = [t.strftime("%Y-%m-%d") for t in trades["time"].iloc[::sample_stride]]

    # ── PNG equity curve ──
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(trades["time"], eq, color="#f59e0b", lw=1.8)
    ax.fill_between(trades["time"], eq, 0, where=(eq >= 0), color="#f59e0b", alpha=0.15)
    ax.fill_between(trades["time"], eq, 0, where=(eq < 0), color="#ef4444", alpha=0.15)
    ax.axhline(0, color="#444", lw=0.6)
    ax.set_facecolor("#0a0a0a"); fig.patch.set_facecolor("#0a0a0a")
    for sp in ax.spines.values(): sp.set_color("#333")
    ax.tick_params(colors="#888", labelsize=9)
    ax.grid(alpha=0.15, color="#444")
    ax.set_title(
        f"EdgePredictor BTC — Honest Holdout "
        f"({trades['time'].iat[0]:%b %Y} – {trades['time'].iat[-1]:%b %Y})",
        color="#e5e5e5", fontsize=11, pad=10)
    ax.set_ylabel("Cumulative P&L (USD, 0.01 lot)", color="#aaa")
    stats = (f"{len(trades):,} trades | WR {wr:.1f}% | PF {pf:.2f} | "
             f"Total ${total_pnl:+.0f} | MaxDD ${-max_dd:.0f}")
    ax.text(0.01, 0.97, stats, transform=ax.transAxes, color="#f5c518",
            fontsize=9, va="top",
            bbox=dict(facecolor="#111", edgecolor="#333", boxstyle="round,pad=0.4"))
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved PNG: {OUT_PNG}")

    # ── JSON entry matching midas/oracle schema ──
    btc_entry = {
        "name":              "EdgePredictor BTC",
        "asset":             "BTCUSD",
        "color":             "#f59e0b",
        "timeframe":         "M5",
        "image":             "/btc_backtest.png",
        "period":            f"{trades['time'].iat[0]:%b %Y} – {trades['time'].iat[-1]:%b %Y} (clean holdout)",
        "calendar_days":     int(days),
        "base_lot":          0.01,
        "pip_value_per_lot": 1.0,
        "total_trades":      int(len(trades)),
        "trades_per_day":    round(float(trades_per_day), 2),
        "win_rate":          round(float(wr), 1),
        "profit_factor":     round(float(pf), 2),
        "total_pnl":         round(total_pnl, 1),
        "max_dd":            round(-float(max_dd), 1),
        "avg_win":           round(float(avg_win), 2),
        "avg_loss":          round(float(avg_loss), 2),
        "rr_ratio":          round(float(avg_win / abs(avg_loss)) if avg_loss else 0.0, 2),
        "expectancy":        round(float(expectancy), 2),
        "max_win_streak":    int(max_w),
        "max_loss_streak":   int(max_l),
        "long_trades":       int((trades["direction"] == 1).sum()),
        "short_trades":      int((trades["direction"] == -1).sum()),
        "active_rules":      int(trades["rule"].nunique()),
        "total_rules":       26,
        "regimes":           regimes,
        "top_rules":         top_rules,
        "equity_curve":      eq_curve,
        "equity_dates":      eq_dates,
    }

    data = json.load(open(OUT_JSON)) if OUT_JSON.exists() else {}
    data["btc"] = btc_entry
    ordered = {k: data[k] for k in ["midas", "oracle", "btc"] if k in data}
    for k, v in data.items():
        if k not in ordered: ordered[k] = v
    with open(OUT_JSON, "w") as f:
        json.dump(ordered, f, indent=2)
    print(f"Updated JSON: {OUT_JSON}")

    print(f"\nFINAL:  n={len(trades)}  PF={pf:.2f}  WR={wr:.1f}%  "
          f"DD=${-max_dd:.0f}  Total=${total_pnl:+.0f}")


if __name__ == "__main__":
    main()
