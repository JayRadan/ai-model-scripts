"""
Regenerate the Oracle /backtest-page assets from the v7.2-lite smoothed pipeline:
  • per-cluster stats
  • top 5 rules by PF
  • equity_curve + equity_dates (sampled for the interactive chart)
  • total stats for the summary cards
  • oracle_backtest.png — same visual style as zigzag_backtest.png

Mirrors the metrics schema of public/backtest_data.json's 'midas' entry so the
/backtest page can render Oracle identically.

Runs the SAME clean-holdout backtest as 01_validate_v72_lite.py.  Inputs:
setups_*_v72l.csv + swing_v5_xauusd.csv.  No retraining — just exercises the
already-validated pipeline and writes results to disk.
"""
from __future__ import annotations
import glob, json, os, sys, time as _time
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/jay/Desktop/new-model-zigzag/model_pipeline")
import paths as P

# Pull reusable functions from the validation script
sys.path.insert(0, "/home/jay/Desktop/new-model-zigzag/experiments/v72_lite_deploy")
import importlib
val = importlib.import_module("01_validate_v72_lite")


OUT_PNG   = "/home/jay/Desktop/new-model/commercial/website/public/oracle_backtest.png"
OUT_JSON  = "/home/jay/Desktop/new-model/commercial/website/public/backtest_data.json"

# $ per R  — matches Midas's ~$0.50/R reference (avg_loss $-2.05 / 4R hard SL).
DOLLAR_PER_R = 0.51


def main():
    t0 = _time.time()
    print("Loading setups (v72l, step=1 smoothed)...")
    train, test = val.load_and_split()
    print("Loading swing + physics...")
    swing, atr = val.load_swing_with_physics()

    print("\nTraining v7.2-lite base (matches 01_validate_v72_lite exactly)...")
    mdls, thrs = val.train_conf(train, val.V72L_FEATS, "v72l-conf")
    tc = val.confirm(train, mdls, thrs, val.V72L_FEATS)
    exit_mdl = val.train_exit(tc, swing, atr)

    print("Simulating TRAIN trades to fit meta head...")
    tt = val.simulate(tc, swing, atr, exit_mdl)
    tc["direction"] = tc["direction"].astype(int); tc["cid"] = tc["cid"].astype(int)
    md = tt.merge(tc[["time","cid","rule"] + val.V72L_FEATS], on=["time","cid","rule"], how="left")
    md["meta_label"] = (md["pnl_R"] > 0).astype(int)
    md = md.sort_values("time").reset_index(drop=True)
    s_idx = int(len(md) * 0.80)
    mtr = md.iloc[:s_idx]
    meta_mdl = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             eval_metric="logloss", verbosity=0)
    meta_mdl.fit(mtr[val.META_FEATS].fillna(0).values, mtr["meta_label"].values)
    META_THRESHOLD = 0.675

    print("\nHoldout: confirm → meta filter → simulate")
    tec = val.confirm(test, mdls, thrs, val.V72L_FEATS)
    tec["direction"] = tec["direction"].astype(int); tec["cid"] = tec["cid"].astype(int)
    pm = meta_mdl.predict_proba(tec[val.META_FEATS].fillna(0).values)[:, 1]
    tec_m = tec[pm >= META_THRESHOLD].copy()
    print(f"  confirmed {len(tec):,} → after meta {len(tec_m):,}")
    trades = val.simulate(tec_m, swing, atr, exit_mdl)
    trades["time"] = pd.to_datetime(trades["time"])
    trades = trades.sort_values("time").reset_index(drop=True)

    # ── Aggregate stats (with $ conversion) ─────────────────────────
    trades["pnl_usd"] = trades["pnl_R"] * DOLLAR_PER_R
    wins = trades[trades["pnl_usd"] > 0]
    losses = trades[trades["pnl_usd"] <= 0]
    pf = wins["pnl_usd"].sum() / max(-losses["pnl_usd"].sum(), 1e-9)
    wr = 100.0 * len(wins) / max(len(trades), 1)
    total_pnl = float(trades["pnl_usd"].sum())
    eq = trades["pnl_usd"].cumsum().values
    max_dd = float((np.maximum.accumulate(eq) - eq).max()) if len(eq) else 0.0
    avg_win = float(wins["pnl_usd"].mean()) if len(wins) else 0.0
    avg_loss = float(losses["pnl_usd"].mean()) if len(losses) else 0.0
    expectancy = float(trades["pnl_usd"].mean())

    days = (trades["time"].iat[-1] - trades["time"].iat[0]).days or 1
    trades_per_day = len(trades) / max(days, 1)

    # ── Win / loss streaks ──────────────────────────────────────────
    signs = np.sign(trades["pnl_usd"].values)
    max_win_streak = 0; max_loss_streak = 0; cur_w = 0; cur_l = 0
    for s in signs:
        if s > 0:  cur_w += 1; cur_l = 0
        elif s < 0: cur_l += 1; cur_w = 0
        if cur_w > max_win_streak: max_win_streak = cur_w
        if cur_l > max_loss_streak: max_loss_streak = cur_l

    # ── Per-cluster ─────────────────────────────────────────────────
    cluster_names = {0: "Uptrend", 1: "MeanRevert", 2: "TrendRange",
                     3: "Downtrend", 4: "HighVol"}
    cluster_colors = {0: "#f5c518", 1: "#3b82f6", 2: "#00E5FF",
                      3: "#ef4444", 4: "#10b981"}
    regimes = []
    for cid in sorted(trades["cid"].unique()):
        grp = trades[trades["cid"] == cid]
        w = grp[grp["pnl_usd"] > 0]; l = grp[grp["pnl_usd"] <= 0]
        regimes.append({
            "name":   f"C{cid} {cluster_names.get(cid, '?')}",
            "color":  cluster_colors.get(cid, "#888888"),
            "trades": int(len(grp)),
            "wr":     float(100.0 * len(w) / max(len(grp), 1)),
            "pf":     float(w["pnl_usd"].sum() / max(-l["pnl_usd"].sum(), 1e-9)),
            "pnl":    float(grp["pnl_usd"].sum()),
        })

    # ── Top rules by PF (min 15 trades for stability) ──────────────
    rule_stats = []
    for rule, grp in trades.groupby("rule"):
        if len(grp) < 15: continue
        w = grp[grp["pnl_usd"] > 0]; l = grp[grp["pnl_usd"] <= 0]
        rule_stats.append({
            "name":   rule,
            "pf":     float(w["pnl_usd"].sum() / max(-l["pnl_usd"].sum(), 1e-9)),
            "trades": int(len(grp)),
        })
    rule_stats.sort(key=lambda r: -r["pf"])
    top_rules = rule_stats[:5]

    # ── Equity curve — downsample to ~200 evenly-spaced points ──────
    N_POINTS = 200
    idxs = np.linspace(0, len(trades) - 1, N_POINTS).astype(int)
    equity_curve = [float(v) for v in eq[idxs]]
    equity_dates = [trades["time"].iat[i].strftime("%Y-%m-%d") for i in idxs]

    # ── PNG (matches Midas's zigzag_backtest.png visual style) ──────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(trades["time"], eq, color="#6366f1", linewidth=1.8)
    ax.fill_between(trades["time"], eq, 0,
                    where=(eq >= 0), color="#6366f1", alpha=0.15)
    ax.fill_between(trades["time"], eq, 0,
                    where=(eq < 0), color="#ef4444", alpha=0.15)
    ax.axhline(0, color="#444", linewidth=0.6)
    ax.set_facecolor("#0a0a0a")
    fig.patch.set_facecolor("#0a0a0a")
    for spine in ax.spines.values(): spine.set_color("#333")
    ax.tick_params(colors="#888", labelsize=9)
    ax.grid(alpha=0.15, color="#444")
    ax.set_title(
        f"EdgePredictor Oracle — XAUUSD M5 · clean holdout Dec 2024 – Apr 2026",
        color="#e5e5e5", fontsize=11, pad=12)
    ax.set_ylabel("Cumulative P&L ($)", color="#aaa", fontsize=10)
    stats_txt = (f"{len(trades)} trades | PF {pf:.2f} | WR {wr:.1f}% | "
                 f"Expectancy ${expectancy:+.2f} | MaxDD ${-max_dd:.0f}")
    ax.text(0.02, 0.96, stats_txt, transform=ax.transAxes, color="#f5c518",
            fontsize=9, verticalalignment="top",
            bbox=dict(facecolor="#111", edgecolor="#333", boxstyle="round,pad=0.4"))
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved PNG: {OUT_PNG}")

    # ── Build Oracle entry (schema matches Midas) ───────────────────
    oracle = {
        "name":              "EdgePredictor Oracle",
        "asset":             "XAUUSD",
        "color":             "#6366f1",
        "timeframe":         "M5",
        "image":             "/oracle_backtest.png",
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
        "max_win_streak":    int(max_win_streak),
        "max_loss_streak":   int(max_loss_streak),
        "long_trades":       int((trades["direction"] == 1).sum()),
        "short_trades":      int((trades["direction"] == -1).sum()),
        "active_rules":      int(trades["rule"].nunique()),
        "total_rules":       26,
        "regimes":           regimes,
        "top_rules":         top_rules,
        "equity_curve":      equity_curve,
        "equity_dates":      equity_dates,
    }

    # Merge into backtest_data.json preserving order: midas, oracle, rest
    data = json.load(open(OUT_JSON))
    data["oracle"] = oracle
    ordered = {k: data[k] for k in ["midas", "oracle"] if k in data}
    for k, v in data.items():
        if k not in ordered: ordered[k] = v
    with open(OUT_JSON, "w") as f:
        json.dump(ordered, f, indent=2)
    print(f"  Updated JSON: {OUT_JSON}")
    print(f"\nFINAL:  n={len(trades)}  PF={pf:.2f}  WR={wr:.1f}%  "
          f"DD=${-max_dd:.0f}  Total=${total_pnl:+.0f}  ({_time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
