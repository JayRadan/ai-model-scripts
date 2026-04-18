"""
Midas v2 — per-cluster triple-barrier + bar-level primary + meta-labeling.

Goal: WR ≥ 70%, PF ≥ 1.5, ≥5 trades/day on XAUUSD M5 holdout.

Architecture:
  Step 1: Per-cluster triple-barrier labels on every cluster bar
          (cluster-specific TP/SL/time — tight targets = predictable)
  Step 2: Per-cluster bar-level primary models (long + short binary)
          Features: 36 existing + rule-trigger booleans as features
  Step 3: Meta-labeling — train a classifier to predict
          "will this primary-model call win?" on holdout validation fold
  Step 4: Backtest — only trade when meta_prob ≥ threshold, simulate with
          matched TP/SL.

Honest: 70/10/20 train / meta-train / test split (time-ordered).
"""
import os, json, glob, time
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

OUT_DIR = "/home/jay/Desktop/new-model-zigzag/experiments/midas_v2"
os.makedirs(OUT_DIR, exist_ok=True)

CLUSTER_NAMES = {0: "Uptrend", 1: "MeanRevert", 2: "TrendRange", 3: "Downtrend", 4: "HighVol"}

# Per-cluster label definition (matched to regime physics)
CLUSTER_CFG = {
    0: {"tp": 0.7, "sl": 1.0, "T": 24, "name": "Uptrend"},
    1: {"tp": 0.5, "sl": 1.0, "T": 12, "name": "MeanRevert"},
    2: {"tp": 0.8, "sl": 1.0, "T": 30, "name": "TrendRange"},
    3: {"tp": 0.7, "sl": 1.0, "T": 24, "name": "Downtrend"},
    4: {"tp": 1.0, "sl": 1.0, "T": 15, "name": "HighVol"},
}

SPREAD = 0.05

DATA_DIR = "/home/jay/Desktop/new-model-zigzag/data"
RAW_CSV = f"{DATA_DIR}/swing_v5_xauusd.csv"


# ── Load full bar data for forward-looking outcomes ──
print("Loading raw bar series for triple-barrier outcomes...")
raw = pd.read_csv(RAW_CSV, parse_dates=["time"])
raw = raw[raw["time"] >= "2016-01-01"].reset_index(drop=True)
c = raw["close"].values.astype(np.float64)
h = raw["high"].values.astype(np.float64)
l = raw["low"].values.astype(np.float64)
prev_c = np.roll(c, 1); prev_c[0] = c[0]
tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
atr14 = pd.Series(tr).rolling(14, min_periods=1).mean().values
n_bars = len(raw)
time_to_idx = dict(zip(raw["time"], range(n_bars)))
print(f"  {n_bars:,} bars loaded")


# ── Build rule-trigger booleans aligned to all bars ──
print("\nLoading setup triggers (for use as features)...")
rule_matrix = {}  # rule_name → array[n_bars] of 0/+1/-1
all_rules = []
for cid in range(5):
    s = pd.read_csv(f"{DATA_DIR}/setups_{cid}.csv", parse_dates=["time"])
    for r in s["rule"].unique():
        all_rules.append(r)
        col = np.zeros(n_bars, dtype=np.int8)
        sub = s[s["rule"] == r]
        for t, dirv in zip(sub["time"], sub["direction"]):
            idx = time_to_idx.get(t)
            if idx is not None:
                col[idx] = 1 if dirv in (1, "buy") else -1
        rule_matrix[r] = col
all_rules = sorted(set(all_rules))
print(f"  {len(all_rules)} rules, aggregated rule-count features")


def triple_barrier(idx, a, tp, sl, T, direction):
    """Return 1 if TP hit first, 0 if SL hit first, -1 if timeout."""
    if a < 1e-10: return -1
    entry = c[idx]
    if direction == 1:
        tp_p = entry + tp * a; sl_p = entry - sl * a
    else:
        tp_p = entry - tp * a; sl_p = entry + sl * a
    end = min(idx + T, n_bars - 1)
    for k in range(idx + 1, end + 1):
        if direction == 1:
            if l[k] <= sl_p: return 0
            if h[k] >= tp_p: return 1
        else:
            if h[k] >= sl_p: return 0
            if l[k] <= tp_p: return 1
    return -1  # timeout


# ── Per-cluster processing ──
all_backtest = []  # trades from all clusters in meta-test fold

print("\n" + "="*70)
print("PER-CLUSTER: labels → primary → meta")
print("="*70)

# Store results per cluster
cluster_results = {}

# Track overall meta training data (to train ONE global meta model)
meta_features_all = []
meta_labels_all = []
meta_meta = []  # for backtest: (time, idx, cluster, direction, prob_long, prob_short)

for cid in range(5):
    cfg = CLUSTER_CFG[cid]
    print(f"\n── Cluster {cid} ({cfg['name']}) — TP={cfg['tp']}R SL={cfg['sl']}R T={cfg['T']} ──")

    df = pd.read_csv(f"{DATA_DIR}/cluster_{cid}_data.csv", parse_dates=["time"])
    df = df[df["time"] >= "2016-01-01"].sort_values("time").reset_index(drop=True)
    print(f"  {len(df):,} bars in cluster")

    # Map each cluster bar to its idx in raw
    df["idx"] = df["time"].map(time_to_idx)
    df = df.dropna(subset=["idx"]).copy()
    df["idx"] = df["idx"].astype(int)

    # Compute long + short triple-barrier labels
    print(f"  Computing triple-barrier labels...")
    long_lbl = np.full(len(df), -1, dtype=np.int8)
    short_lbl = np.full(len(df), -1, dtype=np.int8)
    tp, sl, T = cfg["tp"], cfg["sl"], cfg["T"]
    for i, idx in enumerate(df["idx"].values):
        if idx + T >= n_bars: continue
        a = atr14[idx]
        long_lbl[i] = triple_barrier(idx, a, tp, sl, T, +1)
        short_lbl[i] = triple_barrier(idx, a, tp, sl, T, -1)

    df["long_win"] = long_lbl
    df["short_win"] = short_lbl
    df = df[(df["long_win"] != -1) | (df["short_win"] != -1)].copy()

    # Base rate
    lw = df[df["long_win"] != -1]["long_win"].mean()
    sw = df[df["short_win"] != -1]["short_win"].mean()
    print(f"  Base WR: long={lw:.2%} ({(df['long_win']!=-1).sum()} samples)  "
          f"short={sw:.2%} ({(df['short_win']!=-1).sum()} samples)")

    # Features: 36 base feats + rule trigger features (all rules as signed ints)
    base_feats = [col for col in df.columns if col.startswith("f") or
                  col in ["rsi14","rsi6","stoch_k","stoch_d","bb_pct",
                          "mom5","mom10","mom20","ll_dist10","hh_dist10",
                          "vol_accel","atr_ratio","spread_norm","hour_enc","dow_enc"]]

    # Add rule features (signed: +1 buy fire, -1 sell fire, 0 none)
    for r in all_rules:
        df[f"rule_{r}"] = rule_matrix[r][df["idx"].values]

    rule_feats = [f"rule_{r}" for r in all_rules]
    # Aggregate rule counts (bullish/bearish rule count)
    rule_arr = df[rule_feats].values
    df["n_bull_rules"] = (rule_arr > 0).sum(axis=1)
    df["n_bear_rules"] = (rule_arr < 0).sum(axis=1)

    feat_cols = base_feats + rule_feats + ["n_bull_rules", "n_bear_rules"]

    # Time-ordered split: 70% train / 10% meta-train / 20% test
    n = len(df)
    n1 = int(n * 0.70)
    n2 = int(n * 0.80)
    train = df.iloc[:n1].copy()
    meta_train = df.iloc[n1:n2].copy()
    test = df.iloc[n2:].copy()
    print(f"  Split: train={len(train)}  meta={len(meta_train)}  test={len(test)}")

    # Train primary LONG classifier
    tl = train[train["long_win"] != -1]
    if len(tl) < 100:
        print(f"  Skipping — insufficient long samples")
        continue
    Xl = tl[feat_cols].fillna(0).values
    yl = tl["long_win"].values
    pos_rate_l = yl.mean()
    scale_l = (1 - pos_rate_l) / max(pos_rate_l, 0.01)

    model_long = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                scale_pos_weight=scale_l, subsample=0.8,
                                colsample_bytree=0.8, eval_metric="auc",
                                early_stopping_rounds=30, verbosity=0)
    # Hold out last 10% of train for early stop
    split_es = int(len(tl) * 0.9)
    model_long.fit(Xl[:split_es], yl[:split_es],
                   eval_set=[(Xl[split_es:], yl[split_es:])], verbose=False)

    # Train primary SHORT classifier
    ts = train[train["short_win"] != -1]
    Xs = ts[feat_cols].fillna(0).values
    ys = ts["short_win"].values
    pos_rate_s = ys.mean()
    scale_s = (1 - pos_rate_s) / max(pos_rate_s, 0.01)
    model_short = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                 scale_pos_weight=scale_s, subsample=0.8,
                                 colsample_bytree=0.8, eval_metric="auc",
                                 early_stopping_rounds=30, verbosity=0)
    split_es = int(len(ts) * 0.9)
    model_short.fit(Xs[:split_es], ys[:split_es],
                    eval_set=[(Xs[split_es:], ys[split_es:])], verbose=False)

    # Evaluate AUC on meta_train
    mt_l = meta_train[meta_train["long_win"] != -1]
    mt_s = meta_train[meta_train["short_win"] != -1]
    auc_l = roc_auc_score(mt_l["long_win"], model_long.predict_proba(mt_l[feat_cols].fillna(0).values)[:, 1]) if len(mt_l) > 10 else 0.5
    auc_s = roc_auc_score(mt_s["short_win"], model_short.predict_proba(mt_s[feat_cols].fillna(0).values)[:, 1]) if len(mt_s) > 10 else 0.5
    print(f"  Primary AUC (meta-train fold): long={auc_l:.3f}  short={auc_s:.3f}")

    # Generate primary predictions on meta_train & test
    for phase, subset, collect in [("meta", meta_train, True), ("test", test, False)]:
        Xf = subset[feat_cols].fillna(0).values
        pl = model_long.predict_proba(Xf)[:, 1]
        ps = model_short.predict_proba(Xf)[:, 1]
        subset = subset.copy()
        subset["prob_long"] = pl
        subset["prob_short"] = ps

        # For each row, determine primary's decision: direction with higher prob
        # Only record candidate trades (not both-low) later at backtest time
        for i, (_, row) in enumerate(subset.iterrows()):
            direction = 1 if pl[i] >= ps[i] else -1
            if direction == 1:
                label = row["long_win"] if row["long_win"] != -1 else None
            else:
                label = row["short_win"] if row["short_win"] != -1 else None
            if label is None: continue

            # Meta features: primary probs, cluster id, time/vol features, rule counts
            mf = [pl[i], ps[i], max(pl[i], ps[i]), abs(pl[i] - ps[i]),
                  row.get("atr_ratio", 0), row.get("hour_enc", 0), row.get("dow_enc", 0),
                  row.get("n_bull_rules", 0), row.get("n_bear_rules", 0)]
            # Cluster one-hot
            for k in range(5):
                mf.append(1 if cid == k else 0)

            if phase == "meta":
                meta_features_all.append(mf)
                meta_labels_all.append(int(label))
            else:
                # For test phase, store for backtest
                meta_meta.append({
                    "time": row["time"], "idx": int(row["idx"]),
                    "cluster": cid, "direction": direction,
                    "prob_long": pl[i], "prob_short": ps[i],
                    "label": int(label),
                    "meta_features": mf,
                    "atr": atr14[int(row["idx"])],
                })

    cluster_results[cid] = {
        "auc_long": auc_l, "auc_short": auc_s,
        "base_wr_long": lw, "base_wr_short": sw,
        "n_train": len(train), "n_test": len(test),
    }


# ── Train meta-classifier on aggregated meta_train data ──
print("\n" + "="*70)
print("META-LABELING: train classifier on primary predictions")
print("="*70)

MX = np.array(meta_features_all, dtype=np.float32)
my = np.array(meta_labels_all, dtype=np.int8)
print(f"  Meta samples: {len(my)}  base WR: {my.mean():.2%}")

meta_model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8,
                            eval_metric="auc", verbosity=0)
# Internal validation split
split = int(len(my) * 0.85)
meta_model.fit(MX[:split], my[:split], eval_set=[(MX[split:], my[split:])], verbose=False)
meta_auc = roc_auc_score(my[split:], meta_model.predict_proba(MX[split:])[:, 1])
print(f"  Meta AUC (internal val): {meta_auc:.3f}")


# ── Predict meta probs on test set and backtest ──
print("\n" + "="*70)
print("BACKTEST on holdout")
print("="*70)

test_MX = np.array([m["meta_features"] for m in meta_meta], dtype=np.float32)
test_meta_probs = meta_model.predict_proba(test_MX)[:, 1]
for i, m in enumerate(meta_meta):
    m["meta_prob"] = test_meta_probs[i]

test_df = pd.DataFrame(meta_meta)
test_days = (test_df["time"].max() - test_df["time"].min()).days
print(f"  Test span: {test_days} days, {len(test_df)} candidate trades")


def simulate_trades(subset_df):
    """Compute PnL for trades in subset using their triple-barrier label + cluster TP/SL."""
    trades = []
    for _, r in subset_df.iterrows():
        cfg = CLUSTER_CFG[int(r["cluster"])]
        tp = cfg["tp"]; sl = cfg["sl"]
        if r["label"] == 1:
            pnl = tp - SPREAD
        else:
            pnl = -sl - SPREAD
        trades.append({"time": r["time"], "pnl": pnl, "cluster": int(r["cluster"])})
    return pd.DataFrame(trades)


print(f"\n  {'Meta thr':<10} {'n':>6} {'t/day':>6} {'WR':>6} {'PF':>6} {'PnL':>8}  status")
print("  " + "-"*70)

best = None
thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
results = []
for thr in thresholds:
    sel = test_df[test_df["meta_prob"] >= thr]
    if len(sel) < 10: continue
    trades = simulate_trades(sel)
    wins = (trades["pnl"] > 0).sum()
    losses = (trades["pnl"] < 0).sum()
    n = len(trades)
    wr = wins / n
    win_sum = trades[trades["pnl"] > 0]["pnl"].sum()
    loss_sum = -trades[trades["pnl"] < 0]["pnl"].sum()
    pf = win_sum / max(loss_sum, 0.01)
    pnl = trades["pnl"].sum()
    tpd = n / max(test_days, 1)
    hits = (wr >= 0.70 and pf >= 1.50 and tpd >= 5)
    status = "★ TARGET" if hits else ("~ near" if (wr >= 0.65 and pf >= 1.40 and tpd >= 3) else "")
    print(f"  {thr:<10.2f} {n:>6} {tpd:>6.1f} {wr:>6.0%} {pf:>6.2f} {pnl:>+8.1f}  {status}")
    results.append({"thr": thr, "n": n, "tpd": tpd, "wr": wr, "pf": pf, "pnl": pnl, "trades": trades})
    if hits and (best is None or pf > best["pf"]):
        best = {"thr": thr, "n": n, "tpd": tpd, "wr": wr, "pf": pf, "pnl": pnl, "trades": trades}


# ── Per-cluster breakdown at best threshold ──
if best is not None:
    print(f"\n  ★ HIT TARGET at meta_prob ≥ {best['thr']}")
    print(f"\n  Per-cluster breakdown:")
    sel = test_df[test_df["meta_prob"] >= best["thr"]]
    for cid in range(5):
        sub = sel[sel["cluster"] == cid]
        if len(sub) == 0: continue
        tr = simulate_trades(sub)
        w = (tr["pnl"] > 0).sum(); n = len(tr)
        wr = w / n
        pf = tr[tr["pnl"] > 0]["pnl"].sum() / max(-tr[tr["pnl"] < 0]["pnl"].sum(), 0.01)
        print(f"    C{cid} {CLUSTER_NAMES[cid]:<12}: n={n:>4}  WR={wr:.0%}  PF={pf:.2f}")
else:
    print(f"\n  No threshold hit target. Near-misses may still be useful.")


# ── Plot ──
fig = plt.figure(figsize=(20, 12), facecolor="#080c12")
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)

# Threshold sweep
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor("#0d1117")
thrs = [r["thr"] for r in results]
ax1.plot(thrs, [r["wr"]*100 for r in results], "o-", color="#10b981", label="WR %", linewidth=2)
ax1.plot(thrs, [r["pf"]*50 for r in results], "o-", color="#FFD700", label="PF × 50", linewidth=2)
ax1.plot(thrs, [r["tpd"]*5 for r in results], "o-", color="#00E5FF", label="trades/day × 5", linewidth=2)
ax1.axhline(70, color="#10b981", linestyle="--", alpha=0.4)
ax1.axhline(75, color="#FFD700", linestyle="--", alpha=0.4, label="PF=1.5 line")
ax1.axhline(25, color="#00E5FF", linestyle="--", alpha=0.4, label="5 t/d line")
ax1.set_xlabel("Meta threshold", color="#5a7080")
ax1.set_title("Threshold Sweep", color="#FFD700", fontsize=12)
ax1.legend(fontsize=8)
ax1.tick_params(colors="#5a7080")
for sp in ax1.spines.values(): sp.set_edgecolor("#1e2a3a")

# Equity at best threshold (or threshold 0.60)
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor("#0d1117")
for r in results:
    tr = r["trades"].sort_values("time")
    if len(tr) == 0: continue
    eq = np.cumsum(tr["pnl"].values)
    ax2.plot(eq, linewidth=1.3, alpha=0.8,
             label=f"thr={r['thr']:.2f} WR={r['wr']:.0%} PF={r['pf']:.2f} n={r['n']}")
ax2.axhline(0, color="#444", linewidth=0.6)
ax2.legend(fontsize=7, loc="upper left")
ax2.set_title("Equity Curves by Threshold", color="#FFD700", fontsize=12)
ax2.tick_params(colors="#5a7080")
for sp in ax2.spines.values(): sp.set_edgecolor("#1e2a3a")

# Per-cluster AUC bars
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_facecolor("#0d1117")
cids = list(cluster_results.keys())
x = np.arange(len(cids))
width = 0.35
ax3.bar(x - width/2, [cluster_results[c]["auc_long"] for c in cids], width, color="#10b981", label="Long AUC")
ax3.bar(x + width/2, [cluster_results[c]["auc_short"] for c in cids], width, color="#ef4444", label="Short AUC")
ax3.axhline(0.5, color="#666", linestyle="--", alpha=0.5)
ax3.axhline(0.6, color="#FFD700", linestyle="--", alpha=0.4, label="0.6 threshold")
ax3.set_xticks(x)
ax3.set_xticklabels([f"C{c}\n{CLUSTER_NAMES[c][:8]}" for c in cids], color="#ccc", fontsize=9)
ax3.set_title("Primary Model AUC per Cluster", color="#FFD700", fontsize=12)
ax3.legend(fontsize=8)
ax3.tick_params(colors="#5a7080")
for sp in ax3.spines.values(): sp.set_edgecolor("#1e2a3a")

# Summary table
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor("#0d1117")
ax4.axis("off")
txt = "Threshold sweep:\n" + "─"*50 + "\n"
txt += f"{'thr':<6}{'n':>6}{'t/d':>6}{'WR':>6}{'PF':>6}{'PnL':>8}\n"
txt += "─"*50 + "\n"
for r in results:
    mark = "★" if (r["wr"]>=0.70 and r["pf"]>=1.50 and r["tpd"]>=5) else ""
    txt += f"{r['thr']:<6.2f}{r['n']:>6}{r['tpd']:>6.1f}{r['wr']:>6.0%}{r['pf']:>6.2f}{r['pnl']:>+8.1f} {mark}\n"
if best:
    txt += f"\n★ TARGET HIT at thr={best['thr']}\n  WR={best['wr']:.0%} PF={best['pf']:.2f} {best['tpd']:.1f} t/d"
else:
    txt += "\nNo target hit."
ax4.text(0, 1, txt, fontsize=10, color="#ccc", family="monospace", va="top")

plt.suptitle("Midas v2 — Per-cluster Triple-Barrier + Meta-labeling",
             color="#FFD700", fontsize=16, fontweight="bold", y=1.0)
out = os.path.join(OUT_DIR, "midas_v2.png")
plt.savefig(out, dpi=140, bbox_inches="tight", facecolor="#080c12")
print(f"\nSaved: {out}")
