"""
Midas v3 — rule-gated setups + tight-TP retraining + meta-labeling.

Learned from v2: rules are the edge (bar-level AUC = 0.50 without them).
Keep rules as gate. Change the label to tight TP. Add meta filter on top.

Architecture:
  Step 1: Load rule-gated setups (setups_k.csv) — ~72k prefiltered moments
  Step 2: Re-label each with tight triple-barrier per cluster
          (raises base WR from ~44% to ~60%+)
  Step 3: Retrain per-rule primary confirm models on tight label
  Step 4: Meta-layer on [primary_prob, cluster, time, vol, rule_agreement]
          trained on 10% middle fold
  Step 5: Backtest on 20% holdout, sweep meta threshold.

Target: WR ≥ 70%, PF ≥ 1.5, ≥5 trades/day.
"""
import os, json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

OUT_DIR = "/home/jay/Desktop/new-model-zigzag/experiments/midas_v3"
os.makedirs(OUT_DIR, exist_ok=True)

CLUSTER_NAMES = {0: "Uptrend", 1: "MeanRevert", 2: "TrendRange", 3: "Downtrend", 4: "HighVol"}
CLUSTER_CFG = {
    0: {"tp": 0.7, "sl": 1.0, "T": 24},
    1: {"tp": 0.5, "sl": 1.0, "T": 12},
    2: {"tp": 0.8, "sl": 1.0, "T": 30},
    3: {"tp": 0.7, "sl": 1.0, "T": 24},
    4: {"tp": 1.0, "sl": 1.0, "T": 15},
}
SPREAD = 0.05

DATA_DIR = "/home/jay/Desktop/new-model-zigzag/data"
RAW_CSV = f"{DATA_DIR}/swing_v5_xauusd.csv"

# ── Load raw for triple-barrier ──
print("Loading raw M5...")
raw = pd.read_csv(RAW_CSV, parse_dates=["time"])
raw = raw[raw["time"] >= "2016-01-01"].reset_index(drop=True)
h = raw["high"].values.astype(np.float64)
l = raw["low"].values.astype(np.float64)
c = raw["close"].values.astype(np.float64)
prev_c = np.roll(c, 1); prev_c[0] = c[0]
tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
atr14 = pd.Series(tr).rolling(14, min_periods=1).mean().values
n_bars = len(raw)
t2i = dict(zip(raw["time"], range(n_bars)))
print(f"  {n_bars:,} bars")


def triple_barrier(idx, a, tp, sl, T, direction):
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
    return -1


# ── Load rule-gated setups (all clusters) ──
print("\nLoading rule-gated setups...")
setups_parts = []
for cid in range(5):
    s = pd.read_csv(f"{DATA_DIR}/setups_{cid}.csv", parse_dates=["time"])
    s["cluster_id"] = cid
    setups_parts.append(s)
setups = pd.concat(setups_parts, ignore_index=True)
# Map to bar index
setups["idx"] = setups["time"].map(t2i)
setups = setups.dropna(subset=["idx"]).copy()
setups["idx"] = setups["idx"].astype(int)
print(f"  {len(setups):,} total rule-gated setups across all clusters")

# ── Compute tight triple-barrier label per setup ──
print("\nComputing tight triple-barrier labels (per cluster TP)...")
labels = np.full(len(setups), -1, dtype=np.int8)
for i, (idx, cid, dirv) in enumerate(zip(setups["idx"].values,
                                         setups["cluster_id"].values,
                                         setups["direction"].values)):
    cfg = CLUSTER_CFG[int(cid)]
    if idx + cfg["T"] >= n_bars: continue
    a = atr14[idx]
    direction = 1 if dirv in (1, "buy") else -1
    labels[i] = triple_barrier(idx, a, cfg["tp"], cfg["sl"], cfg["T"], direction)

setups["tight_label"] = labels
setups = setups[setups["tight_label"] != -1].reset_index(drop=True)
print(f"  Usable (non-timeout): {len(setups):,}")
print(f"  Overall base WR (tight TP): {setups['tight_label'].mean():.2%}")
for cid in range(5):
    sub = setups[setups["cluster_id"] == cid]
    if len(sub):
        print(f"    C{cid} {CLUSTER_NAMES[cid]:<12}: n={len(sub):>5}  base WR={sub['tight_label'].mean():.2%}")


# ── Time-ordered split ──
setups = setups.sort_values("time").reset_index(drop=True)
n = len(setups)
train_end = int(n * 0.70)
meta_end = int(n * 0.80)
train = setups.iloc[:train_end].reset_index(drop=True)
meta_train = setups.iloc[train_end:meta_end].reset_index(drop=True)
test = setups.iloc[meta_end:].reset_index(drop=True)
print(f"\nSplit: train={len(train)}  meta={len(meta_train)}  test={len(test)}")

# Feature columns (exclude meta cols)
skip_cols = {"time", "idx", "direction", "rule", "cluster_id", "tight_label",
             "label", "outcome", "pnl", "reward", "target"}  # v3 leak fix: exclude prior-label columns
feat_cols = [c for c in setups.columns if c not in skip_cols]
print(f"  Features: {len(feat_cols)}")


# ── Train per-rule primary confirm models on tight label ──
print("\n" + "="*70)
print("PRIMARY: per-rule classifiers on tight triple-barrier label")
print("="*70)

primary_models = {}
primary_aucs = {}

for (cid, rule), g in train.groupby(["cluster_id", "rule"]):
    if len(g) < 100: continue
    X = g[feat_cols].fillna(0).values
    y = g["tight_label"].values
    if y.mean() in (0.0, 1.0): continue  # no class diversity

    pos = y.mean(); scale = (1 - pos) / max(pos, 0.01)
    model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                          scale_pos_weight=scale, subsample=0.8,
                          colsample_bytree=0.8, eval_metric="auc",
                          early_stopping_rounds=25, verbosity=0)
    sp = int(len(g) * 0.9)
    try:
        model.fit(X[:sp], y[:sp], eval_set=[(X[sp:], y[sp:])], verbose=False)
    except Exception:
        continue
    primary_models[(cid, rule)] = model

    # AUC on meta_train fold
    mg = meta_train[(meta_train["cluster_id"] == cid) & (meta_train["rule"] == rule)]
    if len(mg) > 20 and mg["tight_label"].nunique() > 1:
        auc = roc_auc_score(mg["tight_label"], model.predict_proba(mg[feat_cols].fillna(0).values)[:, 1])
        primary_aucs[(cid, rule)] = auc

print(f"\n  Trained {len(primary_models)} per-rule primary models")
print(f"  AUC summary (meta-train fold):")
aucs = sorted(primary_aucs.items(), key=lambda x: -x[1])
for (cid, rule), auc in aucs[:10]:
    print(f"    C{cid} {rule:<25} AUC={auc:.3f}")
print(f"  Median AUC: {np.median(list(primary_aucs.values())):.3f}")
print(f"  Best AUC: {max(primary_aucs.values()):.3f}")


# ── Score primary probs on meta_train and test ──
def score_primary(df):
    probs = np.full(len(df), np.nan)
    for (cid, rule), model in primary_models.items():
        mask = (df["cluster_id"] == cid) & (df["rule"] == rule)
        if mask.sum() == 0: continue
        X = df.loc[mask, feat_cols].fillna(0).values
        probs[mask.values] = model.predict_proba(X)[:, 1]
    return probs


meta_train["primary_prob"] = score_primary(meta_train)
test["primary_prob"] = score_primary(test)

# Drop rows with no primary model for that rule
meta_train = meta_train.dropna(subset=["primary_prob"]).reset_index(drop=True)
test = test.dropna(subset=["primary_prob"]).reset_index(drop=True)
print(f"\n  After primary scoring: meta={len(meta_train)}  test={len(test)}")


# ── Build meta features ──
def meta_feats(df):
    mf = pd.DataFrame()
    mf["primary_prob"] = df["primary_prob"]
    mf["atr_ratio"] = df.get("atr_ratio", 0)
    mf["hour"] = df.get("hour_enc", 0)
    mf["dow"] = df.get("dow_enc", 0)
    mf["rsi14"] = df.get("rsi14", 0)
    mf["stoch_k"] = df.get("stoch_k", 0)
    mf["bb_pct"] = df.get("bb_pct", 0)
    mf["vol_accel"] = df.get("vol_accel", 0)
    # Cluster one-hot
    for k in range(5):
        mf[f"c{k}"] = (df["cluster_id"] == k).astype(int)
    # Direction
    mf["is_long"] = (df["direction"].isin([1, "buy"])).astype(int)
    return mf.fillna(0).values


# ── Train meta-classifier ──
print("\n" + "="*70)
print("META: classifier on [primary_prob, context] → will this pass win?")
print("="*70)

MX_train = meta_feats(meta_train)
my_train = meta_train["tight_label"].values
print(f"  Meta samples: {len(my_train)}  base WR: {my_train.mean():.2%}")

scale = (1 - my_train.mean()) / max(my_train.mean(), 0.01)
meta_model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                            scale_pos_weight=scale, subsample=0.8,
                            colsample_bytree=0.8, eval_metric="auc",
                            early_stopping_rounds=25, verbosity=0)
sp = int(len(my_train) * 0.85)
meta_model.fit(MX_train[:sp], my_train[:sp],
               eval_set=[(MX_train[sp:], my_train[sp:])], verbose=False)
meta_auc = roc_auc_score(my_train[sp:], meta_model.predict_proba(MX_train[sp:])[:, 1])
print(f"  Meta AUC (internal val): {meta_auc:.3f}")


# ── Backtest on test fold ──
print("\n" + "="*70)
print("BACKTEST on holdout — threshold sweep")
print("="*70)

MX_test = meta_feats(test)
test["meta_prob"] = meta_model.predict_proba(MX_test)[:, 1]
test_days = (test["time"].max() - test["time"].min()).days
print(f"  Test span: {test_days} days, {len(test)} candidate trades")


def simulate(df):
    trades = []
    for _, r in df.iterrows():
        cfg = CLUSTER_CFG[int(r["cluster_id"])]
        pnl = (cfg["tp"] - SPREAD) if r["tight_label"] == 1 else (-cfg["sl"] - SPREAD)
        trades.append({"time": r["time"], "pnl": pnl, "cluster": int(r["cluster_id"])})
    return pd.DataFrame(trades)


print(f"\n  {'thr':<8}{'n':>7}{'t/day':>7}{'WR':>7}{'PF':>7}{'PnL':>9}  status")
print("  " + "-"*60)

results = []
best = None
for thr in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
    sel = test[test["meta_prob"] >= thr]
    if len(sel) < 10: continue
    tr = simulate(sel)
    n = len(tr); wins = (tr["pnl"] > 0).sum()
    wr = wins / n
    win_sum = tr[tr["pnl"] > 0]["pnl"].sum()
    loss_sum = -tr[tr["pnl"] < 0]["pnl"].sum()
    pf = win_sum / max(loss_sum, 0.01)
    pnl = tr["pnl"].sum()
    tpd = n / max(test_days, 1)
    hits = (wr >= 0.70 and pf >= 1.50 and tpd >= 5)
    near = (wr >= 0.65 and pf >= 1.40 and tpd >= 3)
    status = "★ TARGET" if hits else ("~ near" if near else "")
    print(f"  {thr:<8.2f}{n:>7}{tpd:>7.1f}{wr:>7.0%}{pf:>7.2f}{pnl:>+9.1f}  {status}")
    results.append({"thr": thr, "n": n, "tpd": tpd, "wr": wr, "pf": pf, "pnl": pnl, "trades": tr})
    if hits and (best is None or pf > best["pf"]):
        best = results[-1]


# ── Per-cluster breakdown at best (or at 0.60) ──
show_thr = best["thr"] if best else 0.60
print(f"\n  Per-cluster at meta_prob ≥ {show_thr}:")
sel = test[test["meta_prob"] >= show_thr]
for cid in range(5):
    sub = sel[sel["cluster_id"] == cid]
    if len(sub) == 0: continue
    tr = simulate(sub)
    n = len(tr); wr = (tr["pnl"] > 0).sum() / n
    pf = tr[tr["pnl"] > 0]["pnl"].sum() / max(-tr[tr["pnl"] < 0]["pnl"].sum(), 0.01)
    print(f"    C{cid} {CLUSTER_NAMES[cid]:<12}: n={n:>4}  WR={wr:.0%}  PF={pf:.2f}")


# ── Plot ──
fig = plt.figure(figsize=(20, 12), facecolor="#080c12")
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)

ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor("#0d1117")
thrs = [r["thr"] for r in results]
ax1.plot(thrs, [r["wr"]*100 for r in results], "o-", color="#10b981", label="WR %", linewidth=2)
ax1.plot(thrs, [r["pf"]*50 for r in results], "o-", color="#FFD700", label="PF × 50", linewidth=2)
ax1.plot(thrs, [min(r["tpd"]*5, 150) for r in results], "o-", color="#00E5FF", label="trades/day × 5", linewidth=2)
ax1.axhline(70, color="#10b981", linestyle="--", alpha=0.4)
ax1.axhline(75, color="#FFD700", linestyle="--", alpha=0.4)
ax1.axhline(25, color="#00E5FF", linestyle="--", alpha=0.4)
ax1.set_xlabel("Meta threshold", color="#5a7080")
ax1.set_title("Threshold Sweep (dashed = target lines)", color="#FFD700", fontsize=12)
ax1.legend(fontsize=8)
ax1.tick_params(colors="#5a7080")
for sp in ax1.spines.values(): sp.set_edgecolor("#1e2a3a")

ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor("#0d1117")
for r in results:
    tr = r["trades"].sort_values("time")
    if len(tr) == 0: continue
    eq = np.cumsum(tr["pnl"].values)
    ax2.plot(eq, linewidth=1.3, alpha=0.85,
             label=f"thr={r['thr']:.2f} WR={r['wr']:.0%} PF={r['pf']:.2f} n={r['n']}")
ax2.axhline(0, color="#444", linewidth=0.6)
ax2.legend(fontsize=7, loc="upper left")
ax2.set_title("Equity Curves by Threshold", color="#FFD700", fontsize=12)
ax2.tick_params(colors="#5a7080")
for sp in ax2.spines.values(): sp.set_edgecolor("#1e2a3a")

# AUC distribution
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_facecolor("#0d1117")
auc_vals = list(primary_aucs.values())
ax3.hist(auc_vals, bins=20, color="#10b981", alpha=0.7, edgecolor="white")
ax3.axvline(0.5, color="#666", linestyle="--", alpha=0.6)
ax3.axvline(np.median(auc_vals), color="#FFD700", linestyle="--", label=f"median={np.median(auc_vals):.3f}")
ax3.set_xlabel("Primary AUC", color="#5a7080")
ax3.set_title(f"Per-rule AUC distribution ({len(auc_vals)} rules)", color="#FFD700", fontsize=12)
ax3.legend(fontsize=9)
ax3.tick_params(colors="#5a7080")
for sp in ax3.spines.values(): sp.set_edgecolor("#1e2a3a")

# Summary
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor("#0d1117")
ax4.axis("off")
txt = f"Base WR (tight label): {setups['tight_label'].mean():.0%}\n"
txt += f"Meta AUC: {meta_auc:.3f}\n\n"
txt += "Threshold sweep:\n" + "─"*50 + "\n"
txt += f"{'thr':<6}{'n':>6}{'t/d':>6}{'WR':>6}{'PF':>6}{'PnL':>8}\n"
txt += "─"*50 + "\n"
for r in results:
    mark = "★" if (r["wr"]>=0.70 and r["pf"]>=1.50 and r["tpd"]>=5) else ""
    txt += f"{r['thr']:<6.2f}{r['n']:>6}{r['tpd']:>6.1f}{r['wr']:>6.0%}{r['pf']:>6.2f}{r['pnl']:>+8.1f} {mark}\n"
if best:
    txt += f"\n★ TARGET at thr={best['thr']}\n  WR={best['wr']:.0%} PF={best['pf']:.2f} {best['tpd']:.1f} t/d"
else:
    txt += "\nNo target hit."
ax4.text(0, 1, txt, fontsize=10, color="#ccc", family="monospace", va="top")

plt.suptitle("Midas v3 — Rule-gated + Tight TP + Meta-labeling",
             color="#FFD700", fontsize=16, fontweight="bold", y=1.0)
out = os.path.join(OUT_DIR, "midas_v3.png")
plt.savefig(out, dpi=140, bbox_inches="tight", facecolor="#080c12")
print(f"\nSaved: {out}")
