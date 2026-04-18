"""
Regime v3 — rebalanced supervised classifier, drop-in for current K=5.

Lessons from v2:
  - ±0.3% thresholds made directional classes 1-3% → XGBoost collapsed
  - Volatility regime was predictable (AUC 0.82) — real signal
  - Persistence improved to 47h (vs 8h KMeans) — flicker fixed

v3 fixes:
  - Loosen thresholds so each class has >= 10% base rate
  - Balanced class weights (sample_weight ∝ 1/class_freq)
  - Same 5 cluster IDs (0-4) matching production router — drop-in replacement
  - Honest test: per-class AUC, confusion, forward-behavior match

Label rules (loosened):
  0 Uptrend    : fwd_ret > +0.15% AND trend_cons > 0.50
  1 MeanRevert : |fwd_ret| < 0.15% AND fwd_vol in bottom 50%
  2 TrendRange : fwd_vol in middle 50-80% AND |fwd_ret| < 0.25%
  3 Downtrend  : fwd_ret < -0.15% AND trend_cons > 0.50
  4 HighVol    : fwd_vol in top 20%
"""
import os, json, time
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

OUT_DIR = "/home/jay/Desktop/new-model-zigzag/experiments/regime_v3_rebalanced"
os.makedirs(OUT_DIR, exist_ok=True)

RAW_CSV = "/home/jay/Desktop/new-model-zigzag/data/swing_v5_xauusd.csv"
WINDOW = 288
STEP = 12

NAMES = {0: "Uptrend", 1: "MeanRevert", 2: "TrendRange", 3: "Downtrend", 4: "HighVol"}
COLORS = {0: "#f5c518", 1: "#3b82f6", 2: "#00E5FF", 3: "#ef4444", 4: "#a855f7"}

print("Loading XAUUSD M5...")
raw = pd.read_csv(RAW_CSV, parse_dates=["time"])
raw = raw[raw["time"] >= "2016-01-01"].sort_values("time").reset_index(drop=True)
c = raw["close"].values.astype(np.float64)
h = raw["high"].values.astype(np.float64)
l = raw["low"].values.astype(np.float64)
o = raw["open"].values.astype(np.float64)
n_bars = len(raw)
print(f"  {n_bars:,} bars")


def fingerprint(c_, h_, l_, o_):
    n = len(c_)
    if n < 10: return None
    r = np.diff(c_) / c_[:-1]
    br = (h_ - l_) / c_
    fp = np.zeros(7)
    fp[0] = r.sum(); fp[1] = r.std()
    mean_r = r.mean()
    fp[2] = np.mean(np.sign(r) == np.sign(mean_r)) if abs(mean_r) > 1e-12 else 0.5
    fp[3] = r.sum() / (r.std() + 1e-9)
    fp[4] = br.mean()
    total_range = (h_.max() - l_.min()) / c_.mean()
    fp[5] = total_range / (br.mean() + 1e-9)
    if len(r) > 2:
        denom = r[:-1].std() * r[1:].std()
        fp[6] = np.corrcoef(r[:-1], r[1:])[0, 1] if denom > 1e-12 else 0.0
    return fp


def forward(c_, h_, l_):
    n = len(c_)
    if n < 10: return None
    r = np.diff(c_) / c_[:-1]
    mean_r = r.mean()
    return {
        "fwd_ret_pct": r.sum() * 100,
        "fwd_vol_pct": r.std() * 100,
        "trend_cons": np.mean(np.sign(r) == np.sign(mean_r)) if abs(mean_r) > 1e-12 else 0.5,
    }


# ── Compute ──
print(f"\nComputing fingerprint + forward outcomes (step={STEP})...")
t0 = time.time()
records = []
for idx in range(WINDOW, n_bars - WINDOW, STEP):
    fp = fingerprint(c[idx-WINDOW:idx], h[idx-WINDOW:idx], l[idx-WINDOW:idx], o[idx-WINDOW:idx])
    if fp is None: continue
    fwd = forward(c[idx:idx+WINDOW], h[idx:idx+WINDOW], l[idx:idx+WINDOW])
    if fwd is None: continue
    records.append({"time": raw["time"].iloc[idx], "idx": idx,
                    "fp0": fp[0], "fp1": fp[1], "fp2": fp[2], "fp3": fp[3],
                    "fp4": fp[4], "fp5": fp[5], "fp6": fp[6], **fwd})
df = pd.DataFrame(records)
print(f"  {len(df):,} points in {time.time()-t0:.1f}s")


# ── Loose labels ──
vol_50 = df["fwd_vol_pct"].quantile(0.50)
vol_80 = df["fwd_vol_pct"].quantile(0.80)
vol_20 = df["fwd_vol_pct"].quantile(0.20)

labels = np.full(len(df), -1, dtype=np.int8)

for i in range(len(df)):
    r = df.iloc[i]
    # Priority 4: HighVol (overrides everything)
    if r["fwd_vol_pct"] >= vol_80:
        labels[i] = 4
    # Priority 0: Uptrend
    elif r["fwd_ret_pct"] > 0.15 and r["trend_cons"] > 0.50:
        labels[i] = 0
    # Priority 3: Downtrend
    elif r["fwd_ret_pct"] < -0.15 and r["trend_cons"] > 0.50:
        labels[i] = 3
    # Priority 1: MeanRevert (quiet sideways)
    elif abs(r["fwd_ret_pct"]) < 0.15 and r["fwd_vol_pct"] < vol_50:
        labels[i] = 1
    # Priority 2: TrendRange (mid-vol chop or small directional move)
    else:
        labels[i] = 2

df["label"] = labels

print("\nLabel distribution:")
for cls in range(5):
    sub = df[df["label"] == cls]
    pct = len(sub) / len(df) * 100
    print(f"  {cls} {NAMES[cls]:<12}: {pct:>5.1f}%  "
          f"fwd_ret={sub['fwd_ret_pct'].mean():+.3f}%  "
          f"fwd_vol={sub['fwd_vol_pct'].mean():.3f}%")


# ── Train balanced ──
print("\n" + "="*70)
print("TRAINING XGBoost multi-class (balanced sample weights)")
print("="*70)

feat_cols = [f"fp{i}" for i in range(7)]
df = df.sort_values("time").reset_index(drop=True)
s1, s2 = int(len(df)*0.70), int(len(df)*0.85)
X_tr, y_tr = df.iloc[:s1][feat_cols].values, df.iloc[:s1]["label"].values
X_vl, y_vl = df.iloc[s1:s2][feat_cols].values, df.iloc[s1:s2]["label"].values
X_te, y_te = df.iloc[s2:][feat_cols].values, df.iloc[s2:]["label"].values
print(f"  train={len(X_tr)}  val={len(X_vl)}  test={len(X_te)}")

# Balanced weights
w_tr = compute_sample_weight("balanced", y_tr)

model = XGBClassifier(
    n_estimators=600, max_depth=5, learning_rate=0.04,
    objective="multi:softprob", num_class=5,
    subsample=0.8, colsample_bytree=0.8,
    early_stopping_rounds=50, verbosity=0
)
model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_vl, y_vl)], verbose=False)

probs = model.predict_proba(X_te)
preds = probs.argmax(axis=1)

print(f"\n  Accuracy: {(preds == y_te).mean():.3f}")
print(f"\n  Per-class AUC:")
aucs = []
for cls in range(5):
    y_bin = (y_te == cls).astype(int)
    if y_bin.sum() < 10:
        aucs.append(0.5); continue
    a = roc_auc_score(y_bin, probs[:, cls])
    aucs.append(a)
    print(f"    {cls} {NAMES[cls]:<12}: AUC={a:.3f}  base={y_bin.mean():.0%}")

# Confusion matrix
cm = confusion_matrix(y_te, preds, labels=list(range(5)))
print(f"\n  Confusion matrix (rows=true, cols=pred):")
print(f"  {'':<14}" + " ".join(f"{NAMES[i][:7]:>8}" for i in range(5)))
for i in range(5):
    print(f"  {NAMES[i]:<14}" + " ".join(f"{cm[i][j]:>8}" for j in range(5)))


# ── Forward-behavior validation — THE KEY TEST ──
print("\n" + "="*70)
print("FORWARD-BEHAVIOR — does each predicted class match its label?")
print("="*70)

test_df = df.iloc[s2:].copy().reset_index(drop=True)
test_df["pred"] = preds
# Confidence margin
sorted_p = np.sort(probs, axis=1)
test_df["margin"] = sorted_p[:, -1] - sorted_p[:, -2]
test_df["max_prob"] = sorted_p[:, -1]

print(f"\n  {'Predicted':<14} {'n':>6} {'fwd_ret%':>10} {'fwd_vol%':>10} {'status'}")
print("  " + "-"*60)
verdicts = []
for cls in range(5):
    sub = test_df[test_df["pred"] == cls]
    if len(sub) == 0:
        print(f"  {NAMES[cls]:<14}: NEVER PREDICTED"); verdicts.append("✗"); continue
    ok = ""
    if cls == 0 and sub["fwd_ret_pct"].mean() > 0.05: ok = "✓"
    elif cls == 3 and sub["fwd_ret_pct"].mean() < -0.05: ok = "✓"
    elif cls == 4 and sub["fwd_vol_pct"].mean() > vol_80 * 0.9: ok = "✓"
    elif cls == 1 and sub["fwd_vol_pct"].mean() < vol_50: ok = "✓"
    elif cls == 2: ok = "~"  # no strong criterion
    else: ok = "⚠"
    verdicts.append(ok)
    print(f"  {NAMES[cls]:<14}: {len(sub):>6}  {sub['fwd_ret_pct'].mean():>+10.3f}  {sub['fwd_vol_pct'].mean():>10.3f}  {ok}")


# High-confidence subset
hc = test_df[test_df["margin"] > 0.2]
print(f"\n  High-confidence predictions (margin > 0.2): {len(hc)}/{len(test_df)} = {len(hc)/len(test_df)*100:.0f}%")
for cls in range(5):
    sub = hc[hc["pred"] == cls]
    if len(sub) == 0: continue
    print(f"    {NAMES[cls]:<14}: n={len(sub)}  fwd_ret={sub['fwd_ret_pct'].mean():+.3f}%  vol={sub['fwd_vol_pct'].mean():.3f}%")


# Persistence
print("\n" + "="*70)
print("PERSISTENCE (vs KMeans median 3h, v2 supervised median 4h)")
print("="*70)
seq = test_df["pred"].values
flips = np.where(np.diff(seq) != 0)[0]
if len(flips) > 1:
    runs = np.diff(np.concatenate([[0], flips, [len(seq)]]))
    print(f"  Mean: {runs.mean():.1f} points ({runs.mean()*STEP/12:.1f} hours)")
    print(f"  Median: {np.median(runs):.1f}")
    print(f"  % runs ≤ 2: {(runs<=2).mean()*100:.0f}%")


# ── Plots ──
fig = plt.figure(figsize=(22, 14), facecolor="#080c12")
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

# Label distribution
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor("#0d1117")
counts = [(df["label"] == c).sum() for c in range(5)]
ax1.bar(range(5), counts, color=[COLORS[c] for c in range(5)])
ax1.set_xticks(range(5)); ax1.set_xticklabels([NAMES[c] for c in range(5)], color="#ccc")
ax1.set_title("Label distribution (balanced thresholds)", color="#FFD700", fontsize=12)
ax1.tick_params(colors="#5a7080")
for sp in ax1.spines.values(): sp.set_edgecolor("#1e2a3a")

# AUC bars
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor("#0d1117")
ax2.bar(range(5), aucs, color=[COLORS[c] for c in range(5)])
ax2.axhline(0.5, color="#666", linestyle="--", alpha=0.5)
ax2.axhline(0.6, color="#FFD700", linestyle="--", alpha=0.4, label="target 0.6")
ax2.set_xticks(range(5)); ax2.set_xticklabels([NAMES[c] for c in range(5)], color="#ccc")
ax2.set_ylabel("AUC", color="#ccc"); ax2.set_title("Per-class test AUC", color="#FFD700", fontsize=12)
ax2.legend(fontsize=8); ax2.tick_params(colors="#5a7080")
for sp in ax2.spines.values(): sp.set_edgecolor("#1e2a3a")

# Forward return per predicted class (ALL)
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_facecolor("#0d1117")
for cls in range(5):
    sub = test_df[test_df["pred"] == cls]
    if len(sub) == 0: continue
    ax3.bar(cls, sub["fwd_ret_pct"].mean(), color=COLORS[cls],
            yerr=sub["fwd_ret_pct"].std()/np.sqrt(len(sub)))
ax3.axhline(0, color="#666"); ax3.set_xticks(range(5))
ax3.set_xticklabels([NAMES[c] for c in range(5)], color="#ccc")
ax3.set_ylabel("Mean fwd return %", color="#ccc")
ax3.set_title("Forward return per predicted class — ALL TEST", color="#FFD700", fontsize=12)
ax3.tick_params(colors="#5a7080")
for sp in ax3.spines.values(): sp.set_edgecolor("#1e2a3a")

# Forward return per predicted class (high-conf)
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor("#0d1117")
for cls in range(5):
    sub = hc[hc["pred"] == cls]
    if len(sub) == 0: continue
    ax4.bar(cls, sub["fwd_ret_pct"].mean(), color=COLORS[cls],
            yerr=sub["fwd_ret_pct"].std()/np.sqrt(max(len(sub),1)))
ax4.axhline(0, color="#666"); ax4.set_xticks(range(5))
ax4.set_xticklabels([NAMES[c] for c in range(5)], color="#ccc")
ax4.set_ylabel("Mean fwd return %", color="#ccc")
ax4.set_title("Forward return — HIGH CONFIDENCE (margin > 0.2)", color="#FFD700", fontsize=12)
ax4.tick_params(colors="#5a7080")
for sp in ax4.spines.values(): sp.set_edgecolor("#1e2a3a")

# Confusion
ax5 = fig.add_subplot(gs[2, 0])
ax5.set_facecolor("#0d1117")
im = ax5.imshow(cm, cmap="Blues")
ax5.set_xticks(range(5)); ax5.set_xticklabels([NAMES[c][:7] for c in range(5)], color="#ccc")
ax5.set_yticks(range(5)); ax5.set_yticklabels([NAMES[c][:7] for c in range(5)], color="#ccc")
ax5.set_xlabel("Predicted", color="#ccc"); ax5.set_ylabel("True", color="#ccc")
ax5.set_title("Confusion matrix", color="#FFD700", fontsize=12)
for i in range(5):
    for j in range(5):
        ax5.text(j, i, str(cm[i][j]), ha="center", va="center",
                 color="white" if cm[i][j] > cm.max()/2 else "black", fontsize=8)
plt.colorbar(im, ax=ax5)

# Summary
ax6 = fig.add_subplot(gs[2, 1])
ax6.set_facecolor("#0d1117")
ax6.axis("off")
txt = "Regime v3 — Rebalanced Supervised\n" + "═"*50 + "\n\n"
txt += f"Accuracy: {(preds == y_te).mean():.1%}\n\n"
txt += "Per-class AUC:\n"
for cls in range(5):
    txt += f"  {cls} {NAMES[cls]:<12} {aucs[cls]:.3f}\n"
txt += "\nFwd return per pred class (status):\n"
for cls in range(5):
    sub = test_df[test_df["pred"] == cls]
    if len(sub) == 0: continue
    txt += f"  {cls} {NAMES[cls]:<12} {sub['fwd_ret_pct'].mean():+.3f}% {verdicts[cls]}\n"
if len(flips) > 1:
    txt += f"\nPersistence median: {np.median(runs):.0f} points"
    txt += f"\n(KMeans=3, v2=4, now {np.median(runs):.0f})"
ax6.text(0, 1, txt, fontsize=10, color="#ccc", family="monospace", va="top")

plt.suptitle("Regime v3 — Rebalanced Supervised Classifier",
             color="#FFD700", fontsize=16, fontweight="bold", y=0.995)
out = os.path.join(OUT_DIR, "regime_v3.png")
plt.savefig(out, dpi=140, bbox_inches="tight", facecolor="#080c12")
print(f"\nSaved: {out}")

model.save_model(os.path.join(OUT_DIR, "regime_v3_classifier.json"))
with open(os.path.join(OUT_DIR, "regime_v3_meta.json"), "w") as f:
    json.dump({
        "names": NAMES, "window": WINDOW, "step": STEP, "feat_cols": feat_cols,
        "test_accuracy": float((preds == y_te).mean()),
        "per_class_auc": {NAMES[c]: float(aucs[c]) for c in range(5)},
        "persistence_median_hours": float(np.median(runs) * STEP / 12) if len(flips) > 1 else None,
        "vol_thresholds": {"vol_20": float(vol_20), "vol_50": float(vol_50), "vol_80": float(vol_80)},
    }, f, indent=2)
print(f"Saved: regime_v3_classifier.json + meta.json")
