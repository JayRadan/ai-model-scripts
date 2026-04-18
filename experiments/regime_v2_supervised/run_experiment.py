"""
Regime v2 — supervised clusterer.

Current K=5 KMeans defines clusters by fingerprint centroids. Problem:
audit showed C3 "Downtrend" had +0.05% forward return (wrong-labeled),
and classifier flickered every 3 hours.

New approach: DEFINE clusters by forward outcomes (the truth), then train
a classifier to predict them from current state only.

Labels (defined by next-288-bar behavior):
  Bullish     : fwd_ret > +0.3% AND trend_consistency > 0.55
  Bearish     : fwd_ret < -0.3% AND trend_consistency > 0.55
  MeanRevert  : |fwd_ret| < 0.25% AND fwd_autocorr < -0.02
  Breakout    : fwd_vol in top 25% AND |fwd_ret| > 0.4%
  Quiet       : everything else

Classifier: XGBoost multi-class on fingerprint features (same 7 as today).
Output: per-class AUC, confusion matrix, probability calibration curve.

Success criteria: per-class AUC > 0.60, Bearish class forward return
must be negative (that's the old C3 problem).
"""
import os, json, time
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

OUT_DIR = "/home/jay/Desktop/new-model-zigzag/experiments/regime_v2_supervised"
os.makedirs(OUT_DIR, exist_ok=True)

RAW_CSV = "/home/jay/Desktop/new-model-zigzag/data/swing_v5_xauusd.csv"
WINDOW = 288     # match current selector
STEP = 12        # hourly granularity — same as audit


CLASS_NAMES = {0: "Bullish", 1: "Bearish", 2: "MeanRevert", 3: "Breakout", 4: "Quiet"}
CLASS_COLORS = {0: "#10b981", 1: "#ef4444", 2: "#3b82f6", 3: "#FFD700", 4: "#6b7280"}


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
    """Same 7 features as production selector — computed on past window."""
    n = len(c_)
    if n < 10: return None
    r = np.diff(c_) / c_[:-1]
    br = (h_ - l_) / c_
    fp = np.zeros(7)
    fp[0] = r.sum()
    fp[1] = r.std()
    mean_r = r.mean()
    fp[2] = np.mean(np.sign(r) == np.sign(mean_r)) if abs(mean_r) > 1e-12 else 0.5
    fp[3] = r.sum() / (r.std() + 1e-9)
    fp[4] = br.mean()
    total_range = (h_.max() - l_.min()) / c_.mean()
    fp[5] = total_range / (br.mean() + 1e-9)
    if len(r) > 2:
        r1, r2 = r[:-1], r[1:]
        denom = r1.std() * r2.std()
        fp[6] = np.corrcoef(r1, r2)[0, 1] if denom > 1e-12 else 0.0
    return fp


def forward_features(c_, h_, l_):
    """Forward-outcome features — used to define labels, not features."""
    n = len(c_)
    if n < 10: return None
    r = np.diff(c_) / c_[:-1]
    br = (h_ - l_) / c_
    mean_r = r.mean()
    trend_consistency = np.mean(np.sign(r) == np.sign(mean_r)) if abs(mean_r) > 1e-12 else 0.5
    # autocorr
    if len(r) > 2:
        denom = r[:-1].std() * r[1:].std()
        ac = np.corrcoef(r[:-1], r[1:])[0, 1] if denom > 1e-12 else 0.0
    else:
        ac = 0.0
    return {
        "fwd_ret_pct": r.sum() * 100,
        "fwd_vol_pct": r.std() * 100,
        "fwd_trend_cons": trend_consistency,
        "fwd_trend_sign": np.sign(mean_r),
        "fwd_autocorr": ac,
        "fwd_br_mean": br.mean(),
    }


# ── Compute fingerprint + forward outcomes at every STEP bar ──
print(f"\nComputing fingerprints + forward outcomes (step={STEP} bars)...")
t0 = time.time()
records = []
for idx in range(WINDOW, n_bars - WINDOW, STEP):
    past_start = idx - WINDOW
    fp = fingerprint(c[past_start:idx], h[past_start:idx], l[past_start:idx], o[past_start:idx])
    if fp is None: continue
    fwd = forward_features(c[idx:idx+WINDOW], h[idx:idx+WINDOW], l[idx:idx+WINDOW])
    if fwd is None: continue
    records.append({
        "time": raw["time"].iloc[idx], "idx": idx,
        "fp0": fp[0], "fp1": fp[1], "fp2": fp[2], "fp3": fp[3],
        "fp4": fp[4], "fp5": fp[5], "fp6": fp[6],
        **fwd,
    })

df = pd.DataFrame(records)
print(f"  {len(df):,} points computed in {time.time()-t0:.1f}s")


# ── Define behavioral labels based on forward outcomes ──
print("\nDefining behavioral labels from forward outcomes...")
labels = np.full(len(df), 4, dtype=np.int8)  # default Quiet

# Determine vol cutoff
vol_top25 = df["fwd_vol_pct"].quantile(0.75)

for i in range(len(df)):
    r = df.iloc[i]
    if r["fwd_ret_pct"] > 0.3 and r["fwd_trend_cons"] > 0.55:
        labels[i] = 0  # Bullish
    elif r["fwd_ret_pct"] < -0.3 and r["fwd_trend_cons"] > 0.55:
        labels[i] = 1  # Bearish
    elif abs(r["fwd_ret_pct"]) < 0.25 and r["fwd_autocorr"] < -0.02:
        labels[i] = 2  # MeanRevert
    elif r["fwd_vol_pct"] >= vol_top25 and abs(r["fwd_ret_pct"]) > 0.4:
        labels[i] = 3  # Breakout
    # else 4 Quiet

df["label"] = labels

print("\nLabel distribution:")
for cls in range(5):
    pct = (labels == cls).mean() * 100
    sub = df[df["label"] == cls]
    print(f"  {cls} {CLASS_NAMES[cls]:<11}: {pct:>5.1f}%  "
          f"fwd_ret={sub['fwd_ret_pct'].mean():+.3f}%  "
          f"fwd_vol={sub['fwd_vol_pct'].mean():.3f}%  "
          f"ac={sub['fwd_autocorr'].mean():+.3f}")


# ── Train supervised classifier ──
print("\n" + "="*70)
print("TRAINING supervised regime classifier (XGBoost multi-class)")
print("="*70)

feat_cols = [f"fp{i}" for i in range(7)]
df = df.sort_values("time").reset_index(drop=True)
split = int(len(df) * 0.70)
val_split = int(len(df) * 0.85)

X_train = df.iloc[:split][feat_cols].values
y_train = df.iloc[:split]["label"].values
X_val = df.iloc[split:val_split][feat_cols].values
y_val = df.iloc[split:val_split]["label"].values
X_test = df.iloc[val_split:][feat_cols].values
y_test = df.iloc[val_split:]["label"].values
print(f"  train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")

model = XGBClassifier(
    n_estimators=500, max_depth=5, learning_rate=0.05,
    objective="multi:softprob", num_class=5,
    subsample=0.8, colsample_bytree=0.8,
    early_stopping_rounds=40, verbosity=0
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# Test performance
probs_test = model.predict_proba(X_test)
preds_test = probs_test.argmax(axis=1)
print(f"\n  Accuracy: {(preds_test == y_test).mean():.3f}")
print(f"\n  Per-class AUC (one-vs-rest):")
for cls in range(5):
    y_bin = (y_test == cls).astype(int)
    if y_bin.sum() < 5:
        print(f"    {cls} {CLASS_NAMES[cls]:<11}: insufficient samples")
        continue
    auc = roc_auc_score(y_bin, probs_test[:, cls])
    print(f"    {cls} {CLASS_NAMES[cls]:<11}: AUC={auc:.3f}  base={y_bin.mean():.0%}")


# Confusion matrix
cm = confusion_matrix(y_test, preds_test, labels=list(range(5)))
print(f"\n  Confusion matrix (rows=true, cols=pred):")
print(f"  {'':<12} " + " ".join(f"{CLASS_NAMES[i][:7]:>8}" for i in range(5)))
for i in range(5):
    row = f"  {CLASS_NAMES[i]:<12} "
    row += " ".join(f"{cm[i][j]:>8}" for j in range(5))
    print(row)


# ── Forward-behavior validation: predicted class vs actual outcomes ──
print("\n" + "="*70)
print("FORWARD-BEHAVIOR VALIDATION (did predicted class match reality?)")
print("="*70)

test_df = df.iloc[val_split:].copy().reset_index(drop=True)
test_df["pred"] = preds_test
test_df["prob_max"] = probs_test.max(axis=1)
# Margin = best prob - 2nd best prob
sorted_probs = np.sort(probs_test, axis=1)
test_df["margin"] = sorted_probs[:, -1] - sorted_probs[:, -2]

print(f"\n  {'Predicted class':<15} {'n':>6} {'fwd_ret%':>10} {'fwd_vol%':>10} {'autocorr':>10}")
print("  " + "-"*58)
for cls in range(5):
    sub = test_df[test_df["pred"] == cls]
    if len(sub) == 0: continue
    ok = ""
    if cls == 0 and sub["fwd_ret_pct"].mean() > 0.1: ok = "✓"
    if cls == 1 and sub["fwd_ret_pct"].mean() < -0.1: ok = "✓"
    if cls == 2 and sub["fwd_autocorr"].mean() < 0: ok = "✓"
    print(f"  {CLASS_NAMES[cls]:<13}: {len(sub):>6}  {sub['fwd_ret_pct'].mean():>+10.3f}  {sub['fwd_vol_pct'].mean():>10.3f}  {sub['fwd_autocorr'].mean():>+10.3f}  {ok}")


# High-confidence subset
print("\n  High-confidence predictions (margin > 0.3):")
hc = test_df[test_df["margin"] > 0.3]
print(f"    {len(hc):,}/{len(test_df):,} = {len(hc)/len(test_df)*100:.0f}%")
for cls in range(5):
    sub = hc[hc["pred"] == cls]
    if len(sub) == 0: continue
    print(f"    {CLASS_NAMES[cls]:<13}: n={len(sub)}  fwd_ret={sub['fwd_ret_pct'].mean():+.3f}%  fwd_ac={sub['fwd_autocorr'].mean():+.3f}")


# ── Persistence of predicted class (compare to old KMeans) ──
seq = test_df["pred"].values
flips = np.where(np.diff(seq) != 0)[0]
if len(flips) > 1:
    runs = np.diff(np.concatenate([[0], flips, [len(seq)]]))
    print(f"\n  Persistence of predicted class:")
    print(f"    Mean run length: {runs.mean():.1f} points ({runs.mean()*STEP/12:.1f} hours)")
    print(f"    Median run length: {np.median(runs):.1f}")
    print(f"    % runs ≤ 2: {(runs<=2).mean()*100:.0f}%  (vs. 45% for KMeans)")


# ── Plots ──
fig = plt.figure(figsize=(22, 14), facecolor="#080c12")
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

# (1) Label distribution
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor("#0d1117")
counts = [(df["label"] == c).sum() for c in range(5)]
ax1.bar(range(5), counts, color=[CLASS_COLORS[c] for c in range(5)])
ax1.set_xticks(range(5))
ax1.set_xticklabels([CLASS_NAMES[c] for c in range(5)], color="#ccc")
ax1.set_title("Behavioral label distribution (ground truth)", color="#FFD700", fontsize=12)
ax1.tick_params(colors="#5a7080")
for sp in ax1.spines.values(): sp.set_edgecolor("#1e2a3a")

# (2) Per-class AUC bars
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor("#0d1117")
aucs = []
for cls in range(5):
    y_bin = (y_test == cls).astype(int)
    if y_bin.sum() < 5: aucs.append(0); continue
    aucs.append(roc_auc_score(y_bin, probs_test[:, cls]))
ax2.bar(range(5), aucs, color=[CLASS_COLORS[c] for c in range(5)])
ax2.axhline(0.5, color="#666", linestyle="--", alpha=0.5)
ax2.axhline(0.6, color="#FFD700", linestyle="--", alpha=0.4, label="0.6 target")
ax2.set_xticks(range(5))
ax2.set_xticklabels([CLASS_NAMES[c] for c in range(5)], color="#ccc")
ax2.set_ylabel("AUC", color="#ccc")
ax2.set_title("Per-class test AUC (one-vs-rest)", color="#FFD700", fontsize=12)
ax2.legend(fontsize=8)
ax2.tick_params(colors="#5a7080")
for sp in ax2.spines.values(): sp.set_edgecolor("#1e2a3a")

# (3) Forward-return per predicted class (the crux)
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_facecolor("#0d1117")
for cls in range(5):
    sub = test_df[test_df["pred"] == cls]
    if len(sub) == 0: continue
    ax3.bar(cls, sub["fwd_ret_pct"].mean(),
            color=CLASS_COLORS[cls],
            yerr=sub["fwd_ret_pct"].std() / np.sqrt(len(sub)))
ax3.axhline(0, color="#666", linewidth=0.6)
ax3.set_xticks(range(5))
ax3.set_xticklabels([CLASS_NAMES[c] for c in range(5)], color="#ccc")
ax3.set_ylabel("Mean forward return %", color="#ccc")
ax3.set_title("Forward return by predicted class (all test)", color="#FFD700", fontsize=12)
ax3.tick_params(colors="#5a7080")
for sp in ax3.spines.values(): sp.set_edgecolor("#1e2a3a")

# (4) Forward-return per predicted class (high-conf only)
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor("#0d1117")
for cls in range(5):
    sub = hc[hc["pred"] == cls]
    if len(sub) == 0: continue
    ax4.bar(cls, sub["fwd_ret_pct"].mean(),
            color=CLASS_COLORS[cls],
            yerr=sub["fwd_ret_pct"].std() / np.sqrt(max(len(sub), 1)))
ax4.axhline(0, color="#666", linewidth=0.6)
ax4.set_xticks(range(5))
ax4.set_xticklabels([CLASS_NAMES[c] for c in range(5)], color="#ccc")
ax4.set_ylabel("Mean forward return %", color="#ccc")
ax4.set_title("Forward return — HIGH CONFIDENCE (margin > 0.3)", color="#FFD700", fontsize=12)
ax4.tick_params(colors="#5a7080")
for sp in ax4.spines.values(): sp.set_edgecolor("#1e2a3a")

# (5) Confusion matrix
ax5 = fig.add_subplot(gs[2, 0])
ax5.set_facecolor("#0d1117")
im = ax5.imshow(cm, cmap="Blues", aspect="auto")
ax5.set_xticks(range(5)); ax5.set_xticklabels([CLASS_NAMES[c][:7] for c in range(5)], color="#ccc")
ax5.set_yticks(range(5)); ax5.set_yticklabels([CLASS_NAMES[c][:7] for c in range(5)], color="#ccc")
ax5.set_xlabel("Predicted", color="#ccc"); ax5.set_ylabel("True", color="#ccc")
ax5.set_title("Confusion matrix (test set)", color="#FFD700", fontsize=12)
for i in range(5):
    for j in range(5):
        ax5.text(j, i, str(cm[i][j]), ha="center", va="center",
                 color="white" if cm[i][j] > cm.max()/2 else "black", fontsize=9)
plt.colorbar(im, ax=ax5)

# (6) Summary
ax6 = fig.add_subplot(gs[2, 1])
ax6.set_facecolor("#0d1117")
ax6.axis("off")
txt = "Supervised Regime Classifier — Test Set\n"
txt += "═" * 50 + "\n\n"
txt += f"Accuracy:     {(preds_test == y_test).mean():.1%}\n"
txt += f"Samples:      {len(y_test):,}\n\n"
txt += "Per-class AUC:\n"
for cls in range(5):
    txt += f"  {CLASS_NAMES[cls]:<11} {aucs[cls]:.3f}\n"
txt += "\nForward return per pred class:\n"
for cls in range(5):
    sub = test_df[test_df["pred"] == cls]
    if len(sub) == 0: continue
    txt += f"  {CLASS_NAMES[cls]:<11} {sub['fwd_ret_pct'].mean():+.3f}%  (n={len(sub)})\n"
if len(flips) > 1:
    txt += f"\nMedian run length: {np.median(runs):.1f} (vs. 3 for KMeans)"
ax6.text(0, 1, txt, fontsize=10, color="#ccc", family="monospace", va="top")

plt.suptitle("Regime v2 — Supervised Classifier (forward-outcome labels)",
             color="#FFD700", fontsize=16, fontweight="bold", y=0.995)
out = os.path.join(OUT_DIR, "regime_v2.png")
plt.savefig(out, dpi=140, bbox_inches="tight", facecolor="#080c12")
print(f"\nSaved: {out}")

# Save model + scaler info for later export to MQL5
model.save_model(os.path.join(OUT_DIR, "regime_v2_classifier.json"))
with open(os.path.join(OUT_DIR, "regime_v2_meta.json"), "w") as f:
    json.dump({
        "class_names": CLASS_NAMES,
        "window": WINDOW, "step": STEP,
        "feat_cols": feat_cols,
        "label_rules": {
            "Bullish": "fwd_ret>+0.3% AND trend_cons>0.55",
            "Bearish": "fwd_ret<-0.3% AND trend_cons>0.55",
            "MeanRevert": "|fwd_ret|<0.25% AND autocorr<-0.02",
            "Breakout": "fwd_vol top25% AND |fwd_ret|>0.4%",
            "Quiet": "otherwise",
        },
        "test_accuracy": float((preds_test == y_test).mean()),
        "per_class_auc": {CLASS_NAMES[c]: float(aucs[c]) for c in range(5)},
    }, f, indent=2)
print(f"Saved: {os.path.join(OUT_DIR, 'regime_v2_classifier.json')}")
