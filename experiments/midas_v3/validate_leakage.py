"""
Leakage validation for midas_v3.

Four tests:
  1. Label shuffle — retrain primary on shuffled y; AUC should → 0.5
  2. Feature importance inspection — any features that encode direction/outcome?
  3. Time-ordering check — is test strictly after train in wall-clock time?
  4. Walk-forward — train on rolling windows, test on next slice.

If #1 shows AUC >> 0.5 on shuffled labels → definitive leak.
"""
import os, json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

DATA_DIR = "/home/jay/Desktop/new-model-zigzag/data"
RAW_CSV = f"{DATA_DIR}/swing_v5_xauusd.csv"

CLUSTER_CFG = {
    0: {"tp": 0.7, "sl": 1.0, "T": 24},
    1: {"tp": 0.5, "sl": 1.0, "T": 12},
    2: {"tp": 0.8, "sl": 1.0, "T": 30},
    3: {"tp": 0.7, "sl": 1.0, "T": 24},
    4: {"tp": 1.0, "sl": 1.0, "T": 15},
}

# ── Load raw ──
print("Loading raw...")
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


# Load setups
parts = []
for cid in range(5):
    s = pd.read_csv(f"{DATA_DIR}/setups_{cid}.csv", parse_dates=["time"])
    s["cluster_id"] = cid
    parts.append(s)
setups = pd.concat(parts, ignore_index=True)
setups["idx"] = setups["time"].map(t2i)
setups = setups.dropna(subset=["idx"]).copy()
setups["idx"] = setups["idx"].astype(int)

# Compute label
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
setups = setups[setups["tight_label"] != -1].sort_values("time").reset_index(drop=True)

# Same split
n = len(setups)
train = setups.iloc[:int(n*0.70)].copy()
test = setups.iloc[int(n*0.80):].copy()

skip = {"time", "idx", "direction", "rule", "cluster_id", "tight_label"}
feat_cols = [c_ for c_ in setups.columns if c_ not in skip]

# ── TEST 1: Label shuffle ──
print("\n" + "="*70)
print("TEST 1: Label shuffle (AUC should → 0.5 after shuffling y)")
print("="*70)

# Pick rule with highest AUC in prior run as test case
target_rule = ("C4", "R2d_continuation")  # AUC 0.946 before
for (cid, rule) in [(4, "R2d_continuation"), (0, "R3d_oversold"), (3, "R1c_bouncefade")]:
    g_tr = train[(train["cluster_id"] == cid) & (train["rule"] == rule)]
    g_te = test[(test["cluster_id"] == cid) & (test["rule"] == rule)]
    if len(g_tr) < 100 or len(g_te) < 20: continue

    X_tr = g_tr[feat_cols].fillna(0).values
    X_te = g_te[feat_cols].fillna(0).values
    y_tr = g_tr["tight_label"].values
    y_te = g_te["tight_label"].values

    # Real
    m = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                      eval_metric="auc", verbosity=0)
    m.fit(X_tr, y_tr)
    auc_real = roc_auc_score(y_te, m.predict_proba(X_te)[:, 1]) if y_te.std() > 0 else 0.5

    # Shuffled labels
    y_sh = y_tr.copy()
    np.random.seed(42)
    np.random.shuffle(y_sh)
    m2 = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                       eval_metric="auc", verbosity=0)
    m2.fit(X_tr, y_sh)
    auc_sh = roc_auc_score(y_te, m2.predict_proba(X_te)[:, 1]) if y_te.std() > 0 else 0.5

    print(f"  C{cid} {rule:<25}: real AUC={auc_real:.3f}  shuffled AUC={auc_sh:.3f}  "
          f"{'⚠ LEAK' if abs(auc_sh-0.5)>0.05 else 'ok'}")


# ── TEST 2: Feature importance inspection ──
print("\n" + "="*70)
print("TEST 2: Top features (any suspicious?)")
print("="*70)
for (cid, rule) in [(4, "R2d_continuation"), (0, "R3d_oversold")]:
    g_tr = train[(train["cluster_id"] == cid) & (train["rule"] == rule)]
    if len(g_tr) < 100: continue
    X = g_tr[feat_cols].fillna(0).values
    y = g_tr["tight_label"].values
    m = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                      eval_metric="auc", verbosity=0)
    m.fit(X, y)
    imp = m.feature_importances_
    top10 = np.argsort(imp)[::-1][:10]
    print(f"\n  C{cid} {rule} — top 10 features:")
    for i in top10:
        print(f"    {feat_cols[i]:<20} importance={imp[i]:.4f}")


# ── TEST 3: Time ordering check ──
print("\n" + "="*70)
print("TEST 3: Time ordering — is test strictly after train?")
print("="*70)
print(f"  Train max time:  {train['time'].max()}")
print(f"  Test min time:   {test['time'].min()}")
print(f"  Gap: {(test['time'].min() - train['time'].max()).total_seconds()/3600:.1f} hours")
overlap = test[test['time'] <= train['time'].max()]
print(f"  Test rows at or before train max: {len(overlap)}  {'⚠' if len(overlap)>0 else 'ok'}")


# ── TEST 4: Forward-walk ──
print("\n" + "="*70)
print("TEST 4: Forward-walk (train on earlier, test on next month)")
print("="*70)
# Split by year-month: train on month N, test on month N+1
setups["ym"] = setups["time"].dt.to_period("M")
months = sorted(setups["ym"].unique())
print(f"  {len(months)} months in data; doing 6 rolling folds")

# For speed, do one rule across multiple months
for (cid, rule) in [(4, "R2d_continuation")]:
    g = setups[(setups["cluster_id"] == cid) & (setups["rule"] == rule)]
    print(f"\n  C{cid} {rule}: {len(g)} total setups")
    aucs = []
    for i in range(len(months) - 12, len(months) - 1, 2):  # 6 folds near end
        tr_months = months[max(0, i-24):i]  # ~2yr train window
        te_month = months[i]
        tr = g[g["ym"].isin(tr_months)]
        te = g[g["ym"] == te_month]
        if len(tr) < 50 or len(te) < 10 or te["tight_label"].nunique() < 2: continue
        m = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05,
                          eval_metric="auc", verbosity=0)
        m.fit(tr[feat_cols].fillna(0).values, tr["tight_label"].values)
        p = m.predict_proba(te[feat_cols].fillna(0).values)[:, 1]
        auc = roc_auc_score(te["tight_label"].values, p)
        aucs.append(auc)
        print(f"    Train end={tr_months[-1]} Test={te_month}  n_te={len(te)}  AUC={auc:.3f}")
    if aucs:
        print(f"\n  Mean forward-walk AUC: {np.mean(aucs):.3f}  (in-sample was 0.946)")
        print(f"  {'⚠ INFLATED' if np.mean(aucs) < 0.65 else 'looks real'}")
