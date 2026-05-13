"""Train XGB classifier on current-bar features → forward-12h regime label.

Honest holdout split at 2024-12-12 (same as production gate).

Key metric: not raw accuracy (5-class problem ~20% baseline), but
directional precision: when model says Uptrend, how often does
forward return go UP? That's what we'd use as a /decide gate.
"""
import os, numpy as np, pandas as pd, pickle
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

PROJECT = "/home/jay/Desktop/new-model-zigzag"
OUT = f"{PROJECT}/experiments/v92_supervised_regime"
HOLDOUT = pd.Timestamp("2024-12-12")
NAMES = {0:'Uptrend',1:'MeanRevert',2:'TrendRange',3:'Downtrend',4:'HighVol'}

def train(name):
    print(f"\n{'='*72}\n  {name}\n{'='*72}")
    df = pd.read_parquet(f"{OUT}/labeled_{name.lower()}.parquet")
    df['time'] = pd.to_datetime(df['time'])
    feat_cols = [c for c in df.columns if c not in ('time','label','fwd_ret','fwd_vol')]
    print(f"  feature cols ({len(feat_cols)}): {feat_cols}")

    train_m = df['time'] < HOLDOUT
    test_m  = df['time'] >= HOLDOUT
    Xtr, ytr = df.loc[train_m, feat_cols].values, df.loc[train_m,'label'].values
    Xte, yte = df.loc[test_m,  feat_cols].values, df.loc[test_m, 'label'].values
    fwd_ret_te = df.loc[test_m, 'fwd_ret'].values
    print(f"  train: {len(ytr):,}   test: {len(yte):,}")

    mdl = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        objective='multi:softprob', num_class=5,
        tree_method='hist', n_jobs=-1, eval_metric='mlogloss',
        early_stopping_rounds=20)
    mdl.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)

    proba = mdl.predict_proba(Xte)
    pred  = proba.argmax(axis=1)
    print(f"\n  HOLDOUT accuracy: {(pred==yte).mean()*100:.1f}%")

    print("\n  classification report (test):")
    print(classification_report(yte, pred, target_names=[NAMES[i] for i in range(5)], digits=3))

    cm = confusion_matrix(yte, pred)
    print("  confusion matrix (rows=true, cols=pred):")
    print("       ", "  ".join(f"{NAMES[i][:5]:>5}" for i in range(5)))
    for i in range(5):
        print(f"  {NAMES[i][:5]:>5} ", "  ".join(f"{cm[i,j]:>5d}" for j in range(5)))

    # Directional precision: gating utility
    print("\n  GATING UTILITY (directional precision at confidence thresholds):")
    print(f"  {'thr':>5}  {'pred=Up':>20}  {'pred=Down':>22}")
    for thr in [0.30, 0.40, 0.50, 0.60]:
        up_m   = (pred == 0) & (proba[:,0] >= thr)
        dn_m   = (pred == 3) & (proba[:,3] >= thr)
        up_acc = (fwd_ret_te[up_m] > 0).mean() if up_m.sum() else float('nan')
        dn_acc = (fwd_ret_te[dn_m] < 0).mean() if dn_m.sum() else float('nan')
        up_R   = fwd_ret_te[up_m].mean()*100 if up_m.sum() else float('nan')
        dn_R   = fwd_ret_te[dn_m].mean()*100 if dn_m.sum() else float('nan')
        print(f"  {thr:>5.2f}  N={up_m.sum():>5}  {up_acc*100:>4.1f}% acc  mean={up_R:>+5.2f}%   "
              f"N={dn_m.sum():>5}  {dn_acc*100:>4.1f}% acc  mean={dn_R:>+5.2f}%")

    out_path = f"{OUT}/regime_clf_{name.lower()}.pkl"
    with open(out_path, 'wb') as f:
        pickle.dump({'model': mdl, 'features': feat_cols}, f)
    print(f"\n  saved {out_path}")

if __name__ == "__main__":
    train("XAU")
    train("BTC")
