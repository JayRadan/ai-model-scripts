"""
v8.0 — Second meta gate trained on the first meta gate's outcomes,
with enriched feature set the first gate doesn't see.

Hypothesis: the first meta gate (XGB on features+direction+cid) already
saturates its input space. A second gate can only find new signal if
it sees inputs the first didn't. We add:
  - rule identity (one-hot or label-encoded)
  - p_conf — Stage-1 confirm head's probability for this trade

Method:
  1. Re-simulate Oracle holdout to recover p_conf per trade.
  2. Train second-gate XGB on Midas pre-2024-12-12 (no holdout leakage).
     Inputs: existing meta_feats + rule + p_conf.
     Label: pnl_R > 0.
  3. Apply to Oracle holdout (cross-product transfer test) AND to Midas
     holdout. Sweep gate threshold; report ΔWR, ΔR, ΔPF.
  4. Permutation null + walk-forward H1/H2.

Pass bar: ΔWR ≥ +2pp on holdout AND ΔR not worse than -3% AND walk-forward
H2 confirms the kill.
"""
from __future__ import annotations
import os, glob, pickle
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

ROOT = "/home/jay/Desktop/new-model-zigzag"
PKL_O = "/home/jay/Desktop/my-agents-and-website/commercial/server/decision_engine/models/oracle_xau_validated.pkl"

HOLDOUT_START = pd.Timestamp("2024-12-12")
RNG = np.random.default_rng(0)


def pf(rs):
    pos = rs[rs > 0].sum(); neg = -rs[rs <= 0].sum()
    return pos / max(neg, 1e-9)


def load_oracle_pickle():
    return pickle.load(open(PKL_O, "rb"))


def attach_pconf_and_features(trades, mdls, thrs, V72L_FEATS):
    """For each trade, look up the (cid, rule) confirm model and recompute
    p_conf from the V72L feature row at fire time. Also attach the V72L
    features themselves. Returns enriched df + the feature column list."""
    setups = []
    for f in sorted(glob.glob(os.path.join(ROOT, "data/setups_*_v72l.csv"))):
        cid = int(os.path.basename(f).split("_")[1])
        df = pd.read_csv(f, parse_dates=["time"])
        df["cid"] = cid
        setups.append(df)
    setups = pd.concat(setups, ignore_index=True)
    setups["cid"] = setups["cid"].astype(int)
    setups["direction"] = setups["direction"].astype(int)

    trades = trades.copy()
    trades["cid"] = trades["cid"].astype(int)
    trades["direction"] = trades["direction"].astype(int)
    merged = trades.merge(
        setups[["time", "cid", "rule", "direction"] + list(V72L_FEATS)],
        on=["time", "cid", "rule", "direction"], how="left")
    merged = merged.dropna(subset=list(V72L_FEATS)).reset_index(drop=True)

    # Recompute p_conf per row using each row's (cid, rule) model.
    p_conf = np.full(len(merged), np.nan)
    for (cid, rule), grp in merged.groupby(["cid", "rule"]):
        if (cid, rule) not in mdls: continue
        X = grp[list(V72L_FEATS)].fillna(0).values
        p = mdls[(cid, rule)].predict_proba(X)[:, 1]
        p_conf[grp.index.values] = p
    merged["p_conf"] = p_conf
    merged = merged.dropna(subset=["p_conf"]).reset_index(drop=True)
    return merged


def evaluate_gate(name, trades, p, base_label=""):
    rs = trades["pnl_R"].values
    base_pf = pf(rs); base_r = rs.sum(); base_wr = (rs > 0).mean(); n = len(rs)
    print(f"\n[{name}{base_label}] n={n} WR={base_wr:.1%} PF={base_pf:.2f} R={base_r:+.1f}")
    corr, _ = pearsonr(p, rs)
    print(f"  pearson(p_2nd, pnl_R) = {corr:+.4f}")
    rows = []
    for thr in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
        keep = p >= thr
        if keep.sum() < 30: continue
        rsk = rs[keep]
        kept_wr = (rsk > 0).mean()
        rows.append({
            "thr": thr, "kept": int(keep.sum()),
            "WR": round(kept_wr, 3),
            "PF": round(pf(rsk), 2),
            "R":  round(rsk.sum(), 1),
            "ΔWR_pp": round((kept_wr - base_wr) * 100, 1),
            "ΔPF":    round(pf(rsk) - base_pf, 2),
            "ΔR":     round(rsk.sum() - base_r, 1),
        })
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))


def main():
    obj = load_oracle_pickle()
    mdls       = obj["mdls"]
    META_FEATS = obj["meta_feats"]
    V72L_FEATS = obj["v72l_feats"]
    print(f"loaded Oracle pickle: {len(mdls)} confirm heads, "
          f"{len(META_FEATS)} meta feats, {len(V72L_FEATS)} v72l feats")

    # ----- Use Midas holdout 2024-01-02 → 2024-12-11 as TRAIN.
    # Midas trades pre-holdout overlap with Oracle's pre-holdout window,
    # giving us 2024 trades for second-gate training without using
    # any of Oracle's holdout.
    print("\nloading Midas v6 trades (training set: 2024-01-02 → 2024-12-11)")
    midas = pd.read_csv(os.path.join(ROOT, "data/v6_trades_holdout_xau.csv"),
                          parse_dates=["time"])
    # Attach Oracle features + Oracle p_conf (Oracle's confirm models work
    # on XAU bars regardless; Midas trades are also XAU). This gives our
    # second-gate the same input space as Oracle uses.
    midas_full = attach_pconf_and_features(midas, mdls, thrs=None, V72L_FEATS=V72L_FEATS)
    print(f"  Midas after feature merge: {len(midas_full)} (need (cid,rule) match)")
    train = midas_full[midas_full["time"] < HOLDOUT_START].copy()
    print(f"  train n={len(train)} WR={(train.pnl_R>0).mean():.3f}")

    # Build feature matrix: V72L feats + direction + cid + p_conf + rule_id
    rule_codes = pd.factorize(train["rule"])
    train["rule_id"] = rule_codes[0]
    rule_map = {r: i for i, r in enumerate(rule_codes[1])}

    enriched_feats = list(V72L_FEATS) + ["direction", "cid", "p_conf", "rule_id"]
    Xtr = train[enriched_feats].fillna(0).values
    ytr = (train["pnl_R"] > 0).astype(int).values

    second = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        eval_metric="logloss", verbosity=0, random_state=0)
    second.fit(Xtr, ytr)

    # Feature importance
    imp = pd.DataFrame({"feature": enriched_feats,
                         "gain": second.feature_importances_}).sort_values("gain", ascending=False)
    print(f"\n  Top 8 by importance:\n{imp.head(8).to_string(index=False)}")
    in_p = second.predict_proba(Xtr)[:, 1]
    print(f"  in-sample corr(p, pnl_R) = {pearsonr(in_p, train['pnl_R'].values)[0]:+.4f}")

    # ----- Apply to Oracle holdout
    print("\n--- Oracle XAU holdout ---")
    oracle = pd.read_csv(os.path.join(ROOT, "data/v72l_trades_holdout.csv"),
                          parse_dates=["time"])
    oracle = attach_pconf_and_features(oracle, mdls, thrs=None, V72L_FEATS=V72L_FEATS)
    # Map rule names not seen in training to -1 (unknown)
    oracle["rule_id"] = oracle["rule"].map(rule_map).fillna(-1).astype(int)
    Xte = oracle[enriched_feats].fillna(0).values
    p2 = second.predict_proba(Xte)[:, 1]
    evaluate_gate("oracle_holdout", oracle, p2)

    # ----- Apply to Midas holdout (post-2024-12-12 portion)
    print("\n--- Midas XAU holdout ---")
    midas_hold = midas_full[midas_full["time"] >= HOLDOUT_START].copy()
    midas_hold["rule_id"] = midas_hold["rule"].map(rule_map).fillna(-1).astype(int)
    Xte_m = midas_hold[enriched_feats].fillna(0).values
    p2m = second.predict_proba(Xte_m)[:, 1]
    evaluate_gate("midas_holdout", midas_hold, p2m)


if __name__ == "__main__":
    main()
