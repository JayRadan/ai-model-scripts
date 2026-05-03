"""
v8.7 — Hard-SL Trade Forensics

Goal: take all trades that hit -4R hard SL (~20-30% of all trades on each
product), study their pre-entry feature signatures with multivariate
methods, and determine if there is exploitable structure that distinguishes
them from winners — pre-entry, before any post-entry path information.

Different from v8.5 (kill switch): v8.5 used POST-entry per-bar features
and got AUC 0.83 but lost money because winners-in-drawdown looked like
hard_sls. This experiment uses ONLY the features available at fire time
(the 18 v72L physics features + 14 H1/H4 + 21 candle micro features +
direction + cid). If pre-entry features can distinguish, we have a real
filter that doesn't suffer the post-entry confusion.

Pipeline per product:
  1. Load holdout trades + merge feature rows (from setups files at fire time)
  2. Three classes: winners (pnl_R > 0), losers (-4 < pnl_R <= 0), hard_sl (= -4R)
  3. Univariate: for each feature, Mann-Whitney U test winners vs hard_sl,
     Cohen's d effect size. Identify features with |d| > 0.20 AND p < 0.001.
  4. Multivariate: PCA of all features. Plot 1st-2nd PC of hard_sl vs others.
     Mahalanobis distance from winner centroid — are hard_sls outliers?
  5. Sub-type discovery: K-means on hard_sl-only in PCA space (k=3..5).
     For each sub-cluster, identify top-3 distinguishing features vs
     winner centroid.
  6. Predictive test: train XGB binary (hard_sl vs winner) on H1, test on
     H2. Walk-forward AUC. If > 0.55 it's predictive; if > 0.65 it's
     ship-relevant.
  7. Filter simulation: at the AUC>0.65 threshold, what fraction of
     hard_sls would we catch? What collateral damage on winners?
     Compute ΔR if we skip predicted-hardsl entries.

Output: per-product reports + a top-line summary.
"""
from __future__ import annotations
import os, glob, sys
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, chi2
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import json

ROOT = "/home/jay/Desktop/new-model-zigzag"
OUT  = os.path.dirname(__file__)

# 18 v72L physics + direction + cid (matches META_FEATS shape)
V72L_FEATS = [
    "hurst_rs", "ou_theta", "entropy_rate", "kramers_up", "wavelet_er",
    "vwap_dist", "hour_enc", "dow_enc",
    "quantum_flow", "quantum_flow_h4", "quantum_momentum", "quantum_vwap_conf",
    "quantum_divergence", "quantum_div_strength",
    "vpin", "sig_quad_var", "har_rv_ratio", "hawkes_eta",
]
V6_FEATS = V72L_FEATS[:14]   # Midas v6 has no v72L extras


def cohens_d(a, b):
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) < 5 or len(b) < 5: return 0.0
    pooled_std = np.sqrt(((len(a)-1)*a.std()**2 + (len(b)-1)*b.std()**2) / (len(a)+len(b)-2))
    if pooled_std == 0: return 0.0
    return (np.mean(b) - np.mean(a)) / pooled_std


def load_setups_with_feats(feat_glob):
    setups = []
    for f in sorted(glob.glob(os.path.join(ROOT, "data", feat_glob))):
        cid = int(os.path.basename(f).split("_")[1])
        df = pd.read_csv(f, parse_dates=["time"])
        df["cid"] = cid
        setups.append(df)
    return pd.concat(setups, ignore_index=True) if setups else pd.DataFrame()


def attach(trades, setups, feat_cols):
    out = trades.copy()
    out["cid"] = out["cid"].astype(int)
    out["direction"] = out["direction"].astype(int)
    return out.merge(setups[["time", "cid", "rule", "direction"] + feat_cols],
                       on=["time","cid","rule","direction"],
                       how="left").dropna(subset=feat_cols).reset_index(drop=True)


def univariate_report(df, feat_cols, label):
    win = df[df.pnl_R > 0]
    hsl = df[df.exit == "hard_sl"]
    rows = []
    for f in feat_cols:
        a = win[f].dropna().values
        b = hsl[f].dropna().values
        if len(a) < 10 or len(b) < 10: continue
        try:
            _, p = mannwhitneyu(a, b, alternative="two-sided")
        except Exception:
            p = float("nan")
        d = cohens_d(a, b)
        rows.append({"feature": f, "win_med": np.median(a),
                      "hsl_med": np.median(b), "diff": np.median(b) - np.median(a),
                      "cohen_d": round(d, 3), "p": round(p, 5),
                      "n_win": len(a), "n_hsl": len(b)})
    out = pd.DataFrame(rows).sort_values("cohen_d", key=abs, ascending=False)
    return out


def multivariate_analysis(df, feat_cols):
    win = df[df.pnl_R > 0][feat_cols].fillna(0).values
    hsl = df[df.exit == "hard_sl"][feat_cols].fillna(0).values
    others = df[(df.pnl_R <= 0) & (df.exit != "hard_sl")][feat_cols].fillna(0).values
    if len(win) < 30 or len(hsl) < 30:
        return None

    sc = StandardScaler().fit(win)
    Xw = sc.transform(win); Xh = sc.transform(hsl); Xo = sc.transform(others) if len(others) else None
    pca = PCA(n_components=min(8, len(feat_cols))).fit(Xw)
    Pw = pca.transform(Xw); Ph = pca.transform(Xh)

    # Mahalanobis distance from winner centroid (using winner cov)
    cov = np.cov(Xw.T) + 1e-6 * np.eye(Xw.shape[1])
    inv_cov = np.linalg.inv(cov)
    mu = Xw.mean(axis=0)
    def mahal(X):
        d = X - mu
        return np.sqrt(np.einsum("ij,jk,ik->i", d, inv_cov, d))
    md_w = mahal(Xw); md_h = mahal(Xh)

    # 95th percentile threshold from winners — what fraction of hard_sl exceed it?
    thr95 = np.percentile(md_w, 95)
    frac_hsl_outside = (md_h > thr95).mean()
    expected = 0.05  # by definition, 5% of winners exceed thr95
    chi2_stat = ((frac_hsl_outside - expected)**2 / expected) * len(md_h)

    # K-means on hard_sl in PCA space
    sub_clusters = {}
    for k in [3, 4, 5]:
        if len(Ph) >= 3 * k:
            km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(Ph[:, :3])
            labels = km.labels_
            sub_clusters[k] = {"sizes": np.bincount(labels).tolist(),
                                 "centroids": km.cluster_centers_.tolist()}

    return {
        "pca_explained_var": pca.explained_variance_ratio_.tolist(),
        "winner_md_p50":  float(np.median(md_w)),
        "winner_md_p95":  float(thr95),
        "hardsl_md_p50":  float(np.median(md_h)),
        "hardsl_md_p95":  float(np.percentile(md_h, 95)),
        "frac_hsl_outside_winner_p95": float(frac_hsl_outside),
        "expected_under_null":          expected,
        "chi2_excess":                  float(chi2_stat),
        "sub_clusters": sub_clusters,
    }


def predictive_test(df, feat_cols):
    """Train XGB binary (hard_sl vs winner) on H1, test on H2.
    Returns (test_auc, threshold_at_balanced_acc, kept_pf, lift_summary)."""
    win = df[df.pnl_R > 0].copy(); win["y"] = 0
    hsl = df[df.exit == "hard_sl"].copy(); hsl["y"] = 1
    pool = pd.concat([win, hsl]).sort_values("time").reset_index(drop=True)
    if len(pool) < 100 or pool.y.mean() < 0.05:
        return None
    n_tr = len(pool) // 2
    tr = pool.iloc[:n_tr]; te = pool.iloc[n_tr:]
    Xtr = tr[feat_cols + ["direction", "cid"]].fillna(0).values
    ytr = tr["y"].values
    Xte = te[feat_cols + ["direction", "cid"]].fillna(0).values
    yte = te["y"].values
    if len(set(ytr)) < 2 or len(set(yte)) < 2:
        return None
    mdl = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                         eval_metric="logloss", verbosity=0, random_state=0,
                         tree_method="hist")
    mdl.fit(Xtr, ytr)
    p = mdl.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, p)

    # If we filter at p > thr (skip predicted-hardsl), what's the impact
    # on the FULL holdout (using ALL trades, including the losers between
    # winners and hard_sl which weren't in train)?
    # We score the WHOLE H2 partition of df.
    H2_start = len(df) // 2
    h2 = df.iloc[H2_start:].copy()
    Xh = h2[feat_cols + ["direction", "cid"]].fillna(0).values
    h2["p_hsl"] = mdl.predict_proba(Xh)[:, 1]

    rs0 = h2.pnl_R.values
    base_R = rs0.sum(); base_pf = rs0[rs0>0].sum() / max(-rs0[rs0<=0].sum(), 1e-9)
    base_wr = (rs0 > 0).mean()
    rows = []
    for thr in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]:
        keep = (h2["p_hsl"] < thr).values
        rs = h2.loc[keep, "pnl_R"].values
        if len(rs) < 30: continue
        pf_ = rs[rs>0].sum() / max(-rs[rs<=0].sum(), 1e-9)
        # How many of the SKIPPED trades were actually hard_sl?
        skipped = h2[~keep]
        n_skipped = len(skipped)
        n_hsl_caught = (skipped.exit == "hard_sl").sum()
        n_winners_killed = (skipped.pnl_R > 0).sum()
        rows.append({
            "thr": thr, "kept": int(keep.sum()),
            "skipped": int(n_skipped),
            "hsl_caught": int(n_hsl_caught),
            "winners_killed": int(n_winners_killed),
            "WR": round((rs > 0).mean(), 4),
            "PF": round(pf_, 2),
            "R": round(rs.sum(), 1),
            "ΔR": round(rs.sum() - base_R, 1),
            "ΔWR_pp": round(((rs > 0).mean() - base_wr) * 100, 1),
        })
    return auc, base_R, base_pf, base_wr, pd.DataFrame(rows)


def analyze_one(name, trades_path, feat_cols, feat_glob):
    print("\n" + "="*78)
    print(f"=== {name} ===")
    setups = load_setups_with_feats(feat_glob)
    if setups.empty:
        print("  ⚠ no setups files — skip"); return None
    trades = pd.read_csv(os.path.join(ROOT, trades_path), parse_dates=["time"])
    df = attach(trades, setups, feat_cols).sort_values("time").reset_index(drop=True)
    n_total = len(df); n_hsl = (df.exit == "hard_sl").sum()
    n_win = (df.pnl_R > 0).sum()
    print(f"  trades after merge: {n_total}  winners={n_win} ({n_win/n_total:.1%})  "
          f"hard_sl={n_hsl} ({n_hsl/n_total:.1%})")

    # 1. Univariate
    uv = univariate_report(df, feat_cols, name)
    print("\n  Top 10 features by |Cohen d| (winner vs hard_sl):")
    print(uv.head(10).to_string(index=False))
    uv.to_csv(os.path.join(OUT, f"univariate_{name.lower().replace(' ','_')}.csv"),
                index=False)
    # Hits: |d|>0.20 AND p<0.001
    sig = uv[(uv["cohen_d"].abs() > 0.20) & (uv["p"] < 0.001)]
    print(f"\n  Significant features (|d|>0.20, p<0.001): {len(sig)}")
    if len(sig):
        print(sig[["feature","cohen_d","p","win_med","hsl_med"]].to_string(index=False))

    # 2. Multivariate
    mv = multivariate_analysis(df, feat_cols)
    if mv:
        print(f"\n  Multivariate (PCA + Mahalanobis):")
        print(f"    PC1+PC2+PC3 explain {sum(mv['pca_explained_var'][:3])*100:.1f}% of variance")
        print(f"    winner Mahalanobis dist (p50/p95): {mv['winner_md_p50']:.2f} / {mv['winner_md_p95']:.2f}")
        print(f"    hardsl Mahalanobis dist (p50/p95): {mv['hardsl_md_p50']:.2f} / {mv['hardsl_md_p95']:.2f}")
        print(f"    fraction of hard_sl outside winner 95% ellipsoid: "
              f"{mv['frac_hsl_outside_winner_p95']*100:.1f}% (chance under null: 5%)")
        print(f"    chi^2 excess: {mv['chi2_excess']:.1f}")
        print(f"    K-means hard_sl sub-cluster sizes (k=3,4,5): "
              f"{[mv['sub_clusters'].get(k, {}).get('sizes', []) for k in [3,4,5]]}")
        with open(os.path.join(OUT, f"multivariate_{name.lower().replace(' ','_')}.json"), "w") as f:
            json.dump(mv, f, indent=2)

    # 3. Predictive test
    res = predictive_test(df, feat_cols)
    if res is not None:
        auc, base_R, base_pf, base_wr, sweep = res
        print(f"\n  Predictive test (XGB binary winner-vs-hardsl):")
        print(f"    H2 test AUC = {auc:.3f}")
        print(f"    H2 baseline:  R={base_R:+.0f}  PF={base_pf:.2f}  WR={base_wr:.1%}")
        print(f"\n    Filter sweep (skip when p_hsl >= thr):")
        print(sweep.to_string(index=False))
        sweep.to_csv(os.path.join(OUT, f"filter_sweep_{name.lower().replace(' ','_')}.csv"),
                       index=False)
        verdict = "✅ PROMISING" if auc > 0.65 and (sweep["ΔR"] > 0).any() else \
                  "⚠ marginal" if auc > 0.55 else "❌ no signal"
        print(f"\n  VERDICT: {verdict}  (AUC {auc:.3f}; "
              f"any threshold with ΔR>0? {(sweep['ΔR'] > 0).any()})")
        return {"name": name, "auc": auc, "verdict": verdict, "sweep": sweep}
    return None


def main():
    products = [
        ("Oracle XAU", "data/v72l_trades_holdout.csv",   V72L_FEATS, "setups_*_v72l.csv"),
        ("Midas XAU",  "data/v6_trades_holdout_xau.csv", V6_FEATS,   "setups_*_v6.csv"),
        ("Oracle BTC", "data/v72l_trades_holdout_btc.csv", V72L_FEATS, "setups_*_v72l_btc.csv"),
    ]
    results = []
    for name, path, feats, glob_pat in products:
        r = analyze_one(name, path, feats, glob_pat)
        if r: results.append(r)

    print("\n" + "="*78)
    print("SUMMARY")
    print("="*78)
    for r in results:
        print(f"  {r['name']:12s} AUC={r['auc']:.3f}  {r['verdict']}")


if __name__ == "__main__":
    main()
