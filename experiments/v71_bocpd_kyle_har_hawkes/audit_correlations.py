"""Audit v7.1 features: correlations with existing features + label signal."""
import glob, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, "/home/jay/Desktop/new-model-zigzag/model_pipeline")
import paths as P

OLD = ["hurst_rs", "ou_theta", "entropy_rate", "kramers_up", "wavelet_er",
       "vwap_dist", "hour_enc", "dow_enc",
       "quantum_flow", "quantum_flow_h4", "quantum_momentum", "quantum_vwap_conf",
       "quantum_divergence", "quantum_div_strength"]
V7_KEPT = ["vpin", "sig_quad_var"]
NEW = ["bocpd_recent_cp", "kyle_lambda", "har_rv_ratio", "hawkes_eta"]

dfs = []
for f in sorted(glob.glob(P.data("setups_*_v71.csv"))):
    dfs.append(pd.read_csv(f))
df = pd.concat(dfs, ignore_index=True)
df = df[df["label"].notna()]
print(f"Total labeled setups: {len(df):,}\n")

print("="*80)
print("NEW feature correlations vs existing+v7-kept features (top 3 abs corr)")
print("="*80)
others = OLD + V7_KEPT
for new in NEW:
    col = df[new]
    if col.std() < 1e-9:
        print(f"  {new:<20} CONSTANT — DEAD"); continue
    corrs = [(e, df[new].corr(df[e])) for e in others if e in df.columns]
    corrs = [(e, c) for e, c in corrs if np.isfinite(c)]
    corrs.sort(key=lambda x: -abs(x[1]))
    top3 = corrs[:3]
    max_abs = abs(top3[0][1]) if top3 else 0
    flag = "REDUNDANT" if max_abs > 0.7 else ("suspect" if max_abs > 0.5 else "ok")
    top_str = " ".join(f"{e}:{c:+.2f}" for e, c in top3)
    print(f"  {new:<20} std={col.std():.4f}  top3: {top_str}   [{flag}]")

print("\n" + "="*80)
print("LABEL correlation ranking (all features, new marked *)")
print("="*80)
labels = df["label"].astype(float)
all_corr = []
for feat in OLD + V7_KEPT + NEW:
    if feat in df.columns and df[feat].std() > 1e-9:
        c = df[feat].corr(labels)
        all_corr.append((feat, c))
all_corr.sort(key=lambda x: -abs(x[1]))
print(f"  {'feature':<22} {'corr(feat, label)':>18}")
for feat, c in all_corr:
    mark = "* NEW" if feat in NEW else ("kept-from-v7" if feat in V7_KEPT else "")
    print(f"  {feat:<22} {c:>+18.4f}   {mark}")

print("\n" + "="*80)
print("Pairwise correlations WITHIN the 4 NEW features")
print("="*80)
print(df[NEW].corr().round(2).to_string())
