"""Audit v7 features: correlations with existing features + label signal."""
import glob, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, "/home/jay/Desktop/new-model-zigzag/model_pipeline")
import paths as P

OLD = ["hurst_rs", "ou_theta", "entropy_rate", "kramers_up", "wavelet_er",
       "vwap_dist", "hour_enc", "dow_enc",
       "quantum_flow", "quantum_flow_h4", "quantum_momentum", "quantum_vwap_conf",
       "quantum_divergence", "quantum_div_strength"]
NEW = ["vpin", "sig_levy_area", "sig_quad_var", "sig_time_weighted_drift", "te_h1_m5"]

dfs = []
for f in sorted(glob.glob(P.data("setups_*_v7.csv"))):
    dfs.append(pd.read_csv(f))
df = pd.concat(dfs, ignore_index=True)
df = df[df["label"].notna()]
print(f"Total labeled setups: {len(df):,}\n")

print("="*80)
print("NEW feature correlations vs EXISTING features (top 3 abs corr)")
print("="*80)
for new in NEW:
    col = df[new]
    if col.std() < 1e-9:
        print(f"  {new:<26} CONSTANT — DEAD FEATURE"); continue
    corrs = [(e, df[new].corr(df[e])) for e in OLD if e in df.columns]
    corrs = [(e, c) for e, c in corrs if np.isfinite(c)]
    corrs.sort(key=lambda x: -abs(x[1]))
    top3 = corrs[:3]
    max_abs = abs(top3[0][1]) if top3 else 0
    top_str = " ".join(f"{e}:{c:+.2f}" for e, c in top3)
    flag = "REDUNDANT" if max_abs > 0.7 else ("suspect" if max_abs > 0.5 else "ok")
    print(f"  {new:<26} std={col.std():.4f}  top3: {top_str}   [{flag}]")

print("\n" + "="*80)
print("NEW feature correlations WITH LABEL (higher abs = more entry signal)")
print("="*80)
labels = df["label"].astype(float)
all_corr = []
for feat in OLD + NEW:
    if feat in df.columns and df[feat].std() > 1e-9:
        c = df[feat].corr(labels)
        all_corr.append((feat, c))
all_corr.sort(key=lambda x: -abs(x[1]))
print(f"  {'feature':<28} {'corr(feat, label)':>18}")
for feat, c in all_corr:
    mark = "* NEW" if feat in NEW else ""
    print(f"  {feat:<28} {c:>+18.4f}   {mark}")

# Pairwise correlations WITHIN the new block (catch internal redundancy)
print("\n" + "="*80)
print("Pairwise correlations WITHIN NEW features")
print("="*80)
sub = df[NEW].corr()
print(sub.round(2).to_string())
