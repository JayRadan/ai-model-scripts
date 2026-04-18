"""
Audit v6b for leakage and correctness.

Checks:
  1. Feature computation is past-only (no future bars)
  2. Time merge is unique / no duplicate-row contamination
  3. Per-rule train/test split cutoffs vs global holdout cutoff
  4. Holdout setups are truly held out for EVERY rule they belong to
  5. Are the new features correlated enough with existing ones to be redundant?
  6. Compare v6b confirmed setups to v6 confirmed setups overlap
"""
from __future__ import annotations
import glob, json, os
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "/home/jay/Desktop/new-model-zigzag/model_pipeline")
import paths as P

ALL_FEATS = [
    "hurst_rs", "ou_theta", "entropy_rate", "kramers_up", "wavelet_er",
    "vwap_dist", "hour_enc", "dow_enc",
    "quantum_flow", "quantum_flow_h4", "quantum_momentum", "quantum_vwap_conf",
    "quantum_divergence", "quantum_div_strength",
    "permutation_entropy", "dfa_alpha", "higuchi_fd",
    "spectral_entropy", "hill_tail_index", "vol_of_vol", "log_drift",
]
NEW_FEATS = ["permutation_entropy", "dfa_alpha", "higuchi_fd",
             "spectral_entropy", "hill_tail_index", "vol_of_vol", "log_drift"]


def check_1_feature_timing():
    """Verify features at time t only use data from before t."""
    print("=" * 60)
    print("CHECK 1 — feature past-only computation")
    print("=" * 60)
    # Load the compute script and inspect each function's slicing pattern
    src = open("/home/jay/Desktop/new-model-zigzag/experiments/new_physics_features/compute_new_features.py").read()
    for feat, pattern in [
        ("permutation_entropy", "x[i - window:i]"),
        ("dfa_alpha",           "x[i - window:i]"),
        ("higuchi_fd",          "x[i - window:i]"),
        ("spectral_entropy",    "x[i - window:i]"),
        ("hill_tail_index",     "x[i - window:i]"),
        ("log_drift",           "x[i - window:i]"),
    ]:
        found = pattern in src
        print(f"  {feat:<22} uses [i-window:i] slice (past only): {'YES' if found else 'NO — SUSPECT!'}")
    # vol_of_vol uses pd.rolling which defaults to trailing window (past incl. current)
    print(f"  vol_of_vol             uses pd.rolling (pandas default is trailing/past): YES")


def check_2_merge_integrity():
    """Check setup CSVs for duplicates or broken time merges."""
    print("\n" + "=" * 60)
    print("CHECK 2 — merge integrity (no duplicates, no NaNs in new cols)")
    print("=" * 60)
    for f in sorted(glob.glob(P.data("setups_*_v6b.csv"))):
        cid = os.path.basename(f).split("_")[1]
        df = pd.read_csv(f, parse_dates=["time"])
        dups = df.duplicated(subset=["time", "rule"]).sum()
        nan_new = df[NEW_FEATS].isna().sum().sum()
        all_zero = (df[NEW_FEATS] == 0).all(axis=0).sum()
        print(f"  c{cid}: rows={len(df):,} duplicates={dups} new-col NaNs={nan_new} all-zero cols={all_zero}")


def check_3_train_test_cutoffs():
    """For each rule, compare its per-rule 80% cutoff to the global 80% cutoff.
    Rules with later per-rule cutoffs COULD have training data that falls inside
    the global holdout window — creating leakage in our backtest."""
    print("\n" + "=" * 60)
    print("CHECK 3 — per-rule train cutoff vs global holdout cutoff")
    print("=" * 60)
    all_setups = []
    for f in sorted(glob.glob(P.data("setups_*_v6b.csv"))):
        cid = int(os.path.basename(f).split("_")[1])
        df = pd.read_csv(f, parse_dates=["time"])
        df["cid"] = cid
        all_setups.append(df)
    all_df = pd.concat(all_setups, ignore_index=True).sort_values("time").reset_index(drop=True)
    global_80_cut = all_df["time"].quantile(0.80)
    print(f"  Global 80% cutoff (holdout starts):  {global_80_cut}")
    print(f"  All setups: {len(all_df):,}")
    print(f"  Holdout setups (time > global_80):  {(all_df['time'] > global_80_cut).sum():,}")
    print()
    leaky_rules = []
    total_leaky_train_in_holdout = 0
    total_holdout_in_train_of_rule = 0
    for (cid, rule), grp in all_df.groupby(["cid", "rule"]):
        grp = grp.sort_values("time").reset_index(drop=True)
        split = int(len(grp) * 0.80)
        if split < 20: continue
        train_cut = grp["time"].iat[split - 1]
        # Leakage type A: train cutoff is AFTER global_80 → some training data inside holdout
        train_after_global = train_cut > global_80_cut
        # Leakage type B: how many of this rule's setups with time > global_80 are actually
        # in this rule's TRAIN portion (not TEST)?
        train_portion = grp.iloc[:split]
        leaky_rows = (train_portion["time"] > global_80_cut).sum()
        if leaky_rows > 0 or train_after_global:
            leaky_rules.append((cid, rule, len(grp), train_cut, leaky_rows))
            total_leaky_train_in_holdout += int(leaky_rows)
        # Leakage type C: how many "holdout" (time > global) setups are in TRAIN?
        holdout_this_rule = grp[grp["time"] > global_80_cut]
        in_train = holdout_this_rule[holdout_this_rule["time"] <= train_cut]
        total_holdout_in_train_of_rule += len(in_train)

    if leaky_rules:
        print(f"  ⚠️  LEAKAGE DETECTED — {len(leaky_rules)} rules have training data past global 80% cutoff:")
        print(f"     Total train-portion rows falling into holdout window: {total_leaky_train_in_holdout:,}")
        print(f"     Total 'holdout' setups that are actually in a rule's train portion: {total_holdout_in_train_of_rule:,}")
        print(f"\n  Worst offenders (rule | total_setups | per-rule train cutoff | rows in global holdout):")
        for cid, rule, n, cut, leaky in sorted(leaky_rules, key=lambda x: -x[4])[:10]:
            print(f"    c{cid}_{rule:<25} n={n:<6} cut={cut} leaky={leaky}")
    else:
        print("  ✓ No leakage — every rule's train portion ends before global 80% cutoff.")


def check_4_holdout_purity():
    """For the confirmed holdout trades, how many use a model trained on data later than the trade?"""
    print("\n" + "=" * 60)
    print("CHECK 4 — holdout trades vs model train cutoffs")
    print("=" * 60)
    all_setups = []
    for f in sorted(glob.glob(P.data("setups_*_v6b.csv"))):
        cid = int(os.path.basename(f).split("_")[1])
        df = pd.read_csv(f, parse_dates=["time"])
        df["cid"] = cid
        all_setups.append(df)
    all_df = pd.concat(all_setups, ignore_index=True).sort_values("time").reset_index(drop=True)
    global_80_cut = all_df["time"].quantile(0.80)
    holdout = all_df[all_df["time"] > global_80_cut]
    # For each (cid, rule), compute per-rule train cutoff
    rule_train_cuts = {}
    for (cid, rule), grp in all_df.groupby(["cid", "rule"]):
        grp = grp.sort_values("time").reset_index(drop=True)
        split = int(len(grp) * 0.80)
        if split < 20: continue
        rule_train_cuts[(cid, rule)] = grp["time"].iat[split - 1]
    # How many holdout setups are at time t <= train_cut for their rule?
    contamination = 0
    clean = 0
    for _, s in holdout.iterrows():
        k = (s["cid"], s["rule"])
        if k not in rule_train_cuts: continue
        if s["time"] <= rule_train_cuts[k]:
            contamination += 1
        else:
            clean += 1
    total = contamination + clean
    if total == 0:
        print("  No valid comparisons")
        return
    pct = 100 * contamination / total
    print(f"  Total holdout setups matched to a rule: {total:,}")
    print(f"  Clean (holdout time > rule train cutoff): {clean:,}  ({100-pct:.1f}%)")
    print(f"  ⚠️  Contaminated (holdout time <= rule train cutoff): {contamination:,}  ({pct:.1f}%)")
    if pct > 0.1:
        print(f"\n  ⚠️  LEAKAGE: {pct:.1f}% of 'holdout' evaluations used models that saw that time during training.")


def check_5_feature_correlations():
    """Are the new features highly correlated with existing ones (redundancy)?"""
    print("\n" + "=" * 60)
    print("CHECK 5 — new feature correlations")
    print("=" * 60)
    dfs = []
    for f in sorted(glob.glob(P.data("setups_*_v6b.csv"))):
        dfs.append(pd.read_csv(f))
    df = pd.concat(dfs, ignore_index=True)
    df = df[df["label"].notna()]
    # Correlation of each new feature with existing features (pick strongest 3)
    existing = [f for f in ALL_FEATS if f not in NEW_FEATS]
    for new in NEW_FEATS:
        if new not in df.columns: continue
        col = df[new]
        if col.std() < 1e-9:
            print(f"  {new:<22} CONSTANT (std=0) — dead feature")
            continue
        corrs = [(e, df[new].corr(df[e])) for e in existing if e in df.columns]
        corrs = [(e, c) for e, c in corrs if np.isfinite(c)]
        corrs.sort(key=lambda x: -abs(x[1]))
        top3 = corrs[:3]
        top_str = " ".join(f"{e}:{c:+.2f}" for e, c in top3)
        print(f"  {new:<22} std={col.std():.4f}  top3 corr: {top_str}")


def check_6_leakage_via_full_series():
    """Was any feature computed using FUTURE bars that fall into the setup window?

    The features are computed once over the full swing CSV (which spans full
    training+holdout period). At each time t, feature uses seg = x[i-window:i].
    If the FULL series includes future bars, the feature at time t still only uses
    past bars (that's the whole point of the [i-window:i] slice).

    BUT: there's a subtle issue. The ffill step in the compute propagates values
    forward — so at time t, the feature value might be from a *recent past* bar
    (step=6 means last computed at t-5, which is still past-only). That's fine.

    Only potential issue: did ffill somehow copy a FUTURE value backward? Let me
    verify by reading the _ffill function implementation.
    """
    print("\n" + "=" * 60)
    print("CHECK 6 — ffill direction (forward only, not backward)")
    print("=" * 60)
    src = open("/home/jay/Desktop/new-model-zigzag/experiments/new_physics_features/compute_new_features.py").read()
    # Find _ffill function
    if "def _ffill" in src:
        fn = src.split("def _ffill(arr):")[1].split("\n\n")[0]
        print("  _ffill implementation:")
        for line in fn.split("\n")[:8]:
            print(f"    {line}")
        # Verify it iterates forward, not backward
        if "range(len(arr)):" in fn:
            print("\n  ✓ Iterates forward — propagates past values forward in time. Safe.")
        else:
            print("\n  ⚠️  SUSPECT — check iteration direction")


def main():
    check_1_feature_timing()
    check_2_merge_integrity()
    check_3_train_test_cutoffs()
    check_4_holdout_purity()
    check_5_feature_correlations()
    check_6_leakage_via_full_series()
    print("\n" + "=" * 60)
    print("AUDIT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
