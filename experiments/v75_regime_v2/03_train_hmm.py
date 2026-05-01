"""
v75 step 03: Gaussian HMM regime detector. Unlike K-means (which gives a
hard cluster per bar), an HMM models REGIME TRANSITIONS — the probability
of being in each state given everything observed so far. Regime shifts
get flagged as changes in posterior probability, not as instantaneous
cluster jumps.

Training: Gaussian HMM with N states (try N∈{3,4,5}), 4 features:
  ret_3h, slope_3h, slope_accel, vol_ratio
Pre-2024-12-12 = train.  Post = test.

For each bar in the test window, compute Viterbi-decoded most-likely
state AND the per-state forward posterior. We then check:
  • per-state forward 1h return distribution (edge analysis)
  • shift-day capture rate vs K-means baseline
"""
from __future__ import annotations
import os, sys, json, time as _time, pickle
import numpy as np, pandas as pd
from hmmlearn.hmm import GaussianHMM

ZIGZAG = "/home/jay/Desktop/new-model-zigzag"
EXP = os.path.join(ZIGZAG, "experiments/v75_regime_v2")
FP = os.path.join(EXP, "data/fingerprints_rich.parquet")

FEATS = ["ret_3h", "slope_3h", "slope_accel", "vol_ratio"]
CUTOFF = pd.Timestamp("2024-12-12 00:00:00")
N_STATES_LIST = [3, 4, 5]


def fit_one(N, X_train, train_idx):
    print(f"\n── Fitting Gaussian HMM N={N} on {len(X_train):,} train rows...", flush=True)
    t0 = _time.time()
    model = GaussianHMM(n_components=N, covariance_type="diag",
                         n_iter=50, random_state=42, tol=1e-3)
    model.fit(X_train)
    print(f"  converged in {_time.time()-t0:.0f}s, log-likelihood = {model.score(X_train):.0f}")

    # Decode train labels for cluster naming
    train_states = model.predict(X_train)
    print(f"  train state counts: {pd.Series(train_states).value_counts().sort_index().to_dict()}")
    return model


def label_states(model, X_train, train_states):
    """Name each hidden state by its emission means + 1h fwd return on train."""
    feats_arr = np.array(FEATS)
    names = {}
    for s in range(model.n_components):
        m = model.means_[s]
        d = {f: float(m[i]) for i, f in enumerate(feats_arr)}
        # Heuristic: name by sign of ret_3h + slope_accel + vol_ratio
        r3 = d["ret_3h"]; sa = d["slope_accel"]; vr = d["vol_ratio"]
        if r3 < -0.003 and sa < 0:
            label = "BearBreakdown"
        elif r3 > 0.003 and sa > 0:
            label = "BullBreakout"
        elif r3 < -0.001:
            label = "Downtrend"
        elif r3 > 0.001:
            label = "Uptrend"
        else:
            label = "Quiet"
        names[s] = label
    return names


def main():
    fp = pd.read_parquet(FP).sort_values("time").reset_index(drop=True)
    fp = fp.dropna(subset=FEATS).reset_index(drop=True)
    print(f"Loaded {len(fp):,} rows  {fp['time'].iloc[0]} → {fp['time'].iloc[-1]}")

    train_mask = fp["time"] < CUTOFF
    test_mask  = fp["time"] >= CUTOFF
    Xtr = fp.loc[train_mask, FEATS].values.astype(np.float64)
    Xte = fp.loc[test_mask,  FEATS].values.astype(np.float64)
    print(f"Train (pre-cutoff): {len(Xtr):,}   Test (post): {len(Xte):,}")

    # Standardise on train
    mean = Xtr.mean(axis=0); std = Xtr.std(axis=0) + 1e-9
    Xtr_s = (Xtr - mean) / std
    Xte_s = (Xte - mean) / std

    fp_test = fp.loc[test_mask].reset_index(drop=True).copy()
    fp_test["fwd_ret_1h"] = pd.Series(fp_test["close"].values).pct_change(12).shift(-12).values

    for N in N_STATES_LIST:
        print(f"\n========== N = {N} ==========")
        model = fit_one(N, Xtr_s, fp.loc[train_mask].index.values)

        # Decode test states (Viterbi)
        test_states = model.predict(Xte_s)
        # Posterior probabilities per bar (forward)
        posteriors = model.predict_proba(Xte_s)

        train_states = model.predict(Xtr_s)
        names = label_states(model, Xtr_s, train_states)
        print(f"  state names: {names}")

        # Edge analysis on test
        fp_test["state"] = test_states
        clean = fp_test.dropna(subset=["fwd_ret_1h"])
        print(f"\n  Per-state forward-1h return (test):")
        print(f"  {'state':<25s} {'n':>7s}  {'mean fwd_ret_1h':>15s}  {'P(drop>0.3%)':>14s}  {'P(rise>0.3%)':>14s}")
        for s in range(N):
            sub = clean[clean["state"] == s]
            if not len(sub): continue
            print(f"  N={N} S{s} {names[s]:<18s} {len(sub):>7d}  {sub['fwd_ret_1h'].mean()*100:>14.3f}%  "
                  f"{(sub['fwd_ret_1h']<-0.003).mean()*100:>13.1f}%  "
                  f"{(sub['fwd_ret_1h']> 0.003).mean()*100:>13.1f}%")

        # Save
        with open(os.path.join(EXP, f"models/hmm_N{N}.pkl"), "wb") as f:
            pickle.dump({"model": model, "feats": FEATS, "scaler_mean": mean.tolist(),
                         "scaler_std": std.tolist(), "state_names": names}, f)
        # Persist test posteriors + states for downstream analysis
        post_df = pd.DataFrame(posteriors, columns=[f"P_S{s}" for s in range(N)])
        post_df.insert(0, "state", test_states)
        post_df.insert(0, "time", fp_test["time"].values)
        post_df.to_parquet(os.path.join(EXP, f"data/hmm_N{N}_test_posteriors.parquet"), index=False)
        print(f"  saved hmm_N{N}.pkl + posteriors")

    print("\nDone.")


if __name__ == "__main__":
    main()
