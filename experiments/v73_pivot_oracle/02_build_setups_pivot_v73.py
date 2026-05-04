"""
v7.3 Pivot Oracle — build per-cluster setups using only RP_* turning-point rules.

Inputs:
  data/swing_v5_xauusd.csv                                    (canonical bars + f01..f20)
  experiments/v73_pivot_oracle/data/features_v73.csv          (from 00)
  experiments/v73_pivot_oracle/data/cluster_per_bar_v73.csv   (from 01)

Outputs (one CSV per cluster, schema = Oracle's setups_*_v72l.csv exactly):
  experiments/v73_pivot_oracle/data/setups_0_v73p.csv
  experiments/v73_pivot_oracle/data/setups_1_v73p.csv
  ...
  experiments/v73_pivot_oracle/data/setups_4_v73p.csv

Each row = one fire of one of the 5 RP_* rules at a bar that belonged to
cluster C{cid}. Schema matches Oracle so 03_validate_v73.py can ingest verbatim.

Detectors are causal (use only [..i] info on bar i). Direction encodes the
expected pivot direction (+1 long / -1 short).

The `label` column is computed via the SAME ML-decided exit framework Oracle
uses (forward sim with hard SL=4*ATR, max_hold=60). For the setup-builder we
use a simple conservative label: 1 if max-favorable-excursion in next 60 bars
beats max-adverse-excursion before SL hits, else 0. This is just a placeholder
target — the real entry classifier in 03 retrains label from forward sim.
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/jay/Desktop/new-model-zigzag/model_pipeline")
import paths as P

DATA_DIR = "/home/jay/Desktop/new-model-zigzag/experiments/v73_pivot_oracle/data"
FEATURES_PATH = os.path.join(DATA_DIR, "features_v73.csv")
CLUSTERS_PATH = os.path.join(DATA_DIR, "cluster_per_bar_v73.csv")

ZZ_THR_ATR = 0.5
SL_HARD = 4.0       # match Oracle
MAX_HOLD = 60       # match Oracle


# ---------- helpers ----------

def compute_atr(H, L, C, n=14):
    tr = np.concatenate([
        [H[0] - L[0]],
        np.maximum.reduce([H[1:] - L[1:], np.abs(H[1:] - C[:-1]), np.abs(L[1:] - C[:-1])]),
    ])
    return pd.Series(tr).rolling(n, min_periods=n).mean().values


def compute_rsi(C, n=14):
    diff = np.diff(C, prepend=C[0])
    up = np.where(diff > 0, diff, 0.0); dn = np.where(diff < 0, -diff, 0.0)
    roll_up = pd.Series(up).rolling(n, min_periods=n).mean().values
    roll_dn = pd.Series(dn).rolling(n, min_periods=n).mean().values
    rs = np.divide(roll_up, roll_dn, out=np.zeros_like(roll_up), where=roll_dn > 0)
    return 100.0 - (100.0 / (1.0 + rs))


def label_max_excursion(direction: int, entry_idx: int, atr_e: float,
                         H: np.ndarray, L: np.ndarray, C: np.ndarray) -> int:
    """1 if MFE > 1*ATR before SL=4*ATR hits within MAX_HOLD bars."""
    if not np.isfinite(atr_e) or atr_e <= 0: return 0
    end = min(entry_idx + 1 + MAX_HOLD, len(H))
    sl_dist = SL_HARD * atr_e
    target = 1.0 * atr_e
    entry_px = C[entry_idx]
    for j in range(entry_idx + 1, end):
        if direction == +1:
            if (entry_px - L[j]) >= sl_dist: return 0
            if (H[j] - entry_px) >= target: return 1
        else:
            if (H[j] - entry_px) >= sl_dist: return 0
            if (entry_px - L[j]) >= target: return 1
    return 0


# ---------- detectors ----------

def detect_RP_a_fractal(H, L):
    fires = []; n = len(H)
    for i in range(4, n):
        if L[i-2] < L[i-3] and L[i-2] < L[i-4] and L[i-1] > L[i-2] and L[i] > L[i-2]:
            fires.append((i, +1, "RP_a_fractal"))
        if H[i-2] > H[i-3] and H[i-2] > H[i-4] and H[i-1] < H[i-2] and H[i] < H[i-2]:
            fires.append((i, -1, "RP_a_fractal"))
    return fires


def detect_RP_b_zigzag(H, L, C, atr):
    fires = []; n = len(H)
    leg_dir = 0; last_pivot_price = C[0]
    for i in range(1, n):
        a = atr[i] if np.isfinite(atr[i]) and atr[i] > 0 else None
        if a is None: continue
        thr = ZZ_THR_ATR * a
        if leg_dir >= 0:
            if H[i] > last_pivot_price: last_pivot_price = H[i]; leg_dir = +1
            elif (last_pivot_price - C[i]) >= thr:
                fires.append((i, -1, "RP_b_zigzag")); last_pivot_price = L[i]; leg_dir = -1
        if leg_dir <= 0:
            if L[i] < last_pivot_price: last_pivot_price = L[i]; leg_dir = -1
            elif (C[i] - last_pivot_price) >= thr:
                fires.append((i, +1, "RP_b_zigzag")); last_pivot_price = H[i]; leg_dir = +1
    return fires


def detect_RP_c_momdiv(H, L, C, atr, rsi):
    fires = []; n = len(H)
    leg_dir = 0; last_pivot_price = C[0]; last_pivot_idx = 0
    last_low_price = None; last_low_rsi = None
    last_high_price = None; last_high_rsi = None
    for i in range(1, n):
        a = atr[i] if np.isfinite(atr[i]) and atr[i] > 0 else None
        if a is None: continue
        thr = ZZ_THR_ATR * a
        if leg_dir >= 0:
            if H[i] > last_pivot_price: last_pivot_price = H[i]; last_pivot_idx = i; leg_dir = +1
            elif (last_pivot_price - C[i]) >= thr:
                pidx = last_pivot_idx; pprice = last_pivot_price; prsi = rsi[pidx]
                if last_high_price is not None and pprice > last_high_price and prsi < last_high_rsi:
                    fires.append((i, -1, "RP_c_momdiv"))
                last_high_price = pprice; last_high_rsi = prsi
                last_pivot_price = L[i]; last_pivot_idx = i; leg_dir = -1
        if leg_dir <= 0:
            if L[i] < last_pivot_price: last_pivot_price = L[i]; last_pivot_idx = i; leg_dir = -1
            elif (C[i] - last_pivot_price) >= thr:
                pidx = last_pivot_idx; pprice = last_pivot_price; prsi = rsi[pidx]
                if last_low_price is not None and pprice < last_low_price and prsi > last_low_rsi:
                    fires.append((i, +1, "RP_c_momdiv"))
                last_low_price = pprice; last_low_rsi = prsi
                last_pivot_price = H[i]; last_pivot_idx = i; leg_dir = +1
    return fires


def detect_RP_d_wickrej(O, H, L, C):
    fires = []; n = len(H)
    for i in range(1, n):
        body = abs(C[i] - O[i]); rng = H[i] - L[i]
        if rng <= 0: continue
        upper = H[i] - max(O[i], C[i]); lower = min(O[i], C[i]) - L[i]
        if lower >= 2 * max(body, 1e-9) and C[i] >= O[i] and C[i] >= L[i] + rng / 2:
            fires.append((i, +1, "RP_d_wickrej"))
        if upper >= 2 * max(body, 1e-9) and C[i] <= O[i] and C[i] <= L[i] + rng / 2:
            fires.append((i, -1, "RP_d_wickrej"))
    return fires


def detect_RP_e_volclimax(O, H, L, C, atr):
    fires = []; n = len(H)
    for i in range(4, n):
        a = atr[i] if np.isfinite(atr[i]) and atr[i] > 0 else None
        if a is None: continue
        rng = H[i] - L[i]
        if rng < 2.0 * a: continue
        in_upper = C[i] >= L[i] + rng / 2; in_lower = C[i] <= L[i] + rng / 2
        prior_down = C[i-1] < C[i-4]; prior_up = C[i-1] > C[i-4]
        if prior_down and in_upper: fires.append((i, +1, "RP_e_volclimax"))
        if prior_up and in_lower: fires.append((i, -1, "RP_e_volclimax"))
    return fires


def main():
    print("Loading swing CSV...", flush=True)
    swing = pd.read_csv(P.data("swing_v5_xauusd.csv"), parse_dates=["time"])
    swing = swing.sort_values("time").reset_index(drop=True)
    O = swing["open"].values.astype(np.float64); H = swing["high"].values.astype(np.float64)
    L = swing["low"].values.astype(np.float64);  C = swing["close"].values.astype(np.float64)
    atr = compute_atr(H, L, C, 14)
    rsi = compute_rsi(C, 14)

    print(f"Loading features {FEATURES_PATH}...", flush=True)
    feats = pd.read_csv(FEATURES_PATH, parse_dates=["time"])

    print(f"Loading clusters {CLUSTERS_PATH}...", flush=True)
    clusters = pd.read_csv(CLUSTERS_PATH, parse_dates=["time"])

    print("Running 5 RP_* detectors...", flush=True)
    all_fires = []
    for fn_name, fires in [
        ("RP_a", detect_RP_a_fractal(H, L)),
        ("RP_b", detect_RP_b_zigzag(H, L, C, atr)),
        ("RP_c", detect_RP_c_momdiv(H, L, C, atr, rsi)),
        ("RP_d", detect_RP_d_wickrej(O, H, L, C)),
        ("RP_e", detect_RP_e_volclimax(O, H, L, C, atr)),
    ]:
        print(f"  {fn_name}: {len(fires):,} fires", flush=True)
        all_fires.extend(fires)

    print(f"\nTotal: {len(all_fires):,} fires across 5 rules", flush=True)

    fires_df = pd.DataFrame(all_fires, columns=["bar_idx", "direction", "rule"])
    print("Computing per-fire labels (MFE-vs-SL)...", flush=True)
    labels = []
    for r in fires_df.itertuples(index=False):
        labels.append(label_max_excursion(int(r.direction), int(r.bar_idx), atr[r.bar_idx], H, L, C))
    fires_df["label"] = labels
    fires_df["time"] = swing["time"].iloc[fires_df["bar_idx"].values].values
    fires_df["idx"] = fires_df["bar_idx"]
    fires_df["atr"] = atr[fires_df["bar_idx"].values]
    fires_df["entry_price"] = C[fires_df["bar_idx"].values]

    # Merge features (keeps the 21 base + 18 v72L + 4 pivot context)
    fires_df = fires_df.merge(feats.drop(columns=["bar_idx"]), on="time", how="left")
    # Merge cluster id
    fires_df = fires_df.merge(clusters.drop(columns=["bar_idx"]), on="time", how="left")

    # Match Oracle's setups_*_v72l.csv schema EXACTLY (so 03_validate can ingest verbatim).
    # Oracle expects: f01..f20, rsi14,rsi6,stoch_k,stoch_d,bb_pct,mom5,mom10,mom20,
    # ll_dist10,hh_dist10,vol_accel,atr_ratio,spread_norm,hour_enc,dow_enc,time,idx,
    # direction,rule,atr,entry_price,label, then v72L feats.
    # Some of those columns aren't in our features file — fill with 0.
    EXPECTED_AUX = ["rsi14","rsi6","stoch_k","stoch_d","bb_pct","mom5","mom10","mom20",
                     "ll_dist10","hh_dist10","vol_accel","atr_ratio","spread_norm"]
    for c in EXPECTED_AUX:
        if c not in fires_df.columns: fires_df[c] = 0.0

    print(f"\nWriting per-cluster setups...", flush=True)
    K = clusters["cid"].nunique()
    for cid in range(K):
        sub = fires_df[fires_df["cid"] == cid].copy()
        sub = sub.drop(columns=["cid"])  # cid added back inside Oracle's loader
        out_path = os.path.join(DATA_DIR, f"setups_{cid}_v73p.csv")
        sub.to_csv(out_path, index=False)
        n_long = (sub["direction"] == 1).sum(); n_short = (sub["direction"] == -1).sum()
        wr = sub["label"].mean() if len(sub) else 0
        print(f"  C{cid}: {len(sub):>7,} fires  ({n_long:,}L/{n_short:,}S)  pos-rate {wr:.1%}  -> {out_path}")

    print("\nPer-rule fires across all clusters:")
    print(fires_df.groupby("rule").size().to_string())


if __name__ == "__main__":
    main()
