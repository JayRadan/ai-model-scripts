"""Test each deployed pkl directly against current holdout setups.

Loads commercial/server/decision_engine/models/{product}_validated.pkl
and runs Stage-1 confirm + Stage-2 meta + exit simulation against the
holdout setups. Prints PF/WR/n/DD so we can see exactly what the live
deployed model produces.

Note: cluster IDs in deployed pkl mdls dict may NOT match current
setups' cluster IDs (since selectors differ). We try anyway — if a
cluster has no setups under the deployed cid scheme, that's
information too.
"""
import os, sys, pickle, glob
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/jay/Desktop/new-model-zigzag/experiments/v72_lite_deploy")
import importlib
val = importlib.import_module("01_validate_v72_lite")

GLOBAL_CUTOFF = pd.Timestamp("2024-12-12 00:00:00")
COMM = "/home/jay/Desktop/my-agents-and-website/commercial/server/decision_engine/models"
ZIGZAG = "/home/jay/Desktop/new-model-zigzag"


def test_xau_v72l(product: str):
    pkl_path = f"{COMM}/{product}_validated.pkl"
    print(f"\n=== {product} ===")
    p = pickle.load(open(pkl_path, "rb"))
    print(f"  pkl: meta_thr={p.get('meta_threshold')}  heads={len(p['mdls'])}")

    feats = list(p["v72l_feats"])
    meta_feats = list(p.get("meta_feats") or [])

    # Load XAU setups (current state, v9.3 cluster IDs)
    setups = []
    for f in sorted(glob.glob(f"{ZIGZAG}/data/setups_*_v72l.csv")):
        cid = int(os.path.basename(f).split("_")[1])
        df = pd.read_csv(f, parse_dates=["time"])
        df["cid"] = cid
        setups.append(df)
    setups = pd.concat(setups, ignore_index=True).sort_values("time").reset_index(drop=True)
    setups = setups.loc[:, ~setups.columns.duplicated()]
    test = setups[setups["time"] >= GLOBAL_CUTOFF].reset_index(drop=True)
    print(f"  holdout setups: {len(test):,}")

    # Stage-1 using deployed pkl heads. If (cid,rule) not in pkl, skip.
    rows = []
    matched_pairs = 0; unmatched_pairs = 0
    for (cid, rule), m in p["mdls"].items():
        sub = test[(test["cid"] == cid) & (test["rule"] == rule)]
        if not len(sub):
            unmatched_pairs += 1
            continue
        matched_pairs += 1
        Xs = sub[feats].fillna(0).values
        prob = m.predict_proba(Xs)[:, 1]
        thr = p["thrs"][(cid, rule)]
        keep = sub[prob >= thr].copy()
        rows.append(keep)
    print(f"  pkl→holdout pair match: {matched_pairs}/{len(p['mdls'])} (unmatched: {unmatched_pairs})")
    confirmed = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    print(f"  Stage-1 confirmed: {len(confirmed):,}")

    # Stage-2 meta if present
    if p.get("meta_mdl") is not None and len(confirmed) > 0:
        confirmed["direction"] = confirmed["direction"].astype(int)
        confirmed["cid"] = confirmed["cid"].astype(int)
        Xm = confirmed[meta_feats].fillna(0).values
        pm = p["meta_mdl"].predict_proba(Xm)[:, 1]
        confirmed = confirmed[pm >= p["meta_threshold"]].reset_index(drop=True)
        print(f"  Meta passed: {len(confirmed):,}")

    if len(confirmed) == 0:
        print("  → NO TRADES (cluster ID mismatch?)")
        return None

    # Simulate
    swing, atr = val.load_swing_with_physics()
    trades = val.simulate(confirmed, swing, atr, p["exit_mdl"])
    if len(trades) == 0:
        print("  → simulate produced no trades")
        return None
    w = trades[trades["pnl_R"] > 0]; l = trades[trades["pnl_R"] <= 0]
    pf = w["pnl_R"].sum() / max(-l["pnl_R"].sum(), 1e-9)
    wr = len(w) / len(trades)
    eq = trades["pnl_R"].cumsum().values
    dd = (np.maximum.accumulate(eq) - eq).max() if len(eq) > 0 else 0
    print(f"  RESULT: n={len(trades)} WR={wr:.1%} PF={pf:.2f} DD=-{dd:.1f}R Total={trades['pnl_R'].sum():+.0f}R")
    return {"n": len(trades), "wr": wr, "pf": pf, "dd": dd, "total": trades["pnl_R"].sum()}


if __name__ == "__main__":
    products = sys.argv[1:] if len(sys.argv) > 1 else ["oracle_xau"]
    for prod in products:
        test_xau_v72l(prod)
