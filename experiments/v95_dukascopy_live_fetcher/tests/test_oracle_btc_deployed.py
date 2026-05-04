"""Test deployed Oracle BTC pkl against fresh v9.3 BTC holdout setups."""
import os, sys, pickle, glob
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/jay/Desktop/new-model-zigzag/experiments/v72_lite_btc_deploy")
import importlib
val = importlib.import_module("01_validate_v72_lite_btc")

GLOBAL_CUTOFF = pd.Timestamp("2024-12-12 00:00:00")
COMM = "/home/jay/Desktop/my-agents-and-website/commercial/server/decision_engine/models"
ZIGZAG = "/home/jay/Desktop/new-model-zigzag"

p = pickle.load(open(f"{COMM}/oracle_btc_validated.pkl", "rb"))
print(f"deployed BTC pkl: meta_thr={p['meta_threshold']}  heads={len(p['mdls'])}")

feats = list(p["v72l_feats"]); meta_feats = list(p["meta_feats"])

setups = []
for f in sorted(glob.glob(f"{ZIGZAG}/data/setups_*_v72l_btc.csv")):
    cid = int(os.path.basename(f).split("_")[1])
    df = pd.read_csv(f, parse_dates=["time"])
    df["cid"] = cid
    setups.append(df)
setups = pd.concat(setups, ignore_index=True).sort_values("time").reset_index(drop=True)
setups = setups.loc[:, ~setups.columns.duplicated()]
test = setups[setups["time"] >= GLOBAL_CUTOFF].reset_index(drop=True)
print(f"holdout setups: {len(test):,}")

rows = []; matched = 0; unmatched = 0
for (cid, rule), m in p["mdls"].items():
    sub = test[(test["cid"] == cid) & (test["rule"] == rule)]
    if not len(sub): unmatched += 1; continue
    matched += 1
    Xs = sub[feats].fillna(0).values
    prob = m.predict_proba(Xs)[:, 1]
    thr = p["thrs"][(cid, rule)]
    rows.append(sub[prob >= thr].copy())
print(f"pkl→holdout pair match: {matched}/{len(p['mdls'])} (unmatched: {unmatched})")
confirmed = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
print(f"Stage-1 confirmed: {len(confirmed):,}")

if p.get("meta_mdl") is not None and len(confirmed) > 0:
    confirmed["direction"] = confirmed["direction"].astype(int)
    confirmed["cid"] = confirmed["cid"].astype(int)
    Xm = confirmed[meta_feats].fillna(0).values
    pm = p["meta_mdl"].predict_proba(Xm)[:, 1]
    confirmed = confirmed[pm >= p["meta_threshold"]].reset_index(drop=True)
    print(f"Meta passed: {len(confirmed):,}")

if len(confirmed) == 0: print("NO TRADES"); sys.exit()

swing, atr = val.load_swing_with_physics()
trades = val.simulate(confirmed, swing, atr, p["exit_mdl"])
if len(trades) == 0: print("simulate produced 0 trades"); sys.exit()

w = trades[trades["pnl_R"] > 0]; l = trades[trades["pnl_R"] <= 0]
pf = w["pnl_R"].sum() / max(-l["pnl_R"].sum(), 1e-9)
wr = len(w) / len(trades)
eq = trades["pnl_R"].cumsum().values
dd = (np.maximum.accumulate(eq) - eq).max() if len(eq) > 0 else 0
print(f"\nDEPLOYED BTC RESULT: n={len(trades)} WR={wr:.1%} PF={pf:.2f} DD=-{dd:.1f}R Total={trades['pnl_R'].sum():+.0f}R")
