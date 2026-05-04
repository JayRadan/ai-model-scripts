"""Test deployed Midas v6 pkl against fresh setups (14 feats, no meta)."""
import os, sys, pickle, glob
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/jay/Desktop/new-model-zigzag/experiments/v6_xau_deploy")
import importlib
val = importlib.import_module("01_validate_v6")

COMM = "/home/jay/Desktop/my-agents-and-website/commercial/server/decision_engine/models"
ZIGZAG = "/home/jay/Desktop/new-model-zigzag"

p = pickle.load(open(f"{COMM}/midas_xau_validated.pkl", "rb"))
print(f"deployed Midas pkl: heads={len(p['mdls'])}  feats={len(p['v72l_feats'])}")

feats = list(p["v72l_feats"])

# Midas uses per-rule chronological 80/20, not global cutoff.
setups = []
for f in sorted(glob.glob(f"{ZIGZAG}/data/setups_*_v6.csv")):
    base = os.path.basename(f).replace(".csv","")
    import re
    if not re.fullmatch(r"setups_\d+_v6", base): continue
    cid = int(os.path.basename(f).split("_")[1])
    df = pd.read_csv(f, parse_dates=["time"])
    df["cid"] = cid
    setups.append(df)
setups = pd.concat(setups, ignore_index=True).sort_values("time").reset_index(drop=True)
setups = setups.loc[:, ~setups.columns.duplicated()]
print(f"all setups: {len(setups):,}")

# Per-rule 80/20 split
test_rows = []
for (cid, rule), m in p["mdls"].items():
    sub = setups[(setups["cid"] == cid) & (setups["rule"] == rule)].sort_values("time").reset_index(drop=True)
    if len(sub) < 5: continue
    s = int(len(sub) * 0.8)
    test_rows.append(sub.iloc[s:])
test_setups = pd.concat(test_rows, ignore_index=True) if test_rows else pd.DataFrame()
print(f"holdout setups (per-rule 20%): {len(test_setups):,}")

rows = []; matched = 0
for (cid, rule), m in p["mdls"].items():
    sub = test_setups[(test_setups["cid"] == cid) & (test_setups["rule"] == rule)]
    if not len(sub): continue
    matched += 1
    Xs = sub[feats].fillna(0).values
    prob = m.predict_proba(Xs)[:, 1]
    thr = p["thrs"][(cid, rule)]
    rows.append(sub[prob >= thr].copy())
print(f"pkl→holdout pair match: {matched}/{len(p['mdls'])}")
confirmed = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
print(f"Stage-1 confirmed: {len(confirmed):,}")

if len(confirmed) == 0: print("NO TRADES"); sys.exit()

swing, atr = val.load_swing_with_physics()
trades = val.simulate(confirmed, swing, atr, p["exit_mdl"])
if len(trades) == 0: print("simulate produced 0 trades"); sys.exit()

w = trades[trades["pnl_R"] > 0]; l = trades[trades["pnl_R"] <= 0]
pf = w["pnl_R"].sum() / max(-l["pnl_R"].sum(), 1e-9)
wr = len(w) / len(trades)
eq = trades["pnl_R"].cumsum().values
dd = (np.maximum.accumulate(eq) - eq).max() if len(eq) > 0 else 0
print(f"\nDEPLOYED MIDAS RESULT: n={len(trades)} WR={wr:.1%} PF={pf:.2f} DD=-{dd:.1f}R Total={trades['pnl_R'].sum():+.0f}R")
