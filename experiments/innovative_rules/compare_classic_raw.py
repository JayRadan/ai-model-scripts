"""
Compare raw base-rate (no ML) of existing XAU V5 rules vs innovative ones.
If classic rules also fail raw, Phase 1 alone isn't predictive — we
need Phase 2 (ML confirmation) to decide.
"""
import glob, os
import numpy as np
import pandas as pd

ROOT = "/home/jay/Desktop/new-model-zigzag"
TP_MULT, SL_MULT = 6.0, 2.0

print(f"{'rule':<30} {'n':<6} {'WR':<6}  {'raw PF (6:2)':<12}")
print("-" * 62)
for f in sorted(glob.glob(f"{ROOT}/data/setups_*.csv")):
    cid = os.path.basename(f).replace("setups_","").replace(".csv","")
    df = pd.read_csv(f)
    for rule, grp in df.groupby("rule"):
        if "label" not in grp.columns: continue
        lbl = grp["label"].astype(int).values
        if len(lbl) < 50: continue
        wr = lbl.mean()
        pf = (wr * TP_MULT) / ((1-wr) * SL_MULT + 1e-9)
        flag = "✓" if pf >= 1.0 else "✗"
        print(f"  C{cid} {rule:<24} {len(lbl):<6} {wr:.1%}  {pf:<6.2f}  {flag}")

print("\n(Note: existing labels were computed at TP=2, SL=1. PF at 6:2 is a projection;\n"
      "raw WR is the honest number.)")
