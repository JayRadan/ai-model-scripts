"""
Truest possible replication: fetch DJI bars with the server's OWN
dukascopy_source._fetch_dukascopy (same offer_side, same 8700-bar window,
same in-progress-bar drop), then recompute Q for today's 3 live trade bars
and compare to the funnel-logged Q.

If this matches the live log → the offline sim just needs to source bars the
same way (8700-bar window via dukascopy_source). If it STILL differs → the
residual is Dukascopy real-time revision (irreducible offline).
"""
import sys, pickle
from pathlib import Path
import numpy as np, pandas as pd

DE = Path("/home/jay/Desktop/my-agents-and-website/commercial/server/decision_engine")
sys.path.insert(0, str(DE))
import hermes_features as hf
import dukascopy_source as duk
from configs.hermes_dji import HERMES_DJI as CFG
B = pickle.load(open(DE / "models/hermes_dji_validated.pkl", "rb"))
QM, FC = B["q_model"], B["feat_cols"]

# Fetch EXACTLY like the server: 8700 M1 bars, OFFER_SIDE_BID, drop in-progress.
N = duk.TARGET_BARS
print(f"fetching {N} M1 DJI bars via server dukascopy_source ...")
df = duk._fetch_dukascopy("DJIUSD", n_bars=N, interval_min=1)
print(f"  got {len(df):,} bars, {df.time.iloc[0]} → {df.time.iloc[-1]}")

targets = ["2026-06-29 13:28:00", "2026-06-29 13:59:00", "2026-06-29 15:26:00"]
live_q  = {"2026-06-29 13:28:00": 3.18, "2026-06-29 13:59:00": 3.00, "2026-06-29 15:26:00": 3.35}

def qrow(frame, ts):
    r = frame[frame["time"] == pd.Timestamp(ts)]
    if len(r) == 0: return None, None
    X = r[FC].fillna(0).to_numpy(np.float32)
    return float(QM.predict(X)[0]), r.iloc[0]

print(f"\n{'bar (UTC)':<22}{'LIVE log':>9}{'server-src incr':>17}{'dist_abs':>10}{'cdir':>6}{'tfk_line':>11}")
for ts in targets:
    sub = df[df["time"] <= pd.Timestamp(ts)].copy()
    inc = hf.compute_all_features(sub, CFG)
    q, row = qrow(inc, ts)
    if q is None:
        print(f"{ts:<22}  (bar not in fetched series)"); continue
    print(f"{ts:<22}{live_q[ts]:>9.2f}{q:>17.2f}{float(row['dist_abs']):>10.3f}"
          f"{int(row['committed_dir']):>6}{float(row['tfk_line']):>11.1f}")

print("\nClose to live log → fix = sim must source bars via dukascopy_source (8700-bar window).")
print("Still off → residual is Dukascopy real-time revision; offline can't reproduce a past live tick.")
