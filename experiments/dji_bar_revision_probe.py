"""
Measure Dukascopy recent-bar REVISION: fetch the last few hours of DJI M1,
wait, re-fetch, and diff overlapping bars. If the most-recent K bars change
between two fetches taken minutes apart, that proves the 'provisional bar'
instability — and quantifies how many bars of LAG are needed before a bar
is settled (Jay's idea: decide on settled data, not the freshest bar).
"""
import sys, time, pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np, pandas as pd

DE = Path("/home/jay/Desktop/my-agents-and-website/commercial/server/decision_engine")
sys.path.insert(0, str(DE))
import dukascopy_source as duk

def fetch():
    return duk._fetch_dukascopy("DJIUSD", n_bars=400, interval_min=1)

print("fetch #1 ..."); a = fetch(); t1 = a.time.iloc[-1]
print(f"  last bar {t1}, {len(a)} bars")
print("waiting 150s for new/revised ticks ...")
time.sleep(150)
print("fetch #2 ..."); b = fetch(); t2 = b.time.iloc[-1]
print(f"  last bar {t2}, {len(b)} bars")

m = a.merge(b, on="time", suffixes=("_1", "_2"))
changed = []
for col in ["open", "high", "low", "close"]:
    d = (m[f"{col}_1"] - m[f"{col}_2"]).abs()
    nz = m.loc[d > 1e-6, ["time"]].copy()
    for ts in nz["time"]:
        changed.append((ts, col))
if not changed:
    print("\nNo overlapping bars changed between the two fetches.")
    print(f"(new bars appeared: {sorted(set(b.time)-set(a.time))[-5:] if set(b.time)-set(a.time) else 'none — market may be closed'})")
else:
    cdf = pd.DataFrame(changed, columns=["time", "col"])
    print(f"\n{len(cdf)} OHLC values revised across {cdf.time.nunique()} bars:")
    for ts, g in cdf.groupby("time"):
        row1 = m[m.time == ts].iloc[0]
        mins_old = (t2 - ts).total_seconds() / 60
        deltas = {c: f"{row1[f'{c}_1']:.2f}->{row1[f'{c}_2']:.2f}" for c in g.col}
        print(f"  {ts}  ({mins_old:>4.0f} min before last bar)  {deltas}")
    youngest_stable = None
    print("\n=> bars within ~{} min of the live edge are still moving."
          .format(int((t2 - cdf.time.min()).total_seconds()/60)))
