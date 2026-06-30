"""
Resolve the Hermes DJI Q mystery: does the LIVE server leak HTF look-ahead, or not?

Live funnel logged 3 HRD-L opens today at q = 3.00 / 3.18 / 3.35
(bars 13:28, 13:59, 15:26 UTC). The 3-day sim said max Q 2.46.

Compute Q for those exact bars THREE ways:
  (A) BATCH    : compute_all_features over the WHOLE frame (resample+ffill leaks future)
  (B) INCREMENT: compute_all_features on bars[:t+1] only  (== what the server does live)
  (C) Δ shows whether live (==B) is leak-free, and how far batch (A) inflates it.
"""
import sys, pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np, pandas as pd

DE = Path("/home/jay/Desktop/my-agents-and-website/commercial/server/decision_engine")
sys.path.insert(0, str(DE))
import hermes_features as hf
from configs.hermes_dji import HERMES_DJI as CFG
B = pickle.load(open(DE / "models/hermes_dji_validated.pkl", "rb"))
QM, FC = B["q_model"], B["feat_cols"]

import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_IDX_AMERICA_E_D_J_IND
end = datetime.now(timezone.utc); start = end - timedelta(days=6)
raw = dukascopy_python.fetch(instrument=INSTRUMENT_IDX_AMERICA_E_D_J_IND,
        interval=dukascopy_python.INTERVAL_MIN_1, offer_side=dukascopy_python.OFFER_SIDE_BID,
        start=start, end=end)
df = raw.reset_index().rename(columns={"timestamp": "time"})
df["time"] = pd.to_datetime(df["time"]).dt.tz_convert("UTC").dt.tz_localize(None)
df = df.sort_values("time").reset_index(drop=True)
if "tick_volume" not in df.columns:
    vc = [c for c in df.columns if "vol" in c.lower()][0]; df["tick_volume"] = df[vc]
print(f"fetched {len(df):,} bars, {df.time.iloc[0]} → {df.time.iloc[-1]}")

targets = ["2026-06-29 13:28:00", "2026-06-29 13:59:00", "2026-06-29 15:26:00"]
live_q  = {targets[0]: 3.18, targets[1]: 3.00, targets[2]: 3.35}  # from funnel (bar_ts)

# (A) batch
batch = hf.compute_all_features(df.copy(), CFG)
def qrow(frame, ts):
    r = frame[frame["time"] == pd.Timestamp(ts)]
    if len(r) == 0: return None
    X = r[FC].fillna(0).to_numpy(np.float32)
    return float(QM.predict(X)[0])

print(f"\n{'bar (UTC)':<22}{'LIVE log':>9}{'BATCH(A)':>10}{'INCR(B)':>10}{'B-batchΔ':>10}")
for ts in targets:
    qb = qrow(batch, ts)
    # (B) incremental: only bars up to and including ts
    sub = df[df["time"] <= pd.Timestamp(ts)].copy()
    inc = hf.compute_all_features(sub, CFG)
    qi = qrow(inc, ts)
    lv = live_q[ts]
    print(f"{ts:<22}{lv:>9.2f}{qb:>10.2f}{qi:>10.2f}{qi-qb:>10.2f}")

print("\nIf INCR(B) ≈ LIVE log  → live is leak-free (causal); batch backtests over-state.")
print("If BATCH(A) ≈ LIVE log → live itself leaks (real bug).")
