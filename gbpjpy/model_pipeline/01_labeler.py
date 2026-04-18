"""
GBPJPY M5 causal swing labeler.
Reads swing_GBPJPY_5m.csv, adds proper BUY/FLAT/SELL labels based on
forward TP/SL outcome, outputs labeled_gbpjpy.csv.
"""
import pandas as pd
import numpy as np
import paths as P

# --- Parameters ---
DATA_PATH = P.data("swing_v5_gbpjpy.csv")
OUT_PATH = P.data("labeled_gbpjpy.csv")
MIN_DATE = "2016-01-01"

LOOKBACK = 5
PRIOR_WINDOW = 15
MIN_SWING_ATR = 0.8
TP_MULT = 2.0
SL_MULT = 1.0
MAX_FWD = 40
SPREAD_PIPS = 3.0
POINT = 0.001  # GBPJPY is 3-decimal

print("Loading data...")
df = pd.read_csv(DATA_PATH)
df["time"] = pd.to_datetime(df["time"], format="%Y.%m.%d %H:%M")
df = df.sort_values("time").reset_index(drop=True)

print(f"  Total rows: {len(df):,}")
df = df[df["time"] >= MIN_DATE].reset_index(drop=True)
print(f"  After {MIN_DATE} filter: {len(df):,}")

# Compute ATR-14
highs = df["high"].values
lows = df["low"].values
closes = df["close"].values

tr = np.maximum(highs - lows,
     np.maximum(np.abs(highs - np.roll(closes, 1)),
                np.abs(lows - np.roll(closes, 1))))
tr[0] = highs[0] - lows[0]
atr14 = pd.Series(tr).rolling(14).mean().values

# Spread in price terms
spread_price = SPREAD_PIPS * POINT

print("Labeling with forward TP/SL outcome...")
n = len(df)
labels = np.ones(n, dtype=int)  # default FLAT=1

for i in range(PRIOR_WINDOW, n - MAX_FWD):
    a = atr14[i]
    if np.isnan(a) or a < 1e-10:
        continue

    tp_long = closes[i] + TP_MULT * a + spread_price
    sl_long = closes[i] - SL_MULT * a
    tp_short = closes[i] - TP_MULT * a - spread_price
    sl_short = closes[i] + SL_MULT * a

    # Check long
    for k in range(1, MAX_FWD + 1):
        if highs[i + k] >= tp_long:
            labels[i] = 0  # BUY
            break
        if lows[i + k] <= sl_long:
            break

    if labels[i] == 1:
        # Check short
        for k in range(1, MAX_FWD + 1):
            if lows[i + k] <= tp_short:
                labels[i] = 2  # SELL
                break
            if highs[i + k] >= sl_short:
                break

df["entry_class"] = labels

counts = df["entry_class"].value_counts().sort_index()
total = len(df)
print(f"\nLabel distribution:")
print(f"  BUY  (0): {counts.get(0, 0):>7,}  ({100*counts.get(0,0)/total:.1f}%)")
print(f"  FLAT (1): {counts.get(1, 0):>7,}  ({100*counts.get(1,0)/total:.1f}%)")
print(f"  SELL (2): {counts.get(2, 0):>7,}  ({100*counts.get(2,0)/total:.1f}%)")

df.to_csv(OUT_PATH, index=False)
print(f"\nSaved: {OUT_PATH}")
