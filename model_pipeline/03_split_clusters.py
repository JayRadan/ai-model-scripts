"""
Split labeled_v4.csv into one file per regime cluster, applying per-cluster
relabel rules.

v4 label encoding: {0=BUY, 1=FLAT, 2=SELL}
"""
import pandas as pd
import paths as P

BUY, FLAT, SELL = 0, 1, 2

# Per-cluster rules on entry_class:
#  C0 Ranging   — keep all (BUY / FLAT / SELL)
#  C1 Downtrend — kill buys (BUY → FLAT)       → classes {FLAT, SELL}
#  C2 Shock     — keep all (trade both ways, erratic regime)
#  C3 Uptrend   — kill sells (SELL → FLAT)     → classes {BUY, FLAT}
RULES = {
    0: None,
    1: "drop_buys",
    2: None,
    3: "drop_sells",
}

CLUSTER_NAMES = {0: "Ranging", 1: "Downtrend", 2: "Shock_News", 3: "Uptrend"}

print("Loading labeled_v4.csv...")
df = pd.read_csv(P.data("labeled_v4.csv"), parse_dates=["time"])
print(f"  {len(df):,} rows, {len(df.columns)} columns")
print(f"  time range: {df['time'].min()} → {df['time'].max()}")

fp = pd.read_csv(P.data("regime_fingerprints_K4.csv"))

for ci, rule in RULES.items():
    weeks = fp[fp["cluster"] == ci][["week_start", "week_end"]]
    parts = []
    for _, row in weeks.iterrows():
        mask = (df["time"] >= row["week_start"]) & (df["time"] <= row["week_end"])
        parts.append(df[mask])
    cdf = pd.concat(parts).sort_values("time").reset_index(drop=True).copy()

    if rule == "drop_all":
        cdf["entry_class"] = FLAT
    elif rule == "drop_buys":
        cdf.loc[cdf["entry_class"] == BUY, "entry_class"] = FLAT
    elif rule == "drop_sells":
        cdf.loc[cdf["entry_class"] == SELL, "entry_class"] = FLAT

    out = P.data(f"cluster_{ci}_data.csv")
    cdf.to_csv(out, index=False)
    dist = cdf["entry_class"].value_counts().sort_index().to_dict()
    names = {0: "BUY", 1: "FLAT", 2: "SELL"}
    pretty = {names[k]: v for k, v in dist.items()}
    print(f"C{ci} {CLUSTER_NAMES[ci]:<10} {len(cdf):>9,} rows | {pretty}")
