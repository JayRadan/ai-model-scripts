"""
Split labeled EURUSD data into per-cluster CSVs.
C0 Uptrend: drop sells (SELL→FLAT) — only buy signals
C1 Ranging: keep all directions
C2 Downtrend: drop buys (BUY→FLAT) — only sell signals
C3 High Vol: keep all directions
"""
import pandas as pd
import paths as P

df = pd.read_csv(P.data("labeled_eurusd.csv"), parse_dates=["time"])
fp = pd.read_csv(P.data("regime_fingerprints_K4.csv"))

week_to_cluster = dict(zip(fp["week"], fp["cluster"]))
df["week"] = df["time"].dt.isocalendar().year.astype(str) + "-W" + df["time"].dt.isocalendar().week.astype(str).str.zfill(2)
df["cluster"] = df["week"].map(week_to_cluster)
df = df.dropna(subset=["cluster"])
df["cluster"] = df["cluster"].astype(int)

CLUSTER_NAMES = {0: "Uptrend", 1: "Ranging", 2: "Downtrend", 3: "HighVol"}

for cid in range(4):
    sub = df[df["cluster"] == cid].copy()
    if cid == 0:  # Uptrend: drop sells
        sub.loc[sub["entry_class"] == 2, "entry_class"] = 1
    elif cid == 2:  # Downtrend: drop buys
        sub.loc[sub["entry_class"] == 0, "entry_class"] = 1

    out = P.data(f"cluster_{cid}_{CLUSTER_NAMES[cid]}.csv")
    sub.to_csv(out, index=False)

    buys = (sub["entry_class"] == 0).sum()
    sells = (sub["entry_class"] == 2).sum()
    flats = (sub["entry_class"] == 1).sum()
    print(f"C{cid} {CLUSTER_NAMES[cid]:>10}: {len(sub):>7,} rows  BUY={buys:>6,}  FLAT={flats:>6,}  SELL={sells:>6,}")
