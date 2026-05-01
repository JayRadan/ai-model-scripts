"""
v76 step 01: for every Oracle/Midas holdout trade, build the trade bar's
288-bar normalized + downsampled signature, query top-K nearest analogs,
aggregate their forward 4h/8h returns into a directional signal.

Output:  data/analog_signals.parquet  (one row per unique trade time)
         columns: time, bar_idx, n_neighbors, mean_fwd_4h, std_fwd_4h,
                  pct_pos_4h, pct_neg_4h, mean_fwd_8h, …
"""
from __future__ import annotations
import os, sys, time as _time, pickle
import numpy as np, pandas as pd

ZIGZAG = "/home/jay/Desktop/new-model-zigzag"
sys.path.insert(0, os.path.join(ZIGZAG, "model_pipeline"))
import paths as P

EXP = os.path.join(ZIGZAG, "experiments/v76_knn_analogs")

ORACLE_HOLDOUT = P.data("v72l_trades_holdout.csv")
MIDAS_HOLDOUT  = P.data("v6_trades_holdout_xau.csv")
SWING          = P.data("swing_v5_xauusd.csv")

WINDOW_BARS = 288
DOWNSAMPLE  = 12
K_NEIGHBORS = 100


def main():
    t0 = _time.time()

    print("Loading swing + index...", flush=True)
    df = pd.read_csv(SWING, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)
    closes = df["close"].values.astype(np.float64)
    time_to_idx = {t: i for i, t in enumerate(df["time"].values)}

    with open(os.path.join(EXP, "data/balltree.pkl"), "rb") as f:
        bundle = pickle.load(f)
    tree = bundle["tree"]
    meta = bundle["meta"]
    fwd4 = meta["fwd_4h"].values.astype(np.float64)
    fwd8 = meta["fwd_8h"].values.astype(np.float64)
    print(f"  index: {len(meta):,} historical windows")

    print("\nLoading holdout trade times (Oracle + Midas)...", flush=True)
    trade_times = []
    for csv in [ORACLE_HOLDOUT, MIDAS_HOLDOUT]:
        d = pd.read_csv(csv, parse_dates=["time"])
        trade_times.append(d["time"])
    all_times = pd.concat(trade_times).drop_duplicates().sort_values().reset_index(drop=True)
    print(f"  {len(all_times):,} unique trade timestamps")

    # Build query matrix
    print("\nBuilding query vectors...", flush=True)
    query_idx = []
    queries = []
    for t in all_times:
        i = time_to_idx.get(t.to_datetime64())
        if i is None or i < WINDOW_BARS - 1: continue
        win = closes[i - WINDOW_BARS + 1 : i + 1]
        m, s = win.mean(), win.std()
        if s < 1e-9: continue
        z = (win - m) / s
        queries.append(z[::DOWNSAMPLE].astype(np.float32))
        query_idx.append(i)
    queries = np.stack(queries)
    query_idx = np.array(query_idx)
    print(f"  built {len(queries):,} queries (24-D)")

    print(f"\nKNN search K={K_NEIGHBORS} for {len(queries):,} queries...", flush=True)
    t1 = _time.time()
    dist, ind = tree.query(queries, k=K_NEIGHBORS)
    print(f"  done in {_time.time()-t1:.0f}s")

    # Aggregate per query
    print("\nAggregating analog forward returns...", flush=True)
    n_q = len(queries)
    out = []
    for q in range(n_q):
        nb_fwd4 = fwd4[ind[q]]
        nb_fwd8 = fwd8[ind[q]]
        out.append({
            "bar_idx": int(query_idx[q]),
            "time":    df["time"].iloc[query_idx[q]],
            "mean_dist": float(dist[q].mean()),
            "min_dist":  float(dist[q].min()),
            "n_neighbors": K_NEIGHBORS,
            "mean_fwd_4h": float(nb_fwd4.mean()),
            "std_fwd_4h":  float(nb_fwd4.std()),
            "pct_pos_4h":  float((nb_fwd4 > 0.001).mean()),
            "pct_neg_4h":  float((nb_fwd4 < -0.001).mean()),
            "mean_fwd_8h": float(nb_fwd8.mean()),
            "std_fwd_8h":  float(nb_fwd8.std()),
            "pct_pos_8h":  float((nb_fwd8 > 0.001).mean()),
            "pct_neg_8h":  float((nb_fwd8 < -0.001).mean()),
        })
    sigdf = pd.DataFrame(out)
    sigdf.to_parquet(os.path.join(EXP, "data/analog_signals.parquet"), index=False)
    print(f"  saved {len(sigdf):,} rows → analog_signals.parquet")

    # Quick sanity: distribution of pct_neg_4h
    print(f"\nDistribution of pct_neg_4h across queries:")
    for q in [0.1, 0.25, 0.5, 0.75, 0.9, 0.95]:
        print(f"   {q*100:>4.0f}%-ile: {sigdf['pct_neg_4h'].quantile(q):.3f}")
    print(f"\nQueries with strong bearish (pct_neg_4h ≥ 0.60): {(sigdf['pct_neg_4h']>=0.60).sum()}")
    print(f"Queries with strong bullish  (pct_pos_4h ≥ 0.60): {(sigdf['pct_pos_4h']>=0.60).sum()}")
    print(f"Queries with ambiguous        (both <0.50):       "
          f"{((sigdf['pct_pos_4h']<0.5)&(sigdf['pct_neg_4h']<0.5)).sum()}")

    print(f"\nTotal: {_time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
