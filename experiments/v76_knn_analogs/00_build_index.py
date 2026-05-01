"""
v76 step 00: build the analog-search index.

For each pre-2024-12-12 bar i (with i ≥ 288 + 48):
  • Take closes[i-287..i] (288 bars, including bar i itself).
  • Z-normalize: window = (closes - mean) / std.
  • Downsample to 24 points (every 12th close).
  • Record forward 4h return: fwd_4h = (closes[i+48] - closes[i]) / closes[i].
  • Record forward 8h return: fwd_8h = (closes[i+96] - closes[i]) / closes[i].

Save:
  data/window_vectors.npy   (N × 24 float32)
  data/window_meta.parquet  (N rows: bar_idx, time, fwd_4h, fwd_8h)
  data/kdtree.pkl           (sklearn BallTree, very fast for 24D)
"""
from __future__ import annotations
import os, sys, time as _time, pickle
import numpy as np, pandas as pd
from sklearn.neighbors import BallTree

ZIGZAG = "/home/jay/Desktop/new-model-zigzag"
sys.path.insert(0, os.path.join(ZIGZAG, "model_pipeline"))
import paths as P

OUT = os.path.join(ZIGZAG, "experiments/v76_knn_analogs/data")
SWING = P.data("swing_v5_xauusd.csv")
CUTOFF = pd.Timestamp("2024-12-12 00:00:00")

WINDOW_BARS = 288    # 24h on M5
DOWNSAMPLE  = 12     # take every 12th close → 24-D vector
FWD_4H = 48
FWD_8H = 96
MIN_DATE = "2018-01-01"


def main():
    t0 = _time.time()
    print(f"Loading {SWING}", flush=True)
    df = pd.read_csv(SWING, parse_dates=["time"])
    df = df.sort_values("time").reset_index(drop=True)
    df = df[df["time"] >= MIN_DATE].reset_index(drop=True)
    print(f"  {len(df):,} bars  {df['time'].iloc[0]} → {df['time'].iloc[-1]}")

    closes = df["close"].values.astype(np.float64)
    times  = df["time"].values
    n = len(df)

    print(f"\nExtracting {WINDOW_BARS}-bar windows, downsampled to "
          f"{WINDOW_BARS // DOWNSAMPLE}-D vectors...", flush=True)
    # Indices we keep: need 288 trailing bars + 96 forward + pre-cutoff
    last_train_i = np.searchsorted(times.astype("datetime64[ns]"),
                                    np.datetime64(CUTOFF, "ns")) - 1
    print(f"  cutoff bar index: {last_train_i:,}  ({pd.Timestamp(times[last_train_i])})")

    # eligible indices: i ≥ WINDOW_BARS-1 AND i + FWD_8H < n AND i ≤ last_train_i
    i_min = WINDOW_BARS - 1
    i_max = min(last_train_i, n - FWD_8H - 1)
    print(f"  indexing range: [{i_min:,} ... {i_max:,}]  → {(i_max-i_min):,} windows")

    # Strided extraction. To balance index size vs query precision, take a
    # window EVERY 6 bars (50%-overlap) — huge sample reduction without
    # losing pattern recall (24h shapes don't change every 5 min).
    INDEX_STRIDE = 6
    sample_idx = np.arange(i_min, i_max + 1, INDEX_STRIDE, dtype=np.int64)
    print(f"  stride={INDEX_STRIDE} → {len(sample_idx):,} indexed windows")

    vectors = np.zeros((len(sample_idx), WINDOW_BARS // DOWNSAMPLE), dtype=np.float32)
    fwd_4h  = np.zeros(len(sample_idx), dtype=np.float32)
    fwd_8h  = np.zeros(len(sample_idx), dtype=np.float32)

    for k, i in enumerate(sample_idx):
        win = closes[i - WINDOW_BARS + 1 : i + 1]   # length 288
        m, s = win.mean(), win.std()
        if s < 1e-9:
            continue
        z = (win - m) / s
        vectors[k] = z[::DOWNSAMPLE].astype(np.float32)   # 24 points
        fwd_4h[k] = (closes[i + FWD_4H] - closes[i]) / closes[i]
        fwd_8h[k] = (closes[i + FWD_8H] - closes[i]) / closes[i]
        if k % 50_000 == 0 and k > 0:
            print(f"    {k:,} / {len(sample_idx):,}  ({_time.time()-t0:.0f}s)", flush=True)

    print(f"\nVectors: {vectors.shape}  ({vectors.nbytes/1e6:.1f} MB)")
    np.save(os.path.join(OUT, "window_vectors.npy"), vectors)

    meta = pd.DataFrame({
        "bar_idx": sample_idx, "time": times[sample_idx],
        "fwd_4h": fwd_4h, "fwd_8h": fwd_8h,
    })
    meta.to_parquet(os.path.join(OUT, "window_meta.parquet"), index=False)
    print(f"  saved meta ({len(meta):,} rows)")

    print(f"\nBuilding BallTree index over {len(vectors):,} 24-D vectors...", flush=True)
    t1 = _time.time()
    tree = BallTree(vectors, leaf_size=40, metric="euclidean")
    print(f"  built in {_time.time()-t1:.0f}s")
    with open(os.path.join(OUT, "balltree.pkl"), "wb") as f:
        pickle.dump({"tree": tree, "vectors": vectors, "meta": meta}, f, protocol=4)
    print(f"  saved balltree.pkl  ({os.path.getsize(os.path.join(OUT,'balltree.pkl'))/1e6:.1f} MB)")

    print(f"\nTotal: {_time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
