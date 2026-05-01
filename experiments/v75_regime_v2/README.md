# v7.5 Regime Detector v2 — Experiment

**Goal**: build a regime classifier that reacts to regime *shifts* faster than
the current K=5 K-means on 7 features (288-bar window, non-overlapping
training samples).

## Why

On 2026-05-01 gold rallied yesterday and reverted hard today. The current
classifier said C4 HighVol with 288-bar return still positive (yesterday's
rally dominates the trailing 24h). Option D's directional gate armed R3*
uptrend rules and all 3 products bought the falling market.

The current system is **regime-shift blind** — K=5 K-means trained on
non-overlapping 24h snapshots can't capture transitions, only steady states.

## Approach

1. **Richer features (15 per bar)** — multi-scale momentum (1h / 3h / 24h),
   slope acceleration, volatility ratio, structural HH/LL counts, position
   within recent range, reversal indicators.
2. **Per-bar rolling training samples** (`STEP=1` instead of 288) — every
   bar is a potential cluster center, so transitions get represented.
3. **K-means K∈{5,6,7,8}** plus a stretch HMM run — pick whichever gives
   the cleanest regime *flips* on known shift days (2026-04-30 evening,
   2026-05-01 morning).
4. **Validation**: side-by-side comparison vs production v1 classifier on
   the last 90 days. If the new classifier flips earlier on shift days
   without thrashing during normal trends, it's a win.
5. **Optional**: retrain the per-cluster confirm heads on the new clusters
   and run on the full holdout for a PF comparison vs the validated 3.48.

## Scripts (in order)

```
00_compute_rich_fingerprints.py   # build the 15-feature per-bar matrix
01_train_kmeans.py                # train K∈{5..8}, save selectors
02_compare_to_v1.py               # plot v1 vs v2 cluster timelines + shift-day diagnostics
03_apply_to_holdout.py (optional) # full backtest with new clusters
```

## NO production touch

Reads `data/swing_v5_xauusd.csv` only. Writes to this folder only. The live
server, Connector EAs, website, etc. are untouched.
