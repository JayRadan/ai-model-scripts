# v7.6 kNN Analog Regime Detector — Experiment

## Idea

Instead of compressing the past 288 bars into 5-15 abstract features
(K-means/HMM did this and failed), keep the **raw 288-bar shape** as the
matching unit and search history for similar past patterns.

For every M5 bar:
1. Take the last 288 closes (24h window).
2. Z-normalize to itself (so we match shape, not absolute price).
3. Downsample to 24 points (every 12th bar) to make it tractable.
4. Search **pre-2024-12-12** history for the top-100 most-similar windows.
5. Look at what those analogs did in their next 48 bars (4h forward).
6. If ≥70% went same direction with meaningful magnitude → directional signal.
7. Apply the signal as a veto on Oracle/Midas trades.

## Pipeline

```
00_build_index.py    extract + normalize + downsample all pre-cutoff windows; cKDTree
01_query_holdout.py  for every Oracle/Midas holdout trade, find K=100 analogs +
                     compute their forward-4h aggregate
02_apply_veto.py     apply the analog signal as a veto on the trade lists,
                     compare PF/R/DD baseline vs gated
```

## NO production touch

Reads `data/swing_v5_xauusd.csv` and the existing holdout trade dumps
(`v72l_trades_holdout.csv`, `v6_trades_holdout_xau.csv`). Writes only into
this folder.

## Hypothesis

If the trained Oracle/Midas miss because today's pattern *is* historically
predictive but their 18 v72L features didn't capture that shape, an analog
search will recover the signal. If the pattern is genuinely random
(no shape memory in M5 gold), the analogs' forward returns will average
to ~zero and the experiment will return a clean "no edge" — the correct
answer to a real question.
