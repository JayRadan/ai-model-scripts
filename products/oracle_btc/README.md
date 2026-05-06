# Oracle BTC — XAU's Oracle architecture ported to BTCUSD

Same v7.2-lite two-stage architecture as Oracle XAU, but uses its own
BTC-specific regime selector and the **full directional override**
relabel rule (because BTC's K-means produces a different cluster
structure than gold).

## At a glance

| | Value |
|---|---|
| Asset | BTCUSD M5 |
| Architecture | v7.2-lite (18 features) + per-cluster confirm + meta gate |
| Regime detector | K=5 K-means + v9.4 full_directional relabel — **v83c: 4h-step** |
| Trained on | Dukascopy 2018-01-02 → 2024-12-12 |
| Holdout window | 2024-12-12 → 2026-05-01 |
| **Holdout PF (v7.2)** | **2.85** |
| **Holdout PF (v83c)** | **3.03** (+0.18) |
| **Holdout WR (v83c)** | **64.4%** (was 61.8%) |
| **Holdout n (v83c)** | **1,033** (was 1,899) |
| Meta threshold | **0.625** |
| Deployed pkl | `commercial/.../models/oracle_btc_validated.pkl` |
| Deploy commit | `bd411c8` (v83c) |

## Why BTC needs a different relabel rule

BTC's K-means with 8 fingerprint features (same as XAU) produces a
**MeanRevert** cluster that's actually capturing trending periods —
not just choppy ones. Inspection of price during MeanRevert-labeled
windows showed clearly directional moves. The XAU-style highvol-only
relabel (only retag C4) is insufficient.

Solution: **full directional override** — if any window's
`weekly_return_pct` exceeds ±1% (over 24h), retag it as Up/Down
regardless of which K-means cluster K-means picked. The 1% threshold
is wider than XAU's 0.3% because BTC's natural M5 volatility is much
higher.

After relabel:
- C0 Uptrend grew from 13% → 28% of bars
- C3 Downtrend grew from 19% → 28%
- C4 HighVol shrunk from 6% → 1% (almost gone — BTC's HighVol windows
  were almost all directional)

## Architecture (identical to Oracle XAU)

```
M5 BTC bar  ──►  K=5 BTC regime selector ──►  cluster id
                  + v9.4 full_directional:        │
                  any cluster → Up/Down           │
                  if |24h ret| > 1%               ▼
                                          per-cluster rule scanner
                                          (28 rules)
                                                │
                                                ▼
                                          per-rule confirm head
                                          p_conf >= per-rule thr ?
                                                │ pass
                                                ▼
                                          meta gate (XGB)
                                          p_win >= 0.600 ?
                                                │ pass
                                                ▼
                                            OPEN TRADE
```

## Holdout breakdown by cluster

| Cluster | n | WR | PF | Total R |
|---|---|---|---|---|
| C0 Uptrend | 1,409 | 63.8% | **3.20** | +3,876.6 |
| C3 Downtrend | 179 | 67.0% | **3.99** | +563.1 |
| C1 MeanRevert | 86 | 53.5% | 1.29 | +28.9 |
| C2 TrendRange | 194 | 47.4% | 1.21 | +76.3 |
| C4 HighVol | 31 | 51.6% | 1.36 | +19.3 |

C0 carries 74% of trades and the bulk of profit — exactly as the
directional override intended. The 24-hour BTC return distribution is
heavily skewed toward up moves over this holdout, so C0 dominates.

## Pipeline (retrain from scratch)

### 1. Build BTC-specific regime selector

```bash
python products/oracle_btc/01_build_selector_btc.py
# → writes data/regime_selector_btc_K5.json
# → adds "relabel": {"mode":"full_directional","threshold":0.01,
#                    "up_cid":0,"down_cid":3}
```

### 2. Prepare BTC bars (compute tech features + assign cluster + split)

```bash
python products/oracle_btc/02_prepare_btc.py
# → writes data/cluster_{0..4}_data_btc.csv
# Contains the v9.4 full directional override INSIDE the cluster
# assignment loop — every window with |return| > 1% is retagged.
```

### 3. Build setup signals

```bash
python products/oracle_btc/03_build_setup_signals_btc.py
# → writes data/setups_{0..4}_btc.csv
```

### 4. Compute physics features (14 features)

```bash
python products/oracle_btc/04_compute_physics_features_btc.py
# → writes data/setups_{0..4}_v6_btc.csv
```

### 5. Compute v7.2-lite features

```bash
python products/oracle_btc/05_compute_v72l_features_btc.py
# → writes data/setups_{0..4}_v72l_btc.csv
```

### 6. Train + validate

```bash
python products/oracle_btc/06_validate_v72_lite_btc.py
# → writes:
#   - meta_threshold_v72l_btc.txt (0.600)
#   - data/v72l_trades_holdout_btc.csv
#   - prints PF/WR/DD per cluster
```

### 7. Pickle for deployment

```bash
cd ../my-agents-and-website/commercial/server/decision_engine
rm -f /tmp/oracle_btc_pipeline_cache.pkl
python scripts/pickle_validated_models.py oracle_btc
# → writes new-model-zigzag/models/oracle_btc_validated.pkl
```

### 8. Copy + commit + push (same as Oracle XAU step 8)

## Critical files

```
data/swing_v5_btc.csv                        ← raw BTC M5 + tick_volume
data/regime_selector_btc_K5.json             ← BTC selector + v9.4 relabel
data/cluster_{0..4}_data_btc.csv             ← post-relabel split
data/setups_{0..4}_v72l_btc.csv              ← all features
models/oracle_btc_validated.pkl              ← current production
```

## After retrain: regenerate block boundaries

```bash
python products/_shared/build_block_boundaries.py    # rebuilds + embeds
cp data/regime_selector_btc_K5.json \
   ../my-agents-and-website/commercial/server/decision_engine/data/
```

Without fresh boundaries, live `/decide` for BTC freezes on the last
training-time block until the cache extrapolation kicks in (which only
works if the cache reaches back to the last embedded boundary).

## Live performance caveat

Models trained on Dukascopy bars; customer brokers send their own bars.
Live PF may differ from backtest. BTC volume is especially broker-dependent
(some brokers report tick volume, others volume-of-quotes, etc.) so the
flow_4h_mean feature can land in different places. This is one reason the
deployed PF (2.98 → 2.85) regressed slightly with v9.4 — but DD improvement
(-78 → -50, **-36%**) more than makes up for it.

Path A (server-side Dukascopy fetch) is enabled in production now — closes
this gap for live decisions.

## v83c Holdout (4h regime + range filter + kill-switch)

| Cluster | n | WR | PF | Total R |
|---|---|---|---|---|
| C0 Uptrend  | 295 | 74.6% | **5.83** | +1,108.0 |
| C3 Downtrend | 222 | 65.3% | **3.89** | +649.4 |
| C1 MeanRevert | 250 | 60.4% | 1.78 | +223.5 |
| C2 TrendRange | 172 | 57.0% | 1.65 | +152.2 |
| C4 HighVol | 94 | 54.3% | 2.08 | +171.0 |

Full experiment: `experiments/v83_range_position_filter/`
