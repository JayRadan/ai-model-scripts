# Midas XAU — entry-level XAUUSD pattern scanner

Same 28-rule pattern catalog as Oracle, but **no meta gate** — trades
every per-rule confirmation that passes Stage 1. Higher trade count,
lower selectivity, simpler pricing tier.

## At a glance

| | Value |
|---|---|
| Asset | XAUUSD M5 |
| Architecture | v6 (14 physics features) — **no meta gate** |
| Regime detector | K=5 K-means + v9.4 highvol_only relabel (±0.3%/24h) — **v83c: 4h-step** |
| Trained on | Dukascopy 2018-01-02 → 2024-12-12 |
| Holdout window | 2024-12-12 → 2026-05-01 |
| **Holdout PF (v6)** | **2.55** (was 1.67 pre-v9.4) |
| **Holdout PF (v83c)** | **5.25** (+2.70) |
| **Holdout WR (v83c)** | **73.4%** (was 59.7%) |
| **Holdout n (v83c)** | **1,693** (was 1,582) |
| Meta threshold | **0.675** |
| Deployed pkl | `commercial/.../models/midas_xau_validated.pkl` |
| Deploy commit | `bd411c8` (v83c) |

## Why no meta gate?

Midas is positioned as the higher-trade-frequency entry product. The meta
gate would cut ~30-40% of trades to chase higher PF — Midas customers
prefer more frequent activity over Oracle-tier selectivity.

This means **Midas was the biggest beneficiary of the v9.4 regime relabel**:
because there's no Stage 2 to filter bad-regime mistakes, fixing the regime
labeller had a 53% PF impact (1.67 → 2.55).

## Holdout breakdown by cluster

| Cluster | n | WR | PF | Total R |
|---|---|---|---|---|
| C0 Uptrend | 477 | 66.2% | **3.74** | +1,552.1 |
| C3 Downtrend | 626 | 64.9% | **3.58** | +1,932.5 |
| C1 MeanRevert | 155 | 53.5% | 1.50 | +122.8 |

## v83c Holdout (4h regime + range filter + kill-switch)

| Cluster | n | WR | PF | Total R |
|---|---|---|---|---|
| C0 Uptrend  | 982 | 78.2% | **7.48** | +4,108.6 |
| C3 Downtrend | 283 | 70.3% | **4.89** | +917.6 |
| C1 MeanRevert | 35 | 71.4% | 2.98 | +47.8 |
| C2 TrendRange | 51 | 62.7% | 2.19 | +69.4 |
| C4 HighVol | 342 | 64.0% | 2.38 | +525.2 |

Full experiment: `experiments/v83_range_position_filter/`
| C2 TrendRange | 302 | 43.0% | 0.87 | -85.5 |
| C4 HighVol | 22 | 40.9% | 0.90 | -4.7 |

C2/C4 are net-negative — without a meta gate, they leak ~90 R combined.
Argument for cluster-tradeability flags: disable C2 + C4 entirely → projected
PF closer to 3.0, but at cost of trade count.

## Pipeline (retrain from scratch)

**Reuses Oracle XAU's pipeline steps 1-4** (regime selector + cluster
split + setups + physics). Then a different validation script.

### 1-4. Run Oracle XAU steps 1-4 first

```bash
python products/oracle_xau/01_build_selector.py
python products/oracle_xau/02_split_clusters_with_relabel.py
python products/oracle_xau/03_build_setup_signals.py
python products/oracle_xau/04_compute_physics_features.py
```

### 5. Train + validate Midas v6

```bash
python products/midas_xau/01_validate_v6.py
# → writes:
#   - experiments/v6_xau_deploy/_v6_validated_raw.pkl  (raw trained objects)
#   - data/v6_trades_holdout_xau.csv                    (holdout trade tape)
#   - prints PF/WR/DD per cluster
```

### 6. Pickle for deployment

```bash
cd ../my-agents-and-website/commercial/server/decision_engine
python scripts/pickle_validated_models.py midas_xau
# → writes new-model-zigzag/models/midas_xau_validated.pkl
```

### 7. Copy + commit + push (same as Oracle XAU step 8)

## Critical files

```
data/swing_v5_xauusd.csv
data/labeled_v4.csv
data/cluster_{0..4}_data.csv         (relabeled, from Oracle pipeline)
data/setups_{0..4}_v6.csv             (with physics features)
products/midas_xau/_v6_validated_raw.pkl
models/midas_xau_validated.pkl
```

## v9.4 block boundaries shared with Oracle XAU

Midas reuses Oracle XAU's `regime_selector_K4.json`, which carries
the v9.4 `block_boundary_times`. So Midas's live regime decisions
also align to training-time v99 blocks. No separate boundary file
is needed. After every retrain run `_shared/build_block_boundaries.py`
once and Midas inherits the update via the shared selector.
