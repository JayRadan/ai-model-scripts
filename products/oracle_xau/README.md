# Oracle XAU — flagship two-stage XAUUSD model

The premium product. Per-cluster rule scanner + per-rule confirmation heads
(28 active rules) + meta-labeling gate → cleanest signals, highest PF, the
strictest filtering of the four products.

## At a glance

| | Value |
|---|---|
| Asset | XAUUSD M5 |
| Architecture | v7.2-lite (18 v72l features) + per-cluster confirm + meta gate |
| Regime detector | K=5 K-means + v9.4 highvol_only relabel (±0.3%/24h) — **v83c: 4h-step** |
| Trained on | Dukascopy 2018-01-02 → 2024-12-12 |
| Holdout window | 2024-12-12 → 2026-05-01 |
| **Holdout PF (v7.2)** | **3.45** |
| **Holdout PF (v83c)** | **4.18** (+0.73) |
| **Holdout WR (v83c)** | **69.1%** (was 65.9%) |
| **Holdout n (v83c)** | **1,487** (was 1,903) |
| Meta threshold | **0.775** |
| Deployed pkl | `commercial/.../models/oracle_xau_validated.pkl` |
| Deploy commit | `bd411c8` (v83c — 4h regime + range filter + kill-switch) |

## Architecture

```
M5 bar  ──►  K=5 regime selector ──►  cluster id (0..4)
                  + v9.4 relabel:        │
                  HighVol → Down/Up      │
                  if |24h ret| > 0.3%    ▼
                                    per-cluster rule scanner
                                    (28 rules, gated by cluster)
                                          │
                                          ▼
                                    per-rule confirm head (XGB)
                                    p_conf >= per-rule threshold ?
                                          │ pass
                                          ▼
                                    meta gate (XGB, 18 v72l + cid + dir)
                                    p_win >= 0.675 ?
                                          │ pass
                                          ▼
                                       OPEN TRADE
```

Per-cluster trade-rule allocation (after relabel):

| Cluster | Tradeable rules | Direction filter |
|---|---|---|
| C0 Uptrend | R3a-R3i (9 buy patterns) | longs only |
| C1 MeanRevert | R0a-R0i (9 mean-revert) | both |
| C2 TrendRange | R0d/e/g/h (4 breakouts) | both |
| C3 Downtrend | R1a-R1h (8 sell patterns) | shorts only |
| C4 HighVol | R2a/b/c/d (4 vol-breakout) | both — but rare after relabel |

## Holdout breakdown by cluster

| Cluster | n | WR | PF | Total R |
|---|---|---|---|---|
| C0 Uptrend  | 1,064 | 68.6% | **3.75** | +3,392.7 |
| C3 Downtrend | 634 | 67.2% | **4.23** | +2,162.2 |
| C1 MeanRevert | 51 | 47.1% | 0.98 | -1.8 |
| C2 TrendRange | 36 | 50.0% | 1.10 | +5.3 |
| C4 HighVol | 118 | 47.5% | 0.90 | -21.7 |

C0 + C3 carry 89% of trades and ~95% of profit — the relabel intentionally
funnels strongly-directional bars into these two clusters.

## v83c Holdout (4h regime + range filter + kill-switch)

| Cluster | n | WR | PF | Total R |
|---|---|---|---|---|
| C0 Uptrend  | 641 | 74.9% | **6.63** | +2,725.6 |
| C3 Downtrend | 273 | 75.8% | **6.58** | +1,101.9 |
| C1 MeanRevert | 77 | 64.9% | 2.31 | +79.7 |
| C2 TrendRange | 89 | 50.6% | 1.03 | +3.4 |
| C4 HighVol | 407 | 60.4% | 2.03 | +547.2 |

Improvements from v7.2:
- **4h-step regime** (window=288, step=48) — catches regime changes 6× faster
- **Range-position filter** — skips R1a/R1b C3 shorts below 65% of 20-bar range, R3e C0 longs above 35%
- **Consecutive-loss kill-switch** — 3 SL in same (regime, dir) → 12h cooldown

Full experiment: `experiments/v83_range_position_filter/`

## Pipeline (retrain from scratch)

Step numbers correspond to script filenames. Run from the **repo root**
unless noted.

### 1. Build the regime selector (K-means on 8 features incl. flow_4h_mean)

```bash
python products/oracle_xau/01_build_selector.py
# → writes data/regime_selector_K4.json + data/regime_fingerprints_K4.csv
```

The selector JSON contains:
- `feat_names`: 8-element list (last one is `flow_4h_mean`)
- `centroids`: 5 PCA-space centroids
- `cluster_names`: {"0":"Uptrend", "1":"MeanRevert", "2":"TrendRange",
                     "3":"Downtrend", "4":"HighVol"}
- `relabel`: `{"mode":"highvol_only","threshold":0.003,"highvol_cid":4,
              "up_cid":0,"down_cid":3}` ← **v9.4 critical**

### 2. Split labeled bars into per-cluster CSVs (with v9.4 relabel)

```bash
python products/oracle_xau/02_split_clusters_with_relabel.py
# → writes data/cluster_{0..4}_data.csv
```

Note: this uses the post-hoc relabel rule (HighVol → Down/Up by return
sign) so per-cluster training data matches what live inference will see.

### 3. Build setup signals (per-rule pattern scanner)

```bash
python products/oracle_xau/03_build_setup_signals.py
# → writes data/setups_{0..4}.csv
```

### 4. Compute physics features (14 features incl. quantum_flow MTF)

```bash
python products/oracle_xau/04_compute_physics_features.py
# → writes data/setups_{0..4}_v6.csv
```

### 5. Compute v7.2-lite features (4 extras: vpin, sig_quad_var,
   har_rv_ratio, hawkes_eta)

```bash
python products/oracle_xau/05_compute_v72l_features.py
# → writes data/setups_{0..4}_v72l.csv
```

### 6. Train + validate (per-rule confirm + exit + meta + threshold sweep)

```bash
python products/oracle_xau/06_validate_v72_lite.py
# → writes:
#   - meta_threshold_v72l.txt (selected threshold; current: 0.675)
#   - data/v72l_trades_holdout.csv (full holdout trade tape)
#   - prints PF/WR/DD per cluster
```

### 7. Pickle for deployment (bundle all models + threshold)

```bash
cd ../my-agents-and-website/commercial/server/decision_engine
rm -f /tmp/oracle_deployed_pipeline_cache.pkl
python scripts/pickle_validated_models.py oracle_xau
# → writes new-model-zigzag/models/oracle_xau_validated.pkl
```

### 8. Copy to commercial repo + tag rollback + push

```bash
cp new-model-zigzag/models/oracle_xau_validated.pkl \
   ../my-agents-and-website/commercial/server/decision_engine/models/
cp new-model-zigzag/data/regime_selector_K4.json \
   ../my-agents-and-website/commercial/server/decision_engine/data/
cd ../my-agents-and-website/commercial
git tag -a "pre-oracle-$(date +%Y%m%d)" -m "Pre-Oracle-XAU retrain"
git add server/decision_engine/models/oracle_xau_validated.pkl \
        server/decision_engine/data/regime_selector_K4.json
git commit -m "Retrain Oracle XAU on $(date +%Y-%m-%d) data"
git push origin main
```

Render auto-deploys ~60s after push.

## Critical files (must keep)

```
data/swing_v5_xauusd.csv       ← raw M5 + tick_volume + label
data/labeled_v4.csv            ← ATR-based 0/1/2 labels per bar
products/_shared/quantum_flow.py
products/_shared/regime_selector_xau.json   (canonical, with v9.4 relabel
                                              + 2,051 block_boundary_times)
models/oracle_xau_validated.pkl              (current production)
```

## After retrain: regenerate block boundaries

The selector JSON ships with `block_boundary_times` — every 288-bar
block START timestamp from the training CSV. Live `/decide` reads
this list to align inference-time blocks to training-time blocks
exactly (no UTC-midnight approximation).

```bash
python products/_shared/build_block_boundaries.py    # rebuilds + embeds
cp data/regime_selector_K4.json \
   ../my-agents-and-website/commercial/server/decision_engine/data/
```

The live server extrapolates beyond the last embedded boundary by
counting 288 bars in its Dukascopy cache, so the regime keeps
advancing daily even between retrains. But if retrain skips more
than ~30 days (the live cache window), the cache won't bridge to
the embedded list and classify falls back to UTC-midnight anchor.

## Rollback

```bash
cd ../my-agents-and-website/commercial
git reset --hard pre-v9.4   # last known-good before v9.4 regime change
git push --force-with-lease origin main
```
