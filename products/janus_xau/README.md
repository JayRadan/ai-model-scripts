# Janus XAU — pivot-score turning-point detector

Different architecture from Oracle/Midas: instead of a fixed catalog of
28 rules, Janus uses an XGBoost **pivot-score model** trained to detect
the specific bars that mark trend turning points (zigzag pivots). When
the score crosses a threshold, a synthetic "RP_score" rule fires and
goes through the same per-cluster confirm + meta gate as Oracle.

Niche product: very selective, sparse fires, large bars-per-trade variance.

## At a glance

| | Value |
|---|---|
| Asset | XAUUSD M5 |
| Architecture | v7.4 pivot-score → per-cluster confirm → meta gate |
| Regime detector | K=5 K-means + v9.4 highvol_only relabel — **v83c: 4h-step** |
| Trained on | Dukascopy 2018-01-02 → 2024-12-12 |
| Holdout window | 2024-12-12 → 2026-05-01 |
| **Holdout PF** | **2.96** |
| **Holdout WR** | **60.1%** |
| **Holdout DD** | **-334 R** |
| **Holdout n** | **6,269** |
| Pivot score threshold | `0.3` (`SCORE_THR` constant) |
| Meta threshold | **0.775** |
| Deployed pkl | `commercial/.../models/janus_xau_validated.pkl` |
| Deploy commit | `e4c6c94` (v83c — 4h regime + kill-switch) |

## Architecture

```
M5 bar  ──►  K=5 regime selector + v9.4 relabel ──►  cluster id
                                                          │
                                                          ▼
                                          pivot-score XGB on 32 features
                                          (h1/h4 swing distances, RSI,
                                           momentum, round-number dist,
                                           close-streak, etc.)
                                                p_pivot >= 0.3 ?
                                                          │ pass
                                                          ▼
                                              direction model XGB
                                              (long if p_long >= 0.5
                                               else short)
                                                          │
                                                          ▼
                                          per-cluster confirm head (XGB,
                                          18 v72l features)
                                                p_conf >= per-rule thr ?
                                                          │ pass
                                                          ▼
                                          meta gate XGB (20 features)
                                                p_win >= 0.775 ?
                                                          │ pass
                                                          ▼
                                                     OPEN TRADE
```

Per-cluster contribution (after meta filter):

| Cluster | n | WR | PF | Total R |
|---|---|---|---|---|
| C0 Uptrend | 2,324 | 60.0% | **3.33** | +9,203 |
| C3 Downtrend | 1,685 | 65.3% | **3.77** | +6,589 |
| C1 MeanRevert | 659 | 67.8% | **3.64** | +2,151 |
| C2 TrendRange | 1,048 | 50.7% | 1.70 | +1,454 |
| C4 HighVol | 553 | 52.8% | 1.53 | +494 |

## Critical deploy fix (v9.4)

The pickle script's max-PF threshold sweep selected **0.85**, but the
canonical validated threshold (`meta_threshold_v74.txt`) is **0.775**.
v9.4 commit `132e443` rewrote the deployed pkl's `meta_threshold` field
from 0.85 → 0.775 to match the validation report. Pickle script also
patched to read the txt file first.

## Pipeline (retrain from scratch)

### 1. Reuse Oracle XAU regime selector

```bash
python products/oracle_xau/01_build_selector.py
# Janus reuses data/regime_selector_K4.json
```

### 2. Regenerate cluster_per_bar mapping (with v9.4 relabel)

```bash
python products/janus_xau/01_regen_cluster_per_bar.py
# → writes experiments/v73_pivot_oracle/data/cluster_per_bar_v73.csv
# (the pivot pipeline's own per-bar cluster lookup)
```

### 3. Compute v74 pivot-context features (55 features per bar)

```bash
python products/janus_xau/02_compute_features_v74.py
# → writes experiments/v74_pivot_score/data/features_v74.csv (~570 MB)
```

### 4. Label every bar (is_pivot_25, best_R, best_dir)

```bash
python products/janus_xau/03_label_every_bar.py
# → writes experiments/v74_pivot_score/data/labels_v74.csv
```

### 5. Train pivot-score + direction models

```bash
python products/janus_xau/04_train_pivot_score.py
# → writes pivot_score_mdl.pkl + pivot_dir_mdl.pkl
```

### 6. Build setups using pivot scores (per cluster)

```bash
python products/janus_xau/05_build_setups_v74.py
# → writes setups_{0..4}_v74.csv
```

### 7. Train + validate (per-cluster confirm + meta + threshold sweep)

```bash
python products/janus_xau/06_validate_v74.py
# → writes:
#   - experiments/v74_pivot_score/reports/meta_threshold_v74.txt   (0.775)
#   - experiments/v74_pivot_score/reports/v74_trades_holdout.csv
#   - prints PF/WR/DD per cluster
```

### 8. Pickle for deployment

```bash
python products/janus_xau/07_pickle_janus.py
# → writes new-model-zigzag/models/janus_xau_validated.pkl
# Reads meta_threshold_v74.txt (0.775) — matches what was validated.
```

### 9. Copy + commit + push (same pattern as Oracle XAU step 8)

## Why Janus has higher DD than Oracle/Midas

Janus has ~3-4x the trade count (6,269 vs 1,903) and individual trades have
higher MFE/MAE swings because pivot-detected entries often cluster around
volatile turning points. Per-trade expectancy is excellent (+2.83 R), but
the absolute drawdown in R units scales with trade volume.

To compare DD apples-to-apples, normalize: Janus DD/n = -334/6269 = -0.053
R/trade vs Oracle's -56/1903 = -0.029 R/trade. Roughly 1.8x deeper per
trade — still acceptable given PF 2.96.

## Critical files

```
data/swing_v5_xauusd.csv
data/regime_selector_K4.json                                    (with v9.4 relabel
                                                                  + block_boundary_times)
products/janus_xau/data/cluster_per_bar_v73.csv
products/janus_xau/data/{features,labels}_v74.csv
products/janus_xau/models/pivot_*.json
products/janus_xau/reports/meta_threshold_v74.txt               (0.775)
models/janus_xau_validated.pkl
```

## Watch out: Janus pickle threshold drift

`07_pickle_janus.py` historically picked its OWN max-PF threshold
(landed at 0.85) which differed from the validated 0.775 saved in
`meta_threshold_v74.txt`. v9.4 patched the pickle script to read the
saved threshold first; the deployed pkl now matches the validation
report. Don't revert this — re-introducing the max-PF policy will
silently make Janus more selective than the holdout PF reflects.
