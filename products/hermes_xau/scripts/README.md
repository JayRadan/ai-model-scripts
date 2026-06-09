# hermes_xau — pipeline scripts

End-to-end scripts for the **hermes_xau** product, in execution order.

| Step | Script | Purpose | Output |
|---|---|---|---|
| 1 | `01_download_m1_dukascopy.py` | Fetch M1 bars for XAUUSD (Spot Gold) from Dukascopy | `data/m1_xau_full.parquet` |
| 2 | `02_aggregate_orderflow_from_ticks.py` | Aggregate tick stream into per-bar order-flow features (imbalance, VPIN proxy, signed flow, tick intensity, etc.) | `data/m1_xau_orderflow.parquet` |
| 3 | `03_train_q_production.py` | Train the XGBRegressor Q-model on MFE≥2R candidates + write the .pkl bundle | `commercial/server/decision_engine/models/hermes_xau_validated.pkl` |
| 10 | `10_sim_today.py` | Simulate today's trades under the **deployed** config (Q≥4.0, time_block_utc=(20, 1) + dist_cap=3.0). Outputs trade-by-trade table + equity PNG | `scripts/_out/sim_today.png` |

## Deployed config (as of 2026-06-09)

| Parameter | Value | Source of truth |
|---|---|---|
| Q threshold | **4.0** | `commercial/server/decision_engine/configs/hermes_xau.py` |
| Filter | **time_block_utc=(20, 1) + dist_cap=3.0** | Same config |
| SL / TRAIL | 6×ATR / 2×ATR | Same config |
| BE trigger | +0.5R | Same config |
| Multi-slot | 4 slots, switch 0.5, cooldown 5 | Same config |

## How to reproduce the production bundle

```bash
cd ~/Desktop/new-model-zigzag
# Step 1 — refresh M1 bars from Dukascopy (~1 min)
python3 products/hermes_xau/scripts/01_download_m1_dukascopy.py

# Step 2 — aggregate tick stream into order-flow features (~5 min)
python3 products/hermes_xau/scripts/02_aggregate_orderflow_from_ticks.py

# Step 3 — train and write the production bundle (~1-3 min)
python3 products/hermes_xau/scripts/03_train_q_production.py
```

After step 3, the new `hermes_xau_validated.pkl` is in the server's models/ directory.
**Commit it via the commercial repo** to deploy.

## How to verify a config change without redeploying

```bash
# Edit the config in commercial/server/decision_engine/configs/hermes_xau.py
# Then run the sim with the new constants:
python3 products/hermes_xau/scripts/10_sim_today.py
```

`10_sim_today.py` mirrors the deployed config — keep its top-of-file constants
in sync with `commercial/server/decision_engine/configs/hermes_xau.py` whenever
you push a config change.

## Cross-product helpers

| Path | Purpose |
|---|---|
| `products/_shared/scripts/20_filter_variants_sweep.py` | Sweep time-block + trend-gate variants across all 6 products on 30 days unseen data |
| `products/_shared/scripts/21_filter_ideas_backtest.py` | Idea-screening: baseline vs each filter |
| `products/_shared/scripts/22_last_week_sim.py` | Quick "how did last week go?" sim under current deployed config |
