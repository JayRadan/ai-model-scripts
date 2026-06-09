# Shared backtest / sweep scripts

Cross-product scripts that run against multiple bundles at once.

## 2026-06-09 filter tuning sweep

| Script | Purpose | Output |
|---|---|---|
| `20_filter_variants_sweep.py` | Sweep time-blocking UTC windows AND trend-gate slope thresholds across all 6 products on 30 days unseen Dukascopy bars. Best variant per product. | Stdout — per-product per-variant table |
| `21_filter_ideas_backtest.py` | Compare 5 filter ideas (time, trend, vol-brake, combinations) against baseline. The "is this idea worth trying?" prior to the variant sweep. | Stdout — baseline vs filter table |
| `22_last_week_sim.py` | Simulate the last calendar week (Mon-Sun) for all 6 products under their CURRENT deployed config. Quick "how did we do last week?" check. | Stdout — per-product summary |

### How to run
```bash
cd ~/Desktop/new-model-zigzag
python3 products/_shared/scripts/20_filter_variants_sweep.py
python3 products/_shared/scripts/21_filter_ideas_backtest.py
python3 products/_shared/scripts/22_last_week_sim.py
```

All three fetch fresh data via `dukascopy_python` (XAU/USD, BTC/USD, US30).
Bundles load from `commercial/server/decision_engine/models/`.

### 2026-06-09 sweep verdict

| Product | Best filter | Δ$ | Δ DD |
|---|---|---:|---:|
| hermes_xau | time_block=(20, 1) | +3.5% | 0 |
| hermes_btc | trend_slope_block=1.5 | +10% | −22% |
| hermes_dji | — (baseline best) | 0 | 0 |
| atlas_xau | time_block=(18, 2) | **+98%** | **−58%** |
| atlas_btc | — (baseline best) | 0 | 0 |
| atlas_dji | time_block=(19, 2) | +44% | −44% |

Deployed in commit `c497f45` (commercial repo).

## Other scripts (pre-existing)

| Script | Purpose |
|---|---|
| `build_regime_selector.py` | Train the regime k-means selector used by Oracle |
| `port_and_test.py` | Cross-instrument port tester |
| `visualize_regime.py` | Plot regime clusters for debugging |
