# Atlas BTC — M1 Kalman + TFK Dual-Indicator STRICT Reversal (Bitcoin)

> **Version:** v1 (deployed 2026-06-04)
> **Architecture:** Same as Atlas XAU but tuned for BTCUSD microstructure
> **Bundle:** `atlas_btc_validated.pkl` (Q-regressor trained on MFE≥2R movers)
> **Entry gate:** STRICT pattern + Q ≥ **3.0**
> **Exit:** SL 6×ATR · TRAIL 2×ATR · MAX_HOLD 300 · BE@+0.5R · 4 slots


## 🧠 2026-06-09 — No filter applied

Backtest sweep tested time-filter and trend-gate variants — **none beat
baseline**. Atlas BTC's STRICT 2-bar reversal pattern already filters trend
context internally; additional filters cut volume without removing losers.

Both config fields exist (`time_block_utc = (0, 0)`, `trend_slope_block = 0.0`)
but are disabled. Baseline ($1,506 @0.10 on 30d unseen) is the current best.


## Holdout (post-Sep 2025, 8 months unseen)

| Q | trades | WR | PF | sumR |
|---:|---:|---:|---:|---:|
| 1.0 | 4,675 | 72.3% | 1.32 | +2,401 |
| 2.5 | 3,710 | 72.5% | 1.36 | +2,112 |
| **3.0** | **2,779** | **72.7%** | **1.37** | **+1,660** |
| 4.0 | 960 | 73.8% | 1.43 | +626 |

Q=3.0 chosen for best PF at usable trade volume.

## Entry rule (STRICT)

Identical to Atlas XAU — see `products/atlas_xau/README.md` for details.

## Training data

- `data/m1_btc_orderflow.parquet` (741k bars, Nov 2024 → May 2026)
- 7,855 MFE≥2R candidates (72.8% of all candidates have a ≥2R favourable move)

## Scripts

| Script | Purpose |
|---|---|
| `scripts/10_sim_today.py` | Pull fresh Dukascopy bars, simulate today's trades with deployed config |
