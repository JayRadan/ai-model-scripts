# Hermes DJI — M1 Dow Jones (US30)

> **🚀 STATUS 2026-06-30 — REPLACED with the `edge_pullback` engine (commit `064aeee`).**
> The old combined-Q / `q_thr=3.0` strategy (which lost −$708 live over May–Jun) is **gone**.
> New strategy = **pullback + XGB-expected-R** (see below). Server engine: `decision_engine/edge_pullback.py`,
> routed by bundle `version="edge_pullback_v1_hermes_dji"`. **1 slot, ~11 trades/day.**
> Rollback: `models/hermes_dji_validated.pkl.bak_pre_edge_pullback_2026-06-30`.
>
> **Deployed config:** entry = `committed_dir != 0` AND `|close−tfk_line|/ATR ≤ 1.0` (pullback to TFK line),
> direction = `committed_dir`; XGBRegressor predicts gross R, take if `pred_R ≥ 0.559`; exit = EA-side
> **SL 6×ATR + trail 2×ATR**, max_hold 300. Bundle trained through 2026-06-30 on 8y M1 (~700k candidates).
>
> **8-year walk-forward (train-only thresholds, causal HTF, 1pt spread):** +4811R, **12/12 windows positive**,
> +0.30R/trade, maxDD 149R, ~10/day, daily Sharpe ~3.0. Shorts profitable; made +R in the 2022 bear (not beta).
>
> **⚠️ HONEST CAVEATS:** thin edge (**+0.27R/trade net@1pt**), **spread-sensitive** (breakeven ≈ 3pt — needs a
> tight US30 broker); **never forward-tested** before deploy. ~57% of *days* are positive — **6–9-day losing
> streaks are NORMAL** (a 2-week loss is expected variance, not a failure). Monitor; revert via the `.bak` if it bleeds.

---
<details><summary>Historical (pre-2026-06-30, the retired q_thr strategy)</summary>

5th product; M1 mirror of Hermes XAU/BTC adapted for the Dow Jones index. Slug: `hermes_dji`.
Deployed 2026-05-27.

## 🆕 Current deployed config (2026-06-17)

| Param | Value | Notes |
|---|---|---|
| `near_thr` | 0.50 | pullback band |
| `counter_thr` | 1.5 | counter setup threshold |
| `q_thr` | **2.5** | lowered 3.0 → 2.5 on 2026-06-17 ("more activity" pivot — was firing 0–2 trades/day) |
| `time_block_utc` | (0, 0) | disabled |
| `trend_slope_block` | 0.0 | disabled |
| `sl_hard_atr` | 6.0 | |
| `trail_atr` | 2.0 | **static** (no adaptive buckets — DJI doesn't use them) |
| `use_orderflow` | **False** | orderflow tested neutral on Dow; M1-only ships |
| `max_concurrent` | 4 | |

## 🧠 2026-06-09 — No filter applied

Backtest sweep tested time-filter and trend-gate variants — **every variant
hurt Hermes DJI**. Baseline PF is already 10.30 (the highest of any product),
so further filtering just cuts profitable trades without removing losers.

Both config fields exist (`time_block_utc = (0, 0)`, `trend_slope_block = 0.0`)
but are disabled — left baseline as-is.

If a future regime shift hurts DJI, the config fields are pre-wired for an
instant tuning push without code changes.


## Architecture

- **Timeframe:** M1
- **Regime:** TFK indicator `committed_dir` (see `tfk.py`)
- **Entry (combined-Q):** pullback `|close − tfk_line|/atr ≤ 0.50` **OR** counter `≥ 1.5×ATR` wrong-side
- **ML:** single XGBRegressor Q on 29 standard features. **No orderflow** — orderflow was tested
  on Dow and came out neutral/marginal, so the simpler M1-only model ships (also avoids
  server-side tick fetching for Dow).
- **Exits:** hard SL 6×ATR + trail 2×ATR give-back + BE-on-new-entry (0.5R) + max_hold 300 bars.
  Exit params are owned by the live config at
  `commercial/server/decision_engine/configs/hermes_dji.py`.
- **Multi-pos:** 4 concurrent slots, switch delta 0.5, cooldown 5 bars.
- **Gate:** Q ≥ 3.0.

> ⚠️ **PF under re-validation.** Earlier backtests (portfolio-sim PF 8.45 / raw 9.00) were
> inflated by an HTF (m5/m15/h1) look-ahead bug. Those numbers must **not** be quoted externally.
> Honest causal numbers are being re-measured — now on correct canonical Dukascopy bars after the
> 2026-05-29 `DJIUSD → E_D&J-Ind` symbol fix in `dukascopy_source.py`.

## Files

| File | Purpose |
|---|---|
| `tfk.py` | TFK indicator (Pine v6 port) |
| `train_bundle_combined.py` | combined-Q trainer (NEAR OR counter), M1-only — **the shipped one** |
| `train_bundle_combined_with_flow.py` | variant with orderflow (tested, not shipped) |
| `train_bundle_combined_join_flow.py` | variant joining flow features (tested, not shipped) |
| `run_portfolio_sim.py` | multi-pos + BE + switch portfolio simulation |
| `scripts/01_download_m1_dukascopy.py` | M1 OHLCV download |
| `scripts/02_download_ticks_dukascopy.py` | tick download (for the flow experiments) |
| `scripts/03_aggregate_orderflow_from_ticks.py` | tick → M1 orderflow features |
| `scripts/09_build_website_backtest_json.py` | backtest JSON for the website |
| `hermes_dji_validated.pkl` | shipped bundle (local copy; live copy in commercial repo) |
| `hermes_dji_native_std*.pkl`, `*_with_flow.pkl` | research/variant bundles |

## Data dependencies (all in repo `data/`)

- `data/m1_dji_full.parquet` — M1 OHLCV
- `data/m1_dji_orderflow.parquet` — aggregated orderflow (only for flow variants)
- `data/ticks/dji/YYYY-MM-DD.parquet` — raw tick files

## Shared code

Feature engineering (`add_standard_features`) comes from `products/_shared/m1_with_orderflow.py`
(shared with Hermes XAU/BTC). TFK is duplicated per product in `tfk.py`.

</details>
