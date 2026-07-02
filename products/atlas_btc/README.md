# Atlas BTC — STRICT 2-bar candle reversal (Kalman + Hermes TFK), 8-YEAR deep retrain (Bitcoin)

> **🆕 2026-07-02 — time-boxed-patience trail DEPLOYED (edge_predictor commit `42b6334`).**
> Server-side trail tightens 2×ATR → **0.75×ATR after 30 bars held** (`tight_after_bars`/`tight_trail_atr` in
> `configs/atlas_btc.py`, read by `decide_atlas.decide_exit` via getattr — defaults off, other atlas products
> unchanged; EA unchanged). Model/labels/q_thr untouched. 8y WF overlay on the deploy-validation harness
> (`experiments/atlas_xau_entry_exit_lab/run_lab_btc.py`): dev 2022-24 sumR +15,302→+19,464 medPF 1.43→1.70
> (12/12), untouched holdout 2025+ **sumR +3,976→+6,280, medPF 1.26→1.68, WR 72→79%, 6/6 windows**. Third
> independent confirmation of the same exit mechanism (after atlas_xau + hermes_dji, both edge_pullback).
> Rollback: `git revert 42b6334`.

> **✅ DEPLOYED (verified 2026-06-30):** Atlas strict-candle 2-bar reversal (TFK regime + M1 Kalman
> confluence + strong prev-bar body), **8-year deep retrain** (2026-06-26). **M1 BTCUSD, 1 slot,
> SL 6×ATR / trail 2×ATR / BE@0.5R, q≥3.0.** Live bundle has **no version** (standard atlas path).
> Redeployed after the original short-window train overfit. Catalog: [`../README.md`](../README.md).

> **Version:** **strict-candle 8y-deep** (`atlas_btc_8y_2026-06-26`) — **DEPLOYED, commit `cd911ca`**
> **Architecture:** identical engine to Atlas DOW/XAU — M1 Kalman regime + M1 Hermes TFK, **STRICT 2-bar candle-reversal** confirmation gated by both-lines + `kf_age≥3`; XGBRegressor Q trained only on MFE≥2R candidates. **54 features** (Kalman state + TFK + 18 standard + 14 order-flow).
> **Bundle:** `atlas_btc_validated.pkl` (8-year retrain, `q_thr=3.0`, SL 6×ATR / TRAIL 2×ATR / MAXH 300, BE @ +0.5R, `max_concurrent=1`, spread 0.30 R-units)
> **Trained on:** 8 years M1 BTC tick→orderflow **2018 → 2026** (4.08M bars; ticks backfilled to 2018 to match DOW's depth)
> **Holdout (unseen, post-2025-09):** PF **1.32** @ Q≥3 (WR 71.4%, +1,882R) · PF 1.43 @ Q≥4
> **Walk-forward:** **18/18 quarterly windows PF>1.0** (2022→2026, median 1.38, WR 68–73%, incl. 2022 crypto crash) — more robust than DOW's 10/10
> **Backup:** `atlas_btc_validated.pkl.bak_pre_8y_2026-06-26` · **Rollback:** `git revert cd911ca` (commercial repo)
> **⚠️ Live caveat:** offline edge is strong but live BTC *was* losing pre-retrain — likely an execution gap (real spread/slippage > 0.30-ATR modeled). Deep retrain fixes *overfitting*, not necessarily *execution*. Monitor live.

## 🆕 2026-06-26 — 8-YEAR deep retrain (DEPLOYED, commit `cd911ca`)

**Why:** live BTC was losing while Atlas DOW (identical recipe) was profitable. Root cause:
the deployed BTC bundle was trained on only ~9 months (2024-12 → 2025-09) — a thin recent
window — while DOW had 8 years. Backfilled XAU+BTC Dukascopy ticks to 2018, rebuilt the full
8-year orderflow parquet, and retrained with the **exact Atlas DOW recipe**.

**Result:** the BTC edge **survives deep cross-regime training** — holdout PF 1.32 @ Q≥3,
and **walk-forward 18/18 quarterly windows PF>1.0** (2022→2026, incl. the 2022 crash). By
contrast **Atlas XAU's edge VANISHES on 8-year data (PF 0.95) — confirmed dead, NOT redeployed.**

**Entry (STRICT):** BUY = TFK GREEN + Kalman RED + prior bar strong-bear (body ≥ 0.8×ATR)
closing below BOTH lines + current bar green + kf_age ≥ 3. SELL = mirror.
**Exit:** hard SL 6×ATR, trail 2×ATR, max-hold 300 M1 bars, BE @ +0.5R on new signal. `q_thr=3.0`.

**Reproduce** (`experiments/atlas_retrain_like_dow/`): `backfill_ticks.py` (ticks→2018) →
`aggregate_8y.py` (→ `data/m1_btc_orderflow_8y.parquet`) → `build_btc_bundle.py` (writes the
deployed bundle) · WF: `wf_btc.py`. Recipe mirrors `products/atlas_dji/scripts/03_train_q_production.py`.

---

## ⛔ SUPERSEDED 2026-06-23 — Band-pullback v1 (deployed e442bf2, **REVERTED 2026-06-25** d187ecc)

> The band-pullback engine below was deployed 2026-06-23 but **reverted on 2026-06-25**
> ("restore all 6 products to first-deploy") back to the strict-candle architecture, which
> was then deep-retrained above. **It is NOT live** — the WF 2.38 / PF 4.89 figures here
> are not the deployed model. Kept for history only.

The ushape_m15 product (everything documented below this section) was **fully
replaced** by the Kalman-band-pullback engine — same architecture as Hermes XAU.
Entry is a **rejection of the M1 Kalman ±2σ envelope, confirmed by a candle color
flip, gated to with-trend by M30 TFK (PRO)**.

**Entry rules** (deployed bundle `rules`):
| Side | Rule |
|---|---|
| SHORT | M30 TFK = −1 **AND** bar i−1: High ≥ `kf_upper(k=2.0)` **AND** green **AND** bar i: red → SHORT at open(i+1) |
| LONG  | M30 TFK = +1 **AND** bar i−1: Low ≤ `kf_lower(k=2.0)` **AND** red **AND** bar i: green → LONG at open(i+1) |

**Exit:** band-exit (opposite envelope touch + color flip), hard SL **3.0R**, max-hold **200 M1 bars** (~3h).

**Cascade (3 XGB heads, keep iff all pass):** `MFE ≥ mfe_t`, `Q ≥ q_t`, `BL ≤ bl_t`.
Deployed BTC thresholds (both slots): **`mfe_t=0.40, q_t=−5.0 (off), bl_t=1.0 (off)`** — BTC keeps on the MFE head alone (highest trade volume of the three band products: ~28 trades/day in live sim).

**34 features** — same `feat_cols` as Hermes XAU (Kalman state + candle geometry + TFK natives + classic context). **No order-flow.**

**Trained on** Dukascopy M1 BTC 2024-10-31 → 2026-05-02 (BTC has ~18mo of usable Dukascopy data). Research in `experiments/hermes_band_pullback/` (`train_production_btc_dji.py`, `sweep_btc_dji.py`, `wf_btc_dji.py`).

**Today's live sim (2026-06-24):** 28 trades · WR 82.1% · **PF 4.89** · +22.09R · DD 3.0R · $+33.13.

---

<details>
<summary>📦 ARCHIVED — pre-2026-06-23 ushape_m15 / strict-candle architecture (no longer deployed)</summary>

## 2026-06-17 — Full architecture replacement: ushape_m15

Replaces the prior STRICT-2-bar reversal candle with M15-macro + M1 U-shape edge
(same change as Atlas XAU deploy, commit 7bf5a37).

| | Before (strict_candle) | Now (ushape_m15) |
|---|---|---|
| Macro regime | M1 TFK committed_dir | **M15 Kalman kf_dir** |
| Entry trigger | strong bear/bull prev bar + close past lines | **M1 Kalman U-shape edge** |
| 8mo backtest PF | 1.31 | **1.58** (+21%) |
| 8mo backtest DD | ~95R | **57R** (−40%) |
| Trade rate | ~7/day | ~6/day @ Q≥1.5 |

### 3-day live sim (06-14 → 06-17) — head-to-head

| Date | OLD strict_candle $ | NEW ushape_m15 $ |
|---|---:|---:|
| 06-14 | +$6,890 | +$2,190 |
| 06-15 | +$8,365 | +$3,261 |
| 06-16 | −$8,423 | −$15,084 |
| 06-17 | −$9,543 | **+$11,677** |
| **TOTAL** | **−$2,710** | **+$2,044** |

NEW wins by $4,754 over 3 days. ($ figures inflated by R_to_USD calibration —
actual likely 50-100× smaller; PF/sumR comparisons are valid.)

### 14-day live sim (06-04 → 06-17)

75 trades · 8 green days vs 5 red days · +$3,761 @ 0.10 lot (scale inflated)

### Files changed in this deploy

| File | Change |
|---|---|
| `commercial/server/decision_engine/configs/atlas_btc.py` | `entry_mode="ushape_m15"`, `macro_tf_min=15`, `q_thr=1.5` |
| `commercial/server/decision_engine/models/atlas_btc_validated.pkl` | New 40-feature bundle (backup: `.bak_pre_m15_ushape_2026-06-17`) |
| `products/atlas_btc/scripts/03_train_q_production.py` | Replaced with M15 U-shape training recipe |
| `products/atlas_btc/scripts/10_sim_today.py` | Rewritten for new architecture |

Server-side `decide_atlas.py` and `atlas_features.py` already support `ushape_m15`
from the XAU deploy (commit 7bf5a37) — only config flag flip needed for BTC.

EA changes: **none**. ATB- magic numbers, slot management unchanged.

---

## Pre-2026-06-17 strict_candle config (archived)

> **Prior version:** STRICT 2-bar reversal candle + Kalman/TFK confluence
> **Prior bundle:** `atlas_btc_validated.pkl.bak_pre_m15_ushape_2026-06-17` (54 features)

## Strict_candle deployed config (archived for reference)

| Param | Value | Notes |
|---|---|---|
| `strong_body_atr` | 0.8 | |
| `kf_age_min` | **3** | raised 1 → 3 on 2026-06-12 after −$107 chop day (kage=1 let too many fast Kalman flips through) |
| `require_both_lines` | True | |
| `q_thr` | **1.5** | lowered 2.0 → 1.5 on 2026-06-17 (more activity pivot) |
| `time_block_utc` | (0, 0) | disabled — 24/7 market |
| `trend_slope_block` | 0.0 | disabled |
| `sl_hard_atr` | 6.0 | |
| `trail_atr` | **1.0** | tightened 1.5 → 1.0 on 2026-06-16 (DD −51% on 8mo holdout, config-only no retrain) |
| `use_orderflow` | True | |
| `max_concurrent` | 4 | |

---



## 🧠 2026-06-09 — kf_age_min 3 → 1 + q_thr 3.0 → 2.0

Diagnosis on today's XAU sharp fall (same Atlas architecture as BTC) showed
the `kage ≥ 3` requirement was filtering out the 1-2 bar Kalman flips that
happen during sharp impulse pullbacks. BTC is the most-affected product
because it's 24/7 with frequent sharp moves.

8-month holdout sweep (per-product, multi-pos sim):

| Variant | Trades | WR | PF | DD | $@0.10 |
|---|---:|---:|---:|---:|---:|
| kage≥3 Q≥3.0 (was deployed) | 2,522 | 71.1% | 1.38 | 128 | $34,386 |
| **kage≥1 Q≥2.0 (deployed)** | **4,308** | **68.7%** | 1.28 | 142 | **+$45,172** |

**+31% $ at +11% DD** — biggest absolute lift of the deployment. WR dips
2.4pp but trade volume rises 71%, so absolute winners grow meaningfully.


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

</details>
