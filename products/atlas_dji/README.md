# Atlas DJI — M1 Kalman + TFK STRICT-candle reversal (Dow Jones)

> **🆕 2026-07-02 — time-boxed-patience trail DEPLOYED (edge_predictor commit `51b3f6b`).**
> Server-side trail tightens 2×ATR → **0.75×ATR after 30 bars held** (`tight_after_bars`/`tight_trail_atr`
> in `configs/atlas_dji.py`; engine read shipped with atlas_btc's `42b6334`; EA unchanged). Model/q_thr
> untouched. 8y WF overlay (`experiments/atlas_xau_entry_exit_lab/run_lab_dji_atlas.py`): dev 2022-24
> sumR +2,445→+4,542 medPF 1.25→1.55 (12/12), untouched holdout 2025+ **+1,243→+2,089R, medPF 1.27→1.54,
> WR 69→78%, 6/6 windows**. Fourth independent confirmation across 2 engines / 4 products.
> Rollback: `git revert 51b3f6b`.

> **✅ DEPLOYED (verified 2026-06-30):** Atlas strict-candle 2-bar reversal, **q≥3.0, 1 slot,
> SL 6×ATR / trail 2×ATR / BE@0.5R**, M1 DJIUSD. **The one historically robust edge** (walk-forward
> 10/10 windows positive). Live bundle has **no version** (standard atlas path). Catalog:
> [`../README.md`](../README.md). ⚠️ Missing a `sim_today` monitor script (only 4 scripts present).

> **⚠️ 2026-06-24 ROLLED BACK to strict-candle (commit `6a393bb`).** The band-pullback
> below was breakeven/look-ahead-inflated; the honest 8y retrain + **10/10-window
> walk-forward** show the STRICT-candle Atlas DJI is the real edge: **PF 1.19, WR 70.7%,
> +3,175R/2.4y, DD 302R, ~9–13 t/d** (config q_thr=3.0 ≈ 9/day; Q≥1.0 ≈ 13/day).
> Restored `atlas_dji_validated.pkl.bak_pre_band_pullback_2026-06-23`. Recipe + WF:
> `experiments/full_8y_retrain/` (retrain_all_8y.py, atlas_dji_wf.py). See
> [[atlas_dji_real_edge_2026-06-24]] in memory.

> **Version (no longer deployed):** band-pullback v1 (`hermes_band_pullback_v1_dji_2026-06-23`)
> **Architecture:** M1 Kalman ±2σ band rejection + M30 TFK with-trend (PRO) gate + 3-model cascade (MFE / Q / big-loss), single-position cooldown=5. Identical engine to Hermes XAU & Atlas BTC.
> **Bundle:** `atlas_dji_validated.pkl` (34 features, `band_k=2.0`, `sl_R=3.0`, `maxh_m1=200`, `tfk_tf=30min`)
> **Walk-forward (DJI, 18 folds):** 18/18 +ve PF **2.23** / $+30k (see [memory: Atlas BTC+DJI band-pullback](../../))
> **Backup of pre-band bundle:** `atlas_dji_validated.pkl.bak_pre_band_pullback_2026-06-23`
> **Rollback:** in the commercial repo `git revert e442bf2`, then `mv atlas_dji_validated.pkl.bak_pre_band_pullback_2026-06-23 atlas_dji_validated.pkl`, push.

## 🆕 2026-06-23 — Band-pullback replacement (DEPLOYED, commit e442bf2)

The STRICT-reversal product (everything documented below this section) was **fully
replaced** by the Kalman-band-pullback engine — same architecture as Hermes XAU /
Atlas BTC. Entry is a **rejection of the M1 Kalman ±2σ envelope, confirmed by a
candle color flip, gated to with-trend by M30 TFK (PRO)**.

**Entry rules** (deployed bundle `rules`):
| Side | Rule |
|---|---|
| SHORT | M30 TFK = −1 **AND** bar i−1: High ≥ `kf_upper(k=2.0)` **AND** green **AND** bar i: red → SHORT at open(i+1) |
| LONG  | M30 TFK = +1 **AND** bar i−1: Low ≤ `kf_lower(k=2.0)` **AND** red **AND** bar i: green → LONG at open(i+1) |

**Exit:** band-exit (opposite envelope touch + color flip), hard SL **3.0R**, max-hold **200 M1 bars** (~3h).

**Cascade (3 XGB heads, keep iff all pass):** `MFE ≥ mfe_t`, `Q ≥ q_t`, `BL ≤ bl_t`.
Deployed DJI thresholds (both slots): **`mfe_t=0.40, q_t=−5.0 (off), bl_t=1.0 (off)`** — MFE head only.

**34 features** — same `feat_cols` as Hermes XAU / Atlas BTC. **No order-flow** (DJI never used ticks — fair test was neutral).

**Trained on** Dukascopy M1 DJI 2018-01-02 → 2026-05-27 (full 8y). Research in `experiments/hermes_band_pullback/` (`train_production_btc_dji.py`, `sweep_btc_dji.py`, `wf_btc_dji.py`).

**Today's live sim (2026-06-24):** 31 trades · WR 87.1% · **PF 6.19** · +37.83R · DD 4.1R · $+56.74 (best of the three band products today).

---

<details>
<summary>📦 ARCHIVED — pre-2026-06-23 STRICT-reversal architecture (no longer deployed)</summary>

## Current deployed config (2026-06-17)

| Param | Value | Notes |
|---|---|---|
| `strong_body_atr` | 0.8 | |
| `kf_age_min` | **1** | reverted 5 → 1 on 2026-06-16 — 5 live days showed kage=5 was net-negative ($141 vs $13 for kage=1) despite favorable 8mo backtest |
| `require_both_lines` | True | |
| `q_thr` | 3.0 | unchanged |
| `time_block_utc` | (19, 2) | block 19:00–02:00 UTC (late US + Asia overnight) |
| `trend_slope_block` | 0.0 | disabled |
| `sl_hard_atr` | 6.0 | |
| `trail_atr` | 2.0 | static (no adaptive buckets on DJI) |
| `use_orderflow` | **False** | disabled 2026-06-16 — Dukascopy DJI tick fetch hangs 60–90s, caused err 5203/HTTP 1003/502. Q model treats missing flow features as 0. |
| `max_concurrent` | 4 | |

### 2026-06-16 — Orderflow disabled
Dukascopy's DJI tick stream is unreliable and reliably hangs the decide
pipeline. After the 60–90 s waits caused err 5203 cascades on every EA call,
disabled `use_orderflow`. Matches Hermes DJI which never used orderflow.
The trained Q model still expects 14 flow columns — they're filled with 0
(neutral signal) at inference, and the discriminator handles this gracefully.

### 2026-06-16 — kage 5 → 1 rollback
The 06-12 tightening to `kage_min=5` showed +PF / −DD on the 8mo backtest but
5 live trading days flipped the result: kage=1 made $141 vs kage=5 made $13.
Recent regime didn't match the holdout average. Rolled back to kage=1; Atlas
XAU (kage=4) and Atlas BTC (kage=3) tunes are unchanged.

---



## 🧠 2026-06-09 — kf_age_min 3 → 1 (q_thr unchanged)

Same diagnosis as BTC. The `kage ≥ 3` requirement was rejecting too many
real setups during sharp DJI moves where Kalman flips for just 1-2 bars
before the trend resumes.

8-month holdout sweep:

| Variant | Trades | WR | PF | DD | $@0.10 |
|---|---:|---:|---:|---:|---:|
| kage≥3 Q≥3.0 (was deployed) | 1,549 | 68.9% | 1.28 | 111 | $5,597 |
| **kage≥1 Q≥3.0 (deployed)** | **1,846** | 68.2% | **1.32** | 118 | **+$7,588** |

**+36% $ at +6% DD** — cleanest XAU/BTC-style result on DJI. PF and DD
both improved; only trade-off is a 0.7pp WR drop, more than offset by
the larger trade count.


## 🧠 2026-06-09 — Time-of-day filter (deployed)

After a 30-day post-deploy backtest sweep, added `time_block_utc=(19, 2)` to
block entries between **19:00 and 02:00 UTC** (late US session + Asia overnight).

DJI cash close is around 21:00 UTC, so this blocks the thin pre-close session
plus all overnight where Atlas's reversal pattern stops out frequently.

| | Baseline | + time-block 19-02 |
|---|---|---|
| Trades | 335 | 257 (-23%) |
| WR | 68.7% | **74.7%** |
| PF | 1.36 | **1.83** |
| sumR | +207 | **+296** |
| DD | 92.30 | **51.67** (-44%) |
| $@0.10 | $1,549 | **$2,224** (+44%) |

Disabled with `time_block_utc = (0, 0)`.


## Holdout (post-Sep 2025, 8 months unseen)

| Q | trades | WR | PF | sumR |
|---:|---:|---:|---:|---:|
| 1.0 | 3,465 | 69.5% | 1.24 | +1,425 |
| 2.0 | 3,380 | 69.6% | 1.25 | +1,425 |
| **3.0** | **1,678** | **69.1%** | **1.26** | **+758** |
| 4.0 | 160 | 68.1% | 1.66 | +199 (too thin) |

Q=3.0 chosen for best balance of PF and volume.

## Entry rule (STRICT)

Identical to Atlas XAU — see `products/atlas_xau/README.md` for details.

## Training data

- `data/m1_dji_orderflow.parquet` (2.79M bars, 2018 → May 2026)
- 28,604 MFE≥2R candidates (70.6% of all candidates have a ≥2R favourable move)

## Scripts

| Script | Purpose |
|---|---|
| `scripts/10_sim_today.py` | Pull fresh Dukascopy bars, simulate today's trades with deployed config |

</details>
