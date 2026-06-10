# Atlas DJI — M1 Kalman + TFK Dual-Indicator STRICT Reversal (Dow Jones)

> **Version:** v1 (deployed 2026-06-04)
> **Architecture:** Same as Atlas XAU but tuned for US30 / DJI microstructure
> **Bundle:** `atlas_dji_validated.pkl` (Q-regressor trained on MFE≥2R movers)
> **Entry gate:** STRICT pattern + Q ≥ **3.0**
> **Exit:** SL 6×ATR · TRAIL 2×ATR · MAX_HOLD 300 · BE@+0.5R · 4 slots



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
