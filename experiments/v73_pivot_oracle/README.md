# v7.3 Pivot Oracle — Experiment

**Status:** scaffolding only — no models trained yet.
**Goal:** more trades/day, smaller wins, smarter entries by detecting **turning points** in real time.
**Isolation:** this folder is a sandbox. It does NOT touch:
- `models/` (production XGB pickles)
- `live_deployment/` (legacy MT5 EAs)
- `commercial/server/decision_engine/` (live decision engine)
- any existing `oracle_xau` / `oracle_btc` / `midas_xau` artifacts

All outputs land in `experiments/v73_pivot_oracle/{data,models,reports}/`.

## What's different from Oracle (v7.2-lite)

| | v7.2-lite Oracle | v7.3 Pivot Oracle |
|---|---|---|
| Rule families | 28 chart-pattern rules (BB, NR4, 3-bar reversal, etc.) | **1 family only — turning-point pivots** (RP_*) |
| Detector latency | 3-10 bars (full pattern) | 1-2 bars (pivot confirmation) |
| Exit | SL=4×ATR, ML-exit head, 60-bar max-hold | TP1=0.5×ATR (50%) + TP2=1×ATR (50%), SL=0.8×ATR, 8-bar time-stop |
| Target trades/day | ~3 | 15-25 |
| Target WR | 68% | 55-60% (acceptable trade-off) |
| Target PF | 3.96 | 1.8-2.5 |
| Meta gate | yes (0.675) | optional — start without |

## The single rule family — `RP_*` (Reversal Pivot)

Five sub-detectors, all firing on the **just-closed bar** (no look-ahead):

- **RP_a fractal_pivot** — Williams fractal (N=2 or N=3). Bar[-2] is local high/low vs neighbours.
- **RP_b zigzag_realtime** — running ZigZag with adaptive threshold (0.3-0.8×ATR). Fire when new leg confirmed by counter-move.
- **RP_c momentum_divergence** — RSI/MACD divergence between last two swings.
- **RP_d wick_rejection** — bar with wick ≥ 2× body in reversal direction + close in opposite half.
- **RP_e volume_climax** — abnormal range bar (range > 2× ATR) closing against prior leg.

Each gets a per-rule XGBoost confirmation head trained on multi-scale pivot labels.

## Multi-scale labelling

Each pivot is labelled at three scales:
- **micro** — counter-move ≥ 0.3×ATR
- **mid** — counter-move ≥ 0.8×ATR
- **major** — counter-move ≥ 2.0×ATR

One classifier per (rule, scale) → 5 × 3 = **15 confirm heads**.

## Folder layout (mirrors v72_lite_deploy)

```
v73_pivot_oracle/
├── README.md                          # this file
├── 00_label_pivots.py                 # generate multi-scale pivot labels from swing CSV
├── 01_compute_v73_features.py         # 18 v72-lite features + 6 pivot-specific (slope, leg-len, etc.)
├── 02_build_setups_pivot.py           # for each RP_* rule, emit setups_<cid>_v73p.csv
├── 03_validate_v73_pivot.py           # train 15 confirm heads + holdout forward-sim
├── 04_backtest_tiered_exits.py        # apply TP1/TP2/time-stop exit logic, report PnL
├── 05_compare_to_oracle.py            # side-by-side metrics vs frozen v7.2-lite holdout
├── data/                              # pivot labels, setups CSVs (gitignored if large)
├── models/                            # XGB native + ONNX (sandbox only, NOT shipped)
└── reports/                           # backtest PNGs, holdout summary JSON
```

## How to run (when scripts are filled in)

```bash
cd /home/jay/Desktop/new-model-zigzag
python experiments/v73_pivot_oracle/00_label_pivots.py
python experiments/v73_pivot_oracle/01_compute_v73_features.py
python experiments/v73_pivot_oracle/02_build_setups_pivot.py
python experiments/v73_pivot_oracle/03_validate_v73_pivot.py
python experiments/v73_pivot_oracle/04_backtest_tiered_exits.py
python experiments/v73_pivot_oracle/05_compare_to_oracle.py
```

## Promotion criteria (do NOT ship until ALL pass)

- Holdout PF ≥ 1.8
- Holdout WR ≥ 55%
- Trades/day median ≥ 10
- Max DD ≤ 1.5× v7.2-lite max DD at same lot size
- Per-scale PF: at least micro+mid both > 1.4 (don't promote on majors alone — that's just slower Oracle)

If promoted: register a **new server slug** `oracle_xau_v73` in `commercial/server/decision_engine/configs/`. **Never overwrite the `oracle_xau` pickle.** Run both side-by-side on a test account first.

## Open design questions

1. Adaptive vs fixed pivot threshold? Probably adaptive (×ATR) since gold's vol regime swings widely.
2. Gate RP family by regime? Likely skip C1 (MeanRevert) where pivots whipsaw worst.
3. Lot-size scaling by scale tier? micro=0.5×, mid=1×, major=1.5×.
