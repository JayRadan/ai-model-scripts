# v104 — Kalman Regime (Kinematic State-Space) + color-flip entries

Experiment: replace the TFK regime indicator with the **Kalman Regime Features
(Kinematic State-Space)** Pine v6 indicator, and change the entry logic to a
pure **color flip**:

- regime color = sign of Kalman velocity (`kf.v >= 0` → green, else red)
- **red → green flip = BUY**, **green → red flip = SELL**
- entry fills at the open of the bar after the flip
- exit policy unchanged from v103: hard SL = 4×ATR, trail = 3×ATR, max_hold = 300
- an XGBRegressor Q-filter is trained on the flip events (predict forward R),
  then a Q threshold sweep is run on the holdout

This differs from v103, which gated entries on pullback/counter distance to the
TFK line. Here the **flip itself is the entry**.

## Files

| file | purpose |
|---|---|
| `kalman.py` | Python port of the Pine v6 "Kalman Regime Features" indicator |
| `tfk.py` | the old TFK indicator (copied in for a head-to-head on identical bars) |
| `00_download_m1_dukascopy.py` | download XAU/USD M1 OHLCV 2018→2026 from Dukascopy |
| `01_train_test_flip.py` | full flip pipeline: candidates → labels → train Q → holdout sweep; runs Kalman **and** TFK head-to-head on the same bars |

## How to run

```bash
# 1. Get data (needs Dukascopy network access — blocked in some sandboxes)
python 00_download_m1_dukascopy.py            # → data/m1_xau_full.parquet

# 2. Full train + test, head-to-head Kalman vs TFK
python 01_train_test_flip.py                  # reads data/m1_xau_full.parquet

# Smoke test the code path without real data (results meaningless):
python 01_train_test_flip.py --synthetic
```

## Indicator → pipeline mapping (the only design choices)

The Kalman filter feeds the flip pipeline through three interfaces, mirroring how
TFK fed v103:

| pipeline need | TFK source | Kalman source |
|---|---|---|
| regime direction | `committed_dir` (hysteresis) | `sign(kf.v)` — `kf.v >= 0 ? +1 : -1` |
| price-space line | `tfk_line` | `kf.p` (filtered price) — only used for the `dist_at_flip` feature now |
| 6 native features | force, velocity, x_est, regime_w, trend_raw, trend | f_velPct, f_velSignif, f_innovZ, f_volState, f_accel, f_velRaw |

All generic features (RSI, ATR, EMA-distances, slopes, M5/M15/H1 HTF context),
the labeling, the exit policy, the train/test cutoff (2025-09-01), and the
XGBRegressor hyperparameters are identical between the two indicators, so the
head-to-head isolates the regime indicator + flip logic.

## Known consideration — whipsaw

A faithful port of the indicator uses `kf.v >= 0` with **no confirmation**, so
the color flips every time velocity crosses zero. On noisy data this produces
many marginal flips (the smoke test showed ~17× more flips than TFK). The
Q-filter is meant to prune these, but a velocity dead-band / N-bar confirmation
is an obvious follow-up if raw-flip whipsaw hurts the holdout. Not added yet —
the experiment first measures the indicator exactly as written.

## Status

Code complete and smoke-tested end-to-end on synthetic bars. **Real results
pending Dukascopy data** — the download endpoint is blocked by the network
policy in the current execution environment (HTTP 403). Run step 1 where
Dukascopy is reachable, or load an existing `m1_xau_full.parquet` into `data/`.
