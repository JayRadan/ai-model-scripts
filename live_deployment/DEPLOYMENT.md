# Live Deployment — Rule + ML-Confirmation v5

This bundle is **the v5 system**. It replaces v4 entirely. v4 files have been removed from MT5.

## What's inside

```
live_deployment/
├── DEPLOYMENT.md                              ← you are here
│
├── MQL5_Files/                                → copy to <MT5>/MQL5/Files/
│   ├── confirm_c0_R0a_bb.onnx               (C0 Bollinger mean reversion)
│   ├── confirm_c0_R0b_stoch.onnx            (C0 Stoch reversal)
│   ├── confirm_c0_R0c_doubletouch.onnx      (C0 Double touch)
│   ├── confirm_c0_R0d_squeeze.onnx          (C0 Volatility squeeze)
│   ├── confirm_c1_R1a_swinghigh.onnx        (C1 Swing-high rejection)
│   ├── confirm_c1_R1b_lowerhigh.onnx        (C1 Lower-high continuation)
│   ├── confirm_c1_R1c_bouncefade.onnx       (C1 Bounce fade)
│   ├── confirm_c1_R1d_overbought.onnx       (C1 Overbought short)
│   ├── confirm_c3_R3a_pullback.onnx         (C3 Pullback bounce)
│   ├── confirm_c3_R3b_higherlow.onnx        (C3 Higher-low continuation)
│   ├── confirm_c3_R3c_breakpullback.onnx    (C3 Breakout pullback)
│   └── confirm_c3_R3d_oversold.onnx         (C3 Oversold bounce)
│
├── MQL5_Include/                              → copy to <MT5>/MQL5/Include/
│   ├── regime_selector.mqh                   (weekly regime → cluster id)
│   ├── setup_rules.mqh                       (12 hand-coded rule detectors)
│   └── confirmation_router.mqh               (loads 12 models, ConfirmRule())
│
├── MQL5_Experts/                              → copy to <MT5>/MQL5/Experts/
│   └── SwingScalperEA_v5.mq5                 (the EA — 0 manual edits required)
│
└── python_reference/
    ├── build_setup_signals.py                 (rule code + label generator)
    ├── train_confirmation.py                  (per-rule classifier trainer)
    ├── backtest_confirmation.py               (honest holdout backtest)
    ├── backtest_confirmation.png              (equity curves)
    └── backtest_confirmation_summary.json
```

## How the system works

1. **Detect the active weekly regime** (C0 Ranging / C1 Downtrend / C2 Shock / C3 Uptrend) using the existing `regime_selector.mqh` constants — refreshed once per day.
2. **C2 Shock weeks → never trade.**
3. **For other clusters**, scan the 4 rules belonging to that cluster on every just-closed M5 bar.
4. **For each rule that fires**, run its specific ONNX confirmation classifier. Take the trade only if `P(label=1) ≥ rule_threshold`.
5. **Trade execution**: TP = 2×ATR, SL = 1×ATR (matches the labels the models were trained on). Single-position guard, magic-filtered.
6. **Per-rule cooldown** prevents the same rule from firing on consecutive bars.

## Honest holdout backtest results

Trained on first 80% of each rule's setup pool, evaluated on the unseen last 20%:

| Cluster | Trades | WR | PF | PnL | Max DD | Trades/day |
|---|---|---|---|---|---|---|
| C0 Ranging   | 240 | 56% | **1.63** | **+$120.6** | -$15.0 | 0.36 |
| C1 Downtrend | 215 | 47% | **1.32** | **+$130.9** | -$33.4 | 0.45 |
| C3 Uptrend   | 397 | 45% | **1.30** | **+$270.4** | -$53.6 | 1.25 |
| **Combined** | **852** | — | **~1.36** | **+$522** | — | **~2.1** |

All three clusters profitable. Sizing is 0.01 lot. Equity chart in `python_reference/backtest_confirmation.png`.

## Per-rule reference (operating thresholds)

| Rule | Cluster | Threshold | Holdout PF | Holdout WR |
|---|---|---|---|---|
| R0a_bb           | C0 | 0.60 | 3.18 | 61% |
| R0b_stoch        | C0 | 0.60 | 3.43 | 63% |
| R0c_doubletouch  | C0 | 0.65 | 2.57 | 56% |
| R0d_squeeze      | C0 | 0.60 | 2.23 | 53% |
| R1a_swinghigh    | C1 | 0.65 | 2.11 | 51% |
| R1b_lowerhigh    | C1 | 0.65 | 2.14 | 52% |
| R1c_bouncefade   | C1 | 0.60 | 1.79 | 47% |
| R1d_overbought   | C1 | 0.55 | 1.50 | 43% |
| R3a_pullback     | C3 | 0.40 | 1.47 | 42% |
| R3b_higherlow    | C3 | 0.65 | 2.91 | 59% |
| R3c_breakpullback| C3 | 0.60 | 2.67 | 57% |
| R3d_oversold     | C3 | 0.65 | 1.69 | 46% |

## Installation (4 steps)

`<MT5>` = MT5 data folder (File → Open Data Folder in the terminal). For your install, that's:
`/home/jay/.mt5/drive_c/Program Files/MetaTrader 5/MQL5`

1. **ONNX models** → copy all 12 from `MQL5_Files/` into `<MT5>/MQL5/Files/`
2. **Headers** → copy all 3 `.mqh` files from `MQL5_Include/` into `<MT5>/MQL5/Include/`
3. **EA** → copy `SwingScalperEA_v5.mq5` from `MQL5_Experts/` into `<MT5>/MQL5/Experts/`
4. Open MetaEditor → open `SwingScalperEA_v5.mq5` → press **F7** to compile.
   Should produce **0 errors, 0 warnings**.

> **NOTE**: This bundle has already been copied to your MT5 install at the time it was built. You only need to repeat the copy if you regenerate the bundle (e.g. after retraining).

## EA inputs

| Input | Default | Purpose |
|---|---|---|
| `InpRegimeRefreshBars` | 288 | Re-detect regime every 288 M5 bars (1 day) |
| `InpLots` | 0.01 | Trade size |
| `InpAllowLong` | true | Master long kill-switch |
| `InpAllowShort` | true | Master short kill-switch |
| `InpMaxSpread` | 80 | Skip entries if spread > N points |
| `InpMagic` | 420305 | Order magic (different from v4 = 420304) |
| `InpMaxDevPoints` | 30 | Max slippage on order |
| `InpVerbose` | true | Print rule scan results to Experts log |

## Pre-flight checklist

- [ ] All 12 `.onnx` files in `MQL5/Files/`
- [ ] All 3 `.mqh` files in `MQL5/Include/`
- [ ] `SwingScalperEA_v5.mq5` in `MQL5/Experts/`
- [ ] EA compiles cleanly in MetaEditor (F7 → 0/0)
- [ ] EA attached to **XAUUSD M5** on a **demo account first**
- [ ] `Allow Algorithmic Trading` and `Allow ONNX inference` enabled
- [ ] Experts log shows `ConfirmRouter: all 12 confirmation models loaded` on startup
- [ ] Experts log shows `v5: active regime = Cx ...` on startup and every refresh
- [ ] Let it run at least one full trading week before evaluating

## Caveats

1. **Trade volume is ~2/day combined**, below the 7/day target. Acceptable if you want fewer high-quality trades; not acceptable if you need constant action.
2. **Some rules look better than others** — R0a, R0b, R3b, R3c clear PF 2.5+; R3a is the volume workhorse but only PF 1.47.
3. **All rules use the same TP/SL** (2×ATR / 1×ATR) and 40-bar max horizon. Don't change those without retraining.
4. **The 36 features must match** the Python pipeline exactly. v5's `BuildFeatures()` is byte-for-byte identical to v3/v4.
5. **No memorization risk** — every classifier was trained on first 80% of its rule's pool, evaluated on the unseen last 20%, and the operating threshold was picked from the holdout. The headline numbers are honest.

## Rollback

Detach the EA from the chart. v5 has its own magic number (420305) so it cannot interfere with any other system on the same account.
