# Deployment — Regime-Specialist Live Trading

This is the **Option A** deployment bundle. Models were trained on 100% of cluster data, so the backtest numbers are in-sample and optimistic. Use as a pilot, monitor closely, and run Option B (clean retrain) before scaling up.

## Files you need

| File | Purpose | Goes where |
|---|---|---|
| `models/regime_0_Ranging.onnx` | C0 Ranging model (3-class, 36 features) | `MQL5/Files/Common/` |
| `models/regime_1_Downtrend.onnx` | C1 Downtrend model (binary, 36 features) | `MQL5/Files/Common/` |
| `models/regime_3_Uptrend.onnx` | C3 Uptrend model (binary, 36 features) | `MQL5/Files/Common/` |
| `regime_selector.mqh` | Weekly regime detection (constants + math) | `MQL5/Include/` |
| `regime_router.mqh` | Multi-model ONNX router | `MQL5/Include/` |
| `deployment_config.json` | Thresholds + stats (reference) | project folder |
| `runtime_inference.py` | Python reference implementation | project folder |

## MT5 setup

1. Copy the three `.onnx` files into `<MT5_data_folder>/MQL5/Files/Common/`.
2. Copy `regime_selector.mqh` and `regime_router.mqh` into `<MT5_data_folder>/MQL5/Include/`.
3. Modify your existing EA (see diff below) to use the router.
4. Compile in MetaEditor. Ensure no errors.
5. Attach the EA to XAUUSD M5.

## How to modify your existing EA (SwingScalperEA_v3.mq5)

**Add the include at the top (after `#include <Trade/Trade.mqh>`):**
```mql5
#include <regime_router.mqh>
```

**Replace the single-model inputs:**
```mql5
// OLD
input string InpModelFile = "swing_model_v3.onnx";

// NEW
input string InpModelC0 = "regime_0_Ranging.onnx";
input string InpModelC1 = "regime_1_Downtrend.onnx";
input string InpModelC3 = "regime_3_Uptrend.onnx";
input int    InpRegimeRefreshBars = 288;   // refresh regime every N M5 bars (1 day)
```

**In `OnInit()` — replace the single `OnnxCreate` with:**
```mql5
if(!RR_LoadModels(InpModelC0, InpModelC1, InpModelC3))
{
    Print("Failed to load one or more ONNX models");
    return INIT_FAILED;
}
RR_RefreshRegime(MODEL_TF, _Symbol);
```

**In `OnDeinit()`:**
```mql5
RR_ReleaseModels();
```

**In your bar handler — before running the model, refresh regime periodically:**
```mql5
static int bars_since_refresh = 0;
if(++bars_since_refresh >= InpRegimeRefreshBars)
{
    RR_RefreshRegime(MODEL_TF, _Symbol);
    bars_since_refresh = 0;
}
```

**Replace your single-model inference block with:**
```mql5
// Compute 36 features into feat[] (unchanged from v3)
float feat[36];
// ... your existing feature computation ...

double win_prob = 0.0;
RR_SIGNAL signal = RR_Predict(feat, win_prob);

if(signal == RR_BUY)  { /* open long  with ATR TP/SL */ }
if(signal == RR_SELL) { /* open short with ATR TP/SL */ }
// RR_FLAT → do nothing
```

**TP/SL are unchanged from v3:** `TP = entry ± 2×ATR(14)`, `SL = entry ± 1×ATR(14)`, max hold 40 bars.

## Per-cluster thresholds (already baked into `regime_selector.mqh`)

| Cluster | Name | Threshold | Model | In-sample PF |
|---|---|---|---|---|
| C0 | Ranging | 0.40 | 3-class | 1.88 |
| C1 | Downtrend | 0.40 | SELL-or-FLAT | 2.14 |
| C2 | Shock_News | — | *never trade* | — |
| C3 | Uptrend | 0.35 | BUY-or-FLAT | 1.29 |

## Pre-flight checklist

- [ ] Copy 3 `.onnx` files to `MQL5/Files/Common/`
- [ ] Copy 2 `.mqh` files to `MQL5/Include/`
- [ ] Modify EA per the diff above
- [ ] Compile in MetaEditor — should produce no errors
- [ ] Attach to XAUUSD M5 on a **demo account first**
- [ ] Watch the Experts log for `RR: active regime = ...` messages at startup and every day
- [ ] Verify at least one `RR: OnnxRun ... failed` is NOT printed
- [ ] Let it run at least one full trading week before comparing to backtest
- [ ] Monitor per-regime PnL separately (log which cluster produced each trade)

## Known caveats (repeat from `deployment_config.json`)

1. **In-sample backtest**: the 100%-trained models saw the "holdout" during training. Real live PF will likely be lower than the headline numbers (C0 1.88, C1 2.14, C3 1.29). The walk-forward F1 (0.32 / 0.53 / 0.52) is the more honest generalization estimate.
2. **C0 only delivers ~2 trades/day active** — below the 7/day target. Acceptable if you're OK with low volume during ranging weeks.
3. **The 21 base features (f01..f20)** must be computed live exactly as your existing feature pipeline does. Do not change their formulas; the models were trained on those specific values.
4. **The 15 tech features (rsi14, stoch_k, mom5, bb_pct, etc.)** are defined in `labeler.py::compute_tech_features()` — your MQL5 code must match those formulas exactly. Your existing v3 EA already does this; do not change it.
5. **Regime refresh cost**: `RR_RefreshRegime` walks 1440 M5 bars (5 trading days) and runs a K-means nearest-centroid search. It's cheap but should not run on every tick — call it once per day as shown above.

## Next step: Option B

After you've pilot-tested Option A and want honest generalization numbers:

1. Modify `train_regime_models.py` to train on first 80% of each cluster only (leave the last 20% untouched).
2. Save those models as `regime_*_honest.json`.
3. Run the existing `backtest_regimes.py` against the honest models.
4. Compare with the in-sample numbers above — delta tells you how much the models are overfit.
5. Once you trust the numbers, retrain on 100% again for live and ship.
