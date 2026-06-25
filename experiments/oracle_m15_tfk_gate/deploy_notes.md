# Oracle XAU smart exit deployment (2026-06-22)

## What changed
- Added **smart_exit_mdl** (XGBRegressor) — predicts future-upside R from current trade state
- Added **M15 TFK ANTI gate** as the regime filter (replaces cluster-block of C1+C2)
- Removed trail floor (smart model handles it directly)
- Kept existing exit_mdl as fallback (do not call by default)

## Honest holdout 2.4y (2024-01-01 → 2026-05-01)
- trades: 11,321
- WR: 58.0%
- PF: 1.76
- sumR: +14234
- DD: 271
- avg duration: 49.1 M5 bars (245 min)
- exit mix: hard_sl=3032, smart=4965, max_hold=3324

## Server-side integration

### 1. Entry gate (replaces cluster-block of C1+C2)
At each setup, compute the **causal** M15 TFK direction from the swing close
series:
```python
# rolling resample of M5 swing into M15 bars
# compute TFK on the M15 bars (same compute_tfk as production)
# forward-fill back to M5 grid (causal: each M5 bar sees only the
#   COMPLETED M15 bar before it, NOT the in-progress M15 bar)
```
Then **only fire the setup** if `m15_tfk_dir == -setup_direction` (i.e. M15
trend is opposite the trade direction — counter-trend setups are what Oracle's
rules are designed for).

The per-cluster confirm + meta filter pipeline stays unchanged. Cluster ID is
still used internally for routing to the right confirm model.

### 2. Exit policy (replaces old XGBClassifier exit)
For each bar of an open trade, build the feature vector in this order:
```
['unrealized_R', 'bars_held', 'pnl_vel_3', 'pnl_vel_5', 'mfe_so_far_R', 'mae_so_far_R', 'dd_from_peak_R', 'progress_to_sl', 'progress_to_trail', 'hour_utc', 'dow', 'm15_dir_x_dir', 'cid', 'direction', 'bars_remaining', 'frac_time', 'trade_range_R', 'horizon', 'hurst_rs', 'ou_theta', 'entropy_rate', 'kramers_up', 'wavelet_er', 'quantum_flow', 'quantum_flow_h4', 'vwap_dist']
```
Definitions per feature (all R-units are ATR-normalized using ATR at entry):
- unrealized_R: d*(C[bar]-ep)/atr_at_entry
- bars_held: k (1-based)
- pnl_vel_3: unrealized_R - unrealized_R_3_bars_ago
- pnl_vel_5: unrealized_R - unrealized_R_5_bars_ago
- mfe_so_far_R: running max favorable in R
- mae_so_far_R: running max adverse in R (positive number)
- dd_from_peak_R: mfe_so_far_R - unrealized_R
- progress_to_sl: unrealized_R / 5.0
- progress_to_trail: unrealized_R / 2.0
- hour_utc, dow: time of the CURRENT bar (not entry bar)
- m15_dir_x_dir: m15_tfk_dir_at_current_bar * trade_direction
- cid: cluster id (numeric)
- direction: +1 or -1
- bars_remaining: 60 - k
- frac_time: k / 60
- trade_range_R: mfe_so_far_R - mae_so_far_R
- horizon: 60 (constant for now; computed live as bars available)
- 8 context features at the current bar: hurst_rs, ou_theta, entropy_rate,
  kramers_up, wavelet_er, quantum_flow, quantum_flow_h4, vwap_dist
  (these come from the same physics pipeline as the entry models)

Then:
```python
predicted_upside = bundle["smart_exit_mdl"].predict(X_features.reshape(1,-1))[0]
should_exit = (
    bars_held >= bundle["smart_exit_min_hold"]
    and unrealized_R >= bundle["smart_exit_min_pnl_R"]
    and predicted_upside < bundle["smart_exit_upside_thr"]
)
```

Plus the usual safety net:
- if unrealized_R <= -bundle["smart_exit_sl_hard"]: EXIT (hard SL)
- if bars_held >= bundle["smart_exit_max_hold"]: EXIT (time fallback)

### 3. Files written by this script
- `oracle_xau_validated_smart_exit_2026-06-22.pkl`  ← NEW bundle (side-by-side, no overwrite)
- The existing `oracle_xau_validated.pkl` is UNCHANGED — current production keeps loading it.

### 4. To activate locally (when you're ready)
```bash
# Make a backup first
cp /home/jay/Desktop/new-model-zigzag/products/models/oracle_xau_validated.pkl /home/jay/Desktop/new-model-zigzag/products/models/oracle_xau_validated.pkl.bak_pre_smart_exit_2026-06-22
# Swap in the new bundle
mv /home/jay/Desktop/new-model-zigzag/products/models/oracle_xau_validated_smart_exit_2026-06-22.pkl /home/jay/Desktop/new-model-zigzag/products/models/oracle_xau_validated.pkl
```

### 5. To deploy to the commercial server (when you're ready)
```bash
cp /home/jay/Desktop/new-model-zigzag/products/models/oracle_xau_validated.pkl /home/jay/Desktop/my-agents-and-website/commercial/server/decision_engine/models/oracle_xau_validated.pkl
# then restart server
```

### 6. Rollback (if needed after activation)
```bash
cp /home/jay/Desktop/new-model-zigzag/products/models/oracle_xau_validated.pkl.bak_pre_smart_exit_2026-06-22 /home/jay/Desktop/new-model-zigzag/products/models/oracle_xau_validated.pkl
```

## Honest caveats
- The smart exit + M15 ANTI gate **trades half the volume** of the cluster-gate
  pipeline. sumR is lower (~+14k vs +25k baseline) but PF/DD profile is
  similar. This is risk-reduction.
- The M15 ANTI gate is **reactive** (15-min reaction) vs cluster gate
  (4h reaction). That's the win.
- Context features (hurst_rs, ou_theta, etc.) MUST be computed live by the
  server pipeline — they are not in the swing CSV by default. This already
  works in the current deployment (the old RL exit uses them too).
