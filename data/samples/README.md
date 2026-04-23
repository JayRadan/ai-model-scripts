# Training Data Samples

The first 200 rows of each core CSV used by the training pipeline. Full
files are gitignored (swing files are ~1 GB, cluster/setups up to ~120 MB
each) — regenerate them by running the pipeline end-to-end from an MT5
M5 export. These samples exist so a new agent/developer can open them,
see the columns, and understand the shape of each stage's artefact
without needing to run the whole pipeline first.

Every sample is a **strict prefix** of the real file (same columns, same
dtypes, just truncated). You can't train off them — use the full
pipeline output for that.

## What each file is

| Sample | Produced by | Rows (full) | Role |
|---|---|---|---|
| `swing_v5_xauusd.csv` | MT5 export → `model_pipeline/01_labeler_v4.py` | ~1 M | Raw M5 OHLC + spread for XAUUSD. The root of every XAU pipeline. |
| `swing_v5_btc.csv` | MT5 export → `experiments/v72_lite_btc_deploy/01_prepare_btc.py` | ~1 M | Raw M5 OHLC + spread for BTCUSD. |
| `labeled_v4.csv` | `01_labeler_v4.py` | ~445 K | `swing_v5_xauusd` + 21 micro-features + H1/H4 context + `label` column (1 = forward-R ≥ TP, 0 otherwise). |
| `cluster_0_data.csv` / `cluster_0_data_btc.csv` | `03_split_clusters_k5*.py` | ~90 K / ~180 K | Bars assigned to regime cluster 0 (Uptrend) after K=5 K-means on the 7-feature weekly fingerprint. One `cluster_{cid}_*.csv` per cluster — the sample shows cid=0. |
| `setups_0_v6.csv` / `_btc.csv` | `04_build_setup_signals*.py` + `04b_compute_physics_features*.py` | ~18 K / ~35 K | Every bar where a v6 rule fired in cluster 0, enriched with 14 physics features. The per-rule XGB confirmation heads are trained on these. |
| `setups_0_v72l.csv` / `_btc.csv` | Same + `00_compute_v72l_features_step1*.py` | same | v7.2-lite variant: adds 4 extra features (`vpin`, `sig_quad_var`, `har_rv_ratio`, `hawkes_eta`) on top of v6's 14. |
| `regime_fingerprints_K4.csv` / `_btc_K5.csv` | `02_build_selector_k5*.py` | ~2.5 K / ~3.4 K | One row per weekly window: 7-dim fingerprint + assigned cid. The K-means centroids stored in `regime_selector_*.json` are derived from these. |
| `v6_trades_holdout_xau.csv` | `experiments/v6_xau_deploy/01_validate_v6.py` | 2,636 | Per-trade holdout backtest result for v6 Midas XAU (PF 2.24 / WR 57% / DD -80R). |
| `v72l_trades_holdout.csv` | `experiments/v72_lite_deploy/01_validate_v72_lite.py` | 1,368 | Per-trade holdout for Oracle XAU (PF 3.96 / WR 68.5%). |
| `v72l_trades_holdout_btc.csv` | `experiments/v72_lite_btc_deploy/01_validate_v72_lite_btc.py` | 2,440 | Per-trade holdout for Oracle BTC. |

## Column quick-reference

### `swing_v5_<inst>.csv`
Minimum needed by every downstream step:

```
time, open, high, low, close, spread
```

The samples you see here include additional computed columns (micro-features,
H1/H4 context) that `01_labeler_v4.py` already added — the raw MT5 export
has only the 6 columns above.

### `labeled_v4.csv`
Adds the supervision target:

```
... + f01_CPR … f20_NCDE, rsi14, rsi6, stoch_*, bb_pct, mom5/10/20, ll_dist10,
hh_dist10, vol_accel, atr_ratio, spread_norm, hour_enc, dow_enc, idx,
direction, rule, atr, entry_price, label
```

### `setups_<cid>_v6.csv` / `_v72l.csv`
What the per-rule XGB classifiers eat:

```
<every column in labeled_v4> +
hurst_rs, ou_theta, entropy_rate, kramers_up, wavelet_er, vwap_dist,
quantum_flow, quantum_flow_h4, quantum_momentum, quantum_vwap_conf,
quantum_divergence, quantum_div_strength     (← 14 v6-core)

v7.2-lite also has:
vpin, sig_quad_var, har_rv_ratio, hawkes_eta (← 4 extras)
```

Feature semantics are defined in `model_pipeline/04b_compute_physics_features.py`
and `experiments/v72_lite_deploy/00_compute_v72l_features_step1.py`.

### `regime_fingerprints_<K>.csv`

```
week_start, weekly_return_pct, volatility_pct, trend_consistency,
trend_strength, volatility, range_vs_atr, return_autocorr, cid
```

`cid` is the assigned K-means cluster; the CSV is what the K=5 centroids
in `data/regime_selector_*.json` were fitted on.

### Holdout trade dumps

```
time, cid, rule, direction, bars, pnl_R, exit
```

`pnl_R` is realised PnL in units of entry-ATR. `exit` ∈ {`hard_sl`,
`ml_exit`, `max`}.

## Regenerating the full files

The full pipeline takes ~6 min on a recent CPU (XAU, ~1 M bars). See the
top-level `README.md` § 3 for the step-by-step commands. Short version:

```bash
# From raw MT5 export → trained + validated bundle:
python model_pipeline/01_labeler_v4.py
python model_pipeline/02_build_selector_k5.py
python model_pipeline/03_split_clusters_k5.py
python model_pipeline/04_build_setup_signals.py
python model_pipeline/04b_compute_physics_features.py
python experiments/v72_lite_deploy/00_compute_v72l_features_step1.py  # adds 4 extras
python experiments/v72_lite_deploy/01_validate_v72_lite.py            # produces holdout + pickle-ready objects

# Or for Midas v6:
python experiments/v6_xau_deploy/01_validate_v6.py
```
