# How to retrain all 4 products on new data

Last updated: 2026-05-04. Documents the EXACT pipeline used to produce the
currently-deployed pkls. Run these in order to retrain on fresh data later.

## Required input files

```
data/swing_v5_xauusd.csv      — XAUUSD 5m bars from Dukascopy + tick_volume + label
data/swing_v5_btc.csv         — BTCUSD 5m bars from Dukascopy + tick_volume + label
data/labeled_v4.csv           — XAU bars labeled by 01_labeler_v4.py (ATR-based)
```

## Critical params (do not change without testing)

- `MIN_DATE = "2016-01-01"` in selector scripts (NOT 2020 — that broke v9.3)
- `random_state = 42` everywhere (KMeans, XGB)
- `WINDOW = 288` and `STEP = 288` for regime fingerprint
- `K = 5` clusters
- `Cutoff = 2024-12-12` for train/test holdout split

## Critical lesson learned

**ALWAYS commit the selector script BEFORE running it.** The v9.3 winner (PF 4.17)
was lost because we edited the script 4 minutes after the winning run without
git-tracking the original version. We can reproduce PF ~3.40 now, not PF 4.17.

## Reproducible target after May 4 forensic dig

The pipeline in this repo is **deterministic** and reproduces:

```
Oracle XAU: PF 3.40, WR 63.7%, n=1602, DD -62R   (verified twice on May 4)
Midas XAU:  PF 2.96, WR 61.1%, n=2583, DD -90R
Janus XAU:  PF 2.66, WR 57.8%, n=6652, DD -307R
Oracle BTC: PF 2.98, WR 60.9%, n=2092, DD -78R
```

The currently DEPLOYED pkls (May 3 23:55) perform somewhat differently:
- Oracle XAU deployed: PF 3.70 (better than retrain — happy accident)
- Midas XAU deployed:  PF 2.92 (~equal to retrain)
- Oracle BTC deployed: PF 2.98 (identical to retrain — same selector seed)
- Janus XAU deployed:  PF 2.78 (slightly better than retrain)

**Production was NOT touched on May 4** because deployed pkls match or beat
fresh retrain. New retrain only when training data refreshes substantially.

## What we tried but couldn't recover (PF 4.17)

The May 3 v9.3 winner produced PF 4.17 with n=1402 trades. Today's
pipeline produces PF 3.40 with n=1602 trades — using IDENTICAL selector
centroids (verified to 4 decimals against FINDINGS.md flow values).

Tested hypotheses for the lost 0.77 PF:
- ❌ MIN_DATE in selector (tried 2016, 2018, 2020 — same result)
- ❌ Random state (locked at 42)
- ❌ Selector script (centroids match yesterday's exactly)
- ❌ Cluster split logic (script in git is unchanged)
- ❌ Junk-file pollution (cleaned, didn't change result)

Possible unrecoverable causes:
- Some intermediate file (cluster_per_bar mapping) had subtle differences
- Floating-point non-determinism somewhere in physics features
- The selector script edited at 21:45 had something not in current version

The 16% gap in confirmed train setups (14,010 winner vs 16,272 today)
suggests per-rule training data differed — but we cannot trace why
when selector and cluster scripts produce identical centroids.

**Conclusion:** PF 3.40 is the reproducible target. PF 4.17 was a
moment-in-time result that depended on lost intermediate state.

---

## Oracle XAU (v7.2-lite, K=5 K-means + flow_4h_mean)

```bash
# 1. Build XAU regime selector (8 features incl. flow_4h_mean)
python experiments/v93_flow_in_regime/01_selector_with_flow_feature.py
# → writes data/regime_selector_K4.json + data/regime_fingerprints_K4.csv

# 2. Split XAU data into per-cluster files
python model_pipeline/03_split_clusters_k5.py
# → writes data/cluster_{0..4}_data.csv

# 3. Build setup signals for Oracle (per-rule scanner)
python model_pipeline/04_build_setup_signals.py
# → writes data/setups_{0..4}.csv

# 4. Compute physics features + merge into setups
python model_pipeline/04b_compute_physics_features.py
# → writes data/setups_{0..4}_v6.csv

# 5. Compute v72-lite features (vpin, sig_quad_var, har_rv_ratio, hawkes_eta)
python experiments/v72_lite_deploy/00_compute_v72l_features_step1.py
# → writes data/setups_{0..4}_v72l.csv

# 6. Train + validate Oracle XAU
python experiments/v72_lite_deploy/01_validate_v72_lite.py
# → writes meta_threshold_v72l.txt + holdout trades CSV
# → prints holdout PF/WR/DD

# 7. Pickle for deployment
cd ../my-agents-and-website/commercial/server/decision_engine
rm -f /tmp/oracle_deployed_pipeline_cache.pkl   # force fresh retrain
python scripts/pickle_validated_models.py oracle_xau
# → writes new-model-zigzag/models/oracle_xau_validated.pkl
# Copy to: commercial/server/decision_engine/models/oracle_xau_validated.pkl
```

## Midas XAU (v6, K=5 K-means + flow_4h_mean, NO meta gate)

Reuses Oracle's selector + cluster split + setups + physics. Different validate.

```bash
# Steps 1-4 same as Oracle XAU above (selector + cluster split + setups + physics).
# Then:

# 5. Train + validate Midas v6 (writes _v6_validated_raw.pkl)
python experiments/v6_xau_deploy/01_validate_v6.py

# 6. Pickle for deployment
cd ../my-agents-and-website/commercial/server/decision_engine
python scripts/pickle_validated_models.py midas_xau
# → writes new-model-zigzag/models/midas_xau_validated.pkl
# Copy to: commercial/server/decision_engine/models/midas_xau_validated.pkl
```

## Janus XAU (v7.4 pivot-score, K=5 K-means)

Reuses Oracle's selector. Has its own pivot-score cascade.

```bash
# Steps 1-2 same as Oracle (selector + cluster split). Then:

# 3. Regenerate cluster_per_bar_v73.csv (Janus v74 dependency)
python experiments/v73_pivot_oracle/00b_regen_cluster_per_bar.py
# → writes experiments/v73_pivot_oracle/data/cluster_per_bar_v73.csv

# 4. Compute Janus v73 features (43 features per bar)
python experiments/v73_pivot_oracle/00_compute_features_v73.py

# 5. Compute v74 pivot-context features (extends to 55)
python experiments/v74_pivot_score/00_compute_features_v74.py

# 6. Label every bar (is_pivot_25 etc.)
python experiments/v74_pivot_score/01_label_every_bar.py

# 7. Train pivot-score + pivot-direction models
python experiments/v74_pivot_score/02_train_pivot_score.py

# 8. Build setups using pivot scores
python experiments/v74_pivot_score/03_build_setups_v74.py

# 9. Train + validate Janus
python experiments/v74_pivot_score/04_validate_v74.py
# → writes experiments/v74_pivot_score/reports/meta_threshold_v74.txt

# 10. Pickle for deployment
python experiments/v74_pivot_score/scripts/pickle_janus.py
# → writes new-model-zigzag/models/janus_xau_validated.pkl
# Copy to: commercial/server/decision_engine/models/janus_xau_validated.pkl
```

## Oracle BTC (v7.2-lite, K=5 K-means + flow_4h_mean, separate selector)

Has its OWN selector and pipeline (BTC-specific).

```bash
# 1. Build BTC regime selector (8 features incl. flow_4h_mean, BTC data)
python experiments/v93_flow_in_regime/02_selector_btc_with_flow.py
# → writes data/regime_selector_btc_K5.json

# 2. Cluster split + setup signals + physics + v72l (BTC track)
python experiments/v72_lite_btc_deploy/01_prepare_btc.py
python experiments/v72_lite_btc_deploy/04_build_setup_signals_btc.py
python experiments/v72_lite_btc_deploy/04b_compute_physics_features_btc.py
python experiments/v72_lite_btc_deploy/00_compute_v72l_features_step1_btc.py

# 3. Train + validate Oracle BTC
python experiments/v72_lite_btc_deploy/01_validate_v72_lite_btc.py

# 4. Pickle for deployment
cd ../my-agents-and-website/commercial/server/decision_engine
rm -f /tmp/oracle_btc_pipeline_cache.pkl    # force fresh retrain
python scripts/pickle_validated_models.py oracle_btc
# → writes new-model-zigzag/models/oracle_btc_validated.pkl
# Copy to: commercial/server/decision_engine/models/oracle_btc_validated.pkl
```

---

## Deployment (after all 4 pkls + selectors are produced)

```bash
# Copy 4 pkls + 2 selector JSONs to commercial repo
cd /home/jay/Desktop/my-agents-and-website
for prod in oracle_xau midas_xau oracle_btc janus_xau; do
  cp new-model-zigzag/models/${prod}_validated.pkl \
     commercial/server/decision_engine/models/
done
cp new-model-zigzag/data/regime_selector_K4.json \
   commercial/server/decision_engine/data/
cp new-model-zigzag/data/regime_selector_btc_K5.json \
   commercial/server/decision_engine/data/

# Test deployed pkls before pushing (to catch silent regressions)
python /tmp/test_deployed_pkls.py oracle_xau   # see /tmp/test_deployed_pkls.py
python /tmp/test_deployed_btc.py
python /tmp/test_deployed_midas.py

# Tag rollback point + commit + push
cd commercial
git tag -a "pre-retrain-$(date +%Y%m%d)" -m "Pre-retrain rollback point"
git add server/decision_engine/models/ server/decision_engine/data/
git commit -m "Retrain all 4 products on $(date +%Y-%m-%d) data"
git push origin main
# Render auto-deploys ~2-3 min
```

## Junk-file warning

The pipeline has a bug where `04b_compute_physics_features.py` and
`00_compute_v72l_features_step1.py` glob `setups_*.csv` and re-process their
own outputs, creating recursive `setups_0_v6_v6.csv` style junk files. 

**Both scripts have been patched (May 4 2026)** to filter to canonical
`setups_{N}.csv` only via regex. If you re-introduce that bug, expect
training to silently use stale data.

Cleanup command if junk files appear:
```bash
rm data/setups_*_v6_v6*.csv data/setups_*_v72l_v6*.csv data/setups_*_v6_btc*.csv data/setups_*_btc_v6_v6*.csv
```
