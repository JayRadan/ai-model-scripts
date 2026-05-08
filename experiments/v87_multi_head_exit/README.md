# v87 Multi-Head Exit — DISPROVEN AND REMOVED FROM PRODUCTION

> **Status:** ❌ Net-negative on unseen data. Removed from deploy
> 2026-05-08 (commit `350f7b8` on `JayRadan/edge_predictor:main`).
> **Do not re-deploy without re-validation against the v88 eval harness.**

---

## What it tried

Four binary XGBoost heads predicting at every in-position bar:
- `label_gb`: P(currently >1R profit AND about to give back >0.5R)
- `label_up`: P(meaningful upside remaining, >0.5R more from here)
- `label_sl`: P(trade ultimately hits hard SL)
- `label_nh`: P(new high before 1R drawdown)

Combined into a policy: exit if `(p_gb > 0.6 AND p_up < 0.4) OR p_sl > 0.6`,
unless `p_nh > 0.7 AND p_gb < 0.6` (hope override).

## Why the +265R deploy commit was wrong

The original commit (`496e7be`) measured on training/overlapping data.
On a strict chronological 70/30 holdout of v84 RL trades, the policy
destroyed profit on both products.

## Disprove evidence

Wide threshold sweep over 1,470 combinations (`06_threshold_sweep.py`)
on the unseen 30%:

| Metric | XAU | BTC |
|---|---|---|
| Baseline (no multi-head) | PF 4.34, +1196R | PF 4.43, +962R |
| Best multi-head combo | PF 2.71, +329R | PF 2.65, +288R |
| Loss vs baseline | **-866R** | **-674R** |
| Combos beating baseline | **0 / 1470** | **0 / 1470** |

Even the most conservative threshold combinations destroyed ~70% of
total profit. Higher win-rate (~85%) but tiny winners — the head
correctly identifies givebacks but kills runners that drive the bulk
of profit. **Structural failure, not a tuning issue.**

## Files in this folder

| File | Status |
|---|---|
| `01_train_and_test.py` | Original training script (XAU, in-sample) |
| `02_proper_train_test.py` | Train/test split version |
| `03_train_btc.py` | BTC variant |
| `04_sweep_btc_thresholds.py` | BTC-specific sweep |
| `05_holdout_test.py` | Holdout test (showed less impressive numbers) |
| `06_threshold_sweep.py` | **Definitive disprove** — 1470-combo wide sweep |
| `multi_head_exit_bundle.pkl` | XAU bundle (kept locally for forensics) |
| `multi_head_exit_oracle_btc.pkl` | BTC bundle (kept locally for forensics) |

## Deploy / rollback history

- **2026-05-07** — Bundles deployed in commit `496e7be` claiming +265R lift
- **2026-05-08** — Wide threshold sweep on proper unseen window showed
  the lift was overlapping-data artifact. Bundles removed in commit `350f7b8`.
  See [`server/decision_engine/loader.py`](../../../my-agents-and-website/commercial/server/decision_engine/loader.py)
  — `if os.path.exists(multi_head_exit_<product>.pkl)` gates the multi-head
  block, so simply removing the file disabled the policy. No code change
  needed.

## Replacement

The v88 reverse-setup RL exit
([../v88_exit_rl/13_reverse_setup_exit.py](../v88_exit_rl/13_reverse_setup_exit.py))
is the validated successor. It improves PF and lowers MaxDD on both
products and was deployed in commit `d1d22f1`.

## Why this is preserved

Kept in the repo so future "what about a multi-head exit?" questions
can be redirected here with the disprove evidence rather than re-run.
