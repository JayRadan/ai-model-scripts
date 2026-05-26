# Janus XAU — RETIRED 2026-05-11

> **Status:** RETIRED. Removed from website + server + customer billing on 2026-05-11
> (commits `4e0ee0e`, `2ac70b5`, `180a0e5` in `JayRadan/edge_predictor`).

This folder is kept for historical reference only. **Do not redeploy without rerunning
the full holdout validation — Janus underperformed Oracle XAU on the unseen window
that mattered.**

## What Janus was

A standalone XAUUSD pivot-score model based on `experiments/v7.4_pivot_score/`. Used:

- **Pivot-score classifier** (XGB on pivot/swing features) to detect turning points
- **Walk-forward backtest** showed PF ~2.07 on its training holdout
- **Bundle files in [models/](models/)**:
  - `pivot_dir_v74.json` — direction classifier
  - `pivot_score_v74.json` — pivot strength regressor

Marketed as "third product" alongside Oracle. Deploy commit was `~7c12ae8` (around 2026-04-25).

## Why retired

Once Oracle v89 (maturity-aware q_entry) shipped 2026-05-10, Janus's edge collapsed by
comparison:

- Oracle XAU PF 6.44 vs Janus PF 2.07 (both on overlapping windows post-2024-12-12)
- Janus's pivot signal was redundant with Oracle's stretch features (`stretch_100`,
  `stretch_200`, `pct_to_extreme_50`)
- Customer feedback flagged confusion about which gold product to use

Decision: kill the SKU. Customers were grandfathered/refunded as appropriate.

## If you want to revive it

The training pipeline is at `experiments/v7.4_pivot_score/`. Bundle files in
[models/](models/) are still loadable if needed. **Re-run a 60-day fresh holdout
against current Oracle XAU PF before re-deploying** — anything below Oracle's PF
makes Janus a strict downgrade in the customer's eyes.
