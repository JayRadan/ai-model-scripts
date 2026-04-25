# v7.4 Pivot Score — let ML detect the pivot, no hand-coded rules

**Hypothesis:** v73 failed because hand-coded pivot rules (fractal/zigzag/wickrej/etc.)
each only check one condition and fire on every micro-wiggle. Real M5 pivots are
*confluence events*. Train ONE XGBoost on every bar to score `P(this bar is a pivot)`
using regime features + new H1/H4/session/round-number confluence features.

**Pipeline:**
1. `00_compute_features_v74.py` — 18 v72L + 4 pivot-context (from v73) + ~10 new
   confluence features (H1/H4 swing distances, session HOD/LOD, round-number,
   RSI_H1, streak count, etc.). Per-bar.
2. `01_label_every_bar.py` — for each bar i, forward-simulate best-possible long
   AND short with 4×ATR safety SL within 60 bars. `is_pivot = max(R_long, R_short) > 1.5`.
3. `02_train_pivot_score.py` — single XGB classifier on every bar. Save model.
4. `03_build_setups_v74.py` — for each bar where `p_pivot > thr`, emit a setup
   in Oracle's schema, partitioned by cluster (5 files setups_{cid}_v74.csv).
5. `04_validate_v74.py` — literal copy of Oracle's validate, ingests v74 setups.
   Same per-(cid, rule) confirm + exit head + meta gate.

Reuses from v73:
- `data/cluster_per_bar_v73.csv` (regime per bar)
- 18 v72L feature compute logic

Promotion gate: PF ≥ 2.5, WR ≥ 55%, ≥ 5 trades/day, max DD ≤ 1.5× Oracle DD.
