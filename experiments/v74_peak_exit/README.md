# v7.4 Peak-Exit — experimental exit head retrain

**Goal:** retrain ONLY the Janus exit head with a strict peak-detector label
("exit AT the actual peak of the trade") instead of the current label
("exit if no future bar within 60 beats current"). The current label produces
99% holds → trades almost always close at max-hold.

**Isolation:**
- No other v74 artifacts are touched.
- Reads (read-only):
  - `experiments/v74_pivot_score/data/features_v74.csv`
  - `experiments/v74_pivot_score/data/labels_v74.csv`
  - `experiments/v73_pivot_oracle/data/cluster_per_bar_v73.csv`
  - `models/janus_xau_validated.pkl` (just to grab pivot_score, dir, confirm, meta)
- Writes:
  - `experiments/v74_peak_exit/models/peak_exit_v74.json`
  - `experiments/v74_peak_exit/reports/comparison.json`

**Production deploy** is conditional on the comparison:
- If holdout PF improves OR holds with shorter avg-bars-held → swap exit_mdl in
  `commercial/server/decision_engine/models/janus_xau_validated.pkl`
- If PF degrades → keep original, do nothing in commercial

## Files

- `00_retrain_peak_exit.py` — generates peak-detector labels from holdout
  trades, trains new XGBoost exit head, runs holdout sim with both heads,
  prints side-by-side comparison.
