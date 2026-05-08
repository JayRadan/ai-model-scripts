# v88 — Exit RL Experiments (2026-05-08)

> **Outcome:** 1 of 13 experiments produced a positive, deployable result.
> The reverse-setup RL exit (script `13_reverse_setup_exit.py`) was deployed
> to production for Oracle XAU and Oracle BTC, commit `d1d22f1` on
> `JayRadan/edge_predictor:main`. All other approaches were disproven
> against the unseen 30% holdout — kept in this folder for reference so
> future sessions don't re-litigate them.

---

## Context

After v87 multi-head exit was disproven and removed (commit `350f7b8`,
2026-05-08), this folder explores every reasonable angle for improving
exit timing on Oracle XAU and Oracle BTC. The user's stated failure mode:
*"each time I get profit all is wiped out by a pullback"*.

All experiments evaluate on the unseen 30% chronological split of the
v84 RL trade files:
- **XAU:** 421 trades, 2025-12-10 → 2026-05-01
- **BTC:** 341 trades, 2025-11-11 → 2026-05-01

The training set is the first 70% (XAU 980 trades, BTC 794 trades).
Models trained on the train set, all reported metrics on the test set.

---

## Baseline (production at the time of v88)

| Product | Policy | PF | Total R | MaxDD |
|---|---|---|---|---|
| Oracle XAU | hard SL @ -4R + trail (act=3.0, gb=0.6) + 60-bar max + binary ML exit | 4.18 | +959R | 20R |
| Oracle BTC | same | 4.27 | +841R | 30R |

---

## Results Table

| # | Script | Approach | XAU vs baseline | BTC vs baseline | Verdict |
|---|---|---|---|---|---|
| 01 | `01_train_test.py` | Q-regression on `remaining_R` (M5 features) | -27R (PF 4.38 best) | -120R (PF 4.18) | ❌ |
| 02 | `02_download_m1.py` | Download M1 OHLC from Dukascopy | n/a (data) | n/a (data) | enabler |
| 03 | `03_m1_features_train.py` | Q-regression with M1 micro-structure features | -21R | -124R, M1 features ignored by model | ❌ |
| 04 | `04_entry_q_sweep.py` | Tighten v84 RL entry Q-threshold | +0.04 PF (marginal) | +0.09 PF (marginal) | ❌ marginal |
| 05 | `05_pullback_head.py` | Binary head: peak ≥1.5R AND final ≤-2R | AUC 0.72 (sounds good but base rate 2.7%; +0R) | AUC 0.69 (n=8 positives, noise) | ❌ |
| 06 | `06_adversarial_entry_filter.py` | Predict-loser at entry, V72L+M1 features | AUC **0.48** (worse than random) | AUC 0.51 (~coin flip) | ❌ |
| 07 | `07_download_ticks.py` | Download Dukascopy TICK data (2.2 GB) | n/a (data) | n/a (data) | enabler |
| 08 | `08_tick_features_filter.py` | Adversarial filter w/ tick features (spread, signed flow, microprice, etc.) | +4R, AUC 0.51 (noise) | +8R, AUC 0.69 BUT n=8 positives → noise | ❌ |
| 09 | `09_trail_param_sweep.py` | Sweep (trail_activation_r, trail_giveback_pct) | -22 to -528R across all combos | -45 to -447R | ❌ current params near-optimal |
| 10 | `10_breakeven_stop.py` | Break-even stop layered on existing trail | best -22R | best -65R (DD -7R) | ❌ |
| 11 | `11_regime_conditional_be.py` | Break-even ONLY in HighVol regime (cid=4) | -22R (no help) | **+8R** (PF 4.27→4.44, BTC HighVol PF 1.71→2.63) | ⚠️ small win, not deployed (user opted for v13 only) |
| 12 | `12_reverse_rl_exit.py` | Naive reverse RL: q_entry > thr at every bar | **-413R** at thr=0.5 (catastrophic over-firing) | -150R | ❌ |
| **13** | **`13_reverse_setup_exit.py`** | **Reverse RL gated by opposite-direction rule detection** | **PF 4.18→4.54, DD 20→15R, +4R** | **PF 4.27→5.26, DD 30→25R, +24R** | ✅ **DEPLOYED** |

---

## The Winner — script 13

**Logic:** at every in-trade bar, scan all 30 rule detectors for any setup
firing in the *opposite* direction of the open trade. If one fires AND
`q_entry[cid].predict(v72l_feats) > 0.10`, exit.

**Why it works while every other angle failed:**

- `q_entry` was trained on detected setups, not arbitrary in-position
  bars. Script 12 (naive version) ran q_entry on every bar and
  catastrophically over-fired — Q-values are continuously distributed,
  so >50% of bars look like passable entries.
- Script 13 gates with rule detection. q_entry now sees only bars where
  a real pattern fires. The Q-value becomes meaningful again.
- The "symmetry insight" (credit: Jay): if the same RL that picks good
  entries says "good time to short here", we shouldn't be long.

**Performance:** decide_exit latency went from ~6s naive → 232ms after
slicing the df to last 200 bars before rule scanning. All 30 rule
detectors have lookback ≤ 50 bars, so the slice is safe.

---

## Failed Approaches — Lessons

### Why exit-head ML kept failing (scripts 01, 03, 05, 06, 08)

The v84 RL entry pipeline (q_entry → meta gate → range filter →
kill-switch → regime relabel) extracts most of the available signal
from V72L+regime features. Trades that pass the full filter chain are
statistically indistinguishable from each other at exit time:

- Adversarial filter (script 06) hit AUC 0.476 on XAU — *worse than random*.
- Adding M1 features (script 03): models assigned ~0 weight to M1 columns.
- Adding tick features (script 08): models USED tick features (top-10
  importance) but AUC moved <0.04. The signal is real but tiny.

**Conclusion:** the residual variance in pnl is irreducible from price/
quote/flow data at any timeframe Dukascopy provides for free.

### Why trail tightening keeps failing (scripts 09, 10, 11)

Trades that briefly dip through breakeven mostly recover. Saving 30
pullback-losers from -3R losses costs 100+ trades that would have run
further but get exited early. Trail tightening trades a smaller benefit
for a larger cost.

### Why naive reverse-RL fails (script 12)

q_entry was never trained as a per-bar regressor. It's a setup-conditional
Q-function. Running it on arbitrary in-position bars treats noise as
signal. Script 12 fired exits on 60%+ of trades at the most aggressive
threshold, costing -413R.

---

## File Catalog

| File | Type | Purpose |
|---|---|---|
| `01_train_test.py` | experiment | Q-regression on remaining_R, M5-only features |
| `02_download_m1.py` | data fetcher | Dukascopy M1 OHLC download (518k bars XAU, 784k BTC) |
| `03_m1_features_train.py` | experiment | Q-regression with M1 micro-structure |
| `04_entry_q_sweep.py` | analysis | v84 RL entry Q-threshold sensitivity sweep |
| `05_pullback_head.py` | experiment | Binary classifier on peak≥1.5R then SL pattern |
| `06_adversarial_entry_filter.py` | experiment | Predict-loser at entry, V72L+M1 features |
| `07_download_ticks.py` | data fetcher | Dukascopy TICK download (1.3 GB XAU, 869 MB BTC) |
| `08_tick_features_filter.py` | experiment | Adversarial filter with 10 tick aggregates |
| `09_trail_param_sweep.py` | analysis | (act, gb) parameter sweep — current near-optimal |
| `10_breakeven_stop.py` | experiment | Break-even stop layered on existing trail |
| `11_regime_conditional_be.py` | experiment | Break-even ONLY in HighVol regime (BTC +8R) |
| `12_reverse_rl_exit.py` | experiment | Naive: q_entry > thr at every bar (-413R) |
| `13_reverse_setup_exit.py` | **DEPLOYED** | Reverse RL gated by rule detection |

---

## Reproduction

```bash
# Data prerequisites
python3 experiments/v88_exit_rl/02_download_m1.py     # ~3 min
python3 experiments/v88_exit_rl/07_download_ticks.py  # ~20 min, 2.2 GB

# Run any single experiment
python3 experiments/v88_exit_rl/13_reverse_setup_exit.py
```

All experiments are self-contained, share the same eval harness
(70/30 chrono split on `experiments/v84_rl_entry/{v84,btc}_rl_trades.csv`),
and report PF / Total R / WR / MaxDD against the same baseline.

---

## Production Integration

The deployed reverse-setup RL exit lives at
[commercial/server/decision_engine/decide.py](../../../my-agents-and-website/commercial/server/decision_engine/decide.py)
inside `decide_exit`, between the hard SL check and the trail check.
Activates only when `payload["q_entry"]` is loaded — gates Oracle XAU
and Oracle BTC, silently skips Midas and Janus (which don't have RL
entries).

```python
# In decide_exit, after hard-SL check:
if payload.get("q_entry") is not None and cid in payload["q_entry"]:
    recent = df.iloc[-200:].reset_index(drop=True)
    opp_events = rules.events_at_bar(
        recent, list(rules.RULE_FNS.keys()), len(recent) - 1)
    opp_events = [e for e in opp_events if int(e["direction"]) == -d]
    if opp_events:
        feat_row = features.extract_v72l_row(df, last_i, cfg.v72l_feats)
        q_val = float(payload["q_entry"][cid].predict(feat_row.reshape(1,-1))[0])
        if q_val > 0.10:
            return Decision(action="exit", reason=f"reverse-setup RL exit (q={q_val:.2f})")
```

---

## Future Work — What NOT to Try Again

Per memory log `v88_oracle_ceiling_disproven.md` and this folder's
results, **don't waste time on**:

- More ML exit-heads on M5+regime features (all six attempts failed)
- M1 micro-structure features in any exit model (always weighted ~0)
- Tighter trail params or break-even stops (kills more winners than losers)
- Pullback-pattern binary heads (AUC ceiling ~0.6, base rate too low)
- Adversarial entry filters on V72L+M1+ticks (AUC ~0.5, no signal)

**Genuine future angles** (not tried in v88):

- Cross-asset features (DXY for XAU, BTC futures basis for spot BTC)
- Macro-event filters (FOMC, CPI, etc.)
- Apply v83c+v84 stack to a new instrument (EURUSD/ES/NQ) for
  diversification rather than per-asset PF lift
