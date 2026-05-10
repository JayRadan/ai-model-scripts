# v89 Smart Exit + Maturity-Aware Entry (2026-05-09 → 2026-05-10)

> **Outcome:** the original "smart exit" target failed (offline Q-learning
> exit didn't beat v88 reverse-setup), but the diagnostic work uncovered a
> bigger lever: **trend-maturity features at entry**. Adding three maturity
> features to q_entry produced the largest single-step improvement of
> the entire Oracle stack:
>
> - **Oracle XAU**: PF 4.60 → **6.44**, DD 36R → **27R**, WR 71.4% → **77.4%**
> - **Oracle BTC**: PF 4.69 → **5.27**, DD 43R → **36R**, WR 72.0% → **75.1%**
>
> **Deployed 2026-05-10**, commit `7cb5a8f` on `JayRadan/edge_predictor:main`.

---

## What was tested + what shipped

| Script | Question | Result |
|---|---|---|
| `01_optimal_stopping.py` | Backward FQI Q_hold ensemble + 3 aux classifiers (rec/brk/sl) → does this exit better than v88? | ❌ -321R XAU / -323R BTC; aux classifiers overfit (train AUC 0.95 → test 0.81) |
| `02_ablation.py` | Per-rule sweep — which gate actually helps? | Q-stop with eps=1.0 helps modestly; winner-giveback / deep-loser hurt or noise |
| `03_combined_vs_prod.py` | v88 + v89 Q-stop layered combo | +18R XAU / -9R BTC vs current — within noise |
| `04_hard_sl_sweep.py` | Tighter hard SL (-1.5 → -4)? | -4R is optimal; tighter = more whipsaw, much lower R |
| `05_dd_diagnostic.py` | Where does the MaxDD actually come from? | Streaks of consecutive losers (XAU 5-trade window, BTC 11-trade streak) — not single bad trades |
| `06_hard_sl_predictor.py` | Predict hard-SL trades at entry? | Train AUC 1.0 / Test AUC 0.43-0.47 — completely overfit, no generalization |
| `07_micro_regime_diagnosis.py` | Are losers concentrated in counter-trend pullbacks at entry? | Opposite — counter-trend entries are HIGHER quality (RL is mean-reverting) |
| `08_trend_maturity_diagnosis.py` | Are losers concentrated in **stretched / mature trend** entries? | **YES** — clean monotonic relationship: PF 9.7→1.8 as stretch_100 grows |
| `09_maturity_filter_backtest.py` | Backtest skip-tail / size-tier policies | A: skip stretch>15 → BTC pure win (+0.04 PF, +12R) but tiny |
| `10_train_with_maturity.py` | Add maturity to q_entry training? | **+5-9% R lift** on every threshold |
| `11_retrain_q_with_maturity.py` | Full retrain + bundle save | NEW q_entry passes 2-3× more setups at same threshold; HUGE R but distribution shifted |
| `12_full_pipeline_with_new_q.py` | Does lift survive existing confirm/meta? | **YES** — confirm+meta handle new distribution natively |
| `13_proper_full_pipeline_test.py` | Test against true production baseline (PF 4.60 ref) | At q>3.0: XAU PF 6.44 / DD 27R; BTC PF 5.27 / DD 36R ★ |
| `14_high_q_and_stretch_check.py` | Higher Q sweep + does it actually filter stretched entries? | YES — at q>3.0 stretched-entry share drops 38% (XAU), 36% (BTC) |

---

## What ships in production (commit `7cb5a8f`)

### Maturity features (3 floats, signed by trade direction)

```python
def extract_maturity_features(df, idx, direction):
    # 14-bar ATR computed inline from H/L/C
    stretch_100  = (close[idx] - 100bar_low) / atr  if d=+1 else (100bar_high - close[idx]) / atr
    stretch_200  = same on 200-bar window
    pct_to_extreme_50 = position in last 50-bar range, 0=opposite, 1=at top of move
    return [stretch_100, stretch_200, pct_to_extreme_50]
```

Lives in [`commercial/server/decision_engine/features.py`](../../../my-agents-and-website/commercial/server/decision_engine/features.py).

### q_entry retraining

- Same train window as v84 (setups before 2024-12-12)
- Same per-cluster XGBRegressor architecture (n_est=300, depth=4, lr=0.05)
- Features: V72L (18) + maturity (3) = **21 input dim**
- 5 models per product (one per regime cluster), saved into bundle
- Confirm + meta + exit heads from v84 are KEPT UNCHANGED

### Threshold recalibration

`rl_min_q`: 0.3 → **3.0** for both products. The new q_entry's distribution
is shifted upward (V72L+maturity gives stronger signals); 3.0 produces
firing rate similar to v84's 0.3 while concentrating on higher-quality setups.

### Adaptive min_q (preserved from v88)

Trending market (24h |return| > 1%) → relax min_q toward 2.0
Choppy market (24h |return| < 0.3%) → tighten min_q toward 4.0
Both scale relative to base value of 3.0.

---

## Reproduction

```bash
# Phase 1 — retrain q_entry
python3 experiments/v89_smart_exit/11_retrain_q_with_maturity.py
# saves products/models/oracle_*_validated_v89mat.pkl

# Phase 2 — validate against production through full pipeline
python3 experiments/v89_smart_exit/13_proper_full_pipeline_test.py

# Phase 3 — verify higher-Q + stretched-entry filtering
python3 experiments/v89_smart_exit/14_high_q_and_stretch_check.py
```

All three should reproduce the deploy numbers above.

---

## Failed approaches kept for posterity

Scripts 01–07 explored an offline Q-learning exit agent (per spec from Jay).
The math is sound, but the data shows the residual variance after the v84
pipeline isn't predictable from M5+regime+maturity features at exit time.
The right answer was upstream — at entry — not in exit timing.

The smart-exit work is preserved in this folder so future "what about an
offline-RL exit?" questions can be redirected here with concrete disprove
evidence.

---

## Future work

- Periodic retraining cadence (recency lift was 5-9% per script 16 ablation)
- Add maturity features to **confirm** and **meta** heads as well — currently
  only q_entry uses them; downstream gates might gain additional signal
- Cross-asset features (DXY for XAU, BTC futures basis)
- Quarterly walk-forward retrains to track distribution shift
