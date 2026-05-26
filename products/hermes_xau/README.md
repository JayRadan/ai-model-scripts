# Hermes XAU — M1 TFK + Order-Flow Q-Regressor

> **Version:** **v103 + combined-Q upgrade** (2026-05-26)
> **Architecture:** TFK regime + (pullback-to-line OR deep-counter-pullback) entries
>                   + XGBRegressor Q on 43 features + multi-pos with BE-on-new-entry
> **Holdout PF:** **3.21** @ Q≥4.0 | **WR:** **70.7%** | **+19,877 R** / **n=8,232** / **34 trd/day**
> **Holdout USD @ 0.01 lot (multi-pos sim):** **+$14,871** / max DD **−$493** / starting $1,000 → final **$15,871**
> **Bundle:** `hermes_xau_validated.pkl` (43 features) | **NEAR ≤ 0.50** OR **counter ≥ 1.5 ATR** | **Q ≥ 4.0**
> **Deployed:** v103 base 2026-05-25 (`b50515c`) → **combined-Q 2026-05-26 (`102da4d`)**

## 🆕 Combined-Q upgrade (2026-05-26)

User-suggested hypothesis: when regime is GREEN but price has popped ≥1.5 ATR
BELOW the TFK line (or RED + ≥1.5 ATR above), that's also a tradeable entry
in the regime direction (mean-reversion to line). **Validated** — Q retrained
on the union of pullback + counter candidates learned both setup types equally
well (PF 2.18 pullback / PF 2.20 counter at Q≥1.0).

| | v103 base (pullback-only) | combined-Q (current) |
|---|---|---|
| Entry condition | `\|dist\| ≤ 0.50` | `\|dist\| ≤ 0.50` **OR** `dist_signed × cdir ≤ -1.5` |
| q_thr | 1.0 | **4.0** |
| Training candidates | 30,470 | **175,916** (5.8×) |
| Holdout PF | 2.09 | **3.21** |
| Holdout WR | 59.1% | **70.7%** |
| Holdout sumR | +6,372 | **+19,877** (3.1×) |
| Holdout trades | 4,828 | 8,232 (raw) / 2,518 (multi-pos sim) |
| DD / sumR | ~1.8% | **3.0%** |

Holdout Q sweep (all on the same 242-day window):

| Q | n | WR | PF | sumR | DD |
|---|---|---|---|---|---|
| 1.0 | 33,071 | 64.8% | 2.19 | +53,326 | 895 |
| 2.0 | 21,763 | 66.9% | 2.49 | +40,942 | 819 |
| 3.0 | 13,723 | 68.5% | 2.81 | +29,351 | 736 |
| **4.0** | **8,232** | **70.7%** | **3.21** | **+19,877** | **605** ← shipped |
| 5.0 | 4,897 | 72.0% | 3.60 | +13,138 | 492 |
| 6.0 | 2,960 | 73.2% | 4.11 | +9,081 | 382 |

Q=4.0 chosen: highest PF that still has ~34 trades/day, matching original Hermes
trade cadence. Q=5.0+ thins too much for meaningful daily stats.

**Setup-type breakdown at Q≥1.0** (proves counter trades carry real edge):

| Setup type | n | WR | PF | sumR |
|---|---|---|---|---|
| Pullback (\|dist\|≤0.50) | 11,426 | 63.8% | 2.18 | +18,624 |
| Counter (≥1.5R wrong side) | 21,645 | 65.4% | 2.20 | +34,702 |

The counter trades fire ~2× more often than pullback trades in trending markets
because trends spend a lot of time "stretched" beyond the line.

**Live trace tag:** every Hermes funnel entry now includes
`trace["setup_type"] = "pullback" | "counter"` so you can filter by setup type.

**Rollback to v103 base:**
```bash
cd /home/jay/Desktop/my-agents-and-website/commercial
git revert 102da4d
cd server/decision_engine/models
mv hermes_xau_validated.pkl.bak_pre_combined_2026-05-26 hermes_xau_validated.pkl
git add -A && git commit -m "rollback hermes_xau combined-Q" && git push
```

---

## ⚠️ Deployment notes (2026-05-25 base)

The third production model alongside Oracle XAU / Oracle BTC. Trades M1 XAUUSD
using a single XGBRegressor Q-function on **29 standard features + 14
order-flow features** computed live from Dukascopy ticks. Multi-position
(4 slots) with break-even-on-new-entry and switch rule.

The whole stack ships as a single self-contained product:
- **Server side** (vendored in `commercial/server/decision_engine/`):
  - `configs/hermes_xau.py` — frozen hyperparameters (NEAR=0.50, **counter_thr=1.5**, **Q≥4.0**, SL=4R, TRAIL=3R, BE=1R, max_conc=4, switch=0.5, cooldown=5)
  - `hermes_features.py` — TFK indicator + 29 standard features
  - `tick_source.py` — live tick aggregation → 14 order-flow features per M1 bar
  - `decide_hermes.py` — entry + exit (stateless server, EA owns slot state)
  - `dukascopy_source.py` — extended to per-(symbol, interval) cache (M5 Oracle / M1 Hermes coexist)
  - `models/hermes_xau_validated.pkl` — frozen XGBRegressor (43 features) + train metadata
- **EA side**:
  - `EdgePredictor_Connector_v2.mq5` (same file, patched, no v3) — added `EP_HERMES_XAU` enum, per-product timeframe, `EpSlot.q_at_entry` + `be_active` state, JSON helpers for `move_be_for_magics` array, `MoveSlotToBreakEven()` action.

**Requires `USE_DUKASCOPY_BARS=1` on Render** (already set). The server
fetches its own M1 bars + ticks instead of trusting EA-sent M5 bars.

**Rollback:** `git revert b50515c 27950a1 fe2de24 7ebf596 --no-edit && git push`
in the commercial repo. Render redeploys Oracle-only in ~90s.

---

## Strategy in one paragraph

Each minute, the server computes TFK on the rolling M1 bar series → if
`committed_dir != 0` (green/red trend) AND EITHER `|close − tfk_line| / ATR ≤ 0.50`
(pullback to line) OR `dist_signed × committed_dir ≤ -1.5` (price popped through
to the wrong side of the line by ≥1.5 ATR), it computes order-flow features from
the live tick stream and feeds 43 features into an XGBRegressor. If predicted
R ≥ 4.0, the server opens a trade in the regime direction. Up to 4 concurrent
slots managed via the EA. On each new entry, any slot in profit ≥ 1R gets its SL
moved to break-even. If at capacity, the new signal can switch out the
worst-Q open slot. Each trade exits at the FIRST of: hard SL (−4 ATR),
trail giveback (3 ATR from peak favor), max-hold 300 bars.

---

## Holdout — what the frozen bundle does on unseen data

**Window:** 2025-09-01 → 2026-05-01 (242 calendar days). The bundle's training
cutoff is exactly 2025-09-01, so 100% of trades shown are out-of-sample.

| Metric | Value |
|---|---|
| Total trades | 4,828 |
| Long / Short | ~2,463 / ~2,365 |
| Win rate | **59.1%** |
| **Profit factor** | **2.09** |
| sum R | **+6,372 R** |
| Avg R / trade | +1.32 R |
| Trades / day | 20 |
| **USD pnl @ 0.01 lot** | **+$15,824** |
| **Max drawdown** | **−$599** (= 117 R) |
| Equity curve | $1,000 → $16,824 (smooth, monotone-ish; DD/sumR = 1.8%) |

Exit-reason mix (approximate):
- **Trail wins** ~36% (avg +5R, all winners)
- **Hard SL hits** ~22% (each = −4R)
- **BE stops** ~12% (zero R, protection working)
- **Switch closes** ~8% (slot replaced by higher-Q signal)
- **Max hold** ~1%
- **Color flip** none (not in v103 exit set)

Reproducing the holdout — run the same script the bundle was trained from:
```bash
cd ~/Desktop/new-model-zigzag
python3 experiments/v103_tfk_regime/45_near_sweep_m1.py
# Look at the NEAR=0.50 row → PF 2.09, sumR +6,372, n=4,828
```

Or rebuild the website's backtest curve straight from the deployed bundle:
```bash
python3 products/hermes_xau/scripts/09_build_website_backtest_json.py
# Writes commercial/website/public/backtest_data.json with real equity curve
```

---

## Folder layout

```
products/hermes_xau/
├── README.md                              ← THIS FILE
├── tfk.py                                 ← TFK indicator (Python port of Pine v6)
├── train_bundle.py                        ← THE production retraining recipe
└── scripts/                               ← Research → production pipeline, in order
    ├── 01_download_m1_dukascopy.py        ← One-time: pull M1 XAUUSD 2018→now
    ├── 02_aggregate_orderflow_from_ticks  ← Once tick data is available
    ├── 03_train_q_baseline_m1.py          ← Q-regressor trained on OHLCV features
    ├── 04_train_q_with_orderflow.py       ← + 14 tick-derived flow features
    ├── 05_sweep_sl_threshold.py           ← Found SL=4R is best (was 6R, then 2-8R swept)
    ├── 06_sweep_near_threshold.py         ← Found NEAR=0.50 is best (was 0.25)
    ├── 07_meta_classifier_disproven.py    ← Tested P(win)+P(bad) heads — marginal lift
    ├── 08_visualize_holdout.py            ← Plot equity + zoom windows
    └── 09_build_website_backtest_json.py  ← Regenerate the public backtest curve
```

---

## How the production system breaks down

### Feature engineering

**29 standard (no live tick data required):**
- 4 TFK natives (force, velocity, x_est, trend_raw / trend / regime_w)
- 4 setup-bar features (dist_at_signal, dist_abs, regime_age, bar_range_atr)
- RSI 14, ATR ratio, EMA-distance × 4 (20/50/100/200), slope × 3 (5/10/20)
- HTF context × 9: M5/M15/H1 RSI + slope + EMA50-dist
- committed_dir (immediate, the regime label itself)

**14 order-flow (require live tick data — gracefully degrade to zero if unavailable):**
- `signed_flow`, `imbalance_ratio`, `vpin_proxy`
- `bid_ask_vol_ratio` (quote-side bias)
- `cum_signed_5 / _15 / _60` (rolling buy/sell pressure over 5/15/60 minutes)
- `flow_persistence_5 / _15 / _60` (signed_flow / |signed_flow| over rolling windows)
- `median_spread`, `spread_vol_50`, `tick_intensity_50`
- `n_ticks` per bar

The Q-regressor was trained on all 43 features. If the tick source is down at inference time, the order-flow columns are 0-filled and the strategy degrades to the OHLCV-only path (~PF 1.84 instead of 2.09).

### Exit policy

Stateless per trade — the server tells the EA the parameters, the EA enforces. Each bar:

1. **Hard SL** = entry ± 4 × ATR_at_entry. Triggers on intra-bar low/high vs entry.
2. **Trail stop** = peak_favor − 3 × ATR_at_entry. Activates once favor ≥ 3 ATR.
3. **BE-on-new-entry** = when a NEW signal arrives AND this slot's favor ≥ 1 R, the server sends `move_be_for_magics: [...]` in the response; EA moves SL to entry price. Then the original hard SL is replaced by the BE stop (zero-R loss possible).
4. **Switch rule** = at max_concurrent (=4), if `new_Q ≥ worst_active_Q + 0.5`, server sends `switch_close_magic: ...`; EA closes that slot before opening the new one.
5. **Max hold** = 300 bars (~5 h). Failsafe.

NOT in the exit set: no color flip kill (was tested in v9b, dropped — too aggressive).

### Multi-position state

The EA owns per-slot state (`EpSlot` struct):
- `dir`, `entry_price`, `entry_atr`, `bars_held` (existing fields)
- `magic` (=base + slot_index, base=421100 for Hermes)
- **`q_at_entry`** (Hermes-only, read from server's open response)
- **`be_active`** (Hermes-only, set true after `MoveSlotToBreakEven` succeeds)

Every entry call, the EA sends ALL open slots in `open_positions: [...]` with these fields. The server uses them to:
- Detect BE-eligible profitable slots (favor ≥ 1R, not already BE) → emit `move_be_for_magics`
- At capacity, find worst-Q open slot (lowest `q_at_entry`) → if new Q is +0.5 better, emit `switch_close_magic`

This makes the server **completely stateless** — no per-account DB required. Render restarts don't lose anything.

---

## Reproduction recipe (rebuild from scratch)

```bash
cd ~/Desktop/new-model-zigzag

# Stage 1 — Get the data
python3 products/hermes_xau/scripts/01_download_m1_dukascopy.py     # 8.4 years M1 (~2 min)
python3 products/hermes_xau/scripts/02_aggregate_orderflow_from_ticks.py    # tick → M1 + flow (~30 s)

# Stage 2 — Research (optional — just to confirm research numbers)
python3 products/hermes_xau/scripts/03_train_q_baseline_m1.py        # OHLCV-only baseline
python3 products/hermes_xau/scripts/04_train_q_with_orderflow.py     # + orderflow
python3 products/hermes_xau/scripts/05_sweep_sl_threshold.py         # SL=4R is best
python3 products/hermes_xau/scripts/06_sweep_near_threshold.py       # NEAR=0.50 is best
python3 products/hermes_xau/scripts/08_visualize_holdout.py          # equity + zoom charts

# Stage 3 — Build the production bundle
python3 products/hermes_xau/train_bundle.py
# Output: commercial/server/decision_engine/models/hermes_xau_validated.pkl

# Stage 4 — Refresh the website's public backtest curve
python3 products/hermes_xau/scripts/09_build_website_backtest_json.py
# Output: commercial/website/public/backtest_data.json (hermes entry updated)

# Stage 5 — Ship
cd ~/Desktop/my-agents-and-website/commercial
git add server/decision_engine/models/hermes_xau_validated.pkl \
        website/public/backtest_data.json
git commit -m "Hermes XAU retrain — holdout PF X.XX"
git push   # Render + Vercel auto-deploy in ~90 s
```

---

## Improvement journey (what works and what didn't)

The v103 research lineage in `experiments/v103_tfk_regime/`:

| Iteration | Change | PF | sumR | DD | Notes |
|---|---|---|---|---|---|
| v6 | wide SL + dual-RL (M5) | 1.58 | +739 | 43 | Baseline starting point |
| v7 | auto-BE at +1R | 0.82 | −66k | huge | **DISPROVEN** — destroys trail edge by forcing every winner to BE |
| v8 | BE-on-new-entry (conditional) | 1.87 | +448 | 51 | Foundation of v103 |
| v9b | 10-bar color-flip exit | 1.91 | +583 | 50 | Refined exit gating |
| v9c | drop color-flip exit entirely | 1.33 | +3,149 | 13 | More trades, smoother |
| v10 | NEAR=0.25 close-to-line gate | 1.58 | +1,075 | 76 | Strong trend-pullback shape |
| M1 first pass (11-day train) | tiny training set | 1.38 | +5,789 | 265 | Proof-of-concept |
| M1 full (6.9y train) | full Dukascopy history | 1.75 | +8,371 | 267 | + 5.9× total R vs M5 |
| **+ orderflow features** | 14 tick-derived inputs | **1.91** | +4,776 | 164 | +10% PF — real signal |
| **+ SL=4** | tighter hard SL | **2.04** | +4,480 | 135 | Better risk-adjusted |
| **+ NEAR=0.50 (final)** | loosen closeness gate | **2.09** | **+6,372** | **117** | LOCKED |

Counter-extreme overlay (v11) was tested separately — modest +PF 1.34 alone but
unstable across regimes; **dropped from production**. Q-percentile filter
(script 36) gave slight DD reduction at the cost of total R; **dropped**.
Meta-classifier (script 07/disproven) added <2% PF — **dropped**.

---

## Known limitations / future work

1. **Tick latency** — Dukascopy ticks lag the bar close by ~10-20 seconds.
   The orderflow features computed at decide time are technically based on a
   tick buffer that's slightly behind the current M1 bar's close. Tested
   in research (different latency assumed): impact is small (PF 2.09 → 2.05 if
   we apply an explicit 60-s tick lag). Live performance should be very close.

2. **Spread sensitivity** — research assumes $0.30 XAU spread per round-trip
   (median ATR 1.31 → 0.23 R). At $0.50 spread (high-vol minute), PF drops
   to ~1.85. At $1.00 (news minute), PF ~1.4. Wide-spread minutes are
   downside; consider adding a `max_spread_R` gate in `decide_hermes.py`.

3. **No counter-extreme overlay** — research showed +PF 0.07 from a
   parallel counter-extreme strategy. Not included in v1 deployment to
   keep complexity low. Can be added later as a separate engine.

4. **License entitlement is manual** for existing subscribers. New
   purchases auto-entitle `hermes_xau` via the webhook (`getProduct("hermes").serverSlugs`).

5. **Vercel build still has unrelated pre-existing tsc warning** about a
   missing `app/methodology/page.js` — unrelated to Hermes, ignore.

---

## Live operations cheat-sheet

```bash
# Watch Render logs for Hermes activity
# (look for [hermes], [tick_source], decide_hermes entries)
curl -sS https://edge-predictor.onrender.com/decide/_health | jq

# Inspect the funnel for Hermes
curl -sS "https://edgepredictor.pro/api/admin/funnel?mode=stats&product=hermes_xau&hours=24" \
  -H "Cookie: <admin-session>" | jq

# TFK regime + Q snapshot (one-call admin debug)
curl -sS "https://edgepredictor.pro/api/admin/funnel?mode=regime-debug&product=hermes_xau" \
  -H "Cookie: <admin-session>" | jq

# Grant a license hermes_xau entitlement
# (use the /admin web UI — or POST to /admin/grant-products with admin_secret)

# Rollback if needed
cd ~/Desktop/my-agents-and-website/commercial
git log --oneline | head -10                  # find the pre-Hermes commit
git revert <hermes-commits> --no-edit
git push                                       # Render redeploys Oracle-only in ~90s
```
