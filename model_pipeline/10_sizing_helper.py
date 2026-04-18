"""
Position sizing helper.

Given account size and max acceptable drawdown %, computes:
  - the lot size that scales the historical max drawdown to fit
  - expected daily, weekly, monthly, yearly $ + % return
  - the realistic "live degraded" version (PF dropped 15%)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import paths as P

# Holdout reference at 0.01 lot (from backtest_v6_summary.json)
with open(P.data("backtest_v6_summary.json")) as f:
    s = json.load(f)

REF_LOT          = 0.01
REF_PNL_TOTAL    = s["total_pnl"]            # $ at REF_LOT
REF_MAX_DD       = abs(s["max_dd"])          # $ at REF_LOT
REF_CALENDAR_DAYS= s["calendar_days"]
REF_ACTIVE_DAYS  = s["active_days"]
REF_DAILY_PNL    = REF_PNL_TOTAL / REF_CALENDAR_DAYS
REF_WEEKLY_PNL   = REF_DAILY_PNL * 7
REF_MONTHLY_PNL  = REF_DAILY_PNL * 30
REF_YEARLY_PNL   = REF_DAILY_PNL * 365


def size_for(account: float, max_dd_pct: float, live_degradation: float = 0.15):
    """
    account       : USD
    max_dd_pct    : 0..1, e.g. 0.15 for 15% max drawdown tolerance
    live_degradation : haircut applied to all PnL (default 15%)
    """
    # Scale: choose lot so that REF_MAX_DD scaled equals account * max_dd_pct
    target_dd = account * max_dd_pct
    scale = target_dd / REF_MAX_DD
    lot = REF_LOT * scale

    pnl_factor = scale * (1 - live_degradation)
    daily   = REF_DAILY_PNL  * pnl_factor
    weekly  = REF_WEEKLY_PNL * pnl_factor
    monthly = REF_MONTHLY_PNL* pnl_factor
    yearly  = REF_YEARLY_PNL * pnl_factor

    return {
        "account": account,
        "max_dd_pct_target": max_dd_pct,
        "max_dd_dollars": target_dd,
        "lot_size": lot,
        "live_degradation_applied": live_degradation,
        "daily_$": daily,
        "weekly_$": weekly,
        "monthly_$": monthly,
        "yearly_$": yearly,
        "daily_%": daily / account * 100,
        "weekly_%": weekly / account * 100,
        "monthly_%": monthly / account * 100,
        "yearly_%": yearly / account * 100,
    }


def main():
    print("="*72)
    print("POSITION SIZING HELPER  (based on v6 honest holdout)")
    print("="*72)
    print(f"  Reference run @ lot=0.01")
    print(f"  Total PnL    : ${REF_PNL_TOTAL:+.2f}")
    print(f"  Max drawdown : ${REF_MAX_DD:.2f}")
    print(f"  Span         : {REF_CALENDAR_DAYS} calendar days, {REF_ACTIVE_DAYS} active")
    print(f"  Daily $      : ${REF_DAILY_PNL:+.4f}")
    print(f"  Weekly $     : ${REF_WEEKLY_PNL:+.4f}")
    print()
    print("  Assumptions for live projection:")
    print("  - 15% PnL haircut for live execution friction")
    print("  - Lot scales linearly with account")
    print("  - Drawdown scales linearly too")
    print()

    # Show table for several account sizes at 15% max DD tolerance
    print(f"  ── Suggested sizing @ 15% max drawdown tolerance ──")
    print(f"  {'account':>10}{'lot':>9}{'daily $':>12}{'weekly $':>12}{'monthly $':>12}{'yearly $':>12}{'daily %':>9}{'monthly %':>11}")
    for acct in [1000, 2500, 5000, 10000, 25000, 50000, 100000]:
        r = size_for(acct, 0.15)
        print(f"  ${acct:>9,}{r['lot_size']:>9.3f}"
              f"{r['daily_$']:>+12.2f}{r['weekly_$']:>+12.2f}"
              f"{r['monthly_$']:>+12.2f}{r['yearly_$']:>+12.2f}"
              f"{r['daily_%']:>+8.3f}%{r['monthly_%']:>+10.2f}%")

    print()
    print(f"  ── Same accounts, different DD tolerances ──")
    print(f"  {'account':>10}{'DD%':>8}{'lot':>9}{'monthly $':>12}{'monthly %':>12}{'yearly %':>11}")
    for acct in [1000, 5000, 10000, 25000]:
        for dd in [0.05, 0.10, 0.20, 0.30]:
            r = size_for(acct, dd)
            print(f"  ${acct:>9,}{dd*100:>7.0f}%{r['lot_size']:>9.3f}"
                  f"{r['monthly_$']:>+12.2f}{r['monthly_%']:>+11.2f}%{r['yearly_%']:>+10.1f}%")

    # Explicit answer to "can I get 1-2% daily?"
    print()
    print("="*72)
    print("REALITY CHECK — to get the daily return targets you asked for:")
    print("="*72)
    for target_daily_pct in [0.10, 0.20, 0.50, 1.00, 2.00]:
        # daily $ needed = account * target/100
        # we need daily_$_per_lot * required_lot = target_daily_$
        # daily_$_per_lot at 0.01 lot, with degradation:
        daily_per_001_live = REF_DAILY_PNL * (1 - 0.15)
        # example for $10k account:
        acct = 10_000
        target_daily_dollar = acct * target_daily_pct / 100
        required_lot = target_daily_dollar / daily_per_001_live * 0.01
        # implied DD at that lot:
        scale = required_lot / 0.01
        implied_dd = REF_MAX_DD * scale
        implied_dd_pct = implied_dd / acct * 100
        print(f"  {target_daily_pct:>5.2f}% daily on ${acct:,} → "
              f"need lot={required_lot:.2f}, "
              f"implied max DD = ${implied_dd:,.0f} ({implied_dd_pct:.0f}% of account)")

    # Save full table
    out = {
        "reference": {
            "lot": REF_LOT,
            "total_pnl_dollars": REF_PNL_TOTAL,
            "max_dd_dollars": REF_MAX_DD,
            "calendar_days": REF_CALENDAR_DAYS,
            "active_days": REF_ACTIVE_DAYS,
            "daily_pnl": REF_DAILY_PNL,
            "weekly_pnl": REF_WEEKLY_PNL,
        },
        "sizing_table": {str(a): size_for(a, 0.15) for a in
                         [1000, 2500, 5000, 10000, 25000, 50000, 100000]},
    }
    with open(P.data("sizing_helper.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved: {P.data('sizing_helper.json')}")


if __name__ == "__main__":
    main()
