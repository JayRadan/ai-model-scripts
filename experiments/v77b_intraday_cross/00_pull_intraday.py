"""
Pull 1h cross-instrument data — yfinance limit: last 730 days.
That covers ~2024-05-01 → today, which fully spans our holdout
(2024-12-12 → 2026-04-13). Training signal pre-holdout will be
short (~7 months) but we have 2200 Midas trades in the holdout
to evaluate against.

Symbols: DXY (DX=F futures — more reliable on yfinance 1h than DX-Y.NYB),
         SPX (ES=F mini-futures), TNX (^TNX), VIX (^VIX).
"""
from __future__ import annotations
import os, sys
import pandas as pd
import yfinance as yf

OUT = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT, exist_ok=True)

# Try futures-symbol fallbacks for DXY/SPX since yfinance often refuses
# the cash index at 1h.
SYMBOLS = [
    ("EURUSD=X", "eurusd_1h.parquet"),  # primary DXY proxy for gold
    ("ES=F",     "spx_1h.parquet"),
    ("^TNX",     "tnx_1h.parquet"),
    ("^VIX",     "vix_1h.parquet"),
    ("UUP",      "uup_1h.parquet"),     # DXY ETF — secondary proxy
]
START = "2024-05-05"   # ~730 days back from 2026-05-01
END   = "2026-05-01"


def pull(sym, fname):
    print(f"[{sym}]…", flush=True)
    df = yf.download(sym, start=START, end=END, interval="1h",
                      auto_adjust=False, progress=False)
    if df is None or df.empty:
        print(f"  ⚠ empty"); return
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index().rename(columns={"Datetime":"time","Date":"time"})
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert(None)
    keep = [c for c in ["time","Open","High","Low","Close","Volume"] if c in df.columns]
    df = df[keep].rename(columns=str.lower)
    out = os.path.join(OUT, fname)
    df.to_parquet(out, index=False)
    print(f"  → {out} ({len(df):,} rows, {df.time.min()} → {df.time.max()})")


if __name__ == "__main__":
    for sym, fname in SYMBOLS:
        try: pull(sym, fname)
        except Exception as e: print(f"  ✗ {sym}: {e}", file=sys.stderr)
