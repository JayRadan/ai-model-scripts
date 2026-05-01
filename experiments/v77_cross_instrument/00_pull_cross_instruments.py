"""
Pull cross-instrument hourly data: DXY, SPX, TNX (US10Y yield), VIX.

Why these four:
  DXY  — dollar strength; gold is inversely tied
  SPX  — risk-on / risk-off; flight-to-safety bid for gold
  TNX  — real yields competing with non-yielding gold
  VIX  — fear; gold is a fear hedge

yfinance limits: 1h goes back ~730 days. That's enough — our holdout
window started 2024-12-12, and we want training data 2-3 years before
that. We pull from 2022-06-01 onward.

Output: experiments/v77_cross_instrument/data/{dxy,spx,tnx,vix}_1h.parquet
"""
from __future__ import annotations
import os, sys
import pandas as pd
import yfinance as yf

OUT_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT_DIR, exist_ok=True)

# Symbol → output filename
SYMBOLS = {
    "DX-Y.NYB": "dxy_1d.parquet",   # Dollar Index (NYBOT)
    "^GSPC":    "spx_1d.parquet",   # S&P 500
    "^TNX":     "tnx_1d.parquet",   # 10Y Treasury yield (×10)
    "^VIX":     "vix_1d.parquet",   # CBOE Volatility Index
}

# Daily resolution — cross-instrument regime is slow-moving, and daily
# gives us full history (vs 730-day cap on 1h). XAU bars get
# forward-filled with the latest *closed* daily bar.
START = "2018-01-01"
END   = "2026-05-01"


def pull(symbol: str, fname: str) -> None:
    print(f"[{symbol}] downloading…", flush=True)
    df = yf.download(symbol, start=START, end=END, interval="1d",
                      auto_adjust=False, progress=False)
    if df is None or df.empty:
        print(f"  ⚠ empty for {symbol}")
        return
    # yfinance sometimes returns a column MultiIndex when only one symbol —
    # flatten it.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index().rename(columns={"Datetime": "time", "Date": "time"})
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert(None)
    keep = [c for c in ["time", "Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].rename(columns=str.lower)
    out = os.path.join(OUT_DIR, fname)
    df.to_parquet(out, index=False)
    print(f"  → {out}  ({len(df):,} rows, {df['time'].min()} → {df['time'].max()})")


def main() -> None:
    for sym, fname in SYMBOLS.items():
        try:
            pull(sym, fname)
        except Exception as e:
            print(f"  ✗ {sym}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
