"""
v76 step 02: apply the analog signal as a veto on Oracle/Midas holdout
trades. Two decision rules tested:

  rule A (pct-based): block BUY if pct_neg_4h ≥ T_PCT_NEG
                      block SELL if pct_pos_4h ≥ T_PCT_POS
  rule B (mean-based): block BUY if mean_fwd_4h < -T_MEAN
                       block SELL if mean_fwd_4h > +T_MEAN

For each rule, sweep thresholds and report PF / total R / DD vs baseline.
Also compute the FORWARD-ALIGNMENT TEST: is there ANY rule + threshold
where the veto improves PF? If not — analogs aren't predictive.
"""
from __future__ import annotations
import os, sys
import numpy as np, pandas as pd

ZIGZAG = "/home/jay/Desktop/new-model-zigzag"
sys.path.insert(0, os.path.join(ZIGZAG, "model_pipeline"))
import paths as P

EXP = os.path.join(ZIGZAG, "experiments/v76_knn_analogs")
ORACLE_HOLDOUT = P.data("v72l_trades_holdout.csv")
MIDAS_HOLDOUT  = P.data("v6_trades_holdout_xau.csv")


def stats(df):
    if not len(df): return ("n=0", 0, 0, 0, 0)
    ww = df[df.pnl_R > 0]; ll = df[df.pnl_R <= 0]
    pf = (ww.pnl_R.sum() / -ll.pnl_R.sum()) if len(ll) and ll.pnl_R.sum() < 0 else float("inf")
    cum = df.pnl_R.cumsum().values
    dd = float((cum - np.maximum.accumulate(cum)).min())
    return (f"n={len(df):>4d}  WR={len(ww)/len(df)*100:>5.1f}%  "
            f"PF={pf:>5.2f}  R={df.pnl_R.sum():+8.1f}  DD={dd:+7.1f}",
            len(df), pf, df.pnl_R.sum(), dd)


def apply(trades, sig, label):
    trades = trades.copy()
    if "pnl_R" in trades.columns and trades["pnl_R"].abs().mean() > 2:
        trades["pnl_R"] = trades["pnl_R"] / 4.0
    trades = trades.merge(sig, on="time", how="left")
    trades["dir_str"] = np.where(trades["direction"] > 0, "BUY", "SELL")
    base_str, base_n, base_pf, base_R, base_DD = stats(trades)
    print(f"\n=== {label} ===  baseline: {base_str}")

    print(f"  rule A — pct-based veto (block BUY if pct_neg ≥ T, SELL if pct_pos ≥ T):")
    for T in [0.40, 0.45, 0.50, 0.55]:
        veto = ((trades["dir_str"] == "BUY")  & (trades["pct_neg_4h"] >= T)) | \
                ((trades["dir_str"] == "SELL") & (trades["pct_pos_4h"] >= T))
        keep = trades[~veto.fillna(False)]
        s, n, pf, R, dd = stats(keep)
        delta_pf = pf - base_pf
        flag = "✓" if R > base_R and pf >= base_pf - 0.05 else "✗"
        print(f"    T={T:.2f}  blocked={veto.fillna(False).sum():>3d}  "
              f"{s}  ΔPF={delta_pf:+.2f}  ΔR={R-base_R:+.1f}  {flag}")

    print(f"  rule B — mean-based veto (block BUY if mean_fwd_4h < -M, SELL if > +M):")
    for M in [0.001, 0.0015, 0.002, 0.003, 0.005]:
        veto = ((trades["dir_str"] == "BUY")  & (trades["mean_fwd_4h"] < -M)) | \
                ((trades["dir_str"] == "SELL") & (trades["mean_fwd_4h"] >  M))
        keep = trades[~veto.fillna(False)]
        s, n, pf, R, dd = stats(keep)
        delta_pf = pf - base_pf
        flag = "✓" if R > base_R and pf >= base_pf - 0.05 else "✗"
        print(f"    M={M:.4f}  blocked={veto.fillna(False).sum():>3d}  "
              f"{s}  ΔPF={delta_pf:+.2f}  ΔR={R-base_R:+.1f}  {flag}")


def main():
    sig = pd.read_parquet(os.path.join(EXP, "data/analog_signals.parquet"))
    sig["time"] = pd.to_datetime(sig["time"])
    print(f"Loaded {len(sig):,} analog signals")

    o = pd.read_csv(ORACLE_HOLDOUT, parse_dates=["time"])
    m = pd.read_csv(MIDAS_HOLDOUT,  parse_dates=["time"])
    apply(o, sig, "Oracle XAU")
    apply(m, sig, "Midas XAU")

    # Headline diagnostic: does the analog signal predict the trade outcome
    # AT ALL?  Compute correlation between mean_fwd_4h and the trade's
    # direction-adjusted pnl_R.
    print("\n=== sanity: correlation between analog signal and trade outcome ===")
    for label, csv in [("Oracle", ORACLE_HOLDOUT), ("Midas", MIDAS_HOLDOUT)]:
        d = pd.read_csv(csv, parse_dates=["time"])
        if d["pnl_R"].abs().mean() > 2: d["pnl_R"] = d["pnl_R"] / 4.0
        d = d.merge(sig, on="time", how="left").dropna(subset=["mean_fwd_4h"])
        d["dir_signed_signal"] = d["mean_fwd_4h"] * d["direction"]
        # Positive = signal AGREED with trade direction
        c = np.corrcoef(d["dir_signed_signal"].values, d["pnl_R"].values)[0,1]
        print(f"  {label}: corr(signal*direction, pnl_R) = {c:+.4f}  ← needs to be >+0.05 to be useful")


if __name__ == "__main__":
    main()
