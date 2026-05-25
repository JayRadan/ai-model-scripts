"""Visualize XAU M1 v10 trades."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

from tfk import compute_tfk

CUTOFF = pd.Timestamp("2024-12-12 00:00:00")


def main():
    print("loading M1 + trades ...", flush=True)
    m1 = pd.read_parquet(ROOT / "data" / "m1_xau_full.parquet")
    m1 = m1.sort_values("time").reset_index(drop=True)

    # Compute TFK for line + color
    tfk_out = compute_tfk(m1)
    O = tfk_out["open"].to_numpy()
    H = tfk_out["high"].to_numpy()
    L = tfk_out["low"].to_numpy()
    C = tfk_out["close"].to_numpy()
    line = tfk_out["tfk_line"].to_numpy()
    cdir = tfk_out["committed_dir"].to_numpy()
    times = tfk_out["time"].to_numpy()

    trades = pd.read_csv(HERE / "trade_charts" / "xau_m1_full_q1.0_trades.csv",
                         parse_dates=["time"])
    trades["entry_idx"] = trades["entry_idx"].astype(int)
    trades["exit_idx"] = trades["exit_idx"].astype(int)
    trades["direction"] = trades["direction"].astype(int)
    print(f"  {len(trades):,} M1 trades  range: {trades.time.min()} → {trades.time.max()}", flush=True)

    out_dir = HERE / "trade_charts"; out_dir.mkdir(exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Equity + R distribution
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={"height_ratios": [3, 2]})
    s = trades.sort_values("time").reset_index(drop=True)
    eq = np.cumsum(s["pnl_R"].to_numpy())
    a0.plot(s["time"], eq, lw=1.0, color="#00C896")
    a0.fill_between(s["time"], 0, eq, alpha=0.12, color="#00C896")
    a0.axhline(0, color="black", lw=0.5)
    wins = trades[trades.pnl_R > 0]; losses = trades[trades.pnl_R <= 0]
    pf = wins.pnl_R.sum() / max(-losses.pnl_R.sum(), 1e-9)
    eq_arr = eq; peak = np.maximum.accumulate(eq_arr); dd = (peak - eq_arr).max()
    a0.set_title(f"XAU M1 FULL — Q≥1.0 with $0.30 spread — n={len(trades)} "
                 f"PF={pf:.2f} sumR={trades.pnl_R.sum():+.0f}R DD={dd:.0f}R WR={(trades.pnl_R>0).mean()*100:.1f}%",
                 fontsize=12)
    a0.set_ylabel("cum R"); a0.grid(alpha=0.2)
    bins = np.arange(-8, 12, 0.4)
    a1.hist(wins["pnl_R"], bins=bins, alpha=0.7, color="#00C896", label=f"wins n={len(wins)}")
    a1.hist(losses["pnl_R"], bins=bins, alpha=0.7, color="#FF3B69", label=f"losses n={len(losses)}")
    a1.axvline(0, color="black", lw=0.5)
    a1.set_xlabel("trade pnl_R (after spread)"); a1.set_ylabel("count")
    a1.legend(); a1.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_dir / "xau_m1_full_equity.png", dpi=110)
    plt.close()
    print(f"  wrote xau_m1_full_equity.png", flush=True)

    # Zoom windows — pick 4 clusters (M1 windows = ~4 hours = 240 bars)
    window_bars = 240
    trades_sorted = trades.sort_values("entry_idx").reset_index(drop=True)
    densities = []
    for i in range(len(trades_sorted)):
        ei = trades_sorted.iloc[i]["entry_idx"]
        cnt = int(((trades_sorted["entry_idx"] >= ei - window_bars // 2) &
                   (trades_sorted["entry_idx"] <= ei + window_bars // 2)).sum())
        densities.append((cnt, ei))
    densities.sort(reverse=True)
    selected_centers = []
    for cnt, ei in densities:
        if not any(abs(ei - c) < window_bars for c in selected_centers):
            selected_centers.append(ei)
        if len(selected_centers) >= 4: break
    selected_centers.sort()

    for w_idx, center in enumerate(selected_centers):
        start = max(0, int(center) - window_bars // 2)
        end = min(len(tfk_out), int(center) + window_bars // 2)
        local_times = times[start:end]
        in_win = trades[(trades.entry_idx >= start) & (trades.entry_idx < end)]
        fig, ax = plt.subplots(figsize=(20, 9))
        ax.plot(local_times, C[start:end], color="#444", lw=0.7)
        local_color = cdir[start:end]; local_line = line[start:end]
        seg = 0
        for i in range(1, len(local_color) + 1):
            if i == len(local_color) or local_color[i] != local_color[seg]:
                col = "#00C896" if local_color[seg] > 0 else "#FF3B69"
                ax.plot(local_times[seg:i], local_line[seg:i], color=col, lw=2.2, alpha=0.85)
                seg = i
        for r in in_win.itertuples():
            ei = r.entry_idx; xi = r.exit_idx; d = r.direction
            col = "#00C896" if r.pnl_R > 0 else "#FF3B69"
            mk = "^" if d == 1 else "v"
            ax.scatter(times[ei], O[ei], marker=mk, s=140, color=col,
                       edgecolor="black", linewidth=1.0, zorder=5)
            ax.scatter(times[xi], C[xi], marker="x", s=90, color=col, linewidth=2, zorder=5)
            ax.plot([times[ei], times[xi]], [O[ei], C[xi]], color=col, lw=0.8, alpha=0.5)
            ax.text(times[xi], C[xi], f" {r.pnl_R:+.1f}R", fontsize=7, color=col, va="center")
        from matplotlib.lines import Line2D
        legend = [
            Line2D([0], [0], color="#00C896", lw=2.2, label="TFK green"),
            Line2D([0], [0], color="#FF3B69", lw=2.2, label="TFK red"),
            Line2D([0], [0], marker="^", color="w", markerfacecolor="#00C896",
                   markeredgecolor="black", markersize=12, label="long (win)", lw=0),
            Line2D([0], [0], marker="v", color="w", markerfacecolor="#FF3B69",
                   markeredgecolor="black", markersize=12, label="short (loss)", lw=0),
            Line2D([0], [0], marker="x", color="black", markersize=10, lw=0, label="exit"),
        ]
        ax.legend(handles=legend, loc="upper left", fontsize=9)
        ww = in_win[in_win.pnl_R > 0]; ll = in_win[in_win.pnl_R <= 0]
        ax.set_title(f"XAU M1 — cluster {w_idx+1}/4 — {local_times[0]} to {local_times[-1]}\n"
                     f"n={len(in_win)} wins={len(ww)} losses={len(ll)} sumR={in_win.pnl_R.sum():+.1f}R",
                     fontsize=11)
        ax.set_ylabel("price"); ax.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(out_dir / f"xau_m1_full_cluster_{w_idx+1}.png", dpi=110)
        plt.close()
        print(f"  wrote cluster_{w_idx+1}.png  ({len(in_win)} trades, {in_win.pnl_R.sum():+.1f}R)", flush=True)


if __name__ == "__main__":
    main()
