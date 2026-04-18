"""Synthetic sanity tests for BOCPD, Kyle's λ, HAR-RV ratio, Hawkes η."""
import numpy as np, sys
sys.path.insert(0, "/home/jay/Desktop/new-model-zigzag/experiments/v71_bocpd_kyle_har_hawkes")
from compute_v71_features import (compute_bocpd, compute_kyle_lambda,
                                   compute_har_rv_ratio, compute_hawkes_eta)

rng = np.random.default_rng(42)


# ---- Test 1: BOCPD — mass-on-recent-r spikes after a mean shift
print("TEST 1 — BOCPD on synthetic step change (recent-r mass)")
seg1 = rng.normal(0.0, 1.0, 1000)       # baseline
seg2 = rng.normal(5.0, 1.0, 200)        # mean shifts by +5σ at t=1000
seg3 = rng.normal(0.0, 1.0, 200)        # back to baseline at t=1200
x = np.concatenate([seg1, seg2, seg3])
cp = compute_bocpd(x, r_max=200, step=1)
# Recent-r mass should be near 0 during steady state (run length has grown),
# then spike toward 1 shortly after the shift
print(f"  Recent-r mass at 999 (steady):   {cp[999]:.4f}  (expect near 0)")
print(f"  Recent-r mass at 1005 (post-shift): {cp[1005]:.4f}  (expect HIGH)")
print(f"  Recent-r mass at 1020 (post-shift): {cp[1020]:.4f}  (expect HIGH)")
print(f"  Mean during 500–900 (steady):    {np.nanmean(cp[500:900]):.4f}")
assert cp[1020] > np.nanmean(cp[500:900]) + 0.2, f"FAIL: no BOCPD spike after shift ({cp[1020]:.4f} vs steady {np.nanmean(cp[500:900]):.4f})"
print("  ✓ BOCPD recent-r mass rises after mean shift")

# Past-only verification: corrupt future, past must be identical
x2 = x.copy()
x2[1300:] = rng.normal(100.0, 10.0, len(x) - 1300)
cp2 = compute_bocpd(x2, r_max=200, step=1)
max_diff_past = np.abs(cp[:1300] - cp2[:1300])
max_d = np.nanmax(max_diff_past)
print(f"  Max |diff| on past bars (corrupt future): {max_d:.2e}")
assert max_d < 1e-12, f"FAIL: BOCPD not strict past-only, max diff = {max_d}"
print("  ✓ BOCPD strictly past-only")


# ---- Test 2: Kyle's λ — thin market (small range, big moves) vs liquid
print("\nTEST 2 — Kyle's λ")
# Thin: big price moves relative to range
c_thin = 100 + np.cumsum(rng.normal(0, 0.5, 1000))
h_thin = c_thin + 0.1  # tiny range
l_thin = c_thin - 0.1
lam_thin = compute_kyle_lambda(c_thin, h_thin, l_thin, window=60)
# Liquid: same price path but big range
h_liq = c_thin + 2.0
l_liq = c_thin - 2.0
lam_liq = compute_kyle_lambda(c_thin, h_liq, l_liq, window=60)
m_thin = np.nanmean(lam_thin)
m_liq = np.nanmean(lam_liq)
print(f"  Thin-market λ mean: {m_thin:.4f}  (expect HIGH)")
print(f"  Liquid-market λ mean: {m_liq:.4f}  (expect LOW)")
assert m_thin > m_liq * 3, "FAIL: thin-market λ should be much higher than liquid"
print("  ✓ λ correctly higher in thin market")


# ---- Test 3: HAR-RV ratio — quiet month → noisy day → ratio >> 1
print("\nTEST 3 — HAR-RV ratio")
# Quiet prefix (very low vol) then recent noisy (high vol)
c_quiet = 1000 + np.cumsum(rng.normal(0, 0.01, 9000))
c_noisy = c_quiet[-1] + np.cumsum(rng.normal(0, 1.0, 400))
c = np.concatenate([c_quiet, c_noisy])
ratio = compute_har_rv_ratio(c, short_bars=288, long_bars=8640)
# Mean over last 200 bars should be well above 1 (short-vol spike)
late_mean = np.nanmean(ratio[-200:])
print(f"  HAR-RV ratio in recent noisy period: {late_mean:.2f}  (expect >> 1)")
assert late_mean > 2.0, f"FAIL: expected elevated ratio, got {late_mean:.2f}"
print("  ✓ HAR-RV ratio spikes when short-term vol elevated")


# ---- Test 4: Hawkes η — bursty event series → η > 1
print("\nTEST 4 — Hawkes η (event clustering)")
# Baseline: Gaussian noise → a few random events
base = rng.normal(0, 1.0, 3000)
# Then inject a cluster: 5 large shocks within 40 bars
cluster_idx = 2500 + rng.integers(0, 40, 5)
for idx in cluster_idx:
    base[idx] += 8.0 * np.sign(rng.normal())
# Cumulate into a price
c_bursty = 1000 * np.exp(np.cumsum(base * 0.001))
eta = compute_hawkes_eta(c_bursty, short_window=60, long_window=600)
# Just after cluster, short-window rate >> long-window rate
peak_eta = np.nanmax(eta[2540:2600])
base_eta = np.nanmean(eta[1000:2400])
print(f"  Baseline η before cluster:  {base_eta:.3f}")
print(f"  Peak η after cluster:       {peak_eta:.3f}")
assert peak_eta > max(base_eta * 1.5, 1.5), f"FAIL: η should spike post-cluster ({peak_eta:.2f} vs {base_eta:.2f})"
print("  ✓ Hawkes η detects event clustering")


print("\nALL SANITY TESTS PASSED ✓")
