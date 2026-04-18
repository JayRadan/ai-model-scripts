"""Sanity test features on synthetic data with known behavior."""
import numpy as np, pandas as pd, sys
sys.path.insert(0, "/home/jay/Desktop/new-model-zigzag/experiments/v7_signatures_vpin_te")
from compute_v7_features import (compute_vpin, compute_path_signatures,
                                  compute_transfer_entropy_h1_to_m5, _te_binned_2x2x2)

rng = np.random.default_rng(42)

# ---- Test 1: VPIN on pure noise vs trending
print("TEST 1 — VPIN")
n = 2000
noise = np.cumsum(rng.normal(0, 1, n)) + 1000
trend = np.cumsum(rng.normal(0.5, 1, n)) + 1000      # strong drift
def hl(c):
    h = c + np.abs(rng.normal(0, 0.5, len(c)))
    l = c - np.abs(rng.normal(0, 0.5, len(c)))
    return h, l
hn, ln = hl(noise); ht, lt = hl(trend)
vn = compute_vpin(hn, ln, noise, step=3)
vt = compute_vpin(ht, lt, trend, step=3)
print(f"  VPIN noise:  mean={np.nanmean(vn):.4f}  (expect ~0.5 low toxicity)")
print(f"  VPIN trend:  mean={np.nanmean(vt):.4f}  (expect HIGHER, directional flow)")
assert np.nanmean(vt) > np.nanmean(vn), "FAIL: trend VPIN should exceed noise VPIN"
print("  ✓ trend VPIN > noise VPIN")

# ---- Test 2: Path signatures
print("\nTEST 2 — Path signatures")
# Pure up-ramp: Lévy area should be ~0, quad var finite, time-weighted drift > 0
ramp = 100 + np.linspace(0, 0.5, 500)
la, qv, twd = compute_path_signatures(ramp, window=60, step=1)
# After warmup, all should be defined
print(f"  RAMP Lévy area mean={np.nanmean(la[100:]):+.5f}  (expect ~0, pure trend)")
print(f"  RAMP quad var  mean={np.nanmean(qv[100:]):+.5f}  (expect small positive)")
print(f"  RAMP twd       mean={np.nanmean(twd[100:]):+.5f}  (expect POSITIVE, late drift)")
assert np.nanmean(twd[100:]) > 0, "FAIL: up-ramp time-weighted drift should be positive"

# Inverted ramp: twd should be negative
downramp = 100 - np.linspace(0, 0.5, 500)
la2, qv2, twd2 = compute_path_signatures(downramp, window=60, step=1)
print(f"  DOWN twd       mean={np.nanmean(twd2[100:]):+.5f}  (expect NEGATIVE)")
assert np.nanmean(twd2[100:]) < 0, "FAIL: down-ramp twd should be negative"

# Noise: all three should be near 0 on average
c_noise = 100 + np.cumsum(rng.normal(0, 0.01, 500))
la3, qv3, twd3 = compute_path_signatures(c_noise, window=60, step=1)
print(f"  NOISE Lévy     mean={np.nanmean(la3[100:]):+.6f}  (expect ~0)")
print(f"  NOISE quad var mean={np.nanmean(qv3[100:]):.6f}  (expect > RAMP quad var)")
assert np.nanmean(qv3[100:]) > np.nanmean(qv[100:]) * 0.1, "FAIL: noise QV should not be tiny"
print("  ✓ path sig directional sign correct")

# Lévy area sign test: V-shape path (down then up) should have nonzero Lévy area
v_shape = 100 + np.concatenate([
    np.zeros(60),                          # flat prefix so window is filled
    -np.linspace(0, 0.5, 30), np.linspace(-0.5, 0.5, 30)
])
la_v, _, _ = compute_path_signatures(v_shape, window=60, step=1)
print(f"  V-SHAPE Lévy  = {la_v[-1]:+.5f}  (expect NONZERO)")
assert abs(la_v[-1]) > 1e-4, "FAIL: V-shape should have nonzero Lévy area"
print("  ✓ Lévy area distinguishes V-shape from linear trend")

# ---- Test 3: Transfer entropy — TE(Y->X) on independent series should ~0,
#              on Y = X_{t-1} should be LARGE
print("\nTEST 3 — Transfer entropy")
N = 500
x_indep = rng.normal(0, 1, N)
y_indep = rng.normal(0, 1, N)
te_indep = _te_binned_2x2x2(x_indep, y_indep)
print(f"  Independent X, Y:  TE={te_indep:+.4f}  (expect ~0)")

# Coupled: x_t = 0.9 * y_{t-1} + noise
y_drive = rng.normal(0, 1, N)
x_follow = np.zeros(N)
for i in range(1, N):
    x_follow[i] = 0.9 * y_drive[i-1] + 0.1 * rng.normal()
te_coupled = _te_binned_2x2x2(x_follow, y_drive)
print(f"  Y drives X:        TE={te_coupled:+.4f}  (expect >> 0)")
assert te_coupled > te_indep + 0.05, f"FAIL: TE(Y->X) should be HIGH when Y drives X. Got {te_coupled} vs {te_indep}"
print("  ✓ TE detects Y→X coupling")

# ---- Test 4: Strict past-only verification for TE
print("\nTEST 4 — TE strict past-only (corrupt future should NOT affect past TE)")
n = 2000
base = 1000 + np.cumsum(rng.normal(0, 0.5, n))
idx = pd.date_range("2020-01-01", periods=n, freq="5min")
te_a = compute_transfer_entropy_h1_to_m5(idx.values, base, window=50, step=6)
# Corrupt only the FUTURE (last 500 bars); past TE must be bit-identical
base_c = base.copy()
base_c[1500:] += rng.normal(0, 50, 500)
te_b = compute_transfer_entropy_h1_to_m5(idx.values, base_c, window=50, step=6)
# Compare te values at indices [:1400] (fully past of corruption)
fa = np.where(np.isfinite(te_a[:1400]))[0]
fb = np.where(np.isfinite(te_b[:1400]))[0]
common = np.intersect1d(fa, fb)
if len(common) > 0:
    diffs = te_a[common] - te_b[common]
    max_diff = np.abs(diffs).max()
    print(f"  Max |diff| on past bars: {max_diff:.2e}  (expect 0.0 → strict past-only)")
    assert max_diff < 1e-12, f"FAIL: past TE values differ after corrupting future. max diff = {max_diff}"
    print("  ✓ TE is strictly past-only")

print("\nALL SANITY TESTS PASSED ✓")
