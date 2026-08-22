"""Synthetic validation: can each estimator recover a KNOWN, non-stationary pumping signal?

The test is deliberately unkind to v1: the pumping frequency drifts, which is what happens
when an irrigation schedule changes. v1's band-pass is fixed; v2's mode selection is not.
"""

import numpy as np
from amp import amp_v1, amp_v2

rng = np.random.default_rng(0)
fs = 24.0                                    # hourly samples per day
days = 3 * 365
t = np.arange(0, days, 1 / fs)

# --- ground truth pumping: seasonal amplitude, drifting frequency, duty-cycle harmonic
A_true = 0.10 + 0.07 * np.sin(2 * np.pi * (t / 365.0) - 1.2)      # m, peaks ~March
f_true = 1.00 + 0.08 * (t / days)                                  # cpd, drifts 1.00 -> 1.08
phase = 2 * np.pi * np.cumsum(f_true) / fs
pump = A_true * (np.sin(phase) + 0.35 * np.sin(2 * phase))         # + duty-cycle harmonic

# --- confounders present in real groundwater records
trend = -0.55 * (t / 365.0)                                        # long-term decline, m/yr
seasonal = 0.9 * np.sin(2 * np.pi * t / 365.0 + 0.6)
rain = np.zeros_like(t)
for d in rng.choice(days, 90, replace=False):                      # recharge pulses
    i = int(d * fs)
    rain[i:] += rng.gamma(2.0, 0.10) * np.exp(-np.arange(len(t) - i) / (fs * 12))
tide = 0.012 * np.sin(2 * np.pi * 1.93 * t)
noise = rng.normal(0, 0.004, t.size)

signal = trend + seasonal + rain + pump + tide + noise

r1 = amp_v1(signal, fs)
r2 = amp_v2(signal, fs)

inner = slice(int(60 * fs), int((days - 60) * fs))                 # ignore filter edges
truth = A_true[inner]

def score(est, name):
    e = est["amp"][inner]
    s = np.polyfit(truth, e, 1)[0]                                 # scale (arbitrary units)
    r = np.corrcoef(truth, e)[0, 1]
    rel = np.mean(np.abs(e / s - truth)) / np.mean(truth)
    print(f"  {name:8s} corr(A_true) = {r:6.3f}   scale = {s:6.3f}   mean rel. err = {rel*100:5.1f}%")
    return r

print(f"ground truth: A varies {A_true.min():.3f}-{A_true.max():.3f} m, f drifts "
      f"{f_true[0]:.3f} -> {f_true[-1]:.3f} cpd")
print(f"v1 identified f_pump = {r1['f_pump']:.3f} cpd (fixed band)")
print(f"v2 identified f_pump = {r2['f_pump']:.3f} cpd, f_std = {r2['f_std']:.3f}, "
      f"IMFs = {r2['n_imf']}, candidates = {r2.get('n_cand')}")
print("\namplitude recovery:")
c1 = score(r1, "AMP v1")
c2 = score(r2, "AMP v2")

if r2["found"]:
    fe = r2["freq"][inner]
    ft = f_true[inner]
    print(f"\nfrequency tracking (v2 only, v1 cannot do this):")
    print(f"  corr(f_true) = {np.corrcoef(ft, fe)[0,1]:6.3f}   "
          f"mean abs err = {np.mean(np.abs(fe-ft)):.4f} cpd   "
          f"drift recovered = {fe[-1]-fe[0]:+.4f} cpd (true {ft[-1]-ft[0]:+.4f})")
print(f"\nverdict: v2 {'beats' if c2 > c1 else 'does not beat'} v1 on amplitude "
      f"({c2:.3f} vs {c1:.3f})")
