"""Fair comparison of AMP v1 and v2 across the non-stationarity regimes that actually
occur in irrigation pumping. The fundamental stays locked to the solar day (1 cpd);
what varies is amplitude, duty cycle, phase and intermittency."""

import numpy as np
from amp import amp_v1, amp_v2

fs, days = 24.0, 3 * 365
t = np.arange(0, days, 1 / fs)
yr = t / 365.0


def build(regime, seed=0):
    rng = np.random.default_rng(seed)
    A = 0.10 + 0.07 * np.sin(2 * np.pi * yr - 1.2)          # seasonal intensity (truth)
    duty = np.full_like(t, 0.35)                             # harmonic weight
    phase0 = np.zeros_like(t)
    if regime == "duty":        # irrigation shifts from long slow draw to short hard draw
        duty = 0.15 + 0.55 * (t / days)
    if regime == "phase":       # pumping start time drifts across the day, seasonally
        phase0 = 1.5 * np.sin(2 * np.pi * yr)
    if regime == "intermittent":  # pumping stops in the wet season
        A = A * (0.15 + 0.85 * (np.sin(2 * np.pi * yr - 1.2) > -0.2))
    ph = 2 * np.pi * 1.00 * t + phase0
    pump = A * (np.sin(ph) + duty * np.sin(2 * ph))
    rain = np.zeros_like(t)
    for d in rng.choice(days, 90, replace=False):
        i = int(d * fs)
        rain[i:] += rng.gamma(2.0, 0.10) * np.exp(-np.arange(len(t) - i) / (fs * 12))
    sig = (-0.55 * yr + 0.9 * np.sin(2 * np.pi * yr + 0.6) + rain + pump
           + 0.012 * np.sin(2 * np.pi * 1.93 * t) + rng.normal(0, 0.004, t.size))
    # ground truth "pumping stress" = RMS envelope of the pump term
    truth = A * np.sqrt(0.5 * (1 + duty ** 2))
    return sig, truth


inner = slice(int(60 * fs), int((days - 60) * fs))
print(f"{'regime':<14}{'v1 narrow':>11}{'v1 wide':>10}{'v2 single':>11}{'v2 band':>10}   (corr with true pumping stress)")
print("-" * 68)
rows = {}
for regime in ["stationary", "duty", "phase", "intermittent"]:
    sig, truth = build(regime)
    T = truth[inner]
    out = {}
    out["v1 narrow"] = amp_v1(sig, fs, f_pump=1.0, bandwidth=0.001)["amp"]
    out["v1 wide"] = amp_v1(sig, fs, f_pump=1.0, bandwidth=0.05)["amp"]
    out["v2 single"] = amp_v2(sig, fs, mode="single")["amp"]
    out["v2 band"] = amp_v2(sig, fs, mode="band")["amp"]
    cor = {k: np.corrcoef(T, v[inner])[0, 1] for k, v in out.items()}
    rows[regime] = cor
    print(f"{regime:<14}" + "".join(f"{cor[k]:>11.3f}" for k in
          ["v1 narrow", "v1 wide", "v2 single", "v2 band"]))
print("-" * 68)
mean = {k: np.mean([rows[r][k] for r in rows]) for k in rows["duty"]}
print(f"{'mean':<14}" + "".join(f"{mean[k]:>11.3f}" for k in
      ["v1 narrow", "v1 wide", "v2 single", "v2 band"]))
