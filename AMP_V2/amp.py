"""AMP estimators: v1 (Ouedraogo et al. 2023) and v2 (HGT-based).

v1 reproduces the published method -- Gaussian high-pass to strip the long-term trend,
FFT to identify the pumping frequency, narrow band-pass at that frequency, envelope.
Ouedraogo, Hsu & Wang (2023), J. Hydrol. Eng. 28(4), doi:10.1061/JHYEFF.HEENG-5760.

v2 replaces the fixed band-pass with the Hilbert-Gauss Transform, so the pumping mode is
found adaptively and returns instantaneous amplitude *and* instantaneous frequency. v1
assumes the pumping signal sits at a fixed frequency; v2 does not, which matters because
irrigation schedules shift with crop stage, drought and tariff.
"""

from __future__ import annotations

import numpy as np
from gafd import hgt, instantaneous_mean
from scipy.signal import hilbert


def gaussian_highpass(x: np.ndarray, fs: float, f_cut: float) -> np.ndarray:
    """Strip components below ``f_cut`` (cycles/day) with a Gaussian moving average."""
    M = max(2, int(round(fs / f_cut / 2)))
    M = min(M, x.size // 2 - 2)
    return x - instantaneous_mean(x, M)


def dominant_frequency(x: np.ndarray, fs: float, lo: float = 0.5, hi: float = 5.0) -> float:
    """Frequency of the largest spectral line in [lo, hi] cycles/day."""
    X = np.fft.rfft(x * np.hanning(x.size))
    fr = np.fft.rfftfreq(x.size, d=1.0 / fs)
    band = (fr >= lo) & (fr <= hi)
    return float(fr[band][np.argmax(np.abs(X[band]))])


def amp_v1(x: np.ndarray, fs: float, f_pump: float | None = None,
           bandwidth: float = 0.05, trend_cut: float = 0.1,
           smooth_days: float = 30.0) -> dict:
    """Published AMP: high-pass, narrow band-pass at the pumping frequency, envelope."""
    hp = gaussian_highpass(x, fs, trend_cut)
    f0 = dominant_frequency(hp, fs) if f_pump is None else f_pump
    X = np.fft.rfft(hp)
    fr = np.fft.rfftfreq(hp.size, d=1.0 / fs)
    X[~((fr > f0 - bandwidth) & (fr < f0 + bandwidth))] = 0.0
    bp = np.fft.irfft(X, n=hp.size)
    env = np.abs(hilbert(bp))
    M = max(2, int(round(smooth_days * fs / 2)))
    M = min(M, env.size // 2 - 2)
    return {"amp": instantaneous_mean(env, M), "f_pump": f0, "band": bp}


def amp_v2(x: np.ndarray, fs: float, f_lo: float = 0.6, f_hi: float = 2.6,
           trend_cut: float = 0.1, smooth_days: float = 30.0,
           mode: str = "band") -> dict:
    """HGT-based AMP: adaptive mode selection, instantaneous amplitude and frequency.

    The pumping IMF is the candidate whose mean instantaneous frequency lies in
    [f_lo, f_hi] with the smallest frequency spread (Eq. 11-12 of Lin et al. 2023) --
    i.e. the most coherent oscillation in the pumping band.
    """
    hp = gaussian_highpass(x, fs, trend_cut)
    modes = hgt(hp, fs)
    cand = [m for m in modes if not m["residual"] and f_lo <= m["f_mean"] <= f_hi]
    if not cand:
        return {"amp": np.zeros_like(x), "freq": np.full_like(x, np.nan),
                "f_pump": np.nan, "f_std": np.nan, "found": False, "n_imf": len(modes)}
    # The paper's rule (smallest f_std, Eq. 12) assumes a single clean mode. Pumping is a
    # duty-cycled square-ish wave, so its fundamental and 2 cpd harmonic sit one octave
    # apart -- EMD's resolution limit -- and routinely land in different IMFs. Selecting
    # one mode then discards most of the pumping energy. Combine the band instead.
    if mode == "single":
        best = min(cand, key=lambda m: m["f_std"])
        env, frq, w = best["amp"], best["freq"], None
    else:
        env = np.sqrt(np.sum([m["amp"] ** 2 for m in cand], axis=0))     # energy sum
        w = np.array([m["energy"] for m in cand], dtype="float64")
        w = w / w.sum()
        frq = np.sum([wi * m["freq"] for wi, m in zip(w, cand, strict=True)], axis=0)
    best = min(cand, key=lambda m: m["f_std"])
    M = max(2, int(round(smooth_days * fs / 2)))
    M = min(M, env.size // 2 - 2)
    return {"amp": instantaneous_mean(env, M),
            "freq": instantaneous_mean(frq, M),
            "f_pump": best["f_mean"], "f_std": best["f_std"],
            "found": True, "n_imf": len(modes), "n_cand": len(cand)}
