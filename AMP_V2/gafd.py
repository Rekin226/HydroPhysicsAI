"""Gaussian Average Filtering Decomposition (GAFD) + Hilbert transform = HGT.

Implements the Hilbert-Gauss Transform of Lin, Tan, Ku & Tian (2023), Sensors 23(3785),
doi:10.3390/s23083785, as an EMD/EEMD alternative free of mode mixing, redundant
decomposition and boundary effects.

GAFD sifts intrinsic mode functions with iterated Gaussian moving-average (low-pass)
filters instead of cubic-spline envelopes:

    w[m]   = exp(-(alpha*m/M)**2 / 2),  -M <= m <= M          (Eq. 1)
    w_G    = w / sum(w)                                        (Eq. 2)
    m_i[n] = sum_m w_G[m] * s_e[m+n]     (instantaneous mean)  (Eq. 3)
    r[n]   = s[n] - m_i[n]               (prototype IMF)       (Eq. 4)
    M      = 2 * floor(eps * N / N_e)                          (Eq. 5)

The IMF is the high-pass residue ``r``; the decomposition then continues on the
low-frequency instantaneous mean ``m_i``, giving an EMD-like cascade from high to low
frequency.

Hilbert transform of each IMF gives the analytic signal (Eq. 7-8), hence instantaneous
amplitude and instantaneous frequency.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve, hilbert

ALPHA = 4.0728   # paper's value: end values 0.025% of window max
EPS = 1.8        # paper's value for the window-length rule (Eq. 5)


def _n_extrema(x: np.ndarray) -> int:
    """Number of local extrema by the second-derivative (sign-change) test."""
    d = np.diff(x)
    nz = d[d != 0]
    if nz.size < 2:
        return 0
    return int(np.sum(np.sign(nz[1:]) != np.sign(nz[:-1])))


def gaussian_window(M: int, alpha: float = ALPHA) -> np.ndarray:
    """Normalised discrete Gaussian window of length 2M+1 (Eq. 1-2)."""
    m = np.arange(-M, M + 1, dtype="float64")
    w = np.exp(-((alpha * m / M) ** 2) / 2.0)
    return w / w.sum()


def _extend(x: np.ndarray, M: int) -> np.ndarray:
    """Double-symmetrical reflection extension (the paper's choice of the four styles).

    Reflects about the boundary *sample* and also about the boundary *value*, so the
    extension continues the local slope instead of folding it back. This is what kills
    the boundary effect that plagues EMD.
    """
    n = x.size
    k = min(M, n - 1)
    left = 2.0 * x[0] - x[k:0:-1]
    right = 2.0 * x[-1] - x[-2:-k - 2:-1]
    pad_l = np.full(M, left[0] if k else x[0])
    pad_r = np.full(M, right[-1] if k else x[-1])
    if k:
        pad_l[M - k:] = left
        pad_r[:k] = right
    return np.concatenate([pad_l, x, pad_r])


def instantaneous_mean(x: np.ndarray, M: int, alpha: float = ALPHA) -> np.ndarray:
    """Gaussian moving average of ``x`` with window half-length ``M`` (Eq. 3)."""
    w = gaussian_window(M, alpha)
    xe = _extend(x, M)
    return fftconvolve(xe, w, mode="same")[M:M + x.size]


def window_halflength(x: np.ndarray, eps: float = EPS) -> int | None:
    """M = 2*floor(eps*N/N_e) (Eq. 5); None when the M < N/2 - 1 criterion fails (Eq. 6)."""
    N = x.size
    ne = _n_extrema(x)
    if ne < 2:
        return None
    M = 2 * int(np.floor(eps * N / ne))
    if M < 1 or M >= N / 2 - 1:
        return None
    return M


def gafd(x: np.ndarray, max_imf: int = 12, eps: float = EPS,
         energy_ratio: float = 1e4, energy_diff: float = 1e-8) -> list[np.ndarray]:
    """Decompose ``x`` into IMFs, high frequency first, plus a final residual.

    Stop criteria (all three from the paper): energy ratio of the original signal to the
    residual exceeds ``energy_ratio``; energy difference between neighbouring IMFs falls
    below ``energy_diff``; or the window rule (Eq. 6) fails.
    """
    x = np.asarray(x, dtype="float64")
    e0 = float(np.sum(x ** 2)) or 1.0
    imfs: list[np.ndarray] = []
    resid = x.copy()
    prev_e = None
    for _ in range(max_imf):
        M = window_halflength(resid, eps)
        if M is None:
            break
        mean = instantaneous_mean(resid, M)
        imf = resid - mean
        e = float(np.sum(imf ** 2))
        imfs.append(imf)
        resid = mean
        if e0 / max(float(np.sum(resid ** 2)), 1e-30) > energy_ratio:
            break
        if prev_e is not None and abs(e - prev_e) / e0 < energy_diff:
            break
        prev_e = e
    imfs.append(resid)
    return imfs


def hilbert_spectrum(imf: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Instantaneous amplitude and frequency of one IMF (Eq. 7-8).

    ``fs`` is the sampling frequency; the returned frequency carries its units
    (samples/day in => cycles/day out).
    """
    z = hilbert(np.asarray(imf, dtype="float64"))
    a = np.abs(z)
    phase = np.unwrap(np.angle(z))
    w = np.gradient(phase)
    return a, fs * w / (2.0 * np.pi)


def hgt(x: np.ndarray, fs: float, **kw) -> list[dict]:
    """Full Hilbert-Gauss Transform: GAFD then Hilbert on every IMF.

    Returns one dict per IMF with keys ``imf``, ``amp``, ``freq``, ``f_mean``, ``f_std``
    (Eq. 10). The trailing residual is included and flagged ``residual``.
    """
    imfs = gafd(x, **kw)
    out = []
    for i, c in enumerate(imfs):
        a, f = hilbert_spectrum(c, fs)
        inner = slice(len(c) // 20, -len(c) // 20 or None)   # drop edges from the stats
        out.append({
            "imf": c, "amp": a, "freq": f,
            "f_mean": float(np.mean(f[inner])),
            "f_std": float(np.std(f[inner], ddof=1)),
            "energy": float(np.sum(c ** 2)),
            "residual": i == len(imfs) - 1,
        })
    return out
