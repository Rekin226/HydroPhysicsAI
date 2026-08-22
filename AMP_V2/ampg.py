"""AMP-G: a multi-band pumping signature for groundwater records.

Generalises the single-band AMP of Ouedraogo, Hsu & Wang (2023). AMP measures one scalar
at one frequency (1 cpd). An unconstrained survey of 34 Choushui wells over 11 years finds
a median of 3.9 coherent spectral lines per well, with the sub-daily band (<0.5 cpd,
median 1.92 cm) as large as the diurnal one (2.69 cm), and a harmonic ladder at 2-5 cpd.
AMP-G measures the whole structure.

Three ideas:

1. **Detect, don't assume.** Coherent lines are found against a local median noise floor
   over the full record, so nothing is presupposed about where pumping lives.

2. **Complex demodulation per line.** For each line f_k the analytic amplitude and phase
   are A_k(t) = LP[x(t) e^{-2 pi i f_k t}]. Bandwidth is set by the low-pass, so the
   time-frequency trade-off is explicit and per-band rather than global.

3. **Duty cycle from harmonic ratios.** A pump switching on and off is a rectangular pulse
   train; its n-th Fourier coefficient goes as sin(n pi d)/(n pi) for duty fraction d, so

       A2/A1 = |cos(pi d)|   =>   d = arccos(A2/A1) / pi        (taking d <= 0.5)

   which converts the harmonic ladder into **hours pumped per day**. Amplitude alone
   measures pumping *rate*; amplitude with duration gives a proxy for pumped *volume*,
   which is the quantity electricity consumption actually tracks.

Caveat carried through the API: the aquifer acts as a low-pass filter between pumping and
the observed head, so higher harmonics are attenuated and the recovered duty is an
*apparent* duty cycle, biased high. The frequency dependence of that attenuation is itself
diagnostic of transmissivity and storage -- exploited later, not here.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from gafd import instantaneous_mean

# Exact astronomical / thermal constituents (cycles per day) used for attribution.
CONSTITUENTS = {
    "O1": 0.929536, "P1": 0.997262, "S1": 1.000000, "K1": 1.002738,
    "N2": 1.895982, "M2": 1.932274, "S2": 2.000000,
}


def detect_lines(x: np.ndarray, fs: float, f_lo: float = 0.02, f_hi: float = 6.0,
                 snr: float = 8.0, floor_win: int = 2001) -> list[dict]:
    """Coherent spectral lines above a local median noise floor.

    Returns dicts with ``f`` (cycles/day), ``amp`` (same units as ``x``) and ``snr``,
    strongest first. Nothing about pumping is assumed.
    """
    x = np.asarray(x, dtype="float64")
    N = x.size
    A = np.abs(np.fft.rfft(x * np.hanning(N))) / N * 2
    fr = np.fft.rfftfreq(N, d=1.0 / fs)
    band = (fr >= f_lo) & (fr <= f_hi)
    A, fr = A[band], fr[band]
    import pandas as pd
    floor = pd.Series(A).rolling(floor_win, center=True, min_periods=200).median().to_numpy()
    floor = np.maximum(floor, 1e-12)
    idx, _ = find_peaks(A / floor, height=snr, distance=max(2, floor_win // 10))
    out = [{"f": float(fr[i]), "amp": float(A[i]), "snr": float(A[i] / floor[i])} for i in idx]
    return sorted(out, key=lambda d: -d["amp"])


def demodulate(x: np.ndarray, fs: float, f0: float, bw: float = 0.02) -> np.ndarray:
    """Complex demodulation at ``f0``: returns the analytic envelope A(t) (complex).

    ``bw`` (cycles/day) sets the low-pass half-width, hence the time resolution. The
    default admits seasonal modulation (sidebands at ~0.0027 cpd) with margin while
    excluding neighbouring lines.
    """
    n = np.arange(x.size, dtype="float64")
    z = np.asarray(x, dtype="float64") * np.exp(-2j * np.pi * f0 * n / fs)
    M = max(2, int(round(fs / bw / 2)))
    M = min(M, x.size // 2 - 2)
    return 2.0 * (instantaneous_mean(z.real, M) + 1j * instantaneous_mean(z.imag, M))


def duty_from_harmonics(a1: np.ndarray, a2: np.ndarray) -> np.ndarray:
    """Apparent duty fraction from the first two harmonics: d = arccos(A2/A1)/pi."""
    r = np.clip(np.abs(a2) / np.maximum(np.abs(a1), 1e-12), 0.0, 1.0)
    return np.arccos(r) / np.pi


def fit_duty_response(amps: np.ndarray, ns: np.ndarray) -> tuple[float, float]:
    """Jointly fit duty fraction and aquifer attenuation from the harmonic ladder.

    A rectangular pump train has |c_n| ~ |sin(n pi d)|/(n pi). The aquifer low-passes it,
    so the observed amplitude is A_n = exp(-alpha * n) |sin(n pi d)|/(n pi) * K. Fitting
    ``d`` and ``alpha`` together removes the bias that reading A2/A1 alone incurs, and
    ``alpha`` is itself a measure of the aquifer's frequency response.

    Needs at least three harmonics; returns (nan, nan) otherwise.
    """
    if amps.size < 3 or np.any(amps <= 0):
        return np.nan, np.nan
    y = np.log(amps / amps[0])

    def resid(p):
        d, alpha = p
        m = np.abs(np.sin(ns * np.pi * d)) / (ns * np.pi)
        if np.any(m <= 0):
            return np.full(ns.size, 1e3)
        return (np.log(m / m[0]) - alpha * (ns - ns[0])) - y

    from scipy.optimize import least_squares
    best, bcost = (np.nan, np.nan), np.inf
    for d0 in (0.1, 0.2, 0.3, 0.45):
        try:
            r = least_squares(resid, [d0, 0.3], bounds=([0.01, 0.0], [0.5, 5.0]))
            if r.cost < bcost:
                bcost, best = r.cost, (float(r.x[0]), float(r.x[1]))
        except Exception:
            pass
    return best


def classify(lines: list[dict], f_tol: float = 0.01) -> dict:
    """Attribute detected lines to physical drivers.

    - ``fundamental``  line nearest 1.000 cpd (solar-day-locked pumping / thermal S1)
    - ``harmonics``    lines at n x fundamental, n = 2..5
    - ``rotation``     lines below 0.5 cpd (irrigation rotation, weekly schedules)
    - ``tidal``        lines within ``f_tol`` of a non-S1 astronomical constituent
    - ``other``        everything else
    """
    out = {"fundamental": None, "harmonics": [], "rotation": [], "tidal": [], "other": []}
    diurnal = [l for l in lines if abs(l["f"] - 1.0) < 0.05]
    if diurnal:
        out["fundamental"] = max(diurnal, key=lambda l: l["amp"])
    f0 = out["fundamental"]["f"] if out["fundamental"] else 1.0
    for l in lines:
        if out["fundamental"] is not None and l is out["fundamental"]:
            continue
        tid = [k for k, v in CONSTITUENTS.items() if k != "S1" and abs(l["f"] - v) < f_tol]
        n = round(l["f"] / f0)
        if l["f"] < 0.5:
            out["rotation"].append(l)
        elif 2 <= n <= 5 and abs(l["f"] - n * f0) < 0.05:
            out["harmonics"].append({**l, "n": int(n)})
        elif tid:
            out["tidal"].append({**l, "constituent": tid[0]})
        else:
            out["other"].append(l)
    out["harmonics"].sort(key=lambda l: l["n"])
    return out


def ampg(x: np.ndarray, fs: float, bw: float = 0.02, snr: float = 8.0) -> dict:
    """Full AMP-G signature of one record.

    Returns time series (same length as ``x``):
      ``amp_fund``   fundamental amplitude -- the v1-comparable quantity
      ``amp_harm``   RMS amplitude of the duty-cycle harmonics
      ``amp_rot``    RMS amplitude of the sub-daily rotation band
      ``duty``       apparent duty fraction from A2/A1
      ``volume``     amp_fund * duty -- rate x duration, the volume proxy
      ``amp_total``  RMS over fundamental + harmonics (all pumping-attributed diurnal power)
    plus the classification and the detected line list.
    """
    lines = detect_lines(x, fs, snr=snr)
    cls = classify(lines)
    zeros = np.zeros(x.size)
    if cls["fundamental"] is None:
        return {"found": False, "lines": lines, "classes": cls,
                **{k: zeros for k in ("amp_fund", "amp_harm", "amp_rot", "duty",
                                      "volume", "amp_total")}}

    a1 = np.abs(demodulate(x, fs, cls["fundamental"]["f"], bw))
    harm = [np.abs(demodulate(x, fs, h["f"], bw)) for h in cls["harmonics"]]
    a2 = next((h for h, m in zip(harm, cls["harmonics"]) if m["n"] == 2), None)
    rot = [np.abs(demodulate(x, fs, r["f"], min(bw, r["f"] / 3))) for r in cls["rotation"]]

    amp_harm = np.sqrt(np.sum([h ** 2 for h in harm], axis=0)) if harm else zeros
    amp_rot = np.sqrt(np.sum([r ** 2 for r in rot], axis=0)) if rot else zeros
    duty = duty_from_harmonics(a1, a2) if a2 is not None else np.full(x.size, np.nan)

    # Response-corrected duty from the whole ladder (bias-free where >=3 harmonics exist).
    ns = np.array([1] + [h["n"] for h in cls["harmonics"]], dtype="float64")
    ladder = np.array([cls["fundamental"]["amp"]] + [h["amp"] for h in cls["harmonics"]])
    duty_fit, alpha = fit_duty_response(ladder, ns)
    duty_eff = duty_fit if np.isfinite(duty_fit) else np.nanmedian(duty)
    volume = a1 * (duty_eff if np.isfinite(duty_eff) else 0.5)

    return {"found": True, "lines": lines, "classes": cls,
            "amp_fund": a1, "amp_harm": amp_harm, "amp_rot": amp_rot,
            "duty": duty, "duty_fit": duty_fit, "alpha": alpha, "volume": volume,
            "amp_total": np.sqrt(a1 ** 2 + amp_harm ** 2),
            "f0": cls["fundamental"]["f"],
            "n_harm": len(cls["harmonics"]), "n_rot": len(cls["rotation"]),
            "n_tidal": len(cls["tidal"])}
