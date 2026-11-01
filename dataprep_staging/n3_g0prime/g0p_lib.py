"""Shared code of N3.G0prime.v1 (synthetic, model-only) and of N3G' (the N3 pile months'
second, gating role, registered only if G0' passes): three conversion estimators that do not
regress the pile increment on the noisy nest drawdown change by OLS, their moving-block
bootstrap, the first-stage F statistics and the G0' synthetic world.

Why: N3's G0 failed (registered result sha 80e260d0) because the pile-FE OLS slope of the
monthly pile increment on the observed nest dD is attenuated by the nest-vs-cell head error
(errors in variables; median rho_hat 0.18-0.82 at rho = 1, coverage 0.000-0.40). User ruling
R7.n3_g0prime (user_rulings_2026-11-01b.json) allows one attenuation-robust estimator to be
declared and checked by a fresh, disclosed G0' run.

The estimators, in the DECLARED ORDER (judge rank 14; all three are always reported):

(a) ``harmonic_ratio``: per pile, the annual harmonic (sin, cos of the calendar month) of the
    pile increment and of the nest dD, each by OLS with a pile intercept; pooled ratio
    sign(sum_p b_y,p . b_x,p) * sqrt(sum_p |b_y,p|^2 / sum_p |b_x,p|^2). Phase-free in each
    pile; biased upward in |b_x| (and so downward in the ratio) at low SNR (Rician bias).
(b) ``iv_harmonics``: 2SLS of the increment on dD with a pile fixed effect, instruments the
    per-pile annual sin and cos (2 per pile).
(c) ``iv_model_heads``: 2SLS with a pile fixed effect, one instrument: the frozen ZR model's
    own nest dD at the same months (``model_side.json`` world, the series g_mod is built on).

``ols_fe`` (N3's estimator, ``n3lib.fe_slope``) is computed on the same draws as a reference
line (never selectable). Each estimator's g_mod is the SAME estimator on the noise-free model
world (as N3's g_mod is N3's regression on the model), a frozen number; rho = g / g_mod; the
90 % interval is the percentile interval of a 3-month moving-block bootstrap within pile
(n3lib.block_indices, B 2000, the same resampled rows for every estimator); the verdict is
N2's shared rule through ``n3lib.conversion_verdict`` (bootstrap_scorer.py sha f8990d61,
cited, never edited).

Pure functions on arrays; nothing here opens a product, a forward archive or an observation.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
N3_DIR = REPO / "dataprep_staging" / "n3_piles"
for _p in (str(N3_DIR), str(REPO / "dataprep_staging"), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import g0_n3 as G0  # noqa: E402  (frozen: grid, thresholds, AR(1) error, truths)
import n3lib as L  # noqa: E402  (frozen: bootstrap indices, verdict via N2's shared rule)

# ----------------------------------------------------------------------------- identity
E = "dataprep_staging/n3_g0prime"
ID = "N3.G0prime.v1"                       # the declaration (kind 'declaration'), G0'
N3G_ID = "N3Gprime.piles_conversion.v1"    # the gating registration (judge's N3G'.v1)
WP = "N3"
TEST_12 = "N3GP.conversion_12m"
TEST_24 = "N3GP.conversion_24m"
RULING_FILE = "results/dataprep/rulings/user_rulings_2026-11-01b.json"
RULING_ID = "R7.n3_g0prime"
RULING_SHA = "5f53909bcc9154d8a22da902402fcf41264facccd1bf7cf29eb28c48c8f6b6ab"
N3_REG_SHA = "afa017d76f974c4edb3b34bba38aa23ab9242d76b6d384f6a4d9e786b62df5c5"
N3_G0_RESULT = "results/dataprep/reports/N3/g0/result.json"
N3_G0_RESULT_SHA = "80e260d005c58eac8d6d112bc91c15e94009ae496180797877c982b2d1d25fcf"
MODEL_SIDE = "results/dataprep/reports/N3/mapping/model_side.json"
MODEL_SIDE_SHA = "1b40afe910a9ab5f020b5fc4557bfd75e97529eaf1e07c41b221755cd44e2149"
MAPPING = "results/dataprep/reports/N3/mapping/mapping.json"
MAPPING_SHA = "eea0d2d8f8f8d4c0a24a2e3fb8df59b0bb6d7cf949434959af7711d8a6543a38"
OUT_DIR = "results/dataprep/reports/N3.G0prime.v1"
UTC = dt.timezone.utc
DEADLINE = L.DEADLINE                      # 2026-11-30T16:00Z (N3G' registration)

# ----------------------------------------------------------------------------- the design
ESTIMATORS = ("harmonic_ratio", "iv_harmonics", "iv_model_heads")    # declared order a, b, c
REFERENCE = "ols_fe"
ALL_LINES = ESTIMATORS + (REFERENCE,)
F_OF = {"harmonic_ratio": "f_harmonic", "iv_harmonics": "f_harmonic",
        "iv_model_heads": "f_model"}
HORIZONS = {12: L.WINDOW, 24: L.WINDOW24}
G_MOD_KEY = {12: "g_mod_12", 24: "g_mod_24"}
# the frozen G0 grid and thresholds (imported, so a changed g0_n3.py changes the bound sha)
SD_GRID, PHI_GRID, RHOS, REPS = G0.SD_GRID, G0.PHI_GRID, G0.RHOS, G0.REPS
P_CORRECT, P_FALSE_MAX, COVERAGE_MIN, CORRECT = (G0.P_CORRECT, G0.P_FALSE_MAX,
                                                 G0.COVERAGE_MIN, G0.CORRECT)
# added cells (judge rank 14 and the adversarial review of lens B R2 / lens C R3): nest/pile
# annual amplitude ratio and phase lag of the nest's seasonal head behind the pile cell's
AMP_RATIOS = (0.7, 1.0, 1.3)
LAGS_MONTHS = (0, 2)
MISMATCH = tuple((a, g) for a in AMP_RATIOS for g in LAGS_MONTHS)
# actual-vs-forecast head deviation (added by the builder, not in the judge's text): 'none' =
# G0's world; 'cell' = an independent AR(1) with the cell's sd and phi, seen by the nest AND
# converted into the pile at the model's own OLS slope. Without it the true head change IS the
# model's (the instrument of (c)); (c) could still fail by variance, but the realistic case
# (actual heads deviate from the frozen forecast) would be untested. dF is uncorrelated with
# every instrument, so it only adds variance: (c) stays consistent, (a), (b) are not weaker.
FORECAST_DEV = ("none", "cell")
# cell classes (reported per estimator and horizon in every case) and the declared gate modes.
# A nest whose seasonal swing is amp x the pile cell's is NOT identifiable from nest heads:
# every nest-based estimator measures rho / amp (amp != 1), and a 2-month lag rotates the
# harmonic by 60 deg ((b) reads ~cos 60 = 0.5 x rho, (c) up to rho / cos 60). A gate over the
# 'amp' class cannot be passed by any estimator (adversarial review of the n3p build).
CELL_CLASSES = ("g0", "fdev", "lag", "amp")
GATE_MODES = {"all": CELL_CLASSES,              # the judge's text: 'a pass must hold there too'
              "identifiable": ("g0", "fdev")}   # mismatch cells a report-only sensitivity


def cell_class(amp: float, lag: int, fdev: str) -> str:
    """'g0' = N3 G0's own cells (amp 1, lag 0, no deviation); 'fdev' = + forecast deviation;
    'lag' = amp 1, lag 2 (either fdev); 'amp' = amp != 1 (any lag, either fdev)."""
    if amp != 1.0:
        return "amp"
    if lag != 0:
        return "lag"
    return "g0" if fdev == "none" else "fdev"


F_MIN = 10.0        # weak-instrument gate: MODEL-SIDE first-stage F (per-draw F reported)
SEED = 20261116
PERIOD_MONTHS = 12
OMEGA = 2.0 * math.pi / PERIOD_MONTHS


def sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def jsonable(obj):
    """Strict-JSON copy: inf -> 'inf' / '-inf' (an exact first stage), NaN -> None."""
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if math.isfinite(v) else (None if math.isnan(v) else
                                           ("inf" if v > 0 else "-inf"))
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def month_index(labels) -> np.ndarray:
    """'YYYY-MM' -> 12 * year + month - 1 (so the calendar harmonics are month-of-year)."""
    return np.asarray([12 * int(m[:4]) + int(m[5:7]) - 1 for m in labels], dtype=float)


def harmonics(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.sin(OMEGA * t), np.cos(OMEGA * t)


# ----------------------------------------------------------------------------- estimators
def _group_sums(y, x, z, s, c, idx) -> dict:
    """Within-group demeaned sums over the rows ``idx`` (shape (B, n_g)) of one group."""
    Y, X, Z, S, C = (np.asarray(v, float)[idx] for v in (y, x, z, s, c))
    Y, X, Z, S, C = (a - a.mean(axis=1, keepdims=True) for a in (Y, X, Z, S, C))
    g11, g12, g22 = (S * S).sum(1), (S * C).sum(1), (C * C).sum(1)
    qx = np.stack([(S * X).sum(1), (C * X).sum(1)], axis=1)
    qy = np.stack([(S * Y).sum(1), (C * Y).sum(1)], axis=1)
    det = g11 * g22 - g12 ** 2
    ok = det > 1e-9 * np.maximum(g11 * g22, 1e-300)
    with np.errstate(invalid="ignore", divide="ignore"):
        inv = np.stack([np.stack([g22, -g12], 1), np.stack([-g12, g11], 1)], 1)
        inv = inv / np.where(ok, det, np.nan)[:, None, None]
    bx = np.einsum("bij,bj->bi", inv, qx)
    by = np.einsum("bij,bj->bi", inv, qy)
    return {"xx": (X * X).sum(1), "xy": (X * Y).sum(1), "zy": (Z * Y).sum(1),
            "zx": (Z * X).sum(1), "zz": (Z * Z).sum(1),
            "xPx": (qx * bx).sum(1), "xPy": (qx * by).sum(1),
            "bx2": (bx * bx).sum(1), "by2": (by * by).sum(1), "bxy": (bx * by).sum(1)}


def _lines(t: dict) -> dict:
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"ols_fe": t["xy"] / t["xx"],
                "iv_model_heads": t["zy"] / t["zx"],
                "iv_harmonics": t["xPy"] / t["xPx"],
                "harmonic_ratio": np.sign(t["bxy"]) * np.sqrt(t["by2"] / t["bx2"])}


def _total(y, x, z, s, c, groups, index_of) -> dict:
    tot = None
    for g in index_of:
        sel = np.flatnonzero(groups == g)
        t = _group_sums(y[sel], x[sel], z[sel], s[sel], c[sel], index_of[g])
        tot = t if tot is None else {k: tot[k] + t[k] for k in tot}
    return tot


def first_stage_f(t: dict, n: int, n_groups: int) -> dict:
    """First-stage F of dD on the instruments, after the pile fixed effect (point sums)."""
    out = {}
    k = 2 * n_groups
    df = n - n_groups - k
    ssr0 = float(t["xx"][0])
    expl = float(t["xPx"][0])
    out["f_harmonic"] = (float((expl / k) / ((ssr0 - expl) / df))
                         if df > 0 and ssr0 - expl > 1e-12 * max(ssr0, 1e-300) else
                         (math.inf if df > 0 and expl > 0 else float("nan")))
    zz, zx = float(t["zz"][0]), float(t["zx"][0])
    df1 = n - n_groups - 1
    e1 = zx * zx / zz if zz > 0 else float("nan")
    out["f_model"] = (float(e1 / ((ssr0 - e1) / df1))
                      if df1 > 0 and np.isfinite(e1) and ssr0 - e1 > 1e-12 * max(ssr0, 1e-300)
                      else (math.inf if df1 > 0 and np.isfinite(e1) and e1 > 0
                            else float("nan")))
    return out


def estimate_all(y, x, z, months, groups, *, B: int = L.BOOT_B, block: int = L.BOOT_BLOCK,
                 rng=None, seed: int = L.BOOT_SEED, level: float = L.LEVEL) -> dict:
    """Every line's point estimate and central ``level`` percentile block-bootstrap interval
    (the same resampled rows for all lines), plus the first-stage F statistics.
    ``months``: 'YYYY-MM' labels of the rows; ``z``: the model's nest dD at those months."""
    rng = rng if rng is not None else np.random.default_rng(seed)
    y, x, z = (np.asarray(v, float) for v in (y, x, z))
    groups = np.asarray(groups)
    s, c = harmonics(month_index(months))
    keys = list(dict.fromkeys(groups.tolist()))
    point_idx = {g: np.arange(int((groups == g).sum()))[None, :] for g in keys}
    boot_idx = {g: L.block_indices(int((groups == g).sum()), B, block, rng) for g in keys}
    tp = _total(y, x, z, s, c, groups, point_idx)
    tb = _total(y, x, z, s, c, groups, boot_idx)
    pt, bs = _lines(tp), _lines(tb)
    a = (1.0 - level) / 2.0
    out = {"n": int(len(y)), "n_groups": len(keys), **first_stage_f(tp, len(y), len(keys))}
    for k in ALL_LINES:
        b = bs[k][np.isfinite(bs[k])]
        lo, hi = (np.quantile(b, [a, 1.0 - a]).tolist() if b.size else [np.nan, np.nan])
        out[k] = {"g": float(pt[k][0]), "lo": float(lo), "hi": float(hi),
                  "n_boot_finite": int(b.size)}
    return out


def verdicts(est: dict, g_mod: dict) -> dict:
    """N2's shared rule (n3lib.conversion_verdict) per line, rho = g / g_mod[line]."""
    return {k: L.conversion_verdict(est[k]["g"], est[k]["lo"], est[k]["hi"], g_mod[k])
            for k in ALL_LINES}


# ----------------------------------------------------------------------------- the world
def window_world(world: dict, span) -> dict:
    """Per primary pile: the model's window increments, nest dD, month labels and the
    reference month (``span[0]`` - 1, where the nest head path is set to 0)."""
    arr = G0.world_arrays(world, L.PRIMARY, span)
    for a in arr.values():
        a["ref"] = L.prev_month(span[0])
        if a["months"] != L.months(*span):
            raise ValueError("model world months are not the full window")
    return arr


def g_mod_lines(arr: dict) -> dict:
    """Each line on the noise-free model world (y = column increment, x = z = model nest dD):
    the frozen g_mod of that estimator, plus the model-side first-stage F."""
    y = np.concatenate([a["inc"] for a in arr.values()])
    x = np.concatenate([a["dd"] for a in arr.values()])
    m = sum((a["months"] for a in arr.values()), [])
    g = np.concatenate([[k] * len(a["inc"]) for k, a in arr.items()])
    est = estimate_all(y, x, x, m, g, B=2, rng=np.random.default_rng(0))
    return {"g_mod": {k: est[k]["g"] for k in ALL_LINES},
            "model_side_f": {"f_harmonic": est["f_harmonic"], "f_model": est["f_model"]}}


def seasonal_fit(dd: np.ndarray, months: list[str], ref: str) -> tuple[float, float]:
    """Annual harmonic (A, B) of the nest head path D (D(ref) = 0, D(m) = cumulative dD) by
    OLS on [1, t, cos wt, sin wt] over ref + the window months."""
    t = month_index([ref] + list(months))
    D = np.concatenate([[0.0], np.cumsum(np.asarray(dd, float))])
    X = np.column_stack([np.ones_like(t), t - t[0], np.cos(OMEGA * t), np.sin(OMEGA * t)])
    beta, *_ = np.linalg.lstsq(X, D, rcond=None)
    return float(beta[2]), float(beta[3])


def mismatch_dd(dd: np.ndarray, months: list[str], ref: str, amp: float,
                lag: int) -> np.ndarray:
    """The nest dD whose annual head component is ``amp`` x the model's, delayed by ``lag``
    months: D' = D - H(t) + amp * H(t - lag), H the fitted annual harmonic."""
    if amp == 1.0 and lag == 0:
        return np.asarray(dd, float).copy()
    A, Bc = seasonal_fit(dd, months, ref)
    t = month_index([ref] + list(months))

    def H(tt):
        return A * np.cos(OMEGA * tt) + Bc * np.sin(OMEGA * tt)
    delta = amp * H(t - lag) - H(t)
    return np.asarray(dd, float) + np.diff(delta)


def draw(arr: dict, rho: float, sd: float, phi: float, amp: float, lag: int, fdev: str,
         g_ols_mod: float, rng) -> tuple:
    """One synthetic window: y (printed pile increments, cm), x (observed nest dD, m), z (the
    model's nest dD, the instrument of (c)), month labels and pile keys."""
    ys, xs, zs, ms, gs = [], [], [], [], []
    for k, a in arr.items():
        n = len(a["inc"])
        e = G0.ar1(n + 1, sd, phi, rng)
        dF = np.diff(G0.ar1(n + 1, sd, phi, rng)) if fdev == "cell" else np.zeros(n)
        x = mismatch_dd(a["dd"], a["months"], a["ref"], amp, lag) + dF + np.diff(e)
        y = np.round(rho * (a["inc"] + g_ols_mod * dF) / L.PRINT_RES_CM) * L.PRINT_RES_CM
        ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(a["dd"])
        ys.append(y[ok])
        xs.append(x[ok])
        zs.append(np.asarray(a["dd"], float)[ok])
        ms += [m for m, o in zip(a["months"], ok, strict=True) if o]
        gs.append(np.array([k] * int(ok.sum())))
    return (np.concatenate(ys), np.concatenate(xs), np.concatenate(zs), ms,
            np.concatenate(gs))


def noise_cells() -> list[tuple]:
    """(sd, phi, amp, lag, fdev) in canonical order (the seed spawn order)."""
    return [(sd, phi, a, g, f) for sd in SD_GRID for phi in PHI_GRID for a, g in MISMATCH
            for f in FORECAST_DEV]


def cell_tag(sd, phi, amp, lag, fdev) -> str:
    return f"sd {sd} phi {phi} amp {amp} lag {lag} fdev {fdev}"
