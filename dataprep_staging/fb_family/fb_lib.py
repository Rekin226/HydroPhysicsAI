"""FB.family_power.v1 (round-8 rank 16, FB part): the engine of a CPU-only synthetic power
study of a prospective Bayes-factor (BF) test between two recipe families, {R0, Q0} vs
{R1c, X5R1} (named by recipe id only), on the 2026-12..2027-10 window at E02's wells, per
county (Changhua = N2C's 74 wells; Yunlin = the 84).

Nothing here reads an observation. Inputs are model-only or pre-origin-derived:

* FV D_k (FV.decidability.v1, ``osse_fv.departure``): the datum-anchored departure of truth
  recipe k (FV v1's climatology truth forwards, baseline scenario). The hypothesis mean of k is
  taken against the REGISTERED E02 forward (cf2c7154), mu_k = departure(h_k, h_ZR), because
  the 2028 scorer subtracts E02's registered forecast; mu_k = FV's D_k (against ZR0) + FV's
  report-only ZR0 -> ZR offset (both FV-defined; the offset is common to every hypothesis).
* U1.decomp_power.v1: the registered, DR-bounded noise STRUCTURE (sub-group share a_s,
  persistent per-well share a_w, AR(1) phi, whole-window dropout, DR groups and sub-groups),
  drawn exactly as ``power_e02d.draw`` (``draw_scaled`` reproduces it bit for bit when the
  scale is constant in time; tested), but SCALED per well-month by E02d's registered window-A
  sigma_idio (sigma_idio_A.csv, sd_fit_rel + ensemble + drift terms, metres; lead ruling B3).
* the realised-forcing misspecification (S3): FV's unit responses (``osse_fv.basis`` of the
  truth forward's 58 Yunlin impulse arms, 2023-01..2027-10) times a calendar-year block
  bootstrap of FV's anomaly pool (``osse_fv.draw_multipliers``; pump_electricity preorigin v3,
  design_time_evidence, as FV v2) added to the truth world; the hypothesis means stay at
  climatology, so this is a gating misspecification arm.

Statistic (shared by the power study and the 2028 scorer; one implementation, here):

    e = E02d's relative operator of (obs - ZR) (minus the leave-one-out median of the well's
        DR group that month, >= 5 others), per well and month;
    ybar_w = mean of e over the well's months (>= 6, else the well is not scored);
    m_kw  = the SAME operator and mean applied to mu_k on the REALISED mask (mu_k set missing
            wherever obs is missing), so the plug-in mean sees the same medians' membership;
    v_gw  = E[ybar_w^2] under U1 working cell g (a_s, a_w, phi; sigma_idio scale) with mu = 0
            and no missingness, calibrated by simulation; a well with n < T months uses
            v_gw * T / n (conservative for positively correlated months);
    p(ybar | k, g) = prod_w N(ybar_w; m_kw, v_gw) over the scored wells (working
            independence; the worlds carry U1's correlated sub-group shocks);
    p(ybar | H) = mean over the members k of H and the working cells g;
    log BF(A:B) = log p(ybar | A) - log p(ybar | B), equal prior odds.
    Evaluable iff >= 20 wells are scored (E02d's county floor).

Candidates (lead ruling B1): ONLY the family pair per county gates and may be registered; the
six truth pairs are computed report-only and can never pass or register.

World arms per truth (obs - ZR = mu_k [+ R_k(a)] + x):
    base           gating: U1's a_s x a_w x phi x dropout grid;
    forcing        gating: + R_k(a), on a_s x a_w x phi x {0, max dropout};
    month_missing  gating: each well-month missing with probability MONTH_MISSING, on
                   a_s x a_w x phi x {0, max dropout};
    heavy          report-only: U1's heavy-tailed well type (a_s max, phi max, dropout {0, max});
    scale_x1.5, scale_x2  report-only: the worlds' noise times 1.5 / 2 while the likelihood
                   keeps v (scale misspecification), at the grid's top cell, dropout {0, max}.

Gate (round-8 plan #16): min over truths x gating rows of P(correct strong) >= 0.8 AND max of
P(wrong strong) <= 0.05, strong = BF >= 20. Under a CORRECT model P(wrong strong) <= 1/B =
0.05 by the universal likelihood-ratio bound, so the wrong-side half of the gate sits at its
bound: it can fail only through misspecification or Monte Carlo noise (SE about 0.005 at 2000
replicates) and is disclosed as such.

Seeds are required arguments everywhere (no default). RNG: np.random.default_rng(
SeedSequence(seed, spawn_key=(county index, stream, a, b))): stream 0 = worlds (truth index,
world cell index), 1 = calibration (0, working cell index), 2 = self-check, 3 = forcing draws
(truth index, world cell index).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STAGING = HERE.parent
for _p in (str(HERE), str(STAGING), str(STAGING / "e02d"), str(STAGING / "e02c"),
           str(STAGING / "fv_decidability")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import e02dlib as E  # noqa: E402  (E02d's relative operator; bound by U1.decomp_power.v1)
import power_e02d as U1P  # noqa: E402  (U1.decomp_power.v1's world draw; bound)

VERSION = "2.0.0"
POWER_ID = "FB.family_power.v1"
BF_ID = "FB.family_bf_2026-12.v1"
WP = "FB"
LEDGER_ROW = "E29 (FB family BF, ledger only)"
U1_ID = "U1.decomp_power.v1"
E02D_ID = "E02d.decomposed_prospective_2026-12.v1"
FV1_ID = "FV.decidability.v1"
FV2_ID = "FV.decidability.v2_postfailure"
N2C_ID = "N2C.changhua_response.prospective_2026-11.v1"
N2C_REPORT = f"results/dataprep/reports/{N2C_ID}/report.json"
REG_DEADLINE = "2026-11-30T16:00:00Z"
NOT_BEFORE = "2028-01-15T00:00:00Z"
NO_LATER_THAN = "2028-01-31T23:59:59Z"

WINDOW = ("2026-12", "2027-10")
WINDOW_MONTHS = [f"{y}-{m:02d}" for y, m in [(2026, 12)] + [(2027, i) for i in range(1, 11)]]
T_WIN = 11
FV_WINDOW_SLICE = (1, 12)          # FV's departure covers 2026-11..2027-10; drop 2026-11
DATA_SCOPE = "heads:2026-12..2027-10"

TRUTHS = ("R0", "Q0", "R1c", "X5R1")
FAMILIES = {"R0Q0": ("R0", "Q0"), "R1cX5R1": ("R1c", "X5R1")}
FAMILY_PAIR = "R0Q0_vs_R1cX5R1"
PAIRS = ((FAMILY_PAIR, FAMILIES["R0Q0"], FAMILIES["R1cX5R1"]),
         ("R0_vs_Q0", ("R0",), ("Q0",)),
         ("R0_vs_R1c", ("R0",), ("R1c",)),
         ("R0_vs_X5R1", ("R0",), ("X5R1",)),
         ("Q0_vs_R1c", ("Q0",), ("R1c",)),
         ("Q0_vs_X5R1", ("Q0",), ("X5R1",)),
         ("R1c_vs_X5R1", ("R1c",), ("X5R1",)))
COUNTIES = ("Changhua", "Yunlin")
EXPECTED_WELLS = {"Changhua": 74, "Yunlin": 84}

B_STRONG = 20.0
P_CORRECT_MIN = 0.80
P_WRONG_MAX = 0.05
MIN_WELLS = E.MIN_WELLS_COUNTY     # 20
MIN_MONTHS = E.MIN_MONTHS          # 6
MIN_OTHERS = E.MIN_OTHERS          # 5
MONTH_MISSING = 0.2
SCALE_CELLS = (1.5, 2.0)
HORIZON = 58                       # FV's anomaly months 2023-01..2027-10
REPLICATES = 2000
CAL_REPLICATES = 2000
SMOKE = {"replicates": 40, "cal_replicates": 40}
V_FLOOR = 1e-12
STREAM_WORLD, STREAM_CAL, STREAM_SELF, STREAM_FORCING = 0, 1, 2, 3
SELF_TOL = 1e-9
GATING_ARMS = ("base", "forcing", "month_missing")
WRONG_SIDE_DISCLOSURE = (
    "the wrong-side half of the gate (P(wrong strong) <= 0.05 at B = 20) sits at the universal "
    "likelihood-ratio bound P <= 1/B: under a correct model it cannot fail, so it measures only "
    "misspecification (working independence, plug-in means, realised forcing, missingness) and "
    "Monte Carlo noise (SE about 0.005 at 2000 replicates)")


class EngineeringFailure(RuntimeError):
    """A self-check or input check failed: no power number is reported."""


# ----------------------------------------------------------------------------- candidates
def pair_table() -> dict:
    return {p: {"A": list(a), "B": list(b), "gating": p == FAMILY_PAIR} for p, a, b in PAIRS}


def candidates() -> list[dict]:
    """Every (pair, county) candidate, in a fixed order; only the family pair gates."""
    return [{"key": f"{p}@{c}", "pair": p, "county": c, "A": list(a), "B": list(b),
             "gating": p == FAMILY_PAIR} for c in COUNTIES for p, a, b in PAIRS]


def candidate(key: str) -> dict:
    for c in candidates():
        if c["key"] == key:
            return c
    raise KeyError(f"unknown candidate {key!r}; known: {[c['key'] for c in candidates()]}")


def family_keys() -> list[str]:
    return [c["key"] for c in candidates() if c["gating"]]


def require_family(keys) -> None:
    """Lead ruling B1: refuse any key that is not the family pair of a county."""
    bad = [k for k in keys if k not in family_keys()]
    if bad:
        raise ValueError(f"only the family pair may gate or register ({family_keys()}); "
                         f"refused: {bad}")


# ----------------------------------------------------------------------------- cells
def _f(x) -> float:
    return round(float(x), 6)


def county_grid(u1_county: dict, dropout) -> dict:
    """The U1.decomp_power.v1 grid of one county, unchanged (a_s, a_w, phi, dropout)."""
    return {"a_s": [_f(x) for x in u1_county["a_s"]], "a_w": [_f(x) for x in u1_county["a_w"]],
            "phi": [_f(x) for x in u1_county["phi"]], "dropout": [_f(x) for x in dropout]}


def working_cells(grid: dict) -> list[dict]:
    """The BF's nuisance grid: U1's a_s x a_w x phi (uniform prior), no dropout."""
    out = [{"a_s": a_s, "a_w": a_w, "phi": phi, "heavy": False}
           for a_s in grid["a_s"] for a_w in grid["a_w"] for phi in grid["phi"]]
    for i, c in enumerate(out):
        c["cell"] = i
    return out


def world_cells(grid: dict) -> list[dict]:
    """Every world cell of a county with its arm (see the module docstring)."""
    out = []
    ends = (0.0, max(grid["dropout"]))

    def add(arm, a_s, a_w, phi, drop, *, heavy=False, gating=True, scale=1.0, miss=0.0,
            forcing=False):
        out.append({"arm": arm, "a_s": a_s, "a_w": a_w, "phi": phi, "dropout": drop,
                    "heavy": heavy, "gating": gating, "scale": scale, "month_missing": miss,
                    "forcing": forcing})

    for a_s in grid["a_s"]:
        for a_w in grid["a_w"]:
            for phi in grid["phi"]:
                for drop in grid["dropout"]:
                    add("base", a_s, a_w, phi, drop)
    for a_s in grid["a_s"]:
        for a_w in grid["a_w"]:
            for phi in grid["phi"]:
                for drop in ends:
                    add("forcing", a_s, a_w, phi, drop, forcing=True)
    for a_s in grid["a_s"]:
        for a_w in grid["a_w"]:
            for phi in grid["phi"]:
                for drop in ends:
                    add("month_missing", a_s, a_w, phi, drop, miss=MONTH_MISSING)
    top = (max(grid["a_s"]), max(grid["a_w"]), max(grid["phi"]))
    for drop in ends:
        add("heavy", top[0], 0.0, top[2], drop, heavy=True, gating=False)
    for k in SCALE_CELLS:
        for drop in ends:
            add(f"scale_x{k:g}", *top, drop, gating=False, scale=k)
    for i, c in enumerate(out):
        c["cell"] = i
    return out


def rng_for(seed: int, county_i: int, stream: int, a: int, b: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(int(seed),
                                                        spawn_key=(county_i, stream, a, b)))


# ----------------------------------------------------------------------------- world draw
def draw_scaled(rng, D: int, T: int, scale: np.ndarray, subgroups, cell: dict,
                dropout: float, month_missing: float = 0.0) -> np.ndarray:
    """U1.decomp_power.v1's world (power_e02d.draw: same structure, same RNG order) with a
    per well-month scale (W, T) in metres; (D, W, T), NaN = missing. ``month_missing`` > 0
    then drops each remaining well-month independently (its draw comes last)."""
    import pandas as pd

    s = np.asarray(scale, float)
    if s.ndim == 1:
        s = s[:, None] * np.ones((1, T))
    s = s[None]
    W = s.shape[1]
    subs = pd.factorize(np.asarray(subgroups))[0]
    a_s, a_w, phi = cell["a_s"], cell["a_w"], cell["phi"]
    C = rng.standard_normal((D, 1, T))
    S = rng.standard_normal((D, subs.max() + 1, T))[:, subs, :]
    if cell["heavy"]:
        bad = rng.random((D, W, 1)) < U1P.P_BAD
        A = np.zeros((D, W, 1))
        mult = np.where(bad, U1P.SCALE_BAD, 1.0)
        a_w = 0.0
    else:
        A = rng.standard_normal((D, W, 1))
        mult = 1.0
    eps = rng.standard_normal((D, W, T))
    U = np.empty_like(eps)
    U[..., 0] = eps[..., 0]
    q = math.sqrt(max(0.0, 1.0 - phi ** 2))
    for t in range(1, T):
        U[..., t] = phi * U[..., t - 1] + q * eps[..., t]
    r = math.sqrt(max(0.0, 1.0 - a_s - a_w))
    x = s * mult * (math.sqrt(a_s) * S + math.sqrt(a_w) * A + r * U) + U1P.COMMON_SCALE * C
    if dropout > 0:
        keep = rng.random((D, W, 1)) >= dropout
        x = np.where(keep, x, np.nan)
    if month_missing > 0:
        keep = rng.random((D, W, T)) >= month_missing
        x = np.where(keep, x, np.nan)
    return x


def forcing_response(rng, D: int, G: np.ndarray, pool: np.ndarray) -> np.ndarray:
    """(D, W, T) realised-forcing responses sum_h (a_h - 1) G[h]: a = osse_fv.
    draw_multipliers(pool) per replicate (calendar-year block bootstrap of FV's pool)."""
    import osse_fv as FV

    a = np.stack([FV.draw_multipliers(pool, rng)[0] for _ in range(D)])
    if a.shape[1] != G.shape[0]:
        raise EngineeringFailure(f"anomaly horizon {a.shape[1]} != basis {G.shape[0]}")
    return np.tensordot(a - 1.0, G, axes=(1, 0))


# ----------------------------------------------------------------------------- statistic
def window_means(y: np.ndarray, groups, min_months: int = MIN_MONTHS,
                 min_others: int = MIN_OTHERS) -> tuple[np.ndarray, np.ndarray]:
    """(ybar, n): the mean over months of E02d's relative operator of ``y`` (..., W, T) and
    the number of months it used; ybar is NaN where n < ``min_months``."""
    e = E.relative(y, groups, min_others)
    fin = np.isfinite(e)
    n = fin.sum(axis=-1)
    s = np.where(fin, e, 0.0).sum(axis=-1)
    ybar = np.where(n >= min_months, s / np.maximum(n, 1), np.nan)
    return ybar, n


def mu_stack(mu: dict, tags=TRUTHS) -> np.ndarray:
    return np.stack([np.asarray(mu[t], float) for t in tags])


def realised_means(mus: np.ndarray, mask: np.ndarray, groups) -> np.ndarray:
    """(D, K, W) plug-in means on the realised mask (S4): mu_k (K, W, T) set missing where
    ``mask`` (D, W, T) is False, then the relative operator and window mean. A full mask
    gives the same means for every replicate (computed once)."""
    mask = np.asarray(mask, bool)
    D = mask.shape[0]
    if mask.all():
        m, _ = window_means(mus, groups)                       # (K, W)
        return np.broadcast_to(m[None], (D,) + m.shape)
    X = np.where(mask[None], mus[:, None], np.nan)             # (K, D, W, T)
    m, _ = window_means(X, groups)                             # (K, D, W)
    return np.transpose(m, (1, 0, 2))


def hypothesis_means(mu: dict, groups, tags=TRUTHS) -> np.ndarray:
    """(K, W) means on the full mask (engineering: finite at every well)."""
    m, _ = window_means(mu_stack(mu, tags), groups)
    if not np.isfinite(m).all():
        raise EngineeringFailure("a hypothesis mean is not finite at every well")
    return m


def calibrate(rng, T: int, cd: dict, cell: dict, n: int) -> np.ndarray:
    """v (W,): E[ybar_w^2] under a working cell with mu = 0 and no missingness."""
    x = draw_scaled(rng, n, T, cd["scale"], cd["subgroups"], cell, 0.0)
    ybar, _ = window_means(x, cd["groups"])
    v = np.nanmean(ybar ** 2, axis=0)
    if not np.isfinite(v).all():
        raise EngineeringFailure("calibration: a well has no finite window mean")
    return np.maximum(v, V_FLOOR)


def loglik(ybar: np.ndarray, n: np.ndarray, means: np.ndarray, v: np.ndarray, T: int
           ) -> tuple[np.ndarray, np.ndarray]:
    """(LL (D, K, G), scored wells (D,)). ybar, n: (D, W); means (K, W) or (D, K, W); v
    (G, W). A well is scored iff ybar is finite and n >= MIN_MONTHS; its variance is
    v * T / n."""
    ybar = np.atleast_2d(ybar)
    n = np.atleast_2d(n)
    D = ybar.shape[0]
    means = np.asarray(means, float)
    if means.ndim == 2:
        means = np.broadcast_to(means[None], (D,) + means.shape)
    ok = np.isfinite(ybar) & (n >= MIN_MONTHS)
    infl = np.where(ok, T / np.maximum(n, 1), 1.0)
    yb = np.where(ok, ybar, 0.0)
    K, G = means.shape[1], v.shape[0]
    out = np.empty((D, K, G))
    for g in range(G):
        ve = v[g][None, :] * infl
        const = np.where(ok, -0.5 * np.log(2.0 * math.pi * ve), 0.0).sum(axis=-1)
        for k in range(K):
            mk = np.where(ok, means[:, k, :], 0.0)
            r = np.where(ok, (yb - mk) ** 2 / ve, 0.0).sum(axis=-1)
            out[:, k, g] = const - 0.5 * r
    return out, ok.sum(axis=-1)


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))).squeeze(axis)


def log_marginal(LL: np.ndarray, idx) -> np.ndarray:
    """log p(ybar | H) for the members ``idx`` of H: equal member weights, uniform cells."""
    sub = LL[:, list(idx), :]
    flat = sub.reshape(sub.shape[0], -1)
    return _logsumexp(flat, axis=1) - math.log(flat.shape[1])


def log_bf(LL: np.ndarray, a_idx, b_idx) -> np.ndarray:
    """log BF(A:B) per replicate."""
    return log_marginal(LL, a_idx) - log_marginal(LL, b_idx)


def reading(lbf: float, n_scored: int, b: float = B_STRONG) -> str:
    """The registered BF readings (one county): A = {R0, Q0}, B = {R1c, X5R1}."""
    if n_scored < MIN_WELLS or not np.isfinite(lbf):
        return "not_evaluable"
    if lbf >= math.log(b):
        return "closer_to_R0Q0"
    if lbf <= -math.log(b):
        return "closer_to_R1cX5R1"
    return "not_decisive"


def score_pair(resid: np.ndarray, groups, mus: np.ndarray, v: np.ndarray, T: int,
               a_idx, b_idx) -> dict:
    """The BF statistic of ONE county window (resid = obs - ZR, (W, T), NaN = missing; mus
    (K, W, T) the hypothesis departures): the function the 2028 scorer wraps (same code path
    as the power study, means on the realised mask)."""
    r = np.asarray(resid, float)[None]
    ybar, n = window_means(r, groups)
    means = realised_means(np.asarray(mus, float), np.isfinite(r), groups)
    LL, k = loglik(ybar, n, means, v, T)
    lbf = float(log_bf(LL, a_idx, b_idx)[0])
    return {"log_bf": lbf, "bf": float(math.exp(max(min(lbf, 700.0), -700.0))),
            "wells_scored": int(k[0]), "reading": reading(lbf, int(k[0]))}


# ----------------------------------------------------------------------------- ledger
FAMILY_TEXT = {"R0Q0": "{R0,Q0}", "R1cX5R1": "{R1c,X5R1}"}
NOT_DECIDING = ("this does not decide whether Changhua pumping matters or the response class "
                "(N2C decides that)")
PRECEDENCE = ("if FB and N2C disagree, N2C stands; FB's line is printed beside N2C's "
              "verdict")
LEDGER_TEXT = {
    "closer_to_R0Q0": "{county}: the window's per-well relative pattern is closer to the "
                      "{R0,Q0} forwards (climatology forcing) (BF {bf_txt}); " + NOT_DECIDING,
    "closer_to_R1cX5R1": "{county}: the window's per-well relative pattern is closer to the "
                         "{R1c,X5R1} forwards (climatology forcing) (BF {bf_txt}); "
                         + NOT_DECIDING,
    "not_decisive": "{county}: the window does not decide between the {R0,Q0} and "
                    "{R1c,X5R1} forwards (BF {bf_txt}); " + NOT_DECIDING,
    "not_evaluable": "{county}: not evaluable (fewer than 20 scored wells); " + NOT_DECIDING}


def ledger_line(reading_: str, county: str, bf: float | None, n2c_verdict: str) -> str:
    """The pre-worded E29 ledger line, always beside N2C's verdict (lead ruling S1)."""
    bf_txt = "n/a" if bf is None or not np.isfinite(bf) else f"{bf:.3g}"
    base = LEDGER_TEXT[reading_].replace("{county}", county).replace("{bf_txt}", bf_txt)
    return f"{LEDGER_ROW}: {base}. Beside N2C's verdict '{n2c_verdict}': {PRECEDENCE}."


# ----------------------------------------------------------------------------- self-checks
def _loglik_reference(ybar, n, means, v, T) -> np.ndarray:
    D, W = ybar.shape
    means = np.broadcast_to(means[None], (D,) + means.shape) if means.ndim == 2 else means
    out = np.zeros((D, means.shape[1], v.shape[0]))
    for d in range(D):
        for k in range(means.shape[1]):
            for g in range(v.shape[0]):
                s = 0.0
                for w in range(W):
                    if np.isfinite(ybar[d, w]) and n[d, w] >= MIN_MONTHS:
                        ve = v[g, w] * T / n[d, w]
                        s += -0.5 * math.log(2 * math.pi * ve) \
                            - (ybar[d, w] - means[d, k, w]) ** 2 / (2 * ve)
                out[d, k, g] = s
    return out


def self_check(cd: dict, mus: np.ndarray, v: np.ndarray, seed: int, county_i: int,
               T: int = T_WIN, n: int = 4) -> dict:
    """Engineering checks on the run's own design (own stream; no power number): a county-
    month shock cancels; the vectorised log-likelihood (realised-mask means) equals a scalar
    loop; log BF is antisymmetric; identical hypotheses give log BF = 0."""
    rng = rng_for(seed, county_i, STREAM_SELF, 0, 0)
    cell = {"a_s": max(cd["a_s"]), "a_w": 0.0, "phi": 0.5, "heavy": False}
    x = draw_scaled(rng, n, T, cd["scale"], cd["subgroups"], cell, 0.2, MONTH_MISSING)
    shock = rng.standard_normal((n, 1, T)) * 5.0
    e1 = E.relative(x, cd["groups"])
    e2 = E.relative(x + shock, cd["groups"])
    both = np.isfinite(e1) & np.isfinite(e2)
    cancel = float(np.max(np.abs(e1[both] - e2[both]))) if both.any() else 0.0
    nan_same = bool((np.isfinite(e1) == np.isfinite(e2)).all())
    y = mus[0][None] + x
    ybar, nm = window_means(y, cd["groups"])
    means = realised_means(mus, np.isfinite(y), cd["groups"])
    LL, _ = loglik(ybar, nm, means, v, T)
    ref = _loglik_reference(ybar, nm, np.asarray(means), v, T)
    ll_err = float(np.max(np.abs(LL - ref)))
    K = mus.shape[0]
    a, b = [0], (list(range(1, K)) if K > 1 else [0])
    anti = float(np.max(np.abs(log_bf(LL, a, b) + log_bf(LL, b, a))))
    LL2, _ = loglik(ybar, nm, np.stack([means[:, 0], means[:, 0]], axis=1), v, T)
    ident = float(np.max(np.abs(log_bf(LL2, [0], [1]))))
    checks = {"common_shock_cancels_max_abs": cancel, "common_shock_nan_pattern_same": nan_same,
              "loglik_vs_reference_max_abs": ll_err, "antisymmetry_max_abs": anti,
              "identical_hypotheses_max_abs": ident}
    ok = (cancel <= SELF_TOL and nan_same and ll_err <= 1e-6 * max(1.0, float(np.max(np.abs(
        ref)))) and anti <= SELF_TOL and ident <= SELF_TOL)
    return {"ok": bool(ok), **checks}


# ----------------------------------------------------------------------------- power
def county_cd(design_c: dict, scale: np.ndarray) -> dict:
    """The per-county draw dict: scale = sigma_idio (W, T) in metres, U1's groups."""
    return {"scale": np.asarray(scale, float), "groups": list(design_c["groups"]),
            "subgroups": list(design_c["subgroups"]), "a_s": design_c["grid"]["a_s"]}


def _check_county_inputs(county: str, dc: dict, inp: dict, T: int) -> None:
    W = dc["wells"]
    sc = np.asarray(inp["scale"], float)
    if sc.shape != (W, T) or not np.all(np.isfinite(sc)) or np.any(sc <= 0):
        raise EngineeringFailure(f"{county}: sigma_idio scale is not a positive ({W}, {T})")
    for t in TRUTHS:
        if np.asarray(inp["mu"][t]).shape != (W, T) or not np.isfinite(inp["mu"][t]).all():
            raise EngineeringFailure(f"{county} {t}: mu is not a finite ({W}, {T})")
        G = np.asarray(inp["G"][t])
        if G.shape != (HORIZON, W, T) or not np.isfinite(G).all():
            raise EngineeringFailure(f"{county} {t}: basis is not a finite ({HORIZON}, {W}, "
                                     f"{T})")


def run_county(design: dict, county: str, inp: dict, pool: np.ndarray, seed: int, reps: int,
               cal: int, log=print) -> dict:
    """Calibration, means, self-check and every (truth, world cell) of a county."""
    ci = COUNTIES.index(county)
    dc = design["counties"][county]
    T = design["window"]["months"]
    _check_county_inputs(county, dc, inp, T)
    cd = county_cd(dc, inp["scale"])
    mus = mu_stack(inp["mu"])
    full = hypothesis_means(inp["mu"], cd["groups"])
    wcells = working_cells(dc["grid"])
    v = np.vstack([calibrate(rng_for(seed, ci, STREAM_CAL, 0, c["cell"]), T, cd, c, cal)
                   for c in wcells])
    sc = self_check(cd, mus, v, seed, ci, T)
    out = {"self_check": sc, "calibration": {"v": v.round(12).tolist(),
                                             "working_cells": wcells},
           "means_full_mask": {t: full[i].round(12).tolist() for i, t in enumerate(TRUTHS)},
           "mu": {t: mus[i].round(12).tolist() for i, t in enumerate(TRUTHS)},
           "separation_rms_m": {}, "rows": []}
    for p, a, b in PAIRS:
        ma = full[[TRUTHS.index(t) for t in a]].mean(0)
        mb = full[[TRUTHS.index(t) for t in b]].mean(0)
        out["separation_rms_m"][p] = float(np.sqrt(np.mean((ma - mb) ** 2)))
    if not sc["ok"]:
        return out
    cands = [c for c in candidates() if c["county"] == county]
    lnb = math.log(design["thresholds"]["b_strong"])
    for ti, t in enumerate(TRUTHS):
        for cell in world_cells(dc["grid"]):
            rng = rng_for(seed, ci, STREAM_WORLD, ti, cell["cell"])
            x = draw_scaled(rng, reps, T, cd["scale"], cd["subgroups"], cell, cell["dropout"],
                            cell["month_missing"]) * cell["scale"]
            y = mus[ti][None] + x
            if cell["forcing"]:
                y = y + forcing_response(rng_for(seed, ci, STREAM_FORCING, ti, cell["cell"]),
                                         reps, np.asarray(inp["G"][t], float), pool)
            ybar, n = window_means(y, cd["groups"])
            means = realised_means(mus, np.isfinite(y), cd["groups"])
            LL, k = loglik(ybar, n, means, v, T)
            ev = k >= MIN_WELLS
            row = {kk: cell[kk] for kk in ("arm", "cell", "gating", "heavy", "a_s", "a_w",
                                           "phi", "dropout", "scale", "month_missing")}
            row.update({"truth": t, "p_not_evaluable": float(np.mean(~ev)),
                        "by_candidate": {}})
            for c in cands:
                if t not in c["A"] and t not in c["B"]:
                    continue
                ai = [TRUTHS.index(s) for s in c["A"]]
                bi = [TRUTHS.index(s) for s in c["B"]]
                lbf = log_bf(LL, ai, bi) if t in c["A"] else log_bf(LL, bi, ai)
                row["by_candidate"][c["key"]] = {
                    "p_correct_strong": float(np.mean(ev & (lbf >= lnb))),
                    "p_wrong_strong": float(np.mean(ev & (lbf <= -lnb))),
                    "median_log_bf_true": float(np.median(lbf[ev])) if ev.any() else None}
            out["rows"].append(row)
        log(f"{county} truth {t}: {len(out['rows'])} rows")
    return out


_WHERE = ("arm", "truth", "cell", "a_s", "a_w", "phi", "dropout")


def _arm_summary(pairs) -> dict:
    out = {}
    for r, x in pairs:
        d = out.setdefault(r["arm"], {"p_correct_strong_min": 1.0, "p_wrong_strong_max": 0.0})
        d["p_correct_strong_min"] = min(d["p_correct_strong_min"], x["p_correct_strong"])
        d["p_wrong_strong_max"] = max(d["p_wrong_strong_max"], x["p_wrong_strong"])
    return out


def decide(rows: list[dict], key: str, thresholds: dict) -> dict:
    """One candidate over its gating rows: min P(correct strong) >= p_correct_min AND max
    P(wrong strong) <= p_wrong_max (worst truth x worst row, never pooled). A truth pair is
    report-only (lead ruling B1): its numbers are reported and it never passes."""
    gating_key = key in family_keys()
    g = [(r, r["by_candidate"][key]) for r in rows
         if r["gating"] and key in r["by_candidate"]]
    if not g:
        return {"evaluated": False, "pass": False, "report_only": not gating_key,
                "why": "no gating row"}
    pc = min(x["p_correct_strong"] for _, x in g)
    pw = max(x["p_wrong_strong"] for _, x in g)
    worst_c = min(g, key=lambda z: z[1]["p_correct_strong"])[0]
    worst_w = max(g, key=lambda z: z[1]["p_wrong_strong"])[0]
    ro = [(r, r["by_candidate"][key]) for r in rows
          if not r["gating"] and key in r["by_candidate"]]
    meets = bool(pc >= thresholds["p_correct_min"] and pw <= thresholds["p_wrong_max"])
    return {"evaluated": True, "rows": len(g), "report_only": not gating_key,
            "p_correct_strong_min": pc, "p_wrong_strong_max": pw,
            "worst_correct": {k: worst_c[k] for k in _WHERE},
            "worst_wrong": {k: worst_w[k] for k in _WHERE},
            "gating_by_arm": _arm_summary(g), "report_only_arms": _arm_summary(ro),
            "meets_thresholds_report_only": None if gating_key else meets,
            "pass": bool(gating_key and meets)}


def registration_rule(decisions: dict) -> dict:
    """Pre-declared (lead ruling B1): the BF test may be registered ONLY for the family pair
    of a county that passes, before 2026-11-30T16:00Z; a truth pair is never registrable."""
    bad = [k for k, d in decisions.items() if d.get("pass") and k not in family_keys()]
    if bad:
        raise ValueError(f"a non-family candidate is marked pass: {bad}")
    passing = sorted(k for k, d in decisions.items() if d.get("pass"))
    return {"registrable": passing, "any": bool(passing), "deadline": REG_DEADLINE,
            "only": family_keys(),
            "if_none": f"no BF test is registered; {LEDGER_ROW} records 'family BF not "
                       f"decidable by design ({POWER_ID})'; C20/C21 and every page line "
                       "unchanged"}


def run(design: dict, inputs: dict, seed: int, reps: int, cal: int, *, smoke: bool,
        log=print) -> dict:
    """The study. ``inputs``: {"pool": (Y, 12), county: {"mu": {truth: (W, T)}, "scale":
    (W, T), "G": {truth: (58, W, T)}, "ids": [...]}}. Raises EngineeringFailure when a
    self-check fails."""
    if seed != design["seed"]:
        raise EngineeringFailure(f"seed {seed} != the design's seed {design['seed']}")
    pool = np.asarray(inputs["pool"], float)
    if pool.ndim != 2 or pool.shape[1] != 12 or not np.isfinite(pool).all():
        raise EngineeringFailure("anomaly pool is not a finite (years, 12) table")
    res = {"id": POWER_ID, "version": VERSION, "smoke": smoke, "seed": seed,
           "replicates": reps, "cal_replicates": cal, "design": design, "counties": {},
           "decisions": {}}
    for c in COUNTIES:
        r = run_county(design, c, inputs[c], pool, seed, reps, cal, log)
        if not r["self_check"]["ok"]:
            raise EngineeringFailure(f"{c}: self-check failed {r['self_check']}")
        r["ids"] = list(inputs[c].get("ids", []))
        res["counties"][c] = r
        for cand in candidates():
            if cand["county"] == c:
                res["decisions"][cand["key"]] = decide(r["rows"], cand["key"],
                                                       design["thresholds"])
    res["registration"] = registration_rule(res["decisions"])
    res["wrong_side_disclosure"] = WRONG_SIDE_DISCLOSURE
    return res


# ----------------------------------------------------------------------------- design
def thresholds() -> dict:
    return {"b_strong": B_STRONG, "ln_b_strong": math.log(B_STRONG),
            "p_correct_min": P_CORRECT_MIN, "p_wrong_max": P_WRONG_MAX,
            "rule": "family pair only, per county: min over truths x gating rows (arms "
                    f"{list(GATING_ARMS)}) of P(correct strong) >= p_correct_min AND max of "
                    "P(wrong strong) <= p_wrong_max; heavy and scale cells report-only; truth "
                    "pairs report-only; not evaluable counts as neither",
            "wrong_side": WRONG_SIDE_DISCLOSURE}


def design_spec(u1_doc: dict, u1_sha: str, input_sha: dict, seed: int) -> dict:
    """The one source of the study's design: U1's registered grid and groups (copied from its
    declaration), the arms, candidates, thresholds, scorer constants, input sha256 and the
    seed (required, no default)."""
    u1d = u1_doc["design"]
    counties = {}
    for c in COUNTIES:
        uc = u1d["counties"][c]
        grid = county_grid(uc, u1d["dropout"])
        wc = world_cells(grid)
        counties[c] = {"wells": int(uc["wells"]), "groups": list(uc["groups"]),
                       "subgroups": list(uc["subgroups"]),
                       "u1_scale": [_f(x) for x in uc["scale"]], "grid": grid,
                       "n_working_cells": len(working_cells(grid)),
                       "n_world_cells": len(wc),
                       "n_gating_world_cells": sum(1 for x in wc if x["gating"])}
    return {
        "version": VERSION, "id": POWER_ID, "seed": int(seed),
        "window": {"first": WINDOW[0], "last": WINDOW[1], "months": T_WIN,
                   "fv_window_slice": list(FV_WINDOW_SLICE)},
        "truths": list(TRUTHS), "families": {k: list(v) for k, v in FAMILIES.items()},
        "family_pair": FAMILY_PAIR, "pairs": pair_table(),
        "candidates": [c["key"] for c in candidates()], "gating_candidates": family_keys(),
        "u1": {"id": U1_ID, "sha256": u1_sha, "dropout": [_f(x) for x in u1d["dropout"]],
               "p_bad": U1P.P_BAD, "scale_bad": U1P.SCALE_BAD,
               "common_scale": U1P.COMMON_SCALE},
        "noise_scale": f"E02d's window-A sigma_idio per well-month ({E02D_ID}, "
                       "sigma_idio_A.csv, metres): sd_fit_rel, ensemble and drift terms",
        "arms": {"gating": list(GATING_ARMS), "month_missing": MONTH_MISSING,
                 "scale_cells_report_only": list(SCALE_CELLS), "forcing_horizon": HORIZON},
        "counties": counties, "thresholds": thresholds(),
        "scorer": {"min_wells": MIN_WELLS, "min_months": MIN_MONTHS, "min_others": MIN_OTHERS,
                   "operator": "e02dlib.relative on DR's groups (U1's), then the per-well "
                               "window mean; hypothesis means on the realised mask",
                   "partial_well_variance": "v * T / n",
                   "prior": "equal prior odds; equal member weights within a family; uniform "
                            "over U1's a_s x a_w x phi working cells"},
        "replicates": REPLICATES, "cal_replicates": CAL_REPLICATES,
        "rng": "np.random.default_rng(SeedSequence(seed, spawn_key=(county index, stream, a, "
               "b))); stream 0 worlds (truth, world cell), 1 calibration (0, working cell), "
               "2 self-check, 3 forcing draws (truth, world cell)",
        "input_sha256": dict(sorted(input_sha.items()))}


# ----------------------------------------------------------------------------- synthetic
def synthetic_u1(seed: int) -> dict:
    """A SYNTHETIC stand-in for U1.decomp_power.v1's declaration (real counts and grid
    shapes, made-up groups and scales). Tests and smoke runs only; sets nothing."""
    rng = np.random.default_rng(seed)
    zones = {"Yunlin": ["prox", "mid", "dist"], "Changhua": ["prox", "mid+dist"]}
    counties = {}
    for c in COUNTIES:
        w = EXPECTED_WELLS[c]
        gl = [f"{c}:{z}" for z in zones[c]]
        sl = [f"{c}:{x}" for x in ("L1", "L2", "L34")]
        sc = np.exp(rng.normal(0.0, 0.3, w))
        counties[c] = {"wells": w, "groups": [gl[i % len(gl)] for i in range(w)],
                       "subgroups": [sl[(i // 2) % 3] for i in range(w)],
                       "scale": (sc / np.median(sc)).round(6).tolist(),
                       "a_s": [0.0, 0.07, 0.14], "a_w": [0.0, 0.3, 0.5], "phi": [0.0, 0.6]}
    return {"id": "SYNTHETIC.U1", "design": {"counties": counties,
                                             "dropout": [0.0, 0.2, 0.4]}}


def synthetic_inputs(design: dict, seed: int, sep_m: float = 0.5, med_sd_m: float = 0.3,
                     forcing_m: float = 0.05) -> dict:
    """SYNTHETIC departures, sigma_idio-like scales, bases and pool for ``design``: the
    {R1c, X5R1} family carries a well-specific departure of RMS ``sep_m`` (Changhua), small
    elsewhere. Tests and smoke only."""
    rng = np.random.default_rng(seed + 1)
    T = design["window"]["months"]
    out = {"pool": 1.0 + rng.normal(0.0, 0.1, (8, 12))}
    for c in COUNTIES:
        dc = design["counties"][c]
        W = dc["wells"]
        scale = np.asarray(dc["u1_scale"], float)[:, None] * med_sd_m * \
            np.linspace(1.0, 1.2, T)[None, :]
        base = rng.normal(0.0, 0.05, (W, 1)) * np.ones((1, T))
        pump = rng.normal(0.0, sep_m if c == "Changhua" else 0.05, (W, 1)) * \
            np.linspace(0.8, 1.2, T)[None, :]
        mu = {"R0": base + rng.normal(0.0, 0.01, (W, T)),
              "Q0": base + rng.normal(0.0, 0.01, (W, T)),
              "R1c": base + pump, "X5R1": base + 0.9 * pump}
        G = {t: rng.normal(0.0, forcing_m, (HORIZON, W, T)) / math.sqrt(HORIZON)
             for t in TRUTHS}
        out[c] = {"mu": mu, "scale": scale, "G": G,
                  "ids": [f"{'07' if c == 'Changhua' else '09'}{i:06d}" for i in range(W)]}
    return out
