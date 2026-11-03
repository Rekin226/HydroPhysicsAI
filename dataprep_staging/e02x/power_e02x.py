"""U1.ext_transfer_power.v1: synthetic P-gate of E02x's per-county departure-transfer test at
N3F's external wells, for the primary window (11 months, 2026-12..2027-10) and the declared
fallback (10 months, 2027-01..2027-10), each with its own decision.

Synthetic only: no observation, residual or window head is opened. The design is the
registered U1.decomp_power.v1 grid (DR-bounded: a_s, a_w, phi, dropout, coverage; DR's groups,
sub-groups and donor scales at the 158 wells) PLUS the departure-estimation error of the
transfer, and the external wells' metadata (county, DR group and sub-group from cell and
layer; wells_e02x.ext_table). ``design_spec`` is the one source of the design; ``run``
refuses unless the registered declaration carries exactly it and the seed passed equals the
registered seed. The seed is a required argument everywhere (no default).

World (per county, independently; replicate d, month t; all shocks N(0, 1)):

    donor v:    x_v = s_v ( sqrt(a_s) S[sub(v), t] + sqrt(a_w) A_v
                            + sqrt(1 - a_s - a_w) U_v[t] ) + K (C[t] + G[g(v), t])
    external i: x_i = s_i ( sqrt(a_s) S[sub(i), t] + sqrt(a_x) X[g(i), t] + sqrt(a_w) A_i
                            + sqrt(1 - a_s - a_x - a_w) U_i[t] ) + K (C[t] + G[g(i), t])

C a county-month and G a group-month shock shared by donors and external wells (the
departure the transfer removes; K = 3, they cancel exactly up to the donor median's own
error), S the sub-group (layer-group) shock DR's zone grouping leaves, A a persistent
per-well offset (lead drift), U a stationary AR(1) (phi), X an external-only group-month
shock (transfer mismatch: the uncalibrated wells' group departure differs from the donors').

Departure-estimation error enters two ways, both declared: (i) sampling error of the donor
median, simulated, not assumed (the donors are drawn with their own idiosyncratic parts and
drop out for the whole window with probability ``donor_dropout`` in U1.decomp_power.v1's
registered grid {0, 0.2, 0.4}; a group with < 5 reporting
donors leaves its external wells unscored, E02d's min_others); (ii) the mismatch share a_x in
{0, the county's top a_s} (the DR-bounded sub-group allowance of U1.decomp_power.v1).

External scales s_i: the donor county's DR scale vector (sd_fit_rel / county median) at the
quantiles (k - 0.5) / n, assigned in station-id order: the external sd_fit_rel is NOT read
before this study (it is read afterwards, logged, by sdfit_e02x.py), so the design cannot
depend on it.

Scorer: EXACTLY E02x's (e02xlib.transfer_hits on the DR groups, min donors 5, then E02d's
county_verdict: >= 6 months, >= 20 wells, coverage3 gate-3 upper90 clusters = wells AND point
coverage in [0.85, 0.95]); Changhua's 23 wells against the same 20-well floor (20 of 23). The
vectorised path is checked against county_verdict (self-check). True coverage p: the frozen
half-width z s_i k_p with k_p the p-quantile of |e| / (z s_i) over a calibration sample of the
same cell WITHOUT any dropout.

Calibration (disclosed): k_p is calibrated without dropout, so false-fail is conservative and
power optimistic (``CALIBRATION_DISCLOSURE``).

Arithmetic (lead ruling B1, constants only): with the 20-well floor and external dropout up to
0.4, P(evaluable) bounds the power rule's minimum: Changhua (23 wells) 0.297 at 0.2 and 0.005
at 0.4, so it cannot pass by construction and is report-only by arithmetic (it is still run,
report-only); Yunlin (37) 0.818 at 0.4, so its power is at most 0.818 x the conditional power
(``e02xlib.arithmetic_bounds``, recorded in the design).

Decision (per window, per county): FF = max P(FAIL) over gating cells at p = 0.90; PW = min
P(FAIL) over gating cells at p = 0.75; the county passes iff FF <= 0.12 and PW >= 0.80 (a
not-evaluable replicate is neither a fail nor a pass, as in U1.decomp_power.v1).
Registration (pre-declared): Yunlin passes -> E02x may be registered for that window with
Yunlin gated, otherwise E02x is NOT registered for that window (C14f-r N, 'undecidable by
design at the external wells'); Changhua report-only by arithmetic (gated only if its bound
reached 0.80 AND it passed, which the declared constants exclude);
Nantou is report-only under every outcome (R8.3).

    python -I dataprep_staging/e02x/power_e02x.py smoke --seed <int> --out <dir outside results/>
        (SYNTHETIC inputs only: the real design is never computed before registration)
    python -I dataprep_staging/e02x/power_e02x.py run --seed <the registered seed>
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent), str(HERE.parent / "e02d")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import e02xio as io  # noqa: E402
import e02xlib as L  # noqa: E402

VERSION = "1.0.0"
CODE_FILES = (f"{io.X}/power_e02x.py", f"{io.X}/e02xlib.py", f"{io.X}/e02xio.py",
              f"{io.X}/wells_e02x.py", "dataprep_staging/e02d/e02dlib.py",
              "dataprep_staging/harness/coverage3.py", "dataprep_staging/harness/coverage2.py",
              "dataprep_staging/harness/coverage.py")
RESULT_DIR = f"results/dataprep/reports/{L.POWER_ID}"
COVERAGE = (0.75, 0.80, 0.90)
PHI_CAP = 0.95
P_BAD, SCALE_BAD = 0.10, 3.0
COMMON_SCALE = 3.0
REPLICATES = 2000
CAL_REPLICATES = 500
SMOKE = {"replicates": 60, "cal_replicates": 40}
WINDOW_ORDER = ("primary", "fallback")
CALIBRATION_DISCLOSURE = (
    "k_p (the true-coverage calibration) is drawn WITHOUT external or donor dropout, while the "
    "scored replicates drop wells and donors: dropout thins the donor median and adds noise to "
    "e, so a band calibrated to a true p covers less than p under dropout. The false-fail rate "
    "at p = 0.90 is therefore conservative (over-stated) and the power at p = 0.75 optimistic "
    "(over-stated)")


def _uniq(xs) -> list[float]:
    out = []
    for x in xs:
        x = round(float(x), 6)
        if x not in out:
            out.append(x)
    return sorted(out)


# ----------------------------------------------------------------------------- design
def county_design(u1c: dict, ext: pd.DataFrame, county: str) -> dict:
    """One county's design: U1.decomp_power.v1's donors and grid, the external wells and the
    departure-estimation axes (declared derivation)."""
    sd = np.asarray(u1c["scale"], float)
    e = ext[(ext["county"] == county) & (ext["role"] == "gate_candidate")].sort_values(
        "station_id")
    n = len(e)
    q = (np.arange(n) + 0.5) / n if n else np.array([])
    s_ext = np.quantile(np.sort(sd), q) if n else np.array([])
    a_s = _uniq(u1c["a_s"])
    return {
        "donors": {"wells": int(len(sd)), "groups": list(map(str, u1c["groups"])),
                   "subgroups": list(map(str, u1c["subgroups"])),
                   "scale": [round(float(x), 6) for x in sd]},
        "ext": {"wells": int(n), "station_id": e["station_id"].astype(str).tolist(),
                "groups": e["group"].astype(str).tolist(),
                "subgroups": e["subgroup"].astype(str).tolist(),
                "scale": [round(float(x), 6) for x in s_ext],
                "scale_rule": "donor scale quantiles (k - 0.5) / n, station-id order"},
        "a_s": a_s, "a_w": _uniq(u1c["a_w"]),
        "phi": _uniq(min(PHI_CAP, max(0.0, float(p))) for p in u1c["phi"]),
        "a_x": _uniq([0.0, max(a_s)]),
        "ext_dropout": _uniq(u1c.get("dropout_u1") or []),
        "donor_dropout": _uniq(u1c.get("dropout_u1") or []),     # = U1's grid (lead, S2)
        "from_u1": {k: u1c["from_dr"][k] for k in sorted(u1c["from_dr"])}}


def design_spec(u1: dict, ext: pd.DataFrame, seed: int, u1_sha: str = "",
                ext_prov: dict | None = None) -> dict:
    """The registered design (pure): U1.decomp_power.v1's grid (``u1`` = its registered
    ``design``) plus the transfer axes, both windows, the seed."""
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an int (a required argument, no default)")
    counties = {}
    for c in L.GATED:
        u1c = dict(u1["counties"][c])
        u1c["dropout_u1"] = u1["dropout"]
        counties[c] = county_design(u1c, ext, c)
        if counties[c]["ext"]["wells"] != L.EXPECTED_EXT[c]:
            raise io.EngineeringFailure(f"{c}: {counties[c]['ext']['wells']} external wells, "
                                        f"expected {L.EXPECTED_EXT[c]}")
        bad = sorted(set(counties[c]["ext"]["groups"]) - set(counties[c]["donors"]["groups"]))
        if bad:
            raise io.EngineeringFailure(f"{c}: external groups without donors {bad}")
    return io.jsonable({
        "version": VERSION, "seed": seed, "self_check_seed": seed + 1,
        "u1_decomp_power": {"id": L.U1D_ID, "sha256": u1_sha,
                            "dr_report_sha256": u1.get("dr_report_sha256"),
                            "wells_e02d_sha256": u1.get("wells_e02d_sha256")},
        "ext_wells": ext_prov or {},
        "counties": counties,
        "arithmetic_bounds": L.arithmetic_bounds(
            {c: counties[c]["ext"]["wells"] for c in L.GATED}, u1["dropout"]),
        "calibration_disclosure": CALIBRATION_DISCLOSURE,
        "windows": {w: {"months": L.WINDOWS[w]["months"], "first": L.WINDOWS[w]["first"],
                        "last": L.WINDOWS[w]["last"]} for w in WINDOW_ORDER},
        "coverage": list(COVERAGE),
        "heavy_report_only": {"p_bad": P_BAD, "scale_bad": SCALE_BAD,
                              "cells": "a_s = max, phi = max, a_x = max, ext dropout {0, max}, "
                                       "donor dropout 0, coverage {0.75, 0.90}"},
        "common_scale": COMMON_SCALE, "replicates": REPLICATES,
        "cal_replicates": CAL_REPLICATES,
        "scorer": {"min_donors": L.MIN_DONORS, "min_months": L.MIN_MONTHS,
                   "min_wells": L.MIN_WELLS_COUNTY, "gate": L.D.gate_kwargs(),
                   "point_window_required": True, "z": L.Z90,
                   "changhua_floor": f"{L.MIN_WELLS_COUNTY} of {L.EXPECTED_EXT['Changhua']}"},
        "decision": {"false_fail_max": L.FALSE_FAIL_MAX, "false_fail_at": L.FF_AT,
                     "power_min": L.POWER_MIN, "power_at": L.POWER_AT,
                     "fail": "evaluable AND NOT (gate-3 pass AND point coverage in "
                             "[0.85, 0.95])",
                     "not_evaluable": "reported; it is neither a fail nor a pass"},
        "rng": "np.random.default_rng(SeedSequence(seed, spawn_key=(window index, county "
               "index, cell index, 0 draws | 1 calibration)))"})


def cells(cd: dict) -> list[dict]:
    out = []
    for a_s in cd["a_s"]:
        for a_w in cd["a_w"]:
            for phi in cd["phi"]:
                for a_x in cd["a_x"]:
                    if a_s + a_x + a_w > 1.0:
                        raise io.EngineeringFailure("a_s + a_x + a_w > 1")
                    for de in cd["ext_dropout"]:
                        for dd in cd["donor_dropout"]:
                            for p in COVERAGE:
                                out.append({"a_s": a_s, "a_w": a_w, "phi": phi, "a_x": a_x,
                                            "ext_dropout": de, "donor_dropout": dd,
                                            "coverage": p, "heavy": False, "gating": True})
    for de in (0.0, max(cd["ext_dropout"])):
        for p in (0.75, 0.90):
            out.append({"a_s": max(cd["a_s"]), "a_w": 0.0, "phi": max(cd["phi"]),
                        "a_x": max(cd["a_x"]), "ext_dropout": de, "donor_dropout": 0.0,
                        "coverage": p, "heavy": True, "gating": False})
    for i, c in enumerate(out):
        c["cell"] = i
    return out


# ----------------------------------------------------------------------------- world
def _ar1(rng, shape, phi: float) -> np.ndarray:
    eps = rng.standard_normal(shape)
    U = np.empty_like(eps)
    U[..., 0] = eps[..., 0]
    q = math.sqrt(max(0.0, 1.0 - phi ** 2))
    for t in range(1, shape[-1]):
        U[..., t] = phi * U[..., t - 1] + q * eps[..., t]
    return U


def _codes(cd: dict) -> dict:
    g_all = sorted(set(cd["donors"]["groups"]) | set(cd["ext"]["groups"]))
    s_all = sorted(set(cd["donors"]["subgroups"]) | set(cd["ext"]["subgroups"]))
    gi = {g: i for i, g in enumerate(g_all)}
    si = {s: i for i, s in enumerate(s_all)}
    return {"ng": len(g_all), "ns": len(s_all),
            "gd": np.array([gi[g] for g in cd["donors"]["groups"]], int),
            "ge": np.array([gi[g] for g in cd["ext"]["groups"]], int),
            "sd": np.array([si[s] for s in cd["donors"]["subgroups"]], int),
            "se": np.array([si[s] for s in cd["ext"]["subgroups"]], int)}


def masks_keys(cd: dict) -> tuple[dict, list]:
    dg = np.asarray(cd["donors"]["groups"], dtype=str)
    keys = [f"group:{g}" for g in cd["ext"]["groups"]]
    return {k: dg == k.split(":", 1)[1] for k in sorted(set(keys))}, keys


def draw(rng, n: int, T: int, cd: dict, cell: dict, ext_drop: float, donor_drop: float
         ) -> tuple[np.ndarray, np.ndarray]:
    """(x_donor (n, Wd, T), x_ext (n, We, T)) synthetic residuals (NaN = dropped well)."""
    k = _codes(cd)
    sd = np.asarray(cd["donors"]["scale"], float)[None, :, None]
    se = np.asarray(cd["ext"]["scale"], float)[None, :, None]
    Wd, We = sd.shape[1], se.shape[1]
    a_s, a_w, a_x, phi = cell["a_s"], cell["a_w"], cell["a_x"], cell["phi"]
    C = rng.standard_normal((n, 1, T))
    G = rng.standard_normal((n, k["ng"], T))
    S = rng.standard_normal((n, k["ns"], T))
    X = rng.standard_normal((n, k["ng"], T))
    if cell["heavy"]:
        md = np.where(rng.random((n, Wd, 1)) < P_BAD, SCALE_BAD, 1.0)
        me = np.where(rng.random((n, We, 1)) < P_BAD, SCALE_BAD, 1.0)
        Ad, Ae, a_w = np.zeros((n, Wd, 1)), np.zeros((n, We, 1)), 0.0
    else:
        md = me = 1.0
        Ad, Ae = rng.standard_normal((n, Wd, 1)), rng.standard_normal((n, We, 1))
    Ud, Ue = _ar1(rng, (n, Wd, T), phi), _ar1(rng, (n, We, T), phi)
    rd = math.sqrt(max(0.0, 1.0 - a_s - a_w))
    re_ = math.sqrt(max(0.0, 1.0 - a_s - a_x - a_w))
    xd = (sd * md * (math.sqrt(a_s) * S[:, k["sd"], :] + math.sqrt(a_w) * Ad + rd * Ud)
          + COMMON_SCALE * (C + G[:, k["gd"], :]))
    xe = (se * me * (math.sqrt(a_s) * S[:, k["se"], :] + math.sqrt(a_x) * X[:, k["ge"], :]
                     + math.sqrt(a_w) * Ae + re_ * Ue)
          + COMMON_SCALE * (C + G[:, k["ge"], :]))
    if donor_drop > 0:
        xd = np.where(rng.random((n, Wd, 1)) >= donor_drop, xd, np.nan)
    if ext_drop > 0:
        xe = np.where(rng.random((n, We, 1)) >= ext_drop, xe, np.nan)
    return xd, xe


def calibrate(rng, T: int, cd: dict, cell: dict, n: int) -> float:
    """k_p: the p-quantile of |e| / (z s_i) without any dropout."""
    xd, xe = draw(rng, n, T, cd, cell, 0.0, 0.0)
    m, keys = masks_keys(cd)
    e = L.transfer_relative(xe, xd, m, keys)
    s = np.asarray(cd["ext"]["scale"], float)[None, :, None]
    u = (np.abs(e) / (L.Z90 * s))[np.isfinite(e)]
    return float(np.quantile(u, cell["coverage"]))


def score_batch(xd: np.ndarray, xe: np.ndarray, cd: dict, k: float) -> dict:
    """Vectorised E02x verdicts of a batch: every scored external well has all T months
    (whole-window dropout of wells and donors), so gate-3 runs on per-well totals, batched by
    the well count."""
    from harness import coverage3

    s = np.asarray(cd["ext"]["scale"], float)[None, :, None]
    m, keys = masks_keys(cd)
    hits, _, _ = L.transfer_hits(xe, xd, s * k + 0 * xe, m, keys)
    D_, W, T = hits.shape
    nm = np.isfinite(hits).sum(axis=2)
    scored = nm >= L.MIN_MONTHS
    kw = scored.sum(axis=1)
    fail = np.zeros(D_, bool)
    ne = kw < L.MIN_WELLS_COUNTY
    cov = np.full(D_, np.nan)
    g = L.D.gate_kwargs()
    for kk in np.unique(kw[~ne]):
        ds = np.flatnonzero((kw == kk) & ~ne)
        tot = np.nansum(np.where(scored[ds][:, :, None], hits[ds], 0.0), axis=2)
        Y = np.stack([tot[i][scored[d]] for i, d in enumerate(ds)])
        mm = np.stack([nm[d][scored[d]] for d in ds]).astype(float)
        if not np.all(mm == mm[0]):
            raise io.EngineeringFailure("unequal month counts in a batch")
        r = coverage3.gate_from_totals_batch(Y, mm[0], g["method"], g["target"], g["level"],
                                            g["window"], g["n_eff_window_min"], g["conf"])
        inw = (r["p"] >= L.POINT_WINDOW[0]) & (r["p"] <= L.POINT_WINDOW[1])
        fail[ds] = ~(r["pass"] & inw)
        cov[ds] = r["p"]
    return {"fail": fail & ~ne, "not_evaluable": ne, "coverage": cov, "hits": hits}


def run_cell(cd: dict, cell: dict, T: int, wi: int, ci: int, reps: int, cal: int,
             seed: int) -> dict:
    key = (wi, ci, cell["cell"])
    rc = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=key + (1,)))
    k = calibrate(rc, T, cd, cell, cal)
    rd = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=key + (0,)))
    fails, nes, covs = [], [], []
    done = 0
    while done < reps:
        n = min(500, reps - done)
        xd, xe = draw(rd, n, T, cd, cell, cell["ext_dropout"], cell["donor_dropout"])
        sb = score_batch(xd, xe, cd, k)
        fails.append(sb["fail"])
        nes.append(sb["not_evaluable"])
        covs.append(sb["coverage"])
        done += n
    f, ne, cv = np.concatenate(fails), np.concatenate(nes), np.concatenate(covs)
    p = float(f.mean())
    return {**cell, "months": T, "k_p": k, "replicates": reps, "P_fail": p,
            "mc_se": math.sqrt(max(p * (1 - p), 0.0) / reps),
            "P_not_evaluable": float(ne.mean()),
            "mean_coverage_hat": float(np.nanmean(cv)) if np.isfinite(cv).any() else None}


def self_check(cd: dict, seed: int, T: int = 11, n: int = 12) -> dict:
    """The vectorised verdicts equal E02d's county_verdict on the same replicates, and a
    shock shared by a group's donors and external wells cancels (identical hits)."""
    rng = np.random.default_rng(seed)
    cell = {"a_s": max(cd["a_s"]), "a_w": max(cd["a_w"]), "phi": max(cd["phi"]),
            "a_x": max(cd["a_x"]), "heavy": False, "coverage": 0.85}
    xd, xe = draw(rng, n, T, cd, cell, 0.2, 0.2)
    sb = score_batch(xd, xe, cd, 1.0)
    mism = 0
    for d in range(n):
        v = L.county_verdict(sb["hits"][d], np.ones(xe.shape[1], bool))
        fail_ref, ne_ref = v["verdict"] == "fail", v["verdict"] == "not_evaluable"
        mism += int(fail_ref != bool(sb["fail"][d]) or ne_ref != bool(sb["not_evaluable"][d])
                    or (not ne_ref and abs(v["coverage"] - sb["coverage"][d]) > 1e-12))
    rng2 = np.random.default_rng(seed + 1)
    yd, ye = draw(rng2, 4, T, cd, cell, 0.0, 0.0)
    k = _codes(cd)
    shift = 7.0 * np.random.default_rng(seed + 2).standard_normal((4, k["ng"], T))
    h1 = score_batch(yd, ye, cd, 1.0)["hits"]
    h2 = score_batch(yd + shift[:, k["gd"], :], ye + shift[:, k["ge"], :], cd, 1.0)["hits"]
    same = bool(np.array_equal(np.isnan(h1), np.isnan(h2))
                and np.array_equal(np.nan_to_num(h1), np.nan_to_num(h2)))
    return {"parity_mismatches": mism, "replicates": n, "group_shock_cancels": same,
            "ok": mism == 0 and same}


def decide(rows: list[dict]) -> dict:
    g = [r for r in rows if r["gating"]]
    ff = [r for r in g if abs(r["coverage"] - L.FF_AT) < 1e-9]
    pw = [r for r in g if abs(r["coverage"] - L.POWER_AT) < 1e-9]
    wf = max(ff, key=lambda r: r["P_fail"])
    wp = min(pw, key=lambda r: r["P_fail"])
    ok = wf["P_fail"] <= L.FALSE_FAIL_MAX and wp["P_fail"] >= L.POWER_MIN
    return {"false_fail_max": wf["P_fail"], "false_fail_cell": wf["cell"],
            "power_min_at_0.75": wp["P_fail"], "power_cell": wp["cell"],
            "power_min_at_0.80_report_only": min(
                (r["P_fail"] for r in g if abs(r["coverage"] - 0.80) < 1e-9), default=None),
            "heavy_report_only": {r["cell"]: r["P_fail"] for r in rows if r["heavy"]},
            "p_gate_pass": bool(ok)}


def registration_rule(decisions: dict, bounds: dict | None = None) -> dict:
    """Per window: {registrable, roles} from the per-county decisions (pre-declared). A county
    that is report-only by arithmetic (``bounds``, e02xlib.arithmetic_bounds) stays
    report-only whatever its simulated decision (lead ruling B1: Changhua)."""
    bounds = bounds if bounds is not None else L.arithmetic_bounds()
    out = {}
    for w, dec in decisions.items():
        y = bool(dec["Yunlin"]["p_gate_pass"]) and not bounds["Yunlin"][
            "report_only_by_arithmetic"]
        ch = (bool(dec["Changhua"]["p_gate_pass"])
              and not bounds["Changhua"]["report_only_by_arithmetic"])
        roles = {"Yunlin": "primary" if y else "not_registered",
                 "Changhua": ("primary" if y and ch else "report-only" if y
                              else "not_registered"),
                 "Nantou": "report-only" if y else "not_registered"}
        out[w] = {"registrable": y, "roles": roles}
    return out


def run(design: dict, reps: int, cal: int, seed: int, *, smoke: bool, log=print) -> dict:
    if seed != design["seed"]:
        raise io.Refused(f"seed {seed} != the design's seed {design['seed']}")
    t0 = time.time()
    res = {"id": L.POWER_ID, "smoke": smoke, "design": design, "seed": seed,
           "replicates": reps, "cal_replicates": cal,
           "created": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "code_sha256": {f: io.sha256(io.repo_path(f)) for f in CODE_FILES},
           "self_check": {}, "results": {}, "decisions": {}}
    for c in L.GATED:
        res["self_check"][c] = self_check(design["counties"][c], design["self_check_seed"])
    res["engineering_failure"] = not all(v["ok"] for v in res["self_check"].values())
    if res["engineering_failure"]:
        return res
    for wi, w in enumerate(WINDOW_ORDER):
        T = design["windows"][w]["months"]
        for ci, c in enumerate(L.GATED):
            cd = design["counties"][c]
            rows = [run_cell(cd, cell, T, wi, ci, reps, cal, seed) for cell in cells(cd)]
            res["results"].setdefault(w, {})[c] = rows
            res["decisions"].setdefault(w, {})[c] = decide(rows)
            log(f"{w} {c} {T} months: {res['decisions'][w][c]['p_gate_pass']} "
                f"({time.time() - t0:.0f} s)")
    res["registration"] = registration_rule(res["decisions"], design["arithmetic_bounds"])
    res["runtime_s"] = round(time.time() - t0, 1)
    return res


def synthetic_inputs(seed: int) -> tuple[dict, str, pd.DataFrame, dict]:
    """A SYNTHETIC stand-in for (U1.decomp_power.v1's design, the external-well table): the
    real counts and DR-like grids, made-up scales and groups. Smoke runs and tests only; no
    number from it sets anything."""
    rng = np.random.default_rng(seed)
    u1 = {"dropout": [0.0, 0.2, 0.4], "dr_report_sha256": "synthetic",
          "wells_e02d_sha256": "synthetic", "counties": {}}
    rows = []
    zones = {"Yunlin": ["prox", "mid", "dist"], "Changhua": ["prox", "mid+dist"]}
    nd = {"Yunlin": 84, "Changhua": 74}
    for c in L.GATED:
        gl = [f"{c}:{z}" for z in zones[c]]
        sl = [f"{c}:{x}" for x in ("L1", "L2", "L34")]
        sc = np.exp(rng.normal(0, 0.3, nd[c]))
        u1["counties"][c] = {
            "groups": [gl[i % len(gl)] for i in range(nd[c])],
            "subgroups": [sl[i % 3] for i in range(nd[c])],
            "scale": (sc / np.median(sc)).round(6).tolist(),
            "a_s": [0.0, 0.07, 0.14], "a_w": [0.0, 0.3, 0.5], "phi": [0.0, 0.6],
            "from_dr": {"grouping": "zone", "icc_remaining": 0.07}}
        for i in range(L.EXPECTED_EXT[c]):
            rows.append({"station_id": f"{'09' if c == 'Yunlin' else '07'}9{i:05d}",
                         "county": c, "group": gl[i % len(gl)], "subgroup": sl[i % 3],
                         "role": "gate_candidate"})
    for i in range(L.EXPECTED_EXT["Nantou"]):
        rows.append({"station_id": f"089{i:05d}", "county": "Nantou", "group": "Nantou:prox",
                     "subgroup": "Nantou:L1", "role": "report_only"})
    return u1, "synthetic", pd.DataFrame(rows), {"synthetic": True}


def load_inputs(prereg_dir=None) -> tuple[dict, str, pd.DataFrame, dict]:
    import wells_e02x as WX

    u1, u1_sha = WX.u1_design(prereg_dir)
    ext, prov = WX.ext_table(prereg_dir)
    return u1, u1_sha, ext, prov


def _reorder(inputs, seed: int) -> tuple:
    u1, u1_sha, ext, prov = inputs
    return u1, ext, seed, u1_sha, prov


def main(argv=None, prereg_dir=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("smoke", help="tiny replicate counts on SYNTHETIC inputs only (sets "
                                     "nothing; the real design is never run before it is "
                                     "registered)")
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("run", help="the registered study (refuses unless registered)")
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "smoke":
            out = a.out.resolve()
            res_root = io.repo_path("results").resolve()
            if out == res_root or res_root in out.parents:
                raise io.Refused("a smoke run never writes under results/")
            design = design_spec(*_reorder(synthetic_inputs(a.seed), a.seed))
            res = run(design, SMOKE["replicates"], SMOKE["cal_replicates"], a.seed,
                      smoke=True)
            name = "smoke_result.json"
        else:
            u1, u1_sha, ext, prov = load_inputs(prereg_dir)
            out = a.out or io.repo_path(RESULT_DIR)
            doc, sha = io.registered(L.POWER_ID, prereg_dir)
            design = design_spec(u1, ext, a.seed, u1_sha, prov)
            if doc.get("design") != design:
                raise io.Refused("the registered design differs from design_spec(inputs, "
                                 "seed): wrong seed or changed inputs")
            now = {f: io.sha256(io.repo_path(f)) for f in CODE_FILES}
            if doc.get("code_sha256") != now:
                raise io.Refused("bound code changed since registration")
            if (out / "result.json").exists():
                raise io.Refused("result.json exists (run once)")
            res = run(design, REPLICATES, CAL_REPLICATES, a.seed, smoke=False)
            res["declaration"] = {"id": L.POWER_ID, "sha256": sha}
            name = "result.json"
        h = io.write_once(out / name, res)
        print(f"wrote {out / name} sha256 {h}; engineering failure "
              f"{res['engineering_failure']}")
        for w, d in res.get("decisions", {}).items():
            print(w, {c: v["p_gate_pass"] for c, v in d.items()})
        return 1 if res["engineering_failure"] else 0
    except io.Refused as exc:
        print(f"REFUSED: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
