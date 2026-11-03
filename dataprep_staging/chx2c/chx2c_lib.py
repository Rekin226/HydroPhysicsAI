"""Constants and pure functions of I1.CHX.v1's single exploratory follow-up (round 8 rank 8,
user ruling R8.10): the two-contrast narrowing on the 23 spent Changhua screens.

The question: does the direction pass of I1.CHX.v1 survive when the Yunlin placebo contrast,
which the panel already carries, enters the same regression? It can only NARROW E25's
wording; it can never promote anything, and E25's status is unchanged in every branch.

What is fixed here (and hash-bound by the declaration, ``prereg_chx2c.py``):

* dependent variable: CHX's own, e = obs - B (B = Changhua held at its 2012 pattern);
* regressors: c_CH = A - B (CHX's primary contrast, panel key ``c_ch``) and c_YL = A - C
  (the Yunlin placebo contrast, panel key ``c_placebo_ch``);
* estimator: CHX's, unchanged: rows from ``chx_lib.long_panel``/``eligible`` (flags 0-1,
  >= 12 scored months per screen), the 'twoway' fixed effects of ``chx_lib.fe_basis``
  projected out of e and of BOTH contrasts (exact least squares, ``chx_lib.fe_resid``), the
  pinned scorer's per-well demeaning (``bootstrap_scorer.within``, a no-op after the
  transform, kept so one regressor is bit-for-bit ``beta_response``), then least squares on
  the per-well cross-product sums;
* interval: the pinned 5 km block bootstrap, unchanged (CHX's blocks, the 07100111-141 nest
  as one block, B 2000, seed 20261101, the same ``default_rng(seed).integers(0, n_blocks,
  (B, n_blocks))`` draw over the sorted block ids, 5 % / 95 % quantiles, numpy's default
  rule); each draw solves the 2 x 2 normal equations of the resampled block sums; singular
  draws are dropped and counted (one regressor: exactly the pinned zero-denominator rule);
* collinearity rule (model only, computed before any head is read): the cosine of the two
  TRANSFORMED contrasts over every screen x window month (23 x 120, the same full grid as
  CHX's informative gate; after the two-way transform each has mean 0, so this is their
  Pearson correlation). If |r| > 0.9 (or r is not finite) the reading is fixed to 'not
  separable': nothing is fitted, no look is logged and no head is read;
* readings (one-sided, narrowing only): the Changhua coefficient's 90 % interval lies wholly
  above 0 (lo > 0, point > 0) -> 'direction survives a Yunlin spill-over control
  (after-the-fact, exploratory)'; anything else -> 'direction not separable from Yunlin
  spill-over'.

Why the offset does not matter for the Changhua coefficient: under linear superposition the
both-counties-at-2012 offset is D = B + C - A, and obs - B = (obs - D) - c_YL. Subtracting a
regressor from the dependent variable shifts only that regressor's coefficient (by exactly
-1, draw by draw), so b_CH and its interval are the same with either offset; the Yunlin
coefficient here is b_YL = k_YL - 1 (k_YL = the observed response to Yunlin's pattern in units
of the model's own), and k_YL = b_YL + 1 is reported beside it (report only).

No function here reads a file.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STAGING = HERE.parent
for _p in (str(STAGING / "ch_x"), str(STAGING / "n2_policy"), str(STAGING)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bootstrap_scorer as BS  # noqa: E402
import chx_const as K  # noqa: E402
import chx_lib as L  # noqa: E402

# ----------------------------------------------------------------------------- identity
ID = "I1.CHX.v1.twocontrast"
PARENT_ID = K.CHX_ID                              # I1.CHX.v1
PARENT_TEST = K.CHX_TEST
WP = K.WP                                         # I1
UNIT = K.CHX_UNIT                                 # I1CHX:ext_changhua_23 (spent)
SCOPE = K.CHX_SCOPE                               # heads:2013-01..2022-12
D = "dataprep_staging/chx2c"
LIB = f"{D}/chx2c_lib.py"
SCORER = f"{D}/score_chx2c.py"
PREREG = f"{D}/prereg_chx2c.py"
TESTS = "tests/test_chx2c.py"
RULINGS = "results/dataprep/rulings/user_rulings_2026-11-03.json"
RULING_ID = "R8.10"

#: the parent's files, read only (never written by this code)
PARENT_PREREG = f"results/dataprep/preregistration/{PARENT_ID}.json"
PARENT_REPORT = f"results/dataprep/reports/{PARENT_ID}/report.json"
PANEL = "results/twin_runs/chx/hindcast/chx_panel.npz"
PANEL_SUMMARY = "results/twin_runs/chx/hindcast/chx_panel.json"
#: this follow-up's own outputs (new files; the parent's report.json is never written)
REPORT = f"results/dataprep/reports/{PARENT_ID}/twocontrast_report.json"
ATTEMPTS = f"results/dataprep/reports/{PARENT_ID}/twocontrast_attempts.jsonl"
if REPORT == PARENT_REPORT or Path(REPORT).name == "report.json":
    raise RuntimeError("the two-contrast report must not be the parent's report.json")

#: the parent's frozen files this code imports or calls, at the sha256 I1.CHX.v1 froze
PARENT_FROZEN = {
    f"{K.X}/chx_const.py": "0808daf6c5ef78fc41d1606bf53461e2831b27b14690c6792046fa35e3414c46",
    f"{K.X}/chx_lib.py": "539623d68d538deb29b981e76f9d5b181c05c0efe291533e6ecf9f7c1528e8f5",
    f"{K.X}/chx_io.py": "0a5a0d152ca4ede03b8e07fdd570d9b527af375b2c3b0c542ba5f384a79d438d",
    f"{K.X}/score_chx.py": "d6346ba6c8ec3e1046e22c5d6e15d3bda354e7c8820dbcbe70eee5af5e9d2f76",
    K.SCORER: K.SCORER_SHA,
}
OWN_FROZEN = (LIB, SCORER, PREREG)

# ----------------------------------------------------------------------------- the look
#: logged by ``score_chx2c.py look`` AFTER registration and AFTER the model-only collinearity
#: passed (no look is logged when |r| > 0.9); before any head is read
LOOK_PURPOSE = "after_the_fact"
LOOK_UNITS = (UNIT,)
LOOK_SCOPE = SCOPE
LOOK_NOTE = ("I1.CHX.v1 two-contrast exploratory follow-up (R8.10, the single exploratory "
             "slot), logged after the model-only collinearity passed: observed heads at the 23 "
             "spent Changhua screens, 2013-01..2022-12, are read once on a successful scoring "
             "run for the reproduction pre-gate and the joint regression; a crash re-run, if "
             "one happens, is a second read recorded in twocontrast_attempts.jsonl; narrowing "
             "only; cannot promote")

# ----------------------------------------------------------------------------- panel
#: the only chx_panel.npz keys this code loads. All are model output or metadata; the panel
#: holds no observation (hindcast.py writes model heads only). Observed heads come from
#: L1_clean v3 through chx_io.l1_monthly, in the scorer, after the look.
PANEL_KEYS = ("months", "sid_ch", "block_ch", "A_ch", "B_ch", "C_ch", "c_ch", "c_placebo_ch")
DEPENDENT_OFFSET = "B_ch"
REGRESSORS = ("c_ch", "c_placebo_ch")             # Changhua, Yunlin (placebo) contrasts
REGRESSOR_NAMES = ("changhua", "yunlin")
FORM = K.CHX_PRIMARY_FORM                         # 'twoway', unchanged

# ----------------------------------------------------------------------------- thresholds
COLLINEAR_MAX = 0.9          # |r| > 0.9 -> 'not separable', fixed without fitting
PREGATE_TOL = 1e-6           # reproduction of CHX's published primary (and placebo)
ROUTE_TOL = 1e-9             # this code with one regressor vs the frozen CHX code
SINGULAR_DET_REL = 1e-12     # a draw is singular if det <= this x product of its diagonal
SINGULAR_SHARE_MAX = 0.05    # more singular draws than this share -> 'not separable'
#: (practically never fires: once the model-only |r| <= 0.9 a resampled 2 x 2 Gram matrix is
#: singular only if a draw keeps blocks whose contrasts are exactly proportional)

READING_SURVIVES = ("direction survives a Yunlin spill-over control (after-the-fact, "
                    "exploratory)")
READING_NOT_SEPARABLE = "direction not separable from Yunlin spill-over"
READING_NOT_EVALUABLE = ("two-contrast check not evaluable (reproduction pre-gate failed); "
                         "no joint coefficient was computed")
READINGS = {"survives": READING_SURVIVES, "not_separable": READING_NOT_SEPARABLE,
            "not_evaluable": READING_NOT_EVALUABLE}
E25_STATUS = "unchanged in every branch (the reading changes wording only)"
CANNOT_PROMOTE = ("after-the-fact, exploratory narrowing on spent units; it cannot promote "
                  "E25, C20/C21 or any claim, in any branch")


# ----------------------------------------------------------------------------- transform
def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    den = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
    return float(np.dot(a, b) / den) if den > 0 else float("nan")


def collinearity_model_only(c1_wide, c2_wide, sids, months, form: str = FORM) -> dict:
    """The declared collinearity statistic from the panel alone (no head): both contrasts over
    every screen x window month, the primary's fixed effects projected out (as CHX's
    informative gate does for one contrast), then their cosine (= Pearson r, both have mean 0
    after the two-way transform). ``separable`` is False when |r| > COLLINEAR_MAX."""
    c1, c2 = np.asarray(c1_wide, float), np.asarray(c2_wide, float)
    W, T = c1.shape
    if c2.shape != (W, T) or len(months) != T or len(sids) != W:
        raise ValueError("contrasts, screens and months differ in shape")
    if not (np.isfinite(c1).all() and np.isfinite(c2).all()):
        raise ValueError("collinearity: the contrast panels must be finite")
    rows = {"well": np.repeat(np.asarray([str(s) for s in sids]), T),
            "month": np.tile(np.asarray([str(m) for m in months]), W),
            "t": np.tile(L.month_t([str(m) for m in months]), W)}
    t1, t2 = L.transform(c1.ravel(), c2.ravel(), rows, form)
    r = _cosine(t1, t2)
    return {"r": r, "abs_r": abs(r) if np.isfinite(r) else float("nan"),
            "threshold": COLLINEAR_MAX, "form": form, "finite": bool(np.isfinite(r)),
            "grid": [int(W), int(T)], "separable": bool(np.isfinite(r)
                                                         and abs(r) <= COLLINEAR_MAX),
            "rms_transformed_m": [float(np.sqrt(np.mean(t1 ** 2))),
                                  float(np.sqrt(np.mean(t2 ** 2)))]}


def joint_rows(sids, months, obs, flag, zr, regressors: list[np.ndarray]) -> dict:
    """CHX's eligible rows (``chx_lib.long_panel`` on the FIRST regressor, then ``eligible``)
    with every regressor attached on exactly those rows."""
    sids = [str(s) for s in sids]
    months = [str(m) for m in months]
    rows = L.eligible(L.long_panel(sids, months, obs, zr, regressors[0], flag))
    wi = {s: i for i, s in enumerate(sids)}
    ti = {m: j for j, m in enumerate(months)}
    iw = np.array([wi[str(w)] for w in rows["well"]], int)
    it = np.array([ti[str(m)] for m in rows["month"]], int)
    X = np.column_stack([np.asarray(c, float)[iw, it] for c in regressors])
    if not np.isfinite(X).all():
        raise ValueError("a regressor is not finite on a scored row (engineering)")
    if not np.array_equal(X[:, 0], rows["c"]):
        raise ValueError("first regressor differs from CHX's row contrast (engineering)")
    rows["X"] = X
    return rows


def transform_rows(rows: dict, form: str = FORM, Q: np.ndarray | None = None
                   ) -> tuple[np.ndarray, np.ndarray]:
    """(e~, X~): e = obs - ZR and every regressor with the declared fixed effects projected out
    (``chx_lib.fe_basis``/``fe_resid``, as ``chx_lib.transform``), then the pinned scorer's
    per-well demeaning (``bootstrap_scorer.within``), as ``beta_response`` applies it."""
    w = rows["well"]
    e = rows["obs"] - rows["zr"]
    X = rows["X"]
    cols = []
    if form == "detrended":
        uniq, inv = BS._codes(w)
        et = BS.within(e, inv, len(uniq), rows["t"], detrend=True)
        cols = [BS.within(X[:, j], inv, len(uniq), rows["t"], detrend=True)
                for j in range(X.shape[1])]
    else:
        Q = L.fe_basis(w, rows["month"], form) if Q is None else Q
        et = L.fe_resid(e, Q)
        cols = [L.fe_resid(X[:, j], Q) for j in range(X.shape[1])]
    uniq, inv = BS._codes(w)
    n = len(uniq)
    et = BS.within(et, inv, n)
    Xt = np.column_stack([BS.within(c, inv, n) for c in cols])
    return et, Xt


def well_cross_sums(et, Xt, wells) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(well ids, per-well X~'e~ (n, k), per-well X~'X~ (n, k, k))."""
    uniq, inv = BS._codes(wells)
    n, k = len(uniq), Xt.shape[1]
    sxy = np.column_stack([np.bincount(inv, weights=Xt[:, j] * et, minlength=n)
                           for j in range(k)])
    sxx = np.empty((n, k, k))
    for a in range(k):
        for b in range(k):
            sxx[:, a, b] = np.bincount(inv, weights=Xt[:, a] * Xt[:, b], minlength=n)
    return uniq, sxy, sxx


def _ok(den: np.ndarray) -> np.ndarray:
    """Non-singular resampled cross-product matrices (one regressor: den > 0, the pinned rule)."""
    diag = np.diagonal(den, axis1=-2, axis2=-1)
    pos = (diag > 0).all(axis=-1)
    if den.shape[-1] == 1:
        return pos
    det = np.linalg.det(den)
    return pos & (det > SINGULAR_DET_REL * np.prod(diag, axis=-1))


def joint_bootstrap(sxy: np.ndarray, sxx: np.ndarray, well_blocks, B: int = BS.B_BOOT,
                    seed: int = BS.SEED, level: float = BS.LEVEL) -> dict:
    """The pinned block bootstrap (``bootstrap_scorer.block_bootstrap``'s exact draw) with
    each draw solving the k x k normal equations of the resampled block sums."""
    ub, binv = BS._codes(well_blocks)
    nb, k = len(ub), sxy.shape[1]
    sxy_b = np.stack([np.bincount(binv, weights=sxy[:, j], minlength=nb) for j in range(k)],
                     axis=1)
    sxx_b = np.empty((nb, k, k))
    for a in range(k):
        for b in range(k):
            sxx_b[:, a, b] = np.bincount(binv, weights=sxx[:, a, b], minlength=nb)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, nb, size=(B, nb))
    cnt = np.zeros((B, nb))
    np.add.at(cnt, (np.repeat(np.arange(B), nb), draw.ravel()), 1.0)
    num = cnt @ sxy_b                                           # (B, k)
    den = (cnt @ sxx_b.reshape(nb, k * k)).reshape(B, k, k)     # (B, k, k)
    ok = _ok(den)
    if k == 1:
        betas = (num[ok, 0] / den[ok, 0, 0])[:, None]
    else:
        betas = np.linalg.solve(den[ok], num[ok][..., None])[..., 0]
    q = ((1.0 - level) / 2.0, 1.0 - (1.0 - level) / 2.0)
    ci = []
    for j in range(k):
        if betas.shape[0] == 0:
            ci.append([float("nan"), float("nan")])
        else:
            lo, hi = (float(v) for v in np.quantile(betas[:, j], q))
            ci.append([lo, hi])
    return {"ci90": ci, "n_blocks": int(nb), "B": int(B), "seed": int(seed),
            "level": float(level), "n_draws_used": int(ok.sum()),
            "n_draws_singular": int((~ok).sum())}


def joint_fit(rows: dict, blocks: dict, form: str = FORM, B: int = BS.B_BOOT,
              seed: int = BS.SEED) -> dict:
    """Point coefficients (least squares on the summed per-well cross-products) and their 90 %
    block-bootstrap intervals, one per column of ``rows['X']``."""
    w = rows["well"]
    if len(w) == 0:
        raise ValueError("no scored row")
    miss = sorted(set(w) - set(blocks))
    if miss:
        raise ValueError(f"wells without a block: {miss[:5]}")
    et, Xt = transform_rows(rows, form)
    uniq, sxy, sxx = well_cross_sums(et, Xt, w)
    Sxy, Sxx = sxy.sum(axis=0), sxx.sum(axis=0)
    k = Xt.shape[1]
    if not _ok(Sxx[None])[0]:
        beta = [float("nan")] * k
    elif k == 1:
        beta = [float(Sxy[0] / Sxx[0, 0])]
    else:
        beta = [float(v) for v in np.linalg.solve(Sxx, Sxy)]
    boot = joint_bootstrap(sxy, sxx, [blocks[u] for u in uniq], B, seed)
    return {"beta": beta, "ci90": boot["ci90"], "bootstrap": boot, "n_wells": int(len(uniq)),
            "n_rows": int(len(w)), "form": form,
            "scored_rows_r": (_cosine(Xt[:, 0], Xt[:, 1]) if k == 2 else None)}


# ----------------------------------------------------------------------------- reading
def reading(fit: dict | None, collinearity: dict, pregate: dict | None) -> dict:
    """The declared reading. Order: model-only collinearity (|r| > 0.9 fixes 'not separable'
    without a fit); a failed reproduction pre-gate gives 'not evaluable' (no joint statistic);
    then the scored-row collinearity and the singular-draw share (each can only narrow); then
    the Changhua coefficient's interval: lo > 0 and point > 0 -> 'survives', else 'not
    separable'. E25's status is unchanged in every branch."""
    why = []
    if not collinearity["separable"]:
        key = "not_separable"
        ar = collinearity["abs_r"]
        why.append(f"model-only |r| {ar:.3f} > {COLLINEAR_MAX} (fixed without fitting)"
                   if np.isfinite(ar) else "model-only r not finite (fixed without fitting)")
    elif pregate is not None and not pregate["passed"]:
        key = "not_evaluable"
        why.append("reproduction pre-gate failed")
    else:
        if fit is None:
            raise ValueError("a reading past the gates needs the joint fit")
        b, (lo, hi) = fit["beta"][0], fit["ci90"][0]
        boot = fit["bootstrap"]
        r_rows = fit.get("scored_rows_r")
        narrow = []
        if r_rows is not None and not (np.isfinite(r_rows) and abs(r_rows) <= COLLINEAR_MAX):
            narrow.append(f"scored-row |r| {abs(r_rows):.3f} > {COLLINEAR_MAX}")
        if boot["n_draws_singular"] > SINGULAR_SHARE_MAX * boot["B"]:
            narrow.append(f"{boot['n_draws_singular']} singular draws > "
                          f"{SINGULAR_SHARE_MAX:.0%} of {boot['B']}")
        finite = all(np.isfinite(v) for v in (b, lo, hi))
        if narrow:
            key = "not_separable"
            why += narrow
        elif finite and lo > 0.0 and b > 0.0:
            key = "survives"
            why.append("Changhua coefficient's 90 % interval lies above 0")
        else:
            key = "not_separable"
            why.append("Changhua coefficient's 90 % interval reaches 0" if finite else
                       "Changhua coefficient not finite")
    return {"key": key, "e25_note": READINGS[key], "why": why, "e25_status": E25_STATUS,
            "cannot_promote": CANNOT_PROMOTE}


# ----------------------------------------------------------------------------- pre-gate
def pregate(published: dict, frozen_route: dict, own_route: dict) -> dict:
    """Reproduction pre-gate: CHX's frozen code on the re-read heads (``frozen_route``:
    score_chx.score_panel's primary and placebo) and this code with the Yunlin regressor
    dropped (``own_route``: joint_fit on c_CH alone) must both give CHX's published primary
    beta and interval to PREGATE_TOL, the same rows and wells, and agree with each other to
    ROUTE_TOL; the frozen route must also reproduce the published placebo."""
    pp, pl = published["primary"], published["placebo"]
    fr, fl = frozen_route["primary"], frozen_route["placebo"]
    checks = {}

    def close(name, a, b, tol):
        d = abs(float(a) - float(b)) if np.isfinite(a) and np.isfinite(b) else float("inf")
        checks[name] = {"abs_diff": d, "tol": tol, "ok": bool(d <= tol)}

    close("frozen_beta_vs_published", fr["beta"], pp["beta"], PREGATE_TOL)
    close("frozen_lo_vs_published", fr["ci90"][0], pp["ci90"][0], PREGATE_TOL)
    close("frozen_hi_vs_published", fr["ci90"][1], pp["ci90"][1], PREGATE_TOL)
    close("own_beta_vs_published", own_route["beta"][0], pp["beta"], PREGATE_TOL)
    close("own_lo_vs_published", own_route["ci90"][0][0], pp["ci90"][0], PREGATE_TOL)
    close("own_hi_vs_published", own_route["ci90"][0][1], pp["ci90"][1], PREGATE_TOL)
    close("own_beta_vs_frozen", own_route["beta"][0], fr["beta"], ROUTE_TOL)
    close("own_lo_vs_frozen", own_route["ci90"][0][0], fr["ci90"][0], ROUTE_TOL)
    close("own_hi_vs_frozen", own_route["ci90"][0][1], fr["ci90"][1], ROUTE_TOL)
    close("placebo_beta_vs_published", fl["beta"], pl["beta"], PREGATE_TOL)
    close("placebo_lo_vs_published", fl["ci90"][0], pl["ci90"][0], PREGATE_TOL)
    close("placebo_hi_vs_published", fl["ci90"][1], pl["ci90"][1], PREGATE_TOL)
    counts = {"n_rows": (fr["n_rows"], own_route["n_rows"], pp["n_rows"]),
              "n_wells": (fr["n_wells"], own_route["n_wells"], pp["n_wells"])}
    for k, v in counts.items():
        checks[k] = {"values": list(map(int, v)), "ok": len(set(map(int, v))) == 1}
    return {"passed": all(c["ok"] for c in checks.values()), "checks": checks}


def published_numbers(parent_report: dict) -> dict:
    """CHX's published primary and placebo (report.json) in the pre-gate's shape."""
    p = parent_report["primary"]
    q = parent_report["placebo_report_only"]
    return {"primary": {"beta": p["beta_primary"]["beta"], "ci90": p["beta_primary"]["ci90"],
                        "n_rows": p["n_rows"], "n_wells": p["n_wells"]},
            "placebo": {"beta": q["beta_primary"]["beta"], "ci90": q["beta_primary"]["ci90"]}}


def frozen_numbers(score_panel_out: dict) -> dict:
    """``score_chx.score_panel``'s output in the pre-gate's shape."""
    p, q = score_panel_out["primary"], score_panel_out["placebo_report_only"]
    return {"primary": {"beta": p["beta_primary"]["beta"], "ci90": p["beta_primary"]["ci90"],
                        "n_rows": p["n_rows"], "n_wells": p["n_wells"]},
            "placebo": {"beta": q["beta_primary"]["beta"], "ci90": q["beta_primary"]["ci90"]}}
