"""Pure functions of N2CS (round-8 rank #6). No function here reads a file.

The estimator wraps the pinned shared N2/N2C scorer ``dataprep_staging/n2_policy/
bootstrap_scorer.py`` (sha256 f8990d61..., cited, never edited): the two-contrast twoway
regression is reduced by Frisch-Waugh to the pinned per-well slope of e~ on the Changhua
contrast residualised on the Yunlin contrast, so ``beta_response`` (5 km block cluster
bootstrap, B 2000, seed 20261101, 90 %) gives the Changhua coefficient and its interval, and
``informative_gate`` the RMS / cluster / month part of the informative gate. What this module
adds is declared in n2cs_const: the three forms' fixed effects, the readings, the collinearity
part of the informative gate, the OSSE worlds, the G0 gate per form and the choice rule.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent / "n2_policy")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bootstrap_scorer as BS  # noqa: E402
import n2cs_const as K  # noqa: E402


# ----------------------------------------------------------------------------- months
def month_list(first: str, last: str) -> list[str]:
    y0, m0 = int(first[:4]), int(first[5:7])
    y1, m1 = int(last[:4]), int(last[5:7])
    a, b = 12 * y0 + m0 - 1, 12 * y1 + m1 - 1
    if b < a:
        raise ValueError(f"{first} > {last}")
    return [f"{k // 12}-{k % 12 + 1:02d}" for k in range(a, b + 1)]


def month_t(months) -> np.ndarray:
    return np.array([int(str(m)[:4]) * 12 + int(str(m)[5:7]) - 1 for m in months], float)


WINDOW_MONTHS = month_list(*K.WINDOW)


# ----------------------------------------------------------------------------- rows
def long_rows(sids, months, e: np.ndarray, c_ch: np.ndarray, c_yl: np.ndarray) -> dict:
    """Wide (W, T) arrays -> long rows with finite e, c_CH and c_YL (row-major: well, month).
    Wells with fewer than MIN_WELL_MONTHS rows are dropped and listed."""
    sids = np.asarray([str(s) for s in sids])
    e, c_ch, c_yl = (np.asarray(v, float) for v in (e, c_ch, c_yl))
    W, T = e.shape
    if c_ch.shape != (W, T) or c_yl.shape != (W, T) or len(months) != T or len(sids) != W:
        raise ValueError("panel arrays differ in shape")
    well = np.repeat(sids, T)
    mon = np.tile(np.asarray([str(m) for m in months]), W)
    keep = np.isfinite(e.ravel()) & np.isfinite(c_ch.ravel()) & np.isfinite(c_yl.ravel())
    u, n = np.unique(well[keep], return_counts=True)
    ok = set(u[n >= K.MIN_WELL_MONTHS])
    keep &= np.isin(well, list(ok))
    return {"well": well[keep], "month": mon[keep], "t": month_t(mon[keep]),
            "e": e.ravel()[keep], "ch": c_ch.ravel()[keep], "yl": c_yl.ravel()[keep],
            "dropped_wells": sorted(set(sids) - ok)}


def first_diff(rows: dict) -> dict:
    """Per-well differences of consecutive scored months (t and t - 1 both scored), labelled
    with the later month."""
    order = np.lexsort((rows["t"], rows["well"]))
    w, t = rows["well"][order], rows["t"][order]
    nxt = (w[1:] == w[:-1]) & (t[1:] - t[:-1] == 1.0)
    i1 = np.flatnonzero(nxt) + 1
    i0 = i1 - 1
    out = {"well": w[i1], "month": rows["month"][order][i1], "t": t[i1]}
    for k in ("e", "ch", "yl"):
        if k in rows:
            v = rows[k][order]
            out[k] = v[i1] - v[i0]
    return out


# ----------------------------------------------------------------------------- fixed effects
def fe_basis(wells, months, form: str) -> np.ndarray:
    """Orthonormal basis of the form's fixed effects on these rows (rows of differences for
    'twoway_fd'): per-well intercept (+ per-well linear trend for 'twoway') + month-of-sample
    effects. Rank-deficient designs are fine (SVD)."""
    if form not in K.FORMS:
        raise ValueError(f"unknown form {form!r}")
    wells = np.asarray(wells).astype(str)
    months = np.asarray([str(m)[:7] for m in months])
    n = len(wells)
    if n == 0:
        return np.zeros((0, 0))
    rows = np.arange(n)
    uw, iw = np.unique(wells, return_inverse=True)
    D = np.zeros((n, len(uw)))
    D[rows, iw] = 1.0
    cols = [D]
    if form == "twoway":
        t = month_t(months)
        cols.append(D * (t - t.mean())[:, None])
    um, im = np.unique(months, return_inverse=True)
    M = np.zeros((n, len(um)))
    M[rows, im] = 1.0
    cols.append(M)
    U, s, _ = np.linalg.svd(np.hstack(cols), full_matrices=False)
    return U[:, s > s[0] * 1e-10]


def prepare(rows: dict, form: str) -> tuple[dict, np.ndarray]:
    """(rows the form fits on, their fixed-effect basis)."""
    r = first_diff(rows) if form == "twoway_fd" else rows
    return r, fe_basis(r["well"], r["month"], form)


def resid(v, Q: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    return v - Q @ (Q.T @ v)


def partial(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    """``a`` residualised on ``b`` (no intercept: both are already fixed-effect residuals)."""
    bb = float(b @ b)
    g = float(a @ b) / bb if bb > 0 else 0.0
    return a - g * b, g


# ----------------------------------------------------------------------------- estimator
def estimate(rows: dict, blocks: dict, form: str, *, prepared=None, B: int = BS.B_BOOT,
             seed: int = BS.SEED, yunlin: bool = True) -> dict:
    """The two-contrast twoway regression under ``form``: beta_CH (the estimand) with the
    pinned scorer's interval on the Frisch-Waugh regressor, and (``yunlin``) beta_YL the same
    way (report only)."""
    r, Q = prepared if prepared is not None else prepare(rows, form)
    if len(r["well"]) == 0:
        raise ValueError("no scored row")
    miss = sorted(set(r["well"]) - set(blocks))
    if miss:
        raise ValueError(f"wells without a block: {miss[:5]}")
    et, cht, ylt = resid(r["e"], Q), resid(r["ch"], Q), resid(r["yl"], Q)
    ch_p, g = partial(cht, ylt)
    br = BS.beta_response(et, ch_p, r["well"], blocks, B=B, seed=seed)
    out = {"form": form, "beta_ch": br["beta"], "ci90_ch": list(br["ci90"]),
           "bootstrap_ch": br["bootstrap"], "n_wells": br["n_wells"],
           "n_rows": int(len(r["well"])), "gamma_ch_on_yl": g,
           "corr_ch_yl": _corr(cht, ylt)}
    if yunlin:
        yl_p, _ = partial(ylt, cht)
        by = BS.beta_response(et, yl_p, r["well"], blocks, B=B, seed=seed)
        out.update({"beta_yl_report_only": by["beta"], "ci90_yl_report_only": list(by["ci90"])})
    return out


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    den = math.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b) / den if den > 0 else float("nan")


def refit_interval(rows: dict, blocks: dict, form: str, *, B: int = BS.B_BOOT,
                   seed: int = BS.SEED, level: float = BS.LEVEL) -> dict:
    """Report only: the 2 x 2 refit per draw (re-partialling in every draw) with the pinned
    scorer's draws (same sorted blocks, same generator call)."""
    r, Q = prepare(rows, form)
    et, cht, ylt = resid(r["e"], Q), resid(r["ch"], Q), resid(r["yl"], Q)
    X = np.stack([cht, ylt], axis=1)
    uw, iw = np.unique(r["well"].astype(str), return_inverse=True)
    wb = np.array([blocks[w] for w in uw])
    ub, ib = np.unique(wb.astype(str), return_inverse=True)
    rb = ib[iw]
    nb = len(ub)
    sxx = np.zeros((nb, 2, 2))
    sxy = np.zeros((nb, 2))
    np.add.at(sxx, rb, X[:, :, None] * X[:, None, :])
    np.add.at(sxy, rb, X * et[:, None])
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, nb, size=(B, nb))
    cnt = np.zeros((B, nb))
    np.add.at(cnt, (np.repeat(np.arange(B), nb), draw.ravel()), 1.0)
    A = np.einsum("bk,kij->bij", cnt, sxx)
    y = cnt @ sxy
    ok = np.abs(np.linalg.det(A)) > 1e-18
    betas = np.linalg.solve(A[ok], y[ok][:, :, None])[:, 0, 0] if ok.any() else np.array([])
    q = ((1.0 - level) / 2.0, 1.0 - (1.0 - level) / 2.0)
    lo, hi = (float(v) for v in np.quantile(betas, q)) if betas.size else (math.nan,) * 2
    full = np.linalg.lstsq(X, et, rcond=None)[0]
    return {"beta_ch": float(full[0]), "ci90_ch": [lo, hi], "n_draws_used": int(ok.sum()),
            "B": int(B), "seed": int(seed)}


def reading(beta: float, lo: float, hi: float, informative: bool) -> str:
    """The declared one-sided size-class reading."""
    if not informative:
        return "not_informative"
    if not (np.isfinite(beta) and np.isfinite(lo) and np.isfinite(hi)):
        return "inconclusive"
    if hi < K.SIZE_X:
        return "overstated_x2"
    if lo > K.SIZE_X:
        return "not_overstated_x2"
    return "inconclusive"


# ----------------------------------------------------------------------------- informative
def informative_gate(c_ch: np.ndarray, c_yl: np.ndarray, sids, blocks: dict, months,
                     n_recorded_months: int, form: str) -> dict:
    """Model-only, before any head, under the CHOSEN ``form`` (lead ruling Q2): both contrasts
    at every well and window month transformed by the form; the pinned gate (RMS >= 0.05 m,
    >= 5 clusters, >= 6 recorded Changhua months) on the Frisch-Waugh regressor (c_CH~
    residualised on c_YL~), and |corr(c_CH~, c_YL~)| <= 0.8 as an additional check."""
    c_ch, c_yl = np.asarray(c_ch, float), np.asarray(c_yl, float)
    if not (np.isfinite(c_ch).all() and np.isfinite(c_yl).all()):
        raise ValueError("informative_gate: the contrast panel must be finite")
    W, T = c_ch.shape
    rows = long_rows(sids, months, np.zeros((W, T)), c_ch, c_yl)
    if rows["dropped_wells"]:
        raise ValueError("informative_gate: a full panel drops no well")
    r, Q = prepare(rows, form)
    cht, ylt = resid(r["ch"], Q), resid(r["yl"], Q)
    chp, _ = partial(cht, ylt)
    g = BS.informative_gate(chp, r["well"], blocks, n_recorded_months)
    corr = _corr(cht, ylt)
    g.update({"form": form, "regressor": "c_CH~ residualised on c_YL~ (Frisch-Waugh)",
              "corr_ch_yl": corr, "max_abs_corr": K.MAX_ABS_CORR,
              "rms_ch_transformed_m_report_only": float(np.sqrt(np.mean(cht ** 2))),
              "rms_yl_transformed_m_report_only": float(np.sqrt(np.mean(ylt ** 2))),
              "pinned_gate_passes": bool(g["passes"])})
    g["passes"] = bool(g["passes"] and np.isfinite(corr) and abs(corr) <= K.MAX_ABS_CORR)
    return g


# ----------------------------------------------------------------------------- OSSE worlds
def ar1(rng, n: int, T: int, s: float, phi: float) -> np.ndarray:
    """Stationary AR(1) rows (synth_n2c.ar1's recursion)."""
    eps = rng.standard_normal((n, T))
    x = np.empty((n, T))
    x[:, 0] = s * eps[:, 0]
    q = math.sqrt(1.0 - phi * phi) * s
    for k in range(1, T):
        x[:, k] = phi * x[:, k - 1] + q * eps[:, k]
    return x


def confounder(xi: np.ndarray, a: np.ndarray, rho: float) -> np.ndarray:
    """Unit-sd series with correlation exactly ``rho`` to ``a`` (synth_n2c.confounder, from a
    pre-drawn ``xi``)."""
    a = np.asarray(a, float) - np.mean(a)
    sd = a.std()
    a = a / sd if sd > 0 else a
    xi = np.asarray(xi, float) - np.mean(xi)
    if sd > 0:
        xi = xi - (xi @ a) / (a @ a) * a
    xs = xi.std()
    xi = xi / xs if xs > 0 else xi
    return rho * a + math.sqrt(1.0 - rho * rho) * xi


def load_z(c: np.ndarray) -> np.ndarray:
    """Standardised per-well RMS of the well-demeaned contrast (zeros if constant)."""
    c = np.asarray(c, float)
    amp = np.sqrt(np.mean((c - c.mean(axis=1, keepdims=True)) ** 2, axis=1))
    sd = amp.std()
    return (amp - amp.mean()) / sd if sd > 0 else np.zeros_like(amp)


def world(panel: dict, j: int, k: float, conf: str, rng, params: dict, *,
          k_yl: float | None = None, kappa_mult: float = 1.0, noise_scale: float = 1.0,
          miss_frac: float = 0.0, drop_wells: int = 0) -> np.ndarray:
    """One synthetic e = obs - ZR (W, T) for replay ``j`` (draw order: n2cs_const.SEEDING)."""
    c_ch, c_yl = panel["c_ch"][j], panel["c_yl"][j]
    W, T = c_ch.shape
    spec = K.CONFOUNDERS[conf]
    ub, binv = np.unique(panel["block"], return_inverse=True)
    sig, share, phi = params["sigma_tot_m"], params["block_share"], params["phi"]
    b = ar1(rng, len(ub), T, noise_scale * sig * math.sqrt(share), phi)[binv]
    wn = ar1(rng, W, T, noise_scale * sig * math.sqrt(1.0 - share), phi)
    xi = rng.standard_normal(T)
    z = rng.standard_normal(W)
    g = confounder(xi, panel["a_ch"][j], spec["rho"])
    if spec["loading"] == "shaped":
        lr = spec["load_rho"]
        z = lr * load_z(c_ch) + math.sqrt(1.0 - lr * lr) * z
        load = 1.0 + spec["het_mult"] * z
    else:
        load = np.ones(W)
    kap = kappa_mult * spec.get("kappa_mult", 1.0) * params["kappa_m"]
    e = (k * c_ch + (k if k_yl is None else k_yl) * c_yl + b + wn
         + kap * load[:, None] * g[None, :])
    if miss_frac > 0:
        e = np.where(rng.random((W, T)) < miss_frac, np.nan, e)
    if drop_wells > 0:
        e = e.copy()
        e[rng.permutation(W)[:drop_wells], :] = np.nan
    return e


def panel_informative(panel: dict, forms=K.FORMS) -> dict:
    """{form: [the informative gate of each replay's contrast under that form]} (all window
    months recorded)."""
    blocks = dict(zip(panel["sid"], panel["block"], strict=True))
    T = panel["c_ch"].shape[2]
    return {f: [informative_gate(panel["c_ch"][j], panel["c_yl"][j], panel["sid"], blocks,
                                 panel["months"], T, f) for j in range(panel["c_ch"].shape[0])]
            for f in forms}


def _templates(panel: dict, j: int, forms) -> dict:
    W, T = panel["c_ch"][j].shape
    rows = long_rows(panel["sid"], panel["months"], np.zeros((W, T)), panel["c_ch"][j],
                     panel["c_yl"][j])
    return {f: prepare(rows, f) for f in forms}


def score_world(panel: dict, j: int, e: np.ndarray, forms, informative: dict, *,
                templates: dict | None = None, B: int = BS.B_BOOT) -> dict:
    """Every form's beta_CH, interval and reading on one world (pinned bootstrap);
    ``informative`` {form: bool} for this replay."""
    blocks = dict(zip(panel["sid"], panel["block"], strict=True))
    rows = long_rows(panel["sid"], panel["months"], e, panel["c_ch"][j], panel["c_yl"][j])
    out = {}
    for f in forms:
        if not informative[f]:
            out[f] = {"reading": "not_informative", "beta": math.nan, "lo": math.nan,
                      "hi": math.nan}
            continue
        if templates is not None and not np.isnan(e).any():
            r0, Q = templates[f]
            r = dict(r0)
            src = first_diff(rows) if f == "twoway_fd" else rows
            if len(src["well"]) != len(r0["well"]):
                raise RuntimeError("a full world's rows differ from the template")
            r["e"] = src["e"]
            prep = (r, Q)
        else:
            prep = prepare(rows, f)
        s = estimate(rows, blocks, f, prepared=prep, B=B, yunlin=False)
        lo, hi = s["ci90_ch"]
        out[f] = {"reading": reading(s["beta_ch"], lo, hi, True), "beta": s["beta_ch"],
                  "lo": lo, "hi": hi}
    return out


def plumbing_exact(panel: dict, forms=K.FORMS, ks=((0.5, 1.0), (1.0, 1.0), (2.0, 0.3))
                   ) -> dict:
    """No noise, no confounder: beta_CH must equal k_CH exactly for every replay and form."""
    out, ok = {}, True
    for j in range(panel["c_ch"].shape[0]):
        tm = _templates(panel, j, forms)
        for kc, ky in ks:
            e = kc * panel["c_ch"][j] + ky * panel["c_yl"][j]
            for f in forms:
                r0, Q = tm[f]
                rows = long_rows(panel["sid"], panel["months"], e, panel["c_ch"][j],
                                 panel["c_yl"][j])
                src = first_diff(rows) if f == "twoway_fd" else rows
                et, cht, ylt = resid(src["e"], Q), resid(src["ch"], Q), resid(src["yl"], Q)
                chp, _ = partial(cht, ylt)
                b = BS.slope_fe(et, chp, src["well"])
                good = bool(np.isfinite(b) and abs(b - kc) <= K.PLUMBING_TOL)
                ok &= good
                out[f"{j}:{f}:{kc}:{ky}"] = b
    return {"ok": bool(ok), "tol": K.PLUMBING_TOL, "betas": out}


def cells() -> list[dict]:
    """Gate cells (k outer, gate confounder inner), then the report-only cells."""
    gate = [dict(K.GATE_VARIANTS[v], k=k, variant=v, role="gate")
            for k in K.K_GRID for v in K.GATE_VARIANTS]
    rep = [dict(c, role="report_only") for c in K.REPORT_CELLS]
    return gate + rep


def osse_cell(panel: dict, params: dict, cell: dict, rng, reps: int, informative: dict,
              forms=K.FORMS, B: int = BS.B_BOOT) -> dict:
    J = panel["c_ch"].shape[0]
    tms = {j: _templates(panel, j, forms) for j in range(J)}
    cnt = {f: {r: 0 for r in K.READINGS} for f in forms}
    betas = {f: [] for f in forms}
    cover = {f: 0 for f in forms}
    n_inf = {f: 0 for f in forms}
    for r in range(reps):
        j = r % J
        e = world(panel, j, cell["k"], cell["conf"], rng, params, k_yl=cell.get("k_yl"),
                  kappa_mult=cell.get("kappa_mult", 1.0), miss_frac=cell.get("miss_frac", 0.0),
                  drop_wells=cell.get("drop_wells", 0))
        inf = {f: bool(informative[f][j]["passes"]) for f in forms}
        s = score_world(panel, j, e, forms, inf, templates=tms[j], B=B)
        for f in forms:
            cnt[f][s[f]["reading"]] += 1
            if inf[f]:
                n_inf[f] += 1
                betas[f].append(s[f]["beta"])
                cover[f] += bool(s[f]["lo"] <= cell["k"] <= s[f]["hi"])
    out = {"cell": {k: v for k, v in cell.items()}, "replicates": reps,
           "n_informative": n_inf, "forms": {}}
    for f in forms:
        c = cnt[f]
        n_inf_f = n_inf[f]
        p = {r: c[r] / reps for r in K.READINGS}
        pi = ({r: c[r] / n_inf_f for r in ("overstated_x2", "not_overstated_x2",
                                           "inconclusive")} if n_inf_f else None)
        out["forms"][f] = {"counts": c, "p": p, "p_given_informative": pi,
                           "median_beta": float(np.median(betas[f])) if betas[f] else None,
                           "coverage_k_given_informative": (cover[f] / n_inf_f if n_inf_f
                                                            else None)}
    return out


def _strict_false(cell_form: dict, reading_name: str) -> float:
    p = cell_form["p"][reading_name]
    pi = cell_form["p_given_informative"]
    return max(p, pi[reading_name]) if pi is not None else p


def form_gate(results: list[dict], form: str) -> dict:
    """G0 for one form under each gate variant (k = 1 and k = 1/3 only)."""
    def get(k, var):
        hit = [c for c in results if c["cell"]["role"] == "gate"
               and c["cell"]["variant"] == var and abs(c["cell"]["k"] - k) < 1e-12]
        if len(hit) != 1:
            raise ValueError(f"gate cell k={k} variant={var} missing")
        return hit[0]["forms"][form]

    per = {}
    for var in K.GATE_VARIANTS:
        c1, c3 = get(K.K_TRUE, var), get(K.K_OVER, var)
        power = c1["p"]["not_overstated_x2"]
        false_no = _strict_false(c3, "not_overstated_x2")
        false_ov = _strict_false(c1, "overstated_x2")
        per[var] = {"p_not_overstated_k1": power, "p_not_overstated_k1_3_strict": false_no,
                     "p_overstated_k1_strict": false_ov,
                     "passes": bool(power >= K.G0["p_not_overstated_k1_min"]
                                    and false_no <= K.G0["p_not_overstated_k1_3_max"]
                                    and false_ov <= K.G0["p_overstated_k1_max"])}
    return {"form": form, "per_variant": per,
            "power_min": min(v["p_not_overstated_k1"] for v in per.values()),
            "false_not_overstated_max": max(v["p_not_overstated_k1_3_strict"]
                                            for v in per.values()),
            "passes": all(v["passes"] for v in per.values())}


def choose(gates: list[dict]) -> dict:
    """The declared choice rule (n2cs_const.CHOICE_RULE)."""
    order = {f: i for i, f in enumerate(K.FORMS)}
    best_power = max(g["power_min"] for g in gates)
    ok = [g for g in gates if g["passes"]]
    if not ok:
        return {"passed": False, "form": None, "osse_power_X": best_power,
                "n_line": f"size class not decidable prospectively (OSSE power "
                          f"{best_power:.2f})", "role": "bookkeeping, not evidence"}
    ok.sort(key=lambda g: (-g["power_min"], g["false_not_overstated_max"], order[g["form"]]))
    g = ok[0]
    return {"passed": True, "form": g["form"], "osse_power_X": g["power_min"],
            "false_not_overstated_max": g["false_not_overstated_max"]}


def run_osse(panel: dict, params: dict, *, seed: int, reps: int = K.REPLICATES,
             B: int = BS.B_BOOT, forms=K.FORMS) -> dict:
    """The declared OSSE: plumbing, per-replay informative gate, every cell, G0 per form, the
    choice. ``seed`` is required (no default)."""
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an int")
    plumb = plumbing_exact(panel, forms)
    if not plumb["ok"]:
        return {"engineering_failure": True, "plumbing": plumb}
    inf = panel_informative(panel)
    cl = cells()
    ss = np.random.SeedSequence(seed).spawn(len(cl))
    res = [osse_cell(panel, params, c, np.random.default_rng(ss[i]), reps, inf, forms, B)
           for i, c in enumerate(cl)]
    gates = [form_gate(res, f) for f in forms]
    return {"engineering_failure": False, "plumbing": {"ok": True, "tol": K.PLUMBING_TOL},
            "replay_informative": inf, "cells": res, "gates": gates, "choice": choose(gates),
            "seed": seed, "replicates": reps, "B": B, "bootstrap_seed": BS.SEED}


def synthetic_panel(seed: int, n_blocks: int = 20, scale: float = 0.3, T: int = K.N_WINDOW
                    ) -> dict:
    """A stand-in panel for tests and smoke runs (no model output)."""
    rng = np.random.default_rng(seed)
    J, W = len(K.REPLAY_YEARS), K.N_WELLS
    a_ch = rng.normal(0, 0.1, (J, T))
    a_yl = 0.6 * a_ch + 0.8 * rng.normal(0, 0.1, (J, T))
    lam = rng.uniform(0.3, 1.7, W) * scale / 0.1
    mu = rng.uniform(0.0, 0.6, W) * scale / 0.1
    c_ch = -lam[None, :, None] * a_ch[:, None, :] + rng.normal(0, 0.02, (J, W, T))
    c_yl = -mu[None, :, None] * a_yl[:, None, :] + rng.normal(0, 0.02, (J, W, T))
    return {"c_ch": c_ch, "c_yl": c_yl, "a_ch": a_ch, "a_yl": a_yl,
            "sid": np.array([f"07{i:06d}" for i in range(W)]),
            "block": np.array([f"{b}_0" for b in rng.integers(0, n_blocks, W)]),
            "months": np.array(WINDOW_MONTHS[:T] if T <= K.N_WINDOW
                               else month_list("2026-12", "2028-12")[:T])}
