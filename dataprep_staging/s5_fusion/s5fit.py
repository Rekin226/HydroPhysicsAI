"""WP S5: the fusion problem (operators of every instrument), fitting and hyperparameters.

Instrument operators (plan WP S5 table)
---------------------------------------
* Leveling (benchmark b, survey t):  y = u(x_b, t) + c_b + r_b (t - tbar_b) + e,
  c_b diffuse, r_b ~ N(0, sig_rb^2) (benchmark instability / sub-lattice deviation),
  e ~ N(0, sig_lev^2); c_b and r_b are eliminated analytically (restricted likelihood).
* GNSS (station s of family F, month t):  y = u(x_s, t) + c_s + (D_F + d_s)(t - T_REF)
  + cm_{F, month} + e;  c_s diffuse, D_F ~ N(0, 50^2) (family frame rate), d_s ~ N(0, sig_d^2)
  (station excess rate: deep and tectonic motion), cm ~ N(0, sig_cm^2) (family common mode, a
  separate term), e ~ N(0, sig_gnss^2 + sd_month^2).
* InSAR (unit = 4 km block x network segment; month group e):
  y = u(block, t_e) + o_unit + plane_e(x) + aps + e: o_unit diffuse (so only within-segment
  changes count: LiCSBAS-bridged gap offsets never enter), a free plane per month (the
  per-epoch reference nuisance, projected out), aps with exponential covariance
  sig_aps^2 exp(-d / ell_aps) and white sig_ins^2, both divided by the epochs averaged.
* Rings and piles (sensor s): y = f_s u(x_s, t) + c_{s, segment} + r_s (t - tbar) + e;
  the depth fraction f_s is estimated once from the pre-origin fit without them (regression
  of the series on the fused u with per-segment offsets, bounded 0.2-1.2), c diffuse per
  segment, r_s ~ N(0, sig_rc^2), e ~ N(0, sig_cmp^2).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import s5const as C
import s5model as M
from scipy import optimize

FAMILIES = ("lev", "gnss", "insar", "cmp")


# ============================================================================ problem
@dataclass
class Problem:
    lat: M.Lattice
    tb: M.TimeBasis
    lev: pd.DataFrame
    gnss: pd.DataFrame
    insar: dict
    cmp: pd.DataFrame
    cell_xy: np.ndarray
    families: tuple
    fdict: dict = field(default_factory=dict)
    gnames: list = field(default_factory=list)
    gindex: dict = field(default_factory=dict)
    _cache_ell: float | None = None

    @property
    def K(self) -> int:
        return self.tb.K

    @property
    def m(self) -> int:
        return self.lat.m

    @property
    def L(self) -> int:
        return self.tb.K * self.lat.m

    @property
    def ng(self) -> int:
        return len(self.gnames)

    @property
    def N(self) -> int:
        return self.L + self.ng


def make_problem(data: dict, knots, families=FAMILIES, *, masks: dict | None = None,
                 fdict: dict | None = None, lattice: M.Lattice | None = None) -> Problem:
    """``masks``: {family: boolean keep array over that family's rows (insar: over groups,
    or a dict {"groups": bool array} )}."""
    masks = masks or {}
    g = data["grid"]
    if lattice is None:
        lattice = M.Lattice(M.build_lattice(g["cell_xy_km"], C.LATTICE_KM, C.LATTICE_REACH_KM),
                            nu=C.MATERN_NU)
    tb = M.TimeBasis(knots)
    tmax = tb.knots[-1] + 1e-9

    def sel(df, fam):
        if fam not in families or df is None or len(df) == 0:
            return df.iloc[0:0] if df is not None else pd.DataFrame()
        k = np.ones(len(df), bool) if fam not in masks else np.asarray(masks[fam], bool)
        k &= (df["t"].to_numpy() >= tb.knots[0]) & (df["t"].to_numpy() <= tmax)
        return df[k].reset_index(drop=True)

    lev = sel(data["lev"], "lev")
    gn = sel(data["gnss"], "gnss")
    cm = sel(data["cmp"], "cmp")
    ins = {"units": data["insar"]["units"], "unit_cells": data["insar"]["unit_cells"],
           "groups": []}
    if "insar" in families:
        keep = masks.get("insar")
        for i, gr in enumerate(data["insar"]["groups"]):
            if keep is not None and not keep[i]:
                continue
            if gr["t"] < tb.knots[0] or gr["t"] > tmax:
                continue
            ins["groups"].append(gr)
    pr = Problem(lattice, tb, lev, gn, ins, cm, g["cell_xy_km"], tuple(families),
                 dict(fdict or {}))
    names = ["mu_v"]
    if len(gn):
        for (f, s) in gn[["family", "station"]].drop_duplicates().itertuples(index=False):
            names += [f"goff:{f}:{s}", f"grate:{f}:{s}"]
        for f in sorted(gn.family.unique()):
            names.append(f"gframe:{f}")
        for (f, mth) in gn[["family", "month"]].drop_duplicates().sort_values(
                ["family", "month"]).itertuples(index=False):
            names.append(f"gcm:{f}:{mth}")
    used_units = sorted({int(u) for gr in ins["groups"] for u in gr["unit"]})
    names += [f"ioff:{u}" for u in used_units]
    pr.gnames = names
    pr.gindex = {n: i for i, n in enumerate(names)}
    _precompute(pr)
    return pr


def _precompute(pr: Problem) -> None:
    tb = pr.tb
    pr._lev_sites = []
    for sid, grp in pr.lev.groupby("sid", sort=True):
        t = grp.t.to_numpy()
        if t.size < 2:
            continue
        pr._lev_sites.append({"sid": sid, "t": t, "y": grp.y.to_numpy(), "H": tb.rows(t),
                              "xy": grp[["x_km", "y_km"]].to_numpy()[0], "rows": grp.index})
    pr._gnss_sites = []
    for (f, s), grp in pr.gnss.groupby(["family", "station"], sort=True):
        t = grp.t.to_numpy()
        gi = [0, pr.gindex[f"goff:{f}:{s}"], pr.gindex[f"grate:{f}:{s}"],
              pr.gindex[f"gframe:{f}"]]
        cmi = [pr.gindex[f"gcm:{f}:{mth}"] for mth in grp.month]
        n = t.size
        ucm = sorted(set(cmi))
        G = np.zeros((n, 4 + len(ucm)))
        G[:, 0] = t - C.T_REF
        G[:, 1] = 1.0
        G[:, 2] = t - C.T_REF
        G[:, 3] = t - C.T_REF
        pos = {c: j for j, c in enumerate(ucm)}
        for r, c in enumerate(cmi):
            G[r, 4 + pos[c]] = 1.0
        pr._gnss_sites.append({"key": (f, s), "t": t, "y": grp.y.to_numpy(),
                               "sd": grp.sd.to_numpy(), "H": tb.rows(t), "G": G,
                               "gidx": np.array(gi + ucm),
                               "xy": grp[["x_km", "y_km"]].to_numpy()[0], "rows": grp.index})
    pr._cmp_sites = []
    for sen, grp in pr.cmp.groupby("sensor", sort=True):
        t = grp.t.to_numpy()
        if t.size < 2:
            continue
        pr._cmp_sites.append({"sensor": sen, "t": t, "y": grp.y.to_numpy(),
                              "seg": grp.seg.to_numpy(), "H": tb.rows(t),
                              "xy": grp[["x_km", "y_km"]].to_numpy()[0], "rows": grp.index})
    U = pr.insar["units"]
    pr._unit_xy = U[["x_km", "y_km"]].to_numpy() if len(U) else np.zeros((0, 2))
    for gr in pr.insar["groups"]:
        gr["_h"] = tb.rows([gr["t"]])[0]
        gr["_gidx"] = np.array([pr.gindex[f"ioff:{int(u)}"] for u in gr["unit"]])


def set_ell(pr: Problem, ell: float) -> None:
    if pr._cache_ell == ell:
        return
    pr.lat.setup(ell)
    for s in pr._lev_sites + pr._gnss_sites + pr._cmp_sites:
        s["phi"] = pr.lat.weights(s["xy"][None])[0]
    if len(pr.insar["groups"]):
        Pc = pr.lat.weights(pr.cell_xy)
        pr._phi_units = np.stack([Pc[c].mean(0) for c in pr.insar["unit_cells"]])
    pr._cache_ell = ell


# ============================================================================ contributions
def contribution(pr: Problem, fam: str, h: dict) -> M.Contribution:
    set_ell(pr, h["ell_km"])
    asm = M.Assembler(pr.K, pr.m, pr.ng)
    if fam == "lev":
        for s in pr._lev_sites:
            W, ld, ne = M.site_local(s["t"], np.zeros(s["t"].size, int), h["sig_lev"],
                                     h["sig_rb"])
            asm.add_site(s["phi"], s["H"], W, s["y"], gidx=[0],
                         G=(s["t"] - C.T_REF)[:, None], logdet=ld, n_eff=ne)
    elif fam == "gnss":
        for s in pr._gnss_sites:
            v = h["sig_gnss"] ** 2 + s["sd"] ** 2
            asm.add_site(s["phi"], s["H"], np.diag(1.0 / v), s["y"], gidx=s["gidx"],
                         G=s["G"], logdet=float(np.sum(np.log(v))), n_eff=s["t"].size)
    elif fam == "cmp":
        for s in pr._cmp_sites:
            f = float(pr.fdict.get(s["sensor"], 1.0))
            W, ld, ne = M.site_local(s["t"], s["seg"], h["sig_cmp"], h["sig_rc"])
            asm.add_site(f * s["phi"], s["H"], W, s["y"], gidx=[0],
                         G=f * (s["t"] - C.T_REF)[:, None], logdet=ld, n_eff=ne)
    elif fam == "insar":
        for gr in pr.insar["groups"]:
            u = gr["unit"]
            W, ld, ne = M.ramp_month(pr._unit_xy[u], gr["nep"], h["sig_ins"], h["sig_aps"],
                                     h["ell_aps_km"])
            asm.add_month(pr._phi_units[u], gr["_h"], W, gr["y"], gr["_gidx"], logdet=ld,
                          n_eff=ne)
    else:
        raise ValueError(fam)
    return asm.result()


def lev_subset_contribution(pr: Problem, h: dict, sids) -> M.Contribution:
    """Leveling contribution of the benchmarks in ``sids`` only (to subtract for a block)."""
    set_ell(pr, h["ell_km"])
    sids = set(sids)
    asm = M.Assembler(pr.K, pr.m, pr.ng)
    for s in pr._lev_sites:
        if s["sid"] not in sids:
            continue
        W, ld, ne = M.site_local(s["t"], np.zeros(s["t"].size, int), h["sig_lev"],
                                 h["sig_rb"])
        asm.add_site(s["phi"], s["H"], W, s["y"], gidx=[0], G=(s["t"] - C.T_REF)[:, None],
                     logdet=ld, n_eff=ne)
    return asm.result()


def subtract(a: M.Contribution, b: M.Contribution) -> M.Contribution:
    return M.Contribution(a.P - b.P, a.b - b.b, a.yWy - b.yWy, a.logdetR - b.logdetR,
                          a.n_eff - b.n_eff)


def total_contribution(pr: Problem, h: dict, families=None) -> M.Contribution:
    out = M.empty_contribution(pr.N)
    for fam in families or pr.families:
        out += contribution(pr, fam, h)
    return out


def glob_prec(pr: Problem, h: dict) -> np.ndarray:
    p = np.empty(pr.ng)
    for i, n in enumerate(pr.gnames):
        if n == "mu_v":
            p[i] = 1.0 / C.MU_V_SD ** 2
        elif n.startswith("grate:"):
            p[i] = 1.0 / h["sig_d"] ** 2
        elif n.startswith("gframe:"):
            p[i] = 1.0 / C.FRAME_RATE_SD ** 2
        elif n.startswith("gcm:"):
            p[i] = 1.0 / h["sig_cm"] ** 2
        else:                                   # offsets: diffuse
            p[i] = 1.0 / C.VAGUE_SD ** 2
    return p


def fit(pr: Problem, h: dict, *, contrib: M.Contribution | None = None,
        want_cov: bool = False) -> M.Posterior:
    c = contrib if contrib is not None else total_contribution(pr, h)
    J = pr.tb.prior_precision(h["sig_v0"], h["q"], h["sig_s"])
    set_ell(pr, h["ell_km"])
    return M.solve(c, J, pr.lat, glob_prec(pr, h), want_cov=want_cov)


# ============================================================================ predictions
def field_rows(pr: Problem, xy: np.ndarray, t: np.ndarray, *, seasonal: bool = True,
               factor: np.ndarray | None = None) -> np.ndarray:
    """Rows (n, N) of u(x_n, t_n) (+ mu_v), optionally multiplied by ``factor``."""
    phis = pr.lat.weights(np.atleast_2d(xy))
    if len(phis) == 1 and len(t) > 1:
        phis = np.repeat(phis, len(t), 0)
    H = pr.tb.rows(t, seasonal=seasonal)
    G = M.functional_rows(pr.tb, phis, H, pr.ng, mu_coef=np.asarray(t) - C.T_REF)
    if factor is not None:
        G *= np.asarray(factor)[:, None]
    return G


def site_mean(pr: Problem, post: M.Posterior, s: dict, fam: str) -> np.ndarray:
    """Posterior mean of the operator part H theta (+ globals) at a site's observations."""
    th = post.theta
    Z = th[:pr.L].reshape(pr.K, pr.m)
    mu = th[pr.L + 0]
    base = s["H"] @ (Z @ s["phi"]) + mu * (s["t"] - C.T_REF)
    if fam == "cmp":
        return float(pr.fdict.get(s["sensor"], 1.0)) * base
    if fam == "gnss":
        g = th[pr.L + s["gidx"]]
        return s["H"] @ (Z @ s["phi"]) + s["G"] @ g
    return base


def local_residuals(t, seg, r, sig, sig_r) -> np.ndarray:
    """Residual after the local offsets (per segment) and random velocity, posterior mean."""
    t = np.asarray(t, float)
    tau = t - t.mean()
    segs, inv = np.unique(seg, return_inverse=True)
    E = np.zeros((t.size, segs.size))
    E[np.arange(t.size), inv] = 1.0
    Lm = np.column_stack([E, tau])
    prec = Lm.T @ Lm / sig ** 2
    prec[-1, -1] += 1.0 / sig_r ** 2
    est = np.linalg.solve(prec + 1e-12 * np.eye(len(prec)), Lm.T @ r / sig ** 2)
    return r - Lm @ est


def outliers(pr: Problem, post: M.Posterior, h: dict, fam: str) -> list:
    """Row ids (column ``rid`` of the family table, else the problem's row index) with
    |standardised residual| > 5."""
    rows = _outlier_rows(pr, post, h, fam)
    tab = {"lev": pr.lev, "gnss": pr.gnss, "cmp": pr.cmp}[fam]
    if "rid" in tab.columns:
        return sorted(int(x) for x in tab.loc[rows, "rid"])
    return sorted(int(x) for x in rows)


def _outlier_rows(pr: Problem, post: M.Posterior, h: dict, fam: str) -> list:
    set_ell(pr, h["ell_km"])
    out = []
    if fam == "lev":
        for s in pr._lev_sites:
            e = local_residuals(s["t"], np.zeros(s["t"].size, int),
                                s["y"] - site_mean(pr, post, s, "lev"), h["sig_lev"],
                                h["sig_rb"])
            out += list(np.asarray(s["rows"])[np.abs(e / h["sig_lev"]) > C.OUTLIER_Z])
    elif fam == "gnss":
        for s in pr._gnss_sites:
            e = s["y"] - site_mean(pr, post, s, "gnss")
            z = e / np.sqrt(h["sig_gnss"] ** 2 + s["sd"] ** 2)
            out += list(np.asarray(s["rows"])[np.abs(z) > C.OUTLIER_Z])
    elif fam == "cmp":
        for s in pr._cmp_sites:
            e = local_residuals(s["t"], s["seg"], s["y"] - site_mean(pr, post, s, "cmp"),
                                h["sig_cmp"], h["sig_rc"])
            out += list(np.asarray(s["rows"])[np.abs(e / h["sig_cmp"]) > C.OUTLIER_Z])
    return out


def estimate_f(pr: Problem, post: M.Posterior, cmp: pd.DataFrame) -> dict:
    """Depth fraction per ring/pile sensor: regression of the series on the fused u (posterior
    mean, from a fit WITHOUT rings and piles) with per-segment offsets; bounded."""
    out = {}
    for sen, grp in cmp.groupby("sensor"):
        t = grp.t.to_numpy()
        if t.size < 4:
            continue
        G = field_rows(pr, grp[["x_km", "y_km"]].to_numpy()[0], t)
        u = G @ post.theta
        segs, inv = np.unique(grp.seg.to_numpy(), return_inverse=True)
        X = np.zeros((t.size, segs.size + 1))
        X[np.arange(t.size), inv] = 1.0
        X[:, -1] = u
        beta, res, *_ = np.linalg.lstsq(X, grp.y.to_numpy(), rcond=None)
        dof = max(t.size - X.shape[1], 1)
        s2 = float(np.sum((grp.y.to_numpy() - X @ beta) ** 2) / dof)
        cov = s2 * np.linalg.pinv(X.T @ X)
        f = float(beta[-1])
        out[sen] = {"f_raw": f, "f_se": float(np.sqrt(max(cov[-1, -1], 0.0))),
                    "f": float(np.clip(f, *C.F_BOUNDS)), "n": int(t.size)}
    return out


# ============================================================================ hyperparameters
def optimise(pr: Problem, h0: dict, names: tuple, fams_var: tuple, maxfev: int,
             log=None) -> tuple[dict, dict]:
    """Nelder-Mead on log-hyperparameters ``names`` (bounded by clipping), maximising the
    restricted marginal likelihood; families not in ``fams_var`` are assembled once."""
    h = dict(h0)
    fixed_fams = [f for f in pr.families if f not in fams_var]
    ell_varies = "ell_km" in names
    fixed = None
    if not ell_varies and fixed_fams:
        fixed = total_contribution(pr, h, fixed_fams)
    lo = np.log([C.HYPER_BOUNDS[n][0] for n in names])
    hi = np.log([C.HYPER_BOUNDS[n][1] for n in names])
    hist = []
    t0 = time.time()

    def obj(x):
        xc = np.clip(x, lo, hi)
        pen = 1e3 * float(np.sum((x - xc) ** 2))
        hh = dict(h)
        hh.update({n: float(np.exp(v)) for n, v in zip(names, xc, strict=True)})
        try:
            if ell_varies or not fixed_fams:
                c = total_contribution(pr, hh)
            else:
                c = M.Contribution(fixed.P.copy(), fixed.b.copy(), fixed.yWy, fixed.logdetR,
                                   fixed.n_eff)
                for f in fams_var:
                    c += contribution(pr, f, hh)
            v = fit(pr, hh, contrib=c).m2loglik + pen
        except np.linalg.LinAlgError:
            v = 1e18
        hist.append({"x": {n: float(np.exp(a)) for n, a in zip(names, xc, strict=True)},
                     "m2ll": float(v), "s": round(time.time() - t0, 1)})
        if log is not None and len(hist) % 10 == 0:
            log(f"  eval {len(hist)} m2ll {v:.3f} best {min(r['m2ll'] for r in hist):.3f}")
        return v

    x0 = np.log([h[n] for n in names])
    res = optimize.minimize(obj, x0, method="Nelder-Mead",
                            options={"maxfev": maxfev, "xatol": 1e-3, "fatol": 0.05,
                                     "initial_simplex": _simplex(x0)})
    xb = np.clip(res.x, lo, hi)
    h.update({n: float(np.exp(v)) for n, v in zip(names, xb, strict=True)})
    info = {"names": list(names), "m2ll": float(res.fun), "nfev": int(res.nfev),
            "converged": bool(res.success), "message": str(res.message),
            "at_bound": [n for n, v, a, b in zip(names, xb, lo, hi, strict=True)
                         if abs(v - a) < 1e-6 or abs(v - b) < 1e-6],
            "seconds": round(time.time() - t0, 1), "history_tail": hist[-5:]}
    return h, info


def _simplex(x0: np.ndarray, step: float = 0.35) -> np.ndarray:
    S = np.tile(x0, (len(x0) + 1, 1))
    for i in range(len(x0)):
        S[i + 1, i] += step
    return S


def dump(obj, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=_js))


def _js(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
