"""WP S5 repair round 2: the repaired fusion model (R1-R5 of s5r2_const), built on the round-1
problem structure (s5fit / s5model, imported read-only) and the repair-round-1 operators
(s5r_core: GNSS gap segments, leveling segments).

R1  Temporal prior (per lattice node, separable with the Matern K_ZZ as in round 1). With
    knots t_0 < ... < t_{T-1}, d_j = t_j - t_{j-1}:
        v_k = b + beta_k                                   (k = 0..T-1)
        u_k = sum_{j=1..k} d_j (b + (beta_{j-1} + beta_j) / 2) + eps_k   (k = 1..T-1; u_0 = 0)
    b ~ N(0, sig_v0^2) long-term velocity, beta_k ~ N(0, sig_be^2) white knot-velocity
    anomalies (they do not persist), eps_k ~ N(0, sig_ep^2) reverting displacement anomalies.
    The covariance of the round-1 basis coefficients (u_1..u_{T-1}, v_0..v_{T-1}) is
    A diag(s^2) A' (full rank); the seasonal pair keeps sig_s. Between knots u is the cubic
    Hermite interpolant (s5model.TimeBasis.rows, unchanged).
R2  Leveling: y = u + c_b(seg) + r_b (t - tbar) + camp_{county, month} + e; camp ~ N(0,
    sig_camp^2) global parameters (names ``lcamp:<county>:<YYYY-MM>``).
R3  GNSS: the white part sig_gnss^2 + sd^2 plus station-level sig_gc^2 exp(-|dt| / tau_g).
R4  Rings, piles: sig_cmp^2 I + sig_rc^2 tau tau' + sig_cc^2 exp(-|dt| / tau_c).
R5  InSAR: per-unit random rate r_u (t - tref_u), r_u ~ N(0, sig_ir^2) (names ``irate:<u>``).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import s5const as C
import s5fit as F
import s5model as M
import s5r2_const as K
import s5r_core as RC
from scipy import linalg as sla
from scipy import optimize


# ============================================================================ R1 temporal prior
def temporal_cov_map(knots) -> np.ndarray:
    """A (Kt, 2T): basis coefficients (u_1..u_{T-1}, v_0..v_{T-1}) as a linear map of
    z = (b, beta_0..beta_{T-1}, eps_1..eps_{T-1})."""
    kn = np.asarray(knots, float)
    T = kn.size
    Kt = 2 * T - 1
    A = np.zeros((Kt, 2 * T))
    ib, ibe, iep = 0, 1, 1 + T
    for k in range(1, T):                       # u_k rows 0..T-2
        r = k - 1
        for j in range(1, k + 1):
            d = kn[j] - kn[j - 1]
            A[r, ib] += d
            A[r, ibe + j - 1] += d / 2.0
            A[r, ibe + j] += d / 2.0
        A[r, iep + k - 1] = 1.0
    for k in range(T):                          # v_k rows T-1..2T-2
        r = T - 1 + k
        A[r, ib] = 1.0
        A[r, ibe + k] = 1.0
    return A


def prior_precision2(tb: M.TimeBasis, h: dict) -> np.ndarray:
    T = tb.T
    A = temporal_cov_map(tb.knots)
    s2 = np.r_[h["sig_v0"] ** 2, np.full(T, h["sig_be"] ** 2), np.full(T - 1, h["sig_ep"] ** 2)]
    Ct = (A * s2) @ A.T
    J = np.zeros((tb.K, tb.K))
    c = sla.cho_factor(Ct, lower=True, check_finite=False)
    Ji = sla.cho_solve(c, np.eye(tb.Kt), check_finite=False)
    J[:tb.Kt, :tb.Kt] = 0.5 * (Ji + Ji.T)
    J[tb.ic, tb.ic] = J[tb.isn, tb.isn] = 1.0 / h["sig_s"] ** 2
    return J


# ============================================================================ problem
def make_problem2(data: dict, knots, families=F.FAMILIES, *, masks=None, fdict=None,
                  lattice=None) -> F.Problem:
    """Round-1 problem + repair-round-1 segments (s5r_core) + R2 campaign and R5 unit-rate
    globals. ``data['lev']`` needs a ``camp`` column; ``data['gnss']`` may carry ``seg``."""
    pr = RC.make_problem_v2(data, knots, families, masks=masks, fdict=fdict, lattice=lattice)
    # ---- R2 campaigns
    if len(pr.lev) and "camp" in pr.lev.columns:
        for s in pr._lev_sites:
            camps = pr.lev.loc[s["rows"], "camp"].astype(str).to_numpy()
            idx = []
            for cname in camps:
                nm = f"lcamp:{cname}"
                if nm not in pr.gindex:
                    pr.gnames.append(nm)
                    pr.gindex[nm] = len(pr.gnames) - 1
                idx.append(pr.gindex[nm])
            uc = sorted(set(idx))
            G = np.zeros((s["t"].size, 1 + len(uc)))
            G[:, 0] = s["t"] - C.T_REF
            pos = {c: j for j, c in enumerate(uc)}
            for r, c in enumerate(idx):
                G[r, 1 + pos[c]] = 1.0
            s["G2"] = G
            s["gidx2"] = np.array([0] + uc, int)
    else:
        for s in pr._lev_sites:
            s["G2"] = (s["t"] - C.T_REF)[:, None]
            s["gidx2"] = np.array([0], int)
    # ---- R5 unit rates (reference time = mean of the unit's group times)
    times: dict[int, list] = {}
    for gr in pr.insar["groups"]:
        for u in gr["unit"]:
            times.setdefault(int(u), []).append(gr["t"])
    pr._unit_tref = {u: float(np.mean(v)) for u, v in times.items()}
    for u in sorted(times):
        nm = f"irate:{u}"
        if nm not in pr.gindex:
            pr.gnames.append(nm)
            pr.gindex[nm] = len(pr.gnames) - 1
    for gr in pr.insar["groups"]:
        gr["_ridx"] = np.array([pr.gindex[f"irate:{int(u)}"] for u in gr["unit"]], int)
        gr["_rdt"] = np.array([gr["t"] - pr._unit_tref[int(u)] for u in gr["unit"]])
    return pr


def glob_prec2(pr: F.Problem, h: dict) -> np.ndarray:
    p = F.glob_prec(pr, h)
    for i, n in enumerate(pr.gnames):
        if n.startswith("lcamp:"):
            p[i] = 1.0 / h["sig_camp"] ** 2
        elif n.startswith("irate:"):
            p[i] = 1.0 / h["sig_ir"] ** 2
    return p


# ============================================================================ local covariances
def expcov(t: np.ndarray, sig: float, tau: float) -> np.ndarray:
    t = np.asarray(t, float)
    return sig ** 2 * np.exp(-np.abs(t[:, None] - t[None, :]) / tau)


def gnss_R(s: dict, h: dict) -> np.ndarray:
    return np.diag(h["sig_gnss"] ** 2 + s["sd"] ** 2) + expcov(s["t"], h["sig_gc"], h["tau_g"])


def cmp_R(t: np.ndarray, h: dict) -> np.ndarray:
    tau = t - t.mean()
    return (h["sig_cmp"] ** 2 * np.eye(t.size) + h["sig_rc"] ** 2 * np.outer(tau, tau)
            + expcov(t, h["sig_cc"], h["tau_c"]))


def _segE(seg) -> np.ndarray:
    segs, inv = np.unique(seg, return_inverse=True)
    E = np.zeros((len(seg), segs.size))
    E[np.arange(len(seg)), inv] = 1.0
    return E


def _add_month2(asm: M.Assembler, Phi_e, h_t, W, y, gidx, G, logdet, n_eff) -> None:
    """InSAR month group with a general global design G (n, ng_e) over ``gidx``."""
    nz = np.flatnonzero(h_t)
    PW = Phi_e.T @ W
    Mm = PW @ Phi_e
    v = PW @ y
    PWG = PW @ G
    for a in nz:
        for c in nz:
            asm.P4[a, :, c, :] += h_t[a] * h_t[c] * Mm
        asm.blat[a] += h_t[a] * v
        blk = asm.Plg[a]
        blk[:, gidx] += h_t[a] * PWG
    WG = W @ G
    asm.Pgg[np.ix_(gidx, gidx)] += G.T @ WG
    asm.bg[gidx] += WG.T @ y
    asm.yWy += float(y @ W @ y)
    asm.logdetR += float(logdet)
    asm.n_eff += int(n_eff)


# ============================================================================ contributions
def contribution2(pr: F.Problem, fam: str, h: dict, sids=None) -> M.Contribution:
    F.set_ell(pr, h["ell_km"])
    asm = M.Assembler(pr.K, pr.m, pr.ng)
    if fam == "lev":
        keep = None if sids is None else set(sids)
        for s in pr._lev_sites:
            if keep is not None and s["sid"] not in keep:
                continue
            W, ld, ne = M.site_local(s["t"], s["seg"], h["sig_lev"], h["sig_rb"])
            asm.add_site(s["phi"], s["H"], W, s["y"], gidx=s["gidx2"], G=s["G2"], logdet=ld,
                         n_eff=ne)
    elif fam == "gnss":
        for s in pr._gnss_sites:
            R = gnss_R(s, h)
            cR = sla.cho_factor(R, lower=True, check_finite=False)
            W = sla.cho_solve(cR, np.eye(len(R)), check_finite=False)
            asm.add_site(s["phi"], s["H"], 0.5 * (W + W.T), s["y"], gidx=s["gidx"], G=s["G"],
                         logdet=2.0 * float(np.sum(np.log(np.diag(cR[0])))),
                         n_eff=s["t"].size)
    elif fam == "cmp":
        for s in pr._cmp_sites:
            f = float(pr.fdict.get(s["sensor"], 1.0))
            W, ld = M.projected_precision(cmp_R(s["t"], h), _segE(s["seg"]))
            asm.add_site(f * s["phi"], s["H"], W, s["y"], gidx=[0],
                         G=f * (s["t"] - C.T_REF)[:, None], logdet=ld,
                         n_eff=s["t"].size - len(np.unique(s["seg"])))
    elif fam == "insar":
        for gr in pr.insar["groups"]:
            u = gr["unit"]
            W, ld, ne = M.ramp_month(pr._unit_xy[u], gr["nep"], h["sig_ins"], h["sig_aps"],
                                     h["ell_aps_km"])
            n = len(u)
            G = np.zeros((n, 2 * n))
            G[np.arange(n), np.arange(n)] = 1.0
            G[np.arange(n), n + np.arange(n)] = gr["_rdt"]
            gidx = np.r_[gr["_gidx"], gr["_ridx"]]
            _add_month2(asm, pr._phi_units[u], gr["_h"], W, gr["y"], gidx, G, ld, ne)
    else:
        raise ValueError(fam)
    return asm.result()


def total2(pr: F.Problem, h: dict, families=None) -> M.Contribution:
    out = M.empty_contribution(pr.N)
    for fam in families or pr.families:
        out += contribution2(pr, fam, h)
    return out


def fit2(pr: F.Problem, h: dict, *, contrib=None, want_cov=False) -> M.Posterior:
    c = contrib if contrib is not None else total2(pr, h)
    J = prior_precision2(pr.tb, h)
    F.set_ell(pr, h["ell_km"])
    return M.solve(c, J, pr.lat, glob_prec2(pr, h), want_cov=want_cov)


# ============================================================================ rows and residuals
def camp_cols(pr: F.Problem, camps) -> np.ndarray:
    """(n, N) rows with 1 at each survey's campaign global (0 where not a parameter)."""
    G = np.zeros((len(camps), pr.N))
    for r, cname in enumerate(camps):
        j = pr.gindex.get(f"lcamp:{cname}")
        if j is not None:
            G[r, pr.L + j] = 1.0
    return G


def lev_mean(pr, post, s) -> np.ndarray:
    th = post.theta
    Z = th[:pr.L].reshape(pr.K, pr.m)
    return s["H"] @ (Z @ s["phi"]) + s["G2"] @ th[pr.L + s["gidx2"]]


def outliers2(pr: F.Problem, post: M.Posterior, h: dict, fam: str) -> list:
    """rids with |standardised residual| > 5 (round-1 rule) under the repaired noise models."""
    F.set_ell(pr, h["ell_km"])
    rows = []
    if fam == "lev":
        for s in pr._lev_sites:
            e = F.local_residuals(s["t"], s["seg"], s["y"] - lev_mean(pr, post, s),
                                  h["sig_lev"], h["sig_rb"])
            rows += list(np.asarray(s["rows"])[np.abs(e / h["sig_lev"]) > C.OUTLIER_Z])
    elif fam == "gnss":
        for s in pr._gnss_sites:
            e = s["y"] - F.site_mean(pr, post, s, "gnss")
            z = e / np.sqrt(h["sig_gnss"] ** 2 + h["sig_gc"] ** 2 + s["sd"] ** 2)
            rows += list(np.asarray(s["rows"])[np.abs(z) > C.OUTLIER_Z])
    elif fam == "cmp":
        sd = np.sqrt(h["sig_cmp"] ** 2 + h["sig_cc"] ** 2)
        for s in pr._cmp_sites:
            e = F.local_residuals(s["t"], s["seg"], s["y"] - F.site_mean(pr, post, s, "cmp"),
                                  sd, h["sig_rc"])
            rows += list(np.asarray(s["rows"])[np.abs(e / sd) > C.OUTLIER_Z])
    tab = {"lev": pr.lev, "gnss": pr.gnss, "cmp": pr.cmp}[fam]
    return sorted(int(x) for x in tab.loc[rows, "rid"])


# ============================================================================ optimiser
def optimise2(pr: F.Problem, h0: dict, names: tuple, fams_var: tuple, maxfev: int,
              restart: int = 0, log=None) -> tuple[dict, dict]:
    """Nelder-Mead on log-hyperparameters (bounds by clipping + quadratic penalty), maximising
    the restricted marginal likelihood; one restart from the best evaluated point. The
    returned point is the best EVALUATED point (penalty-free value reported)."""
    h = dict(h0)
    fixed_fams = [f for f in pr.families if f not in fams_var]
    ell_varies = "ell_km" in names
    fixed = total2(pr, h, fixed_fams) if (not ell_varies and fixed_fams) else None
    lo = np.log([K.HYPER_BOUNDS[n][0] for n in names])
    hi = np.log([K.HYPER_BOUNDS[n][1] for n in names])
    hist = []
    t0 = time.time()

    def obj(x):
        xc = np.clip(x, lo, hi)
        pen = 1e3 * float(np.sum((x - xc) ** 2))
        hh = dict(h)
        hh.update({n: float(np.exp(v)) for n, v in zip(names, xc, strict=True)})
        try:
            if fixed is None:
                c = total2(pr, hh)
            else:
                c = M.Contribution(fixed.P.copy(), fixed.b.copy(), fixed.yWy, fixed.logdetR,
                                   fixed.n_eff)
                for f in fams_var:
                    c += contribution2(pr, f, hh)
            m2 = fit2(pr, hh, contrib=c).m2loglik
        except (np.linalg.LinAlgError, ValueError):
            m2 = 1e18
        hist.append((xc.copy(), m2, m2 + pen))
        if log is not None and len(hist) % 25 == 0:
            log(f"  eval {len(hist)} m2ll {m2:.3f} best {min(r[1] for r in hist):.3f} "
                f"({time.time() - t0:.0f} s)")
        return m2 + pen

    x0 = np.log([h[n] for n in names])
    res = optimize.minimize(obj, x0, method="Nelder-Mead",
                            options={"maxfev": maxfev, "xatol": 1e-3, "fatol": 0.05,
                                     "initial_simplex": F._simplex(x0)})
    passes = [{"nfev": int(res.nfev), "converged": bool(res.success),
               "message": str(res.message)}]
    if restart:
        xb = min(hist, key=lambda r: r[2])[0]
        res2 = optimize.minimize(obj, xb, method="Nelder-Mead",
                                 options={"maxfev": restart, "xatol": 1e-3, "fatol": 0.05,
                                          "initial_simplex": F._simplex(xb, 0.15)})
        passes.append({"nfev": int(res2.nfev), "converged": bool(res2.success),
                       "message": str(res2.message)})
    best = min(hist, key=lambda r: r[2])
    h.update({n: float(np.exp(v)) for n, v in zip(names, best[0], strict=True)})
    info = {"names": list(names), "m2ll_best_evaluated": float(best[1]),
            "n_evaluations": len(hist), "passes": passes,
            "converged_last_pass": passes[-1]["converged"],
            "at_bound": [n for n, v, a, b in zip(names, best[0], lo, hi, strict=True)
                         if abs(v - a) < 1e-6 or abs(v - b) < 1e-6],
            "seconds": round(time.time() - t0, 1)}
    return h, info


# ============================================================================ predictions
def lev_pred_heldout(pr, post, h, xy, t_tr, y_tr, camps_tr, t_new, camps_new):
    """Predictive of a benchmark's new surveys given its own training surveys (exact local
    offset and random velocity conditioned on the training surveys; campaign offsets of the
    training surveys are posterior parameters; a new survey's campaign that is not a parameter
    adds sig_camp^2)."""
    t_tr, y_tr, t_new = (np.asarray(a, float) for a in (t_tr, y_tr, t_new))
    G_tr = F.field_rows(pr, xy, t_tr) + camp_cols(pr, camps_tr)
    G_nw = F.field_rows(pr, xy, t_new) + camp_cols(pr, camps_new)
    tb = t_tr.mean()
    L_tr = np.column_stack([np.ones(t_tr.size), t_tr - tb])
    L_nw = np.column_stack([np.ones(t_new.size), t_new - tb])
    s2 = h["sig_lev"] ** 2
    Lam = L_tr.T @ L_tr / s2
    Lam[1, 1] += 1.0 / h["sig_rb"] ** 2
    Li = np.linalg.inv(Lam)
    Mm = Li @ L_tr.T / s2
    g = G_nw - L_nw @ Mm @ G_tr
    mean = g @ post.theta + L_nw @ Mm @ y_tr
    newc = np.array([f"lcamp:{c}" not in pr.gindex for c in camps_new], float)
    var = (post.var_of(g) + np.einsum("ij,jk,ik->i", L_nw, Li, L_nw) + s2
           + newc * h["sig_camp"] ** 2)
    return mean, np.sqrt(var)


def field_increment_var(pr, post, xy, t0, t1) -> np.ndarray:
    t1 = np.atleast_1d(np.asarray(t1, float))
    t0 = np.broadcast_to(np.asarray(t0, float), t1.shape).copy()
    g = F.field_rows(pr, xy, t1) - F.field_rows(pr, xy, t0)
    return post.var_of(g)


def campaign_ids(lev: pd.DataFrame, town: pd.Series) -> np.ndarray:
    """County (first three characters of the panel's town) x survey month."""
    county = lev["sid"].map(town).astype(str).str[:3]
    return (county + ":" + lev["date"].astype(str).str[:7]).to_numpy()
