"""WP S5 repair round, amendment 4: the InSAR-supported monthly anomaly (stage 2).

The annual fusion (round 1) carries annual knots and one constant seasonal harmonic. Stage 2
models what InSAR sees at monthly resolution beyond that: the InSAR month-group residuals of
an annual fit, r_e = y_e - (u_annual + seasonal + unit offset) (posterior means), are

    r_e = Phi_e a_{m(e)} + plane_e + aps_e + white_e

* a_m (m lattice nodes, the annual model's node lattice): a monthly anomaly field, Matern
  (nu 3/2, range ell_a, unit variance) in space, stationary AR(1) in months:
  a_1 ~ N(0, S), a_m = rho a_{m-1} + w_m, w_m ~ N(0, (1 - rho^2) S), S = sig_a^2 K_ZZ(ell_a);
* plane_e: a free plane per month group (the per-epoch reference nuisance), projected out
  exactly as in the annual operator (``s5model.ramp_month``);
* aps_e + white_e: exponential covariance sig_aps2^2 exp(-d / ell_aps2) + sig_ins2^2, divided by
  the number of epochs averaged per unit (stage-2 noise hyperparameters).

APS is white across months, the anomaly is not: that is what separates them. Exact Gaussian
inference: the posterior precision is block tridiagonal in months (block LDL', no dense
matrix); restricted marginal likelihood as in the annual model. Two-stage approximation: the
annual posterior mean is held fixed (its uncertainty is added in quadrature downstream).

Support. The plane of each month is invisible to InSAR, so the variance reduction is measured
on the PLANE-FREE anomaly: for month m with footprint F_m (cells inside a 4 km unit with an
InSAR value that month, at least ``S2_MIN_FOOTPRINT`` cells), psi(x) = phi(x) - x~(x)'
(X_F'X_F)^-1 X_F' Phi_F with x~ = (1, east, north) centred on F_m, and
VR_dt(x, m) = 1 - psi' Sigma_m psi / psi' S psi. A cell-month is supported when x is in F_m and
VR_dt >= 0.5.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg as sla
from scipy import optimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import s5model as M  # noqa: E402
import s5r_const as R  # noqa: E402

LOG2PI = math.log(2.0 * math.pi)


# ============================================================================ inputs
def residual_groups(pr, post) -> list[dict]:
    """InSAR month-group residuals of an annual fit (``s5fit.Problem``, posterior): y - (field
    at the unit, incl. the seasonal pair) - unit offset (posterior means). The fan-wide mean
    velocity is constant within a month and drops out with the per-month plane."""
    Z = post.theta[:pr.L].reshape(pr.K, pr.m)
    out = []
    for gr in pr.insar["groups"]:
        u = np.asarray(gr["unit"], int)
        fit = pr._phi_units[u] @ (Z.T @ gr["_h"]) + post.theta[pr.L + gr["_gidx"]]
        out.append({"seg": int(gr["seg"]), "month": str(gr["month"]), "t": float(gr["t"]),
                    "unit": u, "r": np.asarray(gr["y"], float) - fit,
                    "nep": np.asarray(gr["nep"], float)})
    return out


def month_range(first: str, last: str) -> list[str]:
    return [str(p) for p in pd.period_range(first, last, freq="M")]


# ============================================================================ block tridiagonal
class BTD:
    """Symmetric positive-definite block-tridiagonal matrix with diagonal blocks ``D`` (M, m, m)
    and one constant symmetric off-diagonal block ``B`` (m, m): block LDL' factorisation."""

    def __init__(self, D: np.ndarray, B: np.ndarray):
        self.M, self.m = D.shape[0], D.shape[1]
        self.B = B
        self.cf = []
        self.logdet = 0.0
        prev = None
        for k in range(self.M):
            Dt = D[k].copy()
            if k > 0:
                Dt -= B @ sla.cho_solve(prev, B, check_finite=False)
            Dt = 0.5 * (Dt + Dt.T)
            c = sla.cho_factor(Dt, lower=True, check_finite=False)
            self.cf.append(c)
            self.logdet += 2.0 * float(np.sum(np.log(np.diag(c[0]))))
            prev = c

    def _inv(self, k: int, X: np.ndarray) -> np.ndarray:
        return sla.cho_solve(self.cf[k], X, check_finite=False)

    def solve(self, b: np.ndarray) -> np.ndarray:
        """x with P x = b; b of shape (M, m) or (M, m, n)."""
        z = [b[0]]
        for k in range(1, self.M):
            z.append(b[k] - self.B @ self._inv(k - 1, z[k - 1]))
        x = [None] * self.M
        x[-1] = self._inv(self.M - 1, z[-1])
        for k in range(self.M - 2, -1, -1):
            x[k] = self._inv(k, z[k] - self.B @ x[k + 1])
        return np.stack(x)

    def marginals(self) -> np.ndarray:
        """Diagonal blocks (M, m, m) of P^-1 (backward recursion)."""
        S = [None] * self.M
        S[-1] = self._inv(self.M - 1, np.eye(self.m))
        for k in range(self.M - 2, -1, -1):
            E = self._inv(k, self.B)                       # Dt_k^-1 B
            S[k] = self._inv(k, np.eye(self.m)) + E @ S[k + 1] @ E.T
            S[k] = 0.5 * (S[k] + S[k].T)
        return np.stack(S)

    def functional_cov(self, month: np.ndarray, G: np.ndarray) -> np.ndarray:
        """Covariance G P^-1 G' of functionals: row i is G[i] (m,) applied to block month[i]."""
        month = np.asarray(month, int)
        n = len(month)
        cov = np.zeros((n, n))
        Y = None
        for k in range(self.M):
            Gk = np.zeros((self.m, n))
            sel = np.flatnonzero(month == k)
            Gk[:, sel] = G[sel].T
            Y = Gk if k == 0 else Gk - self.B @ self._inv(k - 1, Y)
            cov += Y.T @ self._inv(k, Y)
        return 0.5 * (cov + cov.T)


# ============================================================================ stage-2 problem
class Stage2:
    """InSAR month-group residuals on the annual model's node lattice."""

    def __init__(self, nodes_km: np.ndarray, cell_xy_km: np.ndarray, unit_xy_km: np.ndarray,
                 unit_cells: list, months: list, groups: list, nu: float = 1.5):
        self.Z = np.asarray(nodes_km, float)
        self.m = len(self.Z)
        self.cell_xy = np.asarray(cell_xy_km, float)
        self.unit_xy = np.asarray(unit_xy_km, float)
        self.unit_cells = unit_cells
        self.months = list(months)
        self.M = len(self.months)
        if self.M < 2:
            raise ValueError("stage 2 needs at least two months")
        mi = {mo: i for i, mo in enumerate(self.months)}
        self.groups = []
        for g in groups:
            if g["month"] not in mi:
                continue
            gg = dict(g)
            gg["mi"] = mi[g["month"]]
            self.groups.append(gg)
        self.nu = nu
        self._dzz = M.dist(self.Z, self.Z)
        self._dcz = M.dist(self.cell_xy, self.Z)
        self._lat_key = None
        self._noise_key = None
        self.footprint = self.observed_cells()

    # ---- lattice of the anomaly (range ell_a)
    def lattice(self, ell: float) -> None:
        if self._lat_key == ell:
            return
        K = M.matern(self._dzz, ell, self.nu) + R.S2_JITTER * np.eye(self.m)
        cf = sla.cho_factor(K, lower=True, check_finite=False)
        self.K = K
        self.Kinv = sla.cho_solve(cf, np.eye(self.m), check_finite=False)
        self.Kinv = 0.5 * (self.Kinv + self.Kinv.T)
        self.logdetK = 2.0 * float(np.sum(np.log(np.diag(cf[0]))))
        self.Phi_cells = sla.cho_solve(cf, M.matern(self._dcz, ell, self.nu).T,
                                       check_finite=False).T
        self.Phi_units = np.stack([self.Phi_cells[c].mean(0) for c in self.unit_cells])
        self._cf = cf
        self._lat_key = ell
        self._pf = {}

    def point_weights(self, xy_km: np.ndarray) -> np.ndarray:
        k = M.matern(M.dist(np.atleast_2d(xy_km), self.Z), self._lat_key, self.nu)
        return sla.cho_solve(self._cf, k.T, check_finite=False).T

    # ---- noise of every group (cached by the three noise hyperparameters)
    def noise(self, sig_i: float, sig_aps: float, ell_aps: float) -> None:
        key = (sig_i, sig_aps, ell_aps)
        if self._noise_key == key:
            return
        self._W, ld, ne = [], 0.0, 0
        for g in self.groups:
            W, l_, n_ = M.ramp_month(self.unit_xy[g["unit"]], g["nep"], sig_i, sig_aps, ell_aps)
            self._W.append(W)
            ld += l_
            ne += n_
        self._ld, self._ne = ld, ne
        self._yWy = float(sum(g["r"] @ W @ g["r"] for g, W in zip(self.groups, self._W,
                                                                  strict=True)))
        self._noise_key = key

    # ---- assembly and solve
    def solve(self, h: dict) -> dict:
        self.lattice(h["ell_a"])
        self.noise(h["sig_ins2"], h["sig_aps2"], h["ell_aps2"])
        rho, sa2 = h["rho"], h["sig_a"] ** 2
        m, Mn = self.m, self.M
        A = np.zeros((Mn, m, m))
        b = np.zeros((Mn, m))
        for g, W in zip(self.groups, self._W, strict=True):
            Phi = self.Phi_units[g["unit"]]
            PW = Phi.T @ W
            A[g["mi"]] += PW @ Phi
            b[g["mi"]] += PW @ g["r"]
        Sinv = self.Kinv / sa2
        c = 1.0 / (1.0 - rho * rho)
        D = A.copy()
        for k in range(Mn):
            w = 1.0 if k in (0, Mn - 1) else 1.0 + rho * rho
            D[k] += c * w * Sinv
        B = -c * rho * Sinv
        P = BTD(D, B)
        x = P.solve(b)
        logdetS = m * math.log(sa2) + self.logdetK
        # Q = C_t^-1 (x) S^-1 with C_t the AR(1) correlation matrix, |C_t| = (1-rho^2)^(M-1)
        logdetQ = -m * (Mn - 1) * math.log(1.0 - rho * rho) - Mn * logdetS
        quad = self._yWy - float(np.sum(b * x))
        m2 = quad + P.logdet - logdetQ + self._ld + self._ne * LOG2PI
        return {"x": x, "P": P, "m2ll": m2, "quad": quad, "logdetP": P.logdet,
                "logdetQ": logdetQ, "logdetR": self._ld, "n_eff": self._ne}

    def m2ll_null(self, sig_i: float, sig_aps: float, ell_aps: float) -> float:
        """-2 log L of the residuals without an anomaly (noise only)."""
        self.noise(sig_i, sig_aps, ell_aps)
        return self._yWy + self._ld + self._ne * LOG2PI

    # ---- footprint and the plane-free operator
    def observed_cells(self) -> np.ndarray:
        """(M, A) boolean: cell inside a unit with an InSAR value in that month."""
        A = len(self.cell_xy)
        out = np.zeros((self.M, A), bool)
        for g in self.groups:
            for u in g["unit"]:
                out[g["mi"], self.unit_cells[int(u)]] = True
        return out

    def plane_free(self, k: int, phi: np.ndarray, xy: np.ndarray) -> np.ndarray | None:
        """psi rows (n, m) of the plane-free anomaly of month k at points xy with weights phi;
        None when month k has fewer than S2_MIN_FOOTPRINT footprint cells."""
        F = np.flatnonzero(self.footprint[k])
        if F.size < R.S2_MIN_FOOTPRINT:
            return None
        key = k
        if key not in self._pf:
            cxy = self.cell_xy[F]
            c0 = cxy.mean(0)
            XF = np.column_stack([np.ones(F.size), cxy - c0])
            self._pf[key] = (c0, np.linalg.solve(XF.T @ XF, XF.T @ self.Phi_cells[F]))
        c0, Bk = self._pf[key]
        xt = np.column_stack([np.ones(len(xy)), np.asarray(xy, float) - c0])
        return phi - xt @ Bk

    # ---- moments at the cells
    def cell_moments(self, h: dict, sol: dict) -> dict:
        """Posterior mean and variance of the anomaly at every cell and month, its prior
        variance, and the variance reduction of the plane-free anomaly (vr_dt; NaN in months
        without a footprint)."""
        self.lattice(h["ell_a"])
        S = sol["P"].marginals()
        sa2 = h["sig_a"] ** 2
        Phi = self.Phi_cells
        mean = sol["x"] @ Phi.T                                            # (M, A)
        var = np.stack([np.einsum("ij,ij->i", Phi @ S[k], Phi) for k in range(self.M)])
        prior = sa2 * np.einsum("ij,ij->i", Phi @ self.K, Phi)
        vr_dt = np.full((self.M, len(Phi)), np.nan)
        for k in range(self.M):
            psi = self.plane_free(k, Phi, self.cell_xy)
            if psi is None:
                continue
            pv = sa2 * np.einsum("ij,ij->i", psi @ self.K, psi)
            qv = np.einsum("ij,ij->i", psi @ S[k], psi)
            vr_dt[k] = 1.0 - qv / np.maximum(pv, 1e-12)
        var = np.clip(var, 0.0, None)
        support = self.footprint & np.nan_to_num(vr_dt >= R.VR_A_MIN, nan=False)
        return {"mean": mean, "var": var, "prior": prior, "vr": 1.0 - var / prior[None, :],
                "vr_dt": vr_dt, "support": support, "marginals": S}

    def point_vr_dt(self, h: dict, sol: dict, xy: np.ndarray, mi: np.ndarray,
                    marg: np.ndarray | None = None) -> np.ndarray:
        """VR of the plane-free anomaly at points xy (km) in months mi (NaN without a
        footprint)."""
        self.lattice(h["ell_a"])
        S = sol["P"].marginals() if marg is None else marg
        phi = self.point_weights(xy)
        out = np.full(len(xy), np.nan)
        sa2 = h["sig_a"] ** 2
        for k in np.unique(mi):
            sel = np.flatnonzero(mi == k)
            psi = self.plane_free(int(k), phi[sel], np.asarray(xy)[sel])
            if psi is None:
                continue
            pv = sa2 * np.einsum("ij,ij->i", psi @ self.K, psi)
            qv = np.einsum("ij,ij->i", psi @ S[k], psi)
            out[sel] = 1.0 - qv / np.maximum(pv, 1e-12)
        return out


# ============================================================================ hyperparameters
def _to_x(h: dict, names) -> np.ndarray:
    out = []
    for n in names:
        if n == "rho":
            p = min(max(h[n] / R.S2_BOUNDS["rho"][1], 1e-6), 1 - 1e-6)
            out.append(math.log(p / (1 - p)))
        else:
            out.append(math.log(h[n]))
    return np.array(out)


def _from_x(x: np.ndarray, names) -> tuple[dict, float]:
    h, pen = {}, 0.0
    for n, v in zip(names, x, strict=True):
        if n == "rho":
            vc = min(max(v, -12.0), 12.0)
            h[n] = R.S2_BOUNDS["rho"][1] / (1.0 + math.exp(-vc))
        else:
            lo, hi = (math.log(b) for b in R.S2_BOUNDS[n])
            vc = min(max(v, lo), hi)
            h[n] = math.exp(vc)
        pen += (v - vc) ** 2
    return h, 1e3 * pen


def _nm(obj, x0, maxfev, step=0.35):
    S = np.tile(x0, (len(x0) + 1, 1))
    for i in range(len(x0)):
        S[i + 1, i] += step
    return optimize.minimize(obj, x0, method="Nelder-Mead",
                             options={"maxfev": maxfev, "xatol": 1e-3, "fatol": 0.05,
                                      "initial_simplex": S})


def _at_bound(h: dict) -> list[str]:
    out = []
    for n, v in h.items():
        lo, hi = R.S2_BOUNDS[n]
        if n == "rho":
            if v > 0.999 * hi or v < 1e-4:
                out.append(n)
        elif abs(math.log(v) - math.log(lo)) < 1e-6 or abs(math.log(v) - math.log(hi)) < 1e-6:
            out.append(n)
    return out


def fit_null(s2: Stage2, h0: dict, maxfev: int | None = None) -> tuple[dict, dict]:
    """Noise-only model (no anomaly): sig_ins2, sig_aps2, ell_aps2."""
    maxfev = R.S2_MAXFEV_NULL if maxfev is None else maxfev
    names = R.S2_NOISE
    hist = []

    def obj(x):
        h, pen = _from_x(x, names)
        v = s2.m2ll_null(h["sig_ins2"], h["sig_aps2"], h["ell_aps2"]) + pen
        hist.append({"h": h, "m2ll": float(v)})
        return v

    res = _nm(obj, _to_x(h0, names), maxfev)
    best = min(hist, key=lambda r: r["m2ll"])
    h = dict(best["h"])
    return h, {"m2ll": best["m2ll"], "nfev": int(res.nfev), "converged": bool(res.success),
               "returned": "best evaluated point", "at_bound": _at_bound(h)}


def fit_hyper(s2: Stage2, h0: dict, maxfev: int | None = None,
              restart: int | None = None, log=None) -> tuple[dict, dict]:
    """Nelder-Mead on the transformed stage-2 hyperparameters (restricted marginal
    likelihood; bounds by clipping with a quadratic penalty, as ``s5fit.optimise``), then one
    restart from the optimum with a fresh simplex (step 0.2); returns the best evaluated
    point over both passes."""
    maxfev = R.S2_MAXFEV if maxfev is None else maxfev
    restart = R.S2_MAXFEV_RESTART if restart is None else restart
    names = R.S2_NAMES
    hist = []
    t0 = time.time()

    def obj(x):
        h, pen = _from_x(x, names)
        try:
            v = s2.solve(h)["m2ll"] + pen
        except np.linalg.LinAlgError:
            v = 1e18
        hist.append({"h": h, "m2ll": float(v), "s": round(time.time() - t0, 1)})
        if log is not None and len(hist) % 50 == 0:
            log(f"  stage-2 eval {len(hist)} m2ll {v:.3f} best "
                f"{min(r['m2ll'] for r in hist):.3f}")
        return v

    res1 = _nm(obj, _to_x(h0, names), maxfev)
    n1 = len(hist)
    res2 = _nm(obj, res1.x, restart, step=0.2) if restart > 0 else res1
    best = min(hist, key=lambda r: r["m2ll"])
    h = dict(best["h"])                     # the best evaluated point over both passes
    info = {"m2ll": best["m2ll"], "nfev_first": n1, "nfev_restart": len(hist) - n1,
            "converged_first": bool(res1.success), "converged_restart": bool(res2.success),
            "m2ll_first": float(res1.fun), "m2ll_restart": float(res2.fun),
            "returned": "best evaluated point", "at_bound": _at_bound(h),
            "seconds": round(time.time() - t0, 1), "history_tail": hist[-5:]}
    return h, info
