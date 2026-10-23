"""WP S5 space-time subsidence fusion: model core (numpy / scipy, CPU only).

Model (plan WP S5)
------------------
Vertical displacement u(x, t) (mm, positive = sinking, u(x, T_REF) = 0) is

    u(x, t) = mu_v (t - T_REF) + sum_i phi_i(x) w_i(t) + phi(x) . (a cos 2 pi t + b sin 2 pi t)

* Space: a predictive-process basis on a regular node lattice, phi(x) = k(x, Z) K_ZZ^-1, with a
  unit-variance Matern (nu = 3/2) covariance of range ``ell``.
* Time: an integrated random walk on velocity (continuous white-noise acceleration) at annual
  knots, state (u_k, v_k) per node; u_0 = 0, v_0 ~ N(0, sig_v0^2 K), increments of intensity q.
  Between knots u is the cubic Hermite interpolant (the IRW conditional mean).
* Separable: the latent prior covariance is C_t (x) K_ZZ, so its precision is
  blockdiag(J_t, I_2 / sig_s^2) (x) K_ZZ^-1, with the seasonal pair (a, b) ~ sig_s^2 K_ZZ.
* Instruments enter through their own operators (``s5fit``): leveling and partial-column
  instruments with diffuse per-site (per-segment) offsets and a per-site random velocity, both
  eliminated analytically (restricted likelihood); GNSS with explicit offsets, station rates,
  family frame rates and family-month common modes; InSAR month groups with free per-month
  planar ramps (projected out) and explicit per-(block, segment) offsets.

Inference is exact Gaussian (dense Cholesky of the posterior precision); hyperparameters by
maximising the (restricted) marginal likelihood.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import linalg as sla

SQ3 = math.sqrt(3.0)
SQ5 = math.sqrt(5.0)
LOG2PI = math.log(2.0 * math.pi)


# ============================================================================ kernels, lattice
def dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.atleast_2d(np.asarray(a, float))
    b = np.atleast_2d(np.asarray(b, float))
    d2 = (a[:, None, 0] - b[None, :, 0]) ** 2 + (a[:, None, 1] - b[None, :, 1]) ** 2
    return np.sqrt(d2)


def matern(d: np.ndarray, ell: float, nu: float = 1.5) -> np.ndarray:
    if nu == 0.5:
        return np.exp(-d / ell)
    if nu == 1.5:
        r = SQ3 * d / ell
        return (1.0 + r) * np.exp(-r)
    if nu == 2.5:
        r = SQ5 * d / ell
        return (1.0 + r + r * r / 3.0) * np.exp(-r)
    raise ValueError(f"nu {nu} not supported")


def build_lattice(cell_xy_km: np.ndarray, spacing: float, reach: float) -> np.ndarray:
    """Regular lattice nodes (km) within ``reach`` of any cell centre."""
    lo = cell_xy_km.min(0) - reach
    hi = cell_xy_km.max(0) + reach
    xs = np.arange(lo[0], hi[0] + 1e-9, spacing)
    ys = np.arange(lo[1], hi[1] + 1e-9, spacing)
    gx, gy = np.meshgrid(xs, ys)
    cand = np.column_stack([gx.ravel(), gy.ravel()])
    keep = np.zeros(len(cand), bool)
    for i0 in range(0, len(cand), 2000):
        keep[i0:i0 + 2000] = dist(cand[i0:i0 + 2000], cell_xy_km).min(1) <= reach
    return cand[keep]


class Lattice:
    """Node lattice with a Matern covariance of range ``ell`` (set by :meth:`setup`)."""

    def __init__(self, nodes_km: np.ndarray, nu: float = 1.5, jitter: float = 1e-7):
        self.Z = np.asarray(nodes_km, float)
        self.m = len(self.Z)
        self.nu = nu
        self.jitter = jitter
        self._dzz = dist(self.Z, self.Z)
        self.ell = None

    def setup(self, ell: float) -> None:
        if self.ell == ell:
            return
        K = matern(self._dzz, ell, self.nu) + self.jitter * np.eye(self.m)
        self.cf = sla.cho_factor(K, lower=True, check_finite=False)
        self.Kinv = sla.cho_solve(self.cf, np.eye(self.m), check_finite=False)
        self.Kinv = 0.5 * (self.Kinv + self.Kinv.T)
        self.logdetK = 2.0 * float(np.sum(np.log(np.diag(self.cf[0]))))
        self.ell = ell

    def weights(self, xy_km: np.ndarray) -> np.ndarray:
        """Predictive-process weights phi (n, m)."""
        k = matern(dist(xy_km, self.Z), self.ell, self.nu)
        return sla.cho_solve(self.cf, k.T, check_finite=False).T


# ============================================================================ time basis
class TimeBasis:
    """Temporal basis: [u_1..u_{T-1}, v_0..v_{T-1}, cos, sin] at knots t_0 < ... < t_{T-1}."""

    def __init__(self, knots):
        self.knots = np.asarray(knots, float)
        if np.any(np.diff(self.knots) <= 0):
            raise ValueError("knots must increase")
        self.T = len(self.knots)
        self.Kt = 2 * self.T - 1
        self.K = self.Kt + 2
        self.ic = self.Kt
        self.isn = self.Kt + 1

    def iu(self, k: int) -> int | None:
        return None if k == 0 else k - 1

    def iv(self, k: int) -> int:
        return self.T - 1 + k

    def rows(self, t, seasonal: bool = True) -> np.ndarray:
        """(n, K) design rows of u(t) (trend + seasonal)."""
        t = np.atleast_1d(np.asarray(t, float))
        H = np.zeros((t.size, self.K))
        kn = self.knots
        if np.any(t < kn[0] - 1e-9):
            raise ValueError(f"time before the first knot {kn[0]}")
        for n, tt in enumerate(t):
            if tt >= kn[-1]:
                k = self.T - 1
                iu = self.iu(k)
                if iu is not None:
                    H[n, iu] += 1.0
                H[n, self.iv(k)] += tt - kn[-1]
                continue
            k = int(np.searchsorted(kn, tt, side="right") - 1)
            d = kn[k + 1] - kn[k]
            s = (tt - kn[k]) / d
            h00 = 2 * s ** 3 - 3 * s ** 2 + 1
            h10 = s ** 3 - 2 * s ** 2 + s
            h01 = -2 * s ** 3 + 3 * s ** 2
            h11 = s ** 3 - s ** 2
            iu0 = self.iu(k)
            if iu0 is not None:
                H[n, iu0] += h00
            H[n, self.iv(k)] += h10 * d
            H[n, self.iu(k + 1)] += h01
            H[n, self.iv(k + 1)] += h11 * d
        if seasonal:
            H[:, self.ic] = np.cos(2 * np.pi * t)
            H[:, self.isn] = np.sin(2 * np.pi * t)
        return H

    def knot_row(self, k: int) -> np.ndarray:
        """Row selecting the trend displacement u_k at knot k (no seasonal term)."""
        h = np.zeros(self.K)
        iu = self.iu(k)
        if iu is not None:
            h[iu] = 1.0
        return h

    def prior_precision(self, sig_v0: float, q: float, sig_s: float) -> np.ndarray:
        """(K, K) precision J of the scalar temporal prior (IRW + seasonal pair)."""
        T = self.T
        Jf = np.zeros((2 * T, 2 * T))
        Jf[T, T] += 1.0 / sig_v0 ** 2
        for k in range(T - 1):
            d = self.knots[k + 1] - self.knots[k]
            Qd = q * np.array([[d ** 3 / 3.0, d ** 2 / 2.0], [d ** 2 / 2.0, d]])
            Qi = np.linalg.inv(Qd)
            D = np.zeros((2, 2 * T))
            D[0, k + 1], D[0, k], D[0, T + k] = 1.0, -1.0, -d
            D[1, T + k + 1], D[1, T + k] = 1.0, -1.0
            Jf += D.T @ Qi @ D
        J = np.zeros((self.K, self.K))
        J[:self.Kt, :self.Kt] = Jf[1:, 1:]
        J[self.ic, self.ic] = J[self.isn, self.isn] = 1.0 / sig_s ** 2
        return J


# ============================================================================ local projections
def _logdet_chol(c) -> float:
    return 2.0 * float(np.sum(np.log(np.diag(c))))


def projected_precision(R: np.ndarray, E: np.ndarray) -> tuple[np.ndarray, float]:
    """Restricted precision W = R^-1 - R^-1 E (E' R^-1 E)^-1 E' R^-1 for diffuse effects with
    design E, and its log-determinant term log|R| + log|E' R^-1 E|."""
    cR = sla.cho_factor(R, lower=True, check_finite=False)
    Ri = sla.cho_solve(cR, np.eye(len(R)), check_finite=False)
    RiE = Ri @ E
    S = E.T @ RiE
    cS = sla.cho_factor(S, lower=True, check_finite=False)
    W = Ri - RiE @ sla.cho_solve(cS, RiE.T, check_finite=False)
    W = 0.5 * (W + W.T)
    return W, _logdet_chol(cR[0]) + _logdet_chol(cS[0])


def site_local(t: np.ndarray, seg: np.ndarray, sig: float, sig_r: float,
               extra_var: np.ndarray | None = None) -> tuple[np.ndarray, float, int]:
    """Point site with diffuse per-segment offsets and a random velocity ~ N(0, sig_r^2):
    restricted precision, log-det term, effective n."""
    t = np.asarray(t, float)
    n = t.size
    tau = t - t.mean()
    R = sig ** 2 * np.eye(n) + sig_r ** 2 * np.outer(tau, tau)
    if extra_var is not None:
        R = R + np.diag(extra_var)
    segs, inv = np.unique(seg, return_inverse=True)
    E = np.zeros((n, segs.size))
    E[np.arange(n), inv] = 1.0
    W, ld = projected_precision(R, E)
    return W, ld, n - segs.size


def ramp_month(xy_km: np.ndarray, nep: np.ndarray, sig_i: float, sig_a: float,
               ell_a: float) -> tuple[np.ndarray, float, int]:
    """InSAR month group: white + exponential APS covariance (each block averaged over its
    ``nep`` epochs) with a free plane (offset, east, north) projected out."""
    n = len(xy_km)
    C = sig_i ** 2 * np.eye(n) + sig_a ** 2 * np.exp(-dist(xy_km, xy_km) / ell_a)
    s = 1.0 / np.sqrt(np.asarray(nep, float))
    R = C * np.outer(s, s)
    xc = xy_km - xy_km.mean(0)
    X = np.column_stack([np.ones(n), xc[:, 0], xc[:, 1]])
    W, ld = projected_precision(R, X)
    return W, ld, n - 3


# ============================================================================ assembly
@dataclass
class Contribution:
    """Data part of the posterior precision (no prior): P (N, N), b (N,), y'Wy, log-det
    terms and effective n."""
    P: np.ndarray
    b: np.ndarray
    yWy: float = 0.0
    logdetR: float = 0.0
    n_eff: int = 0

    def __iadd__(self, o: Contribution):
        self.P += o.P
        self.b += o.b
        self.yWy += o.yWy
        self.logdetR += o.logdetR
        self.n_eff += o.n_eff
        return self


class Assembler:
    """Accumulates the data part of the posterior precision for one family."""

    def __init__(self, K: int, m: int, n_glob: int):
        self.K, self.m, self.ng = K, m, n_glob
        self.L = K * m
        self.N = self.L + n_glob
        self.P = np.zeros((self.N, self.N))
        self.P4 = self.P[:self.L, :self.L].reshape(K, m, K, m)
        self.Plg = self.P[:self.L, self.L:].reshape(K, m, n_glob)
        self.Pgg = self.P[self.L:, self.L:]
        self.b = np.zeros(self.N)
        self.blat = self.b[:self.L].reshape(K, m)
        self.bg = self.b[self.L:]
        self.yWy = 0.0
        self.logdetR = 0.0
        self.n_eff = 0
        self._A, self._phi, self._r = [], [], []
        for v in (self.P4, self.Plg, self.Pgg):
            if not np.shares_memory(v, self.P):
                raise RuntimeError("block view of P is a copy")

    # ---- point sites (shared phi)
    def add_site(self, phi, Ht, W, y, gidx=None, G=None, logdet=0.0, n_eff=0) -> None:
        WHt = W @ Ht
        self._A.append(Ht.T @ WHt)
        self._phi.append(np.asarray(phi, float))
        self._r.append(WHt.T @ y)
        if gidx is not None and len(gidx):
            gidx = np.asarray(gidx, int)
            WG = W @ G
            B = Ht.T @ WG                                           # (K, ng_s)
            self.Plg[:, :, gidx] += B[:, None, :] * np.asarray(phi)[None, :, None]
            self.Pgg[np.ix_(gidx, gidx)] += G.T @ WG
            self.bg[gidx] += WG.T @ y
        self.yWy += float(y @ W @ y)
        self.logdetR += float(logdet)
        self.n_eff += int(n_eff)

    def _flush(self) -> None:
        if not self._A:
            return
        A = np.stack(self._A)
        Phi = np.stack(self._phi)
        Rr = np.stack(self._r)
        self.blat += Rr.T @ Phi
        S = len(A)
        for k in range(self.K):
            X = (A[:, k, :, None] * Phi[:, None, :]).reshape(S, self.K * self.m)
            self.P4[k] += (Phi.T @ X).reshape(self.m, self.K, self.m)
        self._A, self._phi, self._r = [], [], []

    # ---- InSAR month groups (shared time)
    def add_month(self, Phi_e, h, W, y, gidx, logdet=0.0, n_eff=0) -> None:
        nz = np.flatnonzero(h)
        PW = Phi_e.T @ W                                            # (m, n)
        M = PW @ Phi_e
        v = PW @ y
        gidx = np.asarray(gidx, int)
        for a in nz:
            for c in nz:
                self.P4[a, :, c, :] += h[a] * h[c] * M
            self.blat[a] += h[a] * v
            blk = self.Plg[a]
            blk[:, gidx] += h[a] * PW
        self.Pgg[np.ix_(gidx, gidx)] += W
        self.bg[gidx] += W @ y
        self.yWy += float(y @ W @ y)
        self.logdetR += float(logdet)
        self.n_eff += int(n_eff)

    def result(self) -> Contribution:
        self._flush()
        L = self.L
        self.P[L:, :L] = self.P[:L, L:].T
        return Contribution(self.P, self.b, self.yWy, self.logdetR, self.n_eff)


def empty_contribution(N: int) -> Contribution:
    return Contribution(np.zeros((N, N)), np.zeros(N))


# ============================================================================ solve
@dataclass
class Posterior:
    theta: np.ndarray
    chol: np.ndarray            # lower Cholesky factor of P
    logdetP: float
    m2loglik: float
    parts: dict = field(default_factory=dict)
    cov: np.ndarray | None = None

    def var_of(self, G: np.ndarray) -> np.ndarray:
        """Posterior variances of the linear functionals in the rows of G."""
        if self.cov is not None:
            return np.einsum("ij,ij->i", G @ self.cov, G)
        V = sla.solve_triangular(self.chol, np.asarray(G, float).T, lower=True,
                                 check_finite=False)
        return np.einsum("ij,ij->j", V, V)

    def cov_of(self, G: np.ndarray) -> np.ndarray:
        if self.cov is not None:
            return G @ self.cov @ G.T
        V = sla.solve_triangular(self.chol, np.asarray(G, float).T, lower=True,
                                 check_finite=False)
        return V.T @ V


def add_prior(P: np.ndarray, J: np.ndarray, Kinv: np.ndarray) -> None:
    """P[latent] += J (x) Kinv, block by block (only the nonzero blocks of J)."""
    m = Kinv.shape[0]
    for k, j in zip(*np.nonzero(J), strict=True):
        P[k * m:(k + 1) * m, j * m:(j + 1) * m] += J[k, j] * Kinv


def prior_logdet(J: np.ndarray, lat: Lattice, glob_prec: np.ndarray) -> float:
    sJ, ldJ = np.linalg.slogdet(J)
    if sJ <= 0:
        raise np.linalg.LinAlgError("temporal prior precision not positive definite")
    return lat.m * ldJ - J.shape[0] * lat.logdetK + float(np.sum(np.log(glob_prec)))


def solve(contrib: Contribution, J: np.ndarray, lat: Lattice, glob_prec: np.ndarray, *,
          want_cov: bool = False) -> Posterior:
    """Posterior of the latent field and global parameters given the data contribution."""
    K, m = J.shape[0], lat.m
    P = contrib.P.copy()
    add_prior(P, J, lat.Kinv)
    L = K * m
    idx = np.arange(L, P.shape[0])
    P[idx, idx] += glob_prec
    # lower factor; the upper triangle of ``c`` is left as garbage and never read
    c, _ = sla.cho_factor(P, lower=True, overwrite_a=True, check_finite=False)
    theta = sla.cho_solve((c, True), contrib.b, check_finite=False)
    ldP = _logdet_chol(c)
    ldQ = prior_logdet(J, lat, glob_prec)
    quad = contrib.yWy - float(contrib.b @ theta)
    m2 = quad + ldP - ldQ + contrib.logdetR + contrib.n_eff * LOG2PI
    post = Posterior(theta, c, ldP, m2, parts={"quad": quad, "logdetP": ldP,
                                               "logdetQ": ldQ, "logdetR": contrib.logdetR,
                                               "n_eff": contrib.n_eff})
    if want_cov:
        inv, info = sla.lapack.dpotri(c, lower=1)
        if info != 0:
            raise np.linalg.LinAlgError(f"dpotri info {info}")
        inv = np.tril(inv) + np.tril(inv, -1).T
        post.cov = inv
    return post


# ============================================================================ functionals
def latent_functional(K: int, m: int, h: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Row over the latent block for sum_k h_k phi . w_k."""
    return np.outer(h, phi).ravel()


def functional_rows(tb: TimeBasis, phis: np.ndarray, hs: np.ndarray, n_glob: int,
                    mu_coef: np.ndarray | None = None, mu_index: int = 0) -> np.ndarray:
    """Rows (n, N) for targets sum_k hs[n, k] phis[n] . w_k + mu_coef[n] * mu_v."""
    n = len(phis)
    L = tb.K * phis.shape[1]
    G = np.zeros((n, L + n_glob))
    G[:, :L] = (hs[:, :, None] * phis[:, None, :]).reshape(n, L)
    if mu_coef is not None:
        G[:, L + mu_index] = mu_coef
    return G
