"""State-space disaggregation with posterior draws (plan W1 step 5).

Per district: log x_t = log i_t + u_t on the months with a positive indicator i_t (other
months are structural zeros), u ~ stationary AR(1) in calendar months (variance s^2, lag-k
correlation rho^k, mean 0). The annual district totals A_y (printed to 0.01 x 10^8 m3, a
relative error near 0.1 %) are imposed exactly by normalising within each year,
x_t = A_y i_t e^{u_t} / sum_{s in y} i_s e^{u_s}, so every draw is benchmarked like the
Denton series. A near-exact Gaussian annual constraint on the unnormalised model was tried
first and rejected: its curved constraint manifold made both independence Metropolis-Hastings
(acceptance 0.03-2 %) and elliptical slice sampling (lag-1 autocorrelation 0.99) useless.
Observed aggregates with their own Gaussian errors (config.SS_*): the Jan-May totals
2015-2021 (unless held out) and the crop-1 share of the MOA office totals 2012-2015 written
as the linear constraint sum_{Feb-Jun} x - share * sum_year x = 0.

Hyperparameters (rho, s) on a fixed grid, weighted by their Laplace marginal likelihood
(uniform prior over the grid). Given (rho, s), the posterior of u is sampled exactly by elliptical slice
sampling against the Laplace approximation (mode, Gauss-Newton precision) as the reference
Gaussian (an independence Metropolis-Hastings sampler from that approximation was tried first
and rejected: acceptance 0.03-2 % in 100+ dimensions). Draws allocate across the grid in
proportion to the weights.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from w1_canal.config import (
    BENCH_YEARS,
    CALENDAR,
    SS_ESS_BURN,
    SS_ESS_THIN,
    SS_N_DRAWS,
    SS_RHO_GRID,
    SS_SEED,
    SS_SIGMA_CROPSHARE,
    SS_SIGMA_GRID,
    SS_SIGMA_JANMAY_MIN,
    SS_SIGMA_JANMAY_REL,
)

CROP1_MONTHS = tuple(m for m, (c, _) in CALENDAR.items() if c == 1)


@dataclass
class Obs:
    A: np.ndarray        # (K, T) aggregation weights on x
    y: np.ndarray        # (K,)
    s: np.ndarray        # (K,) 1-sigma errors
    labels: list
    annual: np.ndarray   # (n_years,) exact annual totals


def build_obs(years, annual: np.ndarray, janmay: dict | None, crop_share: dict | None,
              holdout_year: int | None = None) -> Obs:
    years = list(years)
    T = 12 * len(years)
    rows, y, s, lab = [], [], [], []
    for yr, v in (janmay or {}).items():
        if yr == holdout_year or yr not in years:
            continue
        i = years.index(yr)
        a = np.zeros(T)
        a[12 * i:12 * i + 5] = 1
        rows.append(a)
        y.append(v)
        s.append(max(SS_SIGMA_JANMAY_MIN, SS_SIGMA_JANMAY_REL * v))
        lab.append(f"janmay_{yr}")
    for yr, sh in (crop_share or {}).items():
        if yr not in years:
            continue
        i = years.index(yr)
        a = np.zeros(T)
        for m in CROP1_MONTHS:
            a[12 * i + m - 1] += 1
        a[12 * i:12 * i + 12] -= sh
        rows.append(a)
        y.append(0.0)
        s.append(SS_SIGMA_CROPSHARE * annual[i])
        lab.append(f"crop1share_{yr}")
    A = np.array(rows) if rows else np.zeros((0, T))
    return Obs(A, np.array(y, float), np.array(s, float), lab, np.asarray(annual, float))


def ar1_cov(t: np.ndarray, rho: float, sig: float) -> np.ndarray:
    return sig ** 2 * rho ** np.abs(np.subtract.outer(t, t))


class Posterior:
    """log posterior of u (positive months only) for fixed (rho, sig)."""

    def __init__(self, ind: np.ndarray, obs: Obs, rho: float, sig: float):
        self.pos = np.where(ind > 1e-12)[0]
        self.ip = ind[self.pos]
        self.yr = self.pos // 12
        self.ny = len(ind) // 12
        self.annual = obs.annual
        self.Ap = obs.A[:, self.pos]
        self.y, self.w = obs.y, 1 / obs.s ** 2
        S = ar1_cov(self.pos.astype(float), rho, sig)
        self.L = np.linalg.cholesky(S + 1e-12 * np.eye(len(S)))
        self.Si = np.linalg.inv(S)
        self.logdetS = 2 * np.log(np.diag(self.L)).sum()
        self.obs = obs

    def x(self, u):
        e = self.ip * np.exp(u - u.max())
        tot = np.zeros(self.ny)
        np.add.at(tot, self.yr, e)
        return self.annual[self.yr] * e / tot[self.yr]

    def jac(self, u):
        """d(A_obs x)/du: x_t (delta_ts - x_s / A_y) within each year."""
        x = self.x(u)
        Ax = self.Ap * x[None, :]
        # subtract, per year, (A_obs x restricted to the year) * x_s / A_y
        J = Ax.copy()
        for k in range(self.ny):
            sel = self.yr == k
            if not sel.any():
                continue
            J[:, sel] -= np.outer(Ax[:, sel].sum(1), x[sel] / self.annual[k])
        return J

    def negLogPost(self, u):
        r = self.y - self.Ap @ self.x(u)
        return 0.5 * u @ self.Si @ u + 0.5 * (self.w * r * r).sum()

    def grad(self, u):
        r = self.y - self.Ap @ self.x(u)
        return self.Si @ u - self.jac(u).T @ (self.w * r)

    def gn_hessian(self, u):
        J = self.jac(u)
        return self.Si + J.T @ (self.w[:, None] * J)

    def mode(self):
        res = minimize(self.negLogPost, np.zeros(len(self.pos)), jac=self.grad,
                       method="L-BFGS-B", options={"maxiter": 5000, "gtol": 1e-9})
        u = res.x
        for _ in range(20):          # polish with damped Newton (GN Hessian)
            g = self.grad(u)
            H = self.gn_hessian(u)
            step = np.linalg.solve(H, g)
            f0 = self.negLogPost(u)
            t = 1.0
            while self.negLogPost(u - t * step) > f0 and t > 1e-6:
                t /= 2
            u = u - t * step
            if np.abs(t * step).max() < 1e-10:
                break
        return u

    def log_marginal(self, u, H):
        r = self.y - self.Ap @ self.x(u)
        sign, logdetH = np.linalg.slogdet(H)
        return float(-0.5 * u @ self.Si @ u - 0.5 * (self.w * r * r).sum()
                     - 0.5 * self.logdetS - 0.5 * logdetH
                     - 0.5 * np.log(2 * np.pi / self.w).sum())


def ess_draws(post: Posterior, u0, H, n_draws: int, rng, thin: int = SS_ESS_THIN,
              burn: int = SS_ESS_BURN):
    """Elliptical slice sampling against the Laplace approximation N(u0, H^-1) as the
    reference Gaussian: target = N(u0, H^-1) * exp(loglik_ratio), with loglik_ratio the exact
    log posterior minus the Gaussian log density. Exact for the posterior, rejection-free,
    and close to independent draws when the approximation is good."""
    n = len(u0)
    Lc = np.linalg.cholesky(np.linalg.inv(H))

    def llr(nu):
        return -post.negLogPost(u0 + nu) + 0.5 * nu @ H @ nu

    nu = np.zeros(n)
    ll = llr(nu)
    keep, n_eval = [], 0
    n_iter = burn + thin * n_draws
    for k in range(n_iter):
        nu_p = Lc @ rng.standard_normal(n)
        logy = ll + np.log(rng.random())
        th = rng.uniform(0, 2 * np.pi)
        lo, hi = th - 2 * np.pi, th
        while True:
            cand = nu * np.cos(th) + nu_p * np.sin(th)
            lc = llr(cand)
            n_eval += 1
            if lc > logy:
                nu, ll = cand, lc
                break
            if th < 0:
                lo = th
            else:
                hi = th
            th = rng.uniform(lo, hi)
        if k >= burn and (k - burn) % thin == 0 and len(keep) < n_draws:
            keep.append(u0 + nu)
    return np.array(keep), n_eval / max(k + 1, 1)


def run(ind: np.ndarray, obs: Obs, n_draws: int = SS_N_DRAWS, seed: int = SS_SEED) -> dict:
    """Hyperparameter weights, posterior draws of x (n_draws, T) and diagnostics."""
    T = len(ind)
    fits = []
    for rho in SS_RHO_GRID:
        for sig in SS_SIGMA_GRID:
            p = Posterior(ind, obs, rho, sig)
            u = p.mode()
            H = p.gn_hessian(u)
            fits.append((rho, sig, p, u, H, p.log_marginal(u, H)))
    lm = np.array([f[5] for f in fits])
    w = np.exp(lm - lm.max())
    w /= w.sum()
    rng = np.random.default_rng(seed)
    alloc = rng.multinomial(n_draws, w)
    draws, accs = [], []
    for (rho, sig, p, u, H, _), k in zip(fits, alloc, strict=True):
        if k == 0:
            continue
        U, a = ess_draws(p, u, H, int(k), rng)
        X = np.zeros((len(U), T))
        X[:, p.pos] = np.array([p.x(uu) for uu in U])
        draws.append(X)
        accs.append({"rho": rho, "sigma": sig, "n": int(k), "evals_per_draw": round(float(a), 2)})
    X = np.vstack(draws)
    X = X[rng.permutation(len(X))]
    best = fits[int(np.argmax(lm))]
    xmode = np.zeros(T)
    xmode[best[2].pos] = best[2].x(best[3])
    return {"draws": X, "weights": [{"rho": f[0], "sigma": f[1], "log_ml": round(float(f[5]), 3),
                                     "w": round(float(wi), 5)} for f, wi in zip(fits, w, strict=True)],
            "ess": accs, "mode_best": xmode, "best": {"rho": best[0], "sigma": best[1]},
            "obs_labels": obs.labels}


def interval(samples: np.ndarray, level: float = 0.90):
    lo, hi = np.percentile(samples, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return float(lo), float(hi)


def janmay_of(X: np.ndarray, years, year: int) -> np.ndarray:
    i = list(years).index(year)
    return X[:, 12 * i:12 * i + 5].sum(1)


def bench_years_in(years):
    return [y for y in BENCH_YEARS if y in list(years)]
