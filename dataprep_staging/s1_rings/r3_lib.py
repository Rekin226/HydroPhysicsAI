"""WP S1a repair round r3 (S1ar): the repaired evidence, membership and pooling rules.

Declaration S1a.rings_bridged_rules.v3. Each change answers a defect of the r3 diagnosis
(``results/dataprep/reports/S1a.repair.md``, Part 1, written before this module):

* **M1, length-aware b_deep error.** ``sigma_bdeep^2 = max(sd_jk, 10 ppm)^2 +
  (sigma_x / L_deep)^2``: the deep section's excess shortening over a gap is a few mm whatever
  its length, so its scale error grows as 1/L_deep (diagnosis 1.2); the jackknife over intervals
  misses that section-wide part. It feeds the b_deep likelihood ratio and the pool weights.
* **M2, pile line out of membership.** The pile-implied scale is no longer an evidence line
  (miscalibrated, not independent of the jump being corrected, upstream gate failed; 1.3). It is
  still computed (check (b), product card) and still shapes the 2022 path.
* **M3, empirical-Bayes prior.** P(A) of site i uses the A share ``pi_i`` estimated by EM over
  the OTHER sites' line likelihood ratios with the Laplace rule
  ``pi <- (sum_j P_j + 1) / (n + 2)`` (start 0.5), instead of the fixed 0.5 (1.3).
* **M4, pools by confident posterior membership.** The pooled scale of group g for site i
  comes from every other site j with a finite b_deep and ``P(g | j; pi_i) >= 0.95``, whatever
  its vote count or forcing status (1.1, 1.4), inverse-variance weighted with M1's sd,
  DerSimonian-Laird tau. No qualifying site: the reference value (as in v1), flagged.
* **M5, common-mode term.** ``sigma_b^2 = se^2 + tau^2 + sigma_cm^2`` with
  ``sigma_cm = sigma_c * sum_j(w_j / L_j) / sum_j(w_j)``: the shared excess of a strong dry
  season does not average out over sites (1.4).

Unchanged from v1 (``s1lib``): ``gap_stats``, the offset line (likelihood ratio 10 in a
cluster), the likelihood-ratio formula and vote rule, the assignment rule (>= 2 agreeing lines,
none against, not forced; with M2 the two lines are offset and b_deep), the forced list, the
offset ``a = J_top_mid - b z_top``, the member gap compaction and its sd, the mixture primary.
"""
from __future__ import annotations

import math

import numpy as np

from s1_rings import s1const as K
from s1_rings import s1lib as L

#: M4 threshold on the posterior membership probability for entering a pool.
POOL_P_MIN = 0.95
#: M3 EM iterations (Laplace rule); convergence tolerance.
EM_MAX_IT = 500
EM_TOL = 1e-10


def bdeep_sd(gs: dict, sigma_x: float) -> float:
    """M1: length-aware sd of b_deep (strain); NaN when b_deep is NaN."""
    if not np.isfinite(gs["b_deep"]) or not np.isfinite(gs["sd_deep"]) or gs["deep_len"] <= 0:
        return float("nan")
    return math.sqrt(max(gs["sd_deep"], K.SD_FLOOR) ** 2 + (sigma_x / gs["deep_len"]) ** 2)


def lines_r3(gs: dict, sigma_x: float, offset_lr: float = K.OFFSET_LR) -> dict:
    """Evidence lines of v3: the offset cluster and the site's own b_deep (M1 sd)."""
    return {"offset": L.offset_line(gs["J_top_mid"], offset_lr),
            "b_deep": L.line_lr(gs["b_deep"], bdeep_sd(gs, sigma_x))}


def lr_product(lines: dict) -> float:
    lr = 1.0
    for v in lines.values():
        lr *= v["lr"]
    return lr


def posterior(lr: float, prior: float) -> float:
    return prior * lr / (prior * lr + (1.0 - prior))


def em_prior(lrs) -> tuple[float, int]:
    """M3: A share by EM with the Laplace rule over the given sites' likelihood ratios."""
    lrs = [float(x) for x in lrs]
    n = len(lrs)
    pi = 0.5
    for it in range(1, EM_MAX_IT + 1):
        new = (sum(posterior(x, pi) for x in lrs) + 1.0) / (n + 2.0)
        if abs(new - pi) < EM_TOL:
            return new, it
        pi = new
    return pi, EM_MAX_IT


def pool_r3(group: str, site: str, sites: list[str], lrs: dict, prior: float, gss: dict,
            sds: dict, sigma_c: float) -> dict:
    """M4/M5: pooled scale of ``group`` for ``site`` (left out) from the other sites whose
    posterior membership (with ``site``'s prior) is >= POOL_P_MIN."""
    members, vals, sd, wl = [], [], [], []
    for j in sites:
        if j == site or not np.isfinite(gss[j]["b_deep"]) or not np.isfinite(sds[j]):
            continue
        pa = posterior(lrs[j], prior)
        pg = pa if group == "A" else 1.0 - pa
        if pg >= POOL_P_MIN:
            members.append(j)
            vals.append(gss[j]["b_deep"])
            sd.append(sds[j])
            wl.append(gss[j]["deep_len"])
    if not members:
        m, t = K.REF[group]
        return {"b": m, "sigma": t, "n": 0, "se": float("nan"), "tau": float("nan"),
                "sigma_cm": float("nan"), "sites": [], "fallback": True}
    pr = L.pool(vals, sd)
    w = 1.0 / np.maximum(np.asarray(sd), K.SD_FLOOR) ** 2
    sig_cm = sigma_c * float(np.sum(w / np.asarray(wl)) / np.sum(w))
    sig = math.sqrt(pr["se"] ** 2 + pr["tau"] ** 2 + sig_cm ** 2)
    return {"b": pr["mean"], "sigma": sig, "n": pr["n"], "se": pr["se"], "tau": pr["tau"],
            "sigma_cm": sig_cm, "sites": members, "fallback": False}


def evaluate_r3(pos: dict, pre_win: dict, post_win: dict, forced: set, sigma_x: float,
                sigma_c: float, gs_cache: dict | None = None, offset_lr: float = K.OFFSET_LR,
                piles: dict | None = None, pile_wins: dict | None = None) -> dict:
    """v3 evidence, membership, pooling and members for a set of sites.

    ``piles`` (site -> (daily Series, depth)) only adds the pile-implied scale to the result
    (``SiteResult.pile``) for check (b); it is never an evidence line (M2)."""
    sites = list(pos)
    gss, sds, lines, lrs = {}, {}, {}, {}
    for s in sites:
        gs = gs_cache[s] if gs_cache and s in gs_cache else L.gap_stats(pos[s], pre_win[s],
                                                                         post_win[s])
        gss[s] = gs
        sds[s] = bdeep_sd(gs, sigma_x)
        lines[s] = lines_r3(gs, sigma_x, offset_lr)
        lrs[s] = lr_product(lines[s])
    res = {}
    for s in sites:
        prior, n_it = em_prior([lrs[j] for j in sites if j != s])
        pa = posterior(lrs[s], prior)
        pe = None
        if piles and s in piles and piles[s] is not None:
            q, depth = piles[s]
            pw = (pile_wins or {}).get(s, (pre_win[s], post_win[s]))
            pe = L.pile_evidence(pos[s], gss[s], q, depth, pw[0], pw[1])
        r = L.SiteResult(site=s, gs=gss[s], lines=lines[s], p_A=pa,
                         assign=L.assign(lines[s], s in forced), pile=pe)
        r.assign["prior_A"] = prior
        r.assign["em_iterations"] = n_it
        r.assign["sd_bdeep"] = sds[s]
        for g in "AB":
            gsc = pool_r3(g, s, sites, lrs, prior, gss, sds, sigma_c)
            m = L.member_gap(gss[s], gsc["b"], gsc["sigma"])
            m["pool"] = gsc
            r.members[g] = m
        r.primary = L.mixture(r.members, r.p_A, r.assign["assigned"])
        res[s] = r
    return res


def global_pool(res: dict, group: str, sigma_x: float) -> dict:
    """Pooled scale of ``group`` over ALL sites (none left out) with posterior membership
    >= POOL_P_MIN under the all-site EM prior (check (b) and the product card)."""
    sites = list(res)
    lrs = {s: lr_product(res[s].lines) for s in sites}
    prior, _ = em_prior([lrs[s] for s in sites])
    mem, vals, sds = [], [], []
    for s in sites:
        gs = res[s].gs
        sd = bdeep_sd(gs, sigma_x)
        pa = posterior(lrs[s], prior)
        pg = pa if group == "A" else 1.0 - pa
        if np.isfinite(gs["b_deep"]) and np.isfinite(sd) and pg >= POOL_P_MIN:
            mem.append(s)
            vals.append(gs["b_deep"])
            sds.append(sd)
    if not mem:
        m, t = K.REF[group]
        return {"mean": m, "se": float("nan"), "tau": float("nan"), "n": 0, "sites": [],
                "prior_A": prior, "fallback": True}
    pr = L.pool(vals, sds)
    return {**pr, "sites": mem, "prior_A": prior, "fallback": False}


# ----------------------------------------------------------------------------- corrected (a)
def ar1_factor(resid: np.ndarray) -> float:
    """sqrt((1 + phi) / (1 - phi)) with phi the lag-1 autocorrelation (clipped to [0, 0.95])."""
    r = np.asarray(resid, float)
    r = r[np.isfinite(r)]
    if len(r) < 4:
        return 1.0
    r = r - r.mean()
    den = float(np.sum(r * r))
    phi = float(np.sum(r[1:] * r[:-1]) / den) if den > 0 else 0.0
    phi = min(max(phi, 0.0), 0.95)
    return math.sqrt((1.0 + phi) / (1.0 - phi))


def trend_endpoints(p, gs: dict, pre_win, post_win) -> dict:
    """Window-fit trend of the total separation at the last pre (t_lp) and the first post
    (t_fp) survey, with AR(1)-inflated se (corrected check (a))."""
    S = p[gs["bot"]] - p[gs["top"]]
    t_m = float(L.years([gs["t_mid"]])[0])
    out = {}
    for name, w, t in (("pre", pre_win, gs["t_lp"]), ("post", post_win, gs["t_fp"])):
        x = S.loc[w[0]:w[1]].dropna()
        tt = L.years(x.index)
        f = L.window_fit(tt, x.to_numpy(), t_m)
        X = np.stack([np.ones_like(tt), tt - t_m, np.sin(2 * np.pi * tt),
                      np.cos(2 * np.pi * tt)], 1)
        resid = x.to_numpy() - X @ f.beta
        ty = float(L.years([t])[0])
        k = ar1_factor(resid)
        out[name] = {"trend": f.trend(ty), "se": f.se_trend(ty), "ar1_factor": k,
                     "se_inflated": k * f.se_trend(ty), "n": f.n}
    return out


def corrected_a_site(r, te: dict) -> dict:
    """Trend-based bridged gap compaction of each member and the primary (corrected (a))."""
    L_tot = r.gs["L"]
    mem = {}
    for g in "AB":
        m = r.members[g]
        C = te["pre"]["trend"] - te["post"]["trend"] / (1.0 + m["b"])
        sd = math.sqrt((m["sigma"] * L_tot) ** 2 + te["pre"]["se_inflated"] ** 2
                       + te["post"]["se_inflated"] ** 2)
        mem[g] = {"C": C, "sigma_C": sd, "b": m["b"]}
    return L.mixture(mem, r.p_A, r.assign["assigned"])
