"""P2a step 3: validation of the Chow-Lin split on the truly monthly irrigation meters
(pre-registered: P2a.meter_validation.v1, P2a.sd_coverage.v1), placebo profiles, the
December phase-B edge extrapolation at pseudo-origins, and the sd model used by the product.

Writes ``_work_p2abc/validation_<variant>.json`` and ``sd_model_<variant>.json``.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib  # noqa: E402
import s2_chowlin as s2  # noqa: E402

SIZES = (1, 3, 10, 30, 100, 300)
STRATA = {"1": (1,), "3": (3,), "10": (10,), "30": (30,), "100+300": (100, 300)}
Z90 = 1.6448536269514722
N_PLACEBO = 20
warnings.filterwarnings("ignore", category=UserWarning)   # KMeans duplicate points


# ----------------------------------------------------------------------------- stats
def icc_oneway(hits: np.ndarray, groups: np.ndarray) -> float:
    """One-way ANOVA intra-class correlation of a 0/1 outcome within clusters."""
    df = pd.DataFrame({"h": hits.astype(float), "g": groups})
    g = df.groupby("g")["h"]
    n_i = g.size().to_numpy().astype(float)
    k, N = len(n_i), n_i.sum()
    if k < 2 or N <= k:
        return 0.0
    mean_i = g.mean().to_numpy()
    grand = df["h"].mean()
    msb = np.sum(n_i * (mean_i - grand) ** 2) / (k - 1)
    msw = np.sum((df["h"] - df["g"].map(g.mean())) ** 2) / (N - k)
    n0 = (N - np.sum(n_i**2) / N) / (k - 1)
    den = msb + (n0 - 1) * msw
    return float(max(0.0, (msb - msw) / den)) if den > 0 else 0.0


def wilson(p: float, n_eff: float, z: float = 1.959963984540054) -> tuple[float, float]:
    if n_eff <= 0:
        return (0.0, 1.0)
    den = 1 + z**2 / n_eff
    c = (p + z**2 / (2 * n_eff)) / den
    h = z * np.sqrt(p * (1 - p) / n_eff + z**2 / (4 * n_eff**2)) / den
    return float(c - h), float(c + h)


def coverage_gate(hits: np.ndarray, groups: np.ndarray, target: float = 0.90) -> dict:
    n = len(hits)
    p = float(np.mean(hits)) if n else float("nan")
    m = n / max(len(np.unique(groups)), 1)
    icc = icc_oneway(hits, groups)
    n_eff = n / (1 + (m - 1) * icc)
    lo, hi = wilson(p, n_eff)
    ok = lo <= target <= hi
    point_needed = n_eff >= 30
    if point_needed:
        ok = ok and (0.85 <= p <= 0.95)
    return {"n": int(n), "n_groups": int(len(np.unique(groups))), "coverage": p, "icc": icc,
            "n_eff": float(n_eff), "wilson95": [lo, hi], "point_window_applies": point_needed,
            "passed": bool(ok)}


# ----------------------------------------------------------------------------- set-up
def setup(name: str, mt: pd.DataFrame, M: np.ndarray):
    v = lib.VARIANTS[name]
    tab = pd.read_parquet(lib.WORK / f"{name}_phase.parquet")
    phase = tab["phase"].to_numpy()
    end = lib.tidx(v["end"])
    e_last = lib.tidx("2022-12") if name == "preorigin" else lib.tidx("2025-06")
    W0 = lib.tidx("2012-01")
    keep = mt["kept"].to_numpy() & np.isfinite(mt["x"].to_numpy())
    irr = mt["cls"].to_numpy() == "irrigation"
    XW = M[:, W0:e_last + 1]
    val = keep & irr & (phase == lib.MONTHLY) & np.isfinite(XW).all(1) & (np.nansum(XW, 1) > 0)
    vi = np.where(val)[0]
    bim = keep & irr & np.isin(phase, [lib.PHASE_A, lib.PHASE_B])
    tree = cKDTree(mt[["x", "y"]].to_numpy()[bim])
    nph = phase[bim][tree.query(mt[["x", "y"]].to_numpy()[vi], k=1)[1]]
    ph = np.where(nph == lib.PHASE_A, "A", "B")
    prof = np.load(lib.WORK / f"{name}_profile.npz")["prof"]
    ev = np.arange(W0 + 1, e_last)          # 2012-02 .. last-1 (pilot ev slice)
    return {"vi": vi, "ph": ph, "prof": prof, "end": end, "ev": ev,
            "xy": mt[["x", "y"]].to_numpy()[vi], "truth": np.nan_to_num(M[vi].astype(float))}


def estimates(S: dict, name: str, prof: np.ndarray | None = None) -> dict:
    """even, prorata and Chow-Lin monthly estimates (N, T_ALL) of the validation meters."""
    prof = S["prof"] if prof is None else prof
    W = s2.windows(name)
    truth, ph = S["truth"], S["ph"]
    est = {m: np.zeros_like(truth) for m in ("even", "prorata", "chowlin")}
    for p_ in ("A", "B"):
        k = np.where(ph == p_)[0]
        x = truth[k]
        rec = np.nan_to_num(lib.even_split(x, p_))
        est["even"][k] = rec
        est["prorata"][k] = np.nan_to_num(lib.prorata(x, p_, prof))
        lo, hi = W[p_]
        cl = rec.copy()
        cl[:, lo:hi] = lib.ChowLin(p_, prof, lo, hi).run(rec)
        est["chowlin"][k] = cl
    return est


def labels(xy: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    k_ = max(1, len(xy) // n)
    if k_ == 1:
        return np.zeros(len(xy), int)
    return KMeans(n_clusters=k_, n_init=3, random_state=seed).fit_predict(xy)


def group_skill(est: dict, truth: np.ndarray, lab: np.ndarray, ev: np.ndarray) -> dict:
    sse = {m: 0.0 for m in est}
    for gid in np.unique(lab):
        gi = np.where(lab == gid)[0]
        tr = truth[gi].sum(0)[ev]
        for m in est:
            sse[m] += float(np.sum((est[m][gi].sum(0)[ev] - tr) ** 2))
    return {m: 1 - sse[m] / sse["even"] for m in est if m != "even"}


def group_errors(est: dict, truth: np.ndarray, lab: np.ndarray, ev: np.ndarray, tag: str):
    """Long table of group-month errors: group, month index, even value, Chow-Lin error."""
    rows = []
    for gid in np.unique(lab):
        gi = np.where(lab == gid)[0]
        even = est["even"][gi].sum(0)[ev]
        err = est["chowlin"][gi].sum(0)[ev] - truth[gi].sum(0)[ev]
        rows.append(pd.DataFrame({"group": f"{tag}{gid}", "t": ev, "even": even, "err": err,
                                  "n_meters": len(gi)}))
    return pd.concat(rows, ignore_index=True)


# ----------------------------------------------------------------------------- sd model
def fit_q90(df: pd.DataFrame) -> dict:
    """q90 of |err| / even per calendar month (zero pairs excluded)."""
    d = df[df["even"] > 0]
    z = (d["err"].abs() / d["even"]).to_numpy()
    moy = lib.DATES[d["t"].to_numpy()].month.to_numpy()
    return {int(m): float(np.quantile(z[moy == m], 0.90)) for m in range(1, 13)}


def sd_model_and_coverage(S: dict, est: dict) -> tuple[dict, dict]:
    xs = S["xy"][:, 0]
    west = xs <= np.median(xs)
    halves = {"W": np.where(west)[0], "E": np.where(~west)[0]}
    errs = {}
    for h, idx in halves.items():
        sub = {m: est[m][idx] for m in est}
        for n in SIZES:
            lab = labels(S["xy"][idx], n) if n > 1 else np.arange(len(idx))
            errs[(h, n)] = group_errors(sub, S["truth"][idx], lab, S["ev"], f"{h}{n}_")
    q = {h: {n: fit_q90(errs[(h, n)]) for n in SIZES} for h in halves}
    cov = {}
    for sname, sizes in STRATA.items():
        hits, grp = [], []
        for fit_h, test_h in (("W", "E"), ("E", "W")):
            for n in sizes:
                d = errs[(test_h, n)]
                d = d[d["even"] > 0]
                moy = lib.DATES[d["t"].to_numpy()].month.to_numpy()
                qq = np.array([q[fit_h][n][int(m)] for m in moy])
                hits.append((d["err"].abs().to_numpy() <= qq * d["even"].to_numpy()))
                grp.append(("fit" + fit_h + "_") + d["group"].to_numpy().astype(object))
        cov[sname] = coverage_gate(np.concatenate(hits), np.concatenate(grp))
    # product sd model: q90 fitted on all validation groups (both halves pooled)
    full_q = {}
    for n in SIZES:
        full_q[n] = fit_q90(pd.concat([errs[("W", n)], errs[("E", n)]], ignore_index=True))
    return {"q90_by_n_moy": {str(n): full_q[n] for n in SIZES}, "sizes": list(SIZES),
            "z90": Z90, "rule": "sd = q90(n_eff, moy) / 1.645 x even-split value of the "
                                "Chow-Lin component; q90 interpolated log-linearly in n_eff, "
                                "clamped to [1, 300]"}, cov


# ----------------------------------------------------------------------------- edge
def edge_check(name: str, mt: pd.DataFrame, M: np.ndarray, S: dict) -> dict:
    """Phase-B December extrapolation at pseudo-origins Dec 2014..Dec 2021 (pre-origin data
    only): the production recipe with the window and profile cut at each pseudo-origin."""
    tab = pd.read_parquet(lib.WORK / f"{name}_phase.parquet")
    phase = tab["phase"].to_numpy()
    act = tab["active_meter"].to_numpy()
    kB = np.where(S["ph"] == "B")[0]
    truth = S["truth"][kB]
    rec_full = np.nan_to_num(lib.even_split(truth, "B"))
    lo = lib.tidx(lib.VARIANTS[name]["cl0"])
    rows = []
    for Y in range(2014, 2022):
        e = lib.tidx(f"{Y}-12")
        keep, irr, bill = mt["kept"].to_numpy(), mt["cls"].to_numpy() == "irrigation", \
            mt["bill"].to_numpy()
        Z = np.nan_to_num(M.astype(float))
        selA = keep & act & irr & (bill == "M2") & (phase == lib.PHASE_A)
        selB = keep & act & irr & (bill == "M1") & (phase == lib.PHASE_B)
        a, b = Z[selA].sum(0), Z[selB].sum(0)
        sb = b.copy()
        sb[e] = sb[e - 1]
        u = lib.stagger_deconvolve(a, b, t_end=e + 1, share_b=sb)
        prof = np.concatenate([u, np.full(lib.T_ALL - len(u), u[-1])])
        rec = rec_full.copy()
        rec[:, e:] = 0.0                       # nothing after the pseudo-origin is known
        out = lib.ChowLin("B", prof, lo, e + 1).run(rec)
        xhat = out[:, -1]
        own = np.mean([rec_full[:, lib.tidx(f"{Y - j}-12")] for j in (1, 2, 3)], axis=0)
        rows.append(pd.DataFrame({"Y": Y, "k": np.arange(len(kB)), "xhat": xhat, "own": own,
                                  "truth": truth[:, e], "prev_even": rec_full[:, e - 1]}))
    d = pd.concat(rows, ignore_index=True)
    res = {"n_meters": int(len(kB)), "pseudo_origins": list(range(2014, 2022))}
    for m in ("xhat", "own"):
        e_ = d[m] - d["truth"]
        res[f"meter_rmse_{m}"] = float(np.sqrt(np.mean(e_**2)))
        res[f"total_abs_err_pct_{m}"] = float(
            d.groupby("Y").apply(lambda g: abs(g[m].sum() - g["truth"].sum()) / g["truth"].sum()
                                 ).mean() * 100)
    # q90 of |err| / xhat by group size (groups: k-means of the B-assigned meters)
    xyB = S["xy"][kB]
    for m in ("xhat", "own"):
        q = {}
        for n in SIZES:
            lab = labels(xyB, n) if n > 1 else np.arange(len(kB))
            g = d.assign(g=lab[d["k"].to_numpy()]).groupby(["Y", "g"])[[m, "truth"]].sum()
            g = g[g[m] > 0]
            q[str(n)] = float(np.quantile((g[m] - g["truth"]).abs() / g[m], 0.90))
        res[f"q90_rel_by_n_{m}"] = q
    # Deviation (recorded in the report): the product fills the open December pair with the
    # meter's own 3-year same-month mean, chosen after this pre-origin pseudo-origin backtest
    # showed it beats the Chow-Lin extrapolation; its error here is therefore optimistic.
    res["chosen"] = "own"
    res["q90_rel_by_n"] = res["q90_rel_by_n_own"]
    return res


# ----------------------------------------------------------------------------- main
def main() -> None:
    t0 = time.time()
    mt, M, _ = lib.load_work()
    for name in ("preorigin", "full"):
        S = setup(name, mt, M)
        est = estimates(S, name)
        res = {"n_validation_meters": int(len(S["vi"])),
               "pseudo_phase_counts": {p: int((S["ph"] == p).sum()) for p in ("A", "B")},
               "eval_months": [lib.ym(S["ev"][0]), lib.ym(S["ev"][-1])], "group_skill": {}}
        for n in SIZES + (len(S["vi"]),):
            lab = labels(S["xy"], n)
            key = str(n) if n in SIZES else "fan"
            res["group_skill"][key] = group_skill(est, S["truth"], lab, S["ev"])
        res["criterion"] = "chowlin skill >= 0.15 at every n in 1..300"
        res["passed"] = bool(all(res["group_skill"][str(n)]["chowlin"] >= 0.15 for n in SIZES))
        res["min_skill_1_300"] = float(min(res["group_skill"][str(n)]["chowlin"] for n in SIZES))
        print(name, "skill", {k: round(v["chowlin"], 4) for k, v in res["group_skill"].items()},
              "passed", res["passed"], flush=True)
        sdm, cov = sd_model_and_coverage(S, est)
        res["sd_coverage"] = cov
        res["sd_coverage_passed"] = bool(all(c["passed"] for c in cov.values()))
        print(name, "coverage", {k: (round(c["coverage"], 3), [round(x, 3) for x in c["wilson95"]],
                                     c["passed"]) for k, c in cov.items()}, flush=True)
        if name == "preorigin":
            # placebo profiles (year-shuffled) on the same meters
            end = S["end"]
            plac = {str(n): [] for n in SIZES}
            plac["fan"] = []
            labs = {n: labels(S["xy"], n) for n in SIZES + (len(S["vi"]),)}
            for seed in range(1, N_PLACEBO + 1):
                pp = lib.year_shuffle(S["prof"][:end + 1], seed)
                pp = np.concatenate([pp, np.full(lib.T_ALL - len(pp), pp[-1])])
                e2 = estimates(S, name, prof=pp)
                for n, lab in labs.items():
                    key = str(n) if n in SIZES else "fan"
                    plac[key].append(group_skill({"even": e2["even"], "chowlin": e2["chowlin"]},
                                                 S["truth"], lab, S["ev"])["chowlin"])
            res["placebo_skill"] = {k: {"mean": float(np.mean(v)), "min": float(np.min(v)),
                                        "max": float(np.max(v))} for k, v in plac.items()}
            res["real_beats_all_placebos"] = {
                k: bool(res["group_skill"][k]["chowlin"] > res["placebo_skill"][k]["max"])
                for k in plac}
            res["edge_december_B"] = edge_check(name, mt, M, S)
            sdm["edge_q90_rel_by_n"] = res["edge_december_B"]["q90_rel_by_n"]
            print("placebo", {k: round(v["mean"], 3) for k, v in res["placebo_skill"].items()},
                  flush=True)
            print("edge", {k: v for k, v in res["edge_december_B"].items()
                           if k != "pseudo_origins"}, flush=True)
        lib.save_json(res, lib.WORK / f"validation_{name}.json")
        lib.save_json(sdm, lib.WORK / f"sd_model_{name}.json")
    print("seconds", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
