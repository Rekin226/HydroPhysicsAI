"""Registered tests P2r2.a1 (Chow-Lin skill on truly monthly meters) and P2r2.a2 (coverage of
the split's sd), per county, on the C0 v3 products (pre-registration P2abc.r2.c0_based.v1).

Artificial-gap test of plan 4.4 ("Billing (P2a)"): every kept irrigation meter whose detected
phase is monthly, flag 0 and finite in every month 2012-01..e_last (pre-origin e_last 2022-12,
full 2026-06), with positive total and coordinates, is given the phase (A/B) of its nearest kept
irrigation bimonthly meter of the same county (first-pump coordinates; Chiayi: of Yunlin or
Chiayi, the Yunlin profile county), summed into that phase's pairs, split evenly ("recorded")
and re-split by the production operator of the variant (the county's profile, window
2011-01..variant end, the production pair-eligibility rule on the meter's own flags and method
codes). Groups: k-means on coordinates (n_init 3, random_state 0), N // n clusters for n in
{1, 3, 10, 30, 100, 300} (n = 1: every meter alone) plus fan-wide; evaluation months
2012-02..e_last-1. Skill = 1 - SSE_chowlin / SSE_even pooled over groups.

sd model (P2r2.a2): z = |chowlin - truth| / even per group-month (zero even values excluded);
q90 per (n, calendar month) fitted on one half of the meters (split at the county's median
easting; k-means groups formed within each half), scored on the other half, both directions;
hits pooled per stratum {1}, {3}, {10}, {30}, {100, 300}; clusters = groups; harness.coverage3
rule gate-3. A stratum with fewer than 20 clusters is pooled with the next smaller stratum
before testing. The product's sd table is q90 fitted on all groups of both halves (pooled).

Usage: python -I t1_meter_validation.py <out_dir> [--variant preorigin|full] [--placebo]
"""
from __future__ import annotations

import hashlib
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2lib as L  # noqa: E402
import scoring  # noqa: E402
from harness import coverage3  # noqa: E402

SIZES = (1, 3, 10, 30, 100, 300)
STRATA = [("1", (1,)), ("3", (3,)), ("10", (10,)), ("30", (30,)), ("100+300", (100, 300))]
N_PLACEBO = 20
MIN_K = 20
warnings.filterwarnings("ignore", category=UserWarning)


def code_sha() -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), L.CODE_DIR / "p2lib.py")}


# ----------------------------------------------------------------------------- set-up
def setup(variant: str, county: str, cds: dict, profs: dict) -> dict:
    cd = cds[county]
    e_last = L.tidx("2022-12") if variant == "preorigin" else L.tidx("2026-06")
    W0 = L.tidx("2012-01")
    irr = cd.mclass == "irrigation"
    ph = cd.phase
    Xw, Fw = cd.M[:, W0:e_last + 1], cd.F[:, W0:e_last + 1]
    xy = np.column_stack([cd.m_x, cd.m_y])
    fin = np.isfinite(xy).all(1)
    val = (irr & (ph == "monthly") & np.isfinite(Xw).all(1) & (Fw == 0).all(1)
           & (np.nansum(Xw, 1) > 0) & fin)
    vi = np.where(val)[0]
    # donors of the pseudo phase: kept irrigation bimonthly meters with coordinates
    don_xy, don_ph = [], []
    for c in ([county] if county != "chiayi" else ["yunlin", "chiayi"]):
        d = cds[c]
        sel = (d.mclass == "irrigation") & np.isin(d.phase, ["A", "B"]) & \
            np.isfinite(d.m_x) & np.isfinite(d.m_y)
        don_xy.append(np.column_stack([d.m_x, d.m_y])[sel])
        don_ph.append(d.phase[sel])
    don_xy, don_ph = np.concatenate(don_xy), np.concatenate(don_ph)
    nph = don_ph[cKDTree(don_xy).query(xy[vi], k=1)[1]]
    return {"vi": vi, "ph": nph.astype(str), "xy": xy[vi], "truth": np.nan_to_num(cd.M[vi]),
            "F": cd.F[vi], "Mm": cd.Mm[vi], "ev": np.arange(W0 + 1, e_last),
            "prof": profs[L.PROFILE_OF[county]], "variant": variant, "T": cd.T}


def estimates(S: dict, prof: np.ndarray | None = None) -> dict:
    """even, pro-rata and Chow-Lin monthly estimates (N, T) of the validation meters."""
    prof = S["prof"] if prof is None else prof
    v = L.VARIANTS[S["variant"]]
    truth, ph = S["truth"], S["ph"]
    est = {m: np.zeros_like(truth) for m in ("even", "prorata", "chowlin")}
    for p_ in ("A", "B"):
        k = np.where(ph == p_)[0]
        if not len(k):
            continue
        x = truth[k]
        rec = np.nan_to_num(L.even_split(x, p_))
        est["even"][k] = rec
        st = L.pair_starts(p_, 0, S["T"], S["T"])
        s = x[:, st] + x[:, st + 1]
        i1, i2 = np.maximum(prof[st], 1e-6), np.maximum(prof[st + 1], 1e-6)
        w = i1 / (i1 + i2)
        pr = np.zeros_like(x)
        pr[:, st] = s * w
        pr[:, st + 1] = s * (1 - w)
        est["prorata"][k] = pr
        vals, _, _, _ = L.apply_chowlin(rec, S["F"][k], S["Mm"][k], np.arange(len(k)), p_,
                                        prof, v["cl0"], v["win_hi"][p_])
        est["chowlin"][k] = vals
    return est


def labels(xy: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    k_ = max(1, len(xy) // n)
    if n == 1:
        return np.arange(len(xy))
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
    rows = []
    for gid in np.unique(lab):
        gi = np.where(lab == gid)[0]
        even = est["even"][gi].sum(0)[ev]
        err = est["chowlin"][gi].sum(0)[ev] - truth[gi].sum(0)[ev]
        rows.append(pd.DataFrame({"group": f"{tag}{gid}", "t": ev, "even": even, "err": err,
                                  "n_meters": len(gi)}))
    return pd.concat(rows, ignore_index=True)


def fit_q90(df: pd.DataFrame) -> dict:
    d = df[df["even"] > 0]
    z = (d["err"].abs() / d["even"]).to_numpy()
    moy = L.DATES[d["t"].to_numpy()].month.to_numpy()
    return {int(m): float(np.quantile(z[moy == m], 0.90)) for m in range(1, 13)}


def pooled_strata(k_by_stratum: dict) -> list[tuple[str, tuple]]:
    """Registered pooling rule: a stratum with < MIN_K clusters merges into the next smaller
    stratum (repeatedly)."""
    out = [(name, sizes, k_by_stratum[name]) for name, sizes in STRATA]
    i = len(out) - 1
    while i > 0:
        name, sizes, k = out[i]
        if k < MIN_K:
            pn, ps, pk = out[i - 1]
            out[i - 1] = (f"{pn}+{name}", ps + sizes, pk + k)
            out.pop(i)
        i -= 1
    return [(n, s) for n, s, _ in out]


def sd_model_and_coverage(S: dict, est: dict) -> tuple[dict, dict]:
    xs = S["xy"][:, 0]
    west = xs <= np.median(xs)
    halves = {"W": np.where(west)[0], "E": np.where(~west)[0]}
    errs = {}
    for h, idx in halves.items():
        sub = {m: est[m][idx] for m in est}
        for n in SIZES:
            lab = labels(S["xy"][idx], n)
            errs[(h, n)] = group_errors(sub, S["truth"][idx], lab, S["ev"], f"{h}{n}_")
    q = {h: {n: fit_q90(errs[(h, n)]) for n in SIZES} for h in halves}
    k_by = {}
    for name, sizes in STRATA:
        k_by[name] = int(sum(errs[(h, n)]["group"].nunique() for h in halves for n in sizes))
    strata = pooled_strata(k_by)
    cov = {"k_before_pooling": k_by, "strata": {}}
    for sname, sizes in strata:
        hits, grp = [], []
        for fit_h, test_h in (("W", "E"), ("E", "W")):
            for n in sizes:
                d = errs[(test_h, n)]
                d = d[d["even"] > 0]
                moy = L.DATES[d["t"].to_numpy()].month.to_numpy()
                qq = np.array([q[fit_h][n][int(m)] for m in moy])
                hits.append(d["err"].abs().to_numpy() <= qq * d["even"].to_numpy())
                grp.append(("fit" + fit_h + "_") + d["group"].to_numpy().astype(object))
        g = coverage3.coverage_gate(np.concatenate(hits), np.concatenate(grp))
        g["sizes"] = list(sizes)
        cov["strata"][sname] = g
    cov["all_pass"] = bool(all(g["pass"] for g in cov["strata"].values()))
    cov["all_calibrated_envelope"] = bool(all(g["calibrated"] for g in cov["strata"].values()))
    cov["label"] = ("calibrated" if cov["all_pass"] and cov["all_calibrated_envelope"]
                    else "uncalibrated")
    full_q = {n: fit_q90(pd.concat([errs[("W", n)], errs[("E", n)]], ignore_index=True))
              for n in SIZES}
    sdm = {"q90_by_n_moy": {str(n): full_q[n] for n in SIZES}, "sizes": list(SIZES),
           "z90": L.Z90,
           "rule": "sd = q90(n_eff, moy) / 1.645 x even-split value of the Chow-Lin component; "
                   "q90 interpolated linearly in log(n_eff) between the fitted sizes, clamped "
                   "to [1, 300]; n_eff = (sum e)^2 / sum e^2 over the contributing Chow-Lin "
                   "meters' even-split energies e"}
    return sdm, cov


def run(variant: str, out_dir: Path, placebo: bool) -> dict:
    t0 = time.time()
    l1 = L.l1_pumps()
    prod = L.open_c0(variant)
    cds = {c: L.CountyData(variant, c, prod=prod, l1=l1) for c in L.COUNTIES}
    pz = np.load(L.WORK / f"{variant}_profiles.npz")
    profs = {c: pz[f"prof_{c}"] for c in ("yunlin", "changhua")}
    res = {"variant": variant, "code_sha256": code_sha(), "counties": {}}
    tests = [f"P2r2.a1.skill_{c}" for c in L.COUNTIES] + \
        [f"P2r2.a2.sdcov_{c}" for c in ("yunlin", "changhua")]
    res["authorization"] = scoring.authorize(tests)
    for county in L.COUNTIES:
        S = setup(variant, county, cds, profs)
        est = estimates(S)
        N = len(S["vi"])
        r = {"n_validation_meters": int(N),
             "pseudo_phase_counts": {p: int((S["ph"] == p).sum()) for p in ("A", "B")},
             "eval_months": [L.ym(S["ev"][0]), L.ym(S["ev"][-1])], "group_skill": {},
             "n_groups": {}}
        labs = {}
        for key, n in [(str(n), n) for n in SIZES] + [("fan", N)]:
            if key != "fan" and n > N:
                continue
            labs[key] = labels(S["xy"], n) if key != "fan" else np.zeros(N, int)
            r["n_groups"][key] = int(len(np.unique(labs[key])))
            r["group_skill"][key] = group_skill(est, S["truth"], labs[key], S["ev"])
        gated = [str(n) for n in SIZES if str(n) in r["group_skill"] and N // n >= 1]
        r["sizes_scored"] = gated
        r["min_skill_scored"] = float(min(r["group_skill"][k]["chowlin"] for k in gated))
        r["criterion"] = "chowlin skill >= 0.15 at every n in {1, 3, 10, 30, 100, 300}"
        r["passed"] = bool(all(r["group_skill"][str(n)]["chowlin"] >= 0.15 for n in SIZES
                               if str(n) in r["group_skill"])
                           and all(N // n >= 1 for n in SIZES))
        print(variant, county, N, {k: round(v["chowlin"], 4) for k, v in
                                   r["group_skill"].items()}, "passed", r["passed"], flush=True)
        if county != "chiayi":
            sdm, cov = sd_model_and_coverage(S, est)
            r["sd_coverage"] = cov
            r["sd_model"] = sdm
            print(variant, county, "coverage", {k: (round(g["coverage"], 3), g["k_clusters"],
                                                    g["pass"], g["calibrated"])
                                                for k, g in cov["strata"].items()},
                  cov["label"], flush=True)
        if placebo and county != "chiayi":
            T = S["T"]
            end = L.tidx("2022-12") if variant == "preorigin" else T - 1
            plac = {k: [] for k in labs}
            for seed in range(1, N_PLACEBO + 1):
                pp = L.year_shuffle(S["prof"][:end + 1], seed)
                pp = np.concatenate([pp, S["prof"][end + 1:]]) if end + 1 < T else pp
                e2 = estimates(S, prof=pp)
                for k, lab in labs.items():
                    plac[k].append(group_skill({"even": e2["even"], "chowlin": e2["chowlin"]},
                                               S["truth"], lab, S["ev"])["chowlin"])
            r["placebo_skill"] = {k: {"mean": float(np.mean(v)), "min": float(np.min(v)),
                                      "max": float(np.max(v))} for k, v in plac.items()}
            r["real_beats_all_placebos"] = {
                k: bool(r["group_skill"][k]["chowlin"] > r["placebo_skill"][k]["max"])
                for k in plac}
        res["counties"][county] = r
    res["seconds"] = round(time.time() - t0, 1)
    L.save_json(res, out_dir / f"t1_meter_validation_{variant}.json")
    return res


if __name__ == "__main__":
    out = Path(sys.argv[1])
    variant = "preorigin"
    if "--variant" in sys.argv:
        variant = sys.argv[sys.argv.index("--variant") + 1]
    run(variant, out, "--placebo" in sys.argv)
