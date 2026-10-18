"""WP S5 repair round: the registered monthly-field procedure and test ``S5.monthly_field.v1``.

    tools/cpu_slot.sh python -I s5r_mtest.py hyper    stage-2 hyperparameters (pre-origin)
                                                      -> S5_evidence/repair/hyper_stage2.json
    tools/cpu_slot.sh python -I s5r_mtest.py score    test M1 -> S5_evidence/repair/m1.json

Pre-origin only (U7): the annual fit is the registered round-1 model on pre-origin data
WITHOUT GNSS (leveling 2012-2021, InSAR run P, rings <= 2021, piles <= 2022-12;
hyper_pre.json, f from hyper_pre.json); stage 2 (``s5r_stage2``) is fitted on that fit's
InSAR run-P month-group residuals 2020-01..2022-12. GNSS never enters the predictions.

Test M1 (gate of the monthly component). Targets: WRA monthly GNSS (gnss_clean/preorigin/v2,
flags 0-1, n_days >= 10) 2020-01..2022-12 at stations with >= 12 months, in months with >= 7
such stations. Every station-month residual g = y - u_annual (the GNSS-out annual field incl.
the seasonal pair and the mean velocity) and the stage-2 anomaly prediction p (posterior mean
at the station point) pass through ONE linear operator T = I - X (X'X)^+ X' whose nuisance
design X holds a station offset and a station rate (GNSS offset and excess-rate terms) and a
plane (1, east, north) per month (family common mode and the plane InSAR cannot see). Scored:
station-months whose station cell lies in the month's InSAR footprint (a 4 km unit with a
run-P value that month) and whose PLANE-FREE anomaly variance reduction at the station point
(``Stage2.point_vr_dt``) is >= 0.5 - the product's support rule.

* evaluable: >= 100 scored station-months at >= 8 stations;
* skill (gate): RMSE(T(g - p)) / RMSE(T g) < 1 over the scored station-months AND the
  station-cluster bootstrap of that ratio (2000 draws, seed 5006) gives P(ratio >= 1) <= 0.10;
* sd label: coverage3 gate-3 (station clusters) of |T(g - p)| <= 1.645 sd, where
  sd^2 = diag(T (S_annual + S_anomaly + S_noise) T') with S_annual the annual posterior
  covariance at the station-months, S_anomaly the stage-2 posterior covariance and S_noise
  diag(sig_gnss^2 + sd_month^2) + sig_cm^2 (same month); pass -> the monthly sd is labelled
  "calibrated at monthly resolution on pre-origin WRA station-months", else "uncalibrated".

Report only: the same numbers on all station-months, by year, corr(T g, T p), and the
stage-2 likelihood ratio against a noise-only model.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import s5const as C  # noqa: E402
import s5data as D  # noqa: E402
import s5fit as F  # noqa: E402
import s5hyper as Hy  # noqa: E402
import s5r_const as R  # noqa: E402
import s5r_stage2 as S2  # noqa: E402

FAMS_ANNUAL = ("lev", "insar", "cmp")


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def rmse(e) -> float:
    e = np.asarray(e, float)
    return float(np.sqrt(np.mean(e ** 2))) if e.size else float("nan")


def score_gate() -> dict:
    from harness import prereg3
    r = prereg3.assert_can_score(R.MTEST_ID, C.WP, test_id=R.MTEST_TEST)
    return {"prereg_sha256": r["sha256"],
            "post_registration_looks": len(r["post_registration_looks"]),
            "unverifiable_prior_looks": len(r["unverifiable_prior_looks"])}


# ============================================================================ set-up
def build_pre(evidence: Path = C.EVIDENCE) -> dict:
    """GNSS-out pre-origin annual fit and the stage-2 problem on its InSAR residuals."""
    base = json.loads((evidence / "hyper_pre.json").read_text())
    h, fdict = base["hyper"], base["fdict"]
    data, qc = Hy.load("pre")
    m = Hy.masks_from(data, base["flagged"])
    pr = F.make_problem(data, C.KNOTS_PRE, FAMS_ANNUAL, masks=m, fdict=fdict)
    post = F.fit(pr, h)
    groups = S2.residual_groups(pr, post)
    months = S2.month_range(min(g["month"] for g in groups), max(g["month"] for g in groups))
    s2 = S2.Stage2(pr.lat.Z, pr.cell_xy, pr._unit_xy, pr.insar["unit_cells"], months, groups)
    return {"base": base, "h": h, "data": data, "masks": m, "pr": pr, "post": post, "s2": s2,
            "qc": qc}


def noise_start(h: dict) -> dict:
    return {"sig_ins2": h["sig_ins"], "sig_aps2": h["sig_aps"], "ell_aps2": h["ell_aps_km"]}


# ============================================================================ hyperparameters
def hyper(out: Path, evidence: Path = C.EVIDENCE) -> dict:
    """Registered stage-2 procedure: the noise-only model first (sig_ins2, sig_aps2, ell_aps2
    from the annual model's InSAR noise), then the full model from its optimum with sig_a
    3.0, ell_a = ell_km and rho 0.5 (Nelder-Mead + one restart; best evaluated point)."""
    t0 = time.time()
    S = build_pre(evidence)
    s2 = S["s2"]
    log(f"stage 2: {len(s2.groups)} month groups, {s2.M} months, {s2.m} nodes "
        f"({time.time() - t0:.0f} s)")
    hn0 = noise_start(S["h"])
    hn, info_n = S2.fit_null(s2, hn0)
    log(f"noise-only {json.dumps({k: round(v, 3) for k, v in hn.items()})} m2ll "
        f"{info_n['m2ll']:.3f}")
    h0 = dict(hn)
    h0.update({"sig_a": R.S2_START["sig_a"], "ell_a": S["h"]["ell_km"],
               "rho": R.S2_START["rho"]})
    h2, info = S2.fit_hyper(s2, h0, log=log)
    sol = s2.solve(h2)
    out_d = {"hyper": h2, "start": h0, "fit": info, "m2ll": sol["m2ll"],
             "null": {"hyper": hn, "start": hn0, **info_n},
             "lr_null_minus_full": info_n["m2ll"] - sol["m2ll"],
             "groups": len(s2.groups), "months": s2.months, "nodes": s2.m,
             "units": int(len(s2.unit_xy)), "fit_window_end": "2022-12",
             "annual_fit": "pre-origin, families lev+insar+cmp (no GNSS), hyper_pre.json",
             "seconds": round(time.time() - t0, 1)}
    F.dump(out_d, out)
    log(f"stage-2 hyperparameters {json.dumps({k: round(v, 3) for k, v in h2.items()})}; "
        f"null - full {out_d['lr_null_minus_full']:.1f} -> {out}")
    return out_d


# ============================================================================ test M1
def nuisance_design(st: np.ndarray, mi: np.ndarray, t: np.ndarray, xy: np.ndarray
                    ) -> np.ndarray:
    """Station offset + station rate + per-month plane columns over the station-months."""
    cols = []
    for s in np.unique(st):
        sel = st == s
        c1 = sel.astype(float)
        c2 = np.where(sel, t - t[sel].mean(), 0.0)
        cols += [c1, c2]
    xc = xy - xy.mean(0)
    for k in np.unique(mi):
        sel = mi == k
        cols += [sel.astype(float), np.where(sel, xc[:, 0], 0.0), np.where(sel, xc[:, 1], 0.0)]
    return np.column_stack(cols)


def m1_rows(S: dict, h2: dict) -> tuple[pd.DataFrame, dict]:
    """Station-month table, the operator T, predictions and predictive sd (no target)."""
    pr, post, s2, h = S["pr"], S["post"], S["s2"], S["h"]
    gn = S["data"]["gnss"][S["masks"]["gnss"]]
    gn = gn[(gn.family == R.M1_FAMILY) & (gn.month >= R.M1_WINDOW[0])
            & (gn.month <= R.M1_WINDOW[1])].copy()
    n_st = gn.groupby("station").size()
    gn = gn[gn.station.isin(n_st[n_st >= R.M1_MIN_MONTHS].index)]
    n_m = gn.groupby("month").station.nunique()
    gn = gn[gn.month.isin(n_m[n_m >= R.M1_PLANE_MIN_OTHERS + 1].index)]
    gn = gn.sort_values(["station", "t"]).reset_index(drop=True)
    mi_of = {mo: i for i, mo in enumerate(s2.months)}
    gn = gn[gn.month.isin(mi_of)].reset_index(drop=True)
    mi = gn.month.map(mi_of).to_numpy()
    xy = gn[["x_km", "y_km"]].to_numpy()
    t = gn.t.to_numpy()
    # annual prediction (GNSS-out fit) and its posterior covariance
    G = np.vstack([F.field_rows(pr, xy[i], t[i:i + 1]) for i in range(len(gn))])
    u_ann = G @ post.theta
    cov_ann = post.cov_of(G)
    # stage-2 anomaly at the station points
    sol = s2.solve(h2)
    s2.lattice(h2["ell_a"])
    phi = s2.point_weights(xy)
    p = np.einsum("ij,ij->i", phi, sol["x"][mi])
    cov_an = sol["P"].functional_cov(mi, phi)
    vr = s2.point_vr_dt(h2, sol, xy, mi)
    # support: station cell inside the month's InSAR footprint, plane-free VR >= 0.5
    g = D.grid()
    cell = D.cell_of(g, 1000.0 * xy[:, 0], 1000.0 * xy[:, 1])
    observed = np.array([bool(c >= 0 and s2.footprint[k, c])
                         for k, c in zip(mi, cell, strict=True)])
    # noise: white GNSS + family common mode (same month)
    same = (mi[:, None] == mi[None, :]).astype(float)
    cov_noise = np.diag(h["sig_gnss"] ** 2 + gn.sd.to_numpy() ** 2) + h["sig_cm"] ** 2 * same
    X = nuisance_design(gn.station.to_numpy(), mi, t, xy)
    T = np.eye(len(gn)) - X @ np.linalg.pinv(X)
    Sig = cov_ann + cov_an + cov_noise
    sd_e = np.sqrt(np.clip(np.einsum("ij,jk,ik->i", T, Sig, T), 1e-12, None))
    tab = pd.DataFrame({"station": gn.station.to_numpy(), "month": gn.month.to_numpy(),
                        "t": t, "mi": mi, "cell": cell, "u_annual": u_ann, "p": p,
                        "Tp": T @ p, "sd_e": sd_e, "vr": vr, "observed": observed,
                        "supported": observed & (np.nan_to_num(vr, nan=-1.0)
                                                 >= R.VR_A_MIN),
                        "_y": gn.y.to_numpy()})
    info = {"n_station_months": int(len(gn)), "stations": int(gn.station.nunique()),
            "months": int(len(np.unique(mi))), "nuisance_columns": int(X.shape[1]),
            "rank_T": int(round(float(np.trace(T))))}
    return tab, {"T": T, **info}


def boot_ratio(e, z, clusters, n=C.BOOT_N, seed=R.M1_SEED) -> dict:
    e, z = np.asarray(e, float), np.asarray(z, float)
    cl = np.asarray(clusters).astype(str)
    u = np.unique(cl)
    idx = {c: np.flatnonzero(cl == c) for c in u}
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for i in range(n):
        pick = np.concatenate([idx[c] for c in rng.choice(u, size=u.size, replace=True)])
        out[i] = rmse(e[pick]) / max(rmse(z[pick]), 1e-12)
    return {"p05": float(np.quantile(out, 0.05)), "p50": float(np.quantile(out, 0.5)),
            "p95": float(np.quantile(out, 0.95)), "p_ge_1": float(np.mean(out >= 1.0)),
            "n_boot": n, "seed": seed, "k_clusters": int(u.size)}


def coverage(hits, clusters) -> dict:
    from harness import coverage3
    hits = np.asarray(hits, bool)
    if hits.size == 0:
        return {"n": 0, "pass": False, "note": "no targets"}
    _, cl = np.unique(np.asarray(clusters).astype(str), return_inverse=True)
    return coverage3.coverage_gate(hits.astype(int), cl)


def _block(tab: pd.DataFrame) -> dict:
    out = {"n": int(len(tab)), "stations": int(tab.station.nunique())}
    if not len(tab):
        return out
    out.update({"rmse_monthly": rmse(tab.err), "rmse_annual_only": rmse(tab.z),
                "ratio": rmse(tab.err) / max(rmse(tab.z), 1e-12),
                "corr_z_Tp": float(np.corrcoef(tab.z, tab.Tp)[0, 1]) if len(tab) > 2 else None,
                "rms_z_score": float(np.sqrt(np.mean((tab.err / tab.sd_e) ** 2)))})
    return out


def score(out: Path, evidence: Path = C.EVIDENCE, gate=score_gate) -> dict:
    t0 = time.time()
    S = build_pre(evidence)
    hs = json.loads((evidence / "repair" / "hyper_stage2.json").read_text())
    h2 = hs["hyper"]
    tab, info = m1_rows(S, h2)
    T = info.pop("T")
    log(f"M1 {info['n_station_months']} station-months at {info['stations']} stations; "
        f"{int(tab.supported.sum())} supported ({time.time() - t0:.0f} s)")
    reg = gate()
    # ---------------- score (first comparison of a prediction with the withheld GNSS)
    g = tab.pop("_y").to_numpy() - tab.u_annual.to_numpy()
    tab["z"] = T @ g
    tab["err"] = tab.z - tab.Tp
    tab["hit"] = np.abs(tab.err) <= R.Z90 * tab.sd_e
    Sc = tab[tab.supported]
    res = {"test_id": R.MTEST_TEST, "registration": reg, "stage2_hyper": h2, **info,
           "n_scored": int(len(Sc)), "stations_scored": int(Sc.station.nunique())}
    evaluable = res["n_scored"] >= R.M1_MIN_TARGETS and \
        res["stations_scored"] >= R.M1_MIN_STATIONS
    res["scored"] = _block(Sc)
    if evaluable:
        res["scored"]["boot_ratio"] = boot_ratio(Sc.err, Sc.z, Sc.station)
        res["scored"]["coverage"] = coverage(Sc.hit, Sc.station)
    res["criteria"] = {"evaluable": bool(evaluable),
                       "skill": bool(evaluable and res["scored"]["ratio"] < 1.0
                                     and res["scored"]["boot_ratio"]["p_ge_1"]
                                     <= R.M1_BOOT_P_MAX)}
    res["pass"] = bool(all(res["criteria"].values()))
    res["sd_label"] = ("calibrated at monthly resolution on pre-origin WRA station-months"
                       if evaluable and res["scored"]["coverage"]["pass"] else "uncalibrated")
    res["all_station_months"] = _block(tab)
    res["all_station_months"]["coverage"] = coverage(tab.hit, tab.station)
    tab["year"] = np.floor(tab.t).astype(int)
    res["scored_by_year"] = {int(y): _block(gg) for y, gg in Sc.assign(
        year=np.floor(Sc.t).astype(int)).groupby("year")}
    res["lr_null_minus_full"] = hs.get("lr_null_minus_full")
    res["seconds"] = round(time.time() - t0, 1)
    tab.to_parquet(out.with_name("m1_targets.parquet"), index=False)
    F.dump(res, out)
    log(f"M1 pass {res['pass']} {json.dumps(res['criteria'])}; sd {res['sd_label']}")
    return res


def main(argv=None) -> int:
    warnings.simplefilter("ignore", RuntimeWarning)
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["hyper", "score"])
    a = ap.parse_args(argv)
    R.EVIDENCE_R.mkdir(parents=True, exist_ok=True)
    if a.what == "hyper":
        out = R.EVIDENCE_R / "hyper_stage2.json"
        if out.exists():
            raise SystemExit(f"{out} exists; refusing to overwrite")
        hyper(out)
    else:
        out = R.EVIDENCE_R / "m1.json"
        if out.exists():
            raise SystemExit(f"{out} exists; refusing to overwrite")
        score(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
