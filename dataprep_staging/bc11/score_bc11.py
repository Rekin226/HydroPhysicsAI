"""Score I1.backcast_2010_11.v1 ONCE: 2011 heads from a 2010-12 start, months no model has
fitted (design-time, window flagged; user rulings R8.6a-c).

Order (no 2010-11 head is opened before step 5):

0. the registration is intact (``prereg3.load_prereg``) and every frozen code file matches
   its ``frozen_code`` sha256;
1. every pre-gate file (availability, E0, swap, panel, baselines) and every G0 result file
   matches the sha256 the registration froze; the availability stop rule and E0 passed;
2. heads/L1_pre2012 v2: its MANIFEST allows design_time_evidence (allowed_uses read before
   gating), its sha256 is the registered one, its wells are the registered wells and it was
   built from the registered availability output under the product_build unit;
3. the panel matches its json, was built at the declared code head with the registered
   theta files, and holds only registered wells; the 2012-22 baselines are reproduced
   exactly from L1_clean v3 (2012-22 only, spent; the CL / rho / sd files are bound);
4. ``prereg3.assert_can_score`` for every test (logs the scoring looks);
5. heads: L1_pre2012 v2, use design_time_evidence, window_end 2011-12, flags 0-1;
6. every gate with the role the registration froze (``roles``): G1 + G1b (primary), G2 per
   county, the Yunlin arm, G3; then the report-only rows (full persistence, the CF arm, the
   F/L memory line). report.json once (mode 'x'); every attempt appended to attempts.jsonl.

    python dataprep_staging/bc11/score_bc11.py --verify-registration
    python dataprep_staging/bc11/score_bc11.py score
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent / "ch_x"), str(HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
#: CPU entry point: this checkout's hydrophysics (the hash-bound copy), not an editable install
if str(HERE.parents[1]) not in sys.path[:1]:
    sys.path.insert(0, str(HERE.parents[1]))

import bc11_const as K  # noqa: E402
import bc11_lib as BL  # noqa: E402
import chx_io as IO  # noqa: E402

SCORER_VERSION = "1.0.0"
CHX, X = K.CHX, K.X
FROZEN = (f"{X}/bc11_const.py", f"{X}/bc11_lib.py", f"{X}/build_l1pre_v2.py",
          f"{X}/page_run.py", f"{X}/baselines_bc11.py", f"{X}/g0_bc11.py",
          f"{X}/score_bc11.py", f"{X}/run_bc11.sh",
          f"{CHX}/avail_bc11.py", f"{CHX}/avail_lib.py", f"{CHX}/avail_l1pre.py",
          f"{CHX}/bc_lib.py", f"{CHX}/build_l1pre.py", f"{CHX}/chx_const.py",
          f"{CHX}/chx_io.py", f"{CHX}/chx_lib.py", f"{CHX}/hindcast.py",
          "dataprep_staging/n2_policy/bootstrap_scorer.py", K.G0_PARAMS_FROM,
          "dataprep_staging/fv_decidability_v2/stat_fv.py",
          "dataprep_staging/fv_decidability_v2/osse_fv2.py",
          "dataprep_staging/fv_decidability/osse_fv.py",
          "dataprep_staging/h1_heads_qc/screens.py", "dataprep_staging/h1_heads_qc/config.py",
          "hydrophysics/twin/hybrid.py", "hydrophysics/twin/coverage.py",
          "hydrophysics/twin/drift_diag.py")
PREGATES = {"availability": K.AVAIL_OUT, "e0": K.E0_OUT, "swap": K.SWAP_OUT,
            "panel": K.PANEL, "panel_json": K.PANEL_JSON, "baselines": K.BASELINES,
            "baselines_json": K.BASELINES_JSON}
S_LABEL = "S (design-time, window flagged)"
NARROW_PHYSICS = "the physics level is a policy-comparison scenario only"
WORDING = {
    "g1_pass": f"{S_LABEL}: E04's recipe (CL and rho refitted on 2012-22) beat climatology "
               "and level-adjusted climatology on 2011 heads from a 2010-12 start, with "
               "90 % coverage {cov:.2f}, months no model has fitted (ratios {r_cl:.2f} / "
               "{r_adj:.2f}; 90 % upper bounds {ub_cl:.2f} / {ub_adj:.2f}); beside E04",
    "g1_fail": "E04's recipe did not beat both climatologies with calibrated 90 % intervals "
               "on 2011 heads from a 2010-12 start (design-time); E04 unchanged",
    "g2_pass": f"{S_LABEL} ({{county}}): the page physics, free-running on recorded pumping "
               "from 2007, was within E07's fair margin of climatology on 2011 heads (ratio "
               "{ratio:.2f} <= 1.25); beside E07, which stays N",
    "g2_fail": f"{{county}}: {NARROW_PHYSICS} (2011 backcast, design-time; ratio {{ratio:.2f}})",
    "yr_pass": f"{S_LABEL}: direction of the page model's Yunlin pumping response supported "
               "in 2010-11 (slope {beta:.2f} [{lo:.2f}, {hi:.2f}], well x calendar-month and "
               "month-of-sample effects)",
    "yr_fail": "direction of the page model's Yunlin pumping response not supported in "
               "2010-11 (design-time)",
    "g3_pass": "recorded 2007-11 pumping improved the page physics' 2011 heads over pumping "
               "climatology (design-time; ledger only)",
    "g3_fail": "recorded 2007-11 pumping did not improve the page physics' 2011 heads over "
               "pumping climatology (design-time; ledger only)",
    "not_evaluable": "not evaluable (too few scored wells after QC); ledger only",
    "report_only": "report-only (its synthetic G0 was not decidable or its pre-gate failed); "
                   "ledger only",
}


def frozen_code() -> dict:
    return {p: IO.sha_or_missing(HERE.parents[1] / p) for p in FROZEN}


def _kw(prereg_dir, rows_dir) -> dict:
    return {} if prereg_dir is None else {"prereg_dir": prereg_dir, "rows_dir": rows_dir}


def verify_registration(prereg_dir=None, rows_dir=None) -> tuple[dict, str]:
    from harness import prereg3

    try:
        doc, sha = prereg3.load_prereg(K.ID, **_kw(prereg_dir, rows_dir))
    except prereg3.PreregistrationError as e:
        raise SystemExit(f"REFUSED: {K.ID} is not registered or not intact: {e}") from e
    bad = {p: (doc["frozen_code"].get(p), h) for p, h in frozen_code().items()
           if doc["frozen_code"].get(p) != h}
    if bad:
        raise SystemExit(f"REFUSED: frozen code differs from the registration: {bad}")
    return doc, sha


def registered_ids(doc: dict) -> dict:
    t = next(t for t in doc["tests"] if t["test_id"] == K.TEST_G1)
    return {c: sorted(map(str, t["external_units"][K.UNIT_OF_COUNTY[c]]["ids"]))
            for c in K.COUNTIES}


def check_files(doc: dict) -> None:
    probs = []
    for key, path in PREGATES.items():
        if IO.sha_or_missing(IO.rp(path)) != doc["pregates"][key]["sha256"]:
            probs.append(f"{path} differs from the registration")
    for key, path in K.G0_FILES.items():
        if IO.sha_or_missing(IO.rp(path)) != doc["g0"][key]["sha256"]:
            probs.append(f"{path} differs from the registration")
    if not doc["pregates"]["availability"]["passed"] or not doc["pregates"]["e0"]["passed"]:
        probs.append("the availability stop rule or E0 did not pass")
    if probs:
        raise SystemExit("REFUSED (step 1): " + "; ".join(probs))


def check_product(doc: dict) -> dict:
    from common.manifest import load_manifest

    m, pdir = load_manifest(IO.rp(K.PRODUCT))
    probs = []
    if K.PRODUCT_USE not in m.get("allowed_uses", []):
        probs.append(f"L1_pre2012 v2 does not allow {K.PRODUCT_USE}")
    if IO.sha256(pdir / "MANIFEST.json") != doc["pregates"]["product"]["manifest_sha256"]:
        probs.append("L1_pre2012 v2's MANIFEST differs from the registration")
    par = m.get("parameters") or {}
    ids = registered_ids(doc)
    if sorted(par.get("wells") or []) != sorted(ids["yunlin"] + ids["changhua"]):
        probs.append("L1_pre2012 v2's wells differ from the registered units")
    if (par.get("availability") or {}).get("sha256") != doc["pregates"]["availability"][
            "sha256"]:
        probs.append("L1_pre2012 v2 was not built from the registered availability output")
    if (par.get("build_look") or {}).get("unit") != K.PRODUCT_UNIT:
        probs.append("L1_pre2012 v2 was not built under the product_build unit")
    if probs:
        raise SystemExit("REFUSED (step 2): " + "; ".join(probs))
    return {"allowed_uses": m["allowed_uses"], "gate": m.get("gate")}


def check_panel(doc: dict) -> dict:
    pz = IO.rp(K.PANEL)
    summ = json.loads(IO.rp(K.PANEL_JSON).read_text())
    probs = []
    if IO.sha256(pz) != summ["panel_sha256"]:
        probs.append("panel sha256 differs from its json")
    if summ.get("code_head") != doc["recipe"]["worktree_head"]:
        probs.append("panel not built at the declared commit")
    if summ.get("thetas_sha256") != doc["recipe"]["thetas_sha256"]:
        probs.append("page theta files differ")
    with np.load(pz, allow_pickle=False) as z:
        P = {k: z[k] for k in z.files}
    ids = registered_ids(doc)
    if not set(np.asarray(P["sid"]).astype(str)) <= set(ids["yunlin"] + ids["changhua"]):
        probs.append("panel wells outside the registered units")
    if [str(m) for m in P["months"]] != BL.WIN_M:
        probs.append("panel months are not 2010-01..2011-12")
    if probs:
        raise SystemExit("REFUSED (step 3): " + "; ".join(probs))
    return P


def check_baselines(base: dict, obs_fit: np.ndarray) -> dict:
    """The 2012-22 ingredients must be reproduced exactly from the re-read record."""
    fit = BL.fit_stat(obs_fit)
    ok = all(np.allclose(fit[k], base[k], rtol=0, atol=1e-12, equal_nan=True)
             for k in ("cl_month", "rho_used", "sd", "n_fit"))
    if not ok:
        raise SystemExit("REFUSED (step 3): the frozen baselines are not reproduced from "
                         "L1_clean v3 2012-22")
    return fit


# ----------------------------------------------------------------------------- arithmetic
def _line(key: str, **kw) -> str:
    return WORDING[key].format(**kw) if kw else WORDING[key]


def score_all(doc: dict, P: dict, base: dict, fit: dict, ids: list[str], obs: np.ndarray,
              flag: np.ndarray, obs_fit: np.ndarray) -> dict:
    """Every gate from the registered roles and the arrays (no file access)."""
    roles = doc["roles"]
    county = np.asarray(base["county"]).astype(str)
    if [str(s) for s in base["sid"]] != list(ids):
        raise SystemExit("ENGINEERING: baselines and registered wells differ in order")
    out = {}
    # G1 + G1b (primary)
    fc = BL.stat_forecast(fit, obs)
    g1 = BL.g1_score(fc, obs, county)
    if roles["g1"] != "gating":
        line = _line("report_only")
    elif g1["verdict"] == "not_evaluable":
        line = _line("not_evaluable")
    elif g1["passed"]:
        line = _line("g1_pass", cov=g1["coverage90"], r_cl=g1["ratio_vs_cl"],
                     r_adj=g1["ratio_vs_cl_level_adjusted"], ub_cl=g1["ub90_vs_cl"],
                     ub_adj=g1["ub90_vs_cl_level_adjusted"])
    else:
        line = _line("g1_fail")
    out["G1"] = {"role": roles["g1"], "result": g1, "line": line}
    # physics rows at the placed wells
    pos = {s: i for i, s in enumerate(ids)}
    rows = [pos[str(s)] for s in P["sid"]]
    pc = np.asarray(P["county"]).astype(str)
    mf = np.asarray(P["native_fit"], float)
    of = obs_fit[rows]
    ow = obs[rows]
    F1, F0 = np.asarray(P["F1"], float), np.asarray(P["F0"], float)
    s = slice(BL.N_PRE, None)
    g2 = BL.g2_score(mf, of, F1[:, s], ow[:, s], pc)
    out["G2"] = {}
    for c in K.COUNTIES:
        r = g2[c]
        role = roles["g2"][c]
        if role != "gating":
            line = _line("report_only")
        elif r["verdict"] == "not_evaluable":
            line = _line("not_evaluable")
        elif r["passed"]:
            line = _line("g2_pass", county=c, ratio=r["rmse_ratio_fair_cl"])
        else:
            line = _line("g2_fail", county=c, ratio=r["rmse_ratio_fair_cl"])
        out["G2"][c] = {"role": role, "result": r, "line": line}
    out["G2"]["pooled_report_only"] = g2["pooled_report_only"]
    # G3
    datum = np.asarray(base["phys_datum"], float)[rows]
    g3 = BL.g3_score(F1[:, s], F0[:, s], datum, ow[:, s])
    role = roles["g3"]
    line = (_line("report_only") if role != "gating" else _line("not_evaluable")
            if g3["verdict"] == "not_evaluable" else _line("g3_pass" if g3["passed"]
                                                           else "g3_fail"))
    out["G3"] = {"role": role, "result": g3, "line": line}
    # Yunlin response arm
    yl = [pos[str(x)] for x in P["sid_yl"]]
    yr = BL.score_yr(P, obs[yl], flag[yl])
    role = roles["yr"]
    if role != "gating":
        line = _line("report_only")
    elif yr["verdict"] in ("not_evaluable", "not_informative"):
        line = _line("not_evaluable") if yr["verdict"] == "not_evaluable" else \
            "the page model's Yunlin contrast is not informative; ledger only"
    elif yr["passed"]:
        b = yr["beta_primary"]
        line = _line("yr_pass", beta=b["beta"], lo=b["ci90"][0], hi=b["ci90"][1])
    else:
        line = _line("yr_fail")
    out["YR"] = {"role": role, "result": yr, "line": line}
    # report only: the CF origin-update arm and the F/L memory line
    cf = BL.cf_forecast(mf, of, F1, ow, datum)
    m = np.isfinite(ow[:, s]) & np.isfinite(datum)[:, None]
    cl_rows = fc["cl"][rows][:, s]
    phys = F1[:, s] - datum[:, None]
    out["report_only"] = {
        "cf_ratio_vs_cl": BL.equal_station_ratio(BL.station_msq(cf[:, s], ow[:, s], m),
                                                 BL.station_msq(cl_rows, ow[:, s], m)),
        "f_l_memory_line": {
            "lead_ratio_stat_vs_cl": BL.lead_ratios(fc["pred"][rows][:, s], cl_rows,
                                                    ow[:, s], m),
            "lead_ratio_physics_f1_vs_cl": BL.lead_ratios(phys, cl_rows, ow[:, s], m),
            "lead_ratio_cf_vs_cl": BL.lead_ratios(cf[:, s], cl_rows, ow[:, s], m),
            "forcing_departure_p90_m": BL.forcing_departure_p90(F1[:, s], F0[:, s])},
        "dropped_not_placed": [str(x) for x in P.get("not_placed", [])]}
    return out


# ----------------------------------------------------------------------------- run
def _attempt(rec: dict) -> None:
    p = IO.rp(K.ATTEMPTS)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def run_score(prereg_dir=None, rows_dir=None, l1=None) -> dict:
    out = IO.rp(K.REPORT)
    if out.exists():
        raise SystemExit(f"{K.REPORT} exists; {K.ID} is scored once")
    l1 = l1 or IO.l1_monthly
    doc, sha = verify_registration(prereg_dir, rows_dir)
    check_files(doc)
    prod = check_product(doc)
    P = check_panel(doc)
    from baselines_bc11 import load as load_base

    base = load_base(K.BASELINES)
    ids = [str(s) for s in base["sid"]]
    reg = registered_ids(doc)
    if sorted(ids) != sorted(reg["yunlin"] + reg["changhua"]):
        raise SystemExit("REFUSED (step 3): baseline wells differ from the registered units")
    obs_fit, _, fit_notes = l1(ids, BL.FIT_M, product=K.CL_PRODUCT, use=K.CL_USE,
                               window_end=K.CL_WINDOW_END)
    fit = check_baselines(base, obs_fit)
    from harness import prereg3

    acc = prereg3.assert_can_score(K.ID, K.WP, **_kw(prereg_dir, rows_dir))
    obs, flag, notes = l1(ids, BL.WIN_M, product=K.PRODUCT, use=K.PRODUCT_USE,
                          window_end=K.WINDOW[1])
    res = score_all(doc, P, base, fit, ids, obs, flag, obs_fit)
    rep = {"prereg_id": K.ID, "prereg_sha256": sha, "scorer_version": SCORER_VERSION,
           "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "product": prod, "guard_notes": notes, "fit_guard_notes": fit_notes,
           "post_registration_looks": acc["post_registration_looks"],
           "unverifiable_prior_looks": acc["unverifiable_prior_looks"][:20],
           "roles": doc["roles"], **res}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as f:
        json.dump(rep, f, indent=1, default=float)
    _attempt({"time": rep["created"], "step": "score", "rc": 0,
              "report_sha256": IO.sha256(out)})
    print(f"{K.ID}: G1 {res['G1']['result']['verdict']} ({res['G1']['role']}); G2 "
          + ", ".join(f"{c} {res['G2'][c]['result']['verdict']}" for c in K.COUNTIES)
          + f"; YR {res['YR']['result']['verdict']}; G3 {res['G3']['result']['verdict']} "
          f"-> {K.REPORT}")
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--verify-registration", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("score")
    a = ap.parse_args(argv)
    if a.verify_registration:
        _, sha = verify_registration()
        print(f"{K.ID} registered and intact (sha256 {sha[:16]})")
        return 0
    if a.cmd != "score":
        ap.error("use --verify-registration or score")
    run_score()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
