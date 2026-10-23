"""Package the WP H2 products (heads/L2_filled_preorigin, heads/L2_filled_full).

Releases (each a new version directory; existing versions are never touched):

* ``v1_fill``   the registered v1 configuration (H2.fill_acceptance.v1). Gate: all six v1
  fill tests. It failed (N, S, E coverage), so by the v1 packaging rule the fill is published
  for ``design_time_evidence`` and ``after_the_fact_analysis`` only (full variant:
  ``after_the_fact_analysis`` only).
* ``v1_bridge`` (pre-origin only) the daily-bridge component of v1 (flag 0-2 rows), released
  under the v1 packaging rule declared before any fresh result: gate = H2.fresh.d alone, for
  ``calibration_target`` (U2: observed and daily-bridged months) and after-the-fact analysis.
* ``v2_fill``   the repair round (H2.fill_acceptance.v2: v2 sd factors). Gate: the five v2
  fill tests and the carried-over v1 H2.fresh.d. If it passes: pre-origin
  ``initial_conditions``, ``idw_field`` (flags 0-4), ``calibration_target`` (flags 0-2),
  ``model_design``, ``after_the_fact_analysis``, ``design_time_evidence``; full variant
  ``statistical_layer_history`` and ``after_the_fact_analysis``. If it fails: as v1_fill.

Each version gets ``fill.parquet`` (``s5_build.build``), ``validation.json``,
``PRODUCT_CARD.md`` and ``MANIFEST.json`` (plan 2.3; schema-checked; written last).

Usage: python -I s8_package.py RELEASE VARIANT     (RELEASE v1_fill|v1_bridge|v2_fill;
                                                    VARIANT pre|full)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2])]

from common import guard, manifest  # noqa: E402

from h2_heads_fill import config as C  # noqa: E402
from h2_heads_fill import repair_v2 as R  # noqa: E402
from h2_heads_fill import s5_build, scoring  # noqa: E402

V1_TESTS = ("H2.fresh.N", "H2.fresh.S", "H2.fresh.E", "H2.fresh.H", "H2.fresh.T", "H2.fresh.d")
V2_TESTS = tuple(f"H2.fresh2.{f}" for f in R.FAMILIES_V2)
CODE = sorted(str(p.relative_to(C.REPO)) for p in C.CODE_DIR.glob("*.py"))
LIB = ["dataprep_staging/harness/gaps.py", "dataprep_staging/harness/gaps_guarded.py",
       "dataprep_staging/harness/coverage.py", "dataprep_staging/harness/coverage2.py",
       "dataprep_staging/harness/coverage3.py", "dataprep_staging/harness/scorers.py",
       "dataprep_staging/harness/prereg3.py", "dataprep_staging/common/guard.py",
       "dataprep_staging/common/manifest.py", "dataprep_staging/common/paths.py",
       "dataprep_staging/common/uses.py", "hydrophysics/twin/grid.py"]
L1 = "results/dataprep/heads/L1_clean/v3"
EV = "results/dataprep/reports/H2_evidence"
INPUTS_COMMON = [f"{L1}/MANIFEST.json", f"{L1}/heads_monthly.parquet",
                 f"{L1}/heads_daily.parquet",
                 "results/dataprep/validation_units/v3/MANIFEST.json",
                 "results/dataprep/validation_units/v3/wells.csv",
                 "results/twin/basemap.npz", "results/twin_forward/datum_gate_innov.npz",
                 f"{EV}/design/design_params.json", f"{EV}/design/bridge_form.json",
                 f"{EV}/port_check.json",
                 "results/dataprep/preregistration/H2.fill_acceptance.v1.json"]
USES_PRE_PASS = ["initial_conditions", "idw_field", "calibration_target", "model_design",
                 "after_the_fact_analysis", "design_time_evidence"]
USES_FULL_PASS = ["statistical_layer_history", "after_the_fact_analysis"]
ALL_USES = ["scoring_target", "baseline", "datum_fit", "prospective_scorer",
            "calibration_target", "initial_conditions", "idw_field", "prior",
            "prior_width_input", "model_design", "selection", "forcing_frozen",
            "forcing_recorded_arm", "validator", "statistical_layer_history",
            "after_the_fact_analysis", "design_time_evidence"]


def load_records(tests, reg_dir):
    return {t: json.loads((reg_dir / f"{t}.json").read_text()) for t in tests}


def acceptance_rows(res: dict) -> list[dict]:
    rows = []
    for t, r in res.items():
        for s, v in r["strata_clean"].items():
            c = v["coverage_gate"]
            rows.append({
                "test": t, "stratum": s, "criterion": "coverage of the 90 % interval, rule "
                "gate-3 (clean truth)", "threshold": "cluster-adjusted interval contains 0.90"
                "; point in 0.85-0.95 when n_eff_lower >= 30",
                "measured": f"{c['coverage']:.3f} [{c['lo']:.3f}, {c['hi']:.3f}], n {c['n']},"
                            f" k {c['k_clusters']}, n_eff_lower {c['n_eff_lower']:.0f}, "
                            f"calibrated {c['calibrated']}", "pass": bool(c["pass"])})
            b = {k: x for k, x in v["baselines"].items() if x["defined"]}
            rows.append({
                "test": t, "stratum": s, "criterion": "RMSE below every defined baseline "
                "(backfill, clim+AR(1), STK, IDW on IDW cells; clean truth)",
                "threshold": "product RMSE < baseline RMSE",
                "measured": "; ".join(f"{k} {x['rmse_product']:.3f} vs {x['rmse_baseline']:.3f}"
                                      for k, x in b.items()),
                "pass": bool(v["rmse_pass"])})
    return rows


def release_spec(release: str, variant: str) -> dict:
    v1 = load_records(V1_TESTS, R.REG_V1)
    if release == "v1_fill":
        recs, passed = v1, all(r["pass"] for r in v1.values())
        params, mode = R.PARAMS_V1, "fill"
        crit = ("H2.fill_acceptance.v1: every stratum of the six fill tests passes coverage "
                "(gate-3) and beats every defined baseline on clean truth")
    elif release == "v1_bridge":
        if variant != "pre":
            raise SystemExit("the bridge component is a pre-origin release only")
        recs = {"H2.fresh.d": v1["H2.fresh.d"]}
        passed = v1["H2.fresh.d"]["pass"]
        params, mode = R.PARAMS_V1, "bridge"
        crit = ("v1 packaging rule (declared in H2.fill_acceptance.v1 before any fresh "
                "result): the daily-bridge component is released with gate H2.fresh.d alone "
                "when the conjunctive gate fails and H2.fresh.d passes")
    elif release == "v2_fill":
        v2 = load_records(V2_TESTS, R.REG_V2)
        recs = dict(v2) | {"H2.fresh.d": v1["H2.fresh.d"]}
        passed = all(r["pass"] for r in recs.values())
        params, mode = R.PARAMS_V2, "fill"
        crit = ("H2.fill_acceptance.v2: every stratum of the five v2 fill tests passes "
                "coverage (gate-3) and beats every defined baseline on clean truth, and the "
                "carried-over v1 H2.fresh.d passed")
    else:
        raise SystemExit(release)
    if variant == "pre":
        fwe = C.PRE_END
        if release == "v1_bridge":
            allowed = ["calibration_target", "after_the_fact_analysis"] if passed else \
                ["after_the_fact_analysis"]
        else:
            allowed = USES_PRE_PASS if passed else ["design_time_evidence",
                                                    "after_the_fact_analysis"]
    else:
        fwe = C.FULL_END
        allowed = USES_FULL_PASS if passed else ["after_the_fact_analysis"]
    forbidden = [u for u in ALL_USES if u not in allowed]
    return {"records": recs, "passed": bool(passed), "params": params, "mode": mode,
            "criteria": crit, "allowed": allowed, "forbidden": forbidden, "fwe": fwe}


def card(release, variant, version, spec, stats, ic) -> str:
    product = C.PRODUCT_PRE if variant == "pre" else C.PRODUCT_FULL
    passed = spec["passed"]
    window = ("pre-origin: every fitted parameter saw 2012-01..2022-12 only" if variant == "pre"
              else "full: fitted on 2012-01..2026-09, 2023-26 test heads included")
    what = {"v1_fill": "registered v1 configuration (complete fill)",
            "v1_bridge": "daily-bridge component of v1 (observations and flag-2 months only)",
            "v2_fill": "repair-round configuration v2 (complete fill; v2 sd factors)"}[release]
    L = [f"# {product} {version}: {what}", "", f"{window}.", ""]
    L.append(f"Monthly groundwater heads of the L1 v3 wells with flag 0/1 data: "
             f"{stats['rows']} rows, {stats['wells']} wells, {stats['months']} months. Every "
             f"non-observed value carries a 1-sigma sd (head_sd_m), a flag and a method.")
    L.append("")
    L.append(f"Gate: {'passed' if passed else 'FAILED'}. {spec['criteria']}. Allowed uses: "
             f"{', '.join(spec['allowed'])}. Details in validation.json.")
    L.append("")
    L.append("## Flags in this version")
    L.append("")
    names = {0: "observed (L1 flag 0; method observed or observed_partial)",
             1: "observed, L1 barometric correction",
             2: "daily bridge of a partial or outage month (same instrument)",
             3: "router fill (neighbour ridge + softImpute; uses other wells)",
             4: "own-record temporal model (GP in months with no observation anywhere)",
             6: "trailing-edge fill after the well's last observation (beyond the record end; "
                "never for calibration)",
             8: "missing, not filled (outside the tested envelope)"}
    for f, n in sorted(stats["by_flag"].items(), key=lambda x: int(x[0])):
        L.append(f"- {f}: {n} rows - {names.get(int(f), '')}")
    L.append("")
    L.append("## Use")
    L.append("")
    if variant == "pre":
        L += ["- Calibration targets (if allowed): flags 0-2 only (the guard drops 3-8), weighted "
              "by u2_weight = sigma_obs^2 / (sigma_obs^2 + sd^2) (U2). Flag 3/4 fills are "
              "functions of other calibration wells: never targets.",
              "- Initial conditions and IDW head fields (if allowed): flags 0-4.",
              "- Never scoring targets, baselines, datum fits or the prospective scorer (U1)."]
        if ic:
            L.append(f"- ic_field_2012-01.npz: January 2012 head fields per layer on the "
                     f"model grid (idw, ked, gp; registered decision '{ic['decision']}', "
                     f"leave-one-well-out RMSE {ic['rmse_lowo']}).")
    else:
        L += ["- Statistical-layer history and after-the-fact analysis only (if allowed). Never "
              "for calibration, initial conditions, design or selection of a model scored on "
              "2023-26 (U3).",
              "- Feb-Mar 2026 (the network outage) is daily-bridged (flag 2)."]
    L.append("")
    L.append("## Columns")
    L.append("")
    L.append("sid, date (month start, ISO), month, head_m (= value), head_sd_m (= sd; 0 for "
             "observations), flag, method, gap_class, gap_run_len, edge_dist, bridged_share, "
             "l1_n_clean_days, sigma_obs_m, u2_weight, in_model (one of the twin's 158 wells), "
             "layer, county, product_version, source_ids.")
    L.append("")
    L.append("## Known limits")
    L.append("")
    L += ["- Edge fills are weak (leading up to 66 months, trailing up to 48; largest sd "
          "factors).",
          "- Truth for every test is the L1 v3 C20 monthly mean; 'clean' truth excludes months "
          "with or next to flag-9 days and the three pumping-cycle wells (07080132, 07080122, "
          "10050211), whose flagged episodes may be pump regimes (H1a).",
          "- Neighbour regression can carry an undetected L1 offset into another well's fill "
          "(plan risk; H1a stage-1 detection misses some 1-2 m episodes).",
          "- L1 partial months (3-19 clean days) the bridge cannot complete stay observations "
          "(method observed_partial) with U2's sigma_obs(k); they are not router inputs.",
          "- With 14 gated strata, a calibrated configuration fails some stratum by chance "
          "about half the time (A1 coverage rule at level 0.95 per stratum)."]
    L.append("")
    return "\n".join(L)


def package(release: str, variant: str) -> Path:
    spec = release_spec(release, variant)
    product = C.PRODUCT_PRE if variant == "pre" else C.PRODUCT_FULL
    d = manifest.new_version_dir(product)
    stats = s5_build.build(variant, d, params_path=spec["params"], mode=spec["mode"],
                           with_ic=(release != "v1_bridge"))
    meta_in = f"{L1}/station_meta_preorigin.parquet" if variant == "pre" else \
        f"{L1}/station_meta_clean.parquet"
    ic = json.loads((R.REG_V1 / "H2.ic.lowo.json").read_text())
    after = None
    ap = C.EVIDENCE_DIR / "after_the_fact" / "full_variant_checks.json"
    if variant == "full" and ap.exists():
        after = json.loads(ap.read_text())
    params = json.loads(Path(spec["params"]).read_text())
    val = {"schema": "h2-validation/1", "product": product, "version": d.name,
           "release": release, "gate_passed": spec["passed"], "gate_criteria": spec["criteria"],
           "acceptance": acceptance_rows(spec["records"]),
           "registered_tests": {t: {"pass": r["pass"], "strata_clean": {
               s: {"coverage": v["coverage_gate"], "baselines": v["baselines"],
                   "rmse_product": v["rmse"].get("rmse_product"), "pass": v["pass"]}
               for s, v in r["strata_clean"].items()} if "strata_clean" in r else None,
               "reported": r.get("reported")} for t, r in spec["records"].items()},
           "ic_decision": {k: ic[k] for k in ("decision", "rmse_lowo", "n_wells",
                                              "rmse_by_layer", "n_by_layer",
                                              "bootstrap_diff_vs_idw_p05_p50_p95",
                                              "native_cov90")},
           "design_params": params, "build_stats": stats}
    if after is not None:
        val["after_the_fact"] = after
    scoring.jdump(val, d / "validation.json")
    (d / "PRODUCT_CARD.md").write_text(card(release, variant, d.name, spec, stats,
                                            stats.get("ic")))
    regs = sorted({str(p.relative_to(C.REPO)) for t in spec["records"]
                   for p in (R.REG_V1 / f"{t}.json", R.REG_V2 / f"{t}.json") if p.exists()})
    extra_in = [meta_in, str(Path(spec["params"]).relative_to(C.REPO)),
                str((R.REG_V1 / "H2.ic.lowo.json").relative_to(C.REPO))] + regs
    if release == "v2_fill":
        extra_in.append("results/dataprep/preregistration/H2.fill_acceptance.v2.json")
    if after is not None:
        extra_in.append(str(ap.relative_to(C.REPO)))
    m = manifest.build_manifest(
        d, product=product, layer="L2", inputs=INPUTS_COMMON + sorted(set(extra_in)),
        code=CODE + LIB,
        parameters={"release": release, "mode": spec["mode"], "design": params,
                    "fit_window": [C.START, C.PRE_END if variant == "pre" else C.FULL_END],
                    "seeds": "harness.gaps.rng_for(family, r); split-conformal seed "
                             f"{C.SEED_SPLIT}", "bridge_form": C.BRIDGE_FORM,
                    "envelope": {"min_obs_months": C.MIN_OBS_MONTHS,
                                 "max_internal_run": C.MAX_INTERNAL_RUN,
                                 "max_lead": C.MAX_LEAD, "max_trail": C.MAX_TRAIL,
                                 "sd_floor": C.SD_FLOOR}},
        fit_window_end=spec["fwe"], validation_report="validation.json",
        gate={"passed": spec["passed"], "criteria": spec["criteria"]},
        allowed_uses=spec["allowed"], forbidden_uses=spec["forbidden"],
        personal_data="none (station ids, layer and county codes; model-grid cell centroids "
                      "in the IC file; no well coordinates, no contact fields)")
    errs = manifest.validate_manifest(m, d) + guard.time_slicing_errors(m, d)
    if errs:
        raise SystemExit("; ".join(errs))
    manifest.write_manifest(d, m)
    print(d, spec["passed"], json.dumps(stats["by_flag"], default=str))
    return d


if __name__ == "__main__":
    package(sys.argv[1], sys.argv[2])
