"""WP P4.0 report after the repair round: reads engine/v2 (and v1), writes
``results/dataprep/reports/P40.{md,json}`` (the first-round report is kept as ``P40.r1.*``)
and appends the repair round's ledger rows.

    cd dataprep_staging
    tools/cpu_slot.sh python -m p4_changhua.report_p40_repair \
        results/dataprep/changhua_abstraction/engine/v2

Not part of the product's code: it only reads the products (whose manifests are re-hashed
here) and the evidence files, and it changes no product file.
"""
from __future__ import annotations

import datetime as _dt
import json
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import check, ledger, manifest  # noqa: E402
from common.paths import LEDGER_ROWS, REPORTS_DIR, from_repo, rel_to_repo  # noqa: E402

from p4_changhua import report_p40 as r1  # noqa: E402
from p4_changhua import report_p40_repair_text  # noqa: E402
from p4_changhua.report_p40 import fmt  # noqa: E402

EVID = REPORTS_DIR / "P40_evidence"
REP = EVID / "repair"
CODE_DIR = "dataprep_staging/p4_changhua"
CODE = [f"{CODE_DIR}/{f}" for f in (
    "engine.py", "build_engine.py", "inputs_p4.py", "__init__.py", "engine_tlm.py",
    "profiles_v2.py", "build_engine_v2.py", "repair_theta_status.py", "diag_riv.py",
    "tests/test_engine.py", "tests/test_tlm.py", "prereg_p40.py", "prereg_p40_repair.py",
    "looks_p40.py", "looks_p40_repair.py", "probe_fresh_blocking.py", "report_p40.py",
    "report_p40_repair.py", "report_p40_repair_text.py")]
PREREG_A = "results/dataprep/preregistration/P40.engine_acceptance.v1.json"
PREREG_R = "results/dataprep/preregistration/P40.repair_linearisation.v1.json"
FIT_SUMMARY = "results/twin_runs/stage3_datum_gate_cwa/stage3_fit_summary.json"
FOLD_THETAS = "results/twin_runs/stage3_datum_gate_cwa/stage3_fold_thetas.json"
JOBS = "results/twin_runs/par8/jobs.sh"
QLOG = "results/twin_runs/gap_queue.log"
MERGE_MARK = "results/twin_runs/par8/done/merge"
V1 = "results/dataprep/changhua_abstraction/engine/v1"


def _at(tree, path: str):
    """Value at a ``_numeric_diff`` path such as ``/operator/lambda[1]``."""
    import re

    x = tree
    for part in path.strip("/").split("/"):
        m = re.fullmatch(r"(.*?)((?:\[\d+\])*)", part)
        if m.group(1):
            x = x[m.group(1)]
        for i in re.findall(r"\[(\d+)\]", m.group(2)):
            x = x[int(i)]
    return float(x)


# ----------------------------------------------------------------------------- ledger
def repair_rows(pv: str) -> list[dict]:
    rows = []

    def add(nid, sources, note=""):
        seen = set()
        for src, role in sources:
            if src in seen:
                continue
            seen.add(src)
            rows.append({"number_id": f"P40.{nid}", "source": src, "role": role,
                         "product_version": pv, "note": note})

    model = [(r1.THETA, "input"), (r1.GRID, "input"), (r1.BASEMAP, "input"),
             (r1.POLYGON, "input"), (r1.STATIONS, "input")]
    forcing = model + [(r1.HEADS, "input")] + [(c, "input") for c in r1.CENSUS] + [
        (r, "input") for r in r1.RAIN]
    county = [(r1.COUNTIES, "input")]
    insample = model + [(r1.HEADS, "fit_target")] + [(c, "input") for c in r1.CENSUS] + [
        (r, "input") for r in r1.RAIN]
    add("theta_status", [(r1.THETA, "input"), (FIT_SUMMARY, "input"), (FOLD_THETAS, "input"),
                         (JOBS, "input"), (QLOG, "input"), (MERGE_MARK, "input")],
        "theta is batch 8's merged cwa theta (repair round)")
    add("identity_with_v1", forcing + county + [(V1, "validator")],
        "engine/v2 outputs and numbers against engine/v1")
    add("R1_tlm_vs_fd", forcing + county, "tangent-linear model vs central differences")
    add("R2_energy_variance", forcing + county, "tangent-linear vs stepwise MC, energy errors")
    add("R3_volumetric_variance", forcing + county,
        "stepwise vs tangent-linear MC, volumetric errors with base lift feedback")
    add("lift_clip", forcing + county)
    add("profiles_v2", insample, "reference refits and local profiles, in sample")
    add("slices_v2", insample)
    add("fisher_reference", insample)
    add("residual_autocorrelation", insample, "lag-1 autocorrelation of in-sample residuals")
    add("fresh_unit_blocking_probe",
        [("results/dataprep/preregistration/looks.jsonl", "input"),
         ("results/dataprep/validation_units/v3/units.json", "input")],
        "read-only prereg3 probe of the look log (probe_fresh_blocking.py)")
    return rows


# ----------------------------------------------------------------------------- tables
def reading(sl: dict, pr: dict) -> str:
    if pr[0.5] < 0 and sl[0.5] < 0:
        return "binding (loss falls beyond the bound)"
    if pr[-0.5] < 0 and sl[-0.5] < 0:
        return "not binding (loss falls inward)"
    return "mixed (slice and profile disagree)"


def pinned_rows(P: dict) -> list[dict]:
    out = []
    for s in P["slices"]:
        key = (s["name"], s["index"])
        rec = {"entry": f"{s['name']}[{s['index']}]", "side": s["side"]}
        for which in ("filled", "observed"):
            sl = {r["offset_outward"]: r[f"dloss_x_N_over_s2_{which}"] for r in s["rows"]}
            pp = next(p for p in P[which]["profiles"] if (p["name"], p["index"]) == key)
            pr = {r["offset_outward"]: r["dprofile_x_N_over_s2"] for r in pp["rows"]}
            rec[which] = {"slice": sl, "profile": pr, "reading": reading(sl, pr),
                          "stop_reasons": [r["stop_reason"] for r in pp["rows"]
                                           if r["offset_outward"] != 0.0],
                          "max_remaining": max(r["predicted_remaining_relative_decrease"]
                                               for r in pp["rows"])}
        out.append(rec)
    return out


def pinned_table(rows: list[dict]) -> list[str]:
    L = ["| entry | side | target | slice at -0.5 / +0.5 | profile at -0.5 / +0.5 | "
         "reading |", "|---|---|---|---|---|---|"]
    for r in rows:
        for which in ("filled", "observed"):
            x = r[which]
            L.append(f"| {r['entry']} | {r['side']} | {which} | {x['slice'][-0.5]:.1f} / "
                     f"{x['slice'][0.5]:.1f} | {x['profile'][-0.5]:.1f} / "
                     f"{x['profile'][0.5]:.1f} | {x['reading']} |")
    return L


# ----------------------------------------------------------------------------- content
def acceptance(V: dict, extra: dict) -> list[dict]:
    a1, a2, R = V["A1"]["cases"], V["A2"], V["repair"]
    r1_, r2_, r3_ = (R["R1_tlm_vs_finite_difference"], R["R2_energy_error_variance"],
                     R["R3_volumetric_noise_variance"])
    ts, idn = V["theta"]["status_evidence"], R["identity_with_v1"]
    d = V["datum"]
    w = {k: [a2[k][c]["scale_rel"] for c in ("changhua_uniform", "single_cells_worst",
                                             "random_field_131", "frozen_lift_energy")]
         for k in ("float64", "float32_stored")}
    acc = [
        {"criterion": "A1 sparse rollout vs torch rollout (CPU, CUDA hidden, 2 threads), "
                      "deliverable forcing: the pre-registered acceptance case (engine/v2 build)",
         "threshold": "max |dh| <= 1e-5 m over all layers, cells and 131 steps 2012-02..2022-12",
         "measured": f"{fmt(a1['base']['max_abs_dh_m'])} m (rms {fmt(a1['base']['rms_dh_m'])})",
         "passed": a1["base"]["max_abs_dh_m"] <= 1e-5},
        {"criterion": "A1 reported case: forcing + synthetic Changhua abstraction (188 GWh/yr), "
                      "lift feedback on the added energy",
         "threshold": "max |dh| <= 1e-5 m", "measured": f"{fmt(a1['extra']['max_abs_dh_m'])} m",
         "passed": a1["extra"]["max_abs_dh_m"] <= 1e-5},
        {"criterion": "A1 reported case: MODFLOW-RIV switching + synthetic canal recharge "
                      "(verification configuration, not calibrated)",
         "threshold": "max |dh| <= 1e-5 m",
         "measured": f"{fmt(a1['riv']['max_abs_dh_m'])} m (fails; see the RIV diagnostic in "
                     "key numbers, which is post hoc and not an acceptance row)",
         "passed": a1["riv"]["max_abs_dh_m"] <= 1e-5},
        {"criterion": "A2 adjoint kernels vs forward rollout, float64: changhua_uniform / "
                      "single_cells_worst / random_field_131 / frozen_lift_energy",
         "threshold": "max |K - h_fwd| / max |h_fwd| <= 1e-6 per case",
         "measured": " / ".join(fmt(x) for x in w["float64"]),
         "passed": all(x <= 1e-6 for x in w["float64"])},
        {"criterion": "A2 adjoint kernels vs forward rollout, stored float32: changhua_uniform / "
                      "single_cells_worst / random_field_131 / frozen_lift_energy",
         "threshold": "max |K - h_fwd| / max |h_fwd| <= 1e-6 per case",
         "measured": " / ".join(fmt(x) for x in w["float32_stored"]),
         "passed": all(x <= 1e-6 for x in w["float32_stored"])},
        {"criterion": "[repair declaration R1] tangent-linear model (lift feedback included) vs "
                      "central differences of the stepwise twin: energy / volumetric direction",
         "threshold": "max |dh_TLM - dh_FD| / max |dh_FD| <= 1e-6 each",
         "measured": f"{fmt(r1_['energy']['tlm_vs_fd_rel'])} / "
                     f"{fmt(r1_['volumetric']['tlm_vs_fd_rel'])} (frozen-lift response: "
                     f"{fmt(r1_['energy']['frozen_vs_fd_rel'])} / "
                     f"{fmt(r1_['volumetric']['frozen_vs_fd_rel'])})",
         "passed": bool(r1_["passed"])},
        {"criterion": "[repair declaration R2] energy-error head variance: fresh stepwise Monte "
                      "Carlo (200 draws) / tangent-linear J Sigma J^T, median over well-months",
         "threshold": f"inside the chi2 95 % band [{r2_['chi2_95_band'][0]:.4f}, "
                      f"{r2_['chi2_95_band'][1]:.4f}]",
         "measured": f"{r2_['mc_over_tlm']['median']:.3f} (p05-p95 "
                     f"{r2_['mc_over_tlm']['p05_p95'][0]:.2f}-"
                     f"{r2_['mc_over_tlm']['p05_p95'][1]:.2f}); frozen-lift K Sigma K^T: "
                     f"{r2_['mc_over_frozen_formula']['median']:.3f}",
         "passed": bool(r2_["passed"])},
        {"criterion": "[repair declaration R3] independent volumetric noise with the base "
                      "pumping's lift feedback: stepwise / tangent-linear variance, same 200 "
                      "draws, median over well-months",
         "threshold": "inside [0.98, 1.02]",
         "measured": f"{r3_['step_over_tlm_mc_paired']['median']:.5f}; frozen-lift formula: "
                     f"{r3_['step_over_frozen_formula']['median']:.3f} (sum ratio "
                     f"{r3_['step_over_frozen_formula']['sum_ratio']:.3f})",
         "passed": bool(r3_["passed"])},
        {"criterion": "[repair check] the theta is batch 8's merged cwa theta (verifier major "
                      "issue)",
         "threshold": "merge job (deps fit_full + folds 0-4) wrote it before the v1 build; "
                      "5-fold gate in meta; parameters equal the full fit; sha256 unchanged",
         "measured": f"written {ts['theta']['mtime_utc'][:19]}Z, merge marker "
                     f"{ts['batch8']['done_markers_utc']['merge'][:19]}Z, batch end "
                     f"{ts['batch8']['done_markers_utc']['ring'][:19]}Z; n_folds "
                     f"{ts['theta']['meta_gate']['n_folds']}; parameter diff "
                     f"{ts['theta']['params_max_abs_diff_vs_full_fit_summary']}; sha256 "
                     f"{ts['theta']['sha256'][:8]}... = v1's",
         "passed": bool(ts["is_batch8_merged_theta"])},
        {"criterion": "[repair check] engine/v2 reproduces engine/v1 (same theta, inputs and "
                      "kernel code): stored files",
         "threshold": "kernels and operator byte-identical; every array of engine_arrays.npz "
                      "identical",
         "measured": f"bytes {idn['kernels_and_operator_byte_identical']}, arrays "
                     f"{idn['engine_arrays_identical']}",
         "passed": bool(idn["kernels_and_operator_byte_identical"]
                        and idn["engine_arrays_identical"])},
        {"criterion": "[repair check] engine/v2 reproduces engine/v1: every number of "
                      "validation.json's core block",
         "threshold": "identical (timings and the theta status excluded)",
         "measured": (f"{len(idn['validation_numbers_differing'])} differ: "
                      + ", ".join(f"{x['path']} (relative {fmt(x['rel_diff'])})"
                                  for x in extra["identity_rel"])
                      + "; all are estimates of scipy's iterative eigensolver (eigsh, random "
                        "start vector), not engine outputs"
                      if idn["validation_numbers_differing"] else "0 differ"),
         "passed": bool(idn["validation_numbers_identical"])},
        {"criterion": "[engine check] sparse operator vs FlowModel matvec",
         "threshold": "<= 1e-12 relative", "measured": fmt(V["operator"]["rel_max_matvec"]),
         "passed": V["operator"]["rel_max_matvec"] <= 1e-12},
        {"criterion": "[engine check] datum prior replica vs calibrate_flow._profile_well_datum",
         "threshold": "<= 1e-9 m", "measured": f"{fmt(d['max_abs_engine_vs_torch_m'])} m",
         "passed": d["max_abs_engine_vs_torch_m"] <= 1e-9},
        {"criterion": "[engine check] in-sample R2 2012-22 reproduces the theta's recorded value",
         "threshold": "|dR2| <= 1e-4",
         "measured": f"{d['r2_insample']:.6f} vs {d['r2_insample_meta']:.6f}",
         "passed": abs(d["r2_insample"] - d["r2_insample_meta"]) <= 1e-4},
        {"criterion": "[engine check] inputs identical to the scout's load_twin_inputs output",
         "threshold": "all arrays identical",
         "measured": f"{sum(V['inputs']['pilot_inputs_identical'].values())}/"
                     f"{len(V['inputs']['pilot_inputs_identical'])}",
         "passed": all(V["inputs"]["pilot_inputs_identical"].values())},
        {"criterion": "[engine check] bound-pinned entries match the theta's bounds_hit",
         "threshold": "identical", "measured": f"{len(V['bound_pinned']['pinned'])} entries, "
                                               "identical",
         "passed": bool(V["bound_pinned"]["counts_match_meta"])},
        {"criterion": "[engine check] unit tests (p4_changhua/tests, synthetic) and ruff",
         "threshold": "all pass", "measured": extra["unit_tests"],
         "passed": extra["unit_tests_passed"]},
        {"criterion": "[engine check] manifest re-hash (common.check): engine/v2, and v1 kept "
                      "unchanged",
         "threshold": "status ok, 0 problems, both",
         "measured": f"v2 {extra['check_v2']['status']} ({extra['check_v2']['n_hashed']} "
                     f"records, {len(extra['check_v2']['hash_problems'])} problems); v1 "
                     f"{extra['check_v1']['status']} ({extra['check_v1']['n_hashed']} records, "
                     f"{len(extra['check_v1']['hash_problems'])} problems)",
         "passed": extra["check_v2"]["status"] == "ok" and extra["check_v1"]["status"] == "ok"},
    ]
    return acc


def key_numbers(V: dict, prow: list[dict], extra: dict) -> list[str]:
    R = V["repair"]
    a1, a2 = V["A1"]["cases"], V["A2"]
    rd = R["riv_diagnostic"]
    r8, r12 = rd["riv"]["cg_tol_1e-08"], rd["riv"]["cg_tol_1e-12"]
    r1_, r2_, r3_ = (R["R1_tlm_vs_finite_difference"], R["R2_energy_error_variance"],
                     R["R3_volumetric_noise_variance"])
    k, op, d, lf, tm = (V["kernels"], V["operator"], V["datum"],
                        V["lift_feedback_diagnostic"], V["timing_s"])
    w64 = max(v["scale_rel"] for v in a2["float64"].values() if isinstance(v, dict)
              and "scale_rel" in v)
    w32 = max(v["scale_rel"] for v in a2["float32_stored"].values() if isinstance(v, dict)
              and "scale_rel" in v)
    P = R["profiles_v2"]
    ts = V["theta"]["status_evidence"]
    out = [
        f"theta: results/twin_runs/stage3_datum_gate_cwa/stage3_theta.json sha256 "
        f"{V['theta']['sha256']} = batch 8's merged cwa theta (written "
        f"{ts['theta']['mtime_utc'][:19]}Z by merge_folds; batch 8 finished "
        f"{ts['batch8']['done_markers_utc']['ring'][:19]}Z, "
        f"{ts['timeline_hours']['batch_end_to_v1_build']} h before the v1 build)",
        f"A1 sparse vs torch max |dh|: base {fmt(a1['base']['max_abs_dh_m'])} m (acceptance), "
        f"extra {fmt(a1['extra']['max_abs_dh_m'])} m, riv {fmt(a1['riv']['max_abs_dh_m'])} m "
        "(threshold 1e-5 m; riv FAILS)",
        f"RIV diagnostic (post hoc, rerun inside the v2 build): torch's own per-step CG error "
        f"{fmt(r8['torch_one_step_error_max_m'])} m; reference CG tightened to 1e-12: riv "
        f"{fmt(r12['max_abs_dh_m'])} m, base {fmt(rd['base']['cg_tol_1e-12']['max_abs_dh_m'])} "
        f"m; {r8['mask_disagreements']} switch disagreements",
        f"A2 adjoint vs forward, worst case: float64 {fmt(w64)}, stored float32 {fmt(w32)} "
        "(threshold 1e-6)",
        f"engine/v2 vs v1: kernels and operator byte-identical "
        f"{R['identity_with_v1']['kernels_and_operator_byte_identical']}, arrays identical "
        f"{R['identity_with_v1']['engine_arrays_identical']}; validation numbers differing "
        f"{len(extra['identity_rel'])}"
        + ("" if not extra["identity_rel"] else " (" + ", ".join(
            f"{x['path']} rel {fmt(x['rel_diff'])}" for x in extra["identity_rel"])
           + "; eigsh estimates)"),
        f"R1 tangent-linear vs central difference: energy {fmt(r1_['energy']['tlm_vs_fd_rel'])}, "
        f"volumetric {fmt(r1_['volumetric']['tlm_vs_fd_rel'])} (frozen-lift "
        f"{r1_['energy']['frozen_vs_fd_rel']:.3f} / {r1_['volumetric']['frozen_vs_fd_rel']:.3f});"
        f" lift clipped in {R['lift_clip']['base_path']['n_clipped']} of "
        f"{R['lift_clip']['base_path']['n']} base cell-steps",
        f"R2 energy errors (39 factors): stepwise MC / TLM median "
        f"{r2_['mc_over_tlm']['median']:.3f}; MC / frozen-lift K Sigma K^T "
        f"{r2_['mc_over_frozen_formula']['median']:.3f}; TLM / frozen median "
        f"{r2_['tlm_over_frozen_formula']['median']:.3f} (sum "
        f"{r2_['tlm_over_frozen_formula']['sum_ratio']:.3f}); per-draw nonlinear share "
        f"{r2_['per_draw_nonlinear_rms_share']:.3f}",
        f"R3 volumetric noise (0.3 x Q_base) with base lift feedback: stepwise / TLM paired "
        f"median {r3_['step_over_tlm_mc_paired']['median']:.5f}; stepwise / frozen formula "
        f"median {r3_['step_over_frozen_formula']['median']:.3f} (sum "
        f"{r3_['step_over_frozen_formula']['sum_ratio']:.3f}); TLM / frozen paired median "
        f"{r3_['tlm_mc_over_frozen_mc_paired']['median']:.3f} (p05-p95 "
        f"{r3_['tlm_mc_over_frozen_mc_paired']['p05_p95'][0]:.2f}-"
        f"{r3_['tlm_mc_over_frozen_mc_paired']['p05_p95'][1]:.2f})",
        f"Kernels: 158 wells x 2148 cells x {k['n_lags']} lags x 4 layers + pump kernel, "
        f"float32, 5 x 244,356,608 bytes; tail beyond {k['n_lags']} lags "
        f"{fmt(k['tail_beyond_n_lags']['max_rel_to_max_gain'])} of the max steady gain",
        f"Operator: {op['unknowns']} unknowns, nnz {op['nnz']}, cond {op['cond']:.3e}, matvec "
        f"rel {fmt(op['rel_max_matvec'])}, symmetric",
        f"Lift feedback, synthetic Changhua 188 GWh/yr: Changhua wells mean "
        f"{lf['changhua_wells_mean_dh_step_m']:.2f} m stepwise vs "
        f"{lf['changhua_wells_mean_dh_frozen_m']:.2f} m frozen lift; max feedback "
        f"{lf['max_abs_feedback_m']:.2f} m",
        f"Datum prior N(0, 5 m^2), 12 months/obs: replica vs fit_flow "
        f"{fmt(d['max_abs_engine_vs_torch_m'])} m; shrinkage {d['shrink_min']:.3f}-"
        f"{d['shrink_median']:.3f}; free datum differs by max "
        f"{d['free_datum_max_abs_diff_m']:.3f} m",
        f"In-sample R2 2012-22 {d['r2_insample']:.5f} (theta meta {d['r2_insample_meta']:.5f}); "
        f"with datum {d['r2_with_datum']:.5f}",
        f"Residual lag-1 autocorrelation (in sample, observed months, theta-hat) "
        f"{P['residual_autocorrelation_theta_hat']['phi']:.3f}: iid Fisher sd too small by "
        f"about {P['residual_autocorrelation_theta_hat']['sd_factor_ar1']:.2f}x (variance "
        f"{P['residual_autocorrelation_theta_hat']['variance_factor_ar1']:.2f}x)",
    ]
    for which in ("filled", "observed"):
        ref = P[which]["reference"]
        out.append(
            f"Reference refit of the 28 interior entries ({which} targets): objective "
            f"{ref['objective_mse_theta_hat']:.4f} -> {ref['objective_mse']:.4f} "
            f"({ref['relative_decrease_vs_theta_hat'] * 100:.1f} %), stop "
            f"'{ref['stop_reason']}' after {ref['n_iter']} iterations, predicted remaining "
            f"decrease {fmt(ref['predicted_remaining_relative_decrease'])}")
    out.append("Bound-pinned readings (slice + profile; filled / observed): " + "; ".join(
        f"{r['entry']} {r['filled']['reading'].split(' (')[0]} / "
        f"{r['observed']['reading'].split(' (')[0]}" for r in prow))
    out.append(f"Timing: core build {tm['total']} s; repair block {R['t_total_s']} s in all; "
               f"profiles v2 {P['t_s']} s ({P['n_model_evaluations']} model evaluations)")
    pb = extra["probe"]["units"]
    n_blk = sum(1 for v in pb.values() if v["blocking_p40"] > 0)
    out.append(f"A1 look log under harness.prereg3: P40 looks eligible to block a test registered "
               f"now on a fresh unit: {max(v['blocking_p40'] for v in pb.values())} looks on "
               f"{n_blk} of {len(pb)} fresh units (before this round's looks: "
               f"{max(v['blocking_p40'] for v in extra['probe_before']['units'].values())}; "
               "all other WPs' looks there: "
               f"{max(v['blocking_all_wps'] - v['blocking_p40'] for v in pb.values())} blocking, "
               f"up to {max(v['unverifiable_all_wps'] for v in pb.values())} unverifiable)")
    out.append("Claude API tokens: 0; GPU: none")
    return out


# ----------------------------------------------------------------------------- main
def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    vdir = from_repo(argv[0])
    m, _ = manifest.load_manifest(vdir)
    V = json.loads((vdir / "validation.json").read_text())
    R = V["repair"]
    for f in ("P40.md", "P40.json"):        # keep the first-round report once
        src, dst = REPORTS_DIR / f, REPORTS_DIR / f.replace("P40.", "P40.r1.")
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
    c2 = check.check_manifest(vdir / "MANIFEST.json")
    c1 = check.check_manifest(from_repo(V1) / "MANIFEST.json")
    (REP / "manifest_check.json").write_text(json.dumps({"v2": c2, "v1": c1}, indent=1,
                                                        default=str) + "\n")
    ut = (REP / "unit_tests.txt").read_text().strip().splitlines()
    V1j = json.loads((from_repo(V1) / "validation.json").read_text())
    extra = {"unit_tests": "; ".join(x.strip(". ") for x in ut if x.strip(". ")),
             "unit_tests_passed": any("passed" in x for x in ut)
             and not any("failed" in x or "error" in x.lower() for x in ut)
             and any("All checks passed" in x for x in ut),
             "check_v1": c1, "check_v2": c2,
             "identity_rel": [{"path": x["path"], "abs_diff": x["abs_diff"],
                               "rel_diff": x["abs_diff"] / abs(_at(V1j, x["path"]))}
                              for x in R["identity_with_v1"]["validation_numbers_differing"]],
             "probe": json.loads((REP / "fresh_blocking_probe_at_report.json").read_text()),
             "probe_before": json.loads(
                 (REP / "fresh_blocking_probe_before_looks.json").read_text())}
    pv = f"{m['product']}/{m['version']}"
    rows = r1.build_rows(pv) + repair_rows(pv)
    ledger.append_rows("P40", rows)
    all_rows = ledger.read_all()
    unl = ledger.unledgered({r["number_id"] for r in rows})
    n_conf = len(ledger.conflicts(all_rows))
    n_frag = len(ledger._read(ledger.fragment_path("P40")))
    acc = acceptance(V, extra)
    prow = pinned_rows(R["profiles_v2"])
    text = report_p40_repair_text.texts(V, prow, extra)
    gate = bool(V["A1"]["passed_base"] and V["A2"]["passed"])
    out = {
        "wp": "P40", "status": "passed" if gate else "failed",
        "date": _dt.date.today().isoformat(), "round": "repair",
        "summary": text["summary"],
        "gate": {"pre_registered": "A1 acceptance case (deliverable forcing) and A2 (float64 "
                                   "and stored float32), P40.engine_acceptance.v1",
                 "passed": gate, "A1_all_reported_cases_passed": V["A1"]["passed_all"],
                 "failed_reported_case": f"A1 riv: {fmt(V['A1']['cases']['riv']['max_abs_dh_m'])}"
                                         " m > 1e-5 m",
                 "repair_declaration": R["acceptance"]},
        "acceptance": acc,
        "products": [
            {"path": rel_to_repo(vdir) + "/", "layer": m["layer"],
             "manifest": rel_to_repo(vdir / "MANIFEST.json") + " (re-hashes "
                         f"{c2['status']}, {c2['n_hashed']} records). Files: G_layer0-3_f32.npy, "
                         "K_pump_f32.npy, M_operator.npz, engine_arrays.npz (byte- and "
                         "value-identical to v1), validation.json (core + repair block)",
             "allowed_uses": ", ".join(m["allowed_uses"]) + " (current version; built on batch "
                             "8's merged cwa theta)",
             "fit_window_end": m["fit_window_end"]},
            {"path": V1 + "/", "layer": "L3",
             "manifest": f"{V1}/MANIFEST.json (kept unchanged, re-hashes {c1['status']}); "
                         "SUPERSEDED by v2: identical content, but its manifest and "
                         "validation.json wrongly call the theta provisional",
             "allowed_uses": "as v2; prefer v2", "fit_window_end": "2022-12"}],
        "code_paths": [str(from_repo(c)) for c in CODE],
        "key_numbers": key_numbers(V, prow, extra),
        "deviations_from_plan": text["deviations"],
        "issues_for_downstream": text["issues"],
        "repair_round": text["repair_round"],
        "report_path": str(REPORTS_DIR / "P40.md"),
        "preregistrations": [PREREG_A, PREREG_R],
        "evidence": [rel_to_repo(vdir / "validation.json"), rel_to_repo(REP / "runs.json"),
                     rel_to_repo(REP / "build_engine_v2.log"),
                     rel_to_repo(REP / "build_engine_core.log"),
                     rel_to_repo(REP / "trial_build_engine_v2.log"),
                     rel_to_repo(REP / "unit_tests.txt"),
                     rel_to_repo(REP / "manifest_check.json"),
                     rel_to_repo(REP / "build_result_v2.json"),
                     rel_to_repo(REP / "cutoff_build_engine_v2.stdout"),
                     rel_to_repo(REP / "cutoff_build_engine_core.log"),
                     rel_to_repo(REP / "cutoff_core_validation.json"),
                     rel_to_repo(REP / "fresh_blocking_probe_before_looks.json"),
                     rel_to_repo(REP / "fresh_blocking_probe_after_looks.json"),
                     rel_to_repo(REP / "fresh_blocking_probe_at_report.json")],
        "ledger": {"fragment": rel_to_repo(LEDGER_ROWS / "P40.csv"), "rows": n_frag,
                   "unledgered": unl, "conflicts": n_conf},
        "pinned_readings": prow,
        "claude_api_tokens": 0, "gpu": "none",
        "manifest_errors": manifest.validate_manifest(m, vdir),
    }
    md = write_md(out, V, prow, text)
    (REPORTS_DIR / "P40.json").write_text(json.dumps(out, ensure_ascii=False, indent=1,
                                                     default=float) + "\n")
    (REPORTS_DIR / "P40.md").write_text(md)
    print(json.dumps({"status": out["status"], "unledgered": unl, "conflicts": n_conf,
                      "rows": n_frag, "acceptance": [(a["criterion"][:70], a["passed"])
                                                     for a in acc]},
                     ensure_ascii=False, indent=1))
    return 0


def write_md(out: dict, V: dict, prow: list[dict], text: dict) -> str:
    R = V["repair"]
    L = ["# WP P4.0: linear-response engine (report, repair round 2026-10-12/13)\n",
         "> **Repair round: read the 'Repair round' section at the end first.** engine/v1 was "
         "built on batch 8's merged cwa theta, not a provisional one; engine/v2 republishes the "
         "identical engine with corrected provenance and the repair diagnostics. The first-"
         "round report is kept as `P40.r1.md` / `P40.r1.json`.\n",
         f"Status: **{out['status']}**. {out['summary']}\n",
         "## Acceptance\n",
         "Pre-registrations: `" + PREREG_A + "` (declaration, 2026-10-12T06:23:42Z) names the "
         "deliverable-forcing case as the A1 acceptance case and reports the extra and RIV "
         "cases against the same threshold; A2 must pass in float64 and for the stored float32 "
         "kernels. `" + PREREG_R + "` (declaration, harness.prereg2, 2026-10-12T15:51:46Z, "
         "before any R1-R3 number was computed on the twin) registers the repair diagnostics "
         "R1-R3; it is not part of the P4.0 gate. Rows marked [repair check] or [engine check] "
         "are not plan criteria. The post-hoc RIV diagnostic is reported below, not as an "
         "acceptance row.\n",
         "| criterion | threshold | measured | pass |", "|---|---|---|---|"]
    for a in out["acceptance"]:
        L.append(f"| {a['criterion']} | {a['threshold']} | {a['measured']} | "
                 f"{'yes' if a['passed'] else '**no**'} |")
    L.append("")
    L.append("## Products\n")
    for p in out["products"]:
        L.append(f"- `{p['path']}` ({p['layer']}): {p['manifest']}. Allowed uses: "
                 f"{p['allowed_uses']}; fit_window_end {p['fit_window_end']}.")
    L.append("")
    L += text["md_sections"]
    L.append("### Bound-pinned entries (engine/v2, profiles v2)\n")
    L.append(text["pinned_intro"] + "\n")
    L += pinned_table(prow)
    L.append("")
    L.append("## Deviations from plan\n")
    L += [f"{i}. {x}" for i, x in enumerate(out["deviations_from_plan"], 1)]
    L.append("")
    L.append("## Issues for downstream\n")
    L += [f"- {x}" for x in out["issues_for_downstream"]]
    L.append("")
    L.append("## Repair round (2026-10-12/13)\n")
    L.append(text["repair_intro"] + "\n")
    L.append("| verifier finding | severity | response | evidence |")
    L.append("|---|---|---|---|")
    for r in out["repair_round"]:
        L.append(f"| {r['finding']} | {r['severity']} | {r['response']} | {r['evidence']} |")
    L.append("")
    L.append("## Provenance\n")
    L.append("- Code: " + ", ".join(f"`{rel_to_repo(Path(c))}`" for c in out["code_paths"])
             + ".")
    L.append("- Evidence: " + ", ".join(f"`{e}`" for e in out["evidence"]) + "; first round: "
             "`results/dataprep/reports/P40_evidence/` (build_engine.log, riv_diagnostic.json, "
             "resume2_check.json).")
    L.append(f"- Ledger: `{out['ledger']['fragment']}` ({out['ledger']['rows']} rows; "
             f"unledgered numbers {len(out['ledger']['unledgered'])}; conflicts "
             f"{out['ledger']['conflicts']}).")
    L.append("- Look log: " + text["looks"])
    L.append("- No GPU (CUDA hidden, torch on CPU, `cpu_slot.sh --slots 2`), 0 Claude API "
             "tokens, nothing written under `hydrophysics/`, `tests/` or L0, nothing staged in "
             "git. Census columns read: " + ", ".join(V["inputs"]["census_columns_read"])
             + " (never 戶名1 or 用電地址2).")
    L.append(f"- Repair block of validation.json: builder {R['builder']} {R['builder_version']}, "
             f"engine_tlm {R['engine_tlm_version']}.")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
