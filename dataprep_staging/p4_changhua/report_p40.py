"""WP P4.0 report: reads the engine product's validation.json and the RIV diagnostic, writes
``results/dataprep/reports/P40.{md,json}`` and the WP's ledger rows.

    cd dataprep_staging
    tools/cpu_slot.sh python -m p4_changhua.report_p40 results/dataprep/changhua_abstraction/engine/v1

Not part of the product's code: it only reads the product (whose manifest is re-hashed here)
and the evidence files, and it changes no product file.
"""
from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import check, ledger, manifest  # noqa: E402
from common.paths import LEDGER_ROWS, REPORTS_DIR, from_repo, rel_to_repo  # noqa: E402

THETA = "results/twin_runs/stage3_datum_gate_cwa/stage3_theta.json"
RIV = "results/twin_runs/temporal_rivers/stage3_theta.json"
GRID = "results/twin_forward/datum_gate_innov.npz"
BASEMAP = "results/twin/basemap.npz"
COUNTIES = "chou-shui-data/data/water/Taiwan_countyV02/TW_COUNTY.gpkg"
POLYGON = "chou-shui-data/data/Zhuoshui Alluvial Fan/Zhuoshui Alluvial Fan.json"
STATIONS = "AMP_V2/data/fan_stations.parquet"
CENSUS = ["AMP_V2/data/tpc_pumps.parquet", "AMP_V2/data/pump_kwh_all.parquet"]
HEADS = "AMP_V2/data/wells"
RAIN = ["chou-shui-data/data/rf_timeseries.csv", "chou-shui-data/data/rf_stations.csv",
        "chou-shui-data/data/gw_stations.csv", "results/et/openmeteo_et0_2012_2022.npz"]
RIVSHP = ["chou-shui-data/data/water/river_TWD97.shp", "chou-shui-data/data/water/river_TWD97.dbf",
          "chou-shui-data/data/water/river_TWD97.shx"]
PILOT_IN = "results/dataprep/_pilots/physics_inversion_hooks/inputs_cwa.npz"
PREREG = "results/dataprep/preregistration/P40.engine_acceptance.v1.json"
EVID = REPORTS_DIR / "P40_evidence"
CODE_DIR = "dataprep_staging/p4_changhua"
CODE = [f"{CODE_DIR}/{f}" for f in ("engine.py", "build_engine.py", "inputs_p4.py", "__init__.py",
                                    "tests/test_engine.py", "prereg_p40.py", "diag_riv.py",
                                    "looks_p40.py", "report_p40.py")]


def fmt(x, p=3):
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, int):
        return str(x)
    if abs(x) != 0 and (abs(x) < 1e-3 or abs(x) >= 1e5):
        return f"{x:.{p - 1}e}"
    return f"{x:.{p}g}"


# ----------------------------------------------------------------------------- ledger
def _src(*groups):
    out = []
    for g in groups:
        out += g
    return out


def build_rows(pv: str) -> list[dict]:
    """Ledger rows: one per (number, source), with the source's role for that number."""
    rows = []

    def add(nid, sources, note=""):
        seen = set()
        for src, role in sources:
            if src in seen:
                continue
            seen.add(src)
            rows.append({"number_id": f"P40.{nid}", "source": src, "role": role,
                         "product_version": pv, "note": note})

    model = [(THETA, "input"), (GRID, "input"), (BASEMAP, "input"), (POLYGON, "input"),
             (STATIONS, "input")]
    forcing = _src(model, [(HEADS, "input")], [(c, "input") for c in CENSUS],
                   [(r, "input") for r in RAIN])
    county = [(COUNTIES, "input")]
    insample = _src(model, [(HEADS, "fit_target")], [(c, "input") for c in CENSUS],
                    [(r, "input") for r in RAIN])
    riv = _src(forcing, [(RIV, "input")], [(s, "input") for s in RIVSHP])
    for k in ("base", "extra"):
        add(f"A1_{k}_max_dh", _src(forcing, county if k == "extra" else []))
    add("A1_riv_max_dh", riv)
    add("riv_diag_torch_one_step_error", riv,
        "torch per-step CG error from its own state, RIV and base configurations")
    add("riv_diag_tight_cg", riv, "sparse vs torch with the reference CG tolerance 1e-12")
    for k in ("changhua_uniform", "single_cells_worst", "random_field_131",
              "frozen_lift_energy"):
        add(f"A2_{k}", _src(forcing, county))
    add("operator_check", model)
    add("kernel_tail", model)
    add("quasi_steady_share", _src(model, county, [(HEADS, "input")]))
    add("pump_weights", [(THETA, "input")])
    add("lift_feedback", _src(forcing, county))
    add("datum_replica", insample)
    add("r2_insample", insample)
    add("inputs_identical", _src(forcing, county, [(PILOT_IN, "validator")]))
    add("cells_wells_by_county", _src(model, county, [(HEADS, "input")]))
    add("forcing_variance", _src(forcing, county))
    add("bound_pinned", [(THETA, "input")])
    add("level_slices", insample)
    add("fisher_interior", insample)
    add("theta_stationarity", insample, "gradient and Gauss-Newton refit of the level loss")
    add("pinned_profiles", insample)
    add("kernel_bytes", model)
    add("timing", forcing)
    return rows


# ----------------------------------------------------------------------------- content
def acceptance_rows(V: dict, rd: dict, extra: dict) -> list[dict]:
    a1, a2 = V["A1"]["cases"], V["A2"]
    acc = [
        {"criterion": "A1 sparse rollout vs torch rollout (CPU, CUDA hidden, 2 threads), "
                      "deliverable forcing: the pre-registered acceptance case",
         "threshold": "max |dh| <= 1e-5 m (all layers, cells, 131 steps)",
         "measured": f"{fmt(a1['base']['max_abs_dh_m'])} m (rms {fmt(a1['base']['rms_dh_m'])})",
         "passed": a1["base"]["max_abs_dh_m"] <= 1e-5},
        {"criterion": "A1 reported case: forcing + synthetic Changhua abstraction "
                      "(188 GWh/yr), lift feedback on the added energy",
         "threshold": "max |dh| <= 1e-5 m", "measured": f"{fmt(a1['extra']['max_abs_dh_m'])} m",
         "passed": a1["extra"]["max_abs_dh_m"] <= 1e-5},
        {"criterion": "A1 reported case: MODFLOW-RIV switching + canal recharge "
                      "(verification configuration, not calibrated)",
         "threshold": "max |dh| <= 1e-5 m", "measured": f"{fmt(a1['riv']['max_abs_dh_m'])} m",
         "passed": a1["riv"]["max_abs_dh_m"] <= 1e-5}]
    for lab, key in (("float64", "float64"), ("float32 stored", "float32_stored")):
        for case in ("changhua_uniform", "single_cells_worst", "random_field_131",
                     "frozen_lift_energy"):
            v = a2[key][case]["scale_rel"]
            acc.append({"criterion": f"A2 adjoint kernels vs forward rollout: {case} "
                                     f"({lab})",
                        "threshold": "max |K - h_fwd| / max |h_fwd| <= 1e-6",
                        "measured": fmt(v), "passed": v <= 1e-6})
    d = V["datum"]
    rb, rr = rd["base"]["cg_tol_1e-12"], rd["riv"]["cg_tol_1e-12"]
    acc += [
        {"criterion": "[diagnostic, post hoc] RIV case with the reference CG tolerance "
                      "tightened to 1e-12 in-process (base case in brackets)",
         "threshold": "explains the RIV shortfall; not a registered case",
         "measured": f"{fmt(rr['max_abs_dh_m'])} m ({fmt(rb['max_abs_dh_m'])} m); torch "
                     f"per-step error at its default 1e-8: "
                     f"{fmt(rd['riv']['cg_tol_1e-08']['torch_one_step_error_max_m'])} m; "
                     f"switch disagreements {rd['riv']['cg_tol_1e-08']['mask_disagreements']}",
         "passed": rr["max_abs_dh_m"] <= 1e-5},
        {"criterion": "[engine check] sparse operator vs FlowModel matvec",
         "threshold": "<= 1e-12 relative (not a plan criterion)",
         "measured": fmt(V["operator"]["rel_max_matvec"]),
         "passed": V["operator"]["rel_max_matvec"] <= 1e-12},
        {"criterion": "[engine check] datum prior replica vs "
                      "calibrate_flow._profile_well_datum",
         "threshold": "<= 1e-9 m", "measured": f"{fmt(d['max_abs_engine_vs_torch_m'])} m",
         "passed": d["max_abs_engine_vs_torch_m"] <= 1e-9},
        {"criterion": "[engine check] in-sample R2 (2012-22) reproduces the theta's "
                      "recorded value",
         "threshold": "|dR2| <= 1e-4",
         "measured": f"{d['r2_insample']:.6f} vs {d['r2_insample_meta']:.6f}",
         "passed": abs(d["r2_insample"] - d["r2_insample_meta"]) <= 1e-4},
        {"criterion": "[engine check] inputs identical to the scout's load_twin_inputs output",
         "threshold": "all arrays identical",
         "measured": f"{sum(V['inputs']['pilot_inputs_identical'].values())}/"
                     f"{len(V['inputs']['pilot_inputs_identical'])}",
         "passed": all(V["inputs"]["pilot_inputs_identical"].values())},
        {"criterion": "[engine check] bound-pinned entries match the theta's bounds_hit",
         "threshold": "identical counts", "measured": "7 entries; counts identical",
         "passed": bool(V["bound_pinned"]["counts_match_meta"])},
        {"criterion": "[engine check] synthetic unit tests (p4_changhua/tests) and ruff",
         "threshold": "all pass", "measured": extra["unit_tests"],
         "passed": extra["unit_tests_passed"]},
        {"criterion": "[engine check] product manifest re-hash (common.check)",
         "threshold": "status ok, 0 problems",
         "measured": f"{extra['manifest_check']['status']}, "
                     f"{extra['manifest_check']['n_hashed']} records, "
                     f"{len(extra['manifest_check']['hash_problems'])} problems",
         "passed": extra["manifest_check"]["status"] == "ok"}]
    return acc


def key_numbers(V: dict, rd: dict) -> list[str]:
    a1, a2 = V["A1"]["cases"], V["A2"]
    op, k, d, fv = V["operator"], V["kernels"], V["datum"], V["forcing_variance"]
    lf, bp, tm = V["lift_feedback_diagnostic"], V["bound_pinned"], V["timing_s"]
    w64 = max(v["scale_rel"] for v in a2["float64"].values() if isinstance(v, dict)
              and "scale_rel" in v)
    w32 = max(v["scale_rel"] for v in a2["float32_stored"].values() if isinstance(v, dict)
              and "scale_rel" in v)
    r8, r12 = rd["riv"]["cg_tol_1e-08"], rd["riv"]["cg_tol_1e-12"]
    fi = bp["fisher_interior"]
    pr = bp["profiles"]
    return [
        f"A1 sparse vs torch max |dh|: base {fmt(a1['base']['max_abs_dh_m'])} m (acceptance), "
        f"extra {fmt(a1['extra']['max_abs_dh_m'])} m, riv {fmt(a1['riv']['max_abs_dh_m'])} m "
        f"(threshold 1e-5 m; riv FAILS)",
        f"RIV diagnostic: torch's own per-step CG error {fmt(r8['torch_one_step_error_max_m'])} m"
        f" max (median {fmt(r8['torch_one_step_error_median_m'])}); with the reference CG "
        f"tightened to 1e-12: riv {fmt(r12['max_abs_dh_m'])} m, base "
        f"{fmt(rd['base']['cg_tol_1e-12']['max_abs_dh_m'])} m; 0 switch disagreements",
        f"A2 adjoint vs forward, worst case: float64 {fmt(w64)}, stored float32 {fmt(w32)} "
        f"(threshold 1e-6)",
        f"Kernels: 158 wells x 2148 cells x {k['n_lags']} lags x 4 layers + pump kernel, "
        f"float32, 5 x 244,356,608 bytes; tail beyond {k['n_lags']} lags "
        f"{fmt(k['tail_beyond_n_lags']['max_rel_to_max_gain'])} of the max steady gain",
        f"Pump weights per m3/day abstracted: layer 0 {k['pump_weights']['0']:+.4f} "
        f"(return flow {op['scalars']['return_frac']:.4f} minus shallow share "
        f"{op['scalars']['f_shallow']:.4f}), layer 1 {k['pump_weights']['1']:+.4f}",
        f"Quasi-steady share of the Changhua steady gain (top decile pairs): lag0 "
        f"{k['changhua_quasi_steady_share_top_decile']['lag0']:.3f}, lag2 "
        f"{k['changhua_quasi_steady_share_top_decile']['lag2']:.3f}, lag11 "
        f"{k['changhua_quasi_steady_share_top_decile']['lag11']:.3f}",
        f"Operator: {op['unknowns']} unknowns, nnz {op['nnz']}, LU fill {op['lu_fill_nnz']}, "
        f"cond {op['cond']:.3e}, matvec rel {fmt(op['rel_max_matvec'])}, symmetric",
        f"Timing (1-2 threads): LU {tm['lu']} s, sparse rollout {tm['sparse_rollout_131']} s "
        f"vs torch {tm['torch_rollout_131']} s, Green's functions {tm['greens_all_layers']} s, "
        f"convolution of all wells {tm['convolve_all_wells_131']} s, build {tm['total']} s",
        f"Lift feedback, synthetic Changhua 188 GWh/yr: Changhua wells mean "
        f"{lf['changhua_wells_mean_dh_step_m']:.2f} m stepwise vs "
        f"{lf['changhua_wells_mean_dh_frozen_m']:.2f} m frozen lift; max feedback "
        f"{lf['max_abs_feedback_m']:.2f} m; feedback {lf['feedback_share_of_rms_response']:.3f}"
        f" of the rms response",
        f"Var h = K Sigma K^T: independent noise MC/formula median "
        f"{fv['independent_noise']['median_ratio_mc_over_formula']:.4f} "
        f"({fv['independent_noise']['share_in_band']:.3f} in the chi2 95% band); energy errors "
        f"with lift feedback, nonlinear MC/linear median "
        f"{fv['lift_feedback_mc']['median_ratio_nonlinear_mc_over_linear']:.3f} (p05-p95 "
        f"{fv['lift_feedback_mc']['p05_p95_ratio'][0]:.2f}-"
        f"{fv['lift_feedback_mc']['p05_p95_ratio'][1]:.2f}); factor responses exact "
        f"({fmt(max(fv['factor_responses_scale_rel']))})",
        f"Datum prior N(0, 5 m^2), 12 months/obs: replica vs fit_flow {fmt(d['max_abs_engine_vs_torch_m'])}"
        f" m; sigma_within {d['sigma_within_m']:.3f} m; shrinkage {d['shrink_min']:.3f}-"
        f"{d['shrink_median']:.3f} (min-median); free datum differs by max "
        f"{d['free_datum_max_abs_diff_m']:.3f} m (median {d['free_datum_median_abs_diff_m']:.3f})",
        f"In-sample R2 2012-22 {d['r2_insample']:.5f} (theta meta {d['r2_insample_meta']:.5f});"
        f" with datum {d['r2_with_datum']:.5f}",
        "Bound-pinned (7, = bounds_hit): log_T_distal hi in layers 0, 2, 3; log_L lo in mid[0], "
        "distal[0], distal[2]; log_spread_km at the 0.5 km floor",
        f"theta-hat is not stationary for its own level loss: Gauss-Newton refit within the "
        f"bounds lowers it {bp['base_loss']:.4f} -> {pr['reference_refit_loss']:.4f} "
        f"({pr['reference_refit_dloss'] / bp['base_loss'] * 100:.1f} %); cond(J^T J) "
        f"{fi['cond_JTJ']:.3e} over {bp['n_interior']} interior entries",
        f"Inputs: {V['inputs']['n_active']} cells (Changhua "
        f"{V['inputs']['cells_by_county']['彰化縣']}, Yunlin "
        f"{V['inputs']['cells_by_county']['雲林縣']}, Nantou "
        f"{V['inputs']['cells_by_county']['南投縣']}, Chiayi "
        f"{V['inputs']['cells_by_county']['嘉義縣']}), {V['inputs']['n_wells']} wells "
        f"(Changhua {V['inputs']['wells_by_county']['彰化縣']}, Yunlin "
        f"{V['inputs']['wells_by_county']['雲林縣']}), 131 steps 2012-02..2022-12; 14/14 arrays "
        f"identical to the scout's inputs",
        f"theta: {THETA} sha256 {V['theta']['sha256']} (provisional)",
        "Claude API tokens: 0; GPU: none"]


def pinned_table(V: dict) -> list[str]:
    bp = V["bound_pinned"]
    prof = {(p["name"], p["index"]): p for p in bp["profiles"]["rows"]}
    lines = ["| entry | side | slice dloss x N/s2 at outward -0.5 / +0.5 | profile at -0.5 / "
             "+0.5 | reading |", "|---|---|---|---|---|"]
    for s in bp["slices"]:
        r = {row["offset_outward"]: row["dloss_x_N_over_s2"] for row in s["rows"]}
        p = {row["offset_outward"]: row["dprofile_x_N_over_s2"]
             for row in prof[(s["name"], s["index"])]["rows"]}
        binding = p[0.5] < 0 and r[0.5] < 0
        reading = ("bound binding (the loss falls beyond it)" if binding else
                   "loss falls inward: not pressed against the bound" if p[-0.5] < 0 and
                   r[-0.5] < 0 else "mixed (slice and profile disagree): ridge-like")
        lines.append(f"| {s['name']}[{s['index']}] | {s['side']} | {r[-0.5]:.1f} / "
                     f"{r[0.5]:.1f} | {p[-0.5]:.1f} / {p[0.5]:.1f} | {reading} |")
    return lines


DEVIATIONS = [
    "Built on the current cwa theta (stage3_datum_gate_cwa/stage3_theta.json, sha256 "
    "789d0e45..., written 2026-10-11 19:06 UTC), not on batch 8's merged theta, which does not "
    "exist yet (task instruction). The plan's acceptance 'on the merged theta' is therefore "
    "evaluated on a provisional theta; rebuild as engine/v2 once the merged theta exists.",
    "Code lives in dataprep_staging/p4_changhua/ (decision D12) instead of "
    "results/dataprep/_staging_code/.",
    "The RIV verification configuration is the cwa theta plus the temporal_rivers recipe's "
    "river conductances and a synthetic canal recharge (fraction 0.25); the deliverable has no "
    "rivers, so no calibrated RIV model exists to verify on.",
    "Kernels hold 180 lags (2012-01..2026-12, the whole record) instead of 131, so each "
    "float32 kernel is 244 MB rather than the plan's 178 MB; the Green's functions took 49 s "
    "for all 4 layers against the plan's 166 s estimate.",
    "Bound-pinned profiles are approximate: chord Gauss-Newton with a fixed finite-difference "
    "Jacobian at theta-hat, 4 iterations, trust box 0.5 per iteration (interior shifts often "
    "reach the 1.5 trust limit), on the level loss with the datum profiled; no torch "
    "re-optimisation. The plan asks for profiling or widened-bound sensitivities; widened "
    "bounds were evaluated only as slices and profiles up to 1 log unit beyond each bound.",
    "A scratch trial of build_engine.py (24 lags, 4 Monte Carlo draws) ran at 06:29Z after "
    "the pre-registration (06:23Z) and before the product build (06:40Z); build_engine.py "
    "was edited between them (diff not recoverable; A1, the impulse A2 cases, the datum and "
    "the Jacobian are identical between the runs, only the profile-refit diagnostic differs). "
    "Both runs and the later RIV diagnostic are logged in the A1 look log (logged late).",
    "The RIV diagnostic (diag_riv.py) and its tightened-tolerance comparison are post hoc, "
    "run after the product to explain the RIV shortfall; they change no threshold, case or "
    "product file. The reference's CG tolerance was changed only inside that process "
    "(flow._cg.__defaults__); no repository file was edited.",
    "The engine forcing is the twin's deliverable forcing (Yunlin census AMP_V2 "
    "tpc_pumps/pump_kwh_all, identical to the new TPC L0 Yunlin sheet per the task update); "
    "the Changhua census is not an input of P4.0 (C0/P4m). The 188 GWh/yr synthetic Changhua "
    "abstraction used in the tests is uniform over Changhua's cells with Yunlin's monthly "
    "shape, a verification load only.",
    "report_p40.py, diag_riv.py and looks_p40.py were added or rewritten after the v1 build; "
    "they are not in v1's manifest code list because they only read the product.",
]

ISSUES = [
    "Provisional theta: engine/v1 replicates the cwa theta of 2026-10-11 (sha256 "
    "789d0e45...). When batch 8's merged theta is written, rebuild engine/v2 with build_engine "
    "(about 9 min in 2 CPU slots) and rerun anything built on v1. The checker will flag v1's "
    "inputs if stage3_theta.json is rewritten in place.",
    "The convolution engine (stored kernels) is the frozen-lift linearisation. For a large "
    "added abstraction it overstates drawdown: for the synthetic 188 GWh/yr Changhua load the "
    "Changhua wells' mean drawdown is 9.42 m at frozen lift against 7.65 m with lift feedback "
    "(max difference 6.69 m; feedback 26 % of the rms response). P4m/C0, which add the Changhua "
    "census to a model calibrated without it, must use SparseTwin.rollout (E_extra, stepwise) "
    "or linearise about the path that already includes the added abstraction "
    "(energy_rate_jacobian at that path); the kernels alone are valid for perturbations "
    "around a given path.",
    "Var h = K Sigma_q K^T is exact for volumetric (m3/day) abstraction errors (Monte Carlo "
    "median ratio 0.9985). For energy (kWh) errors with lift feedback the linearised formula "
    "overstates the nonlinear variance: median ratio 0.695 (p05-p95 0.46-0.94), outside the "
    "chi2 band 0.81-1.21. Treat it as conservative for energy-forcing uncertainty, or use the "
    "stepwise Monte Carlo (about 46 s per 200 draws).",
    "RIV mode is verified to 2.7e-5 m against torch at torch's default CG tolerance (the "
    "registered threshold 1e-5 m is missed), and to 2.7e-9 m with torch's CG tightened to "
    "1e-12, with zero switch disagreements: the shortfall is the reference solver's per-step "
    "CG error (2.75e-5 m from its own state), not the sparse engine. Quote it that way; the "
    "deliverable twin has no rivers. The engine refuses delay-bed and aquitard-store models "
    "and per-class efficiencies (eta_classes); a recalibration ladder that adds them needs an "
    "engine extension.",
    "theta-hat is not a stationary point of its own fit_flow level loss (500 Adam epochs; "
    "all 35 entries were free in the calibration): residual-sum-of-squares gradients reach "
    "1.0e3 m^2 per log unit (log_T_proximal[0]), and a bounded Gauss-Newton refit lowers the "
    "in-sample loss by 2.8 % (3.388 -> 3.293) with interior shifts at the trust limit. The "
    "Fisher sd at theta-hat (stored in validation.json) is local curvature at a "
    "non-stationary point, iid-month and uncalibrated (phi about 0.74 shrinks the effective "
    "sample about 6.7-fold); do not quote it as an uncertainty. The refit is in-sample only "
    "and says nothing about out-of-sample skill.",
    "Bound-pinned entries (profiled, never differentiated at the bound, per plan): the "
    "leakance floors log_L_mid[0], log_L_distal[0] and the spread floor are binding (the loss "
    "keeps falling beyond them); log_L_distal[2] is mixed; the distal log T upper bounds in "
    "layers 2 and 3 are not binding (lower T fits better at fixed others and in the profile), "
    "and layer 0 is mixed (slice favours lower T, profile favours higher). Distal T is "
    "ridge-like; treat all seven as non-identified and widen bounds only in sensitivities.",
    "Datum convention: the engine's fit_flow datum prior N(0, 5 m^2) with 12 months per "
    "independent observation replicates calibrate_flow to 7e-15 m; datum shrinkage is "
    "0.965-0.99, so the datum is effectively free (the free datum differs by at most 0.38 m). "
    "Mean (level) information about abstraction is therefore absorbed by the datum, as the "
    "plan's P4 formulation states.",
    "Kernel files: 5 x 244 MB float32 under changhua_abstraction/engine/v1 (G_layer0-3, "
    "K_pump); load with np.load(mmap_mode='r'). K_pump is drawdown (m, positive down) per "
    "m3/day abstracted for one monthly step, combining the pump split and return flow of the "
    "theta; G_layer<l> is head rise per m3/day injected in layer l. engine_arrays.npz carries "
    "the operator pieces, hindcast heads, the base abstraction rate Q_rate (cell aggregates), "
    "datum, steady gains and the county of every cell and well. Allowed uses: model_design, "
    "selection, design_time_evidence, after_the_fact_analysis.",
    "The quasi-steady share of the Changhua steady gain is 0.66 at lag 0 and 0.86 by lag 2 "
    "for the strongest well-cell pairs: Changhua heads respond within one to three months, so "
    "monthly forcing timing errors (billing pair split, July mask) map almost directly into "
    "head errors (P2, P4m).",
    "72 of the 2148 cells fall outside every county polygon and were assigned to the nearest "
    "one (cell_county in engine_arrays.npz).",
]


def write_md(out: dict, V: dict, rd: dict) -> str:
    L = []
    L.append("# WP P4.0: linear-response engine (report, 2026-10-12)\n")
    L.append(f"Status: **{out['status']}** (provisional theta). {out['summary']}\n")
    L.append("## Acceptance\n")
    L.append("Pre-registration: `results/dataprep/preregistration/P40.engine_acceptance.v1.json` "
             "(declaration, 2026-10-12T06:23:42Z, before any acceptance number was computed). "
             "It names the deliverable-forcing case as the A1 acceptance case and reports the "
             "two other A1 cases against the same threshold; A2 must pass in float64 and again "
             "for the stored float32 kernels. Rows marked [diagnostic] or [engine check] are "
             "not plan criteria.\n")
    L.append("| criterion | threshold | measured | pass |")
    L.append("|---|---|---|---|")
    for a in out["acceptance"]:
        L.append(f"| {a['criterion']} | {a['threshold']} | {a['measured']} | "
                 f"{'yes' if a['passed'] else '**no**'} |")
    L.append("")
    L.append("## Product\n")
    for p in out["products"]:
        L.append(f"- `{p['path']}` ({p['layer']}); manifest `{p['manifest']}`; allowed uses: "
                 f"{p['allowed_uses']}; fit_window_end {p['fit_window_end']}.")
    L.append("- Files: `G_layer0-3_f32.npy` (158 wells x 2148 cells x 180 lags, head per m3/day "
             "injected in that layer), `K_pump_f32.npy` (drawdown per m3/day abstracted), "
             "`M_operator.npz` (scipy.sparse backward-Euler operator), `engine_arrays.npz` "
             "(operator pieces, hindcast heads 2012-01..2022-12, base abstraction rate, datum, "
             "steady gains, cell/well counties), `validation.json` (every number below).")
    L.append("")
    L.append("## What the engine is\n")
    L.append("`dataprep_staging/p4_changhua/engine.py` ports the physics scout's sparse-LU "
             "replica (`_pilots/physics_inversion_hooks`: b_operator, c_greens, inv_lib) and "
             "adds what P4.0 asks for:\n")
    L.append("- `assemble_operator` builds the backward-Euler operator M = D + K + leakage + GHB "
             "from the repository's own `forward.build_model` (read-only), with cell-masked "
             "T/S/leakance multipliers (`region_multiplier`), checked against "
             "`FlowModel._matvec_from` (relative 3.1e-16, exactly symmetric).")
    L.append("- `SparseTwin`: LU factorisation and a stepwise rollout with lift feedback "
             "(energy converted with the previous step's pump-layer head), added abstraction "
             "with the same feedback (`E_extra`), direct sources, canal recharge, and "
             "MODFLOW-RIV switching (one cached factorisation per connection state); "
             "`verify_against_torch` runs `forward.rollout` on CPU for comparison.")
    L.append("- `adjoint_greens`: exact Green's functions by one adjoint rollout per well (M "
             "symmetric), all 4 layers; `pump_kernel`, `steady_gain` (truncation tail), "
             "`forward_impulse`, and `convolve` (per-well FFT; valid at fixed lift without RIV "
             "switching).")
    L.append("- Datum: `profile_datum` and `fit_flow_level_loss` replicate fit_flow's per-well "
             "datum prior; `solve_with_datum` and `fisher_with_datum` use the same convention "
             "in linear inversions (free datum = sd inf).")
    L.append("- Forcing uncertainty (plan 4.8): `head_var_independent`, `factor_responses` + "
             "`head_var_factors`, `separable_factors`, `energy_rate_jacobian`.")
    L.append("- Bound-pinned parameters: `bound_pinned`, `theta_vector`, `perturb_theta`; "
             "slices, Fisher at interior entries and chord Gauss-Newton profiles in "
             "`build_engine.py`.\n")
    L.append("## RIV shortfall and its diagnosis\n")
    r8b, r12b = rd["base"]["cg_tol_1e-08"], rd["base"]["cg_tol_1e-12"]
    r8r, r12r = rd["riv"]["cg_tol_1e-08"], rd["riv"]["cg_tol_1e-12"]
    L.append("The RIV case misses the reported threshold (2.71e-5 m against 1e-5 m). "
             "`diag_riv.py` (post hoc) measured torch's own per-step solve error, restarting "
             "the exact sparse one-step solve from torch's previous head, and repeated the "
             "comparison with torch's CG tolerance tightened in-process from 1e-8 to 1e-12:\n")
    L.append("| configuration | torch CG tol | sparse vs torch max abs dh (m) | torch per-step "
             "error max / median (m) | switch disagreements |")
    L.append("|---|---|---|---|---|")
    for lab, a, tol in (("base", r8b, "1e-8 (default)"), ("base", r12b, "1e-12"),
                        ("riv", r8r, "1e-8 (default)"), ("riv", r12r, "1e-12")):
        L.append(f"| {lab} | {tol} | {fmt(a['max_abs_dh_m'])} | "
                 f"{fmt(a['torch_one_step_error_max_m'])} / "
                 f"{fmt(a['torch_one_step_error_median_m'])} | "
                 f"{a.get('mask_disagreements', 'n/a')} |")
    L.append("\nThe difference is the reference solver's CG truncation, which is larger with "
             "the river conductances on the diagonal; the sparse direct solve agrees with a "
             "tightly converged torch rollout to 2.7e-9 m. The registered result stays a fail "
             "for the RIV case.\n")
    L.append("## Diagnostics (not gates)\n")
    lf = V["lift_feedback_diagnostic"]
    fv = V["forcing_variance"]
    L.append(f"- Lift feedback (synthetic Changhua {lf['case'].split()[-2]} GWh/yr): mean "
             f"drawdown change at Changhua wells {lf['changhua_wells_mean_dh_step_m']:.2f} m "
             f"stepwise vs {lf['changhua_wells_mean_dh_frozen_m']:.2f} m at frozen lift; max "
             f"feedback {lf['max_abs_feedback_m']:.2f} m; feedback is "
             f"{lf['feedback_share_of_rms_response']:.3f} of the rms response.")
    L.append(f"- Var h = K Sigma K^T, independent cell-month volumetric noise (sd 0.3 x base "
             f"rate): MC/formula median {fv['independent_noise']['median_ratio_mc_over_formula']:.4f}, "
             f"p05-p95 {fv['independent_noise']['p05_p95_ratio'][0]:.3f}-"
             f"{fv['independent_noise']['p05_p95_ratio'][1]:.3f}, "
             f"{fv['independent_noise']['share_in_band']:.3f} inside the chi2 95 % band "
             f"{fv['independent_noise']['chi2_95_band'][0]:.3f}-"
             f"{fv['independent_noise']['chi2_95_band'][1]:.3f} (200 draws).")
    L.append(f"- Energy errors through the full stepwise rollout (39 factors: 10 % zone-year "
             f"levels and zone annual cycles): nonlinear MC / linear formula median "
             f"{fv['lift_feedback_mc']['median_ratio_nonlinear_mc_over_linear']:.3f}, p05-p95 "
             f"{fv['lift_feedback_mc']['p05_p95_ratio'][0]:.3f}-"
             f"{fv['lift_feedback_mc']['p05_p95_ratio'][1]:.3f}: the frozen-lift formula "
             f"overstates the variance when lift feedback acts.")
    d = V["datum"]
    L.append(f"- Datum: replica vs fit_flow {fmt(d['max_abs_engine_vs_torch_m'])} m; vs the "
             f"theta's stored datum {fmt(d['max_abs_engine_vs_theta_meta_m'])} m; sigma_within "
             f"{d['sigma_within_m']:.4f} m; shrinkage min {d['shrink_min']:.3f}, median "
             f"{d['shrink_median']:.3f}; R2 in sample {d['r2_insample']:.5f} (with datum "
             f"{d['r2_with_datum']:.5f}); free datum max diff {d['free_datum_max_abs_diff_m']:.3f}"
             f" m.")
    k = V["kernels"]
    L.append(f"- Kernel truncation: the tail beyond 131 lags is "
             f"{fmt(k['tail_beyond_131']['max_rel_to_max_gain'])} and beyond 180 lags "
             f"{fmt(k['tail_beyond_n_lags']['max_rel_to_max_gain'])} of the largest steady gain. "
             f"Quasi-steady share of the Changhua gain (top decile pairs): " +
             ", ".join(f"{kk} {vv:.3f}" for kk, vv in
                       k["changhua_quasi_steady_share_top_decile"].items()) + ".")
    tm = V["timing_s"]
    L.append(f"- Timing: LU {tm['lu']} s; sparse rollout {tm['sparse_rollout_131']} s against "
             f"torch {tm['torch_rollout_131']} s (2 threads); Green's functions "
             f"{tm['greens_all_layers']} s; convolution of all wells {tm['convolve_all_wells_131']}"
             f" s; whole build {tm['total']} s.\n")
    L.append("### Bound-pinned entries\n")
    bp = V["bound_pinned"]
    L.append(f"The seven entries at a bound equal the theta's `bounds_hit`. Loss changes are "
             f"scaled as N/s2 x dloss (N = 20,698 well-months, s2 = "
             f"{V['datum']['sigma_within_m'] ** 2:.3f} m^2): iid chi2 units, overconfident "
             f"about 6.7-fold for autocorrelated months. Slices hold every other entry; "
             f"profiles re-optimise the 28 interior entries (approximate, see deviations) and "
             f"are measured from the refit at theta-hat (loss "
             f"{bp['profiles']['reference_refit_loss']:.4f}, against {bp['base_loss']:.4f} at "
             f"theta-hat). Outward = beyond the bound.\n")
    L += pinned_table(V)
    L.append("")
    L.append("## Deviations from plan\n")
    L += [f"{i}. {x}" for i, x in enumerate(out["deviations_from_plan"], 1)]
    L.append("")
    L.append("## Issues for downstream\n")
    L += [f"- {x}" for x in out["issues_for_downstream"]]
    L.append("")
    L.append("## Provenance\n")
    L.append(f"- Code: {', '.join('`' + c + '`' for c in out['code_paths'])}.")
    L.append("- Evidence: " + ", ".join(f"`{e}`" for e in out["evidence"]) + ".")
    L.append(f"- Ledger: `{out['ledger']['fragment']}` ({out['ledger']['rows']} rows; "
             f"unledgered numbers {len(out['ledger']['unledgered'])}; conflicts "
             f"{out['ledger']['conflicts']}).")
    L.append("- Look log: 7 looks recorded late for this WP (trial run, product build, RIV "
             "diagnostic; `p4_changhua/looks_p40.py`).")
    L.append("- No GPU (CUDA hidden, torch on CPU), 0 Claude API tokens, nothing written under "
             "`hydrophysics/`, `tests/` or L0, nothing staged in git. Census columns read: "
             + ", ".join(V["inputs"]["census_columns_read"]) + " (never 戶名1 or 用電地址2).")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    vdir = from_repo(argv[0])
    m, _ = manifest.load_manifest(vdir)
    mpath = vdir / "MANIFEST.json"
    V = json.loads((vdir / "validation.json").read_text())
    rd = json.loads((EVID / "riv_diagnostic.json").read_text())
    if rd["theta_sha256"] != V["theta"]["sha256"]:
        raise SystemExit("riv diagnostic ran on a different theta")
    mc = check.check_manifest(mpath)
    ut = (EVID / "unit_tests_rerun.txt").read_text().strip().splitlines()
    extra = {"unit_tests": "; ".join(x.strip(". ") for x in ut if x.strip(". ")),
             "unit_tests_passed": any("passed" in x for x in ut)
             and not any("failed" in x or "error" in x.lower() for x in ut)
             and any("All checks passed" in x for x in ut),
             "manifest_check": mc}
    (EVID / "manifest_check_resume.json").write_text(json.dumps(mc, indent=1, default=str)
                                                     + "\n")
    pv = f"{m['product']}/{m['version']}"
    rows = build_rows(pv)
    ledger.append_rows("P40", rows)
    all_rows = ledger.read_all()
    unl = ledger.unledgered({r["number_id"] for r in rows})
    n_conf = len(ledger.conflicts(all_rows))
    acc = acceptance_rows(V, rd, extra)
    gate = bool(V["A1"]["passed_base"] and V["A2"]["passed"])
    status = "passed" if gate else "failed"
    summary = (
        "P4.0 ports the physics scout's sparse-LU replica of the calibrated twin and adds a "
        "stepwise rollout with lift feedback, MODFLOW-RIV switching, the fit_flow datum prior, "
        "Var h = K Sigma_q K^T propagation and bound-pinned profiling "
        "(dataprep_staging/p4_changhua/engine.py). It was built on the cwa theta of "
        "2026-10-11 (sha256 789d0e45..., provisional: batch 8's merged theta does not exist "
        "yet). Against the repository's torch rollout on CPU the sparse rollout matches to "
        f"{fmt(V['A1']['cases']['base']['max_abs_dh_m'])} m on the deliverable forcing (the "
        "pre-registered acceptance case, threshold 1e-5 m) and to "
        f"{fmt(V['A1']['cases']['extra']['max_abs_dh_m'])} m with a synthetic 188 GWh/yr "
        "Changhua abstraction and lift feedback. The third reported case, a non-calibrated "
        f"RIV configuration, misses the threshold at {fmt(V['A1']['cases']['riv']['max_abs_dh_m'])}"
        " m. A post-hoc diagnostic traces this to the reference solver's CG tolerance: torch's "
        "own per-step error is 2.75e-5 m, and with its CG tightened to 1e-12 the difference "
        "falls to 2.7e-9 m with no switch disagreements. Adjoint Green's functions for all 4 "
        "layers (158 wells x 2148 cells x 180 lags) match forward impulses, the convolution of "
        "a random 131-month field and the frozen-lift energy pathway to <= 2.7e-15 relative "
        "in float64 and <= 4.9e-8 for the stored float32 kernels (threshold 1e-6). Kernels "
        "are stored float32 under results/dataprep/changhua_abstraction/engine/v1 (manifest "
        "re-hashes ok). Diagnostics for downstream: the frozen-lift kernels overstate the "
        "drawdown of a large added abstraction (Changhua wells 9.42 m vs 7.65 m with "
        "feedback), the linearised Var h overstates energy-error variance (ratio 0.69), and "
        "theta-hat is not stationary for its own level loss (a bounded refit lowers it "
        "2.8 %). No GPU, no Claude API, nothing under hydrophysics/ or tests/ edited.")
    out = {
        "wp": "P40", "status": status, "date": _dt.date.today().isoformat(),
        "summary": summary,
        "gate": {"pre_registered": "A1 acceptance case (deliverable forcing) and A2 "
                                   "(float64 and stored float32)",
                 "passed": gate, "A1_all_reported_cases_passed": V["A1"]["passed_all"],
                 "failed_reported_case": "A1 riv: "
                                         f"{fmt(V['A1']['cases']['riv']['max_abs_dh_m'])} m "
                                         "> 1e-5 m"},
        "acceptance": acc,
        "products": [{"path": rel_to_repo(vdir), "layer": m["layer"],
                      "manifest": rel_to_repo(mpath),
                      "allowed_uses": ", ".join(m["allowed_uses"]),
                      "fit_window_end": m["fit_window_end"]}],
        "code_paths": CODE,
        "key_numbers": key_numbers(V, rd),
        "deviations_from_plan": DEVIATIONS,
        "issues_for_downstream": ISSUES,
        "preregistration": rel_to_repo(from_repo(PREREG)),
        "evidence": [rel_to_repo(vdir / "validation.json"),
                     rel_to_repo(EVID / "build_engine.log"),
                     rel_to_repo(EVID / "riv_diagnostic.json"),
                     rel_to_repo(EVID / "riv_diagnostic.log"),
                     rel_to_repo(EVID / "unit_tests_rerun.txt"),
                     rel_to_repo(EVID / "manifest_check_resume.json"),
                     rel_to_repo(EVID / "manifest_check.txt")],
        "ledger": {"fragment": rel_to_repo(LEDGER_ROWS / "P40.csv"), "rows": len(rows) + 1,
                   "unledgered": unl, "conflicts": n_conf},
        "theta": V["theta"],
        "riv_diagnostic": rd,
        "claude_api_tokens": 0, "gpu": "none",
        "manifest_errors": manifest.validate_manifest(m, vdir),
    }
    md = write_md(out, V, rd)
    (REPORTS_DIR / "P40.json").write_text(json.dumps(out, ensure_ascii=False, indent=1,
                                                     default=float) + "\n")
    (REPORTS_DIR / "P40.md").write_text(md)
    print(json.dumps({"status": status, "unledgered": unl, "conflicts": n_conf,
                      "acceptance": [(a["criterion"][:60], a["passed"]) for a in acc]},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
