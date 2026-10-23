"""Prose of the P4.0 report after the repair round (numbers formatted from engine/v2's
validation.json, never typed by hand). Used by ``report_p40_repair``."""
from __future__ import annotations

from p4_changhua.report_p40 import fmt


def _pct(x: float) -> str:
    return f"{x * 100:.1f} %"


def texts(V: dict, prow: list[dict], extra: dict) -> dict:
    R = V["repair"]
    a1, a2 = V["A1"]["cases"], V["A2"]
    ts = V["theta"]["status_evidence"]
    dm = ts["batch8"]["done_markers_utc"]
    r1_, r2_, r3_ = (R["R1_tlm_vs_finite_difference"], R["R2_energy_error_variance"],
                     R["R3_volumetric_noise_variance"])
    rd = R["riv_diagnostic"]
    r8b, r12b = rd["base"]["cg_tol_1e-08"], rd["base"]["cg_tol_1e-12"]
    r8r, r12r = rd["riv"]["cg_tol_1e-08"], rd["riv"]["cg_tol_1e-12"]
    idn = R["identity_with_v1"]
    P = R["profiles_v2"]
    ac = P["residual_autocorrelation_theta_hat"]
    reff, refo = P["filled"]["reference"], P["observed"]["reference"]
    lf, d, k, tm = V["lift_feedback_diagnostic"], V["datum"], V["kernels"], V["timing_s"]
    w64 = max(v["scale_rel"] for v in a2["float64"].values() if isinstance(v, dict)
              and "scale_rel" in v)
    w32 = max(v["scale_rel"] for v in a2["float32_stored"].values() if isinstance(v, dict)
              and "scale_rel" in v)
    e_tlm = r2_["mc_over_tlm"]["median"]
    e_frz = r2_["mc_over_frozen_formula"]["median"]
    e_tf = r2_["tlm_over_frozen_formula"]
    v_pair = r3_["step_over_tlm_mc_paired"]["median"]
    v_frz = r3_["step_over_frozen_formula"]
    v_fb = r3_["tlm_mc_over_frozen_mc_paired"]
    rall = R["acceptance"]["all"]
    irel = extra["identity_rel"]
    irel_txt = ("identical" if not irel else
                f"identical except {len(irel)} estimates of the iterative eigensolver "
                f"(operator condition number and extreme eigenvalues; relative difference at "
                f"most {fmt(max(x['rel_diff'] for x in irel))})")
    pb = extra["probe"]["units"]
    n_fresh = len(pb)
    n_blk_units = sum(1 for v in pb.values() if v["blocking_p40"] > 0)
    n_blk_looks = max(v["blocking_p40"] for v in pb.values())
    n_other_blk = max(v["blocking_all_wps"] - v["blocking_p40"] for v in pb.values())
    n_unv = max(v["unverifiable_all_wps"] for v in pb.values())

    agree = [r["entry"] for r in prow if r["filled"]["reading"] == r["observed"]["reading"]]
    stops = {}
    for which in ("filled", "observed"):
        for r in prow:
            for s in r[which]["stop_reasons"]:
                stops[(which, s)] = stops.get((which, s), 0) + 1
    n_prof = sum(stops.values())
    n_conv = sum(v for (w, s), v in stops.items() if s == "converged")
    max_rem = max(max(r["filled"]["max_remaining"], r["observed"]["max_remaining"])
                  for r in prow)

    def lst(x):
        return ", ".join(x) if x else "none"

    s2h = P["s2_theta_hat"]

    def rem_chi2(ref, which):
        return (ref["predicted_remaining_relative_decrease"] * ref["objective_mse"]
                * P["n_rows"][which] / s2h)

    rem_f, rem_o = rem_chi2(reff, "filled"), rem_chi2(refo, "observed")
    ref_conv = reff["stop_reason"] == "converged" and refo["stop_reason"] == "converged"
    prof_abs = [abs(v) for r in prow for w in ("filled", "observed")
                for k_, v in r[w]["profile"].items() if k_ != 0.0]
    med_prof = float(sorted(prof_abs)[len(prof_abs) // 2]) if prof_abs else float("nan")
    ref_txt = (
        "converged" if ref_conv else
        f"stopped at the {reff['n_iter']}-iteration cap ('{reff['stop_reason']}' / "
        f"'{refo['stop_reason']}') with a predicted remaining decrease of "
        f"{fmt(reff['predicted_remaining_relative_decrease'])} / "
        f"{fmt(refo['predicted_remaining_relative_decrease'])} (back-filled / observed), i.e. "
        f"about {rem_f:.0f} / {rem_o:.0f} in the N/s2 units of the profile table, against a "
        f"median |profile difference| of {med_prof:.1f}")

    summary = (
        "Repair round. The first round labelled engine/v1 'provisional' because the task "
        "context said batch 8's merged theta did not exist yet; the verifier was right that it "
        "did. stage3_datum_gate_cwa/stage3_theta.json (sha256 789d0e45..., unchanged) was "
        f"written by batch 8's merge job at {ts['theta']['mtime_utc'][:16]}Z after the full "
        f"fit and all five folds; it carries the 5-fold gate and equals the full fit, and batch "
        f"8 finished at {dm['ring'][:16]}Z, {ts['timeline_hours']['batch_end_to_v1_build']} h "
        "before the v1 build, so the plan's acceptance 'on the merged theta' was met. A new "
        "version, changhua_abstraction/engine/v2, was built with the unchanged v1 kernel code: "
        + ("its kernels, operator and arrays are byte- and value-identical to v1"
           if idn["kernels_and_operator_byte_identical"] and idn["engine_arrays_identical"]
           else "its kernels, operator or arrays DIFFER from v1")
        + " and the validation numbers are "
        f"{irel_txt}; its manifest carries the corrected theta status and a gate text that "
        "names the failed RIV case. The "
        "pre-registered acceptance case matches torch on CPU (CUDA hidden, 2 threads) to "
        f"{fmt(a1['base']['max_abs_dh_m'])} m against 1e-5 m and the synthetic Changhua case "
        f"to {fmt(a1['extra']['max_abs_dh_m'])} m; the RIV verification case still fails at "
        f"{fmt(a1['riv']['max_abs_dh_m'])} m, which a post-hoc diagnostic traces to torch's "
        f"CG truncation ({fmt(r12r['max_abs_dh_m'])} m with its CG tightened). The adjoint "
        f"kernels match forward impulses to {fmt(w64)} (float64) and {fmt(w32)} (stored "
        "float32) relative against 1e-6. The lift feedback is first order, not second order "
        "as engine.py's docstring said: a new tangent-linear model (engine_tlm) matches "
        f"central differences of the stepwise twin to {fmt(r1_['energy']['tlm_vs_fd_rel'])} "
        f"(energy) and {fmt(r1_['volumetric']['tlm_vs_fd_rel'])} (volumetric), and its "
        f"variance matches fresh stepwise Monte Carlo (energy errors median {e_tlm:.3f}; "
        f"volumetric noise, same draws, {v_pair:.4f}), so the registered repair declaration "
        f"{'passes' if rall else 'FAILS'}. The frozen-lift K Sigma K^T overstates the "
        f"head variance of energy errors (MC/formula median {e_frz:.3f}) and, contrary to the "
        "first round's claim, misstates (mostly overstates) that of volumetric errors once the "
        f"base pumping's lift feedback acts (median {v_frz['median']:.3f}, sum ratio "
        f"{v_frz['sum_ratio']:.3f}). The "
        "bound-pinned profiles were rerun for back-filled and observed-only targets with a "
        f"Jacobian refreshed every iteration, but the reference refits {ref_txt}"
        + ("" if ref_conv else ", so the profile readings stay convergence-limited and only "
           "the slices are free of that")
        + f"; the in-sample residual autocorrelation {ac['phi']:.3f} makes the iid Fisher "
        f"sd about {ac['sd_factor_ar1']:.1f}-fold too small (not 6.7-fold, which was a "
        "variance factor). A first v2 build attempt ended with its session before publishing "
        "anything; the second produced v2. Every repair run is in the A1 look log, now through "
        "harness.prereg3 with declared members: the in-sample looks on the 158 calibration "
        f"wells 2012-22 are eligible under prereg3's rule to block later registrations on "
        f"{n_blk_units} of the {n_fresh} A1 fresh units, which A1 must rule on (issues for "
        "downstream). No GPU was used, 0 Claude API tokens were spent, and nothing under "
        "hydrophysics/, tests/ or L0 was written or staged in git.")

    md = []
    md.append("## What the engine is\n")
    md.append("`dataprep_staging/p4_changhua/engine.py` (unchanged since v1) ports the physics "
              "scout's sparse-LU replica (`_pilots/physics_inversion_hooks`: b_operator, "
              "c_greens, inv_lib) and adds what P4.0 asks for: `assemble_operator` (M = D + K + "
              "leakage + GHB from `forward.build_model`, read-only, with cell-masked T/S/"
              f"leakance multipliers; matvec check {fmt(V['operator']['rel_max_matvec'])}, "
              "exactly symmetric); `SparseTwin` (LU, stepwise rollout with lift feedback, "
              "added abstraction with the same feedback, direct sources, canal recharge, "
              "MODFLOW-RIV switching; `verify_against_torch`); `adjoint_greens`, `pump_kernel`, "
              "`steady_gain`, `forward_impulse`, `convolve`; the fit_flow datum prior "
              "(`profile_datum`, `fit_flow_level_loss`, `solve_with_datum`, "
              "`fisher_with_datum`); frozen-lift forcing-variance tools (`head_var_independent`, "
              "`factor_responses`, `head_var_factors`, `separable_factors`, "
              "`energy_rate_jacobian`); bound-pinned helpers (`bound_pinned`, `theta_vector`, "
              "`perturb_theta`).\n")
    md.append("Repair round additions, in new modules because engine/v1's manifest hashes "
              "engine.py and build_engine.py:\n")
    md.append("- `engine_tlm.py`: `rate_factor_dh` (c'(h) = c(h)/lift, 0 where the lift is "
              "clipped at 2 m), `tlm_rollout` (the exact first-order response of "
              "`SparseTwin.rollout` to energy, abstraction-rate or source perturbations, lift "
              "feedback included, batched over columns; `frozen=True` gives the kernels' "
              "frozen-lift response), `tlm_factor_responses` (J Sigma J^T for factor-structured "
              "covariances via `engine.head_var_factors`), `tlm_draw_responses` (Monte Carlo "
              "through the tangent-linear model). Rivers are refused (not differentiable at a "
              "switch). Its docstring records that `engine.energy_rate_jacobian`'s 'second "
              "order' remark is wrong.")
    md.append("- `profiles_v2.py`: bounded Gauss-Newton refits that stop on a relative "
              "tolerance, a stall or an iteration cap (`gn_refit`, "
              "`central_jacobian`, `predicted_remaining`) and `lag1_autocorr`.")
    md.append("- `repair_theta_status.py`: the theta-status evidence; `build_engine_v2.py`: the "
              "v2 build (core = build_engine.main, unchanged; then theta status, identity with "
              "v1, R1-R3, the RIV diagnostic, profiles v2, manifest).\n")
    md.append("## RIV shortfall and its diagnosis (post hoc; rerun inside the v2 build)\n")
    md.append(f"The RIV case misses the reported threshold ({fmt(a1['riv']['max_abs_dh_m'])} m "
              "against 1e-5 m). `diag_riv.run_case` measures torch's own per-step solve error "
              "(the exact sparse one-step solve restarted from torch's previous head) and "
              "repeats the comparison with torch's CG tolerance tightened in-process from 1e-8 "
              "to 1e-12 (no repository file edited):\n")
    md.append("| configuration | torch CG tol | sparse vs torch max abs dh (m) | torch per-step "
              "error max / median (m) | switch disagreements |")
    md.append("|---|---|---|---|---|")
    for lab, a, tol in (("base", r8b, "1e-8 (default)"), ("base", r12b, "1e-12"),
                        ("riv", r8r, "1e-8 (default)"), ("riv", r12r, "1e-12")):
        md.append(f"| {lab} | {tol} | {fmt(a['max_abs_dh_m'])} | "
                  f"{fmt(a['torch_one_step_error_max_m'])} / "
                  f"{fmt(a['torch_one_step_error_median_m'])} | "
                  f"{a.get('mask_disagreements', 'n/a')} |")
    md.append("\nThe miss is the reference solver's CG truncation (larger with the river "
              "conductances on the diagonal); the registered result stays a fail. The numbers "
              "equal the first round's diag_riv run.\n")
    md.append("## Linearisation and forcing-variance diagnostics (repair declaration R1-R3)\n")
    md.append(f"- **R1** (threshold 1e-6): tangent-linear vs central difference of the stepwise "
              f"rollout {fmt(r1_['energy']['tlm_vs_fd_rel'])} for the energy direction (spread "
              f"synthetic Changhua energy, eps {r1_['energy']['eps']:g}) and "
              f"{fmt(r1_['volumetric']['tlm_vs_fd_rel'])} for the volumetric direction "
              f"(eps {r1_['volumetric']['eps']:g}); the frozen-lift response misses by "
              f"{r1_['energy']['frozen_vs_fd_rel']:.3f} and "
              f"{r1_['volumetric']['frozen_vs_fd_rel']:.3f} (relative to the largest response). "
              "The frozen-lift tangent model reproduces the stored float32 kernels' convolution "
              f"to {fmt(r1_['energy']['frozen_tlm_vs_stored_kernel_rel'])}. The lift is "
              f"clipped at 2 m in {R['lift_clip']['base_path']['n_clipped']} of "
              f"{R['lift_clip']['base_path']['n']} base-path cell-steps (min lift "
              f"{R['lift_clip']['base_path']['min_lift_m']:.1f} m).")
    md.append(f"- **R2** (39 energy-error factors, fresh 200-draw stepwise Monte Carlo, seed "
              f"{r2_['seed']}): MC / tangent-linear median {e_tlm:.3f} (p05-p95 "
              f"{r2_['mc_over_tlm']['p05_p95'][0]:.2f}-{r2_['mc_over_tlm']['p05_p95'][1]:.2f}; "
              f"chi2 band {r2_['chi2_95_band'][0]:.3f}-{r2_['chi2_95_band'][1]:.3f}); MC / "
              f"frozen-lift K Sigma K^T {e_frz:.3f} (first round, other seed: 0.695). The "
              f"tangent-linear variance is {e_tf['median']:.3f} of the frozen-lift one (median; "
              f"sum ratio {e_tf['sum_ratio']:.3f}; deterministic, no sampling error). Per draw, "
              f"the stepwise response departs from the tangent-linear one by "
              f"{_pct(r2_['per_draw_nonlinear_rms_share'])} rms for these 10 % energy errors: "
              "the first round's 0.695 was the omitted first-order feedback, not "
              "non-linearity.")
    md.append(f"- **R3** (independent cell-month abstraction-rate noise, sd 0.3 x base rate, "
              f"base path, seed {r3_['seed']}): stepwise / tangent-linear variance from the same "
              f"draws, median {v_pair:.5f} (band 0.98-1.02; per-draw departure "
              f"{_pct(r3_['per_draw_nonlinear_rms_share'])} rms). Against the frozen-lift "
              f"formula on the stored kernels the stepwise variance is {v_frz['median']:.3f} "
              f"(median; sum ratio {v_frz['sum_ratio']:.3f}; p05-p95 "
              f"{v_frz['p05_p95'][0]:.2f}-{v_frz['p05_p95'][1]:.2f}); free of sampling error, "
              f"the feedback alone scales it by {v_fb['median']:.3f} (paired TLM / frozen MC; "
              f"p05-p95 {v_fb['p05_p95'][0]:.2f}-{v_fb['p05_p95'][1]:.2f}, so the feedback "
              "raises the variance at some well-months and lowers it at most). The first "
              "round's 0.9985 compared the formula with Monte Carlo through the same "
              "frozen-lift linear system: an implementation check, not a test against the "
              "twin.\n")
    md.append("## Other diagnostics (not gates; identical to v1)\n")
    md.append(f"- Lift feedback (synthetic Changhua 188 GWh/yr): mean drawdown change at "
              f"Changhua wells {lf['changhua_wells_mean_dh_step_m']:.2f} m stepwise vs "
              f"{lf['changhua_wells_mean_dh_frozen_m']:.2f} m at frozen lift; max feedback "
              f"{lf['max_abs_feedback_m']:.2f} m; feedback {lf['feedback_share_of_rms_response']:.3f}"
              " of the rms response.")
    md.append(f"- Datum: replica vs fit_flow {fmt(d['max_abs_engine_vs_torch_m'])} m; vs the "
              f"theta's stored datum {fmt(d['max_abs_engine_vs_theta_meta_m'])} m; sigma_within "
              f"{d['sigma_within_m']:.4f} m; shrinkage min {d['shrink_min']:.3f}, median "
              f"{d['shrink_median']:.3f}; R2 in sample {d['r2_insample']:.5f} (with datum "
              f"{d['r2_with_datum']:.5f}); free datum max diff {d['free_datum_max_abs_diff_m']:.3f}"
              " m.")
    md.append(f"- Kernels: tail beyond 131 lags {fmt(k['tail_beyond_131']['max_rel_to_max_gain'])}"
              f" and beyond {k['n_lags']} lags {fmt(k['tail_beyond_n_lags']['max_rel_to_max_gain'])}"
              " of the largest steady gain; quasi-steady share of the Changhua gain (top "
              "decile pairs): " + ", ".join(f"{kk} {vv:.3f}" for kk, vv in
                                            k["changhua_quasi_steady_share_top_decile"].items())
              + ".")
    md.append(f"- Timing: LU {tm['lu']} s; sparse rollout {tm['sparse_rollout_131']} s vs torch "
              f"{tm['torch_rollout_131']} s; Green's functions {tm['greens_all_layers']} s; core "
              f"build {tm['total']} s; repair block in all {R['t_total_s']} s.\n")
    md.append("## In-sample level-loss diagnostics (engine/v2, profiles v2)\n")
    for lab, ref in (("back-filled targets (fit_flow's convention)", reff),
                     ("observed months only (plan U2)", refo)):
        md.append(f"- Reference refit of the 28 interior entries, {lab}: objective "
                  f"{ref['objective_mse_theta_hat']:.4f} -> {ref['objective_mse']:.4f} "
                  f"({_pct(ref['relative_decrease_vs_theta_hat'])} lower), stop "
                  f"'{ref['stop_reason']}' after {ref['n_iter']} iterations "
                  f"({ref['n_eval']} model runs), predicted remaining decrease "
                  f"{fmt(ref['predicted_remaining_relative_decrease'])}, last step at the trust "
                  f"box {ref['last_step_hit_trust_box']}; R2 {ref['r2']:.4f}; residual "
                  f"autocorrelation {ref['residual_autocorrelation']['phi']:.3f}.")
    md.append(f"- Profile points (4 offsets x 7 entries x 2 targets = {n_prof}): {n_conv} "
              "stopped on the relative tolerance; the others: " + ", ".join(
                  f"{w} {s} {v}" for (w, s), v in sorted(stops.items()) if s != "converged")
              + f"; largest predicted remaining decrease {fmt(max_rem)}. 'no_decrease' means "
              "the chord step with the reference's Jacobian could not lower the objective "
              "within four halvings: the refit stalled, it did not reach a stationary point.")
    md.append(f"- Residual lag-1 autocorrelation at theta-hat (observed consecutive months, "
              f"{ac['n_pairs']} pairs): {ac['phi']:.3f}, variance factor "
              f"{ac['variance_factor_ar1']:.2f}, sd factor {ac['sd_factor_ar1']:.2f}. The Fisher "
              "sd at the reference refits (validation.json repair.profiles_v2.<target>."
              "fisher_at_reference) is given iid and AR(1)-inflated; both stay uncalibrated"
              + ("" if ref_conv else ", and the references are not stationary points")
              + ".\n")
    if not ref_conv:
        md.append(f"The reference refits {ref_txt}. A profile point re-optimised from such a "
                  "reference can lower the objective further for reasons unrelated to the "
                  "pinned entry, so negative profile differences partly measure the "
                  "unfinished refit (the verifier's point about v1 still applies, if less "
                  "strongly). Only the slices (every other entry at theta-hat) are free of "
                  "this; rely on readings where slice and profile agree.\n")

    pinned_intro = (
        "Seven entries sit on a bound (equal to the theta's `bounds_hit`). Values are N/s2 x "
        f"dloss (s2 = {P['s2_theta_hat']:.3f} m^2; N = {int(P['n_rows']['filled'])} "
        f"well-months back-filled, {int(P['n_rows']['observed'])} observed), i.e. iid chi2 "
        f"units, about {ac['variance_factor_ar1']:.1f}-fold too large for autocorrelated "
        "months. Slices move one entry with every other entry at theta-hat; profiles "
        "re-optimise the 28 interior entries from the reference refit of the same target "
        f"(which {'converged' if ref_conv else 'did not converge: ' + ref_txt}) and are "
        "measured from it. Outward = beyond the bound. Readings: "
        "binding = both slice and profile fall beyond the bound; not binding = both fall "
        f"inward; mixed otherwise. Same reading for both targets: {lst(agree)}.")

    deviations = [
        "Corrected: engine/v1 was built on batch 8's merged cwa theta, as the plan requires; "
        "the first round's statement that the theta was provisional (from stale task context) "
        "was wrong. engine/v2 republishes the identical engine with corrected provenance. v1 "
        "is kept unchanged (published versions are immutable), so its manifest and "
        "validation.json still carry the wrong 'provisional' label; v2's manifest records that "
        "it supersedes v1.",
        "Code lives in dataprep_staging/p4_changhua/ (decision D12) instead of "
        "results/dataprep/_staging_code/.",
        "The RIV verification configuration is the cwa theta plus the temporal_rivers recipe's "
        "river conductances and a synthetic canal recharge (fraction 0.25); the deliverable has "
        "no rivers, so no calibrated RIV model exists to verify on.",
        f"Kernels hold {k['n_lags']} lags (2012-01..2026-12) instead of 131, so each float32 "
        "kernel is 244 MB rather than 178 MB; the Green's functions took "
        f"{tm['greens_all_layers']} s against the plan's 166 s estimate.",
        "Bound-pinned profiles: v2 replaces v1's four-iteration chord steps (which mostly "
        "stopped at the iteration cap) with a Gauss-Newton reference refit (central-difference"
        " Jacobian refreshed every iteration, up to 15 iterations) and chord profiles from it, "
        "for back-filled and observed-only targets; the reference refits "
        + ("converged" if ref_conv else ref_txt + ", so the profiles remain "
           "convergence-limited (a longer refit would cost about 35 s per iteration and target"
           " in two CPU slots; not run)")
        + ". They remain approximate: finite-"
        "difference Jacobians, chord iterations for the profile points, the level loss with "
        "the datum profiled at fixed kappa, no torch re-optimisation, widened bounds "
        "evaluated only up to 1 log unit (slices) and 0.5 (profiles). v1's profile numbers are "
        "still in v1/ and v2's core block (identical code) and are superseded by the "
        "repair.profiles_v2 block.",
        "First round: a scratch trial of build_engine.py (24 lags, 4 draws) ran after the "
        "pre-registration and before the v1 build, and build_engine.py was edited between them "
        "(diff not recoverable; only the profile-refit diagnostic differed). Repair round: a "
        "reduced trial of build_engine_v2 (24 lags, 6 draws, 1 profile iteration) ran after "
        "the repair declaration; build_engine_v2.py was then edited only in the v1-identity "
        "skip rule and the manifest gate text. A first full build (2026-10-12T16:02Z) printed "
        "the core, R1-R3 and RIV numbers (equal to v2's) and ended with its session during "
        "the profiles, in a staging directory, before anything was published; "
        "build_engine_v2.py was then edited only in the theta-status string, and a second "
        "build (2026-10-13) produced v2. All runs are in the look log.",
        "The RIV diagnostic is post hoc (run after v1 to explain the RIV shortfall); it "
        "changes no threshold or case. It is now rerun inside the v2 build, so its numbers sit "
        "in v2's validation.json and diag_riv.py is in v2's manifest code list.",
        "The engine forcing is the twin's deliverable forcing (Yunlin census AMP_V2 "
        "tpc_pumps/pump_kwh_all, identical to the TPC L0 Yunlin sheet); the Changhua census is "
        "not an input of P4.0 (C0/P4m). The 188 GWh/yr synthetic Changhua abstraction is a "
        "verification load only.",
        "Repair code was added as new modules (engine_tlm.py, profiles_v2.py, "
        "build_engine_v2.py, repair_theta_status.py) instead of edits, because engine/v1's "
        "manifest hashes engine.py and build_engine.py. engine.energy_rate_jacobian's "
        "docstring therefore still calls the lift feedback 'second order' (wrong); the "
        "correction is in engine_tlm.py, v2's manifest and this report, and the docstring "
        "should be fixed at the Phase-1 port.",
        "Pre-registration wording (first round): the declaration preceded every computation "
        "in this WP, but the base case's 1.66e-6 m on this theta was already known from the "
        "scout (its prior_exposure says so) when base was named the acceptance case, while "
        "RIV, unmeasured, became a reported case. The designation is defensible (the merged "
        "theta has no rivers) but it was not blind, contrary to the first report's wording.",
        "report_p40*.py, looks_p40*.py, probe_fresh_blocking.py and the prereg scripts only "
        "read products or the look log and are not in any manifest's code list.",
        "The first build attempt's staging directory "
        "results/dataprep/changhua_abstraction/engine/_stage_v2_20261013T000244/ (about 1.2 GB:"
        " core outputs identical to v1, no manifest, never a version) is still on disk: this "
        "session was not permitted to delete it. Its core log and validation.json are copied "
        "to P40_evidence/repair/cutoff_*. It can be deleted by hand.",
    ]

    issues = [
        "A1 look log, for A1 and every WP that will register tests on A1 fresh units (H2, H3, "
        "P3, P4m, S1, S5, I1): this WP's in-sample computations on the calibration heads (datum "
        "replica, R2, level-loss slices, profiles, Fisher, residual autocorrelation; 158 wells, "
        "2012-01..2022-12, pooled, no hold-out split, no choice made) are logged through "
        "harness.prereg3 with their true members. Under prereg3's rule a look made after the "
        "fresh units were declared (2026-10-12T06:21Z) on overlapping members and windows "
        "blocks every later registration on those units: a read-only probe "
        f"(probe_fresh_blocking.py) finds {n_blk_looks} P40 looks eligible to block a test "
        f"registered now on {n_blk_units} of the {n_fresh} fresh units with a 2012-22 head "
        f"scope ({n_blk_looks - 2} repair-round in-sample looks, one per run, plus the first "
        "round's 2 in-sample looks, which became comparable because they share unit ids), "
        "against 0 before this round's looks. Other WPs' looks "
        f"there: {n_other_blk} blocking and up to {n_unv} logged without members (prereg/"
        "prereg2), which prereg3 only discloses. The plan (4.5) treats whole-window 2012-22 "
        "fits as design-time history and orders P4.0's profiling before the fresh-unit tests, "
        "so A1 should rule whether pooled in-sample replications without a hold-out split "
        "block fresh units. Until it does, assert_can_score will refuse fresh-unit tests "
        "registered after these looks.",
        "Use engine/v2 (results/dataprep/changhua_abstraction/engine/v2). It and v1 replicate "
        "batch 8's merged cwa theta (sha256 789d0e45...); no rebuild is needed for the merged "
        "theta, and the first round's instruction to rebuild and rerun everything built on v1 "
        "is withdrawn (v1's content is identical; only its labels are wrong). The plan's "
        "Phase-1 rebuild after the port to hydrophysics/dataprep remains. The checker will "
        "flag v1 and v2 if stage3_theta.json is rewritten in place.",
        "The deliverable page still uses the registered datum_gate model, to which the "
        "prospective registrations are bound (commit 709f660); the engine replicates the cwa "
        "recalibration, as plan P4.0 asks. Batch 8 scored within noise of the registered "
        "model, but any P4 statement about the deliverable should name which theta it rests "
        "on.",
        "The stored kernels are the frozen-lift linearisation and overstate the drawdown of a "
        f"large added abstraction: {lf['changhua_wells_mean_dh_frozen_m']:.2f} m vs "
        f"{lf['changhua_wells_mean_dh_step_m']:.2f} m with feedback at Changhua wells for "
        "188 GWh/yr. P4m and C0 add the Changhua census to a model calibrated without it: use "
        "SparseTwin.rollout (E_extra, stepwise) for the added load, and engine_tlm.tlm_rollout "
        "about the path that includes it for perturbations.",
        "Forcing-uncertainty propagation (plan 4.8): K Sigma K^T on the stored kernels is "
        "exact only for the frozen-lift linear system. The twin's lift feedback is first "
        "order and acts on every perturbation, volumetric ones included: the frozen-lift "
        f"formula overstates energy-error variance (stepwise MC / formula median {e_frz:.3f}) "
        f"and misstates volumetric-error variance (median {v_frz['median']:.3f}, sum ratio "
        f"{v_frz['sum_ratio']:.3f}, p05-p95 {v_frz['p05_p95'][0]:.2f}-"
        f"{v_frz['p05_p95'][1]:.2f}). Use engine_tlm.tlm_factor_responses + "
        "engine.head_var_factors for factor covariances, or engine_tlm.tlm_draw_responses for "
        f"independent noise: they match the stepwise Monte Carlo (median {e_tlm:.3f} and "
        f"{v_pair:.4f}). Keep the stepwise Monte Carlo for RIV configurations and for errors "
        f"large enough that non-linearity matters ({_pct(r2_['per_draw_nonlinear_rms_share'])}"
        " per-draw departure at 10 % energy errors).",
        f"RIV mode matches torch only to {fmt(a1['riv']['max_abs_dh_m'])} m at torch's default "
        "CG tolerance, so the registered 1e-5 m is missed; with the reference CG tightened to "
        f"1e-12 the match is {fmt(r12r['max_abs_dh_m'])} m with 0 switch disagreements. Quote "
        "it that way. The engine refuses delay-bed and aquitard-store models and "
        "eta_classes; a recalibration ladder that adds them needs an engine extension. "
        "engine_tlm refuses rivers.",
        "theta-hat is not stationary for its own level loss: the 15-iteration refit lowers it by "
        f"{_pct(reff['relative_decrease_vs_theta_hat'])} (back-filled) and "
        f"{_pct(refo['relative_decrease_vs_theta_hat'])} (observed months), in sample"
        + ("" if ref_conv else ", and still predicts a further "
           f"{_pct(reff['predicted_remaining_relative_decrease'])} / "
           f"{_pct(refo['predicted_remaining_relative_decrease'])}")
        + ". The "
        "Fisher sd in v1's and v2's core block (at theta-hat) is local curvature at a "
        "non-stationary point; the Fisher at the reference refits (repair.profiles_v2) is "
        f"iid and, AR(1)-inflated by {ac['sd_factor_ar1']:.2f}x (phi {ac['phi']:.3f} computed "
        "here), still uncalibrated. Do not quote either as an uncertainty.",
        "Bound-pinned entries (slices with profiles from "
        + ("converged" if ref_conv else "unconverged") + " references): binding for both targets: "
        + lst([r["entry"] for r in prow if r["filled"]["reading"].startswith("binding")
               and r["observed"]["reading"].startswith("binding")])
        + "; not binding for both: "
        + lst([r["entry"] for r in prow if r["filled"]["reading"].startswith("not binding")
               and r["observed"]["reading"].startswith("not binding")])
        + "; mixed or target-dependent: "
        + lst([r["entry"] for r in prow if r["entry"] not in agree
               or r["filled"]["reading"].startswith("mixed")])
        + ". " + ("" if ref_conv else "The profile half of each reading is "
                  "convergence-limited; the slices are not. ")
        + "Treat all seven as non-identified (plan P4.0) and widen their bounds only in "
        "sensitivities; a recalibration rung that fits observed months only (U2) should "
        "expect the observed-target readings.",
        f"The datum is effectively free (shrinkage {d['shrink_min']:.3f}-{d['shrink_median']:.3f};"
        f" the free datum differs by at most {d['free_datum_max_abs_diff_m']:.2f} m), so "
        "mean-level information about abstraction is absorbed by the datum.",
        "Kernel files: 5 x 244 MB float32 under engine/v2; load with np.load(mmap_mode='r'). "
        "K_pump is drawdown (positive down) per m3/day abstracted for one monthly step; "
        "G_layer<l> is head rise per m3/day injected in layer l. engine_arrays.npz holds the "
        "operator pieces, hindcast heads, Q_rate, datum, steady gains and cell/well counties.",
        "The allowed-use vocabulary (common/uses.py, owned by A0) has no use for plan 4.8's "
        "propagation of forcing uncertainty into head intervals. A consumer of the kernels "
        "for forcing_uncertainty/v1 must declare model_design (pre-origin, allowed) or "
        "after_the_fact_analysis until A0 adds a use.",
        "Changhua heads respond within one to three months: the quasi-steady share of the "
        f"steady gain is {k['changhua_quasi_steady_share_top_decile']['lag0']:.2f} at lag 0 and "
        f"{k['changhua_quasi_steady_share_top_decile']['lag2']:.2f} by lag 2, so monthly "
        "forcing timing errors (billing pair split, July mask) map almost directly into head "
        "errors (P2, P4m).",
        "72 of the 2148 cells fall outside every county polygon and were assigned to the "
        "nearest county (cell_county in engine_arrays.npz).",
    ]

    repair_round = [
        {"finding": "The report wrongly says the engine was built on a provisional theta "
                    "(batch 8 had finished; the theta is its merged theta)",
         "severity": "major",
         "response": "Correct; fixed. Verified independently: the merge job (deps fit_full + "
                     f"folds 0-4) wrote the theta at {ts['theta']['mtime_utc'][:19]}Z and its "
                     f"done marker at {dm['merge'][:19]}Z after a first merge attempt failed "
                     "on a commit-only difference; batch 8 finished at "
                     f"{dm['ring'][:19]}Z; meta carries the 5-fold gate; parameters equal the "
                     "full fit; no twin process running. engine/v2 republishes the identical "
                     "engine with the corrected status (byte-identical kernels and operator, "
                     f"identical arrays, validation numbers {irel_txt}); v1 is kept unchanged "
                     "and superseded. Deviation 1, the "
                     "downstream issues and the summary no longer call the theta provisional "
                     "and drop the rebuild instruction.",
         "evidence": "engine/v2 validation.json theta.status_evidence; "
                     "p4_changhua/repair_theta_status.py; v2 MANIFEST.json supersedes"},
        {"finding": "engine.energy_rate_jacobian calls the lift feedback of an energy "
                    "perturbation 'second order'; it is first order, so the 0.695 gap is an "
                    "omitted first-order term, not non-linearity",
         "severity": "minor",
         "response": "Correct; fixed by a new tangent-linear model (engine_tlm.py) that "
                     f"includes the feedback. R1: {fmt(r1_['energy']['tlm_vs_fd_rel'])} vs "
                     f"central differences (frozen-lift {r1_['energy']['frozen_vs_fd_rel']:.3f}); "
                     f"R2: stepwise MC / TLM median {e_tlm:.3f}, per-draw non-linear share "
                     f"{_pct(r2_['per_draw_nonlinear_rms_share'])}. engine.py is not edited "
                     "(v1's manifest hashes it); the wrong docstring is documented and left "
                     "for the Phase-1 port.",
         "evidence": "validation.json repair.R1_tlm_vs_finite_difference, "
                     "repair.R2_energy_error_variance; P40.repair_linearisation.v1"},
        {"finding": "'Var h = K Sigma K^T is exact for volumetric errors (0.9985)' is "
                    "overstated: the MC used the same frozen-lift linear system",
         "severity": "minor",
         "response": "Correct; claim withdrawn. R3 with the base pumping's lift feedback: "
                     f"stepwise / frozen formula median {v_frz['median']:.3f} (sum "
                     f"{v_frz['sum_ratio']:.3f}); stepwise / TLM, same draws, {v_pair:.4f}. "
                     "0.9985 is now described as an implementation check.",
         "evidence": "validation.json repair.R3_volumetric_noise_variance"},
        {"finding": "Manifest gate omits the failed RIV case; P40.json lists the post-hoc RIV "
                    "diagnostic as a passed acceptance row",
         "severity": "minor",
         "response": "Fixed in v2: the gate criteria name the RIV failure and carry per-case "
                     "pass flags (RIV false); the acceptance list has no diagnostic row (the "
                     "diagnostic is in key numbers and its own section). v1's manifest is "
                     "unchanged.",
         "evidence": "engine/v2 MANIFEST.json gate; P40.json acceptance"},
        {"finding": "'Pre-registered before any acceptance number was computed' overstates "
                    "the blinding (base's value was known from the scout; RIV unmeasured)",
         "severity": "minor",
         "response": "Correct; reworded (deviation 10). The repair declaration was written "
                     "before any R1-R3 number was computed on the twin.",
         "evidence": "P40.engine_acceptance.v1 prior_exposure; "
                     "P40.repair_linearisation.v1"},
        {"finding": "Level-loss diagnostics use back-filled targets; U2 recalibration fits "
                    "observed months only",
         "severity": "minor",
         "response": "Addressed: slices, reference refits, profiles and Fisher are computed "
                     "for back-filled and observed-only targets; same reading for both "
                     f"targets: {lst(agree)}.",
         "evidence": "validation.json repair.profiles_v2"},
        {"finding": "Bound-pinned profiles are capped by the trust region (refits stopped at "
                    "the iteration cap)",
         "severity": "minor",
         "response": ("Addressed: " if ref_conv else "Partly addressed: ")
                     + "reference refits with a Jacobian refreshed every iteration, up to 15 "
                     f"iterations, {ref_txt}; profile points {n_conv} of {n_prof} stopped on "
                     f"the tolerance (largest predicted remaining {fmt(max_rem)})."
                     + ("" if ref_conv else " The profile readings therefore stay "
                        "convergence-limited; the slices are not, and only readings where "
                        "slice and profile agree should be relied on.")
                     + " Still approximate (deviation 5).",
         "evidence": "validation.json repair.profiles_v2"},
        {"finding": "'About 6.7-fold overconfident' mixes a variance factor with an sd claim; "
                    "phi 0.74 was not computed here",
         "severity": "minor",
         "response": f"Correct; phi computed in this WP ({ac['phi']:.3f} at theta-hat): "
                     f"variance factor {ac['variance_factor_ar1']:.2f}, sd factor "
                     f"{ac['sd_factor_ar1']:.2f}.",
         "evidence": "validation.json repair.profiles_v2.residual_autocorrelation_theta_hat"},
        {"finding": "diag_riv.py produced acceptance-table evidence but is in no manifest",
         "severity": "minor",
         "response": "Fixed: the RIV diagnostic is rerun inside the v2 build (numbers equal "
                     "the first round's) and diag_riv.py is in v2's code list.",
         "evidence": "validation.json repair.riv_diagnostic; v2 MANIFEST.json code"},
        {"finding": "No allowed use covers plan 4.8's forcing-uncertainty propagation",
         "severity": "minor (downstream)",
         "response": "Not changed here: common/uses.py belongs to A0 and existing manifests "
                     "depend on the vocabulary. Flagged for A0 in the downstream issues.",
         "evidence": "issues for downstream"},
    ]
    repair_intro = (
        "An independent verifier confirmed every acceptance number of the first round and "
        "raised one major issue (the provisional-theta label) and nine minor ones. The table "
        "lists each with the response. Nothing in the registered acceptance changed: A1 base "
        "and A2 pass, the RIV case fails, and the thresholds are the pre-registered ones. The "
        "repair adds engine/v2 (same content as v1, corrected provenance), a tangent-linear "
        "model with its registered verification (P40.repair_linearisation.v1: "
        f"{'pass' if rall else 'FAIL'}), longer in-sample refits and profiles for both "
        "target conventions, and the RIV diagnostic "
        "inside a manifest. A first build attempt ended with its session before publishing; "
        "its printed numbers equal v2's and it is logged as a look.")
    looks = ("7 first-round looks (logged late, looks_p40.py, harness.prereg) and the repair "
             "round's looks (the trial, the cut-off first build and the product build of "
             "build_engine_v2, each on the acceptance, repair and in-sample units; "
             "looks_p40_repair.py, harness.prereg3, numerical units with members 'none' and "
             "in-sample units with the 158 calibration wells over 2012-01..2022-12). The "
             f"in-sample looks are eligible to block later registrations on {n_blk_units} of "
             f"{n_fresh} A1 fresh units (see issues for downstream).")
    return {"summary": summary, "md_sections": md, "pinned_intro": pinned_intro,
            "deviations": deviations, "issues": issues, "repair_round": repair_round,
            "repair_intro": repair_intro, "looks": looks}
