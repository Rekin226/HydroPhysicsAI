"""WP P4.0 repair round: ``changhua_abstraction/engine/v<N>`` on batch 8's merged cwa theta.

    cd dataprep_staging
    CUDA_VISIBLE_DEVICES= tools/cpu_slot.sh --slots 2 python -m p4_changhua.build_engine_v2

Why a new version (verifier, 2026-10-12): engine/v1's manifest, validation.json and report
called the theta "provisional: batch 8's merged theta does not exist yet". It was batch 8's
merged theta (``repair_theta_status``). v1 is left untouched; this builder publishes v<N+1>
with the corrected provenance, a gate text that names the failed RIV case, and the repair
diagnostics.

Steps

1. core: ``build_engine.main`` (unchanged since v1, so the kernels come from hash-identical
   code) in staging mode under the product root; the staged files become the new version;
2. theta status: ``repair_theta_status.collect`` must confirm batch 8's merged theta, and the
   theta's status fields are rewritten from that evidence;
3. identity with v1: every staged output against v1's (sha256, array contents, numbers);
4. registered repair diagnostics (declaration ``P40.repair_linearisation.v1``):
   R1 tangent-linear model vs central differences of the stepwise rollout (energy and
   volumetric directions); R2 energy-error variance: tangent-linear ``J Sigma J^T`` vs a
   fresh stepwise Monte Carlo (the frozen-lift ``K Sigma K^T`` reported alongside); R3
   independent volumetric noise with the base pumping's lift feedback: stepwise Monte Carlo
   vs the same draws through the tangent-linear model (and the frozen-lift formula);
5. the RIV diagnostic of ``diag_riv`` rerun inside the build, so its evidence sits in a
   manifest;
6. in-sample profiles of the bound-pinned entries (``profiles_v2``): slices, converged
   reference refits and local profiles for back-filled targets and observed months only,
   Fisher information at the converged reference, residual lag-1 autocorrelation;
7. validation.json (core block + ``repair`` block), new version directory, manifest.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse  # noqa: E402
import datetime as _dt  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import manifest  # noqa: E402
from common.hashing import sha256_file  # noqa: E402
from common.paths import DATAPREP, REPO, REPORTS_DIR, TWIN_GRID_NPZ, rel_to_repo  # noqa: E402

from p4_changhua import build_engine as be  # noqa: E402
from p4_changhua import engine, engine_tlm, inputs_p4, profiles_v2  # noqa: E402
from p4_changhua import repair_theta_status as rts  # noqa: E402

BUILDER_VERSION = "0.2.0"
PREREG_ID = "P40.repair_linearisation.v1"
EVID = REPORTS_DIR / "P40_evidence" / "repair"
SEED_FD, SEED_E, SEED_Q = 2029, 2027, 2028
EPS_E, EPS_Q = 1e-4, 1e-3
TOL_R1 = 1e-6
R3_BAND = (0.98, 1.02)
PROFILE_OFFSETS = (-0.5, -0.25, 0.25, 0.5)
SLICE_OFFSETS = (-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0)
log = be.log


def _rel(a, b) -> float:
    return float(np.abs(np.asarray(a) - np.asarray(b)).max() / np.abs(np.asarray(b)).max())


def _ratio_stats(num, den, lo, hi, floor=1e-6) -> dict:
    sel = den > floor * den.max()
    r = num[sel] / den[sel]
    return {"median": float(np.median(r)), "p05_p95": [float(np.percentile(r, 5)),
                                                      float(np.percentile(r, 95))],
            "share_in_chi2_band": float(((r >= lo) & (r <= hi)).mean()),
            "sum_ratio": float(num[sel].sum() / den[sel].sum()), "n": int(sel.sum())}


def _numeric_diff(a, b, path="", skip=()) -> list[tuple[str, float]]:
    """Leaves of two JSON trees that differ (numbers by absolute difference)."""
    out = []
    if any(s in path for s in skip):
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if any(s in f"{path}/{k}" for s in skip):
                continue
            if k not in a or k not in b:
                out.append((f"{path}/{k}", float("inf")))
            else:
                out += _numeric_diff(a[k], b[k], f"{path}/{k}", skip)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((path, float("inf")))
        else:
            for i, (x, y) in enumerate(zip(a, b, strict=True)):
                out += _numeric_diff(x, y, f"{path}[{i}]", skip)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(
            a, bool):
        if a != b:
            out.append((path, abs(float(a) - float(b))))
    elif a != b:
        out.append((path, float("inf")))
    return out


def identity_with_v1(stage: Path, V: dict) -> dict:
    v1 = DATAPREP / be.PRODUCT / "v1"
    m1 = json.loads((v1 / "MANIFEST.json").read_text())
    sha1 = {o["path"]: o["sha256"] for o in m1["outputs"]}
    files = {}
    for f in sorted(stage.iterdir()):
        files[f.name] = {"sha256_equal": sha256_file(f) == sha1.get(f.name)}
    with np.load(stage / "engine_arrays.npz", allow_pickle=False) as a, \
            np.load(v1 / "engine_arrays.npz", allow_pickle=False) as b:
        arr = {k: bool(np.array_equal(a[k], b[k])) for k in a.files}
        arr_keys_equal = sorted(a.files) == sorted(b.files)
    V1 = json.loads((v1 / "validation.json").read_text())
    diffs = _numeric_diff(V, V1, skip=("timing_s", "t_torch_s", "t_sparse_s", "t_greens_s",
                                       "t_mc_s", "t_jacobian_s", "/created", "theta/status",
                                       "profiles/t_s"))
    return {"v1": rel_to_repo(v1), "files": files,
            "kernels_and_operator_byte_identical": all(
                files[k]["sha256_equal"] for k in files if k.endswith(".npy")
                or k == "M_operator.npz"),
            "engine_arrays_identical": bool(arr_keys_equal and all(arr.values())),
            "engine_arrays_differing_keys": [k for k, v in arr.items() if not v],
            "validation_numbers_differing": [{"path": p, "abs_diff": d} for p, d in diffs],
            "validation_numbers_identical": not diffs}


def energy_factors(d, Ex, zone3, ch, nT):
    """The 39 energy-error factors of engine/v1 (build_engine step 7), rebuilt identically."""
    tt = np.arange(nT)
    year = (tt + 1) // 12
    fac = []
    A = Ex.shape[0]
    for z in sorted(set(zone3[ch].tolist())):
        cz = ch & (zone3 == z)
        for y in sorted(set(year.tolist())):
            f = np.zeros((A, nT))
            f[np.ix_(cz, year == y)] = 0.10 * Ex[np.ix_(cz, year == y)]
            fac.append(f)
        for trig in (np.cos, np.sin):
            f = np.zeros((A, nT))
            f[cz] = 0.10 * Ex[cz] * trig(2 * np.pi * ((tt + 1) % 12) / 12.0)[None, :]
            fac.append(f)
    return np.stack(fac)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-lags", type=int, default=be.N_LAGS)
    ap.add_argument("--mc-draws", type=int, default=200)
    ap.add_argument("--profile-iters", type=int, default=15)
    ap.add_argument("--skip-profiles", action="store_true")
    ap.add_argument("--core-skip-profiles", action="store_true",
                    help="trial only: skip build_engine's own (v1) profile diagnostics")
    ap.add_argument("--scratch", default=None,
                    help="trial run: everything under this directory; no version directory, "
                         "no manifest, no evidence files")
    args = ap.parse_args(argv)
    global EVID
    t_start = time.perf_counter()
    os.chdir(REPO)
    ts = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    if args.scratch:
        stage_root = Path(args.scratch).resolve() / f"v2trial_{ts}"
        EVID = stage_root / "evidence_repair"
    else:
        stage_root = DATAPREP / be.PRODUCT / f"_stage_v2_{ts}"
    stage_root.mkdir(parents=True)
    EVID.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- 1. core (v1 code)
    core = ["--scratch", str(stage_root), "--n-lags", str(args.n_lags), "--mc-draws",
            str(args.mc_draws)] + (["--skip-profiles"] if args.skip_profiles
                                   or args.core_skip_profiles else [])
    if args.core_skip_profiles and not args.scratch:
        raise SystemExit("--core-skip-profiles is for trial runs (--scratch) only")
    log(f"core: build_engine.main {core} (build_engine.py sha256 "
        f"{sha256_file(Path(be.__file__))})")
    rc = be.main(core)
    builds = sorted(stage_root.glob("_build_*"))
    if len(builds) != 1:
        raise SystemExit(f"expected one staged build under {stage_root}, found {builds}")
    stage = builds[0]
    V = json.loads((stage / "validation.json").read_text())
    log(f"core finished rc={rc}; passed {V['passed']}")
    import torch

    from hydrophysics.twin.forward import load_members

    if torch.cuda.is_available():
        raise SystemExit("CUDA is visible")

    # ---------------------------------------------------------------- 2. theta status
    status = rts.collect()
    if not status["is_batch8_merged_theta"] or status["theta"]["sha256"] != V["theta"]["sha256"]:
        raise SystemExit(f"theta status not confirmed: {status['verdict']}")
    V["theta"]["status"] = ("batch 8's merged cwa theta (stage3_datum_gate_cwa, written by "
                            "merge_folds 2026-10-11T19:06:12Z after fit_full and folds 0-4, "
                            "on the rerun that followed the first merge attempt's commit-only "
                            "refusal (commit 709f660); 5-fold gate in meta; parameters equal "
                            "the full fit; batch 8 finished 2026-10-11T20:30:55Z); verified in "
                            "the P4.0 repair round. engine/v1's 'provisional' label was wrong.")
    V["theta"]["status_evidence"] = status
    log(f"theta status: {status['verdict']}")

    # ---------------------------------------------------------------- 3. identity with v1
    R: dict = {"builder": "p4_changhua.build_engine_v2", "builder_version": BUILDER_VERSION,
               "engine_tlm_version": engine_tlm.__version__, "prereg": PREREG_ID,
               "core_rc": rc}
    if args.n_lags == be.N_LAGS and args.mc_draws == 200 and not args.skip_profiles:
        R["identity_with_v1"] = identity_with_v1(stage, V)
        log(f"identity with v1: kernels/operator byte-identical "
            f"{R['identity_with_v1']['kernels_and_operator_byte_identical']}, arrays "
            f"{R['identity_with_v1']['engine_arrays_identical']}, numbers identical "
            f"{R['identity_with_v1']['validation_numbers_identical']}")

    # ---------------------------------------------------------------- reload the twin
    member = load_members([str(be.THETA)])[0]
    meta = member.meta
    d = be.load_everything(meta)
    grid, ge, h0, E, R_ = d["grid"], d["ge"], d["h0"], d["E_tot"], d["R"]
    oi, ol = d["oi"], d["ol"]
    obs = (ol, oi)
    A, nT, L = grid.n_active, E.shape[1], 4
    ch = d["cell_county"] == be.CH
    model, sc, zone3, tw = be.build(member, grid, ge)
    H_base, Q_base = tw.rollout(h0, E=E, R=R_, return_rates=True)
    with np.load(stage / "engine_arrays.npz", allow_pickle=False) as z:
        R["hindcast_equals_staged"] = float(np.abs(z["H_hindcast"] - H_base).max())
    Ex = be.synthetic_changhua_energy(d)
    H_extra = tw.rollout(h0, E=E, R=R_, E_extra=Ex)
    Es_base = tw.spread(E)
    Es_extra = Es_base + tw.spread(Ex)
    wts = tw.pump_weights()
    K32 = np.load(stage / "K_pump_f32.npy", mmap_mode="r")
    Hb_w, He_w = H_base[ol, oi, 1:], H_extra[ol, oi, 1:]
    R["lift_clip"] = {"base_path": engine_tlm.clipped_lift_count(tw, H_base),
                      "extra_path": engine_tlm.clipped_lift_count(tw, H_extra)}

    def sources(rate):                       # (A, T) abstraction rate -> (L, A, T) sources
        q = np.zeros((L,) + rate.shape)
        for lay, w in wts.items():
            q[lay] += w * rate
        return q

    # ---------------------------------------------------------------- 4. R1 (registered)
    t0 = time.perf_counter()
    r1 = {}
    dE = tw.spread(Ex)
    fd = (tw.rollout(h0, E_extra=Es_extra + EPS_E * dE, R=R_, spread_extra=False)
          - tw.rollout(h0, E_extra=Es_extra - EPS_E * dE, R=R_, spread_extra=False)) / (2 * EPS_E)
    tl = engine_tlm.tlm_rollout(tw, H_extra, Es_extra, dEs=dE)
    fr = engine_tlm.tlm_rollout(tw, H_extra, Es_extra, dEs=dE, frozen=True)
    kern = -engine.convolve(K32, engine.energy_rate_jacobian(tw, H_extra) * dE)
    r1["energy"] = {"direction": "spread synthetic Changhua energy (188 GWh/yr), extra path",
                    "eps": EPS_E, "tlm_vs_fd_rel": _rel(tl, fd),
                    "tlm_vs_fd_rel_wells": _rel(tl[ol, oi, 1:], fd[ol, oi, 1:]),
                    "frozen_vs_fd_rel": _rel(fr, fd),
                    "frozen_tlm_vs_stored_kernel_rel": _rel(fr[ol, oi, 1:], kern),
                    "fd_max_abs_m_per_unit": float(np.abs(fd).max())}
    dr = 0.3 * Q_base * np.random.default_rng(SEED_FD).standard_normal((A, nT))
    qd = sources(dr)
    fd = (tw.rollout(h0, E=E, R=R_, q_extra=EPS_Q * qd)
          - tw.rollout(h0, E=E, R=R_, q_extra=-EPS_Q * qd)) / (2 * EPS_Q)
    tl = engine_tlm.tlm_rollout(tw, H_base, Es_base, drate=dr)
    fr = engine_tlm.tlm_rollout(tw, H_base, Es_base, drate=dr, frozen=True)
    r1["volumetric"] = {"direction": "independent abstraction-rate field, sd 0.3 x base rate "
                                     f"(seed {SEED_FD}), base path",
                        "eps": EPS_Q, "tlm_vs_fd_rel": _rel(tl, fd),
                        "tlm_vs_fd_rel_wells": _rel(tl[ol, oi, 1:], fd[ol, oi, 1:]),
                        "frozen_vs_fd_rel": _rel(fr, fd),
                        "fd_max_abs_m_per_unit": float(np.abs(fd).max())}
    r1["threshold_rel"] = TOL_R1
    r1["passed"] = bool(r1["energy"]["tlm_vs_fd_rel"] <= TOL_R1
                        and r1["volumetric"]["tlm_vs_fd_rel"] <= TOL_R1)
    r1["t_s"] = round(time.perf_counter() - t0, 1)
    R["R1_tlm_vs_finite_difference"] = r1
    log(f"R1: {r1}")
    del fd, tl, fr

    from scipy.stats import chi2

    nd = args.mc_draws
    lo, hi = (float(x) for x in chi2.ppf([0.025, 0.975], nd - 1) / (nd - 1))

    # ---------------------------------------------------------------- 4. R2 (registered)
    t0 = time.perf_counter()
    F_E = energy_factors(d, Ex, zone3, ch, nT)
    S_E = np.stack([tw.spread(f) for f in F_E])
    U_t = engine_tlm.tlm_factor_responses(tw, H_extra, Es_extra, S_E, "energy", obs)
    U_f = engine_tlm.tlm_factor_responses(tw, H_extra, Es_extra, S_E, "energy", obs,
                                          frozen=True)
    V_t, V_f = engine.head_var_factors(U_t), engine.head_var_factors(U_f)
    c_ex = engine.energy_rate_jacobian(tw, H_extra)
    V_k = engine.head_var_factors(engine.factor_responses(K32, c_ex[None] * S_E))
    rng = np.random.default_rng(SEED_E)
    acc, acc2 = np.zeros_like(V_t), np.zeros_like(V_t)
    dev2 = lin2 = 0.0
    for _ in range(nd):
        xi = rng.standard_normal(F_E.shape[0])
        hd = tw.rollout(h0, E_extra=Es_extra + np.tensordot(xi, S_E, axes=1), R=R_,
                        spread_extra=False)[ol, oi, 1:]
        acc += hd
        acc2 += hd ** 2
        lin = np.tensordot(xi, U_t, axes=1)
        dev2 += float(((hd - He_w - lin) ** 2).sum())
        lin2 += float((lin ** 2).sum())
    V_mc = (acc2 - acc ** 2 / nd) / (nd - 1)
    r2 = {"factors": f"{F_E.shape[0]} energy-error factors of engine/v1 (10 % zone-year levels "
                     "and zone annual cycles of the synthetic Changhua abstraction), extra path",
          "n_draws": nd, "seed": SEED_E, "chi2_95_band": [lo, hi],
          "mc_over_tlm": _ratio_stats(V_mc, V_t, lo, hi),
          "mc_over_frozen_formula": _ratio_stats(V_mc, V_f, lo, hi),
          "tlm_over_frozen_formula": _ratio_stats(V_t, V_f, 0, np.inf),
          "frozen_tlm_vs_kernel_formula_rel": _rel(V_f, V_k),
          "per_draw_nonlinear_rms_share": math.sqrt(dev2 / lin2),
          "max_head_sd_tlm_m": float(np.sqrt(V_t.max())),
          "max_head_sd_frozen_m": float(np.sqrt(V_f.max()))}
    r2["passed"] = bool(lo <= r2["mc_over_tlm"]["median"] <= hi)
    r2["t_s"] = round(time.perf_counter() - t0, 1)
    R["R2_energy_error_variance"] = r2
    log(f"R2: {r2}")
    del U_t, U_f

    # ---------------------------------------------------------------- 4. R3 (registered)
    t0 = time.perf_counter()
    sd_q = 0.3 * Q_base

    def draw(i):
        return sd_q * np.random.default_rng([SEED_Q, i]).standard_normal((A, nT))

    T_resp = np.zeros((len(oi), nT, nd))
    F_resp = np.zeros((len(oi), nT, nd))
    for s, blk in engine_tlm.tlm_draw_responses(tw, H_base, Es_base, draw, nd, "rate", obs):
        T_resp[:, :, s:s + blk.shape[-1]] = blk
    for s, blk in engine_tlm.tlm_draw_responses(tw, H_base, Es_base, draw, nd, "rate", obs,
                                                frozen=True):
        F_resp[:, :, s:s + blk.shape[-1]] = blk
    acc, acc2 = np.zeros((len(oi), nT)), np.zeros((len(oi), nT))
    dev2 = lin2 = 0.0
    for i in range(nd):
        hd = tw.rollout(h0, E=E, R=R_, q_extra=sources(draw(i)))[ol, oi, 1:] - Hb_w
        acc += hd
        acc2 += hd ** 2
        dev2 += float(((hd - T_resp[:, :, i]) ** 2).sum())
        lin2 += float((T_resp[:, :, i] ** 2).sum())
    V_step = (acc2 - acc ** 2 / nd) / (nd - 1)
    V_tmc = T_resp.var(axis=-1, ddof=1)
    V_fmc = F_resp.var(axis=-1, ddof=1)
    V_form = engine.head_var_independent(K32, sd_q ** 2)
    r3 = {"noise": "independent cell-month abstraction-rate errors, sd 0.3 x base rate, base "
                   "path (lift feedback of the base census pumping active)",
          "n_draws": nd, "seed": SEED_Q, "band": list(R3_BAND), "chi2_95_band": [lo, hi],
          "step_over_tlm_mc_paired": _ratio_stats(V_step, V_tmc, *R3_BAND),
          "per_draw_nonlinear_rms_share": math.sqrt(dev2 / lin2),
          "step_over_frozen_formula": _ratio_stats(V_step, V_form, lo, hi),
          "tlm_mc_over_frozen_mc_paired": _ratio_stats(V_tmc, V_fmc, 0, np.inf),
          "frozen_mc_over_frozen_formula": _ratio_stats(V_fmc, V_form, lo, hi),
          "max_head_sd_step_m": float(np.sqrt(V_step.max())),
          "max_head_sd_formula_m": float(np.sqrt(V_form.max()))}
    r3["passed"] = bool(R3_BAND[0] <= r3["step_over_tlm_mc_paired"]["median"] <= R3_BAND[1])
    r3["t_s"] = round(time.perf_counter() - t0, 1)
    R["R3_volumetric_noise_variance"] = r3
    log(f"R3: {r3}")
    del T_resp, F_resp
    R["acceptance"] = {"R1": r1["passed"], "R2": r2["passed"], "R3": r3["passed"],
                       "all": bool(r1["passed"] and r2["passed"] and r3["passed"])}

    # ---------------------------------------------------------------- 5. RIV diagnostic
    from p4_changhua import diag_riv

    t0 = time.perf_counter()
    rd = {"theta_sha256": V["theta"]["sha256"]}
    rd["base"] = diag_riv.run_case(tw, sc, h0, E, R_, None, "base")
    mr = be.riv_member(member)
    _, sc_r, zone3_r, tw_r = be.build(mr, grid, ge)
    months = np.arange(1, nT + 1) % 12
    sw = np.zeros((A, nT))
    sw[zone3_r == 1] = 2e-3 * np.clip(np.sin(2 * np.pi * (months - 2) / 12.0), 0, None)[None, :]
    rd["riv"] = diag_riv.run_case(tw_r, sc_r, h0, E, R_, sw, "riv")
    rd["t_s"] = round(time.perf_counter() - t0, 1)
    R["riv_diagnostic"] = rd
    del tw_r
    log(f"RIV diagnostic done in {rd['t_s']}s")

    # ---------------------------------------------------------------- 6. profiles v2
    if not args.skip_profiles:
        R["profiles_v2"] = profiles(member, meta, d, grid, ge, h0, E, R_, ol, oi,
                                    H_base, args.profile_iters)

    # ---------------------------------------------------------------- 7. outputs
    R["t_total_s"] = round(time.perf_counter() - t_start, 1)
    V["repair"] = R
    V["acceptance"]["repair_linearisation_declaration"] = R["acceptance"]["all"]
    V["engine_version_note"] = ("engine.py, build_engine.py, inputs_p4.py and __init__.py are "
                                "hash-identical to engine/v1's; the repair block comes from "
                                f"build_engine_v2 {BUILDER_VERSION} and engine_tlm "
                                f"{engine_tlm.__version__}")
    (stage / "validation.json").write_text(json.dumps(V, ensure_ascii=False, indent=1,
                                                      default=float) + "\n")
    core_log = stage_root / "evidence" / "build_engine.log"
    if args.scratch:
        (EVID / "build_engine_v2.log").write_text("\n".join(be.LOG) + "\n")
        log(f"trial written to {stage}")
        return 0
    vdir = manifest.new_version_dir(be.PRODUCT)
    for f in sorted(stage.iterdir()):
        shutil.move(str(f), vdir / f.name)
    stage.rmdir()
    if core_log.exists():
        shutil.move(str(core_log), EVID / "build_engine_core.log")
    shutil.rmtree(stage_root / "evidence", ignore_errors=True)
    stage_root.rmdir()
    write_manifest(vdir, V, meta, d)
    out = {"version_dir": rel_to_repo(vdir), "passed": V["passed"],
           "acceptance": V["acceptance"], "repair_acceptance": R["acceptance"]}
    (EVID / "build_result_v2.json").write_text(json.dumps(out, indent=1) + "\n")
    log(f"wrote {rel_to_repo(vdir)}; passed {V['passed']}; repair {R['acceptance']}; total "
        f"{time.perf_counter() - t_start:.0f}s")
    (EVID / "build_engine_v2.log").write_text("\n".join(be.LOG) + "\n")
    return 0 if V["passed"] else 1


# ============================================================================ profiles
def profiles(member, meta, d, grid, ge, h0, E, R_, ol, oi, H_base, n_iter) -> dict:
    from hydrophysics.twin.calibrate_flow import _r2
    from hydrophysics.twin.forward import Member

    t_all = time.perf_counter()
    theta = member.theta
    target, dmask = d["obs_filled"], np.isfinite(d["obs"])
    pinned = engine.bound_pinned(theta, meta)
    entries = engine.theta_vector(theta, meta)
    pin_names = {f"{p.name}[{p.index}]" for p in pinned}
    interior = [e for e in entries if e["name"] not in pin_names]
    lo_b = np.array([-np.inf if e["lo"] is None else e["lo"] for e in interior])
    hi_b = np.array([np.inf if e["hi"] is None else e["hi"] for e in interior])
    x_hat = np.array([e["value"] for e in interior])
    pred_hat = H_base[ol, oi, 1:]
    _, kap, s2, n_obs = engine.profile_datum(pred_hat, target, dmask)
    N = {"filled": float(target.size), "observed": float(dmask.sum())}
    n_eval = [0]

    def predict(th):
        n_eval[0] += 1
        _, _, _, twp = be.build(Member(label="profile", theta=th, meta=meta), grid, ge)
        return twp.rollout(h0, E=E, R=R_)[ol, oi, 1:]

    def residuals(pr):
        n_ = dmask.sum(axis=1)
        rb = np.where(dmask, target - pr, 0.0).sum(axis=1) / np.maximum(n_, 1)
        dd = np.where(n_ > 0, n_ * rb / np.maximum(n_ + kap, 1e-12), 0.0)
        e = pr + dd[:, None] - target
        tail = np.sqrt(kap) * dd
        return {"filled": np.concatenate([e.ravel(), tail]),
                "observed": np.concatenate([e[dmask], tail]), "e": e}

    def with_interior(th, x):
        for e, v in zip(interior, x, strict=True):
            th = engine.perturb_theta(th, e["key"], e["index"], float(v))
        return th

    def make_eval(th_fixed, which):
        return lambda x: residuals(predict(with_interior(th_fixed, x)))[which]

    def losses(pr):
        lf = engine.fit_flow_level_loss(pr, target, dmask)[0]
        lo_ = engine.fit_flow_level_loss(pr, target, dmask, loss_mask=dmask)[0]
        return {"filled": lf, "observed": lo_}

    out = {"objective": "fit_flow level loss with the per-well datum profiled at fixed kappa "
                        "(kappa at theta-hat); 'filled' = back-filled targets on every "
                        "well-month (fit_flow's convention), 'observed' = observed months "
                        "only (plan U2); in-sample 2012-01..2022-12",
           "n_rows": N, "s2_theta_hat": s2, "n_interior": len(interior),
           "pinned": [f"{p.name}[{p.index}]" for p in pinned]}
    base = losses(pred_hat)
    out["base_loss"] = base
    res_hat = residuals(pred_hat)
    out["residual_autocorrelation_theta_hat"] = profiles_v2.lag1_autocorr(res_hat["e"], dmask)
    # slices (every other entry at theta-hat)
    t0 = time.perf_counter()
    slices = []
    for p in pinned:
        sgn = 1.0 if p.side == "hi" else -1.0
        rows = []
        for off in SLICE_OFFSETS:
            lv = losses(predict(engine.perturb_theta(theta, p.name, p.index,
                                                     p.bound + sgn * off)))
            rows.append({"offset_outward": off,
                         **{f"dloss_x_N_over_s2_{k}": (lv[k] - base[k]) * N[k] / s2
                            for k in N}})
        slices.append({"name": p.name, "index": p.index, "side": p.side, "rows": rows})
    out["slices"] = slices
    out["t_slices_s"] = round(time.perf_counter() - t0, 1)
    log(f"profiles v2: slices in {out['t_slices_s']}s")
    for which in ("filled", "observed"):
        t0 = time.perf_counter()
        f_ref = make_eval(theta, which)
        ref = profiles_v2.gn_refit(f_ref, x_hat, lo_b, hi_b, n_iter=n_iter, trust=0.5,
                                   tol=1e-6)
        J, _ = profiles_v2.central_jacobian(f_ref, ref["x"])
        rem = profiles_v2.predicted_remaining(J, ref["res"], ref["x"], lo_b, hi_b)
        g = J.T @ ref["res"]
        F = J.T @ J / s2
        try:
            sd = np.sqrt(np.clip(np.diag(np.linalg.inv(F)), 0, None))
        except np.linalg.LinAlgError:
            sd = np.full(len(interior), np.nan)
        pr_ref = predict(with_interior(theta, ref["x"]))
        ac = profiles_v2.lag1_autocorr(residuals(pr_ref)["e"], dmask)
        obj_hat = float(res_hat[which] @ res_hat[which])
        blk = {"reference": {"objective_mse": ref["obj"] / N[which],
                             "objective_mse_theta_hat": obj_hat / N[which],
                             "relative_decrease_vs_theta_hat": (obj_hat - ref["obj"]) / obj_hat,
                             "fit_flow_loss": losses(pr_ref)[which],
                             "r2": float(_r2(pr_ref, target) if which == "filled" else
                                         _r2(pr_ref[dmask], target[dmask])),
                             "n_iter": ref["n_iter"], "n_eval": ref["n_eval"],
                             "stop_reason": ref["reason"],
                             "last_step_hit_trust_box": ref["last_step_hit_trust_box"],
                             "predicted_remaining_relative_decrease": rem,
                             "max_abs_gradient_over_s2": float(np.abs(g).max() / s2),
                             "interior_shift": [{"name": e["name"], "shift": float(v)}
                                                for e, v in zip(interior, ref["x"] - x_hat,
                                                                strict=True)],
                             "residual_autocorrelation": ac},
               "fisher_at_reference": {
                   "convention": "iid months of variance s2 (fit_flow's), datum profiled at "
                                 "fixed kappa, pinned entries held at their bounds; "
                                 "UNCALIBRATED; sd_ar1 inflates by sqrt((1+phi)/(1-phi)) with "
                                 "the residual lag-1 autocorrelation phi at the reference "
                                 "(a crude effective-sample correction)",
                   "phi": ac["phi"], "sd_factor_ar1": ac["sd_factor_ar1"],
                   "cond": float(np.linalg.cond(F)),
                   "entries": [{"name": e["name"], "value": float(v), "sd_iid": float(s),
                                "sd_ar1": float(s * ac["sd_factor_ar1"])}
                               for e, v, s in zip(interior, ref["x"], sd, strict=True)]},
               "t_reference_s": round(time.perf_counter() - t0, 1)}
        log(f"profiles v2 [{which}]: reference refit {ref['reason']} after {ref['n_iter']} "
            f"iterations, decrease {blk['reference']['relative_decrease_vs_theta_hat']:.4f}, "
            f"remaining {rem:.2e}, {blk['t_reference_s']}s")
        t0 = time.perf_counter()
        prof = []
        for p in pinned:
            sgn = 1.0 if p.side == "hi" else -1.0
            rows = [{"offset_outward": 0.0, "dprofile_x_N_over_s2": 0.0, "stop_reason":
                     "reference", "n_iter": 0, "max_abs_shift_from_reference": 0.0,
                     "predicted_remaining_relative_decrease": rem}]
            for off in PROFILE_OFFSETS:
                th_f = engine.perturb_theta(theta, p.name, p.index, p.bound + sgn * off)
                pf = profiles_v2.gn_refit(make_eval(th_f, which), ref["x"], lo_b, hi_b,
                                          J_fixed=J, n_iter=n_iter, trust=0.5, tol=1e-6)
                rows.append({"offset_outward": off,
                             "dprofile_x_N_over_s2": (pf["obj"] - ref["obj"]) / s2,
                             "stop_reason": pf["reason"], "n_iter": pf["n_iter"],
                             "n_eval": pf["n_eval"],
                             "max_abs_shift_from_reference": float(
                                 np.abs(pf["x"] - ref["x"]).max()),
                             "predicted_remaining_relative_decrease":
                                 profiles_v2.predicted_remaining(J, pf["res"], pf["x"], lo_b,
                                                                 hi_b)})
            rows.sort(key=lambda r: r["offset_outward"])
            prof.append({"name": p.name, "index": p.index, "side": p.side, "rows": rows})
        blk["profiles"] = prof
        blk["t_profiles_s"] = round(time.perf_counter() - t0, 1)
        log(f"profiles v2 [{which}]: profiles in {blk['t_profiles_s']}s")
        out[which] = blk
    out["method"] = ("reference: bounded Gauss-Newton, central-difference Jacobian (step "
                     "1e-3) recomputed at every accepted iterate, trust box 0.5 per iteration, "
                     f"up to {n_iter} iterations, stop at relative decrease < 1e-6; profiles: "
                     "chord Gauss-Newton from the converged reference with its Jacobian, same "
                     "box, iteration cap and tolerance; slices: every other entry at theta-hat")
    out["n_model_evaluations"] = n_eval[0]
    out["t_s"] = round(time.perf_counter() - t_all, 1)
    return out


# ============================================================================ manifest
def write_manifest(vdir: Path, V: dict, meta: dict, d: dict) -> None:
    import glob

    files, dirs = inputs_p4.input_files(meta)
    shp = sorted(glob.glob(str(REPO / "chou-shui-data/data/water/river_TWD97.*")))
    inputs = ([rel_to_repo(be.THETA), rel_to_repo(be.RIV_THETA), rel_to_repo(TWIN_GRID_NPZ),
               rel_to_repo(be.COUNTIES), rel_to_repo(be.PILOT / "inputs_cwa.npz"),
               "results/twin/basemap.npz", rel_to_repo(rts.FIT_SUMMARY),
               rel_to_repo(rts.FOLD_THETAS), rel_to_repo(rts.JOBS)]
              + files + [rel_to_repo(p) for p in shp])
    inputs = list(dict.fromkeys(inputs))
    roles = {rel_to_repo(be.THETA): "calibrated parameters: batch 8's merged cwa theta",
             rel_to_repo(be.RIV_THETA): "river conductances for the RIV verification config "
                                        "only",
             rel_to_repo(be.PILOT / "inputs_cwa.npz"): "design-time cross-check of the inputs "
                                                        "(identity), not an input to any output",
             rel_to_repo(rts.FIT_SUMMARY): "theta-status evidence: the full fit's parameters",
             rel_to_repo(rts.FOLD_THETAS): "theta-status evidence: the merged fold thetas",
             rel_to_repo(rts.JOBS): "theta-status evidence: batch 8's merge dependencies"}
    here = Path(__file__).resolve().parent
    code = [here / "build_engine.py", here / "engine.py", here / "inputs_p4.py",
            here / "__init__.py", here / "build_engine_v2.py", here / "engine_tlm.py",
            here / "profiles_v2.py", here / "repair_theta_status.py", here / "diag_riv.py",
            REPO / "dataprep_staging/harness/build_units.py",
            REPO / "hydrophysics/twin/forward.py", REPO / "hydrophysics/twin/flow.py",
            REPO / "hydrophysics/twin/calibrate_flow.py", REPO / "hydrophysics/twin/inputs.py",
            REPO / "hydrophysics/twin/pumping.py", REPO / "hydrophysics/twin/spread.py",
            REPO / "hydrophysics/twin/heads.py", REPO / "hydrophysics/twin/rivers.py",
            REPO / "hydrophysics/twin/scenario.py", REPO / "hydrophysics/twin/boundaries.py",
            REPO / "hydrophysics/twin/zones.py", REPO / "hydrophysics/twin/grid.py"]
    a1 = V["A1"]["cases"]
    rep = V["repair"]
    def pf(ok):
        return "pass" if ok else "FAIL"

    gate_txt = (
        "plan 3 WP P4.0 acceptance as pre-registered (P40.engine_acceptance.v1): A1 sparse vs "
        "torch <= 1e-5 m on the acceptance case (the deliverable forcing, "
        f"{a1['base']['max_abs_dh_m']:.3g} m: {pf(V['A1']['passed_base'])}) and A2 adjoint vs "
        "forward <= 1e-6 relative in float64 and for the stored float32 kernels "
        f"({pf(V['A2']['passed'])}). Reported against the same A1 threshold: the synthetic "
        f"Changhua case ({a1['extra']['max_abs_dh_m']:.3g} m: "
        f"{pf(a1['extra']['max_abs_dh_m'] <= be.TOL_A1_M)}) and the RIV verification "
        f"configuration ({a1['riv']['max_abs_dh_m']:.3g} m: "
        f"{pf(a1['riv']['max_abs_dh_m'] <= be.TOL_A1_M)}; the miss is traced post hoc to the "
        "reference's CG truncation, validation.json repair.riv_diagnostic). Repair declaration "
        "P40.repair_linearisation.v1 (tangent-linear model, not part of the P4.0 gate): "
        f"{pf(rep['acceptance']['all'])}.")
    m = manifest.build_manifest(
        vdir, product=be.PRODUCT, layer="L3",
        inputs=[{"path": p, **({"role": roles[p]} if p in roles else {})} for p in inputs],
        input_trees=[{"path": x, "role": "head records read by build_head_field"} for x in dirs],
        code=[rel_to_repo(c) for c in code],
        parameters={"theta": {k: v for k, v in V["theta"].items() if k != "status_evidence"},
                    "n_lags": V["kernels"]["n_lags"], "layers": [0, 1, 2, 3],
                    "kernel_dtype": "float32", "grid_hash": d["grid_hash"],
                    "mask_hash": d["mask_hash"], "months": V["inputs"]["months"],
                    "census_columns_read": inputs_p4.CENSUS_COLS,
                    "datum_convention": V["datum"]["convention"],
                    "thresholds": {"A1_m": be.TOL_A1_M, "A2_rel": be.TOL_A2_REL,
                                   "R1_rel": TOL_R1, "R2": "chi2 95 % band",
                                   "R3": list(R3_BAND)},
                    "prereg": ["results/dataprep/preregistration/P40.engine_acceptance.v1.json",
                               f"results/dataprep/preregistration/{PREREG_ID}.json"],
                    "layer_note": "model response functions and operator of the calibrated "
                                  "twin on the twin grid; no observed values",
                    "linearisation_note": "the stored kernels are the frozen-lift "
                                          "linearisation; the lift feedback is first order "
                                          "(engine_tlm.tlm_rollout gives the exact first-order "
                                          "response); see validation.json repair.R1-R3",
                    "files": {"G_layer<l>_f32.npy": "(158 wells, 2148 cells, n_lags) m per "
                                                    "m3/day, unit source in layer l",
                              "K_pump_f32.npy": "drawdown kernel, m per m3/day abstracted "
                                                "(pump split and return flow of the theta)",
                              "M_operator.npz": "scipy.sparse backward-Euler operator",
                              "engine_arrays.npz": "operator pieces, hindcast heads, base "
                                                   "abstraction rates, datum, steady gains"}},
        fit_window_end="2022-12",
        validation_report="validation.json",
        gate={"passed": V["passed"], "criteria": gate_txt,
              "A1_cases": {k: {"max_abs_dh_m": a1[k]["max_abs_dh_m"],
                               "passed": a1[k]["max_abs_dh_m"] <= be.TOL_A1_M,
                               "role": "acceptance case" if k == "base" else "reported case"}
                           for k in ("base", "extra", "riv")},
              "A1_all_reported_cases_passed": V["A1"]["passed_all"],
              "A2_passed": V["A2"]["passed"],
              "repair_linearisation_declaration": rep["acceptance"]},
        allowed_uses=["model_design", "selection", "design_time_evidence",
                      "after_the_fact_analysis"],
        forbidden_uses=["scoring_target", "baseline", "datum_fit", "calibration_target",
                        "prospective_scorer", "prior", "prior_width_input", "forcing_frozen",
                        "forcing_recorded_arm", "initial_conditions", "idw_field", "validator",
                        "statistical_layer_history"],
        personal_data="none: kernels and operator fields carry no personal data; Q_rate holds "
                      "1 km cell aggregates of the spread, cleaned census abstraction (no "
                      "names, addresses, meter ids or pump coordinates); well ids are public "
                      "WRA station ids",
        extra={"supersedes": {"version": "v1", "reason": "v1's manifest and validation.json "
                              "labelled the theta provisional ('batch 8's merged theta does "
                              "not exist yet'); it was batch 8's merged theta (same sha256). "
                              "Same theta, inputs and kernel code; v1 is kept unchanged.",
                              "identity_with_v1": {k: v for k, v in rep.get(
                                  "identity_with_v1", {}).items() if k != "files"}}})
    manifest.write_manifest(vdir, m)


if __name__ == "__main__":
    raise SystemExit(main())
