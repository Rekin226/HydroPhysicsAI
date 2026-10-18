"""WP P4.0 build: the linear-response engine on the cwa theta, its acceptance checks, and the
kernel product ``results/dataprep/changhua_abstraction/engine/v<N>/``.

    cd dataprep_staging
    CUDA_VISIBLE_DEVICES= tools/cpu_slot.sh --slots 2 python -m p4_changhua.build_engine

Steps (pre-registration ``P40.engine_acceptance.v1`` fixes the thresholds and cases):

1. inputs: the twin's inputs for the theta's recipe (``inputs_p4.load_inputs``: the repository
   loaders, census columns explicit), cross-checked against the scout's ``inputs_cwa.npz``;
2. operator: ``M`` assembled from the rebuilt torch model, checked against its matvec;
3. A1: sparse rollout vs ``forward.rollout`` (CPU torch) for the deliverable's forcing, for
   the forcing plus a synthetic Changhua abstraction (lift feedback on the added energy) and
   for a MODFLOW-RIV configuration with canal recharge (time-varying operator);
4. kernels: adjoint Green's functions for all 4 layers, ``N_LAGS`` lags, the pump kernel,
   steady gains (truncation tail), stored float32;
5. A2: kernels vs forward impulses (Changhua uniform, single cells in every layer), the
   convolution engine vs the stepwise linear rollout, and the frozen-lift energy pathway;
6. datum: the engine's ``fit_flow`` datum prior vs ``calibrate_flow._profile_well_datum``
   and the theta's recorded datum; R2 in sample;
7. forcing variance: factor responses (exact), independent-noise ``K Sigma K^T`` vs Monte
   Carlo through the linear rollout, and a lift-feedback Monte Carlo through the full
   stepwise rollout against the linearised ``K Sigma K^T``;
8. bound-pinned parameters: listed against the theta's ``bounds_hit``; exact slices inside
   and beyond each bound; finite-difference Gauss-Newton Jacobian of the interior entries,
   their Fisher sd with the pinned entries held (uncalibrated), and chord Gauss-Newton
   profiles of each pinned entry with the interior entries re-optimised.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse  # noqa: E402
import datetime as _dt  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import scipy.sparse.linalg as spla  # noqa: E402

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from common import manifest  # noqa: E402
from common.hashing import grid_hash, mask_hash, sha256_file  # noqa: E402
from common.paths import DATAPREP, REPO, REPORTS_DIR, TWIN_GRID_NPZ, rel_to_repo  # noqa: E402

from p4_changhua import __version__, engine, inputs_p4  # noqa: E402

torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))

THETA = REPO / "results/twin_runs/stage3_datum_gate_cwa/stage3_theta.json"
RIV_THETA = REPO / "results/twin_runs/temporal_rivers/stage3_theta.json"
PILOT = REPO / "results/dataprep/_pilots/physics_inversion_hooks"
COUNTIES = REPO / "chou-shui-data/data/water/Taiwan_countyV02/TW_COUNTY.gpkg"
PRODUCT = "changhua_abstraction/engine"
EVID = REPORTS_DIR / "P40_evidence"
N_LAGS = 180            # 2012-01 .. 2026-12: covers every month of the 2012-26 record
CH, YL = "彰化縣", "雲林縣"
TOL_A1_M = 1e-5
TOL_A2_REL = 1e-6
SEED = 2026
CH_GWH_YR = 188.0       # in-grid Changhua cleaned census, 2012-22 (plan revision note)

LOG = []


def log(msg: str) -> None:
    line = f"[{_dt.datetime.now().strftime('%H:%M:%S')}] {msg}"
    LOG.append(line)
    print(line, flush=True)


def rel_err(got: np.ndarray, ref: np.ndarray, floor: float = 1e-3) -> dict:
    """Scale-relative max error (the acceptance metric) and the elementwise relative error
    over entries with |ref| >= floor x max|ref|."""
    got, ref = np.asarray(got, np.float64), np.asarray(ref, np.float64)
    d = np.abs(got - ref)
    scale = float(np.abs(ref).max())
    sel = np.abs(ref) >= floor * scale
    return {"scale_rel": float(d.max() / scale), "max_abs": float(d.max()), "scale": scale,
            "elem_rel_max": float((d[sel] / np.abs(ref[sel])).max()) if sel.any() else 0.0,
            "n": int(ref.size)}


# ============================================================================ inputs
def load_everything(meta: dict) -> dict:
    from harness.build_units import cell_counties

    t0 = time.perf_counter()
    inp = inputs_p4.load_inputs(meta)
    log(f"inputs loaded in {time.perf_counter() - t0:.1f}s: {inp.grid.n_active} cells, "
        f"{len(inp.sids)} wells, {len(inp.dates)} months")
    g = inp.grid
    z = np.load(TWIN_GRID_NPZ, allow_pickle=False)
    same_grid = ((g.nx, g.ny) == (int(z["nx"]), int(z["ny"]))
                 and abs(g.dx - float(z["dx"])) < 1e-9 and abs(g.x0 - float(z["x0"])) < 1e-6
                 and abs(g.y0 - float(z["y0"])) < 1e-6
                 and np.array_equal(g.mask, np.asarray(z["mask"], bool)))
    cc, _, n_miss = cell_counties(g.centroids())
    stn = inp.stations.set_index("sid")
    wc = np.array([str(stn.loc[s, "CountyName"]) if s in stn.index else "" for s in inp.sids])
    E_cls = np.stack([inp.E_by_class[c] for c in inp.E_by_class])
    d = {"inp": inp, "grid": g, "same_grid": bool(same_grid),
         "grid_hash": grid_hash(g.nx, g.ny, g.dx, g.x0, g.y0, g.mask),
         "mask_hash": mask_hash(g.mask), "cell_county": cc, "cell_county_nearest": n_miss,
         "well_county": wc, "E_cls": E_cls, "classes": list(inp.E_by_class),
         "E_tot": E_cls.sum(axis=0)[:, 1:], "R": inp.recharge_field.numpy()[:, 1:],
         "ge": inp.ground_elev.numpy(), "h0": inp.initial_heads(0, n_layers=4).numpy(),
         "oi": inp.obs_idx, "ol": inp.obs_layer, "sids": np.array(inp.sids),
         "obs": inp.obs_h[:, 1:], "obs_filled": inp.obs_h_filled[:, 1:],
         "dates": np.array([str(x.date()) for x in inp.dates])}
    # design-time cross-check: the scout's load_twin_inputs output (all census columns)
    pc = {}
    with np.load(PILOT / "inputs_cwa.npz", allow_pickle=False) as p:
        pairs = {"E_cls": E_cls, "recharge": inp.recharge_field.numpy(), "h0": d["h0"],
                 "obs_h": inp.obs_h, "obs_h_filled": inp.obs_h_filled, "obs_idx": inp.obs_idx,
                 "obs_layer": inp.obs_layer, "ground_elev": d["ge"], "mask": g.mask,
                 "well_xy": inp.well_xy}
        for k, v in pairs.items():
            a, b = np.asarray(p[k]), np.asarray(v)
            pc[k] = bool(a.shape == b.shape and np.array_equal(a, b, equal_nan=True))
        pc["sids"] = bool(np.array_equal(p["sids"].astype(str), d["sids"]))
        pc["cell_county"] = bool(np.array_equal(p["cell_county"].astype(str), cc))
        pc["well_county"] = bool(np.array_equal(p["well_county"].astype(str), wc))
        pc["classes"] = bool(list(p["classes"].astype(str)) == d["classes"])
    d["pilot_inputs_identical"] = pc
    log(f"grid equals {rel_to_repo(TWIN_GRID_NPZ)}: {same_grid}; identical to the scout's "
        f"inputs: {all(pc.values())} ({sum(pc.values())}/{len(pc)})")
    return d


# ============================================================================ helpers
def build(member, grid, ge, **mults):
    from hydrophysics.twin.forward import build_model

    model, sc, zone3 = build_model(grid, member, "cpu")
    return model, sc, zone3, engine.SparseTwin(model, sc, ge, **mults)


def synthetic_changhua_energy(d: dict) -> np.ndarray:
    """A synthetic added abstraction (kWh/month, (A, T)): uniform over Changhua's in-grid
    cells, total ``CH_GWH_YR``, with the Yunlin census's own monthly shape (2012-22)."""
    ch = d["cell_county"] == CH
    yl = d["cell_county"] == YL
    shape = d["E_tot"][yl].sum(axis=0)
    shape = shape / shape.mean()
    per_cell = CH_GWH_YR * 1e6 / 12.0 / ch.sum()
    Ex = np.zeros_like(d["E_tot"])
    Ex[ch] = per_cell * shape[None, :]
    return Ex


def riv_member(member):
    """cwa theta + the temporal_rivers MODFLOW-RIV recipe and conductances + a canal-recharge
    fraction of 0.25: a verification configuration, not a calibrated model."""
    import copy

    from hydrophysics.twin.forward import Member

    rv = json.loads(RIV_THETA.read_text())
    th = copy.deepcopy(member.theta)
    th["log_C_riv"] = list(rv["theta"]["log_C_riv"])
    th["log_sw_scale"] = math.log(0.25)
    meta = dict(member.meta)
    for k in ("rivers", "river_set", "river_layer", "river_stage_depth", "river_rbot_depth",
              "river_groups", "river_c_split", "river_skip_apex", "river_stage_season",
              "dem_npz"):
        meta[k] = rv["meta"].get(k)
    meta["river_shp"] = None            # the repository default path
    meta["sw_layer"] = None
    return Member(label="cwa+riv(verification)", theta=th, meta=meta)


# ============================================================================ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-lags", type=int, default=N_LAGS)
    ap.add_argument("--mc-draws", type=int, default=200)
    ap.add_argument("--skip-profiles", action="store_true")
    ap.add_argument("--scratch", default=None,
                    help="trial run: write everything under this directory; no product "
                         "version, no manifest, no evidence files")
    args = ap.parse_args(argv)
    global EVID
    if args.scratch:
        EVID = Path(args.scratch).resolve() / "evidence"
    t_start = time.perf_counter()
    os.chdir(REPO)
    if torch.cuda.is_available():
        raise SystemExit("CUDA is visible; run with CUDA_VISIBLE_DEVICES=''")
    EVID.mkdir(parents=True, exist_ok=True)
    root = Path(args.scratch).resolve() if args.scratch else DATAPREP / PRODUCT
    stage = root / f"_build_{_dt.datetime.now().strftime('%Y%m%dT%H%M%S')}"
    stage.mkdir(parents=True)
    V: dict = {"wp": "P40", "engine_version": __version__, "created": _dt.date.today().isoformat(),
               "torch_threads": torch.get_num_threads(), "cuda_visible": torch.cuda.is_available(),
               "prereg": "P40.engine_acceptance.v1"}
    from hydrophysics.twin import calibrate_flow as cf
    from hydrophysics.twin.forward import load_members

    assert engine.RETURN_FRAC_MAX == cf.RETURN_FRAC_MAX
    assert engine.WELL_DATUM_SD == cf.WELL_DATUM_SD_DEFAULT
    assert engine.WELL_DATUM_MONTHS_PER_OBS == cf.WELL_DATUM_MONTHS_PER_OBS
    theta_sha = sha256_file(THETA)
    member = load_members([str(THETA)])[0]
    meta = member.meta
    V["theta"] = {"path": rel_to_repo(THETA), "sha256": theta_sha,
                  "mtime_utc": _dt.datetime.utcfromtimestamp(THETA.stat().st_mtime).isoformat(),
                  "git_commit_in_meta": meta.get("git_commit"),
                  "r2_insample_meta": meta.get("r2_insample"),
                  "gate_in_meta": meta.get("gate"),
                  "status": "provisional: batch 8's merged theta not confirmed by the task; "
                            "the file's meta carries a 5-fold gate"}
    log(f"theta {rel_to_repo(THETA)} sha256 {theta_sha}")

    # ---------------------------------------------------------------- 1. inputs
    d = load_everything(meta)
    V["inputs"] = {"n_active": int(d["grid"].n_active), "n_wells": int(len(d["sids"])),
                   "months": [d["dates"][0], d["dates"][-1]], "n_steps": int(d["E_tot"].shape[1]),
                   "grid_equals_twin_grid_npz": d["same_grid"], "grid_hash": d["grid_hash"],
                   "mask_hash": d["mask_hash"], "classes": d["classes"],
                   "census_columns_read": inputs_p4.CENSUS_COLS,
                   "pilot_inputs_identical": d["pilot_inputs_identical"],
                   "cells_by_county": {k: int((d["cell_county"] == k).sum())
                                       for k in sorted(set(d["cell_county"]))},
                   "wells_by_county": {k: int((d["well_county"] == k).sum())
                                       for k in sorted(set(d["well_county"]))},
                   "cell_county_nearest_polygon": int(d["cell_county_nearest"])}
    grid, ge, h0, E, R = d["grid"], d["ge"], d["h0"], d["E_tot"], d["R"]
    oi, ol = d["oi"], d["ol"]
    A, nT = grid.n_active, E.shape[1]
    ch = d["cell_county"] == CH

    # ---------------------------------------------------------------- 2. operator
    t0 = time.perf_counter()
    model, sc, zone3, tw = build(member, grid, ge)
    t_build = time.perf_counter() - t0
    opchk = engine.verify_operator(tw.op, model)
    t0 = time.perf_counter()
    lu = tw.lu()
    t_lu = time.perf_counter() - t0
    ev_max = spla.eigsh(tw.op.M, k=1, which="LA", return_eigenvectors=False)[0]
    ev_min = spla.eigsh(tw.op.M, k=1, sigma=0, which="LM", return_eigenvectors=False)[0]
    V["operator"] = {**opchk, "lu_fill_nnz": int(lu.L.nnz + lu.U.nnz),
                     "cond": float(ev_max / ev_min), "lambda": [float(ev_min), float(ev_max)],
                     "scalars": {"eta": tw.eta, "head_extra_m": tw.hx, "f_shallow": tw.f_sh,
                                 "return_frac": tw.r_ret, "recharge_frac": tw.rfrac,
                                 "spread_km": float(member.theta.get("spread_km", float("nan"))),
                                 "pump_weights": tw.pump_weights()},
                     "rivers": tw.has_rivers}
    log(f"operator: {opchk}; LU {t_lu:.3f}s; cond {ev_max / ev_min:.3e}")

    # ---------------------------------------------------------------- 3. A1 (torch)
    t0 = time.perf_counter()
    H_base, Q_base = tw.rollout(h0, E=E, R=R, return_rates=True)
    t_roll = time.perf_counter() - t0
    a1 = {}
    r = engine.verify_against_torch(tw, sc, h0, E, R, H_sparse=H_base)
    r.pop("H_torch")
    a1["base"] = {**r, "t_sparse_s": round(t_roll, 3)}
    log(f"A1 base: max|dh| {r['max_abs_dh_m']:.3e} m (torch {r['t_torch_s']}s, sparse "
        f"{t_roll:.3f}s)")
    Ex = synthetic_changhua_energy(d)
    H_extra, Q_extra = tw.rollout(h0, E=E, R=R, E_extra=Ex, return_rates=True)
    r = engine.verify_against_torch(tw, sc, h0, E, R, E_extra=Ex, H_sparse=H_extra)
    r.pop("H_torch")
    a1["extra"] = {**r, "E_extra_GWh_per_yr": float(Ex.sum() / 1e6 / (nT / 12)),
                   "n_cells": int(ch.sum())}
    log(f"A1 extra (synthetic Changhua {a1['extra']['E_extra_GWh_per_yr']:.1f} GWh/yr): "
        f"max|dh| {r['max_abs_dh_m']:.3e} m")
    mr = riv_member(member)
    model_r, sc_r, zone3_r, tw_r = build(mr, grid, ge)
    mid = zone3_r == 1
    months = np.arange(1, nT + 1) % 12          # step t is calendar month (t+1) % 12, Jan = 0
    sw = np.zeros((A, nT))
    sw[mid] = 2e-3 * np.clip(np.sin(2 * np.pi * (months - 2) / 12.0), 0, None)[None, :]
    t0 = time.perf_counter()
    H_riv = tw_r.rollout(h0, E=E, R=R, sw=sw, month0=1)
    t_riv = time.perf_counter() - t0
    r = engine.verify_against_torch(tw_r, sc_r, h0, E, R, sw=sw, month0=1, H_sparse=H_riv)
    r.pop("H_torch")
    conn = H_riv[tw_r.riv_layer][tw_r.riv_idx][:, :-1] > tw_r.riv_rbot[:, None]
    n_switch = int((np.diff(conn.astype(int), axis=1) != 0).sum())
    a1["riv"] = {**r, "t_sparse_s": round(t_riv, 3), "n_river_cells": int(tw_r.riv_idx.size),
                 "groups": list(model_r.rivers.names), "n_factorisations": tw_r.n_factorisations,
                 "connected_fraction": float(conn.mean()), "n_switch_events": n_switch,
                 "sw_synthetic": "2 mm/day x max(0, sin(2 pi (month - 2) / 12)) on mid-zone cells, "
                                 "fraction 0.25"}
    log(f"A1 riv: max|dh| {r['max_abs_dh_m']:.3e} m; {tw_r.n_factorisations} factorisations, "
        f"{n_switch} switch events, connected {conn.mean():.3f}")
    a1_pass = all(v["max_abs_dh_m"] <= TOL_A1_M for v in a1.values())
    V["A1"] = {"threshold_m": TOL_A1_M, "cases": a1, "passed_base": a1["base"]["max_abs_dh_m"]
               <= TOL_A1_M, "passed_all": a1_pass}
    del model_r, tw_r

    # ---------------------------------------------------------------- 4. kernels
    nL = args.n_lags
    t0 = time.perf_counter()
    G = engine.adjoint_greens(tw, oi, ol, nL, layers=(0, 1, 2, 3), dtype=np.float64)
    t_greens = time.perf_counter() - t0
    wts = tw.pump_weights()
    K = engine.pump_kernel(G, wts, dtype=np.float64)
    log(f"adjoint Green's functions: {len(oi)} wells x {A} cells x {nL} lags x 4 layers in "
        f"{t_greens:.1f}s")
    gain = engine.steady_gain(tw, oi, ol, layers=(0, 1, 2, 3))
    K_steady = -sum(float(w) * gain[lay] for lay, w in wts.items())

    def tail(n):
        t = K_steady - K[:, :, :n].sum(axis=2)
        return {"max_rel_to_max_gain": float(np.abs(t).max() / np.abs(K_steady).max()),
                "median_well_l1_share": float(np.median(np.abs(t).sum(1)
                                                        / np.abs(K_steady).sum(1)))}

    wch = d["well_county"] == CH
    cum = np.cumsum(K, axis=2) / np.where(np.abs(K_steady) > 0, K_steady, 1)[:, :, None]
    gch = np.abs(K_steady[np.ix_(wch, ch)])
    big = gch > np.percentile(gch, 90)
    quasi = {f"lag{lag}": float(np.median(cum[np.ix_(wch, ch)][:, :, lag][big]))
             for lag in (0, 1, 2, 5, 11, 23)}
    del cum
    V["kernels"] = {"n_lags": nL, "layers": [0, 1, 2, 3], "pump_weights": wts,
                    "t_greens_s": round(t_greens, 1),
                    "tail_beyond_131": tail(131), "tail_beyond_n_lags": tail(nL),
                    "changhua_quasi_steady_share_top_decile": quasi}
    log(f"kernel tail beyond 131 lags {V['kernels']['tail_beyond_131']}, beyond {nL} "
        f"{V['kernels']['tail_beyond_n_lags']}; quasi-steady {quasi}")

    # ---------------------------------------------------------------- 5. A2
    def a2_cases(Kx, Gx, label):
        out = {}
        q0 = tw.layer_sources(ch.astype(np.float64))
        fwd = engine.forward_impulse(tw, q0, nL, oi, ol)
        adj = -np.asarray(Kx[:, ch, :], np.float64).sum(axis=1)
        out["changhua_uniform"] = rel_err(adj, fwd)
        rng = np.random.default_rng(SEED)
        worst = {"scale_rel": 0.0}
        per = []
        for lay in range(4):
            for c in rng.choice(A, size=8, replace=False):
                qq = np.zeros((4, A))
                qq[lay, c] = 1.0
                f = engine.forward_impulse(tw, qq, nL, oi, ol)
                e = rel_err(np.asarray(Gx[lay][:, c, :], np.float64), f)
                per.append({"layer": lay, "cell": int(c), "scale_rel": e["scale_rel"]})
                if e["scale_rel"] >= worst["scale_rel"]:
                    worst = {**e, "layer": lay, "cell": int(c)}
        out["single_cells_worst"] = worst
        out["single_cells_n"] = len(per)
        rate = np.random.default_rng(SEED + 1).uniform(0, 1e3, (A, nT))
        q = np.stack([tw.layer_sources(rate[:, t]) for t in range(nT)], axis=-1)
        Hl = tw.rollout_linear(q)[..., 1:][ol, oi, :]
        out["random_field_131"] = rel_err(-engine.convolve(Kx, rate), Hl)
        out["frozen_lift_energy"] = rel_err(-engine.convolve(Kx, dq_lin), dh_frozen)
        out["passed"] = all(v["scale_rel"] <= TOL_A2_REL for k, v in out.items()
                            if isinstance(v, dict) and "scale_rel" in v)
        log(f"A2 {label}: " + ", ".join(f"{k} {v['scale_rel']:.2e}" for k, v in out.items()
                                        if isinstance(v, dict) and "scale_rel" in v))
        return out

    t0 = time.perf_counter()
    H_extra_frozen = tw.rollout(h0, E=E, R=R, E_extra=Ex, frozen_head=H_base[1, :, :-1])
    H_base_frozen = tw.rollout(h0, E=E, R=R, frozen_head=H_base[1, :, :-1])
    frozen_equals_base = float(np.abs(H_base_frozen - H_base).max())
    dh_frozen = (H_extra_frozen - H_base_frozen)[ol, oi, 1:]
    c_rate = engine.energy_rate_jacobian(tw, H_base)
    dq_lin = c_rate * tw.spread(Ex)
    a2_64 = a2_cases(K, G, "float64")
    # store float32, reload memory-mapped, re-check
    for lay in range(4):
        np.save(stage / f"G_layer{lay}_f32.npy", G[lay].astype(np.float32))
    np.save(stage / "K_pump_f32.npy", K.astype(np.float32))
    K32 = np.load(stage / "K_pump_f32.npy", mmap_mode="r")
    G32 = {lay: np.load(stage / f"G_layer{lay}_f32.npy", mmap_mode="r") for lay in range(4)}
    a2_32 = a2_cases(K32, G32, "float32 (stored)")
    t_a2 = time.perf_counter() - t0
    t0 = time.perf_counter()
    _ = engine.convolve(K32, Q_base)
    t_conv = time.perf_counter() - t0
    # lift feedback on the added abstraction: stepwise (feedback) vs frozen lift / kernel
    dh_step = (H_extra - H_base)[ol, oi, 1:]
    fb = dh_step - dh_frozen
    V["A2"] = {"threshold_rel": TOL_A2_REL, "float64": a2_64, "float32_stored": a2_32,
               "passed": bool(a2_64["passed"] and a2_32["passed"]),
               "frozen_head_rollout_equals_base_max_abs_m": frozen_equals_base}
    V["lift_feedback_diagnostic"] = {
        "case": f"synthetic Changhua abstraction {a1['extra']['E_extra_GWh_per_yr']:.1f} GWh/yr",
        "max_abs_drawdown_step_m": float(np.abs(dh_step).max()),
        "max_abs_feedback_m": float(np.abs(fb).max()),
        "rms_feedback_m": float(np.sqrt((fb ** 2).mean())),
        "feedback_share_of_rms_response": float(np.sqrt((fb ** 2).mean())
                                                / np.sqrt((dh_step ** 2).mean())),
        "changhua_wells_mean_dh_step_m": float(dh_step[wch].mean()),
        "changhua_wells_mean_dh_frozen_m": float(dh_frozen[wch].mean()),
        "note": "difference between the stepwise rollout (lift feedback on base and added "
                "abstraction) and the kernel/frozen-lift response; the convolution engine "
                "omits it"}
    log(f"lift feedback: {V['lift_feedback_diagnostic']}")

    # ---------------------------------------------------------------- 6. datum
    from hydrophysics.twin.calibrate_flow import _profile_well_datum, _r2, well_datum_vector

    pred = H_base[ol, oi, 1:]
    dmask = np.isfinite(d["obs"])
    target = d["obs_filled"]
    de, kap, s2, n_obs = engine.profile_datum(pred, target, dmask)
    dt_, kt_, s2t, _ = _profile_well_datum(torch.as_tensor(pred), torch.as_tensor(target),
                                           torch.as_tensor(dmask), engine.WELL_DATUM_SD)
    dmeta = well_datum_vector(meta, d["sids"])
    loss, _, _ = engine.fit_flow_level_loss(pred, target, dmask)
    d_free, _, _, _ = engine.profile_datum(pred, target, dmask, sd=np.inf)
    shrink = n_obs / np.maximum(n_obs + kap, 1e-12)
    V["datum"] = {"convention": "fit_flow: N(0, 5 m^2), kappa_i = min(n_i, 12) s^2 / sd^2, "
                                "observed months only",
                  "max_abs_engine_vs_torch_m": float(np.abs(de - dt_.numpy()[:, 0]).max()),
                  "s2_engine_vs_torch": [s2, float(s2t)],
                  "max_abs_engine_vs_theta_meta_m": float(np.abs(de - dmeta).max()),
                  "sigma_within_m": math.sqrt(s2),
                  "sigma_within_meta_m": meta.get("well_datum_stats", {}).get("sigma_within_m"),
                  "shrink_min": float(shrink[n_obs > 0].min()),
                  "shrink_median": float(np.median(shrink[n_obs > 0])),
                  "level_loss_with_datum": loss,
                  "r2_insample": float(_r2(pred, target)),
                  "r2_insample_meta": meta.get("r2_insample"),
                  "r2_with_datum": float(_r2(pred + de[:, None], target)),
                  "r2_with_datum_meta": meta.get("well_datum_stats", {}).get(
                      "r2_insample_with_datum"),
                  "free_datum_max_abs_diff_m": float(np.abs(d_free - de).max()),
                  "free_datum_median_abs_diff_m": float(np.median(np.abs(d_free - de)))}
    log(f"datum: {V['datum']}")

    # ---------------------------------------------------------------- 7. forcing variance
    rng = np.random.default_rng(SEED + 2)
    tt = np.arange(nT)
    F3 = np.zeros((3, A, nT))
    F3[0][ch] = 100.0 * np.cos(2 * np.pi * tt / 12)[None, :]
    F3[1][ch] = 100.0 * np.sin(2 * np.pi * tt / 12)[None, :]
    F3[2][ch] = 100.0
    U3 = engine.factor_responses(K, F3)
    fac_err = []
    for j in range(3):
        qj = np.stack([tw.layer_sources(F3[j][:, t]) for t in range(nT)], axis=-1)
        Hj = tw.rollout_linear(qj)[..., 1:][ol, oi, :]
        fac_err.append(rel_err(-U3[j], Hj)["scale_rel"])
    sd_q = 0.3 * Q_base
    V_ind = engine.head_var_independent(K, sd_q ** 2)
    nd = args.mc_draws
    acc = np.zeros((len(oi), nT))
    acc2 = np.zeros((len(oi), nT))
    t0 = time.perf_counter()
    for _ in range(nd):
        dq = sd_q * rng.standard_normal((A, nT))
        qd = np.stack([tw.layer_sources(dq[:, t]) for t in range(nT)], axis=-1)
        hd = tw.rollout_linear(qd)[..., 1:][ol, oi, :]
        acc += hd
        acc2 += hd ** 2
    V_mc = (acc2 - acc ** 2 / nd) / (nd - 1)
    t_mc1 = time.perf_counter() - t0
    sel = V_ind > 1e-6 * V_ind.max()
    ratio = V_mc[sel] / V_ind[sel]
    # chi-square band for a variance estimated from nd draws
    from scipy.stats import chi2

    lo, hi = chi2.ppf([0.025, 0.975], nd - 1) / (nd - 1)
    # lift feedback Monte Carlo: energy error on the synthetic Changhua abstraction, as
    # independent factors: a 10 % level error per zone and year, and a 10 % error on each
    # zone's annual cycle (cos and sin); xi ~ N(0, I)
    year = (tt + 1) // 12                       # step t is month t+1 of the record
    fac = []
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
    F_E = np.stack(fac)
    S_E = np.stack([tw.spread(f) for f in F_E])          # spread is linear: once per factor
    c_ex = engine.energy_rate_jacobian(tw, H_extra)
    U_E = engine.factor_responses(K, c_ex[None] * S_E)
    V_lin = engine.head_var_factors(U_E)
    Es_base = tw.spread(E) + tw.spread(Ex)
    acc[:] = 0.0
    acc2[:] = 0.0
    t0 = time.perf_counter()
    for _ in range(nd):
        xi = rng.standard_normal(F_E.shape[0])
        hd = tw.rollout(h0, E_extra=Es_base + np.tensordot(xi, S_E, axes=1), R=R,
                        spread_extra=False)[ol, oi, 1:]
        acc += hd
        acc2 += hd ** 2
    V_nl = (acc2 - acc ** 2 / nd) / (nd - 1)
    t_mc2 = time.perf_counter() - t0
    sel2 = V_lin > 1e-6 * V_lin.max()
    ratio2 = V_nl[sel2] / V_lin[sel2]
    V["forcing_variance"] = {
        "factor_responses_scale_rel": fac_err,
        "independent_noise": {"sd_q": "0.3 x base abstraction rate, independent cell-months",
                              "n_draws": nd, "median_ratio_mc_over_formula": float(
                                  np.median(ratio)),
                              "p05_p95_ratio": [float(np.percentile(ratio, 5)),
                                                float(np.percentile(ratio, 95))],
                              "chi2_95_band": [float(lo), float(hi)],
                              "share_in_band": float(((ratio >= lo) & (ratio <= hi)).mean()),
                              "max_head_sd_m": float(np.sqrt(V_ind.max())),
                              "t_mc_s": round(t_mc1, 1)},
        "lift_feedback_mc": {"energy_error": "synthetic Changhua abstraction: independent "
                                             "10 % level errors per zone and year and 10 % "
                                             "errors on each zone's annual cycle "
                                             f"({F_E.shape[0]} factors)",
                             "n_draws": nd,
                             "median_ratio_nonlinear_mc_over_linear": float(np.median(ratio2)),
                             "p05_p95_ratio": [float(np.percentile(ratio2, 5)),
                                               float(np.percentile(ratio2, 95))],
                             "chi2_95_band": [float(lo), float(hi)],
                             "max_head_sd_linear_m": float(np.sqrt(V_lin.max())),
                             "t_mc_s": round(t_mc2, 1)}}
    log(f"forcing variance: {V['forcing_variance']}")
    del G

    # ---------------------------------------------------------------- 8. bound-pinned
    from hydrophysics.twin.forward import Member

    theta = member.theta
    pinned = engine.bound_pinned(theta, meta)
    counts = engine.pinned_counts(pinned)
    bh = meta.get("bounds_hit", {})
    bh_nz = {z: {b: {"lo": v["lo"], "hi": v["hi"]} for b, v in zz.items()
                 if v.get("lo") or v.get("hi")} for z, zz in bh.items()}
    bh_nz = {z: v for z, v in bh_nz.items() if v}
    entries = engine.theta_vector(theta, meta)
    pin_names = {f"{p.name}[{p.index}]" for p in pinned}
    interior = [e for e in entries if e["name"] not in pin_names]
    V["bound_pinned"] = {"pinned": [p.__dict__ for p in pinned], "counts": counts,
                         "meta_bounds_hit_nonzero": bh_nz, "counts_match_meta": counts == bh_nz,
                         "n_free_entries": len(entries), "n_interior": len(interior)}
    log(f"bound-pinned: {[(p.name, p.index, p.side) for p in pinned]}; match meta "
        f"{counts == bh_nz}")
    target_mask = dmask

    def evaluate(th, want_res=False, kappa_fixed=None):
        mm = Member(label="profile", theta=th, meta=meta)
        _, _, _, twp = build(mm, grid, ge)
        Hp = twp.rollout(h0, E=E, R=R)
        pr = Hp[ol, oi, 1:]
        ls, dd, s2p = engine.fit_flow_level_loss(pr, target, target_mask)
        out = {"loss": ls, "r2": float(_r2(pr, target)), "s2": s2p}
        if want_res:
            kp = kappa_fixed
            n_ = target_mask.sum(axis=1)
            rb = np.where(target_mask, target - pr, 0.0).sum(axis=1) / np.maximum(n_, 1)
            dd = np.where(n_ > 0, n_ * rb / np.maximum(n_ + kp, 1e-12), 0.0)
            out["res"] = np.concatenate([(pr + dd[:, None] - target).ravel(),
                                         np.sqrt(kp) * dd])
        return out

    base_eval = evaluate(theta)
    V["bound_pinned"]["base_loss"] = base_eval["loss"]
    V["bound_pinned"]["base_loss_matches_engine"] = abs(base_eval["loss"] - loss)
    N = float(target.size)
    if not args.skip_profiles:
        t0 = time.perf_counter()
        slices = []
        for p in pinned:
            sgn = 1.0 if p.side == "hi" else -1.0       # outward direction
            rows = []
            for off in (-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0):
                v = p.bound + sgn * off
                ev = evaluate(engine.perturb_theta(theta, p.name, p.index, v))
                rows.append({"offset_outward": off, "value": v, "loss": ev["loss"],
                             "dloss": ev["loss"] - base_eval["loss"],
                             "dloss_x_N_over_s2": (ev["loss"] - base_eval["loss"]) * N / s2,
                             "r2": ev["r2"]})
            slices.append({"name": p.name, "index": p.index, "side": p.side, "rows": rows})
        t_slices = time.perf_counter() - t0
        log(f"slices done in {t_slices:.1f}s")
        # FD Gauss-Newton Jacobian of the interior entries (fixed kappa)
        kap_hat = kap.copy()
        r0 = evaluate(theta, True, kap_hat)["res"]
        h = 1e-3
        J = np.zeros((r0.size, len(interior)))
        t0 = time.perf_counter()
        for j, e in enumerate(interior):
            rp = evaluate(engine.perturb_theta(theta, e["key"], e["index"], e["value"] + h),
                          True, kap_hat)["res"]
            rm = evaluate(engine.perturb_theta(theta, e["key"], e["index"], e["value"] - h),
                          True, kap_hat)["res"]
            J[:, j] = (rp - rm) / (2 * h)
        t_jac = time.perf_counter() - t0
        g = J.T @ r0
        JTJ = J.T @ J
        try:
            step = -np.linalg.solve(JTJ, g)
        except np.linalg.LinAlgError:
            step = np.full(len(interior), np.nan)
        pred_dloss = float((np.sum((r0 + J @ step) ** 2) - np.sum(r0 ** 2)) / N)
        F = JTJ / s2
        try:
            C = np.linalg.inv(F)
            sd = np.sqrt(np.clip(np.diag(C), 0, None))
            corr = C / np.outer(np.where(sd > 0, sd, 1), np.where(sd > 0, sd, 1))
        except np.linalg.LinAlgError:
            sd = np.full(len(interior), np.nan)
            corr = np.full((len(interior), len(interior)), np.nan)
        iu = np.triu_indices(len(interior), 1)
        top = np.argsort(-np.abs(corr[iu]))[:8]
        V["bound_pinned"]["fisher_interior"] = {
            "convention": "fit_flow level loss, iid months of variance s2, datum profiled "
                          "with kappa fixed; pinned entries held at their bounds "
                          "(profiled, no Laplace sd); UNCALIBRATED, overconfident "
                          "(months are autocorrelated, phi ~0.74)",
            "fd_step": h, "t_jacobian_s": round(t_jac, 1),
            "entries": [{"name": e["name"], "value": e["value"], "sd": float(s),
                         "gradient": float(gg)} for e, s, gg in zip(interior, sd, g, strict=True)],
            "cond_JTJ": float(np.linalg.cond(JTJ)),
            "gauss_newton_step_norm": float(np.linalg.norm(step)),
            "gauss_newton_predicted_dloss": pred_dloss,
            "top_correlations": [{"a": interior[iu[0][k]]["name"],
                                  "b": interior[iu[1][k]]["name"],
                                  "corr": float(corr[iu][k])} for k in top]}
        log(f"Jacobian {J.shape} in {t_jac:.1f}s; cond {np.linalg.cond(JTJ):.3e}; GN step "
            f"norm {np.linalg.norm(step):.3e}, predicted dloss {pred_dloss:.3e}")
        # chord Gauss-Newton profiles of each pinned entry
        lo_b = np.array([-np.inf if e["lo"] is None else e["lo"] for e in interior])
        hi_b = np.array([np.inf if e["hi"] is None else e["hi"] for e in interior])
        x_hat = np.array([e["value"] for e in interior])

        def with_interior(th, x):
            th2 = th
            for e, v in zip(interior, x, strict=True):
                th2 = engine.perturb_theta(th2, e["key"], e["index"], float(v))
            return th2

        TRUST = 0.5      # per-iteration box on every interior entry (log / logit units)

        def refit(th_fixed, n_iter=4):
            from scipy.optimize import lsq_linear

            x = x_hat.copy()
            cur = evaluate(with_interior(th_fixed, x), True, kap_hat)
            best_loss = cur["loss"]
            n_eval = 1
            for _ in range(n_iter):
                sol = lsq_linear(J, -cur["res"],
                                 bounds=(np.maximum(lo_b - x, -TRUST),
                                         np.minimum(hi_b - x, TRUST)), method="bvls")
                dx = sol.x
                accepted = False
                for _ls in range(4):
                    trial = evaluate(with_interior(th_fixed, x + dx), True, kap_hat)
                    n_eval += 1
                    if trial["loss"] < best_loss:
                        x, cur, best_loss, accepted = x + dx, trial, trial["loss"], True
                        break
                    dx = dx / 2
                if not accepted:
                    break
            return best_loss, x, n_eval

        t0 = time.perf_counter()
        prof = []
        ref_loss, x_ref, ne_ref = refit(theta)
        for p in pinned:
            sgn = 1.0 if p.side == "hi" else -1.0
            rows = []
            for off in (-0.5, -0.25, 0.0, 0.25, 0.5):
                th_f = engine.perturb_theta(theta, p.name, p.index, p.bound + sgn * off)
                lp, xp, ne = refit(th_f)
                rows.append({"offset_outward": off, "profile_loss": lp,
                             "dprofile": lp - ref_loss,
                             "dprofile_x_N_over_s2": (lp - ref_loss) * N / s2,
                             "n_eval": ne, "max_abs_interior_shift": float(
                                 np.abs(xp - x_hat).max())})
            prof.append({"name": p.name, "index": p.index, "side": p.side, "rows": rows})
        t_prof = time.perf_counter() - t0
        V["bound_pinned"]["slices"] = slices
        V["bound_pinned"]["profiles"] = {"method": "chord Gauss-Newton (fixed FD Jacobian of "
                                                   "the interior entries at theta-hat), 4 "
                                                   "iterations, trust box 0.5 per iteration "
                                                   "within the theta bounds, halving line "
                                                   "search; an approximate profile",
                                         "reference_refit_loss": ref_loss,
                                         "reference_refit_n_eval": ne_ref,
                                         "reference_refit_interior_shift": [
                                             {"name": e["name"], "shift": float(v)}
                                             for e, v in zip(interior, x_ref - x_hat,
                                                             strict=True)
                                             if abs(v) > 0.05],
                                         "reference_refit_dloss": ref_loss - base_eval["loss"],
                                         "rows": prof, "t_s": round(t_prof, 1)}
        log(f"profiles done in {t_prof:.1f}s; refit at theta-hat dloss "
            f"{ref_loss - base_eval['loss']:.3e}")

    # ---------------------------------------------------------------- outputs
    V["timing_s"] = {"build_model_and_operator": round(t_build, 2), "lu": round(t_lu, 3),
                     "sparse_rollout_131": round(t_roll, 3),
                     "torch_rollout_131": a1["base"]["t_torch_s"],
                     "greens_all_layers": round(t_greens, 1),
                     "convolve_all_wells_131": round(t_conv, 2), "a2_checks": round(t_a2, 1),
                     "total": round(time.perf_counter() - t_start, 1)}
    V["acceptance"] = {"A1_sparse_vs_torch": V["A1"]["passed_base"],
                       "A1_all_cases": V["A1"]["passed_all"],
                       "A2_adjoint_vs_forward": V["A2"]["passed"]}
    V["passed"] = bool(V["A1"]["passed_base"] and V["A2"]["passed"])
    sp.save_npz(stage / "M_operator.npz", tw.op.M)
    np.savez_compressed(
        stage / "engine_arrays.npz",
        nx=grid.nx, ny=grid.ny, dx=grid.dx, x0=grid.x0, y0=grid.y0, mask=grid.mask,
        dates=d["dates"], obs_idx=oi, obs_layer=ol, sids=d["sids"].astype(str),
        well_county=d["well_county"].astype(str), cell_county=d["cell_county"].astype(str),
        zone3=zone3, stor=tw.op.stor, bdiag=tw.op.bdiag, brhs=tw.boundary_rhs(h0),
        T=tw.op.T, S=tw.op.S, Lk=tw.op.Lk, ia=tw.op.ia, ib=tw.op.ib, h0=h0, ground_elev=ge,
        H_hindcast=H_base, Q_rate=Q_base, datum=de, datum_kappa=kap,
        steady_gain_l0=gain[0], steady_gain_l1=gain[1], steady_gain_l2=gain[2],
        steady_gain_l3=gain[3], K_steady=K_steady,
        pump_weight_layers=np.array(list(wts.keys())),
        pump_weight_values=np.array(list(wts.values())),
        scalars=np.array([tw.eta, tw.hx, tw.f_sh, tw.r_ret, tw.rfrac]),
        scalar_names=np.array(["eta", "head_extra_m", "f_shallow", "return_frac",
                               "recharge_frac"]))
    (stage / "validation.json").write_text(json.dumps(V, ensure_ascii=False, indent=1,
                                                      default=float) + "\n")
    (EVID / "build_engine.log").write_text("\n".join(LOG) + "\n")
    if args.scratch:
        log(f"trial run written to {stage}; passed {V['passed']}")
        return 0 if V["passed"] else 1
    # move into a fresh version directory and write the manifest
    vdir = manifest.new_version_dir(PRODUCT)
    for f in sorted(stage.iterdir()):
        shutil.move(str(f), vdir / f.name)
    stage.rmdir()
    files, dirs = inputs_p4.input_files(meta)
    shp = sorted(glob.glob(str(REPO / "chou-shui-data/data/water/river_TWD97.*")))
    inputs = ([rel_to_repo(THETA), rel_to_repo(RIV_THETA), rel_to_repo(TWIN_GRID_NPZ),
               rel_to_repo(COUNTIES), rel_to_repo(PILOT / "inputs_cwa.npz"),
               "results/twin/basemap.npz"] + files + [rel_to_repo(p) for p in shp])
    inputs = list(dict.fromkeys(inputs))
    roles = {rel_to_repo(THETA): "calibrated parameters (provisional)",
             rel_to_repo(RIV_THETA): "river conductances for the RIV verification config only",
             rel_to_repo(PILOT / "inputs_cwa.npz"): "design-time cross-check of the inputs "
                                                     "(identity), not an input to any output"}
    code = [Path(__file__), Path(engine.__file__), Path(inputs_p4.__file__),
            Path(__file__).with_name("__init__.py"),
            REPO / "dataprep_staging/harness/build_units.py",
            REPO / "hydrophysics/twin/forward.py", REPO / "hydrophysics/twin/flow.py",
            REPO / "hydrophysics/twin/calibrate_flow.py", REPO / "hydrophysics/twin/inputs.py",
            REPO / "hydrophysics/twin/pumping.py", REPO / "hydrophysics/twin/spread.py",
            REPO / "hydrophysics/twin/heads.py", REPO / "hydrophysics/twin/rivers.py",
            REPO / "hydrophysics/twin/scenario.py", REPO / "hydrophysics/twin/boundaries.py",
            REPO / "hydrophysics/twin/zones.py", REPO / "hydrophysics/twin/grid.py"]
    m = manifest.build_manifest(
        vdir, product=PRODUCT, layer="L3",
        inputs=[{"path": p, **({"role": roles[p]} if p in roles else {})} for p in inputs],
        input_trees=[{"path": x, "role": "head records read by build_head_field"} for x in dirs],
        code=[rel_to_repo(c) for c in code],
        parameters={"theta": V["theta"], "n_lags": nL, "layers": [0, 1, 2, 3],
                    "kernel_dtype": "float32", "grid_hash": d["grid_hash"],
                    "mask_hash": d["mask_hash"], "months": V["inputs"]["months"],
                    "census_columns_read": inputs_p4.CENSUS_COLS,
                    "datum_convention": V["datum"]["convention"],
                    "thresholds": {"A1_m": TOL_A1_M, "A2_rel": TOL_A2_REL},
                    "prereg": "results/dataprep/preregistration/P40.engine_acceptance.v1.json",
                    "layer_note": "model response functions and operator of the calibrated "
                                  "twin on the twin grid; no observed values",
                    "files": {"G_layer<l>_f32.npy": "(158 wells, 2148 cells, n_lags) m per "
                                                    "m3/day, unit source in layer l",
                              "K_pump_f32.npy": "drawdown kernel, m per m3/day abstracted "
                                                "(pump split and return flow of the theta)",
                              "M_operator.npz": "scipy.sparse backward-Euler operator",
                              "engine_arrays.npz": "operator pieces, hindcast heads, base "
                                                   "abstraction rates, datum, steady gains"}},
        fit_window_end="2022-12",
        validation_report="validation.json",
        gate={"passed": V["passed"], "criteria": "plan 3 WP P4.0 acceptance (pre-registration "
              "P40.engine_acceptance.v1): A1 sparse vs torch <= 1e-5 m; A2 adjoint vs forward "
              "<= 1e-6 relative (float64 and stored float32)"},
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
        extra={"provisional": "built on the cwa theta of 2026-10-12 03:06 (sha256 above); "
                              "rebuild (v<N+1>) on batch 8's merged theta once confirmed"})
    manifest.write_manifest(vdir, m)
    V_out = {"version_dir": rel_to_repo(vdir), "passed": V["passed"],
             "acceptance": V["acceptance"]}
    (EVID / "build_result.json").write_text(json.dumps(V_out, indent=1) + "\n")
    log(f"wrote {rel_to_repo(vdir)}; passed {V['passed']}; total "
        f"{time.perf_counter() - t_start:.0f}s")
    (EVID / "build_engine.log").write_text("\n".join(LOG) + "\n")
    return 0 if V["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
