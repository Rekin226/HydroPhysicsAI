"""A0: model-only identifiability audit of the policy quantities (Laplace / Gauss-Newton).

Question: which policy quantities can the EXISTING calibration data identify at all?
For each frozen theta among R0 (stage3_datum_gate_cwa), Q0 (_q0), R4p (_r4p) and R1c
(_ccz), a Laplace approximation of the posterior of the calibrated parameters is built
from the CALIBRATION DESIGN only, then propagated to the policy quantities.

Design (declared before any result was seen; nothing here is scored or registered)
- Flow parameters theta: the free keys of ``twin.uncertainty.flatten`` (fixed eta and
  head_extra, tied county copies and R4p's fixed pump-layer shares held; derived readouts
  dropped). Data: the 2012-02..2022-12 head well-months the fit used, restricted to the
  months actually observed (the fit's loss also saw back-filled copies; those carry no
  information), each well's datum profiled out exactly (Schur complement per well; the
  calibration's datum prior of sd 5 m is ignored, which is conservative).
- Head noise, PRIMARY ("iid"): independent months with each well's own in-sample
  residual sd (within-well, after the fitted datum). No per-well sd is on disk, so it is
  recomputed once from the calibration's own fit (same theta, datum, months and values
  the fit used; no new data) and checked against the published pooled
  ``well_datum_stats.sigma_within_m``. SENSITIVITY ("ar1"): per-well AR(1) with the
  lag-1 autocorrelation of the same calibration residuals (GLS on observed months).
- Column parameters phi (Ske, Skv, tau, h_pc0 per zone): the column JSON the model pairs
  with, fitted second on the leveling benchmarks (calibrate_coupled --target leveling)
  given the flow heads. Design: ``calibrate_coupled.leveling_rows`` (benchmarks, survey
  months, re-zero at each benchmark's first survey), values discarded unread. PRIMARY
  noise: iid with the column JSON's published in-sample leveling RMSE. SENSITIVITY
  ("rw"): random-walk error in the cumulative series (iid increments, sd set so the
  implied cumulative RMS equals the published RMSE; no value read).
- Two-stage (sequential) posterior, as the calibration was sequential:
  Sigma_theta = (J_h' W_h J_h + P_theta)^-1, Sigma_phi = A^-1 + G Sigma_theta G',
  A = J_lphi' W_l J_lphi + P_phi, G = -A^-1 J_lphi' W_l J_ltheta (the column refits when the
  heads move: memory "the column pairs with its head field"). Weak priors centred on the
  fit: sd 2 in log/logit units (as twin.uncertainty), 5 m for h_pc0. Bounds ignored
  (Gaussian; at-bound parameters listed).
- Jacobians: one-sided finite differences, eps 1e-3 (into the interior for a parameter
  on its upper bound), through the frozen model on the CPU; CG tol 1e-8. Column
  Jacobians: central differences on the cheap column recurrence.
- Quantities (free-running model: hindcast 2012-2022, then month-of-year climatology to
  2032-12, ``forward.run`` with gain 0; cut30 = irrigation x0.7 from 2026-01, fan-wide):
  head change cut30 - baseline at the well cells (each well's own layer), mean over the
  Changhua and over the Yunlin wells, at Dec 2026 / Dec 2028 / Dec 2032 (1, 3, 7 years);
  avoided subsidence (baseline - cut30, cm) over the fan / Yunlin / Changhua cells at the
  same months; Changhua T and S per layer (geometric mean over the Changhua cells);
  pump-layer shares where estimated; Ske and Skv per column zone; layer-0 recharge
  conductance (n/a: recharge is a fixed-fraction flux; recharge_frac reported instead).
- Self-validation (a): for R0, the forward-difference columns of 5 parameters against
  central differences (heads and quantities) and against reverse-mode VJPs through
  ``_rollout`` (heads), relative error < 5 %. (b): 50 synthetic replicates per model:
  noise drawn with the design's residual sd, the two-stage linearised (Gauss-Newton) refit,
  and the coverage of the nominal 90 % intervals of every quantity (window 0.85-0.95).
- DECLARED RULE (fixed by the caller): identifiable iff posterior sd/|mean| <= 0.5 AND
  the coverage check passes; otherwise "not identifiable without new data". Positive
  quantities (T, S, Ske, Skv) are Gaussian in log space: sd/mean = sqrt(exp(s^2) - 1).

Usage (CPU only; HYDROMIND_GW_DATA set; from the repo root):
  python -I dataprep_staging/n1_gaps/identifiability.py evals --model R0 --work DIR
  python -I dataprep_staging/n1_gaps/identifiability.py validate --model R0 --work DIR
  python -I dataprep_staging/n1_gaps/identifiability.py analyze --work DIR
Intermediate arrays (one npz per model evaluation) go to ``--work`` (outside the repo);
the JSON results go to results/dataprep/reports/N1_gaps_2026-10-24/A0/ (never overwritten).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from n1_gaps import gaps_lib as G  # noqa: E402

OUT = G.OUT_ROOT / "A0"
MODELS = {
    "R0": ("stage3_datum_gate_cwa", ["coupled_leveling"]),
    "Q0": ("stage3_datum_gate_q0", ["coupled_leveling"]),
    "R4p": ("stage3_datum_gate_r4p", ["coupled_leveling"]),
    "R1c": ("stage3_datum_gate_ccz", ["coupled_leveling_z3", "coupled_leveling"]),
}
EPS = 1e-3
PRIOR_SD = 2.0
PRIOR_SD_HPC0 = 5.0
N_REP = 50
N_REP_DIAG = 2000
Z90 = 1.6448536269514722
COVER_WINDOW = (0.85, 0.95)
RATIO_MAX = 0.5
HORIZON = 120                       # 2023-01 .. 2032-12
SPLIT = 36                          # 2023-01 .. 2025-12 shared by baseline and cut30
YEARS = {"1y_dec2026": 179, "3y_dec2028": 203, "7y_dec2032": 251}   # months from 2012-01
SCEN = "cut30:irrigation=0.7@2026-01"
DERIVED_EXTRA = ("pump_layer_shares", "tau_ret_steps")
VAL_PARAMS = (("log_T_mid", 1), ("log_S_mid", 1), ("log_C_coast", 1),
              ("return_frac_logit", 0), ("recharge_frac_logit", 0))


# ----------------------------------------------------------------------------- set-up
def free_index(member) -> tuple[np.ndarray, list, tuple]:
    from hydrophysics.twin.uncertainty import flatten
    from hydrophysics.twin.zone_county import tied_copy_keys

    fixed = tuple(member.meta.get("fixed") or ()) + tuple(tied_copy_keys(member.theta,
                                                                         member.meta))
    if member.meta.get("fix_eta") is not None:
        fixed += ("log_eta",)
    if member.meta.get("fix_head_extra") is not None:
        fixed += ("log_head_extra",)
    fixed += DERIVED_EXTRA
    if "pump_layer_logits" in member.theta and member.meta.get("fixed_pump_layer_shares"):
        fixed += ("pump_layer_logits",)
    vec, index = flatten(member.theta, fixed=fixed)
    return vec, index, fixed


def load_member(run: str):
    from hydrophysics.twin.forward import load_members

    m = load_members([str(G.RUNS / run / "stage3_theta.json")])[0]
    summ = json.loads((G.RUNS / run / "stage3_fit_summary.json").read_text())
    shares = (summ.get("cfg", {}).get("recipe", {}) or {}).get("pump_layer_shares")
    if shares:
        m.meta["fixed_pump_layer_shares"] = shares
    return m


class Ctx:
    """The frozen model's inputs and the calibration design (no observation kept)."""

    def __init__(self, name: str):
        import torch

        from hydrophysics.config import Config
        from hydrophysics.twin.calibrate_coupled import leveling_rows
        from hydrophysics.twin.forward import (
            attach_sw_recharge,
            build_model,
            extra_hist,
            future_extra,
            future_forcing,
            future_sw,
            parse_scenario,
            sw_hist,
        )
        from hydrophysics.twin.inputs import input_options, load_twin_inputs
        from hydrophysics.twin.scenario import BASELINE
        from hydrophysics.twin.zone_county import DEFAULT_MAP, county_flags_for_grid

        self.torch = torch
        self.name = name
        self.run, self.col_dirs = MODELS[name]
        self.member = m = load_member(self.run)
        self.vec, self.index, self.fixed = free_index(m)
        inp = load_twin_inputs(dx=1000.0, meter_filter=m.meta.get("meter_filter", "none"),
                               cap_duty=float(m.meta.get("cap_duty", 1.0)),
                               **input_options(m.meta))
        attach_sw_recharge(inp, m.meta)
        if m.meta.get("eta_classes"):
            raise SystemExit("per-class eta models are not covered")
        self.inp = inp
        self.h0 = inp.initial_heads(0)
        self.E = inp.E_total[:, 1:]
        self.r = inp.recharge_field[:, 1:]
        self.sw, self.q = sw_hist(inp), extra_hist(inp)
        _, _, zoc = build_model(inp.grid, m, "cpu")
        self.zoc = np.asarray(zoc, int)
        self.fut = {}
        scen_c, rain = parse_scenario(SCEN)
        for key, scen in (("base", BASELINE), ("cut", scen_c)):
            Ef, rf, fd = future_forcing(inp, scen, HORIZON, rain_scale=1.0, zone_of_cell=zoc)
            self.fut[key] = (Ef, rf, fd, future_sw(inp, scen, HORIZON, zoc),
                             future_extra(inp, HORIZON))
        self.month0 = int(self.fut["base"][2][0].month - 1)
        self.ch_cell = np.asarray(county_flags_for_grid(inp.grid, DEFAULT_MAP)[0], bool)
        self.ch_well = self.ch_cell[inp.obs_idx]
        # the leveling design the column used (benchmark cells, survey months); the
        # values returned by the library are discarded unread
        A, T = inp.grid.n_active, len(inp.dates)
        ddir = str(Config().data_dir)
        _H, _O, M, cells, names = leveling_rows(ddir, inp, np.zeros((1, A, T)), np.arange(A))
        del _H, _O
        self.lev_mask = np.asarray(M, bool)
        self.lev_cell = np.asarray(cells, int)
        self.lev_first = self.lev_mask.argmax(axis=1)
        self.lev_ucell, self.lev_row = np.unique(self.lev_cell, return_inverse=True)
        self.columns = []
        for d in self.col_dirs:
            p = G.RUNS / self.run / d / "vep_zonal_leveling.json"
            cj = json.loads(p.read_text())
            zone = self.zoc % 3 if cj.get("column_zones_collapsed") else self.zoc
            if zone.max() >= len(cj["zonal"]):
                raise SystemExit(f"{p}: zone ids {zone.max()} vs {len(cj['zonal'])} sets")
            self.columns.append({"dir": d, "path": p, "json": cj, "zone": zone,
                                 "sets": cj["zonal"]})

    # ------------------------------------------------------------------ one model run
    def evaluate(self, vec: np.ndarray, full: bool = False) -> dict:
        from hydrophysics.twin.forward import Member, build_model, rollout
        from hydrophysics.twin.uncertainty import unflatten

        torch = self.torch
        inp = self.inp
        th = unflatten(vec, self.index, self.member.theta)
        mem = Member(self.member.label, th, self.member.meta)
        model, sc, _ = build_model(inp.grid, mem, "cpu")
        h, st = rollout(model, sc, self.h0, self.E, self.r, inp.ground_elev,
                        sw_field=self.sw, q_extra=self.q, return_state=True)
        h_end = h[..., -1]
        Eb, rb, _, swb, qb = self.fut["base"]
        Ec, rc, _, swc, qc = self.fut["cut"]

        def sl(x, a, b):
            return None if x is None else x[..., a:b]

        h1, st1 = rollout(model, sc, h_end, Eb[:, :SPLIT], rb[:, :SPLIT], inp.ground_elev,
                          sw_field=sl(swb, 0, SPLIT), q_extra=sl(qb, 0, SPLIT), u0=st,
                          return_state=True, month0=self.month0, apex_from=h_end)
        hb = rollout(model, sc, h1[..., -1], Eb[:, SPLIT:], rb[:, SPLIT:], inp.ground_elev,
                     sw_field=sl(swb, SPLIT, None), q_extra=sl(qb, SPLIT, None), u0=st1,
                     month0=0, apex_from=h_end)
        hc = rollout(model, sc, h1[..., -1], Ec[:, SPLIT:], rc[:, SPLIT:], inp.ground_elev,
                     sw_field=sl(swc, SPLIT, None), q_extra=sl(qc, SPLIT, None), u0=st1,
                     month0=0, apex_from=h_end)
        if not (torch.equal(Eb[:, :SPLIT], Ec[:, :SPLIT]) and torch.equal(rb, rc)):
            raise SystemExit("cut30 differs from the baseline before 2026-01")
        base = torch.cat([h, h1[..., 1:], hb[..., 1:]], dim=-1).numpy()
        cut = torch.cat([h, h1[..., 1:], hc[..., 1:]], dim=-1).numpy()
        hn = h.numpy()
        lay, idx = inp.obs_layer, inp.obs_idx
        out = {"wells": hn[lay, idx, 1:],                          # (W, 131) hindcast
               "bench_hm": hn.mean(axis=0)[self.lev_ucell]}         # (nbc, 132)
        ks = list(YEARS.values())
        out["dh_wells"] = cut[lay, idx][:, ks] - base[lay, idx][:, ks]       # (W, 3) m
        mb, mc = base.mean(axis=0), cut.mean(axis=0)                         # (A, 252)
        for ci, c in enumerate(self.columns):
            sb = column_subs(mb, c["zone"], c["sets"])
            scut = column_subs(mc, c["zone"], c["sets"])
            out[f"avoided_{ci}"] = (sb - scut)[:, ks] * 100.0                  # (A, 3) cm
        if full:
            out["path_base"], out["path_cut"] = mb, mc
        return out


def column_subs(hm: np.ndarray, zone: np.ndarray, sets: list[dict],
                override: dict | None = None) -> np.ndarray:
    """The zonal VEP column (``compaction.vep_compaction``) on layer-mean heads (n, T) ->
    compaction (n, T) m; ``override`` = {(zone, key): value}."""
    import torch

    from hydrophysics.twin.compaction import VEP_KEYS, vep_compaction

    h = torch.as_tensor(hm, dtype=torch.float64)
    out = torch.zeros_like(h)
    for z, p in enumerate(sets):
        sel = torch.as_tensor(zone == z)
        if not bool(sel.any()):
            continue
        par = []
        for k in VEP_KEYS:
            v = float(p[k]) if override is None else float(override.get((z, k), p[k]))
            par.append(torch.tensor([v], dtype=torch.float64))
        out[sel] = vep_compaction(h[sel], *par)
    return out.numpy()


def fd_sign(ctx: Ctx) -> np.ndarray:
    from hydrophysics.twin.uncertainty import effective_bounds

    lo, hi = effective_bounds(ctx.index, ctx.member.meta)
    return np.where(np.abs(ctx.vec - hi) < 1e-6, -1.0, 1.0)


def save_npz(path: Path, d: dict) -> None:
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **d)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- evals
def cmd_evals(args) -> int:
    os.chdir(G.REPO)
    work = Path(args.work) / args.model
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    ctx = Ctx(args.model)
    G.log(f"{args.model}: {len(ctx.index)} free parameters; fixed {list(ctx.fixed)}; "
          f"{ctx.lev_mask.shape[0]} leveling benchmarks, {int(ctx.lev_mask.sum())} survey "
          f"entries; {len(ctx.inp.sids)} wells ({time.perf_counter() - t0:.0f}s set-up)")
    sign = fd_sign(ctx)
    dpath = work / "design.npz"
    if not dpath.exists():
        save_npz(dpath, {"vec": ctx.vec, "sign": sign,
                         "index_keys": np.array([k for k, _ in ctx.index]),
                         "index_pos": np.array([i for _, i in ctx.index]),
                         "obs_mask": np.isfinite(ctx.inp.obs_h[:, 1:]),
                         "obs_idx": ctx.inp.obs_idx, "obs_layer": ctx.inp.obs_layer,
                         "sids": np.array(ctx.inp.sids), "ch_well": ctx.ch_well,
                         "ch_cell": ctx.ch_cell, "zoc": ctx.zoc,
                         "lev_mask": ctx.lev_mask, "lev_cell": ctx.lev_cell,
                         "lev_row": ctx.lev_row, "lev_ucell": ctx.lev_ucell})
    base_p = work / "eval_base.npz"
    if not base_p.exists():
        t = time.perf_counter()
        b = ctx.evaluate(ctx.vec, full=True)
        b.update(calibration_residual_summary(ctx, b["wells"]))
        save_npz(base_p, b)
        G.log(f"{args.model}: base evaluation {time.perf_counter() - t:.0f}s; in-sample R2 "
              f"{float(b['r2_insample']):+.4f} (published {ctx.member.meta.get('r2_insample')}); "
              f"pooled within-well sd {float(b['pooled_sd']):.3f} m (published "
              f"{ctx.member.meta.get('well_datum_stats', {}).get('sigma_within_m')})")
    order = range(len(ctx.index))
    for j in (reversed(order) if args.reverse else order):
        p = work / f"eval_{j:03d}.npz"
        if p.exists():
            continue
        t = time.perf_counter()
        v = ctx.vec.copy()
        v[j] += sign[j] * EPS
        save_npz(p, ctx.evaluate(v))
        G.log(f"{args.model}: column {j + 1}/{len(ctx.index)} {ctx.index[j][0]}"
              f"[{ctx.index[j][1]}] ({time.perf_counter() - t:.0f}s)")
    G.log(f"{args.model}: evaluations done ({time.perf_counter() - t0:.0f}s)")
    return 0


def calibration_residual_summary(ctx: Ctx, pred: np.ndarray) -> dict:
    """Per-well in-sample residual sd (within-well, after the fitted datum) and lag-1
    autocorrelation of the calibration fit itself, on the months the fit observed. The
    observation array is used here only and not stored."""
    from hydrophysics.twin.calibrate_flow import _r2, well_datum_vector

    obs = ctx.inp.obs_h[:, 1:]
    mask = np.isfinite(obs)
    d = well_datum_vector(ctx.member.meta, ctx.inp.sids)
    d = np.zeros(obs.shape[0]) if d is None else d
    r = np.where(mask, obs - pred - d[:, None], 0.0)
    n = mask.sum(axis=1)
    rbar = r.sum(axis=1) / np.maximum(n, 1)
    e = np.where(mask, r - rbar[:, None], 0.0)
    sd = np.sqrt((e ** 2).sum(axis=1) / np.maximum(n, 1))
    pooled = math.sqrt(float((e ** 2).sum() / mask.sum()))
    pair = mask[:, 1:] & mask[:, :-1]
    num = (e[:, 1:] * e[:, :-1] * pair).sum(axis=1)
    den = np.sqrt(((e[:, 1:] ** 2) * pair).sum(axis=1) * ((e[:, :-1] ** 2) * pair).sum(axis=1))
    rho = np.where(den > 0, num / np.maximum(den, 1e-300), 0.0)
    r2 = _r2(pred, ctx.inp.obs_h_filled[:, 1:])
    del obs
    return {"sd_w": sd, "rho_w": rho, "n_w": n, "pooled_sd": np.array(pooled),
            "r2_insample": np.array(r2)}


# ----------------------------------------------------------------------------- validate
def cmd_validate(args) -> int:
    """(a) for one model: the forward differences of VAL_PARAMS against central
    differences (heads + quantities) and against reverse-mode VJPs (heads)."""
    os.chdir(G.REPO)
    work = Path(args.work) / args.model
    ctx = Ctx(args.model)
    sign = fd_sign(ctx)
    keys = [(k, i) for k, i in VAL_PARAMS if (k, i) in ctx.index]
    js = [ctx.index.index(ki) for ki in keys]
    for j in js:
        p = work / f"eval_{j:03d}_minus.npz"
        if p.exists():
            continue
        t = time.perf_counter()
        v = ctx.vec.copy()
        v[j] -= sign[j] * EPS
        save_npz(p, ctx.evaluate(v))
        G.log(f"validate: opposite side of {ctx.index[j]} ({time.perf_counter() - t:.0f}s)")
    p = work / "reverse_mode.npz"
    if not p.exists():
        t = time.perf_counter()
        save_npz(p, reverse_mode(ctx, js))
        G.log(f"validate: reverse-mode VJPs ({time.perf_counter() - t:.0f}s)")
    return 0


def reverse_mode(ctx: Ctx, js: list[int], n_v: int = 2, seed: int = 11) -> dict:
    """VJPs v' dH/dtheta_j of the hindcast heads at the wells through ``_rollout`` with
    autograd (the implicit solve's adjoint), for random v on the observed months. Each
    parameter enters as a scalar delta along its exact model-field direction (T, S, C are
    gathered linearly from theta; the logits are passed straight to _rollout)."""
    import torch

    from hydrophysics.twin.calibrate_flow import _rollout
    from hydrophysics.twin.forward import Member, build_model
    from hydrophysics.twin.uncertainty import unflatten

    inp = ctx.inp
    mem0 = Member(ctx.member.label, unflatten(ctx.vec, ctx.index, ctx.member.theta),
                  ctx.member.meta)
    model, sc, _ = build_model(inp.grid, mem0, "cpu")
    fields0 = {k: getattr(model, k).detach().clone()
               for k in ("log_T", "log_S", "log_L", "log_C_coast", "log_C_apex")}
    dirs = {}
    for j in js:
        v = ctx.vec.copy()
        v[j] += 1.0
        m1, _, _ = build_model(inp.grid, Member(ctx.member.label,
                                                unflatten(v, ctx.index, ctx.member.theta),
                                                ctx.member.meta), "cpu")
        dirs[j] = {k: getattr(m1, k).detach() - fields0[k] for k in fields0}
    delta = torch.zeros(len(js), dtype=torch.float64, requires_grad=True)
    f = {k: fields0[k].clone() for k in fields0}
    s = {k: sc[k] for k in ("recharge_frac_logit", "return_frac_logit", "pump_split_logit",
                            "log_eta", "log_head_extra") if k in sc}
    for a, j in enumerate(js):
        name = ctx.index[j][0]
        if name in s:
            s[name] = s[name] + delta[a]
        else:
            for k in f:
                f[k] = f[k] + delta[a] * dirs[j][k]
    model.set_apex_heads(ctx.h0)
    with torch.enable_grad():
        h, _ = _rollout(model, f["log_T"], f["log_S"], f["log_L"], ctx.h0, ctx.E.shape[-1],
                        recharge_field=ctx.r, recharge_scale=s.get("recharge_frac_logit"),
                        recharge_layer=0, E=ctx.E, log_eta=s.get("log_eta"),
                        log_head_extra=s.get("log_head_extra"), ground_elev=inp.ground_elev,
                        pump_layer=1, log_C_coast=f["log_C_coast"], log_C_apex=f["log_C_apex"],
                        pump_split_logit=s.get("pump_split_logit"),
                        return_frac_logit=s.get("return_frac_logit"),
                        spread_W=sc.get("spread_W"), return_state=True)
        hw = h[torch.as_tensor(inp.obs_layer), torch.as_tensor(inp.obs_idx), 1:]
        rng = np.random.default_rng(seed)
        mask = np.isfinite(inp.obs_h[:, 1:])
        V = np.stack([np.where(mask, rng.standard_normal(mask.shape), 0.0) for _ in range(n_v)])
        g = []
        for a in range(n_v):
            (gr,) = torch.autograd.grad((hw * torch.as_tensor(V[a])).sum(), delta,
                                        retain_graph=a < n_v - 1)
            g.append(gr.numpy())
    return {"js": np.array(js), "V": V, "vjp": np.stack(g), "wells0": hw.detach().numpy()}


# ----------------------------------------------------------------------------- analysis
def head_information(Jw: np.ndarray, mask: np.ndarray, sd: np.ndarray, rho: np.ndarray | None):
    """Per well: Q_w (inverse noise covariance on its observed months) and the datum-
    profiled information J'QJ - (J'Q1)(1'QJ)/(1'Q1). Returns (H, per-well blocks)."""
    W, T, p = Jw.shape
    H = np.zeros((p, p))
    blocks = []
    t_all = np.arange(T)
    for w in range(W):
        m = mask[w]
        if m.sum() < 2:
            blocks.append(None)
            continue
        J = Jw[w, m]                                   # (n, p)
        t = t_all[m]
        if rho is None:
            Q = np.eye(len(t)) / sd[w] ** 2
        else:
            r = float(np.clip(rho[w], 0.0, 0.98))
            C = sd[w] ** 2 * r ** np.abs(t[:, None] - t[None, :])
            Q = np.linalg.inv(C)
        q1 = Q.sum(axis=1)
        JQ = J.T @ Q
        a = JQ.sum(axis=1)
        Hw = JQ @ J - np.outer(a, a) / q1.sum()
        H += Hw
        blocks.append({"m": m, "J": J, "Q": Q, "q1": q1, "a": a, "t": t,
                       "L": np.linalg.cholesky(np.linalg.inv(Q))})
    return H, blocks


def head_score(blocks, eps_rows) -> np.ndarray:
    """sum_w J'Q e - (J'Q1)(1'Q e)/(1'Q1) for one noise draw (list of per-well vectors)."""
    b = 0.0
    for blk, e in zip(blocks, eps_rows, strict=True):
        if blk is None:
            continue
        Qe = blk["Q"] @ e
        b = b + blk["J"].T @ Qe - blk["a"] * Qe.sum() / blk["q1"].sum()
    return b


def lev_jacobians(ctx_d: dict, cj: dict, zone_cell: np.ndarray, base_hm: np.ndarray,
                  hms: list[np.ndarray], sign: np.ndarray) -> dict:
    """Leveling predictions (re-zeroed at each benchmark's first survey) at the design's
    entries: J wrt theta (from the perturbed hindcasts) and wrt phi (central FD)."""
    from hydrophysics.twin.compaction import VEP_KEYS

    mask, row, first = ctx_d["lev_mask"], ctx_d["lev_row"], ctx_d["lev_mask"].argmax(axis=1)
    ent = mask.copy()
    ent[np.arange(len(first)), first] = False             # re-zero entries are identically 0
    zb = zone_cell[ctx_d["lev_ucell"]]                    # zone of each unique bench cell
    sets = cj["zonal"]

    def pred(hm, override=None):
        s = column_subs(hm, zb, sets, override)[row]       # (n_b, T)
        s = s - s[np.arange(len(first)), first][:, None]
        return s[ent]

    p0 = pred(base_hm)
    Jt = np.stack([(pred(h) - p0) / (sign[j] * EPS) for j, h in enumerate(hms)], axis=1)
    phi_index = [(z, k) for z in range(len(sets)) for k in VEP_KEYS]
    phi0 = np.array([float(sets[z][k]) for z, k in phi_index])
    cols = []
    for z, k in phi_index:
        e = 1e-4 if k != "h_pc0" else 1e-3
        pp = pred(base_hm, {(z, k): float(sets[z][k]) + e})
        pm = pred(base_hm, {(z, k): float(sets[z][k]) - e})
        cols.append((pp - pm) / (2 * e))
    # entry -> survey order within its benchmark (for the random-walk variant)
    rr, tt = np.nonzero(ent)
    return {"Jt": Jt, "Jp": np.stack(cols, axis=1), "phi_index": phi_index, "phi0": phi0,
            "ent_row": rr, "ent_t": tt, "n_ent": int(ent.sum()),
            "n_bench": int(mask.shape[0])}


def rw_pred(rows: np.ndarray, ts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(pred, order): each entry's previous survey of the same benchmark (-1 for the first
    after the re-zero survey), and the entries sorted by benchmark then month."""
    order = np.lexsort((ts, rows))
    pred = np.full(len(rows), -1)
    for a in range(1, len(order)):
        i, k = order[a], order[a - 1]
        if rows[i] == rows[k]:
            pred[i] = k
    return pred, order


def rw_diff(X: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Cumulative entries -> increments between consecutive surveys (rows of X)."""
    prev = np.where((pred >= 0)[:, None] if X.ndim == 2 else pred >= 0,
                    X[np.maximum(pred, 0)], 0.0)
    return X - prev


def rw_cum(u: np.ndarray, pred: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Increments -> cumulative entries (the inverse of ``rw_diff`` for a vector)."""
    us = u[order]
    cs = np.cumsum(us)
    start = pred[order] < 0                            # first entry of each benchmark
    gid = np.cumsum(start) - 1
    before = (cs - us)[start]                          # cumulative sum before each group
    c = np.empty_like(u)
    c[order] = cs - before[gid]
    return c


def posterior(Hh: np.ndarray, Jt: np.ndarray, Jp: np.ndarray, Wl, prior_phi: np.ndarray):
    """Two-stage Laplace: (Sigma_joint, Sigma_theta, A^-1, G). ``Wl`` is the leveling
    weight: a scalar 1/sigma^2 (iid) or a matrix."""
    p = Hh.shape[0]
    St = np.linalg.inv(Hh + np.eye(p) / PRIOR_SD ** 2)
    JpW = Jp.T * Wl if np.isscalar(Wl) else Jp.T @ Wl
    A = JpW @ Jp + np.diag(1.0 / prior_phi ** 2)
    Ai = np.linalg.inv(A)
    Gm = -Ai @ (JpW @ Jt)
    Sp = Ai + Gm @ St @ Gm.T
    Stp = St @ Gm.T
    S = np.block([[St, Stp], [Stp.T, Sp]])
    return S, St, Ai, Gm, JpW


def quantities(ctx_d, base, evals, sign, index, vec, theta, meta, cols, lev, run) -> list:
    """Each quantity: name, value at the fit, gradient over [theta; phi_col0; phi_col1..],
    scale ('lin' natural units, 'log' for positive parameters)."""
    p = len(index)
    q = []
    nphi = [len(lv["phi_index"]) for lv in lev]
    ntot = p + sum(nphi)
    ch_w = ctx_d["ch_well"].astype(bool)
    for yi, yk in enumerate(YEARS):
        for county, sel in (("changhua", ch_w), ("yunlin", ~ch_w)):
            v0 = float(base["dh_wells"][sel, yi].mean())
            g = np.zeros(ntot)
            g[:p] = [(float(e["dh_wells"][sel, yi].mean()) - v0) / (sign[j] * EPS)
                     for j, e in enumerate(evals)]
            q.append({"name": f"cut30_head_change_{county}_wells_m_{yk}", "group": "heads",
                      "value": v0, "grad": g, "scale": "lin", "n_wells": int(sel.sum())})
    ch_c = ctx_d["ch_cell"].astype(bool)
    off = p
    for ci, c in enumerate(cols):
        key = f"avoided_{ci}"
        for yi, yk in enumerate(YEARS):
            for region, sel in (("fan", np.ones_like(ch_c)), ("yunlin", ~ch_c),
                                ("changhua", ch_c)):
                v0 = float(base[key][sel, yi].mean())
                g = np.zeros(ntot)
                g[:p] = [(float(e[key][sel, yi].mean()) - v0) / (sign[j] * EPS)
                         for j, e in enumerate(evals)]
                g[off:off + nphi[ci]] = avoided_phi_grad(base, c, lev[ci], sel, YEARS[yk])
                q.append({"name": f"cut30_avoided_subsidence_{region}_cm_{yk}",
                          "group": "subsidence", "column": c["dir"], "value": v0,
                          "grad": g, "scale": "lin"})
        off += nphi[ci]
    q += changhua_ts(ctx_d, index, vec, theta, meta, ntot)
    if "pump_split_logit" in theta:
        j = index.index(("pump_split_logit", 0))
        f = 1.0 / (1.0 + math.exp(-vec[j]))
        g = np.zeros(ntot)
        g[j] = f * (1 - f)
        q.append({"name": "pump_share_shallow_layer0", "group": "pump_shares", "value": f,
                  "grad": g, "scale": "lin"})
    if "recharge_frac_logit" in theta:
        j = index.index(("recharge_frac_logit", 0))
        f = 1.0 / (1.0 + math.exp(-vec[j]))
        g = np.zeros(ntot)
        g[j] = f * (1 - f)
        q.append({"name": "recharge_frac_layer0_info_only", "group": "recharge",
                  "value": f, "grad": g, "scale": "lin"})
    off = p
    for ci, c in enumerate(cols):
        names = c["json"].get("zone_names") or ["proximal", "mid", "distal"]
        for a, (z, k) in enumerate(lev[ci]["phi_index"]):
            if k not in ("log_ske", "log_skv"):
                continue
            g = np.zeros(ntot)
            g[off + a] = 1.0
            q.append({"name": f"{k[4:].capitalize()}_{names[z]}", "group": "column",
                      "column": c["dir"], "value": float(lev[ci]["phi0"][a]), "grad": g,
                      "scale": "log"})
        off += nphi[ci]
    return q


def avoided_phi_grad(base, c, lv, sel, k) -> np.ndarray:
    out = []
    for z, key in lv["phi_index"]:
        e = 1e-4 if key != "h_pc0" else 1e-3
        vals = []
        for s in (1, -1):
            ov = {(z, key): float(c["sets"][z][key]) + s * e}
            cz = c["zone"] == z
            sb = column_subs(base["path_base"][cz], np.full(cz.sum(), z), c["sets"], ov)
            sc = column_subs(base["path_cut"][cz], np.full(cz.sum(), z), c["sets"], ov)
            full = base[f"avoided_{c['ci']}"][:, list(YEARS.values()).index(k)].copy()
            full[cz] = (sb - sc)[:, k] * 100.0
            vals.append(float(full[sel].mean()))
        out.append((vals[0] - vals[1]) / (2 * e))
    return np.array(out)


def changhua_ts(ctx_d, index, vec, theta, meta, ntot) -> list:
    """Geometric-mean T and S over the Changhua cells per layer (log scale); the
    gradient through ``build_model``'s exact (linear) gather of the zone parameters."""
    from hydrophysics.twin.forward import Member, build_model
    from hydrophysics.twin.uncertainty import unflatten

    grid = ctx_d["grid"]
    ch = ctx_d["ch_cell"].astype(bool)

    def fields(v):
        m, _, _ = build_model(grid, Member("x", unflatten(v, index, theta), meta), "cpu")
        return m.log_T.detach().numpy()[:, ch].mean(axis=1), m.log_S.detach().numpy()[:, ch].mean(axis=1)

    T0, S0 = fields(vec)
    gT = np.zeros((len(T0), ntot))
    gS = np.zeros((len(S0), ntot))
    for j, (k, _) in enumerate(index):
        if not (k.startswith("log_T") or k.startswith("log_S")):
            continue
        v = vec.copy()
        v[j] += 1.0
        T1, S1 = fields(v)
        gT[:, j], gS[:, j] = T1 - T0, S1 - S0
    out = []
    for lay in range(len(T0)):
        out.append({"name": f"changhua_T_layer{lay + 1}_m2day", "group": "changhua_TS",
                    "value": float(T0[lay]), "grad": gT[lay], "scale": "log"})
        out.append({"name": f"changhua_S_layer{lay + 1}", "group": "changhua_TS",
                    "value": float(S0[lay]), "grad": gS[lay], "scale": "log"})
    return out


def summarise(qd: dict, sd: float) -> dict:
    v = qd["value"]
    if qd["scale"] == "log":
        ratio = math.sqrt(math.expm1(sd ** 2))
        med = math.exp(v)
        mean = math.exp(v + sd ** 2 / 2)
        return {"mean": mean, "median": med, "sd": mean * ratio, "sd_over_abs_mean": ratio,
                "sd_log": sd, "ci90": [math.exp(v - Z90 * sd), math.exp(v + Z90 * sd)]}
    ratio = sd / abs(v) if v != 0 else float("inf")
    return {"mean": v, "sd": sd, "sd_over_abs_mean": ratio,
            "ci90": [v - Z90 * sd, v + Z90 * sd]}


def coverage(deltas: np.ndarray, grads: np.ndarray, sds: np.ndarray) -> np.ndarray:
    z = deltas @ grads.T                          # (n, nq)
    return (np.abs(z) <= Z90 * sds[None, :]).mean(axis=0)


def analyze_model(name: str, work: Path) -> dict:
    from hydrophysics.twin.forward import load_members  # noqa: F401
    from hydrophysics.twin.grid import build_grid
    from hydrophysics.twin.inputs import resolve_paths

    run, col_dirs = MODELS[name]
    wd = work / name
    d = dict(np.load(wd / "design.npz", allow_pickle=False))
    member = load_member(run)
    vec, index, fixed = free_index(member)
    if not np.allclose(vec, d["vec"]):
        raise SystemExit(f"{name}: theta changed since the evaluations")
    sign = d["sign"]
    p = len(index)
    base = dict(np.load(wd / "eval_base.npz", allow_pickle=False))
    evals = [dict(np.load(wd / f"eval_{j:03d}.npz", allow_pickle=False)) for j in range(p)]
    d["grid"] = build_grid(resolve_paths()["polygon"], dx=1000.0)
    mask = d["obs_mask"]
    Jw = np.stack([(e["wells"] - base["wells"]) / (sign[j] * EPS)
                   for j, e in enumerate(evals)], axis=-1)            # (W, T, p)
    cols = []
    for ci, cd in enumerate(col_dirs):
        cp = G.RUNS / run / cd / "vep_zonal_leveling.json"
        cj = json.loads(cp.read_text())
        zone = d["zoc"] % 3 if cj.get("column_zones_collapsed") else d["zoc"]
        cols.append({"dir": cd, "path": cp, "json": cj, "zone": zone, "sets": cj["zonal"],
                     "ci": ci})
    lev = [lev_jacobians(d, c["json"], c["zone"], base["bench_hm"],
                         [e["bench_hm"] for e in evals], sign) for c in cols]
    sd_w, rho_w = base["sd_w"], base["rho_w"]
    res = {"model": name, "run": run, "n_params": p, "fixed": list(fixed),
           "params": [f"{k}[{i}]" for k, i in index],
           "at_bound": at_bound_list(index, vec, member.meta),
           "design": {"n_wells": int(mask.shape[0]), "n_well_months_observed": int(mask.sum()),
                      "head_residual_sd_m_p10_p50_p90": np.percentile(sd_w, [10, 50, 90]).tolist(),
                      "head_residual_pooled_sd_m": float(base["pooled_sd"]),
                      "published_sigma_within_m": member.meta.get("well_datum_stats", {}).get(
                          "sigma_within_m"),
                      "insample_r2_rebuilt": float(base["r2_insample"]),
                      "insample_r2_published": member.meta.get("r2_insample"),
                      "head_residual_lag1_rho_p10_p50_p90": np.percentile(rho_w, [10, 50, 90]).tolist(),
                      "columns": []}}
    for c, lv in zip(cols, lev, strict=True):
        res["design"]["columns"].append({
            "dir": c["dir"], "n_benchmarks": lv["n_bench"],
            "n_entries_incl_rezero": int(d["lev_mask"].sum()),
            "published_n_sites_pairs": [c["json"]["leveling"]["n_sites"],
                                        c["json"]["leveling"]["n_pairs"]],
            "published_rmse_cm": c["json"]["leveling"]["rmse_cm"],
            "phi": [f"{k}[{z}]" for z, k in lv["phi_index"]],
            "phi_values": lv["phi0"].tolist()})
    qs = quantities(d, base, evals, sign, index, vec, member.theta, member.meta, cols, lev, run)
    grads = np.stack([q["grad"] for q in qs])
    nphi = [len(lv["phi_index"]) for lv in lev]
    rng_master = np.random.default_rng({"R0": 101, "Q0": 102, "R4p": 103, "R1c": 104}[name])

    variants = {}
    for var in ("iid", "ar1"):
        Hh, blocks = head_information(Jw, mask, sd_w, None if var == "iid" else rho_w)
        # joint covariance over [theta, phi_col0, phi_col1, ...]: each column is its own
        # second stage on the same theta
        Sig_parts, stage = [], []
        for lv, c in zip(lev, cols, strict=True):
            sig_l = c["json"]["leveling"]["rmse_cm"] / 100.0
            prior_phi = np.array([PRIOR_SD_HPC0 if k == "h_pc0" else PRIOR_SD
                                  for _, k in lv["phi_index"]])
            pred, order = rw_pred(lv["ent_row"], lv["ent_t"])
            if var == "iid":
                Wl, Drw = 1.0 / sig_l ** 2, None
                Jt, Jp = lv["Jt"], lv["Jp"]
            else:
                # survey order k of each entry (1 = first after the re-zero survey)
                k_ord = np.ones(lv["n_ent"])
                for i in order:
                    if pred[i] >= 0:
                        k_ord[i] = k_ord[pred[i]] + 1
                sig_u = sig_l / math.sqrt(float(k_ord.mean()))
                Jt, Jp = rw_diff(lv["Jt"], pred), rw_diff(lv["Jp"], pred)
                Wl = 1.0 / sig_u ** 2
                Drw = {"sig_u": sig_u, "mean_k": float(k_ord.mean())}
            S, St, Ai, Gm, JpW = posterior(Hh, Jt, Jp, Wl, prior_phi)
            stage.append({"St": St, "Ai": Ai, "Gm": Gm, "JpW": JpW, "Jt": Jt,
                          "sig_l": sig_l, "Drw": Drw, "pred": pred, "order": order})
            Sig_parts.append((Ai, Gm))
        St = stage[0]["St"]
        ntot = p + sum(nphi)
        Sj = np.zeros((ntot, ntot))
        Sj[:p, :p] = St
        offs = np.cumsum([p] + nphi)
        for a, (Ai, Gm) in enumerate(Sig_parts):
            ia = slice(offs[a], offs[a + 1])
            Sj[:p, ia] = St @ Gm.T
            Sj[ia, :p] = Gm @ St
            for b, (_Bi, Gb) in enumerate(Sig_parts):
                ib = slice(offs[b], offs[b + 1])
                Sj[ia, ib] = Gm @ St @ Gb.T + (Ai if a == b else 0.0)
        sds = np.sqrt(np.einsum("qi,ij,qj->q", grads, Sj, grads))
        seed = int(rng_master.integers(1 << 62))
        dl = deltas_shared(N_REP, seed, blocks, stage, None)
        cov50 = coverage(dl, grads, sds)
        dl2 = deltas_shared(N_REP_DIAG, seed + 1, blocks, stage, None)
        cov2k = coverage(dl2, grads, sds)
        # share of the posterior variance the data reproduce (1 = data-driven width,
        # well below 1 = the weak prior sets the width)
        vfrac = (dl2 @ grads.T).var(axis=0) / sds ** 2
        variants[var] = {"sds": sds, "cov50": cov50, "cov2k": cov2k, "Sj": Sj, "vfrac": vfrac,
                         "Hh": Hh, "blocks": blocks, "stage": stage, "dl": dl}
    # misspecification diagnostic: the iid posterior's intervals under AR(1)/RW noise
    iid, ar1 = variants["iid"], variants["ar1"]
    dmis = deltas_shared(N_REP_DIAG, 7_000 + len(name), iid["blocks"], iid["stage"],
                         ar1["blocks"], rw_noise=[s["Drw"] for s in ar1["stage"]])
    cov_mis = coverage(dmis, grads, iid["sds"])

    out_q = []
    for a, q in enumerate(qs):
        row = {"name": q["name"], "group": q["group"], "scale": q["scale"]}
        if "column" in q:
            row["column"] = q["column"]
        for var in ("iid", "ar1"):
            V = variants[var]
            s = summarise(q, float(V["sds"][a]))
            c50, c2k = float(V["cov50"][a]), float(V["cov2k"][a])
            ok_cov = COVER_WINDOW[0] <= c50 <= COVER_WINDOW[1]
            ident = (s["sd_over_abs_mean"] <= RATIO_MAX) and ok_cov
            row[var] = {**s, "coverage_50rep": c50, "coverage_2000rep_diag": c2k,
                        "replicate_var_over_posterior_var_2000rep": float(V["vfrac"][a]),
                        "coverage_pass": ok_cov,
                        "verdict": "identifiable" if ident else "not identifiable without new data"}
        row["iid_coverage_under_ar1_rw_noise_2000rep"] = float(cov_mis[a])
        out_q.append(row)
    res["quantities"] = out_q
    res["n_rep"] = N_REP
    res["information_matrix_heads_iid"] = variants["iid"]["Hh"].tolist()
    res["posterior_sd_params_iid"] = {
        f"{k}[{i}]": float(math.sqrt(variants["iid"]["Sj"][j, j]))
        for j, (k, i) in enumerate(index)}
    res["posterior_sd_params_ar1"] = {
        f"{k}[{i}]": float(math.sqrt(variants["ar1"]["Sj"][j, j]))
        for j, (k, i) in enumerate(index)}
    for ci in range(len(lev)):
        off = p + sum(nphi[:ci])
        res["design"]["columns"][ci]["posterior_sd_phi_iid"] = [
            float(math.sqrt(variants["iid"]["Sj"][off + a, off + a])) for a in range(nphi[ci])]
        res["design"]["columns"][ci]["posterior_sd_phi_ar1"] = [
            float(math.sqrt(variants["ar1"]["Sj"][off + a, off + a])) for a in range(nphi[ci])]
        res["design"]["columns"][ci]["posterior_sd_phi_given_heads_iid"] = [
            float(math.sqrt(variants["iid"]["stage"][ci]["Ai"][a, a])) for a in range(nphi[ci])]
    res["gradients"] = {q["name"] + (f"@{q['column']}" if "column" in q else ""):
                        q["grad"].tolist() for q in qs}
    res["provenance_inputs"] = [G.source(G.RUNS / run / "stage3_theta.json")] + [
        G.source(c["path"]) for c in cols]
    save_npz(wd / "replicates_iid.npz",
             {"deltas": variants["iid"]["dl"], "grads": grads, "sds": variants["iid"]["sds"],
              "values": np.array([q["value"] for q in qs]),
              "names": np.array([q["name"] + (f"@{q['column']}" if "column" in q else "")
                                 for q in qs]),
              "nphi": np.array(nphi)})
    return res


def deltas_shared(n, seed, blocks, stage, noise_blocks, rw_noise=None):
    """``n`` two-stage linearised refits; the head draw is shared by every column's
    second stage. ``noise_blocks`` overrides the head-noise covariance (misspecification
    check); ``rw_noise`` (per column: {"sig_u"}) draws random-walk leveling noise into a
    posterior built on the cumulative (iid) design."""
    rng = np.random.default_rng(seed)
    nb = noise_blocks if noise_blocks is not None else blocks
    St = stage[0]["St"]
    rows = []
    for _ in range(n):
        eps = [None if blk is None else blk["L"] @ rng.standard_normal(len(blk["t"]))
               for blk in nb]
        dth = St @ head_score(blocks, eps)
        parts = [dth]
        for a, s in enumerate(stage):
            n_ent = s["Jt"].shape[0]
            if rw_noise is not None:
                el = rw_cum(rw_noise[a]["sig_u"] * rng.standard_normal(n_ent),
                            s["pred"], s["order"])
            elif s["Drw"] is not None:
                # the posterior's own random-walk model works on increments: iid there
                el = s["Drw"]["sig_u"] * rng.standard_normal(n_ent)
            else:
                el = s["sig_l"] * rng.standard_normal(n_ent)
            parts.append(s["Ai"] @ (s["JpW"] @ (el - s["Jt"] @ dth)))
        rows.append(np.concatenate(parts))
    return np.stack(rows)


def at_bound_list(index, vec, meta) -> list:
    from hydrophysics.twin.uncertainty import effective_bounds, find_at_bound

    lo, hi = effective_bounds(index, meta)
    return [f"{index[j][0]}[{index[j][1]}]" for j in find_at_bound(vec, lo, hi)]


def validation_report(work: Path, name: str = "R0") -> dict:
    run, col_dirs = MODELS[name]
    wd = work / name
    d = dict(np.load(wd / "design.npz", allow_pickle=False))
    member = load_member(run)
    vec, index, _ = free_index(member)
    sign = d["sign"]
    base = dict(np.load(wd / "eval_base.npz", allow_pickle=False))
    rm = dict(np.load(wd / "reverse_mode.npz", allow_pickle=False))
    mask = d["obs_mask"]
    out = {"model": name, "eps": EPS, "params": []}
    keys = ["wells", "dh_wells"] + [f"avoided_{ci}" for ci in range(len(col_dirs))]
    worst = 0.0
    worst_q = 0.0
    for a, j in enumerate(rm["js"]):
        ep = dict(np.load(wd / f"eval_{j:03d}.npz", allow_pickle=False))
        em = dict(np.load(wd / f"eval_{j:03d}_minus.npz", allow_pickle=False))
        h = sign[j] * EPS
        row = {"param": f"{index[j][0]}[{index[j][1]}]", "fd_sign": float(sign[j])}
        for k in keys:
            fwd = (ep[k] - base[k]) / h
            cen = (ep[k] - em[k]) / (2 * h)
            if k == "wells":
                fwd, cen = fwd[mask], cen[mask]
            rel = float(np.linalg.norm(fwd - cen) / max(np.linalg.norm(cen), 1e-300))
            row[f"rel_err_forward_vs_central_{k}"] = rel
            if k == "wells":
                worst = max(worst, rel)
        # the reported quantities: county-mean head change and regional-mean avoided
        # subsidence at the three horizons (the per-cell fields above are diagnostics;
        # a few cells sit on the column's preconsolidation kink)
        ch_w, ch_c = d["ch_well"].astype(bool), d["ch_cell"].astype(bool)
        qerr = {}
        for yi, yk in enumerate(YEARS):
            for county, sel in (("changhua", ch_w), ("yunlin", ~ch_w)):
                f = (ep["dh_wells"][sel, yi].mean() - base["dh_wells"][sel, yi].mean()) / h
                c = (ep["dh_wells"][sel, yi].mean() - em["dh_wells"][sel, yi].mean()) / (2 * h)
                qerr[f"head_{county}_{yk}"] = float(abs(f - c) / max(abs(c), 1e-300))
            for ci in range(len(col_dirs)):
                for region, sel in (("fan", np.ones_like(ch_c)), ("yunlin", ~ch_c),
                                    ("changhua", ch_c)):
                    k = f"avoided_{ci}"
                    f = (ep[k][sel, yi].mean() - base[k][sel, yi].mean()) / h
                    c = (ep[k][sel, yi].mean() - em[k][sel, yi].mean()) / (2 * h)
                    qerr[f"avoided{ci}_{region}_{yk}"] = float(abs(f - c) / max(abs(c), 1e-300))
        row["rel_err_forward_vs_central_quantities"] = qerr
        row["rel_err_forward_vs_central_quantities_max"] = max(qerr.values())
        worst_q = max(worst_q, max(qerr.values()))
        fwd_full = (ep["wells"] - base["wells"]) / h
        cen_full = (ep["wells"] - em["wells"]) / (2 * h)
        for b in range(rm["V"].shape[0]):
            vj = float(rm["vjp"][b, a])
            vf = float((rm["V"][b] * fwd_full).sum())
            vc = float((rm["V"][b] * cen_full).sum())
            row[f"vjp{b}_reverse"] = vj
            row[f"vjp{b}_forward_fd"] = vf
            row[f"vjp{b}_central_fd"] = vc
            row[f"vjp{b}_rel_err_forward_fd_vs_reverse"] = abs(vf - vj) / max(abs(vj), 1e-300)
            row[f"vjp{b}_rel_err_central_fd_vs_reverse"] = abs(vc - vj) / max(abs(vj), 1e-300)
            worst = max(worst, row[f"vjp{b}_rel_err_forward_fd_vs_reverse"])
        out["params"].append(row)
    out["reverse_mode_heads_reproduce_base_max_abs_m"] = float(
        np.abs(rm["wells0"] - base["wells"]).max())
    out["worst_rel_err_heads"] = worst
    out["worst_rel_err_quantities_forward_vs_central"] = worst_q
    out["pass_lt_5pct"] = bool(worst < 0.05 and worst_q < 0.05)
    return out


def cmd_analyze(args) -> int:
    os.chdir(G.REPO)
    work = Path(args.work)
    models = args.models.split(",")
    res = {"what": "A0 identifiability audit: two-stage Laplace (Gauss-Newton) posterior of "
                   "each frozen model's calibrated flow and column parameters given the "
                   "calibration design only, propagated to the policy quantities",
           "rule": {"identifiable_iff": f"sd/|mean| <= {RATIO_MAX} AND the nominal 90 % "
                                        f"interval covers in [{COVER_WINDOW[0]}, "
                                        f"{COVER_WINDOW[1]}] of {N_REP} synthetic replicates",
                    "primary_design": "iid",
                    "note_mc": "with 50 replicates a perfectly calibrated 90 % interval lands "
                               "outside [0.85, 0.95] about 23 % of the time (binomial); the "
                               "2000-replicate coverage is reported as a diagnostic only"},
           "settings": {"eps": EPS, "prior_sd_log": PRIOR_SD, "prior_sd_hpc0_m": PRIOR_SD_HPC0,
                        "horizons_months_from_2012_01": YEARS, "scenario": SCEN,
                        "forward": "free run (gain 0): hindcast 2012-2022 then month-of-year "
                                   "climatology 2023-01..2032-12",
                        "variants": {"iid": "heads iid per-well sd (calibration residual), "
                                            "leveling iid published RMSE",
                                     "ar1": "heads AR(1) per well (calibration residual "
                                            "lag-1), leveling random-walk increments"}},
           "models": {}}
    for name in models:
        G.log(f"analyze {name}")
        out = dict(res)
        out["models"] = {name: analyze_model(name, work)}
        out["provenance"] = {"code": "dataprep_staging/n1_gaps/identifiability.py",
                             "git_head": G.git_head(),
                             "status": "post-hoc model-only audit; no test scored, registered "
                                       "or looked at; no head or leveling value from 2023 on "
                                       "read"}
        G.write_new(OUT / f"identifiability_{name}.json", out)
    return 0


def cmd_selfcheck(args) -> int:
    """Write self-validation (a) and, if present, the nonlinear check (b) for R0."""
    os.chdir(G.REPO)
    work = Path(args.work)
    out = {"what": "A0 self-validation", "a_jacobian": validation_report(work, "R0")}
    nl = nonlinear_report(work, "R0")
    if nl is not None:
        out["b_nonlinear_R0"] = nl
    out["provenance"] = {"code": "dataprep_staging/n1_gaps/identifiability.py",
                         "git_head": G.git_head()}
    G.write_new(OUT / args.out_name, out)
    return 0


def cmd_nonlinear(args) -> int:
    """(b), nonlinear: the full model re-run at theta_hat + delta_theta_r (and the column at
    phi_hat + delta_phi_r) for the iid replicates ``--start``..``--stop``."""
    os.chdir(G.REPO)
    wd = Path(args.work) / args.model
    rep = dict(np.load(wd / "replicates_iid.npz", allow_pickle=False))
    ctx = Ctx(args.model)
    p = len(ctx.index)
    nphi = [int(x) for x in rep["nphi"]]
    for r in range(args.start, min(args.stop, rep["deltas"].shape[0])):
        f = wd / f"nonlin_{r:03d}.npz"
        if f.exists():
            continue
        t = time.perf_counter()
        dl = rep["deltas"][r]
        off = p
        for c, n in zip(ctx.columns, nphi, strict=True):
            from hydrophysics.twin.compaction import VEP_KEYS

            keys = [(z, k) for z in range(len(c["sets"])) for k in VEP_KEYS]
            c["sets"] = [dict(x) for x in c["json"]["zonal"]]
            for a, (z, k) in enumerate(keys):
                c["sets"][z][k] = float(c["sets"][z][k]) + float(dl[off + a])
            off += n
        save_npz(f, ctx.evaluate(ctx.vec + dl[:p]))
        G.log(f"nonlinear replicate {r} ({time.perf_counter() - t:.0f}s)")
    return 0


def nonlinear_report(work: Path, name: str):
    wd = work / name
    files = sorted(wd.glob("nonlin_*.npz"))
    if not files:
        return None
    rep = dict(np.load(wd / "replicates_iid.npz", allow_pickle=False))
    d = dict(np.load(wd / "design.npz", allow_pickle=False))
    base = dict(np.load(wd / "eval_base.npz", allow_pickle=False))
    names = [str(x) for x in rep["names"]]
    ch_w, ch_c = d["ch_well"].astype(bool), d["ch_cell"].astype(bool)
    rows = []
    for f in files:
        r = int(f.stem.split("_")[1])
        e = dict(np.load(f, allow_pickle=False))
        vals = {}
        for yi, yk in enumerate(YEARS):
            for county, sel in (("changhua", ch_w), ("yunlin", ~ch_w)):
                vals[f"cut30_head_change_{county}_wells_m_{yk}"] = float(
                    e["dh_wells"][sel, yi].mean())
            for ci, cd in enumerate(MODELS[name][1]):
                for region, sel in (("fan", np.ones_like(ch_c)), ("yunlin", ~ch_c),
                                    ("changhua", ch_c)):
                    vals[f"cut30_avoided_subsidence_{region}_cm_{yk}@{cd}"] = float(
                        e[f"avoided_{ci}"][sel, yi].mean())
        rows.append((r, vals))
    out = {"n_replicates": len(rows), "quantities": {}}
    for qi, nm in enumerate(names):
        if nm not in rows[0][1]:
            continue
        v0, sd = float(rep["values"][qi]), float(rep["sds"][qi])
        dn = np.array([vals[nm] - v0 for _, vals in rows])
        dlin = np.array([float(rep["deltas"][r] @ rep["grads"][qi]) for r, _ in rows])
        out["quantities"][nm] = {
            "coverage_nonlinear": float((np.abs(dn) <= Z90 * sd).mean()),
            "coverage_linear_same_reps": float((np.abs(dlin) <= Z90 * sd).mean()),
            "rms_nonlinear_minus_linear_over_sd": float(
                np.sqrt(np.mean((dn - dlin) ** 2)) / sd)}
    _ = base
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("evals", "validate"):
        s = sub.add_parser(c)
        s.add_argument("--model", required=True, choices=list(MODELS))
        s.add_argument("--work", required=True)
        s.add_argument("--threads", type=int, default=2)
        s.add_argument("--reverse", action="store_true",
                       help="evals: walk the parameters from the last (a second worker)")
    s = sub.add_parser("nonlinear")
    s.add_argument("--model", default="R0", choices=list(MODELS))
    s.add_argument("--work", required=True)
    s.add_argument("--start", type=int, default=0)
    s.add_argument("--stop", type=int, default=N_REP)
    s.add_argument("--threads", type=int, default=2)
    s = sub.add_parser("selfcheck")
    s.add_argument("--work", required=True)
    s.add_argument("--out-name", default="self_validation_R0.json")
    s.add_argument("--threads", type=int, default=2)
    s = sub.add_parser("analyze")
    s.add_argument("--work", required=True)
    s.add_argument("--models", default="R0,Q0,R4p,R1c")
    s.add_argument("--out-name", default="identifiability.json")
    s.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    import torch

    torch.set_num_threads(args.threads)
    return {"evals": cmd_evals, "validate": cmd_validate, "analyze": cmd_analyze,
            "nonlinear": cmd_nonlinear, "selfcheck": cmd_selfcheck}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
