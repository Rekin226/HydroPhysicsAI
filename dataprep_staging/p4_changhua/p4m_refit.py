"""WP P4m T2b: CPU emulation of the R1 recalibration (theta bias from zero-Changhua
calibration), as declared in ``P4m.metered_diagnostics.v1``.

    cd dataprep_staging
    tools/cpu_slot.sh python -m p4_changhua.p4m_refit

P4.0's reference refit (``profiles_v2.gn_refit``: bounded Gauss-Newton of the 28 interior
theta entries, the 7 bound-pinned entries held, a central-difference Jacobian at every
accepted iterate, trust box 0.5, tol 1e-6, 15 iterations; ``fit_flow``'s level loss on
observed months 2012-02..2022-12 with the per-well datum profiled at the kappa of theta-hat)
is rerun with the metered Changhua energy added at s = 0, 0.5 and 1. The s = 0 refit
replicates P4.0's observed-target reference. The profiled objective L(s) gives the scale the
heads accept once theta may move (s_theta); the frozen-theta argmin of the same objective
(s_frozen_cal) is the like-for-like comparison. refit(1) - refit(0) is the expected direction
of the R1 theta shift. Everything is in-sample and approximate (CPU emulation, unconverged
references, fixed kappa, pinned entries held, no torch re-optimisation).
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import datetime as _dt  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from p4_changhua import engine, looks_p4m, profiles_v2  # noqa: E402
from p4_changhua import p4m_lib as L  # noqa: E402

OUT = L.STAGE / "refit"
S_LEVELS = (0.0, 0.5, 1.0)
N_ITER = 15
STEP = 1e-3


def main() -> int:
    from scipy.optimize import minimize_scalar

    from hydrophysics.twin.forward import Member

    OUT.mkdir(parents=True, exist_ok=True)
    created = _dt.datetime.now(_dt.timezone.utc).isoformat()
    t_all = time.perf_counter()
    S = L.setup()
    looks_p4m.record(["p4m.T2b.insample"], ids=list(S["sids"]), window="2012-01..2022-12",
                     description="heads of the 158 calibration wells, observed months "
                                 "2012-02..2022-12: the theta's own level loss refitted on CPU "
                                 "with metered Changhua forcing (in-sample)",
                     purpose="insample_diagnostic", prereg_id=L.PRE_DIAG,
                     tag=f"[p4m_refit:T2b:{created}]",
                     note="T2b theta-refit emulation at s = 0, 0.5, 1; frozen-theta curve of "
                          "the same objective")
    Ech, _, _ = L.changhua_forcing(S)
    member, meta = S["member"], S["meta"]
    theta = member.theta
    obs = S["obs"]
    target = S["inp"].obs_h_filled[:, 1:].astype(np.float64)
    dmask = np.isfinite(obs)
    pinned = engine.bound_pinned(theta, meta)
    entries = engine.theta_vector(theta, meta)
    pin_names = {f"{p.name}[{p.index}]" for p in pinned}
    interior = [e for e in entries if e["name"] not in pin_names]
    lo_b = np.array([-np.inf if e["lo"] is None else e["lo"] for e in interior])
    hi_b = np.array([np.inf if e["hi"] is None else e["hi"] for e in interior])
    x_hat = np.array([e["value"] for e in interior])
    pred_hat = L.at_wells(S, L.rollout(S))
    _, kap, s2, _ = engine.profile_datum(pred_hat, target, dmask)
    N = float(dmask.sum())
    n_eval = [0]

    def predict(th, s):
        n_eval[0] += 1
        tw_p, _, _ = L.build_twin(Member(label="p4m_refit", theta=th, meta=meta), S["grid"],
                                  S["ge"])
        H = tw_p.rollout(S["h0"], E=S["E"], R=S["R"], E_extra=(s * Ech) if s else None,
                         apex_from=S["h0"])
        return H[S["ol"], S["oi"], 1:]

    def residuals(pr):
        n_ = dmask.sum(axis=1)
        rb = np.where(dmask, target - pr, 0.0).sum(axis=1) / np.maximum(n_, 1)
        dd = np.where(n_ > 0, n_ * rb / np.maximum(n_ + kap, 1e-12), 0.0)
        e = pr + dd[:, None] - target
        return np.concatenate([e[dmask], np.sqrt(kap) * dd])

    def with_interior(th, x):
        for e, v in zip(interior, x, strict=True):
            th = engine.perturb_theta(th, e["key"], e["index"], float(v))
        return th

    out = {"wp": "P4m", "prereg": L.PRE_DIAG, "created": created,
           "objective": "fit_flow level loss, observed months 2012-02..2022-12, per-well datum "
                        "profiled at kappa from theta-hat (P4.0's 'observed' reference); "
                        "value = |r|^2 / N_observed",
           "settings": {"n_iter": N_ITER, "trust": 0.5, "tol": 1e-6, "jacobian_step": STEP,
                        "n_interior": len(interior), "pinned": sorted(pin_names)},
           "N_observed": N, "s2_theta_hat": s2, "levels": {}}
    # frozen-theta curve of the same objective
    t0 = time.perf_counter()
    fz = {}

    def Lfz(s):
        k = round(float(s), 6)
        if k not in fz:
            r = residuals(predict(theta, s))
            fz[k] = float(r @ r) / N
        return fz[k]

    curve = [Lfz(s) for s in L.S_GRID]
    k = int(np.argmin(curve))
    lo, hi = L.S_GRID[max(k - 1, 0)], L.S_GRID[min(k + 1, len(L.S_GRID) - 1)]
    s_fz = (float(minimize_scalar(Lfz, bounds=(lo, hi), method="bounded",
                                  options={"xatol": 1e-3}).x) if lo != hi else lo)
    out["frozen"] = {"grid": list(L.S_GRID), "L": curve, "s_frozen_cal": s_fz,
                     "L_at_s_frozen_cal": Lfz(s_fz), "t_s": round(time.perf_counter() - t0, 1)}
    L.log(f"frozen curve: s_frozen_cal {s_fz:.4f}; L(0) {curve[0]:.5f} L(1) "
          f"{curve[L.S_GRID.index(1.0)]:.5f}")
    for s in S_LEVELS:
        t0 = time.perf_counter()

        def f(x, s=s):
            return residuals(predict(with_interior(theta, x), s))

        ref = profiles_v2.gn_refit(f, x_hat, lo_b, hi_b, n_iter=N_ITER, trust=0.5, tol=1e-6)
        J, _ = profiles_v2.central_jacobian(f, ref["x"], STEP)
        rem = profiles_v2.predicted_remaining(J, ref["res"], ref["x"], lo_b, hi_b)
        # pinned entries: does the descent direction push against the bound at the refit?
        th_ref = with_interior(theta, ref["x"])
        pin_rows = []
        for p in pinned:
            rp = residuals(predict(engine.perturb_theta(th_ref, p.name, p.index,
                                                        p.value + STEP), s))
            rm = residuals(predict(engine.perturb_theta(th_ref, p.name, p.index,
                                                        p.value - STEP), s))
            g = float(((rp - rm) / (2 * STEP)) @ ref["res"]) * 2.0
            outward = (-g > 0) if p.side == "hi" else (-g < 0)
            pin_rows.append({"name": f"{p.name}[{p.index}]", "side": p.side,
                             "gradient_dobj_dtheta": g / N, "descent_pushes_outward": bool(outward)})
        pr_ref = predict(th_ref, s)
        out["levels"][str(s)] = {
            "objective": ref["obj"] / N, "objective_at_theta_hat": Lfz(s),
            "relative_decrease_vs_theta_hat": (Lfz(s) - ref["obj"] / N) / Lfz(s),
            "n_iter": ref["n_iter"], "n_eval": ref["n_eval"], "stop_reason": ref["reason"],
            "last_step_hit_trust_box": ref["last_step_hit_trust_box"],
            "history": [h / N for h in ref["history"]],
            "predicted_remaining_relative_decrease": rem,
            "x": [{"name": e["name"], "value": float(v), "shift_from_theta_hat": float(v - x0),
                   "lo": e["lo"], "hi": e["hi"],
                   "at_bound": bool((e["lo"] is not None and v <= e["lo"] + 1e-9)
                                    or (e["hi"] is not None and v >= e["hi"] - 1e-9))}
                  for e, v, x0 in zip(interior, ref["x"], x_hat, strict=True)],
            "pinned_at_refit": pin_rows, "t_s": round(time.perf_counter() - t0, 1)}
        np.save(OUT / f"pred_refit_s{s}.npy", pr_ref)
        L.log(f"refit s={s}: {ref['reason']} after {ref['n_iter']} it, L {ref['obj'] / N:.5f} "
              f"(theta-hat {Lfz(s):.5f}), remaining {rem:.2e}, {time.perf_counter() - t0:.0f}s")
        L.jdump(out, OUT / "results_partial.json")
    Ls = [out["levels"][str(s)]["objective"] for s in S_LEVELS]
    a = (Ls[2] - 2 * Ls[1] + Ls[0]) / (2 * 0.5 ** 2)          # curvature of the parabola
    b = (Ls[1] - Ls[0]) / 0.5 - a * 0.5
    s_theta = float(-b / (2 * a)) if a > 0 else None
    out["s_theta"] = s_theta
    out["parabola"] = {"a": a, "b": b, "c": Ls[0], "convex": bool(a > 0),
                       "vertex_outside_0_1": bool(s_theta is not None and not 0 <= s_theta <= 1)}
    out["theta_bias_share_of_gap"] = (
        (s_theta - s_fz) / (1 - s_fz) if (s_theta is not None and s_fz < 1) else None)
    x0 = {r["name"]: r["value"] for r in out["levels"]["0.0"]["x"]}
    x1 = {r["name"]: r["value"] for r in out["levels"]["1.0"]["x"]}
    xh = {e["name"]: e["value"] for e in interior}
    out["dtheta_refit1_minus_refit0"] = sorted(
        [{"name": n, "dtheta": x1[n] - x0[n], "theta_hat": xh[n], "refit0": x0[n],
          "refit1": x1[n]} for n in x0], key=lambda r: -abs(r["dtheta"]))
    # replication of P4.0's observed reference
    v = json.loads((L.ENGINE_V2 / "validation.json").read_text())
    ref40 = v["repair"]["profiles_v2"]["observed"]["reference"]
    out["p40_replication"] = {"p40_objective_mse": ref40["objective_mse"],
                              "p4m_s0_objective": out["levels"]["0.0"]["objective"],
                              "abs_diff": abs(ref40["objective_mse"]
                                              - out["levels"]["0.0"]["objective"]),
                              "p40_stop": ref40["stop_reason"]}
    # in-sample T1-type scores at the refits (scored 2013-01..2022-12)
    tm = L.step_mask(S)
    obs_m = L.masked_obs(S, tm)
    G = L.groups(S)
    sc = {}
    for s in S_LEVELS:
        pr = np.load(OUT / f"pred_refit_s{s}.npy")
        pw = L.per_well_stats(obs_m, pr)
        sc[str(s)] = {g: L.group_metrics(pw, G[g]) for g in ("CH", "YL", "all")}
    out["insample_scores_at_refits"] = sc
    out["n_model_evaluations"] = n_eval[0]
    out["t_total_s"] = round(time.perf_counter() - t_all, 1)
    L.jdump(out, OUT / "results.json")
    L.log(f"s_theta {s_theta}; s_frozen_cal {s_fz:.4f}; share {out['theta_bias_share_of_gap']}; "
          f"P40 replication diff {out['p40_replication']['abs_diff']:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
