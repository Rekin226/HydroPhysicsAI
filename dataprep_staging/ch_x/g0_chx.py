"""G0 of I1.CHX.v1: the synthetic decidability gate, run on R1c's own contrast at the 23
external Changhua screens with no observation (the lead runs it; model output + declared
parameters only).

World (``chx_lib.world``), per screen and window month 2013-01..2022-12:

    e = obs - ZR = k c + block AR(1) + well AR(1) + kappa g_t (1 + z_i) + s h_t

* c: R1c's Changhua-only contrast A - B (hindcast.py chx), the panel the scorer will use;
* noise and kappa: N2C's published, already declared values (``synth_n2c.PARAMS``:
  sigma_tot 1.318 m, 5 km block share 0.439, lag-1 phi 0.859, kappa); disclosed: they come
  from the 74 calibration screens, so if the externals are noisier G0 overstates power;
* g: one common series, unit sd, correlation rho with the Changhua pumping anomaly a_CH;
* z_i: a per-screen loading, N(0, 1) per replicate (``G0_HET_MULT`` 1, ``G0_HET_CORR`` 0): the
  two-way primary absorbs a common-only confounder exactly, so without it the false-pass
  gate would be vacuous; report-only cells correlate it 0.5 with the contrast amplitude;
* h: the declared common-trend hinge (0 to 2016-07, then linear to 1 m at 2022-12), with the
  sign that biases the per-well detrended slope upward (``trend_sign``; absorbed exactly by
  the two-way form); report-only cells reverse it or drop it;
* the availability mask: each screen observed on its last min(n, 120) window months, n =
  N3F's logged pre-origin month count (wells.csv ``n_fit_months``; a count, no value).

Order: the informative gate on R1c's TRANSFORMED contrast first (if it fails, no cell runs
and the reading is 'not_informative'); exact plumbing (no noise: the slope equals k to 1e-9;
a failure is engineering); then the grid, scored exactly as the primary
(``chx_lib.score_direction``, form ``chx_const.CHX_PRIMARY_FORM``). Gate: P(direction | k = 1,
rho = -0.83) >= 0.60 and max over rho in {0, -0.83} of P(direction | k = 0) <= 0.20 (trend on,
heterogeneous loading); otherwise 'not decidable' and the scorer opens no head.

    python dataprep_staging/ch_x/g0_chx.py --panel results/twin_runs/chx/hindcast/chx_panel.npz \\
        --out results/twin_runs/chx/g0
    python dataprep_staging/ch_x/g0_chx.py --smoke --out <scratch dir>   (synthetic panel)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent), str(HERE.parent / "n2c_reading")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import chx_const as K  # noqa: E402
import chx_io as IO  # noqa: E402
import chx_lib as L  # noqa: E402


def declared_params() -> dict:
    """N2C's declared noise/confounder values, imported (no new look)."""
    import synth_n2c as SN

    P = SN.PARAMS
    return {"sigma_tot_m": float(P["sigma_tot_m"]["value"]),
            "block_share": float(P["block_share"]["value"]),
            "phi": float(P["phi"]["value"]), "kappa_m": float(P["kappa_m"]["value"]),
            "source": K.G0_PARAMS_FROM, "source_sha256": IO.sha256(HERE.parent / "n2c_reading"
                                                                    / "synth_n2c.py")}


def panel_from_npz(path: Path, counts: np.ndarray | None) -> dict:
    z = np.load(path, allow_pickle=False)
    months = [str(m) for m in z["months"]]
    c = np.asarray(z["c_ch"], float)
    sid = np.asarray(z["sid_ch"]).astype(str)
    if counts is None:
        counts = IO.ext_counts(list(sid))
    return build_panel(c, np.asarray(z["a_ch"], float), sid,
                       np.asarray(z["block_ch"]).astype(str), months, counts)


def build_panel(c, a_ch, sid, block, months, counts, form: str = K.CHX_PRIMARY_FORM
                ) -> dict:
    T = len(months)
    c = np.asarray(c, float)
    mask = L.g0_mask(counts, T)
    h = L.hinge(months)
    blocks = dict(zip(sid, block, strict=True))
    return {"c": c, "mask": mask, "a_ch": np.asarray(a_ch, float),
            "sid": np.asarray(sid), "block": np.asarray(block), "months": months, "hinge": h,
            "trend_sign": L.trend_sign(c, mask, h), "load_z": L.load_z(c),
            "informative": L.informative_from_contrast(c, sid, blocks, months, form),
            "counts": np.asarray(counts, int), "form": form}


def synthetic_panel(seed: int = 0, W: int = K.N_EXT_CHANGHUA, scale: float = 1.0,
                    form: str = K.CHX_PRIMARY_FORM) -> dict:
    """A stand-in (smoke runs, tests): no model output."""
    months = L.month_list(*K.CHX_WINDOW)
    T = len(months)
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    a = 0.12 * np.sin(2 * np.pi * t / 12) + 0.05 * rng.standard_normal(T)
    lag = 0.6 * a + 0.4 * np.r_[a[0], a[:-1]]
    c = scale * (-rng.uniform(0.8, 2.5, W)[:, None] * lag[None, :] + 0.001 * t[None, :])
    sid = np.array([f"07{i:06d}" for i in range(W)])
    x = rng.uniform(170000, 205000, W)
    y = rng.uniform(2640000, 2675000, W)
    bl = L.well_blocks(sid, x, y)
    return build_panel(c, a, sid, np.array([bl[s] for s in sid]), months,
                       rng.integers(24, 132, W), form)


def run(panel: dict, params: dict, reps: int, B: int, seed: int = K.G0_SEED) -> dict:
    form = panel["form"]
    res = L.run_g0(panel, params, reps=reps, seed=seed, B=B, form=form)
    res.update({"design": {
        "world": "e = k c + block AR(1) + well AR(1) + kappa g_t (1 + het z_i) + s h_t, NaN "
                 "outside the mask; scored by chx_lib.score_direction (form "
                 f"{form!r}, pinned FE slope, 5 km blocks with the 07100111-141 nest as one "
                 "block, decide())", "form": form,
        "params": params, "grid": {"k": list(K.G0_K), "rho": list(K.G0_RHO)},
        "het": {"mult": K.G0_HET_MULT, "corr": K.G0_HET_CORR},
        "report_only_cells": list(K.G0_REPORT_CELLS),
        "replicates_per_cell": reps, "bootstrap_B": B, "seed": seed,
        "seeding": "SeedSequence(seed).spawn(n_grid + n_report): grid cells k outer, rho "
                   "inner, then G0_REPORT_CELLS in order",
        "trend": {"break": K.TREND_BREAK, "amp_m": K.TREND_AMP_M,
                  "sign": panel["trend_sign"]},
        "mask": "last min(n_fit_months, 120) window months per screen",
        "n_screens": int(len(panel["sid"])), "n_blocks": int(len(set(panel["block"]))),
        "informative_gate_on_contrast": panel["informative"]}})
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--panel", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true", help="synthetic panel, few replicates")
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--form", choices=K.FORMS[:2], default=None,
                    help="--smoke only: the primary form to simulate (default the frozen one)")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="--smoke only: multiply the stand-in contrast")
    a = ap.parse_args(argv)
    if not a.smoke and (a.form is not None or a.scale != 1.0):
        ap.error("--form and --scale are for --smoke only; the registered run is fixed")
    if a.smoke:
        panel = synthetic_panel(scale=a.scale, form=a.form or K.CHX_PRIMARY_FORM)
        reps, B = a.reps or 40, 400
    else:
        if a.panel is None:
            ap.error("--panel is required (or --smoke)")
        summ = json.loads((a.panel.parent / "chx_panel.json").read_text())
        if IO.sha256(a.panel) != summ["panel_sha256"]:
            raise SystemExit("panel sha256 differs from chx_panel.json")
        panel, reps, B = panel_from_npz(a.panel, None), K.G0_REPLICATES, 2000
        if a.reps is not None:
            raise SystemExit("--reps is for --smoke only; the registered run is fixed")
    params = declared_params()
    a.out.mkdir(parents=True, exist_ok=True)
    target = a.out / (f"g0_smoke_{panel['form']}_x{a.scale:g}.json" if a.smoke else "g0.json")
    if target.exists():
        raise SystemExit(f"{target} exists; G0 runs once")
    res = run(panel, params, reps, B)
    res["created"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    res["panel"] = None if a.smoke else str(a.panel)
    res["panel_sha256"] = None if a.smoke else IO.sha256(a.panel)
    res["smoke"] = bool(a.smoke)
    res["smoke_scale"] = a.scale if a.smoke else None
    res["code_sha256"] = {p: IO.sha256(HERE / p) for p in ("g0_chx.py", "chx_lib.py",
                                                           "chx_const.py")}
    with target.open("x") as f:
        json.dump(res, f, indent=1, default=float)
    g = res.get("gate") or {}
    print(f"G0 {'(smoke) ' if a.smoke else ''}form {panel['form']}; plumbing "
          f"{res['plumbing']['ok']}; informative {g.get('informative')}; power "
          f"{g.get('power_k1_rho083')}; false {g.get('false_pass_k0_max_rho')}; "
          f"decidable {g.get('decidable')} -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
