"""Model-only runs of I1.backcast_2010_11.v1 with the PAGE model (the lead runs this through
``run_bc11.sh``, on the GPU, from a clean detached worktree at the ONE declared E0 commit
``bc11_const.WT_HEAD`` = 4c2723e; an E0 failure there means not evaluable: stop).

``PageRun`` is a staging copy of ``ch_x/hindcast.py``'s ``R1cRun`` (hash-bound to E26 and
I1.CHX.v1, never edited) WITHOUT its zone_county_split guard (R1cRun refuses the page theta,
which has no county split) and over the page's 18 parameter sets (stage3_datum_gate, its 5
folds and 12 posterior draws), rebuilt as ``forward.py`` rebuilds them: the same inputs
(``load_twin_inputs`` with the page run's rain file, ``input_options(meta)``,
``attach_sw_recharge``), ``build_model`` per set, ``rollout`` from the page inputs' COMMON
month-0 field ``inp.initial_heads(0)`` (forward.py's ``h0``, shared by every set).
Heads are the ensemble mean over the 18 sets (the page's heads_mean averages 36 members = 18
sets x 2 initial fields, which differ only after the 2022-12 origin).

No observation is read or printed. Outputs are written once (mode 'x'); stdout carries only
shapes, check flags and sha256s.

Subcommands
-----------
``e0 --out DIR``
    E0: the native 2012-22 rollout of the ensemble must reproduce the registered page forward
    (datum_gate_innov.npz, sha cf2c7154), baseline scenario, months 2012-01..2022-12, every
    layer and cell, within E0_TOL_M = 0.01 m -> e0.json. A failed E0 writes its e0.json and
    exits 3 (logged): the backcast is then NOT EVALUABLE (stop).
``swap --out DIR``
    The pre-declared Yunlin+Changhua swap gate on 2012-22: census preorigin v3 energy in place
    of the page's native energy, per county (energy ratio, and the county-only contrast with
    the county at its 2012 pattern from 2013-01, every county cell and layer over 2013-22:
    correlation and RMS ratio of the per-cell-demeaned contrasts) -> swap.json. G3 gates only
    if both counties pass.
``backcast --stations PARQUET --out DIR``
    From 2007-01 with the common month-0 field (spin-up 2007-01..2009-12), recharge = the
    page inputs' 2012-22 calendar-month climatology (climatological rain), arms F1 (recorded
    v3 energy), F0 (v3's per-cell calendar-month climatology over 2007-11: no year-to-year
    pumping information) and B_yl (Yunlin cells at their 2007 monthly pattern from 2008-01,
    every other cell recorded). Heads at the product's screens over 2010-01..2011-12, the
    page forward's own 2012-22 heads at the same screens, the Yunlin contrast and its county
    pumping anomaly -> bc11_panel.npz + bc11_panel.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent / "ch_x"), str(HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bc11_const as K  # noqa: E402
import bc11_lib as BL  # noqa: E402
import chx_io as IO  # noqa: E402
import chx_lib as L  # noqa: E402
import hindcast as HC  # noqa: E402  (E26's frozen helpers: census_v3_energy, climatology)


def log(msg: str) -> None:
    print(f"{dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} {msg}", flush=True)


def theta_shas() -> dict:
    return {p: IO.sha_or_missing(IO.rp(p)) for p in K.PAGE_THETAS}


# ----------------------------------------------------------------------------- the model
class PageRun:
    """The page ensemble (18 parameter sets), heads only (no column)."""

    def __init__(self, thetas=K.PAGE_THETAS, device: str = "cpu", rf: str = K.PAGE_RF,
                 compile_matvec: bool = False):
        import torch

        from hydrophysics.twin.flow import set_compile_matvec
        from hydrophysics.twin.forward import (
            N_LAYERS,
            attach_sw_recharge,
            extra_hist,
            load_members,
            sw_hist,
        )
        from hydrophysics.twin.inputs import input_options, load_twin_inputs

        set_compile_matvec(bool(compile_matvec))
        self.torch = torch
        self.members = load_members([str(IO.rp(p)) for p in thetas])
        if len(self.members) != K.PAGE_N_SETS:
            raise SystemExit(f"{len(self.members)} parameter sets, expected {K.PAGE_N_SETS}")
        meta = self.members[0].meta
        self.meta = meta
        if meta.get("eta_classes"):
            raise SystemExit("the page theta is single-class (no eta_classes); not the page")
        inp = load_twin_inputs({"rf_timeseries": str(IO.rp(rf))}, dx=1000.0,
                               meter_filter=meta.get("meter_filter", "none"),
                               cap_duty=float(meta.get("cap_duty", 1.0)),
                               **input_options(meta))
        attach_sw_recharge(inp, meta)
        self.inp = inp
        if sw_hist(inp) is not None or extra_hist(inp) is not None:
            raise SystemExit("the page theta carries no canal field and no extra abstraction")
        self.E = np.asarray(inp.E_total.cpu().numpy(), float)
        self.recharge = np.asarray(inp.recharge_field.cpu().numpy(), float)
        self.h0 = inp.initial_heads(0, n_layers=N_LAYERS)
        self.pump_layer = int(meta.get("pump_layer", 1))
        self.recharge_layer = int(meta.get("recharge_layer", 0))
        self.device = device

    def run_member(self, i: int, E: np.ndarray, recharge: np.ndarray,
                   month0: int | None = None) -> np.ndarray:
        """Heads ``(L, A, T)`` of set ``i``; month 0 is the set's initial field, the steps are
        months 1..T-1 (forward.py's convention, R1cRun.run_from's call)."""
        from hydrophysics.twin.forward import build_model, rollout

        t = self.torch
        model, scalars, _ = build_model(self.inp.grid, self.members[i], self.device)
        Et = t.as_tensor(np.asarray(E, float)[..., 1:], dtype=t.float64)
        Rt = t.as_tensor(np.asarray(recharge, float)[:, 1:], dtype=t.float64)
        with t.no_grad():
            H = rollout(model, scalars, self.h0, Et, Rt, self.inp.ground_elev,
                        self.pump_layer, self.recharge_layer, month0=month0)
        return H.cpu().numpy()

    def ensemble(self, E: np.ndarray, recharge: np.ndarray, month0: int | None = None
                 ) -> np.ndarray:
        acc = None
        for i in range(len(self.members)):
            h = self.run_member(i, E, recharge, month0)
            acc = h if acc is None else acc + h
        return acc / len(self.members)

    def native(self) -> np.ndarray:
        """The native 2012-22 free run (forward.py's hindcast at gain 0)."""
        return self.ensemble(self.E, self.recharge)

    def county_cells(self, county: str) -> np.ndarray:
        """Active cells whose centroid lies in the county polygon (zone_county's rule)."""
        from hydrophysics.twin.zone_county import county_of_points

        c, _ = county_of_points(self.inp.grid.centroids(), str(IO.rp(K.COUNTY_GPKG)))
        return np.asarray(c).astype(str) == K.COUNTY_ZH[county]

    def grid_cells(self, x, y) -> np.ndarray:
        g = self.inp.grid
        out = []
        for a, b in zip(np.asarray(x, float), np.asarray(y, float), strict=True):
            i = g.active_index(float(a), float(b))
            out.append(-1 if i is None else int(i))
        return np.array(out, int)


def page_forward(keys=("heads_mean", "dates", "scenario_names")) -> dict:
    """The registered page forward (sha-checked), model output only."""
    p = IO.rp(K.PAGE_FORWARD)
    if IO.sha256(p) != K.PAGE_FORWARD_SHA:
        raise SystemExit(f"{K.PAGE_FORWARD} differs from cf2c7154")
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in keys}


def _write_json_once(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(obj, f, indent=1, default=float)


def _stamp(device: str) -> dict:
    return {"created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "code_head": HC.code_head(), "declared_code_head": K.WT_HEAD,
            "device": device,
            "code_sha256": K.code_shas(),
            "thetas_sha256": theta_shas(), "rain": K.PAGE_RF,
            "rain_sha256": IO.sha_or_missing(IO.rp(K.PAGE_RF))}


# ----------------------------------------------------------------------------- E0
def e0_compare(native: np.ndarray, ref: dict) -> dict:
    """max |native - page| over every layer, cell and month 2012-01..2022-12 (baseline)."""
    names = [str(s) for s in ref["scenario_names"]]
    dates = [str(d)[:7] for d in ref["dates"]]
    if names[0] != "baseline" or dates[0] != K.FIT[0] or \
            dates[K.PAGE_FIT_MONTHS - 1] != K.FIT[1]:
        raise SystemExit("the page forward's axis or scenario order is not the declared one")
    page = np.asarray(ref["heads_mean"][0][..., :K.PAGE_FIT_MONTHS], float)
    if native.shape != page.shape:
        raise SystemExit(f"native {native.shape} and page {page.shape} differ")
    d = float(np.max(np.abs(native - page)))
    return {"max_abs_diff_m": d, "tol_m": K.E0_TOL_M, "passed": bool(d <= K.E0_TOL_M),
            "shape": list(page.shape)}


def e0(out: Path, device: str, model=None, compile_matvec: bool = False) -> dict:
    target = out / "e0.json"
    if target.exists():
        raise SystemExit(f"{target} exists")
    m = PageRun(device=device, compile_matvec=compile_matvec) if model is None else model
    if len(m.inp.dates) != len(BL.FIT_M) or str(m.inp.dates[0])[:7] != K.FIT[0]:
        raise SystemExit("the page inputs' axis is not 2012-01..2022-12")
    rep = {"what": "E0: the page ensemble's native 2012-22 rollout against the registered "
                   "page forward cf2c7154 (model only)", **_stamp(device),
           "page_forward": K.PAGE_FORWARD, "page_forward_sha256": K.PAGE_FORWARD_SHA,
           **e0_compare(m.native(), page_forward())}
    _write_json_once(target, rep)
    log(f"E0 max |diff| {rep['max_abs_diff_m']:.2e} m -> {'PASS' if rep['passed'] else 'FAIL'}")
    if not rep["passed"]:
        raise SystemExit(K.E0_FAIL_RC)  # the logged E0 failure: not evaluable, stop
    return rep


# ----------------------------------------------------------------------------- swap
def swap(out: Path, device: str, model=None, compile_matvec: bool = False) -> dict:
    target = out / "swap.json"
    if target.exists():
        raise SystemExit(f"{target} exists")
    m = PageRun(device=device, compile_matvec=compile_matvec) if model is None else model
    months = BL.FIT_M
    E, R = m.E, m.recharge
    E3, info = HC.census_v3_energy(m, months)
    ws = L.window_slice(months, f"{K.SWAP_PATTERN_YEAR + 1}-01", K.FIT[1])
    nat = m.ensemble(E, R, month0=1)
    v3 = m.ensemble(E3, R, month0=1)
    gates = {}
    for c in K.COUNTIES:
        cells = m.county_cells(c)
        if not cells.any():
            raise SystemExit(f"ENGINEERING: no {c} cell")
        Bn = m.ensemble(L.pattern_energy(E, months, cells, K.SWAP_PATTERN_YEAR), R, month0=1)
        B3 = m.ensemble(L.pattern_energy(E3, months, cells, K.SWAP_PATTERN_YEAR), R, month0=1)
        cn = (nat - Bn)[:, cells][..., ws]
        c3 = (v3 - B3)[:, cells][..., ws]
        gates[c] = {**BL.swap_gate_county(cn, c3, float(E[cells].sum()), float(E3[cells].sum())),
                    "n_cells": int(cells.sum())}
    passed = all(g["passed"] for g in gates.values())
    rep = {"what": "the declared Yunlin+Changhua v3-vs-native energy swap gate on 2012-22 "
                   "(model only)", **_stamp(device), "census_v3": info, "counties": gates,
           "bands": {"energy": list(K.SWAP_ENERGY_BAND), "corr_min": K.SWAP_CORR_MIN,
                     "rms": list(K.SWAP_RMS_BAND)},
           "passed": bool(passed),
           "report_only": {"max_abs_head_diff_m_v3_vs_native": float(np.max(np.abs(v3 - nat)))}}
    _write_json_once(target, rep)
    log("swap gate: " + ", ".join(f"{c} {g['passed']}" for c, g in gates.items())
        + f" -> {'PASS' if passed else 'FAIL'}")
    return rep


# ----------------------------------------------------------------------------- backcast
def read_stations(path: Path):
    import pandas as pd

    st = pd.read_parquet(path, columns=["sid", "x", "y", "layer_code", "county"])
    st["sid"] = st["sid"].astype(str)
    st["county"] = st["county"].astype(str).str.lower()
    return st.sort_values("sid").reset_index(drop=True)


def backcast(stations: Path, out: Path, device: str, model=None,
             compile_matvec: bool = False) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    target = out / Path(K.PANEL).name
    if target.exists():
        raise SystemExit(f"{target} exists")
    st = read_stations(stations)
    m = PageRun(device=device, compile_matvec=compile_matvec) if model is None else model
    lay = st["layer_code"].astype(str).map({"1": 0, "2": 1, "3": 2, "4": 3}).fillna(-1)
    st = st.assign(layer=lay.astype(int).to_numpy(), cell=m.grid_cells(st["x"], st["y"]))
    placed = ((st["layer"] >= 0) & (st["cell"] >= 0)).to_numpy()
    E3, info = HC.census_v3_energy(m, BL.RUN_M)
    R = HC.calendar_climatology(m.recharge, BL.FIT_M, BL.RUN_M)
    yl = m.county_cells("yunlin")
    arms = {"F1": E3, "F0": HC.calendar_climatology(E3, BL.RUN_M, BL.RUN_M),
            "B_yl": L.pattern_energy(E3, BL.RUN_M, yl, K.PATTERN_YEAR)}
    H = {a: m.ensemble(Ea, R, month0=1) for a, Ea in arms.items()}
    if not all(np.isfinite(h).all() for h in H.values()):
        raise SystemExit("ENGINEERING: non-finite heads")
    ws = L.window_slice(BL.RUN_M, *K.WINDOW)
    sp = st[placed].reset_index(drop=True)
    layer, cell = sp["layer"].to_numpy(int), sp["cell"].to_numpy(int)
    ref = page_forward()
    panel = {"months": np.array(BL.WIN_M), "run_months": np.array(BL.RUN_M),
             "fit_months": np.array(BL.FIT_M), "sid": sp["sid"].to_numpy(str),
             "county": sp["county"].to_numpy(str), "x": sp["x"].to_numpy(float),
             "y": sp["y"].to_numpy(float), "layer": layer, "cell": cell,
             "not_placed": st.loc[~placed, "sid"].to_numpy(str),
             "native_fit": np.asarray(ref["heads_mean"][0][layer, cell, :K.PAGE_FIT_MONTHS],
                                      float)}
    for a in arms:
        panel[a] = L.screen_series(H[a], layer, cell)[:, ws]
    isy = panel["county"] == "yunlin"
    panel["sid_yl"] = panel["sid"][isy]
    panel["A_yl"] = panel["F1"][isy]
    panel["B_yl"] = panel["B_yl"][isy]
    panel["c_yl"] = panel["A_yl"] - panel["B_yl"]
    bl = L.well_blocks(panel["sid_yl"], panel["x"][isy], panel["y"][isy])
    panel["block_yl"] = np.array([bl[s] for s in panel["sid_yl"]])
    panel["a_yl"] = L.county_anomaly(E3, BL.RUN_M, yl)[ws]
    panel["yunlin_cells"] = yl
    with target.open("xb") as fh:
        np.savez(fh, **panel)
    summary = {"what": "I1.backcast_2010_11.v1 page-model backcast panel (model only)",
               **_stamp(device), "stations": str(stations),
               "stations_sha256": IO.sha256(stations), "census_v3": info,
               "n_placed": int(placed.sum()), "n_not_placed": int((~placed).sum()),
               "forcing": {"F1": "pump_electricity preorigin v3 cells_by_class value, all "
                                 "classes, 2007-01..2011-12 (forcing_frozen)",
                           "F0": "F1's per-cell calendar-month climatology over 2007-11",
                           "B_yl": f"Yunlin cells at their {K.PATTERN_YEAR} monthly pattern "
                                   f"from {K.PATTERN_YEAR + 1}-01, other cells F1",
                           "recharge": "the page inputs' 2012-22 recharge field (page rain "
                                       "file), calendar-month climatology (climatological "
                                       "rain)",
                           "initial_field": "the page inputs' common month-0 field "
                                            "(2012-01, inp.initial_heads(0)) at 2007-01",
                           "spin_up": f"{K.BC_RUN[0]}..{K.SPINUP_END}"},
               "native_fit": f"{K.PAGE_FORWARD} heads_mean[baseline] months 0..131",
               "panel": str(target), "panel_sha256": IO.sha256(target),
               "keys": {k: list(np.asarray(v).shape) for k, v in panel.items()}}
    _write_json_once(out / Path(K.PANEL_JSON).name, summary)
    log(f"wrote {target} sha256 {summary['panel_sha256']}")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("e0", "swap", "backcast"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True, type=Path)
        p.add_argument("--device", default=os.environ.get("BC11_DEVICE", "cpu"))
        p.add_argument("--compile-matvec", action="store_true")
        if name == "backcast":
            p.add_argument("--stations", required=True, type=Path,
                           help="L1_pre2012 v2's station_meta.parquet")
    a = ap.parse_args(argv)
    import attempts_bc11 as AT

    if a.cmd == "e0":
        return AT.run_step("e0", lambda: e0(a.out, a.device, compile_matvec=a.compile_matvec),
                           [str(a.out / "e0.json")])
    if a.cmd == "swap":
        return AT.run_step("swap", lambda: swap(a.out, a.device,
                                                compile_matvec=a.compile_matvec),
                           [str(a.out / "swap.json")])
    return AT.run_step("hindcast", lambda: backcast(a.stations, a.out, a.device,
                                                    compile_matvec=a.compile_matvec),
                       [str(a.out / Path(K.PANEL).name), str(a.out / Path(K.PANEL_JSON).name)])


if __name__ == "__main__":
    raise SystemExit(main())
