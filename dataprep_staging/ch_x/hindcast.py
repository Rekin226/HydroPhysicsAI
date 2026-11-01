"""Model-only free-run hindcasts of R1c for I1.CHX.v1 and I1.CH_backcast.v1 (the lead runs this,
in the frozen worktree, through ``run_chx.sh`` / ``run_backcast.sh``).

No observation is read or printed. ``load_twin_inputs`` loads the calibration's own inputs
(the 158 calibration wells' spent 2012-22 heads build R1c's month-0 field, exactly as every
R1c run does); nothing here compares a model value with an observation. Outputs are written
once (mode 'x'); stdout carries only shapes, check flags and sha256s.

Subcommands
-----------
``chx --out DIR``
    Three free-run hindcasts of R1c's main parameter set over 2012-01..2022-12 (no
    assimilation): A recorded pumping in both counties; B Changhua cells at their 2012
    monthly pattern from 2013-01 (Yunlin recorded); C (placebo) the Yunlin cells (centroid in
    雲林縣 and not in R1c's Changhua map) at their 2012 pattern, every other cell recorded. Heads at each external screen's cell and layer (N3F's
    export, metadata columns only), the Changhua and Yunlin pumping anomalies, and the
    engineering checks -> DIR/chx_panel.npz + chx_panel.json.
``s0 --out DIR``
    The backcast's pre-gates on 2012-22 (model only). (a) Code-path identity: the forcing
    assembled by ``bc_forcing`` rolled out by ``run_from`` equals the native rollout (max
    |diff| <= 0.01 m; a plumbing check: it does not test the 2007 placement, which has no
    reference run). (b) The declared forcing-swap gate: census preorigin v3 energy in place
    of R1c's model_census v1 must keep the Changhua 2012-22 energy within
    S0_SWAP_ENERGY_BAND and R1c's Changhua-only contrast (Changhua at its 2012 pattern) at
    every Changhua cell and layer over 2013-22 with per-cell-demeaned pooled correlation >=
    S0_SWAP_CORR_MIN and RMS ratio in S0_SWAP_RMS_BAND. Report only: the climatological
    recharge's effect (v3 energy, recharge climatology vs native recharge).
``backcast --stations PARQUET --out DIR``
    A and B (Changhua at its 2007 pattern from 2008-01) from 2007-01 with R1c's month-0
    field, census preorigin v3 energy, recharge as R1c's 2012-22 calendar-month
    climatology; heads at the L1_pre2012 screens over 2010-01..2011-12 -> bc_panel.npz.

Usage (repository root; PYTHONPATH pinned to the worktree by the runner):
    $PY dataprep_staging/ch_x/hindcast.py chx --out results/twin_runs/chx/hindcast
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import chx_const as K  # noqa: E402
import chx_io as IO  # noqa: E402
import chx_lib as L  # noqa: E402

RUN = L.month_list(*K.RUN_MONTHS)
WIN = L.month_list(*K.CHX_WINDOW)
BC_RUN = L.month_list(*K.BC_RUN)
BC_WIN = L.month_list(*K.BC_WINDOW)


def log(msg: str) -> None:
    print(f"{dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} {msg}", flush=True)


def code_head() -> str:
    """Short commit of the checkout that provides ``hydrophysics`` (the frozen worktree)."""
    import hydrophysics

    root = Path(hydrophysics.__file__).resolve().parents[1]
    try:
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "--short=7", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain",
                                "--untracked-files=no"], capture_output=True, text=True,
                               check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return head + ("-dirty" if dirty else "")


# ----------------------------------------------------------------------------- the model
class R1cRun:
    """R1c's main parameter set rebuilt as forward.py and policy_lib.TwinModel rebuild it, with
    no column (heads only)."""

    def __init__(self, theta: str = K.R1C_THETA, device: str = "cpu"):
        import torch

        from hydrophysics.twin.forward import (
            attach_sw_recharge,
            build_model,
            extra_hist,
            load_members,
            sw_hist,
        )
        from hydrophysics.twin.inputs import input_options, load_twin_inputs
        from hydrophysics.twin.zone_county import county_flags_from_meta

        self.torch = torch
        self.member = m = load_members([str(IO.rp(theta))])[0]
        self.meta = m.meta
        inp = load_twin_inputs(dx=1000.0, meter_filter=m.meta.get("meter_filter", "none"),
                               cap_duty=float(m.meta.get("cap_duty", 1.0)),
                               **input_options(m.meta))
        attach_sw_recharge(inp, m.meta)
        self.inp = inp
        self.model, self.scalars, _ = build_model(inp.grid, m, device)
        ec = m.meta.get("eta_classes")
        if ec:
            self.E = np.stack([np.asarray(inp.E_by_class[c], float) for c in ec])
        else:
            self.E = np.asarray(sum(inp.E_by_class.values()), float)
        self.recharge = np.asarray(inp.recharge_field.cpu().numpy(), float)
        self._sw, self._extra = sw_hist(inp), extra_hist(inp)
        ch = county_flags_from_meta(inp.grid, m.meta)
        if ch is None:
            raise SystemExit("R1c's meta records no zone_county_split; not R1c")
        self.changhua = np.asarray(ch, bool)
        if self.E.ndim != 2:
            raise SystemExit("R1c is expected to carry single-class energy (no eta_classes)")
        self.h0 = inp.initial_heads(0)
        self.pump_layer = int(m.meta.get("pump_layer", 1))
        self.recharge_layer = int(m.meta.get("recharge_layer", 0))
        self.device = device

    def run_from(self, E: np.ndarray, recharge: np.ndarray, h0=None, month0: int | None = None,
                 sw=None, q_extra=None) -> np.ndarray:
        """Heads ``(L, A, T)`` on an axis of T months whose month 0 is the initial field:
        ``E`` ``(A, T)`` (or ``(K, A, T)``) and ``recharge`` ``(A, T)`` on that axis; the
        rollout's steps are months 1..T-1 (forward.py's convention)."""
        from hydrophysics.twin.forward import rollout

        t = self.torch
        Et = t.as_tensor(np.asarray(E, float)[..., 1:], dtype=t.float64)
        Rt = t.as_tensor(np.asarray(recharge, float)[:, 1:], dtype=t.float64)
        with t.no_grad():
            H = rollout(self.model, self.scalars, self.h0 if h0 is None else h0, Et, Rt,
                        self.inp.ground_elev, self.pump_layer, self.recharge_layer,
                        sw_field=sw, q_extra=q_extra, month0=month0)
        return H.cpu().numpy()

    def heads(self, E: np.ndarray | None = None) -> np.ndarray:
        """The native 2012-22 free run (policy_lib.TwinModel.heads' call)."""
        return self.run_from(self.E if E is None else E, self.recharge, sw=self._sw,
                             q_extra=self._extra)

    def yunlin_cells(self) -> np.ndarray:
        """Active cells whose centroid lies in 雲林縣 (zone_county's polygon rule) and that
        R1c's county map does not flag Changhua: the placebo arm's cells."""
        from hydrophysics.twin.zone_county import county_of_points

        county, _ = county_of_points(self.inp.grid.centroids(), str(IO.rp(K.COUNTY_GPKG)))
        return (np.asarray(county).astype(str) == K.YUNLIN_ZH) & ~self.changhua

    def grid_cells(self, x, y) -> np.ndarray:
        g = self.inp.grid
        out = []
        for a, b in zip(np.asarray(x, float), np.asarray(y, float), strict=True):
            i = g.active_index(float(a), float(b))
            out.append(-1 if i is None else int(i))
        return np.array(out, int)


def theta_sha() -> str:
    return IO.sha256(IO.rp(K.R1C_THETA))


# ----------------------------------------------------------------------------- chx
def chx(out: Path, device: str, check_twinmodel: bool) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    target = out / "chx_panel.npz"
    if target.exists():
        raise SystemExit(f"{target} exists; panels are never rebuilt (archive it first)")
    ext = IO.ext_wells()
    ch, yl = IO.county_rows(ext, "changhua"), IO.county_rows(ext, "yunlin")
    if len(ch) != K.N_EXT_CHANGHUA:
        raise SystemExit(f"{len(ch)} kept Changhua externals, expected {K.N_EXT_CHANGHUA}")
    m = R1cRun(device=device)
    if len(m.inp.dates) != len(RUN) or str(m.inp.dates[0])[:7] != RUN[0]:
        raise SystemExit("R1c's input axis is not 2012-01..2022-12")
    checks = {}
    for name, t in (("changhua", ch), ("yunlin", yl)):
        cells = m.grid_cells(t["x"], t["y"])
        checks[f"cells_match_export_{name}"] = bool(np.array_equal(cells,
                                                                   t["cell"].to_numpy(int)))
    if not all(checks.values()):
        raise SystemExit(f"ENGINEERING: export cells differ from R1c's grid: {checks}")
    E = m.E
    yun = m.yunlin_cells()
    if not yun.any():
        raise SystemExit("ENGINEERING: no Yunlin cell found for the placebo arm")
    checks["n_cells"] = {"changhua": int(m.changhua.sum()), "yunlin": int(yun.sum()),
                         "other": int((~m.changhua & ~yun).sum())}
    arms = {"A": E, "B": L.pattern_energy(E, RUN, m.changhua, K.PATTERN_YEAR),
            "C": L.pattern_energy(E, RUN, yun, K.PATTERN_YEAR)}
    H = {}
    for a, Ea in arms.items():
        H[a] = m.heads(Ea)
        log(f"arm {a}: heads {H[a].shape}")
    if check_twinmodel:
        sys.path.insert(0, str(HERE.parent))
        from n1_policy.policy_lib import TwinModel
        tm = TwinModel(str(IO.rp(K.R1C_THETA)), str(IO.rp(K.R1C_COLUMN)), device=device)
        d = float(np.max(np.abs(tm.heads() - H["A"])))
        checks["twinmodel_max_abs_diff_m"] = d
        checks["twinmodel_reproduced"] = bool(d <= 1e-9)
        if d > 1e-9:
            raise SystemExit(f"ENGINEERING: arm A differs from policy_lib.TwinModel by {d} m")
    checks["finite"] = bool(all(np.isfinite(h).all() for h in H.values()))
    if not checks["finite"]:
        raise SystemExit("ENGINEERING: non-finite heads")
    ws = L.window_slice(RUN, *K.CHX_WINDOW)
    panel = {"months": np.array(WIN), "run_months": np.array(RUN)}
    for name, t in (("ch", ch), ("yl", yl)):
        lay, cel = t["layer"].to_numpy(int), t["cell"].to_numpy(int)
        for a in arms:
            panel[f"{a}_{name}"] = L.screen_series(H[a], lay, cel)[:, ws]
        panel[f"c_{name}"] = panel[f"A_{name}"] - panel[f"B_{name}"]
        panel[f"c_placebo_{name}"] = panel[f"A_{name}"] - panel[f"C_{name}"]
        panel[f"sid_{name}"] = t["station_id"].to_numpy(str)
        panel[f"x_{name}"] = t["x"].to_numpy(float)
        panel[f"y_{name}"] = t["y"].to_numpy(float)
        panel[f"layer_{name}"] = lay
        panel[f"cell_{name}"] = cel
        bl = L.well_blocks(panel[f"sid_{name}"], panel[f"x_{name}"], panel[f"y_{name}"])
        panel[f"block_{name}"] = np.array([bl[s] for s in panel[f"sid_{name}"]])
    panel["a_ch"] = L.county_anomaly(E, RUN, m.changhua)[ws]
    panel["a_yl"] = L.county_anomaly(E, RUN, yun)[ws]
    panel["changhua_cells"] = m.changhua
    panel["yunlin_cells"] = yun
    with target.open("xb") as fh:
        np.savez(fh, **panel)
    summary = {
        "what": "I1.CHX.v1 hindcast panel (model only; no observation read)",
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "theta": K.R1C_THETA, "theta_sha256": theta_sha(), "code_head": code_head(),
        "declared_code_head": K.WT_HEAD, "device": device,
        "ext_wells": K.EXT_WELLS, "ext_wells_sha256": IO.sha256(IO.rp(K.EXT_WELLS)),
        "rows_sha256_changhua": IO.rows_sha(ch), "rows_sha256_yunlin": IO.rows_sha(yl),
        "n_changhua": int(len(ch)), "n_yunlin": int(len(yl)),
        "arms": {"A": "recorded both counties",
                 "B": f"Changhua cells at their {K.PATTERN_YEAR} monthly pattern from "
                      f"{K.PATTERN_YEAR + 1}-01, rest recorded",
                 "C": f"Yunlin cells at their {K.PATTERN_YEAR} pattern, every other cell "
                      "recorded (placebo)"},
        "checks": checks, "panel": str(target), "panel_sha256": IO.sha256(target),
        "keys": {k: list(np.asarray(v).shape) for k, v in panel.items()}}
    with (out / "chx_panel.json").open("x") as f:
        json.dump(summary, f, indent=1)
    log(f"wrote {target} sha256 {summary['panel_sha256']}")
    return summary


# ----------------------------------------------------------------------------- backcast
def calendar_climatology(R: np.ndarray, months: list[str], out_months: list[str]
                         ) -> np.ndarray:
    """``(A, len(out_months))``: each output month gets the mean of R over the same
    calendar month of ``months``."""
    mo = np.array([int(m[5:7]) for m in months])
    clim = np.stack([np.asarray(R, float)[:, mo == k].mean(axis=1) for k in range(1, 13)],
                    axis=1)
    om = np.array([int(m[5:7]) for m in out_months])
    return clim[:, om - 1]


def census_v3_energy(m: R1cRun, months: list[str], path: str = K.CENSUS_V3_FILE
                     ) -> tuple[np.ndarray, dict]:
    """All-class cell energy ``(A, T)`` of pump_electricity preorigin v3 on ``months`` (use
    forcing_frozen through the guard), after checking the grid against R1c's."""
    from common import guard

    prod = guard.open_product(IO.rp(K.CENSUS_V3), "forcing_frozen")
    info = json.loads((IO.rp(K.CENSUS_V3) / "build_info.json").read_text())["grid"]
    g = m.inp.grid
    ok = (int(info["nx"]) == g.nx and int(info["ny"]) == g.ny
          and abs(float(info["dx"]) - float(g.dx)) < 1e-6
          and abs(float(info["x0"]) - float(g.x0)) < 1e-6
          and abs(float(info["y0"]) - float(g.y0)) < 1e-6
          and int(info["n_active"]) == g.n_active)
    if not ok:
        raise SystemExit("ENGINEERING: census preorigin v3 grid differs from R1c's grid")
    with np.load(prod.path(Path(path).name), allow_pickle=False) as z:
        dates = [str(d)[:7] for d in z["dates"]]
        val = np.asarray(z["value"], float)
        flag = np.asarray(z["flag"])
    idx = [dates.index(mo) for mo in months]
    E = val[:, :, idx].sum(axis=0)
    if not np.isfinite(E).all():
        raise SystemExit("census v3 energy has non-finite cell-months on the axis")
    return E, {"product": K.CENSUS_V3, "file_sha256": IO.sha256(prod.path(
               Path(path).name)), "use": "forcing_frozen", "guard_notes": list(prod.notes),
               "n_flag8_cell_months": int((flag[:, :, idx] == 8).sum())}


def bc_forcing(m: R1cRun, months: list[str], source: str) -> tuple[np.ndarray, np.ndarray]:
    """(E, recharge) on ``months``: ``source`` 'native' (R1c's own inputs; the S0 path, axis
    must be 2012-22) or 'v3' (census preorigin v3 energy; recharge = R1c's 2012-22
    calendar-month climatology unless the axis is R1c's own, where the native field is
    used)."""
    if source == "native":
        if months != RUN:
            raise ValueError("the native source exists on 2012-01..2022-12 only")
        return m.E, m.recharge
    E, _ = census_v3_energy(m, months)
    R = m.recharge if months == RUN else calendar_climatology(m.recharge, RUN, months)
    return E, R


def swap_gate(cn: np.ndarray, c3: np.ndarray, e_native: float, e_v3: float) -> dict:
    """S0's declared forcing-swap gate from the two Changhua contrasts ``(..., T)`` (native and
    v3 energy) and the two Changhua energy totals."""
    dn = cn - cn.mean(axis=-1, keepdims=True)
    d3 = c3 - c3.mean(axis=-1, keepdims=True)
    den = float(np.sqrt((dn ** 2).sum() * (d3 ** 2).sum()))
    corr = float((dn * d3).sum() / den) if den > 0 else float("nan")
    rn = float(np.sqrt(np.mean(dn ** 2)))
    rms = float(np.sqrt(np.mean(d3 ** 2)) / rn) if rn > 0 else float("nan")
    er = e_v3 / e_native if e_native > 0 else float("nan")
    ok = {"energy_ratio_ok": bool(K.S0_SWAP_ENERGY_BAND[0] <= er <= K.S0_SWAP_ENERGY_BAND[1]),
          "contrast_corr_ok": bool(np.isfinite(corr) and corr >= K.S0_SWAP_CORR_MIN),
          "contrast_rms_ratio_ok": bool(K.S0_SWAP_RMS_BAND[0] <= rms <= K.S0_SWAP_RMS_BAND[1])}
    return {"energy_ratio_changhua_2012_22": er, "contrast_corr": corr,
            "contrast_rms_ratio": rms, **ok, "passed": all(ok.values()),
            "bands": {"energy": list(K.S0_SWAP_ENERGY_BAND), "corr_min": K.S0_SWAP_CORR_MIN,
                      "rms": list(K.S0_SWAP_RMS_BAND)}}


def s0(out: Path, device: str, model=None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    target = out / "s0.json"
    if target.exists():
        raise SystemExit(f"{target} exists")
    m = R1cRun(device=device) if model is None else model
    if m._sw is not None or m._extra is not None:
        raise SystemExit("R1c is expected to carry no canal field and no extra abstraction")
    native = m.heads()
    E, R = bc_forcing(m, RUN, "native")
    path = m.run_from(E, R, month0=1)
    d = float(np.max(np.abs(path - native)))
    E3, R3 = bc_forcing(m, RUN, "v3")
    ch = m.changhua
    ws = L.window_slice(RUN, *K.CHX_WINDOW)
    v3 = m.run_from(E3, R3, month0=1)
    Bn = m.run_from(L.pattern_energy(E, RUN, ch, K.PATTERN_YEAR), R, month0=1)
    B3 = m.run_from(L.pattern_energy(E3, RUN, ch, K.PATTERN_YEAR), R3, month0=1)
    cn = (path - Bn)[:, ch][..., ws]
    c3 = (v3 - B3)[:, ch][..., ws]
    swap = swap_gate(cn, c3, float(E[ch].sum()), float(E3[ch].sum()))
    clim = m.run_from(E3, calendar_climatology(m.recharge, RUN, RUN), month0=1)
    passed = bool(d <= K.S0_TOL_M and swap["passed"])
    rep = {"what": "S0: code-path identity and the declared census-v3 forcing-swap gate on "
                   "2012-22 (model only)",
           "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "theta_sha256": theta_sha(), "code_head": code_head(),
           "max_abs_diff_m": d, "tol_m": K.S0_TOL_M, "identity_passed": bool(d <= K.S0_TOL_M),
           "swap_gate": swap, "passed": passed,
           "report_only": {
               "max_abs_head_diff_m_v3_vs_native": float(np.max(np.abs(v3 - native))),
               "max_abs_head_diff_m_changhua_2013_22": float(
                   np.max(np.abs((v3 - native)[:, ch][..., ws]))),
               "max_abs_head_diff_m_recharge_climatology": float(
                   np.max(np.abs(clim - v3))),
               "note": "the 2007 placement of the 2012-01 field has no reference run; the "
                       "36-month spin-up is the declared remedy"}}
    with target.open("x") as f:
        json.dump(rep, f, indent=1)
    log(f"S0 identity {rep['identity_passed']}; swap gate {swap['passed']} -> "
        f"{'PASS' if passed else 'FAIL'}")
    return rep


def backcast(stations: Path, out: Path, device: str, model=None) -> dict:
    import pandas as pd

    out.mkdir(parents=True, exist_ok=True)
    target = out / "bc_panel.npz"
    if target.exists():
        raise SystemExit(f"{target} exists")
    st = pd.read_parquet(stations, columns=["sid", "x", "y", "layer_code", "county"])
    st = st[st["county"].astype(str).str.lower() == "changhua"].sort_values("sid")
    m = R1cRun(device=device) if model is None else model
    lay = st["layer_code"].astype(str).map({"1": 0, "2": 1, "3": 2, "4": 3}).fillna(-1)
    st = st.assign(layer=lay.astype(int).to_numpy(), cell=m.grid_cells(st["x"], st["y"]))
    placed = (st["layer"] >= 0) & (st["cell"] >= 0)
    E, R = bc_forcing(m, BC_RUN, "v3")
    arms = {"A": E, "B": L.pattern_energy(E, BC_RUN, m.changhua, K.BC_PATTERN_YEAR)}
    H = {a: m.run_from(Ea, R, month0=1) for a, Ea in arms.items()}
    if not all(np.isfinite(h).all() for h in H.values()):
        raise SystemExit("ENGINEERING: non-finite heads")
    ws = L.window_slice(BC_RUN, *K.BC_WINDOW)
    sp = st[placed]
    panel = {"months": np.array(BC_WIN), "run_months": np.array(BC_RUN),
             "sid": sp["sid"].to_numpy(str), "x": sp["x"].to_numpy(float),
             "y": sp["y"].to_numpy(float), "layer": sp["layer"].to_numpy(int),
             "cell": sp["cell"].to_numpy(int),
             "not_placed": st.loc[~placed, "sid"].to_numpy(str)}
    for a in arms:
        panel[a] = L.screen_series(H[a], panel["layer"], panel["cell"])[:, ws]
    panel["c"] = panel["A"] - panel["B"]
    bl = L.well_blocks(panel["sid"], panel["x"], panel["y"])
    panel["block"] = np.array([bl[s] for s in panel["sid"]])
    panel["a_ch"] = L.county_anomaly(E, BC_RUN, m.changhua)[ws]
    with target.open("xb") as fh:
        np.savez(fh, **panel)
    summary = {"what": "I1.CH_backcast.v1 hindcast panel (model only)",
               "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
               "theta_sha256": theta_sha(), "code_head": code_head(),
               "stations": str(stations), "stations_sha256": IO.sha256(stations),
               "n_placed": int(placed.sum()), "n_not_placed": int((~placed).sum()),
               "forcing": {"energy": "pump_electricity preorigin v3 cells_by_class value, all "
                                     "classes, 2007-01..2011-12 (forcing_frozen)",
                           "recharge": "R1c's 2012-22 recharge field, calendar-month "
                                       "climatology", "canal": "none (R1c has none)",
                           "initial_field": "R1c's own month-0 field (2012-01) at 2007-01",
                           "spin_up": "2007-01..2009-12"},
               "panel": str(target), "panel_sha256": IO.sha256(target)}
    with (out / "bc_panel.json").open("x") as f:
        json.dump(summary, f, indent=1)
    log(f"wrote {target} sha256 {summary['panel_sha256']}")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("chx", "s0", "backcast"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True, type=Path)
        p.add_argument("--device", default=os.environ.get("CHX_DEVICE", "cpu"))
        if name == "chx":
            p.add_argument("--check-twinmodel", action="store_true",
                           help="also rebuild arm A through policy_lib.TwinModel (needs R1c's "
                                "column JSON) and require identity to 1e-9 m")
        if name == "backcast":
            p.add_argument("--stations", required=True, type=Path,
                           help="L1_pre2012's station_meta.parquet")
    a = ap.parse_args(argv)
    if a.cmd == "chx":
        chx(a.out, a.device, a.check_twinmodel)
    elif a.cmd == "s0":
        s0(a.out, a.device)
    else:
        backcast(a.stations, a.out, a.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
