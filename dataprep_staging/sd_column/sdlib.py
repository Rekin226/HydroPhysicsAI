"""Shared model-side layer of round 7 ranks 9 and 10 (``dataprep_staging/sd_column``).

* SD.cut_response_spread.v1 (rank 9, model-only declaration): cut30 avoided subsidence of
  every leveling-equivalent column on its own paired heads, main parameter set only.
* X5.gnss_seasonal_column.v1 (rank 10, design-time test): the same forwards' free hindcast
  (2012-01..2022-12) gives the column's seasonal subsidence at the GNSS station cells.

Every forward is ONE parameter set (the calibration's main theta), ``--ic-members 0`` and
``--gain 0.0``. With ``--column-heads free`` the column is driven by the free projection
from the hindcast's own end state, so with gain 0 the saved ``heads_mean`` IS the head field
that drives the column (``forward.run``: ``heads_col`` = ``heads`` when the restart gain is
0). That makes the elastic/inelastic split exact: the elastic term of the VEP recurrence is
``Ske (h0 - h_t)`` on the layer-mean head (``compaction.vep_compaction``); the rest (the
preconsolidation-gated, delayed term and the aquitard term) is inelastic.

Nothing in this module reads an observation. Paths are repository-relative and resolve
against ``common.paths.REPO`` (``DATAPREP_REPO`` overrides it), the directory that holds
``results/``, ``chou-shui-data/`` and this code.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
X = "dataprep_staging/sd_column"

SD_ID = "SD.cut_response_spread.v1"
GSC_ID = "X5.gnss_seasonal_column.v1"

# --------------------------------------------------------------------------- the forwards
#: main 4c2723e: FS.start_invariance.v1 reproduced the page run with it (S0 repro exit 0,
#: 2026-11-01); hydrophysics/twin is unchanged from it to origin/main fc0ea4a except the page
#: files (template, claims, viewer). Every theta's commit is an ancestor of it.
CODE = "4c2723e"
WT_DIR = "HydroPhysicsAI-sdrun-4c2723e"           # under $HOME, clean, detached at CODE
RF_VENDOR = "chou-shui-data/data/rf_timeseries_vendor_2012_2022.csv"
RF_DEFAULT = "chou-shui-data/data/rf_timeseries.csv"
CUT30 = "cut30:irrigation=0.7@2026-01"
CUT30_DESC = "cut30: irrigation x0.7 from 2026-01"
BASE_DESC = "baseline: baseline (no change)"
FLAGS = ("--horizon 120 --gain 0.0 --ic-members 0 --seed 0 --update-method innovation "
         "--restart-taper-km 5 --apex-hold calibrated --column-heads free "
         "--column-hpc0-fast-days 365 --device cuda --compile-matvec --save-members none")
NUDGE_FLAGS = "--hindcast-gain 1.0 --hindcast-every 1"
RUNS = "results/twin_runs"
FWD = "results/twin_forward"
LOG = "results/twin_runs/sd_column"
N_CELLS = 2148
N_T = 252                                          # 2012-01..2032-12
ORIGIN = 131                                       # 2022-12
DATE0 = (2012, 1)

#: forward key -> theta run directory, rain file, extra flags, and its columns in rheology
#: order (column ids of COLUMNS). The rain is the file each theta was calibrated with: the
#: page theta predates the 2026-10-11 CWA rebuild and read the vendor cache (as FS); every
#: other theta records rf_timeseries.csv and its sha256 in its meta (checked at scoring).
FORWARDS = {
    "page": {"run": "stage3_datum_gate", "rf": RF_VENDOR, "extra": "",
             "columns": ("page", "page_tau30", "page_aquitard")},
    "q0": {"run": "stage3_datum_gate_q0", "rf": RF_DEFAULT, "extra": "",
           "columns": ("q0", "q0_s29")},
    "q0cs": {"run": "stage3_datum_gate_q0_cs", "rf": RF_DEFAULT, "extra": "",
             "columns": ("q0_cs", "q0_csk1", "q0_csk01_s29")},
    "cwa": {"run": "stage3_datum_gate_cwa", "rf": RF_DEFAULT, "extra": "",
            "columns": ("cwa", "cwa_tau30")},
    "q0nh": {"run": "stage3_datum_gate_q0_nh", "rf": RF_DEFAULT, "extra": NUDGE_FLAGS,
             "columns": ("q0_nh",)},
    "ccz": {"run": "stage3_datum_gate_ccz", "rf": RF_DEFAULT, "extra": "",
            "columns": ("ccz",)},
    "q0c": {"run": "stage3_datum_gate_q0c_csk1", "rf": RF_DEFAULT, "extra": "",
            "columns": ("q0c_csk1",)},
    "x5r1cs": {"run": "stage3_datum_gate_x5r1_cs", "rf": RF_DEFAULT, "extra": "",
               "columns": ("x5r1_cs",)},
}

#: column id -> (run directory, column subdirectory, tier, archive check). Tier A: the page
#: theta's own heads; B: the Q0 family and R0 (cwa) on their own heads; CH: the Changhua-
#: pumping tier, reported apart, never in a Yunlin reading. Archive check: ("members",
#: archive, member label, rheology) = the archive's per-member December fields of the main
#: member (ic 0); ("single", archive) = a one-theta archive whose hindcast months are the main
#: member's (the ic perturbation enters at the origin only); ("shared", column) = no stored
#: trajectory, checked through the column of the same forward (same heads) plus the CPU
#: rebuild; None = no stored trajectory that may be read. A column whose hindcast is not
#: checked against an independent stored trajectory passes on its paired theta and the CPU
#: rebuild only, carries the UNVERIFIED label, and every reading is reported with and
#: without it.
COLUMNS = {
    "page": ("stage3_datum_gate", "coupled_leveling", "A",
             ("members", "datum_gate_innov", "stage3_datum_gate", "coupled_leveling")),
    "page_tau30": ("stage3_datum_gate", "coupled_leveling_tau30", "A",
                   ("members", "datum_gate_innov", "stage3_datum_gate", "tau30y")),
    "page_aquitard": ("stage3_datum_gate", "coupled_leveling_aquitard", "A",
                      ("shared", "page")),
    "q0": ("stage3_datum_gate_q0", "coupled_leveling", "B", ("single", "datum_gate_q0")),
    "q0_s29": ("stage3_datum_gate_q0_s29", "coupled_leveling", "B",
               ("single", "datum_gate_q0_s29")),
    "q0_cs": ("stage3_datum_gate_q0_cs", "coupled_leveling", "B",
              ("single", "datum_gate_q0_cs")),
    "q0_csk1": ("stage3_datum_gate_q0_csk1", "coupled_leveling", "B",
                ("single", "datum_gate_q0_csk1")),
    "q0_csk01_s29": ("stage3_datum_gate_q0_csk01_s29", "coupled_leveling", "B",
                     ("single", "datum_gate_q0_csk01_s29")),
    "cwa": ("stage3_datum_gate_cwa", "coupled_leveling", "B",
            ("members", "datum_gate_cwa", "stage3_datum_gate_cwa", "coupled_leveling")),
    "cwa_tau30": ("stage3_datum_gate_cwa", "coupled_leveling_tau30", "B",
                  ("members", "datum_gate_cwa", "stage3_datum_gate_cwa", "tau30y")),
    "q0_nh": ("stage3_datum_gate_q0_nh", "coupled_leveling", "B", None),
    "ccz": ("stage3_datum_gate_ccz", "coupled_leveling", "CH",
            ("members", "datum_gate_ccz", "stage3_datum_gate_ccz", "coupled_leveling")),
    "q0c_csk1": ("stage3_datum_gate_q0c_csk1", "coupled_leveling", "CH",
                 ("single", "datum_gate_q0c_csk1")),
    "x5r1_cs": ("stage3_datum_gate_x5r1_cs", "coupled_leveling", "CH", None),
}
#: columns without an independent stored-trajectory check (label; readings with and without)
UNVERIFIED = {
    "page_aquitard": "unverified trajectory (shared heads): no archive stores its trajectory; "
                     "checked only through the page column of the same forward and the CPU "
                     "rebuild of the same load_columns code",
    "q0_nh": "unverified pairing (no clean reference): its only stored trajectory, "
             "datum_gate_q0_nh.npz, assimilated heads to 2025-06 (origin 2025-06, "
             "extend_record_to) and is never read (judge rank 9)",
    "x5r1_cs": "unverified pairing (no stored trajectory); Changhua tier only",
}
#: candidates the declaration lists but that cannot run (listed, never substituted)
DROPPED_AT_DECLARATION = {
    "page_aquitard_tau30": "results/twin_runs/stage3_datum_gate/coupled_leveling_aquitard_"
                           "tau30/ holds no vep_zonal_leveling.json (the fit never wrote one)",
}
PAGE_COLUMN = "page"
PAGE_COLUMN_SHA8 = "d290c592"
#: inclusion bar (judge rank 9): out-of-fold leveling R2 >= page - 0.02, from the published
#: column JSONs (spent fit metrics, used only as the inclusion rule they already were)
BAR_DELTA = 0.02
#: hindcast reproduction tolerances (metres)
TOL_SINGLE_M = 5e-4
TOL_MEMBERS_EXTRA_M = 5e-4
TOL_REBUILD_M = 1e-4


def sha(p) -> str:
    p = Path(p)
    if not p.exists():
        return "missing"
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha(d) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def theta_path(run: str) -> str:
    return f"{RUNS}/{run}/stage3_theta.json"


def column_path(col: str) -> str:
    run, sub, _, _ = COLUMNS[col]
    return f"{RUNS}/{run}/{sub}/vep_zonal_leveling.json"


def forward_of(col: str) -> str:
    for k, f in FORWARDS.items():
        if col in f["columns"]:
            return k
    raise KeyError(col)


def out_stem(fwd: str) -> str:
    return f"{FWD}/sd_{fwd}"


def forward_command(fwd: str, py: str = "$PY") -> str:
    """The exact forward command line of ``fwd`` (the runner prints the same with DRY=1)."""
    f = FORWARDS[fwd]
    vep = ",".join(column_path(c) for c in f["columns"])
    extra = f" {f['extra']}" if f["extra"] else ""
    return (f"{py} -m hydrophysics.twin.forward --theta {theta_path(f['run'])} "
            f"--scenario '{CUT30}' {FLAGS}{extra} --rf-timeseries {f['rf']} "
            f"--vep-json {vep} --out {out_stem(fwd)}")


def month_index(ym: str) -> int:
    y, m = (int(x) for x in ym.split("-")[:2])
    return 12 * (y - DATE0[0]) + (m - DATE0[1])


HORIZONS = {"2026-12": month_index("2026-12"), "2028-12": month_index("2028-12"),
            "2032-12": month_index("2032-12")}
PRIMARY_HORIZON = "2028-12"
F_SUBS_EARLY = "2026-06"           # six months into the cut (2026-01)
F_SUBS_LATE = "2028-12"


# --------------------------------------------------------------------------- inclusion
def inclusion(repo: Path, cols=None) -> dict:
    """Out-of-fold R2 of each declared column (published JSON field ``r2_outoffold``), the
    bar, and the decision. Tier CH is listed but never barred (reported apart)."""
    cols = list(cols or COLUMNS)
    page = json.loads((repo / column_path(PAGE_COLUMN)).read_text())["r2_outoffold"]
    bar = page - BAR_DELTA
    out = {"page_r2_outoffold": page, "bar": bar, "columns": {}}
    for c in cols:
        p = repo / column_path(c)
        if not p.exists():
            out["columns"][c] = {"included": False, "reason": "column JSON missing"}
            continue
        r2 = json.loads(p.read_text()).get("r2_outoffold")
        tier = COLUMNS[c][2]
        ok = tier == "CH" or (r2 is not None and r2 >= bar)
        out["columns"][c] = {"r2_outoffold": r2, "tier": tier, "included": bool(ok),
                             "reason": ("Changhua tier: reported apart, no bar" if tier == "CH"
                                        else ("meets the bar" if ok else "below the bar"))}
    return out


def param_hash(col_json: dict) -> str:
    """Hash of a column's parameter sets (zonal/banded/shared): two columns with the same hash
    are one distinct column (q0 and q0_s29 differ only in their fold seed)."""
    from_keys = col_json.get("zonal") or col_json.get("banded") or {
        k: col_json[k] for k in ("log_ske", "log_skv", "log_tau", "h_pc0") if k in col_json}
    return canonical_sha(from_keys)[:16]


def distinct_columns(repo: Path, cols) -> dict:
    """column id -> the id of the first listed column with the same parameters."""
    seen, out = {}, {}
    for c in cols:
        h = param_hash(json.loads((repo / column_path(c)).read_text()))
        out[c] = seen.setdefault(h, c)
    return out


# --------------------------------------------------------------------------- npz layout
def load_npz(path) -> dict:
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def rheology_labels(f: dict) -> list[str]:
    return [str(r) for r in f["rheology_labels"]]


def column_subs(z: dict, ri: int) -> np.ndarray:
    """(n_scen, A, T) subsidence of rheology ``ri`` of a forward npz (one member)."""
    if "subs_mean_by_rheology" in z:
        return np.asarray(z["subs_mean_by_rheology"][ri], dtype="float64")
    if ri != 0:
        raise ValueError("a one-column forward has rheology 0 only")
    return np.asarray(z["subs_mean"], dtype="float64")


def expected_labels(fwd: str) -> list[str]:
    """forward.load_columns labels: the column directory name, de-duplicated by index."""
    out = []
    for c in FORWARDS[fwd]["columns"]:
        lab = COLUMNS[c][1]
        if lab in out:
            lab = f"{lab}_{len(out)}"
        out.append(lab)
    return out


def check_forward_npz(z: dict, fwd: str) -> list[str]:
    """Layout checks of a scored forward before any value is used (empty = ok)."""
    e = []
    names = [str(s) for s in z["scenario_names"]]
    if names != ["baseline", "cut30"]:
        e.append(f"scenario names {names}")
    scen = [str(s) for s in z["scenarios"]]
    if scen != [BASE_DESC, CUT30_DESC]:
        e.append(f"scenario strings {scen}")
    d = [str(x)[:7] for x in z["dates"]]
    if len(d) != N_T or d[0] != "2012-01" or d[-1] != "2032-12":
        e.append(f"dates {d[:1]}..{d[-1:]} ({len(d)})")
    if int(z["origin"]) != ORIGIN:
        e.append(f"origin {int(z['origin'])}")
    if int(z["n_members"]) != 1:
        e.append(f"n_members {int(z['n_members'])}")
    lab = [str(m) for m in z["member_labels"]]
    if lab != [FORWARDS[fwd]["run"]]:
        e.append(f"member labels {lab}")
    if float(z["gain"]) != 0.0:
        e.append(f"gain {float(z['gain'])}")
    hg = float(z["hindcast_gain"])
    if hg != (1.0 if FORWARDS[fwd]["extra"] else 0.0):
        e.append(f"hindcast_gain {hg}")
    if rheology_labels(z) != expected_labels(fwd):
        e.append(f"rheology labels {rheology_labels(z)} != {expected_labels(fwd)}")
    fo = json.loads(str(z["forward_options"]))
    for k, v in (("column_heads", "free"), ("update_method", "innovation"),
                 ("apex_hold", "calibrated"), ("column_hpc0_fast_days", 365.0),
                 ("save_members", "none")):
        if fo.get(k) != v:
            e.append(f"forward_options[{k!r}] = {fo.get(k)!r}")
    for k in ("record_end", "extend_record_to", "climatology_end", "recorded_pumping"):
        if k in fo:
            e.append(f"forward_options carries {k}")
    if any(float(b) != 0.0 for b in fo.get("column_zone_blend_km", [])):
        e.append("a column is blended")
    if tuple(np.shape(z["heads_mean"])) != (2, 4, N_CELLS, N_T):
        e.append(f"heads_mean shape {np.shape(z['heads_mean'])}")
    return e


# --------------------------------------------------------------------------- reproduction
def hindcast_repro(z: dict, ri: int, check, repo: Path) -> dict:
    """Max |new - stored| subsidence over the hindcast (months 0..ORIGIN) of rheology ``ri``
    against the column's stored trajectory (``check`` from COLUMNS). Reads model output only."""
    new = column_subs(z, ri)[0, :, :ORIGIN + 1]
    if check is None:
        return {"route": "none", "pass": None, "note": "no stored trajectory (unverifiable)"}
    kind = check[0]
    if kind == "single":
        a = load_npz(repo / f"{FWD}/{check[1]}.npz")
        o = int(a["origin"])
        if o < ORIGIN:
            return {"route": "single", "pass": False, "note": f"archive origin {o}"}
        old = np.asarray(a["subs_mean"], dtype="float64")[0, :, :ORIGIN + 1]
        dmax = float(np.abs(new - old).max())
        return {"route": "single", "archive": f"{FWD}/{check[1]}.npz",
                "max_abs_m": dmax, "tol_m": TOL_SINGLE_M, "pass": dmax <= TOL_SINGLE_M}
    if kind == "members":
        _, arch, member, rheo = check
        m = load_npz(repo / f"{FWD}/{arch}.members.npz")
        mem = [str(x) for x in m["member"]]
        ic = [int(x) for x in m["ic"]]
        rh = [str(x) for x in m["rheology"]]
        idx = [i for i in range(len(mem)) if mem[i] == member and ic[i] == 0 and rh[i] == rheo]
        if len(idx) != 1:
            return {"route": "members", "pass": False,
                    "note": f"{len(idx)} sidecar rows for {member}/ic0/{rheo}"}
        q = float(m["subs_scale_m"])
        ye = [int(t) for t in m["ye_idx"]]
        keep = [k for k, t in enumerate(ye) if t <= ORIGIN]
        old = np.asarray(m["subs_members_yr"][0, idx[0]], dtype="float64")[:, keep] * q
        newd = new[:, [ye[k] for k in keep]]
        dmax = float(np.abs(newd - old).max())
        tol = 0.5 * q + TOL_MEMBERS_EXTRA_M
        return {"route": "members", "archive": f"{FWD}/{arch}.members.npz",
                "member": member, "rheology": rheo, "decembers": len(keep),
                "quantum_m": q, "max_abs_m": dmax, "tol_m": tol, "pass": dmax <= tol}
    if kind == "shared":
        return {"route": "shared", "via": check[1], "pass": None,
                "note": "decided with the via column and the CPU rebuild"}
    raise ValueError(f"unknown archive check {check!r}")


# --------------------------------------------------------------------------- the grid
def grid_and_masks(repo: Path) -> dict:
    """The 1 km grid from the fan polygon (no observation), the easting zones (0 proximal,
    1 mid, 2 distal; zones.fan_zones at 205/182 km) and the county flags (zone_county
    DEFAULT_MAP; True = Changhua)."""
    import os

    from hydrophysics.twin.grid import build_grid
    from hydrophysics.twin.inputs import resolve_paths
    from hydrophysics.twin.zone_county import DEFAULT_MAP, county_flags_for_grid
    from hydrophysics.twin.zones import fan_zones

    cwd = os.getcwd()
    os.chdir(repo)
    try:
        grid = build_grid(resolve_paths()["polygon"], dx=1000.0)
        ch = np.asarray(county_flags_for_grid(grid, DEFAULT_MAP)[0], bool)
    finally:
        os.chdir(cwd)
    zone = np.asarray(fan_zones(grid.centroids()), int)
    return {"grid": grid, "changhua": ch, "zone": zone}


def check_grid(z: dict, grid) -> list[str]:
    e = []
    if not np.array_equal(np.asarray(z["mask"], bool), np.asarray(grid.mask, bool)):
        e.append("grid mask differs from the forward's")
    for k in ("nx", "ny", "dx", "x0", "y0"):
        if float(z[k]) != float(getattr(grid, k)):
            e.append(f"grid {k} differs")
    return e


# --------------------------------------------------------------------------- the column
def rebuild_column(repo: Path, col: str, grid, meta: dict):
    """The column exactly as forward.main builds it (forward.load_columns, CPU), and the
    per-cell elastic coefficient Ske (None when the column is blended or banded)."""
    import os

    from hydrophysics.twin import forward as F
    from hydrophysics.twin.zone_county import column_zone_map, column_zone_weights

    cwd = os.getcwd()
    os.chdir(repo)
    try:
        cols = F.load_columns(column_path(col), None, None, "cpu",
                              zone_of_cell=column_zone_map(grid, meta),
                              zone_weights=column_zone_weights(grid, meta),
                              hpc0_fast_days=365.0, band_xy=grid.centroids(),
                              county_sha256=meta.get("zone_county_sha256"))
    finally:
        os.chdir(cwd)
    _, module, params = cols[0]
    ske = None
    if hasattr(module, "cols") and getattr(module, "zone_w", None) is None:
        zone = module.zone.cpu().numpy()
        per = np.array([math.exp(float(c.log_ske.detach().reshape(-1)[0])) for c in module.cols])
        ske = per[zone]
    elif hasattr(module, "log_ske") and not hasattr(module, "cols"):
        ske = np.full(grid.n_active, math.exp(float(module.log_ske.detach().reshape(-1)[0])))
    return module, params, ske


def driver(heads: np.ndarray) -> np.ndarray:
    """Layer-mean head (n_scen, A, T) from heads_mean (n_scen, 4, A, T), as forward.compaction
    (float32 before the column)."""
    return np.asarray(heads, dtype="float64").mean(axis=1).astype("float32")


def compaction_cpu(module, drv: np.ndarray) -> np.ndarray:
    import torch

    with torch.no_grad():
        return module(torch.as_tensor(drv, dtype=torch.float32)).cpu().numpy().astype(
            "float64")


# --------------------------------------------------------------------------- aggregation
def avoided_cm(subs: np.ndarray) -> np.ndarray:
    """(A, T) cut30 avoided subsidence in cm from Dec 2022 (+ = less sinking); ``subs``
    (2, A, T) metres, scenario 0 the baseline."""
    fb = subs[0] - subs[0][:, ORIGIN:ORIGIN + 1]
    fc = subs[1] - subs[1][:, ORIGIN:ORIGIN + 1]
    return (fb - fc) * 100.0


def elastic_avoided_cm(ske: np.ndarray, drv: np.ndarray) -> np.ndarray:
    """(A, T) elastic part of the avoided subsidence (cm): Ske (h_cut - h_base), re-zeroed at
    the origin (zero there: the runs share the record)."""
    d = np.asarray(drv[1], dtype="float64") - np.asarray(drv[0], dtype="float64")
    el = ske[:, None] * d * 100.0
    return el - el[:, ORIGIN:ORIGIN + 1]


GROUPS = ("fan", "yunlin", "changhua", "yunlin_inland", "yunlin_proximal", "yunlin_mid",
          "yunlin_distal", "changhua_proximal", "changhua_mid", "changhua_distal")


def group_masks(zone: np.ndarray, ch: np.ndarray) -> dict:
    yl = ~ch
    return {"fan": np.ones_like(ch), "yunlin": yl, "changhua": ch,
            "yunlin_inland": yl & (zone <= 1),
            "yunlin_proximal": yl & (zone == 0), "yunlin_mid": yl & (zone == 1),
            "yunlin_distal": yl & (zone == 2),
            "changhua_proximal": ch & (zone == 0), "changhua_mid": ch & (zone == 1),
            "changhua_distal": ch & (zone == 2)}


def summarise(av: np.ndarray, el: np.ndarray | None, dh: np.ndarray, dh2: np.ndarray,
              masks: dict, horizons: dict = HORIZONS) -> dict:
    """Group means at each horizon: avoided, elastic, inelastic (cm), avoided driver head and
    layer-2 head (m), cm per m of avoided driver head, distal shares."""
    out = {}
    for h, t in horizons.items():
        blk = {}
        for g, m in masks.items():
            if not m.any():
                continue
            a = float(av[m, t].mean())
            r = {"n_cells": int(m.sum()), "avoided_cm": a,
                 "avoided_head_m": float(dh[m, t].mean()),
                 "avoided_head_L2_m": float(dh2[m, t].mean())}
            r["cm_per_m"] = (a / r["avoided_head_m"] if abs(r["avoided_head_m"]) >= 1e-3
                             else None)
            if el is not None:
                e = float(el[m, t].mean())
                r.update(elastic_cm=e, inelastic_cm=a - e,
                         elastic_share=(e / a if abs(a) > 1e-9 else None))
            blk[g] = r
        tot = float(av[:, t].sum())
        yl = masks["yunlin"]
        blk["distal_share_fan"] = (float(av[masks["yunlin_distal"] | masks["changhua_distal"],
                                            t].sum()) / tot if abs(tot) > 1e-12 else None)
        ytot = float(av[yl, t].sum())
        blk["distal_share_yunlin"] = (float(av[masks["yunlin_distal"], t].sum()) / ytot
                                      if abs(ytot) > 1e-12 else None)
        out[h] = blk
    return out


def f_subs(av: np.ndarray, mask: np.ndarray) -> float | None:
    """Share of the Yunlin cut30 avoided subsidence at 2028-12 already there 6 months into the
    cut (2026-06): the part a seasonal-band test exercises (declaration, never a gate)."""
    late = float(av[mask, month_index(F_SUBS_LATE)].mean())
    early = float(av[mask, month_index(F_SUBS_EARLY)].mean())
    return early / late if abs(late) > 1e-9 else None


def storage_ratio(theta: dict, col_json: dict) -> dict:
    """Distal and mid (Yunlin) column Ske over the flow model's own storage there (sum over
    layers of exp(log_S_<zone>)); page about 11x."""
    th = theta.get("theta", theta)
    names = col_json.get("zone_names") or ["proximal", "mid", "distal"]
    out = {}
    for z in ("mid", "distal"):
        if f"log_S_{z}" not in th or z not in names:
            out[z] = None
            continue
        s = float(sum(math.exp(v) for v in np.atleast_1d(th[f"log_S_{z}"])))
        ske = math.exp(float(col_json["zonal"][names.index(z)]["log_ske"]))
        out[z] = {"ske": ske, "flow_storage_sum_layers": s, "ratio": ske / s}
    return out


# --------------------------------------------------------------------------- townships
#: the dated page build identical (sha256 26f15e1b...) to twin_app.html on 2026-11-01; a dated
#: file, so a later page rebuild cannot change the township map under the declaration
PAGE_HTML = "results/twin/twin_app_20261101a.html"


def page_townships(repo: Path, page_html: str = PAGE_HTML) -> dict:
    """Township index per cell, names and the page's headline-eligible mask (not the pooled
    Changhua panel, not low-confidence), from the page payload (model output)."""
    import re

    from hydrophysics.twin.app.prep import unpack

    s = (repo / page_html).read_text(encoding="utf-8")
    m = re.search(r'<script[^>]*id="payload"[^>]*>(.*?)</script>', s, re.S)
    if m is None:
        raise ValueError(f"no payload in {page_html}")
    D = json.loads(m.group(1))
    town = np.asarray(unpack(D["arrays"]["town"])).astype(int).ravel()
    names = [t["en"] for t in D["towns"]]
    skill = D.get("townSkill") or [{"low": True}] * len(names)
    elig = np.array([(not t["pooled"]) and (not s_["low"])
                     for t, s_ in zip(D["towns"], skill, strict=True)])
    return {"town": town, "names": names, "eligible": elig, "page_html": page_html}


def top_towns(av_t: np.ndarray, town: np.ndarray, elig: np.ndarray, n: int = 5) -> list[int]:
    """Eligible townships with the largest positive mean avoided subsidence, largest first."""
    nt = len(elig)
    cnt = np.bincount(town, minlength=nt).astype(float)
    sm = np.bincount(town, weights=av_t, minlength=nt)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = sm / cnt
    v = np.where(elig & np.isfinite(mean) & (mean > 0), mean, -np.inf)
    order = [int(i) for i in np.argsort(-v, kind="stable") if np.isfinite(v[i])]
    return order[:n]


# --------------------------------------------------------------------------- readings
def reading_inland(values: dict, factor: float = 1.5) -> dict:
    """Declared reading on {column: inland Yunlin avoided cm}: same sign everywhere and
    max/min <= factor -> robust; else column-dependent with the range."""
    v = np.array(list(values.values()), dtype=float)
    if v.size == 0:
        return {"reading": "not evaluable", "n": 0}
    same = bool(np.all(v > 0) or np.all(v < 0))
    lo, hi = float(np.min(np.abs(v))), float(np.max(np.abs(v)))
    ratio = hi / lo if lo > 0 else math.inf
    robust = same and ratio <= factor
    return {"n": int(v.size), "min_cm": float(v.min()), "max_cm": float(v.max()),
            "ratio": ratio, "same_sign": same,
            "reading": ("inland Yunlin robust within x1.5" if robust
                        else f"column-dependent x{ratio:.2f} ({v.min():.3f}-{v.max():.3f} cm)")}


def reading_rank(tops: dict, ref: str, need: int = 4) -> dict:
    """Top-5 township overlap of every column with the reference column (>= need of 5 in
    every column -> ranking column-robust) and whether the leader is the same everywhere."""
    r = tops[ref]
    ov = {c: len(set(t) & set(r)) for c, t in tops.items()}
    lead = {c: (t[0] if t else None) for c, t in tops.items()}
    need = min(need, len(r))              # fewer than 5 eligible towns: all of them
    robust = all(v >= need for v in ov.values())
    same_leader = len({x for x in lead.values()}) == 1
    return {"overlap_with_reference": ov, "min_overlap": min(ov.values()) if ov else None,
            "need": need,
            "reading": "ranking column-robust" if robust else "ranking column-dependent",
            "leader_same_in_every_column": same_leader,
            "C11_leader_wording": "kept" if same_leader else "lost"}
