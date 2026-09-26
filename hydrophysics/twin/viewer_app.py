"""Build the twin's decision page: one self-contained HTML file, no server.

    python -m hydrophysics.twin.viewer_app --forward results/twin_forward/<run>.npz \\
        --basis results/twin_forward/response_basis.npz --out results/twin/twin_app.html

What the page is for (spec ``docs/superpowers/specs/2026-09-23-twin-decision-app-redesign.md``).
A water authority asks *what does this policy change, where, by when, how sure are we,
and does it protect the high-speed rail and my township*. So the page opens on the answer:
a generated headline and six tiles comparing the current policy with the named baseline
("business as usual"), over a large 2D map of the policy's change from baseline on a
fixed, symmetric scale, linked to a time series with the fitted, tested and untested
periods marked. The 3D block model is a context view in a drawer, built only when opened.

Where the interactivity comes from. A solver run per policy is minutes, so the page ships
a **response basis**: one run per water-use class with that class retired, from 2026 and
from 2030. The page forms ``Δ(policy) = Σ_k (1 - f_k) · (retired_k - baseline)``. The basis
is one parameter set, so each class's rows are rescaled to the ensemble's solved run for
that class (``prep.calibrate_basis``; classes without one take the mean factor, marked
assumed), and moving a slider onto a solved scenario does not make the numbers jump. The
builder reports the superposition error before rescaling (against the basis file's
``check_*`` runs and the solved scenarios). When the policy *is* a solved scenario of the
forward run, the page uses that run's ensemble-mean fields and its paired members. A basis
that misses a solved run by more than ``BASIS_MAX_REL`` before rescaling was built for
another model (the 2026-09-24 basis against the per-well-datum model misses retiring
aquaculture by ~42 %): it is not shipped, the fast estimate is off, and only the solved
policies have numbers; the page says why.

Rheology axis. A forward run with two columns (``rheology_labels``) carries every member
under each; the page's fields, members and agreement are the first column's (the
deliverable), and the second column is the creep note on the baseline only.

Tests. The model card, the trust tab and the caveats quote the flow model's k-fold on
levels and on head changes (``stage3_flow.csv`` beside ``--theta``), the fair held-out-years
verdict (the ``--temporal`` screen's scorecard: error after each well's constant offset
against the best simple forecast), the column's leveling and ring skill, the policy check,
and the same numbers for the previous deliverable (``--prev-dir``, ``--prev-temporal``,
``--alt-forward``) side by side.

Model artefacts. A forward run made with the 2026-09-23 fixes (``twin.forward
--column-hpc0-fast-days`` and ``--column-heads free``; ``_artefacts_fixed_upstream``) has no
start-up or restart step, so nothing is removed and forward change is measured from the
last fitted year-end, the ledger's base (``meta.yRef``); the builder still measures the
steps and the page reports what it finds. For an older run the proximal column jumps when
it settles onto the first heads and again after the restart from observed heads at the
origin; both steps are removed from every field (``prep.artefact_steps``), what the restart
leaves elsewhere is bridged at year-end resolution (``prep.restart_bridge``), and forward
change is measured from the first year-end after the restart. Policy differences are
unaffected either way: the steps are identical in every scenario.

Per-member output (``<forward stem>.members.npz``, ``--save-members yearly``) gives the
page its real agreement: per-cell sign agreement over the parameter sets (the map hatches
below ``prep.AGREE_MIN``), per-township agreement, per-year run bands, run ranges for the
area and rail tiles, and every run's fan-average series for "play runs one by one".
``--rheo-forward`` (a run with a second column) puts the creep-ceiling note on the
baseline number, never on a policy difference.

Observations. By default the page is **public**: the leveling and the wells enter only as
aggregates (chain and township skill, the fan-average of the layer-2 wells). ``--private``
embeds every benchmark and well with its location and series, for a local page that must
never be committed.

Every number on the page is computed here or in ``app/prep.py`` from the npz files (and
the optional inputs below); the page itself only superposes, averages and samples. Arrays
are quantised, byte-shuffled and gzipped (``prep.pack``), so the page is about 2 MB.

Optional inputs, each of which the page survives without ("not loaded" panels):
``--hsr`` (THSR centreline traced from OSM, ``app/geo.py``), ``--members-csv`` (paired
member deltas; default ``<forward stem>.members.csv``), ``--leveling`` (skill against the
benchmarks, default ``$HYDROMIND_GW_DATA``), ``--wells`` (``auto`` loads the calibration's
wells and the class energies through ``twin.inputs``, about a minute on CPU),
``--temporal`` (the held-out-years test: the tested horizon and caveat 1),
``--column-csv`` (column skill), ``--theta`` (the learned stress radius).
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import io
import json
import os
import re
import sys
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .app import geo, prep
from .zones import ZONE_NAMES, fan_zones

TEMPLATE = os.path.join(os.path.dirname(__file__), "app", "template.html")
THREE_CDN = "https://cdnjs.cloudflare.com/ajax/libs/three.js/r134/three.min.js"
THREE_LOCAL = os.path.join(os.path.dirname(__file__), "app", "vendor", "three.min.js")
MAX_BYTES = 8_000_000                 # the builder refuses to write a page larger than this
TARGET_BYTES = 5_000_000
# the deliverable (STATE.md §0, 2026-09-27): the per-well-datum flow model, its per-zone
# column fitted on the leveling, and the 30-year creep-ceiling column beside it
DELIVERABLE = "results/twin_runs/stage3_datum_gate"
# the held-out-years screen of the deliverable recipe (fit 2012-2019, free-run 2020-2022,
# scored after each well's constant offset); the other ``temporal_*`` screens beside it are
# the variants that were tried
DEFAULT_TEMPORAL = ("results/twin_runs/temporal_ref10_ic_ge_split208_tmin58_datum_sd5/"
                    "stage3_temporal_pred.npz")
TEMPORAL_GLOB = "results/twin_runs/temporal_*/scorecard.json"
# the fair verdict of the screens written before ``calibrate_flow`` stored it
# (``twin.rescore_temporal``)
DEFAULT_RESCORE = "results/twin/temporal_fair_rescore.csv"
# the rheology axis: ``auto`` is the forward run itself when it carries two columns
# (``rheology_labels``); the second column's skill table
DEFAULT_RHEO = "auto"
DEFAULT_RHEO_COLUMN = f"{DELIVERABLE}/coupled_leveling_tau30/stage4_column.csv"
DEFAULT_COLUMN = f"{DELIVERABLE}/coupled_leveling/stage4_column.csv"
DEFAULT_THETA = f"{DELIVERABLE}/stage3_theta.json"
# the previous deliverable (STATE.md §0b), projected with the same apex handling: quoted
# next to every answer and in the test table, because the ensemble varies parameters only
DEFAULT_ALT = "results/twin_forward/physical_spread_apex.npz"
DEFAULT_ALT_THETA = "results/twin_runs/stage3_spreadL_gate/stage3_theta.json"
DEFAULT_PREV_DIR = "results/twin_runs/stage3_spreadL_gate"
DEFAULT_PREV_TEMPORAL = "results/twin_runs/temporal_ref10"
# the MLCW compaction rings the column is checked against (their count)
RINGS_CSV = "results/twin/stage2_vep_mlcw.csv"
# columns fitted with a longer creep ceiling (``--tau-max-years``), for the creep caveat
TAU_ALT_GLOB = "results/twin_runs/*/coupled_leveling_tau*/vep_*.json"
HSR_REACH_KM = 1.5          # half the 2 km smoothing plus half the 1 km distortion window
# the fast estimate superposes a per-class response basis. When the basis misses a solved
# run of the forward model by more than this share of its fan-mean response (before any
# rescaling), it was built for another model and is not shipped: only the full-model
# policies are shown
BASIS_MAX_REL = 0.15

# Per-zone aquifer geometry, metres below ground, from the median screen depth of the
# zone-50 wells in each layer (AMP_V2 station metadata; see heads.py for the QC). The
# proximal layer-4 depth is extrapolated (no zone-50 well is screened there); it only
# places a slab in the drawing, nothing numerical uses it.
LAYER_DEPTHS = {
    "proximal": [67.0, 125.0, 205.0, 250.0],
    "mid": [35.5, 119.8, 214.0, 289.0],
    "distal": [66.4, 104.9, 203.9, 276.0],
}
AQUIFER_THICK = {"proximal": [40.0, 45.0, 45.0, 40.0],
                 "mid": [30.0, 45.0, 45.0, 40.0],
                 "distal": [35.0, 40.0, 45.0, 40.0]}
CLASS_LABELS = {"irrigation": ("Irrigation", "農業灌溉"), "aquaculture": ("Aquaculture", "養殖"),
                "livestock": ("Livestock", "畜牧"), "domestic": ("Domestic", "民生"),
                "industry": ("Industry", "工業"), "other": ("Other", "其他")}


def _encode_image(raw: np.ndarray, long_side: int = 1200, quality: int = 60) -> str:
    """Re-encode a basemap JPEG at ``long_side`` px and ``quality`` -> base64."""
    from PIL import Image

    img = Image.open(io.BytesIO(raw.tobytes())).convert("RGB")
    s = long_side / max(img.size)
    if s < 1:
        img = img.resize((round(img.size[0] * s), round(img.size[1] * s)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True, progressive=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _parse_policy(desc: str) -> tuple[list[float], int] | None:
    """``"cut30: irrigation x0.7 from 2026-01"`` -> (factors per class, start year)."""
    f = [1.0] * len(prep.CLASSES)
    hits = re.findall(r"(\w+) x([\d.]+) from (\d{4})", desc)
    if not hits:
        return None
    start = int(hits[0][2])
    for cls, val, yr in hits:
        if cls not in prep.CLASSES or int(yr) != start:
            return None
        f[prep.CLASSES.index(cls)] = float(val)
    return f, start


def _load_inputs():
    """The calibration's wells and class energies via ``twin.inputs`` (CPU, ~1 min)."""
    from .inputs import load_twin_inputs

    return load_twin_inputs(verbose=False)


def _ground_idw(cent: np.ndarray, log=print) -> np.ndarray | None:
    """Ground elevation at the cell centres, interpolated from the wells' ground heights.

    Only a fallback: the basemap's terrain (``--basemap``) is used when present.
    """
    try:
        from ..subsidence import idw_interp
        from .calibrate_flow import DEFAULT_PATHS

        stn = pd.read_parquet(DEFAULT_PATHS["stations"])
    except (ImportError, OSError, KeyError, ValueError) as e:
        log(f"ground elevation unavailable ({type(e).__name__}: {e}); section and 3D use 0 m")
        return None
    need = {"GroundwaterZoneIdentifier", "LocationByTWD97", "GroundHeight"}
    if not need <= set(stn.columns):
        log(f"ground elevation unavailable (stations lack {sorted(need - set(stn.columns))})")
        return None
    stn = stn[stn.GroundwaterZoneIdentifier == 50]
    xy = stn["LocationByTWD97"].astype(str).str.split(expand=True)
    x = pd.to_numeric(xy[0], errors="coerce") if 0 in xy else pd.Series(np.nan, stn.index)
    y = pd.to_numeric(xy[1], errors="coerce") if 1 in xy else pd.Series(np.nan, stn.index)
    g = pd.to_numeric(stn["GroundHeight"], errors="coerce")
    # the same plausibility window as the head field's station parser
    ok = (x.between(140000, 240000) & y.between(2580000, 2700000) & np.isfinite(g)).to_numpy()
    if ok.sum() < 3:
        log("ground elevation unavailable (fewer than 3 stations); section and 3D use 0 m")
        return None
    src = np.column_stack([x.to_numpy()[ok], y.to_numpy()[ok]]).astype("float64")
    return idw_interp(np.asarray(cent, dtype="float64"), src,
                      g.to_numpy()[ok].astype("float64").reshape(-1, 1))[:, 0]


def _leveling_dir(leveling: str | None) -> str | None:
    if leveling in (None, "none"):
        return None
    if leveling == "auto":
        leveling = os.environ.get("HYDROMIND_GW_DATA", "chou-shui-data/data")
    return leveling if os.path.exists(os.path.join(leveling, "ls_cache")) else None


def _geometry(fw) -> SimpleNamespace:
    """Grid, time axis and zones of a forward run."""
    g = SimpleNamespace()
    g.mask = fw["mask"].astype(bool)
    g.nx, g.ny, g.dx = int(fw["nx"]), int(fw["ny"]), float(fw["dx"])
    g.x0, g.y0 = float(fw["x0"]), float(fw["y0"])
    g.dates = [str(d)[:10] for d in fw["dates"]]
    g.origin = int(fw["origin"])
    g.ye, g.years = prep.year_ends(g.dates)
    g.y_obs = int(np.searchsorted(g.ye, g.origin, side="right") - 1)   # last fitted year-end
    # forward changes are measured from the first year-end after the restart has settled;
    # a run with the restart removed upstream has nothing to settle, so they are measured
    # from the last fitted year-end, the ledger's base (STATE.md)
    g.fixed = _artefacts_fixed_upstream(fw)
    g.y_ref = g.y_obs if g.fixed else min(
        int(np.searchsorted(g.ye, g.origin + prep.RESTART_WINDOW, side="left")), len(g.ye) - 2)
    g.A = int(g.mask.sum())
    g.rows, g.cols = np.nonzero(g.mask)
    g.cent = np.column_stack([g.x0 + (g.cols + 0.5) * g.dx, g.y0 + (g.rows + 0.5) * g.dx])
    g.gate_all = {}
    with contextlib.suppress(Exception):
        g.gate_all = json.loads(str(fw["gate"]))
    g.zb = [float(v) for v in str(g.gate_all.get("zone_boundaries", "205,182")).split(",")]
    g.zone = fan_zones(g.cent, proximal_km=g.zb[0], distal_km=g.zb[1])
    return g


def _artefacts_fixed_upstream(fw) -> str:
    """Non-empty (a description) when the forward run removed both model steps at the
    source: the proximal start-up load (A1) and the restart (``column_heads == "free"``).

    A1 counts as fixed when the run released it (``column_hpc0_fast_days`` set, or a
    non-empty ``hpc0_released``), or when every column was refitted with the offset
    guard (``hpc0_guard_days``), which leaves nothing to release (review D1)."""
    if "forward_options" not in fw.files:
        return ""
    try:
        opt = json.loads(str(fw["forward_options"]))
    except ValueError:
        return ""
    guards = opt.get("hpc0_guard_days") or []
    a1 = (bool(opt.get("column_hpc0_fast_days")) or any(opt.get("hpc0_released") or [])
          or (bool(guards) and all(g is not None and float(g) > 0 for g in guards)))
    a2 = opt.get("column_heads") == "free"
    return "start-up load released, column on free-running heads" if (a1 and a2) else ""


def _forward_options(fw) -> dict:
    with contextlib.suppress(KeyError, ValueError):
        return json.loads(str(fw["forward_options"]))
    return {}


def rheology_labels(fw) -> list[str]:
    """The forward run's compaction columns (``twin.forward --vep-json a,b``), first = own."""
    return [str(v) for v in fw["rheology_labels"]] if "rheology_labels" in fw.files else []


def _boundaries(fwd: np.ndarray, rate: np.ndarray, x_km: np.ndarray, zb: list[float]) -> list:
    """Per zone line, the median forward change and final-year rate in the first column of
    cells east and west of it (cm, cm/yr)."""
    out = []
    for km in sorted(zb):
        b = prep.boundary_contrast(fwd, x_km, km)
        if not (b["n_east"] and b["n_west"]):
            continue
        r = prep.boundary_contrast(rate, x_km, km)
        out.append({**b, "rate_east": r["east"], "rate_west": r["west"]})
    return out


def _forward_fields(fw, g, log) -> SimpleNamespace:
    """Year-end fields of the forward run, with the proximal column's model steps removed."""
    f = SimpleNamespace()
    # a run with a rheology axis carries every member under each column, and its
    # ``subs_mean`` pools them; the page's numbers are the first column's (the deliverable),
    # the second is the creep note on the baseline (``_rheology_block``)
    f.rheo_labels = rheology_labels(fw)
    f.rheo_ref = f.rheo_labels[0] if len(f.rheo_labels) > 1 else None
    if f.rheo_ref is not None and "subs_mean_by_rheology" in fw.files:
        raw = fw["subs_mean_by_rheology"][0].astype("float64") * 100.0
        log(f"rheology axis {f.rheo_labels}: the page shows the first column ({f.rheo_ref})")
    else:
        raw = fw["subs_mean"].astype("float64") * 100.0            # (S, A, T) cm
    prox = g.zone == ZONE_NAMES.index("proximal")
    # a forward run made with the artefact fixes of 2026-09-23 (twin.forward
    # --column-hpc0-fast-days and --column-heads free) has no steps to remove
    fixed = g.fixed
    if fixed:
        log("forward run has the start-up and restart fixes applied upstream "
            f"({fixed}); no artefact steps removed")
    # measured in the proximal cells either way; removed only when not fixed upstream
    measured = prep.artefact_steps(raw[0], g.origin, prox)
    steps = prep.artefact_steps(raw[0], g.origin, prox & (not fixed))
    f.subs = np.stack([prep.remove_artefacts(s, steps) for s in raw])
    # what is left of the restart elsewhere (a rebound of a few cm), bridged at year-end
    # resolution; computed on the baseline and applied to every scenario alike
    bridge = (np.zeros(g.A) if fixed else prep.restart_bridge(f.subs[0], g.ye, g.y_obs))
    f.subs = np.stack([prep.apply_bridge(s, bridge, g.origin) for s in f.subs])
    sstd = fw["subs_std"].astype("float64") * 100.0
    f.heads = fw["heads_mean"]                                      # (S, L, A, T) m
    f.L = f.heads.shape[1]
    f.base_ye = f.subs[0][:, g.ye]
    shift = raw[0][:, g.ye] - f.base_ye
    f.band = prep.band_from_std(raw[0][:, g.ye], sstd[0][:, g.ye]) - shift[None]
    f.head_ye = f.heads[0][:, :, g.ye].astype("float64")
    fan = prep.model_artefacts(raw[0].mean(0), g.origin)
    free = prep.artefact_steps(raw[0], g.origin, ~prox)           # the start-up elsewhere
    other = bridge[~prox] if (~prox).any() else np.zeros(1)
    fwd = f.subs[0][:, g.ye[-1]] - f.subs[0][:, g.ye[g.y_ref]]
    x_km = g.cent[:, 0] / 1000.0
    opt = _forward_options(fw)

    def _cell(key, fn):
        return float(fn(measured[key][prox])) if prox.any() else 0.0

    f.artefacts = {
        **fan, "zone": "proximal", "zone_km": g.zb[0], "n_cells": int(prox.sum()),
        # fixed upstream: nothing is removed, and the steps below are what a check of
        # this run still finds (the same measure the removal would use)
        "fixed_upstream": fixed, "removed": not fixed,
        "hpc0_fast_days": opt.get("column_hpc0_fast_days"),
        "column_heads": opt.get("column_heads"),
        "restart_taper_km": opt.get("restart_taper_km"),
        "startup_cell_median": _cell("startup", np.median),
        "startup_cell_max": _cell("startup", np.max),
        "restart_cell_median": _cell("restart", np.median),
        "restart_cell_max": _cell("restart", np.max),
        "other_startup_p95_cm": float(np.percentile(np.abs(free["startup"][~prox]), 95))
        if (~prox).any() else 0.0,
        "bridge_median_cm": float(np.median(other)), "bridge_p05_cm": float(np.percentile(other, 5)),
        "bridge_max_abs_cm": float(np.abs(other).max()),
        # the zone lines: the column's parameters change there, so the field steps; measured
        # on the forward change and on the final year's rate
        "boundaries": _boundaries(fwd, f.subs[0][:, g.ye[-1]] - f.subs[0][:, g.ye[-2]],
                                  x_km, g.zb),
        "windows": steps["windows"], "ref_year": g.years[g.y_ref]}
    names = [str(s) for s in fw["scenario_names"]] if "scenario_names" in fw.files else \
        [str(s).split(":")[0] for s in fw["scenarios"]]
    desc = [str(s) for s in fw["scenarios"]]
    f.solved = []
    for s in range(1, f.subs.shape[0]):
        pol = _parse_policy(desc[s])
        if pol is None:
            continue
        fan_d = (f.subs[s] - f.subs[0]).mean(0)
        f.solved.append({"name": names[s], "desc": desc[s], "factors": pol[0],
                         "start": pol[1], "_dsub": f.subs[s][:, g.ye] - f.base_ye,
                         "_dh2": f.heads[s, 1][:, g.ye] - f.heads[0, 1][:, g.ye],
                         "fast": prep.fast_share(fan_d, g.dates, pol[1]),
                         # when the benefit arrives and whether the late sinking slows
                         "timing": prep.effect_timing(f.base_ye.mean(0),
                                                      f.subs[s][:, g.ye].mean(0),
                                                      g.years, pol[1])})
    # the ledger's (STATE.md) forward subsidence is Dec of the last fitted year to the
    # horizon on the raw field, start-up and restart steps included; the page counts from
    # ``y_ref`` with them removed, and says which base it uses
    f.raw_fwd_fan = float((raw[0][:, g.ye[-1]] - raw[0][:, g.ye[g.y_obs]]).mean())
    # per scenario, what the page's corrections took off the fan average at each year-end
    # (zero when fixed upstream); the per-member series are shifted by the same amount
    f.names = names
    f.fan_shift = raw[:, :, g.ye].mean(axis=1) - f.subs[:, :, g.ye].mean(axis=1)
    f.cell_shift0 = raw[0][:, g.ye] - f.base_ye                     # (A, Y), the baseline's
    return f


def _q(x, q=(10, 50, 90)) -> list[float]:
    """Percentiles to four significant figures (angular distortions are ~1e-5)."""
    return [float(f"{v:.4g}") for v in np.percentile(np.asarray(x, dtype="float64"), q)]


def _member_fields_block(forward_npz: str, members_npz: str | None, g, f, solved: list,
                         town_idx: np.ndarray, n_towns: int, hsr_out: dict | None,
                         log, rheology: str | None = None) -> tuple[dict, dict | None]:
    """Per-member results from ``<forward stem>.members.npz`` (``twin.forward
    --save-members yearly``), when it exists.

    For each solved scenario this adds:
    - ``solvedAgree<i>``: an (A, Y) uint8 percentage of parameter sets that agree with
      the sign of the ensemble-mean change (policy minus baseline, since ``yRef``). The
      map hatches where it is below ``prep.AGREE_MIN``.
    - ``bandYr`` / ``bandYrHead``: the real per-year p10/p25/p50/p75/p90 over the runs of
      the fan-average change (subsidence since ``yRef``, cm; layer-2 head, m).
    - ``townAgree``: per township, the sets whose township-average change is a benefit at
      the horizon.
    - ``area`` / ``hsr``: p10/p50/p90 over the sets of the rate-area count (per threshold)
      and of the rail's largest angular distortion, the policy's and the paired change.

    ``memberFields`` carries the baseline's per-year bands, its area and rail ranges, and
    every run's fan-average series per scenario (``fan``), which the page plays one run at
    a time. A parameter set's initial fields are one opinion, so agreement counts sets.
    Without the file nothing is added, and the page keeps its fallback (the |change| rule
    and the constructed bands).

    ``rheology`` (a run with a rheology axis) keeps the members of that column only, and
    the baseline's per-cell band (``f.band``) is then taken from them: the run's
    ``subs_std`` pools both columns."""
    if members_npz == "auto":
        members_npz = os.path.splitext(forward_npz)[0] + ".members.npz"
    if not (members_npz and os.path.exists(members_npz)):
        log("no per-member yearly fields (--save-members yearly): map agreement falls back "
            "to the |change| rule")
        return {}, None
    try:
        mf = prep.load_member_fields(members_npz, rheology)
    except ValueError as e:
        log(f"{e}; ignored")
        return {}, None
    if mf["years"] != list(g.years) or mf["subs"].shape[2] != g.A:
        log(f"{members_npz}: years/cells do not match the forward run; ignored")
        return {}, None
    if rheology is not None:
        f.band = (np.percentile(mf["subs"][0].astype("float64"), (10, 25, 75, 90), axis=0)
                  - f.cell_shift0[None])
        log(f"members: {mf['subs'].shape[1]} runs of the {rheology} column; the baseline "
            "band is their per-cell p10/p25/p75/p90")
    names = mf["scenario_names"]
    if names[0] != "baseline":
        log(f"{members_npz}: the first scenario is not the baseline; ignored")
        return {}, None
    sets, n_sets = mf["sets"], len(dict.fromkeys(mf["sets"]))
    qs = (10, 25, 50, 75, 90)
    # every run's fan average, on the page's field (the corrections the page applied to
    # the ensemble mean, if any, are applied to every run alike)
    fan_s = prep.member_fan(mf["subs"])                                  # (S, M, Y) cm
    fan_h = prep.member_fan(mf["headL2"])                                # (S, Mh, Y) m
    for s, n in enumerate(names):
        if n in f.names:
            fan_s[s] -= f.fan_shift[f.names.index(n)][None]
    fan = {"subs": {n: np.round(fan_s[s], 3).tolist() for s, n in enumerate(names)},
           "head": {n: np.round(fan_h[s], 3).tolist() for s, n in enumerate(names)}}
    labels = [f"{m} · ic {i}" for m, i in zip(mf["member"], mf["ic"], strict=True)]
    thr = (1, 2, 3)
    base_area = {t: prep.area_by_set(mf["subs"][0], sets, t) for t in thr}
    hsr_args = None
    if hsr_out is not None:
        hsr_args = (np.array(hsr_out["idx"]), np.array(hsr_out["w"]), hsr_out["step"],
                    np.array(hsr_out["mixed"], dtype=bool))

    def fwd_of(s):
        x = mf["subs"][s]
        return x[..., -1] - x[..., g.y_ref]

    base_hsr = prep.hsr_max_by_set(fwd_of(0), sets, *hsr_args) if hsr_args else None
    arrays = {}
    for i, x in enumerate(solved):
        if x["name"] not in names:
            continue
        s = names.index(x["name"])
        d = prep.member_delta(mf, s, g.y_ref)
        agree = prep.cell_agreement(d, sets)
        arrays[f"solvedAgree{i}"] = prep.pack(np.rint(agree * 100.0).astype("uint8"), None,
                                              "uint8")
        x["bandYr"] = [[round(float(v), 4) for v in row] for row in prep.fan_member_band(d)]
        dh = mf["headL2"][s].astype("float64") - mf["headL2"][0]
        x["bandYrHead"] = [[round(float(v), 4) for v in row]
                           for row in np.percentile(dh.mean(axis=1), qs, axis=0)]
        x["townAgree"] = prep.township_agreement(d, sets, town_idx, n_towns)
        area = {}
        for t in thr:
            pa = prep.area_by_set(mf["subs"][s], sets, t)
            area[str(t)] = {"pol": _q(pa), "d": _q(pa - base_area[t])}
        x["area"] = area
        if hsr_args:
            ph = prep.hsr_max_by_set(fwd_of(s), sets, *hsr_args)
            x["hsr"] = {"pol": _q(ph), "d": _q(ph - base_hsr)}
        # the same share over all runs, for the log: the initial fields change nothing
        dm = d[..., -1]
        sign = np.sign(dm.mean(axis=0))
        runs = ((np.sign(dm) == sign[None]) & (sign[None] != 0)).mean(axis=0)
        ok = np.abs(dm.mean(axis=0)) >= 0.1
        log(f"members {x['name']}: {n_sets} sets; cells where >= {prep.AGREE_MIN:.0%} of sets "
            f"agree at the horizon: {int((agree[:, -1] >= prep.AGREE_MIN).sum())}/{g.A} "
            f"({int((agree[ok, -1] >= prep.AGREE_MIN).sum())}/{int(ok.sum())} where the mean "
            f"change is >= 1 mm; over all {dm.shape[0]} runs "
            f"{int((runs[ok] >= prep.AGREE_MIN).sum())}); fan-average change p10/p50/p90 "
            f"{x['bandYr'][0][-1]:+.2f} / {x['bandYr'][2][-1]:+.2f} / "
            f"{x['bandYr'][4][-1]:+.2f} cm")
    info = {"n": int(mf["subs"].shape[1]), "n_sets": n_sets, "agreeMin": prep.AGREE_MIN,
            "rheology": rheology,
            "source": os.path.basename(members_npz), "labels": labels,
            "baseBandYr": np.round(np.percentile(fan_s[0], qs, axis=0), 3).tolist(),
            "baseHeadBandYr": np.round(np.percentile(fan_h[0], qs, axis=0), 3).tolist(),
            "baseFwd": _q(fan_s[0][:, -1] - fan_s[0][:, g.y_ref]),
            "area": {str(t): _q(base_area[t]) for t in thr},
            "hsr": None if base_hsr is None else _q(base_hsr),
            "fan": fan}
    return arrays, info


def _response_basis(basis_npz, g, solved, log
                    ) -> tuple[dict | None, dict, dict | None, dict | None]:
    """The per-class response basis, rescaled to the ensemble's solved runs.

    Returns ``(basis, error, calibration, off)``. ``off`` is set (and the basis dropped)
    when the basis misses a solved run of this forward model by more than
    ``BASIS_MAX_REL`` of its fan-mean response before rescaling: a basis built for another
    model cannot be rescaled into this one's answer by one factor per class, so the page
    shows the full-model policies only and says why."""
    if not (basis_npz and os.path.exists(basis_npz)):
        log("no response basis: the levers are disabled; solved scenarios still show")
        return None, {}, None, None
    bs = np.load(basis_npz, allow_pickle=False)
    bnames = [str(n) for n in bs["scenario_names"]]
    bsubs = bs["subs_mean"].astype("float64") * 100.0
    bd = prep.basis_deltas(bsubs, bs["heads_mean"].astype("float64"), bnames, g.ye)
    berr = prep.basis_error(bd["dsub"], bsubs, bnames, g.ye, g.y_ref, solved)
    ens = {k: v for k, v in berr.items() if v["kind"] == "ensemble"}
    if ens:
        worst = max(ens, key=lambda k: ens[k]["rel"])
        if ens[worst]["rel"] > BASIS_MAX_REL:
            gate = {}
            with contextlib.suppress(KeyError, ValueError):
                gate = json.loads(str(bs["gate"]))
            off = {"worst": worst, "rel": ens[worst]["rel"], "fan_cm": ens[worst]["fan_cm"],
                   "max_rel": BASIS_MAX_REL, "source": os.path.basename(basis_npz),
                   "basis_commit": gate.get("git_commit"),
                   "basis_r2_kfold": (gate.get("gate") or {}).get("r2_kfold")}
            log(f"fast estimate OFF: the basis ({off['source']}, commit "
                f"{off['basis_commit']}) misses the solved run {worst} by "
                f"{100 * off['rel']:.1f} % of its response (> {100 * BASIS_MAX_REL:.0f} %): "
                "built for another model; only the full-model policies are shown")
            return None, berr, None, off
    calib = prep.calibrate_basis(bd["dsub"], bd["dheadL2"], solved, g.y_ref)
    bd["dsub"] = prep.apply_calibration(bd["dsub"], calib["subs"])
    bd["dheadL2"] = prep.apply_calibration(bd["dheadL2"], calib["head"])
    bd["dhead_end"] = prep.apply_calibration(bd["dhead_end"], calib["head"])
    if bd["missing"]:
        log(f"basis rows missing (left at zero): {bd['missing']}")
    return bd, berr, calib, None


def _wells_block(inp, g, head_ye, public: bool) -> tuple[list | None, dict | None, dict | None]:
    """Class energies, the layer-2 wells' fan-average series and (private) the well points."""
    if inp is None:
        return None, None, None
    nyr = len(inp.dates) / 12.0
    # the projection repeats the record's month-of-year mean (scenario.climatology),
    # so the 2023-2032 annual energy per class is the record's annual mean
    class_gwh = [float(inp.E_by_class[c].sum() / nyr / 1e6) if c in inp.E_by_class
                 else 0.0 for c in prep.CLASSES]
    obs_years = g.years[:g.y_obs + 1]
    ann = np.full((len(inp.sids), len(obs_years)), np.nan)
    yrs = inp.dates.year.to_numpy()
    for j, y in enumerate(obs_years):
        sel = yrs == y
        if sel.any():
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                ann[:, j] = np.nanmean(inp.obs_h[:, sel], axis=1)
    # an aggregate only: the annual mean over the layer-2 wells with a record that year,
    # observed and modelled at the same wells
    l2 = np.nonzero(inp.obs_layer == 1)[0]
    fo, fm, fn = [], [], []
    for j in range(len(obs_years)):
        ok = l2[np.isfinite(ann[l2, j])]
        fn.append(int(len(ok)))
        fo.append(round(float(ann[ok, j].mean()), 3) if len(ok) else None)
        fm.append(round(float(head_ye[1][inp.obs_idx[ok], j].mean()), 3) if len(ok) else None)
    fan = {"years": obs_years, "obs": fo, "model": fm, "n": fn, "n_wells": int(len(l2))}
    points = None
    if not public:
        points = {"x": np.round((inp.well_xy[:, 0] - g.x0) / 1000.0, 3).tolist(),
                  "y": np.round((inp.well_xy[:, 1] - g.y0) / 1000.0, 3).tolist(),
                  "cell": inp.obs_idx.astype(int).tolist(),
                  "layer": inp.obs_layer.astype(int).tolist(),
                  "obs": prep.pack(ann, 0.01)}
    return class_gwh, fan, points


def _leveling_block(ldir, g, base_monthly, town_idx, n_towns, public: bool, log,
                    extra: dict | None = None):
    """Leveling support: the chain skill, township skill and (private) the benchmarks.

    ``extra`` maps a name to another model's monthly baseline field on the same grid; its
    chain skill against the same benchmarks goes to ``chain["extra"][name]``."""
    if not ldir:
        return None, None, None
    from .leveling import load_panel, site_subsidence, site_xy

    try:
        # only reading the data is allowed to fail quietly; a bug in the comparison below
        # must surface, not turn into a page that silently lacks its leveling skill
        panel = load_panel(ldir)
    except (OSError, KeyError, ValueError) as e:
        log(f"leveling unavailable ({type(e).__name__}: {e})")
        return None, None, None
    T_obs = g.origin + 1
    obs = site_subsidence(panel, g.dates[0], str(pd.Timestamp(g.dates[g.origin])
                                                 + pd.offsets.MonthBegin(1))[:10],
                          min_obs=5, max_rate=0.5)
    obs_years = g.years[:g.y_obs + 1]
    lev = prep.leveling_support(obs, site_xy(panel), g.cent, g.dx, base_monthly[:, :T_obs],
                                g.dates[:T_obs], obs_years)
    if not len(lev["cell"]):
        log("leveling unavailable (no benchmark falls on the grid)")
        return None, None, None
    def _chain(lv):
        diff = np.concatenate([p - o for p, o in lv["pairs"]])
        return {"r2": prep.pooled_r2(lv["pairs"]), "n": int(len(lv["cell"])),
                "rmse": float(np.sqrt(np.mean(diff ** 2))), "bias": float(np.mean(diff))}

    chain = _chain(lev)
    chain["extra"] = {}
    for name, m in (extra or {}).items():
        lv = prep.leveling_support(obs, site_xy(panel), g.cent, g.dx, m[:, :T_obs],
                                   g.dates[:T_obs], obs_years)
        if len(lv["cell"]):
            chain["extra"][name] = _chain(lv)
    town_skill = prep.township_skill(lev, town_idx, n_towns)
    lev_out = None
    if not public:
        lim = prep.snap(float(np.percentile(np.abs(lev["resid"]), 95)), (2.0, 5.0, 10.0, 20.0))
        lev_out = {"x": np.round((lev["x"] - g.x0) / 1000.0, 3).tolist(),
                   "y": np.round((lev["y"] - g.y0) / 1000.0, 3).tolist(),
                   "cell": lev["cell"].astype(int).tolist(),
                   "resid": np.round(lev["resid"], 2).tolist(),
                   "bias": np.round(lev["bias"], 2).tolist(), "lim": lim,
                   "years": obs_years, "series": prep.pack(lev["series"], 0.01)}
    return chain, town_skill, (lev_out, lev)


def _hsr_block(hsr_csv, hsr_stations_csv, g) -> dict | None:
    hsr = geo.load_hsr(hsr_csv)
    if hsr is None:
        return None
    idx, w, ok = prep.bilinear_weights(hsr.x_twd97.to_numpy(), hsr.y_twd97.to_numpy(),
                                       g.x0, g.y0, g.dx, g.mask)
    if ok.sum() < 8:
        return None
    h = hsr[ok].reset_index(drop=True)
    ch = h.chainage_km.to_numpy() - h.chainage_km.iloc[0]
    stations = []
    st = pd.read_csv(hsr_stations_csv) if hsr_stations_csv and \
        os.path.exists(hsr_stations_csv) else None
    if st is not None:
        for _, r in st.iterrows():
            d = np.hypot(h.x_twd97 - r.x_twd97, h.y_twd97 - r.y_twd97)
            if d.min() < 1500:
                stations.append({"zh": r.name_zh, "en": r.name_en,
                                 "ch": round(float(ch[int(d.argmin())]), 2)})
    step = float(np.median(np.diff(ch))) if len(ch) > 1 else 0.25
    # where the rail's gradient reads cells on both sides of a zone line, the gradient is
    # the model's parameter boundary; the rail tile leaves those points out
    mixed = prep.zone_mixed(idx[ok], w[ok], g.zone, step, HSR_REACH_KM)
    return {"ch": np.round(ch, 3).tolist(),
            "x": np.round((h.x_twd97.to_numpy() - g.x0) / 1000.0, 3).tolist(),
            "y": np.round((h.y_twd97.to_numpy() - g.y0) / 1000.0, 3).tolist(),
            "idx": idx[ok].astype(int).tolist(),
            "w": np.round(w[ok], 4).tolist(), "stations": stations,
            "step": step, "mixed": mixed.astype(int).tolist(), "reach": HSR_REACH_KM,
            "thr": 1000, "attrib": geo.ATTRIB}


def _rivers_block(rivers_csv, g) -> list | None:
    rv = geo.load_rivers(rivers_csv)
    if rv is None or not len(rv):
        return None
    out = []
    for (name, zh, _part), gr in rv.groupby(["river", "name_zh", "part"], sort=False):
        out.append({"en": name, "zh": zh,
                    "pts": np.round(np.column_stack([(gr.x_twd97 - g.x0) / 1000.0,
                                                     (gr.y_twd97 - g.y0) / 1000.0]),
                                    2).tolist()})
    return out


def _temporal_block(temporal_npz, temporal_key="auto",
                    rescore_csv: str | None = DEFAULT_RESCORE) -> dict | None:
    """The held-out-years test: fit to month ``T_fit``, run free for the rest.

    ``temporal_key`` names the prediction array; ``auto`` takes ``pred`` (the layout of
    ``calibrate_flow``'s ``stage3_temporal_pred.npz``) or else ``temporal_spreadL`` (the
    older combined file). When a ``scorecard.json`` sits beside the file (or in its
    ``coupled_leveling/``) its shape and per-well scores are added, and every other
    ``temporal_*`` screen (``TEMPORAL_GLOB``) is listed under ``others``: the variants that
    were tried and how they did.

    The verdict is the **fair** one when the scorecard carries it (``verdict_fair``,
    pre-registered 2026-09-24): each well's constant offset is taken from the fitted years
    only, late-start wells are left out, and the model's error after that offset must be
    within ``fair_k`` of the best simple forecast (climatology, climatology plus trend,
    persistence) with a shape R² no worse than climatology's minus ``fair_shape_tol``. The
    raw RMSE (``rmse_model`` against ``rmse_clim``) is kept: most of it is that offset. A
    screen whose scorecard predates the fair verdict takes it from ``rescore_csv``
    (``twin.rescore_temporal``)."""
    if not (temporal_npz and os.path.exists(temporal_npz)):
        return None
    tz = np.load(temporal_npz, allow_pickle=False)
    if temporal_key == "auto":
        temporal_key = next((k for k in ("pred", "temporal_spreadL") if k in tz.files), "")
    if temporal_key not in tz.files or "clim" not in tz.files:
        return None
    T_fit = int(tz["T_fit"])
    o = tz["obs"][:, T_fit:]

    def _rmse(p):
        p = p[:, T_fit:] if p.shape[1] == tz["obs"].shape[1] else p
        ok_ = np.isfinite(p) & np.isfinite(o[:, :p.shape[1]])
        return float(np.sqrt(np.mean((p - o[:, :p.shape[1]])[ok_] ** 2)))

    rm, rc = _rmse(tz[temporal_key]), _rmse(tz["clim"])
    out = {"rmse_model": rm, "rmse_clim": rc, "months": int(o.shape[1]),
           "n_wells": int(o.shape[0]), "fit_months": T_fit, "passed": bool(rm <= rc),
           "source": os.path.relpath(temporal_npz)}
    here = os.path.dirname(os.path.abspath(temporal_npz))
    rescored = _rescored(rescore_csv)
    sc = next((x for x in (_scorecard(os.path.join(here, "coupled_leveling", "scorecard.json")),
                           _scorecard(os.path.join(here, "scorecard.json"))) if x), None)
    if sc:
        out.update({k: sc.get(k) for k in ("r2_shape", "r2_well_median", "level_err_m",
                                           "verdict")})
        # the scorecard's raw numbers are the calibration's own (NaN-aware) scoring
        if sc.get("rmse") is not None and sc.get("rmse_clim") is not None:
            out["rmse_model"], out["rmse_clim"] = sc["rmse"], sc["rmse_clim"]
            out["passed"] = bool(sc["rmse"] <= sc["rmse_clim"])
        fair = sc.get("fair") or rescored.get(os.path.basename(here))
        if fair:
            out["fair"] = fair
            out["passed"] = fair["verdict"] == "PASS"
            if fair.get("n_wells"):
                out["n_wells"] = int(fair["n_wells"])
    th = _theta(os.path.join(here, "stage3_theta.json"))
    out["spread_km"] = _spread_of(th)
    import glob

    others = []
    for p in sorted(glob.glob(TEMPORAL_GLOB)):
        if os.path.dirname(os.path.abspath(p)) == here:
            continue
        o2 = _scorecard(p)
        if o2 and o2.get("rmse") is not None:
            name = os.path.basename(os.path.dirname(p))
            o2["fair"] = o2.get("fair") or rescored.get(name)
            others.append({"name": name, **o2})
    out["others"] = others
    return out


def _rescored(path: str | None) -> dict:
    """``{screen: fair verdict}`` from ``twin.rescore_temporal``'s table (empty without it)."""
    if not (path and os.path.exists(path)):
        return {}
    d = pd.read_csv(path)
    out = {}
    for _, r in d.iterrows():
        out[str(r["screen"])] = _fair_of({k: r.get(k) for k in d.columns}, prefix="")
    return {k: v for k, v in out.items() if v}


def _fair_of(t: dict, prefix: str = "fair_") -> dict | None:
    """The fair held-out-years verdict's fields (``drift_diag.fair_temporal_verdict``)."""
    verdict = t.get("verdict_fair", t.get(f"{prefix}verdict_fair"))
    if verdict not in ("PASS", "FAIL"):
        return None

    def num(k):
        v = t.get(prefix + k)
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        return round(v, 4) if np.isfinite(v) else None

    return {"verdict": verdict, "ratio": num("rmse_ratio_fair"), "k": num("fair_k"),
            "shape_tol": num("fair_shape_tol"), "rmse": num("rmse_datum_model_m"),
            "rmse_raw": num("rmse_raw_model_m"), "datum_share": num("datum_share_of_mse"),
            "best": t.get(prefix + "best_baseline"), "rmse_best": num("rmse_best_baseline_m"),
            "rmse_clim": num("rmse_clim_m"), "rmse_clim_trend": num("rmse_clim_trend_m"),
            "shape": num("r2_shape_datum_model"), "shape_clim": num("r2_shape_clim"),
            "shape_clim_trend": num("r2_shape_clim_trend"),
            "level_err_rms": num("level_err_rms_m"), "n_wells": num("n_wells_scored"),
            "n_excluded": num("n_wells_excluded_short_fit")}


def _scorecard(path: str) -> dict | None:
    """The temporal block of a ``calibrate_flow`` scorecard, the fields the page quotes."""
    try:
        with open(path, encoding="utf-8") as fh:
            t = json.load(fh).get("temporal") or {}
    except (OSError, ValueError):
        return None

    def num(k):
        v = t.get(k)
        return None if v is None or not np.isfinite(float(v)) else round(float(v), 3)

    ext = [k for k, on in (("delay storage", t.get("delay_storage") not in (None, "off")),
                           ("rivers", t.get("rivers") not in (None, "none")),
                           ("canal water", isinstance(t.get("sw_recharge"), str)
                            and bool(t.get("sw_recharge"))))
           if on]
    return {"rmse": num("rmse_model_m"), "rmse_clim": num("rmse_clim_m"),
            "r2_shape": num("r2_shape_model"), "r2_well_median": num("r2_well_median_model"),
            "level_err_m": num("level_err_model_m"), "verdict": t.get("verdict"),
            "extensions": ext, "fair": _fair_of(t)}


def _theta(path: str | None) -> dict:
    """``{"theta": ..., "meta": ...}`` of a calibration, empty when unreadable."""
    with contextlib.suppress(OSError, ValueError, TypeError), open(path, encoding="utf-8") as fh:
        return json.load(fh)
    return {}


def _spread_of(th: dict) -> float | None:
    t = th.get("theta") or {}
    if "spread_km" in t:
        return float(t["spread_km"])
    if "log_spread_km" in t:
        return float(np.exp(t["log_spread_km"]))
    return None


def _flow_scores(run_dir: str | None, gate_all: dict | None = None) -> dict:
    """The head tests of a flow calibration: the k-fold on levels (``r2_kfold`` against
    IDW) and on head changes after each well's mean (``r2_anom_kfold``, the anomaly verdict
    pre-registered 2026-09-26, ``twin.kfold_scores``). Read from ``stage3_flow.csv`` of
    ``run_dir``, else from its ``stage3_kfold_anom.csv``, else from the forward run's gate;
    ``None`` where a number was never computed."""
    gate = (gate_all or {}).get("gate") or {}
    out = {"r2_kfold": gate.get("r2_kfold"), "r2_idw": gate.get("r2_idw"),
           "verdict": gate.get("verdict"), "n_folds": gate.get("n_folds"),
           "r2_anom": gate.get("r2_anom_kfold"), "r2_anom_idw": gate.get("r2_anom_idw"),
           "verdict_anom": gate.get("verdict_anom"), "anom_well_median": None,
           "anom_well_median_idw": None}
    rows = []
    for name in ("stage3_flow.csv", "stage3_kfold_anom.csv"):
        p = os.path.join(run_dir or "", name)
        if run_dir and os.path.exists(p):
            with contextlib.suppress(OSError, ValueError, pd.errors.ParserError):
                rows.append(pd.read_csv(p).iloc[0])

    def pick(col, key, cast=float):
        for r in rows:
            v = r.get(col)
            if v is not None and not (isinstance(v, float) and not np.isfinite(v)):
                out[key] = cast(v)
                return

    pick("r2_kfold", "r2_kfold")
    pick("r2_idw", "r2_idw")
    pick("verdict_kfold", "verdict", str)
    pick("n_folds", "n_folds", int)
    pick("r2_anom_kfold", "r2_anom")
    pick("r2_anom_idw", "r2_anom_idw")
    pick("verdict_anom_kfold", "verdict_anom", str)
    pick("r2_anom_well_median_kfold", "anom_well_median")
    pick("r2_anom_well_median_idw", "anom_well_median_idw")
    return out


def _column_scores(column_csv: str | None) -> dict | None:
    """The per-zone column's leveling skill out of fold and against the independent rings."""
    if not (column_csv and os.path.exists(column_csv)):
        return None
    cc = pd.read_csv(column_csv)
    r = cc[cc.config == "zonal"] if "zonal" in set(cc.config) else cc.iloc[[0]]
    out = {"config": str(r.config.iloc[0]), "r2_oof": float(r.r2_outoffold.iloc[0]),
           "rings": float(r.rings_independent_r2.iloc[0]), "n_rings": None}
    with contextlib.suppress(OSError, ValueError, KeyError, IndexError):
        out["n_rings"] = int(pd.read_csv(RINGS_CSV)["n_sites"].iloc[0])
    return out


def _policy_verdict(run_dir: str | None) -> str | None:
    """The policy-response gate (``twin.policy_gate``) of a calibration's own column."""
    sc = os.path.join(run_dir or "", "coupled_leveling", "scorecard.json")
    with contextlib.suppress(OSError, ValueError, AttributeError), open(sc, encoding="utf-8") as fh:
        return (json.load(fh).get("policy_response") or {}).get("verdict")
    return None


def _rheology_block(rheo_npz: str | None, rheo_column_csv: str | None,
                    column: dict | None, log) -> dict | None:
    """What a longer creep ceiling adds, from a forward run with two columns.

    The run (``twin.forward`` with two ``rheology_labels``) carries every member under
    both columns; ``<stem>.members.csv`` pairs them by (scenario, member, ic). Reported:
    the paired difference (second column minus first) of the baseline's forward
    subsidence (mean, p10, p90), the largest change it makes to any policy's paired
    response (which is why the note sits on the baseline number only), the two columns'
    leveling skill out of fold, the second column's creep ceiling, and whether that run
    still carried the start-up and restart steps (it then over- or under-states the
    total, but the steps are common to both columns)."""
    if not (rheo_npz and os.path.exists(rheo_npz)):
        return None
    csv = os.path.splitext(rheo_npz)[0] + ".members.csv"
    z = np.load(rheo_npz, allow_pickle=False)
    labels = [str(v) for v in z["rheology_labels"]] if "rheology_labels" in z.files else []
    if len(labels) < 2 or not os.path.exists(csv):
        return None
    m = pd.read_csv(csv)
    if "rheology" not in m.columns:
        return None
    key = ["scenario", "member", "ic"] if "ic" in m.columns else ["scenario", "member"]
    p = m.pivot_table(index=key, columns="rheology", values="subs_forward_cm")
    ref, alt = labels[0], labels[1]
    if ref not in p.columns or alt not in p.columns:
        return None
    d = (p[alt] - p[ref]).dropna()
    base = d.xs("baseline", level="scenario")
    shift = 0.0
    for sc in d.index.get_level_values("scenario").unique():
        if sc != "baseline":
            ds = d.xs(sc, level="scenario") - base
            shift = max(shift, float(np.abs(ds.mean())))
    alt_col = None
    if rheo_column_csv and os.path.exists(rheo_column_csv):
        cc = pd.read_csv(rheo_column_csv)
        r = cc[cc.config == "zonal"] if "zonal" in set(cc.config) else cc.iloc[[0]]
        alt_col = {"r2_oof": float(r.r2_outoffold.iloc[0]),
                   "rings": float(r.rings_independent_r2.iloc[0])}
    tau = None
    if rheo_column_csv:
        import glob

        for q in sorted(glob.glob(os.path.join(os.path.dirname(rheo_column_csv),
                                               "vep_*.json"))):
            with contextlib.suppress(OSError, ValueError), open(q, encoding="utf-8") as fh:
                v = json.load(fh).get("tau_max_years")
                tau = float(v) if v is not None else tau
    b_ref = p[ref].dropna().xs("baseline", level="scenario")
    b_alt = p[alt].dropna().xs("baseline", level="scenario")
    out = {"ref": ref, "alt": alt, "n": int(len(base)),
           "dBase": round(float(base.mean()), 3), "dBaseP": _q(base),
           # the baseline's forward subsidence under each column (fan mean over the runs)
           "baseRef": round(float(b_ref.mean()), 3), "baseAlt": round(float(b_alt.mean()), 3),
           "policyShift": round(shift, 4), "tauYears": tau,
           "levOof": None if column is None else round(column["r2_oof"], 3),
           "levOofAlt": None if alt_col is None else round(alt_col["r2_oof"], 3),
           "artefacts": not _artefacts_fixed_upstream(z),
           "source": os.path.basename(rheo_npz)}
    log(f"rheology {alt} vs {ref} ({out['source']}): baseline forward "
        f"{out['baseRef']:.2f} -> {out['baseAlt']:.2f} cm, "
        f"{out['dBase']:+.2f} cm (p10/p90 {out['dBaseP'][0]:+.2f}/{out['dBaseP'][2]:+.2f}), "
        f"largest change to a policy response {shift:.3f} cm; leveling out of fold "
        f"{out['levOofAlt']} vs {out['levOof']}; run "
        + ("still carried the start-up/restart steps" if out["artefacts"] else "fixed"))
    return out


def _spread_km(theta_json: str | None, gate_all: dict) -> float | None:
    if gate_all.get("pump_spread_km") is not None:
        return float(gate_all["pump_spread_km"])
    with contextlib.suppress(Exception):
        with open(theta_json, encoding="utf-8") as fh:
            th = json.load(fh).get("theta", {})
        if "spread_km" in th:
            return float(th["spread_km"])
        if "log_spread_km" in th:
            return float(np.exp(th["log_spread_km"]))
    return None


def _alt_block(alt_npz: str | None, alt_theta: str | None, members: dict | None,
               solved: list[dict], log, prev_dir: str | None = None) -> dict | None:
    """Another model's answers: the previous deliverable (``prev_dir`` set) or another
    calibration that passes the same head gate.

    Reads the other forward run's gate (head k-fold R²) and its paired members
    (``<stem>.members.csv``; its first column when it has a rheology axis), and for every
    solved scenario the two runs share, its ensemble-mean response. ``ratio`` per class is
    the other model's response over this one's for the single-class solved runs (None where
    there is none), which the page uses to say what that model would answer for a slider
    policy. With ``prev_dir`` (the previous deliverable's flow run) the block also carries
    ``prev``: its head tests, column skill and policy verdict, for the test table.
    """
    if not (alt_npz and os.path.exists(alt_npz)):
        return None
    csv = os.path.splitext(alt_npz)[0] + ".members.csv"
    if not os.path.exists(csv):
        log(f"structural alternative skipped: no {csv}")
        return None
    z = np.load(alt_npz, allow_pickle=False)
    gate_all = {}
    with contextlib.suppress(KeyError, ValueError):
        gate_all = json.loads(str(z["gate"]))
    mc = pd.read_csv(csv)
    rl = rheology_labels(z)
    if "rheology" in mc.columns and len(rl) > 1:
        mc = mc[mc.rheology == rl[0]]
    am = prep.paired_deltas(mc)
    resp = {k: {"subs_mean": v["subs_mean"], "head_mean": v["head_mean"],
                "subs_p": v["subs_p"], "n_sets": v["n_sets"], "agree_sets": v["agree_sets"]}
            for k, v in am.items() if k != "base"}
    ratio = [None] * len(prep.CLASSES)
    for x in solved:
        k = prep.single_lever(x["factors"])
        m = (members or {}).get(x["name"])
        if k is None or not m or x["name"] not in resp or abs(m["subs_mean"]) < 1e-9:
            continue
        ratio[k] = resp[x["name"]]["subs_mean"] / m["subs_mean"]
    # the same timing as this model's solved runs (first rheology column, as the page)
    timing = {}
    sm = z["subs_mean_by_rheology"][0] if len(rl) > 1 and "subs_mean_by_rheology" in z.files \
        else z["subs_mean"]
    ye, years = prep.year_ends(z["dates"])
    anames = [str(v) for v in z["scenario_names"]] if "scenario_names" in z.files else \
        [str(v).split(":")[0] for v in z["scenarios"]]
    fan = sm[:, :, ye].astype("float64").mean(axis=1) * 100.0              # (S, Y) cm
    for x in solved:
        if x["name"] in anames and anames.index(x["name"]) > 0:
            timing[x["name"]] = prep.effect_timing(fan[0], fan[anames.index(x["name"])],
                                                   years, x["start"])
    out = {"source": os.path.basename(alt_npz),
           "timing": timing,
           "timingKind": prep.timing_kind(list(timing.values())),
           "r2_kfold": (gate_all.get("gate") or {}).get("r2_kfold"),
           "spread_km": _spread_km(alt_theta, gate_all), "resp": resp, "ratio": ratio,
           "baseFwd": round(float(np.mean(am["base"]["fwd"])), 3), "prev": None}
    if prev_dir and os.path.isdir(prev_dir):
        out["prev"] = {"name": os.path.basename(os.path.normpath(prev_dir)),
                       **_flow_scores(prev_dir, gate_all),
                       "column": _column_scores(os.path.join(prev_dir, "coupled_leveling",
                                                             "stage4_column.csv")),
                       "policy": _policy_verdict(prev_dir), "temporal": None, "chain": None}
    return out


def _tau_note(column_csv: str | None, record_years: int) -> str | None:
    """"11 vs 30 yr": the deliverable column's creep ceiling against the longest ceiling
    any other leveling-calibrated column was fitted with (``TAU_ALT_GLOB``). A column
    without ``tau_max_years`` used the default ceiling, the record length."""
    import glob

    own = None
    if column_csv:
        for p in sorted(glob.glob(os.path.join(os.path.dirname(column_csv), "vep_*.json"))):
            with contextlib.suppress(OSError, ValueError), open(p, encoding="utf-8") as fh:
                v = json.load(fh).get("tau_max_years")
                own = float(v) if v is not None else own
    own = own if own is not None else float(record_years)
    alts = set()
    for p in glob.glob(TAU_ALT_GLOB):
        with contextlib.suppress(OSError, ValueError), open(p, encoding="utf-8") as fh:
            v = json.load(fh).get("tau_max_years")
            if v is not None and abs(float(v) - own) > 0.5:
                alts.add(float(v))
    return f"{own:g} vs {max(alts):g} yr" if alts else None


def _baseline_monthly(npz: str | None, g) -> np.ndarray | None:
    """Another forward run's monthly baseline subsidence (cm) on this grid, its first
    column when it has a rheology axis; ``None`` when the grids or months differ."""
    if not (npz and os.path.exists(npz)):
        return None
    z = np.load(npz, allow_pickle=False)
    if len(rheology_labels(z)) > 1 and "subs_mean_by_rheology" in z.files:
        s = z["subs_mean_by_rheology"][0][0]
    else:
        s = z["subs_mean"][0]
    if s.shape != (g.A, len(g.dates)) or [str(d)[:10] for d in z["dates"]] != g.dates:
        return None
    return s.astype("float64") * 100.0


def _prev_temporal(run_dir: str | None, rescore_csv: str | None) -> dict | None:
    """The held-out-years verdict of another model's screen (fair when known)."""
    if not (run_dir and os.path.isdir(run_dir)):
        return None
    sc = _scorecard(os.path.join(run_dir, "scorecard.json")) or {}
    fair = sc.get("fair") or _rescored(rescore_csv).get(os.path.basename(
        os.path.normpath(run_dir)))
    if not (sc or fair):
        return None
    return {"name": os.path.basename(os.path.normpath(run_dir)), "fair": fair,
            "rmse": sc.get("rmse"), "rmse_clim": sc.get("rmse_clim"),
            "verdict": sc.get("verdict")}


def _modelcard(fw, g, f, forward_npz, chain, column, temporal, spread_km, public,
               flow_dir: str | None = None) -> dict:
    bh = g.gate_all.get("bounds_hit") or {}
    n_at = sum(v.get("lo", 0) + v.get("hi", 0) for z in bh.values() for v in z.values())
    n_tr = sum(v.get("n", 0) for z in bh.values() for v in z.values())
    ga = g.gate_all
    hs = _flow_scores(flow_dir, ga)
    # the per-well datum (``calibrate_flow --well-datum``): aggregates only, never the
    # per-well offsets, which carry station ids
    wd = ga.get("well_datum_stats") or {}
    datum = None
    if ga.get("well_datum_mode") or wd:
        datum = {"mode": ga.get("well_datum_mode"), "sd_m": ga.get("well_datum_sd"),
                 **{k: (round(float(wd[k]), 3) if isinstance(wd.get(k), (int, float))
                        else wd.get(k))
                    for k in ("n", "rms_m", "max_abs_m", "mean_abs_m",
                              "r2_insample_with_datum") if k in wd}}
    return {
        "verdict": hs["verdict"], "r2_kfold": hs["r2_kfold"],
        "r2_idw": hs["r2_idw"], "n_folds": hs["n_folds"],
        "r2_anom": hs["r2_anom"], "r2_anom_idw": hs["r2_anom_idw"],
        "verdict_anom": hs["verdict_anom"], "anom_well_median": hs["anom_well_median"],
        "anom_well_median_idw": hs["anom_well_median_idw"],
        "well_datum": datum, "policy": _policy_verdict(flow_dir),
        "spread_max_km": ga.get("spread_max_km"),
        "rheology": f.rheo_labels, "rheology_ref": f.rheo_ref,
        "fix_eta": ga.get("fix_eta"), "fix_head_extra": ga.get("fix_head_extra"),
        "return_flow": ga.get("return_flow"), "learn_spread": ga.get("learn_spread"),
        "pump_spread_km": ga.get("pump_spread_km"), "spread_km": spread_km,
        "n_wells": ga.get("n_wells"),
        "git_commit": ga.get("git_commit"), "temporal_gate": ga.get("temporal_gate"),
        "zone_boundaries": g.zb, "bounds_at": n_at, "bounds_tracked": n_tr,
        "spread_at_bound": bool(bh.get("global", {}).get("log_spread_km", {}).get("hi", 0)),
        "n_members": int(fw["n_members"]) if "n_members" in fw.files else None,
        "hindcast_r2": [round(float(v), 4) for v in fw["hindcast_r2"]]
        if "hindcast_r2" in fw.files else [],
        "member_labels": [str(v) for v in fw["member_labels"]]
        if "member_labels" in fw.files else [],
        "chain": chain, "column": column, "temporal": temporal,
        "record": [g.dates[0][:7], g.dates[g.origin][:7]], "horizon": g.dates[-1][:7],
        "record_years": g.years[g.y_obs] - g.years[0] + 1,
        "free_from": g.dates[g.origin][:7],
        "artefacts": f.artefacts, "public": public,
        "source": os.path.basename(forward_npz),
    }


def build(forward_npz: str, basis_npz: str | None, out_html: str,
          townships_csv: str | None = None,
          basemap_npz: str | None = None, log=print, *, hsr_csv: str | None = geo.HSR_CSV,
          hsr_stations_csv: str | None = geo.HSR_STATIONS_CSV,
          rivers_csv: str | None = geo.RIVERS_CSV, members_csv: str | None = "auto",
          members_npz: str | None = "auto",
          leveling: str | None = "auto", wells: str | None = "auto",
          temporal_npz: str | None = DEFAULT_TEMPORAL, temporal_key: str = "auto",
          column_csv: str | None = DEFAULT_COLUMN, theta_json: str | None = DEFAULT_THETA,
          alt_npz: str | None = DEFAULT_ALT, alt_theta: str | None = DEFAULT_ALT_THETA,
          rheo_npz: str | None = DEFAULT_RHEO,
          rheo_column_csv: str | None = DEFAULT_RHEO_COLUMN,
          prev_dir: str | None = DEFAULT_PREV_DIR,
          prev_temporal: str | None = DEFAULT_PREV_TEMPORAL,
          rescore_csv: str | None = DEFAULT_RESCORE,
          three: str = "cdn", public: bool = True, max_bytes: int | None = None) -> dict:
    """Build the page. Returns the payload (a dict of plain values and packed arrays).

    ``public`` (the default) ships observations only as aggregates: township and chain
    skill from the leveling, the fan-average of the layer-2 wells. ``public=False`` adds
    every benchmark and well with its location and observed series, for a local page that
    must never be committed. ``alt_npz`` is another model's forward run, by default the
    previous deliverable (``prev_dir`` its flow run, ``prev_temporal`` its held-out-years
    screen); the page quotes its response next to the ensemble's agreement, because the
    ensemble varies parameters only, and compares its tests with this model's.
    ``rheo_npz="auto"`` takes the rheology axis from the forward run itself when it has
    two columns.
    """
    fw = np.load(forward_npz, allow_pickle=False)
    g = _geometry(fw)
    A, y_obs, y_ref = g.A, g.y_obs, g.y_ref
    sizes = {}

    # ---- ground and imagery -------------------------------------------------------------
    ground, imagery, attrib = None, {}, []
    if basemap_npz and os.path.exists(basemap_npz):
        bm = np.load(basemap_npz, allow_pickle=False)
        if "dem" in bm.files and len(bm["dem"]) == A:
            ground = bm["dem"].astype("float64")
        for key, name in (("PHOTO2", "photo"), ("EMAP", "emap")):
            if key in bm.files:
                imagery[name] = _encode_image(bm[key])
                sizes[f"img:{name}"] = len(imagery[name])
        attrib = [str(a) for a in bm["attrib"]] if "attrib" in bm.files else []
    if ground is None:
        ground = _ground_idw(g.cent, log)
    ground_ok = ground is not None
    if ground is None:
        ground = np.zeros(A)

    # ---- townships, forward run, basis --------------------------------------------------
    cell_town = pd.Series(dtype=object)
    if townships_csv and os.path.exists(townships_csv):
        cell_town = pd.read_csv(townships_csv).set_index("cell")["town"]
    town_idx, towns = prep.township_index(cell_town, A)
    f = _forward_fields(fw, g, log)
    base_ye, solved = f.base_ye, f.solved
    basis, berr, calib, basis_off = _response_basis(basis_npz, g, solved, log)

    # fixed difference scales, chosen once (spec §3.2)
    if basis is not None:
        sc_s = prep.delta_scale(basis["dsub"], y_ref, prep.DELTA_SNAPS_CM)
        sc_h = prep.delta_scale(basis["dheadL2"], y_ref, prep.DELTA_SNAPS_M)
    else:
        # without a basis, the largest solved response sets the scale, rounded up so that
        # no solved policy saturates it (a basis-free page shows those only)
        vs = [(float(np.percentile(np.abs(x["_dsub"][:, -1] - x["_dsub"][:, y_ref]), 98)),
               float(np.percentile(np.abs(x["_dh2"][:, -1]), 98)), x["name"]) for x in solved]
        v, vh, lever = max(vs) if vs else (0.0, 0.0, None)
        sc_s = {"limit": prep.snap_up(v, prep.DELTA_SNAPS_CM) if v > 0 else 1.0, "p98": v,
                "lever": lever}
        vh = max((h for _, h, _ in vs), default=0.0)
        sc_h = {"limit": prep.snap_up(vh, prep.DELTA_SNAPS_M) if vh > 0 else 1.0, "p98": vh,
                "lever": lever}
    # year-on-year rates from Dec to Dec; the first year-end is left out, its "rate" is the
    # whole first year including the column's start-up
    rates = np.diff(base_ye, axis=1) if base_ye.shape[1] > 1 else base_ye
    scales = {"dsubs": sc_s, "dhead": sc_h,
              "absSubs": prep.abs_limits(base_ye[:, -1]),
              "absHead": prep.abs_limits(f.head_ye[1]),
              "rate": [0.0, float(np.ceil(np.percentile(rates, 98) * 2) / 2)],
              "fwd": prep.abs_limits(base_ye[:, -1] - base_ye[:, y_ref])}

    # ---- members ------------------------------------------------------------------------
    if members_csv == "auto":
        members_csv = os.path.splitext(forward_npz)[0] + ".members.csv"
    members = None
    if members_csv and os.path.exists(members_csv):
        mcsv = pd.read_csv(members_csv)
        if f.rheo_ref is not None and "rheology" in mcsv.columns:
            mcsv = mcsv[mcsv.rheology == f.rheo_ref]
        members = prep.paired_deltas(mcsv)
    for x in solved:
        x["members"] = (members or {}).get(x["name"])
    hsr_out = _hsr_block(hsr_csv, hsr_stations_csv, g)
    member_arrays, member_info = _member_fields_block(forward_npz, members_npz, g, f, solved,
                                                      town_idx, len(towns), hsr_out, log,
                                                      f.rheo_ref)
    alt = _alt_block(alt_npz, alt_theta, members, solved, log, prev_dir)
    if alt and alt["prev"] is not None:
        alt["prev"]["temporal"] = _prev_temporal(prev_temporal, rescore_csv)

    # ---- observations -------------------------------------------------------------------
    inp = None
    if wells == "auto":
        try:
            inp = _load_inputs()
        except Exception as e:
            log(f"wells and class energies unavailable ({type(e).__name__}: {e})")
    class_gwh, wells_fan, wells_out = _wells_block(inp, g, f.head_ye, public)
    if wells_out is not None:
        sizes["wells"] = len(wells_out["obs"]["z"])
    extra = {}
    if alt and alt["prev"] is not None:
        pm = _baseline_monthly(alt_npz, g)
        if pm is not None:
            extra["prev"] = pm
    chain, town_skill, lev_pair = _leveling_block(_leveling_dir(leveling), g, f.subs[0],
                                                  town_idx, len(towns), public, log, extra)
    if chain is not None:
        ex = chain.pop("extra", {})
        if "prev" in ex:
            alt["prev"]["chain"] = ex["prev"]
            log(f"previous model ({alt['source']}): full-chain hindcast R² "
                f"{ex['prev']['r2']:+.3f} against {chain['r2']:+.3f} here")
    lev_out, lev = lev_pair if lev_pair else (None, None)
    if lev_out is not None:
        sizes["leveling"] = len(json.dumps(lev_out))
    if hsr_out is not None:
        sizes["hsr"] = len(json.dumps(hsr_out))
    rivers_out = _rivers_block(rivers_csv, g)

    # ---- model card ---------------------------------------------------------------------
    temporal = _temporal_block(temporal_npz, temporal_key, rescore_csv)
    column = _column_scores(column_csv)
    modelcard = _modelcard(fw, g, f, forward_npz, chain, column, temporal,
                           _spread_km(theta_json, g.gate_all), public,
                           os.path.dirname(theta_json) if theta_json else None)
    if rheo_npz == "auto":
        rheo_npz = forward_npz if len(f.rheo_labels) > 1 else None
    rheology = _rheology_block(rheo_npz, rheo_column_csv, column, log)
    # the held-out-years test ran the model free for ``months``: that is the tested horizon
    y_tested = min(y_obs + (int(np.ceil(temporal["months"] / 12)) if temporal else 0),
                   len(g.years) - 1)

    # ---- payload ------------------------------------------------------------------------
    arrays = {
        "cols": prep.pack(g.cols, None, "int16"), "rows": prep.pack(g.rows, None, "int16"),
        "zone": prep.pack(g.zone, None, "uint8"), "town": prep.pack(town_idx, None, "uint8"),
        "ground": prep.pack(ground, 0.1),
        "subsBase": prep.pack(base_ye, 0.01),
        # the final-year rate at full precision, so the area tile counts what Python counts
        "rateBase": prep.pack(base_ye[:, -1] - base_ye[:, -2], 0.001),
        "subsBand": prep.pack(f.band, 0.01),
        "headBase": prep.pack(f.head_ye, 0.01),
    }
    if basis is not None:
        for key, arr in (("dSubs", basis["dsub"]), ("dHeadL2", basis["dheadL2"]),
                         ("dHeadEnd", basis["dhead_end"])):
            arrays[key] = prep.pack(arr, max(0.001, float(np.abs(arr).max()) / 30000.0))
    for i, x in enumerate(solved):
        for key, arr in ((f"solvedSubs{i}", x["_dsub"]), (f"solvedHead{i}", x["_dh2"])):
            arrays[key] = prep.pack(arr, max(0.001, max(float(np.abs(arr).max()), 1e-6)
                                             / 30000.0))
    arrays.update(member_arrays)
    for k, v in arrays.items():
        sizes[k] = len(v["z"])

    payload = {
        "meta": {"nx": g.nx, "ny": g.ny, "dx": g.dx, "x0": g.x0, "y0": g.y0, "nA": A,
                 "L": int(f.L), "years": g.years, "yObs": y_obs, "yRef": y_ref,
                 "yTested": y_tested, "starts": list(prep.STARTS),
                 "t0": g.dates[0][:7], "rawFwdFan": round(f.raw_fwd_fan, 3),
                 "vintage": g.dates[g.origin][:7], "groundOk": ground_ok},
        "arrays": arrays,
        "img": imagery, "attrib": attrib,
        "towns": towns, "townSkill": town_skill,
        "classes": [{"key": c, "en": CLASS_LABELS[c][0], "zh": CLASS_LABELS[c][1],
                     "gwh": None if class_gwh is None else round(class_gwh[k], 3)}
                    for k, c in enumerate(prep.CLASSES)],
        "basis": None if basis is None else {"labels": basis["labels"],
                                             "missing": basis["missing"]},
        "basisError": berr, "calib": calib, "basisOff": basis_off,
        "solved": [{k: v for k, v in x.items() if not k.startswith("_")} for x in solved],
        "members": None if members is None else {"base": members["base"]},
        "memberFields": member_info,
        "scales": scales,
        "hsr": hsr_out, "rivers": rivers_out, "leveling": lev_out, "wells": wells_out,
        "wellsFan": wells_fan,
        "modelcard": modelcard,
        "palettes": prep.PALETTES, "vScale": prep.VERT_SCALE,
        "layerDepths": LAYER_DEPTHS, "layerThick": AQUIFER_THICK,
        "zoneNames": list(ZONE_NAMES),
        "alt": alt,
        "caveatWords": {"tau": _tau_note(column_csv, modelcard["record_years"])},
        "timingKind": prep.timing_kind([x["timing"] for x in solved]),
        "rheology": rheology,
    }

    _summary(payload, solved, members, basis, base_ye, g, towns, town_idx, town_skill,
             lev, hsr_out, log)

    html = _render(payload, three, log)
    size = len(html.encode("utf-8"))
    limit = MAX_BYTES if max_bytes is None else max_bytes
    log("payload (base64 bytes): " + ", ".join(
        f"{k} {v / 1e3:.0f} KB" for k, v in sorted(sizes.items(), key=lambda kv: -kv[1])))
    if size > limit:
        raise SystemExit(f"page would be {size / 1e6:.2f} MB, above the {limit / 1e6:.1f} MB "
                         "guard; refusing to write it")
    os.makedirs(os.path.dirname(out_html) or ".", exist_ok=True)
    with open(out_html, "w", encoding="utf-8") as fh:
        fh.write(html)
    note = "" if size <= TARGET_BYTES else f" (above the {TARGET_BYTES / 1e6:.0f} MB target)"
    log(f"wrote {out_html} ({size / 1e6:.2f} MB{note}, {A} cells, {len(g.years)} year-ends, "
        f"{0 if basis is None else len(basis['labels'])} basis rows, {len(solved)} solved; "
        + ("public: observations as aggregates only)" if public
           else "PRIVATE: carries observed points, never commit it)"))
    return payload


def _render(payload: dict, three: str, log=print) -> str:
    with open(TEMPLATE, encoding="utf-8") as fh:
        page = fh.read()
    blob = json.dumps(payload, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False, default=_json_default)
    blob = blob.replace("</", "<\\/")
    inline = ""
    if three == "inline":
        src = os.environ.get("TWIN_THREE_JS", THREE_LOCAL)
        if os.path.exists(src):
            with open(src, encoding="utf-8") as fh:
                inline = "<script>" + fh.read() + "</script>"
        else:
            log(f"--three inline: no pinned three.min.js at {src}; falling back to cdn")
            three = "cdn"
    page = page.replace("__THREE_MODE__", three)
    page = page.replace("__THREE_CDN__", THREE_CDN if three == "cdn" else "")
    page = page.replace("<!--__THREE_INLINE__-->", inline)
    return page.replace("__PAYLOAD__", blob)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def _summary(payload, solved, members, basis, base_ye, g, towns, town_idx, town_skill,
             lev, hsr_out, log) -> None:
    """Print what a reviewer checks before sharing the page (spec §8)."""
    yrs = payload["meta"]["years"]
    y0 = g.y_ref
    fwd = base_ye[:, -1] - base_ye[:, y0]
    log(f"baseline: fan-mean forward subsidence Dec {yrs[y0]}-Dec {yrs[-1]} {fwd.mean():.2f} "
        + ("cm (the last fitted year-end; steps fixed upstream)" if g.fixed
           else "cm (after the restart has settled)")
        + "; cells above 1/2/3 cm/yr in the final year: "
        + "/".join(str(prep.rate_area(base_ye, None, t)) for t in (1, 2, 3)))
    for x in solved:
        m = x.get("members")
        fan = (x["_dsub"][:, -1] - x["_dsub"][:, y0]).mean()
        if m:
            s, h = m["subs_p"], m["head_p"]
            log(f"solved {x['name']}: Δ forward subsidence mean {m['subs_mean']:+.2f}, "
                f"p10/p50/p90 {s[0]:+.2f} / {s[1]:+.2f} / {s[2]:+.2f} cm; Δ layer-2 head mean "
                f"{m['head_mean']:+.2f}, {h[0]:+.2f} / {h[1]:+.2f} / {h[2]:+.2f} m; "
                f"{m['agree']}/{m['n']} runs agree (ensemble-mean field {fan:+.2f} cm)")
        else:
            log(f"solved {x['name']}: ensemble-mean Δ {fan:+.2f} cm (no members.csv)")
    if basis is not None:
        resp = [(basis["labels"][r], (basis["dsub"][r][:, -1] - basis["dsub"][r][:, y0]).mean())
                for r in range(len(basis["labels"]))]
        log("basis (rescaled), retiring each class: "
            + ", ".join(f"{k} {v:+.2f} cm" for k, v in resp))
    cal = payload.get("calib")
    if cal:
        log("basis rescaled to the ensemble, per class (subsidence / head, source): " + ", ".join(
            f"{c} {s:.3f}/{h:.3f} ({src})" for c, s, h, src in
            zip(prep.CLASSES, cal["subs"], cal["head"], cal["source"], strict=True)))
    art = payload["modelcard"]["artefacts"]
    if art["fixed_upstream"]:
        log(f"model artefacts: fixed upstream ({art['fixed_upstream']}), nothing removed; a "
            f"check of the {art['n_cells']} {art['zone']} cells still finds a first-months "
            f"step of median {art['startup_cell_median']:.1f} (max "
            f"{art['startup_cell_max']:.1f}) cm and a restart step of median "
            f"{art['restart_cell_median']:.1f} (max {art['restart_cell_max']:.1f}) cm")
    else:
        log(f"model artefacts: fan mean start-up {art['startup_cm']:.2f} cm, restart "
            f"{art['restart_cm']:.2f} cm; per cell in the {art['n_cells']} {art['zone']} "
            f"cells start-up median {art['startup_cell_median']:.1f} (max "
            f"{art['startup_cell_max']:.1f}) cm, restart median "
            f"{art['restart_cell_median']:.1f} (max {art['restart_cell_max']:.1f}) cm, removed "
            f"from every field; the start-up elsewhere is seasonal noise (p95 "
            f"{art['other_startup_p95_cm']:.1f} cm), left in; the restart year elsewhere "
            f"bridged by median {art['bridge_median_cm']:+.2f} cm (p5 "
            f"{art['bridge_p05_cm']:+.2f}, max |{art['bridge_max_abs_cm']:.1f}|)")
    for b in art["boundaries"]:
        log(f"zone line {b['km']:g} km E: median forward subsidence first column east "
            f"{b['east']:.1f} cm, west {b['west']:.1f} cm; final-year rate east "
            f"{b['rate_east']:.2f}, west {b['rate_west']:.2f} cm/yr")
    for k, v in payload["basisError"].items():
        log(f"superposition ({v['kind']}) vs {k}: fan {v['fan_cm']:.3f} cm "
            f"({100 * v['rel']:.1f} %), p95 cell {v['p95_cell_cm']:.3f} cm")
    sc = payload["scales"]
    what = (f"a 30 % cut of {sc['dsubs']['lever']}" if basis is not None
            else f"the solved run {sc['dsubs']['lever']}")
    log(f"fixed Δ scales: ±{sc['dsubs']['limit']:g} cm (p98 {sc['dsubs']['p98']:.2f} for "
        f"{what}), ±{sc['dhead']['limit']:g} m head; absolute "
        f"subsidence p2-p98 {sc['absSubs'][0]:.1f}-{sc['absSubs'][1]:.1f} cm")
    top = np.argsort(-fwd)[:5]
    lev_cells = lev["cell"] if lev else np.zeros(0, dtype=int)
    log("top 5 cells by baseline forward subsidence (cell, township, cm, benchmarks within "
        "the cell, their mean bias):")
    for c in top:
        t = towns[town_idx[c]]
        sel = np.nonzero(lev_cells == c)[0]
        b = np.mean(lev["bias"][sel]) if len(sel) else float("nan")
        log(f"  cell {c} ({g.cent[c, 0] / 1000:.1f}, {g.cent[c, 1] / 1000:.1f} km) {t['zh']} "
            f"{t['en']}: {fwd[c]:.1f} cm; leveling n={len(sel)} bias {b:+.1f} cm")
    tf = [(i, fwd[town_idx == i].mean()) for i in range(len(towns)) if (town_idx == i).any()]
    log("top 5 townships by baseline forward subsidence (mean cm, leveling n, bias, R², "
        "confidence):")
    for i, v in sorted(tf, key=lambda kv: -kv[1])[:5]:
        s = (town_skill or [{}] * len(towns))[i]
        log(f"  {towns[i]['zh']} {towns[i]['en']}: {v:.1f} cm; n={s.get('n')} "
            f"bias {s.get('bias')} R² {s.get('r2')} {'LOW' if s.get('low') else 'ok'}")
    for x in solved:
        fs = x.get("fast") or {}
        if fs.get("dec") is not None:
            log(f"solved {x['name']}: share of the {yrs[-1]} fan change reached by Dec "
                f"{x['start']} {100 * fs['dec']:.0f} %, peak in the first year "
                f"{100 * fs['peak']:.0f} % ({fs['peak_month']})")
    alt = payload.get("alt")
    for who, tim in [("this model", {x["name"]: x["timing"] for x in solved})] + (
            [(f"previous model {alt['source']}", alt.get("timing") or {})] if alt else []):
        for name, t in tim.items():
            ys = t["rate_years_span"]
            log(f"timing, {who}, {name}: avoided " + ", ".join(
                f"{a['avoid']:.2f} cm by Dec {a['year']}" for a in t["at"])
                + f"; fan sinking rate {ys[0]}-{ys[1]} {t['rate_base']:.2f} -> "
                f"{t['rate_pol']:.2f} cm/yr (slows {t['slows']}, grows {t['grows']})")
    if alt:
        log(f"structural alternative {alt['source']} (head k-fold {alt['r2_kfold']}, spread "
            f"{alt['spread_km']} km): " + ", ".join(
                f"{k} {v['subs_mean']:+.2f} cm ({v['agree_sets']}/{v['n_sets']} sets)"
                for k, v in alt["resp"].items()))
    wf = payload.get("wellsFan")
    if wf and len(wf["obs"]) > 1 and None not in wf["obs"][-2:] + wf["model"][-2:]:
        log(f"layer-2 wells, fan mean {wf['years'][-2]}->{wf['years'][-1]}: observed "
            f"{wf['obs'][-1] - wf['obs'][-2]:+.2f} m, model at the same wells "
            f"{wf['model'][-1] - wf['model'][-2]:+.2f} m")
    log(f"baseline forward, ledger basis (raw, Dec {yrs[g.y_obs]}-Dec {yrs[-1]}): "
        f"{payload['meta']['rawFwdFan']:.2f} cm; page basis (Dec {yrs[y0]}-"
        + (", fixed upstream" if g.fixed else ", steps removed") + f"): {fwd.mean():.2f} cm")
    if hsr_out:
        ad = prep.angular_distortion(prep.sample(fwd, np.array(hsr_out["idx"]),
                                                 np.array(hsr_out["w"])), hsr_out["step"])
        mx = np.array(hsr_out["mixed"], dtype=bool)
        clean = ad[~mx].max() if (~mx).any() else float("nan")
        log(f"HSR: {len(hsr_out['ch'])} points over {hsr_out['ch'][-1]:.1f} km; baseline max "
            f"angular distortion {yrs[y0]}-{yrs[-1]} 1/{1 / max(ad.max(), 1e-12):,.0f} at km "
            f"{hsr_out['ch'][int(ad.argmax())]:.2f}; away from the zone lines "
            f"({mx.sum() * hsr_out['step']:.1f} km of track excluded) "
            f"1/{1 / max(clean, 1e-12):,.0f}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="build the twin's decision page")
    ap.add_argument("--forward", required=True, help="a twin.forward .npz (the deliverable run)")
    ap.add_argument("--basis", default=None, help="the per-class response basis .npz")
    ap.add_argument("--townships", default="results/twin_runs/cell_townships.csv")
    # accepted and ignored so that older command lines still run; the page steps by year
    ap.add_argument("--quarter", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--delta-step", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--basemap", default="results/twin/basemap.npz",
                    help="orthophoto, base map and terrain from twin/basemap.py")
    ap.add_argument("--hsr", default=geo.HSR_CSV, help="THSR centreline csv ('none' to omit)")
    ap.add_argument("--members-csv", default="auto",
                    help="paired member deltas (default <forward stem>.members.csv)")
    ap.add_argument("--members-npz", default="auto",
                    help="per-member yearly fields from twin.forward --save-members yearly "
                         "(default <forward stem>.members.npz): real per-cell agreement, "
                         "yearly fan bands and township agreement; 'none' to use the "
                         "fallback")
    ap.add_argument("--leveling", default="auto",
                    help="data dir holding ls_cache/ (default $HYDROMIND_GW_DATA; 'none')")
    ap.add_argument("--wells", default="auto",
                    help="'auto' loads the calibration wells and class energies; 'none'")
    ap.add_argument("--temporal", default=DEFAULT_TEMPORAL,
                    help="held-out-years predictions npz (stage3_temporal_pred.npz of the "
                         "deliverable's screen; its scorecard gives the fair verdict)")
    ap.add_argument("--column-csv", default=DEFAULT_COLUMN, help="column skill table")
    ap.add_argument("--theta", default=DEFAULT_THETA,
                    help="the calibration's theta json (for the learned stress radius)")
    ap.add_argument("--alt-forward", default=DEFAULT_ALT,
                    help="another model's forward run, by default the previous deliverable "
                         "(its <stem>.members.csv is read); 'none' to omit")
    ap.add_argument("--alt-theta", default=DEFAULT_ALT_THETA,
                    help="that model's theta json (for its stress radius)")
    ap.add_argument("--rheo-forward", default=DEFAULT_RHEO,
                    help="a forward run with a second column (rheology axis), for the creep "
                         "note on the baseline; its <stem>.members.csv is read; 'auto' (the "
                         "forward run itself when it has two columns) or 'none'")
    ap.add_argument("--rheo-column-csv", default=DEFAULT_RHEO_COLUMN,
                    help="that second column's skill table (leveling out of fold)")
    ap.add_argument("--prev-dir", default=DEFAULT_PREV_DIR,
                    help="the previous deliverable's flow run (head tests, column), compared "
                         "with this model in the test table; 'none'")
    ap.add_argument("--prev-temporal", default=DEFAULT_PREV_TEMPORAL,
                    help="the previous deliverable's held-out-years screen dir; 'none'")
    ap.add_argument("--rescore-csv", default=DEFAULT_RESCORE,
                    help="twin.rescore_temporal's table: the fair verdict of older screens")
    ap.add_argument("--three", choices=("cdn", "inline", "none"), default="cdn")
    ap.add_argument("--private", action="store_true",
                    help="embed every benchmark and well with its location and observed "
                         "series; for a local page only, never commit the output")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    def opt(v):
        return None if v in (None, "none") else v

    build(args.forward, args.basis, args.out, townships_csv=opt(args.townships),
          basemap_npz=opt(args.basemap),
          hsr_csv=opt(args.hsr), members_csv=opt(args.members_csv),
          members_npz=opt(args.members_npz), leveling=args.leveling,
          wells=args.wells, temporal_npz=opt(args.temporal), column_csv=opt(args.column_csv),
          theta_json=opt(args.theta), alt_npz=opt(args.alt_forward),
          alt_theta=opt(args.alt_theta), rheo_npz=opt(args.rheo_forward),
          rheo_column_csv=opt(args.rheo_column_csv), prev_dir=opt(args.prev_dir),
          prev_temporal=opt(args.prev_temporal), rescore_csv=opt(args.rescore_csv),
          three=args.three, public=not args.private,
          log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    sys.exit(main())
