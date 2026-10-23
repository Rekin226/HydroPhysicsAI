"""WP H2 repair round 3 (after the independent verification of 2026-10-14): shared constants
and helpers. The v1/v2 modules are hashed by the registrations and the published manifests
and stay unchanged; everything new lives in ``r3_*`` and ``s1x_*`` modules.

What the verifier found and what this round does about it:

* **Initial condition.** ``H2.ic.lowo`` compared the GP field with plain per-layer IDW, but
  the calibrated stage-3 runs (``results/twin_runs/stage3_datum_gate_cwa``) build the initial
  condition with ``no_backfill = False`` (every well's month-0 value of the backfilled series,
  so a well that starts later contributes its first observed head) and
  ``ic_merged_proximal = True`` with zones ``205,182,208``. :func:`twin_ic_field` reproduces
  that construction by calling the twin's own functions (``calibrate_flow._idw_initial_heads``,
  ``_merged_proximal_heads``, ``_ic_zone_map``, ``prepare_series``) on CPU.
* **Family E freshness.** The pilot's E r0-1 well-gaps are design-time units and are the same
  cells as H2's (E masks are deterministic per well); :func:`pilot_e_gaps` lists them so that
  they are excluded from the E gates like H2's own design realizations.
* **Use section.** A registered replication of the daily-bridge test on realizations never
  generated (family d, r = 60..69) carries a packaging rule declared before it is run: the
  full-variant daily-bridge component (Feb-Mar 2026 bridged) for the statistical layer's
  history. A registered confirmatory test of the initial-condition field against the twin's
  actual construction on fresh months (January 2013..2022) carries the rule for releasing a
  January 2012 field for ``initial_conditions``.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from h2_heads_fill import config as C

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # never the GPU (Phase-0 rule)

WP = C.WP
PREREG_ID_R3 = "H2.repair3.v1"
TEST_D3 = "H2.fresh3.d"
TEST_IC = "H2.ic.twin"
FRESH_D3 = range(60, 70)                 # family d realizations never generated before
IC_MONTHS = tuple(f"{y}-01" for y in range(2013, 2023))   # fresh months of H2.ic.twin
IC_MONTH0 = "2012-01"                    # the twin's month 0 (H2.ic.lowo, post-look here)
ZONE_BOUNDARIES = "205,182,208"          # stage-3 recipe (stage3_fit_summary.json)
SITE_KM = 0.1                            # wells within 100 m form one site (nested wells)
N_BOOT = 2000
EV_R3 = C.EVIDENCE_DIR / "repair3"
REG_R3 = EV_R3 / "registered"
PILOT_DIR = C.REPO / "results/dataprep/_pilots/head_gapfill_pilot"
STAGE3_SUMMARY = C.REPO / "results/twin_runs/stage3_datum_gate_cwa/stage3_fit_summary.json"
PARAMS_V1 = C.EVIDENCE_DIR / "design" / "design_params.json"
PARAMS_V2 = C.EVIDENCE_DIR / "design_v2" / "design_params_v2.json"
#: E realizations that H2 generated (design r0-4, v1 fresh r10-19, v2 fresh r30-59)
E_R_DESIGN = tuple(range(0, 5))
E_R_V1 = tuple(range(10, 20))
E_R_V2 = tuple(range(30, 60))


def stratum_of_L(L: int) -> str:
    return "E_lead66" if int(L) == 66 else "E_trail48"


def pilot_e_gaps() -> set[tuple[str, str]]:
    """(sid, stratum) well-gaps of the pilot's family E realizations r0-1 (identity only; the
    pilot's values are not used). Read through the loader guard for after-the-fact analysis
    (the pilot is design-time evidence with fit_window_end 2026-09)."""
    import sys
    sys.path.insert(0, str(C.REPO / "dataprep_staging"))
    from common import guard
    p = guard.open_product(PILOT_DIR, "after_the_fact_analysis")
    e = p.read_table("out/pilot_E.parquet", columns=["real", "sid", "L"])
    e["sid"] = e["sid"].astype(str).str.zfill(8)
    return {(s, stratum_of_L(L)) for s, L in zip(e.sid, e.L, strict=True)}


def site_ids(xy: np.ndarray, km: float = SITE_KM) -> np.ndarray:
    """Cluster wells into sites (single linkage at ``km``): nested wells share a site."""
    n = len(xy)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    d = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1]) / 1e3
    for i, j in zip(*np.nonzero(np.triu(d <= km, 1)), strict=True):
        a, b = find(i), find(j)
        if a != b:
            parent[b] = a
    roots = np.array([find(i) for i in range(n)])
    _, lab = np.unique(roots, return_inverse=True)
    return lab


def monthly_values(h) -> np.ndarray:
    """(W, T) L1 flag-0/1 monthly value: the C20 mean, else the partial-month mean (the
    definition of ``s6_ic.jan2012``, any clean-day count)."""
    return np.where(np.isfinite(h.Y), h.Y, h.partial_mean)


def backfilled(vals: np.ndarray) -> np.ndarray:
    """The twin's ``prepare_series(backfill=True)`` row by row (linear inside, constant at
    the edges); rows without any value stay NaN."""
    from hydrophysics.twin.calibrate_flow import prepare_series
    out = np.full_like(vals, np.nan)
    for i, s in enumerate(vals):
        if np.isfinite(s).any():
            out[i] = prepare_series(s, backfill=True)
    return out


class TwinIC:
    """The stage-3 initial-condition construction on the model grid (CPU torch)."""

    def __init__(self, zone_boundaries: str = ZONE_BOUNDARIES):
        from h2_heads_fill import io_h2
        from hydrophysics.twin import calibrate_flow as cf
        self.cf = cf
        self.grid, self.dem = io_h2.grid_and_dem()
        self.zone = cf._ic_zone_map(self.grid, zone_boundaries)
        self.centroids = self.grid.centroids()

    def field(self, xy, vals, lay0, cells, merged: bool = True) -> np.ndarray:
        """(4, A) field: per-layer IDW of the finite ``vals`` (``_idw_initial_heads``), then
        the proximal zones overwritten by the merged-aquifer IDW (``_merged_proximal_heads``,
        ``layer_of=None`` as in stage 3)."""
        ok = np.isfinite(vals)
        xy, vals, lay0, cells = xy[ok], vals[ok], lay0[ok], cells[ok]
        h0 = self.cf._idw_initial_heads(self.grid, xy, vals, lay0, n_layers=4)
        if merged:
            h0, _ = self.cf._merged_proximal_heads(self.grid, h0, xy, vals, self.zone,
                                                    self.zone[cells], layer_of=None)
        return h0.cpu().numpy()

    def is_proximal(self, cells) -> np.ndarray:
        from hydrophysics.twin.zones import PROXIMAL, PROXIMAL_W
        return np.isin(self.zone[np.asarray(cells)], (PROXIMAL, PROXIMAL_W))


def idw_point_merged(xy, vals, lay0, xy_te, lay_te, zone_tr, zone_te, prox_te):
    """Point (well-location) version of the twin's construction (the verifier's form): per-
    layer IDW at the point; inside a proximal zone the IDW of that zone's wells of any layer
    (all proximal wells when the zone has none)."""
    from h2_heads_fill import ic_field
    from hydrophysics.twin.zones import PROXIMAL, PROXIMAL_W
    ok = np.isfinite(vals)
    xy, vals, lay0, zone_tr = xy[ok], vals[ok], lay0[ok], zone_tr[ok]
    out = ic_field.idw_layered(xy, vals, lay0, xy_te, lay_te)
    any_prox = np.isin(zone_tr, (PROXIMAL, PROXIMAL_W))
    for k in range(len(xy_te)):
        if prox_te[k]:
            m = zone_tr == zone_te[k]
            if not m.any():
                m = any_prox
            if m.any():
                out[k] = ic_field.idw_predict(xy[m], vals[m], xy_te[[k]])[0]
    return out


def rmse(e) -> float:
    e = np.asarray(e, float)
    e = e[np.isfinite(e)]
    return float(np.sqrt(np.mean(e ** 2))) if e.size else float("nan")


def cluster_boot_diff(err_a, err_b, clusters, n=N_BOOT, seed=C.SEED_SPLIT):
    """p05/p50/p95 of RMSE(a) - RMSE(b) over a cluster bootstrap (clusters resampled with
    replacement, all their cells kept)."""
    err_a, err_b = np.asarray(err_a, float), np.asarray(err_b, float)
    cl = pd.factorize(np.asarray(clusters))[0]
    groups = [np.flatnonzero(cl == k) for k in range(cl.max() + 1)]
    rng = np.random.default_rng(seed)
    d = np.empty(n)
    for b in range(n):
        ix = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
        d[b] = rmse(err_a[ix]) - rmse(err_b[ix])
    return [float(np.quantile(d, q)) for q in (0.05, 0.5, 0.95)]


def jdump(obj, path):
    from h2_heads_fill import scoring
    scoring.jdump(obj, path)
