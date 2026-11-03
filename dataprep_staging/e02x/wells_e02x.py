"""The external-well and donor tables of E02x, from METADATA only (no head value, no residual).

* external wells: N3F's registered kept wells (N3F.ext_level.prospective_2026-11.v1,
  ``external_predictions.kept_ids``, 66) with their forward-grid cell and twin layer from
  N3F's registered export ``results/twin/prospective_2026-11_ext/wells.csv`` read with
  ``usecols`` = station_id, county, cell, layer, kept (the file's pre-origin columns
  datum_m and residual_sd_m are never loaded here), sha-checked against N3F's registration;
  the column zone (prox / mid / dist) of each cell from ``twin.zone_county.column_zone_map``
  on the frozen forward's grid and meta (model metadata).
* donors: the 158 E02 wells of DR's frozen ``wells_e02d.csv`` (sha from the registered
  U1.decomp_power.v1 design), metadata columns only for the power study; the artefact
  (build_artefact_e02x.py) also reads their layer and cell for the member spread.
* DR's grouping, groups and sub-groups per county from the registered U1.decomp_power.v1
  design (derived from DR, registered 9467e8af).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import e02xio as io  # noqa: E402
import e02xlib as L  # noqa: E402

EXT_META_COLS = ["station_id", "county", "cell", "layer", "kept"]
DONOR_META_COLS = ["station_id", "county", "layer", "layer_group", "zone", "cell", "grouping",
                   "group", "subgroup"]


def n3f_registration(prereg_dir=None) -> tuple[dict, str]:
    """N3F's registered document (verified against INDEX)."""
    return io.registered(L.N3F_ID, prereg_dir)


def u1_design(prereg_dir=None) -> tuple[dict, str]:
    doc, sha = io.registered(L.U1D_ID, prereg_dir)
    if "design" not in doc:
        raise io.EngineeringFailure(f"{L.U1D_ID} carries no design")
    return doc["design"], sha


def ext_meta(prereg_dir=None) -> tuple[pd.DataFrame, dict]:
    """(kept external wells: station_id, county, cell, layer (twin, 1-based), provenance)."""
    doc, sha = n3f_registration(prereg_dir)
    ex = doc.get("external_predictions") or {}
    kept = sorted(map(str, ex.get("kept_ids") or []))
    want = ((ex.get("files") or {}).get("wells.csv") or {}).get("sha256")
    p = io.repo_path(f"{ex.get('dir') or L.N3F_DIR}/wells.csv")
    if not p.is_file() or io.sha256(p) != want:
        raise io.EngineeringFailure(f"{p} is not N3F's registered wells.csv")
    w = pd.read_csv(p, usecols=EXT_META_COLS, dtype={"station_id": str})
    w = w[w["kept"].astype(bool)].copy()
    if sorted(w["station_id"]) != kept:
        raise io.EngineeringFailure("N3F wells.csv kept wells != the registered kept_ids")
    w["county"] = w["station_id"].map(L.county_of)
    got = w["county"].value_counts().to_dict()
    if got != L.EXPECTED_EXT:
        raise io.EngineeringFailure(f"external wells by county {got} != {L.EXPECTED_EXT}")
    if (w["cell"] < 0).any() or not w["layer"].isin([0, 1, 2, 3]).all():
        raise io.EngineeringFailure("a kept external well has no cell or a layer outside 0-3")
    w["layer"] = w["layer"].astype(int) + 1          # N3F's 0-based twin layer -> 1..4
    w["cell"] = w["cell"].astype(int)
    prov = {"n3f_registration_sha256": sha, "wells_csv": io.rec(p),
            "kept_ids_sha256": io.sha_lines(kept), "columns_read": EXT_META_COLS}
    return w.sort_values("station_id").reset_index(drop=True), prov


def zones_at(cells, forward=None) -> np.ndarray:
    """Column zone ids at forward-grid active cells (model metadata: grid and meta only)."""
    from hydrophysics.twin.grid import FanGrid
    from hydrophysics.twin.zone_county import column_zone_map

    fwd = io.repo_path(forward or L.FORWARD)
    if io.sha256(fwd) != L.MODEL_SHA:
        raise io.EngineeringFailure("forward npz sha256 differs from cf2c7154")
    with np.load(fwd, allow_pickle=False) as z:
        grid = FanGrid(int(z["nx"]), int(z["ny"]), float(z["dx"]), float(z["x0"]),
                       float(z["y0"]), z["mask"])
        meta = json.loads(str(z["gate"]))
    zoc = np.asarray(column_zone_map(grid, meta)).astype(int)
    zz = zoc[np.asarray(cells, int)]
    bad = sorted(set(zz.tolist()) - set(L.D.ZONE_IDS))
    if bad:
        raise io.EngineeringFailure(f"column zone ids {bad} outside 0..2 (DR's rule)")
    return zz


def design_groups(design: dict) -> tuple[dict, dict, dict]:
    """({county: DR group labels}, {county: sub-group labels}, {county: grouping}) from the
    registered U1.decomp_power.v1 design."""
    g, s, how = {}, {}, {}
    for c in L.GATED:
        cd = design["counties"][c]
        g[c] = sorted(set(map(str, cd["groups"])))
        s[c] = sorted(set(map(str, cd["subgroups"])))
        how[c] = str(cd["from_dr"]["grouping"])
        if how[c] not in L.GROUPINGS:
            raise io.EngineeringFailure(f"{c}: DR grouping {how[c]!r}")
    return g, s, how


def ext_table(prereg_dir=None, forward=None) -> tuple[pd.DataFrame, dict]:
    """The external-well table (metadata only): station_id, county, cell, layer,
    layer_group, zone, group, subgroup, donor_key, role."""
    w, prov = ext_meta(prereg_dir)
    design, dsha = u1_design(prereg_dir)
    zz = zones_at(w["cell"], forward)
    w["zone"] = [L.D.ZONE_NAMES[int(z)] for z in zz]
    w["layer_group"] = [L.D.LAYER_GROUP.get(int(x), "L34") for x in w["layer"]]
    g, s, how = design_groups(design)
    t = L.ext_groups(w, g, s, how)
    miss = t[(t["role"] == "gate_candidate") & (t["group"].isna() | t["subgroup"].isna())]
    if len(miss):
        raise io.EngineeringFailure(f"{len(miss)} gated external wells fall in no DR group")
    prov.update({"u1_decomp_power_sha256": dsha, "grouping": how,
                 "dr_report_sha256": design.get("dr_report_sha256"),
                 "wells_e02d_sha256": design.get("wells_e02d_sha256")})
    return t, prov


def donors_table(design: dict, cols=DONOR_META_COLS) -> pd.DataFrame:
    """DR's frozen wells_e02d.csv (the 158 donors), sha-checked against the registered
    U1.decomp_power.v1 design, ``cols`` only."""
    p = io.repo_path(f"{L.DR_DIR}/wells_e02d.csv")
    if io.sha256(p) != design.get("wells_e02d_sha256"):
        raise io.EngineeringFailure("wells_e02d.csv differs from the registered design's")
    d = pd.read_csv(p, usecols=list(cols), dtype={"station_id": str})
    if len(d) != L.N_DONORS:
        raise io.EngineeringFailure(f"wells_e02d.csv has {len(d)} wells, expected 158")
    return d.sort_values("station_id").reset_index(drop=True)
