"""W3r after-the-fact analysis: the DTM point against agency GroundHeight at the fan wells
under both candidate survey epochs (never gating, never fed back).

Declared in ``W3r.corrected_test_design.v1`` (A1 index) BEFORE this runs. The corrected test
of ``repair_wells`` is not evaluable: only 3 wells at 2 sites have a GroundHeight surveyed in
2005-2022, and 0 fresh sites remain. The stations with a consistent GroundHeight were almost
all established in 1968-2000; their survey epoch is undocumented (establishment, or the bulk
modification of 2009-2011), and the leveling displacement between the two dates has a median
of 0.33 m. This script reports, for every one of those stations, the residual under BOTH
epoch assumptions, so that the user sees the range the well reference allows; it cannot
choose between them and it is not a test of the product.

Residual (as in ``repair_wells``): r = H_pt + [u(t_s) - u(t_sheet)] - GroundHeight, with t_s
= EstablishDate (assumption A) or ModificationDate (assumption B); also without correction,
and with the containing 1 km cell mean instead of H_pt. Station set: fan stations passing
rules E1, E2, E3, E5 and E6 at any establishment date up to 2022-12-31 (E4 dropped), plus the
3 wells that also pass E4, listed individually. The look is logged with the A1 tool
(``harness.prereg3.record_look``, purpose ``after_the_fact_analysis``) before any residual is
computed.

Usage: python -I repair_aftfact.py <out_dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from harness import prereg3  # noqa: E402
from w3_dtm import repair_wells as rw  # noqa: E402

DECLARATION = "W3r.corrected_test_design.v1"
UNIT = "W3r:fan_wells_groundheight_consistent"
SCOPE = "station_metadata_GroundHeight+w3_dtm_well_points"
RULES = ["E1_coords_dtm", "E2_groundheight", "E3_stickup", "E5_flat", "E6_no_postorigin_edit"]


def station_set(t: pd.DataFrame) -> pd.DataFrame:
    est = pd.to_numeric(t["EstablishDate"], errors="coerce")
    return t[(t["set"] == "fan") & t[RULES].all(axis=1)
             & (est <= int(rw.DESIGN["establish_window"][1]))].copy()


def main(out_dir: str) -> int:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prereg3.load_prereg(DECLARATION)                 # the declaration must exist first
    v3 = rw.verify_v3()
    t = rw.eligibility(rw.station_table())
    s = station_set(t)
    members = {UNIT: {"description": "fan monitoring stations whose agency GroundHeight is "
                                     "a real value consistent with WellElevation (rules E1, "
                                     "E2, E3, E5, E6 of w3_dtm/repair_wells.py, any "
                                     "establishment date up to 2022-12-31); members are "
                                     "station-metadata records, not head records",
                      "ids": [f"station_meta:{x}" for x in sorted(s["sid"])]}}
    look = prereg3.record_look(
        [UNIT], SCOPE, "W3", "after_the_fact_analysis", members=members,
        note=f"declared in {DECLARATION}; residuals under both survey-epoch assumptions, "
             "uncorrected and cell-mean variants; never gating, never fed back; these "
             "stations were already looked at by W3 diagnostics (2026-10-12) and H1b "
             "(2026-10-13)")
    dm, info = rw.displacement_model()
    r = rw.well_residuals(s, dm)
    cells = pd.read_parquet(rw.V3 / "ground_elevation_cells.parquet",
                            columns=["cell", "z_sheet_mean"]).set_index("cell")["z_sheet_mean"]
    inc = r["cell"] >= 0
    zc = np.full(len(r), np.nan)
    zc[inc.to_numpy()] = cells.reindex(r.loc[inc, "cell"]).to_numpy()
    r["r_cell_A"] = zc + r["corr_establish"] - r["gh_used"]
    r["r_cell_B"] = zc + r["corr_modification"] - r["gh_used"]
    x0, y0 = rw.grid_origin()
    res = {}
    for lab, col in (("A_survey_at_establishment", "r_primary"),
                     ("B_survey_at_modification", "r_modification_epoch"),
                     ("uncorrected", "r_uncorrected"),
                     ("cell_mean_A", "r_cell_A"), ("cell_mean_B", "r_cell_B")):
        sv = rw.site_values(r, col)
        res[lab] = {"sites": rw.summarize(sv["r"]), "wells": rw.summarize(r[col]),
                    "block_bootstrap_90_sites": rw.block_bootstrap(sv, x0, y0)
                    if len(sv) >= 2 else None}
    strata = {}
    for lab, sel in (("changhua_sid07", r["sid"].str.startswith("07")),
                     ("yunlin_sid09", r["sid"].str.startswith("09")),
                     ("other_counties", ~r["sid"].str[:2].isin(["07", "09"]))):
        d = r[sel.to_numpy()]
        strata[lab] = {k: rw.summarize(rw.site_values(d, c)["r"])
                       for k, c in (("A", "r_primary"), ("B", "r_modification_epoch"))}
    post = r[r["E4_establish"]][["sid", "EstablishDate", "ModificationDate", "r_primary",
                                 "r_modification_epoch", "r_uncorrected", "corr_establish"]]
    rec = {
        "label": "after-the-fact analysis (declared in W3r.corrected_test_design.v1); never "
                 "gating, never fed back into ground_elevation; the survey epoch of agency "
                 "GroundHeight is undocumented, so neither assumption can be chosen",
        "look": {k: look.get(k) for k in ("time", "wp", "purpose", "data_scope", "note")},
        "v3_sha256_verified": v3,
        "displacement_model": info,
        "n_wells": int(len(r)), "n_sites": int(r["site"].nunique()),
        "residuals": res, "strata_sites": strata,
        "epoch_correction_A_m": rw.summarize(r["corr_establish"]),
        "epoch_correction_B_m": rw.summarize(r["corr_modification"]),
        "post2005_wells": post.round(4).to_dict(orient="records"),
        "sign": "r = DTM (moved to the survey epoch) minus agency GroundHeight; negative = "
                "DTM below the agency ground",
    }
    keep = ["sid", "site", "x", "y", "model_well", "cell", "EstablishDate",
            "ModificationDate", "t_establish", "t_modification", "dtm_point_sheet_m",
            "dtm_point_epoch", "dtm_local_sd_m", "gh_used", "corr_establish",
            "corr_modification", "r_primary", "r_modification_epoch", "r_uncorrected",
            "r_cell_A", "r_cell_B"]
    r[keep].to_parquet(out / "aftfact_station_residuals.parquet", index=False)
    (out / "aftfact.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False,
                                                 default=float) + "\n", encoding="utf-8")
    print(json.dumps(rec, indent=1, ensure_ascii=False, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
