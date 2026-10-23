"""WP S1b repair round r3 (S1br): after-the-fact diagnostics of the second look (logged look).

    python -I dataprep_staging/s1_rings/s1b/s1b_r3_diag_after.py

Reads only evidence already computed: ``repair_r3/retest/second_look.json`` (this round's rows,
variants and decompositions) and round 1's ``registered_rows.csv``, plus the ring-site
coordinates. No product is opened and nothing new is scored. Reports:

1. InSAR per site: gap change I_gap, bracket ends C_pre / C_post, excess rates and the primary
   value, this round (v3, M3 pixels) against round 1 (v2), and the change of each;
2. InSAR minus GNSS gap change at the co-located sites, this round and round 1, and the
   correlation of the InSAR change (v3 minus v2, M3 cells minus plane cells) with northing and
   easting (the reference term S4b documented);
3. GNSS per site: the bracket shift from v2 to v4 and which step change caused it (from the
   declared decomposition "rings_bridged/v3 with round-1 validators");
4. (d1) on the cell variants with the sites' coordinates.

Writes ``repair_r3/retest/diag_after.json``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from s1_rings.s1b import s1b_const as K  # noqa: E402
from s1_rings.s1b import s1b_io as IO1  # noqa: E402
from s1_rings.s1b.s1b_run import dump  # noqa: E402

RT = K.EVIDENCE / "repair_r3" / "retest"
KEYS = ("I_gap_mm", "C_pre_mm", "C_post_mm", "C_primary_mm", "C_raw_mm", "excess_rate_pre",
        "excess_rate_post", "rate_inst_pre", "rate_inst_post", "rate_ring_pre",
        "rate_ring_post", "sigma_pre_mm", "sigma_post_mm", "G_mm", "seasonal_mm")


def rows_by_site(rows: list[dict]) -> dict:
    return {r["site"]: r for r in rows}


def corr(a, b) -> float | None:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return None
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def main() -> int:
    from s1_rings.s1b import s1b_r3_looks as LK
    LK.after("insar_and_gnss_site_diagnostics")
    sl = json.loads((RT / "second_look.json").read_text())
    r1 = pd.read_csv(K.EVIDENCE / "registered_rows.csv")
    xy = IO1.site_xy()
    ins3 = rows_by_site(sl["primary"]["rows"]["d_insar"])
    insM = rows_by_site(sl["variants"]["insar_cells_M3"]["rows"]["d_insar"])
    insP = rows_by_site(sl["variants"]["insar_cells_plane"]["rows"]["d_insar"])
    ins1 = {r.site: r._asdict() for r in r1[r1.test == "d_insar"].itertuples()}
    g3 = rows_by_site(sl["primary"]["rows"]["c_wra"])
    g1 = {r.site: r._asdict() for r in r1[r1.test == "c_wra"].itertuples()}
    gdec = rows_by_site(sl["decomposition"]["v3_product_round1_validators"]["rows"]["c_wra"])
    gcat = rows_by_site(sl["variants"]["gnss_catalogue_steps_only_in_gap"]["rows"]["c_wra"])
    out = {"insar": [], "gnss": []}
    for s in K.SITES:
        a, b = ins3[s], ins1.get(s, {})
        row = {"site": s, "site_en": K.SITE_EN[s], "x": xy[s][0], "y": xy[s][1]}
        for k in KEYS:
            row[f"v3_{k}"] = a.get(k)
            row[f"r1_{k}"] = b.get(k)
        row["d_I_gap_mm"] = (a.get("I_gap_mm") or np.nan) - (b.get("I_gap_mm") or np.nan)
        row["cells_M3_I_gap_mm"] = insM[s].get("I_gap_mm")
        row["cells_plane_I_gap_mm"] = insP[s].get("I_gap_mm")
        row["plane_minus_M3_mm"] = (insP[s].get("I_gap_mm") or np.nan) - (
            insM[s].get("I_gap_mm") or np.nan)
        lo, hi = sorted([a.get("C_pre_mm") or np.nan, a.get("C_post_mm") or np.nan])
        row["v3_primary_outside_by_mm"] = (0.0 if lo <= a["C_primary_mm"] <= hi else
                                           (a["C_primary_mm"] - hi if a["C_primary_mm"] > hi
                                            else a["C_primary_mm"] - lo))
        if s in g3 and g3[s].get("evaluable"):
            row["insar_minus_gnss_gap_v3_mm"] = a["I_gap_mm"] - g3[s]["G_mm"]
        if s in g1:
            row["insar_minus_gnss_gap_r1_mm"] = b.get("I_gap_mm") - g1[s]["G_mm"]
        out["insar"].append(row)
    df = pd.DataFrame(out["insar"])
    out["insar_summary"] = {
        "corr_dIgap_northing": corr(df.d_I_gap_mm, df.y),
        "corr_dIgap_easting": corr(df.d_I_gap_mm, df.x),
        "corr_plane_minus_M3_northing": corr(df.plane_minus_M3_mm, df.y),
        "corr_plane_minus_M3_easting": corr(df.plane_minus_M3_mm, df.x),
        "median_d_I_gap_mm": float(np.nanmedian(df.d_I_gap_mm)),
        "range_d_I_gap_mm": [float(np.nanmin(df.d_I_gap_mm)), float(np.nanmax(df.d_I_gap_mm))],
        "median_v3_excess_post_minus_pre": float(np.nanmedian(df.v3_excess_rate_post
                                                              - df.v3_excess_rate_pre)),
        "median_r1_excess_post_minus_pre": float(np.nanmedian(df.r1_excess_rate_post
                                                              - df.r1_excess_rate_pre)),
        "median_v3_inst_rate_post": float(np.nanmedian(df.v3_rate_inst_post)),
        "median_r1_inst_rate_post": float(np.nanmedian(df.r1_rate_inst_post)),
        "median_v3_inst_rate_pre": float(np.nanmedian(df.v3_rate_inst_pre)),
        "median_r1_inst_rate_pre": float(np.nanmedian(df.r1_rate_inst_pre)),
        "insar_minus_gnss_v3": {"median": float(np.nanmedian(df.insar_minus_gnss_gap_v3_mm)),
                                "rms": float(np.sqrt(np.nanmean(
                                    df.insar_minus_gnss_gap_v3_mm ** 2)))},
        "insar_minus_gnss_r1": {"median": float(np.nanmedian(df.insar_minus_gnss_gap_r1_mm)),
                                "rms": float(np.sqrt(np.nanmean(
                                    df.insar_minus_gnss_gap_r1_mm ** 2)))},
    }
    for s, a in g3.items():
        if not a.get("evaluable"):
            continue
        d, c = gdec.get(s, {}), gcat.get(s, {})
        out["gnss"].append({
            "site_en": K.SITE_EN[s], "station": a["station"], "G_v4": a["G_mm"],
            "G_v2": d.get("G_mm"), "G_v4_catalogue_only": c.get("G_mm"),
            "C_pre_v4": a["C_pre_mm"], "C_post_v4": a["C_post_mm"],
            "C_pre_v2": d.get("C_pre_mm"), "C_post_v2": d.get("C_post_mm"),
            "C_pre_v4_cat": c.get("C_pre_mm"), "C_post_v4_cat": c.get("C_post_mm"),
            "exr_pre_v4": a["excess_rate_pre"], "exr_pre_v2": d.get("excess_rate_pre"),
            "exr_post_v4": a["excess_rate_post"], "exr_post_v2": d.get("excess_rate_post"),
            "C_primary": a["C_primary_mm"], "C_A": a["C_A_mm"], "C_B": a["C_B_mm"]})
    dump(out, RT / "diag_after.json")
    pd.set_option("display.width", 250)
    cols = ["site_en", "v3_I_gap_mm", "r1_I_gap_mm", "d_I_gap_mm", "cells_M3_I_gap_mm",
            "cells_plane_I_gap_mm", "v3_C_pre_mm", "v3_C_post_mm", "r1_C_pre_mm", "r1_C_post_mm",
            "v3_C_primary_mm", "v3_primary_outside_by_mm", "v3_excess_rate_pre",
            "v3_excess_rate_post", "r1_excess_rate_pre", "r1_excess_rate_post",
            "insar_minus_gnss_gap_v3_mm", "insar_minus_gnss_gap_r1_mm"]
    print(df[cols].round(1).to_string())
    print(json.dumps(out["insar_summary"], indent=1))
    print(pd.DataFrame(out["gnss"]).round(1).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
