"""Independent re-verification of heads/L1_station_ground v1 and v2 (WP H1b).

    cpu_slot.sh python -I h1b_verify.py

Checks (written to results/dataprep/reports/H1b_evidence/verify.json):

1. Rebuild: v1's compute (registered code, unchanged) and v2's compute rerun in memory equal
   the published tables (every column, NaN-equal).
2. Independent bilinear: the DTM height at the station points from the published mosaic with
   scipy's RegularGridInterpolator (not W3's Mosaic.bilinear) equals dtm_point_sheet_m.
3. Independent leave-one-block-out: hits and coverage of v1's T1 and of v2's post-look check
   recomputed with a separate loop (explicit rank arithmetic, no h1b_lib calibration code).
4. The registered fresh-unit members equal v1's calibration stations.
5. The v2 rule (stick-up > 5 m) recomputed from H1a v3's station table selects the same 11
   stations, and no consistent station has a stick-up in (3.6, 6.8) m.
6. Every flag-5 value of both versions equals the DTM well point; flag-0 values equal the
   agency value; the two versions differ only at the 11 stations plus the sd/tau columns.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(1, str(HERE.parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import guard  # noqa: E402
from common.paths import DATAPREP, PREREG_DIR  # noqa: E402

from h1_heads_qc import h1b_build as b1  # noqa: E402
from h1_heads_qc import h1b_lib as lib  # noqa: E402
from h1_heads_qc import h1b_v2 as b2  # noqa: E402

PROD = DATAPREP / lib.PRODUCT


def frames_equal(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    out = {"same_columns": list(a.columns) == list(b.columns), "rows": [len(a), len(b)],
           "differing_columns": []}
    for c in a.columns:
        if c not in b.columns:
            out["differing_columns"].append(c)
            continue
        x, y = a[c].reset_index(drop=True), b[c].reset_index(drop=True)
        if pd.api.types.is_float_dtype(x) and pd.api.types.is_float_dtype(y):
            eq = np.array_equal(x.to_numpy(), y.to_numpy(), equal_nan=True)
        else:
            eq = x.astype(str).equals(y.astype(str))
        if not eq:
            out["differing_columns"].append(c)
    out["equal"] = out["same_columns"] and not out["differing_columns"] and len(a) == len(b)
    return out


def lobo_independent(r, s, blocks, z=1.6448536269514722, level=0.90):
    r, s, blocks = np.asarray(r, float), np.asarray(s, float), np.asarray(blocks)
    hits = np.zeros(len(r), bool)
    for b in np.unique(blocks):
        tr = blocks != b
        need = []
        for ri, si in zip(r[tr], s[tr], strict=True):
            q = (abs(ri) / z) ** 2 - si ** 2
            need.append(math.sqrt(q) if q > 0 else 0.0)
        need.sort()
        n = len(need)
        k = min(math.ceil((n + 1) * level), n)
        tau = need[k - 1]
        for i in np.flatnonzero(~tr):
            hits[i] = abs(r[i]) <= z * math.sqrt(s[i] ** 2 + tau ** 2)
    return hits


def main() -> int:
    from harness import prereg3
    from scipy.interpolate import RegularGridInterpolator
    lk = prereg3.record_look(
        [lib.FRESH_UNIT], data_scope="station_metadata_GroundHeight+w3_dtm_well_points",
        wp=lib.WP, purpose="reproduce",
        note="h1b_verify.py: rebuild of v1 (registered code) and v2 in memory, independent "
             "bilinear and leave-one-block-out implementations, registered member set, v2 rule")
    out: dict = {"time_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                 "look_time": lk.get("time") if isinstance(lk, dict) else None}
    t1 = guard.open_product(PROD / "v1", "after_the_fact_analysis")
    t2 = guard.open_product(PROD / "v2", "after_the_fact_analysis")
    pub1 = pd.read_parquet(t1.path("station_ground.parquet"))   # all rows, flag 8 included
    pub2 = pd.read_parquet(t2.path("station_ground.parquet"))

    # 1. rebuild
    r1 = b1.compute(False, "v1")
    r2 = b2.compute_v2("v2")
    out["rebuild_v1"] = frames_equal(r1["tab"], pub1)
    out["rebuild_v2"] = frames_equal(r2["tab"], pub2)

    # 2. independent bilinear on the published mosaic
    w3 = guard.open_product(lib.W3_DIR, "design_time_evidence")
    with np.load(w3.path(lib.W3_MOSAIC), allow_pickle=False) as z:
        H = z["H"].astype("float64")
        ext = z["extent"]
        step = float(z["step"])
    e = ext[0] + step * np.arange(H.shape[1])
    n = ext[2] + step * np.arange(H.shape[0])
    f = RegularGridInterpolator((n, e), H, method="linear", bounds_error=False,
                                fill_value=np.nan)
    inp = lib.open_inputs()
    meta = inp["meta"].reset_index(drop=True)
    pts = np.column_stack([meta["y_twd97"].to_numpy(float), meta["x_twd97"].to_numpy(float)])
    ok = np.isfinite(pts).all(axis=1)
    hb = np.full(len(meta), np.nan)
    hb[ok] = f(pts[ok])
    dtm = lib.dtm_well_points(meta["x_twd97"].to_numpy(), meta["y_twd97"].to_numpy(),
                              inp["mosaic"], lib.displacement_model(), r1["sd_pt_mad"],
                              r1["a_epoch"])
    mine = dtm["dtm_point_sheet_m"].to_numpy()
    both = np.isfinite(hb) & np.isfinite(mine)
    out["independent_bilinear"] = {
        "n_compared": int(both.sum()),
        "max_abs_diff_m": float(np.max(np.abs(hb[both] - mine[both]))) if both.any() else None,
        "finite_only_scipy": int((np.isfinite(hb) & ~np.isfinite(mine)).sum()),
        "finite_only_w3": int((~np.isfinite(hb) & np.isfinite(mine)).sum())}

    # 3. independent leave-one-block-out (v1 registered T1 and v2 post-look)
    from w3_dtm import validation as w3val
    g = lib.model_grid()
    for name, pub, res in (("v1", pub1, r1), ("v2", pub2, r2)):
        cal = pub["sid"].isin(set(res["cal_sids"])).to_numpy()
        x = meta.set_index("sid").loc[pub["sid"], "x_twd97"].to_numpy(float)
        y = meta.set_index("sid").loc[pub["sid"], "y_twd97"].to_numpy(float)
        blocks = w3val.block_ids(x[cal], y[cal], g["x0"], g["y0"])
        r = (pub["raw_ground_height_m"] - pub["dtm_point_ref_m"]).to_numpy(float)[cal]
        s = pub["dtm_sd_w3_m"].to_numpy(float)[cal]
        hits = lobo_independent(r, s, blocks)
        cov = hits.mean()
        out[f"independent_lobo_{name}"] = {
            "n": int(cal.sum()), "k_blocks": int(len(np.unique(blocks))),
            "coverage": float(cov), "reported": res["cov"]["coverage"],
            "equal": bool(abs(cov - res["cov"]["coverage"]) < 1e-12)}

    # 4. registered members = v1 calibration stations
    doc = json.loads((PREREG_DIR / f"{lib.PREREG_ID}.json").read_text(encoding="utf-8"))
    members = sorted(x[len(lib.NS):] for x in
                     doc["tests"][0]["external_units"][lib.FRESH_UNIT]["ids"])
    out["registered_members_equal_v1_calibration"] = {
        "n_registered": len(members), "n_v1": len(r1["cal_sids"]),
        "equal": members == sorted(r1["cal_sids"])}

    # 5. v2 rule from H1a's table
    gh = pd.to_numeric(meta["GroundHeight"], errors="coerce")
    we = pd.to_numeric(meta["WellElevation"], errors="coerce")
    stick = we - gh
    real = gh.notna() & (gh != 0)
    sel = sorted(meta.loc[real & (stick > 5.0), "sid"])
    gap = meta.loc[real & (stick > 3.6) & (stick < 6.8), "sid"].tolist()
    out["v2_rule"] = {"selected": sel, "n": len(sel),
                      "equals_v2": sel == sorted(r2["reported"]["inconsistent_stations"]["sids"]),
                      "stations_in_gap_3.6_6.8": gap,
                      "max_consistent_stickup_m": float(stick[real & (stick <= 5.0)].max())}

    # 6. value rules and v1-v2 differences
    for name, pub in (("v1", pub1), ("v2", pub2)):
        f5 = pub["flag"] == 5
        f0 = pub["flag"] == 0
        out[f"value_rules_{name}"] = {
            "flag5_equal_dtm": bool(np.array_equal(pub.loc[f5, "value"],
                                                   pub.loc[f5, "dtm_point_ref_m"])),
            "flag0_equal_agency": bool(np.array_equal(pub.loc[f0, "value"],
                                                      pub.loc[f0, "raw_ground_height_m"])),
            "flags": {int(k): int(v) for k, v in pub["flag"].value_counts().items()}}
    dv = pub1["value"].to_numpy() != pub2["value"].to_numpy()
    dv &= ~(pub1["value"].isna() & pub2["value"].isna()).to_numpy()
    out["v1_v2_value_differences"] = {"n": int(dv.sum()),
                                      "sids": pub1.loc[dv, "sid"].tolist(),
                                      "equal_to_v2_rule": sorted(pub1.loc[dv, "sid"]) == sel}
    out["all_pass"] = bool(out["rebuild_v1"]["equal"] and out["rebuild_v2"]["equal"]
                           and out["independent_bilinear"]["max_abs_diff_m"] is not None
                           and out["independent_bilinear"]["max_abs_diff_m"] < 1e-6
                           and out["independent_bilinear"]["finite_only_scipy"] == 0
                           and out["independent_bilinear"]["finite_only_w3"] == 0
                           and out["independent_lobo_v1"]["equal"]
                           and out["independent_lobo_v2"]["equal"]
                           and out["registered_members_equal_v1_calibration"]["equal"]
                           and out["v2_rule"]["equals_v2"] and not out["v2_rule"]
                           ["stations_in_gap_3.6_6.8"]
                           and all(out[f"value_rules_{v}"]["flag5_equal_dtm"]
                                   and out[f"value_rules_{v}"]["flag0_equal_agency"]
                                   for v in ("v1", "v2"))
                           and out["v1_v2_value_differences"]["equal_to_v2_rule"])
    b1.EVIDENCE.mkdir(parents=True, exist_ok=True)
    (b1.EVIDENCE / "verify.json").write_text(json.dumps(out, indent=1, ensure_ascii=False,
                                                        default=float) + "\n")
    print(json.dumps(out, indent=1, ensure_ascii=False, default=float)[:4000], flush=True)
    return 0 if out["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
