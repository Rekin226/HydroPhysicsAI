"""Post-look reviewed build ``heads/L1_station_ground/v2`` (WP H1b; v1 is the registered build).

After the registered build v1 (pre-registration ``H1b.ground_height.v1``: T1 and C1-C4 pass)
an after-the-fact look at the stations with |agency GroundHeight - DTM| > 5 m showed that 11
agency GroundHeight values are inconsistent with the agency's own WellElevation: the implied
casing stick-up ``WellElevation - GroundHeight`` is 6.84-29.39 m, while the other 185 valid
stations with a WellElevation have at most 3.53 m. Nine of the 11 store GroundHeight 0.80 m and
one 0.99 m (a second placeholder, probably the casing height typed into the ground field).

v2 changes exactly two things, both post-look and labelled as such:

1. A real-looking agency GroundHeight whose stick-up exceeds ``STICKUP_MAX_M`` (5 m, inside the
   observed 3.53-6.84 m gap; a metadata-only rule, the DTM is not used to decide) is treated
   like the 0.0 code: raw value kept, ``raw_status = 'inconsistent_with_wellelevation'``,
   replaced by the DTM well point (flag 5).
2. tau is refitted with the registered rule (split-conformal 90 %) on the remaining consistent
   calibration stations; its leave-one-10 km-block-out coverage is reported as a post-look
   check (not a registered test; every station was seen by v1's scoring).

Everything else (inputs, DTM recomputation, C1-C4) is v1's code, imported unchanged. The gate
stays closed: W3's gate failed and no user decision is recorded.

    cpu_slot.sh python -I h1b_v2.py
"""
from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(1, str(HERE.parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import guard, hashing, manifest  # noqa: E402
from common.paths import DATAPREP, PREREG_DIR, TWIN_GRID_NPZ, rel_to_repo  # noqa: E402

from h1_heads_qc import h1b_build as b1  # noqa: E402
from h1_heads_qc import h1b_lib as lib  # noqa: E402

STICKUP_MAX_M = 5.0
V1_DIR = DATAPREP / lib.PRODUCT / "v1"
CODE = [HERE, *b1.CODE]
log = b1.log


def inconsistent(meta: pd.DataFrame) -> np.ndarray:
    """Agency GroundHeight real-looking (finite, not 0.0) but more than STICKUP_MAX_M below the
    agency's own WellElevation."""
    gh = pd.to_numeric(meta["GroundHeight"], errors="coerce").to_numpy(dtype="float64")
    we = pd.to_numeric(meta["WellElevation"], errors="coerce").to_numpy(dtype="float64")
    return np.isfinite(gh) & (gh != 0.0) & np.isfinite(we) & ((we - gh) > STICKUP_MAX_M)


def compute_v2(version_name: str) -> dict:
    from harness import coverage3
    from w3_dtm import validation as w3val
    g = lib.model_grid()
    mw = set(g["model_wells"])
    inp = lib.open_inputs()
    meta = inp["meta"].reset_index(drop=True)
    w3m = inp["w3_manifest"]
    sd_pt_mad = float(w3m["parameters"]["point_sd_mad_m"])
    a_epoch = float(w3m["parameters"]["epoch_error_a_m_per_yr"])
    dm = lib.displacement_model()
    dtm = lib.dtm_well_points(meta["x_twd97"].to_numpy(), meta["y_twd97"].to_numpy(),
                              inp["mosaic"], dm, sd_pt_mad, a_epoch)
    c2 = lib.reproduction(meta, dtm, inp["w3"])
    bad = inconsistent(meta)
    log(f"inconsistent agency GroundHeight (stick-up > {STICKUP_MAX_M} m): {int(bad.sum())}")

    # calibration on consistent stations (registered rule, post-look station set)
    work = meta.join(dtm)
    work["raw_status"] = lib.raw_status(work["GroundHeight"])
    work.loc[bad, "raw_status"] = "inconsistent_with_wellelevation"
    cal = lib.calibration_mask(work)
    x = work["x_twd97"].to_numpy(dtype="float64")
    y = work["y_twd97"].to_numpy(dtype="float64")
    gh = pd.to_numeric(work["GroundHeight"], errors="coerce").to_numpy(dtype="float64")
    z_all = work["ground_elev_dtm_ref_m"].to_numpy(dtype="float64")
    s_w3 = work["sd_m"].to_numpy(dtype="float64")
    blocks = w3val.block_ids(x[cal], y[cal], g["x0"], g["y0"])
    r = gh[cal] - z_all[cal]
    s = s_w3[cal]
    hits, blk = lib.lobo_coverage(r, s, blocks)
    cov = coverage3.coverage_gate(hits.astype(int), blocks)
    tau = lib.conformal_tau(lib.tau_needed(r, s))
    ok = bool(cov["pass"] and cov["calibrated"])
    log(f"post-look coverage: n={cov['n']} k={cov['k_clusters']} {cov['coverage']:.3f} "
        f"[{cov['lo']:.3f}, {cov['hi']:.3f}] pass={cov['pass']} calibrated={cov['calibrated']};"
        f" tau={tau:.3f} m")
    sd_status = ("calibrated post-look (H1b v2 review: leave-one-10km-block-out coverage of "
                 f"consistent agency GroundHeight {cov['coverage']:.3f}, rule gate-3; not a "
                 "registered test)" if ok else
                 "uncalibrated (H1b v2 post-look coverage "
                 f"{cov['coverage']:.3f}, pass={cov['pass']}, calibrated={cov['calibrated']})")

    tab = lib.assemble(meta, dtm, mw, tau, sd_status, f"{lib.PRODUCT}/{version_name}")
    # the inconsistent stations: replaced like the 0.0 code (raw value kept)
    idx = np.flatnonzero(bad)
    zb = tab.loc[idx, "dtm_point_ref_m"].to_numpy(dtype="float64")
    if not np.isfinite(zb).all():
        raise SystemExit("an inconsistent station has no DTM value")
    tab.loc[idx, "raw_status"] = "inconsistent_with_wellelevation"
    tab.loc[idx, "value"] = zb
    tab.loc[idx, "sd"] = tab.loc[idx, "dtm_sd_cal_m"]
    tab.loc[idx, "flag"] = np.uint8(5)
    tab.loc[idx, "method"] = "w3_dtm20m_point_2017.5_replaces_inconsistent_agency_value"
    tab.loc[idx, "epoch"] = lib.T_REF
    tab.loc[idx, "sd_status"] = sd_status
    tab.loc[idx, "source_ids"] = ("ground_elevation/v3 (dtm20m_mosaic, leveling epoch "
                                  "correction) at heads/L1_clean/v3 coordinates")
    tab.loc[idx, "well_bottom_elev_m"] = tab.loc[idx, "value"] - tab.loc[idx, "well_depth_m"]
    tab["flag"] = tab["flag"].astype(np.uint8)
    tab["agency_stickup_m"] = (tab["well_elevation_m"] - tab["raw_ground_height_m"])

    c1 = lib.completeness(tab)
    c3 = lib.integrity(tab, meta)
    c3["note"] = ("agency_values_unchanged is checked on the flag-0 rows only; the "
                  f"{int(bad.sum())} inconsistent stations are flag 5 with their raw value kept")
    log(f"C1 pass={c1['pass']} ({c1['replaced_flag5']} replaced); C3 pass={c3['pass']}")

    rep: dict = {}
    rep["inconsistent_stations"] = {
        "n": int(bad.sum()), "rule": f"WellElevation - GroundHeight > {STICKUP_MAX_M} m",
        "sids": tab.loc[idx, "sid"].tolist(),
        "raw_ground_height_m": [float(v) for v in tab.loc[idx, "raw_ground_height_m"]],
        "stickup_m": [float(v) for v in tab.loc[idx, "agency_stickup_m"]],
        "agency_minus_dtm_m": [float(v) for v in tab.loc[idx, "agency_minus_dtm_m"]],
        "model_wells": int(tab.loc[idx, "model_well"].sum())}
    st_ok = (tab["raw_status"] == "valid") & np.isfinite(tab["agency_stickup_m"])
    rep["stickup_consistent_valid"] = lib.describe(tab.loc[st_ok, "agency_stickup_m"])
    rep["residual_agency_minus_dtm_calibration_set"] = lib.describe(r)
    rep["w3_sd_as_published_coverage"] = coverage3.coverage_gate(
        (np.abs(r) <= lib.Z90 * s).astype(int), blocks)
    rep["tau_all_m"] = tau
    rep["tau_lobo_m"] = lib.describe(blk["tau_train_m"])
    f5 = (tab["flag"] == 5).to_numpy()
    rep["sd_cal_flag5"] = lib.describe(tab.loc[f5, "sd"])
    steep = f5 & (tab["dtm_local_sd_m"].to_numpy() > 5.0)
    rep["flag5_steep_terrain_local_sd_gt_5m"] = {"n": int(steep.sum()),
                                                 "sids": tab.loc[steep, "sid"].tolist()}
    we = tab["well_elevation_m"].to_numpy(dtype="float64")
    stick_cal = we[cal] - gh[cal]
    s_med, s_mad = float(np.nanmedian(stick_cal)), lib.mad_sd(stick_cal)
    d_proxy = (we[f5] - s_med) - tab.loc[f5, "value"].to_numpy(dtype="float64")
    half = lib.Z90 * np.sqrt(tab.loc[f5, "sd"].to_numpy() ** 2 + s_mad ** 2)
    rep["proxy_minus_dtm_replaced"] = lib.describe(d_proxy)
    rep["proxy_within_interval_share"] = float(np.mean(np.abs(d_proxy) <= half))
    rep["proxy_stickup_median_m"] = s_med
    rep["proxy_stickup_mad_sd_m"] = s_mad
    gross = tab[tab["agency_dtm_gross_discrepancy"] & (tab["flag"] == 0)]
    rep["remaining_gross_discrepancies_flag0"] = {
        "n": int(len(gross)), "sids": gross["sid"].tolist(),
        "residuals_m": [float(v) for v in gross["agency_minus_dtm_m"]],
        "dtm_local_sd_m": [float(v) for v in gross["dtm_local_sd_m"]]}
    m0 = tab["model_well"] & (tab["flag"] == 5)
    rep["model_wells_flag5"] = int(m0.sum())
    rep["model_wells_new_value_vs_wells_mode_field"] = {
        "zero_code": int((m0 & (tab["raw_status"] == "zero_missing_code")).sum()),
        "missing_nan": int((m0 & (tab["raw_status"] == "missing_nan")).sum()),
        "inconsistent": int((m0 & (tab["raw_status"] == "inconsistent_with_wellelevation"))
                            .sum())}
    return {"tab": tab, "blk": blk, "c1": c1, "c2": c2, "c3": c3, "cov": cov, "post_look_ok": ok,
            "tau": tau, "sd_status": sd_status, "reported": rep, "notes": inp["notes"],
            "w3_gate": w3m["gate"], "w3_probe": lib.w3_policy_probe(), "n_cal": int(cal.sum()),
            "n_blocks": int(len(blk)), "sd_pt_mad": sd_pt_mad, "a_epoch": a_epoch,
            "cal_sids": sorted(work.loc[cal, "sid"].astype(str).tolist())}


def card_v2(res: dict, gate: dict, version_name: str) -> str:
    c1, cov, rep = res["c1"], res["cov"], res["reported"]
    rr = rep["residual_agency_minus_dtm_calibration_set"]
    inc = rep["inconsistent_stations"]
    return f"""# heads/L1_station_ground/{version_name} (WP H1b, post-look reviewed build)

One ground elevation per monitoring station of heads/L1_clean/v3 (344 rows; key `sid`).
v1 is the registered build (pre-registration {lib.PREREG_ID}: T1 and C1-C4 passed). This v2
adds one post-look metadata rule and refits tau on the consistent stations; both changes were
designed after v1's scores were seen.

- **flag 0**: agency `GroundHeight` as stored ({res['c3']['n_flag0']} stations). No sd.
- **flag 5** ({c1['replaced_flag5']} stations, incl. all {c1['model_wells_zero']} model wells whose
  GroundHeight is 0.0): W3 DTM well point at 2017.5 (bilinear 20 m MOI DTM, leveling epoch
  correction; ground_elevation/v3, recomputed at the H1a v3 coordinates) where the agency value
  is the 0.0 code, missing, or **inconsistent with the agency's own WellElevation** (stick-up
  WellElevation - GroundHeight > {STICKUP_MAX_M:.0f} m: {inc['n']} stations, {inc['model_wells']}
  of them model wells; 9 store 0.80 m and one 0.99 m). `raw_ground_height_m` keeps the agency
  value; `rate_2012_2021_m_per_yr` shifts the 2017.5 value to other epochs (uncalibrated).
- **flag 8**: no value ({c1['flag8_not_replaced']} stations without coordinates).
- `sd` (flag 5) = sqrt(sd_w3^2 + tau^2), tau = {res['tau']:.3f} m (split-conformal 90 %, the
  registered rule, on {res['n_cal']} consistent stations). {res['sd_status']}. Leave-one-10 km-
  block-out coverage {cov['coverage']:.3f} (Wilson {cov['lo']:.3f}..{cov['hi']:.3f},
  {cov['k_clusters']} blocks). v1's registered tau (1.508 m) was inflated by the 11
  inconsistent agency values. The sd is calibrated against agency GroundHeight
  (construction-era surveys), so it includes the agency-vs-DTM epoch and datum mismatch.
- Agency minus DTM at the consistent calibration stations: median {rr['median']:+.3f} m,
  MAD-sd {rr['mad_sd']:.3f} m (n {rr['n']}). Flag-5 values are not shifted by this offset.
- Steep terrain: {rep['flag5_steep_terrain_local_sd_gt_5m']['n']} flag-5 stations have a DTM
  local sd > 5 m; the remaining flag-0 gross discrepancies are
  {rep['remaining_gross_discrepancies_flag0']['sids']} (DTM local sd 15.7 m: the DTM, not the
  agency value, is off there).
- `well_bottom_elev_m` = value - WellDepth (assumes WellDepth from ground level);
  `agency_stickup_m` = WellElevation - raw GroundHeight.

**Gate: {'passed' if gate['passed'] else 'NOT passed'}.** {gate['criteria']}.
The guard grants `design_time_evidence` and `after_the_fact_analysis` only. No head data was
read. fit_window_end 2022-12.
"""


def main() -> int:
    v1m, _ = manifest.load_manifest(V1_DIR)
    v1val = json.loads((V1_DIR / "validation.json").read_text(encoding="utf-8"))
    version_name = b1.next_version_name(lib.PRODUCT)
    res = compute_v2(version_name)
    own_v1 = bool(v1val["h1b_verdict"])
    own = bool(own_v1 and res["post_look_ok"] and res["c1"]["pass"] and res["c2"]["pass"]
               and res["c3"]["pass"])
    w3_ok = bool(res["w3_gate"].get("passed", False))
    crit = (f"registered H1b verdict on v1 ({lib.PREREG_ID}): "
            f"{'pass' if own_v1 else 'FAIL'}; this post-look reviewed build: coverage "
            f"{res['cov']['coverage']:.3f} ({'pass' if res['post_look_ok'] else 'FAIL'}, not "
            f"registered), C1 {'pass' if res['c1']['pass'] else 'FAIL'}, C2 "
            f"{'pass' if res['c2']['pass'] else 'FAIL'}, C3 "
            f"{'pass' if res['c3']['pass'] else 'FAIL'}; upstream W3 ground_elevation/v3 gate "
            f"{'passed' if w3_ok else 'FAILED'} and no user decision on it is recorded, so the "
            f"DTM-derived values stay closed to gated uses")
    gate = {"passed": bool(own and w3_ok), "criteria": crit, "own_criteria_passed": own,
            "post_look": True, "registered_build": "v1",
            "upstream": {"ground_elevation/v3": res["w3_gate"]}}
    policy = {"product": lib.PRODUCT, "version": version_name, "fit_window_end": "2022-12",
              "allowed_uses": lib.ALLOWED_USES, "forbidden_uses": lib.FORBIDDEN_USES,
              "gate": gate}
    c4 = lib.policy_probe(policy)
    if not c4["pass"]:
        gate["passed"] = False
    res["c4"] = c4
    log(f"C4 pass={c4['pass']}; own={own}; gate passed={gate['passed']}")
    val = {
        "wp": lib.WP, "product": lib.PRODUCT, "version": version_name,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "post_look_reviewed_build": True, "registered_build": "v1",
        "registered_verdict_v1": {"h1b_verdict": own_v1,
                                  "T1_coverage": v1val["registered"]["T1_sd_coverage_lobo"]
                                  ["gate"]["coverage"],
                                  "tau_v1_m": v1val["registered"]["T1_sd_coverage_lobo"]
                                  ["tau_all_m"],
                                  "v1_manifest_sha256": hashing.sha256_file(
                                      V1_DIR / "MANIFEST.json")},
        "post_look": {"sd_coverage_lobo": {"pass": res["post_look_ok"], "gate": res["cov"],
                                           "n_stations": res["n_cal"],
                                           "n_blocks": res["n_blocks"], "tau_all_m": res["tau"]},
                      "rule": f"agency GroundHeight inconsistent when WellElevation - "
                              f"GroundHeight > {STICKUP_MAX_M} m"},
        "conformance": {"C1_completeness": res["c1"], "C2_reproduction": res["c2"],
                        "C3_integrity": res["c3"], "C4_use_policy": c4},
        "gate": gate, "upstream_w3_policy": res["w3_probe"],
        "reported_not_gated": res["reported"], "inputs_opened": res["notes"],
        "parameters": {"t_ref": lib.T_REF, "level": lib.LEVEL, "z90": lib.Z90,
                       "est_cutoff": lib.EST_CUTOFF, "gross_m": lib.GROSS_M,
                       "stickup_max_m": STICKUP_MAX_M,
                       "seen_before_registration": list(lib.SEEN_BEFORE_REGISTRATION),
                       "w3_point_sd_mad_m": res["sd_pt_mad"],
                       "w3_epoch_error_a_m_per_yr": res["a_epoch"],
                       "block_m": 10000.0, "block_origin": "twin grid (x0, y0)"},
    }
    out = manifest.new_version_dir(lib.PRODUCT)
    if out.name != version_name:
        raise SystemExit(f"version directory {out.name} != expected {version_name}")
    res["tab"].to_parquet(out / "station_ground.parquet", index=False)
    res["blk"].to_csv(out / "calibration_blocks.csv", index=False)
    (out / "validation.json").write_text(json.dumps(val, indent=1, ensure_ascii=False,
                                                    default=float) + "\n")
    (out / "PRODUCT_CARD.md").write_text(card_v2(res, gate, version_name), encoding="utf-8")
    prereg_path = PREREG_DIR / f"{lib.PREREG_ID}.json"
    m = manifest.build_manifest(
        out, product=lib.PRODUCT, layer="L1",
        inputs=[str(lib.H1A_DIR / lib.H1A_TABLE), str(lib.H1A_DIR / "station_meta_clean.parquet"),
                str(lib.H1A_DIR / "MANIFEST.json"), str(lib.W3_DIR / lib.W3_MOSAIC),
                str(lib.W3_DIR / lib.W3_WELLS), str(lib.W3_DIR / "MANIFEST.json"),
                str(lib.LEVELING), str(TWIN_GRID_NPZ), str(prereg_path),
                str(V1_DIR / "MANIFEST.json"), str(V1_DIR / "validation.json")],
        code=[str(p) for p in CODE],
        parameters={**val["parameters"], "tau_m": res["tau"],
                    "calibration_rule": "registered rule of H1b.ground_height.v1 (split-"
                                        "conformal 90 %) on the consistent stations "
                                        "(post-look station set)",
                    "preregistration": {"id": lib.PREREG_ID, "path": rel_to_repo(prereg_path),
                                        "applies_to": "v1 (registered build)"}},
        fit_window_end="2022-12", validation_report="validation.json", gate=gate,
        allowed_uses=lib.ALLOWED_USES, forbidden_uses=lib.FORBIDDEN_USES,
        personal_data="none (public monitoring-station ids and elevations; no coordinates, "
                      "names, contact or address fields are copied)",
        extra={"product_card": "PRODUCT_CARD.md",
               "build_note": "post-look reviewed build: agency GroundHeight inconsistent with "
                             "WellElevation (stick-up > 5 m) replaced like the 0.0 code; tau "
                             "refitted on the consistent stations; v1 is the registered build",
               "flags_in_use": {"0": "agency GroundHeight as stored (real, consistent value)",
                                "5": "W3 DTM well point (2017.5) replacing GroundHeight 0.0, "
                                     "NaN or a value inconsistent with WellElevation",
                                "8": "no value (no coordinates)"},
               "release_condition": "a recorded user decision on the failed W3 gate (waiver or "
                                    "new test) and a new version of this product built from "
                                    "the released ground elevation"})
    for k in ("allowed_uses", "forbidden_uses", "fit_window_end", "gate"):
        if m[k] != policy[k]:
            raise SystemExit(f"manifest {k} differs from the probed policy")
    manifest.write_manifest(out, m)
    log(f"wrote {out}/MANIFEST.json (gate passed={gate['passed']})")
    probes = {}
    for use in lib.ALLOWED_USES + lib.FORBIDDEN_USES:
        try:
            p = guard.open_product(out, use)
            df = p.read_table("station_ground.parquet")
            probes[use] = {"result": "granted", "rows_readable": int(len(df)),
                           "flags_readable": sorted(int(f) for f in df["flag"].unique())}
        except PermissionError as exc:
            probes[use] = {"result": type(exc).__name__, "message": str(exc)[:300]}
    hp = manifest.verify_hashes(m, out)
    b1.EVIDENCE.mkdir(parents=True, exist_ok=True)
    (b1.EVIDENCE / f"guard_probe_{version_name}.json").write_text(json.dumps(
        {"product": rel_to_repo(out), "probes": probes, "verify_hashes_problems": hp,
         "time_utc": _dt.datetime.now(_dt.timezone.utc).isoformat()}, indent=1,
        ensure_ascii=False) + "\n")
    (b1.EVIDENCE / f"calibration_sids_{version_name}.json").write_text(
        json.dumps({"n": res["n_cal"], "sids": res["cal_sids"]}, indent=1) + "\n")
    log(json.dumps({u: v["result"] for u, v in probes.items()}))
    log(f"verify_hashes problems: {hp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
