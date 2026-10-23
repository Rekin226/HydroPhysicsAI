"""Build ``results/dataprep/heads/L1_station_ground/v<N>/`` (WP H1b, plan 3 WP H1 part b).

Usage (through the CPU-slot wrapper, with ``python -I``):

    cpu_slot.sh python -I h1b_build.py --dry OUTDIR   # development: the agency GroundHeight of
                                                      # every valid station is replaced by a
                                                      # synthetic value (seeded), so no real
                                                      # agency-vs-DTM residual is computed;
                                                      # no pre-registration, manifest or ledger
    cpu_slot.sh python -I h1b_build.py                # the product (needs H1b.ground_height.v1)

The product run first calls ``harness.prereg3.assert_can_score`` (refuses if the registered
inputs changed or a look predates the registration; logs the scoring look), computes
everything in memory, then creates the next version directory and writes the outputs, the
validation report, the product card and the manifest. Registered gate (H1b's own verdict):
T1 (leave-one-10 km-block-out coverage of the calibrated sd, rule gate-3, calibrated) and the
conformance checks C1-C4. The product gate additionally requires W3's gate (failed, no user
decision recorded): it stays closed until a decision on W3 is recorded and a new version is
built carrying it.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))           # dataprep_staging
sys.path.insert(1, str(HERE.parents[2]))           # repository (read-only imports)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import guard, manifest  # noqa: E402
from common.paths import DATAPREP, PREREG_DIR, REPO, TWIN_GRID_NPZ, rel_to_repo  # noqa: E402

from h1_heads_qc import h1b_lib as lib  # noqa: E402

DRY_SEED = 20261013
CODE = [HERE, HERE.parent / "h1b_lib.py",
        REPO / "dataprep_staging/w3_dtm/build.py", REPO / "dataprep_staging/w3_dtm/leveling.py",
        REPO / "dataprep_staging/w3_dtm/mosaic.py",
        REPO / "dataprep_staging/w3_dtm/validation.py",
        REPO / "dataprep_staging/w3_dtm/dtm_io.py",
        REPO / "dataprep_staging/harness/coverage3.py",
        REPO / "dataprep_staging/harness/coverage2.py",
        REPO / "dataprep_staging/harness/coverage.py"]
EVIDENCE = DATAPREP / "reports" / "H1b_evidence"


def log(msg: str) -> None:
    print(msg, flush=True)


def next_version_name(product: str) -> str:
    base = DATAPREP / product
    ns = [int(p.name[1:]) for p in base.iterdir()
          if p.is_dir() and re.fullmatch(r"v[1-9]\d*", p.name)] if base.is_dir() else []
    return f"v{max(ns, default=0) + 1}"


def _modified_after_origin(sids: list[str]) -> dict:
    """Reported only: calibration stations whose agency metadata row was modified on or after
    2023-01-01 (H1a v3 station_meta_clean, opened for after-the-fact analysis)."""
    p = guard.open_product(lib.H1A_DIR, "after_the_fact_analysis")
    m = p.read_table("station_meta_clean.parquet", columns=["sid", "ModificationDate"])
    m["sid"] = m["sid"].astype(str)
    late = m[m["sid"].isin(set(sids)) & (pd.to_numeric(m["ModificationDate"], errors="coerce")
                                          >= 20230101)]
    return {"n_calibration_stations": len(sids), "n_modified_on_or_after_2023_01_01":
            int(len(late)), "sids": sorted(late["sid"].tolist())}


def compute(dry: bool, version_name: str) -> dict:
    from harness import coverage3
    from w3_dtm import validation as w3val
    g = lib.model_grid()
    mw = set(g["model_wells"])
    inp = lib.open_inputs()
    meta = inp["meta"].reset_index(drop=True)
    w3m = inp["w3_manifest"]
    sd_pt_mad = float(w3m["parameters"]["point_sd_mad_m"])
    a_epoch = float(w3m["parameters"]["epoch_error_a_m_per_yr"])
    log(f"stations {len(meta)}; W3 point_sd_mad {sd_pt_mad:.4f} m, epoch a {a_epoch:.5f} m/yr")
    dm = lib.displacement_model()
    dtm = lib.dtm_well_points(meta["x_twd97"].to_numpy(), meta["y_twd97"].to_numpy(),
                              inp["mosaic"], dm, sd_pt_mad, a_epoch)
    c2 = lib.reproduction(meta, dtm, inp["w3"])
    log(f"C2 reproduction: pass={c2['pass']} max|diff|={c2['max_abs_diff_all']:.3g} "
        f"coords max {c2['coord_max_diff_m']}")

    meta_used = meta.copy()
    status = lib.raw_status(meta["GroundHeight"])
    z_all = dtm["ground_elev_dtm_ref_m"].to_numpy(dtype="float64")
    if dry:      # synthetic agency values at valid stations: no real residual is computed
        rng = np.random.default_rng(DRY_SEED)
        v = (status == "valid") & np.isfinite(z_all)
        meta_used.loc[v, "GroundHeight"] = z_all[v] + rng.normal(0.3, 0.7, int(v.sum()))
        log(f"DRY: synthetic GroundHeight at {int(v.sum())} valid stations (seed {DRY_SEED})")

    work = meta_used.join(dtm)
    work["raw_status"] = lib.raw_status(work["GroundHeight"])
    cal = lib.calibration_mask(work)
    x = work["x_twd97"].to_numpy(dtype="float64")
    y = work["y_twd97"].to_numpy(dtype="float64")
    gh = pd.to_numeric(work["GroundHeight"], errors="coerce").to_numpy(dtype="float64")
    s_w3 = work["sd_m"].to_numpy(dtype="float64")
    blocks = w3val.block_ids(x[cal], y[cal], g["x0"], g["y0"])
    r = gh[cal] - z_all[cal]
    s = s_w3[cal]
    hits, blk = lib.lobo_coverage(r, s, blocks)
    cov = coverage3.coverage_gate(hits.astype(int), blocks)
    need = lib.tau_needed(r, s)
    tau = lib.conformal_tau(need)
    t1_pass = bool(cov["pass"] and cov["calibrated"])
    log(f"T1: n={cov['n']} k={cov['k_clusters']} coverage={cov['coverage']:.3f} "
        f"[{cov['lo']:.3f}, {cov['hi']:.3f}] pass={cov['pass']} calibrated={cov['calibrated']}"
        f" -> {t1_pass}; tau(all)={tau:.3f} m")
    sd_status = ("calibrated (H1b.T1 pass: leave-one-10km-block-out coverage of agency "
                 f"GroundHeight {cov['coverage']:.3f}, rule gate-3)" if t1_pass else
                 "uncalibrated (H1b.T1 failed: leave-one-10km-block-out coverage "
                 f"{cov['coverage']:.3f}, rule gate-3 pass={cov['pass']}, "
                 f"calibrated={cov['calibrated']})")
    pv = f"{lib.PRODUCT}/{version_name}"
    tab = lib.assemble(meta_used, dtm, mw, tau, sd_status, pv)
    c1 = lib.completeness(tab)
    c3 = lib.integrity(tab, meta_used)
    log(f"C1 completeness pass={c1['pass']} ({c1['replaced_flag5']} replaced, "
        f"{c1['flag8_not_replaced']} flag 8); C3 integrity pass={c3['pass']}")

    # ---------------------------------------------------------------- reported (not gated)
    rep: dict = {}
    rep["residual_agency_minus_dtm_calibration_set"] = lib.describe(r)
    hits0 = np.abs(r) <= lib.Z90 * s
    rep["w3_sd_as_published_coverage"] = coverage3.coverage_gate(hits0.astype(int), blocks)
    cal_tab = tab[cal]
    by = {}
    for key in ("county", "model_well"):
        for val, idx in cal_tab.groupby(key).groups.items():
            ii = np.isin(cal_tab.index, idx)
            by[f"{key}={val}"] = {"n": int(ii.sum()), "coverage_lobo": float(hits[ii].mean()),
                                  "residual_median": float(np.median(r[ii]))}
    rep["coverage_lobo_by_stratum"] = by
    seen = (work["raw_status"].to_numpy() == "valid") & np.isfinite(z_all) & work["sid"].isin(
        lib.SEEN_BEFORE_REGISTRATION).to_numpy()
    rep["seen_stations_excluded"] = {"sids": sorted(work.loc[seen, "sid"].tolist()),
                                     "residuals_m": [float(v) for v in (gh - z_all)[seen]]}
    rep["tau_all_m"] = tau
    rep["tau_lobo_m"] = lib.describe(blk["tau_train_m"])
    rep["sd_cal_flag5"] = lib.describe(tab.loc[tab["flag"] == 5, "sd"])
    rep["sd_w3_flag5"] = lib.describe(tab.loc[tab["flag"] == 5, "dtm_sd_w3_m"])
    # stick-up proxy at the replaced stations (agency WellElevation minus the median stick-up)
    we = pd.to_numeric(work["WellElevation"], errors="coerce").to_numpy(dtype="float64")
    stick = we[cal] - gh[cal]
    s_med, s_mad = float(np.nanmedian(stick)), lib.mad_sd(stick)
    f5 = (tab["flag"] == 5).to_numpy()
    d_proxy = (we[f5] - s_med) - z_all[f5]
    half = lib.Z90 * np.sqrt(tab.loc[f5, "sd"].to_numpy() ** 2 + s_mad ** 2)
    rep["stickup_valid_stations"] = lib.describe(stick)
    rep["proxy_minus_dtm_replaced"] = lib.describe(d_proxy)
    rep["proxy_within_interval_share"] = float(np.mean(np.abs(d_proxy) <= half)) if f5.any() \
        else None
    from scipy import stats
    we_z_cal = we[cal] - z_all[cal]
    we_z_rep = we[f5] - z_all[f5]
    mwu = stats.mannwhitneyu(we_z_cal[np.isfinite(we_z_cal)], we_z_rep[np.isfinite(we_z_rep)])
    rep["exchangeability_wellelev_minus_dtm"] = {
        "calibration": lib.describe(we_z_cal), "replaced": lib.describe(we_z_rep),
        "mannwhitney_p": float(mwu.pvalue)}
    gross = tab[tab["agency_dtm_gross_discrepancy"]]
    rep["gross_discrepancies_gt_5m"] = {"n": int(len(gross)), "sids": gross["sid"].tolist(),
                                        "residuals_m": [float(v) for v in
                                                        gross["agency_minus_dtm_m"]]}
    m0 = tab["model_well"] & (tab["raw_status"] == "zero_missing_code")
    rep["model_wells_zero_new_value"] = lib.describe(tab.loc[m0, "value"])
    rep["replaced_value"] = lib.describe(tab.loc[tab["flag"] == 5, "value"])
    rep["replaced_after_origin_stations"] = int((tab["established_after_origin"]
                                                 & (tab["flag"] == 5)).sum())
    rep["well_bottom_elev_replaced"] = lib.describe(tab.loc[tab["flag"] == 5,
                                                            "well_bottom_elev_m"])
    rep["post_origin_metadata_modification"] = _modified_after_origin(
        work.loc[cal, "sid"].astype(str).tolist())
    return {"tab": tab, "blk": blk, "c1": c1, "c2": c2, "c3": c3, "cov": cov, "t1_pass": t1_pass,
            "tau": tau, "sd_status": sd_status, "reported": rep, "notes": inp["notes"],
            "w3_gate": w3m["gate"], "w3_probe": lib.w3_policy_probe(), "n_cal": int(cal.sum()),
            "n_blocks": int(len(blk)), "sd_pt_mad": sd_pt_mad, "a_epoch": a_epoch,
            "cal_sids": sorted(work.loc[cal, "sid"].astype(str).tolist())}


def gate_and_policy(res: dict, version_name: str) -> tuple[dict, dict, bool]:
    own = bool(res["t1_pass"] and res["c1"]["pass"] and res["c2"]["pass"] and res["c3"]["pass"])
    w3_ok = bool(res["w3_gate"].get("passed", False))
    passed = own and w3_ok
    crit = (f"H1b own criteria (pre-registration {lib.PREREG_ID}): T1 sd coverage "
            f"{'pass' if res['t1_pass'] else 'FAIL'} ({res['cov']['coverage']:.3f}), C1 "
            f"{'pass' if res['c1']['pass'] else 'FAIL'}, C2 "
            f"{'pass' if res['c2']['pass'] else 'FAIL'}, C3 "
            f"{'pass' if res['c3']['pass'] else 'FAIL'}; upstream W3 ground_elevation/v3 gate "
            f"{'passed' if w3_ok else 'FAILED'} ({res['w3_gate'].get('criteria', '')}) and no "
            f"user decision on it is recorded, so the DTM-derived values stay closed to gated "
            f"uses; release needs a recorded decision on W3 and a new version of this product")
    gate = {"passed": passed, "criteria": crit, "own_criteria_passed": own,
            "upstream": {"ground_elevation/v3": res["w3_gate"]}}
    policy = {"product": lib.PRODUCT, "version": version_name, "fit_window_end": "2022-12",
              "allowed_uses": lib.ALLOWED_USES, "forbidden_uses": lib.FORBIDDEN_USES,
              "gate": gate}
    c4 = lib.policy_probe(policy)
    if not c4["pass"]:
        gate["passed"] = False
    res["c4"] = c4
    verdict = own and c4["pass"]
    return gate, policy, verdict


def card(res: dict, gate: dict, version_name: str) -> str:
    c1, cov, rep = res["c1"], res["cov"], res["reported"]
    rr = rep["residual_agency_minus_dtm_calibration_set"]
    return f"""# heads/L1_station_ground/{version_name} (WP H1b)

One ground elevation per monitoring station of heads/L1_clean/v3 (344 rows; key `sid`).

- **flag 0**: agency `GroundHeight` as stored ({res['c3']['n_flag0']} stations). No sd.
- **flag 5**: agency value was the missing-value code 0.0 or NaN; replaced by the W3 DTM well
  point (bilinear 20 m MOI DTM, epoch-corrected to 2017.5 with WRA leveling; ground_elevation/v3)
  recomputed at the H1a v3 coordinates ({c1['replaced_flag5']} stations, including all
  {c1['model_wells_zero']} model wells whose GroundHeight is 0.0). `epoch` = 2017.5;
  `rate_2012_2021_m_per_yr` shifts it to other epochs (leveling mean rate, uncalibrated).
- **flag 8**: no value ({c1['flag8_not_replaced']} stations without coordinates).
- `sd` (flag 5) = sqrt(sd_w3^2 + tau^2), tau = {res['tau']:.3f} m (split-conformal, 90 %, on
  {res['n_cal']} stations with a real agency value). Status: {res['sd_status']}.
  Leave-one-10 km-block-out coverage {cov['coverage']:.3f} (Wilson {cov['lo']:.3f}..{cov['hi']:.3f},
  {cov['k_clusters']} blocks). The sd is calibrated against agency GroundHeight (construction-era
  surveys), so it includes the agency-vs-DTM epoch and datum mismatch.
- Agency minus DTM at the calibration stations: median {rr['median']:+.3f} m, MAD-sd
  {rr['mad_sd']:.3f} m (n {rr['n']}). The flag-5 values are not shifted by this offset (plan:
  replace by the DTM); mixing flag-0 and flag-5 rows mixes the two references.
- `well_bottom_elev_m` = value - WellDepth (assumes WellDepth is measured from ground level),
  for transducer-floor checks; `well_elevation_m` is the agency WellElevation.
- `agency_dtm_gross_discrepancy`: |agency - DTM| > {lib.GROSS_M:.0f} m (informational, flag
  unchanged; {rep['gross_discrepancies_gt_5m']['n']} stations).

**Gate: {'passed' if gate['passed'] else 'NOT passed'}.** {gate['criteria']}.
The loader guard grants `design_time_evidence` and `after_the_fact_analysis` only; `validator`
and `model_design` are declared but refused (GateNotPassed) until the gate opens. No head data
was read to build or validate this product. fit_window_end 2022-12 (station metadata of
heads/L1_clean/v3 pre-origin table; W3 inputs end 2021-05). Stations established after
2022-12 ({rep['replaced_after_origin_stations']} replaced) are marked
`established_after_origin`.
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", metavar="OUTDIR", help="development run with synthetic agency "
                    "values at valid stations; no pre-registration, manifest or ledger")
    ap.add_argument("--note", default=None, help="build note stored in the manifest")
    a = ap.parse_args(argv)
    dry = a.dry is not None
    score = None
    if dry:
        out = Path(a.dry)
        out.mkdir(parents=True, exist_ok=True)
        version_name = "dry"
    else:
        from harness import prereg3
        score = prereg3.assert_can_score(lib.PREREG_ID, lib.WP)
        log(f"assert_can_score ok: {score['tests']}; post-registration looks "
            f"{len(score['post_registration_looks'])}")
        version_name = next_version_name(lib.PRODUCT)
    res = compute(dry, version_name)
    gate, policy, verdict = gate_and_policy(res, version_name)
    log(f"C4 policy pass={res['c4']['pass']}; H1b verdict={verdict}; product gate "
        f"passed={gate['passed']}")

    val = {
        "wp": lib.WP, "product": lib.PRODUCT, "version": version_name,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "dry_run_synthetic_agency_values": dry,
        "preregistration": None if score is None else {
            "id": lib.PREREG_ID, "sha256": score["sha256"], "tests": score["tests"],
            "inputs_verified": score["inputs_verified"],
            "post_registration_looks": score["post_registration_looks"],
            "unverifiable_prior_looks": score["unverifiable_prior_looks"][:20]},
        "registered": {
            "T1_sd_coverage_lobo": {"test_id": lib.TEST_ID, "pass": res["t1_pass"],
                                    "gate": res["cov"], "n_stations": res["n_cal"],
                                    "n_blocks": res["n_blocks"], "tau_all_m": res["tau"]},
            "C1_completeness": res["c1"], "C2_reproduction": res["c2"],
            "C3_integrity": res["c3"], "C4_use_policy": res["c4"]},
        "h1b_verdict": verdict, "gate": gate,
        "upstream_w3_policy": res["w3_probe"],
        "reported_not_gated": res["reported"],
        "inputs_opened": res["notes"],
        "parameters": {"t_ref": lib.T_REF, "level": lib.LEVEL, "z90": lib.Z90,
                       "est_cutoff": lib.EST_CUTOFF, "gross_m": lib.GROSS_M,
                       "seen_before_registration": list(lib.SEEN_BEFORE_REGISTRATION),
                       "w3_point_sd_mad_m": res["sd_pt_mad"],
                       "w3_epoch_error_a_m_per_yr": res["a_epoch"],
                       "block_m": 10000.0, "block_origin": "twin grid (x0, y0)"},
    }
    if dry:
        publish(Path(a.dry), res, gate, policy, val, None, a.note, manifest_too=False)
        log(f"DRY outputs in {a.dry}")
        return 0
    out = manifest.new_version_dir(lib.PRODUCT)
    if out.name != version_name:
        raise SystemExit(f"version directory {out.name} != expected {version_name}")
    publish(out, res, gate, policy, val, score, a.note,
            prereg_path=PREREG_DIR / f"{lib.PREREG_ID}.json", evidence_dir=EVIDENCE)
    return 0


def publish(out: Path, res: dict, gate: dict, policy: dict, val: dict, score: dict | None,
            note: str | None, *, manifest_too: bool = True, prereg_path: Path | None = None,
            evidence_dir: Path | None = None) -> dict | None:
    """Write the outputs into ``out``; with ``manifest_too`` also the manifest, then probe the
    published product through the guard and re-verify its hashes (evidence written to
    ``evidence_dir``)."""
    version_name = val["version"]
    tab, blk = res["tab"], res["blk"]
    tab.to_parquet(out / "station_ground.parquet", index=False)
    blk.to_csv(out / "calibration_blocks.csv", index=False)
    (out / "validation.json").write_text(json.dumps(val, indent=1, ensure_ascii=False,
                                                    default=float) + "\n")
    (out / "PRODUCT_CARD.md").write_text(card(res, gate, version_name), encoding="utf-8")
    if not manifest_too:
        return None
    m = manifest.build_manifest(
        out, product=lib.PRODUCT, layer="L1",
        inputs=[str(lib.H1A_DIR / lib.H1A_TABLE), str(lib.H1A_DIR / "station_meta_clean.parquet"),
                str(lib.H1A_DIR / "MANIFEST.json"), str(lib.W3_DIR / lib.W3_MOSAIC),
                str(lib.W3_DIR / lib.W3_WELLS), str(lib.W3_DIR / "MANIFEST.json"),
                str(lib.LEVELING), str(TWIN_GRID_NPZ), str(prereg_path)],
        code=[str(p) for p in CODE],
        parameters={**val["parameters"], "tau_m": res["tau"],
                    "calibration_rule": "split-conformal 90 % on tau_i = sqrt(max(0, (|r|/z)^2 - "
                                        "sd_w3^2)), rank ceil((n+1) 0.9)",
                    "preregistration": {"id": lib.PREREG_ID, "path": rel_to_repo(prereg_path),
                                        "sha256": None if score is None else score["sha256"]},
                    "guard_uses_of_inputs": {"heads/L1_clean/v3": "model_design (station table) "
                                             "and after_the_fact_analysis (ModificationDate, "
                                             "reported only)",
                                             "ground_elevation/v3": "design_time_evidence"}},
        fit_window_end="2022-12", validation_report="validation.json", gate=gate,
        allowed_uses=lib.ALLOWED_USES, forbidden_uses=lib.FORBIDDEN_USES,
        personal_data="none (public monitoring-station ids and elevations; no coordinates, "
                      "names, contact or address fields are copied)",
        extra={"product_card": "PRODUCT_CARD.md", "build_note": note,
               "flags_in_use": {"0": "agency GroundHeight as stored (real value)",
                                "5": "W3 DTM well point (2017.5) replacing GroundHeight 0.0 "
                                     "or NaN, sd calibrated per H1b.T1",
                                "8": "no value (no coordinates)"},
               "release_condition": "a recorded user decision on the failed W3 gate (waiver or "
                                    "new test) and a new version of this product built from "
                                    "the released ground elevation"})
    for k in ("allowed_uses", "forbidden_uses", "fit_window_end", "gate"):
        if m[k] != policy[k]:
            raise SystemExit(f"manifest {k} differs from the probed policy")
    manifest.write_manifest(out, m)
    log(f"wrote {out}/MANIFEST.json (gate passed={gate['passed']})")

    # guard probes on the published product (evidence, outside the product directory)
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
    ev = {"product": rel_to_repo(out), "probes": probes, "verify_hashes_problems": hp,
          "time_utc": _dt.datetime.now(_dt.timezone.utc).isoformat()}
    if evidence_dir is not None:
        evidence_dir.mkdir(parents=True, exist_ok=True)
        (evidence_dir / f"guard_probe_{version_name}.json").write_text(
            json.dumps(ev, indent=1, ensure_ascii=False) + "\n")
        (evidence_dir / f"calibration_sids_{version_name}.json").write_text(
            json.dumps({"n": res["n_cal"], "sids": res["cal_sids"]}, indent=1) + "\n")
    log(json.dumps({u: v["result"] for u, v in probes.items()}))
    log(f"verify_hashes problems: {hp}")
    return ev


if __name__ == "__main__":
    raise SystemExit(main())
