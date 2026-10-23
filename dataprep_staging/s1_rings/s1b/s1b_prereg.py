"""WP S1b: write the pre-registration ``S1b.ring_bridge_checks.v1`` (harness.prereg3) BEFORE
any S1b statistic is computed on real displacement values.

    python -I dataprep_staging/s1_rings/s1b/s1b_prereg.py --dry-run   (validate, list blocks)
    python -I dataprep_staging/s1_rings/s1b/s1b_prereg.py              (write once)

Registers the plan's independent checks of the ring bridge (WP S1, part b): (c) GNSS-implied
gap compaction at the ring sites (WRA co-located family; NGL family near the sites) and (d)
InSAR gap subsidence at all 14 sites (bracket containment d2 and the cross-site correlation
d1, each against the raw baseline). Validators only: no S1a parameter is changed by anything
here.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from common.hashing import sha256_file  # noqa: E402
from common.paths import rel_to_repo  # noqa: E402
from harness import prereg3  # noqa: E402

from s1_rings.s1b import s1b_const as K  # noqa: E402
from s1_rings.s1b import s1b_io as IO  # noqa: E402

CODE = ["__init__.py", "s1b_const.py", "s1b_core.py", "s1b_io.py", "s1b_run.py",
        "s1b_synth.py", "s1b_looks.py", "s1b_prereg.py"]
S1A_CODE_USED = ["s1lib.py", "s1const.py"]       # s1lib.line_lr (report-only evidence line)
SHARED = ["common/guard.py", "common/manifest.py", "common/uses.py", "common/paths.py",
          "common/hashing.py"]
DATA = [
    K.RINGS_V2 / "MANIFEST.json", K.RINGS_V2 / "ring_corrections.csv",
    K.GNSS_FULL_V2 / "MANIFEST.json", K.GNSS_FULL_V2 / "wra" / "daily.parquet",
    K.GNSS_FULL_V2 / "ngl" / "daily.parquet", K.GNSS_FULL_V2 / "stations.csv",
    K.GNSS_FULL_V2 / "steps.csv", K.GNSS_FULL_V2 / "common_mode.parquet",
    K.INSAR_V2 / "MANIFEST.json", K.INSAR_V2 / "vertical_ml.npz",
    K.INSAR_V2 / "vertical_cells.npz", K.RING_STATIONS, K.TWIN_GRID,
    K.EVIDENCE / "synthetic_operating_characteristics.json",
    K.EVIDENCE / "synthetic_operating_characteristics_v2.json",
]
WINDOW = "2020-01..2025-05"


def ring_files() -> list[Path]:
    m = json.loads((K.RINGS_V2 / "MANIFEST.json").read_text())
    return [K.RINGS_V2 / o["path"] for o in m["outputs"]
            if o["path"].startswith("rings_corrected_")]


def units() -> tuple[dict, dict]:
    """External units (members by id) of the four registered tests, and the twin cells of the
    ring sites (for the disclosure dry run only)."""
    st = __import__("pandas").read_csv(K.GNSS_FULL_V2 / "stations.csv")
    xy = IO.site_xy()
    prs = IO.pairs(st, xy)
    cells = {s: IO.twin_cell(*xy[s]) for s in K.SITES}
    wra = prs["wra_colocated"]
    ngl = prs["ngl_near"]
    ext = {
        "s1b.c.wra_colocated": {
            "description": "ring sites with a co-located WRA GNSS station (<= 0.3 km): the "
                           "ring top-to-bottom interval and the station's gnss_clean/full/v2 "
                           "daily heights, 2020-01..2025-05 (pre-gap window, gap endpoints, "
                           "post-gap window). Twin cells: "
                           + ", ".join(f"{K.SITE_EN[p['site']]} {cells[p['site']]}"
                                       for p in wra),
            "ids": sorted({f"ring_site:{p['site']}" for p in wra}
                          | {f"wra_gnss:{p['station']}" for p in wra}),
            "window": WINDOW},
        "s1b.c.ngl_near": {
            "description": "ring sites with an NGL GNSS station within 5 km (gnss_clean/full/v2 "
                           "ngl daily heights), 2020-01..2025-05",
            "ids": sorted({f"ring_site:{p['site']}" for p in ngl}
                          | {f"ngl_gnss:{p['station']}" for p in ngl}),
            "window": WINDOW},
        "s1b.d.insar_ring_sites": {
            "description": "the 14 ring sites: ring top-to-bottom interval and the insar_069A/v2 "
                           "multilooked vertical-equivalent pixels around each site (3 x 3), "
                           "2020-01..2025-05. Declared by site id; the pixels lie in twin "
                           "cells " + ", ".join(f"{K.SITE_EN[s]} {cells[s]}" for s in K.SITES)
                           + " (see the disclosure of S4b's all-cell InSAR looks)",
            "ids": sorted(f"ring_site:{s}" for s in K.SITES) + ["insar_069A:ring_site_pixels"],
            "window": WINDOW},
    }
    return ext, {"cells": cells, "pairs": prs}


def disclosure_cells(tests: list[dict], cells: dict) -> list[dict]:
    """Dry run, never written as a unit: which earlier looks WOULD block the InSAR tests if
    their unit were declared by the ring sites' twin cells instead of site ids."""
    idx = prereg3.membership_index()
    u = "s1b.disclosure.insar_ring_cells"
    v = {"description": "hypothetical", "cells": sorted({c for c in cells.values() if c >= 0}),
         "window": WINDOW}
    idx.members[idx.canon(u)] = prereg3._declared_members(v)
    hyp = [{"test_id": t["test_id"] + "#cells", "data_scope": t["data_scope"],
            "fresh_units": [u]} for t in tests if t["test_id"].startswith("S1b.d")]
    now = _dt.datetime.now(_dt.timezone.utc)
    out = []
    for h in prereg3.blocking_looks(hyp, None, idx=idx, test_time=now):
        out.append({"test_id": h["test_id"], "look": h["look"],
                    "pairs": [list(p) for p in h["pairs"]][:6]})
    return out


def code_hashes() -> list[dict]:
    files = ([HERE / c for c in CODE] + [HERE.parent / c for c in S1A_CODE_USED]
             + [HERE.parents[1] / c for c in SHARED])
    return [{"path": rel_to_repo(f), "sha256": sha256_file(f)} for f in files]


def doc(ext: dict, meta: dict) -> dict:
    na = prereg3.not_applicable
    pr = meta["pairs"]
    proc_common = (
        "Ring series per site: the total interval S(t) = depth_raw(bottom ring) - "
        "depth_raw(top ring) of rings_bridged/v2's per-site survey file (first member file "
        "for both-member sites; depth_raw is identical in both), at surveys where both rings "
        "are read; c(t) = -1000 (S - S0) mm (shortening positive). t_lp / t_fp = the last "
        "pre-gap / first post-gap survey of ring_corrections.csv. Raw gap change = "
        "1000 (S(t_lp) - S(t_fp)) (checked against -dS_obs_cm). Product values: C_primary "
        "(P(A)-weighted mixture at both-member sites), C_A, C_B from ring_corrections.csv. "
        "Windows: pre = 2020-01-01..2021-12-31, post = 2023-02-01..2025-05-31, each clipped "
        "to the instrument's data and to the ring's gap (post starts at t_fp); the ring is "
        "fitted on exactly the clipped window. Pre fit (instrument and ring separately): "
        "a + r (t - t_mid) + annual sin/cos, OLS, covariance inflated by (1 + rho)/(1 - rho) "
        "of the lag-1 residual autocorrelation (rho clipped to [0, 0.95]). Post fit: rate "
        "with the pre-window annual term subtracted (one intercept per InSAR network "
        "segment). Excess over the gap from window w: E_w = (r_inst,w - r_ring,w) (t_fp - "
        "t_lp) + the instrument-minus-ring annual change between t_lp and t_fp. Instrument "
        "change G = mean over +-h days around t_fp minus that around t_lp. Instrument-implied "
        "ring gap compaction C_w = G - E_w; sigma_w^2 = var(E_w) + var(endpoints) + 2 sd_s^2 "
        "(sd_s: ring_corrections.csv) (+ InSAR: n_bridges x 6.0^2 mm^2; + GNSS: sum of se^2 "
        "of the product's steps dated strictly inside (t_lp, t_fp)). Bracket = [min_w(C_w - "
        "2 sigma_w), max_w(C_w + 2 sigma_w)]. Evaluable: >= 240 instrument samples (GNSS "
        "days) or >= 20 epochs (InSAR) spanning >= 240 d and >= 8 ring surveys in each "
        "window, and both endpoint means available.")
    fam_rule = (
        "family PASS = at least ceil(0.8 n) of the n evaluable sites have C_primary inside "
        "their bracket AND RMSE(C_primary - C_mid) < RMSE(C_raw - C_mid) over the evaluable "
        "sites (C_mid = (C_pre + C_post)/2), with n >= 1. The family enters the S1b verdict "
        "only with n >= 3; with fewer it is reported, not judged.")
    tests = [
        {"test_id": "S1b.c_wra_colocated", "data_scope": "rings+gnss_wra:" + WINDOW,
         "fresh_units": ["s1b.c.wra_colocated"],
         "external_units": {"s1b.c.wra_colocated": ext["s1b.c.wra_colocated"]},
         "pairs": pr["wra_colocated"],
         "procedure": proc_common + " GNSS: gnss_clean/full/v2 WRA daily heights, flags "
                      "0, 1, 3 (9 excluded), product step corrections kept, no common mode "
                      "removed; subsidence = -1000 h; endpoints = mean of the daily values "
                      "within +-3 d (>= 3 days; else +-7 d); endpoint variance = sigma_hf^2 "
                      "(1 + rho_hf)/(1 - rho_hf) (1/n_lp + 1/n_fp) from stations.csv.",
         "threshold": fam_rule},
        {"test_id": "S1b.c_ngl_near", "data_scope": "rings+gnss_ngl:" + WINDOW,
         "fresh_units": ["s1b.c.ngl_near"],
         "external_units": {"s1b.c.ngl_near": ext["s1b.c.ngl_near"]},
         "pairs": pr["ngl_near"],
         "procedure": "as S1b.c_wra_colocated, with gnss_clean/full/v2 NGL stations within "
                      "5 km of a ring site (from coordinates: " + ", ".join(
                          f"{K.SITE_EN[p['site']]}-{p['station']} {p['d_km']} km"
                          for p in pr["ngl_near"]) + ").",
         "threshold": fam_rule + " (expected n = 1: reported, outside the verdict)"},
        {"test_id": "S1b.d2_insar_bracket", "data_scope": "rings+insar_069A:" + WINDOW,
         "fresh_units": ["s1b.d.insar_ring_sites"],
         "external_units": {"s1b.d.insar_ring_sites": ext["s1b.d.insar_ring_sites"]},
         "procedure": proc_common + " InSAR: insar_069A/v2 vertical_ml.npz, per epoch the "
                      "median of the 3 x 3 multilooked pixels around the pixel nearest to the "
                      "site, using pixels in the LiCSBAS15 mask whose segment id equals the "
                      "network segment at every epoch, >= 4 valid; subsidence = -value. "
                      "Endpoints: mean of the epochs within +-18 d that lie in the segment of "
                      "the epoch nearest to the survey (>= 1); endpoint variance 6.6^2 "
                      "(1/n_lp + 1/n_fp) mm^2; n_bridges = segment(t_fp) - segment(t_lp). All "
                      "14 sites.",
         "threshold": fam_rule},
        {"test_id": "S1b.d1_insar_correlation", "data_scope": "rings+insar_069A:" + WINDOW,
         "fresh_units": ["s1b.d.insar_ring_sites"],
         "external_units": {"s1b.d.insar_ring_sites": ext["s1b.d.insar_ring_sites"]},
         "procedure": "Across the sites with an InSAR gap change (endpoint means as in d2, "
                      "no fit): Pearson r between the InSAR gap-mean subsidence rate I/gap "
                      "and the product's gap-mean compaction rate C_primary/gap, and r_raw "
                      "with C_raw/gap (gap = t_fp - t_lp in years).",
         "threshold": "PASS = r_bridged >= the one-sided 5 % critical r for the n sites "
                      "(H0 rho = 0; 0.458 for n = 14) AND r_bridged > r_raw"},
    ]
    return {
        "id": K.PREREG_ID, "wp": K.WP, "kind": "test",
        "title": "S1b: independent checks of the ring bridge (rings_bridged/v2) - (c) "
                 "GNSS-implied and (d) InSAR-implied gap compaction at the 14 ring sites, "
                 "each against the raw baseline",
        "tests": tests,
        "zones": na("no zones; ring sites"),
        "priors": {"report_only_evidence_line": "S1a's s1lib.line_lr against the plan's "
                                                "reference groups A 182.2 +- 11.4 ppm, B 1.9 "
                                                "+- 14.3 ppm (never fed back)"},
        "burn_in": na("no model run"),
        "folds": na("no fitting across sites; each site is checked on its own"),
        "fresh_units": sorted(ext),
        "thresholds": {
            "c_wra / c_ngl / d2": "per family: >= 80 % of evaluable sites contain C_primary "
                                  "in the 2-sigma excess-rate bracket and RMSE(primary) < "
                                  "RMSE(raw) against the bracket mid-point; judged only with "
                                  ">= 3 evaluable sites",
            "d1": "r_bridged >= one-sided 5 % critical r and r_bridged > r_raw",
            "S1b_verdict": "PASS = every judged family (c_wra, c_ngl if n >= 3, d2) AND d1 "
                           "pass. A PASS says the bridged gap compaction is consistent with "
                           "GNSS and InSAR within the declared (uncalibrated) uncertainty; it "
                           "does not open rings_bridged/v2 for any use (its own gate failed, "
                           "and its inputs and these validators are gate-failed products "
                           "read as after_the_fact_analysis).",
            "report_only": [
                "per-site membership evidence: whether the bracket contains C_A, C_B, both "
                "or neither (sites carried as both members in particular); S1a's evidence "
                "line (likelihood ratio A:B) of the instrument-implied scale "
                "b_mid = (C_mid - C_raw)/L, b_sd from the bracket; NEVER fed back into S1a",
                "neighbour GNSS pairs for sites without a co-located WRA station (nearest "
                "station of either family within 6 km: " + ", ".join(
                    f"{K.SITE_EN[p['site']]}-{p['station']} {p['d_km']} km"
                    for p in pr["neighbour"]) + ")",
                "variants after the registered run: GNSS flags 0-1 only; in-gap GNSS steps "
                "not of basis metadata_or_catalogue added back (KTES 2022-08-11 in "
                "particular); ngl_offfan common mode removed; no annual term; InSAR per-cell "
                "values (twin cells) instead of pixels; InSAR without segment intercepts; "
                "1-sigma bracket; SJES excluded (the plan flags an unresolved 2023 offset)",
                "d1 variants: changes instead of rates; without Beichen; the pilot's 9 "
                "sites; Spearman rank correlation",
                "operating characteristics of the rules on availability-preserving "
                "synthetic data (design-time; synthetic_operating_characteristics_v2.json)"]},
        "gate_scores": na("conjunctive pass/fail; no weighted score"),
        "primary_model": {
            "product_checked": "results/dataprep/rings_bridged/v2 (S1a; gate failed)",
            "configuration": {"gnss_flags": list(K.GNSS_FLAGS), "steps_mode": "product",
                              "common_mode": None, "seasonal": True,
                              "insar_source": "pixels", "segment_intercepts": True,
                              "tol_sigma": K.TOL_SIGMA, "exclude_stations": []},
            "constants": {k: getattr(K, k) for k in (
                "COLOCATED_KM", "NGL_NEAR_KM", "NEIGHBOUR_KM", "PRE_GAP_END", "PRE_WIN",
                "POST_WIN", "MIN_SPAN_DAYS", "MIN_GNSS_DAYS", "MIN_INSAR_EPOCHS",
                "MIN_RING_SURVEYS", "GNSS_FLAGS", "GNSS_END_HALF_D",
                "GNSS_END_HALF_D_FALLBACK", "GNSS_END_MIN", "INSAR_PIX_HALF",
                "INSAR_PIX_MIN_VALID", "INSAR_END_HALF_D", "INSAR_END_MIN",
                "INSAR_EPOCH_SD_MM", "INSAR_BRIDGE_SD_MM", "TOL_SIGMA",
                "FAMILY_MIN_PASS_FRAC", "FAMILY_MIN_SITES_GATE", "CORR_ALPHA", "SEED",
                "SYN_REAL")},
            "guard_use": "after_the_fact_analysis for all three products (each gate failed)",
            "code": code_hashes()},
        "inputs": ([rel_to_repo(p) for p in DATA] + [rel_to_repo(p) for p in ring_files()]
                   + [rel_to_repo(HERE / c) for c in CODE]
                   + [rel_to_repo(HERE.parent / c) for c in S1A_CODE_USED]),
        "interpretation": (
            "(c) is the plan's GNSS check: the bridged gap compaction must lie inside the "
            "bracket that the GNSS excess rate allows, using the pre- and post-gap excess "
            "rates as the two ends (the plan's '+-30-40 ppm excess-rate bracket') widened by "
            "2 sigma. (d) extends the pilot's InSAR comparison to all 14 sites with the same "
            "bracket (d2) and the pilot's cross-site correlation (d1). Each is reported "
            "against the raw (uncorrected) gap change. A family failure says the bridged "
            "values are inconsistent with that instrument at more than 20 % of the sites or no "
            "better than raw; the per-site membership evidence is reported for S1a's open "
            "membership issue but changes no S1a parameter (validators only). Operating "
            "characteristics stated before scoring (availability-preserving synthetic run, "
            "seed 1802, 200 realizations; synthetic_operating_characteristics_v2.json): "
            "median bracket half-widths 1.7-5.9 cm at the WRA co-located sites and 2.9-9.1 cm "
            "for InSAR (Beichen widest); per-site size 1-5.5 %; family pass rate 0.995 (WRA) "
            "and 0.995 (InSAR) when the product carries the correct member, 0.96 / 0.965-0.98 "
            "with the registered P(A)-weighted mixture; 0.00 when every site carries the other "
            "group's member or the raw value; 0.885-0.94 under a uniform +-25 % group-A scale "
            "error, so a 25 % scale error is NOT detectable by (c) or (d2). The (d1) pass rate "
            "is 0.16-0.22 in the synthetic world, whose independent uniform excess rates "
            "(0-40 mm/yr) swamp the cross-site compaction spread; it is indicative only (the "
            "pilot's real 9-site r was 0.87, design-time). "
            "Status of the result: the product checked and both validators are gate-failed "
            "and opened only for after_the_fact_analysis, so S1b's verdict is after-the-fact "
            "evidence, not a gate opening any use. U7: the checks use 2023-25 ring, GNSS and "
            "InSAR observations by necessity (the instrument change exists only after the "
            "gap); they select no model, no member and no parameter, and feed no gate of a "
            "model scored on 2023-26. Same U7 tension as S1a's, flagged for the user."),
        "anti_leakage": (
            "No S1b output feeds S1a (rings_bridged/v2 is read, never written). U1: no imputed "
            "value is a target (the product's 2022 path is not used; only surveyed endpoints "
            "and observed GNSS/InSAR values). U7 flagged (interpretation)."),
        "design_time_exposure": [
            "Logged design look (2026-10-13T20:49Z, S1b): the subsidence_gap_sources pilot's "
            "gnss_vs_ring.csv / gnss_vs_ring_stepfixed.csv (GNSS-implied scales 25-569 ppm at "
            "7 sites) and insar_vs_ring_gap.csv (InSAR gap subsidence at 9 sites; r 0.87 "
            "bridged vs 0.10 raw); S1a's after-the-fact GNSS sensitivity on gnss_clean/full/v1 "
            "(GNSS-implied scales at 9 sites, S1a.md); rings_bridged/v2 ring_corrections.csv "
            "(the product checked); gnss_clean/full/v2 stations.csv and steps.csv; insar_069A "
            "card and network segments; coordinates",
            "S3's A3 (excess rates of GNSS minus ring at co-located sites in 2020-21 and "
            "2023-25: median 21.1 vs 15.0 mm/yr, per-site changes HNES -12.2, KTES -8.7, SSES "
            "-8.0 mm/yr) and A2 (GNSS-ring 2021 extra at co-located pairs) were read in the "
            "task brief; they use the same GNSS and ring records as (c)",
            "S4b's looks at InSAR over all active cells (2020-01..2026-01) and its G1 / "
            "after-the-fact InSAR-GNSS changes at the co-located WRA stations, including the "
            "pilot window 2021-12-05 -> 2023-02-20 (nearly the ring gap); no ring value in "
            "them; by another WP",
            "Two availability-preserving synthetic runs (logged integrity_check looks "
            "2026-10-13T20:55Z and 2026-10-14T00:09Z): dates, flags, masks and the step "
            "catalogue only; every value synthetic. MIN_SPAN_DAYS (270 -> 240) and "
            "INSAR_END_MIN (2 -> 1) were set from availability alone (Beichen)",
            "Consequence: S1b is a registered check with declared rules, but NOT a fresh test "
            "of rings_bridged: earlier versions of both comparisons were seen before the "
            "design. It is reported as design-exposed after-the-fact evidence."],
        "disclosure_cell_declaration": {
            "note": "dry run at registration: looks that would block the InSAR tests if their "
                    "unit were declared by the ring sites' twin cells (A1 membership by "
                    "cells) instead of site ids; the units are declared by site id because "
                    "the test's units are ring sites, and these looks hold no ring data",
            "would_block": disclosure_cells(tests, meta["cells"])},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    ext, meta = units()
    d = doc(ext, meta)
    if a.dry_run:
        errs = prereg3.validate_prereg(d)
        idx = prereg3.membership_index()
        for t in d["tests"]:
            for u, v in (t.get("external_units") or {}).items():
                idx.members[idx.canon(u)] = prereg3._declared_members(v)
        now = _dt.datetime.now(_dt.timezone.utc)
        block = prereg3.blocking_looks(d["tests"], None, idx=idx, test_time=now)
        unver = prereg3.unverifiable_looks(d["tests"], None, idx=idx)
        wraps = sorted({u for x in unver for u in x["units"]})
        print(json.dumps({"errors": errs, "blocking": block[:10], "n_unverifiable": len(unver),
                          "unverifiable_units": wraps[:60],
                          "would_block_cells": d["disclosure_cell_declaration"]["would_block"],
                          "n_inputs": len(d["inputs"])}, indent=1, ensure_ascii=False,
                         default=str))
        return 0
    path, sha = prereg3.write_prereg(d)
    print(path, sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
