"""WP C0: read-only resume check after a usage-limit interruption (2026-10-13).

The earlier session built and scored everything; it stopped before writing the WP report. This
script re-verifies, without rebuilding or re-scoring anything:

1. every file recorded in the three C0 manifests (inputs, code, outputs) re-hashes;
2. the pre-registration is intact: hash chain, inputs unchanged, the A1 tool still allows the
   registered tests (asked with ``log=False``, so no score is logged);
3. an independent spot check of the delivered arrays, written separately from ``checks.py``
   (in-grid energy by county 2012-2022, domestic share, domestic-excluded identity, class-sum
   identity, NaN <=> flag 8 in the last month, flag values);
4. ledger conflicts across every fragment.

It then logs one ``integrity_check`` look in the A1 look log, appends ledger rows for the
numbers it produced and saves its evidence under ``reports/C0_evidence/resume/``.

Usage: python -I resume_c0.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib  # noqa: E402
from common import ledger  # noqa: E402
from common.check import check_manifest  # noqa: E402
from harness import prereg2  # noqa: E402

PE = lib.PRODUCT_ROOT
VERSIONS = {"census_L1": PE / "census_L1" / "v1", "preorigin": PE / "preorigin" / "v1",
            "full": PE / "full" / "v1"}
OUT = lib.EVIDENCE / "resume"


def spot(v: str) -> dict:
    d = VERSIONS[v]
    c = np.load(d / "cells_by_class.npz", allow_pickle=False)
    t = np.load(d / "cells_total.npz", allow_pickle=False)
    dates = np.array([str(x)[:7] for x in c["dates"]])
    classes = [str(x) for x in c["classes"]]
    counties = [str(x) for x in c["counties"]]
    w = (dates >= "2012-01") & (dates <= "2022-12")
    vb = c["value_by_county"]
    jd = classes.index("domestic")
    o: dict = {"n_months": int(len(dates)), "first": dates[0], "last": dates[-1],
               "classes": classes, "counties": counties}
    for i, k in enumerate(counties):
        tot = float(np.nansum(vb[i][..., w]))
        o[f"{k}_in_grid_GWh_per_yr_2012_2022"] = tot / 1e6 / 11.0
        o[f"{k}_domestic_share_2012_2022"] = float(np.nansum(vb[i][jd][..., w]) / tot)
    va, vn = t["value_all"], t["value_nodomestic"]
    o["nodomestic_identity_max_abs_kwh"] = float(np.nanmax(np.abs((va - c["value"][jd]) - vn)))
    o["all_equals_class_sum_max_abs_kwh"] = float(np.nanmax(np.abs(np.nansum(c["value"], 0)
                                                                   - va)))
    o["county_sum_equals_value_max_abs_kwh"] = float(np.nanmax(np.abs(np.nansum(vb, 0)
                                                                      - c["value"])))
    f = c["flag"]
    o["flag_values"] = sorted(int(x) for x in np.unique(f))
    o["last_month_flag_counts"] = {str(int(x)): int((f[..., -1] == x).sum())
                                   for x in np.unique(f[..., -1])}
    o["nan_iff_flag8_all_months"] = bool(np.array_equal(np.isnan(c["value"]), f == 8))
    o["sd_all_nan"] = bool(np.isnan(c["sd"]).all())
    return o


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    res: dict = {"time_utc": prereg2._now(), "purpose": "resume integrity check (read-only)"}

    man = [check_manifest(VERSIONS[v] / "MANIFEST.json") for v in VERSIONS]
    res["manifests"] = [{k: m.get(k) for k in ("path", "status", "n_hashed", "schema_errors",
                                               "hash_problems")} for m in man]
    res["manifests_ok"] = all(m["status"] == "ok" for m in man)
    res["files_rehashed"] = int(sum(m.get("n_hashed", 0) for m in man))

    a = prereg2.audit()
    res["prereg_audit"] = {
        "chains_intact": a["chains_intact"], "index_records": a["index_records"],
        "looks_records": a["looks_records"],
        "c0_refused": [x for x in a["tests_refused_under_repaired_rule"] if x["wp"] == lib.WP],
        "c0_changed_inputs": [x for x in a["tests_with_changed_inputs"] if x["wp"] == lib.WP],
        "c0_unknown_prereg_looks": [x for x in a["looks_with_unknown_prereg_id"]
                                    if x["wp"] == lib.WP]}
    try:
        ok = prereg2.assert_can_score(lib.PREREG_ID, lib.WP, log=False)
        res["prereg_status"] = {"allowed": True, "sha256": ok["sha256"],
                                "tests": ok["tests"], "inputs_verified": ok["inputs_verified"],
                                "post_registration_looks_other_wp": [
                                    x for x in ok["post_registration_looks"]
                                    if x["wp"] != lib.WP]}
    except prereg2.PreregistrationError as exc:
        res["prereg_status"] = {"allowed": False, "error": str(exc)[:500]}

    res["spot"] = {v: spot(v) for v in ("preorigin", "full")}
    pre, full = res["spot"]["preorigin"], res["spot"]["full"]
    res["spot_checks"] = {
        "pre_equals_full_headline": all(
            abs(pre[f"{k}_in_grid_GWh_per_yr_2012_2022"]
                - full[f"{k}_in_grid_GWh_per_yr_2012_2022"]) < 1e-9 for k in lib.COUNTIES),
        "identities_le_1e-6_kwh": all(
            s[k] <= 1e-6 for s in (pre, full) for k in (
                "nodomestic_identity_max_abs_kwh", "all_equals_class_sum_max_abs_kwh",
                "county_sum_equals_value_max_abs_kwh")),
        "nan_iff_flag8": pre["nan_iff_flag8_all_months"] and full["nan_iff_flag8_all_months"],
        "preorigin_no_flag8": 8 not in pre["flag_values"],
        "sd_nan_observed": pre["sd_all_nan"] and full["sd_all_nan"]}

    rows = ledger.read_all()
    res["ledger"] = {"rows_all_fragments": len(rows), "conflicts": ledger.conflicts(rows),
                     "c0_rows_before": sum(r["wp"] == lib.WP for r in rows)}
    res["seconds"] = round(time.time() - t0, 1)
    lib.save_json(res, OUT / "resume_check.json")

    prereg2.record_look(
        sorted(["C0:product_descriptive_stats", "C0:product_masks_internal",
                "C0:product_variant_consistency"]),
        "c0_products:2007-01..2026-07", lib.WP, "integrity_check",
        note="[C0-resume-1] resume check 2026-10-13 after a usage-limit interruption: re-hash "
             "of the three pump_electricity manifests, prereg2.audit and assert_can_score("
             "log=False) on C0.census_product.v1, an independent spot check of the delivered "
             "cell arrays (in-grid GWh/yr 2012-2022 by county, domestic share, identities, "
             "NaN<=>flag 8), ledger conflicts; the already scored check and diagnostic files "
             "were re-read for the report; nothing rebuilt or re-scored.")

    pv = lib.rel(VERSIONS["preorigin"])
    fv = lib.rel(VERSIONS["full"])
    lv = lib.rel(VERSIONS["census_L1"])
    ev = lib.rel(OUT / "resume_check.json")
    new = []
    for nid, ver, srcs, note in (
        ("C0.resume.manifests_rehash", pv, [pv, fv, lv],
         f"{len(man)} manifests, {res['files_rehashed']} files re-hashed, "
         f"ok={res['manifests_ok']}"),
        ("C0.resume.spot_in_grid_GWh_per_yr", pv, [pv, fv],
         "independent recomputation: " + ", ".join(
             f"{k} {pre[f'{k}_in_grid_GWh_per_yr_2012_2022']:.2f}" for k in lib.COUNTIES)),
        ("C0.resume.spot_domestic_share", pv, [pv, fv],
         "independent recomputation: " + ", ".join(
             f"{k} {pre[f'{k}_domestic_share_2012_2022']:.3f}" for k in ("yunlin", "changhua"))),
        ("C0.resume.spot_identities", fv, [pv, fv], f"{res['spot_checks']}"),
        ("C0.resume.last_month_flags_full", fv, [fv],
         f"2026-07 class x cell flags {full['last_month_flag_counts']}"),
    ):
        for s in srcs:
            new.append({"number_id": nid, "source": s, "role": "input", "product_version": ver,
                        "note": note})
        new.append({"number_id": nid, "source": ev, "role": "validator", "product_version": ver,
                    "note": note})
    ledger.append_rows(lib.WP, new)
    rows = ledger.read_all()
    bad = ledger.conflicts(rows)
    res["ledger"]["c0_rows_after"] = sum(r["wp"] == lib.WP for r in rows)
    res["ledger"]["conflicts_after"] = bad
    lib.save_json(res, OUT / "resume_check.json")
    print("manifests_ok", res["manifests_ok"], "files", res["files_rehashed"],
          "prereg allowed", res["prereg_status"]["allowed"], "spot", res["spot_checks"],
          "ledger conflicts", len(bad), "c0 rows", res["ledger"]["c0_rows_after"],
          "seconds", res["seconds"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
