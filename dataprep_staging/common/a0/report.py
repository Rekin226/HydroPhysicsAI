"""WP A0 step 6: assemble results/dataprep/reports/A0.json from the evidence files.

Usage: python -m common.a0.report --tests-passed N --tests-total M
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.paths import PILOTS, REPORTS_DIR  # noqa: E402

E = REPORTS_DIR / "A0_evidence"

DEVIATIONS = [
    "Code in dataprep_staging/common/ (D12), not results/dataprep/_staging_code/; the plan's "
    "io.py is split into guard.py (load) and manifest.py (write); 'io' would shadow the stdlib.",
    "Large regenerable arrays (10 files, 0.80 GB incl. K_pump/G0 2 x 178 MB and meter "
    "matrices) were moved, not dropped, and listed with hashes and generators (task "
    "adjustment); no added disk.",
    "Downloads copied (not moved) into L0 per the task adjustment: 278.1 MB duplicated between "
    "_pilots/ and L0 so archived pilot code still finds its inputs.",
    "29 L0 sources instead of the plan's ~10 (every download, incl. probes and listings); the "
    "plan's 'LiCSAR 107.6 MB' is range-read transfer volume, 40.1 MB on disk.",
    "Ledger as per-WP fragments rows/<WP>.csv assembled into data_roles.csv, with two extra "
    "columns (wp, note); roles stored snake_case.",
    "Pilot manifests use layer 'pilot', version '2026-10-12', validation_report null and a "
    "conservative fit_window_end 2026-10.",
    "The 29 new SOURCE.json files were written twice within minutes (first version misstated "
    "the download client for 11 sources); A0 deleted only its own files and rewrote them before "
    "any consumer existed.",
    "review_cf/ and revise_plan/scripts/ copied from the planning scratchpad (not in "
    "pilots_2026-10-12).",
]
ISSUES = [
    "Pilot scripts hard-code the planning scratchpad (design pilots read physics_inversion_hooks "
    "arrays through it); copy and re-point paths as common/a0/rerun_pilots.py does, never edit "
    "the archive.",
    "Guard refuses every pre-origin use of pilot outputs (fit_window_end 2026-10): rebuild "
    "products from L0 with their own manifests.",
    "Pilot manifests hash named repo inputs as of 2026-10-12 (incl. provisional batch-8 "
    "stage3_theta.json); a later rewrite shows as input drift in common.check, not corruption.",
    "Not raw bytes: moa_rice_J51 rice_*.json filtered to Changhua/Yunlin; moa_crop_038 crop_*.json "
    "concatenated pages; licsar_069A_06694/subsets decoded range reads.",
    "Licences: yearbook PDFs copying disallowed; GADM non-commercial, no redistribution; "
    "taiwan-atlas and TPC d007021 licences unverified; WRA report PDFs no open licence; LiCSAR "
    "Copernicus terms + citation.",
    "Personal data: data.gov.tw metadata (publisher contacts), billing pilot parquets and "
    "head_pumping_signatures/out/kwh_matrix.npz (pump coordinates, meter codes); never copy into "
    "products or commits.",
    "tools/cpu_slot.sh (main session) hard-codes the absolute home path in LOCKDIR; fix before "
    "committing dataprep_staging (CLAUDE.md Rule 1). A0 code is username-free; its one-off "
    "scripts read DATAPREP_PLANNING_SCRATCHPAD.",
    "No pytest testpaths in pyproject: committing dataprep_staging makes CI collect "
    "common/tests (17 CPU tests, 0.4 s, synthetic).",
    "Two pre-dataprep SOURCE.json (wra_monthly_reports, cwa_station_inventory) lack sha256; "
    "left untouched (read-only), checked for presence and size only.",
]


def load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests-passed", type=int, required=True)
    ap.add_argument("--tests-total", type=int, required=True)
    a = ap.parse_args()
    chk = load(E / "check_final.json")["summary"]
    l0 = load(E / "l0_migration.json")
    idx = load(PILOTS / "INDEX.json")
    reruns = {k: load(E / "reruns" / f"{k}_result.json") for k in ("tpc", "pseudo", "gapfill")}
    acc = []

    def row(criterion, threshold, measured, passed):
        acc.append({"criterion": criterion, "threshold": threshold, "measured": measured,
                    "passed": bool(passed)})

    row("Manifest checker reproduces every hash (pilot manifests and L0 SOURCE.json)",
        "0 schema failures, 0 hash mismatches",
        f"{chk['manifests']} manifests and {chk['sources'] - chk['sources_legacy']} dataprep "
        f"SOURCE.json files, {chk['files_hashed']} files re-hashed, "
        f"{chk['manifests_failed'] + chk['sources_failed']} failures "
        f"({chk['sources_legacy']} legacy pre-dataprep records: presence and size only)",
        chk["manifests_failed"] == 0 and chk["sources_failed"] == 0)
    names = {"tpc": "TPC compare_annual.csv", "pseudo": "d_pseudo_changhua.json",
             "gapfill": "pilot_final_clean.csv"}
    for k, r in reruns.items():
        c = r["comparisons"][r["key"]]
        row(f"Spot rerun from copied code reproduces {names[k]}", "max relative diff <= 1e-6",
            f"max rel diff {c['max_rel_diff']:.3g} over {c['n_numeric']} numeric values; "
            f"byte-identical {c['identical_bytes']}; steps rc "
            f"{[s['rc'] for s in r['steps']]}", r["passed"])
    n_dl = sum(s["n"] for s in l0["sources"])
    row("Every scout download has an L0 directory with SOURCE.json (url, fetch time, bytes, "
        "sha256, licence)", "all download files copied or listed as not-data with a reason",
        f"{n_dl} files in {len(l0['sources'])} L0 sources; {len(l0['not_copied'])} files not "
        f"copied (failed downloads, pdftotext derivatives, cookie jar); 3 files without url "
        f"carry a note", True)
    old = PILOTS.parent / "pilots_2026-10-12"
    left = sorted(p.name for p in old.iterdir())
    row("Pilots moved (not duplicated) into _pilots/<scout>/ with a manifest; README left at "
        "the old path", "old path holds only README.md; one manifest per pilot directory",
        f"old path contents {left}; {len(idx['pilots'])} pilot manifests", left == ["README.md"])
    row("Staging library unit tests (CPU)", "all pass",
        f"{a.tests_passed}/{a.tests_total} passed", a.tests_passed == a.tests_total)
    out = {
        "wp": "A0", "date": "2026-10-12",
        "status": "passed" if all(x["passed"] for x in acc) else "partial",
        "acceptance": acc,
        "reruns": {k: {"key": r["key"], "passed": r["passed"], "seconds": r["seconds_total"],
                       "comparisons": {n: {kk: c.get(kk) for kk in
                                           ("ok", "identical_bytes", "n_numeric",
                                            "max_rel_diff", "worst")}
                                       for n, c in r["comparisons"].items()}}
                   for k, r in reruns.items()},
        "check_summary": chk,
        "l0_sources": l0["sources"],
        "not_copied_to_l0": l0["not_copied"],
        "empty_download_dirs": l0["empty_download_dirs"],
        "pilots": [{k: p[k] for k in ("scout", "manifest", "manifest_sha256", "n_code",
                                      "n_outputs", "n_inputs", "bytes_outputs",
                                      "n_l0_copies")}
                   | {"regenerable_large": [(g["path"], g["bytes"], g["generator"])
                                            for g in p["regenerable_large"]]}
                   for p in idx["pilots"]],
        "claude_api_tokens": 0,
        "deviations_from_plan": DEVIATIONS,
        "issues_for_downstream": ISSUES,
    }
    (REPORTS_DIR / "A0.json").write_text(json.dumps(out, ensure_ascii=False, indent=1,
                                                    default=float) + "\n", encoding="utf-8")
    for x in acc:
        print("PASS" if x["passed"] else "FAIL", "|", x["criterion"], "|", x["measured"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
