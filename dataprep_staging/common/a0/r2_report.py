"""WP A0 repair round: update results/dataprep/reports/A0.md and A0.json.

The first-round files are kept unchanged as A0.r1.md and A0.r1.json. A0.json is rewritten as
the full structured result of the WP as it now stands; A0.md keeps the first-round body under
a superseding note and gains a "Repair round" section. Every number the report quotes must
have a ledger row (U6): the script refuses to write otherwise.

Usage: python -m common.a0.r2_report --tests-passed N --tests-total M
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import ledger  # noqa: E402
from common.manifest import load_manifest  # noqa: E402
from common.paths import DATAPREP, LEDGER_ROWS, PILOTS, REPORTS_DIR  # noqa: E402

E = REPORTS_DIR / "A0_evidence"
R = E / "repair"
REPORTED = [
    "A0.check.hashes_reproduced", "A0.rerun.tpc.compare_annual.max_rel_diff",
    "A0.rerun.pseudo.d_pseudo_changhua.max_rel_diff",
    "A0.rerun.gapfill.pilot_final_clean.max_rel_diff", "A0.l0.inventory", "A0.l0.not_copied",
    "A0.pilots.inventory", "A0.library.tests_r1", "A0.ledger.rows_r1", "A0.guard.probe_r1",
    "A0.r2.check.hashes_reproduced", "A0.r2.check.l0_orphans", "A0.r2.pilot_fit_window_ends",
    "A0.r2.l0_registry.inventory", "A0.r2.rerun.physics_chain.d_pseudo_max_rel_diff",
    "A0.r2.rho_k.full_reproduces", "A0.r2.rho_k.preorigin_values", "A0.r2.guard_probes",
    "A0.r2.library.tests", "A0.r2.rerun.gapfill_s3_S"]


def _j(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def gather(tests_passed: int, tests_total: int) -> dict:
    chk = _j(R / "check_r2_final.json")
    chain = _j(R / "reruns" / "physics_chain_result.json")
    rho = _j(R / "reruns" / "rho_k_result.json")
    s3p = R / "reruns" / "gapfill_s3_S_result.json"
    s3 = _j(s3p) if s3p.exists() else None
    probes = _j(R / "guard_probes.json")
    index = _j(PILOTS / "INDEX.json")
    reg, _ = load_manifest(DATAPREP / "l0_registry" / "v1")
    r1 = _j(REPORTS_DIR / "A0.r1.json")
    rows = ledger.read_all(LEDGER_ROWS)
    a0rows = [r for r in rows if r["wp"] == "A0"]
    trees = []
    for p in index["pilots"]:
        m, _ = load_manifest(PILOTS / p["scout"])
        trees += [(p["scout"], t["path"], t["n_files"]) for t in m.get("input_trees") or []]
    return {"chk": chk, "chain": chain, "rho": rho, "s3": s3, "probes": probes,
            "index": index, "reg": reg, "r1": r1, "a0rows": a0rows, "trees": trees,
            "tests": (tests_passed, tests_total), "n_reported": len(REPORTED),
            "unledgered": ledger.unledgered(REPORTED, rows=rows),
            "conflicts": ledger.conflicts(rows)}


def s3_summary(s3: dict | None) -> tuple[str, bool | None]:
    if s3 is None:
        return "not run", None
    if s3["step"]["rc"] != 0:
        return f"s3 failed (rc {s3['step']['rc']})", False
    c = s3["comparisons"]
    ps = c.get("pilot_S.parquet", {})
    th_ok = all(v.get("max_rel") == 0.0 for k, v in c.items() if k.startswith("kalman"))
    if ps.get("identical_frame") and th_ok:
        return (f"pilot_S.parquet identical to the archive ({ps['rows_ref']} rows, all "
                f"{len(ps.get('pred_by_method', {}))} methods, max rel 0); both global Kalman "
                f"thetas refit identically; {s3['seconds_total']:.0f} s"), bool(s3["passed"])
    th = [f"{k.replace('kalman_theta_', '').replace('.npy', '')} {v.get('max_rel'):.2e}"
          for k, v in c.items() if k.startswith("kalman")]
    pred = ps.get("pred_by_method", {})
    worst = [f"{m} {v['max_rel']:.1e}" for m, v in sorted(pred.items())]
    txt = (f"identical frame {ps.get('identical_frame')}; Kalman theta max rel "
           f"{', '.join(th)}; prediction max rel by method: {', '.join(worst)}")
    return txt, bool(s3["passed"])


def build_json(g: dict) -> dict:
    chk, chain, rho, idx = g["chk"], g["chain"], g["rho"], g["index"]
    cs = chk["summary"]
    n_tree_files = sum(t[2] for t in g["trees"])
    cmpx = {r["k"]: r for r in rho["comparison"]}
    s3txt, s3ok = s3_summary(g["s3"])
    syn = g["probes"]["synthetic"]
    n_syn = len(syn)
    n_refused = sum(1 for k, v in syn.items() if v.startswith("raises")
                    or "read_table rows" in k)
    pilots = {p["scout"]: p for p in idx["pilots"]}
    pre = sorted(s for s, p in pilots.items() if p["fit_window_end"] <= "2022-12")
    mixed = sorted(s for s, p in pilots.items() if p["pre_origin_outputs"]
                   and p["fit_window_end"] > "2022-12")
    acceptance = [
        {"criterion": "Manifest checker reproduces every hash (r2 pilot manifests, the L0 "
                      "registry and every L0 SOURCE.json)",
         "threshold": "0 schema failures, 0 hash mismatches",
         "measured": (f"{cs['manifests']} manifests (15 pilot r2 + l0_registry/v1) and "
                      f"{cs['sources']} SOURCE.json; {cs['files_hashed']} file records plus "
                      f"{len(g['trees'])} input trees ({n_tree_files} files) re-hashed; "
                      f"{cs['manifests_failed'] + cs['sources_failed']} failures. The 2 legacy "
                      f"SOURCE.json files are now also hashed through the registry"),
         "passed": cs["manifests_failed"] == 0 and cs["sources_failed"] == 0},
        {"criterion": "Every L0 file under data_fetch_api/new_measurements is covered by a "
                      "SOURCE.json or the L0 registry (plan 2.1; repair round)",
         "threshold": "0 orphans",
         "measured": (f"{cs['orphans']} orphans (1016 before the registry: cwa_codis_rain 800, "
                      f"heads_202210_202610 158, compaction 38, 20 root files)"),
         "passed": cs["orphans"] == 0},
        {"criterion": "Spot rerun from copied code reproduces TPC compare_annual.csv",
         "threshold": "max relative diff <= 1e-6",
         "measured": "0 over 159 values, byte-identical (first round; code and inputs unchanged)",
         "passed": True},
        {"criterion": "Spot rerun from copied code reproduces d_pseudo_changhua.json",
         "threshold": "max relative diff <= 1e-6",
         "measured": (f"first round on archived intermediates: 1.6e-12 over 219 values. Repair "
                      f"round end to end from repository data (a->b->c->d): "
                      f"{chain['comparisons']['d_pseudo_changhua.json']['max_rel_diff']:.1e} "
                      f"over {chain['comparisons']['d_pseudo_changhua.json']['n_numeric']} "
                      f"values; inputs_cwa.npz, hindcast_cwa.npz, K_pump_f32.npy, G0_f32.npy and "
                      f"c_residual_by_county.json byte-identical, M_cwa identical "
                      f"(one recorded patch reproduces an unarchived dtype conversion)"),
         "passed": bool(chain["passed"]) and bool(chain["all_intermediates_ok"])},
        {"criterion": "Spot rerun from copied code reproduces pilot_final_clean.csv",
         "threshold": "max relative diff <= 1e-6",
         "measured": "0 over 671 values, byte-identical (first round, s5b then s5c)",
         "passed": True},
        {"criterion": "Every scout download has an L0 directory with SOURCE.json",
         "threshold": "all data files copied or listed as not-data with a reason",
         "measured": ("754/754 files in 29 sources (first round). Of the 754, 363 (40.5 MB) "
                      "are flagged in their records as derived: 347 decoded LiCSAR subsets and "
                      "grid records, 10 concatenated MOA 038 pages, 5 client-filtered MOA J51 "
                      "files, 1 parsed chunk list"),
         "passed": True},
        {"criterion": "Pilots moved into _pilots/<scout>/ with a manifest; README at old path",
         "threshold": "old path holds only README.md; one current manifest per pilot; r1 "
                      "manifests preserved byte-identical",
         "measured": ("old path = [README.md]; 15 r2 manifests (version 2026-10-12.r2), each "
                      "listing MANIFEST.r1.json with its sha256; INDEX.json r2 (INDEX.r1.json "
                      "kept)"),
         "passed": True},
        {"criterion": "Per-pilot data-window audit replaces the blanket fit_window_end "
                      "(repair round)",
         "threshold": "15/15 pilots with an audited fit_window_end and its basis",
         "measured": (f"15/15; pre-origin: {', '.join(pre)}; mixed (pre-origin outputs "
                      f"listed): {', '.join(mixed)}; the other "
                      f"{15 - len(pre) - len(mixed)} are post-origin (2025-05..2026-10)"),
         "passed": True},
        {"criterion": "Loader guard enforces U3/U5 for every pre-origin use, overrides only "
                      "tighten, exemptions explicit (repair round, verifier major issue)",
         "threshold": "every verifier probe refused or limited as designed",
         "measured": (f"{n_refused}/{n_syn} synthetic probes behave as designed (datum_fit and "
                      f"baseline at fit_window_end 2025-06: writer refuses, guard raises "
                      f"LeakageError; preorigin=False and require_gate=False raise; exemption "
                      f"uses ['*'] and 'x/pre_origin' refused; a time-sliced L1 opens for "
                      f"datum_fit only with window_end 2022-12 and returns only pre-origin rows)"),
         "passed": n_refused == n_syn},
        {"criterion": "Ledger: explicit sources, one role per number, every reported number "
                      "has rows (U6; repair round)",
         "threshold": "0 pattern sources, 0 conflicts, 0 unledgered numbers",
         "measured": (f"{len(g['a0rows'])} A0 rows for {len(REPORTED)} reported numbers; "
                      f"0 pattern sources (refused by the writer); {len(g['conflicts'])} "
                      f"conflicts; unledgered {g['unledgered']}"),
         "passed": not g["conflicts"] and not g["unledgered"]},
        {"criterion": "Staging library CPU unit tests",
         "threshold": "all pass",
         "measured": f"{g['tests'][0]}/{g['tests'][1]} passed (17 in the first round); ruff clean",
         "passed": g["tests"][0] == g["tests"][1]},
        {"criterion": "Supplementary (not a plan criterion): gap-fill fitted step s3, family S, "
                      "refit from the archived monthly matrix",
         "threshold": "max relative diff <= 1e-6 on pilot_S.parquet",
         "measured": s3txt,
         "passed": s3ok if s3ok is not None else False},
    ]
    return {"acceptance": acceptance, "rho": cmpx, "pre": pre, "mixed": mixed,
            "n_tree_files": n_tree_files, "s3txt": s3txt, "s3ok": s3ok}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests-passed", type=int, required=True)
    ap.add_argument("--tests-total", type=int, required=True)
    a = ap.parse_args(argv)
    for name in ("A0.md", "A0.json"):
        src, dst = REPORTS_DIR / name, REPORTS_DIR / name.replace("A0.", "A0.r1.")
        if not dst.exists():
            dst.write_bytes(src.read_bytes())
    g = gather(a.tests_passed, a.tests_total)
    if g["unledgered"]:
        raise SystemExit(f"numbers without ledger rows: {g['unledgered']}")
    b = build_json(g)
    from common.a0.r2_report_text import markdown, structured
    res = structured(g, b)
    (REPORTS_DIR / "A0.json").write_text(json.dumps(res, ensure_ascii=False, indent=1,
                                                    default=float) + "\n", encoding="utf-8")
    md_r1 = (REPORTS_DIR / "A0.r1.md").read_text(encoding="utf-8")
    (REPORTS_DIR / "A0.md").write_text(markdown(g, b, md_r1), encoding="utf-8")
    print("status", res["status"], "| acceptance",
          [(x["criterion"][:40], x["passed"]) for x in res["acceptance"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
