"""WP A0fix: ledger rows and report (results/dataprep/reports/A0fix.md and A0fix.json).

Reads the evidence written under results/dataprep/reports/A0fix_evidence/ (time-encoding
probe before and after the fix, pytest and ruff output, the mutation run of the new tests
against the old guard, the checker run, the rerun of A0's guard probes) and writes the
report and the ledger fragment rows/A0fix.csv. No data is computed here.

Usage: python -m common.a0fix.report
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import ledger  # noqa: E402
from common.hashing import sha256_file  # noqa: E402
from common.paths import REPO, REPORTS_DIR, STAGING, rel_to_repo  # noqa: E402

WP = "A0fix"
EV = REPORTS_DIR / "A0fix_evidence"
CODE = [STAGING / "common" / p for p in (
    "guard.py", "__init__.py", "tests/test_library.py", "a0fix/__init__.py",
    "a0fix/guard_time_probe.py", "a0fix/report.py")] + [STAGING / "README.md"]


#: The tests WP A0fix added to common/tests/test_library.py.
NEW_TESTS = ("test_time_slice_integer_year_column", "test_time_slice_epoch_seconds",
             "test_time_slice_yyyymm_range", "test_time_slice_mixed_and_invalid_types",
             "test_time_slicing_declaration", "test_time_slice_checks_rows_outside_the_use_flags")


def _j(name: str) -> dict:
    return json.loads((EV / name).read_text(encoding="utf-8"))


def _case(probe: dict, name: str) -> dict:
    return next(c for c in probe["cases"] if c["case"] == name)


def gather() -> dict:
    before, after = _j("guard_time_probe_before.json"), _j("guard_time_probe_after.json")
    chk = _j("check_after.json")
    a0p = json.loads((REPORTS_DIR / "A0_evidence/repair/guard_probes.json").read_text())
    rer = _j("r2_guard_probe_rerun.json")
    n_probe = sum(len(v) for v in a0p.values())
    same = sum(rer[s].get(k) == v for s in a0p for k, v in a0p[s].items())
    py = (EV / "pytest_after.txt").read_text().splitlines()
    passed = sum(" PASSED" in ln for ln in py)
    failed = sum(" FAILED" in ln or " ERROR" in ln for ln in py)
    new_tests = [ln.split("::")[1].split()[0] for ln in py
                 if ln.split("::")[-1].split()[0] in NEW_TESTS]
    mut = (EV / "mutation_old_guard_lines.txt").read_text().splitlines()
    mut_fail = len(mut)
    mut_raise = sum("DID NOT RAISE" in ln for ln in mut)
    common_code = {}
    for m in chk["manifests"]:
        mp = Path(m["path"])
        man = json.loads((REPO / mp).read_text())
        cc = [c["path"] for c in man.get("code", []) if c["path"].startswith(
            "dataprep_staging/common/") and "/a0/" not in c["path"]]
        if cc:
            common_code[m["path"]] = {"status": m["status"], "common_code": cc}
    hash_common = [h for m in chk["manifests"] for h in m.get("hash_problems", [])
                   if str(h.get("path", "")).startswith("dataprep_staging/common/")]
    fails = {m["path"]: sorted({h["path"] for h in m["hash_problems"]})
             for m in chk["manifests"] if m["status"] != "ok"}
    src_fail = {s["path"]: s["problems"][:1] + [f"... {len(s['problems'])} problems"]
                for s in chk["sources"] if s["status"] == "fail"}
    orph = {}
    for o in chk.get("orphans", []):
        top = o.split("/")[2]
        orph[top] = orph.get(top, 0) + 1
    sha_b = dict(ln.split()[::-1] for ln in (EV / "common_sha256_before.txt").read_text()
                 .splitlines())
    sha_a = dict(ln.split()[::-1] for ln in (EV / "common_sha256_after.txt").read_text()
                 .splitlines())
    changed = sorted(k for k in sha_a if sha_a[k] != sha_b.get(k))
    yr = _case(after, "integer year, encoding year")
    yr6 = _case(after, "integer year, encoding year, window_end 2022-06")
    return dict(before=before["summary"], after=after["summary"], chk=chk["summary"],
                n_probe=n_probe, same=same, passed=passed, failed=failed,
                new_tests=new_tests, mut_fail=mut_fail, mut_raise=mut_raise,
                common_code=common_code, hash_common=hash_common, fails=fails,
                src_fail=src_fail, orph=orph, changed=changed,
                ruff=(EV / "ruff_after.txt").read_text().strip(),
                vb=_case(before, "integer year, no encoding (verifier case)"),
                eb=_case(before, "epoch seconds, no encoding (verifier case)"),
                va=_case(after, "integer year, no encoding (verifier case)"),
                ea=_case(after, "epoch seconds, no encoding (verifier case)"),
                yr=yr, yr6=yr6, after_cases=after["cases"], before_cases=before["cases"])


def acceptance(g: dict) -> list[dict]:
    after = {c["case"]: c for c in g["after_cases"]}
    oor = [k for k in after if k.startswith("YYYYMM") and "encoding yyyymm" in k
           and k != "YYYYMM integers, encoding yyyymm"]
    mixed = ["mixed csv strings (ISO, YYYYMM, YYYY), no encoding",
             "bare year strings in parquet, no encoding", "float year in parquet, encoding year",
             "nullable Int64 year with NA, encoding year",
             "invalid ISO strings (month 13, slash, padded)"]
    epoch = [k for k in after if k.startswith("epoch")]
    reads = [c for c in g["after_cases"] if c["outcome"] == "read"]
    undeclared = after["YYYYMM integers, no encoding (undeclared)"]["outcome"]
    unknown = after["unknown encoding 'epoch'"]["outcome"]
    return [
        {"criterion": "Integer YEAR column no longer leaks (verifier case: time_slicing on "
                      "'year' with no declared encoding, datum_fit, window_end 2022-12)",
         "threshold": "refused with an error (no rows returned)",
         "measured": f"before: {g['vb']['rows_returned']} of {g['vb']['n_rows']} rows returned, "
                     f"{g['vb']['leaked_rows']} post-window (2023, 2025); after: "
                     f"{g['va']['outcome']} ({g['va'].get('error', '')[:60]}...)",
         "passed": g["va"]["outcome"] == "refused"},
        {"criterion": "Declared integer 'year' column compared at year granularity",
         "threshold": "window 2022-12 returns 2020-2022; window 2022-06 returns 2020-2021; 0 "
                      "post-window rows",
         "measured": f"2022-12: {g['yr']['returned_true_end']}; 2022-06: "
                     f"{g['yr6']['returned_true_end']}; leaked "
                     f"{g['yr']['leaked_rows'] + g['yr6']['leaked_rows']}",
         "passed": g["yr"]["returned_true_end"] == ["2020-12", "2021-12", "2022-12"]
         and g["yr6"]["returned_true_end"] == ["2020-12", "2021-12"]},
        {"criterion": "Epoch seconds (and ms, days) cannot pass a time slice under any "
                      "declaration",
         "threshold": f"all {len(epoch)} epoch probes refused",
         "measured": f"before: {sum(_c(g, 'before', k)['leaked_rows'] > 0 for k in epoch)} of "
                     f"{len(epoch)} leaked (verifier case: all {g['eb']['rows_returned']} rows "
                     f"returned, {g['eb']['leaked_rows']} post-window); after: "
                     f"{sum(after[k]['outcome'] == 'refused' for k in epoch)} of {len(epoch)} "
                     f"refused",
         "passed": all(after[k]["outcome"] == "refused" for k in epoch)},
        {"criterion": "YYYYMM integers validated to 190001-209912 with month 01-12",
         "threshold": "every out-of-range probe refused; valid YYYYMM sliced correctly",
         "measured": f"{sum(after[k]['outcome'] == 'refused' for k in oor)} of {len(oor)} "
                     f"out-of-range probes refused (202213, 202300, 189912, 210001, 20211115/"
                     f"20230115; "
                     f"unit tests add 202200, 2022, 0, -202212); valid: "
                     f"{after['YYYYMM integers, encoding yyyymm']['returned_true_end']}",
         "passed": all(after[k]["outcome"] == "refused" for k in oor)
         and after["YYYYMM integers, encoding yyyymm"]["returned_true_end"]
         == ["2021-11", "2022-12"]},
        {"criterion": "Mixed, float, missing and malformed time columns refused",
         "threshold": "every such probe refused with TimeEncodingError",
         "measured": f"{sum(after[k]['outcome'] == 'refused' for k in mixed)} of {len(mixed)} "
                     f"probes refused (unit tests add object str/int mix, bool, period, "
                     f"non-ASCII digits, 10 malformed ISO strings)",
         "passed": all(after[k]["outcome"] == "refused"
                       and "TimeEncodingError" in after[k]["error"] for k in mixed)},
        {"criterion": "Only declared encodings accepted (iso default, yyyymm, year)",
         "threshold": "undeclared integer columns and unknown encodings refused",
         "measured": f"undeclared YYYYMM: {undeclared}; unknown 'epoch': {unknown}",
         "passed": after["YYYYMM integers, no encoding (undeclared)"]["outcome"] == "refused"
         and after["unknown encoding 'epoch'"]["outcome"] == "refused"},
        {"criterion": "No post-window row returned by any probe after the fix",
         "threshold": "0 leaking cases of 29",
         "measured": f"before: {g['before']['cases_with_leaks']} cases, "
                     f"{g['before']['leaked_rows']} rows; after: "
                     f"{g['after']['cases_with_leaks']} cases, {g['after']['leaked_rows']} rows "
                     f"({len(reads)} cases read correctly, {g['after']['refused']} refused)",
         "passed": g["after"]["cases_with_leaks"] == 0},
        {"criterion": "New tests for year columns, epoch seconds, out-of-range YYYYMM and "
                      "mixed types detect the defect",
         "threshold": "each new test fails against the pre-fix guard",
         "measured": f"{len(g['new_tests'])} new tests; {g['mut_fail']} of {len(g['new_tests'])} "
                     f"fail on the old guard ({g['mut_raise']} with 'DID NOT RAISE', 1 on the "
                     f"missing time_keys helper)",
         "passed": g["mut_fail"] == len(g["new_tests"]) and len(g["new_tests"]) >= 4},
        {"criterion": "Every existing test keeps passing; ruff clean",
         "threshold": "25/25 existing + all new pass; ruff 0 findings",
         "measured": f"{g['passed']} passed, {g['failed']} failed (25 existing + "
                     f"{len(g['new_tests'])} new); ruff: {g['ruff']}",
         "passed": g["failed"] == 0 and g["passed"] == 25 + len(g["new_tests"])
         and g["ruff"] == "All checks passed!"},
        {"criterion": "A0's guard probes (synthetic loopholes and real pilots) unchanged",
         "threshold": "every outcome identical to A0_evidence/repair/guard_probes.json",
         "measured": f"{g['same']} of {g['n_probe']} identical",
         "passed": g["same"] == g["n_probe"]},
        {"criterion": "No manifest's code hash invalidated by the edit (else build v<N+1>)",
         "threshold": "manifests listing dataprep_staging/common/ files re-hash ok; 0 checker "
                      "problems on common/ files",
         "measured": f"changed common files: {g['changed']} (listed as code by no manifest); "
                     + "; ".join(f"{k}: {v['status']}" for k, v in g["common_code"].items())
                     + f"; checker problems on common/: {len(g['hash_common'])}; l0_registry/v2 "
                       f"not needed",
         "passed": not g["hash_common"]
         and all(v["status"] == "ok" for v in g["common_code"].values())},
    ]


def _c(g: dict, which: str, name: str) -> dict:
    return next(c for c in g[f"{which}_cases"] if c["case"] == name)


NOTES = [
    "Import: sys.path.insert(0, '<repo>/dataprep_staging'); from common import guard, manifest, "
    "ledger, source (library version 0.2.1). Nothing else changed in the API; Product and "
    "Access gained a time_encoding field.",
    "Every product that declares time_slicing must also declare its encoding: "
    "time_slicing = {'column': ..., 'params_fit_window_end': 'YYYY-MM', 'encoding': 'iso' | "
    "'yyyymm' | 'year'}. Before manifest.write_manifest, call guard.time_slicing_errors(m, d) "
    "and fail the build on any message. It requires the encoding key, checks every row of the "
    "column in every parquet/csv output, and requires at least one output to carry the column. "
    "manifest.validate_manifest does not check the encoding (see deviations).",
    "Encodings: 'iso' takes ISO-8601 strings (YYYY-MM, YYYY-MM-DD, YYYY-MM-DD[T ]hh:mm[:ss[.f]]"
    "[Z|+hh:mm]) or a datetime64 column (naive or tz-aware), compared on the month written in "
    "the value. 'yyyymm' takes integers 190001-209912 with month 01-12. 'year' takes integer "
    "calendar years 1900-2099. A year Y is readable only when Y-12 <= window_end: "
    "window_end 2022-12 reads through 2022, and 2022-06 excludes 2022. Annual products keyed by "
    "integer year (leveling, crop statistics, TPC annual) declare 'year'.",
    "Refused with guard.TimeEncodingError (a ValueError), before any row is returned: integer "
    "columns not declared yyyymm or year (epoch seconds, ms and days can never pass); values "
    "outside the declared range (2023 declared yyyymm, 202301 declared year, YYYYMMDD); "
    "floats (an integer column with gaps reads as float, so drop or fill those rows, or use a "
    "nullable Int64 column without NA); booleans; pandas periods; missing values; bare year "
    "strings '2022'; non-ASCII digits; object columns mixing strings and numbers; and unknown "
    "encodings (raised when a window is opened). A categorical column is checked through its "
    "categories.",
    "The whole table is checked, including rows whose flags the declared use would drop. An "
    "undated row therefore breaks the slice for every use, so keep undated rows in an output "
    "without the time column. Opened with window_end, read_table still refuses an output that "
    "lacks the time column, and path() stays refused.",
    "A manifest without an encoding key is read as 'iso' (only ISO strings or datetime64 pass). "
    "This keeps A0's r2 test and S2's builder (time_slicing on 'month' strings, no encoding) "
    "working. S2 should still add \"encoding\": \"iso\", because time_slicing_errors flags its "
    "absence. Product.notes records the encoding and whether it was declared or defaulted.",
    "Helpers for builders and consumers: guard.time_keys(series, encoding) gives the YYYYMM key "
    "of the last month each row covers; guard.slice_mask(series, encoding, window_end); "
    "guard.TIME_ENCODINGS; guard.DEFAULT_TIME_ENCODING.",
    "Unchanged from A0: pre-origin rho(k) for sigma_obs is 0.331 m (k=20), 0.494 (15), "
    "0.651 (10), 0.804 (5), 0.862 (3). A pre-origin use of an observed 2012-2026 product needs "
    "time_slicing with an honest params_fit_window_end, opened with window_end <= 2022-12.",
]

DEVIATIONS = [
    "manifest.validate_manifest (the writer's schema check) does not validate "
    "time_slicing.encoding. Editing common/manifest.py would change a code hash recorded by "
    "county_electricity/full/v1 and county_electricity/preorigin/v1 (P1's products). Those "
    "versions would then fail the checker, and only P1 can rebuild them. The check lives in "
    "guard.time_slicing_errors instead, which builders call before writing, and the guard "
    "fails closed at read time. Fold it into manifest.py at the next manifest.py revision or "
    "at the Phase-1 port.",
    "A missing encoding defaults to 'iso' rather than raising, for backward compatibility "
    "with A0 r2 manifests and S2's builder. The default accepts only unambiguous calendar "
    "values (ISO strings, datetime64), and integers always need a declaration, so it cannot "
    "reintroduce the leak. 'Explicit' is enforced on the builder side by time_slicing_errors.",
    "The guard used to accept pandas Period columns. It now refuses them, because periods are "
    "not one of the declared encodings. This is a tightening, and no product used them.",
    "ISO strings with a UTC offset are compared on the calendar month written in the string, "
    "with no UTC conversion. Tz-aware datetime64 values are compared in their own time zone.",
]


def issues(g: dict) -> list[str]:
    out = [
        "Manifest checker (whole tree, 2026-10-12 13:55): "
        f"{g['chk']['manifests']} manifests ({g['chk']['manifests_failed']} failed), "
        f"{g['chk']['sources']} SOURCE.json ({g['chk']['sources_failed']} failed, "
        f"{g['chk']['sources_legacy']} legacy), {g['chk']['files_hashed']} files re-hashed, "
        f"{g['chk']['orphans']} L0 orphans. None of this comes from A0fix: no problem touches "
        "dataprep_staging/common/. A0's final run had 0/0/0. The new failures come from WPs "
        "that wrote after it:",
    ]
    for k, v in g["fails"].items():
        out.append(f"  {k}: code changed after the build ({', '.join(v)}). "
                   "ground_elevation/v3 passes. W3 should keep a version's code immutable, or "
                   "record superseded versions, so that v1 and v2 do not fail every check.")
    for k, v in g["src_fail"].items():
        out.append(f"  {k}: its files[] entries use 'path' where common/source.py expects "
                   f"'file' ({v[-1].replace('... ', '')}, each 'None missing'), so none of "
                   "its files is verified, "
                   "and all are counted as orphans. The fix belongs to the owner (the main "
                   "session) by adding 'file' keys, or to a later common revision that accepts "
                   "'path' as an alias. source.py is registry code, so that revision would need "
                   "l0_registry/v2. The checker hashed none of raw/ or extracted/ (entries "
                   "unresolved) and only listed their file names.")
    out.append("  L0 orphans by source: " + ", ".join(f"{k} {v}" for k, v in
                                                     sorted(g["orphan_items"])) +
               ". W2 (openmeteo_et0) and S4 (licsar_*) should write SOURCE.json for their "
               "downloads.")
    out += [
        "S2 (s2_deep_benchmarks/build.py) declares time_slicing on 'month' strings without "
        "an encoding. It reads correctly (default 'iso'), but add \"encoding\": \"iso\" and "
        "call guard.time_slicing_errors(m, d) before write_manifest.",
        "H1, H2, S1, S3, S5, W1 and any annual product (leveling, crop statistics, TPC "
        "annual) must declare their slice encoding. A post-2022 row with a malformed or "
        "missing date now makes read_table raise for every windowed use, so clean or drop "
        "such rows before the build.",
        "The A0 verifier's open defect (integer time columns read as YYYYMM) is closed. "
        "A0's acceptance claim 'loader guard enforces U3/U5 for every pre-origin use' now "
        "holds for integer and epoch time columns too (29 probe cases, 0 leaks).",
    ]
    return out


def ledger_rows(g: dict) -> list[dict]:
    ev = rel_to_repo(EV)
    guard_py = "dataprep_staging/common/guard.py"
    old = f"{ev}/guard_before_A0fix.py"
    probe = "dataprep_staging/common/a0fix/guard_time_probe.py"
    tests = "dataprep_staging/common/tests/test_library.py"
    pv = "dataprep_staging/common@0.2.1"
    rows = []

    def add(nid, src, role, note):
        rows.append({"number_id": nid, "source": src, "role": role, "product_version": pv,
                     "note": note})

    for nid, which in (("A0fix.probe.before.leaks", "before"),
                       ("A0fix.probe.after.leaks", "after")):
        add(nid, probe, "input", "synthetic probe products (temporary directory, no real data)")
        add(nid, old if which == "before" else guard_py, "input",
            f"guard under test ({which} the fix)")
        add(nid, f"{ev}/guard_time_probe_{which}.json", "validator", "probe outcome record")
    add("A0fix.tests.passed", tests, "validator", "31 tests (25 existing + 6 new)")
    add("A0fix.tests.passed", guard_py, "input", "guard under test")
    add("A0fix.tests.passed", f"{ev}/pytest_after.txt", "validator", "pytest -v output")
    add("A0fix.tests.fail_on_old_guard", tests, "validator", "new tests run on the old guard")
    add("A0fix.tests.fail_on_old_guard", old, "input", "pre-fix guard (+ error alias)")
    add("A0fix.tests.fail_on_old_guard", f"{ev}/mutation_old_guard_lines.txt", "validator",
        "failure lines")
    add("A0fix.a0_probes.identical", "dataprep_staging/common/a0/r2_guard_probe.py", "input",
        "A0 probe script rerun")
    add("A0fix.a0_probes.identical", "results/dataprep/reports/A0_evidence/repair/"
        "guard_probes.json", "validator", "A0 outcomes")
    add("A0fix.a0_probes.identical", f"{ev}/r2_guard_probe_rerun.json", "validator",
        "rerun outcomes")
    add("A0fix.check.summary", "dataprep_staging/common/check.py", "input", "checker CLI")
    add("A0fix.check.summary", f"{ev}/check_after.json", "validator", "checker report")
    add("A0fix.check.common_code_ok", "results/dataprep/l0_registry/v1", "validator",
        "lists common/check.py, common/source.py")
    add("A0fix.check.common_code_ok", "results/dataprep/county_electricity/full/v1",
        "validator", "lists common/manifest.py, ledger.py, hashing.py")
    add("A0fix.check.common_code_ok", "results/dataprep/county_electricity/preorigin/v1",
        "validator", "lists common/manifest.py, ledger.py, hashing.py")
    add("A0fix.check.common_code_ok", f"{ev}/common_sha256_after.txt", "input",
        "common/*.py hashes after the edit (changed: guard.py, __init__.py)")
    return rows


def write_md(g: dict, acc: list[dict], iss: list[str]) -> str:
    ok = all(a["passed"] for a in acc)
    L = ["# WP A0fix: type-safe time slicing in the loader guard (report, 2026-10-12)", "",
         f"Status: **{'passed' if ok else 'partial'}** "
         f"({sum(a['passed'] for a in acc)}/{len(acc)} acceptance criteria). The A0 verifier's "
         "open major defect is fixed in `dataprep_staging/common/guard.py`. Integer time "
         "columns are no longer read as YYYYMM by default. A time slice accepts only a "
         "declared encoding (`iso`, `yyyymm` or `year`) and refuses everything else with "
         "`guard.TimeEncodingError` before it returns a row. No GPU, no Claude API (0 tokens), "
         "nothing under `hydrophysics/`, `tests/` or L0 written, nothing staged in git.", "",
         "## The defect (reproduced before the fix)", "",
         "`_month_strings` turned every integer column into `YYYY-MM` as `v // 100, v % 100`. "
         "A year 2023 became `'0020-23'`, an epoch second 1673740800 became `'16737408-00'` "
         "and an epoch day 19372 became `'0193-72'`; all sort before `'2022-12'`, so they "
         "passed a `window_end = 2022-12` slice. The old guard also ignored any declared "
         "encoding. The probe "
         "(`common/a0fix/guard_time_probe.py`, 29 synthetic products opened for `datum_fit`) "
         f"leaked in {g['before']['cases_with_leaks']} of 29 cases "
         f"({g['before']['leaked_rows']} post-window rows) on the pre-fix guard:", "",
         f"- integer year, no encoding: {g['vb']['rows_returned']} of {g['vb']['n_rows']} rows "
         f"returned, including {g['vb']['leaked_rows']} after the window (2023, 2025);",
         f"- epoch seconds, no encoding: all {g['eb']['rows_returned']} rows returned, "
         f"{g['eb']['leaked_rows']} after the window;",
         "- the same for epoch ms and days, and under every declaration (`yyyymm`, `year`, an "
         "unknown `epoch`);",
         "- invalid YYYYMM values were accepted without complaint: 202213 was read as "
         "`'2022-13'`, 189912 as `'1899-12'` and 20211115 as `'202111-15'`. None of these "
         "probes leaked, but only because the malformed strings happened to sort on the "
         "right side of the window.", "",
         "## The fix", "",
         "- `time_slicing` now carries an `encoding`, one of `guard.TIME_ENCODINGS`:",
         "  - `iso` (the default when the key is absent): ISO-8601 strings or datetime64, "
         "compared at month granularity;",
         "  - `yyyymm`: integers 190001-209912 with month 01-12;",
         "  - `year`: integers 1900-2099, compared at year granularity (a row of year Y is "
         "readable only when `Y-12 <= window_end`).",
         "- `guard.time_keys` and `guard.slice_mask` validate every row of the table (before "
         "the flag filter). They raise on undeclared integer columns, out-of-range values, "
         "floats, booleans, periods, missing values, bare year strings, non-ASCII digits and "
         "mixed-type columns. A categorical column is checked through its categories.",
         "- `check_use` resolves the encoding when a window is opened and raises on an unknown "
         "one. `Product.notes` records the encoding and whether it was declared or defaulted.",
         "- `guard.time_slicing_errors(m, d)` is the builders' pre-write check: it requires an "
         "explicit encoding and checks the column in every parquet/csv output.",
         "- Six new tests cover integer year columns, epoch seconds/ms/days, out-of-range "
         "YYYYMM, mixed and invalid types, the declaration helper, and rows dropped by the flag "
         "filter. README usage notes were updated, and the library version is 0.2.1.", "",
         f"After the fix: {g['after']['cases_with_leaks']} leaking cases of 29 "
         f"({g['after']['read']} read correctly, {g['after']['refused']} refused).", "",
         "## Acceptance", "", "| criterion | threshold | measured | pass |", "|---|---|---|---|"]
    for a in acc:
        L.append(f"| {a['criterion']} | {a['threshold']} | {a['measured'].replace('|', '/')} | "
                 f"{'pass' if a['passed'] else 'FAIL'} |")
    L += ["", "## Library usage notes for downstream WPs", ""] + [f"- {n}" for n in NOTES]
    L += ["", "## Deviations from the task text", ""] + [f"- {d}" for d in DEVIATIONS]
    L += ["", "## Issues for downstream", ""] + [
        (f"- {i}" if not i.startswith("  ") else f"  - {i.strip()}") for i in iss]
    L += ["", "## Files", "",
          "- Code (changed): `dataprep_staging/common/guard.py`, `common/__init__.py` "
          "(version 0.2.1), `common/tests/test_library.py` (+6 tests), "
          "`dataprep_staging/README.md`.",
          "- Code (new): `dataprep_staging/common/a0fix/guard_time_probe.py`, "
          "`common/a0fix/report.py`.",
          f"- Evidence: `{rel_to_repo(EV)}/` holds the pre-fix guard copy, the probe before and "
          "after, pytest and ruff output, the mutation run, the checker report and the A0 "
          "probe rerun.",
          "- Ledger: `results/dataprep/ledger/rows/A0fix.csv`.", ""]
    return "\n".join(L)


def main() -> int:
    g = gather()
    g["orphan_items"] = list(g["orph"].items())
    acc = acceptance(g)
    iss = issues(g)
    rows = ledger_rows(g)
    ledger.append_rows(WP, rows)
    status = "passed" if all(a["passed"] for a in acc) else "partial"
    rep = {
        "wp": WP, "status": status, "date": "2026-10-12",
        "acceptance": acc,
        "key_numbers": [
            f"Probe (29 synthetic time encodings, datum_fit, window_end 2022-12/06): before "
            f"{g['before']['cases_with_leaks']} leaking cases / {g['before']['leaked_rows']} "
            f"rows; after {g['after']['cases_with_leaks']} / {g['after']['leaked_rows']} "
            f"({g['after']['read']} read correctly, {g['after']['refused']} refused)",
            f"Tests: {g['passed']} passed, {g['failed']} failed (25 existing + "
            f"{len(g['new_tests'])} new); {g['mut_fail']}/{len(g['new_tests'])} new tests fail "
            f"on the pre-fix guard; ruff clean",
            f"A0 guard probes rerun: {g['same']}/{g['n_probe']} identical outcomes",
            f"Checker: {g['chk']['manifests']} manifests ({g['chk']['manifests_failed']} failed, "
            f"W3 code edits), {g['chk']['sources']} sources ({g['chk']['sources_failed']} "
            f"failed, TPC SOURCE.json schema), {g['chk']['orphans']} orphans; 0 problems on "
            f"common/ code; l0_registry/v1 and county_electricity v1 ok; no v2 needed",
            "Claude API tokens: 0; GPU: none",
        ],
        "usage_notes": NOTES, "deviations_from_plan": DEVIATIONS, "issues_for_downstream": iss,
        "code": [{"path": rel_to_repo(p), "sha256": sha256_file(p)} for p in CODE],
        "evidence_dir": rel_to_repo(EV), "ledger_rows": len(rows),
        "checker": {"summary": g["chk"], "failed_manifests": g["fails"],
                    "failed_sources": g["src_fail"], "orphans_by_source": g["orph"],
                    "manifests_with_common_code": g["common_code"]},
    }
    (REPORTS_DIR / f"{WP}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1) + "\n",
                                            encoding="utf-8")
    (REPORTS_DIR / f"{WP}.md").write_text(write_md(g, acc, iss), encoding="utf-8")
    for a in acc:
        print("PASS" if a["passed"] else "FAIL", a["criterion"][:70], "|", a["measured"][:120])
    print(status, len(rows), "ledger rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
