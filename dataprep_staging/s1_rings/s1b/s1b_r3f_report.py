"""WP S1b repair round r3, fix round: numbers, quote check, ledger rows, provenance and JSON.

    python -I dataprep_staging/s1_rings/s1b/s1b_r3f_report.py

* ``S1b_evidence/repair_r3/fix_round/numbers.json``: the fix-round numbers from
  ``step_policy_grid.json``;
* ``fix_round/numbers_check.json``: every quoted fix-round number recomputed and found in
  ``reports/S1b.repair.md``;
* ``fix_round/PROVENANCE.json``: both declarations intact (0 changed inputs), snapshot hashes,
  Part 1 / Part 2 body identity with their snapshots, the grid's sha256, this round's looks;
* ``ledger/rows/S1b.csv``: one row per (number, source) of the fix round (``S1b.r3f.*``);
* ``reports/S1b.repair.json``: the r3 JSON is archived read-only to
  ``fix_round/S1b.repair.r3_round.json`` (first run only) and the report JSON is rewritten with
  the fix round's acceptance rows, summary and a ``fix_round`` block.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from common import ledger  # noqa: E402
from common.hashing import sha256_file  # noqa: E402
from common.paths import rel_to_repo  # noqa: E402
from harness import prereg3  # noqa: E402

from s1_rings.s1b import s1b_const as K  # noqa: E402
from s1_rings.s1b.s1b_run import dump  # noqa: E402

R3 = K.EVIDENCE / "repair_r3"
FIX = R3 / "fix_round"
REPORTS = K.DATAPREP / "reports"
MD = REPORTS / "S1b.repair.md"
JS = REPORTS / "S1b.repair.json"
ARCH = FIX / "S1b.repair.r3_round.json"
DECL_FIX = "S1b.r3f.step_policy_symmetry.v1"
DECL_R3 = "S1b.r3.rings_v3_second_look.v1"
RINGS3 = "results/dataprep/rings_bridged/v3"
GNSS4 = "results/dataprep/gnss_clean/full/v4"
GNSS2 = "results/dataprep/gnss_clean/full/v2"
R3DIR = "results/dataprep/reports/S1b_evidence/repair_r3"
PV = {RINGS3: "rings_bridged/v3", GNSS4: "gnss_clean/full/v4", GNSS2: "gnss_clean/full/v2",
      R3DIR: "S1b r3 evidence"}
CELLS6 = ("v2|product|noSJES", "v2|catalogue_in_gap|noSJES", "v2|catalogue_all|noSJES",
          "v4|product|noSJES", "v4|catalogue_in_gap|noSJES", "v4|catalogue_all|noSJES")
R3_CELL = "v4|catalogue_in_gap|noSJES"
CODE = ["s1b_r3f_looks.py", "s1b_r3f_symmetry.py", "s1b_r3f_prereg.py", "s1b_r3f_report.py",
        "tests/test_s1b_r3f.py"]


def load(p: Path) -> dict:
    return json.loads(Path(p).read_text())


def lr_vals(cell: dict, rules, keys=("LR_correct_vs_up25", "LR_correct_vs_down25")) -> list:
    return [cell["lr"][m][r][k] for m in cell["lr"] for r in rules for k in keys
            if cell["lr"][m][r][k] is not None]


def numbers() -> dict:
    g = load(FIX / "step_policy_grid.json")
    c = g["cells"]
    rr = ("range_ppm:30", "range_ppm:40")
    site = {k: {o["station"]: o for o in c[k]["per_site"] if o["evaluable"]} for k in c}
    other5 = [k for k in CELLS6 if k != R3_CELL]
    n = {
        "counts": {k: {r: f"{v['readings'][r]['count']}/{v['readings'][r]['n']}"
                       for r in v["readings"]} for k, v in c.items()},
        "family_pass": {k: {r: v["readings"][r]["family_pass"] for r in v["readings"]}
                        for k, v in c.items()},
        "rmse": {k: [round(v["rmse_mm"]["primary_vs_mid"], 1), round(v["rmse_mm"]["raw_vs_mid"], 1)]
                 for k, v in c.items()},
        "summary": g["summary"],
        "lr_range_six": [round(min(sum((lr_vals(c[k], rr) for k in CELLS6), [])), 2),
                         round(max(sum((lr_vals(c[k], rr) for k in CELLS6), [])), 2)],
        "lr_range_r3_cell": [round(min(lr_vals(c[R3_CELL], rr)), 2),
                             round(max(lr_vals(c[R3_CELL], rr)), 2)],
        "lr_range_other5": [round(min(sum((lr_vals(c[k], rr) for k in other5), [])), 2),
                            round(max(sum((lr_vals(c[k], rr) for k in other5), [])), 2)],
        "lr_range_v2_product": [round(min(lr_vals(c["v2|product|noSJES"], rr)), 2),
                                round(max(lr_vals(c["v2|product|noSJES"], rr)), 2)],
        "lr_k1_six": [round(min(sum((lr_vals(c[k], ("k:1",)) for k in CELLS6), [])), 2),
                      round(max(sum((lr_vals(c[k], ("k:1",)) for k in CELLS6), [])), 2)],
        "tuku": {k: [round(site[k]["TKJS"]["C_pre_mm"], 1), round(site[k]["TKJS"]["C_post_mm"], 1)]
                 for k in CELLS6},
        "tuku_primary": round(site[R3_CELL]["TKJS"]["C_primary_mm"], 1),
        "kecuo_catalogue_all": {v: [round(site[f"{v}|catalogue_all|noSJES"]["KTES"][x], 1)
                                    for x in ("C_pre_mm", "C_post_mm")] for v in ("v2", "v4")},
        "addback_max_mm": max(x[k] for v in g["addback_integrity"].values() for x in v.values()
                              for k in ("max_abs_value_plus_corr_minus_raw_mm",
                                        "max_abs_corr_minus_step_sum_mm")),
        "declaration_sha256": g["declaration_sha256"],
    }
    tk = n["tuku"]
    n["tuku_gap_above_top_detector_free"] = {
        "v2": round(n["tuku_primary"] - max(tk["v2|catalogue_all|noSJES"]), 1),
        "v4": round(n["tuku_primary"] - max(tk["v4|catalogue_all|noSJES"]), 1)}
    return n


def quotes(n: dict) -> list[tuple[str, str]]:
    cs, s = n["counts"], n["summary"]["counts_over_six_cells"]
    q = []
    for k in CELLS6:
        q.append((f"{k} range30", cs[k]["range_ppm:30"]))
        q.append((f"{k} rmse", f"{n['rmse'][k][0]:.1f} / {n['rmse'][k][1]:.1f}"))
    for k in ("v2|catalogue_all|withSJES", "v4|catalogue_all|withSJES"):
        q.append((f"{k} rmse", f"{n['rmse'][k][0]:.1f} / {n['rmse'][k][1]:.1f}"))
    q += [("range min-max", f"{s['range_ppm:30']['min']}-{s['range_ppm:30']['max']}/9"),
          ("k0 min-max", f"{s['k:0']['min']}-{s['k:0']['max']}/9"),
          ("k1 min-max", f"{s['k:1']['min']}-{s['k:1']['max']}/9"),
          ("lr six", f"{n['lr_range_six'][0]:.2f}-{n['lr_range_six'][1]:.2f}"),
          ("lr r3 cell", f"{n['lr_range_r3_cell'][0]:.2f}-{n['lr_range_r3_cell'][1]:.2f}"),
          ("lr other5", f"{n['lr_range_other5'][0]:.2f}-{n['lr_range_other5'][1]:.2f}"),
          ("lr v2 product", f"{n['lr_range_v2_product'][0]:.2f}-"
                            f"{n['lr_range_v2_product'][1]:.2f}"),
          ("lr k1", f"{n['lr_k1_six'][0]:.2f}-{n['lr_k1_six'][1]:.2f}"),
          ("tuku v4 product", "{:.1f} .. {:.1f}".format(*n["tuku"]["v4|product|noSJES"])),
          ("tuku v2 product", "{:.1f} .. {:.1f}".format(*n["tuku"]["v2|product|noSJES"])),
          ("tuku v2 cat_all", "{:.1f} .. {:.1f}".format(*n["tuku"]["v2|catalogue_all|noSJES"])),
          ("tuku v4 cat_all", "{:.1f} .. {:.1f}".format(*n["tuku"]["v4|catalogue_all|noSJES"])),
          ("tuku gap v4", f"{n['tuku_gap_above_top_detector_free']['v4']:.1f} mm (v4)"),
          ("tuku gap v2", f"{n['tuku_gap_above_top_detector_free']['v2']:.1f} mm (v2)"),
          ("kecuo v2", "{:.1f} .. {:.1f} on v2".format(*n["kecuo_catalogue_all"]["v2"])),
          ("kecuo v4", "{:.1f} .. {:.1f} on v4".format(*n["kecuo_catalogue_all"]["v4"])),
          ("addback", f"{n['addback_max_mm']:.3f} mm"),
          ("decl sha", n["declaration_sha256"][:8])]
    return q


def check_quotes(q) -> dict:
    text = MD.read_text()
    miss = [(a, b) for a, b in q if b not in text]
    return {"n_quoted": len(q), "n_found": len(q) - len(miss), "missing": miss}


def ledger_rows() -> list[dict]:
    spec = {
        "S1b.r3f.step_policy_grid_v2": [(GNSS2, "validator", "after the fact; step policies "
                                         "product / catalogue_in_gap / catalogue_all"),
                                        (RINGS3, "input", "values under test")],
        "S1b.r3f.step_policy_grid_v4": [(GNSS4, "validator", "after the fact; step policies "
                                         "product / catalogue_in_gap / catalogue_all"),
                                        (RINGS3, "input", "values under test")],
        "S1b.r3f.step_policy_grid_summary": [(GNSS2, "validator", "after the fact"),
                                             (GNSS4, "validator", "after the fact"),
                                             (RINGS3, "input", "values under test")],
        "S1b.r3f.step_policy_grid_lr": [(R3DIR, "input", "synthetic oc_*.json (r3) and the "
                                         "grid counts (after the fact)")],
        "S1b.r3f.addback_integrity": [(GNSS2, "validator", "value, value_raw, correction_m, "
                                       "steps"),
                                      (GNSS4, "validator", "value, value_raw, correction_m, "
                                       "steps")],
    }
    return [{"number_id": num, "source": src, "role": role, "product_version": PV[src],
             "note": note} for num, items in spec.items() for src, role, note in items]


def body_identity() -> dict:
    cur = MD.read_text()
    idx = [json.loads(x) for x in (R3 / "snapshots" / "index.jsonl").read_text().splitlines()
           if x.strip()]
    path = {r["label"]: K.DATAPREP.parents[1] / r["path"] for r in idx}
    s1 = path["part1_diagnosis"].read_text()
    s2 = path["part2_design"].read_text()
    p1c = cur[cur.index("## Part 1."):cur.index("## Part 2.")]
    p2c = cur[cur.index("## Part 2."):cur.index("## Part 3.")]
    p1s = s1[s1.index("## Part 1."):]
    p2s = s2[s2.index("## Part 2."):]
    root = K.DATAPREP.parents[1]
    return {"part1_body_identical_up_to_separator":
                p1c.rstrip().removesuffix("---").rstrip() == p1s.rstrip(),
            "part2_body_identical_up_to_separator":
                p2c.rstrip().removesuffix("---").rstrip() == p2s.rstrip(),
            "snapshot_hashes_ok": all(sha256_file(root / r["path"]) == r["sha256"]
                                      for r in idx)}


def provenance() -> dict:
    out = {}
    for d in (DECL_FIX, DECL_R3):
        doc, sha = prereg3.load_prereg(d)
        out[d] = {"sha256": sha, "changed_inputs": prereg3.changed_inputs(doc)}
    looks = [json.loads(x) for x in
             (K.DATAPREP / "preregistration" / "looks.jsonl").read_text().splitlines()
             if x.strip()]
    fix_looks = [{k: lk.get(k) for k in ("time", "purpose", "prereg_id")} for lk in looks
                 if lk.get("wp") == K.WP and lk.get("time", "") >= "2026-10-18T02:2"]
    return {"declarations": out, "bodies": body_identity(),
            "grid_sha256": sha256_file(FIX / "step_policy_grid.json"),
            "code": {c: sha256_file(HERE / c) for c in CODE}, "looks_fix_round": fix_looks}


SUMMARY = (
    "S1b stays partial (excluded). The fix round confirms the r3 verifier's major issue and "
    "replaces the one-sided r3 statement with a declared after-the-fact step-policy grid. The "
    "r3 '8/9 with both GNSS artefacts removed' holds in one of six cells (v4 / catalogue_in_gap) "
    "and needs v4's TKJS 2023-24 detector steps; with SJES excluded the tolerance-reading count "
    "is 6-8/9 over the grid and 7/9 under the detector-free (catalogue-only) policy on either "
    "GNSS version, with Yuanchang and Tuku outside. Likelihood ratios correct : +-25 % for the "
    "range readings span 0.33-6.60 over the grid: weak, mixed and step-policy dependent. The "
    "four minor issues are corrected as errata (Parts 1-2 stay unedited) and the r3 verifier's "
    "look is disclosed. No threshold, registered result, verdict or status changed; no product "
    "version; CPU only, 0 API tokens, nothing staged or committed.")


def report_json(n: dict, prov: dict, led: dict, chk: dict) -> dict:
    d = load(ARCH)
    s6 = n["summary"]["counts_over_six_cells"]
    acc = d["acceptance"]
    acc[1]["measured"] = (
        "7/10 and 7/10 (outside: Yuanchang mixture, Kecuo KTES in-gap step, Xizhou SJES data-gap "
        "offset). Fix round: with SJES excluded the count depends on the GNSS step policy, "
        f"{s6['range_ppm:30']['min']}-{s6['range_ppm:30']['max']}/9 over the declared grid "
        "(GNSS v2/v4 x product/catalogue_in_gap/catalogue_all), 7/9 under the detector-free "
        "policy on either version; the r3 '8/9' is the v4/catalogue_in_gap cell only")
    acc[10]["measured"] = (
        acc[10]["measured"] + f"; fix round, 9-site grid: range readings "
        f"{n['lr_range_six'][0]:.2f}-{n['lr_range_six'][1]:.2f} (r3 cell "
        f"{n['lr_range_r3_cell'][0]:.2f}-{n['lr_range_r3_cell'][1]:.2f}, other five cells "
        f"{n['lr_range_other5'][0]:.2f}-{n['lr_range_other5'][1]:.2f})")
    acc.insert(4, {
        "criterion": "k = 1 bracket (reported reading), WRA family (SECOND LOOK)",
        "threshold": ">= 80 % AND RMSE clause",
        "measured": "8/10 (conditional, second look); 7/10 with the ring noise at the residual "
                    "sd (declared variant, fail); discriminating at 10 sites only under N1",
        "passed": False, "units": "sites"})
    acc += [
        {"criterion": "Fix round, verifier major: symmetric step-policy grid replacing the "
                      "one-sided artefact-removed statement (after the fact, declared "
                      f"{DECL_FIX})",
         "threshold": "declared before computation; no cell selected; r3 statement stands only "
                      "if range +-30/40 pass in all six cells",
         "measured": f"range +-30/40 pass in {s6['range_ppm:30']['n_cells_family_pass']} of 6 "
                     "cells; r3 statement withdrawn; prediction 7/9 (v2/catalogue_in_gap) "
                     "confirmed",
         "passed": False, "units": "cells"},
        {"criterion": "Fix round: minor issues 1-5",
         "threshold": "corrected or disclosed",
         "measured": "E1-E4 errata in F.4 (Parts 1-2 unedited); r3 verifier look disclosed "
                     "2026-10-18T02:26:24Z",
         "passed": True, "units": "procedure"},
        {"criterion": "Fix round: provenance and quotes",
         "threshold": "0 changed declared inputs; snapshots intact; quoted numbers match",
         "measured": f"fix declaration changed {len(prov['declarations'][DECL_FIX]['changed_inputs'])}"
                     f", r3 declaration changed {len(prov['declarations'][DECL_R3]['changed_inputs'])}"
                     f"; Part 1 body identical {prov['bodies']['part1_body_identical_up_to_separator']}; "
                     f"{chk['n_found']}/{chk['n_quoted']} quoted numbers found; ledger "
                     f"+{led['rows_appended']} rows, {led['conflicts']} conflicts over "
                     f"{led['rows_all']}",
         "passed": bool(not prov["declarations"][DECL_FIX]["changed_inputs"]
                        and not prov["declarations"][DECL_R3]["changed_inputs"]
                        and chk["n_found"] == chk["n_quoted"] and led["conflicts"] == 0),
         "units": "checks"}]
    d.update({
        "round": "S1br fix round (after the r3 verification) of repair round r3 of WP S1b",
        "date": "2026-10-18", "acceptance": acc, "summary": SUMMARY,
        "fix_round": {"numbers": n, "provenance": prov, "ledger": led, "quote_check": chk,
                      "declaration": DECL_FIX,
                      "evidence": rel_to_repo(FIX), "archived_r3_json": rel_to_repo(ARCH)},
        "issues_for_downstream": d["issues_for_downstream"] + [
            "S3 / gnss_clean (fix round): TKJS's July-August 2023 steps are handled differently "
            "by v2 and v4 (v4 adds -37.1 mm 2023-07-09 and -22.8 mm 2023-08-16, paired with "
            "opposite steps); under the detector-free policy Tuku's bracket lies 16.5-19.8 mm "
            "below the bridged value, so the TKJS step history decides a site of check (c)"],
        "data_needed_if_failed": d["data_needed_if_failed"] + (
            " Fix round: add TKJS's equipment and monument log for July-August 2023.")})
    d["code_paths"] = d["code_paths"] + [f"dataprep_staging/s1_rings/s1b/{c}" for c in CODE]
    return d


def main(argv=None) -> int:
    FIX.mkdir(parents=True, exist_ok=True)
    if not ARCH.exists():
        shutil.copyfile(JS, ARCH)
        os.chmod(ARCH, 0o444)
    n = numbers()
    dump(n, FIX / "numbers.json")
    chk = check_quotes(quotes(n))
    dump(chk, FIX / "numbers_check.json")
    before = len(ledger._read(ledger.fragment_path(K.WP)))
    path = ledger.append_rows(K.WP, ledger_rows())
    allr = ledger.read_all()
    led = {"fragment": rel_to_repo(path), "rows_appended": len(ledger._read(path)) - before,
           "rows_fragment": len(ledger._read(path)), "rows_all": len(allr),
           "conflicts": len(ledger.conflicts(allr))}
    prov = provenance()
    dump(prov, FIX / "PROVENANCE.json")
    dump(report_json(n, prov, led, chk), JS)
    print(json.dumps({"quotes": chk, "ledger": led, "bodies": prov["bodies"],
                      "decl": {k: len(v["changed_inputs"])
                               for k, v in prov["declarations"].items()}},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
