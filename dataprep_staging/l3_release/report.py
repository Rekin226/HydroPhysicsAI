"""WP report of the L3 release: results/dataprep/reports/L3.md and L3.json.

    PYTHONPATH=<repo>/dataprep_staging python -m l3_release.report

Reads the release's RELEASE.json and REBUILD_CHECK.json, the inventory and the post-move
check written to results/dataprep/reports/L3_evidence/, and the ledger, then writes the
acceptance table (criterion, threshold, measured, pass/fail), deviations from the plan and
issues for downstream. It computes nothing that the release itself does not record.
"""
from __future__ import annotations

import json
from pathlib import Path

from common import ledger
from common.paths import REPO

from l3_release import config as C
from l3_release import io_det
from l3_release import ledger_rows as LR

EVID = REPO / "results/dataprep/reports/L3_evidence"


def _load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    rel_dir = REPO / C.RELEASE_DIR / C.RELEASE
    rel = _load(rel_dir / "RELEASE.json")
    rb = _load(rel_dir / "REBUILD_CHECK.json")
    inv = _load(EVID / "inventory.json")
    chk = _load(EVID / "check_final.json")
    gate = rel["release_gate"]
    comps = [c for v in rel["variants"].values() for c in v["components"]]
    pre = rel["variants"]["preorigin"]["components"]
    full = rel["variants"]["full"]["components"]
    n_out = sum(len(c["outputs"]) for c in comps)
    copies = [o for c in comps for o in c["outputs"] if o["mode"] == "copy"]
    n_copy_ident = sum(bool(o["identical_to_upstream"]) for o in copies)
    grid_files = [o for c in comps for o in c["outputs"] if o["kind"] == "grid"]
    pre_ok = all((c["fit_window_end"] <= C.ORIGIN) or
                 (c["non_exempt_fit_window_end"] or "9999") <= C.ORIGIN for c in pre)
    pre_dates = []
    for c in pre:
        for o in c["outputs"]:
            s = o["summary"]
            end = (s.get("dates") or [None, None])[1] if s.get("dates") else (
                (s.get("range") or [None, None])[1])
            if end:
                pre_dates.append(str(end)[:7])
    pre_last = max(pre_dates) if pre_dates else None
    full_pre_uses = sorted({u for c in full for u in c["allowed_uses"]} &
                           {"calibration_target", "datum_fit", "baseline", "initial_conditions",
                            "idw_field", "prior", "prior_width_input", "model_design",
                            "selection", "forcing_frozen", "design_time_evidence"})
    probe_n = sum(len(v) for v in chk["guard_probe"].values())
    probe_bad = sum(len(v["guard_mismatches"]) for v in chk["manifests"].values())
    rows = ledger.read_all()
    conflicts = ledger.conflicts(rows)
    unl = ledger.unledgered(LR.release_number_ids(), rows=rows)
    n_l3 = sum(r["wp"] == C.LEDGER_WP for r in rows)
    excl_versions = sum(max(1, len(x["versions"])) for x in C.EXCLUDED)
    acc = [
        {"criterion": "Only products whose WP passed, or partial WPs' sub-products that passed "
                      "their own gate, enter; each upstream re-checked by "
                      "guard.assert_releasable (manifest valid, gate passed, every hash "
                      "reproduced)",
         "threshold": "every upstream product", "measured":
             f"{len(rel['upstream_products'])} upstream product versions, all gates passed and "
             f"re-hashed: " + ", ".join(u["dir"].replace("results/dataprep/", "")
                                        for u in rel["upstream_products"]),
         "passed": all(u["gate"]["passed"] for u in rel["upstream_products"])},
        {"criterion": "Plan 2.8 item 1: every manifest validates (release components; hashes "
                      "re-checked after the move into place)",
         "threshold": "0 schema errors, 0 hash problems",
         "measured": f"{chk['n_manifests']} component manifests: schema errors "
                     f"{sum(len(v['schema_errors']) for v in chk['manifests'].values())}, hash "
                     f"problems {sum(len(v['hash_problems']) for v in chk['manifests'].values())}"
                     f"; RELEASE.json files re-hashed {chk['release_json_files']['checked']}, "
                     f"problems {len(chk['release_json_files']['problems'])}",
         "passed": bool(chk["passed"]) and gate["item1_every_manifest_validates"]["passed"]},
        {"criterion": "Plan 2.8 item 2: every included product's validation report meets its "
                      "acceptance criteria; every component's own checks pass",
         "threshold": "all upstream gates passed; all component validations passed",
         "measured": f"upstream {sum(v['gate_passed'] for v in gate['item2_validation_meets_acceptance']['upstream'].values())}"
                     f"/{len(gate['item2_validation_meets_acceptance']['upstream'])} gates "
                     f"passed; components {sum(gate['item2_validation_meets_acceptance']['components'].values())}"
                     f"/{len(gate['item2_validation_meets_acceptance']['components'])} passed "
                     f"(accepted exceptions: "
                     f"{sum(len(c['accepted_exceptions']) for c in comps)}, declared before the "
                     f"build); heads/recovered_wells/v3 excluded because plan H3 acceptance part "
                     f"1 failed (3/15)",
         "passed": gate["item2_validation_meets_acceptance"]["passed"]},
        {"criterion": "Plan 2.8 item 3: the ledger has rows for the release; data_roles.csv "
                      "assembled from the per-WP fragments",
         "threshold": "0 unledgered release numbers, 0 conflicts",
         "measured": f"{n_l3} L3 rows for {len(LR.release_number_ids())} release numbers; "
                     f"unledgered {len(unl)}; conflicts across {len(rows)} rows of all "
                     f"fragments {len(conflicts)}",
         "passed": not unl and not conflicts},
        {"criterion": "Plan 2.8 item 4: product card with coverage, known limits, flags in use",
         "threshold": "PRODUCT_CARD.md present", "measured":
             f"PRODUCT_CARD.md sha256 {gate['item4_product_card']['sha256'][:16]}...",
         "passed": gate["item4_product_card"]["passed"]},
        {"criterion": "Plan 2.8 item 5: two builds from scratch give identical hashes",
         "threshold": "every file identical, same file set",
         "measured": f"{rb['compare']['n_identical']} of {rb['compare']['n_files_a']} files "
                     f"identical; differing {len(rb['compare']['differing'])}; one-sided "
                     f"{len(rb['compare']['only_in_a']) + len(rb['compare']['only_in_b'])}; "
                     f"tree digest {rb['compare']['tree_digest_a'][:16]}...",
         "passed": bool(rb["compare"]["identical"])},
        {"criterion": "Grid arrays on the twin's 1 km grid with value/sd/flag, dates and the "
                      "grid-mask hash",
         "threshold": "mask and grid hash equal the twin's; A on the second-to-last axis; "
                      "value/sd/flag valid (plan 2.4)",
         "measured": f"{len(grid_files)} grid array files, all with mask hash "
                     f"{rel['grid']['mask_hash'][:12]}... and grid hash "
                     f"{rel['grid']['grid_hash'][:12]}...; twin grid reproduced by build_grid "
                     f"on the fan polygon: {rel['grid']['polygon_crosscheck']['passed']}; "
                     f"accepted exceptions (declared before the build): "
                     + "; ".join(f"{c['variant']}/{c['name']}: {x['error']}"
                                 for c in comps for x in c["accepted_exceptions"])
                     + "; ET0 stays at its 61 native points (deviation)",
         "passed": all(c["validation_passed"] for c in comps)
         and rel["grid"]["polygon_crosscheck"]["passed"]},
        {"criterion": "Pre-origin variant: fit_window_end <= 2022-12 (or non-exempt window <= "
                      "2022-12 with listed exemptions) and no dated value after 2022-12 (U3)",
         "threshold": "every pre-origin component", "measured":
             f"{len(pre)} components; windows ok {pre_ok}; latest dated value {pre_last}; "
             f"components with inherited exemptions: "
             + ", ".join(f"{c['name']} ({len(c['exemptions'])})" for c in pre
                         if c["exemptions"]),
         "passed": pre_ok and (pre_last is None or pre_last <= C.ORIGIN)},
        {"criterion": "Full variant: diagnostic arm and analysis uses only (U4)",
         "threshold": "no pre-origin use allowed", "measured":
             f"{len(full)} components; pre-origin uses allowed: {full_pre_uses or 'none'}",
         "passed": not full_pre_uses},
        {"criterion": "Loader guard on the release: allowed uses open, every other use is "
                      "refused (U5)",
         "threshold": "0 mismatches", "measured":
             f"{probe_n} component-use probes, mismatches {probe_bad}",
         "passed": probe_bad == 0},
        {"criterion": "Every product version under results/dataprep is classified; every "
                      "excluded one listed with its reason",
         "threshold": "0 unclassified", "measured":
             f"included {inv['counts']['included']}, excluded {inv['counts']['excluded']}, "
             f"unclassified {inv['counts']['unclassified']} manifests; RELEASE.json lists "
             f"{excl_versions} excluded versions or WP-level items",
         "passed": inv["counts"]["unclassified"] == 0 and not inv["included_not_found"]},
        {"criterion": "Byte copies equal the upstream files",
         "threshold": "sha256 equal to the upstream manifest", "measured":
             f"{n_copy_ident} of {len(copies)} copied files identical ({n_out} outputs in all)",
         "passed": n_copy_ident == len(copies)},
    ]
    status = "passed" if all(a["passed"] for a in acc) else "partial"
    summary = (
        f"The first L3 model-ready release is {C.RELEASE_DIR}/{C.RELEASE}/. It has "
        f"{len(pre)} pre-origin and {len(full)} full components plus the grid, built from "
        f"{len(rel['upstream_products'])} upstream product versions whose gates passed (C0, "
        "P2abc, P2def extension, H1a, H2 bridge and IC field, S2 printed piles, W1, W2, P6). "
        f"{excl_versions} product versions or WP-level items are excluded with reasons, among "
        "them every subsidence product except the printed piles (rings, GNSS, InSAR, fused "
        "field and DTM failed their gates) and the recovered wells (plan H3 acceptance part 1 "
        "failed, 3/15). Every byte was read through the loader guard with a declared use, and "
        "every release use is one the upstream guard grants. Pre-origin components end by "
        "2022-12, or run through inherited exemptions with a non-exempt window ending "
        "2022-12. Full components allow no pre-origin use. "
        f"Two builds from scratch gave {rb['compare']['n_identical']} identical files out of "
        f"{rb['compare']['n_files_a']}. The release chooses no arm or member (U7). No GPU, no "
        "Claude API, nothing under hydrophysics/ or tests/, nothing staged.")
    deviations = [
        "Layout: each component is an L3 product version <release>/<variant>/<component>/v1/ "
        "with its own MANIFEST.json, so the A0 guard enforces per-component uses; RELEASE.json "
        "and PRODUCT_CARD.md sit at the release root, which has no MANIFEST.json of its own "
        "(the schema requires a v<N> directory).",
        "Forcing arrays already on the grid are byte copies of the upstream files (same "
        "sha256), not re-packed into one layout; they already carry value/sd/flag, dates and "
        "the mask and grid hashes. Re-written files: W1's grid and draws (upstream 'dates' are "
        "pickled object arrays, never unpickled; dates reconstructed from district_monthly.csv "
        "and length-checked), P6's member (value/sd/flag keys added: flag 5, sd NaN because the "
        "variants are its uncertainty), H2's IC field (value/sd/flag as (4, A, 1), flag 3), and "
        "the H1a and pre-origin S2 tables (guarded read_table output; H1a gains sd, method, "
        "product_version, source_ids).",
        "ET0 stays at its native 61 points rather than an (A, T) array: the twin's recharge "
        "builder IDWs these series itself, and the file keeps the keys that loader reads. "
        "Grid arrays of ET0 would have needed a new interpolation with no sd.",
        "Tables read through the guard carry only the rows the read use grants: flag-8 "
        "(missing) and flag-9 (suspect) head months and flag-8 pile months are not carried; "
        "byte-copied tables keep every row and the release guard filters them per use.",
        "Uses were tightened where the basis is missing: prior_width_input is not granted on "
        "the electricity components (C0 fill sd fails the plan's coverage rule per the C0 "
        "verifier; several P2abc sds are uncalibrated).",
        "Validators and evidence products whose gates passed (P5 head signatures, N1 "
        "interventions, P4m diagnostics, P2f layer response, W4 rain check, A1 units) are "
        "excluded as not model data, with reasons; Phase-1 (I1) artefacts are excluded as out "
        "of scope.",
        "The release name keeps the task's date (release_20261012); the build ran on "
        "2026-10-17 and manifests carry that creation date.",
        "No pre-registration: the release runs no statistical test; its inclusion rule and "
        "accepted exceptions were fixed in config.py before the first build.",
    ]
    issues = [
        "I1 (after the freeze): open release components through common.guard.open_product with "
        "the use you make; the preorigin variant is for calibration, initial conditions, priors, "
        "design and frozen forcing; the full variant only for the recorded arm, statistical "
        "layer, validators and after-the-fact analysis. Nothing is wired into hydrophysics/.",
        "Declare before any result is read (U7): the electricity arm (recorded vs Chow-Lin), the "
        "synchronized-zero members, the domestic-excluded sensitivity member, the public-well "
        "variants and the W1 draw handling.",
        "Grid arrays carry every flag: array consumers must apply the per-use flags (forcing "
        "0-5; flag 9 aggregates are excluded by that rule).",
        "C0's December-2022 fill sd is uncalibrated (coverage 0.808); the release labels it so. "
        "Several P2abc and W1 sds are uncalibrated too: plan 4.8 propagation must not take them "
        "at face value.",
        "The 2 flag-3 head months of 09100211 (2024-09, 2024-10) carry no sd upstream; they "
        "serve validator and after-the-fact uses only.",
        "heads/recovered_wells/v3 is not in the release (H3 acceptance part 1 failed). If the "
        "user or A1 waives it for rung R5 (D9), a release_20261012b can add it as a separate "
        "component; nothing else would change.",
        "Subsidence validation 2023-26 rests on the printed piles only (rings, GNSS, InSAR and "
        "the fused field failed). The 2019-2021 printed pile months are not corroborated by the "
        "feeds.",
        "The 2026-07 census month is partial; the per-meter July-2026 imputation (P2abc) is "
        "not carried at cell level; census_extension covers 2026-07..08 (flag 5, uncalibrated).",
        "A new release is needed when an included product gets a new version or when a "
        "user decision opens a failed gate (W3/H1b, S1a, S3, S4b, S5, P2d, H3).",
        "The component manifests list dataprep_staging/common/*.py and hydrophysics/twin/grid.py "
        "as code: editing them makes the checker flag the release; build a new release instead.",
    ]
    out = {
        "wp": "L3", "status": status, "date": rel["created"], "summary": summary,
        "release_dir": f"{C.RELEASE_DIR}/{C.RELEASE}",
        "included": [f"{c['variant']}/{c['name']} <- {c['upstream']['dir']}" for c in comps],
        "excluded": [{"product": x["product"], "versions": x["versions"], "wp": x["wp"],
                      "category": x["category"], "reason": x["reason"]} for x in C.EXCLUDED],
        "acceptance": acc,
        "identical_rebuild": bool(rb["compare"]["identical"]),
        "deviations_from_plan": deviations, "issues_for_downstream": issues,
        "key_numbers": [
            f"components: preorigin {len(pre)}, full {len(full)}, plus grid; outputs {n_out}",
            f"files in the release (hashed): {rb['compare']['n_files_a']}; identical rebuild "
            f"{rb['compare']['identical']}",
            f"grid: {rel['grid']['n_active']} active cells, mask hash "
            f"{rel['grid']['mask_hash']}, grid hash {rel['grid']['grid_hash']}",
            f"inventory: included {inv['counts']['included']}, excluded "
            f"{inv['counts']['excluded']}, unclassified {inv['counts']['unclassified']}",
            f"guard probes {probe_n}, mismatches {probe_bad}",
            f"ledger: {n_l3} L3 rows; {len(rows)} rows across fragments; conflicts "
            f"{len(conflicts)}",
        ],
        "code_paths": [c["path"] for c in rel["code"]] + [
            "dataprep_staging/l3_release/verify.py", "dataprep_staging/l3_release/inventory.py",
            "dataprep_staging/l3_release/report.py"],
        "evidence": [str((EVID / x).relative_to(REPO)) for x in sorted(
            p.name for p in EVID.iterdir())] if EVID.is_dir() else [],
        "claude_api_tokens": 0, "gpu": "none",
    }
    rp = REPO / "results/dataprep/reports"
    (rp / "L3.json").write_text(io_det.dumps(out), encoding="utf-8")
    md = [f"# WP L3: first model-ready release `{C.RELEASE}` (report, {rel['created']})", "",
          f"Status: **{status}**.", "", summary, "",
          "## Acceptance", "", "| criterion | threshold | measured | pass |", "|---|---|---|---|"]
    for a in acc:
        md.append(f"| {a['criterion']} | {a['threshold']} | {a['measured']} | "
                  f"{'pass' if a['passed'] else 'FAIL'} |")
    md += ["", "## Included components", "", "| variant/component | upstream | WP status | "
           "allowed uses |", "|---|---|---|---|"]
    for c in comps:
        md.append(f"| {c['variant']}/{c['name']} | `{c['upstream']['dir']}` | "
                  f"{c['upstream']['wp']}: {c['upstream']['wp_final_status']} | "
                  f"{', '.join(c['allowed_uses'])} |")
    md += ["", "## Excluded", "", "| product | versions | WP | category | reason |",
           "|---|---|---|---|---|"]
    for x in C.EXCLUDED:
        md.append(f"| `{x['product']}` | {', '.join(x['versions']) or '-'} | {x['wp']} "
                  f"({x['wp_final_status']}) | {x['category']} | {x['reason']} |")
    md += ["", "## Deviations from the plan", ""] + [f"- {d}" for d in deviations]
    md += ["", "## Issues for downstream", ""] + [f"- {d}" for d in issues]
    md += ["", "## Files", "",
           f"- Release: `{C.RELEASE_DIR}/{C.RELEASE}/` (RELEASE.json, PRODUCT_CARD.md, "
           "REBUILD_CHECK.json, grid/v1, preorigin/*/v1, full/*/v1).",
           "- Code: `dataprep_staging/l3_release/` (config, io_det, components, build, verify, "
           "inventory, ledger_rows, report).",
           "- Evidence: `results/dataprep/reports/L3_evidence/` (inventory.json, compare.json, "
           "check_final.json, build logs).",
           f"- Ledger: `results/dataprep/ledger/rows/{C.LEDGER_WP}.csv`; assembled "
           "`results/dataprep/ledger/data_roles.csv`.",
           "- Resources: no GPU, 0 Claude API tokens, CPU work in the shared slots.", ""]
    (rp / "L3.md").write_text("\n".join(md), encoding="utf-8")
    print(f"status {status}; wrote {rp / 'L3.md'} and L3.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
