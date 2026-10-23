"""Text of the A0 repair-round report (used by common.a0.r2_report)."""
from __future__ import annotations

from common.paths import REPO

_R = REPO.as_posix()

CODE = [
    "dataprep_staging/README.md", "dataprep_staging/.gitignore",
    "dataprep_staging/common/__init__.py", "dataprep_staging/common/paths.py",
    "dataprep_staging/common/hashing.py", "dataprep_staging/common/flags.py",
    "dataprep_staging/common/uses.py", "dataprep_staging/common/manifest.py",
    "dataprep_staging/common/guard.py", "dataprep_staging/common/ledger.py",
    "dataprep_staging/common/source.py", "dataprep_staging/common/gitinfo.py",
    "dataprep_staging/common/check.py", "dataprep_staging/common/tests/test_library.py",
    "dataprep_staging/common/tests/conftest.py",
    "dataprep_staging/common/a0/reorganize_pilots.py", "dataprep_staging/common/a0/l0_sources.py",
    "dataprep_staging/common/a0/pilot_manifests.py", "dataprep_staging/common/a0/rerun_pilots.py",
    "dataprep_staging/common/a0/ledger_rows.py", "dataprep_staging/common/a0/report.py",
    "dataprep_staging/common/a0/r2_window_audit.py",
    "dataprep_staging/common/a0/r2_pilot_manifests.py",
    "dataprep_staging/common/a0/r2_l0_registry.py",
    "dataprep_staging/common/a0/r2_rerun_chain.py", "dataprep_staging/common/a0/r2_rho_k.py",
    "dataprep_staging/common/a0/r2_rerun_gapfill_s3.py",
    "dataprep_staging/common/a0/r2_guard_probe.py",
    "dataprep_staging/common/a0/r2_ledger_rows.py", "dataprep_staging/common/a0/r2_report.py",
    "dataprep_staging/common/a0/r2_report_text.py",
]

DISPOSITIONS = [
    ("major", "datum_fit and baseline were not pre-origin uses (U3/U5 gap)",
     "fixed: both are in PRE_ORIGIN_USES (design_time_evidence too); the writer refuses a "
     "manifest that allows a pre-origin use the guard could never grant; observed products "
     "spanning 2012-2026 serve datum fits, baselines and calibration targets only through a "
     "declared time slice (window_end <= 2022-12, rows filtered, raw paths refused)"),
    ("minor", "guard escape hatches (preorigin=False, require_gate=False) unrecorded",
     "fixed: overrides only tighten (relaxing raises ValueError); verify=False is noted"),
    ("minor", "exemptions permissive ('*', optional non_exempt_fit_window_end, name check)",
     "fixed: explicit uses from EXEMPTABLE_USES only, non_exempt_fit_window_end required and "
     "<= origin, fit_window_end = max(non-exempt, data_end), every post-origin exempt input must "
     "list the use; pre-origin name check matches preorigin/pre_origin/pre-origin and is backed "
     "by the name-independent reachability check"),
    ("minor", "pilot input provenance partial (module-mediated reads not recorded)",
     "improved: physics_inversion_hooks records the files its repository loaders read by "
     "default (labelled 'via', possible inputs); directories read as a whole are tree-hashed "
     "(input_trees). Still partial and labelled so: hashes taken at A0 time, not run time"),
    ("minor", "landuse_crop_priors personal_data 'none' despite pump coordinates",
     "fixed in the r2 manifest (target.npz pump_xy, pump_hp, pump_town, pump_cls, pump_cell)"),
    ("minor", "checker blind to L0 directories without SOURCE.json",
     "fixed: the checker reports orphans (--fail-on-orphans); l0_registry/v1 hashes the 1016 "
     "orphans and the 29 files of the 2 legacy records; 0 orphans now"),
    ("minor", "derived files counted among the 754 data files; duplicates; mixed publishers",
     "clarified (L0 is not modified): 363 of 754 files are flagged derived in their records; "
     "6 duplicate groups listed; wra_reports_misc's licence string covers a Control Yuan "
     "document too (issue for downstream)"),
    ("minor", "reruns tested post-processing only",
     "extended: the pseudo-Changhua chain was rerun end to end from repository data; the "
     "gap-fill fitted step s3 (family S, fresh Kalman MLE) was rerun as a supplementary check"),
    ("minor", "ledger patterns, prefix collisions, unledgered numbers",
     "fixed: pattern sources refused, sources normalised, directory/file conflicts detected, "
     "every reported number has rows (unledgered() check before the report is written)"),
    ("minor", "blanket fit_window_end 2026-10 on pilots",
     "fixed: per-pilot audited windows and per-output windows; design-time evidence only where "
     "pre-origin"),
    ("minor", ".gitignore gaps; cpu_slot.sh home path",
     "gitignore extended (json, txt, html, gpkg, xlsx, geojson, ...); cpu_slot.sh is the main "
     "session's shared tool, in use by other agents, and was not edited (flagged again)"),
    ("minor", "SOURCE.json rewritten once; old-path README from an earlier template",
     "explained, no action: both happened in the first round before any reader; the repair "
     "round writes nothing into L0 (the registry lives under results/) and edits no "
     "first-round script (all new code is in r2_* files)"),
]


def _pilot_table(g: dict) -> str:
    lines = ["| pilot | fit_window_end | allowed uses | pre-origin outputs |", "|---|---|---|---|"]
    for p in g["index"]["pilots"]:
        lines.append(f"| {p['scout']} | {p['fit_window_end']} | {', '.join(p['allowed_uses'])} "
                     f"| {len(p['pre_origin_outputs']) or ''} |")
    return "\n".join(lines)


def structured(g: dict, b: dict) -> dict:
    acc = b["acceptance"]
    core = [x for x in acc if not x["criterion"].startswith("Supplementary")]
    sup = [x for x in acc if x["criterion"].startswith("Supplementary")]
    status = "passed" if all(x["passed"] for x in core) else "partial"
    rho = b["rho"]
    chain = g["chain"]["comparisons"]["d_pseudo_changhua.json"]
    cs = g["chk"]["summary"]
    sup_txt = ("A supplementary refit of the gap-fill pilot's fitted step (s3, family S, fresh "
               "Kalman MLE) reproduced pilot_S.parquet exactly." if sup and sup[0]["passed"]
               else f"The supplementary s3 refit did not match to 1e-6 ({b['s3txt']}); it is "
               "not a plan criterion and is reported as failed.")
    summary = (
        "Repair round after independent verification. The verifier's major issue is fixed: "
        "datum_fit and baseline (and design_time_evidence) are now pre-origin uses. The manifest "
        "writer refuses any allowed pre-origin use that the guard could never grant. Observed "
        "products spanning 2012-2026 can serve datum fits, baselines and calibration targets "
        "only through a declared time slice ending by 2022-12. Every guard override can now "
        "only tighten, exemptions must name their uses explicitly and carry "
        "non_exempt_fit_window_end, and read_table can no longer drop the flag filter. "
        f"The 15 pilot manifests were revised to r2 (r1 kept byte-identical as MANIFEST.r1.json) "
        f"with per-pilot fit windows from a data-window audit: only physics_inversion_hooks "
        f"(2022-12) and remediation_plan (2020-12) are pre-origin; head_gapfill_pilot and "
        f"landuse_crop_priors expose only their pre-origin outputs as design-time evidence; the "
        f"rest are after-the-fact analysis only. A new L0 registry (l0_registry/v1) hashes the "
        f"1016 L0 files no SOURCE.json covered, and the checker now reports orphans: "
        f"{cs['manifests']} manifests and {cs['sources']} SOURCE.json re-hash with 0 failures "
        f"and 0 orphans. The pseudo-Changhua chain was rerun end to end from repository data: "
        f"every intermediate is byte- or value-identical and d_pseudo_changhua.json matches to "
        f"{chain['max_rel_diff']:.1e}. One recorded patch was needed, for an unarchived "
        f"string conversion. The audit found that plan U2's rho(k) used 2023-26 heads; "
        f"recomputed on 2012-22 data it is {rho[20]['rmse_2012_2022']:.3f}/"
        f"{rho[15]['rmse_2012_2022']:.3f}/{rho[10]['rmse_2012_2022']:.3f} m at k=20/15/10, "
        f"1.1-1.4% above the plan's values. The ledger fragment was rewritten with explicit "
        f"sources ({len(g['a0rows'])} rows, every reported number covered, 0 conflicts), and "
        f"{g['tests'][0]}/{g['tests'][1]} library tests pass. {sup_txt} No GPU, no Claude API, "
        f"nothing under hydrophysics/ or tests/ touched, nothing staged in git.")
    key_numbers = [
        f"Checker r2: {cs['manifests']} manifests + {cs['sources']} SOURCE.json, "
        f"{cs['files_hashed']} file records + {len(g['trees'])} input trees "
        f"({b['n_tree_files']} files) re-hashed, 0 failures, 0 L0 orphans",
        "L0 registry v1: 1045 files, 841.8 MB (1016 orphans: cwa_codis_rain 800, "
        "heads_202210_202610 158, compaction 38, 20 root files; 29 legacy-record files)",
        "Pilot fit_window_end: physics_inversion_hooks 2022-12, remediation_plan 2020-12, "
        "landuse_crop_priors 2024-12 (3 pre-origin outputs), head_gapfill_pilot 2026-09 (22 "
        "pre-origin outputs), others 2025-05..2026-10",
        f"Physics chain end to end: d_pseudo_changhua.json max rel {chain['max_rel_diff']:.1e} "
        f"over {chain['n_numeric']} values; K_pump/G0/inputs/hindcast byte-identical; "
        f"{g['chain']['seconds_total']:.0f} s",
        f"rho(k) 2012-2022 vs 2012-2026: k=20 {rho[20]['rmse_2012_2022']:.4f} vs "
        f"{rho[20]['rmse_2012_2026']:.4f}; k=15 {rho[15]['rmse_2012_2022']:.4f} vs "
        f"{rho[15]['rmse_2012_2026']:.4f}; k=10 {rho[10]['rmse_2012_2022']:.4f} vs "
        f"{rho[10]['rmse_2012_2026']:.4f} m (full variant reproduces the archive byte for byte)",
        f"Gap-fill s3 family S refit: {b['s3txt']}",
        "First round (unchanged): TPC compare_annual 0 over 159; d_pseudo 1.6e-12 over 219; "
        "pilot_final_clean 0 over 671; L0 754 files / 278.1 MB / 29 sources (363 flagged derived)",
        f"Library: {g['tests'][0]}/{g['tests'][1]} tests pass, ruff clean; ledger "
        f"{len(g['a0rows'])} A0 rows, 0 conflicts, 0 unledgered numbers",
        "Claude API tokens: 0; GPU: none",
    ]
    deviations = [
        "Code lives in dataprep_staging/common/ (D12), not results/dataprep/_staging_code/; the "
        "plan's io.py is split into guard.py (loading) and manifest.py (writing).",
        "Large regenerable arrays (10 files, 0.80 GB) were moved, not dropped, and are listed "
        "with hashes and generating scripts (task adjustment).",
        "Downloads were copied, not moved, into L0 (task adjustment): 278.1 MB duplicated.",
        "L0 holds 29 sources against the plan's ~10 (every download included); 363 of the 754 "
        "files are derived records (flagged per file).",
        "The ledger is per-WP fragments (rows/<WP>.csv) with two extra columns (wp, note).",
        "Repair round: pilot directories are not v<N> product directories, so a pilot manifest "
        "revision renames the r1 file to MANIFEST.r1.json (bytes unchanged, hashed by r2) and "
        "writes r2 as MANIFEST.json; INDEX.json and the pilots README likewise (r1 copies kept).",
        "Repair round: design_time_evidence is treated as a pre-origin use (plan 4.5 read with "
        "U3/U7): post-origin pilots now allow after_the_fact_analysis only.",
        "Repair round: pre-convention L0 directories are read-only, so instead of writing a "
        "SOURCE.json into them A0 built results/dataprep/l0_registry/v1 (layer 'registry', a "
        "schema extension) whose inputs are those files with their sha256.",
        "Repair round: the A0 ledger fragment was rewritten (first-round rows used pattern "
        "sources the repaired ledger refuses); the r1 fragment is kept in ledger/superseded/.",
        "Repair round: manifest schema dataprep-manifest/2 adds non_exempt_fit_window_end, "
        "time_slicing, per-output fit_window_end and input_trees (r1 manifests still validate).",
        "Repair round: the end-to-end chain rerun needed one recorded patch to a_build_inputs.py "
        "(cell_county saved as a string array), reproducing a conversion the scout did outside "
        "the archived scripts; attempt 1 without it is kept as evidence.",
        "Repair round: two supplementary checks beyond the plan (gap-fill s3 refit, rho(k) on "
        "pre-origin months).",
        "First-round reports are kept as A0.r1.md and A0.r1.json.",
    ]
    issues = [
        "U7 for plan U2: rho(k) (sigma_obs of calibration targets) was computed from complete "
        "months of 2012-2026, i.e. with the 2023-26 test heads. Pre-origin values: "
        f"{rho[20]['rmse_2012_2022']:.3f} m (k=20), {rho[15]['rmse_2012_2022']:.3f} (k=15), "
        f"{rho[10]['rmse_2012_2022']:.3f} (k=10), {rho[5]['rmse_2012_2022']:.3f} (k=5), "
        f"{rho[3]['rmse_2012_2022']:.3f} (k=3). H2, A1 and I1 should use these (evidence: "
        "results/dataprep/reports/A0_evidence/repair/reruns/rho_k_result.json).",
        "Design pilots are post-origin through their inputs: design_physics_inversion, "
        "design_bayesian_fusion and design_ml_transfer used the TPC Changhua candidate series "
        "(coefficients fitted on 2016-01..2025-06), the canal-pilot deliveries (annual totals "
        "2012-2025 in every fold) and the LUIMAP 2024 prior. Design choices resting on them "
        "(plan 4.5: zones, burn-in, basis, prior log-sd, Design 1) must be labelled "
        "selection-contaminated or re-derived on pre-origin inputs (P1/W1 pre-origin variants) "
        "by P4/A1. physics_inversion_hooks itself (inversions e/f/g on twin inputs 2012-2022) is "
        "pre-origin.",
        "head_gapfill_pilot: the artificial-gap method ranking (s3/s3b/s5/s5b/s5c, incl. "
        "pilot_final_clean.csv) is pre-origin design evidence for H2; partial_month_error, the "
        "Feb-Mar 2026 daily bridge, fill_comb and the s8 test-period router check are "
        "post-origin.",
        "TPC and billing pilots are wholly post-origin (TPC 2016-01..2026-08; census to "
        "2025-07): mu_elec, the source separation and the Chow-Lin design must be refitted "
        "pre-origin by P1/P2a (U3), as the plan already requires.",
        "Guard usage: observed L1 products that span 2012-2026 must declare time_slicing with an "
        "honest params_fit_window_end (QC thresholds fitted on 2023+ data make slicing "
        "impossible); consumers of datum fits, baselines and calibration targets then open with "
        "window_end <= 2022-12. Exemptions need non_exempt_fit_window_end and explicit uses; "
        "overrides only tighten.",
        "physics_inversion_hooks: the scout converted cell_county to a string array outside the "
        "archived scripts; any reuse of a_build_inputs.py must add that conversion (see "
        "r2_rerun_chain.py). The archived theta matches the current batch-8 theta (value check), "
        "but batch 8 is provisional; if stage3_theta.json is rewritten the checker will flag "
        "the pilot's inputs.",
        "subsidence_gap_sources/out/dbm_daily_raw.csv contains a 2027-03 timestamp, an "
        "impossible date in the raw deep-benchmark feed (S2 data quality).",
        "L0: 363 of the 754 scout files are derived records (decoded LiCSAR subsets, "
        "concatenated or filtered MOA JSON, a parsed chunk list); 6 groups of byte-identical "
        "files (LiCSAR grid records, 3 identical catalogue HTML pages, moa_crop_038 p0.json = "
        "t.json, 3 identical moa_irrigation_stats CSVs, 3 identical hydroinfo pages); "
        "wra_reports_misc's licence string ('WRA website publication') also covers the Control "
        "Yuan document cy53001.pdf. Licences as before: yearbooks no copying, GADM "
        "non-commercial, taiwan-atlas and TPC d007021 unverified, LiCSAR Copernicus terms.",
        "Pre-convention L0 directories (cwa_codis_rain, heads_202210_202610, compaction, the 20 "
        "root files) still have no SOURCE.json; their owner should add one (cwa_codis_rain has "
        "per-file .meta.json sidecars with url, fetch time and sha256). Until then "
        "l0_registry/v1 hashes them.",
        "The registry manifest lists common/check.py and common/source.py as its code: editing "
        "either makes l0_registry/v1 fail the checker; build l0_registry/v2 after such an edit.",
        "Pilot provenance stays partial: hashes are of 2026-10-12, not pilot run time; bare "
        "roots in unresolved_input_patterns (e.g. billing's AMP_V2/data/) are unrecorded.",
        "Personal data: billing census/meters parquet, head_pumping_signatures kwh_matrix.npz "
        "and landuse_crop_priors target.npz hold pump coordinates or attributes; "
        "datagovtw_metadata and the TPC pilot's dl_meta hold publisher contacts; "
        "ls-wra-dbm-obs-stations.parquet holds contact names and phones. Never copy them.",
        "dataprep_staging/tools/cpu_slot.sh (main session's) hard-codes the absolute home path; "
        "fix before committing dataprep_staging (CLAUDE.md Rule 1). Committing it would also "
        "make CI collect common/tests (25 CPU-only tests, < 1 s).",
        "The checker now re-hashes 9 input trees (AMP_V2/data/wells, 475 MB, in 5 pilot "
        "manifests; heads_202210_202610 in 3) and the 842 MB registry: a cold run reads several "
        "GB; run it in a CPU slot.",
    ]
    products = [
        {"path": f"{_R}/results/dataprep/_pilots/",
         "layer": "pilot",
         "manifest": "results/dataprep/_pilots/<scout>/MANIFEST.json (r2, 15) + INDEX.json (r2); "
                     "r1 kept as MANIFEST.r1.json / INDEX.r1.json",
         "allowed_uses": "after_the_fact_analysis for all; design_time_evidence (pre-origin use) "
                         "only for physics_inversion_hooks, remediation_plan, and the listed "
                         "pre-origin outputs of head_gapfill_pilot (22) and landuse_crop_priors "
                         "(3)",
         "fit_window_end": "per pilot: 2020-12 .. 2026-10 (see INDEX.json)"},
        {"path": f"{_R}/results/dataprep/l0_registry/v1/",
         "layer": "registry",
         "manifest": "results/dataprep/l0_registry/v1/MANIFEST.json (inputs = 1045 L0 files "
                     "with sha256) + registry.json",
         "allowed_uses": "none (provenance and hashes only; every use forbidden)",
         "fit_window_end": "2026-10 (contents span to the fetch month)"},
        {"path": f"{_R}/data_fetch_api/new_measurements/ (29 sources "
                 "written in the first round; unchanged in the repair round)",
         "layer": "L0",
         "manifest": "SOURCE.json per source (dataprep-source/1)",
         "allowed_uses": "raw input for L1+ builders; untrusted data; licence per SOURCE.json",
         "fit_window_end": "n/a (raw; fetched 2026-10-12)"},
        {"path": f"{_R}/results/dataprep/ledger/",
         "layer": "ledger",
         "manifest": f"rows/A0.csv ({len(g['a0rows'])} rows, explicit sources) assembled into "
                     "data_roles.csv; superseded/A0.r1.csv",
         "allowed_uses": "data-role ledger (U6)",
         "fit_window_end": "n/a"},
    ]
    return {
        "wp": "A0", "date": "2026-10-12", "round": "repair", "status": status,
        "summary": summary, "acceptance": acc, "products": products,
        "code_paths": [f"{_R}/{c}" for c in CODE],
        "key_numbers": key_numbers, "deviations_from_plan": deviations,
        "issues_for_downstream": issues,
        "report_path": f"{_R}/results/dataprep/reports/A0.md",
        "verifier_dispositions": [{"severity": s, "issue": i, "disposition": d}
                                  for s, i, d in DISPOSITIONS],
        "pilot_windows": [{k: p[k] for k in ("scout", "fit_window_end", "allowed_uses",
                                             "pre_origin_outputs", "u3_u7")}
                          for p in g["index"]["pilots"]],
        "rho_k": b["rho"], "check_summary": g["chk"]["summary"],
        "physics_chain": {k: g["chain"][k] for k in ("passed", "all_intermediates_ok",
                                                     "seconds_total", "steps")},
        "guard_probes": g["probes"], "claude_api_tokens": 0,
        "first_round": {"report": "results/dataprep/reports/A0.r1.json"},
    }


def markdown(g: dict, b: dict, md_r1: str) -> str:
    res = structured(g, b)
    rho = b["rho"]
    head = md_r1.split("\n", 1)
    body_r1 = head[1] if len(head) > 1 else ""
    note = (
        "# WP A0: provenance, layout and pilot preservation (report, 2026-10-12)\n\n"
        f"Status after the repair round: **{res['status']}**. This file has two parts: the "
        "first-round report (below, unchanged; also kept as `A0.r1.md`) and the **Repair "
        "round** section at the end, which supersedes the first round wherever they differ: "
        "pilot fit windows and allowed uses, the guard and manifest rules, the ledger rows, the "
        "test count, and the issues for downstream WPs.\n\n"
        "---\n\n## First round (unchanged)\n")
    disp = "\n".join(f"| {s} | {i} | {d} |" for s, i, d in DISPOSITIONS)
    acc = "\n".join(f"| {x['criterion']} | {x['threshold']} | {x['measured']} | "
                    f"{'yes' if x['passed'] else 'NO'} |" for x in res["acceptance"])
    rho_rows = "\n".join(
        f"| {k} | {rho[k]['n_2012_2026']} | {rho[k]['rmse_2012_2026']:.4f} | "
        f"{rho[k]['n_2012_2022']} | {rho[k]['rmse_2012_2022']:.4f} | "
        f"{100 * rho[k]['rmse_rel_change']:+.1f} % |" for k in sorted(rho, reverse=True))
    chain = g["chain"]
    steps = ", ".join(f"{s['cmd'].split()[-2]} {s['seconds']:.0f} s" for s in chain["steps"])
    issues = "\n".join(f"- {x}" for x in res["issues_for_downstream"])
    devs = "\n".join(f"- {x}" for x in res["deviations_from_plan"][5:])
    return note + body_r1 + f"""

---

## Repair round (2026-10-12, after independent verification)

I fixed the verifier's one major issue and acted on every minor one. Nothing was
overwritten. Revised pilot manifests, a new registry product, a rewritten ledger fragment
and this section were added, and the first-round files are kept (`MANIFEST.r1.json`,
`INDEX.r1.json`, `README.r1.md`, `ledger/superseded/A0.r1.csv`, `A0.r1.md`, `A0.r1.json`). No
GPU and no Claude API were used, nothing under `hydrophysics/`, `tests/` or L0 was written,
and nothing was staged in git. All new code is in `dataprep_staging/common/` (library) and
`common/a0/r2_*.py`; no first-round one-off script was edited.

### Verifier findings and what changed

| severity | finding | disposition |
|---|---|---|
{disp}

### Library (staging library 0.2.0, manifest schema `dataprep-manifest/2`)

- **Pre-origin uses.** `datum_fit`, `baseline` and `design_time_evidence` are now
  pre-origin uses, alongside calibration, initial conditions, IDW fields, priors, prior
  widths, model design, selection and frozen forcing.
  - A per-well datum is a calibrated offset of the scored model.
  - A climatology baseline is a model fitted to observations.
  - Design-time evidence sets design choices for models scored on 2023-26.
  - A test-period reference statistic, such as an NSE denominator, is part of the score and
    is read under `scoring_target`.
- **Reachability.** The manifest writer refuses any allowed pre-origin use that the guard
  could never grant. The pre-origin name check accepts every spelling of the name.
- **Time slices.** A product declares
  `time_slicing = {{column, params_fit_window_end}}`. A consumer then opens it with
  `window_end <= 2022-12`, `read_table` returns rows up to that month only, and raw `path()`
  access is refused. This is how an observed L1 product that spans 2012-2026 can serve datum
  fits and calibration targets without leaking 2023-26 rows.
- **Per-output windows.** For pilots that mix windows, a pre-origin use can read only the
  outputs whose window ends by the origin.
- **Exemptions.**
  - Uses must be listed explicitly, never `*`. They can never include targets, fits or
    selection.
  - `non_exempt_fit_window_end` is required.
  - `fit_window_end` must equal max(non-exempt end, the exemptions' `data_end`).
  - Every post-origin exempt input must list the use.
- **Overrides only tighten.** `preorigin=False` on a pre-origin use raises, and so does
  `require_gate=False` on a gated use. `verify=False` is recorded in `Product.notes`.
- **Flag filter.** `read_table` always reads the flag column, so passing a column subset no
  longer bypasses U1 or U2.
- **Ledger.** Pattern sources are refused and sources are normalised. A directory now
  conflicts with a file inside it, and `unledgered()` finds reported numbers that have no
  row.
- **Checker.** It reports L0 orphans (`--fail-on-orphans`) and honours the L0 registry. It
  also re-hashes input trees.
- **Tests and lint.** The tests cover each of these rules: {g['tests'][0]}/{g['tests'][1]}
  pass (17 in the first round), and ruff is clean. `.gitignore` now also blocks JSON, text,
  HTML, geo and office formats.
- **Verifier probes** (`A0_evidence/repair/guard_probes.json`).
  - The writer refuses a manifest with `fit_window_end` 2025-06 that allows `datum_fit` or
    `baseline`, and the guard raises `LeakageError` if that writer check is bypassed.
  - The other loopholes now raise.
  - On the real pilots, `design_time_evidence` is refused for the TPC pilot, opens for
    `physics_inversion_hooks`, and exposes only the 22 pre-origin outputs of
    `head_gapfill_pilot`. Its `partial_month_error.csv` raises.

### Pilot manifests r2 and the data-window audit

`common/a0/r2_window_audit.py` collected two kinds of evidence for every pilot: the date
range of each tabular and array output (date columns only), and every 2023-2026 year
literal in its scripts (`A0_evidence/repair/window_audit.json`). I then read each pilot's
data path, counting inputs as U3 requires. Each r2 manifest records the result under
`parameters.fit_window_audit`, along with the basis and the U3/U7 status.

{_pilot_table(g)}

What the audit establishes:

- **`physics_inversion_hooks` is pre-origin.**
  - It reads twin inputs through `hydrophysics.twin.inputs`, whose default window is
    2012-01..2022-12.
  - It uses the stage-3 theta calibrated on that window.
  - The theta file was rewritten at 03:06, after the pilot ran at 00:41. However, the
    archived operator's T, S, f_sh, r_ret and rfrac equal the current values, and the rain
    table's sha256 equals the one recorded in the theta's meta.
- **`remediation_plan`** uses leveling from 2012, 2016 and 2020 only.
- **`head_gapfill_pilot` is mixed.**
  - The artificial-gap chain (s3/s3b/s5/s5b/s5c) runs on the 2012-22 matrix with the
    audit's 'hist' events. Its 22 outputs carry `fit_window_end` 2022-12.
  - The partial-month errors, the daily bridge and the test-period checks use 2023-26 heads.
- **`landuse_crop_priors`** uses the LUIMAP 2024 snapshot, the plan's listed U3 exemption,
  so its window is 2024-12. `target.npz`, `town_kwh.csv` and `heads_amp.csv` are built
  without the snapshot and are pre-origin.
- **The three design pilots are post-origin through their inputs.** They read the TPC
  Changhua candidates (fitted on 2016-01..2025-06), the canal-pilot deliveries (2012-2025
  totals) and the LUIMAP prior.

r2 also fixes `landuse_crop_priors` `personal_data`: its `target.npz` holds pump coordinates
and attributes. It also records the module-mediated inputs of `physics_inversion_hooks`
(entries marked `via`) and tree-hashes the directories that scripts read as a whole.

### Finding: plan U2's rho(k) used the 2023-26 heads

`s4b_partial.py` estimated the partial-month error from complete months of 2012-2026. That
weight of the calibration likelihood is a hyperparameter, which U7 forbids setting from 2023+
observations. I reran it twice from hash-checked copies:
- unmodified, which reproduces the archive byte for byte;
- keeping only days before 2023-01-01.

| k | n (2012-2026) | rmse (2012-2026) | n (2012-2022) | rmse (2012-2022) | change |
|---|---|---|---|---|---|
{rho_rows}

n counts error samples: two per complete month, from the first and the last k days. At
k = 10-20 the pre-origin values are 1.1-1.4 % higher, and at k = 3-5 they are 0.7 % higher. At
two decimals they are 0.33, 0.49 and 0.65 m (the plan has 0.33, 0.49 and 0.64). Downstream
should use the pre-origin values.

### L0 registry (`results/dataprep/l0_registry/v1/`)

1016 files under `data_fetch_api/new_measurements/` had no SOURCE.json:
- `cwa_codis_rain` (800 files): the CWA rain the twin now uses;
- `heads_202210_202610` (158): the 2023-26 test heads;
- `compaction` (38);
- 20 vendor-API root files.

The two legacy records (29 files) also carry no sha256. These directories are read-only for
this work, so the registry lists all 1045 files as manifest inputs with sha256, bytes, mtime
and a provenance note per group. It has layer `registry`, and every use is forbidden. The
checker re-hashes it, so a later change to any of these L0 files fails the check, and the
orphan count is 0. The station file with contact fields and the owner-only download log were
hashed, never parsed.

### Reruns

- **Pseudo-Changhua chain, end to end** (`common/a0/r2_rerun_chain.py`; {steps}).
  - Every intermediate was regenerated from repository data. `inputs_cwa.npz`,
    `hindcast_cwa.npz`, `K_pump_f32.npy`, `G0_f32.npy` and `c_residual_by_county.json` are
    byte-identical; `M_cwa` is identical.
  - `d_pseudo_changhua.json` matches to
    {chain['comparisons']['d_pseudo_changhua.json']['max_rel_diff']:.1e} over
    {chain['comparisons']['d_pseudo_changhua.json']['n_numeric']} values.
  - Attempt 1 ran the archived scripts unchanged. Every array of `inputs_cwa.npz` was
    value-identical, but `cell_county` was saved as an object array, which `b_operator.py`
    refuses. The archived file holds it as `<U3` and was written after both scripts were
    saved, so the scout converted it outside the archived code. Attempt 2 adds that
    conversion as a recorded patch.
  - Torch threads were set to 1 for the CPU budget.
- **Gap-fill fitted step** (supplementary, `common/a0/r2_rerun_gapfill_s3.py`). s3 for
  family S was refit with fresh global Kalman MLE fits and four realizations, and compared
  with `out/pilot_S.parquet`: {b['s3txt']}.
- **rho(k):** see the table above.

### Ledger

`rows/A0.csv` now has {len(g['a0rows'])} rows, each with one explicit source, covering all
{g['n_reported']}
numbers this report quotes. The first-round numbers include the L0 inventory, the unit tests
and the guard probe, which had no rows before. `data_roles.csv` is reassembled with 0
conflicts. The first-round fragment is in `ledger/superseded/A0.r1.csv`.

### Acceptance (WP as it now stands)

| Criterion | Threshold | Measured | Pass |
|---|---|---|---|
{acc}

### Deviations from the plan (repair round)

{devs}

### Issues for downstream WPs (current list; supersedes the first-round list)

{issues}

### Evidence (repair round)

All paths below are under `results/dataprep/reports/A0_evidence/repair/`:
- `window_audit.json`: the per-pilot window evidence;
- `check_r2_final.json` and `check_r2_final.out`: the final full checker run (`check_r2.*`:
  the same run before the ledger rewrite);
- `guard_probes.json`;
- `reruns/physics_chain_result.json` and `reruns/physics_chain/` (attempt 1:
  `reruns/physics_chain_attempt1*`);
- `reruns/rho_k_result.json` and `reruns/rho_k/`;
- `reruns/gapfill_s3_S_result.json` and `reruns/gapfill_s3_S/`.

Products: `results/dataprep/_pilots/*/MANIFEST.json` (r2), `_pilots/INDEX.json`,
`results/dataprep/l0_registry/v1/`, `results/dataprep/ledger/rows/A0.csv`.
"""
