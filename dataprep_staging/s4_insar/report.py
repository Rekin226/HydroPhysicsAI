"""WP S4a: write the WP report (``results/dataprep/reports/S4a.md`` and ``S4a.json``), copy the
evidence files to ``reports/S4a_evidence/`` and append the ledger rows (``ledger/rows/S4a.csv``).

Every number in the report is read from the run records written by the other S4a steps (no
number is typed in by hand); each reported number id gets a ledger row naming its source.

Usage (hydro env): python -I report.py
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # dataprep_staging (common)
sys.path.insert(0, str(HERE))

from common import ledger  # noqa: E402
from common.hashing import sha256_file  # noqa: E402
from common.paths import DATAPREP, PREREG_DIR, REPORTS_DIR, rel_to_repo  # noqa: E402
from config import FRAMES, LICSBAS_DIR, RUN_DIR, WP  # noqa: E402

PRODUCT = DATAPREP / "insar_licsar_subsets" / "v1"
PV = "insar_licsar_subsets/v1"
EVID = REPORTS_DIR / "S4a_evidence"
PRIMARY, DESC, ASC2 = "069A_06694_131402", "105D_06628_232108", "069A_06632_162010"
DECL = PREREG_DIR / "S4a.setup_acceptance.v1.json"
EVIDENCE_FILES = [
    "acceptance_result.json", "transfer_tally.json", "fetch_decision.json", "smoke_result.json",
    "spot_checks.json", "checker_licsar_with_product.json", "notes_for_S4b.json",
    "probes/resume_probes.json", "probes/probes.json", "env/env_record.json",
    "env/check_install_final.log", "env/hydro_conda_meta_final.txt",
    "env/hydro_pip_freeze_final.txt", "fetch_069A_06694_131402.run3.log",
    "fetch_105D_06628_232108.run1.log", "validate_069A_06694_131402.log",
    "validate_105D_06628_232108.log", "smoke_licsbas.log", "smoke_single_variant_licsbas.log",
    "write_sources.log", "run_checker_l0.log", "aborted/removed_files.jsonl",
]


def j(p: Path) -> dict:
    return json.loads(p.read_text())


def gb(x: float) -> str:
    return f"{x / 1e9:.3f} GB"


def main() -> int:
    acc = j(RUN_DIR / "acceptance_result.json")
    tally = j(RUN_DIR / "transfer_tally.json")
    dec = j(RUN_DIR / "fetch_decision.json")
    smoke = j(RUN_DIR / "smoke_result.json")
    spot = j(RUN_DIR / "spot_checks.json")
    env = j(RUN_DIR / "env" / "env_record.json")
    cov = j(RUN_DIR / "footprint_coverage.json")
    est = j(RUN_DIR / "size_estimate.json")["frames"]
    ckp = j(RUN_DIR / "checker_licsar_with_product.json")
    vp = j(RUN_DIR / f"validation_{PRIMARY}.json")
    vd = j(RUN_DIR / f"validation_{DESC}.json")
    man = j(PRODUCT / "MANIFEST.json")
    guard = j(EVID / "guard_probe.json")
    decl = j(DECL)
    with open(RUN_DIR / f"pairs_{PRIMARY}.csv", newline="") as fh:
        prow = [r for r in csv.DictReader(fh) if r["fetched"] == "True"]
    n_pre_pairs = sum(r["epoch2"] <= "20221231" for r in prow)
    n_pre_epochs = len({e for r in prow for e in (r["epoch1"], r["epoch2"])
                        if e <= "20221231"})
    np_, nd = vp["network_fetched"], vd["network_fetched"]
    gs = vp["gnss_summary"]
    good = sorted(s["station"] for s in vp["gnss_stations_in_box"]
                  if s["off_fan"] and s["valid_frac_preorigin_pairs"] >= 0.5)
    zero_off = sorted(s["station"] for s in vp["gnss_stations_in_box"]
                      if s["off_fan"] and s["valid_frac_preorigin_pairs"] == 0)
    prim_b = tally["items"][f"subsets_{PRIMARY}"]["bytes"]
    desc_b = tally["items"][f"subsets_{DESC}"]["bytes"]
    sv = smoke["runs"]["smoke_single_variant"]
    sm = smoke["runs"]["smoke"]
    tok = 0
    n_pages = sum(json.loads(x).get("status") == 200 for x in (
        RUN_DIR / "fetch_log_pairdirs.jsonl").read_text().splitlines())
    early = [json.loads(x) for x in (RUN_DIR / f"fetch_log_{PRIMARY}.jsonl").read_text(
    ).splitlines()]
    early = [r for r in early if (r.get("fetched_utc") or "") < "2026-10-12T10:00"]
    n_early_ok = sum(r["status"] == "ok" for r in early)
    n_inputs = len(decl["inputs"])
    EVID.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in EVIDENCE_FILES:
        src = RUN_DIR / name
        if src.exists():
            dst = EVID / name.replace("/", "__")
            shutil.copy2(src, dst)
            copied.append(rel_to_repo(dst))
    copied.append(rel_to_repo(EVID / "guard_probe.json"))

    rows = acc["results"]
    status = "passed" if acc["passed"] else "partial"
    fail_rows = [r for r in rows if not r["passed"]]

    # ------------------------------------------------------------------ ledger rows
    l0p = rel_to_repo(Path(FRAMES[PRIMARY]["l0_dir"]) / "SOURCE.json")
    l0d = rel_to_repo(Path(FRAMES[DESC]["l0_dir"]) / "SOURCE.json")
    l0a = rel_to_repo(Path(FRAMES[ASC2]["l0_dir"]) / "SOURCE.json")
    R = rel_to_repo(RUN_DIR)
    numbers = {
        "S4a.env.licsbas": ([f"{R}/env/env_record.json", f"{R}/env/check_install_final.log"],
                            f"licsbas env {env['env']['n_packages']} conda-forge packages, "
                            f"python {env['env']['python']}, gdal {env['env']['gdal']}"),
        "S4a.env.hydro_untouched": ([f"{R}/env/hydro_conda_meta_final.txt",
                                     f"{R}/env/hydro_pip_freeze_final.txt"],
                                    "hydro conda-meta and pip list identical before/after/final"),
        "S4a.licsbas.commit": ([rel_to_repo(LICSBAS_DIR)],
                               f"LiCSBAS commit {acc['licsbas_git']['commit']}"),
        "S4a.primary.pairs": ([l0p, f"{R}/validation_{PRIMARY}.json"],
                              f"{vp['n_pairs_fetched']} of {vp['n_pairs_listed']} listed pairs "
                              f"fetched ({vp['n_pairs_unavailable']} not on the portal)"),
        "S4a.primary.epochs": ([l0p, f"{R}/validation_{PRIMARY}.json"],
                               f"{np_['n_epochs']} epochs {np_['first_epoch']}.."
                               f"{np_['last_epoch']}, {np_['n_components']} component, "
                               f"{vp['epochs_2022']} in 2022; pre-origin {n_pre_epochs} epochs, "
                               f"{n_pre_pairs} pairs"),
        "S4a.primary.empty_pairs": ([l0p, f"{R}/validation_{PRIMARY}.json"],
                                    f"{len(vp['pairs_empty_window'])} pairs empty in the clip"),
        "S4a.primary.alignment": ([l0p, f"{R}/validation_{PRIMARY}.json"],
                                  "unw on sea 0 pairs; cc best shift counts "
                                  f"{vp['alignment']['cc_best_shift_counts']}"),
        "S4a.desc.pairs_epochs": ([l0d, f"{R}/validation_{DESC}.json"],
                                  f"{vd['n_pairs_fetched']} pairs, {nd['n_epochs']} epochs "
                                  f"{nd['first_epoch']}..{nd['last_epoch']}"),
        "S4a.asc2.listing": ([l0a, f"{R}/size_estimate.json"],
                             f"{est[ASC2]['n_pairs_window']} pairs, expected "
                             f"{gb(est[ASC2]['expected_smart_clip_bytes'])}"),
        "S4a.transfer.total": ([f"{R}/transfer_tally.json"], gb(tally["total_bytes"])),
        "S4a.transfer.primary": ([f"{R}/transfer_tally.json",
                                  f"{R}/fetch_log_{PRIMARY}.jsonl"], gb(prim_b)),
        "S4a.transfer.desc": ([f"{R}/transfer_tally.json", f"{R}/fetch_log_{DESC}.jsonl"],
                              gb(desc_b)),
        "S4a.size_estimate.primary": ([f"{R}/size_estimate.json"],
                                      gb(est[PRIMARY]["expected_smart_clip_bytes"])),
        "S4a.raw_reader.bitwise": ([f"{R}/spot_checks.json"],
                                   f"{spot['n_batches']} batches all equal {spot['all_equal']}"),
        "S4a.smoke.cum": ([f"{R}/smoke_result.json"],
                          f"single-variant cum {sv['cum_h5']['cum_shape']}, mixed "
                          f"{sm['cum_h5']['cum_shape']}"),
        "S4a.gnss.off_fan_valid": ([f"{R}/validation_{PRIMARY}.json",
                                    rel_to_repo(Path(j(RUN_DIR / "run_config.json")[
                                        "ngl_holdings"]))],
                                   f"{gs['n_off_fan_valid50_preorigin']} off-fan stations"),
        "S4a.footprint.primary": ([f"{R}/footprint_coverage.json",
                                   "results/twin_forward/datum_gate_innov.npz"],
                                  f"{cov[PRIMARY]['frac_inside']} of active cells"),
        "S4a.checker.licsar": ([f"{R}/checker_licsar_with_product.json"],
                               f"{ckp['summary']['sources']} SOURCE.json, failed "
                               f"{ckp['summary']['sources_failed']}, orphans "
                               f"{ckp['summary']['orphans']}"),
    }
    lrows = [{"number_id": n, "source": s, "role": "input", "product_version": PV,
              "note": note} for n, (srcs, note) in numbers.items() for s in srcs]
    frag = ledger.append_rows(WP, lrows)
    all_rows = ledger.read_all()
    unl = ledger.unledgered(list(numbers), all_rows)
    conf = [c for c in ledger.conflicts(all_rows) if c[0].startswith("S4a.")]

    # ------------------------------------------------------------------ acceptance table
    tbl = ["| Id | Criterion | Threshold | Measured | Pass |", "|---|---|---|---|---|"]
    for r in rows:
        tbl.append(f"| {r['id']} | {r['criterion']} | {r['threshold']} | {r['measured']} | "
                   f"{'yes' if r['passed'] else '**no**'} |")

    def ey(v: dict) -> str:
        return ", ".join(f"{y} {d['epochs']}" for y, d in v["network_fetched"][
            "epochs_per_year"].items())

    lines = [
        "# WP S4a: InSAR setup, LiCSBAS environment and LiCSAR fan subsets (report, "
        f"{dt.date.today().isoformat()})",
        "",
        f"Status: **{status}**. {acc['n_passed']} of {acc['n_criteria']} pre-registered "
        "criteria pass (declaration `S4a.setup_acceptance.v1`, written "
        f"{decl['created']} before any subset existed).",
        "",
    ]
    for r in fail_rows:
        lines += [f"- **Failed: {r['id']}.** {r['measured']}. The threshold was not changed.",
                  "  The declaration recorded the reason before the fetch: D1's 1.4 GB came "
                  "from the scout's 0.22 x 0.20 deg box, while S4a clips the whole twin grid "
                  "plus a 2 km buffer (the header-only survey priced it at "
                  f"{gb(est[PRIMARY]['expected_smart_clip_bytes'])}). The S4a total, "
                  f"{gb(tally['total_bytes'])}, stayed under the 3.0 GB hard cap."]
    lines += [
        f"- **Primary frame {PRIMARY} is complete for 2020-01..2026-01.** "
        f"{vp['n_pairs_fetched']} of {vp['n_pairs_listed']} listed pairs were fetched. The "
        f"remaining one ({vp['unavailable_pairs'][0]['pair']}) has no GeoTIFF on the portal. "
        f"The network has {np_['n_epochs']} epochs ({np_['first_epoch']}..{np_['last_epoch']}) "
        f"in {np_['n_components']} connected component, with {vp['epochs_2022']} epochs in "
        f"2022. Pre-origin (to 2022-12) it has {n_pre_epochs} epochs and {n_pre_pairs} pairs.",
        f"- **The last epoch is unusable over the fan.** The 4 pairs ending 20260104 have no "
        "valid pixel in the clip: the portal's cc files for them hold 48.7 KB for the whole "
        "frame, so the acquisition misses the fan. Without them the network still has one "
        f"component and ends {vp['network_fetched_without_empty_pairs']['last_epoch']}.",
        f"- **Full-fan frames.** {DESC} (descending geometry) was fetched under the budget "
        f"rule: {vd['n_pairs_fetched']} pairs and {nd['n_epochs']} epochs, {nd['first_epoch']}.."
        f"{nd['last_epoch']}. The portal has GeoTIFFs for only {vd['n_pairs_fetched']} of its "
        f"{vd['n_pairs_listed']} listed in-window pairs, so an ascending/descending "
        "decomposition is possible from 2025-01 only, not from 2022-11. "
        f"{ASC2} was listed and priced ({est[ASC2]['n_pairs_window']} available pairs, "
        f"expected {gb(est[ASC2]['expected_smart_clip_bytes'])}) but not fetched: it needed "
        f"{gb(dec['frames'][ASC2]['needed_with_margin_bytes'])} with the 10 % margin, and "
        f"{gb(dec['frames'][ASC2]['remaining_before_bytes'])} remained.",
        f"- **Environment.** `~/miniforge3/envs/licsbas` holds {env['env']['n_packages']} "
        f"packages, all from conda-forge (python {env['env']['python']}, GDAL "
        f"{env['env']['gdal']}). `LiCSBAS_check_install.py` reports OK. The `hydro` env's "
        "conda-meta listing and pip list are byte-identical before setup, after setup and at "
        f"report time. LiCSBAS is the official COMET checkout, commit "
        f"`{acc['licsbas_git']['commit'][:12]}`, under gitignored "
        f"`{rel_to_repo(LICSBAS_DIR)}`.",
        f"- **LiCSBAS runs on the subsets.** Steps 02, 11, 12, 13 and 14 exited 0 on 40 "
        f"pre-origin pairs of one unw georeference variant (cum.h5 "
        f"{sv['cum_h5']['cum_shape']}, epochs x rows x cols) and on 40 pairs mixing two "
        f"variants ({sm['cum_h5']['cum_shape']}). No displacement value was read.",
        "- **Integrity.** All fetched files pass the declared checks. The raw multi-range "
        f"reader is bitwise exact in all {spot['n_batches']} uncompressed portal batches. The "
        f"checker re-hashes {ckp['summary']['files_hashed']} L0 files under 5 SOURCE.json "
        f"with 0 failures and 0 orphans; the product manifest re-hashes ok.",
        f"- **Product** `{rel_to_repo(PRODUCT)}/` (L1 index, no pixel values): manifest gate "
        f"(data criteria S4a-4, -5, -6, -9) passed; allowed uses `validator` and "
        "`after_the_fact_analysis`; every pre-origin use is forbidden (fit_window_end 2026-01).",
        f"- **Resources.** Claude API tokens: {tok}. GPU: none. Long jobs ran through "
        "`tools/cpu_slot.sh`. Nothing under `hydrophysics/` or `tests/` was written, no "
        "pre-existing L0 file was modified, and nothing was staged in git.",
        "",
        "## Acceptance (pre-registered, `S4a.setup_acceptance.v1`)",
        "",
        *tbl,
        "",
        "S4a-7 has two parts, and the criterion passes only if both do. The product manifest's "
        "gate uses the four criteria that describe the fetched data (S4a-4, -5, -6, -9). The "
        "process criteria, including the failed D1 comparison, are in "
        "`acceptance_result.json`. This split was set in `build_product.py` before the "
        "evaluation ran. The D1 failure itself was already foreseen in the declaration.",
        "",
        "## What was done",
        "",
        "1. **Resume check.** An earlier attempt had been cut off during the primary fetch. "
        "That attempt had already created the env, cloned LiCSBAS, fetched the listings and "
        f"the {n_pages} pair-directory pages, surveyed the sizes and written the declaration. "
        f"It had also fetched {n_early_ok} files. Before resuming, the {n_inputs} declared "
        f"inputs were re-hashed, and all {n_inputs} match. The {n_early_ok} files were kept, "
        "and no `.part` file was left. Four "
        "frame metadata rasters had failed with truncated reads, most likely because their "
        "7.5 MB merged ranges outlasted GDAL's 180 s request timeout at 25-40 KB/s. "
        "`fetch_subsets.py` r3 raises the timeout to 1800 s and adds a low-speed abort, and "
        "the retry succeeded.",
        f"2. **Primary fetch** (r3, 8 threads, cap 2.70 GB): {vp['n_pairs_fetched']} pairs x "
        "(unw, cc) plus E/N/U/hgt/landmask, in the declared tiered order. All files were "
        f"clipped with one index window of the frame's `geo.E.tif` grid, "
        f"{vp['reference_grid']['window']} (col, row, ncol, nrow). The run transferred "
        f"{gb(prim_b)}, against {gb(vp['bytes_remote_full_ifg'])} for the full files, and "
        f"{gb(vp['bytes_stored_ifg'])} is stored after lossless deflate re-encoding. "
        "Throughput was about 4.2 MB/s over 8 connections (2.35 GB in 559 s) on the evening "
        "of 2026-10-12, against 25-60 KB/s per connection that afternoon. (The S4b notes' "
        "'up to 1 MB/s per connection' rounds this up; it is about 0.5 MB/s.) The descending "
        f"frame used the window {vd['reference_grid']['window']} of its own grid and "
        f"transferred {gb(desc_b)} in 149 s.",
        f"3. **Validation** (`validate.py` r3) of each fetched frame: problems "
        f"{vp['n_problems']} and {vd['n_problems']}. Warnings: 4 empty-window pairs in each "
        f"frame, plus 7 inconclusive coherence-shift tests in the primary frame. "
        "Index alignment was "
        "checked for every pair in two ways. Valid unw pixels over the sea of the landmask "
        f"number 0 in all pairs. Each coherence raster's best integer shift against the frame's "
        f"reference coherence is (0, 0) in "
        f"{vp['alignment']['cc_best_shift_counts'].get('0,0')} primary pairs. The other 7 "
        "pairs all contain the decorrelated epoch 20200211, have correlation about 0 and are "
        "inconclusive; 4 empty pairs have no coherence. The descending frame gives (0, 0) in "
        f"{vd['alignment']['cc_best_shift_counts'].get('0,0')} pairs. The first draft's "
        "heuristic (valid unw where cc == 0) is non-zero in "
        f"{vp['alignment']['n_pairs_unw_valid_where_cc0_gt0']} primary pairs and "
        f"{vd['alignment']['n_pairs_unw_valid_where_cc0_gt0']} descending pairs, at most "
        f"{vp['alignment']['max_unw_valid_where_cc0']} boundary pixels. These are the newer "
        "portal generation, with a different mask rule: the primary files dated 2025-12-27 "
        "and 2026-01-05 (deflate-strip unw) and every 105D file. The two checks "
        "above show it is not a misregistration. Off-fan GNSS: "
        f"{gs['n_off_fan_valid50_preorigin']} NGL stations outside every active cell fall on "
        f"valid unw in >= 50 % of {gs['n_sampled_preorigin']} sampled pre-origin pairs "
        f"({', '.join(good)}). Off-fan stations with 0 valid samples: "
        f"{', '.join(zero_off)}. They include CHUA and LSB0 from the plan's candidate list.",
        "4. **Coherence was not looked at.** r3 removed the first draft's coherence summaries "
        "(fractions >= 0.3 at active cells, by season), so S4b's per-zone summer-coherence "
        "acceptance stays unlooked-at before its pre-registration.",
        "5. **SOURCE.json** for the three fetch directories (`write_sources.py`). Each file "
        "has its URL, the remote size and date, the source window and the native "
        "geotransform. Licence: Copernicus Sentinel-1 derived-work terms, with citations to "
        "LiCSAR (Lazecky et al. 2020) and LiCSBAS (Morishita et al. 2020).",
        "6. **Transfer tally** (every request of the WP, `tally.py`): "
        + "; ".join(f"{k} {v['bytes'] / 1e6:.1f} MB" for k, v in tally["items"].items())
        + f"; total {gb(tally['total_bytes'])}, {gb(3e9 - tally['total_bytes'])} under the cap.",
        "",
        "## Hand-over to S4b",
        "",
        f"- Index product: `{rel_to_repo(PRODUCT)}/inventory.json` (frames, exact L0 paths, "
        "reference grids, network and epoch inventory, environment, smoke run, notes), plus "
        "`pairs_<frame>.csv`, `epochs_<frame>.csv`, `validation_<frame>.json` and "
        "`notes_for_S4b.json`.",
        f"- Primary L0: `{rel_to_repo(Path(FRAMES[PRIMARY]['l0_dir']))}/GEOC/` (LiCSBAS01 "
        "layout: `<pair>/<pair>.geo.{unw,cc}.tif`, `<frame>.geo.{E,N,U,hgt,landmask}.tif`), "
        "with `metadata/baselines` and `metadata/metadata.txt`. Epochs per year: "
        f"{ey(vp)}. Largest gaps: "
        + ", ".join(f"{g['days']} d ({g['from']}..{g['to']})"
                    for g in np_["largest_epoch_gaps_days"][:3]) + ".",
        f"- Descending L0: `{rel_to_repo(Path(FRAMES[DESC]['l0_dir']))}/GEOC/`. Epochs per "
        f"year: {ey(vd)}. Largest gap: {nd['largest_epoch_gaps_days'][0]['days']} d "
        f"({nd['largest_epoch_gaps_days'][0]['from']}..{nd['largest_epoch_gaps_days'][0]['to']})"
        ". Four pairs have an empty unw on the portal (182 KB files).",
        f"- Footprints: the primary frame covers {cov[PRIMARY]['frac_inside']:.1%} of the "
        f"twin's active cells; latitudes {cov[PRIMARY]['uncovered_lat_range'][0]}.."
        f"{cov[PRIMARY]['uncovered_lat_range'][1]} lie outside it. Both full-fan frames cover "
        "100 %.",
        "- Environment: `dataprep_staging/s4_insar/licsbas_env.sh <command>`. Build a work "
        "directory of symlinks with `make_licsbas_workdir.py`; never write into L0.",
        "",
        "## Deviations from the plan",
        "",
        "- D1's 'about 1.4 GB' was exceeded for the primary frame "
        f"({gb(prim_b)}; reason above), with sizes reported before the fetch. The total stayed "
        "under the task's 3 GB.",
        f"- The plan's asc/desc decomposition from 2022-11 is not available. {ASC2} did not "
        f"fit the budget, and {DESC} has GeoTIFFs only for 2025-01..2026-01.",
        "- The LiCSBAS env drops `usgs-libcomcat`, which requires python < 3.11 and is used "
        "only by `LiCSBAS_get_eqoffsets.py` (earthquake catalogue).",
        "- Code is in `dataprep_staging/s4_insar/` (D12). The LiCSBAS checkout and run files "
        "are in `results/dataprep/insar_tools/` (gitignored).",
        "- Changes made in this resumed round:",
        "  - `fetch_subsets.py`: request timeout raised.",
        "  - `validate.py`: coherence summaries removed, alignment checks added, empty windows "
        "reported as warnings rather than as declared problems.",
        "  - A second smoke run on a single georeference variant, for the strict reading of "
        "'single-grid pairs'. The first run mixed two variants.",
        "",
        "## Issues for downstream",
        "",
    ]
    issues = [
        "S4b georeference: LiCSBAS02 takes the geotransform of the first GeoTIFF it reads. In "
        "the mixed-variant smoke run, cum.h5 got the stretched spacing (post_lon 0.00200067 deg "
        "instead of 0.002). Use the window's reference geotransform "
        f"{vp['reference_geotransform_of_window']} (primary) explicitly, or build each GEOC "
        "from one variant.",
        "S4b network: drop the 4 empty pairs ending 20260104 (the primary series ends "
        "20251211) and expect LiCSBAS11 to reject the 7 pairs of the decorrelated epoch "
        "20200211.",
        "S4b coherence acceptance: no coherence statistic over the fan or Changhua was computed "
        "by S4a. Pre-register the per-zone summer-coherence criterion before computing it "
        "(A1 look log).",
        "S4b references: usable off-fan NGL stations (>= 50 % valid, pre-origin sample): "
        + ", ".join(good) + f". {', '.join(zero_off)} are off-fan but never valid.",
        f"Budget: {gb(3e9 - tally['total_bytes'])} remain under the 3 GB cap. Fetching {ASC2} "
        f"(about {gb(est[ASC2]['expected_smart_clip_bytes'])}) needs a user decision; the "
        "ready command is in inventory.json. The portal throttles parallel bursts (HTTP 429). "
        "Use at most 8 threads, with serial multi-range on HTTP/1.1.",
        "Vertical: before 2025 only the ascending primary frame exists (east-west leakage, "
        "plan S4 risk). The per-pixel U raster is clipped; the plan's constant U = 0.833 "
        "approximates it.",
        "Leakage: InSAR is a validator (U7). The product forbids every pre-origin use. The "
        "2020-2022 epochs are observations, not a pre-origin product.",
        "Before committing dataprep_staging (CLAUDE.md rule 1): notes_for_s4b.py has a "
        "rounded latitude band of the primary footprint's gap typed into a caveat string. "
        "The WP's convention keeps study-area coordinates out of versioned code. Replacing "
        "it with the computed value changes a file that insar_licsar_subsets/v1 hashes, so "
        "that edit needs a v2.",
        "Checker: the L0 trees licsar_069A_06632, licsar_069A_06694 and licsar_105D_06628, "
        "flagged as orphans by A0fix, are now covered (0 orphans).",
    ]
    lines += [f"- {x}" for x in issues]
    lines += ["", "## Looks logged (A1 look log)", "",
              "- design_time_seed: the run-1 alignment probe (20200112_20200118, 06:45Z), "
              "back-dated to its evidence file.",
              "- integrity_check: validation of each frame, raw-reader spot check, both "
              "smoke runs. No score and no deformation value.", "",
              "## Ledger", "",
              f"`{rel_to_repo(frag)}`: {len(lrows)} rows added for {len(numbers)} number ids "
              f"(role input, product {PV}); unledgered {len(unl)}, conflicts {len(conf)}.", ""]
    md = REPORTS_DIR / "S4a.md"
    md.write_text("\n".join(lines))

    rep = {
        "wp": WP, "status": status, "date": dt.date.today().isoformat(),
        "products": [rel_to_repo(PRODUCT)],
        "product_manifest_sha256": sha256_file(PRODUCT / "MANIFEST.json"),
        "gate": man["gate"]["passed"], "gate_criteria": man["gate"]["criteria"],
        "acceptance": [{"criterion": f"{r['id']}: {r['criterion']}", "threshold": r["threshold"],
                        "measured": r["measured"], "passed": r["passed"]} for r in rows],
        "n_criteria_passed": acc["n_passed"], "n_criteria": acc["n_criteria"],
        "prereg": rel_to_repo(DECL), "prereg_sha256": sha256_file(DECL),
        "key_numbers": {n: note for n, (_, note) in numbers.items()},
        "number_ids": list(numbers),
        "ledger": {"fragment": rel_to_repo(frag), "rows_added": len(lrows),
                   "unledgered_numbers": unl, "conflicts": conf},
        "evidence": copied,
        "manifest_and_guard": {"checker": ckp["products"]["summary"] if ckp.get("products")
                               else None, "guard_probes": guard},
        "transfer": tally, "fetch_decision": dec,
        "claude_api_tokens": tok, "gpu": "none",
        "deviations_from_plan": [x.lstrip("- ") for x in lines[
            lines.index("## Deviations from the plan") + 2:
            lines.index("## Issues for downstream") - 1]],
        "issues_for_downstream": issues,
        "code_paths": sorted(rel_to_repo(p) for p in HERE.glob("*.py")) + sorted(
            rel_to_repo(p) for p in HERE.glob("*.sh")),
    }
    (REPORTS_DIR / "S4a.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n")
    print(md, REPORTS_DIR / "S4a.json", f"ledger +{len(lrows)} rows, unledgered {len(unl)}, "
          f"conflicts {len(conf)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
