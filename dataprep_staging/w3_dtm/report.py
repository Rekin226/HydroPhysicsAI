"""Write the WP W3 report (``results/dataprep/reports/W3.md`` and ``W3.json``) and the W3 ledger
rows from the product and evidence files (continuation round, 2026-10-12).

Every number in the report is read here from one of:

* ``results/dataprep/ground_elevation/v3/validation.json`` and ``MANIFEST.json`` (the product);
* ``results/dataprep/reports/W3_evidence/diagnostics_v3.json`` (after-the-fact diagnostics,
  first round, ``diagnostics.py``);
* ``results/dataprep/reports/W3_evidence/continuation/verify_v3.json`` (``verify.py``);
* ``results/dataprep/reports/W3_evidence/continuation/check_ground_elevation.json``
  (``python -m common.check``) and ``ground_elevation/SUPERSEDED.json`` (``supersede.py``);
* ``data_fetch_api/new_measurements/moi_dtm20m_2025/SOURCE.json`` (the D4 size check);
* ``results/dataprep/reports/W3_evidence/continuation/a1_record.json`` and the declaration it
  names (``a1_record.py``: the pre-registration and W3's looks under the A1 tool).

Usage: python -I report.py [--rows-dir SCRATCH --out-dir SCRATCH]   (dry run first)
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))

from common import ledger  # noqa: E402
from common.hashing import sha256_file  # noqa: E402
from common.paths import DATAPREP, LEDGER_ROWS, REPO, REPORTS_DIR, rel_to_repo  # noqa: E402
from common.source import check_source_file  # noqa: E402

WP = "W3"
PV = "ground_elevation/v3"
V3 = DATAPREP / "ground_elevation" / "v3"
EV = REPORTS_DIR / "W3_evidence"
CONT = EV / "continuation"

# ledger sources (one role per source per number; '#' names the part of a file that is used)
DTM = "data_fetch_api/new_measurements/moi_dtm20m_2025"
SRCJ = DTM + "/SOURCE.json"
LEV = "chou-shui-data/data/ls_cache/ls-wra-lsp-obs__choushui_panel.parquet"
LEV_IN = LEV + "#elevations_of_benchmarks_inside_the_held_out_block"
LEV_OUT = LEV + "#increments_of_benchmarks_outside_the_held_out_block"
LEV_NAME = LEV + "#benchmark_name"
GRID = "results/twin_forward/datum_gate_innov.npz"
SRTM = "results/twin/basemap.npz"
ST_XY = "AMP_V2/data/fan_stations.parquet#sid_LocationByTWD97"
ST_META = "AMP_V2/data/fan_stations.parquet#GroundHeight_WellElevation_BaseElevation"
PREREG = "results/dataprep/preregistration/W3.dtm_heldout_bias.json"
PV3 = "results/dataprep/ground_elevation/v3"
PROD = "results/dataprep/ground_elevation"
CODE = ["build.py", "dtm_io.py", "mosaic.py", "leveling.py", "validation.py", "prereg.py",
        "l0_source.py", "diagnostics.py", "verify.py", "supersede.py", "a1_record.py",
        "report.py"]
A1REC = "results/dataprep/preregistration/W3.dtm_heldout_bias.record.json"
UNITS = "results/dataprep/validation_units/v1/units.json"
RESID = "results/dataprep/ground_elevation/v3/benchmark_residuals_heldout.parquet"


def jl(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def f(x, nd=3, sign=True) -> str:
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def utc_mtime(p: Path) -> str:
    return dt.datetime.fromtimestamp(p.stat().st_mtime, dt.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows-dir", default=None, help="ledger rows directory (default: the real "
                    "one; a scratch directory for a dry run)")
    ap.add_argument("--out-dir", default=str(REPORTS_DIR), help="where W3.md/W3.json go")
    a = ap.parse_args(argv)
    rows_dir = Path(a.rows_dir) if a.rows_dir else LEDGER_ROWS
    out_dir = Path(a.out_dir)
    val = jl(V3 / "validation.json")
    man = jl(V3 / "MANIFEST.json")
    diag = jl(EV / "diagnostics_v3.json")
    ver = jl(CONT / "verify_v3.json")
    chk = jl(CONT / "check_ground_elevation.json")
    a1 = jl(CONT / "a1_record.json")
    a1doc = jl(REPO / a1["declaration"]["path"])
    a1ov = a1["a1_fresh_units_overlap"]
    a1T = a1["timeline_utc"]
    a1_reg = [x for x in a1["looks"] if not x["seeded"]]
    a1_unreg = [x for x in a1["looks"] if x["seeded"]]
    a1_first = min(a1T[k] for k in ("first_scored_output", "diagnostics_v1"))[:19] + "Z"
    a1_last = a1T["verify_v3"][:19] + "Z"
    a1_decl = a1T["a1_fresh_units_declared"][:19] + "Z"
    a1_scopes = sorted({x["data_scope"] for x in a1["looks"]})
    sup = jl(DATAPREP / "ground_elevation" / "SUPERSEDED.json")
    src = jl(REPO / SRCJ)
    pre = jl(REPO / PREREG)
    src_chk = check_source_file(REPO / SRCJ)

    P, S, E, PS = (val["primary"], val["secondary"], val["epoch_correction_error"],
                   val["product_summary"])
    pb = P["per_benchmark"]
    pt = S["point_dtm_bilinear"]["per_benchmark"]
    sr = S["srtm_cell"]["per_benchmark"]
    cm = S["cell_median"]["per_benchmark"]
    un = S["uncorrected_cell_mean"]["per_benchmark"]
    bf = S["buffer_10km_cell_mean"]["per_benchmark"]
    cov = S["coverage_point_in_cell_90"]
    lobo = E["coverage_leave_one_block_out"]
    sc = src["size_check"]
    dec = diag["decomposition"]
    lw = diag["local_window"]
    st = diag["site_type_from_name"]
    hd = diag["dtm_provenance"]["height_datum"]
    sp = diag["spatial"]
    am = diag["agency_metadata_at_wells"]
    rk = ver["siting_ranks"]
    rb = ver["rebuild"]
    lk, lkb = ver["leak_test"], ver["leak_test_buffer"]
    raw = ver["raw_sheet_check"]
    ind = ver["independent_cell_means"]
    il = ver["independent_leveling"]
    gp = ver["guard_probes"]
    ms = {m["path"]: m for m in chk["manifests"]}
    v1 = ms["results/dataprep/ground_elevation/v1/MANIFEST.json"]
    v2 = ms["results/dataprep/ground_elevation/v2/MANIFEST.json"]
    v3c = ms["results/dataprep/ground_elevation/v3/MANIFEST.json"]
    t_pre = pre["created"]
    t_first_out = utc_mtime(DATAPREP / "ground_elevation/v1/ground_elevation_grid.npz")
    t_v = {v: utc_mtime(DATAPREP / f"ground_elevation/{v}/MANIFEST.json")
           for v in ("v1", "v2", "v3")}
    yr = S["per_year"]
    strata = S["strata_per_benchmark"]
    res = pd.read_parquet(V3 / "benchmark_residuals_heldout.parquet")
    yr_unc = {str(y): float(res.loc[res["year"] == y, "r_mean_uncorrected"].median())
              for y in (2012, 2016, 2020)}
    bm = res.groupby("sid").agg(r=("r_mean", "median"), blk=("block", "first"),
                                z=("cell_mean_sheet", "first"))
    bg = bm.groupby("blk").agg(n=("r", "size"), med=("r", "median"), z=("z", "median"))
    wb, bb = bg["med"].idxmin(), bg["med"].idxmax()
    prov = [(k, kk, vv["n"], vv["median"]) for k, v in diag["dtm_provenance"].items()
            if k != "height_datum" for kk, vv in v.items()]
    X = {"yr_unc": yr_unc,
         "worst": {"block": wb, "n": int(bg.at[wb, "n"]), "median": float(bg.at[wb, "med"]),
                   "cell_z": float(bg.at[wb, "z"])},
         "best": {"block": bb, "n": int(bg.at[bb, "n"]), "median": float(bg.at[bb, "med"]),
                  "cell_z": float(bg.at[bb, "z"])},
         "prov_all": [min(x[3] for x in prov), max(x[3] for x in prov)],
         "prov_n10": [min(x[3] for x in prov if x[2] >= 10),
                      max(x[3] for x in prov if x[2] >= 10)],
         "n_prov_groups": len(prov)}
    X["a1"] = {"doc_created": a1doc["created"][:19] + "Z", "rec": a1, "first": a1_first,
               "last": a1_last, "decl": a1_decl, "n_reg": len(a1_reg),
               "n_unreg": len(a1_unreg)}
    granted = sorted(u for u, o in gp.items() if o == "granted")
    gate_refused = sorted(u for u, o in gp.items() if o == "GateNotPassed")
    not_allowed = sorted(u for u, o in gp.items() if o == "UseNotAllowed")

    base = [(DTM, "input"), (GRID, "input")]
    heldout = base + [(LEV_IN, "validator"), (LEV_OUT, "input"), (PREREG, "input")]
    epoch = base + [(LEV, "input")]
    N = []

    def num(i, text, sources):
        N.append({"id": f"W3.{i}", "text": text, "sources": sources})

    # ---------------------------------------------------------------- L0 and product build
    num("l0.size_check",
        f"D4 size check before any archive download (HTTP HEAD on all 25 archives of the "
        f"index): {sc['all_archives_total_bytes'] / 1e9:.3f} GB in all (unsheeted island file "
        f"{sc['whole_island_unsheeted_bytes'] / 1e9:.3f} GB); downloaded 4 county archives "
        f"{sc['downloaded_archives_bytes'] / 1e6:.1f} MB (threshold about 1 GB)",
        [(SRCJ, "input")])
    num("mosaic.sheets",
        f"Mosaic: {PS['sheets_used']} sheets used, {PS['duplicate_sheets_identical']} "
        f"byte-identical duplicates read once ({PS['duplicate_sheets_different']} differing), "
        f"{PS['overlap']['n_points']:,} overlapping lattice points all equal (max |diff| "
        f"{PS['overlap']['max_abs_diff_m']} m); {PS['sheets_feeding_active_cells']} sheets "
        f"feed active cells, acquired {PS['active_cell_sheet_acquisition_range'][0]}.."
        f"{PS['active_cell_sheet_acquisition_range'][1]} (mosaic "
        f"{PS['mosaic_sheet_acquisition_range'][0]}..{PS['mosaic_sheet_acquisition_range'][1]}"
        f"); {PS['sheets_excluded_late']} sheets after 2022-12 needed", base)
    num("cells.coverage",
        f"Cells: {PS['n_cells']} active cells all have DTM points; {PS['cells_full_coverage']} "
        f"fully covered, {PS['cells_partial_coverage']} coastal cells partially (minimum "
        f"{PS['min_coverage']:.3f} of 2500 lattice points); {PS['cells_mixed_epochs']} cells "
        f"mix sheet epochs; cell epochs {PS['cell_epoch_range'][0]:.2f}.."
        f"{PS['cell_epoch_range'][1]:.2f}", base)
    num("cells.sd_within",
        f"Within-cell sd of the 20 m points (representativeness): median "
        f"{PS['sd_within_median_m']:.3f} m, p95 {PS['sd_within_p95_m']:.2f} m", base)
    num("epoch.corrections",
        f"Epoch correction (leveling increments 2011-2021, IDW k 8, power 2; "
        f"{PS['leveling_increments_dropped_as_resets']} increments dropped as resets): "
        f"|correction to 2017.5| max {PS['corr_ref_abs_max_m']:.3f} m (p95 "
        f"{PS['corr_ref_abs_p95_m']:.3f}); to 1 Jan 2012 {PS['corr_2012_range_m'][0]:+.3f}.."
        f"{PS['corr_2012_range_m'][1]:+.3f} m; to 1 Jan 2023 (extrapolated, flag 6) "
        f"{PS['corr_2023_range_m'][0]:+.3f}..{PS['corr_2023_range_m'][1]:+.3f} m; cell rate "
        f"2012-2021 {PS['rate_2012_2021_range_m_per_yr'][0] * 100:+.1f}.."
        f"{PS['rate_2012_2021_range_m_per_yr'][1] * 100:+.1f} cm/yr", epoch)
    num("zref.vs_srtm",
        f"Static field z_ref (2017.5): median {PS['z_ref_median_m']:.2f} m, minimum "
        f"{PS['z_ref_min_m']:.2f} m, {PS['cells_z_ref_below_0']} cells below 0 m and "
        f"{PS['cells_z_ref_below_2']} below 2 m (SRTM basemap: {PS['srtm_cells_below_0']} and "
        f"{PS['srtm_cells_below_2']}); SRTM minus z_ref median "
        f"{PS['srtm_minus_z_ref_median_m']:+.2f} m (IQR "
        f"{PS['srtm_minus_z_ref_iqr_m'][0]:+.2f}..{PS['srtm_minus_z_ref_iqr_m'][1]:+.2f})",
        epoch + [(SRTM, "input")])
    num("wells.coverage",
        f"Well points (H1b): {PS['stations']} stations, {PS['stations_with_dtm']} with a DTM "
        f"value ({PS['stations_xy_swapped_repaired_locally']} 'y x' coordinates swapped "
        f"locally, {PS['stations_xy_missing']} without coordinates); "
        f"{PS['model_wells_groundheight_zero_with_dtm']} of the "
        f"{PS['model_wells_groundheight_zero']} model wells with GroundHeight 0.0 (of "
        f"{PS['model_wells']}) get a value", base + [(ST_XY, "input"), (LEV, "input")])
    num("wells.metadata_vs_dtm",
        f"Station metadata minus DTM (epoch-corrected to 2017.5): GroundHeight (non-zero, n "
        f"{PS['groundheight_nonzero_n']}) median {PS['groundheight_nonzero_minus_dtm_median_m']:+.3f}"
        f" m (IQR {PS['groundheight_nonzero_minus_dtm_iqr_m'][0]:+.2f}.."
        f"{PS['groundheight_nonzero_minus_dtm_iqr_m'][1]:+.2f}); WellElevation (n "
        f"{PS['wellelevation_n']}) {PS['wellelevation_minus_dtm_median_m']:+.3f} m (IQR "
        f"{PS['wellelevation_minus_dtm_iqr_m'][0]:+.2f}.."
        f"{PS['wellelevation_minus_dtm_iqr_m'][1]:+.2f})",
        base + [(ST_XY, "input"), (ST_META, "validator"), (LEV, "input")])

    # ---------------------------------------------------------------- held-out validation
    num("primary.bias",
        f"PRIMARY (pre-registered, gate): median over {P['n_benchmarks']} held-out benchmarks "
        f"({P['n_observations']} observations 2012-2021, {P['n_blocks']} blocks of 10 km) of "
        f"(epoch-corrected cell mean minus benchmark) = {P['median_m']:+.3f} m; block-bootstrap "
        f"90 % interval {P['block_bootstrap_90ci_m'][0]:+.3f}..{P['block_bootstrap_90ci_m'][1]:+.3f}"
        f"; threshold |bias| <= {P['threshold_m']:.2f} m: FAILED", heldout)
    num("primary.spread",
        f"Primary residual spread per benchmark: mean {pb['mean']:+.3f} m, IQR "
        f"{pb['iqr'][0]:+.3f}..{pb['iqr'][1]:+.3f}, MAD-sd {pb['mad_sd']:.3f} m, "
        f"|r| > 1 m {pb['frac_abs_gt_1m']:.3f}, |r| > 3 m {pb['frac_abs_gt_3m']:.3f}; "
        f"{S['blocks_with_abs_median_gt_0.30']} of {P['n_blocks']} blocks have |median| > 0.30 m "
        f"(block medians {S['per_block_median_range_m'][0]:+.2f}..{S['per_block_median_range_m'][1]:+.2f}"
        f"; lowest block {X['worst']['block']}: {X['worst']['n']} benchmark(s), cell mean "
        f"{X['worst']['cell_z']:.0f} m)",
        heldout)
    num("secondary.variants",
        f"Secondary (pre-registered, never gating), median per benchmark: point DTM (bilinear, "
        f"epoch-corrected) {pt['median']:+.3f} m (MAD-sd {pt['mad_sd']:.3f}, n {pt['n']}); cell "
        f"median {cm['median']:+.3f}; uncorrected cell mean {un['median']:+.3f}; 10 km buffer "
        f"{bf['median']:+.3f} (median training set "
        f"{S['buffer_10km_cell_mean']['n_train_benchmarks_median']:.0f} benchmarks)", heldout)
    num("secondary.srtm",
        f"SRTM basemap cell value minus benchmark on the same {sr['n']} benchmarks: median "
        f"{sr['median']:+.3f} m, MAD-sd {sr['mad_sd']:.2f}, |r| > 3 m {sr['frac_abs_gt_3m']:.3f}"
        f" (DTM cell mean: MAD-sd {pb['mad_sd']:.2f}, |r| > 3 m {pb['frac_abs_gt_3m']:.3f})",
        [(SRTM, "input"), (GRID, "input"), (LEV_IN, "validator")])
    num("secondary.per_year",
        "Per year, DTM cell mean vs SRTM (median, per observation): " + "; ".join(
            f"{y} {yr[y]['cell_mean']['median']:+.3f} vs {yr[y]['srtm_cell']['median']:+.3f} m "
            f"(n {yr[y]['cell_mean']['n']})" for y in ("2012", "2016", "2020"))
        + "; uncorrected DTM cell mean " + ", ".join(
            f"{y} {X['yr_unc'][y]:+.3f}" for y in ("2012", "2016", "2020"))
        + " (plan facts, SRTM at the nearest centroid: +1.18, +1.23, +1.43)",
        heldout + [(SRTM, "input")])
    num("secondary.strata",
        "Strata (median per benchmark): " + "; ".join(
            f"{k} {v['median']:+.3f} (n {v['n']})" for k, v in strata.items()), heldout)
    num("secondary.coverage_sd_point",
        f"Coverage of value +- 1.645 sd_point on point-in-cell residuals: {cov['coverage']:.3f} "
        f"(cluster-adjusted Wilson 90 % {cov['wilson90'][0]:.3f}..{cov['wilson90'][1]:.3f}, "
        f"n_eff {cov['n_eff']:.0f}): plan 4.4 rule failed, sd_point uncalibrated", heldout)
    num("epoch.error",
        f"Epoch-correction error sigma = a |dt|, a = {E['a_m_per_yr']:.4f} m/yr from "
        f"{E['n']} held-out displacements; in-sample 90 % coverage "
        f"{E['coverage_of_1.645_a_dt']:.3f}; leave-one-block-out (added after registration, "
        f"reported only) {lobo['coverage']:.3f}, Wilson 90 % {lobo['wilson90'][0]:.3f}.."
        f"{lobo['wilson90'][1]:.3f}: plan 4.4 rule passed", heldout)
    num("prereg.timing",
        f"Pre-registration written {t_pre}; first scored output {t_first_out}; v1/v2/v3 "
        f"manifests {t_v['v1']} / {t_v['v2']} / {t_v['v3']}; v1 code = the code hashed in the "
        f"pre-registration ({sup['superseded'][0]['code_equals_preregistration_code']})",
        [(PREREG, "input"), (PROD, "input")])
    num("prereg.a1_record",
        f"A1 record (continuation, {a1doc['created'][:19]}Z): declaration "
        f"W3.dtm_heldout_bias.record (sha256 {a1['declaration']['sha256'][:12]}) puts the "
        f"legacy pre-registration into the A1 index; {len(a1['looks'])} looks logged "
        f"({len(a1_reg)} registered, logged late; {len(a1_unreg)} unregistered, back-dated to "
        f"their evidence files); every benchmark of A1's fresh.folds.leveling_all "
        f"({a1ov['fresh.folds.leveling_all']['scored_by_w3']}/"
        f"{a1ov['fresh.folds.leveling_all']['benchmarks']}) and fresh.folds.leveling_CH "
        f"({a1ov['fresh.folds.leveling_CH']['scored_by_w3']}/"
        f"{a1ov['fresh.folds.leveling_CH']['benchmarks']}) was scored by W3 ({a1_first}.."
        f"{a1_last}) before A1 declared them fresh ({a1_decl})",
        [(A1REC, "input"), (UNITS, "input"), (RESID, "input")])

    # ---------------------------------------------------------------- diagnostics
    num("diag.decomposition",
        f"After the fact: cell mean minus DTM point at the benchmarks median "
        f"{dec['cell_mean_minus_dtm_point_at_benchmarks']['median']:+.3f} m; benchmark vs the "
        f"maximum of the 3 x 3 (60 m) DTM window {lw['window_3x3']['r_local_max']['median']:+.3f}"
        f" m, vs its minimum {lw['window_3x3']['r_local_min']['median']:+.3f} m; "
        f"{lw['window_3x3']['frac_benchmark_above_local_max']:.3f} of benchmarks lie above the "
        f"3 x 3 maximum", [(PV3, "input")])
    num("diag.site_type",
        "After the fact, point residual by site type from the benchmark name: " + "; ".join(
            f"{k} {v['r_point']['median']:+.3f} m (n {v['r_point']['n']})"
            for k, v in st.items()), [(PV3, "input"), (LEV_NAME, "input")])
    num("diag.datum_label",
        "After the fact, point residual by the sheet header's height-datum label: " + "; ".join(
            f"{k} {v['median']:+.3f} m (n {v['n']})" for k, v in hd.items())
        + f"; neighbour correlation of point residuals within 2 km "
          f"{sp['nearest_neighbour_corr_within_2km']:+.3f} (n {sp['n_pairs']})",
        [(PV3, "input")])
    num("diag.provenance",
        f"After the fact, point residual by sheet agency, sensor, acquisition date and DTM "
        f"producer: group medians {X['prov_all'][0]:+.3f}..{X['prov_all'][1]:+.3f} m over "
        f"{X['n_prov_groups']} groups ({X['prov_n10'][0]:+.3f}..{X['prov_n10'][1]:+.3f} m for "
        f"groups of n >= 10)", [(PV3, "input")])
    num("diag.siting_ranks",
        f"After the fact (continuation): rank of the DTM value at the benchmark among its "
        f"cell's 20 m points median {rk['benchmark_dtm_point_rank_in_cell']['median']:.3f} "
        f"({rk['benchmark_dtm_point_rank_in_cell']['frac_above_cell_median']:.3f} above the "
        f"cell median; 0.5 expected for random siting); the benchmark elevation itself ranks "
        f"{rk['benchmark_elevation_rank_in_cell_at_sheet_epoch']['median']:.3f}; monitoring "
        f"stations {rk['station_dtm_point_rank_in_cell']['median']:.3f} (n "
        f"{rk['station_dtm_point_rank_in_cell']['n']}), model wells "
        f"{rk['model_well_dtm_point_rank_in_cell']['median']:.3f} (n "
        f"{rk['model_well_dtm_point_rank_in_cell']['n']})", [(PV3, "input")])
    num("diag.agency_vs_point",
        f"After the fact, agency metadata minus DTM point (sheet epoch): GroundHeight "
        f"{am['groundheight_nonzero_minus_dtm_point']['median']:+.3f} m (n "
        f"{am['groundheight_nonzero_minus_dtm_point']['n']}), WellElevation "
        f"{am['wellelevation_minus_dtm_point']['median']:+.3f} m, BaseElevation "
        f"{am['baseelevation_minus_dtm_point']['median']:+.3f} m (IQR "
        f"{am['baseelevation_minus_dtm_point']['iqr'][0]:+.2f}.."
        f"{am['baseelevation_minus_dtm_point']['iqr'][1]:+.2f})",
        [(PV3, "input"), (ST_META, "validator")])

    # ---------------------------------------------------------------- verification
    rbr = rb["residuals"]
    num("verify.rebuild",
        f"Continuation re-validation: full rebuild from L0 reproduces v3 exactly (mosaic max "
        f"|diff| {rb['H_max_abs_diff']}, cell mean/median/sd {rb['cell_mean_max_abs_diff']}/"
        f"{rb['cell_median_max_abs_diff']}/{rb['cell_sd_max_abs_diff']}, {rbr['rows_new']} "
        f"residual rows max |diff| {rbr['max_abs_diff']}, buffer variant "
        f"{rb['residuals_buffer']['max_abs_diff']}, bootstrap interval equal "
        f"{rb['bootstrap_equal_v3']}); rebuild {rb['mosaic_seconds']} s + validation "
        f"{rb['validation_seconds']} s", [(PV3, "input"), (DTM, "input"), (LEV, "input"),
                                          (GRID, "input"), (SRTM, "input")])
    num("verify.independent",
        f"Independent re-implementations: cell means by floor() assignment max |diff| "
        f"{ind['max_abs_diff_vs_v3_z_sheet_mean']} over {ind['cells']} cells (point counts equal "
        f"{ind['n_points_equal']}); {raw['sheets_checked']} random sheets parsed raw, "
        f"{raw['points_compared']:,} points vs the mosaic max |diff| {raw['max_abs_diff_m']} m "
        f"({raw['points_overwritten_by_later_sheet']:,} points superseded by later sheets); "
        f"leveling campaign elevations from the panel: {il['rows']} rows, "
        f"{il['rows_unmatched']} unmatched, max |diff| {il['max_abs_diff_m']} m",
        [(PV3, "input"), (DTM, "input"), (LEV, "input"), (GRID, "input")])
    num("verify.leak_test",
        f"Hold-out leakage test (+{lk['trend_m_per_yr']:.2f} m/yr synthetic trend on the "
        f"benchmarks of each block, validation rerun): {lk['blocks_tested']} blocks, max change "
        f"of the block's own epoch corrections {lk['max_abs_change_in_block_m']} m (the "
        f"perturbation moved other blocks' corrections in "
        f"{lk['blocks_where_perturbation_moved_other_blocks']} of {lk['blocks_tested']}); "
        f"buffer variant (trend within 10 km): {lkb['max_abs_change_in_block_m']} m, moved "
        f"others in {lkb['blocks_where_perturbation_moved_other_blocks']} of "
        f"{lkb['blocks_tested']}", [(PV3, "input"), (LEV, "input"), (DTM, "input"),
                                    (GRID, "input")])
    num("verify.guard",
        f"Loader guard on v3: granted {granted}; refused for the failed gate {gate_refused}; "
        f"refused as not allowed or forbidden {len(not_allowed)} uses", [(PV3, "input")])
    num("verify.checker",
        f"Checker: v3 {v3c['status']} ({v3c['n_hashed']} files re-hashed); v1 {v1['status']} and "
        f"v2 {v2['status']} (code edited in place after their build; every data output "
        f"byte-identical to v3, see SUPERSEDED.json); L0 SOURCE.json {src_chk['status']} "
        f"({src_chk['n_files']} files)", [(PROD, "input"), (SRCJ, "input")])

    # ---------------------------------------------------------------- ledger
    rows = []
    for n in N:
        for s, role in n["sources"]:
            rows.append({"number_id": n["id"], "source": s, "role": role,
                         "product_version": PV, "wp": WP, "note": n["text"][:160]})
    if a.rows_dir:                       # dry run: start from a copy of the real fragment
        rows_dir.mkdir(parents=True, exist_ok=True)
        real = ledger.fragment_path(WP)
        if real.exists() and not ledger.fragment_path(WP, rows_dir).exists():
            ledger.fragment_path(WP, rows_dir).write_text(real.read_text(encoding="utf-8"),
                                                          encoding="utf-8")
    lp = ledger.append_rows(WP, rows, rows_dir=rows_dir)
    w3_rows = [r for r in ledger.read_all(rows_dir) if r["wp"] == WP]
    missing = ledger.unledgered([n["id"] for n in N], rows_dir=rows_dir)
    conf = ledger.conflicts(w3_rows)

    # ---------------------------------------------------------------- acceptance
    acc = [
        {"criterion": "D4: download size checked before downloading; proceed only under about "
                      "1 GB",
         "threshold": "< ~1 GB, checked first",
         "measured": f"HEAD on all 25 archives first: {sc['all_archives_total_bytes'] / 1e9:.3f}"
                     f" GB in all; 4 county archives downloaded, "
                     f"{sc['downloaded_archives_bytes'] / 1e9:.3f} GB",
         "passed": sc["downloaded_archives_bytes"] < 1e9},
        {"criterion": "Aggregate the DTM to every active 1 km cell by mean and median with the "
                      "within-cell sd as representativeness error",
         "threshold": "all 2148 active cells",
         "measured": f"{PS['n_cells'] - ind['cells_without_points']}/{PS['n_cells']} cells; "
                     f"{PS['cells_full_coverage']} full, {PS['cells_partial_coverage']} partial "
                     f"(min {PS['min_coverage']:.2f}); sd_within median "
                     f"{PS['sd_within_median_m']:.2f} m",
         "passed": ind["cells_without_points"] == 0},
        {"criterion": "Epoch correction with cumulative leveling displacement between sheet "
                      "epoch and model epoch; its error model checked held out (plan 4.4)",
         "threshold": "Wilson 90 % interval contains 0.90 and point in 0.85-0.95",
         "measured": f"sd_epoch = {E['a_m_per_yr']:.4f} m/yr x |dt|; leave-one-block-out "
                     f"coverage {lobo['coverage']:.3f} (Wilson {lobo['wilson90'][0]:.3f}.."
                     f"{lobo['wilson90'][1]:.3f}); metric added after registration, reported "
                     f"only",
         "passed": bool(lobo["passes_rule_4.4"])},
        {"criterion": "PRIMARY GATE (plan WP W3, pre-registered W3.dtm_heldout_bias): |median "
                      "bias| of the epoch-corrected cell mean against leveling benchmarks held "
                      "out in 10 km blocks",
         "threshold": "|median| <= 0.30 m",
         "measured": f"{P['median_m']:+.3f} m (90 % block bootstrap "
                     f"{P['block_bootstrap_90ci_m'][0]:+.3f}..{P['block_bootstrap_90ci_m'][1]:+.3f}"
                     f"; {P['n_benchmarks']} benchmarks, {P['n_blocks']} blocks)",
         "passed": bool(P["passed"])},
        {"criterion": "Benchmarks used for the epoch correction are not the ones validated "
                      "(10 km block hold-out), verified by perturbation",
         "threshold": "0 change in a block's corrections when its own benchmarks change",
         "measured": f"{lk['max_abs_change_in_block_m']} m over {lk['blocks_tested']} blocks "
                     f"(buffer variant {lkb['max_abs_change_in_block_m']} m); the perturbation "
                     f"reached other blocks in {lk['blocks_where_perturbation_moved_other_blocks']}"
                     f"/{lk['blocks_tested']}",
         "passed": lk["max_abs_change_in_block_m"] == 0
         and lkb["max_abs_change_in_block_m"] == 0},
        {"criterion": "Well-point DTM elevation written to replace GroundHeight 0.0 (feeds H1b)",
         "threshold": "every model well with GroundHeight 0.0",
         "measured": f"{PS['model_wells_groundheight_zero_with_dtm']}/"
                     f"{PS['model_wells_groundheight_zero']}; {PS['stations_with_dtm']}/"
                     f"{PS['stations']} stations (flag 5; sd_m uncalibrated); usable only after "
                     f"a gate decision",
         "passed": PS["model_wells_groundheight_zero_with_dtm"]
         == PS["model_wells_groundheight_zero"]},
        {"criterion": "Within-cell sd as the error of a point in its cell (plan W3 Validation; "
                      "plan 4.4 coverage rule; secondary, not a gate)",
         "threshold": "Wilson 90 % interval contains 0.90 and point in 0.85-0.95",
         "measured": f"coverage {cov['coverage']:.3f} (Wilson {cov['wilson90'][0]:.3f}.."
                     f"{cov['wilson90'][1]:.3f}); sd_point labelled uncalibrated",
         "passed": bool(cov["passes_rule_4.4"])},
        {"criterion": "Pre-registration written before the held-out test (plan 4.5)",
         "threshold": "created before the first scored output; same design constants",
         "measured": f"{t_pre} < {t_first_out}; v1 code = registered code; recorded in the "
                     f"A1 index as W3.dtm_heldout_bias.record ({a1doc['created'][:19]}Z)",
         "passed": t_pre < t_first_out
         and sup["superseded"][0]["code_equals_preregistration_code"]
         and a1["declaration"]["load_prereg_ok"]},
        {"criterion": "No post-origin data anywhere (U3, U7)",
         "threshold": "fit_window_end <= 2022-12",
         "measured": f"fit_window_end {man['fit_window_end']} (leveling to 2021-05; sheets in "
                     f"the grid to {PS['active_cell_sheet_acquisition_range'][1][:6]})",
         "passed": man["fit_window_end"] <= "2022-12"},
        {"criterion": "Reproducible from L0 (re-validation in the continuation round)",
         "threshold": "rebuild equals v3",
         "measured": f"max |diff| 0 on mosaic, cells, {rbr['rows_new']} residual rows (both "
                     f"variants) and the bootstrap interval; independent cell means, raw sheets "
                     f"and leveling values also 0",
         "passed": rbr["max_abs_diff"] == 0 and rb["residuals_buffer"]["max_abs_diff"] == 0
         and rb["H_max_abs_diff"] == 0 and ind["max_abs_diff_vs_v3_z_sheet_mean"] == 0
         and raw["max_abs_diff_m"] == 0 and il["max_abs_diff_m"] == 0},
        {"criterion": "Manifest checker re-hashes every product version",
         "threshold": "all versions ok",
         "measured": f"v3 {v3c['status']}; v1 {v1['status']}, v2 {v2['status']} (code edited in "
                     f"place after their build; outputs byte-identical to v3; recorded in "
                     f"ground_elevation/SUPERSEDED.json)",
         "passed": all(x["status"] == "ok" for x in (v1, v2, v3c))},
    ]

    issues = [
        "Gate FAILED, so the loader refuses ground_elevation/v3 for forcing_frozen, "
        "forcing_recorded_arm, model_design and validator; only design_time_evidence and "
        "after_the_fact_analysis open. H1b (GroundHeight 0.0 replacement) and I1 (SRTM -> DTM "
        "swap) therefore wait for a recorded user decision. W3 did not loosen the threshold, "
        "re-define the statistic or build a rescue variant.",
        f"Decision material: the DTM cell mean sits {P['median_m']:+.2f} m below the benchmarks, "
        f"the DTM point {pt['median']:+.2f} m (MAD-sd {pt['mad_sd']:.2f}); SRTM sits "
        f"{sr['median']:+.2f} m above them (MAD-sd {sr['mad_sd']:.2f}). Benchmarks sit on "
        f"locally high ground (their DTM value ranks "
        f"{rk['benchmark_dtm_point_rank_in_cell']['median']:.2f} in their cell; bridges "
        f"{st['bridge_橋']['r_point']['median']:+.2f} m), so part of the gap is siting, not DTM "
        f"error. The remaining point offset (~{pt['median']:+.2f} m) cannot be split between "
        f"sub-20 m siting (benchmarks on structures the bare-earth grid smooths away) and a "
        f"vertical-datum difference between the LiDAR DTM and the WRA leveling network with the "
        f"data at hand; an independent bare-ground control set (for example MOI LiDAR check "
        f"points or e-GNSS stations) would separate them.",
        "Options for the user (not W3's call): (a) record a waiver and release v3 for "
        "forcing/model design with the bias disclosed (a new version carrying the decision); "
        "(b) pre-register a new test under a new id (any siting allowance chosen now is post "
        "hoc and must be labelled as such); (c) keep SRTM. The plan's impact note applies: "
        "1 m is about 3 % of the lift z - h + 40 m.",
        f"I1: swapping SRTM for z_ref lowers cell ground by a median "
        f"{PS['srtm_minus_z_ref_median_m']:.2f} m (IQR {PS['srtm_minus_z_ref_iqr_m'][0]:.2f}.."
        f"{PS['srtm_minus_z_ref_iqr_m'][1]:.2f}); cells below 0 m rise from "
        f"{PS['srtm_cells_below_0']} to {PS['cells_z_ref_below_0']} and below 2 m from "
        f"{PS['srtm_cells_below_2']} to {PS['cells_z_ref_below_2']}. Use value_ref (2017.5, "
        f"static) or value (1 Jan 2012..2023; 2022 and 2023 extrapolated, flag 6).",
        f"H1b: monitoring stations also sit above their cell's typical ground (DTM rank "
        f"{rk['station_dtm_point_rank_in_cell']['median']:.2f}), so use the well-point table "
        f"(well_ground_elevation.parquet, ground_elev_dtm_ref_m), not the cell value, for "
        f"GroundHeight. Agency GroundHeight (non-zero) is "
        f"{PS['groundheight_nonzero_minus_dtm_median_m']:+.2f} m above the DTM point at "
        f"2017.5 (construction-era surveys plus later subsidence, or the same offset as the "
        f"benchmarks); the table's sd_m is uncalibrated.",
        f"H1a: {PS['stations_xy_swapped_repaired_locally']} station coordinates stored as 'y x' "
        f"were swapped locally by W3 and {PS['stations_xy_missing']} stations have none; H1a "
        f"owns the official repair, after which the well-point table should be rebuilt.",
        "Uncertainty fields: sd_epoch passed the leave-one-block-out coverage check (added after "
        "registration, reported only); sd_point failed (0.689), and sd (cell mean) and the "
        "well-point sd_m have no independent check. All except sd_epoch are uncalibrated (plan "
        "4.7). None of them carries the systematic benchmark offset.",
        f"A1: W3's pre-registration (written {t_pre} by w3_dtm/prereg.py, about a minute "
        f"before the A1 tool existed) is now in preregistration/INDEX.jsonl through the "
        f"declaration W3.dtm_heldout_bias.record ({a1doc['created'][:19]}Z), and W3's "
        f"{len(a1['looks'])} looks are in looks.jsonl: {len(a1_reg)} registered scoring looks "
        f"(logged late, real times in the note) and {len(a1_unreg)} unregistered looks "
        f"back-dated to their evidence files (sd_epoch leave-one-block-out coverage, "
        f"diagnostics, siting ranks). Data scopes: {', '.join(a1_scopes)}.",
        f"A1's fresh leveling units are not unseen: all "
        f"{a1ov['fresh.folds.leveling_all']['benchmarks']} benchmarks of "
        f"fresh.folds.leveling_all and all {a1ov['fresh.folds.leveling_CH']['benchmarks']} of "
        f"fresh.folds.leveling_CH are among the {a1ov['w3_benchmarks']} benchmarks W3 scored "
        f"(elevations 2012-2021, displacements 2011-2021) between {a1_first} and {a1_last}, "
        f"before A1 declared them fresh at {a1_decl}. A1's count of 0 looks on fresh units "
        f"came from an incomplete log: W3's looks were not yet in it. The tool now refuses any "
        f"later test on those units with either W3 data scope. Whether the units stay fresh "
        f"for other uses of the leveling panel (for example a subsidence model scored on "
        f"leveling displacement) is A1's decision.",
        "Checker: ground_elevation v1 and v2 fail permanently (their build code was edited in "
        "place and not kept); every data output is byte-identical to v3 "
        "(ground_elevation/SUPERSEDED.json). Consumers open v3. Do not edit build.py, dtm_io.py, "
        "mosaic.py, leveling.py or validation.py, or v3 fails too: a change needs a v4 build.",
        "Leveling ends with the 2021 main campaign (panel's last survey 2021-09); epochs "
        "2022-01-01 and 2023-01-01 are linear extrapolations of the last four increments (flag "
        "6, never for calibration). If S5 (InSAR/GNSS epoch fields) is built, a v4 can switch "
        "the correction; its effect on the static field is small (|corr_ref| <= 0.06 m).",
        "Sheet-header quirks (L0): 28 of the 633 sheets used label the height datum "
        "'TWD97[2010]'; their point residual is "
        f"{hd['TWD97[2010]']['median']:+.3f} m vs {hd['TWVD2001']['median']:+.3f} m for "
        "TWVD2001-labelled sheets, so it is treated as a header typo (edge points match the "
        "neighbours exactly).",
    ]

    deviations = [
        "The pre-registration was written by the W3-local tool w3_dtm/prereg.py (the A1 tool "
        "did not exist yet); the continuation round put it into A1's INDEX.jsonl as the "
        "declaration W3.dtm_heldout_bias.record and logged W3's looks late in looks.jsonl "
        "(a1_record.py, following the W4 precedent), without changing the registered design.",
        "The product is ground_elevation/v3, not v1 as the plan names it: v1-v3 were built in "
        "the first round from the same inputs and design, with documentation-only edits to the "
        "builder in between. Editing the builder in place broke the v1/v2 code hashes; "
        "SUPERSEDED.json records it.",
        "The epoch correction uses WRA leveling (the plan allows S5 or leveling); S5 was not "
        "available. A time-varying z(t) was produced (optional in the plan): value at 1 January "
        "2012..2023.",
        "The leave-one-block-out coverage of sd_epoch was added after the pre-registration "
        "(reported only, never gating).",
        "Point in cell means the active cell containing the benchmark (pre-registered); the "
        "plan's SRTM facts used the nearest centroid within 0.71 km. The SRTM numbers reproduce "
        "the plan's within 0.01 m per year.",
        "The continuation round added verify.py (rebuild, independent checks, leakage test, "
        "siting ranks), supersede.py, a1_record.py and report.py; no product version was "
        "rebuilt and nothing "
        "was downloaded.",
    ]

    code_paths = [str(HERE.parent / c) for c in CODE]
    rep = {
        "wp": WP, "status": "partial", "date": "2026-10-12",
        "products": [rel_to_repo(V3)],
        "superseded_versions": ["results/dataprep/ground_elevation/v1",
                                "results/dataprep/ground_elevation/v2"],
        "gate": man["gate"],
        "acceptance": acc,
        "key_numbers": [n["text"] for n in N],
        "number_ids": [n["id"] for n in N],
        "prereg": PREREG, "prereg_sha256": sha256_file(REPO / PREREG),
        "ledger": {"fragment": rel_to_repo(lp), "rows_w3": len(w3_rows),
                   "unledgered_numbers": missing, "conflicts": conf},
        "evidence": [rel_to_repo(EV / "diagnostics_v3.json"),
                     rel_to_repo(CONT / "verify_v3.json"),
                     rel_to_repo(CONT / "verify_v3.log"),
                     rel_to_repo(CONT / "check_ground_elevation.json"),
                     rel_to_repo(CONT / "a1_record.json"), A1REC,
                     "results/dataprep/ground_elevation/SUPERSEDED.json"],
        "manifest_and_guard": {"v3_checker": v3c["status"], "v1_checker": v1["status"],
                               "v2_checker": v2["status"], "guard_probes": gp},
        "claude_api_tokens": 0, "gpu": "none",
        "deviations_from_plan": deviations,
        "issues_for_downstream": issues,
        "code_paths": code_paths,
    }
    (out_dir / "W3.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n",
                                         encoding="utf-8")
    write_md(rep, val, PS, P, S, E, sc, pt, sr, cm, un, bf, cov, lobo, yr, strata, dec, lw, st,
             hd, sp, am, rk, rb, lk, lkb, raw, ind, il, granted, gate_refused, t_pre,
             t_first_out, t_v, v1, v2, v3c, src_chk, out_dir, X)
    print(f"wrote {out_dir / 'W3.json'} and W3.md; ledger rows {len(w3_rows)}, "
          f"unledgered {missing}, conflicts {conf}")
    return 0


def table(rows, head) -> str:
    out = ["| " + " | ".join(h.replace("|", "\\|") for h in head) + " |",
           "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c).replace("|", "\\|") for c in r) + " |" for r in rows]
    return "\n".join(out)


def write_md(rep, val, PS, P, S, E, sc, pt, sr, cm, un, bf, cov, lobo, yr, strata, dec, lw, st,
             hd, sp, am, rk, rb, lk, lkb, raw, ind, il, granted, gate_refused, t_pre,
             t_first_out, t_v, v1, v2, v3c, src_chk, out_dir, X) -> None:
    pb = P["per_benchmark"]
    acc_rows = [(a["criterion"], a["threshold"], a["measured"],
                 "yes" if a["passed"] else "**no**") for a in rep["acceptance"]]

    def srow(name, d):
        return (name, f(d["median"]), f"{f(d['iqr'][0])}..{f(d['iqr'][1])}",
                f(d["mad_sd"], 2, False), f(d["frac_abs_gt_1m"], 2, False),
                f(d["frac_abs_gt_3m"], 3, False), d["n"])

    res_rows = [srow("**epoch-corrected cell mean (primary)**", pb),
                srow("cell median", cm), srow("uncorrected cell mean", un),
                srow("10 km buffer variant (cell mean)", bf),
                srow("point DTM (bilinear, epoch-corrected)", pt),
                srow("SRTM basemap cell value (current twin field)", sr)]
    yr_rows = [(y, f(yr[y]["cell_mean"]["median"]), f(X["yr_unc"][y]),
                f(yr[y]["srtm_cell"]["median"]), p,
                yr[y]["cell_mean"]["n"]) for y, p in (("2012", "+1.18 (n 618)"),
                                                      ("2016", "+1.23 (n 734)"),
                                                      ("2020", "+1.43 (n 795)"))]
    st_rows = [(k.replace("_", " "), f(v["median"]), v["n"]) for k, v in strata.items()]
    site_rows = [(k.split("_")[0] + (f" ({k.split('_', 1)[1]})" if "_" in k else ""),
                  f(v["r_point"]["median"]), f(v["r_mean"]["median"]), v["r_point"]["n"])
                 for k, v in st.items()]
    rk_rows = [("DTM value at the benchmark", rk["benchmark_dtm_point_rank_in_cell"]),
               ("benchmark elevation (moved to the sheet epoch)",
                rk["benchmark_elevation_rank_in_cell_at_sheet_epoch"]),
               ("DTM value at monitoring stations", rk["station_dtm_point_rank_in_cell"]),
               ("DTM value at the 158 model wells", rk["model_well_dtm_point_rank_in_cell"])]
    rk_rows = [(a, f(b["median"], 3, False), f"{b['iqr'][0]:.2f}..{b['iqr'][1]:.2f}",
                f(b["frac_above_cell_median"], 2, False), b["n"]) for a, b in rk_rows]
    rbr = rb["residuals"]
    txt = f"""# WP W3: ground elevation from the MOI 20 m DTM (report, 2026-10-12)

Status: **partial**. The product `results/dataprep/ground_elevation/v3/` is built, re-verified
and documented, but its single acceptance criterion **failed**. The pre-registered held-out
bias of the epoch-corrected 1 km cell mean against leveling benchmarks is
**{P['median_m']:+.3f} m**, with a 90 % block-bootstrap interval of
{f(P['block_bootstrap_90ci_m'][0])}..{f(P['block_bootstrap_90ci_m'][1])} m over
{P['n_benchmarks']} benchmarks in {P['n_blocks']} blocks of 10 km. The threshold is
|bias| <= 0.30 m. The threshold was not changed, and no rescue
variant was built.

- **What the gate failure means.** The loader guard refuses v3 for `forcing_frozen`,
  `forcing_recorded_arm`, `model_design` and `validator`. Only `design_time_evidence` and
  `after_the_fact_analysis` open. H1b (GroundHeight repair) and I1 (SRTM to DTM swap) wait
  for a recorded user decision.
- **How large the gap is, and how much is siting.** The field the twin uses today, SRTM, sits
  **{sr['median']:+.3f} m** above the same benchmarks with MAD-sd {sr['mad_sd']:.2f} m. The
  DTM cell mean sits {P['median_m']:+.2f} m below them with MAD-sd {pb['mad_sd']:.2f} m. The
  DTM value at the benchmark point sits {pt['median']:+.2f} m below, MAD-sd
  {pt['mad_sd']:.2f} m. After-the-fact diagnostics show that benchmarks sit on locally high
  ground: their DTM value ranks {rk['benchmark_dtm_point_rank_in_cell']['median']:.2f} among
  their cell's 20 m points, where 0.5 would be random. So about half of the gap is the cell
  mean lying below the benchmark site, not DTM error.
- **What stays unexplained.** The rest, about {abs(pt['median']):.2f} m, cannot be split with
  the data at hand. It is either siting finer than 20 m or a vertical-datum difference.
- **This round's checks.** The continuation round rebuilt everything from L0 and reproduced v3
  exactly (max |diff| 0). A perturbation test showed that the 10 km block hold-out does not
  leak: each block's corrections changed by 0 m when its own benchmarks changed.
- **Pre-registration index.** The pre-registration now sits in A1's hash-chained index
  (`W3.dtm_heldout_bias.record`) with W3's looks logged. Every benchmark of A1's fresh
  leveling folds is among those W3 scored before A1 declared them fresh (issue for A1).
- **Resources.** 0 Claude API tokens, no GPU. Long jobs ran through `tools/cpu_slot.sh`.
  Nothing under `hydrophysics/`, `tests/` or L0 was written in this round, and nothing was
  staged in git.

## What was done

1. **L0 and the size check (D4).**
   - HTTP HEAD requests on all 25 archives of the dataset's index came first, before any
     archive download: {sc['all_archives_total_bytes'] / 1e9:.3f} GB in all (the unsheeted
     island file alone is {sc['whole_island_unsheeted_bytes'] / 1e9:.3f} GB).
   - Only the four county archives covering the active cells were downloaded (Changhua,
     Yunlin, Nantou, Chiayi County; {sc['downloaded_archives_bytes'] / 1e6:.1f} MB), with the
     index CSV and the header schema.
   - The record is `data_fetch_api/new_measurements/moi_dtm20m_2025/SOURCE.json`: Government
     Open Data License v1.0, per-file sha256, header quirks. The checker reports it
     {src_chk['status']}.
   - The archives are read in memory (zipfile and numpy text parsing), never extracted or
     executed.
2. **Mosaic** (`mosaic.py`): a dense 20 m lattice in EPSG:3826 around the grid.
   - {PS['sheets_used']} sheets were used. {PS['duplicate_sheets_identical']} byte-identical
     duplicates across county archives were read once.
   - All {PS['overlap']['n_points']:,} overlapping lattice points agree exactly.
   - The {PS['sheets_feeding_active_cells']} sheets feeding active cells were acquired
     {PS['active_cell_sheet_acquisition_range'][0]}..{PS['active_cell_sheet_acquisition_range'][1]}
     (airborne LiDAR). No sheet acquired after 2022-12 lies in or near the lattice extent.
3. **Cell aggregation**: mean, median, within-cell sd, min/max/p05/p95 and the epoch mix for
   all {PS['n_cells']} active cells.
   - {PS['cells_full_coverage']} cells are fully covered and {PS['cells_partial_coverage']}
     coastal cells partially (minimum {PS['min_coverage']:.2f}).
   - The within-cell sd has a median of {PS['sd_within_median_m']:.2f} m and a p95 of
     {PS['sd_within_p95_m']:.2f} m.
4. **Epoch correction** (`leveling.py`). The data are the WRA leveling main campaigns
   (Apr-Jul) 2011-2021: annual increments per benchmark, with
   {PS['leveling_increments_dropped_as_resets']} resets of more than 0.30 m dropped,
   interpolated by IDW (k 8, power 2) and accumulated.
   - The outputs are `value` at 1 January 2012..2023, `value_ref` at 2017.5 and the
     uncorrected sheet-epoch aggregates.
   - The corrections are small. The correction to 2017.5 is at most
     {PS['corr_ref_abs_max_m']:.3f} m. The correction to 2012 is
     {f(PS['corr_2012_range_m'][0])}..{f(PS['corr_2012_range_m'][1])} m.
   - 2022 and 2023 are linear extrapolations (flag 6).
5. **Well points for H1b**: a bilinear DTM value at every station, epoch-corrected to 2017.5,
   with a local 100 m sd.
   - {PS['stations_with_dtm']} of {PS['stations']} stations have a value, including all
     {PS['model_wells_groundheight_zero']} model wells whose GroundHeight is 0.0.
6. **Pre-registration then test.**
   - `preregistration/W3.dtm_heldout_bias.json` was written at {t_pre} with its sha256 in
     the ledger.
   - The first scored output followed at {t_first_out}, built with the code hashed in the
     pre-registration.
   - The pre-registration lists what had been looked at before it was written. That
     includes a synthetic check that already showed a DTM-only siting offset at the
     benchmark locations: the cell mean is 0.42 m below the DTM point. The primary statistic
     was not changed in response.
7. **Continuation round (this session).**
   - `verify.py` ran a full rebuild, independent re-implementations, a leakage test, guard
     probes and the siting ranks.
   - `supersede.py` wrote `ground_elevation/SUPERSEDED.json` for v1 and v2.
   - `a1_record.py` put the pre-registration into A1's index (declaration
     `W3.dtm_heldout_bias.record`, {X['a1']['doc_created']}) and logged W3's
     {len(X['a1']['rec']['looks'])} looks in `looks.jsonl`.
   - The checker ran, and `report.py` wrote the ledger rows and this report.

## Acceptance

{table(acc_rows, ["Criterion", "Threshold", "Measured", "Pass"])}

The plan's own acceptance (row 4) failed. The other rows are the plan's technique steps and
the Phase-0 rules.

## Held-out validation results

All rows are per benchmark: the median of a benchmark's own observations, then statistics
over benchmarks. All variants are pre-registered secondary metrics; only the first row
gates.

{table(res_rows, ["Field minus benchmark", "Median (m)", "IQR (m)", "MAD-sd (m)", "|r| > 1 m",
                  "|r| > 3 m", "n"])}

Per year (per observation) against the plan's SRTM facts:

{table(yr_rows, ["Year", "DTM cell mean, corrected (m)", "Uncorrected (m)", "SRTM here (m)",
                  "Plan SRTM facts (m)", "n"])}

With the epoch correction the residual does not drift:
{', '.join(f(yr[y]['cell_mean']['median'], 2) for y in ('2012', '2016', '2020'))} m in 2012, 2016
and 2020. Without it, it drifts with subsidence
({', '.join(f(X['yr_unc'][y], 2) for y in ('2012', '2016', '2020'))} m), as the static SRTM
field does. So the bias is static, not an epoch effect. Its spatial spread is wide:
{S['blocks_with_abs_median_gt_0.30']} of {P['n_blocks']} blocks have |median| > 0.30 m. The
block medians run from {f(X['worst']['median'], 2)} m (block {X['worst']['block']}:
{X['worst']['n']} benchmark, in a foothill cell with a mean of {X['worst']['cell_z']:.0f} m) to
{f(X['best']['median'], 2)} m (block {X['best']['block']}: {X['best']['n']} benchmarks, cell mean
{X['best']['cell_z']:.0f} m).

Strata (median per benchmark):

{table(st_rows, ["Stratum", "Median (m)", "n"])}

Uncertainty fields (plan 4.4 / 4.7):

- **sd_epoch** = {E['a_m_per_yr']:.4f} m/yr × |dt|. Its leave-one-block-out coverage is
  {lobo['coverage']:.3f} (Wilson 90 % {lobo['wilson90'][0]:.3f}..{lobo['wilson90'][1]:.3f}),
  which passes. This check was added after registration and is reported only.
- **sd_point** (sd_within with sd_epoch) covers {cov['coverage']:.3f} of point-in-cell
  residuals (Wilson {cov['wilson90'][0]:.3f}..{cov['wilson90'][1]:.3f}), which fails. It is
  **uncalibrated**.
- **sd** (cell mean) and the well-point **sd_m** have no independent check, so both are
  **uncalibrated**.
- None of these fields includes the systematic benchmark offset.

## Why the gate fails (after-the-fact analysis, never fed back)

These numbers come from `W3_evidence/diagnostics_v3.json` (first round) and
`W3_evidence/continuation/verify_v3.json` (siting ranks). They explain the result. They do
not change the product, its gate or its threshold.

- **Decomposition.** Of the {P['median_m']:+.2f} m, the cell mean minus the DTM point at the
  benchmark accounts for {f(dec['cell_mean_minus_dtm_point_at_benchmarks']['median'])} m. The
  DTM point minus the benchmark accounts for {f(pt['median'])} m.
- **Benchmarks sit high in their cells.**

{table(rk_rows, ["Rank among the cell's 20 m points", "Median", "IQR", "Share above cell median",
                 "n"])}

- **Local relief.** Against the highest DTM value of the surrounding 3 × 3 (60 m) window, the
  benchmark is {f(lw['window_3x3']['r_local_max']['median'])} m off. Against the lowest it is
  {f(lw['window_3x3']['r_local_min']['median'])} m off.
  {lw['window_3x3']['frac_benchmark_above_local_max']:.2f} of benchmarks lie above the window
  maximum.
- **Site type** (from the benchmark name):

{table(site_rows, ["Site", "Point residual (m)", "Cell residual (m)", "n"])}

- **DTM provenance.** The point residual barely depends on the sheet's agency, sensor,
  acquisition date or producer. The group medians run {f(X['prov_all'][0])}..
  {f(X['prov_all'][1])} m over {X['n_prov_groups']} groups ({f(X['prov_n10'][0])}..
  {f(X['prov_n10'][1])} m for groups of at least 10 benchmarks; `dtm_provenance` in the
  diagnostics file). The sheets whose header labels the height datum `TWD97[2010]` give
  {f(hd['TWD97[2010]']['median'])} m against {f(hd['TWVD2001']['median'])} m for
  TWVD2001-labelled sheets. The correlation of point residuals between neighbours within
  2 km is {f(sp['nearest_neighbour_corr_within_2km'])}: there is no spatially coherent error
  field.
- **Agency metadata at wells** (DTM point at the sheet epoch): GroundHeight
  {f(am['groundheight_nonzero_minus_dtm_point']['median'])} m
  (n {am['groundheight_nonzero_minus_dtm_point']['n']}), WellElevation
  {f(am['wellelevation_minus_dtm_point']['median'])} m.
- **Reading.** Many benchmarks are on structures: by name, {st['bridge_橋']['r_point']['n']}
  are at bridges, {st['school_國小_國中']['r_point']['n']} at schools and
  {st['temple_宮_廟']['r_point']['n']} at temples. For the median benchmark,
  {rk['benchmark_dtm_point_rank_in_cell']['median'] * 100:.0f} % of its cell's 20 m points lie
  lower than the DTM at the benchmark site, so the cell-mean statistic carries a siting offset
  by construction. The remaining point offset of about
  {f(pt['median'], 2)} m has two candidate explanations:
  - sub-20 m siting, where the bare-earth grid smooths a raised structure away;
  - a vertical-datum difference between the LiDAR DTM and the WRA leveling network.

  The data at hand cannot separate them. An independent bare-ground control set would.

## Verification in the continuation round

| Check | Result |
|---|---|
| Rebuild from L0 with the v3 code | mosaic, epochs and sheet index identical; cell mean/median/sd max diff {rb['cell_mean_max_abs_diff']}/{rb['cell_median_max_abs_diff']}/{rb['cell_sd_max_abs_diff']}; {rbr['rows_new']} residual rows max diff {rbr['max_abs_diff']} (buffer variant {rb['residuals_buffer']['max_abs_diff']}); bootstrap interval identical ({rb['bootstrap_equal_v3']}); {rb['mosaic_seconds'] + rb['validation_seconds']:.0f} s |
| Cell means re-implemented (floor assignment of lattice points) | max diff {ind['max_abs_diff_vs_v3_z_sheet_mean']} over {ind['cells']} cells; point counts equal |
| Raw parse of {raw['sheets_checked']} random sheets vs the mosaic | {raw['points_compared']:,} points, max diff {raw['max_abs_diff_m']} m |
| Leveling campaign values re-derived from the panel | {il['rows']} rows, {il['rows_unmatched']} unmatched, max diff {il['max_abs_diff_m']} m |
| Hold-out leakage test (+{lk['trend_m_per_yr']:.2f} m/yr on a block's own benchmarks) | {lk['blocks_tested']} blocks: own-block correction change {lk['max_abs_change_in_block_m']} m; the perturbation moved other blocks in {lk['blocks_where_perturbation_moved_other_blocks']}/{lk['blocks_tested']} |
| Same, buffer variant (trend within 10 km) | own-block change {lkb['max_abs_change_in_block_m']} m; others moved in {lkb['blocks_where_perturbation_moved_other_blocks']}/{lkb['blocks_tested']} |
| Loader guard on v3 | granted: {', '.join(granted)}; GateNotPassed: {', '.join(gate_refused)}; every other use not allowed or forbidden |
| Manifest checker | v3 {v3c['status']} ({v3c['n_hashed']} files); v1 {v1['status']}, v2 {v2['status']} (code edited in place after the build) |
| A1 tool record | declaration in INDEX.jsonl ({X['a1']['rec']['declaration']['index_records']} record), load_prereg ok ({X['a1']['rec']['declaration']['load_prereg_ok']}); {X['a1']['n_reg']} registered + {X['a1']['n_unreg']} unregistered looks; probe tests on A1's fresh leveling folds with W3's scopes refused {len(X['a1']['rec']['blocking_probe']['blocked_by_unregistered_w3_looks'])}/{len(X['a1']['rec']['blocking_probe']['tests'])} |

## Manifests and versions

`ground_elevation/v3` is the product. Its manifest has fit_window_end 2021-05, layer L2 and
the gate failed.

- **Allowed uses.** Gated: `forcing_frozen`, `forcing_recorded_arm`, `model_design`,
  `validator`. Gate-exempt: `design_time_evidence`, `after_the_fact_analysis`.
- **Forbidden uses.** Scoring and calibration targets, baselines, datum fits, selection.
- **v1 and v2.** Both were built minutes before v3 from the same inputs. All seven of their
  data outputs are byte-identical to v3's; only `PRODUCT_CARD.md` and `validation.json`
  differ. Their manifests fail the checker because `build.py`, `mosaic.py` and
  `validation.py` were later edited in place and the earlier editions were not kept. v1's
  code hashes equal the pre-registration's. `ground_elevation/SUPERSEDED.json` records all
  this.

## Ledger

{rep['ledger']['rows_w3']} rows in `ledger/rows/W3.csv`, covering {len(rep['number_ids'])}
reported numbers plus the pre-registration and its A1 record. Unledgered numbers:
{len(rep['ledger']['unledgered_numbers'])}. Conflicts: {len(rep['ledger']['conflicts'])}.

The leveling panel plays two roles in the held-out numbers, so it is split by the part used:
the elevations of benchmarks inside the held-out block are the `validator`, and the
increments of benchmarks outside it are an `input`.

## Deviations from the plan

{chr(10).join('- ' + d for d in rep['deviations_from_plan'])}

## Issues for downstream

{chr(10).join('- ' + d for d in rep['issues_for_downstream'])}

## Code

{chr(10).join('- `' + rel_to_repo(c) + '`' for c in rep['code_paths'])}
"""
    (out_dir / "W3.md").write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
