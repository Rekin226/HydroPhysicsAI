"""WP S3 repair round 3 (S3r): markdown tables, the number file and ledger rows for the v3
round (diagnosis, design, registered re-test, products).

    python -I dataprep_staging/s3_gnss/v3/report_v3.py EVIDENCE_DIR TUNE_JSON [--ledger]

EVIDENCE_DIR = ``results/dataprep/reports/S3_evidence/repair_2026-10-18``. Prints the tables
used in ``results/dataprep/reports/S3.repair.md`` Parts 3-4, writes
``EVIDENCE_DIR/report_numbers_v3.json`` (every number quoted, with its id) and, with
``--ledger``, appends one ledger row per number and source to
``results/dataprep/ledger/rows/S3.csv``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common import ledger  # noqa: E402
from common.paths import REPO  # noqa: E402

PRE = "gnss_clean/preorigin/v3"
FULL = "gnss_clean/full/v3"
PRE_DIR = "results/dataprep/gnss_clean/preorigin/v3"
FULL_DIR = "results/dataprep/gnss_clean/full/v3"
V2_PRE_DIR = "results/dataprep/gnss_clean/preorigin/v2"
V1_PRE_DIR = "results/dataprep/gnss_clean/preorigin/v1"
V2_TRIALS = "results/dataprep/reports/S3_evidence/repair_2026-10-13/acceptance/" \
            "injection_trials_preorigin.csv"
RINGS = "chou-shui-data/data/ls_cache/clean"
PILES = ["chou-shui-data/data/ls_cache/ls-wra-dbm-obs__TKJS_281_29.parquet",
         "chou-shui-data/data/ls_cache/ls-wra-dbm-obs__NTUH_282_29.parquet",
         "chou-shui-data/data/ls_cache/ls-wra-dbm-obs__SJES.parquet",
         "chou-shui-data/data/ls_cache/ls-wra-dbm-obs__STES.parquet"]
TWIN = "results/twin_forward/datum_gate_innov.npz"
A2_KEYS = ("cm_removed_ngl_offfan", "cm_removed_wra_hf", "uncorrected",
           "cm_removed_pilot_onfan_fullband")
DIAG_PV = "gnss_clean/preorigin/v2 (failed v2 test; S3r diagnosis)"
DESIGN_PV = "gnss_clean/design (pre-origin, S3r design-time evidence)"


def f(x, nd=3):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return "nan" if not np.isfinite(x) else f"{x:.{nd}f}"
    return str(x)


def _rel(p: Path) -> str:
    return Path(p).resolve().relative_to(REPO).as_posix()


def a1_table(v: dict, label: str) -> list[str]:
    a1 = v["A1_injected_steps"]
    out = [f"| {label} | family | n | recovered | found within 3 d | 2 cm | 3 cm | 5 cm | 10 cm "
           "| size err p90 (mm, found) | extra steps / trial | pass (>= 0.90) |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for fam in ("wra", "ngl"):
        s = a1[fam]
        bs = s.get("by_size", {})
        out.append(f"| | {fam} | {s['n']} | {f(s['recovered'])} | {f(s['found_within_3d'])} | "
                   + " | ".join(f(bs.get(k, {}).get("recovered")) for k in
                                ("2cm", "3cm", "5cm", "10cm"))
                   + f" | {f(s.get('size_err_mm_p90'), 1)} | "
                     f"{f(s.get('extra_steps_per_trial'), 2)} | {s['pass']} |")
    return out


def main(argv: list[str]) -> int:
    ev = Path(argv[0])
    tune_path = Path(argv[1])
    tune = json.loads(tune_path.read_text())
    tune_rel = _rel(tune_path)
    acc = ev / "acceptance"
    vp = json.loads((acc / "validation_preorigin.json").read_text())
    vf = json.loads((acc / "validation_full.json").read_text())
    nums: dict[str, object] = {}
    rows: list[dict] = []

    def num(nid: str, value, sources: list[tuple[str, str]], pv: str, note: str = ""):
        nums[nid] = value
        for s, role in sources:
            rows.append({"number_id": nid, "source": s, "role": role, "product_version": pv,
                         "note": note})

    md: list[str] = []
    # ------------------------------------------------------------------ Part 1 diagnosis
    diag = ev / "diag"
    an = json.loads((diag / "a1_anatomy.json").read_text())
    eb = json.loads((diag / "a1_error_budget.json").read_text())
    src_d = [(V2_TRIALS, "input"), (_rel(diag / "a1_trials_with_covariates.csv"), "input")]
    for fam in ("wra", "ngl"):
        a = an[fam]["all"]
        num(f"S3.v3.diag.A1v2.{fam}.fail_rate", a["fail"], src_d, DIAG_PV)
        num(f"S3.v3.diag.A1v2.{fam}.not_found_rate", a["not_found"], src_d, DIAG_PV)
        num(f"S3.v3.diag.A1v2.{fam}.size_fail_rate", a["size"], src_d, DIAG_PV)
        num(f"S3.v3.diag.A1v2.{fam}.by_sigma_hf", an[fam]["by_sigma_hf_mm"], src_d, DIAG_PV)
        num(f"S3.v3.diag.A1v2.{fam}.by_days_to_base_step", an[fam]["by_days_to_base_step"],
            src_d, DIAG_PV)
        num(f"S3.v3.diag.A1v2.{fam}.excursion_not_found", an[fam]["excursion_flagged_not_found"],
            src_d, DIAG_PV)
        src_n = [(_rel(diag / f"null_errors_{fam}.csv"), "input"), (V2_PRE_DIR, "input")]
        for k in ("null_sd_mm", "null_share_gt5", "z_null_share_gt_zmin",
                  "corr_station_share_null_vs_trials"):
            num(f"S3.v3.diag.null_v2.{fam}.{k}", eb[fam][k], src_n, DIAG_PV)
        nv = pd.read_csv(diag / f"null_vs_trials_{fam}.csv")
        num(f"S3.v3.diag.null_v2.{fam}.z_null_madsd_range",
            [round(float(nv.z_null_madsd.min()), 3), round(float(nv.z_null_madsd.max()), 3)],
            src_n, DIAG_PV)
    a2v = {}
    for ver in ("v1", "v2"):
        T = pd.read_csv(diag / f"a2_pairs_{ver}.csv")
        a2v[ver] = T
        src = [(_rel(diag / f"a2_pairs_{ver}.csv"), "input"),
               (V1_PRE_DIR if ver == "v1" else V2_PRE_DIR, "input"), (RINGS, "validator")] + \
            [(p, "validator") for p in PILES]
        for r in T.itertuples():
            num(f"S3.v3.diag.A2{ver}.{r.gnss}.{r.reference}.extras",
                {"ref": r.ref_extra, "raw_all": r.raw_all_extra,
                 "clean_flag01": getattr(r, "clean_flag01_extra", None),
                 "clean_flag013": getattr(r, "clean_flag013_extra", None),
                 "ngl_offfan_removed": getattr(r, "cmrm_ngl_offfan_extra", None),
                 "wra_hf_removed": getattr(r, "cmrm_wra_hf_extra", None)}, src, DIAG_PV)
    a2d = json.loads((diag / "a2_diagnosis.json").read_text())
    for ver in ("v1", "v2"):
        num(f"S3.v3.diag.A2{ver}.cm_own", a2d[ver]["cm_own"],
            [(_rel(diag / "a2_diagnosis.json"), "input"),
             (V1_PRE_DIR if ver == "v1" else V2_PRE_DIR, "input")], DIAG_PV)
    inv = pd.read_csv(diag / "a2_candidate_inventory.csv")
    num("S3.v3.diag.A2.inventory.ngl_nearest_ring_m",
        {r.station: int(r.ring_dist_m) for r in inv[inv.family == "ngl"].itertuples()},
        [(_rel(diag / "a2_candidate_inventory.csv"), "input")], DIAG_PV,
        "geometry only; no value read")
    # ------------------------------------------------------------------ Part 2 design
    dn = ev / "design_null"
    nm = json.loads((dn / "null_models.json").read_text())
    nc = json.loads((dn / "null_cm.json").read_text())
    nl = json.loads((dn / "null_level_cm.json").read_text())
    for fam in ("wra", "ngl"):
        num(f"S3.v3.design.null_models.{fam}", nm[fam], [(_rel(dn / "null_models.json"),
                                                          "design_time_evidence")], DESIGN_PV)
        num(f"S3.v3.design.null_cm.{fam}", nc[fam], [(_rel(dn / "null_cm.json"),
                                                      "design_time_evidence")], DESIGN_PV)
        num(f"S3.v3.design.null_level_cm.{fam}", nl[fam],
            [(_rel(dn / "null_level_cm.json"), "design_time_evidence")], DESIGN_PV)
    s1 = tune["stage1"]
    src_t = [(tune_rel, "design_time_evidence")]
    num("S3.v3.design.stage1.grid", s1["grid"], src_t, DESIGN_PV)
    num("S3.v3.design.stage1.chosen", s1["chosen"], src_t, DESIGN_PV)
    num("S3.v3.design.stage2.grid", tune["stage2_grid"], src_t, DESIGN_PV)
    num("S3.v3.design.chosen_params", tune["chosen_params"], src_t, DESIGN_PV)
    num("S3.v3.design.reference", tune["reference"], src_t, DESIGN_PV)
    num("S3.v3.design.reference_full", tune["reference_full"], src_t, DESIGN_PV)
    num("S3.v3.design.hp_window", {"window": tune["chosen_params"]["hp_window"],
                                   "by": tune["cm_evidence"]["hp_window_selected_by"]},
        src_t, DESIGN_PV)
    md += ["## Design run 2, stage 1 (seed 515)", "",
           "| win | z_min | delta_min (mm) | 2+3 cm | 2 cm | 3 cm | WRA | NGL | found | "
           "false / off-fan station-yr | WRA base steps |", "|---|---|---|---|---|---|---|---|---"
           "|---|---|"]
    for r in s1["grid"]:
        md.append(f"| {r['win']} | {r['z_min']} | {1000 * r['delta_min']:.0f} | "
                  f"{f(r['recovered_2_3cm'])} | {f(r['recovered_2cm'])} | "
                  f"{f(r['recovered_3cm'])} | {f(r['recovered_wra'])} | "
                  f"{f(r['recovered_ngl'])} | {f(r['found_within_3d'])} | "
                  f"{f(r['false_per_station_year_offfan'])} | {r['wra_detected_steps_base']} |")
    md += ["", "## Design run 2, stage 2 (seed 616)", "",
           "| size model | recovered | WRA | NGL | 2 cm | 3 cm | 5 cm | 10 cm | size err sd "
           "(mm) | p90 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in tune["stage2_grid"]:
        b = r["by_size"]
        md.append(f"| order {r['size_order']}, +-{r['local_half']} d | {f(r['recovered'])} | "
                  f"{f(r['recovered_wra'])} | {f(r['recovered_ngl'])} | {f(b.get('2cm'))} | "
                  f"{f(b.get('3cm'))} | {f(b.get('5cm'))} | {f(b.get('10cm'))} | "
                  f"{f(r['size_err_mm_sd_found'], 2)} | {f(r['size_err_mm_p90_found'], 2)} |")
    md += ["", f"Chosen: {json.dumps(tune['chosen_params'])}", ""]
    # ------------------------------------------------------------------ A1 (registered)
    md += ["## A1 injected steps (v3, seed 707)", ""] + a1_table(vp, "pre-origin (gate)") + \
        [""] + a1_table(vf, "full records (reported)") + [""]
    tri = acc / "injection_trials_preorigin.csv"
    for var, v, pv, d in (("preorigin", vp, PRE, PRE_DIR), ("full", vf, FULL, FULL_DIR)):
        for fam in ("wra", "ngl"):
            s = v["A1_injected_steps"][fam]
            src = [(d, "input"), (_rel(acc / f"injection_trials_{var}.csv"), "input")]
            num(f"S3.v3.A1.{var}.{fam}.recovered", s["recovered"], src, pv,
                "synthetic steps seed 707 in the v3 product pipeline")
            num(f"S3.v3.A1.{var}.{fam}.n", s["n"], src, pv)
            num(f"S3.v3.A1.{var}.{fam}.found_within_3d", s["found_within_3d"], src, pv)
            num(f"S3.v3.A1.{var}.{fam}.size_err_mm_p90", s.get("size_err_mm_p90"), src, pv)
            num(f"S3.v3.A1.{var}.{fam}.pass", s["pass"], src, pv)
            for k, b in s.get("by_size", {}).items():
                num(f"S3.v3.A1.{var}.{fam}.recovered_{k}", b["recovered"], src, pv)
        g = v.get("A1_secondary_harness_family_G", {})
        for fam, s in g.items():
            num(f"S3.v3.A1G.{var}.{fam}.recovered", s["recovered"],
                [(d, "input"), (_rel(acc / f"injection_harness_g_{var}.csv"), "input")], pv,
                "harness family G r=20..29 (reported)")
            md.append(f"- harness family G r = 20..29, {var}, {fam}: recovered "
                      f"{f(s['recovered'])} of {s['n']} (found {f(s['found_within_3d'])}; dense "
                      f"enough {f(s.get('recovered_dense_enough'))})")
    tr = pd.read_csv(tri)
    md += ["", "A1 failure modes (pre-origin, v3):", ""]
    for fam, t in tr.groupby("family"):
        for a, g in t.groupby(t.abs_size_m.round(3)):
            nf = int((~g.found).sum())
            big = int((g.found & (g.size_err_m.abs() > 0.005)).sum())
            exc = int(((~g.found) & g.flagged_as_excursion).sum())
            md.append(f"- {fam} {a * 100:.0f} cm: {len(g)} trials, not found {nf} (excursion or "
                      f"transient {exc}), found with |size err| > 5 mm {big}")
            num(f"S3.v3.A1.preorigin.{fam}.{a * 100:.0f}cm.counts",
                {"n": len(g), "not_found": nf, "excursion": exc, "size_gt5": big},
                [(_rel(tri), "input")], PRE)
        f_ = t[t.found]
        num(f"S3.v3.A1.preorigin.{fam}.size_err_sd_mm", float(1000 * f_.size_err_m.std()),
            [(_rel(tri), "input")], PRE)
        st = t.groupby("station").recovered.mean().sort_values()
        md.append(f"- {fam} worst stations: " + ", ".join(f"{k} {x:.3f}" for k, x in
                                                         st.head(5).items()))
        num(f"S3.v3.A1.preorigin.{fam}.by_station", {k: round(float(x), 3) for k, x in
                                                     st.items()}, [(_rel(tri), "input")], PRE)
        if fam == "ngl":
            wo = float(t[t.station != "ERLN"].recovered.mean())
            num("S3.v3.A1.preorigin.ngl.without_ERLN_posthoc", wo, [(_rel(tri), "input")], PRE,
                "post-hoc subset, not a test")
            md.append(f"- ngl without ERLN (post-hoc subset, not a test): {wo:.3f}")
    md.append("")
    # ------------------------------------------------------------------ A2 (second look)
    md += ["## A2 2021 extra compaction (v3; second look on spent units)", "",
           "| variant | GNSS series | evaluable pairs | same sign | pooled ratio | pooled GNSS "
           "extra (mm) | / uncorrected | unique stations: n, same sign, ratio | pass |",
           "|---|---|---|---|---|---|---|---|---|"]
    srcA2 = [(RINGS, "validator")] + [(p, "validator") for p in PILES]
    for var, v, pv, d in (("preorigin", vp, PRE, PRE_DIR), ("full", vf, FULL, FULL_DIR)):
        a2 = v["A2_2021_extra"]
        for k in A2_KEYS:
            s = a2.get(k, {})
            u = s.get("unique_gnss", {})
            md.append(f"| {var} | {k} | {s.get('n_evaluable')} | "
                      f"{f(s.get('same_sign_fraction'), 3)} | "
                      f"{f(s.get('pooled_amplitude_ratio'), 3)} | "
                      f"{f(s.get('pooled_gnss_extra_mm'), 1)} | "
                      f"{f(s.get('ratio_to_uncorrected_pooled'), 2)} | "
                      f"{u.get('n')}, {f(u.get('same_sign_fraction'), 2)}, "
                      f"{f(u.get('pooled_amplitude_ratio'), 2)} | {s.get('pass')} |")
            for kk in ("same_sign_fraction", "pooled_amplitude_ratio", "pooled_gnss_extra_mm",
                       "ratio_to_uncorrected_pooled", "n_evaluable", "pass"):
                num(f"S3.v3.A2.{var}.{k}.{kk}", s.get(kk), [(d, "input")] + srcA2, pv,
                    "second look on spent units")
            num(f"S3.v3.A2.{var}.{k}.unique", u, [(d, "input")] + srcA2, pv)
        for col in ("ngl_offfan", "wra_hf", "wra_onfan_fullband"):
            ce = a2.get(f"cm_own_extra_{col}")
            if ce:
                num(f"S3.v3.A2.{var}.cm_own_extra.{col}", {k: ce.get(k) for k in (
                    "extra_mm", "other_years_sd_mm", "other_years_n", "rank_abs_of_year")},
                    [(d, "input")], pv)
        num(f"S3.v3.A2.{var}.pass", a2.get("pass"), [(d, "input")] + srcA2, pv,
            "second look on spent units")
    md += ["", "Pairs (v3 pre-origin variant):", "",
           "| kind | GNSS | reference | ref extra (mm) | ngl_offfan removed | wra_hf removed | "
           "uncorrected | on-fan removed |", "|---|---|---|---|---|---|---|---|"]
    for r in vp["A2_2021_extra"]["pairs"]:
        md.append(f"| {r['kind']} | {r['gnss']} | {r['reference']} | {f(r['ref_extra_mm'], 1)} "
                  f"| {f(r.get('cm_removed_ngl_offfan_extra_mm'), 1)} | "
                  f"{f(r.get('cm_removed_wra_hf_extra_mm'), 1)} | "
                  f"{f(r.get('uncorrected_extra_mm'), 1)} | "
                  f"{f(r.get('cm_removed_pilot_onfan_fullband_extra_mm'), 1)} |")
    # ------------------------------------------------------------------ A3
    a3 = vf["A3_excess_stability"]
    md += ["", "## A3 excess-rate stability (reported, v3 full)", "",
           "| GNSS | ring site | pre excess | post excess | change (mm/yr) | months pre/post |",
           "|---|---|---|---|---|---|"]
    for r in a3["table"]:
        md.append(f"| {r['gnss']} | {r['site']} | {f(r['pre_excess_mm_yr'], 1)} | "
                  f"{f(r['post_excess_mm_yr'], 1)} | {f(r['excess_change_mm_yr'], 1)} | "
                  f"{r['pre_n_months']}/{r['post_n_months']} |")
    t3 = pd.DataFrame(a3["table"])
    both = t3.dropna(subset=["pre_excess_mm_yr", "post_excess_mm_yr"])
    src3 = [(FULL_DIR, "input"), (RINGS, "input")]
    num("S3.v3.A3.median_pre_excess", a3["median_pre_excess_mm_yr"], src3, FULL)
    num("S3.v3.A3.median_post_excess", a3["median_post_excess_mm_yr"], src3, FULL)
    num("S3.v3.A3.n_pre", int(t3.pre_excess_mm_yr.notna().sum()), src3, FULL)
    num("S3.v3.A3.n_post", int(t3.post_excess_mm_yr.notna().sum()), src3, FULL)
    num("S3.v3.A3.n_both", int(len(both)), src3, FULL)
    num("S3.v3.A3.median_change_both", float(both.excess_change_mm_yr.median())
        if len(both) else None, src3, FULL)
    md.append(f"\nMedian excess: pre {f(a3['median_pre_excess_mm_yr'], 1)} "
              f"({int(t3.pre_excess_mm_yr.notna().sum())} sites), post "
              f"{f(a3['median_post_excess_mm_yr'], 1)} ({int(t3.post_excess_mm_yr.notna().sum())}"
              f" sites); {len(both)} sites in both, median change "
              f"{f(float(both.excess_change_mm_yr.median()) if len(both) else None, 1)} mm/yr.")
    # ------------------------------------------------------------------ CM checks
    md += ["", "## Common-mode scatter (median station residual sd, mm; v3)", "",
           "| variant | stations | none | ngl_offfan | wra_hf | wra_onfan_fullband (diag) |",
           "|---|---|---|---|---|---|"]
    for var, v, pv, d in (("preorigin", vp, PRE, PRE_DIR), ("full", vf, FULL, FULL_DIR)):
        s = v["cm_scatter_wra"]
        md.append(f"| {var} | WRA members | {f(s['none'], 2)} | {f(s['ngl_offfan'], 2)} | "
                  f"{f(s['wra_hf'], 2)} | {f(s['wra_onfan_fullband'], 2)} |")
        n = v["cm_scatter_ngl_fan"]
        md.append(f"| {var} | NGL fan (non-reference) | {f(n['none'], 2)} | "
                  f"{f(n['ngl_offfan'], 2)} | | |")
        num(f"S3.v3.cm_scatter.{var}.wra", s, [(d, "input")], pv)
        num(f"S3.v3.cm_scatter.{var}.nglfan", n, [(d, "input")], pv)
        num(f"S3.v3.pca.{var}", v["pca_wra"], [(d, "input")], pv)
    # ------------------------------------------------------------------ twin scores
    md += ["", "## Frozen forward archive against GNSS rates (after-the-fact, v3 full)", "",
           "| family | series | n | model R2 | RMSE (cm/yr) | bias | continue R2 / RMSE | "
           "zero R2 / RMSE |", "|---|---|---|---|---|---|---|---|"]
    for fam in ("wra", "ngl"):
        ts = vf[f"twin_scores_{fam}"]
        for k, s in ts.items():
            if not isinstance(s, dict) or "model_archive" not in s:
                continue
            ma, bc, bz = s["model_archive"], s["base_continue_2020_22"], s["base_zero"]
            md.append(f"| {fam} | {k} | {s['n_stations']} | {f(ma.get('r2'), 2)} | "
                      f"{f(ma.get('rmse'), 2)} | {f(ma.get('bias'), 2)} | "
                      f"{f(bc.get('r2'), 2)} / {f(bc.get('rmse'), 2)} | "
                      f"{f(bz.get('r2'), 2)} / {f(bz.get('rmse'), 2)} |")
            num(f"S3.v3.twin.{fam}.{k}", {"n": s["n_stations"], "model": ma, "continue": bc,
                                          "zero": bz},
                [(FULL_DIR, "validator"), (TWIN, "input")], FULL)
    # ------------------------------------------------------------------ products
    md += ["", "## v3 products", ""]
    for var, pv, d in (("preorigin", PRE, PRE_DIR), ("full", FULL, FULL_DIR)):
        if not (REPO / d).exists():
            md.append(f"- {var}: not built")
            continue
        for fam in ("wra", "ngl"):
            dd = pd.read_parquet(REPO / d / fam / "daily.parquet",
                                 columns=["station", "flag", "method", "date"])
            src = [(d, "input")]
            shares = {int(k): round(float(x), 4) for k, x in
                      dd.flag.value_counts(normalize=True).sort_index().items()}
            num(f"S3.v3.product.{var}.{fam}.flag_shares", shares, src, pv)
            num(f"S3.v3.product.{var}.{fam}.n_daily", int(len(dd)), src, pv)
            num(f"S3.v3.product.{var}.{fam}.n_stations", int(dd.station.nunique()), src, pv)
            md.append(f"- {var} {fam}: {dd.station.nunique()} stations, {len(dd)} daily rows, "
                      f"flag shares {shares}")
        st = pd.read_csv(REPO / d / "steps.csv")
        src = [(d, "input")]
        w = st[st.family == "wra"]
        yr = pd.to_datetime(w.date).dt.year
        cmo = st[st.basis == "common_mode_only"]
        num(f"S3.v3.steps.{var}", {"n": int(len(st)), "wra": int(len(w)),
                                   "wra_2020_2022": int((yr <= 2022).sum()),
                                   "wra_2023_2025": int((yr >= 2023).sum()),
                                   "common_mode_only": int(len(cmo))}, src, pv)
        md.append(f"- {var} steps: {len(st)} (WRA {len(w)}: {(yr <= 2022).sum()} in 2020-22, "
                  f"{(yr >= 2023).sum()} in 2023-25); common-mode-only {len(cmo)}")
        sta = pd.read_csv(REPO / d / "stations.csv")
        notes = sta.get("notes", pd.Series(dtype=str)).fillna("")
        dropped = [x for s_ in notes for x in s_.split(" | ") if "re-test" in x]
        exc = [x for s_ in notes for x in s_.split(" | ") if x.startswith("excursion")]
        num(f"S3.v3.notes.{var}", {"retest_dropped": len(dropped), "excursions": len(exc)},
            src, pv)
        md.append(f"- {var}: {len(dropped)} detected steps dropped by the re-test; "
                  f"{len(exc)} excursions flagged 9")
    for var, v in (("preorigin", vp), ("full", vf)):
        src = [(_rel(acc / f"validation_{var}.json"), "input")]
        num(f"S3.v3.run.{var}.seconds", v["seconds"], src, PRE if var == "preorigin" else FULL)
        num(f"S3.v3.run.{var}.cpu_env", v.get("cpu_env"), src,
            PRE if var == "preorigin" else FULL)
    (ev / "report_numbers_v3.json").write_text(json.dumps(nums, indent=1, ensure_ascii=False,
                                                          default=str))
    print("\n".join(md))
    if "--ledger" in argv:
        p = ledger.append_rows("S3", rows)
        print(f"\nledger: {len(rows)} rows -> {p}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
