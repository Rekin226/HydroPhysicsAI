"""WP S3: build a ``gnss_clean`` product version v2 (v2 code, repair round 2026-10-12/13; plan WP
S3; layout plan 2.2, manifest 2.3).

    dataprep_staging/tools/cpu_slot.sh python -I dataprep_staging/s3_gnss/v2/build.py \
        --variant {preorigin,full} --tune TUNE_JSON --evidence EVIDENCE_DIR

v2 against v1: pre-origin station sets from criteria valid at the origin (``pipeline``), the
step re-test (``model``), the rule's ``wra_hf`` window (``tune.py`` r3), the full-variant
jump-cluster flag (``products``), A2 on each offered common mode (``validate``), gate from the
v2 pre-registration. The v1 versions and their code stay untouched.

Writes ``results/dataprep/gnss_clean/<variant>/v2/``:

* ``wra/daily.parquet``, ``wra/monthly.parquet`` (+ ``wra/ncku_daily.parquet`` and
  ``wra/ncku_monthly.parquet`` in the full variant: NCKU stations 2023-04..2025-03);
* ``ngl/daily.parquet``, ``ngl/monthly.parquet``;
* ``common_mode.parquet`` (daily) and ``common_mode_monthly.parquet``: separate series, never
  removed from ``value``;
* ``steps.csv``, ``stations.csv``, ``validation.json``, ``PRODUCT_CARD.md``, ``MANIFEST.json``.

``preorigin``: every series cut at 2022-12-31 before any fit (``fit_window_end`` 2022-12; the
pre-origin uses). ``full``: complete records (``fit_window_end`` 2026-09; validator, scoring
target and after-the-fact uses only; no time slice is declared because the per-station fits
see the whole record). The gate of both variants is the pre-origin acceptance (U7); the
full-period results are reported in ``validation.json`` but never gate.

The tables are rebuilt here and must match the hashes the validation run recorded.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))   # dataprep_staging
sys.path.insert(0, str(HERE))              # v2 modules first
import build_tables as BT  # noqa: E402
import io_gnss as io  # noqa: E402
import pipeline as P  # noqa: E402
import products as PR  # noqa: E402
import run_validation as RV  # noqa: E402
from common import manifest  # noqa: E402
from common.paths import rel_to_repo  # noqa: E402

PREREG_ID = "S3.gnss_clean_acceptance.v2"
VERSION = "v2"
CODE = ["io_gnss.py", "model.py", "pipeline.py", "products.py", "build_tables.py",
        "injection.py", "validate.py", "run_validation.py", "build.py", "tune.py",
        "units_v2.py", "prereg_v2.py", "record_looks_r3.py"]
#: v1 provenance the verifier asked to reference (minor issue): the gate of the v1 versions was
#: computed by the REGISTERED v1 validate.py, archived byte-exact after the resume round
V1_REGISTERED_VALIDATE = {
    "path": "results/dataprep/reports/S3_evidence/acceptance/registered_code/validate.py",
    "sha256": "461cd6a1cfa732079a26bd266178c208fd3421acacdc333c10e47254d48528dc",
    "note": "gnss_clean/{preorigin,full}/v1 MANIFEST.json list the post-registration "
            "dataprep_staging/s3_gnss/validate.py (5a06c7ac...); the v1 pre-origin "
            "validation.json (the v1 gate) was produced by this registered file. v1 manifests "
            "are never rewritten; this record is the reference."}
#: v1 code that produced the L0 download (unchanged, outside v2/)
CODE_V1 = ["fetch_ngl.py"]

ALLOWED = {
    "preorigin": ["calibration_target", "datum_fit", "baseline", "model_design", "selection",
                  "prior", "prior_width_input", "design_time_evidence", "scoring_target",
                  "validator", "after_the_fact_analysis", "statistical_layer_history"],
    "full": ["validator", "scoring_target", "after_the_fact_analysis",
             "statistical_layer_history"],
}
FORBIDDEN = {
    "preorigin": ["initial_conditions", "idw_field", "forcing_frozen", "forcing_recorded_arm",
                  "prospective_scorer"],
    "full": ["calibration_target", "datum_fit", "baseline", "model_design", "selection",
             "prior", "prior_width_input", "design_time_evidence", "initial_conditions",
             "idw_field", "forcing_frozen", "forcing_recorded_arm", "prospective_scorer"],
}


def inputs(variant: str) -> list[str]:
    ls = io.LS_CACHE
    out = [rel_to_repo(p) for p in sorted(ls.glob("ls-wra-gnss-obs__*.parquet"))]
    if variant == "full":
        out += [rel_to_repo(p) for p in sorted(ls.glob("ls-ncku-gnss1day-obs__*.parquet"))]
    stations = P.NGL_ALL if variant == "full" else P.ngl_policy(P.ORIGIN_END)[0]
    out += [rel_to_repo(io.ngl_tenv3_path(k)) for k in stations]
    out += [rel_to_repo(io.NGL_S3 / "steps.txt"), rel_to_repo(io.NGL_S3 / "SOURCE.json"),
            rel_to_repo(io.NGL_S3B / "SOURCE.json"), rel_to_repo(io.NGL_REVIEW / "SOURCE.json"),
            "results/twin_forward/datum_gate_innov.npz",
            "chou-shui-data/data/mlcw_stations.csv",
            "chou-shui-data/data/ls_cache/ls-wra-dbm-obs__meta.parquet"]
    out += [rel_to_repo(p) for p in sorted((ls / "clean").glob("ls-wra-mlcw-obs__*.parquet"))]
    for f in ("TKJS_281_29", "NTUH_282_29", "SJES", "STES"):
        out.append(rel_to_repo(ls / f"ls-wra-dbm-obs__{f}.parquet"))
    return out


def ngl_offfan_note(val: dict) -> str:
    """Card line on what removing ``ngl_offfan`` does to the 2021 extra (this variant's A2)."""
    a2 = val.get("A2_2021_extra", {})
    n = a2.get("cm_removed_ngl_offfan", {})
    u = a2.get("uncorrected", {})
    own = a2.get("cm_own_extra_ngl_offfan", {})
    if not n or not u or "pooled_gnss_extra_mm" not in n:
        return "- `ngl_offfan` removal effect on the 2021 extra: not evaluable in this variant."
    return (f"- Removing `ngl_offfan` changes the pooled 2021 extra at the co-located ring/pile "
            f"sites from {u['pooled_gnss_extra_mm']:.1f} mm (uncorrected) to "
            f"{n['pooled_gnss_extra_mm']:.1f} mm ({n.get('ratio_to_uncorrected_pooled', float('nan')):.2f} "
            f"of uncorrected; pooled ratio to the rings/piles {n['pooled_amplitude_ratio']:.3f}). "
            f"The off-fan median itself has a 2021 extra of {own.get('extra_mm', float('nan')):+.1f} mm "
            f"(height), against an interannual sd of {own.get('other_years_sd_mm', float('nan')):.1f} mm "
            "in its other years: interannual statistics of a single site after this removal "
            "carry reference noise of that size. Check the effect before removing it from series "
            "compared with rings.")


def card(variant: str, tab: dict, gate: dict, val: dict) -> str:
    st = tab["stations"]
    lines = [f"# gnss_clean/{variant} product card", "",
             "Cleaned daily GNSS heights of the WRA network (and, in the full variant, the "
             "four NCKU stations) and of the Nevada Geodetic Laboratory (NGL) stations in and "
             "around the Choushui fan (plan WP S3).", "",
             "## Coverage", ""]
    for fam in ("wra", "ncku", "ngl"):
        s = st[(st.family == fam) & (st.role != "dropped")]
        if len(s):
            lines.append(f"- {fam}: {len(s)} stations, {s.first_day.min()}..{s.last_day.max()}"
                         f" ({int(s.n_days.sum())} station-days)")
    dropped = st[st.role == "dropped"]
    if len(dropped):
        lines.append(f"- dropped: {', '.join(dropped.station)} (plan WP S3)")
    if variant == "preorigin":
        lines.append("- station sets use only criteria valid at the origin: every WRA cache "
                     "station with >= 60 pre-origin days (LNJS, YSLL and ANES included: the "
                     "plan's drop/flag rests on 2023-25 evidence); NGL fan-box stations and "
                     "JSAN (twin grid); off-fan reference from the plan's height-based "
                     "candidates inside the box")
    else:
        lines.append("- full variant: the plan's WRA drops (LNJS, YSLL) and ANES flag apply "
                     "(full-period evidence); NGL adds LSB0 and the regional continuity set "
                     "(CKSV, JUNA, TCMS, S01R); jump-cluster days (>= 3 consecutive months with "
                     ">= 3 consecutive-day jumps > 3 cm) carry flag 9")
    lines += ["", "## Values and flags", "",
              "- `value` = height (m) with the step catalogue removed; `value_raw` = L0 "
              "height; no common mode is removed (see `common_mode.parquet`).",
              "- flag 0 observed; 1 corrected by a step backed by metadata, the plan "
              "catalogue, or evident in the station's own series; 3 corrected by a step "
              "significant only after the neighbour common mode is removed; 9 suspect "
              "(gross or screen outlier, excursion, after an untied offset).",
              "- U1: scoring targets, baselines and datum fits read flags 0-1 only, so data "
              "after a flag-3 step are excluded from them.",
              "- `sd`: station high-frequency noise and step-correction sd in quadrature "
              "(daily); monthly sd adds serial correlation and the median's efficiency.",
              "", "## Common modes", "",
              "- `ngl_offfan`: daily median residual of the off-fan NGL reference stations "
              f"({', '.join(val.get('reference', []))}); the plan's off-fan common mode.",
              f"- `wra_hf`: WRA network median residual minus its {val.get('params', {}).get('hp_window')}"
              "-day running mean (the WRA processing noise; the slow part is real fan-wide "
              "motion and stays in).",
              ngl_offfan_note(val),
              "- `wra_onfan_fullband`: the pilot's on-fan median (diagnostic only; it "
              "contains real fan-wide deformation).", "",
              "## Gate", "",
              f"- passed: {gate['passed']} ({gate['criteria']})", "",
              "## Known limits", "",
              "- Step causes (antenna, monument) are unverified; WRA publishes no equipment "
              "log. NGL equipment changes come from NGL's steps.txt.",
              "- GNSS carries full-column and tectonic motion: compare with rings through "
              "the excess, not as a bridge.",
              "- Most NGL fan stations end 2024-04-16 (probe R-j unresolved).",
              ]
    return "\n".join(lines) + "\n"


def catalogue_note(steps: pd.DataFrame, end) -> str:
    """How the plan catalogue entered this variant: for each catalogue date inside the window,
    the evidence of the product step within 15 days (computed, not asserted)."""
    parts = []
    for k, items in P.WRA_CATALOGUE.items():
        for d, _ in items:
            d = pd.Timestamp(d)
            if end is not None and d > end:
                continue
            s = steps[(steps.family == "wra") & (steps.station == k)] if len(steps) else steps
            near = s[(pd.to_datetime(s.date) - d).abs().dt.days <= 15] if len(s) else s
            ev = ", ".join(f"{r.date} {r.evidence} {100 * r.size_m:+.1f} cm flag {r.flag_after}"
                           for r in near.itertuples()) or "no step within 15 days"
            parts.append(f"{k} {d.date()}: {ev}")
    return ("plan catalogue (compiled by the audit from full-period series); a run uses only "
            "dates <= its window end. 'detected+catalogue' = found by the scan itself, the "
            "catalogue only labels it; 'catalogue' = forced. " + "; ".join(parts))


def write_tables(d: Path, tab: dict) -> None:
    (d / "wra").mkdir()
    (d / "ngl").mkdir()
    tab["wra_daily"].to_parquet(d / "wra" / "daily.parquet", index=False)
    tab["wra_monthly"].to_parquet(d / "wra" / "monthly.parquet", index=False)
    if "ncku_daily" in tab:
        tab["ncku_daily"].to_parquet(d / "wra" / "ncku_daily.parquet", index=False)
        tab["ncku_monthly"].to_parquet(d / "wra" / "ncku_monthly.parquet", index=False)
    tab["ngl_daily"].to_parquet(d / "ngl" / "daily.parquet", index=False)
    tab["ngl_monthly"].to_parquet(d / "ngl" / "monthly.parquet", index=False)
    cms = tab["cms"].copy()
    cms.index = pd.DatetimeIndex(cms.index).strftime("%Y-%m-%d")
    cms.index.name = "date"
    cms.reset_index().to_parquet(d / "common_mode.parquet", index=False)
    tab["cms_monthly"].to_parquet(d / "common_mode_monthly.parquet", index=False)
    tab["steps"].to_csv(d / "steps.csv", index=False)
    tab["stations"].to_csv(d / "stations.csv", index=False)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["preorigin", "full"], required=True)
    ap.add_argument("--tune", required=True)
    ap.add_argument("--evidence", required=True)
    a = ap.parse_args(argv)
    ev = Path(a.evidence)
    p, ref, _ = RV.load_tuned(a.tune, a.variant)
    end = P.ORIGIN_END if a.variant == "preorigin" else None
    product = f"gnss_clean/{a.variant}"
    val_pre = json.loads((ev / "validation_preorigin.json").read_text())
    val = json.loads((ev / f"validation_{a.variant}.json").read_text())
    expected = json.loads((ev / f"tables_{a.variant}.json").read_text())
    version = f"{product}/{VERSION}"   # the version the validation run hashed
    tab = BT.build(p, ref, end, version=version)
    got = BT.table_hashes(tab)
    if got != expected:
        raise SystemExit(f"tables differ from the validation run: {got} != {expected}")
    d = manifest.new_version_dir(product)
    if d.name != VERSION:
        d.rmdir()                          # the empty directory this call just created
        raise SystemExit(f"{product}/{d.name}: the validation run hashed tables for {VERSION}")
    write_tables(d, tab)
    a1 = val_pre["A1_injected_steps"]
    a2 = val_pre["A2_2021_extra"]
    gate = {"passed": bool(a1["wra"]["pass"] and a1["ngl"]["pass"] and a2["pass"]),
            "criteria": f"plan 3 WP S3 acceptance, pre-registration {PREREG_ID}: A1 injected "
                        "steps >= 2 cm recovered within +-5 mm and +-3 days at a rate >= 0.90 "
                        "(WRA and NGL separately; seed 404) and A2 the 2021 extra compaction "
                        "survives the removal of EACH offered common mode, ngl_offfan and "
                        "wra_hf (same sign >= 0.75, pooled ratio >= 0.5), both on the "
                        "pre-origin variant (U7)",
            "A1_wra_recovered": a1["wra"]["recovered"], "A1_ngl_recovered": a1["ngl"]["recovered"],
            "A2_pass": a2["pass"], "A3": "reported (full variant)"}
    for cm in ("cm_removed_ngl_offfan", "cm_removed_wra_hf"):
        gate[f"A2_{cm}"] = {k: a2.get(cm, {}).get(k) for k in
                            ("n_evaluable", "same_sign_fraction", "pooled_amplitude_ratio",
                             "pass")}
    report = {"variant": a.variant, "gate": gate, "gate_source": "validation_preorigin.json",
              "this_variant": val,
              "note": "full-period numbers (full variant) are after-the-fact and never gate (U7)"}
    (d / "validation.json").write_text(json.dumps(BT.clean_json(report), indent=1,
                                                  allow_nan=False, default=str))
    (d / "PRODUCT_CARD.md").write_text(card(a.variant, tab, gate, val))
    code = [rel_to_repo(HERE / c) for c in CODE if (HERE / c).is_file()]
    code += [rel_to_repo(HERE.parent / c) for c in CODE_V1]
    code.append("hydrophysics/twin/mechanism.py")
    drop, flag = P.wra_policy(end)
    ngl_st, ref_cand = P.ngl_policy(end)
    m = manifest.build_manifest(
        d, product=product, layer="L1", inputs=inputs(a.variant), code=code,
        parameters={"detector": p.as_dict(), "reference_stations": ref,
                    "tune_file": rel_to_repo(Path(a.tune)),
                    "tune_rule": "pre-origin injected-step recovery (seeds 101/202) subject to "
                                 "<= 0.05 false detections per off-fan station-year",
                    "window_end": None if end is None else "2022-12-31",
                    "acceptance_seed": RV.SEED, "harness_family_G_r": "10..19",
                    "prereg": PREREG_ID,
                    "supersedes": {"version": f"{product}/v1",
                                   "v1_registered_validate": V1_REGISTERED_VALIDATE},
                    "wra_dropped": drop, "wra_flagged": flag,
                    "wra_catalogue": P.WRA_CATALOGUE,
                    "wra_catalogue_note": catalogue_note(tab["steps"], end),
                    "ngl_stations": ngl_st, "ngl_reference_candidates": ref_cand,
                    "jump_cluster_flag": None if end is not None else {
                        "jump_m": PR.JUMP_M, "per_month": PR.JUMP_PER_MONTH,
                        "months": PR.JUMP_MONTHS,
                        "note": "full variant only; rule written after seeing 2023-25 jump "
                                "counts (plan: Flag ANES)"},
                    "repair_round": "v2 (2026-10-12/13): see results/dataprep/reports/S3.md, "
                                    "Repair round"},
        fit_window_end="2022-12" if a.variant == "preorigin" else "2026-09",
        validation_report="validation.json", gate=gate,
        allowed_uses=ALLOWED[a.variant], forbidden_uses=FORBIDDEN[a.variant],
        personal_data="none (station codes, public station coordinates and school names; no "
                      "contact fields are read)",
        extra={"evidence": rel_to_repo(ev)})
    manifest.write_manifest(d, m)
    print(f"wrote {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
