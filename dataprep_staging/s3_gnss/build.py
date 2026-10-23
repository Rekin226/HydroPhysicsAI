"""WP S3: build a ``gnss_clean`` product version (plan WP S3; layout plan 2.2, manifest 2.3).

    dataprep_staging/tools/cpu_slot.sh python -I dataprep_staging/s3_gnss/build.py \
        --variant {preorigin,full} --tune TUNE_JSON --evidence EVIDENCE_DIR

Writes ``results/dataprep/gnss_clean/<variant>/v<N>/``:

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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import build_tables as BT  # noqa: E402
import io_gnss as io  # noqa: E402
import pipeline as P  # noqa: E402
import run_validation as RV  # noqa: E402
from common import manifest  # noqa: E402
from common.paths import rel_to_repo  # noqa: E402

PREREG_ID = "S3.gnss_clean_acceptance.v1"
CODE = ["io_gnss.py", "model.py", "pipeline.py", "products.py", "build_tables.py",
        "injection.py", "validate.py", "run_validation.py", "build.py", "tune.py",
        "fetch_ngl.py", "record_looks.py", "prereg_s3.py"]

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
    out += [rel_to_repo(io.ngl_tenv3_path(k)) for k in P.NGL_ALL]
    out += [rel_to_repo(io.NGL_S3 / "steps.txt"), rel_to_repo(io.NGL_S3 / "SOURCE.json"),
            rel_to_repo(io.NGL_S3B / "SOURCE.json"), rel_to_repo(io.NGL_REVIEW / "SOURCE.json"),
            "results/twin_forward/datum_gate_innov.npz",
            "chou-shui-data/data/mlcw_stations.csv",
            "chou-shui-data/data/ls_cache/ls-wra-dbm-obs__meta.parquet"]
    out += [rel_to_repo(p) for p in sorted((ls / "clean").glob("ls-wra-mlcw-obs__*.parquet"))]
    for f in ("TKJS_281_29", "NTUH_282_29", "SJES", "STES"):
        out.append(rel_to_repo(ls / f"ls-wra-dbm-obs__{f}.parquet"))
    return out


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
              "- `wra_hf`: WRA network median residual minus its 31-day running mean (the "
              "WRA processing noise; the slow part is real fan-wide motion and stays in).",
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
    p, ref, _ = RV.load_tuned(a.tune)
    end = P.ORIGIN_END if a.variant == "preorigin" else None
    product = f"gnss_clean/{a.variant}"
    val_pre = json.loads((ev / "validation_preorigin.json").read_text())
    val = json.loads((ev / f"validation_{a.variant}.json").read_text())
    expected = json.loads((ev / f"tables_{a.variant}.json").read_text())
    version = f"{product}/v1"          # the version the validation run hashed
    tab = BT.build(p, ref, end, version=version)
    got = BT.table_hashes(tab)
    if got != expected:
        raise SystemExit(f"tables differ from the validation run: {got} != {expected}")
    d = manifest.new_version_dir(product)
    if d.name != "v1":
        d.rmdir()                          # the empty directory this call just created
        raise SystemExit(f"{product}/{d.name}: the validation run hashed tables for v1")
    write_tables(d, tab)
    a1 = val_pre["A1_injected_steps"]
    a2 = val_pre["A2_2021_extra"]
    gate = {"passed": bool(a1["wra"]["pass"] and a1["ngl"]["pass"] and a2["pass"]),
            "criteria": f"plan 3 WP S3 acceptance, pre-registration {PREREG_ID}: A1 injected "
                        "steps >= 2 cm recovered within +-5 mm and +-3 days at a rate >= 0.90 "
                        "(WRA and NGL separately) and A2 the 2021 extra compaction survives "
                        "the common-mode removal (same sign >= 0.75, pooled ratio >= 0.5), "
                        "both on the pre-origin variant (U7)",
            "A1_wra_recovered": a1["wra"]["recovered"], "A1_ngl_recovered": a1["ngl"]["recovered"],
            "A2_same_sign": a2["cm_removed_adopted_wra_hf"].get("same_sign_fraction"),
            "A2_pooled_ratio": a2["cm_removed_adopted_wra_hf"].get("pooled_amplitude_ratio"),
            "A2_n_evaluable": a2["cm_removed_adopted_wra_hf"].get("n_evaluable"),
            "A3": "reported (full variant)"}
    report = {"variant": a.variant, "gate": gate, "gate_source": "validation_preorigin.json",
              "this_variant": val,
              "note": "full-period numbers (full variant) are after-the-fact and never gate (U7)"}
    (d / "validation.json").write_text(json.dumps(BT.clean_json(report), indent=1,
                                                  allow_nan=False, default=str))
    (d / "PRODUCT_CARD.md").write_text(card(a.variant, tab, gate, val))
    code = [rel_to_repo(HERE / c) for c in CODE if (HERE / c).is_file()]
    code.append("hydrophysics/twin/mechanism.py")
    m = manifest.build_manifest(
        d, product=product, layer="L1", inputs=inputs(a.variant), code=code,
        parameters={"detector": p.as_dict(), "reference_stations": ref,
                    "tune_file": rel_to_repo(Path(a.tune)),
                    "tune_rule": "pre-origin injected-step recovery (seeds 101/202) subject to "
                                 "<= 0.05 false detections per off-fan station-year",
                    "window_end": None if end is None else "2022-12-31",
                    "acceptance_seed": RV.SEED, "harness_family_G_r": "0..9",
                    "prereg": PREREG_ID,
                    "wra_dropped": P.WRA_DROP, "wra_flagged": P.WRA_FLAG,
                    "wra_catalogue": P.WRA_CATALOGUE},
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
