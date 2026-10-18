"""Build ``results/dataprep/deep_benchmarks_printed/v<N>/`` (WP S2 repair round, 2026-10-12).

Version 2 of the printed deep-benchmark product. It changes no printed number. It answers three
findings of the independent verifier of round 3:

* **One row per sensor-month.** v1 ``printed_monthly.parquet`` carries every printing: the
  007001 reports print each month twice (the report-month column, and the prior-year column of
  the report twelve months later), so 245 of the 712 verified sensor-months appear twice and a
  scoring consumer would double-count them. v2 keeps one **primary** printing per sensor and
  data month: the verified printing of the earliest vintage (the report whose caption month is
  earliest); if no printing is verified, the earliest printing with its flag (9 or 8). The
  other verified printing is kept in ``reprint_*`` columns. The cumulative table is reduced
  the same way per (sensor, printed period), and its ``month`` is the period's last month.
* **A guarded route to the integrated-report values.** The 2024-01..2026-01 values of the
  integrated reports (series 007003; ``results/twin/report_extract/deep_piles/deep_piles.csv``,
  two-read verified by ``hydrophysics.twin.report_extract``) had no manifest. v2 carries them
  as ``integrated_monthly.parquet``: printed numbers and transcription checks only (the
  feed-derived columns of the transcription file are left out).
* **Uncertainty labelled (plan 4.7).** ``sd`` is the rounding error of one printed decimal. Its
  nominal 95 % interval is checked against the two printings of the same month (coverage in
  ``validation.json``) and labelled "uncalibrated" in the method column, the README and the
  manifest.

Time convention: the time-slice column ``month`` is the data month (the month the value
describes), as for every observed product of the plan. A value re-printed in a later report is
the agency's value for that data month; its vintage is recorded (``vintage_month``).

Gate (fixed here before the first run; the 007001 part is v1's gate unchanged): at least 90 %
of the printed 007001 monthly values verified by two independent reads, no verified row whose
month headers disagree with its caption, and at least 90 % of the non-null integrated values
two-read verified.

Usage: python -I printed_v2.py [--scratch DIR]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import flags as F  # noqa: E402
from common import guard, manifest  # noqa: E402
from common.paths import DATAPREP, L0_NEW, REPO, rel_to_repo  # noqa: E402

from s2_deep_benchmarks import s2lib as L  # noqa: E402

PRODUCT = "deep_benchmarks_printed"
V1 = DATAPREP / PRODUCT / "v1"
INTEGRATED_CSV = L.PRINTED_CSV
INTEGRATED_REPORT = L.PRINTED_REPORT
INTEGRATED_SOURCE = L0_NEW / "wra_monthly_reports" / "SOURCE.json"
CY_SOURCE = L0_NEW / "wra_cy_monthly_reports_007001" / "SOURCE.json"
GATE_MIN_VERIFIED = 0.90
ORIGIN = L.ORIGIN_MONTH
HERE = Path(__file__).resolve().parent
COMMON = HERE.parent / "common"
CODE = [Path(__file__), Path(L.__file__), HERE / "transcribe_007001.py", HERE / "cy007001.py",
        HERE / "fetch_007001.py", COMMON / "__init__.py", COMMON / "guard.py",
        COMMON / "manifest.py", COMMON / "flags.py", COMMON / "hashing.py",
        COMMON / "paths.py", COMMON / "uses.py", COMMON / "gitinfo.py"]
FLAG_RANK = {0: 0, 9: 1, 8: 2}
UNCAL = ("sd = rounding error of one printed decimal only; uncalibrated (plan 4.7: its nominal "
         "95 % interval does not cover the differences between two printings of the same "
         "month, see validation.json coverage)")


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------------------- pure helpers
def roc_to_month(s: str) -> str:
    """'113/11' -> '2024-11' (ROC year + 1911)."""
    y, m = s.strip().split("/")
    return f"{int(y) + 1911:04d}-{int(m):02d}"


def period_bounds(label: str) -> tuple[str, str]:
    """'112/4~113/11' -> ('2023-04', '2024-11')."""
    a, b = label.split("~")
    return roc_to_month(a), roc_to_month(b)


def primary_printings(df: pd.DataFrame, keys: list[str], vintage_col: str = "vintage_month"
                      ) -> pd.DataFrame:
    """One row per ``keys``: the verified (flag 0) printing of the earliest vintage, else the
    earliest printing by flag rank (9 before 8). Adds n_printings, n_verified_printings and the
    other verified printing (reprint_value, reprint_report, reprint_vintage_month,
    reprint_diff_mm = reprint - primary)."""
    d = df.copy()
    d["_rank"] = d["flag"].map(FLAG_RANK).fillna(3).astype(int)
    d = d.sort_values(keys + ["_rank", vintage_col], kind="mergesort")
    out = []
    for _, g in d.groupby(keys, sort=True):
        p = g.iloc[0].to_dict()
        ver = g[g["flag"] == 0]
        p["n_printings"] = int(len(g))
        p["n_verified_printings"] = int(len(ver))
        rep = ver.iloc[1] if (p["flag"] == 0 and len(ver) > 1) else None
        p["reprint_value"] = float(rep["value"]) if rep is not None else np.nan
        p["reprint_report"] = str(rep["report"]) if rep is not None else ""
        p["reprint_vintage_month"] = str(rep[vintage_col]) if rep is not None else ""
        p["reprint_diff_mm"] = (float(rep["value"]) - float(p["value"])) if rep is not None \
            else np.nan
        out.append(p)
    return pd.DataFrame(out).drop(columns="_rank").reset_index(drop=True)


def coverage_rounding(primary: pd.DataFrame) -> dict:
    """Coverage of the nominal 95 % interval of the difference of two printings,
    +-1.96 sqrt(2) sd, on every sensor-month with two verified printings."""
    j = primary[primary["reprint_diff_mm"].notna()]
    if not len(j):
        return {"n_pairs": 0}
    half = 1.96 * np.sqrt(2.0) * j["sd"].astype(float)
    inside = j["reprint_diff_mm"].abs() <= half + 1e-9
    a = j["reprint_diff_mm"].abs()
    return {"n_pairs": int(len(j)), "nominal": 0.95,
            "coverage": float(inside.mean()),
            "n_identical": int((a <= 1e-9).sum()), "n_within_0p1mm": int((a <= 0.1 + 1e-9).sum()),
            "n_over_2mm": int((a > 2.0 + 1e-9).sum()), "max_abs_mm": float(a.max()),
            "calibrated": bool(inside.mean() >= 0.90),
            "note": "two printings of the same data month (report-month column and the "
                    "prior-year column a year later); the rounding sd alone cannot explain "
                    "the revisions"}


def rounding_sd_cm(v) -> float:
    """Rounding sd (cm) of a printed value with its printed decimals (as s2lib.read_printed)."""
    if pd.isna(v):
        return float("nan")
    s = f"{abs(float(v)):.6f}".rstrip("0")
    dec = len(s.split(".")[1]) if "." in s and s.split(".")[1] else 0
    return 10.0 ** (-max(dec, 1)) / 2.0 / math.sqrt(3.0)


# ------------------------------------------------------------------------------- tables
def monthly_v2(pm1: pd.DataFrame, version: str) -> pd.DataFrame:
    d = pm1.copy()
    d["vintage_month"] = d["report_caption_month"]
    p = primary_printings(d, ["sensor_id", "month"])
    p["method"] = [
        f"{m}; primary printing (earliest verified vintage {v}) of {n} printing(s); {UNCAL}"
        for m, v, n in zip(p["method"], p["vintage_month"], p["n_printings"], strict=True)]
    p["product_version"] = version
    cols = ["sensor_id", "station", "depth_m", "month", "value", "sd", "flag", "column",
            "report", "report_caption_month", "vintage_month", "printed_mm", "positive_means",
            "headers_consistent", "n_printings", "n_verified_printings", "reprint_value",
            "reprint_report", "reprint_vintage_month", "reprint_diff_mm", "method",
            "product_version", "source_ids"]
    return p[cols].sort_values(["sensor_id", "month"]).reset_index(drop=True)


def cumulative_v2(pc1: pd.DataFrame, version: str) -> pd.DataFrame:
    d = pc1.copy()
    # a period printed as '-' has no label: its end is the caption month minus 12 (cum_prev)
    b = [period_bounds(lab) if "~" in str(lab) else
         ("", str(pd.Period(m, freq="M") - (12 if c == "cum_prev" else 0)))
         for lab, m, c in zip(d["period_label"], d["month"], d["column"], strict=True)]
    d["period_start"] = [x[0] for x in b]
    d["period_end"] = [x[1] for x in b]
    d["report_caption_month"] = d["month"]
    d["vintage_month"] = d["month"]
    p = primary_printings(d, ["sensor_id", "period_start", "period_end"])
    p["month"] = p["period_end"]
    p["method"] = [f"transcribed cumulative change over the printed period {lab} "
                   f"({s}..{e}); month = last month of the period; primary printing "
                   f"(earliest verified vintage {v}) of {n} printing(s); {UNCAL}"
                   for lab, s, e, v, n in zip(p["period_label"], p["period_start"],
                                              p["period_end"], p["vintage_month"],
                                              p["n_printings"], strict=True)]
    p["product_version"] = version
    cols = ["sensor_id", "station", "depth_m", "month", "period_label", "period_start",
            "period_end", "value", "sd", "flag", "column", "report", "report_caption_month",
            "vintage_month", "printed_mm", "positive_means", "n_printings",
            "n_verified_printings", "reprint_value", "reprint_report", "reprint_vintage_month",
            "reprint_diff_mm", "method", "product_version", "source_ids"]
    return p[cols].sort_values(["sensor_id", "month", "period_start"]).reset_index(drop=True)


def integrated_v2(raw: pd.DataFrame, version: str) -> pd.DataFrame:
    p = raw.copy()
    p["depth_m"] = p["depth_m"].astype("Int64")
    p["sensor_id"] = [L.PRINTED_TO_SENSOR.get((a, int(b)) if pd.notna(b) else (a, -1), "")
                      for a, b in zip(p["pile"], p["depth_m"], strict=True)]
    ver = p["verified"].astype(bool)
    p["value"] = p["month_cm"] * 10.0
    p["year_value"] = p["year_cm"] * 10.0
    p["sd"] = [rounding_sd_cm(v) * 10.0 for v in p["month_cm"]]
    p["year_sd"] = [rounding_sd_cm(v) * 10.0 for v in p["year_cm"]]
    p["flag"] = np.where(p["month_cm"].isna(), 8, np.where(ver, 0, 9)).astype("int64")
    p["sd"] = np.where(p["flag"] == 8, np.nan, p["sd"])
    note = []
    for s in p["sensor_id"]:
        if s == "TKJS(1)":
            note.append("printed depth label 300 m; the integrated reports print 土庫國中 at "
                        "400 m in other months, and no feed covers these months: label "
                        "unverifiable")
        elif s == "DLIO":
            note.append("Pingtung pile, outside the model domain")
        elif s == "WNES":
            note.append("灣內國小, off the alluvial fan")
        else:
            note.append("")
    p["label_note"] = note
    p["method"] = [f"integrated WRA monthly report (series 007003) {f} p.{pg}: printed monthly "
                   f"value, compression mm (printed cm x 10); two-read verified "
                   f"(hydrophysics.twin.report_extract); {UNCAL}"
                   for f, pg in zip(p["file"], p["page"], strict=True)]
    p.loc[p["flag"] == 8, "method"] = "printed as '-' (no value)"
    p["product_version"] = version
    p["source_ids"] = "007003:" + p["file"].astype(str)
    cols = ["sensor_id", "pile", "depth_m", "county", "township", "month", "value", "sd",
            "flag", "year_value", "year_sd", "roc_year", "roc_month", "agree", "agree_with",
            "chain_resid", "chain_ok", "label_note", "file", "page", "method",
            "product_version", "source_ids"]
    return p[cols].sort_values(["sensor_id", "month"]).reset_index(drop=True)


# ------------------------------------------------------------------------------- build
def compute(version: str) -> dict:
    prod = guard.open_product(V1, "validator")          # verifies every v1 output hash
    pm1 = pd.read_parquet(prod.path("printed_monthly.parquet"))
    pc1 = pd.read_parquet(prod.path("printed_cumulative.parquet"))
    v1val = json.loads(prod.path("validation.json").read_text(encoding="utf-8"))
    raw_int = pd.read_csv(INTEGRATED_CSV)
    pm = monthly_v2(pm1, version)
    pc = cumulative_v2(pc1, version)
    ig = integrated_v2(raw_int, version)
    # checks
    ver = pm1["flag"] == 0
    n_ver_rows = int(ver.sum())
    pre = pm[pm.month <= ORIGIN]
    pre_ver = pre[pre.flag == 0]
    late_only = pre_ver[pre_ver.vintage_month > ORIGIN]
    val = {
        "v1_product": rel_to_repo(V1), "v1_gate": prod.manifest["gate"],
        "v1_monthly_rows": int(len(pm1)), "v1_verified_printings": n_ver_rows,
        "monthly_rows": int(len(pm)),
        "monthly_flag_counts": {str(k): int(v) for k, v in pm.flag.value_counts().items()},
        "verified_sensor_months": int((pm.flag == 0).sum()),
        "sensor_months_with_two_verified_printings": int((pm.n_verified_printings > 1).sum()),
        "primary_is_report_month_column": int(((pm.flag == 0)
                                               & (pm.column == "report_month")).sum()),
        "primary_is_prior_year_column": int(((pm.flag == 0) & (pm.column == "prior_year")).sum()),
        "first_verified_month": str(pm[pm.flag == 0].month.min()),
        "last_verified_month": str(pm[pm.flag == 0].month.max()),
        "first_month_any_row": str(pm.month.min()),
        "preorigin": {
            "verified_sensor_months": int(len(pre_ver)),
            "known_only_from_a_later_vintage": int(len(late_only)),
            "later_vintage_months": sorted(late_only.month.unique().tolist()),
            "pairs_with_two_verified_printings": int(pre_ver.reprint_diff_mm.notna().sum()),
            "pairs_differing": int((pre_ver.reprint_diff_mm.abs() > 1e-9).sum()),
            "pairs_differing_over_2mm": int((pre_ver.reprint_diff_mm.abs() > 2.0).sum()),
            "convention": "time slicing on the data month; a re-printed value is the agency's "
                          "value for that data month (vintage recorded in vintage_month)"},
        "coverage_rounding_sd_monthly": coverage_rounding(pm),
        "coverage_rounding_sd_cumulative": coverage_rounding(pc),
        "cumulative_rows": int(len(pc)), "v1_cumulative_rows": int(len(pc1)),
        "cumulative_flag_counts": {str(k): int(v) for k, v in pc.flag.value_counts().items()},
        "integrated_rows": int(len(ig)),
        "integrated_flag_counts": {str(k): int(v) for k, v in ig.flag.value_counts().items()},
        "integrated_months": [str(ig.month.min()), str(ig.month.max())],
        "integrated_sensors": {s: int(n) for s, n in ig[ig.flag == 0].groupby("sensor_id").size()
                               .items()},
        "integrated_unmapped_rows": int((ig.sensor_id == "").sum()),
    }
    # gate
    v1_share = float(v1val.get("share_monthly_verified", float("nan")))
    nn = ig[ig.flag != 8]
    int_share = float((nn.flag == 0).mean()) if len(nn) else float("nan")
    hdr_bad = int(((pm1.flag == 0) & ~pm1.headers_consistent.astype(bool)).sum())
    share_007001 = float(n_ver_rows / max(int(pm1["printed_mm"].notna().sum()), 1))
    val["gate"] = {
        "share_007001_printed_values_verified": share_007001,
        "share_007001_v1_reported": v1_share,
        "verified_rows_header_inconsistent": hdr_bad,
        "share_integrated_values_verified": int_share,
        "threshold": GATE_MIN_VERIFIED,
        "passed": bool(share_007001 >= GATE_MIN_VERIFIED and hdr_bad == 0
                       and int_share >= GATE_MIN_VERIFIED)}
    # long-table checks
    val["table_checks"] = {k: F.validate_long_table(t) for k, t in
                           (("printed_monthly", pm), ("printed_cumulative", pc),
                            ("integrated_monthly", ig))}
    val["duplicate_keys"] = {
        "printed_monthly": int(pm.duplicated(["sensor_id", "month"]).sum()),
        "printed_cumulative": int(pc.duplicated(["sensor_id", "period_start", "period_end"])
                                  .sum()),
        "integrated_monthly": int(ig[ig.sensor_id != ""].duplicated(["sensor_id", "month"])
                                  .sum())}
    return {"pm": pm, "pc": pc, "ig": ig, "val": val, "prod": prod}


def card(val: dict, name: str) -> str:
    g = val["gate"]
    return "\n".join([
        f"# deep_benchmarks_printed/{name}: printed deep-benchmark values (one row per "
        "sensor-month)", "",
        "WP S2 repair round (2026-10-12), dataprep_staging/s2_deep_benchmarks/printed_v2.py. "
        "Built from deep_benchmarks_printed/v1 (Table 1 of the WRA 007001 Changhua-Yunlin "
        "reports) and from the integrated-report transcription (series 007003). No printed "
        "number is changed.", "",
        "## Tables", "",
        f"- printed_monthly.parquet: 007001 monthly values, one row per sensor and data month "
        f"({val['monthly_rows']} rows; verified months {val['first_verified_month']}.."
        f"{val['last_verified_month']}; the 2019-04 rows are flag 8). The primary printing is "
        "the verified one of the earliest vintage; reprint_* columns keep the other printing. "
        "value = compression mm.",
        "- printed_cumulative.parquet: 007001 cumulative changes, one row per sensor and printed "
        "period; month = the period's last month.",
        f"- integrated_monthly.parquet: integrated-report (007003) monthly values "
        f"{val['integrated_months'][0]}..{val['integrated_months'][1]} (compression mm), "
        "printed numbers and transcription checks only.",
        "- table1_rows.parquet, table1_reads.parquet: every 007001 row and read (copied from "
        "v1; no month column, so a time slice cannot read them).", "",
        "## Flags", "", "0 verified by two independent reads; 8 printed as '-'; 9 reads "
        "disagree or month headers inconsistent with the caption.", "",
        "## Uncertainty", "", "sd is the rounding error of one printed decimal and is "
        "UNCALIBRATED (plan 4.7): its nominal 95 % interval covers "
        f"{val['coverage_rounding_sd_monthly'].get('coverage', float('nan')):.2f} of the "
        "differences between two printings of the same month "
        f"({val['coverage_rounding_sd_monthly'].get('n_pairs')} pairs, "
        f"{val['coverage_rounding_sd_monthly'].get('n_over_2mm')} differ by more than 2 mm).", "",
        "## Time slicing", "", "month is the data month. A value re-printed a year later is "
        "the agency's value for that data month (vintage_month records the report). "
        f"{val['preorigin']['known_only_from_a_later_vintage']} pre-origin sensor-months "
        f"({', '.join(val['preorigin']['later_vintage_months'])}) are known only from a "
        "2023 re-printing.", "",
        f"## Gate: {g['passed']}", "",
        f"007001 values verified {g['share_007001_printed_values_verified']:.3f} (>= 0.9); "
        f"verified rows with inconsistent headers {g['verified_rows_header_inconsistent']}; "
        f"integrated values verified {g['share_integrated_values_verified']:.3f} (>= 0.9).", ""])


def manifest_for(d: Path, res: dict) -> dict:
    prod, val = res["prod"], res["val"]
    inputs = [V1 / "MANIFEST.json", V1 / "printed_monthly.parquet",
              V1 / "printed_cumulative.parquet", V1 / "table1_rows.parquet",
              V1 / "table1_reads.parquet", V1 / "validation.json", INTEGRATED_CSV,
              INTEGRATED_REPORT, INTEGRATED_SOURCE, CY_SOURCE]
    tok = prod.manifest["parameters"].get("claude_api_tokens")
    return manifest.build_manifest(
        d, product=PRODUCT, layer="L1", inputs=[rel_to_repo(p) for p in inputs],
        code=[rel_to_repo(p) for p in CODE],
        parameters={"fitted_parameters": "none",
                    "primary_printing": "verified printing of the earliest vintage",
                    "gate_min_verified": GATE_MIN_VERIFIED,
                    "claude_api_tokens_this_version": 0,
                    "claude_api_tokens_v1_transcription": tok,
                    "claude_api_tokens_integrated_transcription": json.loads(
                        INTEGRATED_REPORT.read_text(encoding="utf-8")).get("tokens")},
        fit_window_end=str(max(res["pm"].month.max(), res["ig"].month.max())),
        validation_report="validation.json",
        gate={"passed": bool(val["gate"]["passed"]),
              "criteria": "share of printed 007001 monthly values verified by two reads >= 0.9, "
                          "no verified row with month headers inconsistent with its caption "
                          "(v1 gate unchanged), and share of non-null integrated values "
                          "two-read verified >= 0.9"},
        time_slicing={"column": "month", "params_fit_window_end": "2019-04", "encoding": "iso"},
        allowed_uses=["validator", "scoring_target", "after_the_fact_analysis", "datum_fit",
                      "calibration_target", "baseline", "design_time_evidence"],
        forbidden_uses=["selection", "model_design", "prior", "prior_width_input",
                        "initial_conditions", "idw_field", "forcing_frozen",
                        "forcing_recorded_arm", "statistical_layer_history"],
        personal_data="none (printed station names and depths only; the reports' maintenance "
                      "staff names are never copied)",
        extra={"supersedes": "deep_benchmarks_printed/v1 (same printed numbers; v1 lists every "
                             "printing, v2 one primary printing per sensor-month)",
               "notes": [
                   "Printed values of agency report series; nothing is fitted, so the slice "
                   "parameter end is the first month of the series. value = compression in mm.",
                   "One row per sensor and data month in printed_monthly.parquet (primary = "
                   "verified printing of the earliest vintage); read it, not table1_rows, for "
                   "scoring, so that no month is counted twice.",
                   "Time slicing uses the data month; a re-printed value is the agency's value "
                   "for that data month (vintage_month). "
                   f"{val['preorigin']['known_only_from_a_later_vintage']} pre-origin "
                   "sensor-months are known only from a 2023 re-printing; "
                   f"{val['preorigin']['pairs_differing']} of "
                   f"{val['preorigin']['pairs_with_two_verified_printings']} pre-origin "
                   "double printings differ, "
                   f"{val['preorigin']['pairs_differing_over_2mm']} by more than 2 mm.",
                   "sd is rounding only and UNCALIBRATED (plan 4.7); coverage of its nominal 95 % "
                   "interval on re-printings in validation.json.",
                   "integrated_monthly.parquet: series 007003, 2024-01..2026-01, transcribed by "
                   "hydrophysics.twin.report_extract (two reads; the model never computed a "
                   "value). 土庫國中 printed at 300 m (2025-05, 06, 08) carries an unverifiable "
                   "depth label.",
                   "Role: validator of twin subsidence 2023-26 and of the deep-benchmark feeds; "
                   "pre-origin uses only through a time slice ending by 2022-12; never fitted, "
                   "never in a gate or choice of a model (U7).",
                   "Claude API: none in this version (v1: 263,527 input / 56,309 output tokens; "
                   "integrated transcription: see parameters)."]})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", type=Path, default=None)
    a = ap.parse_args(argv)
    t0 = dt.datetime.now()
    if a.scratch is None:
        base = DATAPREP / PRODUCT
        ns = [int(q.name[1:]) for q in base.glob("v*") if q.name[1:].isdigit()]
        version = f"{PRODUCT}/v{max(ns, default=0) + 1}"
    else:
        version = f"{PRODUCT}/scratch"
    res = compute(version)
    val = res["val"]
    bad = {k: v for k, v in val["table_checks"].items() if v}
    if bad or any(val["duplicate_keys"].values()):
        raise SystemExit(f"checks failed: {bad} {val['duplicate_keys']}")
    val["runtime_s"] = (dt.datetime.now() - t0).total_seconds()
    out = a.scratch if a.scratch is not None else manifest.new_version_dir(PRODUCT)
    if a.scratch is None and f"{PRODUCT}/{out.name}" != version:
        raise SystemExit(f"version race: computed for {version}, got {out}")
    out.mkdir(parents=True, exist_ok=True)
    res["pm"].to_parquet(out / "printed_monthly.parquet", index=False)
    res["pc"].to_parquet(out / "printed_cumulative.parquet", index=False)
    res["ig"].to_parquet(out / "integrated_monthly.parquet", index=False)
    for f in ("table1_rows.parquet", "table1_reads.parquet"):
        shutil.copyfile(res["prod"].path(f), out / f)
    (out / "validation.json").write_text(json.dumps(val, ensure_ascii=False, indent=1,
                                                    default=str) + "\n", encoding="utf-8")
    (out / "README.md").write_text(card(val, out.name), encoding="utf-8")
    log(json.dumps({k: val[k] for k in ("monthly_rows", "verified_sensor_months", "gate",
                                        "coverage_rounding_sd_monthly", "preorigin")},
                   ensure_ascii=False, default=str, indent=1))
    if a.scratch is not None:
        return 0
    m = manifest_for(out, res)
    errs = guard.time_slicing_errors(m, out)
    if errs:
        raise SystemExit(f"time_slicing errors (manifest not written): {errs}")
    manifest.write_manifest(out, m)
    log(f"wrote {out} ({rel_to_repo(REPO / out)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
