"""Transcribe table 1 (彰雲地區深層水準樁 觀測成果) of the 007001 reports into an L1 product.

Reads ``data_fetch_api/new_measurements/wra_cy_monthly_reports_007001/`` (SOURCE.json lists the
PDFs) and writes ``results/dataprep/deep_benchmarks_printed/v<N>/``.

Per report, three independent reads are possible (``cy007001``):

* ``text_parse``: a deterministic parse of the table page's poppler layout text;
* ``pdf_model``: ``claude-opus-5-5`` on the table page as a PDF document, structured output;
* ``text_model``: ``claude-sonnet-5-5`` on the layout text, only when the first two disagree
  on any row or the deterministic parse fails.

A row (station, depth) is verified when two reads agree on its depth, its four values, its
two month headers and the caption month (flag 0); otherwise every value of the row is flag 9.
The model only copies printed numbers (schema: null for a printed ``-``); nothing is computed.
Raw API responses are cached in ``results/dataprep/reports/S2_evidence/cy007001_api_cache/``
and token usage is recorded.

Gate of the transcription product (fixed before the first run): at least 90 % of the
printed monthly values verified, and every verified row's month headers consistent with its
caption (current column = caption month, previous column 12 months earlier).

Usage: python -I transcribe_007001.py [--no-api] [--scratch DIR]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import flags as F  # noqa: E402
from common import guard, manifest  # noqa: E402
from common.paths import L0_NEW, REPO, REPORTS_DIR, rel_to_repo  # noqa: E402

from s2_deep_benchmarks import cy007001 as C  # noqa: E402

PRODUCT = "deep_benchmarks_printed"
L0_DIR = L0_NEW / "wra_cy_monthly_reports_007001"
CACHE = REPORTS_DIR / "S2_evidence" / "cy007001_api_cache"
MODEL_PDF = "claude-opus-5-5"
MODEL_TEXT = "claude-sonnet-5-5"
GATE_MIN_VERIFIED = 0.90
ROUND_SD_MM = 0.1 / 2.0 / math.sqrt(3.0)     # one printed decimal (mm), uniform rounding


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------------------- API
def load_key() -> None:
    """ANTHROPIC_API_KEY from the repository .env (never printed), as report_extract expects
    it in the environment."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    env = REPO / ".env"
    for ln in env.read_text(encoding="utf-8").splitlines():
        if ln.startswith("ANTHROPIC_API_KEY="):
            os.environ["ANTHROPIC_API_KEY"] = ln.split("=", 1)[1].strip().strip('"').strip("'")
            return
    raise SystemExit("ANTHROPIC_API_KEY not found in .env")


def page_pdf_bytes(pdf: Path, page: int) -> bytes:
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "page.pdf"
        subprocess.run(["pdfseparate", "-f", str(page), "-l", str(page), str(pdf), str(out)],
                       check=True)
        return out.read_bytes()


def call(client, model: str, content: list[dict]) -> tuple[dict, dict]:
    resp = client.messages.create(
        model=model, max_tokens=4000, system=C.SYSTEM,
        messages=[{"role": "user", "content": content}],
        output_config={"format": {"type": "json_schema", "schema": C.SCHEMA}})
    text = "".join(b.text for b in resp.content if b.type == "text")
    return json.loads(text), resp.usage.model_dump(mode="json")


def pdf_content(data: bytes) -> list[dict]:
    import base64
    return [{"type": "document",
             "source": {"type": "base64", "media_type": "application/pdf",
                        "data": base64.b64encode(data).decode()}},
            {"type": "text", "text": "Transcribe the table 彰雲地區深層水準樁 觀測成果 on this "
                                     "page."}]


def text_content(text: str) -> list[dict]:
    return [{"type": "text",
             "text": "Below is the poppler layout text of one report page. Transcribe the "
                     "table 彰雲地區深層水準樁 觀測成果. The county column is printed vertically "
                     "(彰/化, 雲/林) and its characters may appear on separate lines.\n\n" + text}]


# ------------------------------------------------------------------------------- per report
def read_report(pdf: Path, client, use_api: bool) -> dict:
    stem = pdf.stem
    rec = {"file": pdf.name, "page": C.find_table_page(pdf)}
    if rec["page"] is None:
        rec["error"] = "table caption not found"
        return rec
    text = C.page_text(pdf, rec["page"])
    try:
        rec["text_parse"] = C.parse_layout(text)
    except (ValueError, StopIteration) as exc:
        rec["text_parse_error"] = str(exc)
    cache = CACHE / f"{stem}.json"
    cached = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    if "pdf_model" not in cached and use_api:
        t, u = call(client, MODEL_PDF, pdf_content(page_pdf_bytes(pdf, rec["page"])))
        cached["pdf_model"] = {"model": MODEL_PDF, "table": t, "usage": u,
                               "called": dt.datetime.now().isoformat(timespec="seconds")}
        CACHE.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(cached, ensure_ascii=False, indent=1), encoding="utf-8")
    if "pdf_model" in cached:
        rec["pdf_model"] = cached["pdf_model"]["table"]
    need_third = ("text_parse" not in rec) or ("pdf_model" not in rec) or bool(
        (~C.compare(C.rows_frame(rec["text_parse"], stem, "a"),
                    C.rows_frame(rec["pdf_model"], stem, "b"))["agree"]).any())
    if need_third and "text_model" not in cached and use_api:
        t, u = call(client, MODEL_TEXT, text_content(text))
        cached["text_model"] = {"model": MODEL_TEXT, "table": t, "usage": u,
                                "called": dt.datetime.now().isoformat(timespec="seconds")}
        cache.write_text(json.dumps(cached, ensure_ascii=False, indent=1), encoding="utf-8")
    if "text_model" in cached:
        rec["text_model"] = cached["text_model"]["table"]
    rec["usage"] = {k: cached[k]["usage"] for k in ("pdf_model", "text_model") if k in cached}
    return rec


def resolve(rec: dict, key: str) -> pd.DataFrame:
    """One row per (station, depth): the values of the first agreeing pair of reads."""
    reads = {r: C.rows_frame(rec[r], key, r) for r in ("text_parse", "pdf_model", "text_model")
             if r in rec}
    pairs = [("text_parse", "pdf_model"), ("text_parse", "text_model"),
             ("pdf_model", "text_model")]
    out = {}
    for a, b in pairs:
        if a not in reads or b not in reads:
            continue
        cmp = C.compare(reads[a], reads[b])
        for r in cmp[cmp.agree].itertuples():
            k = (r.station, r.depth_m)
            if k not in out:
                row = reads[a][(reads[a].station == r.station)
                               & (reads[a].depth_m == r.depth_m)].iloc[0].to_dict()
                row.update({"flag": 0, "agreeing_reads": f"{a}+{b}"})
                out[k] = row
    # rows no pair agrees on: keep the first read's values, flag 9
    for r in ("text_parse", "pdf_model", "text_model"):
        if r not in reads:
            continue
        for row in reads[r].to_dict("records"):
            k = (row["station"], row["depth_m"])
            if k not in out:
                row.update({"flag": 9, "agreeing_reads": ""})
                out[k] = row
    df = pd.DataFrame(list(out.values()))
    if len(df):
        df["n_reads"] = len(reads)
    return df


# ------------------------------------------------------------------------------- tables
def month_shift(m: str, k: int) -> str:
    return str(pd.Period(m, freq="M") + k)


def compression(v, positive_means) -> float:
    """Printed value -> compression (mm) using the table's own sign note."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    if positive_means == "compression":
        return float(v)
    if positive_means == "rebound":
        return -float(v)
    return np.nan


def long_monthly(rows: pd.DataFrame, version: str) -> pd.DataFrame:
    out = []
    for r in rows.to_dict("records"):
        cap = f"{int(r['caption_roc_year']) + 1911}-{int(r['caption_month']):02d}" \
            if pd.notna(r.get("caption_roc_year")) and pd.notna(r.get("caption_month")) \
            else None
        mc, mp = C.ym_label_to_month(r["col_month_curr"]), C.ym_label_to_month(r["col_month_prev"])
        hdr_ok = cap is not None and mc == cap and mp == month_shift(cap, -12)
        for col, m, v in (("report_month", mc or cap, r["month_curr_mm"]),
                          ("prior_year", mp or (month_shift(cap, -12) if cap else None),
                           r["month_prev_mm"])):
            if m is None:
                continue
            flag = int(r["flag"])
            sign_known = r.get("positive_means") in ("compression", "rebound")
            if flag == 0 and not (hdr_ok and sign_known):
                flag = 9
            if v is None or (isinstance(v, float) and np.isnan(v)):
                flag = 8
            meth = (f"transcribed from 007001 table 1 ({col} column); reads "
                    f"{r['agreeing_reads'] or 'disagree'}; positive printed = "
                    f"{r.get('positive_means')}")
            if not hdr_ok:
                meth += "; month headers inconsistent with the caption"
            comp = compression(v, r.get("positive_means"))
            if flag != 8 and np.isnan(comp):
                comp = -float(v)      # sign note missing: flag 9 row keeps the common sign
                meth += "; sign note not printed (value negated as in most reports)"
            out.append({"sensor_id": r["sensor_id"], "station": r["station"],
                        "depth_m": r["depth_m"], "month": m, "column": col,
                        "report": r["file"], "report_caption_month": cap,
                        "printed_mm": v, "positive_means": r.get("positive_means"),
                        "value": comp if flag != 8 else np.nan,
                        "sd": ROUND_SD_MM if flag != 8 else np.nan,
                        "flag": flag, "headers_consistent": bool(hdr_ok),
                        "method": meth, "product_version": version,
                        "source_ids": f"007001:{r['file']}"})
    df = pd.DataFrame(out)
    return df.sort_values(["sensor_id", "month", "column"]).reset_index(drop=True)


def long_cumulative(rows: pd.DataFrame, version: str) -> pd.DataFrame:
    out = []
    for r in rows.to_dict("records"):
        cap = f"{int(r['caption_roc_year']) + 1911}-{int(r['caption_month']):02d}"
        for col, lab, v in (("cum_prev", r["col_cum_prev"], r["cum_prev_mm"]),
                            ("cum_curr", r["col_cum_curr"], r["cum_curr_mm"])):
            flag = 8 if v is None or (isinstance(v, float) and np.isnan(v)) else int(r["flag"])
            comp = compression(v, r.get("positive_means"))
            if flag != 8 and np.isnan(comp):
                flag, comp = 9, -float(v)
            out.append({"sensor_id": r["sensor_id"], "station": r["station"],
                        "depth_m": r["depth_m"], "month": cap, "column": col,
                        "period_label": lab, "report": r["file"], "printed_mm": v,
                        "positive_means": r.get("positive_means"),
                        "value": comp if flag != 8 else np.nan,
                        "sd": ROUND_SD_MM if flag != 8 else np.nan, "flag": flag,
                        "method": "transcribed cumulative change (period in period_label; "
                                  "month = report caption month)",
                        "product_version": version, "source_ids": f"007001:{r['file']}"})
    return pd.DataFrame(out)


def cross_report(monthly: pd.DataFrame) -> dict:
    """Values printed twice (report month m and the prior-year column of report m+12)."""
    v = monthly[monthly.flag == 0]
    a = v[v.column == "report_month"][["sensor_id", "month", "value", "report"]]
    b = v[v.column == "prior_year"][["sensor_id", "month", "value", "report"]]
    j = a.merge(b, on=["sensor_id", "month"], suffixes=("_m", "_m12"))
    j["diff_mm"] = j.value_m12 - j.value_m      # compression, sign notes applied
    same = j.diff_mm.abs() < 1e-9
    big = j.diff_mm.abs() > 2.0 + 1e-9
    return {"n_pairs": int(len(j)), "n_identical": int(same.sum()),
            "n_within_0p1mm": int((j.diff_mm.abs() <= 0.1 + 1e-9).sum()),
            "n_over_2mm": int(big.sum()),
            "median_abs_diff_mm": float(j.diff_mm.abs().median()) if len(j) else float("nan"),
            "differences": j[~same].round(3).to_dict("records")}


# ------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-api", action="store_true")
    ap.add_argument("--scratch", type=Path, default=None)
    a = ap.parse_args(argv)
    src = json.loads((L0_DIR / "SOURCE.json").read_text(encoding="utf-8"))
    pdfs = [L0_DIR / f["file"] for f in src["files"] if f["file"].endswith(".pdf")]
    client = None
    if not a.no_api:
        import anthropic
        load_key()
        client = anthropic.Anthropic(timeout=600.0)
    base = REPO / "results" / "dataprep" / PRODUCT
    ns = [int(q.name[1:]) for q in base.glob("v*") if q.name[1:].isdigit()] if base.exists() \
        else []
    version = f"{PRODUCT}/v{max(ns, default=0) + 1}" if a.scratch is None \
        else f"{PRODUCT}/scratch"
    recs, rows, reads = [], [], []
    for pdf in pdfs:
        rec = read_report(pdf, client, not a.no_api)
        recs.append(rec)
        if "error" in rec:
            log(pdf.name, rec["error"])
            continue
        key = pdf.stem.replace("cy_", "")
        for r in ("text_parse", "pdf_model", "text_model"):
            if r in rec:
                fr = C.rows_frame(rec[r], key, r)
                fr["file"] = pdf.name
                reads.append(fr)
        res = resolve(rec, key)
        res["file"] = pdf.name
        rows.append(res)
        log(pdf.name, "page", rec["page"], "reads", [r for r in ("text_parse", "pdf_model",
                                                                  "text_model") if r in rec],
            "rows", len(res), "verified", int((res.flag == 0).sum()) if len(res) else 0,
            rec.get("text_parse_error", ""))
    rows = pd.concat(rows, ignore_index=True)
    reads = pd.concat(reads, ignore_index=True)
    monthly = long_monthly(rows, version)
    cum = long_cumulative(rows, version)
    usage = [{"file": r["file"], **r.get("usage", {})} for r in recs]
    tok = {k: int(sum(u.get(m, {}).get(k, 0) or 0 for u in usage
                      for m in ("pdf_model", "text_model")))
           for k in ("input_tokens", "output_tokens")}
    vals = monthly[monthly.flag != 8]
    share_ver = float((vals.flag == 0).mean()) if len(vals) else 0.0
    ver_rows = rows[rows.flag == 0]
    hdr_bad = int((~monthly[monthly.flag.isin([0, 9])].headers_consistent).sum())
    val = {"schema": "s2-cy007001-transcription/1",
           "built": dt.datetime.now().isoformat(timespec="seconds"),
           "n_reports": len(pdfs), "n_with_table": int(sum("error" not in r for r in recs)),
           "reports_without_table": [r["file"] for r in recs if "error" in r],
           "text_parse_errors": {r["file"]: r["text_parse_error"] for r in recs
                                 if "text_parse_error" in r},
           "n_rows": int(len(rows)), "n_rows_verified": int(len(ver_rows)),
           "n_rows_third_read": int(sum("text_model" in r for r in recs)),
           "agreeing_pairs": rows.agreeing_reads.value_counts().to_dict(),
           "n_monthly_values": int(len(vals)), "share_monthly_verified": share_ver,
           "n_header_inconsistent_values": hdr_bad,
           "cross_report": cross_report(monthly),
           "unmapped_stations": sorted({f"{s} {d}" for s, d, sid in
                                        zip(rows.station, rows.depth_m, rows.sensor_id,
                                            strict=True) if not sid}),
           "months": [str(monthly.month.min()), str(monthly.month.max())],
           "models": {"pdf": MODEL_PDF, "text": MODEL_TEXT}, "tokens": tok, "usage": usage,
           "gate_rule": f"share of printed monthly values verified >= {GATE_MIN_VERIFIED} and "
                        "no verified row with month headers inconsistent with its caption"}
    gate_ok = share_ver >= GATE_MIN_VERIFIED and int(
        (~monthly[monthly.flag == 0].headers_consistent).sum()) == 0
    val["gate_passed"] = bool(gate_ok)
    errs = {k: F.validate_long_table(df) for k, df in (("monthly", monthly), ("cum", cum))}
    val["table_checks"] = errs
    if any(errs.values()):
        raise SystemExit(f"table checks failed: {errs}")
    log(json.dumps({k: val[k] for k in ("n_reports", "n_with_table", "n_rows",
                                        "n_rows_verified", "share_monthly_verified",
                                        "n_header_inconsistent_values", "months", "tokens",
                                        "gate_passed")}, ensure_ascii=False))
    out = a.scratch
    if out is None:
        out = manifest.new_version_dir(PRODUCT)
        if f"{PRODUCT}/{out.name}" != version:
            raise SystemExit(f"version race: {version} vs {out}")
    out.mkdir(parents=True, exist_ok=True)
    monthly.to_parquet(out / "printed_monthly.parquet", index=False)
    cum.to_parquet(out / "printed_cumulative.parquet", index=False)
    rows.to_parquet(out / "table1_rows.parquet", index=False)
    reads.to_parquet(out / "table1_reads.parquet", index=False)
    (out / "validation.json").write_text(json.dumps(val, ensure_ascii=False, indent=1,
                                                    default=str) + "\n", encoding="utf-8")
    if a.scratch is not None:
        return 0
    (out / "README.md").write_text(card(val, out.name), encoding="utf-8")
    inputs = [L0_DIR / "SOURCE.json", *pdfs]
    inputs += sorted(CACHE.glob("cy_*.json"))
    m = manifest.build_manifest(
        out, product=PRODUCT, layer="L1", inputs=[rel_to_repo(p) for p in inputs],
        code=[rel_to_repo(Path(__file__)), rel_to_repo(Path(C.__file__)),
              rel_to_repo(Path(__file__).with_name("fetch_007001.py"))],
        parameters={"models": {"pdf": MODEL_PDF, "text": MODEL_TEXT},
                    "verification": "two independent reads agree per row",
                    "gate_min_verified": GATE_MIN_VERIFIED, "fitted_parameters": "none",
                    "claude_api_tokens": tok},
        fit_window_end=str(monthly.month.max()),
        validation_report="validation.json",
        gate={"passed": bool(gate_ok), "criteria": val["gate_rule"]},
        time_slicing={"column": "month", "params_fit_window_end": str(monthly.month.min()),
                      "encoding": "iso"},
        allowed_uses=["validator", "scoring_target", "after_the_fact_analysis", "datum_fit",
                      "calibration_target", "baseline", "design_time_evidence"],
        forbidden_uses=["selection", "model_design", "prior", "prior_width_input",
                        "initial_conditions", "idw_field", "forcing_frozen",
                        "forcing_recorded_arm", "statistical_layer_history"],
        personal_data="none (printed station names and depths only)",
        extra={"notes": [
            "Printed values of an agency report series; nothing is fitted, so the slice "
            "parameter end is the first month of the series. value = compression in mm "
            "(the negated printed number); printed_mm keeps the printed sign (positive = "
            "rebound).",
            "Role: validator of the deep-benchmark feeds (WP S2 test C) and of twin "
            "subsidence; never fitted, never in a gate or choice of a model (U7).",
            "Claude API used only to copy printed numbers (approved for S2 transcription)."]})
    errs = guard.time_slicing_errors(m, out)
    if errs:
        raise SystemExit(f"time_slicing errors (manifest not written): {errs}")
    manifest.write_manifest(out, m)
    log(f"wrote {out}")
    return 0


def card(val: dict, vname: str) -> str:
    return "\n".join([
        f"# {PRODUCT}/{vname}: printed deep-benchmark table of the 007001 reports",
        "",
        "Table 1 (彰雲地區深層水準樁 觀測成果) of the WRA 彰化與雲林地區地層下陷監測及分析 monthly "
        "data-analysis reports, transcribed by WP S2 "
        "(dataprep_staging/s2_deep_benchmarks/transcribe_007001.py).",
        "",
        f"- reports: {val['n_reports']} ({val['n_with_table']} with the table); months "
        f"{val['months'][0]}..{val['months'][1]}",
        f"- rows verified by two independent reads: {val['n_rows_verified']} of {val['n_rows']}",
        f"- printed monthly values verified: {val['share_monthly_verified']:.3f}",
        f"- gate ({val['gate_rule']}): {val['gate_passed']}",
        "",
        "Tables: printed_monthly.parquet (one row per sensor, value month and column: "
        "report_month or prior_year; value = compression mm), printed_cumulative.parquet, "
        "table1_rows.parquet (resolved rows), table1_reads.parquet (every read).",
        "Flags: 0 verified; 8 printed as -; 9 reads disagree or headers inconsistent.",
        ""])


if __name__ == "__main__":
    raise SystemExit(main())
