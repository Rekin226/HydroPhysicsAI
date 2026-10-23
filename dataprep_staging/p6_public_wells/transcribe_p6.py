"""Transcribe P6's table and chart pages with the Claude API (decision D6; transcription only).

Pages (``JOBS``):

* ``p2_at2_2``: phase-2 main plan, annex table 2-2 (附表2-2), public wells disposed 100-108
  by agency and year, counts and abstraction cuts (萬噸/年). The text layer keeps the numbers
  of each row but loses the empty cells, so the column of a number is ambiguous there.
* ``p2_at2_5``: annex table 2-5 (附表2-5), irrigation-association and Taiwan Sugar wells by
  year (a cross-check of 附表2-2).
* ``b110_irr_yl``: WRA 110 briefing p. 18, chart 灌溉抽水量比較 (Yunlin county Jan-May
  irrigation pumping 100-110, 萬噸): the WRA side of the pre-origin WRA-to-model ratio check.

Two independent reads per page: ``pdf_read`` (claude-opus-5-5 on the page as a one-page PDF)
and ``img_read`` (claude-sonnet-5-5 on the page rendered to PNG at 200 dpi). Tables are copied
cell by cell as printed strings (no arithmetic); charts give marker values off the axis.
Raw responses and token usage are cached in ``results/dataprep/reports/P6_evidence/api_cache``
(one JSON per page; a cached read is never re-requested). The API never fills a value.

Usage: python -I transcribe_p6.py [--no-api]
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import REPO  # noqa: E402

from p6_public_wells import p6const as K  # noqa: E402

TABLE_SYSTEM = (
    "You transcribe tables from Taiwanese government planning documents (Ministry of Economic "
    "Affairs, Water Resources Agency). Copy every table on the page exactly as printed: the "
    "title, the full column headers (join multi-line header text top to bottom, including the "
    "group header such as 彰化地區 or 雲林地區, separated by '|'), and for every body row the "
    "row label and one string per column, left to right, in the printed column order. Copy each "
    "cell's text verbatim (digits, commas, decimal points, dashes, footnote marks and any small "
    "note text inside the cell). An empty cell is the empty string ''. Never compute, sum, "
    "infer, shift or fill a value; never merge or drop columns; every row has exactly as many "
    "cells as the table has columns after the row-label column. Also copy the notes printed "
    "under each table.")

TABLE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["tables"],
    "properties": {
        "tables": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["title", "row_label_header", "column_headers", "rows", "notes"],
                "properties": {
                    "title": {"type": "string"},
                    "row_label_header": {"type": "string"},
                    "column_headers": {"type": "array", "items": {"type": "string"}},
                    "rows": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["row_label", "cells"],
                            "properties": {
                                "row_label": {"type": "string"},
                                "cells": {"type": "array", "items": {"type": "string"}},
                            },
                        },
                    },
                    "notes": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    },
}
TABLE_USER = ("Transcribe every table on this page cell by cell (title, row-label header, "
              "column headers, every row's label and cells as printed, notes).")

CHART_SYSTEM = (
    "You transcribe charts from Taiwanese government briefing slides (Water Resources Agency, "
    "Ministry of Economic Affairs). For every chart on the page, read the value of each plotted "
    "marker or bar off the axis it belongs to, for every x-axis label (ROC years such as 100 = "
    "2011). Report what is plotted only: do not interpolate, extrapolate, smooth, sum or "
    "otherwise compute anything; if a series has no marker or bar at an x label, give null. "
    "Read each value as precisely as the gridlines allow (about 1 % of the axis range). For "
    "each series give the axis it is read on ('left' or 'right'), and give both axes' ranges "
    "and units. Copy chart titles, legend labels and axis titles exactly as printed "
    "(Traditional Chinese). Also copy verbatim every sentence on the page outside the chart.")

CHART_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["charts", "printed_sentences"],
    "properties": {
        "charts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["title", "left_axis_title", "left_axis_min", "left_axis_max",
                             "right_axis_title", "right_axis_min", "right_axis_max",
                             "x_labels", "series"],
                "properties": {
                    "title": {"type": "string"},
                    "left_axis_title": {"type": "string"},
                    "left_axis_min": {"type": ["number", "null"]},
                    "left_axis_max": {"type": ["number", "null"]},
                    "right_axis_title": {"type": "string"},
                    "right_axis_min": {"type": ["number", "null"]},
                    "right_axis_max": {"type": ["number", "null"]},
                    "x_labels": {"type": "array", "items": {"type": "string"}},
                    "series": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["legend_label", "kind", "axis", "points"],
                            "properties": {
                                "legend_label": {"type": "string"},
                                "kind": {"type": "string"},
                                "axis": {"type": "string", "enum": ["left", "right"]},
                                "points": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "required": ["x_label", "value"],
                                        "properties": {
                                            "x_label": {"type": "string"},
                                            "value": {"type": ["number", "null"]},
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            },
        },
        "printed_sentences": {"type": "array", "items": {"type": "string"}},
    },
}
CHART_USER = ("Transcribe every chart on this page (title, both axes' titles and ranges, x "
              "labels, and for every series its legend label, kind (line/bar), axis and the "
              "value of every plotted point), and copy the sentences outside the chart.")

JOBS = [
    {"key": "p2_at2_2", "kind": "table"},
    {"key": "p2_at2_5", "kind": "table"},
    {"key": "b110_irr_yl", "kind": "chart"},
]


def load_key() -> None:
    """ANTHROPIC_API_KEY from the repository .env, as report_extract.py does (never printed)."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    for ln in (REPO / ".env").read_text(encoding="utf-8").splitlines():
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


def page_png_bytes(pdf: Path, page: int, dpi: int = K.RENDER_DPI) -> bytes:
    with tempfile.TemporaryDirectory() as td:
        stem = Path(td) / "p"
        subprocess.run(["pdftoppm", "-f", str(page), "-l", str(page), "-r", str(dpi), "-png",
                        "-singlefile", str(pdf), str(stem)], check=True)
        return Path(str(stem) + ".png").read_bytes()


def call(client, model: str, block: dict, kind: str) -> tuple[dict, dict, str]:
    system, schema, user = ((TABLE_SYSTEM, TABLE_SCHEMA, TABLE_USER) if kind == "table"
                            else (CHART_SYSTEM, CHART_SCHEMA, CHART_USER))
    resp = client.messages.create(
        model=model, max_tokens=16000, system=system,
        messages=[{"role": "user", "content": [block, {"type": "text", "text": user}]}],
        output_config={"effort": "high", "format": {"type": "json_schema", "schema": schema}})
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"{model} refused: {resp.stop_details}")
    text = "".join(b.text for b in resp.content if b.type == "text")
    return json.loads(text), resp.usage.model_dump(mode="json"), str(resp.stop_reason)


def read_page(job: dict, client, use_api: bool) -> dict:
    src, page = K.PAGES[job["key"]]
    pdf = K.SRC[src]
    cache = K.API_CACHE / f"{job['key']}.json"
    cached = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    jobs = {
        "pdf_read": (K.MODEL_PDF, lambda: {
            "type": "document", "source": {
                "type": "base64", "media_type": "application/pdf",
                "data": base64.b64encode(page_pdf_bytes(pdf, page)).decode()}}),
        "img_read": (K.MODEL_IMG, lambda: {
            "type": "image", "source": {
                "type": "base64", "media_type": "image/png",
                "data": base64.b64encode(page_png_bytes(pdf, page)).decode()}}),
    }
    for name, (model, make_block) in jobs.items():
        if name in cached or not use_api:
            continue
        out, usage, stop = call(client, model, make_block(), job["kind"])
        cached[name] = {"model": model, "kind": job["kind"], "result": out, "usage": usage,
                        "stop_reason": stop,
                        "called": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                        "source": {"file": pdf.name, "page": page}}
        K.API_CACHE.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(cached, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{job['key']} {name}: {model} in {usage.get('input_tokens')} / out "
              f"{usage.get('output_tokens')} tokens", flush=True)
    return cached


def load_cached(key: str) -> dict:
    return json.loads((K.API_CACHE / f"{key}.json").read_text(encoding="utf-8"))


def token_usage() -> dict:
    """Token usage per model over every cached P6 API call."""
    out: dict = {}
    for p in sorted(K.API_CACHE.glob("*.json")):
        for name, rec in json.loads(p.read_text(encoding="utf-8")).items():
            u = out.setdefault(rec["model"], {"calls": 0, "input_tokens": 0,
                                              "output_tokens": 0, "pages": []})
            u["calls"] += 1
            u["input_tokens"] += int(rec["usage"].get("input_tokens") or 0)
            u["output_tokens"] += int(rec["usage"].get("output_tokens") or 0)
            u["pages"].append(f"{p.stem}:{name}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-api", action="store_true")
    args = ap.parse_args(argv)
    client = None
    if not args.no_api:
        load_key()
        import anthropic
        client = anthropic.Anthropic(timeout=600.0)
    for job in JOBS:
        read_page(job, client, not args.no_api)
    print(json.dumps(token_usage(), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
