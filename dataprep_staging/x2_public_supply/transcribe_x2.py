"""Transcribe the WRA 112 Changhua-Yunlin briefing's public-well charts with the Claude API.

Pages (existing L0 ``wra_reports_misc/wra_cy_subsidence_results_112.pdf``, read only):

* ``b112_p23``: 彰化公井抽水量 (Changhua public-well Jan-May pumping 100-112, 萬噸; series
  彰化台水1~5月 and 農田水利署彰化管理處1~5月) and the township charts below it.
* ``b112_p26``: 雲林公井抽水量 (the same for Yunlin).
* ``b110_p10`` / ``b110_p19``: the same charts in the 110 briefing (2011-2021; a pre-origin
  publication, used for the pre-origin variant; the 112 briefing was created 2024-04).

The series cover the public wells under the action plan's control (the closure-ramp wells).
Two independent reads per page, as dataprep_staging/p6_public_wells/transcribe_p6.py does:
``pdf_read`` (claude-opus-5-5 on the page as a one-page PDF) and ``img_read``
(claude-sonnet-5-5 on the 200-dpi PNG). The model reads plotted values only; it never computes
or fills a value. Responses and token usage are cached in
``reports/X2_public_supply/evidence/api_cache`` (a cached read is never re-requested).
Reconciliation is in ``reconcile_charts.py``.

Usage: python -I transcribe_x2.py [--no-api]
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

from x2_public_supply import x2const as K  # noqa: E402

MODEL_PDF = "claude-opus-5-5"
MODEL_IMG = "claude-sonnet-5-5"
RENDER_DPI = 200
BRIEF112 = K.L0_MISC / "wra_cy_subsidence_results_112.pdf"
BRIEF110 = K.L0_MISC / "wra_cy_subsidence_results_110.pdf"
JOBS = {"b112_p23": (BRIEF112, 23), "b112_p26": (BRIEF112, 26),
        "b110_p10": (BRIEF110, 10), "b110_p19": (BRIEF110, 19)}

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
    "type": "object", "additionalProperties": False,
    "required": ["charts", "printed_sentences"],
    "properties": {
        "charts": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["title", "left_axis_title", "left_axis_min", "left_axis_max",
                         "right_axis_title", "right_axis_min", "right_axis_max", "x_labels",
                         "series"],
            "properties": {
                "title": {"type": "string"},
                "left_axis_title": {"type": "string"},
                "left_axis_min": {"type": ["number", "null"]},
                "left_axis_max": {"type": ["number", "null"]},
                "right_axis_title": {"type": "string"},
                "right_axis_min": {"type": ["number", "null"]},
                "right_axis_max": {"type": ["number", "null"]},
                "x_labels": {"type": "array", "items": {"type": "string"}},
                "series": {"type": "array", "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["legend_label", "kind", "axis", "points"],
                    "properties": {
                        "legend_label": {"type": "string"},
                        "kind": {"type": "string"},
                        "axis": {"type": "string", "enum": ["left", "right"]},
                        "points": {"type": "array", "items": {
                            "type": "object", "additionalProperties": False,
                            "required": ["x_label", "value"],
                            "properties": {"x_label": {"type": "string"},
                                           "value": {"type": ["number", "null"]}}}}}}}}}},
        "printed_sentences": {"type": "array", "items": {"type": "string"}},
    },
}
CHART_USER = ("Transcribe every chart on this page (title, both axes' titles and ranges, x "
              "labels, and for every series its legend label, kind (line/bar), axis and the "
              "value of every plotted point), and copy the sentences outside the chart.")


def load_key() -> None:
    """ANTHROPIC_API_KEY from the repository .env, as report_extract.py does (never printed)."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    for ln in (K.REPO / ".env").read_text(encoding="utf-8").splitlines():
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


def page_png_bytes(pdf: Path, page: int, dpi: int = RENDER_DPI) -> bytes:
    with tempfile.TemporaryDirectory() as td:
        stem = Path(td) / "p"
        subprocess.run(["pdftoppm", "-f", str(page), "-l", str(page), "-r", str(dpi), "-png",
                        "-singlefile", str(pdf), str(stem)], check=True)
        return Path(str(stem) + ".png").read_bytes()


def call(client, model: str, block: dict) -> tuple[dict, dict, str]:
    resp = client.messages.create(
        model=model, max_tokens=16000, system=CHART_SYSTEM,
        messages=[{"role": "user", "content": [block, {"type": "text", "text": CHART_USER}]}],
        output_config={"effort": "high",
                       "format": {"type": "json_schema", "schema": CHART_SCHEMA}})
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"{model} refused")
    text = "".join(b.text for b in resp.content if b.type == "text")
    return json.loads(text), resp.usage.model_dump(mode="json"), str(resp.stop_reason)


def read_page(key: str, client, use_api: bool) -> dict:
    pdf, page = JOBS[key]
    cache = K.API_CACHE / f"{key}.json"
    cached = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    makers = {
        "pdf_read": (MODEL_PDF, lambda: {"type": "document", "source": {
            "type": "base64", "media_type": "application/pdf",
            "data": base64.b64encode(page_pdf_bytes(pdf, page)).decode()}}),
        "img_read": (MODEL_IMG, lambda: {"type": "image", "source": {
            "type": "base64", "media_type": "image/png",
            "data": base64.b64encode(page_png_bytes(pdf, page)).decode()}}),
    }
    for name, (model, make) in makers.items():
        if name in cached or not use_api:
            continue
        out, usage, stop = call(client, model, make())
        cached[name] = {"model": model, "kind": "chart", "result": out, "usage": usage,
                        "stop_reason": stop,
                        "called": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                        "source": {"file": pdf.name, "page": page}}
        K.API_CACHE.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(cached, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{key} {name}: {model} in {usage.get('input_tokens')} / out "
              f"{usage.get('output_tokens')} tokens", flush=True)
    return cached


def token_usage() -> dict:
    out: dict = {}
    for p in sorted(K.API_CACHE.glob("*.json")):
        for rec in json.loads(p.read_text(encoding="utf-8")).values():
            u = out.setdefault(rec["model"], {"calls": 0, "input_tokens": 0, "output_tokens": 0})
            u["calls"] += 1
            u["input_tokens"] += int(rec["usage"].get("input_tokens") or 0)
            u["output_tokens"] += int(rec["usage"].get("output_tokens") or 0)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-api", action="store_true")
    a = ap.parse_args(argv)
    client = None
    if not a.no_api:
        load_key()
        import anthropic
        client = anthropic.Anthropic(timeout=600.0)
    for key in JOBS:
        read_page(key, client, not a.no_api)
    print(json.dumps(token_usage(), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
