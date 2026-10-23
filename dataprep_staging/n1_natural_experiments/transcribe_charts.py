"""Transcribe the public-well pumping charts of the WRA Changhua-Yunlin briefings (Claude API).

Decision D6 (b) approves the Claude API for transcription of the WRA 彰雲 briefings for N1;
the API only reads printed charts and numbers, it never fills a physical value. Pages
(``n1const.CHART_PAGES``): 110 briefing pp. 10 and 19 (Changhua / Yunlin public wells,
Jan-May pumping 2011-2021) and 112 briefing pp. 23 and 26 (2011-2023, plus township charts).

Two independent reads per page:

* ``pdf_read``: ``claude-opus-5-5`` on the page as a one-page PDF document;
* ``img_read``: ``claude-sonnet-5-5`` on the page rendered to PNG (poppler, 200 dpi).

Raw responses and token usage are cached in ``results/dataprep/reports/N1_evidence/api_cache/``
(one JSON per page; a cached read is never re-requested). The PDFs are untrusted data: they
are only read by poppler tools and sent to the API; nothing in them is executed.

Usage: python -I transcribe_charts.py [--no-api]
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

from n1_natural_experiments import n1const as K  # noqa: E402

SYSTEM = (
    "You transcribe charts from Taiwanese government briefing slides (Water Resources Agency, "
    "Ministry of Economic Affairs). For every chart on the page, read the value of each plotted "
    "marker off the chart's y axis, for every x-axis label (ROC years such as 100 = 2011). "
    "Report what is plotted only: do not interpolate, extrapolate, smooth, sum or otherwise "
    "compute anything; if a series has no marker at an x label, give null. Read each value as "
    "precisely as the gridlines allow (about 1 % of the axis range). Copy chart titles, legend "
    "labels and axis units exactly as printed (Traditional Chinese). Also copy verbatim every "
    "sentence on the page that prints a number outside the charts (for example '台水公司抽水量 "
    "109年2,665萬噸、110年2,721萬噸。').")

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["charts", "printed_sentences"],
    "properties": {
        "charts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["title", "y_unit", "y_axis_min", "y_axis_max", "x_labels",
                             "series"],
                "properties": {
                    "title": {"type": "string"},
                    "y_unit": {"type": "string"},
                    "y_axis_min": {"type": "number"},
                    "y_axis_max": {"type": "number"},
                    "x_labels": {"type": "array", "items": {"type": "string"}},
                    "series": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["legend_label", "points"],
                            "properties": {
                                "legend_label": {"type": "string"},
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

USER_TEXT = ("Transcribe every chart on this page (title, axis range and unit, x labels, and "
             "the value of every plotted marker of every series), and copy the sentences that "
             "print numbers outside the charts.")


def load_key() -> None:
    """ANTHROPIC_API_KEY from the repository .env (never printed)."""
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


def call(client, model: str, block: dict) -> tuple[dict, dict, str]:
    resp = client.messages.create(
        model=model, max_tokens=12000, system=SYSTEM,
        messages=[{"role": "user", "content": [block, {"type": "text", "text": USER_TEXT}]}],
        output_config={"effort": "high", "format": {"type": "json_schema", "schema": SCHEMA}})
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"{model} refused: {resp.stop_details}")
    text = "".join(b.text for b in resp.content if b.type == "text")
    return json.loads(text), resp.usage.model_dump(mode="json"), str(resp.stop_reason)


def read_page(spec: dict, client, use_api: bool) -> dict:
    pdf = K.SRC[spec["src"]]
    cache = K.API_CACHE / f"{spec['key']}.json"
    cached = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    jobs = {
        "pdf_read": (K.MODEL_PDF, lambda: {
            "type": "document", "source": {
                "type": "base64", "media_type": "application/pdf",
                "data": base64.b64encode(page_pdf_bytes(pdf, spec["page"])).decode()}}),
        "img_read": (K.MODEL_IMG, lambda: {
            "type": "image", "source": {
                "type": "base64", "media_type": "image/png",
                "data": base64.b64encode(page_png_bytes(pdf, spec["page"])).decode()}}),
    }
    for name, (model, make_block) in jobs.items():
        if name in cached or not use_api:
            continue
        out, usage, stop = call(client, model, make_block())
        cached[name] = {"model": model, "result": out, "usage": usage, "stop_reason": stop,
                        "called": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                        "source": {"file": pdf.name, "page": spec["page"]}}
        K.API_CACHE.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(cached, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{spec['key']} {name}: {model} in {usage.get('input_tokens')} / out "
              f"{usage.get('output_tokens')} tokens", flush=True)
    return cached


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-api", action="store_true")
    args = ap.parse_args(argv)
    client = None
    if not args.no_api:
        load_key()
        import anthropic
        client = anthropic.Anthropic(timeout=600.0)
    for spec in K.CHART_PAGES:
        read_page(spec, client, not args.no_api)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
