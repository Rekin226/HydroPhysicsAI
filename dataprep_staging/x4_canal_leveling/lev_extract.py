"""Transcribe the WRA leveling tables and statements with the Claude API, twice, and reconcile.

    python -I lev_extract.py <plan.json> <out_dir> [--workers 4]

Follows hydrophysics/twin/report_extract.py: for every page of the plan (lev_pages.py) the
printed numbers are transcribed (a) from the page as a one-page PDF by ``MODEL_PDF`` and
(b) from the poppler layout text of the page by ``MODEL_TEXT``. Records are matched on a
kind-specific key and a field is accepted only when both transcriptions give the same value;
on any disagreement a third transcription (``MODEL_PDF`` on the text) is made. Each field
of a record is accepted when the PDF transcription agrees with either text transcription
(``value`` holds the accepted fields, None elsewhere; ``verified`` = every field accepted). The model
transcribes; it never computes, fills or extrapolates a value (system prompt), and every
value keeps its page reference. Raw responses are cached per page under ``<out_dir>/raw``.

The API key is read from the repository-root ``.env`` into the process environment and is
never printed. Downloads are untrusted data; only their text and page bytes are sent.
"""
from __future__ import annotations

import base64
import concurrent.futures as cf
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import REPO, rel_to_repo  # noqa: E402

MODEL_PDF = "claude-opus-5-5"
MODEL_TEXT = "claude-sonnet-5-5"
NUM = {"type": ["number", "null"]}
STR = {"type": ["string", "null"]}


def _obj(props: dict) -> dict:
    return {"type": "object", "additionalProperties": False, "required": list(props),
            "properties": props}


SCHEMAS = {
    "area_county": _obj({"rows": {"type": "array", "items": _obj({
        "county": {"type": "string", "description": "縣市 as printed, e.g. 彰化, 雲林, 合計"},
        "year_label": {"type": "string", "description": "column year label as printed, e.g. "
                                                        "'90(91)' or '104'"},
        "area_km2": NUM})}}),
    "max_rate": _obj({"rows": {"type": "array", "items": _obj({
        "region": {"type": "string", "description": "地區 row as printed; for 雲林 write "
                                                    "'雲林沿海' or '雲林內陸' when split"},
        "roc_year": {"type": "integer"}, "rate_cm_yr": NUM,
        "marked": {"type": "boolean", "description": "true if the value carries a * mark"}})}}),
    "survey_dates": _obj({"table_roc_year": {"type": ["integer", "null"]},
                          "rows": {"type": "array", "items": _obj({
                              "region": {"type": "string"}, "start": STR, "end": STR,
                              "note": STR})}}),
    "interval": _obj({
        "county": {"type": "string", "description": "彰化 or 雲林 from the table title"},
        "intervals": {"type": "array", "items": _obj({
            "start": {"type": "string", "description": "觀測期距 start as printed, e.g. '100.5'"},
            "end": {"type": "string", "description": "觀測期距 end as printed, e.g. '101.5'"},
            "max_rate_cm_yr": NUM,
            "max_rate_township": {"type": ["string", "null"],
                                  "description": "最大下陷速率發生地點, e.g. 溪州鄉"},
            "area_gt2_km2": {"type": ["number", "null"],
                             "description": "速率超過2cm/yr之面積; null if printed '-' or blank"},
            "area_gt3_km2": {"type": ["number", "null"],
                             "description": "速率超過3cm/yr之面積"},
            "class_areas": {"type": "array", "items": _obj({
                "rate_class": {"type": "string", "description": "e.g. '2.0~3.0cm'"},
                "km2": NUM})}})}}),
    "statement": _obj({"statements": {"type": "array", "items": _obj({
        "county": {"type": "string", "description": "彰化 or 雲林 only"},
        "survey_roc_year": {"type": ["integer", "null"],
                            "description": "the year of the leveling result described (the "
                                           "later year of a span such as 111~112年)"},
        "townships_over_3cm": {"type": "array", "items": {"type": "string"},
                               "description": "townships the text says exceed 3 cm/yr or form "
                                              "the significant-subsidence area, as printed"},
        "says_none_over_3cm": {"type": "boolean"},
        "max_rate_cm_yr": NUM, "max_rate_township": STR, "significant_area_km2": NUM})}}),
    "hsr": _obj({"rows": {"type": "array", "items": _obj({
        "location": {"type": "string"}, "chainage": {"type": "string"},
        "structure": STR, "roc_year": {"type": "integer"}, "settlement_cm": NUM,
        "angular_ratio": STR})}}),
}
KIND_SCHEMA = {"interval_chc": "interval", "interval_yl": "interval",
               "brief_statement": "statement", "statement": "statement"}

SYSTEM = (
    "You transcribe printed tables and sentences from a Taiwanese government report "
    "(Water Resources Agency, land subsidence). Copy numbers and names exactly as printed. "
    "Never compute, round, convert, infer or fill a value: if a value is not printed, or is "
    "printed as '-', '--' or blank, return null. Do not transcribe groundwater levels or "
    "rainfall. Return only the JSON."
)
TASK = {
    "area_county": "Transcribe table 歷年主要縣市顯著下陷面積統計表 (significant subsidence area, "
                   "km2, by county and year column).",
    "max_rate": "Transcribe table 地層下陷地區最大年下陷速率統計表 (maximum annual subsidence rate "
                "by region and year, cm/yr).",
    "survey_dates": "Transcribe table 水準檢測外業期程 (leveling field-survey dates by region).",
    "interval": "Transcribe the 下陷面積分析表 on this page: one entry per 觀測期距 column, with "
                "the maximum subsidence rate, its township (the vertical characters under the "
                "rate, e.g. 溪/州/鄉 -> 溪州鄉), the areas over 2 and over 3 cm/yr and every "
                "rate-class area row printed for that column.",
    "statement": "Transcribe the statements about the Changhua (彰化) and Yunlin (雲林) "
                 "leveling results on this page: for each county and survey year, the townships "
                 "said to exceed 3 cm/yr (or listed as the significant-subsidence area), whether "
                 "the text says none exceeds 3 cm/yr, the maximum rate and its township, and the "
                 "significant-subsidence area. Ignore other counties.",
    "hsr": "Transcribe the 高鐵 (high-speed rail) settlement table: one row per pier location "
           "and year, the annual settlement in cm as printed (keep the sign) and the angular "
           "ratio in parentheses.",
}


def load_key() -> None:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    env = REPO / ".env"
    for line in env.read_text(encoding="utf-8").splitlines():
        if line.startswith("ANTHROPIC_API_KEY="):
            os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip().strip('"')
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not found in .env")


def page_text(pdf: str, page: int) -> str:
    return subprocess.run(["pdftotext", "-f", str(page), "-l", str(page), "-layout", pdf, "-"],
                          check=True, capture_output=True, text=True).stdout


def page_pdf(pdf: str, page: int) -> bytes:
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "p.pdf"
        subprocess.run(["pdfseparate", "-f", str(page), "-l", str(page), pdf, str(out)],
                       check=True)
        return out.read_bytes()


def transcribe(client, model: str, schema: dict, content: list[dict]) -> tuple[dict, dict]:
    resp = client.messages.create(
        model=model, max_tokens=16000, system=SYSTEM,
        messages=[{"role": "user", "content": content}],
        output_config={"effort": "medium",
                       "format": {"type": "json_schema", "schema": schema}})
    if resp.stop_reason == "refusal":
        return {"_refusal": True}, resp.usage.model_dump(mode="json")
    text = "".join(b.text for b in resp.content if b.type == "text")
    return json.loads(text), resp.usage.model_dump(mode="json")


def pdf_content(data: bytes, task: str) -> list[dict]:
    return [{"type": "document", "source": {"type": "base64", "media_type": "application/pdf",
                                            "data": base64.b64encode(data).decode()}},
            {"type": "text", "text": task}]


def text_content(text: str, task: str) -> list[dict]:
    return [{"type": "text", "text": task + " Below is the poppler layout text of the page; "
             "columns may be misaligned, so match values to columns carefully.\n\n" + text}]


# ----------------------------------------------------------------------------- records
def _n(x):
    if x is None:
        return None
    try:
        return round(float(x), 3)
    except (TypeError, ValueError):
        return str(x).strip()


def _s(x):
    return None if x is None else "".join(str(x).split())


def _cls(x) -> str:
    """Rate-class label -> 'a-b' (e.g. '3.0～5.0cm/yr' and '3.0~5.0cm' -> '3.0-5.0')."""
    t = _s(x) or ""
    t = t.replace("～", "~").replace("〜", "~").replace("公分/年", "").replace("cm/yr", "")
    t = t.replace("cm", "").replace("~", "-")
    m = re.findall(r"\d+(?:\.\d+)?", t)
    return "-".join(f"{float(v):.1f}" for v in m) if m else t


def _chain(x) -> str:
    m = re.search(r"TK\s*\d+\s*\+\s*\d+", str(x or ""))
    return "".join(m[0].split()) if m else (_s(x) or "")


def _loc(x) -> str | None:
    return None if x is None else re.sub(r"[（(].*?[)）]", "", _s(x))


def records(kind: str, table: dict) -> dict:
    """{key: comparable record} for one transcription."""
    k = KIND_SCHEMA.get(kind, kind)
    out = {}
    if table.get("_refusal"):
        return out
    if k == "area_county":
        for r in table.get("rows", []):
            out[(_s(r["county"]), _s(r["year_label"]))] = {"area_km2": _n(r["area_km2"])}
    elif k == "max_rate":
        for r in table.get("rows", []):
            out[(_s(r["region"]), int(r["roc_year"]))] = {"rate_cm_yr": _n(r["rate_cm_yr"]),
                                                          "marked": bool(r["marked"])}
    elif k == "survey_dates":
        for r in table.get("rows", []):
            out[(_s(r["region"]),)] = {"start": _s(r["start"]), "end": _s(r["end"]),
                                       "table_roc_year": table.get("table_roc_year")}
    elif k == "interval":
        cty = _s(table.get("county"))
        for r in table.get("intervals", []):
            cls = {_cls(c["rate_class"]): _n(c["km2"]) for c in r.get("class_areas", [])
                   if c.get("km2") is not None}
            out[(cty, _s(r["start"]), _s(r["end"]))] = {
                "max_rate_cm_yr": _n(r["max_rate_cm_yr"]),
                "max_rate_township": _s(r["max_rate_township"]),
                "area_gt2_km2": _n(r["area_gt2_km2"]), "area_gt3_km2": _n(r["area_gt3_km2"]),
                "class_areas": dict(sorted(cls.items()))}
    elif k == "statement":
        for r in table.get("statements", []):
            key = (_s(r["county"]), r.get("survey_roc_year"))
            out[key] = {"townships_over_3cm": sorted({_s(t) for t in r["townships_over_3cm"]}),
                        "says_none_over_3cm": bool(r["says_none_over_3cm"]),
                        "max_rate_cm_yr": _n(r["max_rate_cm_yr"]),
                        "max_rate_township": _s(r["max_rate_township"]),
                        "significant_area_km2": _n(r["significant_area_km2"])}
    elif k == "hsr":
        for r in table.get("rows", []):
            out[(_chain(r["chainage"]), int(r["roc_year"]))] = {
                "location": _loc(r["location"]), "settlement_cm": _n(r["settlement_cm"]),
                "angular_ratio": _s(r["angular_ratio"])}
    return out


def reconcile(kind: str, rec: dict) -> list[dict]:
    """Field-level reconciliation: a field of a record is verified when the PDF
    transcription's value equals the text (or second text) transcription's value."""
    a = records(kind, rec["pdf"]["table"])
    b = records(kind, rec["text"]["table"])
    c = records(kind, rec["text2"]["table"]) if "text2" in rec else {}
    rows = []
    for key in sorted(set(a) | set(b) | set(c), key=str):
        va, vb, vc = a.get(key), b.get(key), c.get(key)
        value, ok = {}, {}
        for f in (va or {}):
            hit = (vb is not None and vb.get(f) == va[f]) or (vc is not None
                                                              and vc.get(f) == va[f])
            ok[f] = bool(hit)
            value[f] = va[f] if hit else None
        agree_with = "text" if va is not None and va == vb else (
            "text2" if va is not None and va == vc else "")
        rows.append({"key": list(key) if isinstance(key, tuple) else key, "pdf": va,
                     "text": vb, "text2": vc, "value": value, "fields_verified": ok,
                     "verified": bool(va) and all(ok.values()),
                     "any_verified": any(ok.values()), "agree_with": agree_with})
    return rows


def run_page(client, item: dict, raw_dir: Path) -> dict:
    kind = item["kind"]
    sk = KIND_SCHEMA.get(kind, kind)
    tag = hashlib.sha256(f"{item['pdf']}|{item['page']}|{kind}".encode()).hexdigest()[:16]
    cache = raw_dir / f"{Path(item['pdf']).stem}_p{item['page']}_{kind}_{tag}.json"
    if cache.exists():
        rec = json.loads(cache.read_text(encoding="utf-8"))
    else:
        txt = page_text(item["pdf"], item["page"])
        ta, ua = transcribe(client, MODEL_PDF, SCHEMAS[sk],
                            pdf_content(page_pdf(item["pdf"], item["page"]), TASK[sk]))
        tb, ub = transcribe(client, MODEL_TEXT, SCHEMAS[sk], text_content(txt, TASK[sk]))
        rec = {"item": {**item, "pdf": rel_to_repo(item["pdf"])},
               "pdf": {"model": MODEL_PDF, "table": ta, "usage": ua},
               "text": {"model": MODEL_TEXT, "table": tb, "usage": ub}}
        cache.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    if "text2" not in rec and any(not r["verified"] for r in reconcile(kind, rec)):
        txt = page_text(item["pdf"], item["page"])
        tc, uc = transcribe(client, MODEL_PDF, SCHEMAS[sk], text_content(txt, TASK[sk]))
        rec["text2"] = {"model": MODEL_PDF, "table": tc, "usage": uc}
        cache.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"item": rec["item"], "rows": reconcile(kind, rec),
            "usage": {k: rec[k]["usage"] for k in ("pdf", "text", "text2") if k in rec}}


def main(argv: list[str]) -> int:
    import anthropic

    if len(argv) < 2:
        print(__doc__)
        return 2
    plan = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    out = Path(argv[1])
    workers = int(argv[argv.index("--workers") + 1]) if "--workers" in argv else 4
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    load_key()
    client = anthropic.Anthropic(timeout=900.0, max_retries=4)
    results, errors = [], []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_page, client, it, raw): it for it in plan}
        for f in cf.as_completed(futs):
            it = futs[f]
            try:
                r = f.result()
                results.append(r)
                nv = sum(x["verified"] for x in r["rows"])
                print(f"{Path(it['pdf']).name} p{it['page']} {it['kind']}: "
                      f"{nv}/{len(r['rows'])} verified", flush=True)
            except Exception as exc:  # noqa: BLE001 - recorded, page left out
                errors.append({"item": {**it, "pdf": rel_to_repo(it["pdf"])},
                               "error": repr(exc)[:500]})
                print(f"ERROR {Path(it['pdf']).name} p{it['page']}: {exc!r}"[:300], flush=True)
    results.sort(key=lambda r: (r["item"]["pdf"], r["item"]["page"], r["item"]["kind"]))
    tok = {k: sum(u.get(k, 0) or 0 for r in results for u in r["usage"].values())
           for k in ("input_tokens", "output_tokens")}
    (out / "transcriptions.json").write_text(json.dumps(
        {"schema": "x4-transcriptions/1", "models": {"pdf": MODEL_PDF, "text": MODEL_TEXT},
         "pages": results, "errors": errors, "tokens": tok}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(json.dumps({"pages": len(results), "errors": len(errors), "tokens": tok}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
