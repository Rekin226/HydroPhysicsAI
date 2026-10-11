"""Find public data that could close the twin's evidence gaps, with the Claude API.

The audit of 2026-10-11 found the two science scores data-limited: head amplitude needs
forcing the model lacks (canal deliveries, metered pumping, gauge rain) and the policy
mechanism needs subsidence observations past 2021 (leveling, deep benchmarks, InSAR).
This scout asks Claude, with its server-side web search and web fetch tools, to locate
*downloadable* public sources for a list of targets and to return, for each lead, the
URL, what the resource contains (quoted from the page), its period and granularity, and
how it was verified. It finds and describes; it never produces data values. Every lead is
then checked here by fetching the URL and recording the HTTP status and size, so that
what reaches the repository is a verified pointer, not a model's recollection.

Needs ``ANTHROPIC_API_KEY`` in the environment (``.env`` is gitignored; source it first).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import urllib.request
from pathlib import Path

TARGETS = {
    "jiji_intake_history": (
        "Historical DAILY or MONTHLY intake (取水量/引水量) of 集集攔河堰 (Jiji Weir) on the "
        "濁水溪, separately for the 北岸聯絡渠道 (Changhua irrigation) and 南岸聯絡渠道 (Yunlin "
        "irrigation), for 2012-2026. Candidate sources: 經濟部水利署 水文年報 PDFs, 水利署 "
        "中區水資源分署/集集攔河堰管理中心 pages, data.gov.tw / opendata.wra.gov.tw datasets "
        "(水庫每日營運狀況 id 20503 is latest-day only), fhy.wra.gov.tw history pages."),
    "irrigation_deliveries": (
        "Canal delivery / irrigation water supply statistics (灌溉用水量, 供水量) of the "
        "農田水利署 彰化管理處 and 雲林管理處 by year or month or by 工作站, 2012-2026."),
    "leveling_2022_plus": (
        "Precise leveling results (水準測量成果) for the 彰化/雲林 land-subsidence network "
        "for survey years 2022, 2023, 2024, 2025: per-benchmark elevations or annual "
        "subsidence, or at least the published annual subsidence-rate maps/contours with "
        "downloadable files. Sources: 水利署 地層下陷監測資訊整合服務系統 "
        "(web.wra.gov.tw/landsubsidence), data.gov.tw datasets 58361/58362 (等值線), "
        "國土測繪中心 地層下陷區水準網計算成果, 交通部鐵道局 高鐵彰雲嘉水準測量報告, "
        "成功大學 地層下陷防治服務團 (lsprc.ncku.edu.tw) annual reports."),
    "tuku_trial_records": (
        "Records of the 土庫 (Tuku) / 高鐵沿線 groundwater-pumping reduction or well-closure "
        "trials and farmland conversion (黃金廊道, 轉旱作, 封井) with dates, areas and the "
        "measured response, 2013-2026: official reports or open data."),
    "insar_2023_plus": (
        "Any downloadable InSAR deformation time series or velocity product covering the "
        "Choushui River alluvial fan (彰化/雲林) for 2023 or later: Sentinel-1 products "
        "(LiCSAR frames, ASF HyP3 bursts, EGMS-like national services, 中央地質調查所/地礦中心 "
        "releases, journal data-availability links)."),
}

SYSTEM = (
    "You are a data scout for a hydrogeology research group in Taiwan. You search the "
    "public web (search in Traditional Chinese and English) for DOWNLOADABLE data that "
    "matches each target. For every lead you must have opened the page with web_fetch and "
    "you must quote, verbatim, the sentence or table header that proves what it contains. "
    "Never invent a URL, a dataset, a field name or a number; if you did not open it, say "
    "so and mark it unverified. Prefer official sources (gov.tw, edu.tw, journals). Report "
    "dead ends too. Finish with ONE fenced ```json block: a list of objects with keys "
    "target, title, url, publisher, contains (your words), evidence_quote (verbatim), "
    "period, granularity, format, access (open/registration/request), verified "
    "(true only if fetched), notes."
)


def scout(targets: dict[str, str], model: str, max_searches: int, max_fetches: int,
          search_type: str, fetch_type: str) -> dict:
    import anthropic  # local import so the module imports without the SDK
    client = anthropic.Anthropic(timeout=900.0)
    prompt = "Targets:\n" + "\n".join(f"{i + 1}. [{k}] {v}" for i, (k, v) in enumerate(targets.items()))
    prompt += ("\n\nWork target by target. Use web_search to find candidates and web_fetch "
               "to open the most promising pages (including PDFs) before listing them. "
               "Then write the final JSON block.")
    tools = [{"type": search_type, "name": "web_search", "max_uses": max_searches},
             {"type": fetch_type, "name": "web_fetch", "max_uses": max_fetches,
              "citations": {"enabled": True}}]
    resp = client.messages.create(model=model, max_tokens=16000, system=SYSTEM,
                                  messages=[{"role": "user", "content": prompt}], tools=tools)
    return resp.model_dump(mode="json")


def final_text(response: dict) -> str:
    return "\n".join(b.get("text", "") for b in response.get("content", [])
                     if b.get("type") == "text")


def parse_leads(text: str) -> list[dict]:
    blocks = re.findall(r"```json\s*(.*?)```", text, flags=re.S)
    for block in reversed(blocks):
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            data = data.get("leads", [data])
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict) and d.get("url")]
    return []


def check_url(url: str, timeout: int = 60) -> dict:
    """HEAD-like probe: status, content type and size of the first bytes, no parsing."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (data scout)"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 (http(s) only)
            head = r.read(4096)
            return {"status": r.status, "content_type": r.headers.get("Content-Type", ""),
                    "content_length": r.headers.get("Content-Length"),
                    "looks_like": ("pdf" if head[:4] == b"%PDF" else
                                   "html" if b"<html" in head.lower() else "other")}
    except Exception as exc:  # noqa: BLE001 (report, do not raise)
        return {"status": None, "error": str(exc)[:200]}


def run(out: Path, model: str, targets: list[str] | None, max_searches: int, max_fetches: int,
        search_type: str, fetch_type: str, verify: bool = True) -> dict:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set (source .env)")
    chosen = {k: TARGETS[k] for k in (targets or TARGETS)}
    response = scout(chosen, model, max_searches, max_fetches, search_type, fetch_type)
    out.mkdir(parents=True, exist_ok=True)
    (out / "response.json").write_text(json.dumps(response, ensure_ascii=False, indent=1,
                                                  default=str))
    text = final_text(response)
    (out / "report.md").write_text(text)
    leads = parse_leads(text)
    if verify:
        for lead in leads:
            lead["http_check"] = check_url(str(lead["url"]))
    summary = {"ran_on": dt.date.today().isoformat(), "model": model, "targets": list(chosen),
               "usage": response.get("usage"), "stop_reason": response.get("stop_reason"),
               "leads": leads,
               "note": "Leads are pointers found by a language model and probed by HTTP; "
                       "none of their contents has been read into the repository by this run."}
    (out / "leads.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    return summary


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--targets", nargs="*", choices=list(TARGETS), default=None)
    ap.add_argument("--max-searches", type=int, default=25)
    ap.add_argument("--max-fetches", type=int, default=25)
    ap.add_argument("--search-type", default="web_search_20250305")
    ap.add_argument("--fetch-type", default="web_fetch_20260318")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args(argv)
    summary = run(args.out, args.model, args.targets, args.max_searches, args.max_fetches,
                  args.search_type, args.fetch_type, verify=not args.no_verify)
    print(json.dumps({k: v for k, v in summary.items() if k != "leads"}, indent=1))
    for lead in summary["leads"]:
        print(f"- [{lead.get('target')}] {lead.get('title')} -> {lead.get('url')} "
              f"({lead.get('http_check', {}).get('status')})")


if __name__ == "__main__":
    main()
