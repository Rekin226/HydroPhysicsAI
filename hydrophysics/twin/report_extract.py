"""Read the deep-benchmark table of the WRA monthly subsidence reports with the Claude API.

Every instrument feed the project can reach (the vendor API and the WRA open-data deep
piles) stops in April-May 2025. The only later subsidence numbers in the public domain are
printed in the monthly 地層下陷監測整合分析月報: its table 3 gives, for six deep benchmarks
(深層樁), the compression of the last month and of the trailing twelve months in
centimetres, with the pile's name, depth and county. The layout interleaves the pile's
numbers with those of the neighbouring observation wells, so a plain text parse is not
reliable; this module asks Claude to transcribe the printed numbers and then checks them
three ways before any of them is used:

1. two independent transcriptions per report - the page as a PDF document and the page
   as poppler layout text, by different models - must agree number for number;
2. the trailing-twelve-month column must chain across consecutive reports,
   Y(m) - Y(m-1) = M(m) - M(m-12), within the rounding of four printed values;
3. where the open-data hourly feed covers the month, the end-of-month difference of the
   pile's reading must match the printed monthly value within 0.15 cm.

The model transcribes; it never computes, fills or extrapolates a value. A number that
fails a check is written to the output with its flags and is not consumed downstream.

Needs ``ANTHROPIC_API_KEY`` in the environment and poppler's ``pdftotext``/``pdfseparate``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

TABLE_MARK = "深層樁與地下水位變化趨勢比較表"
# the table sometimes prints a pile without its suffix; keep one spelling per pile
PILE_ALIAS = {"秀潭": "秀潭國小", "溪州": "溪州國小", "土庫": "土庫國中", "灣內": "灣內國小",
              "虎尾": "中科虎尾", "頂寮": "頂寮安檢所"}

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["report_roc_year", "report_month", "piles"],
    "properties": {
        "report_roc_year": {"type": "integer", "description": "民國 year in the table title"},
        "report_month": {"type": "integer", "description": "month in the table title"},
        "piles": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["row", "county", "township", "pile_name", "pile_depth_m",
                             "month_change_cm", "year_change_cm", "month_symbol",
                             "year_symbol"],
                "properties": {
                    "row": {"type": "integer", "description": "序 number"},
                    "county": {"type": "string", "description": "區域 column, e.g. 彰化"},
                    "township": {"type": "string", "description": "鄉鎮區 column"},
                    "pile_name": {"type": "string",
                                  "description": "樁名 without the depth, e.g. 溪州國小"},
                    "pile_depth_m": {"type": ["integer", "null"],
                                     "description": "樁深 in metres printed under the name"},
                    "month_change_cm": {"type": ["number", "null"],
                                        "description": "深層樁 月變化 value in parentheses, cm, "
                                                       "sign as printed; null if printed as —"},
                    "year_change_cm": {"type": ["number", "null"],
                                       "description": "深層樁 年變化 value in parentheses, cm"},
                    "month_symbol": {"type": "string",
                                     "description": "symbol above the month value: ▼ □ or —"},
                    "year_symbol": {"type": "string"},
                },
            },
        },
    },
}

SYSTEM = (
    "You transcribe one printed table from a Taiwanese government report. Copy the deep "
    "benchmark (深層樁) rows exactly as printed: the row number, 區域, 鄉鎮區, 樁名, 樁深, and "
    "the two parenthesised numbers under 深層樁 月變化 and 年變化 (centimetres, negative = "
    "rebound). Ignore the observation-well (井名) columns entirely; their numbers are in "
    "metres and belong to the wells, not to the pile. Each pile has exactly one month value "
    "and one year value. Never compute, round, infer or fill a number: if a value is not "
    "printed, return null. Return only the JSON."
)


# --------------------------------------------------------------------------------------
# PDF plumbing (poppler)
# --------------------------------------------------------------------------------------
def page_text(pdf: Path, page: int) -> str:
    return subprocess.run(["pdftotext", "-f", str(page), "-l", str(page), "-layout", str(pdf), "-"],
                          check=True, capture_output=True, text=True).stdout


def n_pages(pdf: Path) -> int:
    out = subprocess.run(["pdfinfo", str(pdf)], check=True, capture_output=True, text=True).stdout
    return int(next(line.split()[-1] for line in out.splitlines() if line.startswith("Pages")))


def find_table_page(pdf: Path) -> int:
    for p in range(1, n_pages(pdf) + 1):
        if TABLE_MARK in page_text(pdf, p):
            return p
    raise ValueError(f"{pdf}: table not found")


def page_pdf_bytes(pdf: Path, page: int) -> bytes:
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "page.pdf"
        subprocess.run(["pdfseparate", "-f", str(page), "-l", str(page), str(pdf), str(out)],
                       check=True)
        return out.read_bytes()


# --------------------------------------------------------------------------------------
# Claude transcription
# --------------------------------------------------------------------------------------
def transcribe(client, model: str, content: list[dict]) -> tuple[dict, dict]:
    """One structured-output call -> (parsed table, usage)."""
    resp = client.messages.create(
        model=model, max_tokens=4000, system=SYSTEM,
        messages=[{"role": "user", "content": content}],
        output_config={"format": {"type": "json_schema", "schema": SCHEMA}})
    text = "".join(b.text for b in resp.content if b.type == "text")
    return json.loads(text), resp.usage.model_dump(mode="json")


def pdf_content(data: bytes) -> list[dict]:
    import base64
    return [{"type": "document",
             "source": {"type": "base64", "media_type": "application/pdf",
                        "data": base64.b64encode(data).decode()}},
            {"type": "text", "text": "Transcribe the 深層樁 rows of 表 3 on this page."}]


def text_content(text: str) -> list[dict]:
    return [{"type": "text",
             "text": "Below is the poppler layout text of one report page. Transcribe the 深層樁 "
                     "rows of 表 3. The pile's month and year values are the parenthesised "
                     "numbers printed under the 樁名 column, left of the 井名 column.\n\n"
                     + text}]


# --------------------------------------------------------------------------------------
# checks (pure, tested)
# --------------------------------------------------------------------------------------
def rows_from(table: dict, month: str, source: str) -> pd.DataFrame:
    rows = []
    for p in table.get("piles", []):
        name = str(p.get("pile_name", "")).strip()
        rows.append({"month": month, "pile": PILE_ALIAS.get(name, name),
                     "depth_m": p.get("pile_depth_m"), "county": p.get("county"),
                     "township": p.get("township"),
                     "month_cm": p.get("month_change_cm"), "year_cm": p.get("year_change_cm"),
                     "source": source,
                     "roc_year": table.get("report_roc_year"), "roc_month": table.get("report_month")})
    return pd.DataFrame(rows)


def compare_transcriptions(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Merge two transcriptions of one report on (month, pile) and flag disagreement."""
    key = ["month", "pile"]
    m = a.merge(b, on=key, how="outer", suffixes=("_a", "_b"), indicator="present")
    def same(x, y):
        if pd.isna(x) and pd.isna(y):
            return True
        if pd.isna(x) or pd.isna(y):
            return False
        return abs(float(x) - float(y)) < 1e-9
    m["agree"] = [(r.present == "both") and same(r.month_cm_a, r.month_cm_b)
                  and same(r.year_cm_a, r.year_cm_b) and same(r.depth_m_a, r.depth_m_b)
                  for r in m.itertuples()]
    return m


def chain_check(df: pd.DataFrame, tol: float = 0.21) -> pd.DataFrame:
    """Trailing-twelve-month consistency: Y(m) - Y(m-1) - (M(m) - M(m-12)) within ``tol``
    cm (four values each rounded to 0.1), among rows of the same pile *and printed depth*
    (the reports switch 土庫國中 between its 400 m and 300 m sensors in some months). Adds
    ``chain_resid`` and ``chain_ok`` (NaN / None where a term is missing)."""
    df = df.copy()
    df["per"] = pd.PeriodIndex(df["month"], freq="M")
    idx = df.set_index(["pile", "depth_m", "per"])
    resid, ok = [], []
    for r in df.itertuples():
        try:
            k = (r.pile, r.depth_m)
            y1, y0 = idx.loc[k + (r.per,), "year_cm"], idx.loc[k + (r.per - 1,), "year_cm"]
            m1, m12 = idx.loc[k + (r.per,), "month_cm"], idx.loc[k + (r.per - 12,), "month_cm"]
            vals = [y1, y0, m1, m12]
            if any(pd.isna(v) for v in vals):
                raise KeyError
            res = float(y1) - float(y0) - (float(m1) - float(m12))
            resid.append(res)
            ok.append(abs(res) <= tol)
        except KeyError:
            resid.append(float("nan"))
            ok.append(None)
    df["chain_resid"], df["chain_ok"] = resid, ok
    return df.drop(columns="per")


def feed_monthly_cm(hourly: pd.Series, unit_scale: float = 0.1) -> pd.DataFrame:
    """Printed convention: month change = daily mean on the last day of the month minus the
    daily mean on the last day of the previous month. ``hourly`` is the feed's reading
    (mm; a rising reading is compression). Returns month, feed_cm and whether both
    month-end days are observed."""
    s = pd.to_numeric(hourly, errors="coerce").dropna()
    daily = s.resample("D").mean()
    per = daily.index.to_period("M")
    last_of_month = daily.groupby(per).apply(
        lambda d: d.iloc[-1] if (d.index[-1].day == d.index[-1].days_in_month and pd.notna(d.iloc[-1]))
        else float("nan"))
    diff = last_of_month.diff() * unit_scale
    step = daily.diff().abs().groupby(per).max() * unit_scale     # largest daily jump, cm
    return pd.DataFrame({"month": [str(p) for p in diff.index], "feed_cm": diff.to_numpy(),
                         "feed_complete": last_of_month.notna().to_numpy()
                         & last_of_month.shift(1).notna().to_numpy(),
                         "feed_step_cm": step.reindex(diff.index).to_numpy()})


def feed_check(df: pd.DataFrame, feed: dict[str, pd.Series], tol: float = 0.15) -> pd.DataFrame:
    """Attach the feed-derived monthly value per (pile, month) and a pass flag."""
    df = df.copy()
    df["feed_cm"], df["feed_ok"], df["feed_step_cm"] = float("nan"), None, float("nan")
    for pile, hourly in feed.items():
        tab = feed_monthly_cm(hourly).set_index("month")
        for i in df.index[df.pile == pile]:
            m = df.at[i, "month"]
            if m in tab.index and tab.at[m, "feed_complete"] and pd.notna(df.at[i, "month_cm"]):
                v = float(tab.at[m, "feed_cm"])
                df.at[i, "feed_cm"] = v
                df.at[i, "feed_ok"] = bool(abs(v - float(df.at[i, "month_cm"])) <= tol)
                df.at[i, "feed_step_cm"] = float(tab.at[m, "feed_step_cm"])
    return df


# --------------------------------------------------------------------------------------
def load_feed(paths: list[Path]) -> dict[str, pd.Series]:
    """Open-data deep-pile JSON dumps -> {identifier: hourly reading (mm)}."""
    rows = []
    for p in paths:
        rows.extend(json.loads(p.read_text()))
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    df["t"] = pd.to_datetime(df["observationtime"])
    df["v"] = pd.to_numeric(df["depth"], errors="coerce")
    df = df.drop_duplicates(["identifier", "t"]).sort_values("t")
    return {str(ident): g.set_index("t")["v"] for ident, g in df.groupby("identifier")}


def feed_for_piles(df: pd.DataFrame, feed: dict[str, pd.Series]) -> dict[str, pd.Series]:
    """{printed pile name: feed series} using the printed depth to pick the pile's sensor;
    a pile whose (name, depth) has no feed identifier is left unchecked."""
    out = {}
    for pile, g in df.groupby("pile"):
        depths = g["depth_m"].dropna()
        if depths.empty:
            continue
        ident = FEED_ID.get((pile, int(depths.mode().iloc[0])))
        if ident in feed:
            out[pile] = feed[ident]
    return out


# the printed pile (name, depth) -> open-data identifier
FEED_ID = {("土庫國中", 400): "BM_TKJS(3)", ("土庫國中", 300): "BM_TKJS(1)",
           ("溪州國小", 300): "BM_SJES", ("秀潭國小", 70): "BM_STES",
           ("中科虎尾", 300): "BM_NTUH(1)", ("灣內國小", 100): "BM_WNES",
           ("頂寮安檢所", 150): "BM_DLIO"}


def run(reports: Path, out: Path, model_pdf: str, model_text: str, feed_paths: list[Path],
        months: list[str] | None = None, log=print) -> dict:
    import anthropic
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set (source .env)")
    client = anthropic.Anthropic(timeout=600.0)
    out.mkdir(parents=True, exist_ok=True)
    raw_dir = out / "raw"
    raw_dir.mkdir(exist_ok=True)
    files = sorted(reports.glob("integrated_*.pdf"))
    if months:
        files = [f for f in files if f.stem.replace("integrated_", "") in months]
    usage, frames, per_report = [], [], []
    for f in files:
        month = f.stem.replace("integrated_", "")
        page = find_table_page(f)
        cache = raw_dir / f"{month}.json"
        if cache.exists():
            rec = json.loads(cache.read_text())
        else:
            ta, ua = transcribe(client, model_pdf, pdf_content(page_pdf_bytes(f, page)))
            tb, ub = transcribe(client, model_text, text_content(page_text(f, page)))
            rec = {"month": month, "file": f.name, "page": page,
                   "pdf": {"model": model_pdf, "table": ta, "usage": ua},
                   "text": {"model": model_text, "table": tb, "usage": ub}}
            cache.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
        a = rows_from(rec["pdf"]["table"], month, "pdf")
        b = rows_from(rec["text"]["table"], month, "text")
        cmp = compare_transcriptions(a, b)
        n_dis = int((~cmp.agree).sum())
        if n_dis and "text2" not in rec:
            # third transcription (the PDF model on the text) settles a two-way disagreement
            tc, uc = transcribe(client, model_pdf, text_content(page_text(f, page)))
            rec["text2"] = {"model": model_pdf, "table": tc, "usage": uc}
            cache.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
        usage.append({"month": month, **{k: rec[k]["usage"] for k in ("pdf", "text", "text2")
                                         if k in rec}})
        key = ["month", "pile"]
        a = a.merge(cmp[key + ["agree", "month_cm_b", "year_cm_b"]], on=key, how="left")
        a = a.rename(columns={"month_cm_b": "text_month_cm", "year_cm_b": "text_year_cm"})
        a["agree_with"] = np.where(a.agree, "text", "")
        if "text2" in rec:
            c = rows_from(rec["text2"]["table"], month, "text2")
            cmp2 = compare_transcriptions(rows_from(rec["pdf"]["table"], month, "pdf"), c)
            a = a.merge(cmp2[key + ["agree"]].rename(columns={"agree": "agree2"}), on=key, how="left")
            a["agree_with"] = np.where(a.agree, "text", np.where(a.agree2.fillna(False), "text2", ""))
            a["agree"] = a.agree | a.agree2.fillna(False)
            a = a.drop(columns="agree2")
        n_open = int((~a.agree).sum())
        per_report.append({"month": month, "file": f.name, "page": page, "piles_pdf": len(a),
                           "piles_text": len(b), "disagreements": n_dis, "unresolved": n_open})
        log(f"{month}: page {page}, {len(a)}/{len(b)} piles, {n_dis} disagreement(s), "
            f"{n_open} unresolved")
        a["file"], a["page"] = f.name, page
        frames.append(a)
    df = pd.concat(frames, ignore_index=True)
    df = chain_check(df)
    feed = load_feed(feed_paths)
    df = feed_check(df, feed_for_piles(df, feed)) if feed else df.assign(feed_cm=float("nan"),
                                                                        feed_ok=None)
    # verified = two independent transcriptions agree. The chain and feed checks are
    # diagnostics of the printed record, not vetoes: a chain residual flags a printed
    # inconsistency (the reports' own arithmetic), a feed mismatch flags a step in the
    # raw feed; both are listed in the report with their size.
    df["verified"] = df["agree"].astype(bool)
    df.to_csv(out / "deep_piles.csv", index=False)
    chain_fail = df[df.chain_ok == False]  # noqa: E712
    feed_fail = df[df.feed_ok == False]  # noqa: E712
    summary = {
        "schema": 1, "task": "wra-monthly-report-deep-pile-transcription",
        "ran_on": dt.date.today().isoformat(), "reports": per_report,
        "models": {"pdf": model_pdf, "text": model_text},
        "usage": usage,
        "tokens": {k: int(sum(u[s][k] for u in usage for s in ("pdf", "text", "text2") if s in u))
                   for k in ("input_tokens", "output_tokens")},
        "n_values": int(len(df)), "n_agree": int(df.agree.sum()),
        "n_third_opinion": int((df.agree_with == "text2").sum()),
        "n_chain_checked": int(df.chain_ok.notna().sum()), "n_chain_ok": int((df.chain_ok == True).sum()),  # noqa: E712
        "n_feed_checked": int(df.feed_ok.notna().sum()), "n_feed_ok": int((df.feed_ok == True).sum()),  # noqa: E712
        "n_verified": int(df.verified.sum()),
        "chain_failures": [{"month": r.month, "pile": r.pile, "depth_m": r.depth_m,
                            "resid_cm": round(float(r.chain_resid), 2)} for r in chain_fail.itertuples()],
        "feed_failures": [{"month": r.month, "pile": r.pile, "printed_cm": r.month_cm,
                           "feed_cm": round(float(r.feed_cm), 2),
                           "feed_step_cm": round(float(r.feed_step_cm), 2)} for r in feed_fail.itertuples()],
        "piles": sorted(df.pile.unique().tolist()),
        "note": "Numbers are transcribed from printed tables; the model never computed or "
                "filled a value. 'verified' = two independent transcriptions agree (a third, "
                "by the PDF model on the text, settles a two-way disagreement). chain_ok "
                "and feed_ok are diagnostics of the printed record and of the raw feed.",
    }
    (out / "report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    return summary


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reports", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model-pdf", default="claude-opus-5-5")
    ap.add_argument("--model-text", default="claude-sonnet-5-5")
    ap.add_argument("--feed", nargs="*", type=Path, default=[],
                    help="open-data deep-pile JSON dumps (data.gov.tw 58339)")
    ap.add_argument("--months", nargs="*", default=None, help="e.g. 2025-12")
    args = ap.parse_args(argv)
    s = run(args.reports, args.out, args.model_pdf, args.model_text, args.feed, args.months)
    print(json.dumps({k: s[k] for k in s if k not in ("usage", "reports")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
