"""JS-vs-Python parity: the analyst's numbers and sentences must equal the page's.

    python -m hydrophysics_analyst parity --page results/twin/twin_app_20261101a.html \\
        --out results/analyst/parity/

Opens the page in headless Chromium (Playwright; ``TWIN_CHROMIUM`` names a binary if
Playwright's own is absent), and for a grid of policies, starts, years and both languages:

1. calls the page's own ``kpis(policyFields(f, si), y)`` and compares it with
   ``query.kpis`` (numbers within 1e-9);
2. sets the URL hash (with a nonce: an unchanged hash fires no ``hashchange``), waits for the
   re-render (``hashchange`` + two animation frames; ``dataset.ready`` is set only at boot),
   and compares the text of ``#headline``, ``#tW .vn``, ``#tC .vn`` and ``#startNote`` with
   the Python render, and ``#rankTbl`` after ``setTab("rank")``.

Writes ``parity.json`` (cases, mismatches, sha). A bundle gate: zero mismatches.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import jsport as J
from . import query as Q
from .page import PageBundle

KP_KEYS = ("n2Head", "yunHead", "fieldHead", "fieldAvoid", "gwh", "gwhPct", "agree", "nRuns",
           "areaBase", "areaPol", "hsrBase", "hsrPol", "town")
_SET_HASH = """async (h) => {
  await new Promise((resolve, reject) => {
    const t = setTimeout(() => reject(new Error("no hashchange")), 8000);
    addEventListener("hashchange", () => requestAnimationFrame(() => requestAnimationFrame(() => {
      clearTimeout(t); resolve(); })), {once: true});
    location.hash = h; });
}"""
_KPIS = """([f, si, y, keys]) => { S.thr = 1; S.hsrThr = 1000;
  const kp = kpis(policyFields(f, si), y), o = {};
  for (const k of keys) o[k] = kp[k] === undefined ? null : kp[k];
  return JSON.stringify(o); }"""
_DOM = """() => {
  const q = s => { const e = document.querySelector(s); return e ? e.textContent : null; };
  const sn = document.getElementById("startNote");
  return {headline: q("#headline"), tW: q("#tW .vn"), tC: q("#tC .vn"),
          startNote: sn && !sn.hidden ? sn.textContent : ""}; }"""
_EVIDENCE = """() => { renderCard(); const q = id => (document.getElementById(id) || {}).textContent || "";
  return [q("evLine"), q("cardBody"), q("tiles")].join("\\n"); }"""
_RANK = """(by) => { S.rank = by; setTab("rank"); renderRank();
  return document.getElementById("rankTbl").textContent; }"""


@dataclass
class ParityReport:
    page_sha256: str
    n_cases: int = 0
    n_mismatch: int = 0
    mismatches: list = field(default_factory=list)

    def sha(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def grid(b: PageBundle, n_random: int = 30, seed: int = 0) -> list[tuple[tuple[float, ...], int, int]]:
    """Page presets + solved runs (deduped) + seeded 5 %-grid lever vectors; both starts; the
    horizon and two earlier years (one before the 2030 start, for the not-yet-acting branch)."""
    c = Q.ctx(b)
    fs: list[tuple[float, ...]] = []
    presets = [[0.7, 1, 1, 1, 1, 1], [1, 0, 1, 1, 1, 1], [0.7, 0, 1, 1, 1, 1], [0.8] * 6]
    for f in presets + [x["factors"] for x in c.solved] + [[1.0] * c.K]:
        t = tuple(float(v) for v in f)
        if t not in fs:
            fs.append(t)
    rng = random.Random(seed)
    for _ in range(n_random):
        cuts = {k: rng.choice([0, 0, 0, 5, 10, 20, 25, 30, 50, 70, 85, 100]) for k in range(c.K)}
        t = Q.Policy.from_cuts(cuts, c.starts[0], c.K).f
        if t not in fs:
            fs.append(t)
    years = [c.years[-1], c.years[-3], c.years[max(c.YR + 1, len(c.years) - 6)]]
    return [(f, s, y) for f in fs for s in c.starts for y in years]


def _num_eq(a, b) -> bool:
    if a is None or (isinstance(a, float) and math.isnan(a)):
        return b is None or (isinstance(b, float) and math.isnan(b))
    if b is None:
        return False
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(_num_eq(x, y) for x, y in zip(a, b, strict=True))
    return abs(float(a) - float(b)) <= 1e-9 * max(1.0, abs(float(a)))


def _python_dom(b: PageBundle, f, start, year, lang) -> dict:
    r = Q.headline(b, Q.Policy(tuple(f), start), year, lang)
    hl = next(s for s in r.snippets if s.key.startswith("hl_"))
    vals = {v.key: v for v in r.values}
    src = r.src
    tW = "–" if src == "none" else vals["head"].text
    tC = "–" if src == "none" or "gwh" not in vals else vals["gwh"].text
    sn = ""
    c = Q.ctx(b)
    if year == c.years[-1] and src not in ("base", "none"):
        r2 = Q.start_compare(b, f, year, lang)
        sn = next((s.text for s in r2.snippets if s.key.startswith("start_note")), "")
    return {"headline": J.strip_tags(hl.text), "tW": tW, "tC": tC, "startNote": sn}


def _python_rank(b: PageBundle, year: int, by: str, lang: str) -> str:
    r = Q.rank(b, year, by, lang)
    c = Q.ctx(b)
    head, *rows = r.table
    return c.tr(lang, "tranking") + "".join(head) + "".join("".join(row) for row in rows)


def run_parity(page_html: Path, b: PageBundle | None = None, *, n_random: int = 30,
               langs=("en", "zh"), launch: dict | None = None) -> ParityReport:
    from playwright.sync_api import sync_playwright
    b = b or PageBundle.load(Path(page_html), check_forward=False, check_template=False,
                             allow_pending=True)
    rep = ParityReport(page_sha256=b.page_sha256)
    cases = grid(b, n_random)
    c = Q.ctx(b)
    exe = os.environ.get("TWIN_CHROMIUM")
    with sync_playwright() as pw:
        browser = pw.chromium.launch(**(launch or ({"executable_path": exe} if exe else {})))
        try:
            page = browser.new_page()
            page.goto(Path(page_html).resolve().as_uri())
            page.wait_for_function("document.body.dataset.ready === '1'", timeout=60000)
            nonce = 0
            for f, start, year in cases:
                si, yi = c.si(start), c.yidx(year)
                js = json.loads(page.evaluate(_KPIS, [list(f), si, yi, list(KP_KEYS)]))
                py = Q.kpis(b, Q.policy_fields(b, Q.Policy(tuple(f), start)), yi).kp
                rep.n_cases += 1
                for k in KP_KEYS:
                    if not _num_eq(py.get(k), js.get(k)):
                        rep.mismatches.append({"what": "kpis", "key": k, "f": f, "start": start,
                                               "year": year, "py": py.get(k), "js": js.get(k)})
                for lang in langs:
                    nonce += 1
                    page.evaluate(_SET_HASH, Q.deep_link(b, Q.Policy(tuple(f), start), year,
                                                         lang=lang, nonce=nonce))
                    dom = page.evaluate(_DOM)
                    want = _python_dom(b, f, start, year, lang)
                    rep.n_cases += 1
                    for k, v in want.items():
                        if dom.get(k) != v:
                            rep.mismatches.append({"what": "dom", "key": k, "f": f, "start": start,
                                                   "year": year, "lang": lang, "py": v,
                                                   "js": dom.get(k)})
            for lang in langs:
                nonce += 1
                page.evaluate(_SET_HASH, Q.deep_link(b, Q.Policy((1.0,) * c.K, c.starts[0]),
                                                     c.years[-1], lang=lang, nonce=nonce))
                # how_sure: every sentence must be in the page's evidence line, drawer list or the
                # answer tiles (tW_r shows only under a policy, so set the first full-model run)
                if c.solved:
                    nonce += 1
                    x = c.solved[0]
                    page.evaluate(_SET_HASH, Q.deep_link(b, Q.Policy(tuple(float(v) for v in x["factors"]),
                                                                     int(x["start"])), c.years[-1],
                                                         lang=lang, nonce=nonce))
                page_ev = page.evaluate(_EVIDENCE)
                for sn in Q.evidence(b, lang).snippets:
                    rep.n_cases += 1
                    if J.strip_tags(sn.text) not in page_ev:
                        rep.mismatches.append({"what": "evidence", "key": sn.key, "lang": lang,
                                               "py": J.strip_tags(sn.text)[:200]})
                for year in (c.years[-1], c.years[-3]):
                    page.evaluate(f"() => setYear({c.yidx(year)})")
                    for by in ("head", "gwh"):
                        js = page.evaluate(_RANK, by)
                        py = _python_rank(b, year, by, lang)
                        rep.n_cases += 1
                        if js != py:
                            rep.mismatches.append({"what": "rank", "by": by, "year": year,
                                                   "lang": lang, "py": py, "js": js})
        finally:
            browser.close()
    rep.n_mismatch = len(rep.mismatches)
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--page", required=True, type=Path)
    ap.add_argument("--ledger", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("results/analyst/parity"))
    ap.add_argument("--n-random", type=int, default=30)
    ap.add_argument("--allow-pending", action="store_true")
    ap.add_argument("--unbound", action="store_true", help="skip the page/ledger pins (dev only)")
    a = ap.parse_args(argv)
    b = PageBundle.load(a.page, a.ledger, allow_pending=a.allow_pending,
                        check_forward=not a.unbound, check_template=not a.unbound)
    rep = run_parity(a.page, b, n_random=a.n_random)
    a.out.mkdir(parents=True, exist_ok=True)
    out = asdict(rep) | {"sha": rep.sha()}
    (a.out / "parity.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"parity: {rep.n_cases} cases, {rep.n_mismatch} mismatches -> {a.out / 'parity.json'}")
    return 0 if rep.n_mismatch == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
