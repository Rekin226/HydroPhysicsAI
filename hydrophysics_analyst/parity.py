"""JS-vs-Python parity: the analyst's numbers and sentences must equal the page's.

    python -m hydrophysics_analyst parity --page results/twin/twin_app_20261111a.html \\
        --out results/analyst/parity/

Opens the page in headless Chromium (Playwright; ``TWIN_CHROMIUM`` names a binary if
Playwright's own is absent), and for a grid of policies, starts, years and both languages:

1. calls the page's own ``kpis(policyFields(f, si), y)`` and compares it with
   ``query.kpis`` (numbers within 1e-9);
2. sets the URL hash (with a nonce: an unchanged hash fires no ``hashchange``), waits for the
   re-render (``hashchange`` + two animation frames; ``dataset.ready`` is set only at boot),
   and compares the text of ``#headline``, ``#tW .vn``, ``#tC .vn`` and ``#startNote`` with
   the Python render, and ``#rankTbl`` after ``setTab("rank")``;
3. how_sure: every port sentence is on the page, and the trust drawer (``evidenceItems()``) is
   the port's line by line, same text and same claims (``display_map.i18n`` of every ``tr`` key
   a line used), except the band-key lines the ledger keeps out of answers; the head band lines
   (R8.13) are in ``#tsNote``/``#pbWrap``; E24's line (R8.11) equals ``e24Line()``.

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
          tWr: q("#tW .r"), tWd: q("#tW .d"),
          startNote: sn && !sn.hidden ? sn.textContent : ""}; }"""
_EVIDENCE = """() => { renderCard(); const q = id => (document.getElementById(id) || {}).textContent || "";
  return [q("evLine"), q("cardBody"), q("tiles")].join("\\n"); }"""
# evidenceItems() twice: the lines as shown, and the same lines with every tr() key marked
# (\u0001key\u0002), so each line's keys, and so its claims, can be compared with the port's
_EV_ITEMS = """() => { const clean = evidenceItems().map(x => x[1]); const orig = tr;
  tr = function(k, v) { return "\\u0001" + k + "\\u0002" + orig(k, v); };
  let keyed; try { keyed = evidenceItems().map(x => x[1]); } finally { tr = orig; }
  const keys = t => [...t.matchAll(/\\u0001([^\\u0002]*)\\u0002/g)].map(m => m[1]);
  return clean.map((t, i) => [t, keys(keyed[i])]); }"""
_KEYED_TR = 'tr = function(k, v) { return "\\u0001" + k + "\\u0002" + orig(k, v) + "\\u0003"; };'
_TOP = """() => { const orig = tr; let out; """ + _KEYED_TR + """
  try { renderTiles(); const r = document.querySelector("#tW .r");
    out = {ev: evLineText(), tile: r ? r.textContent : ""}; }
  finally { tr = orig; renderTiles(); }
  return out; }"""
_BAND = """() => { const orig = tr; let out; """ + _KEYED_TR + """
  try { const s = seriesFor(S.sel, P);
    const cov = S.ts === "head" && (s.bandKind === "runs_head" || s.pb) ? covLine() : "";
    renderPB(); const el = document.getElementById("pbWrap");
    const pb = el && !el.hidden ? [...el.querySelectorAll("h3, p, text")].map(e => e.textContent)
      : [];
    out = {note: s.bandNote, cov, pb}; }
  finally { tr = orig; renderPB(); }
  return out; }"""
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
    presets = [[0.7] * 6, [0.7, 1, 1, 1, 1, 1], [1, 0, 1, 1, 1, 1], [0.7, 0, 1, 1, 1, 1], [0.8] * 6]
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
    _, tWr, tWd = Q.tile_w(b, Q.Policy(tuple(f), start), year, lang)
    return {"headline": J.strip_tags(hl.text), "tW": tW, "tC": tC, "tWr": tWr, "tWd": tWd,
            "startNote": sn}


def _python_rank(b: PageBundle, year: int, by: str, lang: str) -> str:
    r = Q.rank(b, year, by, lang)
    c = Q.ctx(b)
    head, *rows = r.table
    return c.tr(lang, "tranking") + "".join(head) + "".join("".join(row) for row in rows)


NOT_DRAWER = ("ev_nc_pass", "ev_nc_fail", "ev_tp_pass", "ev_tp_fail", "tW_w30", "tW_r",
              "tW_r_w30")                          # evLineText and the head tile's reading


def parse_keyed(t: str) -> tuple[str, list[tuple[str, str, list[str]]], list[str]]:
    """A string rendered with ``tr`` marking each call ``\\x01key\\x02 ... \\x03``: (the clean
    text, its top-level calls as (key, clean text, every key used inside), every key used)."""
    clean: list[str] = []
    stack: list[list] = []
    tops, every = [], []
    i = 0
    while i < len(t):
        ch = t[i]
        if ch == "\x01":
            j = t.index("\x02", i)
            k = t[i + 1:j]
            stack.append([k, len(clean), [k]])
            every.append(k)
            i = j + 1
            continue
        if ch == "\x03":
            k, start, keys = stack.pop()
            if stack:
                stack[-1][2].extend(keys)
            else:
                tops.append((k, "".join(clean[start:]), keys))
        else:
            clean.append(ch)
        i += 1
    return "".join(clean), tops, every


def _both_ways(b: PageBundle, mine, items: list, lang: str, what: str) -> list[dict]:
    """Page lines ``items`` [(text, keys)] against the port's snippets, as multisets: each page
    line is one of the port's, with the same text and the same claims (``display_map.i18n`` of
    every key the page used), and the port has no other line."""
    from . import binding as BD
    dm = b.ledger["display_map"]["i18n"]
    pool = [(J.strip_tags(sn.text), set(BD.snippet_claims(sn.key, b.ledger)), sn.key)
            for sn in mine]
    out = []
    for text, keys in items:
        t = J.strip_tags(text)
        want = {c for k in keys for c in dm.get(k, [])}
        hit = next((x for x in pool if x[0] == t), None)
        if hit is None:
            out.append({"what": f"{what}_missing", "lang": lang, "js": t[:200]})
            continue
        pool.remove(hit)
        if hit[1] != want:
            out.append({"what": f"{what}_claims", "lang": lang, "key": hit[2], "js": t[:120],
                        "py_claims": sorted(hit[1]), "js_claims": sorted(want)})
    out += [{"what": f"{what}_extra", "lang": lang, "key": k, "py": t[:200]} for t, _, k in pool]
    return out


def _evidence_items(b: PageBundle, ev, items: list, lang: str) -> list[dict]:
    """The trust drawer line by line, both ways, text and claims (``ev`` is the full mirror,
    ``query.evidence(full=True)``; its evidence-line and tile-reading lines are checked apart)."""
    return _both_ways(b, [s for s in ev if s.key.split("+")[0] not in NOT_DRAWER], items, lang,
                      "evidence")


def _top_items(b: PageBundle, ev, keyed: dict, lang: str) -> list[dict]:
    """``evLineText`` and the head tile's reading (``#tW .r``), both ways, text and claims."""
    items = []
    for t in (keyed.get("ev"), keyed.get("tile")):
        if t:
            text, _, every = parse_keyed(t)
            items.append((text, every))
    return _both_ways(b, [s for s in ev if s.key.split("+")[0] in NOT_DRAWER], items, lang, "top")


def _band_items(b: PageBundle, band, keyed: dict, lang: str) -> list[dict]:
    """The head series' band at the default view, both ways, text and claims: the band note's
    top-level strings, the coverage line and every title, note and label of the PB band."""
    items = []
    _, tops, _ = parse_keyed(keyed.get("note") or "")
    items += [(t, keys) for _, t, keys in tops]
    for t in [keyed.get("cov") or ""] + list(keyed.get("pb") or []):
        if t:
            text, _, every = parse_keyed(t)
            items.append((text, every))
    return _both_ways(b, band, items, lang, "head_band")


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
                # how_sure, at the page's default view (no policy, the horizon): every port
                # sentence is on the page; the evidence line, the tile reading, the drawer and the
                # head band are the port's line by line, both ways, text and claims
                page_ev = page.evaluate(_EVIDENCE)
                ev = Q.evidence(b, lang, full=True).snippets
                for sn in ev:
                    rep.n_cases += 1
                    if J.strip_tags(sn.text) not in page_ev:
                        rep.mismatches.append({"what": "evidence", "key": sn.key, "lang": lang,
                                               "py": J.strip_tags(sn.text)[:200]})
                rep.n_cases += 3
                rep.mismatches += _evidence_items(b, ev, page.evaluate(_EV_ITEMS), lang)
                rep.mismatches += _top_items(b, ev, page.evaluate(_TOP), lang)
                rep.mismatches += _band_items(b, Q.head_band(b, lang, full=True).snippets,
                                              page.evaluate(_BAND), lang)
                # E24 (R8.11): tile tA's line under Model internals, its one policy and another
                for pol in (Q.Policy(Q.IRR30[:c.K], 2026), Q.Policy((0.7,) * c.K, c.starts[0])):
                    nonce += 1
                    page.evaluate(_SET_HASH, Q.deep_link(b, pol, c.years[-1], lang=lang,
                                                         nonce=nonce, mi=True))
                    js = page.evaluate("() => typeof e24Line === 'function' ? e24Line() : ''")
                    py = "".join(s.text for s in Q.e24_line(b, pol, lang).snippets)
                    rep.n_cases += 1
                    if js != py:
                        rep.mismatches.append({"what": "e24", "f": pol.f, "start": pol.start,
                                               "lang": lang, "py": py[:200], "js": js[:200]})
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
