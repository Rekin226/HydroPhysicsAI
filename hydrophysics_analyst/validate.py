"""The answer validator: runs on every answer, presets included, as a regression net.

Rules (spec §8):
1. numbers — only page sentences and page-formatted values may carry digits; frames, labels and
   echo labels may not (exempt: the page's years and starts, and lever percents equal to the
   route's own params);
2. banned strings — ``page_scoping.template_absent`` anywhere; the ``headline_must_not_use`` and
   band keys never as snippets; the verdict lexicon only in non-page spans (page strings use those
   words in negated or scoped form);
3. withheld — no withheld value, and no answer-block sentence governed only by W claims;
4. labels — every internal or labelled value sits beside its rule's page label;
5. Changhua — ``claims.lever_mentions`` is empty on the whole composed text, echo included;
6. citations — exactly the claims of the values shown (and of a refusal's own sentence);
7. dates — a D claim past its scoring date carries the stale-wording banner;
8. reproduction — re-running the logged route gives the same text;
9. charts — every plotted point is one of the answer's own bound values (never withheld, its
   claims cited), titles are page strings, and internals are drawn only in the internals block.
"""
from __future__ import annotations

import datetime as dt
import os
import re
from dataclasses import dataclass

from hydrophysics.twin import claims as C

from . import binding as BD
from . import query as Q
from .page import PageBundle
from .render import Answer, citations_of, echo_text

_DIGIT = re.compile(r"\d+")
VERDICT = {
    "en": ["validated", "accurate", "proven", "guarantee", "will stop", "predict", "recommend",
           "safe", "protect", "better", "worse", "shield", "best choice", "should choose"],
    "zh": ["保護", "安全", "建議", "準確", "保證", "預測", "較好", "較差", "已驗證", "最佳選擇", "應選"],
}
PAGE_KEYS_EXTRA = {"pp", "jan", "dec", "tHead", "town_name", "changhua_label"}


@dataclass(frozen=True)
class Violation:
    rule: str
    span_index: int | None
    detail: str


def _exempt_numbers(b: PageBundle, ans: Answer) -> set[str]:
    c = Q.ctx(b)
    ok = {str(y) for y in c.years} | {str(s) for s in c.starts}
    for x in ans.params.get("cuts", []) or []:
        ok.add(str(x["pct"]))
    return ok


def _snippet_key_ok(b: PageBundle, key: str) -> bool:
    if key in PAGE_KEYS_EXTRA:
        return True
    return all(k in b.T["en"] for k in key.split("+"))


def validate(ans: Answer, b: PageBundle, *, reproduce: bool = True,
             today: dt.date | None = None) -> list[Violation]:
    out: list[Violation] = []
    lang = ans.lang
    scoping = b.ledger["page_scoping"]
    banned_keys = set(scoping.get("headline_must_not_use", [])) | set(
        scoping.get("answer_must_not_use_band_keys", []))
    exempt = _exempt_numbers(b, ans)
    verdict = VERDICT["en"] + VERDICT["zh"]                 # both lists, whatever the answer language
    withheld_words = BD.lexicon()["en"] + BD.lexicon()["zh"]
    st = {c["id"]: c["status_current"] for c in b.ledger["claims"]}
    vals: dict[str, list] = {}
    for r in ans.results:
        for v in r.values:
            vals.setdefault(v.key, []).append(v)
    all_spans = list(ans.echo) + list(ans.spans)
    for i, s in enumerate(all_spans):
        page = s.kind in ("snippet", "value")
        # 1 numbers
        if not page:
            bad = [n for n in _DIGIT.findall(s.text) if n not in exempt]
            if bad:
                out.append(Violation("numbers", i, f"digits {bad} in a {s.kind} span"))
        # 2 banned strings
        for t in scoping.get("template_absent", []):
            if t in s.text:
                out.append(Violation("banned", i, f"template_absent string {t!r}"))
        if s.kind == "snippet":
            if any(k in banned_keys for k in s.key.split("+")):
                out.append(Violation("banned", i, f"key {s.key} must not be used in an answer"))
            if not _snippet_key_ok(b, s.key):
                out.append(Violation("banned", i, f"snippet key {s.key} is not a page string"))
        if not page:
            low = s.text.lower()
            words = verdict + (withheld_words if s.block != "refusal" else [])
            for w in words:
                if w.lower() in low:
                    out.append(Violation("banned", i, f"verdict/withheld word {w!r} in a {s.kind} span"))
        # 3 withheld
        if s.kind == "value":
            if s.cls == "withheld":
                out.append(Violation("withheld", i, f"withheld value {s.key}"))
            if not any(v.text == s.text for v in vals.get(s.key, [])):
                out.append(Violation("numbers", i, f"value span {s.key} does not match its query value"))
        if (s.kind == "snippet" and s.block not in ("refusal", "internals") and s.claims
                and all("W" in C.codes(st[c]) for c in s.claims if c in st)):
                out.append(Violation("withheld", i, f"answer sentence {s.key} governed only by W claims"))
    # 4 labels
    keys_present = [s.key for s in ans.spans if s.kind in ("snippet", "frame")]
    for i, s in enumerate(ans.spans):
        if s.kind == "value" and s.cls in ("internal", "labelled"):
            v = (vals.get(s.key) or [None])[0]
            need = BD.required_labels(v) if v else ()
            if need and not any(any(k == n or k.startswith(n) or n in k.split("+") for k in keys_present)
                                for n in need):
                out.append(Violation("labels", len(ans.echo) + i, f"{s.key} without its label {need}"))
    # 5 Changhua
    text = echo_text(ans.echo, lang) + "\n" + "\n".join(s.text for s in ans.spans) if ans.echo else \
        "\n".join(s.text for s in ans.spans)
    from .jsport import strip_tags
    miss = C.lever_mentions(strip_tags(text), scoping["changhua_lever_rule"], "zh" if lang == "zh" else "en")
    for m in miss:
        out.append(Violation("changhua", None, f"unlabelled Changhua lever mention: {m[:80]}"))
    # 6 citations
    want = citations_of(ans.spans)
    got = tuple(c for c, _ in ans.citations)
    if set(want) != set(got):
        out.append(Violation("citations", None, f"citations {sorted(got)} != claims stated {sorted(want)}"))
    # 7 dates
    today = today or dt.date.fromisoformat(os.environ.get("ANALYST_TODAY", dt.date.today().isoformat()))
    for cid, sc in BD.scoring().items():
        if (cid in got and "D" in C.codes(st.get(cid, ""))
                and today >= dt.date.fromisoformat(sc["not_before"])
                and "banner_stale_wording" not in ans.banners):
                out.append(Violation("dates", None, f"{cid} is past its scoring date without the banner"))
    # 9 charts: every point is a value this answer already states, under the same rules
    from .render import resolve
    cited = set(got)
    for ci, ch in enumerate(ans.charts):
        if ch.title_key and not _snippet_key_ok(b, ch.title_key):
            out.append(Violation("charts", None, f"chart {ci} title {ch.title_key} is not a page string"))
        classes = set()
        for sr in ch.series:
            for ref in sr.refs:
                try:
                    v = resolve(ans, ref)
                except Exception:  # noqa: BLE001
                    out.append(Violation("charts", None, f"chart {ci} point {ref} is not an answer value"))
                    continue
                classes.add(v.answer_class)
                if v.answer_class == "withheld":
                    out.append(Violation("charts", None, f"chart {ci} plots a withheld value {ref}"))
                if not set(v.claims) <= cited:
                    out.append(Violation("charts", None, f"chart {ci} point {ref} cites {v.claims} not in the answer"))
        if "internal" in classes and ch.block != "internals":
            out.append(Violation("charts", None, f"chart {ci} plots model internals outside the internals block"))
    # 8 reproduction
    if reproduce and ans.preset not in ("clarify",):
        from . import presets as PR
        again = PR.run(b, ans.preset, ans.params, lang)
        if again.text() != ans.text():
            out.append(Violation("reproduce", None, "re-running the route gives a different answer"))
    return out
