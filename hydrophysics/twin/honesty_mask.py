"""The grader-view mask of user ruling R11.4: claim-honesty figures and 6-part figures hidden.

The rule itself is ``RULE_TEXT`` (a module constant, so ``rules()`` and its hash do not depend
on docstrings, which ``python -OO`` strips). It is applied the same way to the grader view of
the inventory (``honesty.grader_view``) and to the text files of a grader workspace
(``dataprep_staging/honesty/make_grader_workspace.py``). ``derivable_pairs`` is the
derivation gate the workspace builder runs after masking; ``mask_hashes`` hides the published
hash prefixes of files the mask changed.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import re
from pathlib import Path

RULE_TEXT = """\
R11.4 (user rulings 2026-11-08, twenty-fourth set) asks for a committed, mechanical masking rule
for the claim-honesty re-audit: the graders must not see the current claim-honesty input, the
per-source audit figures, or a 6-part figure beside a science-trust figure (honesty =
6 x 6-part - 5 x science). Round 9's mask was ad hoc. The rule is defined by patterns with
context, never by a list of statement ids or of known figures. Every example below uses values
no real figure has (71.1, 22.2, 33.3 and the like).

UNIT. A text is masked one unit at a time; terms in one unit never reach a figure in another.
A unit is a block of BLOCK (claims.sentences's non-period boundaries: a blank line, a new list
item, a numbered item or a table row start a new block). An inventory statement is masked with
its block of the pinned file as context (doc statements) or alone (page strings, ledger
paragraphs). A block with no letter (a figure alone, say under a heading) is masked with the
block before it as context. In HTML each string literal and each text between two tags
(outside scripts) is a unit, a JSON script payload is masked as JSON, and script code outside
them is left; log files use one line per unit; data tables (CSV), binary files and code files
(step 5) are not masked. In JSON every string value is a text of its own, and a number is
masked when the nearest key on its path that names a term names an honesty or a 6-part term
("claim_honesty": 71.1; "six_part": {"official": 22.2}).

FIGURE (FIGURE): a number with an optional sign, an optional range end (71.1-71.4, 61-62,
-0.2 to +0.1), an optional "/100" and an optional %. Not a figure: part of an identifier, path,
version, hash, date or time (a letter, digit, _, ., /, #, : or - right before it; a letter,
digit, _ or - plus a letter or digit right after it), a fraction (5/6), a year, year range or
year-month (an integer 1900-2100, alone or with a year or two-digit end), a number named by a
counter word right before it (section 27, step 0, round 9; COUNTER_WORDS), a count (a number
followed by a count noun or a date unit: 66 findings, 1,649 statements, 95 年; COUNT_NOUNS), a
level (a percentage followed by range, interval, band or CI: 80 % range), a physical quantity
(a number without % followed by a unit: 71.1 cm/yr; UNIT_WORDS), or a band (an integer range
at least 5 wide: 90-99).

STATEMENT IDS. An inventory doc id (page, readme, model_card, evidence or ledger, a colon and
ten hex digits; STATEMENT_ID) hashes the unmasked text of its statement, so a quoted id would
let a masked figure be found by trying values: every such id in a text is replaced by
PLACEHOLDERS["id"].

TERMS (EN and ZH, case-insensitive; TERMS):
* H, honesty: honesty (claim honesty, claim-honesty, claim_honesty), audit or audited figure,
  re-audit, grader(s), an audit, auditor or grader (auditor-1...) that gave or scored; 誠實,
  評分者, 複核.
* R, audit source or abbreviation: page, README, MODEL_CARD (model card), evidence, ledger, CH
  (upper case, a word; in this project also Changhua): owns a figure in SOURCE_RANGE [90, 100]
  right after it (SOURCE_GAP: at most 4 spaces, colons, =, * or opening parentheses between;
  a per-source audit figure, or "CH 9x.x"), and nothing else.
* S, 6-part: 6-part, six-part, six_part, 6 part; 六部分, 六項.
* K: a quoted key "six": (counts for the unit rule only).
* A, headline: headline (not the headline's interface parts: headline lever, head lever, row,
  tile(s), function, sentence, text, key, string, wording, "headline and tiles", "headline's
  Yunlin/Changhua"), official figure(s), progress as a word (not the progress module or
  progress_map); 總分, 頭條 (not 頭條槓桿), 進度.
* P, aggregate: overall, total, composite, combined (figure), mean of the six; 整體.
* O, other quantity: science, science trust, a bare part(s), coverage, and the part names head
  response, forecast start, uncertainty band(s), subsidence, Changhua; 科學, 涵蓋.
* Q, statistic: kappa, agreement; 一致.

A 6-part-sized figure is signed, or a decimal in SIX_RANGE [40, 80], or a decimal right after a
change word (rises, falls, by...; CHANGE_WORDS).

DECISION, per figure, left to right:
1. Adjacent owner. The closest term right beside the figure owns it: a term that follows it
   within TRAILING_GAP (at most 12 characters, no digit, none of ;,.|()[]{}, no CJK stop or
   comma, and no "and", "or", "with", "vs", "but", "while"; TRAILING_BREAK) or that precedes
   it within LEADING_GAP (at most 16 characters, no digit, none of ;,.| and no CJK stop or
   comma). An H owner masks the figure when it is signed (+, -, −; not ±) or lies in
   HONESTY_RANGE [50, 100); an S owner when it is signed, has a decimal point or is at least
   SMALL_INTEGER (10); an A owner when it is 6-part-sized; a P owner when it is a decimal in
   [40, 80]; an R owner when it lies in [90, 100]; an O or Q owner keeps it (science 22.2
   stays).
2. Chain. A figure joined to the figure before it only by a connector (CHAIN: ->, →, =>, to,
   /, a comma, and, or, vs) takes that figure's decision when it has no adjacent owner of its
   own (22.2 -> 22.4 stays with science; 71.1 -> 71.4 is masked with the 6-part figure).
   Parallel list: a chain of k figures none of which has an adjacent owner, after at least k
   owned figures in the unit whose last k owners mix a keeping (O, Q) and a masking (H, S, A,
   P) kind, takes the owners of those k figures in order ("science
   (22.2) and 6-part (71.1) ... (22.4, 71.4)": 22.4 stays, 71.4 is masked). A misreading of
   the order is caught by the derivation gate.
3. Scope. The nearest S, A, P, O or Q term before the figure is an S term and the figure is
   signed, at least 10 or a decimal after a change word; or it is an A term at most REACH (80)
   characters before the figure and the figure is 6-part-sized (P terms own figures only
   right beside them, step 1).
   Honesty scope: the nearest H or O term before it is an H term and the figure is signed or
   lies in [50, 100).
4. Unit. The unit holds an H, S or K term and the figure lies in [50, 100), or it holds an A
   term and the figure is a decimal in [40, 80] whose nearest S, A, P, O or Q term before it
   is not an O term: it is masked unless an O or Q term owns it (step 1) or its chain. A list
   such as "from 71.1, 22.2 and 33.3" names its 6-part figure far from the term; science
   trust lies below 50.
5. Code is not masked: Python, shell and JS files are left as they are (registrations hash
   them) and only gated (GATE).

The masked figure (sign, range and % included) is replaced by PLACEHOLDERS["honesty"] when an H
term owns it (adjacent, else the nearest H, S, A or P term before it is H), else by
PLACEHOLDERS["six_part"]. Same text, same output.

HASHES. A file the mask changed no longer has the hash a record may print for it, and a printed
hash prefix of a short file would let the masked figure be found by trying values. In a grader
workspace every hexadecimal run of 7 or more characters that starts the sha256 or sha1 of the
unmasked content of a masked file is replaced by PLACEHOLDERS["hash"] (mask_hashes).

GATE (derivable_pairs). After masking, no unit may hold, within 300 characters of an H, S or A
term, two unmasked figures a and b (at most two decimals, not both integers, not in a data
series of 5 or more numbers joined only by commas, brackets or space; a not kept for an O or Q
term, so not a science or part figure) with a in [40, 80], b in
[30, 75], |a - b| <= 15 and 6a - 5b in [90, 100]: the workspace builder stops on any such
pair. It also stops on any statement id left (STATEMENT_ID). Without the term condition, pairs
within 300 characters of each other with 6a - 5b in [94.6, 99.5] are listed for the operator
to review (free_pairs); they do not stop the build.
"""

RULING = "R11.4 (results/dataprep/rulings/user_rulings_2026-11-08f.json)"
PLACEHOLDERS = {"honesty": "[masked: claim-honesty figure]",
                "six_part": "[masked: 6-part figure]",
                "hash": "[masked: hash of a masked file]",
                "id": "[masked: statement id]"}
HONESTY_RANGE = (50.0, 100.0)
YEARS = (1900, 2100)
SMALL_INTEGER = 10
SIX_RANGE = (40.0, 80.0)
SOURCE_RANGE = (90.0, 100.0)
REACH = 80
BAND_WIDTH = 5
GAP_MAX = (12, 16)
GATE = {"a": (40.0, 80.0), "b": (30.0, 75.0), "max_diff": 15.0, "honesty": (90.0, 100.0),
        "window": 300, "series": 5, "free_honesty": (94.6, 99.5)}

_NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_SIGN = r"\+/-|\+-|±|[+\-−]"
FIGURE = (r"(?<![A-Za-z0-9_./#:\-−])"
          rf"(?P<sign>{_SIGN})?(?P<num>{_NUM})"
          rf"(?:\s*(?:-|–|to)\s*(?P<sign2>{_SIGN})?(?P<num2>{_NUM}))?"
          r"(?P<over>/100)?(?P<pct>\s?%)?"
          r"(?![A-Za-z0-9_]|/\d|:\d|\.\d|-[A-Za-z0-9])")
COUNTER_WORDS = ("section", "sections", "step", "steps", "round", "rounds", "stage", "line",
                 "lines", "caveat", "caveats", "set", "sets", "rule", "ruling", "rulings",
                 "row", "rows", "tier", "phase", "batch", "version", "PR", "§", "第")
COUNT_NOUNS = ("findings?", "statements?", "wells?", "runs?", "rows?", "claims?", "keys?",
               "records?", "files?", "members?", "years?", "months?", "well-months?",
               "days?", "items?", "stations?", "townships?", "hashes", "inputs?", "pages?",
               "sections?", "graders?", "sentences?", "paragraphs?", "口", "個", "筆", "項",
               "cells?", "年", "月", "日")
LEVEL_WORDS = ("range", "interval", "band", "CI", "confidence")
UNIT_WORDS = ("cm", "mm", "m", "km", "Mm3", "m3", "kWh", "GWh", "MW", "kg", "yr", "°C",
              "公尺", "公分", "公釐", "萬噸")
STATEMENT_ID = (r"(?<![A-Za-z0-9_])(?:page|readme|model_card|evidence|ledger):[0-9a-f]{10}"
                r"(?:-\d+)?(?![0-9A-Za-z])")
CHANGE_WORDS = ("rises", "rose", "rise", "falls", "fell", "fall", "drops", "dropped", "moves",
                "moved", "changes", "changed", "gains", "gained", "loses", "lost", "by", "up",
                "down")
TERMS = {
    "H": [r"(?<![A-Za-z])honesty", r"audit(?:ed)?\s+figures?", r"re-?audit(?:ed|s|ing)?",
          r"(?<![A-Za-z])(?:audit(?:or)?s?|graders?)(?:-\d+)?\s+(?:gave|gives|scored|scores)"
          r"(?![A-Za-z])",
          r"(?<![A-Za-z])graders?(?![A-Za-z])", r"誠實", r"評分者", r"複核"],
    "R": [r"(?<![A-Za-z_])(?:page|readme|model[_ ]card|evidence|ledger)(?![A-Za-z_])",
          r"(?<![A-Za-z0-9_-])(?-i:CH)(?![A-Za-z0-9_-])"],
    "S": [r"(?<![A-Za-z0-9])(?:6|six)[-_ ]parts?(?![A-Za-z])", r"六部分", r"六項"],
    "K": [r"(?<=[\"'])six(?=[\"']\s*:)"],
    "A": [r"(?<![A-Za-z0-9_])headline(?![A-Za-z0-9_])(?!'s\s+(?:Yunlin|Changhua)|\s+(?:head\s+)?"
          r"(?:levers?|rows?|tiles?|functions?|sentences?|text|keys?|strings?|wording)"
          r"|\s+and\s+(?:the\s+)?tiles?)",
          r"(?<![A-Za-z])official\s+figures?(?![A-Za-z])",
          r"(?<![A-Za-z0-9_./-])progress(?![A-Za-z0-9_]|\.[A-Za-z]|-\w)",
          r"總分", r"頭條(?!槓桿)", r"進度"],
    "P": [r"(?<![A-Za-z0-9_])(?:overall|total)(?![A-Za-z0-9_])",
          r"(?<![A-Za-z0-9_])composite(?![A-Za-z0-9_])",
          r"(?<![A-Za-z])combined(?:\s+figures?)?(?![A-Za-z])",
          r"(?<![A-Za-z])mean\s+of\s+(?:the\s+)?six(?![A-Za-z])", r"整體"],
    "O": [r"(?<![A-Za-z])science(?:[-_ ]trust)?(?![A-Za-z])",
          r"(?<![-_A-Za-z0-9])parts?(?![-A-Za-z0-9])", r"(?<![A-Za-z])coverage(?![A-Za-z])",
          r"(?<![A-Za-z])head\s+response(?![A-Za-z])", r"(?<![A-Za-z])forecast\s+start",
          r"(?<![A-Za-z])uncertainty\s+bands?", r"(?<![A-Za-z])subsidence(?![A-Za-z])",
          r"(?<![A-Za-z])Changhua(?![A-Za-z])", r"科學", r"涵蓋"],
    "Q": [r"(?<![A-Za-z])kappa(?![A-Za-z])", r"(?<![A-Za-z])agreement(?![A-Za-z])", r"一致"],
}
TRAILING_GAP = r"[^\d;,.|()\[\]{}。；，、]{0,12}"
LEADING_GAP = r"[^\d;,.|。；，、]{0,16}"
TRAILING_BREAK = r"(?<![A-Za-z])(?:and|or|with|vs|but|while)(?![A-Za-z])|和|與|及"
CHAIN = r"[\s*%]*(?:->|→|=>|to|/|,|and|or|vs\.?)?[\s*%]*"
SOURCE_GAP = r"[\s:=*(]{0,4}"
BLOCK = r"\n\s*\n|\n(?=\s*(?:[-*|]|\d+\.)\s)"
HASH_RUN = r"(?<![0-9A-Fa-f])[0-9A-Fa-f]{7,64}(?![0-9A-Fa-f])"
LINE_SUFFIXES = (".log",)
MARKUP_SUFFIXES = (".html", ".htm")
CODE_SUFFIXES = (".py", ".sh", ".js", ".mjs", ".ipynb")
TEXT_SUFFIXES = (".md", ".txt", ".rst", ".tex", ".yaml", ".yml", ".toml", ".cfg",
                 ".ini") + LINE_SUFFIXES + MARKUP_SUFFIXES
MARKUP_UNIT = (r'"(?:[^"\\\n]|\\.)*"|\'(?:[^\'\\\n]|\\.)*\'|`(?:[^`\\]|\\.)*`'
               r"|(?<=>)[^<>]+(?=<)")
SCRIPT = r"(<script[^>]*>)(.*?)(</script>)"
JSON_SUFFIXES = (".json", ".jsonl")

_FIG = re.compile(FIGURE)
_TERM = re.compile("|".join(f"(?P<{k}{i}>{p})" for k, ps in TERMS.items()
                            for i, p in enumerate(ps)), re.I)
_TRAIL = re.compile(TRAILING_GAP)
_LEAD = re.compile(LEADING_GAP)
_BREAK = re.compile(TRAILING_BREAK, re.I)
_CHAIN = re.compile(CHAIN, re.I)
_BLOCK = re.compile(BLOCK)
_HASH = re.compile(HASH_RUN)
_SERIES = re.compile(r"[\s,;\[\]]*")
_SOURCE_GAP = re.compile(SOURCE_GAP)
_MARKUP = re.compile(MARKUP_UNIT)
_SCRIPT = re.compile(SCRIPT, re.S | re.I)
_ANCHORS = ("honest", "audit", "grader", "part", "six", "headline", "progress", "official",
            "overall", "total", "composite", "combined", "mean", "science", "coverage", "kappa",
            "agreement", "head", "forecast", "uncertainty", "subsidence", "changhua", "ch",
            "page", "readme", "model", "evidence", "ledger", "誠實", "評分者", "複核", "六部分",
            "六項", "總分", "頭條", "進度", "整體", "科學", "涵蓋", "一致")
_CAND = re.compile("|".join(re.escape(w) for w in _ANCHORS))
_COUNTER = re.compile(r"(?:(?<![A-Za-z])(?:" + "|".join(
    re.escape(w) for w in COUNTER_WORDS if w.isascii()) + r")|§|第)\s*$", re.I)
_COUNT = re.compile(r"\s*(?:" + "|".join(COUNT_NOUNS) + r")(?![A-Za-z])", re.I)
_LEVEL = re.compile(r"\s*(?:" + "|".join(LEVEL_WORDS) + r")(?![A-Za-z])", re.I)
_UNIT = re.compile(r"\s*(?:" + "|".join(re.escape(u) for u in UNIT_WORDS)
                   + r")(?![A-Za-z0-9])")
_ID = re.compile(STATEMENT_ID)
_LETTER = re.compile(r"[A-Za-z\u4e00-\u9fff]")
_CHANGE = re.compile(r"(?<![A-Za-z])(?:" + "|".join(CHANGE_WORDS) + r")\s*(?:about\s*)?$",
                     re.I)


def rules() -> dict:
    """The mask rule as data; ``rules_sha256`` is its hash (canonical JSON)."""
    return {
        "ruling": RULING, "rule": RULE_TEXT, "placeholders": PLACEHOLDERS,
        "honesty_range": list(HONESTY_RANGE), "small_integer": SMALL_INTEGER,
        "six_range": list(SIX_RANGE), "source_range": list(SOURCE_RANGE), "reach": REACH,
        "band_width": BAND_WIDTH, "gap_max": list(GAP_MAX), "gate": GATE,
        "years_excluded": list(YEARS), "figure": FIGURE, "counter_words": list(COUNTER_WORDS),
        "count_nouns": list(COUNT_NOUNS), "level_words": list(LEVEL_WORDS),
        "unit_words": list(UNIT_WORDS), "statement_id": STATEMENT_ID,
        "change_words": list(CHANGE_WORDS), "terms": TERMS, "trailing_gap": TRAILING_GAP,
        "leading_gap": LEADING_GAP, "trailing_break": TRAILING_BREAK, "chain": CHAIN,
        "source_gap": SOURCE_GAP,
        "block": BLOCK, "hash_run": HASH_RUN,
        "line_suffixes": list(LINE_SUFFIXES), "markup_suffixes": list(MARKUP_SUFFIXES),
        "code_suffixes": list(CODE_SUFFIXES),
        "markup_unit": MARKUP_UNIT, "script": SCRIPT, "text_suffixes": list(TEXT_SUFFIXES),
        "json_suffixes": list(JSON_SUFFIXES),
    }


def rules_sha256() -> str:
    return hashlib.sha256(json.dumps(rules(), sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def _terms(text: str) -> list[tuple[int, int, str]]:
    """(start, end, kind) of every term of ``text``, left to right, not overlapping. The term
    pattern is looked for only around an anchor (``_ANCHORS``; every term holds one), which
    gives the same terms as a scan of the whole text, much faster on long data lines."""
    low = text.lower()
    if len(low) != len(text):                                 # case folding moved offsets
        return [(m.start(), m.end(), m.lastgroup[0]) for m in _TERM.finditer(text)]
    found: dict[int, tuple[int, int, str]] = {}
    for c in _CAND.finditer(low):
        m = _TERM.search(text, max(0, c.start() - 4), min(len(text), c.end() + 24))
        while m is not None and m.start() <= c.start():
            if m.end() >= c.end():
                found.setdefault(m.start(), (m.start(), m.end(), m.lastgroup[0]))
                break
            m = _TERM.search(text, m.end(), min(len(text), c.end() + 24))
    out: list[tuple[int, int, str]] = []
    for t in sorted(found.values()):
        if not out or t[0] >= out[-1][1]:
            out.append(t)
    return out


def _value(s: str) -> float:
    return float(s.replace(",", ""))


def _year(s: str | None) -> bool:
    return s is not None and s.isdigit() and YEARS[0] <= int(s) <= YEARS[1]


def _is_figure(m: re.Match, text: str) -> bool:
    num, num2 = m.group("num"), m.group("num2")
    if _year(num) and not m.group("sign") and not m.group("pct") and (
            num2 is None or _year(num2) or (len(num2) == 2 and num2.isdigit())):
        return False                          # a year, a year range or a year-month
    if num2 and "." not in num + num2 and abs(_value(num2) - _value(num)) >= BAND_WIDTH:
        return False                          # a band
    if _COUNT.match(text, m.end()) or (m.group("pct") and _LEVEL.match(text, m.end())) or \
            (not m.group("pct") and not m.group("over") and _UNIT.match(text, m.end())):
        return False                          # a count, a level or a physical quantity
    return not _COUNTER.search(text[max(0, m.start() - 12):m.start()])


class _Terms:
    """The terms of a unit, indexed for the questions of ``_decide``."""

    def __init__(self, text: str):
        self.t = _terms(text)
        self.starts = [t[0] for t in self.t]
        self.ends = [t[1] for t in self.t]
        self.unit = any(t[2] in "HSK" for t in self.t)
        self.unit_a = any(t[2] == "A" for t in self.t)
        self.last: dict[str, list[int]] = {}
        for kinds in ("SAPOQ", "HO", "HSAP"):
            cur, arr = -1, [-1]
            for i, t in enumerate(self.t):
                cur = i if t[2] in kinds else cur
                arr.append(cur)
            self.last[kinds] = arr          # last[k][n]: last index < n of a kind in k, or -1


def _adjacent(m: re.Match, text: str, terms: _Terms) -> str | None:
    """The kind of the closest term right beside the figure (DECISION step 1), or None."""
    a, b = m.start(), m.end()
    t, best = terms.t, (None, 10**9)
    nb = bisect.bisect_right(terms.ends, a)
    if nb and t[nb - 1][2] != "K" and a - t[nb - 1][1] <= GAP_MAX[1] and \
            _LEAD.fullmatch(text[t[nb - 1][1]:a]) and (
                t[nb - 1][2] != "R" or _SOURCE_GAP.fullmatch(text[t[nb - 1][1]:a])):
        best = (t[nb - 1][2], a - t[nb - 1][1])
    na = bisect.bisect_left(terms.starts, b)
    if na < len(t) and t[na][2] not in "KR" and t[na][0] - b <= GAP_MAX[0] and \
            _TRAIL.fullmatch(text[b:t[na][0]]) and not _BREAK.search(text[b:t[na][0]]) and \
            t[na][0] - b < best[1]:
        best = (t[na][2], t[na][0] - b)
    return best[0]


def _by_owner(m: re.Match, kind: str) -> str | None:
    """The decision an owning term of ``kind`` gives the figure (DECISION step 1)."""
    vals = [_value(m.group("num"))] + ([_value(m.group("num2"))] if m.group("num2") else [])
    signed = any(s in ("+", "-", "−") for s in (m.group("sign"), m.group("sign2")))
    decimal = "." in m.group("num") + (m.group("num2") or "")
    if kind == "H":
        return "honesty" if signed or any(HONESTY_RANGE[0] <= v < HONESTY_RANGE[1]
                                          for v in vals) else None
    if kind == "R":
        return "honesty" if any(SOURCE_RANGE[0] <= v <= SOURCE_RANGE[1] for v in vals) else None
    if kind == "S":
        return "six_part" if signed or decimal or any(v >= SMALL_INTEGER for v in vals) else None
    if kind in ("A", "P"):
        level = decimal and any(SIX_RANGE[0] <= v <= SIX_RANGE[1] for v in vals)
        return "six_part" if level or (kind == "A" and signed) else None
    return "keep" if kind in ("O", "Q") else None


def _decide(m: re.Match, text: str, terms: _Terms, prev: str | None,
            kind: str | None) -> str | None:
    """'honesty', 'six_part', 'keep' (owned by an O or Q term) or None for the figure ``m``
    with adjacent owner ``kind``, given the decision of the figure before it when the two are
    chained (DECISION)."""
    a = m.start()
    vals = [_value(m.group("num"))] + ([_value(m.group("num2"))] if m.group("num2") else [])
    signed = any(s in ("+", "-", "−") for s in (m.group("sign"), m.group("sign2")))
    decimal = "." in m.group("num") + (m.group("num2") or "")
    in_range = any(HONESTY_RANGE[0] <= v < HONESTY_RANGE[1] for v in vals)
    in_six = any(SIX_RANGE[0] <= v <= SIX_RANGE[1] for v in vals)
    change = decimal and _CHANGE.search(text[max(0, a - 20):a]) is not None
    if kind == "A" and change:
        return "six_part"
    if kind is not None and kind != "R":
        return _by_owner(m, kind)
    if kind == "R" and _by_owner(m, "R"):
        return "honesty"
    if prev is not None:
        return prev
    t = terms.t
    nb = bisect.bisect_right(terms.ends, a)
    i = terms.last["HSAP"][nb]
    label = "honesty" if i >= 0 and t[i][2] == "H" else "six_part"
    i = terms.last["SAPOQ"][nb]
    k = t[i][2] if i >= 0 else None
    level = decimal and in_six
    six = (k == "S" and (signed or any(v >= SMALL_INTEGER for v in vals) or change)) or (
        k == "A" and a - t[i][1] <= REACH and (level or signed or change))
    i = terms.last["HO"][nb]
    hon = i >= 0 and t[i][2] == "H" and (signed or in_range)
    if six and hon:
        return label
    if six or hon:
        return "six_part" if six else "honesty"
    if terms.unit and in_range:
        return label
    return "six_part" if terms.unit_a and level and k != "O" else None


_OWNERS = "HSAPOQR"


def _decisions(full: str, figs: list) -> list:
    """The decision for every figure of a unit, left to right (DECISION steps 1-4)."""
    terms = _Terms(full)
    adj = [_adjacent(m, full, terms) for m in figs]
    out: list = [None] * len(figs)
    groups: list[list[int]] = []
    for i in range(len(figs)):
        if i and _CHAIN.fullmatch(full[figs[i - 1].end():figs[i].start()]):
            groups[-1].append(i)
        else:
            groups.append([i])
    owned: list[str] = []                     # adjacent owners met so far, in order
    for g in groups:
        last = owned[-len(g):] if len(g) >= 2 and len(owned) >= len(g) else []
        if last and all(adj[i] is None for i in g) and set(last) & set("OQ") and \
                set(last) & set("HSAP"):
            for i, kind in zip(g, last, strict=True):                 # a parallel list
                out[i] = _by_owner(figs[i], kind)
            continue
        prev = None
        for i in g:
            out[i] = _decide(figs[i], full, terms, prev, adj[i])
            if adj[i] in tuple(_OWNERS) and adj[i] != "R":
                owned.append(adj[i])
            prev = out[i]
    return out


def mask_text(text: str, before: str = "", after: str = "") -> tuple[str, int]:
    """(``text`` masked, number of figures and statement ids masked); ``before`` and ``after``
    are the rest of its unit (terms only: their own figures are not touched)."""
    if not text:
        return text, 0
    out_text, n = text, 0
    full = before + text + after
    low = full.lower()
    if any(w in low for w in _ANCHORS):
        lo, hi = len(before), len(before) + len(text)
        figs = [m for m in _FIG.finditer(full) if _is_figure(m, full)]
        if figs:
            dec = _decisions(full, figs)
            out, pos = [], lo
            for m, k in zip(figs, dec, strict=True):
                if k in (None, "keep") or m.start() < lo or m.end() > hi:
                    continue
                out += [full[pos:m.start()], PLACEHOLDERS[k]]
                pos, n = m.end(), n + 1
            out.append(full[pos:hi])
            out_text = "".join(out)
    out_text, k = _ID.subn(PLACEHOLDERS["id"], out_text)
    return out_text, n + k


def blocks(text: str) -> list[tuple[int, int]]:
    """(start, end) of each unit block of ``text`` (``BLOCK``)."""
    out, pos = [], 0
    for m in _BLOCK.finditer(text):
        out.append((pos, m.start()))
        pos = m.end()
    out.append((pos, len(text)))
    return out


def mask_blocks(text: str, lines: bool = False) -> tuple[str, int]:
    """``text`` masked unit by unit: blocks, or lines when ``lines``."""
    if lines:
        parts = text.split("\n")
        res = [mask_text(p) for p in parts]
        return "\n".join(r[0] for r in res), sum(r[1] for r in res)
    out, pos, n, prev = [], 0, 0, ""
    for a, b in blocks(text):
        out.append(text[pos:a])
        blk = text[a:b]
        s, k = mask_text(blk, "" if _LETTER.search(blk) else prev + "\n")
        out.append(s)
        pos, n, prev = b, n + k, blk
    out.append(text[pos:])
    return "".join(out), n


# ---- JSON, formatting kept ------------------------------------------------------------------
_JTOK = re.compile(r'(?P<s>"(?:[^"\\]|\\.)*")|(?P<n>-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)'
                   r'|(?P<p>[{}\[\]:,])|(?P<w>\s+)|(?P<l>true|false|null|NaN|-?Infinity)')


def _key_kind(key: str) -> str | None:
    """The term a JSON key names: H, S (a 6-part or headline key, or "six"), or another kind
    that stops the search (P, O, Q)."""
    if key.strip().lower() == "six":
        return "S"
    kinds = [t[2] for t in _terms(key)]
    for k in ("H", "S", "A", "P", "O", "Q"):
        if k in kinds:
            return "S" if k == "A" else k
    return None


def _json_walk(raw: str, on_string, on_number) -> tuple[str, int]:
    json.loads(raw)
    out, n, pos = [], 0, 0
    stack: list[list] = []                    # [type, current key, expecting a key]
    for m in _JTOK.finditer(raw):
        if m.start() != pos:
            raise ValueError("unexpected JSON text")
        pos = m.end()
        tok = m.group(0)
        if m.lastgroup == "w":
            out.append(tok)
            continue
        if m.lastgroup == "p":
            if tok in "{[":
                stack.append(["{" if tok == "{" else "[", None, tok == "{"])
            elif tok in "}]":
                stack.pop()
            elif tok == "," and stack and stack[-1][0] == "{":
                stack[-1][2] = True
            out.append(tok)
            continue
        if m.lastgroup == "s" and stack and stack[-1][0] == "{" and stack[-1][2]:
            stack[-1][1], stack[-1][2] = json.loads(tok), False
            out.append(tok)
            continue
        if m.lastgroup == "s":
            new, k = on_string(json.loads(tok))
            if k:
                tok = json.dumps(new, ensure_ascii="\\u" in tok)
                n += k
        elif m.lastgroup == "n":
            keys = [fr[1] for fr in stack if fr[0] == "{" and fr[1] is not None]
            tok, k = on_number(tok, keys)
            n += k
        out.append(tok)
    if pos != len(raw):
        raise ValueError("unexpected JSON text")
    return "".join(out), n


def _mask_number(tok: str, keys: list[str]) -> tuple[str, int]:
    for key in reversed(keys):
        kind = _key_kind(key)
        if kind in ("H", "S"):
            return json.dumps(PLACEHOLDERS["honesty" if kind == "H" else "six_part"]), 1
        if kind is not None:
            return tok, 0
    return tok, 0


def mask_json_text(raw: str) -> tuple[str, int]:
    """A JSON document masked without reformatting it: string values by ``mask_blocks``,
    numbers by their key path (the nearest key naming a term decides). Raises ``ValueError``
    on a document that is not JSON."""
    return _json_walk(raw, mask_blocks, _mask_number)


def mask_file_text(name: str, raw: str) -> tuple[str, int] | None:
    """``raw`` (the content of a file called ``name``) masked by its kind, or None when the
    kind is not masked (binary, data arrays and the like)."""
    suf = Path(name).suffix.lower()
    if suf == ".json":
        try:
            return mask_json_text(raw)
        except ValueError:
            return mask_blocks(raw)
    if suf == ".jsonl":
        res = []
        for ln in raw.split("\n"):
            try:
                res.append(mask_json_text(ln) if ln.strip() else (ln, 0))
            except ValueError:
                res.append(mask_text(ln))
        return "\n".join(r[0] for r in res), sum(r[1] for r in res)
    if suf in MARKUP_SUFFIXES:
        return mask_markup(raw)
    if suf in TEXT_SUFFIXES:
        return mask_blocks(raw, lines=suf in LINE_SUFFIXES)
    return None


def _markup_parts(raw: str, js: bool = False) -> list[tuple[int, int, str]]:
    """(start, end, kind) of the maskable parts of HTML or JS (``js``): 'json' for a JSON
    script payload; 'text' for a string literal's content (in scripts and attributes) or the
    text between two tags outside scripts."""
    lit = re.compile(MARKUP_UNIT.split("|(?<=>)")[0])
    parts: list[tuple[int, int, str]] = []

    def literals(a: int, b: int) -> None:
        parts.extend((m.start() + 1, m.end() - 1, "text") for m in lit.finditer(raw, a, b)
                     if m.end() - m.start() >= 2)

    if js:
        literals(0, len(raw))
        return parts
    pos = 0
    for m in _SCRIPT.finditer(raw):
        parts.extend((t.start(), t.end(), "text") for t in _MARKUP.finditer(raw, pos, m.start())
                     if t.group(0)[:1] not in "\"'`")
        literals(pos, m.start())
        if "application/json" in m.group(1):
            parts.append((m.start(2), m.end(2), "json"))
        else:
            literals(m.start(2), m.end(2))
        pos = m.end()
    parts.extend((t.start(), t.end(), "text") for t in _MARKUP.finditer(raw, pos, len(raw))
                 if t.group(0)[:1] not in "\"'`")
    literals(pos, len(raw))
    return sorted(set(parts))


def mask_markup(raw: str, js: bool = False) -> tuple[str, int]:
    """HTML or JS masked part by part (UNIT): string literals and text between tags as texts,
    a JSON script payload as JSON; code and numbers outside them are left as they are."""
    out, pos, n = [], 0, 0
    for a, b, kind in _markup_parts(raw, js):
        seg = raw[a:b]
        if kind == "json":
            try:
                new, k = mask_json_text(seg)
            except ValueError:
                new, k = seg, 0
        else:
            new, k = mask_text(seg)
        out += [raw[pos:a], new]
        pos, n = b, n + k
    out.append(raw[pos:])
    return "".join(out), n


# ---- hashes of masked files -----------------------------------------------------------------
def digests(content: bytes) -> list[str]:
    """The hex digests a record may print for a file: sha256 and sha1."""
    return [hashlib.sha256(content).hexdigest(), hashlib.sha1(content).hexdigest()]


def mask_hashes(text: str, digs: list[str]) -> tuple[str, int]:
    """``text`` with every hex run of 7+ characters that starts one of ``digs`` replaced."""
    if not digs or not text:
        return text, 0
    n = 0

    def sub(m: re.Match) -> str:
        nonlocal n
        h = m.group(0).lower()
        if any(d.startswith(h) for d in digs):
            n += 1
            return PLACEHOLDERS["hash"]
        return m.group(0)

    return _HASH.sub(sub, text), n


# ---- the derivation gate --------------------------------------------------------------------
def units(name: str, raw: str) -> list[str]:
    """The units of a file's text the mask works on (see UNIT); [] for kinds it skips."""
    suf = Path(name).suffix.lower()
    if suf == ".jsonl":
        return [u for ln in raw.split("\n") if ln.strip() for u in units("x.json", ln)]
    if suf == ".json":
        got: list[str] = []

        def take(s: str) -> tuple[str, int]:
            got.extend(units("x.md", s))
            return s, 0

        try:
            _json_walk(raw, take, lambda tok, keys: (tok, 0))
            return got
        except ValueError:
            suf = ".md"
    if suf in MARKUP_SUFFIXES:
        got: list[str] = []
        for a, b, kind in _markup_parts(raw):
            got += units("x.json", raw[a:b]) if kind == "json" else [raw[a:b]]
        return got
    if suf in LINE_SUFFIXES:
        return raw.split("\n")
    if suf in TEXT_SUFFIXES or suf in CODE_SUFFIXES:
        return [raw[a:b] for a, b in blocks(raw)]
    return []


def _owned_by_science(unit: str, figs: list[re.Match]) -> set[int]:
    """Start offsets of the figures an O or Q term owns, directly or through a chain or a
    parallel list (a science or part figure; the mask's own decision 'keep')."""
    every = [m for m in _FIG.finditer(unit) if _is_figure(m, unit)]
    keep = {m.start() for m, d in zip(every, _decisions(unit, every), strict=True)
            if d == "keep"}
    return {m.start() for m in figs} & keep


def derivable_pairs(unit: str) -> list[tuple[float, float, float]]:
    """(a, b, 6a - 5b) for every pair of unmasked figures near an H, S or A term of a unit
    that could be a 6-part figure a beside a science figure b (GATE); a figure a science or
    part term owns is not taken for a."""
    low = unit.lower()
    if not any(w in low for w in _ANCHORS):
        return []
    anchors = [t for t in _terms(unit) if t[2] in "HSA"]
    if not anchors:
        return []
    w = GATE["window"]
    figs = [m for m in _gate_figures(unit)
            if any(t[0] - w <= m.start() <= t[1] + w for t in anchors)]
    sci = _owned_by_science(unit, figs)
    a_vals = sorted({_value(m.group("num")) for m in figs if m.start() not in sci})
    vals = sorted({_value(m.group("num")) for m in figs})
    ga, gb, gh = GATE["a"], GATE["b"], GATE["honesty"]
    out = []
    for a in a_vals:
        if not ga[0] <= a <= ga[1]:
            continue
        for b in vals:
            h = 6 * a - 5 * b
            if b != a and gb[0] <= b <= gb[1] and abs(a - b) <= GATE["max_diff"] and \
                    gh[0] <= h <= gh[1] and (a != int(a) or b != int(b)):
                out.append((a, b, round(h, 3)))
    return out


def _gate_figures(unit: str) -> list[re.Match]:
    """The unmasked figures of a unit the gates look at: one number, at most two decimals, not
    in a data series."""
    figs = [m for m in _FIG.finditer(unit) if _is_figure(m, unit) and not m.group("num2")]
    data, run = set(), []
    for m in figs:
        if run and not _SERIES.fullmatch(unit[run[-1].end():m.start()]):
            if len(run) >= GATE["series"]:
                data.update(x.start() for x in run)
            run = []
        run.append(m)
    if len(run) >= GATE["series"]:
        data.update(x.start() for x in run)
    return [m for m in figs if m.start() not in data
            and len(m.group("num").partition(".")[2]) <= 2]


def free_pairs(unit: str) -> list[tuple[float, float, float]]:
    """Term-free review list (reported, not a gate): pairs of unmasked figures within
    GATE["window"] characters of each other with 6a - 5b in GATE["free_honesty"]."""
    figs = _gate_figures(unit)
    ga, gb, gh = GATE["a"], GATE["b"], GATE["free_honesty"]
    pos = [m.start() for m in figs]
    out = set()
    for x in figs:
        a = _value(x.group("num"))
        if not ga[0] <= a <= ga[1]:
            continue
        lo = bisect.bisect_left(pos, x.start() - GATE["window"])
        hi = bisect.bisect_right(pos, x.start() + GATE["window"])
        for y in figs[lo:hi]:
            b = _value(y.group("num"))
            if y is x or not gb[0] <= b <= gb[1]:
                continue
            h = 6 * a - 5 * b
            if abs(a - b) <= GATE["max_diff"] and gh[0] <= h <= gh[1] and \
                    (a != int(a) or b != int(b)):
                out.add((a, b, round(h, 3)))
    return sorted(out)
