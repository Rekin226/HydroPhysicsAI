"""The grader-view mask of user ruling R11.4: claim-honesty figures and 6-part figures hidden.

R11.4 (user rulings 2026-11-08, twenty-fourth set) asks for a committed, mechanical masking rule
for the claim-honesty re-audit: the graders must not see the current claim-honesty input, the
per-source audit figures, or the 6-part figures beside science-trust figures (honesty =
6 x 6-part - 5 x science, so a 6-part figure next to a science figure gives honesty away).
Round 9's mask was ad hoc; this module is the rule, hashed (``rules_sha256``) and applied the
same way to the grader view of the inventory (``honesty.grader_view``) and to the text files of
a grader workspace (``dataprep_staging/honesty/make_grader_workspace.py``). It is defined by
patterns with context, never by a list of statement ids or of known figures.

UNIT. A text is masked one unit at a time; terms in one unit never reach a figure in another.
A unit is a block of ``BLOCK`` (``claims.sentences``'s non-period boundaries: a blank line, a
new list item, a numbered item or a table row start a new block), so a sentence sees the terms
of the sentences before it in its paragraph or list item. An inventory statement is masked with
its block as context (doc statements; ``honesty.grader_view`` finds the block in the pinned
file) or alone (page strings, ledger paragraphs). Line-oriented files (``.html``, ``.js``,
``.log``) use one line per unit; data tables (CSV) and binary files are not masked. In JSON,
every string value is a text of its own, and a number is masked when the nearest key on its
path that names a term names an honesty or a 6-part term (``"claim_honesty": 97.4``,
``"six_part": {"official": 52.1}``).

FIGURE (``FIGURE``): a number with an optional sign (``+``, ``-``, ``−``, ``±``, ``+-``,
``+/-``), an optional range end (``97.1-97.6``, ``61-62``, ``-0.2 to +0.1``) and an optional
``%``. Not a figure: part of an identifier, path, version, hash, date or time (a letter, digit,
``_``, ``.``, ``/``, ``#``, ``:`` or ``-`` right before it; a letter, digit, ``_`` or ``-`` plus
a letter or digit right after it), a fraction (``5/6``), a year, year range or year-month (an
integer 1900-2100, alone or with a year or two-digit end), or a number named by a counter word
right before it (``section 27``, ``step 0``, ``round 9``, ``caveat 12``; ``COUNTER_WORDS``).

TERMS (EN and ZH, case-insensitive; ``TERMS``):

* honesty terms H: ``honesty`` (claim honesty, claim-honesty, claim_honesty), audit or audited
  figure, re-audit, grader(s); 誠實, 評分者, 複核.
* 6-part terms S: ``6-part`` / ``six-part`` / ``six_part`` / ``6 part``; 六部分, 六項. A
  quoted key ``"six":`` is a K term: it counts for the 6-part unit rule only.
* adjacent 6-part terms A (a 6-part figure only when right beside a figure): ``headline`` (not
  "headline lever" or "headline head lever"), ``progress`` (whole words, not ``_progress``);
  總分, 頭條 (not 頭條槓桿), 進度.
* other-quantity terms O: ``science`` / ``science trust``, a bare ``part``/``parts`` (a science
  part), ``coverage``; 科學, 涵蓋. Statistic terms Q: ``kappa``, ``agreement``; 一致.

SCOPE. A figure is masked when any of these holds:

* 6-part scope: the nearest S, O or Q term before it in the unit is an S term (O and Q terms
  end the scope; H terms do not), or an S or A term follows it within ``TRAILING_GAP`` (at most
  12 characters, no digit and no ``;,.|()[]{}`` or CJK stop or comma between: ``-0.3
  headline``, ``61-62 % on the 6-part basis``), or it follows an A term within ``LEADING_GAP``
  (at most 16 characters, no digit and no ``;,.|`` or CJK stop or comma between: ``headline
  52.1``, ``progress ~52.1 %``); and the figure is signed, has a decimal point or is at least
  ``SMALL_INTEGER`` (10): a 6-part level or delta of any size, but not a grade, a weight or a
  small count (``graded 0``, ``divided by 6``).
* 6-part unit: the unit holds an S or K term and one of the figure's numbers lies in
  ``HONESTY_RANGE`` [50, 100), wherever it stands (a list such as "from 52.9, 44.2 and 45.6"
  names the 6-part figure far from its term; science trust lies below 50 and is left).
* honesty scope: the nearest H or O term before it in the unit is an H term (an O term ends the
  scope; S and Q terms do not), or an H term follows it within ``TRAILING_GAP``; and the figure
  is signed (a change) or one of its numbers lies in ``HONESTY_RANGE`` [50, 100) (a
  claim-honesty level; 100 itself is the formula's scale and is left, as are grades 0-3, counts
  below 50 and other small numbers).

The masked figure (sign, range and ``%`` included) is replaced by a fixed placeholder: when
both scopes mask it, the closest owning term (the nearest H or S term before it, an adjacent A
term before it, or a trailing S, A or H term; fewest characters away) chooses
``PLACEHOLDERS["honesty"]`` (H) or ``["six_part"]`` (S, A); otherwise the scope that masks it
does. Science-trust figures, part figures and every number outside both scopes are left as
written. Same text, same output.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import re
from pathlib import Path

RULING = "R11.4 (results/dataprep/rulings/user_rulings_2026-11-08f.json)"
PLACEHOLDERS = {"honesty": "[masked: claim-honesty figure]",
                "six_part": "[masked: 6-part figure]"}
HONESTY_RANGE = (50.0, 100.0)
YEARS = (1900, 2100)

_NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_SIGN = r"\+/-|\+-|±|[+\-−]"
FIGURE = (r"(?<![A-Za-z0-9_./#:\-−])"
          rf"(?P<sign>{_SIGN})?(?P<num>{_NUM})"
          rf"(?:\s*(?:-|–|to)\s*(?P<sign2>{_SIGN})?(?P<num2>{_NUM}))?"
          r"(?P<pct>\s?%)?"
          r"(?![A-Za-z0-9_]|/\d|:\d|\.\d|-[A-Za-z0-9])")
COUNTER_WORDS = ("section", "sections", "step", "steps", "round", "rounds", "stage", "line",
                 "lines", "caveat", "caveats", "set", "sets", "rule", "ruling", "rulings",
                 "row", "rows", "tier", "phase", "batch", "version", "PR", "§", "第")
TERMS = {
    "H": [r"(?<![A-Za-z])honesty", r"audit(?:ed)?\s+figures?", r"re-?audit(?:ed|s|ing)?",
          r"(?<![A-Za-z])graders?(?![A-Za-z])", r"誠實", r"評分者", r"複核"],
    "S": [r"(?<![A-Za-z0-9])(?:6|six)[-_ ]parts?(?![A-Za-z])", r"六部分", r"六項"],
    "K": [r"(?<=[\"'])six(?=[\"']\s*:)"],
    "A": [r"(?<![A-Za-z0-9_])headline(?![A-Za-z0-9_])(?!\s+(?:head\s+)?levers?)",
          r"(?<![A-Za-z0-9_])progress(?![A-Za-z0-9_])", r"總分", r"頭條(?!槓桿)", r"進度"],
    "O": [r"(?<![A-Za-z])science(?:[-_ ]trust)?(?![A-Za-z])",
          r"(?<![-_A-Za-z0-9])parts?(?![-A-Za-z0-9])", r"(?<![A-Za-z])coverage(?![A-Za-z])",
          r"科學", r"涵蓋"],
    "Q": [r"(?<![A-Za-z])kappa(?![A-Za-z])", r"(?<![A-Za-z])agreement(?![A-Za-z])", r"一致"],
}
TRAILING_GAP = r"[^\d;,.|()\[\]{}。；，、]{0,12}"
LEADING_GAP = r"[^\d;,.|。；，、]{0,16}"
SMALL_INTEGER = 10                    # 6-part scope leaves unsigned integers below this
GAP_MAX = (12, 16)                    # the two gaps' lengths, checked before the patterns
BLOCK = r"\n\s*\n|\n(?=\s*(?:[-*|]|\d+\.)\s)"
LINE_SUFFIXES = (".html", ".htm", ".js", ".log")
TEXT_SUFFIXES = (".md", ".txt", ".py", ".rst", ".tex", ".yaml", ".yml", ".toml", ".cfg",
                 ".ini", ".sh") + LINE_SUFFIXES
JSON_SUFFIXES = (".json", ".jsonl")

_FIG = re.compile(FIGURE)
_TERM = re.compile("|".join(f"(?P<{k}{i}>{p})" for k, ps in TERMS.items()
                            for i, p in enumerate(ps)), re.I)
_TRAIL = re.compile(TRAILING_GAP)
_LEAD = re.compile(LEADING_GAP)
_BLOCK = re.compile(BLOCK)
_ANCHORS = ("honest", "audit", "grader", "part", "six", "headline", "progress", "science",
            "coverage", "kappa", "agreement", "誠實", "評分者", "複核", "六部分", "六項", "總分",
            "頭條", "進度", "科學", "涵蓋", "一致")      # every term holds one (a fast pre-check)
_CAND = re.compile("|".join(re.escape(w) for w in _ANCHORS))
_COUNTER = re.compile(r"(?:(?<![A-Za-z])(?:" + "|".join(
    re.escape(w) for w in COUNTER_WORDS if w.isascii()) + r")|§|第)\s*$", re.I)


def rules() -> dict:
    """The mask rule as data; ``rules_sha256`` is its hash (canonical JSON)."""
    return {
        "ruling": RULING,
        "rule": __doc__.split("\n\n", 1)[1].strip(),
        "placeholders": PLACEHOLDERS, "honesty_range": list(HONESTY_RANGE),
        "small_integer": SMALL_INTEGER, "gap_max": list(GAP_MAX),
        "years_excluded": list(YEARS), "figure": FIGURE, "counter_words": list(COUNTER_WORDS),
        "terms": TERMS, "trailing_gap": TRAILING_GAP, "leading_gap": LEADING_GAP,
        "block": BLOCK, "line_suffixes": list(LINE_SUFFIXES),
        "text_suffixes": list(TEXT_SUFFIXES), "json_suffixes": list(JSON_SUFFIXES),
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
    return not _COUNTER.search(text[max(0, m.start() - 12):m.start()])


class _Terms:
    """The terms of a unit, indexed for the scope questions of ``_kind`` (linear, not
    quadratic, in the number of figures and terms)."""

    def __init__(self, text: str):
        self.t = _terms(text)
        self.starts = [t[0] for t in self.t]
        self.ends = [t[1] for t in self.t]
        self.has_six = any(t[2] in "SK" for t in self.t)
        self.last: dict[str, list[int]] = {}
        for kinds in ("SOQ", "HO", "HS"):
            cur, arr = -1, [-1]
            for i, t in enumerate(self.t):
                cur = i if t[2] in kinds else cur
                arr.append(cur)
            self.last[kinds] = arr          # last[k][n]: last index < n of a kind in k, or -1


def _kind(m: re.Match, text: str, terms: _Terms) -> str | None:
    """'honesty', 'six_part' or None for the figure ``m`` of ``text`` (see SCOPE)."""
    a, b = m.start(), m.end()
    nb = bisect.bisect_right(terms.ends, a)                  # terms[:nb] end before the figure
    t = terms.t
    i = terms.last["SOQ"][nb]
    six = i >= 0 and t[i][2] == "S"
    i = terms.last["HO"][nb]
    hon = i >= 0 and t[i][2] == "H"
    i = terms.last["HS"][nb]
    owners = [(a - t[i][1], t[i][2])] if i >= 0 else []       # (distance, kind)
    if nb and t[nb - 1][2] == "A" and a - t[nb - 1][1] <= GAP_MAX[1] and \
            _LEAD.fullmatch(text[t[nb - 1][1]:a]):
        six = True
        owners.append((a - t[nb - 1][1], "S"))
    na = bisect.bisect_left(terms.starts, b)
    if na < len(t) and t[na][2] in "SAH" and t[na][0] - b <= GAP_MAX[0] and \
            _TRAIL.fullmatch(text[b:t[na][0]]):
        k = "H" if t[na][2] == "H" else "S"
        six, hon = six or k == "S", hon or k == "H"
        owners.append((t[na][0] - b, k))
    label = min(owners)[1] if owners else None
    vals = [_value(m.group("num"))] + ([_value(m.group("num2"))] if m.group("num2") else [])
    signed = bool(m.group("sign") or m.group("sign2"))
    in_range = any(HONESTY_RANGE[0] <= v < HONESTY_RANGE[1] for v in vals)
    hon = hon and (signed or in_range)
    six = (six and (signed or "." in m.group(0) or any(v >= SMALL_INTEGER for v in vals))) \
        or (terms.has_six and in_range)
    if six and hon:
        return "honesty" if label == "H" else "six_part"
    return "honesty" if hon else "six_part" if six else None


def mask_text(text: str, before: str = "", after: str = "") -> tuple[str, int]:
    """(``text`` masked, number of figures masked); ``before`` and ``after`` are the rest of its
    unit (terms only: their own figures are not touched)."""
    if not text:
        return text, 0
    full = before + text + after
    lo, hi = len(before), len(before) + len(text)
    low = full.lower()
    if not any(w in low for w in _ANCHORS):
        return text, 0
    terms = None
    out, pos, n = [], lo, 0
    for m in _FIG.finditer(full, lo, hi):
        if m.end() > hi or not _is_figure(m, full):
            continue
        terms = terms or _Terms(full)
        k = _kind(m, full, terms)
        if k is None:
            continue
        out += [full[pos:m.start()], PLACEHOLDERS[k]]
        pos, n = m.end(), n + 1
    out.append(full[pos:hi])
    return "".join(out), n


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
    out, pos, n = [], 0, 0
    for a, b in blocks(text):
        out.append(text[pos:a])
        s, k = mask_text(text[a:b])
        out.append(s)
        pos, n = b, n + k
    out.append(text[pos:])
    return "".join(out), n


# ---- JSON, formatting kept ------------------------------------------------------------------
_JTOK = re.compile(r'(?P<s>"(?:[^"\\]|\\.)*")|(?P<n>-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)'
                   r'|(?P<p>[{}\[\]:,])|(?P<w>\s+)|(?P<l>true|false|null)')


def _key_kind(key: str) -> str | None:
    """The term a JSON key names (an A term counts as S here: a key is not prose)."""
    if key.strip().lower() == "six":
        return "S"
    kinds = [t[2] for t in _terms(key)]
    for k in ("H", "S", "A", "O"):
        if k in kinds:
            return "S" if k == "A" else k
    return None


def mask_json_text(raw: str) -> tuple[str, int]:
    """A JSON document masked without reformatting it: string values by ``mask_blocks``,
    numbers by their key path (the nearest key naming a term decides). Raises ``ValueError``
    on a document that is not JSON."""
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
            val = json.loads(tok)
            new, k = mask_blocks(val)
            if k:
                tok = json.dumps(new, ensure_ascii="\\u" in tok)
                n += k
        elif m.lastgroup == "n":
            kind = None
            for fr in reversed(stack):
                if fr[0] == "{" and fr[1] is not None and (kind := _key_kind(fr[1])):
                    break
            if kind in ("H", "S"):
                tok = json.dumps(PLACEHOLDERS["honesty" if kind == "H" else "six_part"])
                n += 1
        out.append(tok)
    if pos != len(raw):
        raise ValueError("unexpected JSON text")
    return "".join(out), n


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
    if suf in TEXT_SUFFIXES:
        return mask_blocks(raw, lines=suf in LINE_SUFFIXES)
    return None
