"""Claim-honesty audit machinery of user ruling R9.3: the statement inventory and the scorer.

R9.3 (user rulings 2026-11-07, sixteenth set): honesty = 100 x (1 - sum(w_i s_i) / (3 sum(w_i))),
where s_i is the worst finding on statement i (contradiction 3, overclaim 2, stale or missing
disclosure 1, none 0) and w_i is its display weight. The inventory is built mechanically from
the pinned rendered page (EN/ZH counted as pairs), the README, the model card, the evidence
update (sections 21-27: section 27 exists since round 9 and is included) and the ledger's own
wording and notes; it is hashed before grading. Grading is done by a blind, independent auditor
and a second grader on a sample; analyst answers are not statements of the inventory. This
module builds the inventory and scores findings. It grades nothing.

INVENTORY (``inventory``): one record per statement.

* PAGE: one record per i18n key the page renders, the EN and ZH strings together (one pair, one
  grade). Both strings come from the built page's ``const T = {en: ..., zh: ...}`` dictionary
  (``claims.i18n``; raw JS string contents, ``{name}`` placeholders kept and listed). A key is
  rendered when, in the page with the dictionary and the payload block removed, it appears as a
  quoted literal (``"k"``, ``'k'`` or a backtick pair: this covers ``tr("k")``, chosen keys,
  lookup tables and ``data-i18n="k"``), or when it starts with a prefix the page concatenates
  into a ``tr(`` key (``tr("ly_" + id)``: every ``"prefix" +`` first argument). A key in the
  dictionary that is never asked for is not rendered and is not a statement.
* README.md, MODEL_CARD.md and docs/TWIN_EVIDENCE_2026-10.md sections 21-27 (from the heading
  ``## 21.`` to the first ``## N.`` heading with N > 27, or the end of the file): fenced code
  blocks are removed (line numbers kept); a markdown heading line is one sentence; the rest is
  cut into sentences by ``claims.sentences``'s rule (a sentence ends at '.', '!' or '?'
  followed by white space, at a CJK stop, at a blank line, or before a list item or a table
  row). A sentence is a statement
  when it carries a claim marker (``MARKERS``): a digit, a ledger claim id, a registration id
  (a ``registry_hashes`` key or the pattern ``REG_PATTERN``), a cited test id, or a word or
  phrase of the fixed lists ``STATUS_WORDS`` / ``CLAIM_VERBS``. Markers are looked for in the
  sentence's visible text (link targets ``](...)`` and bare URLs removed); the record keeps the
  sentence as written, white space collapsed.
* LEDGER (``hydrophysics/twin/claims_v1.json``): each claim's ``wording_en`` and each paragraph
  (blank-line separated) of its ``notes``, non-empty, and each paragraph of the top-level
  ``status_note``. Every one is a statement (the ledger's own wording carries its claims by
  construction); no marker filter.

WEIGHT (fixed; written into the inventory as ``rules.weight_rule``):

* PAGE: the largest ``progress.derived_weight`` among the non-Z ledger claims the key maps to
  (2 for a claim on an answer tile or in the headline function, 1 for any other claim shown on
  the page); the mapping's per-claim overrides and mirror flags are not applied, since display,
  not the science-part count, is weighed here. A key that maps to no non-Z claim weighs 1 when
  its EN text carries a claim marker, or its ZH text carries a digit, claim id or registration
  id, or its visible EN text (HTML tags, ``{placeholders}`` and entities removed) has at least
  ``PAGE_SENTENCE_WORDS`` (6) words, i.e. it is a sentence, not a label: it is shown on the
  page and says something checkable. Otherwise it weighs 0 (interface label; it is listed, and
  a finding on it is reported, but it counts nothing). A key maps to a claim
  through ``display_map.i18n``, ``display_map.card_rows``, ``display_map.tiles`` or a claim's
  ``text_keys`` entry ``<kind>:<key>``.
* README, MODEL_CARD, evidence sections 21-27: 0.5 (progress.py's weight for a row shown only
  in the docs).
* LEDGER wording and notes: 0.5.

Each record also lists the claim ids it maps to (page: as above; docs: the ledger ids named in
the sentence and every claim whose ``doc_anchors`` "present" string for that file occurs in it;
ledger: its own row) and what it should be checked against: the registrations, tests and
reports the mapped claims cite, plus the registration ids and report paths the statement names.

The output JSON carries ``sha256``, the sha256 of the canonical record list (``json.dumps``
with sorted keys, no ASCII escaping, separators ``(",", ":")``), ``rules_sha256`` of the rules
block, and the sha256 of every input file. Same inputs, same records, same hashes.

SCORE (``score``): reads a findings file
``{"inventory_sha256": ..., "grader": ..., "findings": [{"id", "statement", "grade",
"category", "evidence"}]}``. It refuses a findings file made against another inventory hash, a
finding on an unknown statement id, a grade outside 0-3 or not an integer, a category that does
not match its grade (3 contradiction, 2 overclaim, 1 stale_or_missing_disclosure, 0 none), and
a duplicate finding id. Several findings on one statement are allowed; the worst one is s_i
(R9.3). A statement with no finding has s_i = 0. Reported: the overall figure, the figure per
source (the same formula over that source's statements), the weight and grade counts per source,
and every graded statement. With ``--sample-check`` (the second grader's findings file, which
must list its ``sample`` of statement ids): exact agreement and linear and quadratic weighted
Cohen's kappa between the two graders' worst grades on the sampled ids, and every disagreement.

CLI::

    python -m hydrophysics.twin.honesty inventory --page PAGE.html --out INVENTORY.json
    python -m hydrophysics.twin.honesty score --inventory INVENTORY.json --findings F.json \
        [--sample-check SECOND.json] [--json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

from hydrophysics.twin import claims as C
from hydrophysics.twin import progress as P

SCHEMA = "twin-honesty-inventory/1"
FINDINGS_SCHEMA = "twin-honesty-findings/1"
README = "README.md"
MODEL_CARD = "MODEL_CARD.md"
EVIDENCE = "docs/TWIN_EVIDENCE_2026-10.md"
LEDGER_PATH = "hydrophysics/twin/claims_v1.json"
EVIDENCE_SECTIONS = (21, 27)
SOURCES = ("page", "readme", "model_card", "evidence", "ledger")
DOC_WEIGHT = 0.5
LEDGER_WEIGHT = 0.5
PAGE_UNMAPPED_MARKED = 1.0
PAGE_SENTENCE_WORDS = 6
GRADES = {"none": 0, "stale_or_missing_disclosure": 1, "overclaim": 2, "contradiction": 3}

REG_PATTERN = r"(?<![A-Za-z0-9_./-])[A-Z][A-Za-z0-9]*(?:\.[A-Za-z0-9_-]+)*\.v\d+[a-z]?(?![A-Za-z0-9_])"
REPORT_PATTERN = r"(?<![A-Za-z0-9_/])(?:results|docs|dataprep_staging|hydrophysics|tests)/" \
                 r"[A-Za-z0-9_./-]*[A-Za-z0-9_/]"
STATUS_WORDS = (
    "supported", "unsupported", "not supported", "supports", "support", "decided", "decides",
    "undecided", "withdrawn", "withdraws", "narrowed", "pending", "passed", "passes", "pass",
    "failed", "fails", "fail", "validated", "verified", "confirmed", "contradicted",
    "registered", "pre-registered", "preregistered", "unregistered", "untested", "tested",
    "evaluable", "not evaluable", "decidable", "undecidable", "not decidable", "estimated",
    "measured", "model-only", "report-only", "design-time", "prospective", "prospectively",
    "held-out", "out-of-sample", "post-hoc", "within noise", "verdict", "scored", "unchanged",
    "closed", "superseded", "inconclusive", "consistent", "inconsistent", "not run",
    "model internals", "forecast of record", "scenario", "labelled",
)
CLAIM_VERBS = (
    "shows", "show", "showed", "demonstrates", "proves", "proven", "predicts", "predicted",
    "forecasts", "explains", "causes", "beats", "outperforms", "improves", "improved",
    "reduces", "reduced", "increases", "agrees", "disagrees", "matches", "exceeds", "covers",
    "covered", "excludes", "depends", "does not depend", "is within", "arrives", "leaves",
    "found", "finds",
)
_WORDS = re.compile(r"(?<![A-Za-z0-9-])(" + "|".join(
    re.escape(w) for w in sorted(STATUS_WORDS + CLAIM_VERBS, key=len, reverse=True))
    + r")(?![A-Za-z0-9-])", re.I)
_LINK = re.compile(r"\]\([^)]*\)")
_URL = re.compile(r"https?://\S+")
_FENCE = re.compile(r"^\s*(```|~~~)")
_SECTION = re.compile(r"^## (\d+)\.", re.M)
_HEADING = re.compile(r"^#{1,6} [^\n]*", re.M)
_PREFIX = re.compile(r'^\s*["\'`]([A-Za-z_][A-Za-z0-9_]*)["\'`]\s*\+')
_PAYLOAD = re.compile(r'<script id="payload" type="application/json">.*?</script>', re.S)
_PLACEHOLDER = re.compile(r"\{([A-Za-z0-9_]+)\}")
_PARA = re.compile(r"\n\s*\n")


class InventoryError(ValueError):
    """The inputs cannot give a defined inventory."""


class FindingsError(ValueError):
    """A findings file the scorer refuses."""


# ---- shared helpers -------------------------------------------------------------------------
def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def sha_of(obj) -> str:
    return hashlib.sha256(canonical(obj)).hexdigest()


def _id_regex(ids) -> re.Pattern | None:
    ids = sorted(set(ids), key=len, reverse=True)
    if not ids:
        return None
    return re.compile(r"(?<![A-Za-z0-9_.-])(" + "|".join(re.escape(i) for i in ids)
                      + r")(?![A-Za-z0-9_]|-[A-Za-z0-9])")


class _Vocab:
    """What a statement can name: ledger claim ids, registration ids and test ids."""

    def __init__(self, ledger: dict):
        self.claims = {c["id"]: c for c in ledger["claims"]}
        self.claim_re = _id_regex(self.claims)
        regs = set(ledger.get("registry_hashes", {}))
        tests: set[str] = set()
        for c in ledger["claims"]:
            cites = c.get("cites") or {}
            regs.update(cites.get("registrations") or [])
            tests.update(cites.get("tests") or [])
        self.reg_re = _id_regex(regs)
        self.test_re = _id_regex(tests - regs)
        self.reg_generic = re.compile(REG_PATTERN)
        self.report_re = re.compile(REPORT_PATTERN)

    def find(self, rx, text: str) -> list[str]:
        return sorted(set(rx.findall(text))) if rx is not None else []

    def claim_ids(self, text: str) -> list[str]:
        return self.find(self.claim_re, text)

    def registrations(self, text: str) -> list[str]:
        return sorted(set(self.find(self.reg_re, text)) | set(self.reg_generic.findall(text)))

    def tests(self, text: str) -> list[str]:
        return self.find(self.test_re, text)

    def reports(self, text: str) -> list[str]:
        return sorted({r.rstrip("./") for r in self.report_re.findall(text)})

    def markers(self, text: str) -> list[str]:
        vis = _URL.sub(" ", _LINK.sub("]", text))
        out = []
        if re.search(r"\d", vis):
            out.append("digit")
        if self.claim_ids(vis):
            out.append("claim_id")
        if self.registrations(vis):
            out.append("registration")
        if self.tests(vis):
            out.append("test")
        out += sorted({f"word:{w.lower()}" for w in _WORDS.findall(vis)})
        return out

    def check_against(self, claim_ids, text: str) -> dict:
        regs, tests, reps = set(self.registrations(text)), set(self.tests(text)), \
            set(self.reports(text))
        for cid in claim_ids:
            cites = self.claims[cid].get("cites") or {}
            regs.update(cites.get("registrations") or [])
            tests.update(cites.get("tests") or [])
            reps.update(cites.get("reports") or [])
        return {"registrations": sorted(regs), "tests": sorted(tests), "reports": sorted(reps)}


def _stable_ids(prefix: str, texts: list[str]) -> list[str]:
    """``prefix:<sha1 of the text, 10 hex>`` with ``-2``, ``-3``... for repeated texts, so an id
    does not move when an unrelated sentence is added above it."""
    seen: dict[str, int] = {}
    out = []
    for t in texts:
        h = hashlib.sha1(t.encode()).hexdigest()[:10]
        seen[h] = seen.get(h, 0) + 1
        out.append(f"{prefix}:{h}" + (f"-{seen[h]}" if seen[h] > 1 else ""))
    return out


# ---- the page -------------------------------------------------------------------------------
def rendered_keys(page_src: str, keys) -> tuple[list[str], list[str]]:
    """(rendered keys, the concatenation prefixes found) by the PAGE rule of the docstring."""
    t0 = page_src.index("const T = {")
    body = page_src[:t0] + page_src[page_src.index("\n}};", t0):]
    body = _PAYLOAD.sub("", body)
    prefixes = sorted({m.group(1) for a in C._tr_first_args(body)
                       if (m := _PREFIX.match(a))})
    lits = set(re.findall(r'"([A-Za-z_][A-Za-z0-9_]*)"', body)) | \
        set(re.findall(r"'([A-Za-z_][A-Za-z0-9_]*)'", body)) | \
        set(re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", body))
    out = [k for k in keys if k in lits or any(k.startswith(p) for p in prefixes)]
    return out, prefixes


def visible_words(s: str) -> list[str]:
    """The words of a page string as shown: HTML tags, placeholders and entities removed."""
    s = re.sub(r"&[A-Za-z#0-9]+;", " ", _PLACEHOLDER.sub(" ", re.sub(r"<[^>]*>", " ", s)))
    return re.findall(r"[A-Za-z][A-Za-z'’-]*", s)


def _key_claims(key: str, ledger: dict) -> list[str]:
    dm = ledger.get("display_map", {})
    ids = set(dm.get("i18n", {}).get(key, [])) | set(dm.get("card_rows", {}).get(key, [])) | \
        set(dm.get("tiles", {}).get(key, []))
    ids |= {c["id"] for c in ledger["claims"]
            if any(tk.split(":", 1)[-1] == key for tk in c.get("text_keys", [])
                   if not tk.startswith("doc:"))}
    return sorted(ids)


def page_records(page: Path, page_label: str, ledger: dict, pmap: dict, vocab: _Vocab) -> list:
    src = page.read_text(encoding="utf-8")
    en, zh = C.i18n(page, "en"), C.i18n(page, "zh")
    if set(en) != set(zh):
        raise InventoryError(f"EN/ZH key parity broken: {sorted(set(en) ^ set(zh))[:10]}")
    keys, _ = rendered_keys(src, sorted(en))
    recs = []
    for k in keys:
        cids = _key_claims(k, ledger)
        real = [c for c in cids if not c.startswith("Z") and c in vocab.claims]
        both = en[k] + " " + zh[k]
        if real:
            ws = [(P.derived_weight(vocab.claims[c], ledger, pmap)) for c in real]
            w, basis = max(ws, key=lambda t: t[0])
            basis = f"claim weight (progress.derived_weight): {basis}"
        else:
            marks = vocab.markers(en[k])
            zh_mark = re.search(r"\d", zh[k]) or vocab.claim_ids(zh[k]) or \
                vocab.registrations(zh[k])
            if marks or zh_mark:
                w, basis = PAGE_UNMAPPED_MARKED, "page text with a claim marker, no non-Z claim"
            elif len(visible_words(en[k])) >= PAGE_SENTENCE_WORDS:
                w, basis = PAGE_UNMAPPED_MARKED, "page sentence (>= 6 words), no non-Z claim"
            else:
                w, basis = 0.0, "interface label: no non-Z claim, no marker, < 6 words"
        recs.append({
            "id": f"page:{k}", "source": "page",
            "location": {"file": page_label, "key": k},
            "text": {"en": en[k], "zh": zh[k]},
            "placeholders": sorted(set(_PLACEHOLDER.findall(both))),
            "weight": float(w), "weight_basis": basis,
            "claim_ids": cids,
            "markers": vocab.markers(en[k]),
            "check_against": vocab.check_against(real, both),
        })
    return recs


# ---- the docs -------------------------------------------------------------------------------
def _strip_fences(text: str) -> str:
    out, inside = [], False
    for ln in text.split("\n"):
        if _FENCE.match(ln):
            inside = not inside
            out.append("")
        else:
            out.append("" if inside else ln)
    return "\n".join(out)


def sentences_with_lines(text: str) -> list[tuple[int, str]]:
    """(1-based line of the first character, sentence with white space collapsed), by the
    sentence rule of ``claims.sentences``."""
    out = []

    def emit(chunk: str, off: int) -> None:
        pos = 0
        bounds = [(m.start(), m.end()) for m in C._SENTENCE.finditer(chunk)]
        for a, b in bounds + [(len(chunk), len(chunk))]:
            seg = chunk[pos:a]
            s = " ".join(seg.split())
            if s:
                first = off + pos + (len(seg) - len(seg.lstrip()))
                out.append((text.count("\n", 0, first) + 1, s))
            pos = b

    start = 0                                  # a heading line is one sentence, never cut
    for m in _HEADING.finditer(text):
        emit(text[start:m.start()], start)
        emit_line = " ".join(m.group(0).split())
        out.append((text.count("\n", 0, m.start()) + 1, emit_line))
        start = m.end()
    emit(text[start:], start)
    return out


def evidence_slice(text: str, lo: int = EVIDENCE_SECTIONS[0],
                   hi: int = EVIDENCE_SECTIONS[1]) -> tuple[str, int]:
    """(the text of sections lo..hi, the line offset of its first line)."""
    heads = [(int(m.group(1)), m.start()) for m in _SECTION.finditer(text)]
    starts = [p for n, p in heads if n == lo]
    if not starts:
        raise InventoryError(f"{EVIDENCE}: no '## {lo}.' heading")
    if not any(n == hi for n, _ in heads):
        raise InventoryError(f"{EVIDENCE}: no '## {hi}.' heading")
    a = starts[0]
    ends = [p for n, p in heads if n > hi and p > a]
    b = ends[0] if ends else len(text)
    return text[a:b], text.count("\n", 0, a)


def doc_records(source: str, rel: str, text: str, ledger: dict, vocab: _Vocab,
                line0: int = 0) -> list:
    sec_heads = [(text.count("\n", 0, m.start()) + 1 + line0, int(m.group(1)))
                 for m in _SECTION.finditer(text)]
    anchors = {c["id"]: (c.get("doc_anchors") or {}).get(rel, {}).get("present", [])
               for c in ledger["claims"]}
    kept = []
    for line, s in sentences_with_lines(_strip_fences(text)):
        marks = vocab.markers(s)
        if marks:
            kept.append((line + line0, s, marks))
    ids = _stable_ids(source, [s for _, s, _ in kept])
    recs = []
    for rid, (line, s, marks) in zip(ids, kept, strict=True):
        cids = set(vocab.claim_ids(s)) | {c for c, ps in anchors.items()
                                          if any(" ".join(p.split()) in s for p in ps)}
        loc = {"file": rel, "line": line}
        if source == "evidence":
            loc["section"] = max((n for ln, n in sec_heads if ln <= line), default=None)
        recs.append({
            "id": rid, "source": source, "location": loc,
            "text": {"en": s, "zh": None}, "placeholders": [],
            "weight": DOC_WEIGHT, "weight_basis": "docs statement (README, MODEL_CARD or "
                                                  "evidence sections 21-27): 0.5",
            "claim_ids": sorted(cids), "markers": marks,
            "check_against": vocab.check_against(sorted(cids), s),
        })
    return recs


# ---- the ledger -----------------------------------------------------------------------------
def ledger_records(ledger: dict, vocab: _Vocab, rel: str = LEDGER_PATH) -> list:
    recs = []

    def add(rid, loc, text, cids):
        recs.append({
            "id": rid, "source": "ledger", "location": dict(loc, file=rel),
            "text": {"en": " ".join(text.split()), "zh": None}, "placeholders": [],
            "weight": LEDGER_WEIGHT, "weight_basis": "ledger wording or notes: 0.5",
            "claim_ids": cids, "markers": vocab.markers(text),
            "check_against": vocab.check_against(cids, text),
        })

    for i, para in enumerate(p for p in _PARA.split(ledger.get("status_note", "")) if p.strip()):
        add(f"ledger:status_note:{i}", {"field": "status_note", "paragraph": i}, para,
            vocab.claim_ids(para))
    for c in ledger["claims"]:
        cid = c["id"]
        if c.get("wording_en", "").strip():
            add(f"ledger:{cid}:wording_en", {"claim": cid, "field": "wording_en"},
                c["wording_en"], [cid])
        paras = [p for p in _PARA.split(c.get("notes") or "") if p.strip()]
        for i, para in enumerate(paras):
            add(f"ledger:{cid}:notes:{i}", {"claim": cid, "field": "notes", "paragraph": i},
                para, [cid])
    return recs


# ---- the inventory --------------------------------------------------------------------------
def rules() -> dict:
    return {
        "ruling": "R9.3 (user_rulings_2026-11-07c.json)",
        "formula": "honesty = 100 x (1 - sum(w_i s_i) / (3 sum(w_i))); s_i = worst finding on "
                   "statement i: contradiction 3, overclaim 2, stale_or_missing_disclosure 1, "
                   "none 0; w_i = display weight",
        "sources": {"page": "every i18n key the pinned page renders, EN+ZH as one statement",
                    "readme": README, "model_card": MODEL_CARD,
                    "evidence": f"{EVIDENCE} sections {EVIDENCE_SECTIONS[0]}-"
                                f"{EVIDENCE_SECTIONS[1]}",
                    "ledger": f"{LEDGER_PATH}: status_note paragraphs, each claim's "
                              "wording_en, each notes paragraph"},
        "extraction": __doc__.split("INVENTORY (``inventory``): one record per statement.")[1]
                      .split("WEIGHT (fixed")[0].strip(),
        "weight_rule": "WEIGHT (fixed" + __doc__.split("WEIGHT (fixed")[1]
                       .split("Each record also lists")[0].rstrip(),
        "weights": {"page": "max progress.derived_weight over the non-Z claims the key maps "
                            "to (2 answer tile or headline, 1 other page claim); no non-Z "
                            f"claim: {PAGE_UNMAPPED_MARKED} with a claim marker or >= "
                            f"{PAGE_SENTENCE_WORDS} visible words, else 0",
                    "readme": DOC_WEIGHT, "model_card": DOC_WEIGHT, "evidence": DOC_WEIGHT,
                    "ledger": LEDGER_WEIGHT},
        "status_words": list(STATUS_WORDS), "claim_verbs": list(CLAIM_VERBS),
        "registration_pattern": REG_PATTERN, "report_pattern": REPORT_PATTERN,
    }


def _rel(p: Path, root: Path) -> str:
    try:
        return p.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return p.name


def build_inventory(page: Path, root: Path = C.REPO, ledger_path: Path | None = None,
                    map_path: Path = P.MAP) -> dict:
    root = Path(root)
    ledger_path = Path(ledger_path) if ledger_path else root / LEDGER_PATH
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    pmap = P.load_map(map_path)
    vocab = _Vocab(ledger)
    page_label = _rel(page, root)
    files = {README: root / README, MODEL_CARD: root / MODEL_CARD, EVIDENCE: root / EVIDENCE}
    for rel, p in files.items():
        if not p.is_file():
            raise InventoryError(f"missing input {rel}")
    recs = page_records(page, page_label, ledger, pmap, vocab)
    recs += doc_records("readme", README, files[README].read_text(encoding="utf-8"), ledger,
                        vocab)
    recs += doc_records("model_card", MODEL_CARD,
                        files[MODEL_CARD].read_text(encoding="utf-8"), ledger, vocab)
    ev, line0 = evidence_slice(files[EVIDENCE].read_text(encoding="utf-8"))
    recs += doc_records("evidence", EVIDENCE, ev, ledger, vocab, line0)
    recs += ledger_records(ledger, vocab, _rel(ledger_path, root))
    ids = [r["id"] for r in recs]
    if len(ids) != len(set(ids)):
        raise InventoryError("duplicate statement ids")
    ru = rules()
    inputs = {page_label: C.sha256(page), _rel(ledger_path, root): C.sha256(ledger_path),
              _rel(Path(map_path), root): C.sha256(Path(map_path))}
    inputs.update({rel: C.sha256(p) for rel, p in files.items()})
    return {"schema": SCHEMA, "sha256": sha_of(recs), "rules_sha256": sha_of(ru),
            "inputs": inputs, "commit": _commit(root), "summary": summarize(recs),
            "rules": ru, "records": recs}


def _commit(root: Path) -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                              capture_output=True, text=True).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def summarize(recs: list) -> dict:
    out = {}
    for s in SOURCES:
        rs = [r for r in recs if r["source"] == s]
        out[s] = {"statements": len(rs), "weight": sum(r["weight"] for r in rs),
                  "zero_weight": sum(1 for r in rs if r["weight"] == 0)}
    out["total"] = {"statements": len(recs), "weight": sum(r["weight"] for r in recs),
                    "zero_weight": sum(1 for r in recs if r["weight"] == 0)}
    return out


def load_inventory(path: Path) -> dict:
    inv = json.loads(Path(path).read_text(encoding="utf-8"))
    if inv.get("schema") != SCHEMA:
        raise FindingsError(f"{path}: not an inventory ({inv.get('schema')!r})")
    if sha_of(inv["records"]) != inv["sha256"]:
        raise FindingsError(f"{path}: records do not hash to the recorded sha256 (edited "
                            "after hashing)")
    return inv


# ---- the scorer -----------------------------------------------------------------------------
def load_findings(path_or_obj, inv: dict) -> dict:
    """The findings file, checked against the inventory (see the docstring's refusals)."""
    f = path_or_obj if isinstance(path_or_obj, dict) else \
        json.loads(Path(path_or_obj).read_text(encoding="utf-8"))
    if f.get("inventory_sha256") != inv["sha256"]:
        raise FindingsError(f"findings were made against inventory {f.get('inventory_sha256')!r}"
                            f", not {inv['sha256']!r}")
    known = {r["id"] for r in inv["records"]}
    seen: set[str] = set()
    for i, x in enumerate(f.get("findings", [])):
        fid = x.get("id")
        if not isinstance(fid, str) or not fid:
            raise FindingsError(f"finding #{i}: needs a string id")
        if fid in seen:
            raise FindingsError(f"duplicate finding id {fid!r}")
        seen.add(fid)
        if x.get("statement") not in known:
            raise FindingsError(f"{fid}: unknown statement id {x.get('statement')!r}")
        g = x.get("grade")
        if isinstance(g, bool) or not isinstance(g, int) or not 0 <= g <= 3:
            raise FindingsError(f"{fid}: grade {g!r} is not an integer in 0-3")
        cat = x.get("category")
        if GRADES.get(cat) != g:
            raise FindingsError(f"{fid}: category {cat!r} does not match grade {g} "
                                f"({GRADES})")
        if g > 0 and not str(x.get("evidence") or "").strip():
            raise FindingsError(f"{fid}: a finding above 0 needs its evidence")
    if "sample" in f:
        bad = [s for s in f["sample"] if s not in known]
        if bad:
            raise FindingsError(f"sample names unknown statement ids: {bad[:5]}")
        if len(set(f["sample"])) != len(f["sample"]):
            raise FindingsError("sample lists a statement twice")
        off = sorted({x["statement"] for x in f["findings"]} - set(f["sample"]))
        if off:
            raise FindingsError(f"findings outside the declared sample: {off[:5]}")
    return f


def worst(findings: list) -> dict[str, dict]:
    """statement id -> its worst finding (R9.3: s_i is the worst finding on statement i)."""
    out: dict[str, dict] = {}
    for x in findings:
        cur = out.get(x["statement"])
        if cur is None or x["grade"] > cur["grade"]:
            out[x["statement"]] = x
    return out


def honesty(recs: list, s: dict[str, int]) -> float | None:
    wsum = sum(r["weight"] for r in recs)
    if wsum <= 0:
        return None
    return 100.0 * (1.0 - sum(r["weight"] * s.get(r["id"], 0) for r in recs) / (3.0 * wsum))


def kappa(pairs: list[tuple[int, int]], kind: str = "linear", k: int = 4) -> float | None:
    """Cohen's weighted kappa on grades 0..k-1; None when the expected disagreement is 0."""
    n = len(pairs)
    if not n:
        return None
    obs = [[0.0] * k for _ in range(k)]
    for a, b in pairs:
        obs[a][b] += 1.0 / n
    ra = [sum(row) for row in obs]
    cb = [sum(obs[i][j] for i in range(k)) for j in range(k)]

    def wt(i, j):
        d = abs(i - j) / (k - 1)
        return d if kind == "linear" else d * d

    do = sum(wt(i, j) * obs[i][j] for i in range(k) for j in range(k))
    de = sum(wt(i, j) * ra[i] * cb[j] for i in range(k) for j in range(k))
    return None if de == 0 else 1.0 - do / de


def score(inv: dict, findings: dict, sample: dict | None = None) -> dict:
    recs = inv["records"]
    w = worst(findings.get("findings", []))
    s = {sid: x["grade"] for sid, x in w.items()}
    by_id = {r["id"]: r for r in recs}
    per = {}
    for src in SOURCES:
        rs = [r for r in recs if r["source"] == src]
        counts = {g: sum(1 for r in rs if s.get(r["id"], 0) == g) for g in range(4)}
        per[src] = {"honesty": honesty(rs, s), "statements": len(rs),
                    "weight": sum(r["weight"] for r in rs),
                    "weighted_grade": sum(r["weight"] * s.get(r["id"], 0) for r in rs),
                    "grade_counts": counts}
    graded = sorted(({"statement": sid, "grade": x["grade"], "category": x["category"],
                      "finding": x["id"], "weight": by_id[sid]["weight"],
                      "source": by_id[sid]["source"], "evidence": x.get("evidence", "")}
                     for sid, x in w.items() if x["grade"] > 0),
                    key=lambda d: (-d["grade"], -d["weight"], d["statement"]))
    out = {"inventory_sha256": inv["sha256"], "grader": findings.get("grader"),
           "honesty": honesty(recs, s), "per_source": per,
           "statements": len(recs), "weight": sum(r["weight"] for r in recs),
           "findings": len(findings.get("findings", [])),
           "statements_with_a_finding": len(w),
           "graded": graded,
           "zero_weight_findings": [g["statement"] for g in graded if g["weight"] == 0]}
    if sample is not None:
        out["sample_check"] = sample_check(inv, findings, sample)
    return out


def sample_check(inv: dict, first: dict, second: dict) -> dict:
    ids = list(second["sample"]) if "sample" in second else None
    if ids is None:
        raise FindingsError("the second grader's file must list its sample of statement ids")
    a = {k: v["grade"] for k, v in worst(first.get("findings", [])).items()}
    wb = worst(second.get("findings", []))
    b = {k: v["grade"] for k, v in wb.items()}
    pairs = [(a.get(i, 0), b.get(i, 0)) for i in ids]
    recs = [r for r in inv["records"] if r["id"] in set(ids)]
    dis = [{"statement": i, "auditor": x, "second": y,
            "auditor_evidence": (worst(first.get("findings", [])).get(i) or {}).get("evidence"),
            "second_evidence": (wb.get(i) or {}).get("evidence")}
           for i, (x, y) in zip(ids, pairs, strict=True) if x != y]
    return {"grader": second.get("grader"), "sample_size": len(ids),
            "exact_agreement": (sum(1 for x, y in pairs if x == y) / len(pairs)) if pairs
            else None,
            "kappa_linear": kappa(pairs, "linear"), "kappa_quadratic": kappa(pairs, "quadratic"),
            "honesty_on_sample_auditor": honesty(recs, a),
            "honesty_on_sample_second": honesty(recs, b),
            "disagreements": dis}


# ---- output and CLI -------------------------------------------------------------------------
def _f(v, nd=1) -> str:
    return "-" if v is None else f"{v:.{nd}f}"


def render_score(res: dict) -> str:
    out = [f"Claim honesty (R9.3) against inventory {res['inventory_sha256'][:12]} "
           f"(grader {res['grader'] or '-'})",
           f"overall: {_f(res['honesty'])}   statements {res['statements']}, weight "
           f"{res['weight']:g}, findings {res['findings']} on {res['statements_with_a_finding']}"
           " statements", "",
           f"{'source':12s} {'honesty':>8s} {'stmts':>6s} {'weight':>8s}  grades 0/1/2/3"]
    for src, p in res["per_source"].items():
        gc = p["grade_counts"]
        out.append(f"{src:12s} {_f(p['honesty']):>8s} {p['statements']:>6d} {p['weight']:>8g}  "
                   f"{gc[0]}/{gc[1]}/{gc[2]}/{gc[3]}")
    if res["graded"]:
        out += ["", "graded statements (worst finding):"]
        out += [f"  {g['grade']} {g['category']:28s} w={g['weight']:g} {g['statement']}"
                for g in res["graded"]]
    if res["zero_weight_findings"]:
        out.append(f"findings on zero-weight statements (count nothing): "
                   f"{', '.join(res['zero_weight_findings'])}")
    sc = res.get("sample_check")
    if sc:
        out += ["", f"sample check (second grader {sc['grader'] or '-'}, {sc['sample_size']} "
                    f"statements): exact agreement {_f(sc['exact_agreement'], 3)}, weighted "
                    f"kappa linear {_f(sc['kappa_linear'], 3)}, quadratic "
                    f"{_f(sc['kappa_quadratic'], 3)}"]
        out += [f"  {d['statement']}: auditor {d['auditor']} / second {d['second']}"
                for d in sc["disagreements"]]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hydrophysics.twin.honesty",
                                 description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("inventory", help="build the statement inventory")
    a1.add_argument("--page", type=Path, required=True, help="the pinned built page")
    a1.add_argument("--out", type=Path, required=True)
    a1.add_argument("--root", type=Path, default=C.REPO, help="repository root (docs)")
    a1.add_argument("--ledger", type=Path, default=None)
    a1.add_argument("--map", type=Path, default=P.MAP)
    a2 = sub.add_parser("score", help="score findings against an inventory")
    a2.add_argument("--inventory", type=Path, required=True)
    a2.add_argument("--findings", type=Path, required=True)
    a2.add_argument("--sample-check", type=Path, default=None,
                    help="the second grader's findings file (with its 'sample' list)")
    a2.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "inventory":
        inv = build_inventory(a.page, a.root, a.ledger, a.map)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(inv, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"inventory {a.out}: sha256 {inv['sha256']}")
        for src, v in inv["summary"].items():
            print(f"  {src:11s} statements {v['statements']:5d}  weight {v['weight']:8g}  "
                  f"(zero-weight {v['zero_weight']})")
        return 0
    inv = load_inventory(a.inventory)
    first = load_findings(a.findings, inv)
    second = load_findings(a.sample_check, inv) if a.sample_check else None
    if second is not None and "sample" not in second:
        raise FindingsError("the second grader's file must list its sample of statement ids")
    res = score(inv, first, second)
    if a.json:
        json.dump(res, sys.stdout, indent=1, ensure_ascii=False)
        sys.stdout.write("\n")
    else:
        print(render_score(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
