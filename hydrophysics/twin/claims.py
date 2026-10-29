"""Claims ledger of the decision page: what each displayed number claims, and on what record.

The ledger (``claims_v1.json`` beside this module) gives every number the page template or the
README/MODEL_CARD can show an id and a status (S supported by a registered test, D decided by a
registered prospective test, N narrowed wording the record supports, L model-internal and
labelled so, W withdrawn from the answer; U and X are not allowed). ``tests/test_twin_claims.py``
checks it against the template (always), the docs (always) and, where the gitignored files
exist, the pre-registration index and the built page.

Nothing here reads observation values. The page payload is parsed with every number replaced by
a placeholder, so only key paths and types are ever seen.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LEDGER = HERE / "claims_v1.json"
TEMPLATE = HERE / "app" / "template.html"
PREREG = REPO / "results" / "dataprep" / "preregistration"
CODES = set("SDNLUXW")


def load_ledger(path: Path = LEDGER) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def codes(status: str) -> list[str]:
    return status.split("+")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---- the page payload: structure only ----------------------------------------------------
def payload_structure(page: Path) -> dict:
    """The page payload with every number replaced by 0 (values never materialise)."""
    html = page.read_text(encoding="utf-8")
    m = re.search(r'<script id="payload" type="application/json">(.*?)</script>', html, re.S)
    if not m:
        raise ValueError("no payload block in the page")
    return json.loads(m.group(1).replace("<\\/", "</"), parse_float=lambda _: 0.0,
                      parse_int=lambda _: 0, parse_constant=lambda _: 0.0)


def numeric_paths(D: dict) -> set[str]:
    """Normalised key paths of every numeric leaf (bools excluded); list elements become '[]'."""
    out: set[str] = set()

    def walk(o, path):
        if isinstance(o, bool) or o is None or isinstance(o, str):
            return
        if isinstance(o, (int, float)):
            out.add(path)
        elif isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{path}.{k}" if path else str(k))
        elif isinstance(o, list):
            for v in o:
                walk(v, path + "[]")

    walk(D, "")
    return out


# ---- the template ---------------------------------------------------------------------------
def template_src(template: Path = TEMPLATE) -> str:
    return template.read_text(encoding="utf-8")


def template_js(template: Path = TEMPLATE) -> str:
    src = template_src(template)
    return src[src.index("const THREE_MODE"):]


def i18n(template: Path = TEMPLATE, lang: str = "en") -> dict[str, str]:
    """key -> string of one language block of ``const T = {en: {...}, zh: {...}}``."""
    src = template_src(template)
    t0 = src.index("const T = {")
    en0 = src.index("\nen: {", t0)
    zh0 = src.index("\nzh: {", en0)
    end = src.index("\n}};", zh0)
    block = src[en0:zh0] if lang == "en" else src[zh0:end]
    pat = re.compile(r'(?:^|[,{\s])([A-Za-z_][A-Za-z0-9_]*):\s*("(?:[^"\\]|\\.)*"|`(?:[^`\\]|\\.)*`)',
                     re.M)
    return {m.group(1): m.group(2)[1:-1] for m in pat.finditer(block)}


_LIT = re.compile(r"(?<![{A-Za-z_\d])[-+−]?\d+(?:[.,]\d+)?(?![}A-Za-z_\d])")


def numeric_keys(d: dict[str, str]) -> set[str]:
    """Keys whose string shows a number: a {placeholder} or a digit literal outside one."""
    return {k for k, s in d.items()
            if re.search(r"\{[A-Za-z0-9_]+\}", s) or _LIT.search(re.sub(r"\{[^}]*\}", "", s))}


def displayed_numeric_keys(template: Path = TEMPLATE) -> set[str]:
    """Numeric keys in EITHER language (a number shown only in Chinese counts too)."""
    en, zh = i18n(template, "en"), i18n(template, "zh")
    return numeric_keys(en) | {k for k in numeric_keys(zh) if k in en}


NUMFMT = re.compile(r"\b(f1|f2|sg|pctf|inv|r3)\(|\.toFixed\(|\.toExponential\(|\.toPrecision\("
                    r"|toLocaleString\(|Math\.round\(|\$\{[^}]*D\.[A-Za-z_]")
FORMATTERS = {"f1", "f2", "sg", "pctf", "inv", "r3", "rgbAt", "lut"}
_FN = re.compile(r"(?:async\s+)?function\s+([A-Za-z0-9_$]+)\s*\(|const\s+([A-Za-z0-9_$]+)\s*="
                 r"\s*(?:\([^)]*\)|[A-Za-z0-9_]+)\s*=>")


def _function_bodies(template: Path = TEMPLATE) -> dict[str, str]:
    lines = template_js(template).split("\n")
    starts = [(i, m.group(1) or m.group(2)) for i, ln in enumerate(lines) if (m := _FN.match(ln))]
    starts.append((len(lines), None))
    return {name: "\n".join(lines[a:b]) for (a, name), (b, _) in zip(starts, starts[1:], strict=False)}


def number_functions(template: Path = TEMPLATE) -> dict[str, dict]:
    """Top-level JS functions that format a number (or interpolate a payload value) -> the
    dictionary keys and payload paths they use."""
    out = {}
    for name, body in _function_bodies(template).items():
        if name in FORMATTERS or not NUMFMT.search(body):
            continue
        out[name] = {"tr_keys": sorted(tr_keys(body)),
                     "D_paths": sorted(set(re.findall(r"\bD\.([A-Za-z0-9_.]+)", body)))}
    return out


def _tr_first_args(body: str) -> list[str]:
    """The source text of the first argument of every ``tr(`` call in ``body``."""
    out = []
    for m in re.finditer(r"(?<![A-Za-z0-9_$.])tr\(", body):
        if body[:m.start()].endswith("function "):      # the translator's own definition
            continue
        j, depth, quote = m.end(), 0, None
        while j < len(body):
            ch = body[j]
            if quote:
                if ch == "\\":
                    j += 1
                elif ch == quote:
                    quote = None
            elif ch in "\"'`":
                quote = ch
            elif ch in "([{":
                depth += 1
            elif ch in ")]}":
                if depth == 0:
                    break
                depth -= 1
            elif ch == "," and depth == 0:
                break
            j += 1
        out.append(body[m.end():j])
    return out


_KEY_LIT = re.compile(r'"([A-Za-z_][A-Za-z0-9_]*)"')


def tr_keys(body: str) -> set[str]:
    """Every key literal in the first argument of a ``tr(`` call: a plain ``tr("k")`` and the
    branches of a chosen key, ``tr(c ? "a" : "b")``."""
    return {k for a in _tr_first_args(body) for k in _KEY_LIT.findall(a)}


def computed_tr_calls(body: str) -> list[str]:
    """``tr(`` first arguments whose key cannot be read from the source: a concatenation
    (``tr("ly_" + id)``) or no key literal at all (``tr(t)``). ``tr_keys`` cannot see these keys."""
    return [a.strip() for a in _tr_first_args(body) if "+" in a or not _KEY_LIT.search(a)]


def function_closure(name: str, template: Path = TEMPLATE) -> set[str]:
    """``name`` and every top-level function it calls, transitively (by name)."""
    bodies = _function_bodies(template)
    seen, todo = set(), [name]
    while todo:
        n = todo.pop()
        if n in seen or n not in bodies or n == "tr":    # tr itself: the lookup, no key
            continue
        seen.add(n)
        todo += [c for c in re.findall(r"(?<![A-Za-z0-9_$.])([A-Za-z_$][A-Za-z0-9_$]*)\s*\(",
                                        bodies[n]) if c in bodies and c not in seen]
    return seen


def function_keys(name: str, template: Path = TEMPLATE, follow: bool = False) -> set[str]:
    """Dictionary keys a top-level function asks for (``tr("key")`` and ternary-chosen keys);
    with ``follow``, also those of every top-level function it calls. Keys built by
    concatenation are not visible here: see ``computed_tr_calls``."""
    bodies = _function_bodies(template)
    names = function_closure(name, template) if follow else {name}
    return {k for n in names for k in tr_keys(bodies[n])}


def function_source(name: str, template: Path = TEMPLATE, follow: bool = False) -> str:
    """The source of a top-level function (and, with ``follow``, of everything it calls)."""
    bodies = _function_bodies(template)
    names = function_closure(name, template) if follow else {name}
    return "\n".join(bodies[n] for n in sorted(names))


def card_rows(template: Path = TEMPLATE) -> set[str]:
    """kv_* labels of the model-card detail rows (their numbers are inline, not in the dict)."""
    js = template_js(template)
    body = js[js.index("function renderCard()"):js.index("function build3D()")]
    keys = set(re.findall(r'add\((?:FIXED\(\)\s*\?\s*)?tr\("(kv_[a-z_]+)"\)', body))
    keys |= set(re.findall(r':\s*tr\("(kv_steps_[a-z_]+)"\)', body))
    return keys


def tiles(template: Path = TEMPLATE) -> tuple[list[str], int]:
    """(TILE_IDS, PRIMARY): the first PRIMARY tiles are the answer row, the rest model internals."""
    js = template_js(template)
    m = re.search(r"const TILE_IDS = \[([^\]]*)\], PRIMARY = (\d+);", js)
    if not m:
        raise ValueError("TILE_IDS / PRIMARY not found in the template")
    return re.findall(r'"([A-Za-z]+)"', m.group(1)), int(m.group(2))


def js_names_leaf(js: str, path: str) -> bool:
    """Is the leaf key of a payload path named anywhere in the page's JS?"""
    segs = [s for s in path.replace("[]", "").split(".") if s and not s.isdigit()]
    return bool(segs) and re.search(r"(?<![A-Za-z0-9_])" + re.escape(segs[-1])
                                    + r"(?![A-Za-z0-9_])", js) is not None


# ---- the registry (gitignored) ---------------------------------------------------------------
def prereg_index() -> tuple[dict[str, dict], set[str]]:
    ids, tests = {}, set()
    for ln in (PREREG / "INDEX.jsonl").read_text().splitlines():
        if ln.strip():
            r = json.loads(ln)
            ids[r["id"]] = r
            tests.update(r.get("test_ids", []))
    return ids, tests
