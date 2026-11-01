"""Bind the analyst to one built decision page and one claims ledger.

    bundle = PageBundle.load(Path("results/twin/twin_app_20261101a.html"))

Binding checks (spec §4.1), all fatal:

(a) the page's forward-run sha equals the ledger's ``page.forward_sha256``;
(b) the built page, with ``viewer_app._render``'s four substitutions reversed, hashes to the
    ledger's ``page.template_sha256`` — so the page's ``T`` strings and answer functions are
    exactly the ones the ledger tests checked;
(c) the ledger loads strictly (no ``PENDING`` registration) unless ``allow_pending`` in dev
    mode, which puts a banner on every answer;
(d) individual observations (``wells``/``leveling``) are never read.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from hydrophysics.twin import claims
from hydrophysics.twin.release import read_payload

REPO = Path(__file__).resolve().parents[1]
LEDGER = REPO / "hydrophysics" / "twin" / "claims_v1.json"
TEMPLATE = REPO / "hydrophysics" / "twin" / "app" / "template.html"
FORBIDDEN_ARRAYS = ("wells", "leveling")

_PAYLOAD_RE = re.compile(r'(<script id="payload" type="application/json">)(.*?)(</script>)', re.S)
_THREE_SRC_RE = re.compile(r'<script type="text/plain" id="three-src">.*?</script>', re.S)
_THREE_MODE_RE = re.compile(r'const THREE_MODE = "[^"]*", THREE_CDN = "[^"]*";')
_T_PAIR = re.compile(r'(?:^|[,{\s])([A-Za-z_][A-Za-z0-9_]*):\s*("(?:[^"\\]|\\.)*"|`(?:[^`\\]|\\.)*`)',
                     re.M)


class BindingError(RuntimeError):
    """The page and the ledger do not belong together, or a deploy-mode rule is broken."""


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def parse_T(text: str, lang: str) -> dict[str, str]:
    """``claims.i18n`` on a string (it takes a Path). Parses only after ``const THREE_MODE`` so an
    inlined three.js can never be mistaken for the dictionary."""
    i = text.find("const THREE_MODE")
    src = text[i:] if i >= 0 else text
    t0 = src.index("const T = {")
    en0 = src.index("\nen: {", t0)
    zh0 = src.index("\nzh: {", en0)
    end = src.index("\n}};", zh0)
    block = src[en0:zh0] if lang == "en" else src[zh0:end]
    out = {m.group(1): m.group(2)[1:-1] for m in _T_PAIR.finditer(block)}
    bad = [k for k, v in out.items() if "\\" in v]
    if bad:                                     # claims.i18n returns raw literals: no unescaping
        raise BindingError(f"T strings with escapes are not supported: {bad[:5]}")
    return out


def unrender(page_text: str) -> str:
    """Reverse ``viewer_app._render`` (viewer_app.py:1804-1824) to recover the template."""
    m = _PAYLOAD_RE.search(page_text)
    if not m:
        raise BindingError("no payload script in the page")
    s = page_text[:m.start(2)] + "__PAYLOAD__" + page_text[m.end(2):]
    close = s.index("</script>", s.index('<script id="payload"')) + len("</script>")
    ws = 1 if s[close:close + 1] == "\n" else 0          # template: "</script>\n<!--marker-->\n"
    head, rest = s[:close + ws], s[close + ws:]
    three = _THREE_SRC_RE.match(rest)
    rest = rest[three.end():] if three else rest
    s = head + "<!--__THREE_INLINE__-->" + rest
    s, n = _THREE_MODE_RE.subn('const THREE_MODE = "__THREE_MODE__", THREE_CDN = "__THREE_CDN__";',
                              s, count=1)
    if n != 1:
        raise BindingError("THREE_MODE line not found")
    return s


def unpack_f32(p: dict) -> np.ndarray:
    """The page's JS ``unpack`` (template.html:583-598): int arrays raw; scaled arrays as
    float32 of ``q * s`` computed in double; only the int16 minimum is NaN."""
    dt = {"i16": "int16", "u8": "uint8", "i32": "int32", "u16": "uint16"}[p["t"]]
    raw = gzip.decompress(base64.b64decode(p["z"]))
    width = np.dtype(dt).itemsize
    if width > 1:
        raw = np.frombuffer(raw, dtype="uint8").reshape(width, -1).T.copy().tobytes()
    q = np.frombuffer(raw, dtype=dt).reshape(-1)
    if p.get("s") is None:
        return q.copy()
    out = (q.astype(np.float64) * float(p["s"])).astype(np.float32)
    if dt == "int16":
        out[q == np.iinfo(np.int16).min] = np.nan
    return out


@dataclass
class PageBundle:
    payload: dict
    T: dict[str, dict[str, str]]
    ledger: dict
    page_path: Path
    page_sha256: str
    ledger_sha256: str
    strings_sha256: str
    mode: Literal["dev", "deploy"] = "dev"
    banners: list[str] = field(default_factory=list)
    _arrays: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    # ---- construction ---------------------------------------------------------------
    @classmethod
    def load(cls, page_html: Path, ledger_json: Path | None = None, *,
             mode: Literal["dev", "deploy"] = "dev", allow_pending: bool = False,
             check_forward: bool = True, check_template: bool = True) -> PageBundle:
        page_html, ledger_json = Path(page_html), Path(ledger_json or LEDGER)
        if mode == "deploy" and (allow_pending or not check_forward or not check_template):
            raise BindingError("deploy mode runs every binding check on a strict ledger")
        banners: list[str] = []
        try:
            ledger = claims.load_ledger(ledger_json, allow_pending=allow_pending)
        except claims.PendingRegistrationError as e:
            raise BindingError(str(e)) from e
        if claims.pending_registrations(ledger):
            banners.append("dev_pending")
        text = page_html.read_text(encoding="utf-8")
        payload = read_payload(page_html)
        if check_forward:                                   # (a)
            fwd = (((payload.get("release") or {}).get("sources") or {}).get("forward") or {})
            want = ledger.get("page", {}).get("forward_sha256")
            if fwd.get("sha256") != want:
                raise BindingError(f"forward run {fwd.get('sha256')} is not the ledger's {want}")
        if check_template:                                  # (b)
            got = sha256_bytes(unrender(text).encode("utf-8"))
            want = ledger.get("page", {}).get("template_sha256")
            if got != want:
                raise BindingError(f"page template {got[:12]} is not the ledger's {str(want)[:12]}")
        T = {"en": parse_T(text, "en"), "zh": parse_T(text, "zh")}
        strings = json.dumps(T, sort_keys=True, ensure_ascii=False).encode("utf-8")
        if not check_forward or not check_template:
            banners.append("dev_unbound")
        return cls(payload=payload, T=T, ledger=ledger, page_path=page_html,
                   page_sha256=sha256_bytes(text.encode("utf-8")),
                   ledger_sha256=claims.sha256(ledger_json),
                   strings_sha256=sha256_bytes(strings), mode=mode, banners=banners)

    # ---- access ---------------------------------------------------------------------
    def array(self, name: str) -> np.ndarray:
        if name in FORBIDDEN_ARRAYS or name.startswith(FORBIDDEN_ARRAYS):
            raise BindingError(f"individual observations ({name}) are out of scope")
        if name not in self._arrays:
            p = (self.payload.get("arrays") or {}).get(name)
            if p is None:
                raise KeyError(name)
            self._arrays[name] = unpack_f32(p)
        return self._arrays[name]

    def has_array(self, name: str) -> bool:
        return name in (self.payload.get("arrays") or {})

    def __getitem__(self, key: str):
        if key in FORBIDDEN_ARRAYS:
            raise BindingError(f"individual observations ({key}) are out of scope")
        return self.payload[key]

    def get(self, key: str, default=None):
        if key in FORBIDDEN_ARRAYS:
            raise BindingError(f"individual observations ({key}) are out of scope")
        return self.payload.get(key, default)

    @property
    def claims_by_id(self) -> dict[str, dict]:
        return {c["id"]: c for c in self.ledger["claims"]}

    @property
    def meta(self) -> dict:
        return self.payload["meta"]
