"""Deterministic query layer: a line-by-line numpy port of the decision page's runtime math.

Every public function returns a :class:`QueryResult` whose numbers are formatted exactly as
the page formats them, so an answer can be string-identical to what the page shows at the
deep-linked state. Source functions (hydrophysics/twin/app/template.html): solvedMatch,
policyInto, kpis, headWhere, policyPhrase, renderHeadline, renderTiles (answer row),
renderRank, renderTowns (the internals row), railLine / renderHsr note, renderStartNote,
hashWrite.

Arithmetic rules that make the port bit-exact (spec §5):
- the page's Float32Array accumulations are emulated as double math then a float32 store;
- KPI sums are float64 in the JS sequential order (``np.cumsum``, never pairwise ``np.sum``);
- formatting goes through :mod:`hydrophysics_analyst.jsport`.

Withdrawn (W) quantities — fan-average avoided subsidence, the area tile, subsidence bands, the
cm ranking — are computed only inside :class:`Internal` and never leave this module through a
public result (spec P4).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from typing import Literal
from urllib.parse import urlencode

import numpy as np

from . import jsport as J
from .page import PageBundle

F32, F64 = np.float32, np.float64
HSR_THR = 1000          # S.hsrThr default; not in the URL hash, so fixed (spec §5)
AREA_THR = 1            # S.thr default; feeds the withdrawn area tile only


# ---------------------------------------------------------------------------------------
# result types
# ---------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Policy:
    """``f[k]`` is the page's factor for class k (1 = unchanged, 0 = retired), as the URL hash
    carries it; ``start`` is a calendar year in ``meta.starts``."""
    f: tuple[float, ...]
    start: int

    @classmethod
    def from_cuts(cls, cuts: dict[int, int] | list[int], start: int, K: int = 6) -> Policy:
        """Cuts in percent per class (5 % grid). The factor is the hash-rounded value
        ``+(1 - c/100).toFixed(2)`` (hashWrite), which differs from ``1 - c/100`` by one ulp
        for some cuts; answers claim parity with the linked state, so the hash value is used."""
        c = dict(enumerate(cuts)) if isinstance(cuts, list) else dict(cuts)
        f = tuple(float(J.to_fixed(1 - c.get(k, 0) / 100, 2)) for k in range(K))
        return cls(f=f, start=int(start))

    def is_base(self) -> bool:
        return all(v == 1 for v in self.f)


@dataclass(frozen=True)
class Value:
    key: str
    value: float | int | None
    text: str
    unit: str | None
    paths: tuple[str, ...] = ()
    functions: tuple[str, ...] = ()
    quantity: str = ""
    region: str = ""
    aggregation: str = ""
    claims: tuple[str, ...] = ()
    status: tuple[tuple[str, str], ...] = ()
    answer_class: str = "context"


@dataclass(frozen=True)
class Snippet:
    key: str                        # i18n key (or "+"-joined keys for a composed page sentence)
    lang: str
    text: str                       # rendered with the page's tr(), may contain <b>
    values: tuple[str, ...] = ()    # Value.key of the numbers this sentence carries
    claims: tuple[str, ...] = ()


@dataclass(frozen=True)
class Withheld:
    claim: str
    reason: str


@dataclass(frozen=True)
class QueryResult:
    kind: str
    values: tuple[Value, ...]
    snippets: tuple[Snippet, ...]
    src: str
    policy: Policy | None
    year: int | None
    deep_link: str
    page_sha256: str
    ledger_sha256: str
    withheld: tuple[Withheld, ...] = ()
    table: tuple[tuple[str, ...], ...] = ()

    def value(self, key: str) -> Value:
        return next(v for v in self.values if v.key == key)


@dataclass
class PolicyFields:
    f: tuple[float, ...]
    si: int
    dS: np.ndarray
    dH: np.ndarray
    dW: np.ndarray
    hasW: bool
    src: Literal["base", "solved", "basis", "none"]
    solved: int
    w30: bool = False              # ruling R8.9: dW from the all30 full-model run (n2Unit.all30)


@dataclass
class Internal:
    """Everything kpis computes, including the withdrawn quantities. Never returned publicly."""
    kp: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------------------
# context: the page's derived constants (template ``M, A, Y, YR, K`` and ``derive()``)
# ---------------------------------------------------------------------------------------
class Ctx:
    def __init__(self, b: PageBundle):
        self.b = b
        M = b.meta
        self.M = M
        self.A = int(M["nA"])
        self.years = [int(y) for y in M["years"]]
        self.Y = len(self.years)
        self.YO = int(M["yObs"])
        self.YR = int(M["yRef"] if M.get("yRef") is not None else M["yObs"])
        self.starts = [int(s) for s in M["starts"]]
        self.classes = b["classes"]
        self.K = len(self.classes)
        self.towns = b["towns"]
        self.solved = b["solved"] or []
        self.n2u = b.get("n2Unit")
        self.mc = b["modelcard"]
        self.town = b.array("town").astype(np.int64)
        self.subsBase = b.array("subsBase")
        self.has_basis = b.has_array("dSubs")
        self._pooled = np.array([bool(t.get("pooled")) for t in self.towns])

    # helpers mirroring the page
    def nochp(self) -> bool:
        p = self.mc.get("pumping")
        return bool(p) and p.get("changhua") is False

    def yidx(self, year: int) -> int:
        return self.years.index(int(year))

    def si(self, start: int) -> int:
        return self.starts.index(int(start))

    def CN(self, k: int, lang: str) -> str:
        return self.classes[k]["zh" if lang == "zh" else "en"]

    def TN(self, i: int, lang: str) -> str:
        return self.towns[i]["zh" if lang == "zh" else "en"]

    def tr(self, lang: str, key: str, v: dict | None = None) -> str:
        return J.tr(self.b.T, lang, key, v)


_CTX: dict[int, Ctx] = {}


def ctx(b: PageBundle) -> Ctx:
    c = _CTX.get(id(b))
    if c is None or c.b is not b:
        c = _CTX[id(b)] = Ctx(b)
    return c


# ---------------------------------------------------------------------------------------
# policy fields (policyInto)
# ---------------------------------------------------------------------------------------
def _acc(acc: np.ndarray, w: float, row) -> np.ndarray:
    """JS ``acc[i] += w * row[i]`` into a Float32Array: double math, float32 store."""
    return (acc.astype(F64) + w * np.asarray(row, dtype=F64)).astype(F32)


def _jsum(x: np.ndarray) -> float:
    """JS sequential double sum (``for ... s += v``), not numpy's pairwise sum."""
    x = np.asarray(x, dtype=F64)
    return float(np.cumsum(x)[-1]) if x.size else 0.0


def solved_match(c: Ctx, f, start_year: int) -> int:
    for i, s in enumerate(c.solved):
        if int(s["start"]) == int(start_year) and all(
                abs(v - f[k]) < 1e-6 for k, v in enumerate(s["factors"])):
            return i
    return -1


def w30(c: Ctx) -> dict | None:
    """``W30()``: the headline lever's N2-unit block (ruling R8.9), when the page carries it."""
    return (c.n2u or {}).get("all30") or None


def is_w30(c: Ctx, f, start_year: int) -> bool:
    """``isW30(f, startYear)``: the policy is the all30 run (every class x0.7, its start)."""
    w = w30(c)
    return bool(w) and int(w["start"]) == int(start_year) and all(
        abs(v - f[k]) < 1e-6 for k, v in enumerate(w["factors"]))


def policy_fields(b: PageBundle, p: Policy) -> PolicyFields:
    c = ctx(b)
    si = c.si(p.start)
    n = c.A * c.Y
    out = PolicyFields(f=tuple(p.f), si=si, dS=np.zeros(n, F32), dH=np.zeros(n, F32),
                       dW=np.zeros(c.Y, F32), hasW=False, src="base", solved=-1)
    if p.is_base():
        return out
    m, u = solved_match(c, p.f, c.starts[si]), c.n2u
    if m >= 0 and b.has_array(f"solvedSubs{m}"):
        out.dS = b.array(f"solvedSubs{m}").astype(F32).copy()
        out.dH = b.array(f"solvedHead{m}").astype(F32).copy()
        out.src, out.solved = "solved", m
        if u and u.get("solved") and m < len(u["solved"]) and u["solved"][m]:
            out.dW = np.asarray(u["solved"][m], dtype=F64).astype(F32)
            out.hasW = True
        return out
    if not c.has_basis:
        out.src = "none"
        return out
    bS, bH = b.array("dSubs"), b.array("dHeadL2")
    for k in range(c.K):
        w = 1 - p.f[k]
        if not w:
            continue
        off = (si * c.K + k) * n
        out.dS = _acc(out.dS, w, bS[off:off + n])
        out.dH = _acc(out.dH, w, bH[off:off + n])
        if u and u.get("basis"):
            out.dW = _acc(out.dW, w, u["basis"][si * c.K + k])
    out.hasW = bool(u and u.get("basis"))
    out.src = "basis"
    if is_w30(c, p.f, c.starts[si]):            # R8.9: its own full-model run
        out.dW = np.asarray(w30(c)["series"], dtype=F64).astype(F32)
        out.hasW = out.w30 = True
    return out


def _dfwd(c: Ctx, pol: PolicyFields, y: int) -> np.ndarray:
    """``dFwd(pol, c, y)`` for every cell: float32 reads subtracted in double."""
    d = pol.dS.reshape(c.A, c.Y)
    return d[:, y].astype(F64) - d[:, c.YR].astype(F64)


# ---------------------------------------------------------------------------------------
# rail (hsrProfile, angDist)
# ---------------------------------------------------------------------------------------
def _hsr_profile(c: Ctx, field: np.ndarray) -> list[float]:
    H = c.b["hsr"]
    out = []
    for n, ix in enumerate(H["idx"]):
        s = 0.0
        for q in range(4):
            if ix[q] >= 0:
                s += H["w"][n][q] * float(field[ix[q]])
        out.append(s)
    return out


def _ang_dist(c: Ctx, prof: list[float]) -> list[float]:
    H = c.b["hsr"]
    st, n = float(H["step"]), len(prof)
    k = max(1, int(J.js_round(2 / st)))
    s = [v / 100 for v in prof]
    if k > 1 and n > k:
        pad = k // 2
        padded = [s[max(0, min(n - 1, i))] for i in range(-pad, n + k - 1 - pad)]
        s2 = []
        for i in range(n):
            a = 0.0
            for q in range(k):
                a += padded[i + q]
            s2.append(a / k)
        s = s2
    h = max(1, int(J.js_round(1 / st / 2)))
    out = []
    for i in range(n):
        lo, hi = max(0, i - h), min(n - 1, i + h)
        out.append(abs(s[hi] - s[lo]) / max((hi - lo) * st * 1000, 1e-9))
    return out


def _fwd_field(c: Ctx, pol: PolicyFields | None, y: int) -> np.ndarray:
    sb = c.subsBase.reshape(c.A, c.Y)
    v = sb[:, y].astype(F64) - sb[:, c.YR].astype(F64)
    if pol is not None:
        v = v + _dfwd(c, pol, y)
    return v.astype(F32)                 # FWDBUF is a Float32Array


# ---------------------------------------------------------------------------------------
# kpis (answer subset + internals)
# ---------------------------------------------------------------------------------------
def _single(f) -> int:
    cut = [k for k, v in enumerate(f) if v < 1 - 1e-9]
    return cut[0] if len(cut) == 1 else -1


def _solo_run(c: Ctx, k: int, start_year: int | None) -> int:
    for i, x in enumerate(c.solved):
        if x.get("members") and _single(x["factors"]) == k and (
                start_year is None or int(x["start"]) == int(start_year)):
            return i
    return -1


def kpis(b: PageBundle, pol: PolicyFields, y: int, *, hsr_thr: int = HSR_THR,
         area_thr: int = AREA_THR) -> Internal:
    """The page's ``kpis(pol, y)`` (template.html ~1435). ``y`` is a year INDEX."""
    c = ctx(b)
    A, Y = c.A, c.Y
    kp: dict = {"y": y}
    atH = y == Y - 1
    d = -_dfwd(c, pol, y)
    dh = pol.dH.reshape(A, Y)[:, y].astype(F64)
    yun = ~c._pooled[c.town]
    kp["yunN"] = int(yun.sum())
    kp["yunAvoid"] = _jsum(d[yun]) / kp["yunN"] if kp["yunN"] else None
    kp["yunHead"] = _jsum(dh[yun]) / kp["yunN"] if kp["yunN"] else None
    kp["n2Head"] = float(pol.dW[y]) if pol.hasW else None
    kp["fieldAvoid"] = _jsum(d) / A
    kp["fieldHead"] = _jsum(dh) / A
    # first strict maximum, like the JS loop (headMax starts at -1e9)
    kp["headMax"] = float(dh.max()) if A and dh.max() > -1e9 else -1e9
    kp["avoid"], kp["head"] = kp["fieldAvoid"], kp["fieldHead"]
    kp["agree"], kp["agreeLB"], kp["noRun"] = None, False, []
    kp["nRuns"] = kp["nAll"] = c.mc.get("n_members")

    def sets_of(m):
        return m.get("n_sets", m.get("n"))

    def agree_of(m):
        return m.get("agree_sets", m.get("agree"))

    sx0 = c.solved[pol.solved] if pol.src == "solved" else None
    if pol.src == "solved":
        m = sx0.get("members")
        if m:
            if atH:
                kp["avoid"] = -(m["subs_mean"] if m.get("subs_mean") is not None else m["subs_p"][1])
                kp["head"] = m["head_mean"] if m.get("head_mean") is not None else m["head_p"][1]
            kp["agree"], kp["nRuns"], kp["nAll"] = agree_of(m), sets_of(m), m["n"]
    elif pol.src == "basis":
        start_year = c.starts[pol.si]
        cut = [k for k in range(c.K) if 1 - pol.f[k]]
        runs = [_solo_run(c, k, start_year) for k in cut]
        kp["noRun"] = [k for k, r in zip(cut, runs, strict=True) if r < 0]
        if cut and not kp["noRun"]:
            ms = [c.solved[i]["members"] for i in runs]
            n = min(sets_of(m) for m in ms)
            kp["agree"] = max(0, sum(agree_of(m) for m in ms) - (len(ms) - 1) * n)
            kp["nRuns"], kp["nAll"], kp["agreeLB"] = n, min(m["n"] for m in ms), len(ms) > 1
    # area above the rate threshold (W: C05) — internals only
    yp = max(0, y - 1)
    sb = c.subsBase.reshape(A, Y).astype(F64)
    rateB = b.array("rateBase").astype(F64) if (atH and b.has_array("rateBase")) else None
    rb = rateB if rateB is not None else sb[:, y] - sb[:, yp]
    dS = pol.dS.reshape(A, Y).astype(F64)
    rp = rb + dS[:, y] - dS[:, yp]
    kp["areaBase"], kp["areaPol"] = int((rb > area_thr).sum()), int((rp > area_thr).sum())
    if b.get("hsr"):
        H = b["hsr"]
        pb = _hsr_profile(c, _fwd_field(c, None, y))
        pp = _hsr_profile(c, _fwd_field(c, pol, y))
        ab, ap, th = _ang_dist(c, pb), _ang_dist(c, pp), 1 / hsr_thr
        mixed = H.get("mixed") or []

        def okp(i):
            return not (mixed and mixed[i])

        def top(a):
            best, j = 0.0, -1
            for i, v in enumerate(a):
                if okp(i) and v > best:
                    best, j = v, i
            return best, j
        (bB, iB), (bP, iP) = top(ab), top(ap)
        kp.update(hsrBase=bB, hsrPol=bP,
                  hsrAtB=H["ch"][iB] if iB >= 0 else math.nan,
                  hsrAtP=H["ch"][iP] if iP >= 0 else math.nan,
                  hsrRawB=max(ab), hsrRawP=max(ap),
                  hsrExcl=sum(1 for v in mixed if v) * H["step"],
                  hsrSegB=sum(1 for i, v in enumerate(ab) if okp(i) and v > th) * H["step"],
                  hsrSegP=sum(1 for i, v in enumerate(ap) if okp(i) and v > th) * H["step"])
    gw = [cl.get("gwh") for cl in c.classes]
    tot = sum(g or 0 for g in gw)
    if gw[0] is None:
        kp["gwh"] = kp["gwhPct"] = None
    else:
        s = 0.0
        for k, g in enumerate(gw):
            s = s + (1 - pol.f[k]) * g
        kp["gwh"] = s
        kp["gwhPct"] = 100 * s / tot
    nT = len(c.towns)
    ts, tb, tn = np.zeros(nT, F64), np.zeros(nT, F64), np.zeros(nT, np.int64)
    bfw = sb[:, y] - sb[:, c.YR]
    for cell in range(A):                         # ts[t] -= dFwd, sequential in cell order
        t = c.town[cell]
        ts[t] += d[cell]
        tb[t] += bfw[cell]
        tn[t] += 1
    kp["town"] = [ts[i] / tn[i] if tn[i] else math.nan for i in range(nT)]
    kp["townBase"] = [tb[i] / tn[i] if tn[i] else math.nan for i in range(nT)]
    kp["src"] = pol.src
    return Internal(kp=kp)


# ---------------------------------------------------------------------------------------
# words
# ---------------------------------------------------------------------------------------
def _join_list(c: Ctx, lang: str, a: list[str]) -> str:
    if len(a) > 1:
        return c.tr(lang, "list_sep").join(a[:-1]) + c.tr(lang, "list_and") + a[-1]
    return a[0] if a else "–"


def policy_phrase(b: PageBundle, f, lang: str) -> str:
    c = ctx(b)
    cuts = [(k, 1 - v) for k, v in enumerate(f) if 1 - v > 1e-6]
    if not cuts:
        return c.tr(lang, "pp_bau")
    if len(cuts) == c.K and all(abs(x - cuts[0][1]) < 1e-6 for _, x in cuts):
        return (c.tr(lang, "pp_all0") if cuts[0][1] >= .999 else
                c.tr(lang, "pp_allcut", {"p": int(J.js_round(cuts[0][1] * 100))}))

    def one(kc):
        k, x = kc
        if x >= .999:
            return c.tr(lang, "pp_ret", {"c": J.lc(c.CN(k, lang))})
        return c.tr(lang, "pp_cut", {"c": J.lc(c.CN(k, lang)), "p": int(J.js_round(x * 100))})
    return J.uc(_join_list(c, lang, [one(x) for x in cuts]))


def class_list(b: PageBundle, ks: list[int], lang: str) -> str:
    c = ctx(b)
    return _join_list(c, lang, [J.lc(c.CN(k, lang)) for k in ks])


def head_where(b: PageBundle, kp: dict, lang: str) -> dict:
    c = ctx(b)
    if c.n2u and kp.get("n2Head") is not None:
        return {"v": kp["n2Head"], "key": "hw_n2", "n2": True,
                "where": c.tr(lang, "hw_n2", {"n": c.n2u["n"]})}
    if c.nochp() and kp.get("yunHead") is not None and kp.get("yunN"):
        return {"v": kp["yunHead"], "key": "hw_yun", "n2": False,
                "where": c.tr(lang, "hw_yun", {"n": J.locale_en(kp["yunN"])})}
    return {"v": kp["fieldHead"], "key": "hw_fan", "n2": False, "where": c.tr(lang, "hw_fan")}


def boff_words(b: PageBundle, lang: str) -> str:
    c = ctx(b)
    bo = b.get("basisOff")
    if not bo:
        return ""
    x = next((s for s in c.solved if s["name"] == bo["worst"]), None)
    nm = policy_phrase(b, x["factors"], lang) if x else bo["worst"]
    esc = (str(bo["source"]).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
           .replace('"', "&quot;"))
    return c.tr(lang, "boff", {"src": esc, "nm": J.lc(nm), "rel": int(J.js_round(100 * bo["rel"])),
                               "max": int(J.js_round(100 * bo["max_rel"]))})


def deep_link(b: PageBundle, p: Policy, year: int, *, layer: str = "dhead", lang: str = "en",
              pins: tuple = (), nonce: int | None = None, sel: int | None = None,
              tab: str | None = None, mi: bool = False) -> str:
    """The page's ``hashWrite`` (template.html:636): ``#p=..&s=..&y=..&l=..&lang=..``."""
    q = [("p", ",".join(J.js_num_str(float(J.to_fixed(v, 2))) for v in p.f)),
         ("s", str(p.start)), ("y", str(year)), ("l", layer), ("lang", lang)]
    if pins:
        q.append(("pins", "|".join(f"{nm}~{','.join(J.js_num_str(v) for v in f)}~{st}"
                                   for nm, f, st in pins)))
    # view keys (spec D5): pages without them ignore unknown keys, so links degrade gracefully
    if sel is not None:
        q.append(("sel", f"t:{int(sel)}"))
    if tab and tab != "towns":
        q.append(("tab", tab))
    if mi:
        q.append(("mi", "1"))
    if nonce is not None:
        q.append(("n", str(nonce)))
    return "#" + urlencode(q, safe=",~|:")


def hash_read(b: PageBundle, h: str) -> dict:
    """The page's ``hashRead`` subset: policy, start, year, layer, lang."""
    from urllib.parse import parse_qs
    c = ctx(b)
    q = {k: v[0] for k, v in parse_qs(h.lstrip("#")).items()}
    out: dict = {}
    if q.get("p"):
        v = [float(x) for x in q["p"].split(",")]
        if len(v) == c.K and all(0 <= x <= 1 for x in v):
            out["f"] = tuple(v)
    if q.get("s") and int(q["s"]) in c.starts:
        out["start"] = int(q["s"])
    if q.get("y") and int(q["y"]) in c.years:
        out["year"] = int(q["y"])
    out["lang"] = "zh" if q.get("lang") == "zh" else "en"
    for k in ("sel", "tab", "mi"):
        if q.get(k):
            out[k] = q[k]
    return out


# ---------------------------------------------------------------------------------------
# public queries
# ---------------------------------------------------------------------------------------
def _result(b: PageBundle, kind: str, values, snippets, src, p, year, lang, *, withheld=(),
            table=(), layer="dhead") -> QueryResult:
    link = deep_link(b, p, year, layer=layer, lang=lang) if p is not None and year else ""
    return QueryResult(kind=kind, values=tuple(values), snippets=tuple(snippets), src=src,
                       policy=p, year=year, deep_link=link, page_sha256=b.page_sha256,
                       ledger_sha256=b.ledger_sha256, withheld=tuple(withheld), table=tuple(table))


def _prov(b: PageBundle, pol: PolicyFields, kp: dict, lang: str) -> tuple[str, list[str]]:
    c = ctx(b)
    if pol.src == "solved":
        return (c.tr(lang, "prov_full", {"n": kp["nRuns"], "k": int(J.js_round(kp["nAll"] / kp["nRuns"]))}),
                ["prov_full"])
    if pol.src == "basis":
        be = b.get("basisError") or {}
        nl = next((e for e in be.values() if e.get("kind") == "nonlinear"), None)
        cal = b.get("calib")
        assumed = ([k for k, v in enumerate(pol.f) if v < 1 and cal["source"][k] == "assumed"]
                   if cal else [])
        s = c.tr(lang, "prov_fast")
        keys = ["prov_fast"]
        if nl:
            s += c.tr(lang, "prov_nl", {"p": int(J.js_round(nl["rel"] * 100))})
            keys.append("prov_nl")
        if assumed:
            s += c.tr(lang, "prov_assumed", {"c": class_list(b, assumed, lang)})
            keys.append("prov_assumed")
        if pol.w30:
            s += c.tr(lang, "prov_w30")
            keys.append("prov_w30")
        return s, keys
    if pol.src == "none":
        return c.tr(lang, "prov_none"), ["prov_none"]
    return c.tr(lang, "prov_base", {"n": c.mc.get("n_members")}), ["prov_base"]


IRR30 = (0.7, 1.0, 1.0, 1.0, 1.0, 1.0)
IRR30_FROM = 2026          # x1.27 was measured for the irrigation x0.7 lever from 2026-01 only


def _irr30m(f, yr) -> bool:
    return _near(f, IRR30) and int(yr) == IRR30_FROM


def _all30_2026(pol, f, yr) -> bool:
    """``renderTiles``' all30 range line: the headline run, or all30 from 2026 on the fast
    estimate."""
    return bool(pol.w30) or (_near(f, (0.7,) * len(f)) and int(yr) == 2026)


def _near(f, g) -> bool:
    return all(abs(a - b) < 1e-6 for a, b in zip(f, g, strict=True))


def _uniform_cut(f) -> bool:
    return not all(v == 1 for v in f) and all(abs(v - f[0]) < 1e-6 for v in f)


def hl_size(c: Ctx, pol: PolicyFields, f, yr: int, lang: str) -> tuple[str, str]:
    """``hlSize(f, yr)``: the headline's size clause per policy (R8.9; lead decisions a, b: the
    x1.27 factor belongs to the irrigation x0.7 lever only, x1.03-1.05 to all30 only)."""
    if pol.w30:
        k = "hl_sz_w30"
    elif _near(f, IRR30):
        k = "hl_sz_irr" if _irr30m(f, yr) else "hl_sz_cls"
    elif not _uniform_cut(f):
        k = "hl_sz_cls"
    elif _near(f, (0.7,) * len(f)) and int(yr) == 2026:
        k = "hl_sz_all"
    else:
        return "", ""
    return k, c.tr(lang, k)


def tile_w(b: PageBundle, p: Policy, year: int, lang: str = "en") -> tuple[str, str, str]:
    """``renderTiles`` for tile tW: (range-line key, range line, detail line)."""
    c = ctx(b)
    y = c.yidx(year)
    pol = policy_fields(b, p)
    if pol.src == "none":
        return "no_run", c.tr(lang, "no_run"), ""
    kp = kpis(b, pol, y).kp
    isB = pol.src == "base"
    w = w30(c)
    if isB:
        rk = "tW_w30" if w else ""
        r = c.tr(lang, "tW_w30", {"yr": w["start"], "h": J.sg(w["series"][c.Y - 1]),
                                   "H": c.years[c.Y - 1]}) if w else ""
        pre = ""
    else:
        sy = c.starts[pol.si]
        rk = "tW_r_w30" if _all30_2026(pol, p.f, sy) else "tW_r"
        r = c.tr(lang, rk)
        pk = ("tW_w30_src" if pol.w30 else "tW_cls_irr" if _irr30m(p.f, sy) else
              "" if _uniform_cut(p.f) else "tW_cls")
        pre = c.tr(lang, pk) if pk else ""
    d = pre + c.tr(lang, "tW_d", {"year": c.years[y],
                                  "y": "–" if kp.get("yunHead") is None else J.sg(kp["yunHead"]),
                                  "f": J.sg(kp["head"]), "v": J.sg(kp["headMax"], 1)})
    return rk, r, d


def w30_sentence(b: PageBundle, lang: str) -> str:
    """``w30Sentence()``: the opening headline names the headline lever (ruling R8.9)."""
    c = ctx(b)
    w = w30(c)
    if not w:
        return ""
    return c.tr(lang, "hl_w30", {"yr": w["start"], "h": J.sg(w["series"][c.Y - 1]),
                                 "where": c.tr(lang, "hw_n2", {"n": c.n2u["n"]}),
                                 "H": c.years[c.Y - 1]})


def headline(b: PageBundle, p: Policy, year: int, lang: str = "en") -> QueryResult:
    """``renderHeadline`` + ``headWhere`` + the provenance chip + tiles tW and tC."""
    c = ctx(b)
    y = c.yidx(year)
    pol = policy_fields(b, p)
    kp = kpis(b, pol, y).kp
    Hy, yr, y0 = c.years[y], p.start, c.years[c.YR]
    rec = f"{c.years[0]}–{c.years[c.YO]}"
    values: list[Value] = []
    snippets: list[Snippet] = []
    pol_words = J.lc(policy_phrase(b, p.f, lang))
    if y <= c.YR:
        key, s = "hl_fit", c.tr(lang, "hl_fit", {"year": Hy, "y0": y0, "a": c.years[0], "b": c.years[c.YO]})
    elif pol.src == "base":
        key, s = "hl_base", c.tr(lang, "hl_base", {"rec": rec, "y0": y0, "year": Hy})
        ws = w30_sentence(b, lang)
        if ws:
            key, s = "hl_base+hl_w30", s + ws
    elif pol.src == "none":
        opts = c.tr(lang, "list_or").join(
            c.tr(lang, "hl_opt", {"p": J.lc(policy_phrase(b, x["factors"], lang)), "s": x["start"]})
            for x in c.solved)
        boff = boff_words(b, lang) if b.get("basisOff") else c.tr(lang, "no_basis")
        key, s = "hl_none", c.tr(lang, "hl_none", {"pol": pol_words, "yr": yr, "boff": boff, "opts": opts})
    elif y < (c.years.index(p.start) if p.start in c.years else -1):   # startYearIdx()
        key, s = "hl_early", c.tr(lang, "hl_early", {"pol": pol_words, "yr": yr, "year": Hy})
    else:
        hw = head_where(b, kp, lang)
        txt = J.sg(hw["v"])
        values.append(Value("head", hw["v"], txt, "m",
                            paths=((("n2Unit.all30.series[]",) if pol.w30 else
                                    ("n2Unit.solved[][]", "n2Unit.basis[][]")) if hw["n2"]
                                   else ("arrays.dHeadL2",)),
                            functions=("headWhere",), quantity="head_change",
                            region="n2_yunlin_wells" if hw["n2"] else
                            ("yunlin_l2" if hw["key"] == "hw_yun" else "fan_l2"),
                            aggregation="mean_own_layer" if hw["n2"] else "cell_mean"))
        _, sz = hl_size(c, pol, p.f, yr, lang)       # its claims are hl_pol's (C01-sz)
        key = "hl_pol"
        s = c.tr(lang, "hl_pol", {"pol": pol_words, "yr": yr, "h": txt, "where": hw["where"],
                                  "year": Hy, "sz": sz})
    snippets.append(Snippet(key, lang, s, tuple(v.key for v in values)))
    prov, pkeys = _prov(b, pol, kp, lang)
    snippets.append(Snippet("+".join(pkeys), lang, prov))
    # answer-row tiles (renderTiles: tW, tC)
    isB = pol.src in ("base", "none")
    if pol.src != "none":
        hw = head_where(b, kp, lang)
        tw = "0.00" if isB else J.sg(hw["v"])
        if not values:
            values.append(Value("head", 0.0 if isB else hw["v"], tw, "m", paths=("n2Unit.solved[][]",),
                                functions=("headWhere",), quantity="head_change",
                                region="n2_yunlin_wells" if hw["n2"] else "fan_l2",
                                aggregation="mean_own_layer" if hw["n2"] else "cell_mean"))
        sy = c.starts[pol.si]
        if not isB:
            rk = "tW_r_w30" if _all30_2026(pol, p.f, sy) else "tW_r"
            snippets.append(Snippet(rk, lang, c.tr(lang, rk)))
        pk = ("tW_w30_src" if pol.w30 else "tW_cls_irr" if _irr30m(p.f, sy) else
              "" if _uniform_cut(p.f) else "tW_cls") if not isB else ("tW_w30" if w30(c) else "")
        if pk:
            w = w30(c)
            snippets.append(Snippet(pk, lang, c.tr(lang, pk, {"yr": w["start"], "h": J.sg(
                w["series"][c.Y - 1]), "H": c.years[c.Y - 1]} if pk == "tW_w30" else None)))
        gwh = kp["gwh"]
        if gwh is not None:
            values.append(Value("gwh", gwh, J.f1(gwh), "GWh/yr", paths=("classes[].gwh",),
                                functions=("renderTiles",), quantity="energy_forgone",
                                region="policy", aggregation="sum_classes"))
            pct = J.pctf(kp["gwhPct"])
            values.append(Value("gwh_pct", kp["gwhPct"], pct, "%", paths=("classes[].gwh",),
                                functions=("renderTiles",), quantity="energy_forgone",
                                region="policy", aggregation="sum_classes"))
            snippets.append(Snippet("tCost", lang, c.tr(lang, "tCost")))
            snippets.append(Snippet("tC_r", lang, c.tr(lang, "tC_r", {"p": pct}), ("gwh_pct",)))
            nyP = Hy - yr + 1
            if isB:
                snippets.append(Snippet("tC_metered", lang, c.tr(lang, "tC_metered")))
            elif nyP > 0:
                tot = int(J.js_round(gwh * nyP))
                values.append(Value("gwh_total", tot, J.js_num_str(tot), "GWh", paths=("classes[].gwh",),
                                    functions=("renderTiles",), quantity="energy_forgone",
                                    region="policy", aggregation="sum_classes_years"))
                snippets.append(Snippet("tC_d", lang, c.tr(lang, "tC_d", {"yr": yr, "year": Hy, "n": nyP,
                                                                         "tot": tot}), ("gwh_total",)))
            else:
                snippets.append(Snippet("tC_d0", lang, c.tr(lang, "tC_d0", {"yr": yr})))
        else:
            snippets.append(Snippet("no_energy", lang, c.tr(lang, "no_energy")))
    if pol.hasW and pol.src in ("solved", "basis"):
        values += _head_series(c, pol, "head_y")         # the chart's points: the same N2-unit arrays
    return _result(b, "headline", values, snippets, pol.src, p, year, lang)


def _head_series(c: Ctx, pol: PolicyFields, prefix: str) -> list[Value]:
    """N2-unit head change at every year-end from the reference year (0 there) to the horizon:
    ``dW[y]``, the value the headline shows when the page's year slider is at ``y``."""
    out = []
    for yi in range(c.YR, c.Y):
        v = float(pol.dW[yi])
        out.append(Value(f"{prefix}{c.years[yi]}", v, J.sg(v), "m",
                         paths=("n2Unit.solved[][]", "n2Unit.basis[][]"), functions=("headWhere",),
                         quantity="head_change", region="n2_yunlin_wells", aggregation="mean_own_layer"))
    return out


def start_compare(b: PageBundle, f, year: int, lang: str = "en") -> QueryResult:
    """N2-unit head for both starts at ``year`` and at the horizon, plus ``start_note`` exactly
    when the page shows it (renderStartNote: horizon, both runs carry N2, within tolerance)."""
    c = ctx(b)
    y, H = c.yidx(year), c.Y - 1
    pols = [policy_fields(b, Policy(tuple(f), s)) for s in c.starts]
    values, snippets = [], []
    for s, pol in zip(c.starts, pols, strict=True):
        if not pol.hasW:
            continue
        for yy, tag in ((y, "year"), (H, "horizon")):
            v = float(pol.dW[yy])
            values.append(Value(f"head_{s}_{tag}", v, J.sg(v), "m", paths=("n2Unit.solved[][]", "n2Unit.basis[][]"),
                                functions=("renderStartNote",), quantity="head_change",
                                region="n2_yunlin_wells", aggregation="mean_own_layer"))
        if pol.src in ("solved", "basis"):
            values += _head_series(c, pol, f"head_{s}_y")
    if (len(c.starts) >= 2 and all(p.src in ("solved", "basis") for p in pols)
            and c.n2u and all(p.hasW for p in pols)):
        a = [float(p.dW[H]) for p in pols]
        if abs(a[0] - a[1]) <= max(.01, .03 * abs(a[0])):
            # FS.start_invariance.v1 pass (renderStartNote): the 2022-12 start moves the levels
            fs = (c.mc.get("results") or {}).get("fs")
            fs_ok = bool((c.mc.get("registrations") or {}).get("fs") and fs
                         and fs.get("wording_case") == "pass")
            tail = c.tr(lang, "start_note_fs") if fs_ok else ""
            tail = "" if tail == "start_note_fs" else tail          # key absent: older template
            snippets.append(Snippet("start_note+start_note_fs" if tail else "start_note", lang,
                                    c.tr(lang, "start_note", {
                                        "s0": c.starts[0], "s1": c.starts[1], "v": J.sg(a[0]),
                                        "n": c.n2u["n"], "H": c.years[-1]}) + tail,
                                    (f"head_{c.starts[0]}_horizon",)))
    p0 = Policy(tuple(f), c.starts[0])
    return _result(b, "start_compare", values, snippets, pols[0].src, p0, year, lang)


def rank(b: PageBundle, year: int, by: Literal["head", "gwh", "cm"], lang: str = "en") -> QueryResult:
    """``renderRank``: per class and start, head rise at N2's wells (or per 100 GWh/yr)."""
    c = ctx(b)
    if by == "cm":
        return _result(b, "rank", [], [], "none", None, year, lang,
                       withheld=[Withheld("C03", "subsidence_size")])
    y = c.yidx(year)
    u = c.n2u

    def hd(k, si):
        if u and u.get("basis"):
            return float(u["basis"][si * c.K + k][y])
        if not c.has_basis:
            return 0.0
        n = c.A * c.Y
        off = (si * c.K + k) * n
        blk = b.array("dHeadL2")[off:off + n].reshape(c.A, c.Y)[:, y]
        return _jsum(blk) / c.A
    rows = []
    for k, cl in enumerate(c.classes):
        a = [hd(k, si) for si in range(len(c.starts))]
        v = [100 * x / cl["gwh"] if cl.get("gwh") else math.nan for x in a] if by == "gwh" else a
        rows.append((k, v))
    rows.sort(key=lambda r: -(r[1][0] if r[1][0] == r[1][0] and r[1][0] else 0))   # stable, like JS
    values, table = [], []
    for k, v in rows:
        cells = []
        for si, x in enumerate(v):
            txt = J.to_fixed(x, 3) if x == x else "NaN"
            cells.append(txt)
            values.append(Value(f"rank_{k}_{c.starts[si]}", x, txt, "m" if by == "head" else "m/100GWh",
                                paths=("n2Unit.basis[][]",) if u and u.get("basis") else ("arrays.dHeadL2",),
                                functions=("renderRank",), quantity="class_rank",
                                region=by, aggregation="per_class"))
        table.append((c.CN(k, lang), *cells))
    if by == "gwh":
        unit = c.tr(lang, "rank_unit_gwh")
        ukey = "rank_unit_gwh"
    elif u and u.get("basis"):
        unit, ukey = c.tr(lang, "rank_unit_head", {"y": c.years[y], "n": u["n"]}), "rank_unit_head"
    else:
        unit, ukey = c.tr(lang, "rank_unit_head_l2", {"y": c.years[y]}), "rank_unit_head_l2"
    snippets = [Snippet(ukey, lang, unit), Snippet("rank_limit", lang, c.tr(lang, "rank_limit"))]
    header = ("", *[str(s) for s in c.starts])
    return _result(b, "rank", values, snippets, "basis" if c.has_basis else "none", None, year, lang,
                   table=(header, *table))


def township(b: PageBundle, p: Policy, year: int, town: int, lang: str = "en") -> QueryResult:
    """The township row's model-internal value (``tw_avoided_val``, C11 L) with the page's
    labels: ``mi_note``, ``tw_low`` (E09) and ``tw_pooled``/``tw_pooled_np`` (spillover)."""
    c = ctx(b)
    y = c.yidx(year)
    pol = policy_fields(b, p)
    if pol.src in ("base", "none"):
        return _result(b, "township", [], [], pol.src, p, year, lang)
    kp = kpis(b, pol, y).kp
    i = int(town)
    tw = c.towns[i]
    dv = kp["town"][i]
    txt = J.f2(dv)
    values = [Value("town_avoided", dv, txt, "cm", paths=("arrays.dSubs",), functions=("renderTowns",),
                    quantity="avoided_subsidence", region="township", aggregation="cell_mean")]
    sk = (b.get("townSkill") or [None] * len(c.towns))[i]
    skill = ""
    if sk:
        skill = c.tr(lang, "tw_skill", {
            "n": sk["n"], "bias": c.tr(lang, "tw_bias", {"v": J.sg(sk["bias"], 1)}) if sk.get("bias") is not None else "",
            "r2": c.tr(lang, "tw_r2", {"v": J.to_fixed(sk["r2"], 2)}) if sk.get("r2") is not None else ""})
    pooled = (c.tr(lang, "tw_pooled") + (c.tr(lang, "tw_pooled_np") if c.nochp() else "")) if tw.get("pooled") else ""
    name = c.TN(i, lang) if tw.get("pooled") else f"{tw['zh']} {tw['en']}"
    snippets = [Snippet("mi_note", lang, c.tr(lang, "mi_note")),
                Snippet("town_name", lang, name),
                Snippet("tw_avoided_val", lang, c.tr(lang, "tw_avoided_val", {"v": txt}), ("town_avoided",))]
    low = bool(sk and sk.get("low"))
    if low:
        snippets.append(Snippet("tw_pooled+tw_low" if pooled else "tw_low", lang,
                                pooled + c.tr(lang, "tw_low", {"s": skill})))
    elif pooled or skill:
        snippets.append(Snippet("tw_pooled+tw_skill" if pooled else "tw_skill", lang, pooled + skill))
    return _result(b, "township", values, snippets, pol.src, p, year, lang)


def rail(b: PageBundle, p: Policy, year: int, lang: str = "en") -> QueryResult:
    """The rail note (``rail_note`` with ``railLine``; C06 L) at the page default threshold."""
    c = ctx(b)
    y = c.yidx(year)
    pol = policy_fields(b, p)
    if not b.get("hsr"):
        return _result(b, "rail", [], [Snippet("no_rail", lang, c.tr(lang, "no_rail"))], pol.src, p, year, lang)
    kp = kpis(b, pol, y).kp
    isB = pol.src in ("base", "none")
    dd = (kp["hsrPol"] - kp["hsrBase"]) / kp["hsrBase"] if kp["hsrBase"] else 0.0
    vals = [Value("rail_base", kp["hsrBase"], J.inv(kp["hsrBase"]), None, paths=("hsr.ch[]",),
                  functions=("renderHsr",), quantity="rail_angular_distortion", region="hsr", aggregation="max"),
            Value("rail_pol", kp["hsrPol"], J.inv(kp["hsrPol"]), None, paths=("hsr.ch[]",),
                  functions=("renderHsr",), quantity="rail_angular_distortion", region="hsr", aggregation="max")]
    if isB or abs(dd) < .002:
        line = c.tr(lang, "rail_same")
    elif dd > 0:
        pv = J.js_round(dd * 1000) / 10
        vals.append(Value("rail_pct", pv, J.js_num_str(pv), "%", paths=("hsr.ch[]",), functions=("railLine",),
                          quantity="rail_angular_distortion", region="hsr", aggregation="change"))
        vals.append(Value("rail_km", kp["hsrAtP"], J.f1(kp["hsrAtP"]), "km", paths=("hsr.ch[]",),
                          functions=("railLine",), quantity="rail_angular_distortion", region="hsr",
                          aggregation="location"))
        line = c.tr(lang, "rail_steeper", {"p": pv, "km": J.f1(kp["hsrAtP"])})
    else:
        pv = J.js_round(-dd * 1000) / 10
        vals.append(Value("rail_pct", pv, J.js_num_str(pv), "%", paths=("hsr.ch[]",), functions=("railLine",),
                          quantity="rail_angular_distortion", region="hsr", aggregation="change"))
        line = c.tr(lang, "rail_gentler", {"p": pv})
    ex = ""
    if kp.get("hsrExcl"):
        vals.append(Value("rail_excl_km", kp["hsrExcl"], J.f1(kp["hsrExcl"]), "km", paths=("hsr.ch[]",),
                          functions=("renderHsr",), quantity="rail_angular_distortion", region="hsr",
                          aggregation="excluded"))
        vals.append(Value("rail_raw", kp["hsrRawB"], J.inv(kp["hsrRawB"]), None, paths=("hsr.ch[]",),
                          functions=("renderHsr",), quantity="rail_angular_distortion", region="hsr",
                          aggregation="max_raw"))
        ex = c.tr(lang, "rail_ex_join") + c.tr(lang, "rail_excl", {"km": J.f1(kp["hsrExcl"]),
                                                                    "raw": J.inv(kp["hsrRawB"])})
    note = c.tr(lang, "rail_note", {"a": J.inv(kp["hsrBase"]), "b": J.inv(kp["hsrPol"]), "line": line, "ex": ex})
    return _result(b, "rail", vals, [Snippet("rail_note", lang, note, tuple(v.key for v in vals))],
                   pol.src, p, year, lang)


def cheapest_for_head(b: PageBundle, target_m: float, year: int, start: int,
                      lang: str = "en") -> QueryResult:
    """Lowest-GWh page lever setting (single-class 5 % cuts, uniform all-class cuts) whose N2
    head at ``year`` reaches ``target_m``. Flagged off until ruling D6 (presets gate it)."""
    c = ctx(b)
    y = c.yidx(year)
    cands = []
    for pct in range(5, 101, 5):
        for k in range(c.K):
            cands.append(Policy.from_cuts({k: pct}, start, c.K))
        cands.append(Policy.from_cuts(dict.fromkeys(range(c.K), pct), start, c.K))
    best = None
    for p in cands:
        pol = policy_fields(b, p)
        if not pol.hasW:
            continue
        kp = kpis(b, pol, y).kp
        if kp["n2Head"] is None or kp["n2Head"] < target_m or kp["gwh"] is None:
            continue
        if best is None or kp["gwh"] < best[1]:
            best = (p, kp["gwh"])
    if best is None:
        return _result(b, "cheapest", [], [], "none", None, year, lang)
    hl = headline(b, best[0], year, lang)
    return QueryResult(kind="cheapest", values=hl.values, snippets=hl.snippets, src=hl.src,
                       policy=best[0], year=year, deep_link=hl.deep_link, page_sha256=hl.page_sha256,
                       ledger_sha256=hl.ledger_sha256)


# ---------------------------------------------------------------------------------------
# evidence (how_sure, phase 5): a stable subset of the trust drawer's evidence list
# ---------------------------------------------------------------------------------------
MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
BESTK = {"clim": "best_clim", "clim_trend": "best_clim_trend", "persist": "best_persist"}


def month_year(iso: str | None, lang: str) -> str:
    """The page's ``monthYear``."""
    if not iso:
        return ""
    y, m = str(iso).split("-")[:2]
    return f"{y}年{int(m)}月" if lang == "zh" else f"{MON[int(m) - 1]} {y}"


def nc_span(nc: dict) -> str:
    """The page's ``ncSpan``: "2025–26" from forecast origin and exclusive end."""
    a = str(nc.get("forecast_origin") or "").split("-")
    b2 = str(nc.get("end_exclusive") or "").split("-")
    y0 = (int(a[0]) + 1 if int(a[1]) == 12 else int(a[0])) if len(a) > 1 else None
    y1 = (int(b2[0]) - 1 if int(b2[1]) == 1 else int(b2[0])) if len(b2) > 1 else None
    return f"{y0}–{str(y1)[2:]}" if y0 and y1 else ""


def _best_name(c: Ctx, lang: str, k) -> str:
    return c.tr(lang, BESTK[k]) if k in BESTK else (k or "–")


def _esc(v) -> str:
    """The page's ``esc``: ``String(v)`` (so a missing value reads "undefined") with &<>" escaped."""
    t = "undefined" if v is None else (v if isinstance(v, str) else J.js_num_str(v))
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def js_keys(d: dict) -> list:
    """``Object.keys`` order: array-index keys ascending, then the others in insertion order."""
    def idx(k) -> bool:
        return (isinstance(k, str) and k.isdigit() and (k == "0" or k[0] != "0")
                and int(k) < 2**32 - 1)
    return sorted((k for k in d if idx(k)), key=int) + [k for k in d if not idx(k)]


def _num0(v) -> float:
    """A JS number in ``Math.min``/``Math.max``: ``null`` counts as 0."""
    return 0.0 if v is None else float(v)


class _Lines:
    """Builds page lines with the page's ``tr`` and records every key each line used, so a
    snippet's key names the page string of the line first and then each sub-string that carries
    claims (``display_map.i18n``): the claims behind the sentence are exactly the page's."""

    def __init__(self, c: Ctx, lang: str):
        self.c, self.lang, self.sn, self.used = c, lang, [], []
        self.dm = (c.b.ledger.get("display_map") or {}).get("i18n") or {}

    def t(self, key: str, v: dict | None = None) -> str:
        self.used.append(key)
        return self.c.tr(self.lang, key, v)

    def take(self) -> list[str]:
        k, self.used = self.used, []
        return k

    def push(self, key: str, text: str, used: list[str] | None = None) -> None:
        used = self.take() if used is None else used
        subs = [k for k in dict.fromkeys(used) if k != key and self.dm.get(k)]
        if key in self.c.b.T["en"] and text:              # an older template: no such line
            self.sn.append(Snippet("+".join([key, *subs]), self.lang, text))

    def add(self, key: str, v: dict | None = None) -> None:
        self.push(key, self.t(key, v))


def cov_span(cv: dict) -> str:
    """The page's ``covSpan``."""
    return nc_span({"forecast_origin": cv.get("forecast_origin"),
                    "end_exclusive": cv.get("end_exclusive")})


def test_counts(c: Ctx) -> tuple[int, int]:
    """The page's ``testCounts`` over ``testRows`` (with ``n1Rows``): PASS and FAIL rows, without
    the N1 rows of a model that is not the deliverable recipe (``count === false``)."""
    mc = c.mc
    tp = mc.get("temporal")
    fr = tp.get("fair") if tp else None
    rows: list[tuple[str | None, bool]] = []
    nc = mc.get("new_data_challenge")
    if nc:
        rows.append(("PASS" if nc.get("beats_best_baseline") else "FAIL", True))
    if fr is not None:
        rows += [(fr.get("verdict"), True), (None, True)]
    elif tp:
        rows.append(("PASS" if tp.get("passed") else "FAIL", True))
    if mc.get("r2_anom") is not None:
        rows.append((mc.get("verdict_anom") or None, True))
    if mc.get("r2_kfold") is not None:
        rows.append((mc.get("verdict") or None, True))
    pr = mc.get("policy_response")
    if pr:
        for kind in ("heads", "subsidence"):
            for row in (pr.get(kind) or {}).values():
                if not row.get("model"):
                    continue
                both = (row.get("verdict_cl") == "model_better"
                        and row.get("verdict_zr") == "model_better")
                rows.append(("PASS" if both else "FAIL",
                             row["model"].get("deliverable_recipe") is True))
    p = sum(1 for v, n in rows if n and v == "PASS")
    f = sum(1 for v, n in rows if n and v == "FAIL")
    return p, f


def _pr_span(r: dict) -> str:
    m = re.match(r"^(\d{4})-\d{2}\.\.(\d{4})", _sub(r.get("window")))
    return f"{m.group(1)}–{m.group(2)[2:]}" if m else ""


def _pr_design(r: dict) -> bool:
    return r.get("design_time") is True or "design-time" in _sub(r.get("window"))


def _sub(v) -> str:
    """``String(v)``."""
    return J._sub_str(v) if not isinstance(v, (dict, list)) else str(v)


def _policy_response_lines(L: _Lines, r: dict) -> None:
    """``policyResponseLines`` (with ``chCapF`` / ``chCapRd``): the registered N1 verdicts."""
    c, f2 = L.c, J.f2
    mc = c.mc
    rg, res = mc.get("registrations") or {}, mc.get("results") or {}
    h, s, ch = r.get("heads") or {}, r.get("subsidence") or {}, r.get("changhua")
    span = _pr_span(r)
    dt = L.t("pr_design") if _pr_design(r) else ""
    dtp = L.t("pr_design_p") if _pr_design(r) else ""

    def passed(row) -> bool:
        return row.get("verdict_cl") == "model_better" and row.get("verdict_zr") == "model_better"

    def sized(row) -> bool:
        return (bool(row) and row.get("slope_obs_on_model") is not None
                and bool(row.get("slope_ci")) and len(row["slope_ci"]) == 2)

    def pr_model(row) -> str:
        i = _sub(((row or {}).get("model") or {}).get("id") or "")
        return L.t("prm_r1c" if i.startswith("R1c") else "prm_r0" if i.startswith("R0")
                   else "prm_none")

    def cap_f() -> str:
        x = res.get("x5clm")
        if not x:
            return L.t("pr_ch_cap_f")
        if not (x.get("caps_c13a_at_N") and x.get("F") is not None):
            return ""
        return L.t("pr_ch_cap_f_res", {"f": f2(x["F"])})

    def cap_rd() -> str:
        x = res.get("x5n2c")
        if not x:
            return L.t("pr_ch_cap_rd")
        if x.get("status") != "estimated" or x.get("false_pass") is None:
            return ""
        return L.t("pr_ch_rd_held" if x["false_pass"] >= .25 else "pr_ch_rd_res",
                   {"p": f2(x["false_pass"])})

    for cty, row in h.items():
        other = bool(row.get("model")) and row["model"].get("deliverable_recipe") is False
        rs = L.t("pr_beats" if passed(row) else "pr_not_beats",
                 {"r": f2(row["rmse_model_m"]), "c": f2(row["rmse_cl_m"])})
        sec = (r.get("secondary") or {}).get(cty + "_heads_deliverable_recipe")
        np_ = (L.t("pr_ch_nopump", {"r": f2(sec["rmse_model_m"]), "c": f2(sec["rmse_cl_m"]),
                                    "q": f2(sec["ratio_cl"]), "u": J.sg(sec["ci_cl"][1], 4)})
               if cty == "changhua" and c.nochp() and sec else "")
        if cty == "changhua" and row.get("model") and row["model"].get("id") == "R1c-m1618":
            sl = (L.t("pr_ch_slope", {"s": f2(row["slope_obs_on_model"]),
                                      "lo": f2(row["slope_ci"][0]), "hi": f2(row["slope_ci"][1])})
                  if sized(row) else "")
            caps = (cap_f() if rg.get("x5clm") else "") + (cap_rd() if rg.get("x5n2c") else "")
            L.add("evc_pr_ch_head", {"span": span, "dt": dt, "res": rs, "sl": sl,
                                     "note": np_ + caps})
            continue
        note = ((L.t("pr_other_note") if other else "") + np_
                + (L.t("pr_ch_untested") if cty == "changhua" else ""))
        L.add("evc_pr_heads", {"cty": L.t("cty_" + cty), "span": span, "dt": dt,
                               "mdl": pr_model(row), "res": rs, "note": note})
    hy = h.get("yunlin")
    if sized(hy):                       # observed on modelled slope
        L.add("evc_pr_heads_size", {"span": span, "dt": dt, "hy": f2(hy["slope_obs_on_model"]),
                                    "hyl": f2(hy["slope_ci"][0]), "hyu": f2(hy["slope_ci"][1]),
                                    "my": pr_model(hy)})
    if s.get("yunlin") and s.get("changhua"):
        ok = [k for k in ("yunlin", "changhua") if passed(s[k])]
        if len(ok) == 2:
            L.add("evc_pr_subs_ok", {"span": span, "dt": dtp,
                                     "list": L.t("list_and").join(L.t("cty_" + k) for k in ok)})
        else:
            def sres(k):
                zr = s[k].get("verdict_zr") == "model_better"
                if s[k].get("verdict_cl") == "model_better":
                    return L.t("pr_s_beat" if zr else "pr_s_lost_zr")
                return L.t("pr_s_lost_cl" if zr else "pr_s_lost_both")

            def ci(row, i):
                ok = row.get("slope_ci") and len(row["slope_ci"]) == 2
                return f2(row["slope_ci"][i]) if ok else "–"
            bb = (r.get("step_b") or {}).get("yunlin")
            sb = ""
            if (bb and bb.get("beta_obs_cm_per_gwh") is not None
                    and bb.get("beta_model_cm_per_gwh") is not None
                    and bb.get("ratio_ci") and len(bb["ratio_ci"]) == 2):
                more = bb["beta_obs_cm_per_gwh"] > bb["beta_model_cm_per_gwh"]
                vk = "vd_" + _sub(bb.get("verdict"))
                sb = L.t("pr_stepb", {"dir": L.t("pr_more" if more else "pr_less"),
                                      "v": (L.t(vk) if vk in c.b.T["en"]
                                            else _sub(bb.get("verdict"))),
                                      "o": J.sg(bb["beta_obs_cm_per_gwh"]),
                                      "m": J.sg(bb["beta_model_cm_per_gwh"]),
                                      "lo": f2(bb["ratio_ci"][0]), "hi": f2(bb["ratio_ci"][1])})
            L.add("evc_pr_subs", {"span": span, "dt": dtp,
                                  "sy": f2(s["yunlin"]["slope_obs_on_model"]),
                                  "syl": ci(s["yunlin"], 0), "syu": ci(s["yunlin"], 1),
                                  "sc": f2(s["changhua"]["slope_obs_on_model"]),
                                  "scl": ci(s["changhua"], 0), "scu": ci(s["changhua"], 1),
                                  "my": pr_model(s["yunlin"]), "mc": pr_model(s["changhua"]),
                                  "ry": sres("yunlin"), "rc": sres("changhua"), "sb": sb})
    if ch and ch.get("status") == "not_supported":       # C13c: withdrawn under every outcome
        L.add("evc_pr_ch_subs", {"o": J.sg(ch["dose_obs_cm_per_gwh"], 2),
                                 "m": J.sg(ch["dose_model_cm_per_gwh"], 2)})
    if h.get("changhua") or ch:                          # C13b: withdrawn whatever any verdict says
        L.add("evc_pr_ch_class", {"np": L.t("pr_ch_np") if c.nochp() else ""})
    L.take()


def _dr_line(L: _Lines, r: dict | None) -> None:
    """``drLine`` (DR.relative_drift.v1, C14b)."""
    if not r:
        return
    cs = [k for k in ("Yunlin", "Changhua") if r.get(k)]
    if not cs:
        return

    def one(k) -> str:
        v, cty = r[k], L.t("cty_" + k.lower())
        if v.get("evaluable") and v.get("excess_drift_6y_m") is not None:
            return L.t("dr_cty", {"cty": cty, "x": J.f2(v["excess_drift_6y_m"])})
        return L.t("dr_ne", {"cty": cty})
    parts = [one(k) for k in cs]
    L.add("evc_pro_dr", {"list": L.t("list_and").join(parts),
                         "trig": L.t("pro_dr_trig") if any(r[k].get("include_drift") for k in cs)
                         else ""})


def _ring_line(L: _Lines) -> None:
    """``ringLine``."""
    r = L.c.mc.get("ring_holdout")
    if not r or not r.get("rings") or not r["rings"].get("cum"):
        return
    cum = r["rings"]["cum"]
    arm = ("model" if "model" in cum
           else next((k for k in js_keys(cum) if not k.startswith("base_")), None))
    bc = cum.get("base_continue")
    if not arm or not cum.get(arm) or not bc or cum[arm].get("r2") is None or bc.get("r2") is None:
        return
    w = r.get("window")
    span = f"{_sub(w[0])[:4]}–{_sub(w[1])[2:4]}" if w and len(w) == 2 else ""
    v = {"span": span, "n": r["rings"].get("n_sites"), "a": J.sg(cum[arm]["r2"], 2),
         "b": J.sg(bc["r2"], 2)}
    if (r.get("max_ref_age_days") is not None and cum[arm].get("rmse") is not None
            and bc.get("rmse") is not None):
        L.add("evc_ring_ref", v | {"ra": J.f2(cum[arm]["rmse"]), "rb": J.f2(bc["rmse"])})
    else:
        L.add("evc_ring", v)


def _pile_line(L: _Lines) -> None:
    """``pileLine``: report-only (ruling U7) when the piles chose between columns."""
    r, lang, f1 = L.c.mc.get("deep_piles"), L.lang, J.f1
    if (not r or not r.get("cum") or not r["cum"].get("model")
            or r["cum"]["model"].get("rmse") is None):
        return
    w = r.get("window")
    span = f"{month_year(w[0], lang)}–{month_year(w[1], lang)}" if w and len(w) == 2 else ""
    a = r.get("after_feeds") or {}
    aw = a.get("window")
    span2 = f"{month_year(aw[0], lang)}–{month_year(aw[1], lang)}" if aw and len(aw) == 2 else ""
    d = r.get("cum_deep") or {}
    dm, db, sel = d.get("model") or {}, d.get("base_continue") or {}, r.get("selection")
    bc = r["cum"].get("base_continue")
    pre = L.t("evc_pile_ro", {"d": sel.get("date"), "u": sel.get("ruling")}) if sel else ""
    txt = L.t("evc_pile", {
        "span": span, "n": r.get("n_sites"), "v": r.get("n_verified_values"),
        "e": f1(r["cum"]["model"]["rmse"]),
        "b": f1(bc["rmse"] if bc else math.nan),
        "nd": "?" if d.get("n_sites") is None else d["n_sites"],
        "ed": "–" if dm.get("rmse") is None else f1(dm["rmse"]),
        "bd": "–" if db.get("rmse") is None else f1(db["rmse"]),
        "kd": "?" if dm.get("n_inside") is None else dm["n_inside"],
        "span2": span2, "e2": f1(a["model"]["rmse"]) if a.get("model") else "–",
        "b2": f1(a["base_continue"]["rmse"]) if a.get("base_continue") else "–"})
    L.push("evc_pile", pre + txt)


def band_keys(b: PageBundle) -> set[str]:
    """The ledger's ``page_scoping.answer_must_not_use_band_keys``."""
    return set(b.ledger["page_scoping"].get("answer_must_not_use_band_keys", []))


def _no_band(b: PageBundle, res: QueryResult) -> QueryResult:
    banned = band_keys(b)
    return replace(res, snippets=tuple(s for s in res.snippets
                                       if not banned & set(s.key.split("+"))))


def evidence(b: PageBundle, lang: str = "en", *, full: bool = False) -> QueryResult:
    """``evLineText``, the head tile's reading at the page's default view (``tW_w30`` when the
    page has the all30 run; none otherwise) and every line of ``evidenceItems`` (template.html
    ~2888) in its order. ``full`` keeps every line, for the parity check; otherwise the lines
    using a band key the ledger keeps out of answers (``answer_must_not_use_band_keys``: the
    per-well interval ``evc_pro_iv``, DR's drift ``evc_pro_dr``, FV v2's forcing-year coverage
    ``evc_fv2_rc_*``) are dropped. Lines the bound page's dictionary leaves empty (an older
    template) are skipped."""
    res = _drawer(b, lang)
    return res if full else _no_band(b, res)


def _drawer(b: PageBundle, lang: str) -> QueryResult:
    c = ctx(b)
    mc, L, f2 = c.mc, _Lines(c, lang), J.f2

    def best(k) -> str:
        return L.t(BESTK[k]) if k in BESTK else (k or "–")
    nc, tp = mc.get("new_data_challenge"), mc.get("temporal")
    if nc:                                                   # evLineText
        L.add("ev_nc_pass" if nc.get("beats_best_baseline") else "ev_nc_fail",
              {"r": f2(nc["metrics"]["model_datum"]["rmse_m"]),
               "c": f2(nc["metrics"][nc["best_baseline"]]["rmse_m"]),
               "b": best(nc["best_baseline"]), "span": nc_span(nc)})
    elif tp and not tp.get("passed"):
        L.add("ev_tp_fail", {"y": c.years[c.YO]})
    else:
        L.add("ev_tp_pass", {"y": c.years[c.YO]})
    rk, r, _ = tile_w(b, Policy((1.0,) * c.K, c.starts[0]), c.years[-1], lang)   # default view
    if rk:
        L.push(rk, r, [])
    # ---- evidenceItems() ----
    rel = b.get("release")
    L.add("ev_status", {"u": _esc(rel.get("uncertainty")) if rel else ""})
    L.add("ev_decision")
    if nc:
        L.add("evc_nc_pass" if nc.get("beats_best_baseline") else "evc_nc_fail",
              {"n": nc.get("stations"), "span": nc_span(nc),
               "r": f2(nc["metrics"]["model_datum"]["rmse_m"]),
               "c": f2(nc["metrics"][nc["best_baseline"]]["rmse_m"]),
               "b": best(nc["best_baseline"])})
    cv = mc.get("coverage")
    if cv:
        L.add("evc_cov", {"n": mc.get("n_members"), "a": J.pctf(100 * cv["picp_90"]),
                          "span": cov_span(cv),
                          "x": f2(cv["inflation_90"]), "c": J.pctf(100 * cv["combined_picp_90"]),
                          "st": cv.get("stations"), "m": J.locale_en(cv["scored_months"])})
    rg = mc.get("registrations") or {}
    res = mc.get("results") or {}
    q = mc.get("prospective")
    if q:
        qa = month_year(str(q["test_start"])[:7], lang) if q.get("test_start") else ""
        qb = month_year(str(q["test_end"])[:7], lang) if q.get("test_end") else ""
        if q.get("test_start") and q.get("test_end"):
            L.add("evc_pro", {"reg": q["registered_on"], "a": qa, "b": qb, "n": q["stations"]})
        else:
            L.add("evc_pro_nodates", {"reg": q["registered_on"], "n": q["stations"]})
        if q.get("interval_bound") and qa and qb:          # band key: parity only
            L.add("evc_pro_iv", {"a": qa, "b": qb,
                                 "out": L.t("pro_out_e02y" if rg.get("e02y") else "pro_out_un")})
        if rg.get("dr"):                                     # drLine, band key: parity only
            _dr_line(L, res.get("dr"))
        if rg.get("e02c"):
            L.add("evc_pro_e02c")
        if rg.get("e02z"):                                   # C30-scen's years-2/3 sub-label (R8.7)
            L.add("evc_pro_e02z")
        if rg.get("e02d"):
            L.add("evc_pro_e02d")
        fm = res.get("fm") if rg.get("fm") else None        # fmLine: only its declared case
        if (fm and fm.get("case") == "within_first_year"
                and fm.get("clause_case") == "window_beyond_memory" and qa and qb):
            L.add("evc_pro_fm", {"m": fm["lstar_months"], "lm": month_year(fm["lstar_month"], lang),
                                 "a": qa, "b": qb, "f": f2(fm["floor_m"])})
        if rg.get("ext"):
            L.add("evc_pro_ext")
            fx = res.get("fvext")                            # fvextLine
            if rg.get("fvext") and fx and fx.get("status") == "scored":
                L.add("evc_fvext_wbd" if fx.get("reading") == "weak_by_design" else "evc_fvext_nsw",
                      {"p": f2(fx["p"])})
        if rg.get("fv"):
            # fvLine: v2's post-failure exploratory level reading once scored; else v1's line
            # fixed beforehand until its report exists, then its declared not-evaluable line
            v2 = res.get("fv2") if rg.get("fv2") else None
            fv = res.get("fv")
            if v2 and v2.get("status") == "scored":
                wbd = v2.get("level") == "weak_by_design"
                L.add("evc_fv2_lvl_wbd" if wbd else "evc_fv2_lvl_nsw",
                      {"p": f2(v2["p_level"])})
            elif not fv:
                L.add("evc_pro_fv")
            else:
                sup = fv.get("superposition") or {}
                why = (L.t("fv_why_sup", {"x": f2(sup["max_abs_window_m"]),
                                          "t": f2(sup["tolerance_m"])})
                       if sup and sup.get("pass") is False else _esc(fv.get("why")))
                L.add("evc_pro_fv_ne", {"why": why})
        if rg.get("fv2"):                      # fv2Lines (the rc lines are band keys)
            w, y = res.get("fv2"), res.get("fv2fy")
            if w and w.get("status") == "scored":
                for cty in ("Yunlin", "Changhua"):
                    v = (w.get("coverage") or {}).get(cty)
                    if v and v.get("p") is not None:
                        wbd = v.get("reading") == "weak_by_design"
                        L.add("evc_fv2_cov_wbd" if wbd else "evc_fv2_cov_nsw",
                              {"cty": L.t("cty_" + cty.lower()), "p": f2(v["p"])})
            if y and y.get("status") == "scored":
                for cty in ("Yunlin", "Changhua"):
                    v = (y.get("rc") or {}).get(cty)
                    if not v or v.get("reading") == "not_represented":
                        continue
                    forcing = L.t("fv2_forcing_rep" if y.get("rain") == "replayed"
                                  else "fv2_forcing_clim")
                    L.add("evc_fv2_rc_omits" if v.get("reading") == "omits" else "evc_fv2_rc_nso",
                          {"cty": L.t("cty_" + cty.lower()), "forcing": forcing,
                           "rc": J.pctf(100 * v["rc"]), "tlo": J.pctf(100 * v["t_lo"]),
                           "thi": J.pctf(100 * v["t_hi"]), "lo": J.pctf(100 * v["lo"]),
                           "hi": J.pctf(100 * v["hi"])})
                if (y.get("rc") or {}).get("Changhua"):
                    L.add("evc_fv2_ch_rep" if y.get("rain") == "replayed" else "evc_fv2_ch_clim")
    if rg.get("no_band"):
        L.add("evc_band_none")
    # the statistical layer: its retrospective score, its registration, the forecast of record
    # (ruling R8.7, claim C30-fc) and FV.stat.v2's two lines (fvsLines, ruling R8.8)
    if mc.get("hybrid"):                                     # the page's `if (mc.hybrid)`
        hy = mc["hybrid"]
        py = hy.get("per_year") or {}
        ys = js_keys(py)
        y1 = ys[0] if ys else ""
        L.add("evc_hyb", {"s": f2(hy["rmse_stat"]), "c": f2(hy["rmse_clim"]),
                          "r": f2(hy["rmse_model"]),
                          "span": nc_span(hy), "y1": y1, "s1": f2(py[y1]["stat"]) if y1 else "",
                          "c1": f2(py[y1]["clim"]) if y1 else "", "p": f2(hy["ratio_phys"])})
        g = hy.get("registration") or {}
        if g.get("test_start") and g.get("test_end"):
            ga = month_year(str(g["test_start"])[:7], lang)
            gb = month_year(str(g["test_end"])[:7], lang)
            L.add("evc_hyb_reg", {"reg": g["registered_on"], "a": ga, "b": gb, "n": g["stations"]})
            if g.get("result") not in ("failed", "not_evaluable"):     # forecastOfRecord
                L.add("evc_hyb_for", {"a": ga, "b": gb})
        f = res.get("fvs2")                                  # fvsLines: under mc.hybrid only
        if (rg.get("fvs2") and f and f.get("status") == "scored"
                and f.get("reading") == "not_shown_weak"):
            v = {"p": f2(f["p"]), "p0": f2(f["p0"]), "q": f2(1 - f["p"])}
            L.add("evc_hyb_fvs", v)
            L.add("evc_hyb_fvs_pass", v)
    _ring_line(L)
    _pile_line(L)
    if mc.get("policy_response"):
        _policy_response_lines(L, mc["policy_response"])
    n2 = res.get("n2rd")                                     # n2rdLine: C01's D part (R8.8)
    if rg.get("n2rd") and n2 and n2.get("wording") == "informative":
        L.add("evc_n2_rd", {"fp": f2(n2["false_pass"]), "ws05": f2(n2["ws05"]),
                            "ws2": f2(n2["ws2"]), "cov": f2(n2["cov"]), "pw": f2(n2["power"]),
                            "pwc": f2(n2["power_canal"]),
                            "pwf": f2(n2["power_forcing"])})
    for k, key in (("chx", "evc_chx"), ("chb", "evc_chb")):  # chDirLines (E25/E26): direction only
        r = res.get(k)
        if rg.get(k) and r:
            txt = L.t(key, {"s": f2(r["beta"]), "lo": f2(r["lo"]), "hi": f2(r["hi"]),
                            "n": r["n_wells"]})
            if k == "chx":                                   # chxSepClause (R8.10): narrowing only
                x = res.get("chx2c")
                if rg.get("chx2c") and x and x.get("reading") == "not_separable":
                    txt += L.t("chx_2c_ns" if x["lo"] <= 0 else "chx_2c_ns_gen",
                               {"s": f2(x["beta"]), "lo": f2(x["lo"]), "hi": f2(x["hi"])})
            elif r.get("p_k1") is not None and r.get("p_k0") is not None:
                txt += L.t("chb_syn", {"k1": f2(r["p_k1"]), "k0": f2(r["p_k0"])})
            L.push(key, txt)
    n3c = rg.get("n3_conv12") or rg.get("n3_conv24")
    ck = "decomp_conv_un" if not rg.get("n3") else "decomp_conv_n3" if n3c else "decomp_conv_g0"
    if ck in b.T["en"]:
        L.add("evc_subs_decomp", {"conv": L.t(ck)})
    if rg.get("n3"):
        pk = ("pile_conv_12" if rg.get("n3_conv12") else
              "pile_conv_24" if rg.get("n3_conv24") else "pile_conv_ro")
        L.add("evc_pile_n3", {"conv": L.t(pk)})
    if rg.get("n3sl"):
        L.add("evc_n3sl")
    p, fl = test_counts(c)
    if p + fl:
        L.add("ev_counts_long", {"p": p, "f": fl})
    return _result(b, "evidence", [], L.sn, "base", None, None, lang)


def head_band(b: PageBundle, lang: str = "en", *, full: bool = False) -> QueryResult:
    """The head series' band at the page's default view (head, fan average): ``seriesFor``'s band
    note (with ``bn_aux``, the layer-2 wells' note), ``covLine`` and, under ruling R8.13
    (PB.decomposed_band.v1), ``renderPB``'s labelled two-part band (title, notes, zone labels,
    county ranges, FV v2's forcing-year clause ``bn_head_fy_pb``, the leave-one-family-out line).
    ``full`` keeps every line, for the parity check; otherwise lines with a ledger band key
    (every R8.13 number, ``bn_head_fy``) are dropped."""
    res = _head_band(b, lang)
    return res if full else _no_band(b, res)


def _head_band(b: PageBundle, lang: str) -> QueryResult:
    c = ctx(b)
    mc, L = c.mc, _Lines(c, lang)
    rg, res = mc.get("registrations") or {}, mc.get("results") or {}
    mf = b.get("memberFields")
    pb = mc.get("pb")
    hb = None if pb else (mf or {}).get("baseHeadBandYr")
    y = res.get("fv2fy") if rg.get("fv2") else None
    fy_ok = bool(y and y.get("status") == "scored" and y.get("agg"))
    if pb:
        L.add("bn_head_pb")
    elif hb:                                               # headBandNote
        if not fy_ok:
            L.add("bn_head", {"n": mf["n"]})
        else:
            a = y["agg"]
            L.add("bn_head_fy", {"n": mf["n"], "n2lo": J.sg(a["n2_lo"]), "n2hi": J.sg(a["n2_hi"]),
                                 "l2lo": J.sg(a["l2_lo"]), "l2hi": J.sg(a["l2_hi"]),
                                 "forcing": L.t("fv2_forcing_rep" if y.get("rain") == "replayed"
                                                else "fv2_forcing_clim")})
    else:
        L.add("bn_head_none")
    wf = b.get("wellsFan")                                 # the fan's layer-2 wells (bn_aux)
    if wf:
        pts = [(o, m) for o, m in zip(wf["obs"], wf["model"], strict=False) if o is not None]
        d = [(_num0(m) - o) for o, m in pts if (_num0(m) - o) == (_num0(m) - o)]
        gap = sum(d) / len(d) if d else math.nan
        L.add("bn_aux", {"n": wf.get("n_wells"), "gap": J.sg(gap, 1), "gapa": J.f1(gap)})
    cv = mc.get("coverage")
    if (pb or hb) and cv:                                    # covLine (kept under R8.13)
        L.add("cov_line", {"n": mc.get("n_members"), "a": J.pctf(100 * cv["picp_90"]),
                           "x": J.f2(cv["inflation_90"]), "span": cov_span(cv),
                           "c": J.pctf(100 * cv["combined_picp_90"])})
    rib = (pb or {}).get("ribbon") or {}
    if pb and rib.get("months"):
        def lab(o) -> str:
            return _esc((o or {}).get(lang) or (o or {}).get("en") or "")

        def esc_t(key, v=None) -> str:
            return _esc(L.t(key, v))
        sh = (pb.get("shared") or {}).get("counties") or {}
        cs = [k for k in ("Yunlin", "Changhua") if sh.get(k)]
        fy = ""
        if fy_ok and mf and mf.get("baseHeadBandYr"):      # fyClause: the clause only (R8.13)
            a = y["agg"]
            fy = L.t("bn_head_fy_pb", {"n2lo": J.sg(a["n2_lo"]), "n2hi": J.sg(a["n2_hi"]),
                                       "l2lo": J.sg(a["l2_lo"]), "l2hi": J.sg(a["l2_hi"]),
                                       "forcing": L.t("fv2_forcing_rep"
                                                      if y.get("rain") == "replayed"
                                                      else "fv2_forcing_clim")})
        fy_keys = L.take()
        L.push("pb_title", esc_t("pb_title"))
        small = (" " + esc_t("pb_small", {"k": rib["small"]})) if rib.get("small") else ""
        L.push("pb_ribbon_h", f"<b>{esc_t('pb_ribbon_h')}</b> {lab(rib.get('label'))}. "
                              f"{esc_t('pb_ribbon_what')} {esc_t('pb_scale')}{small}")
        zones: dict[str, dict] = {}
        for z in rib.get("zones") or []:
            zones[z["zone"]] = {"n": z["n"], "pts": [v for v in z["hw"] if v is not None]}
        for k in sorted(zones):
            pts = zones[k]["pts"]
            if pts:
                L.push("pb_wells", f"{_esc(k)} ({esc_t('pb_wells', {'n': zones[k]['n']})}) "
                                   f"±{J.f1(pts[0])}…±{J.f1(pts[-1])} m")
        if cs:
            see = L.t("pb_see_fy") if fy else ""
            L.push("pb_shared_h", f"<b>{esc_t('pb_shared_h')}</b> "
                                  f"{lab(pb['shared'].get('label'))}. "
                                  f"{esc_t('pb_shared_what', {'see': see})}")
            for k in cs:
                v = sh[k]
                lo, hi = min(_num0(x) for x in v["lo"]), max(_num0(x) for x in v["hi"])
                L.push("pb_range", esc_t("pb_range", {"c": L.t("pb_county_" + k),
                                                      "lo": J.sg(lo, 1), "hi": J.sg(hi, 1)}))
            if fy:
                L.push("bn_head_fy_pb", fy, fy_keys)
            lf = pb.get("lofo") or {}
            lofo = L.t("pb_sep").join(
                L.t("pb_lofo_c", {"c": L.t("pb_county_" + k), "k": lf[k]["inside"],
                                  "n": lf[k]["n"]}) for k in cs if lf.get(k))
            if lofo:
                L.push("pb_lofo", esc_t("pb_lofo", {"list": lofo}))
    L.take()
    return _result(b, "head_band", [], L.sn, "base", None, None, lang)


def e24_line(b: PageBundle, p: Policy, lang: str = "en") -> QueryResult:
    """``e24Line`` (SD.cut_response_spread.v2, ruling R8.11): tile tA's model-form spread, shown
    on the page under Model internals only and only for the irrigation ×0.7 lever from 2026 (the
    run SD v2 measured). Never an answer number: its keys are in ``headline_must_not_use`` and
    no preset puts it in an answer; it is mirrored for the parity check only."""
    c = ctx(b)
    mc, L, f2 = c.mc, _Lines(c, lang), J.f2
    r = (mc.get("results") or {}).get("sd2")
    if ((mc.get("registrations") or {}).get("sd2") and r and r.get("ab") and len(p.f) == len(IRR30)
            and all(abs(v - IRR30[i]) <= 1e-6 for i, v in enumerate(p.f)) and int(p.start) == 2026):
        def pc(v) -> str:
            return J.js_num_str(J.js_round(100 * v))

        def one(t) -> dict:
            inl = L.t("e24_inl_rob" if t["inland_robust"] else "e24_inl_dep",
                      {"x": f2(t["inland_ratio"]), "ilo": J.to_fixed(t["inland_lo_cm"], 3),
                       "ihi": J.to_fixed(t["inland_hi_cm"], 3)})
            a, z = pc(t["coast_lo"]), pc(t["coast_hi"])
            return {"n": t["n_columns"], "lo": f2(t["lo_cm"]), "hi": f2(t["hi_cm"]),
                    "crng": a if a == z else f"{a}–{z}", "inl": inl}
        ta = L.t("def_tA_e24_a", one(r["a"])) if r.get("a") else ""
        L.add("def_tA_e24", one(r["ab"]) | {"ta": ta})
    L.take()
    return _result(b, "e24", [], L.sn, "base", p, None, lang)
