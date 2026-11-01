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
from dataclasses import dataclass, field
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
        return s, keys
    if pol.src == "none":
        return c.tr(lang, "prov_none"), ["prov_none"]
    return c.tr(lang, "prov_base", {"n": c.mc.get("n_members")}), ["prov_base"]


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
                            paths=("n2Unit.solved[][]", "n2Unit.basis[][]") if hw["n2"] else ("arrays.dHeadL2",),
                            functions=("headWhere",), quantity="head_change",
                            region="n2_yunlin_wells" if hw["n2"] else
                            ("yunlin_l2" if hw["key"] == "hw_yun" else "fan_l2"),
                            aggregation="mean_own_layer" if hw["n2"] else "cell_mean"))
        key = "hl_pol"
        s = c.tr(lang, "hl_pol", {"pol": pol_words, "yr": yr, "h": txt, "where": hw["where"], "year": Hy})
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
        if not isB:
            snippets.append(Snippet("tW_r", lang, c.tr(lang, "tW_r")))
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
    return _result(b, "headline", values, snippets, pol.src, p, year, lang)


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


def evidence(b: PageBundle, lang: str = "en") -> QueryResult:
    """``evLineText``, ``ev_decision`` and the registered-test lines of ``evidenceItems`` whose
    inputs are registry flags and dates (template.html ~1653, ~2634). Lines the bound page's
    dictionary leaves empty (an older template) are skipped."""
    c = ctx(b)
    mc, sn = c.mc, []

    def add(key, v=None):
        t = c.tr(lang, key, v)
        if t and t != key:
            sn.append(Snippet(key, lang, t))
    nc, tp = mc.get("new_data_challenge"), mc.get("temporal")
    if nc:
        add("ev_nc_pass" if nc.get("beats_best_baseline") else "ev_nc_fail",
            {"r": J.f2(nc["metrics"]["model_datum"]["rmse_m"]),
             "c": J.f2(nc["metrics"][nc["best_baseline"]]["rmse_m"]),
             "b": _best_name(c, lang, nc["best_baseline"]), "span": nc_span(nc)})
    elif tp and not tp.get("passed"):
        add("ev_tp_fail", {"y": c.years[c.YO]})
    else:
        add("ev_tp_pass", {"y": c.years[c.YO]})
    add("ev_decision")
    add("tW_r")
    rg = mc.get("registrations") or {}
    q = mc.get("prospective")
    if q:
        qa = month_year(str(q["test_start"])[:7], lang) if q.get("test_start") else ""
        qb = month_year(str(q["test_end"])[:7], lang) if q.get("test_end") else ""
        if q.get("test_start") and q.get("test_end"):
            add("evc_pro", {"reg": q["registered_on"], "a": qa, "b": qb, "n": q["stations"]})
        else:
            add("evc_pro_nodates", {"reg": q["registered_on"], "n": q["stations"]})
        # evc_pro_iv (the per-well interval line) is a band key the ledger keeps out of answers
        # (page_scoping.answer_must_not_use_band_keys); it stays in the page's drawer only
        for flag, key in (("e02c", "evc_pro_e02c"), ("ext", "evc_pro_ext")):
            if rg.get(flag):
                add(key)
        if rg.get("fv"):
            # fvLine: the rule fixed beforehand until FV's report exists, then its declared
            # not-evaluable line (the only scored outcome the page build accepts)
            fv = (mc.get("results") or {}).get("fv")
            sup = (fv or {}).get("superposition") or {}
            if not fv:
                add("evc_pro_fv")
            else:
                why = (c.tr(lang, "fv_why_sup", {"x": J.f2(sup["max_abs_window_m"]),
                                                 "t": J.f2(sup["tolerance_m"])})
                       if sup and sup.get("pass") is False else _esc(fv.get("why")))
                add("evc_pro_fv_ne", {"why": why})
    if rg.get("no_band"):
        add("evc_band_none")
    n3c = rg.get("n3_conv12") or rg.get("n3_conv24")
    conv = c.tr(lang, "decomp_conv_un" if not rg.get("n3") else "decomp_conv_n3" if n3c else "decomp_conv_g0")
    if conv and not conv.startswith("decomp_"):
        add("evc_subs_decomp", {"conv": conv})
    if rg.get("n3"):
        add("evc_pile_n3", {"conv": c.tr(lang, "pile_conv_12" if rg.get("n3_conv12") else
                                          "pile_conv_24" if rg.get("n3_conv24") else "pile_conv_ro")})
    if rg.get("n3sl"):
        add("evc_n3sl")
    return _result(b, "evidence", [], sn, "base", None, None, lang)
