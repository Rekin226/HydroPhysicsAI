"""Deterministic refusal guards (round 9, after ruling R9.2): four classes of question that must be
refused whatever the router would read, in English and Chinese. Simplified Chinese is folded to
traditional first (:func:`norm`, ``router.zh``), so every pattern here is written once.

They are written from the claims ledger and the page. Their authors also knew the four failure
classes of held-out v3 and the abridged v3 items printed in docs/analyst/EVAL_2026-11-07.md (v3
is spent); nothing else from v3.

- **Changhua levers** (C13a all-class head lever, C13b class levers, C13c subsidence lever; E20
  pending; ``page_scoping.changhua_lever_rule``). The page's levers are Yunlin's; Changhua is one
  pooled row "not yet split by township", shown only as the labelled spillover of a Yunlin
  policy. Refused when Changhua is named (Changhua, Chang-hua, Zhanghua, 彰化), reached by one
  of its townships (二林, 溪州, Erlin, Lu-kang, Peitou ...), by geography (north / far bank of the
  Choushui, 濁水溪以北, 溪北, the county north of Yunlin, Yunlin and its neighbour, both
  counties), or — for a lever question — by the whole alluvial fan (全扇區). The single
  exception is :func:`pooled_row_ok`.
- **Subsidence size** (C03, C09 withdrawn; C05 area): how much the ground sinks, settles,
  deforms or rebounds, in any wording. Exemptions are listed in :func:`subsidence_size` and in
  ``rules.hard_refusal``; an area-scope word (fan, county-wide, whole, total, 全縣, 整個,
  沖積扇 ...) overrides every one of them.
- **Beyond the horizon**: any year, ROC year, relative lead, decade or spelled year after the
  page's last year (``c.years[-1]``).
- **Single-township water level** (ruling D3; rules_v1.json head_change / township not
  offered): a township reference together with any head word, whatever else is asked.

Each guard only ever returns a refusal decision; none routes.

Round 10 (ruling R10.2a, docs/analyst/GUARDS_ROUND10.md) narrows them by five exemptions, each
with must-refuse boundary twins in tests/test_analyst_guards_r10.py: A1 a Yunlin question that
names Changhua only to leave it out (:func:`yunlin_only_ok`; the pooled row stays answerable only
as a row, :func:`pooled_row_ok`); A2 whether the subsidence size was tested (:func:`how_sure_only`);
A3 one how-sure question about Changhua's direction test (:func:`changhua_how_sure_ok`); A4 one
unnamed township's internals row (:func:`unnamed_township`, clarified by the rules); A5 is in the
rules stage. The round-10 builder used the spent held-out v3/v4 items as development material.
The round-9 code is frozen in ``router.r9`` for the $0 replay (R10.3b). After the adversarial
review of 42a859a the exemptions were narrowed again (tests/test_analyst_guards_r10_review.py): A3
is a strict allow-list, the spillover-without-row exemption was dropped, A2 and A1 gained vetoes.
"""
from __future__ import annotations

import re
import unicodedata

from .. import query as Q
from ..page import PageBundle
from .zh import to_trad


def norm(s: str) -> str:
    """NFKC, then simplified -> traditional (idempotent)."""
    return to_trad(unicodedata.normalize("NFKC", s))


def _has(pats, s: str) -> bool:
    return any(re.search(p, s, re.I) for p in pats)


def _first(pats, s: str):
    out = []
    for p in pats:
        out += [m.start() for m in re.finditer(p, s, re.I)]
    return out


# ---------------------------------------------------------------------------------------------
# levers and counties (shared)
# ---------------------------------------------------------------------------------------------
YUNLIN = [r"\byun[\s\-]*lin\b", "雲林"]
LEVER = [r"\bcut(?:s|ting)?\b", r"\breduc\w*", r"\bretir\w*", r"\bstop\w*", r"\bhalv\w*", r"\bhalf\b",
         r"\blower(?:s|ed|ing)?\s+(?:its|their|the)?\s*(?:pumping|irrigation|aquaculture|use)",
         r"\bcurb\w*", r"\bban(?:s|ned|ning)?\b", r"phase[\s\-]*out", r"\beliminat\w*",
         r"\bpump\w*\s+less\b", r"\bless\s+pumping\b", r"\bscal\w*\s+back\b", r"\btrim\w*",
         r"\d+(?:\.\d+)?\s*(?:%|percent|per\s*cent)", r"\bdrop(?:s|ped|ping)?\s+(?:its|their)?\s*pumping",
         "減", "停", "少抽", "退場", "禁", r"[一二三四五六七八九十\d]\s*成", "%", "降低抽", "降低用水", "槓桿",
         # round 10 fix (review 1): well closures, fallowing, curtailment, using or pumping less
         r"\b(?:shut|shuts|shutting|close|closes|closed|closing|seal\w*|cap(?:s|ped|ping)?)\s+(?:\S+\s+){0,3}?"
         r"wells?\b", r"\bwells?\s+(?:are\s+|being\s+)?(?:closed|shut|sealed|capped)\b",
         r"\bwell\s+(?:closure|shut[\s\-]*down|sealing)s?\b", r"\bfallow\w*", r"\bcurtail\w*",
         r"\b(?:use|uses|using|used)\s+less\s+water\b", r"\bless\s+water\b",
         r"\bpump\w*\s+(?:fewer|less|lower)\b", r"\bfewer\s+(?:hours|wells|pumps|days)\b", r"\blowered\b",
         r"\bcutback\w*", r"\bration\w*", r"\brestrict\w*",
         # second review: a switch to surface or tap water, a moratorium, a freeze, quotas, buyouts
         r"\b(?:switch\w*|convert\w*|replac\w*|substitut\w*|shift\w*|mov(?:e|es|ed|ing)|turn\w*)\s+(?:\S+\s+){0,4}?"
         r"(?:to|with|by|onto)\s+(?:\S+\s+){0,2}?(?:surface|tap|river|reservoir|piped|canal|alternative)",
         r"\bsurface[\s\-]+water\b", r"\btap[\s\-]+water\b", r"\bmoratori\w*", r"\bfreez\w*", r"\bfrozen\b",
         r"\bquotas?\b", r"\bbuy[\s\-]?outs?\b", r"\bbuy(?:s|ing)?\s+(?:\S+\s+){0,2}?out\b",
         r"\bwater[\s\-]+saving\b", r"\bsav(?:e|es|ing)\s+water\b", r"\bconserv\w*", r"\bcaps?\s+on\b",
         "改用", "替代", "配額", "凍結", "節約用水", "節約", "收購", "地面水", "總量管制", "管制",
         "少用水", "關井", "封井", "休耕", "停灌", "節水", "限水", "限抽", "少用"]
PRONOUN = [r"\b(?:it|its|they|their|them|there|locally)\b", "也", "亦", "當地", "該縣", "他們", "它", "那邊",
           "那裡", "那裏", "自己",
           # round 10, second review: demonstratives ("those farmers", "the farmers in question", 那些農民)
           r"\b(?:those|these|such|said)\b", r"\bin\s+question\b", r"\bconcerned\b",
           "那些", "這些", "那個", r"(?<!應)該(?!不|會|是|怎)", r"(?<![因如從彼])此(?!外|時)", "彼"]
CLAUSE = r"[。；;？?！!，,:：]|\band\b|\bwhile\b|\bbut\b|\bthen\b|並且|而且|以及|同時"


def _clauses(s: str) -> list[str]:
    return [c for c in re.split(CLAUSE, s, flags=re.I) if c.strip()]


# ---------------------------------------------------------------------------------------------
# Changhua
# ---------------------------------------------------------------------------------------------
CHANGHUA_WORD = (r"\b(?:chang|zhang)[\s\-_']*hua\b|\bchanghwa\b|\bchang[\s\-]*hwa\b|彰\s*化"
                 # third review: other names of Changhua's ground (Shoka, Changbin, Bagua, the Dadu)
                 r"|\bshoka\b|\bchang[\s\-]?bin\b|彰濱|八卦山|\bba[\s\-]?gua(?:\s*(?:shan|mountain|plateau))?\b"
                 r"|\bdadu\b|大肚溪|烏溪")
_CH_T = r"(?:(?:chang|zhang)[\s\-_']*hua|changhwa|彰化)(?:'s)?(?:\s*county|縣)?"
CHOUSHUI = (r"(?:choushui|chuoshui|zhuoshui|jhuoshuei|jhuoshui|choshui|cho[\s\-]shui|chou[\s\-]shui|"
            r"zhuo[\s\-]shui|jhuo[\s\-]shuei)(?:[\s\-]*(?:river|creek|hsi|xi))?")
RIVER = rf"(?:the\s+)?(?:{CHOUSHUI}|river)"
GEO_EN = [rf"\bnorth(?:ern)?\s+(?:of|side\s+of|bank\s+of|banks\s+of|shore\s+of)\s+{RIVER}",
          rf"\b{CHOUSHUI}(?:'s)?\s+(?:north(?:ern)?|far|other|opposite)\s+(?:bank|side|shore)",
          rf"\b(?:far|other|opposite|northern|north)\s+(?:bank|side|shore)\s+of\s+{RIVER}",
          rf"\bacross\s+{RIVER}", rf"\bboth\s+(?:sides|banks)\s+of\s+{RIVER}",
          rf"\bon\s+either\s+side\s+of\s+{RIVER}",
          r"\bboth\s+counties\b", r"\btwo\s+counties\b", r"\bneighbo(?:u)?ring\s+count(?:y|ies)\b",
          r"\bnorth(?:ern)?\s+neighbo(?:u)?r", r"\bcounty\s+(?:to\s+the\s+)?north\b",
          r"\bnorth(?:ern)?\s+county\b", r"\bcounty\s+(?:above|north\s+of|to\s+the\s+north\s+of)\s+yun",
          r"\b(?:above|north\s+of)\s+yun[\s\-]*lin\b", r"\byun[\s\-]*lin\s+and\s+(?:its|the)\s+neighbo(?:u)?r",
          r"\bits\s+neighbo(?:u)?r(?:ing)?\b",
          # round 10: "the north bank" alone, but not a named Yunlin river's (Beigang, Huwei ...)
          r"\b(?:north(?:ern)?|far)\s+(?:bank|shore)\b(?!\s+of\s+(?:the\s+)?(?:new\s+|xin[\s\-]?)?"
          r"(?:beigang|peikang|huwei|hu[\s\-]wei|xinhuwei|hsinhuwei|yunlin|sanjie|pei[\s\-]kang|xin[\s\-]huwei)\b)",
          r"\bnorth(?:ern)?\s+(?:half|part|side|portion|end)\s+of\s+the\s+(?:alluvial\s+)?fan\b",
          r"\bfan'?s\s+north(?:ern)?\s+(?:half|part|side|portion)"]
GEO_ZH = [r"濁水(?:溪)?(?:以|的)?北", r"濁水溪(?:的)?(?:對岸|兩岸|彼岸|另一邊|另一側)",
          r"(?<![一-鿿])溪北|[在的]溪北", r"(?:兩|二)(?:個)?縣", r"彰雲|雲彰",
          r"鄰縣|隔壁(?:的)?縣", r"北(?:邊|方|側)的縣", r"雲林(?:以北|北邊|北方|的北邊)", r"雲林(?:和|與|跟|及)(?:其)?鄰",
          r"(?:沖積扇|扇)(?:的)?北(?:半部|部|側|邊|段)"]
# the whole fan, which includes Changhua: a Changhua lever only when a lever is asked
WHOLE_FAN = [r"\b(?:choushui|zhuoshui)\s+plain\b", r"\bplain\b", r"\beverywhere\b(?!\s+in\s+yun)",
             r"\b(?:up|down|along)\s+the\s+coast\b", "平原", "到處", "各地", "全區域",
             r"\b(?:whole|entire|full)\s+(?:of\s+the\s+)?(?:choushui\s+)?(?:alluvial\s+)?fan\b",
             r"\b(?:across|throughout|over|all\s+of)\s+the\s+(?:whole\s+|entire\s+)?(?:choushui\s+)?alluvial\s+fan\b",
             r"\bfan[\s\-]*wide\b", "全扇區", "全扇", r"整個(?:濁水溪)?(?:沖積)?扇", "全沖積扇", "沖積扇全區", "扇區全部"]
# Changhua County's townships (the page pools them into one row), in romanisations in use
# (Hanyu, Wade-Giles, Tongyong; "-" stands for an optional space or hyphen). Sihu is left out:
# it is also Yunlin's 四湖. Bare two-character names that are also common words (田中, 大城,
# 花壇, 和美, 大村, 福興) count only with their 鄉/鎮 suffix; 北斗 never as the satellite system.
CH_TOWNS_ZH = ["鹿港", "線西", "伸港", "秀水", "芬園", "員林", "溪湖", "埔鹽", "埔心", "永靖", "社頭", "二水",
               "二林", "田尾", "埤頭", "芳苑", "竹塘", "溪州", "和美鎮", "福興鄉", "花壇鄉", "田中鎮", "大村鄉",
               "大城鄉", "北斗(?!衛星|系統|導航|星)"]
CH_TOWNS_EN = ["lu-kang", "lu-gang", "he-mei", "ho-mei", "xian-xi", "hsien-hsi", "sian-si", "shen-gang",
               "shen-kang", "fu-xing", "fu-hsing", "xiu-shui", "hsiu-shui", "siou-shuei",
               "hua-tan", "fen-yuan", "yuan-lin", "xi-hu", "hsi-hu", "tian-zhong", "tien-chung",
               "tian-jhong", "da-cun", "ta-tsun", "pu-yan", "pu-yen", "pu-xin", "pu-hsin", "pu-sin",
               "yong-jing", "yung-ching", "she-tou", "er-shui", "erh-shui", "er-shuei", "bei-dou",
               "pei-tou", "er-lin", "erh-lin", "tian-wei", "tien-wei", "pi-tou", "fang-yuan", "da-cheng",
               "ta-cheng", "zhu-tang", "chu-tang", "jhu-tang", "xi-zhou", "hsi-chou", "si-jhou", "xi-chou"]
_CH_TOWNS_EN_RE = re.compile(r"\b(?:" + "|".join(n.replace("-", r"[\s\-']?") for n in CH_TOWNS_EN)
                             + r")\b(?!\s+(?:satellite|system|gnss|navigation|positioning))", re.I)
# the one Changhua question that is answered: the pooled township row's labelled spillover of a
# Yunlin policy, asked as that row (township / row / 鄉鎮 / 列)
POOLED_ROW = [r"\brows?\b", r"\bpanel\b", r"\btownships?\b", r"\btowns?\b", r"not\s+yet\s+split",
              "列", "鄉鎮", "未分"]

def changhua_ref(s: str) -> frozenset[str]:
    """How the question reaches Changhua: ``word`` (the name itself), ``town`` (one of its
    townships), ``geo`` (north of the Choushui, both counties ...), ``fan`` (the whole fan);
    empty when it does not."""
    s = norm(s)
    out = set()
    if re.search(CHANGHUA_WORD, s, re.I):
        out.add("word")
    if any(re.search(p, s) for p in CH_TOWNS_ZH) or _CH_TOWNS_EN_RE.search(s):
        out.add("town")
    if _has(GEO_EN + GEO_ZH, s):
        out.add("geo")
    if _has(WHOLE_FAN, s):
        out.add("fan")
    return frozenset(out)


def has_lever(s: str) -> bool:
    return _has(LEVER, norm(s))


# both counties under one lever ("Yunlin and Changhua", 雲林和彰化都 ...) or Changhua following
# suit ("and Changhua too", "so does Changhua", 彰化也一樣, 彰化比照) — the lever is Changhua's too
_CH = r"(?:(?:chang|zhang)[\s\-_']*hua|changhwa|彰化)(?:'s)?(?:\s*county|縣)?"
_YU = r"(?:yun[\s\-]*lin|雲林)(?:'s)?(?:\s*county|縣)?"
_JOIN = (r"\s*(?:,?\s*and|&|plus|as\s+well\s+as|along\s+with|together\s+with|,|和|與|跟|及|以及|、|同)"
         r"\s*(?:(?:in|for|across|to)\s+(?:the\s+)?)?")          # "in Yunlin and in Changhua"
BOTH = [rf"\b{_YU}{_JOIN}{_CH}", rf"{_CH}{_JOIN}{_YU}", rf"\bboth\b[^.?!;]{{0,40}}{_CH}",
        rf"{_CH}[^.?!;]{{0,40}}\bboth\b", rf"(?:{_YU}|{_CH})[^。？?！!；;]{{0,12}}都",
        "兩縣", "兩個縣", "雙方", "彰雲", "雲彰"]
ALSO = [rf"{_CH}\s*,?\s+(?:too|as\s+well|also|likewise)\b", rf"\b(?:so|as)\s+(?:does|do|did|will|would)\s+{_CH}",
        rf"{_CH}\s*[:：,，\-–—]?\s*(?:the\s+)?same\b",                     # "Changhua: same"
        rf"\b(?:appl\w*|extend\w*|imposed|introduced|rolled\s+out)\s+(?:also\s+|too\s+|as\s+well\s+)?"
        rf"(?:to|in|on|across)\s+(?:the\s+)?{_CH}",                       # "applied to Changhua"
        rf"{_CH}\s+(?:does|did|will|would)\s+(?:the\s+same|likewise|too)\b",
        rf"{_CH}\s+(?:matches|matched|matching|follows\s+suit|joins\s+in|copies)\b",
        rf"\b(?:matched|matching|equal|equivalent|same|identical|parallel|similar)\s+(?:\S+\s+){{0,3}}?"
        rf"(?:in|for|by|across)\s+(?:the\s+)?{_CH}",
        rf"{_CH}(?:也|亦|同樣|一樣|跟進|照做|比照|同步|跟著)", rf"(?:也|同樣|一樣|比照|跟進)[^，,。？?]{{0,4}}{_CH}",
        # round 10 fix (review 2): Changhua follows, adopts, mirrors, copies or imitates it
        rf"{_CH}\s+(?:\S+\s+){{0,2}}?(?:follow|follows|followed|adopt\w*|mirror\w*|cop(?:y|ies|ied|ying)|"
        rf"imitat\w*|replicat\w*|emulat\w*|repeat\w*)\b",
        rf"\b(?:followed|adopted|mirrored|copied|imitated|replicated|emulated|repeated|matched)\s+(?:also\s+)?"
        rf"(?:by|in)\s+(?:the\s+)?{_CH}",
        rf"{_CH}[^，,。？?；;]{{0,6}}(?:照著|照樣|仿照|仿效|效法|學雲林|跟雲林)"]
_PREP_TO_COUNTY = re.compile(r"^\s*(?:\S+\s+){0,4}?(?:in|at|for|across|by|throughout|within|of|from|on|among|"
                             r"(?:appl\w*|extend\w*|imposed|introduced)\s+(?:also\s+)?(?:to|in|on))"
                             r"\s+(?:the\s+)?$|^\s*$", re.I)


def _cjk(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


def _yunlin_reflexive(cl: str, m) -> bool:
    """Round 10 (A1): 自己 right after 雲林 (雲林縣自己的魚塭) is Yunlin's own, not a pronoun."""
    return m.group(0) == "自己" and bool(re.search(r"雲林(?:縣)?(?:的)?\s*$", cl[:m.start()]))


def _zh_target(cl: str, m) -> bool:
    """Round 10 (A1): in Chinese a county after the lever and right after 對 / 影響(到) is the
    lever's target (雲林灌溉減三成對彰化有什麼影響), not its owner."""
    return bool(re.search(r"(?:對|影響到?)\s*$", cl[:m.start()]))


def changhua_owns_a_lever(s: str) -> bool:
    """Some lever in the question is not Yunlin's.

    - Both counties under one lever (BOTH) or Changhua following suit (ALSO), when a lever is
      asked at all.
    - Otherwise each lever is owned by the nearest county or pronoun in its clause, measured
      from the lever word itself. In English a county that comes after the lever owns it only
      when it is its object or place ("cut Changhua irrigation", "a cut in Changhua"), not a
      target ("a Yunlin cut does to Changhua's row"); in Chinese the nearest wins either way.
      On a tie Changhua (or a pronoun) wins.
    - A lever in a clause with no county is owned by the last county named before it; with
      none, it is not Yunlin's. Clauses split on punctuation (also ，：) and and / while / 並且 ..."""
    s = norm(s)
    if not has_lever(s):
        return False
    if _has(BOTH + ALSO, s):
        return True
    last = None                                   # the county last named in an earlier clause
    for cl in _clauses(s):
        levs = [m for p in LEVER for m in re.finditer(p, cl, re.I)]
        cty = ([(m, "y") for p in YUNLIN for m in re.finditer(p, cl, re.I)]
               + [(m, "c") for m in re.finditer(CHANGHUA_WORD, cl, re.I)]
               + [(m, "c") for p in PRONOUN for m in re.finditer(p, cl, re.I)
                  if not _yunlin_reflexive(cl, m)])
        for lv in levs:
            zh = _cjk(cl[lv.start()])
            near = []
            for m, k in cty:
                if m.end() <= lv.start():                         # the county comes first
                    near.append((lv.start() - m.end(), k == "y", k))
                elif m.start() >= lv.end() and ((zh and not _zh_target(cl, m))
                                                or _PREP_TO_COUNTY.match(cl[lv.end():m.start()])):
                    near.append((m.start() - lv.end(), k == "y", k))      # it comes after, as object
            if near and min(near)[2] != "y":
                return True
            if not near and last != "y":
                return True
        if cty:
            last = max((m.start(), k) for m, k in cty)[1]        # a pronoun counts as Changhua
    return False


# Changhua's own pumping, classes or levers named right after it ("Changhua's irrigation", 彰化的養殖)
CH_CLASS_NEAR = [r"\b(?:chang|zhang)[\s\-_']*hua(?:'s)?\s+(?:county'?s?\s+)?(?:pumping|irrigation|aquaculture|"
                 r"livestock|industry|industrial|domestic|wells?|farms?|farmers|fish\w*|levers?|cuts?|agricultur\w*)",
                 r"彰化(?:縣)?(?:的)?(?:灌溉|養殖|畜牧|工業|抽水|農業|民生|槓桿|農田|魚塭|減抽)"]


def pooled_row_ok(s: str) -> bool:
    """The pooled Changhua row asked by its name, as a row, under an explicitly Yunlin policy:
    no Changhua township or geography, no head word (C13a), none of Changhua's own classes named,
    and no lever that is Changhua's (:func:`changhua_owns_a_lever`)."""
    s = norm(s)
    return (changhua_ref(s) == {"word"} and _has(POOLED_ROW, s) and _has(YUNLIN, s)
            and not _has(HEAD_WORDS, s) and not _has(CH_CLASS_NEAR, s) and not changhua_owns_a_lever(s))


CLASS_ANY = [r"irrigat\w*", r"agricultur\w*", r"\bfarm\w*", r"\bcrops?\b", r"aquacultur\w*", r"\bfish\w*",
             r"livestock", r"\banimal\w*", r"\bpigs?\b", r"poultry", r"domestic", r"household\w*",
             r"industr\w*", r"factor(?:y|ies)", r"other\s+(?:use|pumping|class)\w*",
             "灌溉", "農業", "農田", "養殖", "魚塭", "漁業", "畜牧", "養豬", "畜產", "民生", "家用", "工業", "工廠"]
ALL_CLASSES = [r"\ball\s+(?:pumping|classes|uses|water)", r"\bevery\s+class", r"across\s+the\s+board",
               r"\ball[\s\-]*class", "全部類別", "所有類別", "全部用水", "所有用水", "各類別", "全部抽水", "所有抽水"]


def changhua_kind(s: str) -> str:
    """Which Changhua lever a refused question reads: ``subsidence`` (C13c), ``class`` (one
    water-use class, C13b) or ``head`` (the all-class head lever, C13a, the one Changhua lever a
    registered test can still decide)."""
    s = norm(s)
    if subsidence_words(s, units=False) or (subsidence_words(s) and not _has(HEAD_WORDS, s)):
        return "subsidence"
    if _has(CLASS_ANY, s) and not _has(ALL_CLASSES, s):
        return "class"
    return "head"


# ---------------------------------------------------------------------------------------------
# subsidence size
# ---------------------------------------------------------------------------------------------
SUBS_WORDS = [r"subsid\w*", r"\bsink\w*", r"\bsank\b", r"\bsunk\w*", r"\bsettl(?:e|es|ed|ing|ement)\b",
              r"\bcompact(?:ion|s|ed|ing)?\b", r"\bdowndrop\w*", r"\buplift\w*",
              "下陷", "沉陷", "沉降", "下沉", "少沉", "多沉", "沉多少", "沉了", "會沉", "沉得",
              "地層", "地盤", "壓密", "塌陷", "地面變形", "地表變形", "地層變形"]
SUBS_UNITS = [r"\bcm\b", r"centimet\w*", r"\bmm\b", r"millimet\w*", r"\binch(?:es)?\b",
              "公分", "厘米", "釐米", "毫米", "公釐", "公厘",
              r"\d\s*(?:cm|mm)\b"]                  # round 10: "3cm" (no word boundary inside)
# the ground surface (never "groundwater" / "ground water" / "surface water") moving down or back
# up; "on the ground" (in practice) is removed before matching
GROUND = (r"(?:\b(?:ground|terrain)\b(?!\s*-?\s*water)|\bland[\s\-]+(?:surface|level|elevation)\b|"
          r"\bsurface\s+elevation\b)")
SUBS_MOVE = [rf"{GROUND}(?:\s+(?:level|surface|elevation))?[^.?!;,]{{0,30}}?\b(?:drop\w*|fall\w*|fell|lower\w*|"
             r"go(?:es)?\s+down|went\s+down|come\s+down|subsid\w*|settl\w*|sink\w*|sank|rebound\w*|rise|rises|"
             r"rising|rose|come\s+back\s+up|comes\s+back\s+up|recover\w*|lift\w*|deform\w*)",
             r"\bland\b[^.?!;,]{0,25}?\b(?:go(?:es)?\s+down|went\s+down|gone\s+down|come\s+down|sink\w*|sank|"
             r"settl\w*|subsid\w*|deform\w*)",
             r"\b(?:ground|land|surface)[\s\-]+deformation\b",
             r"(?:地面|地表(?!水)|地盤|地層)[^，,。？?；;]{0,8}(?:下降|降低|沉|陷|回升|抬升|上升|回彈|變低|變形)"]
DIFFERENTIAL = [r"differential\s+(?:settlement|subsidence|sinking)", r"angular\s+distortion",
                r"\bdistortion\b", "差異沉陷", "差異沉降", "不均勻沉陷", "角變量", "角變形"]
RAIL_WORDS = [r"\bhsr\b", r"high[-\s]speed\s+rail\w*", r"\bthsr\b", r"\brail\w*", r"\btrain\w*", "高鐵", "鐵路"]
# the bare words size / number ("is the size of the response tested?") ask about a size without
# asking for it; every other MAGNITUDE pattern asks for it
SIZE_WORD = r"\b(?:size|number)\s+of\b"
SIZE_ADJ = [r"\b(?:large|larger|small|smaller|big|bigger|significant|substantial|sizable|sizeable|tiny|huge|"
            r"meaningful|material|negligible)\b", "很大", "很小", "大不大", "顯著", "明顯", "可觀", r"\d+\.\d+"]
MAGNITUDE = [r"how\s+(?:much|many|big|large|deep|fast|far\s+(?:down|up))\b", r"\bby\s+how\s+much\b",
             r"\b(?:amount|magnitude|figure|estimate|value)\s+of\b", SIZE_WORD, r"\bin\s+(?:cm|mm|centimet\w*)\b",
             "多少", "幾公分", "多大", "多深", "多快", "幾釐米",
             # an imperative to show it is in WHAT (it blocks the how-sure exemption, not the rail one)
             "給我", "告訴我", "直接", "列出", "輸出", "印出", "顯示"] + SUBS_UNITS
META = [r"how\s+(?:sure|confident|reliable|trustworthy|certain|well\s+tested|far\s+(?:is|are)\b)",
        r"\btest(?:ed|s|ing)?\b", r"\bvalidat\w*", r"\bverif\w*", r"\bwithdrawn?\b", r"\bsupported\b",
        r"\bevidence\b", r"\btrust\w*", r"\bconfiden\w*", r"\breliab\w*", r"\buncertain\w*",
        r"\bwhy\s+(?:is|are|isn't|aren't|was|can't|cannot|won't|don't|do\s+you\s+not|no|not)\b",
        r"\bscored\b",
        "可信", "可靠", "多準", "信心", "驗證", "檢驗", "測試", "撤回", "為什麼", "為何", "證據", "支持", "不確定",
        "評分", "登錄",
        # round 10 (A2): checked against the leveling, did it hold up, 能不能當真 ...
        r"\bcheck(?:ed|s)?\b", r"\bcompared\s+(?:with|to|against)\b", r"\b(?:held|hold|holds)\s+up\b",
        r"\bconfirm\w*", r"\bcorroborat\w*", r"\bbacked\s+up\b", r"\bground[\s\-]*truth\w*",
        r"\blevel(?:l)?ing\b", r"\baccura\w*", r"\bevaluat\w*", r"\bassess\w*", r"\bprove[nd]?\b",
        "檢查", "核對", "比對", "對照", "查證", "證實", "經得起", "當真", "準不準", "準確", "靠得住", "信得過",
        "驗過", "測過", "檢核", "評估", "水準測量"]
# where the size is asked: overrides the named-township and pooled-row exemptions, the rail
# exemption and the how-sure exemption. Bare whole / entire / total / overall / 總 / 平均 / 整個 /
# 整體 count only next to a county or fan word: "Tuku's total" is still Tuku's row.
_WIDE = r"(?:county|fan|alluvial|yun[\s\-]*lin|region)"
SCOPE_AREA = [r"\bfan\b", r"\bfan[\s\-]*(?:average|wide|total|mean)\b", r"\balluvial\b",
              r"\bcounty[\s\-]*wide\b", r"\b(?:across|all\s+of|over)\s+the\s+county\b",
              rf"\b(?:whole|entire|total|overall)\s+(?:of\s+)?(?:the\s+)?(?:\S+\s+){{0,2}}?{_WIDE}",
              rf"\b{_WIDE}(?:'s)?\s+(?:\S+\s+){{0,2}}?(?:total|overall|average|mean|as\s+a\s+whole)\b",
              r"\bregion[\s\-]*wide\b", r"\b(?:all|every|each)\s+(?:the\s+)?townships?\b",
              "全縣", "全雲林", "沖積扇", "全扇", "扇區", "全區", "各鄉鎮", "所有鄉鎮", "全部鄉鎮", "縣內", "全域",
              r"(?:總|平均|整個|整體)[^，,。？?；;]{0,3}(?:縣|沖積扇|扇區|扇|雲林)",
              r"(?:縣|沖積扇|扇區|雲林)[^，,。？?；;]{0,4}(?:總|平均|整體)"]
# a how-sure question: a yes/no or how-sure form ...
YESNO = [r"^\s*(?:please\s+)?(?:is|are|has|have|was|were|does|do|did|can|could|will|would|should|"
         r"how\s+(?:sure|confident|reliable|certain|well|far\s+(?:is|are|has))|why|when|"
         r"tell\s+(?:me|us)\s+(?:whether|if))\b",
         r"[嗎][？?]?\s*$", "有沒有", "是否", "為什麼", "為何", "何時", "什麼時候",
         # round 10 (A2): whether ... anywhere, and the Chinese A-not-A question (能不能, 是不是)
         r"\bwhether\b", r"\bwonder\w*\s+if\b", r"([一-鿿])[不沒]\1"]
# ... that does not ask what the quantity is, or for it to be shown (tell keeps its whether / if)
WHAT = [r"\bwhat(?:'s|\s+is|\s+are|\s+was|\s+were|\s+would\s+be|\s+will\s+be)\b"
        r"(?!\s+(?:the\s+)?(?:evidence|status|test|basis|reason))",
        r"\b(?:give|tell|show|state|quote|print|output|reveal|dump|display|list|report)\b"
        r"(?!\s+(?:me\s+|us\s+)?(?:whether|if)\b)",
        "是多少", "有多少", "多少", "給我", "告訴我", "列出", "輸出", "印出", "顯示"]


def _ground(s: str) -> str:
    return re.sub(r"\bon\s+the\s+ground\b", " ", s, flags=re.I)


def subsidence_words(s: str, *, units: bool = True) -> bool:
    """The question talks about the ground sinking (with ``units``: or about cm / 公分)."""
    s = _ground(norm(s))
    return _has(SUBS_WORDS + SUBS_MOVE + (SUBS_UNITS if units else []), s)


def area_scope(s: str) -> bool:
    return _has(SCOPE_AREA, norm(s))


# round 10 (A2): "減少下陷量" / "subsidence reduction" names the avoided subsidence, not a lever;
# taken out before the lever check of how_sure_only only (a real lever word anywhere still counts)
SUBS_REDUCTION = [r"\breduc\w*\s+(?:in\s+|of\s+)?(?:the\s+)?(?:land\s+|ground\s+)?(?:subsid\w*|sinking|settlement)",
                  r"\b(?:subsid\w*|sinking|settlement)\s+reduc\w*",
                  r"減少(?:的)?(?:地層)?(?:下陷|沉陷|沉降|下沉)(?:量)?", r"(?:下陷|沉陷|沉降|下沉)(?:的)?減少(?:量)?"]
# a bare number (not a year) or a number glued to cm / mm asks a size inside a how-sure form
# a size asked by a sign, zero, a fraction or a multiple (round 10, inside a how-sure form)
SIZE_WORDS_R10 = [r"\b(?:positive|negative|zero|non[\s\-]?zero|nil)\b",
                  r"\b(?:half|halves|halved|halving|double\w*|twice|thirds?|quarters?|percent|fraction)\b",
                  "一半", "減半", "倍", r"[一二兩三四五六七八九十]\s*成", "百分之", "正值", "負值", "為零", "是零", "零"]
BARE_NUMBER = [r"(?<![A-Za-z0-9_.])(?!(?:19|20)\d\d(?!\d))\d+(?:\.\d+)?(?!\d)", r"\d\s*(?:cm|mm)\b",
               r"[一二兩三四五六七八九十半]\s*(?:公分|厘米|釐米|毫米|公釐|公厘)"]


# round 10 fix (review 3): what else ends the how-sure exemption. A place (Yunlin or the county as
# the place of the size), a policy or scenario, a size adjective, a zero / stop verb, a comparison,
# a request to work the number out, a spelled number, or a length unit
WHAT_R10 = [r"\b(?:calculat\w*|comput\w*|summari[sz]\w*|quantif\w*|work\s+out)\b",
            r"(?<!the\s)(?<!this\s)(?<!that\s)(?<!its\s)(?<!subsidence\s)\bestimate\b",
            r"\b(?:say|need|needs|like|want|wants|please|state)\b",
            # 算 as a request (算一下, 幫我算), not 模型算出來的 (what the model computed)
            r"算一下", r"算算", r"(?:幫我|請|你|再)算", r"算給", r"計算一下", r"(?<![主重需只])要", "想知道", "說說",
            "講一下"]
HOW_SURE_VETO = (YUNLIN + [r"\bcount(?:y|ies)\b", "縣"]
                 + [r"\bpolic(?:y|ies)\b", r"\bscenarios?\b", r"\bmeasures?\b", "政策", "方案", "情境", "措施"]
                 + [r"\b(?:considerabl\w*|minor|noticeabl\w*|appreciabl\w*|modest|marginal|measurabl\w*|"
                    r"slight\w*|major|dramatic\w*|notabl\w*|minimal|serious\w*|sever\w*|strong\w*|weak\w*|"
                    r"great\w*|massive|enormous|little|lot|much|real)\b", "嚴重", "很多", "不少", "有限", "微小"]
                 # second review: more size words, ordinal fractions, any digit, magnitude
                 + [r"\b(?:trivial|insignificant|perceptibl\w*|imperceptibl\w*|pronounced|detectabl\w*|"
                    r"undetectabl\w*|noise|noticeabl\w*|magnitude|enough|matter\w*|nothing|nearly|almost|"
                    r"fifths?|sixths?|sevenths?|eighths?|ninths?|tenths?|hundredths?|thousandths?)\b", r"\d",
                    "算不算", "夠不夠", "忽略", "有感", "感覺", "數量級",
                    # third review
                    r"\b(?:moderate\w*|minuscule|miniscule|hefty|paltry|tiny|decimet\w*|coins?|thickness|"
                    r"overestimat\w*|underestimat\w*|over-?stat\w*|under-?stat\w*|exaggerat\w*|range)\b",
                    "偏大", "偏小", "微不足道", "輕微", "毫釐", "公寸", "高估", "低估", "誇大", "範圍", "壹", "貳", r"參(?!數|考|與|加|照)", "肆", "伍", "陸", "柒", "捌",
                    "玖", "拾"]
                 + [r"\bhalt\w*", r"\bends?\b", r"\bended\b", r"\bending\b", r"\bceas\w*", r"\bvanish\w*",
                    r"\bdisappear\w*", r"\bgone\b", "不再", "消失", "結束", "停止"]
                 + [r"\bthan\b", r"\bexceed\w*", r"\bsurpass\w*", r"\b(?:faster|slower|quicker)\b",
                    r"\blimits?\b", r"\bthresholds?\b", r"\b(?:above|below|beyond|within)\s+(?:the\s+)?"
                    r"(?:measurement\s+)?(?:error|noise)", "超過", "大於", "小於", "高於", "低於", "門檻", "上限", "比(?!對)"]
                 + [r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|dozen|hundred)\b",
                    r"[一二兩三四五六七八九十][點点]", r"[點点][一二兩三四五六七八九十]"]
                 + [r"\bmet(?:er|re)s?\b", r"\bfeet\b", r"\bfoot\b", "公尺", "米", "寸", "英寸"]
                 # any other place than the question as a whole: coast, inland, a zone, the rail
                 + [r"\bcoast\w*", r"\binland\b", r"\bzones?\b", r"\bregions?\b", r"\bareas?\b",
                    r"\balong\b", r"\bnear\b", r"\bnorth\w*", r"\bsouth\w*", r"\beast\w*", r"\bwest\w*"]
                 + RAIL_WORDS + ["沿海", "海邊", "內陸", "地區", "區域", "沿線", "附近", "東部", "西部", "南部", "北部"]
                 # a water-use class (a class lever's subsidence), a row or panel, a year
                 + CLASS_ANY + [r"\brows?\b", r"\bpanel\b", "列", r"(?<!\d)(?:19|20)\d\d(?!\d)"])


def how_sure_only(s: str) -> bool:
    """Asks how far the subsidence result is tested, not what it is: how-sure wording (META) in a
    yes/no or how-sure form (YESNO), with no 'what is', no give / show / print / list, no 多少
    (WHAT), no lever, no area-scope word, and no size asked except by the bare words size /
    number (no size adjective, no decimal number either). Round 10 (A2): no bare number either,
    and the avoided-subsidence noun (減少下陷量, subsidence reduction) is not read as a lever; after
    the review, none of HOW_SURE_VETO (Yunlin or the county, a policy or scenario, size adjectives,
    zero verbs, comparisons, spelled numbers, length units) and none of WHAT_R10 either."""
    s = norm(s)
    asks = [p for p in MAGNITUDE if p != SIZE_WORD] + SIZE_ADJ + BARE_NUMBER + SIZE_WORDS_R10
    unlevered = s
    for p in SUBS_REDUCTION:
        unlevered = re.sub(p, " ", unlevered, flags=re.I)
    return (_has(META, s) and _has(YESNO, s) and not _has(WHAT + WHAT_R10, s)
            and not has_lever(unlevered) and not area_scope(s) and not _has(asks + HOW_SURE_VETO, s))


def subsidence_size(s: str) -> bool:
    """A question about how much the ground sinks / settles / rebounds. Exempt here: the rail's
    differential settlement (C06) when no size is asked and no area-scope word is present, and a
    :func:`how_sure_only` question. The caller adds the named-township and pooled-row exemptions,
    which an area-scope word overrides."""
    s = norm(s)
    if not subsidence_words(s):
        return False
    asks_size, scope = _has(MAGNITUDE + SIZE_ADJ, s), area_scope(s)
    # the rail exemption covers the differential settlement only: once those words are taken
    # out, nothing else may speak of the ground sinking ("the avoided subsidence and the HSR
    # differential settlement" is still a size question)
    rest = s
    for d in DIFFERENTIAL:
        rest = re.sub(d, " ", rest, flags=re.I)
    if (_has(RAIL_WORDS, s) and _has(DIFFERENTIAL, s) and not asks_size and not scope
            and not subsidence_words(rest)):
        return False
    return not how_sure_only(s)


# ---------------------------------------------------------------------------------------------
# beyond the page horizon
# ---------------------------------------------------------------------------------------------
EN_NUM = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
          "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
          "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
          "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
          "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100, "a hundred": 100,
          "a couple of": 2, "couple of": 2, "a dozen": 12, "dozen": 12, "a score of": 20}
_EN_N = r"(\d{1,3}|" + "|".join(sorted((re.escape(k) for k in EN_NUM), key=len, reverse=True)) + r")"
_ZH_DIG = {"〇": 0, "零": 0, "一": 1, "二": 2, "兩": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7,
           "八": 8, "九": 9}
_ZH_N = r"((?<!\d)\d{1,3}(?!\d)|[〇零一二兩三四五六七八九十百]+|數十|幾十|十幾)"
EN_LEAD = [  # (pattern, span): a lead reports at base + N; a span of N years ends at base + N - 1
    (rf"\b{_EN_N}[\s\-]+years?\s+(?:later|on|out|ahead|hence|into|in|down\s+the\s+(?:road|line)|"
     rf"after|from|past|since|beyond)\b", False),
    (rf"\b(?:in|after)\s+(?:the\s+)?(?:next\s+|coming\s+|following\s+|another\s+)?{_EN_N}[\s\-]+years?\b",
     False),
    (rf"\b(?:within|over|for|across|during)\s+(?:the\s+)?(?:next\s+|coming\s+|following\s+|another\s+)?"
     rf"{_EN_N}[\s\-]+years?\b", True),
    (rf"\b(?:next|coming|following|another)\s+{_EN_N}[\s\-]+years?\b", True),
    (rf"\b{_EN_N}[\s\-]+years?\s+of\b", True),
    (rf"\b{_EN_N}[\s\-]+year\s+(?:horizon|lead|mark|point|run|outlook|projection)\b", False),
    (rf"\b{_EN_N}[\s\-]+year\s+(?:period|span|window)\b", True),
    (rf"\byear\s+{_EN_N}\s+(?:of|after|into|from|since)\b", True),      # "year 10 of the cut"
    (rf"\bthe\s+{_EN_N}(?:st|nd|rd|th)?\s+year\s+(?:of|after|into|from)\b", True),
]
_ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
        "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12, "fifteenth": 15,
        "twentieth": 20}
EN_DECADES = [(r"\bhalf\s+a\s+century\b", 50), (r"\b(?:a|one)\s+century\b", 100),
              (r"\b(?:several|a\s+few|a\s+couple\s+of|couple\s+of|many)\s+decades\b", 20),
              (r"\bdecades\b", 20), (rf"\b{_EN_N}\s+decades?\b", None),
              (r"\b(?:a|one)\s+decade\b", 10), (r"\bthe\s+next\s+decade\b", 10),
              (r"\bdecade\s+(?:later|from|after|on|out)\b", 10)]
FAR = [r"\bmid[-\s]?century\b", r"\bend\s+of\s+(?:the\s+)?century\b",
       r"\b(?:after|beyond|past)\s+the\s+page\s+(?:ends|stops|finishes|runs\s+out|is\s+over)\b",
       r"\b(?:after|beyond|past)\s+(?:the\s+)?(?:page'?s?\s+)?(?:horizon|last\s+year)\b",
       r"\b(?:after|beyond|past)\s+(?:the\s+)?end\s+of\s+the\s+(?:page|run|projection|horizon)\b",
       "本世紀中", "本世紀末", "半世紀", "百年後", "下個世紀", "頁面結束後", "頁面之後", "超出頁面", "頁面以後"]
ZH_LEAD = [  # (pattern, offset): reports at base + N + offset
    (rf"{_ZH_N}\s*(?:多)?\s*年\s*(?:後|之後|以後|過後)", 0),
    (rf"{_ZH_N}\s*(?:多)?\s*年\s*(?:內|間|期間)", -1),
    (rf"(?:再過|往後|之後)\s*的?\s*{_ZH_N}\s*(?:多)?\s*年", 0),
    (rf"(?:未來|接下來)\s*的?\s*{_ZH_N}\s*(?:多)?\s*年", -1),
    (rf"第\s*{_ZH_N}\s*年", -1),
]
_EN_UNITS = {k: v for k, v in EN_NUM.items() if v < 10 and " " not in k and k not in ("a", "an")}
_EN_TEENS_TENS = {k: v for k, v in EN_NUM.items() if 10 <= v < 100 and " " not in k}


def zh_number(t: str) -> int | None:
    if t.isdigit():
        return int(t)
    if t in ("數十", "幾十"):
        return 20
    if t == "十幾":
        return 11
    if t in ("百", "一百"):
        return 100
    if "十" not in t and "百" not in t:            # digit by digit: 一二五 = 125
        try:
            return int("".join(str(_ZH_DIG[ch]) for ch in t))
        except KeyError:
            return None
    total, cur = 0, 0
    for ch in t:
        if ch == "百":
            total += (cur or 1) * 100
            cur = 0
        elif ch == "十":
            total += (cur or 1) * 10
            cur = 0
        elif ch in _ZH_DIG:
            cur = _ZH_DIG[ch]
        else:
            return None
    return total + cur


def _en_number(t: str) -> int | None:
    t = t.lower().strip()
    return int(t) if t.isdigit() else EN_NUM.get(t, _ORD.get(t))


def spelled_years(s: str) -> list[int]:
    """'twenty thirty-five', 'two thousand and forty' -> [2035, 2040]."""
    out = []
    tens = "|".join(_EN_TEENS_TENS)
    units = "|".join(_EN_UNITS)
    for m in re.finditer(rf"\b(?:twenty|two\s+thousand(?:\s+and)?)[\s\-]+({tens})(?:[\s\-]+({units}))?\b",
                         s, re.I):
        out.append(2000 + _EN_TEENS_TENS[m.group(1).lower()]
                   + (_EN_UNITS[m.group(2).lower()] if m.group(2) else 0))
    for m in re.finditer(rf"\btwo\s+thousand(?:\s+and)?[\s\-]+({units})\b(?![\s\-]+(?:{tens}))", s, re.I):
        out.append(2000 + _EN_UNITS[m.group(1).lower()])
    return out


def _base_year(s: str, c) -> int:
    """The year a relative lead counts from: an explicit page start named in the question, the
    later start when it is asked for, otherwise the first start (the page's 'start now')."""
    named = [int(y) for y in re.findall(r"(?<!\d)(20\d\d)(?!\d)", s) if int(y) in c.starts]
    if named:
        return max(named)
    if _has([r"\blater\s+start", r"\bsecond\s+start", r"\bstart(?:ing)?\s+later\b", "較晚", "晚一點",
             "後一個開始"], s):
        return c.starts[-1]
    return c.starts[0]


def beyond_horizon(s: str, b: PageBundle) -> bool:
    s = norm(s)
    c = Q.ctx(b)
    H = c.years[-1]
    years = [int(y) for y in re.findall(r"(?<!\d)(2[01]\d\d)(?!\d)", s)]          # also "2040s"
    years += [2000 + int(y) for y in re.findall(r"(?:^|[\s(（])['’‘](\d{2})(?![\d%])", s)]   # '35
    years += spelled_years(s)
    if any(y > H for y in years):
        return True
    for m in re.finditer(r"(?:民國|ROC\s*)\s*(\d{2,3}|[〇零一二三四五六七八九十百]+)\s*年?|(?<![\d.])(1[1-9]\d)\s*年",
                         s, re.I):
        roc = zh_number(m.group(1)) if m.group(1) else int(m.group(2))
        if roc is not None and roc + 1911 > H:
            return True
    if re.search(rf"(?:after|beyond|past|post-?|later\s+than)\s*(?:the\s+year\s+)?{H}\b", s, re.I) or \
            re.search(rf"{H}\s*年?\s*(?:以後|之後|後)", s):
        return True
    if _has(FAR, s):
        return True
    base = _base_year(s, c)
    for pat, span in EN_LEAD:
        for m in re.finditer(pat, s, re.I):
            n = _en_number(m.group(1))
            # "for seven years from 2026" is a span, whatever follows it
            span = span or bool(re.search(r"\b(?:for|over|within|during|across)\s+(?:the\s+)?(?:next\s+)?$",
                                          s[:m.start()], re.I))
            if n is not None and base + n - span > H:
                return True
    for m in re.finditer(r"\bthe\s+(" + "|".join(_ORD) + r")\s+year\s+(?:of|after|into|from)\b", s, re.I):
        if base + _ORD[m.group(1).lower()] - 1 > H:
            return True
    for pat, n in EN_DECADES:
        for m in re.finditer(pat, s, re.I):
            k = n if n is not None else (_en_number(m.group(1)) or 0) * 10
            if base + k > H:
                return True
    for pat, off in ZH_LEAD:
        for m in re.finditer(pat, s):
            n = zh_number(m.group(1))
            if n is not None and base + n + off > H:
                return True
    return False


# ---------------------------------------------------------------------------------------------
# single-township water level (ruling D3)
# ---------------------------------------------------------------------------------------------
TOWN_WORDS = [r"\btownships?\b", r"\btowns?\b", r"\bvillages?\b", "鄉", "鎮"]
HEAD_WORDS = [r"water[\s-]*levels?", r"\bheads?\b",
              r"ground[\s-]*water(?![\s-]*(?:pumping|use|extraction|abstraction|withdrawal|draw))",
              r"water[\s-]*table", r"\baquifers?\b", r"piezometric", r"potentiometric", r"\bdrawdown\w*",
              r"\bwater\b[^.?!;,]{0,20}?\b(?:come|comes|coming|go|goes)\s+(?:back\s+)?up\b",
              "水位", r"地下水(?!抽|使用|用量|開採|的抽|超抽)", "水頭", "含水層", "洩降", "地下水面"]


def town_names(t: dict) -> list[str]:
    """A page township's names: English, Chinese, and the Chinese without 鄉/鎮/市."""
    return [n for n in (t["en"], norm(t["zh"]), re.sub("[鄉鎮市]$", "", norm(t["zh"]))) if n and len(n) >= 2]


def township_ref(s: str, b: PageBundle) -> bool:
    """A page township (any name, or its bare name), or the words township/town/village/鄉/鎮."""
    s = norm(s)
    c = Q.ctx(b)
    for t in c.towns:
        if t.get("pooled"):
            continue
        for n in town_names(t):
            if re.search(r"(?<![a-z])" + re.escape(n) + r"(?![a-z])", s, re.I):
                return True
    return _has(TOWN_WORDS, s)


def township_head(s: str, b: PageBundle) -> bool:
    """A township reference with any head word: refused whatever else the question asks."""
    s = norm(s)
    return township_ref(s, b) and _has(HEAD_WORDS, s)


# ---------------------------------------------------------------------------------------------
# round 10 (ruling R10.2a): the narrowed exemptions. Each one only lets a question past a refusal
# (to the rules and the LLM); none routes. Every exemption has must-refuse boundary twins in
# tests/test_analyst_guards_r10.py.
# ---------------------------------------------------------------------------------------------
# finer clauses than CLAUSE: also sentences, so / because / since / whereas / though, brackets
_CLAUSE_X = (CLAUSE + r"|(?<!\d)\.(?!\d)|\bso\b|\bbecause\b|\bsince\b|\bwhereas\b|\b(?:al)?though\b|"
             r"[()（）—–]")
# words that add Changhua to a lever rather than leave it out: any of them ends the A1 exemption
ADDITIVE = [r"\balso\b", r"\btoo\b", r"\bas\s+well\b", r"\bboth\b", r"\beither\b", r"\bneither\b", r"\bnor\b",
            r"\blikewise\b", r"\bin\s+addition\b", r"\badditional\w*", r"\bbesides\b", r"\bplus\b",
            r"\btogether\b", r"\bjointly\b", r"\balong\s+with\b",
            r"(?<!\bstays\sthe\s)(?<!\bstay\sthe\s)(?<!\bremains\sthe\s)(?<!\bremain\sthe\s)(?<!\bkept\sthe\s)"
            r"(?<!\bkeeps\sthe\s)(?<!\bkeep\sthe\s)\bsame\b", r"\bsimilar\w*", r"\bequal\w*",
            r"\bmatch\w*", r"\bnot\s+(?:only|just)\b", r"\bfollow\w*\s+suit\b", r"\bapart\s+from\b",
            r"\bother\s+than\b", r"\btwo\b", r"\bneighbo(?:u)?r\w*\s+and\b",
            "也", "亦", "還", "都", "同樣", "一樣", "一起", "一併", "同時", "跟著", "跟進", "比照", "照做", "同步", "一并", "并且",
            r"兩(?!成)", "雙", "共同", "連同", "以及", "不只", "不僅", "而且", "並且", "除了", "彰雲", "雲彰"]
# the clause leaves Changhua out: each pattern carries its own negation
EXCLUDE = [r"\bnothing\s+(?:on|in|for|from|across|at|north)\b",
           r"\bnot\s+(?:on|in|for|from|at|across)\b", r"\bnot\s+(?:the\s+)?(?=" + _CH_T + r")",
           r"\b(?:is|are|was|were)(?:n't|\s+not)\s+(?:in|part\s+of|included|includ\w*|modell?ed|covered|"
           r"touched|changed|cutting|cut|reducing|pumping\s+less)\b",
           r"\b(?:does|do|did|will|would)(?:n't|\s+not)\s+(?:cut|reduce|change|touch|curb|trim|stop|retire|"
           r"pump\s+less|take\s+part|join)\b", r"\bwon't\s+(?:cut|reduce|change|touch)\b",
           r"\bno\s+(?:cuts?|change|policy|lever)\s+(?:in|for|on|across)\b",
           r"\b(?:excluding|excluded|exclude|except(?:\s+for)?|without|unlike|not\s+including|outside)\b",
           r"\bleav\w*\s+(?:\S+\s+){0,6}?(?:out|alone|aside|as\s+(?:is|it\s+is))\b", r"\bleft\s+(?:out|alone|aside)\b",
           r"\b(?:unchanged|untouched|unaffected|excluded)\b", r"\bas\s+(?:it\s+)?is\b",
           # review finding 5: aside, not counting, ignore, does nothing, stays the same, no change in
           r"\baside\b", r"\bnot\s+counting\b", r"\bignor(?:e|es|ed|ing)\b", r"\bdoes\s+nothing\b",
           r"\b(?:stays?|remains?|kept|keeps?)\s+the\s+same\b", r"\bno\s+change\s+(?:in|for|on|at)\b",
           "沒有動作", "沒動作", "沒有動", r"不在(?:模型|計算|討論)?的?範圍(?:內|之內)?",
           r"\bout\s+of\s+(?:it|this|scope|the\s+(?:model|question|picture|policy|scope))\b",
           r"\bnot\s+(?:in|part\s+of)\s+the\s+model\b",
           r"(?:完全)?(?:不動|不變|不減(?:抽|少)?|不停(?:抽)?|不管|不算|不計|不列入|不包括|不含|不納入|不在模型)",
           r"維持(?:原狀|現狀|不變)", "照舊", "除外", "排除", "不考慮", "不像", "沒有減", "沒減"]
NEGATION = [r"\bnot?\b", r"n't\b", r"\bnever\b", r"\bnothing\b", r"\bnone\b", r"\bwithout\b", r"\bun\w+ed\b",
            r"\bexcept\w*", r"\bexclud\w*", r"[不沒無非未別勿莫]",
            # third review: non-Yunlin, 雲林以外 / 之外, beyond Yunlin's border
            r"\bnon[\s\-]", "以外", "之外", "外的", r"\bbeyond\b", r"\bborder\w*"]
# in the rest of an exclusion clause a comparative also turns "doesn't cut less" into a cut
COMPARATIVE = [r"\bless\b", r"\bfewer\b", r"\bmore\b", r"\bfurther\b", r"\bextra\b", "少", "多", "更"]
# outside the exclusion clauses of A1: no other place (a bank, side, river, county, neighbour,
# the fan, 北 but 北港, 對岸 ...), and Yunlin not set aside
_ELSEWHERE = [r"\b(?:taichung|chiayi|nantou|tainan|kaohsiung|pingtung|taipei|hsinchu|miaoli|taoyuan|"
              r"keelung|yilan|hualien|taitung|penghu|kinmen|matsu)\b",
              "台中", "嘉義", "南投", "台南", "高雄", "屏東", "台北", "新竹", "苗栗", "桃園", "基隆", "宜蘭", "花蓮",
              "台東", "澎湖", "金門", "馬祖",
              r"\bnorth\w*(?!\s+yun)", r"\bbanks?\b", r"\bsides?\b", r"\bacross\b", r"\brivers?\b",
              r"(?<!yunlin\s)\bcount(?:y|ies)\b", r"\bneighbo\w*", r"\bfan\b", r"\bupstream\b", r"\bregion\w*",
              r"北(?!港)", "對岸", "鄰", "扇", "另一邊", "其他縣", "外縣", "溪",
              # "the other farmers", elsewhere, outside (not the class "other uses")
              r"\bother\b(?!\s+(?:uses?|pumping|class\w*|water[\s\-]*uses?)\b)", r"\belsewhere\b",
              r"\boutside\b", "別處", "其餘", "外面", "外地",
              # third review: an upriver counterpart, up the coast
              r"\bcounterpart\w*", r"\bup[\s\-]*(?:river|stream)\b", r"\bup\s+the\s+coast\b"]
CONTRAST = [r"\b(?:but|however|yet|though|although|still|nevertheless|nonetheless|instead|whereas|while)\b",
            "但", "可是", "不過", "然而", "卻", "仍", "反而", "而是",
            # the review's broken exclusions (finding 4): except that, save, bar, until, for now,
            # hardly, far from, not really ...
            r"\bexcept\s+(?:that|when|if|where|for\s+(?:its|their|it|the\s+\w+\s+it))\b", r"\bsave\b",
            r"\bexcept\s+(?:for\s+)?(?:a|an|the|some|one|any|its|their|new)\b", r"\bapart\s+from\b",
            r"\bbar\b", r"\bbarring\b", r"\bbut\s+for\b", r"\buntil\b", r"\btill\b", r"\bfor\s+now\b",
            r"\bhardly\b", r"\bscarcely\b", r"\bbarely\b", r"\bfar\s+from\b", r"\bnot\s+really\b",
            r"\bnot\s+so\b", r"\btemporar\w*", r"\b(?:at\s+first|initially|for\s+the\s+(?:moment|time\s+being))\b",
            r"\bunless\b", r"\bonly\b", r"\balmost\b", r"\bnearly\b", r"\bpartly\b", r"\bpartially\b",
            "除非", "暫時", "目前", "先", "直到", "為止", "只到", "幾乎", "不完全", "並非", "未必", "部分"]
_SET_ASIDE = [r"\bignor\w*", r"\bskip\w*", r"\bset\s+aside\b", r"\bforget\b", "忽略", "撇開", "不管", "不算"]
# third-person words that could bring Changhua back into a later clause ("there" only as a place)
_BACK_REF = [r"\b(?:those|these|such|said)\b", r"\bin\s+question\b", r"\bconcerned\b",
             "那些", "這些", "那個", r"(?<!應)該(?!不|會|是|怎)", r"(?<![因如從彼])此(?!外|時)", "彼",
             r"\b(?:the\s+)?(?:latter|former)\b", r"\bnext\s+door\b", r"\bover\s+there\b",
             r"\bthat\s+(?:side|place|area|one|county|region|bank|part)\b", r"\bthe\s+excluded\b",
             r"\bthe\s+other\s+(?:one|county|side|place)\b", r"\bthe\s+rest\b", r"\bthe\s+neighbo\w*",
             "那一側", "那側", "對側", "另一側", "那頭", "那一頭", "對面", "隔壁", "前者", "後者", r"其(?!他|中|實|餘)", "該地", "此地", "該處", "那一邊",
             "另一個縣", "另一縣", "別的縣", "鄰近",
             r"\b(?:it|its|they|their|them|locally)\b", r"(?<!\bis\s)(?<!\bare\s)\bthere\b(?!\s+(?:is|are|was|were|will|'s))",
             "當地", "該縣", "他們", "它", "那邊", "那裡", "那裏", "那里", "自己"]


def _ch_hit(cl: str) -> bool:
    return bool(re.search(CHANGHUA_WORD, cl, re.I)) or _has(GEO_EN + GEO_ZH, cl)


def _excluded_clause(cl: str) -> bool:
    """A Changhua clause that only leaves Changhua out: an exclusion pattern; once those words are
    taken out, no other negation (no double negative), no lever, no head or subsidence word."""
    if not _has(EXCLUDE, cl):
        return False
    # Yunlin named in the exclusion clause itself ("Changhua stays the same as Yunlin") ties the
    # two counties together; Yunlin inside a geography phrase ("the county north of Yunlin") does not
    keep = [True] * len(cl)
    for p in GEO_EN + GEO_ZH:                      # every geography span, all read on the clause
        for m in re.finditer(p, cl, re.I):
            keep[m.start():m.end()] = [False] * (m.end() - m.start())
    bare = "".join(ch if k else " " for ch, k in zip(cl, keep, strict=True))
    if _has(YUNLIN, bare):
        return False
    rest = cl
    for p in EXCLUDE:
        rest = re.sub(p, " ", rest, flags=re.I)
    return not (_has(NEGATION + COMPARATIVE, rest) or has_lever(rest) or _has(HEAD_WORDS, rest)
                or subsidence_words(rest))


def yunlin_only_ok(s: str) -> bool:
    """A1 (R10.2a): a Yunlin question (a Yunlin lever, or Yunlin's levers ranked) that names
    Changhua — by name or by geography (north of the Choushui, the neighbouring county) — only to
    leave it out ("nothing on the north side of the river", 彰化那邊完全不動, "the neighbouring county
    isn't in the model").

    Every clause that reaches Changhua must be such an exclusion clause (:func:`_excluded_clause`);
    Yunlin is named; no Changhua township and no whole-fan reference; no additive word (also, both,
    too, 也, 都, 除了 ...); no third-person word after the first Changhua clause; and, with the
    exclusion clauses taken out, no lever is Changhua's (:func:`changhua_owns_a_lever`)."""
    s = norm(s)
    ref = changhua_ref(s)
    if not ref or not ref <= {"word", "geo"} or not _has(YUNLIN, s) or _has(ADDITIVE, s):
        return False
    # a contrast after Changhua is named ("..., Changhua unchanged, but more stop pumping") may
    # give the next lever to Changhua: refuse
    first = min([m.start() for m in re.finditer(CHANGHUA_WORD, s, re.I)]
                + [m.start() for p in GEO_EN + GEO_ZH for m in re.finditer(p, s, re.I)])
    if _has(CONTRAST, s[first:]):
        return False
    clauses = [c for c in re.split(_CLAUSE_X, s, flags=re.I) if c.strip()]
    # both counties joined or Changhua following suit, read within clauses ("Unlike Changhua,
    # Yunlin ..." is no list of the two counties)
    if _has(BOTH + ALSO, " ; ".join(clauses)):
        return False
    # a Changhua reference that spans two clauses cannot be judged clause by clause: refuse
    pieces = sum(len(re.findall(CHANGHUA_WORD, c, re.I)) for c in clauses)
    if pieces != len(re.findall(CHANGHUA_WORD, s, re.I)) or (_has(GEO_EN + GEO_ZH, s)
                                                              and not any(_has(GEO_EN + GEO_ZH, c) for c in clauses)):
        return False
    kept, seen_ch = [], False
    for cl in clauses:
        if _ch_hit(cl):
            if not _excluded_clause(cl):
                return False
            seen_ch = True
            continue
        if seen_ch and _has(_BACK_REF, cl):
            return False
        # a later clause that negates something without naming Yunlin may be Changhua's (review 4)
        if seen_ch and _has(NEGATION, cl) and not _has(YUNLIN, cl):
            return False
        # a lever after a Changhua clause must name Yunlin in its own clause (second review: the
        # Yunlin-only scope exemption let "those farmers halve aquaculture" through; dropped)
        if seen_ch and has_lever(cl) and not _has(YUNLIN, cl):
            return False
        # no other place than Yunlin outside the exclusion clauses, and Yunlin itself not left out
        if _has(_ELSEWHERE, cl) or (_has(YUNLIN, cl) and _has(NEGATION + _SET_ASIDE, cl)):
            return False
        kept.append(cl)
    return seen_ch and not changhua_owns_a_lever(" ; ".join(kept))


# A3: how sure is Changhua's direction test (E25/E26: direction only, at wells not tuned on)
CH_TEST = META + [r"\bsigns?\b", r"\bjudged\b", r"\bheld[\s\-]*out\b", r"\bout[\s\-]*of[\s\-]*sample\b",
                  r"\b(?:never|not)\s+(?:been\s+)?(?:tuned|calibrated|fitted|trained)\b", r"\bindependent\b",
                  "跡象", "沒調", "未調", "未校準", "沒校準", "獨立", "樣本外"]
# after the review (finding 1) only direction nouns name the topic: the direction, the right /
# wrong way, the response; a rise, fall or recovery is a value and is refused (A3_VETO)
DIRECTION_TOPIC = [r"\bdirection\w*", r"\b(?:right|wrong|correct|expected)\s+way\b", r"\brespon\w*",
                   r"\breact\w*", "方向", "正確", "反應", "回應"]
# A3 is an allow-list: any of these refuses it (review finding 1)
A3_VETO = (
    # a condition or a scenario: if / when / once / after / under / as ..., 如果 / 時 / 後 ...
    [r"\bif\b", r"\bwhen\b(?!\s+(?:is|will|was|are|does|do|did|would)\b)", r"\bwhenever\b", r"\bonce\b",
     r"\bafter\b", r"\bunder\b", r"\bas\b", r"\bwhile\b", r"\bgiven\b", r"\bprovided\b", r"\bsuppos\w*",
     r"\bassum\w*", r"\bunless\b", r"\bin\s+case\b", r"\bscenario\w*", r"\bpolic(?:y|ies)\b",
     "如果", "若", "假如", "假設", "要是", "一旦", "時", "後", "的話", "當", "政策", "方案", "情境"]
    # Yunlin, both counties or Changhua following suit, any lever or class
    + YUNLIN + BOTH + ALSO + [r"\blevers?\b", "槓桿"] + CLASS_ANY
    # the ground rather than the water level
    + [r"\b(?:ground|land|soil|surface|elevation|terrain)\b", "地面", "地表", "土地", "高程", "地層", "地盤",
       r"地(?!下水|區|方|點)"]
    # a value of the direction: a rise, fall or recovery, more or less
    + [r"\b(?:rise|rises|rising|rose|raise\w*|recover\w*|rebound\w*|go(?:es)?\s+up|went\s+up|come\s+back|"
       r"fall|falls|falling|fell|drop\w*|declin\w*|decreas\w*|increas\w*|lower\w*|higher|less|fewer|more|"
       r"improv\w*|worsen\w*|deplet\w*)\b",
       "上升", "回升", "升高", "下降", "降低", "增加", "減少", "變高", "變低", "恢復", "回彈", "少", "多", "升", "降"]
    # a place word that could be Changhua's own ("there", 那邊)
    + [r"(?<!\bis\s)(?<!\bare\s)(?<!\bwas\s)\bthere\b(?!\s+(?:is|are|was|were|been|any|'s))", "那邊", "那裡",
       "那裏", "當地", "該地", "那頭"]
    # second review: upward / climb, a second subject or question joined by and / or / 和
    + [r"\bupward\w*", r"\bdownward\w*", r"\bclimb\w*", r"\bascend\w*", r"\bdescend\w*", "上揚", "回漲", "漲", "跌",
       r"\band\b", r"\bor\b", r"\bplus\b", "和", "與", "及", "跟", "以及", "或"]
    # third review: the ground measured (leveling, vertical, InSAR, GNSS, benchmarks) and a
    # programme, plan, crackdown, permit or regulation as the thing responded to
    + [r"\blevel(?:l)?ing\b", r"\bvertical\w*", r"\binsar\b", r"\bgnss\b", r"\bgps\b", r"\bbenchmark\w*",
       "水準", "垂直", "衛星",
       r"\bprogram(?:me)?s?\b", r"\bplans?\b", r"\bschemes?\b", r"\bcrackdown\w*", r"\bpermit\w*",
       r"\blicen[cs]\w*", r"\benforc\w*", r"\bregulat\w*", r"\billegal\b", r"\bown\b",
       "計畫", "計劃", "取締", "許可", "執照", "法規", "違法", "非法"])
# the direction asked as the answer itself
DIRECTION_VALUE = [r"\bwhich\s+(?:way|direction)\b", r"\bup\s+or\s+down\b", r"\bdown\s+or\s+up\b",
                   r"\b(?:rise|rises|go\s+up|goes\s+up|increase\w*)\s+or\s+(?:fall|falls|drop|drops|go\s+down|"
                   r"goes\s+down|decrease\w*)\b", r"\b(?:fall|falls|drop|drops)\s+or\s+(?:rise|rises)\b",
                   r"\bin\s+what\s+direction\b", r"\bwhat\s+direction\b",
                   "哪個方向", "什麼方向", "往哪", "升還是降", "上升還是下降", "下降還是上升", "會升會降", "漲還是跌",
                   "上升或下降", "升或降"]
# a head size inside the A3 how-sure form ("by a meter")
HEAD_SIZE = [r"\bdecimet\w*", "公寸", r"\bmet(?:er|re)s?\b", r"\b\d*\s*m\b", r"\bfeet\b", r"\bfoot\b",
             r"\bby\s+(?:a|an|one|two|three|half|several|some|about|around|roughly)\b", "公尺", "米", "公分"]
_SECOND_Q = r"\b(?:and|but|or)\s+(?:is|are|was|were|has|have|does|do|did|can|could|will|would|what|how)\b"


def changhua_how_sure_ok(s: str, b: PageBundle | None = None) -> bool:
    """A3 (R10.2a): one how-sure question about whether Changhua's direction test was run or held
    ("is there any sign that pumping on the Changhua side moves water levels the right way, judged
    at wells the model was never tuned on?").

    Changhua by name or geography only (no Changhua township, no whole fan); one sentence with a
    single question; test wording (CH_TEST) in a yes/no or how-sure form (YESNO); a direction word;
    and no lever, no 'what is' / show (WHAT), no size or magnitude, no digit, no subsidence word, no
    area scope, and no direction asked as the value (which way, up or down, 升還是降). After the
    review it is a strict allow-list: also none of A3_VETO (a condition, Yunlin, both counties or
    following suit, any lever, lever word or class, the ground, a rise / fall / recovery, a place
    word), and the topic must be a direction noun (direction, the right way, the response)."""
    s = norm(s)
    ref = changhua_ref(s)
    if not ref or not ref <= {"word", "geo"}:
        return False
    if b is not None and township_ref(s, b):       # a Yunlin township or the word township
        return False
    sentences = [x for x in re.split(r"[?？。!！;；]|(?<!\d)\.(?!\d)", s) if x.strip()]
    return (len(sentences) == 1 and s.count("嗎") <= 1 and not re.search(_SECOND_Q, s, re.I)
            and _has(CH_TEST, s) and _has(YESNO, s) and _has(DIRECTION_TOPIC, s)
            and not has_lever(s)
            and not _has(WHAT + WHAT_R10 + MAGNITUDE + SIZE_ADJ + DIRECTION_VALUE + HEAD_SIZE
                         + SIZE_WORDS_R10 + A3_VETO, s)
            and not re.search(r"\d", s) and not subsidence_words(s) and not area_scope(s)
            and not _has(ADDITIVE, s))


# A4: one township, not named ("that coastal township", 靠海那個鄉), under a policy: clarified
UNNAMED_TOWN = [r"\b(?:a|one|that|this|the|some|my|our)\s+(?:[\w\-]+\s+){0,3}?(?:township|town(?!ship)|village)\b(?!s)",
                r"(?:那個|這個|某個|一個|某一個|那一個|這一個)[^，,。？?；;、]{0,4}?[鄉鎮](?![鎮鄉])"]
UNNAMED_VETO = [r"\b(?:townships|towns|villages)\b", r"\b(?:which|what|each|every|all|any|per|by|other)\s+"
                r"(?:[\w\-]+\s+){0,2}?(?:township|town|village)\b",
                r"\b(?:most|least|biggest|largest|smallest|worst|best|fastest|slowest|highest|lowest|more|less\s+than)\b",
                r"\b(?:average|mean|typical|median|overall|total|rank\w*|compar\w*|differ\w*)\b",
                "鄉鎮", "各", "每", "所有", "全部", "哪", "最", "平均", "典型", "總", "比較", "排名", "差異"]


def unnamed_township(s: str, b: PageBundle) -> bool:
    """A4 (R10.2a): one township referred to but not named, under a policy (a lever word): its
    internals row is clarified (ambiguous_town), not refused. Not when a page township is named
    (the named exemption covers it), and never for several, each, which, any, the average or the
    most ... township (an area statistic), nor with an area-scope word."""
    s = norm(s)
    c = Q.ctx(b)
    named = any(re.search(r"(?<![a-z])" + re.escape(n) + r"(?![a-z])", s, re.I)
                for t in c.towns for n in (town_names(t) if not t.get("pooled") else []))
    return (not named and _has(UNNAMED_TOWN, s) and not _has(UNNAMED_VETO, s) and has_lever(s)
            and not area_scope(s) and not changhua_ref(s))
