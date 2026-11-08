# FROZEN COPY (round 9): hydrophysics_analyst/router/guards.py as at commit fc7b996, vendored for the
# R10.3b round-9 replay (hydrophysics_analyst.replay_r9). Do not edit: only the relative imports
# changed (one more dot). tests/test_analyst_r9_replay.py checks it still decides as fc7b996 did.
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
"""
from __future__ import annotations

import re
import unicodedata

from ... import query as Q
from ...page import PageBundle
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
         "減", "停", "少抽", "退場", "禁", r"[一二三四五六七八九十\d]\s*成", "%", "降低抽", "降低用水", "槓桿"]
PRONOUN = [r"\b(?:it|its|they|their|them|there|locally)\b", "也", "亦", "當地", "該縣", "他們", "它", "那邊",
           "那裡", "那裏", "自己"]
CLAUSE = r"[。；;？?！!，,:：]|\band\b|\bwhile\b|\bbut\b|\bthen\b|並且|而且|以及|同時"


def _clauses(s: str) -> list[str]:
    return [c for c in re.split(CLAUSE, s, flags=re.I) if c.strip()]


# ---------------------------------------------------------------------------------------------
# Changhua
# ---------------------------------------------------------------------------------------------
CHANGHUA_WORD = r"\b(?:chang|zhang)[\s\-_']*hua\b|\bchanghwa\b|\bchang[\s\-]*hwa\b|彰\s*化"
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
          r"\bnorth(?:ern)?\s+(?:half|part|side|portion|end)\s+of\s+the\s+(?:alluvial\s+)?fan\b",
          r"\bfan'?s\s+north(?:ern)?\s+(?:half|part|side|portion)"]
GEO_ZH = [r"濁水(?:溪)?(?:以|的)?北", r"濁水溪(?:的)?(?:對岸|兩岸|彼岸|另一邊|另一側)",
          r"(?<![一-鿿])溪北|[在的]溪北", r"(?:兩|二)(?:個)?縣", r"彰雲|雲彰",
          r"鄰縣|隔壁(?:的)?縣", r"北(?:邊|方|側)的縣", r"雲林(?:以北|北邊|北方|的北邊)", r"雲林(?:和|與|跟|及)(?:其)?鄰",
          r"(?:沖積扇|扇)(?:的)?北(?:半部|部|側|邊|段)"]
# the whole fan, which includes Changhua: a Changhua lever only when a lever is asked
WHOLE_FAN = [r"\b(?:whole|entire|full)\s+(?:of\s+the\s+)?(?:choushui\s+)?(?:alluvial\s+)?fan\b",
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
        rf"{_CH}(?:也|亦|同樣|一樣|跟進|照做|比照|同步|跟著)", rf"(?:也|同樣|一樣|比照|跟進)[^，,。？?]{{0,4}}{_CH}"]
_PREP_TO_COUNTY = re.compile(r"^\s*(?:\S+\s+){0,4}?(?:in|at|for|across|by|throughout|within|of|from|on|among|"
                             r"(?:appl\w*|extend\w*|imposed|introduced)\s+(?:also\s+)?(?:to|in|on))"
                             r"\s+(?:the\s+)?$|^\s*$", re.I)


def _cjk(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


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
               + [(m, "c") for p in PRONOUN for m in re.finditer(p, cl, re.I)])
        for lv in levs:
            zh = _cjk(cl[lv.start()])
            near = []
            for m, k in cty:
                if m.end() <= lv.start():                         # the county comes first
                    near.append((lv.start() - m.end(), k == "y", k))
                elif m.start() >= lv.end() and (zh or _PREP_TO_COUNTY.match(cl[lv.end():m.start()])):
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
              "公分", "厘米", "釐米", "毫米", "公釐", "公厘"]
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
        "評分", "登錄"]
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
         r"[嗎][？?]?\s*$", "有沒有", "是否", "為什麼", "為何", "何時", "什麼時候"]
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


def how_sure_only(s: str) -> bool:
    """Asks how far the subsidence result is tested, not what it is: how-sure wording (META) in a
    yes/no or how-sure form (YESNO), with no 'what is', no give / show / print / list, no 多少
    (WHAT), no lever, no area-scope word, and no size asked except by the bare words size /
    number (no size adjective, no decimal number either)."""
    s = norm(s)
    asks = [p for p in MAGNITUDE if p != SIZE_WORD] + SIZE_ADJ
    return (_has(META, s) and _has(YESNO, s) and not _has(WHAT, s) and not has_lever(s)
            and not area_scope(s) and not _has(asks, s))


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
