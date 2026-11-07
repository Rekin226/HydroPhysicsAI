"""Deterministic refusal guards (round 9, after ruling R9.2): four classes of question that must be
refused whatever the router would read, in English and Chinese (traditional and simplified).

They are written from the claims ledger and the page, not from any evaluation item:

- **Changhua levers** (C13a/b/c withdrawn, E20 pending; ``page_scoping.changhua_lever_rule``).
  The page's levers are Yunlin's; Changhua is one pooled row "not yet split by township", shown
  only as the labelled spillover of a Yunlin policy. So Changhua is refused whether it is named
  ("Changhua", "chang hua", 彰化), reached by geography (north of the Choushui River, 濁水溪以北,
  溪北, the county to the north, both counties / both banks), or by one of its townships (二林,
  溪州, Erlin ...). Only the word Changhua itself, asked as the pooled township row with no lever
  near it, passes (:func:`changhua_ref` returns ``"word"`` and the caller keeps that exception).
- **Subsidence size** (C03, C09 withdrawn; C05 area): how much the ground sinks, settles or
  rebounds, in any wording ("land settle", 地面少沉, 沉降, cm, 公分, 厘米). Not refused: a named
  Yunlin township (its model-internal row, C11), the rail's differential settlement (C06, the
  rail preset's own labelled quantity), and questions about how far the result is tested
  (how_sure) that do not ask for a size.
- **Beyond the horizon**: any year, relative lead or decade after the page's last year
  (``c.years[-1]``, 2032 on the round-8 page): "2040", "ten years after a 2030 start",
  "in a decade", 十年後, 民國125年, "after 2032", "the 2040s", mid-century.
- **Single-township water level** (ruling D3, rules_v1.json: head_change / township not
  offered): a township reference (a page township, its bare name, or the words township, town,
  village, 鄉, 鎮) together with any head word (water level, head, groundwater, water table,
  aquifer, 水位, 地下水, 水頭, 含水層). Not refused when the question is about subsidence (the
  township's internals row) — the head answer is the mean at the registered Yunlin wells, which
  no township phrase asks for.

Each guard only ever returns a refusal reason (or nothing); none routes.
"""
from __future__ import annotations

import re

from .. import query as Q
from ..page import PageBundle


def _has(pats, s: str) -> bool:
    return any(re.search(p, s, re.I) for p in pats)


# ---------------------------------------------------------------------------------------------
# Changhua
# ---------------------------------------------------------------------------------------------
CHANGHUA_WORD = r"\bchang[\s\-_]*hua\b|\bchanghwa\b|彰化"
CHOUSHUI = (r"(?:choushui|chuoshui|zhuoshui|jhuoshuei|jhuoshui|choshui|cho[\s\-]shui|chou[\s\-]shui)"
            r"(?:[\s\-]*(?:river|creek|hsi|xi))?")
RIVER = rf"(?:the\s+)?(?:{CHOUSHUI}|river)"
GEO_EN = [rf"\bnorth(?:ern)?\s+(?:of|side\s+of|bank\s+of|banks\s+of|shore\s+of)\s+{RIVER}",
          rf"\b{CHOUSHUI}(?:'s)?\s+north(?:ern)?\s+(?:bank|side|shore)",
          rf"\bacross\s+{RIVER}", rf"\bother\s+side\s+of\s+{RIVER}",
          rf"\bboth\s+(?:sides|banks)\s+of\s+{RIVER}",
          rf"\bon\s+either\s+side\s+of\s+{RIVER}",
          r"\bboth\s+counties\b", r"\btwo\s+counties\b", r"\bneighbo(?:u)?ring\s+county\b",
          r"\bnorth(?:ern)?\s+neighbo(?:u)?r", r"\bcounty\s+(?:to\s+the\s+)?north\b",
          r"\bnorth(?:ern)?\s+county\b",
          r"\bnorth(?:ern)?\s+(?:half|part|side|portion|end)\s+of\s+the\s+(?:alluvial\s+)?fan\b",
          r"\bfan'?s\s+north(?:ern)?\s+(?:half|part|side|portion)"]
GEO_ZH = [r"(?:濁水溪|浊水溪|濁水|浊水)(?:以|的)?北", r"(?:濁水溪|浊水溪)(?:的)?(?:對岸|对岸|兩岸|两岸)",
          r"(?<![\u4e00-\u9fff])溪北|[在的]溪北", r"(?:兩|两|二)(?:個|个)?[縣县]", r"彰雲|雲彰|彰云|云彰",
          r"鄰縣|邻县|隔壁(?:的)?縣|隔壁(?:的)?县", r"北(?:邊|边|方|側|侧)的(?:縣|县)",
          r"(?:沖積扇|冲积扇|扇)(?:的)?北(?:半部|部|側|侧|邊|边|段)"]
# Changhua County's townships (the page pools them into one row). Bare two-character names that
# are also common words (田中 "in the field", 大城 "big city", 花壇 "flower bed", 和美, 大村, 福興)
# count only with their 鄉/鎮 suffix; 北斗 never as the satellite system.
CH_TOWNS_ZH = ["鹿港", "線西", "线西", "伸港", "秀水", "芬園", "芬园", "員林", "员林", "溪湖", "埔鹽", "埔盐",
               "埔心", "永靖", "社頭", "社头", "二水", "二林", "田尾", "埤頭", "埤头", "芳苑", "竹塘", "溪州",
               "和美[鎮镇]", "福興[鄉乡]", "福兴[鄉乡]", "花壇[鄉乡]", "花坛[鄉乡]", "田中[鎮镇]", "大村[鄉乡]",
               "大城[鄉乡]", "北斗(?!衛星|卫星|系統|系统|導航|导航|星)"]
CH_TOWNS_EN = ["lukang", "lugang", "hemei", "homei", "xianxi", "hsienhsi", "shengang", "shenkang",
               "fuxing", "fuhsing", "xiushui", "hsiushui", "huatan", "fenyuan", "yuanlin", "xihu",
               "hsihu", "tianzhong", "tienchung", "dacun", "tatsun", "puyan", "puxin", "puhsin",
               "yongjing", "yungching", "shetou", "ershui", "erhshui", "erlin", "erhlin", "tianwei",
               "tienwei", "pitou", "fangyuan", "dacheng", "tacheng", "zhutang", "chutang", "xizhou",
               "hsichou", "sijhou", "xichou",
               r"beidou(?!\s+(?:satellite|system|gnss|navigation|positioning))"]


# the one Changhua question that is answered: the pooled township row's labelled spillover of a
# Yunlin policy, asked as that row (township / row / 鄉鎮 / 列), with no lever and no head
POOLED_ROW = [r"\brows?\b", r"\bpanel\b", r"\btownships?\b", r"\btowns?\b", r"not\s+yet\s+split",
              "列", "鄉鎮", "乡镇", "未分"]


def changhua_ref(s: str) -> frozenset[str]:
    """How the question reaches Changhua: ``word`` (the name itself), ``town`` (one of its
    townships), ``geo`` (north of the Choushui, both counties ...); empty when it does not."""
    out = set()
    if re.search(CHANGHUA_WORD, s, re.I):
        out.add("word")
    if any(re.search(p, s) for p in CH_TOWNS_ZH) or any(
            re.search(rf"\b{p}\b", s, re.I) for p in CH_TOWNS_EN):
        out.add("town")
    if _has(GEO_EN + GEO_ZH, s):
        out.add("geo")
    return frozenset(out)


# ---------------------------------------------------------------------------------------------
# subsidence size
# ---------------------------------------------------------------------------------------------
SUBS_WORDS = [r"subsid\w*", r"\bsink\w*", r"\bsank\b", r"\bsunk\w*", r"\bsettl(?:e|es|ed|ing|ement)\b",
              r"\bcompact(?:ion|s|ed|ing)?\b", r"\bdowndrop\w*", r"\buplift\w*",
              "下陷", "沉陷", "沉降", "下沉", "少沉", "多沉", "沉多少", "沉了", "會沉", "会沉", "沉得",
              "地層", "地层", "地盤", "地盘", "壓密", "压密", "塌陷"]
SUBS_UNITS = [r"\bcm\b", r"centimet\w*", r"\bmm\b", r"millimet\w*", r"\binch(?:es)?\b",
              "公分", "厘米", "釐米", "毫米", "公釐", "公厘"]
# the ground surface (never "groundwater" / "ground water" / "surface water") moving down or back
# up; "on the ground" (in practice) is removed before matching
GROUND = (r"(?:\b(?:ground|terrain)\b(?!\s*-?\s*water)|\bland\s+(?:surface|level|elevation)\b|"
          r"\bsurface\s+elevation\b)")
SUBS_MOVE = [rf"{GROUND}(?:\s+(?:level|surface|elevation))?[^.?!;,]{{0,30}}?\b(?:drop\w*|fall\w*|fell|lower\w*|"
             r"go(?:es)?\s+down|went\s+down|come\s+down|subsid\w*|settl\w*|sink\w*|sank|rebound\w*|rise|rises|"
             r"rising|rose|come\s+back\s+up|comes\s+back\s+up|recover\w*|lift\w*)",
             r"(?:地面|地表(?!水)|地盤|地盘|地層|地层)[^，,。？?；;]{0,8}(?:下降|降低|沉|陷|回升|抬升|上升|回彈|回弹|變低|变低)"]
DIFFERENTIAL = [r"differential\s+(?:settlement|subsidence|sinking)", r"angular\s+distortion",
                r"\bdistortion\b", "差異沉陷", "差异沉陷", "差異沉降", "差异沉降", "不均勻沉陷", "不均匀沉陷",
                "角變量", "角变量", "角變形", "角变形"]
RAIL_WORDS = [r"\bhsr\b", r"high[-\s]speed\s+rail\w*", r"\bthsr\b", r"\brail\w*", r"\btrain\w*",
              "高鐵", "高铁", "鐵路", "铁路"]
MAGNITUDE = [r"how\s+(?:much|many|big|large|deep|fast|far\s+(?:down|up))\b", r"\bby\s+how\s+much\b",
             r"\b(?:amount|magnitude|figure|estimate|number)\s+of\b", r"\bin\s+(?:cm|mm|centimet\w*)\b",
             "多少", "幾公分", "几公分", "多大", "多深", "多快", "幾釐米", "几厘米",
             r"\b(?:give|tell|show|state|quote)\s+(?:(?:me|us)\s+)?(?!(?:me|us)\b)(?!whether|if\b|why|how\s+(?:sure|well|far))"
             r"(?:the\s+)?(?:\w+\s+){0,3}(?:subsid|sink|settl)", "給我", "给我", "告訴我", "告诉我", "直接"] + SUBS_UNITS
META = [r"how\s+(?:sure|confident|reliable|trustworthy|certain|well\s+tested|far\s+(?:is|are)\b)",
        r"\btest(?:ed|s|ing)?\b", r"\bvalidat\w*", r"\bverif\w*", r"\bwithdrawn?\b", r"\bsupported\b",
        r"\bevidence\b", r"\btrust\w*", r"\bconfiden\w*", r"\breliab\w*", r"\buncertain\w*",
        r"\bwhy\s+(?:is|are|isn't|aren't|was|can't|cannot|won't|don't|do\s+you\s+not|no|not)\b",
        r"\bscored\b",
        "可信", "可靠", "多準", "多准", "信心", "驗證", "验证", "檢驗", "检验", "測試", "测试", "撤回", "為什麼",
        "为什么", "為何", "为何", "證據", "证据", "支持", "不確定", "不确定", "評分", "评分", "登錄", "登录"]


def _ground(s: str) -> str:
    return re.sub(r"\bon\s+the\s+ground\b", " ", s, flags=re.I)


def subsidence_words(s: str, *, units: bool = True) -> bool:
    """The question talks about the ground sinking (with ``units``: or about cm / 公分)."""
    s = _ground(s)
    return _has(SUBS_WORDS + SUBS_MOVE + (SUBS_UNITS if units else []), s)


def subsidence_size(s: str) -> bool:
    """A question about how much the ground sinks / settles / rebounds. The caller exempts a
    named Yunlin township (its internals row); this function exempts the rail's differential
    settlement and how-sure questions that do not ask for a size."""
    if not subsidence_words(s):
        return False
    asks_size = _has(MAGNITUDE, s)
    if _has(RAIL_WORDS, s) and _has(DIFFERENTIAL, s) and not asks_size:
        return False
    return asks_size or not _has(META, s)


# ---------------------------------------------------------------------------------------------
# beyond the page horizon
# ---------------------------------------------------------------------------------------------
EN_NUM = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
          "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
          "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
          "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
          "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100, "a hundred": 100,
          "a couple of": 2, "couple of": 2}
_EN_N = r"(\d{1,3}|" + "|".join(sorted((re.escape(k) for k in EN_NUM), key=len, reverse=True)) + r")"
_ZH_DIG = {"零": 0, "一": 1, "二": 2, "兩": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7,
           "八": 8, "九": 9}
_ZH_N = r"((?<!\d)\d{1,3}(?!\d)|[一二兩两三四五六七八九十百]+|數十|数十|幾十|几十|十幾|十几)"
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
]
EN_DECADES = [(r"\bhalf\s+a\s+century\b", 50), (r"\b(?:a|one)\s+century\b", 100),
              (r"\b(?:several|a\s+few|a\s+couple\s+of|couple\s+of|many)\s+decades\b", 20),
              (r"\bdecades\b", 20), (rf"\b{_EN_N}\s+decades?\b", None),
              (r"\b(?:a|one)\s+decade\b", 10), (r"\bthe\s+next\s+decade\b", 10),
              (r"\bdecade\s+(?:later|from|after|on|out)\b", 10)]
FAR = [r"\bmid[-\s]?century\b", r"\bend\s+of\s+(?:the\s+)?century\b", r"\bby\s+21\d\d\b",
       "本世紀中", "本世纪中", "本世紀末", "本世纪末", "半世紀", "半世纪", "百年後", "百年后", "下個世紀", "下个世纪"]
ZH_LEAD = [  # (pattern, offset): reports at base + N + offset
    (rf"{_ZH_N}\s*(?:多)?\s*年\s*(?:後|后|之後|之后|以後|以后|過後|过后)", 0),
    (rf"{_ZH_N}\s*(?:多)?\s*年\s*(?:內|内|間|间|期間|期间)", -1),
    (rf"(?:再過|再过|往後|往后|之後|之后)\s*的?\s*{_ZH_N}\s*(?:多)?\s*年", 0),
    (rf"(?:未來|未来|接下來|接下来)\s*的?\s*{_ZH_N}\s*(?:多)?\s*年", -1),
    (rf"第\s*{_ZH_N}\s*年", -1),
]


def zh_number(t: str) -> int | None:
    if t.isdigit():
        return int(t)
    if t in ("數十", "数十", "幾十", "几十"):
        return 20
    if t in ("十幾", "十几"):
        return 11
    if t == "百" or t == "一百":
        return 100
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
    return int(t) if t.isdigit() else EN_NUM.get(t)


def _base_year(s: str, c) -> int:
    """The year a relative lead counts from: an explicit page start named in the question, the
    later start when it is asked for, otherwise the first start (the page's 'start now')."""
    named = [int(y) for y in re.findall(r"(?<!\d)(20\d\d)(?!\d)", s) if int(y) in c.starts]
    if named:
        return max(named)
    if _has([r"\blater\s+start", r"\bsecond\s+start", r"\bstart(?:ing)?\s+later\b", "較晚", "较晚",
             "晚一點", "晚一点", "後一個開始", "后一个开始"], s):
        return c.starts[-1]
    return c.starts[0]


def beyond_horizon(s: str, b: PageBundle) -> bool:
    c = Q.ctx(b)
    H = c.years[-1]
    for y in re.findall(r"(?<!\d)(2[01]\d\d)(?!\d)", s):
        if int(y) > H:
            return True
    for m in re.finditer(r"(?:民國|民国|ROC\s*)\s*(\d{2,3})\s*年?|(?<![\d.])(1[1-9]\d)\s*年", s, re.I):
        roc = int(m.group(1) or m.group(2))
        if roc + 1911 > H:
            return True
    if re.search(rf"(?:after|beyond|past|post-?|later\s+than)\s*(?:the\s+year\s+)?{H}\b", s, re.I) or \
            re.search(rf"{H}\s*年?\s*(?:以後|以后|之後|之后|後|后)", s):
        return True
    if _has([r"(?:beyond|after|past)\s+(?:the\s+)?(?:page'?s?\s+)?(?:horizon|last\s+year|end\s+of\s+the\s+"
             r"(?:page|run|projection|horizon))"], s):
        return True
    for m in re.finditer(r"(?<!\d)(20\d)0s\b", s):
        if int(m.group(1)) * 10 > H:
            return True
    if _has(FAR, s):
        return True
    base = _base_year(s, c)
    for pat, span in EN_LEAD:
        for m in re.finditer(pat, s, re.I):
            n = _en_number(m.group(1))
            if n is not None and base + n - span > H:
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
TOWN_WORDS = [r"\btownships?\b", r"\btowns?\b", r"\bvillages?\b", "鄉", "鎮", "乡", "镇"]
HEAD_WORDS = [r"water[\s-]*levels?", r"\bheads?\b", r"ground[\s-]*water", r"water[\s-]*table",
              r"\baquifers?\b", r"piezometric", r"potentiometric", r"\bdrawdown\w*",
              r"\bwater\b[^.?!;,]{0,20}?\b(?:come|comes|coming|go|goes)\s+(?:back\s+)?up\b",
              "水位", "地下水", "水頭", "水头", "含水層", "含水层", "洩降", "泄降", "地下水面"]


def township_ref(s: str, b: PageBundle) -> bool:
    """A page township (any name, or its bare name), or the words township/town/village/鄉/鎮."""
    c = Q.ctx(b)
    for t in c.towns:
        if t.get("pooled"):
            continue
        names = [t["en"], t["zh"], re.sub("[鄉鎮市]$", "", t["zh"])]
        for n in names:
            if n and len(n) >= 2 and re.search(r"(?<![a-z])" + re.escape(n) + r"(?![a-z])", s, re.I):
                return True
    return _has(TOWN_WORDS, s)


def township_head(s: str, b: PageBundle) -> bool:
    return township_ref(s, b) and _has(HEAD_WORDS, s) and not subsidence_words(s, units=False)
