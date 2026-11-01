"""Exact ports of the page's JavaScript number formatting and ``tr``.

Source: hydrophysics/twin/app/template.html (``tr`` and the formatters near line 615).
Parity with the browser is checked by ``hydrophysics_analyst.parity``.
"""
from __future__ import annotations

import math
import re
from decimal import ROUND_HALF_UP, Decimal

MINUS = "−"


def to_fixed(x: float, d: int) -> str:
    """``Number.prototype.toFixed`` for |x| < 1e21.

    ECMA-262: if x < 0 emit "-" and format -x; digits come from the exact binary value with
    ties rounded up. So ``(-0.004).toFixed(2) == "-0.00"`` but ``(-0).toFixed(2) == "0.00"``.
    """
    if x != x:
        return "NaN"
    if math.isinf(x):
        return "Infinity" if x > 0 else "-Infinity"
    q = Decimal(abs(x)).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)
    return ("-" if x < 0 else "") + f"{q:.{d}f}"


def js_round(x: float) -> float:
    """``Math.round``: round half toward +infinity. ``floor(x + 0.5)`` fails at 0.49999999999999994."""
    if x != x or math.isinf(x):
        return x
    f = math.floor(x)
    return float(f + (1 if x - f >= 0.5 else 0))


def js_num_str(x: float | int) -> str:
    """``String(number)`` for the values the page formats (integers and short decimals)."""
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, int):
        return str(x)
    if x != x:
        return "NaN"
    if math.isinf(x):
        return "Infinity" if x > 0 else "-Infinity"
    if x == 0:
        return "0"
    if float(x).is_integer() and abs(x) < 1e21:
        return str(int(x))
    r = repr(float(x))                      # shortest round-trip, like JS
    if "e" in r:
        mant, exp = r.split("e")
        e = int(exp)
        if -7 < e < 21:
            return format(Decimal(r), "f")
        return f"{mant}e{'+' if e > 0 else '-'}{abs(e)}"
    return r


def locale_en(x: float) -> str:
    """``x.toLocaleString("en")`` for integers / the values the page passes (max 3 decimals)."""
    if x != x:
        return "NaN"
    q = Decimal(repr(float(x))).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP).normalize()
    s = f"{q:,f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def f1(v: float) -> str:
    return to_fixed(js_round(v * 10) / 10, 1)


def f2(v: float) -> str:
    return to_fixed(js_round(v * 100) / 100, 2)


def sg(v: float, d: int = 2) -> str:
    return ("+" if v > 0 else MINUS if v < 0 else "") + to_fixed(abs(v), d)


def pctf(v: float) -> str:
    return js_num_str(js_round(v)) + "%"


def inv(v: float) -> str:
    return "1/" + locale_en(js_round(1 / v)) if v > 0 else "–"


def r3(v: float | None) -> str:
    return "–" if v is None else sg(v, 3)


def _sub_str(v) -> str:
    if v is None:
        return "undefined"
    if isinstance(v, str):
        return v
    return js_num_str(v)


def tr(T: dict[str, dict[str, str]], lang: str, key: str, v: dict | None = None) -> str:
    """The page's ``tr(k, v)``: ``T[LANG][k]`` → ``T.en[k]`` → the key; ``{a}`` substitution."""
    s = T.get(lang, {}).get(key)
    if s is None:
        s = T.get("en", {}).get(key)
        if s is None:
            return key
    if v:
        for a, val in v.items():                 # JS: s.split("{a}").join(v[a])
            s = _sub_str(val).join(s.split("{" + a + "}"))
    return s


def lc(s: str) -> str:
    return re.sub(r"^[A-Z]", lambda m: m.group(0).lower(), s)


def uc(s: str) -> str:
    return re.sub(r"^[a-z]", lambda m: m.group(0).upper(), s)


def strip_tags(s: str) -> str:
    """What ``textContent`` shows for the page's simple markup (<b>, <i>, <span>)."""
    s = re.sub(r"<[^>]+>", "", s)
    return (s.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')
            .replace("&amp;", "&"))
