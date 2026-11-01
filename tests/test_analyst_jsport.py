"""Golden cases for the JS formatter port (browser-checked by the parity run)."""
from __future__ import annotations

from hydrophysics_analyst import jsport as J


def test_to_fixed_uses_the_binary_value_and_half_up():
    assert J.to_fixed(0.125, 2) == "0.13"
    assert J.to_fixed(1.005, 2) == "1.00"
    assert J.to_fixed(-0.004, 2) == "-0.00"
    assert J.to_fixed(-0.0, 2) == "0.00"
    assert J.to_fixed(2.5, 0) == "3"


def test_math_round():
    assert J.js_round(0.49999999999999994) == 0
    assert J.js_round(-0.5) == 0
    assert J.js_round(2.5) == 3
    assert J.js_round(-2.5) == -2


def test_number_to_string():
    assert J.js_num_str(0.7) == "0.7"
    assert J.js_num_str(1.0) == "1"
    assert J.js_num_str(12) == "12"
    assert J.js_num_str(4.6) == "4.6"


def test_page_formatters():
    assert J.sg(0.054) == "+0.05" and J.sg(-0.004) == "−0.00" and J.sg(0) == "0.00"
    assert J.f1(24.04) == "24.0" and J.f2(0.125) == "0.13"
    assert J.pctf(9.2) == "9%"
    assert J.inv(1 / 58008) == "1/58,008"
    assert J.locale_en(1234) == "1,234"


def test_tr_fallbacks_and_substitution():
    T = {"en": {"a": "x {v} y {v}", "b": "only en"}, "zh": {"a": "甲{v}"}}
    assert J.tr(T, "en", "a", {"v": 0.5}) == "x 0.5 y 0.5"
    assert J.tr(T, "zh", "b") == "only en"
    assert J.tr(T, "zh", "missing") == "missing"
    assert J.tr(T, "en", "a", {"v": None}) == "x undefined y undefined"
    assert J.strip_tags("<b>a</b> &amp; b") == "a & b"
