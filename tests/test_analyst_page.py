"""Binding the analyst to a built page and the ledger (spec §4.1), on a synthetic page."""
from __future__ import annotations

import json

import numpy as np
import pytest

from hydrophysics.twin import claims
from hydrophysics.twin.app import prep
from hydrophysics_analyst import testing
from hydrophysics_analyst.page import TEMPLATE, BindingError, PageBundle, parse_T, unrender


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    d = tmp_path_factory.mktemp("page")
    page = testing.build_synthetic_page(d)
    return page, testing.synthetic_ledger(d, page)


def test_synthetic_page_has_the_injected_blocks(built):
    from hydrophysics.twin.release import read_payload
    d = read_payload(built[0])
    assert d["n2Unit"]["n"] == 84 and len(d["n2Unit"]["basis"]) == 12
    assert d["hsr"] is not None and d["classes"][0]["gwh"] is not None
    assert sum(not t["pooled"] for t in d["towns"]) >= 2


def test_parse_T_equals_claims_i18n_on_the_template():
    text = TEMPLATE.read_text(encoding="utf-8")
    for lang in ("en", "zh"):
        assert parse_T(text, lang) == claims.i18n(TEMPLATE, lang)


def test_unrender_recovers_the_template(built):
    assert unrender(built[0].read_text(encoding="utf-8")) == TEMPLATE.read_text(encoding="utf-8")


def test_full_binding_passes_and_arrays_are_float32(built):
    b = PageBundle.load(built[0], built[1], allow_pending=True)
    a = b.array("subsBase")
    assert a.dtype == np.float32
    ref = prep.unpack(b.payload["arrays"]["subsBase"]).reshape(-1)
    assert np.allclose(a, ref.astype(np.float32), equal_nan=True)


def test_a_one_byte_edit_to_the_page_js_fails_binding(built, tmp_path):
    p = tmp_path / "edited.html"
    text = built[0].read_text(encoding="utf-8").replace('"use strict";', '"use strict"; ', 1)
    p.write_text(text, encoding="utf-8")
    with pytest.raises(BindingError, match="template"):
        PageBundle.load(p, built[1], allow_pending=True)


def test_forward_sha_mismatch_fails(built, tmp_path):
    led = json.loads(built[1].read_text(encoding="utf-8"))
    led["page"]["forward_sha256"] = "0" * 64
    bad = tmp_path / "led.json"
    bad.write_text(json.dumps(led), encoding="utf-8")
    with pytest.raises(BindingError, match="forward"):
        PageBundle.load(built[0], bad, allow_pending=True)


def test_deploy_mode_refuses_dev_switches(built):
    with pytest.raises(BindingError):
        PageBundle.load(built[0], built[1], mode="deploy", allow_pending=True)
    with pytest.raises(BindingError):
        PageBundle.load(built[0], built[1], mode="deploy", check_forward=False)


def test_individual_observations_are_never_read(built):
    b = PageBundle.load(built[0], built[1], allow_pending=True)
    for name in ("wells", "leveling"):
        with pytest.raises(BindingError):
            b.array(name)
        with pytest.raises(BindingError):
            b[name]
