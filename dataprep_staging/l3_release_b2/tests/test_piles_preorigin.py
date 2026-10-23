"""Unit tests of release_20261012b2: the pre-origin publication filter of the printed piles
(synthetic rows, as in release b), the narrowed pile uses of user ruling (b) and the recording
of both user rulings of 2026-10-18."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from common import guard
from l3_release_b2 import components as K
from l3_release_b2 import config as C


def _source(tmp_path, monkeypatch):
    src = {"files": [
        {"file": "listing_p1.html", "note": "listing page"},
        {"file": "cy_2022-11.pdf", "published_roc": "111/12"},
        {"file": "cy_2022-12.pdf", "published_roc": "112/1"},
        {"file": "cy_2022-10.pdf", "published_roc": "111/11",
         "note": "[{'title': 'x', 'published_roc': '111/10'}]"},
        {"file": "cy_2023-10.pdf", "published_roc": "112/11"},
    ]}
    p = tmp_path / "SOURCE.json"
    p.write_text(json.dumps(src), encoding="utf-8")
    monkeypatch.setattr(K, "REPO", tmp_path)
    monkeypatch.setattr(K.C, "PILE_PUBLICATION_SOURCE", "SOURCE.json")


def _row(month, report, vintage, reprint=None, rv=None, n_ver=1):
    two = reprint is not None
    return {"sensor_id": "S", "month": month, "value": 1.0, "sd": 0.03, "flag": 0,
            "report": report, "report_caption_month": vintage, "vintage_month": vintage,
            "n_printings": 2 if two else 1, "n_verified_printings": n_ver,
            "reprint_value": 1.1 if two else np.nan, "reprint_report": reprint or "",
            "reprint_vintage_month": rv or "", "reprint_diff_mm": 0.1 if two else np.nan,
            "method": f"primary printing of {2 if two else 1} printing(s); sd rounding"}


def test_publication_months(tmp_path, monkeypatch):
    _source(tmp_path, monkeypatch)
    pub = K.pile_publication()
    assert pub == {"cy_2022-11.pdf": "2022-12", "cy_2022-12.pdf": "2023-01",
                   "cy_2022-10.pdf": "2022-11", "cy_2023-10.pdf": "2023-11"}


def test_filter_drops_late_primaries_and_reprints(tmp_path, monkeypatch):
    _source(tmp_path, monkeypatch)
    df = pd.DataFrame([
        _row("2022-11", "cy_2022-11.pdf", "2022-11"),                       # kept
        _row("2022-12", "cy_2022-12.pdf", "2022-12"),                       # published 2023-01
        _row("2022-10", "cy_2023-10.pdf", "2023-10"),                       # published 2023-11
        _row("2022-09", "cy_2022-10.pdf", "2022-09", "cy_2023-10.pdf", "2023-10", n_ver=2),
    ])
    ctx: dict = {}
    out, notes, strict = K.transform_piles_preorigin(df, "preorigin", None, ctx)
    assert list(out["month"]) == ["2022-11", "2022-09"]
    r = out.iloc[1]
    assert r["n_printings"] == 1 and r["n_verified_printings"] == 1
    assert np.isnan(r["reprint_value"]) and r["reprint_report"] is None
    assert "of 1 printing(s) published by 2022-12" in r["method"]
    assert ctx["pile_publication"]["dropped_rows"] == 2
    assert ctx["pile_publication"]["reprints_removed"] == 1
    errs, summ = K.check_table(out, "preorigin", strict_columns=False)
    assert not [e for e in errs if "published after" in e]
    assert summ["latest_dated_values"]["report_published"] == "2022-12"


def test_check_table_flags_late_publication():
    df = pd.DataFrame({"month": ["2022-10"], "report_published": ["2023-01"]})
    errs, _ = K.check_table(df, "preorigin", strict_columns=False)
    assert any("report_published reaches 2023-01" in e for e in errs)
    errs, _ = K.check_table(df, "full", strict_columns=False)
    assert not errs


def test_unknown_report_refused(tmp_path, monkeypatch):
    _source(tmp_path, monkeypatch)
    df = pd.DataFrame([_row("2022-05", "cy_2022-05.pdf", "2022-05")])
    with pytest.raises(K.ComponentError):
        K.transform_piles_preorigin(df, "preorigin", None, {})


def test_pinned_source_requires_exact_recorded_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(K, "REPO", tmp_path)
    m = {"git_commit": "abc"}
    rec = {"path": "pkg/mod.py", "sha256": "f" * 64, "git_state": "committed"}
    monkeypatch.setattr(K, "_git_blob_sha256", lambda c, p: "f" * 64)
    assert K._pinned_source(m, rec)["ref"] == "abc:pkg/mod.py"
    monkeypatch.setattr(K, "_git_blob_sha256", lambda c, p: "0" * 64)
    assert K._pinned_source(m, rec) is None
    assert K._pinned_source(m, dict(rec, git_state="dirty")) is None
    data = b"registered bytes\n"
    import hashlib
    sha = hashlib.sha256(data).hexdigest()
    arch = tmp_path / "results/dataprep/reports/X_evidence/code_as_registered" / sha[:12]
    arch.mkdir(parents=True)
    (arch / "mod.py").write_bytes(data)
    got = K._pinned_source(m, {"path": "pkg/mod.py", "sha256": sha, "git_state": "untracked"})
    assert got["source"] == "archive"
    assert K._pinned_source(m, {"path": "pkg/mod.py", "sha256": "1" * 64,
                                "git_state": "untracked"}) is None


# ----------------------------------------------------------------------------- ruling (b)
def _spec(variant, name):
    return next(s for s in C.COMPONENTS if s["variant"] == variant and s["name"] == name)


def _component_manifest(spec):
    """The guard-relevant part of the manifest the builder writes for a component."""
    return {"product": f"model_ready/{C.RELEASE}/{spec['variant']}/{spec['name']}",
            "version": "v1", "fit_window_end": spec["fit_window_end"], "exemptions": [],
            "allowed_uses": list(spec["allowed_uses"]),
            "forbidden_uses": sorted(set(K.U.USES) - set(spec["allowed_uses"])),
            "gate": {"passed": True}}


def test_pile_uses_narrowed_by_ruling():
    sp = _spec("preorigin", "piles_printed")
    assert C.PILE_PRE_USES == ["validator", "after_the_fact_analysis"]
    assert sp["allowed_uses"] == ["validator", "after_the_fact_analysis"]
    assert [rd["use"] for rd in sp["reads"]] == ["validator"]
    assert sp["reads"][0]["window_end"] == "2022-12"
    assert sp["user_rulings"] == [C.RULING_PILES]
    # the full variant's piles are not touched by the ruling
    assert _spec("full", "piles_printed")["allowed_uses"] == [
        "validator", "scoring_target", "after_the_fact_analysis"]


def test_pile_component_refuses_design_time_evidence():
    m = _component_manifest(_spec("preorigin", "piles_printed"))
    for use in ("design_time_evidence", "scoring_target", "calibration_target", "datum_fit",
                "baseline", "selection"):
        with pytest.raises(guard.UseNotAllowed):
            guard.check_use(m, use, require_gate=True if use in K.U.GATE_EXEMPT_USES else None)
    for use in ("validator", "after_the_fact_analysis"):
        guard.check_use(m, use, require_gate=True if use in K.U.GATE_EXEMPT_USES else None)


def test_built_pile_component_refuses_design_time_evidence():
    d = K.REPO / C.RELEASE_DIR / C.RELEASE / "preorigin" / "piles_printed" / "v1"
    if not (d / "MANIFEST.json").is_file():
        pytest.skip(f"{C.RELEASE} not built")
    with pytest.raises(guard.UseNotAllowed):
        guard.open_product(d, "design_time_evidence", require_gate=True)
    guard.open_product(d, "validator")


def test_every_read_uses_an_allowed_use():
    for sp in C.COMPONENTS:
        for rd in sp["reads"]:
            assert rd["use"] in sp["allowed_uses"], (sp["variant"], sp["name"], rd["use"])


def test_uses_check_refuses_a_read_outside_the_allowed_uses():
    sp = dict(_spec("preorigin", "piles_printed"))
    sp["reads"] = [dict(sp["reads"][0], use="design_time_evidence", window_end=None)]
    up_m = {"product": "x", "version": "v1", "fit_window_end": "2022-12", "exemptions": [],
            "allowed_uses": sorted(K.U.USES), "forbidden_uses": [], "gate": {"passed": True}}
    _, errs = K.uses_check(sp, up_m)
    assert any("read use 'design_time_evidence' is not an allowed use" in e for e in errs)
    sp["reads"] = [dict(sp["reads"][0], use="validator")]
    _, errs = K.uses_check(sp, up_m)
    assert not errs


# ----------------------------------------------------------------------------- ruling records
def test_rulings_recorded_on_the_affected_components():
    w1 = _spec("preorigin", "surface_water")
    assert w1["user_rulings"] == [C.RULING_W1]
    assert any(n.startswith("USER RULING 2026-10-18 (a), a ruling, not a gate pass")
               for n in w1["notes"])
    assert "not a gate pass" in C.RULING_W1["status"]
    assert "1.290 -> 1.264 m" in C.RULING_W1["basis"]
    assert "1.848 -> 1.777 m" in C.RULING_W1["basis"]
    piles = _spec("preorigin", "piles_printed")
    assert any(n.startswith("USER RULING 2026-10-18 (b)") for n in piles["notes"])
    others = [s for s in C.COMPONENTS if s is not w1 and s is not piles]
    assert not any(s.get("user_rulings") for s in others)
    assert [d["id"] for d in C.OPEN_USER_DECISIONS] == ["open_c_pile_publication_date"]
    assert "not ruled on" in C.OPEN_USER_DECISIONS[0]["status"]
