"""Unit tests of the pre-origin publication filter of release_20261012b (synthetic rows)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from l3_release_b import components as K


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
