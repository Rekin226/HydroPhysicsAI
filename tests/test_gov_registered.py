"""GOV.heads_window_product.v1 as registered: the registration is in INDEX.jsonl and its
recorded sha256 of heads_window.py and of the MANIFEST template match the current code. A new
file (tests/test_gov_heads_window.py is a GOV-bound input and is not edited). Skips without
the gitignored registry (results/dataprep/preregistration), as in CI.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PREREG = REPO / "results" / "dataprep" / "preregistration"
GOV_ID = "GOV.heads_window_product.v1"
HW_PATH = "dataprep_staging/gov_heads_window/heads_window.py"

pytestmark = pytest.mark.skipif(not (PREREG / "INDEX.jsonl").exists(),
                                reason="the registry (results/, gitignored) is absent")


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _registered() -> tuple[dict, dict]:
    rows = [json.loads(s) for s in (PREREG / "INDEX.jsonl").read_text(
        encoding="utf-8").splitlines() if s.strip()]
    rec = [r for r in rows if r.get("id") == GOV_ID]
    assert len(rec) == 1, f"{GOV_ID} is not uniquely in INDEX.jsonl"
    path = PREREG / f"{GOV_ID}.json"
    assert _sha(path) == rec[0]["sha256"], "the registered file changed"
    return json.loads(path.read_text(encoding="utf-8")), rec[0]


def test_gov_is_registered_and_heads_window_is_the_registered_file():
    doc, _ = _registered()
    rec = [i for i in doc["inputs"] if i["path"] == HW_PATH]
    assert len(rec) == 1
    assert rec[0]["sha256"] == _sha(REPO / HW_PATH)
    assert rec[0]["sha256"].startswith("9a953039")


def test_gov_template_sha_matches_the_current_builder():
    doc, _ = _registered()
    saved = list(sys.path)
    try:
        spec = importlib.util.spec_from_file_location("test_gov_registered_hw", REPO / HW_PATH)
        hw = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hw)
        text = hw.template_text()          # imports common.uses via heads_window._paths()
    finally:
        sys.path[:] = saved
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == doc["template"]["sha256"]


GOV_TEST = "tests/test_gov_heads_window.py"


def test_registered_state_replaces_the_draft_state_check():
    """The registered state that tests/test_gov_heads_window.py's draft-state check cannot see
    (that file is a hash-bound input of GOV and is not edited; tests/conftest.py marks its
    'not registered' check xfail(strict) once GOV is registered): the registered file is the
    INDEX entry, the bound test file is byte-identical to its registered input, and the local
    draft copy, when present, is still what the draft builder produces."""
    doc, rec = _registered()
    bound = [i for i in doc["inputs"] if i["path"] == GOV_TEST]
    assert len(bound) == 1 and bound[0]["sha256"] == _sha(REPO / GOV_TEST)
    assert doc["governance_tests"] == GOV_TEST and doc["id"] == GOV_ID
    saved = list(sys.path)
    try:
        spec = importlib.util.spec_from_file_location("test_gov_registered_hw2", REPO / HW_PATH)
        hw = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hw)
        built = hw.draft_declaration()
        if hw.DRAFT.exists():
            assert json.loads(hw.DRAFT.read_text(encoding="utf-8")) == built
    finally:
        sys.path[:] = saved
    assert built["id"] == GOV_ID == rec["id"]
