"""Registry-state marks for tests whose files are hash-bound inputs of a registration.

tests/test_gov_heads_window.py is an input of GOV.heads_window_product.v1 (its sha256 is in the
registered document), so it is never edited. Its draft-state check
test_draft_declaration_is_not_registered_and_the_local_copy_is_current asserts that GOV is not
yet registered; once GOV is in the registry that check fails by construction. It is then marked
xfail(strict=True): it must still fail (an unregistered registry turns it into a strict XPASS
failure), and tests/test_gov_registered.py asserts the registered state exactly. Without the
gitignored registry (CI) the test skips as before and nothing here applies.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

_INDEX = Path(__file__).resolve().parents[1] / "results" / "dataprep" / "preregistration" / \
    "INDEX.jsonl"
_GOV_ID = "GOV.heads_window_product.v1"
_GOV_DRAFT_TEST = ("test_gov_heads_window.py",
                   "test_draft_declaration_is_not_registered_and_the_local_copy_is_current")


def _registered_ids() -> set[str]:
    if not _INDEX.is_file():
        return set()
    return {json.loads(s)["id"] for s in _INDEX.read_text(encoding="utf-8").splitlines()
            if s.strip()}


def pytest_collection_modifyitems(config, items):
    gov_registered = None
    for item in items:
        if (Path(str(item.fspath)).name, item.name) != _GOV_DRAFT_TEST:
            continue
        if gov_registered is None:
            gov_registered = _GOV_ID in _registered_ids()
        if gov_registered:
            item.add_marker(pytest.mark.xfail(
                strict=True, reason=f"{_GOV_ID} is registered (a draft-state check of a "
                "hash-bound test file); tests/test_gov_registered.py asserts the registered "
                "state"))
