"""JS-vs-Python parity on a synthetic page (local: needs Playwright and a Chromium).

Set ``ANALYST_PAGE`` to also run it on a real built page (dev mode, unbound).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from hydrophysics_analyst import testing
from hydrophysics_analyst.page import PageBundle

pytest.importorskip("playwright.sync_api")


def _run(page: Path, b: PageBundle, n: int):
    from hydrophysics_analyst import parity
    try:
        return parity.run_parity(page, b, n_random=n)
    except Exception as e:  # noqa: BLE001
        if "Executable doesn't exist" in str(e) or "launch" in str(e).lower():
            pytest.skip(f"no Chromium for Playwright: {type(e).__name__}")
        raise


def test_synthetic_page_parity(tmp_path):
    page = testing.build_synthetic_page(tmp_path)
    b = PageBundle.load(page, testing.synthetic_ledger(tmp_path, page), allow_pending=True)
    rep = _run(page, b, 6)
    assert rep.n_cases > 100 and rep.n_mismatch == 0, rep.mismatches[:3]


@pytest.mark.skipif(not os.environ.get("ANALYST_PAGE"), reason="ANALYST_PAGE not set")
def test_real_page_parity():
    page = Path(os.environ["ANALYST_PAGE"])
    b = PageBundle.load(page, check_forward=False, check_template=False, allow_pending=True)
    rep = _run(page, b, 30)
    assert rep.n_mismatch == 0, rep.mismatches[:3]
