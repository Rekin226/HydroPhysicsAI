"""The analyst package: importable without the SDK, outside the hashed hydrophysics/ tree."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


def test_imports_without_anthropic(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", None)
    for m in [k for k in sys.modules if k.startswith("hydrophysics_analyst")]:
        monkeypatch.delitem(sys.modules, m)
    import hydrophysics_analyst
    import hydrophysics_analyst.presets  # noqa: F401
    import hydrophysics_analyst.validate  # noqa: F401
    assert hydrophysics_analyst.__version__


def test_help_lists_commands(capsys):
    from hydrophysics_analyst.__main__ import main
    assert main(["--help"]) == 0
    out = capsys.readouterr().out
    for cmd in ("preset", "ask", "serve", "replay", "eval", "parity", "mcp", "bundle"):
        assert cmd in out


def test_outside_the_release_code_hash():
    """release.provenance hashes every file under hydrophysics/ (release.py:41)."""
    import hydrophysics_analyst
    from hydrophysics.twin import release
    hashed = Path(release.__file__).resolve().parents[1]
    pkg = Path(hydrophysics_analyst.__file__).resolve().parent
    assert hashed not in pkg.parents and pkg != hashed


def test_subpackages_are_found():
    setuptools = pytest.importorskip("setuptools")     # not installed in every CI job
    root = Path(__file__).resolve().parents[1]
    found = setuptools.find_packages(where=str(root), include=["hydrophysics*"])
    assert {"hydrophysics_analyst", "hydrophysics_analyst.router",
            "hydrophysics_analyst.llm"} <= set(found)
