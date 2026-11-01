"""The command line on a synthetic page (no LLM)."""
from __future__ import annotations

import json

import pytest

from hydrophysics_analyst import cli, testing


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    d = tmp_path_factory.mktemp("cli")
    page = testing.build_synthetic_page(d)
    return page, testing.synthetic_ledger(d, page)


def test_preset_json(built, capsys):
    page, led = built
    rc = cli.main(["preset", "policy_head", "--cut", "aquaculture=30", "--start", "2030", "--year", "2032",
                   "--lang", "zh", "--page", str(page), "--ledger", str(led), "--allow-pending", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0 and out["violations"] == [] and out["preset"] == "policy_head"
    assert "#p=1,0.7" in out["deep_link"] and "C01" in dict(map(tuple, out["citations"]))


def test_binding_failure_exits_3(built, tmp_path, capsys):
    page, _ = built
    led = json.loads((built[1]).read_text(encoding="utf-8"))
    led["page"]["template_sha256"] = "0" * 64
    bad = tmp_path / "l.json"
    bad.write_text(json.dumps(led), encoding="utf-8")
    assert cli.main(["preset", "solved_compare", "--year", "2032", "--page", str(page), "--ledger", str(bad),
                     "--allow-pending"]) == 3
