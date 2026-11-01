"""CPU tests of U1.HB.power.v1 (power_u1hb.py, prereg_u1hb_power.py). Synthetic draws only;
nothing is registered in the real registry and no look is logged (registrations go to
pytest's tmp dirs). Run from the repository root:

    ~/miniforge3/envs/hydro/bin/python -m pytest -q -p no:cacheprovider \\
        dataprep_staging/u1_bands/tests
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

TESTS = Path(__file__).resolve().parent
X = TESTS.parent
REPO = X.parents[1]
for p in (str(X), str(REPO / "dataprep_staging")):
    if p not in sys.path:
        sys.path.insert(0, p)
import power_u1hb as P  # noqa: E402
import prereg_u1hb_power as D  # noqa: E402

HAVE_E02 = (REPO / P.PROTOCOL).exists() and (REPO / D.E02 / "manifest.json").exists()
needs_e02 = pytest.mark.skipif(not HAVE_E02, reason="results/twin/prospective_2026-11 absent")


def test_design_is_json_stable_and_grid_is_declared():
    d = P.design_spec()
    assert json.loads(json.dumps(d)) == d
    g = d["grid"]
    assert g["icc_well"] == [0.0, 0.1, 0.3, 0.5, 0.7, "heavy"]
    assert g["icc_month"] == [0.0, 0.05, 0.15, 0.3]
    assert g["coverage"] == [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98]
    assert d["replicates_per_cell"] == 10_000
    assert d["g1"]["module"] == "harness.coverage3" and d["g1"]["rule"] == "gate-3"
    assert d["g1b"]["role"] == "report-only"
    cells = P.cells()
    assert len(cells) == 6 * 4 * 3 * 7 == 504
    assert [c["cell"] for c in cells] == list(range(504))
    assert cells[1]["coverage"] == 0.75 and cells[7]["icc_year"] == 0.1


@pytest.mark.parametrize("cell", [(0.0, 0.0, 0.0, 0.9), (0.7, 0.3, 0.2, 0.75),
                                  ("heavy", 0.15, 0.1, 0.8), (0.3, 0.05, 0.0, 0.98)])
def test_generator_marginal_is_the_declared_coverage(cell):
    aw, am, ay, p = cell
    rng = np.random.default_rng(7)
    h = P.draw_hits(rng, 3000, p, aw, am, ay)
    assert h.shape == (3000, 158, 12) and h.dtype == bool
    vals = h.mean(axis=(1, 2))
    se = vals.std(ddof=1) / math.sqrt(vals.size)
    assert abs(vals.mean() - p) <= 4.5 * max(se, 1e-6)


def test_heavy_arm_has_all_hit_wells_and_bad_wells():
    rng = np.random.default_rng(3)
    p = 0.9
    h = P.draw_hits(rng, 500, p, "heavy", 0.0, 0.0)
    all_hit = h.all(axis=2).mean()
    pi = (1 - p) / (1 - P.HEAVY_P_BAD)
    assert abs(all_hit - (1 - pi)) < 0.01      # bad wells almost never hit all 12 months


def test_draw_rejects_invalid_shares():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        P.draw_hits(rng, 2, 0.9, 0.8, 0.3, 0.0)
    with pytest.raises(ValueError):
        P.draw_hits(rng, 2, 1.0, 0.0, 0.0, 0.0)


def test_self_check_parity_and_bias():
    chk = P.self_check(n_parity=8, n_bias=1000)
    assert chk["ok"], chk
    assert all(r["mismatches"] == 0 for r in chk["parity"])


def test_icc_zero_scores_are_binomial():
    c = {"cell": 0, "icc_well": 0.0, "icc_month": 0.0, "icc_year": 0.0, "coverage": 0.9}
    r = P.run_cell(c, 2000, 11, chunk=500)
    se = math.sqrt(0.9 * 0.1 / 1896)
    assert abs(r["sd_coverage_hat"] - se) < 0.15 * se
    assert abs(r["mean_coverage_hat"] - 0.9) < 4 * se / math.sqrt(2000)
    y = r["yunlin"]
    assert y["rho_anova_well_mean"] < 0.02 and y["window_applies"] == 1.0
    assert 0 <= y["P_fail"]["p"] <= 1 and y["P_fail"]["mc_se"] > 0


def _fake(cov, am, fail_yl, fail_ch=None, g1b=0.3):
    fc = fail_yl if fail_ch is None else fail_ch
    mk = lambda f: {"P_fail": {"p": f, "mc_se": 0.001}, "P_g1b_fail": {"p": g1b}}  # noqa: E731
    return {"cell": 0, "icc_well": 0.0, "icc_month": am, "icc_year": 0.0, "coverage": cov,
            "yunlin": mk(fail_yl), "changhua": mk(fc)}


def test_wording_rules_two_failure_modes():
    # calibrated interval fails often; under-coverage is caught: shock wording, plain pass
    w = P.wording([_fake(0.90, 0.0, 0.25), _fake(0.75, 0.15, 0.95), _fake(0.75, 0.3, 0.1)])
    assert w["yunlin"]["fail_wording"] == "fail_shock"
    assert w["yunlin"]["pass_wording"] == "pass_plain"       # the 0.3 cell is excluded
    # false fail low, power low: plain fail, weak pass
    w = P.wording([_fake(0.90, 0.0, 0.05), _fake(0.75, 0.05, 0.6, 0.9)])
    assert w["yunlin"]["fail_wording"] == "fail_plain"
    assert w["yunlin"]["pass_wording"] == "pass_weak"
    assert w["changhua"]["pass_wording"] == "pass_plain"
    assert w["yunlin"]["pass_wording_text"]["en"] == "no gross under-coverage detected"
    # the thresholds are strict: exactly 0.10 is not above, exactly 0.80 is not below
    w = P.wording([_fake(0.90, 0.0, 0.10), _fake(0.75, 0.0, 0.80)])
    assert (w["yunlin"]["fail_wording"], w["yunlin"]["pass_wording"]) == ("fail_plain",
                                                                          "pass_plain")


def test_fail_shock_keeps_the_failure_and_slices_are_reported():
    for lang in ("en", "zh"):
        plain, shock = P.WORDING["fail_plain"][lang], P.WORDING["fail_shock"][lang]
        assert shock == plain + P.WORDING["fail_shock_suffix"][lang]
        assert "{x}" in shock
    assert P.WORDING["fail_shock"]["en"].startswith("failed its registered gate")
    cells = [dict(_fake(0.90, 0.0, 0.05), icc_year=0.0), dict(_fake(0.90, 0.0, 0.4),
                                                              icc_year=0.2),
             dict(_fake(0.75, 0.0, 0.9), icc_year=0.0), dict(_fake(0.75, 0.0, 0.7),
                                                             icc_year=0.2)]
    w = P.wording(cells)["yunlin"]
    assert w["by_icc_year_report_only"] == {
        "0": {"false_fail_max": 0.05, "power_min_at_0.75": 0.9},
        "0.2": {"false_fail_max": 0.4, "power_min_at_0.75": 0.7}}
    assert (w["fail_wording"], w["pass_wording"]) == ("fail_shock", "pass_weak")
    assert "neither turns a FAIL into a pass" in P.WORDING_RULES["direction"]


def test_engineering_failure_note_matches_the_declaration():
    d = D.doc()
    assert "new declaration id" in P.ENGINEERING_FAILURE_NOTE
    assert "new declaration id" in d["self_checks"]["on_failure"]
    assert "5-10 minutes" in d["run"]["cost"]
    assert "fail_shock" in d["expected_outcome_disclosure"]["fail"]


def test_no_wording_says_calibrated():
    for k, v in P.WORDING.items():
        if k == "never":
            continue
        for lang in ("en", "zh"):
            assert not any(word in v[lang] for word in P.WORDING["never"]), (k, lang)


def test_smoke_cli_writes_and_never_touches_the_registered_dir(tmp_path):
    assert P.main(["smoke", "--out", str(tmp_path / "s")]) == 0
    res = json.loads((tmp_path / "s" / "smoke_result.json").read_text())
    assert res["smoke"] is True and res["engineering_failure"] is False
    assert res["seed"] == P.SMOKE_SEED != P.SEED
    assert len(res["cells"]) == 4 and (tmp_path / "s" / "smoke_result.md").exists()
    assert P.main(["smoke", "--out", str(REPO / P.RESULT_DIR)]) == 1
    assert not (REPO / P.RESULT_DIR / "smoke_result.json").exists()


def test_run_refuses_without_registration(tmp_path):
    with pytest.raises(SystemExit, match="not registered"):
        P.registered_declaration(tmp_path)


@needs_e02
def test_declaration_validates_and_round_trips_in_tmp_registry(tmp_path, monkeypatch):
    from harness import prereg3

    d = D.doc()
    assert prereg3.validate_prereg(d) == []
    assert d["design"] == P.design_spec() and d["code_sha256"] == P.code_sha256()
    assert d["kind"] == "declaration" and d["id"] == "U1.HB.power.v1"
    assert P.consistency_problems() == []
    reg = tmp_path / "prereg"
    path, h = prereg3.write_prereg(d, prereg_dir=reg, rows_dir=tmp_path / "rows",
                                   registries=[])
    got, sha = P.registered_declaration(reg)
    assert sha == h and got["design"] == P.design_spec()
    # a code change after registration is refused
    monkeypatch.setattr(P, "code_sha256", lambda: {**d["code_sha256"],
                                                   P.CODE_FILES[1]: "0" * 64})
    with pytest.raises(SystemExit, match="code changed"):
        P.registered_declaration(reg)
    monkeypatch.undo()
    # a design change after registration is refused
    monkeypatch.setattr(P, "REPLICATES", 500)
    with pytest.raises(SystemExit, match="design differs"):
        P.registered_declaration(reg)


@needs_e02
def test_prereg_cli_validate_is_a_dry_run(capsys):
    if (REPO / "results/dataprep/preregistration/U1.HB.power.v1.json").exists():
        pytest.skip("U1.HB.power.v1 is registered; the dry-run check applies before registration")
    before = sorted((REPO / "results/dataprep/preregistration").glob("U1.*"))
    assert D.main(["--validate", "--quiet"]) == 0
    out = capsys.readouterr().out
    assert "validate_prereg: valid" in out and "not written" in out
    assert sorted((REPO / "results/dataprep/preregistration").glob("U1.*")) == before
    assert D.main(["--write", "--expect-self-sha", "0" * 64]) == 1     # refused, not written
    assert sorted((REPO / "results/dataprep/preregistration").glob("U1.*")) == before
