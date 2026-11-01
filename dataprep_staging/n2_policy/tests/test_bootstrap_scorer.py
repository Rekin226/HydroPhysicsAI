"""Tests of the shared N2/N2C scorer (dataprep_staging/n2_policy/bootstrap_scorer.py) on
synthetic panels only (no head, no forward file, nothing written outside pytest's tmp dirs):

    cd <repository root>
    ~/miniforge3/envs/hydro/bin/python -m pytest -q -p no:cacheprovider \\
        dataprep_staging/n2_policy/tests
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

TESTS = Path(__file__).resolve().parent
N2 = TESTS.parent
REPO = N2.parents[1]
sys.path.insert(0, str(N2))
sys.path.insert(0, str(REPO / "dataprep_staging"))
import bootstrap_scorer as S  # noqa: E402

PREREG = REPO / "results/dataprep/preregistration"
N2_ID = "N2.policy_response.prospective_2026-11.v1"
N2C_ID = "N2C.changhua_response.prospective_2026-11.v1"


def panel(n_wells=40, n_months=12, beta=1.0, noise=0.3, seed=0, offset=True):
    rng = np.random.default_rng(seed)
    wells = np.repeat([f"w{i:03d}" for i in range(n_wells)], n_months)
    months = np.tile([f"{2026 + (10 + k) // 12}-{(10 + k) % 12 + 1:02d}"
                      for k in range(n_months)], n_wells)
    c = rng.normal(0, 0.5, n_wells * n_months) + np.repeat(rng.normal(0, 1, n_wells), n_months)
    lvl = np.repeat(rng.normal(0, 2, n_wells), n_months) if offset else 0.0
    zr = rng.normal(10, 3, n_wells * n_months)
    obs = zr + beta * c + lvl + rng.normal(0, noise, n_wells * n_months)
    xy = rng.uniform([170000, 2600000], [215000, 2650000], (n_wells, 2))
    blocks = dict(zip([f"w{i:03d}" for i in range(n_wells)], S.block_ids(xy[:, 0], xy[:, 1]),
                      strict=True))
    return wells, months, obs, zr, c, blocks


def test_constants_are_the_registered_ones():
    assert (S.B_BOOT, S.SEED, S.LEVEL, S.BLOCK_KM) == (2000, 20261101, 0.90, 5.0)
    assert (S.RATIO_BAND, S.INFORMATIVE_HEAD_M, S.MIN_CLUSTERS, S.MIN_MONTHS) == \
        (1.5, 0.05, 5, 6)
    assert S.FLAGS_SCORED == (0, 1)


@pytest.mark.parametrize("pid", [N2_ID, N2C_ID])
def test_constants_match_the_registry(pid):
    p = PREREG / f"{pid}.json"
    if not p.exists():
        pytest.skip("registry not present (gitignored results tree)")
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d.get("thresholds") == {"informative_head_m": 0.05, "level": 0.9,
                                   "min_clusters": 5, "min_months": 6, "ratio_band": 1.5}
    if pid == N2C_ID:
        t = d["tests"][0]
        assert t["seed"] == S.SEED
        interval = t["metric"]["interval"]
    else:
        interval = d["primary_metric"]["interval"]
    assert interval == "90 % from a 5 km block cluster bootstrap, B 2000, seed 20261101"
    rule = d["decision_rule"] if pid == N2_ID else d["tests"][0]["decision_rule"]
    assert rule["not_supported"] == "beta 90 % interval includes 0, or beta <= 0"
    assert "[1/1.5, 1.5]" in rule["response_supported"]
    assert "RMSE(PR) < RMSE(ZR)" in rule["response_supported"]


def test_block_ids_match_n1_and_e02c():
    rng = np.random.default_rng(3)
    x, y = rng.uniform(150000, 230000, 50), rng.uniform(2580000, 2680000, 50)
    want = np.array([f"{int(np.floor(a / 5000))}_{int(np.floor(b / 5000))}"
                     for a, b in zip(x, y, strict=True)])
    assert (S.block_ids(x, y) == want).all()
    e02c = REPO / "dataprep_staging/e02c/score_e02c.py"
    if e02c.exists():
        sys.path.insert(0, str(e02c.parent))
        import score_e02c as E  # noqa: E402

        assert (E.block_ids(x, y) == S.block_ids(x, y)).all()
    with pytest.raises(ValueError):
        S.block_ids([np.nan], [1.0])


def test_fe_slope_equals_dummy_variable_ols():
    w, m, obs, zr, c, _ = panel(n_wells=12, n_months=9, beta=0.8, seed=1)
    e = obs - zr
    uniq, inv = np.unique(w, return_inverse=True)
    X = np.column_stack([c, np.eye(len(uniq))[inv]])
    coef, *_ = np.linalg.lstsq(X, e, rcond=None)
    assert S.slope_fe(e, c, w) == pytest.approx(coef[0], rel=1e-10)
    t = np.array([int(s[:4]) * 12 + int(s[5:7]) for s in m], float)
    X2 = np.column_stack([c, np.eye(len(uniq))[inv], np.eye(len(uniq))[inv] * t[:, None]])
    coef2, *_ = np.linalg.lstsq(X2, e, rcond=None)
    assert S.slope_fe(e, c, w, t, detrend=True) == pytest.approx(coef2[0], rel=1e-8)


def test_bootstrap_draw_equals_explicit_resample_and_refit():
    w, _, obs, zr, c, blocks = panel(n_wells=20, n_months=6, seed=2)
    e = obs - zr
    uniq, sxy, sxx = S.well_sums(e, c, w)
    wb = np.array([blocks[u] for u in uniq])
    res = S.block_bootstrap(sxy, sxx, wb, B=50, seed=7)
    ub = np.unique(wb)
    draw = np.random.default_rng(7).integers(0, len(ub), size=(50, len(ub)))
    betas = []
    for d in draw:
        rows_e, rows_c, rows_w = [], [], []
        for k, bi in enumerate(d):
            for u in uniq[wb == ub[bi]]:
                sel = w == u
                rows_e.append(e[sel])
                rows_c.append(c[sel])
                rows_w.append(np.full(sel.sum(), f"{u}#{k}"))   # each copy its own FE
        betas.append(S.slope_fe(np.concatenate(rows_e), np.concatenate(rows_c),
                                np.concatenate(rows_w)))
    lo, hi = np.quantile(betas, [0.05, 0.95])
    assert res["lo"] == pytest.approx(lo, rel=1e-10) and res["hi"] == pytest.approx(hi,
                                                                                   rel=1e-10)
    assert res["n_blocks"] == len(ub) and res["n_draws_used"] == 50


@pytest.mark.parametrize("inf,beta,lo,hi,pr,zr,want", [
    (False, 1.0, 0.5, 1.5, 1.0, 2.0, "not_informative"),
    (True, 1.0, -0.1, 2.0, 1.0, 2.0, "not_supported"),
    (True, 1.0, 0.0, 2.0, 1.0, 2.0, "not_supported"),          # 0 on the bound: includes 0
    (True, -0.5, -0.9, -0.1, 1.0, 2.0, "not_supported"),
    (True, 1.0, 0.4, 1.6, 1.0, 2.0, "response_supported"),
    (True, 1.5, 0.9, 2.1, 1.0, 2.0, "response_supported"),     # band inclusive
    (True, 1 / 1.5, 0.2, 1.0, 1.0, 2.0, "response_supported"),
    (True, 1.51, 0.9, 2.1, 1.0, 2.0, "direction_supported_size_not"),
    (True, 0.6, 0.2, 1.0, 1.0, 2.0, "direction_supported_size_not"),
    (True, 1.0, 0.4, 1.6, 2.0, 2.0, "direction_supported_size_not"),   # RMSE tie
    (True, float("nan"), float("nan"), float("nan"), 1.0, 2.0, "not_supported"),
])
def test_decision_rule(inf, beta, lo, hi, pr, zr, want):
    assert S.decide(inf, beta, lo, hi, pr, zr) == want


def test_decision_rule_is_literal_and_flags_the_unnamed_case():
    # an interval wholly below 0 with beta > 0 matches neither registered not_supported clause
    assert S.decide(True, 1.0, -0.9, -0.1, 1.0, 2.0) == "response_supported"
    assert S.interval_below_zero_with_positive_beta(1.0, -0.9, -0.1)
    assert not S.interval_below_zero_with_positive_beta(1.0, 0.4, 1.6)
    assert not S.interval_below_zero_with_positive_beta(-0.5, -0.9, -0.1)
    assert not S.interval_below_zero_with_positive_beta(float("nan"), -0.9, -0.1)


def test_informative_gate():
    w = np.repeat(["a", "b", "c", "d", "e", "f"], 12)
    c = np.tile(np.linspace(-0.1, 0.1, 12), 6)
    blocks = {k: f"{i}_0" for i, k in enumerate("abcdef")}
    g = S.informative_gate(c, w, blocks, 12)
    assert g["passes"] and g["n_clusters"] == 6
    assert not S.informative_gate(c * 0.5, w, blocks, 12)["passes"]
    assert not S.informative_gate(c, w, blocks, 5)["passes"]
    few = {k: "0_0" if k in "abc" else "1_0" for k in "abcdef"}
    assert not S.informative_gate(c, w, few, 12)["passes"]
    lvl = np.repeat(np.arange(6.0), 12)                  # a pure level: demeaned away
    assert S.informative_gate(lvl, w, blocks, 12)["rms_well_demeaned_contrast_m"] == 0.0


def test_rmse_equal_well_and_pooled():
    r = np.array([1.0, 1.0, 1.0, 3.0])
    w = np.array(["a", "a", "a", "b"])
    out = S.rmse(r, w)
    assert out["equal_well"] == pytest.approx(np.sqrt((1 + 9) / 2))
    assert out["pooled"] == pytest.approx(np.sqrt(12 / 4))


def test_score_recovers_beta_and_ignores_levels():
    w, m, obs, zr, c, blocks = panel(n_wells=60, n_months=12, beta=0.9, noise=0.2, seed=4)
    flag = np.zeros(len(w), int)
    inf = S.informative_gate(c, w, blocks, 12)
    out = S.score(w, m, obs, zr, c, flag, blocks, inf, cl=zr)
    assert out["beta_response"]["beta"] == pytest.approx(0.9, abs=0.05)
    lo, hi = out["beta_response"]["ci90"]
    assert lo < 0.9 < hi
    assert out["sha256"] == S.file_sha256()
    assert out["verdict"] in S.VERDICTS
    assert out["detrended_secondary"]["detrended"] is True
    assert out["interval_below_zero_with_positive_beta"] is False


def test_score_row_filters_and_refusals():
    w, m, obs, zr, c, blocks = panel(n_wells=10, n_months=6, seed=5)
    flag = np.zeros(len(w), int)
    flag[:6] = 2                                          # one whole well flagged out
    obs = obs.copy()
    obs[6] = np.nan
    inf = S.informative_gate(c, w, blocks, 6)
    out = S.score(w, m, obs, zr, c, flag, blocks, inf)
    assert out["n_rows_scored"] == len(w) - 7
    assert out["beta_response"]["n_wells"] == 9
    with pytest.raises(ValueError, match="duplicate"):
        S.score(np.r_[w, w[:1]], np.r_[m, m[:1]], np.r_[obs, obs[:1]], np.r_[zr, zr[:1]],
                np.r_[c, c[:1]], np.r_[flag, flag[:1]], blocks, inf)
    with pytest.raises(ValueError, match="block"):
        S.score(w, m, obs, zr, c, flag, {}, inf)


def test_describe_and_no_home_paths():
    d = S.describe()
    assert d["sha256"] == S.file_sha256() and d["version"] == S.VERSION
    pat = re.compile(r"(/home/|-home-)[A-Za-z0-9_]")
    for p in (N2 / "bootstrap_scorer.py", Path(__file__)):
        assert not pat.search(p.read_text(encoding="utf-8")), p
