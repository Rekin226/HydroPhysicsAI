"""I1.CHX.v1.twocontrast (round-8 rank 8, R8.10): the two-contrast exploratory narrowing.

Everything here is SYNTHETIC: panels, observations, the parent's report and both
registrations live in pytest's tmp dirs; nothing is registered in the real registry, no look
is logged there and no real observation is read.

- with one regressor the joint code is CHX's estimator (beta and bootstrap interval);
- the joint regression recovers known coefficients, and the Changhua coefficient does not
  depend on the offset (B or B + C - A);
- the model-only collinearity rule fixes 'not separable' without a fit and without a head;
- the reading map, the pre-gate, the declared panel keys;
- the scorer refuses before registration, before the look, a foreign panel and a second run,
  never writes the parent's report.json, and logs every attempt;
- the draft declaration's key fields and validate_prereg.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pandas")             # chx_io imports it

REPO = Path(__file__).resolve().parents[1]
D = REPO / "dataprep_staging" / "chx2c"
PATHS = [str(D), str(REPO / "dataprep_staging" / "ch_x"),
         str(REPO / "dataprep_staging" / "n2_policy"), str(REPO / "dataprep_staging")]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_saved = list(sys.path)
sys.path[:0] = PATHS
C = _load("chx2c_lib", D / "chx2c_lib.py")
S = _load("score_chx2c", D / "score_chx2c.py")
PR = _load("prereg_chx2c", D / "prereg_chx2c.py")
import chx_io as IO  # noqa: E402
import chx_lib as L  # noqa: E402
import score_chx as SC  # noqa: E402

sys.path[:] = _saved

HAVE_RESULTS = (REPO / "results" / "dataprep" / "preregistration" / "INDEX.jsonl").is_file()
NEED_RESULTS = pytest.mark.skipif(not HAVE_RESULTS, reason="results/ (gitignored) is absent")
WIN = L.month_list(*C.K.CHX_WINDOW)


_FE_BASIS = L.fe_basis
_FE_CACHE: dict = {}


def _fe_basis_memo(wells, months, form):
    """chx_lib.fe_basis, memoised on its exact inputs (same numbers; its SVD takes seconds
    and every synthetic world here shares the same two row sets)."""
    w = np.asarray(wells).astype(str)
    m = np.asarray([str(x)[:7] for x in months])
    key = (form, w.tobytes(), m.tobytes())
    if key not in _FE_CACHE:
        _FE_CACHE[key] = _FE_BASIS(w, m, form)
    return _FE_CACHE[key]


@pytest.fixture(autouse=True)
def _staging_on_path(monkeypatch):
    saved = list(sys.path)
    sys.path[:0] = PATHS
    monkeypatch.setattr(L, "fe_basis", _fe_basis_memo)
    yield
    sys.path[:] = saved


def _no_heads(*a, **k):
    raise AssertionError("a head was read")


# ----------------------------------------------------------------------------- synthetic
def synth(seed: int = 1, k_ch: float = 0.4, k_yl1: float = 0.6, noise: float = 0.05,
          collinear: bool = False) -> tuple[dict, np.ndarray, np.ndarray]:
    """A chx_panel-shaped dict (PANEL_KEYS + coordinates) and synthetic obs with
    obs - B = k_ch c_CH + k_yl1 c_YL + noise."""
    rng = np.random.default_rng(seed)
    W, T = C.K.N_EXT_CHANGHUA, len(WIN)
    sid = np.array([f"07{i:06d}" for i in range(W - 4)] + list(C.K.NEST))
    x = 180000.0 + 6000 * (np.arange(W) % 7)
    y = 2650000.0 + 6000 * (np.arange(W) // 7)
    bl = L.well_blocks(sid, x, y)
    a = 0.4 * np.sin(2 * np.pi * np.arange(T) / 12) + 0.2 * rng.standard_normal(T)
    b = 0.3 * np.cos(2 * np.pi * np.arange(T) / 12) + 0.2 * rng.standard_normal(T)
    c = -rng.uniform(0.5, 3.0, W)[:, None] * a[None, :]
    if collinear:
        cp = 2.0 * c + 0.01 * rng.standard_normal((W, T))
    else:
        cp = rng.uniform(-1.0, 1.0, W)[:, None] * b[None, :]
    B = 10.0 + rng.normal(0, 1, (W, 1)) + np.zeros((W, T))
    A = B + c
    Cm = A - cp
    P = {"months": np.array(WIN), "sid_ch": sid, "x_ch": x, "y_ch": y,
         "block_ch": np.array([bl[s] for s in sid]), "A_ch": A, "B_ch": B, "C_ch": Cm,
         "c_ch": A - B, "c_placebo_ch": A - Cm}
    obs = B + k_ch * P["c_ch"] + k_yl1 * P["c_placebo_ch"] + noise * rng.standard_normal((W, T))
    obs[:5, :30] = np.nan
    obs[7, :110] = np.nan                         # one screen below the 12-month minimum
    flag = np.where(np.isfinite(obs), 0, 9)
    return P, obs, flag


def _blocks(P):
    return dict(zip(P["sid_ch"].astype(str), P["block_ch"].astype(str), strict=True))


# ----------------------------------------------------------------------------- estimator
def test_one_regressor_is_chx_estimator():
    P, obs, flag = synth()
    sid, bl = list(P["sid_ch"]), _blocks(P)
    rows_chx = L.eligible(L.long_panel(sid, WIN, obs, P["B_ch"], P["c_ch"], flag))
    inf = L.informative_from_contrast(P["c_ch"], sid, bl, WIN)
    ref = L.score_direction(rows_chx, bl, inf)
    own = C.joint_fit(C.joint_rows(sid, WIN, obs, flag, P["B_ch"], [P["c_ch"]]), bl)
    assert own["n_rows"] == ref["n_rows"] and own["n_wells"] == ref["n_wells"] == 22
    assert abs(own["beta"][0] - ref["beta_primary"]["beta"]) <= 1e-12
    for j in range(2):
        assert abs(own["ci90"][0][j] - ref["beta_primary"]["ci90"][j]) <= 1e-12
    assert own["bootstrap"]["n_draws_used"] == ref["beta_primary"]["bootstrap"]["n_draws_used"]
    # and the frozen scorer's own call path (score_chx.score_panel) gives the same number
    fz = C.frozen_numbers(SC.score_panel(P, obs, flag))
    assert abs(fz["primary"]["beta"] - own["beta"][0]) <= 1e-12


def test_joint_recovers_known_coefficients_noise_free():
    P, obs, flag = synth(k_ch=0.5, k_yl1=0.3, noise=0.0)
    rows = C.joint_rows(list(P["sid_ch"]), WIN, obs, flag, P["B_ch"],
                        [P["c_ch"], P["c_placebo_ch"]])
    fit = C.joint_fit(rows, _blocks(P))
    assert np.allclose(fit["beta"], [0.5, 0.3], atol=1e-9)
    for (lo, hi), b in zip(fit["ci90"], (0.5, 0.3), strict=True):
        assert abs(lo - b) <= 1e-8 and abs(hi - b) <= 1e-8
    assert fit["bootstrap"]["n_draws_singular"] == 0


def test_changhua_coefficient_does_not_depend_on_the_offset():
    P, obs, flag = synth(seed=4)
    sid, bl = list(P["sid_ch"]), _blocks(P)
    X = [P["c_ch"], P["c_placebo_ch"]]
    fb = C.joint_fit(C.joint_rows(sid, WIN, obs, flag, P["B_ch"], X), bl)
    D_ = P["B_ch"] + P["C_ch"] - P["A_ch"]                    # both counties at 2012
    fd = C.joint_fit(C.joint_rows(sid, WIN, obs, flag, D_, X), bl)
    assert abs(fb["beta"][0] - fd["beta"][0]) <= 1e-9
    assert np.allclose(fb["ci90"][0], fd["ci90"][0], atol=1e-9)
    assert abs(fd["beta"][1] - (fb["beta"][1] + 1.0)) <= 1e-9   # k_YL = b_YL + 1


def test_omitting_the_yunlin_contrast_biases_the_changhua_slope():
    P, obs, flag = synth(seed=5, k_ch=0.0, k_yl1=1.5, noise=0.0)
    P["c_placebo_ch"] = P["c_placebo_ch"] + 0.8 * P["c_ch"]          # correlated contrasts
    P["C_ch"] = P["A_ch"] - P["c_placebo_ch"]
    obs = P["B_ch"] + 1.5 * P["c_placebo_ch"]
    obs[:5, :30] = np.nan
    flag = np.where(np.isfinite(obs), 0, 9)
    sid, bl = list(P["sid_ch"]), _blocks(P)
    one = C.joint_fit(C.joint_rows(sid, WIN, obs, flag, P["B_ch"], [P["c_ch"]]), bl)
    two = C.joint_fit(C.joint_rows(sid, WIN, obs, flag, P["B_ch"],
                                   [P["c_ch"], P["c_placebo_ch"]]), bl)
    assert one["beta"][0] > 0.5                       # a spurious 'direction' alone
    assert abs(two["beta"][0]) <= 1e-9                # gone with the control
    assert abs(two["beta"][1] - 1.5) <= 1e-9


# ----------------------------------------------------------------------------- collinearity
def test_collinearity_is_model_only_and_fixes_the_reading():
    P, _, _ = synth(collinear=True)
    col = C.collinearity_model_only(P["c_ch"], P["c_placebo_ch"], list(P["sid_ch"]), WIN)
    assert col["abs_r"] > 0.99 and not col["separable"] and col["grid"] == [23, 120]
    r = C.reading(None, col, None)
    assert r["key"] == "not_separable" and "without fitting" in r["why"][0]
    P2, _, _ = synth()
    col2 = C.collinearity_model_only(P2["c_ch"], P2["c_placebo_ch"], list(P2["sid_ch"]), WIN)
    assert col2["separable"] and col2["abs_r"] < 0.9
    neg = C.collinearity_model_only(P["c_ch"], -P["c_placebo_ch"], list(P["sid_ch"]), WIN)
    assert neg["r"] < -0.99 and not neg["separable"]            # |r|: sign does not help


def test_collinearity_statistic_equals_the_direct_projection():
    P, _, _ = synth(seed=8)
    sid = list(P["sid_ch"])
    rows = {"well": np.repeat(np.asarray(sid), len(WIN)),
            "month": np.tile(np.asarray(WIN), len(sid))}
    Q = L.fe_basis(rows["well"], rows["month"], "twoway")
    u, v = L.fe_resid(P["c_ch"].ravel(), Q), L.fe_resid(P["c_placebo_ch"].ravel(), Q)
    col = C.collinearity_model_only(P["c_ch"], P["c_placebo_ch"], sid, WIN)
    assert abs(col["r"] - float(np.corrcoef(u, v)[0, 1])) <= 1e-9


# ----------------------------------------------------------------------------- readings
def _fit(b, lo, hi, singular=0, r_rows=0.2):
    return {"beta": [b, 0.0], "ci90": [[lo, hi], [0.0, 0.0]], "scored_rows_r": r_rows,
            "bootstrap": {"B": 2000, "n_draws_singular": singular}}


def test_reading_map_is_one_sided():
    ok, pg = {"separable": True, "abs_r": 0.3}, {"passed": True}
    assert C.reading(_fit(0.2, 0.01, 0.4), ok, pg)["key"] == "survives"
    assert C.reading(_fit(0.2, 0.0, 0.4), ok, pg)["key"] == "not_separable"     # reaches 0
    assert C.reading(_fit(0.2, -0.1, 0.4), ok, pg)["key"] == "not_separable"
    assert C.reading(_fit(-0.2, -0.4, -0.1), ok, pg)["key"] == "not_separable"
    assert C.reading(_fit(0.2, 0.01, 0.4, singular=101), ok, pg)["key"] == "not_separable"
    assert C.reading(_fit(0.2, 0.01, 0.4, r_rows=0.95), ok, pg)["key"] == "not_separable"
    assert C.reading(_fit(np.nan, 0.01, 0.4), ok, pg)["key"] == "not_separable"
    ne = C.reading(None, ok, {"passed": False})
    assert ne["key"] == "not_evaluable" and ne["e25_note"] == C.READING_NOT_EVALUABLE
    sv = C.reading(_fit(0.2, 0.01, 0.4), ok, pg)
    assert sv["e25_note"] == ("direction survives a Yunlin spill-over control "
                              "(after-the-fact, exploratory)")
    assert C.READING_NOT_SEPARABLE == "direction not separable from Yunlin spill-over"
    for r in (sv, ne):
        assert r["e25_status"].startswith("unchanged") and "cannot promote" in r["cannot_promote"]


def test_pregate_reproduces_and_detects_a_mismatch():
    P, obs, flag = synth(seed=2)
    out = SC.score_panel(P, obs, flag)
    fz = C.frozen_numbers(out)
    own = C.joint_fit(C.joint_rows(list(P["sid_ch"]), WIN, obs, flag, P["B_ch"], [P["c_ch"]]),
                      _blocks(P))
    pub = C.published_numbers(json.loads(json.dumps(out, default=float)))
    assert C.pregate(pub, fz, own)["passed"]
    bad = json.loads(json.dumps(pub))
    bad["primary"]["beta"] += 2e-6
    g = C.pregate(bad, fz, own)
    assert not g["passed"] and not g["checks"]["own_beta_vs_published"]["ok"]
    bad2 = json.loads(json.dumps(pub))
    bad2["placebo"]["ci90"][1] += 1e-3
    assert not C.pregate(bad2, fz, own)["passed"]
    bad3 = json.loads(json.dumps(pub))
    bad3["primary"]["n_rows"] += 1
    assert not C.pregate(bad3, fz, own)["checks"]["n_rows"]["ok"]


def test_declared_panel_keys_are_model_or_metadata():
    hindcast = (REPO / "dataprep_staging" / "ch_x" / "hindcast.py").read_text()
    assert "no observation read" in hindcast
    assert set(C.PANEL_KEYS) == {"months", "sid_ch", "block_ch", "A_ch", "B_ch", "C_ch",
                                 "c_ch", "c_placebo_ch"}
    assert not any("obs" in k for k in C.PANEL_KEYS)
    assert C.REPORT != C.PARENT_REPORT and Path(C.REPORT).name == "twocontrast_report.json"
    src = (D / "score_chx2c.py").read_text()
    assert 'open("x"' in src and "write_prereg(" not in src and "SC.run_score" not in src
    for p in D.glob("*.py"):
        assert "write_prereg(" not in p.read_text(), p


# ----------------------------------------------------------------------------- end to end
def _fake_parent(pdir, rows, sids):
    from harness import prereg3

    doc = {"id": C.PARENT_ID, "wp": C.WP, "kind": "declaration",
           "title": "synthetic stand-in for I1.CHX.v1", "tests": {"not_applicable": "x"},
           "zones": {"not_applicable": "x"}, "priors": "synthetic",
           "burn_in": {"not_applicable": "x"}, "folds": {"not_applicable": "x"},
           "fresh_units": {"not_applicable": "x"}, "thresholds": {"not_applicable": "x"},
           "gate_scores": {"not_applicable": "x"}, "primary_model": "synthetic",
           "inputs": {"not_applicable": "x"}, "interpretation": "synthetic",
           "attempt_policy": {"exploratory_followups_max": 1,
                              "pre_declared_alternatives": []},
           "screens": {"ids": sorted(sids)},
           "external_units": {C.UNIT: {"ids": sorted(sids), "window": "2013-01..2022-12",
                                       "description": "synthetic 23 screens"}}}
    prereg3.write_prereg(doc, prereg_dir=pdir, rows_dir=rows, units_registry=None,
                         registries=[])


def _world(tmp_path, monkeypatch, collinear=False, seed=3):
    """A tmp repo with a synthetic panel, parent registration and parent report."""
    monkeypatch.setattr(S, "REPO", tmp_path)
    P, obs, flag = synth(seed=seed, collinear=collinear)
    pz = tmp_path / C.PANEL
    pz.parent.mkdir(parents=True)
    np.savez(pz, **P)
    (tmp_path / C.PANEL_SUMMARY).write_text(json.dumps({"panel_sha256": IO.sha256(pz)}))
    pdir, rows = tmp_path / "prereg", tmp_path / "rows"
    _fake_parent(pdir, rows, list(P["sid_ch"].astype(str)))
    rep = SC.score_panel(P, obs, flag)
    rep.update({"prereg_id": C.PARENT_ID, "direction": True, "panel_sha256": IO.sha256(pz)})
    rp = tmp_path / C.PARENT_REPORT
    rp.parent.mkdir(parents=True)
    rp.write_text(json.dumps(rep, default=float))

    def loader(sids, months):
        return obs, flag, ["synthetic"]

    return pdir, rows, loader


def _register(tmp_path, pdir, rows, **change):
    from harness import prereg3

    d = PR.draft_prereg()
    d["inputs"] = [str(tmp_path / C.PARENT_REPORT), str(tmp_path / C.PANEL)]
    d.update(change)
    return prereg3.write_prereg(d, prereg_dir=pdir, rows_dir=rows, units_registry=None,
                                registries=[])


def _attempts(tmp_path):
    p = tmp_path / C.ATTEMPTS
    return [json.loads(s) for s in p.read_text().splitlines()] if p.is_file() else []


def _events(tmp_path):
    return [a["event"] for a in _attempts(tmp_path)]


def _looks(pdir):
    p = pdir / "looks.jsonl"
    return [json.loads(s) for s in p.read_text().splitlines()] if p.is_file() else []


def test_scorer_refuses_before_registration_and_before_the_look(tmp_path, monkeypatch):
    pdir, rows, loader = _world(tmp_path, monkeypatch)
    with pytest.raises(S.Refused, match=r"step 1"):
        S.run_score(pdir, rows, heads_loader=_no_heads)
    with pytest.raises(S.Refused, match=r"step 1"):
        S.record_look(pdir, rows)
    _register(tmp_path, pdir, rows)
    with pytest.raises(S.Refused, match=r"step 4"):
        S.run_score(pdir, rows, heads_loader=_no_heads)
    assert _events(tmp_path) == ["start", "refused"] * 3
    assert all(a.get("heads_read") is False for a in _attempts(tmp_path) if "heads_read" in a)
    assert not (tmp_path / C.REPORT).exists() and _looks(pdir) == []


def test_a_look_of_the_wrong_kind_does_not_unlock(tmp_path, monkeypatch):
    from harness import prereg3

    pdir, rows, _ = _world(tmp_path, monkeypatch)
    _register(tmp_path, pdir, rows)
    prereg3.record_look([C.UNIT], C.SCOPE, C.WP, "design_time", prereg_id=C.ID,
                        prereg_dir=pdir, rows_dir=rows)
    prereg3.record_look([C.UNIT], "heads:2013-01..2020-12", C.WP, C.LOOK_PURPOSE,
                        prereg_id=C.ID, prereg_dir=pdir, rows_dir=rows)
    with pytest.raises(S.Refused, match=r"step 4"):
        S.run_score(pdir, rows, heads_loader=_no_heads)


def test_end_to_end_scores_once_into_a_new_report(tmp_path, monkeypatch):
    pdir, rows, loader = _world(tmp_path, monkeypatch)
    _, h = _register(tmp_path, pdir, rows)
    out = S.record_look(pdir, rows)
    lk = out["look"]
    assert not out["collinear_stop"] and not (tmp_path / C.REPORT).exists()
    assert lk["purpose"] == "after_the_fact" and lk["prereg_id"] == C.ID
    assert lk["note"] == C.LOOK_NOTE and "read once on a successful" in lk["note"]
    with pytest.raises(S.Refused, match="already logged"):
        S.record_look(pdir, rows)
    parent_before = (tmp_path / C.PARENT_REPORT).read_bytes()
    rep = S.run_score(pdir, rows, heads_loader=loader)
    assert rep["prereg_sha256"] == h and rep["heads_read"] and rep["pregate"]["passed"]
    assert rep["collinearity_model_only"]["separable"]
    assert rep["followup_limit"]["limit"] == 1 and rep["followup_limit"]["counted"] == [C.ID]
    assert rep["reading"]["key"] == "survives"            # synthetic k_ch 0.4, low noise
    assert abs(rep["joint"]["changhua"]["beta"] - 0.4) < 0.05
    assert abs(rep["joint"]["yunlin"]["report_only_k_yl"]["beta"] - 1.6) < 0.1
    assert (tmp_path / C.PARENT_REPORT).read_bytes() == parent_before
    written = json.loads((tmp_path / C.REPORT).read_text())
    assert written["reading"]["e25_status"].startswith("unchanged")
    assert _events(tmp_path) == ["start", "look_logged", "start", "refused", "start",
                                 "scored"]
    assert "beta" not in json.dumps(_attempts(tmp_path))       # no statistic in the log
    with pytest.raises(S.Refused, match="scored once"):
        S.run_score(pdir, rows, heads_loader=_no_heads)


def test_collinear_panel_stops_in_look_without_a_look_or_a_head(tmp_path, monkeypatch):
    pdir, rows, _ = _world(tmp_path, monkeypatch, collinear=True)
    _register(tmp_path, pdir, rows)
    out = S.record_look(pdir, rows)
    rep = out["report"]
    assert out["collinear_stop"] and rep["heads_read"] is False and rep["joint"] is None
    assert rep["reading"]["key"] == "not_separable" and rep["look_logged"] is False
    assert _looks(pdir) == []                                  # NO look was logged
    with pytest.raises(S.Refused, match="scored once"):
        S.run_score(pdir, rows, heads_loader=_no_heads)
    assert _events(tmp_path) == ["start", "collinear_stop", "start", "refused"]


def test_failed_pregate_computes_no_joint_coefficient(tmp_path, monkeypatch):
    pdir, rows, loader = _world(tmp_path, monkeypatch)
    rp = tmp_path / C.PARENT_REPORT
    r = json.loads(rp.read_text())
    r["primary"]["beta_primary"]["beta"] += 1e-4
    rp.write_text(json.dumps(r))
    _register(tmp_path, pdir, rows)
    S.record_look(pdir, rows)
    rep = S.run_score(pdir, rows, heads_loader=loader)
    assert not rep["pregate"]["passed"] and rep["joint"] is None
    assert rep["reading"]["key"] == "not_evaluable"
    assert _events(tmp_path) == ["start", "look_logged", "start", "pregate_failed"]


def test_scorer_refuses_a_foreign_panel_and_a_changed_input(tmp_path, monkeypatch):
    pdir, rows, loader = _world(tmp_path, monkeypatch)
    _register(tmp_path, pdir, rows)
    S.record_look(pdir, rows)
    other, _, _ = synth(seed=11)
    alt = tmp_path / "other.npz"
    np.savez(alt, **other)
    with pytest.raises(S.Refused, match=r"step 2.*panel sha256"):
        S.run_score(pdir, rows, panel=alt, heads_loader=_no_heads)
    np.savez(tmp_path / C.PANEL, **other)                          # registered input changed
    with pytest.raises(S.Refused, match="inputs changed"):
        S.run_score(pdir, rows, heads_loader=_no_heads)


def test_refuses_changed_frozen_code(tmp_path, monkeypatch):
    pdir, rows, _ = _world(tmp_path, monkeypatch)
    _register(tmp_path, pdir, rows)
    real = S._code_sha
    monkeypatch.setattr(S, "_code_sha",
                        lambda p: "0" * 64 if p.endswith("chx_lib.py") else real(p))
    with pytest.raises(S.Refused, match="frozen code differs"):
        S.record_look(pdir, rows)
    with pytest.raises(S.Refused, match="frozen code differs"):
        S.run_score(pdir, rows, heads_loader=_no_heads)


def test_refuses_wrong_followup_of_and_wrong_kind(tmp_path, monkeypatch):
    from harness import prereg3

    pdir, rows, _ = _world(tmp_path, monkeypatch)
    _register(tmp_path, pdir, rows, followup_of="I1.CH_backcast.v1")
    with pytest.raises(S.Refused, match="followup_of"):
        S.record_look(pdir, rows)
    real = prereg3.load_prereg

    def as_test(pid, **kw):
        doc, sha = real(pid, **kw)
        return (doc | {"kind": "test"} if pid == C.ID else doc), sha

    monkeypatch.setattr(prereg3, "load_prereg", as_test)
    with pytest.raises(S.Refused, match="not a declaration"):
        S.run_score(pdir, rows, heads_loader=_no_heads)


def test_refuses_a_missing_parent_report(tmp_path, monkeypatch):
    pdir, rows, _ = _world(tmp_path, monkeypatch)
    _register(tmp_path, pdir, rows)
    (tmp_path / C.PARENT_REPORT).unlink()
    with pytest.raises(S.Refused, match=r"step 0.*missing"):
        S.record_look(pdir, rows)
    with pytest.raises(S.Refused, match=r"step 0.*missing"):
        S.run_score(pdir, rows, heads_loader=_no_heads)


def test_panel_identity_and_shape_checks(tmp_path):
    P, _, _ = synth()
    pz = tmp_path / "p.npz"
    np.savez(pz, **P)
    parent_doc = {"screens": {"ids": sorted(P["sid_ch"].astype(str))}}
    good = {"panel_sha256": IO.sha256(pz)}
    Q = {k: P[k] for k in C.PANEL_KEYS}
    assert S.check_panel(Q, pz, parent_doc, good) == []
    assert any("sha256" in x for x in S.check_panel(Q, pz, parent_doc, {"panel_sha256": "x"}))
    bad_ids = {"screens": {"ids": sorted(P["sid_ch"].astype(str))[:-1] + ["07999999"]}}
    assert any("23 ids" in x for x in S.check_panel(Q, pz, bad_ids, good))
    assert any("window" in x for x in S.check_panel(Q | {"months": Q["months"][:-1]}, pz,
                                                    parent_doc, good))
    assert any("finite" in x for x in S.check_panel(Q | {"c_ch": Q["c_ch"][:, :-1]}, pz,
                                                    parent_doc, good))
    nan = Q["c_placebo_ch"].copy()
    nan[0, 0] = np.nan
    assert any("finite" in x for x in S.check_panel(Q | {"c_placebo_ch": nan}, pz,
                                                    parent_doc, good))
    assert any("A_ch - B_ch" in x for x in S.check_panel(Q | {"c_ch": Q["c_ch"] + 1e-9}, pz,
                                                         parent_doc, good))
    assert any("block" in x for x in S.check_panel(Q | {"block_ch": Q["block_ch"][:-1]}, pz,
                                                   parent_doc, good))
    with pytest.raises(S.Refused, match="lacks"):
        np.savez(tmp_path / "q.npz", **{k: v for k, v in P.items() if k != "C_ch"})
        S.load_panel(tmp_path / "q.npz")


def test_followup_limit_reads_the_parent_and_ignores_names(tmp_path, monkeypatch):
    from harness import prereg3

    pdir, rows, _ = _world(tmp_path, monkeypatch)
    _register(tmp_path, pdir, rows)
    S.record_look(pdir, rows)
    d = PR.draft_prereg()
    d.update({"id": "CH_R.river_arm.v1", "inputs": {"not_applicable": "x"}})   # any name
    prereg3.write_prereg(d, prereg_dir=pdir, rows_dir=rows, units_registry=None,
                         registries=[])
    assert S.followups_of_parent(pdir) == sorted([C.ID, "CH_R.river_arm.v1"])
    with pytest.raises(S.Refused, match="exploratory_followups_max"):
        S.run_score(pdir, rows, heads_loader=_no_heads)
    parent = {"attempt_policy": {"exploratory_followups_max": 1}}
    own = {"attempt_policy": {"repair_ids": ["CH_R.river_arm.v1"]}}
    assert S.check_followup_limit(own, parent, pdir)["counted"] == [C.ID]
    two = {"attempt_policy": {"exploratory_followups_max": 2}}
    assert S.check_followup_limit({}, two, pdir)["limit"] == 2


def test_nan_r_fails_toward_not_separable_and_json_has_no_nan():
    P, _, _ = synth()
    z = np.zeros_like(P["c_ch"])
    col = C.collinearity_model_only(P["c_ch"], z, list(P["sid_ch"]), WIN)
    assert not col["finite"] and not col["separable"]
    r = C.reading(None, col, None)
    assert r["key"] == "not_separable" and "not finite" in r["why"][0]
    txt = json.dumps(S.jsonable({"col": col, "x": [np.float64("inf"), np.nan, np.bool_(1)]}),
                     allow_nan=False)
    assert "NaN" not in txt and '"x": [null, null, true]' in txt


# ----------------------------------------------------------------------------- the draft
def test_draft_key_fields_and_validation(tmp_path):
    from harness import prereg3

    d = PR.draft_prereg()
    assert d["id"] == "I1.CHX.v1.twocontrast" and d["kind"] == "declaration"
    assert d["followup_of"] == "I1.CHX.v1" and d["ruling"]["id"] == "R8.10"
    assert "cannot promote" in d["label"] and "cannot promote" in d["cannot_promote"]
    assert "not_applicable" in d["tests"] and "not_applicable" in d["fresh_units"]
    assert d["attempt_policy"]["exploratory_followups_max"] == 0
    assert d["look"]["purpose"] == "after_the_fact" and d["look"]["units"] == [C.UNIT]
    assert d["gates"]["collinearity"]["rule"].startswith("|r| > 0.9")
    assert d["readings"]["survives"]["e25_note"] == C.READING_SURVIVES
    assert d["outputs"]["never_written"].endswith("I1.CHX.v1/report.json")
    assert set(d["frozen_code"]) == {*C.OWN_FROZEN, *C.PARENT_FROZEN}
    assert "created" not in d
    assert prereg3.validate_prereg(d, units_registry=None, prereg_dir=tmp_path / "p",
                                   registries=[]) == []


@NEED_RESULTS
def test_draft_against_the_real_registry_reads_nothing_new():
    from harness import prereg3

    d = PR.draft_prereg()
    assert prereg3.validate_prereg(d) == []
    assert all(h != PR.MISSING for h in d["frozen_code"].values())
    assert {p: d["frozen_code"][p] for p in C.PARENT_FROZEN} == C.PARENT_FROZEN
    assert all(isinstance(x, dict) for x in d["inputs"])
    left = PR.readiness(d)
    idx = {r["id"]: r for r in prereg3.read_chained(prereg3.PREREG_DIR / "INDEX.jsonl")}
    if C.ID not in idx:                         # before registration: only git remains
        assert all("not committed" in s for s in left), left
        return
    # registered (2026-11-03) and scored once: the registered file is the INDEX entry, it holds
    # exactly what this draft builder produces (the harness adds only its own bookkeeping
    # fields and resolves the inputs), and readiness names exactly the registered state
    reg_path = prereg3.PREREG_DIR / f"{C.ID}.json"
    assert hashlib.sha256(reg_path.read_bytes()).hexdigest() == idx[C.ID]["sha256"]
    reg = json.loads(reg_path.read_text(encoding="utf-8"))
    for k, v in d.items():
        if k == "inputs":
            assert [x["sha256"] for x in reg[k]] == [x["sha256"] for x in v]
        elif k == "disclosures":                # this test file is disclosed, not frozen
            t = reg[k]["tests"]
            assert t["binding"] == "disclosure only, not frozen" and re.fullmatch(
                r"[0-9a-f]{64}", t["sha256_at_declaration"])
            assert {a: b for a, b in reg[k].items() if a != "tests"} == {
                a: b for a, b in v.items() if a != "tests"}
        else:
            assert reg[k] == v, k
    assert sorted(left) == sorted([f"{C.REPORT} exists before the declaration",
                                   f"{C.ATTEMPTS} exists before the declaration",
                                   f"{C.ID} is already registered"]), left
