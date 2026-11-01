"""CPU tests of the E02c and external-well registration scripts and their scorer.

Synthetic observations only: no head product is opened, nothing is written to
results/dataprep/preregistration (every registration and look goes to pytest's tmp dirs), no
look is logged on the real registry. The frozen E02 predictions (registered model output) are
read for station ids and as the forecast the synthetic observations are drawn around.

    cd <repository root>
    ~/miniforge3/envs/hydro/bin/python -m pytest -q -p no:cacheprovider dataprep_staging/e02c/tests
"""
from __future__ import annotations

import datetime as dt
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

TESTS = Path(__file__).resolve().parent
X = TESTS.parent
REPO = X.parents[1]
for _p in (str(X), str(REPO / "dataprep_staging"), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import export_baselines as EB  # noqa: E402
import prereg_e02c as P  # noqa: E402
import prereg_e02c_ext as PX  # noqa: E402
import score_e02c as S  # noqa: E402
from harness import prereg3  # noqa: E402

E02_PRED = REPO / S.E02_DIR / "predictions.csv"
REAL_INDEX = REPO / "results/dataprep/preregistration/INDEX.jsonl"
needs_repo = pytest.mark.skipif(not (E02_PRED.exists() and REAL_INDEX.exists()),
                                reason="gitignored results/ tree not present")
UTC = dt.timezone.utc
BEFORE_DEADLINE = dt.datetime(2026, 10, 30, 12, 0, tzinfo=UTC)
SCORING_DAY = dt.datetime(2028, 1, 20, 12, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _no_real_products(monkeypatch):
    """No test may open a heads product (pre- or post-origin): the guard and the export
    loader raise inside every test."""
    from common import guard

    def refuse(*a, **k):
        raise AssertionError("tests must not open a data product")

    monkeypatch.setattr(guard, "open_product", refuse)
    monkeypatch.setattr(EB, "_open_l1", refuse)


def _fair_stub(wells, frame):
    """Synthetic arrays for the report-only fair rule (the real one reads L1 pre-origin)."""
    rng = np.random.default_rng(9)
    W, M = 20, 12
    truth = rng.normal(size=(W, S.T_FIT + 46 + M))
    return {"pred_full": truth + 1.0, "obs_fit": truth[:, :S.T_FIT],
            "obs_window": truth[:, -M:], "gap": 46}


def _pred() -> pd.DataFrame:
    return pd.read_csv(E02_PRED, dtype={"station_id": str})


def synthetic_obs(pred: pd.DataFrame, seed=1, scale=1.0, shift=0.0) -> pd.DataFrame:
    """obs = predicted + N(0, sigma) with sigma the interval's own: coverage ~0.90."""
    rng = np.random.default_rng(seed)
    sig = (pred["upper_m"] - pred["lower_m"]) / (2 * 1.6448536269514722)
    return pd.DataFrame({"station_id": pred["station_id"], "date": pred["date"],
                         "observed_m": pred["predicted_m"] + shift
                         + scale * sig * rng.standard_normal(len(pred)),
                         "synthetic": True})


def synthetic_baselines(pred: pd.DataFrame, seed=2, off=3.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(pred)
    return pd.DataFrame({"station_id": pred["station_id"], "date": pred["date"],
                         "cl_m": pred["predicted_m"] + off + rng.standard_normal(n),
                         "clt_m": pred["predicted_m"] - off + rng.standard_normal(n),
                         "n_fit_months": 100})


def synthetic_wells(pred: pd.DataFrame) -> pd.DataFrame:
    ids = sorted(pred["station_id"].unique())
    rng = np.random.default_rng(3)
    x = 170000 + rng.uniform(0, 50000, len(ids))
    y = 2610000 + rng.uniform(0, 60000, len(ids))
    return pd.DataFrame({"station_id": ids, "county": [S.county_of(s) for s in ids],
                         "x": x, "y": y, "layer_code": "2", "layer": 1, "cell": 5,
                         "block5": S.block_ids(x, y), "cell_matches_e02": True})


def write_verify(d: Path, csvs) -> None:
    """A verify.json as export_baselines.verify_export writes it (identical rebuild)."""
    (d / "verify.json").write_text(json.dumps({
        "identical": True, "files": {n: S.sha256(d / n) for n in csvs}, "problems": [],
        "summary_sha256": S.sha256(d / "summary.json"), "verified": "synthetic"}))


def make_synth_dir(tmp: Path, pred: pd.DataFrame, **kw) -> Path:
    d = tmp / "synth"
    d.mkdir()
    (d / S.SYNTHETIC_MARKER).write_text("synthetic fixture\n")
    synthetic_obs(pred, **kw).to_csv(d / "observations.csv", index=False)
    synthetic_baselines(pred).to_csv(d / "baselines_e02c.csv", index=False)
    synthetic_wells(pred).to_csv(d / "wells_158.csv", index=False)
    csvs = ("baselines_e02c.csv", "wells_158.csv")
    s = {"files": {n: S.sha256(d / n) for n in csvs},
         "reference": {"chosen": EB.REF_L1}, "cell_check": None}
    (d / "summary.json").write_text(json.dumps(s))
    write_verify(d, csvs)
    return d


def make_product(tmp: Path, grant: bool = True) -> Path:
    """A window-product directory whose MANIFEST (metadata only, no table) does or does not
    grant prospective_scorer; ``new_wells`` lies inside it."""
    d = tmp / "window_product"
    (d / "new_wells").mkdir(parents=True)
    uses = ["prospective_scorer"] if grant else ["after_the_fact_analysis"]
    (d / "MANIFEST.json").write_text(json.dumps({
        "product": "heads_window_synthetic", "version": "v0", "allowed_uses": uses,
        "forbidden_uses": [], "fit_window_end": "2027-10", "gate": {"passed": True}}))
    return d


def make_e02_step5(tmp: Path, pred: pd.DataFrame, obs: pd.DataFrame) -> Path:
    """E02's step-(5) output, as twin.prospective.score writes it, from synthetic heads."""
    man = json.loads((REPO / S.E02_DIR / "manifest.json").read_text())
    return S.rehearse_e02_step5(pred, obs, REPO / S.E02_DIR / "protocol.json", man,
                                tmp / "e02_step5")


def add_score_look(pdir: Path, prereg_id: str, test_id: str) -> None:
    """A score look as prereg3.assert_can_score logs it (tmp registry only)."""
    now = prereg3._now()
    prereg3._v1._append_chained(pdir / "looks.jsonl", {
        "time": now, "logged": now, "wp": "N2", "purpose": "score", "test_id": test_id,
        "prereg_id": prereg_id, "units": ["N2C:synthetic_stand_in"],
        "data_scope": S.DATA_SCOPE, "scope_windows": "2026-11..2027-10", "members": {},
        "seeded": False, "note": "test fixture", "tool": "test"})


def score_args(tmp: Path, product: Path, step5: Path | None, n2c: Path | None, out="score"):
    return SimpleNamespace(rehearsal=False, synthetic=None, window_product=str(product),
                           e02_score_dir=str(step5) if step5 else None,
                           e02_new_wells=str(product / "new_wells"),
                           n2c_report=str(n2c) if n2c else None, out=str(tmp / out))


def power_report(tmp: Path, ff=0.05, pw=0.9, pdir: Path | None = None) -> Path:
    """A result file in power_u1hb.py's format (numbers synthetic), bound to the tmp
    registry's U1.HB.power.v1 declaration when ``pdir`` is given."""
    def rec(f, p):
        return {"false_fail_max": {"value": f}, "power_min_at_0.75": {"value": p},
                "fail_wording": "fail_shock" if f > 0.10 else "fail_plain",
                "pass_wording": "pass_weak" if p < 0.8 else "pass_plain"}
    p = tmp / "power_report.json"
    rec_i = P.index_record(S.POWER_ID, pdir) if pdir is not None else None
    decl = {"id": S.POWER_ID, "sha256": rec_i["sha256"]} if rec_i else None
    p.write_text(json.dumps({"id": S.POWER_ID, "smoke": False, "engineering_failure": False,
                             "declaration": decl,
                             "wording": {"yunlin": rec(ff, pw), "changhua": rec(ff, pw)}}))
    return p


def register_power_decl(pdir: Path, rows: Path) -> None:
    d = {"id": S.POWER_ID, "wp": "U1", "kind": "declaration", "title": "synthetic stand-in",
         "tests": prereg3.not_applicable("declaration"), "zones": prereg3.not_applicable("x"),
         "priors": prereg3.not_applicable("x"), "burn_in": prereg3.not_applicable("x"),
         "folds": prereg3.not_applicable("x"), "fresh_units": prereg3.not_applicable("x"),
         "thresholds": prereg3.not_applicable("x"), "gate_scores": prereg3.not_applicable("x"),
         "primary_model": prereg3.not_applicable("x"), "inputs": prereg3.not_applicable("x"),
         "interpretation": "test fixture only"}
    prereg3.write_prereg(d, prereg_dir=pdir, rows_dir=rows)


# ----------------------------------------------------------------------------- static
def test_no_home_paths_or_user_names():
    pat = re.compile(r"(/home/|-home-)[A-Za-z0-9_]")
    for p in sorted(X.rglob("*.py")):
        assert not pat.search(p.read_text()), p


def test_module_header_and_constants():
    for p in sorted(X.rglob("*.py")):
        assert "from __future__ import annotations" in p.read_text(), p
    assert dt.datetime(2026, 10, 31, 16, 0, tzinfo=UTC) == S.DEADLINE
    assert S.NOT_BEFORE.date() == dt.date(2028, 1, 15)
    assert set(P.FROZEN) == {f"{S.E}/{n}" for n in ("score_e02c.py", "export_baselines.py",
                                                    "prereg_e02c.py")}
    assert set(PX.FROZEN_EXT) == {f"{S.E}/{n}" for n in ("score_e02c.py",
                                                         "export_baselines.py",
                                                         "prereg_e02c_ext.py")}
    assert set(P.CODE) == {"hydrophysics/twin/bands.py", "hydrophysics/twin/drift_diag.py",
                           "dataprep_staging/harness/coverage3.py"}
    assert "dataprep_staging/common/guard.py" in P.CODE_AT_REGISTRATION
    assert not set(P.CODE) & set(P.CODE_AT_REGISTRATION)
    assert PX.unit_id("2026-12") == "N3F:ext_heads_2026-12_2027-11"


@needs_repo
def test_unit_ids_verified_from_registry():
    i1 = json.loads((REPO / f"results/dataprep/preregistration/{S.I1_PREREG}.json").read_text())
    assert i1["fresh_units"] == [S.UNIT]
    ids = sorted(_pred()["station_id"].unique())
    assert len(ids) == 158 and sum(s.startswith("09") for s in ids) == 84
    assert sum(s.startswith("07") for s in ids) == 74
    assert P.i1_unit() == i1["tests"][0]["external_units"][S.UNIT]
    ext = PX.n1_ext_ids()
    assert len(ext) == 138 and not set(ext) & set(ids)


@needs_repo
def test_doc_validates_and_binds():
    d = P.doc()
    assert prereg3.validate_prereg(d) == []
    assert [t["test_id"] for t in d["tests"]] == [S.TEST_COV, S.TEST_LEVEL]
    for t in d["tests"]:
        assert t["fresh_units"] == [S.UNIT] and t["data_scope"] == S.DATA_SCOPE
    assert d["binds"]["e02"]["predictions.csv"]["sha256"] == S.E02_PRED_SHA
    assert d["binds"]["e02"]["protocol.json"]["sha256"] == S.E02_PROTOCOL_SHA
    for k, (_, h) in S.SECONDARIES.items():
        assert d["binds"]["secondaries_report_only"][k]["predictions.csv"]["sha256"] == h
    assert "2023-26" in d["disclosures"]["U7_interval_family"]
    assert "zh" in P.U7_LABEL and "2023" in P.U7_LABEL["zh"]
    assert "no X1" in d["disclosures"]["scope"]
    assert d["power"]["flags"]["status"] == P.PENDING
    assert d["frozen_code"] == {p: S.sha256(REPO / p) for p in P.FROZEN}
    # shared library modules are recorded, not hashed inputs (a routine edit must not leave
    # E02c unscorable); the ext script is not an E02c input
    for p in P.CODE_AT_REGISTRATION + (f"{S.E}/prereg_e02c_ext.py",):
        assert p not in d["inputs"]
    for p in P.CODE + P.FROZEN:
        assert p in d["inputs"]
    assert set(d["binds"]["code_at_registration_report_only"]) == set(P.CODE_AT_REGISTRATION)
    assert "design_phase_engineering_note" in d["disclosures"]
    assert "not_scorable" in d["scoring"]["if_order_breached"]
    assert "ONCE" in d["attempt_policy"]
    assert any("N2C" in s for s in d["scoring"]["same_session_order"][:3])
    assert P.check_blocking(d) == []


def test_power_flags(tmp_path):
    f = P.power_flags(power_report(tmp_path, ff=0.12, pw=0.7))
    assert f["Yunlin"]["fail_may_reflect_shock"] and f["Changhua"]["pass_no_gross_only"]
    f = P.power_flags(power_report(tmp_path, ff=0.05, pw=0.95))
    assert not f["Yunlin"]["fail_may_reflect_shock"]
    assert not f["Changhua"]["pass_no_gross_only"]
    bad = tmp_path / "bad.json"
    good = json.loads(power_report(tmp_path).read_text())
    for patch in ({"id": "other"}, {"smoke": True}, {"engineering_failure": True},
                  {"wording": {"yunlin": good["wording"]["yunlin"]}}):
        bad.write_text(json.dumps({**good, **patch}))
        with pytest.raises(SystemExit):
            P.power_flags(bad)
    inconsistent = json.loads(json.dumps(good))
    inconsistent["wording"]["yunlin"]["false_fail_max"]["value"] = 0.5
    bad.write_text(json.dumps(inconsistent))
    with pytest.raises(SystemExit):
        P.power_flags(bad)
    undetermined = json.loads(json.dumps(good))
    undetermined["wording"]["changhua"] = {"fail_wording": "not determined",
                                           "pass_wording": "not determined"}
    bad.write_text(json.dumps(undetermined))
    f = P.power_flags(bad)
    assert f["Changhua"]["fail_may_reflect_shock"] and f["Changhua"]["pass_no_gross_only"]
    assert P.power_flags(None)["status"] == P.PENDING
    f = P.power_flags(None, conservative=True)
    assert f["status"] == P.CONSERVATIVE
    assert all(f[c]["fail_may_reflect_shock"] and f[c]["pass_no_gross_only"]
               for c in S.COUNTIES_COV)


# ----------------------------------------------------------------------------- scorer parts
def test_cov_outcome_table():
    c = {"county": "Yunlin", "evaluable": True, "why": None,
         "G1": {"pass": True, "coverage": 0.91, "n": 900},
         "G2": {"balanced": True, "miss_below": 0.04, "miss_above": 0.05}}
    strong = {"fail_may_reflect_shock": False, "pass_no_gross_only": False}
    weak = {"fail_may_reflect_shock": True, "pass_no_gross_only": True}
    r = S.cov_outcome(True, c, strong)
    assert r["verdict"] == "coverage_supported" and "passed its registered gate" in r["wording"]
    assert "balanced" in r["wording"] and "calibrated" not in r["wording"]
    assert "no gross under-coverage" in S.cov_outcome(True, c, weak)["wording"]
    r = S.cov_outcome(False, c, strong)
    assert r["verdict"] == "pooled_gate_failed_county_report_only"
    assert r["wording"] == ("registered pooled gate failed; Yunlin stratum within 0.85-0.95 "
                            "(report-only)")
    c2 = dict(c, G1={"pass": False, "coverage": 0.71, "n": 900})
    r = S.cov_outcome(True, c2, weak)
    assert r["verdict"] == "coverage_not_supported" and "one year's shock" in r["wording"]
    assert "below" in r["wording"]
    assert S.cov_outcome(None, c, strong)["verdict"] == "not_evaluable"
    # gate-3 passed without the point window (n_eff_lower < 30), coverage outside: variant
    c3 = dict(c, G1={"pass": True, "coverage": 0.81, "n": 300})
    r = S.cov_outcome(False, c3, strong)
    assert r["verdict"] == "pooled_gate_failed_county_report_only"
    assert "within 0.85-0.95" not in r["wording"]
    assert "coverage 0.81 passed gate-3 outside 0.85-0.95" in r["wording"]
    assert S.cov_outcome(True, dict(c, evaluable=False, why="k"), strong)["verdict"] == \
        "not_evaluable"


def test_paired_bootstrap_deterministic_and_signed():
    rng = np.random.default_rng(0)
    rows = []
    for w in range(40):
        for m in range(12):
            o = rng.normal()
            rows.append({"station_id": f"w{w}", "date": pd.Timestamp(2026, 11, 1)
                         + pd.DateOffset(months=m), "observed_m": o,
                         "a": o + 0.1 * rng.normal(), "b": o + 1.0 * rng.normal()})
    f = pd.DataFrame(rows)
    blocks = {f"w{w}": str(w // 4) for w in range(40)}
    r1 = S.paired_bootstrap(f, "a", "b", blocks, B=300)
    r2 = S.paired_bootstrap(f, "a", "b", blocks, B=300)
    assert r1 == r2 and r1["mse_diff"] < 0 and r1["a_better_block"]
    assert r1["n_blocks"] == 10 and r1["block_ci"][0] <= r1["mse_diff"] <= r1["block_ci"][1]
    r3 = S.paired_bootstrap(f, "b", "a", blocks, B=300)
    assert not r3["a_better_block"]


def test_fair_verdict_gap_shapes():
    rng = np.random.default_rng(5)
    W, M, gap = 30, 12, 46
    truth = rng.normal(size=(W, 1)) + np.sin(np.arange(S.T_FIT + gap + M) / 2)[None]
    obs_fit = truth[:, :S.T_FIT] + 0.1 * rng.normal(size=(W, S.T_FIT))
    obs_win = truth[:, -M:] + 0.1 * rng.normal(size=(W, M))
    r = S.fair_verdict_gap(truth + 2.0, obs_fit, obs_win)
    assert r["n_wells_scored"] == W and r["n_cells"] == W * M
    assert r["label"].startswith("report-only") and r["fair_k"] == 1.25
    with pytest.raises(ValueError):
        S.fair_verdict_gap(truth, obs_fit[:, :-1], obs_win)


def test_rehearsal_dir_rules(tmp_path):
    with pytest.raises(S.ScoringRefused):
        S.assert_rehearsal_dir(tmp_path)
    with pytest.raises(S.ScoringRefused):
        S.assert_rehearsal_dir(REPO / "results")
    (tmp_path / S.SYNTHETIC_MARKER).write_text("x")
    S.assert_rehearsal_dir(tmp_path)
    pd.DataFrame({"station_id": ["a"], "date": ["2026-11-01"], "observed_m": [1.0],
                  "synthetic": [False]}).to_csv(tmp_path / "observations.csv", index=False)
    with pytest.raises(S.ScoringRefused):
        S.read_synthetic_obs(tmp_path)


def test_real_mode_refuses_before_not_before(tmp_path):
    a = SimpleNamespace(rehearsal=False, synthetic=None, window_product=str(tmp_path),
                        e02_score_dir=str(tmp_path), e02_new_wells=str(tmp_path),
                        n2c_report=str(tmp_path / "n2c.json"), new_wells=str(tmp_path),
                        e02c_report=str(tmp_path / "e.json"), out=str(tmp_path / "o"),
                        window="2026-11")
    with pytest.raises(S.ScoringRefused):
        S.run_score(a, prereg_dir=tmp_path, rows_dir=tmp_path)
    with pytest.raises(S.ScoringRefused):
        S.run_score_ext(a, prereg_dir=tmp_path, rows_dir=tmp_path)
    assert S.main(["score", "--out", str(tmp_path / "o"), "--window-product", str(tmp_path),
                   "--e02-score-dir", str(tmp_path), "--e02-new-wells", str(tmp_path),
                   "--n2c-report", str(tmp_path / "n2c.json")]) == 2
    with pytest.raises(SystemExit):                 # real mode needs every provenance path
        S.main(["score", "--out", str(tmp_path / "o"), "--window-product", str(tmp_path)])
    assert not (tmp_path / "o").exists()


def test_product_grants_scorer_reads_manifest_only(tmp_path):
    assert S.product_grants_scorer(make_product(tmp_path))[0]
    (tmp_path / "b").mkdir()
    ok, why = S.product_grants_scorer(make_product(tmp_path / "b", grant=False))
    assert not ok and "prospective_scorer" in why
    assert not S.product_grants_scorer(tmp_path / "nowhere")[0]
    assert S._inside(tmp_path / "window_product" / "new_wells", tmp_path / "window_product")
    assert not S._inside(tmp_path, tmp_path / "window_product")


def test_order_status_records(tmp_path):
    pdir = tmp_path / "prereg"
    pdir.mkdir()
    st = S.n2c_order_status(pdir, None)
    assert not st["ok"] and "step (2)" in st["why"]
    rep_ = tmp_path / "n2c.json"
    rep_.write_text(json.dumps({"prereg_id": "other"}))
    assert not S.n2c_order_status(pdir, rep_)["ok"]
    rep_.write_text(json.dumps({"prereg_id": S.N2C_ID, "verdict": "x"}))
    st = S.n2c_order_status(pdir, rep_)
    assert not st["ok"] and "step (1)" in st["why"]
    add_score_look(pdir, S.N2C_ID, S.N2C_TEST)
    assert S.n2c_order_status(pdir, rep_)["ok"]
    rec = tmp_path / "n2c_ns.json"
    rec.write_text(json.dumps({"prereg_id": S.N2C_ID, "verdict": S.NOT_SCORABLE}))
    assert S.n2c_order_status(tmp_path / "empty", rec)["ok"]
    # EXT after E02c: a report that scored needs its looks; a not-evaluable one does not
    e = tmp_path / "e02c.json"
    assert not S.e02c_done_status(pdir, e)["ok"]
    e.write_text(json.dumps({"prereg_id": S.ID, "T_cov": {}}))
    assert not S.e02c_done_status(pdir, e)["ok"]
    e.write_text(json.dumps({"prereg_id": S.ID, "verdict": "not_evaluable"}))
    assert S.e02c_done_status(pdir, e)["ok"]
    e.write_text(json.dumps({"prereg_id": S.ID, "verdict": S.NOT_SCORABLE}))
    assert S.e02c_done_status(pdir, e)["ok"]


# ----------------------------------------------------------------------------- rehearsal
@needs_repo
def test_rehearsal_pass_and_fail(tmp_path):
    pred = _pred()
    d = make_synth_dir(tmp_path, pred)
    a = SimpleNamespace(rehearsal=True, synthetic=str(d), out=str(tmp_path / "r1"))
    rep = S.run_score(a)
    assert rep["mode"].startswith("REHEARSAL")
    cov = rep["T_cov"]
    assert cov["G0"]["pass"] is True
    for c in S.COUNTIES_COV:
        g1 = cov["counties"][c]["G1"]
        assert g1["rule"] == "gate-3" and g1["method"] == "upper90" and g1["calibrated"]
        assert cov["counties"][c]["G1b_report_only"]["label"].startswith("uncalibrated")
        assert cov["counties"][c]["verdict"] == "coverage_supported"
    assert rep["T_level"]["verdict"] == "level_forecast_supported"
    assert rep["e02_own_verdict_unchanged"]["checks"]["enough_stations"]
    with pytest.raises(FileExistsError):                 # the report is written once
        S.run_score(a)
    # a forecast offset by 3 sigma: coverage collapses and the level rule fails
    d2 = tmp_path / "s2"
    d2.mkdir()
    for n in ("baselines_e02c.csv", "wells_158.csv", "summary.json", S.SYNTHETIC_MARKER):
        (d2 / n).write_bytes((d / n).read_bytes())
    o = synthetic_obs(pred, shift=0.0)
    sig = (pred["upper_m"] - pred["lower_m"]) / (2 * 1.6448536269514722)
    o["observed_m"] = o["observed_m"] + 3 * sig
    o.to_csv(d2 / "observations.csv", index=False)
    b = pd.read_csv(d2 / "baselines_e02c.csv", dtype={"station_id": str})
    b["cl_m"] = o["observed_m"].to_numpy()                # an oracle baseline
    b.to_csv(d2 / "baselines_e02c.csv", index=False)
    rep2 = S.run_score(SimpleNamespace(rehearsal=True, synthetic=str(d2),
                                       out=str(tmp_path / "r2")))
    assert rep2["T_cov"]["G0"]["pass"] is False
    for c in S.COUNTIES_COV:
        assert rep2["T_cov"]["counties"][c]["verdict"] == "coverage_not_supported"
    assert rep2["T_level"]["verdict"] == "level_forecast_not_supported"


# ----------------------------------------------------------------------------- end to end
@needs_repo
def test_write_refuses_without_slots(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "_now", lambda: BEFORE_DEADLINE)
    pdir, rows = tmp_path / "prereg", tmp_path / "rows"
    rc = P.main(["--write", "--quiet", "--expect-self-sha", S.sha256(Path(P.__file__))],
                prereg_dir=pdir, rows_dir=rows)
    assert rc == 1 and not (pdir / f"{S.ID}.json").exists()


@needs_repo
def test_register_and_score_in_tmp_registry(tmp_path, monkeypatch):
    pred = _pred()
    pdir, rows = tmp_path / "prereg", tmp_path / "rows"
    d = make_synth_dir(tmp_path, pred)
    pr = power_report(tmp_path, ff=0.2, pw=0.5)
    monkeypatch.setattr(S, "_now", lambda: BEFORE_DEADLINE)
    self_sha = S.sha256(Path(P.__file__))
    # without the power declaration in the registry: refused
    assert P.main(["--write", "--quiet", "--baselines-dir", str(d), "--power-report",
                   str(pr), "--expect-self-sha", self_sha], prereg_dir=pdir,
                  rows_dir=rows) == 1
    register_power_decl(pdir, rows)
    # a report not bound to the registered declaration: refused
    assert P.main(["--write", "--quiet", "--baselines-dir", str(d), "--power-report",
                   str(pr), "--expect-self-sha", self_sha], prereg_dir=pdir,
                  rows_dir=rows) == 1
    pr = power_report(tmp_path, ff=0.2, pw=0.5, pdir=pdir)
    assert P.main(["--write", "--quiet", "--baselines-dir", str(d), "--power-report",
                   str(pr), "--expect-self-sha", "0" * 64], prereg_dir=pdir,
                  rows_dir=rows) == 1
    # after the deadline: refused
    monkeypatch.setattr(S, "_now", lambda: S.DEADLINE)
    assert P.main(["--write", "--quiet", "--baselines-dir", str(d), "--power-report",
                   str(pr), "--expect-self-sha", self_sha], prereg_dir=pdir,
                  rows_dir=rows) == 1
    monkeypatch.setattr(S, "_now", lambda: BEFORE_DEADLINE)
    assert P.main(["--write", "--quiet", "--baselines-dir", str(d), "--power-report",
                   str(pr), "--expect-self-sha", self_sha], prereg_dir=pdir,
                  rows_dir=rows) == 0
    doc, _ = prereg3.load_prereg(S.ID, prereg_dir=pdir, rows_dir=rows)
    assert doc["power"]["flags"]["Yunlin"]["fail_may_reflect_shock"]
    assert doc["power"]["flags"]["Changhua"]["pass_no_gross_only"]
    assert not (REPO / "results/dataprep/preregistration" / f"{S.ID}.json").exists() or \
        S.ID in REAL_INDEX.read_text()                    # nothing new in the real registry
    # scoring day. E02's step-(5) output from synthetic heads; no product table is read
    monkeypatch.setattr(S, "_now", lambda: SCORING_DAY)
    monkeypatch.setattr(S, "fair_inputs_real", _fair_stub)
    obs = synthetic_obs(pred)
    step5 = make_e02_step5(tmp_path, pred, obs)
    product = make_product(tmp_path)
    n2c = tmp_path / "n2c_report.json"
    real_load = S.load_e02_step5

    def no_read(*a, **k):
        raise AssertionError("a window row was read before the order check")
    monkeypatch.setattr(S, "load_e02_step5", no_read)
    # N2C's step (2) report missing: refused, nothing read, no look
    with pytest.raises(S.ScoringRefused, match="step \\(2\\)"):
        S.run_score(score_args(tmp_path, product, step5, n2c), prereg_dir=pdir, rows_dir=rows)
    n2c.write_text(json.dumps({"prereg_id": S.N2C_ID, "verdict": "synthetic"}))
    # report present but N2C's step-(1) look missing: refused, nothing read, no look
    with pytest.raises(S.ScoringRefused, match="step \\(1\\)"):
        S.run_score(score_args(tmp_path, product, step5, n2c), prereg_dir=pdir, rows_dir=rows)
    assert prereg3.read_chained(pdir / "looks.jsonl") == []
    # new_wells outside the product directory: refused before anything
    a = score_args(tmp_path, product, step5, n2c)
    a.e02_new_wells = str(tmp_path)
    with pytest.raises(S.ScoringRefused, match="inside the window product"):
        S.run_score(a, prereg_dir=pdir, rows_dir=rows)
    add_score_look(pdir, S.N2C_ID, S.N2C_TEST)
    # the look is logged BEFORE any window row is read
    seen = {}

    def load_after_look(*a, **k):
        seen["looks"] = {(lk["prereg_id"], lk["test_id"])
                         for lk in prereg3.read_chained(pdir / "looks.jsonl")}
        return real_load(*a, **k)
    monkeypatch.setattr(S, "load_e02_step5", load_after_look)
    rep = S.run_score(score_args(tmp_path, product, step5, n2c), prereg_dir=pdir,
                      rows_dir=rows)
    assert {(S.ID, S.TEST_COV), (S.ID, S.TEST_LEVEL)} <= seen["looks"]
    e02 = json.loads((step5 / "report.json").read_text())
    assert rep["e02_own_verdict_unchanged"] == e02            # G0 IS E02's verdict
    assert rep["T_cov"]["G0"]["pass"] is e02["checks"]["interval_coverage"]
    assert rep["e02_step5"]["report.json_sha256"] == S.sha256(step5 / "report.json")
    for c in S.COUNTIES_COV:
        assert rep["T_cov"]["counties"][c]["verdict"] in (
            "coverage_supported", "coverage_not_supported")
        if rep["T_cov"]["counties"][c]["verdict"] == "coverage_supported":
            assert "no gross under-coverage" in rep["T_cov"]["counties"][c]["wording"]
    assert rep["T_level"]["verdict"] == "level_forecast_supported"
    assert rep["T_level_fair_rule_report_only"]["n_wells_scored"] == 20
    for k in S.SECONDARIES:
        sec = rep["secondaries_report_only"][k]
        assert "error" not in sec and sec["label"].startswith("report-only")
    assert rep["order"]["ok"] and rep["order"]["n2c_score_look"]
    looks = prereg3.read_chained(pdir / "looks.jsonl")
    assert {(lk["test_id"], lk["purpose"]) for lk in looks if lk["prereg_id"] == S.ID} == {
        (S.TEST_COV, "score"), (S.TEST_LEVEL, "score")}
    with pytest.raises(FileExistsError):                      # the report is written once
        S.run_score(score_args(tmp_path, product, step5, n2c), prereg_dir=pdir, rows_dir=rows)
    # the external-well test may follow: E02c's report exists and its looks are logged
    assert S.e02c_done_status(pdir, tmp_path / "score" / "report.json")["ok"]


@needs_repo
def test_step5_inconsistencies_are_engineering_failures(tmp_path):
    pred = _pred()
    step5 = make_e02_step5(tmp_path, pred, synthetic_obs(pred))
    man = json.loads((REPO / S.E02_DIR / "manifest.json").read_text())
    proto = REPO / S.E02_DIR / "protocol.json"
    frame, rep, prov = S.load_e02_step5(step5, pred, proto, man)
    assert len(frame) == prov["rows"] and set(frame["county"]) <= {"Yunlin", "Changhua"}
    with pytest.raises(S.EngineeringFailure, match="manifest"):
        S.load_e02_step5(step5, pred, proto, dict(man, rows=0))
    ev = pd.read_csv(step5 / "evaluation.csv", dtype={"station_id": str})
    ev.loc[0, "upper_m"] += 1.0                                  # a widened interval
    ev.to_csv(step5 / "evaluation.csv", index=False)
    with pytest.raises(S.EngineeringFailure, match="upper_m"):
        S.load_e02_step5(step5, pred, proto, man)
    ev.loc[0, "upper_m"] -= 1.0
    ev.to_csv(step5 / "evaluation.csv", index=False)
    r = json.loads((step5 / "report.json").read_text())
    (step5 / "report.json").write_text(json.dumps(dict(r, coverage=0.5)))
    with pytest.raises(S.EngineeringFailure, match="reproduce"):
        S.load_e02_step5(step5, pred, proto, man)


@needs_repo
def test_power_conservative_fallback_registers(tmp_path, monkeypatch):
    pred = _pred()
    pdir, rows = tmp_path / "prereg", tmp_path / "rows"
    d = make_synth_dir(tmp_path, pred)
    monkeypatch.setattr(S, "_now", lambda: BEFORE_DEADLINE)
    sha = S.sha256(Path(P.__file__))
    base = ["--write", "--quiet", "--baselines-dir", str(d), "--expect-self-sha", sha]
    assert P.main(base + ["--power-conservative", "--power-report",
                          str(power_report(tmp_path))], prereg_dir=pdir, rows_dir=rows) == 1
    assert P.main(base + ["--power-conservative"], prereg_dir=pdir, rows_dir=rows) == 0
    doc, _ = prereg3.load_prereg(S.ID, prereg_dir=pdir, rows_dir=rows)
    assert doc["power"]["flags"]["status"] == P.CONSERVATIVE
    assert doc["power"]["report"] == P.CONSERVATIVE


@needs_repo
def test_baselines_block_needs_reference_and_verify(tmp_path):
    d = make_synth_dir(tmp_path, _pred())
    assert P.baselines_block(d)["status"] == "ok"
    v = json.loads((d / "verify.json").read_text())
    (d / "verify.json").write_text(json.dumps(dict(v, identical=False)))
    assert P.baselines_block(d)["status"] == "not verified"
    (d / "verify.json").unlink()
    assert P.baselines_block(d)["status"] == "incomplete"
    write_verify(d, ("baselines_e02c.csv", "wells_158.csv"))
    s = json.loads((d / "summary.json").read_text())
    (d / "summary.json").write_text(json.dumps(dict(s, reference=None)))
    assert P.baselines_block(d)["status"] == "no reference decision"


@needs_repo
def test_unavailable_scorer_product_is_not_evaluable(tmp_path, monkeypatch):
    pred = _pred()
    pdir, rows = tmp_path / "prereg", tmp_path / "rows"
    d = make_synth_dir(tmp_path, pred)
    register_power_decl(pdir, rows)
    monkeypatch.setattr(S, "_now", lambda: BEFORE_DEADLINE)
    assert P.main(["--write", "--quiet", "--baselines-dir", str(d), "--power-report",
                   str(power_report(tmp_path, pdir=pdir)), "--expect-self-sha",
                   S.sha256(Path(P.__file__))], prereg_dir=pdir, rows_dir=rows) == 0
    monkeypatch.setattr(S, "_now", lambda: SCORING_DAY)

    def no_read(*a, **k):
        raise AssertionError("read a window row")
    monkeypatch.setattr(S, "load_e02_step5", no_read)
    product = make_product(tmp_path, grant=False)
    rep = S.run_score(score_args(tmp_path, product, tmp_path, None, out="s"),
                      prereg_dir=pdir, rows_dir=rows)
    assert rep["verdict"] == "not_evaluable" and "prospective_scorer" in rep["why"]
    assert prereg3.read_chained(pdir / "looks.jsonl") == []


# ----------------------------------------------------------------------------- export
def _fw(T=252, nx=6, ny=5, seed=0):
    rng = np.random.default_rng(seed)
    mask = np.ones((ny, nx), bool)
    mask[0, 0] = False
    C = int(mask.sum())
    hm = rng.normal(size=(3, 4, C, T)).astype("float32")
    return {"dates": np.array([str(d) for d in pd.date_range("2012-01-01", periods=T,
                                                             freq="MS")]),
            "origin": np.array(131), "heads_mean": hm,
            "heads_std": np.full((3, 4, C, T), 0.2, "float32"),
            "scenario_names": np.array(["baseline", "cut30", "retire_aqua"]),
            "mask": mask, "nx": np.array(nx), "ny": np.array(ny), "dx": np.array(1000.0),
            "x0": np.array(0.0), "y0": np.array(0.0)}


def test_obs_matrix_drops_post_origin_rows():
    long = pd.DataFrame({"sid": ["a", "a", "a", "b"],
                         "date": ["2012-01-15", "2022-12-01", "2023-01-01", "2015-06-01"],
                         "value": [1.0, 2.0, 99.0, 3.0], "flag": [0, 1, 0, 9]})
    ax = EB.axis("2027-10")
    M = EB.obs_matrix(long, ["a", "b"], ax)
    assert M[0, 0] == 1.0 and M[0, 131] == 2.0
    assert np.isnan(M[:, 132:]).all() and np.isnan(M[1]).all()


def test_baselines_table_matches_fair_baselines():
    from hydrophysics.twin.drift_diag import fair_baselines

    rng = np.random.default_rng(1)
    ax = EB.axis("2027-10")
    obs = np.full((3, len(ax)), np.nan)
    obs[:, :132] = rng.normal(size=(3, 132)) + np.arange(132) * 0.01
    obs[2, :120] = np.nan
    t = EB.baselines_table(["a", "b", "c"], obs, ax, *S.WINDOW)
    b = fair_baselines(obs.copy(), 132, month_offset=0)
    j = np.flatnonzero(ax >= pd.Timestamp("2026-11-01"))
    assert len(t) == 36
    assert np.allclose(t[t.station_id == "a"].cl_m, b["clim"][0, j])
    assert np.allclose(t[t.station_id == "b"].clt_m, b["clim_trend"][1, j])
    assert (t[t.station_id == "c"].n_fit_months == 12).all()


def test_ext_predictions_recipe_and_rule():
    fw = _fw()
    wells = pd.DataFrame({"station_id": ["09000001", "09000002", "07000003", "07000004"],
                          "county": ["Yunlin", "Yunlin", "Changhua", "Changhua"],
                          "x": [1500.0, 2500.0, 3500.0, 500.0],
                          "y": [1500.0, 2500.0, 3500.0, 500.0],
                          "layer_code": ["2", "1", "3", "2"], "layer": [1, 0, 2, 1],
                          "cell": [0, 5, 10, -1]})
    T = 252
    rng = np.random.default_rng(2)
    obs = np.full((4, T), np.nan)
    for w in range(3):
        h = fw["heads_mean"][0, wells.layer[w], wells.cell[w]].astype(float)
        obs[w, :132] = h[:132] + 4.0 + 0.3 * rng.normal(size=132)
    obs[1, :112] = np.nan                                  # only 20 fitted months
    obs[3, :132] = 1.0                                     # no active cell
    pred, base, sel = EB.ext_predictions(wells, obs, fw, "2026-11", "2027-10")
    assert sel.kept.tolist() == [True, False, True, False]
    assert sel.reason.tolist()[1] == "< 24 pre-origin months"
    assert sorted(pred.station_id.unique()) == ["07000003", "09000001"]
    assert abs(sel.datum_m[0] - 4.0) < 0.1 and len(pred) == 24 and len(base) == 24
    p = pred[pred.station_id == "09000001"]
    h = fw["heads_mean"][0, 1, 0, 178:190].astype(float)
    assert np.allclose(p.predicted_m - h, sel.datum_m[0])
    half = (p.upper_m - p.lower_m) / 2 / EB.Z90
    assert np.allclose(half, np.sqrt(0.2 ** 2 + sel.residual_sd_m[0] ** 2), atol=1e-6)
    assert (p.prediction_origin == "2022-12-01").all()


def test_well_table_cells_and_blocks():
    fw = _fw()
    meta = pd.DataFrame({"sid": ["a", "b", "c"], "x_twd97": [1500.0, 500.0, np.nan],
                         "y_twd97": [1500.0, 500.0, 1.0], "layer_code": ["2", "1", "5"],
                         "CountyName": ["x", "y", "z"]})
    t = EB.well_table(meta, ["a", "b", "c", "d"], fw)
    assert t.cell.tolist()[1:] == [-1, -1, -1] and t.cell[0] >= 0
    assert t.layer.tolist() == [1, 0, -1, -1]
    assert t.block5.tolist()[2:] == ["none", "none"]


def test_reference_rule_both_branches():
    pred = pd.DataFrame({"station_id": ["a"] * 2 + ["b"] * 2,
                         "date": ["2026-11-01", "2026-12-01"] * 2,
                         "baseline_m": [1.0, 2.0, 5.0, 6.0]})
    base = pd.DataFrame({"station_id": pred.station_id, "date": pred.date,
                         "cl_m": pred.baseline_m + 0.01, "clt_m": pred.baseline_m + 0.31,
                         "n_fit_months": 100})
    t, info = EB.reference_rule(pred, base)
    assert info["chosen"] == EB.REF_L1 and np.allclose(t.cl_m, base.cl_m)
    assert np.allclose(t.clt_m, base.clt_m)
    base2 = base.assign(cl_m=pred.baseline_m + 0.4, clt_m=pred.baseline_m + 0.7)
    t, info = EB.reference_rule(pred, base2)
    assert info["chosen"] == EB.REF_E02 and info["n_wells_over_threshold"] == 2
    assert np.allclose(t.cl_m, pred.baseline_m)
    assert np.allclose(t.clt_m, pred.baseline_m + 0.3)            # L1's trend departure
    assert list(t.columns[:5]) == ["station_id", "date", "cl_m", "clt_m", "n_fit_months"]


def test_verify_export(tmp_path):
    def build(o):
        o.mkdir(parents=True)
        pd.DataFrame({"x": [1, 2]}).to_csv(o / "a.csv", index=False)
        s = {"files": {"a.csv": S.sha256(o / "a.csv")}, "reference": {"chosen": EB.REF_L1}}
        (o / "summary.json").write_text(json.dumps(s))
        return s
    out = tmp_path / "exp"
    build(out)
    r = EB.verify_export(out, build)
    assert r["identical"] and r["files"] == {"a.csv": S.sha256(out / "a.csv")}
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".verify_")]
    with pytest.raises(FileExistsError):                         # written once
        EB.verify_export(out, build)
    out2 = tmp_path / "exp2"
    build(out2)
    pd.DataFrame({"x": [1, 3]}).to_csv(out2 / "a.csv", index=False)   # drifted on disk
    assert not EB.verify_export(out2, build)["identical"]


def test_ext_protocol_is_e02_shaped():
    p = EB.ext_protocol("2026-11", "2027-10", "2026-10-30")
    assert p["test_end"] == "2027-10-31" and p["min_stations"] == 30
    assert p["max_rmse_ratio"] == 1.0 and p["coverage_bounds"] == [0.85, 0.95]


# ----------------------------------------------------------------------------- ext prereg
def _synthetic_ext_export(tmp: Path, n: int, window="2026-11") -> Path:
    """An export directory shaped like export_baselines.py ext's, from synthetic numbers on
    real N1:ext_wells ids (ids only)."""
    ids = PX.n1_ext_ids()[:n]
    w = PX.WINDOWS[window]
    months = pd.date_range(f"{w['first']}-01", f"{w['last']}-01", freq="MS")
    rows = [{"station_id": s, "layer": 2, "date": str(m.date()),
             "prediction_origin": "2022-12-01", "predicted_m": 1.0, "baseline_m": 1.1,
             "lower_m": 0.0, "upper_m": 2.0, "model_sha256": S.MODEL_SHA}
            for s in ids for m in months]
    d = tmp / "ext"
    d.mkdir()
    pd.DataFrame(rows).to_csv(d / "predictions.csv", index=False)
    pd.DataFrame([{"station_id": r["station_id"], "date": r["date"], "cl_m": 1.1,
                   "clt_m": 1.2, "n_fit_months": 60} for r in rows]).to_csv(
        d / "baselines.csv", index=False)
    pd.DataFrame({"station_id": ids, "county": [S.county_of(s) for s in ids],
                  "kept": True, "reason": "kept"}).to_csv(d / "wells.csv", index=False)
    (d / "protocol.json").write_text(json.dumps(EB.ext_protocol(w["first"], w["last"],
                                                                "2026-10-30")))
    (d / "manifest.json").write_text("{}")
    files = {k: S.sha256(d / k) for k in ("predictions.csv", "baselines.csv", "wells.csv",
                                          "protocol.json", "manifest.json")}
    (d / "summary.json").write_text(json.dumps({"window": [w["first"], w["last"]],
                                                "files": files}))
    write_verify(d, ("predictions.csv", "baselines.csv", "wells.csv"))
    return d


@needs_repo
def test_ext_prereg_validates_writes_and_orders(tmp_path, monkeypatch):
    pdir, rows = tmp_path / "prereg", tmp_path / "rows"
    d = _synthetic_ext_export(tmp_path, 40)
    doc = PX.doc(d, "2026-11", pdir)
    assert prereg3.validate_prereg(doc, prereg_dir=pdir) == []
    t = doc["tests"][0]
    assert t["fresh_units"] == ["N3F:ext_heads_2026-11_2027-10"]
    assert t["external_units"][t["fresh_units"][0]]["ids"] == PX.n1_ext_ids()[:40]
    assert "E08 and C02 are unchanged" in doc["page_mapping"]["never"]
    monkeypatch.setattr(S, "_now", lambda: BEFORE_DEADLINE)
    sha = S.sha256(Path(PX.__file__))
    assert PX.main(["--write", "--quiet", "--ext-dir", str(d), "--expect-self-sha", sha],
                   prereg_dir=pdir, rows_dir=rows) == 0
    assert set(doc["frozen_code"]) == set(PX.FROZEN_EXT)
    assert f"{S.E}/prereg_e02c.py" not in doc["inputs"]
    # scoring refuses before E02c is scored, even on scoring day
    monkeypatch.setattr(S, "_now", lambda: SCORING_DAY)
    product = make_product(tmp_path)
    a = SimpleNamespace(rehearsal=False, synthetic=None, window_product=str(product),
                        new_wells=str(product / "new_wells"),
                        e02c_report=str(tmp_path / "missing.json"),
                        out=str(tmp_path / "x"), window="2026-11")
    with pytest.raises(S.ScoringRefused, match="E02c must be scored first"):
        S.run_score_ext(a, prereg_dir=pdir, rows_dir=rows)
    assert prereg3.read_chained(pdir / "looks.jsonl") == []


@needs_repo
def test_ext_prereg_refuses_small_or_late(tmp_path, monkeypatch):
    pdir, rows = tmp_path / "prereg", tmp_path / "rows"
    d = _synthetic_ext_export(tmp_path, 10, "2026-12")
    sha = S.sha256(Path(PX.__file__))
    monkeypatch.setattr(S, "_now", lambda: BEFORE_DEADLINE)
    assert PX.main(["--write", "--quiet", "--window", "2026-12", "--ext-dir", str(d),
                    "--expect-self-sha", sha], prereg_dir=pdir, rows_dir=rows) == 1
    monkeypatch.setattr(S, "_now", lambda: dt.datetime(2026, 12, 1, tzinfo=UTC))
    (tmp_path / "b").mkdir()
    big = _synthetic_ext_export(tmp_path / "b", 40, "2026-12")
    assert PX.main(["--write", "--quiet", "--window", "2026-12", "--ext-dir", str(big),
                    "--expect-self-sha", sha], prereg_dir=pdir, rows_dir=rows) == 1
    assert not (pdir / f"{PX.prereg_id('2026-12')}.json").exists()


def test_ext_rehearsal(tmp_path):
    d = tmp_path / "synth"
    d.mkdir()
    (d / S.SYNTHETIC_MARKER).write_text("x")
    ids = [f"0900{i:04d}" for i in range(35)]
    months = pd.date_range("2026-11-01", periods=12, freq="MS")
    rng = np.random.default_rng(4)
    rows, obs = [], []
    for s in ids:
        for m in months:
            o = rng.normal()
            rows.append({"station_id": s, "layer": 2, "date": str(m.date()),
                         "prediction_origin": "2022-12-01", "predicted_m": o + 0.2 *
                         rng.normal(), "baseline_m": o + rng.normal(), "lower_m": o - 0.5,
                         "upper_m": o + 0.5 if rng.random() < 0.9 else o - 0.1,
                         "model_sha256": S.MODEL_SHA})
            obs.append({"station_id": s, "date": str(m.date()), "observed_m": o,
                        "synthetic": True})
    (d / "ext").mkdir()
    pd.DataFrame(rows).to_csv(d / "ext" / "predictions.csv", index=False)
    (d / "ext" / "protocol.json").write_text(json.dumps(EB.ext_protocol("2026-11", "2027-10",
                                                                        "2026-10-30")))
    pd.DataFrame(obs).to_csv(d / "observations.csv", index=False)
    rep = S.run_score_ext(SimpleNamespace(rehearsal=True, synthetic=str(d),
                                          out=str(tmp_path / "o"), window="2026-11"))
    assert rep["n_wells"] == 35 and rep["verdict"] in ("passed", "not_supported")
    assert rep["evaluate"]["rmse_ratio"] < 1.0
    # fewer than 30 wells: not informative
    f = S.join_obs(pd.DataFrame(rows[:12 * 20]), pd.DataFrame(obs))
    assert S.ext_outcome(f, None)["verdict"] == "not_informative"
