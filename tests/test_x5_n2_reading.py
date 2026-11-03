"""X5.n2_reading.v1 (round-8 rank 4): the harness of N2's synthetic operating characteristics,
on synthetic stand-in panels and fabricated records only.

- engineering reproduction: the harness (synth_n2c.score_world -> the shared scorer) equals the
  reference computation of N2's registered statistic; exact plumbing; the baseline
  reproduction check;
- the seed has no default and refused seeds are refused (study, draft, runner, CLI);
- the wording map (weak / informative / decisive, X5.n2c_reading.v1's thresholds) on
  FABRICATED cell records: no test draws a world on a model contrast or computes a pass rate
  of N2;
- the runner refuses an unregistered or other id, a tampered registration, changed code, a
  wrong seed, an existing report, a panel from other replays or another forward, and appends
  every attempt;
- the draft declaration's key fields (skipped when the gitignored results/ tree is absent).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("pandas")

REPO = Path(__file__).resolve().parents[1]
D = REPO / "dataprep_staging" / "x5_n2_reading"
for _p in (str(D), str(REPO / "dataprep_staging")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import synth_n2 as S  # noqa: E402

RESULTS = REPO / "results" / "dataprep"
HAVE_RESULTS = (RESULTS / "preregistration" / "INDEX.jsonl").exists() and \
    (REPO / S.PRODUCT / "MANIFEST.json").exists()
needs_results = pytest.mark.skipif(not HAVE_RESULTS, reason="gitignored results/ tree absent")
SEED = S.TEST_SEED                             # refused by draft_prereg, simulate and run


def _panel(n_rep: int = 2) -> dict:
    p = S.synthetic_panel(seed=3)
    return {**p, "c": p["c"][:n_rep], "a_yl": p["a_yl"][:n_rep], "replays": p["replays"][:n_rep]}


# ----------------------------------------------------------------------------- static
def test_scorer_is_the_bound_one():
    assert S.BS.file_sha256() == S.SCORER_SHA
    assert S.SCORER_SHA.startswith("f8990d61")
    assert S.SCORER in S.CODE_FILES and "dataprep_staging/n2c_reading/synth_n2c.py" in \
        S.CODE_FILES


def test_no_home_paths_in_staged_files():
    pat = re.compile(r"(/home/|-home-)[A-Za-z0-9_]")
    for p in sorted(D.rglob("*")):
        if p.is_file() and p.suffix in (".py", ".sh"):
            assert not pat.search(p.read_text()), p


def test_runner_syntax_and_dry_run():
    PR = pytest.importorskip("prereg_n2_reading")
    sh = D / "run_n2_reading.sh"
    subprocess.run(["bash", "-n", str(sh)], check=True)
    out = subprocess.run(["bash", str(sh), "forward"], cwd=REPO,
                         env={**os.environ, "DRY": "1"}, capture_output=True, text=True,
                         check=True).stdout.splitlines()
    assert out == [PR.command()]
    assert "--device cpu" in out[0] and out[0].count("--recorded-pumping") == 7
    assert "--save-members none" in out[0] and "stage3_datum_gate/stage3_theta.json" in out[0]


def test_runner_run_needs_a_seed():
    p = subprocess.run(["bash", str(D / "run_n2_reading.sh"), "run"], cwd=REPO,
                       capture_output=True, text=True)
    assert p.returncode == 1 and "seed" in p.stderr


def test_reading_thresholds_are_n2c_readings():
    assert S.READING == S.SC.READING == {"false_pass_max": 0.25, "wrong_size_max": 0.25,
                                         "coverage_min": 0.80}
    assert S.N2C_READING_ID in S.READING_SOURCE
    assert set(S.WORDING) == {"weak", "informative", "pass_controlled_low_power", "decisive",
                              "not_estimable", "verdict_word"}
    assert S.POWER_MIN == 0.80


# ----------------------------------------------------------------------------- seed
@pytest.mark.parametrize("bad", [None, True, 1.5, "7", -1, 2 ** 63])
def test_seed_has_no_default_and_must_be_an_int(bad):
    with pytest.raises(ValueError):
        S.check_seed(bad)
    with pytest.raises(ValueError):
        S.design_spec(bad)


@pytest.mark.parametrize("bad", S.REFUSED_SEEDS)
def test_refused_seeds(bad):
    with pytest.raises(ValueError, match="refused"):
        S.design_spec(bad)
    with pytest.raises(ValueError, match="refused"):
        S.simulate(_panel(), bad)


def test_test_seed_is_refused_for_real_runs(tmp_path):
    assert S.check_seed(SEED) == SEED
    with pytest.raises(ValueError, match="real runs"):
        S.check_seed(SEED, real=True)
    with pytest.raises(ValueError, match="real runs"):
        S.simulate(_panel(), SEED)
    with pytest.raises(ValueError, match="real runs"):
        S.main(["run", "--seed", str(SEED), "--prereg-id", S.DECL_ID,
                "--panel", str(tmp_path / "p.npz"), "--out-dir", str(tmp_path)])
    PR = pytest.importorskip("prereg_n2_reading")
    with pytest.raises(ValueError, match="real runs"):
        PR.draft_prereg(SEED)


def test_design_spec_needs_the_seed_argument():
    with pytest.raises(TypeError):
        S.design_spec()                         # no default
    d = S.design_spec(SEED)
    assert d["seed"] == SEED and d["grid"]["k"][0] == 0.0 and d["grid"]["rho"] == [0.0, -0.87]
    assert d["scorer"]["sha256"] == S.SCORER_SHA


def test_cli_run_without_seed_fails():
    with pytest.raises(SystemExit) as e:
        S.main(["run", "--prereg-id", S.DECL_ID])
    assert e.value.code == 2


# ----------------------------------------------------------------------------- engineering
def test_reference_statistic_equals_the_harness():
    pan = _panel(1)
    T = pan["c"].shape[2]
    wells = np.repeat(pan["sid"], T)
    bof = dict(zip(pan["sid"], pan["block"], strict=True))
    for _lab, _k, e in S.engineering_worlds(pan["c"][0])[1:3]:
        h = S.harness_statistic(pan, 0, e)
        r = S.reference_statistic(e.ravel(), pan["c"][0].ravel(), wells, bof, T)
        for k in ("beta", "lo", "hi", "rms", "rmse_pr", "rmse_zr"):
            assert abs(h[k] - r[k]) <= 1e-8, k
        assert h["verdict"] == r["verdict"]


def test_reference_statistic_is_well_fixed_effects():
    pan = _panel(1)
    T = pan["c"].shape[2]
    c = pan["c"][0]
    wells = np.repeat(pan["sid"], T)
    bof = dict(zip(pan["sid"], pan["block"], strict=True))
    off = np.arange(c.shape[0])[:, None] * 5.0
    r = S.reference_statistic((0.7 * c + off).ravel(), c.ravel(), wells, bof, T, B=50)
    assert abs(r["beta"] - 0.7) < 1e-9


def test_exact_plumbing_and_gate_on_a_stand_in_panel(monkeypatch):
    pan = _panel(1)
    assert S.SC.exact_plumbing(pan)["ok"]
    monkeypatch.setattr(S, "engineering_worlds", lambda c: [("k1", 1.0, 1.0 * c)])
    g = S.engineering_gate([("stand-in", pan)])
    assert g["ok"] and g["verdict_agreement"] == "1/1" and g["exact_plumbing_ok"]
    assert max(g["max_abs_diff"].values()) <= g["tol"]


def test_baseline_reproduction_check():
    a = np.linspace(0, 30, 84 * 252).reshape(84, 252)
    assert S.reproduction(a, a.copy())["ok"]
    assert S.reproduction(a, a + 5e-4)["ok"]
    assert not S.reproduction(a, a + 2e-3)["ok"]
    assert not S.reproduction(a, a[:, :-1])["ok"]
    b = a.copy()
    b[0, 0] = np.nan
    assert not S.reproduction(a, b)["ok"]


def test_forward_meta_problems_on_a_fake_forward():
    f = {"dates": np.array([f"{2012 + t // 12}-{t % 12 + 1:02d}-01" for t in range(252)]),
         "origin": np.int64(131), "horizon": np.int64(120),
         "member_labels": np.array(["stage3_datum_gate"]), "gain": 1.0, "hindcast_gain": 0.0,
         "forward_options": json.dumps(S.FWD_OPTIONS)}
    assert S.forward_meta_problems(f) == []
    g = {**f, "forward_options": json.dumps({**S.FWD_OPTIONS, "save_members": "yearly"})}
    assert S.forward_meta_problems(g)
    h = {**f, "member_labels": np.array(["stage3_datum_gate_ccz"])}
    assert S.forward_meta_problems(h)


def test_world_noise_parts_and_county_switch():
    pan = _panel(1)
    cell = {"k": 0.0, "rho": 0.0, "kappa_mult": 0.0, "county_mult": 1.0}
    e1 = S.world(pan, 0, cell, np.random.default_rng(1))
    e2 = S.world(pan, 0, cell, np.random.default_rng(1))
    assert np.array_equal(e1, e2) and e1.shape == pan["c"][0].shape
    e0 = S.world(pan, 0, {**cell, "county_mult": 0.0}, np.random.default_rng(1))
    assert not np.array_equal(e0, e1)
    p = S.PARAMS
    assert 0 < p["share_county"]["value"] < p["share_0_5km"]["value"] < 1


def test_kappa_is_capped_unique_and_taken_out_of_the_county_noise():
    p = S.PARAMS
    sig2 = p["sigma_tot_m"]["value"] ** 2
    k = p["kappa_m"]
    assert k["inputs"]["cap_binds"] and abs(k["value"] ** 2 - p["share_county"]["value"]
                                             * sig2) < 1e-12
    assert abs(k["inputs"]["r2_canal_unique"] - (0.7435338902539914 + 0.001664693967455777)) \
        < 1e-15
    for cell in S.cells():
        v = S.noise_variances(cell)
        assert min(v["county"], v["block"], v["well"]) >= 0
        if v["kappa_sq"] <= p["share_county"]["value"] * v["sigma_sq"] + 1e-12:
            assert abs(v["county"] + v["block"] + v["well"] + v["kappa_sq"]
                       - v["sigma_sq"]) < 1e-9
    raw = next(c for c in S.cells() if c.get("kappa_m") == p["kappa_raw_m"]["value"])
    assert S.noise_variances(raw)["county"] == 0.0
    f = S.ar1_within_factor(p["phi"]["value"], 12)
    assert 0 < f < 1 and abs(f - p["confounder_within_factor"]["value"]) < 1e-15
    assert S.ar1_within_factor(0.0, 12) == pytest.approx(11 / 12)


def test_report_only_cells_declared():
    ro = [c for c in S.cells() if c["report_only"]]
    assert {c.get("kappa_mult") for c in ro} >= {0.5, 0.25}
    assert any(c.get("forcing_error_lambda") == 0.8 for c in ro)
    assert any(c["rho"] == -0.52 for c in ro)
    assert any(c.get("sigma_m") == S.PARAMS["sigma_page_kfold_m"]["value"] for c in ro)
    assert [round(x, 2) for x in S.corr_bound(0.8683200707914109, 0.8771508726475611)] == \
        [0.52, 1.0]


def test_forcing_error_panel_rescales():
    pan = _panel(3)
    fe = S.forcing_error_panel(pan, 0.8)

    def rms(x):
        return float(np.sqrt(np.mean((x - x.mean(axis=1, keepdims=True)) ** 2)))

    for j in range(3):
        assert rms(fe["c"][j]) == pytest.approx(rms(pan["c"][j]))
        assert not np.allclose(fe["c"][j], pan["c"][j])
    assert np.allclose(S.forcing_error_panel(pan, 1.0)["c"], pan["c"])


def test_rehearsal_pin(tmp_path):
    f = tmp_path / "fake.npz"
    f.write_bytes(b"not the rehearsal")
    with pytest.raises(SystemExit, match="pinned rehearsal"):
        S.check_rehearsal(f)
    with pytest.raises(SystemExit, match="missing"):
        S.check_rehearsal(tmp_path / "absent.npz")
    assert S.REHEARSAL_SHA.startswith("6795e125")


def test_gate_record_bound_to_code():
    PR = pytest.importorskip("prereg_n2_reading")
    assert PR.gate_record_problems(None)
    rec = {"ok": True, "harness_sha256": S.gate_code_sha(), "scorer_sha256": S.SCORER_SHA}
    assert PR.gate_record_problems(rec) == []
    bad = {**rec, "harness_sha256": {**rec["harness_sha256"],
                                     f"{S.X}/synth_n2.py": "0" * 64}}
    assert "other code" in PR.gate_record_problems(bad)[0]
    assert PR.gate_record_problems({**rec, "ok": False})


# ----------------------------------------------------------------------------- wording
def _rec(k, rho, p_sup, cov=0.85, p_inf=None, report_only=False, kappa=1.0, county=1.0):
    return {"k": k, "rho": rho, "kappa_mult": kappa, "county_mult": county,
            "report_only": report_only, "p": {"response_supported": p_sup},
            "p_supported_given_informative": p_inf,
            "p_supported_stricter": max(p_sup, p_inf) if p_inf is not None else p_sup,
            "coverage90_of_k": cov}


def _grid(fp=0.05, ws05=0.1, ws2=0.05, cov=0.85, pw=0.9, fp_inf=None):
    out = []
    for k in S.GRID["k"]:
        for r in S.GRID["rho"]:
            p = {0.0: fp if r else 0.0, 0.5: ws05, 2.0: ws2, 1.0: pw}.get(k, 0.3)
            out.append(_rec(k, r, p, cov=cov if k == 1.0 else 0.8,
                            p_inf=fp_inf if (k == 0.0 and r) else None))
    return out


def test_wording_pass_controlled_low_power():
    r = S.reading(_grid(pw=0.79))
    assert r["wording"] == "pass_controlled_low_power" and r["power_floor"] == 0.80
    assert "pass-controlled but low-power" in r["display_text"][0]


def test_wording_false_pass_is_the_max_over_rho():
    g = _grid()
    next(c for c in g if c["k"] == 0.0 and c["rho"] == 0.0)["p"]["response_supported"] = 0.3
    g[[i for i, c in enumerate(g) if c["k"] == 0.0 and c["rho"] == 0.0][0]][
        "p_supported_stricter"] = 0.3
    r = S.reading(g)
    assert r["wording"] == "weak" and r["false_pass_max_over_rho"] == 0.3


def test_wording_texts_carry_power_and_the_forcing_caveat():
    for key in ("weak", "informative", "pass_controlled_low_power", "decisive"):
        assert "{pw:.2f}" in S.WORDING[key]
    for key in ("informative", "pass_controlled_low_power", "decisive"):
        assert "no forcing error" in S.WORDING[key]
    assert "never used the word 'decisive'" in S.READING_SOURCE


def test_wording_decisive():
    r = S.reading(_grid())
    assert r["wording"] == "decisive" and "decisive by design" in r["display_text"][0]
    assert r["display_text"][1] == S.WORDING["verdict_word"]


def test_wording_weak_on_false_pass_including_the_stricter_rate():
    assert S.reading(_grid(fp=0.25))["wording"] == "weak"
    r = S.reading(_grid(fp=0.10, fp_inf=0.30))
    assert r["wording"] == "weak" and r["false_pass_unconditional_rho_m087"] == 0.10
    assert r["false_pass_max_over_rho"] == 0.30


@pytest.mark.parametrize("kw", [{"ws05": 0.25}, {"ws2": 0.4}, {"cov": 0.79}])
def test_wording_informative(kw):
    assert S.reading(_grid(**kw))["wording"] == "informative"


def test_report_only_cells_never_change_the_wording():
    fe = {**_rec(1.0, 0.0, 0.1, report_only=True), "forcing_error_lambda": 0.8}
    g = _grid() + [_rec(0.0, -0.87, 0.99, report_only=True, kappa=2.0),
                   _rec(1.0, 0.0, 0.0, cov=0.0, report_only=True, county=0.0), fe]
    r = S.reading(g)
    assert r["wording"] == "decisive" and r["power_k1_forcing_error_report_only"] == 0.1


# ----------------------------------------------------------------------------- runner
def _register(tmp: Path, seed: int, *, tamper: bool = False, code: dict | None = None) -> Path:
    from harness import prereg as P1

    d = tmp / "prereg"
    d.mkdir(parents=True, exist_ok=True)
    doc = {"id": S.DECL_ID, "design": json.loads(json.dumps(S.design_spec(seed))),
           "frozen_code": code if code is not None else S.frozen_code(),
           "replays": {"replays_json_sha256": "r" * 64}}
    path = d / f"{S.DECL_ID}.json"
    path.write_text(json.dumps(doc))
    sha = S.sha256(path)
    if tamper:
        path.write_text(json.dumps({**doc, "note": "edited"}))
    P1._append_chained(d / "INDEX.jsonl", {"id": S.DECL_ID, "sha256": sha})
    return d


def _panel_file(tmp: Path, replays_sha: str = "r" * 64, fwd: str = "f" * 64) -> Path:
    p = tmp / "panel.npz"
    np.savez(p, **{**_panel(1), "replays_json_sha256": np.str_(replays_sha),
                   "forward_sha256": np.str_(fwd)})
    return p


def _fake_sim(panel, seed):
    return {"exact_plumbing": {"ok": True}, "reading": {"status": "not_estimable",
                                                       "wording": "not_estimable",
                                                       "display_text": ["x", "y"]}}


def _forward_logged(att: Path, fwd: str = "f" * 64):
    S.append_attempt(att, {"event": "forward", "rc": 0, "npz_sha256": fwd})


def test_runner_refuses_unregistered_and_other_ids(tmp_path):
    (tmp_path / "prereg").mkdir()
    with pytest.raises(SystemExit, match="not registered"):
        S.registered_declaration(S.DECL_ID, SEED, tmp_path / "prereg")
    d = _register(tmp_path, SEED)
    with pytest.raises(SystemExit, match="REFUSED"):
        S.registered_declaration("X5.n2c_reading.v1", SEED, d)


def test_runner_refuses_tampered_registration(tmp_path):
    d = _register(tmp_path, SEED, tamper=True)
    with pytest.raises(SystemExit, match="sha256"):
        S.registered_declaration(S.DECL_ID, SEED, d)


def test_runner_refuses_changed_code(tmp_path):
    code = {**S.frozen_code(), S.SCORER: "0" * 64}
    d = _register(tmp_path, SEED, code=code)
    with pytest.raises(SystemExit, match="code changed"):
        S.registered_declaration(S.DECL_ID, SEED, d)


def test_runner_refuses_another_seed(tmp_path):
    d = _register(tmp_path, SEED)
    with pytest.raises(SystemExit, match="registered seed"):
        S.registered_declaration(S.DECL_ID, SEED + 1, d)
    doc, _ = S.registered_declaration(S.DECL_ID, SEED, d)
    assert doc["design"]["seed"] == SEED


def test_runner_refuses_wrong_panel_and_runs_once(tmp_path):
    d = _register(tmp_path, SEED)
    out, att = tmp_path / "out", tmp_path / "out" / "attempts.jsonl"
    kw = {"prereg_dir": d, "attempts": att, "simulate_fn": _fake_sim}
    with pytest.raises(SystemExit, match="registered replays"):
        S.cmd_run(SEED, S.DECL_ID, _panel_file(tmp_path, replays_sha="x" * 64), out, **kw)
    with pytest.raises(SystemExit, match="last successful forward"):
        S.cmd_run(SEED, S.DECL_ID, _panel_file(tmp_path), out, **kw)
    _forward_logged(att)
    assert S.cmd_run(SEED, S.DECL_ID, _panel_file(tmp_path), out, **kw) == 0
    rep = json.loads((out / "report.json").read_text())
    assert rep["seed"] == SEED and rep["id"] == S.DECL_ID
    recs = [json.loads(x) for x in att.read_text().splitlines()]
    assert recs[-1]["event"] == "run" and recs[-1]["rc"] == 0 and recs[-1]["seed"] == SEED
    with pytest.raises(SystemExit, match="run once"):
        S.cmd_run(SEED, S.DECL_ID, _panel_file(tmp_path), out, **kw)
    with pytest.raises(ValueError, match="refused"):
        S.cmd_run(S.SMOKE_SEED, S.DECL_ID, _panel_file(tmp_path), out, **kw)


def test_runner_logs_an_engineering_failure_without_a_report(tmp_path):
    d = _register(tmp_path, SEED)
    out, att = tmp_path / "out", tmp_path / "out" / "attempts.jsonl"
    _forward_logged(att)

    def fail(panel, seed):
        return {"exact_plumbing": {"ok": False}, "engineering_failure": "exact plumbing"}

    rc = S.cmd_run(SEED, S.DECL_ID, _panel_file(tmp_path), out, prereg_dir=d, attempts=att,
                   simulate_fn=fail)
    assert rc == 3 and not (out / "report.json").exists()
    assert json.loads(att.read_text().splitlines()[-1])["rc"] == 3


def test_smoke_refuses_results(tmp_path):
    with pytest.raises(SystemExit, match="results"):
        S.main(["smoke", "--out", str(REPO / "results" / "x5_n2_smoke")])


# ----------------------------------------------------------------------------- draft
@needs_results
def test_draft_declaration_key_fields(monkeypatch):
    from harness import prereg3

    PR = pytest.importorskip("prereg_n2_reading")
    with pytest.raises(ValueError):
        PR.draft_prereg(None)
    monkeypatch.setattr(S, "REAL_REFUSED_SEEDS", ())    # the test seed, for the draft only
    d = PR.draft_prereg(SEED)
    assert prereg3.validate_prereg(d) == []
    assert d["kind"] == "declaration" and d["id"] == S.DECL_ID
    assert d["design"] == S.design_spec(SEED) and d["frozen_code"] == S.frozen_code()
    ap = d["attempt_policy"]
    assert ap["score_once"] is True and ap["exploratory_followups_max"] == 0
    assert d["thresholds"]["reading"]["false_pass_max"] == 0.25
    assert "design_time_evidence" in d["product_use"]["allowed_uses"]
    assert "forcing_recorded_arm" in d["product_use"]["forbidden_uses"]
    assert d["shared_scorer_binding"]["sha256"] == S.SCORER_SHA
    assert any("X1" in s for s in d["disclosures"])
    assert any("U7" in s for s in d["disclosures"])
    assert "metadata only" in d["product_use"]["x1"]["use_made"]
    assert d["design"]["reading"]["power_min"] == 0.80
    assert PR.gate_record_problems(d["thresholds"]["engineering_gate_record"]) == []
    assert PR.runner_problems() == []


@needs_results
def test_published_fields_match_the_reports():
    PR = pytest.importorskip("prereg_n2_reading")
    if not all((REPO / p).exists() for p in PR.CITED):
        pytest.skip("cited reports absent")
    assert PR.published_problems() == []
