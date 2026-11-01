"""CPU tests of the X5.n2c_reading.v1 scripts: synthetic panels and fake forward files only (no
model output or observation is read), nothing registered in the real registry (round trips use
pytest's tmp dirs), no look logged.

    cd <repository root>
    ~/miniforge3/envs/hydro/bin/python -m pytest -q -p no:cacheprovider \\
        dataprep_staging/n2c_reading/tests
"""
from __future__ import annotations

import json
import os
import re
import subprocess
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
import prereg_n2c_reading as P  # noqa: E402
import synth_n2c as S  # noqa: E402

DECL = REPO / f"results/dataprep/preregistration/{S.DECL_ID}.json"


# ----------------------------------------------------------------------------- static
def test_no_home_paths_in_staged_files():
    pat = re.compile(r"(/home/|-home-)[A-Za-z0-9_]")
    for p in sorted(X.rglob("*")):
        if p.is_file() and p.suffix in (".py", ".sh", ".json"):
            assert not pat.search(p.read_text()), p


def test_runner_syntax_and_dry_run():
    subprocess.run(["bash", "-n", str(X / "run_n2c_reading.sh")], check=True)
    out = subprocess.run(["bash", str(X / "run_n2c_reading.sh"), "forward"], cwd=REPO,
                         env={**os.environ, "DRY": "1"}, capture_output=True, text=True,
                         check=True).stdout.splitlines()
    assert out == [P.command()]
    assert "--device cpu" in out[0] and out[0].count("--recorded-pumping") == 7


@pytest.mark.skipif(DECL.exists(), reason="declaration registered")
@pytest.mark.parametrize("cmd", ["check", "forward", "extract", "run"])
def test_runner_refuses_without_registration(cmd):
    p = subprocess.run(["bash", str(X / "run_n2c_reading.sh"), cmd], cwd=REPO,
                       capture_output=True, text=True)
    assert p.returncode == 1 and f"declare {S.DECL_ID} first" in p.stderr


def test_declaration_validates_and_binds_the_scorer():
    from harness import prereg3

    d = P.doc()
    assert prereg3.validate_prereg(d) == []
    assert d["design"] == S.design_spec() and d["code_sha256"] == S.code_sha256()
    b = d["shared_scorer_binding"]
    assert b["sha256"] == S.BS.file_sha256() and b["version"] == S.BS.VERSION
    assert any("same_session" in s for s in b["binds"])
    assert P.runner_problems() == []
    assert P.scorer_problems() == [] or not (REPO / "results/dataprep/preregistration").exists()
    assert "never" in d["evidence_and_ledger_mapping"]
    assert d["thresholds"]["reading"]["false_pass_max"] == 0.25
    assert any("X1" in s for s in d["rule_risks_disclosure"])


def test_published_fields_match_the_reports():
    if not (REPO / "results/dataprep/reports/I1_changhua_local/report.json").exists():
        pytest.skip("reports not present (gitignored results tree)")
    assert P.published_problems() == []


def test_kappa_derivation():
    k = S.PARAMS["kappa_m"]
    i = k["inputs"]
    assert k["value"] == pytest.approx(np.sqrt(i["canal_r2"]) * i["rmse_r0_m"]
                                       / np.sqrt(1 - i["r2_r0"]))
    assert 0.29 < k["value"] < 0.31


def test_registered_round_trip_in_tmp_registry(tmp_path, monkeypatch):
    from harness import prereg3

    d = P.doc()
    d["inputs"] = [x for x in d["inputs"]
                   if (REPO / (x["path"] if isinstance(x, dict) else x)).exists()]
    reg = tmp_path / "prereg"
    _, h = prereg3.write_prereg(d, prereg_dir=reg, rows_dir=tmp_path / "rows", registries=[])
    got, sha = S.registered_declaration(reg)
    assert sha == h and got["shared_scorer_binding"]["sha256"] == S.BS.file_sha256()
    monkeypatch.setattr(S, "code_sha256", lambda: {**d["code_sha256"],
                                                   S.CODE_FILES[2]: "0" * 64})
    with pytest.raises(SystemExit, match="code changed"):
        S.registered_declaration(reg)
    monkeypatch.undo()
    monkeypatch.setattr(S, "REPLICATES", 400)
    with pytest.raises(SystemExit, match="design differs"):
        S.registered_declaration(reg)


def test_run_refuses_without_registration(tmp_path):
    with pytest.raises(SystemExit, match="not registered"):
        S.registered_declaration(tmp_path)


# ----------------------------------------------------------------------------- worlds
def test_confounder_has_the_declared_correlation_and_scale():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.1, 12)
    for rho in (0.0, -0.83):
        g = S.confounder(rng, a, rho)
        assert g.std() == pytest.approx(1.0) and abs(g.mean()) < 1e-12
        assert np.corrcoef(g, a)[0, 1] == pytest.approx(rho, abs=1e-12)


def test_ar1_is_stationary():
    x = S.ar1(np.random.default_rng(1), 20000, 12, 2.0, 0.86)
    assert x[:, 0].std() == pytest.approx(2.0, rel=0.03)
    assert x[:, -1].std() == pytest.approx(2.0, rel=0.03)
    assert np.corrcoef(x[:, 5], x[:, 6])[0, 1] == pytest.approx(0.86, abs=0.02)


def test_confounded_world_biases_beta_up():
    pan = S.synthetic_panel(3)
    rng = np.random.default_rng(4)
    b = [S.score_world(pan, j % 7, S.world(pan, j % 7, 0.0, -0.83, rng, kappa_mult=3.0,
                                           noise_scale=0.0))["beta_response"]["beta"]
         for j in range(7)]
    assert np.median(b) > 0.5


def test_exact_plumbing_and_simulate_is_deterministic():
    pan = S.synthetic_panel(5)
    assert S.exact_plumbing(pan)["ok"]
    r1 = S.simulate(pan, replicates=14, seed=9, log=lambda m: None)
    r2 = S.simulate(pan, replicates=14, seed=9, log=lambda m: None)
    assert r1["cells"] == r2["cells"] and r1["reading"] == r2["reading"]
    assert len(r1["cells"]) == 12 + 2
    for c in r1["cells"]:
        assert sum(c["p"].values()) == pytest.approx(1.0)
    assert "power" in r1["reading"]["display_keys"]


def test_reading_rules():
    def cell(k, rho, sup, cov=0.9):
        return {"k": k, "rho": rho, "kappa_mult": 1.0, "report_only": False,
                "p": {"response_supported": sup}, "coverage90_of_k": cov}

    base = [cell(k, r, 0.1) for k in S.GRID["k"] for r in S.GRID["rho"]]
    assert S.reading(base)["display_keys"] == ["power"]
    fp = [cell(k, r, 0.3 if (k == 0 and r == -0.83) else 0.1)
          for k in S.GRID["k"] for r in S.GRID["rho"]]
    assert S.reading(fp)["display_keys"] == ["reported_not_shown", "power"]
    ws = [cell(k, r, 0.25 if (k == 2.0 and r == -0.83) else 0.1)
          for k in S.GRID["k"] for r in S.GRID["rho"]]
    assert S.reading(ws)["display_keys"] == ["rates_printed", "power"]
    cv = [cell(k, r, 0.1, cov=0.79 if (k == 1.0 and r == 0.0) else 0.9)
          for k in S.GRID["k"] for r in S.GRID["rho"]]
    assert "rates_printed" in S.reading(cv)["display_keys"]


def test_reading_uses_the_stricter_informative_conditional_rate():
    def cell(k, rho, sup, sup_inf=None, cov=0.9):
        c = {"k": k, "rho": rho, "kappa_mult": 1.0, "report_only": False,
             "p": {"response_supported": sup}, "coverage90_of_k": cov,
             "p_supported_given_informative": sup_inf}
        c["p_supported_stricter"] = max(sup, sup_inf) if sup_inf is not None else sup
        return c

    # 3 of 7 replays informative: unconditional 0.15 < 0.25 <= conditional 0.35
    cs = [cell(k, r, 0.15 if (k == 0 and r == -0.83) else 0.05,
               0.35 if (k == 0 and r == -0.83) else 0.1)
          for k in S.GRID["k"] for r in S.GRID["rho"]]
    rd = S.reading(cs)
    assert "reported_not_shown" in rd["display_keys"]
    assert rd["false_pass_unconditional"] == 0.15 and rd["false_pass_given_informative"] == 0.35
    assert rd["false_pass_k0_rho_m083"] == 0.35


def test_run_cell_reports_both_rates():
    pan = S.synthetic_panel(6)
    c = S.run_cell(pan, {"k": 1.0, "rho": 0.0, "kappa_mult": 1.0, "report_only": False},
                   np.random.default_rng(0), 14)
    assert 0.0 <= c["p_informative"] <= 1.0
    if c["p_supported_given_informative"] is not None:
        assert c["p_supported_stricter"] >= c["p"]["response_supported"]


def test_plumbing_cell_failure_is_a_terminal_reading(monkeypatch):
    pan = S.synthetic_panel(5)
    r = S.simulate(pan, replicates=7, seed=9, log=lambda m: None)
    pc = r["plumbing_cell"]
    assert pc["ok"] and pc["noise_scale"] == S.PLUMBING_NOISE_SCALE and pc["kappa_mult"] == 0.0
    assert pc["replicates"] == S.PLUMBING_REPLICATES
    assert abs(pc["median_beta"] - 1.0) < 0.05                   # low noise: plumbing only
    monkeypatch.setattr(S, "PLUMBING_BAND", (5.0, 6.0))
    r = S.simulate(pan, replicates=7, seed=9, log=lambda m: None)
    assert "engineering_failure" not in r and "cells" not in r
    assert r["reading"]["status"] == "not_estimable"
    assert r["reading"]["display_keys"] == ["not_estimable"]
    assert "not estimable" in r["reading"]["display_text"][0]
    assert S.render_md({**r, "created": "t", "scorer": S.BS.describe()}).count("terminal") == 1


def test_exact_plumbing_failure_writes_no_panel_summary(monkeypatch):
    pan = S.synthetic_panel(5)
    monkeypatch.setattr(S, "exact_plumbing", lambda panel: {"ok": False, "per_replay": {}})
    r = S.simulate(pan, replicates=7, seed=9, log=lambda m: None)
    assert "engineering_failure" in r and "replay_amplitudes" not in r


def test_smoke_refuses_the_registered_tree():
    with pytest.raises(SystemExit, match="must not go under"):
        S.main(["smoke", "--out", str(REPO / "results/dataprep/reports/x")])


# ----------------------------------------------------------------------------- extract
def test_extract_on_fake_files(tmp_path, monkeypatch):
    sc = S.changhua_screens()
    names = ["baseline"] + [f"RP{y}" for y in S.REPLAY_YEARS]
    hm = np.zeros((len(names), 4, 2148, 252), np.float32)
    rep = {"replays": {}}
    for i, y in enumerate(S.REPLAY_YEARS):
        hm[i + 1][sc["layer"], sc["cell"], 178:190] = 0.01 * (i + 1)
        rep["replays"][f"rp{y}"] = {"file": f"rp{y}.npz", "sha256": f"s{y}",
                                    "window": {"a_changhua": [0.01] * 12}}
    dates = np.array([f"{y}-{m:02d}-01" for y in range(2012, 2033) for m in range(1, 13)])
    fo = {**S.FWD_OPTIONS,
          "recorded_pumping": {f"RP{y}": {"sha256": f"s{y}"} for y in S.REPLAY_YEARS}}
    fwd = tmp_path / "f.npz"

    def save(fo, label="stage3_datum_gate_ccz"):
        np.savez(fwd, scenario_names=np.array(names), dates=dates, heads_mean=hm,
                 forward_options=np.str_(json.dumps(fo)), origin=np.int64(131),
                 horizon=np.int64(120), member_labels=np.array([label]),
                 gain=np.float64(1.0), hindcast_gain=np.float64(0.0))

    save(fo)
    q = np.zeros((len(names), 2, 2148, 21), np.int16)
    for path in (tmp_path / "f.members.npz", tmp_path / "arch.members.npz"):
        np.savez(path, scenario_names=np.array(names),
                 head_member=np.array(["stage3_datum_gate_ccz"] * 2), head_ic=np.array([0, 1]),
                 headL2_members_yr=q, head_scale_m=np.float64(0.01))
    rj = tmp_path / "replays.json"
    rj.write_text(json.dumps(rep))
    xy = tmp_path / "wells.csv"
    xy.write_text("sid,layer,x,y\n" + "".join(f"{s},1,{190000 + 3000 * i},{2650000}\n"
                                              for i, s in enumerate(sc["sid"])))
    monkeypatch.setattr(S, "ARCHIVE_MEMBERS", str(tmp_path / "arch.members.npz"))
    monkeypatch.setattr(S, "WELLS_XY", str(xy))
    info = S.extract(fwd, rj, tmp_path / "panel.npz")
    assert info["shape_c"] == [7, 74, 12] and info["reproduction"]["ok"]
    pan = S.load_panel(tmp_path / "panel.npz")
    assert np.allclose(pan["c"][2], 0.03)
    save(fo, label="stage3_datum_gate_chc")
    with pytest.raises(SystemExit, match="member_labels"):
        S.extract(fwd, rj, tmp_path / "panel2.npz")
    save({**fo, "column_heads": "restart"})
    with pytest.raises(SystemExit, match="forward_options"):
        S.extract(fwd, rj, tmp_path / "panel2.npz")
    fo["recorded_pumping"]["RP2015"]["sha256"] = "other"
    save(fo)
    with pytest.raises(SystemExit, match="RP2015"):
        S.extract(fwd, rj, tmp_path / "panel2.npz")
