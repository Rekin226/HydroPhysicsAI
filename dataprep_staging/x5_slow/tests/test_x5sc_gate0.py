"""CPU tests of the I1.X5SC.gate0 scripts (no calibrate_coupled fit, no GPU, no write outside
pytest's tmp dirs). They read the gitignored results/ tree (theta, q0_nh column JSON) and
the leveling panel's inventory columns, so they live here, outside the repo's testpaths:

    cd <repository root>
    ~/miniforge3/envs/hydro/bin/python -m pytest -q -p no:cacheprovider \\
        dataprep_staging/x5_slow/tests
"""
from __future__ import annotations

import datetime as dt
import fcntl
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
TESTS = Path(__file__).resolve().parent
X = TESTS.parent
sys.path.insert(0, str(X))
sys.path.insert(0, str(TESTS))
import dump_fit_heads as D  # noqa: E402
import e2common as C  # noqa: E402
import e2flags as FL  # noqa: E402
import evaluate_e2 as E  # noqa: E402
import fake_arms as F  # noqa: E402
import jobmeta as J  # noqa: E402
import make_synthetic as M  # noqa: E402

NEEDED = (C.SPEC["theta"], C.SPEC["q0_nh_column"], f"{C.SPEC['data_src']}/{C.SPEC['panel']}")
if not all(C.repo_path(p).exists() for p in NEEDED):
    pytest.skip("needs the gitignored results/ tree and the leveling panel",
                allow_module_level=True)

PAST = "2000-01-01T00:00:00.000000Z"


@pytest.fixture(scope="session")
def heads(tmp_path_factory):
    return F.write_heads_npz(tmp_path_factory.mktemp("heads") / "fit_heads.npz")


@pytest.fixture(scope="session")
def synth(tmp_path_factory, heads):
    out = tmp_path_factory.mktemp("synth")
    assert M.main(["--heads", str(heads), "--out", str(out)]) == 0
    return out


def _truth(synth, t):
    return json.loads((synth / f"truth_{t}" / "truth_column.json").read_text())["zonal"]


# --------------------------------------------------------------------------- generator
def test_inventory_never_reads_elev(monkeypatch):
    import pyarrow.parquet as pq

    seen = []
    real = pq.read_table

    def spy(path, columns=None, **kw):
        seen.append(list(columns) if columns is not None else None)
        return real(path, columns=columns, **kw)

    monkeypatch.setattr(pq, "read_table", spy)
    inv = C.panel_inventory(C.repo_path(C.SPEC["data_src"]) / C.SPEC["panel"])
    assert seen == [C.SPEC["inventory_columns"]]
    assert "elev_m" not in inv.columns
    assert inv["datetime"].min() >= pd.Timestamp("2012-01-01")
    assert inv["datetime"].max() < pd.Timestamp("2022-01-01")


def test_truth_heads_slice_ends_2021(heads):
    H = C.truth_heads(heads, layers=True)
    assert str(H["dates"][-1].date()) == "2021-12-01"
    assert H["h"].shape == (H["grid"].n_active, 120)
    assert H["layers"].shape == (4, H["grid"].n_active, 120)


def test_provenance_is_checked(heads, tmp_path):
    prov = C.heads_provenance(heads)
    assert M.check_provenance(prov)["argv_flags"] == C.SPEC["calibrate_common"]
    bad = dict(prov, argv=D.hindcast_argv("d", "o")[:6] + ["0"] + D.hindcast_argv("d", "o")[7:])
    with pytest.raises(SystemExit):
        M.check_provenance(bad)
    with pytest.raises(SystemExit):
        M.check_provenance(dict(prov, wells_dir="results/dataprep/twin_inputs/clean_heads/v2"))
    z = dict(np.load(heads))
    z.pop("provenance_json")
    np.savez(tmp_path / "noprov.npz", **z)
    with pytest.raises(ValueError):
        C.heads_provenance(tmp_path / "noprov.npz")


def test_panel_schema_and_inventory(synth):
    import pyarrow.parquet as pq

    src = C.repo_path(C.SPEC["data_src"]) / C.SPEC["panel"]
    inv = C.panel_inventory(src)
    for t in ("A", "N", "V"):
        p = synth / f"truth_{t}" / "data" / C.SPEC["panel"]
        assert p.is_file() and not p.is_symlink()
        assert pq.read_schema(p).remove_metadata().equals(pq.read_schema(src).remove_metadata())
        df = pd.read_parquet(p)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values(["sid", "datetime"]).reset_index(drop=True)
        for c in ("sid", "datetime", "x_3826", "y_3826"):
            assert (df[c].to_numpy() == inv[c].to_numpy()).all(), c
        assert df[["lon", "lat", "town", "name"]].isna().all().all()
        assert np.isfinite(df["elev_m"]).all()


def test_mirror_is_symlinks_except_panel(synth):
    src = C.repo_path(C.SPEC["data_src"])
    d = synth / "truth_A" / "data"
    assert sorted(os.listdir(d)) == sorted(os.listdir(src))
    assert sorted(os.listdir(d / "ls_cache")) == sorted(os.listdir(src / "ls_cache"))
    for e in os.listdir(d):
        assert (d / e).is_symlink() or e == "ls_cache"
    for e in os.listdir(d / "ls_cache"):
        assert (d / "ls_cache" / e).is_symlink() == (e != M.PANEL_NAME)


def test_truths_differ_by_their_slow_term_only(synth):
    """Common noise: elev_N - elev_A is exactly the aquitard term at the benchmark cell."""
    man = json.loads((synth / "generator_manifest.json").read_text())
    H = C.truth_heads(man["heads_npz"])
    grid, dates = H["grid"], H["dates"]
    zone = C.zone_map(grid, C.repo_path(C.SPEC["theta"]))

    def read(t):
        df = pd.read_parquet(synth / f"truth_{t}" / "data" / C.SPEC["panel"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df.sort_values(["sid", "datetime"]).reset_index(drop=True)

    a, n = read("A"), read("N")
    cell = np.array([grid.cell_of(x, y) for x, y in zip(a["x_3826"], a["y_3826"],
                                                       strict=True)])
    mon = np.array([C.nearest_month(dates, t) for t in a["datetime"]])
    sel = (cell >= 0) & (zone[np.maximum(cell, 0)] == C.DISTAL)
    aq = C.aquitard_series(man["truths"]["A"]["distal_params"], H["h"][cell[sel]])
    want = aq[np.arange(sel.sum()), mon[sel]]
    np.testing.assert_allclose((n["elev_m"] - a["elev_m"]).to_numpy()[sel], want, atol=1e-9)
    off = (cell >= 0) & (zone[np.maximum(cell, 0)] != C.DISTAL)
    np.testing.assert_allclose((n["elev_m"] - a["elev_m"]).to_numpy()[off], 0.0, atol=1e-12)


def test_truth_calibration_and_selfcheck(synth):
    man = json.loads((synth / "generator_manifest.json").read_text())
    for t in ("A", "V"):
        assert abs(man["truths"][t]["diagnostics"]["bg2016_distal_cm"] - 2.5) < 1e-3
    a = man["truths"]["A"]["distal_params"]
    assert math.isclose(math.exp(a["log_ska"] - a["log_tau_a"]) * 365.25, C.SENS_TRUTH)
    assert 0.0 < a["h_a0"] < 30.0
    assert man["truths"]["V"]["distal_params"]["h_pc0"] > 0.0
    sc = man["selfcheck"]
    assert set(sc["max_abs_diff_m"]) == {"A", "N", "V"}
    assert max(sc["max_abs_diff_m"].values()) <= M.SELFCHECK_TOL_M
    n = man["noise"]
    assert math.isclose(n["per_survey_sd_cm"], n["residual_sd_cm"] / math.sqrt(2.0))
    assert man["heads_provenance_checked"]["wells_dir"].endswith("clean_heads/v1/wells")


def test_calibrate_reads_the_synthetic_panel(synth):
    """leveling_rows' own reader on the synthetic dir: the inventory's benchmarks."""
    from hydrophysics.twin.leveling import load_panel, site_subsidence

    panel = load_panel(str(synth / "truth_A" / "data"))
    obs = site_subsidence(panel, "2012-01-01", "2022-12-01", min_obs=5, max_rate=None)
    assert len(obs) == 888


def test_generator_refuses_to_overwrite(synth, heads):
    with pytest.raises(SystemExit):
        M.main(["--heads", str(heads), "--out", str(synth)])


# --------------------------------------------------------------------------- dump
def test_dump_captures_the_hindcast_and_stops():
    import types

    import torch

    calls = []

    def real(model, scalars, inp, h0, E_, R, gain, every, **kw):
        calls.append((gain, every, kw))
        return torch.ones(4, 3, 5, dtype=torch.float64)

    def main(argv):
        calls.append(argv)
        heads = cc.hindcast_with_nudging("m", "s", "inp", 0, 0, 0, 1.0, 1, datum=1,
                                         apex_from=0, update_method="innovation")
        raise AssertionError(f"main continued past the hindcast with {heads}")

    cc = types.SimpleNamespace(hindcast_with_nudging=real, main=main)
    argv = D.hindcast_argv("DATA", "OUT")
    h, inp, call = D.capture(cc, argv)
    assert h.shape == (4, 3, 5) and h.dtype == np.float64 and inp == "inp"
    assert call == {"gain": 1.0, "every": 1, "update_method": "innovation",
                    "apex_from_given": True, "datum_given": True}
    assert cc.hindcast_with_nudging is real                  # restored
    assert calls[0] == argv
    assert argv[2:2 + len(C.SPEC["calibrate_common"])] == C.SPEC["calibrate_common"]
    cc2 = types.SimpleNamespace(hindcast_with_nudging=real, main=lambda a: None)
    with pytest.raises(SystemExit):
        D.capture(cc2, argv)


@pytest.mark.parametrize("form", list(C.SPEC["forms"]))
def test_calibrate_accepts_the_arm_flags(form, synth, monkeypatch):
    """calibrate_coupled.main parses and validates every arm's exact argv (all its flag
    refusals run before pick_device), stopped at pick_device: no input is loaded."""
    from hydrophysics.twin import calibrate_coupled as cc

    class Stop(Exception):
        pass

    def stop(dev):
        raise Stop(dev)

    saved = {k: v for k, v in vars(cc).items() if k.isupper()}
    monkeypatch.setattr(cc, "pick_device", stop)
    d = json.loads((synth / "generator_manifest.json").read_text())["truths"]["A"]["data_dir"]
    try:
        for o in C.SPEC["origins"]:
            av = C.fit_args(form, o, d, str(synth / "never"))
            assert av[:2] == ["-m", "hydrophysics.twin.calibrate_coupled"]
            with pytest.raises(Stop):
                cc.main(av[2:])
            k = F.constraint_keys(form)
            assert cc.column_constraints() == {key: k[key] for key in cc.column_constraints()}
    finally:
        for key, v in saved.items():
            setattr(cc, key, v)
    assert not (synth / "never").exists()


# --------------------------------------------------------------------------- runner pieces
def test_e2flags_jobs_and_argv(synth, tmp_path):
    lines = FL.jobs(synth)
    assert len(lines) == 12 and len({ln.split("|")[0] for ln in lines}) == 12
    man = json.loads((synth / "generator_manifest.json").read_text())
    for ln in lines:
        a, t, f, o, d = ln.split("|")
        assert d == man["truths"][t]["data_dir"]
        av = FL.argv(a, synth, tmp_path)
        assert av == C.fit_args(f, o, d, str(tmp_path.resolve() / a))
        assert av.count("--data") == 1 and av[av.index("--data") + 1] == d
    forms = [ln.split("|")[2] for ln in lines]
    for k in range(3):                                   # each worker: 2 aq + 2 vt
        assert sorted(forms[k::3]) == ["aq", "aq", "vt", "vt"]


def test_run_e2_sh_syntax_and_dry_run():
    sh = X / "run_e2.sh"
    assert subprocess.run(["bash", "-n", str(sh)]).returncode == 0
    r = subprocess.run(["bash", str(sh)], env={**os.environ, "DRY": "1"},
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    cmds = [ln for ln in r.stdout.splitlines() if ln.startswith("HYDROMIND_GW_DATA=")]
    assert len(cmds) == 12
    for ln in cmds:
        env, rest = ln.split(" ", 1)
        d = env.split("=", 1)[1]
        assert f"--data {d} " in rest and "-m hydrophysics.twin.calibrate_coupled" in rest


def test_guard_refuses_real_data(synth, tmp_path):
    runs = tmp_path / "runs"
    real = str(C.repo_path(C.SPEC["data_src"]))
    o = C.SPEC["origins"][0]
    arm = runs.resolve() / C.arm_name("A", "aq", o)
    # --data on the real dir
    assert F.start_arm(runs, synth, "A", "aq", o, data_dir=real) == J.EXIT_REFUSED
    # HYDROMIND_GW_DATA on the real dir while --data is synthetic
    assert F.start_arm(runs, synth, "A", "aq", o, env=real) == J.EXIT_REFUSED
    # an argv whose --data differs from the guarded dir
    d = json.loads((synth / "generator_manifest.json").read_text())["truths"]["A"]["data_dir"]
    av = C.fit_args("aq", o, real, str(arm))
    assert F.start_arm(runs, synth, "A", "aq", o, argv=av) == J.EXIT_REFUSED
    # a dropped --data
    av = C.fit_args("aq", o, d, str(arm))
    i = av.index("--data")
    assert F.start_arm(runs, synth, "A", "aq", o, argv=av[:i] + av[i + 2:]) == J.EXIT_REFUSED
    rows = J.read_ledger(runs)
    assert [r["event"] for r in rows] == ["refused"] * 4
    assert not (arm / "job.json").exists()


def test_guard_refuses_a_symlinked_panel(synth, tmp_path, monkeypatch):
    import shutil

    man = json.loads((synth / "generator_manifest.json").read_text())
    fake = tmp_path / "synth"
    shutil.copytree(synth, fake, symlinks=True)
    d = fake / "truth_N" / "data"
    p = d / C.SPEC["panel"]
    p.unlink()
    p.symlink_to(C.repo_path(C.SPEC["data_src"]) / C.SPEC["panel"])
    man["truths"]["N"]["data_dir"] = str(d.resolve())
    (fake / "generator_manifest.json").write_text(json.dumps(man))
    monkeypatch.setenv("HYDROMIND_GW_DATA", str(d))
    with pytest.raises(SystemExit, match="regular file"):
        J.guard(tmp_path / "arm", "N", "vt", "2016-04", str(d),
                C.fit_args("vt", "2016-04", str(d.resolve()), str(tmp_path / "arm")), man)


def test_ledger_retry_rule(synth, tmp_path):
    runs = tmp_path / "runs"
    o = C.SPEC["origins"][1]
    tN = _truth(synth, "N")
    F.write_arm(runs, synth, "N", "vt", o, tN, rc=1, output=False)          # crash
    arm = runs.resolve() / C.arm_name("N", "vt", o)
    F.write_arm(runs, synth, "N", "vt", o, tN, rc=0)                        # retry: allowed
    assert (arm / "job.attempt1.json").exists() and (arm / "job.json").exists()
    assert json.loads((arm / "job.json").read_text())["attempt"] == 2
    assert F.start_arm(runs, synth, "N", "vt", o) == J.EXIT_COMPLETED       # never rerun
    st = J.arm_status(J.read_ledger(runs), arm.name)
    assert st == {"attempts": 2, "completed": [2], "open": [], "last": 2}
    lines = (runs / C.LEDGER).read_text().splitlines()
    (runs / C.LEDGER).write_text("\n".join([lines[0].replace('"start"', '"Start"')] + lines[1:])
                                 + "\n")
    with pytest.raises(SystemExit, match="hash chain"):
        J.read_ledger(runs)


def test_redaction():
    import io

    src = io.StringIO("leveling r2 0.61 rings +0.123 and rings_x nan\nrings_independent_r2 ...\n"
                      "row 1 0.003\n")
    dst = io.StringIO()
    J.redact_stream(src, dst)
    out = dst.getvalue()
    assert "0.123" not in out and "0.003" not in out and "0.61" in out


# --------------------------------------------------------------------------- evaluator
def _oracle_runs(synth, runs, collapse_a=False, bad_withhold=False):
    """aq: the truth's own distal term (A), a floored term (N), a weak term (V);
    vt: tau on its 3650 d floor (A, N), the truth V column (V)."""
    tA, tN, tV = _truth(synth, "A"), _truth(synth, "N"), _truth(synth, "V")
    floor = F.aq(1e-4, 10.0, 0.0)
    for o in C.SPEC["origins"]:
        a = [dict(s) for s in tA]
        a[1].update(floor)
        if collapse_a:
            a[2]["h_a0"] = 0.0
        F.write_arm(runs, synth, "A", "aq", o, a,
                    withhold="2016-05:2022-12" if bad_withhold and o == "2019-04" else None)
        n = [dict(s) for s in tN]
        n[1].update(floor)
        n[2].update(floor)
        F.write_arm(runs, synth, "N", "aq", o, n)
        v = [dict(s) for s in tN]
        v[1].update(floor)
        v[2].update(F.aq(0.02, 20.0, 30.0))
        F.write_arm(runs, synth, "V", "aq", o, v)
        for t in ("A", "N"):
            s = [dict(x) for x in tN]
            s[2].update(log_tau=math.log(3650.0), h_pc0=0.0)
            F.write_arm(runs, synth, t, "vt", o, s)
        F.write_arm(runs, synth, "V", "vt", o, [dict(x) for x in tV])


def _decl(tmp_path, runs, synth, created=PAST):
    return str(F.fake_declaration(tmp_path / "decl.json", runs, synth, created))


def test_evaluator_oracle(synth, tmp_path):
    runs = tmp_path / "runs"
    _oracle_runs(synth, runs)
    decl = _decl(tmp_path, runs, synth)
    args = ["--runs", str(runs), "--synth", str(synth), "--declaration", decl]
    assert E.main(args) == 0
    out = runs / "e2_report.json"
    rep = json.loads(out.read_text())
    assert "123.0" not in out.read_text()                   # rings R2 never reported
    aq, vt = rep["forms"]["aq"], rep["forms"]["vt"]
    assert aq["G0-A1"] and aq["G0-A2"] and aq["G0-A3"] and aq["G0-N"] and aq["G0-V"]
    assert aq["eligible_for_b"] is True
    m = rep["arms"]["A_aq_r2016"]["metrics"]
    assert abs(m["background_ratio"] - 1.0) < 1e-9 and abs(m["sens_ratio"] - 1.0) < 1e-9
    assert m["next_year_with_truth_heads"]["fit_vs_truth_distal"]["rmse"] < 1e-9
    assert vt["G0-A3"] is False and vt["eligible_for_b"] is False      # tau on its floor
    assert rep["arms"]["V_vt_r2019"]["metrics"]["background_ratio"] == pytest.approx(1.0)
    assert rep["verdict"]["eligible_forms_for_I1.X5SC"] == ["aq"]
    assert rep["verdict"]["b_primary"] == "aq"
    assert rep["verdict"]["aquitard_level_sensitivity_claimable"] is True
    rows = J.read_ledger(runs)
    assert rows[-1]["event"] == "evaluated"
    assert rows[-1]["report_sha256"] == C.file_sha256(out)
    with pytest.raises(SystemExit):                          # scored once
        E.main(args)
    out.unlink()                                             # even without the report
    with pytest.raises(SystemExit, match="evaluation"):
        E.main(args)
    with pytest.raises(SystemExit, match="evaluated"):       # no arm may run after scoring
        J.preflight(runs, synth, decl)


def test_evaluator_collapsed_and_nonconforming(synth, tmp_path):
    runs = tmp_path / "runs"
    _oracle_runs(synth, runs, collapse_a=True)
    rep = E.evaluate(runs, synth, json.loads(Path(_decl(tmp_path, runs, synth)).read_text()))
    aq = rep["forms"]["aq"]
    assert aq["G0-A1"] is False and aq["G0-A3"] is False and aq["eligible_for_b"] is False
    runs2 = tmp_path / "runs2"
    _oracle_runs(synth, runs2, bad_withhold=True)
    d2 = F.fake_declaration(tmp_path / "decl2.json", runs2, synth, PAST)
    rep2 = E.evaluate(runs2, synth, json.loads(d2.read_text()))
    assert rep2["arms"]["A_aq_r2019"]["status"] == "non-conforming"
    assert rep2["forms"]["aq"]["G0-A1"] is None
    assert rep2["forms"]["aq"]["eligible_for_b"] is False
    assert "G0-A1" in rep2["forms"]["aq"]["not_evaluable"]


def test_evaluator_refusals(synth, tmp_path):
    runs = tmp_path / "runs"
    tN = _truth(synth, "N")
    o = C.SPEC["origins"][0]
    F.write_arm(runs, synth, "N", "vt", o, tN)
    decl = json.loads(Path(_decl(tmp_path, runs, synth)).read_text())
    with pytest.raises(SystemExit, match="never started"):
        E.readiness(runs, synth, decl)
    assert F.start_arm(runs, synth, "N", "aq", o) == 0             # an open attempt
    with pytest.raises(SystemExit, match="has not ended"):
        E.readiness(runs, synth, decl)
    with open(runs / E.LOCK_NAME, "w") as fh:                       # the runner's lock
        fcntl.flock(fh, fcntl.LOCK_EX)
        with pytest.raises(SystemExit, match="lock"):
            E.readiness(runs, synth, decl)
    other = dict(decl, frozen=dict(decl["frozen"], runs_dir=str(tmp_path)))
    with pytest.raises(SystemExit, match="runs dir"):
        E.readiness(runs, synth, other)


def test_arm_before_registration_does_not_conform(synth, tmp_path):
    runs = tmp_path / "runs"
    _oracle_runs(synth, runs)
    future = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=1)).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ")
    decl = json.loads(Path(_decl(tmp_path, runs, synth, created=future)).read_text())
    rep = E.evaluate(runs, synth, decl)
    assert all(r["status"] == "non-conforming" for r in rep["arms"].values())
    assert rep["verdict"]["b_runs"] is False


def test_at_bound_tolerance():
    p = {"zonal": [{}, {}, {"log_ska": math.log(1e-4) + 0.05, "log_tau_a": math.log(7305.0),
                             "h_a0": 0.2}],
         "aquitard_bounds": {"log_ska": [math.log(1e-4), 0.0],
                             "log_tau_a": [math.log(3652.5), math.log(36525.0)],
                             "h_a0": [0.0, 30.0]}}
    b = E.bound_flags(p, "aq", 0.01)
    assert b["log_ska"]["at_bound"] and b["h_a0"]["at_bound"]
    assert not b["log_tau_a"]["at_bound"]
    q = {"zonal": [{}, {}, {"log_tau": math.log(3650.0) + 0.03}], "tau_min_days": None,
         "tau_min_days_zone": {"distal": 3650.0}, "tau_ceiling_days": 36525.0}
    assert not E.bound_flags(q, "vt", 0.01)["log_tau"]["at_bound"]
    q["zonal"][2]["log_tau"] = math.log(3650.0) + 0.02
    assert E.bound_flags(q, "vt", 0.01)["log_tau"]["at_bound"]


def test_fake_constraint_keys_match_calibrate():
    k = F.constraint_keys("aq")
    assert k["creep_zones"] == "mid+distal" and k["aquitard_ha0_min"] == 0.0
    assert k["aquitard_bounds"]["log_tau_a"][0] == pytest.approx(math.log(3652.5))
    v = F.constraint_keys("vt")
    assert v["tau_min_days_zone"] == {"distal": 3650.0}
    assert v["tau_ceiling_days"] == pytest.approx(100 * 365.25)


# --------------------------------------------------------------------------- declaration
def test_declaration_validates(synth, tmp_path):
    import prereg_x5sc_gate0 as P
    from harness import prereg3

    d = P.doc(frozen=J.frozen_values(synth))
    assert prereg3.validate_prereg(d) == []
    assert d["id"] == C.DECL_ID and d["kind"] == "declaration"
    assert "HAS been fitted on real data" in d["no_prior_look_statement"]
    assert P.AQ_COLUMN_2026_10 in d["no_prior_look_statement"]
    assert "aqd" not in json.dumps(d["arms"]) and len(d["arms"]) == 12
    assert d["frozen"]["report"] == C.REPORT


def test_no_user_paths_in_committed_files():
    files = [p for p in X.rglob("*") if p.is_file() and p.suffix in (".py", ".sh")
             and "__pycache__" not in p.parts]
    assert len(files) >= 10
    home = str(Path.home())
    homes = "/" + "home" + "/"
    for p in files:
        text = p.read_text(encoding="utf-8")
        assert home not in text, p
        assert homes not in text, p
