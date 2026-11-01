"""CPU tests of FH.forcing_sensitivity.v1 (round-7 rank 7): no forward run, no GPU, nothing
registered, no look logged, nothing written outside pytest's tmp dirs. Every forward output
here is SYNTHETIC (the forbidden keys hindcast_r2 and gate are present in every fake and must
never be loaded): a free-run hindcast whose 2020-22 months depart from the climatology run by
a known growing F, and a restart run whose difference M decays at a known rate. The tmp
registry round trip hashes the gitignored results/ files and skips when they are absent.

    cd <repository root>
    ~/miniforge3/envs/hydro/bin/python -m pytest -q -p no:cacheprovider \\
        --basetemp=<scratch dir> dataprep_staging/fm_origin_memory/tests
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

TESTS = Path(__file__).resolve().parent
X = TESTS.parent
REPO = X.parents[1]
sys.path.insert(0, str(X))
sys.path.insert(0, str(TESTS))
sys.path.insert(0, str(REPO / "dataprep_staging"))
import fm_common as C  # noqa: E402
import prereg_fh as P  # noqa: E402
import score_fh as S  # noqa: E402
from test_fm_origin_memory import base_heads, fake_npz  # noqa: E402

DECL = REPO / f"results/dataprep/preregistration/{S.PREREG_ID}.json"
N_NEW = 216                                       # 2012-01 .. 2029-12
needs_results = pytest.mark.skipif(
    not all((REPO / p).exists() for p in P.INPUT_SHA256),
    reason="results/ inputs (FS forwards, thetas, columns, forward inputs) not present")


def lead_curve(fn, n_t: int, i0: int = S.I0) -> np.ndarray:
    """(n_t,) fn(L) at lead L = t - i0 >= 1, zero before."""
    t = np.arange(n_t)
    return np.where(t - i0 >= 1, fn(np.maximum(t - i0, 1).astype(np.float64)), 0.0)


def fake_new(path: Path, arm: str, hm: np.ndarray, *, fo_over=None, gain=None,
             drop=()) -> Path:
    fo = {**S.SPEC[arm]["fo"], "hpc0_guard_days": [365.0, 365.0], **(fo_over or {})}
    fo = {k: v for k, v in fo.items() if v is not None}
    fake_npz(path, S.NEW_SCEN, hm, origin=S.I0)
    z = dict(np.load(path, allow_pickle=False))
    z.update(horizon=120, gain=S.SPEC[arm]["gain"] if gain is None else gain,
             hindcast_gain=0.0, n_members=36, member_labels=np.array(C.PAGE_MEMBERS),
             forward_options=json.dumps(fo), scenarios=np.array(S.NEW_SCEN_DESC))
    for k in drop:
        z.pop(k)
    np.savez(path, **z)
    return path


F_SLOPE, M_AMP, M_TAU = 0.02, 0.5, 4.0


def world(t: Path, f_slope=F_SLOPE, m_amp=M_AMP, m_tau=M_TAU, c19_jitter=0.0) -> dict:
    fs = base_heads(5)                                       # 2012-01..2032-12
    fcur = lead_curve(lambda L: f_slope * L, 252)
    fcur[S.I0 + S.N_F + 1:] = 0.0                            # only the 2020-22 hindcast months
    s1 = fs + fcur[None, None, None, :]
    s0 = s1 + lead_curve(lambda L: 0.7 * np.exp(-(L - 1) / 3.5), 252, S.I0_FS)[None, None,
                                                                               None, :]
    c19 = base_heads(1, N_NEW)
    if c19_jitter:
        c19[0, C.LAYERS[3], C.CELLS[3], 40] += c19_jitter
    n19 = c19 + lead_curve(lambda L: m_amp * np.exp(-(L - 1) / m_tau), N_NEW)[None, None,
                                                                              None, :]
    return {"S1": fake_npz(t / "S1.npz", S.FS_SCEN, s1, origin=S.I0_FS),
            "S0": fake_npz(t / "S0.npz", S.FS_SCEN, s0, origin=S.I0_FS),
            "C19": fake_new(t / "C19.npz", "C19", c19),
            "N19": fake_new(t / "N19.npz", "N19", n19)}


def expected_lstar(f_slope=F_SLOPE, m_amp=M_AMP, m_tau=M_TAU, n=36):
    for L in range(1, n + 1):
        if f_slope * L > m_amp * math.exp(-(L - 1) / m_tau):
            return L
    return None


def attempts(path: Path, files: dict, codes=("4c2723e", "4c2723e"), rc=0) -> Path:
    with path.open("w") as fh:
        for arm, code in zip(S.ARMS, codes, strict=True):
            fh.write(json.dumps({"event": "forward", "attempt": 1, "arm": arm, "rc": rc,
                                 "code": code, "npz_sha256": C.sha256(files[arm])}) + "\n")
    return path


@pytest.fixture
def registered(monkeypatch):
    monkeypatch.setattr(C, "load_registered", lambda pid, *a, **k: (
        {"frozen_code": {p: C.sha256(REPO / p) for p in S.FROZEN},
         "input_digests": P.INPUT_DIGESTS}, "f" * 64))
    monkeypatch.setattr(C, "check_sha", lambda path, want, what: C.sha256(path))
    d = P.INPUT_DIGESTS[0]
    monkeypatch.setattr(S, "dir_digest", lambda path, pattern: (d["n_files"], d["sha256"]))


def run_main(f: dict, att: Path, out: Path | None) -> int:
    a = [f"--c19={f['C19']}", f"--n19={f['N19']}", f"--fs-s1={f['S1']}",
         f"--fs-s0={f['S0']}", f"--attempts={att}"]
    return S.main(a + ([f"--out={out}"] if out else []))


# ----------------------------------------------------------------------------- static
def test_declaration_validates_and_binds():
    from harness import prereg3

    sys.path.insert(0, str(REPO / "dataprep_staging/fs_start"))
    import prereg_fs as FSP

    d = P.doc()
    assert prereg3.validate_prereg(d) == []
    assert P.runner_problems() == [] and P.scorer_problems() == []
    assert d["kind"] == "declaration" and d["id"] == "FH.forcing_sensitivity.v1"
    assert set(d["frozen_code"]) == set(S.FROZEN)
    declared = {r["path"]: r["sha256"] for r in d["inputs"] if isinstance(r, dict)}
    assert {p: declared[p] for p in S.FROZEN} == d["frozen_code"]
    # the forward inputs, thetas and columns are FS's, the FS files are FS's scored ones
    assert P.FORWARD_INPUTS == FSP.FORWARD_INPUTS and P.INPUT_DIGESTS == FSP.INPUT_DIGESTS
    assert all(FSP.INPUT_SHA256[p] == h for p, h in P.THETA_INPUTS.items())
    assert (P.RF, P.RF_SHA256, P.COMMON_SH, P.VEP) == (FSP.RF, FSP.RF_SHA256, FSP.COMMON_SH,
                                                       FSP.VEP)
    assert P.ARM_FLAGS["C19"] == FSP.ARM_FLAGS["S1"] and P.ARM_FLAGS["N19"] == FSP.ARM_FLAGS["S0"]
    assert FSP.CODES[0] == P.CODE
    assert all(declared[p] == h for p, h in S.FS_FILES.values())
    for a in S.ARMS:
        cmd = d["recipe"][f"forward_{a}"]
        assert "--record-end 2019-12" in cmd
        assert cmd.startswith(f"env HYDROMIND_GW_DATA={P.NOGW} PYTHONPATH=")
        assert "-u" not in cmd.split()[:3] and P.NOGW.startswith(P.LOGDIR + "/")
        assert "--scenario" not in cmd and "--data" not in cmd.split()
    th = d["thresholds"]
    assert "p90_F(L) > p90_M(L)" in th["lstar_force"]
    assert "< 0.001 m" in " ".join(th["engineering_checks_before_F_or_M_is_computed"])
    assert S.E1_TOL_M == 1e-3 and S.E1_TOL_M * 50 <= C.FLOOR_M
    assert "gw_data_env" in d["disclosed"] and "0.85" in d["power"]
    assert th["wordings"] == S.WORDINGS
    assert all("this model" in w and "2020-22" in w for k, w in S.WORDINGS.items()
               if k != "not_evaluable")
    assert "SEALED" in d["attempt_policy"]["values_written_before_scoring"]


def test_runner_syntax_and_dry_run():
    subprocess.run(["bash", "-n", str(X / "run_fh.sh")], check=True)
    for a in S.ARMS:
        out = subprocess.run(["bash", str(X / "run_fh.sh"), "forward", a], cwd=REPO,
                             env={**os.environ, "DRY": "1"}, capture_output=True, text=True,
                             check=True).stdout.splitlines()
        assert out == [P.command(a)]
    out = subprocess.run(["bash", str(X / "run_fh.sh"), "forward", "arms"], cwd=REPO,
                         env={**os.environ, "DRY": "1"}, capture_output=True, text=True,
                         check=True).stdout.splitlines()
    assert out == [P.command(a) for a in S.ARMS]


@pytest.mark.skipif(DECL.exists(), reason="declaration registered; refusal test not applicable")
@pytest.mark.parametrize("cmd", [["check"], ["forward", "C19"], ["score"],
                                 ["archive", "x"]])
def test_runner_refuses_without_registration(cmd):
    log = REPO / "results/twin_runs/fh_force"
    before = sorted(log.rglob("*")) if log.exists() else None
    p = subprocess.run(["bash", str(X / "run_fh.sh"), *cmd], cwd=REPO, capture_output=True,
                       text=True)
    assert (sorted(log.rglob("*")) if log.exists() else None) == before   # nothing written
    assert p.returncode == 1 and "declare FH.forcing_sensitivity.v1 first" in p.stderr


def _parse(cmds: dict, src: str | None, tmp_path: Path) -> dict:
    code = r"""
import argparse, json, shlex, sys
import hydrophysics.twin.forward as F
class Stop(Exception):
    pass
got = {}
orig = argparse.ArgumentParser.parse_args
def parse(self, a=None, ns=None):
    got["ns"] = orig(self, a, ns)
    return got["ns"]
def stop(*_a, **_k):
    raise Stop
argparse.ArgumentParser.parse_args = parse
F.set_compile_matvec = stop
out = {"file": F.__file__}
for arm, cmd in json.loads(sys.argv[1]).items():
    toks = shlex.split(cmd)
    try:
        F.main(toks[toks.index("hydrophysics.twin.forward") + 1:])
    except Stop:
        pass
    ns = got.pop("ns")
    out[arm] = [len(ns.theta), ns.gain, ns.update_method, ns.restart_taper_km, ns.ic_members,
                ns.column_heads, ns.apex_hold, ns.rf_timeseries, len(ns.scenario),
                ns.record_end, ns.climatology_end, ns.data, ns.horizon, ns.hindcast_gain]
print(json.dumps(out))
"""
    env = {**os.environ, "PYTHONPATH": src or str(REPO)}
    p = subprocess.run([sys.executable, "-P", "-c", code, json.dumps(cmds)], cwd=tmp_path,
                       env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-2000:]
    return json.loads(p.stdout.strip().splitlines()[-1])


def _want(arm):
    return [3, S.SPEC[arm]["gain"], "innovation", 5.0, 1, "free", "calibrated", P.RF, 0,
            "2019-12", None, None, 120, 0.0]


def test_forward_parser_accepts_every_arm(tmp_path):
    got = _parse({a: P.command(a) for a in S.ARMS}, None, tmp_path)
    for a in S.ARMS:
        assert got[a] == _want(a), (a, got[a])


@pytest.fixture(scope="module")
def declared_src(tmp_path_factory) -> Path:
    """The declared code's (4c2723e) hydrophysics package, extracted with git archive."""
    if subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", f"{P.CODE}^{{commit}}"],
                      capture_output=True).returncode != 0:
        pytest.skip(f"commit {P.CODE} not in this clone")
    d = tmp_path_factory.mktemp("declared")
    tar = d / "pkg.tar"
    subprocess.run(["git", "-C", str(REPO), "archive", "-o", str(tar), P.CODE, "hydrophysics"],
                   check=True)
    (d / "src").mkdir()
    subprocess.run(["tar", "-xf", str(tar), "-C", str(d / "src")], check=True)
    return d / "src"


def test_declared_code_parses_every_arm(tmp_path, declared_src):
    """The declared code (4c2723e) parses each command (each forward stopped right after
    its argument parse)."""
    got = _parse({a: P.command(a) for a in S.ARMS}, str(declared_src), tmp_path)
    assert got["file"].startswith(str(declared_src))
    for a in S.ARMS:
        assert got[a] == _want(a), (a, got[a])


_DDIR = r"""
import argparse, inspect, json, os
import hydrophysics.twin.forward as F
src = inspect.getsource(F.main)
lines = ["cfg = Config(data_dir=args.data) if args.data else Config()",
         "ddir = str(cfg.data_dir) if cfg.data_dir and os.path.isdir(str(cfg.data_dir)) "
         "else None"]
assert all(s in src for s in lines), "forward.main resolves its data dir differently"
assert src.count("validate_against_leveling(ddir") >= 1 and "if ddir is not None:" in src
ns = {"Config": F.Config, "os": os, "args": argparse.Namespace(data=None)}
for s in lines:
    exec(s, ns)
print(json.dumps({"file": F.__file__, "ddir": ns["ddir"],
                  "env": os.environ.get("HYDROMIND_GW_DATA")}))
"""


def _ddir(src: Path, cwd: Path, env_over: dict) -> dict:
    env = {k: v for k, v in os.environ.items() if k != "HYDROMIND_GW_DATA"}
    env.update(env_over, PYTHONPATH=str(src))
    p = subprocess.run([sys.executable, "-P", "-c", _DDIR], cwd=cwd, env=env,
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-2000:]
    return json.loads(p.stdout.strip().splitlines()[-1])


def test_declared_code_reads_no_data_dir_under_the_runner_env(tmp_path, declared_src):
    """Under the exact environment of the declared command, the declared code's forward.main
    resolves no data dir (so it neither reads nor prints leveling and cannot crash in
    validate_against_leveling before writing the npz). Control: with the variable unset it
    would fall back to the bundled synthetic sample_data, the crash review found."""
    import shlex

    for a in S.ARMS:
        toks = shlex.split(P.command(a))
        assert toks[0] == "env" and "=" not in toks[3]          # env NAME=V NAME=V python
        env = dict(t.split("=", 1) for t in toks[1:3])
        assert set(env) == {"HYDROMIND_GW_DATA", "PYTHONPATH"}
        assert not (REPO / env["HYDROMIND_GW_DATA"]).exists()
        # the forward runs from the repository root; resolve from a root-like cwd where
        # the relative path is equally absent
        got = _ddir(declared_src, tmp_path, {"HYDROMIND_GW_DATA": env["HYDROMIND_GW_DATA"]})
        assert got["file"].startswith(str(declared_src))
        assert got["env"] == P.NOGW and got["ddir"] is None, got
    ctl = _ddir(declared_src, tmp_path, {})
    assert ctl["env"] is None and ctl["ddir"] is not None
    assert ctl["ddir"].endswith("sample_data")
    assert not (Path(ctl["ddir"]) / "ls_cache").exists()      # the panel the crash needs


def test_whitelist_has_no_never_key():
    assert not set(C.FM_KEYS + C.FH_META_KEYS) & set(C.NEVER_KEYS)


@needs_results
def test_declaration_round_trips_in_tmp_registry(tmp_path):
    from harness import prereg3

    path, h = prereg3.write_prereg(P.doc(), prereg_dir=tmp_path / "prereg",
                                   rows_dir=tmp_path / "rows", registries=[])
    got, sha = C.load_registered(S.PREREG_ID, prereg_dir=tmp_path / "prereg",
                                 rows_dir=tmp_path / "rows")
    assert sha == h and C.check_frozen(got, S.FROZEN) == got["frozen_code"]
    assert S.check_input_digests(got) == P.INPUT_DIGESTS


# ----------------------------------------------------------------------------- scoring
def test_synthetic_within_first_year(tmp_path, registered, monkeypatch):
    f = world(tmp_path)
    seen = []
    orig = np.lib.npyio.NpzFile.__getitem__

    def spy(self, key):
        seen.append(key)
        return orig(self, key)
    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", spy)
    att = attempts(tmp_path / "att.jsonl", f)
    out = tmp_path / "rep.json"
    assert run_main(f, att, out) == 0
    assert set(seen) <= set(C.FM_KEYS + C.FH_META_KEYS)
    assert not set(seen) & set(C.NEVER_KEYS)
    r = json.loads(out.read_text())
    pr = r["primary"]
    assert pr["lstar_force_months"] == expected_lstar() == 7
    assert pr["lstar_force_month"] == "2020-07" and pr["case"] == "within_first_year"
    assert "about 7 months" in pr["wording"]
    assert r["e1"]["ok"] and r["e1"]["arms"]["C19"]["max_abs_diff_m"] == 0.0
    rep = r["reported_not_gating"]
    assert rep["stays_above_lead"] == 7
    assert rep["n2_unit_yunlin84"]["first_exceeds"] == 7
    assert rep["M_2019_efold_n2_unit"]["tau_1e_months"] == pytest.approx(M_TAU, abs=1e-3)
    assert rep["M_2022_from_fs"]["n2_unit_at_leads_m"]["1"] == pytest.approx(0.7, abs=1e-5)
    for k in ("fs_start_S1", "fs_start_S0"):
        assert set(r["files"][k]["keys_read"]) <= set(C.FM_KEYS)


def test_synthetic_within_record_and_not_reached(tmp_path, registered):
    f = world(tmp_path, f_slope=0.0008)
    att = attempts(tmp_path / "att.jsonl", f)
    out = tmp_path / "rep.json"
    assert run_main(f, att, out) == 0
    pr = json.loads(out.read_text())["primary"]
    want = expected_lstar(f_slope=0.0008)
    assert 12 < want <= 36 and pr["lstar_force_months"] == want
    assert pr["case"] == "within_record"
    t2 = tmp_path / "b"
    t2.mkdir()
    f = world(t2, f_slope=0.0)
    att = attempts(t2 / "att.jsonl", f)
    assert run_main(f, att, t2 / "rep.json") == 0
    pr = json.loads((t2 / "rep.json").read_text())["primary"]
    assert pr["lstar_force_months"] is None and pr["case"] == "not_reached"


def test_not_evaluable_when_the_start_carries_nothing(tmp_path, registered):
    f = world(tmp_path, m_amp=0.01)
    att = attempts(tmp_path / "att.jsonl", f)
    out = tmp_path / "rep.json"
    assert run_main(f, att, out) == 0
    pr = json.loads(out.read_text())["primary"]
    assert not pr["evaluable"] and pr["case"] == "not_evaluable"
    assert pr["lstar_force_months"] is None


def test_e1_failure_computes_nothing(tmp_path, registered, capsys):
    f = world(tmp_path, c19_jitter=2e-3)
    att = attempts(tmp_path / "att.jsonl", f)
    out = tmp_path / "rep.json"
    assert run_main(f, att, out) == 3
    assert not out.exists() and "do not share the 2019-12 state" in capsys.readouterr().err


def test_e1_numeric_noise_passes_and_is_recorded(tmp_path, registered):
    """A float32-ulp-sized state difference (1e-5 m) is inside the declared numeric band: E1
    passes, the report records the match as not bit-identical, and L*_force is unchanged."""
    f = world(tmp_path, c19_jitter=1e-5)
    att = attempts(tmp_path / "att.jsonl", f)
    out = tmp_path / "rep.json"
    assert run_main(f, att, out) == 0
    r = json.loads(out.read_text())
    assert r["e1"]["ok"] and r["e1"]["tol_m"] == 1e-3
    assert not r["e1"]["arms"]["C19"]["bit_identical"]
    assert r["e1"]["arms"]["C19"]["max_abs_diff_m"] == pytest.approx(1e-5, rel=0.1)
    assert r["primary"]["lstar_force_months"] == expected_lstar()


@pytest.mark.parametrize("over", [
    {"fo_over": {"record_end": "2018-12"}}, {"gain": 1.0},
    {"fo_over": {"climatology_end": "2018-12"}}, {"fo_over": {"update_method": "field"}},
    {"drop": ("scenarios",)}])
def test_recipe_metadata_is_checked(tmp_path, registered, over):
    f = world(tmp_path)
    hm = base_heads(1, N_NEW)
    f["C19"] = fake_new(tmp_path / "C19b.npz", "C19", hm, **over)
    att = attempts(tmp_path / "att.jsonl", f)
    assert run_main(f, att, tmp_path / "rep.json") == 2


def test_attempt_log_binds_files_and_code(tmp_path, registered):
    f = world(tmp_path)
    att = attempts(tmp_path / "att.jsonl", f, codes=("4c2723e", "730f65b"))
    assert run_main(f, att, tmp_path / "rep.json") == 1
    att = attempts(tmp_path / "att.jsonl", f, rc=1)
    assert run_main(f, att, tmp_path / "rep.json") == 1
    att = attempts(tmp_path / "att.jsonl", {**f, "N19": f["C19"]})
    assert run_main(f, att, tmp_path / "rep.json") == 1
    assert run_main(f, tmp_path / "none.jsonl", tmp_path / "rep.json") == 1
    assert not (tmp_path / "rep.json").exists()


def test_scored_once(tmp_path, registered):
    f = world(tmp_path)
    att = attempts(tmp_path / "att.jsonl", f)
    out = tmp_path / "rep.json"
    assert run_main(f, att, out) == 0
    assert run_main(f, att, out) == 1


def test_unregistered_is_engineering(tmp_path):
    f = world(tmp_path)
    att = attempts(tmp_path / "att.jsonl", f)
    assert run_main(f, att, tmp_path / "rep.json") == 1
    assert S.main(["--verify-registration"]) == 1


# ----------------------------------------------------------------------------- mechanism
def _tiny_inputs(T=24):
    """A tiny synthetic fan (the fixture of tests/test_twin_forward.py) whose pumping and
    recharge step up after the first year, so a climatology of the first year differs from
    the recorded second year."""
    import pandas as pd
    import torch

    from hydrophysics.twin.grid import FanGrid
    from hydrophysics.twin.inputs import TwinInputs

    g = FanGrid(nx=6, ny=3, dx=1000.0, x0=200_000.0, y0=0.0, mask=np.ones((3, 6), dtype=bool))
    A = g.n_active
    well_xy = np.array([[200_500.0, 500.0], [204_500.0, 1500.0], [205_500.0, 2500.0]])
    obs_h = np.tile(np.array([[3.0], [4.0], [6.0]]), (1, T)) + 0.1 * np.arange(T)
    late = (np.arange(T) >= 12).astype(float)
    E = {"irrigation": np.full((A, T), 100.0) * (1.0 + late),
         "aquaculture": np.full((A, T), 10.0) + 5.0 * late}
    return TwinInputs(grid=g, hf=None, dates=pd.date_range("2012-01-01", periods=T, freq="MS"),
                      obs_h=obs_h, obs_h_filled=obs_h.copy(),
                      obs_idx=np.array([g.active_index(x, y) for x, y in well_xy]),
                      obs_layer=np.array([0, 1, 1]), well_xy=well_xy, sids=["a", "b", "c"],
                      ground_elev=torch.full((A,), 12.0, dtype=torch.float64), E_by_class=E,
                      recharge_field=torch.full((A, T), 1e-4, dtype=torch.float64)
                      * torch.as_tensor(1.0 + 2.0 * late))


def _tiny_theta(tmp_path) -> str:
    theta = {"log_T_proximal": [np.log(500.0)], "log_S_proximal": [np.log(1e-3)],
             "log_T_mid": [np.log(300.0)] * 4, "log_S_mid": [np.log(1e-3)] * 4,
             "log_T_distal": [np.log(100.0)] * 4, "log_S_distal": [np.log(1e-3)] * 4,
             "log_L_mid": [np.log(1e-4)] * 3, "log_L_distal": [np.log(1e-4)] * 3,
             "log_eta": np.log(0.3), "log_head_extra": np.log(30.0),
             "recharge_frac_logit": 0.0, "log_C_coast": [np.log(50.0)] * 4,
             "log_C_apex": [np.log(50.0)]}
    meta = {"param_mode": "zonal", "boundaries": "coast-apex", "zone_boundaries": "205,203",
            "meter_filter": "none", "eta_classes": None}
    p = tmp_path / "stage3_theta.json"
    p.write_text(json.dumps({"theta": theta, "meta": meta}))
    return str(p)


def _design_check(tmp: str) -> dict:
    """On a tiny synthetic fan, with the forward's own run() and truncate_record() (of
    whichever hydrophysics package is first on sys.path): the record-end runs share the full
    run's hindcast prefix bit for bit (the premise of E1); with gain 0 the innovation update
    leaves both initial fields identical (C19 has no initial-field spread, as declared);
    with gain 1 the start moves the heads (M > 0); and the recorded second year departs from
    the first-year climatology (F > 0)."""
    import hydrophysics.twin.forward as fw
    from hydrophysics.twin.compaction import VEPColumn
    from hydrophysics.twin.forward import load_members, parse_scenario, run
    from hydrophysics.twin.inputs import truncate_record

    inp = _tiny_inputs()
    members = load_members([_tiny_theta(Path(tmp))])
    scen = [(parse_scenario("baseline:irrigation=1")[0], 1.0)]
    kw = dict(ic_members=1, ic_sigma=0.5, seed=0, col=VEPColumn(n_sites=1, dt_days=30.0),
              device="cpu", log=lambda *_: None, restart_taper_km=5.0, column_heads="free",
              apex_hold="calibrated", update_method="innovation")
    s1 = run(inp, members, scen, horizon=6, gain=0.0, **kw)          # fs_start_S1's role
    cut = truncate_record(inp, "2012-12")
    c19 = run(cut, members, scen, horizon=12, gain=0.0, **kw)
    n19 = run(cut, members, scen, horizon=12, gain=1.0, **kw)
    t0 = len(cut.dates)                                                # 12 = origin + 1
    prefix = [float(np.max(np.abs(r["heads_mean"][..., :t0] - s1["heads_mean"][..., :t0])))
              for r in (c19, n19)]
    m = np.abs(n19["heads_mean"][0, :, :, t0:] - c19["heads_mean"][0, :, :, t0:])
    f = np.abs(s1["heads_mean"][0, :, :, t0:24] - c19["heads_mean"][0, :, :, t0:24])
    return {"file": fw.__file__, "prefix_max_abs_diff": prefix,
            "c19_std_max": float(np.abs(c19["heads_std"]).max()),
            "m_max": float(m.max()), "f_max": float(f.max())}


def test_the_design_on_the_declared_code(tmp_path, declared_src):
    """_design_check run against the declared code (4c2723e), in a subprocess whose
    hydrophysics is the git-archived package."""
    pytest.importorskip("torch")
    pytest.importorskip("pyproj")
    code = ("import json, sys; sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2])"
            "; import test_fh_forcing as T; print(json.dumps(T._design_check(sys.argv[3])))")
    env = {**os.environ, "PYTHONPATH": str(declared_src)}
    env.pop("HYDROMIND_GW_DATA", None)
    p = subprocess.run([sys.executable, "-P", "-c", code, str(declared_src), str(TESTS),
                        str(tmp_path)], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-3000:]
    got = json.loads(p.stdout.strip().splitlines()[-1])
    assert got["file"].startswith(str(declared_src))
    assert got["prefix_max_abs_diff"] == [0.0, 0.0]                    # bit for bit
    assert got["c19_std_max"] == 0.0                                   # identical fields
    assert got["m_max"] > 1e-3 and got["f_max"] > 1e-3
