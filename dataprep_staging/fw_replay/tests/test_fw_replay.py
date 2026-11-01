"""CPU tests of the FW.weather_replay.v1 scripts (no forward run, no GPU, nothing registered,
no look logged, nothing written outside pytest's tmp dirs). Every forward output and capture
here is SYNTHETIC, with known elastic and inelastic parts. The engine itself
(hydrophysics/twin/replay.py) is tested in tests/test_twin_replay.py. Run from the repository
root:

    ~/miniforge3/envs/hydro/bin/python -m pytest -q -p no:cacheprovider \\
        --basetemp=<scratch dir> dataprep_staging/fw_replay/tests
"""
from __future__ import annotations

import json
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
sys.path.insert(0, str(REPO / "dataprep_staging"))
import manifest_rain as MR  # noqa: E402
import prereg_fw as P  # noqa: E402
import score_fw as S  # noqa: E402

DECL = REPO / f"results/dataprep/preregistration/{S.PREREG_ID}.json"
ENGINE = {n: f"{i}" * 64 for i, n in enumerate(S.ENGINE)}
CODE = "abc1234"
A, NS, NR, NY, NP = S.N_CELLS, len(S.SCEN_NAMES), len(S.RHEOLOGY), len(S.YEARS), \
    len(S.PROJ_YEARS)
ZONE = (np.arange(A) % 3).astype("int64")
CHANGHUA = np.arange(A) < 300
COUNTY = {"path": P.COUNTY_MAP, "sha256": "c" * 64}
COUNTY_PROV = {"source": P.COUNTY_MAP, "file_sha256": "c" * 64, "county_sha256": "d" * 64,
               "error": None}


# ----------------------------------------------------------------------------- synthetic
def parts(share: float, avoided_el: float = 0.02, scale_tau30: float = 0.5):
    """(el, inel) (S, R, A, Y): baseline subsides linearly; cut30 and cut30_lift avoid
    ``avoided_el`` elastic and a share-matched inelastic amount from 2026; retire_aqua half."""
    t = np.clip(np.arange(NY) - S.YEARS.index(2025), 0, None) / 7.0      # 0 .. 1 at 2032
    a_in = avoided_el * share / (1.0 - share)
    el = np.zeros((NS, NR, A, NY))
    inel = np.zeros((NS, NR, A, NY))
    base_el = 0.01 * np.arange(NY)[None, :] * np.ones((A, 1))
    base_in = 0.005 * np.arange(NY)[None, :] * np.ones((A, 1))
    for r, k in enumerate((1.0, scale_tau30)):
        for s, name in enumerate(S.SCEN_NAMES):
            f = {"baseline": 0.0, "cut30": 1.0, "retire_aqua": 0.5, "cut30_lift": 1.0}[name]
            el[s, r] = base_el - f * avoided_el * t[None, :]
            inel[s, r] = base_in - k * f * a_in * t[None, :]
    lift = S.SCEN_NAMES.index("cut30_lift")
    el[lift, :, :, S.YEARS.index(2030):] += 0.5 * avoided_el     # half the elastic returns
    return el, inel


def newlows(n_runs: int, base_cells: int, cut_cells: int):
    """(S, R, M, A, Yp) per-run new-low flags and (S, R, A, Yp) month counts."""
    ry = np.zeros((NS, NR, n_runs, A, NP), dtype="uint8")
    y = S.PROJ_YEARS.index(2027)
    ry[0, :, :, :base_cells, y] = 1
    ry[1, :, :, :cut_cells, y] = 1
    low = ry.sum(axis=2).astype("int64") * 2
    return ry, low


def dates() -> np.ndarray:
    return np.array([f"{2012 + i // 12}-{i % 12 + 1:02d}-01T00:00:00" for i in range(S.N_T)])


def fake_forward(path: Path, arm: str, el, inel, page: bool = False, heads_shift=0.0):
    spec = S.arm_spec("C" if page else arm, "energy")
    names, descs = (S.REF_NAMES, S.SCEN_DESC[:3]) if page else (S.SCEN_NAMES, S.SCEN_DESC)
    fo = dict(S._FO, save_members="yearly") if page else dict(S._FO)
    pay = dict(dates=dates(), origin=S.ORIGIN, horizon=S.HORIZON,
               scenario_names=np.array(names), scenarios=np.array(descs),
               member_labels=np.array(spec["members"]), n_members=spec["n_members"],
               gain=1.0, hindcast_gain=0.0, rheology_labels=np.array(S.RHEOLOGY),
               gate=json.dumps({"git_commit": S.THETA_GIT}),
               forward_options=json.dumps({**fo, "hpc0_guard_days": [365.0, 365.0]}))
    if not page:
        sbr = np.zeros((NR, NS, A, S.N_T), dtype="float32")
        sbr[..., S._ye_index()] = np.moveaxis(el + inel, 0, 1)
        pay["subs_mean_by_rheology"] = sbr
    if page or arm == "C":
        pay["heads_mean"] = np.full((len(names), 2, 3, 4), 5.0 + heads_shift, dtype="float32")
        pay["subs_mean"] = np.full((len(names), 3, 4), 0.1, dtype="float32")
    np.savez_compressed(path, **pay)
    return path


def fake_capture(path: Path, arm: str, el, inel, ry, low, fields="energy", engine=None,
                 replay_over=None, total_check=0.0, county=COUNTY_PROV, split=True):
    spec = S.arm_spec(arm, fields)
    rp = None if spec["replay"] is None else {
        **spec["replay"], "years_text": "x", "trend_factors": {},
        "annual_anomaly_Y_over_L": {"irrigation": {str(y): 1.0 for y in S.SEQ_STARTS}}}
    if replay_over:
        rp = {**rp, **replay_over}
    rec = {"projection_forcing": spec["projection_forcing"], "replay": rp,
           "code_sha256": engine or ENGINE, "forward_argv": [], "county": county}
    n = spec["n_members"]
    np.savez_compressed(
        path, record=json.dumps(rec), dates=np.array(["x"]), origin=np.int64(S.ORIGIN),
        years=np.array(S.YEARS), proj_years=np.array(S.PROJ_YEARS),
        scenario_names=np.array(S.SCEN_NAMES), rheology_labels=np.array(S.RHEOLOGY),
        zone_ids=np.stack([ZONE, ZONE]),
        changhua=CHANGHUA.astype("int8") if split else np.full(A, -1, dtype="int8"),
        group_names=np.array(S.GROUPS if split else S.GROUPS[:4]), n_runs=np.int64(n),
        elastic_ye_mean=el,
        inelastic_ye_mean=inel, aquitard_ye_mean=np.zeros_like(el), newlow_runs=low,
        newlow_run_year=ry[:, :, :n], run_group_elastic=np.zeros((NS, NR, n, 6, NY)),
        run_group_inelastic=np.zeros((NS, NR, n, 6, NY)),
        total_check_max_abs_m=np.float64(total_check))
    return path


def build_world(root: Path, shares: dict[str, float], clim_share=0.0, fields="energy",
                pipeline=True):
    out = root / "out"
    out.mkdir(parents=True)
    files = {}
    for arm in S.arms(fields):
        sh = clim_share if arm == "C" else shares.get(arm, 0.12)
        el, inel = parts(sh)
        n = S.arm_spec(arm, fields)["n_members"]
        ry, low = newlows(n, 400 if arm != "C" else 0, 100 if arm != "C" else 0)
        fake_forward(out / f"{arm}.npz", arm, el, inel)
        fake_capture(out / f"{arm}.fw.npz", arm, el, inel, ry, low, fields=fields)
        files[arm] = (out / f"{arm}.npz", out / f"{arm}.fw.npz")
    el, inel = parts(clim_share)
    ref = fake_forward(root / "page.npz", "C", el, inel, page=True)
    gate = root / "gate.json"
    gate.write_text(json.dumps({"passed": True, "years": "2012-2022", "months": 132,
                                "fields": {"recharge": {}}, "code_sha256": ENGINE,
                                **({"pipeline": {"passed": True, "recharge_equal": True}}
                                   if pipeline else {})}))
    return {"out": out, "files": files, "ref": ref, "gate": gate}


def attempts(path: Path, w: dict, code=CODE, repro=True, gate=True, skip=(),
             failed_repro=False) -> Path:
    with path.open("w") as fh:
        if failed_repro:
            fh.write(json.dumps({"event": "repro", "rc": 3, "code": code, "time": "t0",
                                 "c_npz_sha256": "0" * 64}) + "\n")
        for arm, (npz, cap) in w["files"].items():
            if arm in skip:
                continue
            fh.write(json.dumps({"event": "forward", "attempt": 1, "arm": arm, "rc": 0,
                                 "code": code, "npz_sha256": S.sha256(npz),
                                 "capture_sha256": S.sha256(cap)}) + "\n")
        if gate:
            fh.write(json.dumps({"event": "gate", "rc": 0, "code": code,
                                 "gate_sha256": S.sha256(w["gate"])}) + "\n")
        if repro:
            fh.write(json.dumps({"event": "repro", "rc": 0, "code": code,
                                 "c_npz_sha256": S.sha256(w["files"]["C"][0])}) + "\n")
    return path


def args(w: dict, att: Path, out: Path | None = None) -> list[str]:
    a = [f"--outputs={w['out']}", f"--ref={w['ref']}", f"--gate={w['gate']}",
         f"--attempts={att}"]
    return a + ([f"--out={out}"] if out else [])


def register(monkeypatch, fields="energy"):
    def fake_load(prereg_dir=None, rows_dir=None):
        return {"frozen_code": {p: S.sha256(REPO / p) for p in S.FROZEN},
                "input_digests": P.INPUT_DIGESTS,
                "recipe": {"replay_fields": fields, "engine_code_sha256": ENGINE,
                           "run_code": CODE, "county_map": COUNTY}}, "f" * 64
    monkeypatch.setattr(S, "load_registered", fake_load)
    d = P.INPUT_DIGESTS[0]
    monkeypatch.setattr(S, "dir_digest", lambda path, pattern: (d["n_files"], d["sha256"]))


@pytest.fixture
def registered(monkeypatch):
    register(monkeypatch)


@pytest.fixture
def any_ref(monkeypatch):
    orig = S.load_forward

    def load(path, arm, *a, **k):
        d = orig(path, arm, *a, **k)
        if k.get("page"):
            monkeypatch.setattr(S, "REF_SHA256", d["sha256"])
        return d
    monkeypatch.setattr(S, "load_forward", load)


SHARES_HIGH = {f"W{y}": s for y, s in zip(S.SEQ_STARTS, np.linspace(0.06, 0.26, 11),
                                          strict=True)}
SHARES_LOW = {f"W{y}": s for y, s in zip(S.SEQ_STARTS, np.linspace(0.01, 0.11, 11),
                                         strict=True)}


@pytest.fixture(scope="module")
def world_high(tmp_path_factory):
    return build_world(tmp_path_factory.mktemp("fw_high"), SHARES_HIGH)


# ----------------------------------------------------------------------------- scoring
def test_synthetic_inelastic_present(world_high, registered, any_ref, tmp_path, capsys):
    w = world_high
    att = attempts(tmp_path / "att.jsonl", w)
    out = tmp_path / "rep.json"
    assert S.main(args(w, att, out)) == 0
    rep = json.loads(out.read_text())
    assert rep["verdict"] == "scored" and rep["wording_case"] == "inelastic_present"
    sh = rep["summary"]["share"]
    assert abs(sh["median"] - 0.16) < 1e-9 and sh["n"] == 11
    assert abs(sh["min"] - 0.06) < 1e-9 and abs(sh["max"] - 0.26) < 1e-9
    rr = rep["reported_not_gating"]
    assert abs(rr["tau30_share"]["median"] - 0.16 * 0.5 / (1 - 0.16 + 0.16 * 0.5)) < 1e-9
    assert abs(rr["retire_aqua_share"]["median"] - 0.16) < 1e-9
    assert set(rr["share_by_zone_and_county_median"]) == {"proximal", "mid", "distal",
                                                         "yunlin_rest", "changhua"}
    for g in ("fan", "proximal", "mid", "distal", "yunlin_rest", "changhua"):
        lift = rr["lift"]["W2012"][g]
        assert abs(lift["three_years_after_dec2032"]["total_m"]
                   - (lift["cut30_kept_dec2032"]["total_m"] - 0.01)) < 1e-9
    assert set(rr["lift_remaining_fraction_by_group_median"]) == {
        "fan", "proximal", "mid", "distal", "yunlin_rest", "changhua"}
    cr = rr["crossings"]["W2015"]
    assert cr["baseline_from_2026"]["cells_half_runs"] == 400
    assert cr["avoided_by_cut30_from_2026_half_runs"] == 300
    assert cr["baseline_from_2026"]["cells_half_runs_by_group"]["changhua"] == 300
    assert sum(cr["avoided_by_cut30_from_2026_half_runs_by_group"][z]
               for z in ("proximal", "mid", "distal")) == 300
    assert rr["crossings"]["C"]["baseline_after_origin"]["cells_any_run"] == 0
    src = rr["baseline_new_lows_by_source_year"]
    # projected 2027 replays the 5th year of each order: every source year once
    assert all(int(v) > 0 for v in src["cell_months"].values())
    prev = rr["prevented_lows_by_source_year"]
    assert prev["total"] == 300 * 36 * 11 and len(prev["listed"]) == 11
    assert rep["summary"]["source_years_cut_prevents_lows"] == prev["listed"]
    assert rr["attribution"]["RW"] is None and rr["attribution"]["RD"]["share"]["n"] == 11
    assert rr["pumping_annual_anomaly_by_source_year"]["values"]["irrigation"]["2015"] == 1.0
    assert rr["county_split"]["C"]["source"] == P.COUNTY_MAP
    wd = rep["wording"]
    assert "16.0 %" in wd["C15"] and "X5.record_low_tf.v1" in wd["C15"]
    assert "pumping anomalies only" in wd["C15"] and "cm of lasting benefit" in wd["C25"]
    assert "pumping-anomaly years" in wd["C25"] and "source years 2012, 2013" in wd["C27"]
    for t in wd.values():                    # pumping-only: no weather or dry-year claim
        assert "dry-year" not in t and "replayed weather" not in t and "weather and" not in t
    assert "no dry year was replayed" in wd["C15"]
    assert "not an interval" in wd["spread_line"]
    assert rep["registration"]["engine_code"] == ENGINE
    assert "VERDICT: scored (wording case inelastic_present" in capsys.readouterr().out


def test_synthetic_all_elastic(registered, any_ref, tmp_path):
    w = build_world(tmp_path / "w", SHARES_LOW)
    out = tmp_path / "rep.json"
    assert S.main(args(w, attempts(tmp_path / "att.jsonl", w), out)) == 0
    rep = json.loads(out.read_text())
    assert rep["wording_case"] == "mostly_elastic"
    assert abs(rep["summary"]["share"]["median"] - 0.06) < 1e-9
    wd = rep["wording"]
    assert wd["C15"].startswith("Mostly elastic (median 6.0 %, below the 10 % rule) also")
    assert "closed for pumping anomalies only; a rain-year replay was not run" in wd["C15"]
    assert "a rain-year replay was not run" in wd["C25"]
    assert "a rain-year replay was not run" in wd["TWIN_EVIDENCE_s13"]
    assert "All elastic" not in wd["C15"]


def test_all_elastic_only_when_every_share_is_zero(registered, any_ref, tmp_path):
    w = build_world(tmp_path / "w", {f"W{y}": 0.0 for y in S.SEQ_STARTS})
    out = tmp_path / "rep.json"
    assert S.main(args(w, attempts(tmp_path / "att.jsonl", w), out)) == 0
    wd = json.loads(out.read_text())["wording"]
    assert wd["C15"].startswith("All elastic also under replayed 2012-22 pumping anomalies")
    assert wd["C27"].startswith("no inelastic channel")


def test_weather_mode_runs_the_weather_only_arms_and_words_weather(monkeypatch, any_ref,
                                                                   tmp_path):
    register(monkeypatch, "energy+weather")
    w = build_world(tmp_path / "w", SHARES_LOW, fields="energy+weather")
    assert set(S.arms("energy+weather")) - set(S.arms("energy")) == set(S.RED_WEA)
    out = tmp_path / "rep.json"
    assert S.main(args(w, attempts(tmp_path / "att.jsonl", w), out)) == 0
    rep = json.loads(out.read_text())
    assert rep["reported_not_gating"]["attribution"]["RW"]["share"]["n"] == 11
    wd = rep["wording"]
    assert "forcing years (weather and detrended pumping)" in wd["C15"]
    assert "the mean-forcing caveat is closed." in wd["C15"]
    assert "pumping anomalies only" not in wd["C15"]
    # a weather-mode world missing its RW files is an engineering failure
    att2 = attempts(tmp_path / "att2.jsonl", w)
    os.remove(w["files"]["RW2017"][1])
    assert S.main(args(w, att2, tmp_path / "r2.json")) in (1, 2)
    assert not (tmp_path / "r2.json").exists()


def test_not_evaluable_when_the_cut_avoids_nothing(registered, any_ref, tmp_path,
                                                   monkeypatch):
    real = parts

    def tiny(share, avoided_el=0.02, scale_tau30=0.5):
        return real(share, 1e-5 if share == 0.5 else avoided_el, scale_tau30)
    monkeypatch.setattr(sys.modules[__name__], "parts", tiny)
    shares = {**SHARES_HIGH, "W2012": 0.5, "W2013": 0.5, "W2014": 0.5}
    w = build_world(tmp_path / "w", shares)
    out = tmp_path / "rep.json"
    assert S.main(args(w, attempts(tmp_path / "att.jsonl", w), out)) == 0
    rep = json.loads(out.read_text())
    assert rep["verdict"] == "not_evaluable" and rep["wording_case"] is None
    assert rep["summary"]["n_evaluable"] == 8 and "not evaluable" in rep["wording"]["ledger_note"]


# ----------------------------------------------------------------------------- engineering
def test_repro_mode_reads_c_only_and_a_failed_repro_is_exit_3(registered, any_ref, tmp_path):
    w = build_world(tmp_path / "w", SHARES_HIGH)
    att = attempts(tmp_path / "att.jsonl", w, repro=False)
    os.remove(w["files"]["W2015"][1])                      # repro never reads another arm
    assert S.main(args(w, att) + ["--repro"]) == 0
    el, inel = parts(0.0)
    fake_forward(w["ref"], "C", el, inel, page=True, heads_shift=0.02)
    assert S.main(args(w, att) + ["--repro"]) == 3


@pytest.mark.parametrize("break_it, msg", [
    ("engine", "engine code"), ("record", "replay record"), ("total", "capture total"),
    ("parts", "parts total"), ("log", "not the capture"), ("nogate", "forcing gate"),
    ("norepro", "reproduction"), ("failedrepro", "failed reproduction"),
    ("pipeline", "pipeline layer"), ("county", "county split came from")])
def test_engineering_failures_refuse_before_any_value(break_it, msg, registered, any_ref,
                                                      tmp_path, capsys):
    w = build_world(tmp_path / "w", SHARES_HIGH, pipeline=break_it != "pipeline")
    att = attempts(tmp_path / "att.jsonl", w, gate=break_it != "nogate",
                   repro=break_it != "norepro", failed_repro=break_it == "failedrepro")
    npz, cap = w["files"]["RR2016"]
    el, inel = parts(0.12)
    ry, low = newlows(1, 400, 100)
    if break_it == "engine":
        fake_capture(cap.with_suffix(".tmp.npz"), "RR2016", el, inel, ry, low,
                     engine={**ENGINE, "forward.py": "0" * 64})
    elif break_it == "record":
        fake_capture(cap.with_suffix(".tmp.npz"), "RR2016", el, inel, ry, low,
                     replay_over={"pumping": "detrended"})
    elif break_it == "total":
        fake_capture(cap.with_suffix(".tmp.npz"), "RR2016", el + 1e-3, inel, ry, low)
    elif break_it == "parts":
        fake_capture(cap.with_suffix(".tmp.npz"), "RR2016", el, inel, ry, low,
                     total_check=1e-3)
    elif break_it == "log":
        fake_capture(cap.with_suffix(".tmp.npz"), "RR2016", el, inel * 1.0, ry, low * 3)
    elif break_it == "county":
        fake_capture(cap.with_suffix(".tmp.npz"), "RR2016", el, inel, ry, low,
                     county={**COUNTY_PROV, "source": "rebuilt from x", "file_sha256": None})
    if break_it in ("engine", "record", "total", "parts", "log", "county"):
        os.replace(cap.with_suffix(".tmp.npz"), cap)
        if break_it != "log":
            att = attempts(tmp_path / "att.jsonl", w)
    out = tmp_path / "rep.json"
    rc = S.main(args(w, att, out))
    assert rc in (1, 2) and not out.exists()
    assert msg in capsys.readouterr().err


def test_a_capture_without_a_county_split_is_kept_and_its_reason_reported(
        registered, any_ref, tmp_path):
    w = build_world(tmp_path / "w", SHARES_HIGH)
    for arm, (_, cap) in w["files"].items():
        n = S.arm_spec(arm, "energy")["n_members"]
        ry, low = newlows(n, 400 if arm != "C" else 0, 100 if arm != "C" else 0)
        os.remove(cap)
        fake_capture(cap, arm, *parts(0.0 if arm == "C" else SHARES_HIGH.get(arm, 0.12)),
                     ry, low, split=False,
                     county={"source": None, "error": "FileNotFoundError: x"})
    out = tmp_path / "rep.json"
    assert S.main(args(w, attempts(tmp_path / "att.jsonl", w), out)) == 0
    rep = json.loads(out.read_text())
    rr = rep["reported_not_gating"]
    assert set(rr["share_by_zone_and_county_median"]) == {"proximal", "mid", "distal"}
    assert rr["county_split"]["W2012"]["error"].startswith("FileNotFoundError")
    assert rep["wording_case"] == "inelastic_present"


def test_scored_once(world_high, registered, any_ref, tmp_path):
    out = tmp_path / "rep.json"
    out.write_text("{}")
    assert S.main(args(world_high, attempts(tmp_path / "a.jsonl", world_high), out)) == 1


def test_real_registration_absent_is_engineering(world_high, tmp_path):
    if DECL.exists():
        pytest.skip("declaration registered")
    assert S.main(args(world_high, tmp_path / "none.jsonl", tmp_path / "r.json")) == 1


def test_sequences_and_arm_specs():
    assert S.cyclic(2015) == (2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2012, 2013, 2014)
    assert S.years_text(S.cyclic(2012)) == "2012-2022"
    assert S.years_text(S.cyclic(2013)) == "2013-2022,2012"
    assert S.years_text(S.cyclic(2022)) == "2022,2012-2021"
    assert S.arm_spec("RR2020", "energy")["replay"]["pumping"] == "raw"
    assert S.arm_spec("W2020", "energy+weather")["n_members"] == 36
    assert S.arm_spec("C", "energy")["replay"] is None
    rw = S.arm_spec("RW2015", "energy")["replay"]
    assert rw["pumping"] == "climatology" and rw["fields"] == "energy+weather"
    from hydrophysics.twin.replay import cyclic_sequences, format_years

    assert [format_years(s) for s in cyclic_sequences()] == \
        [S.years_text(S.cyclic(y)) for y in S.SEQ_STARTS]


# ----------------------------------------------------------------------------- declaration
def test_declaration_validates_and_binds():
    from harness import prereg3

    d = P.doc()
    assert prereg3.validate_prereg(d) == []
    assert P.runner_problems() == [] and P.scorer_problems() == []
    assert d["recipe"]["replay_fields"] in S.FIELD_MODES
    assert set(d["recipe"]["engine_code_sha256"]) == set(S.ENGINE)
    assert d["recipe"]["sequences"]["W2015"] == "2015-2022,2012-2014"
    txt = json.dumps(d, ensure_ascii=False)
    for must in ("X5.record_low_tf.v1", "not an interval", "d290c592", "cut30_lift",
                 "0.1", "model-only", "needs_user_ruling", "public drought record"):
        assert must in txt
    assert "step C is not cited" in txt.lower() or "Step C is not cited" in txt
    assert "drew new minima in the hindcast" not in txt
    assert d["recipe"]["county_map"]["path"] == P.COUNTY_MAP
    m = d["evidence_and_ledger_mapping"]
    # the declared texts are the scorer's own wording function with placeholders
    assert m["case_mostly_elastic"] == S.wording("mostly_elastic", S.PLACEHOLDER_FILL,
                                                 d["recipe"]["replay_fields"])
    assert "all_elastic" not in json.dumps(m)
    if d["recipe"]["replay_fields"] == "energy":
        assert "RW2015" not in json.dumps(d["recipe"]["arms_run"])
        for case in ("case_inelastic_present", "case_mostly_elastic"):
            for t in m[case].values():
                assert "dry-year" not in t and "weather and" not in t
        assert "closed for pumping anomalies only" in m["case_mostly_elastic"]["C15"]


def test_declaration_in_weather_mode(tmp_path):
    mf = tmp_path / P.RAIN_MANIFEST
    mf.parent.mkdir(parents=True)
    mf.write_text(json.dumps({"allowed_uses": ["forcing_frozen"], "fit_window_end": "2022-12",
                              "inputs": [{"path": P.RF, "sha256": P.RF_SHA256}]}))
    d = P.doc(None, tmp_path)
    assert d["recipe"]["replay_fields"] == "energy+weather"
    assert "RW2015" in d["recipe"]["arms_run"] and "forward_RW2015" in d["recipe"]
    assert "--replay-pumping climatology" in d["recipe"]["forward_RW2015"]
    m = d["evidence_and_ledger_mapping"]
    assert "the mean-forcing caveat is closed." in m["case_mostly_elastic"]["C15"]
    assert "launch weather" in d["recipe"]["runner"]


def test_pregate_record_and_readiness(tmp_path):
    assert P.pregate_record("abc1234", tmp_path) is None
    f = tmp_path / P.PREGATE_LOG
    f.parent.mkdir(parents=True)
    f.write_text(json.dumps({"event": "pregate", "code": "abc1234", "rc": 3}) + "\n"
                 + json.dumps({"event": "pregate", "code": "abc1234", "rc": 0,
                               "gate_sha256": "e" * 64, "time": "t"}) + "\n")
    assert P.pregate_record("abc1234", tmp_path)["rc"] == 0
    assert P.pregate_record("abc9999", tmp_path) is None


def test_replay_fields_follow_the_rain_manifest(tmp_path):
    assert P.replay_fields(tmp_path)[0] == "energy"
    m = tmp_path / P.RAIN_MANIFEST
    m.parent.mkdir(parents=True)
    rec = {"allowed_uses": ["forcing_frozen", "design_time_evidence"], "fit_window_end":
           "2022-12", "inputs": [{"path": P.RF, "sha256": P.RF_SHA256}]}
    m.write_text(json.dumps(rec))
    assert P.replay_fields(tmp_path)[0] == "energy+weather"
    m.write_text(json.dumps({**rec, "allowed_uses": ["design_time_evidence"]}))
    assert P.replay_fields(tmp_path)[0] == "energy+weather"
    m.write_text(json.dumps({**rec, "allowed_uses": ["after_the_fact_analysis"]}))
    assert P.replay_fields(tmp_path)[0] == "energy"
    m.write_text(json.dumps({**rec, "inputs": [{"path": P.RF, "sha256": "0" * 64}]}))
    assert P.replay_fields(tmp_path)[0] == "energy"


def test_runner_syntax_and_dry_run_matches_the_declared_commands():
    subprocess.run(["bash", "-n", str(X / "run_fw.sh")], check=True)
    for arm in S.ARMS:
        p = subprocess.run(["bash", str(X / "run_fw.sh"), "forward", arm], cwd=REPO,
                           capture_output=True, text=True, env={**os.environ, "DRY": "1"})
        assert p.returncode == 0, p.stderr
        assert p.stdout.strip() == P.command(arm, "<replay_fields>")
    p = subprocess.run(["bash", str(X / "run_fw.sh"), "forward", "primary", "2/3"], cwd=REPO,
                       capture_output=True, text=True, env={**os.environ, "DRY": "1"})
    got = [ln.rsplit("/", 1)[-1] for ln in p.stdout.split("\n") if ln.strip()]
    assert got == ["W2013", "W2016", "W2019", "W2022"]
    p = subprocess.run(["bash", str(X / "run_fw.sh"), "forward", "weather"], cwd=REPO,
                       capture_output=True, text=True, env={**os.environ, "DRY": "1"})
    got = [ln.rsplit("/", 1)[-1] for ln in p.stdout.split("\n") if ln.strip()]
    assert got == list(S.RED_WEA)


@pytest.mark.skipif(DECL.exists(), reason="declaration registered; refusal test not applicable")
@pytest.mark.parametrize("cmd", [["check"], ["gate"], ["forward", "C"], ["repro"], ["score"],
                                 ["launch", "primary"], ["forward", "all"]])
def test_runner_refuses_without_registration(cmd):
    p = subprocess.run(["bash", str(X / "run_fw.sh"), *cmd], cwd=REPO, capture_output=True,
                       text=True)
    assert p.returncode == 1 and f"declare {S.PREREG_ID} first" in p.stderr


@pytest.mark.skipif(DECL.exists(), reason="declaration registered; pregate refuses then")
def test_runner_pregate_refusals_write_nothing():
    for cmd, msg in ((["pregate"], "7-char run code"), (["pregate", "zzz"], "7-char run code"),
                     (["pregate", "0000000"], "worktree")):
        p = subprocess.run(["bash", str(X / "run_fw.sh"), *cmd], cwd=REPO,
                           capture_output=True, text=True,
                           env={**os.environ, "HOME": "/nonexistent-home"})
        assert p.returncode == 1 and msg in p.stderr, p.stderr
    p = subprocess.run(["bash", str(X / "run_fw.sh"), "forward", "nonsense"], cwd=REPO,
                       capture_output=True, text=True, env={**os.environ, "DRY": "1"})
    assert p.returncode == 1 and "unknown arm or group" in p.stderr


def test_prereg_cli_validate_is_a_dry_run(capsys):
    reg = REPO / "results/dataprep/preregistration"
    before = sorted(reg.glob("FW.*")) if reg.exists() else []
    P.main(["--validate", "--quiet"])
    out = capsys.readouterr().out
    assert "validate_prereg: valid" in out and "recipe/binding checks: ok" in out
    assert "--run-code not given" in out
    assert (sorted(reg.glob("FW.*")) if reg.exists() else []) == before


def test_no_home_paths_in_staged_files():
    import re

    files = [*X.rglob("*.py"), *X.rglob("*.sh"), REPO / "hydrophysics/twin/replay.py",
             REPO / "tests/test_twin_replay.py"]
    for p in files:
        assert not re.search(r"/home/\w", p.read_text()), p


# ----------------------------------------------------------------------------- rain manifest
def _rain(tmp: Path, empty=False, short=False) -> tuple[Path, Path]:
    import datetime as dt

    days = [MR.FIRST + dt.timedelta(days=i) for i in range((MR.LAST - MR.FIRST).days + 1)]
    if short:
        days = days[:-1]
    rows = ["date time,rf1,rf2"]
    for i, d in enumerate(days):
        v = "" if (empty and i == 10) else "1.5"
        rows.append(f"{d.isoformat()} 00:00,{v},0")
    rain = tmp / "rain.csv"
    rain.write_text("﻿" + "\n".join(rows) + "\n", encoding="utf-8")
    st = tmp / "st.csv"
    st.write_text("﻿rf_id,rf_num,TM_X97,TM_Y97\nrf1,1,0,0\nrf2,2,0,0\n", encoding="utf-8")
    return rain, st


def test_rain_manifest_structure_gate_and_build(tmp_path):
    rain, st = _rain(tmp_path)
    rec = MR.structure(rain, st)
    assert rec["n_days"] == 4018 and rec["daily_and_complete"] and rec["empty_cells"] == 0
    assert MR.gate(rec)["passed"]
    m = MR.build(tmp_path / "prod" / MR.VERSION, rain, st)
    assert m["allowed_uses"] == ["forcing_frozen", "design_time_evidence"]
    assert "scoring_target" in m["forbidden_uses"] and m["fit_window_end"] == "2022-12"
    assert {o["path"] for o in m["outputs"]} == {"rain_record.json", "validation.json"}
    assert "1.5" not in (tmp_path / "prod" / MR.VERSION / "rain_record.json").read_text()
    for kw, check in (({"empty": True}, "no_empty_cell"),
                      ({"short": True}, "daily_2012-01-01_to_2022-12-31")):
        d = tmp_path / str(check)
        d.mkdir()
        r2, s2 = _rain(d, **kw)
        g = MR.gate(MR.structure(r2, s2))
        assert not g["passed"] and not g["checks"][check]
