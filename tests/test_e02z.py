"""E02z.level_years23 (round-8 rank #17): E02c's strict level rule per county on E02y's year-2/3
windows of the frozen forecast, its decidability engine, its scorer and its draft registration.

Everything runs on synthetic arrays, synthetic product directories and fake step-(5) rows in
pytest's tmp dir: no model output, window head or observation is read, and no statistic is
computed on a real seed (the engine runs on its synthetic smoke context and smoke seed only).
Checks that need the gitignored registry (results/dataprep) skip when it is absent (CI).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import datetime as dt  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

REPO = Path(__file__).resolve().parents[1]
X = REPO / "dataprep_staging" / "e02z"
for _p in (str(X), str(REPO / "dataprep_staging" / "e02c"), str(REPO / "dataprep_staging")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import decide_e02z as DZ  # noqa: E402
import e02z_const as K  # noqa: E402
import e02z_lib as L  # noqa: E402
import prereg_e02z as P  # noqa: E402
import score_e02c as S  # noqa: E402
import score_e02z as SZ  # noqa: E402

PREREG = REPO / "results" / "dataprep" / "preregistration"
NEED_REGISTRY = pytest.mark.skipif(not (PREREG / f"{K.E02Y_ID}.json").exists(),
                                   reason="the pre-registration registry is gitignored, absent")
REGISTERED = (PREREG / f"{K.ID}.json").exists()
STATION = re.compile(r"\b(?:0[79]|10)[0-9A-Z]{6}\b")
SEED = 424242
NOW_B = dt.datetime(2029, 1, 20, tzinfo=dt.timezone.utc)


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ----------------------------------------------------------------------------- constants
def test_windows_axis_and_dates():
    ax = pd.date_range(f"{K.AXIS_START}-01", periods=K.T_RUN, freq="MS")
    assert K.T_RUN == 214 and str(ax[-1].date()) == "2029-10-01"
    for w, (first, last) in K.WINDOWS.items():
        t0, t1 = K.FWD_WIN[w]
        assert (str(ax[t0].date())[:7], str(ax[t1 - 1].date())[:7]) == (first, last)
        assert list(L.window_months(w)) == list(ax[t0:t1])
        assert K.SCOPE[w] == f"heads:{first}..{last}"
    assert K.NOT_BEFORE == {"B": "2029-01-15T00:00:00Z", "C": "2030-01-15T00:00:00Z"}
    assert K.REG_DEADLINE == "2027-11-01T00:00:00Z"
    assert set(K.UNIT.values()).isdisjoint(K.E02Y_UNIT.values())
    assert "--horizon 82" in K.FWD_FLAGS and "recorded" not in K.FWD_FLAGS


def test_wording_pre_words_the_narrowing():
    nar = K.WORDING["level_not_supported"]
    assert "2.048 m" in nar and "1.667 m" in nar and "average-weather scenario" in nar
    assert "not a forecast" in K.WORDING["level_supported"]
    for txt in [*K.WORDING.values(), *K.DECIDE_SUFFIX.values()]:
        assert not any(n in txt for n in K.NEVER)
    line = L.wording("level_not_supported", "Yunlin", "C",
                     decide={"reading": "weak_by_design", "p": 0.31})
    assert line.startswith("Yunlin:") and "2028-11..2029-10" in line and "0.31" in line
    assert L.wording("level_not_evaluable", "Changhua", "B", why="x").endswith(
        K.DECIDE_SUFFIX["not_run"])


def test_pinned_files_unchanged():
    assert sha(REPO / K.E02C_SCORER) == K.E02C_SCORER_SHA
    assert sha(REPO / K.E02C_EXPORT) == K.E02C_EXPORT_SHA
    assert sha(REPO / K.GOV) == K.GOV_SHA


def test_no_station_ids_in_code():
    for p in sorted(X.glob("*.py")) + [X / "run_e02z.sh"]:
        assert not STATION.search(p.read_text(encoding="utf-8")), p.name


# ----------------------------------------------------------------------------- seeds
@pytest.mark.parametrize("bad", [None, True, -1, 1.5, "7", *K.REFUSED_SEEDS])
def test_check_seed_refuses(bad):
    with pytest.raises(L.Refused):
        L.check_seed(bad)


def test_seeds_have_no_default_and_cover_fv_stat_v2():
    assert L.check_seed(SEED) == SEED
    for f, name in ((P.draft_prereg, "seed"), (DZ.run, "seed"), (DZ.score_worlds, "seed")):
        assert inspect.signature(f).parameters[name].default is inspect.Parameter.empty
    sv2 = REPO / "dataprep_staging" / "fv_stat_v2" / "stat_v2.py"
    src = sv2.read_text(encoding="utf-8")
    for s in ("ST.SEED", "ST.SMOKE_SEED", "SMOKE_SEED"):
        assert s in src
    assert {20261103, 77770001, 51120017, K.FVSTAT2_SEED} <= set(K.REFUSED_SEEDS)


# ----------------------------------------------------------------------------- the rule
def _frame(n_y=25, n_c=22, zr_off=0.0, cl_off=2.0, clt_off=2.5, noise=0.3, seed=1, win="B"):
    """Synthetic annotated rows and baselines: obs = truth + noise."""
    rng = np.random.default_rng(seed)
    ids = [f"09{i:06d}" for i in range(n_y)] + [f"07{i:06d}" for i in range(n_c)]
    months = L.window_months(win)
    st = np.repeat(ids, len(months))
    dates = np.tile(months, len(ids))
    truth = np.repeat(rng.normal(0, 3, len(ids)), len(months))
    obs = truth + rng.normal(0, noise, truth.size)
    rows = pd.DataFrame({"station_id": st, "date": dates, "predicted_m": truth + zr_off,
                         "lower_m": truth - 5, "upper_m": truth + 5, "observed_m": obs})
    base = pd.DataFrame({"station_id": st, "date": dates, "cl_m": truth + cl_off,
                         "clt_m": truth + clt_off})
    blocks = {s: f"b{i % 12}" for i, s in enumerate(ids)}
    return S.annotate(rows), base, blocks


def test_county_rule_supported_not_supported_not_evaluable():
    f, base, blocks = _frame()
    lv = L.window_level(f, base, blocks, SEED, "B")["counties"]
    assert {c: v["verdict"] for c, v in lv.items()} == {"Yunlin": "level_supported",
                                                       "Changhua": "level_supported"}
    f, base, blocks = _frame(zr_off=3.0)
    lv = L.window_level(f, base, blocks, SEED, "B")["counties"]
    assert all(v["verdict"] == "level_not_supported" for v in lv.values())
    f, base, blocks = _frame(n_c=10)
    lv = L.window_level(f, base, blocks, SEED, "B")["counties"]
    assert lv["Changhua"]["verdict"] == "level_not_evaluable"
    assert lv["Yunlin"]["verdict"] == "level_supported"


def test_county_needs_ten_blocks():                         # review S5
    f, base, blocks = _frame()
    few = {s: (f"b{i % 9}" if s.startswith("07") else b)
           for i, (s, b) in enumerate(blocks.items())}
    lv = L.window_level(f, base, few, SEED, "B")["counties"]
    assert lv["Changhua"]["verdict"] == "level_not_evaluable" and lv["Changhua"]["n_blocks"] == 9
    assert "5 km blocks" in lv["Changhua"]["why"]
    assert lv["Yunlin"]["verdict"] == "level_supported" and lv["Yunlin"]["n_blocks"] == 12


def test_counties_are_decided_on_their_own_and_outside_months_ignored():
    f, base, blocks = _frame(seed=3)
    a = L.window_level(f, base, blocks, SEED, "B")["counties"]["Yunlin"]
    g = f.copy()
    ch = g["county"] == "Changhua"
    g.loc[ch, "observed_m"] += 50.0                         # Changhua rows only
    extra = g.iloc[:5].copy()
    extra["date"] = pd.Timestamp("2027-10-01")               # outside window B
    g = pd.concat([g, extra], ignore_index=True)
    b = L.window_level(g, base, blocks, SEED, "B")["counties"]["Yunlin"]
    assert json.dumps(a, sort_keys=True, default=L.jsonable) == json.dumps(
        b, sort_keys=True, default=L.jsonable)


def test_the_seed_is_the_bootstrap_seed():
    f, base, blocks = _frame(seed=4)
    a = L.window_level(f, base, blocks, SEED, "B")["counties"]["Yunlin"]
    b = L.window_level(f, base, blocks, SEED + 1, "B")["counties"]["Yunlin"]
    assert a["vs_CL"]["seed"] == SEED and b["vs_CL"]["seed"] == SEED + 1
    assert a["vs_CL"]["block_ci"] != b["vs_CL"]["block_ci"]


# ----------------------------------------------------------------------------- decidability
def test_noise_paths_are_seeded_and_span_both_windows():
    ctx, _ = DZ.synthetic_context()
    a = DZ.noise_path(ctx, SEED, K.STREAM_NOISE, 0, 3)
    b = DZ.noise_path(ctx, SEED, K.STREAM_NOISE, 0, 3)
    c = DZ.noise_path(ctx, SEED + 1, K.STREAM_NOISE, 0, 3)
    assert np.array_equal(a["B"], b["B"]) and np.array_equal(a["C"], b["C"])
    assert not np.array_equal(a["B"], c["B"])
    assert a["B"].shape == a["C"].shape == (ctx["W"], 12)


def test_departure_removes_the_fitted_datum():
    W = 4
    mask = np.zeros((W, K.T_FIT), bool)
    mask[:, ::3] = True
    zr0 = np.random.default_rng(0).normal(size=(W, K.T_RUN))
    t = zr0 + 1.7                                            # pure datum offset
    D = DZ.departure(t, zr0, mask)
    assert all(np.allclose(D[w], 0.0) for w in K.WINDOWS)
    t2 = t.copy()
    t2[:, K.FWD_WIN["C"][0]:] += 0.5
    assert np.allclose(DZ.departure(t2, zr0, mask)["C"], 0.5)


def test_hindcast_identity_and_forward_meta():
    h = np.zeros((3, K.T_RUN))
    ref = np.zeros((3, 190))
    assert DZ.hindcast_identity("R0", h, ref)["hindcast_max_abs_m"] == 0.0
    ref[1, 10] = 1e-9
    with pytest.raises(L.EngineeringFailure, match="hindcast"):
        DZ.hindcast_identity("R0", h, ref)
    meta = {"scenario_names": np.array(["baseline"]),
            "dates": np.array([str(d.date()) for d in pd.date_range("2012-01-01",
                                                                   periods=K.T_RUN,
                                                                   freq="MS")]),
            "origin": np.array(K.ORIGIN_IDX), "horizon": np.array(K.HORIZON),
            "member_labels": np.array(["stage3_datum_gate_cwa"]), "n_members": np.array(1),
            "forward_options": np.array(json.dumps({"update_method": "innovation",
                                                    "column_heads": "free"})),
            "gain": np.array(1.0), "hindcast_gain": np.array(0.0),
            "gate": np.array(json.dumps({"git_commit": "x"}))}
    hh = np.zeros((1, 3, K.T_RUN))
    assert DZ.check_forward_meta("R0", meta, hh, ["stage3_datum_gate_cwa"])["members"] == 1
    bad = dict(meta, horizon=np.array(58))
    with pytest.raises(L.EngineeringFailure, match="horizon"):
        DZ.check_forward_meta("R0", bad, hh, ["stage3_datum_gate_cwa"])
    arm = dict(meta, forward_options=np.array(json.dumps(
        {"update_method": "innovation", "column_heads": "free", "recorded_pumping": {"a": 1}})))
    with pytest.raises(L.EngineeringFailure, match="recorded-pumping"):
        DZ.check_forward_meta("R0", arm, hh, ["stage3_datum_gate_cwa"])


def test_selfcheck_engine_path_equals_scorer_path():
    ctx, D = DZ.synthetic_context()
    sc = DZ.selfcheck(ctx, D, K.SMOKE_SEED)
    assert len(sc) == K.N_SELFCHECK * len(K.WINDOWS) and all(x["identical"] for x in sc)


def test_cell_reading_rule_on_fabricated_records():
    ctx, _ = DZ.synthetic_context()
    win = [{"pass": True, "verdict": "level_supported"}] * 8
    lose = [{"pass": False, "verdict": "level_not_supported"}] * 8
    nul = [{"pass": False, "verdict": "level_not_supported"}] * 8
    per = {t: (win if i < 3 else lose) for i, t in enumerate(K.TAGS)}
    r = DZ.cell_reading(ctx, "B", "Yunlin", per, nul, 8)
    assert r["P"] == 0.75 and r["reading"] == "not_shown_weak"
    per = {t: (win if i < 2 else lose) for i, t in enumerate(K.TAGS)}
    assert DZ.cell_reading(ctx, "B", "Yunlin", per, nul, 8)["reading"] == "weak_by_design"
    r = DZ.cell_reading(ctx, "B", "Yunlin", per, [{"pass": True, "verdict": "x"}] * 8, 8)
    assert r["reading"] == "not_evaluable" and not r["null"]["gate_n_passed"]
    assert set(r["per_truth_report_only"]) == set(K.TAGS)


def test_reading_for_and_suffix():
    assert DZ.reading_for(None, "B", "Yunlin") == {"reading": "not_run"}
    rep = {"created": "2027-03-01T00:00:00+00:00",
           "readings": {"B": {"Yunlin": {"reading": "not_shown_weak", "P": 0.8}}}}
    assert DZ.reading_for(rep, "B", "Yunlin") == {"reading": "not_shown_weak", "p": 0.8,
                                                  "why": ""}
    assert DZ.reading_for(rep, "C", "Yunlin") == {"reading": "not_run"}
    late = dict(rep, created="2029-01-15T00:00:00+00:00")             # review B1
    assert DZ.reading_for(late, "B", "Yunlin") == {"reading": "not_run"}
    assert DZ.reading_for({k: v for k, v in rep.items() if k != "created"}, "B",
                          "Yunlin") == {"reading": "not_run"}
    assert "optimistic" in K.DECIDE_SUFFIX["not_shown_weak"]


def test_smoke_runs_and_writes_no_rate(tmp_path):
    rep = DZ.run_smoke(tmp_path / "smoke", k=2)
    assert rep["status"] == "ran ok"
    txt = (tmp_path / "smoke" / "smoke.json").read_text()
    assert '"P"' not in txt and "readings" not in txt
    with pytest.raises(L.EngineeringFailure, match="results"):
        DZ.run_smoke(REPO / "results" / "e02z_smoke_never")


def _events(p: Path) -> list[str]:
    return [json.loads(x)["event"] for x in p.read_text().splitlines()]


def test_run_refuses_seed_report_and_registration(tmp_path, monkeypatch):
    att, out = tmp_path / "att.jsonl", tmp_path / "rep.json"
    with pytest.raises(L.Refused, match="refused"):
        DZ.run(K.SMOKE_SEED, out=str(out), attempts=str(att))
    assert _events(att) == ["start", "refused"]
    out.write_text("{}")
    with pytest.raises(L.Refused, match="written once"):
        DZ.run(SEED, out=str(out), attempts=str(att))
    out.unlink()
    from harness import prereg3

    def boom(*a, **k):
        raise prereg3.PreregistrationError("no such registration")

    monkeypatch.setattr(prereg3, "load_prereg", boom)
    with pytest.raises(L.Refused, match="not registered"):
        DZ.run(SEED, out=str(out), attempts=str(att))
    assert _events(att)[-2:] == ["start", "refused"] and not out.exists()
    assert [json.loads(x).get("attempt") for x in att.read_text().splitlines()
            if json.loads(x)["event"] == "start"] == [1, 2, 3]


def test_run_refuses_after_window_b_opens_or_a_score_exists(tmp_path, monkeypatch):
    att, out = tmp_path / "att.jsonl", tmp_path / "rep.json"            # review B1
    late = dt.datetime(2029, 1, 15, tzinfo=dt.timezone.utc)
    with pytest.raises(L.Refused, match="before 2029-01-15"):
        DZ.run(SEED, out=str(out), attempts=str(att), now=late)
    sc = tmp_path / "report_C.json"
    sc.write_text("{}")
    monkeypatch.setitem(K.SCORE_REPORT, "C", str(sc))
    with pytest.raises(L.Refused, match="score report exists"):
        DZ.run(SEED, out=str(out), attempts=str(att),
               now=dt.datetime(2027, 1, 1, tzinfo=dt.timezone.utc))
    assert _events(att) == ["start", "refused", "start", "refused"] and not out.exists()


def test_registration_deadline_is_enforced(monkeypatch):            # review S6
    assert L.deadline_problem("2026-12-01T00:00:00.000Z") is None
    assert "not before" in L.deadline_problem("2027-11-01T00:00:00.000Z")
    assert L.deadline_problem(None)
    from harness import prereg3

    monkeypatch.setattr(prereg3, "load_prereg",
                        lambda *a, **k: ({"created": "2027-11-02T00:00:00Z"}, "x"))
    with pytest.raises(L.Refused, match="not before"):
        DZ.load_registered()
    with pytest.raises(ValueError, match="registered before"):
        P.draft_prereg(SEED, ids=[str(i) for i in range(K.N_WELLS)], check_files=False,
                       now=dt.datetime(2027, 11, 1, tzinfo=dt.timezone.utc))


def test_fv1_reference_forwards_are_pinned(monkeypatch, tmp_path):    # review S4
    V2 = DZ.V2
    assert K.FV1_FWD_SHA == V2.V1_FORWARD_SHA
    monkeypatch.setattr(K, "fv1_npz", lambda t: str(tmp_path / f"{t}.npz"))
    assert len(P.fv1_forward_problems()) == 5
    (tmp_path / "ZR0.npz").write_bytes(b"x")
    assert any("ZR0" in x for x in P.fv1_forward_problems())
    ctx, _ = DZ.synthetic_context()
    with pytest.raises(L.EngineeringFailure, match="pinned sha256"):
        DZ.load_departures(ctx)
    with pytest.raises(ValueError, match="reference forwards"):
        P.draft_prereg(SEED, ids=[str(i) for i in range(K.N_WELLS)])


def test_run_refuses_a_different_registered_design(monkeypatch):
    doc = {"decidability": {"seed": SEED + 1, "k_truth": K.K_TRUTH, "k_null": K.K_NULL,
                            "streams": {"noise": K.STREAM_NOISE, "null": K.STREAM_NULL,
                                        "selfcheck": K.STREAM_SELFCHECK},
                            "truths": [list(t) for t in K.TRUTHS], "p_weak": K.P_WEAK}}
    with pytest.raises(L.EngineeringFailure, match="seed"):
        DZ.check_registered(doc, SEED)
    doc["decidability"]["seed"] = SEED
    DZ.check_registered(doc, SEED)


# ----------------------------------------------------------------------------- scorer
def _window_product(root: Path, window) -> Path:
    spec = importlib.util.spec_from_file_location("gov_hw_test_e02z", REPO / K.GOV)
    HW = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(HW)
    d = root / "heads_window" / "v1"
    d.mkdir(parents=True)
    (d / "validation.json").write_text("{}\n")
    rec = {"path": "heads.parquet", "sha256": "0" * 64}
    m = HW.instantiate(HW.template(), version="v1", created="2029-01-10", inputs=[], code=[],
                       git_commit="0" * 40, fit_window_end="2029-10", outputs=[rec],
                       validation_report="validation.json",
                       gate={"passed": True, "criteria": "synthetic"}, window=window)
    (d / "MANIFEST.json").write_text(json.dumps(m, indent=1))
    return d


def test_precondition_is_e02z_own_covering_rule(tmp_path):
    ok, why = SZ.precondition(_window_product(tmp_path / "a", ("2026-11", "2029-10")), "B")
    assert ok and "covering_rule" in why
    ok, why = SZ.precondition(_window_product(tmp_path / "b", ("2027-11", "2028-10")), "C")
    assert not ok and "does not cover" in why
    ok, why = SZ.precondition(_window_product(tmp_path / "c", ("2027-11", "2028-10")), "B")
    assert ok


def _e02y_report(root: Path, win: str, **extra) -> Path:
    p = root / "e02y" / "report.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"prereg_id": K.E02Y_ID, "window": win, **extra}))
    return p


def _bind(rep: Path) -> None:
    """Write E02y's rows_provenance from the step-(5) files (as score_e02d.py records it)."""
    d = json.loads(rep.read_text())
    d["rows_provenance"] = {f"{n}_sha256": sha(rep.parent / "step5" / n)
                            for n in S.E02_STEP5_FILES}
    rep.write_text(json.dumps(d))


def test_e02y_order(tmp_path, monkeypatch):
    assert not SZ.e02y_status(tmp_path / "none.json", "B", tmp_path)["ok"]
    r = _e02y_report(tmp_path, "C")
    assert "window 'C'" in SZ.e02y_status(r, "B", tmp_path)["why"]
    monkeypatch.setattr(S, "assert_order", lambda req, d: list(req))      # E02y did not look
    r = _e02y_report(tmp_path, "B", verdict="not_evaluable", why="no product")
    st = SZ.e02y_status(r, "B", tmp_path)
    assert st["ok"] and not st["evaluable"] and not st["e02y_looked"]
    r = _e02y_report(tmp_path, "B")
    assert "not logged" in SZ.e02y_status(r, "B", tmp_path)["why"]
    monkeypatch.setattr(S, "assert_order", lambda req, d: [])             # E02y looked
    for v in ("not_evaluable", K.NOT_SCORABLE):                           # review S3
        r = _e02y_report(tmp_path, "B", verdict=v)
        st = SZ.e02y_status(r, "B", tmp_path)
        assert not st["ok"] and "engineering error" in st["why"]
    r = _e02y_report(tmp_path, "B")
    assert "step (5) files missing" in SZ.e02y_status(r, "B", tmp_path)["why"]
    s5 = r.parent / "step5"
    s5.mkdir()
    for n in S.E02_STEP5_FILES:
        (s5 / n).write_text("{}")
    assert "rows_provenance" in SZ.e02y_status(r, "B", tmp_path)["why"]   # review Q6
    _bind(r)
    st = SZ.e02y_status(r, "B", tmp_path)
    assert st["ok"] and st["evaluable"] and st["step5"] == str(s5)
    (s5 / "evaluation.csv").write_text("{} ")
    assert not SZ.e02y_status(r, "B", tmp_path)["ok"]


def _step5(root: Path, ids, win="B", manifest=None, **override) -> Path:
    s5 = root / "step5"
    s5.mkdir(parents=True, exist_ok=True)
    (s5 / "report.json").write_text(json.dumps({"registration_manifest": manifest or {"m": 1}}))
    months = [str(m.date()) for m in L.window_months(win)][:3]
    cols = {"station_id": np.repeat(ids, 3), "date": np.tile(months, len(ids)),
            "predicted_m": 0.0, "lower_m": -1.0, "upper_m": 1.0, "baseline_m": 0.0,
            "observed_m": 0.1}
    cols.update(override)
    pd.DataFrame(cols).to_csv(s5 / "evaluation.csv", index=False)
    return s5


def test_rows_contract_before_the_look(tmp_path):
    ids = [f"09{i:06d}" for i in range(4)] + [f"07{i:06d}" for i in range(4)]
    man = {"m": 1}
    assert SZ.rows_contract_problems(_step5(tmp_path / "g", ids), ids, "B", man) == []
    p = SZ.rows_contract_problems(_step5(tmp_path / "m", ids), ids, "B", {"m": 2})
    assert any("registered manifest" in x for x in p)
    s5 = _step5(tmp_path / "c", ids)
    pd.read_csv(s5 / "evaluation.csv").drop(columns="observed_m").to_csv(
        s5 / "evaluation.csv", index=False)
    assert "lacks columns" in SZ.rows_contract_problems(s5, ids, "B", man)[0]
    lost = [s.lstrip("0") for s in ids]
    p = SZ.rows_contract_problems(_step5(tmp_path / "z", ids, station_id=np.repeat(lost, 3)),
                                  ids, "B", man)
    assert any("leading zeros" in x for x in p)
    p = SZ.rows_contract_problems(_step5(tmp_path / "o", ids), ids[:4], "B", man)
    assert any("not the 158" in x for x in p)
    p = SZ.rows_contract_problems(_step5(tmp_path / "d", ids, date="not a month"), ids, "B",
                                  man)
    assert any("does not parse" in x for x in p)
    p = SZ.rows_contract_problems(_step5(tmp_path / "w", ids, win="C"), ids, "B", man)
    assert any("no row" in x for x in p)


@pytest.mark.skipif(REGISTERED, reason="E02z is registered")
def test_scorer_refuses_before_registration(tmp_path):
    att = tmp_path / "att.jsonl"
    with pytest.raises(SZ.Refused, match="not registered"):
        SZ.score("B", tmp_path, tmp_path / "r.json", now=NOW_B, out=str(tmp_path / "o.json"),
                 attempts=str(att), prereg_dir=tmp_path)
    assert _events(att) == ["score_invoked", "score_refused"]


CREATED = "2026-12-01T00:00:00.000000Z"


def _registered(monkeypatch, frozen=None, ids=None, created=CREATED):
    from harness import prereg3

    ids = ids or []
    doc = {"frozen_code": frozen if frozen is not None else {}, "created": created,
           "thresholds": {"boot_seed": SEED},
           "tests": [{"test_id": K.TEST[w], "external_units": {K.UNIT[w]: {"ids": ids}}}
                     for w in K.WINDOWS]}
    monkeypatch.setattr(prereg3, "load_prereg", lambda *a, **k: (doc, "x" * 64))
    return doc


def test_scorer_refuses_early_twice_late_registered_and_unbound(tmp_path, monkeypatch):
    _registered(monkeypatch)
    att, out = tmp_path / "att.jsonl", tmp_path / "o.json"
    early = dt.datetime(2029, 1, 14, 23, tzinfo=dt.timezone.utc)
    kw = {"out": str(out), "attempts": str(att), "prereg_dir": tmp_path}
    with pytest.raises(SZ.Refused, match="not scored before 2029-01-15"):
        SZ.score("B", tmp_path, tmp_path / "r.json", now=early, **kw)
    with pytest.raises(SZ.Refused, match="2030-01-15"):
        SZ.score("C", tmp_path, tmp_path / "r.json", now=NOW_B, **kw)
    with pytest.raises(SZ.Refused, match="heads_window.py is not bound"):
        SZ.score("B", tmp_path, tmp_path / "r.json", now=NOW_B, **kw)
    _registered(monkeypatch, created="2027-11-01T00:00:00.000000Z")       # review S6
    with pytest.raises(SZ.Refused, match="not before 2027-11-01"):
        SZ.score("B", tmp_path, tmp_path / "r.json", now=NOW_B, **kw)
    out.write_text("{}")
    with pytest.raises(SZ.Refused, match="scored once"):
        SZ.score("B", tmp_path, tmp_path / "r.json", now=NOW_B, **kw)
    assert all(e in ("score_invoked", "score_refused") for e in _events(att))


def test_scorer_imports_the_engine_at_module_top():                     # review S2
    src = (X / "score_e02z.py").read_text(encoding="utf-8")
    assert SZ.DZ is DZ and "\nimport decide_e02z as DZ" in src
    body = src[src.index("def score("):]
    assert "import decide_e02z" not in body


def _session(tmp_path, monkeypatch, ids, *, window=("2026-11", "2029-10"), looked=True,
             verdict=None):
    """A synthetic window-B session: registration, E02y files and looks stubbed; the product,
    step (5), baselines and wells are synthetic files."""
    from harness import prereg3

    bound = {K.GOV: K.GOV_SHA, K.E02C_SCORER: K.E02C_SCORER_SHA}
    _registered(monkeypatch, frozen=bound, ids=ids)
    pdir = tmp_path / "prereg"
    pdir.mkdir()
    (pdir / f"{K.E02Y_ID}.json").write_text("{}")
    monkeypatch.setattr(K, "E02Y_SHA", sha(pdir / f"{K.E02Y_ID}.json"))
    monkeypatch.setattr(K, "DECIDE_REPORT", str(tmp_path / "no_decide.json"))
    monkeypatch.setattr(S, "assert_order", lambda req, d: [] if looked else list(req))
    monkeypatch.setattr(SZ, "baseline_problems", lambda doc: [])
    man = {"m": 1}
    monkeypatch.setattr(SZ, "abs_files", lambda w: (None, None, man))
    f, base, blocks = _frame(seed=7)
    bdir = tmp_path / "base"
    bdir.mkdir()
    base.assign(window="B").to_csv(bdir / K.BASE_FILE, index=False)
    pd.DataFrame({"station_id": list(blocks), "block5": list(blocks.values())}).to_csv(
        tmp_path / "wells.csv", index=False)
    monkeypatch.setattr(K, "BASE_DIR", str(bdir))
    monkeypatch.setattr(K, "E02C_WELLS", str(tmp_path / "wells.csv"))
    looks = []
    monkeypatch.setattr(prereg3, "assert_can_score",
                        lambda *a, **k: looks.append(k.get("test_id")) or
                        {"post_registration_looks": [], "unverifiable_prior_looks": []})
    rep = _e02y_report(tmp_path, "B", **({"verdict": verdict} if verdict else {}))
    _step5(rep.parent, ids, manifest=man)
    _bind(rep)
    prod = _window_product(tmp_path, window)
    return prod, rep, pdir, looks


def _run(tmp_path, prod, rep, pdir, now=NOW_B):
    return SZ.score("B", prod, rep, now=now, out=str(tmp_path / "o.json"),
                    attempts=str(tmp_path / "a.jsonl"), prereg_dir=pdir)


def test_scorer_not_evaluable_when_e02y_did_not_look(tmp_path, monkeypatch):
    ids = [f"09{i:06d}" for i in range(3)]
    prod, rep, pdir, looks = _session(tmp_path, monkeypatch, ids, looked=False,
                                      verdict="not_evaluable", window=("2026-11", "2027-10"))
    r = _run(tmp_path, prod, rep, pdir)
    assert set(r["verdicts"].values()) == {"level_not_evaluable"} and looks == []
    assert r["looked"] is False and "does not cover" in r["precondition"]["why"]
    assert r["window_product_manifest_sha256"] == sha(prod / "MANIFEST.json")


def test_scorer_refuses_a_wrong_product_once_e02y_looked(tmp_path, monkeypatch):  # review S3
    ids = [f"09{i:06d}" for i in range(3)]
    prod, rep, pdir, looks = _session(tmp_path, monkeypatch, ids,
                                      window=("2026-11", "2027-10"))
    with pytest.raises(SZ.Refused, match="engineering error"):
        _run(tmp_path, prod, rep, pdir)
    assert looks == [] and not (tmp_path / "o.json").exists()


def test_scorer_reads_every_input_before_the_look(tmp_path, monkeypatch):    # review S7
    ids = [f"09{i:06d}" for i in range(3)]
    prod, rep, pdir, looks = _session(tmp_path, monkeypatch, ids)
    (tmp_path / "wells.csv").unlink()
    with pytest.raises(SZ.Refused, match="pre-look inputs"):
        _run(tmp_path, prod, rep, pdir)
    assert looks == []


def test_scorer_refuses_a_bad_rows_contract_before_the_look(tmp_path, monkeypatch):
    ids = [f"09{i:06d}" for i in range(3)]
    prod, rep, pdir, looks = _session(tmp_path, monkeypatch, ids)
    ev = rep.parent / "step5" / "evaluation.csv"
    t = pd.read_csv(ev, dtype={"station_id": str})
    t.assign(station_id=t["station_id"].str.lstrip("0")).to_csv(ev, index=False)
    _bind(rep)
    with pytest.raises(SZ.Refused, match="rows contract"):
        _run(tmp_path, prod, rep, pdir)
    assert looks == [] and not (tmp_path / "o.json").exists()


def test_scorer_environment_failure_after_look_is_rerun(tmp_path, monkeypatch):  # review Q4
    ids = [f"09{i:06d}" for i in range(3)]
    prod, rep, pdir, looks = _session(tmp_path, monkeypatch, ids)
    monkeypatch.setattr(SZ, "rows_contract_problems", lambda *a: [])

    def gone(*a, **k):
        raise FileNotFoundError("evaluation.csv vanished")

    monkeypatch.setattr(S, "load_e02_step5", gone)
    with pytest.raises(SZ.Refused, match="rerun with unchanged code"):
        _run(tmp_path, prod, rep, pdir)
    assert looks == [K.TEST["B"]] and not (tmp_path / "o.json").exists()
    assert _events(tmp_path / "a.jsonl")[-1] == "environment_failure_after_look"


def test_scorer_after_look_failure_is_final_and_disclosed(tmp_path, monkeypatch):
    ids = [f"09{i:06d}" for i in range(3)]
    prod, rep, pdir, looks = _session(tmp_path, monkeypatch, ids)
    monkeypatch.setattr(SZ, "rows_contract_problems", lambda *a: [])

    def boom(*a, **k):
        raise S.EngineeringFailure("inconsistent step (5)")

    monkeypatch.setattr(S, "load_e02_step5", boom)
    late = dt.datetime(2029, 2, 3, tzinfo=dt.timezone.utc)
    r = _run(tmp_path, prod, rep, pdir, now=late)
    assert looks == [K.TEST["B"]] and r["session_breach"] is True
    assert set(r["verdicts"].values()) == {"engineering_failure_after_look"}
    assert "inconsistent" in r["error"] and set(r["versions"]) >= {"numpy", "pandas"}
    with pytest.raises(SZ.Refused, match="scored once"):
        _run(tmp_path, prod, rep, pdir, now=late)


def test_scorer_full_synthetic_window_and_late_decidability(tmp_path, monkeypatch):
    f, base, blocks = _frame(seed=7)
    ids = sorted(f["station_id"].unique())
    prod, rep, pdir, looks = _session(tmp_path, monkeypatch, ids)
    monkeypatch.setattr(SZ, "rows_contract_problems", lambda *a: [])
    monkeypatch.setattr(S, "load_e02_step5", lambda *a, **k: (f, {"passed": True}, {"rows": 1}))
    dec = tmp_path / "decide.json"                                         # review B1
    dec.write_text(json.dumps({"created": "2029-01-16T00:00:00+00:00", "readings": {
        "B": {c: {"reading": "not_shown_weak", "P": 0.9} for c in K.COUNTIES}}}))
    monkeypatch.setattr(K, "DECIDE_REPORT", str(dec))
    r = _run(tmp_path, prod, rep, pdir)
    assert r["verdicts"] == {"Yunlin": "level_supported", "Changhua": "level_supported"}
    assert r["seed"] == SEED and r["page_mapping"]["C30-fc"].startswith("untouched")
    assert r["decidability_report_in_time"] is False
    assert all(d["reading"] == "not_run" for d in r["decidability"].values())
    assert "S-limited" in r["page_mapping"]["Yunlin"] and looks == [K.TEST["B"]]
    assert _events(tmp_path / "a.jsonl")[-1] == "score_written"


# ----------------------------------------------------------------------------- export
def test_export_build_on_synthetic_matrix():
    pytest.importorskip("hydrophysics.twin.drift_diag")
    import export_baselines_e02z as EX

    ids = [f"09{i:06d}" for i in range(K.N_WELLS)]
    ax = pd.date_range("2012-01-01", "2029-10-01", freq="MS")
    rng = np.random.default_rng(2)
    obs = np.full((len(ids), len(ax)), np.nan)
    obs[:, :K.T_FIT] = rng.normal(size=(len(ids), K.T_FIT))
    obs[:, K.T_FIT:] = 99.0                    # never used: the export slices to 2022-12
    pred = {w: pd.DataFrame({"station_id": np.repeat(ids, 12),
                             "date": np.tile([str(m.date()) for m in L.window_months(w)],
                                             len(ids)), "baseline_m": 0.0})
            for w in K.WINDOWS}
    obs_clean = obs.copy()
    obs_clean[:, K.T_FIT:] = np.nan
    t, ref = EX.build(pred, obs_clean, ax)
    assert len(t) == 2 * 12 * K.N_WELLS and set(t["window"]) == {"B", "C"}
    assert set(ref) == {"B", "C"}
    t2, _ = EX.build(pred, obs, ax)            # post-2022 values never reach a baseline
    pd.testing.assert_frame_equal(t, t2)


# ----------------------------------------------------------------------------- draft
def test_draft_needs_a_fresh_seed():
    with pytest.raises(L.Refused):
        P.draft_prereg(20261031, ids=[str(i) for i in range(K.N_WELLS)], check_files=False)
    with pytest.raises(TypeError):
        P.draft_prereg(ids=[], check_files=False)        # the seed has no default


def _draft():
    return P.draft_prereg(SEED, ids=[f"x{i:03d}" for i in range(K.N_WELLS)], check_files=False)


def test_draft_key_fields():
    d = _draft()
    assert d["id"] == K.ID and d["kind"] == "test" and d["wp"] == K.WP
    assert [t["test_id"] for t in d["tests"]] == [K.TEST["B"], K.TEST["C"]]
    assert d["fresh_units"] == [K.UNIT["B"], K.UNIT["C"]]
    assert d["scoring"]["B"]["not_before"] == "2029-01-15T00:00:00Z"
    assert d["scoring"]["C"]["not_before"] == "2030-01-15T00:00:00Z"
    assert d["scoring"]["registration_deadline"] == K.REG_DEADLINE
    assert "covering_rule" in d["scoring"]["precondition"]
    pm = d["page_mapping"]
    assert "C30-scen" in pm["claim"] and "C30-fc" in pm["claim"]
    assert any("C30-fc is never moved" in x for x in pm["never"])
    assert pm["expected_outcome"]["rc5"]["frozen_run_rmse_m"] == 2.048
    assert "ONCE" in d["attempt_policy"]
    assert d["thresholds"]["boot_seed"] == SEED == d["decidability"]["seed"]
    assert d["thresholds"]["min_blocks_county"] == 10
    assert "no CPU fallback" in d["decidability"]["forwards"]["device"]
    assert d["decidability"]["forwards"]["reference_forwards_sha256"] == {
        K.fv1_npz(t): K.FV1_FWD_SHA[t] for t in (K.ZR0, *K.TAGS)}
    assert "decide SEED" in d["decidability"]["runner"] and "--seed" not in d[
        "decidability"]["runner"]
    assert "before 2029-01-15" in d["decidability"]["when"]
    assert set(d["environment"]) >= {"numpy", "pandas"}
    assert "pooled" in d["page_mapping"]["expected_outcome"]["why"]
    assert d["decidability"]["k_truth"] == K.K_TRUTH
    assert set(d["disclosures"]) >= {"multiplicity", "U7", "same_origin", "builder_reads"}
    assert "R8.14" in d["rulings"] and "R7.heads_years23" in d["rulings"]
    assert all(t["metric"]["interval"].count(str(SEED)) == 1 for t in d["tests"])
    DZ.check_registered(d, SEED)                          # the engine accepts its own design


@NEED_REGISTRY
def test_draft_validates_against_prereg3_with_e02y_ids():
    d = P.draft_prereg(SEED, check_files=False)
    ids = d["tests"][0]["external_units"][K.UNIT["B"]]["ids"]
    assert len(ids) == K.N_WELLS == len(set(ids))
    assert P.validate(d) == []
    assert f"{P.PREREG}/{K.E02Y_ID}.json" in d["inputs"]


@NEED_REGISTRY
def test_draft_with_files_refuses_without_exported_baselines():
    if all((REPO / K.BASE_DIR / n).is_file() for n in K.BASE_FILES):
        pytest.skip("baselines exported")
    with pytest.raises(ValueError, match="export the baselines first"):
        P.draft_prereg(SEED)


def test_draft_never_writes():
    src = (X / "prereg_e02z.py").read_text(encoding="utf-8")
    assert "write_prereg(" not in src.replace("write_prereg is never called", "")


# ----------------------------------------------------------------------------- runner
@pytest.mark.skipif(shutil.which("bash") is None, reason="no bash")
def test_runner_dry_forward_commands():
    r = subprocess.run(["bash", str(X / "run_e02z.sh"), "forward"], env={**os.environ,
                       "DRY": "1"}, capture_output=True, text=True, check=True)
    lines = [x for x in r.stdout.splitlines() if "hydrophysics.twin.forward" in x]
    assert len(lines) == 5
    assert all("--horizon 82" in x and "--recorded-pumping" not in x for x in lines)
    assert "e02z_zr0" in lines[0] and lines[0].count("--theta") == 3
    assert all(x.count("--theta") == 1 for x in lines[1:])
