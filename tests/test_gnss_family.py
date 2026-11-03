"""X5.gnss_seasonal_family.v1 (round-8 rank 11): the scale-free GNSS seasonal family screen,
on FABRICATED stand-in stations and records only. No test draws a world on the frozen model
side, runs G-sep or G0 on it, or opens a GNSS, pile or electricity value.

- constants: the deduplicated screen, report-only arms, bands, refused seeds (no default);
- estimator: median-of-years robustness, T1 scale-free / T2 scale-carrying, screen decision,
  the (a)-(d) table, cluster/far, U2 bound direction, qc_jumps months, pile feed choice, F1;
- engines: 64 declared cells, determinism across workers, seeds required;
- gates: the runners refuse without the registered declaration, with changed code, with a
  missing or unlogged G-sep record (monkeypatched paths; results/ is never written);
- draft registration: key fields, closure on a failed gate, validation (skipped when the
  gitignored results/ tree is absent).
"""
from __future__ import annotations

import inspect
import json
import math
import re
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

REPO = Path(__file__).resolve().parents[1]
D = REPO / "dataprep_staging" / "gnss_family"
for _p in (str(D), str(REPO / "dataprep_staging" / "sd_column"), str(REPO / "dataprep_staging")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gsf_const as K  # noqa: E402
import gsf_estimator as F  # noqa: E402
import gsf_synth as G  # noqa: E402

RESULTS = REPO / "results" / "dataprep"
HAVE_RESULTS = (RESULTS / "preregistration" / "INDEX.jsonl").exists() and \
    (REPO / K.V1_MODEL_JSON).exists()
needs_results = pytest.mark.skipif(not HAVE_RESULTS, reason="gitignored results/ tree absent")
SEED = 777


def _stack_from(stations: dict, members: list[str], obs_member: str, scale=None):
    E = pytest.importorskip("gsc_estimator")
    pairs = {}
    for s, st in stations.items():
        prep = E.prepare(np.asarray(st["m"]))
        mod = {mb: E.coefficients(prep, np.asarray(st["mod"][mb], float)) for mb in members}
        f = 1.0 if scale is None else scale(st)
        pairs[s] = {"years": prep["years"], "obs": f * mod[obs_member], "mod": mod}
    return F.stack(pairs, members, {s: st["zone"] for s, st in stations.items()})


# ----------------------------------------------------------------------------- constants
def test_screen_is_deduplicated_tier_ab():
    assert K.SCREEN == ("cwa", "cwa_tau30", "page", "page_aquitard", "page_tau30", "q0",
                        "q0_cs", "q0_csk01_s29")
    assert K.DEDUP == {"q0_s29": "q0", "q0_csk1": "q0_cs"}
    assert K.REPORT_ONLY_ARMS == ("ccz", "q0_nh", "q0c_csk1", "x5r1_cs")
    assert not set(K.SCREEN) & set(K.REPORT_ONLY_ARMS)
    assert set(K.SCREEN) | set(K.DEDUP) | set(K.SCREEN_REPORT_ONLY) == set(K.TIER_AB)


def test_bands_and_removal_are_declared():
    assert pytest.approx((-math.log(1.25), math.log(1.25))) == K.T1_BAND
    assert pytest.approx((-math.log(1.25), math.log(1.42) + math.log(1.25))) == K.T2_BAND
    assert K.REMOVAL == "ngl_offfan" and K.USE == "design_time_evidence"
    assert K.USE in K.GATE_EXEMPT
    assert K.RULING_LABEL == "S (design-time, window flagged, outcome partly known)"


def test_seeds_have_no_default_and_v1_seeds_are_refused():
    import gsf_gates as GT
    import prereg_gsf as P

    for fn in (G.run_g0, G.run_gsep, GT.draft_declaration, P.draft_prereg):
        p = inspect.signature(fn).parameters["seed"]
        assert p.default is inspect.Parameter.empty, fn.__name__
    for bad in (None, True, 1.5, "1"):
        with pytest.raises(TypeError):
            K.check_seed(bad)
    for bad in (20261101, 20261102):
        with pytest.raises(ValueError):
            K.check_seed(bad)
    assert K.check_seed(SEED) == SEED


# ----------------------------------------------------------------------------- estimator
def test_median_vector_resists_one_bad_year():
    V = np.array([[0.010, 0.002], [0.011, 0.001], [0.200, -0.300]])
    assert F.amplitude(V) == pytest.approx(math.hypot(0.011, 0.001))
    assert abs(F.amplitude(V) - np.hypot(*V.mean(axis=0))) > 0.05


def test_t1_cancels_a_common_frame_scale_t2_carries_it():
    st = G.standin_stations(3)
    mem = ["page", "low", "near"]
    a = F.point(_stack_from(st, mem, "page"))
    b = F.point(_stack_from(st, mem, "page", scale=lambda s: 1.42))
    assert b["T1"][0] == pytest.approx(a["T1"][0], rel=1e-9)
    assert b["T2"][0] == pytest.approx(1.42 * a["T2"][0], rel=1e-9)


def test_screen_keeps_the_true_member_and_drops_the_far_one():
    st = G.standin_stations(3)
    mem = ["page", "low", "near"]
    res = F.screen(_stack_from(st, mem, "page"), np.random.default_rng(1), 300)
    assert "page" in res["survivors"] and "near" in res["survivors"]
    assert res["members"]["low"]["excluded"]
    ex = F.screen_fast(_stack_from(st, mem, "page"), np.random.default_rng(1), 300)
    assert list(ex) == [res["members"][m]["excluded"] for m in mem]


def test_t2_band_tolerates_the_frame_over_read_only_upward():
    st = G.standin_stations(4)
    mem = ["page", "low", "near"]
    up = F.screen(_stack_from(st, mem, "page", scale=lambda s: 1.42),
                  np.random.default_rng(2), 300)
    assert not up["members"]["page"]["T2"]["excluded"]
    down = F.screen(_stack_from(st, mem, "page", scale=lambda s: 0.5),
                    np.random.default_rng(2), 300)
    assert down["members"]["page"]["T2"]["excluded"]


def test_reading_table_is_exhaustive():
    c, f = ["page", "q0"], ["q0_csk01_s29"]
    assert F.reading(["page"], c, f) == "a"
    assert F.reading(["q0_csk01_s29"], c, f) == "b"
    assert F.reading(["q0", "q0_csk01_s29"], c, f) == "c"
    assert F.reading([], c, f) == "d"


def test_cluster_far_and_far_pairs_from_cut30():
    cut = {"page": 1.40, "q0_cs": 1.16, "q0_csk01_s29": 0.471, "cwa": 1.42}
    cf = F.cluster_far(cut, list(cut))
    assert cf == {"cluster": ["cwa", "page", "q0_cs"], "far": ["q0_csk01_s29"]}
    fp = F.far_pairs(cut, list(cut))
    assert fp["page"] == ["q0_csk01_s29"] and fp["q0_csk01_s29"] == ["cwa", "page", "q0_cs"]
    assert F.screened_range(cut, ["page", "q0_cs"]) == [1.16, 1.40]


def test_u2_bound_is_per_site_and_an_upper_bound():
    rng = np.random.default_rng(5)
    mod = np.array([[0.010, 0.003], [0.011, 0.002], [0.009, 0.004]])
    s, share = 1.2, 0.7
    sites = {"TKJS": {"gnss": {"obs": s * mod, "mod": mod},
                      "pile": {"obs": share * mod, "mod": mod}, "gated": True},
             "NTUH": {"gnss": {"obs": 5 * mod, "mod": mod},
                      "pile": {"obs": share * mod[:1], "mod": mod[:1]}, "gated": False}}
    r = F.u2_bound(sites, rng, 200)
    assert r["bound_site"] == "TKJS"                     # report-only NTUH never read
    assert r["sites"]["TKJS"]["s_point"] == pytest.approx(s / share)
    assert r["s_max"] >= s and r["sites"]["NTUH"]["gated"] is False
    assert F.u2_bound({}, rng, 10)["evaluable"] is False
    only_ro = {"NTUH": sites["NTUH"]}
    assert F.u2_bound(only_ro, rng, 10)["evaluable"] is False


def test_spring_mask_and_u2_constants():
    m = np.arange(12 * 2017, 12 * 2018)
    assert list(np.flatnonzero(~F.non_spring(m))) == [2, 3, 4]
    assert K.MIN_YEARS_U2 == 2 and K.PILE_VARIANT == "preorigin"
    assert K.PILES["NTUH(1)"]["gnss_cell"] == 771 and K.PILES["NTUH(1)"]["pile_cell"] == 769


def test_qc_jump_months_and_drop():
    dates = pd.date_range("2021-03-01", "2021-04-30", freq="D").strftime("%Y-%m-%d")
    v = np.zeros(len(dates))
    v[5::2] = 0.05                                     # alternating 5 cm jumps in March
    v[31:] = 0.0
    daily = pd.DataFrame({"station": "TKJS", "date": dates, "value_raw": v,
                          "value": v, "flag": 0, "family": "wra"})
    q = F.qc_jump_months(daily)
    assert q == {("TKJS", "2021-03")}
    kept = F.drop_months(daily, q)
    assert set(kept["date"].str[:7]) == {"2021-04"}
    two = daily.copy()
    two["value_raw"] = 0.0
    two.loc[10, "value_raw"] = 0.05
    assert F.qc_jump_months(two) == set()               # 2 jumps < 3: kept


def test_pile_monthly_one_feed_units_and_variant():
    rows = []
    for d in pd.date_range("2017-01-01", "2017-02-28", freq="D").strftime("%Y-%m-%d"):
        rows.append({"sensor_id": "TKJS(1)", "feed": "raw", "date": d, "value": 2.0, "flag": 0,
                     "variant": "preorigin"})
        rows.append({"sensor_id": "TKJS(1)", "feed": "raw", "date": d, "value": 7.0, "flag": 0,
                     "variant": "full"})
    for d in pd.date_range("2017-01-01", "2017-01-12", freq="D").strftime("%Y-%m-%d"):
        rows.append({"sensor_id": "TKJS(1)", "feed": "vendor", "date": d, "value": 9.0,
                     "flag": 0, "variant": "preorigin"})
    m, y, feed = F.pile_monthly(pd.DataFrame(rows), "TKJS(1)")
    assert feed == "raw" and list(m) == [12 * 2017, 12 * 2017 + 1]
    assert np.allclose(y, 0.002)


def test_f1_beta_recovers_a_known_slope():
    m = np.arange(12 * 2020, 12 * 2023)
    rng = np.random.default_rng(0)
    e = rng.normal(size=m.size)
    elec = {"distal": dict(zip(m.tolist(), e.tolist(), strict=True))}
    y = 0.003 * e + 0.01 * np.cos(2 * np.pi * (m % 12) / 12)
    b = F.f1_beta({"A": (m, y)}, elec, {"A": "distal"})
    assert b["distal"] == pytest.approx(0.003, rel=0.05)
    assert math.isnan(b["mid"])


# ----------------------------------------------------------------------------- engines
def test_g0_cells_are_the_declared_grid():
    c = G.cells()
    assert len(c) == 64 and len({G.cell_key(x) for x in c}) == 64
    assert sum(x["stress"] == "none" for x in c) == 48
    assert {x["stress"] for x in c} == {"none", *K.STRESS}
    assert "proximal_seasonal_truth" in K.STRESS and K.N_REP >= 1000
    assert {x["model_err"] for x in c} == set(K.MODEL_ERR)


def test_g0_engine_deterministic_across_workers_on_a_standin():
    st = G.standin_stations(5)
    mem = ["page", "low", "near"]
    cut = {"page": 1.40, "low": 0.47, "near": 1.42}
    kw = dict(n_rep=3, B=40, cell_list=G.cells()[:2] + G.cells()[-2:], truths=["page", "low"])
    a = G.run_g0(st, mem, cut, seed=SEED, workers=1, **kw)
    b = G.run_g0(st, mem, cut, seed=SEED, workers=2, **kw)
    assert json.dumps(a, sort_keys=True, default=float) == \
        json.dumps(b, sort_keys=True, default=float)
    assert set(a["checks"]) == {"true_retained_every_cell", "far_excluded_pooled",
                                "far_excluded_worst_cell", "p_all_excluded_every_cell"}
    with pytest.raises(TypeError):
        G.run_g0(st, mem, cut, seed=None, **kw)


def test_gsep_engine_far_vs_every_cluster_member():
    st = G.standin_stations(6)
    cut = {"page": 1.40, "low": 0.47, "near": 1.42}
    r = G.run_gsep(st, ["page", "low", "near"], cut, seed=SEED, n_rep=3, B=40)
    assert set(r["separation"]) == {"low|page", "low|near"}
    assert r["n_pairs"] == 2 and r["pass"] == (r["n_separated"] == 2)
    assert r["need_T1"] == pytest.approx(r["hw_T1"] + math.log(1.25))
    assert r["need_T2"] == pytest.approx(r["hw_T2"] + math.log(1.25) + math.log(1.42))
    with pytest.raises(TypeError):
        G.run_gsep(st, ["page", "low", "near"], cut, seed=None)


def test_zone_frame_world_scales_distal_only():
    st = G.standin_stations(7)
    cell = {"stress": "zone_frame", "loading_mm": 0.0, "thermal_mm": 0.0, "sigma_mm": 1e-9,
            "jumps": "off", "model_err": 1e-12}
    keep = {s: np.ones(len(v["m"]), bool) for s, v in st.items()}
    W = G.worlds(st, "page", cell, np.random.default_rng(0), 2, keep, 0.0)
    for s, v in st.items():
        f = K.ZONE_FRAME[v["zone"]]
        assert np.allclose(W[s][0], f * v["mod"]["page"], atol=1e-6)


# ----------------------------------------------------------------------------- gates
def _fake_engineering():
    return {"pass": True, "cut30_cm": {"page": 1.4}, "cluster_far": {"cluster": ["page"],
                                                                     "far": ["q0_csk01_s29"]},
            "dedup": {}, "pile_series_digest": "x"}


def test_declaration_draft_fields_and_refusals():
    import gsf_gates as GT

    d = GT.draft_declaration(SEED, engineering=_fake_engineering(), engineering_sha256="e" * 64)
    assert d["kind"] == "declaration" and d["id"] == K.DECL_ID
    assert d["seeds"]["declared"] == SEED and d["seeds"]["g0"] == [SEED, G.STREAM_G0]
    assert d["design"]["g0"]["n_cells"] == 64 and d["design"]["g0"]["pass"] == K.G0_PASS
    assert d["rulings"][K.RULING_ID]["sha256"] == K.RULINGS_SHA
    with pytest.raises(SystemExit):
        GT.draft_declaration(SEED, engineering={"pass": False}, engineering_sha256="e")


def test_runners_refuse_without_registration(monkeypatch, tmp_path):
    import gsf_gates as GT
    from harness import prereg3

    def boom(*a, **k):
        raise prereg3.PreregistrationError("absent")

    monkeypatch.setattr(prereg3, "load_prereg", boom)
    for fn in (GT.run_gsep_once, GT.run_g0_once):
        with pytest.raises(SystemExit, match="not registered"):
            fn()


def test_runners_refuse_changed_code_and_unlogged_gsep(monkeypatch, tmp_path):
    import gsf_gates as GT
    from harness import prereg3

    fake = {"code_sha256": {f"{K.X}/gsf_synth.py": "0" * 64}, "engineering": {"sha256": "x"},
            "seeds": {"declared": SEED}}
    monkeypatch.setattr(prereg3, "load_prereg", lambda *a, **k: (fake, "h"))
    with pytest.raises(SystemExit, match="code changed"):
        GT.declared()
    monkeypatch.setattr(GT, "declared", lambda: (fake, "h"))
    monkeypatch.setattr(K, "GSEP_OUT", str(tmp_path / "gsep.json"))
    monkeypatch.setattr(K, "ATTEMPTS", str(tmp_path / "attempts.jsonl"))
    with pytest.raises(SystemExit, match="gsep.json missing"):
        GT.run_g0_once()
    (tmp_path / "gsep.json").write_text(json.dumps({"pass": True}))
    with pytest.raises(SystemExit, match="not the file G-sep logged"):
        GT.run_g0_once()
    (tmp_path / "gsep.json").write_text(json.dumps({"pass": False}))
    GT.log_attempt({"event": "gsep_run", "gsep_sha256": GT.sha256(tmp_path / "gsep.json")})
    with pytest.raises(SystemExit, match="G-sep failed"):
        GT.run_g0_once()


def test_closure_line_is_honest():
    import gsf_gates as GT

    s = GT.closure_line("G0", {"summary": {"true_retained_min": 0.7}, "checks": {}})
    assert "no GNSS amplitude was read" in s and "not registered" in s and "E24" in s
    s2 = GT.closure_line("G-sep", {"n_separated": 1, "n_pairs": 7})
    assert "G-sep failed" in s2 and "1 of 7" in s2


# ----------------------------------------------------------------------------- draft
def _fake_ms():
    names = sorted(G.standin_stations(1))
    return {"units": {"primary": names}}


def _fake_gates(g0_pass=True):
    return dict(gsep={"pass": True, "n_separated": 3}, gsep_sha256="g" * 64,
                g0={"pass": g0_pass, "checks": {}, "summary": {"true_retained_min": 0.95}},
                g0_sha256="z" * 64, decl_sha256="d" * 64,
                engineering=_fake_engineering(), engineering_sha256="e" * 64)


def test_draft_prereg_key_fields():
    import prereg_gsf as P

    d = P.draft_prereg(SEED, ms=_fake_ms(), **_fake_gates())
    assert d["kind"] == "test" and d["id"] == K.ID and d["seed"] == SEED
    assert [t["test_id"] for t in d["tests"]] == [K.T1_ID, K.T2_ID, K.DIR_ID, K.U2_ID]
    assert set(d["decision_table"]) == {"a", "b", "c", "d"}
    assert "beside, never replacing" in d["decision_table"]["a"]
    assert "Narrowing only" in d["decision_table"]["b"]
    assert d["rulings"][K.RULING_ID]["sha256"] == K.RULINGS_SHA
    assert d["decision_rule"]["label_on_pass"] == K.PASS_LABEL
    assert "decided by the frame-carrying T2" in K.PASS_LABEL
    assert d["disclosures"][0].startswith("DECIDED BY T2")
    assert any("C32" in x and "claims_v1.json" in x for x in d["disclosures"])
    assert d["attempt_policy"]["exploratory_followups_max"] == 0
    assert any("6.1 mm" in x for x in d["disclosures"])
    assert any("annual_amp_m" in x for x in d["disclosures"])
    assert any("R8:gnss_anes_chsg_stations_disclosure" in x for x in d["disclosures"])
    assert any("amp_census_easting" in x for x in d["disclosures"])


def test_draft_prereg_refuses_a_failed_gate():
    import prereg_gsf as P

    with pytest.raises(SystemExit, match="no GNSS amplitude was read"):
        P.draft_prereg(SEED, ms=_fake_ms(), **_fake_gates(g0_pass=False))


def test_draft_prereg_refuses_the_declaration_seed():
    import prereg_gsf as P

    with pytest.raises(ValueError, match="must differ"):
        P.draft_prereg(SEED, ms=_fake_ms(), decl_seed=SEED, **_fake_gates())


def test_scorer_outcome_lines_and_dir_mapping():
    pytest.importorskip("gsc_estimator")
    import score_gsf as SC

    for r in ("a", "b", "c", "d"):
        s = SC.outcome_line(r, ["page"], ["q0_csk01_s29"], K.SCREEN)
        assert "no correction factor" in s and not re.search(r"\d\.\d\d", s)
        if r != "d":
            assert "Decided by T2" in s
    for r in ("a", "b"):
        assert K.PASS_LABEL in SC.OUTCOME_TEXT[r]
    t = {"verdict": "larger", "r_A": 0.5, "ci90": [0.3, 0.6]}
    assert SC.dir_line(t, {"verdict": "inconclusive"})["C27_status"] == SC.DIR_LABEL
    assert SC.dir_line(t, {"verdict": "consistent"})["C27_status"] == "N"
    for v in ("smaller", "consistent", "inconclusive"):
        assert SC.dir_line({**t, "verdict": v}, {"verdict": "inconclusive"})["C27_status"] == "N"


def test_scorer_compute_on_fabricated_inputs():
    pytest.importorskip("gsc_estimator")
    import score_gsf as SC

    st = G.standin_stations(8)
    arms = list(K.SCREEN) + list(K.REPORT_ONLY_ARMS)
    series, stations = {}, {}
    for s, v in st.items():
        base = v["mod"]["page"]
        for a in arms:
            y = v["mod"]["low"] if a == "q0_csk01_s29" else base * (1.0 + 0.01 * len(a) % 3)
            series[f"{a}|{s}|m"] = v["m"]
            series[f"{a}|{s}|y"] = np.asarray(y, float)
        stations[s] = {"zone": v["zone"], "family": v["family"]}
    ms = {"units": {"primary": sorted(st)}, "stations": stations}
    obs = {s: (v["m"], np.asarray(v["mod"]["page"], float)) for s, v in st.items()}
    cut = {m: 1.3 for m in K.SCREEN} | {"q0_csk01_s29": 0.471, "page": 1.40}
    cf = F.cluster_far(cut, list(K.SCREEN))
    res = SC.compute(ms, series, {}, obs, {}, cf, SEED, None, obs_qc=obs, B=200)
    p = res["T1_screen"]
    assert "page" in p["survivors"] and "q0_csk01_s29" in p["excluded"]
    assert p["reading"] == "a" and p["excluded_by"]["q0_csk01_s29"]
    assert res["U2"]["evaluable"] is False
    assert res["report_only"]["qc_jumps_variant"]["screen"]["reading"] == "a"


# ----------------------------------------------------------------------------- hygiene
def test_staged_files_are_clean():
    pat = re.compile(r"(/home/|-home-)[A-Za-z0-9_]")
    for p in sorted(D.glob("*.py")) + [D / "run_gsf.sh"]:
        t = p.read_text()
        assert not pat.search(t), p.name
        if p.suffix == ".py":
            assert "from __future__ import annotations" in t, p.name
            assert "write_prereg(" not in t, p.name


# ----------------------------------------------------------------------------- results
@needs_results
def test_bound_files_and_products():
    import gsf_model as MD
    import prereg_gsf as P

    assert MD.sha256(K.V1_MODEL_JSON) == K.V1_MODEL_SHA
    assert MD.sha256(K.V1_G0) == K.V1_G0_SHA
    assert MD.sha256(K.SD_REPORT) == K.SD_REPORT_SHA
    assert P.rulings_problems() == []
    assert P.manifest_problems() == []


@needs_results
def test_declaration_validates():
    import gsf_gates as GT

    d = GT.draft_declaration(SEED, engineering=_fake_engineering(), engineering_sha256="e" * 64)
    assert GT.validate_declaration(d) == []


@needs_results
def test_pending_draft_validates():
    import prereg_gsf as P
    from harness import prereg3

    d = P.draft_prereg(918273645, pending=True)
    assert prereg3.validate_prereg(d) == []
    assert d["seed"] == 918273645 and d["frozen_code"] and d["gates"]["v1_g0oc"]["pass"] is False


def test_score_failure_after_the_look_writes_the_final_report(monkeypatch, tmp_path):
    pytest.importorskip("gsc_estimator")
    import gsf_model as MD
    import score_gsf as SC
    from harness import prereg3

    monkeypatch.setattr(K, "REPORT", str(tmp_path / "report.json"))
    monkeypatch.setattr(K, "ATTEMPTS", str(tmp_path / "attempts.jsonl"))
    doc = {"model_side": {"pile_series_digest": "d"}}
    monkeypatch.setattr(SC, "verify_registration", lambda *a: {"doc": doc, "sha256": "r"})
    monkeypatch.setattr(MD, "load_v1", lambda *a: ({}, {}))
    monkeypatch.setattr(MD, "load_supplement", lambda: ({"series_digest": "d"}, {}))
    monkeypatch.setattr(SC, "open_products", lambda: (None, None, None))
    monkeypatch.setattr(prereg3, "assert_can_score", lambda *a, **k: {
        "sha256": "r", "tests": [], "post_registration_looks": []})

    def boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(SC, "after_look", boom)
    assert SC.score(log=False) == 5
    rep = json.loads((tmp_path / "report.json").read_text())
    assert rep["status"] == "engineering_failure_after_look" and "disk full" in rep["error"]
    ev = [json.loads(x)["event"] for x in (tmp_path / "attempts.jsonl").read_text().split("\n")
          if x.strip()]
    assert ev == ["score_start", "score_looked", "score_failed_after_look"]
    assert SC.score(log=False) == 1                    # the final report blocks any rerun
