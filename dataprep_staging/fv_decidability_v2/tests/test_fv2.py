"""CPU tests of FV.decidability.v2_postfailure, FV.ext.v1 and FV.stat.v1 (no GPU, no forward
run, nothing registered on the real registry, no look logged there, no observation opened).
Synthetic inputs only; the real registry and results/ tree are read (never written) by the
declaration checks, and only where present.

    cd <repository root>
    ~/miniforge3/envs/hydro/bin/python -m pytest -q -p no:cacheprovider \\
        dataprep_staging/fv_decidability_v2/tests
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

TESTS = Path(__file__).resolve().parent
X = TESTS.parent
REPO = X.parents[1]
for _p in (str(X), str(REPO / "dataprep_staging" / "fv_decidability"),
           str(REPO / "dataprep_staging" / "e02c"), str(REPO / "dataprep_staging"), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ext_fv as XF  # noqa: E402
import forcing_year as FYM  # noqa: E402
import osse_fv as O  # noqa: E402
import osse_fv2 as V2  # noqa: E402
import prereg_fv2 as PR  # noqa: E402
import stat_fv as ST  # noqa: E402

SMALL = V2.Plan(k_truth=6, batch=4)


# ----------------------------------------------------------------------------- static
def test_no_home_paths_or_usernames_in_staged_files():
    pat = re.compile(r"(/home/|-home-)[A-Za-z0-9_]")
    for p in sorted(X.rglob("*")):
        if p.is_file() and p.suffix in (".py", ".sh", ".json", ".md"):
            assert not pat.search(p.read_text(encoding="utf-8")), p


def test_future_annotations_everywhere():
    for p in sorted(X.rglob("*.py")):
        assert "from __future__ import annotations" in p.read_text(encoding="utf-8"), p


def test_runner_carries_the_declared_recipe():
    assert PR.runner_problems() == []
    assert {V2.SELF, V2.FY, V2.RUNNER, O.SELF} <= set(V2.FROZEN)
    assert O.RUNNER not in V2.FROZEN


def test_v1_thresholds_are_byte_identical():
    assert (V2.P_WEAK, V2.K_TRUTH, V2.K_NULL, V2.SEED, V2.REPRO_TOL_M) == (
        0.5, 250, 500, O.SEED, 0.02)
    assert V2.V1_NULL_TOLERANCE == 0.006000375877185671
    assert abs(O.null_tolerance(np.full((10, 12), 3.0), np.full(10, 0.1), np.full(10, 0.5),
                                500)["tolerance"] - 3 / 500) < 1e-12
    assert V2.PLAN.n_batch * V2.BATCH == V2.K_TRUTH and V2.HINDCAST_TOL_M == 0.0
    assert V2.FWD_FLAGS == O.FWD_FLAGS and V2.WT_HEAD == O.WT_HEAD


def test_mc_check_is_non_binding_at_the_frozen_k():
    assert V2.mc_max_half_width(4, 250) < 0.027 and V2.mc_max_half_width(9, 500) < 0.013
    assert V2.mc_max_half_width(FYM.N_SEQ, FYM.K_RC) < 0.016
    assert max(V2.mc_max_half_width(4, V2.K_TRUTH), V2.mc_max_half_width(9, ST.K_STAT),
               V2.mc_max_half_width(FYM.N_SEQ, FYM.K_RC)) < V2.MC_HALF_WIDTH
    assert "non-binding" in V2.mc_note(4, 250)
    pooled = V2._pooled({t: {"r": (0.6, 0.02)} for t in O.TAGS}, lambda r: r["r"])
    assert np.isclose(pooled["p_minus_threshold_in_se"], 0.1 / 0.01)


def test_committed_disclosure_carries_no_observation_values():
    """The builder's accidental look is described, never quoted, in committed files."""
    txt = PR.BUILDER_LOOKS
    assert PR.BUILDER_LOOK_UNIT in txt and "never in a committed file" in txt
    assert not re.search(r"\b0\d{7}\b", txt)                        # no station id
    assert not re.search(r"\b\d\.\d{2}\b", txt)                    # no printed value
    for p in sorted(X.rglob("*.py")):
        assert "0.9" + "3 m" not in p.read_text(encoding="utf-8"), p


def test_readings_one_sided_and_mc_gate():
    ok = {"p": 0.5, "mc_half_width_90": 0.07}
    assert V2.reading(ok) == "weak_by_design"
    assert V2.reading({**ok, "p": 0.5000001}) == "not_shown_weak"
    assert V2.reading({**ok, "mc_half_width_90": 0.0700001}) is None
    txt = json.dumps([V2.WORDING, XF.WORDING, ST.WORDING], ensure_ascii=False).lower()
    assert "informative" not in txt and not re.search(r"(?<!never-)calibrated", txt)
    assert "post-failure" in json.dumps(V2.WORDING) and "post-failure" in json.dumps(
        FYM.WORDING)


# ----------------------------------------------------------------------------- worlds
def test_world_draws_are_v1_draws():
    pool = 1.0 + 0.1 * np.random.default_rng(0).standard_normal((5, 12))
    for _j, tag, d in O.check_plan():
        a, _ = V2.world_series(pool, tag, d)
        b, _ = O.draw_multipliers(pool, O.rng_for(O.STREAM_ANOM, O.TAGS.index(tag), d))
        assert np.array_equal(a, b)


def test_plan_layout():
    assert V2.PLAN.names(0)[:2] == ["baseline", "w000"] and len(V2.PLAN.names(4)) == 51
    assert V2.PLAN.names(4)[-1] == "w249"
    assert [list(SMALL.worlds(b)) for b in range(SMALL.n_batch)] == [[0, 1, 2, 3], [4, 5]]


def test_arm_files_are_forward_compatible(tmp_path):
    from hydrophysics.twin.recorded_pumping import load_recorded_multiplier, parse_recorded_spec

    pool = 1.0 + 0.1 * np.random.default_rng(1).standard_normal((5, 12))
    rec = V2.write_arms(tmp_path / "arms", pool, SMALL)
    with pytest.raises(FileExistsError):
        V2.write_arms(tmp_path / "arms", pool, SMALL)
    specs = V2.recorded_specs("Q0", 1, str(tmp_path / "arms"), SMALL)
    assert len(specs) == 2
    proj = pd.date_range("2023-01-01", periods=O.HORIZON, freq="MS")
    for s in specs:
        name, path, start, county = parse_recorded_spec(s)
        assert county == "yunlin" and name in SMALL.names(1)
        m = load_recorded_multiplier(path, start=start, county=county)
        m.check_window(proj)
        assert m.sha256 == rec["files"]["Q0"][name]["sha256"]
    V2.check_arms(tmp_path / "arms", pool, SMALL)
    p = tmp_path / "arms" / "R1c" / "w003.csv"
    p.write_text(p.read_text().replace("2023-01,", "2023-01,9"))
    with pytest.raises(O.EngineeringFailure, match="w003"):
        V2.check_arms(tmp_path / "arms", pool, SMALL)


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    out = tmp_path_factory.mktemp("smoke") / "run"
    return out, V2.run_smoke(out, k_truth=6, batch=4, k_null=12)


def test_smoke_end_to_end_on_nonlinear_worlds(smoke):
    """Responses with a strong quadratic term (v1's superposition would fail) score."""
    out, rep = smoke
    assert rep["status"] == "scored" and rep["smoke"] is True
    assert "superposition" not in rep and rep["null_world"]["pass"]
    for tag in O.TAGS:
        for b in rep["engineering"]["truths"][tag]:
            assert b["hindcast_max_abs_vs_v1_m"] == 0.0
    assert rep["engineering"]["repro"]["X5R1"]["pass"]
    assert all(c["identical"] for c in rep["engineering"]["evaluate_selfcheck"])
    pl = rep["P"]["level"]
    assert np.isclose(pl["p"], np.mean(list(pl["by_truth"].values())))
    assert rep["readings"]["level"] == V2.reading(pl)
    assert rep["v1"]["status"] == "not_evaluable"
    agree = rep["engineering"]["v1_check_agreement_report_only"]
    assert {a["world"] for a in agree["R0"]} == {0, 1}
    assert (out / "smoke_worlds.json").is_file()


def test_batch_hindcast_identity_is_gate_e(tmp_path):
    P = V2.build_synthetic(tmp_path / "syn", SMALL)
    p = P.batches["Q0"][1]
    with np.load(p, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d["heads_mean"] = d["heads_mean"].copy()
    d["heads_mean"][0, :, :, 5] += np.float32(0.001)
    np.savez_compressed(p, **d)
    with pytest.raises(O.EngineeringFailure, match="hindcast"):
        V2.score_worlds(P, plan=SMALL, smoke=True, k_null=2)


def test_batch_metadata_is_gate_e(tmp_path):
    P = V2.build_synthetic(tmp_path / "syn", SMALL)
    p = P.batches["R0"][0]
    with np.load(p, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    opts = json.loads(str(d["forward_options"]))
    opts["recorded_pumping"]["w001"]["sha256"] = "0" * 64
    d["forward_options"] = json.dumps(opts)
    np.savez_compressed(p, **d)
    with pytest.raises(O.EngineeringFailure, match="w001"):
        V2.score_worlds(P, plan=SMALL, smoke=True, k_null=2)


def test_check_attempts(tmp_path):
    f = tmp_path / "x.npz"
    f.write_bytes(b"abc")
    att = tmp_path / "attempts.jsonl"
    att.write_text(json.dumps({"event": "forward", "truth": "R0", "batch": 2, "rc": 0,
                               "npz_sha256": O.sha256(f), "attempt": 3}) + "\n")
    assert V2.check_attempts(att, {("R0", 2): f})["R0_b2"]["attempt"] == 3
    with pytest.raises(O.EngineeringFailure):
        V2.check_attempts(att, {("R0", 1): f})


# ----------------------------------------------------------------------------- reading 2
@pytest.fixture(scope="module")
def forcing(tmp_path_factory):
    out = tmp_path_factory.mktemp("forcing") / "run"
    return out, FYM.run_smoke(out, k_rc=10, k_null=100)


def test_forcing_smoke(forcing):
    out, rep = forcing
    assert rep["status"] == "scored" and rep["null_world"]["pass"]
    assert rep["sigma_method"] == "recipe"
    assert set(rep["rc"]) == {"Yunlin", "Changhua"}
    for c in ("Yunlin", "Changhua"):
        r = rep["rc"][c]
        assert len(r["by_sequence"]) == FYM.N_SEQ
        assert np.isclose(r["rc"], np.mean(r["by_sequence"]))
        assert r["sampling_11yr"]["t90"][0] <= r["rc"] <= r["sampling_11yr"]["t90"][1]
        assert rep["readings"]["rc"][c] == FYM.rc_reading(r["rc"])
    assert rep["readings"]["rc"]["Yunlin"] == "omits"          # big synthetic forcing
    assert "92.6 %" in rep["wording"]["C14a"]["Yunlin"]["en"]
    assert "90 % interval over the 11 recorded years" in rep["wording"]["C14a"]["Yunlin"]["en"]
    assert "rain-year range only" in rep["wording"]["changhua"]["en"]
    assert "no Changhua pumping" in rep["wording"]["changhua"]["en"]
    assert rep["readings"]["c14b_trigger"] == "either_county"
    assert set(rep["lead_ratio"]) == {"Yunlin", "Changhua", "pooled"}
    assert "non-binding" in rep["mc_gate_note"]
    assert set(rep["lead_ratio"]["Yunlin"]) == set(FYM.LEADS)
    assert "not a predictive interval" in rep["wording"]["C14c"]["en"]
    assert rep["icc_year_report_only"]["Yunlin"]["report_only"]
    assert len(rep["engineering"]["replay"]) == FYM.N_SEQ


def test_rc_reading_thresholds():
    assert FYM.rc_reading(0.8499) == "omits"
    assert FYM.rc_reading(0.85) == "no_line" and FYM.rc_reading(0.8799) == "no_line"
    assert FYM.rc_reading(0.88) == "not_shown_to_omit" and FYM.rc_reading(None) is None


def test_null_coverage_matches_analytic():
    rng = np.random.default_rng(4)
    W = 60
    zr = rng.normal(0, 1, (W, 12))
    sd = rng.uniform(0.2, 0.8, W)
    sig = np.sqrt(sd ** 2 + 0.1)[:, None]
    lo, hi = zr - O.Z90 * sig, zr + O.Z90 * sig
    county = np.array(["Yunlin"] * 30 + ["Changhua"] * 30)
    cov = FYM.coverage_draws(zr, lo, hi, np.zeros((1, W, 12)), sd, np.full(W, 0.6), county,
                             400, FYM.STREAM_RC_NULL)
    for c in ("Yunlin", "Changhua"):
        an = FYM.analytic_coverage(zr, lo, hi, sd, county == c)
        assert abs(cov[c].mean() - an) < 0.01 and an > 0.9


def test_sigma_by_lead_and_declared_alternative():
    W, T = 5, FYM.T_NEED
    ens = np.tile(np.linspace(0.05, 0.5, T), (W, 1))
    sd = np.linspace(0.2, 0.6, W)
    sig = np.sqrt(ens ** 2 + sd[:, None] ** 2)[:, O.WIN[0]:O.WIN[1]]
    lo, hi = -O.Z90 * sig, O.Z90 * sig
    r = FYM.sigma_by_lead(lo, hi, ens)
    assert r["method"] == "recipe" and r["reconstruction_max_abs_m"] < 1e-12
    assert np.allclose(r["sigma"][:, -1], np.sqrt(0.25 + sd ** 2))
    r2 = FYM.sigma_by_lead(lo, hi, ens * 0.5)                  # does not reproduce
    assert r2["method"] == "sigma_const" and np.allclose(r2["sigma"][:, 0], sig.mean(axis=1))


def test_lead_ratio_and_icc_on_known_fields():
    n, W, T = FYM.N_SEQ, 4, FYM.T_NEED
    common = np.linspace(-1, 1, n)[:, None, None] * np.ones((n, W, T))
    county = np.array(["Yunlin", "Yunlin", "Changhua", "Changhua"])
    lr = FYM.lead_ratio(common, np.full((W, T), 2.0), county)
    sd = float(np.std(np.linspace(-1, 1, n), ddof=1))
    assert np.isclose(lr["Yunlin"]["47-58"]["median_ratio"], sd / 2.0)
    icc = FYM.icc_year(common[:, :, O.WIN[0]:O.WIN[1]], np.zeros(W), np.zeros(W), county)
    assert np.isclose(icc["Yunlin"]["icc_forcing_only"], 1.0)


def _replay_set(tmp_path):
    FP, idx = FYM.build_synthetic(tmp_path / "syn")
    return FP, idx, json.loads(idx.read_text())


def test_replay_index_checks(tmp_path):
    FP, idx, ix = _replay_set(tmp_path)
    wells = pd.read_csv(FP.v1.wells, dtype={"station_id": str}).sort_values("station_id")
    cells, layers = wells["cell"].to_numpy(), wells["layer"].to_numpy()
    rp = FYM.load_replay(idx, cells, layers)
    assert rp["F"].shape == (FYM.N_SEQ, len(cells), FYM.T_NEED)
    assert np.all(rp["F"][:, :, :O.T_FIT] == 0.0)
    bad = dict(ix)
    bad["sequences"] = list(reversed(ix["sequences"]))
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(O.EngineeringFailure, match="order"):
        FYM.load_replay(p, cells, layers)
    # a replay whose hindcast differs from the climatology run
    q = Path(ix["sequences"][3]["npz"])
    with np.load(q, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d["heads_mean"] = d["heads_mean"].copy()
    d["heads_mean"][..., 10] += np.float32(0.01)
    np.savez_compressed(q, **d)
    bad2 = dict(ix)
    bad2["sequences"] = [dict(r) for r in ix["sequences"]]
    bad2["sequences"][3]["sha256"] = O.sha256(q)
    p2 = tmp_path / "bad2.json"
    p2.write_text(json.dumps(bad2))
    with pytest.raises(O.EngineeringFailure, match="hindcast"):
        FYM.load_replay(p2, cells, layers)
    with pytest.raises(FileExistsError):
        FYM.build_index(idx, ix["climatology"]["npz"], [r["npz"] for r in ix["sequences"]])


def test_capture_records_set_the_rain_mode_and_are_checked(tmp_path):
    FP, idx = FYM.build_synthetic(tmp_path / "syn", fields="energy")
    ix = json.loads(idx.read_text())
    assert ix["rain"] == "climatology" and ix["sequences"][4]["years"][0] == 2016
    good = {"projection_forcing": "replay:2015-2022,2012-2014",
            "replay": {"years": FYM.seq_years(3, 10), "pumping": "detrended",
                       "fields": "energy+weather"}}
    assert FYM.check_capture(good, 3) == []
    assert FYM.check_capture(good, 4)                          # wrong sequence
    assert FYM.check_capture({**good, "replay": {**good["replay"], "pumping": "raw"}}, 3)
    assert FYM.check_capture({"projection_forcing": "climatology", "replay": None}, None) == []
    c, seqs = FYM.default_layout()
    assert c.endswith("fw_replay/C.npz") and seqs[0].endswith("W2012.npz")
    assert seqs[-1].endswith("W2022.npz")


def test_replay_unavailable_is_not_evaluable():
    rep = FYM.score_forcing(FYM.ForcingPaths(), unavailable="rank 11 not run")
    assert rep["status"] == "not_evaluable" and "rank 11" in rep["why"]
    assert rep["readings"] is None and "not evaluable" in rep["wording"]["en"]


def test_pumping_only_replay_gives_changhua_no_line(tmp_path):
    """Energy-only replay: no Changhua forcing exists in the model, so no reassuring line."""
    rep = FYM.run_smoke(tmp_path / "e", k_rc=6, k_null=60, fields="energy", amp_changhua=0.0)
    assert rep["status"] == "scored" and rep["forcing"]["rain"] == "climatology"
    assert rep["readings"]["rc"]["Changhua"] == FYM.NOT_REPRESENTED
    assert "Changhua" not in rep["wording"]["C14a"]
    ch = rep["wording"]["changhua"]["en"]
    assert "no Changhua forcing-year variation is represented" in ch
    assert "rain-year range only" not in ch
    txt = json.dumps(rep["wording"], ensure_ascii=False)
    assert "weather" not in txt and "pumping years only" in txt


def test_c14b_trigger_is_declared(tmp_path):
    decl = {"readings": {"reading_2": {"c14b_trigger": "pooled"}}}
    assert FYM.declared_c14b_trigger(decl) == "pooled"
    assert FYM.declared_c14b_trigger(None) == FYM.C14B_DEFAULT == "either_county"
    with pytest.raises(O.EngineeringFailure, match="c14b_trigger"):
        FYM.declared_c14b_trigger({"readings": {"reading_2": {"c14b_trigger": "PENDING"}}})


def test_unavailable_needs_a_predeclared_ground(tmp_path):
    import datetime as dt

    att = tmp_path / "attempts.jsonl"
    fw = tmp_path / "fw_report.json"
    early = dt.datetime(2026, 12, 1, tzinfo=dt.timezone.utc)
    late = dt.datetime(2027, 2, 1, tzinfo=dt.timezone.utc)
    assert FYM.unavailable_grounds(att, fw, now=early) == []
    assert FYM.unavailable_grounds(att, fw, now=late)                  # absent after 01-31
    fw.write_text(json.dumps({"verdict": "scored"}))
    assert FYM.unavailable_grounds(att, fw, now=late) == []
    fw.write_text(json.dumps({"verdict": "not_evaluable"}))
    assert FYM.unavailable_grounds(att, fw, now=early)
    fw.write_text(json.dumps({"verdict": "scored"}))
    FYM.log_replay_gate_e(att, "ix.json", "replay forwards: seq 3: hindcast differs")
    g = FYM.unavailable_grounds(att, fw, now=early)
    assert len(g) == 1 and "hindcast" in g[0]
    # with a declaration and no ground, the scorer refuses
    decl = {"readings": {"reading_2": {"c14b_trigger": "either_county"}}}
    with pytest.raises(O.EngineeringFailure, match="pre-declared ground"):
        FYM.score_forcing(FYM.ForcingPaths(), unavailable="x", decl=decl,
                          attempts=tmp_path / "none.jsonl", now=early)


def test_gate_e_on_fw_content_is_a_replay_gate_e(tmp_path):
    FP, idx, ix = _replay_set(tmp_path)
    wells = pd.read_csv(FP.v1.wells, dtype={"station_id": str}).sort_values("station_id")
    cells, layers = wells["cell"].to_numpy(), wells["layer"].to_numpy()
    bad = dict(ix)
    bad["schema"] = "other"
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(O.EngineeringFailure) as e1:                  # FV-side: retry free
        FYM.load_replay(p, cells, layers)
    assert not isinstance(e1.value, FYM.ReplayGateE)
    q = Path(ix["sequences"][2]["npz"])
    with np.load(q, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d["heads_mean"] = d["heads_mean"].copy()
    d["heads_mean"][..., 7] += np.float32(0.01)
    np.savez_compressed(q, **d)
    bad2 = dict(ix)
    bad2["sequences"] = [dict(r) for r in ix["sequences"]]
    bad2["sequences"][2]["sha256"] = O.sha256(q)
    p2 = tmp_path / "bad2.json"
    p2.write_text(json.dumps(bad2))
    with pytest.raises(FYM.ReplayGateE, match="hindcast"):          # FW content: no retry
        FYM.load_replay(p2, cells, layers)


# ----------------------------------------------------------------------------- FV.ext.v1
def test_synthetic_cl_is_month_of_year_mean():
    y = np.tile(np.arange(12, dtype=float), (2, 11))
    cl = XF.synthetic_cl(y)
    assert np.array_equal(cl[0], np.array([10, 11, *range(10)], float))   # Nov..Oct


def test_ext_smoke(tmp_path):
    rep = XF.run_smoke(tmp_path / "ext", k_truth=6, batch=4, k_null=20)
    assert rep["status"] == "scored" and rep["null_world"]["pass"]
    assert "non-binding" in rep["mc_gate_note"]
    assert set(rep["truth_worlds"]) == set(O.TAGS)
    assert rep["readings"]["ext_level"] in ("weak_by_design", "not_shown_weak", None)
    assert all(c["identical"] for c in rep["engineering"]["evaluate_selfcheck"])
    assert rep["engineering"]["n_wells"] == 40


# ----------------------------------------------------------------------------- FV.stat.v1
def test_stat_grid_and_seeds():
    assert [(c["rho"], c["share"]) for c in ST.cells()][:2] == [(0.80, 0.0), (0.80, 0.15)]
    assert len(ST.cells()) == 9 and ST.SMOKE_SEED != ST.SEED
    assert sum(ST.E04_ORIGINS.values()) == sum(ST.E04_COUNTIES.values()) == 147
    assert ST.T_FIT == 178 and ST.WIN == (178, 190)


def test_ar1_unit_and_anomalies():
    x = ST.ar1_unit(400, 600, 0.86, np.random.default_rng(2))
    assert abs(x.var() - 1.0) < 0.05
    a, b = x[:, 1:].ravel(), x[:, :-1].ravel()
    assert abs(np.corrcoef(a, b)[0, 1] - 0.86) < 0.01
    county = np.array(["Yunlin"] * 200 + ["Changhua"] * 200)
    an = ST.anomalies(county, 0.86, 0.3, np.random.default_rng(3))
    r = np.corrcoef(an[:200, 100], an[200:, 100])
    assert an.shape == (400, ST.T_ALL) and np.isfinite(r).all()
    c = np.mean([np.corrcoef(an[0], an[k])[0, 1] for k in range(1, 50)])
    assert 0.15 < c < 0.45


def test_e04_frame_follows_the_registered_origins():
    lay = ST.e04_layout()
    y = ST.anomalies(lay.county, 0.86, 0.0, ST.rng_for(1, 0, 0, ST.SMOKE_SEED))
    f, info = ST.e04_frame(lay, y)
    assert info["n_kept"] == 147 and len(f) == 147 * 12
    assert np.allclose(f["upper_m"] - f["predicted_m"], f["predicted_m"] - f["lower_m"])


def test_stat_smoke_uses_the_smoke_seed(tmp_path):
    rep = ST.run_smoke(tmp_path / "stat", k_stat=3, k_t1=1)
    assert rep["status"] == "scored" and rep["seed"] == ST.SMOKE_SEED
    assert rep["engineering"]["rho_recovery"]["pass"]
    assert all(c["identical"] for c in rep["engineering"]["evaluate_selfcheck"])
    assert len(rep["e04_cells"]) == 9 and len(rep["i1_t1_t2_report_only"]["cells"]) == 9
    assert "readings" not in rep["i1_t1_t2_report_only"]
    assert set(rep["readings"]) >= {"e04", "wording"} and "t1" not in rep["readings"]
    assert rep["builder_preview"] is None                       # smoke seed: no preview note
    assert all("previewed at K = 8" in ST.WORDING[k]["en"]
               for k in ("weak_by_design", "not_shown_weak"))


# ----------------------------------------------------------------------------- declarations
@pytest.mark.parametrize("which", ["v2", "ext", "stat"])
def test_declarations_validate_and_wait_for_the_lead(tmp_path, which):
    from harness import prereg3

    d, pend = PR.build(which, prereg_dir=tmp_path)
    d = json.loads(json.dumps(d, default=O._jsonable))
    assert prereg3.validate_prereg(d) == []
    assert d["kind"] == "declaration" and pend                     # tmp registry: not ready
    assert PR.TESTS not in d["inputs"] and d["disclosures"]["tests"]["path"] == PR.TESTS
    assert "accident" in d["disclosures"]["builder"]
    if which == "v2":
        assert d["label"] == "post-failure, exploratory"
        assert d["thresholds"]["null_tolerance"] == V2.V1_NULL_TOLERANCE
        assert d["attempt_policy"]["exploratory_followups_max"] == 0
        assert any("--replay-status" in x for x in pend)
        assert "superposition" not in json.dumps(d["gates"]).lower()
    if which == "stat":
        assert "DECLARED seed" in d["disclosures"]["builder_smoke"]
        assert any(PR.STAT_PREVIEW_UNIT in x for x in pend)
    if which == "ext":
        assert d["n3f"]["pinned"].startswith(XF.N3F_ID) and XF.PRODUCT_USE == (
            "design_time_evidence")
        assert any(XF.EXT_LOOK_UNIT in x for x in pend)
    assert any(PR.BUILDER_LOOK_UNIT in x for x in pend)


def test_builder_looks_are_logged_in_a_tmp_registry(tmp_path):
    from harness import prereg3

    rows = tmp_path / "rows"
    rows.mkdir()
    with pytest.raises(ValueError, match="--station"):
        PR.log_builder_look("builder", prereg_dir=tmp_path, rows_dir=rows)
    assert PR.main(["--log-look", "builder", "--station", "00000001", "--note",
                    "synthetic values"], prereg_dir=tmp_path, rows_dir=rows) == 0
    assert PR.main(["--log-look", "stat_preview"], prereg_dir=tmp_path, rows_dir=rows) == 0
    assert PR.builder_look_problems(tmp_path, stat=True) == []
    with pytest.raises(prereg3.PreregistrationError, match="already"):
        PR.log_builder_look("stat_preview", prereg_dir=tmp_path, rows_dir=rows)
    _, pend = PR.build("stat", prereg_dir=tmp_path)
    assert not any("look" in x for x in pend)
    d, pend = PR.build("v2", prereg_dir=tmp_path, replay_status="t", c14b_trigger="pooled")
    assert d["readings"]["reading_2"]["c14b_trigger"] == "pooled"
    assert not any("c14b" in x or PR.BUILDER_LOOK_UNIT in x for x in pend)


def test_ext_look_is_logged_once_for_the_exact_export(tmp_path):
    P = XF.build_synthetic(tmp_path / "syn", SMALL)
    reg = tmp_path / "reg"
    rows = tmp_path / "rows"
    reg.mkdir()
    rows.mkdir()
    pd.read_csv(Path(P.ext_dir) / "predictions.csv").to_csv(
        Path(P.ext_dir) / "baselines.csv", index=False)
    assert XF.look_logged(reg, P.ext_dir) is None
    assert PR.ext_null_block(P.ext_dir, reg)["status"] == PR.PENDING   # no read before it
    XF.log_look(reg, rows, P.ext_dir)
    assert XF.look_logged(reg, P.ext_dir)["units"] == [XF.EXT_LOOK_UNIT]
    (Path(P.ext_dir) / "wells.csv").write_text(
        (Path(P.ext_dir) / "wells.csv").read_text() + "\n")
    assert XF.look_logged(reg, P.ext_dir) is None                      # other file: not it


def test_write_refuses_when_not_ready(tmp_path, capsys):
    sha = O.sha256(X / "prereg_fv2.py")
    rc = PR.main(["--decl", "stat", "--write", "--quiet", "--expect-self-sha", sha],
                 prereg_dir=tmp_path, rows_dir=tmp_path / "rows")
    assert rc == 1 and "REFUSED" in capsys.readouterr().out
    assert not (tmp_path / f"{ST.ID}.json").exists()


def test_write_and_bind_in_a_tmp_registry(tmp_path, monkeypatch):
    """A declaration registers in a tmp registry once, and the scorers' loader accepts it."""
    real = PR.build

    def ready(which, prereg_dir=None, replay_status=None, c14b_trigger=None):
        d, _ = real(which, prereg_dir, replay_status, c14b_trigger)
        d["inputs"] = [p for p in d["inputs"] if O.rp(p).is_file()]
        return d, []

    monkeypatch.setattr(PR, "build", ready)
    sha = O.sha256(X / "prereg_fv2.py")
    rows = tmp_path / "rows"
    rows.mkdir()
    assert PR.main(["--decl", "stat", "--write", "--quiet", "--expect-self-sha", sha],
                   prereg_dir=tmp_path, rows_dir=rows) == 0
    doc, dsha = V2.load_registered(ST.ID, ST.FROZEN, prereg_dir=tmp_path, rows_dir=rows)
    assert doc["id"] == ST.ID and len(dsha) == 64
    from harness import prereg3

    with pytest.raises(prereg3.PreregistrationError, match="already exists"):
        PR.main(["--decl", "stat", "--write", "--quiet", "--expect-self-sha", sha],
                prereg_dir=tmp_path, rows_dir=rows)
