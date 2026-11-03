"""FB.family_power.v1, the FB BF test draft and its 2028 scorer (round-8 rank 16, FB part), on
SYNTHETIC inputs only (fb_lib.synthetic_u1 / synthetic_inputs, made-up scales and departures).

- the statistic: E02d's relative operator removes a county-month shock; the vectorised log-
  likelihood equals a scalar loop and inflates a partial well's variance by T / n; log BF is
  antisymmetric and 0 for identical hypotheses; hypothesis means on the realised mask (S4);
  score_pair's readings at B = 20;
- the world: draw_scaled equals U1's power_e02d.draw for a time-constant scale; the arms
  (base / forcing / month_missing gating; heavy / scale report-only); the sigma_idio units and
  lead gate (B3);
- lead ruling B1: only the family pair gates or registers; truth pairs are report-only and are
  refused by the registration rule, the BF draft and bf_problems;
- the gate on FABRICATED rows and the registration rule; an end-to-end run on a reduced grid;
- runner: seeds have no default; refused seeds (builder, U1, FV); unregistered, crash and
  engineering failures are logged; smoke seeds are logged; the CLI has no --out;
- drafts validate with prereg3.validate_prereg (injected inputs -> TEST.* ids) and carry the
  key clauses (E29 ledger only, no page line, N2C precedence, no_later_than 2028-01-31);
- score_fb.py: refusals (unregistered, before 2028-01-15, N2C report, bound GOV / E02c code),
  the GOV covering-rule precondition, the heads contract, the model-only datum check.

Checks that need the gitignored results/ tree skip when it is absent (CI).
"""
from __future__ import annotations

import copy
import datetime as dt
import importlib.util
import inspect
import json
import math
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

REPO = Path(__file__).resolve().parents[1]
FB_DIR = REPO / "dataprep_staging" / "fb_family"
for _p in (str(FB_DIR), str(REPO / "dataprep_staging")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fb_inputs as FI  # noqa: E402
import fb_io as io  # noqa: E402
import fb_lib as L  # noqa: E402
import power_fb as PW  # noqa: E402
import prereg_fb as PR  # noqa: E402
import score_fb as SC  # noqa: E402
from harness import prereg3  # noqa: E402

HAVE_RESULTS = (REPO / "results" / "dataprep" / "preregistration" / "INDEX.jsonl").is_file()
NEED_RESULTS = pytest.mark.skipif(not HAVE_RESULTS, reason="results/ (gitignored) is absent")
SEED = 11                         # a builder seed (PW.BUILDER_SEEDS): never registrable
FAM = L.FAMILY_PAIR
LATE = dt.datetime(2028, 1, 20, tzinfo=dt.timezone.utc)
EARLY = dt.datetime(2027, 6, 1, tzinfo=dt.timezone.utc)


def _design(seed=SEED, reduced=True):
    u1 = L.synthetic_u1(seed)
    if reduced:
        for c in L.COUNTIES:
            u1["design"]["counties"][c].update({"a_s": [0.1], "a_w": [0.2], "phi": [0.5]})
        u1["design"]["dropout"] = [0.0, 0.4]
    return L.design_spec(u1, "synthetic", {"x": "0" * 64}, seed)


def _validate(doc, tmp_path):
    return prereg3.validate_prereg(doc, units_registry=None, prereg_dir=tmp_path,
                                   registries=[])


def _wells():
    return {"Changhua": [f"07{i:06d}" for i in range(74)],
            "Yunlin": [f"09{i:06d}" for i in range(84)]}


def _power():
    return {"path": "results/dataprep/reports/FB.family_power.v1/result.json",
            "sha256": "0" * 64, "declaration_sha256": "1" * 64}


# ----------------------------------------------------------------------------- seeds
@pytest.mark.parametrize("fn", [L.design_spec, L.run, L.synthetic_u1, L.synthetic_inputs,
                                L.self_check, L.run_county, PR.draft_prereg_power,
                                PW.smoke, PW.run_registered, PW.design_now])
def test_seed_has_no_default(fn):
    p = inspect.signature(fn).parameters["seed"]
    assert p.default is inspect.Parameter.empty


def test_draft_power_rejects_non_int_seed():
    with pytest.raises(TypeError):
        PR.draft_prereg_power(None)
    with pytest.raises(TypeError):
        PR.draft_prereg_power(True)


@pytest.mark.parametrize("seed", [*PW.BUILDER_SEEDS, 20261101, 20261103, 20261104])
def test_run_refuses_builder_and_upstream_seeds(tmp_path, seed):
    out = tmp_path / "out"
    assert PW.run_registered(seed, prereg_dir=tmp_path, out=out) == 1
    row = json.loads((out / "attempts.jsonl").read_text().splitlines()[-1])
    assert row["outcome"] == "refused" and "refused" in row["why"]


# ----------------------------------------------------------------------------- statistic
def test_draw_scaled_is_u1_draw_for_constant_scale():
    d = _design(reduced=False)
    dc = d["counties"]["Yunlin"]
    s = np.asarray(dc["u1_scale"]) * 0.3
    for cell in ({"a_s": 0.07, "a_w": 0.3, "phi": 0.6, "heavy": False},
                 {"a_s": 0.14, "a_w": 0.0, "phi": 0.6, "heavy": True}):
        a = L.U1P.draw(np.random.default_rng(5), 6, 11,
                       {"scale": s.tolist(), "subgroups": dc["subgroups"]}, cell, 0.2)
        b = L.draw_scaled(np.random.default_rng(5), 6, 11, np.repeat(s[:, None], 11, 1),
                          dc["subgroups"], cell, 0.2)
        assert np.array_equal(np.isnan(a), np.isnan(b))
        assert np.allclose(a[np.isfinite(a)], b[np.isfinite(b)], rtol=0, atol=1e-12)
    m = L.draw_scaled(np.random.default_rng(1), 50, 11, s, dc["subgroups"],
                      {"a_s": 0.0, "a_w": 0.0, "phi": 0.0, "heavy": False}, 0.0, 0.2)
    assert 0.15 < np.isnan(m).mean() < 0.25


def test_common_shock_cancels_and_self_check():
    d = _design()
    inp = L.synthetic_inputs(d, SEED)
    dc = d["counties"]["Changhua"]
    cd = L.county_cd(dc, inp["Changhua"]["scale"])
    mus = L.mu_stack(inp["Changhua"]["mu"])
    v = np.full((2, dc["wells"]), 0.05)
    sc = L.self_check(cd, mus, v, SEED, 0)
    assert sc["ok"], sc
    x = np.random.default_rng(0).normal(size=(3, dc["wells"], 11))
    e1 = L.E.relative(x, cd["groups"])
    e2 = L.E.relative(x + 7.0 * np.random.default_rng(1).normal(size=(3, 1, 11)),
                      cd["groups"])
    assert np.nanmax(np.abs(e1 - e2)) < 1e-9


def test_loglik_reference_and_partial_inflation():
    rng = np.random.default_rng(3)
    D, W, K, G, T = 3, 25, 2, 2, 11
    ybar = rng.normal(size=(D, W))
    n = np.full((D, W), T)
    n[0, :5] = 6
    n[1, :3] = 5                  # below MIN_MONTHS: not scored
    ybar[2, 0] = np.nan
    means = rng.normal(size=(K, W))
    v = rng.uniform(0.1, 0.5, size=(G, W))
    LL, k = L.loglik(ybar, n, means, v, T)
    assert np.allclose(LL, L._loglik_reference(ybar, n, means, v, T))
    assert k.tolist() == [W, W - 3, W - 1]
    w, d, kk, g = 0, 0, 1, 0
    ve = v[g, w] * T / 6
    one = L.loglik(ybar[[d]][:, [w]], n[[d]][:, [w]], means[:, [w]], v[:, [w]], T)[0]
    want = -0.5 * math.log(2 * math.pi * ve) - (ybar[d, w] - means[kk, w]) ** 2 / (2 * ve)
    assert one[0, kk, g] == pytest.approx(want)
    m3 = np.broadcast_to(means[None], (D, K, W))
    assert np.allclose(L.loglik(ybar, n, m3, v, T)[0], LL)


def test_realised_means_follow_the_mask():
    d = _design()
    inp = L.synthetic_inputs(d, SEED)
    groups = d["counties"]["Changhua"]["groups"]
    mus = L.mu_stack(inp["Changhua"]["mu"])
    rng = np.random.default_rng(4)
    mask = rng.random((3, mus.shape[1], 11)) >= 0.3
    got = L.realised_means(mus, mask, groups)
    for i in range(3):
        for k in range(mus.shape[0]):
            want, _ = L.window_means(np.where(mask[i], mus[k], np.nan), groups)
            assert np.allclose(got[i, k], want, equal_nan=True)
    full = L.realised_means(mus, np.ones_like(mask), groups)
    assert np.allclose(full[0], L.hypothesis_means(inp["Changhua"]["mu"], groups))


def test_log_bf_antisymmetric_identical_and_family_marginal():
    rng = np.random.default_rng(5)
    LL = rng.normal(size=(4, 4, 3)) * 10
    a, b = [0, 1], [2, 3]
    assert np.allclose(L.log_bf(LL, a, b), -L.log_bf(LL, b, a))
    assert np.allclose(L.log_bf(LL, [1], [1]), 0.0)
    want = np.log(np.exp(LL[:, a, :]).reshape(4, -1).mean(1))
    assert np.allclose(L.log_marginal(LL, a), want)


def test_score_pair_readings():
    d = _design()
    inp = L.synthetic_inputs(d, SEED, sep_m=1.0)
    dc = d["counties"]["Changhua"]
    groups = dc["groups"]
    mu = inp["Changhua"]["mu"]
    mus = L.mu_stack(mu)
    v = np.full((1, dc["wells"]), 0.01)
    A = [L.TRUTHS.index("R0"), L.TRUTHS.index("Q0")]
    B = [L.TRUTHS.index("R1c"), L.TRUTHS.index("X5R1")]
    assert L.score_pair(mu["R0"], groups, mus, v, 11, A, B)["reading"] == "closer_to_R0Q0"
    assert L.score_pair(mu["R1c"], groups, mus, v, 11, A, B)["reading"] == \
        "closer_to_R1cX5R1"
    part = np.array(mu["R1c"], float)
    part[::3, ::2] = np.nan                       # per-month gaps: realised-mask means
    assert L.score_pair(part, groups, mus, v, 11, A, B)["reading"] == "closer_to_R1cX5R1"
    sparse = np.array(mu["R0"], float)
    sparse[L.MIN_WELLS - 1:] = np.nan
    r = L.score_pair(sparse, groups, mus, v, 11, A, B)
    assert r["reading"] == "not_evaluable" and r["wells_scored"] < L.MIN_WELLS
    lnb = math.log(L.B_STRONG)
    assert L.reading(lnb, 30) == "closer_to_R0Q0"
    assert L.reading(-lnb, 30) == "closer_to_R1cX5R1"
    assert L.reading(lnb - 1e-9, 30) == "not_decisive"
    assert L.reading(10.0, L.MIN_WELLS - 1) == "not_evaluable"


def test_ledger_line_wording_and_precedence():
    line = L.ledger_line("closer_to_R1cX5R1", "Changhua", 31.0, "not_supported")
    assert line.startswith("E29 (FB family BF, ledger only)")
    assert "the window's per-well relative pattern is closer to the {R1c,X5R1} forwards " \
           "(climatology forcing)" in line
    assert "this does not decide whether Changhua pumping matters or the response class " \
           "(N2C decides that)" in line
    assert "if FB and N2C disagree, N2C stands" in line and "'not_supported'" in line
    for r in L.LEDGER_TEXT:
        assert "N2C decides that" in L.ledger_line(r, "Yunlin", None, "x")


# ----------------------------------------------------------------------------- grids
def test_candidates_cells_and_family_only():
    cs = L.candidates()
    assert len(cs) == 14 and len({c["key"] for c in cs}) == 14
    assert L.family_keys() == [f"{FAM}@Changhua", f"{FAM}@Yunlin"]
    fam = L.candidate(f"{FAM}@Changhua")
    assert fam["A"] == ["R0", "Q0"] and fam["B"] == ["R1c", "X5R1"] and fam["gating"]
    assert sum(c["gating"] for c in cs) == 2
    with pytest.raises(ValueError):
        L.require_family(["Q0_vs_R1c@Yunlin"])
    with pytest.raises(KeyError):
        L.candidate("stack@Changhua")
    d = _design(reduced=False)
    for c in L.COUNTIES:
        dc = d["counties"][c]
        assert dc["n_working_cells"] == 18
        assert dc["n_gating_world_cells"] == 54 + 36 + 36
        assert dc["n_world_cells"] == 54 + 36 + 36 + 2 + 4
    wc = L.world_cells(d["counties"]["Yunlin"]["grid"])
    arms = {w["arm"]: w["gating"] for w in wc}
    assert arms == {"base": True, "forcing": True, "month_missing": True, "heavy": False,
                    "scale_x1.5": False, "scale_x2": False}
    assert all(w["month_missing"] == L.MONTH_MISSING for w in wc if w["arm"] == "month_missing")
    assert {w["scale"] for w in wc if not w["gating"] and not w["heavy"]} == {1.5, 2.0}


def test_forcing_response_shape():
    rng = np.random.default_rng(0)
    G = rng.normal(size=(L.HORIZON, 5, 11))
    pool = 1.0 + rng.normal(0, 0.1, (6, 12))
    R = L.forcing_response(np.random.default_rng(1), 4, G, pool)
    assert R.shape == (4, 5, 11) and np.isfinite(R).all()
    assert np.allclose(L.forcing_response(np.random.default_rng(1), 4, np.zeros_like(G), pool),
                       0.0)


# ----------------------------------------------------------------------------- gate
def _row(truth, gating, pc, pw, key=f"{FAM}@Changhua", arm="base"):
    return {"arm": arm if gating else "heavy", "truth": truth, "cell": 0, "gating": gating,
            "heavy": not gating, "a_s": 0.0, "a_w": 0.0, "phi": 0.0, "dropout": 0.0,
            "by_candidate": {key: {"p_correct_strong": pc, "p_wrong_strong": pw}}}


def test_decide_on_fabricated_rows():
    th = L.thresholds()
    key = f"{FAM}@Changhua"
    ok = [_row(t, True, 0.9, 0.01) for t in L.TRUTHS] + [_row("R0", False, 0.1, 0.5)]
    d = L.decide(ok, key, th)
    assert d["pass"] and not d["report_only"]
    assert d["report_only_arms"]["heavy"]["p_wrong_strong_max"] == 0.5
    low = ok + [_row("Q0", True, 0.79, 0.0, arm="month_missing")]
    d = L.decide(low, key, th)
    assert not d["pass"] and d["worst_correct"]["arm"] == "month_missing"
    wrong = ok + [_row("X5R1", True, 0.95, 0.051, arm="forcing")]
    d = L.decide(wrong, key, th)
    assert not d["pass"] and d["worst_wrong"]["truth"] == "X5R1"
    assert not L.decide([], key, th)["pass"]


def test_truth_pairs_are_report_only_and_never_register():
    th = L.thresholds()
    key = "Q0_vs_R1c@Yunlin"
    rows = [_row(t, True, 0.99, 0.0, key=key) for t in ("Q0", "R1c")]
    d = L.decide(rows, key, th)
    assert d["report_only"] and d["meets_thresholds_report_only"] and not d["pass"]
    rule = L.registration_rule({f"{FAM}@Changhua": {"pass": True}, key: d})
    assert rule["registrable"] == [f"{FAM}@Changhua"] and rule["deadline"] == L.REG_DEADLINE
    with pytest.raises(ValueError):
        L.registration_rule({key: {"pass": True}})
    assert "C20/C21" in L.registration_rule({})["if_none"]
    with pytest.raises(ValueError):
        PR.draft_prereg_bf([key], power=_power(), wells=_wells())
    doc = PR.draft_prereg_bf([f"{FAM}@Yunlin"], power=_power(), wells=_wells())
    assert any("only the family pair" in p for p in PR.bf_problems(doc, [key]))


# ----------------------------------------------------------------------------- run
def test_run_reduced_grid_end_to_end():
    d = _design()
    inp = L.synthetic_inputs(d, SEED, sep_m=0.8)
    res = L.run(d, inp, SEED, 20, 20, smoke=True, log=lambda *a: None)
    assert set(res["decisions"]) == {c["key"] for c in L.candidates()}
    for k, dec in res["decisions"].items():
        assert dec["report_only"] == (k not in L.family_keys())
        if dec["report_only"]:
            assert not dec["pass"]
    for c in L.COUNTIES:
        cr = res["counties"][c]
        assert cr["self_check"]["ok"]
        assert {r["arm"] for r in cr["rows"]} == {"base", "forcing", "month_missing", "heavy",
                                                  "scale_x1.5", "scale_x2"}
        v = np.asarray(cr["calibration"]["v"])
        assert v.shape == (1, d["counties"][c]["wells"]) and (v > 0).all()
        assert np.asarray(cr["mu"]["R0"]).shape == (d["counties"][c]["wells"], 11)
        assert len(cr["ids"]) == d["counties"][c]["wells"]
    assert set(res["registration"]["registrable"]) <= set(L.family_keys())
    assert res["design"] == d
    with pytest.raises(L.EngineeringFailure):
        L.run(d, inp, SEED + 1, 5, 5, smoke=True, log=lambda *a: None)
    bad = copy.deepcopy(inp)
    bad["Yunlin"]["scale"] = bad["Yunlin"]["scale"][:, :5]
    with pytest.raises(L.EngineeringFailure):
        L.run(d, bad, SEED, 5, 5, smoke=True, log=lambda *a: None)


def test_smoke_refuses_results_logs_seed_and_writes_once(tmp_path):
    with pytest.raises(io.Refused):
        PW.smoke(SEED, REPO / "results" / "fb_smoke", prereg_dir=tmp_path)
    out = tmp_path / "smoke"
    p = PW.smoke(SEED, out, prereg_dir=tmp_path, reps=2, cal=2, log=lambda *a: None)
    doc = json.loads(p.read_text())
    assert doc["smoke"] and doc["synthetic"] and doc["seed"] == SEED
    row = json.loads((out / "attempts.jsonl").read_text().splitlines()[0])
    assert row["cmd"] == "smoke" and row["seed"] == SEED
    with pytest.raises(FileExistsError):
        PW.smoke(SEED, out, prereg_dir=tmp_path, reps=2, cal=2, log=lambda *a: None)


def test_run_refuses_unregistered_crash_and_engineering_are_logged(tmp_path, monkeypatch):
    import osse_fv as FV

    (tmp_path / "INDEX.jsonl").write_text("")
    out = tmp_path / "out"
    assert PW.run_registered(20261199, prereg_dir=tmp_path, out=out) == 1

    def boom(seed):
        raise KeyError("synthetic crash")

    monkeypatch.setattr(PW, "seed_problem", boom)
    assert PW.run_registered(20261199, prereg_dir=tmp_path, out=out) == 3

    def eng(seed):
        raise FV.EngineeringFailure("synthetic FV engineering failure")

    monkeypatch.setattr(PW, "seed_problem", eng)
    assert PW.run_registered(20261199, prereg_dir=tmp_path, out=out) == 2
    rows = [json.loads(x) for x in (out / "attempts.jsonl").read_text().splitlines()]
    assert [r["outcome"] for r in rows] == ["refused", "crashed", "engineering_failure"]
    assert "not registered" in rows[0]["why"]
    assert not (out / "result.json").exists()


def test_cli_run_has_no_out():
    with pytest.raises(SystemExit):
        PW.main(["run", "--seed", "20261199", "--out", "/tmp/x"])


# ----------------------------------------------------------------------------- inputs
def test_window_departure_and_basis_are_fv_sliced():
    import osse_fv as FV

    rng = np.random.default_rng(2)
    W = 6
    hk = rng.normal(size=(W, FV.T_RUN))
    hr = rng.normal(size=(W, FV.T_RUN))
    mask = rng.random((W, FV.T_FIT)) < 0.5
    mask[:, 0] = True
    full = FV.departure(hk, hr, mask)
    got = FI.window_departure(hk, hr, mask)
    assert got.shape == (W, L.T_WIN) and np.allclose(got, full[:, 1:12])
    heads = rng.normal(size=(1 + FV.HORIZON, W, FV.T_RUN))
    G = FI.window_basis(heads)
    assert G.shape == (FV.HORIZON, W, L.T_WIN)
    assert np.allclose(G, FV.basis(heads)[:, :, FV.WIN[0] + 1:FV.WIN[1]])


def _ident_and_wells(d):
    rows = []
    for c in L.COUNTIES:
        dc = d["counties"][c]
        s = np.asarray(dc["u1_scale"]) * 0.4
        for i in range(dc["wells"]):
            rows.append({"station_id": f"{c[:1]}{i:04d}", "county": c,
                         "group": dc["groups"][i], "subgroup": dc["subgroups"][i],
                         "sd_fit_rel_m": s[i]})
    return pd.DataFrame(rows)


def test_u1_identity_gate():
    d = _design(reduced=False)
    w = _ident_and_wells(d)
    w = pd.concat([w, pd.DataFrame([{"station_id": "N0001", "county": "Nantou", "group": "x",
                                     "subgroup": "y", "sd_fit_rel_m": 1.0}])])
    got = FI.u1_identity(d, w)
    assert len(got["Changhua"]["ids"]) == 74 and len(got["Yunlin"]["ids"]) == 84
    bad = w.copy().reset_index(drop=True)
    bad.loc[0, "group"] = "Changhua:other"
    with pytest.raises(io.EngineeringFailure):
        FI.u1_identity(d, bad)
    bad = w.copy().reset_index(drop=True)
    bad.loc[3, "sd_fit_rel_m"] *= 1.01
    with pytest.raises(io.EngineeringFailure):
        FI.u1_identity(d, bad)


def _artefact(ident):
    rows = []
    for c, v in ident.items():
        for i, s in enumerate(v["ids"]):
            for j, m in enumerate(L.WINDOW_MONTHS):
                sd, ens, dr = float(v["sd_fit_rel"][i]), 0.05 + 0.01 * j, 0.002 * j
                rows.append({"station_id": s, "county": c, "group": v["groups"][i],
                             "date": f"{m}-01", "lead_months": 48 + j, "sd_fit_rel_m": sd,
                             "sigma_ens_rel_m": ens, "drift_m": dr,
                             "sigma_idio_m": math.sqrt(sd ** 2 + ens ** 2 + dr ** 2)})
    return pd.DataFrame(rows)


def test_sigma_idio_units_and_lead_gate():
    d = _design(reduced=False)
    ident = FI.u1_identity(d, _ident_and_wells(d))
    art = _artefact(ident)
    S = FI.sigma_scale(art, ident)
    assert S["Changhua"].shape == (74, 11) and (S["Yunlin"] > 0).all()
    for mutate in (lambda a: a.assign(lead_months=a["lead_months"] + 1),
                   lambda a: a.assign(sigma_idio_m=a["sigma_idio_m"] * 1.001),
                   lambda a: a.assign(sd_fit_rel_m=a["sd_fit_rel_m"] + 1e-6,
                                      sigma_idio_m=np.sqrt((a["sd_fit_rel_m"] + 1e-6) ** 2
                                                           + a["sigma_ens_rel_m"] ** 2
                                                           + a["drift_m"] ** 2)),
                   lambda a: a.iloc[1:],
                   lambda a: a.assign(group="other")):
        with pytest.raises(io.EngineeringFailure):
            FI.sigma_scale(mutate(art.copy()), ident)


def test_upstream_pins_include_zr0_and_sigma():
    assert FI.UPSTREAM_SHA[FI.ZR0_FORWARD].startswith("a1833d5f")
    assert FI.UPSTREAM_SHA[FI.SIGMA_IDIO_A].startswith("b70da8b0")
    assert set(FI.UPSTREAM_SHA) == set(FI.INPUT_FILES.values())


# ----------------------------------------------------------------------------- drafts
def test_draft_power_validates_and_key_fields(tmp_path):
    u1 = L.synthetic_u1(SEED)
    doc = PR.draft_prereg_power(20261199, u1=(u1, "0" * 64), input_sha={"a": "b"})
    assert doc["id"] == "TEST." + L.POWER_ID and doc["kind"] == "declaration"
    assert _validate(doc, tmp_path) == []
    assert doc["design"]["seed"] == 20261199
    th = doc["thresholds"]
    assert (th["p_correct_min"], th["p_wrong_max"], th["b_strong"]) == (0.8, 0.05, 20.0)
    assert doc["design"]["window"] == {"first": "2026-12", "last": "2027-10", "months": 11,
                                       "fv_window_slice": [1, 12]}
    assert doc["hypotheses"]["gating_pair"] == FAM
    assert "stacking" in doc["not_the_killed_stacking_route"]
    bc = doc["bf_commitments"]
    assert "never moves C20/C21" in bc["never"] and "after N2C" in bc["scored"]
    assert L.NO_LATER_THAN in bc["scored"] and bc["precedence"] == L.PRECEDENCE
    text = json.dumps(doc)
    assert "Changhua pumping in the recipe" not in text
    assert "README" not in text and "separate ruling" not in text
    assert "E29 (FB family BF, ledger only)" in text
    assert "never registered" in doc["disclosures"]["multiplicity"]
    probs = PR.power_problems(doc, prereg_dir=tmp_path)
    assert any("TEST draft" in p for p in probs)
    doc2 = PR.draft_prereg_power(SEED, u1=(u1, "0" * 64), input_sha={"a": "b"})
    assert any("refused" in p for p in PR.power_problems(doc2, prereg_dir=tmp_path))


def test_draft_bf_validates_and_key_fields(tmp_path):
    pairs = [f"{FAM}@Changhua", f"{FAM}@Yunlin"]
    doc = PR.draft_prereg_bf(pairs, power=_power(), wells=_wells())
    assert doc["id"] == "TEST." + L.BF_ID and doc["kind"] == "test"
    assert _validate(doc, tmp_path) == []
    assert [t["candidate"] for t in doc["tests"]] == pairs
    t0 = doc["tests"][0]
    assert t0["data_scope"] == "heads:2026-12..2027-10"
    unit = t0["fresh_units"][0]
    assert len(t0["external_units"][unit]["ids"]) == 74
    assert t0["external_units"][unit]["window"] == "2026-12..2027-10"
    assert doc["scoring"]["not_before"] == L.NOT_BEFORE
    assert doc["scoring"]["no_later_than"] == "2028-01-31T23:59:59Z" == L.NO_LATER_THAN
    assert L.N2C_REPORT in doc["scoring"]["after"]
    pm = doc["page_mapping"]
    assert pm["ledger_row"] == "E29 (FB family BF, ledger only)"
    assert "no page line in any branch" in pm["after_scoring"]
    assert "C20/C21" in pm["never"] and "response-class" in pm["never"]
    assert doc["precedence"] == L.PRECEDENCE
    text = json.dumps(doc)
    assert "README" not in text and "separate ruling" not in text
    assert doc["frozen_code"][SC.GOV] in (SC.GOV_SHA, "MISSING")
    assert SC.SELF in doc["frozen_code"]
    probs = PR.bf_problems(doc, pairs, prereg_dir=tmp_path)
    assert any("TEST draft" in p for p in probs)
    assert any("not registered" in p for p in probs)


def test_draft_bf_rejects_bad_pairs():
    with pytest.raises(ValueError):
        PR.draft_prereg_bf([], wells=_wells(), power=_power())
    with pytest.raises(ValueError):
        PR.draft_prereg_bf([f"{FAM}@Yunlin"] * 2, wells=_wells(), power=_power())
    with pytest.raises(KeyError):
        PR.draft_prereg_bf(["stack@Changhua"], wells=_wells(), power=_power())
    with pytest.raises(ValueError):
        PR.draft_prereg_bf(["R0_vs_Q0@Yunlin"], wells=_wells(), power=_power())


# ----------------------------------------------------------------------------- scorer
def _fake_registered(monkeypatch, frozen):
    doc = {"id": L.BF_ID, "frozen_code": frozen, "tests": [], "power_study": {}}

    def reg(pid, prereg_dir=None):
        if pid == L.BF_ID:
            return doc, "f" * 64
        raise io.Refused(f"{pid} is not registered")

    monkeypatch.setattr(io, "registered", reg)


def test_scorer_refusals(tmp_path, monkeypatch):
    kw = {"out_dir": tmp_path / "o", "attempts": tmp_path / "a.jsonl", "prereg_dir": tmp_path}
    with pytest.raises(SC.Refused, match="not registered"):
        SC.score(tmp_path, "t", tmp_path / "n2c.json", now=LATE, **kw)
    _fake_registered(monkeypatch, {})
    with pytest.raises(SC.Refused, match="before 2028-01-15"):
        SC.score(tmp_path, "t", tmp_path / "n2c.json", now=EARLY, **kw)
    with pytest.raises(SC.Refused, match="missing"):
        SC.score(tmp_path, "t", tmp_path / "n2c.json", now=LATE, **kw)
    rep = tmp_path / "n2c.json"
    rep.write_text(json.dumps({"id": "other", "verdict": "not_supported"}))
    with pytest.raises(SC.Refused, match="does not name"):
        SC.score(tmp_path, "t", rep, now=LATE, **kw)
    rep.write_text(json.dumps({"id": L.N2C_ID, "verdict": "maybe"}))
    with pytest.raises(SC.Refused, match="no verdict"):
        SC.score(tmp_path, "t", rep, now=LATE, **kw)
    rep.write_text(json.dumps({"id": L.N2C_ID, "verdict": "not_supported"}))
    with pytest.raises(SC.Refused, match="heads_window.py is not bound"):
        SC.score(tmp_path, "t", rep, now=LATE, **kw)
    _fake_registered(monkeypatch, {SC.GOV: SC.GOV_SHA, SC.E02C_SCORER: SC.E02C_SCORER_SHA})
    with pytest.raises(SC.Refused, match="score_fb.py is not bound"):
        SC.score(tmp_path, "t", rep, now=LATE, **kw)
    _fake_registered(monkeypatch, {SC.GOV: SC.GOV_SHA, SC.E02C_SCORER: SC.E02C_SCORER_SHA,
                                   SC.SELF: "0" * 64})
    with pytest.raises(SC.Refused, match="frozen code changed"):
        SC.score(tmp_path, "t", rep, now=LATE, **kw)
    rows = [json.loads(x) for x in (tmp_path / "a.jsonl").read_text().splitlines()]
    assert sum(r["event"] == "score_refused" for r in rows) == 8
    assert not (tmp_path / "o" / "report.json").exists()


def test_scorer_datum_check_counts_only():
    rng = np.random.default_rng(0)
    h = rng.normal(size=(5, 11))
    zr = h + rng.normal(size=(5, 1))
    assert SC.datum_offsets_constant(zr, h) == {"ok": True, "wells_failing": 0,
                                                "tolerance_m": 1e-6}
    zr[2, 3] += 1e-5
    r = SC.datum_offsets_constant(zr, h)
    assert not r["ok"] and r["wells_failing"] == 1 and set(r) == {"ok", "wells_failing",
                                                                  "tolerance_m"}


def _window_product(root: Path, window) -> Path:
    spec = importlib.util.spec_from_file_location("gov_hw_test_fb", REPO / SC.GOV)
    HW = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(HW)
    d = root / "heads_window" / "v1"
    d.mkdir(parents=True)
    (d / "validation.json").write_text("{}\n")
    rec = {"path": "heads.parquet", "sha256": "0" * 64}
    m = HW.instantiate(HW.template(), version="v1", created="2028-01-10", inputs=[],
                       code=[], git_commit="0" * 40, fit_window_end="2029-10", outputs=[rec],
                       validation_report="validation.json",
                       gate={"passed": True, "criteria": "synthetic"}, window=window)
    (d / "MANIFEST.json").write_text(json.dumps(m, indent=1))
    return d


def test_scorer_precondition_is_gov_covering_rule(tmp_path):
    ok, why = SC.precondition(_window_product(tmp_path / "a", ("2026-11", "2029-10")))
    assert ok and "covering_rule" in why
    ok, why = SC.precondition(_window_product(tmp_path / "b", ("2027-11", "2029-10")))
    assert not ok and "does not cover" in why


class _FakeProduct:
    def __init__(self, path: Path):
        self.path = path
        self.outputs = {path.name: {"sha256": "0" * 64}}

    def _resolve(self, name):
        return self.path

    def read_table(self, name, columns=None):
        df = pd.read_parquet(self.path)
        return df[[c for c in (columns or df.columns) if c in df.columns]]


def _heads(tmp_path, **cols):
    pytest.importorskip("pyarrow")
    p = tmp_path / "heads.parquet"
    pd.DataFrame(cols).to_parquet(p, index=False)
    return _FakeProduct(p)


def test_scorer_heads_contract_before_the_look(tmp_path):
    ids = [f"07{i:06d}" for i in range(74)]
    good = _heads(tmp_path, station_id=["07000001", "07000002"], date=["2026-12", "2027-01"],
                  observed_m=[1.0, 2.0], flag=[0, 1])
    assert SC.contract_problems(good, "heads.parquet", ids) == []
    assert "not a parquet table listed" in SC.contract_problems(good, "x.csv", ids)[0]
    (tmp_path / "l").mkdir()
    lost = _heads(tmp_path / "l", station_id=["7000001"], date=["2026-12"], observed_m=[1.0],
                  flag=[0])
    assert any("leading zeros" in x for x in SC.contract_problems(lost, "heads.parquet", ids))
    (tmp_path / "m").mkdir()
    nov = _heads(tmp_path / "m", station_id=["07000001"], date=["2026-11"], observed_m=[1.0],
                 flag=[0])
    assert "no scorable row" in SC.contract_problems(nov, "heads.parquet", ids)[0]
    (tmp_path / "f").mkdir()
    flags = _heads(tmp_path / "f", station_id=["07000001"], date=["2026-12"],
                   observed_m=[1.0], flag=[3])
    assert "no scorable row" in SC.contract_problems(flags, "heads.parquet", ids)[0]


# ----------------------------------------------------------------------------- real metadata
@NEED_RESULTS
def test_real_u1_grid_is_copied():
    u1, sha = io.registered(L.U1_ID)
    d = L.design_spec(u1, sha, {"x": "y"}, 20261199)
    for c in L.COUNTIES:
        uc = u1["design"]["counties"][c]
        dc = d["counties"][c]
        assert dc["wells"] == L.EXPECTED_WELLS[c] == uc["wells"]
        assert dc["groups"] == uc["groups"] and dc["subgroups"] == uc["subgroups"]
        n = len(uc["a_s"]) * len(uc["a_w"]) * len(uc["phi"])
        assert dc["n_gating_world_cells"] == n * 3 + n * 2 + n * 2


@NEED_RESULTS
def test_upstream_pins_match_registrations():
    u1, _ = io.registered(L.U1_ID)
    fv2, _ = io.registered(L.FV2_ID)
    e02d, _ = io.registered(L.E02D_ID)
    fv1, _ = io.registered(L.FV1_ID)
    assert FI.UPSTREAM_SHA[FI.WELLS_E02D] == u1["design"]["wells_e02d_sha256"]
    for t in fv2["truths"]:
        assert FI.UPSTREAM_SHA[FI.truth_forward(t["tag"])] == t["v1_forward"]["sha256"]
    assert FI.UPSTREAM_SHA[FI.OBS_MASK] == fv1["inputs_frozen"]["obs_mask.csv"]
    assert FI.UPSTREAM_SHA[FI.ANOMALY_POOL] == fv1["inputs_frozen"]["anomaly_pool.csv"]
    assert FI.UPSTREAM_SHA[FI.SIGMA_IDIO_A] == FI._registered_sha(e02d, FI.SIGMA_IDIO_A)
    assert u1["design"]["seed"] in PW.REFUSED_SEEDS
    assert u1["design"]["self_check_seed"] in PW.REFUSED_SEEDS
    assert fv2["design"]["SEED"] in PW.REFUSED_SEEDS
