"""COLF.column_family_2028.v1 (round-8 rank 7): the column-family decidability OSSE and freeze.

On fabricated members and fabricated tables only, except the needs_results tests, which read
model outputs and registration metadata (never an observation):

- the copied constants equal their hash-bound sources (N3 G0, n3lib, N3S-L's scorer);
- seeds are required (no default) and refused seeds are refused;
- the ranking metrics (pile cumulative RMSE over verified months, township bound distance),
  N3's sensor rule and the anchor-change cell;
- the OSSE engine on fabricated members: per-arm self-check (swap, pile midpoint, the
  label-randomised township null, aliases exactly 0.5), no P after a failed check, nominal
  vs report-only cells, per-pile agreement, multiplicity, labels, determinism;
- the engineering record and the one-run rule at fixed paths;
- the loader's dedup (byte identity per arm) and the freeze file's refusals;
- the scorer refuses before registration, before 2030-01-15, out of session order, and
  writes once;
- the draft declaration validates and carries its key fields (results/ tree only).
"""
from __future__ import annotations

import datetime as dt
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

REPO = Path(__file__).resolve().parents[1]
D = REPO / "dataprep_staging" / "colfreeze"
for _p in (str(D), str(REPO / "dataprep_staging")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import colf_config as C  # noqa: E402
import members as M  # noqa: E402
import osse as O  # noqa: E402
import score_colf as S  # noqa: E402

RESULTS = REPO / "results" / "dataprep"
HAVE_RESULTS = (RESULTS / "preregistration" / "INDEX.jsonl").exists() and all(
    (REPO / p).exists() for p, _ in C.FORWARDS.values()) and (REPO / C.SD_V2["report"]).exists()
needs_results = pytest.mark.skipif(not HAVE_RESULTS, reason="gitignored results/ tree absent")
SEED = C.TEST_SEED


# ----------------------------------------------------------------------------- static
def test_package_conventions():
    pat = re.compile(r"(/home/|-home-)[A-Za-z0-9_]")
    for p in sorted(D.glob("*.py")):
        txt = p.read_text(encoding="utf-8")
        assert not pat.search(txt), p
        assert "\nfrom __future__ import annotations\n" in txt, p
        assert not re.search(r"write_prereg\s*\(", txt), f"{p.name} calls write_prereg"
        assert max(len(line) for line in txt.splitlines()) <= 100, p


def test_copied_constants_match_their_bound_sources():
    g0 = REPO / C.N3["g0_code"]
    lib = REPO / C.N3["n3lib"]
    sl = REPO / C.N3SL["scorer"]
    if not (g0.exists() and lib.exists() and sl.exists()):
        pytest.skip("bound sources absent")
    assert M.sha256(g0) == C.N3["g0_code_sha256"]
    assert M.sha256(lib) == C.N3["n3lib_sha256"]
    assert M.sha256(sl) == C.N3SL["scorer_sha256"]
    g = _literals(g0)
    assert (g["SD_GRID"], g["PHI_GRID"]) == (C.G0_SD_GRID_M, C.G0_PHI_GRID)
    n = _literals(lib)
    assert n["PRINT_RES_CM"] == C.PRINT_RES_CM and n["MIN_MONTHS"] == 8
    assert n["MIN_MONTHS24"] == C.ARMS["r_piles"]["min_months"]
    s = _literals(sl)
    assert (s["TOL"], s["LISTED_FLOOR"], s["L1_MIN_LISTED"]) == (C.TOL, C.LISTED_FLOOR,
                                                                  C.L1_MIN_LISTED)
    assert s["COAST"] == C.COAST and s["NOMINAL"] == C.NOMINAL and s["SURVEYS"] == C.SURVEYS


def _literals(path: Path) -> dict:
    """Module-level literal assignments of a source file (never imported)."""
    import ast
    import contextlib

    out = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(
                node.targets[0], ast.Name):
            with contextlib.suppress(ValueError):
                out[node.targets[0].id] = ast.literal_eval(node.value)
    return out


def test_config_shape():
    assert len(C.TIER_AB) == 11 and set(C.ALIASES) <= set(C.TIER_AB)
    assert set(C.ALIASES.values()) <= set(C.TIER_AB)
    assert [C.ARMS[a]["role"] for a in C.ARM_ORDER] == ["report_only", "report_only", "gating"]
    g = C.ARMS["g_piles"]
    assert g["window"] == ("2028-12", "2029-11") and g["min_months"] == 8
    assert C.ARMS["r_piles"]["window"] == ("2026-12", "2028-11")
    assert (C.P_D, C.P_NARROW) == (0.8, 0.5)
    assert C.NOT_BEFORE.date() == dt.date(2030, 1, 15)
    assert C.D_PENDING_AFTER == "2030-01-15"
    assert min(C.ARMS[a]["min_months"] for a in ("r_piles", "g_piles")) > C.AC_400_MONTHS
    assert C.MEMBER_SPEC["q0_nh"]["label"] == C.UNVERIFIED_PAIRING
    assert C.XIUTAN["pile"] == "秀潭國小" and C.XIUTAN["min_version"] == 3
    assert C.FREEZE_TOWN_MONTHS == ("2025-01", "2029-06")
    assert C.AVAIL_CELLS == ("full", "minimum") and C.RHO_SENS == (0.4, 2.0)
    assert C.ENGINEERING_RECORD == "results/dataprep/reports/COLF/engineering_record.json"
    assert C.LOOK_SCOPE_CORRECTION["chain_hash"].startswith("bef64e1a")


def test_seed_required_and_refused():
    with pytest.raises(TypeError):
        O.draft_prereg()  # the seed has no default
    with pytest.raises(ValueError):
        O.check_seed(None)
    with pytest.raises(ValueError):
        O.check_seed(True)
    with pytest.raises(ValueError):
        O.check_seed(1.5)
    for s in C.REFUSED_SEEDS:
        with pytest.raises(ValueError):
            O.check_seed(s)
    assert O.check_seed(SEED) == SEED
    with pytest.raises(ValueError):
        O.check_seed(SEED, real=True)
    with pytest.raises(TypeError):
        O.run_osse({})  # no default seed
    with pytest.raises(ValueError):
        O.draft_prereg(SEED)


# ----------------------------------------------------------------------------- metrics
def test_choose_sensor_is_n3s_rule():
    n = 12
    full = np.ones(n)
    part = np.full(n, np.nan)
    part[:5] = 1.0
    assert O.choose_sensor({400: full, 300: full}, 400, 300, 8)[0] == 400
    assert O.choose_sensor({400: part, 300: full}, 400, 300, 8)[0] == 300
    short = np.full(n, np.nan)
    short[:7] = 1.0
    assert O.choose_sensor({400: part, 300: short}, 400, 300, 8)[0] == 400
    assert O.choose_sensor({300: part}, 300, None, 8)[0] == 300


def test_rmse_piles_is_per_pile_and_skips_unverified_months():
    m = np.array([[1.0, 1.0, 1.0, 1.0], [0.5, 0.5, 0.5, 0.5]])
    assert O.rmse_piles(m, m.copy()).tolist() == [0.0, 0.0]
    obs = m.copy()
    obs[0, 1] = np.nan                       # a gap: the member skips that month too
    assert O.rmse_piles(m, obs).tolist() == [0.0, 0.0]
    obs2 = m.copy()
    obs2[1] += 0.1
    r = O.rmse_piles(m, obs2)
    want = np.sqrt(np.mean(np.cumsum(np.full(4, 0.1)) ** 2))
    assert r[0] == 0.0 and abs(r[1] - want) < 1e-12
    assert np.isnan(O.rmse_piles(m, np.full((2, 4), np.nan))).all()


def test_outcome_piles_needs_both_piles():
    ra = np.array([[1.0, 1.0], [1.0, 3.0], [2.0, 2.0], [np.nan, 1.0]])
    rb = np.array([[2.0, 2.0], [2.0, 2.0], [2.0, 2.0], [2.0, 2.0]])
    o = O.outcome_piles(ra, rb)
    assert o == {"a": 0.25, "b": 0.0, "tie": 0.75, "split": 0.25}


def test_bound_distance_semantics():
    peaks = np.array([2.0, 4.0, 1.0, 9.0])
    listed = np.array([True, True, False, False])
    upper = np.array([5.0, 5.0, 3.0, 3.0])
    use = np.array([True, True, True, False])
    d = O.bound_distance(peaks, listed, upper, use)
    assert d[0] == pytest.approx(0.5) and d[1] == 0.0 and d[2] == 0.0 and np.isnan(d[3])
    obs = {116: {"listed": listed, "upper": upper, "use": use, "informative": np.asarray(True)},
           117: {"listed": listed, "upper": upper, "use": use, "informative": np.asarray(False)}}
    r = O.rmse_towns({116: peaks, 117: peaks + 100}, obs)
    assert float(r) == pytest.approx(np.sqrt(0.25 / 3))
    obs[116]["informative"] = np.asarray(False)
    assert np.isnan(O.rmse_towns({116: peaks, 117: peaks}, obs))


def test_synth_pile_obs_cells():
    truth = np.array([[1.0] * 12, [1.0] * 12])
    rng = np.random.default_rng(0)
    nom = O.synth_pile_obs(truth, 1e-12, 0.5, 8, 3, rng)
    ac = O.synth_pile_obs(truth, 1e-12, 0.5, 8, 3, rng, anchor="ac300")
    assert np.allclose(nom, 1.0) and np.allclose(ac[:, 0], C.F300) and np.allclose(ac[:, 1], 1.0)
    rho = O.synth_pile_obs(truth, 1e-12, 0.5, 8, 3, rng, rho=2.0)
    assert np.allclose(rho, 2.0)
    mn = O.synth_pile_obs(truth, 1e-12, 0.5, 8, 5, rng, avail="minimum")
    assert (np.isfinite(mn).sum(axis=-1) == 8).all()
    for bad in ({"anchor": "bogus"}, {"avail": "bogus"}):
        with pytest.raises(ValueError):
            O.synth_pile_obs(truth, 0.1, 0.5, 8, 3, rng, **bad)


def test_outcome_ties_and_nan():
    o = O.outcome([1.0, 2.0, 3.0, np.nan], [2.0, 1.0, 3.0, 1.0])
    assert o == {"a": 0.25, "b": 0.25, "tie": 0.5}


# ----------------------------------------------------------------------------- OSSE engine
def _frozen(rates: dict, towns_offset: dict | None = None, town_scale: float = 0.02) -> dict:
    pm = M.months(*C.FREEZE_PILE_MONTHS)
    tm = M.months(*C.FREEZE_TOWN_MONTHS)
    tc = {t: [2 * i, 2 * i + 1] for i, t in enumerate(C.COAST)}
    mem = {}
    for c, r in rates.items():
        t_p = np.arange(len(pm))[None, :] * np.array([[r], [0.6 * r]]) / 100.0
        grad = np.repeat(np.linspace(1.0, 0.2, 11), 2)[:, None]
        t_t = np.arange(len(tm))[None, :] * grad * town_scale * r / 100.0
        if towns_offset and c in towns_offset:
            t_t = t_t.copy()
            t_t[-2:] += np.arange(len(tm))[None, :] * towns_offset[c] / 100.0
        mem[c] = {"piles": t_p.astype(np.float32), "towns": t_t.astype(np.float32),
                  "spec": C.MEMBER_SPEC[c]}
    return {"members": mem, "piles_months": pm, "towns_months": tm, "town_cells": tc,
            "pile_cells": {k: C.PILES[k]["cell"] for k in C.PILE_KEYS}, "full_identity": {}}


def _small_frozen(**kw) -> dict:
    # q0_s29 == q0 everywhere; q0_csk1 == q0_cs on the piles but not at 大城鄉
    return _frozen({"q0": 0.10, "q0_s29": 0.10, "q0_cs": 1.50, "q0_csk1": 1.50},
                   towns_offset={"q0_csk1": 0.5}, town_scale=0.002, **kw)


def test_dedup_is_by_bytes_per_arm():
    dd = M.dedup(_small_frozen())
    assert dd["r_piles"]["members"] == ["q0", "q0_cs"] == dd["g_piles"]["members"]
    assert dd["r_towns"]["members"] == ["q0", "q0_cs", "q0_csk1"]
    assert dd["r_towns"]["evidence"]["q0_csk1"]["townships_differing"] == ["大城鄉"]
    assert dd["r_towns"]["evidence"]["q0_s29"]["bytes_equal_on_arm"]


def test_osse_on_fabricated_members():
    fz = _small_frozen()
    res = O.run_osse(fz, SEED, reps=800, real=False)
    assert res["status"] == "ok" and res["self_check"]["pass"], res["self_check"]["fails"]
    for arm in ("r_piles", "g_piles"):
        sc = res["arms"][arm]["self_check"]
        assert set(sc["aliases"]) == {"q0_s29", "q0_csk1"}
        assert all(v["p_sym"] == 0.5 for v in sc["aliases"].values())
        assert all(r["swap_ok"] for r in sc["null_cells"])
        pr = res["arms"][arm]["pairs"]["q0|q0_cs"]
        assert len(pr["cells"]) == 2 * 4 * 2 and pr["p_min"] >= 0.8
        assert {r["avail"] for r in pr["cells"]} == {"full", "minimum"}
        assert len(pr["report_only_cells"]) == 2 * 4 * 4
        assert set(pr["report_only_p_min"]) == {"ac300", "rho_0.4", "rho_2", "g_mod_cell"}
        assert all("anchor" not in r for r in pr["cells"])
    assert set(res["arms"]["r_towns"]["self_check"]["aliases"]) == {"q0_s29"}
    dec = res["decidability"]
    assert dec["arms"]["g_piles"]["q0|q0_cs"]["label"] == "D"
    assert dec["arms"]["r_piles"]["q0|q0_cs"]["label"] == "R"
    m = dec["multiplicity"]["g_piles"]
    assert m["n"] == 1 and m["expected_wrong_labels"] == pytest.approx(
        1 - dec["arms"]["g_piles"]["q0|q0_cs"]["p"])
    assert dec["expected_wrong_D_labels"] == pytest.approx(m["sum_1_minus_p"])
    assert all(v["label"] == "narrowing" for v in dec["arms"]["r_towns"].values())
    again = O.run_osse(fz, SEED, reps=800, real=False)
    assert again["decidability"] == res["decidability"]
    lines = O.page_lines(res)
    assert lines["status"] == "ok"
    d_lines = [x for x in lines["lines"]["C09"] if x["label"] == "D"]
    assert d_lines and "at pile/column scale 1" in d_lines[0]["line"]
    assert all(x["arm"] != "r_towns" for x in lines["lines"]["C09"])
    assert "no cut number" in lines["lines"]["E24"][0]["line"]
    town = [x["line"] for x in lines["lines"]["C11"] if x["arm"] == "r_towns"]
    assert any("separable only at 大城鄉" in x for x in town)


def test_town_null_passes_where_a_midpoint_null_fails():
    """Review B1: with listed coastal townships the one-sided bound metric makes the
    midpoint non-equidistant; the label-randomised null stays at 0.5."""
    fz = _frozen({"q0": 0.10, "q0_s29": 0.10, "q0_cs": 0.40, "q0_csk1": 0.40},
                 town_scale=8.0)
    x = O._arm_inputs(fz, "r_towns", ["q0", "q0_cs"])
    assert (x["q0_cs"][116] >= C.LISTED_FLOOR).sum() >= C.L1_MIN_LISTED
    mid = O._mid(x["q0"], x["q0_cs"])
    cell = O.nominal_cells("r_towns")[0]
    ss = np.random.SeedSequence(5)
    naive = O._p_sym(O.simulate("r_towns", x["q0"], x["q0_cs"], mid, cell, 2000, ss))
    fair = O._p_sym(O.simulate("r_towns", x["q0"], x["q0_cs"], mid, cell, 2000, ss,
                               swap_labels=True))
    assert abs(naive - 0.5) > C.SELF_CHECK_TOL
    assert abs(fair - 0.5) <= C.SELF_CHECK_TOL
    res = O.run_osse(fz, SEED, reps=800, arms=("r_towns",), real=False)
    assert res["self_check"]["by_arm"]["r_towns"]["pass"], res["self_check"]["fails"]


def test_osse_close_pair_is_not_decidable():
    fz = _frozen({"q0": 0.10, "q0_s29": 0.10, "q0_cs": 0.1001, "q0_csk1": 0.1001})
    res = O.run_osse(fz, SEED, reps=800, arms=("g_piles",), real=False)
    assert res["self_check"]["pass"]
    assert res["decidability"]["arms"]["g_piles"]["q0|q0_cs"]["label"] == "narrowing"


def test_failed_self_check_computes_and_writes_no_p(monkeypatch):
    def failing(arm, *a, **k):
        return {"pass": arm != "r_towns", "fails": ["x"] if arm == "r_towns" else [],
                "null_cells": [], "aliases": {}, "null": "n"}
    monkeypatch.setattr(O, "self_check_arm", failing)
    res = O.run_osse(_small_frozen(), SEED, reps=50, real=False)
    assert res["status"] == "self_check_failed" and "arms" not in res
    assert res["decidability"] == {"status": "self_check_failed", "arms": {}}
    assert "p_min" not in json.dumps(res) and "p_correct" not in json.dumps(res)
    assert O.page_lines(res)["status"].startswith("no lines")


def test_pair_words_carry_labels():
    w = O.pair_words("g_piles", "page|page_aquitard")
    assert C.UNVERIFIED_TRAJECTORY in w
    assert C.UNVERIFIED_PAIRING in O.pair_words("r_piles", "q0|q0_nh")
    assert O.pair_words("r_piles", "q0|q0_cs") == "q0 vs q0_cs"
    assert "大城鄉" in O.pair_words("r_towns", "q0_cs|q0_csk1")
    assert "大城鄉" not in O.pair_words("r_piles", "q0_cs|q0_csk1")


# ----------------------------------------------------------------------------- record / run
def _freeze_file(tmp_path, fz: dict) -> Path:
    p = tmp_path / "predictions.npz"
    arr = {f"{c}__piles": fz["members"][c]["piles"] for c in fz["members"]}
    arr.update({f"{c}__towns": fz["members"][c]["towns"] for c in fz["members"]})
    np.savez(p, **arr, piles_months=np.asarray(fz["piles_months"]),
             towns_months=np.asarray(fz["towns_months"]),
             town_cells_json=np.asarray(json.dumps(fz["town_cells"])))
    return p


def test_engineering_record_and_run_refusals(tmp_path, monkeypatch):
    base = tmp_path / "COLF"
    npz = _freeze_file(tmp_path, _small_frozen())
    with pytest.raises(O.RunRefused, match="engineering record first"):
        O.run_registered(SEED, npz, base=base, real=False)
    rec = O.write_engineering_record(SEED, npz, base=base, real=False)
    assert rec["seed"] == SEED
    with pytest.raises(FileExistsError):
        O.write_engineering_record(SEED + 1, npz, base=base, real=False)
    with pytest.raises(O.RunRefused, match="seed"):
        O.run_registered(SEED + 1, npz, base=base, real=False)
    other = tmp_path / "other.npz"
    other.write_bytes(npz.read_bytes() + b"x")
    with pytest.raises(O.RunRefused, match="frozen predictions"):
        O.run_registered(SEED, other, base=base, real=False)
    monkeypatch.setattr(M, "load_freeze", lambda p, sha: _small_frozen())
    monkeypatch.setattr(O, "run_osse", lambda fz, seed, **k: {
        "status": "self_check_failed", "self_check": {"pass": False, "fails": ["x"]},
        "decidability": {"status": "self_check_failed", "arms": {}}})
    out = O.run_registered(SEED, npz, base=base, real=False)
    assert out["status"] == "self_check_failed" and not (base / "osse_result.json").exists()
    assert (base / "osse_selfcheck_failure_1.json").exists()
    monkeypatch.setattr(O, "run_osse", lambda fz, seed, **k: {"status": "ok", "arms": {}})
    assert O.run_registered(SEED, npz, base=base, real=False)["status"] == "ok"
    with pytest.raises(O.RunRefused, match="runs once"):
        O.run_registered(SEED, npz, base=base, real=False)
    rows = [json.loads(x) for x in (base / "attempts.jsonl").read_text().splitlines()]
    assert [r.get("rc") for r in rows] == [None, 2, 0]


def test_xiutan_reads_any_version_from_v3(tmp_path):
    root = tmp_path / C.XIUTAN["product_root"]
    (root / "v2").mkdir(parents=True)
    (root / "v2" / "MANIFEST.json").write_text(json.dumps({"allowed_uses": ["prospective_scorer"]}))
    assert O._xiutan(tmp_path)["status"] == "excluded"
    (root / "v4").mkdir()
    (root / "v4" / "MANIFEST.json").write_text(json.dumps({"allowed_uses": ["validator"]}))
    assert O._xiutan(tmp_path)["status"] == "excluded"
    (root / "v12").mkdir()
    (root / "v12" / "MANIFEST.json").write_text(
        json.dumps({"allowed_uses": ["prospective_scorer"]}))
    x = O._xiutan(tmp_path)
    assert x["status"] == "lead decision needed" and {v["version"] for v in x["versions"]} == \
        {"v4", "v12"}
    assert "cannot add an arm" in x["after_registration"]


# ----------------------------------------------------------------------------- freeze file
def test_load_freeze_refuses_wrong_hash_and_member_set(tmp_path):
    p = tmp_path / "predictions.npz"
    np.savez(p, **{"q0__piles": np.zeros((2, 3), np.float32),
                   "q0__towns": np.zeros((2, 3), np.float32),
                   "piles_months": np.asarray(["2026-11"]),
                   "towns_months": np.asarray(["2026-01"]),
                   "town_cells_json": np.asarray("{}")})
    with pytest.raises(M.MemberError):
        M.load_freeze(p, "0" * 64)
    with pytest.raises(M.MemberError, match="never add members"):
        M.load_freeze(p, M.sha256(p))


def test_array_sha_is_canonical():
    a = np.arange(6, dtype=np.float64).reshape(2, 3)
    assert M.array_sha(a) == M.array_sha(a.astype(np.float32))
    assert M.array_sha(a) != M.array_sha(a.reshape(3, 2))


# ----------------------------------------------------------------------------- scorer
def _args(tmp_path, **kw):
    base = {"out": tmp_path / "out", "piles_product": tmp_path / "p",
            "towns_product": tmp_path / "t", "freeze_npz": tmp_path / "f.npz",
            "n3_record": None, "n3sl_record": None, "n3_record_sha": None,
            "n3sl_record_sha": None, "final": False}
    base.update(kw)
    return SimpleNamespace(**base)


def test_scorer_refuses_before_registration(tmp_path):
    pytest.importorskip("harness.prereg3")
    pdir = tmp_path / "prereg"
    pdir.mkdir()
    with pytest.raises(S.ScoringRefused, match="not registered"):
        S.run_score(_args(tmp_path), prereg_dir=pdir, rows_dir=tmp_path / "rows")
    assert not (tmp_path / "out" / "report.json").exists()


def test_scorer_writes_once(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "report.json").write_text("{}")
    with pytest.raises(FileExistsError):
        S.run_score(_args(tmp_path))


def test_scorer_refuses_before_not_before_and_out_of_order(tmp_path, monkeypatch):
    pytest.importorskip("harness.prereg3")
    monkeypatch.setattr(S, "load_registered", lambda *a, **k: {"frozen_code": {}})
    monkeypatch.setattr(S, "check_code", lambda doc: None)
    with pytest.raises(S.ScoringRefused, match="not before"):
        S.run_score(_args(tmp_path), now=dt.datetime(2029, 12, 31, tzinfo=C.UTC))
    pdir = tmp_path / "prereg"
    pdir.mkdir()
    (pdir / "looks.jsonl").write_text("")
    with pytest.raises(S.ScoringRefused, match="session order"):
        S.run_score(_args(tmp_path), prereg_dir=pdir,
                    now=dt.datetime(2030, 2, 1, tzinfo=C.UTC))
    rec = tmp_path / "n3.json"
    rec.write_text(json.dumps({"prereg_id": C.N3["id"], "verdict": "not_scorable"}))
    assert not S.order_status(pdir, rec, None)["n3_record"]["ok"]          # no sha
    assert not S.order_status(pdir, rec, None, "0" * 64)["n3_record"]["ok"]
    st = S.order_status(pdir, rec, None, M.sha256(rec))
    assert st["n3_record"]["ok"] and st["n3_record"]["sha256"] == M.sha256(rec)
    assert not st["ok"]
    rec2 = tmp_path / "n3sl.json"
    rec2.write_text(json.dumps({"prereg_id": C.N3SL["id"], "survey_roc": 117}))
    assert S.order_status(pdir, rec, rec2, M.sha256(rec), M.sha256(rec2))["ok"]


def test_product_covers_reads_the_manifest(tmp_path, monkeypatch):
    pytest.importorskip("common.manifest")
    import common.manifest as CM

    monkeypatch.setattr(CM, "load_manifest", lambda d: ({"data_end": "2029-10"}, d))
    assert not S.product_covers(tmp_path)[0]
    monkeypatch.setattr(CM, "load_manifest", lambda d: ({"parameters": {"data_end":
                                                                        "2029-11"}}, d))
    assert S.product_covers(tmp_path)[0]
    monkeypatch.setattr(CM, "load_manifest", lambda d: ({}, d))
    assert "no data_end" in S.product_covers(tmp_path)[1]


def test_observed_piles_gate_and_ranking():
    fz = _small_frozen()
    ms = M.months(*C.ARMS["g_piles"]["window"])
    rows = []
    for key in C.PILE_KEYS:
        spec = C.PILES[key]
        inc = O.pile_increments(fz["members"]["q0"]["piles"], fz["piles_months"], "g_piles")
        for i, m in enumerate(ms):
            rows.append({"pile": spec["pile"], "depth_m": spec["depth_m"], "month": m,
                         "value": inc[C.PILE_KEYS.index(key), i] * 10.0, "flag": 0})
    pf = S.pile_frame(pd.DataFrame(rows))
    obs, info = S.observed_piles(pf, "g_piles")
    assert info["tuku"] == {"sensor_depth_m": 400, "verified_months": 12}
    r = S.rank_arm(fz, "g_piles", ["q0", "q0_cs"], obs)
    assert r["pairs"]["q0|q0_cs"]["winner"] == "q0" and max(r["rmse"]["q0"].values()) < 1e-5
    few = S.pile_frame(pd.DataFrame(rows[:7] + rows[12:]))
    assert S.observed_piles(few, "g_piles")[0] is None
    # a split between the piles is not a ranking
    obs2 = obs.copy()
    obs2[1] = O.pile_increments(fz["members"]["q0_cs"]["piles"], fz["piles_months"],
                                "g_piles")[1]
    r2 = S.rank_arm(fz, "g_piles", ["q0", "q0_cs"], obs2)
    assert r2["pairs"]["q0|q0_cs"]["split"] and r2["pairs"]["q0|q0_cs"]["winner"] is None


def test_scorer_wording_only_ranks_decidable_pairs():
    doc = {"osse_result": {"decidability": {"arms": {"g_piles": {
        "q0|q0_cs": {"p": 0.93, "label": "D"}, "page|q0": {"p": 0.43, "label": "narrowing"},
        "page|page_aquitard": {"p": 0.6, "label": "neither"}}}}}}
    pairs = {"q0|q0_cs": {"winner": "q0", "tie": False, "split": False},
             "page|q0": {"winner": "page", "tie": False, "split": False},
             "page|page_aquitard": {"winner": "page", "tie": False, "split": False}}
    w = S.wording(doc, "g_piles", pairs)
    assert "tracks the status-quo level better" in w["q0|q0_cs"]["line"]
    assert "pile/column scale 1" in w["q0|q0_cs"]["line"]
    assert w["page|q0"]["line"].endswith("not decidable (P=0.43)")
    assert "tracks" not in w["page|page_aquitard"]["line"]
    assert C.UNVERIFIED_TRAJECTORY in w["page|page_aquitard"]["line"]


def test_observed_towns_semantics():
    b = pd.DataFrame([{"survey_year": 2027, "county": C.COAST_COUNTY[t], "township": t,
                       "listed_by_all": i < 3, "listed_by_any": i < 4, "ambiguous": False,
                       "peak_rate_upper_cm_yr": 6.0 if i < 3 else 3.0}
                      for i, t in enumerate(C.COAST)])
    st = pd.DataFrame([{"county": "雲林", "survey_year": 2027}])
    obs, ivs = S.observed_towns({"township_bounds.csv": b, "township_status.csv": st})
    o = obs[116]
    assert o["listed"].sum() == 3 and not o["use"][3] and bool(o["informative"])
    assert not bool(obs[117]["informative"])
    assert ivs[116]["雲林縣"] == ("2026-05", "2027-05", "nominal")


# ----------------------------------------------------------------------------- results tree
def test_f300_and_rho_against_their_sources():
    bm = REPO / "dataprep_staging" / "n3_piles" / "build_mapping.py"
    g0 = REPO / C.N3["g0_code"]
    if not (bm.exists() and g0.exists()):
        pytest.skip("bound sources absent")
    assert f"{C.F300} * z12" in bm.read_text(encoding="utf-8")
    assert set(C.RHO_SENS) | {1.0} == set(_literals(g0)["RHOS"])


@needs_results
def test_g_mod_and_f300_against_n3_model_side():
    ms = json.loads((REPO / C.N3["model_side"]).read_text(encoding="utf-8"))
    assert M.sha256(REPO / C.N3["model_side"]) == C.N3["model_side_sha256"]
    assert ms["g_mod_12"]["g_mod"] == C.G_MOD_CM_PER_M
    assert ms["g_mod_12"]["g_mod_cell_report_only"] == C.G_MOD_CELL_CM_PER_M
    for k in C.PILE_KEYS:
        z = ms["zr_level"][k]
        assert z["report_only_f08_12m_cm"] == pytest.approx(C.F300 * z["zr_12m_cm"])


@needs_results
def test_loader_dedup_evidence_on_the_real_members():
    s = M.summary(M.load_members(REPO))
    dd = s["dedup"]
    assert dd["r_piles"]["members"] == dd["g_piles"]["members"]
    assert "q0_s29" not in dd["r_piles"]["members"] and "q0_csk1" not in dd["r_piles"]["members"]
    assert "q0_csk1" in dd["r_towns"]["members"] and "q0_s29" not in dd["r_towns"]["members"]
    assert dd["r_towns"]["evidence"]["q0_csk1"]["townships_differing"] == ["大城鄉"]
    assert s["full_identity"]["q0_s29"]["bytes_equal_all_scenarios_all_cells"]
    assert not s["full_identity"]["q0_csk1"]["bytes_equal_all_scenarios_all_cells"]
    assert all(not v["undeclared_identical"] for v in dd.values())
    assert len(dd["r_piles"]["members"]) == 9 and len(dd["r_towns"]["members"]) == 10
    assert s["towns_months"] == list(C.FREEZE_TOWN_MONTHS)


@needs_results
def test_draft_validates_with_key_fields():
    pytest.importorskip("harness.prereg3")
    doc = O.draft_prereg(4242)
    assert O.validate(doc) == []
    assert doc["kind"] == "test" and doc["id"] == C.ID
    assert [t["role"] for t in doc["tests"]] == ["report_only", "report_only", "gating"]
    assert doc["thresholds"]["p_d"] == 0.8 and doc["thresholds"]["p_narrow"] == 0.5
    assert "NOMINAL" in doc["thresholds"]["bounds"]
    assert set(doc["members"]["spec"]) == set(C.TIER_AB)
    assert doc["attempt_policy"]["scored_runs"].endswith("(score once)")
    assert doc["attempt_policy"]["fixed_paths"]["engineering_record"] == C.ENGINEERING_RECORD
    for k in ("dedup_q0_csk1", "multiplicity", "per_pile_agreement", "roc118_forfeited",
              "not_comparable_to_n3", "xiutan_after_registration", "pile_column_scale"):
        assert k in doc["disclosures"], k
    assert C.LOOK_SCOPE_CORRECTION["unit"] in doc["disclosures"]["own_unit_ids"]
    assert "towns_null_why" in doc["osse_design"]["self_check"]
    assert doc["xiutan_70m"]["status"] == "excluded"
    assert doc["scoring"]["not_before"] == "2030-01-15"
    assert set(O.readiness(doc)) >= {"frozen predictions pending", "OSSE result pending",
                                     "engineering record pending"}
    assert all("/home/" not in i["path"] for i in doc["inputs"])
    assert doc["frozen_code"] == {f: M.sha256(REPO / f) for f in C.CODE_FILES}
    g = doc["tests"][2]
    assert g["fresh_units"] == [C.G_UNIT] and g["external_units"][C.G_UNIT]["window"] == \
        "2028-12..2029-11"
    with pytest.raises(ValueError):
        O.draft_prereg(4242, osse_result={"seed": 4242, "status": "self_check_failed"})
