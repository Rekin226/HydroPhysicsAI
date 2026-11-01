"""CPU tests of N3.G0prime.v1 (G0') and N3Gprime.piles_conversion.v1 (N3G').

Synthetic worlds only: a synthetic model_side.json written here, a synthetic forward archive
for the ring note, a synthetic N3 registration, ruling record, pile frame and heads. Nothing
is written to results/ (every registration, look and output goes to pytest's tmp dirs) and
no real product, archive or observation is opened.

    cd <repository root>
    ~/miniforge3/envs/hydro/bin/python -m pytest -q -p no:cacheprovider \
        dataprep_staging/n3_g0prime/tests
"""
from __future__ import annotations

import datetime as dt
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

TESTS = Path(__file__).resolve().parent
X = TESTS.parent
REPO = X.parents[1]
for _p in (str(X), str(REPO / "dataprep_staging" / "n3_piles"), str(REPO / "dataprep_staging"),
           str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import g0p_lib as P  # noqa: E402
import g0prime as GP  # noqa: E402
import prereg_g0prime as D  # noqa: E402
import prereg_n3g as N  # noqa: E402
import score_n3g as SG  # noqa: E402
from harness import prereg3  # noqa: E402

L = P.L
G0 = P.G0
UTC = dt.timezone.utc
BEFORE_DEADLINE = dt.datetime(2026, 11, 20, 12, 0, tzinfo=UTC)
SCORING_DAY = dt.datetime(2028, 1, 20, 12, 0, tzinfo=UTC)


# ----------------------------------------------------------------------------- synthetic world
def world(g: float = 0.8, seasonal: float = 1.0, seed: int = 1) -> dict:
    """Model world at the two primary piles: column increment = g x nest dD + creep."""
    rng = np.random.default_rng(seed)
    months = L.months(L.REF_MONTH, L.WINDOW24[1])
    t = P.month_index(months)
    tt = np.concatenate([[t[0] - 1], t])
    piles = {}
    for k, amp, ph in (("tuku", 1.2, 0.3), ("huwei", 0.6, 1.0)):
        D_ = (seasonal * amp * np.sin(P.OMEGA * tt + ph) + 0.02 * (tt - tt[0])
              + 0.1 * np.cumsum(rng.normal(0, 1, len(tt))))
        dd = np.diff(D_)
        piles[k] = {"months": months, "inc_cm": (g * dd + 0.03).round(6).tolist(),
                    "dd_nest_m": dd.round(6).tolist()}
    return {"piles": piles}


def model_side(tmp: Path, **kw) -> Path:
    w = world(**kw)
    ms = {"world": w, "informative_model_side_pass": True}
    for h, key in ((12, "g_mod_12"), (24, "g_mod_24")):
        arr = P.window_world(w, P.HORIZONS[h])
        ms[key] = {"g_mod": P.g_mod_lines(arr)["g_mod"]["ols_fe"]}
    p = tmp / "model_side.json"
    p.write_text(json.dumps(ms))
    return p


# ----------------------------------------------------------------------------- conventions
def test_files_header_and_no_home_paths():
    pat = re.compile(r"(/home/|-home-)[A-Za-z0-9_]")
    for f in sorted(X.rglob("*.py")) + [X / "run_g0prime.sh"]:
        s = f.read_text(encoding="utf-8")
        assert not pat.search(s), f
        if f.suffix == ".py":
            body = re.sub(r'^"""(.|\n)*?"""\n', "", s, count=1)
            assert body.lstrip().startswith("from __future__ import annotations"), f
        assert max(len(x) for x in s.splitlines()) <= 100, f


def test_frozen_grid_is_g0s():
    assert (P.SD_GRID, P.PHI_GRID, P.RHOS, P.REPS) == (G0.SD_GRID, G0.PHI_GRID, G0.RHOS,
                                                       G0.REPS)
    assert (P.P_CORRECT, P.P_FALSE_MAX, P.COVERAGE_MIN) == (0.80, 0.10, 0.85)
    assert P.ESTIMATORS == ("harmonic_ratio", "iv_harmonics", "iv_model_heads")
    assert {(0.7, 0), (0.7, 2), (1.3, 0), (1.3, 2)} <= set(P.MISMATCH)
    assert len(P.noise_cells()) == 2 * 2 * 6 * 2
    assert P.RULING_ID == "R7.n3_g0prime"


# ----------------------------------------------------------------------------- estimators
def test_noise_free_recovery_every_line():
    arr = P.window_world(world(g=0.8), P.HORIZONS[12])
    gm = P.g_mod_lines(arr)
    for k in P.ALL_LINES:
        assert gm["g_mod"][k] == pytest.approx(0.8, abs=1e-5)
    assert gm["model_side_f"]["f_model"] == np.inf
    assert gm["model_side_f"]["f_harmonic"] > 10


def test_iv_consistent_where_ols_attenuates():
    rng = np.random.default_rng(3)
    n = 600
    months = [f"{2000 + i // 12}-{i % 12 + 1:02d}" for i in range(n)] * 2
    g = np.array(["a"] * n + ["b"] * n)
    t = P.month_index(months)
    z = 1.0 * np.sin(P.OMEGA * t) + rng.normal(0, 0.5, 2 * n)       # true dD (the model's)
    y = 0.8 * z
    x = z + rng.normal(0, 1.0, 2 * n)                              # nest error
    est = P.estimate_all(y, x, z, months, g, B=50, rng=rng)
    assert est["ols_fe"]["g"] < 0.5
    assert est["iv_model_heads"]["g"] == pytest.approx(0.8, abs=0.05)
    assert est["iv_harmonics"]["g"] == pytest.approx(0.8, abs=0.1)
    assert est["harmonic_ratio"]["g"] == pytest.approx(0.8, abs=0.1)
    assert est["f_model"] > 10 and est["f_harmonic"] > 10


def test_bootstrap_is_seeded_and_shared():
    arr = P.window_world(world(), P.HORIZONS[12])
    y, x, z, m, g = P.draw(arr, 1.0, 0.75, 0.5, 1.0, 0, "none", 0.8,
                           np.random.default_rng(0))
    a = P.estimate_all(y, x, z, m, g, B=200, rng=np.random.default_rng(5))
    b = P.estimate_all(y, x, z, m, g, B=200, rng=np.random.default_rng(5))
    assert a == b
    v = P.verdicts(a, {k: 0.8 for k in P.ALL_LINES})
    assert set(v) == set(P.ALL_LINES)
    assert all(r["rule"].startswith(L.SHARED_SCORER) for r in v.values() if "rule" in r)


def test_mismatch_scales_and_shifts_the_annual_component():
    arr = P.window_world(world(seasonal=2.0), P.HORIZONS[24])
    a = arr["tuku"]
    same = P.mismatch_dd(a["dd"], a["months"], a["ref"], 1.0, 0)
    assert np.allclose(same, a["dd"])
    A, B = P.seasonal_fit(a["dd"], a["months"], a["ref"])
    A2, B2 = P.seasonal_fit(P.mismatch_dd(a["dd"], a["months"], a["ref"], 1.3, 0),
                            a["months"], a["ref"])
    assert np.hypot(A2, B2) == pytest.approx(1.3 * np.hypot(A, B), rel=1e-6)
    A3, B3 = P.seasonal_fit(P.mismatch_dd(a["dd"], a["months"], a["ref"], 1.0, 2),
                            a["months"], a["ref"])
    assert np.hypot(A3, B3) == pytest.approx(np.hypot(A, B), rel=1e-6)
    lag = (np.arctan2(B, A) - np.arctan2(B3, A3)) % (2 * np.pi)
    assert lag == pytest.approx(2 * P.OMEGA, abs=1e-6) or \
        lag == pytest.approx(2 * np.pi - 2 * P.OMEGA, abs=1e-6)


def test_draw_g0_world_when_no_deviation():
    arr = P.window_world(world(), P.HORIZONS[12])
    y, x, z, m, g = P.draw(arr, 2.0, 0.75, 0.5, 1.0, 0, "none", 0.8,
                           np.random.default_rng(1))
    inc = np.concatenate([a["inc"] for a in arr.values()])
    assert np.allclose(y, np.round(2.0 * inc / 0.1) * 0.1)
    assert np.allclose(z, np.concatenate([a["dd"] for a in arr.values()]))
    assert m[:12] == L.months(*L.WINDOW) and len(g) == 24


def test_mismatch_biases_every_nest_estimator_as_declared():
    """amp != 1: every nest-based estimator reads ~rho / amp (not identifiable from the
    nest); lag 2: (a) unbiased (phase-free), (b) ~cos 60 = 0.5 x rho, (c) above rho."""
    arr = P.window_world(world(g=0.8, seasonal=3.0), P.HORIZONS[24])
    gm = P.g_mod_lines(arr)["g_mod"]

    def rho_hat(amp, lag, rho=1.0):
        y, x, z, m, g = P.draw(arr, rho, 1e-9, 0.5, amp, lag, "none", gm["ols_fe"],
                               np.random.default_rng(2))
        est = P.estimate_all(y, x, z, m, g, B=2, rng=np.random.default_rng(0))
        return {k: est[k]["g"] / gm[k] for k in P.ALL_LINES}

    base = rho_hat(1.0, 0)
    for k in P.ESTIMATORS:
        assert base[k] == pytest.approx(1.0, abs=0.02)
    for amp in (0.7, 1.3):
        r = rho_hat(amp, 0)
        for k in ("harmonic_ratio", "iv_harmonics"):
            assert r[k] == pytest.approx(1.0 / amp, rel=0.05), (amp, k)
        lo, hi = sorted((1.0, 1.0 / amp))
        assert lo - 0.02 <= r["iv_model_heads"] <= hi + 0.02
        assert abs(r["iv_model_heads"] - 1.0 / amp) < abs(r["iv_model_heads"] - 1.0)
    r = rho_hat(1.0, 2)
    assert r["harmonic_ratio"] == pytest.approx(1.0, abs=0.05)
    assert r["iv_harmonics"] == pytest.approx(np.cos(2 * P.OMEGA), abs=0.05)
    assert 1.3 < r["iv_model_heads"] <= 2.05
    assert {P.cell_class(*c[2:]) for c in P.noise_cells()} == set(P.CELL_CLASSES)
    assert P.GATE_MODES["identifiable"] == ("g0", "fdev")


# ----------------------------------------------------------------------------- the gate rule
def _cells(p_ok=0.9, cov=0.9, false=0.0, f=20.0, bad_classes=()):
    out = []
    for nc in P.noise_cells():
        bad = P.cell_class(*nc[2:]) in bad_classes
        for rho in P.RHOS:
            line = {"probs": {"conversion_supported": false if rho != 1.0 else p_ok,
                              "direction_supported_size_not": p_ok if rho != 1.0 else 0.0,
                              "not_supported": 0.0, "not_informative": 0.0},
                    "p_correct": p_ok, "coverage": 0.5 if bad else cov}
            out.append({"sd": nc[0], "phi": nc[1], "amp": nc[2], "lag": nc[3], "fdev": nc[4],
                        "rho": rho, "lines": {k: line for k in P.ALL_LINES},
                        "f_median": {"f_harmonic": f, "f_model": f}})
    return out


def test_judge_rule():
    gm = {k: 0.8 for k in P.ALL_LINES}
    mf = {"f_harmonic": 50.0, "f_model": np.inf}
    assert GP.judge(_cells(), "iv_harmonics", mf, gm)["pass"]
    assert not GP.judge(_cells(cov=0.8), "iv_harmonics", mf, gm)["pass"]
    assert not GP.judge(_cells(false=0.2), "harmonic_ratio", mf, gm)["pass"]
    # the per-draw F is reported, not gated (the judge's gate is the model-side F)
    j = GP.judge(_cells(f=5.0), "iv_model_heads", mf, gm)
    assert j["pass"] and j["report_only_draw_f_median_min"]["g0"] == 5.0
    assert GP.judge(_cells(f=5.0), "ols_fe", mf, gm)["pass"]          # reference: G0's 4
    # gate modes: failures in the mismatch classes only
    cm = _cells(bad_classes=("amp", "lag"))
    ja = GP.judge(cm, "iv_harmonics", mf, gm, "all")
    ji = GP.judge(cm, "iv_harmonics", mf, gm, "identifiable")
    assert not ja["pass"] and ji["pass"]
    assert ja["class_pass"] == ji["class_pass"] == {"g0": True, "fdev": True, "lag": False,
                                                    "amp": False}
    assert not GP.judge(_cells(bad_classes=("fdev",)), "iv_harmonics", mf, gm,
                        "identifiable")["pass"]
    assert not GP.judge(_cells(), "iv_harmonics", {"f_harmonic": 3.0, "f_model": 50.0},
                        gm)["pass"]
    assert not GP.judge(_cells(), "harmonic_ratio", mf, {**gm, "harmonic_ratio": -0.1})["pass"]


def test_decision_order():
    def hz(h, passing, cls=None):
        cls = cls or {}
        return {"horizon_months": h, "g_mod": {k: 0.8 for k in P.ALL_LINES},
                "lines": {k: {"pass": k in passing,
                              "class_pass": {c: c in cls.get(k, ()) for c in P.CELL_CLASSES}}
                          for k in P.ALL_LINES}}
    d = GP.decision(hz(12, {"iv_model_heads", "iv_harmonics"}), hz(24, {"harmonic_ratio"}))
    assert (d["passed"], d["horizon_months"], d["estimator"]) == (True, 12, "iv_harmonics")
    d = GP.decision(hz(12, {"ols_fe"}), hz(24, {"iv_model_heads"}))
    assert (d["horizon_months"], d["estimator"]) == (24, "iv_model_heads")
    d = GP.decision(hz(12, set()), hz(24, {"ols_fe"}))
    assert d["passed"] is False and "C31 keeps N" in d["wording"]
    assert d["fail_class"] == "nest_error"
    d = GP.decision(hz(12, set(), {"iv_harmonics": ("g0",)}), hz(24, set()))
    assert d["fail_class"] == "forecast_deviation"
    d = GP.decision(hz(12, set()), hz(24, set(), {"harmonic_ratio": ("g0", "fdev", "lag")}))
    assert d["fail_class"] == "mismatch" and GP.FAIL_WORDING["mismatch"] in d["wording"]


def test_run_once_and_refuses_undeclared(tmp_path):
    ms = model_side(tmp_path)
    with pytest.raises(GP.NotDeclared):
        GP.run(ms, tmp_path / "o0", reps=2, B=20, prereg_dir=tmp_path / "prereg")
    res = GP.run(ms, tmp_path / "o", reps=2, B=20, check_declared=False)
    assert set(res["horizon_12"]["lines"]) == set(P.ALL_LINES)
    assert res["horizon_12"]["g_mod_ols_equals_registered"]
    assert len(res["horizon_24"]["cells"]) == len(P.noise_cells()) * len(P.RHOS)
    assert res["gate"]["mode"] == "all"
    assert set(res["horizon_12"]["lines"]["iv_harmonics"]["class_pass"]) == set(P.CELL_CLASSES)
    with pytest.raises(FileExistsError):
        GP.run(ms, tmp_path / "o", reps=2, B=20, check_declared=False)


def test_engineering_precheck_fails_fast_and_writes_nothing(tmp_path):
    ms = model_side(tmp_path)
    d = json.loads(ms.read_text())
    d["g_mod_12"]["g_mod"] = d["g_mod_12"]["g_mod"] * 1.01
    ms.write_text(json.dumps(d))
    with pytest.raises(GP.EngineeringFailure, match="registered"):
        GP.run(ms, tmp_path / "o", reps=2, B=20, check_declared=False)
    assert not (tmp_path / "o").exists()


# ----------------------------------------------------------------------------- ring note
def test_ring_attenuation_synthetic(tmp_path):
    pytest.importorskip("scipy")
    import ring_attenuation as R

    nx = ny = 10
    T = 132
    dates = pd.date_range("2012-01-01", periods=T + 12, freq="MS")
    t = np.arange(T + 12)
    h = 10 + 1.5 * np.sin(2 * np.pi * t / 12)
    heads = np.broadcast_to(h, (1, 4, nx * ny, T + 12)).astype(np.float32).copy()
    subs = np.broadcast_to(0.01 * (-(h - h[0])), (1, 1, nx * ny, T + 12)).astype(
        np.float32).copy()
    npz = tmp_path / "fw.npz"
    np.savez(npz, dates=np.array([d.strftime("%Y-%m-%d") for d in dates]),
             origin=np.int64(T - 1), heads_mean=heads, subs_mean_by_rheology=subs,
             mask=np.ones((ny, nx), bool), dx=np.float64(1000.0), x0=np.float64(0.0),
             y0=np.float64(0.0))
    sites = tmp_path / "sites.csv"
    pd.DataFrame({"site": ["A", "B"], "x": [2500.0, 6500.0], "y": [2500.0, 7500.0],
                  "secret_value": [1.0, 2.0]}).to_csv(sites, index=False)
    assert list(R.read_sites(sites).columns) == ["site", "x", "y"]
    with pytest.raises(GP.NotDeclared):                    # the declaration check
        R.run(npz, sites, tmp_path / "ring0", reps=5, prereg_dir=tmp_path / "prereg")
    assert not (tmp_path / "ring0").exists()
    res = R.run(npz, sites, tmp_path / "ring", reps=5, check_declared=False)
    assert res["noise_free"]["median_lag0_mm_per_m"] == pytest.approx(10.0, rel=1e-3)
    lam = res["lambda_range_lag0"]
    assert 0 < lam[0] <= lam[1] < 1
    assert "printed unchanged" in res["note"] and "assumed, not measured" in res["note"]
    assert "account for" not in res["note"]
    with pytest.raises(FileExistsError):
        R.run(npz, sites, tmp_path / "ring", reps=5, check_declared=False)


# ----------------------------------------------------------------------------- registrations
def write_ruling(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    p = root / "user_rulings_synthetic.json"
    p.write_text(json.dumps({"schema": D.RULINGS_SCHEMA, "date": "2026-11-01",
                             "ruling_by": "user", "rulings": [
                                 {"id": P.RULING_ID, "question": "G0' and N3G'?",
                                  "decision": "Yes, both (synthetic)",
                                  "applies_to": [P.ID, L.ID]}]}))
    return p


def fake_n3(tmp: Path, ms: Path, pdir: Path, rows: Path) -> None:
    """A minimal registered N3 (synthetic: same ids, units and fields N3G copies)."""
    mp = tmp / "mapping.json"
    mp.write_text(json.dumps({"piles": {
        "tuku": {"nest": [{"sid": "W1", "layer": 1, "cell": 1}]},
        "huwei": {"nest": [{"sid": "W2", "layer": 1, "cell": 2}]}}}))
    units = {L.UNIT: {"description": "synthetic pile months", "ids": ["dbm:a", "dbm:b"],
                      "window": "2026-12..2028-11"},
             L.UNIT_HEADS_YL: {"description": "synthetic nest heads", "ids": ["W1", "W2"],
                               "window": "2026-09..2026-10, 2027-11..2028-11"}}

    def t(tid, scope):
        return {"test_id": tid, "data_scope": scope, "fresh_units": list(units),
                "external_units": units, "seed": L.BOOT_SEED, "role": "report_only",
                "metric": {"observed": "synthetic"}, "decision_rule": {"rule": "N2's"},
                "informative_gate": "synthetic gate"}
    doc = {"id": L.ID, "wp": "N3", "kind": "test", "title": "synthetic N3",
           "tests": [t(L.TEST_CONV, L.SCOPE_CONV12), t(L.TEST_LEVEL, L.SCOPE12),
                     t(L.TEST_CONV24, L.SCOPE_CONV24)],
           "zones": prereg3.not_applicable("x"), "priors": "none",
           "burn_in": prereg3.not_applicable("x"), "folds": prereg3.not_applicable("x"),
           "fresh_units": list(units), "thresholds": {"a": 1},
           "gate_scores": prereg3.not_applicable("x"), "primary_model": {"m": 1},
           "inputs": [str(ms), str(mp)], "interpretation": "synthetic",
           "model_side": {"files": {"model_side.json": {"path": str(ms),
                                                         "sha256": P.sha256(ms)},
                                    "mapping.json": {"path": str(mp),
                                                     "sha256": P.sha256(mp)}}},
           "page_mapping": {"conversion": {"joint": "C31 -> S"}, "never": "no cut number"},
           "disclosures": {"priors_both": "rings 3.4 vs 9.3"},
           "attempt_policy": "scored once", "external_inputs": {"heads": "N2's feed"}}
    prereg3.write_prereg(doc, prereg_dir=pdir, rows_dir=rows)


def test_declaration_then_g0prime_then_n3g(tmp_path, monkeypatch):
    pdir, rows, rdir, out = (tmp_path / n for n in ("prereg", "rows", "rulings", "out"))
    ms = model_side(tmp_path)
    monkeypatch.setattr(P, "MODEL_SIDE_SHA", P.sha256(ms))
    ruling = write_ruling(rdir)
    g0res = tmp_path / "n3_g0_result.json"
    g0res.write_text(json.dumps({"synthetic": True}))
    monkeypatch.setattr(P, "N3_G0_RESULT", str(g0res))
    kw = {"prereg_dir": pdir, "rows_dir": rows, "rulings_dir": rdir, "out_root": out}
    args = ["--model-side", str(ms), "--ruling", str(ruling), "--quiet"]
    assert D.main(args + ["--validate"], **kw) == 0
    # N3 not registered yet: refused
    sha = P.sha256(X / "prereg_g0prime.py")
    monkeypatch.setattr(L, "_now", lambda: BEFORE_DEADLINE)
    gate = ["--gate", "all"]
    assert D.main(args + gate + ["--write", "--expect-self-sha", sha], **kw) == 1
    fake_n3(tmp_path, ms, pdir, rows)
    n3sha = prereg3.load_prereg(L.ID, prereg_dir=pdir, rows_dir=rows)[1]
    # pinned shas (the real ones) do not match the synthetic files: refused
    d0 = D.doc(ms, ruling, P.RULING_ID, pdir, rows, rdir, "all")
    bad = D.problems_for_write(d0, sha, out)
    assert sum("pinned" in b for b in bad) == 4, bad
    monkeypatch.setattr(P, "N3_REG_SHA", n3sha)
    monkeypatch.setattr(P, "RULING_SHA", P.sha256(ruling))
    monkeypatch.setattr(P, "N3_G0_RESULT_SHA", P.sha256(g0res))
    monkeypatch.setattr(P, "MAPPING_SHA", P.sha256(tmp_path / "mapping.json"))
    assert D.main(args + gate + ["--write", "--expect-self-sha", "0" * 64], **kw) == 1
    # the gate mode must be chosen explicitly; 'identifiable' needs a basis
    assert D.main(args + ["--write", "--expect-self-sha", sha], **kw) == 1
    di = D.doc(ms, ruling, P.RULING_ID, pdir, rows, rdir, "identifiable")
    assert any("--gate-basis" in b for b in D.problems_for_write(di, sha, out))
    assert di["design"]["gating_classes"] == ["g0", "fdev"]
    assert D.main(args + gate + ["--write", "--expect-self-sha", sha], **kw) == 0
    decl, _ = prereg3.load_prereg(P.ID, prereg_dir=pdir, rows_dir=rows)
    assert decl["kind"] == "declaration" and set(decl["frozen_code"]) == set(D.FROZEN)
    assert decl["rulings"][P.RULING_ID]["status"] == "ok"
    assert decl["design"]["gate_mode"] == "all" and "construction" in decl["priors"]
    assert set(decl["design"]["power_by_cell_class"]) == set(P.CELL_CLASSES)
    # G0' runs only now (tiny, synthetic); result written once
    res = GP.run(ms, out / "g0prime", reps=2, B=20, prereg_dir=pdir)
    assert res["declaration_index_record"]["id"] == P.ID
    assert res["gate"]["mode"] == "all"
    # an undeclared run (check disabled) is never accepted by N3G, even with a passing decision
    und = GP.run(ms, tmp_path / "undeclared", reps=2, B=20, check_declared=False)
    und.update({"decision": {"passed": True, "horizon_months": 12,
                             "estimator": "iv_model_heads"}})
    for k in P.ALL_LINES:
        und["horizon_12"]["lines"][k]["pass"] = k in ("iv_model_heads", "ols_fe")
    (tmp_path / "und.json").write_text(json.dumps(und))
    decl_reg = D.registered(P.ID, pdir, rows)
    assert N.g0prime_block(tmp_path / "und.json", decl_reg)["status"].startswith(
        "G0' was not run under the registered declaration")
    # a second declaration attempt is refused once a result exists
    assert D.problems_for_write(decl, sha, out)
    # N3G: G0' (2 reps) cannot pass -> refused; a passing decision is forced to test plumbing
    nargs = ["--ruling", str(ruling), "--quiet"]
    nsha = P.sha256(X / "prereg_n3g.py")
    assert N.main(nargs + ["--write", "--expect-self-sha", nsha], **kw) == 1
    r = json.loads((out / "g0prime" / "result.json").read_text())
    r["decision"] = {"passed": True, "horizon_months": 12, "estimator": "iv_model_heads"}
    forced = tmp_path / "forced_result.json"
    forced.write_text(json.dumps(r))
    nargs += ["--g0prime-result", str(forced)]
    monkeypatch.setattr(P, "REPS", 2)
    monkeypatch.setattr(L, "BOOT_B", 20)
    d0 = N.doc(forced, ruling, pdir, rows, rdir, out)
    assert d0["g0prime"]["status"].startswith("the stored decision does not follow")
    for k in P.ALL_LINES:
        r["horizon_12"]["lines"][k]["pass"] = k in ("iv_model_heads", "ols_fe")
    forced.write_text(json.dumps(r))
    assert N.main(nargs + ["--write", "--expect-self-sha", nsha], **kw) == 0
    doc, _ = prereg3.load_prereg(P.N3G_ID, prereg_dir=pdir, rows_dir=rows)
    assert doc["roles"] == {P.TEST_12: "gating", P.TEST_24: "report_only"}
    t12 = doc["tests"][0]
    assert t12["metric"]["estimator"] == "iv_model_heads"
    assert t12["decision_rule"] == {"rule": "N2's"}
    assert t12["fresh_units"] == [L.UNIT, L.UNIT_HEADS_YL]
    assert doc["page_mapping"]["conversion_copied_from_N3"] == {"joint": "C31 -> S"}
    assert doc["n3_registration"]["status"] == "ok"
    # after the deadline: refused (both registrations)
    monkeypatch.setattr(L, "_now", lambda: P.DEADLINE)
    assert any("DEADLINE" in b for b in D.problems_for_write(decl, sha, tmp_path / "x"))
    d2 = N.doc(forced, ruling, pdir, rows, rdir, out)
    assert any("DEADLINE" in b for b in N.problems_for_write(d2, nsha, pdir))

    # ------------------------------------------------------------- scoring (synthetic)
    monkeypatch.setattr(L, "_now", lambda: SCORING_DAY)
    a = SimpleNamespace(out=tmp_path / "score", horizon="12", piles_product=tmp_path / "np",
                        heads_product=tmp_path / "nh", heads_table="h.parquet",
                        stations=None, final=False)
    with pytest.raises(SG.ScoringRefused):                # no product grants
        SG.run_score(a, prereg_dir=pdir, rows_dir=rows)
    monkeypatch.setattr(SG.S, "product_grants", lambda p, use="prospective_scorer": (True, "ok"))
    with pytest.raises(SG.ScoringRefused, match="N3's step-"):
        SG.run_score(a, prereg_dir=pdir, rows_dir=rows)
    for tid in (L.TEST_CONV, L.TEST_LEVEL):
        prereg3.assert_can_score(L.ID, "N3", test_id=tid, prereg_dir=pdir, rows_dir=rows,
                                 verify_inputs=False)
    wd = world()
    pf, heads = obs_frames(wd)
    from common import guard

    class Fake:
        outputs: tuple = ()

        def __init__(self, tab):
            self.tab = tab

        def read_table(self, name):
            return self.tab
    monkeypatch.setattr(guard, "open_product", lambda p, use: Fake(
        pf_raw(pf) if p == a.piles_product else heads_long(heads)))
    rep = SG.run_score(a, prereg_dir=pdir, rows_dir=rows)
    tc = rep["T_conversion"]
    assert tc["role"] == "gating" and tc["estimator"] == "iv_model_heads"
    assert tc["verdict"] == "conversion_supported" and tc["rho"] == pytest.approx(1, abs=0.05)
    assert set(tc["report_only"]) == set(P.ALL_LINES) - {"iv_model_heads"}
    looks = prereg3.read_chained(pdir / "looks.jsonl")
    assert any(lk.get("prereg_id") == P.N3G_ID and lk["test_id"] == P.TEST_12 for lk in looks)
    with pytest.raises(FileExistsError):
        SG.run_score(a, prereg_dir=pdir, rows_dir=rows)


# ----------------------------------------------------------------------------- obs frames
def obs_frames(wd: dict) -> tuple[pd.DataFrame, dict]:
    """Observed = the model exactly (rho 1, no head error): pile increments and one nest
    well per pile whose head path gives the model's nest dD."""
    rows, heads = [], {}
    for key, sid in (("tuku", "W1"), ("huwei", "W2")):
        p = wd["piles"][key]
        spec = L.PILES[key]
        for m, v in zip(p["months"], p["inc_cm"], strict=True):
            rows.append({"pile": spec["pile"], "depth_m": spec["depth_m"], "month": m,
                         "inc_cm": float(v), "year_cm": np.nan})
        h = -np.concatenate([[0.0], np.cumsum(p["dd_nest_m"])])
        idx = [L.prev_month(p["months"][0])] + p["months"]
        heads[sid] = pd.Series(h, index=idx)
    return pd.DataFrame(rows), heads


def pf_raw(pf: pd.DataFrame) -> pd.DataFrame:
    return pf.assign(flag=0, value=pf["inc_cm"] * 10.0, year_value=np.nan)


def heads_long(heads: dict) -> pd.DataFrame:
    return pd.DataFrame([{"sid": s, "month": m, "value": float(v)}
                         for s, ser in heads.items() for m, v in ser.items()])
