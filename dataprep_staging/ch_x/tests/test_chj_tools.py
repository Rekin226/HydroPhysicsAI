"""CPU tests of the CH-J tools added in round 8 (chj_report.py, chj_produce.py): no head, no registry write, no
look logged in the real registry; everything is stubbed or synthetic.

    cd <repository root>
    ~/miniforge3/envs/hydro/bin/python -m pytest -q -p no:cacheprovider \\
        dataprep_staging/ch_x/tests/test_chj_tools.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

TESTS = Path(__file__).resolve().parent
X = TESTS.parent
REPO = X.parents[1]
for p in (str(X), str(REPO / "dataprep_staging")):
    if p not in sys.path:
        sys.path.insert(0, p)
import chj_report as CR  # noqa: E402
import chx_const as K  # noqa: E402


def _block(fractions_outcome: dict) -> dict:
    return {"status": "report_only", "seeds": {
        s: {"sw_outcome": o, "external": {"amplitude_ratio_median_ch": 0.8 + 0.1 * i}}
        for i, (s, o) in enumerate(fractions_outcome.items())}}


def test_r81b_reading_rejected_for_either_seed_and_never_support():
    r = CR.r81b_reading(_block({"2027": "off_bound", "2029": CR.REJECTED}))
    assert r["reading"] == "co_located_recharge_rejected"
    assert r["wording"] == CR.WORDING["rejected"]
    assert r["chj_pass_claimable"] is False and r["off_bound_is_support"] is False
    r = CR.r81b_reading(_block({"2027": "off_bound", "2029": "off_bound"}))
    assert r["reading"] == "report_only_line" and "not support" in r["wording"]
    assert "0.80" in r["wording"] and "0.90" in r["wording"]
    assert CR.r81b_reading({"status": "not_run"})["reading"] == "not_run"


def test_looks_name_the_two_spent_units_with_the_chj_test():
    assert [u for u, _ in CR.units_and_note()] == [K.CHX_UNIT, K.CHX_UNIT_YL]
    assert all("after the primary" in n for _, n in CR.units_and_note())


def test_run_logs_the_looks_before_any_head_and_writes_once(tmp_path, monkeypatch):
    order = []
    prim = tmp_path / "report.json"
    prim.write_text("{}")
    for name in ("chj_panel.npz", "chx_panel.npz"):
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "chj_metrics.json").write_text(json.dumps(_metrics()))
    P = {"months": np.array(["2013-01", "2013-02"]), "sid_ch": np.array(["a", "b"]),
         "sid_yl": np.array(["c"])}
    monkeypatch.setattr(CR, "PRIMARY", str(prim))
    monkeypatch.setattr(CR.S, "verify_registration", lambda *a: ({"created": "x"}, "sha"))
    monkeypatch.setattr(CR.S, "check_panel", lambda *a: P)
    monkeypatch.setattr(CR, "log_looks", lambda *a: order.append("looks") or [{"look": 1}])

    def heads(sids, months):
        order.append(f"heads:{len(sids)}")
        return np.zeros((len(sids), len(months))), None, ["note"]

    monkeypatch.setattr(CR.IO, "l1_monthly", heads)
    monkeypatch.setattr(CR.S, "chj_block", lambda *a: order.append("block")
                        or _block({"2027": CR.REJECTED, "2029": "off_bound"}))
    a = SimpleNamespace(out=str(tmp_path / "chj_report.json"),
                        panel=str(tmp_path / "chx_panel.npz"),
                        chj_panel=str(tmp_path / "chj_panel.npz"),
                        chj_metrics=str(tmp_path / "chj_metrics.json"), chj_engineering=None)
    rep = CR.run(a)
    assert order == ["looks", "heads:2", "heads:1", "block"]
    assert rep["r8_1b"]["reading"] == "co_located_recharge_rejected"
    saved = json.loads(Path(a.out).read_text())
    assert saved["after_the_fact_looks"] == [{"look": 1}] and saved["heads_read"] is True
    with pytest.raises(SystemExit, match="written once"):
        CR.run(a)
    prim.unlink()
    a.out = str(tmp_path / "other.json")
    with pytest.raises(SystemExit, match="primary is scored first"):
        CR.run(a)
    assert not Path(a.out).exists()


# ----------------------------------------------------------------------------- R8.1c report
def _metrics(d=0.01, p=0.4, fed=None):
    return {"seeds": {"2027": {"sw_fractions_changhua": {"deliveries": d, "percolation": p},
                               "sw_fraction_changhua": max(d, p) if fed is None else fed}}}


def test_report_requires_both_fractions_and_feeds_their_max():
    assert CR.check_sw_fractions(_metrics()) == {
        "2027": {"deliveries": 0.01, "percolation": 0.4, "fed": 0.4}}
    with pytest.raises(SystemExit, match="R8.1c"):
        CR.check_sw_fractions(_metrics(fed=0.01))            # the min fed instead of the max
    bad = _metrics()
    del bad["seeds"]["2027"]["sw_fractions_changhua"]["percolation"]
    with pytest.raises(SystemExit, match="lack a Changhua sw fraction"):
        CR.check_sw_fractions(bad)
    with pytest.raises(SystemExit, match="no seed"):
        CR.check_sw_fractions({"seeds": {}})


# ----------------------------------------------------------------------------- producer
import chj_produce as PR  # noqa: E402
import chx_lib as L  # noqa: E402


def test_producer_refuses_posterior_tools():
    PR.refuse_posterior(["hindcast", "--seed", "2027"])
    for argv in (["-m", "hydrophysics.twin.uncertainty", "--theta", "t.json"],
                 ["dataprep_staging/n1_gaps/identifiability.py"],
                 ["--theta", "run/stage3_posterior.json"]):
        with pytest.raises(SystemExit, match="posterior tool"):
            PR.refuse_posterior(argv)
    with pytest.raises(SystemExit, match="posterior tool"):
        PR.main(["hindcast", "--seed", "2027", "--device", "stage3_posterior"])


def test_sw_fractions_feed_the_max_and_the_outcome_needs_both_at_the_bound():
    th = {"log_sw_scale": [math.log(0.01), math.log(0.15), math.log(0.3), math.log(0.25)]}
    f = PR.sw_fractions(th)
    assert f["deliveries"] == pytest.approx(0.01) and f["percolation"] == pytest.approx(0.3)
    assert f["fed"] == f["percolation"]
    assert L.chj_sw_outcome(f["fed"]) == "off_bound"
    both = PR.sw_fractions({"log_sw_scale": [math.log(0.01)] * 4})
    assert L.chj_sw_outcome(both["fed"]) == "co_located_recharge_rejected"
    with pytest.raises(SystemExit, match="4"):
        PR.sw_fractions({"log_sw_scale": [0.0, 0.0]})


class _Runner:
    """hindcast.R1cRun's interface on synthetic arrays: heads = cumulative energy."""

    def __init__(self, A, T, changhua, cells):
        rng = np.random.default_rng(1)
        self.E = rng.uniform(1.0, 2.0, (A, T))
        self.changhua = changhua
        self._cells = cells
        self.calls = []

    def heads(self, E):
        self.calls.append(np.asarray(E).copy())
        h = -np.cumsum(np.asarray(E), axis=1)
        return np.stack([h, 2 * h])

    def grid_cells(self, x, y):
        return np.asarray([self._cells[(float(a), float(b))] for a, b in zip(x, y, strict=True)])


def _chx_panel(A=6, n_ch=3, n_yl=2):
    run = L.month_list(*K.RUN_MONTHS)
    ch = np.array([True, True, True, False, False, False])[:A]
    P = {"months": np.array(L.month_list(*K.CHX_WINDOW)), "run_months": np.array(run),
         "changhua_cells": ch, "yunlin_cells": ~ch}
    for n, cells in (("ch", [0, 1, 2][:n_ch]), ("yl", [3, 4][:n_yl])):
        P[f"sid_{n}"] = np.array([f"{n}{i}" for i in range(len(cells))])
        P[f"x_{n}"] = np.arange(len(cells), dtype=float) + (0 if n == "ch" else 100)
        P[f"y_{n}"] = np.zeros(len(cells))
        P[f"layer_{n}"] = np.array([0, 1, 0][:len(cells)])
        P[f"cell_{n}"] = np.array(cells)
        P[f"block_{n}"] = np.array([f"b{i}" for i in range(len(cells))])
    return P


def test_chj_arms_are_the_primary_arms_at_the_chx_screens():
    P = _chx_panel()
    cells = {(float(x), 0.0): int(c) for n in ("ch", "yl")
             for x, c in zip(P[f"x_{n}"], P[f"cell_{n}"], strict=True)}
    m = _Runner(6, 132, P["changhua_cells"], cells)
    out = PR.chj_arms(m, P)
    run = L.month_list(*K.RUN_MONTHS)
    assert np.array_equal(m.calls[1], L.pattern_energy(m.E, run, P["changhua_cells"],
                                                       K.PATTERN_YEAR))
    assert out["A_ch"].shape == (3, 120) and out["B_yl"].shape == (2, 120)
    ws = L.window_slice(run, *K.CHX_WINDOW)
    assert np.array_equal(out["A_ch"][1], 2 * -np.cumsum(m.E, axis=1)[1, ws])
    # Yunlin cells are not in the pattern arm: A == B there
    assert np.array_equal(out["A_yl"], out["B_yl"])
    assert not np.array_equal(out["A_ch"], out["B_ch"])
    m2 = _Runner(6, 132, ~P["changhua_cells"], cells)
    with pytest.raises(SystemExit, match="county map differs"):
        PR.chj_arms(m2, P)
    cells[(0.0, 0.0)] = 5
    with pytest.raises(SystemExit, match="cells differ"):
        PR.chj_arms(_Runner(6, 132, P["changhua_cells"], cells), P)


# ----------------------------------------------------------------------------- synthetic tree
HOOK = "abcdef1234567890abcdef1234567890abcdef12"
PINNED = ["log_T_mid", "pump_split_logit"]
Q0_TH = {"log_T_mid": [1.0], "pump_split_logit": -2.0}


def _sha(p):
    import hashlib

    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _json(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj))
    return p


def _git(commit=HOOK):
    return {"git_commit": commit[:7], "git_commit_full": commit, "git_dirty": False}


def _run(root, suffix, seed, theta, meta, rng, cap, folds):
    """One run directory as calibrate_flow / merge_folds / calibrate_coupled / forward
    leave it: theta, flow csv, fold files, per-entry dump, kfold wells, fit summary,
    column and forward npz."""
    import pandas as pd

    rel = f"results/twin_runs/stage3_datum_gate_{suffix}"
    d = root / rel
    sids = [f"c{i}" for i in range(6)] + [f"y{i}" for i in range(6)]
    meta = {**meta, **_git()}
    _json(d / "stage3_theta.json", {"theta": theta, "meta": meta})
    pd.DataFrame([{"seed": seed, "r2_kfold": 0.6, "git_commit": HOOK[:7]}]).to_csv(
        d / "stage3_flow.csv", index=False)
    for k, fold in enumerate(folds):
        _json(d / f"stage3_fold{k}.json", {"fold": k, "n_folds": 5, "seed": seed,
                                           "sids": sids, "cfg": {"git_commit": HOOK[:7]},
                                           **fold})
    _json(d / "stage3_fit_summary.json", {"sids": sids})
    T = 24
    obs = rng.normal(0, 1, (12, T)).cumsum(axis=1)
    np.savez(d / "stage3_per_entry.npz", entry=np.arange(12), pred=obs + rng.normal(
        0, 0.5, (12, T)), obs=obs)
    pd.DataFrame({"entry": range(12), "sid": sids}).to_csv(d / "stage3_kfold_wells.csv",
                                                           index=False)
    _json(d / "coupled_leveling/vep_zonal_leveling.json",
          {"theta": f"{rel}/stage3_theta.json", "fold_seed": seed, "r2_outoffold": 0.65,
           "ske_skv_max": cap, "zone_names": ["a", "b"],
           "zonal": [{"log_ske": -5.0, "log_skv": -3.0}] * 2,
           "r2_outoffold_by_zone": {"county:changhua": {"r2": 0.1},
                                    "county:rest": {"r2": 0.7}}})
    dates = pd.date_range("2012-01-31", "2032-12-31", freq="ME")
    (root / "results/twin_forward").mkdir(parents=True, exist_ok=True)
    np.savez(root / f"results/twin_forward/datum_gate_{suffix}.npz", dates=dates.values,
             subs_mean=np.cumsum(rng.uniform(0, 0.01, (6, len(dates))), axis=1)[None],
             gate=json.dumps(meta), member_labels=np.array([d.name]))
    return d


def _q0_fold_theta(k):
    return {"log_T_mid": [1.0 + 0.1 * k], "pump_split_logit": -2.0 - 0.1 * k}


def _chj_meta(q0_sha, sw_sha):
    return {"freeze_from_sha256": q0_sha, "sw_county_split": True,
            "pump_layer_shares_county": list(K.CHJ_SHARES_CHANGHUA),
            "freeze_partition": {"free": ["log_T_mid_changhua"],
                                 "pinned": PINNED + ["log_sw_scale[1]"]},
            "chj_hook": {"posterior_tools_unsupported": True}, "sw_recharge_sha256": sw_sha}


_LOOKS: list = []                 # the no-harm looks the fixture's stub logged


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A synthetic repository holding every file the producer reads: Q0's registered
    theta, fresh Q0 baselines and CH-J runs of both seeds at the hook commit, the fold
    parents, the canal field's provenance, the hindcasts and the CHX panel."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    S, B, X, _, _ = PR._mods()
    monkeypatch.setattr(PR.IO, "REPO", tmp_path)
    monkeypatch.setattr(S, "REPO", tmp_path)
    monkeypatch.setattr(S, "STATIONS", tmp_path / "stations.parquet")
    monkeypatch.setattr(B, "RUNS", tmp_path / "results/twin_runs")
    monkeypatch.setattr(B, "FWD", tmp_path / "results/twin_forward")
    monkeypatch.setattr(B, "CMAP", tmp_path / "cmap.npz")
    rng = np.random.default_rng(3)
    P = _chx_panel()
    np.savez(tmp_path / "cmap.npz", changhua=P["changhua_cells"])
    hc = tmp_path / "results/twin_runs/chx/hindcast"
    hc.mkdir(parents=True)
    np.savez(hc / "chx_panel.npz", **P)
    _json(hc / "chx_panel.json", {"panel_sha256": _sha(hc / "chx_panel.npz")})
    q0 = _json(tmp_path / K.Q0_THETA, {"theta": Q0_TH, "meta": {"pump_layer": 1}})
    sids = [f"c{i}" for i in range(6)] + [f"y{i}" for i in range(6)]
    pd.DataFrame({"sid": sids, "CountyName": ["彰化縣"] * 6 + ["雲林縣"] * 6}).to_parquet(
        tmp_path / "stations.parquet")
    sw = tmp_path / "results/twin_runs/chx/chj_sw/chj_sw.npz"
    sw.parent.mkdir(parents=True)
    sw.write_bytes(b"canal field")
    _json(sw.with_suffix(".json"), {"output_sha256": _sha(sw)})
    for seed in K.CHJ_SEEDS:
        qs, cs = PR.q0_suffix(seed), PR.chj_suffix(seed)
        qfolds = [{"record": {"entry": [k, k + 5], "theta": _q0_fold_theta(k)}}
                  for k in range(5)]
        _run(tmp_path, qs, seed, Q0_TH, {"pump_layer": 1}, rng, None, qfolds)
        cfolds = []
        for k in range(5):
            src = f"results/twin_runs/stage3_datum_gate_{qs}/stage3_fold{k}.json"
            par = _json(tmp_path / f"results/twin_runs/chx/chj/parents/q0_s{seed}_fold{k}"
                        "_theta.json",
                        {"theta": _q0_fold_theta(k), "meta": {
                            "pump_layer": 1, "chj_fold_parent": {
                                "fold": k, "seed": seed, "n_folds": 5, "source": src,
                                "source_sha256": _sha(tmp_path / src)}}})
            cfolds.append({"record": {"entry": [k, k + 5],
                                      "theta": {**_q0_fold_theta(k),
                                                "log_T_mid_changhua": [2.0]}},
                           "chj_parent": {"path": str(par.relative_to(tmp_path)),
                                          "sha256": _sha(par)}})
        th = {**Q0_TH, "log_T_mid_changhua": [1.4], "pump_layer_logits_ch": [0.0] * 4,
              "log_sw_scale": [math.log(v) for v in (0.01, 0.15, 0.3, 0.25)]}
        d = _run(tmp_path, cs, seed, th, _chj_meta(_sha(q0), _sha(sw)), rng, 0.1, cfolds)
        arrays = {k: rng.normal(0, 1, (3 if k.endswith("ch") else 2, 120))
                  for k in ("A_ch", "B_ch", "A_yl", "B_yl")}
        f = hc.parent / "chj/hindcast" / f"chj_hindcast_s{seed}.npz"
        f.parent.mkdir(parents=True, exist_ok=True)
        np.savez(f, **arrays)
        _json(f.with_suffix(".json"), {
            "npz_sha256": _sha(f), "theta_sha256": _sha(d / "stage3_theta.json"),
            "chx_panel_sha256": _sha(hc / "chx_panel.npz"), "code_head": HOOK[:7],
            "producer_sha256": _sha(PR.__file__)})
    f = hc.parent / "chj/hindcast/q0_hindcast.npz"
    np.savez(f, A_ch=rng.normal(0, 1, (3, 120)), A_yl=rng.normal(0, 1, (2, 120)))
    _json(f.with_suffix(".json"), {"npz_sha256": _sha(f), "theta_sha256": _sha(q0),
                                   "chx_panel_sha256": _sha(hc / "chx_panel.npz"),
                                   "code_head": HOOK[:7],
                                   "producer_sha256": _sha(PR.__file__)})
    _LOOKS.clear()
    monkeypatch.setattr(PR, "log_noharm_look", lambda sids, *a: _LOOKS.append(sorted(sids))
                        or {"purpose": "design_time"})
    return tmp_path


def _engineering(root, producer_sha, **kw):
    rec = {"hook_commit": HOOK, "hook_sha256": "h", "forcing_version": "Q0 default (R8.1d)",
           "forcing_sha256": "f", "pregate": {"passed": True, "max_abs_diff_m": 0.0},
           "frozen_at": "2026-11-02T00:00:00Z", "producer_sha256": producer_sha,
           "sw_sha256": _sha(root / "results/twin_runs/chx/chj_sw/chj_sw.npz")}
    rec.update(kw)
    return _json(root / PR.ENG, rec)


def _edit(path, fn):
    d = json.loads(Path(path).read_text())
    fn(d)
    Path(path).write_text(json.dumps(d))


@pytest.mark.filterwarnings("ignore")
def test_assemble_writes_what_the_frozen_chj_block_reads(tree, monkeypatch):
    import score_chx as SC

    with pytest.raises(SystemExit, match="missing"):
        PR.assemble()
    _engineering(tree, "not-this-producer")
    with pytest.raises(SystemExit, match="sha256 differs"):
        PR.assemble()
    eng = _engineering(tree, _sha(PR.__file__))
    order = []
    real = PR.build_metrics
    monkeypatch.setattr(PR, "build_metrics",
                        lambda *a: order.append(len(_LOOKS)) or real(*a))
    summ = PR.assemble()
    assert order == [1]                          # the look was logged before any read
    assert _LOOKS[0] == [f"c{i}" for i in range(6)] + [f"y{i}" for i in range(6)]
    assert set(SC.PANEL_SUMMARY_KEYS) <= set(summ)
    panel, metrics = tree / PR.PANEL, tree / PR.METRICS
    met = json.loads(metrics.read_text())
    for seed in ("2027", "2029"):
        s = met["seeds"][seed]
        for blk in ("noharm", "q0"):
            assert set(s[blk]) >= {"yunlin_kfold", "yunlin_county_corr", "oof_leveling",
                                   "changhua_leveling"}
            assert all(np.isfinite(s[blk][k]) for k in ("yunlin_kfold", "oof_leveling"))
        assert set(s["forward"]) >= {"subsidence_2023_25_changhua_m",
                                     "subsidence_2023_25_yunlin_m", "rising_share_changhua"}
        assert "rising_share_changhua" in s["forward_q0"]
        assert s["forward"]["subsidence_2023_25_changhua_m"] > 0      # cumulative sinking
        assert s["sw_fraction_changhua"] == pytest.approx(0.3)
        assert s["sw_fractions_changhua"]["deliveries"] == pytest.approx(0.01)
        assert s["runs"] == {"chj": f"chj_s{seed}", "q0": f"q0_f{seed}"}
        assert s["q0_refit_max_abs_theta_diff"] == 0.0
    assert met["run_provenance"]["q0_f2027"]["column_ske_skv_max"] is None
    assert met["run_provenance"]["chj_s2029"]["column_ske_skv_max"] == 0.1
    assert "Yunlin column zones are capped too" in met["definitions"]["columns_r8_1f"]
    assert "[1] = 0.15" in met["definitions"]["sw_pinned_values"]
    assert "seed" in met["seed_note"] and met["sw_fraction_seed_spread"] == 0.0
    assert CR.check_sw_fractions(met)["2029"]["fed"] == pytest.approx(0.3)
    z = np.load(panel)
    assert {"chjA_ch_2027", "chjB_ch_2027", "chjA_yl_2027", "chjA_ch_2029", "q0_ch"} <= set(z)
    doc = {"created": "2026-11-01T14:14:55+00:00"}
    prov = SC.chj_provenance(doc, eng, panel, metrics, met["seeds"].keys())
    assert prov["gating"] and not prov["problems"]
    P = {k: np.asarray(v) for k, v in _chx_panel().items()}
    obs_ch = np.random.default_rng(0).normal(0, 1, (3, 120))
    obs_yl = np.random.default_rng(1).normal(0, 1, (2, 120))
    block = SC.chj_block(P, obs_ch, obs_yl, panel, metrics, doc, eng,
                         direction_fn=lambda *a: {"verdict": "not_supported",
                                                  "direction": False})
    assert set(block["seeds"]) == {"2027", "2029"}
    assert block["seeds"]["2027"]["sw_outcome"] == "off_bound"
    with pytest.raises(SystemExit, match="produced once"):
        PR.assemble()


def _refused(tree, match):
    _engineering(tree, _sha(PR.__file__))
    with pytest.raises(SystemExit, match=match):
        PR.assemble()
    assert not _LOOKS and not (tree / PR.PANEL).exists()   # refused before any read


RUNS = "results/twin_runs/stage3_datum_gate_"


def test_an_archived_q0_baseline_is_refused(tree):
    _edit(tree / f"{RUNS}q0_f2027/stage3_theta.json",
          lambda d: d["meta"].update(git_commit="43ed1df", git_commit_full="43ed1df" * 5))
    _refused(tree, "q0_f2027: theta fitted at 43ed1df")


def test_a_dirty_or_other_commit_fold_or_flow_is_refused(tree):
    _edit(tree / f"{RUNS}chj_s2029/stage3_fold3.json",
          lambda d: d["cfg"].update(git_commit="1234567"))
    _refused(tree, "fold 3 run at 1234567")


def test_column_caps_follow_r8_1f(tree):
    _edit(tree / f"{RUNS}q0_f2029/coupled_leveling/vep_zonal_leveling.json",
          lambda d: d.update(ske_skv_max=0.1))
    _refused(tree, r"cap 0.1, R8.1f requires None for q0")


def test_an_uncapped_chj_column_is_refused(tree):
    _edit(tree / f"{RUNS}chj_s2027/coupled_leveling/vep_zonal_leveling.json",
          lambda d: d.update(ske_skv_max=None))
    _refused(tree, r"R8.1f requires 0.1 for chj")


def test_a_forward_of_another_theta_is_refused(tree):
    f = tree / f"results/twin_forward/datum_gate_{PR.q0_suffix(2027)}.npz"
    z = dict(np.load(f))
    z["gate"] = json.dumps({"git_commit": "43ed1df"})
    np.savez(f, **z)
    _refused(tree, "not rolled out from this theta")


def test_a_hindcast_of_another_producer_or_commit_is_refused(tree):
    _edit(tree / "results/twin_runs/chx/chj/hindcast/q0_hindcast.json",
          lambda d: d.update(code_head=HOOK[:7] + "-dirty"))
    _refused(tree, "not the hook commit")


def test_another_canal_field_is_refused(tree):
    _edit(tree / f"{RUNS}chj_s2027/stage3_theta.json",
          lambda d: d["meta"].update(sw_recharge_sha256="0" * 64))
    _refused(tree, "sw_recharge sha256 differs")


def test_an_unhooked_run_is_refused(tree):
    _edit(tree / f"{RUNS}chj_s2029/stage3_theta.json", lambda d: d["meta"].pop("chj_hook"))
    _refused(tree, "chj_hook")


def test_panel_shapes_cover_every_per_seed_arm_key():
    P = _chx_panel()
    good = {"months": P["months"], "sid_ch": P["sid_ch"], "sid_yl": P["sid_yl"],
            "chjA_ch_2027": np.zeros((3, 120)), "chjA_yl_2027": np.zeros((2, 120)),
            "q0_ch": np.zeros((3, 120))}
    PR.check_panel_shapes(good, P)
    with pytest.raises(SystemExit, match="chjB_ch_2029"):
        PR.check_panel_shapes({**good, "chjB_ch_2029": np.zeros((3, 119))}, P)
    with pytest.raises(SystemExit, match="chjA_yl_2027"):
        PR.check_panel_shapes({**good, "chjA_yl_2027": np.zeros((3, 120))}, P)


def test_the_axis_and_q0_grid_checks():
    PR.check_axis([f"{m}-01" for m in L.month_list(*K.RUN_MONTHS)])
    with pytest.raises(SystemExit, match="axis starts"):
        PR.check_axis(L.month_list("2011-12", "2022-11"))
    with pytest.raises(SystemExit, match="axis starts"):
        PR.check_axis(L.month_list("2012-01", "2022-11"))

    class G:
        def active_index(self, x, y):
            return {0.0: 0, 1.0: 1, 2.0: 2, 100.0: 3, 101.0: 4}.get(x)

    P = _chx_panel()
    PR.check_cells(PR._GridCells(G()), P)
    P["cell_yl"] = np.array([3, 5])
    with pytest.raises(SystemExit, match="yl screens' cells differ"):
        PR.check_cells(PR._GridCells(G()), P)


def test_commit_ok():
    assert PR.commit_ok(HOOK[:7], HOOK) and PR.commit_ok(HOOK, HOOK[:9])
    assert not PR.commit_ok(HOOK[:7] + "-dirty", HOOK)
    assert not PR.commit_ok("43ed1df", HOOK) and not PR.commit_ok("", HOOK)
    assert not PR.commit_ok(HOOK[:5], HOOK)


# ----------------------------------------------------------------------------- hook_applied
def _check(tree, seed=2027, **kw):
    return PR.check_hook_applied(tree / f"{RUNS}chj_s{seed}", K.Q0_THETA, seed, **kw)


def test_hook_applied_checks_every_fold_holds_out_q0s_partition(tree):
    q0dir = tree / f"{RUNS}q0_f2027"
    assert _check(tree, require_folds=5, q0_fold_dir=q0dir)["folds_checked"] == [0, 1, 2, 3, 4]
    assert PR.main(["check-hook", "--run", str(tree / f"{RUNS}chj_s2027"), "--seed", "2027",
                    "--q0-theta", K.Q0_THETA, "--require-folds", "5",
                    "--q0-fold-dir", str(q0dir)]) == 0
    with pytest.raises(SystemExit, match="seed"):
        PR.check_hook_applied(tree / f"{RUNS}chj_s2027", K.Q0_THETA, 2029)
    with pytest.raises(SystemExit, match="not the fresh Q0 folds"):
        _check(tree, q0_fold_dir=tree / f"{RUNS}q0_f2029")
    run = tree / f"{RUNS}chj_s2027"
    f1 = run / "stage3_fold1.json"
    good = f1.read_text()
    for fn, match in ((lambda d: d["record"].update(entry=[0, 3]), "other entries than Q0"),
                      (lambda d: d.update(sids=["x"]), "sids differs"),
                      (lambda d: d.update(chj_parent=json.loads(
                          (run / "stage3_fold0.json").read_text())["chj_parent"]),
                       "parent of fold 0"),
                      (lambda d: d["record"]["theta"].update(log_T_mid=[9.0]), "pinned key")):
        _edit(f1, fn)
        with pytest.raises(SystemExit, match=match):
            _check(tree)
        f1.write_text(good)
    (run / "stage3_fold4.json").unlink()
    with pytest.raises(SystemExit, match=r"\[0, 1, 2, 3\] checked"):
        _check(tree, require_folds=5)


def test_a_parent_rebuilt_from_the_full_data_q0_theta_is_refused(tree):
    """R8.1e: a fold parent with VALID meta (fold, seed, Q0's fold file and its sha256)
    but Q0's full-data theta -- and a CH-J fold consistent with it -- is refused, because
    the parent's theta is not Q0's fold-K theta."""
    run = tree / f"{RUNS}chj_s2027"
    par = tree / "results/twin_runs/chx/chj/parents/q0_s2027_fold2_theta.json"
    _edit(par, lambda d: d.update(theta=dict(Q0_TH)))
    _edit(run / "stage3_fold2.json", lambda d: (
        d["record"]["theta"].update(Q0_TH),
        d["chj_parent"].update(sha256=_sha(par))))
    with pytest.raises(SystemExit, match="not Q0's fold-2 theta"):
        _check(tree)


# ----------------------------------------------------------------------------- looks
def test_looks_are_logged_once(tmp_path, monkeypatch):
    from harness import prereg as V1
    from harness import prereg3

    calls = []
    monkeypatch.setattr(prereg3, "record_look", lambda *a, **k: calls.append(a) or {"new": 1})
    kw = dict(test_id=K.CHJ_TEST, prereg_id=K.CHX_ID, note="n", prereg_dir=tmp_path)
    assert PR.log_look_once(["U"], "heads:2013-01..2022-12", "after_the_fact", **kw) == \
        {"new": 1}
    assert len(calls) == 1
    V1._append_chained(tmp_path / "looks.jsonl", {
        "time": "t", "logged": "t", "wp": K.WP, "purpose": "after_the_fact",
        "test_id": K.CHJ_TEST, "prereg_id": K.CHX_ID, "units": ["U"],
        "data_scope": "heads:2013-01..2022-12", "members": {}, "note": "n"})
    old = PR.log_look_once(["U"], "heads:2013-01..2022-12", "after_the_fact", **kw)
    assert old["time"] == "t" and len(calls) == 1               # re-used, not re-logged
    PR.log_look_once(["U"], "heads:2013-01..2022-12", "after_the_fact",
                     **{**kw, "note": "other"})
    assert len(calls) == 2
    # the no-harm look's fields
    monkeypatch.setattr(PR, "log_look_once", lambda *a, **k: {"args": a, **k})
    r = PR.log_noharm_look(["w1", "w2"])
    assert r["args"][:3] == ([PR.LOOK_UNIT_WELLS, PR.LOOK_UNIT_LEVELING], PR.LOOK_SCOPE,
                             "design_time")
    assert r["members"][PR.LOOK_UNIT_WELLS]["ids"] == ["w1", "w2"]
    assert "2026-10-23T02:42:44.765773Z" in r["note"] and "user ruling 2026-10-23" in r["note"]


# ----------------------------------------------------------------------------- run_chj.sh
def _run_chj(*args, **env):
    import os
    import subprocess

    e = {k: v for k, v in os.environ.items() if k != "CHJ_FORCING_ARGS"}
    e.update(env)
    return subprocess.run(["bash", str(X / "run_chj.sh"), *args], capture_output=True,
                          text=True, env=e)


def test_run_chj_refuses_a_forcing_argument_and_dry_runs_the_registered_commands():
    r = _run_chj("all", "2027", DRY="1", CHJ_FORCING_ARGS="--pump-census c.parquet")
    assert r.returncode != 0 and "R8.1d" in r.stderr and not r.stdout
    r = _run_chj("all", "2029", DRY="1")
    assert r.returncode == 0, r.stderr
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("[")]
    names = [ln.split("]")[0][1:] for ln in lines]
    assert names == (["chj_s2029_fit"] + [f"chj_s2029_fold{k}" for k in range(5)]
                     + ["chj_s2029_merge", "chj_s2029_column", "chj_s2029_forward",
                        "chj_s2029_hindcast"])
    fit, fold3 = lines[0], lines[4]
    assert "--fit-only" in fit and "stage3_datum_gate_q0/stage3_theta.json" in fit
    assert "--only-fold 3" in fold3 and "parents/q0_s2029_fold3_theta.json" in fold3
    assert "--pump-census" not in r.stdout and "--ske-skv-max 0.1" in lines[7]
    for step in ("q0folds", "q0base"):                      # fresh for BOTH seeds
        r = _run_chj(step, "2027", DRY="1")
        assert r.returncode == 0, r.stderr
        assert "stage3_datum_gate_q0_f2027" in r.stdout
        assert "stage3_datum_gate_q0/stage3_theta.json" not in r.stdout
