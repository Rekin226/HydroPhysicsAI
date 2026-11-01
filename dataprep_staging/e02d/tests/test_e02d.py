"""CPU tests of round 7 ranks 4-6 (dataprep_staging/e02d): DR.relative_drift.v1, the
fold-member extraction, the sigma_idio artefact, U1.decomp_power.v1, the E02d / E02y
registrations and the scorer.

Synthetic only: every array, forward run, prediction file and observation is generated here,
inside a fake repository in pytest's tmp dir (code symlinked from this checkout). No head
product is opened, nothing under results/ of the real checkout is read or written, and every
registration and look goes to a tmp registry.

    cd <repository root>
    ~/miniforge3/envs/hydro/bin/python -m pytest -q -p no:cacheprovider dataprep_staging/e02d
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

TESTS = Path(__file__).resolve().parent
X = TESTS.parent
REAL = X.parents[1]
for _p in (str(X), str(REAL / "dataprep_staging")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import build_artefact as BA  # noqa: E402
import dr_drift as DR  # noqa: E402
import e02dio as io  # noqa: E402
import e02dlib as L  # noqa: E402
import members_e02d as ME  # noqa: E402
import power_e02d as PW  # noqa: E402
import prereg_design as PD  # noqa: E402
import prereg_e02d as PE  # noqa: E402
import score_e02d as SC  # noqa: E402
from harness import prereg3  # noqa: E402

UTC = dt.timezone.utc
BEFORE = dt.datetime(2026, 11, 15, 12, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _no_products(monkeypatch):
    from common import guard

    def refuse(*a, **k):
        raise AssertionError("tests must not open a data product")

    monkeypatch.setattr(guard, "open_product", refuse)
    monkeypatch.setattr(DR, "load_e02_arrays", refuse)


# ----------------------------------------------------------------------------- library
def test_loo_median_matches_brute_force():
    rng = np.random.default_rng(0)
    X_ = rng.normal(size=(2, 11, 6))
    X_[rng.random(X_.shape) < 0.25] = np.nan
    med, n = L.loo_median(X_, 1)
    for d in range(2):
        for w in range(11):
            for t in range(6):
                if not np.isfinite(X_[d, w, t]):
                    assert np.isnan(med[d, w, t])
                    continue
                o = [X_[d, v, t] for v in range(11) if v != w and np.isfinite(X_[d, v, t])]
                assert n[d, w, t] == len(o)
                if o:
                    assert abs(np.median(o) - med[d, w, t]) < 1e-12


def test_relative_cancels_group_shocks_and_needs_others():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(12, 5))
    g = np.array([0] * 6 + [1] * 6)
    shock = np.where(g[:, None] == 0, 50.0, -30.0) * rng.normal(size=(1, 5))
    a, b = L.relative(x, g), L.relative(x + shock, g)
    assert np.allclose(a, b, equal_nan=True)
    x[:2] = np.nan                           # group 0 keeps 4 wells: 3 others < MIN_OTHERS
    assert np.isnan(L.relative(x, g)[:6]).all()


def test_icc_merge_and_grouping_rule():
    r = L.anova_icc([1, 2, 3, 4, 5, 6], [0, 0, 1, 1, 2, 2])
    assert abs(r["icc"] - 0.8823529411764706) < 1e-12
    lab, log = L.merge_small(["L1"] * 3 + ["L2"] * 10 + ["L34"] * 4, L.LAYER_ORDER)
    assert len(set(lab)) == 1 or all((lab == g).sum() >= 8 for g in set(lab))
    assert log[0]["merged"] == "L1"
    assert L.choose_grouping(0.10, 0.20, 2, 3)["grouping"] == "zone"
    assert L.choose_grouping(0.20, 0.20, 2, 3)["grouping"] == "layer"
    assert L.choose_grouping(0.01, 0.04, 2, 3)["grouping"] == "county"
    assert L.choose_grouping(0.30, float("nan"), 1, 1)["grouping"] == "county"


def test_drift_recovers_tau_and_rule():
    assert L.var_factor(1, 0.5) == 1.0 and L.var_factor(10, 0.0) == 1.0
    for drift in (0.0, 0.05):
        a, p = DR.synthetic_arrays(5, drift=drift)
        rep, _ = DR.compute(a, p)
        for c in L.COUNTIES:
            t = rep["counties"][c]["drift"]["tau_m_per_year"]
            assert (t < 0.015) if drift == 0 else (0.035 < t < 0.065)
            assert rep["counties"][c]["rule"]["include_drift"] is (drift > 0)
    r = L.drift_rule(0.0, 0.3, 10.0)
    assert r["include_drift"] is False and r["increase"] == 0.0


def test_dr_compute_grouping_gate_and_failure():
    a, p = DR.synthetic_arrays(3, layer_shock=0.4)
    rep, wells = DR.compute(a, p)
    assert rep["datum_gate"]["ok"] and not rep["engineering_failure"]
    assert all(rep["counties"][c]["grouping"]["grouping"] == "layer" for c in L.COUNTIES)
    assert list(wells.columns) == list(DR.WELL_COLS) and len(wells) == 158
    assert wells["group"].str.contains(":").all()
    bad = p.copy()
    bad.loc[0, "predicted_m"] += 1e-3
    rep2, w2 = DR.compute(a, bad)
    assert rep2["engineering_failure"] and w2 is None


# ----------------------------------------------------------------------------- fake repo
def _abs_registration(d: Path, wells: pd.DataFrame, months, centre: np.ndarray, sha: str,
                      sigma: float = 1.0) -> None:
    d.mkdir(parents=True, exist_ok=True)
    if centre is None:
        _protocol_manifest(d, months, sha)
        return
    rows = []
    for i, s in enumerate(wells["station_id"]):
        for j, m in enumerate(months):
            rows.append({"station_id": s, "layer": int(wells.at[i, "layer"]),
                         "date": str(m.date()), "prediction_origin": "2022-12-01",
                         "predicted_m": centre[i, j], "baseline_m": centre[i, j] - 0.1,
                         "lower_m": centre[i, j] - L.Z90 * sigma,
                         "upper_m": centre[i, j] + L.Z90 * sigma, "model_sha256": sha})
    pd.DataFrame(rows).to_csv(d / "predictions.csv", index=False)
    _protocol_manifest(d, months, sha)


def _protocol_manifest(d: Path, months, sha: str) -> None:
    proto = {"schema": 1, "task": "free-running-head-validation",
             "test_start": str(months[0].date()),
             "test_end": str((months[-1] + pd.offsets.MonthEnd(0)).date()),
             "registered_before": "2026-10-11", "independent_holdout": True,
             "model_sha256": sha, "baseline": "synthetic climatology", "max_rmse_ratio": 1.0,
             "min_stations": 100, "min_months": 6, "coverage_bounds": [0.85, 0.95]}
    (d / "protocol.json").write_text(json.dumps(proto))
    man = {"registered_on": "2026-11-01", "predictions_sha256": io.sha256(d / "predictions.csv"),
           "protocol_sha256": io.sha256(d / "protocol.json"), "model_sha256": sha}
    (d / "manifest.json").write_text(json.dumps(man))


def _page_npz(path: Path, a: dict, members: np.ndarray | None = None, labels=None) -> None:
    W, T = a["predicted"].shape
    hm = np.repeat(a["predicted"][None, None], 4, axis=1).astype("float32")
    hs = np.zeros_like(hm)
    if members is not None:
        hs = np.repeat(members.std(axis=0)[None, None], 4, axis=1).astype("float32")
    np.savez_compressed(path, heads_mean=hm, heads_std=hs,
                        dates=np.array([d.isoformat() for d in a["dates"]]),
                        origin=a["origin"], scenario_names=np.array(["baseline"]),
                        n_members=36, member_labels=np.array(labels or ["x"]),
                        forward_options=json.dumps({"update_method": "innovation"}))


def _member_runs(d: Path, a: dict, labels: list[str], rng) -> np.ndarray:
    """36 synthetic members around the page mean; ic pairs share the hindcast."""
    W, T = a["predicted"].shape
    dev = rng.normal(0, 0.3, size=(18, 2, W, T))
    dev[:, 1, :, : a["origin"] + 1] = dev[:, 0, :, : a["origin"] + 1]
    dev -= dev.mean(axis=(0, 1), keepdims=True)
    mem = a["predicted"][None, None] + dev
    d.mkdir(parents=True, exist_ok=True)
    for k, lab in enumerate(labels):
        for ic, val in ((1, mem[k].mean(axis=0)), (0, mem[k, 0])):
            np.savez_compressed(
                d / f"{lab}_ic{ic}.npz",
                heads_mean=np.repeat(val[None, None], 4, axis=1).astype("float32"),
                dates=np.array([x.isoformat() for x in a["dates"]]), origin=a["origin"],
                scenario_names=np.array(["baseline"]), n_members=2 if ic else 1,
                member_labels=np.array([lab]),
                forward_options=json.dumps({"update_method": "innovation"}))
    return mem.reshape(36, W, T)


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """A fake repository: code symlinked, synthetic results; registry in tmp."""
    import common.paths as CP

    root = tmp_path / "repo"
    root.mkdir()
    for n in ("dataprep_staging", "hydrophysics"):
        (root / n).symlink_to(REAL / n)
    monkeypatch.setattr(io, "REPO", root)
    monkeypatch.setattr(CP, "REPO", root)
    a, pred = DR.synthetic_arrays(11)
    e02 = root / L.E02_DIR
    e02.mkdir(parents=True)
    pred.to_csv(e02 / "predictions.csv", index=False)
    pm = pd.DatetimeIndex(sorted(pd.to_datetime(pred["date"]).unique()))
    _abs_registration(e02, None, pm, None, "synthetic")
    fwd = root / L.FORWARD
    fwd.parent.mkdir(parents=True)
    labels = ["stage3_datum_gate"] + [f"fold{i}" for i in range(5)] + [
        f"foldpost{i}" for i in range(12)]
    mem = _member_runs(root / ME.MEMBERS_OUT, a, labels, np.random.default_rng(4))
    _page_npz(fwd, a, mem, labels)
    (root / L.STATIONS).parent.mkdir(parents=True)
    (root / L.STATIONS).write_bytes(b"synthetic")
    (root / L.WELLS_DIR).mkdir(parents=True)
    (root / L.WELLS_DIR / "x.parquet").write_bytes(b"synthetic")
    rul = root / L.RULINGS
    rul.parent.mkdir(parents=True)
    rul.write_text(json.dumps({"rulings": [{"id": "R7.e02d_residual_read", "decision": "Yes"},
                                           {"id": "R7.heads_years23", "decision": "Yes."}]}))
    th = root / ME.THETAS
    th.mkdir(parents=True)
    (th / "manifest.json").write_text(json.dumps({"labels": labels}))
    monkeypatch.setattr(L, "E02_PRED_SHA", io.sha256(e02 / "predictions.csv"))
    monkeypatch.setattr(L, "E02_PROTOCOL_SHA", io.sha256(e02 / "protocol.json"))
    monkeypatch.setattr(L, "MODEL_SHA", io.sha256(fwd))
    monkeypatch.setattr(io, "now", lambda: BEFORE)
    monkeypatch.setattr(PW, "REPLICATES", 40)
    monkeypatch.setattr(PW, "CAL_REPLICATES", 30)
    gov = PD._governance
    monkeypatch.setattr(PD, "_governance", lambda: {"ok": True, "why": "synthetic"})
    return {"gov": gov, "root": root, "a": a, "pred": pred, "labels": labels, "mem": mem,
            "pdir": tmp_path / "prereg", "rows": tmp_path / "rows"}


def _self_sha(mod) -> str:
    return io.sha256(Path(mod.__file__))


DR_WRITE = ["dr", "--write", "--quiet", "--lead-from", "datum-centre"]


def _dr_registered(f, monkeypatch):
    rc = PD.main(DR_WRITE + ["--expect-self-sha", _self_sha(PD)],
                 prereg_dir=f["pdir"], rows_dir=f["rows"])
    assert rc == 0
    monkeypatch.setattr(DR, "load_e02_arrays", lambda *a, **k: f["a"])
    assert DR.run_real(prereg_dir=f["pdir"], rows_dir=f["rows"]) == 0


def test_dr_refuses_unregistered_and_logs_its_look(fake, monkeypatch):
    with pytest.raises(io.Refused):
        DR.run_real(prereg_dir=fake["pdir"], rows_dir=fake["rows"])
    _dr_registered(fake, monkeypatch)
    looks = prereg3.read_chained(fake["pdir"] / "looks.jsonl")
    assert [lk["purpose"] for lk in looks] == ["design_time_estimation"]
    assert looks[0]["units"] == [L.DR_UNIT] and looks[0]["prereg_id"] == L.DR_ID
    with pytest.raises(io.Refused):
        DR.run_real(prereg_dir=fake["pdir"], rows_dir=fake["rows"])     # runs once
    rc = PD.main(DR_WRITE + ["--expect-self-sha", _self_sha(PD)],
                 prereg_dir=fake["pdir"], rows_dir=fake["rows"])
    assert rc == 1                                                      # no second write
    doc = json.loads((fake["pdir"] / f"{L.DR_ID}.json").read_text())
    assert doc["lead_definition"] == "datum_centre"
    assert "excess_drift_over_6y_m" in doc["outputs"]["page_line"]
    rep = json.loads((fake["root"] / DR.REPORT_DIR / "report.json").read_text())
    assert rep["lead_definition"] == "datum_centre"


def test_dr_write_needs_lead_signoff_and_governance(fake, monkeypatch):
    sha = _self_sha(PD)
    rc = PD.main(["dr", "--write", "--quiet", "--expect-self-sha", sha],
                 prereg_dir=fake["pdir"], rows_dir=fake["rows"])
    assert rc == 1 and not (fake["pdir"] / f"{L.DR_ID}.json").exists()
    monkeypatch.setattr(PD, "_governance", lambda: {"ok": False, "why": "denied"})
    rc = PD.main(DR_WRITE + ["--expect-self-sha", sha], prereg_dir=fake["pdir"],
                 rows_dir=fake["rows"])
    assert rc == 1 and not (fake["pdir"] / f"{L.DR_ID}.json").exists()


def test_governance_check_reads_manifest_metadata(fake):
    d = fake["root"] / PD.GOV_CARD
    real = fake["gov"]                           # the unpatched function
    assert real()["ok"] is False                 # no card in the fake repository
    d.mkdir(parents=True)
    (d / "MANIFEST.json").write_text(json.dumps({
        "product": "heads/L1_clean", "version": "v3", "fit_window_end": "2026-09",
        "allowed_uses": ["datum_fit", "design_time_evidence"], "forbidden_uses": [],
        "time_slicing": {"column": "date", "params_fit_window_end": "2022-12",
                         "encoding": "iso"}, "gate": {"passed": True}}))
    g = real()
    assert g["ok"], g
    assert g["product"] == "heads/L1_clean v3" and len(g["manifest_sha256"]) == 64
    (d / "MANIFEST.json").write_text(json.dumps({
        "product": "heads/L1_clean", "version": "v3", "fit_window_end": "2026-09",
        "allowed_uses": ["design_time_evidence"], "forbidden_uses": [], "gate": {}}))
    assert real()["ok"] is False


def test_members_extract_gate(fake):
    root = fake["root"]
    wells = DR.compute(fake["a"], fake["pred"])[1]
    arrays, gate = ME.extract(wells, fake["labels"], root / ME.MEMBERS_OUT, root / L.FORWARD)
    assert gate["ok"], gate
    assert arrays["members"].shape[0] == 36 and gate["months"] == 214
    assert gate["max_abs_std_diff_m_report_only"] < 1e-4
    lab = fake["labels"][3]
    p = root / ME.MEMBERS_OUT / f"{lab}_ic0.npz"
    with np.load(p) as z:
        d = {k: z[k] for k in z.files}
    d["heads_mean"] = d["heads_mean"] + np.float32(0.01)
    np.savez_compressed(p, **d)
    _, gate2 = ME.extract(wells, fake["labels"], root / ME.MEMBERS_OUT, root / L.FORWARD)
    assert not gate2["ok"]


def _members_and_abs(f):
    root = f["root"]
    assert ME.main(["extract"]) == 0
    wells = pd.read_csv(root / BA.DR_DIR / "wells_e02d.csv", dtype={"station_id": str})
    datum = wells["datum_m"].to_numpy(float)
    chk = root / L.ABS_CHECK_DIR
    chk.mkdir(parents=True)
    for n in ("protocol.json", "manifest.json"):
        shutil.copy(root / L.E02_DIR / n, chk / n)
    e02 = pd.read_csv(root / L.E02_DIR / "predictions.csv", dtype={"station_id": str})
    e02[e02["date"] >= "2026-12-01"].to_csv(chk / "predictions.csv", index=False)
    assert BA.abs_check(chk, root / L.E02_DIR)["ok"]
    alt = root / "alt_check"
    alt.mkdir()
    bad = e02[e02["date"] >= "2026-12-01"].copy()
    bad.loc[bad.index[0], "upper_m"] += 1e-6
    bad.to_csv(alt / "predictions.csv", index=False)
    assert not BA.abs_check(alt, root / L.E02_DIR)["ok"]
    a = f["a"]
    for win in ("B", "C"):
        months = L.month_index(L.WINDOWS[win]["first"], L.WINDOWS[win]["last"])
        ti = [int(np.flatnonzero(a["dates"] == m)[0]) for m in months]
        ids = [str(s) for s in a["ids"]]
        pos = [ids.index(s) for s in wells["station_id"]]
        centre = a["predicted"][pos][:, ti] + datum[:, None]
        _abs_registration(root / L.ABS_DIRS[win], wells, months, centre, L.MODEL_SHA)


def _power(f):
    rc = PD.main(["power", "--write", "--quiet", "--expect-self-sha", _self_sha(PD)],
                 prereg_dir=f["pdir"], rows_dir=f["rows"])
    assert rc == 0
    assert PW.main(["run"], prereg_dir=f["pdir"]) == 0


def test_full_chain_registers_and_rehearses(fake, monkeypatch, tmp_path):
    f = fake
    root = f["root"]
    _dr_registered(f, monkeypatch)
    _members_and_abs(f)
    assert BA.main([]) == 0
    art = pd.read_csv(root / BA.OUT / "sigma_idio_A.csv", dtype={"station_id": str})
    assert set(art["date"]) == {str(m.date()) for m in L.month_index("2026-12", "2027-10")}
    assert np.allclose(art["sigma_idio_m"] ** 2, art["sd_fit_rel_m"] ** 2
                       + art["sigma_ens_rel_m"] ** 2 + art["drift_m"] ** 2)
    assert (art["lead_months"].min(), art["lead_months"].max()) == (48, 58)
    _power(f)
    res = json.loads((root / PW.RESULT_DIR / "result.json").read_text())
    assert not res["engineering_failure"] and res["declaration"]["id"] == L.POWER_ID
    for which in ("e02d", "e02y"):
        mod_sha = _self_sha(PE)
        if which == "e02d" and not json.loads(
                (root / PW.RESULT_DIR / "result.json").read_text())["e02d_registrable"]:
            pytest.skip("tiny-replicate P-gate did not pass (synthetic; not a defect)")
        rc = PE.main([which, "--write", "--quiet", "--expect-self-sha", mod_sha],
                     prereg_dir=f["pdir"], rows_dir=f["rows"])
        assert rc == 0, which
    idx = [r["id"] for r in prereg3.read_chained(f["pdir"] / "INDEX.jsonl")]
    assert idx == [L.DR_ID, L.POWER_ID, L.E02D_ID, L.E02Y_ID]
    doc = json.loads((f["pdir"] / f"{L.E02D_ID}.json").read_text())
    assert doc["tests"][0]["fresh_units"] == [L.WINDOWS["A"]["unit"]]
    assert len(doc["tests"][0]["external_units"][L.WINDOWS["A"]["unit"]]["ids"]) == 158
    # rehearsal: calibrated synthetic errors pass, inflated errors fail
    for scale, want in ((1.0, "pass"), (2.5, "fail")):
        syn = tmp_path / f"syn{scale}"
        syn.mkdir()
        (syn / SC.S.SYNTHETIC_MARKER).write_text("synthetic")
        rng = np.random.default_rng(8)
        common = {c: rng.normal(0, 3.0, size=12) for c in L.COUNTIES}
        e02 = pd.read_csv(root / L.E02_DIR / "predictions.csv", dtype={"station_id": str})
        a = art.set_index(["station_id", "date"])["sigma_idio_m"]
        rows = []
        for _, r in e02.iterrows():
            m = pd.Timestamp(r["date"])
            sig = a.get((r["station_id"], r["date"]), 0.3)
            rows.append({"station_id": r["station_id"], "date": r["date"],
                         "observed_m": r["predicted_m"]
                         + common[L.county_of(r["station_id"])][(m.month + 1) % 12]
                         + scale * sig * rng.standard_normal(), "synthetic": True})
        pd.DataFrame(rows).to_csv(syn / "observations.csv", index=False)
        out = tmp_path / f"out{scale}"
        rc = SC.main(["rehearsal", "--window", "A", "--synthetic", str(syn), "--doc",
                      str(f["pdir"] / f"{L.E02D_ID}.json"), "--artefact-dir",
                      str(root / BA.OUT), "--abs-dir", str(root / L.E02_DIR), "--out",
                      str(out)])
        assert rc == 0
        rep = json.loads((out / "rehearsal_report.json").read_text())
        for c in L.COUNTIES:
            assert rep["relative"]["counties"][c]["verdict"] == want, (scale, c)
    # the real scorer refuses before 2028-01-15
    with pytest.raises(io.Refused):
        SC.run_score(argparse.Namespace(
            window="A", window_product=tmp_path, step5_dir=tmp_path, e02c_report=None,
            ext_step5_dir=None, previous_report=None, new_wells=None, out=tmp_path / "real"),
            prereg_dir=f["pdir"], rows_dir=f["rows"])
    _score_window_b(f, monkeypatch, tmp_path)


def _score_window_b(f, monkeypatch, tmp_path):
    """Window B: the scorer logs both score looks BEFORE step (5) reads any window row,
    then scores relative before absolute; a year-1 product is not_evaluable."""
    import hydrophysics.twin.prospective as P

    root = f["root"]
    monkeypatch.setattr(io, "now", lambda: dt.datetime(2029, 1, 20, tzinfo=UTC))
    monkeypatch.setattr(SC.S, "product_grants_scorer", lambda d: (True, "synthetic grant"))
    y1 = tmp_path / "product_y1"
    y1.mkdir()
    (y1 / "MANIFEST.json").write_text(json.dumps({"parameters": {}}))
    ns = dict(window="B", step5_dir=None, e02c_report=None, ext_step5_dir=None,
              previous_report=None)
    r0 = SC.run_score(argparse.Namespace(window_product=y1, new_wells=y1 / "wells",
                                         out=tmp_path / "b_y1", **ns),
                      prereg_dir=f["pdir"], rows_dir=f["rows"])
    assert r0["verdict"] == "not_evaluable" and "year-1 product" in r0["why"]
    prod = tmp_path / "product"
    (prod / "wells").mkdir(parents=True)
    (prod / "MANIFEST.json").write_text(json.dumps(
        {"parameters": {"prospective_scorer_window": ["2027-11", "2029-10"]}}))
    absb = pd.read_csv(root / L.ABS_DIRS["B"] / "predictions.csv", dtype={"station_id": str})
    rng = np.random.default_rng(9)
    obs = absb[["station_id", "date"]].copy()
    obs["observed_m"] = absb["predicted_m"] + rng.normal(0, 0.3, len(absb))
    obs["synthetic"] = True
    calls = []

    def fake_score(registration, new_wells, out, as_of=None):
        looks = prereg3.read_chained(f["pdir"] / "looks.jsonl")
        calls.append([lk.get("test_id") for lk in looks if lk["purpose"] == "score"])
        pred = pd.read_csv(registration / "predictions.csv", dtype={"station_id": str})
        man = json.loads((registration / "manifest.json").read_text())
        SC.S.rehearse_e02_step5(pred, obs, registration / "protocol.json", man, out)
        return {}

    monkeypatch.setattr(P, "score", fake_score)
    with pytest.raises(io.Refused):           # --new-wells outside the product
        SC.run_score(argparse.Namespace(window_product=prod, new_wells=tmp_path,
                                        out=tmp_path / "b_bad", **ns),
                     prereg_dir=f["pdir"], rows_dir=f["rows"])
    assert calls == []
    rep = SC.run_score(argparse.Namespace(window_product=prod, new_wells=prod / "wells",
                                          out=tmp_path / "b", **ns),
                       prereg_dir=f["pdir"], rows_dir=f["rows"])
    assert calls == [["E02yP.rel_2027-11", "E02yP.abs_2027-11"]]     # looks first, in order
    assert rep["score_looks"] == [["E02yP.rel_2027-11"], ["E02yP.abs_2027-11"]]
    assert rep["absolute"]["wells"] == absb["station_id"].nunique()
    assert "report-only" in rep["step5_protocol_verdict"]
    assert set(rep["wording_absolute"]) == set(L.COUNTIES)
    assert (tmp_path / "b" / "step5" / "evaluation.csv").is_file()


def test_e02d_write_refuses_without_upstream(fake):
    rc = PE.main(["e02d", "--write", "--quiet", "--expect-self-sha", _self_sha(PE)],
                 prereg_dir=fake["pdir"], rows_dir=fake["rows"])
    assert rc == 1
    assert not (fake["pdir"] / f"{L.E02D_ID}.json").exists()


def test_power_self_check_and_decision():
    a, p = DR.synthetic_arrays(3, layer_shock=0.3, drift=0.02)
    rep, wells = DR.compute(a, p)
    design = PW.design_spec(io.jsonable(rep), wells)
    for c in L.COUNTIES:
        cd = design["counties"][c]
        assert cd["wells"] in (84, 74) and len(cd["a_s"]) >= 2
        assert PW.self_check(cd)["ok"]
        cell = PW.cells(cd)[2]
        r = PW.run_cell(cd, cell, 11, 0, 30, 20, 5)
        assert 0 <= r["P_fail"] <= 1 and r["k_p"] > 0
    rows = [{"gating": True, "heavy": False, "coverage": cv, "P_fail": pf, "cell": i}
            for i, (cv, pf) in enumerate(((0.9, 0.1), (0.9, 0.15), (0.75, 0.95),
                                          (0.75, 0.85), (0.8, 0.5)))]
    d = PW.decide(rows)
    assert d["p_gate_pass"] and d["false_fail_max"] == 0.15 and d["power_min_at_0.75"] == 0.85
    rows[3]["P_fail"] = 0.7
    assert not PW.decide(rows)["p_gate_pass"]


def test_scorer_wording_and_absolute():
    doc = {"id": L.E02Y_ID, "page_mapping": {"wording": PE.WORDING},
           "tests": [{"test_id": "E02yP.rel_2027-11",
                      "role_by_county": {"Yunlin": "primary", "Changhua": "report-only"}}]}
    out = {"window": "B", "counties": {
        "Yunlin": {"verdict": "pass", "coverage": 0.9},
        "Changhua": {"verdict": "fail", "coverage": 0.7},
    }}
    w = SC.wording(doc, "rel", out, {"Yunlin": "primary", "Changhua": "report-only"})
    assert "passed" in w["Yunlin"] and "report-only" in w["Changhua"]
    wa = SC.wording(doc, "abs", out)
    assert "no gross under-coverage" in wa["Yunlin"] and "one year's shock" in wa["Changhua"]
    assert not any("calibrated" in s for s in list(w.values()) + list(wa.values()))


def test_runner_syntax_and_dry_run(tmp_path):
    sh = X / "run_e02d.sh"
    subprocess.run(["bash", "-n", str(sh)], check=True)
    env = dict(os.environ, DRY="1")
    r = subprocess.run(["bash", str(sh), "members", "fold0"], env=env, capture_output=True,
                       text=True, check=True)
    lines = [s for s in r.stdout.splitlines() if s.strip()]
    assert len(lines) == 2 and "--ic-members 1" in lines[0] and "--ic-members 0" in lines[1]
    assert all("--rf-timeseries chou-shui-data/data/rf_timeseries_vendor_2012_2022.csv" in s
               and "--update-method innovation --restart-taper-km 5" in s for s in lines)


def test_rehearsal_cli_refuses_results_dir():
    assert DR.main(["rehearsal", "--out", str(REAL / "results" / "x")]) == 1


def test_page_number_is_noise_corrected_and_lead_definition():
    a, p = DR.synthetic_arrays(5, drift=0.0)
    rep, _ = DR.compute(a, p)
    for c in L.COUNTIES:
        pn = rep["counties"][c]["page_numbers"]
        assert pn["page_value_key"] == "excess_drift_over_6y_m"
        exc = pn["excess_drift_over_6y_m"]
        rms = pn["report_only"]["rms_split_half_drift_m"]
        assert exc < rms and "lower bound" in pn["label"]
        assert "NOT a lower bound" in pn["report_only"]["label"]
        assert abs(exc - 6 * rep["counties"][c]["drift"]["tau_m_per_year"]) < 1e-12
    a, p = DR.synthetic_arrays(5, drift=0.05)
    r1, w1 = DR.compute(a, p, "datum_centre")
    r2, _ = DR.compute(a, p, "origin")
    for c in L.COUNTIES:
        assert r1["counties"][c]["lead_years"] > 9 and 4 < r2["counties"][c]["lead_years"] < 5
    anc = L.lead_anchor("origin", w1["t_centre_year"].to_numpy(float))
    assert np.all(anc == L.ORIGIN_ANCHOR_YEAR)
    with pytest.raises(ValueError):
        L.lead_anchor("4y", [2017.0])


def test_zone_ids_outside_range_are_an_engineering_failure():
    a, p = DR.synthetic_arrays(3)
    a["zones"] = np.asarray(a["zones"]).copy()
    a["zones"][0] = 4
    rep, wells = DR.compute(a, p)
    assert rep["engineering_failure"] and wells is None
    assert rep["zone_check"]["unexpected_zone_ids"] == [4]


def test_icc_bootstrap_upper_and_power_bound():
    rng = np.random.default_rng(2)
    W, T = 40, 60
    part = np.array(["a"] * 20 + ["b"] * 20)
    shock = rng.normal(0, 1, size=(2, T))
    E = rng.normal(0, 1, size=(W, T)) + 0.6 * shock[(part == "b").astype(int)]
    pt = L.anova_icc(E, L.cell_labels(part, W, T))["icc"]
    b = L.icc_boot_upper(E, [part], B=100)
    assert b["finite"] == 100 and b["upper"] >= pt * 0.9 and b["upper"] > 0.1
    a, p = DR.synthetic_arrays(3, layer_shock=0.3)
    rep, wells = DR.compute(a, p)
    for c in L.COUNTIES:
        ir = rep["counties"][c]["icc_remaining"]
        cd = PW.county_design(io.jsonable(rep), wells, c)
        want = min(PW.AS_CAP, max(max(ir["icc"], 0.0) + PW.AS_MARGIN, ir["boot_upper90"]))
        assert abs(max(cd["a_s"]) - round(want, 6)) < 1e-9


def test_product_covers_months(tmp_path):
    def prod(name, params):
        d = tmp_path / name
        d.mkdir()
        (d / "MANIFEST.json").write_text(json.dumps({"parameters": params}))
        return d
    y1 = prod("y1", {})
    full = prod("full", {"prospective_scorer_window": ["2026-11", "2029-10"]})
    short = prod("short", {"prospective_scorer_window": ["2026-11", "2028-10"]})
    assert SC.product_covers(y1, "A")[0] and not SC.product_covers(y1, "B")[0]
    assert all(SC.product_covers(full, w)[0] for w in "ABC")
    assert SC.product_covers(short, "B")[0] and not SC.product_covers(short, "C")[0]
    assert not SC.product_covers(tmp_path / "missing", "A")[0]


def test_absolute_outcome_uses_all_registration_wells():
    rng = np.random.default_rng(3)
    months = L.month_index("2027-11", "2028-10")
    rows = []
    for i in range(50):
        sid = ("09" if i < 25 else "07") + f"{i:06d}"
        for m in months:
            rows.append({"station_id": sid, "date": str(m.date()), "observed_m": rng.normal(),
                         "lower_m": -1.645, "upper_m": 1.645})
    out = SC.absolute_outcome(pd.DataFrame(rows), "B")
    assert out["wells"] == 50
    assert all(out["counties"][c]["wells"] == 25 for c in L.COUNTIES)
