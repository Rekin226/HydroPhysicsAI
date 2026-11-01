"""CPU tests of the X5.changhua_lever_model.v1 scripts: no forward run, no GPU, nothing
registered, no look logged, nothing written outside pytest's tmp dirs. Every forward file is a
SYNTHETIC fake built here; no head of any real run is read.

    cd <repository root>
    ~/miniforge3/envs/hydro/bin/python -m pytest -q -p no:cacheprovider \\
        dataprep_staging/x5_changhua_lever/tests
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

TESTS = Path(__file__).resolve().parent
X = TESTS.parent
REPO = X.parents[1]
for p in (str(X), str(REPO / "dataprep_staging")):
    if p not in sys.path:
        sys.path.insert(0, p)
import prereg_chlever as P  # noqa: E402
import replay as RP  # noqa: E402
import score_chlever as S  # noqa: E402

DECL = REPO / f"results/dataprep/preregistration/{S.PREREG_ID}.json"
W = S.wells()
DATES = np.array([f"{y}-{m:02d}-01T00:00:00" for y in range(2012, 2033) for m in range(1, 13)])


def fake_npz(path: Path, arm: str, names: tuple[str, ...], heads: dict | None = None,
             fo: dict | None = None, desc: dict | None = None, gate: dict | None = None) -> None:
    """A forward-shaped npz: heads_mean (S, 4, 2148, 252); screens filled from ``heads``."""
    hm = np.zeros((len(names), 4, 2148, 252), np.float32)
    for i, n in enumerate(names):
        if heads and n in heads:
            hm[i][W["layer"], W["cell"], :] = heads[n]
    d = {**S.SCEN_DESC, **(desc or {})}
    np.savez(path, scenario_names=np.array(names),
             scenarios=np.array([d.get(n, f"{n}: recorded") for n in names]),
             dates=DATES, origin=np.int64(131), horizon=np.int64(120), heads_mean=hm,
             member_labels=np.array([f"stage3_datum_gate_{arm}"]), gain=np.float64(1.0),
             hindcast_gain=np.float64(0.0),
             forward_options=np.str_(json.dumps(fo or S.FWD_OPTIONS)),
             gate=np.str_(json.dumps(gate or {"git_commit": "x", "theta": arm})))


def fake_members(path: Path, arm: str, l2: np.ndarray, quantum: float = 0.01) -> None:
    q = np.zeros((1, 2, 2148, 21), np.int16)
    q[0, 0][W["cell"], :] = np.round(l2 / quantum).astype(np.int16)
    np.savez(path, scenario_names=np.array([S.BASE]),
             head_member=np.array([f"stage3_datum_gate_{arm}"] * 2), head_ic=np.array([0, 1]),
             headL2_members_yr=q, head_scale_m=np.float64(quantum))


# ----------------------------------------------------------------------------- static
def test_wells_are_the_class_share_screens():
    assert len(W["sid"]) == 158 and int(W["changhua"].sum()) == 74
    assert S.FROZEN[-1] == "dataprep_staging/x5_class_share/wells_frozen.json"


def test_no_home_paths_in_staged_files():
    pat = re.compile(r"(/home/|-home-)[A-Za-z0-9_]")
    for p in sorted(X.rglob("*")):
        if p.is_file() and p.suffix in (".py", ".sh", ".json"):
            assert not pat.search(p.read_text()), p


def test_runner_syntax_and_dry_run_matches_declared_commands():
    subprocess.run(["bash", "-n", str(X / "run_chlever.sh")], check=True)
    out = subprocess.run(["bash", str(X / "run_chlever.sh"), "forward"], cwd=REPO,
                         env={**os.environ, "DRY": "1"}, capture_output=True, text=True,
                         check=True).stdout.splitlines()
    assert out == [P.command(a) for a in S.ARMS]
    pre = subprocess.run(["bash", str(X / "run_chlever.sh"), "prebuild"], cwd=REPO,
                         env={**os.environ, "DRY": "1"}, capture_output=True, text=True,
                         check=True).stdout.splitlines()
    assert len(pre) == len(S.ARMS) + 1 and all("--scenario" not in s for s in pre)
    assert pre[-1].endswith("--out results/twin_runs/x5_chlever/pre/ccz_m1618_colB")
    assert "coupled_leveling_z3_w1618_tau3650" in pre[-1]


def test_declaration_validates_and_binds():
    from harness import prereg3

    d = P.doc()
    assert prereg3.validate_prereg(d) == []
    assert P.runner_problems() == []
    assert d["frozen_code"] == {p: S.sha256(REPO / p) for p in S.FROZEN}
    declared = {r["path"]: r["sha256"] for r in d["inputs"] if isinstance(r, dict)}
    assert declared == d["frozen_code"]
    m = d["evidence_and_ledger_mapping"]
    assert "No status moves to S or D" in m["scope"]
    assert "speaks for the family" in m["scope"] and "There is no" in m["scope"]
    assert "F < 0.5" in d["thresholds"]["F"]["reading"]
    assert d["thresholds"]["F"]["reading"]["F < 0.5"] == S.WORDING["F_lt_0p5"]
    assert set(d["arms"]) == set(S.ARMS)
    assert "X1" in d["thresholds"]["F"]["replay_source"]
    assert "SEALED" in d["attempt_policy"]["values_written_before_scoring"]
    assert any("forcing_recorded_arm" in s for s in d["rule_risks_disclosure"])


@pytest.mark.skipif(DECL.exists(), reason="declaration registered; refusal test not applicable")
@pytest.mark.parametrize("cmd", ["check", "forward", "score"])
def test_runner_refuses_without_registration(cmd):
    p = subprocess.run(["bash", str(X / "run_chlever.sh"), cmd], cwd=REPO,
                       capture_output=True, text=True)
    assert p.returncode == 1 and f"declare {S.PREREG_ID} first" in p.stderr


@pytest.mark.skipif(DECL.exists(), reason="declaration registered")
def test_scorer_refuses_without_registration(tmp_path, capsys):
    assert S.main(["score", "--out", str(tmp_path / "r.json")]) == 1
    assert not (tmp_path / "r.json").exists()


# ----------------------------------------------------------------------------- replays
def frame(months: list[str], ch: np.ndarray, yl: np.ndarray) -> pd.DataFrame:
    rows = []
    for c, v in (("changhua", ch), ("yunlin", yl)):
        for m, x in zip(months, v, strict=True):
            rows.append({"county": c, "scope": "in_grid", "class": "all", "date": m,
                         "value": float(x)})
            rows.append({"county": c, "scope": "county", "class": "all", "date": m,
                         "value": 99.0})
    return pd.DataFrame(rows)


def test_replay_ratios_mapping_and_fields():
    months = RP.months("2012-01", 132)
    moy = np.array([int(m[5:]) for m in months])
    clim = 10.0 + moy
    rng = np.random.default_rng(0)
    an = rng.normal(0, 0.1, 132)
    df = frame(months, clim * (1 + an), clim * (1 - an))
    df = df.drop(df[(df.county == "changhua") & (df.date == "2016-07")].index)
    r = RP.county_ratios(df)
    assert list(r.index) == months
    assert r.loc["2016-07", "changhua"] == 1.0                  # missing month -> 1
    ref = r.loc[[m for m in months if m != "2016-07"], "changhua"]
    assert np.isfinite(ref).all() and abs(float(np.mean(ref)) - 1) < 0.05
    rep = RP.replay(r, 2014)
    assert rep["month"].iloc[0] == "2023-01" and rep["source"].iloc[0] == "2014-01"
    assert rep["month"].iloc[-1] == "2027-10" and rep["source"].iloc[-1] == "2018-10"
    win = RP.window_anomalies(rep)
    assert win["months"][0] == "2026-11" and win["source"][0] == "2017-11"
    assert len(win["a_changhua"]) == 12
    with pytest.raises(ValueError):
        RP.replay(r, 2019)                                       # would pass 2022-12
    flags = np.zeros(2148, bool)
    flags[:943] = True
    f = RP.per_cell(rep, flags)
    assert f["multiplier"].shape == (58, 2148) and str(f["start"]) == "2023-01"
    assert np.allclose(f["multiplier"][:, 0], rep["r_changhua"])
    assert np.allclose(f["multiplier"][:, -1], rep["r_yunlin"])
    raw = RP.raw_ratios(df)
    fm = RP.filled_months(raw)
    assert fm["changhua"] == {"n_filled": 1, "n_months": 132, "months": ["2016-07"]}
    assert fm["yunlin"]["n_filled"] == 0
    assert RP.filled_months(raw, rep["source"].tolist())["changhua"]["n_filled"] == 1
    assert RP.filled_months(raw, RP.months("2017-01", 58))["changhua"]["n_filled"] == 0
    csv = RP.cut_csv().splitlines()
    assert csv[0] == "month,multiplier" and csv[1] == "2026-01,0.7" and csv[-1] == "2032-12,0.7"
    assert len(csv) == 1 + 84
    assert RP.STARTS == S.REPLAY_YEARS


# ----------------------------------------------------------------------------- step 3
def synthetic_runs(F_true: float = 0.6, scale: dict | None = None, lev: float = 0.4):
    """Screen heads (158, 252) per scenario: an all30 step from 2026-01; county cuts that
    add to it; replays whose window response is F_true x the steady-state response plus a
    per-screen level."""
    rng = np.random.default_rng(1)
    t = np.arange(252)
    step = np.clip((t - 167) / 24.0, 0, 1)                     # ramps to 1 by 2028
    base = rng.normal(0, 1, (158, 1)) + 0.1 * np.sin(t / 6.0)[None, :]
    lam_ch = rng.uniform(0.5, 1.5, 158) * lev / 0.3 * 0.7
    lam_yl = rng.uniform(0.5, 1.5, 158) * lev / 0.3 * 0.3
    ch30 = base + 0.3 * lam_ch[:, None] * step[None, :]
    yl30 = base + 0.3 * lam_yl[:, None] * step[None, :]
    all30 = base + 0.3 * (lam_ch + lam_yl)[:, None] * step[None, :]
    replays = {"replays": {}}
    h = {S.BASE: base, S.ALL30: all30, "CH30": ch30, "YL30": yl30}
    t0, t1 = S.N2C_WINDOW
    for y in S.REPLAY_YEARS:
        a_ch, a_yl = rng.normal(0, 0.1, 12), rng.normal(0, 0.1, 12)
        s = -(lam_ch[:, None] * a_ch[None, :] + lam_yl[:, None] * a_yl[None, :])
        rp_ = base.copy() + np.zeros((158, 252))
        rp_[:, t0:t1] += F_true * s + rng.normal(0, 0.5, (158, 1))
        h[f"RP{y}"] = rp_
        replays["replays"][f"rp{y}"] = {"window": {
            "a_changhua": a_ch.tolist(), "a_yunlin": a_yl.tolist(),
            "source": [f"{y + 3}-11"] + [""] * 11}}
    runs = {"ccz": {"h": h}}
    for arm, k in (scale or {}).items():
        runs[arm] = {"h": {S.BASE: base, S.ALL30: base + k * (all30 - base)}}
    return runs, replays


@pytest.mark.parametrize("F_true,key", [(0.9, "F_ge_0p8"), (0.6, "F_0p5_0p8"),
                                        (0.3, "F_lt_0p5")])
def test_f_gain_recovers_and_bands(F_true, key):
    runs, rep = synthetic_runs(F_true)
    f = S.f_gain(runs["ccz"], W, rep)
    # lambda is estimated from the 2030-32 mean of a ramp that is complete by 2028: exact
    assert f["F"] == pytest.approx(F_true, rel=1e-9)
    assert f["reading_key"] == key and f["evaluable"]
    assert f["caps_c13a_at_N"] is (key == "F_lt_0p5")
    assert len(f["per_replay_report_only"]) == 7


def test_f_not_evaluable_does_not_cap():
    # the 2026-11-01 ruling approved only the F < 0.5 cap
    runs, rep = synthetic_runs(0.9)
    for r in rep["replays"].values():
        r["window"]["a_changhua"] = [0.0] * 12
        r["window"]["a_yunlin"] = [0.0] * 12
    f = S.f_gain(runs["ccz"], W, rep)
    assert f["reading_key"] == "F_not_evaluable" and not f["caps_c13a_at_N"]
    assert "no cap" in f["wording"]


def test_score_spread_T_and_superposition():
    scale = {"ccz_m1618": 0.95, "chc": 1.1, "chs": 0.9, "cpw": 2.0}
    runs, rep = synthetic_runs(0.6, scale)
    q = {a: {"h": {S.BASE: runs["ccz"]["h"][S.BASE],
                   S.ALL30: runs["ccz"]["h"][S.BASE] + k * (runs["ccz"]["h"][S.ALL30]
                                                             - runs["ccz"]["h"][S.BASE])}}
         for a, k in (("q0c", 1.2), ("x5r1", 1.25))}
    res = S.score(runs, q, W, rep, band_arms=["chc", "chs"])
    assert res["evaluable"]
    sp = res["spread"]["by_window"]["2030-32"]
    assert sp == pytest.approx([0.9, 1.1])
    assert res["spread"]["with_disclosed_q0c_x5r1"]["2030-32"] == pytest.approx([0.9, 1.25])
    assert "cpw" in res["spread"]["excluded_report_only"]
    assert res["T_report_only"]["2030-32"] == pytest.approx(1 / 0.95)
    assert res["superposition_report_only"]["by_window"]["2030-32"] == pytest.approx(1.0)
    assert "model-internal lever spread (L)" in res["readings"]["spread_L"]
    assert res["readings"]["caps_c13a_at_N"] is False


def test_score_not_evaluable_when_lever_small_caps_nothing():
    runs, rep = synthetic_runs(0.6, {"ccz_m1618": 1.0}, lev=0.01)
    res = S.score(runs, {}, W, rep, band_arms=[])
    assert not res["evaluable"] and res["evaluable_windows"] == []
    assert res["readings"]["caps_c13a_at_N"] is False
    assert "spread_L" not in res["readings"] and "T_L" not in res["readings"]
    assert "no cap" in res["readings"]["not_evaluable"]
    assert "F" in res                                    # F has its own evaluable_if


def test_score_evaluability_is_per_window():
    # the ramp reaches ~0.52 of the steady lever in 2026-27: 0.036 m vs 0.07 m in 2030-32
    runs, rep = synthetic_runs(0.3, {"ccz_m1618": 1.0, "chc": 1.2}, lev=0.07)
    res = S.score(runs, {}, W, rep, band_arms=["chc"])
    assert res["evaluable_windows"] == ["2030-32"]
    assert set(res["spread"]["by_window"]) == {"2030-32"}
    assert "in 2030-32" in res["readings"]["spread_L"] and "2026-27" in res["readings"][
        "not_evaluable"]
    assert set(res["T_report_only"]) == {"2030-32"}
    # replay contrast too small for F here (RMS s~ < 0.05 m): no F wording, no cap
    assert res["F"]["reading_key"] == "F_not_evaluable"
    assert res["readings"]["caps_c13a_at_N"] is False


# ----------------------------------------------------------------------------- engineering
def test_load_npz_checks(tmp_path):
    p = tmp_path / "a.npz"
    h = {S.BASE: np.ones((158, 252)), S.ALL30: 2 * np.ones((158, 252))}
    fake_npz(p, "chc", S.OTHER_SCEN, h)
    d = S.load_npz(p, S.OTHER_SCEN, W, arm="chc", read=(S.BASE,))
    assert set(d["h"]) == {S.BASE} and float(d["h"][S.BASE].mean()) == 1.0
    fake_npz(p, "chc", S.OTHER_SCEN, h, desc={S.ALL30: "all30: irrigation x0.7"})
    with pytest.raises(S.EngineeringError, match="string"):
        S.load_npz(p, S.OTHER_SCEN, W, arm="chc", read=(S.BASE,))
    fake_npz(p, "chc", S.OTHER_SCEN, h, fo={**S.FWD_OPTIONS, "column_heads": "restart"})
    with pytest.raises(S.EngineeringError, match="forward_options"):
        S.load_npz(p, S.OTHER_SCEN, W, arm="chc", read=(S.BASE,))
    fake_npz(p, "chs", S.OTHER_SCEN, h)
    with pytest.raises(S.EngineeringError, match="member_labels"):
        S.load_npz(p, S.OTHER_SCEN, W, arm="chc", read=(S.BASE,))


def test_disclosed_files_are_checked(tmp_path):
    p = tmp_path / "q.npz"
    names = S.DISCLOSED_SCEN
    h = {S.BASE: np.ones((158, 252)), S.ALL30: 2 * np.ones((158, 252))}
    desc = {"cut30": "cut30: irrigation x0.7 from 2026-01"}
    fake_npz(p, "q0c", names, h, fo=S.DISCLOSED_OPTIONS, desc=desc)
    d = S.load_disclosed(p, "q0c", W)
    assert float((d["h"][S.ALL30] - d["h"][S.BASE]).mean()) == 1.0
    with pytest.raises(S.EngineeringError, match="member_labels"):
        S.load_disclosed(p, "x5r1", W)
    fake_npz(p, "q0c", names, h, fo=S.FWD_OPTIONS, desc=desc)       # save_members yearly
    with pytest.raises(S.EngineeringError, match="forward_options"):
        S.load_disclosed(p, "q0c", W)
    fake_npz(p, "q0c", (S.BASE, S.ALL30), h, fo=S.DISCLOSED_OPTIONS)
    with pytest.raises(S.EngineeringError, match="scenario_names"):
        S.load_disclosed(p, "q0c", W)


def test_declaration_binds_disclosed_files_and_caps_only_f():
    d = P.doc()
    assert set(d["disclosed_files_sha256"]) == set(S.DISCLOSED_NPZ)
    ev = d["thresholds"]["evaluable_if"]
    assert "per window" in ev and "caps nothing" in ev
    assert "no cap" in d["thresholds"]["F"]["evaluable_if"]
    assert "the only cap" in d["evidence_and_ledger_mapping"]["C13a_F"]
    assert "ic-random-stream" in d["thresholds"]["spread_L"]


def test_recorded_arms_bound_to_declared_files():
    rep = {"cuts": {"ch30": {"sha256": "a"}, "yl30": {"sha256": "b"}},
           "replays": {f"rp{y}": {"sha256": str(y)} for y in S.REPLAY_YEARS}}
    dec = S.declared_recorded(rep)
    fo = {"recorded_pumping": {k: {"sha256": v["sha256"], "county": v["county"],
                                   "per_cell": v["per_cell"], "start": v["start"]}
                               for k, v in dec.items()}}
    S.check_recorded(fo, dec)
    fo["recorded_pumping"]["RP2015"]["sha256"] = "x"
    with pytest.raises(S.EngineeringError, match="RP2015"):
        S.check_recorded(fo, dec)
    del fo["recorded_pumping"]["CH30"]
    with pytest.raises(S.EngineeringError, match="recorded arms"):
        S.check_recorded(fo, dec)


def test_attempt_log_binds_files(tmp_path):
    p = tmp_path / "a.npz"
    p.write_bytes(b"x")
    att = tmp_path / "att.jsonl"
    att.write_text(json.dumps({"event": "forward", "arm": "ccz", "rc": 0,
                               "npz_sha256": S.sha256(p)}) + "\n")
    assert S.check_attempts(att, {"ccz": p})["ccz"]
    att.write_text(json.dumps({"event": "forward", "arm": "ccz", "rc": 1,
                               "npz_sha256": S.sha256(p)}) + "\n")
    with pytest.raises(S.EngineeringError):
        S.check_attempts(att, {"ccz": p})


def test_precheck_on_fakes(tmp_path, monkeypatch):
    """single, 18-member and pre-build-only archive kinds, with a column-independence check."""
    monkeypatch.setattr(S, "REPO", tmp_path)
    arms = {k: S.ARMS[k] for k in ("ccz", "ccz_m1618", "rs")}
    monkeypatch.setattr(S, "ARMS", arms)
    pre = tmp_path / "pre"
    fw = tmp_path / "results/twin_forward"
    pre.mkdir()
    fw.mkdir(parents=True)
    rng = np.random.default_rng(2)
    for arm in arms:
        h = rng.normal(0, 1, (158, 252)).astype(np.float32).astype(np.float64)
        fake_npz(pre / f"{arm}.npz", arm, (S.BASE,), {S.BASE: h})
        l2 = rng.normal(0, 1, (158, 21))
        fake_members(pre / f"{arm}.members.npz", arm, l2)
        if arm == "rs":
            fake_npz(fw / "datum_gate_rs.npz", "rs", (S.BASE,), {S.BASE: h + 0.004})
        if arm == "ccz":
            fake_members(fw / "datum_gate_ccz.members.npz", "ccz", l2 + 0.004)
        if arm == "ccz_m1618":
            fake_npz(pre / "ccz_m1618_colB.npz", arm, (S.BASE,), {S.BASE: h})
    res = S.precheck(pre, W)
    assert res["ok_reference"] and res["arms"]["rs"]["ok"]
    assert res["arms"]["rs"]["max_abs_diff_m"] == pytest.approx(0.004, abs=1e-5)
    assert res["arms"]["ccz"]["max_abs_diff_m"] <= 0.01 + 1e-12
    assert res["arms"]["ccz_m1618"]["column_independence"]["ok"]
    assert res["band_arms"] == ["rs"]
    fake_npz(fw / "datum_gate_rs.npz", "rs", (S.BASE,),
             {S.BASE: np.load(pre / "rs.npz")["heads_mean"][0][W["layer"], W["cell"], :] + 0.5})
    res = S.precheck(pre, W)
    assert not res["arms"]["rs"]["ok"] and res["drift_excluded"] == ["rs"]
    assert "ic-random-stream" in res["drift_excluded_reason"]["rs"]
