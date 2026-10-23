"""The decision page builder (``twin.viewer_app`` + ``twin.app.prep``), on a synthetic fan.

CPU only and fast: a 6 x 5 grid, a forward run with 3 scenarios and 4 members, and a
response basis with a baseline, 12 lever rows and a half-cut check. The spec these follow
is docs/superpowers/specs/2026-09-23-twin-decision-app-redesign.md §8.
"""

from __future__ import annotations

import json
import os
import re
import socket

import numpy as np
import pandas as pd
import pytest

from hydrophysics.twin import viewer_app as va
from hydrophysics.twin.app import prep

T = 252                     # Jan 2012 .. Dec 2032
ORIGIN = 131                # Dec 2022
NX, NY, DX, X0, Y0 = 6, 5, 1000.0, 180000.0, 2620000.0


def _mask():
    m = np.ones((NY, NX), dtype=bool)
    m[0, 0] = m[4, 5] = False
    return m


def _ramp(start_month: int) -> np.ndarray:
    t = np.arange(T, dtype="float64")
    return np.clip(t - start_month, 0, None) / 12.0          # years since start


def _write_inputs(tmp_path, members=True):
    mask = _mask()
    A = int(mask.sum())
    rng = np.random.default_rng(0)
    dates = np.array([str(d)[:19] for d in pd.date_range("2012-01-01", periods=T, freq="MS")],
                     dtype="U19")
    w = 1.0 + rng.random(A)                              # per-cell sinking weight
    base_s = 0.01 * np.outer(w, _ramp(0))                # m, 1 cm/yr x w
    base_h = 10.0 - 0.2 * np.outer(w, _ramp(0))
    resp_irr = -0.002 * np.outer(w, _ramp(168))           # irrigation retired from 2026-01
    resp_aqua = -0.001 * np.outer(w[::-1], _ramp(168))
    # a 2030 row that is NOT the 2026 row shifted: a different magnitude
    resp_irr30 = -0.003 * np.outer(w, _ramp(216))

    def heads_of(s):
        return np.stack([base_h - 50 * (s - base_s) * (1 + 0.1 * k) for k in range(4)])

    # forward run: baseline, cut30, retire_aqua (ensemble means)
    fs = np.stack([base_s, base_s + 0.3 * resp_irr, base_s + resp_aqua])
    fh = np.stack([heads_of(x) for x in fs])
    gate = {"zone_boundaries": "205,182", "git_commit": "abc1234", "n_wells": 10,
            "fix_eta": 0.5, "fix_head_extra": 40.0, "return_flow": True, "learn_spread": True,
            "bounds_hit": {"mid": {"log_S": {"lo": 0, "hi": 2, "n": 4}}},
            "gate": {"r2_kfold": 0.8, "r2_idw": 0.7, "verdict": "PASS", "n_folds": 5}}
    fw = tmp_path / "fwd.npz"
    np.savez(fw, dates=dates, origin=np.int64(ORIGIN), heads_mean=fh.astype("float32"),
             heads_std=np.zeros_like(fh, dtype="float32"), subs_mean=fs.astype("float32"),
             subs_std=np.full_like(fs, 0.002, dtype="float32"),
             scenario_names=np.array(["baseline", "cut30", "retire_aqua"]),
             scenarios=np.array(["baseline: baseline (no change)",
                                 "cut30: irrigation x0.7 from 2026-01",
                                 "retire_aqua: aquaculture x0 from 2026-01"]),
             mask=mask, nx=np.int64(NX), ny=np.int64(NY), dx=np.float64(DX),
             x0=np.float64(X0), y0=np.float64(Y0), n_members=np.int64(4),
             member_labels=np.array(["m0", "m1"]), hindcast_r2=np.array([0.9, 0.91]),
             gate=np.array(json.dumps(gate)))
    # basis: baseline + 6 classes x {2026, 2030} + a half-cut check
    names, rows = ["baseline"], [base_s]
    keys = ["irr", "aqua", "live", "dom", "ind", "oth"]
    for yr, start in (("2026", 168), ("2030", 216)):
        for k, key in enumerate(keys):
            names.append(f"{key}0_{yr}")
            if key == "irr":
                r = resp_irr if yr == "2026" else resp_irr30
            elif key == "aqua":
                r = resp_aqua if yr == "2026" else -0.0008 * np.outer(w[::-1], _ramp(start))
            else:
                r = -0.0001 * (k + 1) * np.outer(w, _ramp(start))
            rows.append(base_s + r)
    names.append("check_irr50_2026")
    rows.append(base_s + 0.5 * resp_irr)
    bsubs = np.stack(rows)
    bheads = np.stack([heads_of(x) for x in bsubs])
    bs = tmp_path / "basis.npz"
    np.savez(bs, dates=dates, origin=np.int64(ORIGIN), subs_mean=bsubs.astype("float32"),
             heads_mean=bheads.astype("float32"), scenario_names=np.array(names),
             mask=mask, nx=np.int64(NX), ny=np.int64(NY), dx=np.float64(DX),
             x0=np.float64(X0), y0=np.float64(Y0))
    if members:
        recs = []
        for i, lab in enumerate(["m0", "m1"]):
            for ic in (0, 1):
                j = 2 * i + ic
                fb = 10.0 + 0.1 * j
                recs.append({"scenario": "baseline", "member": lab, "ic": ic,
                             "subs_forward_cm": fb, "subs_end_cm": 40 + j,
                             "head_L2_change_m": 1.0})
                recs.append({"scenario": "cut30", "member": lab, "ic": ic,
                             "subs_forward_cm": fb - (0.5 + 0.1 * j), "subs_end_cm": 39 + j,
                             "head_L2_change_m": 1.5 + 0.1 * j})
                recs.append({"scenario": "retire_aqua", "member": lab, "ic": ic,
                             "subs_forward_cm": fb - (1.0 + 0.2 * j), "subs_end_cm": 38 + j,
                             "head_L2_change_m": 2.0})
        pd.DataFrame(recs).to_csv(tmp_path / "fwd.members.csv", index=False)
    return fw, bs, A


@pytest.fixture
def quiet(monkeypatch):
    """No real-data side inputs, and no network at build time."""
    monkeypatch.setattr(va, "_ground_idw", lambda cent, log=print: None)
    monkeypatch.setattr(va, "_load_inputs", lambda log=print: None)

    def _no_net(*a, **k):
        raise AssertionError("the builder must not touch the network")

    monkeypatch.setattr(socket.socket, "connect", _no_net)


def _build(tmp_path, **kw):
    fw, bs, A = _write_inputs(tmp_path, members=kw.pop("members", True))
    out = tmp_path / "page.html"
    opts = dict(leveling="none", wells="none", temporal_npz=None, column_csv=None,
                hsr_csv=None, rivers_csv=None, theta_json=None, alt_npz=None,
                rheo_npz=None, log=lambda s: None)
    opts.update(kw)
    pl = va.build(str(fw), str(bs), str(out), townships_csv=None, **opts)
    return pl, out, A


def _payload(out) -> dict:
    html = out.read_text(encoding="utf-8")
    m = re.search(r'<script id="payload" type="application/json">(.*?)</script>', html, re.S)
    return json.loads(m.group(1).replace("<\\/", "</"))


def test_observations_require_forward_lineage(tmp_path, quiet):
    obs = tmp_path / "observations.csv"
    obs.write_text("station_id,date,head_m,layer,datum\ns1,2022-12-01,1,1,synthetic\n")
    with pytest.raises(ValueError, match="identify the supplied observation"):
        _build(tmp_path, observations_csv=str(obs))


# 1 ---------------------------------------------------------------------------------------
def test_cli_builds_one_page_without_network(tmp_path, quiet):
    fw, bs, _ = _write_inputs(tmp_path)
    out = tmp_path / "cli.html"
    va.main(["--forward", str(fw), "--basis", str(bs), "--out", str(out), "--quarter", "3",
             "--delta-step", "12", "--townships", "none", "--basemap", "none",
             "--leveling", "none", "--wells", "none", "--temporal", "none",
             "--column-csv", "none", "--hsr", "none", "--alt-forward", "none",
             "--rheo-forward", "none"])
    html = out.read_text(encoding="utf-8")
    assert html.startswith("<!doctype html>")
    assert "fonts.googleapis" not in html and "plotly" not in html.lower()
    assert list(tmp_path.glob("*.html")) == [out]


# 2 ---------------------------------------------------------------------------------------
def test_payload_parses_with_documented_shapes(tmp_path, quiet):
    _, out, A = _build(tmp_path)
    d = _payload(out)
    Y = len(d["meta"]["years"])
    assert Y == 21 and d["meta"]["years"][0] == 2012 and d["meta"]["years"][-1] == 2032
    assert d["meta"]["years"][d["meta"]["yObs"]] == 2022
    shapes = {k: v["sh"] for k, v in d["arrays"].items()}
    assert shapes["subsBase"] == [A, Y]
    assert shapes["subsBand"] == [4, A, Y]
    assert shapes["headBase"] == [4, A, Y]
    assert shapes["dSubs"] == [12, A, Y]
    assert shapes["dHeadL2"] == [12, A, Y]
    assert shapes["dHeadEnd"] == [12, 4, A]
    assert shapes["solvedSubs0"] == [A, Y] and shapes["solvedHead1"] == [A, Y]
    assert shapes["zone"] == [A] and shapes["town"] == [A] and shapes["rateBase"] == [A]
    assert [s["name"] for s in d["solved"]] == ["cut30", "retire_aqua"]
    assert d["solved"][0]["factors"] == [0.7, 1, 1, 1, 1, 1]
    band = prep.unpack(d["arrays"]["subsBand"])
    assert np.all(band[0] <= band[1]) and np.all(band[2] <= band[3])


# 3 ---------------------------------------------------------------------------------------
def test_superposition_reproduces_a_single_lever(tmp_path, quiet):
    _, out, A = _build(tmp_path)
    d = _payload(out)
    rows = prep.unpack(d["arrays"]["dSubs"])
    fw = np.load(tmp_path / "fwd.npz")
    bs = np.load(tmp_path / "basis.npz")
    ye, _ = prep.year_ends(fw["dates"])
    truth = (bs["subs_mean"][1] - bs["subs_mean"][0])[:, ye] * 100.0
    step = d["arrays"]["dSubs"]["s"]
    got = prep.superpose(rows, [0, 1, 1, 1, 1, 1], 0)          # irrigation retired, 2026
    np.testing.assert_allclose(got, truth, atol=step / 2 + 1e-6)
    half = prep.superpose(rows, [0.5, 1, 1, 1, 1, 1], 0)
    np.testing.assert_allclose(half, 0.5 * truth, atol=step + 1e-6)
    # the synthetic basis is exactly linear, so the reported half-cut error is ~0
    assert d["basisError"]["check_irr50_2026"]["fan_cm"] < 1e-3
    assert d["basisError"]["check_irr50_2026"]["kind"] == "nonlinear"
    assert d["basisError"]["cut30"]["kind"] == "ensemble"
    # and the rescaling to the ensemble is the identity when the basis is the ensemble
    np.testing.assert_allclose(d["calib"]["subs"], 1.0, atol=1e-3)
    assert d["calib"]["source"][:2] == ["solved:cut30", "solved:retire_aqua"]
    assert d["calib"]["source"][2:] == ["assumed"] * 4


# 4 ---------------------------------------------------------------------------------------
def test_2030_start_uses_the_solved_2030_rows(tmp_path, quiet):
    _, out, _ = _build(tmp_path)
    d = _payload(out)
    rows = prep.unpack(d["arrays"]["dSubs"])
    labels = d["basis"]["labels"]
    i26, i30 = labels.index("irrigation_2026"), labels.index("irrigation_2030")
    assert i30 == 6 + i26
    bs = np.load(tmp_path / "basis.npz")
    names = [str(n) for n in bs["scenario_names"]]
    ye, _ = prep.year_ends(np.load(tmp_path / "fwd.npz")["dates"])
    truth30 = (bs["subs_mean"][names.index("irr0_2030")]
               - bs["subs_mean"][0])[:, ye] * 100.0
    np.testing.assert_allclose(rows[i30], truth30, atol=d["arrays"]["dSubs"]["s"])
    shifted = np.zeros_like(rows[i26])
    shifted[:, 4:] = rows[i26][:, :-4]                          # the old page's time shift
    assert np.abs(rows[i30][:, -1] - shifted[:, -1]).max() > 0.1
    np.testing.assert_allclose(prep.superpose(rows, [0, 1, 1, 1, 1, 1], 1), rows[i30])


# 5 ---------------------------------------------------------------------------------------
def test_vertical_scale_is_exag_over_1000(tmp_path, quiet):
    _, out, _ = _build(tmp_path)
    d = _payload(out)
    assert d["vScale"] == pytest.approx(1 / 1000)
    html = out.read_text(encoding="utf-8")
    assert "VS = D.vScale" in html and "* exag * VS" in html
    # spec §3.2: {10, 25, 50}, default 25
    assert re.findall(r'<option value="(\d+)"[^>]*>\d+×', html) == ["10", "25", "50"]
    assert '<option value="25" selected>25×</option>' in html
    assert "exag/40" not in html.replace(" ", "")


# 6 ---------------------------------------------------------------------------------------
def test_fixed_delta_scale_is_symmetric_snapped_and_policy_free(tmp_path, quiet):
    _, out, _ = _build(tmp_path)
    d = _payload(out)
    sc = d["scales"]
    assert sc["dsubs"]["limit"] in prep.DELTA_SNAPS_CM
    assert sc["dhead"]["limit"] in prep.DELTA_SNAPS_M
    assert prep.snap(2.19, prep.DELTA_SNAPS_CM) == 2.0
    assert prep.snap(0.3, prep.DELTA_SNAPS_CM) == 0.25
    # the difference scale rounds up, so its own reference policy never saturates it
    assert prep.snap_up(3.28, prep.DELTA_SNAPS_M) == 5.0
    assert prep.snap_up(2.0, prep.DELTA_SNAPS_CM) == 2.0
    assert prep.snap_up(99.0, prep.DELTA_SNAPS_CM) == max(prep.DELTA_SNAPS_CM)
    assert sc["dsubs"]["limit"] >= sc["dsubs"]["p98"]
    rows = prep.unpack(d["arrays"]["dSubs"])
    a = prep.delta_scale(rows, d["meta"]["yObs"], prep.DELTA_SNAPS_CM)
    assert a["limit"] == sc["dsubs"]["limit"]
    # it is a property of the basis alone: nothing about a policy enters it
    assert set(a) == {"limit", "p98", "lever"}
    # the page colours a Δ map against that limit, symmetric about zero
    html = out.read_text(encoding="utf-8")
    assert "fd.sign * x / fd.lim" in html and "lim: D.scales.dsubs.limit" in html


# 7 ---------------------------------------------------------------------------------------
def test_size_guard_refuses_a_large_page(tmp_path, quiet, monkeypatch):
    with pytest.raises(SystemExit):
        _build(tmp_path, max_bytes=10_000)
    assert not (tmp_path / "page.html").exists()
    monkeypatch.setattr(va, "MAX_BYTES", 20_000)
    with pytest.raises(SystemExit):
        _build(tmp_path)


# 8 ---------------------------------------------------------------------------------------
def test_missing_optional_inputs_still_build(tmp_path, quiet):
    pl, out, _ = _build(tmp_path, members=False)
    d = _payload(out)
    assert d["hsr"] is None and d["leveling"] is None and d["wells"] is None
    assert d["wellsFan"] is None
    assert d["members"] is None and d["townSkill"] is None and d["rivers"] is None
    assert all(c["gwh"] is None for c in d["classes"])
    html = out.read_text(encoding="utf-8")
    for s in ("Rail alignment not loaded", "rail alignment not loaded", "Leveling not loaded",
              "class energy not loaded"):
        assert s in html


def test_optional_hsr_is_sampled_bilinearly(tmp_path, quiet):
    hsr = tmp_path / "hsr.csv"
    ch = np.arange(0, 4.01, 0.25)
    pd.DataFrame({"chainage_km": ch, "x_twd97": X0 + 1000 + ch * 700,
                  "y_twd97": Y0 + 4500 - ch * 700}).to_csv(hsr, index=False)
    st = tmp_path / "st.csv"
    pd.DataFrame({"name_zh": ["甲"], "name_en": ["A"], "x_twd97": [X0 + 1000],
                  "y_twd97": [Y0 + 4500]}).to_csv(st, index=False)
    _, out, _ = _build(tmp_path, hsr_csv=str(hsr), hsr_stations_csv=str(st))
    d = _payload(out)
    h = d["hsr"]
    assert h is not None and len(h["idx"]) == len(h["w"]) == len(h["ch"]) >= 8
    assert np.allclose(np.array(h["w"]).sum(1), 1.0, atol=1e-3)
    assert h["stations"][0]["ch"] == 0.0


# 9 ---------------------------------------------------------------------------------------
def test_paired_member_percentiles_match_a_hand_computation():
    recs = []
    for j in range(5):
        recs.append({"scenario": "baseline", "member": f"m{j}", "ic": 0,
                     "subs_forward_cm": 10.0 + j, "subs_end_cm": 40.0,
                     "head_L2_change_m": 1.0})
        recs.append({"scenario": "cut", "member": f"m{j}", "ic": 0,
                     "subs_forward_cm": 10.0 + j - (j + 1) * 0.1, "subs_end_cm": 39.0,
                     "head_L2_change_m": 1.0 + 0.2 * j})
    out = prep.paired_deltas(pd.DataFrame(recs))
    # deltas are -0.1, -0.2, -0.3, -0.4, -0.5; numpy linear percentiles by hand:
    # p10 at position 0.4 -> -0.46, p50 -> -0.3, p90 at position 3.6 -> -0.14
    np.testing.assert_allclose(out["cut"]["subs_p"], [-0.46, -0.3, -0.14], atol=1e-9)
    np.testing.assert_allclose(out["cut"]["head_p"], [0.08, 0.4, 0.72], atol=1e-9)
    assert out["cut"]["agree"] == 5 and out["cut"]["n"] == 5
    assert out["cut"]["n_sets"] == 5 and out["cut"]["agree_sets"] == 5
    assert out["base"]["n"] == 5


def test_agreement_counts_parameter_sets_not_runs():
    """Two initial fields of one parameter set are one opinion (review 2, must-fix 1)."""
    recs = []
    for j in range(3):
        for ic in (0, 1):
            recs.append({"scenario": "baseline", "member": f"m{j}", "ic": ic,
                         "subs_forward_cm": 10.0, "subs_end_cm": 40.0, "head_L2_change_m": 1.0})
            # set m2 is a tiny dis-benefit on average; its two runs straddle zero
            d = {0: -0.5, 1: -0.4, 2: 0.02 if ic == 0 else -0.01}[j]
            recs.append({"scenario": "cut", "member": f"m{j}", "ic": ic,
                         "subs_forward_cm": 10.0 + d, "subs_end_cm": 40.0,
                         "head_L2_change_m": 1.0})
    out = prep.paired_deltas(pd.DataFrame(recs))["cut"]
    assert out["n"] == 6 and out["agree"] == 5
    assert out["n_sets"] == 3 and out["agree_sets"] == 2
    assert out["ic_spread"] == pytest.approx(0.03)


def test_zone_mixed_flags_points_that_read_two_zones():
    zone = np.array([0, 0, 1, 1])
    # points 0-5 read cell 0 or 1 (zone 0); 6-9 read cells 2-3 (zone 1)
    idx = np.array([[0, 1, -1, -1]] * 6 + [[2, 3, -1, -1]] * 4)
    w = np.where(idx >= 0, 0.5, 0.0)
    m = prep.zone_mixed(idx, w, zone, step_km=0.5, reach_km=1.0)   # reach = 2 points
    assert m.tolist() == [False] * 4 + [True] * 4 + [False] * 2


def test_fast_share_reports_the_first_year_jump():
    dates = [str(d)[:10] for d in pd.date_range("2025-01-01", periods=48, freq="MS")]
    d = np.zeros(48)
    d[12:] = -0.6                              # a jump in the first year of a 2026 start
    d[24:] = -0.6 - np.linspace(0, 0.4, 24)    # then a slow drift to -1.0
    out = prep.fast_share(d, dates, 2026)
    assert out["dec"] == pytest.approx(0.6)
    assert out["peak"] == pytest.approx(0.6) and out["peak_month"] == "2026-01"


def test_rail_tile_and_alternative_model_in_the_payload(tmp_path, quiet):
    """The rail carries its zone-line mask; the structural alternative carries its answers."""
    fw, bs, _ = _write_inputs(tmp_path)
    alt = tmp_path / "alt.npz"
    import shutil

    shutil.copy(fw, alt)
    m = pd.read_csv(tmp_path / "fwd.members.csv")
    m.loc[m.scenario != "baseline", "subs_forward_cm"] = (
        m.loc[m.scenario != "baseline", "subs_forward_cm"] + 0.4)
    m.to_csv(tmp_path / "alt.members.csv", index=False)
    pl, out, _ = _build(tmp_path, alt_npz=str(alt))
    a = pl["alt"]
    assert a["r2_kfold"] == 0.8 and set(a["resp"]) == {"cut30", "retire_aqua"}
    mine = pl["solved"][0]["members"]["subs_mean"]
    assert a["resp"]["cut30"]["subs_mean"] == pytest.approx(mine + 0.4)
    assert a["ratio"][0] == pytest.approx((mine + 0.4) / mine)
    assert a["ratio"][2] is None                       # livestock: no solved run
    html = out.read_text(encoding="utf-8")
    assert "D.hsr.mixed" in html and "structWeak" in html


# data-prep units ---------------------------------------------------------------------------
def test_pack_roundtrip_and_never_clips():
    a = np.array([[0.0, 1.234, -2.5], [np.nan, 400.0, 3.0]])
    p = prep.pack(a, 0.01)
    b = prep.unpack(p)
    assert np.isnan(b[1, 0])
    np.testing.assert_allclose(b[np.isfinite(a)], a[np.isfinite(a)], atol=p["s"])
    assert p["s"] > 0.01                     # 400 cm does not fit int16 at 0.01 cm
    u = prep.pack(np.array([3, 0, 7]), None, "uint8")
    assert prep.unpack(u).tolist() == [3, 0, 7]


def test_angular_distortion_of_a_uniform_tilt():
    ch = np.arange(0, 10.01, 0.25)
    prof = 2.0 * ch                          # 2 cm per km along the line
    ad = prep.angular_distortion(prof, 0.25)
    np.testing.assert_allclose(ad[8:-8], 2e-5, rtol=1e-6)   # 0.02 m / 1000 m


def test_bilinear_weights_skip_inactive_corners():
    mask = np.ones((3, 3), dtype=bool)
    mask[1, 1] = False
    # (1700, 1600) m sits between cell centres (1,1) (inactive), (2,1), (1,2) and (2,2)
    idx, w, ok = prep.bilinear_weights(np.array([1700.0]), np.array([1600.0]), 0, 0, 1000,
                                       mask)
    assert ok[0] and np.isclose(w.sum(), 1.0)
    g = prep.active_index_grid(mask)
    assert g[1, 1] not in idx[0][w[0] > 0]


def test_township_skill_flags_low_confidence():
    lev = {"cell": np.array([0, 0, 1, 1, 1, 1, 1]), "bias": np.array([0, 0, 1, 1, 1, 1, 9.0]),
           "pairs": [(np.array([0, 1.0]), np.array([0, 1.0]))] * 7}
    town_idx = np.array([1, 2], dtype="uint8")
    sk = prep.township_skill(lev, town_idx, 3)
    assert sk[0]["n"] == 0 and sk[0]["low"]                   # the pooled panel, empty
    assert sk[1]["n"] == 2 and sk[1]["low"]                   # fewer than 5 benchmarks
    assert sk[2]["n"] == 5 and sk[2]["r2"] == 1.0 and not sk[2]["low"]
    lev["pairs"] = [(np.array([0, 1.0]), np.array([0, -1.0]))] * 7
    assert prep.township_skill(lev, town_idx, 3)[2]["low"]    # negative R² is poor


def test_township_index_pools_the_unlabelled_half():
    idx, towns = prep.township_index(pd.Series({0: "虎尾鎮", 2: "林內鄉"}), 4)
    pos = {t["zh"]: i for i, t in enumerate(towns)}
    assert towns[0]["pooled"] and idx.tolist() == [pos["虎尾鎮"], 0, pos["林內鄉"], 0]
    assert {t["en"] for t in towns[1:]} == {"Huwei", "Linnei"}


def test_model_artefacts_find_a_restart_step():
    s = np.r_[np.linspace(0, 10, 3), np.linspace(10.5, 20, 130)]
    s = np.r_[s[:132], s[131] + np.r_[2.0, 4.0, 5.0, 5.2, 5.4, 5.6],
              s[131] + 5.6 + 0.1 * np.arange(1, 115)]
    art = prep.model_artefacts(s, 131)
    assert art["restart_cm"] == pytest.approx(5.6 - 6 * 0.1, abs=1e-6)
    assert art["trend_cm_yr"] == pytest.approx(1.2)


def test_diverging_palette_survives_deuteranopia():
    pal = prep.PALETTES["diverging"]
    assert prep.deltaE_deutan(pal[0], pal[-1]) >= 20
    assert prep.deltaE_deutan(pal[0], pal[len(pal) // 2]) >= 20


def test_basis_rows_and_policy_parsing():
    rows = prep.basis_rows(["baseline", "irr0_2026", "aqua0_2030", "check_irr50_2026"])
    assert rows == {("irrigation", 2026): 1, ("aquaculture", 2030): 2}
    assert va._parse_policy("cut30: irrigation x0.7 from 2026-01") == (
        [0.7, 1, 1, 1, 1, 1], 2026)
    assert va._parse_policy("baseline: baseline (no change)") is None


def test_template_has_no_remote_fonts_or_plotly():
    with open(va.TEMPLATE, encoding="utf-8") as fh:
        html = fh.read()
    assert "googleapis" not in html and "plotly" not in html.lower()
    assert os.path.basename(va.THREE_CDN) == "three.min.js" and "cdnjs" in va.THREE_CDN


# review fixes (M1-M5, S1) -------------------------------------------------------------------
def test_forward_change_is_measured_after_the_restart(tmp_path, quiet):
    _, out, _ = _build(tmp_path)
    d = _payload(out)
    yrs = d["meta"]["years"]
    assert yrs[d["meta"]["yObs"]] == 2022 and yrs[d["meta"]["yRef"]] == 2023
    # without the held-out test there is no tested horizon beyond the fit
    assert d["meta"]["yTested"] == d["meta"]["yObs"]


def test_historical_test_never_validates_future_calendar_years(tmp_path, quiet):
    rng = np.random.default_rng(1)
    obs = rng.normal(size=(5, 131))
    tz = tmp_path / "temporal.npz"
    np.savez(tz, obs=obs, clim=obs + 0.1, temporal_spreadL=obs + 2.0, T_fit=np.int64(95))
    _, out, _ = _build(tmp_path, temporal_npz=str(tz))
    d = _payload(out)
    tp = d["modelcard"]["temporal"]
    assert tp["months"] == 36 and tp["fit_months"] == 95 and tp["passed"] is False
    assert d["meta"]["years"][d["meta"]["yTested"]] == 2022


def test_artefact_steps_are_removed_only_in_the_selected_cells():
    T, origin = 60, 30
    t = np.arange(T, dtype="float64")
    trend = 0.1 * t
    step = np.where(t >= 2, 50.0, 25.0 * t)                  # a start-up jump in months 0-2
    rest = np.where(t >= origin + 3, 20.0, 0.0)               # a restart jump after the origin
    s = np.stack([trend + step + rest, trend + step + rest, trend])
    sel = np.array([True, False, True])
    st = prep.artefact_steps(s, origin, sel, startup=6, restart=12)
    out = prep.remove_artefacts(s, st)
    np.testing.assert_allclose(out[0], trend, atol=1e-9)      # the steps are gone
    np.testing.assert_allclose(out[1], s[1])                  # an unselected cell is untouched
    np.testing.assert_allclose(out[2], trend, atol=1e-9)      # a clean selected cell stays clean
    assert st["startup"][0] == pytest.approx(50.0) and st["restart"][0] == pytest.approx(20.0)


def test_calibration_rescales_the_basis_to_the_solved_runs():
    A, Y, K = 4, 5, len(prep.CLASSES)
    rows = np.zeros((2 * K, A, Y))
    rows[0] = -np.linspace(0, 2, Y)[None].repeat(A, 0)       # irrigation, 2026
    rows[1] = -np.linspace(0, 1, Y)[None].repeat(A, 0)       # aquaculture, 2026
    heads = -rows
    solved = [{"name": "cut30", "factors": [0.7, 1, 1, 1, 1, 1], "start": 2026,
               "_dsub": 0.9 * 0.3 * rows[0], "_dh2": 0.8 * 0.3 * heads[0]}]
    cal = prep.calibrate_basis(rows, heads, solved, 0)
    assert cal["subs"][0] == pytest.approx(0.9) and cal["head"][0] == pytest.approx(0.8)
    assert cal["source"][0] == "solved:cut30" and cal["source"][1] == "assumed"
    assert cal["subs"][1] == pytest.approx(0.9)                # the mean of the known factors
    scaled = prep.apply_calibration(rows, cal["subs"])
    np.testing.assert_allclose(prep.superpose(scaled, [0.7, 1, 1, 1, 1, 1], 0),
                               solved[0]["_dsub"])
    assert prep.parse_check("check_irr50_2026") == ([0.5, 1, 1, 1, 1, 1], 2026)
    assert prep.parse_check("check_nope_2026") is None


class _FakeInputs:
    """The shape of ``twin.inputs`` that the builder reads: three wells, two in layer 2."""

    def __init__(self, A):
        self.dates = pd.date_range("2012-01-01", periods=132, freq="MS")
        self.sids = ["w0", "w1", "w2"]
        self.obs_h = np.vstack([np.full(132, 5.0), np.full(132, 7.0), np.full(132, 1.0)])
        self.obs_layer = np.array([1, 1, 0])
        self.obs_idx = np.array([0, 1, 2])
        self.well_xy = np.array([[X0 + 1500.0, Y0 + 1500.0]] * 3)
        self.E_by_class = {c: np.full(132, 1e6) for c in prep.CLASSES}


def test_public_page_ships_observations_only_as_aggregates(tmp_path, quiet, monkeypatch):
    monkeypatch.setattr(va, "_load_inputs", lambda: _FakeInputs(0))
    _, out, _ = _build(tmp_path, wells="auto")
    d = _payload(out)
    assert d["wells"] is None and d["leveling"] is None and d["modelcard"]["public"] is True
    wf = d["wellsFan"]
    assert wf["n_wells"] == 2 and wf["obs"][0] == pytest.approx(6.0) and wf["n"][0] == 2
    assert all(c["gwh"] == pytest.approx(12.0) for c in d["classes"])
    _, out2, _ = _build(tmp_path, wells="auto", public=False)
    d2 = _payload(out2)
    assert d2["wells"] is not None and len(d2["wells"]["x"]) == 3
    assert d2["modelcard"]["public"] is False


def _write_member_sidecar(tmp_path, n_sets=3, n_ic=2, split_cell=0):
    """``fwd.members.npz`` as ``twin.forward --save-members yearly`` writes it: every
    member follows the ensemble mean, except that one set shows a harm at ``split_cell``
    under cut30."""
    pytest.importorskip("torch")  # twin.forward imports torch
    from hydrophysics.twin.forward import write_members_sidecar

    fw = np.load(tmp_path / "fwd.npz")
    subs = fw["subs_mean"].astype("float64")                       # (S, A, T) m
    heads = fw["heads_mean"][:, 1].astype("float64")
    dates = pd.to_datetime([str(d) for d in fw["dates"]])
    ye = np.array([i for i, d in enumerate(dates) if d.month == 12])
    M = n_sets * n_ic
    s_m = np.repeat(subs[:, None, :, :][..., ye], M, axis=1).copy()
    d = subs[1] - subs[0]
    s_m[1, -n_ic:, split_cell] = (subs[0] - d)[split_cell][ye]      # the last set disagrees
    my = {"subs": s_m, "headL2": np.repeat(heads[:, None][..., ye], M, axis=1),
          "years": [int(dates[i].year) for i in ye], "ye_idx": ye,
          "member": [f"m{k // n_ic}" for k in range(M)], "ic": [k % n_ic for k in range(M)],
          "rheology": ["column"] * M, "head_member": [f"m{k // n_ic}" for k in range(M)],
          "head_ic": [k % n_ic for k in range(M)]}
    write_members_sidecar(str(tmp_path / "fwd.members.npz"), my,
                          [str(s) for s in fw["scenario_names"]], int(fw["origin"]), dates)


def test_member_sidecar_gives_real_cell_agreement_and_yearly_bands(tmp_path, quiet):
    _write_inputs(tmp_path)
    _write_member_sidecar(tmp_path)
    pl, out, A = _build(tmp_path)
    p = _payload(out)
    assert p["memberFields"]["n_sets"] == 3 and p["memberFields"]["n"] == 6
    ag = prep.unpack(p["arrays"]["solvedAgree0"])                  # cut30, (A, Y) percent
    Y = len(p["meta"]["years"])
    assert ag.shape == (A, Y)
    assert ag[0, -1] < prep.AGREE_MIN * 100 and (ag[1:, -1] == 100).all()
    band = pl["solved"][0]["bandYr"]
    assert len(band) == 5 and len(band[0]) == Y
    assert band[0][-1] <= band[2][-1] <= band[4][-1] < 0             # a benefit, ordered
    assert "townAgree" in pl["solved"][0]


def test_without_the_sidecar_the_page_keeps_its_fallback(tmp_path, quiet):
    pl, out, _ = _build(tmp_path)
    p = _payload(out)
    assert p["memberFields"] is None
    assert not any(k.startswith("solvedAgree") for k in p["arrays"])


def test_artefact_removal_is_skipped_when_fixed_upstream(tmp_path, quiet):
    fw, _, _ = _write_inputs(tmp_path)
    z = dict(np.load(fw))
    z["subs_mean"] = z["subs_mean"].copy()
    z["subs_mean"][:, :, 1:] += 0.5                                # a 50 cm start-up step
    gate = json.loads(str(z["gate"]))
    gate["zone_boundaries"] = "183,181"                            # proximal cells on this grid
    z["gate"] = np.array(json.dumps(gate))
    np.savez(fw, **z)
    f0 = va._forward_fields(np.load(fw), va._geometry(np.load(fw)), lambda s: None)
    z["forward_options"] = np.array(json.dumps({"column_heads": "free",
                                                "hpc0_released": [{"columns": [0]}]}))
    np.savez(fw, **z)
    g = va._geometry(np.load(fw))
    f1 = va._forward_fields(np.load(fw), g, lambda s: None)
    prox = g.zone == 0
    assert prox.any()
    raw = np.load(fw)["subs_mean"][0].astype("float64") * 100.0
    assert np.allclose(f1.subs[0], raw)                            # untouched
    assert not np.allclose(f0.subs[0][prox], raw[prox])             # the fallback removes it


# per-member agreement, bands and the fixed-upstream run (round 3) -------------------------
def test_cell_agreement_counts_sets_against_the_mean_sign():
    # 3 sets x 2 initial fields, 2 cells, 1 year-end. Cell 0: two sets benefit (-), one
    # harms (+) -> 2/3 agree with the (negative) mean. Cell 1: no change -> 0.
    d = np.zeros((6, 2, 1))
    d[:, 0, 0] = [-1.0, -1.2, -0.5, -0.7, 0.4, 0.2]
    sets = ["a", "a", "b", "b", "c", "c"]
    ag = prep.cell_agreement(d, sets)
    assert ag.shape == (2, 1)
    assert ag[0, 0] == pytest.approx(2 / 3) and ag[1, 0] == 0.0
    # the initial fields are one opinion: a set whose two runs split counts by their mean
    d2 = d.copy()
    d2[5, 0, 0] = -1.0                                   # set c: mean (0.4 - 1.0) / 2 < 0
    assert prep.cell_agreement(d2, sets)[0, 0] == pytest.approx(1.0)
    # the mean sign is the ensemble's: mostly harm -> agreement with harm
    assert prep.cell_agreement(-d, sets)[0, 0] == pytest.approx(2 / 3)


def test_township_agreement_and_member_band_by_hand():
    d = np.zeros((4, 3, 2))                               # 4 runs, 3 cells, 2 year-ends
    d[:, :, 1] = [[-1, -1, 1], [-2, -1, 2], [1, 1, -1], [-1, 0, 0]]
    sets = ["a", "b", "c", "d"]
    town = np.array([1, 1, 2])
    ta = prep.township_agreement(d, sets, town, 3)
    assert ta[0] == {"agree": 0, "n": 0}                  # no cell in township 0
    assert ta[1] == {"agree": 3, "n": 4}                  # sets a, b, d benefit
    assert ta[2] == {"agree": 1, "n": 4}                  # only set c
    band = prep.fan_member_band(d)
    fan = d.mean(axis=1)[:, 1]
    np.testing.assert_allclose(band[:, 1], np.percentile(fan, (10, 25, 50, 75, 90)))
    np.testing.assert_allclose(band[:, 0], 0.0)


def test_area_and_rail_ranges_by_set():
    subs = np.zeros((4, 3, 2))                            # (runs, cells, year-ends) cm
    subs[:, :, 1] = [[2.0, 0.5, 3.0], [2.0, 0.5, 1.0], [0.0, 0.0, 0.0], [5.0, 5.0, 5.0]]
    sets = ["a", "a", "b", "c"]
    # set a: mean rates 2, .5, 2 -> 2 cells above 1; set b: 0; set c: 3
    np.testing.assert_array_equal(prep.area_by_set(subs, sets, 1.0), [2, 0, 3])
    idx = np.array([[0, -1, -1, -1], [1, -1, -1, -1], [2, -1, -1, -1]])
    w = np.array([[1.0, 0, 0, 0]] * 3)
    fwd = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 0.0, 0.0], [0.0, 3.0, 3.0]])
    ad = prep.hsr_max_by_set(fwd, sets, idx, w, step_km=0.5)
    assert ad[1] == 0.0 and ad[0] > 0 and ad[2] > 0
    # leaving out every point but the flat end drops the gradient
    mixed = np.array([True, True, False])
    assert prep.hsr_max_by_set(fwd, sets, idx, w, 0.5, mixed)[2] < ad[2]


def _fix_upstream(fw):
    z = dict(np.load(fw))
    z["forward_options"] = np.array(json.dumps({"column_heads": "free",
                                                "column_hpc0_fast_days": 365.0,
                                                "restart_taper_km": 5.0}))
    np.savez(fw, **z)


def test_fixed_upstream_run_counts_from_the_fit_and_ships_member_results(tmp_path, quiet):
    fw, bs, _ = _write_inputs(tmp_path)
    _fix_upstream(fw)
    _write_member_sidecar(tmp_path)
    out = tmp_path / "page.html"
    pl = va.build(str(fw), str(bs), str(out), townships_csv=None, leveling="none",
                  wells="none", temporal_npz=None, column_csv=None, hsr_csv=None,
                  rivers_csv=None, theta_json=None, alt_npz=None, rheo_npz=None,
                  log=lambda s: None)
    p = _payload(out)
    yrs = p["meta"]["years"]
    assert yrs[p["meta"]["yRef"]] == 2022 == yrs[p["meta"]["yObs"]]
    art = p["modelcard"]["artefacts"]
    assert art["fixed_upstream"] and art["removed"] is False
    assert art["hpc0_fast_days"] == 365.0 and art["restart_taper_km"] == 5.0
    mf = p["memberFields"]
    Y = len(yrs)
    assert len(mf["baseBandYr"]) == 5 and len(mf["baseBandYr"][0]) == Y
    assert set(mf["fan"]["subs"]) == {"baseline", "cut30", "retire_aqua"}
    assert np.array(mf["fan"]["subs"]["baseline"]).shape == (6, Y)
    assert len(mf["labels"]) == 6 and set(mf["area"]) == {"1", "2", "3"}
    x = pl["solved"][0]
    assert len(x["bandYrHead"]) == 5 and set(x["area"]) == {"1", "2", "3"}
    # the runs follow the ensemble mean: their fan average is the page's baseline
    base = prep.unpack(p["arrays"]["subsBase"]).mean(axis=0)
    np.testing.assert_allclose(np.array(mf["fan"]["subs"]["baseline"])[0], base, atol=0.05)
    assert sum(t["n"] > 0 for t in x["townAgree"]) >= 1


def test_rheology_note_pairs_the_two_columns(tmp_path, quiet):
    recs = []
    for sc, dpol in (("baseline", 0.0), ("cut30", -0.9)):
        for j in range(3):
            for rheo, add in (("col", 0.0), ("tau30y", 3.0 + 0.1 * j)):
                recs.append({"scenario": sc, "member": f"m{j}", "ic": 0, "rheology": rheo,
                             "subs_forward_cm": 4.0 + j + dpol + add + (0.002 if sc != "baseline"
                                                                        and rheo != "col" else 0)})
    pd.DataFrame(recs).to_csv(tmp_path / "rh.members.csv", index=False)
    np.savez(tmp_path / "rh.npz", rheology_labels=np.array(["col", "tau30y"]))
    col = tmp_path / "tau" / "stage4_column.csv"
    col.parent.mkdir()
    pd.DataFrame({"config": ["zonal"], "r2_outoffold": [0.59],
                  "rings_independent_r2": [0.28]}).to_csv(col, index=False)
    (col.parent / "vep_zonal_leveling.json").write_text(json.dumps({"tau_max_years": 30.0}))
    r = va._rheology_block(str(tmp_path / "rh.npz"), str(col), {"r2_oof": 0.589},
                           lambda s: None)
    assert r["dBase"] == pytest.approx(3.1) and r["n"] == 3
    assert r["policyShift"] == pytest.approx(0.002, abs=1e-6)
    assert r["tauYears"] == 30.0 and r["levOofAlt"] == pytest.approx(0.59)
    assert r["artefacts"] is True                        # no forward_options: not fixed


def test_temporal_block_reads_the_calibration_layout_and_its_scorecard(tmp_path):
    rng = np.random.default_rng(2)
    obs = rng.normal(size=(4, 131))
    d = tmp_path / "temporal_ref"
    d.mkdir()
    np.savez(d / "stage3_temporal_pred.npz", obs=obs, clim=obs + 0.5, pred=obs + 3.0,
             T_fit=np.int64(95))
    (d / "scorecard.json").write_text(json.dumps({"temporal": {
        "rmse_model_m": 3.0, "r2_shape_model": -1.5, "r2_well_median_model": -11.0,
        "level_err_model_m": 3.7, "verdict": "FAIL"}}))
    tp = va._temporal_block(str(d / "stage3_temporal_pred.npz"))
    assert tp["rmse_model"] == pytest.approx(3.0) and tp["passed"] is False
    assert tp["r2_shape"] == pytest.approx(-1.5) and tp["verdict"] == "FAIL"


# the per-well-datum deliverable (2026-09-27): rheology axis, basis check, fair verdict ------
def test_rheology_axis_shows_the_first_column_and_its_members(tmp_path, quiet):
    """A run with two columns: the page's fields and paired members are the first column's;
    ``subs_mean`` (which pools both) is not used."""
    fw, bs, _ = _write_inputs(tmp_path)
    z = dict(np.load(fw))
    first = z["subs_mean"].copy()
    z["subs_mean_by_rheology"] = np.stack([first, first + 0.05]).astype("float32")
    z["subs_mean"] = (first + 0.025).astype("float32")
    z["rheology_labels"] = np.array(["col", "tau30y"])
    np.savez(fw, **z)
    m = pd.read_csv(tmp_path / "fwd.members.csv")
    m2 = m.copy()
    m2["subs_forward_cm"] += 3.0
    m2.loc[m2.scenario != "baseline", "subs_forward_cm"] -= 5.0     # a very different response
    pd.concat([m.assign(rheology="col"), m2.assign(rheology="tau30y")]).to_csv(
        tmp_path / "fwd.members.csv", index=False)
    out = tmp_path / "page.html"
    pl = va.build(str(fw), str(bs), str(out), townships_csv=None, leveling="none",
                  wells="none", temporal_npz=None, column_csv=None, hsr_csv=None,
                  rivers_csv=None, theta_json=None, alt_npz=None, rheo_npz="auto",
                  log=lambda s: None)
    base = prep.unpack(_payload(out)["arrays"]["subsBase"])
    ye, _ = prep.year_ends([str(d) for d in z["dates"]])
    np.testing.assert_allclose(base, first[0][:, ye] * 100.0, atol=0.02)
    assert pl["modelcard"]["rheology_ref"] == "col"
    mine = pl["solved"][0]["members"]
    assert mine["n"] == 4 and mine["subs_mean"] == pytest.approx(-0.65)   # col runs only
    r = pl["rheology"]
    assert r["ref"] == "col" and r["baseAlt"] - r["baseRef"] == pytest.approx(3.0)


def test_basis_built_for_another_model_is_switched_off(tmp_path, quiet):
    fw, bs, _ = _write_inputs(tmp_path)
    b = dict(np.load(bs))
    names = [str(n) for n in b["scenario_names"]]
    s = b["subs_mean"].copy()
    i = names.index("aqua0_2026")
    s[i] = s[0] + 2.0 * (s[i] - s[0])                              # twice the solved response
    b["subs_mean"] = s
    np.savez(bs, **b)
    out = tmp_path / "page.html"
    pl = va.build(str(fw), str(bs), str(out), townships_csv=None, leveling="none",
                  wells="none", temporal_npz=None, column_csv=None, hsr_csv=None,
                  rivers_csv=None, theta_json=None, alt_npz=None, rheo_npz=None,
                  log=lambda s: None)
    off = pl["basisOff"]
    assert off["worst"] == "retire_aqua" and off["rel"] > va.BASIS_MAX_REL
    assert pl["basis"] is None and pl["calib"] is None
    assert "dSubs" not in pl["arrays"] and "retire_aqua" in pl["basisError"]
    # without a basis no solved policy saturates the difference scale
    assert pl["scales"]["dsubs"]["limit"] >= pl["scales"]["dsubs"]["p98"]
    html = out.read_text(encoding="utf-8")
    assert "boffWords" in html and "NOPOL" in html


def test_basis_that_matches_the_solved_runs_stays_on(tmp_path, quiet):
    pl, _, _ = _build(tmp_path)
    assert pl["basisOff"] is None and pl["basis"] is not None


def test_temporal_block_takes_the_fair_verdict(tmp_path):
    obs = np.random.default_rng(3).normal(size=(4, 131))
    d = tmp_path / "temporal_datum"
    (d / "coupled_leveling").mkdir(parents=True)
    np.savez(d / "stage3_temporal_pred.npz", obs=obs, clim=obs + 0.5, pred=obs + 3.0,
             T_fit=np.int64(95))
    t = {"rmse_model_m": 7.7, "rmse_clim_m": 2.0, "verdict": "FAIL", "verdict_fair": "PASS",
         "fair_rmse_ratio_fair": 1.106, "fair_fair_k": 1.25, "fair_rmse_datum_model_m": 1.81,
         "fair_best_baseline": "clim_trend", "fair_rmse_best_baseline_m": 1.64,
         "fair_r2_shape_datum_model": 0.449, "fair_r2_shape_clim": 0.356,
         "fair_n_wells_scored": 147, "fair_datum_share_of_mse": 0.95}
    (d / "coupled_leveling" / "scorecard.json").write_text(json.dumps({"temporal": t}))
    (d / "stage3_theta.json").write_text(json.dumps({"theta": {"spread_km": 10.0}}))
    tp = va._temporal_block(str(d / "stage3_temporal_pred.npz"), rescore_csv=None)
    assert tp["passed"] is True and tp["verdict"] == "FAIL"        # fair, not the legacy one
    assert tp["rmse_model"] == pytest.approx(7.7) and tp["n_wells"] == 147
    f = tp["fair"]
    assert f["ratio"] == pytest.approx(1.106) and f["best"] == "clim_trend"
    assert tp["spread_km"] == pytest.approx(10.0)
    # an old screen without the fair fields takes them from the rescore table
    csv = tmp_path / "rescore.csv"
    pd.DataFrame([{"screen": "old", "verdict_fair": "FAIL", "rmse_ratio_fair": 2.12}]).to_csv(
        csv, index=False)
    assert va._rescored(str(csv))["old"]["ratio"] == pytest.approx(2.12)


def test_flow_scores_read_the_anomaly_kfold_and_the_modelcard_hides_well_offsets(
        tmp_path, quiet):
    run = tmp_path / "flow"
    run.mkdir()
    pd.DataFrame([{"r2_kfold": 0.643, "r2_idw": 0.702, "verdict_kfold": "FAIL",
                   "r2_anom_kfold": 0.342, "r2_anom_idw": 0.605, "verdict_anom_kfold": "FAIL",
                   "n_folds": 5}]).to_csv(run / "stage3_flow.csv", index=False)
    hs = va._flow_scores(str(run), {"gate": {"r2_kfold": 0.9, "verdict": "PASS"}})
    assert hs["r2_kfold"] == pytest.approx(0.643) and hs["verdict"] == "FAIL"
    assert hs["r2_anom"] == pytest.approx(0.342) and hs["verdict_anom"] == "FAIL"
    # an older run keeps its anomaly score in stage3_kfold_anom.csv
    old = tmp_path / "old"
    old.mkdir()
    pd.DataFrame([{"r2_kfold": 0.804, "verdict_kfold": "PASS"}]).to_csv(
        old / "stage3_flow.csv", index=False)
    pd.DataFrame([{"r2_anom_kfold": -1.43, "verdict_anom_kfold": "FAIL"}]).to_csv(
        old / "stage3_kfold_anom.csv", index=False)
    ho = va._flow_scores(str(old))
    assert ho["verdict"] == "PASS" and ho["r2_anom"] == pytest.approx(-1.43)
    # the forward gate's per-well datum carries station ids: only aggregates reach the page
    fw, bs, _ = _write_inputs(tmp_path)
    z = dict(np.load(fw))
    g = json.loads(str(z["gate"]))
    g.update({"well_datum": {"07010211": -5.76}, "well_datum_mode": "fit",
              "well_datum_stats": {"n": 158, "rms_m": 7.6, "max_abs_m": 38.0}})
    z["gate"] = np.array(json.dumps(g))
    np.savez(fw, **z)
    out = tmp_path / "page.html"
    theta = run / "stage3_theta.json"
    theta.write_text(json.dumps({"theta": {"spread_km": 2.7}}))
    pl = va.build(str(fw), str(bs), str(out), townships_csv=None, leveling="none",
                  wells="none", temporal_npz=None, column_csv=None, hsr_csv=None,
                  rivers_csv=None, theta_json=str(theta), alt_npz=None, rheo_npz=None,
                  log=lambda s: None)
    mc = pl["modelcard"]
    assert mc["r2_anom"] == pytest.approx(0.342) and mc["well_datum"]["rms_m"] == 7.6
    assert "07010211" not in out.read_text(encoding="utf-8")


def test_member_fields_keep_one_rheology(tmp_path):
    pytest.importorskip("torch")
    from hydrophysics.twin.forward import write_members_sidecar

    S, A, Y = 2, 3, 4
    dates = pd.date_range("2012-01-01", periods=12 * Y, freq="MS")
    ye = np.arange(11, 12 * Y, 12)
    my = {"subs": None, "headL2": np.zeros((S, 2, A, Y)), "years": [2012, 2013, 2014, 2015],
          "ye_idx": ye, "member": ["m0", "m0", "m1", "m1"], "ic": [0, 0, 0, 0],
          "rheology": ["col", "tau", "col", "tau"], "head_member": ["m0", "m1"],
          "head_ic": [0, 0]}
    my["subs"] = np.zeros((S, 4, A, Y))
    my["subs"][:, [1, 3]] = 0.01                                # the second column's runs
    p = tmp_path / "x.members.npz"
    write_members_sidecar(str(p), my, ["baseline", "cut"], 20, dates)
    allm = prep.load_member_fields(str(p))
    one = prep.load_member_fields(str(p), "col")
    assert allm["subs"].shape[1] == 4 and one["subs"].shape[1] == 2
    assert one["sets"] == ["m0", "m1"] and np.allclose(one["subs"], 0.0)
    with pytest.raises(ValueError):
        prep.load_member_fields(str(p), "nope")


def test_zone_lines_report_the_rate_step():
    x = np.array([181.5, 181.5, 182.5, 182.5])
    b = va._boundaries(np.array([0.0, -2.0, 15.0, 14.0]), np.array([0.0, 0.0, 1.0, 1.1]),
                       x, [182.0])
    assert b[0]["east"] == pytest.approx(14.5) and b[0]["rate_east"] == pytest.approx(1.05)
    assert b[0]["rate_west"] == 0.0


def test_effect_timing_tells_a_one_time_rebound_from_a_slowing():
    """Avoided subsidence at 1, 4 and 10 policy years (clipped to the horizon) and the late
    rate, by hand: a step response is a rebound, a growing one slows the sinking."""
    years = list(range(2020, 2033))                      # year-ends 2020 .. 2032
    y = np.array(years, dtype="float64")
    base = 0.3 * (y - 2019)                              # 0.3 cm/yr throughout
    step = base - np.where(y >= 2026, 0.9, 0.0)          # 0.9 cm at once, then parallel
    t = prep.effect_timing(base, step, years, 2026)
    assert [(a["n"], a["year"]) for a in t["at"]] == [(1, 2026), (4, 2029), (7, 2032)]
    assert [a["avoid"] for a in t["at"]] == pytest.approx([0.9, 0.9, 0.9])
    assert t["rate_base"] == pytest.approx(0.3) and t["rate_pol"] == pytest.approx(0.3)
    assert t["rate_years_span"] == [2028, 2032]
    assert t["first_share"] == pytest.approx(1.0)
    assert not t["slows"] and not t["grows"]
    # the same step from 2030: the same horizon value, reached later
    late = base - np.where(y >= 2030, 0.9, 0.0)
    t30 = prep.effect_timing(base, late, years, 2030)
    assert [a["year"] for a in t30["at"]] == [2030, 2032]            # 4 and 10 clip
    assert t30["at"][-1]["avoid"] == pytest.approx(t["at"][-1]["avoid"])
    assert t30["rate_years_span"] == [2031, 2032] and not t30["slows"]  # after its jump
    # a policy that slows the sinking: 0.4 at once, then 0.1 cm/yr less
    slow = base - np.where(y >= 2026, 0.4 + 0.1 * (y - 2026), 0.0)
    s = prep.effect_timing(base, slow, years, 2026)
    assert [a["avoid"] for a in s["at"]] == pytest.approx([0.4, 0.7, 1.0])
    assert s["rate_pol"] == pytest.approx(0.2) and s["slows"] and s["grows"]
    assert s["first_share"] == pytest.approx(0.4)
    # counted from the year-end before the start, like the page's forward difference
    off = step - 0.05
    assert prep.effect_timing(base, off, years, 2026)["at"][0]["avoid"] == pytest.approx(0.9)
    assert prep.timing_kind([t, t30]) == "rebound"
    assert prep.timing_kind([t, s]) == "slows"
    assert prep.timing_kind([]) is None


def test_payload_carries_the_timing_of_this_and_the_previous_model(tmp_path, quiet):
    """Each solved run carries its timing from the page's own fields; the other model's
    comes from its first column. The synthetic responses grow (a ramp), the other model's
    is made a step, so the two kinds differ."""
    fw, _, _ = _write_inputs(tmp_path)
    z = dict(np.load(fw, allow_pickle=False))
    s = z["subs_mean"].astype("float64")
    t = np.arange(T)
    step = np.where(t >= 168, 1.0, 0.0)[None, :]                   # from Jan 2026
    alt_s = np.stack([s[0], s[0] - 0.009 * step * np.ones((s.shape[1], 1)),
                      s[0] - 0.013 * step * np.ones((s.shape[1], 1))])
    z["subs_mean"] = alt_s.astype("float32")
    alt = tmp_path / "alt.npz"
    np.savez(alt, **z)
    pd.read_csv(tmp_path / "fwd.members.csv").to_csv(tmp_path / "alt.members.csv", index=False)
    pl, out, _ = _build(tmp_path, alt_npz=str(alt))
    d = _payload(out)
    ye, years = prep.year_ends(pd.date_range("2012-01-01", periods=T, freq="MS").astype(str))
    fan = s[:, :, ye].mean(axis=1) * 100.0
    for i, name in enumerate(["cut30", "retire_aqua"]):
        tim = next(x for x in d["solved"] if x["name"] == name)["timing"]
        want = prep.effect_timing(fan[0], fan[i + 1], years, 2026)
        assert [a["year"] for a in tim["at"]] == [2026, 2029, 2032]
        assert [a["avoid"] for a in tim["at"]] == pytest.approx(
            [a["avoid"] for a in want["at"]], abs=1e-4)
        assert tim["rate_base"] == pytest.approx(want["rate_base"], abs=1e-4)
        assert tim["rate_pol"] < tim["rate_base"] and tim["grows"]
    assert d["timingKind"] == "slows"
    a = d["alt"]
    assert a["timingKind"] == "rebound" and set(a["timing"]) == {"cut30", "retire_aqua"}
    assert [x["avoid"] for x in a["timing"]["cut30"]["at"]] == pytest.approx([0.9] * 3, abs=1e-4)
    assert a["timing"]["retire_aqua"]["rate_pol"] == pytest.approx(
        a["timing"]["retire_aqua"]["rate_base"], abs=1e-6)
    html = out.read_text(encoding="utf-8")
    assert "timingCaveat()" in html and 'id="startNote"' in html and "reboundLine(kp)" in html


# evidence reports on the model card ------------------------------------------------------------
def _fake_forward(tmp_path):
    forward = tmp_path / "forward.npz"
    forward.write_bytes(b"forward-run")
    return str(forward)


def test_coverage_block_keeps_only_what_the_page_quotes(tmp_path):
    forward = _fake_forward(tmp_path)
    report = {"model_sha256": va.sha256(forward), "stations": 158, "scored_months": 6186,
              "forecast_origin": "2022-12-01", "end_exclusive": "2026-09-01",
              "prospective_validation": False,
              "metrics": {"model_ensemble": {"picp_90": 0.747, "inflation_90": 1.86, "other": 1},
                          "model_combined": {"picp_90": 0.926}}}
    path = tmp_path / "coverage.json"
    path.write_text(json.dumps(report))
    block = va._coverage_block(str(path), forward)
    assert block == {"picp_90": 0.747, "inflation_90": 1.86, "combined_picp_90": 0.926,
                     "stations": 158, "scored_months": 6186, "forecast_origin": "2022-12-01",
                     "end_exclusive": "2026-09-01", "prospective_validation": False}
    assert va._coverage_block(None, forward) is None
    report["model_sha256"] = "0" * 64
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="coverage report"):
        va._coverage_block(str(path), forward)


def test_prospective_block_reads_the_window_from_the_protocol(tmp_path):
    forward = _fake_forward(tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps(
        {"registered_on": "2026-10-11", "stations": 158, "model_sha256": va.sha256(forward)}))
    block = va._prospective_block(str(tmp_path / "manifest.json"), forward)
    assert block == {"registered_on": "2026-10-11", "stations": 158, "test_start": None,
                     "test_end": None, "result": "pending"}
    (tmp_path / "protocol.json").write_text(json.dumps(
        {"test_start": "2026-11-01", "test_end": "2027-10-31"}))
    block = va._prospective_block(str(tmp_path / "manifest.json"), forward)
    assert (block["test_start"], block["test_end"], block["result"]) == (
        "2026-11-01", "2027-10-31", "pending")


def test_ring_holdout_block_is_absent_until_the_report_exists(tmp_path):
    path = tmp_path / "ring.json"
    assert va._ring_holdout_block(str(path)) is None and va._ring_holdout_block(None) is None
    report = {"rings": {"n_sites": 12, "cum": {"model": {"r2": 0.4}, "base_continue": {"r2": 0.3}},
                        "rate": {}}, "gnss": {"n_sites": 3}, "window": ["2023-01", "2025-12"],
              "independence": ["rings share the column"]}
    path.write_text(json.dumps(report))
    block = va._ring_holdout_block(str(path))
    assert block["rings"]["n_sites"] == 12 and block["window"] == ["2023-01", "2025-12"]
    assert block["independence"] == ["rings share the column"]
    assert block["rings"]["cum"]["base_continue"]["r2"] == 0.3
    assert block["max_ref_age_days"] is None                   # a report from before the fix
    report["max_ref_age_days"] = 62
    path.write_text(json.dumps(report))
    assert va._ring_holdout_block(str(path))["max_ref_age_days"] == 62


def test_policy_response_block_keeps_the_verdicts_the_page_quotes(tmp_path):
    assert va._policy_response_block(None) is None
    row = {"rmse_model_m": 1.4, "rmse_cl_m": 2.2, "rmse_zr_m": 2.3, "n_wells": 75,
           "verdict_cl": "model_better", "verdict_zr": "model_better",
           "slope_obs_on_model": 1.2, "slope_ci": [1.06, 1.36]}
    report = {"window": "2016-01..2018-12 (masked refits)",
              "heads": {"yunlin": row, "changhua": row},
              "subsidence": {"yunlin": {"rms_model_cm": 1.9, "rms_cl_cm": 1.0,
                                        "verdict_cl": "baseline_better",
                                        "slope_obs_on_model": 0.22}, "changhua": {}},
              "changhua": {"status": "not_supported", "dose_obs_cm_per_gwh": -0.5},
              "registrations": {"N1.b": "sha", "N1.a": "sha"}, "inputs": {"x": {}}}
    path = tmp_path / "page_labels.json"
    path.write_text(json.dumps(report))
    block = va._policy_response_block(str(path))
    assert block["heads"]["yunlin"]["rmse_cl_m"] == 2.2 and "n_wells" not in block["heads"]["yunlin"]
    assert block["subsidence"]["yunlin"]["slope_obs_on_model"] == 0.22
    assert block["changhua"]["status"] == "not_supported"
    assert block["registrations"] == ["N1.a", "N1.b"] and "inputs" not in block


def test_three_is_shipped_unevaluated_and_the_page_has_one_dictionary():
    page = va._render({"meta": {}}, "inline", log=lambda s: None)
    marker = '<script type="text/plain" id="three-src">'
    assert marker in page
    head, _, rest = page.partition(marker)
    assert "WebGLRenderer" not in head and "WebGLRenderer" in rest
    with open(va.TEMPLATE, encoding="utf-8") as fh:
        template = fh.read()
    # one i18n mechanism: no inline language ternaries, and both languages carry the same keys
    assert not re.search(r"\bzh\s*\?", template)
    en, zh = template.split("const T = {")[1].split("\nzh: {")
    key = re.compile(r'(?<![\w"])([A-Za-z0-9_]+):\s*"')
    assert set(key.findall(en)) == set(key.findall(zh.split("\n}};")[0]))


# ---------------------------------------------------------------------------------------
# policy-response labels (review 2026-10-24): page_labels v2 names the model of every N1
# row; the page says the Changhua head result is the census model's, flags the design-time
# window and never counts another model's row; the deep-pile line is report-only (U7); the
# policy gate is a model-internal sanity check; the levers act on Yunlin pumping only when
# the model carries none in Changhua; flipped verdicts change the text. The text checks run
# the page in headless Chromium (Playwright) and skip without it.
# ---------------------------------------------------------------------------------------
from types import SimpleNamespace  # noqa: E402

from dataprep_staging.n1_policy import page_labels_v2 as PL2  # noqa: E402

R0 = "results/twin_runs/stage3_datum_gate_m1618/stage3_theta.json"
R1C = "results/twin_runs/stage3_datum_gate_ccz_m1618/stage3_theta.json"


def _head(model_better=True):
    v = "model_better" if model_better else "baseline_better"
    return {"rmse_model_m": 1.38, "rmse_cl_m": 2.21, "rmse_zr_m": 2.29, "n_wells": 75,
            "verdict_cl": v, "verdict_zr": v, "slope_obs_on_model": 1.21,
            "slope_ci": [1.06, 1.36]}


def _lev(model_better=False):
    v = "model_better" if model_better else "baseline_better"
    return {"rms_model_cm": 1.88, "rms_cl_cm": 1.0, "rms_zr_cm": 1.08, "n_benchmarks": 355,
            "verdict_cl": v, "verdict_zr": v, "slope_obs_on_model": 0.22,
            "slope_ci": [0.17, 0.27]}


def _v1(ch_heads_ok=True, subs_ok=False) -> dict:
    return {"product": "n1_policy/page_labels", "version": "v1",
            "window": "2016-01..2018-12 (masked refits; design-time window)",
            "heads": {"yunlin": _head(), "changhua": _head(ch_heads_ok)},
            "subsidence": {"yunlin": _lev(subs_ok), "changhua": _lev(subs_ok)},
            "changhua": {"status": "not_supported", "dose_obs_cm_per_gwh": -0.5,
                         "dose_model_cm_per_gwh": 0.8, "dose_verdict": "inconsistent",
                         "fixes": {"F1": "failed_design_gate",
                                   "F2": "heads vs R1c indistinguishable"}},
            "registrations": {"N1.policy_response.v1": "a" * 64}, "inputs": {}}


def _cmp(rm, rb, verdict="model_better", ratio=0.96, ci=(-0.1, -0.0002)):
    return {"n_wells": 59, "rmse_model": rm, "rmse_base": rb, "ratio": ratio, "ci": list(ci),
            "verdict": verdict}


def _step_c(models=(R0, R1C)):
    row = {"M_vs_CL": _cmp(1.13, 1.19), "M_vs_ZR": _cmp(1.13, 1.20),
           "slope_obs_on_M": {"slope": 0.77, "ci": [0.51, 1.05]}}
    tests = {}
    for county, m in zip(("yunlin", "changhua"), models, strict=True):
        for kind in ("heads", "lev"):
            tests[f"N1P.C_{kind}_{county}"] = {"model": [m, m.replace("stage3_theta", "col")]}
    return {"tests": tests,
            "secondary_cross_county": {"changhua_scored_with_yunlin_model": {"heads": row}}}


def _step_b():
    def t(v, o, m):
        return {"verdict": v, "beta_obs_cm_per_GWh": o, "beta_model_cm_per_GWh": m,
                "ratio_ci": [0.2, 2.2], "model": {"model": {"theta": "t", "column": "c"}}}
    return {"tests": {"N1P.B_dose_yunlin": t("inconclusive", 0.66, 0.26),
                      "N1P.B_dose_changhua": t("inconsistent", -0.5, 0.8)}}


# --- page_labels v2 ---------------------------------------------------------------------
def test_v2_names_the_model_of_every_row_and_keeps_every_v1_value():
    v1 = _v1()
    v2 = PL2.build_v2(v1, _step_b(), _step_c())
    assert v2["version"] == "v2" and v2["design_time"] is True
    assert v2["heads"]["yunlin"]["model"]["id"] == "R0-m1618"
    assert v2["heads"]["yunlin"]["model"]["deliverable_recipe"] is True
    assert v2["heads"]["changhua"]["model"]["id"] == "R1c-m1618"
    assert v2["subsidence"]["changhua"]["model"]["deliverable_recipe"] is False
    for kind in ("heads", "subsidence"):
        for county, row in v1[kind].items():
            assert {k: v2[kind][county][k] for k in row} == row
    sec = v2["secondary"]["changhua_heads_deliverable_recipe"]
    assert sec["ratio_cl"] == 0.96 and sec["ci_cl"][1] == -0.0002
    assert v2["step_b"]["yunlin"]["verdict"] == "inconclusive"
    assert v1["version"] == "v1" and "model" not in v1["heads"]["yunlin"]   # not mutated


def test_v2_refuses_an_unknown_model_and_a_status_it_cannot_derive():
    with pytest.raises(ValueError, match="unknown model"):
        PL2.build_v2(_v1(), _step_b(), _step_c(models=("results/x/stage3_theta.json", R1C)))
    v1 = _v1()
    v1["changhua"]["dose_verdict"] = "consistent"       # would derive "supported"
    with pytest.raises(ValueError, match="derived Changhua status"):
        PL2.build_v2(v1, _step_b(), _step_c())
    assert PL2.changhua_status("inconsistent", {"F2": "model_better vs R1c"}) == "supported"


# --- viewer blocks ----------------------------------------------------------------------
def test_v1_labels_give_the_v1_block_and_v2_adds_the_models(tmp_path):
    p1, p2 = tmp_path / "v1.json", tmp_path / "v2.json"
    p1.write_text(json.dumps(_v1()))
    p2.write_text(json.dumps(PL2.build_v2(_v1(), _step_b(), _step_c())))
    b1, b2 = va._policy_response_block(str(p1)), va._policy_response_block(str(p2))
    assert "model" not in b1["heads"]["changhua"] and "rmse_zr_m" not in b1["heads"]["yunlin"]
    assert set(b1) == {"window", "heads", "subsidence", "changhua", "registrations"}
    assert b2["heads"]["changhua"]["model"]["id"] == "R1c-m1618"
    assert b2["design_time"] is True and "secondary" in b2 and "step_b" in b2


def test_deep_piles_used_for_selection_are_report_only(tmp_path):
    rep = {"scores": {"n_sites": 4, "window": ["2024-01", "2025-12"], "cum": {}},
           "independence": ["(i) four piles only",
                            "(ii) rings hosted; the deep piles themselves were never used in "
                            "calibration or selection"]}
    path = tmp_path / "piles.json"
    path.write_text(json.dumps(rep))
    b = va._deep_pile_block(str(path), None)
    assert b["selection"] == {"date": "2026-10-21", "ruling": "U7", "report_only": True}
    assert b["independence"][0] == "(i) four piles only"
    assert "never used" not in b["independence"][1] and "(U7)" in b["independence"][1]
    assert b["independence_superseded"] == [1]
    assert "never used" in json.loads(path.read_text())["independence"][1]   # report untouched
    rep["scores"]["window"] = ["2026-11", "2027-10"]                          # a fresh window
    path.write_text(json.dumps(rep))
    assert "selection" not in va._deep_pile_block(str(path), None)


def test_pumping_block_finds_no_changhua_pumping_in_a_yunlin_only_run():
    E = np.zeros((5, 24))
    E[3:] = 1e6                                    # cells 3-4: Yunlin townships
    inp = SimpleNamespace(dates=pd.date_range("2012-01", periods=24, freq="MS"),
                          E_by_class={"irrigation": E})
    town = np.array([0, 0, 0, 1, 2])
    b = va._pumping_block(inp, {}, town, log=lambda s: None)
    assert b["changhua"] is False and b["pooled_gwh"] == 0.0 and b["in_grid_gwh"] == 24.0
    E[0] = 1e6
    assert va._pumping_block(inp, {}, town, log=lambda s: None)["changhua"] is True
    own = va._pumping_block(inp, {"pump_census": "census.parquet"}, town, log=lambda s: None)
    assert own["changhua"] is None and own["source"] == "own census"
    assert va._pumping_block(None, {}, town) is None


def test_template_has_the_policy_label_keys_in_both_languages():
    with open(va.TEMPLATE, encoding="utf-8") as fh:
        t = fh.read()
    for key in ("evc_pile_ro", "pr_design", "prm_r1c", "pr_not_beats", "evc_pr_subs_ok",
                "levers_sub_np", "pol_caveat_ch_np", "pooled_np", "hl_yun", "tt_n1_heads",
                "evc_pr_heads_size", "pol_caveat_hg", "hl_hot", "rank_gwh_tie", "cv9_t_split",
                "cv9_b_split", "cv9_pol", "cv9_lift"):
        assert t.count(f"  {key}: ") + t.count(f" {key}: ") >= 2, key
    assert "Policy-response check" not in t
    assert "long-run trend" not in t and "長期趨勢" not in t and "overstat" not in t
    # the Yunlin headline figure names its cell set (named townships, not the county mask)
    yun = [ln for ln in t.splitlines() if ln.lstrip().startswith("hl_yun: ")]
    assert len(yun) == 2 and all("{n}" in ln for ln in yun)
    assert "named Yunlin townships" in yun[0] and "已命名鄉鎮" in yun[1]


# --- the rendered text (headless Chromium) ----------------------------------------------
def _browser():
    sync = pytest.importorskip("playwright.sync_api")
    pw = sync.sync_playwright().start()
    try:
        browser = pw.chromium.launch(**va_exe())
    except Exception as e:  # noqa: BLE001
        pw.stop()
        pytest.skip(f"no Chromium for Playwright: {type(e).__name__}")
    return pw, browser


def va_exe() -> dict:
    from hydrophysics.twin.browser_check import _exe

    return _exe()


def _page_text(tmp_path, labels: dict, js: str):
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(labels))
    _, out, _ = _build(tmp_path, policy_response_json=str(path))
    pw, browser = _browser()
    try:
        page = browser.new_page()
        page.goto(out.as_uri())
        page.wait_for_function("document.body.dataset.ready === '1'", timeout=30000)
        return page.evaluate(js)
    finally:
        browser.close()
        pw.stop()


LINES = "() => policyResponseLines(D.modelcard.policy_response).map(x => x[1])"


def test_page_names_the_census_model_and_the_design_time_window(tmp_path, quiet):
    v2 = PL2.build_v2(_v1(), _step_b(), _step_c())
    js = ("() => { D.modelcard.pumping = {changhua: false}; const rows = testRows(); "
          "return {lines: policyResponseLines(D.modelcard.policy_response).map(x => x[1]), "
          "n1: rows.filter(r => r.count !== undefined).map(r => [r.test, r.verdict, r.count]), "
          "counts: testCounts(), lev: tr('levers_sub_np', {rec: 'x'}), "
          "cav: (polCaveat(), document.getElementById('polCaveat').textContent)}; }")
    got = _page_text(tmp_path, v2, js)
    heads_y, heads_c, size, subs, ch = got["lines"]
    assert "Yunlin" in heads_y and "design-time window" in heads_y and "R0 recipe" in heads_y
    # both head-response slopes, the Changhua one named by its model
    assert "1.21 (1.06–1.36" in size and "census model" in size and "No correction" in size
    assert "census model" in heads_c and "not the one shown" in heads_c
    assert "marginal" in heads_c and "0.96" in heads_c          # the declared secondary
    assert "not supported" in subs and "census model" in subs and "the other way" in subs
    assert "no Changhua pumping" in ch
    # the census model's rows are shown but never counted as this model's tests
    assert [c for _, _, c in got["n1"]] == [True, False, True, False]
    assert all("not counted" in t for t, _, c in got["n1"] if not c)
    assert "not available in this model" in got["cav"]
    # neutral subsidence wording, and the Yunlin head-gain caveat
    assert "elastic response that arrives within about a year" in got["cav"]
    assert "long-run sinking is untested" in got["cav"] and "likely understated" in got["cav"]
    assert not any("long-run trend" in x for x in [got["cav"], *got["lines"]])


def test_flipped_verdicts_change_the_policy_text(tmp_path, quiet):
    v2 = PL2.build_v2(_v1(ch_heads_ok=False, subs_ok=True), _step_b(), _step_c())
    lines = _page_text(tmp_path, v2, LINES)
    assert "does not beat both simple baselines" in lines[1]
    assert "does not beat" not in lines[0]
    assert any("1–3-year response beats both simple baselines" in s for s in lines)
    assert not any("is not supported. On the" in s for s in lines)


def test_policy_gate_is_neutral_and_piles_are_report_only(tmp_path, quiet):
    js = ("() => { D.modelcard.policy = 'PASS'; D.modelcard.deep_piles = {n_sites: 4, "
          "window: ['2024-01', '2025-12'], n_verified_values: 1, cum: {model: {rmse: 1.2}, "
          "base_continue: {rmse: 1.1}}, selection: {date: '2026-10-21', ruling: 'U7'}}; "
          "const before = testCounts(); "
          "const row = testRows().find(r => r.test.startsWith('Policy direction')); "
          "return {row: row && [row.verdict, row.mine], pile: pileLine(), before}; }")
    got = _page_text(tmp_path, _v1(), js)
    assert got["row"] == [None, "PASS"]
    assert got["pile"].startswith("Report-only, not independent skill")


# ---------------------------------------------------------------------------------------
# post-hoc, model-only labels (page_posthoc.json, 2026-10-24): off by default, bound to the
# forward run and basis, and every line says "post hoc, model-only, not a test"
# ---------------------------------------------------------------------------------------
from dataprep_staging.n1_policy import page_posthoc as PP  # noqa: E402


def _posthoc(fw_sha: str, bs_sha: str, **kw) -> dict:
    run = {"n": 36, "share_same_leader": 30 / 36, "share_same_top3_set": 1.0,
           "share_leader_in_top3": 1.0, "leader_counts": {"Baozhong": 30, "Beigang": 6}}
    out = {"status": PP.STATUS, "forward_sha256": fw_sha, "basis_sha256": bs_sha,
           "hotspot": {"cut30": {"start": 2026, "horizon_year": 2032,
                                 "headline": ["Baozhong", "Beigang", "Dongshi"],
                                 "runs": run, "r0_runs": {"n": 36, "share_same_leader": 26 / 36},
                                 "draws": {"n": 500, "share_same_leader": 0.568},
                                 "stable": False}},
           "per_gwh": {"starts": {"2026": {"lo": 2.84, "hi": 3.10, "share_order_kept": 0.032,
                                           "cm_share_order_kept": 1.0},
                                  "2030": {"lo": 2.84, "hi": 3.10, "share_order_kept": 0.03,
                                           "cm_share_order_kept": 1.0}},
                       "ranking_survives": False},
           "elastic": {"start": 2026, "horizon_year": 2032,
                       "policies": {"cut30": {"total_cm": 0.942, "elastic_cm": 0.942,
                                              "inelastic_cm": 0.0}},
                       "cells_below_preconsolidation_after_2026_01": 0, "margin_p5_m": 0.243,
                       "lift": {"policy": "cut30", "lifted": "2030-01", "at_year": 2032,
                                "left_cm": 0.0}}}
    out.update(kw)
    return out


def test_posthoc_block_is_off_by_default_and_bound_to_its_run(tmp_path, quiet):
    fw, bs, _ = _write_inputs(tmp_path)
    from hydrophysics.twin.release import sha256

    good = tmp_path / "posthoc.json"
    good.write_text(json.dumps(_posthoc(sha256(fw), sha256(bs))))
    assert va._posthoc_block(None, str(fw), str(bs)) is None
    b = va._posthoc_block(str(good), str(fw), str(bs))
    assert b["status"] == "post hoc, model-only, not a test"
    assert set(b) == {"status", "hotspot", "per_gwh", "elastic"}
    for bad in (_posthoc("0" * 64, sha256(bs)), _posthoc(sha256(fw), "0" * 64),
                _posthoc(sha256(fw), sha256(bs), status="a test")):
        good.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            va._posthoc_block(str(good), str(fw), str(bs))
    # the CLI leaves it off unless --posthoc is given
    out = tmp_path / "cli.html"
    va.main(["--forward", str(fw), "--basis", str(bs), "--out", str(out),
             "--townships", "none", "--basemap", "none", "--leveling", "none", "--wells",
             "none", "--temporal", "none", "--column-csv", "none", "--hsr", "none",
             "--alt-forward", "none", "--rheo-forward", "none"])
    d = _payload(out)
    assert "posthoc" not in d["modelcard"] and "posthoc" not in d["release"]["sources"]


def test_page_posthoc_copies_the_checks_and_refuses_mixed_models():
    fwd, basis = "f" * 64, "b" * 64
    inputs = {"provenance": {"inputs": [{"path": PP.FORWARD, "sha256": fwd},
              {"path": "results/twin_forward/response_basis_innov.npz", "sha256": basis}]}}
    draws = {"cut30": {"headline_towns_unperturbed": ["A", "B", "C"],
                       "share_draws_same_leader": 0.5, "share_draws_same_top3_set": 0.6,
                       "leader_counts": {"A": 250, "D": 250}}}
    lev = {"per_100gwh": {"value": {"irrigation": 2.8, "other": 3.1}, "lever_order": ["other"],
                          "share_tau_1": 0.03, "ranking_survives": False},
           "cm_ranking_detail": {"share_tau_1": 1.0, "ranking_survives": True},
           "headline_townships_basis": draws}
    split = {"policies": {"cut30": {"by_year": {
        "2026": {"page_total": {"fan": 0.9}},
        "2032": {"page_total": {"fan": 1.0}, "B_direct_on_mean_heads": {
            "elastic": {"fan": 1.0}, "inelastic": {"fan": 0.0}}}},
        "C_lift_after_2029": {"avoided_left_dec2032": {"fan": 0.0}}}}}
    margin = {"p5": 0.24, "min": 0.03}
    docs = {"a2b_page_payload": {"starts": {"2026": lev, "2030": lev},
                                 "provenance": {"draws_per_start": 500,
                                                "page": {"sha256": "p" * 64}}},
            "a2b_innov": {"basis_sha256": basis},
            "transfer": {**inputs, "horizon": "2032-12", "ensemble": {
                k: {"fan_mean_cm": [1, 1], "cell_pearson_r": 1, "top_decile_cell_jaccard": 1}
                for k in ("cut30", "retire_aqua")}},
            "hotspot": {"solved": {"cut30": {
                "page_headline": ["A", "B", "C"], "page_margins": {"lead_minus_second_cm": 0.02},
                "innov_runs": {"n": 36, "share_same_leader": 0.8, "share_same_top3_set": 1.0,
                               "share_leader_in_ref_top3": 1.0, "leader_counts": {"A": 29}},
                "cwa_runs": {"n": 36, "share_same_leader": 0.7, "share_same_top3_set": 1.0},
                "cwa_ensemble_headline": ["B", "A", "C"],
                "proposed_label": {"leader_stable": False, "rule": "r"}}}},
            "elastic_split": split,
            "elastic_context": {**inputs, "scenarios": {
                "baseline": {"cells_below_preconsolidation_after_2026_01": 0,
                             "head_margin_above_preconsolidation_m": margin}}}}
    out = PP.build(docs)
    assert out["status"] == "post hoc, model-only, not a test"
    assert (out["forward_sha256"], out["basis_sha256"]) == (fwd, basis)
    assert out["hotspot"]["cut30"]["draws"] == {"n": 500, "share_same_leader": 0.5,
                                                "share_same_top3_set": 0.6}
    assert out["per_gwh"]["ranking_survives"] is False and out["per_gwh"]["cm_ranking_survives"]
    assert out["per_gwh"]["starts"]["2026"]["lo"] == 2.8
    e = out["elastic"]
    assert e["policies"]["cut30"]["inelastic_cm"] == 0.0
    assert e["policies"]["cut30"]["share_by_first_dec"] == pytest.approx(0.9)
    assert e["lift"]["left_cm"] == 0.0 and e["cells_below_preconsolidation_after_2026_01"] == 0
    docs["a2b_innov"] = {"basis_sha256": "c" * 64}           # a different basis: refused
    with pytest.raises(ValueError):
        PP.build(docs)


def test_page_shows_the_posthoc_labels_only_when_given(tmp_path, quiet):
    fw, bs, _ = _write_inputs(tmp_path)
    from hydrophysics.twin.release import sha256

    ph = tmp_path / "posthoc.json"
    ph.write_text(json.dumps(_posthoc(sha256(fw), sha256(bs))))
    opts = dict(leveling="none", wells="none", temporal_npz=None, column_csv=None,
                hsr_csv=None, rivers_csv=None, theta_json=None, alt_npz=None,
                rheo_npz=None, log=lambda s: None, townships_csv=None)
    pages = {}
    for key, path in (("with", str(ph)), ("without", None)):
        out = tmp_path / f"page_{key}.html"
        va.build(str(fw), str(bs), str(out), posthoc_json=path, **opts)
        pages[key] = out
    js = ("() => { D.towns = [{en: 'Baozhong', zh: 'A'}, {en: 'Beigang', zh: 'B'}, "
          "{en: 'Dongshi', zh: 'C'}]; P = {src: 'solved', solved: "
          "D.solved.findIndex(x => x.name === 'cut30')}; S.year = M.years.indexOf(2032); "
          "S.rank = 'gwh'; renderRank(); "
          "return {hot: hotWords([0, 1, 2]), order: hotWords([1, 0, 2]), "
          "gwh: document.getElementById('rankNote').textContent, "
          "cv: caveats().map(c => c[0] + c[1]).join(' | ')}; }")
    pw, browser = _browser()
    try:
        got = {}
        for key, out in pages.items():
            page = browser.new_page()
            page.goto(out.as_uri())
            page.wait_for_function("document.body.dataset.ready === '1'", timeout=30000)
            got[key] = page.evaluate(js)
    finally:
        browser.close()
        pw.stop()
    w, wo = got["with"], got["without"]
    assert "post hoc, model-only, not a test" in w["hot"] and "leads in only 83%" in w["hot"]
    assert "as a group, not an order" in w["hot"] and w["order"] == ""
    assert "near-tie (2.84–3.10 cm)" in w["gwh"] and "3% for 2026 starts" in w["gwh"]
    assert "not a test" in w["cv"] and "all elastic" in w["cv"] and "not separate" not in w["cv"]
    assert "long-run sinking is untested" in w["cv"]
    assert wo["hot"] == "" and "near-tie" not in wo["gwh"] and "all elastic" not in wo["cv"]
