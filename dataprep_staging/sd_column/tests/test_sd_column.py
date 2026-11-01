"""CPU tests of dataprep_staging/sd_column (round 7 ranks 9 and 10): SD.cut_response_spread.v1
and X5.gnss_seasonal_column.v1. No forward runs, no GPU, nothing registered, no look logged,
nothing written outside pytest's tmp dirs, and no observation is read: every series, forward
npz, archive and station table here is SYNTHETIC. The tests that need the gitignored
results/ tree (real column JSONs) skip when it is absent.

    cd <repository root>
    ~/miniforge3/envs/hydro/bin/python -m pytest -q -p no:cacheprovider \\
        dataprep_staging/sd_column/tests
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

TESTS = Path(__file__).resolve().parent
X = TESTS.parent
REPO = X.parents[1]
sys.path.insert(0, str(X))
sys.path.insert(0, str(REPO / "dataprep_staging"))

import gsc_estimator as E  # noqa: E402
import gsc_model_side as M  # noqa: E402
import gsc_synth as G  # noqa: E402
import score_gsc as SG  # noqa: E402
import sdlib as L  # noqa: E402

pd = pytest.importorskip("pandas")
HPY = Path.home() / "miniforge3/envs/hydro/bin/python"


def ang(m):
    return 2 * math.pi * ((np.asarray(m) % 12) + 0.5) / 12.0


def seasonal(m, amp, phase=2.0, trend=0.0):
    m = np.asarray(m)
    return amp * np.cos(ang(m) - phase) + trend * (m - m[0]) / 12.0


# ----------------------------------------------------------------------------- estimator
def test_amplitude_recovered_exactly_without_noise():
    m = np.arange(12 * 2014 + 7, 12 * 2020)
    y = seasonal(m, 0.017, 1.3, trend=0.05) + 0.004 * np.cos(2 * ang(m))
    co = E.year_coefficients(m, y)
    assert sorted(co) == list(range(2015, 2020))      # 2014 has 5 months
    assert E.amplitude(co) == pytest.approx(0.017, abs=1e-9)
    assert E.phase_months(co) == pytest.approx(1.3 / (2 * math.pi) * 12, abs=1e-6)


def test_fast_path_equals_reference():
    rng = np.random.default_rng(3)
    m = np.arange(12 * 2020, 12 * 2023)
    y = seasonal(m, 0.02) + rng.normal(0, 0.003, m.size)
    co = E.year_coefficients(m, y)
    fast = E.coefficients(E.prepare(m), y)
    assert np.allclose(np.array([co[k] for k in sorted(co)]), fast)


def test_year_needs_min_months():
    m = np.concatenate([np.arange(12 * 2020, 12 * 2020 + 7), np.arange(12 * 2021, 12 * 2023)])
    co = E.year_coefficients(m, seasonal(m, 0.01))
    assert sorted(co) == [2021, 2022]


@pytest.mark.parametrize("lo,hi,want", [
    (0.7, 1.4, "consistent"), (1 / 1.5, 1.5, "consistent"), (0.3, 0.6, "larger"),
    (1.6, 2.0, "smaller"), (0.5, 1.2, "inconclusive"), (float("nan"), 1.0, "inconclusive")])
def test_readings(lo, hi, want):
    assert E.reading(lo, hi) == want


def test_scale_robustness():
    assert E.scaled_reading(0.3, 0.5)["verdict"] == "larger"
    assert E.scaled_reading(0.8, 1.2)["verdict"] == "inconclusive"     # 0.56 at 1/1.42
    assert E.scaled_reading(1.0, 1.05)["verdict"] == "consistent"
    assert E.scaled_reading(1.6, 1.9)["verdict"] == "inconclusive"     # 1.13 at 1/1.42
    assert E.scaled_reading(2.2, 2.6)["verdict"] == "smaller"


def pairs_world(r=1.0, noise=0.0, seed=0, n_wra=10):
    rng = np.random.default_rng(seed)
    obs, mod, fam = {}, {}, {}
    for i, s in enumerate(["HUWE", "DNAN"]):
        m = np.arange(12 * 2014 + 7, 12 * 2020)
        y = seasonal(m, [0.025, 0.015][i], trend=0.03)
        mod[s], fam[s] = (m, y), "ngl"
        obs[s] = (m, r * y + rng.normal(0, noise, m.size) if noise else r * y)
    for j in range(n_wra):
        s = f"W{j}"
        m = np.arange(12 * 2020, 12 * 2023)
        y = seasonal(m, rng.uniform(0.01, 0.04), trend=0.04)
        mod[s], fam[s] = (m, y), "wra"
        obs[s] = (m, r * y + rng.normal(0, noise, m.size) if noise else r * y)
    return obs, mod, fam


@pytest.mark.parametrize("r", [0.4, 1.0, 2.5])
def test_pooled_ratio_exact_on_scaled_model(r):
    obs, mod, fam = pairs_world(r)
    pairs = E.station_pairs(obs, mod)
    assert E.pooled_ratio(pairs) == pytest.approx(r, rel=1e-9)
    res = E.score_stratum(pairs, fam, B=300, nuisance=None)
    assert res["ci90"][0] == pytest.approx(r, rel=1e-9)
    assert res["ci90"][1] == pytest.approx(r, rel=1e-9)


def test_bootstrap_deterministic_and_nuisance_widens():
    obs, mod, fam = pairs_world(1.0, noise=0.003, seed=5)
    pairs = E.station_pairs(obs, mod)
    a = E.bootstrap(pairs, fam, B=400, seed=7)
    b = E.bootstrap(pairs, fam, B=400, seed=7)
    assert (a["lo"], a["hi"]) == (b["lo"], b["hi"])
    c = E.bootstrap(pairs, fam, B=400, seed=7, nuisance=None)
    assert (a["hi"] - a["lo"]) > (c["hi"] - c["lo"])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_nuisance_draw_is_mean_neutral(seed):
    """Noise-free r = 1 world: without the nuisance the interval is the point; with it the
    interval widens around r_A instead of shifting away from it (projection form, signed
    thermal term)."""
    obs, mod, fam = pairs_world(1.0, seed=seed)
    pairs = E.station_pairs(obs, mod)
    ci = E.bootstrap(pairs, fam, B=2000, seed=7)
    assert ci["lo"] < 1.0 < ci["hi"]
    mid = 0.5 * (ci["lo"] + ci["hi"])
    assert abs(mid - 1.0) < 0.1 * (ci["hi"] - ci["lo"])


def test_full_modulus_nuisance_would_shift_upward():
    """The rejected form |v - u| inflates every replicate: kept as a regression guard."""
    obs, mod, fam = pairs_world(1.0, seed=0)
    pairs = E.station_pairs(obs, mod)
    rng = np.random.default_rng(7)
    draws = E._draw(rng, pairs, fam, 2000)
    sh = E.nuisance_shifts(rng, draws, 2000)
    so = E._resampled_sum(draws, {s: p["obs"] for s, p in pairs.items()}, sh)
    sm = E._resampled_sum(draws, {s: p["mod"] for s, p in pairs.items()})
    assert abs(np.median(so / sm) - 1.0) < 0.01
    full = 0.0
    for j, (g, pick, k, valid, ymax) in enumerate(draws):
        pad = np.zeros((len(g), ymax, 2))
        for i, s in enumerate(g):
            pad[i, :len(pairs[s]["obs"])] = pairs[s]["obs"]
        v = np.where(valid[..., None], pad[pick[..., None], k], 0.0)
        mean = v.sum(axis=2) / valid.sum(axis=2)[..., None] - sh[j]
        full = full + np.hypot(mean[..., 0], mean[..., 1]).sum(axis=1)
    # second order in |u|/|v| (small at cm amplitudes), but one-sided
    assert np.mean(full / sm) - 1.0 > 0.002
    assert abs(np.mean(so / sm) - 1.0) < np.mean(full / sm) - 1.0


def test_station_pairs_refuses_mismatched_months():
    m = np.arange(12 * 2020, 12 * 2023)
    with pytest.raises(ValueError):
        E.station_pairs({"A": (m, seasonal(m, 0.01))}, {"A": (m[1:], seasonal(m[1:], 0.01))})


# ----------------------------------------------------------------------------- G0-OC
def test_g0oc_small_grid(monkeypatch):
    monkeypatch.setattr(G, "LOADING_MM", (0.0, 6.0))
    monkeypatch.setattr(G, "THERMAL_MM", (2.0,))
    monkeypatch.setattr(G, "SIGMA_MM", (3.0,))
    monkeypatch.setattr(G, "JUMPS", ("off", "tkjs"))
    _, mod, fam = pairs_world(1.0, seed=2)
    st = {s: {"family": fam[s], "m": mod[s][0], "mod": mod[s][1]} for s in mod}
    res = G.run(st, n_rep=20, B=150, seed=1, strata={"T1n": ["HUWE", "DNAN"]})
    assert len(res["cells"]) == 4
    for c in res["cells"]:
        assert c["r=0.4"]["p_consistent"] <= 0.10
        assert c["r=2.5"]["p_consistent"] <= 0.10
        r1 = c["r=1"]
        # the pass statistic is the SCORED rule (both scales), never above the scale-1 one
        assert 0.0 <= r1["p_consistent"] <= r1["p_consistent_scale1"] + 1e-12
        assert r1["p_S_label"] <= r1["p_consistent"] + 1e-12
        assert sum(r1["readings"].values()) == pytest.approx(1.0)
        assert sum(r1["strata"]["T1n"]["readings"].values()) == pytest.approx(1.0)
        assert r1["strata"]["T1n"]["n_stations"] == 2
    m = res["marginal"]
    assert {"p_consistent_r1", "coverage_r1", "p_S_label_r1", "p_consistent_r1_scale1",
            "T1n_r1", "T1n_r04_p_larger", "T1n_r25_p_smaller"} <= set(m)
    assert set(m["T1n_r1"]) == {"p_larger_or_smaller", "p_inconclusive", "p_consistent"}
    assert res["grid"]["p_miss"] == G.P_MISS and "both" in res["pass_rule"]
    want = (all(c["false_consistent_ok"] for c in res["cells"])
            and m["p_consistent_r1"] >= 0.80 and m["coverage_r1"] >= 0.85)
    assert res["pass"] is want


def test_g0oc_missingness_drops_months(monkeypatch):
    monkeypatch.setattr(G, "P_MISS", 0.5)
    _, mod, fam = pairs_world(1.0, seed=2, n_wra=2)
    st = {s: {"family": fam[s], "m": mod[s][0], "mod": mod[s][1]} for s in mod}
    rng = np.random.default_rng(0)
    c = G.run_cell(st, 1.0, 0.0, 0.0, 2.0, "off", rng, 5, 50)
    assert c["n_stations"] < len(st)       # 3-year WRA windows lose years at 50 % missing


def test_g0oc_worlds_use_declared_terms():
    rng = np.random.default_rng(0)
    m = np.arange(12 * 2020, 12 * 2023)
    st = {"W": {"family": "wra", "m": m, "mod": np.zeros(m.size)}}
    w = G.worlds(st, 1.0, 0.0, 2.0, 1e-9, "off", rng, 4)["W"]
    want = -0.002 * np.cos(ang(m) - 2 * math.pi * 6.5 / 12)        # thermal only
    assert np.allclose(w, want[None, :], atol=1e-6)


# ----------------------------------------------------------------------------- SD library
def test_forward_commands():
    for k, f in L.FORWARDS.items():
        c = L.forward_command(k)
        assert "--gain 0.0" in c and "--ic-members 0" in c and "--column-heads free" in c
        assert "--extend-record-to" not in c and "--record-end" not in c
        assert f"--theta {L.theta_path(f['run'])} " in c
        assert ("--hindcast-gain 1.0 --hindcast-every 1" in c) == (k == "q0nh")
        assert c.endswith(f"--out {L.out_stem(k)}")
    assert "rf_timeseries_vendor_2012_2022" in L.forward_command("page")
    every = [c for f in L.FORWARDS.values() for c in f["columns"]]
    assert sorted(every) == sorted(L.COLUMNS)


def test_runner_dry_matches_declared_commands():
    if not HPY.exists():
        pytest.skip("hydro env python absent")
    env = {"DRY": "1", "PATH": "/usr/bin:/bin", "HOME": str(Path.home())}
    out = subprocess.run(["bash", str(X / "run_sd.sh"), "forward", "all"], cwd=REPO,
                         env=env, capture_output=True, text=True, check=True).stdout
    assert out.splitlines() == [L.forward_command(k) for k in L.FORWARDS]


def test_no_home_paths_or_usernames_in_committed_files():
    for p in sorted(X.rglob("*")):
        if p.is_file() and p.suffix in (".py", ".sh") and "__pycache__" not in p.parts:
            t = p.read_text()
            assert not re.search(r"/home/\w", t), p
            assert not re.search(rf"\b{re.escape(Path.home().name)}\b", t), p


def fake_forward(fwd="page", gain=0.0, scen=("baseline", "cut30"), labels=None):
    T, A = L.N_T, 6
    d = np.array([f"{2012 + i // 12}-{i % 12 + 1:02d}-01T00:00:00" for i in range(T)])
    return {"scenario_names": np.array(scen),
            "scenarios": np.array([L.BASE_DESC, L.CUT30_DESC]), "dates": d,
            "origin": np.array(L.ORIGIN), "n_members": np.array(1),
            "member_labels": np.array([L.FORWARDS[fwd]["run"]]), "gain": np.array(gain),
            "hindcast_gain": np.array(1.0 if fwd == "q0nh" else 0.0),
            "rheology_labels": np.array(labels or L.expected_labels(fwd)),
            "forward_options": np.array(json.dumps({
                "column_heads": "free", "update_method": "innovation",
                "apex_hold": "calibrated", "column_hpc0_fast_days": 365.0,
                "save_members": "none", "column_zone_blend_km": [0.0]})),
            "heads_mean": np.zeros((2, 4, L.N_CELLS, T), dtype="float32")[:, :, :A]
            if False else np.zeros((2, 4, L.N_CELLS, T), dtype="float32")}


def test_check_forward_npz():
    assert L.check_forward_npz(fake_forward("page"), "page") == []
    assert L.check_forward_npz(fake_forward("q0"), "q0") == []
    assert L.expected_labels("q0") == ["coupled_leveling", "coupled_leveling_1"]
    assert any("gain" in e for e in L.check_forward_npz(fake_forward(gain=1.0), "page"))
    assert any("rheology" in e for e in L.check_forward_npz(
        fake_forward(labels=["x", "y", "z"]), "page"))
    assert any("hindcast_gain" in e for e in L.check_forward_npz(fake_forward("q0"), "q0nh"))


def test_hindcast_repro_single_and_members(tmp_path):
    fwd = tmp_path / L.FWD
    fwd.mkdir(parents=True)
    rng = np.random.default_rng(0)
    sub = np.cumsum(rng.normal(0.001, 0.0005, (2, 5, L.N_T)), axis=-1)
    np.savez(fwd / "arch.npz", subs_mean=sub.astype("float32"), origin=np.array(L.ORIGIN))
    z = {"subs_mean": sub}
    ok = L.hindcast_repro(z, 0, ("single", "arch"), tmp_path)
    assert ok["pass"] and ok["max_abs_m"] < 1e-6
    bad = L.hindcast_repro({"subs_mean": sub + 0.002}, 0, ("single", "arch"), tmp_path)
    assert not bad["pass"]
    q = 0.0005
    ye = np.array([i for i in range(L.N_T) if i % 12 == 11])
    yr = np.round(sub[:, None, :, :][..., ye] / q).astype("int16")
    yr = np.concatenate([yr, yr + 3], axis=1)                    # 2 rows: main, other
    np.savez(fwd / "arch.members.npz", subs_members_yr=yr, subs_scale_m=np.array(q),
             ye_idx=ye, member=np.array(["main", "fold0"]), ic=np.array([0, 0]),
             rheology=np.array(["coupled_leveling", "coupled_leveling"]))
    rec = L.hindcast_repro(z, 0, ("members", "arch", "main", "coupled_leveling"), tmp_path)
    assert rec["pass"] and rec["decembers"] == 11
    rec = L.hindcast_repro(z, 0, ("members", "arch", "fold0", "coupled_leveling"), tmp_path)
    assert not rec["pass"]
    assert L.hindcast_repro(z, 0, None, tmp_path)["pass"] is None


def vep_column(params):
    import torch

    from hydrophysics.twin.forward import ZonalColumn
    return ZonalColumn(params, np.array([0, 0, 1, 1, 2, 2]), device=torch.device("cpu"))


def test_elastic_split_is_exact_when_no_inelastic_change():
    pytest.importorskip("torch")
    p = [{"log_ske": math.log(1e-4), "log_skv": math.log(5e-3), "log_tau": math.log(30),
          "h_pc0": 0.0},
         {"log_ske": math.log(9e-3), "log_skv": math.log(4e-2), "log_tau": math.log(3960),
          "h_pc0": 0.0},
         {"log_ske": math.log(2e-2), "log_skv": math.log(3e-2), "log_tau": math.log(57),
          "h_pc0": 0.0}]
    col = vep_column(p)
    t = np.arange(L.N_T)
    base = (5.0 + 0.5 * np.sin(2 * np.pi * t / 12) - 0.01 * np.minimum(t, 100))[None, :]
    base = np.repeat(base, 6, axis=0)
    cut = base + 0.8 * (t >= L.month_index("2026-01"))[None, :]     # cut raises heads
    drv = np.stack([base, cut]).astype("float32")
    subs = np.stack([L.compaction_cpu(col, drv[s]) for s in range(2)])
    av = L.avoided_cm(subs)
    ske = np.exp(np.array([q["log_ske"] for q in p]))[[0, 0, 1, 1, 2, 2]]
    el = L.elastic_avoided_cm(ske, drv)
    th = L.month_index("2032-12")
    assert np.allclose(av[:, :L.ORIGIN + 1], 0.0)
    assert np.allclose(el[:, th], ske * 0.8 * 100, rtol=1e-5)
    assert np.allclose(av[:, th], el[:, th], atol=1e-6)          # all elastic
    assert av[4, th] > av[2, th] > av[0, th] > 0


def test_summarise_groups_and_shares():
    A, T = 6, L.N_T
    zone = np.array([0, 1, 2, 0, 1, 2])
    ch = np.array([False, False, False, True, True, True])
    av = np.zeros((A, T))
    av[:, L.HORIZONS["2028-12"]] = [0.1, 0.5, 2.0, 0.0, 0.2, 1.0]
    dh = np.full((A, T), 0.5)
    s = L.summarise(av, av * 0.9, dh, dh, L.group_masks(zone, ch))["2028-12"]
    assert s["yunlin"]["avoided_cm"] == pytest.approx(2.6 / 3)
    assert s["yunlin_inland"]["avoided_cm"] == pytest.approx(0.3)
    assert s["distal_share_yunlin"] == pytest.approx(2.0 / 2.6)
    assert s["distal_share_fan"] == pytest.approx(3.0 / 3.8)
    assert s["yunlin"]["elastic_share"] == pytest.approx(0.9)
    assert s["yunlin_distal"]["cm_per_m"] == pytest.approx(4.0)


def test_readings_inland_and_rank():
    r = L.reading_inland({"a": 1.0, "b": 1.4, "c": 1.2})
    assert r["reading"] == "inland Yunlin robust within x1.5"
    r = L.reading_inland({"a": 1.0, "b": 2.1})
    assert r["reading"].startswith("column-dependent x2.10")
    assert not L.reading_inland({"a": 1.0, "b": -0.1})["same_sign"]
    rk = L.reading_rank({"page": [1, 2, 3, 4, 5], "q": [1, 2, 3, 4, 9], "w": [2, 1, 3, 9, 8]},
                        "page")
    assert rk["min_overlap"] == 3 and rk["reading"] == "ranking column-dependent"
    assert rk["C11_leader_wording"] == "lost"


def test_top_towns():
    town = np.array([0, 0, 1, 2, 2, 3])
    elig = np.array([True, True, False, True])
    v = np.array([1.0, 3.0, 9.0, 0.5, 0.5, 4.0])
    assert L.top_towns(v, town, elig, 3) == [1, 3, 0]       # town 2 not eligible


def test_storage_ratio_page_like():
    theta = {"theta": {"log_S_distal": [math.log(0.001), math.log(0.000923)],
                       "log_S_mid": [math.log(0.002)]}}
    col = {"zonal": [{"log_ske": math.log(1e-4)}, {"log_ske": math.log(9e-3)},
                     {"log_ske": math.log(0.0215)}]}
    r = L.storage_ratio(theta, col)
    assert r["distal"]["ratio"] == pytest.approx(0.0215 / 0.001923)
    assert r["mid"]["ratio"] == pytest.approx(4.5)


# ----------------------------------------------------------------------------- model side
class FakeGrid:
    """active_index = the cell whose 1 km square holds (x, y) on a 3 x 3 toy grid."""

    def active_index(self, x, y):
        i, j = int(x // 1000), int(y // 1000)
        return i + 3 * j if 0 <= i < 3 and 0 <= j < 3 else None


def stations_csv(path: Path) -> None:
    rows = [
        "family,station,x_twd97,y_twd97,twin_cell,role,note,first_day,last_day,n_days,"
        "n_flag0_or_1_or_3,n_steps,rate_m_yr,annual_amp_m,sigma_hf_m,rho_hf,notes",
        "wra,ANES,500,500,0,member,,2020-01-01,2022-12-31,1080,1,0,NOTREAD,NOTREAD,x,x,",
        "wra,AAAA,1500,500,1,member,,2020-01-01,2022-12-31,1080,1,0,NOTREAD,NOTREAD,x,x,",
        "wra,BBBB,2500,2500,8,member,,2020-01-01,2020-06-30,180,1,0,NOTREAD,NOTREAD,x,x,",
        "wra,OFFG,9999,9999,-1,member,,2020-01-01,2022-12-31,1080,1,0,NOTREAD,NOTREAD,x,x,",
        "ngl,HUWE,500,1500,3,fan box,,2014-08-15,2022-12-31,3000,1,0,NOTREAD,NOTREAD,x,x,",
        "ngl,PINT,1500,1500,-1,off-fan reference,,2007-01-01,2022-12-31,5000,1,0,N,N,x,x,",
        "ngl,LUKN,1500,2500,7,fan box,,2007-01-01,2022-12-31,5000,1,0,NOTREAD,NOTREAD,x,x,",
        "wra,CHSG,2500,1500,8,member,,2020-01-01,2022-12-31,1030,1,0,NOTREAD,NOTREAD,x,x,",
        "ngl,ERLN,1500,500,1,fan box,,2012-01-01,2019-08-03,2700,1,0,NOTREAD,NOTREAD,x,x,",
        "ngl,JSAN,2500,500,2,on grid,,2015-10-21,2019-12-31,1500,1,0,NOTREAD,NOTREAD,x,x,"]
    path.write_text("\n".join(rows) + "\n")


def test_station_rule_reads_metadata_only(tmp_path):
    p = tmp_path / "stations.csv"
    stations_csv(p)
    meta = M.read_station_metadata(p)
    assert list(meta.columns) == M.META_COLUMNS
    ch = np.array([False] * 6 + [True] * 3)
    zone = np.array([2, 1, 0, 2, 1, 0, 2, 1, 0])
    st = M.station_rule(meta, FakeGrid(), ch, zone)
    assert "OFFG" not in st and "PINT" not in st
    assert st["HUWE"]["window"] == ["2014-08-15", "2019-12-31"]
    assert st["LUKN"]["window"] == ["2012-01-01", "2019-12-31"]
    assert st["AAAA"]["eligible"] and st["AAAA"]["county"] == "yunlin"
    assert not st["BBBB"]["eligible"]                    # 181 days
    assert st["ANES"]["report_only_reason"].startswith("seen")
    assert st["LUKN"]["county"] == "changhua"
    assert st["CHSG"]["cell_check"] is False             # (2500, 1500) is cell 5, not 8
    assert st["ERLN"]["county"] == "yunlin" and st["JSAN"]["county"] == "yunlin"
    page = {s: {"amplitude_m": 0.005, "n_years": 4} for s in st}
    page["AAAA"]["amplitude_m"] = 0.0005                  # below the 1 mm floor
    u = M.assign_units(st, page)
    assert u["ngl_stratum"] == ["HUWE"]
    assert u["primary"] == ["HUWE"]                       # AAAA under 1 mm, ANES seen
    assert set(u["changhua_arm"]) == {"LUKN", "ERLN", "CHSG", "BBBB"}   # ERLN: by name
    assert {"JSAN", "ANES", "AAAA"} <= set(u["report_only"])
    assert st["JSAN"]["report_only_reason"] == M.NGL_OTHER_REASON
    assert st["ERLN"]["unit_basis"].endswith("yunlin")
    assert M.unit_scope(st, u["primary"]) == {"window": "2014-08..2019-12",
                                              "data_scope": "gnss:2014-08..2019-12"}
    assert M.unit_scope(st, ["HUWE", "AAAA"])["data_scope"] == "gnss:2014-08..2022-12"


def test_unnamed_ngl_station_is_report_only():
    st = {"XXXX": {"family": "ngl", "county": "yunlin", "eligible": True,
                   "report_only_reason": None}}
    u = M.assign_units(st, {"XXXX": {"amplitude_m": 0.01, "n_years": 5}})
    assert u["primary"] == [] and u["report_only"] == ["XXXX"]
    assert M.NGL_STRATUM == ("DNAN", "HUWE")


def test_model_series_window():
    sub = np.zeros((2, 3, L.N_T))
    sub[0, 1] = np.arange(L.N_T)
    m, y = M.model_series(sub, 1, ["2014-08-15", "2014-12-31"])
    assert list(m) == list(range(12 * 2014 + 7, 12 * 2015))
    assert list(y) == [31.0, 32.0, 33.0, 34.0, 35.0]
    with pytest.raises(ValueError):
        M.model_series(sub, 1, ["2022-06-01", "2023-02-01"])


# ----------------------------------------------------------------------------- GNSS scorer
def daily_frame(stations: dict, r: float, rng, cm_amp=0.004):
    rows, cms = [], {}
    for s, st in stations.items():
        for d in pd.date_range(st["window"][0], st["window"][1], freq="D"):
            m = 12 * d.year + d.month - 1
            ds = d.strftime("%Y-%m-%d")
            cms[ds] = cm_amp * math.sin(d.dayofyear / 365.25 * 2 * math.pi)
            subs = r * float(seasonal(np.array([m]), st["amp"])[0])
            flag = 9 if d.day == 15 else (3 if d.day == 16 else 0)
            v = -subs + cms[ds] + (5.0 if flag else 0.0) + rng.normal(0, 0.0005)
            rows.append({"family": st["family"], "station": s, "date": ds, "value": v,
                         "flag": flag})
    cm = pd.DataFrame({"date": list(cms), "ngl_offfan": list(cms.values()),
                       "wra_hf": 0.0})
    return pd.DataFrame(rows), cm


def gsc_world(r=0.5, seed=0):
    rng = np.random.default_rng(seed)
    st = {"HUWE": {"family": "ngl", "window": ["2016-01-01", "2019-12-31"], "amp": 0.02,
                   "zone": "distal", "unit": "primary", "ngl_stratum": True},
          "DNAN": {"family": "ngl", "window": ["2016-01-01", "2019-12-31"], "amp": 0.012,
                   "zone": "mid", "unit": "primary", "ngl_stratum": True},
          "W1": {"family": "wra", "window": ["2020-01-01", "2022-12-31"], "amp": 0.03,
                 "zone": "distal", "unit": "primary", "ngl_stratum": False},
          "W2": {"family": "wra", "window": ["2020-01-01", "2022-12-31"], "amp": 0.015,
                 "zone": "mid", "unit": "primary", "ngl_stratum": False},
          "ANES": {"family": "wra", "window": ["2020-01-01", "2022-12-31"], "amp": 0.02,
                   "zone": "distal", "unit": "report_only", "ngl_stratum": False},
          "LUKN": {"family": "ngl", "window": ["2016-01-01", "2019-12-31"], "amp": 0.005,
                   "zone": "mid", "unit": "changhua_arm", "ngl_stratum": False}}
    daily, cm = daily_frame(st, r, rng)
    series = {}
    for arm in ("page", "ccz"):
        for s, v in st.items():
            m = M.months_between(*v["window"])
            series[f"{arm}|{s}|m"] = m
            series[f"{arm}|{s}|y"] = seasonal(m, v["amp"])
    doc = {"stations": st, "units": {
        "primary": ["DNAN", "HUWE", "W1", "W2"], "ngl_stratum": ["DNAN", "HUWE"],
        "changhua_arm": ["LUKN"], "report_only": ["ANES"]},
        "arms": {"page": {"full_window": {s: {"amplitude_m": v["amp"]} for s, v in st.items()}},
                 "ccz": {"full_window": {}}}, "f_subs_page": 0.9}
    return doc, series, daily, cm


def test_obs_monthly_flags_common_mode_and_sign():
    doc, series, daily, cm = gsc_world(r=1.0)
    obs = SG.obs_monthly(daily, cm, doc["stations"], "ngl_offfan")
    m, y = obs["W1"]
    assert len(m) == 36
    co = E.year_coefficients(m, y)
    assert E.amplitude(co) == pytest.approx(0.03, rel=0.03)    # flags 3 and 9 dropped
    raw = SG.obs_monthly(daily, cm, doc["stations"], None)
    assert E.amplitude(E.year_coefficients(*raw["W1"])) != pytest.approx(0.03, rel=0.03)


def test_compute_end_to_end_larger():
    doc, series, daily, cm = gsc_world(r=0.4)
    obs = SG.obs_monthly(daily, cm, doc["stations"], "ngl_offfan")
    alt = {"none": SG.obs_monthly(daily, cm, doc["stations"], None)}
    res = SG.compute(doc, series, obs, alt, B=300)
    assert res["T1"]["r_A"] == pytest.approx(0.4, rel=0.05)
    assert res["T1"]["verdict"] == "larger"
    assert res["page_mapping"]["C32_status"] == "N"
    assert "larger than observed" in res["page_mapping"]["line"]
    assert set(res["report_only"]["changhua_arm"]) == {"page", "ccz"}
    assert "ANES" in res["report_only"]["seen_stations_ANES_CHSG"]
    assert res["T1"]["stations"] == ["DNAN", "HUWE", "W1", "W2"]


def test_station_ids_carry_both_registry_forms():
    assert SG.station_ids(["GFES", "ANES"]) == ["ANES", "GFES", "gnss:ANES", "gnss:GFES"]


def test_page_mapping_strata_clash():
    t1 = {"verdict": "consistent", "r_A": 1.1, "ci90": [1.0, 1.2]}
    assert SG.page_mapping(t1, {"verdict": "inconclusive"})["C32_status"].startswith("S")
    pm = SG.page_mapping(t1, {"verdict": "larger"})
    assert pm["C32_status"] == "N" and pm["strata_clash"]


# ----------------------------------------------------------------------------- registrations
def test_gsc_registration_document_validates(monkeypatch, tmp_path):
    import prereg_gsc as PG
    from harness import prereg3

    monkeypatch.setattr(PG, "repo", lambda: tmp_path)
    d = PG.doc()
    assert prereg3.validate_prereg(d, units_registry=None, prereg_dir=tmp_path / "p",
                                   registries=[]) == []
    assert [t["test_id"] for t in d["tests"]] == [SG.T1, SG.T1N]
    assert "model side not frozen: run_gsc.sh model-side" in PG.readiness(d)
    assert d["page_mapping"]["ruling"].startswith("R7.gnss_seasonal_S")
    assert sorted(d["frozen_code"]) == sorted(SG.FROZEN)
    assert d["g0oc"]["power_statement"].startswith("pending")
    assert "both" in d["g0oc_design"]["pass_rule"]
    for t in d["tests"]:
        (u, v), = t["external_units"].items()
        assert "NOT fresh in the U7 sense" in v["description"]


def test_gsc_readiness_needs_logged_g0oc(monkeypatch, tmp_path):
    import prereg_gsc as PG
    from harness import prereg3

    monkeypatch.setattr(PG, "repo", lambda: tmp_path)
    assert not PG.g0_logged(tmp_path, "abc123")
    pdir = tmp_path / "results/dataprep/preregistration"
    pdir.mkdir(parents=True)
    unit = PG.G0_LOOK_UNIT
    prereg3.record_look([unit], "synthetic:2014-08..2022-12", "X5", "integrity_check",
                        members={unit: {"none": "synthetic worlds",
                                        "description": "G0-OC"}},
                        note="g0oc.json sha256 abc123", prereg_dir=pdir,
                        rows_dir=tmp_path / "rows", registries=[])
    assert PG.g0_logged(tmp_path, "abc123")
    assert not PG.g0_logged(tmp_path, "def456")


def test_power_statement_numbers():
    import prereg_gsc as PG

    g = {"marginal": {"p_consistent_r1": 0.12, "coverage_r1": 0.9, "p_S_label_r1": 0.1,
                      "p_consistent_r1_scale1": 0.6,
                      "T1n_r1": {"p_larger_or_smaller": 0.05, "p_inconclusive": 0.9,
                                 "p_consistent": 0.05}},
         "worst_cell": {"p_consistent_r1": 0.0, "p_consistent_r04": 0.0,
                        "p_consistent_r25": 0.02}}
    s = PG.power_statement(g)
    assert "P(consistent | r = 1) = 0.120" in s and "P(inconclusive) = 0.900" in s


from common.paths import REPO as DATA  # noqa: E402  (DATAPREP_REPO may point elsewhere)

needs_results = pytest.mark.skipif(
    not (DATA / L.column_path(L.PAGE_COLUMN)).exists(),
    reason="results/ (column JSONs, thetas) not present in this checkout")


@needs_results
def test_sd_declaration_round_trips_in_tmp_registry(tmp_path):
    import prereg_sd as PS
    from harness import prereg3

    d = PS.doc()
    assert prereg3.validate_prereg(d) == []
    assert d["columns"]["page"]["column_sha256"].startswith(L.PAGE_COLUMN_SHA8)
    assert d["columns"]["page"]["included"]
    path, h = prereg3.write_prereg(d, prereg_dir=tmp_path / "prereg",
                                   rows_dir=tmp_path / "rows", registries=[])
    assert path.exists() and len(h) == 64


needs_stations = pytest.mark.skipif(
    not (DATA / M.STATIONS_CSV).exists() or not (DATA / "chou-shui-data").exists(),
    reason="stations.csv metadata or the fan polygon not present in this checkout")


@needs_stations
def test_real_grid_units_pin_ngl_by_name():
    """Real twin grid and county mask, stations.csv FIRST TEN COLUMNS ONLY (metadata), a
    uniform fake page amplitude: no observation value is read."""
    gm = L.grid_and_masks(DATA)
    st = M.station_rule(M.read_station_metadata(DATA / M.STATIONS_CSV), gm["grid"],
                        gm["changhua"], gm["zone"])
    u = M.assign_units(st, {s: {"amplitude_m": 0.005, "n_years": 4} for s in st})
    assert u["ngl_stratum"] == ["DNAN", "HUWE"]
    assert {s for s in u["primary"] if st[s]["family"] == "ngl"} == {"DNAN", "HUWE"}
    assert "ERLN" in u["changhua_arm"] and st["ERLN"]["county"] == "yunlin"
    assert "JSAN" not in u["primary"]
    assert "ANES" not in u["primary"] and "CHSG" not in u["primary"]
    assert M.unit_scope(st, u["ngl_stratum"])["data_scope"] == "gnss:2014-08..2019-12"
    assert M.unit_scope(st, u["primary"])["data_scope"] == "gnss:2014-08..2022-12"


@needs_results
def test_inclusion_bar_on_published_values():
    inc = L.inclusion(DATA)
    assert inc["bar"] == pytest.approx(inc["page_r2_outoffold"] - 0.02)
    assert all(v["included"] for c, v in inc["columns"].items() if L.COLUMNS[c][2] == "CH")


def test_sd_scorer_frozen_list_matches_runner():
    import score_sd as SS

    sh = (X / "run_sd.sh").read_text()
    for p in SS.FROZEN:
        assert (REPO / p).exists()
        assert p.split("/")[-1] in sh or p.endswith("run_sd.sh")
    assert os.access(X / "run_sd.sh", os.R_OK)


def test_aggregate_readings_and_page_line():
    import score_sd as SS

    A, T = 6, L.N_T
    zone = np.array([0, 1, 2, 0, 1, 2])
    ch = np.array([False, False, False, True, True, True])
    masks = L.group_masks(zone, ch)
    t = np.arange(T)
    on = (t >= L.month_index("2026-01")).astype(float)
    towns = {"town": np.array([0, 1, 2, 3, 3, 3]), "names": ["a", "b", "c", "ch"],
             "eligible": np.array([True, True, True, False])}
    keep = {}
    for c, k in (("page", 1.0), ("page_tau30", 1.2), ("q0", 2.4)):
        ske = np.array([1e-4, 9e-3, 2e-2, 1e-4, 9e-3, 2e-2]) * k
        hb = np.zeros((A, T))
        hc = 0.5 * on[None, :].repeat(A, axis=0)
        drv = np.stack([hb, hc]).astype("float32")
        subs = np.stack([np.zeros((A, T)), -ske[:, None] * hc])         # cut sinks less
        keep[c] = {"subs": subs, "drv": drv, "ske": ske,
                   "heads": np.stack([np.stack([hb] * 4), np.stack([hc] * 4)]),
                   "theta": {"theta": {"log_S_distal": [math.log(0.002)],
                                       "log_S_mid": [math.log(0.002)]}},
                   "col_json": {"zonal": [{"log_ske": math.log(s)} for s in ske[:3]]}}
    agg = SS.aggregate(None, keep, list(keep), masks, towns)
    s = agg["page"]["summary"]["2028-12"]
    assert s["yunlin_distal"]["avoided_cm"] == pytest.approx(1.0)       # 0.02 x 0.5 m
    assert s["yunlin"]["elastic_share"] == pytest.approx(1.0)
    assert agg["page"]["f_subs_yunlin"] == pytest.approx(1.0)
    assert agg["page"]["top5_towns"]["2032-12"] == ["c", "b", "a"]
    rd = SS.readings(agg, {c: c for c in agg}, {"A": ["page", "page_tau30"],
                                                "A+B": ["page", "page_tau30", "q0"]})
    assert rd["A"]["2028-12"]["inland"]["reading"] == "inland Yunlin robust within x1.5"
    assert rd["A+B"]["2028-12"]["inland"]["reading"].startswith("column-dependent x2.40")
    assert rd["A+B"]["rank_2032"]["reading"] == "ranking column-robust"
    line = SS.page_line(rd)
    assert line.startswith("across 3 leveling-equivalent columns adding the Q0 family")
    assert "model-form spread, not an interval" in line
    la = SS.page_line(rd, "A")
    assert la.startswith("across 2 leveling-equivalent columns on the page's own heads")
    assert "robust within x1.5" in la
    assert SS.page_line({"A": {"columns": []}}, "A") is None


def test_reading_sets_with_and_without_unverified():
    import score_sd as SS

    ok = ["page", "page_tau30", "page_aquitard", "q0", "q0_nh", "ccz", "x5r1_cs"]
    sets = SS.reading_sets(ok, {c: c not in L.UNVERIFIED for c in ok})
    assert sets["A"] == ["page", "page_tau30", "page_aquitard"]
    assert sets["A verified only"] == ["page", "page_tau30"]
    assert sets["A+B verified only"] == ["page", "page_tau30", "q0"]
    assert "q0_nh" in sets["A+B"] and "x5r1_cs" not in sets["A+B"]


def test_q0_nh_archive_never_read():
    """judge rank 9: datum_gate_q0_nh.npz assimilated heads to 2025-06 and is never used."""
    import prereg_sd as PS

    assert L.COLUMNS["q0_nh"][3] is None and "q0_nh" in L.UNVERIFIED
    assert all("datum_gate_q0_nh" not in a for a in PS.archives())
    assert L.COLUMNS["page_aquitard"][3][0] == "shared" and "page_aquitard" in L.UNVERIFIED
