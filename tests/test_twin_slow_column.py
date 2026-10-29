"""E1 (2026-10-28): the slow-component column flags of calibrate_coupled.

--aquitard-tau-min-years / --aquitard-ha0-min (per-column aquitard bounds), --creep-zones
(the aquitard term in some zones only) and --tau-min-days-zone (a per-zone VEP tau floor).
All are opt-in: off, the column, its clamp and its JSON are the historical ones; on, the
floors hold, the JSON records them and twin.forward rebuilds the mixed column or refuses
one whose recorded rheology its parameters do not match.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import hydrophysics.twin.calibrate_coupled as cc  # noqa: E402
from hydrophysics.twin.compaction import (  # noqa: E402
    AQUITARD_BOUNDS,
    AQUITARD_KEYS,
    VEP_KEYS,
    VEPColumn,
    aquitard_bounds,
    aquitard_compaction,
    column_compaction,
    zone_creep,
)
from hydrophysics.twin.forward import (  # noqa: E402
    ZonalColumn,
    blended_column,
    check_column_creep,
    load_or_fit_vep,
)

DT = 30.0
YR = 365.25


@pytest.fixture
def e1_globals(monkeypatch):
    """Every module global ``calibrate_coupled.main`` sets, at its default; monkeypatch
    restores the original values after the test."""
    for k in ("TAU_MAX_YEARS", "HPC0_GUARD_DAYS", "TAU_MIN_DAYS", "SKE_MIN", "SKE_SKV_MAX",
              "SKE_CAP_KAPPA", "SKE_CAPS", "SKE_CAP_ZONES", "AQUITARD_TAU_MIN_YEARS",
              "AQUITARD_HA0_MIN", "TAU_MIN_DAYS_ZONE"):
        monkeypatch.setattr(cc, k, None)
    monkeypatch.setattr(cc, "CREEP", "vep")
    monkeypatch.setattr(cc, "CREEP_ZONES", "all")
    monkeypatch.setattr(cc, "DETERMINISTIC", False)
    for k in ("N_BANDS", "BAND_LAMBDA", "BAND_KIND", "BAND_HPC0_GUARD"):
        monkeypatch.setattr(cc, k, getattr(cc, k))
    return cc


def _aq(ska=0.05, tau_a_yr=20.0, h_a0=5.0):
    t = torch.float64
    return (torch.tensor([math.log(ska)], dtype=t), torch.tensor([math.log(tau_a_yr * YR)],
                                                                  dtype=t),
            torch.tensor([h_a0], dtype=t))


def _synthetic_fit(seed: int = 3, n: int = 12, T: int = 40):
    g = torch.Generator().manual_seed(seed)
    H = -torch.cumsum(torch.rand(n, 4, T, generator=g) * 0.4, dim=-1)
    obs = 0.02 * (H[:, :, 0:1] - H).mean(dim=1)
    M = torch.ones(n, T)
    Z = torch.tensor([i % 3 for i in range(n)], dtype=torch.long)
    return H, obs, M, Z


def _all_params(model) -> list[dict]:
    return [{k: float(getattr(c, k).detach()) for k in c.param_keys()} for c in model]


# ---- the aquitard term, analytically ------------------------------------------------------
def test_aquitard_rate_at_constant_head_is_the_analytic_decay():
    """At constant head the interbeds drain their excess h_a0 with time constant tau_a:
    compaction Ska*h_a0*(1 - exp(-t/tau_a)), rate Ska*h_a0/tau_a*exp(-t/tau_a)."""
    ska, tau_yr, ha0 = 0.05, 20.0, 5.0
    T = 361
    h = torch.full((1, T), -3.0, dtype=torch.float64)
    s = aquitard_compaction(h, *_aq(ska, tau_yr, ha0), DT)[0].numpy()
    t = np.arange(T) * DT
    tau = tau_yr * YR
    assert np.allclose(s, ska * ha0 * (1.0 - np.exp(-t / tau)), rtol=0, atol=1e-12)
    # the yearly rate at mid-interval is the continuous rate to second order
    rate = (s[12:] - s[:-12]) / (12 * DT)                   # m/day over each year
    mid = t[:-12] + 6 * DT
    assert np.allclose(rate, ska * ha0 / tau * np.exp(-mid / tau), rtol=1e-3)
    # ~1.2 cm/yr at the start, as Ska*h_a0/tau_a says
    assert rate[0] * YR == pytest.approx(ska * ha0 / tau_yr, rel=0.03)


def test_head_rise_cuts_the_aquitard_rate_by_ska_d_over_tau():
    """A sustained head rise of d (smaller than the excess left) cuts the rate by
    Ska*d/tau_a at once; the cut decays with tau_a and the avoided compaction tends to
    Ska*d."""
    ska, tau_yr, ha0, d, t0 = 0.05, 20.0, 8.0, 2.0, 24
    T = 300
    base = torch.zeros(1, T, dtype=torch.float64)
    up = base.clone()
    up[:, t0:] += d
    s0 = aquitard_compaction(base, *_aq(ska, tau_yr, ha0), DT)[0].numpy()
    s1 = aquitard_compaction(up, *_aq(ska, tau_yr, ha0), DT)[0].numpy()
    a = 1.0 - math.exp(-DT / (tau_yr * YR))
    k = np.arange(t0, T)
    avoided = s0[t0:] - s1[t0:]
    assert np.allclose(avoided, ska * d * (1.0 - (1.0 - a) ** (k - t0 + 1)), atol=1e-12)
    assert np.allclose(s0[:t0], s1[:t0])                          # nothing before the rise
    # first-month rate cut, per day: Ska*d*a/dt ~ Ska*d/tau_a
    cut = ((s0[t0] - s0[t0 - 1]) - (s1[t0] - s1[t0 - 1])) / DT
    assert cut == pytest.approx(ska * d / (tau_yr * YR), rel=0.01)
    assert avoided[-1] < ska * d and avoided[-1] > 0.9 * ska * d * (1 - (1 - a) ** (T - t0))


def test_aquitard_bounds_floors_only_tighten():
    assert aquitard_bounds() == AQUITARD_BOUNDS
    b = aquitard_bounds(10.0, 0.0)
    assert b["log_tau_a"] == (pytest.approx(math.log(10 * YR)), AQUITARD_BOUNDS["log_tau_a"][1])
    assert b["h_a0"] == (0.0, AQUITARD_BOUNDS["h_a0"][1])
    assert b["log_ska"] == AQUITARD_BOUNDS["log_ska"]
    assert aquitard_bounds(1.0, -30.0) == AQUITARD_BOUNDS          # the defaults themselves
    for bad in ((0.5, None), (100.0, None), (None, -31.0), (None, 30.0)):
        with pytest.raises(ValueError):
            aquitard_bounds(*bad)


def test_column_with_bounds_starts_inside_them():
    c = VEPColumn(n_sites=1, creep="aquitard", aquitard_bounds=aquitard_bounds(20.0, 0.0))
    assert float(torch.exp(c.log_tau_a.detach())) == pytest.approx(20 * YR, rel=1e-5)
    d = VEPColumn(n_sites=1, creep="aquitard")
    assert d.aquitard_bounds == AQUITARD_BOUNDS
    assert float(torch.exp(d.log_tau_a.detach())) == pytest.approx(3652.5, rel=1e-6)


# ---- per-zone creep and floors in calibrate_coupled ---------------------------------------
def test_zone_creep_selects_zones_and_their_changhua_copies():
    three = ("proximal", "mid", "distal")
    assert zone_creep("aquitard", "all", three) == ["aquitard"] * 3
    assert zone_creep("aquitard", "distal", three) == ["vep", "vep", "aquitard"]
    assert zone_creep("aquitard", "mid+distal", three) == ["vep", "aquitard", "aquitard"]
    six = three + tuple(f"{z}_changhua" for z in three)
    assert zone_creep("aquitard", "distal", six) == ["vep", "vep", "aquitard"] * 2
    with pytest.raises(ValueError):
        zone_creep("aquitard", "coast", three)


def test_make_per_zone_creep_and_tau_floor(e1_globals):
    cc.CREEP, cc.CREEP_ZONES = "aquitard", "distal"
    cc.AQUITARD_TAU_MIN_YEARS, cc.AQUITARD_HA0_MIN = 10.0, 0.0
    cc.TAU_MIN_DAYS_ZONE = {"distal": 3650.0}
    model = cc._make("zonal", "cpu", 3)
    assert [c.creep for c in model] == ["vep", "vep", "aquitard"]
    assert [getattr(c, "tau_min_days", None) for c in model] == [None, None, 3650.0]
    assert float(torch.exp(model[2].log_tau.detach())) == pytest.approx(3650.0, rel=1e-5)
    assert model[2].aquitard_bounds == aquitard_bounds(10.0, 0.0)
    js = cc.column_json(model)
    assert [set(z) for z in js["zonal"]] == [set(VEP_KEYS)] * 2 + [set(VEP_KEYS + AQUITARD_KEYS)]
    six = cc._make("zonal", "cpu", 6)
    assert [c.creep for c in six] == ["vep", "vep", "aquitard"] * 2
    assert [getattr(c, "tau_min_days", None) for c in six] == [None, None, 3650.0] * 2
    for cfg in ("shared", "weighted", "banded"):
        with pytest.raises(ValueError):
            cc._make(cfg, "cpu")
    cc.CREEP = "vep"
    with pytest.raises(ValueError, match="needs --creep aquitard"):
        cc._make("zonal", "cpu", 3)


def test_clamp_enforces_every_floor(e1_globals):
    cc.CREEP, cc.CREEP_ZONES = "aquitard", "mid+distal"
    cc.AQUITARD_TAU_MIN_YEARS, cc.AQUITARD_HA0_MIN = 10.0, 0.0
    cc.TAU_MIN_DAYS_ZONE = {"distal": 3650.0}
    model = cc._make("zonal", "cpu", 3)
    T = 400                                               # ceiling 12000 d
    for c in model:
        with torch.no_grad():
            c.log_tau.fill_(math.log(30.0))
            if c.creep == "aquitard":
                c.log_tau_a.fill_(math.log(200.0))
                c.h_a0.fill_(-12.0)
                c.log_ska.fill_(5.0)
        cc._clamp_column(c, T)
    taus = [float(torch.exp(c.log_tau)) for c in model]
    assert taus[0] == pytest.approx(30.0, rel=1e-5) and taus[1] == pytest.approx(30.0, rel=1e-5)
    assert taus[2] == pytest.approx(3650.0, rel=1e-5)
    for c in model[1:]:
        assert float(torch.exp(c.log_tau_a)) == pytest.approx(10 * YR, rel=1e-5)
        assert float(c.h_a0) == 0.0
        assert float(c.log_ska) == pytest.approx(AQUITARD_BOUNDS["log_ska"][1], abs=1e-6)
    # a zone floor above the tau ceiling is refused
    cc.TAU_MIN_DAYS_ZONE = {"distal": 20000.0}
    with pytest.raises(ValueError, match="tau-min-days-zone"):
        cc._clamp_column(cc._make("zonal", "cpu", 3)[2], T)


def test_fit_keeps_floors_and_records_them(e1_globals):
    cc.CREEP, cc.CREEP_ZONES = "aquitard", "distal"
    cc.AQUITARD_TAU_MIN_YEARS, cc.AQUITARD_HA0_MIN = 10.0, 0.0
    cc.TAU_MIN_DAYS_ZONE = {"distal": 3650.0}
    H, obs, M, Z = _synthetic_fit(T=140)
    model = cc._make("zonal", "cpu", 3)
    cc._fit(model, H, obs, M, Z, 40, 0.05, rezero=True)
    p = _all_params(model)
    assert math.exp(p[2]["log_tau"]) >= 3650.0 * (1 - 1e-5)
    assert math.exp(p[2]["log_tau_a"]) >= 10 * YR * (1 - 1e-5) and p[2]["h_a0"] >= 0.0
    rec = cc.column_constraints()
    assert rec["creep_zones"] == "distal"
    assert rec["aquitard_tau_min_years"] == 10.0 and rec["aquitard_ha0_min"] == 0.0
    assert rec["aquitard_bounds"]["h_a0"] == [0.0, 30.0]
    assert rec["tau_min_days_zone"] == {"distal": 3650.0}
    js = {**cc.column_json(model), "config": "zonal", "creep": "aquitard", **rec}
    check_column_creep(json.loads(json.dumps(js)))           # what forward accepts


def test_parse_tau_min_days_zone():
    assert cc.parse_tau_min_days_zone(None) is None
    assert cc.parse_tau_min_days_zone(["distal=3650", "mid_changhua=400"]) == {
        "distal": 3650.0, "mid_changhua": 400.0}
    for bad in (["coast=10"], ["distal"], ["distal=0.5"], ["distal=10", "distal=20"]):
        with pytest.raises(ValueError):
            cc.parse_tau_min_days_zone(bad)


# ---- default path unchanged -----------------------------------------------------------------
def test_defaults_are_the_historical_column(e1_globals):
    assert cc.column_constraints() == {"tau_min_days": None, "ske_min": None,
                                       "ske_skv_max": None}
    for creep in ("vep", "aquitard"):
        cc.CREEP = creep
        model = cc._make("zonal", "cpu", 3)
        assert [c.creep for c in model] == [creep] * 3
        assert all(not hasattr(c, "tau_min_days") for c in model)
        assert all(c.aquitard_bounds == AQUITARD_BOUNDS for c in model)
        assert cc.column_aquitard_bounds() is None


@pytest.mark.parametrize("creep", ["vep", "aquitard"])
def test_non_binding_flags_fit_bit_for_bit(e1_globals, creep):
    """Floors set at the default bounds leave every clamp where it was: identical fits."""
    cc.CREEP = creep
    H, obs, M, Z = _synthetic_fit()
    a = cc._make("zonal", "cpu", 3)
    cc._fit(a, H, obs, M, Z, 60, 0.05, rezero=True)
    cc.TAU_MIN_DAYS_ZONE = {"distal": 1.0}
    if creep == "aquitard":
        cc.AQUITARD_TAU_MIN_YEARS, cc.AQUITARD_HA0_MIN = 1.0, -30.0
    b = cc._make("zonal", "cpu", 3)
    cc._fit(b, H, obs, M, Z, 60, 0.05, rezero=True)
    assert _all_params(a) == _all_params(b)


# ---- forward: rebuild and pairing ----------------------------------------------------------
def _mixed_json():
    vep = {"log_ske": math.log(1e-3), "log_skv": math.log(2e-2), "log_tau": math.log(400.0),
           "h_pc0": 0.0}
    aq = {**vep, "log_tau": math.log(3650.0), "log_ska": math.log(2e-2),
          "log_tau_a": math.log(20 * YR), "h_a0": 4.0}
    return {"zonal": [dict(vep), dict(vep), aq], "config": "zonal", "creep": "aquitard",
            "creep_zones": "distal", "tau_min_days_zone": {"distal": 3650.0},
            "aquitard_bounds": {k: list(v) for k, v in aquitard_bounds(10.0, 0.0).items()},
            "hpc0_guard_days": None}


def test_forward_rebuilds_a_mixed_per_zone_column(tmp_path):
    p = _mixed_json()
    path = tmp_path / "vep_zonal_leveling.json"
    path.write_text(json.dumps(p))
    zone = np.array([0, 1, 2, 2])
    col, got = load_or_fit_vep(str(path), None, None, "cpu", zone_of_cell=zone)
    assert [c.creep for c in col.cols] == ["vep", "vep", "aquitard"]
    g = torch.Generator().manual_seed(1)
    h = -torch.cumsum(torch.rand(4, 80, generator=g, dtype=torch.float64) * 0.3, dim=-1)
    with torch.no_grad():
        out = col.double()(h)
    ref = torch.stack([column_compaction(h[i:i + 1], p["zonal"][z])[0]
                       for i, z in enumerate(zone)])
    assert torch.allclose(out, ref, atol=1e-12)
    # the distal cells carry the slow term; the VEP cells do not
    assert not torch.allclose(ref[2], column_compaction(h[2:3], {
        k: p["zonal"][2][k] for k in VEP_KEYS})[0])


def test_forward_refuses_a_column_its_rebuild_would_change(tmp_path):
    good = _mixed_json()
    check_column_creep(good)
    cases = []
    c = _mixed_json()
    c["creep_zones"] = "all"                 # says every zone, carries distal only
    cases.append(c)
    c = _mixed_json()
    c["creep_zones"] = "mid+distal"
    cases.append(c)
    c = _mixed_json()
    c["creep"] = "vep"
    cases.append(c)
    c = _mixed_json()
    del c["zonal"][2]["h_a0"]                # partial keys: the rebuild would drop the term
    cases.append(c)
    c = _mixed_json()
    c["zonal"][2]["log_tau"] = math.log(400.0)    # below its recorded per-zone floor
    cases.append(c)
    c = _mixed_json()
    c["zonal"][2]["h_a0"] = -2.0                  # below the recorded h_a0 floor
    cases.append(c)
    c = _mixed_json()
    c["zone_blend_km"] = 4.0                      # blend would drop the term
    cases.append(c)
    c = _mixed_json()
    c["creep_zones"] = "coast"
    cases.append(c)
    for i, c in enumerate(cases):
        with pytest.raises(ValueError):
            check_column_creep(c, label=f"case {i}")
        path = tmp_path / f"bad_{i}.json"
        path.write_text(json.dumps(c))
        with pytest.raises(ValueError):
            load_or_fit_vep(str(path), None, None, "cpu", zone_of_cell=np.array([0, 1, 2]),
                            zone_weights=lambda km: np.eye(3))


def test_historical_columns_pass_the_creep_check():
    vep = {"log_ske": -7.0, "log_skv": -4.0, "log_tau": 6.0, "h_pc0": 0.0}
    check_column_creep(dict(vep))                                       # Stage-2 shared
    check_column_creep({"zonal": [dict(vep)] * 3, "creep": "vep", "zone_blend_km": 4.0})
    aq = {**vep, "log_ska": -4.0, "log_tau_a": 8.0, "h_a0": -3.0}
    check_column_creep({"zonal": [dict(aq)] * 3, "creep": "aquitard"})
    check_column_creep({"zonal": [dict(aq)] * 6, "creep": "aquitard",
                        "zone_county_split": True})


def test_blend_refuses_the_aquitard_term():
    cols = [VEPColumn(n_sites=1), VEPColumn(n_sites=1), VEPColumn(n_sites=1, creep="aquitard")]
    w = torch.eye(3)
    h = torch.zeros(3, 10)
    with pytest.raises(ValueError):
        blended_column(cols, w, h)
    blended_column(cols[:2] + [VEPColumn(n_sites=1)], w, h)           # VEP-only still works
    p = _mixed_json()["zonal"]
    with pytest.raises(ValueError):
        ZonalColumn(p, np.array([0, 1, 2]), zone_w=np.eye(3))


# ---- calibrate_coupled.main refusals (before any data is read) -----------------------------
@pytest.mark.parametrize("extra", [
    ["--creep-zones", "distal"],                                   # needs --creep aquitard
    ["--aquitard-tau-min-years", "10"],                            # needs --creep aquitard
    ["--creep", "aquitard", "--aquitard-tau-min-years", "0.5"],    # below the default floor
    ["--creep", "aquitard", "--aquitard-ha0-min", "40"],
    ["--creep", "aquitard", "--creep-zones", "distal", "--configs", "shared,zonal"],
    ["--tau-min-days-zone", "distal=3650", "--configs", "shared"],
    ["--tau-min-days-zone", "coast=3650", "--configs", "zonal"],
    ["--tau-min-days-zone", "distal=99999", "--configs", "zonal"],  # above the ceiling
    ["--creep", "aquitard", "--configs", "zonal", "--zone-blend-km", "4"],
])
def test_main_refuses_inconsistent_flags(e1_globals, tmp_path, extra):
    with pytest.raises(SystemExit):
        cc.main(["--theta", str(tmp_path / "missing.json"), "--out", str(tmp_path / "o"),
                 *extra])


def test_main_accepts_consistent_flags_up_to_the_data(e1_globals, tmp_path):
    """The refusals above are the flags' own: a consistent set gets past them and only
    stops at the (missing) theta."""
    with pytest.raises(FileNotFoundError):
        cc.main(["--theta", str(tmp_path / "missing.json"), "--out", str(tmp_path / "o"),
                 "--configs", "zonal", "--creep", "aquitard", "--creep-zones", "mid+distal",
                 "--aquitard-tau-min-years", "10", "--aquitard-ha0-min", "0",
                 "--tau-min-days-zone", "distal=3650", "--tau-max-years", "30"])
