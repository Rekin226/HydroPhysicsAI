"""M1 (2026-10-24): calibrate_coupled --ske-cap-from-theta and --fold-site-tol-m.

Both are opt-in. Off, they must reproduce the historical clamp and the historical
leveling folds exactly; on, the cap is KAPPA times the flow theta's storativity summed
over layers per column zone, the column JSON says so, and twin.forward still loads it.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")

import hydrophysics.twin.calibrate_coupled as cc  # noqa: E402
from hydrophysics.twin import zone_county  # noqa: E402
from hydrophysics.twin.compaction import VEPColumn  # noqa: E402
from hydrophysics.twin.forward import check_supported_column, load_or_fit_vep  # noqa: E402


@pytest.fixture
def cap_globals(monkeypatch):
    for k in ("TAU_MIN_DAYS", "SKE_MIN", "SKE_SKV_MAX", "TAU_MAX_YEARS", "SKE_CAP_KAPPA",
              "SKE_CAPS", "SKE_CAP_ZONES", "HPC0_GUARD_DAYS"):
        monkeypatch.setattr(cc, k, None)
    return cc


def _theta():
    """A zonal theta with the proximal split: one proximal S per part (every layer),
    four per layer elsewhere."""
    return {"log_S_proximal": [math.log(2e-4)], "log_S_proximal_w": [math.log(5e-3)],
            "log_S_mid": [math.log(v) for v in (5e-3, 1e-4, 5e-4, 3e-3)],
            "log_S_distal": [math.log(v) for v in (1e-3, 2e-4, 2e-4, 5e-4)],
            "log_T_mid": [1.0] * 4, "log_Sd_mid": [math.log(9.0)]}


def _synthetic_fit(seed: int = 3, n: int = 12, T: int = 40):
    g = torch.Generator().manual_seed(seed)
    H = -torch.cumsum(torch.rand(n, 4, T, generator=g) * 0.4, dim=-1)
    obs = 0.02 * (H[:, :, 0:1] - H).mean(dim=1)
    M = torch.ones(n, T)
    Z = torch.tensor([i % 3 for i in range(n)], dtype=torch.long)
    return H, obs, M, Z


def _params(model) -> list[list[float]]:
    return [[float(getattr(c, k).detach()) for k in ("log_ske", "log_skv", "log_tau", "h_pc0")]
            for c in model]


# ---- cap off ----------------------------------------------------------------------------
def test_cap_off_is_the_historical_clamp(cap_globals):
    c = VEPColumn(n_sites=1)
    with torch.no_grad():
        c.log_ske.fill_(math.log(0.5))                   # above SKE_MAX
    cc._clamp_column(c, 132)
    assert float(c.log_ske.detach()) == pytest.approx(math.log(cc.SKE_MAX))
    model = cc._make("zonal", "cpu", 3)
    assert all(not hasattr(col, "ske_cap") for col in model)
    assert all(float(col.log_ske.detach()) == pytest.approx(math.log(1e-3)) for col in model)
    assert cc.column_constraints() == {"tau_min_days": None, "ske_min": None,
                                       "ske_skv_max": None}


def test_cap_off_and_a_non_binding_cap_fit_bit_for_bit(cap_globals):
    """A cap above SKE_MAX leaves every bound where it was: the fits are identical."""
    H, obs, M, Z = _synthetic_fit()
    torch.manual_seed(0)
    a = cc._make("zonal", "cpu", 3)
    cc._fit(a, H, obs, M, Z, 60, 0.05, rezero=True)
    cc.SKE_CAPS = [10.0, 10.0, 10.0]
    b = cc._make("zonal", "cpu", 3)
    cc._fit(b, H, obs, M, Z, 60, 0.05, rezero=True)
    assert _params(a) == _params(b)


# ---- the cap ----------------------------------------------------------------------------
def test_flow_zone_storage_sums_layers_and_expands_proximal():
    s = cc.flow_zone_storage(_theta(), {})
    assert s["proximal"] == pytest.approx(4 * 2e-4)
    assert s["proximal_w"] == pytest.approx(4 * 5e-3)
    assert s["mid"] == pytest.approx(5e-3 + 1e-4 + 5e-4 + 3e-3)
    assert s["distal"] == pytest.approx(1e-3 + 2e-4 + 2e-4 + 5e-4)
    assert set(s) == {"proximal", "proximal_w", "mid", "distal"}     # log_Sd not storage
    assert cc.flow_zone_storage({"log_S": [math.log(1e-3)] * 4}) == {
        "": pytest.approx(4e-3)}


def test_caps_are_kappa_times_area_weighted_storage():
    names = ("proximal", "mid", "distal", "proximal_w")
    flow = np.array([0, 0, 0, 3, 1, 1, 2, 2, 2, 2])                 # 3 proximal, 1 proximal_w
    col = np.where(flow == 3, 0, flow)
    s = cc.flow_zone_storage(_theta(), {})
    kappa = 0.7
    caps = cc.ske_caps(s, names, np.eye(4)[flow], col, 3, kappa)
    assert caps[0] == pytest.approx(kappa * (3 * s["proximal"] + 1 * s["proximal_w"]) / 4)
    assert caps[1] == pytest.approx(kappa * s["mid"])
    assert caps[2] == pytest.approx(kappa * s["distal"])
    shared = cc.ske_caps(s, names, np.eye(4)[flow], np.zeros_like(col), 1, kappa)
    assert shared[0] == pytest.approx(kappa * np.mean([s[names[z]] for z in flow]))
    with pytest.raises(ValueError):
        cc.ske_caps(s, names, np.eye(4)[flow], col, 3, 0.0)


def test_theta_ske_caps_reads_the_flow_zone_map(monkeypatch):
    class G:
        n_active = 6

    names = ("proximal", "mid", "distal", "proximal_w")
    flow = np.array([0, 3, 3, 1, 2, 2])
    monkeypatch.setattr(zone_county, "flow_zone_map", lambda grid, meta: (flow, names))
    col = np.array([0, 0, 0, 1, 2, 2])
    caps = cc.theta_ske_caps(_theta(), {"param_mode": "zonal"}, G(), col, 3, 1.0)
    s = cc.flow_zone_storage(_theta(), {})
    assert caps == pytest.approx([(s["proximal"] + 2 * s["proximal_w"]) / 3, s["mid"],
                                  s["distal"]])


def test_capped_columns_never_exceed_kappa_times_storage(cap_globals):
    caps = [2e-4, 3e-3, 1e-3]
    cc.SKE_CAPS = caps
    model = cc._make("zonal", "cpu", 3)
    assert [col.ske_cap for col in model] == caps
    # each column starts at or below its cap, and the clamp holds it there
    for col, cap in zip(model, caps, strict=True):
        assert float(col.log_ske.detach()) <= math.log(cap) + 1e-6       # float32
        with torch.no_grad():
            col.log_ske.fill_(math.log(0.05))
        cc._clamp_column(col, 132)
        assert math.exp(float(col.log_ske.detach())) == pytest.approx(cap, rel=1e-6)
    # a fit that wants a large Ske (synthetic data with Ske = 2e-2) stays on the caps
    H, obs, M, Z = _synthetic_fit()
    m2 = cc._make("zonal", "cpu", 3)
    cc._fit(m2, H, obs, M, Z, 80, 0.05, rezero=True)
    for col, cap in zip(m2, caps, strict=True):
        assert math.exp(float(col.log_ske.detach())) <= cap * (1 + 1e-5)
    # shared and weighted take one cap
    cc.SKE_CAPS = [1e-3]
    assert cc._make("shared", "cpu").ske_cap == 1e-3
    assert cc._make("weighted", "cpu").col.ske_cap == 1e-3
    with pytest.raises(ValueError):
        cc._make("zonal", "cpu", 3)                       # one cap for three columns
    with pytest.raises(ValueError):
        cc._make("banded", "cpu")


def test_cap_below_the_ske_floor_is_refused(cap_globals):
    cc.SKE_CAPS = [1e-7]
    col = cc._make("shared", "cpu")
    with pytest.raises(ValueError, match="floor"):
        cc._clamp_column(col, 132)


# ---- the record, and twin.forward ---------------------------------------------------------
def test_capped_column_records_its_cap_and_forward_loads_it(cap_globals, tmp_path):
    cc.SKE_CAP_KAPPA, cc.SKE_CAPS = 1.0, [2e-4, 3e-3, 1e-3]
    cc.SKE_CAP_ZONES = ["proximal", "mid", "distal"]
    rec = cc.column_constraints()
    assert rec["ske_cap_kappa"] == 1.0
    assert rec["ske_cap"] == [2e-4, 3e-3, 1e-3]
    assert rec["ske_cap_zones"] == ["proximal", "mid", "distal"]
    assert "ske_cap_source" in rec
    model = cc._make("zonal", "cpu", 3)
    p = {**cc.column_json(model), "config": "zonal", "creep": "vep", **rec,
         **cc.fold_site_tol_record(150.0)}
    check_supported_column(p)
    path = tmp_path / "vep_zonal_leveling.json"
    path.write_text(json.dumps(p))
    zone = np.array([0, 1, 2, 2, 1])
    col, back = load_or_fit_vep(str(path), None, None, "cpu", zone_of_cell=zone)
    assert back["ske_cap"] == [2e-4, 3e-3, 1e-3] and back["fold_site_tol_m"] == 150.0
    h = -torch.linspace(0, 3, 12, dtype=torch.float64).repeat(5, 1)
    with torch.no_grad():
        out = col(h)
    assert out.shape == (5, 12) and torch.isfinite(out).all()


# ---- --fold-site-tol-m --------------------------------------------------------------------
def test_tolerance_zero_is_the_historical_partition():
    assert cc.leveling_fold_groups(np.zeros((7, 2)), 0.0) is None
    assert cc.fold_site_tol_record(0.0) == {}
    for seed in (0, 7):
        order = np.random.default_rng(seed).permutation(23)
        want = np.array_split(order, 5)
        got = cc.site_folds(23, 5, seed, None)
        assert len(got) == len(want)
        assert all(np.array_equal(a, b) for a, b in zip(got, want, strict=True))


def test_kfold_sites_default_folds_unchanged(monkeypatch):
    held = []
    monkeypatch.setattr(cc, "_make", lambda config, device, n_zones: object())
    monkeypatch.setattr(cc, "_fit", lambda *a, **k: 0.0)

    def predict(model, H, Z):
        held.append(sorted(int(i) for i in H[:, 0, 0].tolist()))
        return torch.zeros(H.shape[0], 3)

    monkeypatch.setattr(cc, "_predict", predict)
    monkeypatch.setattr(cc, "_r2", lambda p, o, m: 0.0)
    n = 23
    H = torch.arange(n, dtype=torch.float32).reshape(n, 1, 1).expand(n, 1, 3).clone()
    OBS, M, Z = torch.zeros(n, 3), torch.ones(n, 3), torch.zeros(n, dtype=torch.int64)
    cc.kfold_sites("zonal", H, OBS, M, Z, 1, 0.1, "cpu", n_folds=5, seed=0, rezero=False)
    order = np.random.default_rng(0).permutation(n)
    assert held == [sorted(int(i) for i in f) for f in np.array_split(order, 5)]
    # with groups: every group held out together
    held.clear()
    groups = np.repeat(np.arange(8), 3)[:n]
    cc.kfold_sites("zonal", H, OBS, M, Z, 1, 0.1, "cpu", n_folds=5, seed=0, rezero=False,
                   groups=groups)
    assert sorted(i for f in held for i in f) == list(range(n))
    fold_of = {i: k for k, f in enumerate(held) for i in f}
    assert all(fold_of[i] == fold_of[j] for i in range(n) for j in range(n)
               if groups[i] == groups[j])


@pytest.mark.parametrize("tol", [30.0, 250.0, 1200.0])
def test_positive_tolerance_never_splits_sites_within_it(tol):
    rng = np.random.default_rng(11)
    centres = rng.uniform(0, 20_000, size=(40, 2))
    xy = np.concatenate([centres + rng.normal(0, tol / 3, size=centres.shape)
                         for _ in range(3)])                # nests of three benchmarks
    groups = cc.leveling_fold_groups(xy, tol)
    folds = cc.site_folds(len(xy), 5, 0, groups)
    fold_of = np.empty(len(xy), dtype=int)
    for k, f in enumerate(folds):
        fold_of[f] = k
    assert sorted(np.concatenate(folds).tolist()) == list(range(len(xy)))
    d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    i, j = np.nonzero(d <= tol)
    assert np.all(fold_of[i] == fold_of[j])


def test_cli_refuses_inconsistent_options(cap_globals):
    with pytest.raises(SystemExit, match="leveling"):
        cc.main(["--theta", "x.json", "--out", "o", "--fold-site-tol-m", "100"])
    with pytest.raises(SystemExit, match=">= 0"):
        cc.main(["--theta", "x.json", "--out", "o", "--target", "leveling",
                 "--fold-site-tol-m", "-1"])
    with pytest.raises(SystemExit, match="banded"):
        cc.main(["--theta", "x.json", "--out", "o", "--configs", "banded",
                 "--ske-cap-from-theta", "1"])
    with pytest.raises(SystemExit, match="> 0"):
        cc.main(["--theta", "x.json", "--out", "o", "--ske-cap-from-theta", "0"])
