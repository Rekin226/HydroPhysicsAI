"""--creep aquitard: the column JSON carries the aquitard keys and every loader honours
them (calibrate_coupled.column_json, forward.column_from_params / ZonalColumn,
compaction.column_compaction agree)."""
from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hydrophysics.twin import calibrate_coupled as cc  # noqa: E402
from hydrophysics.twin.compaction import (  # noqa: E402
    AQUITARD_BOUNDS,
    AQUITARD_KEYS,
    VEP_KEYS,
    VEPColumn,
    column_compaction,
)
from hydrophysics.twin.forward import ZonalColumn, column_from_params  # noqa: E402


def _heads(n=3, T=60, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    h = 5 - 0.02 * t[None] + 1.5 * np.sin(2 * np.pi * t[None] / 12) + rng.normal(0, .2, (n, T))
    return torch.tensor(h, dtype=torch.float64)


def test_make_zonal_with_creep_writes_and_reads_aquitard_keys(monkeypatch):
    monkeypatch.setattr(cc, "CREEP", "aquitard")
    model = cc._make("zonal", None)
    assert all(c.creep == "aquitard" for c in model)
    js = cc.column_json(model)
    assert set(js["zonal"][0]) == set(VEP_KEYS + AQUITARD_KEYS)
    h = _heads()
    zone = np.array([0, 1, 2])
    col = ZonalColumn(js["zonal"], zone)
    assert all(c.creep == "aquitard" for c in col.cols)
    with torch.no_grad():
        a = col(h.to(torch.float32))
        b = torch.stack([column_compaction(h[i:i + 1].to(torch.float32), js["zonal"][z])[0]
                         for i, z in enumerate(zone)])
    assert torch.allclose(a, b, atol=1e-6)


def test_vep_json_without_aquitard_keys_stays_vep():
    p = {"log_ske": math.log(1e-3), "log_skv": math.log(2e-2), "log_tau": math.log(365.0),
         "h_pc0": 0.0}
    c = column_from_params(p)
    assert c.creep == "vep" and c.param_keys() == VEP_KEYS


def test_clamp_keeps_aquitard_parameters_inside_their_bounds(monkeypatch):
    monkeypatch.setattr(cc, "CREEP", "aquitard")
    c = VEPColumn(n_sites=1, creep="aquitard")
    with torch.no_grad():
        c.log_ska.fill_(10.0)
        c.log_tau_a.fill_(-5.0)
        c.h_a0.fill_(99.0)
    cc._clamp_column(c, 132)
    for k, (lo, hi) in AQUITARD_BOUNDS.items():
        v = float(getattr(c, k))
        assert lo - 1e-5 <= v <= hi + 1e-5   # float32 parameters


def test_creep_is_refused_for_banded_and_weighted(monkeypatch):
    monkeypatch.setattr(cc, "CREEP", "aquitard")
    for cfg in ("weighted", "banded"):
        with pytest.raises(ValueError):
            cc._make(cfg, None)


def test_aquitard_term_gives_level_dependent_rate():
    # a +1 m sustained head rise slows the aquitard column's sinking; a VEP column's
    # step response is a rebound that then stays flat
    from hydrophysics.twin.mechanism import step_response, step_summary

    # heads fall for five years, then hold: after the step no new minimum is set, so a VEP
    # column can only rebound while the aquitard term keeps draining at a head-set rate
    drv = np.maximum(-0.05 * np.arange(120), -0.05 * 60)[None]
    vep = {"log_ske": math.log(1e-3), "log_skv": math.log(2e-2), "log_tau": math.log(100.0),
           "h_pc0": 0.0}
    aq = {**vep, "log_ska": math.log(1e-2), "log_tau_a": math.log(3652.5), "h_a0": 0.0}
    s_vep = step_summary(step_response(vep, drv, 72), 72)
    s_aq = step_summary(step_response(aq, drv, 72), 72)
    assert s_vep["rebound_first_year_m"] < 0
    assert abs(sum(s_vep["rate_change_m_per_yr"])) < 1e-3
    assert sum(s_aq["rate_change_m_per_yr"]) < -1e-4
