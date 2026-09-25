"""The forward twin's rheology axis (G4, 2026-09-23): several compaction columns, each
driven by every flow member's heads, pooled into one subsidence spread."""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("matplotlib")
pytest.importorskip("pyproj")

from hydrophysics.twin.forward import (  # noqa: E402
    load_columns,
    load_members,
    parse_scenario,
    run,
)
from tests.test_twin_forward import _inputs, _theta_file  # noqa: E402


def _column(tmp_path, name, tau_days, skv, tau_max_years=None):
    d = tmp_path / name
    d.mkdir()
    p = d / "vep_shared_leveling.json"
    p.write_text(json.dumps({"log_ske": float(np.log(1e-3)), "log_skv": float(np.log(skv)),
                             "log_tau": float(np.log(tau_days)), "h_pc0": 0.0,
                             "tau_max_years": tau_max_years}))
    return str(p)


def test_load_columns_labels_each_rheology(tmp_path):
    a = _column(tmp_path, "coupled_leveling", 3960.0, 0.1)
    b = _column(tmp_path, "coupled_leveling_tau30", 30 * 365.25, 0.4, tau_max_years=30.0)
    cols = load_columns(f"{a}, {b}", None, None, "cpu")
    assert [c[0] for c in cols] == ["coupled_leveling", "tau30y"]
    same = load_columns(f"{a},{a}", None, None, "cpu")
    assert [c[0] for c in same] == ["coupled_leveling", "coupled_leveling_1"]


def test_run_pools_members_across_rheologies(tmp_path):
    inp = _inputs()
    p, f = _theta_file(tmp_path)
    members = load_members([p, f])
    scen = [(parse_scenario("base:irrigation=1")[0], 1.0),
            (parse_scenario("cut:irrigation=0.5")[0], 1.0)]
    a = _column(tmp_path, "short", 100.0, 0.05)
    b = _column(tmp_path, "long", 3000.0, 0.5, tau_max_years=30.0)
    cols = [(lab, c) for lab, c, _ in load_columns(f"{a},{b}", None, None, "cpu")]
    kw = dict(horizon=3, gain=1.0, ic_members=0, ic_sigma=0.0, seed=0, device="cpu",
              log=lambda *_: None)
    both = run(inp, members, scen, col=cols, **kw)
    one = [run(inp, members, scen, col=c, **kw) for _, c in cols]
    assert both["rheology_labels"] == ["short", "tau30y"]
    assert both["subs_mean_by_rheology"].shape == (2, 2, inp.grid.n_active,
                                                   len(inp.dates) + 3)
    # each rheology slice is exactly the single-column run; the pooled mean is their mean
    for r in range(2):
        assert np.allclose(both["subs_mean_by_rheology"][r], one[r]["subs_mean"], atol=1e-6)
    assert np.allclose(both["subs_mean"],
                       0.5 * (one[0]["subs_mean"] + one[1]["subs_mean"]), atol=1e-6)
    # heads do not depend on the rheology
    assert np.allclose(both["heads_mean"], one[0]["heads_mean"])
    # the pooled spread carries the between-rheology difference
    half_gap = 0.5 * np.abs(one[0]["subs_mean"] - one[1]["subs_mean"])
    assert (both["subs_std"] >= half_gap - 1e-5).all()
    rows = both["rows"]
    assert len(rows) == 2 * len(members) * 2 and set(rows["rheology"]) == {"short", "tau30y"}
    assert both["n_members"] == len(members)
