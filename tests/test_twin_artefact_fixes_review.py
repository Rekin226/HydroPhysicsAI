"""Adversarial review of the 2026-09-23 artefact fixes (A1/A2/A3).

A forward run whose column was refitted with ``calibrate_coupled --hpc0-guard-days``
already has no start-up load (the guard keeps the fast column's ``h_pc0 <= 0``), so
``--column-hpc0-fast-days`` releases nothing and records ``hpc0_released = [None]``. The
app must still recognise the run as fixed upstream; otherwise it re-applies
``artefact_steps`` / ``restart_bridge`` to a field that no longer has the steps, and
subtracts real subsidence (the planned ``forward_blend2_spreadL`` -> app path).
"""
from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("matplotlib")
pytest.importorskip("pyproj")

from hydrophysics.twin import viewer_app as va  # noqa: E402
from hydrophysics.twin.forward import load_or_fit_vep  # noqa: E402


def _guarded_column(tmp_path) -> str:
    # the shape calibrate_coupled writes after --hpc0-guard-days 365: the fast proximal
    # column's offset sits on the guard (0), the slow ones keep theirs
    p = {"zonal": [{"log_ske": -8.08, "log_skv": -1.36, "log_tau": 3.19, "h_pc0": 0.0},
                   {"log_ske": -3.67, "log_skv": -2.34, "log_tau": 8.28, "h_pc0": 2.6},
                   {"log_ske": -4.10, "log_skv": -3.50, "log_tau": 6.71, "h_pc0": -5.4}],
         "config": "zonal", "hpc0_guard_days": 365.0}
    path = tmp_path / "vep_zonal_leveling.json"
    path.write_text(json.dumps(p))
    return str(path)


def _forward_options_like_main(params: dict, column_heads: str, fast_days) -> str:
    # mirrors forward.main's forward_options for one column
    return json.dumps({"restart_taper_km": 5.0, "column_heads": column_heads,
                       "column_hpc0_fast_days": fast_days,
                       "hpc0_released": [params.get("hpc0_released")],
                       "column_zone_blend_km": [0.0], "save_members": "yearly"})


class _FW:
    def __init__(self, d):
        self._d = d
        self.files = list(d)

    def __getitem__(self, k):
        return self._d[k]


@pytest.mark.parametrize("fast_days", [365.0, None])
def test_guarded_refit_run_is_recognised_as_fixed_upstream(tmp_path, fast_days):
    zoc = np.zeros(3, dtype="int64")
    _, params = load_or_fit_vep(_guarded_column(tmp_path), None, None, "cpu",
                                zone_of_cell=zoc, hpc0_fast_days=fast_days)
    # A1 is already fixed in the column itself: no fast column carries a positive offset
    assert all(not (np.exp(c["log_tau"]) < 365.0 and c["h_pc0"] > 0) for c in params["zonal"])
    fw = _FW({"forward_options": np.array(_forward_options_like_main(params, "free",
                                                                      fast_days))})
    assert va._artefacts_fixed_upstream(fw), (
        "a free-column run on a guarded (A1-clean) column is reported as NOT fixed, so the "
        "app re-applies artefact_steps/restart_bridge and removes real subsidence")


class _Reached(Exception):
    pass


def test_policy_gate_loads_a_blended_column(tmp_path, monkeypatch):
    """The plan reruns the policy gate on the refitted ``--zone-blend-km 2`` column
    (``coupled_leveling_blend2``). ``policy_gate.policy_response`` calls
    ``load_or_fit_vep`` without ``zone_weights``, which raises for any blended column."""
    from types import SimpleNamespace

    from hydrophysics.twin import policy_gate as pg

    with open(_guarded_column(tmp_path)) as fh:
        p = json.load(fh)
    p["zone_blend_km"] = 2.0
    path = tmp_path / "vep_blend2.json"
    path.write_text(json.dumps(p))
    zoc = np.array([0, 1, 1, 2], dtype="int64")
    monkeypatch.setattr(pg, "build_model", lambda grid, member, device: (None, {}, zoc))

    def _stop(*a, **k):
        raise _Reached

    monkeypatch.setattr(pg, "sw_hist", _stop)
    inp = SimpleNamespace(grid=SimpleNamespace(centroids=lambda: np.array(
        [[210e3, 0.0], [190e3, 0.0], [182.5e3, 0.0], [170e3, 0.0]])), hf=None,
        E_total=np.zeros((4, 3)))
    member = SimpleNamespace(meta={"zone_boundaries": "205,182"}, theta={})
    with pytest.raises(_Reached):
        pg.policy_response(inp, member, vep_json=str(path), device="cpu", log=lambda *_: None)
