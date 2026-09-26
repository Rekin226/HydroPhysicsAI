"""--apex-hold (2026-09-26): a rollout started from a restarted field keeps the calibrated
apex boundary head when asked to, and re-pins it to its own starting field otherwise
(the behaviour before, which every default run keeps)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("matplotlib")
pytest.importorskip("pyproj")

from test_twin_forward import _inputs, _theta_file  # noqa: E402

from hydrophysics.twin.forward import build_model, load_members, rollout  # noqa: E402


def _setup(tmp_path):
    inp = _inputs()
    p, _ = _theta_file(tmp_path)
    model, scalars, _ = build_model(inp.grid, load_members([p])[0], "cpu")
    h0 = torch.as_tensor(inp.initial_heads(0, n_layers=4), dtype=torch.float64)
    E = torch.zeros((inp.grid.n_active, 3), dtype=torch.float64)
    R = torch.zeros_like(E)
    return inp, model, scalars, h0, E, R


def test_default_rollout_repins_the_apex_to_its_own_start(tmp_path):
    inp, model, scalars, h0, E, R = _setup(tmp_path)
    h_restart = h0 + 7.0
    rollout(model, scalars, h_restart, E, R, inp.ground_elev)
    assert torch.allclose(model.apex_h, h_restart[:, model.apex_idx])


def test_apex_from_keeps_the_calibrated_boundary(tmp_path):
    inp, model, scalars, h0, E, R = _setup(tmp_path)
    h_restart = h0 + 7.0
    rollout(model, scalars, h_restart, E, R, inp.ground_elev, apex_from=h0)
    assert torch.allclose(model.apex_h, h0[:, model.apex_idx])
    # and the boundary it holds changes the solution relative to the re-pinned run
    held = rollout(model, scalars, h_restart, E, R, inp.ground_elev, apex_from=h0)
    repinned = rollout(model, scalars, h_restart, E, R, inp.ground_elev)
    assert not torch.allclose(held, repinned)
