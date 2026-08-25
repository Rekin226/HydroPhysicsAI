import numpy as np
import pandas as pd
import pytest
import torch

from hydrophysics.twin.grid import FanGrid
from hydrophysics.twin.pumping import aggregate_pumps, energy_to_volume


def _grid():
    return FanGrid(nx=4, ny=4, dx=1000.0, x0=0.0, y0=0.0,
                   mask=np.ones((4, 4), dtype=bool))


def test_energy_to_volume_follows_the_hydraulic_relation():
    """Q = eta * E / (rho g h). Doubling lift halves volume; doubling energy doubles it."""
    E = torch.tensor([[1000.0, 1000.0]])          # kWh in the period
    lift = torch.tensor([[10.0, 20.0]])
    q = energy_to_volume(E, lift, torch.zeros(1))
    assert float(q[0, 1]) == pytest.approx(float(q[0, 0]) / 2.0, rel=1e-6)

    q2 = energy_to_volume(E * 2, lift, torch.zeros(1))
    assert float(q2[0, 0]) == pytest.approx(2.0 * float(q[0, 0]), rel=1e-6)


def test_energy_to_volume_is_bounded_as_lift_goes_to_zero():
    E = torch.tensor([[1000.0]])
    q_small = energy_to_volume(E, torch.tensor([[1e-6]]), torch.zeros(1))
    q_ref = energy_to_volume(E, torch.tensor([[2.0]]), torch.zeros(1))
    assert torch.isfinite(q_small).all()
    assert float(q_small[0, 0]) <= 2.0 * float(q_ref[0, 0])


def test_aggregate_pumps_sums_into_the_right_cells():
    g = _grid()
    pumps = pd.DataFrame({"sid": ["a", "b", "c"],
                          "TWD97_X": [500.0, 1500.0, 500.0],
                          "TWD97_Y": [500.0, 500.0, 500.0],
                          "PUMP_HP": [5.0, 5.0, 5.0],
                          "PURPOSE": ["irrigation"] * 3})
    months = pd.date_range("2015-01-01", periods=2, freq="MS")
    kwh = pd.DataFrame({"pump": ["a", "a", "b", "b", "c", "c"],
                        "datetime": list(months) * 3,
                        "electricity_kwh": [10.0, 20.0, 100.0, 200.0, 1.0, 2.0]})
    E, dates = aggregate_pumps(pumps, kwh, g, "2015-01-01", "2015-03-01")
    assert E.shape == (g.n_active, 2)
    i_a = g.active_index(500.0, 500.0)
    i_b = g.active_index(1500.0, 500.0)
    assert E[i_a, 0] == pytest.approx(11.0)        # pumps a and c share a cell
    assert E[i_b, 1] == pytest.approx(200.0)
    assert E.sum() == pytest.approx(333.0)


def test_pumps_outside_the_grid_are_dropped_not_snapped():
    g = _grid()
    pumps = pd.DataFrame({"sid": ["far"], "TWD97_X": [999999.0], "TWD97_Y": [999999.0],
                          "PUMP_HP": [5.0], "PURPOSE": ["irrigation"]})
    kwh = pd.DataFrame({"pump": ["far"], "datetime": [pd.Timestamp("2015-01-01")],
                        "electricity_kwh": [50.0]})
    E, _ = aggregate_pumps(pumps, kwh, g, "2015-01-01", "2015-02-01")
    assert E.sum() == pytest.approx(0.0)
