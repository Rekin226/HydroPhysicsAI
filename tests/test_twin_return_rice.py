"""Fix F2 (--return-rice-share): a paddy-weighted, per-cell irrigation return-flow logit.

Without the field the return flow stays the historical scalar; with it the logit is
``return_frac_logit + return_rice_b * intensity`` cell by cell, and the rollout returns a
larger share of the pumped volume where paddy is denser.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import hydrophysics.twin.calibrate_flow as cf  # noqa: E402


@pytest.fixture(autouse=True)
def _clear():
    cf.set_return_rice(None)
    yield
    cf.set_return_rice(None)


def test_without_the_field_the_logit_is_the_historical_scalar():
    theta = {"return_frac_logit": torch.tensor(0.3, dtype=torch.float64),
             "return_rice_b": torch.tensor(2.0, dtype=torch.float64)}
    assert cf.return_logit(theta) is theta["return_frac_logit"]
    assert cf.return_logit({}) is None


def test_with_the_field_the_logit_is_per_cell_and_differentiable():
    cf.set_return_rice(np.array([0.0, 0.5, 1.0]))
    a = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    lg = cf.return_logit({"return_frac_logit": a, "return_rice_b": b})
    assert torch.allclose(lg, torch.tensor([0.3, 1.3, 2.3], dtype=torch.float64))
    (cf.RETURN_FRAC_MAX * torch.sigmoid(lg)).sum().backward()
    assert a.grad is not None and b.grad is not None and float(b.grad) > 0


def test_a_theta_without_the_slope_ignores_the_field():
    cf.set_return_rice(np.array([0.0, 1.0]))
    a = torch.tensor(-0.2, dtype=torch.float64)
    assert cf.return_logit({"return_frac_logit": a}) is a
