import pytest
import torch

from hydrophysics.twin.compaction import VEPColumn


def test_elastic_compaction_is_fully_recoverable():
    """Pure elastic loading then unloading must return to zero compaction."""
    col = VEPColumn(n_sites=1)
    with torch.no_grad():
        col.log_ske.fill_(torch.log(torch.tensor(1e-3)).item())
        col.log_skv.fill_(-30.0)     # inelastic off
        col.log_tau.fill_(-10.0)     # instantaneous
        col.h_pc0.fill_(-1e3)        # preconsolidation far below -> never gated on
    h = torch.tensor([[0.0, -5.0, -10.0, -5.0, 0.0]])
    s = col(h)
    assert s[0, 0].item() == pytest.approx(0.0, abs=1e-9)
    assert s[0, 2].item() > 0.0                       # head fell -> subsidence
    assert s[0, 4].item() == pytest.approx(0.0, abs=1e-6)   # head recovered -> recovered


def test_elastic_magnitude_follows_ske():
    col = VEPColumn(n_sites=2)
    with torch.no_grad():
        col.log_ske[0] = torch.log(torch.tensor(1e-3))
        col.log_ske[1] = torch.log(torch.tensor(2e-3))
        col.log_skv.fill_(-30.0)
        col.log_tau.fill_(-10.0)
        col.h_pc0.fill_(-1e3)
    h = torch.tensor([[0.0, -10.0], [0.0, -10.0]])
    s = col(h)
    assert s[1, 1].item() == pytest.approx(2.0 * s[0, 1].item(), rel=1e-5)


def test_gradients_flow_to_parameters():
    col = VEPColumn(n_sites=1)
    h = torch.tensor([[0.0, -3.0, -6.0]])
    col(h).sum().backward()
    assert col.log_ske.grad is not None
    assert torch.isfinite(col.log_ske.grad).all()


def test_inelastic_strain_is_never_recovered():
    """Below the preconsolidation head, compaction is permanent."""
    col = VEPColumn(n_sites=1)
    with torch.no_grad():
        col.log_ske.fill_(torch.log(torch.tensor(1e-4)).item())
        col.log_skv.fill_(torch.log(torch.tensor(1e-2)).item())
        col.log_tau.fill_(-10.0)      # instantaneous: isolate the gate
        col.h_pc0.fill_(0.0)          # preconsolidated at h = 0
    h = torch.tensor([[0.0, -10.0, 0.0]])
    s = col(h)
    assert s[0, 1].item() > 0.0
    assert s[0, 2].item() > 0.5 * s[0, 1].item()   # most of it stays after recovery


def test_no_inelastic_above_preconsolidation_head():
    col = VEPColumn(n_sites=1)
    with torch.no_grad():
        col.log_ske.fill_(torch.log(torch.tensor(1e-4)).item())
        col.log_skv.fill_(torch.log(torch.tensor(1e-2)).item())
        col.log_tau.fill_(-10.0)
        col.h_pc0.fill_(-20.0)        # already preconsolidated well below the loading
    h = torch.tensor([[0.0, -10.0, 0.0]])
    s = col(h)
    assert s[0, 2].item() == pytest.approx(0.0, abs=1e-6)   # purely elastic, recovers


def _step_response(tau_days, n_steps=60, dt=30.0):
    col = VEPColumn(n_sites=1, dt_days=dt)
    with torch.no_grad():
        col.log_ske.fill_(-30.0)                                  # elastic off
        col.log_skv.fill_(torch.log(torch.tensor(1e-2)).item())
        col.log_tau.fill_(torch.log(torch.tensor(float(tau_days))).item())
        col.h_pc0.fill_(0.0)
    h = torch.cat([torch.zeros(1, 1), torch.full((1, n_steps - 1), -10.0)], dim=1)
    return col(h)[0]


def test_tau_small_reproduces_the_instantaneous_limit():
    fast = _step_response(1e-3)
    assert fast[1].item() == pytest.approx(fast[-1].item(), rel=1e-3)   # arrives at once


def test_tau_large_suppresses_compaction_within_the_window():
    slow = _step_response(1e6)
    fast = _step_response(1e-3)
    assert slow[-1].item() < 0.05 * fast[-1].item()


def test_intermediate_tau_relaxes_monotonically_toward_equilibrium():
    mid = _step_response(365.0)
    d = torch.diff(mid[1:])
    assert torch.all(d >= -1e-9)                    # monotone approach
    assert mid[-1].item() > mid[2].item()           # still rising: memory present
