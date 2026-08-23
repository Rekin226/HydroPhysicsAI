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
