"""Fixed extra abstraction (``--extra-abstraction``, rung R1e-pw) and the county split of the
canal recharge fractions (``--sw-county-split``, rung R1e-cn), 2026-10-18.

- the loader aligns the months, holds the last one past the file's end, refuses a run that
  starts before it and a file built for another grid;
- the fit and the evaluation subtract the field as static pumping: a zero field reproduces
  the plain fit bit for bit, a non-zero one lowers the heads;
- a model calibrated with it refuses to roll out without it; the projection holds the
  record's last month;
- the county split keeps each component's total exactly, Changhua cells first;
- the meta records the path, the file hash and the split, only when set."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from test_twin_well_datum import _problem  # noqa: E402

from hydrophysics.twin.calibrate_flow import (  # noqa: E402
    _extra_forcing_record,
    _predict_homogeneous,
    fit_flow,
)
from hydrophysics.twin.flow import FlowModel  # noqa: E402
from hydrophysics.twin.forward import extra_hist, future_extra, rollout  # noqa: E402
from hydrophysics.twin.inputs import (  # noqa: E402
    load_extra_abstraction,
    split_sw_by_county,
)

D = torch.float64


def _write_npz(path, q, dates, dx=1000.0):
    np.savez(path, q_m3d=q, dates=np.array(dates), dx=np.float64(dx),
             n_active=np.int64(q.shape[1]))


def _months(a, b):
    return [str(p) for p in pd.period_range(a, b, freq="M")]


def test_loader_aligns_holds_the_last_month_and_checks_the_grid(tmp_path):
    grid = SimpleNamespace(dx=1000.0, n_active=3)
    q = np.zeros((4, 3, 6))
    q[2, 1, :] = np.arange(6.0) + 1.0
    p = tmp_path / "q.npz"
    _write_npz(p, q, _months("2012-01", "2012-06"))
    got = load_extra_abstraction(str(p), grid,
                                 pd.date_range("2012-03-01", periods=7, freq="MS"))
    assert got.shape == (4, 3, 7) and got.dtype == D
    # March..June from the file, then June held for July..September
    assert got[2, 1].tolist() == [3.0, 4.0, 5.0, 6.0, 6.0, 6.0, 6.0]
    assert float(got[[0, 1, 3]].abs().sum()) == 0.0
    with pytest.raises(ValueError, match="starts 2012-01"):
        load_extra_abstraction(str(p), grid, pd.date_range("2011-12-01", periods=2, freq="MS"))
    with pytest.raises(ValueError, match="built for dx"):
        load_extra_abstraction(str(p), SimpleNamespace(dx=1000.0, n_active=4),
                               pd.date_range("2012-01-01", periods=2, freq="MS"))
    q[0, 0, 0] = -1.0
    _write_npz(p, q, _months("2012-01", "2012-06"))
    with pytest.raises(ValueError, match="finite and >= 0"):
        load_extra_abstraction(str(p), grid, pd.date_range("2012-01-01", periods=2, freq="MS"))


def _fit(extra_q=None, epochs=2, lr=0.05):
    g, h0, rch, rf, idx, lay, _ = _problem()
    T = rch.shape[-1]
    obs = torch.full((4, T), 10.0, dtype=D)
    m = FlowModel(g, n_layers=4, dt_days=30.0)
    fit = fit_flow(m, obs, idx, lay, rch, epochs=epochs, lr=lr, h0=h0, recharge_field=rf,
                   extra_q=extra_q)
    return g, m, fit, h0, rf, T


@pytest.mark.filterwarnings("ignore:_cg did not converge")
def test_zero_field_reproduces_the_plain_fit_and_a_real_one_lowers_heads():
    g, m0, f0, h0, rf, T = _fit()
    zero = torch.zeros(4, g.n_active, T, dtype=D)
    _, m1, f1, *_ = _fit(zero)
    assert f1["loss"] == f0["loss"]
    for k in f0["theta"]:
        assert np.array_equal(np.asarray(f0["theta"][k]), np.asarray(f1["theta"][k])), k
    a = _predict_homogeneous(m0, f0, h0, T, recharge_field=rf)
    b = _predict_homogeneous(m1, f1, h0, T, recharge_field=rf, extra_q=zero)
    assert torch.equal(a, b)
    q = zero.clone()
    q[2] = 50.0                                    # m3/day per cell from layer 2
    c = _predict_homogeneous(m0, f0, h0, T, recharge_field=rf, extra_q=q)
    assert (c[2, :, -1] < a[2, :, -1]).all()
    # and the fit sees it: a different optimum from the same start
    _, _, f2, *_ = _fit(q)
    assert f2["loss"] != f0["loss"]


def test_rollout_refuses_a_model_calibrated_with_it_when_it_is_missing():
    with pytest.raises(ValueError, match="--extra-abstraction"):
        rollout(None, {"extra_abstraction": "q.npz"}, torch.zeros(4, 3, dtype=D),
                torch.zeros(3, 2, dtype=D), torch.zeros(3, 2, dtype=D),
                torch.zeros(3, dtype=D))


def test_history_slice_and_projection_hold():
    q = torch.arange(4 * 2 * 5, dtype=D).reshape(4, 2, 5)
    inp = SimpleNamespace(extra_q=q)
    assert torch.equal(extra_hist(inp), q[..., 1:])
    fut = future_extra(inp, 3)
    assert fut.shape == (4, 2, 3) and torch.equal(fut, q[..., -1:].expand(4, 2, 3))
    assert extra_hist(SimpleNamespace(extra_q=None)) is None
    assert future_extra(SimpleNamespace(extra_q=None), 3) is None


def test_county_split_keeps_each_component_and_orders_changhua_first():
    rng = np.random.default_rng(0)
    sw = torch.tensor(rng.uniform(0, 1, (2, 5, 4)), dtype=D)
    ch = np.array([True, False, True, False, False])
    out = split_sw_by_county(sw, ch)
    assert out.shape == (4, 5, 4)
    for k in range(2):
        assert torch.equal(out[2 * k] + out[2 * k + 1], sw[k])
        assert float(out[2 * k][~torch.as_tensor(ch)].abs().sum()) == 0.0
        assert float(out[2 * k + 1][torch.as_tensor(ch)].abs().sum()) == 0.0
    single = split_sw_by_county(sw[0], ch)        # an (A, T) field gives two components
    assert single.shape == (2, 5, 4)
    with pytest.raises(ValueError, match="county flags"):
        split_sw_by_county(sw, ch[:4])


def test_meta_records_path_hash_and_split_only_when_set(tmp_path):
    assert _extra_forcing_record(SimpleNamespace(extra_abstraction=None,
                                                 sw_county_split=False)) == {}
    p = tmp_path / "q.npz"
    p.write_bytes(b"abc")
    rec = _extra_forcing_record(SimpleNamespace(extra_abstraction=str(p),
                                                sw_county_split=True))
    assert rec["extra_abstraction"] == str(p)
    assert rec["extra_abstraction_sha256"] == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    assert rec["sw_county_split"] is True
