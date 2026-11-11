"""Round-12 frozen layer: unit checks on synthetic inputs (no agency data)."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("scipy")

from dataprep_staging.r12_layer import model as M  # noqa: E402


def test_well_dmean_needs_two_full_hydrological_years():
    idx = pd.date_range("2018-06-01", "2021-05-01", freq="MS")
    W = pd.DataFrame({"a": np.r_[np.full(12, 10.0), np.full(12, 9.0), np.full(12, 9.5)],
                      "b": np.r_[np.full(12, 5.0), np.full(4, 4.0), np.full(8, np.nan),
                                 np.full(12, 4.5)]}, index=idx)
    d = M.well_dmean(W)
    assert d.loc["a", 2020] == pytest.approx(-1.0)
    assert d.loc["a", 2021] == pytest.approx(0.5)
    assert np.isnan(d.loc["a", 2019])                 # no earlier year to difference
    assert np.isnan(d.loc["b", 2020])                 # 4 months is not a year
    assert np.isnan(d.loc["b", 2021])                 # its previous year is missing


def test_fit_gamma_recovers_a_known_sensitivity():
    rng = np.random.default_rng(0)
    n, years = 400, M.YEARS
    xy = rng.uniform(0, 30_000, size=(n, 2))
    rate = 0.01 + 0.03 * xy[:, 0] / 30_000
    F = np.tile(rng.normal(0, 1.0, size=len(years)), (n, 1))
    g_true = -0.15
    D = np.full((n, len(years)), np.nan)
    for j in range(3, len(years)):
        drv = F[:, j] - F[:, j - 3:j].mean(1)
        D[:, j] = rate * (1 + g_true * drv) + rng.normal(0, 0.002, size=n)
    D[:, :3] = rate[:, None]
    g, per_year = M.fit_gamma(D, F, xy, years=(2008, 2021))
    assert len(per_year) == 14
    assert g == pytest.approx(g_true, abs=0.05)


def test_to_cells_leaves_far_cells_empty():
    xy = np.array([[0.0, 0.0], [1000.0, 0.0], [0.0, 1000.0]])
    v = np.array([0.02, 0.02, 0.02])
    out = M.to_cells(np.tile(v, 14)[:40], np.tile(xy, (14, 1))[:40],
                     np.array([[500.0, 500.0], [20_000.0, 0.0]]))
    assert out[0] == pytest.approx(0.02)
    assert np.isnan(out[1])
