"""calibrate_coupled --seed: the leveling site folds follow the seed, 0 is the historical
partition, and the column JSON records a non-zero seed (the G2 fresh partition behind the
model card's fresh-fold figures, commit e6fd07b, which shipped without a test).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hydrophysics.twin import calibrate_coupled as cc  # noqa: E402


def _folds(monkeypatch, seed: int, n: int = 23, n_folds: int = 5) -> list[list[int]]:
    """The held-out site sets kfold_sites uses, with the fitting stubbed out."""
    held = []
    monkeypatch.setattr(cc, "_make", lambda config, device, n_zones: object())
    monkeypatch.setattr(cc, "_fit", lambda *a, **k: 0.0)

    def predict(model, H, Z):
        held.append(sorted(int(i) for i in H[:, 0, 0].tolist()))
        return torch.zeros(H.shape[0], 3)

    monkeypatch.setattr(cc, "_predict", predict)
    monkeypatch.setattr(cc, "_r2", lambda p, o, m: 0.0)
    H = torch.arange(n, dtype=torch.float32).reshape(n, 1, 1).expand(n, 1, 3).clone()
    OBS, M, Z = torch.zeros(n, 3), torch.ones(n, 3), torch.zeros(n, dtype=torch.int64)
    cc.kfold_sites("zonal", H, OBS, M, Z, 1, 0.1, "cpu", n_folds=n_folds, seed=seed,
                   rezero=False)
    return held


def test_seed_zero_is_the_historical_partition(monkeypatch):
    order = np.random.default_rng(0).permutation(23)
    want = [sorted(int(i) for i in f) for f in np.array_split(order, 5)]
    assert _folds(monkeypatch, 0) == want


def test_another_seed_gives_another_partition_of_the_same_sites(monkeypatch):
    a, b = _folds(monkeypatch, 0), _folds(monkeypatch, 7)
    assert a != b
    assert sorted(i for f in b for i in f) == list(range(23))     # every site held out once
    assert _folds(monkeypatch, 7) == b                              # deterministic


def test_the_column_json_records_a_non_zero_seed_only():
    assert cc.fold_seed_record(0) == {}
    assert cc.fold_seed_record(7) == {"fold_seed": 7}


def test_the_cli_takes_the_seed(monkeypatch):
    seen = {}

    def stop(self, args=None, namespace=None):
        ns = real(self, args, namespace)
        seen["seed"] = ns.seed
        raise SystemExit(0)

    real = cc.argparse.ArgumentParser.parse_args
    monkeypatch.setattr(cc.argparse.ArgumentParser, "parse_args", stop)
    with pytest.raises(SystemExit):
        cc.main(["--theta", "x.json", "--seed", "7", "--out", "unused"])
    assert seen["seed"] == 7
