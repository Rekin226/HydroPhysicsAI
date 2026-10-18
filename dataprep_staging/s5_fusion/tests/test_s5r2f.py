"""CPU unit tests of the S5 fix-round labelling and checks (synthetic values only)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1]))

import s5r2f_diag as FD  # noqa: E402
import s5r2f_product as FP  # noqa: E402


def test_gate_v5_never_reads_fresh():
    g = FP.gate_v5({"T1F_fresh_pass": True, "T2_second_look_pass": False,
                    "T2b_second_look_pass": True, "T1_second_look_pass": False})
    assert g["passed"] is False and g["status"] == "FAILED"
    assert g["T1F_pass_reading"] == "conditional (second look)"
    assert g["fresh_unit_passes"] == 0
    assert not any("fresh_pass" in k for k in g)


def test_validation_v5_relabels_registration():
    v4 = {"gate": {"T1F_fresh_pass": True},
          "tests": {"t1f": {"registration": {"status": "registered fresh test"}}}}
    row = {"bin_km": [0, 2], "n": 1, "raw_coverage90": 1.0, "median_abs_z": 0.4,
           "share_abs_z_gt3": 0.0}
    fix = {"B_calibration": {"t2": {"by_distance": [row], "overall": {}},
                             "t2b": {"overall": {}}}, "C_product_regime": {}}
    v = FP.validation_v5(v4, fix)
    reg = v["tests"]["t1f"]["registration"]
    assert reg["status_at_registration"] == "registered fresh test"
    assert "second look" in reg["status"]
    assert v4["tests"]["t1f"]["registration"]["status"] == "registered fresh test"


def test_in_window_and_zstats():
    assert list(FD.in_window([2017.99, 2018.0, 2019.99, 2020.0])) == [False, True, True, False]
    rng = np.random.default_rng(1)
    z = rng.standard_normal(20000)
    T = pd.DataFrame({"sid": np.arange(z.size) % 50, "err_fused": z, "sd_fused": 1.0,
                      "hit_fused": np.abs(z) <= 1.6449, "dist_km": 1.0})
    s = FD.zstats(T)
    assert abs(s["median_abs_z"] - 0.6745) < 0.02
    assert abs(s["raw_coverage90"] - 0.90) < 0.01
