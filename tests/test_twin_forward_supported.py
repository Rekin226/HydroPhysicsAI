"""twin.forward refuses a theta, meta or column written by newer code (review code-02).

Before 2026-10-24 ``build_model`` copied the keys it knew and dropped the rest without a
word: the F2 theta's ``return_rice_b`` (and its meta ``return_rice_share``) rolled out as a
different model under the same name. Every theta it can rebuild still loads unchanged.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")

from hydrophysics.twin import forward as F  # noqa: E402

# the keys of a zonal, county-split, delay-bed theta of the kind calibrate_flow writes
THETA = {
    "log_T_proximal": [1.0], "log_S_mid": [-9.0, -9.0], "log_L_distal_changhua": [0.1],
    "log_T_proximal_w_changhua": [1.0], "log_Sd_mid": -3.0, "log_tau_mid": 5.0,
    "log_du0_distal": 0.2, "log_Sa_mid": -4.0, "log_G_mid": -6.0,
    "log_C_coast": [1.0], "log_C_apex": [2.0], "log_C_riv": [0.5, 0.6], "log_sw_scale": 0.0,
    "log_eta": 0.1, "log_head_extra": 1.0, "recharge_frac_logit": -1.0,
    "pump_split_logit": 0.3, "return_frac_logit": -0.5, "pump_layer_logits": [0.0, 1.0],
    "pump_layer_logits_ch": [0.0, 1.0], "log_tau_ret": 1.0, "log_eta_ratio_ch": 0.1,
    "log_spread_km": 1.5, "spread_km": 4.5,
    # readouts written beside the parameters
    "eta": 1.1, "head_extra_m": 2.7, "recharge_frac": 0.27, "C_coast_m2day": [2.7],
    "C_apex_m2day": 7.4, "C_riv_m2day": [1.6, 1.8], "pump_frac_shallow": 0.57,
    "return_frac": 0.1, "sw_scale": 1.0, "tau_ret_steps": 2.7, "eta_ratio_changhua": 1.1,
    "pump_layer_shares": [0.3, 0.7], "pump_layer_shares_ch": [0.3, 0.7], "Sd_mid": 0.05,
    "tau_days_mid": 148.0, "du0_m_distal": 1.2, "Sa_mid": 0.02, "G_per_day_mid": 0.002,
    "S_eff_mid": [0.1, 0.1, 0.1, 0.1], "delay_is_elastic_mid": False,
}
META = {"param_mode": "zonal", "zone_county_split": True, "git_commit": "abc1234",
        "rf_timeseries_sha256": "0" * 64}


def test_a_theta_this_code_rebuilds_is_accepted():
    F.check_supported(THETA, META)
    F.check_supported(THETA, None)


def test_a_newer_theta_key_is_refused_not_dropped():
    with pytest.raises(ValueError, match=r"unknown theta keys \['log_future_term'\]"):
        F.check_supported({**THETA, "log_future_term": 1.84}, META)


def test_a_newer_opt_in_meta_flag_is_refused(monkeypatch):
    monkeypatch.setattr(F, "UNSUPPORTED_META", ("future_share",))
    meta = {**META, "future_share": "future.npz"}
    with pytest.raises(ValueError, match=r"unsupported opt-in meta \['future_share'\]"):
        F.check_supported(THETA, meta)
    # an absent or empty flag is the historical behaviour
    F.check_supported(THETA, {**META, "future_share": None})


def test_the_f2_theta_and_meta_are_rebuilt_since_the_n1_merge():
    meta = {**META, "return_rice_share": "rice.npz", "return_rice_share_sha256": "ab"}
    F.check_supported({**THETA, "return_rice_b": 1.84}, meta)


def test_build_model_refuses_before_building_anything():
    member = F.Member(label="new", theta={**THETA, "log_future_term": 1.84}, meta=META)
    with pytest.raises(ValueError, match="newer code"):
        F.build_model(None, member, "cpu")


def test_a_flagged_column_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "UNSUPPORTED_COLUMN", ("future_flag",))
    path = tmp_path / "vep_zonal_leveling.json"
    path.write_text(json.dumps({"zonal": [], "future_flag": True}))
    with pytest.raises(ValueError, match="future_flag"):
        F.load_or_fit_vep(str(path), None, None, "cpu", zone_of_cell=None)
    F.check_supported_column({"zonal": [], "future_flag": False})
    F.check_supported_column({"zonal": [], "column_zones_collapsed": True})
