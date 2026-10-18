"""WP P4m item (5): the Phase-1 rung-R1 recommendation from pre-origin evidence only (U7).

    cd dataprep_staging
    python -m p4_changhua.p4m_recommend

Reads the staged results of ``p4m_run`` (T1-T3, T2c), ``p4m_refit`` (T2b) and
``p4m_evidence`` (E8, BE, E9), applies the rules declared in ``P4m.metered_diagnostics.v1``
(T5) and writes ``stage/recommendation/recommendation_preorigin.json`` with the hashes of its
inputs. It must run before ``p4m_recorded`` reads any 2023+ head; it refuses if the recorded
arm's results already exist.

The declaration says an independent indicator "points the same way" as a theta-refit scale
below 0.67 without defining it; this script fixes the reading (written before any P4m result
was read; ``rules_operationalised`` in the output):

* E8: the Changhua/Yunlin M2 amplitude ratio at equal easting and layer has its 90 % interval
  above 1 (a more diffusive Changhua aquifer, hence a weaker and broader pumping response);
* BE: the implied Ss ratio Changhua/Yunlin (BE_Yunlin / BE_Changhua) exceeds 1.25 in the
  pump layer (layer index 1) or, without pump-layer data, in all layers pooled (more storage,
  hence a damped seasonal response);
* E9: at Changhua nested sites the observed pump-layer (model layer 1) slope on local metered
  energy is smaller in magnitude than the A1 model's (median |b_obs| < 0.67 median |b_mod|,
  primary radius 3 km): the model over-responds locally.
"""
from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.hashing import sha256_file  # noqa: E402
from common.paths import rel_to_repo  # noqa: E402

from p4_changhua import p4m_lib as L  # noqa: E402

OUT = L.STAGE / "recommendation"
RUN = L.STAGE / "run" / "results.json"
REFIT = L.STAGE / "refit" / "results.json"
EVID = L.STAGE / "evidence" / "results.json"
RECORDED = L.STAGE / "recorded" / "results.json"
DTHETA_MIN = 0.05
S_THETA_MULT = 0.67


def _e8_points(e8: dict) -> tuple[bool | None, str]:
    o = e8.get("ols_log_AM2")
    if not o:
        return None, "E8: too few coherent wells for the county comparison"
    lo, hi = o["ratio_p05_p95"]
    return bool(lo > 1.0), (f"E8: Changhua/Yunlin M2 amplitude ratio {o['changhua_over_yunlin_ratio']:.2f} "
                            f"(90 % {lo:.2f}-{hi:.2f}, uncalibrated)")


def _be_points(be: dict) -> tuple[bool | None, str]:
    r = be.get("ratios", {})
    if "layer1" in r:
        v = r["layer1"]["implied_Ss_CH_over_YL"]
        return bool(v > 1.25), f"BE: implied Ss ratio CH/YL in the pump layer {v:.2f}"
    a = be.get("all_layers", {})
    if a.get("median_CH") and a.get("median_YL"):
        v = a["median_YL"] / a["median_CH"]
        return bool(v > 1.25), f"BE: implied Ss ratio CH/YL, all layers pooled {v:.2f}"
    return None, "BE: not enough accepted epochs"


def _e9_points(e9: dict) -> tuple[bool | None, str]:
    blk = e9["radii"].get(str(e9.get("primary_r_km", 3.0)), {}).get("changhua", {})
    s = blk.get("slopes_median_m_per_GWh", {}).get("layer1")
    if not s or s["n"] < 3:
        return None, "E9: fewer than 3 Changhua pump-layer wells with a slope"
    ratio = abs(s["obs"]) / abs(s["model_A1"]) if s["model_A1"] else float("nan")
    return bool(ratio < S_THETA_MULT), (f"E9: Changhua pump-layer local slope obs/model "
                                        f"{ratio:.2f} (median |b|, r 3 km, n {s['n']})")


def main() -> int:
    if RECORDED.exists():
        raise SystemExit("the recorded arm has already been scored; the recommendation must "
                         "precede it (U7)")
    run = json.loads(RUN.read_text())
    refit = json.loads(REFIT.read_text())
    ev = json.loads(EVID.read_text())
    t2 = run["T2"]["metered"]
    s_theta = refit.get("s_theta")
    out: dict = {"wp": "P4m", "prereg": L.PRE_DIAG,
                 "created": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                 "u7": "pre-origin evidence only: calibration heads 2012-02..2022-12, C0 "
                       "preorigin/v3, H1a v3 pre-origin rows, L0 hourly heads to 2022-12; no "
                       "2023+ observation was read by P4m before this file",
                 "inputs": [{"path": rel_to_repo(p), "sha256": sha256_file(p)}
                            for p in (RUN, REFIT, EVID)],
                 "rules_operationalised": __doc__.split("fixes the reading")[1].strip()}
    # ------------------------------------------------------------ evidence summary
    t1 = run["T1"]
    out["evidence"] = {
        "T1_changhua_anomaly_rmse": {a: t1["groups"]["CH"][a]["anomaly_rmse"]
                                     for a in ("A0", "A1", "A2")},
        "T1_changhua_amplitude_ratio": {a: t1["groups"]["CH"][a]["amplitude_ratio"]
                                        for a in ("A0", "A1", "A2")},
        "T1_yunlin_anomaly_rmse": {a: t1["groups"]["YL"][a]["anomaly_rmse"]
                                   for a in ("A0", "A1", "A2")},
        "T1_dRMSE_changhua_A1_minus_A0": t1["differences"]["CH"]["A1_minus_A0"],
        "T2_s_LS_all": t2["s_LS_all"], "T2_s_LS_changhua": t2["s_LS_changhua"],
        "T2_s_AM_changhua": t2["s_AM_changhua"], "T2_boot_s_LS_all": t2["boot_s_LS_all"],
        "T2_far_below_1": t2["far_below_1"],
        "T2a_yunlin_shape": {k: run["T2"]["yunlin_shape"][k]
                             for k in ("s_LS_all", "s_AM_changhua")},
        "T2a_seasonal_cv": run["T2"]["seasonal_cv"],
        "T2b_s_frozen_cal": refit["frozen"]["s_frozen_cal"], "T2b_s_theta": s_theta,
        "T2b_theta_bias_share": refit.get("theta_bias_share_of_gap"),
        "T2b_stop_reasons": {k: v["stop_reason"] for k, v in refit["levels"].items()},
        "T3_mean_channel_verdict": run["T3"]["mean_channel_verdict"],
        "T3_disappears": run["T3"]["disappears"]}
    # ------------------------------------------------------------ rule: Changhua multiplier
    ind = {"E8": _e8_points(ev["E8"]), "BE": _be_points(ev["BE"]), "E9": _e9_points(ev["E9"])}
    any_same = any(v[0] for v in ind.values())
    mult = bool(s_theta is not None and s_theta < S_THETA_MULT and any_same)
    out["changhua_specific_response_multiplier"] = {
        "recommended": mult,
        "rule": f"s_theta < {S_THETA_MULT} and at least one independent indicator points the "
                "same way",
        "s_theta": s_theta,
        "indicators": {k: {"points_same_way": v[0], "reading": v[1]} for k, v in ind.items()}}
    # ------------------------------------------------------------ rule: widened bounds
    pins = refit["levels"]["1.0"]["pinned_at_refit"]
    out["widened_bound_sensitivity"] = [
        {"entry": p["name"], "side": p["side"], "pushes_outward_at_s1_refit": p[
            "descent_pushes_outward"]} for p in pins if p["descent_pushes_outward"]]
    out["pinned_at_s1_refit_all"] = pins
    # ------------------------------------------------------------ rule: direction
    out["expected_theta_shift"] = [
        {"entry": r["name"], "dtheta": r["dtheta"],
         "direction": "up" if r["dtheta"] > 0 else "down"}
        for r in refit["dtheta_refit1_minus_refit0"] if abs(r["dtheta"]) > DTHETA_MIN]
    # ------------------------------------------------------------ the recommendation
    free = ["the R0 recipe's zonal parameters unchanged (log T, log S per zone and layer, "
            "leakance per zone and interface, coast and apex conductances, pump split, return "
            "flow, spread, recharge fraction), as I1.R1.changhua_census.v1 declares"]
    if mult:
        free.append("a Changhua-specific (cell mask) response multiplier, with a prior, "
                    "because the shared zonal parameters could not absorb the metered series "
                    "in the CPU emulation and independent evidence points the same way")
    out["recommendation"] = {
        "rung": "R1 = recalibrate the batch-8 cwa recipe with the Changhua (+ in-grid Chiayi) "
                "census as forcing",
        "forcing": "C0 pump_electricity/preorigin/v3 (Changhua + in-grid Chiayi, all classes; "
                   "Yunlin from the same product for consistency, or the model's cache, "
                   "whose only material difference is 2022-12). If I1's model_census/v1 is "
                   "used, check its December-2022 convention against v3's fill (C0 issue). The "
                   "no-domestic variant (A2) is the declared sensitivity member.",
        "free_parameters": free,
        "widened_bound_sensitivity_rung": [w["entry"] for w in out["widened_bound_sensitivity"]],
        "expected_theta_shift_from_cpu_emulation": out["expected_theta_shift"],
        "acceptance_tests_proposed": [
            "plan I1 binding (pre-origin): temporal screen (fit 2012-19, score 2020-22) RMSE "
            "ratio within 1 % of R0 and verdict not worse; leveling out of fold (>= 10 km "
            "buffer) not worse; CPU held-out-site proxy not worse",
            "Changhua out-of-fold anomaly RMSE <= R0's and Changhua amplitude ratio closer to "
            "1 (as I1 declared); Yunlin out-of-fold RMSE not worse than R0 by more than 2 %",
            "P4m diagnostics rerun on R1's theta (CPU engine, frozen R1 theta): the "
            "least-squares scale of the metered series in [0.8, 1.25] and the Changhua "
            "amplitude-matched scale in [0.67, 1.5]; if not, the remaining gap is response or "
            "conversion, not theta bias, and the multiplier rule above is re-applied",
            "E10 rerun on R1: the mean-channel coefficient not significantly negative in at "
            "least one variant; residual trends at Changhua wells reported",
            "theta shift of R1 vs R0 compared in sign with the CPU emulation's dtheta (entries "
            "with |dtheta| > 0.05): reported, not gating",
            "never gating (U7): the 2023-25 recorded-forcing arm with metered Changhua "
            "forcing, the 2023-26 frozen challenge, subsidence scores"],
        "burn_in_or_spinup": ("the calibration loss has no burn-in; "
                              + ("the spin-up IC (V4) removes the mean-channel sign problem, "
                                 "so R1 should start from a consistent 2007-2011 spin-up"
                                 if not run["T3"]["mean_channel_significantly_negative"].get(
                                     "V4_spinup_12m_about_A0s", True)
                                 else "the spin-up IC (V4) does not remove the negative mean "
                                      "channel: keep the observed IC and report E10 on R1")),
        "u7": out["u7"]}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "recommendation_preorigin.json"
    if path.exists():
        raise SystemExit(f"{path} exists; a recommendation is written once")
    L.jdump(out, path)
    print(path, sha256_file(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
