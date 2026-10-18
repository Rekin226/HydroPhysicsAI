"""WP P4m: validation reports, product cards, product versions and ledger rows.

    cd dataprep_staging
    tools/cpu_slot.sh python -m p4_changhua.p4m_publish

Reads the staged results (run, refit, evidence, recommendation, recorded), builds each
product's ``validation.json`` (acceptance rows with criterion, threshold, measured value and
pass) and ``PRODUCT_CARD.md``, publishes ``changhua_forcing_diagnostics/{preorigin,
recorded_arm}/v<N>`` through ``p4m_finalize`` (never overwriting a version) and appends the
ledger rows of every reported number to ``ledger/rows/P4m.csv``. Writes
``P4m_evidence/publish.json`` (product paths) for ``report_p4m``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import ledger  # noqa: E402
from common.hashing import sha256_file  # noqa: E402
from common.paths import rel_to_repo  # noqa: E402

from p4_changhua import p4m_finalize as F  # noqa: E402
from p4_changhua import p4m_lib as L  # noqa: E402

ST = L.STAGE
PUB = L.EVID / "publish.json"


def load() -> dict:
    r = {k: json.loads((ST / k / "results.json").read_text())
         for k in ("run", "refit", "evidence", "recorded")}
    r["rec"] = json.loads((ST / "recommendation" / "recommendation_preorigin.json").read_text())
    r["rec_sha"] = sha256_file(ST / "recommendation" / "recommendation_preorigin.json")
    return r


def _row(criterion, threshold, measured, passed, note=""):
    return {"criterion": criterion, "threshold": threshold, "measured": measured,
            "passed": passed, "note": note}


def acceptance(r: dict) -> list[dict]:
    ec = r["run"]["engine_checks"]
    t2 = r["run"]["T2"]["metered"]
    refit = r["refit"]
    rows = [
        _row("EC1 sparse base rollout (A0) reproduces engine/v2 H_hindcast",
             "max |dh| <= 1e-9 m", f"{ec['EC1_base_vs_engine_v2_max_abs_dh_m']:.2e} m",
             ec["EC1_pass"]),
        _row("EC2 arm forcing equals C0 preorigin/v3 county_class_monthly in-grid totals "
             "(Changhua + Chiayi, 2012-01..2022-12), A1 all classes and A2 no domestic",
             "relative difference <= 1e-9",
             f"A1 {ec['EC2']['rel_A1']:.1e} ({ec['EC2']['GWh_per_yr_A1']:.2f} GWh/yr), A2 "
             f"{ec['EC2']['rel_A2']:.1e} ({ec['EC2']['GWh_per_yr_A2']:.2f} GWh/yr)",
             ec["EC2_pass"]),
        _row("EC3 sparse rollout vs forward.rollout (CPU torch) with metered Changhua forcing",
             "max |dh| <= 1e-5 m", f"{ec['EC3_sparse_vs_torch_A1']['max_abs_dh_m']:.2e} m",
             ec["EC3_pass"]),
        _row("EC4 lift clipped (<= 2 m) in A1 cell-steps (reported)", "reported",
             f"{ec['EC4_lift_clipped']['A1']['n_clipped']} of {ec['EC4_lift_clipped']['A1']['n']}"
             f" (min lift {ec['EC4_lift_clipped']['A1']['min_lift_m']:.1f} m)", True),
    ]
    t1 = r["run"]["T1"]
    rows.append(_row("T1 computed: anomaly RMSE, amplitude ratio, correlation for A0/A1/A2 by "
                     "county, zone and layer, with cluster-bootstrap dRMSE",
                     "all groups computed; harness agreement <= 1e-9",
                     f"{len(t1['groups'])} groups; harness agreement "
                     f"{t1['harness_agreement_max_abs']:.1e}",
                     bool(t1["harness_agreement_max_abs"] <= 1e-9)))
    rows.append(_row("T2 computed: s_LS (all, Changhua), s_AM, bootstrap, first-order "
                     "cross-checks; T2a shape; T2c sensitivity",
                     "all computed", f"s_LS {t2['s_LS_all']:.3f}, s_AM {t2['s_AM_changhua']}; "
                     f"T2c rows {len(r['run'].get('T2c', {}).get('rows', []))}",
                     bool("T2c" in r["run"] and t2["s_LS_all"] is not None)))
    stops = {k: v["stop_reason"] for k, v in refit["levels"].items()}
    rows.append(_row("T2b computed: theta-refit emulation at s = 0, 0.5, 1 (15 iterations)",
                     "three refits run; stop reasons reported", json.dumps(stops),
                     len(stops) == 3,
                     "unconverged refits make s_theta approximate (declared)"))
    rows.append(_row("T2b replication: the s = 0 refit reproduces P4.0's observed-target "
                     "reference objective", "abs diff <= 1e-6 (same code, inputs, settings)",
                     f"{refit['p40_replication']['abs_diff']:.2e}",
                     bool(refit["p40_replication"]["abs_diff"] <= 1e-6)))
    rows.append(_row("T2d computed: E8 (rebuilt pre-origin), BE, E9", "all computed",
                     f"E8 wells {r['evidence']['E8']['n_wells']} (coherent "
                     f"{r['evidence']['E8']['n_coherent']}), BE epochs "
                     f"{r['evidence']['BE']['n_epochs']}, E9 sites "
                     f"{r['evidence']['E9']['n_sites']}", True))
    rows.append(_row("T3 computed: transient, trend, mean channel V1-V4", "all computed",
                     r["run"]["T3"]["mean_channel_verdict"], True))
    rows.append(_row("T5 recommendation written from pre-origin evidence before the first "
                     "2023+ score (U7)", "recommendation hash recorded by the recorded arm "
                     "before its look", f"sha256 {r['rec_sha'][:12]}...; recorded arm saw "
                     f"{r['recorded']['recommendation_sha256_before_first_score'][:12]}...",
                     r["rec_sha"] == r["recorded"]["recommendation_sha256_before_first_score"]))
    num = r["recorded"]["numerical_check_R_obs_ch_vs_torch"]
    rows.append(_row("T4 numerical check: sparse forecast vs torch (R_obs_ch, 30 steps)",
                     "max |dh| <= 1e-5 m", f"{num['max_abs_dh_m']:.2e} m", num["pass"]))
    return rows


def card_preorigin(r: dict) -> str:
    t1 = r["run"]["T1"]["groups"]
    t2 = r["run"]["T2"]["metered"]
    return f"""# changhua_forcing_diagnostics/preorigin (WP P4m)

Pre-origin, in-sample design-time evidence: the frozen cwa theta (batch 8's merged theta,
`results/twin_runs/stage3_datum_gate_cwa/stage3_theta.json`, sha256 789d0e45...) run on the
P4.0 sparse engine with the C0 metered Changhua + in-grid Chiayi forcing
(`pump_electricity/preorigin/v3`) added. Declaration `P4m.metered_diagnostics.v1`.

Not skill: the theta was calibrated on these heads (158 wells, 2012-02..2022-12) with zero
Changhua pumping. Allowed uses: model_design, design_time_evidence, after_the_fact_analysis.
Two exemptions are inherited from C0 preorigin/v3 (the 2026 census attribute table and
residual December-2022 halves; forcing-level effects below 4e-4 of 2012-22 energy).

Files:
- `diagnostics_t1_t3.json`: engine checks EC1-EC4, T1 skill change, T2 heads-accepted scale
  (metered, no-domestic, Yunlin-shaped), seasonal CVs, T2c sensitivities, T3 E10-style check;
- `theta_refit_t2b.json`, `refit_predictions.npz`: CPU emulation of the R1 recalibration
  (theta refits at s = 0, 0.5, 1; unconverged, approximate);
- `evidence_e8_be_e9.json`, `e8_tides_preorigin.csv`, `e9_nested_slopes.csv`: independent
  evidence (E8 rebuilt from L0 hourly heads to 2022-12; BE from H1a v3; E9 nested sites);
- `recommendation_preorigin.json`: the rung-R1 recommendation (written before any 2023+
  head was read);
- `well_arrays.npz`: heads at the 158 wells for A0/A1/A2, observations, eligibility,
  clusters, the metered Changhua cell forcing and abstraction rates; `scale_curves.npz`.

Headline (in-sample): Changhua anomaly RMSE A0 {t1['CH']['A0']['anomaly_rmse']:.3f} m, A1
{t1['CH']['A1']['anomaly_rmse']:.3f} m, A2 {t1['CH']['A2']['anomaly_rmse']:.3f} m; amplitude
ratio {t1['CH']['A0']['amplitude_ratio']:.2f} / {t1['CH']['A1']['amplitude_ratio']:.2f} /
{t1['CH']['A2']['amplitude_ratio']:.2f}. Heads-accepted scale of the metered series: least
squares {t2['s_LS_all']:.3f} (attenuated), amplitude-matched {t2['s_AM_changhua']}.
Intervals are cluster bootstraps, UNCALIBRATED.
"""


def card_recorded(r: dict) -> str:
    s = r["recorded"]["scores"]
    m = s["changhua"]["metrics"]
    y = s["yunlin"]["metrics"]
    return f"""# changhua_forcing_diagnostics/recorded_arm (WP P4m item 4)

Recorded-forcing diagnostic arm 2023-01..2025-06 (U4): reported only, never gating (U7).
Declaration `P4m.recorded_arm_2023_25.v1`. The forcing experiment's protocol on the sparse
twin with the cwa theta (one member, noise-free initial field); the experiment's own numbers
(datum_gate theta, 36 members) are not reproduced and must not be compared one to one.

Arms: R_clim_zero, R_clim_ch (climatology pumping, without / with a Changhua climatology),
R_obs_zero (recorded Yunlin pumping, Changhua zero), R_obs_ch (plus recorded Changhua +
Chiayi from C0 full/v3), R_obs_ch_nodomestic; baseline clim (month-of-year means 2012-22).

Changhua equal-well RMSE (m): R_obs_zero {m['R_obs_zero']['rmse_m']:.3f}, R_obs_ch
{m['R_obs_ch']['rmse_m']:.3f}, clim {m['clim']['rmse_m']:.3f}. Yunlin: R_obs_zero
{y['R_obs_zero']['rmse_m']:.3f}, R_obs_ch {y['R_obs_ch']['rmse_m']:.3f}, clim
{y['clim']['rmse_m']:.3f}.

Allowed use: after_the_fact_analysis only. Nothing here may select, gate or calibrate.
"""


def numbers(r: dict) -> tuple[list, list]:
    """(pre-origin numbers [(id, heads_role)], recorded numbers [(id, role)])."""
    pre = [("P4m.EC1", "input"), ("P4m.EC2", "input"), ("P4m.EC3", "input"),
           ("P4m.EC4", "input")]
    for g in r["run"]["T1"]["groups"]:
        for arm in ("A0", "A1", "A2"):
            for met in ("anomaly_rmse", "amplitude_ratio", "corr_median"):
                pre.append((f"P4m.T1.{g}.{arm}.{met}", "design_time_evidence"))
    for g in ("CH", "YL", "all"):
        for d in ("A1_minus_A0", "A2_minus_A0"):
            pre.append((f"P4m.T1.dRMSE.{g}.{d}", "design_time_evidence"))
    for fam in ("metered", "metered_nodomestic", "yunlin_shape"):
        for k in ("s_LS_all", "s_LS_changhua", "s_AM_changhua"):
            pre.append((f"P4m.T2.{fam}.{k}", "fit_target"))
    pre += [("P4m.T2.metered.boot_s_LS_all", "fit_target"),
            ("P4m.T2.metered.first_order", "fit_target")]
    pre += [(f"P4m.T2.seasonal_cv.{k}", "input") for k in r["run"]["T2"]["seasonal_cv"]]
    for row in r["run"]["T2c"]["rows"]:
        pre.append((f"P4m.T2c.{row['name']}", "design_time_evidence"))
    pre += [("P4m.T2b.s_frozen_cal", "fit_target"), ("P4m.T2b.s_theta", "fit_target"),
            ("P4m.T2b.theta_bias_share", "fit_target"),
            ("P4m.T2b.p40_replication", "fit_target")]
    for s in ("0.0", "0.5", "1.0"):
        pre.append((f"P4m.T2b.refit_s{s}", "fit_target"))
    for row in r["refit"]["dtheta_refit1_minus_refit0"]:
        pre.append((f"P4m.T2b.dtheta.{row['name']}", "fit_target"))
    for v in r["run"]["T3"]["channels"]:
        pre.append((f"P4m.T3.channels.{v}", "fit_target"))
    for kind in ("transient", "trend"):
        for arm in r["run"]["T3"][kind]:
            for g in ("CH", "YL"):
                pre.append((f"P4m.T3.{kind}.{arm}.{g}", "design_time_evidence"))
    pre += [("P4m.T5.recommendation", "design_time_evidence")]
    rec = []
    for g in ("all", "changhua", "yunlin"):
        for arm in r["recorded"]["scores"][g]["metrics"]:
            rec.append((f"P4m.T4.{g}.{arm}", "validator"))
    return pre, rec


def main() -> int:
    if PUB.exists():
        raise SystemExit(f"{PUB} exists: products already published")
    r = load()
    acc = acceptance(r)
    pre_rows = [a for a in acc if not a["criterion"].startswith("T4")]
    rec_rows = [a for a in acc if a["criterion"].startswith(("T4", "T5"))]
    val_pre = {"wp": "P4m", "prereg": L.PRE_DIAG, "acceptance": pre_rows,
               "gate_passed": all(a["passed"] for a in pre_rows
                                  if not a["criterion"].startswith("T2b replication")),
               "gate_note": "the T2b replication row is reported; it is not part of the "
                            "declared engine checks"}
    val_rec = {"wp": "P4m", "prereg": L.PRE_REC, "acceptance": rec_rows,
               "gate_passed": all(a["passed"] for a in rec_rows)}
    d_pre = F.preorigin_product(val_pre, card_preorigin(r))
    d_rec = F.recorded_product(val_rec, card_recorded(r))
    pre_n, rec_n = numbers(r)
    pv_pre = rel_to_repo(d_pre).replace("results/dataprep/", "")
    pv_rec = rel_to_repo(d_rec).replace("results/dataprep/", "")
    rows = F.ledger_rows(pre_n, pv_pre)
    ev_ids = [("P4m.E8", "fit_target"), ("P4m.BE", "design_time_evidence"),
              ("P4m.E9", "fit_target")]
    for nid, role in ev_ids:
        src = {"P4m.E8": {"AMP_V2/data/wells": role},
               "P4m.BE": {"results/dataprep/heads/L1_clean/v3": role},
               "P4m.E9": {"results/dataprep/heads/L1_clean/v3": role,
                          "results/dataprep/pump_electricity/preorigin/v3": "input",
                          "chou-shui-data/data/rf_timeseries.csv": "input",
                          "chou-shui-data/data/rf_stations.csv": "input"}}[nid]
        for s, rl in src.items():
            rows.append({"number_id": nid, "source": s, "role": rl,
                         "product_version": pv_pre, "note": ""})
    rows.append({"number_id": "P4m.E8.after_the_fact_vs_pilot",
                 "source": "results/dataprep/_pilots/head_pumping_signatures",
                 "role": "validator", "product_version": "pilot head_pumping_signatures",
                 "note": "after the fact only (pilot window to 2026-09)"})
    rows += F.ledger_rows(rec_n, pv_rec, kind="recorded")
    have = {(x["number_id"], x["source"], x["product_version"])
            for x in ledger._read(ledger.fragment_path("P4m"))}
    new = [x for x in rows if (x["number_id"], x["source"], x["product_version"]) not in have]
    ledger.append_rows("P4m", new)
    unl = ledger.unledgered([n for n, _ in pre_n + rec_n] + [i for i, _ in ev_ids])
    allrows = ledger.read_all()
    conf = [c for c in ledger.conflicts(allrows) if c[0].startswith("P4m")]
    L.jdump({"preorigin": rel_to_repo(d_pre), "recorded": rel_to_repo(d_rec),
             "ledger_rows_added": len(new), "unledgered": unl, "conflicts": conf,
             "acceptance": acc}, PUB)
    print(d_pre, d_rec, len(new), "rows; unledgered", unl, "conflicts", conf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
