"""Ledger rows of WP P5 (plan 4.1, U6): one role per source per reported number.

Usage: ledger_p5.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ledger  # noqa: E402

PV = "head_signatures/v1"
L1 = "results/dataprep/heads/L1_clean/v3"
L0H = "AMP_V2/data/wells"
L0N = "data_fetch_api/new_measurements/heads_202210_202610"
C0P = "results/dataprep/pump_electricity/preorigin/v3"
C0C = "results/dataprep/pump_electricity/census_L1/v3"
PIL = "results/dataprep/_pilots/head_pumping_signatures"
HS = "results/dataprep/head_signatures/v1"
ET0 = "results/dataprep/et0/v1"
W1 = "results/dataprep/surface_water_v4/preorigin/v2"
RAIN = "chou-shui-data/data/rf_timeseries.csv"
PRE = "results/dataprep/preregistration/P5.signatures.v1.json"


def rows_for(number: str, roles: dict, note: str) -> list[dict]:
    return [{"number_id": number, "source": s, "role": r, "product_version": PV, "note": note}
            for s, r in roles.items()]


def main():
    R = []
    heads_in = {L1: "input", L0H: "input", L0N: "input"}
    R += rows_for("P5.build.construction_check", {L0H: "input", L0N: "input", L1: "validator"},
                  "daily mean of P5's kept samples vs L1 v3 value - corr_m: 1,025,861 days, 2 "
                  "> 1 mm, max 4.3 mm, 15 L1 days not covered")
    R += rows_for("P5.build.coverage", heads_in,
                  "34,026 well-months, 267 wells (158 post-origin), 7,315 flag-1 rows")
    R += rows_for("P5.t1a.repro_spearman", heads_in | {PIL: "validator", PRE: "input"},
                  "Spearman of well medians vs the pilot (uncorrected): 0.9999/0.9999 "
                  "(2016-18), 1.0000/0.9997 (2019-21)")
    R += rows_for("P5.t1b.epoch_spearman", heads_in | {PRE: "input"},
                  "epoch check on the corrected product: dr_med 0.972, S1c 0.965")
    R += rows_for("P5.t2.o1_coherence", heads_in | {PRE: "input"},
                  "O1 coherence paired median +0.0023 (50/18 of 68), loading S1 median 0.5 mm")
    R += rows_for("P5.t3.yunlin_claims", {HS: "input", C0P: "fit_target", C0C: "input",
                                          PIL: "design_time_evidence", PRE: "input"},
                  "Yunlin L1-2 claim rule: S1c 4/5 folds (pooled CV R2 0.204), DN 1/5, "
                  "dr_med 3/5; design-time replication")
    R += rows_for("P5.t4.changhua_claims", {HS: "input", C0P: "fit_target", C0C: "input",
                                            PRE: "input"},
                  "Changhua L1-2 claim rule: S1c 1/5, DN 1/5, dr_med 0/5")
    R += rows_for("P5.t4.transfer", {HS: "input", C0P: "validator", C0C: "input", PRE: "input"},
                  "Yunlin-trained models on Changhua: R2 -1.56 (S1c), -0.71 (DN), -1.85 (dr)")
    R += rows_for("P5.t4.rank_check", {HS: "input", C0P: "validator", C0C: "input",
                                       PRE: "input"},
                  "Changhua Spearman S1c 0.266 (p 0.022), DN 0.237, dr 0.290; Yunlin S1c 0.413")
    R += rows_for("P5.t4.magnitude", {HS: "input", C0P: "validator", C0C: "input",
                                      PIL: "design_time_evidence", PRE: "input"},
                  "pilot heads-implied Changhua/Yunlin ratio 0.81 vs census 1.03 (E_K0_1000)")
    R += rows_for("P5.t5.within_well", {HS: "input", C0P: "validator", C0C: "input",
                                        PRE: "input"},
                  "within-well anomaly r: level local -0.689 / placebo -0.654 / regional "
                  "-0.739 (Yunlin); only Yunlin S1c is local beyond all placebos")
    R += rows_for("P5.t6.seasonal_cv", {HS: "input", C0P: "validator", C0C: "input",
                                        PRE: "input"},
                  "seasonal-CV proxy 0.936 (boot90 0.778-1.070, uncalibrated) vs census "
                  "irrigation alpha 0.604: not consistent")
    R += rows_for("P5.t6.timing", {HS: "input", C0P: "validator", C0C: "input", PRE: "input"},
                  "Changhua index vs census irrigation climatology r 0.76 (S1c), 0.78 (DN)")
    R += rows_for("P5.tf.fits", {L1: "fit_target", ET0: "input", W1: "input", RAIN: "input",
                                 C0P: "input", C0C: "input"},
                  "221 wells, median EVP 0.87, 86 % positive pumping gain, 185 at a bound")
    R += rows_for("P5.t7.tf_claims", {HS: "input", C0P: "fit_target", C0C: "input",
                                      PRE: "input"},
                  "TF pumping gain: Yunlin 1/5 folds, Changhua 0/5, transfer R2 -1.02")
    R += rows_for("P5.eemd.eof_modes", {L1: "input", C0P: "input", C0C: "input",
                                        RAIN: "input"},
                  "EOF mode 0: 66 % of variance, r -0.83 with d(fan irrigation kWh)")
    R += rows_for("P5.t8.eemd_claims", {HS: "input", C0P: "fit_target", C0C: "input",
                                        PRE: "input"},
                  "EEMD annual amplitude 3/5 folds (negative relation), EOF loading 0/5")
    R += rows_for("P5.dl.decision", {HS: "input", PRE: "input"},
                  "deep learning not run: no t7/t8 Yunlin claim supported (registered rule)")
    p = ledger.append_rows("P5", R)
    print(p, len(R))


if __name__ == "__main__":
    main()
