"""Pre-registration ``P5.signatures.v1`` (harness.prereg3), written before any P5 test runs.

Usage: prereg_p5.py WORKDIR [--write]   (without --write: validate and probe the look log only)

The registration pins the test code (``p5stats.py``, ``t_run.py``, ``config.py``), the
extracted tables the tests read (stage 1-5 outputs in WORKDIR, none of which compares an index
with electricity or with the pilot) and the pilot table T1 compares with.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import rel_to_repo  # noqa: E402
from harness import prereg3  # noqa: E402

from p5_head_signatures import config as C  # noqa: E402
from p5_head_signatures import inputs as I  # noqa: E402

PREREG_ID = "P5.signatures.v1"
PRE = f"{C.PRE_MONTHS[0]}..{C.PRE_MONTHS[1]}"
EPOCHS = f"{C.EPOCH_HOURLY[0]}-01..{C.EPOCH_10MIN[1]}-12"


def unit_lists(work: Path) -> dict:
    F = pd.read_parquet(work / "well_month_features.parquet", columns=["sid", "month"])
    km = pd.read_parquet(work / "wells_kwh_meta.parquet")
    meta = I.station_meta()
    km = km.merge(meta[["sid", "layer_flag"]], on="sid", how="left")
    be = I.be_table()
    have = set(F.loc[F.month <= C.ORIGIN, "sid"])

    def shallow(cty):
        k = km[(km.county == cty) & km.layer.isin(list(C.LAYERS_SHALLOW)) & (km.layer_flag == 0)
               & (km.sectors3 >= C.SECTORS_MIN)]
        return sorted(set(k.sid) & have)
    yl = shallow("Yunlin")
    ch = shallow("Changhua")
    within = sorted(set(km[km.county.isin(["Yunlin", "Changhua"])
                           & (km.sectors3 >= C.SECTORS_MIN)].sid) & have)
    ep = sorted(set(F.loc[(F.month >= f"{C.EPOCH_HOURLY[0]}-01")
                          & (F.month <= f"{C.EPOCH_10MIN[1]}-12"), "sid"]))
    bw = sorted(set(be.loc[be.accepted.astype(bool), "sid"].astype(str)) & have)
    return {"yl": yl, "ch": ch, "within": within, "epoch": ep, "baro": bw}


def build(work: Path) -> dict:
    u = unit_lists(work)
    ext = {
        "P5:extractor_epoch_wells": {
            "description": "every well with a P5 index month in 2016-01..2021-12 (extractor "
                           "reproduction against the head-signature pilot and the pilot's "
                           "hourly-vs-10-minute epoch check)",
            "ids": u["epoch"], "window": EPOCHS},
        "P5:baro_wells": {
            "description": "wells with an accepted H1a barometric response (L1 v3 "
                           "barometric_efficiency.csv), pre-origin O1 phases",
            "ids": u["baro"], "window": PRE},
        "P5:yunlin_l12": {
            "description": "Yunlin layer-1/2 wells (layer_flag 0) with census coverage "
                           "(sectors3 >= 7): cross-well head-index vs local census kWh; the "
                           "head-signature pilot computed this relation on (most of) these "
                           "wells, so results are design-time replications",
            "ids": u["yl"], "window": PRE},
        "P5:changhua_l12": {
            "description": "Changhua layer-1/2 wells (layer_flag 0) with census coverage: "
                           "head index vs the Changhua census (obtained 2026-10-12, never "
                           "related to these wells' head indices before this registration)",
            "ids": u["ch"], "window": PRE},
        "P5:within_wells": {
            "description": "Yunlin and Changhua wells with census coverage: month-to-month "
                           "anomaly correlations of head indices with local census kWh",
            "ids": u["within"], "window": PRE},
    }
    seasonal_ids = sorted(set(u["yl"]) | set(u["ch"]))
    ext["P5:seasonal_cv_l12"] = {
        "description": "Yunlin and Changhua layer-1/2 wells: seasonal CV of daily-cycle "
                       "indices against the census seasonal CV (alpha)",
        "ids": seasonal_ids, "window": PRE}
    claim_rule = ("per fold k of GroupKFold(5) over 5 km blocks: R2_k (1 - SSE/SS around the "
                  "test fold's mean) > the 95th percentile of R2_k over 1000 permutations of "
                  "the feature across wells (same folds, layer dummy kept) AND R2_k > the "
                  "R2_k of the same model fitted to the placebo target (median census target "
                  "of the census-covered wells 15-25 km away); claim supported iff >= 4 of 5 "
                  "folds pass (plan WP P5 acceptance)")
    model = ("OLS log10(mean monthly E_r2 + 1, 2012-01..2022-12) ~ 1 + g(well median of the "
             "index over pre-origin months, >= 24 months) + [layer == 2]; g = log10(max(x, "
             "1e-4)) for amplitude indices (A_S1c_strain, dr_med, eemd_annual_amp_m), identity "
             "for signed ones (DN, tf_pump_gain, eof_pump_loading)")
    tests = [
        {"test_id": "P5.t1a_repro", "data_scope": f"heads:{EPOCHS}",
         "fresh_units": ["P5:extractor_epoch_wells"],
         "external_units": {"P5:extractor_epoch_wells": ext["P5:extractor_epoch_wells"]},
         "metric": "Spearman across wells between the pilot's well medians "
                   "(_pilots/head_pumping_signatures/out/well_month_features.parquet) and P5's "
                   "uncorrected-variant well medians (dr_med_raw, A_S1c_strain_raw), "
                   "separately in the hourly epoch 2016-2018 and the 10-minute epoch "
                   "2019-2021; wells with >= 6 months in both tables in the epoch; the "
                   "corrected-variant Spearman is reported, not gated",
         "threshold": "each of the 4 Spearman values >= 0.95"},
        {"test_id": "P5.t1b_epoch", "data_scope": f"heads:{EPOCHS}",
         "fresh_units": ["P5:extractor_epoch_wells"],
         "external_units": {"P5:extractor_epoch_wells": ext["P5:extractor_epoch_wells"]},
         "metric": "the pilot's epoch check on the barometrically corrected product: Spearman "
                   "across wells between 2016-2018 and 2019-2021 well medians of dr_med and of "
                   "A_S1c_strain (>= 6 months per epoch); uncorrected variant reported",
         "threshold": "both Spearman values >= 0.95"},
        {"test_id": "P5.t2_baro", "data_scope": f"heads:{PRE}",
         "fresh_units": ["P5:baro_wells"],
         "external_units": {"P5:baro_wells": ext["P5:baro_wells"]},
         "metric": "O1 phase coherence across pre-origin months (|mean exp(i gO1)|, months "
                   "with harmonic fits, >= 12 months) of the corrected minus the uncorrected "
                   "variant, paired per well with an accepted barometric response; the H1a "
                   "check C6 (no flagged object on a >= 20 hPa fall) is inherited because P5's "
                   "sample set equals L1 v3's",
         "threshold": "median paired difference > 0 (H1 acceptance)"},
        {"test_id": "P5.t3_yunlin_claims", "data_scope": f"heads+census:{PRE}",
         "fresh_units": ["P5:yunlin_l12"],
         "external_units": {"P5:yunlin_l12": ext["P5:yunlin_l12"]},
         "metric": f"{model}; features A_S1c_strain (primary), DN, dr_med; {claim_rule}; "
                   "reported: pooled CV R2, township leave-one-out R2, partial r given the "
                   "placebo target and layer, Changhua-sized masked half (split at the median "
                   "northing, both directions)",
         "threshold": "claim supported iff >= 4 of 5 folds pass (each feature reported; "
                      "primary A_S1c_strain)"},
        {"test_id": "P5.t4_changhua", "data_scope": f"heads+census:{PRE}",
         "fresh_units": ["P5:changhua_l12"],
         "external_units": {"P5:changhua_l12": ext["P5:changhua_l12"]},
         "metric": "(a) the same claim rule inside Changhua; (b) transfer: model fitted on "
                   "all Yunlin layer-1/2 wells, R2 on Changhua wells against their own mean, "
                   "vs the 95th percentile of 1000 permutations of the Changhua feature and "
                   "vs the placebo-target transfer R2; (c) rank check: Spearman of the index "
                   "with log local kWh in Changhua, one-sided permutation p (1000), compared "
                   "with Yunlin's (bootstrap 90 %, uncalibrated); (d) magnitude: the pilot's "
                   "implied Changhua/Yunlin ratio 0.81 of local kernel kWh vs the census "
                   "ratio of medians (reported); (e) sensitivity: target without the "
                   "domestic class",
         "threshold": "(a) >= 4 of 5 folds; (b) R2_transfer > perm95 and > placebo; (c) rank "
                      "signal present iff Spearman > 0 and p <= 0.05 (primary A_S1c_strain)"},
        {"test_id": "P5.t5_within", "data_scope": f"heads+census:{PRE}",
         "fresh_units": ["P5:within_wells"],
         "external_units": {"P5:within_wells": ext["P5:within_wells"]},
         "metric": "per well (>= 36 months), anomaly (calendar-month mean removed) Pearson r "
                   "of the index (mean level, DN, A_S1c_strain, dr_med) with local E_K0_1000 "
                   "(r_local), with the median over census-covered wells 15-25 km away "
                   "(r_placebo), with the fan in-grid total (r_regional), the local-only "
                   "partial r (fan total regressed out of both) and the median r over 200 "
                   "year-shuffles of the local kWh anomaly (null); by county",
         "threshold": "local signal beyond placebos iff, with the index's expected sign s "
                      "(level -1, others +1), the medians of s r_local_specific, "
                      "s(r_local - r_placebo), s(r_local - r_regional) and s(r_local - r_null) "
                      "are all > 0 with two-sided sign-test p < 0.05"},
        {"test_id": "P5.t6_seasonal_cv", "data_scope": f"heads+census:{PRE}",
         "fresh_units": ["P5:seasonal_cv_l12"],
         "external_units": {"P5:seasonal_cv_l12": ext["P5:seasonal_cv_l12"]},
         "metric": "per layer-1/2 well (>= 24 months): CV (population sd / mean) of the "
                   "12-value calendar-month climatology (median over years, >= 2 years per "
                   "month) of A_S1c_strain (primary; DN and dr_med reported); proxy = median "
                   "CV Changhua / median CV Yunlin with a 2000-draw well-cluster bootstrap "
                   "90 % interval (uncalibrated); truth = census alpha: CV of the 12-month "
                   "climatology (mean over years) of C0 in-grid irrigation kWh 2012-2022, "
                   "Changhua / Yunlin (all classes, no-domestic and local E_r2 around the same "
                   "wells reported); timing vs census reported",
         "threshold": "proxy consistent with the census iff the census irrigation alpha lies "
                      "inside the proxy's bootstrap 90 % interval; the P4.3 cut 0.75 reported"},
        {"test_id": "P5.t7_tf_claims", "data_scope": f"heads+census+forcing:{PRE}",
         "fresh_units": ["P5:yunlin_l12"],
         "external_units": {"P5:yunlin_l12": ext["P5:yunlin_l12"]},
         "metric": "tf_pump_gain (tf_gains.csv) as the feature of the claim rule in Yunlin "
                   "layer-1/2 wells; Changhua claim, transfer and rank checks reported",
         "threshold": "claim supported iff >= 4 of 5 folds pass (Yunlin)"},
        {"test_id": "P5.t8_eemd_claims", "data_scope": f"heads+census:{PRE}",
         "fresh_units": ["P5:yunlin_l12"],
         "external_units": {"P5:yunlin_l12": ext["P5:yunlin_l12"]},
         "metric": "eemd_annual_amp_m and eof_pump_loading (eemd_features.csv) as features of "
                   "the claim rule in Yunlin layer-1/2 wells; Changhua reported",
         "threshold": "claim supported iff >= 4 of 5 folds pass (Yunlin), per feature"},
    ]
    inputs = [rel_to_repo(C.CODE_DIR / f) for f in ("config.py", "p5stats.py", "t_run.py",
                                                     "inputs.py")]
    inputs += [rel_to_repo(work / f) for f in (
        "well_month_features.parquet", "local_kwh.parquet", "wells_kwh_meta.parquet",
        "regional_kwh.parquet", "tf_gains.csv", "eemd_features.csv", "eof_modes.json")]
    inputs += [rel_to_repo(C.PILOT_DIR / "out" / "well_month_features.parquet"),
               rel_to_repo(C.L1_DIR / "MANIFEST.json"),
               rel_to_repo(C.C0_PRE_DIR / "MANIFEST.json"),
               rel_to_repo(C.C0_CENSUS_DIR / "MANIFEST.json")]
    doc = {
        "id": PREREG_ID, "wp": C.WP, "kind": "test",
        "title": "WP P5: extractor reproduction, barometric step, head-derived intensity "
                 "claims (Yunlin, Changhua census truth), within-well placebos, seasonal-CV "
                 "proxy for alpha, transfer-function and EEMD/EOF benchmarks",
        "plan_section": "docs/DATA_REMEDIATION_PLAN_2026-10.md WP P5, 2.4-2.5, 4.1-4.7; task "
                        "update 2026-10-12 (Changhua census obtained)",
        "tests": tests,
        "zones": prereg3.not_applicable("well-level tests; no zones"),
        "priors": prereg3.not_applicable("no priors: P5 indices are validators, never priors"),
        "burn_in": prereg3.not_applicable("no model run; transfer functions use a 36-month "
                                          "climatology warm-up fixed in s4_tf.py"),
        "folds": "GroupKFold(5) over 5 km blocks (floor(x/5 km), floor(y/5 km)); township "
                 "leave-one-out and a Changhua-sized masked half (median northing) reported",
        "fresh_units": sorted(ext),
        "external_units": ext,
        "design_time_units": [
            "_pilots/head_pumping_signatures (scout, 2026-10-12): Yunlin cross-well CV R2 "
            "0.11-0.22 for layers 1-2 against a permutation p95 <= 0.02 (cv_shallow), "
            "within-well and placebo medians (placebo.log), the epoch check 0.968 / 0.955 "
            "(extras.log), calendar-month medians of normalised S1c/DN/dr by county and layer "
            "(clim_spring.log; read by this session before registration, nothing computed "
            "from them), Changhua S1c/dr medians and the implied ratio 0.81 (cv_shallow.log)",
            "C0 census_L1/preorigin v3 (2026-10-13): Changhua/Yunlin in-grid irrigation "
            "seasonal CV 0.141 / 0.234 (alpha about 0.60), published in the C0 report and the "
            "task text, read before registration",
            "P5 build checks before registration: row counts, wells per period/county, the "
            "day-by-day construction check against L1 v3, census coverage (sectors3) "
            "distribution; no index was related to electricity or to the pilot"],
        "thresholds": {t["test_id"]: t["threshold"] for t in tests},
        "gate_scores": [
            {"name": "P5.t1a_repro", "weight": 0.4,
             "rule": "head_signatures/v1 gate: t1a and t1b and t2 all pass (the plan's P5 "
                     "acceptance: extractor reproduction and the H1 barometric acceptance)"},
            {"name": "P5.t1b_epoch", "weight": 0.3, "rule": "as above"},
            {"name": "P5.t2_baro", "weight": 0.3, "rule": "as above"}],
        "gate_rule": "The product gate is the conjunction of t1a, t1b and t2. t3-t8 decide which "
                     "head-derived claims P5 may make (plan: 'acceptance for any head-derived "
                     "intensity claim'); they do not gate the feature table. Deep learning "
                     "(plan step 5) runs iff a t7 or t8 Yunlin claim is supported (local "
                     "signal beyond placebo in steps 2-3); otherwise it is not run and the "
                     "report says so.",
        "primary_model": f"{model}; primary feature A_S1c_strain (barometrically corrected "
                         "S1 with the strain admittance), primary target E_r2",
        "reported_not_gated": [
            "corrected-vs-pilot Spearman (t1a), uncorrected epoch check (t1b)",
            "all-wells O1 coherence and the loading's S1 amplitude (t2)",
            "pooled CV R2, township LOO R2, partial r, masked half (t3)",
            "magnitude ratio and no-domestic sensitivity (t4)",
            "census alpha for all classes / no domestic / local kWh; timing (t6)",
            "Changhua TF and EEMD/EOF results (t7, t8)"],
        "multiple_comparisons": "Yunlin claims: 3 daily-cycle features (t3) + 1 TF (t7) + 2 "
                                "EEMD/EOF (t8) = 6; Changhua: the same 6 plus transfer and "
                                "rank checks; the primary is A_S1c_strain",
        "seeds": {"base": C.SEED, "rule": "seed = 2027 + 100 x test number + feature index"},
        "inputs": inputs,
        "interpretation": "The scout found that heads give Changhua pumping presence and timing, "
                          "not magnitude, and that only shallow daily-cycle indices carry a "
                          "cross-well signal. With the Changhua census now available, P5 tests "
                          "that verdict out of sample: Yunlin results replicate design-time "
                          "evidence (the pilot saw those wells), while every Changhua "
                          "comparison with the census is new. A supported claim needs the "
                          "plan's fold rule against both the permutation null and the "
                          "distant-kWh placebo; an unsupported claim is reported with its "
                          "numbers and P5 then provides benchmarks and validators only.",
        "u7": "every statistic uses rows dated 2012-01..2022-12 only; no 2023+ observation "
              "enters any gate, threshold or choice here",
    }
    return doc


def main():
    work = Path(sys.argv[1])
    doc = build(work)
    errs = prereg3.validate_prereg(doc)
    print("validation errors:", errs)
    idx = prereg3.membership_index()
    for _, ext in prereg3._external_dicts(doc):
        for u, v in ext.items():
            m = prereg3._declared_members(v)
            if m is not None and not idx.known(u):
                idx.members[idx.canon(u)] = m
    now = dt.datetime.now(dt.timezone.utc)
    block = prereg3.blocking_looks(doc["tests"], None, idx=idx, test_time=now)
    unver = prereg3.unverifiable_looks(doc["tests"], None, idx=idx)
    print("blocking looks:", json.dumps(block[:10], default=str)[:3000], len(block))
    print("unverifiable prior looks:", len(unver))
    print({u: len(v.get("ids", [])) for u, v in doc["external_units"].items()})
    if "--write" in sys.argv and not errs and not block:
        path, sha = prereg3.write_prereg(doc)
        print("written", path, sha)


if __name__ == "__main__":
    main()
