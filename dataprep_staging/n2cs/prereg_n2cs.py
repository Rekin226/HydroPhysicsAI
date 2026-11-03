"""Draft registration of N2CS (harness.prereg3 schema, kind 'test'). DRAFT ONLY: nothing here
writes a registration; the lead validates, runs the blocking probe and registers by
2026-11-30T16:00Z, after the code is merged and the OSSE has passed.

``draft_prereg_n2cs(osse_result, ...)`` refuses unless the OSSE passed G0 (otherwise the N line
is the bookkeeping text of ``n2cs_lib.choose`` and no test is registered). The primary form is
the OSSE's choice, never a later one.

    python -I dataprep_staging/n2cs/prereg_n2cs.py --validate     (needs the OSSE result)
    python -I dataprep_staging/n2cs/prereg_n2cs.py --probe        (read-only blocking probe)
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(HERE), str(REPO / "dataprep_staging" / "n2_policy"),
           str(REPO / "dataprep_staging")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bootstrap_scorer as BS  # noqa: E402
import n2cs_const as K  # noqa: E402

THIS = f"{K.X}/prereg_n2cs.py"
FROZEN = (f"{K.X}/n2cs_const.py", f"{K.X}/n2cs_lib.py", f"{K.X}/score_n2cs.py",
          f"{K.X}/osse_n2cs.py", f"{K.X}/build_r_county.py", f"{K.X}/run_n2cs.sh",
          K.SCORER, K.BUILD_R_BOTH, K.BUILD_R, K.REPLAY, K.SYNTH, K.GOV, K.E02C_SCORER)
MULTIPLICITY = (
    "the 74 Changhua wells' heads of 2026-12..2027-10 are also scored by N2C (the response "
    "verdict, same model R1c, same wells, a superset window), N2's Changhua report-only "
    "statistics, the free-running prospective_2026-11 test, I1.hybrid_tf T1/T2, X1.hybrid_tf_"
    "tpc T3a/T3b, E02c and E02d (the Changhua subset of the 158 wells); N2CS shares its model, "
    "wells and most months with N2C and is NOT independent of it; each test decides only its "
    "own claim, none is selected post hoc, and no family-wise or combined claim is made "
    "(N2CS's size class never strengthens or weakens N2C's verdict)")
U7 = ("2023+ never gates or designs: only the fresh 2026-12..2027-10 Changhua heads score; "
      "2023-2027 X1 electricity enters only as the recorded forcing of the two scoring arms "
      "(X1 allowed use forcing_recorded_arm); the OSSE's forcing variability is the pre-origin "
      "census (pump_electricity/preorigin/v3, design_time_evidence) and its noise parameters "
      "are N2C-reading's published design-time values from spent pre-origin heads; no "
      "post-2022-12 observation entered the forms, the choice, the thresholds or the design")


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rp(s) -> Path:
    return Path(s) if Path(s).is_absolute() else REPO / s


def unit_ids(prereg_dir: Path | None = None) -> list[str]:
    """N2C's 74 Changhua ids, from its registration (the same wells, a new unit id)."""
    d = Path(prereg_dir) if prereg_dir else rp("results/dataprep/preregistration")
    p = d / f"{K.N2C_ID}.json"
    if sha256(p) != K.N2C_SHA:
        raise SystemExit(f"{p} is not N2C's registered file")
    doc = json.loads(p.read_text(encoding="utf-8"))
    ids = sorted(doc["tests"][0]["external_units"][K.N2C_UNIT]["ids"])
    if len(ids) != K.N_WELLS:
        raise SystemExit(f"N2C lists {len(ids)} wells, expected {K.N_WELLS}")
    return ids


def _na(reason: str) -> dict:
    return {"not_applicable": reason}


def _inputs() -> list[str]:
    paths = [f"results/dataprep/preregistration/{K.N2C_ID}.json", K.RULINGS,
             f"results/dataprep/preregistration/{K.OSSE_DECL_ID}.json",
             f"{K.OSSE_OUT}/result.json", K.PANEL_RECORD, K.PANEL, f"{K.REPLAYS}/replays.json",
             K.UNIT_CH, K.UNIT_YL, K.COUNTY_MAP, K.WELLS, K.WELLS_XY, K.R1C_THETA,
             K.R1C_COLUMN, f"{K.ZR_DIR}/predictions.csv", *FROZEN]
    return list(dict.fromkeys(paths))


def draft_prereg_n2cs(osse_result: dict, *, osse_result_sha256: str, osse_declaration_sha256: str,
                      ids: list[str] | None = None, check_files: bool = True) -> dict:
    """The draft registration. ``osse_result`` is the OSSE's result.json (n2cs_lib.run_osse);
    it must have passed G0. ``check_files`` hashes the frozen code into ``frozen_code`` (and
    refuses if the GOV module or E02c's scorer is not at its registered sha256)."""
    ch = (osse_result or {}).get("choice") or {}
    if osse_result.get("engineering_failure"):
        raise ValueError("the OSSE had an engineering failure: fix and rerun before drafting")
    if not ch.get("passed"):
        raise ValueError("the OSSE did not pass G0; no test is registered. N line: "
                         + str(ch.get("n_line")))
    if check_files:
        for f, want in ((K.GOV, K.GOV_SHA), (K.E02C_SCORER, K.E02C_SCORER_SHA)):
            if sha256(rp(f)) != want:
                raise ValueError(f"{f} is not at its registered sha256 {want[:8]}")
    form = ch["form"]
    if form not in K.FORMS:
        raise ValueError(f"unknown form {form!r}")
    gate = next(g for g in osse_result["gates"] if g["form"] == form)
    ids = sorted(ids) if ids is not None else unit_ids()
    if len(ids) != K.N_WELLS:
        raise ValueError(f"expected {K.N_WELLS} ids")
    unit = {K.UNIT: {"description": "monthly heads 2026-12..2027-10 (flags 0-1) at N2C's 74 "
                                    "Changhua wells (station code 07), a new unit id (E02d / "
                                    "N3F December precedent)",
                     "ids": ids, "window": f"{K.WINDOW[0]}..{K.WINDOW[1]}"}}
    power = ch["osse_power_X"]
    test = {
        "test_id": K.TEST, "data_scope": K.SCOPE, "fresh_units": [K.UNIT],
        "external_units": unit,
        "label": "prospective: the window's heads did not exist at registration",
        "what": "Is R1c's first-year Changhua head response to Changhua's own recorded pumping "
                "anomaly overstated by a factor of two or more?",
        "metric": {
            "name": "beta_CH",
            "definition": "e = obs - ZR over the well-months of the 74 Changhua wells in "
                          "2026-12..2027-10 (flags 0-1; ZR = N2C's registered R1c predictions "
                          "with their per-well datum); c_CH = heads_mean[NCSCH] - "
                          "heads_mean[baseline] and c_YL = heads_mean[NCSYL] - "
                          "heads_mean[baseline] of ONE CPU forward of R1c's main set at each "
                          "well's cell and layer; wells with >= 6 scored months; the "
                          f"registered form '{form}': {K.FORM_TEXT[form]}; the Changhua "
                          "coefficient is the estimand",
            "interval": K.INTERVAL, "form": form,
            "report_only": ["the other two pre-declared forms", "beta_YL",
                            "the 2 x 2 refit interval (re-partialling per draw)",
                            "the single-contrast per-well-demeaned slope of e on c_CH"]},
        "decision_rule": {
            "overstated_x2": K.READING_TEXT["overstated_x2"],
            "not_overstated_x2": K.READING_TEXT["not_overstated_x2"],
            "inconclusive": K.READING_TEXT["inconclusive"],
            "not_informative": K.READING_TEXT["not_informative"],
            "not_evaluable": K.READING_TEXT["not_evaluable"],
            "one_sided": "the claim at stake is the size class 'overstated >= x2' versus "
                         "'not overstated by x2'; the bounds are the 5 % and 95 % quantiles",
            "wording": K.WORDING},
        "informative_gate": {
            "rule": "evaluated from the model and the forcing only, before any window head is "
                    f"read: under the registered form '{form}' over every well and window "
                    "month, RMS of the Frisch-Waugh regressor (c_CH~ residualised on c_YL~) >= "
                    f"{K.INFORMATIVE_RMS_M} m, |corr(c_CH~, c_YL~)| <= {K.MAX_ABS_CORR} as an "
                    f"additional check, >= {BS.MIN_CLUSTERS} 5 km clusters and >= "
                    f"{BS.MIN_MONTHS} window months with a recorded Changhua X1 r; otherwise "
                    "'not informative' and no reading",
            "osse": "the same gate on each replay's contrast, per form"},
        "heads_contract": K.HEADS_CONTRACT,
        "seed": BS.SEED}
    doc = {
        "id": K.ID, "wp": K.WP, "kind": "test",
        "title": "N2CS: prospective one-sided Changhua first-year anomaly-response size class "
                 "of R1c (two-contrast twoway regression), 2026-12..2027-10, OSSE-gated",
        "why": "N2C decides R1c's Changhua response verdict with a single contrast; its size "
               "clause cannot separate Changhua's own response from Yunlin spill-over. N2CS "
               "adds a second, one-sided size-class test on the same wells (merges P1-A, "
               "P2-R-A and P3-R1), decisive by design (OSSE G0 passed).",
        "tests": [test], "primary_form": form,
        "forms_declared": {f: K.FORM_TEXT[f] for f in K.FORMS},
        "osse": {"declaration": K.OSSE_DECL_ID, "declaration_sha256": osse_declaration_sha256,
                 "result": f"{K.OSSE_OUT}/result.json", "result_sha256": osse_result_sha256,
                 "choice": ch, "gate_of_chosen_form": gate, "g0": K.G0,
                 "choice_rule": K.CHOICE_RULE, "world_seed": osse_result.get("seed")},
        "code": {"commit": K.WT_HEAD, "worktree": K.WT,
                 "forward_command": "HYDROMIND_GW_DATA=<repo>/chou-shui-data/data PYTHONPATH="
                                    "<checkout of 1b08fb6> python -P -m hydrophysics.twin."
                                    f"forward --theta {K.R1C_THETA} --vep-json {K.R1C_COLUMN} "
                                    f"{K.FWD_FLAGS} --recorded-pumping 'NCSCH=<scoring dir>/"
                                    "r_changhua.npz@2023-01' --recorded-pumping 'NCSYL="
                                    "<scoring dir>/r_yunlin.npz@2023-01' --out <scoring dir>/"
                                    "forward_ncs_r1c",
                 "r_builder": f"python -I {K.X}/build_r_county.py x1 --x1 <X1 version>/"
                              "county_monthly.parquet --first 2023-01 --scope changhua|yunlin "
                              "--out <scoring dir>/r_<scope>.npz (N2C's rule imported from the "
                              "frozen build_r_both.py)",
                 "runner": f"bash {K.X}/run_n2cs.sh score-files / forward-score",
                 "engineering": "CPU forwards of the frozen 1b08fb6 checkout; the scoring "
                                "baseline must equal OSSE forward A's within 1e-6 m at the 74 "
                                "screens",
                 "x1_version": "electricity_x1_tpc/full/v1 or the first later version "
                               "extending it with the same frozen model (N2C's rule); the "
                               "version used is hashed at scoring"},
        "pregate": {"rule": "both unit arms (multiplier exactly 1) reproduce the run's "
                            f"baseline (ZR arm) within {K.PREGATE_TOL_M} m at every layer, "
                            "cell and month, and forwards A and B agree", "record":
                    K.PANEL_RECORD},
        "frozen_code": ({f: sha256(rp(f)) for f in FROZEN} if check_files else {}),
        "code_report_only": ({K.GOV: sha256(rp(K.GOV))} if check_files and rp(K.GOV).exists()
                             else {}),
        "scoring": {
            "not_before": K.NOT_BEFORE, "no_later_than": K.NO_LATER_THAN,
            "order": "in the 2028-01-15 session, AFTER N2C's step (2) (its report written); "
                     "the scorer refuses without N2C's report; then: GOV precondition, the "
                     "informative gate, prereg3.assert_can_score, the statistic and the report",
            "if_order_breached": "scored once whatever; a session after 2028-01-31 or before "
                                 "N2C step (2) is flagged beside the reading, and a breached "
                                 "reading is reported, not shown",
            "precondition": "N2CS's own, from the MANIFEST alone: E02c's frozen "
                            f"score_e02c.product_grants_scorer (sha {K.E02C_SCORER_SHA[:8]}) AND "
                            f"{K.GOV_ID}'s public heads_window.covering_rule(m, "
                            f"{list(K.WINDOW)}) (sha {K.GOV_SHA[:8]}); both files in "
                            "frozen_code and checked at scoring; otherwise not_evaluable, "
                            "nothing read, no look",
            "n2c_step2": f"{K.N2C_REPORT} exists and its 'verdict' is one of "
                         f"{list(BS.VERDICTS)}; otherwise REFUSED",
            "after_look": "any exception after assert_can_score writes the final report "
                          "'engineering_failure_after_look' (disclosed, never re-scored)",
            "scorer": f"{K.X}/score_n2cs.py score --scoring-dir <dir> --window-product <dir> "
                      "--heads-table <name> --n2c-report <N2C report.json>",
            "write_once": "report.json mode 'x'; attempts in results/twin_runs/n2cs/"
                          "attempts.jsonl"},
        "page_mapping": {
            "before_scoring": {"C20": "size clause N -> N+D: 'size class decided by a "
                                      "registered test (N2CS) decisive by design (OSSE power "
                                      f"{power:.2f}); scored not before 2028-01-15'",
                               "C21": "same as C20", "E27": "new row, status D",
                               "README.md": "Limits: the Changhua size clause carries the C20 "
                                            "text", "MODEL_CARD.md": "policy-response heads: "
                                                                     "the size clause carries "
                                                                     "the C21 text"},
            "after_scoring": "E27 and the C20/C21 size clause carry the reading's text "
                             "(decision_rule) with beta_CH and its 90 % interval and the "
                             f"wording '{K.WORDING}', in the ledger, README.md and "
                             "MODEL_CARD.md",
            "never": ["no correction factor follows (ruling 2026-10-29 (4))",
                      "no Changhua head lever is shown by any reading",
                      "C13a, E20, N2C's clauses and the F < 0.5 cap are untouched",
                      "no Changhua subsidence, absolute pumping volume or page-model claim"]},
        "rulings": {K.RULING_ID: {"file": K.RULINGS, "sha256": K.RULINGS_SHA,
                                  "decision": "approved, 11-month window only; the 24-month "
                                              "variant declined"},
                    K.RULING_FACTOR: "no correction factor follows"},
        "disclosures": {
            "multiplicity": MULTIPLICITY, "U7": U7,
            "design_time": "the replays read pump_electricity/preorigin/v3 (logged design look "
                           "N2CS:v3_county_class_monthly_2012_2022); R1c's published "
                           "design-time Changhua slopes (I1.CHX.v1 0.17 [0.02, 0.44] on "
                           "2013-22 screens, I1.CH_backcast.v1 on 2010-11, N1 step C 0.85) "
                           "were known to the builder from the ledger; they set no form, "
                           "threshold or choice (the OSSE chose the form by its frozen rule)",
            "not_independent_of_n2c": "same model, same wells, 11 of N2C's 12 months",
            "osse_limits": "pre-origin census anomalies stand in for the 2023-27 recorded "
                           "anomalies; G0 uses k = 1 and k = 1/3 only; the gate variants are "
                           "the full panel (uniform and shaped confounder), a 25 % thinned "
                           "panel and a 15-of-74 whole-well dropout (shaped confounder)",
            "uniform_confounder": "the county-uniform confounder is absorbed exactly by the "
                                  "month effects of every form, by construction (as any real "
                                  "county-uniform term is); the binding confounder test is "
                                  "the shaped one",
            "intervals": "the primary bootstrap holds the Yunlin partialling fixed across "
                         "draws (the refit interval, re-partialling per draw, is "
                         "report-only); both hold the month effects fixed; power may depend "
                         "on the true Yunlin coefficient (report cells k_YL = 3)",
            "winners_curse": "the form is the most powerful of three on common worlds; its "
                             "power is optimistic by about the Monte Carlo SE (about 0.009 "
                             "at 0.8 with 2000 replicates)",
            "design_device": "the replays use twin.forward's recorded-pumping mechanism for "
                             "synthetic future months under the census product's allowed use "
                             "design_time_evidence (its MANIFEST forbids forcing_recorded_arm, "
                             "which is not the use made here) (X5.n2c_reading.v1's wording)",
            "census_source": "pump_electricity/preorigin/v3 (not model_ready/"
                             "electricity_census/v1, which X5.n2c_reading used)",
            "v3_exemption_vs_u7": "v3's 2026 census attribute table includes pumps installed "
                                  "after the origin (Yunlin 44, Changhua 34; in-grid 2012-2022 "
                                  "energy <= 3.7e-4 relative; C0.v3.exemption.census_"
                                  "attributes.preorigin), allowed for design_time_evidence: "
                                  "a post-origin attribute input to the OSSE design",
            "smoke_runs": "the builder ran the engine's smoke mode on synthetic stand-in "
                          "panels (no model output); no parameter changed after them",
            "pre_look_contract_read": "the scorer reads the heads table's station_id, date "
                                      "(and flag) columns to check the contract seconds before "
                                      "assert_can_score; no observed_m value is read before it"},
        "attempt_policy": "engineering failures may be fixed and re-run before the score look, "
                          "each logged; scored ONCE; a not-informative, not-evaluable or "
                          "inconclusive result is reported as such and never re-scored on this "
                          "unit; at most one post-failure exploratory follow-up (new ids, "
                          "pre-declared alternatives only, never validation); every attempt "
                          "is listed",
        "zones": _na("no fit"), "priors": _na("no prior enters the statistic (see "
                                              "disclosures.design_time)"),
        "burn_in": _na("no fit"), "folds": _na("prospective window"),
        "fresh_units": [K.UNIT],
        "thresholds": {"size_x": K.SIZE_X, "level": BS.LEVEL, "max_abs_corr": K.MAX_ABS_CORR,
                       "informative_head_m": K.INFORMATIVE_RMS_M,
                       "min_clusters": BS.MIN_CLUSTERS, "min_months": BS.MIN_MONTHS,
                       "min_well_months": K.MIN_WELL_MONTHS},
        "gate_scores": _na("no weighted gate; the reading is reported whichever way it falls"),
        "primary_model": {"name": "R1c: stage3_datum_gate_ccz main parameter set + its "
                                  "coupled_leveling column (N2C's primary model)",
                          "theta_sha256": K.R1C_THETA_SHA, "column_sha256": K.R1C_COLUMN_SHA,
                          "status": "failed rung (I1.R1c, G2), disclosed as in N2C"},
        "inputs": _inputs(),
        "interpretation": "R1c's first-year Changhua head-response size class to Changhua's "
                          "own recorded pumping anomaly, net of the Yunlin contrast, on fresh "
                          "months; " + K.WORDING + ". Nothing about the page model, subsidence "
                          "or absolute pumping.",
        "written_by": THIS}
    return doc


def validate(doc: dict) -> list[str]:
    from harness import prereg3

    return prereg3.validate_prereg(doc)


def blocking_probe(doc: dict) -> list:
    """Read-only: the looks that would block this draft now (write_prereg's own check)."""
    from harness import prereg3

    idx = prereg3.membership_index()
    for t in doc["tests"]:
        for u, v in (t.get("external_units") or {}).items():
            m = prereg3._declared_members(v)
            if m is not None and not idx.known(u):
                idx.members[idx.canon(u)] = m
    return prereg3.blocking_looks(doc["tests"], None, idx=idx,
                                  test_time=dt.datetime.now(dt.timezone.utc))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--probe", action="store_true")
    a = ap.parse_args(argv)
    res_p = rp(K.OSSE_OUT) / "result.json"
    rec_p = rp(f"results/dataprep/preregistration/{K.OSSE_DECL_ID}.json")
    if not res_p.exists():
        raise SystemExit("REFUSED: no OSSE result (the OSSE runs before the draft)")
    doc = draft_prereg_n2cs(json.loads(res_p.read_text()), osse_result_sha256=sha256(res_p),
                            osse_declaration_sha256=sha256(rec_p))
    missing = [p for p in doc["inputs"] if not rp(p).exists()]
    if missing:
        print(json.dumps({"inputs_missing": missing}, indent=1))
    if a.validate:
        print(json.dumps({"errors": validate(doc)}, indent=1))
    if a.probe:
        print(json.dumps({"blocking": blocking_probe(doc)}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
