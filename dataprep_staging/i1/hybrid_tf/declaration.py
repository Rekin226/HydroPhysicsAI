"""Task 3: write the DRAFT declaration of the prospective test (never registered here).

Writes, under ``results/dataprep/reports/I1_hybrid_tf/draft_declaration/``:

* ``I1.hybrid_tf.prospective_2026-11.v1.draft.json``: the pre-registration document in the
  ``harness.prereg`` schema, with the placeholders the registrar fills (``<...>``);
* ``register_draft.py``: the script that would register it with ``harness.prereg.write_prereg``
  after review (it refuses to run while a placeholder remains or once the window has begun);
* ``wells.csv``: the frozen well set (R0's 158 scored wells);
* ``validation.json``: the result of ``harness.prereg.validate_prereg`` and
  ``harness.prereg3.validate_prereg`` on the draft (a dry check; nothing is written to the
  pre-registration index).

Run from the repository root:
    PYTHONPATH=dataprep_staging dataprep_staging/tools/cpu_slot.sh \
        ~/miniforge3/envs/hydro/bin/python -m i1.hybrid_tf.declaration
"""
from __future__ import annotations

import json
import sys

from . import config as C
from . import scorer as S
from .prospective import code_hashes, hyperparameters, wells_frozen

sys.path.insert(0, str(C.REPO / "dataprep_staging"))

PREREG_ID = "I1.hybrid_tf.prospective_2026-11.v1"
UNIT = "I1:prospective_heads_2026-11_2027-10_R0wells"
WINDOW = "2026-11..2027-10"
DEADLINE_UTC = "2026-10-31T16:00:00Z"        # 2026-11-01 00:00 Asia/Taipei
SCORE_DEADLINE = "2028-01-31"
PLACEHOLDER = "<FILL AT REGISTRATION>"

REGISTER = '''"""Register the DRAFT {pid} with harness.prereg.write_prereg.

DO NOT RUN before the user has reviewed the draft. Steps, in order:

1. Freeze the code: commit dataprep_staging/i1/hybrid_tf/ (or copy it to a frozen path) and
   check that the sha256 of every file equals primary_model.code_sha256 in the draft.
2. Fit the frozen model once (reads heads 2012-02..2026-09; nothing in the code may change
   afterwards, whatever the fit shows):
       PYTHONPATH=dataprep_staging dataprep_staging/tools/cpu_slot.sh \\
           ~/miniforge3/envs/hydro/bin/python -c "from pathlib import Path; \\
           from i1.hybrid_tf import prospective as P; \\
           p = P.fit('full', '2026-09', Path('results/dataprep/reports/I1_hybrid_tf/frozen')); \\
           print(p['parameters_sha256'])"
3. Put that sha256 and the git commit in the draft where it says {ph}, add
   results/dataprep/reports/I1_hybrid_tf/frozen/parameters.json to "inputs", save.
4. Run this script before {deadline} (UTC). After that the window must move and the
   draft needs new test ids and scopes.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "dataprep_staging"))

from harness import prereg  # noqa: E402

DRAFT = Path(__file__).with_name("{pid}.draft.json")
DEADLINE = dt.datetime.fromisoformat("{deadline_iso}")


def main() -> None:
    text = DRAFT.read_text(encoding="utf-8")
    if "{ph}" in text:
        raise SystemExit("placeholders remain in the draft; fill them first (steps 1-3)")
    if dt.datetime.now(dt.timezone.utc) >= DEADLINE:
        raise SystemExit("the window 2026-11 has begun; move the window and re-draft")
    doc = json.loads(text)
    for k in [k for k in doc if k.startswith("draft_")]:
        doc.pop(k)
    path, sha = prereg.write_prereg(doc)
    print("registered", path, sha)


if __name__ == "__main__":
    main()
'''


def build() -> dict:
    wl = wells_frozen()
    ids = wl["sid"].tolist()
    hp = hyperparameters()
    t1 = {
        "test_id": "I1.hybrid_tf.T1_onestep_2026-11",
        "data_scope": f"heads:{WINDOW}",
        "fresh_units": [UNIT],
        "external_units": {UNIT: {
            "description": "monthly heads 2026-11..2027-10 (flags 0-1) at R0's 158 scored "
                           "wells; not fetched at drafting (the last fetch ends 2026-09-30)",
            "ids": ids, "window": WINDOW}},
        "role": "primary; decides the outcome",
        "setting": "one month ahead with recorded drivers: the forecast of month t uses the "
                   "heads up to t-1 and the drivers recorded for months up to t; any driver "
                   "month not recorded by the scoring deadline takes the frozen rule",
    }
    t2 = {
        "test_id": "I1.hybrid_tf.T2_recorded_sim_2026-11",
        "data_scope": f"heads:{WINDOW}",
        "fresh_units": [UNIT],
        "external_units": t1["external_units"],
        "role": "secondary; reported with the number of comparisons, never decides or "
                "promotes",
        "setting": "simulation with recorded drivers: the pure TF model (no head after "
                   "2026-09) on every window month, against the registered AR(1) layer "
                   "(results/twin/prospective_2026-11_hybrid), climatology and R0's registered "
                   "forecast",
    }
    gate = [{"name": "ratio_arx1_vs_stat_ar1_1:changhua", "weight": 0.5},
            {"name": "ratio_arx1_vs_stat_ar1_1:yunlin", "weight": 0.5}]
    thresholds = {
        "metrics": {
            "scored_well_months": "window months where the head and the previous month's head "
                                  "are both observed (flags 0-1) at a fitted well; wells with "
                                  f">= {S.MIN_MONTHS_PER_WELL} such months",
            "ratio_vs_X": "per county, sqrt(sum of squared errors of arx1 / sum of squared "
                          "errors of X) pooled over the scored well-months",
            "bootstrap": f"paired bootstrap over sites (wells of one site together), "
                         f"{S.N_BOOT} draws, seed {S.SEED}: 5th and 95th percentiles of the "
                         "ratio against stat_ar1_1",
        },
        "baselines": {
            "stat_ar1_1": "the existing AR(1) statistical layer one month ahead: frozen "
                          "climatology + rho * previous-month anomaly (same parameters file)",
            "persist1": "previous month's observed head",
            "clim": "frozen month-of-year climatology of the fit months (flags 0-1)",
            "r0_registered": "R0's registered forecast results/twin/prospective_2026-11/"
                             "predictions.csv (a seasonal free run from 2022-12; reported, "
                             "not deciding, because it is not a one-month-ahead forecast)",
        },
        "county_passes_if_all": [
            f"ratio_vs_stat_ar1_1 <= {S.THRESH_STAT}",
            "95th bootstrap percentile of ratio_vs_stat_ar1_1 < 1.00",
            "ratio_vs_persist1 < 1.00",
            "ratio_vs_clim < 1.00",
        ],
        "minimum_data": {
            "wells_per_county": S.MIN_WELLS,
            "months_per_well": S.MIN_MONTHS_PER_WELL,
            "window_months_with_at_least_100_wells": S.MIN_WINDOW_MONTHS,
            "if_not_met": "not scorable: no claim either way; the window is not extended",
        },
        "decision": [
            "Both counties pass: the ARX layer adds one-month-ahead skill to the AR(1) layer at "
            "monitored wells; it may replace the AR(1) layer for one-month-ahead well "
            "forecasts (a statistical claim at wells only: no physical, regional or policy "
            "claim).",
            "One county passes: reported as that county only; any use there needs the user's "
            "ruling.",
            "No county passes: no added skill; the AR(1) layer stays; the TF layer stays a "
            "diagnostic.",
            "Every ratio, both counties and T2 are reported whatever the outcome; Changhua and "
            "Yunlin are never pooled for the decision.",
        ],
        "threshold_basis": "fixed from design-time evidence on 2012-2022 only (U7): in the "
                           "forward-chaining mimic the ARX layer with every driver recorded "
                           "reached 0.92 (Changhua) and 0.96 (Yunlin) of the AR(1) layer's "
                           "RMSE; 0.97 asks for a 3 % gain. With the drivers a forecast "
                           "actually has (rain recorded; electricity, canal, river cut) it "
                           "reached 0.98 and 1.10, so the design predicts NO added skill unless "
                           "electricity is recorded for the window.",
        "schedule": f"score once heads 2026-11..2027-10 are fetched and built by the H1a L1 v3 "
                    f"code; drivers are taken as recorded on {SCORE_DEADLINE} (later data "
                    f"never enter); score no later than {SCORE_DEADLINE}",
    }
    driver_rules = {
        "recorded_month": {
            "electricity (3 km and county)": "release full/electricity_census or a later C0 "
                                             "full version whose manifest grants "
                                             "statistical_layer_history; month recorded if "
                                             ">= 99 % of active cells carry flags 0-5 "
                                             "(last now: 2026-06)",
            "rain recharge": "rain (twin cache, CODIS route of hydrophysics.twin.cwa_rain) and "
                             "ET0 (W2 et0 full or a later version, Open-Meteo); month recorded "
                             "if it has ended, >= 5 gauges have >= 90 % of days and ET0 is "
                             "complete (last now: 2026-09)",
            "canal (cell and district)": "W1 full; flags 0-5 only; flag 6 (the 2026 "
                                         "rain-driven extrapolation) is not a record (last "
                                         "now: 2025-12)",
            "river H057, H071": "W1 L1 daily yearbook gauges; month recorded with >= 20 daily "
                                "values (last now: 2025-12)",
            "public wells": "P6 member, 2007-01..2022-12 only",
        },
        "not_recorded": "month-of-year climatology of the driver over its recorded fit months "
                        "(frozen in parameters.json); public wells hold their last recorded "
                        "value. Decided before any score was read; the hold-last alternative "
                        "for every driver was far worse at design time (forward-chaining RMSE "
                        "1.3-2.6 x climatology)",
        "never_used": "census extension P2def (its manifest does not grant "
                      "statistical_layer_history); any driver product without a manifest "
                      "except the twin rain cache",
    }
    rules = {
        "U1": "Scoring targets and every baseline (climatology, persistence, the AR(1) rho and "
              "climatology) use observed heads with flags 0-1 only: heads are read through "
              "common.guard and filtered with filter_flags(use='baseline'); window heads with "
              "use scoring_target. No fill ever becomes a target.",
        "U2": "Fit targets are observed months, flags 0-1 (stricter than U2: no daily-bridged "
              "month), equally weighted.",
        "U3": "U3 governs models scored on 2023-26. This model is scored only on "
              "2026-11..2027-10 and no 2023-26 score of it is ever reported. Everything that "
              "designed it (driver set, filters, ridge alpha, guards, fill rules, lags, "
              "thresholds, well set) comes from 2012-2022 design-time evidence read through "
              "the pre-origin components, plus the inventory of driver dates. The full "
              "variants enter only as the statistical layer's history, the purpose U3 gives "
              "them.",
        "U4": "Post-origin forcing never feeds the physics model: R0's registered forecast is "
              "used as it was registered. The full driver variants feed a statistical layer "
              "under the manifest use statistical_layer_history. THIS READING NEEDS THE USER'S "
              "RULING (U4 says post-origin forcing never feeds 'the frozen or prospective "
              "forecasts'); if U4 forbids it, T1 cannot be run as drafted (see "
              "interpretation).",
        "U5": "Every product with a manifest is opened with common.guard.open_product and a "
              "declared use (prospective.SOURCES); the guard checks allowed_uses and "
              "fit_window_end.",
        "U6": "Each scored number gets ledger rows: heads 2026-11..2027-10 (scoring target), "
              "parameters.json and each driver product version (input), this registration "
              "(input). The design numbers' roles are in results/dataprep/reports/"
              "I1_hybrid_tf/data_roles.csv.",
        "U7": "No observation dated 2023-01 or later chooses anything: all choices above are "
              "fixed in this draft from 2012-2022 data. Heads 2023-01..2026-09 enter only as "
              "fit data (coefficients, climatology, rho) in one mechanical run of "
              "prospective.fit at registration; nothing may change after that run.",
    }
    fit_window = {
        "heads": f"{C.HEAD_START}..2026-09 (flags 0-1; release full/heads_observed, use "
                 "statistical_layer_history)",
        "drivers": f"{C.DRIVER_START}..2026-09 as recorded (rules above)",
        "why": "2026-09 is the last fetched head month, so the fit uses every head that exists "
               "in the repository and none of the window. It is the fit window of the "
               "registered AR(1) layer (fitted through 2026-09), so T1 compares like with "
               "like, and it keeps the climatology current (Yunlin heads trend). Fitting "
               "parameters on 2023-26 heads is not a selection (U7) and the model is never "
               "scored on 2023-26 (U3).",
        "fallback_if_U4_strict": "fit 2012-02..2022-12 on the pre-origin components only; "
                                 "drivers after 2022-12 replaced by the frozen rule. The "
                                 "model then reduces to the AR(1) layer plus a climatological "
                                 "TF term; design evidence says it would not beat the AR(1) "
                                 "layer, so registering it is not recommended",
    }
    doc = {
        "id": PREREG_ID,
        "wp": "I1",
        "title": "Hybrid transfer-function layer (ARX, one month ahead) at R0's monitored wells: "
                 "prospective test on heads 2026-11..2027-10",
        "kind": "test",
        "tests": [t1, t2],
        "zones": {"not_applicable": "statistical layer per well; no zones"},
        "priors": {"not_applicable": "no priors; ridge penalty alpha = 1 on standardised "
                                     "features (frozen)"},
        "burn_in": "drivers from 2012-01, filters started at zero anomaly (2012-2022 mean); "
                   "heads fitted from 2012-02",
        "folds": {"not_applicable": "prospective window; no folds"},
        "fresh_units": [UNIT],
        "thresholds": thresholds,
        "gate_scores": gate,
        "primary_model": {
            "name": "arx1: frozen AR(1) climatology-anomaly layer + ridge TF of its one-month "
                    "innovations on 8 drivers x 4 causal filters + 11 month dummies, per well",
            "code": "dataprep_staging/i1/hybrid_tf/ (prospective.py fit/predict, scorer.py "
                    "decide, model.py, config.py)",
            "code_sha256": code_hashes(),
            "git_commit": PLACEHOLDER,
            "parameters_sha256": PLACEHOLDER,
            "hyperparameters": hp,
            "physics": "none in the primary model: R0 added nothing at monitored wells at "
                       "design time and its forecast after 2022-12 is driven by climatology",
        },
        "forecast_setting": t1["setting"],
        "driver_rules": driver_rules,
        "fit_window": fit_window,
        "use_rules": rules,
        "inputs": [
            "results/dataprep/reports/I1_hybrid_tf/design_skill.json",
            "results/dataprep/reports/I1_hybrid_tf/availability.json",
            "results/dataprep/reports/I1_changhua_drivers/report.json",
            "results/twin/prospective_2026-11/predictions.csv",
            "results/twin/prospective_2026-11_hybrid/predictions.csv",
            "results/twin_runs/stage3_datum_gate_cwa/stage3_per_entry.npz",
        ],
        "interpretation": (
            "Disclosure. The design used heads 2012-2022 at R0's 158 wells, units already "
            "spent by R0..R1e and I1_changhua_drivers (design-time evidence, plan 4.5); four "
            "design runs were logged as looks on 2026-10-20 (run 2 added ridge extrapolation "
            "guards, run 3 the ARX family and electricity-recorded variants, run 4 corrected "
            "the canal availability lag). The ARX family was chosen as primary after run 2 "
            "showed the pure TF layer losing to the AR(1) layer once drivers are cut; it nests "
            "the AR(1) layer. The design PREDICTS FAILURE of T1 unless electricity is recorded "
            "for the window: with the drivers a forecast actually has, the forward-chaining "
            "ratio to the AR(1) layer was 0.98 (Changhua) and 1.10 (Yunlin); with every "
            "driver recorded 0.92 and 0.96. Registering this test is useful only if the user "
            "wants that prediction checked on fresh months or can obtain monthly electricity "
            "for 2026-07..2027-10 (a TPC census delivery granting statistical_layer_history). "
            "The test is separate from the registered prospective_2026-11 and "
            "prospective_2026-11_hybrid tests, which stay untouched (plan 4.6)."),
        "draft_status": "DRAFT for review, written 2026-10-20 by dataprep_staging/i1/hybrid_tf/"
                        "declaration.py; NOT registered; placeholders marked "
                        f"{PLACEHOLDER!r}",
        "draft_registration_deadline_utc": DEADLINE_UTC,
    }
    return doc


def main() -> None:
    from harness import prereg, prereg3
    doc = build()
    C.DRAFT.mkdir(parents=True, exist_ok=True)
    wells_frozen().to_csv(C.DRAFT / "wells.csv", index=False)
    (C.DRAFT / f"{PREREG_ID}.draft.json").write_text(
        json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (C.DRAFT / "register_draft.py").write_text(REGISTER.format(
        pid=PREREG_ID, ph=PLACEHOLDER, deadline=DEADLINE_UTC,
        deadline_iso=DEADLINE_UTC.replace("Z", "+00:00")), encoding="utf-8")
    probe = {k: v for k, v in doc.items() if not k.startswith("draft_")}
    v1 = prereg.validate_prereg(probe)
    v3 = prereg3.validate_prereg(probe)
    (C.DRAFT / "validation.json").write_text(json.dumps(
        {"harness.prereg.validate_prereg": v1 or "valid",
         "harness.prereg3.validate_prereg": v3 or "valid",
         "note": "dry validation only; nothing was registered and the index was not touched"},
        indent=1), encoding="utf-8")
    print("v1:", v1 or "valid", "| v3:", v3 or "valid")


if __name__ == "__main__":
    main()
