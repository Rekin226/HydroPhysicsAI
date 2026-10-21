"""Task 5: write the DRAFT T3 declaration (never registered here).

Usage: python -I t3_declaration.py OUT_DIR CV_JSON

Writes into OUT_DIR (a new directory): the draft JSON, ``register_draft.py``, ``wells.csv``
and ``validation.json`` (dry validation by harness.prereg and harness.prereg3; nothing is
written to the pre-registration index or the ledger). The draft copies the registered
``I1.hybrid_tf.prospective_2026-11.v1`` (read-only) and changes only the electricity drivers,
the ids, the inputs and the expected outcome.
"""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from common.paths import REPO, rel_to_repo  # noqa: E402

I1_ID = "I1.hybrid_tf.prospective_2026-11.v1"
I1_JSON = REPO / f"results/dataprep/preregistration/{I1_ID}.json"
DRAFT_ID = "X1.hybrid_tf_tpc.prospective_2026-11.v1"
T3A = "X1.hybrid_tf_tpc.T3a_onestep_2026-11"
T3B = "X1.hybrid_tf_tpc.T3b_vs_I1T1_2026-11"
FILL = "<FILL AT REGISTRATION>"
X1_DIR = REPO / "results/dataprep/electricity_x1_tpc/full/v1"
DEADLINE = "2026-10-31T16:00:00Z"
# I1 design (forward chaining, one month ahead, pooled RMSE / AR(1) layer), design_skill.json
I1_RATIO = {"changhua": {"rain_recorded": 0.9841, "elec_recorded": 0.9218, "all": 0.9234},
            "yunlin": {"rain_recorded": 1.1031, "elec_recorded": 1.0949, "all": 0.9586}}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def expected(cv: dict) -> dict:
    """Expected T3a ratio by linear interpolation between the I1 design's 'electricity cut'
    and 'electricity recorded' arms, weighted by the share f of the census anomaly that X1
    recovers out of fold (anomaly skill 1 - SSE/SSE_clim, county and 3 km disc)."""
    out = {}
    for c in ("changhua", "yunlin"):
        r = I1_RATIO[c]
        fs = {"county_pre": cv["county"][c]["ridge_wx"]["pre"]["anomaly_skill"],
              "county_post": cv["county"][c]["ridge_wx"]["post"]["anomaly_skill"],
              "disc_pre": 1 - cv["discs"]["pre"][c]["ridge_wx"]
              ["pooled_rmse_ratio_vs_disc_clim"] ** 2,
              "disc_post": 1 - cv["discs"]["post"][c]["ridge_wx"]
              ["pooled_rmse_ratio_vs_disc_clim"] ** 2}
        est = {k: round(r["rain_recorded"] - max(f, 0.0) * (r["rain_recorded"]
                                                           - r["elec_recorded"]), 4)
               for k, f in fs.items()}
        out[c] = {"f": {k: round(v, 4) for k, v in fs.items()}, "expected_ratio": est,
                  "range": [min(est.values()), max(est.values())],
                  "I1_design": r}
    return out


def build(cv: dict) -> dict:
    i1 = json.loads(I1_JSON.read_text(encoding="utf-8"))
    doc = copy.deepcopy(i1)
    for k in ("created", "harness_version", "schema"):
        doc.pop(k, None)
    exp = expected(cv)
    doc["id"] = DRAFT_ID
    doc["wp"] = "X1"
    doc["title"] = ("Hybrid ARX layer with TPC-downscaled electricity (X1) at R0's monitored "
                    "wells: prospective test on heads 2026-11..2027-10 (T3)")
    unit = i1["tests"][0]["fresh_units"][0]
    ext = i1["tests"][0]["external_units"]
    doc["tests"] = [
        {"test_id": T3A, "data_scope": "heads:2026-11..2027-10", "fresh_units": [unit],
         "external_units": ext, "role": "primary; decides the outcome",
         "setting": "one month ahead with recorded drivers, electricity from X1 "
                    "(electricity_x1_tpc/full/v1 plus the frozen X1 extension on archived TPC "
                    "d007012 vintages): the forecast of month t uses heads up to t-1 and the "
                    "drivers recorded for months up to t"},
        {"test_id": T3B, "data_scope": "heads:2026-11..2027-10", "fresh_units": [unit],
         "external_units": ext,
         "role": "secondary; reported with the number of comparisons, never decides",
         "setting": f"per county, pooled RMSE of T3a's arx1 over that of the registered {I1_ID}"
                    " T1 arx1 on the same scored well-months: does X1 electricity add to the "
                    "registered layer (census to 2026-06, then climatology)?"}]
    doc["gate_scores"] = [{"name": "ratio_arx1x1_vs_stat_ar1_1:changhua", "weight": 0.5},
                          {"name": "ratio_arx1x1_vs_stat_ar1_1:yunlin", "weight": 0.5}]
    pm = doc["primary_model"]
    pm["name"] = ("arx1 with X1 electricity: the registered I1 ARX layer (frozen AR(1) "
                  "climatology-anomaly layer + ridge TF of its one-month innovations on 8 "
                  "drivers x 4 causal filters + 11 month dummies, per well) with the two "
                  "electricity drivers from X1")
    pm["code"] = ("dataprep_staging/x1_tpc_monthly/t3.py (fit/predict; patches "
                  "i1.hybrid_tf.prospective.driver_panel at run time, files unchanged) + "
                  "extend.py, model.py, panel.py; i1/hybrid_tf/ as registered by " + I1_ID)
    pm["code_sha256_i1_hybrid_tf"] = i1["primary_model"]["code_sha256"]
    pm["code_sha256"] = {n: sha(HERE / n) for n in ("t3.py", "extend.py", "model.py",
                                                    "panel.py", "gate_spec.py", "cv.py",
                                                    "build.py")}
    pm["x1_product"] = {"path": rel_to_repo(X1_DIR),
                        "manifest_sha256": sha(X1_DIR / "MANIFEST.json"),
                        "parameters_sha256": sha(X1_DIR / "parameters.json"),
                        "use": "statistical_layer_history"}
    pm["parameters_sha256"] = FILL
    pm["git_commit"] = FILL
    pm["physics"] = i1["primary_model"]["physics"]
    dr = doc["driver_rules"]
    dr["recorded_month"]["electricity (3 km and county)"] = (
        "X1: a month is recorded if the electricity_x1_tpc/full/v1 county row has flag 0 "
        "(census recorded: >= 99 % of active cells with flags 0-5; identical to the I1 "
        "driver) or 5 (TPC-downscaled, 2019-09..2020-06 and 2026-07..2026-08), or if it is "
        "after 2026-08 and an archived TPC d007012 vintage (dated directory "
        "data_fetch_api/new_measurements/tpc_d007012/<date>/ with a valid SOURCE.json, "
        "fetched on or before 2028-01-31) carries own-county 農林漁牧 for t, t-1 and t-2; "
        "the frozen X1 model (v1 parameters.json, extend.apply) then gives the county value "
        "and the frozen month-of-year disc share the 3 km value. Missing secondary features "
        "(other county, other sectors, national total from archived Energy Administration "
        "sets 236/256, rain, ET0) take their month-of-year training mean.")
    dr["never_used"] = ("census extension P2def; any electricity product other than X1 v1 "
                        "and its frozen extension; any driver product without a manifest "
                        "except the twin rain cache")
    doc["fit_window"]["drivers"] = ("2012-01..2026-09 as recorded (rules above; electricity "
                                    "from X1, so 2019-09..2020-06 and 2026-07..08 are "
                                    "TPC-downscaled instead of climatology)")
    doc["fit_window"]["fallback_if_U4_strict"] = i1["fit_window"]["fallback_if_U4_strict"]
    th = doc["thresholds"]
    th["baselines"]["i1_t1_arx1"] = (f"the registered {I1_ID} T1 arx1 predictions (frozen "
                                     "parameters results/dataprep/reports/I1_hybrid_tf/"
                                     "frozen/parameters.json); T3b only, not deciding")
    th["schedule"] = ("score once heads 2026-11..2027-10 are fetched and built by the H1a L1 "
                      "v3 code and TPC d007012 has published 2027-10 (about 2-6 weeks after "
                      "the month); drivers are taken as recorded on 2028-01-31 (later data "
                      "never enter); score no later than 2028-01-31; if 2027-10 is not "
                      "published by then, its electricity takes the frozen climatology rule")
    th["threshold_basis"] = ("copied unchanged from " + I1_ID + " (design-time evidence on "
                             "2012-2022 only, U7). Expected T3a ratio from the X1 design "
                             "(interpolation, not a head score): Changhua "
                             f"{exp['changhua']['range'][0]}-{exp['changhua']['range'][1]}, "
                             f"Yunlin {exp['yunlin']['range'][0]}-"
                             f"{exp['yunlin']['range'][1]}")
    doc["use_rules"]["U4"] = (
        i1["use_rules"]["U4"] + " T3 adds the X1 product, a post-origin TPC-downscaled "
        "forcing product (plan 2.5 U4 names 'the TPC extension' among the recorded-arm "
        "products); its manifest grants statistical_layer_history, as the C0 full census "
        "does. The same user ruling covers it.")
    doc["use_rules"]["U7"] = (
        i1["use_rules"]["U7"] + " The X1 model and its selection used electricity only "
        "(census 2012-2026H1, TPC 2016-2026-08); no head was read by WP X1.")
    doc["use_rules"]["U6"] = ("Each scored number gets ledger rows: heads 2026-11..2027-10 "
                              "(scoring target), the T3 parameters.json, X1 v1 and every "
                              "archived TPC vintage used (input), this registration (input). "
                              "X1 design roles: results/dataprep/reports/X1_tpc_monthly/"
                              "data_roles.csv.")
    doc["inputs"] = [
        {"path": rel_to_repo(X1_DIR / "MANIFEST.json")},
        {"path": rel_to_repo(X1_DIR / "parameters.json")},
        {"path": rel_to_repo(X1_DIR / "county_monthly.parquet")},
        {"path": rel_to_repo(X1_DIR / "discs_monthly.parquet")},
        {"path": "results/dataprep/reports/X1_tpc_monthly/evidence/cv_final.json"},
        {"path": rel_to_repo(I1_JSON)},
        {"path": "results/dataprep/reports/I1_hybrid_tf/frozen/parameters.json"},
        {"path": "results/dataprep/reports/I1_hybrid_tf/design_skill.json"},
        {"path": "results/twin/prospective_2026-11_hybrid/predictions.csv"},
        {"path": "results/twin/prospective_2026-11/predictions.csv"},
        {"path": "results/dataprep/reports/X1_tpc_monthly/draft_declaration/t3_frozen/"
                 "parameters.json", "note": FILL}]
    doc["interpretation"] = (
        "Disclosure. T3 is the registered I1 ARX test with electricity from WP X1 instead of "
        "the census-then-climatology rule. It shares its fresh units (heads 2026-11..2027-10 "
        f"at R0's 158 wells) with {I1_ID} T1 and T2, so the family has 4 comparisons on the "
        "same heads; all are registered before any window head exists. The X1 model was "
        "chosen by a rule fixed in gate_spec.py before its CV (forward chaining by year, "
        "electricity only); the post-origin X1 folds 2023-2026H1 overlap months that carry "
        "P1 and P2def scoring looks. Expected outcome (design, interpolated, not a head "
        f"score): Changhua ratio to the AR(1) layer {exp['changhua']['range'][0]}-"
        f"{exp['changhua']['range'][1]} (I1 design: 0.984 with electricity cut, 0.922 with "
        f"it recorded), Yunlin {exp['yunlin']['range'][0]}-{exp['yunlin']['range'][1]} "
        "(1.103 and 1.095). So 'Changhua only' is possible, Yunlin is expected to fail, and "
        "the bootstrap condition (95th percentile < 1.00) may fail even in Changhua (the "
        "I1 design's year-to-year ratios with recorded electricity ranged 0.72-1.03).")
    doc["draft_status"] = ("Drafted by dataprep_staging/x1_tpc_monthly/t3_declaration.py; NOT "
                           "registered. Placeholders are filled at registration (steps in "
                           "register_draft.py).")
    doc["draft_registration_deadline_utc"] = DEADLINE
    doc["draft_expected_design_skill"] = exp
    return doc


REGISTER = '''"""Register the DRAFT X1.hybrid_tf_tpc.prospective_2026-11.v1 (T3).

DO NOT RUN before the user has reviewed the draft and ruled on U4 (as for I1). Steps:

1. Freeze the code: commit dataprep_staging/x1_tpc_monthly/ (or copy it to a frozen path)
   and check every sha256 in primary_model.code_sha256 and code_sha256_i1_hybrid_tf.
2. Optional: if TPC has published 2026-09 by then, archive it with
   ``python -I dataprep_staging/x1_tpc_monthly/fetch.py <plan>`` into a NEW dated directory
   (data_fetch_api/new_measurements/tpc_d007012/<date>/). The fit then uses it through the
   frozen extension. Do NOT rebuild the X1 product: T3 is frozen on v1.
3. Fit the frozen model once (reads heads 2012-02..2026-09; nothing may change afterwards):
       PYTHONPATH=dataprep_staging dataprep_staging/tools/cpu_slot.sh \\
           ~/miniforge3/envs/hydro/bin/python -c "from pathlib import Path; \\
           import sys; sys.path.insert(0, 'dataprep_staging/x1_tpc_monthly'); import t3; \\
           p = t3.fit('2026-09', Path('results/dataprep/reports/X1_tpc_monthly/draft_declaration/t3_frozen')); \\
           print(p['parameters_sha256'])"
4. Put that sha256 and the git commit where the draft says <FILL AT REGISTRATION>, drop the
   "note" of the t3_frozen/parameters.json input, save.
5. Run this script before 2026-10-31T16:00:00Z.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "dataprep_staging"))

from harness import prereg  # noqa: E402

DRAFT = Path(__file__).with_name("X1.hybrid_tf_tpc.prospective_2026-11.v1.draft.json")
DEADLINE = dt.datetime.fromisoformat("2026-10-31T16:00:00+00:00")


def main() -> None:
    text = DRAFT.read_text(encoding="utf-8")
    if "<FILL AT REGISTRATION>" in text:
        raise SystemExit("placeholders remain in the draft; fill them first (steps 1-4)")
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


def main(argv: list[str]) -> None:
    out, cv_path = Path(argv[0]), Path(argv[1])
    out.mkdir(parents=True, exist_ok=False)
    cv = json.loads(cv_path.read_text(encoding="utf-8"))
    doc = build(cv)
    from harness import prereg, prereg3
    probe = {k: v for k, v in doc.items() if not k.startswith("draft_")}
    probe["primary_model"] = dict(probe["primary_model"])
    v1 = prereg.validate_prereg(probe)
    v3 = prereg3.validate_prereg(probe)
    (out / f"{DRAFT_ID}.draft.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (out / "register_draft.py").write_text(REGISTER, encoding="utf-8")
    shutil.copyfile(REPO / "results/dataprep/reports/I1_hybrid_tf/draft_declaration/wells.csv",
                    out / "wells.csv")
    (out / "validation.json").write_text(json.dumps({
        "harness.prereg.validate_prereg": v1 or "valid",
        "harness.prereg3.validate_prereg": v3 or "valid",
        "note": "dry validation only; nothing was registered and the index was not touched"},
        indent=1) + "\n", encoding="utf-8")
    print("prereg", v1 or "valid", "| prereg3", v3 or "valid")
    print(json.dumps(doc["draft_expected_design_skill"], indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
