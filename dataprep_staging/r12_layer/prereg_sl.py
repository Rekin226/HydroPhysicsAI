"""Write the declaration SL.head_layer_companion.v1 (report-only companion of N3S-L).

It binds the frozen head-driven subsidence layer (code, coefficient, inputs, base
predictions) and fixes, before any summary of survey years ROC116-117 is read, which
readings are reported when N3S-L is scored and with which wording. It consumes no fresh
unit and scores nothing itself: every official number is read by N3S-L's scorer from
``subsidence_public_reports/prospective`` and only afterwards compared with this layer.

Run from a clean checkout of main after this file is merged, with the precedent non-resolving
``rel_to_repo`` (``--write`` applies it): ``python dataprep_staging/r12_layer/prereg_sl.py``
validates; ``--write`` registers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGING = HERE.parent
REPO = STAGING.parent
sys.path.insert(0, str(STAGING))
from common.paths import DATAPREP  # noqa: E402
from harness import prereg3  # noqa: E402
from harness.prereg import not_applicable  # noqa: E402

ID = "SL.head_layer_companion.v1"
N3SL = "N3SL.township_prospective_roc116-117.v1"
CODE = ["dataprep_staging/r12_layer/model.py", "dataprep_staging/r12_layer/freeze.py",
        "dataprep_staging/r12_layer/power.py", "dataprep_staging/r12_layer/prereg_sl.py",
        "dataprep_staging/r12_leveling/increments.py", "dataprep_staging/r12_leveling/layer.py",
        "dataprep_staging/r12_leveling/backtest.py", "dataprep_staging/r12_leveling/nowcast.py",
        "dataprep_staging/r12_leveling/paths.py"]
COAST = ["麥寮鄉", "北港鎮", "東勢鄉", "水林鄉", "臺西鄉", "四湖鄉", "口湖鄉", "褒忠鄉", "元長鄉",
         "崙背鄉", "大城鄉"]


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rel(p) -> str:
    """Repository-relative path without resolving symlinks (the precedent wrapper)."""
    p = Path(os.path.abspath(p))
    try:
        return p.relative_to(REPO).as_posix()
    except ValueError:
        return p.as_posix()


def doc() -> dict:
    fz = DATAPREP / "reports" / "SL" / "freeze"
    pw = DATAPREP / "reports" / "SL" / "power" / "result.json"
    model = json.loads((fz / "model.json").read_text(encoding="utf-8"))
    power = json.loads(pw.read_text(encoding="utf-8"))
    n3 = [r for r in prereg3.read_chained(prereg3.PREREG_DIR / "INDEX.jsonl")
          if r["id"] == N3SL]
    inputs = [rel(fz / "model.json"), rel(fz / "predictions_base.csv"), rel(pw),
              rel(DATAPREP / "heads" / "L1_yearbook" / "v1" / "MANIFEST.json"),
              rel(DATAPREP / "heads" / "L1_clean" / "v3" / "MANIFEST.json"),
              rel(DATAPREP / "reports" / "N3S-L" / "freeze" / "cells.csv"),
              rel(DATAPREP / "rulings" / "user_rulings_2026-11-12.json")]
    return {
        "id": ID, "wp": "SL", "kind": "declaration",
        "title": "Head-driven statistical subsidence layer: report-only readings on the "
                 "N3S-L survey years ROC116-117",
        "label": "report-only by a failed design gate (synthetic power 0.65 against 0.80)",
        "tests": not_applicable("report-only companion declaration: it consumes no fresh "
                                "unit; N3S-L's scorer reads the official summaries"),
        "fresh_units": not_applicable("no unit is scored by this declaration"),
        "zones": not_applicable("no zones; one fan-wide coefficient"),
        "priors": "Known before this design: the leveling backtest 2009-2021 (layer against "
                  "the page column), the forward test 2013-2021 of the head term (+10.7 % MSE "
                  "skill over the rate layer; Yunlin +15.5 %, Changhua +2.1 %), and the "
                  "official Yunlin areas of 2015-2025 (read as an after-the-fact check).",
        "burn_in": not_applicable("no simulation"),
        "folds": not_applicable("prospective survey years"),
        "gate_scores": not_applicable("report-only readings; no weighted gate"),
        "thresholds": {"area_rate_cm_yr": 3.0, "power_min": 0.80, "false_pass_max": 0.25,
                       "kernel_km": 3.0, "self_weight": 4.0, "driver_smooth_km": 10.0,
                       "rate_years": [2019, 2020, 2021], "min_months_per_year": 9},
        "primary_model": {
            "name": "SL.subsidence_layer.v1, arm HL (rate x (1 + gamma x head-change driver))",
            "gamma_per_m": model["gamma_per_m"],
            "gamma_fit_years": model["gamma_fit_years"],
            "competitors": "C0 (the same rate field, gamma = 0); N3S-L's arms ZR, SL and Z "
                           "as N3S-L reports them",
        },
        "inputs": inputs,
        "binds": {
            "model": model,
            "n3sl_registration": {"id": N3SL, "sha256": n3[0]["sha256"] if n3 else None},
            "power_study": {"decision": power["decision"],
                            "cells": power["cells"], "seed": power["seed"]},
        },
        "frozen_code": {c: sha(REPO / c) for c in CODE},
        "procedure": {
            "c0": "frozen now for every survey year (predictions_base.csv)",
            "hl": "for survey year Y (ROC Y-1911), `freeze year --survey-year Y` is run once "
                  "heads to May of Y are held and before any official statement of that "
                  "survey year is read; it writes predictions_hl_Y.csv and its sha256 is "
                  "appended to the ledger rows of this work package. If it is not frozen in "
                  "time the reading of that year is 'not evaluable'.",
            "official": "only the values N3S-L's scorer reads from "
                        "subsidence_public_reports/prospective, after N3S-L is scored",
            "survey_years": {"116": "2026-05..2027-05 (survey year 2027)",
                             "117": "2027-05..2028-05 (survey year 2028)"},
        },
        "readings": {
            "R1_area_yunlin": "per survey year: |HL - official| and |C0 - official| for the "
                              "Yunlin area sinking faster than 3 cm/yr (1 km cells counted). "
                              "Wording: 'the head-driven layer was closer than a constant "
                              "rate in k of n survey years (report-only)'.",
            "R2_area_changhua": "the same for Changhua, printed without a comparison when "
                                "both arms and the official figure are 0",
            "R3_coast": "HL's and C0's peak rate in the 11 coastal townships read against "
                        "N3S-L's printed bounds with N3S-L's L1 rule and tolerance, printed "
                        "beside ZR, SL and Z; no verdict of its own",
            "townships": COAST,
        },
        "page_mapping": {
            "every_outcome": "report-only: no claim, tile, status or page line changes; the "
                             "readings are printed in the evidence document beside N3S-L's",
            "never": "the layer does not become the subsidence baseline of record on these "
                     "readings (ruling R12.4 needs a passed gating test); no cut number, "
                     "avoided-subsidence size or policy response is derived from it",
        },
        "interpretation": "Whether a rate field modulated by observed year-on-year head "
                          "change tracks the official sinking area better than the same "
                          "field held constant, on two survey years. Two points cannot "
                          "decide it; the readings are context for a later gating design.",
        "rulings": {
            "R12.1": "yearbook heads accepted as heads/L1_yearbook/v1 with disclosures",
            "R12.2": "official county area of future editions may score the layer; applied "
                     "more narrowly here: survey year 2026 (ROC115) is not read, because "
                     "N3S-L seals it until its scoring and excluded it; ROC116-117 only",
            "R12.3": "power study first; gating iff power >= 0.80. Measured 0.6475 at the "
                     "central setting (false pass 0.1935): report-only",
            "R12.4": "baseline of record only after a passed gating test: not reachable here",
            "R12.5": "Yunlin area reading; Changhua through the N3S-L unit (R2, R3)",
            "file": "results/dataprep/rulings/user_rulings_2026-11-12.json",
        },
        "disclosures": {
            "selection": "the layer family (3-year window, 3 km kernel, own weight 4) and the "
                         "head feature were chosen after looking at 2009-2021 leveling "
                         "scores; nine head features were screened on 2018-2021; the "
                         "multiplicative form was chosen after the additive one",
            "gamma_interval": "year-bootstrap 90 % interval of gamma: -0.30 to +0.02 per "
                              "metre; zero is not excluded",
            "areas_read": "the official county areas of 2015-2025 were read; fitted to 2021 "
                          "the layer gives Yunlin 171, 175, 270, 106 km2 for 2022-2025 "
                          "against 239.5, 247.7, 226.4, 0.0 (C0 268)",
            "overlap_with_n3sl_sl": "the rate field resembles N3S-L's report-only arm SL "
                                    "(mean of three years, 3 km); the new element is the "
                                    "head term and the 2001-2011 head record",
            "hl_2026": "HL for survey year 2026 is recorded (Yunlin 215 km2 against C0 268) "
                       "and never scored",
            "power_design": "the power study assumed three survey years including 2026; "
                            "with two years the design is weaker still",
            "yearbook_gate": "the yearbook product's gate thresholds were written after its "
                             "validation numbers were known",
            "interval_start": "ROC116's interval starts 2026-05, before this declaration; "
                              "heads to 2026-09 are held and enter HL's 2027 driver",
        },
        "attempt_policy": "User rule 2026-10-23: engineering failures may be repaired and "
                          "rerun; each reading is computed once.",
        "written_by": "dataprep_staging/r12_layer/prereg_sl.py",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--prereg-dir", type=Path, default=None)
    ap.add_argument("--rows-dir", type=Path, default=None)
    a = ap.parse_args(argv)
    d = doc()
    kw = {}
    if a.prereg_dir is not None:
        kw["prereg_dir"] = a.prereg_dir
    errs = prereg3.validate_prereg(d, **kw)
    print(json.dumps(d, indent=1, ensure_ascii=False, sort_keys=True)[:3000])
    print("validate:", errs or "ok")
    print("this script sha256:", sha(Path(__file__)))
    if errs or not a.write:
        return 1 if errs else 0
    prereg3.rel_to_repo = rel
    if a.rows_dir is not None:
        kw["rows_dir"] = a.rows_dir
    path, h = prereg3.write_prereg(d, **kw)
    print(f"registered {rel(path)} sha256 {h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
