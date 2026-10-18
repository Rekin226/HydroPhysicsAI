"""Pre-registration of the W1 Jan-May benchmark test (plan 4.5), written BEFORE the test runs.

The WP A1 pre-registration tool did not exist when W1 ran (A1 is not upstream of W1), so this
writer follows the convention of results/dataprep/preregistration/README.md directly: one
JSON per registered test, never edited (exclusive create), with the sha256 of the code and
inputs it names; its own sha256 goes to the W1 ledger fragment.

    python -m w1_canal.prereg
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.hashing import sha256_file  # noqa: E402
from common.paths import PREREG_DIR, rel_to_repo  # noqa: E402

from w1_canal import config as C  # noqa: E402

PREREG_ID = "W1_janmay_benchmark_2026-10-12"
CODE = ["config.py", "parse_yearbook.py", "sources.py", "composite.py", "diversion.py",
        "validate.py", "statespace.py", "prereg.py"]


def inputs() -> list:
    files = [p for y in C.YEARBOOK_YEARS for p in sorted((C.YEARBOOK_DIR / f"hyb{y}").glob("*.pdf"))]
    files += [C.JIJI_CSV, C.JIJI_HTML, C.BRIEFING_110_PDF, C.MOA_691_CSV, C.V3_NPZ]
    return [{"path": rel_to_repo(p), "sha256": sha256_file(p)} for p in files]


def document() -> dict:
    return {
        "id": PREREG_ID,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "wp": "W1",
        "plan_section": "docs/DATA_REMEDIATION_PLAN_2026-10.md, section 3 WP W1 (Validation)",
        "writer": "dataprep_staging/w1_canal/prereg.py (A1 tool not available; see module doc)",
        "design_time_contamination": (
            "The 2026-10-12 scout (results/dataprep/_pilots/surface_water_monthly/pilot2.py) "
            "already scored this benchmark on the same seven years, with beta fitted on "
            "2012-2025, and the supply-limited + Denton design, the share grid and the winter "
            "options were chosen after seeing those scores (RMSE 0.40/0.37 against "
            "climatology 0.66/0.56 and v3 0.84/1.06). No fresh Jan-May units exist. This run "
            "is a re-validation with pre-origin fitting and a corrected yearbook parse; its "
            "results are design-time-contaminated evidence (plan 4.5), not a fresh test."),
        "primary_variant": "preorigin: beta, a_chc, a_yl fitted to the 2012-2022 annual "
                           "district totals; composite QC medians from 2012-2022",
        "secondary_variant": "full: the same with 2012-2025",
        "held_out_data": "Jan-May supply per district 2015-2021, WRA 110年度彰雲地區地層下陷"
                         "檢測成果 slide 19 (never used to fit beta or the demand scales)",
        "method_scored": (
            "supply_limited_denton: D = max(0, min(beta*I - R, C_chc + C_yl)), Denton (PFD) "
            "to the annual totals; crop-1 shares and winter weight chosen leave-one-year-out "
            "by the summed squared Jan-May error of both districts over the six training "
            "years; grid s in " + str(list(C.SHARE_GRID)) + ", winter in "
            + str(list(C.WINTER_GRID))),
        "baselines": {
            "v3": "results/twin/surface_water_v3.npz monthly timing x the annual total",
            "clim_fraction_loo": "Jan-May fraction = mean of the other six years x the annual "
                                 "total"},
        "metric": "RMSE (10^8 m3) per district over the 7 years",
        "comparison": {
            "paired_bootstrap": {"statistic": "RMSE(method) - RMSE(baseline)",
                                 "resample": "years with replacement, paired",
                                 "B": C.BOOT_B, "seed": C.BOOT_SEED,
                                 "interval": "percentile", "level": C.BOOT_LEVEL},
            "sign_test": "two-sided binomial on |e_method| < |e_baseline| (ties dropped)"},
        "acceptance_rule": (
            "Pass only if the method beats both v3 and the climatological fraction in both "
            "districts with the 90 % paired-bootstrap interval of the RMSE difference entirely "
            "below zero (4 comparisons). Otherwise v4 ships labelled 'not distinguishable from "
            "climatology on the benchmark' (if a climatology interval includes zero) or 'not "
            "distinguishable from v3 on the benchmark' (if only a v3 interval does). Decided on "
            "the primary (preorigin) variant; the full variant is reported."),
        "supporting_checks_not_gated": {
            "state_space_coverage": (
                "Leave-one-year-out: the state-space model is refitted without the held-out "
                "year's Jan-May value (indicator from that fold's chosen shares); the 90 % "
                "predictive interval of the held-out Jan-May total is checked on 14 "
                "district-years. Pass if the Clopper-Pearson 95 % interval of the coverage, "
                "computed with effective n = 7 (year clusters: hits scaled by 7/14), contains "
                "0.90 (approximation of the A1 cluster-adjusted rule, plan 4.4). Intervals "
                "without this check are labelled uncalibrated."),
            "head_partial_correlation": (
                "Monthly head anomaly (month-of-year mean removed) of the Changhua and Yunlin "
                "layer-1 and layer-2 well means, 2012-2022, against the 3-month delivery "
                "anomaly, partial correlation given the 3-month fan-mean rain anomaly (CWA "
                "CODIS rebuild); preorigin v4 Denton against v3. 'Not worse' = v4 >= v3 in "
                "each of the 4 groups."),
            "crop_season_shares": "crop-1 (Feb-Jun) share of v4 against MOA 691 2012-2015, "
                                  "reported only"},
        "inputs": inputs(),
        "code": [{"path": rel_to_repo(C.HERE / f), "sha256": sha256_file(C.HERE / f)}
                 for f in CODE],
    }


def write() -> Path:
    PREREG_DIR.mkdir(parents=True, exist_ok=True)
    p = PREREG_DIR / f"{PREREG_ID}.json"
    with open(p, "x", encoding="utf-8") as f:          # never overwrite a registration
        json.dump(document(), f, ensure_ascii=False, indent=1)
        f.write("\n")
    p.chmod(0o444)
    return p


if __name__ == "__main__":
    path = write()
    print(path, sha256_file(path))
