"""Write the P6 declaration ``P6.inventory_member.v1`` (harness.prereg3) before any P6 result.

A declaration, not a test: P6 reads no head, ring, pile, GNSS or InSAR value, scores nothing
on a validation unit, and its acceptance (plan WP P6) is a reconciliation of report totals.
The record fixes, before the inventory, the reconciliation, the census comparison, the WRA-to-
model ratio check and the member are computed:

* the reconciliation rule and its threshold (the plan's 5 %), with the disclosure that the
  annex totals and the text layer of annex table 2-2 were seen while locating the tables;
* the transcription and verification rules;
* every assumption of the sensitivity member and its variants (``p6const.py`` is hashed);
* the census comparison metrics and the pre-origin WRA-to-model ratio check (reported, no
  threshold, no effect on the member, which keeps the plan's 2.58).

Usage: python -I prereg_p6.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import rel_to_repo  # noqa: E402
from harness import prereg3  # noqa: E402

from p6_public_wells import p6const as K  # noqa: E402

NA = prereg3.not_applicable
N1_PARSE = K.CODE_DIR.parent / "n1_natural_experiments" / "parse_sources.py"
N1_CONST = K.CODE_DIR.parent / "n1_natural_experiments" / "n1const.py"

LOOK_DISCLOSURE = (
    "Before this record, on 2026-10-14 (look logged 2026-10-14T15:57:12Z, purpose "
    "inventory_read, unit P6:moea_public_well_tables), P6 read the pdftotext text layer of the "
    "MOEA 2019 annex (774 wells, 8,793.9 x 10^4 m3/yr disposed in 100-107), of the phase-2 "
    "main plan (table 4-5; annex tables 2-2, 2-3, 2-5; the text '1,056 wells, 13,098.45') and of "
    "Control Yuan 53001 (1,191 wells under project control), and summed the printed 100-107 "
    "rows of annex table 2-2 by eye (775 wells by printed row totals; 8,793.9 x 10^4 m3/yr). "
    "The reconciliation R1 below was therefore seen before it was declared: it is reported as a "
    "consistency check of the transcription against the annex, with the plan's threshold (5 %, "
    "not chosen here). P6 has read no head, ring, pile, GNSS or InSAR value, no engine "
    "hindcast head and no census kWh value before this record. The WRA-to-model ratio of the "
    "design pilot (design_bayesian_fusion/p6_wra_janmay.json, 2.5759 on 2012-2021) and the "
    "plan's 2.58 are known (design-time evidence); the pre-origin recomputation K1 has not "
    "been run.")

CHECKS = {
    "T1_transcription": (
        "Annex table 2-2 (phase-2 main plan p. 99) and annex table 2-5 (p. 101) are transcribed "
        "cell by cell twice (claude-opus-5-5 on the PDF page; claude-sonnet-5-5 on the 200-dpi "
        "PNG). A cell is accepted when both reads give the same printed number (or both empty) "
        "and the row's accepted numbers, left to right, equal the numbers of the same row in "
        "the pdftotext -layout text layer (same order). A cell where the reads differ is "
        "accepted only if exactly one read makes the row match the text layer and the printed "
        "row and column totals; otherwise the cell is flagged and the table fails T1. Table 4-5 "
        "(p. 47) and annex table 2-3 (p. 100) are parsed from the text layer directly (every "
        "cell is printed). Arithmetic checks: each body row sums to its printed row total and "
        "each column to its printed column total within the printed rounding (0.15 x 10^4 m3/yr "
        "for volume rows, exact for counts), any exception listed with its printed note."),
    "R1_reconcile_annex": (
        "Inventory totals of the public wells disposed in ROC 100-107 (2011-2018) from the "
        "transcribed annex table 2-2: N = sum of the printed row totals 100-107 (wells) and "
        "V = sum of the cohort cells 100-107 (10^4 m3/yr); also N' = sum of the cohort cells. "
        "Compared with the MOEA 2019 annex: 774 wells and 8,793.9 x 10^4 m3/yr. Pass when "
        "|N - 774| / 774 <= 0.05 and |V - 8793.9| / 8793.9 <= 0.05 (plan WP P6 acceptance). "
        "N' and the cause of any N vs N' difference are reported, never used to pass. Gate of "
        "public_wells/v1."),
    "R2_internal_totals": (
        "Reported consistency (not gating): phase-1 totals 100-108 (1,056 wells; 13,094.85 "
        "printed in table 2-2 vs 13,098.45 printed in the text of annex 2) against the cohort "
        "sums; per agency and county against table 4-5 (actual 100-108) and annex table 2-3 "
        "(executed to 108); irrigation-association and Taiwan Sugar wells and volumes by year "
        "against annex table 2-5; the Control Yuan 53001 list (1,191 = Taiwan Water 350, "
        "irrigation associations 590, Taiwan Sugar 12, others 239) against the phase-2 list "
        "(1,185)."),
    "D1_drought_wells": (
        "The 2022 drought-well tables (wra111 annex tables 1-8 and 1-10) are re-parsed from the "
        "L0 text layer with N1's parser (n1_natural_experiments/parse_sources.drought_wells, "
        "hashed); counts must equal the report's table 3 (Changhua 49 = 26 irrigation + 23 "
        "Taiwan Water; Yunlin 521 = 421 + 98 + 2, of which 520 rows are printed and number 280 "
        "is skipped) and the rows must equal natural_experiments/v1 drought_wells.csv."),
    "C1_census": (
        "Census reconciliation (descriptive, no threshold) on pump_electricity/census_L1/v3 and "
        "preorigin/v3 (C0), opened through the guard with use 'prior': (a) pumps by county and "
        "PURPOSE (counts, in grid, kept, horsepower >= 50 and > 10, max), checking the plan's "
        "Yunlin counts 430 public supply, 326 industrial, 91 domestic-and-public, 9 pumps >= 50 "
        "HP, 1,341 > 10 HP (reproduced or the difference reported); (b) for each drought well, "
        "the distance to the nearest census pump of any purpose and of a public-supply purpose, "
        "with shares within 50 m and 200 m by agency and county, and the purpose and HP of the "
        "nearest pump; (c) census kWh 2012-2021 (kept pumps, in grid) by purpose class and of "
        "pumps within 50 m of a drought well; (d) the census-equivalent energy of the inventory "
        "volumes with the K1 WRA m3 per census kWh (reported as an order of magnitude only)."),
    "K1_ratio_preorigin": (
        "Pre-origin recomputation of the WRA-to-model volume ratio (plan P6 step 3's 2.58, a "
        "design-pilot number). WRA side: Yunlin county Jan-May irrigation pumping, series "
        "'雲林合計' (left axis, 萬噸) of the 110 briefing chart p. 18 (published before the "
        "origin; WRA's estimate from registered-well electricity x water/electricity ratio), "
        "the mean of the two reads per year. Model side: engine/v2 (P4.0 replica of the "
        "deliverable) Q_rate (m3/day; step t = month dates[t+1], Feb 2012 - Dec 2022) summed "
        "over cells whose cell_county is 雲林縣, times days of the month, times the county "
        "irrigation share of census energy in that month (C0 preorigin/v3 cells_by_class "
        "value_by_county, Yunlin: irrigation / all classes). Primary ratio = mean over "
        "2013-2021 of WRA / mean over 2013-2021 of model (complete Jan-May in both); also "
        "reported: the pilot's definition (2012-2021 with the model's 2012 missing January), the "
        "per-year ratios and their sd, and WRA m3 per census Jan-May irrigation kWh (Yunlin in "
        "grid, preorigin/v3). Reported only: the member keeps 2.58 whatever K1 gives."),
    "M1_member": (
        "public_wells/v1 sensitivity_member.npz, fixed by p6const.py (hashed here). Cohorts = "
        "the cells of annex table 2-2 (agency column x year 100-108) with their volume v "
        "(10^4 m3/yr; a count without a printed volume gives 0). A cohort of year Y abstracts at "
        "the constant rate v x 1e4 / 365.25 m3/day in every month before Y-07 (disposal at "
        "mid-year) from 2007-01 and nothing from Y-07 on; time axis 2007-01..2022-12 (192 "
        "months). Volumes are physical (WRA); model units = physical / 2.58 (plan P6 step 3, a "
        "stated assumption). Layers: aquifers 3 and 4 equally (twin layers 2 and 3). "
        "Locations: Taiwan Water and irrigation-association cohorts and Yunlin Taiwan Sugar on "
        "the same county's drought wells of that agency (2022 inventory, equal share per "
        "printed row); Changhua Taiwan Sugar, schools and the Yunlin prison on the county's "
        "census pumps with a public-supply PURPOSE (公共用水, 家用及公共給水, 家用及公共用水, "
        "家庭及公共給水); the two factory cohorts on census industrial pumps (工業用水) of their "
        "township (溪州, 元長), falling back to public-supply pumps of the township, then of the "
        "county; equal share per proxy point; a point outside the active grid drops its share "
        "(in-grid fraction reported per cohort). Variants (same arrays, one change each): "
        "annex_2011_2018 (cohorts 100-107 only), timing_early (stop at Y-01), timing_late "
        "(stop at (Y+1)-01), layers_ia_aq2 (irrigation-association cohorts in aquifer 2 = twin "
        "layer 1; Control Yuan 53001: their wells are 120-150 m deep), loc_census_only (every "
        "cohort on the county's public-supply census pumps; factories as primary), "
        "loc_drought_only (cohorts without own-agency drought wells on all of the county's "
        "drought wells), loc_uniform_county (every cohort uniform over the county's active "
        "cells: no post-origin location input). Separate component pending_tw_changhua: the "
        "129 Changhua Taiwan Water wells planned for 109-113 with the Niaozuitan lake (table "
        "4-5), at (7,172.25 - 1,842.3) x 10^4 m3/yr (annex table 2-3), abstracting through "
        "2022-12, on the Changhua Taiwan Water drought wells; never part of the ramp. Checks: "
        "per year, member volume = sum of active cohort volumes x in-grid fraction (relative "
        "error <= 1e-9); no month after 2022-12; no observation of any kind enters the member. "
        "The member never enters the primary model (plan P6 step 3)."),
}


def doc() -> dict:
    inputs = [K.CODE_DIR / "p6const.py", K.CODE_DIR / "p6lib.py",
              K.CODE_DIR / "transcribe_p6.py", K.CODE_DIR / "prereg_p6.py",
              N1_PARSE, N1_CONST] + list(K.SOURCE_JSON) + [
        K.C0_L1 / "MANIFEST.json", K.C0_PRE / "MANIFEST.json", K.ENGINE / "MANIFEST.json"]
    return {
        "id": K.PREREG_ID,
        "wp": K.WP,
        "kind": "declaration",
        "title": "P6 public-well inventory: reconciliation with the MOEA annex, transcription "
                 "rules, census comparison, pre-origin WRA-to-model ratio check and the "
                 "fan-scale sensitivity member (assumptions fixed before construction)",
        "tests": NA("declaration: P6 scores nothing on validation units; checks are "
                    "reconciliations, transcription checks and construction checks"),
        "checks": CHECKS,
        "zones": NA("no zones"),
        "priors": NA("no priors are fitted"),
        "burn_in": NA("no model run"),
        "folds": NA("no folds"),
        "fresh_units": NA("no fresh units: report tables and census attributes only"),
        "thresholds": {"R1_reconcile_annex": "relative difference <= 0.05 for wells and for "
                                             "volume (plan WP P6 acceptance)",
                       "T1_transcription": "every accepted cell agreed by both reads and the "
                                           "text layer; totals within printed rounding",
                       "D1_drought_wells": "counts equal table 3; rows equal N1's",
                       "M1_member": "mass balance relative error <= 1e-9",
                       "C1_census": "none (descriptive)",
                       "K1_ratio_preorigin": "none (reported)"},
        "gate_scores": {"public_wells/v1": "gate passed iff R1 passes and T1, D1 and M1 hold"},
        "primary_model": NA("the member never enters the primary model (plan P6 step 3)"),
        "inputs": [rel_to_repo(p) for p in inputs],
        "interpretation": "R1 is the plan's acceptance, reported as a consistency check "
                          "because the totals were seen before this record; K1 and C1 are "
                          "descriptive; the member is a sensitivity input, never evidence of "
                          "skill.",
        "prior_looks_disclosed": LOOK_DISCLOSURE,
        "written_by": rel_to_repo(Path(__file__)),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    d = doc()
    errs = prereg3.validate_prereg(d)
    if errs:
        raise SystemExit(f"invalid: {errs}")
    if a.dry_run:
        print(json.dumps(d, ensure_ascii=False, indent=1)[:3000])
        return 0
    path, sha = prereg3.write_prereg(d)
    print(f"wrote {rel_to_repo(path)} sha256 {sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
