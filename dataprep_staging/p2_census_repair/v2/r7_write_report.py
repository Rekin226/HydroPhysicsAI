"""WP P2abc round 2 (resumed 2026-10-14): write results/dataprep/reports/P2abc.{md,json}.

Inputs: ``P2abc_evidence/report_tables.json`` (acceptance rows and key numbers with ledger ids,
written by r6_report.py after the ledger rows) and the two product manifests. The narrative
below states only numbers that appear in those tables or in the cited evidence files.

Usage: python -I r7_write_report.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2lib as L  # noqa: E402

R = L.REPO
REP = L.DATAPREP / "reports"
TAB = json.loads((L.EVID / "report_tables.json").read_text(encoding="utf-8"))
MP = json.loads((L.PRODUCT_ROOT / "preorigin" / "v1" / "MANIFEST.json").read_text())
MF = json.loads((L.PRODUCT_ROOT / "full" / "v1" / "MANIFEST.json").read_text())


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


PREREGS = {}
for pid in ("P2abc.r2.c0_based.v1", "P2abc.r2.k1_repair.v1", "P2a.meter_validation.v1",
            "P2a.sd_coverage.v1", "P2a.head_placebo.v1", "P2b.july_backtest.v1",
            "P2c.zero_reproduction.v1"):
    p = L.DATAPREP / "preregistration" / f"{pid}.json"
    PREREGS[pid] = sha(p)

CODE_DIR = L.CODE_DIR
ROUND1_CODE = sorted(str(p) for p in CODE_DIR.parent.glob("*.py"))
ROUND2_CODE = sorted(str(p) for p in CODE_DIR.glob("*.py"))

SUMMARY = (
    "P2abc round 2 rebuilds the billing repair of the plan's WP P2 parts a-c on WP C0's cleaned "
    "census (pump_electricity v3) for Yunlin, Changhua and the 316 kept in-grid Chiayi pumps, "
    "and publishes yunlin_electricity/preorigin/v1 (2007-01..2022-12, allowed for "
    "forcing_frozen and design) and yunlin_electricity/full/v1 (2007-01..2026-07, recorded-"
    "forcing arm and after-the-fact use). Each carries two arms on C0's grid: the recorded "
    "TPC even split, and a Chow-Lin split of the bimonthly irrigation bills by a per-county "
    "fan-wide profile (flag 3, bill totals exact). Both gates passed: the zero-classification "
    "port reproduces the pilot within 0.17 %, and the construction checks pass with 0 "
    "violations after one repair. The first assembly failed item (f): 4,050 and 5,560 cell sds "
    "were NaN where only zero-energy flag-3 or imputed pumps sat. The fix was registered as "
    "P2abc.r2.k1_repair.v1 before the re-assembly was scored. The plan's meter acceptance "
    "(skill >= 0.15 at every group size 1-300) passes in Yunlin (min 0.232, 1,789 meters) and "
    "Changhua (min 0.205, 5,434 meters). Chiayi has only 16 monthly meters, so it is reported, "
    "not gated. The registered placebo-controlled head check fails in both counties: Yunlin "
    "p 0.051, Changhua p 0.786, and the round-1 registration p 0.139. So the Chow-Lin head "
    "gain is reported as not distinguishable from regional-signal injection, and P2a ships on "
    "the meter validation as the plan allows. The Chow-Lin sd is calibrated in Yunlin and "
    "uncalibrated in Changhua, where coverage is 0.861-0.884 for groups of 10 or more: the sd "
    "is too narrow there. The July backtest passes in Yunlin (15.4 %) and Changhua (8.7 %), so "
    "their July-2026 imputation ships. It fails in Chiayi (42.2 %), whose July 2026 stays NaN. "
    "The interrupted session had built, scored and published everything; this resumed session "
    "re-verified the hashes (checker 0 failures), ran independent spot checks (per-pump bill "
    "totals equal C0 to 6e-11 kWh) and wrote the ledger (137 rows, 0 conflicts) and this "
    "report. No GPU and no Claude API were used.")

DEVIATIONS = [
    "Scope (task update): the WP covers Yunlin, Changhua and the 316 kept in-grid Chiayi pumps "
    "and builds on C0's pump_electricity v3 (kept set, classes, cells, flags and method codes "
    "unchanged), not on the AMP_V2 Yunlin cache. The plan's product name "
    "yunlin_electricity is kept.",
    "Profiles are per county. Changhua gets its own fan-wide profile, because its irrigation "
    "seasonality differs (recorded-arm CV of calendar-month means 0.135 against 0.224 in "
    "Yunlin). Chiayi's 316 pumps use the Yunlin profile and follow the Yunlin verdicts, as "
    "registered.",
    "The plan's July-2025 problem does not exist in the 2026 delivery. The July-2025 bills are "
    "present, but from 2025-07 TPC often posts a whole bimonthly bill to its second month "
    "(M2 July zero share 0.404 in Yunlin and 0.329 in Changhua in 2025, against 0.13-0.20 in "
    "2022-2024). The open bill is now the July-2026 phase-A month. P2b was applied to it: "
    "masked NaN/flag 8 by C0, imputed with flag 4 where the backtest passed. The mask table is "
    "incomplete_pair_mask.parquet rather than july2025_mask.parquet. It lists open bills at "
    "the record end, pairs with one missing month, the partial July-2026 monthly cycle and "
    "bills posted to the second month.",
    "The P2b imputation uses the meter's own 3-year same-month mean of the (Jul + Aug)/2 pair "
    "mean, not of the recorded July, because the recorded 2025 July is distorted by the "
    "second-month posting. The backtest truth is (Jul + Aug)/2 of year Y, for Y = 2013-2025, "
    "not the plan's 2013-2024 recorded July.",
    "Pre-origin December 2022 of Dec-Jan-paired (phase-B) meters keeps C0's flag-4 fill and is "
    "not re-split, because the Dec 2022-Jan 2023 bill is open at the origin. Round 1's "
    "own-3-year-mean edge fill is not used.",
    "Synchronized-zero screen: after the first build flagged 75 of 192 pre-origin months for "
    "Chiayi alone, Chiayi was screened with the pooled Yunlin + Chiayi population. This was a "
    "design change after seeing a build output. No registered test involves the screen, and "
    "its flag-9 months enter only as a sensitivity member. The screened months per county "
    "differ from the plan's list. Yunlin's pre-origin months are close to the plan's, Changhua "
    "has its own (key numbers).",
    "The registered construction checks failed on the first assembly (item f, sd of cells "
    "holding only zero-energy flag-3 or imputed pumps). The fix was chosen after reading the "
    "failure and was registered as P2abc.r2.k1_repair.v1 before the re-assembly was scored. "
    "These are construction checks, not out-of-sample tests.",
    "Yunlin's round-2 units (validation meters, July units, zero port) are not fresh. They are "
    "essentially the round-1 units, whose outcomes were known (declared in the "
    "registration). Changhua's and Chiayi's meter and July units are the fresh part. The "
    "head-check wells and 2012-2022 heads are design-time data of the billing pilot. What is "
    "new is the placebo control and the regional covariates.",
    "Chiayi's sd label ('calibrated') and its Chow-Lin shipping follow Yunlin's verdicts, as "
    "registered. They were not checked on Chiayi meters: 16 meters cannot form the groups of "
    "30-300 or the coverage strata.",
    "The pre-origin product's gate includes the zero-port check P2r2.c1. By design, that check "
    "reads the Yunlin record to 2025-06 (the pilot's window) through pump_electricity/full/v3. "
    "It only decides whether the ported classification code reproduces the pilot's counts. "
    "No value, class, flag, sd or parameter of the pre-origin product depends on post-origin "
    "data, and electricity is not a U7 observation class.",
    "The design-time counts of d0_design_counts.py were computed before a p2lib.py edit that "
    "preceded the registration. Changhua's validation-meter count there is 5,406, against "
    "5,434 in the registered test. The registered test's count is authoritative. The earlier "
    "p2lib version was not kept, so the change cannot be shown.",
    "Not delivered here: the plan's synthetic_post2016_{low,high}.npz (P2d) and "
    "layer_response_localonly.csv (P2f) belong to WP P2def.",
]

ISSUES = [
    "P2def conventions: see the report's section 'Conventions for P2def'. In short, "
    "products are under results/dataprep/yunlin_electricity/{preorigin,full}/v1/ with C0's "
    "time axis (index 0 = 2007-01; T 192 / 235), C0's grid (A = 2148 active cells of "
    "results/twin_forward/datum_gate_innov.npz, row-major over masked cells), classes "
    "(irrigation, aquaculture, livestock, domestic, industry, other) and counties "
    "(yunlin, changhua, chiayi). Per-pump arrays are in C0's pump order (uid, meter_idx), with "
    "cell -1 for kept pumps outside the active cells. Install dates and coordinates are not in "
    "these products (privacy); take them from pump_electricity/census_L1/v3.",
    "Changhua's Chow-Lin sd is labelled 'uncalibrated': it is too narrow at groups of 10 or "
    "more meters (coverage 0.861-0.884; registered, not tuned afterwards). The plan-4.8 "
    "forcing-uncertainty propagation must not take it at face value. Any widening must be "
    "fitted on pre-origin data under its own registration.",
    "The December-2022 flag-4 sd in the pre-origin product is C0's fill sd. The C0 verifier "
    "found its R7 coverage 0.808 fails the plan's gate-3, so treat it as uncalibrated even "
    "though C0 labels it calibrated. The July-2026 imputation sd (full) is also uncalibrated: "
    "its leave-one-year-out coverage is about 0.90, but only 13 year clusters (< 20) exist.",
    "The Chow-Lin arm is not a proven improvement at the scale the twin uses. On the meters, a "
    "year-shuffled profile scores almost as well as the real one for single meters (Yunlin "
    "0.224 vs 0.232 at n = 1; Changhua 0.172 vs 0.205). The real profile pulls ahead only for "
    "groups of 100 or more and fan-wide (Yunlin fan 0.501 vs 0.048). The registered head check "
    "finds no placebo-controlled gain. Treat recorded vs Chow-Lin as a forcing-timing "
    "sensitivity, and decide any member choice or weighting before results on pre-origin "
    "evidence only (U7, plan 4.8).",
    "The arms differ only in within-bill timing. 2012-2022 in-grid totals agree to 2e-5, and "
    "annual totals to 0.24 % (Dec-Jan pairs at year ends). The split moves 7.2 % (Yunlin) and "
    "4.4 % (Changhua) of pump energy between months. It raises the in-grid irrigation "
    "seasonality (CV 0.224 -> 0.260 in Yunlin, 0.135 -> 0.165 in Changhua) and moves Yunlin's "
    "peak month from April to May.",
    "Changhua's domestic class is 23.0 % of in-grid energy (43.07 of 187.36 GWh/yr), against "
    "5.5 % in Yunlin. value_total_nodomestic in kwh_by_class_*.npz is the sensitivity member "
    "(P4m).",
    "Full product, recorded-forcing arm only: from 2025-07 the recorded arm keeps TPC's "
    "second-month postings. C0's evensplit alternative re-splits them evenly, and the "
    "Chow-Lin arm re-splits them by the profile. 2026-07 is partial: phase-A bills are imputed "
    "(Yunlin, Changhua) or NaN (Chiayi, flag 8), and monthly meters carry flag 9 from C0. The "
    "M0 July/June ratio of 2025 (1.09-1.10, against 1.16-1.27 in 2022-2024) stays low in the "
    "2026 delivery. July 2025 of monthly meters may be incomplete; this is not resolved.",
    "P2d inputs. Terminated meters are in zero_class / meter_zero_summary (possible_replacement "
    "True; last-12-month energy before termination). Pre-origin: Yunlin 5,423 meters, 7.22 "
    "GWh; Changhua 6,489 meters, 9.37 GWh. The census is frozen at mid-2016 installs, so the "
    "synthetic layer must sit on top of either arm.",
    "Model integration (after the freeze). The calibrated twin's cached forcing is C0's "
    "preorigin/v1 Yunlin even split (C0 report). It differs from this product's recorded arm "
    "(which equals C0 v3) by at most 3.1e-4 monthly before 2022-12 and by -3.35 % in "
    "2022-12, and it has no Changhua pumping. Recalibrate on this product (either arm) or on "
    "C0 v3. Choosing between the recorded and Chow-Lin arms is a design choice that must not "
    "use 2023+ observations.",
    "Staging: results/dataprep/yunlin_electricity/_work_p2abc_v2 (1.4 GB) holds the stage-1 "
    "arrays and zero tables that both manifests list as inputs. Deleting it makes the products "
    "fail the checker. _work_p2abc (293 MB, round 1) is referenced by the round-1 "
    "registrations and r1_reproduce.",
    "Privacy: the per-pump arrays hold well numbers (uid; personal data declared, gitignored). "
    "No coordinates, names, addresses or raw meter numbers are in any output (construction "
    "check g).",
    "ledger/data_roles.csv was not reassembled (shared file). rows/P2abc.csv has 139 rows, and "
    "a cross-fragment check finds 0 conflicts.",
]

CONVENTIONS = """\
### Conventions for P2def (and any consumer)

**Paths.** `results/dataprep/yunlin_electricity/preorigin/v1/` and `.../full/v1/` (MANIFEST.json,
PRODUCT_CARD.md, validation.json in each). Open them with the guard
(`sys.path.insert(0, '<repo>/dataprep_staging'); from common import guard`). The pre-origin
product is allowed for forcing_frozen, model_design, prior, prior_width_input,
design_time_evidence, validator and after_the_fact_analysis; its pre-origin uses pass only
through the two exemptions inherited from C0. The full product is allowed for
forcing_recorded_arm, statistical_layer_history, validator and after_the_fact_analysis.

**Time axis.** Every array has `dates` ('YYYY-MM'); index 0 = 2007-01. Pre-origin T = 192
(to 2022-12), full T = 235 (to 2026-07). Phase-A bills pair (Jan, Feb), (Mar, Apr), ..., so
pairs start at even indices. Phase-B bills pair (Feb, Mar), ..., (Dec, Jan), starting at odd
indices.

**Grid.** A = 2148 active cells of `results/twin_forward/datum_gate_innov.npz` (nx 59, ny 77,
dx 1000 m, x0 162191.011, y0 2600419.535), in row-major order over the masked cells
(`pumping._cell_lookup`). The arrays carry `mask_hash` and `grid_hash`.

**Labels.** classes C = (irrigation, aquaculture, livestock, domestic, industry, other); counties
K = (yunlin, changhua, chiayi). Unit: kWh/month in model units (the cleaned census,
duty cap 1.0).

**`kwh_by_class_recorded.npz`, `kwh_by_class_chowlin.npz`** (one per arm, same keys):
`value, flag (uint8), sd` (C, A, T); `value_by_county, flag_by_county, sd_by_county`
(K, C, A, T); `value_total_all, flag_total_all, sd_total_all, n_eff_total_all` and the
`_nodomestic` versions (A, T); `n_eff` (effective number of Chow-Lin meters), `n_cl` (flag-3
pumps), `n_imp` (imputed pumps); C0's `missing_expected_kwh` (C, A, T) plus `_all` and
`_nodomestic`; grid metadata.

**`pumps_<county>.npz`** (`yunlin`, `changhua`, `chiayi_ingrid`) hold the Chow-Lin arm per kept
pump in C0's pump order: `uid` (well number, personal data), `meter_idx` (C0's county-scoped
meter id), `cell` (index into A; -1 = kept pump outside the active cells: Yunlin 333,
Changhua 2,652), `purpose_class`, `hp`, `hp_share`, and `value, flag, method, sd` (P, T). The
recorded arm per pump is C0's `pump_electricity/<variant>/v3/pumps_<county>.npz` (same order;
identical outside the July-2026 imputation, which `july2026_imputation.parquet` lists per meter).

**Method codes (uint8).** 0 observed, 1 pre_install, 2 observed_before_install_date, 3 missing,
4 incomplete_pair_2026_07, 5 partial_reading_cycle_2026_07, 6 terminated_trailing_run,
7 bill_in_second_month, 8 uneven_pair, 9 fill_dec2022_dec_jan_pair (C0's codes),
10 chowlin_split (flag 3), 11 own_3yr_same_month_mean (flag 4, July-2026 imputation).

**Flags.** 0 observed; 3 Chow-Lin split; 4 C0's December-2022 fill (pre-origin) or the
July-2026 imputation (full); 8 missing (value NaN); 9 suspect. Aggregates take the highest
flag in the order 8 > 9 > 4 > 3 > 0, and a flag-8 aggregate is NaN. Missing post-install
readings count as 0 in aggregates, as in the model; `missing_expected_kwh` indicates them.
`guard.read_table` for forcing_frozen keeps flags 0-5. npz users must apply flags themselves.

**sd.** Flag 3: q90(n_eff, calendar month)/1.645 x the even-split energy of the Chow-Lin
component, with q90 from the county's validation groups (Chiayi uses Yunlin's). Flag 4: C0's
fill sd, or q90_rel/1.645 x the imputed energy. Components add in quadrature. Cross-cell
correlation is not modelled: do not add cell variances to get a fan-total sd.
Labels: Yunlin calibrated, Changhua uncalibrated, Chiayi as Yunlin (not checked); the
July-2026 imputation is uncalibrated.

**Tables (join on `county`, `meter_idx`).**
- `meter_phase.parquet`: billing_code, phase_code, phase_detected, dec_jan_paired,
  class_first_pump, first_pump_in_grid, chowlin_phase (A/B; '' = not in the Chow-Lin set),
  rho, n_pairs_resplit, profile_member, profile_county, chowlin_shipped.
- `zero_class.parquet`: one row per zero meter-month of the window (pre-origin 2012-01..2022-12,
  full 2012-01..2026-06). Columns: zclass (structural, not_yet_used, idle, long_idle,
  terminated, trailing_short, never_used), run_len, sync_suspect, flag (0 or 9), month,
  possible_replacement.
- `meter_zero_summary.parquet`: first_nz, last_nz, terminated, termination,
  last12_kwh_before_termination, c0_terminated (plus the *_t indices).
- `incomplete_pair_mask.parquet`: month, reason (open_bill_jul2026_phaseA,
  pair_one_month_missing, partial_cycle_jul2026_monthly, bill_in_second_month), C0 and arm
  flags and methods.
- `county_class_monthly.parquet`: county x scope (in_grid, county) x class (+ all,
  all_nodomestic) x date x arm, with value, sd, flag, method and counts.
- `profiles.npz`: prof_<county> (the fan-wide profile), a_month_/b_month_<county> (phase-A/B
  population sums), n_A_, n_B_, t_end_.
- `july2026_imputation.parquet` (full only): imputed recorded/Chow-Lin kWh and sd per meter,
  shipped.
"""


def md() -> str:
    acc = TAB["acc"]
    lines = [
        "# WP P2abc report: billing phase, Chow-Lin split, record-end mask and zero classes",
        "",
        "Round 2 (on WP C0's census, Yunlin + Changhua + in-grid Chiayi). Built, scored and "
        "published 2026-10-14 (UTC 2026-10-13 20:15-20:56); report, ledger and spot checks "
        "added by the resumed session on 2026-10-14.",
        "",
        "**Status: passed.** Both product gates pass. Among the non-gating results, the "
        "registered head check fails in both counties, Changhua's Chow-Lin sd is "
        "uncalibrated, and Chiayi's July imputation fails and is withheld.",
        "",
        "## Summary",
        "",
        SUMMARY,
        "",
        "## Products",
        "",
        "| product | content | allowed uses | fit_window_end |",
        "|---|---|---|---|",
        f"| `results/dataprep/yunlin_electricity/preorigin/v1` (L2) | 2007-01..2022-12, both arms, "
        f"phase table, profiles, zero classes, record-end mask | "
        f"{', '.join(MP['allowed_uses'])} | {MP['fit_window_end']} (non-exempt "
        f"{MP.get('non_exempt_fit_window_end')}; 2 exemptions inherited from C0) |",
        f"| `results/dataprep/yunlin_electricity/full/v1` (L2) | 2007-01..2026-07, same layout + "
        f"July-2026 imputation | {', '.join(MF['allowed_uses'])} | {MF['fit_window_end']} |",
        "",
        "Evidence: `results/dataprep/reports/P2abc_evidence/` (tests/, design_time/, "
        "stage1.json, r1_reproduce.json, checker.json, guard_probe.txt, report_facts.json, "
        "report_tables.json). Ledger: `results/dataprep/ledger/rows/P2abc.csv`.",
        "",
        "## Method (as registered in P2abc.r2.c0_based.v1)",
        "",
        "- **Phase.** C0's detected billing phase of the variant. Pre-origin uses "
        "`phase_detected_2012_2022`, full uses `phase_detected`. The billing code (M2 = A, "
        "M1 = B) selects the profile populations together with the detected phase.",
        "- **Profile.** One per county (Chiayi uses Yunlin's). It is a stagger deconvolution of "
        "the in-grid kept irrigation phase-A (M2 & A) and phase-B (M1 & B) population sums, "
        "with lambda 0.01 and a 12-month share window. Only pairs complete by the variant end "
        "enter (pre-origin: 2022-12).",
        "- **Chow-Lin.** Model x = X beta + u with X = [1, p/mean p] and AR(1) u. rho is chosen "
        "per meter by likelihood from {0, 0.5, 0.9}, over the window 2011-01..variant end. A "
        "pair is re-split only when both months are installed flag-0 observations (methods "
        "observed, observed_before_install_date, terminated_trailing_run, bill_in_second_month, "
        "uneven_pair). The split is clipped to [0, Y], so the pair sum is kept. Every other "
        "month keeps C0's value, flag and method.",
        "- **sd.** q90(n_eff, month)/1.645 x the even-split value, from the validation groups "
        "pooled over both easting halves, log-linear in n_eff and clamped to [1, 300].",
        "- **July 2026 (full).** The open phase-A month is imputed with the meter's own 3-year "
        "same-month mean of the Jul-Aug pair mean (flag 4, method 11), where the county's "
        "backtest passed.",
        "- **Zero classes (P2c).** Port of the pilot's s05(e) and s09 per county. Windows are "
        "2012-01..2022-12 (pre-origin) and 2012-01..2026-06 (full). Months flagged by the "
        "synchronized-zero screen are marked flag 9 (sensitivity member only).",
        "",
        "## Pre-registrations and looks",
        "",
    ]
    for pid, h in PREREGS.items():
        lines.append(f"- `{pid}` sha256 `{h[:16]}...`")
    lines += [
        "",
        "The round-1 registrations (Yunlin AMP_V2 cache, 2026-10-12, outside the A1 index) "
        "stand. Four were scored by the interrupted round-1 session; those looks are logged as "
        "back-dated design-time seeds, and r1_reproduce.py recomputed them identically. "
        "P2a.head_placebo.v1 was scored for the first time in round 2. Every round-2 score was "
        "logged by the A1 tool (assert_can_score). This resumed session logged its own reading "
        "of the scored results ([P2-resume-review], [P2-resume-review-heads]) and its "
        "spot-check run ([P2-facts]) before running anything.",
        "",
        "## Acceptance",
        "",
        "| criterion | threshold | measured | pass |",
        "|---|---|---|---|",
    ]
    for a in acc:
        lines.append(f"| {a['criterion']} | {a['threshold']} | {a['measured']} | "
                     f"{'PASS' if a['passed'] else 'FAIL'} |")
    lines += [
        "",
        "Gates (conjunctive, registered): yunlin_electricity/preorigin = "
        "P2r2.k1r.construction_preorigin and P2r2.c1.zero_port_yunlin; "
        "yunlin_electricity/full = P2r2.k1r.construction_full and P2r2.c1.zero_port_yunlin. "
        "Both passed. The meter tests decide whether each county's Chow-Lin arm ships (all "
        "shipped). The July backtest decides whether each county's July-2026 imputation ships "
        "(Yunlin and Changhua yes, Chiayi no). The coverage tests set the sd labels only. The "
        "head check is reported only.",
        "",
        "## Results in brief",
        "",
        "**Meters (P2a).** On pseudo-bimonthly monthly meters, the pre-origin Chow-Lin operator "
        "beats the even split at every group size: Yunlin skill 0.232-0.324, Changhua "
        "0.205-0.414, and fan-wide 0.50 in both. Pro-rata by the same profile is worse than the "
        "even split in Yunlin at every size (n = 1: -0.259). A year-shuffled placebo profile "
        "keeps most of the skill for single meters and small groups (Yunlin n = 1: placebo mean "
        "0.224 against 0.232). Year-specific timing pays off at aggregate scale (Yunlin n = 300: "
        "0.324 vs 0.056; fan: 0.501 vs 0.048). For Changhua the real profile beats every "
        "placebo at every size (n = 1: 0.205 vs placebo max 0.179).",
        "",
        "**Head check (P2a, registered primary).** D_w is the partial-R2 gain of local energy "
        "with the real profile over 20 year-shuffled placebos, at 1-2 km. The regional "
        "covariates (the county total and kWh more than 15 km away) are recomputed from each "
        "energy version. D_w is +0.0054 in Yunlin (p 0.051), -0.0012 in Changhua (p 0.786) and "
        "+0.0027 pooled (p 0.099). Round 1 on the AMP_V2 energy gave +0.0028 (p 0.139). All "
        "fail p < 0.05, so the gain is not distinguishable from regional-signal injection. The "
        "registered fixed-nuisance secondary, which uses the recorded regional covariates in "
        "every fit, gives p 0.003 (Yunlin), 0.005 (Changhua) and 0.002 (pooled). It is a "
        "secondary, changes no decision and is not promoted. Its effect sizes are below 0.01 "
        "partial R2.",
        "",
        "**July (P2b).** Errors are the rolling-origin fan-total |error| over 2013-2025. Yunlin "
        "is 15.4 % (max 32.1 %; 1 km cell relative RMSE 0.345), Changhua 8.7 % (max 16.5 %; "
        "0.214), and Chiayi 42.2 % (max 157 %; 98 phase-A meters), which fails. Imputed in "
        "July 2026: 55,591 of 55,606 open Yunlin phase-A meters and 45,423 of 45,436 Changhua "
        "meters. Chiayi's 101 stay NaN (flag 8).",
        "",
        "**Zeros (P2c).** The port reproduces the pilot's run classes and the irrigation zero "
        "share in April (0.0559 vs 0.0558) and August (0.1564 vs 0.1563) within 0.17 %. The "
        "pre-origin classes and synchronized-zero months per county are in the key numbers.",
        "",
        "**Independent spot checks (resumed session, after the fact; r5_resume.py).** Per-pump "
        "totals over the Chow-Lin window equal C0's to 6e-11 kWh in both variants and all "
        "counties. The recorded arm equals C0's cells on every cell-month without an imputed "
        "pump (2,474,496 and 3,025,045 cell-months, max 2.9e-11 kWh, 0 flag or NaN "
        "differences). A first version of the per-pump check stopped the full window at 2026-06. "
        "That cut the last phase-B pair and showed spurious differences; it was corrected "
        "before this report.",
        "",
        "## Key numbers (ledger ids in brackets)",
        "",
    ]
    lines += [f"- {x}" for x in TAB["kn"]]
    lines += [
        "",
        "## Anti-leakage (U1-U7)",
        "",
        "- Every fitted quantity of the pre-origin product uses electricity to 2022-12 only: "
        "phase, profiles, beta and rho, the sd model (validation months 2012-02..2022-11), the "
        "zero classes and the synchronized-zero screen. So do the Chow-Lin shipping decisions "
        "and sd labels behind it. Its remaining post-origin information is C0's two exemptions, "
        "listed in the manifest with ledger ids "
        "P2abc.r2.exemption.census_attributes.preorigin and "
        "P2abc.r2.exemption.residual_unpaired_dec2022.preorigin and C0's sensitivity runs. The "
        "zero-port gate input is disclosed under deviations.",
        "- No head, ring, pile, GNSS or InSAR observation dated 2023-01 or later enters any "
        "test, gate, threshold or choice (U7). The head check cuts the daily record at "
        "2022-12-31 and uses months 2012-01..2022-11. The July backtest and the zero port use "
        "2023-2025 electricity, and they ship or gate only full-variant content or code "
        "identity.",
        "- Imputed values (flags 3 and 4) are forcing inputs with sd, never targets (U1/U2). "
        "Both manifests forbid scoring_target, baseline, datum_fit, calibration_target and "
        "selection.",
        "- The full product is refused for forcing_frozen and model_design (U4), as the guard "
        "probe shows.",
        "",
        CONVENTIONS,
        "",
        "## Deviations from plan",
        "",
    ]
    lines += [f"- {x}" for x in DEVIATIONS]
    lines += ["", "## Issues for downstream", ""]
    lines += [f"- {x}" for x in ISSUES]
    lines += [
        "",
        "## Provenance",
        "",
        "- Code (round 2): " + ", ".join(f"`{Path(p).relative_to(R)}`" for p in ROUND2_CODE),
        "- Code (round 1, kept unchanged; its registrations stand): "
        + ", ".join(f"`{Path(p).relative_to(R)}`" for p in ROUND1_CODE),
        "- Library: dataprep_staging/common (guard, manifest, ledger, check), harness.prereg3 "
        "(A1). Read-only repository modules: none at run time beyond C0's products.",
        "- Compute: CPU only, single-threaded via cpu_slot.sh. Products were built in 95 s "
        "(pre-origin) and 120 s (full); the spot checks took 21 s.",
        "- Claude API tokens: 0. GPU: none. Nothing under hydrophysics/, tests/ or L0 was "
        "written, and nothing was staged in git.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    acc = [{k: a[k] for k in ("criterion", "threshold", "measured", "passed")}
           for a in TAB["acc"]]
    rep = {
        "wp": "P2abc",
        "status": "passed",
        "summary": SUMMARY,
        "prereg": {pid: {"sha256": h} for pid, h in PREREGS.items()},
        "acceptance": acc,
        "acceptance_number_ids": [a["number_id"] for a in TAB["acc"]],
        "products": [
            {"path": "results/dataprep/yunlin_electricity/preorigin/v1", "layer": "L2",
             "manifest": "results/dataprep/yunlin_electricity/preorigin/v1/MANIFEST.json "
                         "(+ PRODUCT_CARD.md, validation.json)",
             "allowed_uses": ", ".join(MP["allowed_uses"]),
             "fit_window_end": f"{MP['fit_window_end']} (non_exempt_fit_window_end "
                               f"{MP.get('non_exempt_fit_window_end')}; 2 exemptions "
                               "inherited from C0)"},
            {"path": "results/dataprep/yunlin_electricity/full/v1", "layer": "L2",
             "manifest": "results/dataprep/yunlin_electricity/full/v1/MANIFEST.json "
                         "(+ PRODUCT_CARD.md, validation.json)",
             "allowed_uses": ", ".join(MF["allowed_uses"]),
             "fit_window_end": MF["fit_window_end"]},
        ],
        "deviations_from_plan": DEVIATIONS,
        "issues_for_downstream": ISSUES,
        "conventions_for_P2def": CONVENTIONS,
        "key_numbers": TAB["kn"],
        "ledger": {"fragment": "results/dataprep/ledger/rows/P2abc.csv",
                   "rows": TAB["n_ledger_rows"], "number_ids": TAB["ids"]},
        "code_paths": ROUND2_CODE + ROUND1_CODE,
        "evidence": "results/dataprep/reports/P2abc_evidence",
        "claude_api_tokens": 0,
        "gpu": "none",
        "resumed_session": "2026-10-14: verified hashes (checker), logged review looks, ran "
                           "r5_resume.py facts, wrote ledger rows (r6_report.py) and this "
                           "report (r7_write_report.py); nothing rebuilt or re-scored",
    }
    (REP / "P2abc.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1),
                                    encoding="utf-8")
    (REP / "P2abc.md").write_text(md(), encoding="utf-8")
    print("wrote", REP / "P2abc.md", REP / "P2abc.json")


if __name__ == "__main__":
    main()
