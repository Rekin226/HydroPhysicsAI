"""Repair-round pre-registration of WP P2abc (A1 tool ``harness.prereg3``), written after the
code of the round was debugged (synthetic test; Chiayi-only scratch run, look
[P2-r3-debug-chiayi]) and before the repair-round products are built or any registered check
is scored.

Usage: python -I prereg_r3.py [--dry-run]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (_HERE.parents[2], _HERE.parents[1], _HERE.parent / "v2", _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import p3lib as P3  # noqa: E402
import prereg_v2 as P  # noqa: E402  (v2, read-only: none_unit, NONE)
from harness import prereg3  # noqa: E402
from harness.prereg2 import not_applicable  # noqa: E402
from p3lib import L  # noqa: E402

PREREG_ID = "P2abc.r3.repair.v1"
PILOT = L.DATAPREP / "_pilots" / "billing_and_census_gaps"
L0P = L.REPO / "data_fetch_api" / "new_measurements" / \
    "tpc_changhua_yunlin_chiayi_pingtung_2026" / "parquet"


def inputs() -> list[str]:
    out = []
    for v in ("preorigin", "full"):
        d = L.C0_DIR[v]
        out += [L.rel(d / "MANIFEST.json"), L.rel(d / "cells_by_class.npz"),
                L.rel(d / "cells_total.npz"), L.rel(d / "county_class_monthly.parquet")]
        for c in L.COUNTIES:
            t = L.FILETAG[c]
            out += [L.rel(d / f"pumps_{t}.npz"), L.rel(d / f"meters_{t}.parquet")]
    out += [L.rel(L.C0_L1 / "MANIFEST.json"), L.rel(L.C0_L1 / "census_pumps.parquet")]
    out += [L.rel(p) for p in sorted(L.WORK.iterdir()) if p.suffix in (".npz", ".parquet")]
    out += [L.rel(L.EVID / "tests" / f) for f in ("t1_meter_validation_preorigin.json",
                                                   "t1_meter_validation_full.json",
                                                   "t3_july_backtest.json")]
    out += [L.rel(PILOT / f) for f in ("meter_mat.npy", "meters_phase.parquet", "dates.npy",
                                       "zero_runs_summary.csv", "census_zeros.json",
                                       "s05_census_zeros.py")]
    out += [L.rel(L.PRODUCT_ROOT / "_work_p2abc" / "zeros.json")]
    out += [L.rel(L0P / f"{c}_census.parquet") for c in L.COUNTIES]
    return out


def tests() -> list[dict]:
    N = P.none_unit
    T = [{
        "test_id": "P2r3.c1.zero_shipped_yunlin",
        "data_scope": "c0_pump_electricity_v3:2007-01..2025-07",
        "fresh_units": ["P2r3:zero_shipped_yunlin"],
        "design_time_units": ["P2r2:zero_port_yunlin", "P2r1:yunlin_amp_zero_classification"],
        "external_units": {
            "P2r3:zero_shipped_yunlin": N("Yunlin kept meters of C0 full v3 in the billing "
                                          "pilot's view, classified by the shipped "
                                          "zeros3.classify_core (see t6_zero_shipped.py)"),
            "P2r2:zero_port_yunlin": N("round-2 zero check units (same meters and window)"),
            "P2r1:yunlin_amp_zero_classification": N(P.R1_UNITS[
                "P2r1:yunlin_amp_zero_classification"])},
        "what": "plan P2c acceptance on the SHIPPED classifier (zeros3.classify_core, the "
                "function that writes zero_class.parquet of yunlin_electricity v2): pilot view "
                "of the C0 full v3 Yunlin record (2007-01..2025-07; non-observations NaN; bills "
                "posted wholly to the second month split evenly; July 2025 of code-M2 meters "
                "set to 0 as in the AMP_V2 cache; phase by the pilot rule on 2012-01..2025-06; "
                "kz = phase A, B or monthly; window 2012-01..2025-06)",
        "metric": "relative difference to the pilot of the kz zero-months in the shipped "
                  "classes idle (interior <= 12), long_idle (interior > 12), terminated "
                  "(trailing >= 12), trailing_short (trailing < 12); irrigation zero share in "
                  "April and August from the shipped rows (in-span finite months); structural "
                  "meter-months of kept meters (months before install month minus 1, install "
                  "index > 12, record 2007-01..2025-07): shipped C0 pre_install vs the pilot "
                  "archive's 0-or-missing months (install date floored to its month)",
        "threshold": "each of the 7 items within 1 % (plan P2c)",
        "gating": "part of both products' gates (yunlin_electricity/{preorigin,full}/v2); on "
                  "failure the products are published with gate FAILED and the WP is partial",
        "secondary": ["not_yet_used (leading runs)", "never_used_in_window rows",
                      "the pilot's s05 pre-install statistic recomputed (plan figure 878,738, "
                      "all meters, zero readings only) and its decomposition"]}]
    for v in ("preorigin", "full"):
        u = f"P2r3:assembled3_{v}"
        T.append({
            "test_id": f"P2r3.k1.construction_{v}",
            "data_scope": ("c0_pump_electricity_v3:2007-01..2022-12" if v == "preorigin" else
                           "c0_pump_electricity_v3:2007-01..2026-07"),
            "fresh_units": [u],
            "design_time_units": ["P2r3:debug_chiayi_subset"],
            "external_units": {u: N(f"the repair-round assembled yunlin_electricity/{v} "
                                    f"product (built after this registration)"),
                               "P2r3:debug_chiayi_subset": N(
                                   "Chiayi-only scratch assembly used to debug the code "
                                   "(look [P2-r3-debug-chiayi]; all items 0 there)")},
            "what": "construction checks (a)-(l) of t7_construction3.py: (a) Chow-Lin pair "
                    "sums; (b) Chow-Lin arm = C0 outside methods chowlin_split / "
                    "own_3yr_same_month_mean / m0_july2025_suspect; (c) pre-install zeros; "
                    "(d) recorded cells = C0 cells (flag also except the M0 July-2025 HP rule); "
                    "(e) sums; (f) sd and flag rules (every arm); (g) privacy; (h) pre-origin "
                    "manifest inputs (no pump_electricity/full input; post-origin inputs only "
                    "under an exemption entry); (i) zero table rows, exact re-run of "
                    "zeros3.classify, structural = C0 pre_install, sync rows flag 9 and never "
                    "structural, zclass consistent; (j) M0 July-2025 flags exactly on the "
                    "kept code-M0 monthly meters with a flag-0 2025-07 reading (full; none in "
                    "pre-origin); (k) member cells rebuilt per pump from C0, the Chow-Lin arm, "
                    "sync_fill.parquet and july2026_imputation.parquet; fills only on "
                    "sync-suspect zero rows, finite, >= 0; (l) mask table lists every "
                    "sync-suspect zero month and every M0 suspect",
            "metric": "violations per item",
            "threshold": "0 violations on every item; 1e-6 kWh + 1e-9 x |reference| for value "
                         "equalities",
            "gating": f"yunlin_electricity/{v}/v2 gate (with P2r3.c1.zero_shipped_yunlin)"})
    for c in L.COUNTIES:
        u = f"P2r3:july_member_units_{c}"
        T.append({
            "test_id": f"P2r3.b2.july_syncfill_{c}",
            "data_scope": "c0_pump_electricity_v3:2010-07..2025-08",
            "fresh_units": [u],
            "design_time_units": [f"P2r2:july_units_{c}"],
            "external_units": {
                u: N(f"{c}: kept code-M2 meters with full-variant detected phase A, Julys "
                     f"2010-2025, on the recorded synchronized-zero member of full v2"),
                f"P2r2:july_units_{c}": N(f"round-2 July units of {c} (scored; known)")},
            "what": "round 2's July backtest (v2/t3_july_backtest.backtest, unchanged) on the "
                    "recorded synchronized-zero member (every synchronized-zero meter-month "
                    "replaced by its fill), Y = 2013..2025; also Y = 2013..2024 for the member "
                    "and the recorded arm",
            "metric": "mean |fan-total error| (%) over the years",
            "threshold": "reported against the plan's 20 % (no decision)",
            "gating": "none (reported only); the registered round-2 verdicts P2r2.b1 stand and "
                      "ship the July-2026 imputation"})
    return T


def doc() -> dict:
    ts = tests()
    x1 = L.load_json(P3.EVID3 / "design_time" / "x1_explore_2025.json")

    def pr(c, g, k):
        v = x1[c][g][k]
        return {"total_GWh": round(v["total_GWh"], 2), "zshare": round(v["zshare"], 3),
                "next_ratio_zero": round(v["next_ratio_zero"], 2),
                "next_ratio_nonzero": round(v["next_ratio_nonzero"], 2)}
    ev = {c: {k: pr(c, "A", k) for k in ("2019-07", "2020-07", "2021-07", "2022-07",
                                          "2023-07", "2024-07", "2025-07")}
          for c in ("yunlin", "changhua")}
    return {
        "id": PREREG_ID, "wp": L.WP,
        "title": "P2abc repair round: shipped zero classifier scored against the pilot, "
                 "construction checks of yunlin_electricity v2 (M0 July-2025 flag, "
                 "synchronized-zero members), July backtest on the member (reported)",
        "kind": "test", "tests": ts,
        "fresh_units": sorted({u for t in ts for u in t["fresh_units"]}),
        "zones": not_applicable("no zones; counties are the strata"),
        "priors": not_applicable("no priors"),
        "burn_in": not_applicable("no model run"),
        "folds": not_applicable("rolling-origin years as in round 2 (reported test only)"),
        "thresholds": {t["test_id"]: t["threshold"] for t in ts},
        "gate_scores": not_applicable(
            "conjunctive gates, no weights: yunlin_electricity/preorigin/v2 = "
            "P2r3.k1.construction_preorigin and P2r3.c1.zero_shipped_yunlin; "
            "yunlin_electricity/full/v2 = P2r3.k1.construction_full and "
            "P2r3.c1.zero_shipped_yunlin. Chow-Lin shipping, sd labels and the July-2026 "
            "imputation keep their round-2 registered verdicts (P2r2.a1, P2r2.a2, P2r2.b1); "
            "Chiayi's Chow-Lin sd is relabelled 'uncalibrated' (never checked on Chiayi "
            "meters, plan 4.7)."),
        "primary_model": (
            "Round 2's products (yunlin_electricity v1: recorded and Chow-Lin arms, sd, July-2026 "
            "imputation) with: (1) zero tables from zeros3.classify (billing pilot s05(e) run "
            "kinds on the observed matrix; structural column = C0 pre_install; pair-consistent "
            "synchronized-zero marks, flag 9); (2) full product: M0 July 2025 flagged 9 (plan "
            "P2b; code M0, detected phase monthly, flag-0 2025-07 reading; value kept; method "
            "m0_july2025_suspect; aggregate flag 9 when such pumps hold > 10 % of the installed "
            "HP); (3) sensitivity members recorded_syncfill and chowlin_syncfill (plan P2c: "
            "synchronized zeros enter as a sensitivity member only): synchronized-zero "
            "meter-months filled with the meter's own mean of the same calendar month over the "
            "3 most recent eligible earlier years (<= 6 back), flag 4, sd = in-sample q90/1.645 "
            "x fill (uncalibrated); full: members' July-2026 imputation recomputed from the "
            "member record; (4) mask table reasons sync_zero_month and m0_july2025_suspect; "
            "(5) sync_fill.parquet."),
        "inputs": inputs(),
        "interpretation": {
            "repair_items": {
                "P2c": "Round 2's registered zero check (P2r2.c1) scored a re-implementation "
                       "(t4_zero_port.pilot_stats), not the classifier that wrote "
                       "zero_class.parquet (b2_zeros.classify, different rules). The verifier "
                       "ran b2_zeros.classify on the pilot window: long_idle -1.76 %, "
                       "terminated +7.3 % (fails 1 %). This round ships zeros3.classify, whose "
                       "run kinds are the pilot's rules (synthetic equivalence with a literal "
                       "transcription of s05(e): exact), and scores that function. The "
                       "structural class is a separate column (C0 pre_install) checked against "
                       "the pilot's pre-install count on kept meters under one month "
                       "convention.",
                "P2b_m0": "The plan's M0 July-2025 flag 9 is applied in the full product. Its "
                          "trigger (Jul/Jun 1.096 Yunlin, 1.091 Changhua vs 1.16-1.26 in "
                          "2022-24) holds against the plan's 3-year reference but not against "
                          "2012-2024 (0.87-1.30); the flag is applied as the plan says, the "
                          "values are kept.",
                "P2b_2025_phaseA": "Yunlin's phase-A Jul-Aug 2025 bills (zero share 0.164 of "
                                   "active meters, 8.54 GWh) are treated as the plan's "
                                   "synchronized zeros, not as incomplete pairs: the zero-bill "
                                   "meters' next bill is not raised (next_ratio_zero 1.15 vs "
                                   "1.12 for positive-bill meters; a missed cumulative reading "
                                   "would show about 2), the same pattern occurs in 2019 and "
                                   "2021 (pre-origin synchronized months 2019-07/08, 2021-07/08 "
                                   "of the plan's list), the delivery records these months as "
                                   "0 (not missing; both sheets record missing readings as NaN "
                                   "elsewhere) and Changhua shows no excess. The plan's rule "
                                   "for them is flag 9 in the data and a sensitivity member; "
                                   "the members deliver the masked-and-imputed alternative.",
                "evidence_design_time": ev},
            "design_after_seeing_data": [
                "pair-consistent synchronized-zero marks (round 2 flagged 2025-08 but not "
                "2025-07 of the same zero bills)",
                "bills posted wholly to the second month (C0 method 7) split evenly in the "
                "classifier and screen input (posting artefacts are not zero readings); this "
                "changes the screen input from 2025-07 and for a few thousand earlier pairs "
                "(2013, 2021-12/2022-01), so screened months can differ from round 2's; "
                "the Chiayi debug run showed the pooled Yunlin+Chiayi full screen with "
                "2025-07 and 2025-08 flagged and 2021-07 not",
                "the M0 aggregate threshold (10 % of installed HP, C0's threshold for missing "
                "readings) instead of 'any pump'",
                "members fill every county (Chiayi included); no backtest decides the members"],
            "not_fresh": "P2r3.c1 runs on the same Yunlin meters and window as round 2's "
                         "P2r2.c1 and as the verifier's run of b2_zeros.classify; the pilot "
                         "reference numbers and round 2's outcome (0.17 %) are known, and so is "
                         "the verifier's result for the round-2 classifier (look "
                         "[P2-r3-verifier-disclosure]). The new classifier has not been run on "
                         "Yunlin or Changhua data before this registration (only synthetic data "
                         "and Chiayi). P2r3.b2 runs on round-2 July units whose recorded-arm "
                         "outcomes are known (and the verifier's 2013-2024 variant: 13.99 / "
                         "8.29 / 39.1 %).",
            "u3_u7": "Pre-origin product: zero tables on the pre-origin record (to 2022-12), "
                     "members' fills and q90 from data to 2022-12; the zero check (gate) reads "
                     "the Yunlin record to 2025-07 and is declared as an exemption entry of the "
                     "pre-origin manifest (no value depends on it). No head, ring, pile, GNSS or "
                     "InSAR observation enters anything (U7).",
            "code_at_registration": P3.v3_code()},
    }


def main(dry: bool) -> None:
    d = doc()
    errs = prereg3.validate_prereg(d)
    if errs:
        raise SystemExit("\n".join(errs))
    if dry:
        print(json.dumps({k: (v if k != "tests" else [t["test_id"] for t in v])
                          for k, v in d.items() if k not in ("inputs", "interpretation")},
                         ensure_ascii=False, indent=1)[:3000])
        print(len(d["inputs"]), "inputs")
        return
    path, sha = prereg3.write_prereg(d)
    print(path, sha)


if __name__ == "__main__":
    main("--dry-run" in sys.argv)
