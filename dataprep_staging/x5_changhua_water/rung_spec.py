"""Write the stage-2 rung specification (a DRAFT for the lead to register; nothing is run).

Output: results/dataprep/reports/X5.changhua_water.v1/rung_spec_DRAFT.json (refuses to
overwrite). The lead registers it (harness.prereg3) before the GPU batch runs.

Usage: python -I rung_spec.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.hashing import sha256_file  # noqa: E402
from common.paths import rel_to_repo  # noqa: E402

from x5_changhua_water import x5const as K  # noqa: E402

PRE = K.OUT / "preorigin" / "v1"
Q0_RECIPE = ("--param-mode zonal --epochs 500 --n-folds 5 --fix-eta 0.5 --fix-head-extra 40 "
             "--return-flow --l-min 1e-4 --learn-spread --spread-max-km 10 "
             "--ic-merged-proximal --ground-elev dem --zone-boundaries 205,182,208 "
             "--log-t-min-proximal 58 --well-datum fit --well-datum-sd 5 --device cuda "
             "--compile-matvec --log-every 25 --seed 2027 --fold-site-tol-m 200 --pump-split "
             "--wells-dir results/dataprep/twin_inputs/clean_heads/v1/wells "
             "--stations results/dataprep/twin_inputs/clean_heads/v1/fan_stations.parquet "
             "--day-weighted-months")


def main() -> int:
    out = K.REPORT_DIR / "rung_spec_DRAFT.json"
    if out.exists():
        raise SystemExit(f"{out} exists")
    files = {n: {"path": rel_to_repo(PRE / f"x5_t2_{n}.npz"),
                 "sha256": sha256_file(PRE / f"x5_t2_{n}.npz")} for n in ("service", "paddy")}
    spec = {
        "status": "DRAFT, not registered; the lead registers it before any run",
        "suggested_id": "I1.X5.v1", "wp": "I1", "kind": "declaration",
        "title": "Rung X5: Q0 + Changhua irrigation water outside the Jiji supply as canal "
                 "recharge (T2 timing, X4 service-map placement); alternative placement all "
                 "Changhua paddy",
        "base_model": "Q0 (results/twin_runs/par20/jobs.sh RECIPE; I1.R4Q0.v1)",
        "arms": {
            "X5s (primary)": Q0_RECIPE + f" --sw-recharge {files['service']['path']} "
                                         "--sw-components delivered",
            "X5p (declared alternative placement, reported beside, never substituted)":
                Q0_RECIPE + f" --sw-recharge {files['paddy']['path']} --sw-components delivered"},
        "inputs": files,
        "what_enters": (
            "x5_changhua_water/preorigin/v1 value_t2: MOA Changhua office total minus the Jiji "
            "Changhua delivery (whole office, 10^8 m3/yr; 2016-2022 mean 3.50), timed by "
            "calendar x exp(b.z) with z = coherence and river balance (no heads, so the "
            "rung's head scores are not scored on heads that shaped its forcing), Denton "
            "to the annual totals, spread on Changhua cells (in-fan fraction 0.965); "
            "m/day of water APPLIED. The recharge fraction is the calibration's one learned "
            "scalar log_sw_scale in [0.01, 2] (Changhua cells only carry the field). "
            "Forward runs past 2022 use the field's month-of-year climatology "
            "(twin.forward, as R1b)."),
        "partition": "calibrate_flow --seed 2027 --fold-site-tol-m 200; leveling "
                     "calibrate_coupled --seed 2027 (design-time partition, as I1.B.v1); "
                     "promotion needs a confirmation on the never-scored seed 2029",
        "gates_against_Q0_all": [
            "k-fold out-of-fold R2 >= Q0 - 0.01 (Q0 0.648 -> 0.638)",
            "county_corr (G2 definition, score_g2.py) Changhua >= Q0 - 0.02 (0.448), Yunlin "
            ">= Q0 - 0.02 (0.720)",
            "county_mean_rmse Changhua <= Q0 x 1.02 (0.542 m), Yunlin <= Q0 x 1.02 (1.000 m)",
            "leveling out-of-fold R2 >= Q0 - 0.02 (0.633)",
            "temporal screen (--holdout-months 36 --fit-only) fair RMSE ratio <= Q0 x 1.01 "
            "(1.0768) and fair verdict PASS",
            "forward plausibility: county-mean VEP subsidence rate Dec 2022 -> Dec 2025 / 3 "
            ">= 0 cm/yr in Changhua and in the rest of the fan (county map v2)",
            "X5-specific, amplitude: A = median over Changhua held-out entries (>= 60 paired "
            "finite months, stage3_per_entry.npz) of nanstd(interannual(pred)) / "
            "nanstd(interannual(obs)), interannual() = dataprep_staging/i1/"
            "diag_changhua_interannual.py (own-mean anomaly minus month-of-year "
            "climatology, centred 12-month mean); passes if |ln A_rung| < |ln A_Q0|",
            "X5-specific, no loss of held-out correlation: Changhua median per-entry "
            "interannual correlation (same series, >= 24 paired months) >= Q0's - 0.02, in "
            "addition to the county_corr gates above"],
        "before_registration": "compute A_Q0 and Q0's Changhua interannual correlation from "
                               "results/twin_runs/stage3_datum_gate_q0/stage3_per_entry.npz "
                               "with the same code and write them into the thresholds (Q0 "
                               "is spent; these are design-time numbers)",
        "reported_not_gating": ["fitted sw_scale (if it sits at the 0.01 bound, the model "
                                "rejects the term; reported as such)",
                                "P2f Yunlin ratio (as B1)", "reported:amplitude per county",
                                "leveling R2 by county"],
        "attempt_policy": {"scored_runs": "one per arm on seed 2027", "retries": "engineering "
                           "only, before any gate metric", "exploratory_followups_max": 1,
                           "followup_label": "post-failure, exploratory; never validation; "
                                             "cannot promote", "thresholds_frozen": True},
        "disclosed_in_advance": [
            "R1e-cn put both Changhua canal fractions at the 0.01 lower bound and R1b-R1e "
            "never moved Changhua's interannual correlation above 0.52 except canal water "
            "alone (R1b 0.63); the model may reject this term too",
            "a steady extra recharge damps the head response; Q0's Changhua response is "
            "already small (amplitude 0.36 of observed, total anomaly), so damping alone "
            "would move the amplitude ratio AWAY from 1; only the term's year-to-year "
            "variation (whole-office 5.6-8.9 x10^8 m3/yr in 2012-2015 vs 2.0-4.8 after 2016, "
            "partly a 2016 reporting change) adds interannual forcing",
            "the X5 timing agreed with W1's inferred Yunlin monthly values (r 0.965 >= 0.6) "
            "but the calendar alone scored 0.974: the monthly shape is essentially W1's "
            "calendar; the indicators add weak anomaly information (anomaly r 0.31 T3, "
            "0.36 T2)",
            "annual sd is large (definitional sd 2.62 x10^8 m3/yr against a 3.50 level), so "
            "the volume is uncertain by about 75 % even before the 2016 step"],
        "written_by": rel_to_repo(Path(__file__)),
    }
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(spec, fh, indent=1, ensure_ascii=False)
    print(rel_to_repo(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
