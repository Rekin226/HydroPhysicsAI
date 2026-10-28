"""Erratum to X5.storage_priors.v2_postfailure (written once; report.json is left as written).

The declaration states the systematic method term as "half the median inland log10(Ss_M2 /
Ss_BE)". run_x5.systematic() restricted it to wells whose combined source is 'M2+BE' (M2
usable: earth-flagged, or undetermined and inland), which left 2 wells. This script applies
the declared wording (every inland well with both Ss_M2 and Ss_BE) to the same per-well
estimates and recomputes the systematic term and the prior sd of every group of
ss_prior_table.csv. Nothing else changes. Found after the report was written (post-report).

Usage: python -I erratum_sys.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import x5sp_const as K  # noqa: E402
import x5sp_lib as L  # noqa: E402
from common.hashing import sha256_file  # noqa: E402
from common.paths import rel_to_repo  # noqa: E402


def main() -> int:
    out = K.REPORT_DIR / "erratum_systematic.json"
    if out.exists():
        raise SystemExit(f"{out} exists")
    d = pd.read_csv(K.WORK / "well_estimates_derived.csv", dtype={"sid": str})
    pt_path = K.PRIORS_V1 / "ss_prior_table.csv"
    pt = pd.read_csv(pt_path)
    lo, hi = K.N_POR_RANGE
    por = 0.5 * abs(np.log10(hi / lo)) * 0.5
    nlo, nhi = K.NU_RANGE
    nu = 0.5 * abs(np.log10(L.c_nu(nlo) / L.c_nu(nhi))) * 0.5
    s = d[d["inland"] & d["ss_m2"].notna() & d["ss_be"].notna()]
    med = float(np.median(np.log10(s["ss_m2"] / s["ss_be"])))
    disc = 0.5 * abs(med)
    tot = float(np.sqrt(por ** 2 + nu ** 2 + disc ** 2))
    pt["sys_log10_declared"] = tot
    pt["prior_sd_log10_declared"] = np.sqrt(pt["se_median_log10"] ** 2 + tot ** 2)
    keep = ["grouping", "layer_code", "zone", "CountyName", "quantity", "n_wells", "status",
            "median", "median_log10", "se_median_log10", "sys_log10", "prior_sd_log10",
            "sys_log10_declared", "prior_sd_log10_declared"]
    rows = pt[[c for c in keep if c in pt.columns]].replace({np.nan: None}).to_dict("records")
    res = {"prereg_id": K.PREREG_ID, "label": "post-report erratum (implementation deviated "
           "from the declared wording); post-failure exploratory like its parent",
           "declared_wording": "half the median inland log10(Ss_M2/Ss_BE)",
           "implemented_in_report": "restricted to ss_source == 'M2+BE' (n 2): 0.0792 half "
                                    "log10, total 0.1132",
           "declared": {"n_wells": int(len(s)), "median_log10_ss_m2_over_ss_be": med,
                        "method_half_log10": disc, "porosity_half_log10": por,
                        "poisson_half_log10": nu, "total_sys_log10": tot},
           "groups": rows,
           "source": {rel_to_repo(K.WORK / "well_estimates_derived.csv"):
                      sha256_file(K.WORK / "well_estimates_derived.csv"),
                      rel_to_repo(pt_path): sha256_file(pt_path)}}
    with open(out, "x", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res["declared"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
