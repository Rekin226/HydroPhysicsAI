"""Step 10: checks of the packaged ``heads/recovered_wells`` version.

* the manifest checker re-hashes the version (``common.manifest.verify_hashes``);
* the loader guard grants exactly the declared uses (calibration_target: flags 0-2;
  datum_fit and validator: flags 0-1) and refuses the others (scoring_target,
  initial_conditions, idw_field, selection, statistical_layer_history);
* every target row is pre-origin, and every row taken from ``heads/L2_filled_preorigin/v4``
  is identical to its source row (value, sd, flag, method, sigma_obs_m, u2_weight);
* no candidate or admitted well is one of the 158 model wells.

Writes ``H3_evidence/checks.json``.

Usage: python -I s10_checks.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2])]

import numpy as np  # noqa: E402
from common import guard, manifest  # noqa: E402

from h3_recovered_wells import config as C  # noqa: E402
from h3_recovered_wells import inventory  # noqa: E402
from h3_recovered_wells.s9_ledger import product_version  # noqa: E402


def main():
    pv = product_version()
    d = C.REPO / "results/dataprep" / pv
    m, _ = manifest.load_manifest(d / "MANIFEST.json")
    bad = manifest.verify_hashes(m, d)
    probes = {}
    for use in ("calibration_target", "datum_fit", "validator", "after_the_fact_analysis",
                "scoring_target", "initial_conditions", "idw_field", "selection",
                "statistical_layer_history", "prospective_scorer"):
        try:
            p = guard.open_product(d, use)
            t = p.read_table("targets_preorigin.parquet")
            probes[use] = {"granted": True, "rows": int(len(t)),
                           "flags": sorted(int(x) for x in t.flag.unique())}
        except Exception as e:  # noqa: BLE001 - the refusal is the result
            probes[use] = {"granted": False, "error": f"{type(e).__name__}: {str(e)[:160]}"}
    T = guard.open_product(d, "calibration_target").read_table("targets_preorigin.parquet")
    v4 = guard.open_product(C.H2_PRE_DIR, "calibration_target").read_table("fill.parquet")
    src = T[T.method != "observed_daily_snapshot"].merge(
        v4, on=["sid", "month"], suffixes=("", "_v4"), how="left")
    ident = {}
    for c in ("value", "sd", "sigma_obs_m", "u2_weight"):
        ident[c] = float(np.nanmax(np.abs(src[c] - src[f"{c}_v4"]))) if len(src) else 0.0
    ident["flag_equal"] = bool((src.flag == src.flag_v4).all())
    ident["method_equal"] = bool((src.method == src.method_v4).all())
    ident["unmatched_rows"] = int(src.value_v4.isna().sum())
    model = inventory.model_wells()
    rep = {"product_version": pv, "manifest_hash_problems": bad,
           "guard": probes, "identity_with_v4": ident,
           "max_month": T.month.max(), "rows": int(len(T)),
           "admitted_model_wells": sorted(set(T.sid) & model)}
    rep["expected_grants_ok"] = (
        all(probes[u]["granted"] for u in ("calibration_target", "datum_fit", "validator",
                                           "after_the_fact_analysis"))
        and not any(probes[u]["granted"] for u in ("scoring_target", "initial_conditions",
                                                   "idw_field", "selection",
                                                   "statistical_layer_history",
                                                   "prospective_scorer"))
        and set(probes["calibration_target"]["flags"]) <= {0, 1, 2}
        and set(probes["datum_fit"]["flags"]) <= {0, 1})
    (C.EVIDENCE / "checks.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
