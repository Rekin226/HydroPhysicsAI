"""H1a report numbers: collect the key numbers of the run into one JSON for the WP report.

Usage: s9_report.py PRODUCT_DIR WORKDIR OUT_JSON
Reads the product's validation.json and MANIFEST.json, the design-time records and the
stage summaries; writes a flat dictionary of the numbers the report quotes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import REPO  # noqa: E402

EV = REPO / "results" / "dataprep" / "reports" / "H1a_evidence"


def main():
    prod, work, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    V = json.loads((prod / "validation.json").read_text())
    M = json.loads((prod / "MANIFEST.json").read_text())
    dm = json.loads((EV / "design_time" / "baro_methods.json").read_text())["summary"]
    de = json.loads((EV / "design_time" / "baro_epochs.json").read_text())["summary"]
    p0 = json.loads((EV / "p0_profile.json").read_text())["summary"]
    meta = json.loads((work / "meta_summary.json").read_text())
    B = pd.read_csv(prod / "barometric_efficiency.csv", dtype={"sid": str})
    sm = pd.read_parquet(prod / "station_meta_clean.parquet")
    sw = sm[sm.xy_class_raw == "swapped"]
    D = pd.read_parquet(prod / "heads_daily.parquet", columns=["sid", "period"])
    have = set(D.sid)
    k = {
        "criteria": V["criteria"], "gate_passed": V["gate_passed"],
        "injection": V["injection"]["summary"],
        "injection_not_injected": V["injection"]["not_injected"],
        "injection_spurious": V["injection"]["spurious_events"],
        "injection_spurious_per_event": V["injection"]["spurious_per_injected_event"],
        "injection_base_rate": V["injection"]["base_run_share_of_pre_origin_days_flagged_9"],
        "audit": {p: {kk: vv for kk, vv in v.items() if kk != "unmatched_examples"}
                  for p, v in V["validation"]["audit_agreement"].items()},
        "real_events": [{kk: e[kk] for kk in ("sid", "start", "end", "n_days_in_record",
                                              "n_days_flag9", "n_artefact_events",
                                              "n_days_corrected", "flagged",
                                              "artefact_events")}
                        for e in V["validation"]["real_events"]["events"]],
        "o1": V["validation"]["o1_coherence"],
        "pressure_drops": {p: {kk: vv for kk, vv in v.items()
                               if kk != "episodes_on_fall_ge_10hpa"}
                           for p, v in V["validation"]["pressure_drops"].items()},
        "cwa": V["validation"]["cwa_pressure_check"],
        "rescore": V["validation"]["rescore_frozen_challenge"],
        "summary": V["validation"]["summary"],
        "leakprobe": {kk: vv for kk, vv in V["leakage_probe"].items() if kk != "tables"},
        "baro_design_time": {"methods": dm, "epochs": de},
        "baro_rows": {"n_rows": int(len(B)), "wells": int(B.sid.nunique()),
                      "wells_any_accepted": int(B[B.accepted].sid.nunique()),
                      "rows_accepted": int(B.accepted.sum()),
                      "wells_two_epochs": int(B[B.n_epochs == 2].sid.nunique()),
                      "status_counts": B.status.value_counts().to_dict(),
                      "A_K_accepted_median": float(B[B.accepted].A_K.median()),
                      "A_K_accepted_q10_q90": [float(B[B.accepted].A_K.quantile(0.1)),
                                               float(B[B.accepted].A_K.quantile(0.9))],
                      "post_origin_correction_wells": int(B.applies_to_post_origin.sum())},
        "p0": {kk: p0[kk] for kk in ("files", "hist_t0_min", "hist_t1_max", "new_t0_min",
                                     "new_t1_max", "files_with_sentinels", "sentinel_rows",
                                     "m999998_rows", "overlap_2022q4")},
        "n_empty_files": len(p0["empty_files"]),
        "meta": {kk: meta[kk] for kk in ("n_stations", "xy_class_raw", "swapped",
                                         "swapped_in_changhua_yunlin_labels",
                                         "swapped_town_polygon_match_1km",
                                         "swapped_town_match_after_label_repair",
                                         "zero_zero_left_missing", "layer_missing_raw",
                                         "layer_proposed", "ground_height_zero")},
        "swapped_with_data": sorted(set(sw.sid) & have),
        "label_repaired": meta["label_repaired"],
        "wells_with_daily_rows": len(have),
        "manifest": {"product": M["product"], "version": M["version"],
                     "fit_window_end": M["fit_window_end"], "time_slicing": M["time_slicing"],
                     "allowed_uses": M["allowed_uses"], "forbidden_uses": M["forbidden_uses"],
                     "outputs": [{kk: o.get(kk) for kk in ("path", "rows", "bytes",
                                                           "fit_window_end")}
                                 for o in M["outputs"]]},
    }
    out.write_text(json.dumps(k, indent=1, default=str, ensure_ascii=False))
    print(json.dumps({kk: k[kk] for kk in ("gate_passed", "criteria")}, indent=1,
                     default=str, ensure_ascii=False))


if __name__ == "__main__":
    main()
