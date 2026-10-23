"""WP S5 repair round: report numbers and the ledger number list of the repair round.

    python -I s5r_looks.py report_facts && python -I s5r_report.py
        -> S5_evidence/repair/report_numbers_r2.json and ledger_numbers_r2.json

The prose of the repair-round section of S5.md is written by hand from report_numbers_r2.json;
this module only collects the quoted numbers so that each one has a ledger entry
(s5r_ledger.py repair) and a single source.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import s5const as C  # noqa: E402
import s5r_const as R  # noqa: E402


def j(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def product(vn: str) -> Path:
    return C.DATAPREP / C.PRODUCT / vn


def convergence(hp: dict) -> dict:
    out = {}
    for st, v in hp["stages"].items():
        parts = {"first": v["first"], "refit": v["refit"]} if "first" in v else {"single": v}
        for k, x in parts.items():
            tail = x.get("history_tail", [])
            out[f"{st}.{k}"] = {"m2ll_returned": x["m2ll"], "nfev": x["nfev"],
                                "converged": x["converged"], "at_bound": x["at_bound"],
                                "best_in_tail": min((r["m2ll"] for r in tail), default=None)}
    return out


def collect() -> dict:
    ev, er = C.EVIDENCE, R.EVIDENCE_R
    hp = j(ev / "hyper_pre.json")
    out = {"round1": {k: j(ev / f"{k}.json") for k in ("t1", "t2", "t3", "r2")},
           "hyper_pre": hp["hyper"], "convergence_pre": convergence(hp),
           "convergence_t1": {k: {kk: v[kk] for kk in ("m2ll", "nfev", "converged", "at_bound")}
                              for k, v in j(ev / "hyper_t1.json")["stage_A"].items()
                              if isinstance(v, dict)},
           "stage2": j(er / "hyper_stage2.json"), "m1": j(er / "m1.json"),
           "diag_stageA": j(er / "diag_stageA.json"), "diag_t2buffer": j(er / "diag_t2buffer.json"),
           "diag_campaign": j(er / "diag_campaign.json")}
    dry = {}
    for v in ("anomaly", "null"):
        m = j(er / "dryrun_monthly" / v / "repair" / "m1.json")
        h = j(er / "dryrun_monthly" / v / "repair" / "hyper_stage2.json")
        if m:
            dry[v] = {"criteria": m["criteria"], "scored": m["scored"],
                      "all": m["all_station_months"], "hyper": h["hyper"],
                      "lr": h["lr_null_minus_full"]}
    out["dryrun"] = dry
    vs = sorted((p for p in (C.DATAPREP / C.PRODUCT).glob("v*") if (p / "MANIFEST.json").exists()),
                key=lambda p: int(p.name[1:]))
    d = vs[-1]
    out["product_version"] = f"{C.PRODUCT}/{d.name}"
    val = j(d / "validation.json")
    man = j(d / "MANIFEST.json")
    sens = j(d / "sensitivity.json")
    hpv = j(d / "hyperparameters.json")
    steps = pd.read_csv(d / "leveling_steps.csv")
    seg = pd.read_csv(d / "gnss_segments.csv")
    z = np.load(d / "fused_annual.npz", allow_pickle=False)
    zm = np.load(d / "fused_monthly.npz", allow_pickle=False)
    years = z["years"]
    out["v2"] = {
        "manifest": {k: man[k] for k in ("fit_window_end", "gate", "allowed_uses", "layer")},
        "n_inputs": len(man["inputs"]), "n_code": len(man["code"]),
        "n_outputs": len(man["outputs"]), "data_end": man["parameters"]["data_end"],
        "vr_mask_fraction": val["vr_mask_fraction"],
        "monthly_support_fraction": val["monthly_support_fraction"],
        "screening": hpv["screening_full"], "stepped": int(steps.flagged.sum()),
        "scanned": int(len(steps)),
        "steps_flagged": steps[steps.flagged].to_dict("records"),
        "gnss_segments": int(len(seg)), "gnss_stations": int(seg.station.nunique()),
        "gnss_stations_with_breaks": sorted(seg[seg.seg > 0].station.unique().tolist()),
        "sensitivity": {k: v for k, v in sens.items() if k not in ("f", "ring_pile_cells")},
        "median_inc": {int(y): float(np.median(z["inc"][i])) for i, y in enumerate(years)
                       if y >= 2020},
        "median_inc_sd": {int(y): float(np.median(z["inc_sd"][i])) for i, y in enumerate(years)
                          if y >= 2020},
        "change_2023_2026": {"median": float(np.median(z["change_2023_2026"])),
                             "max": float(np.max(z["change_2023_2026"]))},
        "mask_year_2026": float(z["mask_year"][list(years).index(2026)].mean()),
        "monthly": {"support_delivered": float(zm["support"].mean()),
                    "support_vr_rule": float(zm["support_vr"].mean()),
                    "m1_pass": bool(zm["m1_pass"]),
                    "anomaly_sd_median": float(np.nanmedian(zm["anomaly_sd"])),
                    "anomaly_abs_median": float(np.nanmedian(np.abs(zm["anomaly"]))),
                    "months_with_footprint": int((zm["footprint"].sum(1) >= 30).sum())}}
    v1 = product("v1")
    z1 = np.load(v1 / "fused_annual.npz", allow_pickle=False)
    out["v1"] = {"mask_year_2026": float(z1["mask_year"][list(z1["years"]).index(2026)].mean()),
                 "median_inc": {int(y): float(np.median(z1["inc"][i]))
                                for i, y in enumerate(z1["years"]) if y >= 2020}}
    return out


NUMBERS = {
    "S5.M1.stage2": "stage-2 monthly-anomaly hyperparameters (pre-origin, run P residuals of "
                    "the GNSS-out annual fit) and the likelihood ratio against noise only",
    "S5.M1.score": "registered test M1: monthly anomaly vs withheld WRA GNSS months 2020-2022",
    "S5.v2.dryrun_m1": "M1 operating characteristics on synthetic values (design-time)",
    "S5.v2.gnss_segments": "GNSS offset segments of v2 (gaps > 30 d)",
    "S5.v2.steps": "stepped leveling benchmarks of v2",
    "S5.v2.vr_mask": "VR >= 0.5 mask fractions of v2 (2026 excluded)",
    "S5.v2.monthly": "monthly field of v2: support and delivery",
    "S5.v2.sensitivity": "v2 minus the v1 operator, SJES-only breaks, no steps, no rings/piles",
    "S5.v2.field": "v2 annual increments, sd and the 2023-2026 change",
    "S5.v2.compare_v1": "v2 product minus v1 product (SJES neighbourhood)",
    "S5.diag2.stageA": "stage-A optimiser continuation and v2 sensitivity to it",
    "S5.diag2.t2buffer": "T2 with a 10 km buffer (after the fact, design-exposed)",
    "S5.diag2.campaign": "stage A with per-campaign leveling offsets (after the fact)",
}


def main() -> int:
    out = collect()
    F_ = __import__("s5fit")
    F_.dump(out, R.EVIDENCE_R / "report_numbers_r2.json")
    nums = {k: v for k, v in NUMBERS.items()
            if not (k.startswith("S5.diag2.") and out.get("diag_" + k.split(".")[2]) is None)}
    F_.dump({"product_version": out["product_version"], "numbers": nums},
            R.EVIDENCE_R / "ledger_numbers_r2.json")
    print(json.dumps({"product_version": out["product_version"], "numbers": len(nums)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
