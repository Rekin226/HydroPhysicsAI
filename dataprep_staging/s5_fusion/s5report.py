"""WP S5: report numbers, ledger numbers and the acceptance table (S5.md / S5.json helpers).

    python -I s5report.py numbers     -> S5_evidence/ledger_numbers.json and report_numbers.json

The prose of the report is written by hand from report_numbers.json; this module only collects
the quoted numbers so that each one has a ledger entry (s5ledger.py) and a single source.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import s5const as C  # noqa: E402


def j(name: str):
    p = C.EVIDENCE / name
    return json.loads(p.read_text()) if p.exists() else None


def product_dir() -> Path | None:
    base = C.DATAPREP / C.PRODUCT
    vs = sorted((p for p in base.glob("v*") if (p / "MANIFEST.json").exists()),
                key=lambda p: int(p.name[1:])) if base.exists() else []
    return vs[-1] if vs else None


def collect() -> dict:
    hp, h1, t1, t2, t3, r2, dg = (j(n) for n in ("hyper_pre.json", "hyper_t1.json", "t1.json",
                                                  "t2.json", "t3.json", "r2.json",
                                                  "diagnostics.json"))
    d = product_dir()
    val = json.loads((d / "validation.json").read_text()) if d else None
    sens = json.loads((d / "sensitivity.json").read_text()) if d else None
    man = json.loads((d / "MANIFEST.json").read_text()) if d else None
    out = {"product_version": f"{C.PRODUCT}/{d.name}" if d else None,
           "hyper_pre": hp["hyper"], "f": hp["fdict"],
           "hyper_t1": h1["hyper"] if h1 else None,
           "screening_pre": {k: len(v) for k, v in hp["flagged"].items()},
           "T1": t1, "T2": t2, "T3": t3, "R2": r2, "diag": dg,
           "vr_mask_fraction": val["vr_mask_fraction"] if val else None,
           "vr_test_quantiles": val["vr_test_quantiles"] if val else None,
           "sensitivity": {k: v for k, v in (sens or {}).items() if k != "f"},
           "manifest_gate": man["gate"] if man else None,
           "manifest_fit_window_end": man["fit_window_end"] if man else None}
    return out


NUMBERS = {
    "S5.hyper.pre": "pre-origin hyperparameters (procedure H) and depth fractions f",
    "S5.hyper.t1": "T1 stage-A hyperparameters on the training leveling (<= 2019-12)",
    "S5.T1.gate": "T1: scored targets (VR >= 0.5), RMSE fused vs continue-rate baseline, "
                  "coverage3",
    "S5.T1.all": "T1 on all targets (report only)",
    "S5.T2.gate": "T2: coverage3 and RMSE fused vs IDW over 10 km blocks",
    "S5.T3.gnss": "T3 GNSS left out (report only)",
    "S5.T3.cmp": "T3 rings and piles left out (report only)",
    "S5.T3.insar": "T3 InSAR left out (report only)",
    "S5.R2.gnss_wra": "R2 WRA GNSS post-2022 hold-out (after the fact)",
    "S5.R2.gnss_ngl": "R2 NGL GNSS post-2022 hold-out (after the fact)",
    "S5.R2.insar": "R2 InSAR post-2022 hold-out (after the fact)",
    "S5.R2.piles": "R2 piles post-2022 hold-out (after the fact)",
    "S5.R2.rings": "R2 rings post-2022 hold-out (after the fact)",
    "S5.product.vr_mask": "VR >= 0.5 mask fractions of the product (full data)",
    "S5.product.sensitivity": "product without rings and piles (2022-2025 increments)",
    "S5.diag.t1_breakdown": "T1 breakdown by county and year (after the fact)",
    "S5.diag.t2_breakdown": "T2 breakdown by county and interval, rms z (after the fact)",
    "S5.diag.f_tradeoff": "posterior correlations of f with the field (plan risk)",
}


def main(argv=None) -> int:
    out = collect()
    F_ = __import__("s5fit")
    F_.dump(out, C.EVIDENCE / "report_numbers.json")
    F_.dump({"product_version": out["product_version"], "numbers": NUMBERS},
            C.EVIDENCE / "ledger_numbers.json")
    print(json.dumps({"product_version": out["product_version"], "numbers": len(NUMBERS)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
