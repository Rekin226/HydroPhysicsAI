"""WP A0 repair round: replay the verifier's guard probes and probe the r2 pilot manifests.

Synthetic probes (temporary directory, no real data) reproduce each loophole the verifier
demonstrated and record how the repaired guard and manifest writer now respond. Real probes
open r2 pilot manifests for the uses downstream WPs would declare. Output: one JSON file.

Usage: python -m common.a0.r2_guard_probe --out <guard_probes.json>
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402

from common import guard, manifest  # noqa: E402
from common.paths import PILOTS  # noqa: E402
from common.uses import USES  # noqa: E402


def _outcome(fn) -> str:
    try:
        r = fn()
    except Exception as exc:  # noqa: BLE001 - the probe records the exception type
        return f"raises {type(exc).__name__}: {str(exc)[:160]}"
    return f"opens ({r})" if r is not None else "opens"


def _synthetic(tmp: Path, name: str, fwe: str, allowed: list[str], **kw) -> tuple[Path, dict]:
    d = manifest.new_version_dir(name, root=tmp)
    pd.DataFrame({"value": [1.0, 2.0], "sd": [float("nan")] * 2, "flag": [0, 0],
                  "method": ["obs", "obs"], "product_version": d.name, "source_ids": "s",
                  "date": ["2022-06", "2024-06"]}).to_csv(d / "t.csv", index=False)
    (d / "validation.json").write_text("{}")
    code = tmp / "build.py"
    code.write_text("pass\n")
    m = manifest.build_manifest(
        d, product=name, layer="L1", inputs=[], code=[code], parameters={},
        fit_window_end=fwe, validation_report="validation.json",
        gate={"passed": True, "criteria": "probe"}, allowed_uses=allowed,
        forbidden_uses=[], **kw)
    return d, m


def synthetic_probes() -> dict:
    out = {}
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        for use in ("datum_fit", "baseline"):
            d, m = _synthetic(tmp, f"heads/L1_probe_{use}", "2025-06", [use, "scoring_target"])
            out[f"write manifest allowing {use} with fit_window_end 2025-06"] = _outcome(
                lambda d=d, m=m: manifest.write_manifest(d, m))
            out[f"check_use({use}) on that manifest, bypassing the writer"] = _outcome(
                lambda m=m, use=use: guard.check_use(m, use))
        d, m = _synthetic(tmp, "heads/L1_probe_ok", "2022-12",
                          ["calibration_target", "initial_conditions"])
        manifest.write_manifest(d, m)
        out["open(calibration_target, preorigin=False)"] = _outcome(
            lambda: guard.open_product(d, "calibration_target", preorigin=False))
        out["open(initial_conditions, require_gate=False) [gate passed]"] = _outcome(
            lambda: guard.open_product(d, "initial_conditions", require_gate=False))
        ex = {"input": "post-origin input", "data_end": "2025-06", "reason": "probe",
              "ledger_number_id": "probe", "sensitivity": "probe", "uses": ["*"]}
        d2, m2 = _synthetic(tmp, "x/full", "2025-06", ["prior", "selection"],
                            exemptions=[ex])
        out["write 'x/full' with exemption uses ['*'], no non_exempt_fit_window_end"] = \
            _outcome(lambda: manifest.write_manifest(d2, m2))
        out["check_use(prior) on that manifest"] = _outcome(lambda: guard.check_use(m2, "prior"))
        out["check_use(selection) on that manifest"] = _outcome(
            lambda: guard.check_use(m2, "selection"))
        d3, m3 = _synthetic(tmp, "x/pre_origin", "2025-06", ["validator"])
        out["write 'x/pre_origin' with fit_window_end 2025-06 and no exemption"] = _outcome(
            lambda: manifest.write_manifest(d3, m3))
        d4, m4 = _synthetic(tmp, "heads/L1_probe_slice", "2026-09",
                            ["datum_fit", "scoring_target"],
                            time_slicing={"column": "date", "params_fit_window_end": "2022-12"})
        manifest.write_manifest(d4, m4)
        out["time-sliced L1: open(datum_fit) without window_end"] = _outcome(
            lambda: guard.open_product(d4, "datum_fit"))
        out["time-sliced L1: open(datum_fit, window_end=2022-12).read_table rows"] = _outcome(
            lambda: guard.open_product(d4, "datum_fit", window_end="2022-12")
            .read_table("t.csv")["date"].tolist())
    return out


def pilot_probes() -> dict:
    out = {}
    tpc = PILOTS / "tpc_aggregate_electricity"
    for use in ("design_time_evidence", "prior", "validator"):
        out[f"tpc_aggregate_electricity: {use}"] = _outcome(
            lambda use=use: guard.open_product(tpc, use).use)
    out["tpc_aggregate_electricity: after_the_fact_analysis"] = _outcome(
        lambda: guard.open_product(tpc, "after_the_fact_analysis").use)
    out["physics_inversion_hooks: design_time_evidence"] = _outcome(
        lambda: guard.open_product(PILOTS / "physics_inversion_hooks",
                                   "design_time_evidence").path("d_pseudo_changhua.json").name)
    gap = guard.open_product(PILOTS / "head_gapfill_pilot", "design_time_evidence")
    out["head_gapfill_pilot: design_time_evidence, readable outputs"] = (
        f"{len(gap.allowed_outputs or [])} outputs")
    out["head_gapfill_pilot: design_time_evidence -> out/pilot_final_clean.csv"] = _outcome(
        lambda: gap.path("out/pilot_final_clean.csv").name)
    out["head_gapfill_pilot: design_time_evidence -> out/partial_month_error.csv"] = _outcome(
        lambda: gap.path("out/partial_month_error.csv").name)
    lu = guard.open_product(PILOTS / "landuse_crop_priors", "design_time_evidence")
    out["landuse_crop_priors: design_time_evidence -> town_kwh.csv"] = _outcome(
        lambda: lu.path("town_kwh.csv").name)
    out["landuse_crop_priors: design_time_evidence -> prior_cells.csv (LUIMAP)"] = _outcome(
        lambda: lu.path("prior_cells.csv").name)
    out["uses checked"] = sorted(USES)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = {"synthetic": synthetic_probes(), "pilots": pilot_probes()}
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n",
                           encoding="utf-8")
    for sec, d in res.items():
        for k, v in d.items():
            if k != "uses checked":
                print(f"[{sec}] {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
