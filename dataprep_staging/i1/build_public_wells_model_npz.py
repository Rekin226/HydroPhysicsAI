"""Public-supply deep-well abstraction (P6 sensitivity member) in the npz layout
``inputs.load_extra_abstraction`` reads, for ladder rung R1e-pw
(``calibrate_flow --extra-abstraction <npz>``).

* ``q_m3d`` (4 twin layers, 2148 active cells, months 2007-01..2022-12) = (``primary`` +
  ``pending_tw_changhua``) / ``wra_to_model``: the closure ramp of the agency wells of annex
  table 2-2 (abstraction until each cohort's stop month, nothing afterwards) plus the 129
  pending Changhua Taiwan Water wells (constant), both as the P6 member defines them, in
  model units (WRA physical / 2.58, the plan's stated assumption). Positive = abstraction.
* Read from the release's pre-origin component (``forcing_frozen`` is an allowed use), never
  from the upstream product, and every exemption the component declares is carried over.

The member was built as a SENSITIVITY member (plan P6 step 3: it never enters the primary
model). This file exists for a diagnostic rung; I1.R1e_pw.v1 records that a pass cannot
promote it without a ruling on that restriction.

Run from the repository root:
    python dataprep_staging/i1/build_public_wells_model_npz.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "dataprep_staging"))

from common import manifest  # noqa: E402

SRC = REPO / "results/dataprep/model_ready/release_20261012/preorigin/public_wells_member/v1"
COMPONENTS = ("rate_phys_m3d__primary", "rate_phys_m3d__pending_tw_changhua")


def main() -> None:
    z = np.load(SRC / "sensitivity_member.npz", allow_pickle=False)
    src_manifest = json.loads((SRC / "MANIFEST.json").read_text())
    ratio = float(z["wra_to_model"])
    phys = sum(np.asarray(z[k], dtype="float64") for k in COMPONENTS)
    q = phys / ratio
    dates = [str(d) for d in z["dates"]]
    if q.shape != (4, int(z["n_active"]), len(dates)):
        raise SystemExit(f"unexpected member shape {q.shape}")
    if not (np.isfinite(q).all() and (q >= 0).all()):
        raise SystemExit("member is not finite and non-negative")
    # the precomputed model-unit primary must agree with this conversion
    chk = float(np.abs(np.asarray(z["rate_model_m3d__primary"], dtype="float64")
                       - np.asarray(z["rate_phys_m3d__primary"], dtype="float64") / ratio).max())
    if chk > 1e-2:
        raise SystemExit(f"rate_model_m3d__primary disagrees with phys/{ratio}: {chk}")
    d = manifest.new_version_dir("public_wells/model_npz")
    out = d / "public_wells_model.npz"
    np.savez(out, q_m3d=q, dates=np.array(dates), dx=np.float64(z["dx"]),
             n_active=np.int64(z["n_active"]), wra_to_model=np.float64(ratio),
             grid_hash=np.array(str(z["grid_hash"])), mask_hash=np.array(str(z["mask_hash"])),
             components=np.array(COMPONENTS),
             units=np.array("m3/day of abstraction per cell and twin layer, model units "
                            "(WRA physical / wra_to_model); positive = abstraction"),
             source=np.array(str(SRC.relative_to(REPO) / "sensitivity_member.npz")))
    yrs = np.array([int(t[:4]) for t in dates])
    annual = {str(y): {"model_Mm3_per_yr": float(q[:, :, yrs == y].sum(axis=(0, 1)).mean()
                                                 * 365.25 / 1e6),
                       "phys_primary_Mm3_per_yr": float(
                           np.asarray(z[COMPONENTS[0]])[:, :, yrs == y].sum(axis=(0, 1)).mean()
                           * 365.25 / 1e6)}
              for y in range(2007, 2023)}
    val = {"passed": True, "shape": list(q.shape), "months": [dates[0], dates[-1]],
           "components": list(COMPONENTS), "wra_to_model": ratio,
           "model_primary_check_max_abs_m3d": chk,
           "layer_share": (q.sum(axis=(1, 2)) / q.sum()).tolist(),
           "annual": annual,
           "after_record": "the loader holds the last month (2022-12) for later months"}
    (d / "validation.json").write_text(json.dumps(val, indent=1))
    exemptions = [{k: v for k, v in e.items() if k != "inherited_from"}
                  | {"inherited_from": "model_ready/release_20261012/preorigin/"
                                       "public_wells_member/v1"}
                  for e in src_manifest.get("exemptions", [])]
    m = manifest.build_manifest(
        d, product="public_wells/model_npz", layer="L3",
        inputs=[str(SRC / "sensitivity_member.npz"), str(SRC / "MANIFEST.json")],
        code=[str(Path(__file__))],
        parameters={"components": list(COMPONENTS), "units": "model (phys / wra_to_model)",
                    "wra_to_model": ratio, "variant": "primary location, timing and layers"},
        fit_window_end=src_manifest.get("fit_window_end"),
        non_exempt_fit_window_end=src_manifest.get("non_exempt_fit_window_end"),
        exemptions=exemptions,
        validation_report="validation.json",
        gate={"passed": True, "criteria": "shape, finiteness, sign and the member's own "
                                          "model-unit primary reproduced; P6 gate passed"},
        allowed_uses=["forcing_frozen", "model_design", "after_the_fact_analysis"],
        forbidden_uses=["scoring_target", "baseline", "datum_fit", "selection"])
    manifest.write_manifest(d, m)
    print(json.dumps({k: val[k] for k in ("shape", "months", "layer_share")}, indent=1),
          "\nwritten", out)


if __name__ == "__main__":
    main()
