"""The policy page's N1 labels, v2: v1 plus the model each row was scored on (nothing re-scored).

v1 (``page_labels.py``, written once on 2026-10-23) carried the step C verdicts without
saying which model produced them, so the page presented the Changhua head result, scored on
the census model (R1c flow + three-zone column), as the response of the model it shows, which
carries no Changhua pumping. v2 reads the same scored files (their sha256 must equal the ones
v1 recorded), keeps every v1 value, and adds:

- ``model`` per row: an id, the theta and column paths step C recorded, and
  ``deliverable_recipe`` (the R0 recipe, a masked refit of it, or another model);
- ``design_time``: step C's 2016-18 window is design-time (the masked recipe was designed on
  fits that saw it), never a fresh period;
- ``step_b``: the 2007-10 dose-response verdicts per county, which the page's subsidence
  sentence quotes (v1 hard-coded the Yunlin direction in the page text);
- ``secondary``: step C's declared cross-county secondary of the R0 recipe on Changhua
  heads (the deliverable recipe's own Changhua score);
- ``changhua.status`` derived from step B and the registered fixes instead of a constant.

Written once; refuses to overwrite. Reads only scored result files, never observations.
Usage: DATAPREP_REPO=<repo> python -I page_labels_v2.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

# step C's model paths -> ids. The deliverable recipe is R0 (stage3_datum_gate_cwa) and its
# masked refits; the census model R1c (stage3_datum_gate_ccz*) is another model.
MODEL_IDS = {
    "results/twin_runs/stage3_datum_gate_m1618/stage3_theta.json": ("R0-m1618", True),
    "results/twin_runs/stage3_datum_gate_ccz_m1618/stage3_theta.json": ("R1c-m1618", False),
}
FIX_PASS = "model_better"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def model_of(paths: list[str]) -> dict:
    theta, column = paths[0], paths[1] if len(paths) > 1 else None
    if theta not in MODEL_IDS:
        raise ValueError(f"step C scored an unknown model {theta!r}; add it to MODEL_IDS")
    mid, recipe = MODEL_IDS[theta]
    return {"id": mid, "theta": theta, "column": column, "deliverable_recipe": recipe}


def heads_secondary(t: dict) -> dict:
    cl, zr = t["M_vs_CL"], t["M_vs_ZR"]
    return {"rmse_model_m": cl["rmse_model"], "rmse_cl_m": cl["rmse_base"],
            "rmse_zr_m": zr["rmse_base"], "n_wells": cl["n_wells"],
            "ratio_cl": cl["ratio"], "ci_cl": cl["ci"], "verdict_cl": cl["verdict"],
            "verdict_zr": zr["verdict"], "slope_obs_on_model": t["slope_obs_on_M"]["slope"],
            "slope_ci": t["slope_obs_on_M"]["ci"]}


def changhua_status(dose_verdict: str, fixes: dict) -> str:
    """Supported only when the 2007-10 dose-response is consistent or a registered fix
    beat its reference; anything else (inconsistent, inconclusive, failed gates) is not."""
    fixed = any(str(v).startswith(FIX_PASS) for v in fixes.values())
    return "supported" if dose_verdict == "consistent" or fixed else "not_supported"


def build_v2(v1: dict, step_b: dict, step_c: dict) -> dict:
    """v1 + models, design-time flag, step B and the secondary; every v1 value is kept."""
    if v1.get("version") != "v1":
        raise ValueError("expects the v1 label file")
    tests, b = step_c["tests"], step_b["tests"]
    out = json.loads(json.dumps(v1))
    out["version"] = "v2"
    out["design_time"] = "design-time" in v1["window"]
    for kind, prefix in (("heads", "N1P.C_heads_"), ("subsidence", "N1P.C_lev_")):
        for county in out[kind]:
            out[kind][county]["model"] = model_of(tests[prefix + county]["model"])
    out["step_b"] = {county: {"verdict": t["verdict"],
                              "beta_obs_cm_per_gwh": t["beta_obs_cm_per_GWh"],
                              "beta_model_cm_per_gwh": t["beta_model_cm_per_GWh"],
                              "ratio_ci": t["ratio_ci"],
                              "model": t["model"]["model"]}
                     for county, t in ((k.rsplit("_", 1)[1], t) for k, t in b.items())}
    sec = step_c.get("secondary_cross_county", {})
    if "changhua_scored_with_yunlin_model" in sec:
        row = heads_secondary(sec["changhua_scored_with_yunlin_model"]["heads"])
        row["model"] = out["heads"]["yunlin"]["model"]
        out["secondary"] = {"changhua_heads_deliverable_recipe": row}
    ch = out["changhua"]
    status = changhua_status(ch["dose_verdict"], ch.get("fixes", {}))
    if status != ch["status"]:
        raise ValueError(f"derived Changhua status {status!r} differs from v1's {ch['status']!r}")
    ch["status_rule"] = ("supported only if the 2007-10 dose-response is consistent or a "
                         "registered fix beat its reference")
    ch["levers_in_deliverable"] = False
    return out


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common.paths import REPO, rel_to_repo

    rep = REPO / "results/dataprep/reports/N1_policy"
    v1p, out_p = rep / "page_labels.json", rep / "page_labels_v2.json"
    if out_p.exists():
        raise SystemExit(f"{rel_to_repo(out_p)} exists; written once")
    v1 = json.loads(v1p.read_text(encoding="utf-8"))
    files = {}
    for k in ("step_b", "step_c"):
        p = REPO / v1["inputs"][k]["path"]
        if sha(p) != v1["inputs"][k]["sha256"]:
            raise SystemExit(f"{rel_to_repo(p)} changed since v1 was written")
        files[k] = json.loads(p.read_text(encoding="utf-8"))
    out = build_v2(v1, files["step_b"], files["step_c"])
    out["inputs"]["page_labels_v1"] = {"path": rel_to_repo(v1p), "sha256": sha(v1p)}
    out_p.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
