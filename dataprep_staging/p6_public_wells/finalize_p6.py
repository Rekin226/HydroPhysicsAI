"""Post-build checks of public_wells/v<N>: loader-guard probes, manifest checker on the
product and on the new L0 source, rebuild determinism, ruff. Writes
``results/dataprep/reports/P6_evidence/finalize.json``.

Usage: python -I finalize_p6.py <product_dir> <scratch_build_a> <scratch_build_b>
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import guard  # noqa: E402
from common.hashing import sha256_file  # noqa: E402
from common.paths import REPO, STAGING, rel_to_repo  # noqa: E402
from common.source import check_source_file  # noqa: E402

from p6_public_wells import p6const as K  # noqa: E402

PROBES = {"forcing_frozen": True, "model_design": True, "prior": True,
          "design_time_evidence": True, "validator": True, "after_the_fact_analysis": True,
          "scoring_target": False, "calibration_target": False, "selection": False,
          "baseline": False, "initial_conditions": False, "forcing_recorded_arm": False,
          "prior_width_input": False}
NO_MANIFEST = {"MANIFEST.json", "validation.json", "PRODUCT_CARD.md"}


def probes(d: Path) -> dict:
    out = {}
    for use, expect in PROBES.items():
        try:
            p = guard.open_product(d, use)
            df = p.read_table("closures.csv")
            got, note = True, f"{len(df)} rows"
        except Exception as e:  # noqa: BLE001 - a refusal is the expected outcome
            got, note = False, f"{type(e).__name__}: {str(e)[:160]}"
        out[use] = {"opened": got, "expected": expect, "as_expected": got == expect,
                    "note": note}
    return out


def determinism(d: Path, a: Path, b: Path) -> dict:
    res = {"files": {}, "identical_a_b": True, "identical_product_a": True}
    for f in sorted(p.name for p in a.iterdir()):
        ha, hb = sha256_file(a / f), sha256_file(b / f)
        hp = sha256_file(d / f) if (d / f).exists() else None
        res["files"][f] = {"a_eq_b": ha == hb, "product_eq_a": hp == ha}
        res["identical_a_b"] &= ha == hb
        res["identical_product_a"] &= hp == ha
    prod = {p.name for p in d.iterdir()} - NO_MANIFEST
    res["product_files_not_in_scratch"] = sorted(prod - {p.name for p in a.iterdir()})
    return res


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else argv
    d, a, b = (Path(x).resolve() for x in args)
    py = sys.executable
    chk = subprocess.run([py, "-m", "common.check", rel_to_repo(d), "--no-sources", "--json",
                          str(K.EVID / "checker_product.json")], cwd=STAGING,
                         capture_output=True, text=True)
    src = {str(p.parent.name): check_source_file(p) for p in K.SOURCE_JSON}
    ruff = subprocess.run([py, "-m", "ruff", "check", "--line-length", "100",
                           str(K.CODE_DIR)], capture_output=True, text=True)
    out = {"guard_probes": probes(d), "checker_returncode": chk.returncode,
           "checker_tail": chk.stdout.strip().splitlines()[-6:],
           "source_json": {k: {kk: v.get(kk) for kk in ("status", "problems", "n_files")}
                           for k, v in src.items()},
           "determinism": determinism(d, a, b),
           "ruff": ruff.stdout.strip().splitlines()[-3:], "ruff_returncode": ruff.returncode,
           "repo": str(REPO)}
    out["guard_all_as_expected"] = all(v["as_expected"] for v in out["guard_probes"].values())
    (K.EVID / "finalize.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("guard_all_as_expected", "checker_returncode",
                                          "checker_tail", "ruff", "source_json")},
                     ensure_ascii=False, indent=1))
    print("determinism", out["determinism"]["identical_a_b"],
          out["determinism"]["identical_product_a"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
