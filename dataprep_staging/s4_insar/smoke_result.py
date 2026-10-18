"""WP S4a acceptance S4a-3: summarise the LiCSBAS smoke runs written by ``smoke_licsbas.sh``.

Two runs on pre-origin pairs of the primary frame: ``smoke`` (the first 40 fetched pairs,
which mix two unw georeference variants on the one frame index window) and
``smoke_single_variant`` (40 pairs of one unw georeference variant). For each run it reads
``<name>_licsbas.log`` (one ``== rc=0 <step>`` line per step that exited 0; the script runs
under ``set -e``, so a failing step ends the log), ``<name>_workdir.json`` and the work
directory: ``TS_GEOCml2/cum.h5`` (shape of ``cum`` and number of epochs, read with h5py), the
interferograms LiCSBAS11 rejected and the georeference variants of the linked unw files (from
``pairs_<frame>.csv``). No deformation value is read or reported: the check is that the
software runs end to end on the fetched subsets. S4a-3 passes when the single-variant run
passes (the strict reading of "single-grid pairs"); the mixed run is reported alongside.

Writes ``smoke_result.json``. Usage (licsbas env): python -I smoke_result.py
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import RUN_DIR  # noqa: E402

FRAME = "069A_06694_131402"
REQUIRED = ("LiCSBAS02_ml_prep.py", "LiCSBAS11_check_unw.py", "LiCSBAS12_loop_closure.py",
            "LiCSBAS13_sb_inv.py")
MIN_PAIRS = 20
RUNS = ("smoke_single_variant", "smoke")


def summarise(name: str, rows: dict) -> dict:
    log = (RUN_DIR / f"{name}_licsbas.log").read_text(errors="replace")
    ok_steps = re.findall(r"^== rc=0 (\S+)", log, flags=re.M)
    wd = json.loads((RUN_DIR / f"{name}_workdir.json").read_text().splitlines()[-1])
    w = RUN_DIR / name
    pairs = sorted(p.name for p in (w / "GEOC").iterdir() if p.is_dir())
    variants = sorted({rows[p].get("georef_variant") for p in pairs})
    on_ref = all(rows[p].get("on_reference_grid") == "True" for p in pairs)
    cum = w / "TS_GEOCml2" / "cum.h5"
    info: dict = {"exists": cum.is_file()}
    if cum.is_file():
        import h5py
        with h5py.File(cum, "r") as h:
            info.update({"datasets": sorted(h.keys()), "cum_shape": list(h["cum"].shape),
                         "n_imdates": int(len(h["imdates"])) if "imdates" in h else None,
                         "bytes": cum.stat().st_size})
    bad = w / "TS_GEOCml2" / "info" / "11bad_ifg.txt"
    n_bad11 = (len([x for x in bad.read_text().splitlines() if x.strip()])
               if bad.is_file() else None)
    passed = (all(s in ok_steps for s in REQUIRED) and info["exists"]
              and len(pairs) >= MIN_PAIRS and on_ref)
    return {"workdir": name, "n_pairs_linked": len(pairs),
            "first_pair": pairs[0] if pairs else None,
            "last_pair": pairs[-1] if pairs else None, "workdir_record": wd,
            "steps_exit0": ok_steps, "unw_georef_variants": variants,
            "single_variant": len(variants) == 1, "all_pairs_on_frame_index_window": on_ref,
            "n_ifg_rejected_by_step11": n_bad11, "cum_h5": info, "passed": bool(passed)}


def main() -> int:
    with open(RUN_DIR / f"pairs_{FRAME}.csv", newline="") as fh:
        rows = {r["pair"]: r for r in csv.DictReader(fh)}
    runs = {n: summarise(n, rows) for n in RUNS if (RUN_DIR / f"{n}_licsbas.log").exists()}
    strict = runs.get("smoke_single_variant", {})
    out = {"created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "criterion": "S4a-3", "frame": FRAME, "required_steps": list(REQUIRED),
           "min_pairs": MIN_PAIRS, "runs": runs,
           "passed": bool(strict.get("passed") and strict.get("single_variant")),
           "note": "pre-origin pairs only (2020-2022); software test, no values reported"}
    (RUN_DIR / "smoke_result.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({n: {k: r[k] for k in ("n_pairs_linked", "first_pair", "last_pair",
                                            "steps_exit0", "unw_georef_variants", "cum_h5",
                                            "n_ifg_rejected_by_step11", "passed")}
                      for n, r in runs.items()}, indent=1))
    print("S4a-3 passed:", out["passed"])
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
