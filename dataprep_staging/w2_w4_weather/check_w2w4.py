"""Re-hash this WP's products and L0 records with the A0 checker (common.check functions).

Checks MANIFEST.json of et0/v<N> and rain_check/v1, every SOURCE.json under
data_fetch_api/new_measurements/openmeteo_et0/, and lists L0 files there that no SOURCE.json
or registry covers. Writes a JSON report with an overall "status" ("ok" or "fail").

Usage: python -I isorun.py check_w2w4.py --et0 results/dataprep/et0/v1 --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys

from common.check import check_manifest, find_orphans
from common.manifest import find_manifests
from common.source import SOURCE_NAME, check_source_file
from wcommon import DATAPREP, OM_L0, REPO, write_json_new


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--et0", required=True)
    ap.add_argument("--rain", default="results/dataprep/rain_check/v1")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    mans = [check_manifest(REPO / a.et0 / "MANIFEST.json"),
            check_manifest(REPO / a.rain / "MANIFEST.json")]
    srcs = []
    files = sorted(OM_L0.rglob(SOURCE_NAME))
    for p in files:
        r = check_source_file(p)
        r["path"] = p.relative_to(REPO).as_posix()
        srcs.append(r)
    reg = sorted(find_manifests(DATAPREP / "l0_registry"))
    orph = find_orphans(OM_L0, files, reg)
    ok = (all(m["status"] == "ok" for m in mans) and all(s["status"] == "ok" for s in srcs)
          and not orph)
    rep = {"status": "ok" if ok else "fail", "manifests": mans, "sources": srcs,
           "orphans_openmeteo_et0": orph}
    write_json_new(a.out, rep)
    print(json.dumps({"status": rep["status"],
                      "manifests": [(m["path"], m["status"], m.get("n_hashed")) for m in mans],
                      "sources": [(s["path"], s["status"], s["n_files"]) for s in srcs],
                      "orphans": orph}, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
