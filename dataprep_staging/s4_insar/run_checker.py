"""WP S4a acceptance S4a-5 (checker part): run the dataprep checker on every LiCSAR L0 tree.

For each ``data_fetch_api/new_measurements/licsar_*`` directory this runs ``common.check``
with that directory as the sources root (every SOURCE.json in it is re-hashed; files no
SOURCE.json and no L0-registry manifest lists are orphans) and, optionally, on given product
roots (their MANIFEST.json files are re-hashed). Writes
``results/dataprep/insar_tools/s4a_run/checker_licsar.json`` (per-directory reports and a
summary). Read-only.

Usage (hydro env, inside a CPU slot): python -I run_checker.py [product_root ...]
"""
from __future__ import annotations

import contextlib
import datetime as dt
import io
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # dataprep_staging (common)
sys.path.insert(0, str(HERE))

from common import check  # noqa: E402
from common.paths import L0_NEW, rel_to_repo  # noqa: E402
from config import RUN_DIR  # noqa: E402


def run(args: list[str]) -> tuple[int, dict, str]:
    tmp = RUN_DIR / "_checker_tmp.json"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = check.main(args + ["--json", str(tmp)])
    rep = json.loads(tmp.read_text())
    tmp.unlink()
    return rc, rep, buf.getvalue()


def main() -> int:
    products = sys.argv[1:]
    out = {"created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "dirs": {}, "products": None}
    tot = {"sources": 0, "sources_failed": 0, "orphans": 0, "files_hashed": 0}
    for d in sorted(L0_NEW.glob("licsar_*")):
        # roots = the run directory, which holds no MANIFEST.json: L0 records only
        rc, rep, _ = run([str(RUN_DIR), "--sources-root", str(d)])
        s = rep["summary"]
        out["dirs"][rel_to_repo(d)] = {
            "rc": rc, "summary": s, "orphans": rep.get("orphans", []),
            "sources": [{k: v for k, v in r.items() if k in ("path", "status", "n_files",
                                                             "problems")}
                        for r in rep["sources"]]}
        for k in tot:
            tot[k] += s[k]
        print(f"{rel_to_repo(d)}: sources {s['sources']} (failed {s['sources_failed']}), "
              f"orphans {s['orphans']}, files {s['files_hashed']}", flush=True)
    if products:
        rc, rep, _ = run(products + ["--no-sources"])
        out["products"] = {"rc": rc, "summary": rep["summary"], "manifests": rep["manifests"]}
        print("products:", rep["summary"], flush=True)
    out["summary"] = tot
    out["passed_l0"] = tot["sources_failed"] == 0 and tot["orphans"] == 0
    (RUN_DIR / "checker_licsar.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(tot))
    return 0 if out["passed_l0"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
