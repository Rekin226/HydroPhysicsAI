"""Classify every product manifest under results/dataprep against the release definition.

    PYTHONPATH=<repo>/dataprep_staging python -m l3_release_b.inventory [--json OUT]

Each MANIFEST.json (pilots and the release itself skipped) is 'included' (an upstream of a
component), 'excluded' (listed in config.EXCLUDED with its reason) or 'unclassified' (appeared
after the release was defined, or missed). The task requires every excluded product to be
listed with its reason, so the expected count of unclassified versions is 0.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.paths import DATAPREP, rel_to_repo

from l3_release_b import config as C
from l3_release_b import io_det


def classify() -> dict:
    included = {s["upstream"] for s in C.COMPONENTS}
    excluded = {}
    for x in C.EXCLUDED:
        for v in x["versions"]:
            excluded[f"{x['product']}/{v}"] = x
    rows = []
    for mp in sorted(DATAPREP.rglob("MANIFEST.json")):
        rel = rel_to_repo(mp)
        if "/_pilots/" in rel or "/model_ready/" in rel:
            continue
        try:
            m = json.loads(mp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            rows.append({"manifest": rel, "status": "unreadable"})
            continue
        key = f"{m.get('product')}/{m.get('version')}"
        if key in included:
            st, why = "included", ""
        elif key in excluded:
            st, why = "excluded", excluded[key]["reason"]
        else:
            st, why = "unclassified", ""
        rows.append({"manifest": rel, "product": m.get("product"), "version": m.get("version"),
                     "layer": m.get("layer"), "gate_passed": (m.get("gate") or {}).get("passed"),
                     "status": st, "reason": why})
    n = {s: sum(r["status"] == s for r in rows) for s in
         ("included", "excluded", "unclassified", "unreadable")}
    missing_included = sorted(included - {f"{r.get('product')}/{r.get('version')}"
                                          for r in rows})
    return {"counts": n, "rows": rows, "included_not_found": missing_included}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    res = classify()
    print(res["counts"], "included not found:", res["included_not_found"])
    for r in res["rows"]:
        if r["status"] in ("unclassified", "unreadable"):
            print("  ", r["status"], r["manifest"])
    if a.json:
        Path(a.json).write_text(io_det.dumps(res), encoding="utf-8")
    return 0 if not (res["counts"]["unclassified"] or res["counts"]["unreadable"]
                     or res["included_not_found"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
