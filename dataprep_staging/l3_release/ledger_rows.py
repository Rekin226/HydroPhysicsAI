"""Ledger rows of the L3 release (plan 4.1, rule U6; plan 2.8 item 3).

    PYTHONPATH=<repo>/dataprep_staging python -m l3_release.ledger_rows append
    PYTHONPATH=<repo>/dataprep_staging python -m l3_release.ledger_rows assemble

``append`` writes the release's rows to ``results/dataprep/ledger/rows/L3.csv`` (one row per
source per number, exact duplicates skipped, conflicts refused); ``assemble`` rebuilds
``ledger/data_roles.csv`` from every WP fragment (refuses cross-fragment conflicts).
The rows describe which upstream products feed each release number; they do not depend on
the build's result, so they are written before the builds and the build checks them.
"""
from __future__ import annotations

import argparse

from common import ledger
from common.paths import REPO

from l3_release import config as C

PV = f"model_ready/{C.RELEASE}"


def component_number_id(variant: str, name: str) -> str:
    return f"L3.{C.RELEASE}.{variant}.{name}"


def release_number_ids() -> list[str]:
    ids = [f"L3.{C.RELEASE}.grid"]
    ids += [component_number_id(s["variant"], s["name"]) for s in C.COMPONENTS]
    ids += [f"L3.{C.RELEASE}.excluded", f"L3.{C.RELEASE}.release_gate",
            f"L3.{C.RELEASE}.guard_probe", f"L3.{C.RELEASE}.rebuild_identical"]
    return ids


def _excluded_sources() -> list[tuple[str, str]]:
    out = []
    for x in C.EXCLUDED:
        if x["product"] == "(WP P3, no product)":
            out.append(("results/dataprep/reports/P3.json", x["reason"]))
        elif x["product"] == "_pilots/*":
            out.append(("results/dataprep/_pilots", x["reason"]))
        elif "v9999" in x["versions"]:
            out.append(("results/dataprep/reports/S5_evidence/repair/dryrun_product_c/v9999",
                        x["reason"]))
        else:
            for v in x["versions"]:
                out.append((f"results/dataprep/{x['product']}/{v}", x["reason"]))
    return out


def rows() -> list[dict]:
    r = []
    nid = f"L3.{C.RELEASE}.grid"
    r.append({"number_id": nid, "source": C.GRID_SOURCE, "role": "input",
              "product_version": f"{PV}/grid/v1",
              "note": "geometry keys nx, ny, dx, x0, y0, mask of the twin forward file"})
    r.append({"number_id": nid, "source": C.GRID_POLYGON, "role": "validator",
              "product_version": f"{PV}/grid/v1",
              "note": "cross-check: the twin's build_grid on the fan polygon"})
    upstreams = []
    for s in C.COMPONENTS:
        src = f"results/dataprep/{s['upstream']}"
        upstreams.append(src)
        reads = "; ".join(f"{rd['use']}" + (f" window_end {rd['window_end']}"
                                            if rd["window_end"] else "")
                          for rd in s["reads"])
        r.append({"number_id": component_number_id(s["variant"], s["name"]), "source": src,
                  "role": "input",
                  "product_version": f"{PV}/{s['variant']}/{s['name']}/v1",
                  "note": f"upstream product (WP {s['wp']}, gate passed); guard read: {reads}"})
    for src, reason in _excluded_sources():
        r.append({"number_id": f"L3.{C.RELEASE}.excluded", "source": src, "role": "input",
                  "product_version": PV, "note": f"excluded: {reason[:180]}"})
    for src in sorted(set(upstreams)):
        for num in ("release_gate", "rebuild_identical"):
            r.append({"number_id": f"L3.{C.RELEASE}.{num}", "source": src, "role": "input",
                      "product_version": PV,
                      "note": ("upstream product re-hashed by assert_releasable" if num ==
                               "release_gate" else "upstream product read by both builds")})
    for s in C.COMPONENTS:
        r.append({"number_id": f"L3.{C.RELEASE}.guard_probe",
                  "source": f"{C.RELEASE_DIR}/{C.RELEASE}/{s['variant']}/{s['name']}/v1",
                  "role": "input", "product_version": PV,
                  "note": "release component opened for every use"})
    r.append({"number_id": f"L3.{C.RELEASE}.guard_probe",
              "source": f"{C.RELEASE_DIR}/{C.RELEASE}/grid/v1", "role": "input",
              "product_version": PV, "note": "grid component opened for every use"})
    return r


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["append", "assemble", "show"])
    a = ap.parse_args(argv)
    if a.cmd == "show":
        for x in rows():
            print(x)
        return 0
    if a.cmd == "append":
        missing = [s for s, _ in _excluded_sources() if not (REPO / s).exists()]
        if missing:
            print("excluded sources not found:", missing)
            return 1
        p = ledger.append_rows(C.LEDGER_WP, rows())
        print(f"appended to {p}; unledgered {ledger.unledgered(release_number_ids())}")
        return 0
    out, n = ledger.assemble()
    print(f"wrote {out} ({n} rows); conflicts 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
