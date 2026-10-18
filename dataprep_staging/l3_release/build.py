"""Build the L3 model-ready release into an output root.

    PYTHONPATH=<repo>/dataprep_staging python -m l3_release.build --out-root DIR \
        [--created YYYY-MM-DD]

writes ``DIR/release_20261012/`` (refuses an existing one):

* ``grid/v1/``                       the twin grid (geometry, centroids, hashes);
* ``preorigin/<component>/v1/``      pre-origin variant (U3: calibration, initial conditions,
                                     priors, design, frozen forcing of models scored on 2023-26);
* ``full/<component>/v1/``           full variant (U4: recorded-forcing arm, statistical-layer
                                     history, validators, after-the-fact analysis);
* ``PRODUCT_CARD.md`` and ``RELEASE.json``.

Nothing in the content depends on DIR, the wall clock (``--created`` fixes the date) or the
machine's thread count, so two builds from scratch are byte-identical (plan 2.8 item 5);
``l3_release.verify`` compares them.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from common import flags as F
from common import hashing, ledger, manifest
from common import uses as U
from common.paths import ORIGIN

from l3_release import components as K
from l3_release import config as C
from l3_release import io_det
from l3_release import ledger_rows as LR

VARIANT_PURPOSE = {
    "preorigin": ("Pre-origin variant (U3): every component has fit_window_end <= 2022-12, or a "
                  "non-exempt window <= 2022-12 with its listed exemptions, and every dated value "
                  "ends by 2022-12. For calibration targets, initial conditions, priors, model "
                  "design and frozen forcing of anything scored on 2023-26."),
    "full": ("Full variant (U4): 2007/2012-2026 data for the recorded-forcing diagnostic arm, "
             "the statistical layer's history, validators and after-the-fact analysis. No "
             "pre-origin use is granted."),
}


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _component_flags(rec: dict) -> dict:
    tot: dict[str, int] = {}
    for o in rec["validation"]["outputs"]:
        s = o["checks"]["summary"]
        cands = []
        if s.get("kind") == "grid":
            tri = s.get("triples", {})
            if "value" in tri:
                cands.append(tri["value"]["flags"])
            elif tri:
                cands.append(next(iter(tri.values()))["flags"])
        elif "flags" in s:
            cands.append(s["flags"])
        for fc in cands:
            for k, v in fc.items():
                tot[k] = tot.get(k, 0) + int(v)
    return dict(sorted(tot.items(), key=lambda kv: int(kv[0])))


def _coverage(rec: dict) -> str:
    parts = []
    for o in rec["validation"]["outputs"]:
        s = o["checks"]["summary"]
        if s.get("dates") and s.get("kind") in ("grid", "members", "points"):
            d = s["dates"]
            parts.append(f"{d[0]}..{d[1]} ({d[2]} steps)")
            break
    for o in rec["validation"]["outputs"]:
        s = o["checks"]["summary"]
        if s.get("kind") == "table" and "rows" in s:
            ids = [k for k in s if k.startswith("n_")]
            extra = f", {s[ids[0]]} {ids[0][2:]}" if ids else ""
            rng = f", {s['range'][0]}..{s['range'][1]}" if s.get("range") else ""
            off = (f", {s['stations_with_rows_off_grid']} stations with rows off the grid "
                   f"(cell_index -1)" if "stations_with_rows_off_grid" in s else "")
            parts.append(f"{o['path']}: {s['rows']} rows{extra}{rng}{off}")
        elif s.get("kind") == "points":
            parts.append(f"{s['n_points']} points")
    return "; ".join(parts)


def card_text(created: str, ginfo: dict, recs: list[dict], grid_rec: dict) -> str:
    L = []
    n_up = len({r["spec"]["upstream"] for r in recs})
    n_ex = sum(max(1, len(x["versions"])) for x in C.EXCLUDED)
    L += [f"# Model-ready release `{C.RELEASE}` (L3)", "",
          f"First L3 release of the October 2026 data-remediation plan (plan 2.1 L3, 2.3, 2.8), "
          f"built {created}: {len(recs)} components (pre-origin "
          f"{sum(r['spec']['variant'] == 'preorigin' for r in recs)}, full "
          f"{sum(r['spec']['variant'] == 'full' for r in recs)}) from {n_up} product versions "
          f"whose gates passed; {n_ex} product versions (or WP-level items) excluded, each "
          f"with its reason in `RELEASE.json` (`excluded`).", "",
          f"Grid: the twin's 1 km grid, {ginfo['nx']} x {ginfo['ny']} cells, dx "
          f"{ginfo['dx']:.0f} m, {ginfo['n_active']} active cells, {ginfo['crs']}; mask hash "
          f"`{ginfo['mask_hash']}`, grid hash `{ginfo['grid_hash']}`. Cell order: "
          f"{ginfo['cell_order']}. `grid/v1/grid.npz` holds the mask, centroids and "
          f"row/column of every active cell.", "",
          "## How to use", "",
          "- Every component is an L3 product version with a `MANIFEST.json`. Open it with "
          "`common.guard.open_product(<component>/v1, <use>)`; the guard grants only the "
          "listed uses (pre-origin uses through the listed exemptions) and filters table rows "
          "by the use's flags (U1, U2).",
          f"- `preorigin/`: {VARIANT_PURPOSE['preorigin']}",
          f"- `full/`: {VARIANT_PURPOSE['full']}",
          "- Grid arrays are (C, A, T), (K, C, A, T) or (A, T) with `value`, `sd`, `flag` "
          "(uint8), `dates` (YYYY-MM) and `mask_hash`/`grid_hash`; A is the second-to-last axis. "
          "Arrays carry every flag: apply the per-use flags yourself (forcing: 0-5).",
          "- Byte-copied files are identical to the upstream files (same sha256); re-written "
          "files say what changed in their component's `validation.json`.", "",
          "## Components", ""]
    for variant in ("preorigin", "full"):
        L += [f"### {variant}", "",
              "| component | content | upstream (WP, status) | coverage | flags in use | "
              "allowed uses |", "|---|---|---|---|---|---|"]
        for r in [x for x in recs if x["spec"]["variant"] == variant]:
            sp = r["spec"]
            fl = ", ".join(_component_flags(r))
            L.append(f"| `{sp['name']}` | {sp['title']} | `{sp['upstream']}` ({sp['wp']}, "
                     f"{C.WP_STATUS[sp['wp']]}) | {_coverage(r)} | {fl} | "
                     f"{', '.join(sp['allowed_uses'])} |")
        L.append("")
    used = set()
    for r in recs:
        used |= {int(k) for k in _component_flags(r)}
    L += ["## Flags in use", "", "| flag | meaning (plan 2.4) |", "|---|---|"]
    for f in sorted(used):
        L.append(f"| {f} | {F.FLAG_MEANING[F.Flag(f)]} |")
    L += ["", "Product-specific readings: electricity flag 3 = Chow-Lin split, 4 = December-2022 "
          "fill or synchronized-zero fill, 9 = aggregate with missing readings above 10 % of "
          "installed HP (and M0 July 2025, partial 2026-07); surface water 5 = Denton/state-space "
          "inferred, 6 = 2026 rain-driven extension; heads 1 = barometric loading removed, "
          "2 = daily bridge, 3 = step corrected with neighbour corroboration only; ET0 1 = "
          "level-matched from 2023-01; IC field 3 = GP field from other sites.", "",
          "## Known limits", ""]
    L += [f"- {x}" for x in C.KNOWN_LIMITS_GENERAL]
    L += ["", "Per component (full notes and sd labels: `RELEASE.json` and each component's "
          "`validation.json`):", ""]
    for r in recs:
        sp = r["spec"]
        L.append(f"- `{sp['variant']}/{sp['name']}`: "
                 f"{C.CARD_LIMITS[(sp['variant'], sp['name'])]}")
    L += ["", "## Excluded", ""]
    cats: dict[str, list[str]] = {}
    for x in C.EXCLUDED:
        lab = x["product"] + (f" {','.join(x['versions'])}" if x["versions"] else "")
        cats.setdefault(x["category"], []).append(lab)
    for cat in ("gate_failed", "acceptance_failed", "not_model_data", "privacy", "phase1",
                "superseded", "not_a_product"):
        if cat in cats:
            L.append(f"- {cat}: " + "; ".join(cats[cat]))
    L += ["", "Rebuild check (plan 2.8 item 5): `REBUILD_CHECK.json`, written after two builds "
          "from scratch; it is not part of the hashed release content.", ""]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-root", required=True, help="parent directory of the release")
    ap.add_argument("--created", default=K.today(), help="YYYY-MM-DD (fixed for both builds)")
    a = ap.parse_args(argv)
    t0 = time.time()
    out_root = Path(a.out_root).resolve()
    release_dir = out_root / C.RELEASE
    if release_dir.exists():
        print(f"{release_dir} exists; refusing to build over it", file=sys.stderr)
        return 2
    release_dir.mkdir(parents=True)
    g, ginfo = K.load_grid()
    grid_rec = K.build_grid_component(release_dir, g, ginfo, a.created)
    print(f"grid: {ginfo['n_active']} active cells, passed {grid_rec['validation']['passed']}")
    recs = []
    for spec in C.COMPONENTS:
        t1 = time.time()
        rec = K.build_component(spec, release_dir / spec["variant"], ginfo, g, a.created)
        recs.append(rec)
        v = rec["validation"]
        print(f"{spec['variant']}/{spec['name']}: passed {v['passed']} "
              f"({len(v['checks_failed'])} failed, {len(v['accepted_exceptions'])} accepted) "
              f"{time.time() - t1:.1f}s", flush=True)
        for e in v["checks_failed"]:
            print("    FAIL", e)
    # ---- guard probes and manifest verification (plan 2.8 item 1)
    probes, probe_bad = {}, {}
    item1 = {}
    all_dirs = [("grid", grid_rec["dir"], C.GRID_USES)] + [
        (f"{r['spec']['variant']}/{r['spec']['name']}", r["dir"], r["spec"]["allowed_uses"])
        for r in recs]
    for key, d, allowed in all_dirs:
        m, _ = manifest.load_manifest(d)
        schema = manifest.validate_manifest(m, d)
        hashes = manifest.verify_hashes(m, d)
        item1[key] = {"schema_errors": schema, "hash_problems": hashes,
                      "ok": not schema and not hashes}
        res, bad = K.probe_guard(d, allowed)
        probes[key], probe_bad[key] = res, bad
    # ---- ledger rows (plan 2.8 item 3)
    number_ids = LR.release_number_ids()
    unledgered = ledger.unledgered(number_ids)
    # ---- card (item 4), then RELEASE.json
    card = card_text(a.created, ginfo, recs, grid_rec)
    io_det.write_text(release_dir / "PRODUCT_CARD.md", card)
    comps = []
    for r in recs:
        sp, m = r["spec"], r["manifest"]
        d = r["dir"]
        comps.append({
            "name": sp["name"], "variant": sp["variant"], "title": sp["title"],
            "path": _rel(d, release_dir), "product": m["product"], "version": m["version"],
            "layer": m["layer"],
            "manifest_sha256": hashing.sha256_file(d / manifest.MANIFEST_NAME),
            "fit_window_end": m["fit_window_end"],
            "non_exempt_fit_window_end": m.get("non_exempt_fit_window_end"),
            "exemptions": [{k: x.get(k) for k in ("input", "data_end", "uses",
                                                   "ledger_number_id", "inherited_from")}
                           for x in m["exemptions"]],
            "allowed_uses": m["allowed_uses"], "forbidden_uses": m["forbidden_uses"],
            "upstream": r["upstream"],
            "outputs": [{"path": f"{_rel(d, release_dir)}/{o['path']}", "sha256": o["sha256"],
                         "bytes": o["bytes"], "mode": o["mode"], "kind": o["kind"],
                         "upstream_file": o["upstream_file"],
                         "upstream_sha256": o["upstream_sha256"],
                         "identical_to_upstream": o.get("identical_to_upstream"),
                         "summary": o["checks"]["summary"]}
                        for o in r["validation"]["outputs"]],
            "validation_passed": r["validation"]["passed"],
            "accepted_exceptions": r["validation"]["accepted_exceptions"],
            "guard_probe": probes[f"{sp['variant']}/{sp['name']}"],
            "sd_status": sp["sd_status"], "notes": sp["notes"],
            "personal_data": sp["personal_data"],
            "ledger_number_id": LR.component_number_id(sp["variant"], sp["name"]),
        })
    upstream_products = sorted({r["upstream"]["dir"]: r["upstream"] for r in recs}.items())
    item2 = {u["dir"]: {"gate_passed": bool(u["gate"]["passed"]), "wp": u["wp"],
                        "wp_final_status": u["wp_final_status"],
                        "inclusion_basis": u["inclusion_basis"]}
             for _, u in upstream_products}
    comp_ok = {f"{r['spec']['variant']}/{r['spec']['name']}": r["validation"]["passed"]
               for r in recs}
    gate = {
        "item1_every_manifest_validates": {
            "components": item1,
            "upstream": {u["dir"]: u["assert_releasable"] for _, u in upstream_products},
            "passed": all(v["ok"] for v in item1.values())},
        "item2_validation_meets_acceptance": {
            "upstream": item2, "components": comp_ok,
            "passed": all(v["gate_passed"] for v in item2.values())
            and all(comp_ok.values()) and grid_rec["validation"]["passed"]},
        "item3_ledger_rows": {"fragment": f"results/dataprep/ledger/rows/{C.LEDGER_WP}.csv",
                              "number_ids": number_ids, "unledgered": unledgered,
                              "passed": not unledgered},
        "item4_product_card": {"file": "PRODUCT_CARD.md",
                               "sha256": hashing.sha256_file(release_dir / "PRODUCT_CARD.md"),
                               "passed": True},
        "item5_identical_rebuild": ("REBUILD_CHECK.json, written after two builds from scratch "
                                    "(not part of the hashed release content)"),
        "guard_probe": {"mismatches": {k: v for k, v in probe_bad.items() if v},
                        "passed": not any(probe_bad.values())},
    }
    gate["passed_items_1_to_4"] = all(gate[k]["passed"] for k in (
        "item1_every_manifest_validates", "item2_validation_meets_acceptance",
        "item3_ledger_rows", "item4_product_card")) and gate["guard_probe"]["passed"]
    rel = {
        "schema": "dataprep-release/1", "release": C.RELEASE, "created": a.created,
        "layer": "L3", "origin": ORIGIN,
        "plan": "docs/DATA_REMEDIATION_PLAN_2026-10.md sections 2.1 (L3), 2.3, 2.8",
        "inclusion_rule": C.__doc__.strip(),
        "grid": {**ginfo, "component": "grid/v1",
                 "manifest_sha256": hashing.sha256_file(grid_rec["dir"] / "MANIFEST.json"),
                 "polygon_crosscheck": grid_rec["validation"]["polygon_crosscheck"],
                 "guard_probe": probes["grid"]},
        "variants": {v: {"purpose": VARIANT_PURPOSE[v],
                         "components": [c for c in comps if c["variant"] == v]}
                     for v in ("preorigin", "full")},
        "upstream_products": [u for _, u in upstream_products],
        "excluded": C.EXCLUDED,
        "known_limits": C.KNOWN_LIMITS_GENERAL,
        "release_gate": gate,
        "ledger": {"fragment": f"results/dataprep/ledger/rows/{C.LEDGER_WP}.csv",
                   "assembled": "results/dataprep/ledger/data_roles.csv",
                   "number_ids": number_ids},
        "code": [{"path": p, "sha256": hashing.sha256_file(K.REPO / p)} for p in K.CODE_FILES],
        "uses_vocabulary": sorted(U.USES),
        "personal_data": ("none: cell-level arrays, county aggregates and public "
                          "monitoring-station ids, names and coordinates; no per-pump or "
                          "per-meter table, no census name or address column, no contact "
                          "field"),
        "chooses_no_member": ("every arm, member and variant of the included products is "
                              "shipped; the release makes no choice among them (U7)"),
    }
    io_det.write_json(release_dir / "RELEASE.json", rel)
    n_files = sum(1 for p in release_dir.rglob("*") if p.is_file())
    print(f"release {release_dir.name}: {len(recs)} components + grid, {n_files} files; "
          f"items 1-4 passed {gate['passed_items_1_to_4']}; unledgered {len(unledgered)}; "
          f"{time.time() - t0:.0f}s")
    return 0 if gate["passed_items_1_to_4"] else 1


if __name__ == "__main__":
    np.seterr(all="ignore")
    raise SystemExit(main())
