"""WP A0 repair round: rewrite A0's ledger fragment with explicit sources, add rows for every
number the A0 report quotes, and rebuild data_roles.csv.

The first-round fragment used glob and brace patterns as sources (for example
``results/dataprep/_pilots/*/MANIFEST.json``), which the repaired ledger refuses because a
pattern never collides with the explicit path of the same file. It is kept unchanged as
``ledger/superseded/A0.r1.csv`` (outside ``rows/``, so the assembler ignores it) and the new
``rows/A0.csv`` lists one source per row.

Usage: python -m common.a0.r2_ledger_rows
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import ledger  # noqa: E402
from common.manifest import load_manifest  # noqa: E402
from common.paths import (  # noqa: E402
    DATAPREP,
    L0_NEW,
    LEDGER_DIR,
    LEDGER_ROWS,
    PILOTS,
    rel_to_repo,
)
from common.source import SCHEMA_VERSION  # noqa: E402

P = "results/dataprep/_pilots"
EV = "results/dataprep/reports/A0_evidence"
R1 = "_pilots@2026-10-12 (r1)"
R2 = "_pilots@2026-10-12.r2"
L0V = "L0 new_measurements@2026-10-12"
REG = "l0_registry/v1"
SCOUTS = sorted(p.name for p in PILOTS.iterdir() if p.is_dir())


def _row(n, src, role, pv, note=""):
    return {"number_id": n, "source": src, "role": role, "product_version": pv, "note": note}


def source_files() -> tuple[list[str], list[str]]:
    import json
    new, legacy = [], []
    for p in sorted(L0_NEW.rglob("SOURCE.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        (new if rec.get("schema") == SCHEMA_VERSION else legacy).append(rel_to_repo(p))
    return new, legacy


def rows() -> list[dict]:
    new_src, legacy_src = source_files()
    out = []
    # ---------------------------------------------------------------- first-round numbers
    n = "A0.check.hashes_reproduced"
    out += [_row(n, f"{P}/{s}/MANIFEST.r1.json", "validator", R1,
                 "r1 manifest (was MANIFEST.json at the first-round check)") for s in SCOUTS]
    out += [_row(n, s, "validator", L0V) for s in new_src + legacy_src]
    n = "A0.rerun.tpc.compare_annual.max_rel_diff"
    tpc = "_pilots/tpc_aggregate_electricity@2026-10-12"
    out += [
        _row(n, "data_fetch_api/new_measurements/tpc_d007012/2026-10-12/d007012.csv", "input",
             tpc, "L0 copy of the scout's download"),
        _row(n, "AMP_V2/data/tpc_pumps.parquet", "input", tpc,
             "columns selected explicitly; no 戶名1 or 用電地址2"),
        _row(n, "AMP_V2/data/pump_kwh_all.parquet", "input", tpc),
        _row(n, f"{P}/tpc_aggregate_electricity/compare_annual.csv", "validator", tpc,
             "pilot reference output")]
    n = "A0.rerun.pseudo.d_pseudo_changhua.max_rel_diff"
    pih = "_pilots/physics_inversion_hooks@2026-10-12"
    out += [_row(n, f"{P}/physics_inversion_hooks/{f}", "input", pih, "archived pilot input")
            for f in ("inputs_cwa.npz", "hindcast_cwa.npz", "K_pump_f32.npy", "M_cwa.npz")]
    out.append(_row(n, f"{P}/physics_inversion_hooks/d_pseudo_changhua.json", "validator", pih,
                    "pilot reference output"))
    n = "A0.rerun.gapfill.pilot_final_clean.max_rel_diff"
    gap = "_pilots/head_gapfill_pilot@2026-10-12"
    names = ["monthly_long.parquet", "well_meta.csv"] + [
        f"pilot_{x}{fam}.parquet" for fam in "NSEH" for x in ("", "ext_")]
    out += [_row(n, f"{P}/head_gapfill_pilot/out/{f}", "input", gap, "archived pilot input")
            for f in names]
    out.append(_row(n, f"{P}/audit_2026-10-11/artefact_events.csv", "input",
                    "_pilots/audit_2026-10-11@2026-10-12", "'hist' offset events"))
    out.append(_row(n, f"{P}/head_gapfill_pilot/out/pilot_final_clean.csv", "validator", gap,
                    "pilot reference output"))
    n = "A0.l0.inventory"
    out += [_row(n, s, "input", L0V, "754 files, 278.1 MB over the 29 new records")
            for s in new_src]
    out.append(_row(n, f"{EV}/l0_migration.json", "input", L0V, "pilot-to-L0 map"))
    out.append(_row("A0.l0.not_copied", f"{EV}/l0_migration.json", "input", L0V,
                    "9 files not copied with reasons; 8 empty download dirs"))
    out.append(_row("A0.pilots.inventory", f"{P}/INDEX.r1.json", "input", R1,
                    "15 directories, 1.2 GB, 10 regenerable arrays (0.80 GB)"))
    out.append(_row("A0.library.tests_r1", "dataprep_staging/common/tests/test_library.py",
                    "input", "staging library 0.1.0", "17 of 17 passed in the first round"))
    out.append(_row("A0.ledger.rows_r1", "results/dataprep/ledger/superseded/A0.r1.csv",
                    "input", "ledger fragment r1", "11 rows, 0 conflicts (pattern sources)"))
    out.append(_row("A0.guard.probe_r1", f"{EV}/guard_on_pilots.json", "input", R1))
    # ---------------------------------------------------------------- repair-round numbers
    n = "A0.r2.check.hashes_reproduced"
    out += [_row(n, f"{P}/{s}/MANIFEST.json", "validator", R2) for s in SCOUTS]
    out.append(_row(n, f"results/dataprep/{REG}/MANIFEST.json", "validator", REG,
                    "re-hashes the 1045 registered L0 files"))
    out += [_row(n, s, "validator", L0V) for s in new_src + legacy_src]
    n = "A0.r2.check.l0_orphans"
    out.append(_row(n, rel_to_repo(L0_NEW), "input", L0V, "every file under the L0 root"))
    out.append(_row(n, f"results/dataprep/{REG}/MANIFEST.json", "input", REG))
    n = "A0.r2.pilot_fit_window_ends"
    out.append(_row(n, f"{EV}/repair/window_audit.json", "input", R2,
                    "date ranges of outputs and post-origin year mentions in scripts"))
    out += [_row(n, f"{P}/{s}", "input", R2, "scripts read for the window audit")
            for s in SCOUTS]
    n = "A0.r2.l0_registry.inventory"
    m, _ = load_manifest(DATAPREP / REG)
    dirs = sorted({Path(r["path"]).relative_to(rel_to_repo(L0_NEW)).parts[0]
                   for r in m["inputs"]
                   if len(Path(r["path"]).relative_to(rel_to_repo(L0_NEW)).parts) > 1})
    out += [_row(n, f"{rel_to_repo(L0_NEW)}/{d}", "input", REG) for d in dirs]
    out += [_row(n, r["path"], "input", REG, "root file") for r in m["inputs"]
            if len(Path(r["path"]).relative_to(rel_to_repo(L0_NEW)).parts) == 1]
    n = "A0.r2.rerun.physics_chain.d_pseudo_max_rel_diff"
    pm, _ = load_manifest(PILOTS / "physics_inversion_hooks")
    out += [_row(n, r["path"], "input", "repository data read by the archived a/b scripts",
                 r.get("via", "named by the pilot scripts")) for r in pm["inputs"]]
    out += [_row(n, t["path"], "input", "repository data", t.get("via", ""))
            for t in pm.get("input_trees") or []]
    out += [_row(n, f"{P}/physics_inversion_hooks/{f}", "validator", pih,
                 "archived output compared with the regenerated one")
            for f in ("inputs_cwa.npz", "hindcast_cwa.npz", "M_cwa.npz", "K_pump_f32.npy",
                      "G0_f32.npy", "c_residual_by_county.json", "d_pseudo_changhua.json")]
    for n, extra in (("A0.r2.rho_k.full_reproduces", True), ("A0.r2.rho_k.preorigin_values",
                                                             False)):
        out += [_row(n, f"{P}/head_gapfill_pilot/out/{f}", "input", gap)
                for f in ("daily_clean.parquet", "monthly_long.parquet", "well_meta.csv")]
        if extra:
            out.append(_row(n, f"{P}/head_gapfill_pilot/out/partial_month_error.csv",
                            "validator", gap, "archived rho(k) table"))
    n = "A0.r2.guard_probes"
    out.append(_row(n, "synthetic manifests built by common/a0/r2_guard_probe.py", "input",
                    "staging library 0.2.0", "no real data"))
    out += [_row(n, f"{P}/{s}/MANIFEST.json", "input", R2)
            for s in ("tpc_aggregate_electricity", "physics_inversion_hooks",
                      "head_gapfill_pilot", "landuse_crop_priors")]
    out.append(_row("A0.r2.library.tests", "dataprep_staging/common/tests/test_library.py",
                    "input", "staging library 0.2.0"))
    n = "A0.r2.rerun.gapfill_s3_S"
    out += [_row(n, f"{P}/head_gapfill_pilot/out/{f}", "input", gap, "archived monthly matrix")
            for f in ("monthly_long.parquet", "well_meta.csv")]
    out += [_row(n, f, "input", "repository data", "rain features (lib.rain_features)")
            for f in ("chou-shui-data/data/rf_timeseries.csv", "chou-shui-data/data/rf_stations.csv")]
    out += [_row(n, f"{P}/head_gapfill_pilot/out/{f}", "validator", gap,
                 "archived output compared with the refit")
            for f in ("pilot_S.parquet", "kalman_theta_rain.npy", "kalman_theta_norain.npy")]
    return out


def main() -> int:
    frag = LEDGER_ROWS / "A0.csv"
    sup = LEDGER_DIR / "superseded"
    sup.mkdir(exist_ok=True)
    if frag.exists() and not (sup / "A0.r1.csv").exists():
        os.rename(frag, sup / "A0.r1.csv")
    # a second run (r1 already superseded) only appends rows not yet present: append_rows
    # skips exact duplicates and refuses role conflicts
    rs = rows()
    p = ledger.append_rows("A0", rs)
    out, n = ledger.assemble()
    print(f"fragment {p} ({len(rs)} rows); assembled {out} with {n} rows; conflicts "
          f"{len(ledger.conflicts(ledger.read_all()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
