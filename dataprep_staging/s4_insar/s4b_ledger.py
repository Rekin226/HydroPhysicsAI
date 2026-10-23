"""WP S4b: ledger rows (plan 4.1, U6) for every number the S4b report quotes.

    python -I s4b_ledger.py      (hydro env) -> results/dataprep/ledger/rows/S4b.csv

One role per source per number: LiCSAR L0 and LiCSBAS run files are inputs; GNSS heights are
validators of the InSAR numbers; the census is the validator of the amplitude test; the units
registry, the column JSON and the design-pilot files that define the battery subsets are
inputs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import s4b_after as AF  # noqa: E402
import s4b_battery as BT  # noqa: E402
import s4b_config as C  # noqa: E402
import s4b_lib as Lb  # noqa: E402
from common import ledger  # noqa: E402
from common.paths import rel_to_repo  # noqa: E402

PV = "insar_069A/v1"
L0_A = "data_fetch_api/new_measurements/licsar_069A_06694/fetch_2026-10-12_s4a"
L0_D = "data_fetch_api/new_measurements/licsar_105D_06628/fetch_2026-10-12_s4a"
S4A = "insar_licsar_subsets/v1"
PREREG = f"results/dataprep/preregistration/{C.PREREG_ID}.json"


def run_files(name: str) -> list[str]:
    ts = C.RUN_DIR / name / "TS_GEOCml2"
    return [rel_to_repo(ts / "cum.h5"), rel_to_repo(ts / "results" / "mask")]


def rows_for(number_id: str, note: str, inputs: list[str], validators: list[str]) -> list[dict]:
    out = [{"number_id": number_id, "source": s, "role": "input", "product_version": PV,
            "note": note} for s in inputs]
    out += [{"number_id": number_id, "source": s, "role": "validator", "product_version": PV,
             "note": note} for s in validators]
    return out


def explore_rows() -> list[dict]:
    """Rows of the unregistered exploratory ramp diagnostic (s4b_explore_ramp.py)."""
    ex = json.loads((C.EVID / "explore_ramp.json").read_text())
    files = Lb.wra_files()
    st = sorted({r["station"] for v in ex.values() for r in v["rows"]})
    base = [L0_A, S4A, rel_to_repo(C.RUN_DIR / f"geom_{C.PRIMARY}.npz")] + run_files("F")
    out = []
    for k, v in ex.items():
        out += rows_for(f"S4b.explore.ramp_{k}", f"exploratory (unregistered): InSAR-GNSS rate "
                        f"slope {v['slope_east_mm_yr_per_km']:.3f} mm/yr/km, mean "
                        f"{v['mean_diff_rate_mm_yr']:.2f} mm/yr", base,
                        [rel_to_repo(files[s]) for s in st])
    return out


def main() -> int:
    if "--explore" in sys.argv:
        rows = explore_rows()
        p = ledger.append_rows(C.WP, rows)
        print(json.dumps({"fragment": str(p), "rows": len(rows)}))
        return 0
    ev = C.EVID
    g1 = json.loads((ev / "test_gnss.json").read_text())
    c1 = json.loads((ev / "test_coherence.json").read_text())
    y = json.loads((ev / "test_seasonal.json").read_text())
    af = json.loads((ev / "after.json").read_text())
    files = Lb.wra_files()
    wra = {st: rel_to_repo(files[st]) for st in files}
    meta = "chou-shui-data/data/ls_cache/ls-wra-gnss-obs__meta.parquet"
    base = [L0_A, S4A, PREREG, rel_to_repo(C.RUN_DIR / f"geom_{C.PRIMARY}.npz")]
    rows = []
    st1 = g1["primary"]["stations"]
    rows += rows_for("S4b.G1.rms_preorigin_runP",
                     f"G1 RMS {g1['primary']['rms_mm']:.2f} mm, {len(st1)} stations (gate)",
                     base + run_files("P") + [meta, rel_to_repo(C.RUN_DIR / "gnss_inventory.json")],
                     [wra[s] for s in st1])
    for k, r in g1["report_only"].items():
        rows += rows_for(f"S4b.G1.report_only.{k}", f"report-only G1 row {k}: RMS "
                         f"{r['rms_mm']:.2f} mm", base + run_files("P") + [meta],
                         [wra[s] for s in r["stations"]])
    for z, r in c1["zones"].items():
        rows += rows_for(f"S4b.C1.{z.split('.')[-1]}",
                         f"C1 {z}: median coherent fraction {r['median_frac_coherent']:.3f}",
                         [L0_A, S4A, PREREG, rel_to_repo(C.RUN_DIR / "coh_P_summer_preorigin.npz"),
                          rel_to_repo(C.UNITS_V3)], [])
    ybase = base + run_files("P") + [rel_to_repo(C.UNITS_V3),
                                     rel_to_repo(C.RUN_DIR / "battery.json"),
                                     "results/twin_runs/stage3_datum_gate/coupled_leveling/"
                                     "vep_zonal_leveling.json", rel_to_repo(C.TWIN_GRID)] + \
        [rel_to_repo(p) for p in BT.INPUTS]
    census = ["results/dataprep/pump_electricity/preorigin/v1"]
    rows += rows_for("S4b.Y.Y1_rho", f"Y1 rho {y['Y1_all_yunlin']['rho_primary']:.3f}, p05 "
                     f"{y['Y1_all_yunlin']['boot_p05']:.3f}", ybase, census)
    rows += rows_for("S4b.Y.Y2_wins", f"Y2 wins {y['Y2_battery']['wins']}/8", ybase, census)
    stF = af["F_g1"]["loso"]["stations"]
    rows += rows_for("S4b.after.F_g1_rms", f"after the fact: run F G1 RMS "
                     f"{af['F_g1']['loso']['rms_mm']:.2f} mm", base + run_files("F") + [meta],
                     [wra[s] for s in stF])
    stp = af["pilot_window_F"]["loso"]["stations"]
    rows += rows_for("S4b.after.pilot_window_rms", f"after the fact: pilot window RMS "
                     f"{af['pilot_window_F']['loso']['rms_mm']:.2f} mm",
                     base + run_files("F") + [meta], [wra[s] for s in stp])
    ngl = []
    for st in AF.NGL_ONFAN:
        p = next((d / f"{st}.tenv3" for d in AF.NGL_DIRS if (d / f"{st}.tenv3").is_file()), None)
        if p is not None:
            ngl.append(rel_to_repo(p))
    for nm in ("P", "F"):
        r = af[f"ngl_onfan_{nm}"]
        rows += rows_for(f"S4b.after.ngl_onfan_{nm}", f"after the fact: NGL on-fan RMS "
                         f"{r['rms_mm']:.2f} mm (n {r['n']})",
                         base + run_files(nm) + [meta, rel_to_repo(AF.NGL_DIRS[0] / "steps.txt")],
                         ngl + [wra[s] for s in (st1 if nm == "P" else stF)])
    rows += rows_for("S4b.after.p_vs_f", f"after the fact: run P vs F RMS "
                     f"{af['p_vs_f']['rms_diff_mm']:.2f} mm", base + run_files("P")
                     + run_files("F"), [])
    rows += rows_for("S4b.after.c1_all_pairs", "after the fact: C1 over every pre-origin pair",
                     [L0_A, S4A, rel_to_repo(C.RUN_DIR / "coh_P_all_preorigin.npz"),
                      rel_to_repo(C.UNITS_V3)], [])
    rows += rows_for("S4b.after.decomp_2025", "after the fact: 2025 asc/desc decomposition",
                     [L0_A, L0_D, S4A] + run_files("F") + run_files("D")
                     + [rel_to_repo(C.RUN_DIR / f"geom_{C.DESC}.npz")], [])
    rows += rows_for("S4b.product.coverage", "cells with >= 5 valid pixels per zone (run F)",
                     [L0_A, S4A, "results/dataprep/insar_069A/v1", rel_to_repo(C.UNITS_V3)],
                     [])
    rows += rows_for("S4b.licsbas.run_stats", "LiCSBAS ifg/epoch counts and mask rates",
                     [rel_to_repo(C.RUN_DIR / n / "TS_GEOCml2" / "info" / f)
                      for n in ("P", "F", "D") for f in ("13parameters.txt", "15parameters.txt")],
                     [])
    p = ledger.append_rows(C.WP, rows)
    nums = sorted({r["number_id"] for r in rows})
    print(json.dumps({"fragment": str(p), "rows": len(rows), "numbers": len(nums),
                      "unledgered": ledger.unledgered(nums)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
