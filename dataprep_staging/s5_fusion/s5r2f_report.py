"""WP S5 repair round 2, fix round: numbers quoted in the Fix round section of
reports/S5.repair.md, checked against fix_round/fix_after.json, and their ledger rows.

    python -I s5r2f_looks.py fix_report && python -I s5r2f_report.py

Writes fix_round/report_numbers.json, fix_round/numbers_check.json and appends ledger rows to
results/dataprep/ledger/rows/S5.csv (fix_round/ledger_result.json). Roles: the spent leveling
target tables are design_time_evidence; the leveling panel and the grid are input.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import s5const as C  # noqa: E402
import s5r2_const as K  # noqa: E402
from common import ledger  # noqa: E402
from common.paths import rel_to_repo  # noqa: E402

REPORT = C.DATAPREP / "reports" / "S5.repair.md"
FIX = K.EVID / "fix_round"
PV = f"{C.PRODUCT}/v5"
MARK = "## Fix round (S5r fix"


def f(x, nd):
    return f"{x:.{nd}f}"


def n(x):
    return f"{int(x):,}"


def pct(x):
    return f"{100 * x:.2f} %"


def collect(fx: dict) -> list[tuple[str, str, str, str]]:
    A, B, R = fx["A_exposure"], fx["B_calibration"], fx["C_product_regime"]
    out = [
        ("S5.r2f.A.r1t2.increments", "fix_exposure", n(A["t2_round1"]["increments"]),
         "round-1 T2 scored increments"),
        ("S5.r2f.A.r1t2.with_2018_19", "fix_exposure",
         n(A["t2_round1"]["increments_with_2018_19_survey"]),
         "round-1 T2 increments with a 2018-19 survey"),
        ("S5.r2f.A.r1t2.benchmarks", "fix_exposure",
         n(A["t2_round1"]["benchmarks_with_2018_19_increment"]),
         "benchmarks of round-1 T2 increments with a 2018-19 survey"),
        ("S5.r2f.A.r2t2.with_2018_19", "fix_exposure",
         n(A["t2_round2"]["increments_with_2018_19_survey"]),
         "round-2 T2/T2b increments with a 2018-19 survey"),
        ("S5.r2f.A.t1f_scored_in_t2", "fix_exposure",
         n(A["t2_round1"]["t1f_scored_benchmarks_among_them"]),
         "scored T1F benchmarks among round-1 T2 2018-19 benchmarks"),
        ("S5.r2f.A.t1f_all_in_t2", "fix_exposure",
         n(A["t2_round1"]["t1f_all_benchmarks_among_them"]),
         "all T1F benchmarks among round-1 T2 2018-19 benchmarks"),
        ("S5.r2f.A.surveys", "fix_exposure", n(A["leveling_surveys"]),
         "QC'd leveling surveys 2012-2021"),
        ("S5.r2f.A.never_scored", "fix_exposure", n(A["surveys_never_in_scored_t1_t2"]),
         "leveling surveys never in a scored T1/T2 increment"),
        ("S5.r2f.A.never_scored_bm", "fix_exposure", n(A["benchmarks_with_never_scored_survey"]),
         "benchmarks with a never-scored survey"),
    ]
    for key in ("t2", "t2b"):
        o = B[key]["overall"]
        out += [(f"S5.r2f.B.{key}.coverage", f"fix_{key}", f(o["raw_coverage90"], 3),
                 f"{key} raw 90 % coverage, all targets"),
                (f"S5.r2f.B.{key}.median_abs_z", f"fix_{key}", f(o["median_abs_z"], 3),
                 f"{key} median |z|"),
                (f"S5.r2f.B.{key}.gt3", f"fix_{key}", pct(o["share_abs_z_gt3"]),
                 f"{key} share |z| > 3"),
                (f"S5.r2f.B.{key}.median_dist", f"fix_{key}", f(o["median_dist_km"], 2),
                 f"{key} median distance to nearest training benchmark (km)")]
        if key == "t2b":
            out.append(("S5.r2f.B.t2b.rms_z", "fix_t2b", f(o["rms_z"], 2), "t2b rms z"))
        for r in B[key]["by_distance"]:
            lo, hi = r["bin_km"]
            lab = f"{int(lo)}-{int(hi)}" if hi is not None else f"gt{int(lo)}"
            pre = f"S5.r2f.B.{key}.{lab}"
            out += [(f"{pre}.n", f"fix_{key}", n(r["n"]), f"{key} {lab} km: increments"),
                    (f"{pre}.coverage", f"fix_{key}", f(r["raw_coverage90"], 3),
                     f"{key} {lab} km: raw 90 % coverage"),
                    (f"{pre}.rms_z", f"fix_{key}", f(r["rms_z"], 2), f"{key} {lab} km: rms z"),
                    (f"{pre}.median_abs_z", f"fix_{key}", f(r["median_abs_z"], 3),
                     f"{key} {lab} km: median |z|")]
            if r["share_abs_z_gt3"] > 0:
                out.append((f"{pre}.gt3", f"fix_{key}", pct(r["share_abs_z_gt3"]),
                            f"{key} {lab} km: share |z| > 3"))
            if key == "t2" and lo == 0.0:
                out.append((f"{pre}.median_dist", "fix_t2", f(r["median_dist_km"], 2),
                            "t2 0-2 km: median distance (km)"))
    out += [("S5.r2f.C.cells", "fix_regime", n(R["cells"]), "twin-grid cells"),
            ("S5.r2f.C.median_dist", "fix_regime", f(R["distance_quantiles_km"]["0.5"], 2),
             "median distance of twin-grid cells to the nearest benchmark (km)"),
            ("S5.r2f.C.within2", "fix_regime", f(100 * R["share_of_cells"]["<= 2 km"], 1),
             "percent of cells within 2 km of a benchmark"),
            ("S5.r2f.C.within4", "fix_regime", f(100 * R["share_of_cells"]["<= 4 km"], 1),
             "percent of cells within 4 km of a benchmark")]
    return out


def roles(kind: str) -> dict:
    ev, e1 = K.EVID, C.EVIDENCE
    lev, grid = rel_to_repo(C.LEVELING_PANEL), rel_to_repo(C.TWIN_GRID)
    if kind == "fix_exposure":
        return {"input": [lev, grid], "design_time_evidence": [
            rel_to_repo(e1 / "t2_targets.parquet"), rel_to_repo(ev / "t2_targets.parquet"),
            rel_to_repo(ev / "t2b_targets.parquet"), rel_to_repo(ev / "t1f_targets.parquet"),
            rel_to_repo(e1 / "t1_targets.parquet"), rel_to_repo(ev / "t1_targets.parquet")]}
    if kind in ("fix_t2", "fix_t2b"):
        k = kind[4:]
        return {"input": [lev, grid], "design_time_evidence": [
            rel_to_repo(ev / f"{k}_targets.parquet")]}
    if kind == "fix_regime":
        return {"input": [lev, grid]}
    raise ValueError(kind)


def main() -> int:
    fx = json.loads((FIX / "fix_after.json").read_text())
    text = REPORT.read_text(encoding="utf-8")
    sec = text[text.index(MARK):]
    nums = collect(fx)
    missing = [(i, q) for i, _, q, _ in nums if q not in sec]
    (FIX / "numbers_check.json").write_text(json.dumps(
        {"n": len(nums), "found": len(nums) - len(missing), "missing": missing}, indent=1))
    (FIX / "report_numbers.json").write_text(json.dumps(
        {"product_version": PV, "numbers": {i: {"kind": k, "quoted": q, "note": t}
                                            for i, k, q, t in nums}}, indent=1))
    if missing:
        print(json.dumps(missing, indent=1))
        return 1
    rws = []
    for i, k, _q, t in nums:
        for role, srcs in roles(k).items():
            for s in srcs:
                rws.append({"number_id": i, "source": s, "role": role, "product_version": PV,
                            "note": f"fix round: {t}"})
    p = ledger.append_rows(C.WP, rws)
    allr = ledger.read_all()
    res = {"fragment": rel_to_repo(p), "rows_written": len(rws), "numbers": len(nums),
           "conflicts_all_fragments": len(ledger.conflicts(allr)),
           "rows_all_fragments": len(allr),
           "unledgered": ledger.unledgered([i for i, *_ in nums], allr)}
    (FIX / "ledger_result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
