"""WP S5 repair round 2: collect the numbers quoted in reports/S5.repair.md from the evidence,
check each quoted string appears in the report, and write report_numbers.json (for the ledger)
and numbers_check.json.

    python -I s5r2_report.py <product_version>
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

REPORT = C.DATAPREP / "reports" / "S5.repair.md"


def j(p: Path):
    return json.loads(Path(p).read_text())


def f(x, nd):
    return f"{x:.{nd}f}"


def collect() -> list[tuple[str, str, str, str]]:
    """(number_id, kind, quoted string, note)."""
    ev, dg = K.EVID, K.EVID / "diag"
    out = []
    dc = j(dg / "t1_decomp.json")["subsets"]
    for k, lab in (("lev_gnss_insar", "GNSS + InSAR"), ("all", "all instruments"),
                   ("lev", "leveling only")):
        out.append((f"S5.r2.diag.t1decomp.{k}.rmse", "diag_t1", f(dc[k]["rmse"], 2),
                    f"round-1 T1 refit, {lab}: RMSE (mm)"))
        out.append((f"S5.r2.diag.t1decomp.{k}.rms_z", "diag_t1", f(dc[k]["rms_z"], 2),
                    f"round-1 T1 refit, {lab}: rms z"))
    ps = j(dg / "persistence.json")
    out.append(("S5.r2.diag.persistence.lag1", "diag_lev", f(ps["lag1_corr_total"], 2),
                "lag-1 correlation of annual-increment anomalies, leveling 2012-2017"))
    ac = j(dg / "resid_ac.json")
    out.append(("S5.r2.diag.resid_ac.cmp", "diag_ac", f(ac["cmp"]["median_lag1"], 2),
                "median lag-1 autocorrelation of round-1 T3 ring/pile residuals"))
    out.append(("S5.r2.diag.resid_ac.gnss", "diag_ac", f(ac["gnss"]["median_lag1"], 2),
                "median lag-1 autocorrelation of round-1 T3 GNSS residuals"))
    ir = j(dg / "insar_runP.json")["pairs"]
    pr = next(iter(ir.values()))
    out.append(("S5.r2.diag.insar_runP.corr", "diag_insar", f(pr["corr_deplaned"], 3),
                "run P vs leveling 2020-05..2021-09, de-planed correlation"))
    out.append(("S5.r2.diag.insar_runP.rms", "diag_insar", f(pr["rms_diff_deplaned"], 1),
                "run P vs leveling, de-planed RMS difference (mm)"))
    gv = j(dg / "gnss_v2v4.json")["stations_changed"]
    out.append(("S5.r2.diag.gnss_v2v4.GFES", "diag_gnss", f(gv["wra:GFES"]["mean_mm"], 1),
                "gnss_clean preorigin v4 - v2 at GFES, mean (mm)"))
    for chain, kind in (("t1f", "hyper_t1f"), ("main", "hyper_main"), ("t1", "hyper_t1")):
        hp = j(ev / f"hyper_{chain}.json")["hyper"]
        names = (K.STAGES["A"] if chain == "t1" else
                 K.STAGES["A"] + K.STAGES["B"] + K.STAGES["D"]
                 + (K.STAGES["C"] if chain == "main" else ()))
        for n in names:
            out.append((f"S5.r2.hyper.{chain}.{n}", kind, f"{hp[n]:.3g}",
                        f"H2 chain {chain}: {n}"))
    t1f = j(ev / "t1f.json")
    s = t1f["scored"]
    for key, nd, lab in (("rmse_fused", 2, "RMSE fused"), ("rmse_baseline", 2, "RMSE baseline"),
                         ("ratio_fused_baseline", 3, "ratio"), ("rms_z_fused", 2, "rms z"),
                         ("rmse_levonly", 2, "RMSE leveling-only")):
        out.append((f"S5.r2.T1F.{key}", "t1f", f(s[key], nd), f"T1F (fresh): {lab}"))
    out.append(("S5.r2.T1F.coverage", "t1f", f(s["coverage_fused"]["coverage"], 3),
                "T1F (fresh): coverage"))
    out.append(("S5.r2.T1F.boot_p_ge_1", "t1f", f(s["boot_ratio"]["p_ge_1"], 2),
                "T1F: bootstrap P(ratio >= 1)"))
    out.append(("S5.r2.T1F.n_benchmarks_scored", "t1f", str(t1f["n_benchmarks_scored"]),
                "T1F: benchmarks scored"))
    for key in ("t1",):
        d = j(ev / f"{key}.json")
        if "scored" in d:
            s = d["scored"]
            out.append(("S5.r2.T1SL.ratio", "t1", f(s["ratio_fused_baseline"], 3),
                        "T1 second look: ratio"))
            out.append(("S5.r2.T1SL.coverage", "t1", f(s["coverage_fused"]["coverage"], 3),
                        "T1 second look: coverage"))
            out.append(("S5.r2.T1SL.rmse_fused", "t1", f(s["rmse_fused"], 2),
                        "T1 second look: RMSE fused"))
            out.append(("S5.r2.T1SL.rms_z", "t1", f(s["rms_z_fused"], 2),
                        "T1 second look: rms z"))
    for key in ("t2", "t2b"):
        d = j(ev / f"{key}.json")
        out.append((f"S5.r2.{key.upper()}.ratio", key, f(d["ratio_fused_idw"], 3),
                    f"{key} second look: ratio fused / IDW"))
        out.append((f"S5.r2.{key.upper()}.coverage", key, f(d["coverage_fused"]["coverage"], 3),
                    f"{key} second look: coverage"))
        out.append((f"S5.r2.{key.upper()}.rms_z", key, f(d["rms_z"], 2),
                    f"{key} second look: rms z"))
    t3 = j(ev / "t3.json")["families"]
    out.append(("S5.r2.T3.gnss.ratio", "t3_gnss", f(t3["gnss"]["ratio"], 3), "T3 GNSS out"))
    out.append(("S5.r2.T3.gnss.coverage", "t3_gnss", f(t3["gnss"]["coverage"]["coverage"], 3),
                "T3 GNSS out coverage"))
    out.append(("S5.r2.T3.cmp.ratio", "t3_cmp", f(t3["cmp"]["ratio"], 3), "T3 rings+piles out"))
    out.append(("S5.r2.T3.cmp.coverage", "t3_cmp", f(t3["cmp"]["coverage"]["coverage"], 3),
                "T3 rings+piles out coverage"))
    out.append(("S5.r2.T3.insar.r_fused", "t3_insar", f(t3["insar"]["r_fused"], 3),
                "T3 InSAR out r fused"))
    out.append(("S5.r2.T3.insar.r_levonly", "t3_insar", f(t3["insar"]["r_levonly"], 3),
                "T3 InSAR out r leveling-only"))
    vp = sorted(p for p in (C.DATAPREP / C.PRODUCT).glob("v*") if p.name[1:].isdigit())[-1]
    val = j(vp / "validation.json")
    for y in (2020, 2021, 2022, 2023, 2024, 2025):
        out.append((f"S5.r2.v4.median_inc.{y}", "product",
                    f(val["median_increment_mm_yr"][str(y)], 1),
                    f"v4 fan median annual increment {y} (mm/yr)"))
    out.append(("S5.r2.v4.change_2023_2026.median", "product",
                f(val["change_2023_2026"]["median"], 1), "v4 2023-2026 change median (mm)"))
    return out


def main(argv=None) -> int:
    pv = (argv or sys.argv[1:])[0]
    rows = collect()
    txt = REPORT.read_text(encoding="utf-8")
    check = [{"id": nid, "quote": q, "found": q in txt} for nid, _k, q, _n in rows]
    nums = {"product_version": pv,
            "numbers": {nid: {"kind": k, "value": q, "note": n} for nid, k, q, n in rows}}
    (K.EVID / "report_numbers.json").write_text(json.dumps(nums, indent=1, ensure_ascii=False))
    res = {"n": len(check), "found": sum(c["found"] for c in check),
           "missing": [c for c in check if not c["found"]]}
    (K.EVID / "numbers_check.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
