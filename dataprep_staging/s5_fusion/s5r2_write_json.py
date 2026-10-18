"""WP S5 repair round 2: write results/dataprep/reports/S5.repair.json from the evidence.

    python -I s5r2_write_json.py <product_dir_name e.g. v4>
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
from common.paths import rel_to_repo  # noqa: E402


def j(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def cov(c):
    return f"{c['coverage']:.3f} ({c['lo']:.3f}-{c['hi']:.3f})"


def main(argv=None) -> int:
    vname = (argv or sys.argv[1:])[0]
    ev = K.EVID
    t1f, t1, t2, t2b, t3 = (j(ev / f"{n}.json") for n in ("t1f", "t1", "t2", "t2b", "t3"))
    pdir = C.DATAPREP / C.PRODUCT / vname
    val = j(pdir / "validation.json")
    s1f = t1f["scored"]
    acc = [
        {"criterion": "T1F fresh temporal block (registered S5.repair_r2.v1, gating): evaluable",
         "threshold": ">= 20 benchmarks at VR >= 0.5",
         "measured": f"{t1f['n_benchmarks_scored']} benchmarks, {t1f['n_targets_scored']} "
                     f"targets", "passed": t1f["criteria"]["evaluable"], "units": "benchmarks"},
        {"criterion": "T1F skill (fresh window 2018-19, pseudo-origin 2020-01, no InSAR)",
         "threshold": "RMSE_fused / RMSE_continue-rate < 1",
         "measured": f"{s1f['rmse_fused']:.2f} / {s1f['rmse_baseline']:.2f} = "
                     f"{s1f['ratio_fused_baseline']:.3f} (bootstrap 90 % "
                     f"{s1f['boot_ratio']['p05']:.3f}-{s1f['boot_ratio']['p95']:.3f}, "
                     f"P(>= 1) {s1f['boot_ratio']['p_ge_1']:.2f})",
         "passed": t1f["criteria"]["skill"], "units": "ratio"},
        {"criterion": "T1F coverage (coverage3 gate-3)", "threshold": "interval contains 0.90",
         "measured": f"{cov(s1f['coverage_fused'])}, rms z {s1f['rms_z_fused']:.2f}",
         "passed": t1f["criteria"]["coverage"], "units": "coverage"},
    ]
    if t1 and "scored" in t1:
        s = t1["scored"]
        acc.append({"criterion": "T1 registered design 2020-21, repaired model (SECOND LOOK)",
                    "threshold": "ratio < 1 and coverage3 pass (conditional)",
                    "measured": f"ratio {s['ratio_fused_baseline']:.3f} ({s['rmse_fused']:.2f} "
                                f"vs {s['rmse_baseline']:.2f} mm), coverage "
                                f"{cov(s['coverage_fused'])}, rms z {s['rms_z_fused']:.2f}",
                    "passed": bool(t1["pass"]), "units": "ratio / coverage"})
    acc += [
        {"criterion": "T2 registered design (no buffer), repaired model (SECOND LOOK)",
         "threshold": "coverage3 pass and RMSE_fused / RMSE_IDW < 1",
         "measured": f"coverage {cov(t2['coverage_fused'])} (over-covers), ratio "
                     f"{t2['ratio_fused_idw']:.3f}", "passed": bool(t2["pass"]),
         "units": "coverage / ratio"},
        {"criterion": "T2b with plan 4.3's 10 km buffer (corrected design, SECOND LOOK)",
         "threshold": "coverage3 pass and RMSE_fused / RMSE_IDW < 1 (same thresholds)",
         "measured": f"coverage {cov(t2b['coverage_fused'])}, ratio {t2b['ratio_fused_idw']:.3f}",
         "passed": bool(t2b["pass"]), "units": "coverage / ratio"},
        {"criterion": "T3 leave one family out (report only)", "threshold": "none (report only)",
         "measured": (f"GNSS ratio {t3['families']['gnss']['ratio']:.3f}, coverage "
                      f"{t3['families']['gnss']['coverage']['coverage']:.3f}; rings+piles "
                      f"{t3['families']['cmp']['ratio']:.3f}, "
                      f"{t3['families']['cmp']['coverage']['coverage']:.3f}; InSAR r "
                      f"{t3['families']['insar']['r_fused']:.3f} (leveling-only "
                      f"{t3['families']['insar']['r_levonly']:.3f})"),
         "passed": None, "units": "ratio / coverage / r"},
        {"criterion": f"Product gate subsidence_fused/{vname} (registered: T1F AND T2)",
         "threshold": "both pass", "measured": val["gate"]["status"],
         "passed": val["gate"]["passed"] is not False and val["gate"]["passed"] is not None,
         "units": "boolean"},
    ]
    out = {"wp": "S5r", "product": f"{C.PRODUCT}/{vname}", "report": "S5.repair.md",
           "prereg": K.PREREG_ID, "acceptance": acc, "gate": val["gate"],
           "evidence": rel_to_repo(ev), "hyper_main": j(ev / "hyper_main.json")["hyper"],
           "hyper_t1f": j(ev / "hyper_t1f.json")["hyper"],
           "numbers_check": j(ev / "numbers_check.json"), "ledger": j(ev / "ledger_result.json"),
           "claude_api_tokens": 0}
    (C.DATAPREP / "reports" / "S5.repair.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(acc, indent=1, ensure_ascii=False)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
