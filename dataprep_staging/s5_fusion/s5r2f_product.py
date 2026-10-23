"""WP S5 repair round 2, fix round: subsidence_fused/v<N> (v5) = v4's values with corrected labels.

    python -I s5r2f_looks.py fix_report && python -I s5r2f_product.py

No fit. v4's value files are copied byte for byte (fused_annual.npz, gnss_segments.csv,
leveling_steps.csv, hyperparameters.json) and their sha256 is checked against v4's MANIFEST.
fused_cells_annual.parquet is rewritten with product_version = subsidence_fused/v5; every other
column is checked equal to v4's. validation.json and PRODUCT_CARD.md carry the fix-round labels:

- T1F is a test on design-exposed units (the 2018-19 surveys were scored targets of the failed
  round-1 T2, training data of the failed round-1 T1 and inputs of repair round 1's campaign
  diagnostic): its pass is conditional (second look), not fresh (S5.repair.md F.1.1);
- T2's over-coverage is read as a leveling-era sd that is too wide for most increments and has
  heavier-than-Gaussian tails, not as a test-design artefact (F.1.2, F.3).

Gate and allowed uses are unchanged (FAILED; after_the_fact_analysis only). The v4 directory is
not touched; its build code (s5r2_product.py) is not edited.
"""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import pandas as pd  # noqa: E402
import s5const as C  # noqa: E402
import s5product as P1  # noqa: E402
import s5r2_const as K  # noqa: E402
from common import guard, manifest  # noqa: E402
from common.paths import rel_to_repo  # noqa: E402

SRC = C.DATAPREP / C.PRODUCT / "v4"
FIX = K.EVID / "fix_round"
COPY = ["fused_annual.npz", "gnss_segments.csv", "leveling_steps.csv", "hyperparameters.json"]
CODE = ["s5r2f_product.py", "s5r2f_looks.py", "s5r2f_diag.py", "s5const.py", "s5product.py",
        "s5r2_const.py"]
SHARED = ["dataprep_staging/common/guard.py", "dataprep_staging/common/manifest.py",
          "dataprep_staging/common/hashing.py", "dataprep_staging/common/paths.py",
          "dataprep_staging/common/uses.py", "dataprep_staging/common/flags.py"]

T1F_UNITS = ("design-exposed (second look under rule d): the 2018-19 surveys were scored targets "
             "of the failed round-1 T2 (2,699 of 6,942 increments, 844 benchmarks, every T1F "
             "benchmark), training data of the failed round-1 T1, and inputs of repair round 1's "
             "campaign diagnostic that motivated R2")
T2_READING = ("T2 (no buffer) is the regime closer to the product (product cells: median 0.90 km "
              "to the nearest benchmark; T2 targets 2.34 km; T2b targets 11.12 km). Its "
              "over-coverage (0.948) with median |z| 0.443 (Gaussian 0.674) and 1.05 % of |z| > 3 "
              "(Gaussian 0.27 %) says the leveling-era sd is too wide for most increments and "
              "too narrow for a few (heavy tails). T2b's 0.920 coexists with the same shape "
              "(median |z| 0.522, 1.43 % beyond 3). The earlier reading of T2's failure as a "
              "test-design artefact is withdrawn (S5.repair.md F.1.2, F.3)")


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def jload(p: Path):
    return json.loads(Path(p).read_text())


def gate_v5(v4_gate: dict) -> dict:
    g = {"passed": False, "status": "FAILED",
         "T1F_pass_reading": "conditional (second look)",
         "T1F_second_look_pass": bool(v4_gate.get("T1F_fresh_pass")),
         "T1F_units": T1F_UNITS,
         "T2_second_look_pass": bool(v4_gate.get("T2_second_look_pass")),
         "T2b_second_look_pass": bool(v4_gate.get("T2b_second_look_pass")),
         "T2b_pass_reading": "conditional (second look)",
         "T1_second_look_pass": bool(v4_gate.get("T1_second_look_pass")),
         "fresh_unit_passes": 0,
         "t2_overcoverage_reading": T2_READING}
    return g


def validation_v5(v4val: dict, fix: dict) -> dict:
    v = copy.deepcopy(v4val)
    v["gate"] = gate_v5(v4val["gate"])
    reg = ((v.get("tests") or {}).get("t1f") or {}).get("registration")
    if isinstance(reg, dict):
        reg["status_at_registration"] = reg.get("status")
        reg["status"] = "registered test on design-exposed units: second look (conditional)"
    B = fix["B_calibration"]
    v["fix_round"] = {
        "relabel_of": "subsidence_fused/v4",
        "values": "identical to v4 (byte copies; parquet product_version relabelled only)",
        "report": "results/dataprep/reports/S5.repair.md, Fix round",
        "evidence": rel_to_repo(FIX / "fix_after.json"),
        "second_look_leak": ("T1 (stages B-D), T2 and T2b (all hyperparameters) and the |z| > 5 "
                             "screening use H2 main, fitted on leveling that includes the "
                             "withheld surveys or blocks; inherited from the registered design"),
        "t2_by_distance": [{k: r.get(k) for k in ("bin_km", "n", "raw_coverage90",
                                                   "median_abs_z", "share_abs_z_gt3")}
                           for r in B["t2"]["by_distance"]],
        "t2_overall": B["t2"]["overall"], "t2b_overall": B["t2b"]["overall"],
        "product_regime": fix["C_product_regime"]}
    return v


def card(d: Path, v: dict, v4m: dict) -> str:
    t = v["tests"]

    def g(n, *ks):
        x = t.get(n) or {}
        for k in ks:
            x = x.get(k) if isinstance(x, dict) else None
        return x

    def f3(x):
        return "n/a" if x is None else f"{x:.3f}"

    h = v4m["parameters"]["hyper"]
    B = v["fix_round"]
    lines = [
        f"# {C.PRODUCT}/{d.name} (WP S5, repair round 2, fix round, 2026-10-18)", "",
        "Values identical to subsidence_fused/v4; labels corrected (reports/S5.repair.md, "
        "Fix round).", "",
        "Gate: **FAILED**. Allowed use: after_the_fact_analysis only (upstream GNSS, InSAR and "
        "ring inputs are gate-failed products). No S5 test has passed on fresh units.", "",
        "## Tests (S5.repair_r2.v1; thresholds of S5.subsidence_fusion.v1)", "",
        "| Test | Units | Ratio | Coverage | Pass |", "|---|---|---|---|---|",
        f"| T1F temporal block 2018-19 (pseudo-origin 2020-01) | design-exposed (second look) | "
        f"{f3(g('t1f', 'scored', 'ratio_fused_baseline'))} | "
        f"{f3(g('t1f', 'scored', 'coverage_fused', 'coverage'))} | conditional (second look) |",
        f"| T1 registered design 2020-21 | second look | "
        f"{f3(g('t1', 'scored', 'ratio_fused_baseline'))} | "
        f"{f3(g('t1', 'scored', 'coverage_fused', 'coverage'))} | {g('t1', 'pass')} |",
        f"| T2 registered design (no buffer) | second look | {f3(g('t2', 'ratio_fused_idw'))} | "
        f"{f3(g('t2', 'coverage_fused', 'coverage'))} | {g('t2', 'pass')} (over-covers) |",
        f"| T2b with 10 km buffer (plan 4.3) | second look | {f3(g('t2b', 'ratio_fused_idw'))} | "
        f"{f3(g('t2b', 'coverage_fused', 'coverage'))} | conditional (second look) |", "",
        "T1F's units: " + T1F_UNITS + ".", "",
        "## sd (UNCALIBRATED)", "",
        "- Temporal extrapolation across a leveling gap: under-dispersed (T1 second look "
        "coverage 0.771, rms z 1.69).",
        "- Leveling era, with neighbouring leveling: over-dispersed for most increments, with "
        f"heavy tails (T2 median |z| {B['t2_overall']['median_abs_z']:.3f} against 0.674, "
        f"|z| > 3 for {100 * B['t2_overall']['share_abs_z_gt3']:.2f} % against 0.27 %). "
        "Product cells are nearer to leveling (median "
        f"{B['product_regime']['distance_quantiles_km']['0.5']:.2f} km) than T2's targets "
        f"(median {B['t2_overall']['median_dist_km']:.2f} km).",
        "- T2b (10 km buffer) covers 0.920 at the 90 % level but has the same shape "
        f"(median |z| {B['t2b_overall']['median_abs_z']:.3f}); it does not show a calibrated sd "
        "in the product's regime.", "",
        "## Model", "",
        "As v4: Matern 3/2 field on a 4 km lattice x annual-knot temporal basis with a reverting "
        "prior (long-term velocity, white knot-velocity anomalies, reverting displacement "
        "anomalies; a deviation from the plan's random walk on velocity), seasonal pair; "
        "leveling with campaign offsets; GNSS (gnss_clean/full/v4, gap segments) with correlated "
        "station noise; InSAR (insar_069A/v3) with per-month planes, APS and per-unit random "
        "rates; rings and piles with depth fractions and correlated sensor noise. "
        "Hyperparameters: H2 on pre-origin data (U7).", "",
        "Hyperparameters: " + ", ".join(f"{k} {x:.3g}" for k, x in h.items()), "",
        "## Limits", "",
        "- No monthly field (v3's monthly field was built on the round-1 annual model).",
        "- No fresh-unit validation: every leveling survey 2012-2021 except 28 was a scored "
        "T1/T2 target, and T1F's units are design-exposed.",
        "- T1F has no InSAR (none before 2020): the InSAR operator is checked only by second "
        "looks (T1) and T3 (report only).",
        "- The T1, T2 and T2b second looks use H2-main hyperparameters and screening fitted on "
        "leveling that includes their withheld surveys or blocks.",
        "- Depth fractions f are estimated, then held fixed (f uncertainty not propagated).", ""]
    return "\n".join(lines)


def main() -> int:
    t0 = time.time()
    v4m = jload(SRC / "MANIFEST.json")
    v4h = {o["path"]: o["sha256"] for o in v4m["outputs"]}
    for n in v4h:
        if sha(SRC / n) != v4h[n]:
            raise SystemExit(f"v4 file {n} does not match its manifest")
    fix = jload(FIX / "fix_after.json")
    v4val = jload(SRC / "validation.json")
    d = manifest.new_version_dir(C.PRODUCT)
    for n in COPY:
        shutil.copyfile(SRC / n, d / n)
        if sha(d / n) != v4h[n]:
            raise SystemExit(f"copy of {n} differs from v4")
    a = pd.read_parquet(SRC / "fused_cells_annual.parquet")
    b = a.copy()
    b["product_version"] = f"{C.PRODUCT}/{d.name}"
    b.to_parquet(d / "fused_cells_annual.parquet", index=False)
    chk = pd.read_parquet(d / "fused_cells_annual.parquet")
    other = [c for c in a.columns if c != "product_version"]
    if not chk[other].equals(a[other]):
        raise SystemExit("parquet values differ from v4")
    val = validation_v5(v4val, fix)
    (d / "validation.json").write_text(json.dumps(val, indent=1, default=float))
    (d / "PRODUCT_CARD.md").write_text(card(d, val, v4m), encoding="utf-8")
    inputs = [rel_to_repo(SRC / "MANIFEST.json")] + [rel_to_repo(SRC / n) for n in v4h] + [
        rel_to_repo(FIX / "fix_after.json"),
        rel_to_repo(C.DATAPREP / "preregistration" / f"{K.PREREG_ID}.json")]
    params = copy.deepcopy(v4m["parameters"])
    params["relabel_of"] = f"{C.PRODUCT}/v4"
    mm = manifest.build_manifest(
        d, product=C.PRODUCT, layer="L2", inputs=inputs,
        code=[rel_to_repo(HERE / c) for c in CODE] + SHARED, parameters=params,
        fit_window_end=v4m["fit_window_end"], validation_report="validation.json",
        gate={"passed": False, "status": "FAILED",
              "criteria": (f"{K.PREREG_ID}: T1F (temporal block, 2018-19 leveling withheld at a "
                           "pseudo-origin 2020-01; design-exposed units, so a pass is "
                           "conditional (second look)) AND T2 (registered design, second look); "
                           "T2 failed on over-coverage")},
        allowed_uses=["after_the_fact_analysis"], forbidden_uses=P1.FORBIDDEN,
        personal_data="none (agency benchmark and station coordinates only)",
        extra={"upstream_gate_rule": v4m.get("upstream_gate_rule"),
               "supersedes": ["subsidence_fused/v4 (same values; v4's card and validation call "
                              "T1F fresh and read T2's over-coverage as a design artefact)"],
               "relabel_note": ("values identical to v4: four files byte-identical, "
                                "fused_cells_annual.parquet differs only in product_version")})
    errs = manifest.validate_manifest(mm, d)
    if errs:
        raise SystemExit(f"manifest invalid: {errs}")
    manifest.write_manifest(d, mm)
    p = guard.open_product(d, "after_the_fact_analysis")
    print(f"wrote {rel_to_repo(d)}; guard opens {p.use}; {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
