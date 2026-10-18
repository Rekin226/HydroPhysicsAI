"""WP P1 round 2: package ``county_electricity/census_check/v<N>``, write the ledger rows of
every round-2 number, and probe the loader guard on the round-2 and v1 products.

``census_check`` holds the measured census truth used to score the frozen v1 inference
(county x scope x class x month totals in model units, both variants), the evaluation of the
pre-registered items (``P1r2.census_confirmations``, ``P1r2.census_tests``) and the post hoc
diagnostics. It is after-the-fact analysis only: it uses the census to 2026-07 and exists to
record how v1 compares with measurement, not to feed any model (WP C0 owns the census product
that forcing will use).

Usage: python -I r2_build.py <build_r2_dir> <evidence_dir>
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from common import guard, ledger, manifest  # noqa: E402
from common.hashing import sha256_file  # noqa: E402
from common.paths import REPO, rel_to_repo  # noqa: E402
from common.uses import USES  # noqa: E402

CE = "results/dataprep/county_electricity"
NEW = "data_fetch_api/new_measurements/tpc_changhua_yunlin_chiayi_pingtung_2026"
CEN = f"{NEW}/parquet"
V1PRE, V1FULL = f"{CE}/preorigin/v1", f"{CE}/full/v1"
OBS = f"{CE}/tpc_observed/v1"
TPC = "data_fetch_api/new_measurements/tpc_d007012/2026-10-12/d007012.csv"
TPC_SRC = "data_fetch_api/new_measurements/tpc_d007012/2026-10-12/SOURCE.json"
NAT_DIR = "data_fetch_api/new_measurements/moeaea_agri_electricity/2026-10-12"
GRID = "results/twin_forward/datum_gate_innov.npz"
AMP_KWH = "AMP_V2/data/pump_kwh_all.parquet"
PRODUCT = "county_electricity/census_check"
PREREGS = ["results/dataprep/preregistration/P1r2.census_confirmations.json",
           "results/dataprep/preregistration/P1r2.census_tests.json",
           "results/dataprep/preregistration/P1r2.tpc_observed_gate.json"]
CODE = [HERE / n for n in ("census_truth.py", "r2_evaluate.py", "r2_diagnostics.py",
                           "r2_build.py", "r2_prereg.py", "lib.py")] + [
    REPO / "hydrophysics/twin/pumping.py", REPO / "hydrophysics/twin/scenario.py",
    HERE.parent / "common/manifest.py", HERE.parent / "common/ledger.py",
    HERE.parent / "common/hashing.py"]
#: libraries used but deliberately not listed as code (WP A0fix is editing guard.py; listing
#: it would make the checker fail this product on A0fix's next edit); their hashes at build
#: time are recorded in the parameters instead
LIBS = [HERE.parent / "common/guard.py", HERE.parent / "harness/prereg.py",
        HERE.parent / "common/uses.py", HERE.parent / "common/flags.py"]

CARD = """# county_electricity/census_check/{pv}

Measured pump electricity of Changhua and Yunlin (Taiwan Power pump census, obtained
2026-10-12) in model units, used by WP P1 round 2 to score the frozen v1 inference
(`county_electricity/{{preorigin,full}}/v1`). After-the-fact analysis only.

- **Content.** `census_truth_monthly.parquet`: county (changhua, yunlin) x variant (preorigin:
  kWh truncated to <= 2022-12 before cleaning; full: to 2026-07) x series (clean = model
  cleaning `clean_census` cap 1.0, duty window 2012-2022; dedup = cap None) x scope (county,
  in_grid on the twin's active cells) x purpose class x month, GWh. `meter_counts.parquet`:
  meters with a non-missing / non-zero reading per month. `evaluation.json`: the
  pre-registered comparisons (C1-C9, T10-T13). `diagnostics_post_hoc.json`: post hoc only.
  `annual_comparison.parquet`, `truth_series_used.parquet`, `truth_info.json`.
- **Flags.** 0 (observed census aggregates); 9 for 2026-07 (partial month: one billing group
  missing; kept, never scored).
- **Gate.** Construction check (pre-registered C9 'identical' and preorigin == full on every
  month <= 2022-12): passed = {gate}. C9 is not identical because the new sheet holds 222
  pump-month rows (8.3 MWh in total) absent from the AMP_V2 cache; all common rows agree
  exactly.
- **Allowed uses.** after_the_fact_analysis only. WP C0 builds the census forcing product.
- **Licence.** The census is restricted (local research use, no redistribution; SOURCE.json).
- **Personal data.** None: county x class x month totals and monthly meter counts only; no
  pump rows, coordinates, meter ids, names or addresses.
"""


def truth_long(td: Path, pv: str) -> pd.DataFrame:
    tm = pd.read_parquet(td / "truth_monthly.parquet").rename(columns={"gwh": "value"})
    tm["sd"] = np.nan
    partial = tm["date"] == pd.Timestamp("2026-07-01")
    tm["flag"] = np.where(partial, 9, 0).astype("int64")
    tm["method"] = np.where(tm["series"] == "clean", "census_clean_dedupe_cap1",
                            "census_dedupe_nocap")
    tm.loc[partial, "method"] = tm.loc[partial, "method"] + "|partial_month_billing_group_missing"
    tm["unit"] = "GWh/month, model units (cleaned-census kWh)"
    tm["product_version"] = pv
    tm["source_ids"] = np.where(tm["county"] == "changhua",
                                f"{CEN}/changhua_census.parquet;{CEN}/changhua_kwh_long.parquet",
                                f"{CEN}/yunlin_census.parquet;{CEN}/yunlin_kwh_long.parquet")
    return tm


def ledger_rows(ev: dict, obs_val: dict) -> list[dict]:
    C, P, F, O = CEN, f"{PRODUCT}/v1", "county_electricity/full/v1", OBS
    rows = []

    def add(num, src, role, pv, note=""):
        rows.append({"number_id": f"P1.r2.{num}", "source": src, "role": role,
                     "product_version": pv, "note": note})
    c1 = ev["C1_a_vs_census"]
    for var, v1 in (("preorigin", V1PRE), ("full", V1FULL)):
        add(f"C1.a_star_{var}", C, "input", P, f"a* = {c1[var]['a_star']:.4f}")
        add(f"C1.check_{var}", v1, "input", P, f"a_v1 {c1[var]['a_v1']:.4f}: "
            f"{c1[var]['reading']}")
        add(f"C1.check_{var}", C, "validator", P)
        add(f"C1.irrigation_{var}", v1, "input", P,
            f"a*_irr {c1[var]['irrigation']['a_star_irr']:.4f}")
        add(f"C1.irrigation_{var}", C, "validator", P)
        add(f"C1.domestic_excluded_{var}", C, "input", P,
            f"{c1[var]['changhua_domestic_excluded_over_yunlin_all']:.4f}")
    add("C1.a_star_2012_2022", C, "input", P, f"{c1['preorigin']['window_2012_2022']:.4f}")
    add("C2.cv_ratio", C, "input", P,
        f"irrigation CV ratio {ev['C2_misspecification_link']['moy_cv_irrigation']['ratio']:.3f}")
    add("C2.link", V1PRE, "input", P)
    add("C2.link", C, "validator", P)
    c4 = ev["C4_phi_in"]
    add("C4.phi_star", C, "input", P, f"phi* = {c4['phi_star']:.4f}")
    add("C4.phi_star", GRID, "input", P)
    add("C4.check", V1PRE, "input", P, f"z = {c4['z']:.2f}, inside band {c4['inside_band']}")
    add("C4.check", C, "validator", P)
    add("C4.check", GRID, "input", P)
    for w in ev["C5_in_grid_anchor"]:
        tag = w.replace("-", "").replace("..", "_")
        r = ev["C5_in_grid_anchor"][w]
        add(f"C5.in_grid_{tag}", V1PRE, "input", P, f"ratio {r['ratio_in_grid']:.4f}")
        add(f"C5.in_grid_{tag}", C, "validator", P)
        add(f"C5.in_grid_{tag}", GRID, "input", P)
        add(f"C5.county_{tag}", V1PRE, "input", P, f"ratio {r['ratio_county']:.4f}")
        add(f"C5.county_{tag}", C, "validator", P)
    for cty in ("changhua", "yunlin"):
        add(f"C8.coverage_{cty}", C, "input", P, "census / TPC 農林漁牧, annual")
        add(f"C8.coverage_{cty}", O, "validator", P)
    add("C8.ratio_CH_over_YL", C, "input", P)
    add("C8.ratio_CH_over_YL", O, "validator", P)
    add("C9.yunlin_consistency", C, "input", P,
        f"identical {ev['C9_yunlin_consistency']['identical']}")
    add("C9.yunlin_consistency", V1PRE, "validator", P)
    add("C9.yunlin_consistency", V1FULL, "validator", P)
    t10 = ev["T10_ext_yunlin"]
    add("T10.ext_yunlin", F, "input", P, f"sum err {t10['sum_err']:+.4f}, pass {t10['pass']}")
    add("T10.ext_yunlin", C, "validator", P)
    t11 = ev["T11_ext_changhua"]
    for nm in ("scaled_apre", "scaled", "transfer"):
        add(f"T11.ext_changhua_{nm}", F, "input", P,
            f"sum err {t11[nm]['sum_err']:+.4f}, pass {t11[nm]['pass']}")
        add(f"T11.ext_changhua_{nm}", C, "validator", P)
    for nm in ev["T12_monthly_shape"]:
        add(f"T12.shape_{nm}", V1PRE, "input", P)
        add(f"T12.shape_{nm}", C, "validator", P)
    for var, v1 in (("preorigin", V1PRE), ("full", V1FULL)):
        add(f"T13.rank_{var}", v1, "input", P)
        add(f"T13.rank_{var}", C, "validator", P)
    add("diag.census_over_tpc", C, "input", P, "post hoc")
    add("diag.census_over_tpc", TPC, "input", P, "post hoc")
    add("diag.c9_rows", AMP_KWH, "input", P, "post hoc")
    add("diag.c9_rows", C, "input", P, "post hoc")
    # tpc_observed gate
    g = obs_val["gate"]
    add("G1.archive_hash", TPC, "input", O, f"pass {g['G1_archive_hash']}")
    add("G1.archive_hash", TPC_SRC, "validator", O)
    add("G1.archive_hash", NAT_DIR, "input", O)
    add("G2.completeness", TPC, "input", O, f"pass {g['G2_completeness']}")
    add("G2.completeness", NAT_DIR, "input", O)
    add("G3.internal_share", TPC, "input", O,
        f"pass {g['G3_internal_share']}; max {obs_val['G3_internal_share']['max_abs_diff_pp']:.3f}"
        f" pp on {obs_val['G3_internal_share']['rows_over_0.01pp']} rows")
    add("G4.refetch", TPC, "input", O, "identical")
    add("G4.refetch", "results/dataprep/reports/P1_evidence/r2/fetch_log_r2.json", "validator",
        O)
    add("qc_flag9", TPC, "input", O, f"{obs_val['qc_flag9']['rows_flagged_9']} rows")
    add("ag_share_diagnostic", TPC, "input", O, "post hoc")
    return rows


def probe(path: str, use: str, **kw) -> dict:
    try:
        p = guard.open_product(REPO / path, use, **kw)
        out = {"opened": True, "notes": p.notes}
        if kw.get("window_end"):
            t = p.read_table("tpc_county_sector_monthly_observed.parquet")
            out["rows"] = int(len(t))
            out["max_date"] = str(pd.to_datetime(t["date"]).max().date())
            out["flags"] = sorted(int(x) for x in t["flag"].unique())
        return out
    except Exception as exc:          # noqa: BLE001 - the refusal class is the result
        return {"opened": False, "error": type(exc).__name__, "message": str(exc)[:300]}


def main(build: str, evidence: str) -> int:
    b, evd = Path(build), Path(evidence)
    evd.mkdir(parents=True, exist_ok=True)
    td, ed = b / "census_truth", b / "evaluation"
    ev = json.loads((ed / "evaluation.json").read_text(encoding="utf-8"))
    obs_val = json.loads((REPO / OBS / "validation.json").read_text(encoding="utf-8"))
    d = manifest.new_version_dir(PRODUCT)
    pv = f"{PRODUCT}/{d.name}"
    truth_long(td, pv).to_parquet(d / "census_truth_monthly.parquet", index=False)
    for f in ("meter_counts.parquet", "truth_info.json"):
        shutil.copy2(td / f, d / f)
    for f in ("evaluation.json", "diagnostics_post_hoc.json", "annual_comparison.parquet",
              "truth_series_used.parquet"):
        shutil.copy2(ed / f, d / f)
    agree = ev["truth_info"]["preorigin_vs_full_le_2022_12"]
    gate_ok = bool(ev["C9_yunlin_consistency"]["identical"]
                   and all(v["max_abs_GWh"] == 0.0 for v in agree.values()))
    (d / "PRODUCT_CARD.md").write_text(CARD.format(pv=d.name, gate=gate_ok), encoding="utf-8")
    m = manifest.build_manifest(
        d, product=PRODUCT, layer="L2",
        inputs=[f"{CEN}/changhua_census.parquet", f"{CEN}/changhua_kwh_long.parquet",
                f"{CEN}/yunlin_census.parquet", f"{CEN}/yunlin_kwh_long.parquet",
                f"{NEW}/SOURCE.json", GRID, AMP_KWH,
                f"{V1PRE}/MANIFEST.json", f"{V1FULL}/MANIFEST.json",
                f"{V1PRE}/changhua_candidates.parquet", f"{V1FULL}/changhua_candidates.parquet",
                f"{V1FULL}/yunlin_extension.parquet",
                f"{V1PRE}/yunlin_census_class_monthly.parquet",
                f"{V1FULL}/yunlin_census_class_monthly.parquet",
                f"{V1PRE}/decomposition.json", f"{V1FULL}/decomposition.json",
                f"{V1PRE}/validation.json", f"{V1FULL}/validation.json",
                f"{OBS}/MANIFEST.json", f"{OBS}/tpc_county_sector_monthly_observed.parquet",
                TPC] + PREREGS,
        code=CODE,
        parameters={"preregistrations": {rel: sha256_file(REPO / rel) for rel in PREREGS},
                    "cleaning": "clean_census cap 1.0, duty window 2012-01..2022-12; kept "
                                "pumps' full record de-duplicated and HP-split (cap None)",
                    "variants": {"preorigin": "kWh <= 2022-12 before cleaning",
                                 "full": "kWh to 2026-07 (2026-07 partial, flag 9)"},
                    "in_grid": "pumping._cell_lookup, grid of " + GRID,
                    "census_licence": "restricted; local research use only; do not "
                                      "redistribute (SOURCE.json)",
                    "library_hashes_not_listed_as_code": {rel_to_repo(p): sha256_file(p)
                                                          for p in LIBS},
                    "status": "evaluation of frozen v1 products; nothing fitted"},
        fit_window_end="2026-07", validation_report="evaluation.json",
        gate={"passed": gate_ok,
              "criteria": "construction check: pre-registered C9 'identical' (new-L0 Yunlin "
                          "equals v1 YL_clean to 1e-6 GWh on months <= 2025-06) and "
                          "preorigin == full on every month <= 2022-12",
              "C9_identical": ev["C9_yunlin_consistency"]["identical"],
              "preorigin_equals_full": all(v["max_abs_GWh"] == 0.0 for v in agree.values())},
        allowed_uses=["after_the_fact_analysis"],
        forbidden_uses=sorted(u for u in USES if u != "after_the_fact_analysis"),
        personal_data="none (county x class x month totals and monthly meter counts; no pump "
                      "rows, coordinates, meter ids, names or addresses)")
    p = manifest.write_manifest(d, m)
    print("wrote", rel_to_repo(p))
    rows = ledger_rows(ev, obs_val)
    lp = ledger.append_rows("P1", rows)
    print("ledger rows", len(rows), "->", rel_to_repo(lp))
    allrows = ledger.read_all()
    probes = {
        "tpc_observed/v1": {
            "validator": probe(OBS, "validator"),
            "after_the_fact_analysis": probe(OBS, "after_the_fact_analysis"),
            "statistical_layer_history": probe(OBS, "statistical_layer_history"),
            "design_time_evidence (no window)": probe(OBS, "design_time_evidence"),
            "design_time_evidence window_end 2022-12": probe(OBS, "design_time_evidence",
                                                             window_end="2022-12"),
            "prior": probe(OBS, "prior")},
        "census_check/v1": {
            "after_the_fact_analysis": probe(f"{CE}/census_check/{d.name}",
                                             "after_the_fact_analysis"),
            "validator": probe(f"{CE}/census_check/{d.name}", "validator")},
        "preorigin/v1": {"validator": probe(V1PRE, "validator"),
                         "prior": probe(V1PRE, "prior"),
                         "after_the_fact_analysis": probe(V1PRE, "after_the_fact_analysis")},
        "full/v1": {"validator": probe(V1FULL, "validator"),
                    "forcing_recorded_arm": probe(V1FULL, "forcing_recorded_arm")},
        "ledger": {"rows_all_fragments": len(allrows),
                   "conflicts": ledger.conflicts(allrows),
                   "p1_rows": sum(1 for r in allrows if r["wp"] == "P1")},
    }
    (evd / "guard_probes_r2.json").write_text(json.dumps(probes, ensure_ascii=False, indent=1,
                                                         default=str), encoding="utf-8")
    print(json.dumps(probes, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
