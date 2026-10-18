"""WP S3: markdown tables and ledger rows from the evidence and the products.

    python -I dataprep_staging/s3_gnss/report_tables.py EVIDENCE_DIR TUNE_JSON [--ledger]

Prints the markdown tables used in ``results/dataprep/reports/S3.md`` and writes
``EVIDENCE_DIR/report_numbers.json`` (every number quoted, with its id); ``--ledger`` appends
the ledger rows (``results/dataprep/ledger/rows/S3.csv``).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import ledger  # noqa: E402

PRE = "gnss_clean/preorigin/v1"
FULL = "gnss_clean/full/v1"
PRE_DIR = "results/dataprep/gnss_clean/preorigin/v1"
FULL_DIR = "results/dataprep/gnss_clean/full/v1"
RINGS = "chou-shui-data/data/ls_cache/clean"
PILES = ["chou-shui-data/data/ls_cache/ls-wra-dbm-obs__TKJS_281_29.parquet",
         "chou-shui-data/data/ls_cache/ls-wra-dbm-obs__NTUH_282_29.parquet",
         "chou-shui-data/data/ls_cache/ls-wra-dbm-obs__SJES.parquet",
         "chou-shui-data/data/ls_cache/ls-wra-dbm-obs__STES.parquet"]
TWIN = "results/twin_forward/datum_gate_innov.npz"


def f(x, nd=3):
    return "n/a" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def a1_table(v: dict, label: str) -> list[str]:
    a1 = v["A1_injected_steps"]
    out = [f"| {label} | family | n | recovered | found within 3 d | 2 cm | 3 cm | 5 cm | 10 cm "
           "| size err p90 (mm, found) | extra steps / trial | pass (>= 0.90) |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for fam in ("wra", "ngl"):
        s = a1[fam]
        bs = s.get("by_size", {})
        out.append(f"| | {fam} | {s['n']} | {f(s['recovered'])} | {f(s['found_within_3d'])} | "
                   + " | ".join(f(bs.get(k, {}).get("recovered")) for k in
                                ("2cm", "3cm", "5cm", "10cm"))
                   + f" | {f(s.get('size_err_mm_p90'), 1)} | "
                     f"{f(s.get('extra_steps_per_trial'), 2)} | {s['pass']} |")
    return out


def main(argv: list[str]) -> int:
    ev = Path(argv[0])
    tune = json.loads(Path(argv[1]).read_text())
    vp = json.loads((ev / "validation_preorigin.json").read_text())
    vf = json.loads((ev / "validation_full.json").read_text())
    nums: dict[str, object] = {}
    rows: list[dict] = []

    def num(nid: str, value, sources: list[tuple[str, str]], pv: str, note: str = ""):
        nums[nid] = value
        for s, role in sources:
            rows.append({"number_id": nid, "source": s, "role": role, "product_version": pv,
                         "note": note})

    md = ["## A1 injected steps", ""] + a1_table(vp, "pre-origin (gate)") + [""] + \
        a1_table(vf, "full records (reported)") + [""]
    for var, v, pv, d in (("preorigin", vp, PRE, PRE_DIR), ("full", vf, FULL, FULL_DIR)):
        for fam in ("wra", "ngl"):
            s = v["A1_injected_steps"][fam]
            num(f"S3.A1.{var}.{fam}.recovered", s["recovered"],
                [(d, "input")], pv, "synthetic steps seed 303 in the product pipeline")
            for k, b in s.get("by_size", {}).items():
                num(f"S3.A1.{var}.{fam}.recovered_{k}", b["recovered"], [(d, "input")], pv)
        g = v.get("A1_secondary_harness_family_G", {})
        for fam, s in g.items():
            num(f"S3.A1G.{var}.{fam}.recovered", s["recovered"], [(d, "input")], pv,
                "harness family G r=0..9")
    # A2
    md += ["## A2 2021 extra compaction", "",
           "| variant | GNSS series | evaluable pairs | same sign | pooled ratio | pass |",
           "|---|---|---|---|---|---|"]
    for var, v, pv, d in (("preorigin", vp, PRE, PRE_DIR), ("full", vf, FULL, FULL_DIR)):
        a2 = v["A2_2021_extra"]
        for k in ("cm_removed_adopted_wra_hf", "uncorrected", "cm_removed_pilot_onfan_fullband",
                  "cm_removed_ngl_offfan"):
            s = a2.get(k, {})
            md.append(f"| {var} | {k} | {s.get('n_evaluable')} | "
                      f"{f(s.get('same_sign_fraction'), 2)} | "
                      f"{f(s.get('pooled_amplitude_ratio'), 2)} | {s.get('pass')} |")
            src = [(d, "input"), (RINGS, "validator")] + [(p, "validator") for p in PILES]
            num(f"S3.A2.{var}.{k}.same_sign", s.get("same_sign_fraction"), src, pv)
            num(f"S3.A2.{var}.{k}.pooled_ratio", s.get("pooled_amplitude_ratio"), src, pv)
    md += ["", "Pairs (pre-origin variant):", "",
           "| kind | GNSS | reference | dist (m) | ref 2020 (mm) | ref 2021 (mm) | ref extra | "
           "GNSS extra (wra_hf removed) | uncorrected | on-fan removed | NGL off-fan removed |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in vp["A2_2021_extra"]["pairs"]:
        md.append(f"| {r['kind']} | {r['gnss']} | {r['reference']} | {f(r['distance_m'], 0)} | "
                  f"{f(r['ref_D2020_mm'], 1)} | {f(r['ref_D2021_mm'], 1)} | "
                  f"{f(r['ref_extra_mm'], 1)} | {f(r['cm_removed_adopted_wra_hf_extra_mm'], 1)} | "
                  f"{f(r['uncorrected_extra_mm'], 1)} | "
                  f"{f(r['cm_removed_pilot_onfan_fullband_extra_mm'], 1)} | "
                  f"{f(r['cm_removed_ngl_offfan_extra_mm'], 1)} |")
    # A3
    a3 = vf["A3_excess_stability"]
    md += ["", "## A3 excess-rate stability (reported)", "",
           "| GNSS | ring site | pre GNSS | pre ring | pre excess | post GNSS | post ring | "
           "post excess | change (mm/yr) |", "|---|---|---|---|---|---|---|---|---|"]
    for r in a3["table"]:
        md.append(f"| {r['gnss']} | {r['site']} | {f(r['pre_gnss_mm_yr'], 1)} | "
                  f"{f(r['pre_ring_mm_yr'], 1)} | {f(r['pre_excess_mm_yr'], 1)} | "
                  f"{f(r['post_gnss_mm_yr'], 1)} | {f(r['post_ring_mm_yr'], 1)} | "
                  f"{f(r['post_excess_mm_yr'], 1)} | {f(r['excess_change_mm_yr'], 1)} |")
    md.append(f"\nMedian excess: pre {f(a3['median_pre_excess_mm_yr'], 1)}, post "
              f"{f(a3['median_post_excess_mm_yr'], 1)} mm/yr; median |change| "
              f"{f(a3['median_abs_change_mm_yr'], 1)} mm/yr ({a3['n_sites']} sites).")
    src3 = [(FULL_DIR, "input"), (RINGS, "input")]
    num("S3.A3.median_pre_excess", a3["median_pre_excess_mm_yr"], src3, FULL)
    num("S3.A3.median_post_excess", a3["median_post_excess_mm_yr"], src3, FULL)
    num("S3.A3.median_abs_change", a3["median_abs_change_mm_yr"], src3, FULL)
    # CM scatter
    md += ["", "## Common-mode scatter (median station residual sd, mm)", "",
           "| variant | stations | none | ngl_offfan | wra_hf | wra_onfan_fullband (diag) |",
           "|---|---|---|---|---|---|"]
    for var, v, pv, d in (("preorigin", vp, PRE, PRE_DIR), ("full", vf, FULL, FULL_DIR)):
        s = v["cm_scatter_wra"]
        md.append(f"| {var} | WRA members | {f(s['none'], 2)} | {f(s['ngl_offfan'], 2)} | "
                  f"{f(s['wra_hf'], 2)} | {f(s['wra_onfan_fullband'], 2)} |")
        n = v["cm_scatter_ngl_fan"]
        md.append(f"| {var} | NGL fan (non-reference) | {f(n['none'], 2)} | "
                  f"{f(n['ngl_offfan'], 2)} | | |")
        for k, val in s.items():
            num(f"S3.cm_scatter.{var}.wra.{k}", val, [(d, "input")], pv)
        for k, val in n.items():
            num(f"S3.cm_scatter.{var}.nglfan.{k}", val, [(d, "input")], pv)
        pc = v["pca_wra"]
        for k in ("pc1_variance_fraction", "pc1_loadings_positive_fraction",
                  "corr_pc1_wra_hf", "corr_pc1_ngl_offfan_highpass"):
            num(f"S3.pca.{var}.{k}", pc[k], [(d, "input")], pv)
    md += ["", "PCA of the 31-day high-passed WRA residuals: "
           + "; ".join(f"{var}: PC1 {f(v['pca_wra']['pc1_variance_fraction'], 2)} of variance, "
                       f"loadings positive {f(v['pca_wra']['pc1_loadings_positive_fraction'], 2)}"
                       f", CV {f(v['pca_wra']['pc1_loadings_cv'], 2)}, corr with wra_hf "
                       f"{f(v['pca_wra']['corr_pc1_wra_hf'], 2)}, with high-passed ngl_offfan "
                       f"{f(v['pca_wra']['corr_pc1_ngl_offfan_highpass'], 2)}"
                       for var, v in (("pre-origin", vp), ("full", vf)))]
    # twin scores
    md += ["", "## Frozen forward archive against GNSS rates (after-the-fact)", "",
           "| family | window | series | n | model R2 | model RMSE (cm/yr) | model bias | "
           "continue-2020-22 R2 | RMSE | zero R2 | RMSE |", "|---|---|---|---|---|---|---|---|"
           "---|---|---|"]
    for fam in ("wra", "ngl"):
        ts = vf[f"twin_scores_{fam}"]
        for k, s in ts.items():
            if not isinstance(s, dict) or "model_archive" not in s:
                continue
            ma, bc, bz = s["model_archive"], s["base_continue_2020_22"], s["base_zero"]
            md.append(f"| {fam} | {ts['window'][0][:7]}..{ts['window'][1][:7]} | {k} | "
                      f"{s['n_stations']} | {f(ma.get('r2'), 2)} | {f(ma.get('rmse'), 2)} | "
                      f"{f(ma.get('bias'), 2)} | {f(bc.get('r2'), 2)} | {f(bc.get('rmse'), 2)} | "
                      f"{f(bz.get('r2'), 2)} | {f(bz.get('rmse'), 2)} |")
            src = [(FULL_DIR, "validator"), (TWIN, "input")]
            for m_, b in (("model", ma), ("continue", bc), ("zero", bz)):
                num(f"S3.twin.{fam}.{k}.{m_}.r2", b.get("r2"), src, FULL)
                num(f"S3.twin.{fam}.{k}.{m_}.rmse", b.get("rmse"), src, FULL)
    # steps
    md += ["", "## Steps", ""]
    for var, v, pv, d in (("preorigin", vp, PRE, PRE_DIR), ("full", vf, FULL, FULL_DIR)):
        s = v["steps_summary"]
        md.append(f"- {var}: {s['n_steps']} steps; by basis {s['by_family_basis']}; untied "
                  f"{s['untied']}")
        num(f"S3.steps.{var}.n", s["n_steps"], [(d, "input")], pv)
    # tune
    ch = tune["chosen"]
    num("S3.tune_r2.chosen_recovered_2cm", ch["recovered_2cm"],
        [("results/dataprep/reports/S3_evidence/tune_r2/tune.json", "design_time_evidence")],
        "gnss_clean/design (pre-origin tune r2)")
    num("S3.tune_r2.chosen_false_rate", ch["false_per_station_year_offfan"],
        [("results/dataprep/reports/S3_evidence/tune_r2/tune.json", "design_time_evidence")],
        "gnss_clean/design (pre-origin tune r2)")
    (ev / "report_numbers.json").write_text(json.dumps(nums, indent=1, ensure_ascii=False,
                                                       default=str))
    print("\n".join(md))
    if "--ledger" in argv:
        p = ledger.append_rows("S3", rows)
        print(f"\nledger: {len(rows)} rows -> {p}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
