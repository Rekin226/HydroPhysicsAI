"""WP S3: supplementary report numbers (diagnostics of the evaluated acceptance, product
statistics, download sizes) and their ledger rows.

    python -I dataprep_staging/s3_gnss/report_extra.py EVIDENCE_DIR [--ledger]

Every number here is computed from files that already exist: the injection trial tables
written by ``run_validation.py``, the product tables of ``gnss_clean/{preorigin,full}/v1``,
the tune file and the L0 SOURCE.json records. Nothing is re-scored. The tolerance
decomposition of A1 is a post-hoc diagnostic (it explains the failure; it is not a gate and
changes no threshold). Writes ``EVIDENCE_DIR/report_numbers_extra.json``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import ledger  # noqa: E402
from common.paths import REPO  # noqa: E402

PRE, FULL = "gnss_clean/preorigin/v1", "gnss_clean/full/v1"
PRE_DIR, FULL_DIR = f"results/dataprep/{PRE}", f"results/dataprep/{FULL}"
TUNE = "results/dataprep/reports/S3_evidence/tune_r2/tune.json"
NGL_L0 = "data_fetch_api/new_measurements/ngl_gnss"
CAT = {("TKJS", "2022-06"): 0.132, ("TKJS", "2024-01"): -0.095, ("SJES", "2022-03"): 0.100,
       ("GFES", "2020-07"): -0.056}


def main(argv: list[str]) -> int:
    ev = Path(argv[0])
    nums: dict[str, object] = {}
    rows: list[dict] = []

    def num(nid, value, sources, pv, note=""):
        nums[nid] = value
        for s, role in sources:
            rows.append({"number_id": nid, "source": s, "role": role, "product_version": pv,
                         "note": note})

    # A1 decomposition (post-hoc diagnostic of the evaluated test)
    for var, pv in (("preorigin", PRE), ("full", FULL)):
        t = pd.read_csv(ev / f"injection_trials_{var}.csv")
        src = [(f"results/dataprep/reports/S3_evidence/acceptance/injection_trials_{var}.csv",
                "input")]
        for fam, g in t.groupby("family"):
            k = f"S3.A1diag.{var}.{fam}"
            num(f"{k}.not_found_fraction", float((~g.found).mean()), src, pv)
            num(f"{k}.found_size_err_gt5mm_fraction", float((g.found & ~g.recovered).mean()),
                src, pv)
            r10 = g.found & (g.size_err_m.abs() <= 0.010)
            num(f"{k}.recovered_tol10mm_posthoc", float(r10.mean()), src, pv,
                "post-hoc diagnostic: same trials, +-10 mm size tolerance; not a gate")
            for a, gg in g.groupby(g.abs_size_m.round(3)):
                c = f"{int(round(a * 100))}cm"
                num(f"{k}.{c}.not_found", int((~gg.found).sum()), src, pv)
                num(f"{k}.{c}.size_err_gt5mm", int((gg.found & ~gg.recovered).sum()), src, pv)
                num(f"{k}.{c}.excursion_flagged", int(gg.flagged_as_excursion.sum()), src, pv)
            ps = g.groupby("station").recovered.mean().sort_values()
            num(f"{k}.worst_stations", {s: round(float(v), 3) for s, v in ps.head(4).items()},
                src, pv)
            num(f"{k}.recovered_without_worst4_posthoc",
                float(g[~g.station.isin(ps.head(4).index)].recovered.mean()), src, pv,
                "post-hoc subgroup; not a gate")
    # product statistics
    for var, pv, d in (("preorigin", PRE, PRE_DIR), ("full", FULL, FULL_DIR)):
        src = [(d, "input")]
        for fam in ("wra", "ngl"):
            dd = pd.read_parquet(REPO / d / fam / "daily.parquet", columns=["station", "flag"])
            n = len(dd)
            for f in (0, 1, 3, 9):
                num(f"S3.product.{var}.{fam}.flag{f}_fraction", float((dd.flag == f).mean()),
                    src, pv)
            num(f"S3.product.{var}.{fam}.n_daily", int(n), src, pv)
            num(f"S3.product.{var}.{fam}.n_stations", int(dd.station.nunique()), src, pv)
        st = pd.read_csv(REPO / d / "steps.csv")
        st["year"] = st.date.str[:4].astype(int)
        w = st[st.family == "wra"]
        num(f"S3.product.{var}.wra.steps_2020_2022", int((w.year <= 2022).sum()), src, pv)
        num(f"S3.product.{var}.wra.steps_2023_2025", int((w.year >= 2023).sum()), src, pv)
        for (stn, ym), size in CAT.items():
            m = w[(w.station == stn) & (w.date.str[:7] == ym)]
            if len(m):
                num(f"S3.catalogue.{var}.{stn}.{ym}", {"plan_m": size,
                                                         "product_m": round(float(
                                                             m.size_m.iloc[0]), 4),
                                                         "date": m.date.iloc[0]}, src, pv)
    # validation summaries not tabulated by report_tables.py
    for var, pv, d in (("preorigin", PRE, PRE_DIR), ("full", FULL, FULL_DIR)):
        v = json.loads((ev / f"validation_{var}.json").read_text())
        src = [(d, "input")]
        for fam in ("wra", "ngl"):
            s = v["A1_injected_steps"][fam]
            for k in ("found_within_3d", "size_err_mm_p90", "size_err_mm_median",
                      "extra_steps_per_trial", "n"):
                num(f"S3.A1.{var}.{fam}.{k}", s.get(k), src, pv)
            g = v["A1_secondary_harness_family_G"][fam]
            num(f"S3.A1G.{var}.{fam}.recovered_dense_enough", g["recovered_dense_enough"],
                src, pv)
            num(f"S3.A1G.{var}.{fam}.n", g["n"], src, pv)
        a2 = v["A2_2021_extra"]
        src2 = src + [("chou-shui-data/data/ls_cache/clean", "validator")]
        num(f"S3.A2.{var}.ratio_cmr_to_uncorrected", a2["ratio_cmr_to_uncorrected_pooled"],
            src2, pv)
        num(f"S3.A2.{var}.n_pairs", a2["n_pairs"], src2, pv)
        num(f"S3.A2.{var}.n_reference_evaluable", a2["n_evaluable"], src2, pv)
        num(f"S3.A2.{var}.n_evaluable_with_gnss",
            a2["cm_removed_adopted_wra_hf"]["n_evaluable"], src2, pv)
        if "A3_excess_stability" in v:
            T = pd.DataFrame(v["A3_excess_stability"]["table"])
            both = T.dropna(subset=["pre_excess_mm_yr", "post_excess_mm_yr"])
            num("S3.A3.n_sites_both_windows", int(len(both)),
                src + [("chou-shui-data/data/ls_cache/clean", "input")], pv)
            num("S3.A3.median_change_both_windows", float(both.excess_change_mm_yr.median()),
                src + [("chou-shui-data/data/ls_cache/clean", "input")], pv)
    # NGL downloads
    tot, n_files = 0, 0
    for sub in ("2026-10-12_s3", "2026-10-12_s3b"):
        sj = json.loads((REPO / NGL_L0 / sub / "SOURCE.json").read_text())
        for f in sj["files"]:
            tot += int(f.get("bytes") or (REPO / NGL_L0 / sub / f["file"]).stat().st_size)
            n_files += 1
        num(f"S3.l0.{sub}.n_files", len(sj["files"]),
            [(f"{NGL_L0}/{sub}/SOURCE.json", "input")], "L0")
    num("S3.l0.ngl_s3_bytes", tot, [(f"{NGL_L0}/2026-10-12_s3/SOURCE.json", "input"),
                                    (f"{NGL_L0}/2026-10-12_s3b/SOURCE.json", "input")], "L0")
    # tune evidence
    t = json.loads((REPO / TUNE).read_text())
    for k in ("recovered_2cm_wra", "recovered_2cm_ngl", "recovered_3cm"):
        num(f"S3.tune_r2.chosen_{k}", t["chosen"][k], [(TUNE, "design_time_evidence")],
            "gnss_clean/design (pre-origin tune r2)")
    num("S3.tune_r2.reference_stations", t["reference"], [(TUNE, "design_time_evidence")],
        "gnss_clean/design (pre-origin tune r2)")
    (ev / "report_numbers_extra.json").write_text(json.dumps(nums, indent=1, default=str))
    print(json.dumps(nums, indent=1, default=str))
    if "--ledger" in argv:
        p = ledger.append_rows("S3", rows)
        print(f"ledger: {len(rows)} rows -> {p}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
