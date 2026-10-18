"""WP S3 fix round of the S3r repair (2026-10-18): the number file and ledger rows of the fix
round (diagnosis D7, the withdrawn R7 smoke check, the R7' smoke check, the v4 validation runs
and products, the matched-set A3 and twin scoring, the minor-item numbers).

    python -I -B dataprep_staging/s3_gnss/v4/report_v4.py FIX_ROUND_DIR [--ledger]

FIX_ROUND_DIR = ``results/dataprep/reports/S3_evidence/repair_2026-10-18/fix_round``. Writes
``FIX_ROUND_DIR/report_numbers_v4.json`` (every number quoted in S3.repair.md Part 7 and in
S3.repair.json for this round, with its id) and, with ``--ledger``, appends one ledger row per
number and source to ``results/dataprep/ledger/rows/S3.csv``. Numbers are read, not retyped.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common import ledger  # noqa: E402
from common.paths import REPO  # noqa: E402

PRE4, FULL4 = "gnss_clean/preorigin/v4", "gnss_clean/full/v4"
PRE4_DIR = "results/dataprep/gnss_clean/preorigin/v4"
FULL4_DIR = "results/dataprep/gnss_clean/full/v4"
FULL3_DIR = "results/dataprep/gnss_clean/full/v3"
PRE3_DIR = "results/dataprep/gnss_clean/preorigin/v3"
FULL2_DIR = "results/dataprep/gnss_clean/full/v2"
V3_ACC = "results/dataprep/reports/S3_evidence/repair_2026-10-18/acceptance"
RINGS = "chou-shui-data/data/ls_cache/clean"
TWIN = "results/twin_forward/datum_gate_innov.npz"
DIAG_PV = "gnss_clean/full/v3 (S3r fix-round diagnosis)"
DIAG_PV_PRE = "gnss_clean/preorigin/v3 (S3r fix-round diagnosis)"
SMOKE_PV = "gnss_clean/v4 smoke (no product; S3r fix round)"
SMOKE_R7_PV = "gnss_clean/v4 R7 smoke (withdrawn design; S3r fix round)"


def _rel(p: Path) -> str:
    return Path(p).resolve().relative_to(REPO).as_posix()


def r(x, nd=4):
    if x is None:
        return None
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else round(float(x), nd)
    return x


def main(argv: list[str]) -> int:
    fr = Path(argv[0])
    nums: dict[str, object] = {}
    rows: list[dict] = []

    def num(nid: str, value, sources: list[tuple[str, str]], pv: str, note: str = ""):
        nums[nid] = value
        for s, role in sources:
            rows.append({"number_id": nid, "source": s, "role": role, "product_version": pv,
                         "note": note})

    # ------------------------------------------------------------------ D7 diagnosis
    dg = fr / "diag"
    for var, pv, prod in (("full", DIAG_PV, FULL3_DIR), ("preorigin", DIAG_PV_PRE, PRE3_DIR)):
        ev = pd.read_csv(dg / f"events_{var}.csv")
        cp = pd.to_numeric(ev.composition_part_mm, errors="coerce").abs()
        src = [(_rel(dg / f"events_{var}.csv"), "input"), (prod, "input")]
        num(f"S3.fix.diag.events.{var}.n", int(len(ev)), src, pv)
        num(f"S3.fix.diag.events.{var}.composition_abs_mm",
            {"median": r(float(np.nanmedian(cp)), 2), "max": r(float(np.nanmax(cp)), 2),
             "n_ge_10": int((cp >= 10).sum()), "n_ge_5": int((cp >= 5).sum())}, src, pv)
        if var == "full":
            ev["date"] = pd.to_datetime(ev.date)
            old = ev[ev.date <= "2022-12-31"]
            co = pd.to_numeric(old.composition_part_mm, errors="coerce").abs()
            num("S3.fix.diag.events.full.2020_22.composition_abs_mm",
                {"median": r(float(np.nanmedian(co)), 2), "max": r(float(np.nanmax(co)), 2)},
                src, pv)
            big = ev[cp >= 10]
            num("S3.fix.diag.events.full.ge10.date_range",
                [str(big.date.min().date()), str(big.date.max().date())], src, pv)
            for _, e in ev[ev.station.isin(["HLES", "YCES", "ANES", "KTES"])
                           & (ev.date.dt.strftime("%Y-%m-%d").isin(
                               ["2023-02-04", "2023-02-08", "2023-01-07"]))].iterrows():
                num(f"S3.fix.diag.event.{e.station}.{e.date.date()}",
                    {"cm_jump_mm": r(e.cm_jump_mm, 1),
                     "constant_subset_jump_mm": r(e.constant_subset_jump_mm, 1),
                     "composition_part_mm": r(e.composition_part_mm, 1),
                     "n_before_after": f"{e.n_reporting_before}->{e.n_reporting_after}"},
                    src, pv)
            hn = ev[(ev.station == "HNES") & ev.date.dt.strftime("%Y-%m-%d").isin(
                ["2023-01-09", "2023-06-02", "2023-06-14"])]
            for _, e in hn.iterrows():
                num(f"S3.fix.diag.event.HNES.{e.date.date()}",
                    {"cm_jump_mm": r(e.cm_jump_mm, 1),
                     "composition_part_mm": r(e.composition_part_mm, 1),
                     "n_constant_subset": int(e.n_constant_subset)}, src, pv)
        co = pd.read_csv(dg / f"cm_only_steps_{var}.csv")
        srcc = [(_rel(dg / f"cm_only_steps_{var}.csv"), "input"), (prod, "input")]
        num(f"S3.fix.diag.cm_only_steps.{var}",
            [{"station": x.station, "date": x.date, "size_mm": r(x.size_mm, 1),
              "z_own": r(x.z_own_series, 2),
              "composition_part_mm": r(pd.to_numeric(x.composition_part_mm, errors="coerce"), 1)}
             for x in co.itertuples()], srcc, pv)
    # flag-0/1 days 2023-25, v2 and v3 (diagnosis table)
    fl = {}
    for v, d_ in (("v2", FULL2_DIR), ("v3", FULL3_DIR)):
        d = pd.read_parquet(REPO / d_ / "wra" / "daily.parquet", columns=["station", "date", "flag"])
        d = d[(d.date >= "2023-01-01") & (d.date <= "2025-03-31")]
        fl[v] = d.groupby("station").flag.apply(lambda f: int(f.isin([0, 1]).sum())).to_dict()
    for k in ("ANES", "HLES", "HNES", "YCES", "KTES", "SSES", "GFES", "JMES", "SNES", "TKJS"):
        num(f"S3.fix.diag.flag01_days_2023_25.{k}", {"v2": fl["v2"].get(k), "v3": fl["v3"].get(k)},
            [(FULL2_DIR, "input"), (FULL3_DIR, "input")], DIAG_PV)
    a3v = {v: json.loads((REPO / d_ / "validation.json").read_text())["this_variant"]
           ["A3_excess_stability"] for v, d_ in (("v2", FULL2_DIR), ("v3", FULL3_DIR))}
    for v, d_ in (("v2", FULL2_DIR), ("v3", FULL3_DIR)):
        num(f"S3.fix.diag.A3.{v}.median_abs_change_mm_yr", r(a3v[v]["median_abs_change_mm_yr"], 2),
            [(d_, "input"), (RINGS, "input")], DIAG_PV)
    ng = json.loads((dg / "summary_ngl.json").read_text())
    for var in ("preorigin", "full"):
        num(f"S3.fix.diag.ngl.{var}", ng[var], [(_rel(dg / f"events_ngl_{var}.csv"), "input")],
            DIAG_PV if var == "full" else DIAG_PV_PRE)
    pc = json.loads((dg / "product_cm_edges.json").read_text())
    num("S3.fix.diag.product_cm_edges", [{k: r(e[k], 2) if isinstance(e[k], float) else e[k]
                                          for k in ("date", "n_before", "n_after",
                                                    "composition_part_mm")}
                                         for e in pc["edges"]],
        [(_rel(dg / "product_cm_edges.json"), "input"), (FULL3_DIR, "input")], DIAG_PV)
    # minor item 5c: found-step size-failure share by size, registered v3 trials (seed 707)
    t = pd.read_csv(REPO / V3_ACC / "injection_trials_preorigin.csv")
    t["sf"] = t.found & (t.size_err_m.abs() > 0.005)
    for (fam, a), g in t.groupby(["family", "abs_size_m"]):
        num(f"S3.fix.minor.size_fail_share_of_found.{fam}.{int(round(a * 100))}cm",
            r(float(g.sf.sum() / g.found.sum()), 4),
            [(V3_ACC + "/injection_trials_preorigin.csv", "input")], "gnss_clean/preorigin/v3")

    # ------------------------------------------------------------------ smoke checks
    for tag, sd, pv in (("r7_withdrawn", fr / "smoke_r7_withdrawn", SMOKE_R7_PV),
                        ("r7b", fr / "smoke_r7b", SMOKE_PV)):
        s = json.loads((sd / "smoke_v4.json").read_text())
        for var in ("preorigin", "full"):
            c = s[var]["composition"]
            src = [(_rel(sd / "smoke_v4.json"), "input"),
                   (_rel(sd / f"events_r7_{var}.csv"), "input")]
            num(f"S3.fix.smoke.{tag}.{var}.max_abs_diff_full_composition_days_m",
                c["max_abs_diff_full_composition_days_m"], src, pv)
            num(f"S3.fix.smoke.{tag}.{var}.composition_abs_mm",
                {"v3": {k: r(x, 2) for k, x in c["v3_composition_part_abs_mm"].items()},
                 "v4": {k: r(x, 2) for k, x in c["v4_composition_part_abs_mm"].items()}}, src, pv)
            tv = s[var]["tables_vs_v3"]
            num(f"S3.fix.smoke.{tag}.{var}.steps_v3_v4", [tv["steps_v3"], tv["steps_v4"]], src, pv)
            num(f"S3.fix.smoke.{tag}.{var}.ngl_identical", tv["ngl_daily_identical_to_v3"], src, pv)
            if var == "full":
                num(f"S3.fix.smoke.{tag}.full.flag01_days_2023_25", tv["flag01_days_2023_25"],
                    src, pv)
            if "wra_notes" in s[var]:
                num(f"S3.fix.smoke.{tag}.{var}.notes_named", s[var]["wra_notes"], src, pv)
            if var == "preorigin" and "wra_flag_changed_days" in tv:
                num(f"S3.fix.smoke.{tag}.preorigin.flag_changed_days", tv["wra_flag_changed_days"],
                    src, pv)
                num(f"S3.fix.smoke.{tag}.preorigin.common_step_size_change_mm",
                    {k: r(x, 3) for k, x in tv["common_size_abs_change_mm"].items()}, src, pv)
            num(f"S3.fix.smoke.{tag}.{var}.partial_day_cm_change_mm",
                {k: r(x, 2) for k, x in c["partial_days_abs_change_mm"].items()}, src, pv)

    # ------------------------------------------------------------------ v4 validation runs
    acc = fr / "acceptance_v4"
    v4 = {var: json.loads((acc / f"validation_{var}.json").read_text())
          for var in ("preorigin", "full")}
    for var, pv, pdir in (("preorigin", PRE4, PRE4_DIR), ("full", FULL4, FULL4_DIR)):
        v = v4[var]
        src = [(_rel(acc / f"validation_{var}.json"), "input"),
               (_rel(acc / f"injection_trials_{var}.csv"), "input")]
        a1 = v["A1_injected_steps"]
        for fam in ("wra", "ngl"):
            s = a1[fam]
            num(f"S3.v4.A1.{var}.{fam}.recovered", r(s["recovered"]), src, pv,
                "second look on spent seed-707 units")
            num(f"S3.v4.A1.{var}.{fam}.n", s["n"], src, pv)
            num(f"S3.v4.A1.{var}.{fam}.by_size",
                {k: r(x["recovered"]) for k, x in s["by_size"].items()}, src, pv)
            num(f"S3.v4.A1.{var}.{fam}.found_within_3d", r(s["found_within_3d"]), src, pv)
        g = v.get("A1_secondary_harness_family_G", {})
        for fam, s in g.items():
            num(f"S3.v4.A1G.{var}.{fam}.recovered", r(s["recovered"]),
                [(_rel(acc / f"injection_harness_g_{var}.csv"), "input")], pv)
        a2 = v["A2_2021_extra"]
        srca = [(pdir, "validator"), (RINGS, "validator"), (_rel(acc / f"validation_{var}.json"),
                                                            "input")]
        for k in ("cm_removed_ngl_offfan", "cm_removed_wra_hf", "uncorrected"):
            s = a2[k]
            num(f"S3.v4.A2.{var}.{k}", {"n": s["n_evaluable"],
                                         "same_sign": r(s["same_sign_fraction"]),
                                         "pooled_ratio": r(s["pooled_amplitude_ratio"])},
                srca, pv, "second look on spent pair units")
        num(f"S3.v4.A2.{var}.cm_own_extra_ngl_offfan_mm",
            r(a2["cm_own_extra_ngl_offfan"]["extra_mm"], 2), srca, pv)
        num(f"S3.v4.steps.{var}", v["steps_summary"], [(pdir, "input")], pv)
        num(f"S3.v4.run.{var}.seconds", v["seconds"], [(_rel(acc / f"validation_{var}.json"),
                                                         "input")], pv)
    # v3 registered vs v4 second look: trial positions in common
    t3 = pd.read_csv(REPO / V3_ACC / "injection_trials_preorigin.csv")
    t4 = pd.read_csv(acc / "injection_trials_preorigin.csv")
    key = ["family", "trial", "station", "inject_date", "size_m"]
    m = t3.merge(t4, on=key, suffixes=("_v3", "_v4"))
    for fam in ("wra", "ngl"):
        mm = m[m.family == fam]
        num(f"S3.v4.A1.preorigin.{fam}.same_trials_as_v3",
            {"n_same_position": int(len(mm)), "n_v3": int((t3.family == fam).sum()),
             "recovered_v3_on_same": r(float(mm.recovered_v3.mean())),
             "recovered_v4_on_same": r(float(mm.recovered_v4.mean())),
             "changed_outcome": int((mm.recovered_v3 != mm.recovered_v4).sum())},
            [(V3_ACC + "/injection_trials_preorigin.csv", "input"),
             (_rel(acc / "injection_trials_preorigin.csv"), "input")], PRE4)
    # ------------------------------------------------------------------ matched sets
    ms = json.loads((fr / "matched_sets.json").read_text())
    srcm = [(_rel(fr / "matched_sets.json"), "input"), (FULL2_DIR, "input"),
            (FULL3_DIR, "input"), (FULL4_DIR, "input"), (RINGS, "input")]
    for k in ("own", "common_v2_v3_v4", "common_v2_v4"):
        num(f"S3.v4.A3.matched.{k}", {v: {kk: (r(x, 2) if isinstance(x, float) else x)
                                          for kk, x in d.items()}
                                      for v, d in ms["A3"][k].items()}, srcm, FULL4,
            "A3 report only (no threshold)")
    for lab in ("twin_raw_cleaned", "twin_cm_removed_wra_hf", "twin_cm_removed_ngl_offfan"):
        for k in ("own", "common_v2_v3_v4", "common_v2_v4"):
            num(f"S3.v4.{lab}.matched.{k}",
                {v: {"n": d["n"], "model_r2": r(d["model_archive"]["r2"], 3),
                     "model_rmse": r(d["model_archive"]["rmse"], 3),
                     "model_bias": r(d["model_archive"]["bias"], 3),
                     "continue_r2": r(d["base_continue_2020_22"]["r2"], 3),
                     "continue_rmse": r(d["base_continue_2020_22"]["rmse"], 3)}
                 for v, d in ms[lab][k].items()},
                srcm[:4] + [(TWIN, "input")], FULL4, "after-the-fact")
        num(f"S3.v4.{lab}.stations", ms[lab]["stations"], srcm[:4], FULL4)
    if "flag01_days_2023_25" in ms:
        num("S3.v4.flag01_days_2023_25", ms["flag01_days_2023_25"], srcm[1:4], FULL4)
    # ------------------------------------------------------------------ products, checks
    for var, pdir in (("preorigin", PRE4_DIR), ("full", FULL4_DIR)):
        mf = json.loads((REPO / pdir / "MANIFEST.json").read_text())
        num(f"S3.v4.product.{var}.gate", mf.get("gate"), [(pdir, "input")], f"gnss_clean/{var}/v4")
        st = pd.read_csv(REPO / pdir / "stations.csv")
        num(f"S3.v4.product.{var}.stations", st.groupby("family").size().to_dict(),
            [(pdir, "input")], f"gnss_clean/{var}/v4")
        for fam in ("wra", "ngl"):
            d = pd.read_parquet(REPO / pdir / fam / "daily.parquet", columns=["flag"])
            num(f"S3.v4.product.{var}.{fam}.rows_and_flag_shares",
                {"rows": int(len(d)), "flag_share": {str(k): r(float(x), 4) for k, x in
                                                     d.flag.value_counts(normalize=True)
                                                     .sort_index().items()}},
                [(pdir, "input")], f"gnss_clean/{var}/v4")
    for nm in ("checker_gnss_clean_v4.json", "guard_probe_v4.json"):
        p = fr / nm
        if p.exists():
            num(f"S3.v4.{nm.split('.')[0]}", json.loads(p.read_text()) if nm.startswith("guard")
                else {"summary": json.loads(p.read_text()).get("summary")},
                [(_rel(p), "input")], FULL4)
    (fr / "report_numbers_v4.json").write_text(json.dumps(nums, indent=1, ensure_ascii=False,
                                                          default=str))
    print(f"{len(nums)} numbers, {len(rows)} rows")
    if "--ledger" in argv:
        p = ledger.append_rows("S3", rows)
        print(f"ledger: {len(rows)} rows -> {p}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
