"""WP S3 repair round: markdown tables, the number file and ledger rows for the v2 products.

    python -I dataprep_staging/s3_gnss/v2/report_v2.py EVIDENCE_DIR TUNE_JSON DIAG_JSON [--ledger]

DIAG_JSON = the v2 A2 statistics on gnss_clean/preorigin/v1 (``a2_v1_diag.py`` output).

Prints the markdown tables used in the Repair round section of
``results/dataprep/reports/S3.md``, writes ``EVIDENCE_DIR/report_numbers_v2.json`` (every number
quoted, with its id) and, with ``--ledger``, appends the ledger rows to
``results/dataprep/ledger/rows/S3.csv`` (one role per number and source).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from common import ledger  # noqa: E402
from common.paths import REPO  # noqa: E402

PRE = "gnss_clean/preorigin/v2"
FULL = "gnss_clean/full/v2"
PRE_DIR = "results/dataprep/gnss_clean/preorigin/v2"
FULL_DIR = "results/dataprep/gnss_clean/full/v2"
V1_PRE_DIR = "results/dataprep/gnss_clean/preorigin/v1"
RINGS = "chou-shui-data/data/ls_cache/clean"
PILES = ["chou-shui-data/data/ls_cache/ls-wra-dbm-obs__TKJS_281_29.parquet",
         "chou-shui-data/data/ls_cache/ls-wra-dbm-obs__NTUH_282_29.parquet",
         "chou-shui-data/data/ls_cache/ls-wra-dbm-obs__SJES.parquet",
         "chou-shui-data/data/ls_cache/ls-wra-dbm-obs__STES.parquet"]
TWIN = "results/twin_forward/datum_gate_innov.npz"
A2_KEYS = ("cm_removed_ngl_offfan", "cm_removed_wra_hf", "uncorrected",
           "cm_removed_pilot_onfan_fullband")


def f(x, nd=3):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


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
    tune_path = Path(argv[1])
    tune = json.loads(tune_path.read_text())
    tune_rel = tune_path.resolve().relative_to(REPO).as_posix()
    vp = json.loads((ev / "validation_preorigin.json").read_text())
    vf = json.loads((ev / "validation_full.json").read_text())
    diag_path = Path(argv[2])
    diag = json.loads(diag_path.read_text())["A2_v2_statistics_on_v1"]
    diag_rel = diag_path.resolve().relative_to(REPO).as_posix()
    nums: dict[str, object] = {}
    rows: list[dict] = []

    def num(nid: str, value, sources: list[tuple[str, str]], pv: str, note: str = ""):
        nums[nid] = value
        for s, role in sources:
            rows.append({"number_id": nid, "source": s, "role": role, "product_version": pv,
                         "note": note})

    md = ["## A1 injected steps (v2)", ""] + a1_table(vp, "pre-origin (gate)") + [""] + \
        a1_table(vf, "full records (reported)") + [""]
    for var, v, pv, d in (("preorigin", vp, PRE, PRE_DIR), ("full", vf, FULL, FULL_DIR)):
        for fam in ("wra", "ngl"):
            s = v["A1_injected_steps"][fam]
            src = [(d, "input")]
            num(f"S3.v2.A1.{var}.{fam}.recovered", s["recovered"], src, pv,
                "synthetic steps seed 404 in the v2 product pipeline")
            num(f"S3.v2.A1.{var}.{fam}.n", s["n"], src, pv)
            num(f"S3.v2.A1.{var}.{fam}.found_within_3d", s["found_within_3d"], src, pv)
            num(f"S3.v2.A1.{var}.{fam}.size_err_mm_p90", s.get("size_err_mm_p90"), src, pv)
            for k, b in s.get("by_size", {}).items():
                num(f"S3.v2.A1.{var}.{fam}.recovered_{k}", b["recovered"], src, pv)
        g = v.get("A1_secondary_harness_family_G", {})
        for fam, s in g.items():
            num(f"S3.v2.A1G.{var}.{fam}.recovered", s["recovered"], [(d, "input")], pv,
                "harness family G r=10..19")
    # A1 failure decomposition from the trial tables (pre-origin gate)
    tr = pd.read_csv(ev / "injection_trials_preorigin.csv")
    md += ["A1 failure modes (pre-origin, v2):", ""]
    for fam, t in tr.groupby("family"):
        for a, g in t.groupby(t.abs_size_m.round(3)):
            nf = int((~g.found).sum())
            big = int((g.found & (g.size_err_m.abs() > 0.005)).sum())
            md.append(f"- {fam} {a * 100:.0f} cm: {len(g)} trials, not found {nf}, found with "
                      f"|size err| > 5 mm {big}")
            num(f"S3.v2.A1.preorigin.{fam}.{a * 100:.0f}cm.not_found", nf,
                [(f"{_rel(ev)}/injection_trials_preorigin.csv", "input")], PRE)
            num(f"S3.v2.A1.preorigin.{fam}.{a * 100:.0f}cm.size_err_gt5mm", big,
                [(f"{_rel(ev)}/injection_trials_preorigin.csv", "input")], PRE)
        st = t.groupby("station").recovered.mean().sort_values()
        md.append(f"- {fam} worst stations: " + ", ".join(f"{k} {x:.3f}" for k, x in
                                                         st.head(5).items()))
        num(f"S3.v2.A1.preorigin.{fam}.worst_stations", {k: round(float(x), 3) for k, x in
                                                         st.head(5).items()},
            [(f"{_rel(ev)}/injection_trials_preorigin.csv", "input")], PRE)
        loose = float((t.found & (t.size_err_m.abs() <= 0.010)).mean())
        num(f"S3.v2.A1.preorigin.{fam}.posthoc_10mm", loose,
            [(f"{_rel(ev)}/injection_trials_preorigin.csv", "input")], PRE,
            "post-hoc diagnostic, not a gate")
        md.append(f"- {fam} post-hoc +-10 mm tolerance (diagnostic, not a gate): {loose:.3f}")
    md.append("")
    # A2
    md += ["## A2 2021 extra compaction (v2)", "",
           "| variant | GNSS series | evaluable pairs | same sign | pooled ratio | pooled GNSS "
           "extra (mm) | / uncorrected | unique stations: n, same sign, ratio | pass |",
           "|---|---|---|---|---|---|---|---|---|"]
    srcA2 = [(RINGS, "validator")] + [(p, "validator") for p in PILES]
    for var, v, pv, d in (("preorigin", vp, PRE, PRE_DIR), ("full", vf, FULL, FULL_DIR),
                          ("preorigin v1 (diag)", {"A2_2021_extra": diag}, "gnss_clean/"
                           "preorigin/v1", V1_PRE_DIR)):
        a2 = v["A2_2021_extra"]
        tag = var.split()[0] if "v1" not in var else "v1diag"
        for k in A2_KEYS:
            s = a2.get(k, {})
            u = s.get("unique_gnss", {})
            md.append(f"| {var} | {k} | {s.get('n_evaluable')} | "
                      f"{f(s.get('same_sign_fraction'), 3)} | "
                      f"{f(s.get('pooled_amplitude_ratio'), 3)} | "
                      f"{f(s.get('pooled_gnss_extra_mm'), 1)} | "
                      f"{f(s.get('ratio_to_uncorrected_pooled'), 2)} | "
                      f"{u.get('n')}, {f(u.get('same_sign_fraction'), 2)}, "
                      f"{f(u.get('pooled_amplitude_ratio'), 2)} | {s.get('pass')} |")
            src = [(d if tag != "v1diag" else V1_PRE_DIR, "input")] + srcA2
            if tag == "v1diag":
                src.append((diag_rel, "input"))
            for kk in ("same_sign_fraction", "pooled_amplitude_ratio", "pooled_gnss_extra_mm",
                       "ratio_to_uncorrected_pooled", "n_evaluable"):
                num(f"S3.v2.A2.{tag}.{k}.{kk}", s.get(kk), src, pv)
            num(f"S3.v2.A2.{tag}.{k}.unique", u, src, pv)
        for col in ("ngl_offfan", "wra_hf", "wra_onfan_fullband"):
            ce = a2.get(f"cm_own_extra_{col}")
            if ce:
                src = [(d if tag != "v1diag" else V1_PRE_DIR, "input")]
                if tag == "v1diag":
                    src.append((diag_rel, "input"))
                num(f"S3.v2.A2.{tag}.cm_own_extra.{col}", {k: ce.get(k) for k in (
                    "extra_mm", "other_years_sd_mm", "other_years_n", "rank_abs_of_year")},
                    src, pv)
        num(f"S3.v2.A2.{tag}.pass", a2.get("pass"),
            [(d if tag != "v1diag" else V1_PRE_DIR, "input")] + srcA2, pv)
    md += ["", "Common modes' own 2021 extra (height, D2021 - D2020, mm; pre-origin):", ""]
    for var, v in (("v2", vp), ("v1 (diag)", {"A2_2021_extra": diag})):
        for col in ("ngl_offfan", "wra_hf", "wra_onfan_fullband"):
            ce = v["A2_2021_extra"].get(f"cm_own_extra_{col}", {})
            if ce:
                md.append(f"- {var} {col}: {f(ce.get('extra_mm'), 2)} mm; other years sd "
                          f"{f(ce.get('other_years_sd_mm'), 2)} (n {ce.get('other_years_n')}), "
                          f"years at least as large {ce.get('rank_abs_of_year')}; by year "
                          f"{ce.get('extra_by_year_mm')}")
    md += ["", "Pairs (v2 pre-origin variant):", "",
           "| kind | GNSS | reference | ref extra (mm) | ngl_offfan removed | wra_hf removed | "
           "uncorrected | on-fan removed |", "|---|---|---|---|---|---|---|---|"]
    for r in vp["A2_2021_extra"]["pairs"]:
        md.append(f"| {r['kind']} | {r['gnss']} | {r['reference']} | {f(r['ref_extra_mm'], 1)} "
                  f"| {f(r.get('cm_removed_ngl_offfan_extra_mm'), 1)} | "
                  f"{f(r.get('cm_removed_wra_hf_extra_mm'), 1)} | "
                  f"{f(r.get('uncorrected_extra_mm'), 1)} | "
                  f"{f(r.get('cm_removed_pilot_onfan_fullband_extra_mm'), 1)} |")
    # A3
    a3 = vf["A3_excess_stability"]
    md += ["", "## A3 excess-rate stability (reported, v2 full)", "",
           "| GNSS | ring site | pre excess | post excess | change (mm/yr) | months pre/post |",
           "|---|---|---|---|---|---|"]
    for r in a3["table"]:
        md.append(f"| {r['gnss']} | {r['site']} | {f(r['pre_excess_mm_yr'], 1)} | "
                  f"{f(r['post_excess_mm_yr'], 1)} | {f(r['excess_change_mm_yr'], 1)} | "
                  f"{r['pre_n_months']}/{r['post_n_months']} |")
    t3 = pd.DataFrame(a3["table"])
    both = t3.dropna(subset=["pre_excess_mm_yr", "post_excess_mm_yr"])
    src3 = [(FULL_DIR, "input"), (RINGS, "input")]
    num("S3.v2.A3.median_pre_excess", a3["median_pre_excess_mm_yr"], src3, FULL)
    num("S3.v2.A3.median_post_excess", a3["median_post_excess_mm_yr"], src3, FULL)
    num("S3.v2.A3.n_pre", int(t3.pre_excess_mm_yr.notna().sum()), src3, FULL)
    num("S3.v2.A3.n_post", int(t3.post_excess_mm_yr.notna().sum()), src3, FULL)
    num("S3.v2.A3.n_both", int(len(both)), src3, FULL)
    num("S3.v2.A3.median_change_both", float(both.excess_change_mm_yr.median())
        if len(both) else None, src3, FULL)
    md.append(f"\nMedian excess: pre {f(a3['median_pre_excess_mm_yr'], 1)} "
              f"({int(t3.pre_excess_mm_yr.notna().sum())} sites), post "
              f"{f(a3['median_post_excess_mm_yr'], 1)} ({int(t3.post_excess_mm_yr.notna().sum())}"
              f" sites); {len(both)} sites in both, median change "
              f"{f(float(both.excess_change_mm_yr.median()) if len(both) else None, 1)} mm/yr.")
    # CM scatter / PCA
    md += ["", "## Common-mode scatter (median station residual sd, mm; v2)", "",
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
            num(f"S3.v2.cm_scatter.{var}.wra.{k}", val, [(d, "input")], pv)
        for k, val in n.items():
            num(f"S3.v2.cm_scatter.{var}.nglfan.{k}", val, [(d, "input")], pv)
        pc = v["pca_wra"]
        for k in ("pc1_variance_fraction", "pc1_loadings_positive_fraction", "pc1_loadings_cv",
                  "corr_pc1_wra_hf", "corr_pc1_ngl_offfan_highpass"):
            num(f"S3.v2.pca.{var}.{k}", pc[k], [(d, "input")], pv)
        md.append(f"| {var} | PCA | PC1 {f(pc['pc1_variance_fraction'], 2)}, positive "
                  f"{f(pc['pc1_loadings_positive_fraction'], 2)}, CV "
                  f"{f(pc['pc1_loadings_cv'], 2)}, corr wra_hf {f(pc['corr_pc1_wra_hf'], 2)}, "
                  f"corr hp ngl_offfan {f(pc['corr_pc1_ngl_offfan_highpass'], 2)} | | | | |")
    # twin scores
    md += ["", "## Frozen forward archive against GNSS rates (after-the-fact, v2 full)", "",
           "| family | series | n | model R2 | RMSE (cm/yr) | bias | continue R2 / RMSE | "
           "zero R2 / RMSE |", "|---|---|---|---|---|---|---|---|"]
    for fam in ("wra", "ngl"):
        ts = vf[f"twin_scores_{fam}"]
        for k, s in ts.items():
            if not isinstance(s, dict) or "model_archive" not in s:
                continue
            ma, bc, bz = s["model_archive"], s["base_continue_2020_22"], s["base_zero"]
            md.append(f"| {fam} | {k} | {s['n_stations']} | {f(ma.get('r2'), 2)} | "
                      f"{f(ma.get('rmse'), 2)} | {f(ma.get('bias'), 2)} | "
                      f"{f(bc.get('r2'), 2)} / {f(bc.get('rmse'), 2)} | "
                      f"{f(bz.get('r2'), 2)} / {f(bz.get('rmse'), 2)} |")
            src = [(FULL_DIR, "validator"), (TWIN, "input")]
            num(f"S3.v2.twin.{fam}.{k}.n", s["n_stations"], src, FULL)
            for m_, b in (("model", ma), ("continue", bc), ("zero", bz)):
                for q in ("r2", "rmse", "bias"):
                    num(f"S3.v2.twin.{fam}.{k}.{m_}.{q}", b.get(q), src, FULL)
    # products: flags, steps, stations, re-test, jump clusters
    md += ["", "## v2 products", ""]
    for var, pv, d in (("preorigin", PRE, PRE_DIR), ("full", FULL, FULL_DIR)):
        for fam in ("wra", "ngl"):
            dd = pd.read_parquet(REPO / d / fam / "daily.parquet", columns=["station", "flag",
                                                                           "method", "date"])
            src = [(d, "input")]
            shares = {int(k): round(float(x), 4) for k, x in
                      dd.flag.value_counts(normalize=True).sort_index().items()}
            num(f"S3.v2.product.{var}.{fam}.flag_shares", shares, src, pv)
            num(f"S3.v2.product.{var}.{fam}.n_daily", int(len(dd)), src, pv)
            num(f"S3.v2.product.{var}.{fam}.n_stations", int(dd.station.nunique()), src, pv)
            line = (f"- {var} {fam}: {dd.station.nunique()} stations, {len(dd)} daily rows, "
                    f"flag shares {shares}")
            jc = dd[dd.method == "jump_cluster"]
            if len(jc):
                per = jc.groupby("station").date.agg(["min", "max", "size"])
                jcd = {k: f"{r['min']}..{r['max']} ({r['size']} d)" for k, r in per.iterrows()}
                num(f"S3.v2.product.{var}.{fam}.jump_cluster", jcd, src, pv)
                line += f"; jump-cluster flag 9: {jcd}"
            md.append(line)
        st = pd.read_csv(REPO / d / "steps.csv")
        src = [(d, "input")]
        w = st[st.family == "wra"]
        yr = pd.to_datetime(w.date).dt.year
        num(f"S3.v2.steps.{var}.n", int(len(st)), src, pv)
        num(f"S3.v2.steps.{var}.wra_n", int(len(w)), src, pv)
        num(f"S3.v2.steps.{var}.wra_2020_2022", int((yr <= 2022).sum()), src, pv)
        num(f"S3.v2.steps.{var}.wra_2023_2025", int((yr >= 2023).sum()), src, pv)
        small = st[(st.evidence == "detected") & (st.size_m.abs() < tune["chosen_params"]
                                                  ["delta_min"])]
        num(f"S3.v2.steps.{var}.detected_below_delta_min", int(len(small)), src, pv)
        cmo = st[st.basis == "common_mode_only"]
        num(f"S3.v2.steps.{var}.common_mode_only", int(len(cmo)), src, pv)
        md.append(f"- {var} steps: {len(st)} (WRA {len(w)}: {(yr <= 2022).sum()} in 2020-22, "
                  f"{(yr >= 2023).sum()} in 2023-25); common-mode-only {len(cmo)}; detected "
                  f"steps with |size| < delta_min: {len(small)}")
        sta = pd.read_csv(REPO / d / "stations.csv")
        notes = sta.get("notes", pd.Series(dtype=str)).fillna("")
        dropped = [x for s_ in notes for x in s_.split(" | ") if "re-test" in x]
        num(f"S3.v2.retest.{var}.n_dropped", len(dropped), src, pv)
        md.append(f"- {var}: {len(dropped)} detected steps dropped by the re-test")
        for k in ("wra", "ngl"):
            s2 = sta[sta.family == k]
            num(f"S3.v2.stations.{var}.{k}", {"roles": s2.role.value_counts().to_dict(),
                                              "stations": s2.station.tolist()}, src, pv)
    # tune r3
    ch = tune["chosen"]
    src_t = [(tune_rel, "design_time_evidence")]
    pv_t = "gnss_clean/design (pre-origin tune r3)"
    for k in ("delta_min", "z_min", "recovered_2cm", "recovered_2cm_wra", "recovered_2cm_ngl",
              "recovered_3cm", "false_per_station_year_offfan"):
        num(f"S3.v2.tune_r3.chosen_{k}", ch[k], src_t, pv_t)
    num("S3.v2.tune_r3.local_half", tune["chosen_local_half"], src_t, pv_t)
    num("S3.v2.tune_r3.hp_window", tune["chosen_params"]["hp_window"], src_t, pv_t)
    num("S3.v2.tune_r3.hp_window_rule", tune["cm_evidence"]["hp_window_selected_by"], src_t,
        pv_t)
    num("S3.v2.tune_r3.reference", tune["reference"], src_t, pv_t)
    num("S3.v2.tune_r3.reference_full", tune["reference_full"], src_t, pv_t)
    num("S3.v2.tune_r3.seconds", tune["seconds"], src_t, pv_t)
    md += ["", f"Tune r3: chosen {ch['delta_min'] * 1000:.0f} mm / z {ch['z_min']} / half "
               f"{tune['chosen_local_half']} d; 2 cm {ch['recovered_2cm']:.3f} (WRA "
               f"{ch['recovered_2cm_wra']:.3f}, NGL {ch['recovered_2cm_ngl']:.3f}), 3 cm "
               f"{ch['recovered_3cm']:.3f}, false {ch['false_per_station_year_offfan']:.3f}; "
               f"hp window {tune['chosen_params']['hp_window']} d "
               f"({tune['cm_evidence']['hp_window_selected_by']}); reference "
               f"{tune['reference']}; full {tune['reference_full']}"]
    for var, v in (("preorigin", vp), ("full", vf)):
        num(f"S3.v2.run.{var}.seconds", v["seconds"], [(f"{_rel(ev)}/validation_{var}.json",
                                                        "input")], PRE if var == "preorigin"
            else FULL)
        num(f"S3.v2.run.{var}.cpu_env", v.get("cpu_env"),
            [(f"{_rel(ev)}/validation_{var}.json", "input")],
            PRE if var == "preorigin" else FULL)
    (ev / "report_numbers_v2.json").write_text(json.dumps(nums, indent=1, ensure_ascii=False,
                                                          default=str))
    print("\n".join(md))
    if "--ledger" in argv:
        p = ledger.append_rows("S3", rows)
        print(f"\nledger: {len(rows)} rows -> {p}", file=sys.stderr)
    return 0


def _rel(p: Path) -> str:
    return Path(p).resolve().relative_to(REPO).as_posix()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
