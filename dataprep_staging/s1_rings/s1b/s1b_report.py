"""WP S1b: ledger rows, provenance record and the machine-readable report ``S1b.json``.

    python -I dataprep_staging/s1_rings/s1b/s1b_report.py

Reads only S1b's own evidence files (registered.json, registered_rows.csv,
after_the_fact.json, diagnostics.json, synthetic_operating_characteristics_v2.json) and the
pre-registration; re-verifies the registered inputs' hashes; appends this WP's ledger rows;
writes ``S1b_evidence/PROVENANCE.json`` and ``reports/S1b.json``. The narrative report
``reports/S1b.md`` is written by hand from the same files.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import ledger  # noqa: E402
from common.hashing import sha256_file  # noqa: E402
from common.paths import rel_to_repo  # noqa: E402
from harness import prereg3  # noqa: E402

from s1_rings.s1b import s1b_const as K  # noqa: E402

EV = K.EVIDENCE
REPORTS = K.DATAPREP / "reports"
CODE = ["__init__.py", "s1b_const.py", "s1b_core.py", "s1b_io.py", "s1b_run.py", "s1b_synth.py",
        "s1b_looks.py", "s1b_prereg.py", "s1b_diag.py", "s1b_report.py",
        "tests/test_s1b_core.py"]
EVIDENCE_FILES = ["synthetic_operating_characteristics.json",
                  "synthetic_operating_characteristics_v2.json", "registered.json",
                  "registered_rows.csv", "after_the_fact.json", "diagnostics.json"]

GNSS = "results/dataprep/gnss_clean/full/v2"
INSAR = "results/dataprep/insar_069A/v2"
RINGS = "results/dataprep/rings_bridged/v2"
COORDS = "chou-shui-data/data/mlcw_stations.csv"
GRID = "results/twin_forward/datum_gate_innov.npz"
PILOT = "results/dataprep/_pilots/subsidence_gap_sources"
S1A_MD = "results/dataprep/reports/S1a.md"
S3_MD = "results/dataprep/reports/S3.md"
S4B_MD = "results/dataprep/reports/S4b.md"

SUMMARY = (
    "All four registered checks of S1b.ring_bridge_checks.v1 were evaluated and the S1b "
    "verdict is PASS: (c) WRA GNSS co-located 9/10 sites contain the bridged gap compaction in "
    "the 2-sigma excess-rate bracket (raw 5/10; RMSE 33.7 vs 58.3 mm), (d2) InSAR 14/14 (raw "
    "13/14; RMSE 18.4 vs 41.3 mm), (d1) r 0.714 vs raw 0.318 (critical 0.458). Status partial: "
    "the product and both validators are gate-failed and read as after_the_fact_analysis, the "
    "check is design-exposed, d2's containment does not discriminate bridged from raw, and no "
    "part detects a 25 % scale error (synthetic power 0.06-0.12).")
DEVIATIONS = [
    "Validators gate-failed (S3, S4b): read as after_the_fact_analysis (fallback); results are "
    "after-the-fact evidence, not a validator gate",
    "(c) bracket made operational as the pre/post excess-rate bracket widened by 2 sigma; the "
    "80 % and RMSE-vs-raw rules are S1b's (the plan gives no number)",
    "NGL family has one pair within 5 km (Jiaxing-DNAN): reported, not judged",
    "(d) needs no InSAR tie: own change and rates; insar_069A/v2 is stable-area referenced",
    "Units declared by site id; a cell-based dry run found no blocking look",
    "Resumed session: code re-checked, GNSS step-se term and family-level operating "
    "characteristics added, synthetic re-run (logged) before registration",
    "No product version: validation only; evidence and PROVENANCE.json under S1b_evidence/",
]
ISSUES = [
    "S3: SJES carries an uncorrected ~-100 mm offset inside the 2022-12-22..2023-02-24 data gap "
    "(reverses the corrected +94.9 mm 2022-03-29 step); levels across it are wrong by ~10 cm",
    "S3: InSAR's Kecuo gap change agrees with the GNSS series that keeps the in-gap KTES step "
    "(2022-08-11, -33.2 mm); inconclusive (inside an InSAR network bridge)",
    "S4b: post-gap InSAR rates within segments exceed GNSS by a median 27.6 mm/yr (7.8 without "
    "segment intercepts; -22.3 in 2020-21); gap changes agree with GNSS within 6.4 mm RMS at 8 "
    "sites",
    "S1a (report only, never fed back): GNSS supports every evidence-based assignment and reads "
    "A at Yuanchang, neither at Hunan (C_A 0.3 mm outside), both at Kecuo (A with the KTES step "
    "kept); a new S1a line needs a gate-passing GNSS/InSAR product and a new registration",
    "S1b cannot detect a 25 % scale error; the scale half of S1a's (e) stays untested",
    "User decisions: U7 for S1's inherently post-gap checks; S3/S4b gates",
    "Port after the freeze with S1; registered code files must stay unchanged",
]


def _f(x, nd=1):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(
        float(x), nd)


def ledger_rows() -> list[dict]:
    pv = {GNSS: "gnss_clean/full/v2", INSAR: "insar_069A/v2", RINGS: "rings_bridged/v2",
          COORDS: "L0", GRID: "twin grid (datum_gate_innov)", PILOT: "pilot 2026-10-12",
          S1A_MD: "S1a report 2026-10-14", S3_MD: "S3 report (r3 repair)",
          S4B_MD: "S4b report (repair)"}
    spec = {
        "S1b.c_wra_colocated": [(GNSS, "validator", "WRA daily heights at the 10 co-located "
                                 "stations (after_the_fact_analysis; S3 gate failed)"),
                                (RINGS, "input", "values under test: C_primary, C_A, C_B, sd_s, "
                                 "survey depths (after_the_fact_analysis)"),
                                (COORDS, "input", "ring-site coordinates (pairing)"),
                                (PILOT, "design_time_evidence", "gnss_vs_ring*.csv read before "
                                 "the design"),
                                (S1A_MD, "design_time_evidence", "S1a's GNSS sensitivity table "
                                 "read before the design"),
                                (S3_MD, "design_time_evidence", "A3 excess rates / A2 read in "
                                 "the brief before the design")],
        "S1b.c_ngl_near": [(GNSS, "validator", "NGL DNAN daily heights"),
                           (RINGS, "input", "values under test"),
                           (COORDS, "input", "pairing by distance")],
        "S1b.d2_insar_bracket": [(INSAR, "validator", "multilooked vertical-equivalent pixels "
                                  "at the 14 sites (after_the_fact_analysis; S4b gate failed)"),
                                 (RINGS, "input", "values under test"),
                                 (COORDS, "input", "site coordinates (nearest pixel)"),
                                 (PILOT, "design_time_evidence", "insar_vs_ring_gap.csv read "
                                  "before the design"),
                                 (S4B_MD, "design_time_evidence", "S4b's InSAR-GNSS results "
                                  "read in the brief before the design")],
        "S1b.d1_insar_correlation": [(INSAR, "validator", "InSAR gap changes at 14 sites"),
                                     (RINGS, "input", "values under test"),
                                     (COORDS, "input", "site coordinates"),
                                     (PILOT, "design_time_evidence", "pilot r 0.87 / 0.10 at 9 "
                                      "sites read before the design")],
        "S1b.verdict": [(GNSS, "validator", "c"), (INSAR, "validator", "d"),
                        (RINGS, "input", "product checked"), (COORDS, "input", "pairing"),
                        (PILOT, "design_time_evidence", "pilot tables"),
                        (S1A_MD, "design_time_evidence", "S1a GNSS sensitivity")],
        "S1b.c_neighbour_report_only": [(GNSS, "validator", "nearest station within 6 km"),
                                        (RINGS, "input", "values under test"),
                                        (COORDS, "input", "pairing")],
        "S1b.membership_evidence_report_only": [
            (GNSS, "validator", "bracket containment of C_A / C_B"),
            (INSAR, "validator", "bracket containment of C_A / C_B"),
            (RINGS, "input", "members C_A, C_B and P(A); never written")],
        "S1b.variants_report_only": [(GNSS, "validator", "variants"),
                                     (INSAR, "validator", "variants"),
                                     (RINGS, "input", "values under test"),
                                     (GRID, "input", "twin cells for the InSAR cell variant"),
                                     (COORDS, "input", "pairing")],
        "S1b.synthetic_operating_characteristics": [
            (RINGS, "input", "survey dates, sd_s, L, P(A) only (values replaced)"),
            (GNSS, "input", "dates, flags, sigma_hf, rho_hf, step se only"),
            (INSAR, "input", "valid-pixel masks and network segments only"),
            (COORDS, "input", "pairing")],
        "S1b.diag.sjes_unresolved_offset": [(GNSS, "input", "SJES raw and corrected daily "
                                             "heights, steps.csv")],
        "S1b.diag.gnss_insar_gap_change_agreement": [
            (GNSS, "input", "gap change G at 10 co-located stations"),
            (INSAR, "input", "gap change G at the same sites")],
        "S1b.diag.insar_minus_gnss_rates": [(GNSS, "input", "window rates"),
                                            (INSAR, "input", "window rates, with and without "
                                             "segment intercepts")],
    }
    rows = []
    for num, items in spec.items():
        for src, role, note in items:
            rows.append({"number_id": num, "source": src, "role": role,
                         "product_version": pv[src], "note": note})
    return rows


def provenance(prereg_doc: dict, prereg_sha: str, bad_inputs) -> dict:
    looks = []
    for lk in prereg3.read_chained(K.DATAPREP / "preregistration" / "looks.jsonl"):
        if lk.get("wp") == K.WP:
            looks.append({k: lk.get(k) for k in ("time", "purpose", "test_id", "prereg_id",
                                                 "units", "data_scope")})
    return {
        "wp": K.WP, "prereg": K.PREREG_ID, "prereg_sha256": prereg_sha,
        "registered_inputs": prereg_doc["inputs"],
        "registered_inputs_changed_since_registration": bad_inputs,
        "code_now": [{"path": rel_to_repo(HERE / c), "sha256": sha256_file(HERE / c)}
                     for c in CODE],
        "evidence": [{"path": rel_to_repo(EV / f), "sha256": sha256_file(EV / f)}
                     for f in EVIDENCE_FILES],
        "looks": looks,
        "products_read": {"rings_bridged/v2": "after_the_fact_analysis",
                          "gnss_clean/full/v2": "after_the_fact_analysis",
                          "insar_069A/v2": "after_the_fact_analysis"},
        "products_written": [], "claude_api_tokens": 0, "gpu": "none",
        "note": "S1b is validation only: no product version is created and no file of S1a or "
                "of any product is modified."}


def acceptance(reg: dict) -> list[dict]:
    v = reg["verdicts"]

    def fam(k, label, judged=True):
        x = v[k]
        thr = (">= 80 % of evaluable sites contain C_primary in the 2-sigma excess-rate "
               "bracket AND RMSE(primary) < RMSE(raw) vs bracket mid-point; judged only with "
               ">= 3 evaluable sites")
        meas = (f"{x['n_contain_primary']}/{x['n_evaluable']} contain (need "
                f"{x['n_needed']}); raw contained {x['n_contain_raw']}/{x['n_evaluable']}; "
                f"RMSE primary {x['rmse_primary_mm']:.1f} mm vs raw {x['rmse_raw_mm']:.1f} mm"
                + (f"; failing: {', '.join(K.SITE_EN[s] for s in x['sites_failing'])}"
                   if x["sites_failing"] else ""))
        return {"criterion": label, "threshold": thr, "measured": meas
                + ("" if judged else " (n < 3: reported, not judged)"),
                "passed": bool(x["pass"]) if judged else None}

    d1 = reg["d1"]
    rows = [
        fam("c_wra", "(c) GNSS, WRA co-located family (10 sites) [S1b.c_wra_colocated]"),
        fam("c_ngl", "(c) GNSS, NGL family within 5 km (Jiaxing-DNAN) [S1b.c_ngl_near]",
            judged=False),
        fam("d_insar", "(d2) InSAR bracket, all 14 sites [S1b.d2_insar_bracket]"),
        {"criterion": "(d1) InSAR gap-mean subsidence rate vs bridged gap-mean compaction "
                      "rate across sites [S1b.d1_insar_correlation]",
         "threshold": "r_bridged >= one-sided 5 % critical r (0.458 for n = 14) AND r_bridged "
                      "> r_raw",
         "measured": f"n {d1['n']}; r_bridged {d1['r_bridged']:.3f}; r_raw {d1['r_raw']:.3f}; "
                     f"r_crit {d1['r_crit']:.3f}",
         "passed": bool(d1["pass"])},
        {"criterion": "S1b verdict (registered conjunction of the judged checks)",
         "threshold": "c_wra AND d2 AND d1 (c_ngl not judged: n = 1)",
         "measured": "c_wra pass, d2 pass, d1 pass",
         "passed": bool(reg["gate_pass"])},
    ]
    return rows


def main() -> int:
    doc, sha = prereg3.load_prereg(K.PREREG_ID)
    bad = prereg3.changed_inputs(doc)
    reg = json.loads((EV / "registered.json").read_text())
    aft = json.loads((EV / "after_the_fact.json").read_text())
    diag = json.loads((EV / "diagnostics.json").read_text())
    syn = json.loads((EV / "synthetic_operating_characteristics_v2.json").read_text())
    ledger.append_rows(K.WP, ledger_rows())
    prov = provenance(doc, sha, bad)
    (EV / "PROVENANCE.json").write_text(json.dumps(prov, indent=1, ensure_ascii=False))

    # report-only summaries
    var = {n: {"gate_pass": x["gate_pass"],
               **{k: {kk: x["verdicts"][k][kk] for kk in ("n_evaluable", "n_contain_primary",
                                                          "n_contain_raw", "pass")}
                  for k in ("c_wra", "c_neighbour", "d_insar")},
               "d1": {kk: _f(x["d1"][kk], 3) for kk in ("r_bridged", "r_raw")}}
           for n, x in aft["variants"].items()}
    rows = pd.read_csv(EV / "registered_rows.csv")
    g = rows[rows.test == "c_wra"].set_index("site")
    i = rows[rows.test == "d_insar"].set_index("site")
    gi = [(s, float(i.loc[s, "G_mm"]) - float(g.loc[s, "G_mm"])) for s in g.index]
    keep = [d for s, d in gi if s not in ("溪州國小", "客厝國小")]
    agree = {"n": len(keep), "rms_mm": _f(float(np.sqrt(np.mean(np.square(keep)))), 1),
             "median_mm": _f(float(np.median(keep)), 1),
             "per_site_insar_minus_gnss_mm": {K.SITE_EN[s]: _f(d, 1) for s, d in gi}}
    rt = pd.DataFrame(diag["rates_colocated_mm_yr"])
    rates = {c: _f(float(rt[c].median()), 1) for c in ("insar_minus_gnss_pre",
                                                         "insar_minus_gnss_post_seg",
                                                         "insar_minus_gnss_post_noseg")}
    mem = {}
    for t in ("c_wra", "c_ngl", "d_insar"):
        for _, r in rows[(rows.test == t) & rows.evaluable.astype(bool)].iterrows():
            mem.setdefault(r["site_en"], {})[t] = r["membership_evidence"]
    nos = {r["site_en"]: r["membership_evidence"] for r in
           aft["variants"]["insar_no_segment_intercepts"]["rows"]["d_insar"]
           if r.get("evaluable")}
    cat = {r["site_en"]: r["membership_evidence"] for r in
           aft["variants"]["gnss_catalogue_steps_only_in_gap"]["rows"]["c_wra"]
           if r.get("evaluable")}
    for s in mem:
        mem[s]["d_insar_no_segment_intercepts"] = nos.get(s)
        if s in cat:
            mem[s]["c_wra_catalogue_steps_only"] = cat[s]
    hw = {"gnss_wra_mm": {K.SITE_EN[s]: _f((g.loc[s, "hi_mm"] - g.loc[s, "lo_mm"]) / 2, 1)
                          for s in g.index},
          "insar_mm": {K.SITE_EN[s]: _f((i.loc[s, "hi_mm"] - i.loc[s, "lo_mm"]) / 2, 1)
                       for s in i.index}}
    out = {
        "wp": K.WP, "status": "partial", "date": "2026-10-14", "summary": SUMMARY,
        "deviations_from_plan": DEVIATIONS, "issues_for_downstream": ISSUES,
        "prereg": {"id": K.PREREG_ID, "sha256": sha, "created": doc["created"],
                   "registered_inputs_changed": bad},
        "acceptance": acceptance(reg),
        "report_only": {
            "c_neighbour": reg["verdicts"]["c_neighbour"],
            "variants": var, "d1_variants": {k: {kk: _f(v[kk], 3) for kk in
                                                 ("n", "r_bridged", "r_raw", "r_crit")}
                                             | {"pass": v["pass"]}
                                             for k, v in aft["d1_variants"].items()},
            "d1_spearman": aft["d1_spearman"],
            "membership_evidence": mem,
            "bracket_half_widths": hw,
            "gnss_insar_gap_change_agreement_excl_Xizhou_Kecuo": agree,
            "insar_minus_gnss_rate_median_mm_yr": rates,
            "sjes": {"gap_change_mm": diag["sjes"]["gap_change_mm"],
                     "steps": diag["sjes"]["steps"]},
            "synthetic_families": syn["families"]},
        "resources": {"claude_api_tokens": 0, "gpu": "none",
                      "cpu": "cpu_slot.sh, 1 slot per job (synthetic about 2 min, registered "
                             "and variants about 1 min each)"},
        "report_md": "results/dataprep/reports/S1b.md",
        "provenance": rel_to_repo(EV / "PROVENANCE.json"),
    }
    (REPORTS / "S1b.json").write_text(json.dumps(out, indent=1, ensure_ascii=False,
                                                 default=str))
    print(json.dumps({"bad_inputs": bad, "agree": agree, "rates": rates,
                      "membership": mem}, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
