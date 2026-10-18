"""WP S1b repair round r3 (S1br): numbers, ledger rows, provenance and ``reports/S1b.repair.json``.

    python -I dataprep_staging/s1_rings/s1b/s1b_r3_report.py

Reads only this round's evidence (``S1b_evidence/repair_r3/``), S1b's round-1 and r2 evidence
(re-hash checks), the declaration and the look log. Writes:

* ``repair_r3/numbers.json``: every number quoted in ``S1b.repair.md`` Parts 3-6, recomputed
  from the evidence files, and ``repair_r3/numbers_check.json`` (each quoted string found in
  the report);
* ``repair_r3/PROVENANCE.json``: code hashes now against the declared ones, evidence hashes,
  the declared-input re-check, the round-1 and r2 provenance re-checks, the round's looks;
* ``ledger/rows/S1b.csv``: one row per (number, source) of this round (``S1b.r3.*``);
* ``reports/S1b.repair.json``.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from common import ledger  # noqa: E402
from common.hashing import sha256_file  # noqa: E402
from common.paths import REPO, rel_to_repo  # noqa: E402
from harness import prereg3  # noqa: E402

from s1_rings.s1b import s1b_const as K  # noqa: E402
from s1_rings.s1b import s1b_r2_core as R  # noqa: E402
from s1_rings.s1b import s1b_r3_looks as LK  # noqa: E402
from s1_rings.s1b.s1b_run import dump  # noqa: E402

EV = K.EVIDENCE
R3 = EV / "repair_r3"
REPORTS = K.DATAPREP / "reports"
RINGS3 = "results/dataprep/rings_bridged/v3"
RINGS2 = "results/dataprep/rings_bridged/v2"
GNSS4 = "results/dataprep/gnss_clean/full/v4"
GNSS2 = "results/dataprep/gnss_clean/full/v2"
INSAR3 = "results/dataprep/insar_069A/v3"
INSAR2 = "results/dataprep/insar_069A/v2"
ROWS1 = "results/dataprep/reports/S1b_evidence/registered_rows.csv"
R2DIR = "results/dataprep/reports/S1b_evidence/repair"
R3DIR = "results/dataprep/reports/S1b_evidence/repair_r3"
PV = {RINGS3: "rings_bridged/v3", RINGS2: "rings_bridged/v2", GNSS4: "gnss_clean/full/v4",
      GNSS2: "gnss_clean/full/v2", INSAR3: "insar_069A/v3", INSAR2: "insar_069A/v2",
      ROWS1: "S1b round-1 evidence", R2DIR: "S1b r2 evidence", R3DIR: "S1b r3 evidence"}
READ = ("k:0", "range_ppm:30", "range_ppm:40", "mid_ppm:30", "mid_ppm:40", "k:1", "k:2")
CODE_R3 = ["s1b_r3_looks.py", "s1b_r3_diagnose.py", "s1b_r3_snapshot.py", "s1b_r3_io.py",
           "s1b_r3_run.py", "s1b_r3_synth.py", "s1b_r3_after.py", "s1b_r3_prereg.py",
           "s1b_r3_report.py", "s1b_r3_diag_after.py", "s1b_r3_diag_after2.py",
           "tests/test_s1b_r3.py"]


def load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def r(x, nd=3):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return None
    return round(float(x), nd)


def numbers() -> dict:
    sl = load(R3 / "retest" / "second_look.json")
    af = load(R3 / "retest" / "after.json")
    oc = {m: load(R3 / "synthetic" / f"oc_{m}.json") for m in ("N1_sd_s", "N2_resid")}
    p = sl["primary"]
    out = {"readings_primary": {}, "readings_variants": {}, "d1": {}, "synthetic": {},
           "lr": {}, "d2": {}}
    for rule in R.RULES:
        v = p["verdict_by_reading"][rule]
        out["readings_primary"][rule] = {k: v[k] for k in (
            "c_wra_containment", "c_wra_n", "c_wra_raw_containment", "c_wra_family_pass",
            "d2_containment", "d2_n", "d2_family_pass", "conjunction_c_and_d1")}
    fr = p["families"]["c_wra"]
    out["rmse_c_wra_mm"] = {k: r(x, 1) for k, x in fr["rmse_mm"].items()}
    out["rmse_rule_holds_c_wra"] = fr["rmse_rule_holds"]
    out["k_star_c_wra"] = r(fr["k_star_primary"], 2)
    out["range_half_width_ppm_c_wra"] = {k: r(x, 1) for k, x in
                                         fr["range_half_width_ppm"].items()}
    dr = p["families"]["d_insar"]
    out["rmse_d_insar_mm"] = {k: r(x, 1) for k, x in dr["rmse_mm"].items()}
    out["c_ngl"] = {"n_evaluable": p["verdicts"]["c_ngl"]["n_evaluable"],
                    "contains_primary": p["verdicts"]["c_ngl"]["n_contain_primary"]}
    out["c_neighbour"] = {k: p["verdicts"]["c_neighbour"][k] for k in (
        "n_evaluable", "n_contain_primary", "n_contain_raw", "pass")}
    out["registered_rule_gate_members"] = p["gate_members_registered_rule"]
    out["registered_rule_gate_pass"] = p["gate_pass_registered_rule"]
    for key, src in (("primary", p), ("insar_cells_M3", sl["variants"]["insar_cells_M3"]),
                     ("insar_cells_plane", sl["variants"]["insar_cells_plane"])):
        d = src["d1_stats"]["rates"]
        out["d1"][key] = {"n": d["n"], "r_bridged": r(d["r_bridged"]), "r_raw": r(d["r_raw"]),
                          "r_crit": r(src["d1"]["r_crit"]), "pass": src["d1"]["pass"],
                          "williams_p": r(d["williams"]["p_one_sided"]),
                          "boot_ci90": [r(x) for x in d["bootstrap_diff"]["ci90"]],
                          "fisher90_r_bridged": [r(x) for x in d["fisher90_r_bridged"]],
                          "changes_r": [r(src["d1_stats"]["changes"]["r_bridged"]),
                                        r(src["d1_stats"]["changes"]["r_raw"])]}
    for name, v in {**sl["variants"], **sl["decomposition"]}.items():
        vv = v["verdict_by_reading"]
        out["readings_variants"][name] = {
            rule: [vv[rule]["c_wra_containment"], vv[rule]["c_wra_n"],
                   vv[rule]["c_wra_family_pass"]] for rule in READ}
        out["readings_variants"][name]["d2_k2"] = [vv["k:2"]["d2_containment"],
                                                   vv["k:2"]["d2_n"]]
        out["readings_variants"][name]["d1"] = [r(v["d1"].get("r_bridged")),
                                                r(v["d1"].get("r_raw")), v["d1"].get("pass")]
    for m, o in oc.items():
        out["synthetic"][m] = {}
        for fam in ("wra", "wra_noSJES", "insar"):
            res = o["families"][fam]
            out["synthetic"][m][fam] = {
                rule: {sc: [r(res[f"{rule}|{sc}"]["family_pass"]),
                            r(res[f"{rule}|{sc}"]["mean_count"], 2)]
                       for sc in ("correct", "up25", "down25", "up25A", "down25A", "other",
                                  "raw", "mix", "mix_drawn")} for rule in READ}
            if fam == "insar":
                out["synthetic"][m]["d1"] = {sc: res[f"d1|{sc}"] for sc in
                                             ("correct", "mix", "mix_drawn")}
        out["synthetic"][m]["discriminating"] = {
            fam: {rule: bool(o["families"][fam][f"{rule}|correct"]["family_pass"] >= 0.5
                             and max(o["families"][fam][f"{rule}|up25"]["family_pass"],
                                     o["families"][fam][f"{rule}|down25"]["family_pass"])
                             <= 0.3) for rule in READ} for fam in ("wra", "wra_noSJES")}
    for m in ("N1_sd_s", "N2_resid"):
        out["lr"][m] = {}
        for fam in ("wra", "wra_noSJES"):
            out["lr"][m][fam] = {}
            for rule in READ:
                row = af["lr"][m][fam][rule]
                out["lr"][m][fam][rule] = {
                    "n_obs": row["n_obs"], "n": row["n_sites_real"],
                    "p_eq": {sc: r(row[sc]["p_eq"]) for sc in ("correct", "up25", "down25",
                                                                "up25A", "mix_drawn")},
                    "p_le": {sc: r(row[sc]["p_le"]) for sc in ("correct", "up25", "down25",
                                                                "up25A", "mix_drawn")},
                    "LR_correct_vs_up25": r(row["LR_correct_vs_up25"], 2),
                    "LR_correct_vs_down25": r(row["LR_correct_vs_down25"], 2),
                    "LR_mixdrawn_vs_up25": r(row["LR_mixdrawn_vs_up25"], 2),
                    "LR_mixdrawn_vs_down25": r(row["LR_mixdrawn_vs_down25"], 2)}
    out["per_site_wra"] = [{k: (r(v, 1) if isinstance(v, float) else v) for k, v in o.items()
                            if not isinstance(v, dict)} for o in af["per_site_wra"]]
    da = load(R3 / "retest" / "diag_after.json")
    db = load(R3 / "retest" / "diag_after2.json")
    out["diag_insar_summary"] = {k: (r(v, 2) if isinstance(v, float) else v)
                                 for k, v in da["insar_summary"].items()}
    out["diag_insar_sites"] = [{k: (r(v, 1) if isinstance(v, float) else v)
                                for k, v in o.items()} for o in da["insar"]]
    out["diag_gnss_sites"] = [{k: (r(v, 1) if isinstance(v, float) else v)
                               for k, v in o.items()} for o in da["gnss"]]
    out["diag_insar_minus_gnss"] = {
        k: {"all_10": {kk: r(vv, 1) for kk, vv in v["all_10"].items()},
            "without_kecuo_xizhou": {kk: r(vv, 1) for kk, vv in
                                     v["without_kecuo_xizhou"].items()},
            "per_site": {kk: r(vv, 1) for kk, vv in v["per_site"].items()}}
        for k, v in db["insar_minus_gnss"].items()}
    out["diag_artefact_variant_lr"] = {
        m: {rule: {"n_obs": t[rule]["n_obs"], "n": t[rule]["n_sites_real"],
                   "LR_correct_vs_up25": r(t[rule]["LR_correct_vs_up25"], 2),
                   "LR_correct_vs_down25": r(t[rule]["LR_correct_vs_down25"], 2),
                   "LR_mixdrawn_vs_up25": r(t[rule]["LR_mixdrawn_vs_up25"], 2),
                   "family_pass_correct": r(t[rule]["correct"]["family_pass"]),
                   "family_pass_up25": r(t[rule]["up25"]["family_pass"]),
                   "family_pass_down25": r(t[rule]["down25"]["family_pass"])}
            for rule in READ} for m, t in db["artefact_variant_lr"].items()}
    nb = {x["site_en"]: x for x in sl["primary"]["rows"]["c_neighbour"]}
    out["beichen"] = {"insar_M3_pixels_I_gap_mm": r(next(
        x["I_gap_mm"] for x in p["rows"]["d_insar"] if x["site_en"] == "Beichen"), 1),
        "insar_plane_cells_I_gap_mm": r(next(
            x["I_gap_mm"] for x in sl["variants"]["insar_cells_plane"]["rows"]["d_insar"]
            if x["site_en"] == "Beichen"), 1),
        "gnss_SLES_5p7km_G_mm": r(nb["Beichen"]["G_mm"], 1)}
    for key, src in (("primary", p), ("insar_cells_M3", sl["variants"]["insar_cells_M3"]),
                     ("insar_cells_plane", sl["variants"]["insar_cells_plane"])):
        d = src["d1_stats"]
        out["d1"][key]["without_beichen"] = [r(d["without_beichen"]["r_bridged"]),
                                             r(d["without_beichen"]["r_raw"]),
                                             r(d["without_beichen"]["r_crit"]),
                                             d["without_beichen"]["pass"]]
        out["d1"][key]["pilot_9_sites"] = [r(d["pilot_9_sites"]["r_bridged"]),
                                           r(d["pilot_9_sites"]["r_raw"]),
                                           r(d["pilot_9_sites"]["r_crit"]),
                                           d["pilot_9_sites"]["pass"]]
        vv = src["verdict_by_reading"]
        out["d2"][key] = {rule: [vv[rule]["d2_containment"], vv[rule]["d2_n"],
                                 vv[rule]["d2_family_pass"]] for rule in READ}
    return out


def ledger_rows() -> list[dict]:
    spec = {
        "S1b.r3.diagnosis": [(ROWS1, "input", "round-1 registered rows (bracket ends, sigmas)"),
                             (R2DIR, "input", "r2 readings, synthetic and U1 files")],
        "S1b.r3.synthetic_oc_N1": [(RINGS3, "input", "survey dates, sd_s, L, P(A) (values "
                                    "replaced by synthetic ones)"),
                                   (GNSS4, "input", "dates, flags, station noise, step se"),
                                   (INSAR3, "input", "valid-pixel masks and source codes")],
        "S1b.r3.synthetic_oc_N2": [(RINGS3, "input", "survey dates, sd_s, L, P(A); ring-fit "
                                    "residual sd from the surveys"),
                                   (GNSS4, "input", "dates, flags, station noise, step se"),
                                   (INSAR3, "input", "valid-pixel masks and source codes")],
        "S1b.r3.c_wra_plan_literal_k0": [(GNSS4, "validator", "after_the_fact_analysis; "
                                          "second look"),
                                         (RINGS3, "input", "values under test")],
        "S1b.r3.c_wra_plan_literal_readings": [(GNSS4, "validator", "second look"),
                                               (RINGS3, "input", "values under test")],
        "S1b.r3.c_wra_s1b_2sigma_rule": [(GNSS4, "validator", "second look; S1b's own rule"),
                                         (RINGS3, "input", "values under test")],
        "S1b.r3.c_ngl": [(GNSS4, "validator", "DNAN only (n = 1)"),
                         (RINGS3, "input", "values under test")],
        "S1b.r3.c_neighbour": [(GNSS4, "validator", "report only"),
                               (RINGS3, "input", "values under test")],
        "S1b.r3.d1": [(INSAR3, "validator", "M3 pixels; second look"),
                      (RINGS3, "input", "values under test")],
        "S1b.r3.d1_reference_variants": [(INSAR3, "validator", "flag-0 cells, M3 and plane"),
                                         (RINGS3, "input", "values under test")],
        "S1b.r3.d2_readings": [(INSAR3, "validator", "second look; S1b's own check"),
                               (RINGS3, "input", "values under test")],
        "S1b.r3.variants": [(GNSS4, "validator", "declared report-only variants"),
                            (INSAR3, "validator", "declared report-only variants"),
                            (RINGS3, "input", "values under test")],
        "S1b.r3.decomposition_v2_product": [(RINGS2, "input", "round-1 product values"),
                                            (GNSS4, "validator", "repaired validator"),
                                            (INSAR3, "validator", "repaired validator")],
        "S1b.r3.decomposition_round1_validators": [(RINGS3, "input", "values under test"),
                                                   (GNSS2, "validator", "round-1 validator"),
                                                   (INSAR2, "validator", "round-1 validator")],
        "S1b.r3.lr_real_vs_synthetic": [(R3DIR, "input", "second_look.json and oc_*.json")],
        "S1b.r3.per_site_change": [(R3DIR, "input", "second_look.json"),
                                   (ROWS1, "input", "round-1 registered rows")],
        "S1b.r3.after.insar_minus_gnss_by_reference": [
            (R3DIR, "input", "second_look.json rows (InSAR M3 pixels, M3 and plane cells; GNSS "
             "v4 co-located)")],
        "S1b.r3.after.gnss_bracket_shift_v2_v4": [
            (R3DIR, "input", "second_look.json primary, variant and decomposition rows")],
        "S1b.r3.after.artefact_variant_lr": [
            (R3DIR, "input", "second_look.json variant rows and oc_*.json (after the fact)")],
        "S1b.r3.after.beichen_reference": [
            (R3DIR, "input", "second_look.json InSAR rows and the Beichen-SLES neighbour row")],
    }
    rows = []
    for num, items in spec.items():
        for src, role, note in items:
            rows.append({"number_id": num, "source": src, "role": role,
                         "product_version": PV[src], "note": note})
    return rows


def verify_provenance(path: Path, sections) -> dict:
    p = load(path)
    bad, n = [], 0
    for sec in sections:
        for it in p.get(sec, []):
            n += 1
            f = REPO / it["path"]
            if not f.exists() or sha256_file(f) != it["sha256"]:
                bad.append(it["path"])
    return {"path": rel_to_repo(path), "n_checked": n, "mismatches": bad}


def looks_this_round() -> list[dict]:
    recs = prereg3.read_chained(K.DATAPREP / "preregistration" / "looks.jsonl")
    return [{k: x.get(k) for k in ("time", "purpose", "prereg_id", "units", "data_scope")}
            for x in recs if x["wp"] == K.WP and x["time"] >= "2026-10-17T23:00:00"]


def provenance() -> dict:
    doc, sha = prereg3.load_prereg(LK.DECL_ID)
    declared = {c["path"]: c["sha256"] for c in doc["primary_model"]["code"]}
    code_now = []
    for c in CODE_R3:
        f = HERE / c
        code_now.append({"path": rel_to_repo(f), "sha256": sha256_file(f),
                         "declared_sha256": declared.get(rel_to_repo(f))})
    ev = sorted(p for p in R3.rglob("*") if p.is_file() and p.suffix in (".json", ".csv",
                                                                        ".log", ".md", ".jsonl")
                and "PROVENANCE" not in p.name and "numbers" not in p.name)
    reg1, _ = prereg3.load_prereg(K.PREREG_ID)
    return {"wp": K.WP, "round": "r3 (S1br)", "declaration": LK.DECL_ID,
            "declaration_sha256": sha,
            "declared_inputs_changed": prereg3.changed_inputs(doc),
            "registered_v1_inputs_changed": prereg3.changed_inputs(reg1),
            "code_r3": code_now,
            "code_changed_after_declaration": [c["path"] for c in code_now
                                               if c["declared_sha256"] not in (None,
                                                                               c["sha256"])],
            "code_not_declared": [c["path"] for c in code_now if c["declared_sha256"] is None],
            "evidence": [{"path": rel_to_repo(p), "sha256": sha256_file(p)} for p in ev],
            "round1_provenance_recheck": verify_provenance(EV / "PROVENANCE.json",
                                                           ("code_now", "evidence")),
            "r2_provenance_recheck": verify_provenance(EV / "repair" / "PROVENANCE.json",
                                                       ("code", "evidence")),
            "looks": looks_this_round(),
            "products_read": {"rings_bridged/v3": "after_the_fact_analysis",
                              "gnss_clean/full/v4": "after_the_fact_analysis",
                              "insar_069A/v3": "after_the_fact_analysis",
                              "rings_bridged/v2, gnss_clean/full/v2, insar_069A/v2":
                              "after_the_fact_analysis (decompositions only)"},
            "products_written": 0, "claude_api_tokens": 0, "gpu": "none"}


def check_quotes(nums_quoted: list[tuple[str, str]]) -> dict:
    md = (REPORTS / "S1b.repair.md").read_text(encoding="utf-8")
    res = [{"label": lab, "quoted": q, "found": q in md} for lab, q in nums_quoted]
    return {"n": len(res), "n_found": sum(x["found"] for x in res), "items": res}


def fmt(x, nd):
    return f"{x:.{nd}f}"


def quotes(n: dict) -> list[tuple[str, str]]:
    """(label, string) pairs: every headline number, formatted as quoted in the report."""
    pr, lr, sy = n["readings_primary"], n["lr"], n["synthetic"]
    q = []
    for rule, lab in (("k:0", "k0"), ("range_ppm:30", "range30"), ("range_ppm:40", "range40"),
                      ("mid_ppm:30", "mid30"), ("mid_ppm:40", "mid40"), ("k:1", "k1"),
                      ("k:2", "k2")):
        v = pr[rule]
        q.append((f"c_wra {lab} count", f"{v['c_wra_containment']}/{v['c_wra_n']}"))
        q.append((f"c_wra {lab} raw", f"{v['c_wra_raw_containment']}/{v['c_wra_n']}"))
        for m in ("N1_sd_s", "N2_resid"):
            row = lr[m]["wra"][rule]
            q.append((f"LR {m} {lab}", f"{fmt(row['LR_correct_vs_up25'], 2)} / "
                                       f"{fmt(row['LR_correct_vs_down25'], 2)}"))
            c = sy[m]["wra"][rule]["correct"]
            if not rule.startswith("mid"):
                q.append((f"syn {m} {lab} correct", f"{fmt(c[0], 3)} ({fmt(c[1], 2)})"))
        if rule.startswith("mid"):
            a, b = sy["N1_sd_s"]["wra"][rule]["correct"], sy["N2_resid"]["wra"][rule]["correct"]
            q.append((f"syn N1/N2 {lab} correct", f"{fmt(a[0], 3)} / {fmt(b[0], 3)} "
                                                  f"({fmt(a[1], 2)} / {fmt(b[1], 2)})"))
    q += [("rmse primary", fmt(n["rmse_c_wra_mm"]["primary_vs_mid"], 1) + " mm (primary)"),
          ("rmse raw", fmt(n["rmse_c_wra_mm"]["raw_vs_mid"], 1) + " mm (raw)"),
          ("k*", f"k* = {fmt(n['k_star_c_wra'], 2)}"),
          ("d1 M3 px", f"{fmt(n['d1']['primary']['r_bridged'], 3)} / "
                       f"{fmt(n['d1']['primary']['r_raw'], 3)}"),
          ("d1 M3 cells", f"{fmt(n['d1']['insar_cells_M3']['r_bridged'], 3)} / "
                          f"{fmt(n['d1']['insar_cells_M3']['r_raw'], 3)}"),
          ("d1 plane", f"{fmt(n['d1']['insar_cells_plane']['r_bridged'], 3)} / "
                       f"{fmt(n['d1']['insar_cells_plane']['r_raw'], 3)}"),
          ("d1 williams M3", fmt(n["d1"]["primary"]["williams_p"], 3)),
          ("d1 williams plane", fmt(n["d1"]["insar_cells_plane"]["williams_p"], 3)),
          ("d2 k2 M3", f"{n['d2']['primary']['k:2'][0]}/14"),
          ("d2 k2 plane", f"{n['d2']['insar_cells_plane']['k:2'][0]}/14"),
          ("beichen M3", fmt(n["beichen"]["insar_M3_pixels_I_gap_mm"], 1) + " mm"),
          ("beichen plane", fmt(n["beichen"]["insar_plane_cells_I_gap_mm"], 1)),
          ("beichen SLES", fmt(n["beichen"]["gnss_SLES_5p7km_G_mm"], 1)),
          ("insar-gnss M3 median", "+" + fmt(
              n["diag_insar_minus_gnss"]["M3_pixels"]["without_kecuo_xizhou"]["median"], 1)),
          ("insar-gnss M3 rms", fmt(
              n["diag_insar_minus_gnss"]["M3_pixels"]["without_kecuo_xizhou"]["rms"], 1)),
          ("insar-gnss plane median", fmt(
              n["diag_insar_minus_gnss"]["plane_cells"]["without_kecuo_xizhou"]["median"], 1)),
          ("insar-gnss plane rms", fmt(
              n["diag_insar_minus_gnss"]["plane_cells"]["without_kecuo_xizhou"]["rms"], 1)),
          ("dIgap median", fmt(n["diag_insar_summary"]["median_d_I_gap_mm"], 1) + " mm")]
    return q


SUMMARY = (
    "Repair round r3 of WP S1b (independent GNSS (c) and InSAR (d) checks of the ring bridge), "
    "re-run on the REPAIRED products rings_bridged/v3, gnss_clean/full/v4 and insar_069A/v3 "
    "under the plan-literal readings, with S1b's 2-sigma rule labelled beside them. The "
    "diagnosis (Part 1, hashed 23:24:31Z) and the design (Part 2, 23:34:58Z) came before the "
    "declaration S1b.r3.rings_v3_second_look.v1 (23:36:34Z), which hashes the code and inputs. "
    "Every real unit was scored in round 1, so every real-data number is a SECOND LOOK. "
    "Fresh-seed synthetic operating characteristics (seeds 18031/18032, 400 realizations, ring "
    "noise at sd_s and at the ring-fit residual sd) replace r2's withdrawn discounting argument: "
    "the plan-literal k0 reading (the failed criterion) and mid +-30 ppm fail a correct product "
    "too (0.007-0.045), while range +-30/40 ppm and k = 1 discriminate (correct 0.59-0.87, "
    "+-25 % 0.11-0.25). Second look: k0 5/10 (round 1 3/10) FAIL; range +-30/40 ppm 7/10 FAIL "
    "(outside: Yuanchang's membership mixture, KTES's in-gap step handling that flips sign "
    "between GNSS versions, SJES's data-gap offset; 8/9 in the declared variant with both GNSS "
    "artefacts removed); S1b's own k = 2 rule 9/10. Likelihood ratios correct : +-25 % are weak "
    "and mixed (0.33-2.28 for the range readings). InSAR (d) on v3 depends on the reference: "
    "M3 (the product's) d1 r 0.370 < 0.458 and d2 8/14 FAIL; plane cells 0.828 and 14/14. The "
    "M3 gap changes exceed co-located GNSS by a median +23.0 mm (after the fact), the reference "
    "term S4b documented. GNSS v2 -> v4 moved the brackets by 13-68 mm at 5 of 10 sites, more "
    "than the 12-13 mm a 25 % scale error moves the product. Status stays partial (excluded): "
    "no fresh real unit exists for the 2022 gap, the products are gate-failed, NGL stays at "
    "n = 1, and the governing reading is a user decision. CPU only, 0 API tokens, nothing "
    "committed, no product version.")
DATA_NEEDED = (
    "The WRA record of the 2022 probe/cable change with its scale calibration (plan R-d), the "
    "only route to fresh evidence for this gap; WRA GNSS equipment and monument logs for KTES "
    "(2022-08), SJES (2022-12..2023-02) and GFES (2020) to resolve the step artefacts by rule; an "
    "InSAR reference validated on fresh units across 2021-23 (a fan-spanning ramp reference, or "
    "tropospheric/ionospheric corrections that need accounts); a ring survey noise estimate "
    "independent of the product (repeat readings or probe repeatability); at least 3 continuous "
    "GNSS stations within 5 km of ring sites spanning 2020-25 (not in the public NGL archive).")
ISSUES = [
    "S3 / gnss_clean: KTES has opposite in-gap own-series steps in v2 (-33.2 mm 2022-08-11) and "
    "v4 (+27.3 mm 2022-08-26), consistent with an excursion; SJES's data-gap drop persists in "
    "v4; GFES's v4 2020 steps make the 2020-21 GNSS-minus-ring excess rate -23.8 mm/yr "
    "(implausible); v2 -> v4 moved GNSS-ring brackets by 13-68 mm at 5 of 10 sites",
    "S4b / insar_069A/v3: across the ring gap, M3 gap changes exceed co-located GNSS by a median "
    "+23.0 mm at 8 sites (plane -5.9 mm); Beichen M3 2.4 mm over 2.6 years against plane 66.7 "
    "and GNSS SLES 65.6 mm; after-the-fact users must carry both references",
    "S1a (report only, never fed back): GNSS reads A at Yuanchang, where the v3 mixture lies "
    "18.8 mm below the A member; Kecuo's GNSS evidence depends on the KTES step policy",
    "Plan revision: a revised (c) needs a ring noise term independent of sd_s and step rules "
    "fixed in advance, and fresh units that do not exist for the 2022 gap",
    "USER DECISIONS pending: plan-literal reading vs S1b's rule; U7 for S1's post-gap checks; "
    "S3 and S4b gate decisions; which InSAR reference after-the-fact users read first",
]


def report_json(n: dict, prov: dict, led: dict) -> dict:
    doc, sha = prereg3.load_prereg(LK.DECL_ID)
    snaps = [json.loads(x) for x in (R3 / "snapshots" / "index.jsonl").read_text().splitlines()
             if x.strip()]
    pr = n["readings_primary"]
    acc = [
        {"criterion": "Plan-literal (c), the failed criterion: k0 (unwidened excess-rate range), "
                      "WRA family, rings_bridged/v3 vs gnss_clean/full/v4 (SECOND LOOK)",
         "threshold": ">= 80 % of evaluable sites AND RMSE(primary) < RMSE(raw) (S1b's family "
                      "rule; plan gives no count)",
         "measured": f"{pr['k:0']['c_wra_containment']}/10 (raw {pr['k:0']['c_wra_raw_containment']}"
                     "/10); non-discriminating: a correct product passes at 0.007 (N1) / 0.033 "
                     "(N2)", "passed": False, "units": "sites"},
        {"criterion": "Plan-literal (c) with the plan's tolerance: range +-30 / +-40 ppm (SECOND "
                      "LOOK)", "threshold": ">= 80 % AND RMSE clause",
         "measured": f"{pr['range_ppm:30']['c_wra_containment']}/10 and "
                     f"{pr['range_ppm:40']['c_wra_containment']}/10 (Yuanchang mixture, Kecuo "
                     "and Xizhou GNSS artefacts outside); discriminating (range +-40 both noise "
                     "models); 8/9 in the declared variant with both GNSS artefacts removed",
         "passed": False, "units": "sites"},
        {"criterion": "Other plan-literal readings |C - C_mid| <= 30 / 40 ppm (SECOND LOOK)",
         "threshold": ">= 80 % AND RMSE clause",
         "measured": f"{pr['mid_ppm:30']['c_wra_containment']}/10, "
                     f"{pr['mid_ppm:40']['c_wra_containment']}/10; non-discriminating",
         "passed": False, "units": "sites"},
        {"criterion": "S1b's own registered 2-sigma rule (k = 2; not the plan's), WRA family "
                      "(SECOND LOOK, labelled)", "threshold": ">= 80 % AND RMSE clause",
         "measured": f"{pr['k:2']['c_wra_containment']}/10 (raw "
                     f"{pr['k:2']['c_wra_raw_containment']}/10); RMSE "
                     f"{n['rmse_c_wra_mm']['primary_vs_mid']} vs "
                     f"{n['rmse_c_wra_mm']['raw_vs_mid']} mm; non-discriminating (+-25 % passes "
                     "0.83-0.92); conditional (second look)", "passed": True, "units": "sites"},
        {"criterion": "(d1) plan-literal (d): r(InSAR gap-mean rate, bridged rate), "
                      "insar_069A/v3 M3 pixels (product's reference; SECOND LOOK)",
         "threshold": "r_bridged >= 0.458 (one-sided 5 %, n 14) AND > r_raw",
         "measured": f"r {n['d1']['primary']['r_bridged']} vs raw {n['d1']['primary']['r_raw']}"
                     f"; plane-reference cells {n['d1']['insar_cells_plane']['r_bridged']} vs "
                     f"{n['d1']['insar_cells_plane']['r_raw']} (pass); undetermined by the "
                     "InSAR reference", "passed": False, "units": "Pearson r"},
        {"criterion": "(d2) S1b's own InSAR bracket, k = 2 (SECOND LOOK)",
         "threshold": ">= 80 % AND RMSE clause",
         "measured": f"M3 pixels {n['d2']['primary']['k:2'][0]}/14; plane cells "
                     f"{n['d2']['insar_cells_plane']['k:2'][0]}/14", "passed": False,
         "units": "sites"},
        {"criterion": "Plan-literal S1b conjunction (k0 AND d1) on the repaired products",
         "threshold": "both pass", "measured": "fails (k0 fails under either InSAR reference)",
         "passed": False, "units": "boolean"},
        {"criterion": "S1b's own registered conjunction (k = 2 c_wra AND d2 AND d1) on the "
                      "repaired products", "threshold": "all pass",
         "measured": "fails with the product's M3 reference (d2 8/14, d1 0.370); passes, "
                     "conditional (second look), with the plane reference",
         "passed": False, "units": "boolean"},
        {"criterion": "NGL family within 5 km judgeable", "threshold": ">= 3 evaluable sites",
         "measured": "1 (DNAN, Jiaxing 3.1 km; next PUS2 5.33 km)", "passed": False,
         "units": "sites"},
        {"criterion": "Verifier major (r2): discounting argument replaced by fresh-seed "
                      "operating characteristics, both ring-noise models, every scenario",
         "threshold": "alternative-scenario expectations and noise sensitivity shown; 'not "
                      "evidence against the product' withdrawn",
         "measured": "done: seeds 18031/18032, 400 realizations each; k0 and mid +-30 "
                     "non-discriminating; range +-40 discriminating (correct 0.593 / 0.870, "
                     "+-25 % 0.175-0.253); statement withdrawn (Part 5)",
         "passed": True, "units": "procedure"},
        {"criterion": "Evidence on the scale from the discriminating readings (declared LR)",
         "threshold": "reported (no threshold)",
         "measured": "LR correct : +-25 % 0.33-1.30 (range readings where discriminating), "
                     "2.01-2.69 (k = 1, N1); 0.30-3.59 over all declared 10-site rows: weak, "
                     "mixed", "passed": False, "units": "likelihood ratio"},
        {"criterion": "Repair discipline: diagnosis before change; thresholds unchanged; second "
                      "looks labelled; declaration before computation",
         "threshold": "rules a-d",
         "measured": "Part 1 hashed 23:24:31Z, Part 2 23:34:58Z, declaration 23:36:34Z, "
                     "synthetic 23:36:42/45Z, second look 23:40:08Z; 0 thresholds changed; "
                     "first look record logged after the reads it covers (disclosed)",
         "passed": True, "units": "procedure"},
        {"criterion": "Ledger, provenance, tests",
         "threshold": "0 conflicts; declared inputs and code unchanged; round-1/r2 hashes",
         "measured": f"{led['rows_appended']} rows appended, {led['conflicts']} conflicts over "
                     f"{led['rows_all']}; declared inputs changed "
                     f"{len(prov['declared_inputs_changed'])}; code changed after declaration "
                     f"{len(prov['code_changed_after_declaration'])}; round-1 mismatches "
                     f"{len(prov['round1_provenance_recheck']['mismatches'])}, r2 "
                     f"{len(prov['r2_provenance_recheck']['mismatches'])}; 8 unit tests pass",
         "passed": True, "units": "checks"}]
    return {
        "wp": "S1b", "round": "S1br (repair round r3 of WP S1b)", "status": "partial",
        "date": "2026-10-18", "gate": "none (validation only); nothing opens a use",
        "previous_reports": ["results/dataprep/reports/S1b.md", "results/dataprep/reports/S1b.json",
                             "results/dataprep/reports/S1b_evidence/round1/"],
        "report_path": "results/dataprep/reports/S1b.repair.md",
        "snapshots": snaps,
        "declaration": {"id": LK.DECL_ID, "sha256": sha, "created": doc["created"]},
        "registered_verdict_round1": "PASS under S1b's own 2-sigma rule (round 1, v2 products); "
                                     "the governing reading (plan-literal or S1b's rule) is a "
                                     "user decision (r2 verifier minor 5)",
        "numbers": n, "acceptance": acc, "provenance": "results/dataprep/reports/S1b_evidence/"
                                                      "repair_r3/PROVENANCE.json",
        "evidence": "results/dataprep/reports/S1b_evidence/repair_r3/",
        "products": [], "claude_api_tokens": 0, "gpu": "none", "downloads": "none",
        "code_paths": [f"dataprep_staging/s1_rings/s1b/{c}" for c in CODE_R3],
        "ledger": led, "summary": SUMMARY, "data_needed_if_failed": DATA_NEEDED,
        "issues_for_downstream": ISSUES}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    nums = numbers()
    dump(nums, R3 / "numbers.json")
    if "--numbers" in argv:
        print(json.dumps(nums, ensure_ascii=False, default=str)[:200])
        return 0
    if "--json-only" in argv:          # rewrite the JSON report; keep the first run's ledger counts
        prev = load(REPORTS / "S1b.repair.json")
        dump(report_json(nums, load(R3 / "PROVENANCE.json"), prev["ledger"]),
             REPORTS / "S1b.repair.json")
        return 0
    prov = provenance()
    dump(prov, R3 / "PROVENANCE.json")
    before = len(ledger._read(ledger.fragment_path(K.WP)))
    path = ledger.append_rows(K.WP, ledger_rows())
    allr = ledger.read_all()
    led = {"fragment": rel_to_repo(path), "rows_appended": len(ledger._read(path)) - before,
           "rows_fragment": len(ledger._read(path)), "rows_all": len(allr),
           "conflicts": len(ledger.conflicts(allr))}
    qc = check_quotes(quotes(nums))
    dump(qc, R3 / "numbers_check.json")
    dump(report_json(nums, prov, led), REPORTS / "S1b.repair.json")
    print("quotes", qc["n_found"], "/", qc["n"], [x["label"] for x in qc["items"]
                                                   if not x["found"]])
    print(json.dumps({"ledger": rel_to_repo(path), "rows_all": len(allr),
                      "conflicts": len(ledger.conflicts(allr)),
                      "declared_inputs_changed": prov["declared_inputs_changed"],
                      "code_changed_after_declaration": prov["code_changed_after_declaration"],
                      "round1": prov["round1_provenance_recheck"]["mismatches"],
                      "r2": prov["r2_provenance_recheck"]["mismatches"],
                      "registered_v1_inputs_changed": prov["registered_v1_inputs_changed"],
                      "n_looks": len(prov["looks"])}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
