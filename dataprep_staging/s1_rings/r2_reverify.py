"""Repair round (r2) of WP S1a, resumed session: re-verification before the report JSON.

The repair round built rings_bridged/v2 and rewrote S1a.md, but the session ended before
S1a.json was written. This module re-checks the state that the report describes, without
changing any rule, threshold, registered result or product:

* manifests of rings_bridged/v1 and v2: schema validation and re-hash of every input, code
  and output file;
* the three S1a pre-registrations against the sha256 in the registration index;
* the guard probe of v2 for every use (expected: after_the_fact_analysis only);
* the ledger: S1a fragment size and conflicts across all fragments;
* the A1 look log: hash chain intact, S1a records;
* the numbers quoted by the report against the evidence files they come from.

It first logs a look (purpose ``diagnose``): this session re-read the repair-round evidence
summaries (already logged on 2026-10-13 at 08:30-08:41Z) before this record was written.

    tools/cpu_slot.sh python -m s1_rings.r2_reverify        (from dataprep_staging/)
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import guard, ledger, manifest  # noqa: E402
from common.uses import USES  # noqa: E402
from harness import prereg3  # noqa: E402

from s1_rings import s1const as K  # noqa: E402

DP = K.REPO / "results" / "dataprep"
V1 = DP / "rings_bridged" / "v1"
V2 = DP / "rings_bridged" / "v2"
PREREG = DP / "preregistration"
REP = K.EVIDENCE / "repair"
OUT = REP / "reverify_2026-10-14.json"
PREREG_IDS = ("S1a.rings_bridged_rules.v1", "S1a.e_synthetic_mixed.v1",
              "S1a.rings_bridged_rules.v2")


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def log_look() -> dict:
    from s1_rings.seed_looks import site_units
    units = site_units()
    names = sorted(units) + ["S1a:synth.R.mixed_groupcond.v1", "S1a:synth.R.mixed_harness.v1"]
    return prereg3.record_look(
        names, "rings+piles:2014-05..2025-05", "S1a", "diagnose", members=units,
        note=("S1a repair round (r2) resumed in a new session (2026-10-14), logged after the fact "
              "of reading: before this record the session re-read S1a.md, rings_bridged/v2 "
              "MANIFEST.json and validation.json, S1a_evidence/checks_v2.json and repair/"
              "e_decomposition.json, e_bdeep_bias.json, a_power_decomposition.json, "
              "v1_v2_compare.json, ledger_r2.json (all already produced and logged on "
              "2026-10-13, 08:30-08:41Z), to verify the report before writing S1a.json; "
              "re-verification of hashes, guard, ledger and look log follows; no rule, "
              "threshold, registered result or product changed"))


def manifest_check(d: Path) -> dict:
    m, _ = manifest.load_manifest(d)
    return {"product": f"{m['product']}/{m['version']}",
            "validate_manifest": manifest.validate_manifest(m, d),
            "verify_hashes": manifest.verify_hashes(m, d),
            "n_files": len(m.get("inputs", [])) + len(m.get("code", []))
            + len(m.get("outputs", [])),
            "gate_passed": m["gate"]["passed"], "allowed_uses": m["allowed_uses"]}


def prereg_check() -> dict:
    idx = {r["id"]: r for r in prereg3.read_chained(PREREG / "INDEX.jsonl")}
    out = {}
    for i in PREREG_IDS:
        rec = idx.get(i)
        f = PREREG / f"{i}.json"
        out[i] = {"indexed": rec is not None, "created": rec and rec["created"],
                  "sha256_index": rec and rec["sha256"], "sha256_file": sha(f),
                  "match": bool(rec) and rec["sha256"] == sha(f)}
    return out


def guard_probe(d: Path, m_allowed: list[str], gate: bool) -> dict:
    probe = {}
    for use in sorted(USES):
        try:
            p = guard.open_product(d, use)
            t = p.read_table("ring_total_monthly_bridged.csv")
            probe[use] = {"opened": True, "rows": int(len(t)),
                          "flags": sorted(int(x) for x in t["flag"].unique())}
        except Exception as e:  # noqa: BLE001 - the probe records refusals
            probe[use] = {"opened": False, "error": type(e).__name__}
    exp = set(m_allowed) if gate else {"after_the_fact_analysis"}
    return {"opened": sorted(u for u, v in probe.items() if v["opened"]),
            "as_expected": all(v["opened"] == (u in exp) for u, v in probe.items()),
            "after_the_fact_rows": probe.get("after_the_fact_analysis", {}).get("rows")}


def ledger_check() -> dict:
    rows = ledger.read_all()
    own = [r for r in rows if str(r.get("number_id", "")).startswith("S1a.")]
    frag = ledger.fragment_path("S1a")
    with open(frag, encoding="utf-8") as f:
        n_frag = sum(1 for _ in f) - 1
    return {"fragment_rows": n_frag, "rows_all_fragments": len(rows),
            "s1a_number_rows": len(own), "conflicts_all_fragments": len(ledger.conflicts(rows)),
            "superseded_r1_exists": (DP / "ledger" / "superseded" / "S1a.r1.csv").exists()}


def looks_check() -> dict:
    recs = prereg3.read_chained(PREREG / "looks.jsonl")     # raises if the chain is broken
    s1 = [r for r in recs if r.get("wp") == "S1a"]
    return {"chain_intact": True, "n_records_all": len(recs), "n_s1a": len(s1),
            "s1a": [{"time": r["time"], "purpose": r["purpose"], "test_id": r.get("test_id")}
                    for r in s1]}


def number_checks() -> dict:
    """Numbers quoted in S1a.md / S1a.json against the evidence files (rounded as quoted)."""
    e = json.loads((REP / "e_decomposition.json").read_text())
    bd = json.loads((REP / "e_bdeep_bias.json").read_text())
    ap = json.loads((REP / "a_power_decomposition.json").read_text())
    cmp_ = json.loads((REP / "v1_v2_compare.json").read_text())
    va = json.loads((V2 / "validation.json").read_text())
    ck = json.loads((K.EVIDENCE / "checks_v2.json").read_text())
    p, s = e["primary"], e["stress"]
    cf, cfs = p["counterfactuals_after_the_fact"], s["counterfactuals_after_the_fact"]
    ww = p["worst_window"]
    wb = {(r["status"], r["true_group"]): r for r in p["worst_window_by_status"]}
    rel = {r["p_A_bin"]: r["share_truly_A"] for r in p["membership_probability_reliability"]}
    prior_key = [k for k in cf if k.startswith("mixture with prior")][0]
    av = ap["variants_after_the_fact"]
    claims = {
        "e_worst_rmse_cm 2.79": (round(ww["rmse_cm"], 2), 2.79),
        "e_worst_bias_cm -2.05": (round(ww["bias_cm"], 2), -2.05),
        "e_worst_scale_rmse_cm 1.46": (round(ww["rmse_scale_cm"], 2), 1.46),
        "e_worst_scale_bias_cm -0.93": (round(ww["bias_scale_cm"], 2), -0.93),
        "e_worst_membership_rms_cm 2.10": (round(ww["rms_membership_cm"], 2), 2.10),
        "e_worst_membership_bias_cm -1.11": (round(ww["bias_membership_cm"], 2), -1.11),
        "e_worst_mse_terms 2.13/4.42/1.21": ([round(ww[k], 2) for k in
                                             ("mse_scale", "mse_membership", "mse_cross")],
                                            [2.13, 4.42, 1.21]),
        "e_overall_mse_terms 0.95/0.96/0.06": ([round(p["mse_terms_overall"][k], 2) for k in
                                               ("scale", "membership", "cross")],
                                              [0.95, 0.96, 0.06]),
        "e_windows_over 10 / scale-only 2 / of 42": (
            [p["n_windows_over_threshold"][k] for k in ("actual", "scale_component_only", "of")],
            [10, 2, 42]),
        "e_oracle_worst 1.59": (cf["true group known (scale component only)"]
                                ["worst_window_rmse_cm"].__round__(2), 1.59),
        "e_hard_member 3.01 / 10": ([cf["more probable member instead of the mixture"][k]
                                     for k in ("worst_window_rmse_cm", "n_windows_over_1p5cm")],
                                    [3.01, 10]),
        "e_noforce 2.79 / 10": ([round(cf["no near-threshold forcing (evidence group where 2 "
                                          "lines agree)"]["worst_window_rmse_cm"], 2),
                                 cf["no near-threshold forcing (evidence group where 2 lines "
                                    "agree)"]["n_windows_over_1p5cm"]], [2.79, 10]),
        "e_prior_5/7 2.36 / 9": ([round(cf[prior_key]["worst_window_rmse_cm"], 2),
                                  cf[prior_key]["n_windows_over_1p5cm"]], [2.36, 9]),
        "e_stress 3.11 / 36; oracle 2.10 / 4": (
            [round(cfs["stored (registered primary, check)"]["worst_window_rmse_cm"], 2),
             cfs["stored (registered primary, check)"]["n_windows_over_1p5cm"],
             round(cfs["true group known (scale component only)"]["worst_window_rmse_cm"], 2),
             cfs["true group known (scale component only)"]["n_windows_over_1p5cm"]],
            [3.11, 36, 2.1, 4]),
        "e_worst truly-A mixtures 37 forced + 36 unassigned, 782 cm2": (
            [wb[("forced", "A")]["n"], wb[("unassigned", "A")]["n"],
             round(wb[("forced", "A")]["sum_sq_cm2"] + wb[("unassigned", "A")]["sum_sq_cm2"])],
            [37, 36, 782]),
        "e_bdeep votes B 15/101 worst, 88/3891 overall": (
            [p["b_deep_votes_truly_A"]["worst_window"]["b_deep_votes_B"],
             p["b_deep_votes_truly_A"]["worst_window"]["truly_A"],
             p["b_deep_votes_truly_A"]["overall"]["b_deep_votes_B"],
             p["b_deep_votes_truly_A"]["overall"]["truly_A"]], [15, 101, 88, 3891]),
        "e_member_diff median 5.6 (4.1-8.1)": (
            [round(p["member_diff_C_A_minus_C_B_cm"][k], 1) for k in ("median", "p05", "p95")],
            [5.6, 4.1, 8.1]),
        "e_reliability 0.73/0.76/0.99": ([round(rel["[0.2, 0.4)"], 2), round(rel["[0.4, 0.6)"], 2),
                                          round(rel["[0.6, 0.8)"], 2)], [0.73, 0.76, 0.99]),
        "e_window_coverage 42 of 72": ([e["window_coverage"]["n_scored"],
                                        e["window_coverage"]["n_listed"]], [42, 72]),
        "e_bdeep err failing -65/-63, other -16/-20, worst -77/-80": (
            [round(bd["windows_over_threshold"]["A"]["mean_ppm"]),
             round(bd["windows_over_threshold"]["B"]["mean_ppm"]),
             round(bd["windows_within_threshold"]["A"]["mean_ppm"]),
             round(bd["windows_within_threshold"]["B"]["mean_ppm"]),
             round(bd["worst_window"]["A"]["mean_ppm"]), round(bd["worst_window"]["B"]["mean_ppm"])],
            [-65, -63, -16, -20, -77, -80]),
        "e_bdeep corr 0.88 / 0.71": ([round(bd["corr_window_mean_bd_err_vs"]["bias_scale_cm"], 2),
                                      round(bd["corr_window_mean_bd_err_vs"]["bias_membership_cm"],
                                            2)], [0.88, 0.71]),
        "a power 0.72 / 0.962 / 0.999 / 1.000": (
            [round(v["power"], 3) for v in av.values()], [0.72, 0.962, 0.999, 1.0]),
        "a declared Z 0.32 power 0.721": (
            [round(va["a_pooled_standardized_difference"]["pooled_z"], 2),
             round(va["a_pooled_standardized_difference"]["power_25pct"], 3)], [0.32, 0.721]),
        "b 5/5; residual 13/14": ([va["b_pile_implied_scale"]["n_tested"],
                                   va["residual_offset"]["n_within"]], [5, 13]),
        "v1_v2 only declared change, 13 rows, 21/21 files": (
            [cmp_["only_declared_change"], cmp_["total_csv"]["n_rows_changed"],
             cmp_["rings_corrected_files"]["identical"], cmp_["gate_numbers_identical"]],
            [True, 13, 21, True]),
        "v2 flags 0:1953 1:180 3:213 4:117 6:30 8:316": (
            {str(k): int(v) for k, v in ck["flag_counts_primary_total"].items()},
            {"0": 1953, "1": 180, "3": 213, "4": 117, "6": 30, "8": 316}),
        "gate e/a/b/residual false/false/true/true": (
            [va["gate"][k] for k in ("passed", "e", "a", "b", "residual_offset")],
            [False, False, False, True, True]),
    }
    out = {k: {"evidence": v[0], "quoted": v[1], "match": v[0] == v[1]} for k, v in claims.items()}
    out["_prior_counterfactual_rows_kept_stored"] = p["counterfactual_rows_approximated"]
    return out


def ledger_rows() -> list[dict]:
    """Rows of the re-verification number (sources it re-checked; all inputs)."""
    rel = [
        "results/dataprep/rings_bridged/v1", "results/dataprep/rings_bridged/v2",
        "results/dataprep/reports/S1a_evidence/checks_v2.json",
        *(f"results/dataprep/reports/S1a_evidence/repair/{f}" for f in (
            "e_decomposition.json", "e_bdeep_bias.json", "a_power_decomposition.json",
            "v1_v2_compare.json")),
        *(f"results/dataprep/preregistration/{i}.json" for i in PREREG_IDS),
        "results/dataprep/preregistration/looks.jsonl",
    ]
    rows = [{"number_id": "S1a.r2.reverify", "source": s, "role": "input",
             "product_version": "rings_bridged/v2",
             "note": "r2 resumed session 2026-10-14: re-hash, guard probe, prereg sha256, "
                     "look-log chain and the report's numbers re-checked (no value changed)"}
            for s in rel]
    syn = "results/dataprep/reports/S1a_evidence/synthetic/synthetic_rows.parquet"
    pre = "results/dataprep/preregistration/S1a.e_synthetic_mixed.v1.json"
    for s, note in ((syn, "registered (e) rows: forcing-rule statements recomputed (after the "
                          "fact)"), (pre, "registered design")):
        rows.append({"number_id": "S1a.r2.forcing_rule_check", "source": s, "role": "input",
                     "product_version": "rings_bridged/v2", "note": note})
    return rows


def write_ledger() -> dict:
    ledger.append_rows("S1a", ledger_rows())
    allrows = ledger.read_all()
    return ledger_check() | {"conflicts_all_fragments": len(ledger.conflicts(allrows))}


FORCING_OUT = REP / "forcing_rule_check.json"


def log_forcing_look() -> dict:
    return prereg3.record_look(
        ["S1a:synth.R.mixed_groupcond.v1", "S1a:synth.R.mixed_harness.v1"],
        "rings+piles:2014-05..2021-12", "S1a", "diagnose",
        test_id="S1a.e.mixed_groupcond", prereg_id="S1a.e_synthetic_mixed.v1",
        note=("S1a repair round (r2), resumed session: the report's statements on the "
              "near-threshold forcing rule (769 forced realizations, 354 with two agreeing lines, "
              "all with P(A) < 0.01 or > 0.99, none wrong, at most 0.07 cm change) were not in a "
              "persisted evidence file; recomputed from the registered rows "
              "(synthetic_rows.parquet) and written to repair/forcing_rule_check.json; after "
              "the fact, no rule or threshold changed"))


def forcing_rule_check() -> dict:
    """The near-threshold forcing rule in the registered (e) rows (after the fact)."""
    import numpy as np
    import pandas as pd

    from s1_rings import r2_diagnose as D
    R = pd.read_parquet(D.SYN / "synthetic_rows.parquet")
    out = {"source": "S1a_evidence/synthetic/synthetic_rows.parquet (registered "
                     "S1a.e_synthetic_mixed.v1 rows)",
           "status": "after the fact; the registered result stands"}
    for variant in ("primary", "stress"):
        x = R[R.variant == variant].copy().reset_index(drop=True)
        x["assigned"] = x["assigned"].fillna("")
        evg = x["evidence_group"].fillna("").to_numpy(object)
        forced = x["forced"].to_numpy(bool)
        mixed = (x["assigned"] == "").to_numpy()
        p = x["p_A"].to_numpy(float)
        two = forced & (evg != "")
        w0 = D.stored_weight(x)
        w_nf = np.where(mixed & two & (evg == "A"), 1.0,
                        np.where(mixed & two & (evg == "B"), 0.0, w0))
        cf = D.counterfactual(x, w_nf)
        err = x["err_cm"].to_numpy(float)
        x["cf_noforce"] = cf
        t0 = D.window_table(x, "err_cm").set_index("window")
        t1 = D.window_table(x, "cf_noforce").set_index("window")
        out[variant] = {
            "n_rows": int(len(x)), "n_forced": int(forced.sum()),
            "n_forced_two_agreeing_lines": int(two.sum()),
            "n_forced_two_lines_pA_extreme_0.01": int((two & ((p < 0.01) | (p > 0.99))).sum()),
            "n_forced_two_lines_evidence_wrong": int((two & (evg != x["true_group"]
                                                              .to_numpy(object))).sum()),
            "max_abs_row_change_cm": round(float(np.nanmax(np.abs(cf - err))), 4),
            "max_abs_window_rmse_change_cm": round(float(
                (t1.rmse_cm - t0.rmse_cm).abs().max()), 4),
            "n_windows_over_1p5cm": [int((t0.rmse_cm > D.THR_CM).sum()),
                                     int((t1.rmse_cm > D.THR_CM).sum())],
            "rows_kept_stored_error": D.n_fallback(x, w_nf)}
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-look", action="store_true", help="skip logging (unit tests only)")
    ap.add_argument("--ledger-only", action="store_true",
                    help="append the re-verification rows to the S1a ledger fragment and stop")
    ap.add_argument("--forcing", action="store_true",
                    help="log a look, then recompute the forcing-rule statements from the "
                         "registered rows (repair/forcing_rule_check.json) and stop")
    args = ap.parse_args(argv)
    if args.forcing:
        look = None if args.no_look else log_forcing_look()
        res = forcing_rule_check()
        res["look"] = look and {"time": look["time"], "purpose": look["purpose"]}
        FORCING_OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))
        print(json.dumps(res, indent=1, ensure_ascii=False, default=str))
        return
    if args.ledger_only:
        res = json.loads(OUT.read_text())
        res["ledger_after_append"] = write_ledger()
        OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))
        print(json.dumps(res["ledger_after_append"], indent=1))
        return
    look = None if args.no_look else log_look()
    res = {"wp": "S1a", "round": "r2 (repair), resumed session",
           "time": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "status": "re-verification only; no rule, threshold, registered result or product "
                     "changed",
           "look": look and {"time": look["time"], "purpose": look["purpose"]}}
    res["manifests"] = {v.name: manifest_check(v) for v in (V1, V2)}
    m2 = res["manifests"]["v2"]
    res["preregistrations"] = prereg_check()
    res["guard_probe_v2"] = guard_probe(V2, m2["allowed_uses"], m2["gate_passed"])
    res["ledger"] = ledger_check()
    res["looks"] = looks_check()
    res["numbers"] = number_checks()
    res["ok"] = bool(
        all(not x["validate_manifest"] and not x["verify_hashes"]
            for x in res["manifests"].values())
        and all(x["match"] for x in res["preregistrations"].values())
        and res["guard_probe_v2"]["as_expected"]
        and res["ledger"]["conflicts_all_fragments"] == 0
        and all(v["match"] for k, v in res["numbers"].items() if not k.startswith("_")))
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))
    brief = {k: res[k] for k in ("ok", "guard_probe_v2", "ledger")}
    brief["manifests"] = {k: {"validate": v["validate_manifest"], "hash_failures":
                              len(v["verify_hashes"]), "n_files": v["n_files"]}
                          for k, v in res["manifests"].items()}
    brief["prereg_match"] = {k: v["match"] for k, v in res["preregistrations"].items()}
    brief["numbers_mismatch"] = {k: v for k, v in res["numbers"].items()
                                 if not k.startswith("_") and not v["match"]}
    brief["n_s1a_looks"] = res["looks"]["n_s1a"]
    print(json.dumps(brief, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
