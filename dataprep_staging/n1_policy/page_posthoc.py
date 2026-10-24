"""The policy page's post-hoc, model-only labels (nothing scored, no observation read).

Four post-hoc checks of 2026-10-24 (``results/dataprep/reports/N1_gaps_2026-10-24/``) read
only model outputs: A2b, the A2 lever-ranking procedure rerun on the page's basis
(science-04); the transfer of the policy response from R0 to the page's model (science-05);
the stability of the page's headline townships (science-10); and the elastic/inelastic split
of the avoided subsidence (science-13). None is a test. This script copies the numbers the
page quotes from those files into one small json, records each input's sha256 and binds the
result to the forward run and response basis the checks used, so ``viewer_app --posthoc``
can refuse it for any other page. It computes nothing new.

Written once; refuses to overwrite.
Usage: DATAPREP_REPO=<repo> python -I page_posthoc.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

GAPS = "results/dataprep/reports/N1_gaps_2026-10-24"
FILES = {
    "a2b_page_payload": f"{GAPS}/science-04/a2b_page_payload.json",
    "a2b_innov": f"{GAPS}/science-04/a2b_innov.json",
    "transfer": f"{GAPS}/science-05/transfer_innov_vs_cwa.json",
    "hotspot": f"{GAPS}/science-10/hotspot_stability.json",
    "elastic_split": f"{GAPS}/science-13/elastic_split.json",
    "elastic_context": f"{GAPS}/science-13/elastic_context.json",
}
FORWARD = "results/twin_forward/datum_gate_innov.npz"
STATUS = "post hoc, model-only, not a test"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _input_sha(doc: dict, path: str) -> str:
    """The sha256 a post-hoc file recorded for one of its inputs."""
    for x in doc["provenance"]["inputs"]:
        if x["path"] == path:
            return x["sha256"]
    raise ValueError(f"{path} is not among the recorded inputs")


def _hotspot(hs: dict, draws: dict, start: int, horizon: int) -> dict:
    out = {}
    for name, x in hs["solved"].items():
        runs, r0 = x["innov_runs"], x["cwa_runs"]
        d = draws[name]
        if d["headline_towns_unperturbed"] != x["page_headline"]:
            raise ValueError(f"{name}: A2b's headline differs from the page's")
        out[name] = {
            "start": start, "horizon_year": horizon, "headline": x["page_headline"],
            "lead_minus_second_cm": x["page_margins"]["lead_minus_second_cm"],
            "runs": {"n": runs["n"], "share_same_leader": runs["share_same_leader"],
                     "share_same_top3_set": runs["share_same_top3_set"],
                     "share_leader_in_top3": runs["share_leader_in_ref_top3"],
                     "leader_counts": runs["leader_counts"]},
            "r0_runs": {"n": r0["n"], "share_same_leader": r0["share_same_leader"],
                        "share_same_top3_set": r0["share_same_top3_set"]},
            "r0_headline": x["cwa_ensemble_headline"],
            "draws": {"n": sum(d["leader_counts"].values()),
                      "share_same_leader": d["share_draws_same_leader"],
                      "share_same_top3_set": d["share_draws_same_top3_set"]},
            "stable": bool(x["proposed_label"]["leader_stable"]),
            "rule": x["proposed_label"]["rule"]}
    return out


def _per_gwh(a2b: dict) -> dict:
    starts = {}
    for y, v in a2b["starts"].items():
        g = v["per_100gwh"]
        starts[y] = {"lo": min(g["value"].values()), "hi": max(g["value"].values()),
                     "order": g["lever_order"], "share_order_kept": g["share_tau_1"],
                     "cm_share_order_kept": v["cm_ranking_detail"]["share_tau_1"],
                     "n_draws": a2b["provenance"]["draws_per_start"]}
    return {"unit": "fan-mean cm avoided per 100 GWh/yr forgone", "starts": starts,
            "ranking_survives": all(v["per_100gwh"]["ranking_survives"]
                                    for v in a2b["starts"].values()),
            "cm_ranking_survives": all(v["cm_ranking_detail"]["ranking_survives"]
                                       for v in a2b["starts"].values())}


def _elastic(split: dict, ctx: dict, start: int, horizon: int) -> dict:
    pols = {}
    for name, p in split["policies"].items():
        by = p["by_year"]
        tot, first = by[str(horizon)]["page_total"]["fan"], by[str(start)]["page_total"]["fan"]
        b = by[str(horizon)]["B_direct_on_mean_heads"]
        pols[name] = {"total_cm": tot, "elastic_cm": b["elastic"]["fan"],
                      "inelastic_cm": b["inelastic"]["fan"],
                      "share_by_first_dec": first / tot if tot else None}
    lift = split["policies"]["cut30"].get("C_lift_after_2029")
    sc = ctx["scenarios"]
    return {"start": start, "horizon_year": horizon, "policies": pols,
            "heads": "ensemble-mean (not each run); mean forcing only",
            "cells_below_preconsolidation_after_2026_01":
                max(v["cells_below_preconsolidation_after_2026_01"] for v in sc.values()),
            "margin_p5_m": min(v["head_margin_above_preconsolidation_m"]["p5"] for v in sc.values()),
            "margin_min_m": min(v["head_margin_above_preconsolidation_m"]["min"]
                                for v in sc.values()),
            "lift": None if lift is None else {
                "policy": "cut30", "lifted": "2030-01", "at_year": horizon,
                "left_cm": lift["avoided_left_dec2032"]["fan"]}}


def _transfer(tr: dict) -> dict:
    e = tr["ensemble"]
    return {"pairs": "[page model, R0]", "horizon": tr["horizon"],
            **{k: {"fan_mean_cm": e[k]["fan_mean_cm"], "cell_pearson_r": e[k]["cell_pearson_r"],
                   "top_decile_cell_jaccard": e[k]["top_decile_cell_jaccard"]}
               for k in ("cut30", "retire_aqua")}}


def build(docs: dict, start: int = 2026, horizon: int = 2032) -> dict:
    """The page block from the post-hoc files (already loaded); checks they share one model."""
    fwd = {_input_sha(docs["transfer"], FORWARD), _input_sha(docs["elastic_context"], FORWARD)}
    if len(fwd) != 1:
        raise ValueError("the post-hoc files used different forward runs")
    basis = {docs["a2b_innov"]["basis_sha256"],
             _input_sha(docs["transfer"], "results/twin_forward/response_basis_innov.npz")}
    if len(basis) != 1:
        raise ValueError("the post-hoc files used different response bases")
    pp = docs["a2b_page_payload"]
    draws = pp["starts"][str(start)]["headline_townships_basis"]
    return {"product": "n1_policy/page_posthoc", "version": "v1", "status": STATUS,
            "forward_sha256": fwd.pop(), "basis_sha256": basis.pop(),
            "page_sha256": pp["provenance"]["page"]["sha256"],
            "hotspot": _hotspot(docs["hotspot"], draws, start, horizon),
            "per_gwh": _per_gwh(pp),
            "elastic": _elastic(docs["elastic_split"], docs["elastic_context"], start, horizon),
            "transfer": _transfer(docs["transfer"])}


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common.paths import REPO, rel_to_repo

    out_p = REPO / GAPS / "page_posthoc.json"
    if out_p.exists():
        raise SystemExit(f"{rel_to_repo(out_p)} exists; written once")
    docs = {k: json.loads((REPO / p).read_text(encoding="utf-8")) for k, p in FILES.items()}
    out = build(docs)
    out["inputs"] = {k: {"path": p, "sha256": sha(REPO / p)} for k, p in FILES.items()}
    out_p.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
