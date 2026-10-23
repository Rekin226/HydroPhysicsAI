"""Hot-spot and headline-township stability for the decision page (model outputs only).

The page's headline names "Largest benefit in Yunlin: {tt}", the top three eligible
townships (``topTowns``: not pooled, not low-confidence, positive) by the township mean of
the avoided forward subsidence at the horizon. No label says how stable that order is.
This script measures it three ways, all on saved model output:

1. across the page's own ensemble runs (``datum_gate_innov.members.npz``, the 36
   coupled_leveling runs = 18 parameter sets x 2 initial fields) for the two solved
   policies the headline uses (cut30, retire_aqua from 2026);
2. across the R0 sibling's ensemble (``datum_gate_cwa.members.npz``), i.e. the transfer;
3. under the A2 zone x county perturbation (seed 20261223, 500 draws), on the page's
   rescaled basis rows, for the six "retire this class" levers (the ranking panel's rows)
   and for the two solved policies (A2b, science-04).

Also the top-decile cell overlap per run against the ensemble-mean field. No observation
is read; nothing is scored or registered.
Usage: python -I dataprep_staging/n1_gaps/hotspots.py   (CPU, about a minute)
Output: results/dataprep/reports/N1_gaps_2026-10-24/science-10/hotspot_stability.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from n1_gaps import gaps_lib as G  # noqa: E402

OUT = G.OUT_ROOT / "science-10"
A2B = G.OUT_ROOT / "science-04"
SEED, N_DRAWS = 20261223, 500
SOLVED = ("cut30", "retire_aqua")
STABLE = 0.9                     # proposed label threshold: >= 90 % of runs / draws


def member_fields(tag: str) -> dict:
    """Per-run avoided forward subsidence at the horizon (cm), coupled_leveling runs."""
    p = G.FWD / f"datum_gate_{tag}.members.npz"
    z = np.load(p, allow_pickle=False)
    years = [int(y) for y in z["years"]]
    y0, yH = years.index(2022), len(years) - 1
    sel = np.array([str(r) == "coupled_leveling" for r in z["rheology"]])
    S = z["subs_members_yr"][:, sel].astype("float64") * float(z["subs_scale_m"]) * 100.0
    names = [str(n) for n in z["scenario_names"]]
    fwd = S[..., yH] - S[..., y0]                          # (scen, runs, A)
    b = names.index("baseline")
    runs = [f"{m}/ic{i}" for m, i in zip(z["member"][sel], z["ic"][sel], strict=True)]
    return {"path": p, "runs": runs, "horizon": years[yH],
            **{s: -(fwd[names.index(s)] - fwd[b]) for s in SOLVED}}


def stability(fields: np.ndarray, ref_lead: list[int], ref_field: np.ndarray, town, nT,
              names, elig) -> dict:
    leads = [G.top_towns(G.town_means(f, town, nT), elig) for f in fields]
    ref_top = G.top_decile(ref_field)
    jac = [G.jaccard(ref_top, G.top_decile(f)) for f in fields]
    return {"n": len(fields),
            "share_same_leader": float(np.mean([lt[:1] == ref_lead[:1] for lt in leads])),
            "share_same_top3_set": float(np.mean([set(lt) == set(ref_lead) for lt in leads])),
            "share_leader_in_ref_top3": float(np.mean([bool(lt) and lt[0] in ref_lead
                                                       for lt in leads])),
            "leader_counts": pd.Series([names[lt[0]] for lt in leads if lt])
            .value_counts().to_dict(),
            "top_decile_cell_jaccard_vs_ref_p5_p50": [float(np.quantile(jac, 0.05)),
                                                      float(np.median(jac))]}


def margins(tm: np.ndarray, elig: np.ndarray, names) -> dict:
    v = np.where(elig & np.isfinite(tm), tm, -np.inf)
    o = np.argsort(-v)[:5]
    return {"top5": [[names[i], round(float(tm[i]), 4)] for i in o],
            "lead_minus_second_cm": float(tm[o[0]] - tm[o[1]]),
            "third_minus_fourth_cm": float(tm[o[2]] - tm[o[3]])}


def main() -> int:
    os.chdir(G.REPO)
    D = G.load_payload()
    town, names, elig = G.townships(D)
    nT = len(names)
    M = D["meta"]
    yr, yH = M.get("yRef", M["yObs"]), len(M["years"]) - 1
    res = {"what": "stability of the page's headline townships ('Largest benefit in Yunlin') "
                   "and top-decile hot spots; horizon Dec 2032, forward from Dec 2022",
           "eligible_towns": [names[i] for i in range(nT) if elig[i]],
           "excluded_low_confidence_or_pooled": [names[i] for i in range(nT) if not elig[i]],
           "solved": {}, "basis_levers": {}}
    mem = {k: member_fields(k) for k in ("innov", "cwa")}
    for j, s in enumerate(SOLVED):
        page = G.payload_array(D, f"solvedSubs{j}")          # (A, Y) cm, policy - baseline
        page_av = -(page[:, yH] - page[:, yr])
        tm = G.town_means(page_av, town, nT)
        lead = G.top_towns(tm, elig)
        blk = {"page_headline": [names[i] for i in lead], "page_margins": margins(tm, elig, names)}
        for k, m in mem.items():
            ens = m[s].mean(axis=0)
            blk[f"{k}_runs"] = stability(m[s], lead, page_av, town, nT, names, elig)
            blk[f"{k}_ensemble_headline"] = [names[i] for i in
                                             G.top_towns(G.town_means(ens, town, nT), elig)]
            if k == "cwa":
                blk["cwa_ensemble_margins"] = margins(G.town_means(ens, town, nT), elig, names)
        a2b = json.loads((A2B / "a2b_page_payload.json").read_text())
        hb = a2b["starts"]["2026"]["headline_townships_basis"][s]
        blk["a2_perturbation_page_basis"] = {k: hb[k] for k in
                                             ("headline_towns_unperturbed",
                                              "share_draws_same_leader",
                                              "share_draws_same_top3_set", "leader_counts")}
        shares = [blk["innov_runs"]["share_same_leader"],
                  hb["share_draws_same_leader"],
                  1.0 if blk["cwa_ensemble_headline"][:1] == blk["page_headline"][:1] else 0.0]
        blk["proposed_label"] = {
            "leader_stable": bool(min(shares) >= STABLE),
            "top3_set_stable": bool(min(blk["innov_runs"]["share_same_top3_set"],
                                        hb["share_draws_same_top3_set"]) >= STABLE and
                                    set(blk["cwa_ensemble_headline"]) ==
                                    set(blk["page_headline"])),
            "rule": f"stable if >= {STABLE:.0%} of the page's runs AND >= {STABLE:.0%} of the "
                    "A2 zone x county draws keep it, AND the R0 sibling's ensemble agrees"}
        res["solved"][s] = blk
        G.log(f"{s}: headline {blk['page_headline']}; innov runs same leader "
              f"{blk['innov_runs']['share_same_leader']:.0%}, cwa runs "
              f"{blk['cwa_runs']['share_same_leader']:.0%}, A2 draws "
              f"{hb['share_draws_same_leader']:.0%}; label {blk['proposed_label']}")

    # the ranking panel's rows: retire each class outright, page basis, A2 draws regenerated
    rows = -G.page_basis(D)[..., -1]                             # (R, A) cm avoided
    K = len(G.CLASSES)
    grp = G.grid_groups()["grp"]
    rng = np.random.default_rng(SEED)
    for si, start in enumerate((2026, 2030)):
        R_ = rows[si * K:(si + 1) * K]
        F = [np.exp(rng.uniform(np.log(0.5), np.log(2.0), 6))[grp] for _ in range(N_DRAWS)]
        out = {}
        for k, c in enumerate(G.CLASSES):
            base = R_[k]
            lead = G.top_towns(G.town_means(base, town, nT), elig)
            pert = np.stack([base * f for f in F])
            st = stability(pert, lead, base, town, nT, names, elig)
            out[c] = {"headline": [names[i] for i in lead],
                      "margins": margins(G.town_means(base, town, nT), elig, names), **st}
        res["basis_levers"][str(start)] = out
        G.log(f"basis {start}: same leader share " +
              ", ".join(f"{c} {out[c]['share_same_leader']:.0%}" for c in G.CLASSES))
    res["provenance"] = {
        "code": "dataprep_staging/n1_gaps/hotspots.py", "git_head": G.git_head(),
        "inputs": [G.source(G.PAGE)] + [G.source(m["path"]) for m in mem.values()] +
                  [G.source(A2B / "a2b_page_payload.json")],
        "note": "member fields are int16 at 1 mm per cell (members.npz); township means "
                "average that rounding away to well below the margins reported",
        "status": "post-hoc model-only check; no observation read, nothing scored"}
    G.write_new(OUT / "hotspot_stability.json", res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
