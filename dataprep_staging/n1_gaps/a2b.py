"""A2b: the registered A2 ranking check re-run on the page's own response basis (post hoc).

N1.policy_response.v1 declared check A2 ran on ``results/twin_forward/response_basis.npz``,
whose only member is ``stage3_spreadL_gate`` (the 10 km "previous model" the page shows as
its alternative), because the registration named that path as "the page's response basis".
The page uses ``response_basis_innov.npz`` (member ``stage3_datum_gate``) and R0 is the
CWA-recipe sibling ``response_basis_cwa.npz``. A2b runs the IDENTICAL A2 procedure (same
seed 20261223, same 500 log-uniform [0.5, 2] zone x county draws per start, same order of
random calls, same statistics) on:

- ``repro``   response_basis.npz (must reproduce step_a/a2.json exactly: code check);
- ``innov``   response_basis_innov.npz (the page's basis, raw);
- ``cwa``     response_basis_cwa.npz (R0's sibling basis, raw);
- ``page``    the page payload's rescaled basis rows (what the ranking panel draws).

Additions (computed from the same draws; they consume no random numbers): the ranking
per 100 GWh/yr forgone (the page's second ranking mode; class GWh/yr from the page
payload), the top-decile overlap of the retire-aquaculture field, and the stability of the
page's headline townships (topTowns filter) for cut30 and retire_aqua built from the basis.

Model outputs only: no observation value is read, nothing is scored or registered.
Usage: python -I dataprep_staging/n1_gaps/a2b.py   (CPU, about a minute)
Outputs: results/dataprep/reports/N1_gaps_2026-10-24/science-04/ (new files only).
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

SEED = 20261223                       # run_step_a.SEED
N_DRAWS = 500
OUT = G.OUT_ROOT / "science-04"
A2_JSON = G.REPORTS_DIR / "N1_policy" / "step_a" / "a2.json"
BASES = {"repro": G.FWD / "response_basis.npz",
         "innov": G.FWD / "response_basis_innov.npz",
         "cwa": G.FWD / "response_basis_cwa.npz"}


def raw_rows(path: Path) -> tuple[np.ndarray, int]:
    """A2's own reduction: cm avoided at the last year-end per (start x class, cell)."""
    from hydrophysics.twin.app.prep import basis_deltas, year_ends

    z = np.load(path, allow_pickle=False)
    names = [str(n) for n in z["scenario_names"]]
    ye, years = year_ends(pd.DatetimeIndex(z["dates"]))
    b = basis_deltas(z["subs_mean"] * 100.0, z["heads_mean"], names, ye)
    if b["missing"]:
        raise SystemExit(f"{path}: missing basis rows {b['missing']}")
    return -b["dsub"][..., -1], int(years[-1])


def kendall(a, b) -> float:
    from scipy.stats import kendalltau

    return float(kendalltau(a, b).statistic)


def rank_block(base: np.ndarray, draws: np.ndarray) -> dict:
    """Order, tau and survival of a lever ranking (``base`` (K,), ``draws`` (N, K))."""
    taus = np.array([kendall(base, d) for d in draws])
    order = np.argsort(-base)
    swaps = {}
    for i in range(len(order) - 1):
        a, b = order[i], order[i + 1]
        swaps[f"{G.CLASSES[a]}>{G.CLASSES[b]}"] = float(np.mean(draws[:, a] > draws[:, b]))
    return {"lever_order": [G.CLASSES[i] for i in order],
            "share_tau_1": float(np.mean(taus >= 1.0 - 1e-12)),
            "tau_p5_p50": [float(np.quantile(taus, 0.05)), float(np.median(taus))],
            "adjacent_pair_kept_share": swaps,
            "ranking_survives": bool(np.mean(taus >= 1.0 - 1e-12) >= 0.9)}


def a2_procedure(dsub: np.ndarray, grp: np.ndarray, rng, gwh: np.ndarray, town, names,
                 elig, year_end: int) -> dict:
    """run_step_a.a2's loop, statement for statement, plus the additions."""
    from hydrophysics.twin.app.prep import STARTS

    K = len(G.CLASSES)
    A = dsub.shape[1]
    nT = len(names)
    out = {"year_end": year_end, "starts": {}}
    for si, s in enumerate(STARTS):
        R_ = dsub[si * K:(si + 1) * K]
        base_tot = R_.sum(axis=1)
        base_order = np.argsort(-base_tot)
        irr = R_[G.CLASSES.index("irrigation")] * 0.3
        top = irr >= np.quantile(irr, 0.9)
        aq = R_[G.CLASSES.index("aquaculture")]
        top_aq = G.top_decile(aq)
        base_tw = {"cut30": G.town_means(irr, town, nT), "retire_aqua": G.town_means(aq, town, nT)}
        base_lead = {k: G.top_towns(v, elig) for k, v in base_tw.items()}
        taus, jac, tots, jac_aq = [], [], [], []
        lead = {k: [] for k in base_tw}
        for _ in range(N_DRAWS):
            f = np.exp(rng.uniform(np.log(0.5), np.log(2.0), 6))[grp]
            P = R_ * f[None, :]
            tot = P.sum(axis=1)
            taus.append(kendall(base_tot, tot))
            pt = P[G.CLASSES.index("irrigation")] * 0.3
            tp = pt >= np.quantile(pt, 0.9)
            jac.append(float((top & tp).sum() / max((top | tp).sum(), 1)))
            tots.append(tot)
            # additions (no random numbers consumed)
            pa = P[G.CLASSES.index("aquaculture")]
            jac_aq.append(G.jaccard(top_aq, G.top_decile(pa)))
            lead["cut30"].append(G.top_towns(G.town_means(pt, town, nT), elig))
            lead["retire_aqua"].append(G.top_towns(G.town_means(pa, town, nT), elig))
        tots = np.array(tots)
        taus = np.array(taus)
        per_gwh = 100.0 * (tots / A) / gwh[None, :]          # fan-mean cm per 100 GWh/yr
        base_gwh = 100.0 * (base_tot / A) / gwh
        tw = {}
        for k in base_tw:
            b0 = base_lead[k]
            tw[k] = {
                "headline_towns_unperturbed": [names[i] for i in b0],
                "avoided_cm_unperturbed": [round(float(base_tw[k][i]), 4) for i in b0],
                "share_draws_same_leader": float(np.mean([(lt[:1] == b0[:1]) for lt in lead[k]])),
                "share_draws_same_top3_set": float(np.mean([set(lt) == set(b0) for lt in lead[k]])),
                "leader_counts": pd.Series([names[lt[0]] for lt in lead[k] if lt])
                .value_counts().to_dict()}
        out["starts"][str(s)] = {
            "lever_order": [G.CLASSES[i] for i in base_order],
            "lever_total_cm_sum": dict(zip(G.CLASSES, map(float, base_tot), strict=True)),
            "lever_fan_mean_cm": dict(zip(G.CLASSES, (float(v) for v in base_tot / A),
                                          strict=True)),
            "share_tau_1": float(np.mean(taus >= 1.0 - 1e-12)),
            "tau_p5_p50": [float(np.quantile(taus, 0.05)), float(np.median(taus))],
            "top_decile_jaccard_p5_p50": [float(np.quantile(jac, 0.05)), float(np.median(jac))],
            "lever_total_p5_p95": {c: [float(np.quantile(tots[:, k], 0.05)),
                                       float(np.quantile(tots[:, k], 0.95))]
                                   for k, c in enumerate(G.CLASSES)},
            "ranking_survives": bool(np.mean(taus >= 1.0 - 1e-12) >= 0.9),
            # ---- A2b additions
            "cm_ranking_detail": rank_block(base_tot, tots),
            "per_100gwh": {"unit": "fan-mean cm avoided per 100 GWh/yr forgone",
                           "value": dict(zip(G.CLASSES, map(float, base_gwh), strict=True)),
                           "p5_p95": {c: [float(np.quantile(per_gwh[:, k], 0.05)),
                                          float(np.quantile(per_gwh[:, k], 0.95))]
                                      for k, c in enumerate(G.CLASSES)},
                           **rank_block(base_gwh, per_gwh)},
            "top_decile_jaccard_retire_aqua_p5_p50": [float(np.quantile(jac_aq, 0.05)),
                                                     float(np.median(jac_aq))],
            "headline_townships_basis": tw,
        }
        G.log(f"  start {s}: cm order {out['starts'][str(s)]['lever_order']}, "
              f"tau=1 {out['starts'][str(s)]['share_tau_1']:.1%}; per-GWh order "
              f"{out['starts'][str(s)]['per_100gwh']['lever_order']}, tau=1 "
              f"{out['starts'][str(s)]['per_100gwh']['share_tau_1']:.1%}")
    return out


def compare_repro(res: dict, ref: dict) -> dict:
    """Max |difference| between the re-run's A2 fields and step_a/a2.json."""
    keys = ("lever_total_cm_sum", "share_tau_1", "tau_p5_p50", "top_decile_jaccard_p5_p50",
            "lever_total_p5_p95")
    worst, orders, surv = 0.0, True, True

    def flat(x):
        if isinstance(x, dict):
            return [v for k in sorted(x) for v in flat(x[k])]
        if isinstance(x, list):
            return [v for e in x for v in flat(e)]
        return [float(x)]
    for s, r in ref["starts"].items():
        mine = res["starts"][s]
        for k in keys:
            a, b = np.array(flat(mine[k])), np.array(flat(r[k]))
            worst = max(worst, float(np.max(np.abs(a - b))))
        orders &= mine["lever_order"] == r["lever_order"]
        surv &= mine["ranking_survives"] == r["ranking_survives"]
    return {"max_abs_diff": worst, "orders_equal": bool(orders),
            "verdicts_equal": bool(surv), "reproduces": bool(worst < 1e-9 and orders and surv)}


def main() -> int:
    os.chdir(G.REPO)
    D = G.load_payload()
    gwh = np.array([c["gwh"] for c in D["classes"]], float)
    town, names, elig = G.townships(D)
    gg = G.grid_groups()
    grp = gg["grp"]
    if len(grp) != D["meta"]["nA"] or len(town) != len(grp):
        raise SystemExit("grid size mismatch between the page and the rebuilt grid")
    prov = {"code": "dataprep_staging/n1_gaps/a2b.py", "git_head": G.git_head(),
            "seed": SEED, "draws_per_start": N_DRAWS,
            "page": G.source(G.PAGE), "class_gwh_per_yr_from_page": dict(
                zip(G.CLASSES, map(float, gwh), strict=True)),
            "post_hoc": True,
            "status": "post-hoc model-only check; not a registered test, not a look at any "
                      "unit; A2's own result (step_a/a2.json) is left as written"}
    results = {}
    for tag, path in BASES.items():
        G.log(f"A2b {tag}: {G.rel_to_repo(path)}")
        z = np.load(path, allow_pickle=False)
        members = [str(m) for m in z["member_labels"]] if "member_labels" in z.files else []
        dsub, yend = raw_rows(path)
        rng = np.random.default_rng(SEED)
        res = a2_procedure(dsub, grp, rng, gwh, town, names, elig, yend)
        res = {"basis": G.rel_to_repo(path), "basis_members": members,
               "basis_sha256": G.sha256(path), **res, "provenance": prov}
        results[tag] = res
    G.log("A2b page: the payload's rescaled rows")
    rows = -G.page_basis(D)[..., -1]
    rng = np.random.default_rng(SEED)
    res = a2_procedure(rows, grp, rng, gwh, town, names, elig, int(D["meta"]["years"][-1]))
    results["page"] = {"basis": "results/twin/twin_app.html payload dSubs (rescaled per class "
                                "to the ensemble: " + json.dumps(D["calib"]["subs"]) + ")",
                       "basis_members": ["stage3_datum_gate (rescaled)"], **res,
                       "provenance": prov}
    ref = json.loads(A2_JSON.read_text())
    chk = compare_repro(results["repro"], ref)
    G.log(f"repro vs step_a/a2.json: {chk}")
    G.write_new(OUT / "a2_repro_check.json", {"a2_json": G.rel_to_repo(A2_JSON), **chk,
                                               "rerun": results["repro"]})
    G.write_new(OUT / "a2b_innov.json", results["innov"])
    G.write_new(OUT / "a2b_cwa.json", results["cwa"])
    G.write_new(OUT / "a2b_page_payload.json", results["page"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
