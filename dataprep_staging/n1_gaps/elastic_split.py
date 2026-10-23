"""Elastic / inelastic split of the subsidence the page's solved policies avoid.

The page says much of the avoided subsidence is "likely reversible" elastic rebound but
"cannot separate the two (the forward run does not save the elastic and inelastic
parts)". This script separates them from saved model output, without rerunning the flow
model and without editing ``hydrophysics/twin``:

- Column: the page's coupled_leveling zonal VEP column
  (``results/twin_runs/stage3_datum_gate/coupled_leveling/vep_zonal_leveling.json``),
  one parameter set per column zone (``zone_county.column_zone_map`` on the theta's meta).
  Its recurrence (``compaction.vep_compaction``) is the sum of an elastic term
  ``Ske (h0 - h_t)`` and a preconsolidation-gated inelastic term with viscous delay; this
  script re-evaluates the same recurrence term by term and checks the sum against the
  library function to float precision.
- Heads: the ENSEMBLE-MEAN monthly heads of ``datum_gate_innov.npz`` (36 runs), layer
  mean (as ``forward.compaction``). These are the displayed (origin-restarted) heads; the
  page's subsidence was computed run by run on free-running heads. So:
  (A) LINEAR split, the headline method: the elastic term is linear in head, so the
      ensemble mean of the avoided elastic part is exactly ``Ske * dh_bar`` (dh_bar = the
      policy-minus-baseline layer-mean head of the ensemble mean), up to the difference
      between the restarted and the free-running head DIFFERENCE (zero for a linear flow
      model). Inelastic = the page's own avoided total (subs_mean_by_rheology,
      coupled_leveling) minus that elastic part.
  (B) DIRECT check: the column run on the ensemble-mean heads, both terms, for the
      baseline and each policy; its total is compared to the page's total (the gap
      measures the nonlinearity of the column across runs plus the restart path).
  (C) LIFT test (illustration): the policy lifted after Dec 2029 (heads spliced back to
      the baseline from Jan 2030, an abrupt splice; the flow model's own return takes
      months), column re-run, avoided subsidence left at Dec 2032.
  (D) COMMITTED inelastic: the difference in the column's equilibrium inelastic strain
      at Dec 2032 (what the viscous term relaxes toward with no further loading).

No observation is read; nothing is scored or registered. CPU, seconds.
Usage: python -I dataprep_staging/n1_gaps/elastic_split.py
Output: results/dataprep/reports/N1_gaps_2026-10-24/science-13/elastic_split.json
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from n1_gaps import gaps_lib as G  # noqa: E402

OUT = G.OUT_ROOT / "science-13"
RUN = G.RUNS / "stage3_datum_gate"
COLUMN = RUN / "coupled_leveling" / "vep_zonal_leveling.json"
THETA = RUN / "stage3_theta.json"
FORWARD = G.FWD / "datum_gate_innov.npz"
SCEN = ("cut30", "retire_aqua")
YEARS = (2026, 2029, 2032)
LIFT_FROM = "2030-01"


def vep_terms(h: np.ndarray, ske, skv, tau, h_pc0, dt_days: float = 30.0):
    """``compaction.vep_compaction`` term by term: (elastic, inelastic, equilibrium) (n, T)."""
    n, T = h.shape
    decay = np.exp(-dt_days / tau)
    h_pc = h[:, 0] + h_pc0
    eps_i = np.zeros(n)
    eq = np.zeros(n)
    el = np.zeros((n, T))
    inel = np.zeros((n, T))
    eqs = np.zeros((n, T))
    for t in range(1, T):
        below = np.clip(h_pc - h[:, t], 0.0, None)
        eq = eq + skv * below
        h_pc = np.minimum(h_pc, h[:, t])
        eps_i = eq + (eps_i - eq) * decay
        el[:, t] = ske * (h[:, 0] - h[:, t])
        inel[:, t] = eps_i
        eqs[:, t] = eq
    return el, inel, eqs


def library_total(h: np.ndarray, p: dict) -> np.ndarray:
    import torch

    from hydrophysics.twin.compaction import vep_compaction

    t = torch.as_tensor(h, dtype=torch.float64)
    par = [torch.as_tensor([float(p[k])], dtype=torch.float64)
           for k in ("log_ske", "log_skv", "log_tau", "h_pc0")]
    return vep_compaction(t, *par).numpy()


def main() -> int:
    os.chdir(G.REPO)
    from hydrophysics.twin.zone_county import column_zone_map

    col = json.loads(COLUMN.read_text())
    sets = col["zonal"]
    meta = json.loads(THETA.read_text())["meta"]
    grid = G.grid_groups()
    zone = np.asarray(column_zone_map(grid["grid"], meta), int)
    ch = grid["changhua"]
    if zone.max() >= len(sets):
        raise SystemExit(f"column zones {np.unique(zone)} vs {len(sets)} parameter sets")
    z = np.load(FORWARD, allow_pickle=False)
    names = [str(n) for n in z["scenario_names"]]
    dates = pd.DatetimeIndex(z["dates"])
    o = int(z["origin"])
    labels = [str(r) for r in z["rheology_labels"]]
    S_page = z["subs_mean_by_rheology"][labels.index("coupled_leveling")].astype("float64")
    Hbar = z["heads_mean"].astype("float64").mean(axis=1)              # (scen, A, T)
    b = names.index("baseline")
    tix = {y: int(np.nonzero((dates.year == y) & (dates.month == 12))[0][0]) for y in YEARS}
    lift = int(np.nonzero(dates == pd.Timestamp(LIFT_FROM))[0][0])

    ske = np.array([math.exp(sets[k]["log_ske"]) for k in zone])
    par = {k: np.array([float(sets[zz][k]) for zz in zone]) for k in sets[0]}

    def run(h):
        return vep_terms(h, np.exp(par["log_ske"]), np.exp(par["log_skv"]),
                         np.exp(par["log_tau"]), par["h_pc0"])

    # check the term-by-term recurrence against the library, zone by zone, on the baseline
    Eb, Ib, Qb = run(Hbar[b])
    worst = 0.0
    for k, p in enumerate(sets):
        sel = zone == k
        if sel.any():
            worst = max(worst, float(np.abs(library_total(Hbar[b][sel], p)
                                            - (Eb + Ib)[sel]).max()))
    G.log(f"term-by-term recurrence vs compaction.vep_compaction: max |diff| {worst:.2e} m")
    if worst > 1e-9:
        raise SystemExit("the split does not reproduce the library column")

    def fwd(x):                     # forward from Dec 2022, cm
        return (x - x[:, o:o + 1]) * 100.0

    def summ(v: np.ndarray) -> dict:
        return {"fan": float(v.mean()), "yunlin": float(v[~ch].mean()),
                "changhua": float(v[ch].mean()),
                **{f"zone{k}": float(v[zone == k].mean()) for k in range(len(sets))}}

    res = {"what": "avoided forward subsidence (baseline minus policy, from Dec 2022, cm, "
                   "positive = less sinking) split into the column's elastic (reversible) "
                   "and inelastic (permanent once incurred) terms",
           "column_zones": {f"zone{k}": {"name": ("proximal", "mid", "distal")[k]
                                         if k < 3 else f"zone{k}",
                                         "n_cells": int((zone == k).sum()),
                                         "Ske": math.exp(p["log_ske"]),
                                         "Skv": math.exp(p["log_skv"]),
                                         "tau_days": math.exp(p["log_tau"])}
                            for k, p in enumerate(sets)},
           "check_recurrence_vs_library_max_abs_m": worst, "policies": {}}
    for s in SCEN:
        i = names.index(s)
        tot_page = -(fwd(S_page[i]) - fwd(S_page[b]))                     # (A, T)
        dh = Hbar[i] - Hbar[b]
        el_lin = ske[:, None] * dh * 100.0                                 # avoided elastic
        el_lin = el_lin - el_lin[:, o:o + 1]
        inel_lin = tot_page - el_lin
        Ep, Ip, Qp = run(Hbar[i])
        el_dir = -(fwd(Ep) - fwd(Eb))
        in_dir = -(fwd(Ip) - fwd(Ib))
        # lift: policy heads to Dec 2029, baseline heads from Jan 2030
        hl = Hbar[i].copy()
        hl[:, lift:] = Hbar[b][:, lift:]
        El, Il, _ = run(hl)
        lift_tot = -(fwd(El + Il) - fwd(Eb + Ib))
        lift_in = -(fwd(Il) - fwd(Ib))
        blk = {"by_year": {}}
        for y, t in tix.items():
            tp = tot_page[:, t]
            blk["by_year"][str(y)] = {
                "page_total": summ(tp),
                "A_linear": {"elastic": summ(el_lin[:, t]), "inelastic": summ(inel_lin[:, t]),
                             "inelastic_share_fan": float(inel_lin[:, t].mean() / tp.mean())},
                "B_direct_on_mean_heads": {
                    "total": summ(el_dir[:, t] + in_dir[:, t]),
                    "elastic": summ(el_dir[:, t]), "inelastic": summ(in_dir[:, t]),
                    "inelastic_share_fan": float(in_dir[:, t].mean()
                                                 / (el_dir[:, t] + in_dir[:, t]).mean()),
                    "total_vs_page_fan_cm": float((el_dir[:, t] + in_dir[:, t]).mean()
                                                  - tp.mean())}}
        tH = tix[YEARS[-1]]
        blk["C_lift_after_2029"] = {
            "avoided_dec2029_direct": summ(el_dir[:, tix[2029]] + in_dir[:, tix[2029]]),
            "avoided_left_dec2032": summ(lift_tot[:, tH]),
            "of_which_inelastic_dec2032": summ(lift_in[:, tH]),
            "note": "abrupt splice of the ensemble-mean heads back to the baseline in Jan "
                    "2030; the flow model's own return would take months (98 % of the "
                    "response comes in the first year)"}
        blk["D_committed_inelastic_dec2032"] = {
            "equilibrium_inelastic_strain_avoided": summ(-(Qp[:, tH] - Qb[:, tH]) * 100.0),
            "note": "difference in the column's equilibrium inelastic strain (Skv x "
                    "cumulative fall below preconsolidation): where the viscous term "
                    "converges with no further loading; it counts from Jan 2012, so it is "
                    "a policy difference only (both runs share the record to 2025)"}
        res["policies"][s] = blk
        a = blk["by_year"][str(YEARS[-1])]
        G.log(f"{s}: page total {a['page_total']['fan']:.3f} cm by {YEARS[-1]}; linear "
              f"elastic {a['A_linear']['elastic']['fan']:.3f}, inelastic "
              f"{a['A_linear']['inelastic']['fan']:.3f} ({a['A_linear']['inelastic_share_fan']:.0%}); "
              f"direct total {a['B_direct_on_mean_heads']['total']['fan']:.3f} "
              f"(inelastic {a['B_direct_on_mean_heads']['inelastic']['fan']:.3f}); "
              f"lifted 2030: left {blk['C_lift_after_2029']['avoided_left_dec2032']['fan']:.3f}")
    res["caveats"] = [
        "ensemble-mean heads: the elastic term is linear so method A is exact for the "
        "ensemble mean up to the restart-vs-free head difference; the inelastic term is not "
        "linear, so method B on mean heads is an approximation (its total vs the page's "
        "total is reported)",
        "the split is the model's own rheology (a 4-parameter zonal VEP column fitted to "
        "leveling); its elastic/inelastic partition is not separately identified by any "
        "registered test, and the leveling-fitted Ske/Skv trade off",
        "the 2032 numbers are model output only; no observation unit was read"]
    res["provenance"] = {
        "code": "dataprep_staging/n1_gaps/elastic_split.py", "git_head": G.git_head(),
        "inputs": [G.source(FORWARD), G.source(COLUMN), G.source(THETA)],
        "status": "post-hoc model-only analysis; nothing scored or registered"}
    G.write_new(OUT / "elastic_split.json", res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
