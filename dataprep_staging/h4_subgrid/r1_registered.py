"""WP H4 registered run (pre-registration H4.subgrid_operator.v1).

Calls harness.prereg3.assert_can_score first (refuses if the registration postdates a look at
its test data or a registered input changed; logs the scoring looks), then computes, with the
primary configuration ``h4lib.PRIMARY``:

* T1 held-out years: multipliers fitted on every well's 2012-02..2019-12 residuals, scored on
  2020-01..2022-12 (training-mean datum, 2012-19 month-of-year climatology);
* T2 held-out sites: the same, with multipliers fitted per A1 site-grouped fold
  (``fresh.folds.wells_all`` partition; test and buffer wells excluded) and applied to the
  fold's test wells, scored on 2020-01..2022-12;
* both placebo families (39 seeded draws each, the same draws for T1 and T2);
* the registered criteria A1-A8 and the sensitivity members (reported, never gating).

Writes results/dataprep/reports/H4_evidence/registered/{scores.json, arrays.npz}.
Usage: python r1_registered.py [--dry-no-score]  (the dry flag refuses to run: kept so that a
mistaken invocation before registration cannot score)
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import h4lib as L  # noqa: E402
from common.paths import DATAPREP  # noqa: E402
from harness import prereg3  # noqa: E402

PREREG_ID = "H4.subgrid_operator.v1"
OUT = DATAPREP / "reports/H4_evidence/registered"
SENSITIVITY = {
    "recorded_arm": {"arm": "recorded"},
    "r_max_3km": {"r_max_m": 3000.0},
    "all_ingrid_counties": {"counties": ("yunlin", "changhua", "chiayi")},
    "one_multiplier": {"n_mult": 1},
    "layer_weight_ones": {"layer_weights": "ones"},
    "c20_months_only": {"c20_only": True},
    "t_bound_half": {"t_bound_scale": 0.5},
}


def _q(x):
    return [float(v) for v in np.asarray(x, float).ravel()]


def evaluate(d: L.Data, X, m, q, clim) -> dict:
    t1 = L.run_t1(d, X, m, q, clim)
    t2 = L.run_t2(d, X, m, q, clim)
    return {"t1": t1, "t2": t2}


def brief(res: dict, d: L.Data, m: dict, q: np.ndarray) -> dict:
    t1, t2 = res["t1"], res["t2"]
    q4 = q == 4
    return {"t1": {"a": _q(t1["a"]), "summary": t1["summary"],
                   "q4_amplitude_ratio_base": L.amplitude_ratio(d, m, None, q4),
                   "q4_amplitude_ratio_corrected": L.amplitude_ratio(d, m, t1["corr"], q4)},
            "t2": {"a_by_fold": {str(k): _q(v) for k, v in t2["a_by_fold"].items()},
                   "summary": t2["summary"],
                   "q4_amplitude_ratio_corrected": L.amplitude_ratio(d, m, t2["corr"], q4)}}


def criteria(real: dict, plac: dict, cfg: L.Config) -> list[dict]:
    out = []
    for tk, label in (("t1", "held-out years 2020-22"), ("t2", "held-out sites (A1 site folds) "
                                                                "and years 2020-22")):
        s = real[tk]["summary"]["Q4"]
        e0, e1 = s["excess_base_m2"], s["excess_corrected_m2"]
        out.append({"id": f"A{1 if tk == 't1' else 2}",
                    "criterion": f"Q4 excess (sum over Q4 wells of model MSE - climatology MSE) "
                                 f"falls, {label}",
                    "threshold": "corrected < base (strict)",
                    "measured": f"{e0:.3f} -> {e1:.3f} m2 (reduction {e0 - e1:.3f} m2, "
                                f"{(e0 - e1) / e0 if e0 else float('nan'):.1%} of the base "
                                f"excess; {s['n_wells']} Q4 wells)",
                    "value": e0 - e1, "passed": bool(e1 < e0)})
    k = 3
    for tk, label in (("t1", "held-out years"), ("t2", "held-out sites")):
        dr = real[tk]["summary"]["Q4"]["excess_reduction_m2"]
        for fam, name in (("shuffle", "pump positions shuffled within the county"),
                          ("distant", "only pumps more than 15 km away (donor neighbourhoods)")):
            dp = np.asarray(plac[fam][tk], float)
            n_ge = int(np.sum(dp >= dr))
            p = (1 + n_ge) / (1 + dp.size)
            out.append({"id": f"A{k}", "criterion": f"{label}: the real kernel's Q4 excess "
                                                    f"reduction beats the placebo ({name})",
                        "threshold": f"permutation p = (1 + #placebo >= real) / (1 + "
                                     f"{cfg.n_placebo}) <= {cfg.alpha}",
                        "measured": f"real {dr:.3f} m2; placebo median {np.median(dp):.3f}, "
                                    f"max {dp.max():.3f}; {n_ge} of {dp.size} >= real; "
                                    f"p = {p:.3f}",
                        "value": p, "passed": bool(p <= cfg.alpha)})
            k += 1
    for tk, label in (("t1", "held-out years"), ("t2", "held-out sites")):
        sm = real[tk]["summary"]
        rat = {qq: sm[qq]["rmse_ratio_corrected_over_base"] for qq in ("Q1", "Q2", "Q3")}
        out.append({"id": f"A{k}", "criterion": f"{label}: Q1, Q2 and Q3 equal-well RMSE each "
                                                f"rise by no more than 2 %",
                    "threshold": f"corrected / base <= {cfg.q123_tol} in each quartile",
                    "measured": ", ".join(f"{qq} {v:.4f}" for qq, v in rat.items()),
                    "value": max(rat.values()),
                    "passed": bool(all(v <= cfg.q123_tol for v in rat.values()))})
        k += 1
    return out


def compute(cfg: L.Config, builder, sensitivity: dict, gate: dict | None) -> tuple[dict, dict]:
    """Every registered number for configuration ``cfg``; ``builder(cfg)`` returns the Data
    (``h4lib.build_data`` in the registered run; a synthetic builder in the smoke test)."""
    t_start = time.perf_counter()
    d = builder(cfg)
    m = L.masks(d)
    q, sd, cuts = L.quartiles(d, m)
    clim = L.climatology(d, m)
    Gs, Gl, npump = L.operator_matrices(d)
    X = list(L.regressors(d, Gs, Gl))
    real = evaluate(d, X, m, q, clim)
    # placebos (the same draws for T1 and T2)
    support = npump > 0
    plac = {"shuffle": {"t1": [], "t2": [], "a_t1": []},
            "distant": {"t1": [], "t2": [], "a_t1": []}}
    for r in range(cfg.n_placebo):
        rng = np.random.default_rng(cfg.seed_shuffle + r)
        Gs_p, Gl_p, _ = L.operator_matrices(d, pump_xy=L.shuffle_xy(d, rng))
        res = evaluate(d, list(L.regressors(d, Gs_p, Gl_p)), m, q, clim)
        plac["shuffle"]["t1"].append(L.q4_reduction(res["t1"]))
        plac["shuffle"]["t2"].append(L.q4_reduction(res["t2"]))
        plac["shuffle"]["a_t1"].append(_q(res["t1"]["a"]))
        rng = np.random.default_rng(cfg.seed_distant + r)
        cen = L.distant_centres(d, rng, support)
        Gs_p, Gl_p, _ = L.operator_matrices(d, centres=cen, support=support)
        res = evaluate(d, list(L.regressors(d, Gs_p, Gl_p)), m, q, clim)
        plac["distant"]["t1"].append(L.q4_reduction(res["t1"]))
        plac["distant"]["t2"].append(L.q4_reduction(res["t2"]))
        plac["distant"]["a_t1"].append(_q(res["t1"]["a"]))
    crit = criteria(real, plac, cfg)
    passed = all(c["passed"] for c in crit)
    # sensitivity members (reported, never gating)
    sens = {}
    for name, kw in sensitivity.items():
        c2 = L.with_config(cfg, **kw)
        d2 = builder(c2)
        m2 = L.masks(d2)
        q2, _, cuts2 = L.quartiles(d2, m2)
        clim2 = L.climatology(d2, m2)
        G2s, G2l, n2 = L.operator_matrices(d2)
        X2 = list(L.regressors(d2, G2s, G2l))
        r2 = evaluate(d2, X2, m2, q2, clim2)
        sens[name] = {"config_change": {k: (list(v) if isinstance(v, tuple) else v)
                                        for k, v in kw.items()},
                      "wells_with_kernel": int((n2 > 0).sum()), "quartile_cuts_m": cuts2,
                      "quartiles_identical_to_primary": bool(np.array_equal(q2, q)),
                      **brief(r2, d2, m2, q2)}
    spatial = L.run_t2_spatial_only(d, X, m, q)
    sc_ok = m["score_ok"]
    out = {
        "prereg_id": PREREG_ID, "prereg_sha256": gate["sha256"] if gate else None,
        "post_registration_looks": gate["post_registration_looks"] if gate else None,
        "unverifiable_prior_looks_n": len(gate["unverifiable_prior_looks"]) if gate else None,
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "config": cfg.as_dict(), "notes": d.notes,
        "n_wells": int(len(d.sids)), "n_pumps": int(len(d.pump_xy)),
        "wells_with_kernel": int(support.sum()),
        "wells_with_kernel_by_county": {c: int((support & (d.county == c)).sum())
                                        for c in ("YL", "CH")},
        "quartile_cuts_sd_m": cuts,
        "quartile_counts_classified": {f"Q{k}": int((q == k).sum()) for k in (1, 2, 3, 4)},
        "quartile_counts_scored": {f"Q{k}": int(((q == k) & sc_ok).sum()) for k in (1, 2, 3, 4)},
        "q4_scored_by_county": {c: int(((q == 4) & sc_ok & (d.county == c)).sum())
                                for c in ("YL", "CH")},
        "q4_scored_with_kernel": int(((q == 4) & sc_ok & support).sum()),
        "n_scored": int(sc_ok.sum()), "n_fit_wells": int(m["fit_ok"].sum()),
        "primary": brief(real, d, m, q),
        "placebos": {f: {k: (v if k.startswith("a_") else _q(v)) for k, v in plac[f].items()}
                     for f in plac},
        "criteria": crit, "gate_passed": bool(passed),
        "sensitivity": sens,
        "t2_spatial_only_reported": {"summary": spatial["summary"],
                                     "a_by_fold": {str(k): _q(v) for k, v in
                                                   spatial["a_by_fold"].items()}},
        "runtime_s": round(time.perf_counter() - t_start, 1)}
    arrays = dict(
        sids=d.sids, dates=d.dates, county=d.county, layer=d.layer,
        cell=d.cell, quartile=q, sd_fit=sd, n_train=m["n_train"], n_test=m["n_test"],
        score_ok=sc_ok, npump=npump, S_unit=X[0], L_unit=X[1], a_t1=real["t1"]["a"],
        corr_t1=real["t1"]["corr"], corr_t2=real["t2"]["corr"],
        mse_base=real["t1"]["base"]["mse"], mse_clim=real["t1"]["base"]["mse_clim"],
        mse_t1=real["t1"]["cor"]["mse"], mse_t2=real["t2"]["cor"]["mse"],
        T=d.T, S=d.S, w=d.w)
    return out, arrays


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-no-score", action="store_true")
    args = ap.parse_args()
    if args.dry_no_score:
        raise SystemExit("refusing: this script only runs as the registered score")
    gate = prereg3.assert_can_score(PREREG_ID, "H4")
    OUT.mkdir(parents=True, exist_ok=True)
    out, arrays = compute(L.PRIMARY, L.build_data, SENSITIVITY, gate)
    (OUT / "scores.json").write_text(json.dumps(out, indent=1, ensure_ascii=False,
                                                default=float) + "\n")
    np.savez_compressed(OUT / "arrays.npz", **arrays)
    print(json.dumps({"criteria": out["criteria"], "gate_passed": out["gate_passed"],
                      "a_t1": out["primary"]["t1"]["a"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
