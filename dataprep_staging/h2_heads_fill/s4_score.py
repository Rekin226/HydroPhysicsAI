"""Registered scoring of WP H2 (pre-registration ``H2.fill_acceptance.v1``).

For each registered fill test (one artificial-gap family on its fresh realizations), this
script first calls ``harness.prereg3.assert_can_score`` (refuses if the registration is not
intact or postdates a look at the test data; logs the scoring look), then scores the product
as frozen by the design stage (``design_params.json``, hashed in the registration):

* coverage: rule gate-3 (``harness.coverage3.coverage_gate``) per stratum on clean truth,
  with the clusters stored in the run tables;
* accuracy: per stratum, the pooled clean-truth RMSE of the product must be lower than that of
  the twin's backfill, climatology + AR(1), space-time kriging, and same-layer IDW (on the
  cells where IDW is defined, against the product on the same cells).

A stratum passes when both hold; a test passes when all its strata pass. Where the design
stage made a baseline the method of a gap class present in the stratum (plan H2: "a family
where a baseline wins falls back to that baseline"), the product must not be worse than that
baseline (ties allowed) instead of strictly better.

Family E masks are deterministic per well (its first 66 or last 48 months), so a fresh E
realization that picks a well already used by a design realization repeats a design
well-gap. The registered E gate is therefore computed on the fresh well-gaps that no design
realization contains (``E_lead66`` / ``E_trail48`` restricted to those wells); all fresh E
cells are reported.

Reported, not gated: all-truth tables, the model-well subset, per-bridged-share classes of
family d, the within-fresh split-conformal multiplier (``harness.scorers.conformal_split``,
groups = clusters), every method's RMSE, both bridge forms.

Usage: python -I s4_score.py TEST_ID [TEST_ID ...]   (or 'all')
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2])]

import numpy as np  # noqa: E402
from harness import prereg3, scorers  # noqa: E402

from h2_heads_fill import config as C  # noqa: E402
from h2_heads_fill import scoring  # noqa: E402

PREREG_ID = "H2.fill_acceptance.v1"
FAMILY_OF = {"H2.fresh.N": "N", "H2.fresh.S": "S", "H2.fresh.E": "E", "H2.fresh.H": "H",
             "H2.fresh.T": "T", "H2.fresh.d": "d"}
STRATA = {"N": ("N_net1", "N_net2"), "S": ("S01", "S02", "S03", "S06", "S12", "S24"),
          "E": ("E_lead66", "E_trail48"), "H": ("H_half12",), "T": ("T_S", "T_H06"),
          "d": ("d",)}
PARAMS = C.EVIDENCE_DIR / "design" / "design_params.json"


def product_table(fam, params):
    df = scoring.load_runs("pre", [fam], C.FRESH_R)
    if fam == "d":
        if params.get("bridge_disabled"):
            raise SystemExit("design disabled a bridge class; the daily scorer does not "
                             "implement the router fall-back for disabled classes")
        p = scoring.product_daily(df, params)
        p["share_class"] = p["stratum"]
        p["stratum"] = "d"
        return df, p
    return df, scoring.product_monthly(df, params)


def stratum_verdicts(p):
    cov = {r["stratum"]: r for r in scoring.coverage_rows(p)}
    rm = scoring.rmse_table(p, ("product",) + scoring.BASELINES + scoring.REPORTED)
    used = p.groupby("stratum").method_product.agg(lambda x: set(x)).to_dict()
    out = {}
    for rec in rm.to_dict("records"):
        s = rec["stratum"]
        beats = {}
        for b in scoring.BASELINES:
            if b == "idw":
                ref, mine = rec.get("rmse_idw"), rec.get("rmse_product_on_idw_cells")
            else:
                ref, mine = rec.get(f"rmse_{b}"), rec.get("rmse_product")
            if ref is None or not np.isfinite(ref) or rec.get(f"n_{b}", 0) == 0:
                beats[b] = {"defined": False}
            else:
                fb = b in used.get(s, set())
                ok = bool(mine <= ref + 1e-12) if fb else bool(mine < ref)
                beats[b] = {"defined": True, "rmse_baseline": ref, "rmse_product": mine,
                            "design_fallback_in_stratum": fb,
                            "rule": "not worse (fall-back)" if fb else "strictly lower",
                            "product_lower": ok}
        rmse_ok = all(v["product_lower"] for v in beats.values() if v["defined"])
        c = cov[s]
        out[s] = {"coverage_gate": c, "rmse": rec, "baselines": beats, "rmse_pass": rmse_ok,
                  "coverage_pass": bool(c["pass"]), "calibrated": bool(c["calibrated"]),
                  "pass": bool(c["pass"]) and rmse_ok}
    return out


def split_conformal(p):
    res = {}
    for s, g in p.groupby("stratum"):
        e = (g.pred_product - g.truth).to_numpy()
        res[s] = scorers.conformal_split(e, g.cluster.astype(str).to_numpy(),
                                         g.sd_product.to_numpy(), seed=C.SEED_SPLIT)
    return res


def design_e_gaps():
    """(sid, stratum) well-gaps of the design E realizations (deterministic per well)."""
    des = scoring.load_runs("pre", ["E"], {"E": C.DESIGN_R["E"]})
    return set(zip(des.sid, des.stratum, strict=True))


def score_test(tid, params):
    fam = FAMILY_OF[tid]
    ok = prereg3.assert_can_score(PREREG_ID, C.WP, test_id=tid)
    raw, p = product_table(fam, params)
    p_all_fresh = p
    if fam == "E":
        dk = design_e_gaps()
        new_gap = np.array([(a, b) not in dk for a, b in zip(p.sid, p.stratum, strict=True)])
        p = p[new_gap]
    clean = p[p.clean.astype(bool)]
    verdict = stratum_verdicts(clean)
    missing = [s for s in STRATA[fam] if s not in verdict]
    rep = {"test_id": tid, "family": fam, "realizations": [C.FRESH_R[fam].start,
                                                         C.FRESH_R[fam].stop - 1],
           "assert_can_score": {"sha256": ok["sha256"], "inputs_verified": ok["inputs_verified"],
                                "post_registration_looks": ok["post_registration_looks"],
                                "unverifiable_prior_looks": len(ok["unverifiable_prior_looks"])},
           "strata_clean": verdict, "strata_missing": missing,
           "pass": (not missing) and all(v["pass"] for v in verdict.values()),
           "reported": {
               "all_truth": {"coverage": scoring.coverage_rows(p),
                             "rmse": scoring.rmse_table(
                                 p, ("product",) + scoring.BASELINES + scoring.REPORTED
                             ).to_dict("records")},
               "model_wells_clean": {"coverage": scoring.coverage_rows(
                   clean[clean.in_model.astype(bool)]),
                   "rmse": scoring.rmse_table(clean[clean.in_model.astype(bool)],
                                              ("product",) + scoring.BASELINES
                                              ).to_dict("records")},
               "split_conformal_within_fresh_clean": split_conformal(clean),
               "n_cells_all": int(len(raw)), "n_cells_clean_scored": int(len(clean))}}
    if fam == "d":
        rep["reported"]["bridgeable_share"] = float(raw.ok.mean())
        rep["reported"]["not_bridged_reasons"] = raw[~raw.ok].reason.value_counts().to_dict()
        rep["reported"]["share_classes_clean"] = {
            "coverage": scoring.coverage_rows(clean, by="share_class"),
            "rmse": scoring.rmse_table(clean, ("product", "router") + scoring.BASELINES,
                                       by="share_class").to_dict("records")}
        rep["reported"]["partial_mean_rmse_clean"] = scoring.rmse(clean.partial - clean.truth)
    if fam == "d":
        rep["reported"]["bridge_forms_rmse_clean"] = {
            "shape": scoring.rmse(clean.pred_shape - clean.truth),
            "lin": scoring.rmse(clean.pred_lin - clean.truth)}
    if fam == "E":
        fk = set(zip(raw.sid, raw.stratum, strict=True))
        ca = p_all_fresh[p_all_fresh.clean.astype(bool)]
        rep["reported"]["well_gaps_also_in_design"] = {
            "fresh_well_gaps": len(fk), "also_in_design": len(fk & dk),
            "share": len(fk & dk) / max(len(fk), 1),
            "gated_on": "fresh well-gaps not in any design realization"}
        rep["reported"]["all_fresh_E_clean"] = {
            "coverage": scoring.coverage_rows(ca),
            "rmse": scoring.rmse_table(ca, ("product",) + scoring.BASELINES).to_dict("records")}
    out = C.EVIDENCE_DIR / "registered" / f"{tid}.json"
    scoring.jdump(rep, out)
    print(tid, "PASS" if rep["pass"] else "FAIL",
          {s: (v["coverage_gate"]["coverage"], v["coverage_pass"], v["rmse_pass"])
           for s, v in verdict.items()}, flush=True)
    return rep


def main(argv):
    params = json.loads(PARAMS.read_text())
    tids = list(FAMILY_OF) if argv == ["all"] else argv
    for tid in tids:
        score_test(tid, params)


if __name__ == "__main__":
    main(sys.argv[1:])
