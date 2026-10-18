"""Family-E gates recomputed with a consistent freshness exclusion (WP H2 repair round 3).

E masks are deterministic per well (its first 66 or last 48 months), so a realization that
picks a well already used repeats the same cells. Both registrations excluded H2's own earlier
E well-gaps from the E gate (v1: design r0-4; v2: r0-4 and r10-19), but not the pilot's E
r0-1 well-gaps, which they list as design-time units and which are the same cells (plan 4.5:
design-time units cannot serve as fresh validation units). This script recomputes the E strata
of both registered rounds with the pilot's well-gaps excluded as well, with the registered
scoring code (``s4_score.stratum_verdicts``: coverage rule gate-3 on clean truth and RMSE below
every defined baseline), and also reproduces the as-registered verdicts. It changes no
registered record; it is a post-look correction of how the registered exclusion was applied.

It also counts the E-eligible well-gaps that no realization (pilot r0-1, H2 r0-4, r10-19,
r30-59) has used: the pool a further fresh E test could draw from.

Usage: python -I s11_e_fresh.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2])]

import numpy as np  # noqa: E402
from harness import prereg3  # noqa: E402

from h2_heads_fill import config as C  # noqa: E402
from h2_heads_fill import r3_common as R3  # noqa: E402
from h2_heads_fill import runner, scoring  # noqa: E402
from h2_heads_fill.s4_score import stratum_verdicts  # noqa: E402

ROUNDS = {
    "v1": {"params": R3.PARAMS_V1, "fresh": R3.E_R_V1, "seen": R3.E_R_DESIGN,
           "registered": C.EVIDENCE_DIR / "registered" / "H2.fresh.E.json"},
    "v2": {"params": R3.PARAMS_V2, "fresh": R3.E_R_V2, "seen": R3.E_R_DESIGN + R3.E_R_V1,
           "registered": C.EVIDENCE_DIR / "registered_v2" / "H2.fresh2.E.json"},
}


def gaps_of(rs):
    d = scoring.load_runs("pre", ["E"], {"E": list(rs)})
    return set(zip(d.sid, d.stratum, strict=True))


def brief(v):
    c = v["coverage_gate"]
    b = {k: (round(x["rmse_product"], 3), round(x["rmse_baseline"], 3))
         for k, x in v["baselines"].items() if x["defined"]}
    return {"coverage": c["coverage"], "lo": c["lo"], "hi": c["hi"], "n": c["n"],
            "k": c["k_clusters"], "n_eff_lower": c["n_eff_lower"],
            "window_applies": c["window_applies"], "in_window": c["in_window"],
            "contains_target": c["contains_target"], "calibrated": c["calibrated"],
            "coverage_pass": v["coverage_pass"], "rmse_product_vs_baseline": b,
            "rmse_pass": v["rmse_pass"], "pass": v["pass"]}


def main():
    rec = prereg3.record_look(
        ["H2:fresh_E_r10-19", "H2:fresh2_E_r30-59", "H2:design_E_r0-4"],
        "heads:2012-01..2022-12", C.WP, "post_look_repair",
        note="H2 repair round 3: family-E strata of H2.fill_acceptance.v1 and v2 recomputed "
             "with the pilot's E r0-1 well-gaps excluded as well (verifier finding); as-"
             "registered verdicts reproduced; no registered record changed.")
    pilot = R3.pilot_e_gaps()
    out = {"label": "post-look correction of the registered E exclusion (records unchanged)",
           "look_time": rec["time"], "pilot_well_gaps": len(pilot), "rounds": {}}
    for name, spec in ROUNDS.items():
        params = json.loads(spec["params"].read_text())
        raw = scoring.load_runs("pre", ["E"], {"E": list(spec["fresh"])})
        p = scoring.product_monthly(raw, params)
        seen = gaps_of(spec["seen"])
        key = list(zip(p.sid, p.stratum, strict=True))
        as_reg = p[[k not in seen for k in key]]
        consistent = p[[(k not in seen) and (k not in pilot) for k in key]]
        res = {}
        for lab, sub in (("as_registered", as_reg), ("pilot_also_excluded", consistent)):
            ver = stratum_verdicts(sub[sub.clean.astype(bool)])
            res[lab] = {s: brief(v) for s, v in ver.items()}
            res[lab]["n_well_gaps"] = {s: int(sub[sub.stratum == s].sid.nunique())
                                       for s in ("E_lead66", "E_trail48")}
        reg = json.loads(spec["registered"].read_text())
        res["registered_record_coverage"] = {s: v["coverage_gate"]["coverage"]
                                             for s, v in reg["strata_clean"].items()}
        fresh_keys = set(key)
        res["fresh_well_gaps_also_in_pilot"] = {
            s: int(sum(1 for k in fresh_keys - seen if k in pilot and k[1] == s))
            for s in ("E_lead66", "E_trail48")}
        out["rounds"][name] = res
    # pool of never-used E well-gaps
    h, Yb, _ = runner.load_base("pre")
    obs = np.isfinite(Yb)
    targets = np.where(obs.mean(1) >= C.TARGET_MIN_COV)[0]
    lead_ok = [h.sids[i] for i in targets if obs[i, :66].mean() >= 0.8
               and obs[i, 66:].sum() >= 36]
    trail_ok = [h.sids[i] for i in targets if obs[i, -48:].mean() >= 0.8
                and obs[i, :-48].sum() >= 36]
    used = pilot | gaps_of(R3.E_R_DESIGN + R3.E_R_V1 + R3.E_R_V2)
    out["e_pool"] = {"lead_eligible": len(lead_ok), "trail_eligible": len(trail_ok),
                     "lead_never_used": sorted(s for s in lead_ok if (s, "E_lead66") not in used),
                     "trail_never_used": sorted(s for s in trail_ok
                                                if (s, "E_trail48") not in used)}
    R3.jdump(out, R3.EV_R3 / "e_freshness.json")
    for name, res in out["rounds"].items():
        for lab in ("as_registered", "pilot_also_excluded"):
            print(name, lab, {s: (round(v["coverage"], 3), round(v["lo"], 3), round(v["hi"], 3),
                                  v["k"], v["pass"]) for s, v in res[lab].items()
                              if s.startswith("E_")})
    print("pool", {k: (v if isinstance(v, int) else len(v)) for k, v in out["e_pool"].items()})


if __name__ == "__main__":
    main()
