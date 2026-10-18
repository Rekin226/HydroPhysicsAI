"""WP H2 repair round (v2): constants (post-look; the v1 modules stay unchanged).

The registered v1 test (``H2.fill_acceptance.v1``, 2026-10-13T19:33Z) failed on coverage in
four of 14 strata (RMSE passed in all 14): N_net1 0.812 (one-month network outages, under),
S24 0.939 (24-month gaps, over), E_lead66 0.844 (point window) and E_trail48 0.960 (over).
H, T and d passed. The cause is the sd factors: the one-month network-outage factor (0.738) was
estimated on 15 design clusters, and the edge factors on 79-81 design wells.

Repair (decided after the v1 look, before any v2 realization exists):

* factors per gap class re-estimated with the same conformal rule (q_0.90 of |err| / sd over
  clean cells, divided by z_0.95) on the union of the design realizations r = 0..4 and the v1
  fresh realizations (now design-time); family E well-gaps counted once;
* nothing else changes: router and methods per class (all router), gap classes, envelope,
  bridge form and bridge sds (the bridge passed H2.fresh.d and is carried over unchanged);
* fresh v2 realizations never generated before: N, S r = 30..39; H, T r = 30..49; E r =
  30..59, gated on the well-gaps that none of r = 0..4 or 10..19 contains.

The v1 modules (config, methods, runner, scoring, s4_score, ...) are hashed by the v1
registration and are not edited; the v2 scripts import them.
"""
from __future__ import annotations

from h2_heads_fill import config as C

PREREG_ID_V1 = "H2.fill_acceptance.v1"
PREREG_ID_V2 = "H2.fill_acceptance.v2"
FAMILIES_V2 = ("N", "S", "E", "H", "T")
#: realizations that are design-time for v2 (v1 design + v1 fresh)
DESIGN_R_V2 = {f: list(C.DESIGN_R[f]) + list(C.FRESH_R[f]) for f in FAMILIES_V2}
FRESH_R_V2 = {"N": range(30, 40), "S": range(30, 40), "E": range(30, 60), "H": range(30, 50),
              "T": range(30, 50)}
#: realizations whose E well-gaps are excluded from the v2 E gate
E_SEEN_R = list(C.DESIGN_R["E"]) + list(C.FRESH_R["E"])
DESIGN_DIR_V2 = C.EVIDENCE_DIR / "design_v2"
PARAMS_V1 = C.EVIDENCE_DIR / "design" / "design_params.json"
PARAMS_V2 = DESIGN_DIR_V2 / "design_params_v2.json"
REG_V1 = C.EVIDENCE_DIR / "registered"
REG_V2 = C.EVIDENCE_DIR / "registered_v2"
