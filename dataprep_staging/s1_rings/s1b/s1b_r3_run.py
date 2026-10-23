"""WP S1b repair round r3 (S1br): the declared SECOND LOOK on the repaired products.

    tools/cpu_slot.sh python -I dataprep_staging/s1_rings/s1b/s1b_r3_run.py

Refuses to run unless the declaration ``S1b.r3.rings_v3_second_look.v1`` exists and is intact
(``prereg3.load_prereg``) and the synthetic operating characteristics of this round exist (they
are computed first). Logs the ``second_look_score`` look, then runs the REGISTERED round-1
computation functions of ``s1b_run`` (``compute``, ``gnss_row``, ``insar_site``, ``d1``;
unchanged, their file hashes are recorded) on rings_bridged/v3, gnss_clean/full/v4 and
insar_069A/v3 through ``s1b_r3_io.Adapter``, with round 1's PRIMARY configuration and every
constant of ``s1b_const`` unchanged. Then, as declared:

* the bracket readings of ``s1b_r2_core.RULES`` (plan-literal k0, range +-30/40 ppm, mid +-30/40
  ppm; k = 0.5 .. 2.5, k = 2 being S1b's own registered rule) for every family, with S1b's 80 %
  family rule and RMSE clause, and the dependence statistics of (d1);
* the declared report-only variants (GNSS flags 0-1; catalogue-only in-gap steps; SJES excluded;
  both validator artefacts removed; ngl_offfan removed; no annual term; InSAR cells with the M3
  and the plane reference; ring noise term at the residual sd);
* two decompositions: rings_bridged/v2 with the repaired validators, and rings_bridged/v3 with
  the round-1 validators (gnss_clean/full/v2, insar_069A/v2).

Writes ``S1b_evidence/repair_r3/retest/second_look.json`` and ``second_look_rows.csv``. Nothing
here is a registered test (the units were scored in round 1): every number is a second look,
conditional where it passes, never a gate, and nothing is fed back into S1a.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from s1_rings.s1b import s1b_const as K  # noqa: E402
from s1_rings.s1b import s1b_io as IO1  # noqa: E402
from s1_rings.s1b import s1b_r2_core as R  # noqa: E402
from s1_rings.s1b import s1b_r2_readings as RD  # noqa: E402
from s1_rings.s1b import s1b_r3_io as IO3  # noqa: E402
from s1_rings.s1b import s1b_run as RUN  # noqa: E402

OUT = K.EVIDENCE / "repair_r3" / "retest"
SYN = K.EVIDENCE / "repair_r3" / "synthetic"
DECL_ID = "S1b.r3.rings_v3_second_look.v1"
PRIMARY = dict(RUN.PRIMARY)
PLAN_LITERAL = R.PLAN_LITERAL
READ_FAMILIES = ("c_wra", "c_ngl", "c_neighbour", "d_insar")

VARIANTS = {
    "gnss_flags_0_1": {"gnss_flags": (0, 1)},
    "gnss_catalogue_steps_only_in_gap": {"steps_mode": "catalogue_in_gap"},
    "plan_flagged_SJES_excluded": {"exclude_stations": ("SJES",)},
    "validator_artefacts_removed": {"exclude_stations": ("SJES",),
                                    "steps_mode": "catalogue_in_gap"},
    "gnss_ngl_offfan_removed": {"common_mode": "ngl_offfan"},
    "no_seasonal": {"seasonal": False},
    "insar_cells_M3": {"insar_source": "cells"},
}


class AdapterV2Validators(IO3.Adapter):
    """rings from ``rings_dir``; GNSS and InSAR from the round-1 validators (gnss_clean/full/v2,
    insar_069A/v2) with the round-1 pixel rule."""
    insar_pixel_series = staticmethod(IO1.insar_pixel_series)
    insar_cell_series = staticmethod(IO1.insar_cell_series)

    def gnss(self):
        return IO1.gnss()

    def insar(self):
        return IO1.insar()


def readings_block(res: dict) -> dict:
    """Bracket readings, RMSE clause and k* for every family of one ``compute`` result."""
    fams = {t: RD.family_block(res["rows"][t]) for t in READ_FAMILIES}
    verdict = {}
    for rule in R.RULES:
        fr = fams["c_wra"]
        c_ok = bool(fr["rules"][rule]["primary"]["containment_pass"]
                    and fr["rmse_rule_holds"]["mid"])
        d2 = fams["d_insar"]
        d2_ok = bool(d2["rules"][rule]["primary"]["containment_pass"]
                     and d2["rmse_rule_holds"]["mid"])
        verdict[rule] = {"c_wra_family_pass": c_ok,
                         "c_wra_containment": fr["rules"][rule]["primary"]["n_contain"],
                         "c_wra_n": fr["rules"][rule]["primary"]["n"],
                         "c_wra_raw_containment": fr["rules"][rule]["raw"]["n_contain"],
                         "d2_family_pass": d2_ok,
                         "d2_containment": d2["rules"][rule]["primary"]["n_contain"],
                         "d2_n": d2["rules"][rule]["primary"]["n"],
                         "d1_pass": bool(res["d1"]["pass"]),
                         "conjunction_c_and_d1": bool(c_ok and res["d1"]["pass"])}
    return {"families": fams, "verdict_by_reading": verdict}


def d1_stats(res: dict) -> dict:
    rows = res["rows"]["d_insar"]
    return {"rates": RD.d1_block(rows), "changes": RD.d1_block(rows, use_change=True),
            "without_beichen": RUN.d1(rows, drop=("北辰國小",)),
            "pilot_9_sites": RUN.d1(rows, drop=tuple(
                s for s in K.SITES if s not in ("元長國小", "客厝國小", "內寮派駐站", "土庫國中",
                                                "秀潭國小", "宏崙國小", "光復國小", "拯民國小",
                                                "嘉興國小")))}


def slim(res: dict) -> dict:
    return {"config": res["config"], "pairs": res["pairs"], "verdicts": res["verdicts"],
            "d1": res["d1"], "gate_members_registered_rule": res["gate_members"],
            "gate_pass_registered_rule": res["gate_pass"], "rows": res["rows"]}


def with_ring_sd(rings: dict, sd: dict) -> dict:
    out = {}
    for s, r in rings.items():
        q = dict(r)
        q["sd_s_mm"] = float(sd[s])
        out[s] = q
    return out


def run() -> dict:
    from harness import prereg3

    from s1_rings.s1b import s1b_r3_looks as LK
    doc, sha = prereg3.load_prereg(DECL_ID)
    bad = prereg3.changed_inputs(doc)
    if bad:
        raise SystemExit(f"declared inputs or code changed since {DECL_ID}: {bad}")
    for f in ("oc_N1_sd_s.json", "oc_N2_resid.json"):
        if not (SYN / f).exists():
            raise SystemExit(f"synthetic operating characteristics missing: {SYN / f}")
    LK.second_look()
    prev = IO3.install(IO3.Adapter())
    try:
        rings, n_r = IO3.rings(IO3.RINGS_V3)
        xy = IO1.site_xy()
        g = IO3.gnss(IO3.GNSS_FULL_V4)
        ins = IO3.insar(IO3.INSAR_V3)
        notes = n_r + g["notes"] + ins["notes"]
        base = RUN.compute(PRIMARY, rings, xy, g, ins)
        out = {"declaration": DECL_ID, "declaration_sha256": sha,
               "label": "SECOND LOOK on spent units (S1b round-1 units); conditional where it "
                        "passes; never a gate; nothing fed back into S1a",
               "inputs": {"rings": "rings_bridged/v3", "gnss": "gnss_clean/full/v4",
                          "insar": "insar_069A/v3"},
               "guard_notes": notes,
               "raw_check_max_abs_mm": max(abs(r["raw_check_mm"]) for r in rings.values()),
               "primary": {**slim(base), **readings_block(base), "d1_stats": d1_stats(base)}}
        out["variants"] = {}
        for name, over in VARIANTS.items():
            res = RUN.compute({**PRIMARY, **over}, rings, xy, g, ins)
            out["variants"][name] = {**slim(res), **readings_block(res)}
        res = RUN.compute({**PRIMARY, "insar_source": "cells"}, rings, xy, g,
                          IO3.with_plane_cells(ins))
        out["variants"]["insar_cells_plane"] = {**slim(res), **readings_block(res),
                                                "d1_stats": d1_stats(res)}
        out["variants"]["insar_cells_M3"]["d1_stats"] = d1_stats(
            RUN.compute({**PRIMARY, "insar_source": "cells"}, rings, xy, g, ins))
        rsd = {s: IO3.ring_resid_sd(r) for s, r in rings.items()}
        res = RUN.compute(PRIMARY, with_ring_sd(rings, rsd), xy, g, ins)
        out["variants"]["ring_noise_resid_sd"] = {**slim(res), **readings_block(res),
                                                  "ring_resid_sd_mm": rsd}
        # decomposition 1: v2 product, repaired validators
        rings2, _ = IO3.rings(IO3.RINGS_V2)
        res = RUN.compute(PRIMARY, rings2, xy, g, ins)
        out["decomposition"] = {"v2_product_repaired_validators": {**slim(res),
                                                                   **readings_block(res)}}
    finally:
        IO3.restore(prev)
    # decomposition 2: v3 product, round-1 validators
    prev = IO3.install(AdapterV2Validators())
    try:
        g2 = IO1.gnss()
        ins2 = IO1.insar()
        res = RUN.compute(PRIMARY, rings, xy, g2, ins2)
        out["decomposition"]["v3_product_round1_validators"] = {**slim(res),
                                                                **readings_block(res)}
    finally:
        IO3.restore(prev)
    OUT.mkdir(parents=True, exist_ok=True)
    RUN.dump(out, OUT / "second_look.json")
    RUN.rows_frame(base).to_csv(OUT / "second_look_rows.csv", index=False)
    return out


def summary(out: dict) -> str:
    lines = []
    p = out["primary"]
    for rule in R.RULES:
        v = p["verdict_by_reading"][rule]
        lines.append(f"{rule:14s} c_wra {v['c_wra_containment']}/{v['c_wra_n']} (raw "
                     f"{v['c_wra_raw_containment']}) pass {v['c_wra_family_pass']} | d2 "
                     f"{v['d2_containment']}/{v['d2_n']} pass {v['d2_family_pass']} | conj "
                     f"{v['conjunction_c_and_d1']}")
    lines.append("d1 " + json.dumps(p["d1"], default=str, ensure_ascii=False)[:400])
    lines.append("rmse c_wra " + json.dumps({k: round(x, 1) for k, x in
                                             p["families"]["c_wra"]["rmse_mm"].items()}))
    for name, v in {**out["variants"], **out["decomposition"]}.items():
        vv = v["verdict_by_reading"]
        lines.append(f"[{name}] " + " ".join(
            f"{r}:{vv[r]['c_wra_containment']}/{vv[r]['c_wra_n']}" for r in
            ("k:0", "range_ppm:30", "range_ppm:40", "mid_ppm:30", "mid_ppm:40", "k:1", "k:2"))
            + f" d1 r {v['d1'].get('r_bridged', float('nan')):.3f}/"
              f"{v['d1'].get('r_raw', float('nan')):.3f}")
    return "\n".join(lines)


def main() -> int:
    out = run()
    print(summary(out))
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    os._exit(rc)

