"""WP S1b repair round r3, fix round: the declared after-the-fact GNSS step-policy grid.

    tools/cpu_slot.sh python -I dataprep_staging/s1_rings/s1b/s1b_r3f_symmetry.py

The r3 verifier found the r3 variant "both GNSS artefacts removed" (8/9 under range +-30/40 ppm)
one-sided: it removed only the two v4 step effects that put the product outside the bracket and
kept the v4 step decisions that put it inside (Tuku). This script computes the declared grid of
``S1b.r3f.step_policy_symmetry.v1`` (``reports/S1b.repair.md``, Fix round F.2):

    validator version V in {gnss_clean/full/v2, gnss_clean/full/v4}
    x step policy P in {product, catalogue_in_gap, catalogue_all}, SJES excluded
    (+ catalogue_all with SJES kept),

through the REGISTERED round-1 functions of ``s1b_run`` (unchanged) with rings_bridged/v3 and
round 1's PRIMARY configuration. ``catalogue_all`` adds every step whose basis is not
metadata_or_catalogue back to the daily heights at every date (the registered add-back rule of
``s1b_io.gnss_series``, extended from the gap to every date) and keeps only catalogue steps in
the step table. Refuses to run unless the declaration exists and none of its inputs changed.
Logs the ``after_the_fact`` look before reading any product.

Writes ``S1b_evidence/repair_r3/fix_round/step_policy_grid.json``. Every number is after the
fact on spent units, report only, never a gate; nothing is fed back into S1a.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from s1_rings.s1b import s1b_const as K  # noqa: E402
from s1_rings.s1b import s1b_io as IO1  # noqa: E402
from s1_rings.s1b import s1b_r2_core as R  # noqa: E402
from s1_rings.s1b import s1b_r2_readings as RD  # noqa: E402
from s1_rings.s1b import s1b_r3_after as AF  # noqa: E402
from s1_rings.s1b import s1b_r3_io as IO3  # noqa: E402
from s1_rings.s1b import s1b_run as RUN  # noqa: E402

OUT = K.EVIDENCE / "repair_r3" / "fix_round"
SYN = K.EVIDENCE / "repair_r3" / "synthetic"
DECL_ID = "S1b.r3f.step_policy_symmetry.v1"
PRIMARY = dict(RUN.PRIMARY)
CATALOGUE = "metadata_or_catalogue"
WRA10 = ("YCES", "GFES", "TKJS", "HLES", "KTES", "JMES", "SSES", "HNES", "SJES", "STES")
POLICIES = ("product", "catalogue_in_gap", "catalogue_all")
VERSIONS = ("v2", "v4")
READINGS = ("k:0", "range_ppm:30", "range_ppm:40", "mid_ppm:30", "mid_ppm:40", "k:1", "k:2")


def catalogue_all(g: dict) -> dict:
    """``g`` with every non-catalogue step added back to ``value`` from its date onwards and
    removed from the step table (so it carries no variance term)."""
    d = g["daily"].copy()
    st = g["steps"]
    drop = st["basis"] != CATALOGUE
    v = d["value"].to_numpy(float).copy()
    for r in st[drop].itertuples():
        m = ((d["family"] == r.family) & (d["station"] == r.station)
             & (d["date"] >= pd.Timestamp(r.date))).to_numpy()
        v[m] += float(r.size_m)
    d["value"] = v
    return {**g, "daily": d, "steps": st[~drop].reset_index(drop=True)}


def addback_integrity(g: dict) -> dict:
    """Per WRA station of the 10: max |value + correction_m - value_raw| and max |correction_m -
    sum of step sizes dated on or before the day| (mm)."""
    out = {}
    d, st = g["daily"], g["steps"]
    for s in WRA10:
        x = d[(d.family == "wra") & (d.station == s)].sort_values("date")
        t = pd.DatetimeIndex(x["date"])
        corr = np.zeros(len(x))
        for r in st[(st.family == "wra") & (st.station == s)].itertuples():
            corr += float(r.size_m) * (t >= pd.Timestamp(r.date))
        a = x["value"].to_numpy(float) + x["correction_m"].to_numpy(float) \
            - x["value_raw"].to_numpy(float)
        b = x["correction_m"].to_numpy(float) - corr
        out[s] = {"max_abs_value_plus_corr_minus_raw_mm": float(1000 * np.nanmax(np.abs(a))),
                  "max_abs_corr_minus_step_sum_mm": float(1000 * np.nanmax(np.abs(b))),
                  "n_days": int(len(x))}
    return out


def cell_rows(res: dict) -> list[dict]:
    rows = []
    for r in res["rows"]["c_wra"]:
        o = {"site_en": r["site_en"], "station": r["station"],
             "evaluable": bool(r.get("evaluable"))}
        if r.get("evaluable"):
            L = 1000.0 * r["L_m"]
            o.update({k: r.get(k) for k in ("C_primary_mm", "C_raw_mm", "C_pre_mm",
                                            "C_post_mm", "sigma_pre_mm", "sigma_post_mm",
                                            "G_mm", "excess_rate_pre", "excess_rate_post")})
            for rule in READINGS:
                o[f"in_{rule}"] = bool(R.contains(rule, r["C_primary_mm"], r["C_pre_mm"],
                                                  r["C_post_mm"], r["sigma_pre_mm"],
                                                  r["sigma_post_mm"], L))
        rows.append(o)
    return rows


def cell(res: dict, oc_fam: str, ocs: dict) -> dict:
    fb = RD.family_block(res["rows"]["c_wra"])
    rc = {rule: {"n": fb["rules"][rule]["primary"]["n"],
                 "count": fb["rules"][rule]["primary"]["n_contain"],
                 "raw_count": fb["rules"][rule]["raw"]["n_contain"]} for rule in R.RULES}
    rmse_ok = bool(fb["rmse_rule_holds"]["mid"])
    readings = {}
    for rule in READINGS:
        p = fb["rules"][rule]["primary"]
        readings[rule] = {"count": p["n_contain"], "n": p["n"],
                          "raw_count": fb["rules"][rule]["raw"]["n_contain"],
                          "containment_pass": bool(p["containment_pass"]),
                          "family_pass": bool(p["containment_pass"] and rmse_ok)}
    lr = {}
    for m, oc in ocs.items():
        tab = AF.lr_table({"wra": rc, "wra_noSJES": rc, "insar": rc}, oc)[oc_fam]
        lr[m] = {rule: {"n_obs": tab[rule]["n_obs"],
                        "n_sites_synthetic": tab[rule]["n_sites_synthetic"],
                        "p_eq_correct": tab[rule]["correct"]["p_eq"],
                        "p_eq_up25": tab[rule]["up25"]["p_eq"],
                        "p_eq_down25": tab[rule]["down25"]["p_eq"],
                        "LR_correct_vs_up25": tab[rule]["LR_correct_vs_up25"],
                        "LR_correct_vs_down25": tab[rule]["LR_correct_vs_down25"],
                        "LR_mixdrawn_vs_up25": tab[rule]["LR_mixdrawn_vs_up25"]}
                 for rule in READINGS}
    return {"rmse_mm": fb["rmse_mm"], "rmse_rule_holds": rmse_ok, "readings": readings,
            "lr": lr, "synthetic_family": oc_fam, "per_site": cell_rows(res)}


def summarise(cells: dict) -> dict:
    """Min and max count per reading over the six SJES-excluded cells, and whether the r3
    statement may stand (declared rule 2)."""
    six = {k: v for k, v in cells.items() if k.endswith("|noSJES")}
    out = {}
    for rule in READINGS:
        c = [v["readings"][rule]["count"] for v in six.values()]
        out[rule] = {"min": int(min(c)), "max": int(max(c)),
                     "n_cells_family_pass": int(sum(v["readings"][rule]["family_pass"]
                                                    for v in six.values())),
                     "n_cells": len(six)}
    lrs = []
    for v in six.values():
        for m in v["lr"]:
            for rule in ("range_ppm:30", "range_ppm:40"):
                for key in ("LR_correct_vs_up25", "LR_correct_vs_down25"):
                    x = v["lr"][m][rule][key]
                    if x is not None:
                        lrs.append(x)
    stand = all(six[k]["readings"][r]["family_pass"] for k in six
                for r in ("range_ppm:30", "range_ppm:40"))
    return {"counts_over_six_cells": out,
            "lr_range_ppm_readings_over_six_cells": [float(min(lrs)), float(max(lrs))]
            if lrs else None,
            "r3_artefact_statement_may_stand": bool(stand)}


def run() -> dict:
    from harness import prereg3

    from s1_rings.s1b import s1b_r3f_looks as LK
    doc, sha = prereg3.load_prereg(DECL_ID)
    bad = prereg3.changed_inputs(doc)
    if bad:
        raise SystemExit(f"declared inputs or code changed since {DECL_ID}: {bad}")
    LK.fix_symmetry()
    ocs = {m: json.loads((SYN / f"oc_{m}.json").read_text()) for m in AF.MODELS}
    xy = IO1.site_xy()
    prev = IO3.install(IO3.Adapter())
    try:
        rings, notes = IO3.rings(IO3.RINGS_V3)
        gv = {"v4": IO3.gnss(IO3.GNSS_FULL_V4)}
        ins4 = IO3.insar(IO3.INSAR_V3)
    finally:
        IO3.restore(prev)
    gv["v2"] = IO1.gnss()
    ins2 = IO1.insar()
    out = {"declaration": DECL_ID, "declaration_sha256": sha,
           "label": "after the fact on spent S1b units; report only; never a gate; nothing "
                    "fed back into S1a",
           "inputs": {"rings": "rings_bridged/v3",
                      "gnss": {"v2": "gnss_clean/full/v2", "v4": "gnss_clean/full/v4"}},
           "guard_notes": notes + gv["v4"]["notes"] + gv["v2"]["notes"],
           "addback_integrity": {v: addback_integrity(gv[v]) for v in VERSIONS},
           "cells": {}}
    for v in VERSIONS:
        ins = ins4 if v == "v4" else ins2
        adapter = IO3.Adapter() if v == "v4" else _AdapterV2()
        prev = IO3.install(adapter)
        try:
            for p in POLICIES:
                g = catalogue_all(gv[v]) if p == "catalogue_all" else gv[v]
                mode = "catalogue_in_gap" if p == "catalogue_in_gap" else "product"
                cfg = {**PRIMARY, "steps_mode": mode, "exclude_stations": ("SJES",)}
                res = RUN.compute(cfg, rings, xy, g, ins)
                out["cells"][f"{v}|{p}|noSJES"] = cell(res, "wra_noSJES", ocs)
                if p == "catalogue_all":
                    res = RUN.compute({**PRIMARY, "steps_mode": "product"}, rings, xy, g, ins)
                    out["cells"][f"{v}|{p}|withSJES"] = cell(res, "wra", ocs)
        finally:
            IO3.restore(prev)
    out["summary"] = summarise(out["cells"])
    OUT.mkdir(parents=True, exist_ok=True)
    RUN.dump(out, OUT / "step_policy_grid.json")
    return out


class _AdapterV2(IO3.Adapter):
    """Round-1 validators (gnss_clean/full/v2, insar_069A/v2) with the round-1 pixel rule."""
    insar_pixel_series = staticmethod(IO1.insar_pixel_series)
    insar_cell_series = staticmethod(IO1.insar_cell_series)

    def gnss(self):
        return IO1.gnss()

    def insar(self):
        return IO1.insar()


def report(out: dict) -> str:
    lines = []
    for name, c in out["cells"].items():
        s = " ".join(f"{r.split(':')[0][:5]}{r.split(':')[1]}={c['readings'][r]['count']}/"
                     f"{c['readings'][r]['n']}{'P' if c['readings'][r]['family_pass'] else ''}"
                     for r in READINGS)
        lines.append(f"{name:28s} rmse {c['rmse_mm']['primary_vs_mid']:.1f}/"
                     f"{c['rmse_mm']['raw_vs_mid']:.1f} {s}")
        for o in c["per_site"]:
            if o["evaluable"]:
                lines.append(f"    {o['station']} C {o['C_primary_mm']:.1f} "
                             f"[{o['C_pre_mm']:.1f}, {o['C_post_mm']:.1f}] "
                             f"r30 {int(o['in_range_ppm:30'])} r40 {int(o['in_range_ppm:40'])} "
                             f"k0 {int(o['in_k:0'])}")
    lines.append(json.dumps(out["summary"]))
    return "\n".join(lines)


def main() -> int:
    out = run()
    print(report(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
