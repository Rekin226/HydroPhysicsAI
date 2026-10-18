"""WP S1b repair round r3 (S1br): second after-the-fact diagnostic (logged look).

    python -I dataprep_staging/s1_rings/s1b/s1b_r3_diag_after2.py

Reads only evidence already computed (``repair_r3/retest/second_look.json``,
``repair_r3/synthetic/oc_*.json``). Nothing new is scored on a product. Reports, labelled as
NOT declared in ``S1b.r3.rings_v3_second_look.v1``:

1. InSAR minus GNSS gap change at the co-located WRA sites for the three InSAR series (M3
   pixels, M3 cells, plane cells), with and without Kecuo and Xizhou (the two GNSS artefacts);
2. the declared variant "validator_artefacts_removed" read against the 9-site synthetic family
   (P(N = n_obs | s) and the ratios), which the declaration did not list among its LR rows.

Writes ``repair_r3/retest/diag_after2.json``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np  # noqa: E402

from s1_rings.s1b import s1b_const as K  # noqa: E402
from s1_rings.s1b import s1b_r3_after as AF  # noqa: E402
from s1_rings.s1b.s1b_run import dump  # noqa: E402

RT = K.EVIDENCE / "repair_r3" / "retest"
SYN = K.EVIDENCE / "repair_r3" / "synthetic"
ARTEFACT_SITES = ("客厝國小", "溪州國小")


def stats(d: dict) -> dict:
    v = np.array(list(d.values()), float)
    return {"n": int(v.size), "median": float(np.median(v)), "mean": float(np.mean(v)),
            "rms": float(np.sqrt(np.mean(v ** 2))), "min": float(v.min()), "max": float(v.max())}


def main() -> int:
    from s1_rings.s1b import s1b_r3_looks as LK
    LK.after("insar_reference_vs_gnss_and_artefact_variant_lr")
    sl = json.loads((RT / "second_look.json").read_text())
    g = {r["site"]: r for r in sl["primary"]["rows"]["c_wra"] if r.get("evaluable")}
    series = {"M3_pixels": sl["primary"]["rows"]["d_insar"],
              "M3_cells": sl["variants"]["insar_cells_M3"]["rows"]["d_insar"],
              "plane_cells": sl["variants"]["insar_cells_plane"]["rows"]["d_insar"]}
    out = {"label": "after the fact, not declared; report only", "insar_minus_gnss": {}}
    for name, rows in series.items():
        d = {r["site"]: r["I_gap_mm"] - g[r["site"]]["G_mm"] for r in rows if r["site"] in g}
        d8 = {s: v for s, v in d.items() if s not in ARTEFACT_SITES}
        out["insar_minus_gnss"][name] = {"per_site": {K.SITE_EN[s]: v for s, v in d.items()},
                                         "all_10": stats(d), "without_kecuo_xizhou": stats(d8)}
    v = sl["variants"]["validator_artefacts_removed"]["families"]["c_wra"]["rules"]
    rc = {"wra_noSJES": {rule: {"n": v[rule]["primary"]["n"],
                                "count": v[rule]["primary"]["n_contain"],
                                "raw_count": v[rule]["raw"]["n_contain"]}
                         for rule in v}}
    out["artefact_variant_lr"] = {}
    for m in AF.MODELS:
        oc = json.loads((SYN / f"oc_{m}.json").read_text())
        rcf = {"wra": rc["wra_noSJES"], "wra_noSJES": rc["wra_noSJES"],
               "insar": rc["wra_noSJES"]}
        tab = AF.lr_table(rcf, oc)["wra_noSJES"]
        out["artefact_variant_lr"][m] = tab
    dump(out, RT / "diag_after2.json")
    for name, d in out["insar_minus_gnss"].items():
        print(name, {k: round(x, 1) for k, x in d["per_site"].items()})
        print("   all", {k: round(x, 1) for k, x in d["all_10"].items()},
              "| w/o Kecuo, Xizhou", {k: round(x, 1) for k, x in
                                      d["without_kecuo_xizhou"].items()})
    for m, tab in out["artefact_variant_lr"].items():
        for rule in ("k:0", "range_ppm:30", "range_ppm:40", "mid_ppm:40", "k:1", "k:2"):
            row = tab[rule]
            print(m, rule, "obs", row["n_obs"], "/", row["n_sites_real"],
                  {sc: (round(row[sc]["mean_count"], 2), round(row[sc]["p_eq"], 3),
                        round(row[sc]["family_pass"], 3))
                   for sc in ("correct", "up25", "down25", "mix_drawn")},
                  "LRc/u", row["LR_correct_vs_up25"], "LRc/d", row["LR_correct_vs_down25"],
                  "LRm/u", row["LR_mixdrawn_vs_up25"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
