"""WP S1b repair round r3 (S1br): diagnosis numbers from S1b's own round-1 and r2 evidence.

    python -I dataprep_staging/s1_rings/s1b/s1b_r3_diagnose.py

Reads only S1b's evidence files (no product): ``S1b_evidence/registered_rows.csv``,
``synthetic_operating_characteristics_v2.json`` and ``repair/{readings,
synthetic_plan_literal, u1_and_sensitivity}.json``. The look (``s1b_r3_looks.py diagnose``) is
logged before this script runs. Writes ``S1b_evidence/repair_r3/diag/diagnosis.json``:

1. per WRA site: the bracket ends, the plan-literal readings, the sigma decomposition
   (excess-rate fit, endpoints, ring survey noise, GNSS step se) and the size of a 25 %
   group-A scale error in mm and in sigma;
2. the r2 synthetic expectations (seed 1803) for EVERY scenario (correct, +-25 %, other, raw,
   mixture), expected sites inside and family pass rates, and the real counts beside them;
3. ring noise: sd_s against the ring-fit residual sd (r2 sensitivity);
4. (d1): the registered correlations and their dependence statistics;
5. power of the registered 2-sigma rule against +-25 % (round-1 synthetic).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from s1_rings.s1b import s1b_const as K  # noqa: E402
from s1_rings.s1b import s1b_r2_core as R  # noqa: E402
from s1_rings.s1b.s1b_run import dump  # noqa: E402

EV = K.EVIDENCE
OUT = EV / "repair_r3" / "diag"
A_PPM = K.REF_PPM["A"][0]


def wra_sites(df: pd.DataFrame) -> list[dict]:
    out = []
    for r in df[(df.test == "c_wra") & df.evaluable.astype(str).str.lower().eq("true")
                ].itertuples():
        L = 1000.0 * r.L_m
        var_end = float(r.var_end_inst)
        var_ring = 2.0 * float(r.sd_s_mm) ** 2
        var_steps = float(r.var_steps)
        var_e = {}
        for w in ("pre", "post"):
            s = float(getattr(r, f"sigma_{w}_mm"))
            var_e[w] = s ** 2 - float(r.var_common)
        d25 = 0.25 * A_PPM * 1e-6 * L
        s_mean = math.sqrt(0.5 * (r.sigma_pre_mm ** 2 + r.sigma_post_mm ** 2))
        row = {
            "site": r.site, "site_en": r.site_en, "station": r.station, "L_m": r.L_m,
            "p_A": r.p_A, "assigned": r.assigned if isinstance(r.assigned, str) else "",
            "C_primary_mm": r.C_primary_mm, "C_A_mm": r.C_A_mm, "C_B_mm": r.C_B_mm,
            "C_raw_mm": r.C_raw_mm, "C_pre_mm": r.C_pre_mm, "C_post_mm": r.C_post_mm,
            "sigma_pre_mm": r.sigma_pre_mm, "sigma_post_mm": r.sigma_post_mm,
            "range_half_mm": 0.5 * abs(r.C_pre_mm - r.C_post_mm),
            "range_half_ppm": 1e6 * 0.5 * abs(r.C_pre_mm - r.C_post_mm) / L,
            "sigma_mean_ppm": 1e6 * s_mean / L,
            "excess_rate_pre": r.excess_rate_pre, "excess_rate_post": r.excess_rate_post,
            "excess_rate_change": r.excess_rate_post - r.excess_rate_pre,
            "gap_yr": r.gap_yr,
            "var_share": {
                "excess_fit_pre": var_e["pre"] / r.sigma_pre_mm ** 2,
                "excess_fit_post": var_e["post"] / r.sigma_post_mm ** 2,
                "endpoints_mean": var_end / s_mean ** 2,
                "ring_noise_mean": var_ring / s_mean ** 2,
                "gnss_steps_mean": var_steps / s_mean ** 2},
            "sd_terms_mm": {"excess_fit_pre": math.sqrt(max(var_e["pre"], 0.0)),
                            "excess_fit_post": math.sqrt(max(var_e["post"], 0.0)),
                            "endpoints": math.sqrt(var_end), "ring_noise_2sd_s":
                            math.sqrt(var_ring), "gnss_steps": math.sqrt(var_steps)},
            "scale25_mm": d25, "scale25_over_sigma": d25 / s_mean,
            "min_k_primary": R.min_k(r.C_primary_mm, r.C_pre_mm, r.C_post_mm,
                                     r.sigma_pre_mm, r.sigma_post_mm),
            "primary_minus_nearest_end_mm": (
                0.0 if min(r.C_pre_mm, r.C_post_mm) <= r.C_primary_mm <= max(r.C_pre_mm,
                                                                               r.C_post_mm)
                else (r.C_primary_mm - max(r.C_pre_mm, r.C_post_mm)
                      if r.C_primary_mm > max(r.C_pre_mm, r.C_post_mm)
                      else r.C_primary_mm - min(r.C_pre_mm, r.C_post_mm))),
        }
        for rule in R.RULES:
            row[f"in_{rule}"] = R.contains(rule, r.C_primary_mm, r.C_pre_mm, r.C_post_mm,
                                           r.sigma_pre_mm, r.sigma_post_mm, L)
            row[f"A_in_{rule}"] = R.contains(rule, r.C_A_mm, r.C_pre_mm, r.C_post_mm,
                                             r.sigma_pre_mm, r.sigma_post_mm, L)
        out.append(row)
    return out


def synthetic_table(syn: dict, readings: dict) -> dict:
    """Expected sites inside (sum over sites of the per-site containment rates) and family
    pass rates of the r2 synthetic run, for every scenario, with the real count beside."""
    ps = [p for p in syn["per_site"] if p["family"] == "wra_colocated"]
    fam = syn["families"]["wra_colocated"]
    real = readings["families"]["c_wra"]["rules"]
    out = {}
    for rule in R.RULES:
        e = {}
        for sc in ("correct", "up25", "down25", "other", "raw"):
            v = [p.get(f"{rule}|{sc}") for p in ps]
            e[sc] = float(np.sum([x for x in v if x is not None]))
        out[rule] = {"expected_inside": e,
                     "family_pass": {sc: fam.get(f"{rule}|{sc}") for sc in
                                     ("correct", "up25", "down25", "other", "raw", "mix")},
                     "real_primary": real[rule]["primary"]["n_contain"],
                     "real_raw": real[rule]["raw"]["n_contain"], "n": len(ps)}
    return out


def main() -> int:
    df = pd.read_csv(EV / "registered_rows.csv")
    readings = json.loads((EV / "repair" / "readings.json").read_text())
    syn2 = json.loads((EV / "repair" / "synthetic_plan_literal.json").read_text())
    u1 = json.loads((EV / "repair" / "u1_and_sensitivity.json").read_text())
    oc = json.loads((EV / "synthetic_operating_characteristics_v2.json").read_text())
    sites = wra_sites(df)
    out = {"wra_sites": sites, "synthetic_r2_seed1803": synthetic_table(syn2, readings),
           "ring_resid_sd": u1.get("ring_resid_sd"), "ring_sd_variants": u1.get(
               "ring_sd_variants"),
           "d1_rates": readings["d1_rates"], "d1_changes": readings["d1_changes"],
           "synthetic_d1": {k: v for k, v in syn2["families"]["insar"].items()
                            if k.startswith("d1|")},
           "oc_round1_families": oc["families"],
           "ngl_rows": df[df.test == "c_ngl"].to_dict(orient="records"),
           "neighbour_rows": df[df.test == "c_neighbour"][
               ["site_en", "station", "d_km", "evaluable", "contains_primary"]].to_dict(
               orient="records")}
    agg = {}
    for k in ("excess_fit_pre", "excess_fit_post", "endpoints_mean", "ring_noise_mean",
              "gnss_steps_mean"):
        v = [s["var_share"][k] for s in sites]
        agg[k] = {"median": float(np.median(v)), "min": float(np.min(v)),
                  "max": float(np.max(v))}
    out["var_share_summary"] = agg
    out["scale25_over_sigma"] = {"min": min(s["scale25_over_sigma"] for s in sites),
                                 "median": float(np.median([s["scale25_over_sigma"]
                                                            for s in sites])),
                                 "max": max(s["scale25_over_sigma"] for s in sites)}
    out["range_half_over_sigma"] = [s["range_half_mm"] / math.sqrt(0.5 * (
        s["sigma_pre_mm"] ** 2 + s["sigma_post_mm"] ** 2)) for s in sites]
    OUT.mkdir(parents=True, exist_ok=True)
    dump(out, OUT / "diagnosis.json")
    pd.set_option("display.width", 250)
    t = pd.DataFrame([{k: (round(v, 2) if isinstance(v, float) else v) for k, v in s.items()
                       if not isinstance(v, dict)} for s in sites])
    print(t[["site_en", "station", "L_m", "p_A", "C_primary_mm", "C_A_mm", "C_B_mm",
             "C_raw_mm", "C_pre_mm", "C_post_mm", "sigma_pre_mm", "sigma_post_mm",
             "range_half_mm", "range_half_ppm", "sigma_mean_ppm", "scale25_mm",
             "scale25_over_sigma", "min_k_primary", "primary_minus_nearest_end_mm"]
            ].to_string())
    print(t[["site_en"] + [f"in_{r}" for r in R.RULES]].to_string())
    print(t[["site_en"] + [f"A_in_{r}" for r in R.RULES]].to_string())
    print(pd.DataFrame([{"site": s["site_en"], **{k: round(v, 1) for k, v in
                                                  s["sd_terms_mm"].items()},
                         "dex": round(s["excess_rate_change"], 1)} for s in sites]
                       ).to_string())
    print(json.dumps(agg, indent=0))
    print(json.dumps(out["scale25_over_sigma"]))
    for rule, v in out["synthetic_r2_seed1803"].items():
        print(rule, {k: round(x, 2) for k, x in v["expected_inside"].items()},
              "real", v["real_primary"], "raw", v["real_raw"],
              {k: x for k, x in v["family_pass"].items()})
    print("ring_resid_sd", json.dumps(out["ring_resid_sd"], default=str)[:1500])
    print("ring_sd_variants", json.dumps(out["ring_sd_variants"], default=str)[:1500])
    print("d1", json.dumps({k: out["d1_rates"][k] for k in ("n", "r_bridged", "r_raw",
                                                             "r_bridged_raw", "williams")}))
    print("syn d1", json.dumps(out["synthetic_d1"]))
    print("oc r1", json.dumps(out["oc_round1_families"], default=str)[:2500])
    print("ngl", json.dumps(out["ngl_rows"], default=str)[:800])
    print("neighbour", out["neighbour_rows"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
