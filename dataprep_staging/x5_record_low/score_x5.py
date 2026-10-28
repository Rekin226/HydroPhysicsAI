"""Score X5.record_low_tf.v1 once (design-time label): ring-band compaction of the 2020-21
record-low years predicted by the two-input transfer-function model fitted on 2014-2019,
against persistence and the linear head model.

Declared rule (x5const): a site passes when its TF RMSE (pooled over its scored bands and
months, change since the anchor survey) is <= 0.80 x the persistence RMSE AND <= 0.80 x the
linear-head RMSE; success when >= 10 of the 14 ring sites pass. Declared alternatives,
reported and never gating: the VEP-form TF (A = A0, tau_e = 0: the twin column's form plus a
trend) and the nest-mean head input.

Refuses to run if any registered input changed since registration, and refuses to overwrite
results/dataprep/reports/X5.record_low_tf.v1/report.json.

Usage: python score_x5.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import x5const as K  # noqa: E402
import x5lib as L  # noqa: E402
from harness import prereg3  # noqa: E402

OUT = K.OUT / "score"
PARAMS = ("A", "A0", "B", "tau_e_days", "tau_v_days", "r_m_yr", "no_new_low_in_fit")


def band_row(r: dict, key: str) -> dict:
    a = r[key]
    row = {"site": r["site"], "band": r["band"], "input": key, "well": r["well"],
           "well_depth_m": r["well_depth_m"], "scorable": a["scorable"],
           "why": a.get("why", "")}
    if not a["scorable"]:
        return row
    row.update({"n_fit": a["n_fit"], "n_test": a["n_test"], "anchor": a["anchor"],
                "D_fit_m": a["D_fit_m"], "D_test_new_m": a["D_test_new_m"],
                "pers_rate_mm_yr": a["pers_rate_m_yr"] * 1000,
                "lin_k_mm_per_m": a["lin"]["k_m_per_m"] * 1000,
                "lin_r_mm_yr": a["lin"]["r_m_yr"] * 1000})
    for p in PARAMS:
        row[f"tf_{p}"] = a["tf"][p]
        row[f"vep_{p}"] = a["vep"][p]
    for m in ("tf", "vep", "lin", "pers"):
        row[f"rmse_{m}_mm"] = L.rmse(a["errors"][m]) * 1000
        row[f"end_err_{m}_mm"] = float(a["errors"][m][-1]) * 1000
    return row


def decide(sites: pd.DataFrame, meth: str = "tf") -> dict:
    ok = ((sites[f"rmse_{meth}_mm"] <= K.IMPROVE * sites.rmse_pers_mm)
          & (sites[f"rmse_{meth}_mm"] <= K.IMPROVE * sites.rmse_lin_mm)
          & (sites.n_bands_scored > 0))
    n = int(ok.sum())
    return {"n_sites_pass": n, "of": K.N_SITES, "sites_pass": sites.site[ok].tolist(),
            "success": bool(n >= K.MIN_PASS_SITES),
            "n_beats_pers": int((sites[f"rmse_{meth}_mm"] <= K.IMPROVE * sites.rmse_pers_mm)
                                .sum()),
            "n_beats_lin": int((sites[f"rmse_{meth}_mm"] <= K.IMPROVE * sites.rmse_lin_mm)
                               .sum())}


def main() -> int:
    doc, sha = prereg3.load_prereg(K.PREREG_ID)
    bad = prereg3.changed_inputs(doc)
    if bad:
        raise SystemExit(f"registered inputs changed: {bad}")
    path = K.REPORT_DIR / "report.json"
    if path.exists():
        raise SystemExit(f"{path} exists; scored once")
    recs, months = L.build_inputs()
    for r in recs:
        r["primary"] = L.analyse_band(r["h_primary"], r["obs"], months)
        r["nest"] = L.analyse_band(r["h_nest"], r["obs"], months)
    OUT.mkdir(parents=True, exist_ok=True)
    bands = pd.DataFrame([band_row(r, k) for r in recs for k in ("primary", "nest")])
    bands.to_csv(OUT / "bands.csv", index=False)
    s_p = L.site_scores(recs, "primary")
    s_n = L.site_scores(recs, "nest")
    s_p.to_csv(OUT / "sites_primary.csv", index=False)
    s_n.to_csv(OUT / "sites_nest.csv", index=False)
    primary = decide(s_p, "tf")
    syn = K.OUT / "synthetic" / "synthetic_check.json"
    syn_pass = json.loads(syn.read_text())["pass"] if syn.exists() else None
    bp = bands[(bands.input == "primary") & bands.scorable.astype(bool)]
    by_band = {}
    mid = bp.band.map(L.band_mid_depth)
    cls = np.where(mid < 100, "shallow", np.where(mid < 200, "mid", "deep"))
    for j, g in bp.groupby(cls):
        by_band[j] = {"n": int(len(g)),
                      "n_tf_beats_both": int(((g.rmse_tf_mm <= K.IMPROVE * g.rmse_pers_mm)
                                              & (g.rmse_tf_mm <= K.IMPROVE * g.rmse_lin_mm))
                                             .sum()),
                      "median_rmse_mm": {m: float(g[f"rmse_{m}_mm"].median())
                                         for m in ("tf", "vep", "lin", "pers")},
                      "n_new_low_in_fit": int((~g.tf_no_new_low_in_fit.astype(bool)).sum()),
                      "median_D_test_new_m": float(g.D_test_new_m.median())}
    res = {
        "declaration": K.PREREG_ID, "declaration_sha256": sha, "wp": K.WP,
        "label": "design-time evidence for response shape (rings 2012-2021 were design-time "
                 "data for the twin column; no fresh units); never validation",
        "rule": f"site passes if TF RMSE <= {K.IMPROVE} x persistence AND <= {K.IMPROVE} x "
                f"linear-head; success if >= {K.MIN_PASS_SITES} of {K.N_SITES} sites",
        "primary": primary,
        "sites": s_p.to_dict(orient="records"),
        "by_band_primary": by_band,
        "n_bands_scored_primary": int(len(bp)),
        "alternatives_reported_not_gating": {
            "vep_form_primary_input": decide(s_p, "vep"),
            "tf_nest_mean_input": decide(s_n, "tf"),
            "vep_form_nest_mean_input": decide(s_n, "vep")},
        "synthetic_check_pass": syn_pass,
        "files": {"bands": str(OUT / "bands.csv"), "sites_primary": str(OUT / "sites_primary.csv"),
                  "sites_nest": str(OUT / "sites_nest.csv")},
    }
    res["decision"] = (
        "declared success met (design-time label)" if primary["success"] else
        "declared success not met (design-time label)")
    print(json.dumps(res, indent=1, ensure_ascii=False, default=float))
    K.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise SystemExit(f"{path} exists; scored once")
    path.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float) + "\n",
                    encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
