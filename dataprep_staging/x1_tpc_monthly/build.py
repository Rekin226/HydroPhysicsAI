"""Task 4: build the product ``electricity_x1_tpc/full/v<N>`` (new version directory).

Usage: python -I build.py CV_JSON TPC_FILE [TPC_FILE ...]

Monthly electricity of the hybrid layer's two series, for 2012-01 to the last TPC month:
* ``county_monthly.parquet``: in-grid Changhua and Yunlin totals (GWh/month);
* ``discs_monthly.parquet``: the sum within 3 km of each of the 158 wells (GWh/month);
* ``parameters.json``: the frozen primary model per county, the disc shares and the sds;
* ``validation.json``, ``PRODUCT_CARD.md``, ``MANIFEST.json``.

Flags (plan 2.4): 0 census recorded month (>= 99 % of active cells with flags 0-5), value
as in the release; 5 downscaled from TPC (``extend.apply``) with a calibrated sd; 8 missing
(no TPC month). Gate: ``gate_spec`` G1-G5.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import extend as X  # noqa: E402
import gate_spec as G  # noqa: E402
import model as M  # noqa: E402
import panel  # noqa: E402
from common.manifest import build_manifest, new_version_dir, write_manifest  # noqa: E402
from common.paths import rel_to_repo  # noqa: E402
from common.source import validate_source  # noqa: E402

PRODUCT = "electricity_x1_tpc/full"
FIT_END = pd.Timestamp("2026-06-01")
START = pd.Timestamp("2012-01-01")
METHOD = "x1_ridge_wx_tpc_downscale_v1"
COUNTIES = ("changhua", "yunlin")
ALLOWED = ["statistical_layer_history", "forcing_recorded_arm", "after_the_fact_analysis"]
FORBIDDEN = ["scoring_target", "baseline", "datum_fit", "prospective_scorer",
             "calibration_target", "initial_conditions", "idw_field", "prior",
             "prior_width_input", "model_design", "selection", "forcing_frozen",
             "design_time_evidence", "validator"]


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(argv: list[str]) -> None:
    cv_path = Path(argv[0])
    tpc_files = [Path(a) for a in argv[1:]]
    cv = json.loads(cv_path.read_text(encoding="utf-8"))
    if cv["gate_spec_sha256"] != sha(HERE / "gate_spec.py"):
        raise SystemExit("gate_spec.py changed after the CV run")
    if cv["selection"]["primary"] != X.PRIMARY:
        raise SystemExit("extend.PRIMARY differs from the CV selection")
    tpc = panel.tpc_wide(tpc_files)
    nat = panel.national_ag()
    wx, wreads = panel.weather()
    cen = panel.census_hybrid()
    last = tpc[["ag_changhua", "ag_yunlin"]].dropna().index.max()
    months = pd.date_range(START, last, freq="MS", name="date")
    rec = cen.recorded.reindex(months, fill_value=False)
    yc = cen.county.reindex(months)
    yd = cen.discs.reindex(months)
    fitmask = rec & (months <= FIT_END)

    params = {"primary": X.PRIMARY, "fit_window": ["2016-01", "2026-06"],
              "fit_months": "census recorded months only (2019-09..2020-06 excluded)",
              "county": {}, "disc_share": {}, "disc_sdrel": {}, "county_sdlog": {},
              "missing_input_rules": X.__doc__.split("Missing-input rules")[1].strip()}
    county_rows, disc_rows, checks = [], [], {}
    for c in COUNTIES:
        f = M.fit(X.PRIMARY, yc[c].where(fitmask), tpc, nat, wx, c)
        p = X.to_params(f)
        params["county"][c] = p
        sdlog = cv["sd_calibration"][c]["sdlog_all_folds"]
        params["county_sdlog"][c] = sdlog
        ds = X.apply(p, tpc, nat, wx, months)
        ref = M.predict(f, months, tpc, nat, wx)
        both = ds["value"].notna() & ref.notna()
        checks[f"apply_equals_fit_{c}"] = float(np.max(np.abs(ds["value"][both]
                                                              - ref[both])))
        cols = cen.well_county.index[cen.well_county == c]
        sh = M.shares(yd.loc[fitmask, cols], yc.loc[fitmask, c])
        params["disc_share"].update({s: [float(v) for v in sh[s]] for s in cols})
        dsd = cv["disc_sd"][c]
        sdrel = {s: dsd["sdrel_per_well_all"].get(s, dsd["sdrel_pooled_all"]) for s in cols}
        sdrel = {s: (v if np.isfinite(v) else dsd["sdrel_pooled_all"]) for s, v in sdrel.items()}
        params["disc_sdrel"].update(sdrel)
        obs = rec & yc[c].notna()
        val = np.where(obs, yc[c], ds["value"])
        flag = np.where(obs, 0, np.where(ds["value"].notna(), 5, 8))
        sd = np.where(flag == 5, ds["value"] * sdlog, np.where(flag == 0, 0.0, np.nan))
        county_rows.append(pd.DataFrame({
            "date": months.strftime("%Y-%m"), "county": c, "scope": "in_grid_hybrid",
            "value": val, "sd": sd, "sdlog": np.where(flag == 5, sdlog, np.nan),
            "lo90": np.where(flag == 5, ds["value"] * np.exp(-G.Z90 * sdlog), np.nan),
            "hi90": np.where(flag == 5, ds["value"] * np.exp(G.Z90 * sdlog), np.nan),
            "flag": flag.astype("uint8"),
            "method": np.where(flag == 0, "census_release_full_electricity_census_v1",
                               np.where(flag == 5, METHOD, "missing")),
            "n_missing_features": np.where(flag == 5, ds["n_missing_features"], 0),
            "census_raw_value": yc[c].to_numpy(), "tpc_ag_GWh": tpc[f"ag_{c}"].reindex(
                months).to_numpy(), "unit": "GWh/month"}))
        checks[f"observed_reproduces_census_{c}"] = float(np.nanmax(np.abs(
            np.where(obs, val - yc[c], 0.0))))
        shv = sh.reindex(months.month).to_numpy()                   # (T, W)
        for j, s in enumerate(cols):
            dval = np.where(obs, yd[s], ds["value"].to_numpy() * shv[:, j])
            disc_rows.append(pd.DataFrame({
                "date": months.strftime("%Y-%m"), "sid": s, "county": c, "value": dval,
                "sd": np.where(flag == 5, dval * sdrel[s], np.where(flag == 0, 0.0, np.nan)),
                "flag": flag.astype("uint8"),
                "method": np.where(flag == 0, "census_release_full_electricity_census_v1",
                                   np.where(flag == 5, METHOD + "+disc_share_moy",
                                            "missing")),
                "unit": "GWh/month"}))
    cty = pd.concat(county_rows, ignore_index=True)
    dsc = pd.concat(disc_rows, ignore_index=True)
    for t in (cty, dsc):
        t["product_version"] = None
        t["source_ids"] = "tpc_d007012;moeaea_agri_electricity;release_20261012b2/full"

    # gate
    g = cv["gate_G1_G3"]
    srcs = sorted({p.parent for p in tpc_files})
    g5_src = {rel_to_repo(d): validate_source(json.loads((d / "SOURCE.json").read_text(
        encoding="utf-8")), d) == [] for d in srcs}
    g4 = all(v <= 1e-9 for k, v in checks.items() if k.startswith("observed"))
    passed = bool(g["passed"] and g4 and all(g5_src.values()))
    d = new_version_dir(PRODUCT)
    ver = d.name
    cty["product_version"] = f"{PRODUCT}/{ver}"
    dsc["product_version"] = f"{PRODUCT}/{ver}"
    cty.to_parquet(d / "county_monthly.parquet", index=False)
    dsc.to_parquet(d / "discs_monthly.parquet", index=False)
    (d / "parameters.json").write_text(json.dumps(params, ensure_ascii=False, indent=1)
                                       + "\n", encoding="utf-8")
    summ = {}
    for c in COUNTIES:
        r = cv["county"][c][X.PRIMARY]
        summ[c] = {"rmse_ratio_vs_clim_pre": r["pre"]["rmse_ratio_vs_clim"],
                   "rmse_ratio_vs_clim_post": r["post"]["rmse_ratio_vs_clim"],
                   "anomaly_skill_pre": r["pre"]["anomaly_skill"],
                   "anomaly_skill_post": r["post"]["anomaly_skill"],
                   "sdlog": cv["sd_calibration"][c]["sdlog_all_folds"],
                   "coverage_post_with_pre_sd":
                       cv["sd_calibration"][c]["coverage_post_with_pre_sd"],
                   "disc_pooled_ratio_vs_disc_clim_post":
                       cv["discs"]["post"][c][X.PRIMARY]["pooled_rmse_ratio_vs_disc_clim"]}
    flags = cty.groupby(["county", "flag"]).size().unstack(fill_value=0)
    val = {"gate_spec": "dataprep_staging/x1_tpc_monthly/gate_spec.py",
           "gate_spec_sha256": cv["gate_spec_sha256"], "cv": rel_to_repo(cv_path),
           "cv_sha256": sha(cv_path), "G1": g["G1"], "G2": g["G2"], "G3": g["G3"],
           "G4_observed_reproduce_census": g4, "G5_tpc_sources_valid": g5_src,
           "checks": checks, "passed": passed, "summary": summ,
           "flag_counts": {c: {int(k): int(v) for k, v in flags.loc[c].items()}
                           for c in flags.index},
           "downscaled_months": {c: cty[(cty.county == c) & (cty.flag == 5)]["date"].tolist()
                                 for c in COUNTIES},
           "last_month": last.strftime("%Y-%m")}
    (d / "validation.json").write_text(json.dumps(val, ensure_ascii=False, indent=1) + "\n",
                                       encoding="utf-8")
    tail = cty[cty.date >= "2026-05"][["date", "county", "value", "sd", "flag"]]
    card = card_text(ver, val, tail, last)
    (d / "PRODUCT_CARD.md").write_text(card, encoding="utf-8")
    inputs = [*tpc_files, *panel.NAT_FILES, panel.ELEC_FULL / "cells_total.npz",
              panel.ELEC_FULL / "MANIFEST.json", panel.COUNTY_MAP / "zone_county_map.npz",
              panel.HEADS_PRE / "stations.parquet", panel.RAIN, panel.WELLS_CSV, cv_path,
              *[p for p in (panel.ET0_FULL).glob("*.npz")]]
    code = [HERE / n for n in ("build.py", "extend.py", "model.py", "panel.py", "cv.py",
                               "gate_spec.py")]
    m = build_manifest(
        d, product=PRODUCT, layer="L2", inputs=[str(p) for p in inputs],
        code=[str(p) for p in code],
        parameters={"primary": X.PRIMARY, "fit_window": ["2016-01", "2026-06"],
                    "months": [months[0].strftime("%Y-%m"), last.strftime("%Y-%m")],
                    "flags": {"0": "census recorded", "5": "downscaled from TPC",
                              "8": "missing"},
                    "sd": "county: multiplicative sdlog from out-of-fold log errors (all "
                          "folds 2018-2026H1); disc: relative sd per well, out of fold",
                    "census_last_recorded": "2026-06",
                    "tpc_last_month": last.strftime("%Y-%m")},
        fit_window_end=last.strftime("%Y-%m"), validation_report="validation.json",
        gate={"passed": passed, "criteria": "dataprep_staging/x1_tpc_monthly/gate_spec.py "
              "G1-G5 (fixed before the CV; sha256 in validation.json; not entered in the "
              "harness index)"},
        allowed_uses=ALLOWED, forbidden_uses=FORBIDDEN,
        personal_data="none (in-grid county and 3 km disc aggregates; well ids only)",
        extra={"use_note": "Post-origin product (U3/U4): never a pre-origin use. "
               "statistical_layer_history follows the C0 full census it extends; feeding a "
               "PROSPECTIVE statistical layer needs the same U4 ruling as "
               "I1.hybrid_tf.prospective_2026-11.v1 (plan 2.5 U4 names the TPC extension "
               "among the recorded-forcing-arm products)."})
    write_manifest(d, m)
    print("product", d, "passed", passed)
    print(tail.to_string())


def card_text(ver, val, tail, last) -> str:
    s = val["summary"]
    lines = [f"# {PRODUCT}/{ver} (WP X1)", "",
             "Monthly pump electricity of the hybrid layer's two drivers, extended past the "
             "census end with Taiwan Power's published county 農林漁牧 sales (d007012).", "",
             f"- **Coverage.** 2012-01..{last.strftime('%Y-%m')}. In-grid Changhua and Yunlin "
             "totals (`county_monthly.parquet`) and the sum within 3 km of each of the 158 "
             "wells of the hybrid test (`discs_monthly.parquet`). GWh/month.",
             "- **Flags.** 0 = census recorded month (release full/electricity_census, "
             "value_all, flags 0-5, >= 99 % of active cells), copied exactly. 5 = downscaled "
             "from TPC with sd (the census does not record 2019-09..2020-06 and ends "
             "2026-06). 8 = missing.",
             "- **Model.** Ridge regression (alpha 1) of the log census month-of-year anomaly "
             "on TPC county sector anomalies (own agriculture t, t-1, t-2; other county; "
             "residential, services, industry), the national agricultural total, rain and "
             "ET0. Chosen by a rule fixed before the CV (`gate_spec.py`).",
             "- **Skill (forward chaining by year, RMSE ratio to month-of-year climatology).** "
             f"Changhua {s['changhua']['rmse_ratio_vs_clim_pre']} (2018-22), "
             f"{s['changhua']['rmse_ratio_vs_clim_post']} (2023-26H1); Yunlin "
             f"{s['yunlin']['rmse_ratio_vs_clim_pre']}, {s['yunlin']['rmse_ratio_vs_clim_post']}."
             f" Discs (pooled, 2023-26H1): Changhua "
             f"{s['changhua']['disc_pooled_ratio_vs_disc_clim_post']}, Yunlin "
             f"{s['yunlin']['disc_pooled_ratio_vs_disc_clim_post']}.",
             f"- **sd.** Multiplicative 1-sigma {s['changhua']['sdlog']} (Changhua) and "
             f"{s['yunlin']['sdlog']} (Yunlin) in log units, out of fold. Disc sd per well.",
             f"- **Gate.** passed = {val['passed']} (G1-G5 of gate_spec.py).",
             "- **Allowed uses.** statistical_layer_history, forcing_recorded_arm, "
             "after_the_fact_analysis. Never a pre-origin use, target, baseline or validator. "
             "Feeding a prospective statistical layer needs the user's U4 ruling (as for "
             "I1.hybrid_tf.prospective_2026-11.v1).",
             "- **Known limits.** TPC 農林漁牧 includes non-pump load; its Yunlin seasonal "
             "shape differs from the census (profile correlation 0.37). In 2026 Yunlin TPC "
             "rose about 15 % over 2025 while the census rose less: the 2026H1 Yunlin fold "
             "is the worst (RMSE ratio about 2 to climatology). Disc values use a "
             "month-of-year share of the county total; the share error dominates at discs. "
             "Months 2025-07..2026-06 carry earlier scoring looks (P1, P2def).",
             "- **Licence.** TPC and Energy Administration data: Open Government Data "
             "License v1.0. Personal data: none.", "", "Last months:", "",
             "```", tail.to_string(index=False), "```", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    t0 = dt.datetime.now()
    main(sys.argv[1:])
    print("seconds", (dt.datetime.now() - t0).total_seconds())
