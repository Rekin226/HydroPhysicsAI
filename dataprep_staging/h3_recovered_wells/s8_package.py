"""Step 8: package ``heads/recovered_wells/v<N>`` under the admission rule declared in
``H3.recovery.v1`` (``gate_rule``), from the registered results of ``s6_score.py``.

Contents:

* ``recovered_wells.csv``: one row per H3 test well and per other on-grid excluded well with
  data: admitted or not, why, tier (``bridged`` = flags 0-2, ``observed_only`` = flags 0-1),
  envelope statistics on observed months, per-well H3.d_wells verdict;
* ``targets_preorigin.parquet``: the admitted wells' calibration targets 2012-01..2022-12:
  rows of ``heads/L2_filled_preorigin/v4`` (read through the guard for
  ``calibration_target``; flag 2 kept only for wells whose bridged months were admitted)
  plus, if ``H3.snapshot`` passed, well 09030212's once-daily months 2019-2022 (flag 0,
  method ``observed_daily_snapshot``, sigma_obs = max(0.10, sigma_snap));
* ``inventory.csv`` (all 344 stations, non-personal columns), ``rh_probe.json``,
  ``layer_review.csv``, ``validation.json``, ``PRODUCT_CARD.md``, ``MANIFEST.json``.

Usage: python -I s8_package.py
"""
from __future__ import annotations

import datetime as dt
import json
import shutil
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2])]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import guard, manifest  # noqa: E402

from h3_recovered_wells import config as C  # noqa: E402
from h3_recovered_wells import inventory  # noqa: E402

EV = C.EVIDENCE
REG = EV / "registered"
CODE = ["__init__.py", "config.py", "inventory.py", "snapshot.py", "s1_inventory.py",
        "s2_rh_probe.py", "s3_layer_review.py", "s4_prereg.py", "s5_runs.py", "s6_score.py",
        "s7b_snapshot_consistency.py", "s8_package.py"]
H2_CODE = ["__init__.py", "config.py", "io_h2.py", "bridge.py", "runner.py", "methods.py",
           "kriging.py", "scoring.py"]
ALLOWED = ["calibration_target", "datum_fit", "validator", "after_the_fact_analysis"]
FORBIDDEN = ["scoring_target", "prospective_scorer", "baseline", "initial_conditions",
             "idw_field", "prior", "prior_width_input", "model_design", "selection",
             "forcing_frozen", "forcing_recorded_arm", "statistical_layer_history"]
INV_COLS = ["sid", "site_id", "county", "town", "layer_raw", "layer_code", "layer_flag",
            "well_depth", "x", "y", "xy_flag", "cell", "on_grid", "on_grid_raw_xy", "in_model",
            "n_c20", "l1_c20", "l1_any", "c20_lead", "c20_max_internal", "record_first",
            "record_last", "record_end_trail", "twin_hour", "twin_max_gap_d", "raw_month",
            "twin_pass_qc", "reason", "candidate", "candidate_fail"]


def envelope(obs: np.ndarray, rec_last_idx: int) -> dict:
    rs = inventory.run_stats(obs)
    trail = C.N_MONTHS_PRE - 1 - rec_last_idx
    ok = (obs.mean() >= C.MIN_COVERAGE and obs.sum() >= C.MIN_OBS_MONTHS
          and rs["max_internal"] <= C.MAX_INTERNAL_RUN and rs["lead"] <= C.MAX_LEAD
          and trail <= C.MAX_TRAIL)
    return {"coverage": float(obs.mean()), "n_obs": int(obs.sum()), "lead": rs["lead"],
            "max_internal": rs["max_internal"], "record_end_trail": int(trail), "pass": bool(ok)}


def main():
    reg = json.loads((EV / "registered_wells.json").read_text())
    dres = json.loads((REG / f"{C.TEST_D}.json").read_text())
    sres = json.loads((REG / f"{C.TEST_SNAP}.json").read_text())
    inv = pd.read_csv(EV / "inventory" / "inventory.csv",
                      dtype={"sid": str, "layer_raw": str, "layer_code": str, "site_id": str})
    rh = reg["rh_well"]
    pooled_ok = bool(dres["pooled"]["rmse_pass"])
    v4p = guard.open_product(C.H2_PRE_DIR, "calibration_target")
    v4 = v4p.read_table("fill.parquet")
    v4_pv = "heads/L2_filled_preorigin/v4"
    dates = pd.date_range(f"{C.START}-01", f"{C.PRE_END}-01", freq="MS")
    months = [d.strftime("%Y-%m") for d in dates]
    # 09030212: envelope counting its once-daily months (only if H3.snapshot passed)
    mon = inventory.load_monthly()
    c20 = set(mon[(mon.sid == rh) & mon.flag.isin([0, 1]) & mon.c20.astype(bool)]
              .date.dt.strftime("%Y-%m"))
    snap = pd.read_csv(REG / "H3.snapshot_months_2019_2022.csv", dtype={"month": str})
    sig_snap = float(sres["sigma_snap"]["sigma_snap"])
    obs_rh = np.array([m in c20 or (sres["pass"] and m in set(snap.month)) for m in months])
    last_rh = int(np.where(obs_rh)[0].max()) if obs_rh.any() else -1
    env_rh = envelope(obs_rh, last_rh)
    # Post-look hold-back (a tightening of the registered packaging rule, disclosed in the
    # report): the after-the-fact neighbour check (s7b_snapshot_consistency.py) shows the
    # once-daily months 2019-2022 offset from the hourly-era relation with the neighbours,
    # which H3.snapshot (same-instrument representativeness) cannot see.
    sc_path = EV / "after_fact" / "snapshot_consistency.json"
    hold_back = None
    if sc_path.exists():
        sc = json.loads(sc_path.read_text())
        a, lo = sc["once_daily_admitted_months"], sc["loyo_2012_2018"]
        if a["n"] and abs(a["bias"]) > 2 * lo["rmse"]:
            hold_back = (f"held back post-look: once-daily months 2019-2022 sit "
                         f"{a['bias']:+.2f} m (RMSE {a['rmse']:.2f} m, n {a['n']}) from the "
                         f"neighbour model fitted on 2012-2018 (leave-one-year-out RMSE "
                         f"{lo['rmse']:.2f} m, bias {lo['bias']:+.3f}); possible level shift "
                         f"in 2019 not caught by the persistence screen")
    rows, keep = [], {}
    og = inv[(~inv.in_model.astype(bool)) & inv.on_grid.astype(bool) & (inv.n_raw > 0)]
    for _, r in og.iterrows():
        sid = r.sid
        pw = dres["per_well"].get(sid)
        rec = {"sid": sid, "county": r.county, "layer": r.layer_raw, "cell": int(r.cell),
               "coverage_l1_c20": r.l1_c20, "n_c20": int(r.n_c20), "lead": int(r.c20_lead),
               "max_internal": int(r.c20_max_internal),
               "record_end_trail": int(r.record_end_trail),
               "d_test_k_windows": pw["k_windows"] if pw else None,
               "d_test_coverage": (pw["coverage_gate"]["coverage"]
                                   if pw and pw.get("coverage_gate") else None),
               "d_test_pass": pw["pass"] if pw else None}
        if bool(r.candidate):
            rec.update(admitted=True, recovered_by="envelope (observed months)",
                       tier="bridged" if (pooled_ok and pw and pw["pass"]) else "observed_only",
                       reason="")
        elif sid == rh:
            ok_rule = bool(sres["pass"]) and env_rh["pass"]
            held = ok_rule and hold_back is not None
            ok = ok_rule and not held
            rec.update(admitted=ok, recovered_by="R-h once-daily record" if ok else "",
                       tier=("bridged" if (pooled_ok and pw and pw["pass"]) else
                             "observed_only") if ok else "",
                       reason="" if ok else (hold_back if held else (
                           "H3.snapshot failed" if not sres["pass"] else
                           f"envelope with once-daily months fails {env_rh}")),
                       admitted_by_registered_rule=ok_rule,
                       coverage_with_snapshot=env_rh["coverage"])
        else:
            why = r.candidate_fail
            if r.record_last == "2018-12":
                why += "; record stops 2018-12, no later record found (R-h)"
            rec.update(admitted=False, recovered_by="", tier="", reason=why)
        rows.append(rec)
        if rec["admitted"]:
            keep[sid] = rec["tier"]
    W = pd.DataFrame(rows).sort_values(["admitted", "county", "sid"], ascending=[False, True,
                                                                                 True])
    # targets
    parts = []
    for sid, tier in keep.items():
        t = v4[v4.sid == sid].copy()
        if tier != "bridged":
            t = t[t.flag.isin([0, 1])]
        t["source_ids"] = f"{v4_pv}|" + t.source_ids.astype(str)
        parts.append(t)
    if rh in keep and sres["pass"]:
        lay = str(inv.loc[inv.sid == rh, "layer_raw"].iloc[0])
        s = pd.DataFrame({
            "sid": rh, "date": snap.month + "-01", "month": snap.month,
            "value": snap.value.astype(float), "sd": 0.0, "flag": np.uint8(0),
            "method": "observed_daily_snapshot", "gap_class": "", "gap_run_len": 0,
            "edge_dist": 0, "bridged_share": np.nan, "l1_n_clean_days": 0,
            "sigma_obs_m": max(0.10, sig_snap), "u2_weight": 1.0, "in_model": False,
            "layer": lay, "county": "YL",
            "source_ids": "L1v3:heads_daily once-daily readings (day_valid False), "
                          f"{C.TEST_SNAP}",
            "head_m": snap.value.astype(float), "head_sd_m": 0.0})
        s["n_once_daily_readings"] = snap.n_days.astype(int)
        parts.append(s)
    T = pd.concat(parts, ignore_index=True) if parts else v4.iloc[:0].copy()
    T["flag"] = T.flag.astype(np.uint8)
    T["tier"] = T.sid.map(keep)
    T["cell"] = T.sid.map(dict(zip(W.sid, W.cell, strict=True))).astype(int)
    T = T.sort_values(["sid", "month"]).reset_index(drop=True)
    assert T.month.max() <= C.PRE_END and T.flag.isin([0, 1, 2]).all()
    d = manifest.new_version_dir(C.PRODUCT)
    pv = f"{C.PRODUCT}/{d.name}"
    T["product_version"] = pv
    T.to_parquet(d / "targets_preorigin.parquet", index=False)
    W.to_csv(d / "recovered_wells.csv", index=False)
    inv[INV_COLS].to_csv(d / "inventory.csv", index=False)
    shutil.copy(EV / "rh_probe" / "rh_probe.json", d / "rh_probe.json")
    shutil.copy(EV / "layer_review" / "layer_review.csv", d / "layer_review.csv")
    adm = W[W.admitted.astype(bool)]
    val = {
        "release": "h3_v1", "prereg": C.PREREG_ID,
        "registered": {C.TEST_D: {"n_wells_pass": dres["n_wells_pass"],
                                  "n_wells": dres["n_wells"],
                                  "pooled_rmse_pass": pooled_ok,
                                  "pooled": dres["pooled"],
                                  "per_well": {k: {kk: v.get(kk) for kk in
                                                   ("k_windows", "n_cells", "pass",
                                                    "calibrated", "rmse_bridge")}
                                               | {"coverage": v.get("coverage_gate", {})
                                                  .get("coverage"),
                                                  "lo": v.get("coverage_gate", {}).get("lo"),
                                                  "hi": v.get("coverage_gate", {}).get("hi")}
                                               for k, v in dres["per_well"].items()}},
                       C.TEST_SNAP: {k: sres[k] for k in ("pass", "calibrated", "sigma_snap",
                                                          "own_months", "own_rmse", "own_bias",
                                                          "k_quarters", "coverage_gate",
                                                          "screen_2019_2022",
                                                          "snapshot_months_2019_2022")}},
        "envelope_09030212_with_snapshot": env_rh,
        "post_look_hold_back_09030212": hold_back,
        "admitted": {"n": int(len(adm)), "by_county": adm.county.value_counts().to_dict(),
                     "by_tier": adm.tier.value_counts().to_dict(),
                     "by_layer": adm.layer.value_counts().to_dict(), "sids": adm.sid.tolist()},
        "targets": {"rows": int(len(T)), "by_flag": T.flag.value_counts().to_dict(),
                    "by_method": T.method.value_counts().to_dict(),
                    "max_month": T.month.max(), "min_month": T.month.min()},
        "acceptance_plan": {
            "each_recovered_well_passes_on_observed_months": bool(
                adm.shape[0] > 0 and all(
                    (r.tier == "observed_only") or bool(r.d_test_pass)
                    for r in adm.itertuples())),
            "r5_rung_gate": "not evaluated in Phase 0 (I1, after the freeze)"},
    }
    (d / "validation.json").write_text(json.dumps(val, indent=1, ensure_ascii=False,
                                                  default=str))
    card(d, pv, adm, T, dres, sres, env_rh, pooled_ok, hold_back)
    m = manifest.build_manifest(
        d, product=C.PRODUCT, layer="L2",
        inputs=[str(C.H2_PRE_DIR / "MANIFEST.json"), str(C.H2_PRE_DIR / "fill.parquet"),
                str(C.L1_DIR / "MANIFEST.json"), str(C.L1_DIR / "heads_monthly.parquet"),
                str(C.L1_DIR / "heads_daily.parquet"),
                str(C.L1_DIR / "station_meta_preorigin.parquet"),
                str(C.UNITS_DIR / "wells.csv"), str(C.GRID_NPZ), str(C.H2_PARAMS),
                str(C.AMP_STATIONS),
                str(EV / "inventory" / "inventory.csv"), str(EV / "registered_wells.json"),
                str(REG / f"{C.TEST_D}.json"), str(REG / f"{C.TEST_SNAP}.json"),
                str(REG / "H3.snapshot_months_2019_2022.csv"),
                str(EV / "rh_probe" / "rh_probe.json"),
                str(EV / "layer_review" / "layer_review.csv"),
                str(EV / "after_fact" / "snapshot_consistency.json"),
                str(C.REPO / "results/dataprep/preregistration" / f"{C.PREREG_ID}.json")],
        code=[str(C.CODE_DIR / f) for f in CODE]
        + [str(C.REPO / "dataprep_staging/h2_heads_fill" / f) for f in H2_CODE],
        input_trees=[str(C.AMP_WELLS)],
        parameters={"coverage_definition": C.COVERAGE_DEFINITION,
                    "min_coverage": C.MIN_COVERAGE, "min_obs_months": C.MIN_OBS_MONTHS,
                    "max_internal_run": C.MAX_INTERNAL_RUN, "max_lead": C.MAX_LEAD,
                    "max_trail": C.MAX_TRAIL, "layers": list(C.LAYERS),
                    "fresh_d": [C.FRESH_D.start, C.FRESH_D.stop - 1],
                    "sigma_snap": sig_snap, "rh_well": rh,
                    "use_note": "calibration_target only in the pre-declared I1 ladder rung "
                                "R5; the 2023-26 test set stays the registered 158 wells"},
        fit_window_end=C.PRE_END, validation_report="validation.json",
        gate={"passed": True,
              "criteria": f"{C.PREREG_ID} admission rule (gate_rule): envelope on observed "
                          f"months; {C.TEST_D} per-well coverage (gate-3, k >= 20) and pooled "
                          f"RMSE rule for bridged months; {C.TEST_SNAP} for the once-daily "
                          "months of 09030212. Plan acceptance part 2 (R5 rung gate) is "
                          "evaluated by I1 after the freeze."},
        allowed_uses=ALLOWED, forbidden_uses=FORBIDDEN,
        personal_data="none (no names, addresses or contacts; station coordinates only)",
        created=dt.date.today().isoformat())
    manifest.write_manifest(d, m)
    print(d, json.dumps(val["admitted"], ensure_ascii=False), val["targets"])


def card(d, pv, adm, T, dres, sres, env_rh, pooled_ok, hold_back):
    lines = [
        f"# {pv}: recovered wells for rung R5 (WP H3)", "",
        "Wells outside the twin's 158-well set whose pre-origin record qualifies on observed "
        "months alone (plan WP H3), with their calibration targets for 2012-01..2022-12. "
        "Use them only in the pre-declared I1 ladder rung R5; the 2023-26 test set stays the "
        "registered 158 wells.", "",
        f"Admitted: {len(adm)} wells ({adm.county.value_counts().to_dict()}; tiers "
        f"{adm.tier.value_counts().to_dict()}). Targets: {len(T)} well-months "
        f"(flags {T.flag.value_counts().to_dict()}).", "",
        "## How a well qualifies (H3.recovery.v1, declared before any test cell existed)", "",
        "- Envelope on observed months only (L1 v3 C20 months, flag 0/1; fills never count): "
        "not a model well; in an active cell with the L1 repaired coordinates; raw layer "
        "code 1-4; coverage >= 0.80 of 2012-01..2022-12; >= 36 observed months; internal "
        "runs <= 24 months; leading run <= 66 months; record reaching 2022-12.",
        f"- Bridged months (flag 2, rows of heads/L2_filled_preorigin/v4) are kept only for "
        f"wells that pass H3.d_wells (per-well coverage of H2's daily bridge, rule gate-3 "
        f"with k >= 20 windows; {dres['n_wells_pass']}/{dres['n_wells']} test wells passed) "
        f"and only because the pooled bridge RMSE rule passed: {pooled_ok}.",
        f"- 09030212 (R-h find): its 60-minute record stops on 2018-12-31 and continues with "
        f"one 00:00 reading per day in the twin's cache. H3.snapshot: "
        f"{'passed' if sres['pass'] else 'failed'} (sigma_snap "
        f"{sres['sigma_snap']['sigma_snap']:.3f} m). Coverage with its once-daily months: "
        f"{env_rh['coverage']:.3f}. "
        + (f"NOT ADMITTED: {hold_back}. Its once-daily months stay in H3_evidence for a "
           f"sensitivity run only." if hold_back else "Admitted."), "",
        "## Columns", "",
        "`targets_preorigin.parquet` keeps the columns of heads/L2_filled_preorigin/v4 "
        "(value = head_m, sd, flag, method, sigma_obs_m, u2_weight, ...) plus `tier`, `cell` "
        "and, for once-daily months, `n_once_daily_readings`. Likelihood variance: "
        "sigma_obs_m^2 + head_sd_m^2 (u2_weight = sigma_obs^2 / (sigma_obs^2 + sd^2)).", "",
        "## Known limits", "",
        ("- 09030212's once-daily months are not in this version (held back post-look); "
         "if a later version admits them, note that L1's offset-episode detector never "
         "screened them (it needs samples < 70 min apart)." if hold_back else
         "- The once-daily months of 09030212 (method observed_daily_snapshot) were never "
         "screened by L1's offset-episode detector (it needs samples < 70 min apart); H3 "
         "applied its own persistence screen, and their sigma_obs is sigma_snap."),
        "- Most admitted wells were excluded by the twin only because one internal gap "
        "exceeded 180 days (the twin's max-gap rule); their gap months are simply missing "
        "here (no fill enters as a target).",
        "- Two Changhua wells (07170211, 07170221) are admitted because H1a repaired their "
        "swapped coordinates; the twin's raw parse rejects them.",
        "- Layer codes are as stored by WRA (unverified beyond that). Six apex wells with no "
        "code (layer proposals flag 5) are listed in layer_review.csv and NOT admitted.",
        "- No post-origin data: the candidates are outside the new network fetch; a "
        "secondary 2023-26 test set would need a credentialed vendor fetch (not done).",
        "- R5's own rung gate (plan acceptance part 2) is decided by I1 after the freeze.", ""]
    (d / "PRODUCT_CARD.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
