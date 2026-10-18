"""Repair round 1: package ``heads/recovered_wells/v2`` under the admission rule of
``H3.repair.v1`` (``gate_rule``), and write the product root's ``SUPERSEDED.json``.

Changes from v1 (``s8_package.py`` stays as it was; v1 hashes it):

* admission: the first round's post-look hold-back criterion is replaced by the registered
  step screen (``H3.step_screen``), applied to every admitted well; flagged or untestable
  wells are held back (none in the registered result); 09030212 is held back under rule (3)
  of ``H3.repair.v1`` (no informative representativeness test for its once-daily months; a
  persistence screen tuned after viewing), and its screen result is reported;
* flag-2 rows are kept only with method ``daily_bridge_partial`` for wells passing
  ``H3.d_wells`` (family d never tested outage bridges per well);
* ``validation.json`` and the manifest gate state the plan acceptance separately from the
  release rule: the plan criterion "each recovered well passes the H2 coverage strata" is
  recorded as FAILED (the first round recorded it as true by counting observed-only wells
  as passing);
* the R-h probe summary ships with post-origin availability dates redacted
  (``rh_probe_preorigin.json``), and the probe's sources are manifest inputs;
* the card tells validators to use flags 0-1 only (U1).

Usage: python -I r1_package.py
"""
from __future__ import annotations

import datetime as dt
import json
import re
import shutil
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2])]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import guard, manifest  # noqa: E402

from h3_recovered_wells import config as C  # noqa: E402
from h3_recovered_wells import inventory  # noqa: E402
from h3_recovered_wells.s8_package import (  # noqa: E402
    ALLOWED,
    FORBIDDEN,
    H2_CODE,
    INV_COLS,
    envelope,
)

EV = C.EVIDENCE
REG = EV / "registered"
R1 = EV / "repair1"
PREREG_R1 = "H3.repair.v1"
TEST_STEP = "H3.step_screen"
V1 = C.REPO / "results/dataprep/heads/recovered_wells/v1"
CODE = ["__init__.py", "config.py", "inventory.py", "snapshot.py", "s1_inventory.py",
        "s2_rh_probe.py", "s3_layer_review.py", "s4_prereg.py", "s5_runs.py", "s6_score.py",
        "s7b_snapshot_consistency.py", "s8_package.py", "r1_looks.py", "r1_stepscreen.py",
        "r1_prereg.py", "r1_score.py", "r1_package.py"]
RH_SOURCES = ["chou-shui-data/data/raw/gw_timeseries_raw.csv",
              "chou-shui-data/data/gw_timeseries.csv", "chou-shui-data/data/gw_stations.csv",
              "data_fetch_api/new_measurements/gw-wra-gw10min-obs-stations.parquet"]
RH_TREES = ["data_fetch_api/new_measurements/heads_202210_202610"]
_DATE = re.compile(r"^(\d{4})-(\d{2})(-\d{2}.*)?$")


def _redact_key(k):
    """Dictionary keys are redacted like values: a year after 2022, a YYYY-MM.. date after
    2022-12 or a YYYYMMDD date after 2022-12-31."""
    ks = str(k).strip()
    if re.fullmatch(r"(19|20)\d{2}", ks) and int(ks) > 2022:
        return "post-origin (redacted)"
    r = redact_post_origin(ks)
    return r if r != ks else k


def redact_post_origin(x):
    """Replace every date after 2022-12 (YYYY-MM[-DD ...] strings, YYYYMMDD integers or
    strings) with 'post-origin (redacted)'."""
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            k2, v2 = _redact_key(k), redact_post_origin(v)
            if k2 in out and isinstance(out[k2], (int, float)) and isinstance(v2, (int, float)):
                out[k2] += v2
            elif k2 in out:
                out[k2] = (out[k2] if isinstance(out[k2], list) else [out[k2]]) + [v2]
            else:
                out[k2] = v2
        return out
    if isinstance(x, list):
        return [redact_post_origin(v) for v in x]
    if isinstance(x, str):
        m = _DATE.match(x.strip())
        if m and (int(m.group(1)), int(m.group(2))) > (2022, 12):
            return "post-origin (redacted)"
        if re.fullmatch(r"\d{8}", x.strip()) and x.strip() > "20221231":
            return "post-origin (redacted)"
        return x
    if isinstance(x, (int, np.integer)) and 19000101 <= int(x) <= 20991231 \
            and int(x) > 20221231:
        return "post-origin (redacted)"
    return x


def segment_check(obs: np.ndarray, gap: tuple[int, int]) -> dict:
    """Does either segment of a split record meet the envelope on its own?"""
    T = len(obs)
    out = {}
    for name, keep in (("pre", np.arange(T) < gap[0]), ("post", np.arange(T) > gap[1])):
        o = obs & keep
        if not o.any():
            out[name] = {"n_obs": 0, "pass": False}
            continue
        e = envelope(o, int(np.where(o)[0].max()))
        out[name] = {k: e[k] for k in ("n_obs", "coverage", "lead", "record_end_trail",
                                       "max_internal", "pass")}
    out["either_passes"] = bool(out["pre"]["pass"] or out["post"]["pass"])
    return out


def main():
    from h3_recovered_wells import r1_stepscreen as S
    dres = json.loads((REG / f"{C.TEST_D}.json").read_text())
    sres = json.loads((REG / f"{C.TEST_SNAP}.json").read_text())
    stp = json.loads((R1 / "registered" / f"{TEST_STEP}.json").read_text())
    step = {r["sid"]: r for r in stp["per_well"]}
    reg = json.loads((EV / "registered_wells.json").read_text())
    inv = pd.read_csv(EV / "inventory" / "inventory.csv",
                      dtype={"sid": str, "layer_raw": str, "layer_code": str, "site_id": str})
    rh = reg["rh_well"]
    pooled_ok = bool(dres["pooled"]["rmse_pass"])
    v4 = guard.open_product(C.H2_PRE_DIR, "calibration_target").read_table("fill.parquet")
    v4_pv = "heads/L2_filled_preorigin/v4"
    dates = S.DATES
    months = [d.strftime("%Y-%m") for d in dates]
    mon = inventory.load_monthly()
    snap = pd.read_csv(REG / "H3.snapshot_months_2019_2022.csv", dtype={"month": str})
    c20 = set(mon[(mon.sid == rh) & mon.flag.isin([0, 1]) & mon.c20.astype(bool)]
              .date.dt.strftime("%Y-%m"))
    obs_rh = np.array([m in c20 or (sres["pass"] and m in set(snap.month)) for m in months])
    env_rh = envelope(obs_rh, int(np.where(obs_rh)[0].max()))
    rule3 = ("held back under H3.repair.v1 rule (3): its once-daily months 2019-2022 have no "
             "informative representativeness test (H3.snapshot passed only because its own "
             "RMSE, 0.0025 m, is 20x below sigma_snap and the point window switched off; 24 "
             "of 52 estimation wells fail the same test) and the persistence-screen constants "
             "were chosen after the series was viewed. The first round's neighbour-offset "
             f"reason is withdrawn: the registered step screen does not flag it (z "
             f"{step[rh]['z']:.2f}, p {step[rh]['p']:.3f} against {step[rh]['n_null']} model "
             f"wells; {100 * step[rh]['null_share_z_ge_2']:.0f} % of model wells reach z >= 2 "
             f"at this split)")
    rows, keep = [], {}
    og = inv[(~inv.in_model.astype(bool)) & inv.on_grid.astype(bool) & (inv.n_raw > 0)]
    for _, r in og.iterrows():
        sid = r.sid
        pw = dres["per_well"].get(sid)
        st = step.get(sid)
        rec = {"sid": sid, "county": r.county, "layer": r.layer_raw, "cell": int(r.cell),
               "coverage_l1_c20": r.l1_c20, "n_c20": int(r.n_c20), "lead": int(r.c20_lead),
               "max_internal": int(r.c20_max_internal),
               "record_end_trail": int(r.record_end_trail),
               "d_test_k_windows": pw["k_windows"] if pw else None,
               "d_test_coverage": (pw["coverage_gate"]["coverage"]
                                   if pw and pw.get("coverage_gate") else None),
               "d_test_pass": pw["pass"] if pw else None,
               "step_split_first_month": st["gap_first"] if st else None,
               "step_split_gap_months": st["gap_months"] if st else None,
               "step_bias_m": st.get("bias") if st else None,
               "step_rmse_loyo_m": st.get("rmse_loyo") if st else None,
               "step_z": st.get("z") if st else None,
               "step_p": st.get("p") if st else None,
               "step_verdict": st["verdict"] if st else None,
               "step_null_z_q95": st.get("null_z_q95") if st else None,
               "step_shift_unflagged_up_to_m": (st["null_z_q95"] * st["rmse_loyo"]
                                                if st and st.get("rmse_loyo") is not None
                                                and st.get("null_z_q95") is not None
                                                else None)}
        if bool(r.candidate):
            flagged = st is None or st.get("flagged") is not False
            if flagged:
                rec.update(admitted=False, recovered_by="", tier="",
                           reason=f"held back: H3.step_screen {st['verdict'] if st else 'missing'}")
            else:
                rec.update(admitted=True, recovered_by="envelope (observed months)",
                           tier="bridged" if (pooled_ok and pw and pw["pass"]) else
                           "observed_only", reason="")
        elif sid == rh:
            ok_rule = bool(sres["pass"]) and env_rh["pass"]
            rec.update(admitted=False, recovered_by="", tier="", reason=rule3,
                       admitted_by_registered_rule_v1=ok_rule,
                       coverage_with_snapshot=env_rh["coverage"])
        else:
            why = r.candidate_fail
            if r.record_last == "2018-12":
                why += ("; record stops 2018-12, no later record in the routes checked "
                        "(R-h; the credentialed vendor route was not run)")
            rec.update(admitted=False, recovered_by="", tier="", reason=why)
        # segment check for every screened well (gate rule 2 records it)
        if st is not None:
            o = np.array([m in set(mon[(mon.sid == sid) & mon.flag.isin([0, 1])
                                       & mon.c20.astype(bool)].date.dt.strftime("%Y-%m"))
                          for m in months])
            if sid == rh:
                o = obs_rh
            g0 = months.index(st["gap_first"])
            g = (g0, g0 + int(st["gap_months"]) - 1)
            sc = segment_check(o, g)
            rec["segment_pre_passes_envelope"] = sc["pre"]["pass"]
            rec["segment_post_passes_envelope"] = sc["post"]["pass"]
        rows.append(rec)
        if rec["admitted"]:
            keep[sid] = rec["tier"]
    W = pd.DataFrame(rows).sort_values(["admitted", "county", "sid"],
                                       ascending=[False, True, True])
    parts = []
    for sid, tier in keep.items():
        t = v4[v4.sid == sid].copy()
        if tier == "bridged":
            t = t[t.flag.isin([0, 1]) | ((t.flag == 2) & (t.method == "daily_bridge_partial"))]
        else:
            t = t[t.flag.isin([0, 1])]
        t["source_ids"] = f"{v4_pv}|" + t.source_ids.astype(str)
        parts.append(t)
    T = pd.concat(parts, ignore_index=True)
    T["flag"] = T.flag.astype(np.uint8)
    T["tier"] = T.sid.map(keep)
    T["cell"] = T.sid.map(dict(zip(W.sid, W.cell, strict=True))).astype(int)
    T = T.sort_values(["sid", "month"]).reset_index(drop=True)
    assert T.month.max() <= C.PRE_END and T.flag.isin([0, 1, 2]).all()
    assert set(T.loc[T.flag == 2, "method"]) <= {"daily_bridge_partial"}
    # identity with v1 (same admitted wells expected when nothing is flagged)
    v1t = pd.read_parquet(V1 / "targets_preorigin.parquet")
    cols = ["sid", "month", "value", "sd", "flag", "method", "sigma_obs_m", "u2_weight"]
    a = T[cols].sort_values(["sid", "month"]).reset_index(drop=True)
    b = v1t[v1t.sid.isin(set(T.sid))][cols].sort_values(["sid", "month"]).reset_index(
        drop=True)
    same_rows = bool(len(a) == len(b) and (a[["sid", "month", "flag", "method"]]
                                           .equals(b[["sid", "month", "flag", "method"]]))
                     and np.nanmax(np.abs(a[["value", "sd", "sigma_obs_m", "u2_weight"]]
                                          .to_numpy(float)
                                          - b[["value", "sd", "sigma_obs_m", "u2_weight"]]
                                          .to_numpy(float))) == 0.0)
    d = manifest.new_version_dir(C.PRODUCT)
    pv = f"{C.PRODUCT}/{d.name}"
    T["product_version"] = pv
    T.to_parquet(d / "targets_preorigin.parquet", index=False)
    W.to_csv(d / "recovered_wells.csv", index=False)
    inv[INV_COLS].to_csv(d / "inventory.csv", index=False)
    rhp = json.loads((EV / "rh_probe" / "rh_probe.json").read_text())
    red = redact_post_origin(rhp)
    red["_redaction"] = ("dates after 2022-12 (vendor record windows, disuse dates, first "
                         "samples after 2018) replaced by 'post-origin (redacted)'; the full "
                         "probe output is reports/H3_evidence/rh_probe/rh_probe.json "
                         "(availability metadata, never observations)")
    red["_credentialed_route"] = ("not run: the project's existing vendor access exists "
                                  "(data_fetch_api/, gitignored) but this session was not "
                                  "permitted to load it; r1_vendor_probe.py runs the probe "
                                  "with WISENVR_* set in the caller's environment")
    (d / "rh_probe_preorigin.json").write_text(json.dumps(red, indent=1, ensure_ascii=False))
    shutil.copy(EV / "layer_review" / "layer_review.csv", d / "layer_review.csv")
    adm = W[W.admitted.astype(bool)]
    n_pass = int(sum(bool(x) for x in adm.d_test_pass))
    plan_ok = bool(len(adm) > 0 and n_pass == len(adm))
    held = W[(W.sid.isin(list(step))) & ~W.admitted.astype(bool)]
    val = {
        "release": "h3_v2", "prereg": [C.PREREG_ID, PREREG_R1],
        "acceptance_plan": {
            "criterion": "each recovered well passes the H2 coverage strata on its observed "
                         "months (plan WP H3 acceptance, part 1)",
            "each_recovered_well_passes_h2_coverage_stratum": plan_ok,
            "n_admitted": int(len(adm)), "n_admitted_passing_H3.d_wells": n_pass,
            "detail": (f"{n_pass} of {len(adm)} admitted wells pass H3.d_wells (per-well "
                       "daily-bridge coverage, rule gate-3, k >= 20 windows). The others "
                       "FAIL it and enter with observed months only (flags 0-1) under the "
                       "declared fallback, so no fill whose coverage failed is a target; "
                       "the fallback does not make the plan criterion pass."),
            "r5_rung_gate": "not evaluated in Phase 0 (plan acceptance part 2; I1, after "
                            "the freeze)",
            "wp_status": "partial"},
        "release_rule": {
            "rule": f"{C.PREREG_ID} envelope and bridge rule (with its declared fallback) "
                    f"and {PREREG_R1} (step screen for every admitted well; 09030212 rule "
                    "3; bridged rows daily_bridge_partial only)",
            "satisfied": True,
            "admitted": adm.sid.tolist(),
            "held_back_screened": {s: W.loc[W.sid == s, "reason"].iloc[0]
                                   for s in held.sid}},
        "registered": {
            C.TEST_D: {"n_wells_pass": dres["n_wells_pass"], "n_wells": dres["n_wells"],
                       "pooled_rmse_pass": pooled_ok, "pooled": dres["pooled"],
                       "per_well": {k: {kk: v.get(kk) for kk in
                                        ("k_windows", "n_cells", "pass", "calibrated",
                                         "rmse_bridge")}
                                    | {"coverage": v.get("coverage_gate", {}).get("coverage"),
                                       "lo": v.get("coverage_gate", {}).get("lo"),
                                       "hi": v.get("coverage_gate", {}).get("hi")}
                                    for k, v in dres["per_well"].items()}},
            C.TEST_SNAP: {k: sres[k] for k in ("pass", "calibrated", "sigma_snap",
                                               "own_months", "own_rmse", "own_bias",
                                               "k_quarters", "coverage_gate",
                                               "screen_2019_2022",
                                               "snapshot_months_2019_2022")}
            | {"power_note": "uninformative: passes at a well whose own RMSE is 20x below "
                             "sigma_snap (point window off at n_eff_lower 25 < 30); the "
                             "verifier found 24 of 52 estimation wells fail the same test"},
            TEST_STEP: {"prereg_sha256": stp["prereg_sha256"], "rule": stp["rule"],
                        "admitted_v1": stp["admitted_v1"],
                        "per_well": {r["sid"]: {k: r.get(k) for k in
                                                ("gap_first", "gap_months", "n_fit",
                                                 "n_eval", "bias", "rmse_loyo", "z", "p",
                                                 "n_null", "verdict", "neighbours")}
                                     for r in stp["per_well"]},
                        "null_by_split": stp["null_by_split"]}},
        "envelope_09030212_with_snapshot": env_rh,
        "first_round_hold_back_09030212": "withdrawn as a reason (not flagged by the "
                                          "registered step screen); the well stays held "
                                          "back under H3.repair.v1 rule (3)",
        "admitted": {"n": int(len(adm)), "by_county": adm.county.value_counts().to_dict(),
                     "by_tier": adm.tier.value_counts().to_dict(),
                     "by_layer": adm.layer.value_counts().to_dict(), "sids": adm.sid.tolist()},
        "targets": {"rows": int(len(T)), "by_flag": T.flag.value_counts().to_dict(),
                    "by_method": T.method.value_counts().to_dict(),
                    "max_month": T.month.max(), "min_month": T.month.min(),
                    "identical_to_v1_rows": same_rows},
        "use_notes": {"validator": "flags 0-1 only (U1: flag-2 bridged rows are imputed); "
                                   "the guard's default flags for 'validator' include flag "
                                   "2, so filter with flag <= 1",
                      "calibration_target": "rung R5 only; likelihood variance sigma_obs_m^2 "
                                            "+ head_sd_m^2"},
    }
    (d / "validation.json").write_text(json.dumps(val, indent=1, ensure_ascii=False,
                                                  default=str))
    card(d, pv, adm, T, dres, sres, stp, n_pass, pooled_ok, rule3)
    ls = pd.read_csv(EV / "rh_probe" / "rh_probe_wells.csv", dtype={"sid": str})
    ls_files = [f"chou-shui-data/data/ls_cache/gw__{s}__raw.parquet"
                for s in ls[ls.ls_cache_raw.astype(bool)].sid]
    m = manifest.build_manifest(
        d, product=C.PRODUCT, layer="L2",
        inputs=[str(C.H2_PRE_DIR / "MANIFEST.json"), str(C.H2_PRE_DIR / "fill.parquet"),
                str(C.L1_DIR / "MANIFEST.json"), str(C.L1_DIR / "heads_monthly.parquet"),
                str(C.L1_DIR / "heads_daily.parquet"),
                str(C.L1_DIR / "station_meta_preorigin.parquet"),
                str(C.UNITS_DIR / "wells.csv"), str(C.GRID_NPZ), str(C.H2_PARAMS),
                str(C.AMP_STATIONS), str(V1 / "MANIFEST.json"),
                str(V1 / "targets_preorigin.parquet"), str(V1 / "recovered_wells.csv"),
                str(EV / "inventory" / "inventory.csv"), str(EV / "registered_wells.json"),
                str(REG / f"{C.TEST_D}.json"), str(REG / f"{C.TEST_SNAP}.json"),
                str(REG / "H3.snapshot_months_2019_2022.csv"),
                str(R1 / "registered" / f"{TEST_STEP}.json"),
                str(R1 / "registered" / f"{TEST_STEP}_wells.csv"),
                str(R1 / "registered" / f"{TEST_STEP}_null.parquet"),
                str(EV / "rh_probe" / "rh_probe.json"),
                str(EV / "rh_probe" / "rh_probe_wells.csv"),
                str(EV / "layer_review" / "layer_review.csv"),
                str(EV / "after_fact" / "snapshot_consistency.json"),
                str(C.REPO / "results/dataprep/preregistration" / f"{C.PREREG_ID}.json"),
                str(C.REPO / "results/dataprep/preregistration" / f"{PREREG_R1}.json")]
        + [str(C.REPO / p) for p in RH_SOURCES + ls_files],
        code=[str(C.CODE_DIR / f) for f in CODE]
        + [str(C.REPO / "dataprep_staging/h2_heads_fill" / f) for f in H2_CODE],
        input_trees=[str(C.AMP_WELLS)] + [str(C.REPO / p) for p in RH_TREES],
        parameters={"coverage_definition": C.COVERAGE_DEFINITION,
                    "min_coverage": C.MIN_COVERAGE, "min_obs_months": C.MIN_OBS_MONTHS,
                    "max_internal_run": C.MAX_INTERNAL_RUN, "max_lead": C.MAX_LEAD,
                    "max_trail": C.MAX_TRAIL, "layers": list(C.LAYERS),
                    "fresh_d": [C.FRESH_D.start, C.FRESH_D.stop - 1],
                    "sigma_snap": float(sres["sigma_snap"]["sigma_snap"]), "rh_well": rh,
                    "step_screen": stp["rule"],
                    "bridged_methods_kept": ["daily_bridge_partial"],
                    "validator_flags": [0, 1],
                    "use_note": "calibration_target only in the pre-declared I1 ladder rung "
                                "R5; validators use flags 0-1 only; the 2023-26 test set "
                                "stays the registered 158 wells"},
        fit_window_end=C.PRE_END, validation_report="validation.json",
        gate={"passed": True,
              "release_rule_passed": True,
              "plan_acceptance_passed": False,
              "criteria": (
                  f"RELEASE RULE (passed): {C.PREREG_ID} admission rule (envelope on "
                  f"observed months; bridged months only for wells passing {C.TEST_D} "
                  f"while the pooled RMSE rule passes, daily_bridge_partial rows only; "
                  f"declared fallback: other wells observed months only) and {PREREG_R1} "
                  f"({TEST_STEP} applied to every admitted well: none flagged; 09030212 "
                  f"held back by rule 3). PLAN ACCEPTANCE (NOT passed): 'each recovered "
                  f"well passes the H2 coverage strata' FAILS: {n_pass} of {len(adm)} "
                  f"admitted wells pass {C.TEST_D}; the other {len(adm) - n_pass} enter "
                  f"with flags 0-1 only. Plan acceptance part 2 (R5 rung gate) is "
                  f"evaluated by I1 after the freeze. WP H3 status: partial.")},
        allowed_uses=ALLOWED, forbidden_uses=FORBIDDEN,
        personal_data="none (no names, addresses or contacts; station coordinates only)",
        created=dt.date.today().isoformat())
    manifest.write_manifest(d, m)
    sup = {"schema": "dataprep-superseded/1", "product": C.PRODUCT,
           "written": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "writer": "dataprep_staging/h3_recovered_wells/r1_package.py",
           "current": d.name,
           "current_note": (f"{d.name} is the version for use: {len(adm)} wells, "
                            f"{len(T)} pre-origin targets (calibration_target in rung R5 "
                            "only). The plan acceptance criterion (each recovered well "
                            "passes the H2 coverage strata) is NOT met; see validation.json "
                            "acceptance_plan."),
           "superseded": [{
               "version": "v1", "superseded_by": d.name,
               "why": ("v1's validation.json recorded acceptance_plan."
                       "each_recovered_well_passes_on_observed_months = true by counting "
                       "observed-only wells as passing, contradicting the WP verdict (12 of "
                       "15 admitted wells fail H3.d_wells); v1 held 09030212 back on a "
                       "neighbour-offset threshold written after its result and applied to "
                       "no other well; its rh_probe.json carried post-origin availability "
                       "dates and its manifest omitted the probe's sources. Target rows: "
                       + (f"identical in {d.name}." if same_rows else
                          f"changed in {d.name} (see validation.json).")),
               "allowed_uses": ["after_the_fact_analysis"],
               "note": f"v1's own manifest still grants its declared uses; use {d.name}"}]}
    if d.name != "v2" and (d.parent / "v2" / "MANIFEST.json").exists():
        sup["superseded"].append({
            "version": "v2", "superseded_by": d.name,
            "why": ("first repair-round package: its rh_probe_preorigin.json still carried "
                    "two post-origin dates as dictionary keys (a 2023 count of vendor "
                    "record-window start years and the 2024-01-01 disuse date), so the "
                    "card's 'post-origin availability dates are redacted' was not true; its "
                    "card also overstated the step screen's resolution (a shift smaller than "
                    "the LOYO RMSE) instead of the shift it can leave unflagged (null z95 x "
                    "LOYO RMSE). Target rows, gate and acceptance record are identical."),
            "allowed_uses": ["after_the_fact_analysis"],
            "note": f"v2's own manifest still grants its declared uses; use {d.name}"})
    sp = C.REPO / "results/dataprep" / C.PRODUCT / "SUPERSEDED.json"
    if sp.exists():
        sup["previous_record_sha256"] = __import__("hashlib").sha256(sp.read_bytes()).hexdigest()
    sp.write_text(json.dumps(sup, indent=1, ensure_ascii=False))
    print(d, json.dumps(val["admitted"], ensure_ascii=False), val["targets"],
          "plan_acceptance", plan_ok, n_pass)


def card(d, pv, adm, T, dres, sres, stp, n_pass, pooled_ok, rule3):
    nb = stp["null_by_split"]
    q95 = sorted(v["z_q95"] for v in nb.values())
    ge2 = sorted(v["share_z_ge_2"] for v in nb.values())
    sh = sorted(r["null_z_q95"] * r["rmse_loyo"] for r in stp["per_well"]
                if r["sid"] in set(adm.sid) and r.get("rmse_loyo") is not None)
    lines = [
        f"# {pv}: recovered wells for rung R5 (WP H3, repair round)", "",
        "Wells outside the twin's 158-well set whose pre-origin record qualifies on observed "
        "months alone (plan WP H3), with their calibration targets for 2012-01..2022-12. "
        "Use them only in the pre-declared I1 ladder rung R5; the 2023-26 test set stays the "
        "registered 158 wells. Supersedes v1 and v2 (same target rows; corrected acceptance "
        "record, uniform level-shift screen, complete redaction).", "",
        f"Admitted: {len(adm)} wells ({adm.county.value_counts().to_dict()}; tiers "
        f"{adm.tier.value_counts().to_dict()}). Targets: {len(T)} well-months "
        f"(flags {T.flag.value_counts().to_dict()}).", "",
        "## Plan acceptance: NOT met", "",
        f"The plan asks that each recovered well pass the H2 coverage strata. Only {n_pass} "
        f"of the {len(adm)} admitted wells pass the per-well daily-bridge test H3.d_wells "
        f"({dres['n_wells_pass']}/{dres['n_wells']} test wells); H2's class-based bridge sd "
        "over-covers smooth wells and under-covers high-variability Yunlin wells. The other "
        "wells enter under the declared fallback with observed months only (flags 0-1), so "
        "no fill whose coverage failed is a target. The release rule passed; the plan "
        "criterion did not. R5's own rung gate (plan acceptance part 2) is decided by I1.",
        "",
        "## How a well qualifies", "",
        "- Envelope on observed months only (L1 v3 C20 months, flag 0/1; fills never count): "
        "not a model well; in an active cell with the L1 repaired coordinates; raw layer "
        "code 1-4; coverage >= 0.80 of 2012-01..2022-12; >= 36 observed months; internal "
        "runs <= 24 months; leading run <= 66 months; record reaching 2022-12.",
        f"- Bridged months (flag 2, method daily_bridge_partial, rows of "
        f"heads/L2_filled_preorigin/v4) only for wells that pass H3.d_wells, and only "
        f"because the pooled bridge RMSE rule passed ({pooled_ok}).",
        "- Level-shift screen H3.step_screen (H3.repair.v1), applied to every admitted "
        "well: a ridge model on the 4 best-correlated non-sibling model-well neighbours, "
        "fitted before the well's longest gap, gives the mean residual after it, in units "
        "of its leave-one-year-out RMSE (z); a well is held back when z exceeds the model "
        f"wells' null at p <= 0.05. Null 95th percentiles of z: {q95[0]:.2f}-{q95[-1]:.2f}; "
        f"share of model wells with z >= 2: {ge2[0]:.2f}-{ge2[-1]:.2f}. Result: "
        f"{len(stp['admitted_v1']['flagged'])} flagged, "
        f"{len(stp['admitted_v1']['untestable'])} untestable.",
        f"- 09030212 (R-h find; once-daily record 2019-2022): NOT ADMITTED: {rule3}. Its "
        "once-daily months stay in reports/H3_evidence for a sensitivity run only.", "",
        "## Uses", "",
        "- calibration_target: rung R5 only; likelihood variance sigma_obs_m^2 + head_sd_m^2 "
        "(u2_weight = sigma_obs^2 / (sigma_obs^2 + sd^2)).",
        "- datum_fit: the guard serves flags 0-1.",
        "- **validator: use flags 0-1 only** (`flag <= 1`). The guard's default flags for "
        "'validator' include flag 2; bridged rows are imputed and never serve as a check "
        "of a reported number (U1).",
        "- after_the_fact_analysis. Everything else is forbidden (manifest).", "",
        "## Columns", "",
        "`targets_preorigin.parquet` keeps the columns of heads/L2_filled_preorigin/v4 "
        "(value = head_m, sd, flag, method, sigma_obs_m, u2_weight, ...) plus `tier` and "
        "`cell`. `recovered_wells.csv` adds the H3.d_wells and H3.step_screen results per "
        "well and whether each segment of the split record would meet the envelope alone.",
        "",
        "## Known limits", "",
        "- 12 admitted Yunlin wells lost 2019-01/02..2019-07/12 (the 60-minute to 10-minute "
        "transition), and the 3 Changhua wells 4-8 months in 2020-21. The step screen finds "
        "no well-specific level shift across those gaps beyond what continuous model wells "
        "show, but its resolution is coarse: a shift up to the null's 95th percentile of z "
        f"times the well's LOYO RMSE ({sh[0]:.2f}-{sh[-1]:.2f} m; column "
        "step_shift_unflagged_up_to_m) can go unflagged. Per-well datum fits in R5 should "
        "stay per well, not per segment, unless I1 tests otherwise.",
        "- Two Changhua wells (07170211, 07170221) are admitted because H1a repaired their "
        "swapped coordinates; the twin's raw parse rejects them.",
        "- Layer codes are as stored by WRA. Six apex wells with no code (layer proposals "
        "flag 5) are listed in layer_review.csv and NOT admitted.",
        "- R-h: 18 of the 19 on-grid wells whose records stop in 2018-12 have no later record "
        "in the routes checked; the credentialed vendor route was not run in this WP "
        "(rh_probe_preorigin.json). Post-origin availability dates are redacted here.",
        "- No post-origin data in this product; a secondary 2023-26 test set for these wells "
        "needs the vendor fetch (not done).", ""]
    (d / "PRODUCT_CARD.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
