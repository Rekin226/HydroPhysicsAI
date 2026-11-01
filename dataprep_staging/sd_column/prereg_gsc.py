"""Registration X5.gnss_seasonal_column.v1 (harness.prereg3, kind 'test'; round 7 rank 10):
one GNSS seasonal hold-out of the page column's elastic response, design-time (window
flagged). Merges subsidence lens A R1 G2, lens B R1 and lens C R1 into ONE declaration with
ONE scored look; every other column is a report-only arm hashed before the look.

Ready only after: SD.cut_response_spread.v1 is scored; ``run_gsc.sh model-side`` froze the
model side; ``run_gsc.sh check-model`` reproduced it byte for byte; ``run_gsc.sh g0oc``
passed under the SCORED rule (both observation scales) on that model side; and the lead
logged G0-OC as a look on unit G0_LOOK_UNIT whose note carries g0oc.json's sha256::

    prereg3.record_look(['X5GSC:g0oc_synthetic'], 'synthetic:2014-08..2022-12', 'X5',
        'integrity_check', members={'X5GSC:g0oc_synthetic': {'none': 'synthetic worlds on
        the frozen model side', 'description': 'G0-OC of X5.gnss_seasonal_column.v1'}},
        note='g0oc.json sha256 <sha>')

``--validate`` writes nothing. ``--write`` (the lead only) registers once with
``--expect-self-sha``. Every unit window and data scope comes from the frozen model side.

    PY=~/miniforge3/envs/hydro/bin/python
    $PY -I dataprep_staging/sd_column/prereg_gsc.py --validate --quiet
    $PY -I dataprep_staging/sd_column/prereg_gsc.py --write --quiet --expect-self-sha <sha>
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (str(HERE.parent), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import gsc_estimator as E  # noqa: E402
import gsc_model_side as M  # noqa: E402
import gsc_synth as G  # noqa: E402
import score_gsc as S  # noqa: E402
import sdlib as L  # noqa: E402
from harness import prereg3  # noqa: E402

NA = prereg3.not_applicable
ID = L.GSC_ID
THIS = f"{L.X}/prereg_gsc.py"
TESTS = f"{L.X}/tests/test_sd_column.py"
RULINGS = "results/dataprep/rulings/user_rulings_2026-11-01b.json"
G0 = f"results/dataprep/reports/{ID}/g0oc.json"
G0_LOOK_UNIT = "X5GSC:g0oc_synthetic"
LOOKS = "results/dataprep/preregistration/looks.jsonl"
#: the judge's declared windows, used only while the model side is pending (never registered:
#: readiness refuses until the model side exists)
PENDING_SCOPE = {"primary": "2014-08..2022-12", "ngl_stratum": "2014-08..2019-12"}


def repo() -> Path:
    return S.repo()


def model_block(r: Path) -> tuple[dict, dict | None]:
    d = r / M.OUT_DIR
    if not (d / "model_side.json").exists():
        return {"model_side_json": f"{M.OUT_DIR}/model_side.json",
                "model_side_json_sha256": "missing"}, None
    ms, _ = M.load(d)
    return {"model_side_json": f"{M.OUT_DIR}/model_side.json",
            "model_side_json_sha256": L.sha(d / "model_side.json"),
            "model_series_npz": f"{M.OUT_DIR}/model_series.npz",
            "series_digest": ms["series_digest"],
            "primary_arm": ms["primary_arm"], "arms": sorted(ms["arms"]),
            "units": ms["units"], "f_subs_page": ms.get("f_subs_page"),
            "omitted_arms": ms.get("omitted_arms")}, ms


def power_statement(g: dict) -> str:
    """The scored-rule operating characteristics as numbers (declared in the registration)."""
    m, w = g["marginal"], g["worst_cell"]
    s = (f"under the scored rule (band at both scales) P(consistent | r = 1) = "
         f"{m['p_consistent_r1']:.3f} pooled over the grid (worst cell "
         f"{w['p_consistent_r1']:.3f}; scale-1 reading alone {m['p_consistent_r1_scale1']:.3f}"
         f"); coverage {m['coverage_r1']:.3f}; P(consistent | r = 0.4 or 2.5) worst cell "
         f"{max(w['p_consistent_r04'], w['p_consistent_r25']):.3f}; P(S label | r = 1) = "
         f"{m['p_S_label_r1']:.3f}")
    if "T1n_r1" in m:
        s += (f"; T1n at r = 1: P('larger' or 'smaller') = "
              f"{m['T1n_r1']['p_larger_or_smaller']:.3f}, P(inconclusive) = "
              f"{m['T1n_r1']['p_inconclusive']:.3f}")
    return s


def g0_block(r: Path) -> dict:
    p = r / G0
    if not p.exists():
        return {"g0oc_json": G0, "sha256": "missing", "pass": False,
                "pass_rule": G.PASS_RULE, "pooling": G.POOLING,
                "power_statement": "pending: G0-OC not run"}
    g = json.loads(p.read_text())
    return {"g0oc_json": G0, "sha256": L.sha(p), "pass": bool(g["pass"]),
            "pass_rule": g.get("pass_rule"), "pooling": g.get("pooling"),
            "marginal": g["marginal"], "worst_cell": g["worst_cell"],
            "power_statement": power_statement(g),
            "model_side_json_sha256": g.get("model_side_json_sha256")}


UNIT_NOTE = ("design-time units, NOT fresh in the U7 sense: seen in product QC and fusion "
             "(S3, S4b, S5, S1b looks; WRA stations co-located with ring schools); never "
             "compared with the twin before")


def units(ms: dict | None) -> tuple[dict, dict, dict]:
    """(T1 unit, T1n unit, {test: data_scope}) from the frozen model side (members in both
    id forms, windows and scopes derived from the station windows)."""
    out, scopes = [], {}
    for key, name, desc in (
            ("primary", S.UNIT_PRIMARY,
             "Yunlin on-grid GNSS stations of the primary unit (gsc_model_side station rule: "
             "WRA Yunlin cells except ANES, NGL HUWE and DNAN only), monthly heights"),
            ("ngl_stratum", S.UNIT_NGL,
             "NGL HUWE and DNAN (named by the judge, rank 10), the least-seen co-primary "
             "stratum")):
        names = ms["units"][key] if ms else []
        sc = M.unit_scope(ms["stations"], names) if names else {
            "window": PENDING_SCOPE[key], "data_scope": f"gnss:{PENDING_SCOPE[key]}"}
        mem = {"ids": S.station_ids(names)} if names else {
            "none": "model side pending (registration refused until it exists)"}
        out.append({name: {"description": f"{desc}; {UNIT_NOTE}", **mem,
                           "window": sc["window"]}})
        scopes[key] = sc["data_scope"]
    return out[0], out[1], scopes


DECISION = {
    "statistic": "r_A = sum_s A_obs(s) / sum_s A_mod(s): annual-harmonic subsidence amplitude "
                 "(gsc_estimator: per-station trend removed, per-year annual + semiannual "
                 "fit, modulus of the mean annual vector), observed months only, obs and "
                 "model on the same months",
    "interval": f"90 % two-stage station-block bootstrap over years, B {E.B_SCORE}, seed "
                f"{E.SEED}, with the declared nuisance draw (gsc_estimator.NUISANCE)",
    "readings": "CI inside [0.667, 1.5] -> 'consistent'; CI wholly below 0.667 -> 'larger' "
                "(model elastic response larger than observed); wholly above 1.5 -> "
                "'smaller'; otherwise 'inconclusive'",
    "scale_robustness": "the reading must be the same at observation scale s = 1 and s = "
                        "1/1.42, else 'inconclusive'",
    "inclusive_bounds": True}


def page_mapping() -> dict:
    return {
        "ruling": "R7.gnss_seasonal_S (user, 2026-11-01): a pass may carry 'S (design-time, "
                  "window flagged)'; ANES and CHSG report-only",
        "C32_new_claim": "seasonal elastic subsidence at Yunlin GNSS",
        "consistent": "C32 S (design-time, window flagged) if T1n is not 'larger' or "
                      "'smaller'; else N with both readings. C27 gains 'elastic mechanism "
                      "consistent in size at the seasonal scale (design-time)'",
        "larger": "C32 N; C27 'no direction' -> 'short-term elastic size overstated; step C "
                  "and GNSS agree'; C03 note 'size likely overstated, no correction'; C26 "
                  "gains the coastal direction",
        "smaller": "C32 N with the direction; C27 gains it; no correction",
        "inconclusive": "C32 N; C27 unchanged with a power note",
        "changhua_arm": "C13c wording only (explains the Yunlin/Changhua gap); C13c stays W",
        "f_subs": "a declaration (share of the cut30 benefit a seasonal test exercises), not "
                  "a gate; printed beside any C27 wording",
        "never": "no column switch, no subset range, no correction factor; selection on the "
                 "gate-failed product is never gate-exempt; C03, C09, C31 do not move"}


DISCLOSURES = [
    "the planner and review looks: S3 (product QC, A1/A2 second looks, after-the-fact twin "
    "rates 2023-25), S4b (WRA 2020-22 seasonal amplitude as InSAR validator, amplitude-easting "
    "and harmonic looks), S5 (monthly field, 2012-19 NGL hyperparameter fit, WRA 2020-22), "
    "S1b (co-located WRA stations and NGL near rings)",
    "WRA Yunlin stations are mostly co-located with ring schools whose seasonal compaction "
    "mechanism.json scored (ratio 0.37); the direction prior is partly known",
    "ANES and CHSG annual amplitude and rate were seen by a planner (looks.jsonl "
    "2026-11-01T11:14Z, unit R7:gnss_anes_chsg_disclosure): report-only",
    "the page column was chosen knowing the 2024-25 pile seasonal R2 of 0.92 (U7, E06)",
    "stations.csv carries a per-station annual_amp_m computed at product build: the statistic "
    "pre-exists; the drafting agent read stations.csv columns 1-10 only",
    "gnss_clean/preorigin/v4 failed its gate; read only as design_time_evidence (gate-exempt), "
    "flags 0-1",
    "the 2021 record-low year adds non-sinusoidal compaction to the WRA years (report-only "
    "line without 2021)",
    "G0-OC is evaluated on the SCORED rule (band at both observation scales 1 and 1/1.42); "
    "a 'consistent' reading then needs the 90 % interval inside about [0.947, 1.5], so the "
    "S path is hard by construction; the numbers are in g0oc.power_statement",
    "G0-OC pooling: power and coverage are means over the declared nuisance grid (each cell "
    "weighted equally); false 'consistent' is checked in every cell; worst cells reported",
    "G0-OC worlds drop each station-month with probability 0.10 (declared; the real "
    "missingness after flags 3/9 and the >= 10 days rule was not read)",
    "the bootstrap nuisance draw is mean-neutral (projection on the station's annual vector; "
    "signed thermal term); it widens the interval without shifting it",
    "NGL units are pinned by name (judge rank 10): ERLN (Yunlin by the county mask, an S3 "
    "design unit) is in the report-only Changhua arm; JSAN (apex, east of the plan box) is "
    "report-only"]


def doc() -> dict:
    r = repo()
    mb, ms = model_block(r)
    g0 = g0_block(r)
    u1, u2, sc = units(ms)
    fz = {p: L.sha(r / p) for p in S.FROZEN}
    common = {"decision_rule": DECISION, "label": "design-time (window flagged)",
              "use": S.USE, "flags": list(S.FLAGS), "common_mode": S.REMOVAL,
              "min_days_per_month": S.MIN_DAYS, "min_months_per_year": E.MIN_MONTHS,
              "min_years_per_station": E.MIN_YEARS}
    tests = [
        {"test_id": S.T1, "data_scope": sc["primary"],
         "fresh_units": list(u1), "external_units": u1, "role": "primary (scored once)",
         "claim": "C32 (new): seasonal elastic subsidence at Yunlin GNSS", **common},
        {"test_id": S.T1N, "data_scope": sc["ngl_stratum"],
         "fresh_units": list(u2), "external_units": u2,
         "role": "co-primary stratum (scored once in the same session)",
         "claim": "C32's least-seen stratum; can hold C32 at N (page_mapping)", **common}]
    return {
        "id": ID, "wp": S.WP, "kind": "test",
        "title": "GNSS seasonal hold-out of the page column's elastic response at Yunlin "
                 "(pooled amplitude ratio, NGL stratum co-primary); design-time, window "
                 "flagged",
        "priors": "Direction prior 'larger' (ring seasonal ratio 0.37, partly attenuation; "
                  "step C slopes 0.22/0.40); judge: G0-OC ~0.75, informative verdict ~0.55, "
                  "'consistent' ~0.3. The ANES/CHSG look is disclosed and both are "
                  "report-only.",
        "tests": tests,
        "zones": "county from the twin county mask at each station's twin_cell; zone from "
                 "zones.fan_zones (report-only strata distal/mid)",
        "burn_in": "model side: SD forwards' free hindcast 2012-01..2022-12 (main parameter "
                   "set, gain 0, column on its paired free heads)",
        "folds": NA("one pooled statistic; no folds"),
        "fresh_units": NA("listed per test"),
        "thresholds": {"band": list(E.BAND), "scales": list(E.SCALES), "level": E.LEVEL,
                       "a_min_mod_m": M.A_MIN_MOD_M, "min_span_days": M.MIN_SPAN_DAYS,
                       "g0oc_pass": G.PASS},
        "gate_scores": NA("declared reading bands; no weighted score"),
        "primary_model": f"page column {L.PAGE_COLUMN_SHA8} (stage3_datum_gate/"
                         "coupled_leveling) on the page theta's main-member free hindcast",
        "model_side": mb, "g0oc": g0,
        "g0oc_design": {"grid": {"r": G.R_GRID, "loading_mm": G.LOADING_MM,
                                 "thermal_mm": G.THERMAL_MM, "sigma_mm": G.SIGMA_MM,
                                 "rho": G.RHO, "jumps": G.JUMPS, "p_jump": G.P_JUMP,
                                 "jump_mm": G.JUMP_MM, "survive": G.SURVIVE},
                        "p_miss": G.P_MISS, "n_rep": G.N_REP, "B": G.B_G0,
                        "seed": G.SEED, "pass_rule": G.PASS_RULE, "pooling": G.POOLING,
                        "strata": "T1n characterised from the same worlds",
                        "fail": "the test is not run"},
        "report_only": ["zone strata distal/mid", "WRA without 2021", "no common-mode "
                        "removal", "wra_hf removal (WRA)", "phase lag obs - model",
                        "slope form (step C)", "full-window A_mod", "every other column "
                        "(Tier A/B arms) on the primary unit", "Changhua arm (NGL "
                        + ", ".join(M.NGL_CHANGHUA_ARM) + " by name and WRA Changhua "
                        "cells; page, ccz, q0c_csk1, x5r1_cs)", "ANES and CHSG rows",
                        "other on-grid NGL stations (JSAN)"],
        "page_mapping": page_mapping(),
        "disclosures": DISCLOSURES,
        "frozen_code": fz,
        "scoring": {"order": "verify registration and inputs -> MANIFEST use check -> "
                             "prereg3.assert_can_score (logs T1 and T1n) and one report_only "
                             "look -> open GNSS daily values once -> report.json (mode 'x')",
                    "runner": f"bash {L.X}/run_gsc.sh score"},
        "attempt_policy": {
            "scored_runs": "T1 and T1n scored once, in one session; engineering failures "
                           "before the GNSS read are free and logged",
            "pre_declared_alternatives": [],
            "exploratory_followups_max": 1,
            "followup_label": "post-failure, exploratory; never validation; cannot promote",
            "thresholds_frozen": True},
        "rulings": {"R7.gnss_seasonal_S": "applied (S design-time on a pass; ANES and CHSG "
                                          "report-only)",
                    "WRA 2020-22 in the primary": "kept with disclosure (judge default); NGL "
                                                  "stratum co-primary",
                    "ngl_offfan removal": "primary form (judge)"},
        "inputs": ([{"path": p, "sha256": s} for p, s in fz.items()]
                   + [THIS, TESTS, RULINGS, M.STATIONS_CSV, f"{M.PRODUCT}/MANIFEST.json",
                      f"{M.PRODUCT}/PRODUCT_CARD.md", M.SD_REPORT,
                      f"{M.OUT_DIR}/model_side.json", f"{M.OUT_DIR}/model_series.npz", G0]),
        "interpretation": "A pass says the column's seasonal (elastic) subsidence at Yunlin "
                          "GNSS cells matches the observed within x1.5 at both frame scales "
                          "(design-time). It checks the chain damped heads x Ske, not Ske "
                          "alone, and not the 1-3 yr or long-run cut size.",
        "written_by": THIS,
    }


def blocking(d: dict) -> list:
    idx = prereg3.membership_index()
    for t in d["tests"]:
        for u, v in t["external_units"].items():
            if not idx.known(u):
                m = prereg3._declared_members(v)
                if m is not None:
                    idx.members[idx.canon(u)] = m
    return prereg3.blocking_looks(d["tests"], None, idx=idx,
                                  test_time=dt.datetime.now(dt.timezone.utc))


def readiness(d: dict) -> list[str]:
    r = repo()
    out = []
    for rec in d["inputs"]:
        p = rec["path"] if isinstance(rec, dict) else rec
        if not (r / p).exists():
            out.append(f"input {p} is missing")
    if d["model_side"]["model_side_json_sha256"] == "missing":
        out.append("model side not frozen: run_gsc.sh model-side")
    if not d["g0oc"]["pass"]:
        out.append("G0-OC missing or failed (a fail means the test is not run)")
    elif d["g0oc"]["model_side_json_sha256"] != d["model_side"]["model_side_json_sha256"]:
        out.append("G0-OC ran on another model side")
    elif not g0_logged(r, d["g0oc"]["sha256"]):
        out.append(f"G0-OC not logged: no look on {G0_LOOK_UNIT} whose note carries "
                   f"g0oc.json's sha256 {d['g0oc']['sha256']}")
    if (r / f"results/dataprep/preregistration/{ID}.json").exists():
        out.append(f"{ID} is already registered")
    if (r / S.REPORT).exists():
        out.append(f"{S.REPORT} exists")
    return out


def g0_logged(r: Path, g0_sha: str) -> bool:
    """A look on G0_LOOK_UNIT whose note carries the g0oc.json sha256 exists."""
    p = r / LOOKS
    if not p.exists():
        return False
    return any(G0_LOOK_UNIT in (lk.get("units") or []) and g0_sha in str(lk.get("note", ""))
               for lk in prereg3.read_chained(p))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--validate", action="store_true")
    g.add_argument("--write", action="store_true")
    ap.add_argument("--expect-self-sha", default=None)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    d = doc()
    if not a.quiet:
        print(json.dumps(d, indent=1, ensure_ascii=False, sort_keys=True))
    print("document canonical sha256:", L.canonical_sha(d))
    errs = prereg3.validate_prereg(d)
    print("validate_prereg:", errs or "valid")
    block = blocking(d)
    print("blocking looks (dry run):", block or "none")
    pending = readiness(d)
    print("readiness:", pending or "ready to register")
    self_sha = L.sha(Path(__file__).resolve())
    print(f"this script sha256: {self_sha}")
    bad = bool(errs or block or pending)
    if a.validate:
        print("valid; not written" if not bad else "NOT READY: fix the problems above")
        return 1 if bad else 0
    if a.expect_self_sha != self_sha:
        print("REFUSED: --write needs --expect-self-sha equal to this script's sha256")
        return 1
    if bad:
        print("REFUSED: not ready to register")
        return 1
    path, h = prereg3.write_prereg(d)
    print(f"registered {path} sha256 {h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
