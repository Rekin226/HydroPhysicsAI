"""Gates of X5.gnss_seasonal_family.v1 (model-only, synthetic; nothing here reads a GNSS,
pile or electricity value): the design declaration, G-sep and G0.

Order (the lead runs it; nothing runs on import):

1. ``gsf_model.py supplement`` then ``gsf_model.py engineering`` write pile_model.{json,npz}
   and engineering.json once (model-only: v1 model side 65b927cb, SD v2 report 5cfc9ea7,
   the sealed SD forwards at the two pile cells, the dedup check, cut30 cluster/far).
2. ``declare --seed S`` validates ``draft_declaration(S)`` (harness.prereg3, kind
   'declaration', id X5.gnss_seasonal_family.gates.v1: code shas, engineering.json sha256,
   the seed, the G-sep and G0 designs and pass rules, the members and the rulings file).
   The LEAD registers it (prereg3.write_prereg) BEFORE ``gsep`` (X5.n2c_reading / N2CS
   precedent). Nothing here writes a registration.
3. ``gsep`` refuses unless the declaration is registered, the code is unchanged since it and
   engineering.json is the registered one; runs G-sep once on the declared seed; writes
   gsep.json once (mode 'x'); appends attempts.jsonl.
4. ``g0 [--workers N]`` refuses unless gsep.json passed and is the file gsep logged, and the
   same declaration checks hold; runs G0 once; writes g0.json once; appends attempts.jsonl.
   ``--workers`` does not change the result (per-task SeedSequence children).
   A G0 fail is final: no GNSS look, and ``closure_line`` gives the honest closure text.

    PY=~/miniforge3/envs/hydro/bin/python
    $PY -I dataprep_staging/gnss_family/gsf_gates.py declare --seed <int>   (validate only)
    $PY dataprep_staging/gnss_family/gsf_gates.py gsep
    $PY dataprep_staging/gnss_family/gsf_gates.py g0 --workers 8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(HERE), str(REPO / "dataprep_staging" / "sd_column"),
           str(REPO / "dataprep_staging")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gsf_const as K  # noqa: E402
import gsf_model as MD  # noqa: E402
import gsf_synth as G  # noqa: E402

rp, sha256, log_attempt = MD.rp, MD.sha256, MD.log_attempt


def _na(reason: str) -> dict:
    return {"not_applicable": reason}


def design() -> dict:
    """The G-sep and G0 designs exactly as gsf_synth runs them."""
    return {
        "screen_members": list(K.SCREEN), "dedup": K.DEDUP,
        "report_only_arms": {**K.SCREEN_REPORT_ONLY,
                             **{c: "Changhua tier (SD v2 'CH')" for c in K.CH_TIER}},
        "unit": "v1 model side primary unit (11 distal + 8 mid Yunlin stations; NGL HUWE/DNAN "
                "2014-08..2019-12, WRA 2020-01..2022-12), each member's own free-hindcast "
                "series on the station months",
        "estimator": "gsf_estimator: per-year annual vectors (v1), component-wise median, "
                     "amplitude; T2 = median distal amplitude, T1 = T2 / median mid amplitude; "
                     "station-block (within zone) x year bootstrap, obs and members together",
        "screen_rule": {"T1_band_log": list(K.T1_BAND), "T2_band_log": list(K.T2_BAND),
                        "allowance": K.ALLOW, "frame_max": K.FRAME_MAX, "level": K.LEVEL,
                        "rule": "excluded if the 90 % interval of log(T_obs/T_m) lies wholly "
                                "outside the band, on T1 or on T2"},
        "readings": {"a": "survivors meet the cluster and miss the far members",
                     "b": "survivors miss the cluster and meet the far members",
                     "c": "survivors meet both", "d": "no survivor"},
        "cut30": {"source": K.SD_REPORT, "sha256": K.SD_REPORT_SHA,
                  "field": f"per_column[c].summary['{K.CUT30_HORIZON}']['{K.CUT30_GROUP}']"
                           f"['{K.CUT30_FIELD}']", "cluster_x": K.CLUSTER_X, "far_x": K.FAR_X},
        "gsep": {"cells": list(K.GSEP_CELLS), "n_rep": K.GSEP_NREP, "B": K.B_G0,
                 "truth": K.PAGE, "origin": "the LEAD's design choice (review B2), not the "
                                            "plan's",
                 "rule": "pass only if every far member is separated from EVERY cluster "
                         "member on noise-free model points: |dlog T1| > hw_T1 + log 1.25, or "
                         "|dlog T2| > hw_T2 + log 1.25 + log 1.42 (the screen's T2 band); hw = "
                         "max over cells of the median 90 % half-width of log(T_obs/T_page) in "
                         "page-truth worlds; the T1 route needs hw_T1 below about 0.08-0.13 on "
                         "the frozen model side, so a pass is expected through T2 only",
                 "stream": ["seed", G.STREAM_GSEP]},
        "g0": {"cells": G.cells(), "n_cells": len(G.cells()), "truths": "every screen member",
               "n_rep": K.N_REP, "B": K.B_G0, "pass": K.G0_PASS, "cell": K.G0_CELL,
               "world": G.__doc__.split("World (one true member t, one cell):")[1]
               .split("G0 cells:")[0].strip(),
               "params": {"rho": K.RHO, "p_jump": K.P_JUMP, "jump_mm": K.JUMP_MM,
                          "survive": K.SURVIVE, "p_miss": K.P_MISS,
                          "thermal_peak_month": K.THERMAL_PEAK_MONTH,
                          "frame_uniform": K.FRAME_UNIFORM,
                          "ref_seasonal_mm": K.REF_SEASONAL_MM, "zone_frame": K.ZONE_FRAME,
                          "record_low_mm": K.RECORD_LOW_MM,
                          "record_low_ramp": K.RECORD_LOW_RAMP,
                          "coast_loading_mm": K.COAST_LOADING_MM},
               "jump_rule": "none on the gated path (review S1); worlds keep every jump",
               "proximal_seasonal_truth": "the spec's 'proximal seasonal truth' cell",
               "stream": ["seed", G.STREAM_G0]}}


DISCLOSURE = [
    "DECIDED BY T2 (review B3): " + K.DECIDED_BY_T2 + "; a screen pass carries '"
    + K.PASS_LABEL + "'",
    *K.FOUR_DISCLOSURES,
    "the files live in dataprep_staging/gnss_family/ by the lead's choice (the plan named "
    "sd_column/, whose files stay untouched and hash-bound)",
    "n_rep is 1000 per G0 cell (review B1: at 200 the every-cell 0.90 rule fails by chance "
    "about half the time)",
    "no GNSS, pile or electricity value is read by G-sep or G0; the worlds are the frozen "
    "v1 model side's member series plus declared synthetic nuisances",
    "v1's G0-OC (g0oc.json 35ebf516) failed under the scored two-scale rule; this redesign "
    "is a pre-look redesign (no GNSS value was ever read for v1; ruling R8.5)",
    "the builder printed the gnss_clean/preorigin/v4 MANIFEST gate block while checking "
    "allowed uses: it holds the product-QC A2 statistics (2021 extra compaction, pooled "
    "amplitude ratio against rings/piles at 7 units, both offered common modes) and A1 step "
    "recovery rates; no seasonal amplitude; no parameter was set from it",
    "the builder read deep_benchmarks/v2 README.md (product QC gate text) and mapping.csv "
    "metadata columns, and the sensor/variant/feed/flag/month columns of monthly.parquet for "
    "TKJS(1)/NTUH(1) (row counts by feed and flag only; no value column)",
    "the builder ran the engines only on a FABRICATED stand-in unit (gsf_synth."
    "standin_stations), never on the model side, to time and test them",
    "the model numbers quoted by the planner (T1/T2 per arm, from model_side.json) are "
    "model-only; the G-sep pass rule (far vs every cluster member, with the screen's "
    "log 1.25 allowance) is the LEAD's design choice after review B2, not the plan's (the "
    "plan's 'at least 2 members differ from the page' rule is superseded)"]


def draft_declaration(seed: int, *, engineering: dict | None = None,
                      engineering_sha256: str | None = None) -> dict:
    """X5.gnss_seasonal_family.gates.v1 (kind 'declaration'). ``seed`` is required.
    Refuses unless the engineering record exists and passed (or is passed in, for tests)."""
    K.check_seed(seed)
    if engineering is None:
        p = rp(K.ENGINEERING)
        if not p.exists():
            raise SystemExit("REFUSED: engineering.json missing (gsf_model.py engineering)")
        engineering = json.loads(p.read_text())
        engineering_sha256 = sha256(p)
    if not engineering.get("pass"):
        raise SystemExit("REFUSED: the engineering record did not pass")
    code = MD.code_sha256()
    inputs = ([{"path": p, "sha256": h} for p, h in code.items() if h != "missing"]
              + [p for p in (K.V1_MODEL_JSON, K.V1_SERIES_NPZ, K.SD_REPORT, K.V1_G0,
                             K.PILE_MODEL_JSON, K.PILE_MODEL_NPZ, K.ENGINEERING, K.RULINGS,
                             K.PILE_MAPPING) if rp(p).exists()])
    return {
        "id": K.DECL_ID, "wp": K.WP, "kind": "declaration",
        "title": "Gates of X5.gnss_seasonal_family.v1: G-sep (model-only member separation) "
                 "and G0 (synthetic decidability of the scale-free family screen); "
                 "model-only, before any GNSS amplitude is read",
        "priors": "v1 (X5.gnss_seasonal_column.v1) failed its G0-OC under the scored "
                  "two-scale rule and was never registered or scored. Round 8 rank 11 "
                  "redesigns it before any look as one consolidated declaration: a "
                  "scale-free family screen (T1 primary, T2 secondary) of SD v2's Tier A+B "
                  "columns, v1's directional arm as a secondary, a deep-pile frame bound. "
                  "This declaration freezes the two gates and their seed first.",
        "rulings": {K.RULING_ID: {"file": K.RULINGS, "sha256": K.RULINGS_SHA,
                                  "ruling_label": K.RULING_LABEL,
                                  "pass_label": K.PASS_LABEL}},
        "decided_by_T2": K.DECIDED_BY_T2,
        "design": design(), "code_sha256": code,
        "engineering": {"path": K.ENGINEERING, "sha256": engineering_sha256,
                        "cut30_cm": engineering.get("cut30_cm"),
                        "cluster_far": engineering.get("cluster_far"),
                        "dedup": engineering.get("dedup")},
        "seeds": {"declared": seed, "gsep": [seed, G.STREAM_GSEP],
                  "g0": [seed, G.STREAM_G0],
                  "u2_and_screen": "the registration's own seed (streams "
                                   f"{K.STREAM_SCREEN}, {K.STREAM_U2}, {K.STREAM_SCREEN_QC})",
                  "rule": "numpy SeedSequence([seed, stream]); G0 task k uses child k"},
        "zones": "zones.fan_zones distal/mid at each primary station's twin cell (v1 model side)",
        "burn_in": _na("synthetic; AR(1) noise starts stationary"),
        "folds": _na("synthetic declaration; no folds"),
        "gate_scores": _na("declared pass rules; no weighted score"),
        "primary_model": "SD v2 Tier A+B columns (deduplicated, q0_nh report-only) on their own "
                         "free heads (v1 model side); the characterised object is the screen",
        "inputs": inputs,
        "tests": _na("synthetic declaration: no fresh unit is consumed"),
        "fresh_units": _na("no observation is scored"),
        "thresholds": {"g0": K.G0_PASS, "gsep_rule": "far vs every cluster member (B2)",
                       "t1_band_log": list(K.T1_BAND), "t2_band_log": list(K.T2_BAND)},
        "attempt_policy": {
            "scored_runs": "G-sep once, then G0 once; gsep.json and g0.json written once "
                           "(mode 'x') and final, whatever they show",
            "engineering_retries": "free before gsep.json / g0.json exists (crash, missing "
                                   "file); each logged in attempts.jsonl; a fix that changes "
                                   "bound code needs a v2 that discloses the attempts",
            "v2_scope": "code only; never the grid, the pass rules, the seed or the members",
            "g0_fail": "final: no GNSS look; X5.gnss_seasonal_family.v1 is not registered and "
                       "the closure line is written", "exploratory_followups_max": 0,
            "thresholds_frozen": True},
        "interpretation": "G-sep says whether the frozen family has members the screen could "
                          "tell apart from the page column; G0 says whether the declared "
                          "screen keeps the true member and drops members x1.5 off in cut30 "
                          "under declared nuisances. Neither is evidence about the aquifer.",
        "rule_risks_disclosure": DISCLOSURE,
        "no_prior_look_statement": "No G-sep or G0 world on the model side has been drawn; no "
                                   "GNSS value has been read for v1 or this redesign.",
        "written_by": f"{K.X}/gsf_gates.py"}


def validate_declaration(doc: dict) -> list[str]:
    from harness import prereg3

    return prereg3.validate_prereg(doc)


def declared() -> tuple[dict, str]:
    """(registered declaration, sha256) or SystemExit naming why it refuses."""
    from harness import prereg3

    try:
        doc, h = prereg3.load_prereg(K.DECL_ID)
    except Exception as exc:                       # noqa: BLE001 - any failure refuses
        raise SystemExit(f"REFUSED: {K.DECL_ID} is not registered ({type(exc).__name__})") \
            from exc
    now = MD.code_sha256()
    bad = [k for k, v in doc["code_sha256"].items() if now.get(k) != v]
    if bad:
        raise SystemExit(f"REFUSED: code changed since {K.DECL_ID}: {bad}")
    if sha256(K.ENGINEERING) != doc["engineering"]["sha256"]:
        raise SystemExit("REFUSED: engineering.json is not the declared one")
    if sha256(K.RULINGS) != K.RULINGS_SHA:
        raise SystemExit("REFUSED: the rulings file changed")
    return doc, h


def screen_stations() -> dict:
    ms, series = MD.load_v1()
    return MD.primary_stations(ms, series, list(K.SCREEN))


def run_gsep_once() -> dict:
    doc, h = declared()
    out = rp(K.GSEP_OUT)
    if out.exists():
        raise SystemExit(f"REFUSED: {K.GSEP_OUT} exists; G-sep runs once")
    seed = int(doc["seeds"]["declared"])
    log_attempt({"event": "gsep_start", "declaration_sha256": h})
    cut = doc["engineering"]["cut30_cm"]
    res = G.run_gsep(screen_stations(), list(K.SCREEN), cut, seed=seed)
    res.update({"declaration": K.DECL_ID, "declaration_sha256": h, "written": MD.utc()})
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as fh:
        fh.write(json.dumps(res, indent=1, sort_keys=True, default=float) + "\n")
    s = sha256(out)
    log_attempt({"event": "gsep_run", "gsep_sha256": s, "pass": res["pass"]})
    return {"pass": res["pass"], "sha256": s, "n_separated": res["n_separated"],
            "n_pairs": res["n_pairs"]}


def logged_sha(event: str, key: str) -> str | None:
    p = rp(K.ATTEMPTS)
    if not p.exists():
        return None
    ev = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    ev = [r for r in ev if r.get("event") == event]
    return ev[-1].get(key) if ev else None


def run_g0_once(workers: int = 1) -> dict:
    doc, h = declared()
    gp = rp(K.GSEP_OUT)
    if not gp.exists() or sha256(gp) != logged_sha("gsep_run", "gsep_sha256"):
        raise SystemExit("REFUSED: gsep.json missing or not the file G-sep logged")
    if not json.loads(gp.read_text()).get("pass"):
        raise SystemExit("REFUSED: G-sep failed; G0 is not run (closure)")
    out = rp(K.G0_OUT)
    if out.exists():
        raise SystemExit(f"REFUSED: {K.G0_OUT} exists; G0 runs once")
    seed = int(doc["seeds"]["declared"])
    cut = doc["engineering"]["cut30_cm"]
    log_attempt({"event": "g0_start", "declaration_sha256": h, "workers": workers})
    res = G.run_g0(screen_stations(), list(K.SCREEN), cut, seed=seed, workers=workers)
    res.update({"declaration": K.DECL_ID, "declaration_sha256": h,
                "gsep_sha256": sha256(gp), "written": MD.utc()})
    with out.open("x", encoding="utf-8") as fh:
        fh.write(json.dumps(res, indent=1, sort_keys=True, default=float) + "\n")
    s = sha256(out)
    log_attempt({"event": "g0_run", "g0_sha256": s, "pass": res["pass"]})
    return {"pass": res["pass"], "sha256": s, "checks": res["checks"],
            "summary": {k: v for k, v in res["summary"].items() if k != "readings_by_truth"}}


def closure_line(gate: str, rec: dict | None = None) -> str:
    """The honest closure text when G-sep or G0 fails (no GNSS look is taken)."""
    why = ""
    if rec is not None and gate == "G0":
        s = rec.get("summary", {})
        why = (f" (true member retained min {s.get('true_retained_min')}, far members "
               f"excluded pooled {s.get('far_excluded_pooled')} / worst "
               f"{s.get('far_excluded_worst')}, P(all excluded) max "
               f"{s.get('p_all_excluded_max')}; checks {rec.get('checks')})")
    elif rec is not None and gate == "G-sep":
        why = (f" ({rec.get('n_separated')} of {rec.get('n_pairs')} far-cluster pairs "
               "separated)")
    return (f"{K.ID}: {gate} failed{why}; by the declared rule no GNSS amplitude was read and "
            "the test is not registered. Seasonal GNSS cannot narrow the column family under "
            "the declared nuisances; E24 keeps its full range, C27 stays as it is, and no "
            "direction is claimed. Model-only, synthetic; not evidence about the aquifer.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("declare")
    d.add_argument("--seed", type=int, required=True)
    d.add_argument("--print", action="store_true", help="print the declaration JSON")
    sub.add_parser("gsep")
    g = sub.add_parser("g0")
    g.add_argument("--workers", type=int, default=1)
    a = ap.parse_args(argv)
    if a.cmd == "declare":
        doc = draft_declaration(a.seed)
        if a.print:
            print(json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=False, default=float))
        errs = validate_declaration(doc)
        print(json.dumps({"id": doc["id"], "errors": errs or "valid", "self_sha256":
                          sha256(Path(__file__).resolve())}, indent=1))
        return 1 if errs else 0
    if a.cmd == "gsep":
        r = run_gsep_once()
        print(json.dumps(r, default=float))
        if not r["pass"]:
            print(closure_line("G-sep", r))
        return 0 if r["pass"] else 4
    r = run_g0_once(a.workers)
    print(json.dumps(r, default=float))
    if not r["pass"]:
        print(closure_line("G0", r))
    return 0 if r["pass"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
