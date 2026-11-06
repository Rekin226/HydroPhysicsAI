"""Draft declaration of FW.rider.v1 (harness.prereg3 schema, kind 'declaration'), built in code
because dataprep_staging/ ignores *.json. Nothing here writes to the registry: the lead reviews
the draft and registers it with prereg3.write_prereg after this code is on main
(R7.commit_first). No statistic is computed here; the only numbers read are sha256s, the FW v2
and FV v2 registry rows, MANIFEST metadata and the null-world tolerance, which is an analytic
function of E02's registered predictions, E02c's pre-origin climatology and FV v1's noise
export (osse_fv.null_tolerance; no world is drawn).

  prereg_rider.py [--quiet]     prints the draft JSON, its sha256, validate_prereg and readiness
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_rider as R  # noqa: E402  (puts the FV and staging directories on sys.path)

O = R.O  # noqa: E741  (osse_fv, frozen)
V2 = R.V2
FY = R.FY
PENDING = "PENDING"
MISSING = "MISSING"
RULINGS = "results/dataprep/rulings/user_rulings_2026-11-03.json"
RULINGS_SHA = "ec610b692835d955141e9261e6d6046ee1984f39891bcf1c2e77591aad1c69eb"
RULING_ID = "R8.7"
FV2_DECL = f"results/dataprep/preregistration/{R.FV2_ID}.json"
FV2_FORCING_REPORT = V2.REPORT_FORCING
#: dataprep products behind the rider's inputs, the use each is put to, and the route
PRODUCT_USES = (
    {"product": "results/dataprep/twin_rain/preorigin/v2", "use": "design_time_evidence",
     "via": "the rain FW.weather_replay.v2's RD forwards replayed (forcing_frozen there, its "
            "registered use); the rider reads only those model forwards, never the rain"},
    {"product": "results/dataprep/heads/L1_clean/v3", "use": "design_time_evidence",
     "via": "FV v1's pre-origin export noise_ar1.csv (AR(1) sd and lag-1 of free-run hindcast "
            "residuals, 2012-22) and obs_mask.csv (availability only), as FV v2 registered it"},
    {"product": "results/dataprep/heads/L1_clean/v3", "use": "baseline",
     "via": "E02c's registered pre-origin climatology and climatology-plus-trend "
            "(baselines_e02c.csv, months up to 2022-12)"},
    {"product": "results/dataprep/pump_electricity/preorigin/v3", "use": "design_time_evidence",
     "via": "FV v1's anomaly_pool.csv, loaded by osse_fv.load_context and not used by the rider"},
)
NO_MANIFEST = (
    {"input": f"{R.FW_OUTD}/C.npz, RD2012..RD2022.npz and their .fw.npz captures",
     "why": "model output of the registered model-only declaration FW.weather_replay.v2 (its "
            "registration hashed the forward inputs, including the vendor pumping L0 cache, "
            "which has no MANIFEST); read here as truth worlds, model-only"},
    {"input": f"{R.S.E02_DIR}/predictions.csv, protocol.json and manifest.json",
     "why": "E02's registered predictions (registered model output, manifest sha d7286551); "
            "the frozen forecast ZR and its interval, read unchanged"},
    {"input": f"{O.E02C_BASE_DIR}/wells_158.csv",
     "why": "E02c's registered well table (cell, layer, county, 5 km block of each of the 158 "
            "wells; no head value)"},
)


def sha_text(obj) -> str:
    return hashlib.sha256((json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=1)
                           + "\n").encode()).hexdigest()


def _sha(p) -> str:
    q = O.rp(p)
    return O.sha256(q) if q.is_file() else MISSING


def _json(p) -> dict | None:
    q = O.rp(p)
    return json.loads(q.read_text(encoding="utf-8")) if q.is_file() else None


def manifest_checks() -> list[dict]:
    """Every input's MANIFEST allowed_uses / forbidden_uses (metadata only), and the inputs
    that are model output or registrations with no MANIFEST."""
    out = []
    for u in PRODUCT_USES:
        m = _json(f"{u['product']}/MANIFEST.json")
        rec = {**u, "manifest": f"{u['product']}/MANIFEST.json",
               "manifest_sha256": _sha(f"{u['product']}/MANIFEST.json")}
        if m is None:
            rec["allowed"] = PENDING
        else:
            rec.update(allowed=u["use"] in (m.get("allowed_uses") or [])
                       and u["use"] not in (m.get("forbidden_uses") or []),
                       allowed_uses=m.get("allowed_uses"), forbidden_uses=m.get("forbidden_uses"))
        out.append(rec)
    out += [{**x, "manifest": None, "allowed": "no MANIFEST (see why)"} for x in NO_MANIFEST]
    return out


def null_block(P: R.RiderPaths) -> dict:
    """FV v1's null tolerance recomputed from the same inputs (analytic; no world drawn)."""
    try:
        ctx = O.load_context(P.v1)
        t = O.null_tolerance(ctx["zr"] - ctx["cl"], ctx["sd"], ctx["phi"], R.K_NULL)
    except (OSError, KeyError, ValueError, O.EngineeringFailure) as err:
        return {"tolerance": None, "status": PENDING, "why": f"{type(err).__name__}: {err}"}
    return {"tolerance": t["tolerance"], "status": "ok",
            "identical_to_fv_v1": abs(t["tolerance"] - V2.V1_NULL_TOLERANCE) <= 1e-15,
            "fv_v1_tolerance": V2.V1_NULL_TOLERANCE, "formula": t["formula"],
            "rule": f"null world obs = CL + e, {R.K_NULL} draws on stream {R.STREAM_NULL}: "
                    "E02c level false-pass rate <= tolerance, else not evaluable (a possible "
                    "bug)"}


def index_record(prereg_id: str, prereg_dir=None) -> dict | None:
    return R.index_record(prereg_id, prereg_dir)


def _reg(prereg_id: str, prereg_dir=None) -> dict | str:
    r = index_record(prereg_id, prereg_dir)
    return {"id": prereg_id, "sha256": r["sha256"], "created": r.get("created")} if r else PENDING


def ruling(rid: str = RULING_ID) -> dict | None:
    d = _json(RULINGS)
    if not d:
        return None
    return next((r for r in d.get("rulings", []) if r.get("id") == rid), None)


def _na(reason: str) -> dict:
    return {"not_applicable": reason}


BUILDER = (
    "The FW.rider.v1 builder (round 8, 2026-11-06) read code and registrations only: "
    "forcing_year.py, osse_fv.py, osse_fv2.py, score_e02c.py, prereg_fv2.py, the FW v2 runner "
    "and declaration builders; the registered declarations FW.weather_replay.v2, "
    "FV.decidability.v2_postfailure, E02c.prospective_2026-11.v1, FV.stat.v2 and "
    "E02z.level_years23.v1 (their recipe, frozen_code, inputs_frozen, readings and page "
    "mappings). Of FW v2's forward files it read the sha256 and the metadata fields "
    "member_labels, n_members, origin, gain, hindcast_gain, scenario_names, the length and ends "
    "of dates, forward_options (update_method, column_heads, restart_taper_km, apex_hold, "
    "column_hpc0_fast_days) and the heads_mean array header (shape only), and the capture "
    "records' projection_forcing and replay (years, pumping, fields); no head, subsidence or "
    "capture value. Of FW v2's report it read the top-level keys and 'verdict'; of FV v2's "
    "reading-2 report (FW v1's pumping-only replays, published) its status, readings and "
    "engineering block. It meant to run the rider only on synthetic worlds (run_rider.py "
    "smoke, SMOKE_SEED); see accidental_compute for the one exception, whose result nobody "
    "saw.")
ACCIDENTAL_COMPUTE = (
    "2026-11-06T06:05Z, during the builder's own first test run: the smoke guard of "
    "run_rider.score compared resolved paths only, and in the build worktree results/ entries "
    "are symlinks into another checkout, so one test assertion meant to show that a smoke on "
    "the real paths is refused (score(RiderPaths(), seed=SMOKE_SEED, smoke=True)) ran on the "
    "REAL inputs instead. It wrote the replay index (forcing_year.build_index, metadata and "
    "sha256 only) to results/dataprep/reports/FW.rider.v1/replay_index.json (sha256 "
    "c8705388917735be9d004d210e096570294a7639abf3b25055baf1548ff1972c; deleted by the builder "
    "afterwards), read arm C's and the RD arms' heads at the 158 wells, passed gate E and gate N "
    "(500 null worlds) and scored the 11 x 250 worlds on SMOKE_SEED, then returned the report "
    "to the test, which discarded it. The test's output shows only 'gate E passed "
    "(engineering)' and 'gate N passed' and its failure message ('DID NOT RAISE'); no P, pass "
    "rate, RMSE or other statistic was printed, written, logged or seen by anyone. The "
    "registered seed's noise draws were not made. The guard now refuses any path lexically or "
    "resolved under results/ or with a 'results' directory in it, and the tests make every "
    "real read raise. For the lead: log it as an accidental, unseen builder computation before "
    "registering, and decide whether the declared seed stands.")
AFTER_THE_FACT_KNOWN = (
    "Known to the designer before this declaration, all published, used as disclosure only "
    "(U7; they chose no threshold, seed, world or band): FV v2 reading 1's level P 0.964 with "
    "rain at climatology; FV v2 reading 2 on FW v1's pumping-only replays (Yunlin "
    "'not_shown_to_omit', Changhua 'not_represented'); the 2023-26 record (frozen run 2.048 m "
    "against climatology 1.667 m at the 158 wells, so an E02c level fail in 2028 is the "
    "expected outcome); FW v2's scored subsidence verdict (mostly elastic).")
U7 = (
    "No observation after 2022-12 enters selection, calibration or design. The truth worlds "
    "are model forwards: the page parameter set (calibrated on 2012-22) replaying the recorded "
    "2012-22 rain and detrended pumping in the 11 cyclic orders (FW.weather_replay.v2). The "
    "noise inputs (FV v1's export) and E02c's CL/CLT come from pre-origin months only. K, the "
    "seed, the streams, the rule, the bands and their wording were fixed in code before any "
    "RD world value was read by anyone for this rider.")
MODEL_ONLY = (
    "Model-only: the rider reads replay worlds (FW v2's RD arms and arm C) and the frozen "
    "forecast (E02's registered predictions), plus E02c's pre-origin baselines and FV v1's "
    "pre-origin noise export; it never reads an observed head (no window observation, no "
    "heads product, no well record). Its result is a statement about E02c's rule in the "
    "model's own weather, never evidence about the aquifer.")


def draft_prereg(prereg_dir=None, P: R.RiderPaths | None = None) -> dict:
    """FW.rider.v1's declaration (not written)."""
    P = P or R.RiderPaths()
    frozen = {p: _sha(p) for p in R.FROZEN}
    files = R.world_shas(P)
    rl = ruling()
    worlds_in = [v for r in R.world_files(P).values() for v in r.values()]
    inputs = sorted({*R.FROZEN, R.PREREG, *worlds_in, R.FW_REPORT, R.FW_DECL, FV2_DECL,
                     O.E02_PRED, O.E02_PROTOCOL, R.E02_MANIFEST,
                     f"{O.E02C_BASE_DIR}/baselines_e02c.csv", f"{O.E02C_BASE_DIR}/wells_158.csv",
                     *[f"{O.INPUTS_DIR}/{n}" for n in R.FV1_INPUTS],
                     *{f"{u['product']}/MANIFEST.json" for u in PRODUCT_USES}, RULINGS})
    doc = {
        "id": R.ID, "wp": R.WP, "kind": "declaration",
        "title": "Rain-year rider: P(E02c level pass) when the truth worlds are FW.weather_"
                 "replay.v2's 11 RD replays (rain and detrended pumping), FV v2 reading 2's "
                 "world construction scored by E02c's level rule; one-sided caveat only "
                 "(model-only)",
        "label": "model-only rider; one-sided caveat; scored once; no follow-up",
        "estimand": {
            "P": "equal-weight mean over the 11 RD sequences of P(E02c's strict level rule "
                 "passes | truth = ZR + F_s + e), F_s = osse_fv.departure(h_RD_s, h_C, mask) "
                 "on 2026-11..2027-10 at E02c's 158 wells, e FV v1's per-well AR(1) noise",
            "rule_scored": "E02c's T-level rule as registered (score_e02c.t_level through "
                           "osse_fv.world_stats): the frozen forecast beats CL and CLT on "
                           "equal-well RMSE with each 5 km block-bootstrap 90 % interval below "
                           "0; pooled over both counties (by-county RMSEs report-only)",
            "interpretation": "a perfect-model-unknown-weather pass probability of the level "
                              "rule, in the model's own recorded weather years"},
        "model_only": {"statement": MODEL_ONLY, "observations_read": "none",
                       "asserted_by": f"{R.TESTS}::test_the_rider_is_model_only"},
        "worlds": {
            "source_declaration": R.FW_ID, "registration": _reg(R.FW_ID, prereg_dir),
            "report": {"path": R.FW_REPORT, "sha256": _sha(R.FW_REPORT)},
            "arms": list(R.RD_ARMS), "reference": R.CLIM_ARM,
            "members": {"RD": list(R.RD_MEMBERS), "C": "the page ensemble (18 sets x 2 "
                                                       "initial fields)"},
            "replay": "RD(2012 + s) replays the cyclic order of 2012..2022 from 2012 + s: rain "
                      "and pumping as anomalies about FW v2's declared 2012-22 trend "
                      "(--replay-pumping detrended --replay-fields energy+weather, "
                      "--ic-members 0, the page theta alone)",
            "files": files,
            "formula": "obs = ZR + F_s + e; F_s = departure(h_RD_s, h_C, mask) (window minus "
                       "the mean over each well's observed 2012-22 months, FV v1's function); "
                       "equal to reading 2's h_s - h_clim when the hindcasts are identical",
            "why_departure": "no single-member climatology arm exists (C is the 36-run page "
                             "ensemble), so the datum-anchored departure of reading 1 "
                             "(D_T against ZR0) is used: F_s holds the recorded year plus the "
                             "page set's own departure from the ensemble (common to every s, "
                             "reported, never removed)"},
        "reused_code": {
            "from": R.FV2_ID, "registration": _reg(R.FV2_ID, prereg_dir),
            "binding": "imported unchanged; each file pinned to FV v2's registered frozen_code "
                       "sha256 (run_rider.REUSED); a changed file refuses the run",
            "pins": dict(R.REUSED), "current_files_match": not R.reused_code_problems(),
            "functions": ["forcing_year.build_index", "forcing_year.capture_record",
                          "forcing_year.check_capture", "forcing_year.seq_years",
                          "forcing_year._heads", "forcing_year._opts",
                          "forcing_year.summarise_rc", "osse_fv.load_context",
                          "osse_fv.departure", "osse_fv.simulate_ar1", "osse_fv.world_stats",
                          "osse_fv.null_tolerance", "osse_fv._rate", "score_e02c.t_level"]},
        "design": {"seed": R.SEED, "seed_note": "fixed in code before registration; the smoke "
                                                f"seed {R.SMOKE_SEED} is refused for a run",
                   "K_WORLD": R.K_WORLD, "K_NULL": R.K_NULL,
                   "streams": {"world": R.STREAM_WORLD, "null": R.STREAM_NULL},
                   "P_weak": R.P_WEAK, "hindcast_tol_m": R.HINDCAST_TOL_M,
                   "window": "2026-11..2027-10 (E02c's window, axis 2012-01)",
                   "sequences": R.N_SEQ, "bootstrap": "forcing_year.summarise_rc's own stream "
                                                      "(FV v1's seed, 2000 resamples of the 11 "
                                                      "sequences), unchanged"},
        "readings": {"rule": R.READING_RULE, "threshold": R.P_WEAK,
                     "bands": {"P <= 0.5": "weak_in_replayed_years",
                               "P > 0.5": "not_shown_weak_in_replayed_years",
                               "MC half-width > 0.07, gate N failed or replay worlds unusable":
                                   "not_evaluable"},
                     "wording": R.WORDING,
                     "one_sided": "no 'a fail is informative' line and no 'pass expected' "
                                  "line; P is never the chance of a 2028 pass"},
        "caveat_rule": {
            "role": "a caveat line slot in the claims ledger only (E21 notes, quoted beside "
                    "C30-scen's and E21's pre-declared post-2028 level wording); shown nowhere "
                    "before report.json exists, and never on the page",
            "weak_in_replayed_years": "E21 and C30-scen carry the caveat 'a 2028 level fail "
                                      "would not by itself refute the model's average-weather "
                                      "level'; no status changes",
            "not_shown_weak_in_replayed_years": "E21 carries 'the replayed rain years do not "
                                                "show the level test weak'; no status changes",
            "not_evaluable": "the declared not-evaluable sentence in the E21 notes; nothing "
                             "else",
            "never": ["no S, no D and no status change from the rider",
                      "no page line (ruling R8.7 approves page, ledger and README; this "
                      "caveat is ledger-only)",
                      "never cited as support for the physics level or as the chance of a "
                      "2028 pass", "E02c's rule, thresholds and verdict are unchanged"]},
        "gates": {
            "E": "registration intact; frozen code and inputs unchanged; FW v2's report "
                 "present; the replay index (forcing_year.build_index) with rain 'replayed' "
                 "and pumping 'detrended_anomaly'; npz and capture sha256 as declared; axis, "
                 "origin, members, forward options and gain; the 11 RD hindcasts identical to "
                 f"each other within {R.HINDCAST_TOL_M} m at the wells (FV v2 used 0.0 between "
                 "replays of one ensemble); no two sequences identical; finite heads. A "
                 "failure in the content of FW's forwards maps to 'replay worlds unusable' "
                 "(not evaluable, final); any other failure is a free engineering retry before "
                 "any statistic",
            "N": null_block(P),
            "MC": {"rule": f"1.645 x Monte Carlo SE of P <= {R.MC_HALF_WIDTH}",
                   "note": V2.mc_note(R.N_SEQ, R.K_WORLD)}},
        "attempt_policy": {"scored_runs": "scored once (report.json, mode 'x')",
                           "engineering_retries": "free while gate E fails before any "
                                                  "statistic, each logged",
                           "attempts_log": f"{R.REPORT_DIR}/{R.ATTEMPTS_NAME}",
                           "exploratory_followups_max": 0, "thresholds_frozen": True,
                           "report": "every attempt"},
        "u7": U7,
        "manifest_checks": manifest_checks(),
        "rulings": {f"2026-11-03 twelfth set, {RULING_ID}": (rl or {}).get("decision", PENDING),
                    "file": {"path": RULINGS, "sha256": _sha(RULINGS)},
                    "note": "R8.7 approves the forecast-of-record package the rider rides "
                            "with; the rider is a model-only declaration registered before "
                            "its compute (plan gate), and it can change no status"},
        "disclosures": {
            "builder": BUILDER, "accidental_compute": ACCIDENTAL_COMPUTE,
            "after_the_fact_known": AFTER_THE_FACT_KNOWN,
            "optimism": "noise independent between wells; every well observed every window "
                        "month; the model's damped amplitude (about 0.73) in the replayed "
                        "forcing; one parameter set as the truth family",
            "pessimism": "the AR(1) noise is fitted to free-run hindcast residuals that "
                         "already hold structural misfit; F_s adds the set's departure from "
                         "the ensemble on top",
            "rain_cache": "the replayed rain is the vendor cache, which zero-fills gauge rf11 "
                          "for all of 2021-22 and rf15 in 2015 H2 (FW v2's disclosure)",
            "registration_paths": "write_prereg hashes inputs through common.paths."
                                  "rel_to_repo; registering from a linked worktree whose "
                                  "results/ is a symlink can record absolute paths (FW v1's "
                                  "INDEX rows); register from the checkout that holds "
                                  "results/ as a real directory",
            "tests": {"path": R.TESTS, "sha256_at_declaration": _sha(R.TESTS),
                      "binding": "disclosure only, not an input"}},
        "primary_model": {"forecast": "ZR, E02's registered predictions (forward cf2c7154)",
                          "truths": f"{R.FW_ID} RD arms (page parameter set alone)"},
        "thresholds": {"P_weak": R.P_WEAK, "K_WORLD": R.K_WORLD, "K_NULL": R.K_NULL,
                       "mc_half_width_90": R.MC_HALF_WIDTH, "hindcast_tol_m": R.HINDCAST_TOL_M,
                       "null_tolerance": (null_block(P) or {}).get("tolerance")},
        "interpretation": "A model-only operating characteristic of E02c's level rule when "
                          "the weather after the origin is a recorded year rather than its "
                          "average; one-sided; it can only add a caveat to how a 2028 level "
                          "outcome is read. Nothing here is evidence about the aquifer.",
        "written_by": R.PREREG,
        "zones": _na("no fit"), "burn_in": _na("no fit"),
        "folds": _na("synthetic worlds; no folds"),
        "gate_scores": _na("one reading by one threshold; no weighted score"),
        "tests": _na("model-only declaration: synthetic truth worlds, no observation is scored "
                     "and no fresh unit is consumed"),
        "fresh_units": _na("no observation is scored"),
        "priors": "With recorded weather in the truth the frozen average-weather forecast "
                  "carries the year's departure as error, so P is expected below FV v2's 0.96 "
                  "at climatology; how far below is not known (no RD world was read). No "
                  "direction is claimed.",
        "frozen_code": frozen, "inputs": inputs,
    }
    return json.loads(json.dumps(doc, default=O._jsonable))


def _git_problems(paths) -> list[str]:
    out = []
    for p in paths:
        try:
            tracked = subprocess.run(["git", "-C", str(R.REPO), "ls-files", "--error-unmatch",
                                      p], capture_output=True, text=True).returncode == 0
            clean = subprocess.run(["git", "-C", str(R.REPO), "diff", "--quiet", "HEAD", "--",
                                    p], capture_output=True).returncode == 0
        except OSError as err:
            return [f"git: {err}"]
        if not tracked or not clean:
            out.append(f"{p} is not committed as-is (R7.commit_first)")
    return out


def readiness(d: dict, prereg_dir=None, now=None) -> list[str]:
    """Everything that must hold before the lead registers the draft (empty = ready)."""
    now = now or dt.datetime.now(dt.timezone.utc)
    out = []
    if now >= dt.datetime(2028, 1, 15, tzinfo=dt.timezone.utc):
        out.append("E02c's scoring session (2028-01-15) has started: too late for a caveat "
                   "registered before the outcome")
    for k in ("worlds", "reused_code"):
        if d[k]["registration"] == PENDING:
            out.append(f"{k}: source declaration not in the registry")
    if d["worlds"]["report"]["sha256"] == MISSING:
        out.append(f"{R.FW_REPORT} missing (FW v2's forwards stay sealed until it exists)")
    bad = [a for a, r in d["worlds"]["files"].items() if MISSING in r.values()]
    if bad:
        out.append(f"world files missing: {bad}")
    if not d["reused_code"]["current_files_match"]:
        out.append(f"reused code differs from FV v2's frozen_code: {R.reused_code_problems()}")
    n = d["gates"]["N"]
    if n.get("status") != "ok" or not n.get("identical_to_fv_v1"):
        out.append("the null tolerance is not recomputed identically to FV v1's")
    for m in d["manifest_checks"]:
        if m.get("manifest") and m.get("allowed") is not True:
            out.append(f"{m['use']} is not an allowed use of {m['product']} (MANIFEST)")
    if d["rulings"][f"2026-11-03 twelfth set, {RULING_ID}"] == PENDING or \
            d["rulings"]["file"]["sha256"] != RULINGS_SHA:
        out.append(f"ruling {RULING_ID} not found in {RULINGS} at sha {RULINGS_SHA[:8]}")
    if O.rp(R.REPORT).exists():
        out.append(f"{R.REPORT} exists before the declaration")
    out += [f"frozen code {p} missing" for p, h in d["frozen_code"].items() if h == MISSING]
    out += [f"input {p} missing" for p in d["inputs"] if not O.rp(p).exists()]
    out += _git_problems([*R.FROZEN, R.PREREG])
    if index_record(R.ID, prereg_dir):
        out.append(f"{R.ID} is already registered")
    return out


def main(argv=None, prereg_dir=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    d = draft_prereg(prereg_dir)
    from harness import prereg3

    errs = prereg3.validate_prereg(d, prereg_dir=prereg_dir or prereg3.PREREG_DIR)
    if not a.quiet:
        print(json.dumps(d, indent=1, ensure_ascii=False, sort_keys=True))
    print("draft sha256:", sha_text(d))
    print("validate_prereg:", errs or "ok")
    print("readiness:", readiness(d, prereg_dir) or "ready to register")
    print("this script sha256:", O.sha256(Path(__file__)))
    print("not written: the lead registers with prereg3.write_prereg after review")
    return 1 if errs else 0


if __name__ == "__main__":
    raise SystemExit(main())
