"""Draft declaration of E02.decidability.v1 (harness.prereg3 schema, kind 'declaration'), built
in code because dataprep_staging/ ignores *.json. Nothing here writes to the registry: the lead
reviews the draft and registers it with prereg3.write_prereg after this code is on main
(R7.commit_first). No statistic is computed here; the only things read are sha256s, FV v2's
attempts log (hashes), the source declarations' frozen_code, MANIFEST metadata, the rulings
file and the rule fields of E02's protocol. Every path comes from run_e02d1.DecidPaths, so the
tests build the draft from synthetic files only.

  prereg_e02d1.py [--quiet]     prints the draft JSON, its sha256, validate_prereg and readiness
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

import run_e02d1 as R  # noqa: E402  (puts the FV and staging directories on sys.path)

O = R.O  # noqa: E741  (osse_fv, frozen)
V2 = R.V2
PENDING = "PENDING"
MISSING = "MISSING"
#: the latest moment the reading may be registered: E02's window closes 2027-10-31
TOO_LATE = dt.datetime(2027, 11, 1, tzinfo=dt.timezone.utc)
#: dataprep products behind the reading's inputs (relative to DecidPaths.dataprep), the use
#: each is put to, and the route
PRODUCT_USES = (
    {"product": "heads/L1_clean/v3", "use": "design_time_evidence",
     "via": "FV v1's pre-origin export noise_ar1.csv (AR(1) sd and lag-1 of free-run hindcast "
            "residuals, 2012-22) and obs_mask.csv (availability only), as FV v2 registered it"},
    {"product": "heads/L1_clean/v3", "use": "baseline",
     "via": "E02's own baseline_m (the month-of-year climatology of each well's fitted months "
            "before the origin, inside E02's registered predictions.csv) and E02c's pre-origin "
            "CL/CLT (baselines_e02c.csv, loaded by osse_fv.load_context, not used by E02's "
            "rule)"},
    {"product": "pump_electricity/preorigin/v3", "use": "design_time_evidence",
     "via": "FV v1's anomaly_pool.csv: osse_fv2.check_arms re-draws FV v2's registered "
            "multiplier draws from it to verify the arm files; no new draw"},
)

THRESHOLD_JUSTIFICATION = [
    "Section 21 grades D-measured (0.7) when 'a registered synthetic reading says the test can "
    "decide' and D-weak (0.5) when the test is weak by design. A test can decide only if a "
    "right model usually passes AND a model with no skill usually fails, so the reading needs "
    "both P (power) and P0 (the null pass rate); FV v2 gave E02's rule a P (0.736, report-only) "
    "and no null reading, which is why E02 is D unknown.",
    "R8.17b (fifteenth set, user_rulings_2026-11-07b.json sha256 7602157b): FV.stat.v2 read E04 "
    "'not_shown_weak' (P 0.930) but its no-persistence null passed 0.475 of the time, and the "
    "user graded E04 and C30-fc D unknown, not measured: a null passing about half the time "
    "leaves the test unable to decide. The thresholds reproduce that ruling: P 0.930 with P0 "
    "0.475 reads 'unchanged' (D unknown).",
    "'measured' needs P >= 0.8 AND P0 <= 0.2: power 0.8 and a no-skill pass rate of at most "
    "0.2, the operating point of FE.encompass_2026-12.v1's G0 (at least 0.80 correct readings "
    "in its no-information worlds) and close to X5.n2_reading.v1's false-pass bar (< 0.25); "
    "together a pass is at least 4 times likelier under a right model than under a no-skill "
    "one (0.8 / 0.2).",
    "'weak' takes the one-sided P <= 0.5 rule of FV.decidability.v1/v2, FV.stat.v2 and "
    "FW.rider.v1 (strict here, P < 0.5, so P exactly 0.5 reads 'unchanged', the side of no "
    "grade change) and adds its mirror for the null, P0 > 0.5: a model with no skill passing "
    "more often than not means a 2028 pass cannot show skill (R8.17b's logic at its limit).",
    "Between the bands the reading is 'unchanged' (D unknown, 0.6): neither shown decisive nor "
    "shown weak. A 'not_evaluable' run maps the same way.",
    "Point estimates decide, as in every precedent; the Monte Carlo SE of P and P0 and each "
    "one's distance to its thresholds in SE units are reported beside the outcome.",
    "Stricter than E02z's and FV.ext's 'not shown weak' (P > 0.5): E30 and E22 are not re-read; "
    "their own registered readings stand.",
]
PRIORS = (
    "Expected outcome: 'unchanged' (E02 stays D unknown). P: the right-kind worlds are FV v2's "
    "1000 full forwards with fresh noise; FV v2's report-only E02 rate on the same forwards was "
    "0.736 (by truth Q0 0.468, R0 0.492, R1c 0.992, X5R1 0.992), published in the claims ledger "
    "(E02 notes) before this declaration, so P is expected near 0.74, about 4-5 Monte Carlo SE "
    "below 0.8: 'measured' is not expected. P0: in the equal-skill null the ratio half of the "
    "rule is a fair coin (the noise is symmetric and the truth mean is equidistant from the "
    "forecast and the baseline), so P0 is at most about 0.5 and lower by the coverage half; "
    "'weak' through P0 > 0.5 is not expected, nor through P < 0.5. The thresholds were set by "
    "the lead before this build with P 0.736 known; they make the favourable outcome harder, "
    "not easier. No direction is claimed for P0.")
NULL_DEFINITION = {
    "formula": "obs0(w, t) = (ZR(w, t) + B(w, t)) / 2 + e_k(w, t), B = E02's baseline_m, e_k "
               "FV v1's per-well AR(1) noise on stream 72; K_NULL worlds, one cell",
    "property": "the truth mean is equidistant from the frozen forecast and E02's baseline at "
                "every well-month, so MSE(ZR) - MSE(B) = -2 mean_t[(ZR - B) e] per well: zero "
                "in expectation, symmetric about zero; the frozen forecast has exactly no skill "
                "over E02's baseline in expectation (the ratio half passes with probability 0.5; "
                "ties pass)",
    "as_fv_stat_v2": "FV.stat.v2's null is the world in which the tested forecast has no skill "
                     "over climatology (rho = 0: the statistical layer reverts to climatology "
                     "and its forecast is about as good as the baseline). E02's forecast is "
                     "frozen and cannot revert, so the zero skill is put in the truth instead: "
                     "same rule, same noise model, a truth in which the model's information "
                     "beyond the baseline is worth nothing",
    "not_fv_v1_null": "FV v1/v2's null world (obs = CL + e, the physics departure absent) "
                      "gives the frozen forecast NEGATIVE skill (it predicts a departure that "
                      "is not there) and is not a no-skill model; FV v2's report_worlds.json "
                      "holds a report-only E02 rate for that world, which the builder did not "
                      "open and which sets nothing here",
}
U7 = (
    "No observation after 2022-12 enters selection, calibration or design. The right-kind "
    "worlds are FV v2's registered full forwards (four calibrated parameter sets, pumping "
    "anomalies drawn from the pre-origin pool, rain at climatology); the noise inputs (FV v1's "
    "export) and E02's baseline come from pre-origin months only. E02's interval family was "
    "chosen after its 2023-26 coverage was seen (E02's U7 label; the window is untouched): the "
    "coverage half of the rule read here tests that selected form. K, the seed, the streams, "
    "the null, the thresholds and the wording were fixed in code before any world of this "
    "reading was drawn. Known before, used as disclosure only: FV v2's report-only E02 rate "
    "0.736 on the same forwards (other noise draws) and its split by truth.")
MODEL_ONLY = (
    "Model-only: the reading reads model forwards (FV v2's truth batches, FV v1's truth "
    "forwards, ZR0, the registered ZR forward and the X5R1/Q0 reproduction forwards), E02's "
    "registered predictions and pre-origin observation-derived statistics (E02's baseline_m, "
    "E02c's CL/CLT, FV v1's observed-month mask and AR(1) sd/lag-1). It never reads an observed "
    "head, in the window or before it (no heads product, no well record). Its result is a "
    "statement about E02's rule in the model's own worlds, never evidence about the aquifer.")
BUILDER = (
    "The E02.decidability.v1 builder (round 9, 2026-11-07) read code (osse_fv.py, osse_fv2.py, "
    "score_e02c.py, stat_v2.py, prereg_v2.py, run_rider.py, prereg_rider.py, harness.prereg3, "
    "hydrophysics/twin/progress.py), the registered declarations FV.decidability.v2_postfailure "
    "and FV.stat.v2 (key list, frozen_code, inputs_frozen, v1_forwards, design, attempt_policy "
    "and inputs list), INDEX.jsonl rows (id, wp, kind, created), E02's protocol.json and "
    "manifest.json, the MANIFESTs of heads/L1_clean/v3 and pump_electricity/preorigin/v3 "
    "(allowed and forbidden uses), the round-9 rulings file, the claims ledger and the progress "
    "map, and listed file names under results/twin_forward and results/twin_runs/fv2. It "
    "printed the header and first two rows of E02's predictions.csv (registered model output: "
    "one well's predicted, baseline and interval values for 2026-11 and 2026-12; no "
    "observation). It did not open FV v2's report_worlds.json, any forward array, FV v1's "
    "noise, mask or pool exports, E02c's baselines or any observation, and it ran this reading "
    "only on synthetic worlds (osse_fv2.build_synthetic; SMOKE_SEED and the tests' seeds); the "
    "registered seed's draws were never made.")

#: the progress-map qualifier this reading feeds (progress_map_v1.json, E02's D at status 'D')
PROGRESS_SPEC = {"measured_by": R.ID, "reading_report": R.REPORT,
                 "reading_outcomes": dict(R.LEDGER_GRADE), "otherwise": "D-unknown"}


def sha_text(obj) -> str:
    return hashlib.sha256((json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=1)
                           + "\n").encode()).hexdigest()


def _sha(p) -> str:
    q = O.rp(p)
    return O.sha256(q) if q.is_file() else MISSING


def _json(p) -> dict | None:
    q = O.rp(p)
    return json.loads(q.read_text(encoding="utf-8")) if q.is_file() else None


def _product(P: R.DecidPaths, prod: str) -> str:
    return str(Path(P.dataprep) / prod) if Path(P.dataprep).is_absolute() else \
        f"{P.dataprep}/{prod}"


def manifest_checks(P: R.DecidPaths) -> list[dict]:
    """Every input's MANIFEST allowed_uses / forbidden_uses (metadata only), and the inputs
    that are model output or registrations with no MANIFEST."""
    out = []
    for u in PRODUCT_USES:
        man = f"{_product(P, u['product'])}/MANIFEST.json"
        m = _json(man)
        rec = {**u, "product": f"results/dataprep/{u['product']}", "manifest": man,
               "manifest_sha256": _sha(man)}
        if m is None:
            rec["allowed"] = PENDING
        else:
            rec.update(allowed=u["use"] in (m.get("allowed_uses") or [])
                       and u["use"] not in (m.get("forbidden_uses") or []),
                       allowed_uses=m.get("allowed_uses"), forbidden_uses=m.get("forbidden_uses"))
        out.append(rec)
    out += [
        {"input": "FV v2's truth batches fv2_truth_<T>_b<b>.npz, arms.json (and attempts.jsonl, "
                  "read for gate E, not pinned)",
         "manifest": None, "allowed": "no MANIFEST (see why)",
         "why": "model output of the registered model-only declaration "
                "FV.decidability.v2_postfailure (its registration hashed its inputs and code); "
                "read here as right-kind truth worlds, model-only"},
        {"input": "FV v1's truth forwards fv_truth_<T>.npz, ZR0 (fv_zr0.npz), the registered ZR "
                  "forward (datum_gate_innov.npz, sha cf2c7154) and the X5R1/Q0 reproduction "
                  "forwards", "manifest": None, "allowed": "no MANIFEST (see why)",
         "why": "model output (FV.decidability.v1's registered forwards and the page model), "
                "read for FV v2's hindcast-identity, ZR0 and reproduction checks and D_T"},
        {"input": f"{Path(P.v1.pred).parent}/predictions.csv, protocol.json and manifest.json",
         "manifest": None, "allowed": "no MANIFEST (see why)",
         "why": "E02's registered predictions (registered model output, manifest sha d7286551); "
                "the frozen forecast ZR, its interval and its baseline_m, read unchanged"},
        {"input": P.v1.wells, "manifest": None, "allowed": "no MANIFEST (see why)",
         "why": "E02c's registered well table (cell, layer, county, 5 km block of each of the "
                "158 wells; no head value)"}]
    return out


def _reg(prereg_id: str, prereg_dir=None) -> dict | str:
    r = R.index_record(prereg_id, prereg_dir)
    return {"id": prereg_id, "sha256": r["sha256"], "created": r.get("created")} if r else PENDING


def ruling(P: R.DecidPaths, rid: str = R.RULING_ID) -> dict | None:
    d = _json(P.rulings)
    if not d:
        return None
    return next((r for r in d.get("rulings", []) if r.get("id") == rid), None)


def pins_match(P: R.DecidPaths) -> dict:
    """Each reused file's pin against its source declaration's registered frozen_code."""
    decls = {R.FV2_ID: _json(P.fv2_decl) or {}, R.FVS2_ID: _json(P.fvs2_decl) or {}}
    return {p: (decls[R.REUSED_FROM[p]].get("frozen_code") or {}).get(p) == h
            for p, h in R.REUSED.items()}


def attempts_problems(P: R.DecidPaths) -> list[str]:
    npz = {(t, b): O.rp(p) for t, ps in P.fv2.batches.items() for b, p in enumerate(ps)}
    try:
        V2.check_attempts(O.rp(P.fv2.attempts), npz)
    except (OSError, ValueError, KeyError, O.EngineeringFailure) as err:
        return [f"FV v2 attempts: {err}"]
    return []


def _na(reason: str) -> dict:
    return {"not_applicable": reason}


def draft_prereg(prereg_dir=None, P: R.DecidPaths | None = None) -> dict:
    """E02.decidability.v1's declaration (not written)."""
    P = P or R.DecidPaths()
    frozen = {p: _sha(p) for p in R.FROZEN}
    files = R.world_shas(P)
    rl = ruling(P)
    proto = _json(P.v1.protocol) or {}
    inputs = sorted({*R.FROZEN, R.PREREG, *R.world_files(P).values(), P.v1.pred, P.v1.protocol,
                     P.e02_manifest(), P.v1.baselines, P.v1.wells,
                     *[str(Path(P.v1.inputs_dir) / n) for n in R.FV1_INPUTS],
                     *{f"{_product(P, u['product'])}/MANIFEST.json" for u in PRODUCT_USES},
                     P.fv2_decl, P.fvs2_decl, P.rulings})
    pm = pins_match(P)
    doc = {
        "id": R.ID, "wp": R.WP, "kind": "declaration",
        "title": "Synthetic decidability of E02's own pooled rule (pooled 90 % coverage in "
                 "[0.85, 0.95] and equal-station RMSE ratio to climatology <= 1.0): power P in "
                 "FV v2's four truth families and null pass rate P0 in an equal-skill world; "
                 "three pre-declared outcomes (measured / weak / unchanged), model-only",
        "label": "model-only decidability reading (user ruling R9.4); no fresh units; scored "
                 "once; no follow-up",
        "estimand": {
            "P": "equal-weight mean over FV v2's truths R0, Q0, R1c and X5R1 of P(E02's rule "
                 "passes | obs = ZR + D_T + R_T,k + e_k) on 2026-11..2027-10 at E02's 158 "
                 "wells; R_T,k FV v2's world k (all 250 per truth), e_k FV v1's per-well AR(1) "
                 "noise on this reading's seed",
            "P0": "P(E02's rule passes | obs0 = (ZR + B) / 2 + e_k), K_NULL worlds",
            "rule_scored": "E02's registered rule (twin.validation.evaluate arithmetic, "
                           "osse_fv.g0_e02): pooled central 90 % coverage in [0.85, 0.95] AND "
                           "equal-station RMSE ratio to baseline_m <= 1.0 (ties pass), with "
                           ">= 100 stations of >= 6 window months (always met here: 158 x 12)",
            "interpretation": "the operating characteristic of E02's own rule: how often a "
                              "right-kind model passes it and how often a no-skill one does"},
        "rule": {"protocol": P.v1.protocol, "protocol_sha256": _sha(P.v1.protocol),
                 "expected_sha256": R.E02_PROTOCOL_SHA,
                 "stated": {k: proto.get(k, PENDING) for k in R.E02_RULE},
                 "declared": dict(R.E02_RULE),
                 "check": "run_e02d1.protocol_rule: the protocol must state exactly the declared "
                          "rule, else engineering failure"},
        "model_only": {"statement": MODEL_ONLY,
                       "observations_read": "no observed head; pre-origin observation-derived "
                                            "statistics only (E02's baseline_m, E02c CL/CLT, "
                                            "FV v1 mask and AR(1) sd/phi)",
                       "asserted_by": f"{R.TESTS}::test_the_reading_is_model_only"},
        "worlds": {
            "source_declaration": R.FV2_ID, "registration": _reg(R.FV2_ID, prereg_dir),
            "truths": dict(O.TRUTHS), "truth_order": list(R.TAGS),
            "formula": "obs = ZR + D_T + R_T,k + e_k: D_T = osse_fv.departure(batch baseline, "
                       "ZR0, mask), R_T,k = world k minus its batch baseline in the window, as "
                       "osse_fv2.score_worlds builds them (osse_fv2.load_truth, unchanged)",
            "files": files,
            "binding": "every file's sha256 must equal this declaration at scoring "
                       "(run_e02d1.world_problems), and each batch must be the output of its "
                       "last successful forward attempt in FV v2's attempts.jsonl "
                       "(osse_fv2.check_attempts; the log itself is not pinned, as other FV v2 "
                       "scorers append to it)",
            "attempts_match": not attempts_problems(P),
            "fresh": "the noise is fresh (seed, stream 71); the 1000 forward worlds are FV v2's "
                     "own, already scored by its reading 1 (report-only E02 rate 0.736)"},
        "null": NULL_DEFINITION,
        "reused_code": {
            "from": {R.FV2_ID: _reg(R.FV2_ID, prereg_dir), R.FVS2_ID: _reg(R.FVS2_ID,
                                                                          prereg_dir)},
            "binding": "imported unchanged; each file pinned to its source declaration's "
                       "registered frozen_code sha256 (run_e02d1.REUSED); a changed file "
                       "refuses the run",
            "pins": dict(R.REUSED), "pinned_to": dict(R.REUSED_FROM),
            "current_files_match": not R.reused_code_problems(),
            "pins_match_declarations": all(pm.values()),
            "functions": ["osse_fv.load_context", "osse_fv.simulate_ar1", "osse_fv.g0_e02",
                          "osse_fv.departure", "osse_fv.well_heads", "osse_fv.check_zr0_meta",
                          "osse_fv.evaluate_selfcheck", "osse_fv.world_stats",
                          "osse_fv._rate", "osse_fv2.check_arms", "osse_fv2.v1_reference",
                          "osse_fv2.load_truth", "osse_fv2.check_attempts",
                          "score_e02c.annotate", "twin.validation.evaluate"]},
        "design": {**R.design(),
                   "seed_note": "fixed in code before registration (fresh: not FV v1/v2's "
                                "20261101, not FW.rider's or this reading's smoke seed)",
                   "refused_seeds": {str(k): v for k, v in R.REFUSED_SEEDS.items()},
                   "smoke_seed": R.SMOKE_SEED,
                   "K_note": f"K_TRUTH {R.K_TRUTH} worlds per truth (every FV v2 world, one "
                             f"fresh noise draw each; {len(R.TAGS)} x {R.K_TRUTH} = "
                             f"{len(R.TAGS) * R.K_TRUTH}); K_NULL {R.K_NULL} in one null cell",
                   "mc_se_bounds": {"P_max": (0.25 * len(R.TAGS) / R.K_TRUTH) ** 0.5
                                    / len(R.TAGS),
                                    "P0_max": (0.25 / R.K_NULL) ** 0.5,
                                    "note": "largest possible Monte Carlo SE (every rate at "
                                            "0.5); the report gives the realised SE"},
                   "window": "2026-11..2027-10 (E02's window, axis 2012-01)",
                   "selfcheck": f"{R.N_SELFCHECK_TRUTH} right-kind (R0) worlds and 1 null "
                                f"world on stream {R.STREAM_SELFCHECK}"},
        "readings": {"rule": R.OUTCOME_RULE,
                     "outcomes": {"measured": "P >= 0.8 and P0 <= 0.2",
                                  "weak": "P < 0.5 or P0 > 0.5 (when not measured; the two "
                                          "bands cannot overlap)",
                                  "unchanged": "otherwise",
                                  "not_evaluable": "1.645 SE > 0.07 on P or P0, or a "
                                                   "non-finite estimate (mapped as "
                                                   "unchanged)"},
                     "thresholds": R.design()["thresholds"],
                     "justification": THRESHOLD_JUSTIFICATION,
                     "wording": R.WORDING, "weak_why": R.WEAK_WHY,
                     "one_sided_note": "no 'a fail is informative' line; P is never the chance "
                                       "of a 2028 pass"},
        "ledger_mapping": {**R.LEDGER_MAPPING, "grade_keys": dict(R.LEDGER_GRADE),
                           "progress_map_spec": PROGRESS_SPEC,
                           "page": "none: ledger and docs only; any page line needs a user "
                                   "ruling"},
        "gates": {
            "E": "registration intact (INDEX.jsonl entry, load_prereg at its sha256); frozen "
                 "code, reused pins and inputs unchanged; every world file at its declared "
                 "sha256 and each batch the output of its last successful FV v2 forward "
                 "attempt; FV v2's checks (arms, batch metadata, hindcast identity 0.0 m, X5R1 "
                 "within 0.02 m, ZR0 metadata); the frame order; E02's rule as the protocol "
                 "states it; on the self-check worlds E02's rule here == "
                 "twin.validation.evaluate == FV v2's world_stats. A failure writes nothing "
                 "(free engineering retry before any statistic)",
            "MC": {"rule": f"1.645 x Monte Carlo SE of P and of P0 <= {R.MC_HALF_WIDTH}, else "
                           "not_evaluable",
                   "note": V2.mc_note(len(R.TAGS), R.K_TRUTH)}},
        "attempt_policy": {"scored_runs": "scored once (report.json, mode 'x')",
                           "engineering_retries": "free while gate E fails before any "
                                                  "statistic, each logged",
                           "attempts_log": f"{R.REPORT_DIR}/{R.ATTEMPTS_NAME} (written only "
                                           "after the registration check passes)",
                           "exploratory_followups_max": 0, "thresholds_frozen": True,
                           "report": "every attempt"},
        "u7": U7,
        "manifest_checks": manifest_checks(P),
        "rulings": {f"2026-11-07 sixteenth set, {R.RULING_ID}": (rl or {}).get("decision",
                                                                               PENDING),
                    "file": {"path": P.rulings, "sha256": _sha(P.rulings)},
                    "note": "R9.4 drops the E04/N2 companion registrations and asks for this "
                            "reading of E02's own pooled rule with three pre-declared "
                            "outcomes; no fresh units"},
        "disclosures": {
            "builder": BUILDER,
            "after_the_fact_known": "FV v2 reading 1's report-only E02 rate 0.736 (Q0 0.468, R0 "
                                    "0.492, R1c 0.992, X5R1 0.992) on the same forwards; its "
                                    "Changhua coverage split by truth family; the 2023-26 record "
                                    "(frozen run 2.048 m against climatology 1.667 m at the 158 "
                                    "wells); FV v2's worlds are far less climatological than "
                                    "that record (median CL RMSE 2.21-2.25 m), so P is not the "
                                    "chance of a 2028 pass. Disclosure only (U7)",
            "optimism": "noise independent between wells; no county-common year shock and no "
                        "per-well relative drift (U1.HB.power.v1: correlated noise can fail a "
                        "correct interval); rain at climatology; every well observed every "
                        "window month",
            "pessimism": "the AR(1) noise is fitted to free-run hindcast residuals that already "
                         "hold structural misfit, which D_T adds again",
            "registration_paths": "write_prereg hashes inputs through common.paths."
                                  "rel_to_repo; registering from a linked worktree whose "
                                  "results/ is a symlink can record absolute paths (FV v2's own "
                                  "inputs list does); register from the checkout that holds "
                                  "results/ as a real directory",
            "tests": {"path": R.TESTS, "sha256_at_declaration": _sha(R.TESTS),
                      "binding": "disclosure only, not an input"}},
        "primary_model": {"forecast": "ZR, E02's registered predictions (forward cf2c7154)",
                          "truths": f"{R.FV2_ID} full-forward truth worlds (R0, Q0, R1c, X5R1)"},
        "thresholds": {**R.design()["thresholds"], "K_TRUTH": R.K_TRUTH, "K_NULL": R.K_NULL},
        "interpretation": "A model-only operating characteristic of E02's own pooled rule: its "
                          "power in FV v2's truth families and its pass rate for a model with "
                          "no skill over E02's baseline. It sets only E02's D qualifier in the "
                          "progress rubric; it changes no status and is not evidence about the "
                          "aquifer.",
        "written_by": R.PREREG,
        "zones": _na("no fit"), "burn_in": _na("no fit"),
        "folds": _na("synthetic worlds; no folds"),
        "gate_scores": _na("one outcome by fixed thresholds; no weighted score"),
        "tests": _na("model-only declaration: synthetic truth worlds, no observation is scored "
                     "and no fresh unit is consumed"),
        "fresh_units": _na("no observation is scored"),
        "priors": PRIORS,
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


def readiness(d: dict, P: R.DecidPaths | None = None, prereg_dir=None, now=None) -> list[str]:
    """Everything that must hold before the lead registers the draft (empty = ready)."""
    P = P or R.DecidPaths()
    now = now or dt.datetime.now(dt.timezone.utc)
    out = []
    if now >= TOO_LATE:
        out.append("E02's window has closed (2027-10-31): too late for a reading registered "
                   "before the outcome")
    for k, v in d["reused_code"]["from"].items():
        if v == PENDING:
            out.append(f"{k} is not in the registry")
    if d["worlds"]["registration"] == PENDING:
        out.append(f"{R.FV2_ID} is not in the registry")
    out += R.world_problems(P, None)
    out += attempts_problems(P)
    if not d["reused_code"]["current_files_match"]:
        out.append(f"reused code differs from its pins: {R.reused_code_problems()}")
    if not d["reused_code"]["pins_match_declarations"]:
        out.append("run_e02d1.REUSED differs from the source declarations' frozen_code")
    if d["rule"]["protocol_sha256"] != R.E02_PROTOCOL_SHA:
        out.append("E02 protocol.json missing or not the registered one")
    if d["rule"]["stated"] != d["rule"]["declared"]:
        out.append("E02's protocol states another rule than the declared one")
    if _sha(P.v1.pred) != R.E02_PRED_SHA:
        out.append("E02 predictions.csv missing or not the registered one")
    if _sha(P.e02_manifest()) != R.E02_MANIFEST_SHA:
        out.append("E02 manifest.json missing or not the registered one")
    for m in d["manifest_checks"]:
        if m.get("manifest") and m.get("allowed") is not True:
            out.append(f"{m['use']} is not an allowed use of {m['product']} (MANIFEST)")
    key = f"2026-11-07 sixteenth set, {R.RULING_ID}"
    if d["rulings"][key] == PENDING or d["rulings"]["file"]["sha256"] != R.RULINGS_SHA:
        out.append(f"ruling {R.RULING_ID} not found in {P.rulings} at sha "
                   f"{R.RULINGS_SHA[:8]}")
    if O.rp(P.report).exists():
        out.append(f"{P.report} exists before the declaration")
    out += [f"frozen code {p} missing" for p, h in d["frozen_code"].items() if h == MISSING]
    out += [f"input {p} missing" for p in d["inputs"] if not O.rp(p).exists()]
    out += _git_problems([*R.FROZEN, R.PREREG])
    if R.index_record(R.ID, prereg_dir):
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
    print("readiness:", readiness(d, prereg_dir=prereg_dir) or "ready to register")
    print("this script sha256:", O.sha256(Path(__file__)))
    print("not written: the lead registers with prereg3.write_prereg after review")
    return 1 if errs else 0


if __name__ == "__main__":
    raise SystemExit(main())
