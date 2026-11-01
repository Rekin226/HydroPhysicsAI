"""The three round-7 rank-12 declarations (harness.prereg3, kind 'declaration'), each written
BEFORE any of its worlds is drawn:

  --decl v2    FV.decidability.v2_postfailure  the single post-failure exploratory follow-up of
                                               FV.decidability.v1 (osse_fv2.py, forcing_year.py)
  --decl ext   FV.ext.v1                       fresh: N3F's external-well rule on v2's worlds
                                               (ext_fv.py)
  --decl stat  FV.stat.v1                      fresh, CPU: E04 (and I1 T1/T2 report-only) in
                                               AR(1) worlds (stat_fv.py)

User rulings: 2026-11-01 fourth set rank 11 (FV v1: P <= 0.5 = 'weak by design', one-sided);
2026-11-01 sixth set R7.fv_v2_page (the v2 reading may reach the page, labelled
'post-failure, exploratory').

Order for the lead (repository root, hydro env; the builder ran none of this on real data):
  0. prereg_fv2.py --log-look builder --station <id> [--station <id>] --note '<values>'
     prereg_fv2.py --log-look stat_preview         the builder's accidental look and FV.stat's
                                                   K = 8 preview (values only in looks.jsonl)
     ext_fv.py log-look                            design-time read of N3F's export (FV.ext)
  1. bash run_fv2.sh setup                         frozen worktree at 5f3af18 + py.sh (no GPU)
  2. prereg_fv2.py --decl v2 --validate --replay-status '<who, when, status>'
         --c14b-trigger either_county|pooled
     prereg_fv2.py --decl ext --validate
     prereg_fv2.py --decl stat --validate          dry runs: JSON + sha, readiness (no write)
  3. prereg_fv2.py --decl {v2|ext|stat} --write --expect-self-sha <sha of this file> [...]
     (all three before 'run_fv2.sh arms'; v2 also before rank 11's replay forwards are read)
  4. bash run_fv2.sh arms | forward | score-worlds | score-ext | score-forcing | score-stat

--write refuses unless readiness is empty and --expect-self-sha matches. Nothing here writes
outside the registry given (the tests use a tmp registry).
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
REPO = HERE.parents[1]
for _p in (str(HERE), str(REPO / "dataprep_staging" / "fv_decidability"),
           str(REPO / "dataprep_staging" / "e02c"), str(REPO / "dataprep_staging"), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ext_fv as X  # noqa: E402
import forcing_year as FYM  # noqa: E402
import osse_fv as O  # noqa: E402
import osse_fv2 as V2  # noqa: E402
import score_e02c as S  # noqa: E402
import stat_fv as ST  # noqa: E402
from harness import prereg3  # noqa: E402

NA = prereg3.not_applicable
PENDING = "PENDING"
MISSING = "MISSING"
TESTS = f"{V2.F}/tests/test_fv2.py"
RULINGS_R7 = "results/dataprep/rulings/user_rulings_2026-11-01b.json"
RULINGS_4 = "results/dataprep/rulings/user_rulings_2026-11-01.json"
R7_RULING = "R7.fv_v2_page"
NOT_BEFORE_E02 = dt.datetime(2028, 1, 15, tzinfo=dt.timezone.utc)
RUNNER_LINES = (f"ID={V2.ID}", f"WT_HEAD={V2.WT_HEAD}", f'WT="$HOME/{V2.WT_NAME}"',
                f"PY={V2.PY_SH}", f'FLAGS="{V2.FWD_FLAGS}"', f"RUNS={V2.RUNS}",
                f"ARMS={V2.ARMS_DIR}", f"BATCH={V2.BATCH}", f"NBATCH={V2.PLAN.n_batch}",
                f"ZR0_NPZ={O.ZR0_NPZ}")
RUNNER_FRAGMENTS = ('"$PY" -m hydrophysics.twin.forward "${THETAS[@]}" $FLAGS',
                    '--vep-json "$R/$dir/coupled_leveling/vep_zonal_leveling.json"',
                    '${SPECS[@]+"${SPECS[@]}"} --out "$out" > "$sealed" 2> "$err"',
                    'out=results/twin_forward/fv2_truth_${tag}_b${b}')
AFTER_THE_FACT_KNOWN = {   # v1's list (prereg_fv.py), plus what is known since
    "head_rmse_ratio_vs_climatology_2023_26": "about 1.22 (results/twin/new_data_challenge_"
                                              "innov; disclosed by E02c)",
    "ensemble_only_coverage_2023_26": "0.75; combined-sigma 0.92-0.93 (U7, E02c)",
    "combined_sigma_coverage_2023_26": "92.6 % (published after the fact; U7, E02c)",
    "fv_v1": "not evaluable: superposition max 0.2845 m vs 0.02 m (report.json); no world "
             "statistic, P or reading was computed",
    "use_v2": "none of these chose a truth, a noise rule, K, a threshold or a reading here"}


# ----------------------------------------------------------------------------- helpers
def sha_text(obj) -> str:
    return hashlib.sha256((json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=1)
                           + "\n").encode()).hexdigest()


def _sha(p) -> str:
    q = O.rp(p)
    return O.sha256(q) if q.is_file() else MISSING


def _json(p) -> dict | None:
    q = O.rp(p)
    return json.loads(q.read_text(encoding="utf-8")) if q.is_file() else None


def index_record(prereg_id: str, prereg_dir=None) -> dict | None:
    d = Path(prereg_dir) if prereg_dir else prereg3.PREREG_DIR
    rows = [r for r in prereg3.read_chained(d / "INDEX.jsonl") if r["id"] == prereg_id]
    return rows[-1] if rows else None


def all_ids(prereg_dir=None) -> list[str]:
    d = Path(prereg_dir) if prereg_dir else prereg3.PREREG_DIR
    return [r["id"] for r in prereg3.read_chained(d / "INDEX.jsonl")]


def score_looks(prereg_ids, prereg_dir=None) -> list[dict]:
    d = Path(prereg_dir) if prereg_dir else prereg3.PREREG_DIR
    return [lk for lk in prereg3.read_chained(d / "looks.jsonl")
            if lk.get("purpose") == "score" and lk.get("prereg_id") in prereg_ids]


def ruling(rid: str) -> dict | None:
    r = _json(RULINGS_R7) or {}
    return next((x for x in r.get("rulings", []) if x.get("id") == rid), None)


def _worktree_problems() -> list[str]:
    wt = Path.home() / V2.WT_NAME
    try:
        head = subprocess.run(["git", "-C", str(wt), "rev-parse", "--short=7", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(wt), "status", "--porcelain",
                                "--untracked-files=no"], capture_output=True, text=True,
                               check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as err:
        return [f"worktree $HOME/{V2.WT_NAME}: {err} (run_fv2.sh setup)"]
    out = []
    if head != V2.WT_HEAD:
        out.append(f"worktree is at {head}, not {V2.WT_HEAD}")
    if dirty:
        out.append("worktree is dirty")
    py = O.rp(V2.PY_SH)
    if not py.exists() or f"export PYTHONPATH={wt}" not in py.read_text().splitlines():
        out.append(f"{V2.PY_SH} does not pin the worktree (run_fv2.sh setup)")
    return out


def runner_problems() -> list[str]:
    p = O.rp(V2.RUNNER)
    if not p.is_file():
        return [f"{V2.RUNNER} missing"]
    sh = p.read_text()
    lines = set(sh.splitlines())
    out = [f"{V2.RUNNER} does not carry the line {s!r}" for s in RUNNER_LINES if s not in lines]
    out += [f"{V2.RUNNER} does not carry {s!r}" for s in RUNNER_FRAGMENTS if s not in sh]
    if "/home/" in sh:
        out.append(f"{V2.RUNNER} hardcodes a /home path")
    return out


def _common_fields(frozen: dict, inputs: list[str]) -> dict:
    return {"zones": NA("no fit; the frozen forward runs' own zones"),
            "burn_in": NA("no fit"), "folds": NA("synthetic worlds; no folds"),
            "gate_scores": NA("readings by one threshold; no weighted score"),
            "tests": NA("model-only declaration: synthetic worlds, no observation is scored "
                        "and no fresh unit is consumed"),
            "fresh_units": NA("no observation is scored"),
            "frozen_code": frozen, "inputs": sorted(set(inputs))}


def _disclosure_tests() -> dict:
    return {"path": TESTS, "sha256_at_declaration": _sha(TESTS),
            "binding": "disclosure only, not an input: a later test edit does not invalidate "
                       "this declaration"}


BUILDER_LOOK_UNIT = "FV2:builder_accidental_e04_rows"
BUILDER_LOOK_SCOPE = "heads:2012-01..2026-09"
BUILDER_LOOK_PURPOSE = "accidental_builder_look"
STAT_PREVIEW_UNIT = "FV2:stat_preview_k8_declared_seed"
STAT_PREVIEW_SCOPE = "simulation:FV.stat.v1"
STAT_PREVIEW_PURPOSE = "design_time_preview"
BUILDER_LOOKS = (
    "The builder (round 7, 2026-11-01) ran synthetic smokes only. To learn formats and the "
    "v1 outcome it read: FV.decidability.v1's declaration (thresholds, inputs_frozen, "
    "frozen_code shas), report.json (status not_evaluable, superposition diagnostics, "
    "engineering block; it holds no P and no world statistic) and attempts.jsonl (forward "
    "shas and run times); the N3F registration's export metadata (66 kept wells: 23 "
    "Changhua, 6 Nantou, 37 Yunlin; file paths and shas); the header line of the N3F "
    "export's wells.csv; the hybrid (E04) protocol.json and manifest.json (registered "
    "2026-10-11, 147 stations, origins 2026-07/08/09 = 8/6/133). By accident it also printed "
    "the E04 parameters.json pooled summary, ONE well's row of that file (fitted "
    "parameters, including the anomaly at that well's 2026-09 origin, a value derived from "
    "a post-origin observation) and the first row of the hybrid predictions.csv. The "
    f"printed values are recorded only in the lead's look {BUILDER_LOOK_UNIT} "
    "(looks.jsonl, logged before this declaration), never in a committed file. No head, "
    "leveling, GNSS, pile, ring, township or ls_cache value was opened, nothing after "
    "2022-12 except that accidental print, and no sealed log.")


def logged_look(unit: str, purpose: str, prereg_dir=None) -> dict | None:
    d = Path(prereg_dir) if prereg_dir else prereg3.PREREG_DIR
    for lk in prereg3.read_chained(d / "looks.jsonl"):
        if lk.get("units") == [unit] and lk.get("purpose") == purpose:
            return {k: lk.get(k) for k in ("time", "wp", "purpose", "units", "data_scope")}
    return None


def builder_look_problems(prereg_dir=None, stat: bool = False) -> list[str]:
    out = []
    if not logged_look(BUILDER_LOOK_UNIT, BUILDER_LOOK_PURPOSE, prereg_dir):
        out.append(f"the builder's accidental look {BUILDER_LOOK_UNIT} is not logged "
                   "(prereg_fv2.py --log-look builder) before the declaration that cites it")
    if stat and not logged_look(STAT_PREVIEW_UNIT, STAT_PREVIEW_PURPOSE, prereg_dir):
        out.append(f"FV.stat's K = 8 preview {STAT_PREVIEW_UNIT} is not logged "
                   "(prereg_fv2.py --log-look stat_preview)")
    return out


def log_builder_look(which: str, *, stations=None, note: str = "", prereg_dir=None,
                     rows_dir=None) -> dict:
    """The lead logs the builder's accidental look (values in ``note``, typed by the lead from
    the builder's hand-off; never in a committed file) or FV.stat's preview."""
    kw = {k: v for k, v in (("prereg_dir", prereg_dir), ("rows_dir", rows_dir)) if v}
    if logged_look(*((BUILDER_LOOK_UNIT, BUILDER_LOOK_PURPOSE) if which == "builder" else
                     (STAT_PREVIEW_UNIT, STAT_PREVIEW_PURPOSE)), prereg_dir):
        raise prereg3.PreregistrationError(f"look {which} is already logged")
    if which == "builder":
        if not stations or not note.strip():
            raise ValueError("--log-look builder needs --station (each well printed) and "
                             "--note (the printed values, verbatim)")
        members = {BUILDER_LOOK_UNIT: {
            "ids": sorted(map(str, stations)), "window": "2012-01..2026-09",
            "description": "accidental print by the FV v2 builder (2026-11-01) while reading "
                           "file formats: the E04 (twin.hybrid) parameters.json pooled "
                           "summary, one well's fitted row (its 2026-09 origin anomaly is "
                           "derived from a post-origin observation) and the first row of "
                           "the hybrid predictions.csv; it set nothing in FV v2, FV.ext or "
                           "FV.stat"}}
        return prereg3.record_look([BUILDER_LOOK_UNIT], BUILDER_LOOK_SCOPE, V2.WP,
                                   BUILDER_LOOK_PURPOSE, members=members, note=note, **kw)
    members = {STAT_PREVIEW_UNIT: {
        "none": "synthetic AR(1) worlds only; no observation",
        "description": "FV.stat.v1's study run once by the builder at K = 8 per cell on the "
                       "DECLARED seed before registration (pooled P about 0.92, "
                       "'not_shown_weak'); the first 8 worlds per cell of the K = 500 score"}}
    return prereg3.record_look([STAT_PREVIEW_UNIT], STAT_PREVIEW_SCOPE, V2.WP,
                               STAT_PREVIEW_PURPOSE, members=members,
                               note=note or ST.BUILDER_PREVIEW, **kw)


# ----------------------------------------------------------------------------- v2
def v1_block(prereg_dir=None) -> dict:
    rec = index_record(V2.V1_ID, prereg_dir)
    rep = _json(O.REPORT)
    out = {"id": V2.V1_ID, "registration": rec or PENDING, "declared_sha256": V2.V1_DECL_SHA,
           "report": {"path": O.REPORT, "sha256": _sha(O.REPORT)}}
    if rep is not None:
        sup = rep.get("superposition") or {}
        out.update({"status": rep.get("status"), "why": rep.get("why"),
                    "superposition_max_abs_window_m": sup.get("max_abs_window_m"),
                    "superposition_tolerance_m": sup.get("tolerance_m")})
    return out


def v2_null_block() -> dict:
    """v1's declared tolerance, recomputed from v1's bound inputs when they are present."""
    out = {"tolerance": V2.V1_NULL_TOLERANCE, "source": f"{V2.V1_ID} null_world.tolerance",
           "gate": "the null world's E02c level false-pass rate (K_NULL worlds, obs = CL + "
                   "AR(1) noise; v1's stream) must be <= tolerance; otherwise not_evaluable"}
    try:
        ctx = O.load_context(O.Paths())
        t = O.null_tolerance(ctx["zr"] - ctx["cl"], ctx["sd"], ctx["phi"], V2.K_NULL)
        out["recomputed"] = t["tolerance"]
        out["identical_to_v1"] = abs(t["tolerance"] - V2.V1_NULL_TOLERANCE) <= 1e-12
    except (OSError, KeyError, ValueError, O.EngineeringFailure) as err:
        out["recomputed"] = PENDING
        out["why_pending"] = f"{type(err).__name__}: {err}"
    return out


def doc_v2(prereg_dir=None, replay_status: str | None = None,
           c14b_trigger: str | None = None) -> dict:
    v1d = _json(V2.V1_DECL) or {}
    R = "results/twin_runs"
    truths = [{"tag": t, "theta": f"{R}/{d}/stage3_theta.json",
               "column": f"{R}/{d}/coupled_leveling/vep_zonal_leveling.json",
               "v1_forward": {"path": V2.v1_truth_npz(t), "sha256": V2.V1_FORWARD_SHA[t],
                              "role": "hindcast-identity reference (gate E) and v1's check "
                                      "forwards (report-only comparison)"},
               "batches": [V2.batch_npz(t, b) for b in range(V2.PLAN.n_batch)]}
              for t, d in O.TRUTHS]
    frozen = {p: _sha(p) for p in V2.FROZEN}
    inputs = [*V2.FROZEN, V2.PREREG, V2.V1_DECL, O.REPORT, O.E02_PRED, O.E02_PROTOCOL,
              f"{S.E02_DIR}/manifest.json", S.FORWARD, f"{O.E02C_BASE_DIR}/baselines_e02c.csv",
              f"{O.E02C_BASE_DIR}/wells_158.csv",
              f"results/dataprep/preregistration/{S.ID}.json",
              *[t["theta"] for t in truths], *[t["column"] for t in truths],
              *O.zr0_thetas(), O.zr0_column(), O.ZR0_NPZ,
              *[V2.v1_truth_npz(t) for t in O.TAGS], *[p for p, _ in O.REPRO.values()],
              *[f"{O.INPUTS_DIR}/{n}" for n in (*O.INPUT_FILES, "summary.json", "verify.json")],
              RULINGS_4, RULINGS_R7]
    readings = {
        "reading_1": {
            "status": "v1's readings, unchanged (threshold, rule, verdicts, wording intent)",
            "threshold": V2.P_WEAK,
            "rule": "P <= 0.5 -> 'weak_by_design'; P > 0.5 -> 'not_shown_weak'. One-sided: no "
                    "'a fail is informative' reading exists. Applied once, to the point "
                    "estimate (the 90 % Monte Carlo half-width <= 0.07 check is non-binding "
                    "at this K; P - 0.5 is reported in Monte Carlo SE units)",
            "P": "equal-weight mean over the four truths of each truth's pass rate over "
                 "K_TRUTH = 250 full-forward worlds",
            "verdicts": (v1d.get("readings") or {}).get("verdicts") or PENDING,
            "wording": V2.WORDING},
        "reading_2": {
            "status": "NEW, declared as new (not a v1 reading): forcing-year completeness of "
                      "the frozen combined sigma on round-7 rank 11's replay worlds",
            "worlds": "FW.weather_replay.v1's primary arms W2012..W2022 (the page ensemble "
                      "replaying the cyclic order of 2012..2022 starting at 2012+s, detrended "
                      "pumping; hydrophysics.twin.replay) and its arm C (the same ensemble "
                      "and code under climatology); F_s = h_s - h_C (baseline scenario) at "
                      "the 158 wells and on layer 2. Index schema " + FYM.INDEX_SCHEMA + ", "
                      "built by 'osse_fv2.py replay-index' from npz and capture-record "
                      "metadata only, after FW's report exists (FW's runner seals its "
                      "outputs until then)",
            "rc": f"per county share of window well-months with ZR + F_s + e inside E02's "
                  f"frozen [lower, upper], 11 sequences x K_RC = {FYM.K_RC} AR(1) draws "
                  f"(v1's noise inputs; stream {FYM.STREAM_RC}); RC < {FYM.RC_LOW} -> "
                  f"'omits'; RC >= {FYM.RC_HIGH} -> 'not_shown_to_omit'; otherwise no line",
            "lead_ratio": f"median over wells of sd_s(F_s(w, L)) / sigma_w(L), leads "
                          f"{list(FYM.LEADS)}, per county and pooled; C14b -> L if the 47-58 "
                          f"ratio exceeds {FYM.LEAD_RATIO_L} per the declared c14b_trigger",
            "c14b_trigger": c14b_trigger or PENDING,
            "c14b_trigger_meaning": {"either_county": "per county; either county above "
                                                      "1.2 moves C14b to L (cautious)",
                                     "pooled": "the median over the wells of both counties "
                                               "above 1.2 moves C14b to L"},
            "sigma_by_lead": "sqrt(max(sigma_ens, 1e-3)^2 + sd_w^2), sigma_ens = heads_std "
                             "of the registered forward, sd_w solved from the window rows; "
                             "pre-declared alternative 'sigma_const' (window-mean sigma at "
                             "every lead) if the reconstruction misses by > 1e-4 m",
            "icc_year": "REPORT-ONLY: county-common share of the window-mean forcing "
                        "departure (forcing only and with the AR(1) noise)",
            "aggregates": "C14c: min..max over sequences of the N2-unit mean (84 Yunlin "
                          "wells) and the fan layer-2 mean, by lead",
            "uncertainty": "Monte Carlo SE and the 11-year sampling uncertainty (Student t "
                           "90 % interval over the 11 sequence values and a bootstrap over "
                           "sequences, B = 2000) are reported, the t interval is quoted in "
                           "each C14a line, and neither moves a threshold",
            "changhua": {"by_rain_mode": FYM.WORDING["changhua"],
                         "pumping_only": f"rain 'climatology' (FW ran --replay-fields "
                                         f"energy): Changhua's reading is "
                                         f"'{FYM.NOT_REPRESENTED}' and it gets no C14a line "
                                         "(no Changhua forcing-year variation exists in the "
                                         "model)"},
            "rain": "the index states 'replayed' or 'climatology' (pumping years only); the "
                    "wording follows it ('replayed forcing years', never 'replayed weather' "
                    "when only pumping is replayed)",
            "gate_e_on_fw_forwards": "a gate-E failure in the content of FW's forwards "
                                     "(capture records, metadata, axis, members, months, "
                                     "hindcast identity 0.0 m, identical sequences) cannot be "
                                     "retried by FV: it is logged (attempts.jsonl event "
                                     f"{FYM.GATE_E_EVENT}) and maps to 'replay worlds "
                                     "unusable' (not evaluable); FV-side index problems stay "
                                     "engineering retries",
            "reconciling": "the 'omits' wording states the after-the-fact 92.6 % 2023-26 "
                           "coverage is a different quantity, shown separately",
            "wording": FYM.WORDING}}
    page = {
        "ruling": f"{R7_RULING} (user, 2026-11-01): the reading may reach the page, labelled "
                  "'post-failure, exploratory'",
        "before_scoring": "no change; evc_pro_fv_ne (v1 not evaluable) stays",
        "level_weak_by_design": "evc_pro_fv carries the v2 wording (P, 'weak by design', "
                                "'post-failure, exploratory') in place of evc_pro_fv_ne; C30's "
                                "heads part stays N whatever E02c's 2028 level outcome",
        "level_not_shown_weak": "evc_pro_fv states P and 'not shown weak (post-failure, "
                                "exploratory)'; E02c's mapping for C30 applies unchanged",
        "coverage": "per county, the one-sided line beside E02c's coverage line; C14a's "
                    "mapping is unchanged",
        "reading_2": {"C14a": "per county: the 'omits' line (RC < 0.85) or the "
                              "'not_shown_to_omit' line (RC >= 0.88); otherwise nothing",
                      "C14b": "N -> L if the 47-58 lead ratio exceeds 1.2 in either county; "
                              "otherwise stays N with the lead curve cited",
                      "C14c": "reworded: 'run spread, a parameter spread, plus the model's "
                              "own forcing-year range ...; not a predictive interval'",
                      "icc": "report-only (ledger)",
                      "changhua": "the Changhua sentence of the replay's rain mode: 'rain-year "
                                  "range only; the page model carries no Changhua pumping' "
                                  "when rain is replayed; when only pumping is replayed, "
                                  "'no Changhua forcing-year variation is represented' and "
                                  "no Changhua C14a line"},
        "not_evaluable": "evc_pro_fv_ne stays; the v2 not-evaluable sentence goes to the "
                         "ledger and README; no further follow-up exists",
        "never": "no S anywhere; no head-response claim (C01, C24, C13a) and no subsidence "
                 "claim changes; 'a fail is informative' is never written; nothing is scaled "
                 "up for the damped amplitude (ruling 2026-10-29 (4))"}
    attempt = {
        "slot": f"the single exploratory follow-up allowed by {V2.V1_ID}'s attempt_policy "
                "('v2, label post-failure, exploratory, discloses v1'); no further follow-up "
                "exists (exploratory_followups_max 0)",
        "scored_runs": "reading 1 and reading 2 are each scored once (report_worlds.json, "
                       "report_forcing_year.json, open mode 'x'); the first run that passes "
                       "gate E is final, whatever gates N or MC say",
        "engineering_retries": "free while gate E fails (crash, OOM, missing or mismatched "
                               "file, metadata, hindcast identity): nothing is written, every "
                               "attempt is logged in attempts.jsonl, a failed batch is "
                               "archived unopened (run_fv2.sh archive)",
        "pre_declared_alternatives": ["reading 2 lead ratio: sigma_const if the sigma "
                                      "reconstruction misses (labelled)",
                                      "reading 2: 'replay worlds unavailable' (not evaluable) "
                                      "only on a pre-declared ground, checked by the scorer: "
                                      "FW's report states not_evaluable; FW's report is "
                                      "absent on or after 2027-01-31; or FV's gate E failed "
                                      "on the content of FW's forwards (logged)"],
        "exploratory_followups_max": 0, "thresholds_frozen": True,
        "report": "every attempt, failures included"}
    d = {
        "id": V2.ID, "wp": V2.WP, "kind": "declaration",
        "title": "FV v2 (post-failure, exploratory): E02c's level and coverage decidability "
                 "with every synthetic world a full forward, plus a new forcing-year "
                 "completeness reading of the frozen 90 % head interval",
        "label": "post-failure, exploratory",
        "v1": v1_block(prereg_dir),
        "method_change": {
            "only_change": "each world's multiplier series is its own forward scenario "
                           "(--recorded-pumping arm); R_T(world k) = h_T(world k) - h_T(batch "
                           "baseline) in the window. Gate S (superposition) has nothing left "
                           "to check and is removed",
            "unchanged_from_v1": ["truths and their thetas/columns", "ZR (E02's registered "
                                  "predictions) and ZR0 (v1's same-code reference forward, "
                                  "reused, sha pinned)", "D_T (departure against ZR0, "
                                  "observed-month anchor)", "AR(1) noise inputs, rule and "
                                  "stream", "world draws (v1's stream STREAM_ANOM, truth, k)",
                                  "K_TRUTH 250, K_NULL 500, SEED", "P_weak 0.5",
                                  "null tolerance 0.006", "null-world gate",
                                  "X5R1 reproduction (0.02 m, Q0 report-only)",
                                  "statistics (score_e02c imported)", "one-sided readings"],
            "engineering_added": "every batch baseline's 2012-22 hindcast equals v1's truth "
                                 "forward at the 158 wells within 0.0 m; Monte Carlo check "
                                 "1.645 SE <= 0.07 (non-binding at this K)"},
        "priors": (
            "v1 failed on its engineering surrogate, not on science: the superposition of "
            "one-month impulse arms missed the full forwards by 0.28 m. A full forward per "
            "world removes that failure mode. The round-7 planners put 'weak by design' for "
            "the level rule at about 0.5-0.6 and the per-county coverage gates likelier "
            "'not shown weak'. The reading cannot produce support. Reading 2 is new: the "
            "frozen sigma was fitted with recorded forcing, so it may omit the model's own "
            "forcing-year variation; the planners put a Yunlin narrowing at about 0.35."),
        "truths": truths,
        "zr0": {"npz": O.ZR0_NPZ, "sha256": V2.V1_FORWARD_SHA["ZR0"],
                "role": "v1's same-code reference forward (18 parameter sets of "
                        "datum_gate_innov, baseline only, at 5f3af18), reused unchanged"},
        "v1_forwards": dict(V2.V1_FORWARD_SHA),
        "truth_world": {
            "formula": "obs = ZR + D_T + [h_T(world k) - h_T(baseline)] + e on 2026-11..2027-10 "
                       "at E02c's 158 wells",
            "worlds": f"K_TRUTH = {V2.K_TRUTH} per truth, world k = v1's draw k (stream "
                      f"{O.STREAM_ANOM}); forwards of {V2.BATCH} worlds + baseline, "
                      f"{V2.PLAN.n_batch} per truth, {4 * V2.PLAN.n_batch} in all",
            "observed": "all 158 wells x 12 months (optimistic, as v1)"},
        "inputs_frozen": v1d.get("inputs_frozen") or PENDING,
        "replay_v1_use": {"product": f"{O.PUMP} (pump_electricity/preorigin v3)",
                          "use": O.PUMP_USE, "use_confirmation_v1": (v1d.get("replay") or {})
                          .get("use_confirmation", PENDING),
                          "note": "no new product use: the same pool file, read through v1's "
                                  "verified export"},
        "noise": {"source": f"v1's export {O.INPUTS_DIR} (noise_ar1.csv), look "
                            f"{O.LOOK_UNIT} logged by v1", "new_look": "none"},
        "design": {"K_TRUTH": V2.K_TRUTH, "K_NULL": V2.K_NULL, "SEED": V2.SEED,
                   "batch": V2.BATCH, "n_batch": V2.PLAN.n_batch,
                   "streams": {"anomaly": [O.SEED, O.STREAM_ANOM, "truth", "draw"],
                               "noise": [O.SEED, O.STREAM_NOISE, "truth", "draw"],
                               "null": [O.SEED, O.STREAM_NULL, 0, "draw"],
                               "selfcheck": [O.SEED, O.STREAM_SELFCHECK, 0, "draw"],
                               "rc": [O.SEED, FYM.STREAM_RC, "sequence", "draw"],
                               "rc_null": [O.SEED, FYM.STREAM_RC_NULL, 0, "draw"],
                               "bootstrap": [O.SEED, FYM.STREAM_BOOT, "-", "-"]},
                   "truth_order": list(O.TAGS), "window": list(S.WINDOW),
                   "world_names": "w000..w249 per truth; batch b holds w(50b)..w(50b+49)"},
        "statistics": {"code": f"{S.E}/score_e02c.py via {O.SELF} (world_stats, g0_e02, "
                               "null_tolerance, evaluate_selfcheck), imported unchanged",
                       "baselines": f"E02c's registered {O.E02C_BASE_DIR}"},
        "gates": {
            "E": "engineering, passed completely before any N, MC or world number: "
                 "registration, frozen code, v1 inputs and forwards (shas), every world file "
                 "the declared draw, batch metadata and arm shas, every batch baseline's "
                 "2012-22 hindcast within 0.0 m of v1's truth forward, X5R1 within 0.02 m of "
                 "datum_gate_x5r1.npz (Q0 report-only), ZR0 metadata, v1's null tolerance "
                 "recomputed identically, g0_e02 = twin.validation.evaluate on two self-check "
                 "worlds; reading 2: FW's report exists, index and capture shas, capture "
                 "records (C climatology; W sequence s = the declared cyclic years, "
                 "detrended), npz metadata, every replay's 2012-22 hindcast within 0.0 m of "
                 "arm C at the wells, no two sequences identical",
            "N": "reading 1: v1's null world, false pass <= 0.006; reading 2: null-world "
                 f"coverage within {FYM.NULL_COV_TOL} of the analytic value per county",
            "MC": {"rule": f"each reading: 1.645 x Monte Carlo SE <= {V2.MC_HALF_WIDTH}",
                   "reading_1": V2.mc_note(len(O.TAGS), V2.K_TRUTH),
                   "reading_2": V2.mc_note(FYM.N_SEQ, FYM.K_RC)},
            "code_freeze": "no twin module is edited in the frozen worktree while the forwards "
                           "run; the worktree must stay clean (checked per forward)"},
        "null_world": v2_null_block(),
        "readings": readings,
        "page_mapping": page,
        "e02c_registration": index_record(S.ID, prereg_dir) or PENDING,
        "replay_status_at_declaration": replay_status or PENDING,
        "builder_look": logged_look(BUILDER_LOOK_UNIT, BUILDER_LOOK_PURPOSE, prereg_dir)
        or PENDING,
        "fw_report_at_declaration": {"path": FYM.FW_REPORT, "sha256": _sha(FYM.FW_REPORT),
                                     "rule": "preferably absent: register v2 before FW's "
                                             "report is written; if present, --replay-status "
                                             "must say that nobody read an FV-relevant number "
                                             "(head spread by year or county) from it"},
        "run": {"worktree": f"$HOME/{V2.WT_NAME} at {V2.WT_HEAD} (clean, detached; v1's "
                            "commit, so ZR0 and v1's truth forwards are same-code)",
                "py_sh": V2.PY_SH, "flags": V2.FWD_FLAGS,
                "runner": f"bash {V2.RUNNER} setup | arms | forward [TAG [B]] | "
                          "score-worlds | replay-index | score-forcing; sha in frozen_code",
                "gpu": f"{4 * V2.PLAN.n_batch} forwards of {V2.BATCH + 1} scenarios x 58 "
                       "months, one member (v1: 11-18 min per 60 scenarios) -> about 4-6 "
                       "GPU-h on the T4, about 2-3 h wall under 2-3-way MPS; disk about 8-9 GB",
                "cpu": "reading 1 scoring about 15-30 min; reading 2 about 10 min"},
        "sealed_until_scoring": "forward stdout (sealed file, sha logged); batch npz values "
                                "are read only by the scorers (this one and FV.ext.v1); the "
                                "runner prints exit codes and sha256s only; rank 11's replay "
                                "forwards are read for FV only by score-forcing",
        "attempt_policy": attempt,
        "rulings": {"2026-11-01 fourth set, rank 11": "FV v1: RUN; P <= 0.5 = 'weak by "
                                                      "design'; one-sided reading",
                    f"2026-11-01 sixth set, {R7_RULING}": (ruling(R7_RULING) or {}).get(
                        "decision", PENDING)},
        "disclosures": {
            "v1": "FV.decidability.v1 (sha cf2adf0d) was scored not evaluable on 2026-11-01: "
                  "superposition max 0.2845 m against 0.02 m. Its scorer opened the truth "
                  "forwards for gates E and S only; no null-world or truth-world statistic, "
                  "P or reading exists. v1's check forwards are full forwards of five of "
                  "this study's worlds; their window values were read only by v1's gate S "
                  "(max and RMS differences, published in v1's report)",
            "timing": "written on or after 2026-11-01; no window observation exists or is "
                      "read; precedes any E02 or E02c scoring (not before 2028-01-15)",
            "after_the_fact_known": AFTER_THE_FACT_KNOWN,
            "builder": BUILDER_LOOKS,
            "optimism": "rain at climatology (reading 1); noise independent between wells; "
                        "full availability; Yunlin-only pumping anomalies; the model's "
                        "damped amplitude (about 0.73) in reading 2",
            "pessimism": "the AR(1) noise is fitted to free-run hindcast residuals that "
                         "already hold structural misfit; D_T adds a recipe departure on top",
            "truths_share_family": "the four truths share the model family, so "
                                   "'not_shown_weak' may overstate decidability",
            "reading_2_dependency": "reading 2 uses another declaration's forwards "
                                    "(FW.weather_replay.v1); their rain provenance follows "
                                    "that declaration",
            "tests": _disclosure_tests()},
        "primary_model": {"forecast": "ZR, E02's registered predictions (forward cf2c7154)",
                          "truths": list(O.TAGS)},
        "thresholds": {"P_weak": V2.P_WEAK, "K_TRUTH": V2.K_TRUTH, "K_NULL": V2.K_NULL,
                       "null_tolerance": V2.V1_NULL_TOLERANCE, "repro_tol_m": V2.REPRO_TOL_M,
                       "hindcast_identity_tol_m": V2.HINDCAST_TOL_M,
                       "mc_half_width_90": V2.MC_HALF_WIDTH, "rc_low": FYM.RC_LOW,
                       "rc_high": FYM.RC_HIGH, "lead_ratio_L": FYM.LEAD_RATIO_L,
                       "null_cov_tol": FYM.NULL_COV_TOL, "K_RC": FYM.K_RC,
                       "K_RC_NULL": FYM.K_RC_NULL, "sigma_rec_tol_m": FYM.SIGMA_REC_TOL_M},
        "interpretation": (
            "A model-only, post-failure exploratory operating characteristic of E02c's "
            "verdicts under optimistic truths (reading 1, v1's readings) and a model-only "
            "check of whether the frozen interval includes the model's own forcing-year "
            "variation (reading 2, new). Neither is evidence about the aquifer and neither "
            "can support (S) anything."),
        "written_by": V2.PREREG,
        **_common_fields(frozen, inputs)}
    return d


def readiness_v2(d: dict, prereg_dir=None, now=None) -> list[str]:
    now = now or dt.datetime.now(dt.timezone.utc)
    out = []
    v1 = d["v1"]
    if v1["registration"] == PENDING:
        out.append(f"{V2.V1_ID} is not in the registry")
    elif v1["registration"].get("sha256") != V2.V1_DECL_SHA:
        out.append(f"{V2.V1_ID} registry sha differs from {V2.V1_DECL_SHA[:12]}")
    if v1.get("status") != "not_evaluable" or v1.get("why") != "superposition failed":
        out.append("v1's report does not read 'not_evaluable: superposition failed'")
    other = [i for i in all_ids(prereg_dir)
             if i.startswith("FV.decidability.") and i != V2.V1_ID]
    if other:
        out.append(f"v1's follow-up slot is taken: {other}")
    if d["e02c_registration"] == PENDING:
        out.append(f"{S.ID} is not in the registry")
    if score_looks({S.ID}, prereg_dir) or now >= NOT_BEFORE_E02:
        out.append("E02/E02c scoring has started (or 2028-01-15 passed): too late")
    if d["readings"]["reading_2"]["c14b_trigger"] not in FYM.C14B_TRIGGERS:
        out.append("--c14b-trigger either_county|pooled is required (the lead's explicit "
                   "choice; either_county is the cautious one)")
    if d["replay_status_at_declaration"] == PENDING:
        out.append("--replay-status '<who, when: whether rank 11's replay forwards and FW's "
                   "report exist, and that no FV reading was computed from them>' is required")
    if d["rulings"].get(f"2026-11-01 sixth set, {R7_RULING}") in (None, PENDING):
        out.append(f"ruling {R7_RULING} not found in {RULINGS_R7}")
    nw = d["null_world"]
    if nw.get("recomputed") == PENDING or not nw.get("identical_to_v1"):
        out.append("v1's null tolerance is not recomputed identically from v1's inputs")
    if d["inputs_frozen"] == PENDING:
        out.append("v1's inputs_frozen not found")
    for t, h in V2.V1_FORWARD_SHA.items():
        p = O.ZR0_NPZ if t == "ZR0" else V2.v1_truth_npz(t)
        if _sha(p) != h:
            out.append(f"v1 forward {p} is missing or differs from its pinned sha")
    for p in (V2.ARMS_DIR, V2.REPORT_WORLDS, V2.REPORT_FORCING,
              *[V2.batch_npz(t, b) for t in O.TAGS for b in range(V2.PLAN.n_batch)]):
        if O.rp(p).exists():
            out.append(f"{p} exists before the declaration")
    out += [f"frozen code {p} missing" for p, h in d["frozen_code"].items() if h == MISSING]
    out += [f"input {p} missing" for p in d["inputs"] if not O.rp(p).exists()]
    if index_record(V2.ID, prereg_dir):
        out.append(f"{V2.ID} is already registered")
    out += builder_look_problems(prereg_dir)
    return out + runner_problems() + _worktree_problems()


# ----------------------------------------------------------------------------- ext
def n3f_registration(prereg_dir=None) -> tuple[str | None, dict | None, dict | None]:
    """N3F's pinned 2026-11 external-well registration (EXT_DIR and the window are its own;
    a 2026-12 fallback, if ever registered, is not used by this declaration)."""
    rec = index_record(X.N3F_ID, prereg_dir)
    if not rec:
        return None, None, None
    d = Path(prereg_dir) if prereg_dir else prereg3.PREREG_DIR
    doc = json.loads((d / f"{X.N3F_ID}.json").read_text(encoding="utf-8"))
    return X.N3F_ID, rec, doc


def product_use_block() -> dict:
    """The heads product behind N3F's export and FV.ext's use of it, against its MANIFEST
    (allowed_uses / forbidden_uses; metadata only)."""
    man = _json(f"{X.PRODUCT}/MANIFEST.json")
    out = {"product": X.PRODUCT, "use": X.PRODUCT_USE,
           "via": f"N3F's registered export {X.EXT_DIR} (wells.csv residual_sd_m and datum_m, "
                  "predictions.csv baseline_m; written by e02c/export_baselines.py export_ext "
                  "under the uses datum_fit and baseline)",
           "reads": "the dry run (null tolerance from predicted_m - baseline_m and "
                    "residual_sd_m) and the scorer (noise sd, pre-origin record, registered "
                    "baseline)",
           "manifest_sha256": _sha(f"{X.PRODUCT}/MANIFEST.json")}
    if man is None:
        return {**out, "allowed": PENDING}
    ok = X.PRODUCT_USE in (man.get("allowed_uses") or []) and X.PRODUCT_USE not in (
        man.get("forbidden_uses") or [])
    return {**out, "allowed": ok, "allowed_uses": man.get("allowed_uses"),
            "forbidden_uses": man.get("forbidden_uses")}


def ext_null_block(ext_dir: str = X.EXT_DIR, prereg_dir=None) -> dict:
    if not X.look_logged(prereg_dir, ext_dir):
        return {"status": PENDING, "tolerance": None,
                "why": f"the design-time look {X.EXT_LOOK_UNIT} is not logged yet (ext_fv.py "
                       "log-look): the tolerance reads N3F's derived values only after it"}
    try:
        ctx = X.load_ext(X.ExtPaths(ext_dir=ext_dir))
        t = O.null_tolerance(ctx["zr"] - ctx["base_reg"], ctx["sd"], ctx["phi"], X.K_NULL)
        return {"status": "ok", **t, "gate": "null world obs = registered baseline + AR(1) "
                "noise, scored against it with N3F's rule: false pass <= tolerance"}
    except (OSError, KeyError, ValueError, O.EngineeringFailure) as err:
        return {"status": PENDING, "tolerance": None, "why": f"{type(err).__name__}: {err}"}


def doc_ext(prereg_dir=None, ext_dir: str = X.EXT_DIR) -> dict:
    pid, rec, n3f = n3f_registration(prereg_dir)
    reg_files = ((n3f or {}).get("external_predictions") or {}).get("files") or {}
    ext_files = {n: (reg_files.get(n) or {}).get("sha256", PENDING) for n in X.EXT_FILES}
    kept = ((n3f or {}).get("external_predictions") or {}).get("kept_ids") or PENDING
    frozen = {p: _sha(p) for p in X.FROZEN}
    inputs = [*X.FROZEN, V2.PREREG, *[f"{ext_dir}/{n}" for n in X.EXT_FILES], O.ZR0_NPZ,
              S.FORWARD, f"{O.INPUTS_DIR}/noise_ar1.csv", f"{O.INPUTS_DIR}/anomaly_pool.csv",
              *[V2.v1_truth_npz(t) for t in O.TAGS]]
    if pid:
        inputs.append(f"results/dataprep/preregistration/{pid}.json")
    nul = ext_null_block(ext_dir, prereg_dir)
    return {
        "id": X.ID, "wp": X.WP, "kind": "declaration",
        "title": "OSSE: can N3F's external-well level rule pass if a model of the right kind "
                 "is true? FV v2's full-forward worlds read at the 66 never-calibrated wells; "
                 "P <= 0.5 reads 'weak by design' (one-sided)",
        "label": "fresh (first attempt; not post-failure)",
        "n3f": {"id": pid or PENDING, "registration": rec or PENDING,
                "pinned": f"{X.N3F_ID} only (export {X.EXT_DIR}, window {list(S.WINDOW)})"},
        "ext_files": ext_files, "kept_ids": kept,
        "product_use": product_use_block(),
        "design_time_look": {"unit": X.EXT_LOOK_UNIT, "scope": X.EXT_LOOK_SCOPE,
                             "purpose": X.EXT_LOOK_PURPOSE,
                             "logged": X.look_logged(prereg_dir, ext_dir) or PENDING,
                             "rule": "logged by the lead (ext_fv.py log-look) before the "
                                     "dry run, the first step that reads the derived values"},
        "builder_look": logged_look(BUILDER_LOOK_UNIT, BUILDER_LOOK_PURPOSE, prereg_dir)
        or PENDING,
        "worlds": {"source": V2.ID, "note": "the same batch forwards, read at the external "
                                            "wells' cells and layers; no extra forward",
                   "k_truth": V2.K_TRUTH, "truths": list(O.TAGS)},
        "truth_world": {
            "window": "obs = ZR_ext + D_T + [h_T(world k) - h_T(baseline)] + e",
            "pre_origin": "y = registered forward hindcast + datum_m + D_T + e (2012-22, every "
                          "month observed: optimistic)",
            "D_T": "(h_T - h_ZR0) minus its mean over all 2012-22 months (no observed-month "
                   "mask exists at these wells without a new look)",
            "e": f"one stationary AR(1) path 2012-01..2027-10 per world (stream "
                 f"{X.STREAM_NOISE}); sd = wells.csv residual_sd_m; lag-1 = county median "
                 "of v1's noise inputs (overall median for Nantou)"},
        "rule": {"primary": "N3F's: equal-station RMSE ratio to the baseline <= 1.0 AND pooled "
                            "coverage in [0.85, 0.95] (>= 30 wells x >= 6 months), the "
                            "baseline being each world's own pre-origin month-of-year "
                            "climatology (synthetic; the real one is not reused)",
                 "report_only": "the same rule against the registered real-record baseline"},
        "readings": {"threshold": X.P_WEAK, "rule": "P <= 0.5 -> 'weak_by_design'; otherwise "
                     "'not_shown_weak'; one-sided; applied once to the point estimate (the "
                     "1.645 SE <= 0.07 check is non-binding at this K; P - 0.5 is reported in "
                     "Monte Carlo SE units)", "wording": X.WORDING},
        "null_world": nul,
        "gates": {"E": "registration, frozen code, N3F export files as registered, kept "
                       "wells, v2 world files and batch metadata, hindcast identity (0.0 m) "
                       "at the external cells, the declared null tolerance recomputed, "
                       "g0_e02 = twin.validation.evaluate on two self-check worlds",
                  "N": "null-world false pass <= tolerance",
                  "MC": {"rule": f"1.645 SE <= {V2.MC_HALF_WIDTH}",
                         "note": V2.mc_note(len(O.TAGS), X.K_TRUTH)}},
        "page_mapping": {"before_scoring": "no change",
                         "weak_by_design": "E22 gains the interpretation line: a fail would "
                                           "not refute the forecast at the never-calibrated "
                                           "wells",
                         "not_shown_weak": "E22 gains 'not shown weak'; no other reading",
                         "not_evaluable": "no line",
                         "never": "no S; E08 and C02 unchanged"},
        "attempt_policy": {"scored_runs": "scored once (report.json, mode 'x') after v2's "
                                          "forwards", "engineering_retries": "free while gate "
                                                                            "E fails, logged",
                           "exploratory_followups_max": 1, "thresholds_frozen": True,
                           "report": "every attempt"},
        "priors": "The external wells' frozen forecast must beat each well's own climatology; "
                  "a right-kind truth differs from ZR by D_T and its pumping response. "
                  "Planners put 'evaluable' at about 0.8; no direction is claimed.",
        "disclosures": {"builder": BUILDER_LOOKS,
                        "availability_risk": "N3F's own: the external wells may not report "
                                             "in the window; this study assumes they all do",
                        "after_the_fact_known": AFTER_THE_FACT_KNOWN,
                        "tests": _disclosure_tests()},
        "primary_model": {"forecast": "N3F's registered external predictions (forward "
                                      "cf2c7154)", "truths": list(O.TAGS)},
        "thresholds": {"P_weak": X.P_WEAK, "K_TRUTH": X.K_TRUTH, "K_NULL": X.K_NULL,
                       "null_tolerance": nul.get("tolerance"),
                       "mc_half_width_90": V2.MC_HALF_WIDTH},
        "interpretation": "A model-only operating characteristic of N3F's external-well "
                          "level rule. Nothing here is evidence about the aquifer.",
        "written_by": V2.PREREG,
        **_common_fields(frozen, inputs)}


def readiness_ext(d: dict, prereg_dir=None, ext_dir: str = X.EXT_DIR) -> list[str]:
    out = []
    if d["n3f"]["id"] == PENDING:
        out.append("no N3F external-well registration in the registry")
    elif score_looks({d["n3f"]["id"]}, prereg_dir):
        out.append("N3F has a score look: too late")
    for n, h in d["ext_files"].items():
        if h == PENDING or _sha(f"{ext_dir}/{n}") != h:
            out.append(f"N3F export {n} missing or differs from its registered sha256")
    if d["null_world"].get("tolerance") is None:
        out.append("ext null tolerance not computed (needs the logged look and N3F's export)")
    if d["design_time_look"]["logged"] == PENDING:
        out.append(f"design-time look {X.EXT_LOOK_UNIT} not logged (ext_fv.py log-look)")
    if d["product_use"].get("allowed") is not True:
        out.append(f"{X.PRODUCT_USE} is not an allowed use of {X.PRODUCT} (MANIFEST)")
    for p in (V2.ARMS_DIR, X.REPORT,
              *[V2.batch_npz(t, b) for t in O.TAGS for b in range(V2.PLAN.n_batch)]):
        if O.rp(p).exists():
            out.append(f"{p} exists before the declaration (worlds already drawn)")
    out += [f"frozen code {p} missing" for p, h in d["frozen_code"].items() if h == MISSING]
    out += [f"input {p} missing" for p in d["inputs"] if not O.rp(p).exists()]
    if index_record(X.ID, prereg_dir):
        out.append(f"{X.ID} is already registered")
    return out + builder_look_problems(prereg_dir)


# ----------------------------------------------------------------------------- stat
def doc_stat(prereg_dir=None) -> dict:
    man = _json(f"{ST.HYB}/manifest.json") or {}
    i1 = index_record(ST.I1_ID, prereg_dir)
    frozen = {p: _sha(p) for p in ST.FROZEN}
    inputs = [*ST.FROZEN, V2.PREREG, f"{ST.HYB}/manifest.json", f"{ST.HYB}/protocol.json",
              f"results/dataprep/preregistration/{ST.I1_ID}.json"]
    return {
        "id": ST.ID, "wp": ST.WP, "kind": "declaration",
        "title": "Synthetic decidability of the statistical layer's registered test (E04) in "
                 "AR(1) worlds, with I1.hybrid_tf T1/T2 report-only; P <= 0.5 reads 'weak by "
                 "design' (one-sided)",
        "label": "fresh (first attempt; not post-failure)",
        "e04": {"manifest": f"{ST.HYB}/manifest.json", "protocol": f"{ST.HYB}/protocol.json",
                "manifest_sha256": _sha(f"{ST.HYB}/manifest.json"),
                "protocol_sha256": _sha(f"{ST.HYB}/protocol.json"),
                "registered_on": man.get("registered_on", PENDING),
                "stations": man.get("stations", PENDING), "origins": man.get("origins", PENDING),
                "layout_used": {"origins": ST.E04_ORIGINS, "counties": ST.E04_COUNTIES}},
        "i1": {"id": ST.I1_ID, "registration": i1 or PENDING},
        "looks": {"builder": logged_look(BUILDER_LOOK_UNIT, BUILDER_LOOK_PURPOSE, prereg_dir)
                  or PENDING,
                  "stat_preview": logged_look(STAT_PREVIEW_UNIT, STAT_PREVIEW_PURPOSE,
                                              prereg_dir) or PENDING},
        "design": {"rho_grid": list(ST.RHO_GRID), "share_grid": list(ST.SHARE_GRID),
                   "cells": ST.cells(), "K_STAT": ST.K_STAT, "K_T1": ST.K_T1,
                   "seed": ST.SEED, "smoke_seed": ST.SMOKE_SEED,
                   "streams": {"e04": ST.STREAM_E04, "t1": ST.STREAM_T1,
                               "selfcheck": ST.STREAM_SELFCHECK},
                   "axis": "2012-01..2027-10; window 2026-11..2027-10; fitted months before",
                   "driver_rho_t1": ST.DRIVER_RHO},
        "truth_world": "anomaly = sqrt(1 - c) x well AR(1) + sqrt(c) x county AR(1), same "
                       "rho, unit variance; every month observed except E04's registered "
                       "origins (8 wells end 2026-07, 6 end 2026-08)",
        "rule": {"e04": "twin.hybrid.register's arithmetic re-run per world; E02's evaluation "
                        "(ratio <= 1.0 AND pooled coverage in [0.85, 0.95], >= 100 stations "
                        "x >= 6 months)",
                 "t1_t2": "REPORT-ONLY: i1.hybrid_tf.scorer.decide on one-step forecasts "
                          "from the frozen model code with null drivers; T2 ratio to "
                          "climatology"},
        "readings": {"threshold": ST.P_WEAK, "P": "equal-weight mean over the nine cells",
                     "rule": "P <= 0.5 -> 'weak_by_design'; otherwise 'not_shown_weak'; "
                             "one-sided; applied once (the 1.645 SE <= 0.07 check is "
                             "non-binding at this K; P - 0.5 is reported in SE units)",
                     "wording": ST.WORDING, "t1_t2": "no reading"},
        "gates": {"E": "g0_e02 = twin.validation.evaluate on two self-check worlds; the "
                       f"simulation recovers rho (cell 0.86/0.0, first {ST.N_RECOVERY} "
                       f"worlds, mean fitted rho within {ST.RHO_RECOVERY_TOL})",
                  "MC": {"rule": f"1.645 SE <= {V2.MC_HALF_WIDTH}",
                         "note": V2.mc_note(len(ST.cells()), ST.K_STAT)}},
        "page_mapping": {"before_scoring": "no change",
                         "weak_by_design": "E04 gains the interpretation line: a 2028 fail "
                                           "would not refute the statistical layer "
                                           "(the line states 'previewed at K = 8 before "
                                           "registration')",
                         "not_shown_weak": "E04 gains 'not shown weak' (with 'previewed at "
                                           "K = 8 before registration'); no other reading",
                         "not_evaluable": "no line",
                         "never": "no S; T1/T2 get no page line; the 1.8 persistence weight "
                                  "is never cited as support"},
        "attempt_policy": {"scored_runs": "scored once (report.json, mode 'x')",
                           "engineering_retries": "free while gate E fails, logged",
                           "exploratory_followups_max": 1, "thresholds_frozen": True,
                           "report": "every attempt"},
        "priors": "With persistence as assumed, the layer beats climatology in expectation "
                  "at leads 2-13, but its constant +-1.645 sd interval over-covers early "
                  "leads, so pooled coverage may exceed 0.95; common shocks widen both. No "
                  "direction is claimed.",
        "disclosures": {
            "grid": "RHO_GRID is the round-7 judge's grid around the E04 fit's median 0.86; "
                    "the E04 fit itself spans 2012..2026-09 (parameters.json "
                    "fitted_through_exclusive 2026-11-01), so it is not purely pre-origin. "
                    "The post-origin 1.8 persistence weight is not used (U7). SHARE_GRID "
                    "takes values of U1.HB.power.v1's registered ICC_month grid",
            "builder": BUILDER_LOOKS,
            "builder_smoke": "Before this declaration the builder ran the study once at "
                             "K = 8 per cell with the DECLARED seed (an engineering smoke, "
                             "before the smoke seed was separated): pooled P about 0.92, "
                             "cell rates 0.75-1.0, reading 'not_shown_weak'. The grid, "
                             "K_STAT, the share grid, the driver rho and every threshold were "
                             "written before that run and none was changed after it; later "
                             "smokes use SMOKE_SEED only. The seed is not changed now (that "
                             "would be a reroll); the K = 500 score's first 8 worlds per cell "
                             "are the previewed ones. The preview is carried in the report "
                             "(builder_preview) and in the E04 line, and the lead logs it "
                             f"as the look {STAT_PREVIEW_UNIT}",
            "optimism": "no gaps, drift, datum steps or trend; equal scale at every well; "
                        "anomalies exactly AR(1)",
            "tests": _disclosure_tests()},
        "primary_model": {"layer": "E04's registered statistical layer (procedure re-run per "
                                   "world)", "truth": "AR(1) anomalies"},
        "thresholds": {"P_weak": ST.P_WEAK, "K_STAT": ST.K_STAT,
                       "mc_half_width_90": V2.MC_HALF_WIDTH,
                       "rho_recovery_tol": ST.RHO_RECOVERY_TOL},
        "interpretation": "A model-free operating characteristic of E04's registered test "
                          "under its own assumption. Nothing here is evidence about the "
                          "aquifer.",
        "written_by": V2.PREREG,
        **_common_fields(frozen, inputs)}


def readiness_stat(d: dict, prereg_dir=None, now=None) -> list[str]:
    now = now or dt.datetime.now(dt.timezone.utc)
    out = []
    if score_looks({ST.I1_ID}, prereg_dir):
        out.append(f"{ST.I1_ID} has a score look: too late")
    if now >= dt.datetime.fromisoformat(ST.E04_SCORABLE_FROM).replace(tzinfo=dt.timezone.utc):
        out.append(f"E04's window has closed ({ST.E04_SCORABLE_FROM}): E04 may be scored; too "
                   "late")
    if d["e04"]["protocol_sha256"] == MISSING or d["e04"]["manifest_sha256"] == MISSING:
        out.append("E04 manifest/protocol missing")
    if d["i1"]["registration"] == PENDING:
        out.append(f"{ST.I1_ID} is not in the registry")
    if O.rp(ST.REPORT).exists():
        out.append(f"{ST.REPORT} exists before the declaration")
    out += [f"frozen code {p} missing" for p, h in d["frozen_code"].items() if h == MISSING]
    out += [f"input {p} missing" for p in d["inputs"] if not O.rp(p).exists()]
    if index_record(ST.ID, prereg_dir):
        out.append(f"{ST.ID} is already registered")
    return out + builder_look_problems(prereg_dir, stat=True)


# ----------------------------------------------------------------------------- CLI
def build(which: str, prereg_dir=None, replay_status=None,
          c14b_trigger=None) -> tuple[dict, list[str]]:
    if which == "v2":
        d = doc_v2(prereg_dir, replay_status, c14b_trigger)
        return d, readiness_v2(d, prereg_dir)
    if which == "ext":
        d = doc_ext(prereg_dir)
        return d, readiness_ext(d, prereg_dir)
    d = doc_stat(prereg_dir)
    return d, readiness_stat(d, prereg_dir)


def main(argv=None, prereg_dir=None, rows_dir=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--decl", choices=("v2", "ext", "stat"))
    ap.add_argument("--log-look", choices=("builder", "stat_preview"),
                    help="the lead, before any --write: log the builder's accidental look or "
                         "FV.stat's preview (real registry)")
    ap.add_argument("--station", action="append", default=None,
                    help="--log-look builder: a well whose row was printed (repeat)")
    ap.add_argument("--note", default="", help="--log-look builder: the printed values")
    ap.add_argument("--c14b-trigger", choices=FYM.C14B_TRIGGERS, default=None,
                    help="v2 only: the lead's explicit C14b trigger (either_county is the "
                         "cautious choice)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--validate", action="store_true", help="dry run (default)")
    g.add_argument("--write", action="store_true", help="register once (the lead)")
    ap.add_argument("--expect-self-sha", default=None)
    ap.add_argument("--replay-status", default=None,
                    help="v2 only: who, when, and the state of rank 11's replay forwards")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    if a.log_look:
        kw = {k: v for k, v in (("prereg_dir", prereg_dir), ("rows_dir", rows_dir)) if v}
        rec = log_builder_look(a.log_look, stations=a.station, note=a.note, **kw)
        print(f"logged look {a.log_look} at {rec.get('time')}")
        return 0
    if not a.decl:
        ap.error("--decl is required (or --log-look)")
    d, pend = build(a.decl, prereg_dir, a.replay_status, a.c14b_trigger)
    d = json.loads(json.dumps(d, default=O._jsonable))
    errs = prereg3.validate_prereg(d, prereg_dir=prereg_dir or prereg3.PREREG_DIR)
    if not a.quiet:
        print(json.dumps(d, indent=1, ensure_ascii=False, sort_keys=True))
    self_sha = O.sha256(Path(__file__))
    print("draft sha256:", sha_text(d))
    print("validate_prereg:", errs or "ok")
    print("blocking looks (declaration, no tests, no fresh units): none by construction")
    print("readiness:", pend or "ready to register")
    print("this script sha256:", self_sha)
    if not a.write:
        return 1 if errs else 0
    if a.expect_self_sha != self_sha:
        print("REFUSED: --write needs --expect-self-sha equal to this script's sha256")
        return 1
    if errs or pend:
        print("REFUSED:", errs + pend)
        return 1
    kw = {k: v for k, v in (("prereg_dir", prereg_dir), ("rows_dir", rows_dir)) if v}
    path, h = prereg3.write_prereg(d, **kw)
    print(f"registered {path} sha256 {h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
