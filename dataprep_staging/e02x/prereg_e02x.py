"""Declaration builders of round-8 rank 5 (harness.prereg3), plus the blocking probe:

  power  U1.ext_transfer_power.v1 (kind 'declaration'): ``draft_prereg_power(seed)``. The
         synthetic P-gate of E02x per county and window, registered BEFORE any computation;
         it also binds the code of the one logged design-time read (sdfit_e02x.py).
  probe  the prereg3 blocking probe on E02x's window units (E02x:ext_heads_<first>_<last>_
         wells60, the Yunlin 37 + Changhua 23 gate candidates, and the Nantou 6 report-only
         unit): looks.jsonl only, no head; writes results/twin_runs/e02x/probe_<window>.json
         once (the sd_fit_rel read refuses without a cleared probe).
  e02x   E02x.ext_transfer_2026-12.v1 (kind 'test'): ``draft_prereg_e02x(window)``, or the
         pre-declared fallback E02x.ext_transfer_2027-01.v1 (2027-01..2027-10).

``--validate`` prints the draft, its canonical sha256, prereg3.validate_prereg, the blocking
looks and readiness; nothing is written. ``--write`` is the LEAD's step only (refuses on any
missing slot, a failed power rule, a passed deadline or a wrong ``--expect-self-sha``); the
builder never ran it.

    python -I dataprep_staging/e02x/prereg_e02x.py power --validate --seed <int>
    python -I dataprep_staging/e02x/prereg_e02x.py power --write --seed <int> --expect-self-sha <sha>
    python -I dataprep_staging/e02x/prereg_e02x.py probe --window primary [--dry]
    python -I dataprep_staging/e02x/prereg_e02x.py e02x --validate --window primary
    python -I dataprep_staging/e02x/prereg_e02x.py e02x --write --window primary --expect-self-sha <sha>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent), str(HERE.parent / "e02d")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import build_artefact_e02x as BA  # noqa: E402
import e02xio as io  # noqa: E402
import e02xlib as L  # noqa: E402
import members_e02x as ME  # noqa: E402
import power_e02x as PW  # noqa: E402
import sdfit_e02x as SF  # noqa: E402
from harness import prereg3  # noqa: E402

NA = prereg3.not_applicable
THIS = f"{io.X}/prereg_e02x.py"
TESTS = "tests/test_e02x.py"
SCORER = f"{io.X}/score_e02x.py"
RUNNER = f"{io.X}/run_e02x.sh"
FROZEN_E02X = (SCORER, f"{io.X}/e02xlib.py", f"{io.X}/e02xio.py", f"{io.X}/wells_e02x.py",
               f"{io.X}/members_e02x.py", f"{io.X}/sdfit_e02x.py",
               f"{io.X}/build_artefact_e02x.py", f"{io.X}/power_e02x.py", RUNNER)
BOUND = ("dataprep_staging/e02d/e02dlib.py", "dataprep_staging/e02c/score_e02c.py",
         "dataprep_staging/gov_heads_window/heads_window.py",
         "dataprep_staging/harness/coverage3.py", "dataprep_staging/harness/coverage2.py",
         "dataprep_staging/harness/coverage.py")
CODE_REPORT_ONLY = ("hydrophysics/twin/prospective.py", "hydrophysics/twin/validation.py",
                    "hydrophysics/twin/challenge.py", "dataprep_staging/harness/prereg3.py")
ENGINEERING_RESTORE = ("a git-tracked module listed in code_at_registration_report_only may be "
                       "restored to its registered sha256 (git checkout of that blob) before "
                       "the scoring run as an engineering step, logged in the report; frozen "
                       "and bound code is never edited")
PROBE_DIR = SF.PROBE_DIR

FV_EXT_LOOK = ("FV.ext.v1 (2026-11-01T14:19:55Z, unit FV:ext_preorigin_derived_N3Fwells, "
               "purpose design_time_noise_inputs, scope heads:2012-01..2022-12): read N3F's "
               "registered export at these 66 wells, wells.csv residual_sd_m and datum_m and "
               "predictions.csv baseline_m and predicted_m (pre-origin derived), and ran FV v2's "
               "full-forward worlds at the external cells (N3F's level rule decidability). It "
               "used no window head; its noise scale is the absolute residual sd, not the "
               "transferred relative spread; nothing from it sets a rule here")
HISTORY = {
    "FV:ext": FV_EXT_LOOK,
    "N3F export": "export_baselines.py ext read the 66 wells' pre-origin heads "
                  "(L1_clean v3, datum_fit / baseline, sliced to 2022-12) to set each well's "
                  "datum, residual sd and climatology (N3F's registration)",
    "N1.cpu_response.v1": "score looks 2026-10-13 on N1:ext_wells (heads 2013-01..2022-12, "
                          "disjoint window)",
    "U1.decomp_power.v1 / DR": "the donor grid and groups are DR's pre-origin estimates at the "
                               "158 wells; the designers know the 2023 distal drift and the "
                               "E02d design; none enters a rule here beyond the registered grid",
    "round-5/6 diagnoses": "after-the-fact 2023-26 numbers at the 158 wells are known to the "
                           "designers; the 2023-26 heads at the 66 external wells were never "
                           "opened (heads/L1_clean v3 lists post-origin records at the 158 "
                           "wells only)"}


_AB = L.arithmetic_bounds()
CHANGHUA_ARITH = (
    f"Changhua is REPORT-ONLY BY ARITHMETIC (lead ruling B1; grid and floor unchanged): with "
    f"{_AB['Changhua']['wells']} external wells, the {L.MIN_WELLS_COUNTY}-well floor and whole-"
    f"window dropout {list(L.DROPOUT)}, P(>= {L.MIN_WELLS_COUNTY} report) = "
    f"{_AB['Changhua']['p_at_least_floor_by_dropout']} (binomial, constants only), so the "
    f"power rule's minimum is <= {_AB['Changhua']['power_upper_bound']:.3f} < {L.POWER_MIN}: "
    "Changhua cannot pass by construction. Its power cells are still run and reported, "
    "report-only; its E02x verdict is report-only, never mapped")
YUNLIN_BOUND = (
    f"Yunlin: P(>= {L.MIN_WELLS_COUNTY} of {_AB['Yunlin']['wells']} report) = "
    f"{_AB['Yunlin']['p_at_least_floor_by_dropout']}, so its power is at most "
    f"{_AB['Yunlin']['power_upper_bound']:.3f} x the conditional power at dropout "
    f"{max(L.DROPOUT):g}; passing the {L.POWER_MIN} rule needs a conditional power of at "
    f"least {L.POWER_MIN / _AB['Yunlin']['power_upper_bound']:.3f} there")
NANTOU_DISCLOSURE = (
    "Nantou's donors are the E02 wells of the same raw column zone in BOTH counties: a "
    "mixed-county median removes neither county's monthly shock, so under-coverage is expected "
    "at the Nantou wells; report-only (R8.3), never pooled, never mapped")
FALLBACK_DEADLINE_SOURCE = (
    "the fallback window 2027-01..2027-10 (kept by the lead) has its registration deadline "
    "2026-12-31T16:00Z SET BY THE LEAD by analogy to N3F's 'the day before the first scored "
    "month, 16:00Z'; it is not from the round-8 plan")
MEMBERS_DISCLOSURE = (
    "the 36 members at the external cells are read (lead decision Q1: 'extract e02d', 0 GPU) "
    "from E02d's own run files (results/twin_forward/e02d_members), which hold the full field "
    "and whose sha256 E02d's members_gate.json (registered sha bae7b24d) records. Gate (a), "
    "donors vs members_wells.npz within 1e-3 m, is then an IDENTITY check on the same files, "
    "not a reproduction; gate (b), the member mean at the 66 external cells vs the frozen page "
    "run (cf2c7154) within 1e-3 m, is the real check. members_gate.json records the source")


def _tau_note(drp: Path) -> str:
    """Whether E02d's tau rule equals the plan's unconditional formula here (both counties
    triggered DR's 10 % rule): reads DR's report decisions only."""
    if not drp.is_file():
        return "PENDING (DR report)"
    dec = json.loads(drp.read_text(encoding="utf-8")).get("decisions") or {}
    trig = {c: bool((dec.get(c) or {}).get("include_drift")) for c in L.GATED}
    if all(trig.values()):
        return ("E02d's wording is kept; DR's 10 % rule triggered in both counties, so the "
                "band equals the plan's sqrt(sd_fit_rel^2 + sigma_ens_rel^2 + tau_c^2 dt^2) "
                "numerically")
    return f"DR's 10 % rule triggered: {trig}; the band omits tau where it did not (E02d's rule)"


# ----------------------------------------------------------------------------- helpers
def _rulings() -> dict:
    p = io.repo_path(L.RULINGS)
    if not p.is_file():
        return {"status": "PENDING", "path": L.RULINGS}
    r = json.loads(p.read_text(encoding="utf-8"))
    hit = [x for x in r.get("rulings", []) if x.get("id") == L.RULING_ID]
    return {"status": "ok" if hit else "missing", "file": io.rec(p),
            "ruling": hit[0] if hit else None}


def _well_lists(ext) -> dict:
    out = {}
    for c in (*L.GATED, *L.REPORT_ONLY):
        ids = sorted(ext.loc[ext["county"] == c, "station_id"].astype(str))
        out[c] = {"n": len(ids), "ids": ids, "sha256": io.sha_lines(ids)}
    return out


def _gated_ids(ext) -> list[str]:
    return sorted(ext.loc[ext["county"].isin(L.GATED), "station_id"].astype(str))


def _nantou_ids(ext) -> list[str]:
    return sorted(ext.loc[ext["county"].isin(L.REPORT_ONLY), "station_id"].astype(str))


def _ext_unit(win: str, ids: list[str]) -> dict:
    w = L.WINDOWS[win]
    return {L.unit_id(win, len(ids)): {
        "description": f"monthly heads {w['first']}..{w['last']} (flags 0-1) at N3F's "
                       f"never-calibrated external wells of Yunlin and Changhua ({len(ids)}: "
                       "the gate candidates of E02x); a new unit id inside N3F's unit "
                       f"{L.N3F_UNIT}",
        "ids": ids, "window": f"{w['first']}..{w['last']}"}}


def _nantou_unit(win: str, ids: list[str]) -> dict:
    w = L.WINDOWS[win]
    return {L.nantou_unit_id(win, len(ids)): {
        "description": f"monthly heads {w['first']}..{w['last']} (flags 0-1) at N3F's "
                       f"{len(ids)} Nantou external wells: report-only under R8.3, never "
                       "pooled with a county gate",
        "ids": ids, "window": f"{w['first']}..{w['last']}"}}


def _pre_unit(ext) -> dict:
    ids = sorted(ext["station_id"].astype(str))
    return {L.PRE_UNIT: {"description": "pre-origin monthly heads 2012-01..2022-12 at N3F's "
                                        f"{len(ids)} external wells (the sd_fit_rel read; "
                                        "design-time)",
                         "ids": ids, "window": "2012-01..2022-12"}}


def check_blocking(tests: list[dict], prereg_dir=None) -> list:
    """prereg3's blocking looks for new tests whose external units are not yet known."""
    pdir = Path(prereg_dir) if prereg_dir else io.prereg_dir_default()
    idx = prereg3.membership_index(pdir)
    for t in tests:
        for u, v in (t.get("external_units") or {}).items():
            if not idx.known(u):
                idx.members[idx.canon(u)] = prereg3._declared_members(v)
    return prereg3.blocking_looks(tests, None, pdir, idx, test_time=io.now())


def unverifiable(tests: list[dict], prereg_dir=None) -> list:
    pdir = Path(prereg_dir) if prereg_dir else io.prereg_dir_default()
    idx = prereg3.membership_index(pdir)
    for t in tests:
        for u, v in (t.get("external_units") or {}).items():
            if not idx.known(u):
                idx.members[idx.canon(u)] = prereg3._declared_members(v)
    return prereg3.unverifiable_looks(tests, None, pdir, idx)


# ----------------------------------------------------------------------------- power
def draft_prereg_power(seed: int, prereg_dir=None) -> dict:
    """U1.ext_transfer_power.v1 (declaration). ``seed`` is required (no default)."""
    u1, u1_sha, ext, prov = PW.load_inputs(prereg_dir)
    design = PW.design_spec(u1, ext, seed, u1_sha, prov)
    n = {c: len(PW.cells(design["counties"][c])) for c in L.GATED}
    return {
        "id": L.POWER_ID, "wp": L.POWER_WP, "kind": "declaration",
        "title": "Synthetic P-gate of E02x's departure-transfer relative band test at N3F's "
                 "never-calibrated external wells, per county, primary window 2026-12..2027-10 "
                 "(11 months) and fallback 2027-01..2027-10 (10 months); U1.decomp_power.v1's "
                 "grid plus departure-estimation error",
        "why": "E02x is registered only if its exact scorer is decisive by design at the "
               f"external wells: false-fail <= {L.FALSE_FAIL_MAX} at a true {L.FF_AT} and power "
               f">= {L.POWER_MIN} against a {L.POWER_AT} interval, per county, over the "
               "registered DR-bounded grid plus the transfer's own error (round-8 rank 5). "
               "Registered before any computation.",
        "design": design, "n_cells": n,
        "well_list": _well_lists(ext),
        "derivation": {
            "grid": f"a_s, a_w, phi, ext dropout = {L.U1D_ID}'s registered grid per county "
                    "(DR-bounded, unchanged)",
            "departure_estimation_error": "(i) the donor median's sampling error, simulated: "
                                          "the 158 donors are drawn with their own parts and "
                                          f"drop out (whole window) with probability "
                                          f"{list(L.DONOR_DROPOUT)}; < {L.MIN_DONORS} donors "
                                          "in a group leaves its external wells unscored; "
                                          "(ii) an external-only group-month shock of share "
                                          "a_x in {0, the county's top a_s}",
            "external_scale": "the donor scale quantiles (the external sd_fit_rel is read only "
                              "after this registration)",
            "groups": "each external well's DR group (zone partition with DR's merges) and "
                      "sub-group (layer group) from its cell and twin layer (wells_e02x."
                      "ext_table, metadata only)",
            "a_s_disclosure": "out-of-sample sub-group or transfer shocks may exceed the bound; "
                              "a 2028 fail is then reported with its by-zone coverage "
                              "(report-only)"},
        "windows": {w: {"first": L.WINDOWS[w]["first"], "last": L.WINDOWS[w]["last"],
                        "months": L.WINDOWS[w]["months"],
                        "registration_deadline": L.WINDOWS[w]["deadline"].isoformat(),
                        "role": L.WINDOWS[w]["role"]} for w in L.WINDOWS},
        "decision_to_registration": {
            "per_window": "each window has its own decision (its own power run inside this "
                          "study: 11 and 10 months)",
            "Yunlin": "passes -> E02x may be registered for that window with Yunlin gated; "
                      "fails -> E02x is NOT registered for that window and C14f-r is recorded N "
                      "('undecidable by design at the external wells')",
            "Changhua": CHANGHUA_ARITH,
            "Nantou": "report-only under every outcome (R8.3: no pooling)",
            "member_gate": "independently: if the member gate fails at both declared codes, "
                           "E02x is not registered (no fallback interval)"},
        "sd_fit_rel_read": {
            "when": "after this registration, the power run and the cleared blocking probe on "
                    "the window unit; before E02x's artefact and registration",
            "rule": "e_i = r_i - median of r over i's donors that month (>= 5); sd_fit_rel = "
                    f"RMS of e over >= {L.MIN_FIT_MONTHS_SD} months else the well is not "
                    "scored; t_centre = mean decimal year of i's observed pre-origin months",
            "reads": list(SF.READS), "look": {
                "units": [L.PRE_UNIT, L.DR_UNIT], "scope": L.PRE_SCOPE,
                "purpose": "design_time_estimation", "prereg_id": L.POWER_ID},
            "code": "frozen_code below (sdfit_e02x.py refuses unless these sha256 match)",
            "computes_no": "ICC, coverage, drift or any score"},
        "arithmetic_bounds": design["arithmetic_bounds"],
        "changhua_report_only_by_arithmetic": CHANGHUA_ARITH,
        "yunlin_bound": YUNLIN_BOUND,
        "calibration_disclosure": PW.CALIBRATION_DISCLOSURE,
        "self_checks": "vectorised verdicts equal E02d's county_verdict on the same replicates; "
                       "a shock shared by a group's donors and external wells cancels exactly; "
                       "a failure is an engineering failure that gates nothing",
        "smoke_disclosure": "the builder ran self_check and a one-cell timing only on "
                            "power_e02x.synthetic_inputs (made-up scales and groups) and in the "
                            "test suite; the real design was never computed",
        "external_units": _pre_unit(ext),
        "history_disclosure": HISTORY,
        "code_sha256": {f: io.sha256(io.repo_path(f)) for f in PW.CODE_FILES},
        "frozen_code": {f: io.sha256(io.repo_path(f)) for f in SF.FROZEN},
        "run": {"command": f"python -I {io.X}/power_e02x.py run --seed {seed}",
                "result": f"{PW.RESULT_DIR}/result.json", "runs_once": True,
                "cost": "CPU only, about 45-90 minutes (2 windows x 2 counties x "
                        f"{n['Yunlin']} cells x {PW.REPLICATES} replicates); 0 GPU-h"},
        "attempt_policy": "run once under this registration; an engineering failure (self-check) "
                          "gates nothing and needs a new id after a fix to bound code",
        "zones": NA("synthetic draws"), "priors": NA("synthetic draws"),
        "burn_in": NA("synthetic draws"), "folds": NA("synthetic draws"),
        "tests": NA("declaration of a synthetic study; no observation is scored"),
        "fresh_units": NA("no observation is involved; the read's unit is design-time"),
        "thresholds": design["decision"], "gate_scores": NA("no promoting gate"),
        "primary_model": NA("no model: synthetic residuals and E02x's scorer only"),
        "inputs": sorted({f"results/dataprep/preregistration/{L.U1D_ID}.json",
                          f"results/dataprep/preregistration/{L.N3F_ID}.json",
                          f"{L.N3F_DIR}/wells.csv", f"{L.DR_DIR}/wells_e02d.csv", L.FORWARD,
                          THIS, TESTS, *PW.CODE_FILES, *SF.FROZEN}),
        "interpretation": "Decidability of E02x by design at the external wells; it estimates "
                          "nothing from heads.",
        "written_by": THIS}


def power_problems(doc: dict, expect_self_sha, prereg_dir=None) -> list[str]:
    bad = []
    if io.index_record(L.U1D_ID, prereg_dir) is None:
        bad.append(f"{L.U1D_ID} is not registered")
    if io.index_record(L.N3F_ID, prereg_dir) is None:
        bad.append(f"{L.N3F_ID} is not registered")
    if io.repo_path(f"{PW.RESULT_DIR}/result.json").exists():
        bad.append("the power result already exists")
    if io.repo_path(f"{SF.OUT_DIR}/report.json").exists():
        bad.append("the sd_fit_rel read already happened: it must follow this registration")
    if io.now() >= L.WINDOWS["primary"]["deadline"] and io.now() >= L.WINDOWS["fallback"][
            "deadline"]:
        bad.append("both E02x windows' registration deadlines have passed")
    self_sha = io.sha256(Path(__file__))
    if expect_self_sha != self_sha:
        bad.append(f"--expect-self-sha {expect_self_sha!r} != {self_sha}")
    return bad


# ----------------------------------------------------------------------------- probe
def probe_tests(win: str, prereg_dir=None) -> list[dict]:
    import wells_e02x as WX

    ext, _ = WX.ext_table(prereg_dir)
    g, n = _gated_ids(ext), _nantou_ids(ext)
    if len(g) != L.N_GATED or len(n) != L.EXPECTED_EXT["Nantou"]:
        raise io.EngineeringFailure(f"{len(g)} gated / {len(n)} Nantou wells")
    return [{"test_id": L.test_id(win), "data_scope": L.scope(win),
             "fresh_units": [L.unit_id(win, len(g))], "external_units": _ext_unit(win, g)},
            {"test_id": L.nantou_test_id(win), "data_scope": L.scope(win),
             "fresh_units": [L.nantou_unit_id(win, len(n))],
             "external_units": _nantou_unit(win, n)}]


def run_probe(win: str, prereg_dir=None, dry: bool = False) -> dict:
    """The prereg3 blocking probe (looks.jsonl only, no head)."""
    pdir = Path(prereg_dir) if prereg_dir else io.prereg_dir_default()
    tests = probe_tests(win, pdir)
    block = check_blocking(tests, pdir)
    rec = {"window": win, "unit": tests[0]["fresh_units"][0],
           "units": [t["fresh_units"][0] for t in tests], "time": io.utc_stamp(),
           "blocking": block, "unverifiable": unverifiable(tests, pdir),
           "looks_sha256": io.sha256(pdir / "looks.jsonl") if (pdir / "looks.jsonl").is_file()
           else None,
           "rule": prereg3.BLOCKING_RULE}
    if not dry:
        rec["written"] = io.write_once(io.repo_path(f"{PROBE_DIR}/probe_{win}.json"), rec)
    return rec


# ----------------------------------------------------------------------------- E02x
def _slots(win: str, prereg_dir=None) -> dict:
    """Readiness of every upstream artefact (no observation is read)."""
    out = {}
    pr = io.repo_path(f"{PW.RESULT_DIR}/result.json")
    if pr.is_file():
        r = json.loads(pr.read_text(encoding="utf-8"))
        out["power"] = {"status": "ok" if not (r.get("smoke") or r.get("engineering_failure"))
                        else "invalid", "result": io.rec(pr),
                        "declaration": r.get("declaration"),
                        "decisions": (r.get("decisions") or {}).get(win),
                        "registration": (r.get("registration") or {}).get(win)}
    else:
        out["power"] = {"status": "PENDING"}
    gp = io.repo_path(ME.GATE_JSON)
    out["members"] = ({"status": "ok" if json.loads(gp.read_text()).get("ok") else "failed",
                       "gate": io.rec(gp), "npz": io.rec(ME.MEMBERS_NPZ)}
                      if gp.is_file() else {"status": "PENDING"})
    sp = io.repo_path(f"{SF.OUT_DIR}/report.json")
    if sp.is_file():
        s = json.loads(sp.read_text(encoding="utf-8"))
        out["sd_fit_rel"] = {"status": "ok", "report": io.rec(sp),
                             "wells": io.rec(f"{SF.OUT_DIR}/wells_e02x.csv"),
                             "declaration": s.get("declaration"), "look": s.get("look"),
                             "wells_without_sd_fit_rel": s.get("wells_without_sd_fit_rel")}
    else:
        out["sd_fit_rel"] = {"status": "PENDING"}
    ap = io.repo_path(f"{BA.OUT}/summary.json")
    if ap.is_file():
        s = json.loads(ap.read_text(encoding="utf-8"))
        files = {n: io.rec(f"{BA.OUT}/{n}") for n in s.get("files", {})}
        bad = [n for n, h in s.get("files", {}).items() if files[n]["sha256"] != h]
        out["artefact"] = {"status": "ok" if not bad else "mismatch", "summary": io.rec(ap),
                           "files": files, "per_window": s.get("per_window"),
                           "dr_report_sha256": ((s.get("inputs") or {}).get("dr_report") or {})
                           .get("sha256")}
    else:
        out["artefact"] = {"status": "PENDING"}
    pb = io.repo_path(f"{PROBE_DIR}/probe_{win}.json")
    out["probe"] = (json.loads(pb.read_text(encoding="utf-8")) | {"record": io.rec(pb)}
                    if pb.is_file() else {"status": "PENDING"})
    drp = io.repo_path(f"{L.DR_DIR}/report.json")
    out["dr"] = io.rec(drp) if drp.is_file() else {"status": "PENDING"}
    out["tau_note"] = _tau_note(drp)
    out["rulings"] = _rulings()
    return out


def _roles(s: dict) -> dict:
    reg = (s["power"].get("registration") or {}) if s["power"]["status"] == "ok" else {}
    roles = reg.get("roles") or {}
    ch = roles.get("Changhua", "report-only")
    if L.arithmetic_bounds()["Changhua"]["report_only_by_arithmetic"]:
        ch = "report-only (by arithmetic: " + L.arithmetic_bounds()["Changhua"]["bound"] + ")"
    return {"Yunlin": roles.get("Yunlin", "PENDING (power result)"), "Changhua": ch,
            "Nantou": "report-only"}


WORDING = {
    "pass": "{county}: at wells never used in calibration, the per-well spread of heads around "
            "the departure transferred from the calibration wells passed its registered gate "
            "on {first}..{last} (coverage {cov:.2f}; {test}, prospective)",
    "fail": "{county}: at wells never used in calibration, the transferred per-well band "
            "failed its registered gate on {first}..{last} (coverage {cov:.2f}, {side}); the "
            "transferred per-well wording is withdrawn for {county}",
    "not_evaluable": "{county}: transferred per-well band at wells never used in calibration "
                     "undecided ({why})",
    "report_only": "{county}: report-only (not gated: {why}); coverage {cov:.2f} on "
                   "{first}..{last}",
    "never": ["calibrated", "校準", "the full interval passed", "pooled"]}


def draft_prereg_e02x(window: str = "primary", prereg_dir=None) -> dict:
    """E02x.ext_transfer_<first>.v1 (test) for ``window`` in {'primary', 'fallback'}."""
    import wells_e02x as WX

    if window not in L.WINDOWS:
        raise ValueError(f"window {window!r}")
    w = L.WINDOWS[window]
    ext, prov = WX.ext_table(prereg_dir)
    g, n = _gated_ids(ext), _nantou_ids(ext)
    s = _slots(window, prereg_dir)
    roles = _roles(s)
    n3f, n3f_sha = WX.n3f_registration(prereg_dir)
    unit, nunit = _ext_unit(window, g), _nantou_unit(window, n)
    gate = (f"per county: wells with >= {L.MIN_MONTHS} scored months; undecided below "
            f"{L.MIN_WELLS_COUNTY} (Changhua: {L.MIN_WELLS_COUNTY} of "
            f"{L.EXPECTED_EXT['Changhua']}); harness.coverage3.coverage_gate (gate-3, method "
            f"{L.D.GATE_METHOD}, clusters = wells, target {L.TARGET}, level {L.GATE_LEVEL}) AND "
            f"point coverage within {list(L.POINT_WINDOW)} (E02d's gate, e02dlib."
            "county_verdict)")
    metric = {
        "donor_rows": f"E02's step-(5) rows (results/twin/prospective_2026-11, run in "
                      f"E02/E02c's session) restricted to {w['first']}..{w['last']}, verified "
                      "against E02's registered files (score_e02c.load_e02_step5)",
        "ext_rows": f"N3F's step-(5) rows (E02's scorer on {L.N3F_DIR}, run by N3F's own "
                    f"scorer in the same session) restricted to {w['first']}..{w['last']}, "
                    "verified against N3F's registered export (predicted/lower/upper/baseline "
                    "to 1e-9 m; validation.evaluate reproduced where N3F's report exists)",
        "residual": "r = observed_m - predicted_m at donors (E02's centre) and external wells "
                    "(N3F's centre: the frozen forecast plus the well's own pre-origin datum)",
        "departure": f"D_g[t] = median of r over the E02 wells of the external well's DR group "
                     f"g reporting in month t (>= {L.MIN_DONORS}, E02d min_others; else the "
                     "row is not scored); Nantou: the E02 wells of the same raw column zone in "
                     "either county",
        "relative": "e = r_ext - D_g (the centre is the frozen forecast plus the transferred "
                    "zone-median departure)",
        "hit": f"|e| <= {L.Z90} * sigma_idio (sigma_idio_{window}.csv, frozen)",
        "gate": gate,
        "report_only": ["signed misses (below / above)", "coverage by zone (prox / mid / "
                        "dist) and by DR group, each county", "month-cluster gate-3 "
                        "('uncalibrated, k < 20')", "Nantou's coverage and misses",
                        "Changhua's verdict if Changhua is report-only"]}
    test = {
        "test_id": L.test_id(window), "data_scope": L.scope(window),
        "fresh_units": [next(iter(unit))], "external_units": unit,
        "label": "prospective: the window's heads did not exist at registration",
        "claim": "C14f-r (new ledger row): the per-well spread of heads around the departure "
                 "transferred from the 158 calibration wells, at N3F's never-calibrated "
                 f"external wells, {w['first']}..{w['last']}",
        "what": "Does the frozen transferred relative 90 % band (sigma_idio) cover the per-well "
                "part of the head error at wells never used in calibration, per county, under "
                "a cluster-aware gate?",
        "metric": metric, "role_by_county": {c: roles[c] for c in L.GATED},
        "decision_rule": {
            "pass": "the county's gate passes -> C14f-r S (prospective) for that county",
            "fail": "the gate fails -> the transferred per-well wording is withdrawn for that "
                    "county",
            "not_evaluable": f"< {L.MIN_WELLS_COUNTY} wells with >= {L.MIN_MONTHS} months, no "
                             "window product passing the GOV precondition, or no N3F step-(5) "
                             "rows",
            "per_county": "Yunlin decides only Yunlin wording, Changhua only Changhua and only "
                          "if gated; a report-only county's verdict is reported, never mapped"}}
    ntest = {
        "test_id": L.nantou_test_id(window), "data_scope": L.scope(window),
        "fresh_units": [next(iter(nunit))], "external_units": nunit,
        "label": "report-only (R8.3): scored once with the gated test, never a gate",
        "claim": "none (report-only)",
        "metric": {"as": "the gated test's metric, donors = the E02 wells of the same raw "
                         "column zone in either county"},
        "decision_rule": {"report_only": "coverage, signed misses and the gate's verdict are "
                                         "reported; never pooled into a county gate, never "
                                         "page-mapped"}}
    frozen = {f: io.sha256(io.repo_path(f)) for f in FROZEN_E02X}
    return {
        "id": L.prereg_id(window), "wp": L.E02X_WP, "kind": "test",
        "title": f"E02x: departure-transfer relative (per-well) head-band test at N3F's "
                 f"never-calibrated external wells, per county, {w['first']}..{w['last']}, "
                 "frozen forward cf2c7154",
        "why": "Every per-well band test so far scores the 158 calibration wells. At wells never "
               "used to fit the page model, the shared part can be removed only by transferring "
               "the calibration wells' departure; E02x tests whether the frozen per-well band "
               "then covers, the strongest out-of-sample band test held data allow (round-8 "
               "rank 5). Decidability is registered first (U1.ext_transfer_power.v1).",
        "tests": [test, ntest],
        "window": {"name": window, **{k: w[k] for k in ("first", "last", "months", "role",
                                                       "not_before", "no_later_than")},
                   "registration_deadline": w["deadline"].isoformat(),
                   "fallback": "E02x.ext_transfer_2027-01.v1 (2027-01..2027-10) is registered "
                               "only if this primary is not, before 2026-12-31T16:00Z, with "
                               "its own power decision" if window == "primary" else
                               "used because the primary was not registered",
                   "fallback_deadline_source": FALLBACK_DEADLINE_SOURCE},
        "interval": {
            "formula": "sigma_idio = sqrt(sd_fit_rel^2 + sigma_ens_rel^2 [+ tau_c^2 dt^2 where "
                       "DR's 10 % rule triggered]) (E02d's wording)",
            "tau_note": s["tau_note"],
            "sd_fit_rel": "the logged design-time read (sdfit_e02x.py): RMS of the transferred "
                          "pre-origin residual, >= 12 months",
            "sigma_ens_rel": "the 36 rerun members (18 thetas x 2 IC fields) at the external "
                             "cells around the median of the same members at the donors",
            "tau": f"DR.relative_drift.v1 report.json (sha {L.DR_REPORT_SHA}...) decisions; "
                   "Nantou: the larger triggered county value (report-only)",
            "dt": "DR's registered lead definition (datum centre: the external well's "
                  "t_centre_year; origin (2023.0)); a report without lead_definition is an "
                  "engineering failure",
            "centre": "N3F's registered predicted_m plus the transferred departure D_g[t]",
            "artefact": s["artefact"], "members": s["members"], "sd_fit_rel_read": s["sd_fit_rel"]},
        "binds": {"n3f": {"id": L.N3F_ID, "registration_sha256": n3f_sha,
                          "export": n3f["external_predictions"]["files"]},
                  "e02": {nm: io.rec(f"{L.E02_DIR}/{nm}")
                          for nm in ("predictions.csv", "protocol.json", "manifest.json")},
                  "forward": {"path": L.FORWARD, "sha256": L.MODEL_SHA},
                  "dr_report": s["dr"], "power": s["power"], "probe": s["probe"],
                  "rulings": s["rulings"], "ext_meta": prov,
                  "gov": {"id": L.GOV_ID, "registration": io.index_record(L.GOV_ID,
                                                                          prereg_dir)},
                  "bound_code": {f: io.sha256(io.repo_path(f)) for f in BOUND},
                  "code_at_registration_report_only": {f: io.sha256(io.repo_path(f))
                                                       for f in CODE_REPORT_ONLY},
                  "engineering": ENGINEERING_RESTORE},
        "frozen_code": frozen,
        "scoring": {
            "not_before": w["not_before"], "no_later_than": w["no_later_than"],
            "order": "in the 2028-01-15 session, LAST of E02c -> E02d -> N3F -> E02x: E02c's "
                     "report (or a lead-written not-scorable record), E02d's report.json (or "
                     "record) and N3F's report_ext.json (or record) must exist, each with its "
                     "score look logged when it scored; then the precondition; then "
                     "assert_can_score (gated test, then Nantou); then the rows are read",
            "precondition": f"{L.GOV_ID}'s check_precondition with E02x's entry (score_e02x."
                            "gov_entry: months of the window, needs_window True): the window "
                            "product's MANIFEST grants prospective_scorer (flags 0-1), DECLARES "
                            "parameters.prospective_scorer_window covering the window (the "
                            "registered GOV interpretation for 'later E02x') and that window "
                            "does not end after the product's fit_window_end; MANIFEST alone. "
                            "A readable MANIFEST that fails this is not_evaluable (nothing "
                            "read, no look); an absent or unreadable MANIFEST or a wrong "
                            "--window-product path is REFUSED (logged, retryable). Run after "
                            "the order check",
            "code_check": "before anything else the scorer verifies every sha256 of the "
                          "REGISTERED frozen_code and binds.bound_code against the current "
                          "files, and heads_window.py against GOV's registered input "
                          "(9a953039) before executing it",
            "thin_rows": "if N3F's step-(5) has no report.json (N3F's evaluate raised), the "
                         "--n3f-report record must state why ('evaluate_error') and carry "
                         "evaluation.csv's sha256 ('evaluation_sha256'); otherwise refused",
            "scorer": f"{SCORER} score --window {window} --window-product <dir> --step5-dir "
                      "<E02 step-(5) out> --ext-step5-dir <N3F's e02_protocol_run> "
                      "--e02c-report <file> --e02d-report <file> --n3f-report <file> --out <dir>",
            "writes": "report.json once (open mode 'x'); every invocation, refused or not, is "
                      "appended to <out>/attempts.jsonl and results/twin_runs/e02x/"
                      "attempts.jsonl"},
        "sealed_until_scoring": f"every head of {w['first']}..{w['last']} at the 66 external "
                                "wells; no 2023-26 head at them is ever opened",
        "registration_script": {"path": THIS, "sha256": io.sha256(Path(__file__)),
                                "binding": "report-only: the registered document's sha binds "
                                           "what this script wrote"},
        "prereg3_probe": {"units": [next(iter(unit)), next(iter(nunit))],
                          "step": f"{THIS} probe --window {window} (before the sd_fit_rel read; "
                                  "its record is bound above) and again at --write "
                                  "(check_blocking)", "record": s["probe"]},
        "page_mapping": {
            "before_scoring": {"C14f-r": "new ledger row: 'decided by a registered test "
                                         "decisive by design (U1.ext_transfer_power.v1); scored "
                                         "not before 2028-01-15'; grade D-marginal until "
                                         "evaluability at the external wells is established "
                                         "from metadata only",
                               "C14a / C14a-r / E08 / C02": "untouched"},
            "pass": "C14f-r S (prospective) for that county on the ledger and README always, "
                    "and on the page in evc_pro_ext (R8.3); wording 'per-well spread around the "
                    "transferred departure at never-calibrated wells', never the full "
                    "interval, never 'calibrated'",
            "fail": "the transferred per-well wording is withdrawn for that county (N, coverage "
                    "stated) on ledger, README and evc_pro_ext",
            "not_evaluable": "C14f-r stays D-marginal, 'undecided' (reason stated)",
            "nantou": "report-only on the ledger and README; never pooled, never a page verdict",
            "report_only_county": "Changhua's verdict, if Changhua is not gated, is a "
                                  "report-only line; never S",
            "wording": WORDING},
        "readings": {"pass": "decided S for that county and window only",
                     "fail": "the band (not the forecast level) is too narrow or mis-centred at "
                             "unseen wells; the by-zone breakdown says where (report-only)",
                     "not_evaluable": "the external wells did not report enough "
                                      "(availability), or a precondition failed"},
        "disclosures": {
            "FV:ext": FV_EXT_LOOK,
            "history": HISTORY,
            "sd_fit_rel": "the 66 wells' pre-origin heads and DR's 158-well residuals were "
                          "read once, logged, after U1.ext_transfer_power.v1's registration "
                          "and the cleared probe, to set sd_fit_rel and t_centre",
            "product_mix": "donor residuals come from E02's own read (AMP_V2/data/wells); the "
                           "external wells' from heads/L1_clean v3 (N3F's read); at scoring "
                           "both from E02's scorer on the window product. Time-varying "
                           "differences between the two products enter sd_fit_rel and widen "
                           "the band, which raises the risk of a fail ABOVE 0.95 (over-wide)",
            "calibration": PW.CALIBRATION_DISCLOSURE,
            "changhua": CHANGHUA_ARITH,
            "yunlin_bound": YUNLIN_BOUND,
            "nantou": NANTOU_DISCLOSURE,
            "fallback_deadline": FALLBACK_DEADLINE_SOURCE,
            "overlap": f"the window lies inside N3F's unit {L.N3F_UNIT} (same 66 wells, "
                       "2026-11..2027-10); N3F is scored first in the same session, so its "
                       "score look is a post-registration look here (disclosed, by design)",
            "donor_rows": "the donors' window rows are E02c / E02d's scored rows, read again "
                          "after those scores",
            "availability_risk": "heads/L1_clean v3's card lists post-origin records at the "
                                 "158 wells only; the external wells may not report, giving "
                                 "'not evaluable'; known before registration, not looked at; "
                                 "D-marginal until evaluability is established from metadata",
            "U7": "the combined interval family was chosen after its 2023-26 coverage at the "
                  "158 wells was seen; E02x moves the same family to unseen wells with the "
                  "departure transferred; the scored windows are untouched",
            "multiplicity": "N3F (level vs climatology) scores overlapping well-months of the "
                            "same 66 wells; E02c, E02d and others score the donors; each "
                            "decides only its own claim, no family-wise claim",
            "members": MEMBERS_DISCLOSURE},
        "attempt_policy": "engineering failures may be fixed and re-run before the score look, "
                          "each logged in attempts.jsonl; the test is scored ONCE after the "
                          "window closes; a failed or not-evaluable result is reported as such "
                          "and never re-scored on these units; at most one post-failure "
                          "exploratory follow-up (new ids, never validation); every attempt is "
                          "listed",
        "rulings": {"R8.3": s["rulings"],
                    "precedent": "the N3F-style December fallback precedent sets the "
                                 "2026-11-30T16:00Z deadline; the 2027-01 fallback's is "
                                 "2026-12-31T16:00Z"},
        "zones": NA("no fit; DR's zone groups label wells only"), "priors": NA("no prior"),
        "burn_in": NA("no fit"), "folds": NA("prospective window"),
        "fresh_units": [next(iter(unit)), next(iter(nunit))],
        "thresholds": {"target": L.TARGET, "gate_level": L.GATE_LEVEL,
                       "point_window": list(L.POINT_WINDOW), "min_months": L.MIN_MONTHS,
                       "min_wells_county": L.MIN_WELLS_COUNTY, "min_donors": L.MIN_DONORS,
                       "z": L.Z90},
        "gate_scores": NA("no weighted gate"),
        "primary_model": {"name": "datum_gate_innov (frozen forward), N3F's registered centre "
                                  "plus the transferred departure",
                          "forward_sha256": L.MODEL_SHA},
        "inputs": _inputs(s, window),
        "interpretation": "Per-well spread around the transferred departure at never-calibrated "
                          "wells only: not the level, not the shared departure, not the policy "
                          "contrast, not the full interval.",
        "written_by": THIS}


def _inputs(s: dict, window: str) -> list[str]:
    out = {f"{L.E02_DIR}/{n}" for n in ("predictions.csv", "protocol.json", "manifest.json")}
    out |= {f"{L.N3F_DIR}/{n}" for n in ("predictions.csv", "protocol.json", "manifest.json",
                                         "wells.csv")}
    out |= {L.FORWARD, f"{L.DR_DIR}/report.json", f"{L.DR_DIR}/wells_e02d.csv",
            *FROZEN_E02X, *BOUND, TESTS}
    for k, kk in (("power", "result"), ("members", "gate"), ("members", "npz"),
                  ("sd_fit_rel", "report"), ("sd_fit_rel", "wells"), ("artefact", "summary")):
        if isinstance(s[k].get(kk), dict):
            out.add(s[k][kk]["path"])
    if s["artefact"].get("files"):
        out |= {r["path"] for nm, r in s["artefact"]["files"].items() if window in nm}
    if s["rulings"].get("file"):
        out.add(s["rulings"]["file"]["path"])
    if (s["probe"].get("record") or {}).get("path"):
        out.add(s["probe"]["record"]["path"])
    return sorted(out)


def e02x_problems(window: str, d: dict, expect_self_sha, prereg_dir=None) -> list[str]:
    bad = []
    s = _slots(window, prereg_dir)
    if s["power"]["status"] != "ok":
        bad.append(f"power: {s['power']['status']}")
    else:
        reg = io.index_record(L.POWER_ID, prereg_dir)
        if reg is None or (s["power"]["declaration"] or {}).get("sha256") != reg["sha256"]:
            bad.append("the power result was not run under the registered declaration")
        if not (s["power"].get("registration") or {}).get("registrable"):
            bad.append(f"U1.ext_transfer_power.v1: Yunlin did not pass for window {window}: "
                       "E02x is NOT registered; record C14f-r N ('undecidable by design at "
                       "the external wells')")
    for k in ("members", "sd_fit_rel", "artefact"):
        if s[k]["status"] != "ok":
            bad.append(f"{k}: {s[k]['status']}")
    if s["sd_fit_rel"]["status"] == "ok":
        reg = io.index_record(L.POWER_ID, prereg_dir)
        if reg is None or (s["sd_fit_rel"]["declaration"] or {}).get("sha256") != reg["sha256"]:
            bad.append("the sd_fit_rel read did not run under the registered power declaration")
    if s["artefact"]["status"] == "ok" and not str(
            s["artefact"].get("dr_report_sha256") or "").startswith(L.DR_REPORT_SHA):
        bad.append(f"the artefact was not built from DR's report {L.DR_REPORT_SHA}")
    if s["probe"].get("status") == "PENDING" or s["probe"].get("blocking"):
        bad.append(f"probe: {s['probe'].get('status') or s['probe'].get('blocking')}")
    if s["rulings"]["status"] != "ok":
        bad.append(f"ruling {L.RULING_ID}: {s['rulings']['status']}")
    if io.sha256(io.repo_path(L.FORWARD)) != L.MODEL_SHA:
        bad.append("forward npz changed")
    for nm, h in (("predictions.csv", L.D.E02_PRED_SHA),
                  ("protocol.json", L.D.E02_PROTOCOL_SHA)):
        if io.sha256(io.repo_path(f"{L.E02_DIR}/{nm}")) != h:
            bad.append(f"E02 {nm} changed")
    t = d["tests"][0]
    if len(next(iter(t["external_units"].values()))["ids"]) != L.N_GATED:
        bad.append("the gated unit does not hold the 60 Yunlin + Changhua wells")
    if str(t["role_by_county"].get("Yunlin")).startswith("PENDING"):
        bad.append("roles by county are PENDING (power result)")
    if io.now() >= L.WINDOWS[window]["deadline"]:
        bad.append(f"DEADLINE PASSED ({L.WINDOWS[window]['deadline'].isoformat()}): do not "
                   f"register window {window}")
    if window == "fallback" and io.index_record(L.prereg_id("primary"), prereg_dir):
        bad.append("the primary window is registered: the fallback is not used")
    self_sha = io.sha256(Path(__file__))
    if expect_self_sha != self_sha:
        bad.append(f"--expect-self-sha {expect_self_sha!r} != {self_sha}")
    return bad


# ----------------------------------------------------------------------------- CLI
def main(argv=None, prereg_dir=None, rows_dir=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("which", choices=("power", "probe", "e02x"))
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--validate", action="store_true")
    g.add_argument("--write", action="store_true")
    ap.add_argument("--seed", type=int, default=None, help="power: required (no default)")
    ap.add_argument("--window", choices=sorted(L.WINDOWS), default=None)
    ap.add_argument("--dry", action="store_true", help="probe: print, write no record")
    ap.add_argument("--expect-self-sha", default=None)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    pdir = Path(prereg_dir) if prereg_dir else io.prereg_dir_default()
    try:
        if a.which == "probe":
            if a.window is None:
                ap.error("probe needs --window")
            r = run_probe(a.window, pdir, a.dry)
            print(json.dumps({k: r[k] for k in ("unit", "units", "blocking", "unverifiable",
                                                "time")}, indent=1, default=str))
            return 1 if r["blocking"] else 0
        if a.which == "power":
            if a.seed is None:
                ap.error("power needs --seed (no default)")
            d = draft_prereg_power(a.seed, pdir)
        else:
            if a.window is None:
                ap.error("e02x needs --window")
            d = draft_prereg_e02x(a.window, pdir)
    except (io.Refused, OSError) as exc:
        print(f"NOT READY: {exc}")
        return 1
    errs = prereg3.validate_prereg(d, prereg_dir=pdir)
    block = check_blocking(d["tests"], pdir) if isinstance(d["tests"], list) else []
    if not a.quiet:
        print(json.dumps(d, indent=1, ensure_ascii=False, sort_keys=True, default=str))
    print("draft sha256:", io.sha_json(io.jsonable(d)))
    print("validate:", errs or "ok")
    print("blocking looks:", block or "none")
    print("this script sha256:", io.sha256(Path(__file__)))
    if not a.write:
        return 1 if errs else 0
    bad = (power_problems(d, a.expect_self_sha, pdir) if a.which == "power" else
           e02x_problems(a.window, d, a.expect_self_sha, pdir))
    if errs or block or bad:
        print("REFUSED:", errs + [str(b) for b in block] + bad)
        return 1
    kw = {"prereg_dir": pdir}
    if rows_dir is not None:
        kw["rows_dir"] = rows_dir
    path, h = prereg3.write_prereg(io.jsonable(d), **kw)      # the LEAD's step only
    print(f"registered {path} sha256 {h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
