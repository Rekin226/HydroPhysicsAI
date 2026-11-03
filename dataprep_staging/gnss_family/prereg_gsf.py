"""Draft registration of X5.gnss_seasonal_family.v1 (harness.prereg3, kind 'test'; round 8
rank 11). DRAFT ONLY: nothing here writes a registration (write_prereg is never called); the
lead validates, runs the blocking probe and registers after the code is merged and G-sep and
G0 have passed and been logged.

``draft_prereg(seed, ...)`` refuses (SystemExit with the closure line) if the G-sep or G0
record failed. With ``pending=True`` it drafts on placeholder gate blocks so the document's
structure validates before the gates run; readiness then lists what is pending.

    python -I dataprep_staging/gnss_family/prereg_gsf.py --validate --seed <int> --pending
    python -I dataprep_staging/gnss_family/prereg_gsf.py --validate --seed <int>
"""
from __future__ import annotations

import argparse
import datetime as dt
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
import gsf_gates as GT  # noqa: E402
import gsf_model as MD  # noqa: E402
import score_gsf as SC  # noqa: E402

rp, sha256 = MD.rp, MD.sha256
THIS = f"{K.X}/prereg_gsf.py"
TESTS = "tests/test_gnss_family.py"
G0_LOOK_UNIT = "X5GSF:g0_synthetic"
PENDING = "pending"


def _na(reason: str) -> dict:
    return {"not_applicable": reason}


def station_ids(names) -> list[str]:
    """Bare and 'gnss:' forms (v1's convention), so prior looks in either form intersect."""
    return sorted({x for s in names for x in (str(s), f"gnss:{s}")})


PILE_IDS = sorted({x for s in K.PILES for x in (s, f"dbm:{s}",
                                                 f"dbm_cy007001_printed:{s}:pre")})
UNIT_NOTE = ("design-time units, NOT fresh in the U7 sense: seen in product QC and fusion "
             "(S3, S4b amplitude-easting, S5, S1b looks); v1's metadata read; never compared "
             "with the twin's seasonal amplitude before")


def units(ms: dict) -> dict:
    prim = ms["units"]["primary"]
    return {
        K.UNIT_PRIMARY: {"description": "Yunlin primary GNSS stations of v1's frozen model "
                                        "side (11 distal, 8 mid; WRA 2020-01..2022-12, NGL "
                                        "HUWE/DNAN 2014-08..2019-12); " + UNIT_NOTE,
                         "ids": station_ids(prim), "window": "2014-08..2022-12"},
        K.UNIT_NGL: {"description": "NGL HUWE and DNAN, v1's least-seen stratum (DIR's T1n); "
                                    + UNIT_NOTE,
                     "ids": station_ids(K.NGL_STRATUM), "window": "2014-08..2019-12"},
        K.UNIT_PILES: {"description": "deep piles TKJS(1) and NTUH(1), 300 m, deep_benchmarks/"
                                      "v2 pre-origin rows 2016-06..2019-12 (months G4 did not "
                                      "spend; seen in S2 QC/diagnose looks): U2 frame bound",
                       "ids": PILE_IDS, "window": "2016-06..2019-12"}}


DECISION = {
    "primary": "T1 = median distal / median mid station annual amplitude (median of per-year "
               "harmonic vectors); member m excluded on T1 if the 90 % interval of "
               "log(T1_obs/T1_m) lies wholly outside [-log 1.25, log 1.25]",
    "secondary_T2": "T2 = median distal amplitude; m excluded on T2 if the interval of "
                    "log(T2_obs/T2_m) lies wholly outside [-log 1.25, log 1.42 + log 1.25] "
                    "(frame interval [1/1.42, 1] widened by the allowance)",
    "survival": "a screen member survives unless excluded by T1 or T2",
    "interval": "two-stage bootstrap: stations within zone, then years within station; obs "
                "and members together; percentile 90 %; B 2000; SeedSequence([seed, 11])",
    "decided_by": "readings (a)-(c) are decided by T2 (frame-carrying; outcome close to "
                  "known); T1 (frame-free) excludes the far member only if its half-width is "
                  "below about 0.08 - see disclosures[0]",
    "readings": {"a": "survivors meet the cluster, miss the far members",
                 "b": "survivors miss the cluster, meet the far members (narrowing only)",
                 "c": "survivors meet both", "d": "no survivor"},
    "secondary_DIR": "v1's directional arm unchanged: r_A = sum A_obs / sum A_page (mean "
                     "annual vector), v1's bootstrap with its declared nuisance draw, reading "
                     "'larger'/'smaller' only if it holds at both scales 1 and 1/1.42, "
                     "'consistent' needs the interval inside [1/1.5, 1.5] at both scales and "
                     "T1n not opposite (v1's criteria); on the gated months (no extra jump "
                     "rule; v1 unchanged). C27 ONLY: 'larger' carries the S label (with DIR's "
                     "caveat); 'smaller', 'consistent' and 'inconclusive' are N notes. No G0 "
                     "of this declaration covers DIR; v1's G0-OC failed on it (scored-rule "
                     "P(consistent | r = 1) 0.11; P('larger' | r = 0.4) 0.57)",
    "secondary_U2": "per site (GNSS station paired with its own pile; no pooled sums): "
                    "s <= g/p, g = GNSS/page at the station (2020-22), p = pile/page full "
                    "column at the pile cell (pre-origin variant, 2016-06..2019-12, spring "
                    "months Mar-May masked on pile and model alike); bound = upper end of the "
                    "90 % year-bootstrap interval, SeedSequence([seed, 12]); a sensor gates "
                    "with >= 2 years (NTUH(1), one year, is report-only); the reading is the "
                    "tightest gated site's bound; narrows no screen",
    "label_on_pass": K.PASS_LABEL,
    "dir_label_on_larger": "S (design-time, window flagged, outcome partly known; v1's "
                           "directional arm, not covered by this G0; v1's G0-OC failed on its "
                           "consistent path)",
    "inclusive_bounds": False}


REPORT_ONLY = [
    "C-R2's F1 electricity reduced form (zone and pooled beta_obs/beta_mod on Yunlin zone "
    "electricity, pump_electricity/preorigin/v3 cells_total value_all, lag 0), and its "
    "2021-excluded variant",
    "phase lag obs - page per station", "per-station amplitudes (obs, page)",
    "ANES and CHSG rows (seen stations)",
    "Changhua-tier columns ccz, q0c_csk1, x5r1_cs read by the screen rule on the primary unit",
    "q0_nh read by the screen rule (nudged-head pairing unverified)",
    "the qc_jumps-cleaned variant of the screen and DIR (WRA station-months with >= 3 "
    "consecutive-day |delta value_raw| > 3 cm dropped) and its month list; U7 provenance: the "
    "jump-cluster constants were set in the S3 repair round after 2023-25 jump counts were "
    "seen, so the variant never gates",
    "U2's report-only site (NTUH: one pre-origin pile year; its GNSS cell 771 differs from the "
    "pile cell 769)"]

DISCLOSURES = [
    "DECIDED BY T2 (review B3): " + K.DECIDED_BY_T2 + "; a screen pass carries '"
    + K.PASS_LABEL + "'",
    *K.FOUR_DISCLOSURES,
    "C32 (v1's proposed new claim 'seasonal elastic subsidence at Yunlin GNSS') is dropped: it "
    "is not in claims_v1.json and R8.5 applies to this id, E24 and C27 only; DIR maps to C27",
    "no G0 of this declaration covers DIR; v1's G0-OC failed on DIR's consistent path "
    "(scored-rule P(consistent | r = 1) 0.11, scale-1 reading 0.92); its directional power was "
    "P('larger' | r = 0.4) 0.57 with no false 'larger' at r = 1",
    "the gated estimator applies no jump rule beyond the product's own flags (v1's flags "
    "0-1); the qc_jumps-cleaned variant is report-only because its constants postdate a "
    "2023-25 look (U7)",
    "the files live in dataprep_staging/gnss_family/ by the lead's choice (the plan named "
    "sd_column/, whose files stay untouched and hash-bound)",
    "the ring seasonal ratio 0.37 (mechanism.json) and step C's slope 0.22 are known priors "
    "toward 'larger'; WRA Yunlin stations are mostly co-located with ring schools",
    "gnss_clean/preorigin/v4 and deep_benchmarks/v2 failed their gates; read only as "
    "design_time_evidence (gate-exempt); 'selection' is not a use made here: no page column "
    "changes (R8.5: the screen is design-time narrowing, not selection)",
    "E24's full range stays printed; the screened range only ever appears BESIDE it (R8.5)",
    "v1 (X5.gnss_seasonal_column.v1) failed G0-OC under its scored two-scale rule (g0oc.json "
    "35ebf516; scale-1 reading 0.92); it was never registered and no GNSS value was read; "
    "this is its pre-look redesign (R8.5), with v1's directional arm kept unchanged as a "
    "secondary",
    "the dedup removes q0_s29 (identical to q0) and q0_csk1 (identical to q0_cs at every "
    "Yunlin station and pile cell); page_tau30 and cwa_tau30 differ from page and cwa by "
    "<= 2.3 mm in the hindcast and stay as members",
    "2021 is the record-low year (non-sinusoidal compaction): the median of per-year vectors "
    "and G0's record_low_2021 cell address it; WRA stations have three years (2020-22)",
    "U2 compares non-spring pile months 2016-06..2019-12 with GNSS months 2020-22 at the "
    "same site (no common window; stationarity assumed); NTUH(1) has one pre-origin year "
    "(2019) and is report-only; the NTUH GNSS station (cell 771) and pile (cell 769) sit on "
    "different cells; the spring mask is Mar-May (G4 spent pile springs; the mask is the "
    "builder's reading of 'non-spring', flagged to the lead)",
    "the GNSS frame scale may vary with sinking rate; G0's zone_frame cell (distal 1.42, mid "
    "1.0) tests the case in which T1's cancellation fails",
    "the builder printed the gnss_clean/preorigin/v4 MANIFEST gate block (product-QC A1/A2 "
    "statistics, no seasonal amplitude) and read deep_benchmarks/v2 README, mapping.csv "
    "metadata columns and the monthly.parquet sensor/variant/feed/flag/month columns (counts "
    "only); no observation value of any unit",
    "the declared design uses the plan's model-only T1/T2 numbers per arm (model_side.json); "
    "no observed value entered the allowance x1.25, the frame interval or the grid"]


def manifest_problems() -> list[str]:
    """Allowed-use check of every product read at scoring (gate, allowed, forbidden only)."""
    out = []
    for prod in (K.GNSS_PRODUCT, K.PILE_PRODUCT, K.ELEC_PRODUCT):
        p = rp(prod) / "MANIFEST.json"
        if not p.exists():
            out.append(f"{prod}: MANIFEST missing")
            continue
        m = json.loads(p.read_text())
        gate = m.get("gate")
        passed = bool(gate.get("passed")) if isinstance(gate, dict) else bool(gate)
        if K.USE not in (m.get("allowed_uses") or []) or K.USE in (m.get("forbidden_uses")
                                                                    or []):
            out.append(f"{prod}: {K.USE} not allowed")
        if not passed and K.USE not in K.GATE_EXEMPT:
            out.append(f"{prod}: gate failed and {K.USE} is not gate-exempt")
        if prod in (K.GNSS_PRODUCT, K.PILE_PRODUCT) and passed:
            out.append(f"{prod}: gate now passed (the declaration assumed failed; re-check)")
    return out


def rulings_problems() -> list[str]:
    p = rp(K.RULINGS)
    if sha256(p) != K.RULINGS_SHA:
        return [f"{K.RULINGS} is not ec610b69"]
    d = json.loads(p.read_text())
    r = [x for x in d.get("rulings", []) if x.get("id") == K.RULING_ID]
    if len(r) != 1:
        return [f"{K.RULING_ID} not found once"]
    text = r[0].get("decision", "")
    out = [f"{K.RULING_ID} lacks {s!r}" for s in K.RULING_PHRASES if s not in text]
    if K.ID not in r[0].get("applies_to", []):
        out.append(f"{K.RULING_ID} does not apply to {K.ID}")
    return out


def _gate(path: str, event: str, key: str, pending: bool, rec: dict | None,
          rec_sha: str | None) -> tuple[dict, str]:
    if rec is None:
        p = rp(path)
        if p.exists():
            rec, rec_sha = json.loads(p.read_text()), sha256(p)
        elif pending:
            return {"pass": PENDING}, PENDING
        else:
            raise SystemExit(f"REFUSED: {path} missing (run the gate first)")
    return rec, rec_sha


def draft_prereg(seed: int, *, gsep: dict | None = None, gsep_sha256: str | None = None,
                 g0: dict | None = None, g0_sha256: str | None = None,
                 decl_sha256: str | None = None, engineering: dict | None = None,
                 engineering_sha256: str | None = None, ms: dict | None = None,
                 decl_seed: int | None = None, pending: bool = False) -> dict:
    """The registration document. ``seed`` (the scoring seed) is required, fresh, and differs
    from the gates declaration's seed."""
    K.check_seed(seed)
    gsep, gsep_sha256 = _gate(K.GSEP_OUT, "gsep_run", "gsep_sha256", pending, gsep, gsep_sha256)
    g0, g0_sha256 = _gate(K.G0_OUT, "g0_run", "g0_sha256", pending, g0, g0_sha256)
    if gsep.get("pass") is False:
        raise SystemExit(GT.closure_line("G-sep", gsep))
    if g0.get("pass") is False:
        raise SystemExit(GT.closure_line("G0", g0))
    if engineering is None:
        p = rp(K.ENGINEERING)
        if p.exists():
            engineering, engineering_sha256 = json.loads(p.read_text()), sha256(p)
        elif pending:
            engineering, engineering_sha256 = {"cut30_cm": PENDING, "cluster_far": PENDING,
                                               "pile_series_digest": PENDING}, PENDING
        else:
            raise SystemExit("REFUSED: engineering.json missing")
    dp = rp(f"{K.PREREG_DIR_REL}/{K.DECL_ID}.json")
    if decl_sha256 is None:
        decl_sha256 = sha256(dp) if dp.exists() else PENDING
    # unconditional (review minor): the registered declaration's seed and any passed one
    taken = {decl_seed} if decl_seed is not None else set()
    if dp.exists():
        taken.add(json.loads(dp.read_text()).get("seeds", {}).get("declared"))
    if seed in {int(x) for x in taken if x is not None}:
        raise ValueError("the scoring seed must differ from the gates declaration's seed")
    if ms is None:
        ms, _ = MD.load_v1()
    U = units(ms)
    fz = MD.code_sha256()
    common = {"label": K.PASS_LABEL, "use": K.USE, "flags": list(K.GNSS_FLAGS),
              "common_mode": K.REMOVAL, "min_days_per_month": K.MIN_DAYS,
              "min_months_per_year": K.MIN_MONTHS, "min_years_per_station": K.MIN_YEARS,
              "jump_rule": "none beyond the product's flags (gated path)",
              "qc_jumps_report_only": {"family": "wra", "jump_m": K.QC_JUMP_M,
                           "per_month": K.QC_JUMPS_PER_MONTH, "column": "value_raw",
                           "rule": "a WRA station-month with >= 3 consecutive-calendar-day "
                                   "|delta value_raw| > 0.03 m is dropped before the monthly "
                                   "medians (s3_gnss/v4 jump constants)"}}

    def test(tid, scope, unit_names, role, claim):
        return {"test_id": tid, "data_scope": scope, "fresh_units": list(unit_names),
                "external_units": {u: U[u] for u in unit_names}, "role": role, "claim": claim,
                **common}

    tests = [
        test(K.T1_ID, "gnss:2014-08..2022-12", [K.UNIT_PRIMARY], "primary (scored once)",
             "E24 screened model-form spread beside the full range (C11/C26/C29 notes)"),
        test(K.T2_ID, "gnss:2014-08..2022-12", [K.UNIT_PRIMARY],
             "named secondary (same look; part of the screen's survival rule)",
             "the screen's absolute check against the frame interval"),
        test(K.DIR_ID, "gnss:2014-08..2022-12", [K.UNIT_PRIMARY, K.UNIT_NGL],
             "named secondary (v1's directional arm, criteria unchanged)",
             "C27 direction of the seasonal elastic size"),
        test(K.U2_ID, "deep_benchmarks:2016-06..2019-12", [K.UNIT_PILES],
             "named secondary (one-sided frame bound; narrows no screen)",
             "GNSS frame over-read bound at TKJS/NTUH")]
    cf = engineering.get("cluster_far")
    return {
        "id": K.ID, "wp": K.WP, "kind": "test",
        "title": "GNSS seasonal family screen of SD v2's leveling-equivalent columns at Yunlin "
                 "(scale-free distal/mid amplitude ratio primary; distal amplitude, v1's "
                 "directional arm and a deep-pile frame bound as named secondaries); "
                 "design-time, window flagged, outcome partly known",
        "priors": "Direction prior 'larger' (ring seasonal ratio 0.37; step C slope 0.22; N1's "
                  "6.1 mm InSAR median): outcome (b) is the likely screen reading and 'larger' "
                  "the likely DIR reading. Plan: P(G0) ~0.6, informative ~0.7.",
        "tests": tests, "seed": seed,
        "zones": "zones.fan_zones distal/mid at each station's twin cell (v1 model side); "
                 "Yunlin by the twin county mask",
        "burn_in": "model side: SD forwards' free hindcast 2012-01..2022-12 (main parameter "
                   "set, gain 0, column on its paired free heads); stations start 2014-08",
        "folds": _na("one pooled screen; no folds"),
        "fresh_units": _na("listed per test"),
        "thresholds": {"t1_band_log": list(K.T1_BAND), "t2_band_log": list(K.T2_BAND),
                       "allowance": K.ALLOW, "frame_max": K.FRAME_MAX, "level": K.LEVEL,
                       "B": K.B_SCORE, "cluster_x": K.CLUSTER_X, "far_x": K.FAR_X,
                       "min_years_u2": K.MIN_YEARS_U2},
        "gate_scores": _na("declared readings; no weighted score"),
        "primary_model": "SD v2 Tier A+B columns on their own free heads (v1 model side "
                         "65b927cb), deduplicated; page column d290c592 unchanged",
        "decision_rule": DECISION,
        "decision_table": {r: SC.OUTCOME_TEXT[r] for r in ("a", "b", "c", "d")},
        "ledger_texts": {"outcomes": SC.OUTCOME_TEXT, "DIR": SC.DIR_TEXT,
                         "DIR_clash": SC.DIR_CLASH, "U2": SC.U2_TEXT, "never": SC.NEVER},
        "page_mapping": {
            "ruling": f"{K.RULING_ID} ({K.RULINGS}, sha256 {K.RULINGS_SHA}): "
                      f"'{K.RULING_LABEL}', applied as '{K.PASS_LABEL}' (lead, review B3); "
                      "the screen is design-time narrowing, not "
                      "selection; E24 shows the screened range BESIDE the full range, never "
                      "replacing it",
            "a_or_b": "E24 screened members named beside the full family (ledger and README; "
                      "page only under R8.5, R8.11 keeps E24 on tile tA as 'model-form spread, "
                      "not an interval'); notes on C11/C26/C29 (and C03 under (b), C03 stays "
                      "W); label '" + K.PASS_LABEL + "' - decided by T2; member names only, "
                      "no cut30 number beside the label",
            "c_or_d": "README Limits line; E24 unchanged (d: 'understated' note)",
            "DIR": "C27 only: 'larger' carries the S label with DIR's caveat; 'smaller', "
                   "'consistent' and 'inconclusive' are N notes; C32 dropped; never a factor",
            "U2": "ledger note beside T2; narrows nothing",
            "never": SC.NEVER},
        "rulings": {K.RULING_ID: {"file": K.RULINGS, "sha256": K.RULINGS_SHA}},
        "members": {"screen": list(K.SCREEN), "dedup": K.DEDUP,
                    "report_only": list(K.REPORT_ONLY_ARMS)},
        "engineering": {"path": K.ENGINEERING, "sha256": engineering_sha256,
                        "cut30_cm": engineering.get("cut30_cm"), "cluster_far": cf},
        "model_side": {"v1_model_side_json": K.V1_MODEL_JSON,
                       "v1_model_side_sha256": K.V1_MODEL_SHA,
                       "v1_series_sha256": K.V1_SERIES_SHA,
                       "pile_model_json": K.PILE_MODEL_JSON,
                       "pile_series_digest": engineering.get("pile_series_digest")},
        "gates": {"declaration": {"id": K.DECL_ID, "sha256": decl_sha256},
                  "gsep": {"path": K.GSEP_OUT, "sha256": gsep_sha256,
                           "pass": gsep.get("pass"), "n_separated": gsep.get("n_separated")},
                  "g0": {"path": K.G0_OUT, "sha256": g0_sha256, "pass": g0.get("pass"),
                         "checks": g0.get("checks"), "summary": {
                             k: v for k, v in (g0.get("summary") or {}).items()
                             if k != "readings_by_truth"}},
                  "v1_g0oc": {"path": K.V1_G0, "sha256": K.V1_G0_SHA, "pass": False}},
        "report_only": REPORT_ONLY,
        "disclosures": DISCLOSURES,
        "frozen_code": fz,
        "scoring": {"order": SC.__doc__.split("Order inside ``score``")[1].strip(),
                    "runner": f"bash {K.X}/run_gsf.sh score"},
        "attempt_policy": {
            "scored_runs": "every test scored once, in one session; engineering failures "
                           "before the GNSS read are free and logged",
            "pre_declared_alternatives": [], "exploratory_followups_max": 0,
            "followup_label": "none: v1's redesign is this declaration",
            "thresholds_frozen": True},
        "manifest": {"products": {K.GNSS_PRODUCT: "gate failed; design_time_evidence "
                                                  "(gate-exempt)",
                                  K.PILE_PRODUCT: "gate failed; design_time_evidence "
                                                  "(gate-exempt), window_end 2022-12",
                                  K.ELEC_PRODUCT: "design_time_evidence (report-only F1)"}},
        "inputs": ([{"path": p, "sha256": s} for p, s in fz.items() if s != "missing"]
                   + [p for p in (THIS, TESTS, K.RULINGS, K.V1_MODEL_JSON, K.V1_SERIES_NPZ,
                                  K.SD_REPORT, K.V1_G0, K.PILE_MODEL_JSON, K.PILE_MODEL_NPZ,
                                  K.ENGINEERING, K.GSEP_OUT, K.G0_OUT,
                                  f"{K.GNSS_PRODUCT}/MANIFEST.json",
                                  f"{K.PILE_PRODUCT}/MANIFEST.json",
                                  f"{K.ELEC_PRODUCT}/MANIFEST.json")
                      if rp(p).exists()]),
        "interpretation": "A screen reading says which leveling-equivalent columns are "
                          "consistent with the observed coast-to-inland seasonal amplitude "
                          "contrast (frame-free) and the distal amplitude within the frame "
                          "interval, at Yunlin GNSS, design-time with the outcome partly "
                          "known. It checks the chain damped heads x Ske at seasonal "
                          "frequency, not the 1-3 yr or long-run cut size; the page column and "
                          "every number on the page stay.",
        "written_by": THIS}


def readiness(d: dict) -> list[str]:
    out = []
    for rec in d["inputs"]:
        p = rec["path"] if isinstance(rec, dict) else rec
        if not rp(p).exists():
            out.append(f"input {p} is missing")
    g = d["gates"]
    if g["declaration"]["sha256"] == PENDING:
        out.append(f"{K.DECL_ID} not registered")
    for k, ev, key in (("gsep", "gsep_run", "gsep_sha256"), ("g0", "g0_run", "g0_sha256")):
        if g[k]["pass"] is not True:
            out.append(f"{k}: {g[k]['pass']}")
        elif g[k]["sha256"] != GT.logged_sha(ev, key):
            out.append(f"{k}: file is not the one its run logged")
    if g["g0"]["pass"] is True and not g0_logged(g["g0"]["sha256"]):
        out.append(f"G0 not logged: no look on {G0_LOOK_UNIT} whose note carries "
                   f"g0.json's sha256")
    out += manifest_problems() + rulings_problems()
    if rp(f"{K.PREREG_DIR_REL}/{K.ID}.json").exists():
        out.append(f"{K.ID} is already registered")
    if rp(K.REPORT).exists():
        out.append(f"{K.REPORT} exists")
    return out


def g0_logged(g0_sha: str) -> bool:
    from harness import prereg3

    p = rp(K.LOOKS)
    if not p.exists():
        return False
    return any(G0_LOOK_UNIT in (lk.get("units") or []) and g0_sha in str(lk.get("note", ""))
               for lk in prereg3.read_chained(p))


def blocking(d: dict) -> list:
    from harness import prereg3

    idx = prereg3.membership_index()
    for t in d["tests"]:
        for u, v in t["external_units"].items():
            if not idx.known(u):
                m = prereg3._declared_members(v)
                if m is not None:
                    idx.members[idx.canon(u)] = m
    return prereg3.blocking_looks(d["tests"], None, idx=idx,
                                  test_time=dt.datetime.now(dt.timezone.utc))


def main(argv=None) -> int:
    from harness import prereg3

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--validate", action="store_true", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--pending", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    d = draft_prereg(a.seed, pending=a.pending)
    if not a.quiet:
        print(json.dumps(d, indent=1, ensure_ascii=False, sort_keys=True, default=float))
    errs = prereg3.validate_prereg(d)
    print("validate_prereg:", errs or "valid")
    block = blocking(d)
    print("blocking looks (dry run):", block or "none")
    pend = readiness(d)
    print("readiness:", pend or "ready to register (the lead registers; never this script)")
    print(f"this script sha256: {sha256(Path(__file__).resolve())}")
    return 1 if (errs or block or pend) else 0


if __name__ == "__main__":
    raise SystemExit(main())
