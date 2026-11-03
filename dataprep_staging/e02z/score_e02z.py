"""E02z scorer (one session per window: B on 2029-01-15..31, C on 2030-01-15..31), after
E02y's scorer has written that window's report and its step-(5) rows.

Rows. E02z reads no heads table of its own: its rows are the step-(5) output that E02y's
registered scorer (dataprep_staging/e02d/score_e02d.py) produced in-process into
<E02y out>/step5 from the year-2/3 absolute registration (results/twin/prospective_2027-11_y2
or _2028-11_y3, forward cf2c7154), verified here by E02c's frozen ``load_e02_step5`` against
the registered predictions, protocol and manifest (sha256 pinned).

Refusals, in this order, each logged in the attempts log and none writing a report: the
window's report already exists (written once); E02z is not registered; it was registered at or
after REG_DEADLINE (review S6); it is before the window's not_before; heads_window.py (sha
9a953039) or score_e02c.py (sha 1c4e092a) is not bound in frozen_code, or any frozen file
differs from its registered sha256; E02y's registration is not the pinned file.

Then, in order:
1. E02y's window report (a JSON naming E02y and the window) and E02y's score looks of the window
   in looks.jsonl. If E02y LOOKED (review S3): its report must be a scored report, <E02y
   out>/step5 must hold report.json and evaluation.csv whose sha256 equal the report's
   rows_provenance (review Q6), and a 'not_scorable' / 'not_evaluable' record is REFUSED as an
   engineering error. If E02y did not look: its own 'not_evaluable' record or a lead-written
   'not_scorable' record -> 'level_not_evaluable', no look; a scored report without looks is
   REFUSED;
2. the MANIFEST precondition, E02z's own (R8.14; the N2CS precedent, lead ruling B1): E02c's
   frozen ``product_grants_scorer`` AND the registered GOV module's public
   ``heads_window.covering_rule(m, window)``, from the MANIFEST alone; the MANIFEST's sha256
   is recorded (Q6). If E02y looked, a failing precondition (a wrong or non-covering product)
   is REFUSED (review S3); otherwise the window is 'level_not_evaluable' without a look;
3. pre-look work (the N2CS B2 precedent; review S7: every non-observation read happens here):
   the frozen baselines' sha256 and table, E02c's well table and blocks, the registered seed,
   E02y's registered prediction files, then the step-(5) report's registration_manifest, the
   evaluation.csv header, station ids as 8-character strings with their leading zeros, every
   date parsing to a month, every row one of the 158 registered wells, and >= 1 row in the
   window (station_id / date columns only; no observed_m value): failing -> REFUSED;
4. ``prereg3.assert_can_score`` (logs the score look of the window's test); then the rows are
   verified (``load_e02_step5``), each county is decided by E02c's strict rule on its own
   (e02z_lib.county_level, the registered seed) and the pre-worded lines are written. After
   the look (review Q4): an environment failure before any statistic is computed (a missing
   file, an OSError) is logged and REFUSED, to be rerun with unchanged code; any other
   exception writes the final report 'engineering_failure_after_look' (disclosed, never
   re-scored). A session after the window's no_later_than is flagged.

The decidability report counts only if created before window B's not_before (review B1);
otherwise its reading is 'not_run'.

    python -I dataprep_staging/e02z/score_e02z.py score --window B|C --window-product <dir> \\
        --e02y-report <E02y out>/report.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(HERE), str(REPO / "dataprep_staging" / "e02c"), str(REPO / "dataprep_staging"),
           str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import decide_e02z as DZ  # noqa: E402  (imported here: an import error never follows a look)
import e02z_const as K  # noqa: E402
import e02z_lib as L  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import score_e02c as S  # noqa: E402  (E02c's frozen scorer, read-only)

SCORER_VERSION = "1.1.0"
Refused = L.Refused
EVAL_COLUMNS = ("station_id", "date", "predicted_m", "lower_m", "upper_m", "baseline_m",
                "observed_m")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def gov_module():
    """The registered GOV module (its sha256 is checked by the caller)."""
    return _load(L.rp(K.GOV), "gov_heads_window_e02z")


# ----------------------------------------------------------------------------- checks
def precondition(window_product: Path, win: str) -> tuple[bool, str]:
    """R8.14: E02c's frozen grant check AND GOV's covering rule for the window's months, from
    the MANIFEST alone."""
    from common.manifest import load_manifest

    ok, why = S.product_grants_scorer(window_product)
    if not ok:
        return False, f"product_grants_scorer: {why}"
    m, _ = load_manifest(window_product)
    ok, rule = gov_module().covering_rule(m, K.WINDOWS[win])
    return bool(ok), f"product_grants_scorer: {why}; {K.GOV_ID} covering_rule: {rule}"


def e02y_status(report: Path, win: str, prereg_dir) -> dict:
    """E02y's window report and score looks (session order; review S3, Q6). Reads E02y's
    report JSON, looks.jsonl and file hashes only, never a value."""
    rec, why = S._read_record(report, K.E02Y_ID)
    if rec is None:
        return {"ok": False, "why": f"E02y's window report missing: {why}"}
    if rec.get("window") != K.E02Y_WINDOW_KEY[win]:
        return {"ok": False, "why": f"{report} is E02y's window {rec.get('window')!r}, not "
                                    f"{win!r}"}
    missing = S.assert_order([(K.E02Y_ID, t) for t in K.E02Y_TESTS[win]], prereg_dir)
    looked = len(missing) < len(K.E02Y_TESTS[win])
    st = {"e02y_report": str(report), "e02y_report_sha256": L.sha256(report),
          "e02y_looked": looked, "e02y_looks_missing": missing}
    verdict = rec.get("verdict")
    if verdict in (K.NOT_SCORABLE, "not_evaluable"):
        if looked:
            return {**st, "ok": False,
                    "why": f"engineering error: E02y's window {win} was looked at, but the "
                           f"record says {verdict!r}"}
        return {**st, "ok": True, "evaluable": False,
                "why": f"E02y's window {win} is {verdict} ({rec.get('why', '')}): no "
                       "step-(5) rows exist"}
    if missing:
        return {**st, "ok": False, "why": f"E02y's score looks are not logged: {missing}"}
    step5 = Path(report).parent / "step5"
    gone = [n for n in S.E02_STEP5_FILES if not (step5 / n).is_file()]
    if gone:
        return {**st, "ok": False, "why": f"E02y's step (5) files missing in {step5}: {gone}"}
    prov = rec.get("rows_provenance") or {}
    unbound = [n for n in S.E02_STEP5_FILES
               if prov.get(f"{n}_sha256") != L.sha256(step5 / n)]
    if unbound:
        return {**st, "ok": False, "why": f"step (5) {unbound} differ from E02y's "
                                          "rows_provenance sha256"}
    return {**st, "ok": True, "evaluable": True, "step5": str(step5),
            "step5_sha256": {n: prov[f"{n}_sha256"] for n in S.E02_STEP5_FILES},
            "why": "E02y's window report, score looks and step (5) bound by sha256"}


def manifest_sha256(window_product: Path) -> str | None:
    """Review Q6: the sha256 of the window product's MANIFEST E02z checked."""
    from common.manifest import MANIFEST_NAME

    p = Path(window_product)
    p = p / MANIFEST_NAME if p.is_dir() else p
    return L.sha256(p) if p.is_file() else None


def rows_contract_problems(step5: Path, ids, win: str, manifest: dict) -> list[str]:
    """Before the look: metadata and the station_id / date columns only."""
    step5 = Path(step5)
    e = []
    try:
        rep = json.loads((step5 / "report.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"step-(5) report.json unreadable ({exc})"]
    if rep.get("registration_manifest") != manifest:
        e.append("step-(5) report.json does not carry the registered manifest")
    try:
        head = pd.read_csv(step5 / "evaluation.csv", nrows=0)
    except (OSError, ValueError) as exc:
        return e + [f"evaluation.csv unreadable ({exc})"]
    miss = [c for c in EVAL_COLUMNS if c not in head.columns]
    if miss:
        return e + [f"evaluation.csv lacks columns {miss}"]
    df = pd.read_csv(step5 / "evaluation.csv", usecols=["station_id", "date"],
                     dtype={"station_id": str, "date": str}, keep_default_na=False)
    ids = {str(i) for i in ids}
    sid = df["station_id"]
    if not all(isinstance(v, str) and len(v) == 8 for v in sid):
        e.append("station_id values are not 8-character strings")
    stripped = {i.lstrip("0") for i in ids}
    lost = sorted({v for v in set(sid) if v not in ids and v in stripped})
    if lost:
        e.append(f"ids without their leading zeros: {len(lost)}")
    other = sorted(set(sid) - ids)
    if other and not lost:
        e.append(f"{len(other)} station ids are not the 158 registered wells")
    try:
        mo = pd.to_datetime(df["date"], format="mixed").dt.to_period("M").dt.to_timestamp()
    except (ValueError, TypeError) as exc:
        return e + [f"a date does not parse to a month ({exc})"]
    if not (sid.isin(ids) & mo.isin(L.window_months(win))).any():
        e.append("no row of the 158 wells in the window")
    return e


def baseline_problems(doc: dict) -> list[str]:
    files = ((doc.get("baselines") or {}).get("files") or {})
    if K.BASE_FILE not in files:
        return [f"the registration binds no {K.BASE_FILE}"]
    p = L.rp(K.BASE_DIR) / K.BASE_FILE
    if not p.is_file() or L.sha256(p) != files[K.BASE_FILE]:
        return [f"{p} differs from its registered sha256"]
    if L.sha256(L.rp(K.E02C_WELLS)) != K.E02C_WELLS_SHA:
        return ["E02c's wells_158.csv differs from its registered sha256"]
    return []


def abs_files(win: str) -> tuple[pd.DataFrame, Path, dict]:
    d = L.rp(K.ABS_DIRS[win])
    bad = [n for n, h in K.ABS_SHA[win].items() if L.sha256(d / n) != h]
    if bad:
        raise L.EngineeringFailure(f"E02y's registered files changed: {bad}")
    pred = pd.read_csv(d / "predictions.csv", dtype={"station_id": str})
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    return pred, d / "protocol.json", man


# ----------------------------------------------------------------------------- outcome
def page_entry(verdict: str, county: str, win: str) -> str:
    first, last = K.WINDOWS[win]
    if verdict == "level_supported":
        return (f"C30-scen heads part ({county}, {first}..{last}): N -> S-limited, average-"
                "weather scenario level (never C30-fc)")
    if verdict == "level_not_supported":
        return (f"C30-scen heads part ({county}, {first}..{last}): stays N, decided "
                "(narrowed) with the pre-worded line")
    return f"C30-scen heads part ({county}, {first}..{last}): stays N, 'undecided'"


def outcome(level: dict, win: str, decide: dict | None) -> dict:
    """Verdicts, lines and page entries per county from ``window_level``'s result."""
    out = {"verdicts": {}, "wording": {}, "page_mapping": {}, "decidability": {}}
    for c in K.COUNTIES:
        v = level["counties"][c]
        d = DZ.reading_for(decide, win, c)
        out["verdicts"][c] = v["verdict"]
        out["wording"][c] = L.wording(v["verdict"], c, win, why=v.get("why", ""), decide=d)
        out["page_mapping"][c] = page_entry(v["verdict"], c, win)
        out["decidability"][c] = {**d, "grade_bookkeeping": K.GRADE.get(d["reading"], 0.5)}
    out["page_mapping"]["C30-fc"] = "untouched (E04's; R8.7)"
    return out


def score_window_rows(frame, baselines, blocks: dict, seed: int, win: str,
                      decide: dict | None = None) -> dict:
    """Pure: both counties on one window's annotated rows."""
    level = L.window_level(frame, baselines, blocks, seed, win)
    return {"level": level, **outcome(level, win, decide)}


def _not_evaluable(win: str, why: str, decide: dict | None) -> dict:
    lv = {"window": win, "counties": {c: {"county": c, "verdict": "level_not_evaluable",
                                          "why": why} for c in K.COUNTIES}}
    return {"level": lv, **outcome(lv, win, decide)}


def _decide_report() -> tuple[dict | None, str | None, bool]:
    """(report, sha256, in_time). A report created at or after window B's not_before (or
    without 'created') is read as 'not_run' by ``DZ.reading_for`` (review B1)."""
    p = L.rp(K.DECIDE_REPORT)
    if not p.is_file():
        return None, None, False
    rep = json.loads(p.read_text(encoding="utf-8"))
    return rep, L.sha256(p), DZ.report_in_time(rep)


def versions() -> dict:
    return {"numpy": np.__version__, "pandas": pd.__version__,
            "python": sys.version.split()[0]}


# ----------------------------------------------------------------------------- score
def score(win: str, window_product: Path, e02y_report: Path, *, now: dt.datetime | None = None,
          out: str | None = None, attempts: str | None = None, prereg_dir=None,
          rows_dir=None) -> dict:
    from harness import prereg3

    if win not in K.WINDOWS:
        raise Refused(f"unknown window {win!r} (B or C)")
    now = now or L.now_utc()
    outp = L.rp(out or K.SCORE_REPORT[win])
    att = L.rp(attempts or K.SCORE_ATTEMPTS)
    pdir = Path(prereg_dir) if prereg_dir else prereg3.PREREG_DIR
    kw = {"prereg_dir": pdir, **({"rows_dir": rows_dir} if rows_dir else {})}

    def refuse(why: str, event: str = "score_refused"):
        L.log_attempt(att, {"event": event, "window": win, "why": why})
        raise Refused(f"REFUSED: {why}")

    L.log_attempt(att, {"event": "score_invoked", "window": win})
    if outp.exists():
        refuse(f"{outp} exists; window {win} is scored once")
    try:
        doc, reg_sha = prereg3.load_prereg(K.ID, **kw)
    except Exception as exc:                       # noqa: BLE001 - any failure refuses
        refuse(f"{K.ID} is not registered ({type(exc).__name__})")
    why = L.deadline_problem(doc.get("created"))
    if why:
        refuse(why)
    if now < L.parse_utc(K.NOT_BEFORE[win]):
        refuse(f"window {win} is not scored before {K.NOT_BEFORE[win]}")
    frozen = doc.get("frozen_code") or {}
    for f, s in ((K.GOV, K.GOV_SHA), (K.E02C_SCORER, K.E02C_SCORER_SHA)):
        if frozen.get(f) != s:
            refuse(f"{f} is not bound at {s[:8]} in the registration")
    bad = sorted(f for f, s in frozen.items() if not L.rp(f).is_file() or L.sha256(L.rp(f)) != s)
    if bad:
        refuse(f"frozen code changed: {bad}")
    e02y_doc = pdir / f"{K.E02Y_ID}.json"
    if not e02y_doc.is_file() or L.sha256(e02y_doc) != K.E02Y_SHA:
        refuse("E02y's registration is not the pinned file")
    decide, decide_sha, decide_in_time = _decide_report()
    if not decide_in_time:
        decide = None                              # B1: late or undated -> 'not_run'
    rec = {"prereg_id": K.ID, "test_id": K.TEST[win], "window": win,
           "scorer_version": SCORER_VERSION, "registration_sha256": reg_sha,
           "scored_at": now.isoformat(), "versions": versions(),
           "session_breach": bool(now >= L.parse_utc(K.NO_LATER_THAN[win])),
           "decidability_report_sha256": decide_sha,
           "decidability_report_in_time": decide_in_time,
           "window_product": str(window_product),
           "window_product_manifest_sha256": manifest_sha256(Path(window_product))}

    def write(r: dict) -> dict:
        L.write_once(outp, r)
        L.log_attempt(att, {"event": "score_written", "window": win,
                            "verdicts": r.get("verdicts"), "sha256": L.sha256(outp)})
        return r

    # (1) E02y's window report and looks (session order)
    st = e02y_status(Path(e02y_report), win, pdir)
    rec["e02y_order"] = st
    if not st["ok"]:
        refuse(f"session order: {st['why']}; nothing read, no look logged")
    # (2) MANIFEST metadata only
    ok, why = precondition(Path(window_product), win)
    rec["precondition"] = {"ok": bool(ok), "why": why}
    if not st["evaluable"]:
        return write({**rec, **_not_evaluable(win, st["why"], decide), "looked": False})
    if not ok:                                     # S3: E02y looked; the product is wrong
        refuse(f"engineering error: E02y's window {win} was scored but the window product "
               f"given to E02z fails the precondition ({why})")
    # (3) pre-look work: every non-observation read, then the id/date contract (S7)
    probs = baseline_problems(doc)
    if probs:
        refuse("baselines: " + "; ".join(probs))
    try:
        pred, proto, man = abs_files(win)
        base = pd.read_csv(L.rp(K.BASE_DIR) / K.BASE_FILE, dtype={"station_id": str})
        base = base[base["window"] == win]
        wells = pd.read_csv(L.rp(K.E02C_WELLS), dtype={"station_id": str})
        blocks = dict(zip(wells["station_id"], wells["block5"].astype(str), strict=True))
        seed = L.check_seed((doc.get("thresholds") or {}).get("boot_seed"))
        ids = sorted(next(t for t in doc["tests"] if t["test_id"] == K.TEST[win])
                     ["external_units"][K.UNIT[win]]["ids"])
    except (L.Refused, OSError, KeyError, ValueError, StopIteration) as exc:
        refuse(f"pre-look inputs: {type(exc).__name__}: {exc}")
    if base.empty:
        refuse(f"no frozen baselines for window {win}")
    probs = rows_contract_problems(Path(st["step5"]), ids, win, man)
    if probs:
        refuse("rows contract: " + "; ".join(probs))
    # (4) the look, then the rows
    can = prereg3.assert_can_score(K.ID, K.WP, test_id=K.TEST[win], **kw)
    rec["post_registration_looks"] = can["post_registration_looks"]
    rec["unverifiable_prior_looks_n"] = len(can["unverifiable_prior_looks"])
    rec["looked"] = True
    rec["seed"] = seed
    started = False
    try:
        frame, rep5, prov = S.load_e02_step5(Path(st["step5"]), pred, proto, man)
        started = True                             # a statistic may be computed from here
        res = score_window_rows(frame, base, blocks, seed, win, decide)
        rec.update({**res, "rows_provenance": prov,
                    "step5_protocol_verdict_report_only": {k: rep5.get(k) for k in
                                                           ("passed", "rmse_ratio",
                                                            "coverage")}})
    except OSError as exc:
        if started:
            return write(_after_look_failure(rec, win, decide, exc))
        refuse(f"environment failure after the look, before any statistic ({type(exc).__name__}:"
               f" {exc}): logged; rerun with unchanged code",
               event="environment_failure_after_look")
    except Exception as exc:                       # noqa: BLE001 - after the look: disclose
        return write(_after_look_failure(rec, win, decide, exc))
    return write(rec)


def _after_look_failure(rec: dict, win: str, decide: dict | None, exc: Exception) -> dict:
    rec.update({**_not_evaluable(win, "engineering failure after the score look", decide),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback_tail": traceback.format_exc().splitlines()[-6:]})
    for c in K.COUNTIES:
        rec["verdicts"][c] = "engineering_failure_after_look"
        rec["wording"][c] = L.wording("engineering_failure_after_look", c, win,
                                      decide=rec["decidability"][c])
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--window", choices=sorted(K.WINDOWS), required=True)
    s.add_argument("--window-product", type=Path, required=True)
    s.add_argument("--e02y-report", type=Path, required=True)
    a = ap.parse_args(argv)
    try:
        r = score(a.window, a.window_product, a.e02y_report)
    except Refused as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps({"window": r["window"], "verdicts": r["verdicts"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
