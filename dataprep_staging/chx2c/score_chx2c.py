"""Score I1.CHX.v1.twocontrast once: the two-contrast exploratory narrowing of I1.CHX.v1's
direction pass at the 23 spent Changhua screens (R8.10; cannot promote).

Two commands, run by the lead in this order (each step refuses before the next):

``look`` (model only; no head is read):

0. the parent's report.json exists, this follow-up's report does not;
1. the declaration ``I1.CHX.v1.twocontrast`` is registered and intact (INDEX + ledger sha256),
   kind 'declaration' with followup_of I1.CHX.v1, its frozen code matches, its hashed inputs
   are unchanged, and the registered follow-ups of I1.CHX.v1 (``followup_of`` in every
   registered document, this one's pre-declared repair ids exempt) do not exceed the parent's
   ``attempt_policy.exploratory_followups_max``;
2. the panel is CHX's (sha256 = the parent report's panel_sha256), its screens are the
   parent's 23 ids and its window is the parent's; only ``chx2c_lib.PANEL_KEYS`` are loaded
   (model output and metadata; the panel holds no observation);
3. collinearity (model only, the parent's order: score_chx decides 'not_informative' from the
   panel before any look): |r| of the two transformed contrasts over 23 x 120. If |r| > 0.9
   (or r is not finite) the report is written with 'not separable', NO look is logged and NO
   head is ever read; otherwise the ``after_the_fact`` look is logged.

``score`` (once, after the look):

4. steps 0-3 again (the same model-only r), and the look is in looks.jsonl after the
   registration;
5. heads: L1_clean v3 through ``chx_io.l1_monthly`` (CHX's own loader and use, to reproduce
   the parent's read: design_time_evidence, window_end 2022-12, flags 0-1, the 23 screens,
   2013-01..2022-12);
6. reproduction pre-gate: CHX's frozen ``score_chx.score_panel`` and this code with the
   Yunlin regressor dropped both give the published primary (and the frozen route the
   published placebo) to 1e-6; if not, the report says 'not evaluable', no joint coefficient
   is computed, and that is final;
7. the joint regression and the declared reading; the report is written once (mode 'x').
   I1.CHX.v1's report.json is never opened for writing.

Every invocation appends to ``twocontrast_attempts.jsonl`` (start, refusal or failure with its
step, message and whether heads were read, or the outcome with the report sha256; never a
statistic). A crash after step 5 is re-run at most once with unchanged code; the re-run is a
second read of the same heads and its attempt records it.

    python dataprep_staging/chx2c/score_chx2c.py --verify-registration
    python dataprep_staging/chx2c/score_chx2c.py look      # the lead, once, after registering
    python dataprep_staging/chx2c/score_chx2c.py score     # once, after the look
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
import traceback
import uuid
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import chx2c_lib as C  # noqa: E402  (puts ch_x, n2_policy and dataprep_staging on sys.path)
import chx_io as IO  # noqa: E402
import chx_lib as L  # noqa: E402

SCORER_VERSION = "1.1.0"
REPO = IO.REPO                       # data paths resolve here (tests point it at tmp_path)


class Refused(SystemExit):
    """A step refused; nothing past it ran."""


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def rp(p: str | Path) -> Path:
    return Path(p) if Path(p).is_absolute() else REPO / p


def _code_sha(p: str) -> str:
    return IO.sha_or_missing(C.STAGING.parent / p)


def frozen_code() -> dict:
    return {p: _code_sha(p) for p in (*C.OWN_FROZEN, *C.PARENT_FROZEN)}


def jsonable(o):
    """JSON-safe copy: numpy scalars and arrays to Python, non-finite floats to None."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(float(o)) else None
    return o


# ----------------------------------------------------------------------------- attempts
def log_attempt(rec: dict) -> None:
    p = rp(C.ATTEMPTS)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(jsonable({"time": _now(), **rec}), sort_keys=True, default=str)
                + "\n")


# ----------------------------------------------------------------------------- step 1
def _pdir(prereg_dir):
    from harness import prereg3

    return Path(prereg3.PREREG_DIR if prereg_dir is None else prereg_dir)


def _kw(prereg_dir, rows_dir) -> dict:
    return {"prereg_dir": _pdir(prereg_dir)} | ({} if rows_dir is None else
                                               {"rows_dir": rows_dir})


def followups_of_parent(prereg_dir) -> list[str]:
    """Registered ids whose document says ``followup_of`` I1.CHX.v1 (any name)."""
    from harness import prereg3

    d = _pdir(prereg_dir)
    out = []
    for r in prereg3.read_chained(d / "INDEX.jsonl"):
        p = d / f"{r['id']}.json"
        if not p.is_file():
            continue
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(doc, dict) and doc.get("followup_of") == C.PARENT_ID:
            out.append(r["id"])
    return sorted(out)


def check_followup_limit(doc: dict, parent_doc: dict, prereg_dir) -> dict:
    """The parent's exploratory_followups_max against the registered follow-ups (this
    declaration counts; its pre-declared repair ids, if any, are exempt)."""
    limit = int(parent_doc["attempt_policy"]["exploratory_followups_max"])
    exempt = set((doc.get("attempt_policy") or {}).get("repair_ids") or [])
    regs = followups_of_parent(prereg_dir)
    counted = [i for i in regs if i not in exempt]
    if C.ID not in counted:
        counted.append(C.ID)
    if len(counted) > limit:
        raise Refused(f"REFUSED (step 1): {C.PARENT_ID} allows {limit} exploratory "
                      f"follow-up(s) (attempt_policy.exploratory_followups_max); registered: "
                      f"{sorted(counted)}")
    return {"limit": limit, "counted": sorted(counted), "exempt": sorted(exempt)}


def verify_registration(prereg_dir=None, rows_dir=None) -> tuple[dict, str]:
    from harness import prereg3

    try:
        doc, sha = prereg3.load_prereg(C.ID, **_kw(prereg_dir, rows_dir))
    except prereg3.PreregistrationError as err:
        raise Refused(f"REFUSED (step 1): {C.ID} is not registered or not intact: "
                      f"{err}") from err
    if doc.get("kind") != "declaration" or doc.get("followup_of") != C.PARENT_ID:
        raise Refused(f"REFUSED (step 1): {C.ID} is not a declaration with followup_of "
                      f"{C.PARENT_ID}")
    now = frozen_code()
    bad = {p: (doc["frozen_code"].get(p), h) for p, h in now.items()
           if doc["frozen_code"].get(p) != h}
    if bad:
        raise Refused(f"REFUSED (step 1): frozen code differs from the declaration: {bad}")
    changed = prereg3.changed_inputs(doc)
    if changed:
        raise Refused(f"REFUSED (step 1): registered inputs changed: {changed}")
    return doc, sha


# ----------------------------------------------------------------------------- the look
def find_look(doc: dict, prereg_dir) -> dict | None:
    """The lead's after_the_fact look for this declaration, logged after its registration."""
    from harness import prereg3

    created = prereg3.parse_time(doc["created"])
    for lk in prereg3.read_chained(_pdir(prereg_dir) / "looks.jsonl"):
        if (lk.get("purpose") == C.LOOK_PURPOSE and lk.get("prereg_id") == C.ID
                and lk.get("wp") == C.WP and set(C.LOOK_UNITS) <= set(lk.get("units") or [])
                and lk.get("data_scope") == C.LOOK_SCOPE and not lk.get("seeded")
                and prereg3.parse_time(lk["time"]) >= created):
            return {k: lk.get(k) for k in ("time", "logged", "wp", "purpose", "units",
                                           "data_scope", "prereg_id", "note")}
    return None


# ----------------------------------------------------------------------------- step 2
def load_panel(path: Path) -> dict:
    """Only the declared model/metadata keys of chx_panel.npz."""
    z = np.load(path, allow_pickle=False)
    miss = [k for k in C.PANEL_KEYS if k not in z.files]
    if miss:
        raise Refused(f"REFUSED (step 2): panel lacks {miss}")
    return {k: z[k] for k in C.PANEL_KEYS}


def check_panel(P: dict, panel_path: Path, parent_doc: dict, parent_report: dict) -> list[str]:
    probs = []
    if IO.sha256(panel_path) != parent_report.get("panel_sha256"):
        probs.append("panel sha256 differs from the parent report's panel_sha256")
    sid = sorted(np.asarray(P["sid_ch"]).astype(str))
    if sid != sorted(parent_doc["screens"]["ids"]):
        probs.append("panel screens != the parent's declared 23 ids")
    if [str(m) for m in P["months"]] != L.month_list(*C.K.CHX_WINDOW):
        probs.append("panel months are not the parent's window")
    W, T = len(sid), len(P["months"])
    if np.asarray(P["block_ch"]).shape != (W,):
        probs.append("block_ch is not one block per screen")
    for k in ("A_ch", "B_ch", "C_ch", "c_ch", "c_placebo_ch"):
        a = np.asarray(P[k])
        if a.shape != (W, T) or not np.isfinite(a).all():
            probs.append(f"{k} is not a finite ({W}, {T}) array")
    if not probs:
        if not np.array_equal(P["c_ch"], P["A_ch"] - P["B_ch"]):
            probs.append("c_ch != A_ch - B_ch")
        if not np.array_equal(P["c_placebo_ch"], P["A_ch"] - P["C_ch"]):
            probs.append("c_placebo_ch != A_ch - C_ch")
    return probs


# ----------------------------------------------------------------------------- shared steps
def _blocks(P: dict) -> dict:
    sid = np.asarray(P["sid_ch"]).astype(str)
    return dict(zip(sid, np.asarray(P["block_ch"]).astype(str), strict=True))


def _write(rep: dict) -> str:
    out = rp(C.REPORT)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as f:
        json.dump(jsonable(rep), f, indent=1, allow_nan=False)
    return IO.sha256(out)


def prepare(prereg_dir, rows_dir, panel, stage: dict) -> dict:
    """Steps 0-3 (model only, no head): registration, parent, panel, collinearity."""
    from harness import prereg3

    if rp(C.REPORT).exists():
        raise Refused(f"REFUSED (step 0): {C.REPORT} exists; {C.ID} is scored once")
    if not rp(C.PARENT_REPORT).is_file():
        raise Refused(f"REFUSED (step 0): {C.PARENT_REPORT} is missing")
    stage["step"] = 1
    doc, sha = verify_registration(prereg_dir, rows_dir)
    try:
        parent_doc, parent_sha = prereg3.load_prereg(C.PARENT_ID, **_kw(prereg_dir, rows_dir))
    except prereg3.PreregistrationError as err:
        raise Refused(f"REFUSED (step 1): {C.PARENT_ID} is not intact: {err}") from err
    limit = check_followup_limit(doc, parent_doc, prereg_dir)
    stage["step"] = 2
    parent_report = json.loads(rp(C.PARENT_REPORT).read_text(encoding="utf-8"))
    if parent_report.get("prereg_id") != C.PARENT_ID or not parent_report.get("direction"):
        raise Refused("REFUSED (step 2): the parent report is not I1.CHX.v1's direction pass")
    panel_path = rp(C.PANEL) if panel is None else Path(panel)
    P = load_panel(panel_path)
    probs = check_panel(P, panel_path, parent_doc, parent_report)
    if probs:
        raise Refused("REFUSED (step 2): " + "; ".join(probs))
    stage["step"] = 3
    months = [str(m) for m in P["months"]]
    sid = list(np.asarray(P["sid_ch"]).astype(str))
    col = C.collinearity_model_only(P["c_ch"], P["c_placebo_ch"], sid, months)
    rep = {"id": C.ID, "followup_of": C.PARENT_ID, "label": doc.get("label"),
           "prereg_sha256": sha, "parent_prereg_sha256": parent_sha,
           "parent_report_sha256": IO.sha256(rp(C.PARENT_REPORT)),
           "scorer_version": SCORER_VERSION, "created": _now(),
           "panel": str(C.PANEL), "panel_sha256": IO.sha256(panel_path),
           "panel_keys_read": list(C.PANEL_KEYS), "form": C.FORM, "followup_limit": limit,
           "regressors": dict(zip(C.REGRESSOR_NAMES, C.REGRESSORS, strict=True)),
           "dependent": "obs - B_ch (CHX's own)", "cannot_promote": C.CANNOT_PROMOTE,
           "e25_status": C.E25_STATUS, "collinearity_model_only": col}
    return {"doc": doc, "P": P, "parent_report": parent_report, "months": months,
            "sid": sid, "col": col, "rep": rep}


def _collinear_stop(ctx: dict, attempt: str, logged_look: bool) -> dict:
    rep = ctx["rep"] | {"attempt": attempt, "heads_read": False, "look": None,
                        "look_logged": logged_look, "pregate": None, "joint": None,
                        "reading": C.reading(None, ctx["col"], None)}
    h = _write(rep)
    log_attempt({"attempt": attempt, "event": "collinear_stop", "heads_read": False,
                 "look_logged": logged_look, "report_sha256": h})
    print(f"{C.ID}: {rep['reading']['e25_note']} (model only; no look, no head) -> "
          f"{C.REPORT}")
    return rep


def _guarded(command: str, fn):
    attempt = uuid.uuid4().hex[:12]
    stage = {"step": 0, "heads_read": False}
    log_attempt({"attempt": attempt, "event": "start", "command": command,
                 "scorer_version": SCORER_VERSION,
                 "code": {p: _code_sha(p) for p in C.OWN_FROZEN}})
    try:
        return fn(attempt, stage)
    except Refused as err:
        log_attempt({"attempt": attempt, "event": "refused", "step": stage["step"],
                     "heads_read": stage["heads_read"], "message": str(err)})
        raise
    except Exception as err:
        log_attempt({"attempt": attempt, "event": "engineering_failure", "step": stage["step"],
                     "heads_read": stage["heads_read"],
                     "message": f"{type(err).__name__}: {err}",
                     "where": traceback.format_exc(limit=3).splitlines()[-3:]})
        raise


# ----------------------------------------------------------------------------- commands
def record_look(prereg_dir=None, rows_dir=None, panel: Path | None = None) -> dict:
    """The lead's step after registration: the model-only collinearity first; if it fixes
    'not separable' the report is written and NO look is logged; otherwise the
    after_the_fact look is logged (refused twice)."""
    from harness import prereg3

    def body(attempt, stage):
        ctx = prepare(prereg_dir, rows_dir, panel, stage)
        if find_look(ctx["doc"], prereg_dir) is not None:
            raise Refused(f"REFUSED: the {C.LOOK_PURPOSE} look for {C.ID} is already logged")
        if not ctx["col"]["separable"]:
            return {"collinear_stop": True, "report": _collinear_stop(ctx, attempt, False)}
        rec = prereg3.record_look(list(C.LOOK_UNITS), C.LOOK_SCOPE, C.WP, C.LOOK_PURPOSE,
                                  prereg_id=C.ID, note=C.LOOK_NOTE,
                                  **_kw(prereg_dir, rows_dir))
        log_attempt({"attempt": attempt, "event": "look_logged", "heads_read": False,
                     "look_time": rec["time"]})
        return {"collinear_stop": False, "look": rec}

    return _guarded("look", body)


def run_score(prereg_dir=None, rows_dir=None, panel: Path | None = None,
              heads_loader=None) -> dict:
    """The single scoring session. ``heads_loader`` defaults to ``chx_io.l1_monthly`` (tests
    pass a synthetic one)."""
    import score_chx as SC  # frozen; only score_panel is called

    def body(attempt, stage):
        ctx = prepare(prereg_dir, rows_dir, panel, stage)
        rep, P, col = ctx["rep"] | {"attempt": attempt}, ctx["P"], ctx["col"]
        if not col["separable"]:                    # `look` would have stopped; defensive
            return _collinear_stop(ctx, attempt, False)
        stage["step"] = 4
        look = find_look(ctx["doc"], prereg_dir)
        if look is None:
            raise Refused(f"REFUSED (step 4): no '{C.LOOK_PURPOSE}' look on {C.UNIT} "
                          f"({C.LOOK_SCOPE}) for {C.ID} after its registration; run "
                          "'score_chx2c.py look' first")
        rep["look"] = look
        stage["step"] = 5
        loader = IO.l1_monthly if heads_loader is None else heads_loader
        obs, flag, notes = loader(ctx["sid"], ctx["months"])
        stage["heads_read"] = True
        rep.update({"heads_read": True, "guard_notes": notes})
        stage["step"] = 6
        sid, months, blocks = ctx["sid"], ctx["months"], _blocks(P)
        frozen = C.frozen_numbers(SC.score_panel(P, obs, flag))
        own1 = C.joint_fit(C.joint_rows(sid, months, obs, flag, P["B_ch"], [P["c_ch"]]),
                           blocks)
        pg = C.pregate(C.published_numbers(ctx["parent_report"]), frozen, own1)
        rep["pregate"] = pg
        if not pg["passed"]:
            rep.update({"joint": None, "reading": C.reading(None, col, pg)})
            h = _write(rep)
            log_attempt({"attempt": attempt, "event": "pregate_failed", "heads_read": True,
                         "report_sha256": h})
            print(f"{C.ID}: pre-gate failed; {rep['reading']['e25_note']} -> {C.REPORT}")
            return rep
        stage["step"] = 7
        fit = C.joint_fit(C.joint_rows(sid, months, obs, flag, P["B_ch"],
                                       [P["c_ch"], P["c_placebo_ch"]]), blocks)
        b_yl, ci_yl = fit["beta"][1], fit["ci90"][1]
        rep["joint"] = {
            **fit,
            "changhua": {"beta": fit["beta"][0], "ci90": fit["ci90"][0]},
            "yunlin": {"beta": b_yl, "ci90": ci_yl,
                       "report_only_k_yl": {"beta": b_yl + 1.0,
                                            "ci90": [ci_yl[0] + 1.0, ci_yl[1] + 1.0],
                                            "meaning": "observed response to Yunlin's pattern "
                                                       "in units of the model's own (linear "
                                                       "superposition)"}},
            "report_only_changhua_point_in_ratio_band": bool(
                np.isfinite(fit["beta"][0])
                and 1 / C.BS.RATIO_BAND <= fit["beta"][0] <= C.BS.RATIO_BAND),
            "size_clause": "unchanged (CHX's size clause failed); the band flag above is "
                           "report-only and reads nothing"}
        rep["reading"] = C.reading(fit, col, pg)
        h = _write(rep)
        log_attempt({"attempt": attempt, "event": "scored", "heads_read": True,
                     "report_sha256": h})
        print(f"{C.ID}: {rep['reading']['e25_note']} -> {C.REPORT}")
        return rep

    return _guarded("score", body)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--verify-registration", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("look", help="the lead: model-only collinearity, then the look (once)")
    sub.add_parser("score", help="the single scoring session")
    a = ap.parse_args(argv)
    if a.verify_registration:
        _, sha = verify_registration()
        print(f"{C.ID} registered and intact (sha256 {sha[:16]}); frozen code matches")
        return 0
    if a.cmd == "look":
        out = record_look()
        if not out["collinear_stop"]:
            print(out["look"]["time"], out["look"]["purpose"], out["look"]["units"])
        return 0
    if a.cmd == "score":
        run_score()
        return 0
    ap.error("use --verify-registration, look or score")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
