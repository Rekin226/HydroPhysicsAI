"""Score FE.encompass_2026-12.v1 once, in the 2028-01-15..31 session, after E02c, E02d and E04.

Observations. FE reads no heads table of its own: its rows are E02's step-(5) output
(``python -m hydrophysics.twin.prospective score`` on results/twin/prospective_2026-11) and
E04's step-(5) output (the same command on results/twin/prospective_2026-11_hybrid), each
verified against its registered predictions by ``score_e02c.load_e02_step5``. Chain of custody
(review S6): E02's evaluation.csv must be the one E02c's report scored (its
e02_step5['evaluation.csv_sha256']) on the same window product E02c's report names; E04's rows
must cover exactly E02's well-months at the shared wells with identical observed_m (1e-9 m).
Any disagreement is REFUSED or an engineering failure, never a verdict.

Order of a real run: (0) the attempt is logged; the seed is the registered one; (1) FE must be
registered (INDEX.jsonl), else REFUSED; (2) not before 2028-01-15 and not after 2028-01-31,
else REFUSED; (3) the report is written once; (4) registration, frozen code and seed intact;
(4a) E02c's report must be the canonical one (L.E02C_REPORT or a byte-identical copy) with
its done-status passing, else REFUSED; (4b) pre-declared: if E02c's outcome carries no e02_step5 evaluation sha256 (not scorable),
FE writes a final 'not_evaluable (E02c not scorable)' report and stops (never waits, never an
unchained product); (5) the window product must exist with a readable MANIFEST and be E02c's
product, else
REFUSED (review S5: a missing or mistyped path never writes a verdict); E02d's window-A rule
through the GOV module (prospective_scorer, and a declared prospective_scorer_window must
cover 2026-12..2027-10): if it fails, a not_evaluable report, nothing read, no look;
(6) session order: E02c's report, E02d's report (or lead-written records) and E04's and E02's
step-(5) rows must exist, else REFUSED; (7) prereg3.assert_can_score (logs the score look);
(8) only then are the step-(5) rows read. Note: the plan's order is E02c -> E02d -> E04 -> FE,
but E02c's own session order scores E04's step (5) before E02c, so FE enforces "all three done,
FE last" (a disclosed deviation). Role: gating iff decidability.json was written under this
registration before 2026-11-30T16:00Z with G0 passed AND the registration was created by
2026-11-25 (review B2); otherwise report-only.

  score_fe.py score --seed N --prereg-id FE.encompass_2026-12.v1 --window-product <dir>
      --e02-score-dir <E02 step-(5) out> --e04-score-dir <E04 step-(5) out>
      --e02c-report <E02c report.json> --e02d-report <E02d report.json>
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent / "gov_heads_window")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import decide_fe as D  # noqa: E402  (puts FV.stat.v2 and e02c on sys.path)
import fe_lib as L  # noqa: E402
import heads_window as HW  # noqa: E402  (GOV module API; bound by sha256)
import score_e02c as SC  # noqa: E402  (E02c's scorer, bound; load_e02_step5)

O = D.O  # noqa: E741
SCORER_VERSION = "2.0.0"
E02D_TEST = "E02dP.G_idio_2026-12"
GOV_KEY = f"{L.E02D_ID}:A"        # E02d's window-A precondition (2026-12..2027-10)
QUIET_VERDICTS = (SC.NOT_SCORABLE, "not_evaluable")


# ----------------------------------------------------------------------------- pure scoring
def role_from(dec: dict | None, decl_sha: str, created: str | None, *, doc: dict | None = None,
              dec_path=None) -> tuple[str, str]:
    """(role, why) from the decidability record and the registration time (B2 hard rule).
    With ``doc`` and ``dec_path`` the record's traceability is verified first (the registered
    FV v2 npz and ZR0 sha256s, and the attempts.jsonl success row)."""
    cap, why_cap = D.role_cap(created)
    if dec is None:
        return "report_only", "not measured before the deadline"
    if dec.get("declaration_sha256") != decl_sha:
        raise L.EngineeringFailure("decidability.json was not written under this registration")
    if doc is not None:
        D.verify_decidability(dec, Path(dec_path), doc, decl_sha)
    if cap != "gating":
        return "report_only", why_cap
    if not dec.get("on_time"):
        return "report_only", "measured after the deadline"
    if dec.get("role") != "gating" or not (dec.get("g0") or {}).get("pass"):
        return "report_only", "G0 not passed"
    return "gating", "G0 passed"


def score_frame(frame: pd.DataFrame, *, seed: int, role: str, why_role: str = "") -> dict:
    """lambda_P, its reading and wording, plus report-only lambda_C, one-contrast, raw-level
    and per-county fits, on rows (station_id, date, observed_m, stat_m, phys_m, clim_m).
    Pure: no file, no look."""
    seed = D.check_seed(seed)
    pn = L.panel(frame)
    xp, xc, y, m, W = pn["xp"], pn["xc"], pn["y"], pn["m"], len(pn["ids"])
    rep: dict = {"wells_scored": W, "rows_scored": pn["rows"], "role": role,
                 "role_why": why_role, "delta": L.DELTA, "rule": L.READING_RULE}
    if W < L.MIN_WELLS:
        rd = {"reading": "not_evaluable", "small": False}
        why = f"{W} wells with >= {L.MIN_MONTHS} months, fewer than {L.MIN_WELLS}"
        rep.update({"primary": None, "reading": rd, "partial_contrast_rms_m": None,
                    "wording": L.wording(rd, None, n_wells=W, role=role, why=why),
                    "why": why})
        return rep
    rng = D.rng_for(D.STREAMS["score"], 0, 0, seed=seed)
    wts = L.boot_weights(W, L.BOOT_B, rng)
    est = L.estimate([xp, xc], y, m, weights=wts, rng=None)
    xr = L.partial_rms(xp, xc, m)
    rd = L.reading(est["P"], bad_share=est["bad_share"], informative=xr >= L.X_RMS_MIN_M,
                   evaluable=True)
    why = ("more than 1 % degenerate replicates" if est["bad_share"] > L.MAX_BAD_REPLICATES
           else why_role)
    rep.update({
        "primary": {**est["P"], "bad_share": est["bad_share"],
                    "estimand": "encompassing coefficient lambda_P (within wells, two "
                                "contrasts)"},
        "lambda_c_report_only": est["C"], "partial_contrast_rms_m": xr,
        "contrast_rms_report_only_m": L.contrast_rms(xp, m), "reading": rd,
        "wording": L.wording(rd, est["P"], n_wells=W, role=role, why=why),
        "one_contrast_report_only": L.estimate([xp], y, m, weights=wts, rng=None)["P"],
        "raw_level_report_only": {k: v for k, v in L.estimate(
            [xp, xc], y, m, weights=wts, rng=None, demean=False).items() if k in ("P", "C")}})
    cty = np.array([L.county_of(s) for s in pn["ids"]])
    rep["per_county_report_only"] = {}
    for j, c in enumerate(("Yunlin", "Changhua"), start=1):
        sel = cty == c
        if sel.sum() < 2:
            continue
        r = D.rng_for(D.STREAMS["score"], j, 0, seed=seed)
        e = L.estimate([xp[sel], xc[sel]], y[sel], m[sel], rng=r)
        rep["per_county_report_only"][c] = {"wells": int(sel.sum()), "P": e["P"],
                                            "C": e["C"]}
    return rep


def _monthly(f: pd.DataFrame) -> pd.DataFrame:
    f = f.copy()
    f["station_id"] = f["station_id"].astype(str)
    f["date"] = pd.to_datetime(f["date"]).dt.to_period("M").dt.to_timestamp()
    return f


def frame_from_step5(e02: pd.DataFrame, e04: pd.DataFrame, shared: list[str]
                     ) -> tuple[pd.DataFrame, dict]:
    """The shared wells' rows: E02's and E04's step-(5) rows must hold exactly the same
    well-months there with identical observed_m (chain of custody, review S6)."""
    a = _monthly(e02)[["station_id", "date", "observed_m", "predicted_m"]].rename(
        columns={"predicted_m": "phys_m"})
    b = _monthly(e04)[["station_id", "date", "observed_m", "predicted_m", "baseline_m"]].rename(
        columns={"predicted_m": "stat_m", "baseline_m": "clim_m", "observed_m": "observed_e04"})
    a, b = a[a["station_id"].isin(shared)], b[b["station_id"].isin(shared)]
    ka = set(zip(a["station_id"], a["date"], strict=True))
    kb = set(zip(b["station_id"], b["date"], strict=True))
    if ka != kb:
        raise L.EngineeringFailure(f"E02's and E04's step-(5) rows differ at the shared wells "
                                   f"({len(ka - kb)} only in E02, {len(kb - ka)} only in E04)")
    j = a.merge(b, on=["station_id", "date"], how="inner")
    if j.empty:
        raise L.EngineeringFailure("E02's and E04's step-(5) rows share no well-month")
    if not np.allclose(j["observed_m"].to_numpy(float), j["observed_e04"].to_numpy(float),
                       rtol=0, atol=1e-9):
        raise L.EngineeringFailure("E02's and E04's step-(5) observed_m differ")
    prov = {"shared_rows": int(len(j)), "shared_wells": int(j["station_id"].nunique()),
            "e02_rows_all": int(len(e02)), "e04_rows_all": int(len(e04))}
    return j.drop(columns="observed_e04"), prov


# ----------------------------------------------------------------------------- order, chain
def _same_path(a, b) -> bool:
    try:
        return Path(a).resolve() == O.rp(b).resolve()
    except (TypeError, OSError):
        return False


def window_product_status(product, e02c_report) -> dict:
    """S5/S6: the product exists with a readable MANIFEST and is E02c's product. Not ok means
    REFUSED (nothing written)."""
    from common.manifest import load_manifest

    if product is None or not Path(product).exists():
        return {"ok": False, "why": f"window product {product!r} does not exist"}
    try:
        load_manifest(product)
    except (OSError, ValueError) as exc:
        return {"ok": False, "why": f"window product has no readable MANIFEST ({exc})"}
    rec, why = SC._read_record(e02c_report, L.E02C_ID)
    if rec is None:
        return {"ok": False, "why": f"E02c report missing: {why}"}
    if rec.get("window_product") is None:
        return {"ok": False, "why": "E02c's report names no window_product"}
    if not _same_path(product, rec["window_product"]):
        return {"ok": False, "why": "the window product is not the one E02c's report names"}
    return {"ok": True, "e02c_window_product": rec.get("window_product")}


def e02c_canonical(prereg_dir, e02c_report) -> dict:
    """E02c's report must be the canonical one: the path FE registers (E02c's documented
    --out, L.E02C_REPORT) or a byte-identical copy of it, and E02c's own done-status (its score
    looks when it claims a score) must pass. Not ok means REFUSED; a random JSON naming E02c's
    id can never write FE's report."""
    if e02c_report is None or not Path(e02c_report).is_file():
        return {"ok": False, "why": f"E02c report {e02c_report!r} missing"}
    canon = O.rp(L.E02C_REPORT)
    same = _same_path(e02c_report, L.E02C_REPORT)
    copy = canon.is_file() and O.sha256(canon) == O.sha256(e02c_report)
    if not (same or copy):
        return {"ok": False, "why": f"the E02c report is not the canonical {L.E02C_REPORT} "
                                    "(nor a byte-identical copy)"}
    done = SC.e02c_done_status(prereg_dir, e02c_report)
    if not done["ok"]:
        return {"ok": False, "why": done["why"]}
    return {"ok": True, "path": str(e02c_report), "sha256": O.sha256(e02c_report)}


def e02c_not_scorable(e02c_report) -> dict | None:
    """The pre-declared final record when E02c's outcome carries no E02 step-(5) chain (a
    not-scorable record, or a report that scored no step-(5) rows); None otherwise. A missing
    E02c report is not this case: the scorer then refuses (E02c has not finished)."""
    rec, _ = SC._read_record(e02c_report, L.E02C_ID)
    if rec is None:
        return None
    if ((rec.get("e02_step5") or {}).get("evaluation.csv_sha256")):
        return None
    return {"prereg_id": L.ID, "verdict": "not_evaluable",
            "reading": {"reading": "not_evaluable", "small": False},
            "why": "not_evaluable (E02c not scorable): E02c's outcome carries no E02 step-(5) "
                   "evaluation sha256, so FE has no chained rows; final, no value read, no "
                   "look logged",
            "e02c_record": {"path": str(e02c_report), "sha256": O.sha256(e02c_report),
                            "verdict": rec.get("verdict")},
            "wording": L.wording({"reading": "not_evaluable", "small": False}, None,
                                 n_wells=0, role="gating", why="E02c not scorable")}


def order_status(prereg_dir, e02c_report, e02d_report, e02_dir, e04_dir) -> dict:
    """E02c, E02d and E04 done before FE (reads those runs' own records, never a value)."""
    st = {"e02c": SC.e02c_done_status(prereg_dir, e02c_report)}
    rec, why = SC._read_record(e02d_report, L.E02D_ID)
    if rec is None:
        st["e02d"] = {"ok": False, "why": f"E02d report missing: {why}"}
    elif rec.get("window") == "A" and rec.get("verdict") in QUIET_VERDICTS:
        st["e02d"] = {"ok": True, "verdict": rec.get("verdict"),
                      "sha256": O.sha256(e02d_report)}
    elif rec.get("window") != "A":
        st["e02d"] = {"ok": False, "why": "the E02d record names no window, or not window A "
                                          "(2026-12..2027-10)"}
    elif SC.assert_order([(L.E02D_ID, E02D_TEST)], prereg_dir):
        st["e02d"] = {"ok": False, "why": "E02d report claims a score but its look is missing"}
    else:
        st["e02d"] = {"ok": True, "verdict": rec.get("verdict", "scored"),
                      "sha256": O.sha256(e02d_report)}
    for key, d, reg in (("e02_step5", e02_dir, L.E02_DIR), ("e04_step5", e04_dir, L.E04_DIR)):
        gone = [n for n in SC.E02_STEP5_FILES if d is None or not (Path(d) / n).is_file()]
        if gone:
            st[key] = {"ok": False, "why": f"{reg} step (5) has not run: {gone} missing"}
            continue
        man = json.loads((O.rp(reg) / "manifest.json").read_text(encoding="utf-8"))
        r5 = json.loads((Path(d) / "report.json").read_text(encoding="utf-8"))
        st[key] = ({"ok": True, "dir": str(d)} if r5.get("registration_manifest") == man else
                   {"ok": False, "why": f"{d}/report.json does not carry {reg}'s manifest"})
    if st["e02_step5"]["ok"]:
        rec, _ = SC._read_record(e02c_report, L.E02C_ID)
        want = ((rec or {}).get("e02_step5") or {}).get("evaluation.csv_sha256")
        got = O.sha256(Path(e02_dir) / "evaluation.csv")
        if want != got:
            st["e02_chain"] = {"ok": False, "why": "E02's evaluation.csv is not the one E02c's "
                                                   "report scored"}
        else:
            st["e02_chain"] = {"ok": True, "evaluation_csv_sha256": got}
    st["ok"] = all(v["ok"] for v in st.values() if isinstance(v, dict))
    return st


# ----------------------------------------------------------------------------- runner
def _step5(reg: str, d) -> tuple[pd.DataFrame, dict]:
    pred = pd.read_csv(O.rp(reg) / "predictions.csv", dtype={"station_id": str})
    man = json.loads((O.rp(reg) / "manifest.json").read_text(encoding="utf-8"))
    f, _, prov = SC.load_e02_step5(Path(d), pred, O.rp(reg) / "protocol.json", man)
    return f, prov


def run_score(a, *, prereg_dir=None, rows_dir=None, now=None) -> dict:
    from harness import prereg3

    pdir = Path(prereg_dir) if prereg_dir else prereg3.PREREG_DIR
    rdir = Path(rows_dir) if rows_dir else prereg3.LEDGER_ROWS
    out = O.rp(a.out)
    ap = out.parent / L.ATTEMPTS_NAME
    n = D.attempt_no(ap)
    D.log_attempt(ap, {"event": "start", "stage": "score", "attempt": n, "seed": a.seed,
                       "prereg_id": a.prereg_id, "code_sha256": O.sha256(Path(__file__))})
    try:
        seed = D.check_seed(a.seed)
        idx = D.index_record(a.prereg_id, pdir) if a.prereg_id == L.ID else None
        if idx is None:
            raise L.Refused(f"{a.prereg_id} is not registered: FE scores only under its "
                            "registration")
        t = now or dt.datetime.now(dt.timezone.utc)
        if t < dt.datetime.fromisoformat(L.NOT_BEFORE):
            raise L.Refused(f"not before {L.NOT_BEFORE[:10]}")
        if t >= dt.datetime.fromisoformat(L.NO_LATER_THAN_END):
            raise L.Refused(f"no later than {L.NO_LATER_THAN}: the session has closed (a "
                            "lead-written not-scorable record replaces the report)")
        if out.exists():
            raise L.Refused(f"{out} exists: FE is scored once")
        doc, dsha = D.load_registered(a.prereg_id, seed, pdir, rdir)
        if doc.get("frozen_code", {}).get(L.SCORER) != O.sha256(Path(__file__)):
            raise L.Refused("this scorer is not the one the registration froze")
        canon = e02c_canonical(pdir, a.e02c_report)
        if not canon["ok"]:
            raise L.Refused(f"E02c report: {canon['why']}; nothing read, no look logged")
        e02c = e02c_not_scorable(a.e02c_report)
        if e02c is not None:
            O._write_json_once(out, e02c)
            D.log_attempt(ap, {"event": "success", "stage": "score", "attempt": n,
                               "report_sha256": O.sha256(out), "verdict": "not_evaluable"})
            return e02c
        wp = window_product_status(a.window_product, a.e02c_report)
        if not wp["ok"]:
            raise L.Refused(f"window product: {wp['why']}; nothing read, no look logged")
        granted, why = HW.check_precondition(GOV_KEY, a.window_product)
        if not granted:
            rep = {"prereg_id": L.ID, "verdict": "not_evaluable",
                   "why": f"the window heads product does not grant 2026-12..2027-10 under "
                          f"E02d's window-A rule ({why}); no value read, no look logged",
                   "window_product": str(a.window_product)}
            O._write_json_once(out, rep)
            D.log_attempt(ap, {"event": "success", "stage": "score", "attempt": n,
                               "report_sha256": O.sha256(out), "verdict": "not_evaluable"})
            return rep
        order = order_status(pdir, a.e02c_report, a.e02d_report, a.e02_score_dir,
                             a.e04_score_dir)
        if not order["ok"]:
            bad = {k: v.get("why") for k, v in order.items() if isinstance(v, dict)
                   and not v["ok"]}
            raise L.Refused(f"session order / chain: {bad}; nothing read, no look logged")
        dpath = O.rp(a.decidability)
        dec = json.loads(dpath.read_text(encoding="utf-8")) if dpath.is_file() else None
        role, why_role = role_from(dec, dsha, idx.get("created"), doc=doc, dec_path=dpath)
        ok = prereg3.assert_can_score(L.ID, L.WP, prereg_dir=pdir, rows_dir=rdir)
        e02, p02 = _step5(L.E02_DIR, a.e02_score_dir)
        e04, p04 = _step5(L.E04_DIR, a.e04_score_dir)
        shared = doc["tests"][0]["external_units"][L.UNIT]["ids"]
        frame, pj = frame_from_step5(e02, e04, shared)
        rep = score_frame(frame, seed=seed, role=role, why_role=why_role)
        rep.update({"prereg_id": L.ID, "scorer_version": SCORER_VERSION,
                    "registration_sha256": dsha, "order": order, "window_product": wp,
                    "precondition": {"gov_key": GOV_KEY, "why": why},
                    "decidability": ({"path": str(a.decidability), "sha256": O.sha256(dpath)}
                                     if dec else None),
                    "step5": {"e02": p02, "e04": p04, "join": pj},
                    "post_registration_looks": ok["post_registration_looks"],
                    "unverifiable_prior_looks": ok["unverifiable_prior_looks"],
                    "created": O._now()})
        O._write_json_once(out, rep)
    except (L.Refused, L.EngineeringFailure, SC.ScoringRefused) as err:
        D.log_attempt(ap, {"event": type(err).__name__, "stage": "score", "attempt": n,
                           "message": str(err)})
        raise
    except Exception as err:
        D.log_attempt(ap, {"event": "error", "stage": "score", "attempt": n,
                           "message": f"{type(err).__name__}: {err}"})
        raise
    D.log_attempt(ap, {"event": "success", "stage": "score", "attempt": n,
                       "report_sha256": O.sha256(out), "declaration_sha256": dsha})
    return rep


def main(argv=None, prereg_dir=None, rows_dir=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--seed", type=int, required=True, help="FE's registered seed")
    s.add_argument("--prereg-id", required=True)
    s.add_argument("--window-product", required=True)
    s.add_argument("--e02-score-dir", required=True)
    s.add_argument("--e04-score-dir", required=True)
    s.add_argument("--e02c-report", required=True)
    s.add_argument("--e02d-report", required=True)
    s.add_argument("--decidability", default=L.DECIDABILITY)
    s.add_argument("--out", default=L.REPORT)
    a = ap.parse_args(argv)
    try:
        rep = run_score(a, prereg_dir=prereg_dir, rows_dir=rows_dir)
    except (L.Refused, L.EngineeringFailure, SC.ScoringRefused) as err:
        print(f"REFUSED / ENGINEERING FAILURE: {err}", file=sys.stderr)
        return 1
    rd = (rep.get("reading") or {}).get("reading", rep.get("verdict"))
    print(f"wrote {a.out} (reading {rd}, role {rep.get('role', '-')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
