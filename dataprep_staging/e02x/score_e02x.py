"""Score E02x.ext_transfer_2026-12.v1 (or the fallback E02x.ext_transfer_2027-01.v1) ONCE, after
the window closes, LAST in the 2028-01-15 session (E02c -> E02d -> N3F -> E02x).

Rows. Nothing here reads a heads table: the donors' rows are E02's step-(5) output (run in
E02/E02c's session), the external wells' rows N3F's step-(5) output (E02's scorer on N3F's
registered export, run by N3F's own scorer), both verified against their registered files
and restricted to the window's months.

    r = observed - predicted (E02's centre at donors, N3F's at external wells)
    e = r_ext - median of r over the external well's donors that month (>= 5)
    hit = |e| <= z * sigma_idio (the frozen artefact)

Per gated county (role_by_county 'primary'): E02d's gate (>= 6 months, >= 20 wells, gate-3
upper90 clusters = wells AND point coverage in [0.85, 0.95]). A report-only county (Changhua
when its power failed, Nantou always) is reported with the same numbers, never mapped.
Report-only: coverage by zone and by DR group, signed misses, month-cluster gate-3.

Order of a real run: (0) registered (refuses before registration); every sha256 of the
REGISTERED frozen_code and binds.bound_code equals the current file, and heads_window.py is
GOV's registered 9a953039 before it is executed; (1) date >= not_before; (2) report.json
absent (scored once); (3) E02c, E02d and N3F reports exist (order); (4) the window product's
MANIFEST: a wrong path, an absent or unreadable MANIFEST is REFUSED (logged, retryable); a
readable one failing GOV.heads_window_product.v1's check_precondition with E02x's entry is
not_evaluable (nothing read, no look); (5) the artefact matches the registration (model
output); (6) prereg3.assert_can_score (gated test, then Nantou); (7) only then the step-(5)
rows are read; N3F rows without a step-(5) report.json need an --n3f-report record stating
why evaluate raised and carrying evaluation.csv's sha256. Every
invocation is appended to <out>/attempts.jsonl and results/twin_runs/e02x/attempts.jsonl.

    python -I dataprep_staging/e02x/score_e02x.py score --window primary --window-product <dir> \\
        --step5-dir <E02 step-(5) out> --ext-step5-dir <N3F out>/e02_protocol_run \\
        --e02c-report <f> --e02d-report <f> --n3f-report <f> --out <dir>
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent), str(HERE.parent / "e02d"), str(HERE.parent / "e02c")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import e02xio as io  # noqa: E402
import e02xlib as L  # noqa: E402
import score_e02c as S  # noqa: E402  (E02c's frozen scorer: imported, never edited)

SCORER_VERSION = "1.0.0"
ART_DIR = "results/twin/e02x_relative"
RUN_ATTEMPTS = "results/twin_runs/e02x/attempts.jsonl"
N3F_TEST = "N3FP.ext_level_2026-11"
HW_PATH = "dataprep_staging/gov_heads_window/heads_window.py"
FIELD = "prospective_scorer_window"
_YM = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
NOT_SCORABLE = S.NOT_SCORABLE


# ----------------------------------------------------------------------------- GOV
def gov_entry(window: str) -> dict:
    """E02x's entry for GOV.heads_window_product.v1's PRECONDITIONS (added at call time to a
    private copy of the module; heads_window.py is not edited)."""
    w = L.WINDOWS[window]
    return {"months": (w["first"], w["last"]), "needs_window": True, "code": None,
            "function": "GOV covering rule (check_precondition) plus the declared-window and "
                        "data-end requirements of the registered GOV interpretation",
            "registered_rule": "products serving later E02x must declare parameters."
                               f"{FIELD} covering the scored months and ending no later than "
                               "the product's data"}


#: heads_window.py as GOV.heads_window_product.v1 registered it (its 'inputs' record)
HW_SHA = "9a953039d75fc812fa3d537c80d17cd8c8fa9f90f2b4f70af5d90ddcfa7a4f37"


def _hw(expected_sha: str = HW_SHA):
    """A private module object of heads_window.py (its PRECONDITIONS dict is never the one
    another importer sees), executed only if the file is the one GOV registered."""
    path = io.repo_path(HW_PATH)
    got = io.sha256(path)
    if got != expected_sha:
        raise io.EngineeringFailure(f"{HW_PATH} sha256 {got[:8]} != GOV's registered "
                                    f"{expected_sha[:8]}: not executed")
    spec = importlib.util.spec_from_file_location("e02x_heads_window", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def read_manifest(product_dir) -> dict:
    """The window product's MANIFEST, or Refused (logged, retryable): a wrong path, an absent
    or an unreadable MANIFEST is never a verdict (B2)."""
    from common.manifest import load_manifest

    d = Path(product_dir) if product_dir is not None else None
    if d is None or not d.is_dir():
        raise io.Refused(f"--window-product {product_dir} is not a directory")
    if not (d / "MANIFEST.json").is_file():
        raise io.Refused(f"{d} has no MANIFEST.json")
    try:
        m, _ = load_manifest(d)
    except (OSError, ValueError) as exc:
        raise io.Refused(f"{d}/MANIFEST.json is unreadable ({exc})") from exc
    return m


def precondition(product_dir, window: str, hw_sha: str = HW_SHA) -> tuple[bool, str]:
    """GOV's check_precondition with E02x's entry, then the declared window and data end
    (MANIFEST metadata only). Refused (raises) unless the MANIFEST is readable; (False, why)
    -> not_evaluable only for a readable MANIFEST that fails the grant."""
    m = read_manifest(product_dir)
    hw = _hw(hw_sha)
    key = L.prereg_id(window)
    hw.PRECONDITIONS[key] = gov_entry(window)
    try:
        ok, why = hw.check_precondition(key, product_dir)
    finally:
        hw.PRECONDITIONS.pop(key, None)
    if not ok:
        return False, why
    dec = (m.get("parameters") or {}).get(FIELD)
    if dec is None:
        return False, (f"the MANIFEST declares no parameters.{FIELD}: {L.GOV_ID} requires "
                       "products serving E02x to declare it")
    fwe = m.get("fit_window_end")
    if not (isinstance(fwe, str) and _YM.match(fwe)):
        return False, "fit_window_end is not YYYY-MM"
    if str(dec[1]) > fwe:
        return False, f"{FIELD} ends {dec[1]}, after the product's fit_window_end {fwe}"
    return True, f"{why}; declared {FIELD} {dec} ends no later than fit_window_end {fwe}"


def verify_code(doc: dict, prereg_dir=None) -> dict:
    """S1: every sha256 the REGISTERED document records (frozen_code and binds.bound_code)
    equals the current file, and GOV's registered heads_window.py record equals HW_SHA."""
    want = dict(doc.get("frozen_code") or {})
    want.update(((doc.get("binds") or {}).get("bound_code")) or {})
    if not want:
        raise io.Refused("the registration records no frozen or bound code")
    bad = [f for f, h in sorted(want.items()) if io.sha256(io.repo_path(f)) != h]
    if bad:
        raise io.Refused(f"code changed since registration: {bad}")
    gov, gsha = io.registered(L.GOV_ID, prereg_dir)
    rec = [i for i in gov.get("inputs") or [] if str(i.get("path")).endswith(HW_PATH)]
    if not rec or rec[0].get("sha256") != HW_SHA:
        raise io.Refused(f"GOV's registered {HW_PATH} record is not {HW_SHA[:8]}")
    if want.get(HW_PATH, HW_SHA) != HW_SHA:
        raise io.Refused(f"E02x bound {HW_PATH} at a sha other than GOV's")
    return {"files_checked": len(want), "gov_registration_sha256": gsha,
            "heads_window_sha256": HW_SHA}


# ----------------------------------------------------------------------------- order
def _record(path, pid: str) -> tuple[dict | None, str]:
    return S._read_record(path, pid)


def order_status(prereg_dir, e02c_report, e02d_report, n3f_report) -> dict:
    """E02c, E02d and N3F must be done (a report naming the registration, with its score
    look logged when it scored, or a lead-written not-scorable record). Reads those reports
    only, never a window value."""
    out = {"e02c": S.e02c_done_status(prereg_dir, e02c_report)}
    for name, path, pid, test, scored_key in (
            ("e02d", e02d_report, L.E02D_ID, L.E02D_TEST, "relative"),
            ("n3f", n3f_report, L.N3F_ID, N3F_TEST, "evaluate")):
        rec, why = _record(path, pid)
        if rec is None:
            out[name] = {"ok": False, "why": f"{pid} report missing: {why}"}
            continue
        scored = scored_key in rec
        missing = S.assert_order([(pid, test)], prereg_dir) if scored else []
        if missing:
            out[name] = {"ok": False, "why": f"{pid} report claims a score but its look is "
                                             f"missing: {missing}"}
        else:
            out[name] = {"ok": True, "report": str(path), "sha256": io.sha256(path),
                         "verdict": rec.get("verdict", "scored")}
    out["ok"] = all(v["ok"] for v in out.values() if isinstance(v, dict))
    return out


# ----------------------------------------------------------------------------- rows
def _n3f_files(prereg_dir=None) -> tuple[pd.DataFrame, Path, dict]:
    import wells_e02x as WX

    doc, _ = WX.n3f_registration(prereg_dir)
    files = doc["external_predictions"]["files"]
    d = io.repo_path(doc["external_predictions"]["dir"])
    for n in ("predictions.csv", "protocol.json", "manifest.json"):
        if io.sha256(d / n) != files[n]["sha256"]:
            raise io.EngineeringFailure(f"N3F {n} is not the registered file")
    pred = pd.read_csv(d / "predictions.csv", dtype={"station_id": str})
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    return pred, d / "protocol.json", man


def _e02_files() -> tuple[pd.DataFrame, Path, dict]:
    d = io.repo_path(L.E02_DIR)
    for n, h in (("predictions.csv", L.D.E02_PRED_SHA), ("protocol.json", L.D.E02_PROTOCOL_SHA)):
        if io.sha256(d / n) != h:
            raise io.EngineeringFailure(f"E02 {n} changed")
    pred = pd.read_csv(d / "predictions.csv", dtype={"station_id": str})
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    return pred, d / "protocol.json", man


def verified_rows(step5: Path, pred: pd.DataFrame, proto: Path, man: dict, window: str,
                  thin_record: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Step-(5) rows verified against registered predictions, restricted to the window. With
    a step-(5) report.json: score_e02c.load_e02_step5 (columns to 1e-9 m and validation.
    evaluate reproduced). Without one (N3F's evaluate raised: too few wells) only if
    ``thin_record`` (the --n3f-report record) states why ('evaluate_error') and carries
    evaluation.csv's sha256 ('evaluation_sha256'); then the column check, stated. Otherwise
    refused (S4)."""
    step5 = Path(step5)
    if (step5 / "report.json").is_file():
        try:
            rows, _, prov = S.load_e02_step5(step5, pred, proto, man)
        except S.EngineeringFailure as exc:
            raise io.EngineeringFailure(str(exc)) from exc
    else:
        ev_p = step5 / "evaluation.csv"
        if not ev_p.is_file():
            raise io.Refused(f"{step5} holds neither report.json nor evaluation.csv")
        why = str((thin_record or {}).get("evaluate_error") or "").strip()
        want = (thin_record or {}).get("evaluation_sha256")
        if not why or want != io.sha256(ev_p):
            raise io.Refused(f"{step5} has no report.json: the --n3f-report record must state "
                             "why evaluate raised ('evaluate_error') and carry evaluation.csv's "
                             "sha256 ('evaluation_sha256')")
        ev = pd.read_csv(ev_p, dtype={"station_id": str})
        k = ["station_id", "date"]
        m = ev.merge(pred, on=k, how="left", suffixes=("", "_reg"))
        for c in S.EVAL_CHECK_COLS:
            if m[f"{c}_reg"].isna().any() or not np.allclose(
                    m[c].to_numpy(float), m[f"{c}_reg"].to_numpy(float), rtol=0, atol=1e-9):
                raise io.EngineeringFailure(f"{step5}: column {c} differs from the registered "
                                            "predictions")
        rows = S.annotate(ev)
        prov = {"dir": str(step5), "evaluation_sha256": want, "evaluate_error": why,
                "verify": "columns only (no step-(5) report.json; N3F record)"}
    ms = L.months(window)
    rows = rows[pd.to_datetime(rows["date"]).dt.to_period("M").dt.to_timestamp().isin(ms)]
    return rows.reset_index(drop=True), prov


# ----------------------------------------------------------------------------- pure
def outcome(donor_rows: pd.DataFrame, ext_rows: pd.DataFrame, art: pd.DataFrame,
            donors: pd.DataFrame, roles: dict, window: str) -> dict:
    """Per-county verdicts on verified rows (pure; no look, no file). ``donors``:
    station_id, group, zone (DR's frozen table)."""
    ms = L.months(window)
    a = art.drop_duplicates("station_id").sort_values("station_id").reset_index(drop=True)
    ids = a["station_id"].astype(str).tolist()
    fa = L.frame_to_arrays(art, ids, ms, ("predicted_m", "sigma_idio_m"))
    ob = L.frame_to_arrays(ext_rows, ids, ms, ("observed_m", "predicted_m"))
    both = np.isfinite(ob["predicted_m"]) & np.isfinite(fa["predicted_m"])
    if both.any() and np.max(np.abs(ob["predicted_m"][both] - fa["predicted_m"][both])) > 1e-9:
        raise io.EngineeringFailure("N3F step-(5) predicted_m differs from the frozen artefact")
    did = donors["station_id"].astype(str).tolist()
    od = L.frame_to_arrays(donor_rows, did, ms, ("observed_m", "predicted_m"))
    R_ext = ob["observed_m"] - fa["predicted_m"]
    R_d = od["observed_m"] - od["predicted_m"]
    masks = L.donor_masks(a, donors)
    hits, below, e = L.transfer_hits(R_ext, R_d, fa["sigma_idio_m"], masks,
                                     a["donor_key"].tolist())
    county = a["county"].astype(str).to_numpy()
    out = {"window": window, "first": L.WINDOWS[window]["first"],
           "last": L.WINDOWS[window]["last"], "rows_in": int(np.isfinite(R_ext).sum()),
           "rows_scored": int(np.isfinite(hits).sum()),
           "wells_outside_artefact": sorted(set(ext_rows["station_id"].astype(str))
                                            - set(ids))[:20],
           "counties": {}}
    from harness import coverage3

    for c in (*L.GATED, *L.REPORT_ONLY):
        m = county == c
        v = L.county_verdict(hits, m, below)
        v["role"] = roles.get(c, "report-only")
        H = hits[m]
        ok = np.isfinite(H)
        if ok.any():
            v["coverage_all_rows"] = float(np.nanmean(H))
            if v["verdict"] != "not_evaluable":
                tt = np.nonzero(ok)[1]
                g = coverage3.coverage_gate(H[ok], tt, **L.D.gate_kwargs())
                v["month_clusters_report_only"] = {"coverage": g["coverage"],
                                                   "pass": g["pass"],
                                                   "label": "uncalibrated (k < 20)"}
        v["by_zone_report_only"] = L.breakdown(hits, a["zone"], m)
        v["by_group_report_only"] = L.breakdown(hits, a["group"], m)
        out["counties"][c] = v
    return out


def wording(doc: dict, out: dict) -> dict:
    W = doc["page_mapping"]["wording"]
    res = {}
    for c, v in out["counties"].items():
        if v["verdict"] == "not_evaluable":
            res[c] = W["not_evaluable"].format(county=c, why=v["why"])
            continue
        cov = v["coverage"]
        if v.get("role") != "primary":
            why = ("R8.3, never pooled" if c in L.REPORT_ONLY else
                   "its power study did not pass")
            res[c] = W["report_only"].format(county=c, why=why, cov=cov, first=out["first"],
                                             last=out["last"])
            continue
        side = ("below the registered range" if cov < L.POINT_WINDOW[0] else
                "above the registered range" if cov > L.POINT_WINDOW[1] else "gate-3 failed")
        key = "pass" if v["verdict"] == "pass" else "fail"
        res[c] = W[key].format(county=c, cov=cov, side=side, first=out["first"],
                               last=out["last"], test=doc["id"])
    return res


# ----------------------------------------------------------------------------- runner
def _attempt(out: Path, rec: dict) -> None:
    for p in (Path(out) / "attempts.jsonl", io.repo_path(RUN_ATTEMPTS)):
        io.append_attempt(p, {"event": "score", "scorer_version": SCORER_VERSION, **rec})


def _check_artefact(doc: dict, window: str) -> pd.DataFrame:
    name = f"sigma_idio_{window}.csv"
    p = io.repo_path(ART_DIR) / name
    want = (((doc.get("interval") or {}).get("artefact") or {}).get("files") or {}).get(
        name, {}).get("sha256")
    if want is None or io.sha256(p) != want:
        raise io.EngineeringFailure(f"{name} is not the registered artefact")
    return pd.read_csv(p, dtype={"station_id": str})


def run_score(a, prereg_dir=None, rows_dir=None) -> dict:
    import wells_e02x as WX
    from harness import prereg3

    win = a.window
    pid = L.prereg_id(win)
    out = Path(a.out)
    pdir = Path(prereg_dir) if prereg_dir else io.prereg_dir_default()
    base = {"prereg_id": pid, "window": win}
    try:
        doc, sha = io.registered(pid, pdir)              # refuses before registration
        code = verify_code(doc, pdir)                    # S1: the REGISTERED shas
        nb = pd.Timestamp(L.WINDOWS[win]["not_before"], tz="UTC")
        if io.now() < nb:
            raise io.Refused(f"{pid} may not be scored before {nb.date()}")
        if (out / "report.json").exists():
            raise io.Refused(f"{out}/report.json exists: scored once")
        st = order_status(pdir, a.e02c_report, a.e02d_report, a.n3f_report)
        if not st["ok"]:
            why_not = [v["why"] for v in st.values() if isinstance(v, dict) and not v["ok"]]
            raise io.Refused(f"order (E02c -> E02d -> N3F -> E02x): {why_not}")
        ok, why = precondition(a.window_product, win, code["heads_window_sha256"])
        if not ok:                                       # a readable MANIFEST, no grant
            rec = {**base, "verdict": "not_evaluable", "why": f"window product: {why}",
                   "registration_sha256": sha, "order": st}
            io.write_once(out / "report.json", rec)
            _attempt(out, {**base, "outcome": "not_evaluable", "why": rec["why"]})
            return rec
        if not (a.step5_dir and a.ext_step5_dir):
            raise io.Refused("--step5-dir (E02's step (5)) and --ext-step5-dir (N3F's) needed")
        art = _check_artefact(doc, win)                     # model output: before the looks
        design, _ = WX.u1_design(pdir)
        donors = WX.donors_table(design, ("station_id", "group", "zone"))
        tests = [t["test_id"] for t in doc["tests"]]
        kw = {"prereg_dir": pdir}
        if rows_dir:
            kw["rows_dir"] = rows_dir
        looks = [prereg3.assert_can_score(pid, doc["wp"], test_id=t, **kw) for t in tests]
        # only now, after the looks: the rows
        dp, dproto, dman = _e02_files()
        drows, dprov = verified_rows(Path(a.step5_dir), dp, dproto, dman, win)
        ep, eproto, eman = _n3f_files(pdir)
        n3f_rec, _ = _record(a.n3f_report, L.N3F_ID)
        erows, eprov = verified_rows(Path(a.ext_step5_dir), ep, eproto, eman, win, n3f_rec)
        roles = dict(doc["tests"][0]["role_by_county"])
        roles.update({c: "report-only" for c in L.REPORT_ONLY})
        res = outcome(drows, erows, art, donors, roles, win)
        rep = {**base, "scorer_version": SCORER_VERSION, "registration_sha256": sha,
               "outcome": res, "wording": wording(doc, res), "order": st, "code_check": code,
               "window_product_check": why, "rows_provenance": {"donors": dprov,
                                                                "ext": eprov},
               "score_looks": [lk["tests"] for lk in looks],
               "post_registration_looks": [lk["post_registration_looks"] for lk in looks]}
        rep["report_sha256"] = io.write_once(out / "report.json", rep)
        _attempt(out, {**base, "outcome": "scored", "report_sha256": rep["report_sha256"]})
        return rep
    except io.Refused as exc:
        _attempt(out, {**base, "outcome": "refused", "why": str(exc)})
        raise
    except (S.ScoringRefused, prereg3.PreregistrationError) as exc:
        _attempt(out, {**base, "outcome": "refused", "why": str(exc)})
        raise io.Refused(str(exc)) from exc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--window", choices=sorted(L.WINDOWS), required=True)
    s.add_argument("--window-product", required=True, type=Path)
    s.add_argument("--step5-dir", type=Path, default=None)
    s.add_argument("--ext-step5-dir", type=Path, default=None)
    s.add_argument("--e02c-report", type=Path, default=None)
    s.add_argument("--e02d-report", type=Path, default=None)
    s.add_argument("--n3f-report", type=Path, default=None)
    s.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    try:
        rep = run_score(a)
    except (io.Refused, S.ScoringRefused) as exc:
        print(f"REFUSED: {exc}")
        return 1
    print(json.dumps({k: rep.get(k) for k in ("prereg_id", "window", "wording", "verdict")},
                     indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
