"""Score E02d.decomposed_prospective_2026-12.v1 (window A) and E02y.years23_2027-11_2029-10.v1
(windows B and C), once each, after the window closes.

Rows. Nothing here reads a heads table directly: the rows are the step-(5) output of
``hydrophysics.twin.prospective.score`` on the window's absolute registration, verified
against the registered files by ``score_e02c.load_e02_step5`` (bound by sha). Window A: E02's
own step (5) on results/twin/prospective_2026-11, run in E02/E02c's session (E02c is scored
first), rows 2026-12..2027-10 (``--step5-dir``). Windows B, C: this scorer calls
``prospective.score`` itself, in-process and without printing, into <out>/step5, only AFTER
assert_can_score has logged both score looks (``--new-wells`` must lie inside the window
product); nobody runs the module CLI on the year-2/3 registrations by hand.

Relative test (both registrations): r = observed - predicted; e = r minus the median of r
over the OTHER scored wells of the same DR group in that month (>= 5 others); hit = |e| <=
z * sigma_idio of the frozen artefact. Per county: wells with >= 6 scored months, >= 20 such
wells, harness.coverage3 gate-3 (clusters = wells) AND point coverage in [0.85, 0.95].
Absolute test (E02y only): hit = lower_m <= observed_m <= upper_m on every well of the
absolute registration (not only artefact wells), same per-county gate, wording pass_weak /
fail_shock only. The twin.prospective protocol verdict inside step (5) is report-only.

Order of a real run: (1) date >= not_before; (2) the window product's MANIFEST grants
prospective_scorer and, for B and C, declares parameters.prospective_scorer_window covering
the window (for A: if declared, it covers A) - metadata only; otherwise not_evaluable,
nothing read, no look; (3) for A, E02c's report (or a lead-written not-scorable record)
exists; (4) prereg3.assert_can_score (logs the score looks: relative first, then absolute);
(5) only then the step-(5) rows are produced (B, C) and read; relative scored before
absolute. ``rehearsal`` scores
synthetic observations (a directory with a SYNTHETIC marker) against a document and an
artefact directory given on the command line; it logs no look.

    python -I dataprep_staging/e02d/score_e02d.py score --window A --window-product <dir> \\
        --step5-dir <E02 step-(5) out> --e02c-report <E02c report.json> --out <dir>
    python -I dataprep_staging/e02d/score_e02d.py score --window B --window-product <dir> \\
        --new-wells <dir inside the window product> --out <dir> \\
        [--previous-report <window A report.json>]
    python -I dataprep_staging/e02d/score_e02d.py rehearsal --window A --synthetic <dir> \\
        --doc <registration json> --artefact-dir <dir> --abs-dir <dir> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent), str(HERE.parent / "e02c")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import e02dio as io  # noqa: E402
import e02dlib as L  # noqa: E402
import score_e02c as S  # noqa: E402

SCORER_VERSION = "1.0.0"
ART_DIR = "results/twin/e02d_relative"


# ----------------------------------------------------------------------------- pure
def _wells_and_groups(art: pd.DataFrame) -> tuple[list[str], np.ndarray, np.ndarray]:
    a = art.drop_duplicates("station_id").sort_values("station_id")
    return (a["station_id"].astype(str).tolist(), a["group"].astype(str).to_numpy(),
            a["county"].astype(str).to_numpy())


def relative_outcome(frame: pd.DataFrame, art: pd.DataFrame, win: str) -> dict:
    """Per-county relative verdicts on verified step-(5) rows (pure; no look, no file)."""
    w = L.WINDOWS[win]
    months = L.month_index(w["first"], w["last"])
    wells, groups, county = _wells_and_groups(art)
    obs = L.frame_to_arrays(frame, wells, months, ("observed_m", "predicted_m"))
    fa = L.frame_to_arrays(art, wells, months, ("predicted_m", "sigma_idio_m"))
    both = np.isfinite(obs["predicted_m"]) & np.isfinite(fa["predicted_m"])
    if both.any() and np.max(np.abs(obs["predicted_m"][both] - fa["predicted_m"][both])) > 1e-9:
        raise io.EngineeringFailure("step-(5) predicted_m differs from the frozen artefact")
    unknown = sorted(set(frame["station_id"].astype(str)) - set(wells))
    R = obs["observed_m"] - fa["predicted_m"]
    hits, below, e = L.relative_hits(R, fa["sigma_idio_m"], groups)
    out = {"window": win, "first": w["first"], "last": w["last"],
           "rows_in": int(np.isfinite(R).sum()), "rows_scored": int(np.isfinite(hits).sum()),
           "wells_outside_artefact": unknown[:20], "counties": {}}
    for c in L.COUNTIES:
        m = county == c
        v = L.county_verdict(hits, m, below)
        if v["verdict"] != "not_evaluable":
            from harness import coverage3

            H = hits[m]
            ok = np.isfinite(H)
            tt = np.nonzero(ok)[1]
            g = coverage3.coverage_gate(H[ok], tt, **L.gate_kwargs())
            v["month_clusters_report_only"] = {"coverage": g["coverage"], "pass": g["pass"],
                                               "label": "uncalibrated (k < 20)"}
            v["by_group_report_only"] = {
                str(gname): float(np.nanmean(hits[m & (groups == gname)]))
                for gname in sorted(set(groups[m]))}
        out["counties"][c] = v
    return out


def absolute_outcome(frame: pd.DataFrame, win: str) -> dict:
    """Per-county absolute verdicts on every well of the verified step-(5) rows (the
    absolute registration's wells; not restricted to the relative artefact's wells)."""
    w = L.WINDOWS[win]
    months = L.month_index(w["first"], w["last"])
    wells = sorted(frame["station_id"].astype(str).unique())
    county = np.array([L.county_of(s) for s in wells])
    a = L.frame_to_arrays(frame, wells, months, ("observed_m", "lower_m", "upper_m"))
    ok = np.isfinite(a["observed_m"])
    hits = np.where(ok, ((a["observed_m"] >= a["lower_m"]) & (a["observed_m"] <= a["upper_m"]))
                    .astype(float), np.nan)
    below = np.where(ok, (a["observed_m"] < a["lower_m"]).astype(float), np.nan)
    return {"window": win, "wells": len(wells),
            "counties": {c: L.county_verdict(hits, county == c, below) for c in L.COUNTIES}}


def wording(doc: dict, kind: str, out: dict, roles: dict | None = None) -> dict:
    W = doc["page_mapping"]["wording"]
    w = L.WINDOWS[out["window"]]
    res = {}
    for c, v in out["counties"].items():
        if v["verdict"] == "not_evaluable":
            res[c] = (W["rel_not_evaluable"] if kind == "rel" else
                      W["rel_not_evaluable"].replace("per-well band", "absolute interval")
                      ).format(county=c, why=v["why"])
            continue
        cov = v["coverage"]
        side = ("below the registered range" if cov < L.POINT_WINDOW[0] else
                "above the registered range" if cov > L.POINT_WINDOW[1] else "gate-3 failed")
        if kind == "abs":
            key = "abs_pass_weak" if v["verdict"] == "pass" else "abs_fail_shock"
            res[c] = W[key].format(county=c, cov=cov, first=w["first"], last=w["last"])
        elif roles is not None and roles.get(c) != "primary":
            res[c] = W["rel_report_only"].format(county=c, cov=cov)
        else:
            key = "rel_pass" if v["verdict"] == "pass" else "rel_fail"
            res[c] = W[key].format(county=c, cov=cov, side=side, first=w["first"],
                                   last=w["last"], test=doc["id"])
    return res


def ext_report_only(frame: pd.DataFrame) -> dict:
    """N3F ext wells (report-only): relative errors with the median over ext wells only."""
    f = frame.copy()
    f["station_id"] = f["station_id"].astype(str)
    f["county"] = f["station_id"].map(L.county_of)
    months = sorted(pd.to_datetime(f["date"]).dt.to_period("M").dt.to_timestamp().unique())
    wells = sorted(f["station_id"].unique())
    a = L.frame_to_arrays(f, wells, pd.DatetimeIndex(months),
                          ("observed_m", "predicted_m", "lower_m", "upper_m"))
    cty = np.array([L.county_of(s) for s in wells])
    R = a["observed_m"] - a["predicted_m"]
    hw = (a["upper_m"] - a["lower_m"]) / 2.0
    e = L.relative(R, cty, L.MIN_EXT_WELLS_MONTH - 1)
    out = {"label": "report-only: N3F's absolute half-width applied to relative errors"}
    for c in L.COUNTIES:
        m = cty == c
        ok = np.isfinite(e[m])
        out[c] = ({"rows": int(ok.sum()), "wells": int(ok.any(axis=1).sum()),
                   "share_inside": float(np.mean(np.abs(e[m][ok]) <= hw[m][ok])),
                   "abs_e_quantiles_m": [float(np.quantile(np.abs(e[m][ok]), q))
                                         for q in (0.5, 0.9)]}
                  if ok.any() else {"rows": 0, "why": "no ext county-month with >= "
                                                      f"{L.MIN_EXT_WELLS_MONTH} wells"})
    return out


# ----------------------------------------------------------------------------- runners
def product_covers(product_dir, win: str) -> tuple[bool, str]:
    """Whether the window product's MANIFEST (metadata only) declares
    parameters.prospective_scorer_window = [first, last] covering window ``win``. Required
    for B and C (R7.heads_years23 extends prospective_scorer beyond the vocabulary's 'Nov
    2026 - Oct 2027'); for A an absent declaration is accepted (year 1 is the vocabulary's
    own window), a declared one must cover A."""
    from common.manifest import load_manifest

    try:
        m, _ = load_manifest(product_dir)
    except (OSError, ValueError) as exc:
        return False, f"no readable MANIFEST ({exc})"
    w = L.WINDOWS[win]
    dec = (m.get("parameters") or {}).get("prospective_scorer_window")
    if dec is None:
        if win == "A":
            return True, "no declared window; year 1 is the vocabulary's own window"
        return False, ("the MANIFEST declares no parameters.prospective_scorer_window: a "
                       f"year-1 product does not grant {w['first']}..{w['last']}")
    try:
        first, last = (pd.Period(str(x), "M") for x in dec)
    except (TypeError, ValueError) as exc:
        return False, f"unreadable prospective_scorer_window {dec!r} ({exc})"
    if first <= pd.Period(w["first"], "M") and pd.Period(w["last"], "M") <= last:
        return True, f"prospective_scorer_window {first}..{last} covers the window"
    return False, (f"prospective_scorer_window {first}..{last} does not cover "
                   f"{w['first']}..{w['last']}")


def _abs_files(abs_dir: Path) -> tuple[pd.DataFrame, Path, dict]:
    pred = pd.read_csv(abs_dir / "predictions.csv", dtype={"station_id": str})
    man = json.loads((abs_dir / "manifest.json").read_text(encoding="utf-8"))
    return pred, abs_dir / "protocol.json", man


def _verified_rows(step5: Path, abs_dir: Path, win: str) -> tuple[pd.DataFrame, dict, dict]:
    pred, proto, man = _abs_files(abs_dir)
    try:
        rows, rep, prov = S.load_e02_step5(step5, pred, proto, man)
    except S.EngineeringFailure as exc:
        raise io.EngineeringFailure(str(exc)) from exc
    months = L.month_index(L.WINDOWS[win]["first"], L.WINDOWS[win]["last"])
    rows = rows[pd.to_datetime(rows["date"]).isin(months)].reset_index(drop=True)
    return rows, rep, prov


def _check_artefact(doc: dict, art_dir: Path, win: str) -> pd.DataFrame:
    name = f"sigma_idio_{win}.csv"
    p = art_dir / name
    reg = ((doc.get("interval") or {}).get("artefact")
           or (doc.get("artefacts") or {}).get("relative") or {})
    want = (reg.get("files") or {}).get(name, {}).get("sha256")
    if want is None or io.sha256(p) != want:
        raise io.EngineeringFailure(f"{name} is not the registered artefact")
    return pd.read_csv(p, dtype={"station_id": str})


def score_doc(doc: dict, win: str, frame: pd.DataFrame, art: pd.DataFrame,
              previous: dict | None = None) -> dict:
    rep = {"prereg_id": doc["id"], "scorer_version": SCORER_VERSION, "window": win}
    rel = relative_outcome(frame, art, win)
    if doc["id"] == L.E02D_ID:
        rep["relative"] = rel
        rep["wording"] = wording(doc, "rel", rel)
        return rep
    tag = L.WINDOWS[win]["first"]
    roles = next(t["role_by_county"] for t in doc["tests"]
                 if t["test_id"] == f"E02yP.rel_{tag}")
    ab = absolute_outcome(frame, win)
    rep.update({"relative": rel, "relative_roles": roles,
                "wording_relative": wording(doc, "rel", rel, roles),
                "absolute": ab, "wording_absolute": wording(doc, "abs", ab),
                "common_mode": "not decidable (k = 2-3 year clusters), declared"})
    if previous:
        rep["lead_growth_report_only"] = {
            c: {"relative": _diff(rel, previous.get("relative"), c),
                "absolute": _diff(ab, previous.get("absolute"), c)} for c in L.COUNTIES}
    return rep


def _diff(now: dict, prev: dict | None, c: str):
    try:
        return now["counties"][c]["coverage"] - prev["counties"][c]["coverage"]
    except (KeyError, TypeError):
        return None


def run_score(a, prereg_dir=None, rows_dir=None) -> dict:
    from harness import prereg3

    win = a.window
    pid = L.E02D_ID if win == "A" else L.E02Y_ID
    pdir = Path(prereg_dir) if prereg_dir else io.prereg_dir_default()
    doc, sha = io.registered(pid, pdir)
    nb = pd.Timestamp(L.WINDOWS[win]["not_before"], tz="UTC")
    if io.now() < nb:
        raise io.Refused(f"window {win} may not be scored before {nb.date()}")
    out = Path(a.out)
    if (out / "report.json").exists():
        raise io.Refused(f"{out}/report.json exists: scored once")
    ok, why = S.product_grants_scorer(a.window_product)
    if ok:
        ok, why2 = product_covers(a.window_product, win)
        why = f"{why}; {why2}"
    if not ok:
        rec = {"prereg_id": pid, "window": win, "verdict": "not_evaluable",
               "why": f"window product: {why}"}
        io.write_once(out / "report.json", rec)
        return rec
    if win == "A":
        st = S.e02c_done_status(pdir, a.e02c_report)
        if not st["ok"]:
            raise io.Refused(f"order: {st['why']} (E02d is scored after E02c)")
        if not a.step5_dir:
            raise io.Refused("window A needs --step5-dir (E02's own step (5))")
        step5 = Path(a.step5_dir)
    else:
        if a.step5_dir:
            raise io.Refused("windows B and C produce their own step (5) after the score "
                             "looks: do not pass --step5-dir")
        if not a.new_wells or not S._inside(a.new_wells, a.window_product):
            raise io.Refused("--new-wells must lie inside the window product directory whose "
                             "MANIFEST was checked")
        step5 = out / "step5"
        if step5.exists():
            raise io.Refused(f"{step5} exists: step (5) runs once, after the score looks")
    art = _check_artefact(doc, io.REPO / ART_DIR, win)      # model output: before the looks
    prev = json.loads(Path(a.previous_report).read_text()) if a.previous_report else None
    tests = [t["test_id"] for t in doc["tests"]
             if t["data_scope"] == f"heads:{L.WINDOWS[win]['first']}..{L.WINDOWS[win]['last']}"]
    kw = {"prereg_dir": pdir}
    if rows_dir:
        kw["rows_dir"] = rows_dir
    looks = [prereg3.assert_can_score(pid, doc["wp"], test_id=t, **kw) for t in sorted(
        tests, key=lambda t: ".abs_" in t)]
    abs_dir = io.REPO / L.ABS_DIRS[win]
    if win != "A":      # only now, after the looks: step (5), in-process, nothing printed
        from hydrophysics.twin.prospective import score as step5_score

        step5_score(abs_dir, Path(a.new_wells), step5)
    frame, e02rep, prov = _verified_rows(step5, abs_dir, win)
    rep = score_doc(doc, win, frame, art, prev)
    if win == "A" and a.ext_step5_dir:
        ev = pd.read_csv(Path(a.ext_step5_dir) / "evaluation.csv", dtype={"station_id": str})
        rep["ext_report_only"] = ext_report_only(ev)
    rep.update({"registration_sha256": sha, "rows_provenance": prov,
                "step5_report_unchanged": e02rep,
                "step5_protocol_verdict": "report-only: never page-mapped, not a test of "
                                          f"{pid}",
                "window_product_check": why,
                "score_looks": [lk["tests"] for lk in looks],
                "post_registration_looks": [lk["post_registration_looks"] for lk in looks]})
    io.write_once(out / "report.json", rep)
    return rep


def run_rehearsal(a) -> dict:
    S.assert_rehearsal_dir(Path(a.synthetic))
    obs = S.read_synthetic_obs(Path(a.synthetic))
    doc = json.loads(Path(a.doc).read_text(encoding="utf-8"))
    pred, proto, man = _abs_files(Path(a.abs_dir))
    out = Path(a.out)
    step5 = S.rehearse_e02_step5(pred, obs, proto, man, out / "rehearsal_step5")
    frame, _, _ = _verified_rows(step5, Path(a.abs_dir), a.window)
    art = pd.read_csv(Path(a.artefact_dir) / f"sigma_idio_{a.window}.csv",
                      dtype={"station_id": str})
    rep = score_doc(doc, a.window, frame, art)
    rep["rehearsal"] = "SYNTHETIC observations; no look logged"
    io.write_once(out / "rehearsal_report.json", rep)
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--window", choices=("A", "B", "C"), required=True)
    s.add_argument("--window-product", required=True, type=Path)
    s.add_argument("--step5-dir", type=Path, default=None, help="window A only")
    s.add_argument("--new-wells", type=Path, default=None,
                   help="windows B, C: inside the window product")
    s.add_argument("--e02c-report", type=Path, default=None)
    s.add_argument("--ext-step5-dir", type=Path, default=None)
    s.add_argument("--previous-report", type=Path, default=None)
    s.add_argument("--out", required=True, type=Path)
    r = sub.add_parser("rehearsal")
    r.add_argument("--window", choices=("A", "B", "C"), required=True)
    r.add_argument("--synthetic", required=True, type=Path)
    r.add_argument("--doc", required=True, type=Path)
    r.add_argument("--artefact-dir", required=True, type=Path)
    r.add_argument("--abs-dir", required=True, type=Path)
    r.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    try:
        rep = run_score(a) if a.cmd == "score" else run_rehearsal(a)
    except (io.Refused, S.ScoringRefused) as exc:
        print(f"REFUSED: {exc}")
        return 1
    print(json.dumps({k: rep.get(k) for k in ("prereg_id", "window", "wording",
                                              "wording_relative", "wording_absolute",
                                              "verdict")}, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
