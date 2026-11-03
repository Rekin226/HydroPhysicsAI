"""FB BF scorer (2028 session only): FB.family_bf_2026-12.v1, the {R0,Q0} vs {R1c,X5R1}
Bayes factor per registered county, written once to the E29 ledger line (ledger only).

Refusals, in this order, each logged in the attempts log and none writing a report:
the report already exists (written once); the BF test is not registered; it is before
2028-01-15; N2C's step-(2) report is missing, does not name N2C's registration id or has no
pinned-scorer verdict (the N2CS precedent); the registration does not bind heads_window.py at
9a953039 and score_e02c.py at 1c4e092a, or a frozen file changed; the power result is not the
registered, non-smoke one or a registered candidate did not pass it; the model-only datum
check fails (predicted_m - h_ZR constant per well within 1e-6 m over the window; the values
are never printed or written, only the count of failing wells).

Then, in order:
1. the MANIFEST precondition (the N2CS precedent): E02c's frozen ``product_grants_scorer``
   AND the registered GOV module's ``covering_rule(m, 2026-12..2027-10)``, from the MANIFEST
   alone: failing -> 'not_evaluable' report, nothing read, no look;
2. the column contract from the schema and the id/date/flag columns only (no observed_m
   value): failing -> REFUSED (engineering, before the look);
3. ``prereg3.assert_can_score`` per test (logs the score look); everything after it is
   wrapped: any exception writes 'engineering_failure_after_look' (disclosed). obs - ZR (E02's
   registered predicted_m, sha 16ac3c37) per county in U1's well order, then
   fb_lib.score_pair with the registered power result's mu and v; the E29 line beside N2C's
   verdict; a session after 2028-01-31 is flagged as a breach.

    python -I dataprep_staging/fb_family/score_fb.py score --window-product <dir> \
        --heads-table <name.parquet> --n2c-report <N2C report.json>
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import sys
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STAGING = HERE.parent
for _p in (str(HERE), str(STAGING), str(STAGING / "e02c"), str(STAGING / "n2_policy"),
           str(STAGING / "fv_decidability")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fb_inputs as FI  # noqa: E402
import fb_io as io  # noqa: E402
import fb_lib as L  # noqa: E402

GOV = "dataprep_staging/gov_heads_window/heads_window.py"
GOV_ID = "GOV.heads_window_product.v1"
GOV_SHA = "9a953039d75fc812fa3d537c80d17cd8c8fa9f90f2b4f70af5d90ddcfa7a4f37"
E02C_SCORER = "dataprep_staging/e02c/score_e02c.py"
E02C_SCORER_SHA = "1c4e092a43ae42c9e57ec96ebf4120b0911155d1f7d3216b27c42a3f41b6bc3d"
E02_PRED = "results/twin/prospective_2026-11/predictions.csv"
E02_PRED_SHA = "16ac3c3783cea9f3c94560448d06c64ec8dc8fbb450effa28b465469a931cc32"
SELF = f"{io.X}/score_fb.py"
SCORE_OUT = f"results/dataprep/reports/{L.BF_ID}"
ATTEMPTS = f"{SCORE_OUT}/attempts.jsonl"
HEADS_USE = "prospective_scorer"
DATUM_TOL_M = 1e-6
ID_COL, DATE_COL, VALUE_COL, FLAG_COL = "station_id", "date", "observed_m", "flag"
FLAGS_SCORED = (0, 1)


class Refused(io.Refused):
    pass


def _t(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def gov_module():
    """The registered GOV module (its sha256 is checked by the caller)."""
    return _load(io.repo_path(GOV), "gov_heads_window_fb")


# ----------------------------------------------------------------------------- checks
def n2c_step2_problem(path: Path) -> str | None:
    """N2C's report (the N2CS precedent): it must exist, name N2C's registration id and
    carry a pinned-scorer verdict; its sha256 is recorded in FB's report."""
    import bootstrap_scorer as BS

    path = Path(path)
    if not path.is_file():
        return f"N2C's report {path} is missing (session step 2 not done)"
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return f"N2C's report is not JSON ({exc})"
    if not isinstance(doc, dict):
        return "N2C's report is not a JSON object"
    if L.N2C_ID not in (doc.get("id"), doc.get("prereg_id"), doc.get("registration_id")):
        return f"{path} does not name N2C's registration {L.N2C_ID}"
    if doc.get("verdict") not in BS.VERDICTS:
        return f"N2C's report has no verdict in {BS.VERDICTS}"
    return None


def precondition(window_product: Path) -> tuple[bool, str]:
    """E02c's frozen grant check AND GOV's covering rule for FB's window, MANIFEST only."""
    import score_e02c
    from common.manifest import load_manifest

    ok, why = score_e02c.product_grants_scorer(window_product)
    if not ok:
        return False, f"product_grants_scorer: {why}"
    m, _ = load_manifest(window_product)
    ok, rule = gov_module().covering_rule(m, L.WINDOW)
    return bool(ok), f"product_grants_scorer: {why}; {GOV_ID} covering_rule: {rule}"


def contract_problems(product, table: str, ids) -> list[str]:
    """BEFORE the look: schema, dtypes, ids as strings with their leading zeros, month parse
    and at least one scorable row, from the station_id / date / flag columns only."""
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not str(table).endswith(".parquet") or table not in product.outputs:
        return [f"{table}: not a parquet table listed in the MANIFEST outputs"]
    sch = pq.read_schema(product._resolve(table))
    e = [f"column {c!r} missing" for c in (ID_COL, DATE_COL, VALUE_COL) if c not in sch.names]
    if e:
        return e
    if not (pa.types.is_string(sch.field(ID_COL).type)
            or pa.types.is_large_string(sch.field(ID_COL).type)):
        e.append(f"{ID_COL} is {sch.field(ID_COL).type}, not a string")
    if not pa.types.is_floating(sch.field(VALUE_COL).type):
        e.append(f"{VALUE_COL} is {sch.field(VALUE_COL).type}, not a float")
    has_flag = FLAG_COL in sch.names
    if has_flag and not pa.types.is_integer(sch.field(FLAG_COL).type):
        e.append(f"{FLAG_COL} is {sch.field(FLAG_COL).type}, not an integer")
    if e:
        return e
    cols = [ID_COL, DATE_COL] + ([FLAG_COL] if has_flag else [])
    df = product.read_table(table, columns=cols)
    sid = df[ID_COL]
    if not all(isinstance(v, str) for v in sid):
        return [f"{ID_COL} holds non-string values"]
    ids = [str(i) for i in ids]
    stripped = {i.lstrip("0") for i in ids}
    lost = sorted({v for v in set(sid) if v not in ids and v in stripped})
    if lost:
        e.append(f"ids without their leading zeros: {lost[:5]}")
    mine = df[sid.isin(ids)].copy()
    try:
        mine["_m"] = pd.to_datetime(mine[DATE_COL].astype(str)).dt.strftime("%Y-%m")
    except (ValueError, TypeError) as exc:
        e.append(f"a date of the registered wells does not parse to a month ({exc})")
        return e
    ok = mine["_m"].isin(L.WINDOW_MONTHS)
    if has_flag:
        ok &= mine[FLAG_COL].astype(int).isin(FLAGS_SCORED)
    if not ok.any():
        e.append("no scorable row (a registered id, a window month, flag 0-1)")
    return e


def zr_wide(ids) -> np.ndarray:
    """E02's registered predicted_m (with its datum) on the window, (W, 11)."""
    import pandas as pd

    p = io.repo_path(E02_PRED)
    if io.sha256(p) != E02_PRED_SHA:
        raise Refused(f"{E02_PRED} is not E02's registered predictions (sha {E02_PRED_SHA})")
    t = pd.read_csv(p, dtype={"station_id": str}, usecols=["station_id", "date", "predicted_m"])
    t["m"] = t["date"].astype(str).str[:7]
    t = t[t["m"].isin(L.WINDOW_MONTHS) & t["station_id"].isin(set(map(str, ids)))]
    if t.duplicated(["station_id", "m"]).any():
        raise Refused("duplicate (well, month) ZR rows")
    w = t.pivot(index="station_id", columns="m", values="predicted_m")
    return w.reindex(index=list(ids), columns=L.WINDOW_MONTHS).to_numpy(float)


def datum_offsets_constant(zr: np.ndarray, h_zr: np.ndarray, tol: float = DATUM_TOL_M
                           ) -> dict:
    """Model-only: predicted_m - h_ZR must be constant per well over the window (the datum);
    returns pass/fail and the COUNT of failing wells only (no value is reported)."""
    d = np.asarray(zr, float) - np.asarray(h_zr, float)
    fin = np.isfinite(d).all(axis=1)
    spread = np.where(fin, np.nanmax(d, axis=1) - np.nanmin(d, axis=1), np.inf)
    bad = int(np.sum(spread > tol))
    return {"ok": bad == 0, "wells_failing": bad, "tolerance_m": tol}


def h_zr_window(ids) -> np.ndarray:
    """The registered forward's baseline heads at the wells on the window (model-only)."""
    import osse_fv as FV
    import pandas as pd

    w158 = pd.read_csv(io.repo_path(FI.WELLS_158), usecols=["station_id", "cell", "layer"],
                       dtype={"station_id": str}).set_index("station_id").loc[list(ids)]
    meta, h = FV.well_heads(io.repo_path(FI.ZR_FORWARD), w158["cell"].to_numpy(int),
                            w158["layer"].to_numpy(int))
    names = [str(x) for x in meta["scenario_names"]]
    b = h[names.index("baseline")]
    return b[:, FV.WIN[0] + L.FV_WINDOW_SLICE[0]:FV.WIN[0] + L.FV_WINDOW_SLICE[1]]


def heads_wide(product, table: str, ids) -> tuple[np.ndarray, dict]:
    """Window heads, flags 0-1, (W, 11), after the look (contract already checked)."""
    import pandas as pd

    df = product.read_table(table, columns=[ID_COL, DATE_COL, VALUE_COL])
    if FLAG_COL in df.columns:
        df = df[df[FLAG_COL].astype(int).isin(FLAGS_SCORED)]
    df = df.copy()
    df["m"] = pd.to_datetime(df[DATE_COL].astype(str)).dt.strftime("%Y-%m")
    df = df[df["m"].isin(L.WINDOW_MONTHS) & df[ID_COL].isin(set(map(str, ids)))]
    if df.duplicated([ID_COL, "m"]).any():
        raise RuntimeError("duplicate (well, month) heads")
    w = df.pivot(index=ID_COL, columns="m", values=VALUE_COL)
    return (w.reindex(index=list(ids), columns=L.WINDOW_MONTHS).to_numpy(float),
            {"table": table, "table_sha256": product.outputs[table]["sha256"]})


def power_problems(doc: dict, prereg_dir=None) -> tuple[list[str], dict | None]:
    """The registered power result: path and sha from the BF registration, cites the
    registered power declaration, not a smoke, and every registered candidate passed."""
    ps = doc.get("power_study") or {}
    p = io.repo_path(ps.get("path", ""))
    if not ps.get("path") or not p.is_file():
        return ["the power result is missing"], None
    if io.sha256(p) != ps.get("sha256"):
        return ["the power result differs from the registered sha256"], None
    res = json.loads(p.read_text(encoding="utf-8"))
    bad = []
    rec = io.index_record(L.POWER_ID, prereg_dir)
    if rec is None or res.get("declaration", {}).get("sha256") != rec["sha256"]:
        bad.append("the power result does not cite the registered power declaration")
    if res.get("smoke") or res.get("synthetic"):
        bad.append("the power result is a smoke")
    keys = [t["candidate"] for t in doc.get("tests", [])]
    try:
        L.require_family(keys)
    except ValueError as exc:
        bad.append(str(exc))
    for k in keys:
        if not res.get("decisions", {}).get(k, {}).get("pass"):
            bad.append(f"{k} did not pass the power study")
    return bad, res


# ----------------------------------------------------------------------------- score
def score(window_product: Path, heads_table: str, n2c_report: Path, *,
          now: dt.datetime | None = None, out_dir: Path | None = None,
          attempts: Path | None = None, prereg_dir=None) -> dict:
    from harness import prereg3

    now = now or io.now()
    out_dir = Path(out_dir) if out_dir else io.repo_path(SCORE_OUT)
    attempts = Path(attempts) if attempts else io.repo_path(ATTEMPTS)
    out = out_dir / "report.json"

    def refuse(why: str):
        io.append_attempt(attempts, {"event": "score_refused", "why": why})
        raise Refused(f"REFUSED: {why}")

    io.append_attempt(attempts, {"event": "score_invoked"})
    if out.exists():
        refuse(f"{out} exists; FB is scored once")
    try:
        doc, reg_sha = io.registered(L.BF_ID, prereg_dir)
    except io.Refused as exc:
        refuse(f"{L.BF_ID} is not registered ({exc})")
    if now < _t(L.NOT_BEFORE):
        refuse(f"before {L.NOT_BEFORE}")
    why = n2c_step2_problem(Path(n2c_report))
    if why:
        refuse(why)
    frozen = doc.get("frozen_code") or {}
    for f, s in ((GOV, GOV_SHA), (E02C_SCORER, E02C_SCORER_SHA)):
        if frozen.get(f) != s:
            refuse(f"{f} is not bound at {s[:8]} in the registration")
    if SELF not in frozen:
        refuse(f"{SELF} is not bound in the registration")
    bad = sorted(f for f, s in frozen.items()
                 if not io.repo_path(f).exists() or io.sha256(io.repo_path(f)) != s)
    if bad:
        refuse(f"frozen code changed: {bad}")
    pbad, res = power_problems(doc, prereg_dir)
    if pbad:
        refuse("; ".join(pbad))
    n2c_verdict = json.loads(Path(n2c_report).read_text(encoding="utf-8"))["verdict"]
    tests = doc["tests"]
    county_ids = {t["county"]: res["counties"][t["county"]]["ids"] for t in tests}
    for t in tests:
        unit = t["fresh_units"][0]
        if sorted(t["external_units"][unit]["ids"]) != sorted(county_ids[t["county"]]):
            refuse(f"{t['test_id']}: unit ids differ from the power result's wells")
    zr = {c: zr_wide(ids) for c, ids in county_ids.items()}
    dchk = {c: datum_offsets_constant(zr[c], h_zr_window(ids)) for c, ids in county_ids.items()}
    if not all(v["ok"] for v in dchk.values()):
        refuse(f"model-only datum check failed: "
               f"{ {c: v['wells_failing'] for c, v in dchk.items()} } wells")
    rec = {"id": L.BF_ID, "registration_sha256": reg_sha, "scored_at": now.isoformat(),
           "n2c_report_sha256": io.sha256(Path(n2c_report)), "n2c_verdict": n2c_verdict,
           "power_result_sha256": doc["power_study"]["sha256"],
           "session_breach": bool(now > _t(L.NO_LATER_THAN)), "datum_check": dchk,
           "ledger_row": L.LEDGER_ROW, "precedence": L.PRECEDENCE, "tests": {}}

    def write(r: dict) -> dict:
        h = io.write_once(out, r)
        io.append_attempt(attempts, {"event": "score_written", "sha256": h})
        return r

    ok, why = precondition(Path(window_product))
    rec["precondition"] = {"ok": bool(ok), "why": why}
    if not ok:
        for t in tests:
            rec["tests"][t["test_id"]] = {
                "reading": "not_evaluable",
                "ledger_line": L.ledger_line("not_evaluable", t["county"], None, n2c_verdict)}
        return write(rec)
    from common import guard

    product = guard.open_product(Path(window_product), HEADS_USE)
    all_ids = sorted({s for ids in county_ids.values() for s in ids})
    cbad = contract_problems(product, heads_table, all_ids)
    if cbad:
        refuse("heads contract: " + "; ".join(cbad))
    for t in tests:
        can = prereg3.assert_can_score(L.BF_ID, L.WP, test_id=t["test_id"])
        rec["tests"][t["test_id"]] = {
            "post_registration_looks": can["post_registration_looks"],
            "unverifiable_prior_looks_n": len(can["unverifiable_prior_looks"])}
    try:
        design = res["design"]
        for t in tests:
            c = t["county"]
            ids = county_ids[c]
            obs, hrec = heads_wide(product, heads_table, ids)
            cr = res["counties"][c]
            mus = L.mu_stack({k: np.asarray(v, float) for k, v in cr["mu"].items()})
            v = np.asarray(cr["calibration"]["v"], float)
            ai = [L.TRUTHS.index(s) for s in t["hypotheses"]["A"]]
            bi = [L.TRUTHS.index(s) for s in t["hypotheses"]["B"]]
            r = L.score_pair(obs - zr[c], design["counties"][c]["groups"], mus, v, L.T_WIN,
                             ai, bi)
            rec["tests"][t["test_id"]].update({
                "heads": hrec, **r,
                "ledger_line": L.ledger_line(r["reading"], c, r["bf"], n2c_verdict)})
    except Exception as exc:                       # noqa: BLE001 - after the look: disclose
        rec.update({"reading": "engineering_failure_after_look",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback_tail": traceback.format_exc().splitlines()[-6:]})
    return write(rec)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--window-product", type=Path, required=True)
    s.add_argument("--heads-table", required=True)
    s.add_argument("--n2c-report", type=Path, required=True)
    a = ap.parse_args(argv)
    try:
        r = score(a.window_product, a.heads_table, a.n2c_report)
    except Refused as exc:
        print(exc)
        return 1
    print(json.dumps({k: v.get("reading") for k, v in r["tests"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
