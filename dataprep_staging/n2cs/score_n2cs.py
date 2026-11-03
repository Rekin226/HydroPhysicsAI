"""N2CS scorer (2028 session only): the one-sided Changhua first-year size-class reading.

Refusals, in this order, each logged in the attempts log and none writing a report:
the report already exists (written once); N2CS is not registered; it is before 2028-01-15;
N2C's step-(2) report is not at its path or its 'verdict' is not a pinned-scorer verdict; a
frozen file (heads_window.py and score_e02c.py included) differs from its registered sha256;
OSSE forward A (the baseline reproduction reference) is missing.

Then, in order:
1. the MANIFEST precondition, N2CS's own (lead ruling B1): E02c's frozen
   ``product_grants_scorer`` AND the registered GOV module's public
   ``heads_window.covering_rule(m, WINDOW)``, from the MANIFEST alone: failing ->
   'not_evaluable', nothing read, no look;
2. the model-only informative gate under the registered form (no head): failing ->
   'not_informative', no look;
3. the column contract (n2cs_const.HEADS_CONTRACT) from the schema and the id/date/flag
   columns only (no observed_m value): failing -> REFUSED (engineering, before the look);
4. ``prereg3.assert_can_score`` (logs the score look); everything after it is wrapped: any
   exception writes the final report 'engineering_failure_after_look' (disclosed). The heads
   and N2C's registered R1c ZR are read, e = obs - ZR, the registered form's beta_CH and its
   90 % interval, the reading; report-only: the other two forms, beta_YL, the refit interval
   and the single-contrast slope; a session after 2028-01-31 is flagged as a breach.

    python dataprep_staging/n2cs/score_n2cs.py score --scoring-dir <dir> \
        --window-product <dir> --heads-table <name.parquet> --n2c-report <N2C report.json>
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import sys
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(HERE), str(REPO / "dataprep_staging" / "n2_policy"),
           str(REPO / "dataprep_staging"), str(REPO / "dataprep_staging" / "e02c")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bootstrap_scorer as BS  # noqa: E402
import n2cs_const as K  # noqa: E402
import n2cs_lib as L  # noqa: E402

HEADS_USE = "prospective_scorer"
ID_COL, DATE_COL, VALUE_COL, FLAG_COL = "station_id", "date", "observed_m", "flag"


class Refused(SystemExit):
    pass


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rp(s) -> Path:
    return Path(s) if Path(s).is_absolute() else REPO / s


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _t(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def log(rec: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(rec, time=_now().strftime("%Y-%m-%dT%H:%M:%SZ")),
                           sort_keys=True, default=str) + "\n")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def gov_module():
    """The registered GOV module (its sha256 is checked by the caller)."""
    return _load(rp(K.GOV), "gov_heads_window_n2cs")


# ----------------------------------------------------------------------------- checks
def n2c_step2_problem(path: Path) -> str | None:
    """Lead ruling S5: N2C's report, with a pinned-scorer verdict. N2C's registration fixes no
    report path (its scorer is written later), so the path is given at scoring (K.N2C_REPORT is
    the expected default); the report must name N2C's registration id and carry a verdict, and
    the caller records its path's sha256 in N2CS's report."""
    path = Path(path)
    if not path.is_file():
        return f"N2C's report {path} is missing (session step 2 not done)"
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return f"N2C's report is not JSON ({exc})"
    if not isinstance(doc, dict):
        return "N2C's report is not a JSON object"
    if K.N2C_ID not in (doc.get("id"), doc.get("prereg_id"), doc.get("registration_id")):
        return f"{path} does not name N2C's registration {K.N2C_ID}"
    if doc.get("verdict") not in BS.VERDICTS:
        return f"N2C's report has no verdict in {BS.VERDICTS}"
    return None


def precondition(window_product: Path) -> tuple[bool, str]:
    """Lead ruling B1: E02c's frozen grant check AND GOV's covering rule for N2CS's window,
    from the MANIFEST alone."""
    import score_e02c
    from common.manifest import load_manifest

    ok, why = score_e02c.product_grants_scorer(window_product)
    if not ok:
        return False, f"product_grants_scorer: {why}"
    m, _ = load_manifest(window_product)
    ok, rule = gov_module().covering_rule(m, K.WINDOW)
    return bool(ok), f"product_grants_scorer: {why}; {K.GOV_ID} covering_rule: {rule}"


def contract_problems(product, table: str, ids) -> list[str]:
    """Lead ruling B2, BEFORE the look: schema, dtypes, the 74 ids as strings with their
    leading zeros, month parse and at least one scorable row, from the station_id / date /
    flag columns only (no observed_m value is read)."""
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
    df = product.read_table(table, columns=[ID_COL, DATE_COL])
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
        e.append(f"a date of the 74 wells does not parse to a month ({exc})")
        return e
    ok = mine["_m"].isin(L.WINDOW_MONTHS)
    if has_flag:
        ok &= mine[FLAG_COL].astype(int).isin(BS.FLAGS_SCORED)
    if not ok.any():
        e.append("no scorable row (one of the 74 ids, a window month, flag 0-1)")
    return e


# ----------------------------------------------------------------------------- contrast
def contrast_panel(scoring_dir: Path) -> dict:
    """The two contrasts at the 74 screens over the window from the scoring forward, with its
    checks (metadata, arm files and their per-cell path, baseline reproduction against OSSE
    forward A, which must exist: lead ruling S6)."""
    import build_r_county as BR
    import osse_n2cs as O

    fa = rp(K.FWD_A + ".npz")
    if not fa.exists():
        raise Refused(f"REFUSED: OSSE forward A {fa} is missing (the reproduction reference)")
    f = np.load(scoring_dir / "forward_ncs_r1c.npz", allow_pickle=False)
    names = ["baseline", *K.SCORE_ARMS]
    arms = {"NCSCH": scoring_dir / "r_changhua.npz", "NCSYL": scoring_dir / "r_yunlin.npz"}
    flags, _ = BR.load_flags(rp(K.COUNTY_MAP))
    probs = (O.forward_meta_problems(f, names) + O.arm_sha_problems(f, arms)
             + BR.arm_problems(arms["NCSCH"], flags, "changhua")
             + BR.arm_problems(arms["NCSYL"], flags, "yunlin"))
    if probs:
        raise Refused("ENGINEERING: " + "; ".join(probs))
    sc = O.screens()
    t0, t1 = K.FWD_WINDOW
    hm = f["heads_mean"]
    base = np.asarray(hm[0][sc["layer"], sc["cell"], t0:t1], float)
    c_ch = np.asarray(hm[1][sc["layer"], sc["cell"], t0:t1], float) - base
    c_yl = np.asarray(hm[2][sc["layer"], sc["cell"], t0:t1], float) - base
    ba = np.asarray(np.load(fa)["heads_mean"][0][sc["layer"], sc["cell"], t0:t1], float)
    repro = float(np.max(np.abs(ba - base)))
    if repro > K.PREGATE_TOL_M:
        raise Refused(f"ENGINEERING: baseline differs from OSSE forward A by {repro} m")
    rch = json.loads((scoring_dir / "r_changhua.json").read_text())
    return {"sid": sc["sid"], "block": O.blocks_for(sc["sid"]), "c_ch": c_ch, "c_yl": c_yl,
            "n_recorded_ch": int(rch["window_months_recorded"]["changhua"]),
            "forward_sha256": sha256(scoring_dir / "forward_ncs_r1c.npz"),
            "r_files": {k: sha256(v) for k, v in arms.items()},
            "baseline_vs_osse_a_max_abs_m": repro}


# ----------------------------------------------------------------------------- panel
def zr_wide(sids, n2c_doc: dict) -> np.ndarray:
    """N2C's registered R1c ZR (predicted_m with its per-well datum), (74, 11)."""
    import pandas as pd

    p = rp(K.ZR_DIR) / "predictions.csv"
    want = [v["sha256"] for v in (n2c_doc.get("inputs_detail") or {}).values()
            if isinstance(v, dict) and v.get("path") == f"{K.ZR_DIR}/predictions.csv"]
    if not want or sha256(p) != want[0]:
        raise RuntimeError(f"{p} is not N2C's registered R1c ZR")
    t = pd.read_csv(p, dtype={"station_id": str}, usecols=["station_id", "date", "predicted_m"])
    t["m"] = t["date"].astype(str).str[:7]
    t = t[t["m"].isin(L.WINDOW_MONTHS) & t["station_id"].isin(set(map(str, sids)))]
    if t.duplicated(["station_id", "m"]).any():
        raise RuntimeError("duplicate (well, month) ZR rows")
    w = t.pivot(index="station_id", columns="m", values="predicted_m")
    return w.reindex(index=list(sids), columns=L.WINDOW_MONTHS).to_numpy(float)


def heads_wide(product, table: str, sids) -> tuple[np.ndarray, dict]:
    """Window heads, flags 0-1, (74, 11), after the look (contract already checked)."""
    import pandas as pd

    df = product.read_table(table, columns=[ID_COL, DATE_COL, VALUE_COL])
    if FLAG_COL in df.columns:
        df = df[df[FLAG_COL].astype(int).isin(BS.FLAGS_SCORED)]
    df = df.copy()
    df["m"] = pd.to_datetime(df[DATE_COL].astype(str)).dt.strftime("%Y-%m")
    df = df[df["m"].isin(L.WINDOW_MONTHS) & df[ID_COL].isin(set(map(str, sids)))]
    if df.duplicated([ID_COL, "m"]).any():
        raise RuntimeError("duplicate (well, month) heads")
    w = df.pivot(index=ID_COL, columns="m", values=VALUE_COL)
    return (w.reindex(index=list(sids), columns=L.WINDOW_MONTHS).to_numpy(float),
            {"table": table, "table_sha256": product.outputs[table]["sha256"],
             "notes": product.notes})


# ----------------------------------------------------------------------------- score
def page_line(reading: str, beta=None, ci=None) -> str:
    base = K.READING_TEXT[reading]
    if beta is not None and np.isfinite(beta):
        base += f"; beta_CH {beta:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]"
    return f"{base}. {K.WORDING}"


def score(scoring_dir: Path, window_product: Path, heads_table: str, n2c_report: Path, *,
          now: dt.datetime | None = None, out_dir: Path | None = None,
          attempts: Path | None = None) -> dict:
    from harness import prereg3

    now = now or _now()
    out_dir = out_dir or rp(K.SCORE_OUT)
    attempts = attempts or rp(K.ATTEMPTS)
    out = out_dir / "report.json"

    def refuse(why: str):
        log({"event": "score_refused", "why": why}, attempts)
        raise Refused(f"REFUSED: {why}")

    log({"event": "score_invoked"}, attempts)
    if out.exists():
        refuse(f"{out} exists; N2CS is scored once")
    try:
        doc, reg_sha = prereg3.load_prereg(K.ID)
    except Exception as exc:                       # noqa: BLE001 - any failure refuses
        refuse(f"{K.ID} is not registered ({type(exc).__name__})")
    if now < _t(K.NOT_BEFORE):
        refuse(f"before {K.NOT_BEFORE}")
    why = n2c_step2_problem(Path(n2c_report))
    if why:
        refuse(why)
    frozen = doc.get("frozen_code") or {}
    for f, s in ((K.GOV, K.GOV_SHA), (K.E02C_SCORER, K.E02C_SCORER_SHA)):
        if frozen.get(f) != s:
            refuse(f"{f} is not bound at {s[:8]} in the registration")
    bad = sorted(f for f, s in frozen.items() if not rp(f).exists() or sha256(rp(f)) != s)
    if bad:
        refuse(f"frozen code changed: {bad}")
    if not rp(K.FWD_A + ".npz").exists():
        refuse(f"OSSE forward A {K.FWD_A}.npz is missing")
    form = doc["primary_form"]
    rec = {"id": K.ID, "test_id": K.TEST, "registration_sha256": reg_sha, "form": form,
           "n2c_report_sha256": sha256(Path(n2c_report)), "scored_at": now.isoformat(),
           "session_breach": bool(now >= _t(K.NO_LATER_THAN)), "wording": K.WORDING,
           "heads_contract": K.HEADS_CONTRACT}

    def write(r: dict) -> dict:
        out_dir.mkdir(parents=True, exist_ok=True)
        with out.open("x", encoding="utf-8") as fh:
            fh.write(json.dumps(r, indent=1, default=float) + "\n")
        log({"event": "score_written", "reading": r["reading"], "sha256": sha256(out)},
            attempts)
        return r

    ok, why = precondition(Path(window_product))
    rec["precondition"] = {"ok": bool(ok), "why": why}
    if not ok:
        return write(dict(rec, reading="not_evaluable", page_line=page_line("not_evaluable")))
    cp = contrast_panel(Path(scoring_dir))
    blocks = dict(zip(cp["sid"], cp["block"], strict=True))
    inf = L.informative_gate(cp["c_ch"], cp["c_yl"], cp["sid"], blocks, L.WINDOW_MONTHS,
                             cp["n_recorded_ch"], form)
    rec.update({"contrast": {k: v for k, v in cp.items() if k not in ("c_ch", "c_yl", "sid",
                                                                       "block")},
                "informative_gate": inf})
    if not inf["passes"]:
        return write(dict(rec, reading="not_informative",
                          page_line=page_line("not_informative")))
    from common import guard

    product = guard.open_product(Path(window_product), HEADS_USE)
    cbad = contract_problems(product, heads_table, cp["sid"])
    if cbad:
        refuse("heads contract: " + "; ".join(cbad))
    can = prereg3.assert_can_score(K.ID, K.WP, test_id=K.TEST)
    rec["post_registration_looks"] = can["post_registration_looks"]
    rec["unverifiable_prior_looks_n"] = len(can["unverifiable_prior_looks"])
    try:
        obs, hrec = heads_wide(product, heads_table, cp["sid"])
        n2c_doc = json.loads(rp(f"results/dataprep/preregistration/{K.N2C_ID}.json")
                             .read_text())
        zr = zr_wide(cp["sid"], n2c_doc)
        rows = L.long_rows(cp["sid"], L.WINDOW_MONTHS, obs - zr, cp["c_ch"], cp["c_yl"])
        prim = L.estimate(rows, blocks, form)
        lo, hi = prim["ci90_ch"]
        reading = L.reading(prim["beta_ch"], lo, hi, True)
        rep = {f: L.estimate(rows, blocks, f) for f in K.FORMS if f != form}
        single = BS.beta_response(rows["e"], rows["ch"], rows["well"], blocks)
        rec.update({"heads": hrec, "n_rows": prim["n_rows"], "n_wells": prim["n_wells"],
                    "dropped_wells": rows["dropped_wells"], "primary": prim,
                    "reading": reading,
                    "page_line": page_line(reading, prim["beta_ch"], prim["ci90_ch"]),
                    "report_only": {"other_forms": rep,
                                    "refit_interval": L.refit_interval(rows, blocks, form),
                                    "single_contrast_demeaned_slope": single}})
    except Exception as exc:                       # noqa: BLE001 - after the look: disclose
        rec.update({"reading": "engineering_failure_after_look",
                    "page_line": page_line("engineering_failure_after_look"),
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback_tail": traceback.format_exc().splitlines()[-6:]})
    return write(rec)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--scoring-dir", type=Path, required=True)
    s.add_argument("--window-product", type=Path, required=True)
    s.add_argument("--heads-table", required=True)
    s.add_argument("--n2c-report", type=Path, required=True)
    a = ap.parse_args(argv)
    r = score(a.scoring_dir, a.window_product, a.heads_table, a.n2c_report)
    print(json.dumps({"reading": r["reading"], "page_line": r["page_line"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
