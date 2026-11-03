"""Scorer of COLF.column_family_2028.v1: one scoring session, report written once.

Refuses (ScoringRefused, nothing read, nothing written) unless, in this order:
(1) the declaration is registered and intact (prereg3.load_prereg) and its frozen code is
    this code; (2) the date is on or after 2030-01-15; (3) N3's 24-month look (or a
    not_scorable record naming N3) and N3S-L's ROC117 look (or a not_published /
    not_scorable record naming N3S-L) exist; (4) the frozen predictions hash as registered;
(5) each product's MANIFEST grants prospective_scorer (MANIFEST metadata only; an arm whose
    product does not is written 'not_evaluable' only with --final or after 2031-06-30);
(6) prereg3.assert_can_score passes for every test (it logs the score looks).
Only then are the pile and township tables read. Every run appends to attempts.jsonl.

Usage (repository root):
    $PY -I dataprep_staging/colfreeze/score_colf.py score --piles-product <dir>
        --towns-product <dir> --freeze-npz <npz> --n3-record <json|look> --n3sl-record
        <json> --n3-record-sha <sha> --n3sl-record-sha <sha> --out <dir> [--final]
(a record is optional where the owner's score look is logged; a record counts only by path
and sha256)
"""
from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(HERE), str(REPO / "dataprep_staging")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import colf_config as C  # noqa: E402
import members as M  # noqa: E402
import osse as O  # noqa: E402

SCORER_VERSION = "1.0.0"
PILE_TABLE = "integrated_monthly.parquet"            # N3's (score_n3.PILE_TABLE)
TOWN_TABLES = ("township_bounds.csv", "township_status.csv", "county_intervals.csv")
SHORT = {"雲林縣": "雲林", "彰化縣": "彰化"}
RECORD_VERDICTS = ("not_scorable", "not_published", "not_evaluable")


class ScoringRefused(RuntimeError):
    """The scorer refuses to run (registration, date, order or provenance)."""


class EngineeringFailure(ScoringRefused):
    """An input is inconsistent (no verdict; fix, log and re-run)."""


def _now() -> dt.datetime:
    return dt.datetime.now(C.UTC)


# ----------------------------------------------------------------------------- checks
def load_registered(prereg_dir=None, rows_dir=None) -> dict:
    from harness import prereg3

    kw = {k: v for k, v in (("prereg_dir", prereg_dir), ("rows_dir", rows_dir)) if v}
    try:
        doc, _ = prereg3.load_prereg(C.ID, **kw)
    except Exception as exc:  # noqa: BLE001 - not registered, tampered or unreadable
        raise ScoringRefused(f"{C.ID} is not registered (or not intact): {exc}") from exc
    return doc


def check_code(doc: dict, repo: Path = REPO) -> None:
    bad = [f for f, s in doc["frozen_code"].items() if M.sha256(Path(repo) / f) != s]
    if bad:
        raise ScoringRefused(f"frozen code changed: {bad}")


def _score_looks(prereg_dir) -> set:
    from harness import prereg3

    return {(lk.get("prereg_id"), lk.get("test_id"))
            for lk in prereg3.read_chained(Path(prereg_dir) / "looks.jsonl")
            if lk.get("purpose") == "score"}


def _record_ok(path, sha, prereg_id: str, done) -> dict:
    """A lead-written record naming ``prereg_id`` with a not-scorable verdict, or that
    registration's own report for the needed stage (``done(report) -> bool``), bound by
    path AND sha256 (both go into the report)."""
    out = {"path": str(path) if path else None, "sha256": sha, "ok": False}
    if not path or not Path(path).is_file():
        return out
    if not sha or M.sha256(path) != sha:
        out["why"] = "sha256 missing or different"
        return out
    try:
        r = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return out
    out["ok"] = bool(isinstance(r, dict) and r.get("prereg_id") == prereg_id and (
        r.get("verdict") in RECORD_VERDICTS or done(r)))
    return out


def order_status(prereg_dir, n3_record=None, n3sl_record=None, n3_sha=None,
                 n3sl_sha=None) -> dict:
    have = _score_looks(prereg_dir)
    st = {"n3_24m_look": (C.N3["id"], C.N3["level_24m_test"]) in have,
          "n3sl_roc117_look": (C.N3SL["id"], C.N3SL["l1_tests"][117]) in have,
          "n3_record": _record_ok(n3_record, n3_sha, C.N3["id"],
                                  lambda r: r.get("horizon_months") == 24),
          "n3sl_record": _record_ok(n3sl_record, n3sl_sha, C.N3SL["id"],
                                    lambda r: r.get("survey_roc") == 117)}
    st["ok"] = bool((st["n3_24m_look"] or st["n3_record"]["ok"])
                    and (st["n3sl_roc117_look"] or st["n3sl_record"]["ok"]))
    return st


def product_covers(product_dir, month: str = C.PILE_PRODUCT_COVERS) -> tuple[bool, str]:
    """MANIFEST metadata only: the pile product states its last data month (data_end, or
    parameters.data_end) and it is >= ``month``."""
    from common.manifest import load_manifest

    try:
        m, _ = load_manifest(product_dir)
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)
    end = m.get("data_end") or (m.get("parameters") or {}).get("data_end")
    if not end:
        return False, "the MANIFEST states no data_end"
    return str(end)[:7] >= month, f"MANIFEST data_end {str(end)[:7]} (needs >= {month})"


def product_grants(product_dir) -> tuple[bool, str]:
    """MANIFEST metadata only (no table read)."""
    from common import guard
    from common.manifest import load_manifest

    try:
        m, _ = load_manifest(product_dir)
        guard.check_use(m, "prospective_scorer")
    except Exception as exc:  # noqa: BLE001 - any refusal means 'not granted'
        return False, str(exc)
    return True, "MANIFEST grants prospective_scorer"


# ----------------------------------------------------------------------------- piles
def pile_frame(rows):
    """Verified (flag 0) rows -> pile, depth_m, month, inc_cm (score_n3.pile_frame)."""
    import pandas as pd

    r = rows[rows["flag"] == 0].copy()
    r["inc_cm"] = pd.to_numeric(r["value"]) / 10.0
    r["month"] = r["month"].astype(str).str[:7]
    return r[["pile", "depth_m", "month", "inc_cm"]]


def observed_piles(pf, arm: str) -> tuple[np.ndarray | None, dict]:
    """(2, n) printed increments (NaN where not verified) at each pile's declared sensor,
    or None when a pile has fewer than the arm's min_months."""
    a = C.ARMS[arm]
    ms = M.months(*a["window"])
    out, info = np.full((len(C.PILE_KEYS), len(ms)), np.nan), {}
    for p, key in enumerate(C.PILE_KEYS):
        spec = C.PILES[key]
        series = {}
        for depth in (spec["depth_m"], spec["fallback_depth_m"]):
            if depth is None:
                continue
            g = pf[(pf["pile"] == spec["pile"]) & (pf["depth_m"] == depth)]
            g = g.drop_duplicates("month").set_index("month")["inc_cm"]
            series[depth] = np.asarray([g.get(m, np.nan) for m in ms], float)
        depth, s = O.choose_sensor(series, spec["depth_m"], spec["fallback_depth_m"],
                                   a["min_months"])
        n = int(np.isfinite(s).sum()) if s.size else 0
        info[key] = {"sensor_depth_m": depth, "verified_months": n}
        if n < a["min_months"]:
            return None, info
        out[p] = s
    return out, info


# ----------------------------------------------------------------------------- townships
def interval_of(intervals, county: str, roc: int) -> tuple[str, str, str]:
    """The printed survey interval of a county-year, else the nominal one (N3S-L's rule)."""
    if intervals is not None and len(intervals):
        import pandas as pd

        i = intervals[(intervals["county"] == SHORT[county])
                      & (intervals["survey_year"].astype(int) == roc + 1911)]
        if len(i) and pd.notna(i["start_ym"].iloc[-1]) and pd.notna(i["end_ym"].iloc[-1]):
            return str(i["start_ym"].iloc[-1])[:7], str(i["end_ym"].iloc[-1])[:7], "printed"
    s, e = C.NOMINAL[roc]
    return s, e, "nominal"


def _flag(v) -> bool:
    import pandas as pd

    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes")
    return bool(v) if pd.notna(v) else False


def observed_towns(tables: dict) -> tuple[dict, dict]:
    """{y: {'listed', 'upper', 'use', 'informative'}} in COAST order (N3S-L's check_bounds
    semantics) and the per-year intervals."""
    b, st, iv = (tables.get(n) for n in TOWN_TABLES)
    obs, ivs = {}, {}
    for y in C.SURVEYS:
        rows = b[(b["survey_year"].astype(int) == y + 1911) & b["township"].isin(C.COAST)] \
            if b is not None and len(b) else None
        rows = rows.drop_duplicates(["county", "township"]) if rows is not None else None
        listed = np.zeros(len(C.COAST), bool)
        use = np.zeros(len(C.COAST), bool)
        upper = np.full(len(C.COAST), C.LISTED_FLOOR)
        have = set()
        for r in (rows.itertuples() if rows is not None else []):
            i = C.COAST.index(r.township)
            have.add(r.township)
            lab, anyl = _flag(r.listed_by_all), _flag(r.listed_by_any)
            amb = _flag(getattr(r, "ambiguous", False)) or (anyl and not lab)
            use[i] = not amb
            listed[i] = lab and not amb
            upper[i] = float(r.peak_rate_upper_cm_yr)
        yl = [t for t in C.COAST if C.COAST_COUNTY[t] == "雲林縣"]
        yl_ok = bool(st is not None and len(st) and (
            (st["county"] == "雲林") & (st["survey_year"].astype(int) == y + 1911)).any()) \
            and set(yl) <= have
        obs[y] = {"listed": listed, "upper": upper, "use": use,
                  "informative": np.asarray(yl_ok and int(listed.sum()) >= C.L1_MIN_LISTED)}
        ivs[y] = {c: interval_of(iv, c, y) for c in ("雲林縣", "彰化縣")}
    return obs, ivs


def member_town_peaks(frozen: dict, col: str, ivs: dict) -> dict:
    out = {}
    for y in C.SURVEYS:
        pk = {}
        for cty in ("雲林縣", "彰化縣"):
            s, e, _ = ivs[y][cty]
            if s not in frozen["towns_months"] or e not in frozen["towns_months"]:
                raise EngineeringFailure(f"survey interval {s}..{e} outside the frozen months")
            pk[cty] = O.town_peaks(frozen["members"][col]["towns"], frozen["towns_months"],
                                   frozen["town_cells"], (s, e))
        out[y] = np.asarray([pk[C.COAST_COUNTY[t]][i] for i, t in enumerate(C.COAST)])
    return out


# ----------------------------------------------------------------------------- ranking
def rank_arm(frozen: dict, arm: str, cols: list[str], obs, ivs=None) -> dict:
    """RMSE per member and the winner of every pair (the OSSE's rule)."""
    piles = C.ARMS[arm]["kind"] == "piles"
    if piles:
        rm = {c: O.rmse_piles(O.pile_increments(frozen["members"][c]["piles"],
                                                frozen["piles_months"], arm), obs)
              for c in cols}
    else:
        rm = {c: float(O.rmse_towns(member_town_peaks(frozen, c, ivs), obs)) for c in cols}
    pairs = {}
    for a, b in itertools.combinations(sorted(cols), 2):
        o = (O.outcome_piles(rm[a][None], rm[b][None]) if piles
             else O.outcome([rm[a]], [rm[b]]))
        pairs[f"{a}|{b}"] = {"winner": a if o["a"] else b if o["b"] else None,
                             "tie": bool(o["tie"]), "split": bool(o.get("split", 0.0))}
    rmse = ({c: dict(zip(C.PILE_KEYS, map(float, v), strict=True)) for c, v in rm.items()}
            if piles else rm)
    return {"rmse": rmse, "pairs": pairs,
            "rule": "per-pile agreement (both primary piles)" if piles else "bound RMSE"}


def wording(doc: dict, arm: str, pairs: dict) -> dict:
    """A ranking sentence only for pairs registered D (gating) or R (report-only); every
    other pair is tagged 'not decidable (P=...)' whatever the observations say."""
    dec = doc["osse_result"]["decidability"]["arms"].get(arm, {})
    scale = O.SCALE_WORDS[arm]
    out = {}
    for k, v in pairs.items():
        reg = dec.get(k, {})
        lab, p = reg.get("label"), reg.get("p")
        words = O.pair_words(arm, k)
        if lab not in ("D", "R"):
            ptxt = "n/a" if p is None else f"{p:.2f}"
            out[k] = {"registered_label": lab, "role": "not decidable",
                      "line": f"{words}: not decidable (P={ptxt})"}
            continue
        a, b = k.split("|")
        if v.get("split"):
            line = f"{words}: split between the two piles (not ranked)"
        elif v["tie"]:
            line = f"{words}: tie on {O.UNIT_WORDS[arm]}"
        else:
            loser = b if v["winner"] == a else a
            line = (f"{v['winner']} tracks the status-quo level better than {loser} on "
                    f"{O.UNIT_WORDS[arm]}{scale} [{words}]")
        role = "decided (COLF screen, prospective)" if lab == "D" else "report-only"
        out[k] = {"registered_label": lab, "role": role, "line": f"{line} ({role})"}
    return out


# ----------------------------------------------------------------------------- run
def _write_once(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, sort_keys=True, default=float)
        f.write("\n")


def _attempt(out: Path, rec: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "attempts.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": _now().isoformat(), **rec}, ensure_ascii=False,
                           default=str) + "\n")


def run_score(a, prereg_dir=None, rows_dir=None, now: dt.datetime | None = None) -> dict:
    from harness import prereg3

    out = Path(a.out) / "report.json"
    if out.exists():
        raise FileExistsError(f"{out} exists: the report is written once")
    doc = load_registered(prereg_dir, rows_dir)                                     # (1)
    check_code(doc)
    now = now or _now()
    if now < C.NOT_BEFORE:                                                          # (2)
        raise ScoringRefused(f"not before {C.NOT_BEFORE:%Y-%m-%d}")
    pdir = Path(prereg_dir) if prereg_dir else prereg3.PREREG_DIR
    order = order_status(pdir, a.n3_record, a.n3sl_record,                          # (3)
                         getattr(a, "n3_record_sha", None), getattr(a, "n3sl_record_sha", None))
    if not order["ok"]:
        raise ScoringRefused(f"session order: N3 (24 months) and N3S-L (ROC117) first: {order}")
    fz = doc["frozen_predictions"]
    frozen = M.load_freeze(a.freeze_npz, fz["predictions_npz"]["sha256"])          # (4)
    grants = {"piles": product_grants(a.piles_product),                              # (5)
              "towns": product_grants(a.towns_product)}
    cov = product_covers(a.piles_product)
    if grants["piles"][0] and not cov[0]:
        grants["piles"] = (False, f"pile product does not cover {C.PILE_PRODUCT_COVERS}: "
                                  f"{cov[1]}")
    late = now > C.NO_LATER_THAN
    rep = {"prereg_id": C.ID, "scorer_version": SCORER_VERSION, "scored_at": now.isoformat(),
           "order": order, "products": {k: v[1] for k, v in grants.items()},
           "roles": doc["roles"], "arms": {}}
    missing = [k for k, v in grants.items() if not v[0]]
    if missing and not (late or getattr(a, "final", False)):
        raise ScoringRefused(f"no prospective_scorer grant for {missing}: build the products "
                             f"first, or pass --final for the not-evaluable record")
    kw = {"prereg_dir": pdir, **({"rows_dir": rows_dir} if rows_dir else {})}
    tests = [C.ARMS[x]["test_id"] for x in C.ARM_ORDER
             if C.ARMS[x]["kind"] not in missing]
    rep["post_registration_looks"] = {                                               # (6)
        t: prereg3.assert_can_score(C.ID, C.WP, test_id=t, **kw)["post_registration_looks"]
        for t in tests}
    dd = fz["dedup"]
    from common import guard

    pf = tables = None
    if "piles" not in missing:
        pf = pile_frame(guard.open_product(a.piles_product, "prospective_scorer")
                        .read_table(PILE_TABLE))
    if "towns" not in missing:
        prod = guard.open_product(a.towns_product, "prospective_scorer")
        tables = {n: prod.read_table(n) for n in TOWN_TABLES if n in prod.outputs}
    for arm in C.ARM_ORDER:
        kind = C.ARMS[arm]["kind"]
        cols = dd[arm]["members"]
        if kind in missing:
            rep["arms"][arm] = {"verdict": "not_evaluable", "why": grants[kind][1]}
            continue
        if kind == "piles":
            obs, info = observed_piles(pf, arm)
            if obs is None:
                rep["arms"][arm] = {"verdict": "not_evaluable", "sensors": info,
                                    "why": f"fewer than {C.ARMS[arm]['min_months']} verified "
                                           f"months at a primary pile"}
                continue
            r = rank_arm(frozen, arm, cols, obs)
            r["sensors"] = info
        else:
            obs, ivs = observed_towns(tables)
            if not any(bool(o["informative"]) for o in obs.values()):
                rep["arms"][arm] = {"verdict": "not_informative", "why": "no informative "
                                    "survey year (>= 3 listed and Yunlin's list verified)"}
                continue
            r = rank_arm(frozen, arm, cols, obs, ivs)
            r["intervals"] = ivs
        r["verdict"] = "scored"
        r["wording"] = wording(doc, arm, r["pairs"])
        rep["arms"][arm] = r
    _write_once(out, rep)
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--piles-product", type=Path, required=True)
    s.add_argument("--towns-product", type=Path, required=True)
    s.add_argument("--freeze-npz", type=Path, required=True)
    s.add_argument("--n3-record", type=Path)
    s.add_argument("--n3sl-record", type=Path)
    s.add_argument("--n3-record-sha")
    s.add_argument("--n3sl-record-sha")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--final", action="store_true")
    a = ap.parse_args(argv)
    try:
        rep = run_score(a)
    except Exception as exc:
        _attempt(a.out, {"rc": 1, "error": repr(exc)[:300]})
        raise
    _attempt(a.out, {"rc": 0, "arms": {k: v.get("verdict") for k, v in rep["arms"].items()}})
    print(json.dumps({k: v.get("verdict") for k, v in rep["arms"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
