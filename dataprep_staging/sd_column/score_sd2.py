"""Scorer of the model-only declaration SD.cut_response_spread.v2 (round 7 rank 9).

v1's scorer (score_sd.py) with two changes only: the forwards are v1's eight sealed outputs,
bound by the sha256 v1's runner recorded (sd2lib.V1_FORWARDS), and step 2 maps the declared
columns to rheology indices by forward.py's own labelling rule (sd2lib.label_mapping) instead
of v1's directory-name assumption. Steps, checks, tolerances, readings and the report are
v1's: every function below that is not about registration, binding or labels is score_sd's,
called unchanged. Reads model output only; no observation is opened.

    PY=~/miniforge3/envs/hydro/bin/python
    $PY dataprep_staging/sd_column/score_sd2.py verify-registration
    $PY dataprep_staging/sd_column/score_sd2.py score --attempts <v1 attempts.jsonl> --out <report>

0. v2 is registered and intact; v1 is registered with sha256 0d05eb53... and unscored; the
   frozen code (v2's files and v1's score_sd.py, sdlib.py, run_sd.sh at their v1 hashes)
   matches; every declared input still hashes as registered; every key's v1 command is v2's
   command (reuse_plan) and v1's attempts.jsonl binds each npz (bind_v1_forwards).  exit 1
1. E0 as v1.                                                                       exit 2
2. layout as v1, rheology labels by forward.py's rule, verified against the npz labels and
   the --vep-json files of v1's registered command; any ambiguity refuses.       exit 2
3. and 4. as v1 (score_sd.column_checks, aggregate, readings); report.json mode 'x'.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import score_sd as S  # noqa: E402
import sd2lib as L2  # noqa: E402
import sdlib as L  # noqa: E402

PREREG_ID = L2.SD2_ID
FROZEN = (f"{L.X}/score_sd2.py", f"{L.X}/sd2lib.py", f"{L.X}/run_sd2.sh") + S.FROZEN
REPORT = L2.REPORT
Engineering = S.Engineering


def repo() -> Path:
    return S.repo()


# --------------------------------------------------------------------------- step 0
def load_registered(prereg_dir=None, rows_dir=None, prereg_id: str = PREREG_ID):
    from common.paths import LEDGER_ROWS, PREREG_DIR
    from harness import prereg3

    return prereg3.load_prereg(prereg_id, prereg_dir=prereg_dir or PREREG_DIR,
                               rows_dir=rows_dir or LEDGER_ROWS)


def verify_v1(prereg_dir=None, rows_dir=None) -> dict:
    """v1 is registered, intact, the one v2 discloses, unscored, and its frozen code is
    byte-identical to its registration."""
    from harness import prereg3

    try:
        doc, h = load_registered(prereg_dir, rows_dir, L2.V1_ID)
    except prereg3.PreregistrationError as e:
        raise Engineering(1, f"v1 declaration not registered or not intact: {e}") from e
    if h != L2.V1_SHA256:
        raise Engineering(1, f"v1 declaration sha256 {h} is not the disclosed {L2.V1_SHA256}")
    r = repo()
    if doc.get("frozen_code") != L2.V1_FROZEN:
        raise Engineering(1, "v1's registered frozen code is not the pinned V1_FROZEN")
    bad = [p for p, s in L2.V1_FROZEN.items() if L.sha(r / p) != s]
    if bad:
        raise Engineering(1, f"v1 files changed since v1's registration: {bad}")
    if (r / L2.V1_REPORT).exists():
        raise Engineering(1, f"{L2.V1_REPORT} exists: v1 was scored, nothing to supersede")
    return {"doc": doc, "sha256": h}


def verify_registration(prereg_dir=None, rows_dir=None) -> dict:
    from harness import prereg3

    try:
        doc, h = load_registered(prereg_dir, rows_dir)
    except prereg3.PreregistrationError as e:
        raise Engineering(1, f"declaration not registered or not intact: {e}") from e
    r = repo()
    bad = [p for p, s in doc.get("frozen_code", {}).items() if L.sha(r / p) != s]
    if sorted(doc.get("frozen_code", {})) != sorted(FROZEN):
        bad.append("frozen_code lists other files")
    if bad:
        raise Engineering(1, f"frozen code differs from the declaration: {bad}")
    changed = prereg3.changed_inputs(doc)
    if changed:
        raise Engineering(1, f"declared inputs changed: {changed}")
    v1 = verify_v1(prereg_dir, rows_dir)
    if doc.get("supersedes", {}).get("sha256") != v1["sha256"]:
        raise Engineering(1, "the declaration does not disclose the registered v1")
    return {"doc": doc, "sha256": h, "v1": v1}


def bound_npz(attempts: Path, v1_doc: dict) -> dict:
    """forward key -> npz path of v1's reused forward (reuse_plan, then bind_v1_forwards)."""
    plan = L2.reuse_plan(v1_doc.get("forwards", {}))
    if plan["rerun"]:
        raise Engineering(1, f"v1 forwards that are not v2's command (rerun those keys only, "
                             f"under a new declaration): {plan['rerun']}")
    files, e = L2.bind_v1_forwards(attempts, repo())
    if e:
        raise Engineering(1, f"reused forwards not bound: {e}")
    return files


# --------------------------------------------------------------------------- step 2
def forward_checks(r: Path, files: dict, grid, v1_doc: dict) -> dict:
    """score_sd.forward_checks with sd2lib.check_forward_npz2 (the rain check unchanged) and
    each forward's column -> rheology index map recorded."""
    out = {}
    for k, p in files.items():
        z = L.load_npz(p)
        cmd = v1_doc["forwards"][k]["command"]
        e = L2.check_forward_npz2(z, k, r, cmd) + L.check_grid(z, grid)
        meta = json.loads((r / L.theta_path(L.FORWARDS[k]["run"])).read_text())["meta"]
        rf = L.FORWARDS[k]["rf"]
        want = meta.get("rf_timeseries_sha256")
        if want is not None and L.sha(r / rf) != want:
            e.append(f"rain file {rf} does not hash to the theta's rf_timeseries_sha256")
        if want is None and rf != L.RF_VENDOR:
            e.append("a theta without a recorded rain file must read the vendor cache")
        index, _ = L2.label_mapping(k, L.rheology_labels(z), "subs_mean_by_rheology" in z, r,
                                    cmd)
        out[k] = {"npz": str(p.relative_to(r)), "npz_sha256": L.sha(p), "problems": e,
                  "rheology_labels": L.rheology_labels(z), "rheology_index": index}
    return out


def mapping_is_declared_order(fw: dict) -> list[str]:
    """score_sd.column_checks takes rheology index = position in FORWARDS[k]['columns'];
    the label map must say the same for every column (else refuse, never reorder)."""
    bad = []
    for k, rec in fw.items():
        want = {c: i for i, c in enumerate(L.FORWARDS[k]["columns"])}
        if rec["rheology_index"] != want:
            bad.append(f"{k}: label map {rec['rheology_index']} is not the declared order")
    return bad


# --------------------------------------------------------------------------- score
def score(attempts: Path, out: Path, prereg_dir=None, rows_dir=None) -> int:
    r = repo()
    try:
        reg = verify_registration(prereg_dir, rows_dir)
        v1_doc = reg["v1"]["doc"]
        files = bound_npz(attempts, v1_doc)
        gm = L.grid_and_masks(r)
        masks = L.group_masks(gm["zone"], gm["changhua"])
        e0 = S.e0_check(r, masks)
        if not e0["pass"]:
            raise Engineering(2, f"E0: archived page cut30 fan mean {e0['fan_cm']:.4f} cm is "
                                 f"not {S.E0_PUBLISHED_CM} cm")
        fw = forward_checks(r, files, gm["grid"], v1_doc)
        bad = {k: v["problems"] for k, v in fw.items() if v["problems"]}
        if bad:
            raise Engineering(2, f"forward layout/rain: {bad}")
        order = mapping_is_declared_order(fw)
        if order:
            raise Engineering(2, f"rheology mapping: {order}")
        decl = reg["doc"]["columns"]
        cols = [c for c, v in decl.items() if v["included"]]
        cc = S.column_checks(r, files, gm["grid"], cols)
        if not cc["columns"].get(L.PAGE_COLUMN, {}).get("pass"):
            raise Engineering(3, f"the page column does not reproduce: "
                                 f"{cc['columns'].get(L.PAGE_COLUMN)}")
    except Engineering as e:
        print(f"ENGINEERING FAILURE (exit {e.code}): {e}")
        return e.code
    ok = [c for c in cols if cc["columns"][c]["pass"]]
    dropped = {c: cc["columns"][c] for c in cols if c not in ok}
    towns = L.page_townships(r)
    agg = S.aggregate(r, cc["keep"], ok, masks, towns)
    pg = agg[L.PAGE_COLUMN]["summary"][L.PRIMARY_HORIZON]["yunlin"]["avoided_cm"]
    distinct = L.distinct_columns(r, ok)
    rsets = S.reading_sets(ok, {c: cc["columns"][c]["verified"] for c in ok})
    sets = dict(rsets, CH=[c for c in ok if L.COLUMNS[c][2] == "CH"])
    rd = S.readings(agg, distinct, rsets)
    for c in agg:
        agg[c].pop("_top_idx")
    ev = pg >= S.MIN_PAGE_YUNLIN_CM
    report = {
        "id": PREREG_ID, "declaration_sha256": reg["sha256"], "label": S.LABEL,
        "supersedes": {"id": L2.V1_ID, "sha256": reg["v1"]["sha256"],
                       "v1_outcome": "engineering failure at step 2 (scorer exit 2); no "
                                     "cut30 quantity was formed under v1"},
        "reused_forwards": {k: {"npz_sha256": L2.V1_FORWARDS[k]["npz_sha256"],
                                "v1_attempt": L2.V1_FORWARDS[k]["attempt"],
                                "v1_time": L2.V1_FORWARDS[k]["time"]} for k in files},
        "label_rule": L2.LABEL_RULE,
        "evaluable": ev,
        "e0": e0, "forwards": fw,
        "columns_checks": cc["columns"], "dropped_columns": dropped,
        "distinct_of": distinct, "tiers": sets,
        "per_column": agg, "readings": rd if ev else None,
        "changhua_tier": {c: agg[c] for c in sets["CH"]},
        "page_line_L": S.page_line(rd, "A+B") if ev else None,
        "page_line_L_A": S.page_line(rd, "A") if ev else None,
        "page_line_L_verified_only": (
            {s: S.page_line(rd, s) for s in ("A verified only", "A+B verified only")}
            if ev else None),
        "unverified_columns": {c: L.UNVERIFIED[c] for c in ok if c in L.UNVERIFIED},
        "f_subs_page": agg[L.PAGE_COLUMN]["f_subs_yunlin"],
        "towns_source": {"page_html": towns["page_html"],
                         "sha256": L.sha(r / towns["page_html"])},
        "equal_weights": "every distinct column counts once; no observation weighting",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, ensure_ascii=False, sort_keys=True, default=float)
        fh.write("\n")
    print(f"wrote {out} (sha256 {L.sha(out)})")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("verify-registration")
    s = sp.add_parser("score")
    s.add_argument("--attempts", required=True)
    s.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "verify-registration":
        try:
            v = verify_registration()
        except Engineering as e:
            print(f"REFUSED: {e}")
            return e.code
        print(f"registered and intact: {PREREG_ID} sha256 {v['sha256']} (supersedes "
              f"{L2.V1_ID} {v['v1']['sha256'][:8]})")
        return 0
    out = Path(a.out) if a.out else repo() / REPORT
    return score(Path(a.attempts), out)


if __name__ == "__main__":
    raise SystemExit(main())
