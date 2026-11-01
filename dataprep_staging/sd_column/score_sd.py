"""Scorer of the model-only declaration SD.cut_response_spread.v1 (round 7 rank 9).

Run once by the lead, after the declaration is registered and the forwards of
``run_sd.sh forward all`` have finished (sealed stdout). It reads model output only: the
forwards' npz files, the archived page/Q0/R0/Changhua forwards (hindcast months and the
page's pooled cut30 figure), the theta and column JSONs and the page payload's township
map. No observation (heads, leveling, rings, piles, GNSS) is opened.

    PY=~/miniforge3/envs/hydro/bin/python
    $PY dataprep_staging/sd_column/score_sd.py verify-registration
    $PY dataprep_staging/sd_column/score_sd.py score --attempts <attempts.jsonl> --out <report>

Steps (each must pass before the next; a failure exits non-zero and writes nothing):

0. the declaration is registered and intact (INDEX, ledger), the frozen code hashes match,
   every declared input still hashes as registered, and attempts.jsonl binds each npz (the
   last successful forward of each key).                                         exit 1
1. E0: the archived page ensemble (datum_gate_innov.npz, coupled_leveling rheology) gives
   the published fan-average cut30 avoided subsidence 0.942 cm (Dec 2022 -> Dec 2032)
   within 0.0005 cm, with this scorer's code.                                    exit 2
2. layout of every forward npz (scenarios, dates, origin, one member, gain 0, rheology
   labels, forward options), the grid, and the rain file against the theta's recorded
   sha256.                                                                       exit 2
3. per column: the hindcast subsidence (months 2012-01..2022-12, no leveling read)
   reproduces its stored trajectory, and the CPU rebuild of the column on the saved heads
   reproduces the forward's subsidence in BOTH scenarios (only the maximum absolute
   difference is kept; no cut30 value or aggregate is formed or printed). The page column
   failing is an engineering failure (exit 3); any other failing column is dropped and
   listed, never tuned. A column with no independent stored trajectory (sdlib.UNVERIFIED:
   page_aquitard, q0_nh, x5r1_cs) passes on its paired theta and rebuild only, labelled.
4. aggregation (the only step that forms cut30 quantities) and the declared readings, for
   each set with and without the UNVERIFIED columns; report.json is written with mode 'x'
   and is final.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import sdlib as L  # noqa: E402

PREREG_ID = L.SD_ID
FROZEN = (f"{L.X}/score_sd.py", f"{L.X}/sdlib.py", f"{L.X}/run_sd.sh")
REPORT = f"results/dataprep/reports/{PREREG_ID}/report.json"
E0_ARCHIVE = f"{L.FWD}/datum_gate_innov.npz"
E0_PUBLISHED_CM = 0.942
E0_TOL_CM = 0.0005
E0_YUNLIN_REPORT_ONLY_CM = 1.398
MIN_PAGE_YUNLIN_CM = 0.05
LABEL = "model-form spread, not an interval"


class Engineering(Exception):
    def __init__(self, code: int, msg: str):
        super().__init__(msg)
        self.code = code


def repo() -> Path:
    from common.paths import REPO

    return REPO


# --------------------------------------------------------------------------- step 0
def load_registered(prereg_dir=None, rows_dir=None):
    from common.paths import LEDGER_ROWS, PREREG_DIR
    from harness import prereg3

    return prereg3.load_prereg(PREREG_ID, prereg_dir=prereg_dir or PREREG_DIR,
                               rows_dir=rows_dir or LEDGER_ROWS)


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
    return {"doc": doc, "sha256": h}


def bound_npz(attempts: Path) -> dict:
    """forward key -> npz path of its last successful forward, checked by sha256."""
    if not attempts.exists():
        raise Engineering(1, f"{attempts} missing")
    last = {}
    for s in attempts.read_text().splitlines():
        if not s.strip():
            continue
        rec = json.loads(s)
        if rec.get("event") == "forward" and rec.get("rc") == 0:
            last[rec["fwd"]] = rec
    r = repo()
    out = {}
    for k in L.FORWARDS:
        rec = last.get(k)
        if rec is None:
            raise Engineering(1, f"no successful forward of {k} in attempts.jsonl")
        p = r / f"{L.out_stem(k)}.npz"
        if L.sha(p) != rec.get("npz_sha256"):
            raise Engineering(1, f"{p} is not the last successful forward of {k}")
        out[k] = p
    return out


# --------------------------------------------------------------------------- step 1
def e0_check(r: Path, masks: dict | None = None) -> dict:
    z = L.load_npz(r / E0_ARCHIVE)
    labels = L.rheology_labels(z)
    names = [str(n) for n in z["scenario_names"]]
    S = np.asarray(z["subs_mean_by_rheology"][labels.index("coupled_leveling")],
                   dtype="float64")
    sub = np.stack([S[names.index("baseline")], S[names.index("cut30")]])
    av = L.avoided_cm(sub)
    t = L.month_index("2032-12")
    fan = float(av[:, t].mean())
    out = {"archive": E0_ARCHIVE, "archive_sha256": L.sha(r / E0_ARCHIVE),
           "fan_cm": fan, "published_cm": E0_PUBLISHED_CM, "tol_cm": E0_TOL_CM,
           "pass": abs(fan - E0_PUBLISHED_CM) <= E0_TOL_CM}
    if masks is not None:
        out["yunlin_cm_report_only"] = float(av[masks["yunlin"], t].mean())
        out["yunlin_published_cm"] = E0_YUNLIN_REPORT_ONLY_CM
    return out


# --------------------------------------------------------------------------- steps 2-3
def forward_checks(r: Path, files: dict, grid) -> dict:
    out = {}
    for k, p in files.items():
        z = L.load_npz(p)
        e = L.check_forward_npz(z, k) + L.check_grid(z, grid)
        meta = json.loads((r / L.theta_path(L.FORWARDS[k]["run"])).read_text())["meta"]
        rf = L.FORWARDS[k]["rf"]
        want = meta.get("rf_timeseries_sha256")
        if want is not None and L.sha(r / rf) != want:
            e.append(f"rain file {rf} does not hash to the theta's rf_timeseries_sha256")
        if want is None and rf != L.RF_VENDOR:
            e.append("a theta without a recorded rain file must read the vendor cache")
        out[k] = {"npz": str(p.relative_to(r)), "npz_sha256": L.sha(p), "problems": e}
    return out


def column_checks(r: Path, files: dict, grid, cols) -> dict:
    """Hindcast reproduction and CPU rebuild of every column; returns per-column records
    and keeps the rebuilt module, its Ske and the driver heads for step 4."""
    res, keep = {}, {}
    for k, p in files.items():
        z = L.load_npz(p)
        run = L.FORWARDS[k]["run"]
        theta = json.loads((r / L.theta_path(run)).read_text())
        drv = L.driver(z["heads_mean"])
        for ri, c in enumerate(L.FORWARDS[k]["columns"]):
            if c not in cols:
                continue
            cj = json.loads((r / L.column_path(c)).read_text())
            pair = L.sha(r / cj["theta"]) == L.sha(r / L.theta_path(run))
            rep = L.hindcast_repro(z, ri, L.COLUMNS[c][3], r)
            module, _, ske = L.rebuild_column(r, c, grid, theta["meta"])
            sub = L.column_subs(z, ri)
            cpu = np.stack([L.compaction_cpu(module, drv[s]) for s in range(2)])
            dmax = float(np.abs(cpu - sub).max())
            res[c] = {"forward": k, "rheology_index": ri, "paired_theta": pair,
                      "hindcast": rep, "rebuild_max_abs_m": dmax,
                      "rebuild_tol_m": L.TOL_REBUILD_M,
                      "rebuild_pass": dmax <= L.TOL_REBUILD_M,
                      "split_available": ske is not None}
            keep[c] = {"subs": sub, "drv": drv, "ske": ske, "heads": z["heads_mean"],
                       "theta": theta, "col_json": cj}
    for c, rec in res.items():
        h = rec["hindcast"]
        if h["route"] == "shared":
            via = res.get(h["via"])
            h["pass"] = bool(via and via["forward"] == rec["forward"]
                             and via["hindcast"].get("pass") is True)
        base = bool(rec["paired_theta"] and rec["rebuild_pass"])
        rec["verified"] = c not in L.UNVERIFIED
        if c in L.UNVERIFIED:
            # no independent stored trajectory: passes on pairing and rebuild (and, on the
            # shared route, its via column), labelled; readings are given without it too
            rec["pass"] = base and h.get("pass") is not False
            rec["label"] = L.UNVERIFIED[c]
        else:
            rec["pass"] = base and h.get("pass") is True
    return {"columns": res, "keep": keep}


# --------------------------------------------------------------------------- step 4
def aggregate(r: Path, keep: dict, ok: list, masks: dict, towns: dict) -> dict:
    out = {}
    for c in ok:
        k = keep[c]
        av = L.avoided_cm(k["subs"])
        el = None if k["ske"] is None else L.elastic_avoided_cm(k["ske"], k["drv"])
        dh = np.asarray(k["drv"][1], "float64") - np.asarray(k["drv"][0], "float64")
        h2 = np.asarray(k["heads"], "float64")
        dh2 = h2[1, 1] - h2[0, 1]
        tops = {h: L.top_towns(av[:, t], towns["town"], towns["eligible"], 5)
                for h, t in L.HORIZONS.items()}
        out[c] = {"tier": L.COLUMNS[c][2],
                  "summary": L.summarise(av, el, dh, dh2, masks),
                  "f_subs_yunlin": L.f_subs(av, masks["yunlin"]),
                  "storage_ratio": L.storage_ratio(k["theta"], k["col_json"]),
                  "top5_towns": {h: [towns["names"][i] for i in v] for h, v in tops.items()},
                  "_top_idx": tops}
    return out


def readings(agg: dict, distinct: dict, sets: dict) -> dict:
    out = {}
    for name, members in sets.items():
        cols = [c for c in members if c in agg and distinct.get(c, c) == c]
        blk = {"columns": cols}
        for h in L.HORIZONS:
            inl = {c: agg[c]["summary"][h]["yunlin_inland"]["avoided_cm"] for c in cols}
            yl = [agg[c]["summary"][h]["yunlin"]["avoided_cm"] for c in cols]
            ds = [agg[c]["summary"][h]["distal_share_yunlin"] for c in cols]
            ds = [x for x in ds if x is not None]
            rk = L.reading_rank({c: agg[c]["_top_idx"]["2032-12"] for c in cols},
                                L.PAGE_COLUMN) if L.PAGE_COLUMN in cols else None
            blk[h] = {"inland": L.reading_inland(inl),
                      "yunlin_cm_range": [min(yl), max(yl)] if yl else None,
                      "yunlin_distal_share_range": [min(ds), max(ds)] if ds else None}
            if h == "2032-12" and rk is not None:
                blk["rank_2032"] = rk
        out[name] = blk
    return out


SET_WORDS = {"A": "on the page's own heads (Tier A)",
             "A+B": "adding the Q0 family and R0 on their own heads (Tier A+B)"}


def page_line(rd: dict, name: str = "A+B") -> str | None:
    """The L-row line of one set (``rd[name]``); None when the set is empty."""
    if not rd.get(name, {}).get("columns"):
        return None
    a = rd[name][L.PRIMARY_HORIZON]
    n = len(rd[name]["columns"])
    lo, hi = a["yunlin_cm_range"]
    c0, c1 = a["yunlin_distal_share_range"] or (float("nan"), float("nan"))
    return (f"across {n} leveling-equivalent columns {SET_WORDS.get(name, name)} the Yunlin "
            f"cut30 avoided subsidence by {L.PRIMARY_HORIZON} is {lo:.2f}-{hi:.2f} cm; coast "
            f"(distal zone) {100 * c0:.0f}-{100 * c1:.0f} % of it; {a['inland']['reading']} "
            f"({LABEL})")


def reading_sets(ok: list, verified: dict) -> dict:
    """The declared reading sets: A and A+B with every passing column, and the same without
    the UNVERIFIED columns ('<set> verified only')."""
    a = [c for c in ok if L.COLUMNS[c][2] == "A"]
    ab = [c for c in ok if L.COLUMNS[c][2] in ("A", "B")]
    return {"A": a, "A+B": ab,
            "A verified only": [c for c in a if verified[c]],
            "A+B verified only": [c for c in ab if verified[c]]}


def score(attempts: Path, out: Path, prereg_dir=None, rows_dir=None) -> int:
    r = repo()
    try:
        reg = verify_registration(prereg_dir, rows_dir)
        files = bound_npz(attempts)
        gm = L.grid_and_masks(r)
        masks = L.group_masks(gm["zone"], gm["changhua"])
        e0 = e0_check(r, masks)
        if not e0["pass"]:
            raise Engineering(2, f"E0: archived page cut30 fan mean {e0['fan_cm']:.4f} cm is "
                                 f"not {E0_PUBLISHED_CM} cm")
        fw = forward_checks(r, files, gm["grid"])
        bad = {k: v["problems"] for k, v in fw.items() if v["problems"]}
        if bad:
            raise Engineering(2, f"forward layout/rain: {bad}")
        decl = reg["doc"]["columns"]
        cols = [c for c, v in decl.items() if v["included"]]
        cc = column_checks(r, files, gm["grid"], cols)
        if not cc["columns"].get(L.PAGE_COLUMN, {}).get("pass"):
            raise Engineering(3, f"the page column does not reproduce: "
                                 f"{cc['columns'].get(L.PAGE_COLUMN)}")
    except Engineering as e:
        print(f"ENGINEERING FAILURE (exit {e.code}): {e}")
        return e.code
    ok = [c for c in cols if cc["columns"][c]["pass"]]
    dropped = {c: cc["columns"][c] for c in cols if c not in ok}
    towns = L.page_townships(r)
    agg = aggregate(r, cc["keep"], ok, masks, towns)
    pg = agg[L.PAGE_COLUMN]["summary"][L.PRIMARY_HORIZON]["yunlin"]["avoided_cm"]
    distinct = L.distinct_columns(r, ok)
    rsets = reading_sets(ok, {c: cc["columns"][c]["verified"] for c in ok})
    sets = dict(rsets, CH=[c for c in ok if L.COLUMNS[c][2] == "CH"])
    rd = readings(agg, distinct, rsets)
    for c in agg:
        agg[c].pop("_top_idx")
    report = {
        "id": PREREG_ID, "declaration_sha256": reg["sha256"], "label": LABEL,
        "evaluable": pg >= MIN_PAGE_YUNLIN_CM,
        "e0": e0, "forwards": fw,
        "columns_checks": cc["columns"], "dropped_columns": dropped,
        "distinct_of": distinct, "tiers": sets,
        "per_column": agg, "readings": rd if pg >= MIN_PAGE_YUNLIN_CM else None,
        "changhua_tier": {c: agg[c] for c in sets["CH"]},
        "page_line_L": page_line(rd, "A+B") if pg >= MIN_PAGE_YUNLIN_CM else None,
        "page_line_L_A": page_line(rd, "A") if pg >= MIN_PAGE_YUNLIN_CM else None,
        "page_line_L_verified_only": (
            {s: page_line(rd, s) for s in ("A verified only", "A+B verified only")}
            if pg >= MIN_PAGE_YUNLIN_CM else None),
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
        print(f"registered and intact: {PREREG_ID} sha256 {v['sha256']}")
        return 0
    out = Path(a.out) if a.out else repo() / REPORT
    return score(Path(a.attempts), out)


if __name__ == "__main__":
    raise SystemExit(main())
