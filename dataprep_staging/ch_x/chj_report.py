"""CH-J's report (I1.CHX.v1 secondary) after the primary's single scoring session.

score_chx.py writes report.json once (mode 'x'); it was written with chj_secondary 'not_run'
(the hook did not exist yet). This NEW wrapper calls the frozen ``score_chx.chj_block``
UNCHANGED and writes ``results/dataprep/reports/I1.CHX.v1/chj_report.json`` beside it, once
(mode 'x'); report.json is never touched.

Order (each step refuses before the next; no head is opened before step 4):

0. the declaration is registered and intact and every frozen code file matches
   (``score_chx.verify_registration``); report.json exists (the primary was scored);
1. the CH-J metrics carry both Changhua sw fractions and feed their max (ruling R8.1c,
   ``check_sw_fractions``); the CHX hindcast panel passes ``score_chx.check_panel`` (the declared 23 screens and the
   Yunlin screens come from it); the CH-J panel, metrics and engineering record exist;
2. the re-reads are logged BEFORE any head is opened: one ``prereg3.record_look`` per unit
   (the 23 CHX external Changhua screens, I1CHX:ext_changhua_23, and the Yunlin screens,
   I1CHX:ext_yunlin), purpose 'after_the_fact', test I1CHX.chj_externals (a rerun re-uses
   an identical existing row); these units are
   spent (the primary's session read the 23), so CH-J's external gates are report-only for
   good (ruling R8.1b);
3. heads: ``chx_io.l1_monthly`` for both screen sets over the CHX window (as run_score);
4. ``score_chx.chj_block(P, obs_ch, obs_yl, chj_panel, chj_metrics, doc, chj_engineering)``,
   unchanged; the R8.1b reading beside it: a Changhua sw fraction at its 0.01 bound for
   either seed -> 'co-located canal/paddy recharge rejected (CH-J, design-time)'; any other
   outcome a report-only line with the amplitude ratio; off-bound is never support and
   chj_pass is never claimable.

    python dataprep_staging/ch_x/chj_report.py --chj-panel <chj_panel.npz> \\
        --chj-metrics <chj_metrics.json> [--chj-engineering <chj_engineering.json>]

Nothing here prints a head or model value: stdout carries the status words and the path.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import chx_const as K  # noqa: E402
import chx_io as IO  # noqa: E402
import score_chx as S  # noqa: E402

WRAPPER_VERSION = "1.0.0"
REPORT_DIR = f"results/dataprep/reports/{K.CHX_ID}"
OUT = f"{REPORT_DIR}/chj_report.json"
PRIMARY = S.REPORT
HC = "results/twin_runs/chx/hindcast/chx_panel.npz"
CHJ = "results/twin_runs/chx/chj"
RULINGS = "results/dataprep/rulings/user_rulings_2026-11-03.json"
REJECTED = "co_located_recharge_rejected"
WORDING = {
    "rejected": "co-located canal/paddy recharge rejected (CH-J, design-time)",
    "report_only": "CH-J (report-only): Changhua sw fraction off its 0.01 bound; median "
                   "Changhua seasonal amplitude ratio {amp} (seeds {seeds}); not support",
}


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def units_and_note() -> list[tuple[str, str]]:
    """The two after_the_fact looks, one per unit re-read."""
    base = ("CH-J secondary (I1.CHX.v1, test I1CHX.chj_externals) re-reads spent units after "
            "the primary's single scoring session; report-only for good (ruling R8.1b), "
            "never a gate; logged before any head is opened")
    return [(K.CHX_UNIT, base + "; the 23 never-calibrated external Changhua screens"),
            (K.CHX_UNIT_YL, base + "; the Yunlin external screens (CH/YL ratio only)")]


def log_looks(prereg_dir=None, rows_dir=None) -> list[dict]:
    """One after_the_fact look per unit; a rerun (e.g. after a refusal past this step)
    re-uses an identical existing row instead of logging the look twice
    (``chj_produce.log_look_once``)."""
    import chj_produce as PR

    return [PR.log_look_once([u], K.CHX_SCOPE, "after_the_fact", test_id=K.CHJ_TEST,
                             prereg_id=K.CHX_ID, note=note, prereg_dir=prereg_dir,
                             rows_dir=rows_dir)
            for u, note in units_and_note()]


def r81b_reading(block: dict) -> dict:
    """Ruling R8.1b's reading of chj_block's per-seed outcomes (no gating needed: the
    fallback's externals are report-only; the sw outcome reads either way)."""
    seeds = block.get("seeds") or {}
    outcomes = {s: v["sw_outcome"] for s, v in seeds.items()}
    amps = {s: v["external"]["amplitude_ratio_median_ch"] for s, v in seeds.items()}
    if not seeds:
        return {"reading": "not_run", "wording": None, "sw_outcomes": {}}
    rejected = any(o == REJECTED for o in outcomes.values())
    return {"reading": "co_located_recharge_rejected" if rejected else "report_only_line",
            "wording": WORDING["rejected"] if rejected else WORDING["report_only"].format(
                amp=", ".join(f"{float(a):.2f}" for a in amps.values()),
                seeds=", ".join(map(str, amps))),
            "sw_outcomes": outcomes, "amplitude_ratio_median_ch": amps,
            "chj_pass_claimable": False, "off_bound_is_support": False,
            "ruling": "R8.1b", "rulings_file": RULINGS,
            "rulings_sha256": IO.sha_or_missing(IO.rp(RULINGS))}


def check_sw_fractions(metrics: dict) -> dict:
    """Ruling R8.1c: every seed's metrics carry BOTH Changhua fractions (deliveries [0] and
    percolation [2], chj_produce ``sw_fractions_changhua``) and the value fed to chj_block
    (``sw_fraction_changhua``) is exactly their max. SystemExit otherwise (before any look).
    Returns {seed: {deliveries, percolation, fed}}."""
    out, probs = {}, []
    for seed, s in (metrics.get("seeds") or {}).items():
        fr = s.get("sw_fractions_changhua") or {}
        d, p = fr.get("deliveries"), fr.get("percolation")
        fed = s.get("sw_fraction_changhua")
        if d is None or p is None:
            probs.append(f"seed {seed}: the metrics lack a Changhua sw fraction "
                         f"(deliveries {d}, percolation {p})")
        elif fed is None or float(fed) != max(float(d), float(p)):
            probs.append(f"seed {seed}: the fed fraction {fed} is not max(deliveries, "
                         f"percolation) (ruling R8.1c)")
        else:
            out[str(seed)] = {"deliveries": float(d), "percolation": float(p),
                              "fed": float(fed)}
    if not metrics.get("seeds"):
        probs.append("the metrics hold no seed")
    if probs:
        raise SystemExit("REFUSED: " + "; ".join(probs))
    return out


def run(a, prereg_dir=None, rows_dir=None) -> dict:
    out = IO.rp(a.out)
    if out.exists():
        raise SystemExit(f"{a.out} exists; CH-J's report is written once")
    doc, decl_sha = S.verify_registration(prereg_dir, rows_dir)
    prim = IO.rp(PRIMARY)
    if not prim.is_file():
        raise SystemExit(f"{PRIMARY} missing: the primary is scored first (score_chx.py)")
    for p in (a.chj_panel, a.chj_metrics):
        if not IO.rp(p).is_file():
            raise SystemExit(f"{p} missing")
    fractions = check_sw_fractions(json.loads(IO.rp(a.chj_metrics).read_text()))
    P = S.check_panel(IO.rp(a.panel), doc)
    months = [str(m) for m in P["months"]]
    sid_ch = list(np.asarray(P["sid_ch"]).astype(str))
    sid_yl = list(np.asarray(P["sid_yl"]).astype(str))
    looks = log_looks(prereg_dir, rows_dir)                 # before any head is opened
    obs_ch, _, notes_ch = IO.l1_monthly(sid_ch, months)
    obs_yl, _, notes_yl = IO.l1_monthly(sid_yl, months)
    eng = IO.rp(a.chj_engineering) if a.chj_engineering else None
    block = S.chj_block(P, obs_ch, obs_yl, IO.rp(a.chj_panel), IO.rp(a.chj_metrics), doc,
                        eng)
    rep = {"prereg_id": K.CHX_ID, "prereg_sha256": decl_sha, "test_id": K.CHJ_TEST,
           "wrapper": "dataprep_staging/ch_x/chj_report.py",
           "wrapper_version": WRAPPER_VERSION,
           "wrapper_sha256": IO.sha256(Path(__file__)),
           "scorer_sha256": IO.sha256(HERE / "score_chx.py"), "created": _now(),
           "primary_report": PRIMARY, "primary_report_sha256": IO.sha256(prim),
           "panel": str(a.panel), "panel_sha256": IO.sha256(IO.rp(a.panel)),
           "after_the_fact_looks": looks, "heads_read": True,
           "guard_notes": list(notes_ch) + list(notes_yl),
           "externals_report_only": "for good: the primary's session already read the 23 "
                                    "screens (ruling R8.1b); chj_block's own 'status' is "
                                    "recorded as computed, never read as a gate",
           "chj_secondary": block, "r8_1b": r81b_reading(block),
           "r8_1c_sw_fractions": fractions}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as f:
        json.dump(rep, f, indent=1, default=float)
    print(f"{K.CHX_ID} CH-J: status {block['status']}; R8.1b reading "
          f"{rep['r8_1b']['reading']} -> {a.out}")
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--panel", default=HC, help="the CHX hindcast panel (chx_panel.npz)")
    ap.add_argument("--chj-panel", default=f"{CHJ}/chj_panel.npz")
    ap.add_argument("--chj-metrics", default=f"{CHJ}/chj_metrics.json")
    ap.add_argument("--chj-engineering", default=f"{CHJ}/chj_engineering.json")
    ap.add_argument("--out", default=OUT)
    run(ap.parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
