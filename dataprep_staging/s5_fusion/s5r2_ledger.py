"""WP S5 repair round 2: ledger rows (plan 4.1, U6) for every number quoted in
reports/S5.repair.md, from S5_evidence/repair_r2/report_numbers.json (s5r2_report.py).

    python -I s5r2_ledger.py

Roles: the leveling units are not fresh under A1, so withheld leveling target tables are
``design_time_evidence`` (as repair round 1 corrected for T1/T2); training data of a fit are
``fit_target`` for hyperparameters and ``input`` for test predictions and products.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import s5const as C  # noqa: E402
import s5r2_const as K  # noqa: E402
from common import ledger  # noqa: E402
from common.paths import rel_to_repo  # noqa: E402


def _r(p) -> str:
    return rel_to_repo(Path(p))


def sources() -> dict:
    from s1_rings import s1const as K1
    ev, e1 = K.EVID, C.EVIDENCE
    pr = C.DATAPREP / "preregistration"
    return {
        "lev": _r(C.LEVELING_PANEL), "ring_st": _r(C.RING_STATIONS), "grid": _r(C.TWIN_GRID),
        "rings": [_r(C.RING_DIR / f"{s}.parquet") for s in K1.SITES],
        "gnss_pre_v2": _r(C.DATAPREP / "gnss_clean" / "preorigin" / "v2"),
        "gnss_pre_v4": _r(K.GNSS_PRE_V4), "gnss_full_v4": _r(K.GNSS_FULL_V4),
        "insar_p": _r(e1 / "work" / "insar_runP_cells.npz"), "insar_p2": _r(K.INSAR_P2_CELLS),
        "insar_v3": _r(K.INSAR_V3), "piles": _r(C.PILES), "rings_bridged": _r(C.RINGS_BRIDGED),
        "r1_t1": _r(e1 / "t1_targets.parquet"), "r1_t2": _r(e1 / "t2_targets.parquet"),
        "r1_t3g": _r(e1 / "t3_gnss_targets.parquet"), "r1_t3c": _r(e1 / "t3_cmp_targets.parquet"),
        "r1_hyper": _r(e1 / "hyper_pre.json"), "r1_hyper_t1": _r(e1 / "hyper_t1.json"),
        "prereg_r2": _r(pr / f"{K.PREREG_ID}.json"), "prereg_v1": _r(pr / f"{C.PREREG_ID}.json"),
        "hyper_main": _r(ev / "hyper_main.json"), "hyper_t1": _r(ev / "hyper_t1.json"),
        "hyper_t1f": _r(ev / "hyper_t1f.json"),
        "t1f": _r(ev / "t1f_targets.parquet"), "t1": _r(ev / "t1_targets.parquet"),
        "t2": _r(ev / "t2_targets.parquet"), "t2b": _r(ev / "t2b_targets.parquet"),
    }


def rows_for(nid: str, kind: str, note: str, pv: str, S: dict) -> list[dict]:
    pre_aux = [S["ring_st"], S["gnss_full_v4"], S["grid"]]
    pre_obs = [S["lev"], S["gnss_pre_v4"], S["insar_p2"], S["piles"]] + S["rings"]
    t1f_obs = [S["lev"], S["gnss_pre_v4"], S["piles"]] + S["rings"]
    roles: dict[str, list] = {}
    if kind == "diag_t1":
        roles = {"input": [S["lev"], S["gnss_pre_v2"], S["insar_p"], S["piles"]] + S["rings"]
                 + [S["r1_hyper"], S["r1_hyper_t1"]], "design_time_evidence": [S["r1_t1"]]}
    elif kind == "diag_lev":
        roles = {"input": [S["lev"], S["grid"]]}
    elif kind == "diag_insar":
        roles = {"input": [S["insar_p"], S["lev"], S["grid"]]}
    elif kind == "diag_t2":
        roles = {"design_time_evidence": [S["r1_t2"]]}
    elif kind == "diag_ac":
        roles = {"design_time_evidence": [S["r1_t3g"], S["r1_t3c"]], "input": [S["r1_hyper"]]}
    elif kind == "diag_gnss":
        roles = {"input": [S["gnss_pre_v2"], S["gnss_pre_v4"]]}
    elif kind == "hyper_t1f":
        roles = {"fit_target": t1f_obs, "input": pre_aux + [S["prereg_r2"]]}
    elif kind == "hyper_main":
        roles = {"fit_target": pre_obs, "input": pre_aux + [S["prereg_r2"]]}
    elif kind == "hyper_t1":
        roles = {"fit_target": [S["lev"]], "input": [S["hyper_main"], S["prereg_r2"]]}
    elif kind == "t1f":
        roles = {"input": t1f_obs + pre_aux + [S["hyper_t1f"], S["prereg_r2"]],
                 "design_time_evidence": [S["t1f"]]}
    elif kind == "t1":
        roles = {"input": pre_obs + pre_aux + [S["hyper_main"], S["hyper_t1"], S["prereg_r2"]],
                 "design_time_evidence": [S["t1"]]}
    elif kind in ("t2", "t2b"):
        roles = {"input": pre_obs + pre_aux + [S["hyper_main"], S["prereg_r2"]],
                 "design_time_evidence": [S[kind]]}
    elif kind == "t3_gnss":
        roles = {"input": [S["lev"], S["insar_p2"], S["piles"]] + S["rings"] + pre_aux
                 + [S["hyper_main"], S["prereg_r2"]], "design_time_evidence": [S["gnss_pre_v4"]]}
    elif kind == "t3_cmp":
        roles = {"input": [S["lev"], S["gnss_pre_v4"], S["insar_p2"], S["ring_st"],
                           S["gnss_full_v4"], S["grid"], S["hyper_main"], S["prereg_r2"]],
                 "design_time_evidence": [S["piles"]] + S["rings"]}
    elif kind == "t3_insar":
        roles = {"input": [S["lev"], S["gnss_pre_v4"], S["piles"]] + S["rings"] + pre_aux
                 + [S["hyper_main"], S["prereg_r2"]], "design_time_evidence": [S["insar_p2"]]}
    elif kind == "product":
        roles = {"input": [S["lev"], S["gnss_full_v4"], S["insar_v3"], S["piles"],
                           S["rings_bridged"], S["ring_st"], S["grid"], S["hyper_main"]]
                 + S["rings"]}
    else:
        raise ValueError(f"no ledger rule for kind {kind!r} ({nid})")
    out = []
    for role, srcs in roles.items():
        for s in srcs:
            out.append({"number_id": nid, "source": s, "role": role, "product_version": pv,
                        "note": note})
    return out


def main() -> int:
    nums = json.loads((K.EVID / "report_numbers.json").read_text())
    S = sources()
    rws = []
    for nid, d in nums["numbers"].items():
        rws += rows_for(nid, d["kind"], d["note"], nums["product_version"], S)
    p = ledger.append_rows(C.WP, rws)
    allr = ledger.read_all()
    res = {"fragment": rel_to_repo(p), "rows_written": len(rws), "numbers": len(nums["numbers"]),
           "conflicts_all_fragments": len(ledger.conflicts(allr)),
           "rows_all_fragments": len(allr),
           "unledgered": ledger.unledgered(list(nums["numbers"]), allr)}
    (K.EVID / "ledger_result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
