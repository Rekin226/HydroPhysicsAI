"""WP S5: ledger rows (plan 4.1, U6) for every number the S5 report quotes.

    python -I s5ledger.py        (hydro env) -> results/dataprep/ledger/rows/S5.csv

One role per source per number. Conventions (as H2 and S4b):

* procedure H (hyperparameters, depth fractions): the pre-origin observation sources are
  ``fit_target`` (restricted marginal likelihood);
* registered tests: the training sources are ``input``; the withheld observations are the
  ``validator``, recorded through the test's target table in S5_evidence (it holds exactly the
  withheld values; the same L0 file's other rows are the training ``input``);
* product and after-the-fact numbers: every source is ``input`` (the T1/T2 target tables are
  ``validator`` of the breakdowns of T1/T2).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import s5const as C  # noqa: E402
from common import ledger  # noqa: E402
from common.paths import rel_to_repo  # noqa: E402


def _r(p) -> str:
    return rel_to_repo(p)


def sources() -> dict:
    from s1_rings import s1const as K1
    ev = C.EVIDENCE
    return {
        "lev": _r(C.LEVELING_PANEL), "ring_st": _r(C.RING_STATIONS),
        "rings": [_r(C.RING_DIR / f"{s}.parquet") for s in K1.SITES],
        "gnss_pre": _r(C.GNSS_PRE), "gnss_full": _r(C.GNSS_FULL),
        "insar_p": _r(C.WORK / "insar_runP_cells.npz"), "insar_v2": _r(C.INSAR_V2),
        "piles": _r(C.PILES), "rings_bridged": _r(C.RINGS_BRIDGED), "grid": _r(C.TWIN_GRID),
        "units": _r(C.UNITS_V3), "prereg": _r(C.DATAPREP / "preregistration" /
                                             f"{C.PREREG_ID}.json"),
        "hyper_pre": _r(ev / "hyper_pre.json"), "hyper_t1": _r(ev / "hyper_t1.json"),
        "t1": _r(ev / "t1_targets.parquet"), "t2": _r(ev / "t2_targets.parquet"),
        "t3_gnss": _r(ev / "t3_gnss_targets.parquet"), "t3_cmp": _r(ev / "t3_cmp_targets.parquet"),
        "t3_insar": _r(ev / "t3_insar_units.parquet"),
        "r2": {f: _r(ev / f"r2_{f}_targets.parquet") for f in C.AFTER_FAMILIES},
    }


def rows(number_id: str, pv: str, note: str, **roles) -> list[dict]:
    out = []
    for role, srcs in roles.items():
        for s in srcs:
            out.append({"number_id": number_id, "source": s, "role": role,
                        "product_version": pv, "note": note})
    return out


def build(pv: str, numbers: dict) -> list[dict]:
    """``numbers``: {number_id: note}; the id prefix decides the sources and roles."""
    S = sources()
    pre_obs = [S["lev"], S["gnss_pre"], S["insar_p"], S["piles"]] + S["rings"]
    pre_aux = [S["ring_st"], S["gnss_full"], S["grid"]]
    full_obs = [S["lev"], S["gnss_full"], S["insar_v2"], S["piles"], S["rings_bridged"]] + \
        S["rings"]
    full_aux = [S["ring_st"], S["grid"], S["hyper_pre"]]
    out = []
    for nid, note in numbers.items():
        if nid.startswith("S5.hyper."):
            out += rows(nid, pv, note, fit_target=pre_obs, input=pre_aux + [S["prereg"]])
        elif nid.startswith("S5.T1."):
            out += rows(nid, pv, note, input=pre_obs + pre_aux + [S["hyper_pre"], S["hyper_t1"],
                                                                   S["prereg"]],
                        validator=[S["t1"]])
        elif nid.startswith("S5.T2."):
            out += rows(nid, pv, note, input=pre_obs + pre_aux + [S["hyper_pre"], S["prereg"]],
                        validator=[S["t2"]])
        elif nid.startswith("S5.T3.gnss"):
            out += rows(nid, pv, note,
                        input=[S["lev"], S["insar_p"], S["piles"]] + S["rings"] + pre_aux
                        + [S["hyper_pre"], S["prereg"]],
                        validator=[S["gnss_pre"], S["t3_gnss"]])
        elif nid.startswith("S5.T3.cmp"):
            out += rows(nid, pv, note,
                        input=[S["lev"], S["gnss_pre"], S["insar_p"]] + pre_aux
                        + [S["hyper_pre"], S["prereg"]],
                        validator=[S["piles"], S["t3_cmp"]] + S["rings"])
        elif nid.startswith("S5.T3.insar"):
            out += rows(nid, pv, note,
                        input=[S["lev"], S["gnss_pre"], S["piles"]] + S["rings"] + pre_aux
                        + [S["hyper_pre"], S["prereg"]],
                        validator=[S["insar_p"], S["t3_insar"]])
        elif nid.startswith("S5.R2."):
            fam = nid.split(".")[2]
            out += rows(nid, pv, note, input=full_obs + full_aux + [S["prereg"]],
                        validator=[S["r2"][fam]])
        elif nid.startswith("S5.product."):
            out += rows(nid, pv, note, input=full_obs + full_aux + [S["units"]])
        elif nid.startswith("S5.diag.t1"):
            out += rows(nid, pv, note, input=pre_obs + pre_aux + [S["hyper_t1"], S["units"]],
                        validator=[S["t1"]])
        elif nid.startswith("S5.diag.t2"):
            out += rows(nid, pv, note, input=pre_obs + pre_aux + [S["hyper_pre"], S["units"]],
                        validator=[S["t2"]])
        elif nid.startswith("S5.diag."):
            out += rows(nid, pv, note, input=pre_obs + pre_aux + [S["hyper_pre"]])
        else:
            raise ValueError(f"no ledger rule for {nid}")
    return out


def main() -> int:
    nums = json.loads((C.EVIDENCE / "ledger_numbers.json").read_text())
    rws = build(nums["product_version"], nums["numbers"])
    p = ledger.append_rows(C.WP, rws)
    allr = ledger.read_all()
    print(json.dumps({"fragment": str(p), "rows_added_or_present": len(rws),
                      "numbers": len(nums["numbers"]),
                      "conflicts_all_fragments": len(ledger.conflicts(allr)),
                      "rows_all_fragments": len(allr)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
