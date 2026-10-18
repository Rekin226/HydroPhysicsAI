"""Registered test P2r3b.c1.zero_shipped_yunlin (pre-registration P2abc.r3.repair.v2): does the
SHIPPED zero classifier (``zeros3.classify_core``, the function that writes
``zero_class.parquet`` of yunlin_electricity v2) reproduce the billing pilot's class counts
and irrigation zero seasonality (plan P2c acceptance)?

Pilot view of the C0 full v3 Yunlin record (as round 2's t4): months 2007-01..2025-07, months
that are not observations NaN (``zeros3.observed_matrix``), the July-2025 reading of every
code-M2 meter set to 0 (the AMP_V2 cache's partial month the pilot saw), billing phase by the
pilot rule on 2012-01..2025-06, kz = kept meters with phase A, B or monthly, window
2012-01..2025-06.

Gating items (each within 1 %):
* zero-months of kz meters in the shipped classes idle (interior runs <= 12), long_idle
  (interior > 12), terminated (trailing >= 12) and trailing_short (trailing < 12) against the
  pilot's run table (as reproduced exactly by round 1: _work_p2abc/zeros.json);
* irrigation zero share in April and August (zero rows of the shipped output inside each kz
  irrigation meter's first..last use span over the finite months of that span) against 0.0558
  and 0.1563 (pilot census_zeros.json);
* structural (pre-install) meter-months under the pilot's definition with month-precision
  install dates (months before the install month minus one, meters whose install month index
  exceeds 12, record 2007-01..2025-07), kept meters: the shipped ``structural`` rule (C0 method
  pre_install) on C0's kept Yunlin meters against the pilot's own archive (meter_mat.npy,
  meters_phase.parquet: inst floored to its month, dropped) for the pilot's kept meters,
  counting months read 0 or missing (C0 codes both 0). The pilot's s05 statistic itself
  (install index = first month start on or after the install date, all meters, zero readings:
  the plan's 878,738) is recomputed and reported.

Reported only: not_yet_used (leading runs), never_used_in_window, the literal plan figure
878,738 (all meters, zero readings only) and its decomposition, C0's whole-record count.

Usage: python -I t6_zero_shipped.py <out_dir>
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
for _p in (_HERE.parents[2], _HERE.parents[1], _HERE.parent / "v2", _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import p3lib as P3  # noqa: E402
import scoring  # noqa: E402  (v2: authorize)
import zeros3  # noqa: E402
from p3lib import L  # noqa: E402

PREREG3 = "P2abc.r3.repair.v2"
TEST = "P2r3b.c1.zero_shipped_yunlin"
PILOT = L.DATAPREP / "_pilots" / "billing_and_census_gaps"
R1_ZEROS = L.PRODUCT_ROOT / "_work_p2abc" / "zeros.json"
PILOT_ZERO_APR, PILOT_ZERO_AUG = 0.0558, 0.1563
PILOT_STRUCT_LITERAL = 878738


def pilot_run_reference() -> dict:
    r1 = L.load_json(R1_ZEROS)["pilot_reproduction"]
    tot = r1["zero_months_total"]
    sh = r1["month_share_kind_bin"]
    interior = sum(v for k, v in sh.items() if k.startswith("interior|")) * tot
    trailing = sum(v for k, v in sh.items() if k.startswith("trailing|")) * tot
    ile12 = r1["interior_le12_share"] * tot
    tge12 = r1["trailing_ge12_share"] * tot
    return {"idle": ile12, "long_idle": interior - ile12, "terminated": tge12,
            "trailing_short": trailing - tge12,
            "not_yet_used": sum(v for k, v in sh.items() if k.startswith("leading|")) * tot,
            "zero_months_total": tot,
            "pre_install_all_meters": r1["pre_install_months"]}


def pilot_structural(T: int) -> dict:
    """The pilot's s05 pre-install statistic on its own archive: (a) exactly as s05 (install
    index = first month start on or after the install date; all meters, then kept meters) and
    (b) with the install month (the month containing the install date), the month convention
    of C0's install_month, used for the gating comparison."""
    M = np.load(PILOT / "meter_mat.npy").astype(float)[:, :T]
    mt = pd.read_parquet(PILOT / "meters_phase.parquet", columns=["inst", "dropped"])
    dates = pd.DatetimeIndex(np.load(PILOT / "dates.npy"))[:T]
    ok = mt["inst"].notna().to_numpy()
    ti_s05 = np.full(len(mt), -1)
    ti_s05[ok] = dates.searchsorted(pd.DatetimeIndex(mt["inst"][ok]))
    ti_month = np.full(len(mt), -1)
    ti_month[ok] = dates.searchsorted(pd.DatetimeIndex(mt["inst"][ok]), side="right") - 1
    keep = ~mt["dropped"].fillna(False).to_numpy().astype(bool)
    out = {}
    for conv, ti in (("s05", ti_s05), ("install_month", ti_month)):
        for name, pop in (("all_meters", np.ones(len(mt), bool)), ("kept_meters", keep)):
            z = n = p = 0
            for j in np.where((ti > 12) & pop)[0]:
                seg = M[j, : ti[j] - 1]
                z += int((seg == 0).sum())
                n += int(np.isnan(seg).sum())
                p += int((seg > 0).sum())
            out[f"{conv}|{name}"] = {"zero": z, "nan": n, "positive": p, "zero_or_nan": z + n,
                                     "n_meters": int(pop.sum())}
    return out


def shipped_structural(cd: L.CountyData, T: int) -> dict:
    inst = cd.meters["install_month"].astype(str).to_numpy()
    # month index relative to 2007-01 (negative before the record; those meters drop out)
    ti = np.array([int(m[:4]) * 12 + int(m[5:7]) - (2007 * 12 + 1)
                   if (len(m) == 7 and m[4] == "-" and m[:4].isdigit() and m[5:7].isdigit())
                   else -1 for m in inst])
    pre = cd.Mm[:, :T] == 1
    tt = np.arange(T)[None, :]
    conv = (ti[:, None] > 12) & (tt < (ti[:, None] - 1))
    return {"pilot_convention": int((pre & conv).sum()),
            "observed_before_install_pilot_convention": int(((cd.Mm[:, :T] == 2) & conv).sum()),
            "whole_record_no_slack": int(pre.sum()),
            "n_meters": int(len(ti)), "n_meters_install_gt12": int((ti > 12).sum())}


def main(out_dir: str) -> None:
    t0 = time.time()
    res = {"code_sha256": P3.code_sha(Path(__file__), P3.V3 / "zeros3.py", P3.V3 / "p3lib.py",
                                      L.CODE_DIR / "p2lib.py", L.CODE_DIR / "b2_zeros.py")}
    res["authorization"] = scoring.authorize([TEST], prereg_id=PREREG3)
    cd = L.CountyData("full", "yunlin")
    T = L.tidx("2025-07") + 1
    X = zeros3.observed_matrix(cd, T - 1)
    X[cd.code == "M2", T - 1] = 0.0
    W0, W1 = L.tidx("2012-01"), L.tidx("2025-06")
    phase = L.detect_phase(X, W0, W1 + 1)
    kz = np.isin(phase, ["A", "B", "monthly"])
    r = zeros3.classify_core(X, W0, W1)
    sel = kz[r["row"]]
    zc = pd.Series(np.asarray(zeros3.ZCLASSES)[r["zclass"][sel]]).value_counts()
    mine = {k: int(zc.get(k, 0)) for k in zeros3.ZCLASSES}
    ref = pilot_run_reference()
    checks = {}
    for k in ("idle", "long_idle", "terminated", "trailing_short"):
        rel = abs(mine[k] - ref[k]) / ref[k]
        checks[k] = {"shipped": mine[k], "pilot": ref[k], "rel_diff": rel, "pass": rel <= 0.01}
    # irrigation seasonality from the shipped rows
    irr = (cd.mclass == "irrigation") & kz & r["use_in_window"]
    first, last = r["first"], r["last"]
    lo = np.maximum(first, W0)
    hi = np.minimum(last, W1)
    tt = np.arange(W0, W1 + 1)
    span = (tt[None, :] >= lo[:, None]) & (tt[None, :] <= hi[:, None]) & irr[:, None]
    fin = np.isfinite(X[:, W0:W1 + 1]) & span
    moy = L.DATES[W0:W1 + 1].month.to_numpy()
    den = np.array([fin[:, moy == m].sum() for m in range(1, 13)], float)
    rw, tw = r["row"], r["t"]
    insp = irr[rw] & (tw >= lo[rw]) & (tw <= hi[rw]) & (r["kind"] != 3)
    num = np.bincount(L.DATES[tw[insp]].month.to_numpy() - 1, minlength=12).astype(float)
    zmoy = num / np.maximum(den, 1)
    for k, m_, refv in (("irrigation_zero_apr", 3, PILOT_ZERO_APR),
                        ("irrigation_zero_aug", 7, PILOT_ZERO_AUG)):
        rel = abs(zmoy[m_] - refv) / refv
        checks[k] = {"shipped": float(zmoy[m_]), "pilot": refv, "rel_diff": rel,
                     "pass": rel <= 0.01}
    ps = pilot_structural(T)
    ss = shipped_structural(cd, T)
    pk = ps["install_month|kept_meters"]["zero_or_nan"]
    rel = abs(ss["pilot_convention"] - pk) / pk
    checks["structural_kept_meters"] = {"shipped": ss["pilot_convention"], "pilot": pk,
                                        "rel_diff": rel, "pass": rel <= 0.01}
    res["checks"] = checks
    res["passed"] = bool(all(c["pass"] for c in checks.values()))
    res["reported"] = {
        "not_yet_used": {"shipped": mine["not_yet_used"], "pilot": ref["not_yet_used"]},
        "never_used_in_window_kz_rows": mine["never_used_in_window"],
        "zero_months_total_pilot_classes": {"shipped": int(sum(mine[k] for k in (
            "not_yet_used", "idle", "long_idle", "terminated", "trailing_short"))),
            "pilot": ref["zero_months_total"]},
        "irrigation_zero_moy": zmoy.tolist(),
        "structural": {"plan_literal_all_meters_zero_only": PILOT_STRUCT_LITERAL,
                       "pilot_archive": ps, "shipped": ss,
                       "s05_reproduced_exactly": ps["s05|all_meters"]["zero"]
                       == PILOT_STRUCT_LITERAL,
                       "round1_pilot_reproduction": ref["pre_install_all_meters"]},
        "phase_counts": pd.Series(phase).value_counts().to_dict(),
        "n_kz_meters": int(kz.sum())}
    res["seconds"] = round(time.time() - t0, 1)
    L.save_json(res, Path(out_dir) / "t6_zero_shipped.json")
    print({k: (round(v["rel_diff"], 5), v["pass"]) for k, v in checks.items()}, res["passed"])


if __name__ == "__main__":
    main(sys.argv[1])
