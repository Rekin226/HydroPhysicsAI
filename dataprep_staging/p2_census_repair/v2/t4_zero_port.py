"""Registered test P2r2.c1.zero_port_yunlin (P2c): does the zero classification, ported to
the C0 v3 Yunlin record, reproduce the billing pilot's class counts and irrigation zero
seasonality? (pre-registration P2abc.r2.c0_based.v1)

Pilot view of the C0 full Yunlin record: months 2007-01..2025-07 with the July-2025 reading of
every code-M2 meter set to 0 (the AMP_V2 cache's partial month, which the pilot saw: its
Jul-Aug 2025 bill was not issued); kept set = C0's; billing phase detected by the pilot rule on
2012-01..2025-06; kz = kept meters with phase A, B or monthly; window 2012-01..2025-06; the
pilot's s05(e) run loop (zero = reading exactly 0; leading / interior / trailing kinds;
irrigation, aquaculture and livestock zero frequency by calendar month inside each meter's
first..last use span).

Gating (each within 1 % of the pilot): zero-months in interior runs <= 12, interior runs > 12,
trailing runs >= 12 and trailing runs < 12 (references: the pilot's run table as reproduced
exactly by round 1, results/dataprep/yunlin_electricity/_work_p2abc/zeros.json, itself
reproduced byte-identically by r1_reproduce.py); irrigation zero frequency in April and August
within 1 % of 0.0558 and 0.1563 (pilot census_zeros.json). Reported only (they depend on how
pre-install months are coded, 0 in C0 versus partly NaN in the cache): leading runs, the total
zero-month count, the run table cell by cell, and C0's pre-install meter-months.

Usage: python -I t4_zero_port.py <out_dir>
"""
from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2lib as L  # noqa: E402
import scoring  # noqa: E402

BINS = [0, 1, 2, 4, 6, 12, 24, 1000]
BIN_LABELS = ["1", "2", "3-4", "5-6", "7-12", "13-24", ">24"]
PILOT_ZERO_APR, PILOT_ZERO_AUG = 0.0558, 0.1563
R1_ZEROS = L.PRODUCT_ROOT / "_work_p2abc" / "zeros.json"
PILOT_RUNS = L.DATAPREP / "_pilots" / "billing_and_census_gaps" / "zero_runs_summary.csv"


def code_sha() -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), L.CODE_DIR / "p2lib.py")}


def runs_of(z: np.ndarray):
    d = np.diff(np.concatenate([[0], z.astype(int), [0]]))
    return np.where(d == 1)[0], np.where(d == -1)[0]


def pilot_stats(M: np.ndarray, cls: np.ndarray, phase: np.ndarray) -> dict:
    T = M.shape[1]
    kz = np.isin(phase, ["A", "B", "monthly"])
    Mz = np.nan_to_num(M)
    nz = Mz > 0
    first_nz = np.where(nz.any(1), nz.argmax(1), -1)
    last_nz = np.where(nz.any(1), T - 1 - nz[:, ::-1].argmax(1), -1)
    W0, W1 = L.tidx("2012-01"), L.tidx("2025-06") + 1
    mo = L.DATES[W0:W1].month.to_numpy() - 1
    cls_idx = {"irrigation": 0, "aquaculture": 1, "livestock": 2}
    moy_zero = np.zeros((3, 12))
    moy_cnt = np.zeros((3, 12))
    rows = []
    for j in np.where(kz)[0]:
        x = M[j, W0:W1]
        if not np.isfinite(x).any() or np.nansum(x) <= 0:
            continue
        f, l_ = first_nz[j] - W0, last_nz[j] - W0
        z = x == 0
        for s_, e_ in zip(*runs_of(z), strict=True):
            kind = "leading" if s_ == 0 and f > 0 else (
                "trailing" if e_ == len(x) and l_ < len(x) - 1 else "interior")
            rows.append((cls[j], kind, e_ - s_))
        ci = cls_idx.get(cls[j])
        if ci is not None:
            inside = np.zeros(len(x), bool)
            inside[max(f, 0): l_ + 1] = True
            ok = inside & np.isfinite(x)
            np.add.at(moy_zero[ci], mo[ok], z[ok])
            np.add.at(moy_cnt[ci], mo[ok], 1)
    runs = pd.DataFrame(rows, columns=["cls", "kind", "len"])
    runs["len_bin"] = pd.cut(runs["len"], BINS, labels=BIN_LABELS)
    zmoy = moy_zero / np.maximum(moy_cnt, 1)
    out = {"zero_months_total": int(runs["len"].sum()),
           "interior_le12_months": int(runs.loc[(runs.kind == "interior") & (runs.len <= 12),
                                                "len"].sum()),
           "interior_gt12_months": int(runs.loc[(runs.kind == "interior") & (runs.len > 12),
                                                "len"].sum()),
           "trailing_ge12_months": int(runs.loc[(runs.kind == "trailing") & (runs.len >= 12),
                                                "len"].sum()),
           "trailing_lt12_months": int(runs.loc[(runs.kind == "trailing") & (runs.len < 12),
                                                "len"].sum()),
           "leading_months": int(runs.loc[runs.kind == "leading", "len"].sum()),
           "irrigation_zero_moy": zmoy[0].tolist(), "n_kz_meters": int(kz.sum())}
    counts = runs.groupby(["cls", "kind", "len_bin"], observed=True).size()
    return out, counts


def pilot_reference() -> dict:
    r1 = L.load_json(R1_ZEROS)["pilot_reproduction"]
    tot = r1["zero_months_total"]
    sh = r1["month_share_kind_bin"]
    interior = sum(v for k, v in sh.items() if k.startswith("interior|")) * tot
    trailing = sum(v for k, v in sh.items() if k.startswith("trailing|")) * tot
    ile12 = r1["interior_le12_share"] * tot
    tge12 = r1["trailing_ge12_share"] * tot
    return {"zero_months_total": tot, "interior_le12_months": ile12,
            "interior_gt12_months": interior - ile12, "trailing_ge12_months": tge12,
            "trailing_lt12_months": trailing - tge12,
            "leading_months": sum(v for k, v in sh.items() if k.startswith("leading|")) * tot,
            "pre_install_zero_all_meters": r1["pre_install_months"]["zero"],
            "irrigation_zero_apr": PILOT_ZERO_APR, "irrigation_zero_aug": PILOT_ZERO_AUG}


def main(out_dir: str) -> None:
    t0 = time.time()
    res = {"code_sha256": code_sha()}
    res["authorization"] = scoring.authorize(["P2r2.c1.zero_port_yunlin"])
    cd = L.CountyData("full", "yunlin")
    T = L.tidx("2025-07") + 1
    M = cd.M[:, :T].astype(np.float64).copy()
    M[cd.code == "M2", T - 1] = 0.0
    phase = L.detect_phase(M, L.tidx("2012-01"), L.tidx("2025-06") + 1)
    mine, counts = pilot_stats(M, cd.mclass, phase)
    ref = pilot_reference()
    checks = {}
    for k in ("interior_le12_months", "interior_gt12_months", "trailing_ge12_months",
              "trailing_lt12_months"):
        rel = abs(mine[k] - ref[k]) / ref[k]
        checks[k] = {"mine": mine[k], "pilot": ref[k], "rel_diff": rel, "pass": rel <= 0.01}
    for k, m_ in (("irrigation_zero_apr", 3), ("irrigation_zero_aug", 7)):
        v = mine["irrigation_zero_moy"][m_]
        rel = abs(v - ref[k]) / ref[k]
        checks[k] = {"mine": v, "pilot": ref[k], "rel_diff": rel, "pass": rel <= 0.01}
    res["checks"] = checks
    res["passed"] = bool(all(c["pass"] for c in checks.values()))
    pil = pd.read_csv(PILOT_RUNS).set_index(["cls", "kind", "len_bin"])["0"]
    both = pd.concat([counts.rename("mine"), pil.rename("pilot")], axis=1).fillna(0)
    rel = (both["mine"] - both["pilot"]).abs() / both["pilot"].clip(lower=1)
    res["reported"] = {
        "zero_months_total": {"mine": mine["zero_months_total"],
                              "pilot": ref["zero_months_total"]},
        "leading_months": {"mine": mine["leading_months"], "pilot": ref["leading_months"]},
        "run_table_cells": int(len(both)), "run_table_max_rel_diff": float(rel.max()),
        "run_table_cells_over_1pct": int((rel > 0.01).sum()),
        "run_table_worst": both.assign(rel=rel).sort_values("rel", ascending=False).head(10)
        .reset_index().astype(str).to_dict(orient="records"),
        "c0_pre_install_meter_months_kept_yunlin": int((cd.Mm == 1).sum()),
        "pilot_pre_install_zero_all_meters": ref["pre_install_zero_all_meters"],
        "phase_counts": pd.Series(phase).value_counts().to_dict(),
        "n_kz_meters": mine["n_kz_meters"], "irrigation_zero_moy": mine["irrigation_zero_moy"]}
    res["seconds"] = round(time.time() - t0, 1)
    L.save_json(res, Path(out_dir) / "t4_zero_port.json")
    print({k: (round(v["rel_diff"], 5), v["pass"]) for k, v in checks.items()}, res["passed"])


if __name__ == "__main__":
    main(sys.argv[1])
