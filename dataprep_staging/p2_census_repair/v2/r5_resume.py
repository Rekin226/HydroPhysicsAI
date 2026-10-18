"""WP P2abc round 2, resumed session (2026-10-14): look records and report facts.

The interrupted round-2 session built, scored, gated and published
``yunlin_electricity/{preorigin,full}/v1`` but stopped before the WP report and the ledger.
This script does not rebuild or re-score anything. It

* ``log_review``: logs the resumed session's reading of the already-scored results (validation
  files, test outputs, stage-1 and design counts) in the A1 look log, before the facts run;
* ``facts``: reads the two published products (read-only) and writes
  ``results/dataprep/reports/P2abc_evidence/report_facts.json``:
    - array inventory of every npz / parquet output (names, shapes, dtypes; no values of uid),
      the conventions P2def builds on;
    - independent spot checks, separate from the registered construction checks of
      ``t5_construction.py``: per-pump total over the Chow-Lin window equals C0's (the split
      keeps every bill), the recorded arm equals C0's cells where nothing was imputed, annual
      in-grid energy per county and arm, energy moved by the split, seasonal amplitude of the
      in-grid irrigation total per arm, flag counts, flag-3 relative sd;
    - a digest of the registered test outputs (no recomputation).

Usage: python -I r5_resume.py log_review | facts
"""
from __future__ import annotations

import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from numpy.lib import format as npformat

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2lib as L  # noqa: E402

PROD = {v: L.PRODUCT_ROOT / v / "v1" for v in ("preorigin", "full")}
TESTS = L.EVID / "tests"
OUT = L.EVID / "report_facts.json"


# ----------------------------------------------------------------------------- looks
def log_review() -> None:
    import looks_p2 as LK

    LK.record(
        "P2-resume-review",
        {"P2r2:reassembled_preorigin": "published yunlin_electricity/preorigin/v1",
         "P2r2:reassembled_full": "published yunlin_electricity/full/v1",
         "P2r2:val_meters_yunlin": "scored validation meters (re-read)",
         "P2r2:val_meters_changhua": "scored validation meters (re-read)",
         "P2r2:val_meters_chiayi": "scored validation meters (re-read)",
         "P2r2:july_units_yunlin": "scored July units (re-read)",
         "P2r2:july_units_changhua": "scored July units (re-read)",
         "P2r2:july_units_chiayi": "scored July units (re-read)",
         "P2r2:zero_port_yunlin": "scored zero port (re-read)"},
        "c0_pump_electricity_v3:2007-01..2026-07", "report_review",
        "resumed session (2026-10-14) re-read the already-scored results of "
        "P2abc.r2.c0_based.v1 and P2abc.r2.k1_repair.v1 (validation.json of both products, "
        "P2abc_evidence/tests/*.json and logs, stage1.json, d0_design_counts.json, "
        "build_info.json, checker and guard-probe outputs) to write the WP report; nothing "
        "rebuilt or re-scored.")
    # the head-check well units carry their station ids in the pre-registration: no member
    # override (LK.record would declare them memberless electricity objects)
    tag = "P2-resume-review-heads"
    if f"[{tag}]" in LK.logged_tags():
        print("already logged:", tag)
        return
    rec = LK.prereg3.record_look(
        ["P2r2:head_check_wells_changhua", "P2r2:head_check_wells_yunlin"],
        "heads:2012-01..2022-11", L.WP, "report_review",
        note=f"[{tag}] resumed session (2026-10-14) re-read the summary statistics of the "
             "already-scored head checks (t2_head_check_round1.json for P2a.head_placebo.v1, "
             "t2_head_check_round2.json for P2r2.a3.*); no head series was read.")
    print("logged", tag, rec.get("time"))


# ----------------------------------------------------------------------------- inventory
def npz_inventory(path: Path) -> dict:
    out = {}
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            with z.open(n) as f:
                v = npformat.read_magic(f)
                shape, _fo, dt = npformat._read_array_header(f, v)
            out[n[:-4]] = {"shape": list(shape), "dtype": str(dt)}
    return out


def parquet_inventory(path: Path) -> dict:
    f = pq.ParquetFile(path)
    sch = f.schema_arrow
    return {"rows": f.metadata.num_rows,
            "columns": {fld.name: str(fld.type) for fld in sch}}


def inventory(variant: str) -> dict:
    d = PROD[variant]
    inv = {}
    for p in sorted(d.iterdir()):
        if p.suffix == ".npz":
            inv[p.name] = npz_inventory(p)
        elif p.suffix == ".parquet":
            inv[p.name] = parquet_inventory(p)
        else:
            inv[p.name] = {"bytes": p.stat().st_size}
    return inv


# ----------------------------------------------------------------------------- checks
def pump_window_totals(variant: str) -> dict:
    """Per pump, the sum over the Chow-Lin window of (Chow-Lin arm - C0) must be 0: every
    re-split pair lies inside the window and keeps its bill; months outside re-split pairs
    keep C0's value (the July-2026 imputation month of the full variant is excluded)."""
    # The window runs to the variant end: the last phase-B pair of the full variant is
    # (2026-06, 2026-07). The full variant's July-2026 imputation (method 11, open phase-A
    # bills that C0 leaves NaN) is set to 0 before the comparison and counted separately.
    # (A first run of this check stopped the full window at 2026-06, which cuts that last
    # phase-B pair and showed spurious differences; corrected before the report.)
    V = L.VARIANTS[variant]
    lo = V["cl0"]
    hi = V["T"]
    res = {}
    for c in L.COUNTIES:
        tag = L.FILETAG[c]
        p2 = np.load(PROD[variant] / f"pumps_{tag}.npz", allow_pickle=False)
        c0 = np.load(L.C0_DIR[variant] / f"pumps_{tag}.npz", allow_pickle=False)
        if not (p2["uid"] == c0["uid"]).all():
            res[c] = {"uid_order_equal": False}
            continue
        imp = p2["method"][:, lo:hi] == L.M_JULY_IMPUTED
        a = np.where(imp, 0.0, np.nan_to_num(p2["value"][:, lo:hi]))
        b = np.nan_to_num(c0["value"][:, lo:hi])
        d = a.sum(1) - b.sum(1)
        scale = np.abs(b).sum(1)
        m3 = p2["method"][:, lo:hi] == L.M_CHOWLIN
        nanpat = int(((np.isnan(p2["value"][:, lo:hi]) != np.isnan(c0["value"][:, lo:hi]))
                      & ~imp).sum())
        moved = float(np.abs(a - b).sum())
        res[c] = {
            "uid_order_equal": True,
            "n_pumps": int(len(d)),
            "window": [L.ym(lo), L.ym(hi - 1)],
            "max_abs_total_diff_kwh": float(np.abs(d).max()),
            "max_rel_total_diff": float((np.abs(d) / np.maximum(scale, 1e-9)).max()),
            "nan_pattern_differences_outside_imputation": nanpat,
            "pump_months_july2026_imputed": int(imp.sum()),
            "pump_months_chowlin_split": int(m3.sum()),
            "energy_moved_share_in_window": moved / float(np.abs(b).sum()),
        }
    return res


def cell_checks(variant: str) -> dict:
    T = L.VARIANTS[variant]["T"]
    dates = list(L.DATES[:T].strftime("%Y-%m"))
    rec = np.load(PROD[variant] / "kwh_by_class_recorded.npz", allow_pickle=False)
    cl = np.load(PROD[variant] / "kwh_by_class_chowlin.npz", allow_pickle=False)
    c0 = np.load(L.C0_DIR[variant] / "cells_by_class.npz", allow_pickle=False)
    out = {"dates_equal": bool(list(rec["dates"]) == dates == list(cl["dates"]))}
    # recorded arm vs C0 where no pump was imputed
    noimp = rec["n_imp"] == 0
    a, b = rec["value"], c0["value"]
    both = noimp & np.isfinite(a) & np.isfinite(b)
    out["recorded_vs_c0"] = {
        "cell_months_compared": int(both.sum()),
        "max_abs_diff_kwh": float(np.abs(a[both] - b[both]).max()),
        "nan_pattern_differences_no_imputation": int(
            (np.isnan(a) != np.isnan(b))[noimp].sum()),
        "flag_differences_no_imputation": int((rec["flag"] != c0["flag"])[noimp].sum()),
    }
    yr = np.array([int(s[:4]) for s in dates])
    irr = L.CLASSES.index("irrigation")
    counties = list(rec["counties"])
    annual, moved, season, flags, relsd = {}, {}, {}, {}, {}
    for k, c in enumerate(counties):
        vr = np.nan_to_num(rec["value_by_county"][k])     # (C, A, T)
        vc = np.nan_to_num(cl["value_by_county"][k])
        tot_r = vr.sum((0, 1))
        tot_c = vc.sum((0, 1))
        ann = {}
        for y in range(2012, int(yr.max()) + 1):
            s = yr == y
            if s.sum() < 12:
                continue
            ann[str(y)] = {"recorded_GWh": float(tot_r[s].sum() / 1e6),
                           "chowlin_GWh": float(tot_c[s].sum() / 1e6)}
        annual[c] = ann
        s1222 = (yr >= 2012) & (yr <= 2022)
        annual[c]["mean_2012_2022"] = {
            "recorded_GWh_per_yr": float(tot_r[s1222].sum() / 11e6),
            "chowlin_GWh_per_yr": float(tot_c[s1222].sum() / 11e6),
            "max_abs_rel_annual_diff": float(max(
                abs(ann[str(y)]["chowlin_GWh"] / ann[str(y)]["recorded_GWh"] - 1)
                for y in range(2012, 2023) if ann[str(y)]["recorded_GWh"] > 0))}
        ir, ic = vr[irr][:, s1222], vc[irr][:, s1222]
        moved[c] = {"irrigation_cell_level_2012_2022": float(np.abs(ic - ir).sum() / ir.sum()),
                    "irrigation_fan_total_level_2012_2022": float(
                        np.abs(ic.sum(0) - ir.sum(0)).sum() / ir.sum())}
        # seasonal amplitude: CV of the calendar-month means of the in-grid irrigation total
        m = np.array([int(s[5:7]) for s in dates])[s1222]
        cv = {}
        for nm, x in (("recorded", ir.sum(0)), ("chowlin", ic.sum(0))):
            clim = np.array([x[m == mm].mean() for mm in range(1, 13)])
            cv[nm] = {"cv_calendar_means": float(clim.std() / clim.mean()),
                      "peak_month": int(clim.argmax() + 1),
                      "trough_month": int(clim.argmin() + 1),
                      "peak_to_trough": float(clim.max() / clim.min())}
        season[c] = cv
        fl = cl["flag_by_county"][k][irr]
        flags[c] = {str(f): int((fl == f).sum()) for f in np.unique(fl)}
        sel = (fl == 3) & (vc[irr] > 0)
        rs = cl["sd_by_county"][k][irr][sel] / vc[irr][sel]
        relsd[c] = {"n_cell_months": int(sel.sum()),
                    "median_rel_sd": float(np.median(rs)) if sel.any() else None,
                    "q90_rel_sd": float(np.quantile(rs, 0.9)) if sel.any() else None}
    out.update({"annual_in_grid": annual, "energy_moved": moved,
                "irrigation_seasonality_2012_2022": season,
                "flags_irrigation_chowlin_by_county": flags,
                "flag3_irrigation_cell_rel_sd": relsd})
    fa = cl["flag_total_all"]
    out["flags_total_all_chowlin"] = {str(f): int((fa == f).sum()) for f in np.unique(fa)}
    fr = rec["flag_total_all"]
    out["flags_total_all_recorded"] = {str(f): int((fr == f).sum()) for f in np.unique(fr)}
    if variant == "preorigin":
        t = L.tidx("2022-12")
        out["dec2022"] = {
            "in_grid_total_kWh_recorded": float(np.nansum(rec["value_total_all"][:, t])),
            "in_grid_total_kWh_chowlin": float(np.nansum(cl["value_total_all"][:, t])),
            "cells_flag4_recorded": int((rec["flag_total_all"][:, t] == 4).sum()),
            "cells_flag4_chowlin": int((cl["flag_total_all"][:, t] == 4).sum())}
    else:
        t = L.tidx("2026-07")
        out["jul2026"] = {
            "cells_by_flag_chowlin_total_all": {
                str(f): int((cl["flag_total_all"][:, t] == f).sum())
                for f in np.unique(cl["flag_total_all"][:, t])},
            "in_grid_total_kWh_chowlin": float(np.nansum(cl["value_total_all"][:, t])),
            "in_grid_total_kWh_recorded": float(np.nansum(rec["value_total_all"][:, t]))}
    meta = {k: (rec[k].item() if rec[k].ndim == 0 else None)
            for k in ("nx", "ny", "dx", "x0", "y0", "n_active", "mask_hash", "grid_hash")}
    out["grid"] = {k: (v if not isinstance(v, np.generic) else v.item()) for k, v in meta.items()}
    return out


def ccm_summary(variant: str) -> dict:
    d = pd.read_parquet(PROD[variant] / "county_class_monthly.parquet")
    d["year"] = d["date"].str[:4].astype(int)
    out = {}
    for (c, sc, arm), g in d[d["class"].isin(["all", "all_nodomestic", "irrigation",
                                               "domestic"])].groupby(["county", "scope", "arm"]):
        g2 = g[(g.year >= 2012) & (g.year <= 2022)]
        out[f"{c}|{sc}|{arm}"] = {
            cls: float(g2.loc[g2["class"] == cls, "value"].sum() / 11e6)
            for cls in ("all", "all_nodomestic", "irrigation", "domestic")}
    return out


def meter_phase_summary(variant: str) -> dict:
    d = pd.read_parquet(PROD[variant] / "meter_phase.parquet")
    out = {}
    for c, g in d.groupby("county"):
        out[c] = {
            "meters": int(len(g)),
            "code_x_phase": {f"{a}|{b}": int(n) for (a, b), n in
                             g.groupby(["billing_code", "phase_detected"]).size().items()},
            "chowlin_phase": {str(k): int(n) for k, n in g["chowlin_phase"].value_counts().items()},
            "rho": {str(k): int(n) for k, n in g["rho"].round(2).value_counts().items()},
            # chowlin_shipped is True only for meters of the Chow-Lin set (chowlin_phase A/B)
            "chowlin_set_meters": int((g["chowlin_phase"] != "").sum()),
            "chowlin_shipped_all_set_meters": bool(
                g.loc[g["chowlin_phase"] != "", "chowlin_shipped"].all()),
            "chowlin_shipped_outside_set": int(
                g.loc[g["chowlin_phase"] == "", "chowlin_shipped"].sum()),
            "pairs_resplit": int(g["n_pairs_resplit"].sum()),
            "dec_jan_paired": int(g["dec_jan_paired"].sum())}
    return out


def zero_summary(variant: str) -> dict:
    z = pd.read_parquet(PROD[variant] / "zero_class.parquet",
                        columns=["county", "zclass", "sync_suspect", "flag", "month"])
    out = {}
    for c, g in z.groupby("county"):
        out[c] = {"by_class": {k: int(n) for k, n in g["zclass"].value_counts().items()},
                  "sync_suspect_rows": int(g["sync_suspect"].sum()),
                  "flag9_rows": int((g["flag"] == 9).sum()),
                  "sync_months": sorted(g.loc[g["sync_suspect"], "month"].unique().tolist())}
    s = pd.read_parquet(PROD[variant] / "meter_zero_summary.parquet")
    for c, g in s.groupby("county"):
        out[c]["terminated_meters"] = int(g["terminated"].sum())
        out[c]["terminated_last12_GWh"] = float(
            g.loc[g["terminated"], "last12_kwh_before_termination"].sum() / 1e6)
    return out


def test_digest() -> dict:
    def ld(n):
        with open(TESTS / n, encoding="utf-8") as f:
            return json.load(f)
    dig = {}
    for v in ("preorigin", "full"):
        t1 = ld(f"t1_meter_validation_{v}.json")
        dig[f"t1_{v}_keys"] = sorted(t1.keys())[:30]
    r1 = ld("t2_head_check_round1.json")
    r2 = ld("t2_head_check_round2.json")
    dig["head_round1_summary"] = {k: v.get("primary_D_w") for k, v in r1["summary"].items()}
    dig["head_round1_fixed_nuisance"] = {k: v.get("primary_D_w")
                                         for k, v in r1["summary_fixed_nuisance"].items()}
    dig["head_round2_summary"] = {k: {kk: vv for kk, vv in v.items()
                                      if kk in ("primary_D_w", "D_1km", "D_2km", "D_3km",
                                                "D_5km", "cl_minus_rec_1km",
                                                "share_cl_beats_rec_1km",
                                                "primary_pass_p_lt_0.05")}
                                  for k, v in r2["summary"].items()}
    if "summary_fixed_nuisance" in r2:
        dig["head_round2_fixed_nuisance"] = {k: v.get("primary_D_w")
                                             for k, v in r2["summary_fixed_nuisance"].items()}
    return dig


def facts() -> None:
    import looks_p2 as LK

    LK.record(
        "P2-facts",
        {"P2r2:reassembled_preorigin": "published yunlin_electricity/preorigin/v1",
         "P2r2:reassembled_full": "published yunlin_electricity/full/v1"},
        "c0_pump_electricity_v3:2007-01..2026-07", "report_facts",
        "r5_resume.py facts: array inventory, per-pump window totals vs C0, recorded arm vs "
        "C0 cells, annual in-grid energy per arm, energy moved, irrigation seasonality, flags, "
        "flag-3 relative sd, zero and phase summaries of the published products; descriptive "
        "after-the-fact facts for the report, no test, gate or choice depends on them.")
    t0 = time.time()
    out = {"generated_by": "dataprep_staging/p2_census_repair/v2/r5_resume.py facts",
           "products": {v: L.rel(PROD[v]) if hasattr(L, "rel") else str(PROD[v])
                        for v in PROD}}
    for v in ("preorigin", "full"):
        print(v, "inventory", flush=True)
        out[f"inventory_{v}"] = inventory(v)
        print(v, "pump totals", flush=True)
        out[f"pump_window_totals_{v}"] = pump_window_totals(v)
        print(v, "cells", flush=True)
        out[f"cells_{v}"] = cell_checks(v)
        out[f"county_class_monthly_{v}"] = ccm_summary(v)
        out[f"meter_phase_{v}"] = meter_phase_summary(v)
        out[f"zeros_{v}"] = zero_summary(v)
    out["tests"] = test_digest()
    out["seconds"] = round(time.time() - t0, 1)
    L.save_json(out, OUT)
    print("wrote", OUT, out["seconds"], "s")


if __name__ == "__main__":
    cmd = sys.argv[1:]
    if cmd == ["log_review"]:
        log_review()
    elif cmd == ["facts"]:
        facts()
    else:
        raise SystemExit(__doc__)
