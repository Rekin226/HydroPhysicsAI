"""Stage surface_water_v4/preorigin/v2 and full/v4 (W1 repair round 2).

    cpu_slot.sh python -I dataprep_staging/w1_canal/rep2_build.py STAGE_DIR

Reads the published preorigin/v1 and full/v3 (hashes checked against their manifests), the
L0 yearbooks 2012-2022 (pre-origin parse QC) and the v3 allocation (grid sd), and writes
STAGE_DIR/preorigin/ and STAGE_DIR/full/ plus STAGE_DIR/rep2_state.json. rep2_publish.py
publishes them as new versions. Values, flags and members are unchanged; see rep2_config.py
for what changes and why. Nothing is fitted.
"""
from __future__ import annotations

import itertools
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE.parent, HERE.parents[1]):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import manifest  # noqa: E402
from common.flags import validate_grid, validate_long_table  # noqa: E402
from common.paths import DATAPREP  # noqa: E402
from harness import prereg2  # noqa: E402
from scipy import stats  # noqa: E402

from w1_canal import config as C  # noqa: E402
from w1_canal import grid  # noqa: E402
from w1_canal import rep2_config as R2  # noqa: E402
from w1_canal import rep_uncert as U  # noqa: E402
from w1_canal.build import jsonable  # noqa: E402
from w1_canal.parse_yearbook import parse_qc, parse_years  # noqa: E402

BASE = DATAPREP / C.PRODUCT_BASE
PRE_V1 = BASE / "preorigin" / "v1"
FULL_V3 = BASE / "full" / "v3"
DECL_ID = "W1.ext2026_uncertainty.r1"
NAME = C.DISTRICT_NAME
CODE_OF = {v: k for k, v in NAME.items()}
DRAW_KEY = {"chc": "draws_changhua", "yl": "draws_yunlin"}

PRE_COPY = ["janmay_validation.csv", "statespace_loo_coverage.csv", "printed_monthly.csv",
            "printed_annual.csv"]
FULL_COPY = ["janmay_validation.csv", "statespace_loo_coverage.csv", "printed_monthly.csv",
             "printed_annual.csv", "station_rain_monthly.csv", "ext2026_loo_annual.csv",
             "ext2026_loo_monthly.csv", "rep_coverage_annual.csv", "rep_coverage_janmay.csv",
             "rep_coverage_inflow.csv", "rep_double_loo_monthly.csv"]
CHECK_TABLES = ["janmay_validation.csv", "statespace_loo_coverage.csv",
                "ext2026_loo_annual.csv", "ext2026_loo_monthly.csv", "rep_coverage_annual.csv",
                "rep_coverage_janmay.csv", "rep_coverage_inflow.csv",
                "rep_double_loo_monthly.csv"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def verified(vdir: Path) -> dict:
    """Manifest of a published version after re-hashing its outputs (refuse drift)."""
    m, d = manifest.load_manifest(vdir)
    probs = manifest.verify_hashes(m, d, ("outputs",))
    if probs:
        raise RuntimeError(f"{vdir}: {probs}")
    return m


# ----------------------------------------------------------------------------- labels
def hist_label(cov: pd.DataFrame) -> tuple[str, dict]:
    g = U.gate(cov)
    a = g["a1_gate2"]
    misses = ", ".join(f"{m['year']} {NAME[m['district']]} {m['side']}" for m in g["misses"])
    sides = {m["side"] for m in g["misses"]}
    if sides == {"over-predicted"}:
        misses += " (all over-predictions)"
    text = R2.LABEL_HIST.format(hits=g["hits"], n=g["n"], cov=g["coverage"], lo=a["lo"],
                                hi=a["hi"], rel="contains" if a["contains_target"] else
                                "excludes", neff=g["n_years"], misses=misses or "none")
    if not a["pass"]:
        text = "uncalibrated (Jan-May coverage rule failed); " + text
    return text, g


def composite_label(comp: pd.DataFrame, fit_years) -> tuple[pd.DataFrame, dict]:
    pos = comp.value > 0
    rel = comp.sd[pos] / comp.value[pos]
    r1 = rel[comp.flag[pos].isin([1, 4])]
    r3 = rel[comp.flag[pos] == 3]
    base = float(r1.median())
    chk = {"rel_sd_flag_1_4": base,
           "rel_sd_flag_1_4_max_dev": float((r1 - base).abs().max()),
           "rel_sd_flag_3_over_base_max_dev": float((r3 / base - 2).abs().max())
           if len(r3) else None, "n_rows": int(len(comp)), "n_sd": int(comp.sd.notna().sum())}
    out = comp.copy()
    out["sd_basis"] = out.flag.map(R2.SD_BASIS_COMPOSITE).fillna(R2.SD_BASIS_COMPOSITE[3])
    out["interval_calibration"] = R2.LABEL_COMPOSITE.format(rel=base, fy0=min(fit_years),
                                                            fy1=max(fit_years))
    return out, chk


def member_stats(mon: pd.DataFrame, draws: dict) -> pd.DataFrame:
    dates = pd.to_datetime(draws["dates"]).strftime("%Y-%m-%d")
    mm = []
    for d in C.DISTRICTS:
        X = np.asarray(draws[DRAW_KEY[d]], float)
        mm.append(pd.DataFrame({"date": dates, "district": NAME[d], "member_mean": X.mean(0),
                                "member_sd": X.std(0)}))
    mm = pd.concat(mm)
    out = mon.merge(mm, on=["date", "district"], how="left", validate="one_to_one")
    if out.member_mean.isna().any():
        raise RuntimeError("rows without members")
    out["value_minus_member_mean"] = out.value - out.member_mean
    return out


def per_month_labels(mon: pd.DataFrame, dates) -> tuple[np.ndarray, np.ndarray]:
    lab, bas = [], []
    for t in pd.to_datetime(dates).strftime("%Y-%m-%d"):
        r = mon[mon.date == t]
        ls = list(dict.fromkeys(r.interval_calibration))
        bs = list(dict.fromkeys(r.sd_basis))
        lab.append(ls[0] if len(ls) == 1 else "; ".join(
            f"{a}: {b}" for a, b in zip(r.district, r.interval_calibration, strict=True)))
        bas.append(bs[0] if len(bs) == 1 else "; ".join(
            f"{a}: {b}" for a, b in zip(r.district, r.sd_basis, strict=True)))
    return np.array(lab), np.array(bas)


# ----------------------------------------------------------------------------- diagnostics
def runs_pvalue(signs: np.ndarray) -> dict:
    """Exact one-sided Wald-Wolfowitz p (too few runs) by enumerating every arrangement of the
    observed numbers of + and - signs."""
    s = np.asarray(signs, bool)
    n, k = len(s), int(s.sum())
    runs = 1 + int((s[1:] != s[:-1]).sum())
    le, tot = 0, 0
    for pos in itertools.combinations(range(n), k):
        a = np.zeros(n, bool)
        a[list(pos)] = True
        r = 1 + int((a[1:] != a[:-1]).sum())
        tot += 1
        le += r <= runs
    return {"runs": runs, "n_pos": k, "n_neg": n - k, "p_one_sided_few_runs": le / tot}


def residual_diagnostics(loo: pd.DataFrame) -> dict:
    out = {}
    for d in C.DISTRICTS:
        t = loo[loo.district == d].sort_values("year")
        yrs = t.year.to_numpy(int)
        r = (t.obs - t.rain_chain).to_numpy(float)
        rjs = (t.denton_janSep - t.rain_chain_janSep).to_numpy(float)
        sg = r > 0
        # longest run of one sign and where it ends
        best, cur, end = 1, 1, 0
        for i in range(1, len(sg)):
            cur = cur + 1 if sg[i] == sg[i - 1] else 1
            if cur > best:
                best, end = cur, i
        rec = np.isin(yrs, R2.DIAG_RECENT_YEARS)
        rho, p_rho = stats.spearmanr(yrs, r)
        lag1 = float(np.corrcoef(r[:-1], r[1:])[0, 1])
        out[NAME[d]] = {
            "years": yrs.tolist(), "residual_obs_minus_chain": r.round(3).tolist(),
            "mean_all": float(r.mean()), "mean_recent_2020_2025": float(r[rec].mean()),
            "mean_rest": float(r[~rec].mean()),
            "recent_signs": ["+" if x else "-" for x in sg[rec]],
            "longest_same_sign_run": {"length": best,
                                      "years": yrs[end - best + 1:end + 1].tolist(),
                                      "sign": "+" if sg[end] else "-"},
            "p_iid_symmetric_trailing_run_ge_6": float(2 * 0.5 ** 6),
            "runs_test": runs_pvalue(sg),
            "spearman_residual_vs_year": {"rho": float(rho), "p_two_sided": float(p_rho)},
            "lag1_autocorrelation_over_evaluation_years": lag1,
            "janSep_residual_vs_flag5_denton_mean_recent": float(rjs[rec].mean()),
            "note": "post hoc (after the verifier's report); design-time units; residual = "
                    "observed annual district total - single-fold chain (ext2026_loo_annual."
                    "csv); the Jan-Sep residual is against the flag-5 Denton series (a model, "
                    "not an observation: U1, not a score)"}
    return out


# ----------------------------------------------------------------------------- variants
def build_preorigin(stage: Path, state: dict):
    log("preorigin: verify v1 and re-parse the 2012-2022 yearbooks from L0")
    m1 = verified(PRE_V1)
    out = stage / "preorigin"
    out.mkdir(parents=True, exist_ok=False)
    mon = pd.read_csv(PRE_V1 / "district_monthly.csv")
    draws = dict(np.load(PRE_V1 / "district_draws.npz", allow_pickle=True))
    cov = pd.read_csv(PRE_V1 / "statespace_loo_coverage.csv")
    label_hist, g_hist = hist_label(cov)
    # structural zeros: sd 0 (the members are 0), winter-0.5 alternative to its own column
    sz = mon.structural_zero.astype(bool)
    for d in C.DISTRICTS:
        X = np.asarray(draws[DRAW_KEY[d]], float)
        dts = pd.to_datetime(draws["dates"]).strftime("%Y-%m-%d")
        zd = set(mon.date[sz & (mon.district == NAME[d])])
        nz = int((X[:, np.isin(dts, list(zd))] != 0).sum())
        if nz:
            raise RuntimeError(f"{d}: {nz} non-zero members in structural-zero months")
        if (mon.value[sz] != 0).any():
            raise RuntimeError("structural-zero rows with a non-zero value")
    mon["structural_alt_value"] = np.where(sz, mon.sd, np.nan)
    mon["sd"] = np.where(sz, 0.0, mon.sd)
    mon["sd_basis"] = np.where(sz, "structural_zero_members_0",
                               "sd_of_preorigin_v1_state_space_draws")
    mon["interval_calibration"] = np.where(sz, U.RC.LABEL_STRUCTURAL, label_hist)
    mon = member_stats(mon, draws)
    sd_vs_members = float((mon.sd - mon.member_sd).abs()[~sz].max())
    mon = mon.drop(columns="member_sd")
    mon.to_csv(out / "district_monthly.csv", index=False)
    # percolation basis: same sd and labels as the district rows
    pb = pd.read_csv(PRE_V1 / "percolation_basis.csv")
    key = mon.set_index(["date", "district"])
    j = key.loc[list(zip(pb.date, pb.district, strict=True))]
    if not np.array_equal(j.value.to_numpy(), pb.value.to_numpy()):
        raise RuntimeError("percolation basis values differ from the district table")
    pb["sd"] = j.sd.to_numpy()
    pb["sd_basis"] = j.sd_basis.to_numpy()
    pb["interval_calibration"] = j.interval_calibration.to_numpy()
    pb.to_csv(out / "percolation_basis.csv", index=False)
    # composite and gauges
    comp, comp_chk = composite_label(pd.read_parquet(PRE_V1 / "inflow_composite_daily.parquet"),
                                     C.VARIANTS["preorigin"]["years"])
    comp.to_parquet(out / "inflow_composite_daily.parquet", index=False)
    gl = pd.read_parquet(PRE_V1 / "gauges_daily_L1.parquet")
    if gl.sd.notna().any():
        raise RuntimeError("gauges_daily_L1 has an sd")
    gl["interval_calibration"] = R2.LABEL_GAUGES
    gl.to_parquet(out / "gauges_daily_L1.parquet", index=False)
    # draws: unchanged arrays, labels per month
    lab, bas = per_month_labels(mon, draws["dates"])
    draws_out = dict(draws)
    draws_out.update(interval_calibration=lab, sd_basis=bas, note=np.array(
        str(draws["note"]) + "; repair round 2: interval_calibration per month (plan 4.7); "
        "the members are centred on member_mean (district_monthly.csv), not on the Denton "
        "value"))
    np.savez_compressed(out / "district_draws.npz", **draws_out)
    # grid: sd from the repaired table, structural alternative as its own field, labels
    g1 = np.load(PRE_V1 / "grid_sw_v4.npz", allow_pickle=True)
    gdates = pd.to_datetime(g1["dates"])
    alloc = grid.v3_allocation()
    sd_tab, alt_tab = {}, {}
    for d in C.DISTRICTS:
        r = mon[mon.district == NAME[d]].set_index("date").reindex(
            gdates.strftime("%Y-%m-%d"))
        sd_tab[d] = r.sd.to_numpy()
        alt_tab[d] = r.structural_alt_value.fillna(0.0).to_numpy()
    sd_grid = grid.spread_sd(sd_tab, alloc, gdates)
    smon = np.array([mon[(mon.date == t)].structural_zero.all()
                     for t in gdates.strftime("%Y-%m-%d")])
    sd_grid_check = float(np.abs(sd_grid[:, ~smon] - g1["sd"][:, ~smon]).max())
    glab, gbas = per_month_labels(mon, gdates)
    gout = {k: g1[k] for k in g1.files}
    gout.update(sd=sd_grid, value_structural_alt=grid.spread(alt_tab, alloc, gdates),
                interval_calibration=glab, sd_basis=gbas,
                note=np.array(str(g1["note"]) + "; repair round 2 (preorigin/v2): sd from the "
                              "repaired table (structural-zero months 0); value_structural_alt "
                              "= winter-0.5 alternative of the structural-zero months (0 "
                              "elsewhere); interval_calibration and sd_basis per month"))
    gerr = validate_grid(gout["value"], gout["sd"], gout["flag"], gdates)
    np.savez_compressed(out / "grid_sw_v4.npz", **gout)
    for f in PRE_COPY:
        shutil.copy2(PRE_V1 / f, out / f)
    # pre-origin parse QC from L0 (2012-2022 only)
    daily_raw, pmon, pann, pdfs = parse_years(R2.PRE_YEARS, log=log)
    pqc, daily = parse_qc(daily_raw, pmon, pann, C.PRINTED_MONTHLY_TOL)
    gl0 = gl[gl.flag == 0][["date", "station_no", "value"]]
    dd = daily.assign(date=pd.to_datetime(dict(year=daily.year, month=daily.month,
                                               day=daily.day)).dt.strftime("%Y-%m-%d"))
    jj = gl0.merge(dd[["date", "station_no", "q_cms"]], on=["date", "station_no"], how="left")
    parse_identity = {"l1_flag0_station_days": int(len(gl0)),
                      "matched": int(jj.q_cms.notna().sum()),
                      "max_abs_diff": float((jj.value - jj.q_cms).abs().max())}
    pm_v1 = pd.read_csv(PRE_V1 / "printed_monthly.csv")
    printed_identical = bool(pm_v1.reset_index(drop=True).equals(
        pmon.reset_index(drop=True)[pm_v1.columns]))
    v1val = json.loads((PRE_V1 / "validation.json").read_text())
    common = dict(v1val["common_checks"])
    common.pop("parse_vs_pilot", None)
    common["parse_qc"] = pqc
    common["parse_qc_note"] = ("repair round 2: recomputed from the 2012-2022 yearbooks only "
                               "(preorigin/v1 reported 2012-2025, whose 2023-2025 PDFs were not "
                               "inputs of the pre-origin product); the comparison with the "
                               "scout's 2012-2025 parse is reported in full/v4")
    common["yearbook_pdfs"] = [str(Path(p).relative_to(C.REPO)) for p in pdfs]
    common["parse_identity_vs_preorigin_v1_gauges_L1"] = parse_identity
    common["printed_monthly_identical_to_v1"] = printed_identical
    comp_label = comp.interval_calibration.iloc[0]
    rep2 = {
        "round": R2.ROUND_TAG, "label_hist": label_hist, "coverage_hist_statespace_janmay": g_hist,
        "label_structural": U.RC.LABEL_STRUCTURAL, "label_composite": comp_label,
        "composite_sd_check": comp_chk, "grid_sd_unchanged_months_max_abs": sd_grid_check,
        "grid_validation_errors": gerr, "sd_vs_member_sd_max_abs_nonstructural": sd_vs_members,
        "value_minus_member_mean": {NAME[d]: {
            "max_abs": float(mon[mon.district == NAME[d]].value_minus_member_mean.abs().max()),
            "rms": float(np.sqrt((mon[mon.district == NAME[d]].value_minus_member_mean ** 2)
                                 .mean()))} for d in C.DISTRICTS},
        "structural_zero_rows": int(sz.sum()),
        "removed_uses": list(R2.PRE_REMOVED_USES)}
    val = {"product": f"{C.PRODUCT_BASE}/preorigin", "version": None,
           "variant_validation": v1val["variant_validation"], "common_checks": common,
           R2.ROUND: rep2}
    state["preorigin"] = {"manifest_v1": m1["version"], "rep2": rep2,
                          "parse_qc_2012_2022": {k: pqc[k] for k in (
                              "station_years", "printed_monthly_compared",
                              "printed_monthly_off_gt_tol", "printed_annual_compared")},
                          "parse_identity": parse_identity,
                          "printed_identical": printed_identical,
                          "pdfs": common["yearbook_pdfs"]}
    with open(out / "validation_stage.json", "w") as f:
        json.dump(jsonable(val), f, indent=1, ensure_ascii=False)
    state["preorigin"]["labels"] = labels_inventory("preorigin", mon, comp, None, None)
    log(f"preorigin: hist label {g_hist['hits']}/{g_hist['n']}; grid sd check "
        f"{sd_grid_check:.2e}; parse identity {parse_identity}")


def build_full(stage: Path, state: dict):
    log("full: verify v3")
    m3 = verified(FULL_V3)
    out = stage / "full"
    out.mkdir(parents=True, exist_ok=False)
    v3val = json.loads((FULL_V3 / "validation.json").read_text())
    rep1 = v3val["repair_r1"]
    mon = pd.read_csv(FULL_V3 / "district_monthly.csv")
    draws = dict(np.load(FULL_V3 / "district_draws.npz", allow_pickle=True))
    cov = pd.read_csv(FULL_V3 / "statespace_loo_coverage.csv")
    label_hist, g_hist = hist_label(cov)
    old = rep1["coverage_hist_statespace_janmay"]
    if (g_hist["hits"], g_hist["n"]) != (old["hits"], old["n"]) or abs(
            g_hist["a1_gate2"]["lo"] - old["a1_gate2"]["lo"]) > 1e-12:
        raise RuntimeError("historical coverage differs from full/v3")
    loo = pd.read_csv(FULL_V3 / "ext2026_loo_annual.csv")
    note = ("W1 repair round 2: post hoc diagnostic of the serial dependence of the rain "
            "chain's annual leave-one-year-out residuals (verifier minor issue); no setting "
            "changed")
    eval_years = sorted(loo.year.unique().tolist())
    prereg2.record_look([f"W1.annual.{d}.{y}" for d in C.DISTRICTS for y in eval_years],
                        "jiji_annual:2012..2025", "W1", "diagnose", prereg_id=DECL_ID,
                        note=note)
    diag = residual_diagnostics(loo)
    ga = rep1["coverage_annual"]
    serial = diag["yunlin"]["recent_signs"].count("-") == len(R2.DIAG_RECENT_YEARS)
    label26 = R2.LABEL_2026.format(n_years=ga["n_years"], hits=ga["hits"], n=ga["n"],
                                   lo=ga["a1_gate2"]["lo"], hi=ga["a1_gate2"]["hi"],
                                   serial=R2.LABEL_2026_SERIAL if serial else "")
    if not rep1["annual_check_passed"]:
        raise RuntimeError("full/v3 annual check did not pass; label rule differs")
    is26 = pd.to_datetime(mon.date).dt.year >= 2026
    sz = mon.structural_zero.astype(bool)
    mon["interval_calibration"] = np.where(sz, U.RC.LABEL_STRUCTURAL,
                                           np.where(is26, label26, label_hist))
    mon = member_stats(mon, draws)
    sd_vs_members = float((mon.sd - mon.member_sd).abs()[~sz & ~is26].max())
    sd_vs_members26 = float((mon.sd - mon.member_sd).abs()[is26].max())
    mon = mon.drop(columns="member_sd")
    mon.to_csv(out / "district_monthly.csv", index=False)
    pb = pd.read_csv(FULL_V3 / "percolation_basis.csv")
    j = mon.set_index(["date", "district"]).loc[list(zip(pb.date, pb.district, strict=True))]
    if not (np.array_equal(j.value.to_numpy(), pb.value.to_numpy())
            and np.array_equal(j.sd.to_numpy(), pb.sd.to_numpy())):
        raise RuntimeError("percolation basis differs from the district table")
    pb["sd_basis"] = j.sd_basis.to_numpy()
    pb["interval_calibration"] = j.interval_calibration.to_numpy()
    pb.to_csv(out / "percolation_basis.csv", index=False)
    comp, comp_chk = composite_label(pd.read_parquet(FULL_V3 / "inflow_composite_daily.parquet"),
                                     C.VARIANTS["full"]["years"])
    comp.to_parquet(out / "inflow_composite_daily.parquet", index=False)
    gl = pd.read_parquet(FULL_V3 / "gauges_daily_L1.parquet")
    if gl.sd.notna().any():
        raise RuntimeError("gauges_daily_L1 has an sd")
    gl["interval_calibration"] = R2.LABEL_GAUGES
    gl.to_parquet(out / "gauges_daily_L1.parquet", index=False)
    ri = pd.read_csv(FULL_V3 / "basin_rain_index_monthly.csv")
    ri["sd_basis"] = np.where(ri.sd.notna(), R2.SD_BASIS_RAIN, "none")
    ri["interval_calibration"] = np.where(ri.sd.notna(), R2.LABEL_RAIN, R2.LABEL_RAIN_NONE)
    if (ri.sd.isna() & ri.value.notna()).any():
        raise RuntimeError("rain index value without sd")
    ri.to_csv(out / "basin_rain_index_monthly.csv", index=False)
    ip = pd.read_csv(FULL_V3 / "inflow_predicted_monthly.csv")
    ip["interval_calibration"] = ip.interval_calibration + R2.LABEL_INFLOW_SUFFIX
    ip.to_csv(out / "inflow_predicted_monthly.csv", index=False)
    lab, bas = per_month_labels(mon, draws["dates"])
    draws_out = dict(draws)
    draws_out.update(interval_calibration=lab, sd_basis=bas, note=np.array(
        str(draws["note"]) + "; repair round 2: interval_calibration per month (plan 4.7); "
        "2012-2025 members are centred on member_mean (district_monthly.csv), not on the "
        "Denton value; 2026 members are centred on the value before truncation at 0"))
    np.savez_compressed(out / "district_draws.npz", **draws_out)
    g3 = np.load(FULL_V3 / "grid_sw_v4.npz", allow_pickle=True)
    glab, gbas = per_month_labels(mon, g3["dates"])
    gout = {k: g3[k] for k in g3.files}
    gout.update(interval_calibration=glab, sd_basis=gbas,
                note=np.array(str(g3["note"]) + "; repair round 2 (full/v4): "
                              "interval_calibration and sd_basis per month"))
    np.savez_compressed(out / "grid_sw_v4.npz", **gout)
    for f in FULL_COPY:
        shutil.copy2(FULL_V3 / f, out / f)
    rep2 = {
        "round": R2.ROUND_TAG, "label_2026": label26, "label_hist": label_hist,
        "coverage_hist_statespace_janmay": g_hist, "label_structural": U.RC.LABEL_STRUCTURAL,
        "label_composite": comp.interval_calibration.iloc[0], "label_rain": R2.LABEL_RAIN,
        "composite_sd_check": comp_chk,
        "residual_diagnostics_post_hoc": diag,
        "sd_vs_member_sd_max_abs": {"2012_2025_nonstructural": sd_vs_members,
                                    "2026": sd_vs_members26},
        "value_minus_member_mean": {NAME[d]: {
            "2012_2025_max_abs": float(mon[(mon.district == NAME[d]) & ~is26]
                                       .value_minus_member_mean.abs().max()),
            "2012_2025_rms": float(np.sqrt((mon[(mon.district == NAME[d]) & ~is26]
                                            .value_minus_member_mean ** 2).mean())),
            "2026_jan_sep_sum": float(mon[(mon.district == NAME[d]) & is26]
                                      .value_minus_member_mean.sum())} for d in C.DISTRICTS}}
    val = dict(v3val)
    val.update({"product": f"{C.PRODUCT_BASE}/full", "version": None, R2.ROUND: rep2})
    with open(out / "validation_stage.json", "w") as f:
        json.dump(jsonable(val), f, indent=1, ensure_ascii=False)
    state["full"] = {"manifest_v3": m3["version"], "rep2": rep2}
    state["full"]["labels"] = labels_inventory("full", mon, comp, ri, ip)
    log(f"full: label26 serial clause {serial}; Yunlin runs "
        f"{diag['yunlin']['runs_test']}; recent mean {diag['yunlin']['mean_recent_2020_2025']:.3f}")


# ----------------------------------------------------------------------------- inventory
def labels_inventory(variant: str, mon, comp, ri, ip) -> dict:
    """Every interval-bearing column or array of every output with its label (plan 4.7)."""
    row = "row-level column interval_calibration (values: see validation.json)"
    inv = {
        "district_monthly.csv": {
            "sd": row, "ss_p05/ss_p50/ss_p95": row,
            "structural_alt_value": "structural alternative (winter weight 0.5), not an "
                                    "interval",
            "member_mean": "mean of the members in district_draws.npz (not an interval)"},
        "percolation_basis.csv": {"sd": row},
        "district_draws.npz": {"draws_changhua/draws_yunlin": "per-month array "
                               "interval_calibration (same labels as district_monthly.csv)"},
        "grid_sw_v4.npz": {"sd": "per-month array interval_calibration (same labels as "
                                 "district_monthly.csv)",
                           "value_structural_alt": "structural alternative, not an interval"},
        "inflow_composite_daily.parquet": {"sd": comp.interval_calibration.iloc[0]},
        "gauges_daily_L1.parquet": {"sd": R2.LABEL_GAUGES},
        "printed_monthly.csv": {"-": R2.LABEL_NO_INTERVAL},
        "printed_annual.csv": {"-": R2.LABEL_NO_INTERVAL},
    }
    for f in CHECK_TABLES:
        if variant == "preorigin" and f not in PRE_COPY:
            continue
        inv[f] = {"p05/p95/lo/hi (where present)": R2.LABEL_CHECK_TABLE}
    if variant == "full":
        inv["basin_rain_index_monthly.csv"] = {"sd": R2.LABEL_RAIN}
        inv["inflow_predicted_monthly.csv"] = {"sd": str(ip.interval_calibration.iloc[0])}
        inv["station_rain_monthly.csv"] = {"-": R2.LABEL_NO_INTERVAL}
    labs = sorted(set(mon.interval_calibration))
    return {"outputs": inv, "district_labels": labs}


def main(stage: str):
    stage = Path(stage)
    stage.mkdir(parents=True, exist_ok=True)
    state = {"round": R2.ROUND_TAG, "created_utc": pd.Timestamp.utcnow().isoformat()}
    build_full(stage, state)
    build_preorigin(stage, state)
    for v in ("preorigin", "full"):
        for f in ("district_monthly.csv", "percolation_basis.csv",
                  "inflow_composite_daily.parquet", "gauges_daily_L1.parquet",
                  "basin_rain_index_monthly.csv", "inflow_predicted_monthly.csv"):
            p = stage / v / f
            if not p.exists():
                continue
            t = pd.read_csv(p) if f.endswith(".csv") else pd.read_parquet(p)
            if f == "percolation_basis.csv":
                continue                    # flag/method/source_ids are present already
            errs = validate_long_table(t, require_all_columns=False)
            if errs:
                raise ValueError(f"{v}/{f}: {errs}")
    with open(stage / "rep2_state.json", "w") as f:
        json.dump(jsonable(state), f, indent=1, ensure_ascii=False)
    log(f"staged in {stage}")


if __name__ == "__main__":
    main(sys.argv[1])
