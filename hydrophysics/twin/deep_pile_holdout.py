"""Score the frozen projection against the deep benchmarks printed in the WRA monthly reports.

The ring and GNSS continuation (`twin.ring_holdout`) ends in March 2025 because every
instrument feed the project can reach ends there. The WRA's monthly 地層下陷監測整合分析月報
prints, for six deep benchmarks (深層樁), the compression of each month in centimetres;
`twin.report_extract` transcribes those numbers and verifies them. Four of the six piles
stand on the model grid (溪州國小 300 m, 土庫國中 400 m, 秀潭國小 70 m, 中科虎尾 300 m).
This module sums the verified monthly values into each pile's cumulative compression
from the end of 2023, and scores the forward archive's column (ensemble mean, member
spread from the year-end members file) against them over January 2024 to December 2025,
and separately over June 2025 to January 2026, the months after every feed stopped.

Deep piles were never used in any calibration or model choice. They are not a clean
test of the column for two reasons the report carries verbatim: three of the four stand
at the same schools as compaction rings that were a Stage-2 target through 2022, and a
pile measures the interval between its anchor depth and the surface while the column is
total depth (秀潭 at 70 m sees only the top of the system). Baselines: the pile's own
printed trailing-twelve-month compression in the January 2024 report continued, and no
displacement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .mechanism import ring_cells
from .release import atomic_json, sha256
from .ring_holdout import r2_rmse

CAVEATS = [
    "(i) four piles only; a pooled R² over four sites is reported but fragile",
    "(ii) 溪州國小, 土庫國中 and 秀潭國小 host compaction rings that were the Stage-2 target "
    "through 2022; the deep piles themselves were never used in calibration or selection",
    "(iii) a pile measures anchor-to-surface compression (70-400 m); the column is total depth",
    "(iv) monthly values are printed to 0.1 cm; a 24-month sum carries up to ±1.2 cm of "
    "rounding in the worst case, typically much less (checked against the printed "
    "trailing-twelve-month values)",
    "(v) the model runs its calibration climatology forcing from the 2022 origin; recorded "
    "2024-26 pumping is not applied",
]


def reconstruct_cumulative(piles: pd.DataFrame, ref: str = "2023-12", end: str = "2026-01",
                           w_year: float = 0.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cumulative compression (cm) per pile from ``ref`` (0 there) to ``end`` by weighted
    least squares on the printed values: every verified monthly value gives
    c(m) - c(m-1) = M(m); every verified trailing-twelve-month value whose pile was printed
    at the same depth twelve months earlier gives c(m) - c(m-12) = Y(m) (weight ``w_year``,
    the reports' own arithmetic). A month is determined only if its node is connected to
    ``ref`` through constraints; others stay NaN. Returns (month x pile table, constraint
    residual table: pile, kind, month, printed, fitted, resid). Plain arithmetic on printed
    numbers; nothing is interpolated across an unconstrained gap."""
    ok = piles[piles["verified"].astype(bool)].copy()
    ok["per"] = pd.PeriodIndex(ok["month"], freq="M")
    months = pd.period_range(ref, end, freq="M")
    pos = {p: i for i, p in enumerate(months)}
    cum, resid_rows = {}, []
    for pile, g in ok.groupby("pile"):
        g = g.drop_duplicates("per").set_index("per").sort_index()
        depth_at = g["depth_m"].to_dict()
        rows, rhs, wts, meta = [], [], [], []
        for p_, r in g.iterrows():
            if p_ not in pos or (p_ - 1) not in pos:
                continue
            if pd.notna(r["month_cm"]):
                rows.append((pos[p_], pos[p_ - 1]))
                rhs.append(float(r["month_cm"]))
                wts.append(1.0)
                meta.append(("month", str(p_), float(r["month_cm"])))
            if pd.notna(r["year_cm"]) and (p_ - 12) in pos and depth_at.get(p_ - 12) == r["depth_m"]:
                rows.append((pos[p_], pos[p_ - 12]))
                rhs.append(float(r["year_cm"]))
                wts.append(w_year)
                meta.append(("year", str(p_), float(r["year_cm"])))
        n = len(months)
        # connectivity to the reference node
        adj = {i: set() for i in range(n)}
        for a, b in rows:
            adj[a].add(b)
            adj[b].add(a)
        seen, stack = {pos[months[0]]}, [pos[months[0]]]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        A, b, w = np.zeros((len(rows) + 1, n)), np.zeros(len(rows) + 1), np.ones(len(rows) + 1)
        A[0, 0], w[0] = 1.0, 10.0                    # c(ref) = 0
        for k, ((i, j), v, wt) in enumerate(zip(rows, rhs, wts, strict=True), start=1):
            A[k, i], A[k, j], b[k], w[k] = 1.0, -1.0, v, wt
        keep = sorted(seen)
        sol = np.full(n, np.nan)
        if len(rows):
            x, *_ = np.linalg.lstsq(A[:, keep] * w[:, None], b * w, rcond=None)
            sol[keep] = x
        cum[pile] = pd.Series(sol, index=months)
        for (kind, m, printed), (i, j) in zip(meta, rows, strict=True):
            f = sol[i] - sol[j] if (i in seen and j in seen) else float("nan")
            resid_rows.append({"pile": pile, "kind": kind, "month": m, "printed_cm": printed,
                               "fitted_cm": round(float(f), 3), "resid_cm": round(float(f - printed), 3)})
    table = pd.DataFrame(cum)
    return table, pd.DataFrame(resid_rows)


def printed_year_check(piles: pd.DataFrame, cum: pd.DataFrame, month: str = "2024-12") -> dict:
    """The printed trailing-twelve-month value at ``month`` against the sum of the twelve
    printed monthly values (cm): a rounding-level agreement supports both."""
    out = {}
    p = pd.Period(month, freq="M")
    for pile in cum.columns:
        row = piles[(piles.pile == pile) & (piles.month == month)]
        if row.empty or p not in cum.index or (p - 12) not in cum.index:
            continue
        printed = float(row.year_cm.iloc[0])
        summed = float(cum.at[p, pile] - cum.at[p - 12, pile])
        out[pile] = {"printed_cm": printed, "summed_cm": round(summed, 2),
                     "diff_cm": round(summed - printed, 2)}
    return out


def model_at(forward_npz: Path, members_npz: Path, xy: np.ndarray) -> dict:
    """Ensemble-mean cumulative subsidence (m) and year-end member values at the cells
    nearest ``xy`` for the baseline scenario and the published column."""
    with np.load(forward_npz, allow_pickle=False) as z:
        npz = {k: z[k] for k in ("mask", "dx", "x0", "y0")}
        dates = pd.DatetimeIndex(pd.to_datetime(z["dates"])) + pd.offsets.MonthEnd(0)
        mean = np.asarray(z["subs_mean_by_rheology"][0, 0], dtype=float)
        rhe = str(z["rheology_labels"][0])
    cells = ring_cells(npz, xy)
    with np.load(members_npz, allow_pickle=False) as m:
        keep = (m["rheology"] == rhe)
        yr = m["subs_members_yr"][0][keep].astype(float) * float(m["subs_scale_m"])
        years = [int(y) for y in m["years"]]
    return {"cells": cells, "dates": dates, "mean": mean[cells.clip(min=0)],
            "members_yr": yr[:, cells.clip(min=0), :], "years": years, "rheology": rhe}


def score(cum: pd.DataFrame, sites: pd.DataFrame, model: dict, prior_rate: dict[str, float],
          ref: str, t1: str, sub_ref: str, sub_t1: str, deep_min_m: int = 200
          ) -> tuple[pd.DataFrame, dict]:
    ref_p, t1_p = pd.Period(ref, freq="M"), pd.Period(t1, freq="M")
    sref_p, st1_p = pd.Period(sub_ref, freq="M"), pd.Period(sub_t1, freq="M")
    years = (t1_p - ref_p).n / 12.0
    rows = []
    for k, r in sites.reset_index(drop=True).iterrows():
        pile, c = r["pile"], int(model["cells"][k])
        s = cum.get(pile)
        row = {"pile": pile, "depth_m": int(r["depth_m"]), "cell": c,
               "n_months": int(s.notna().sum() - 1) if s is not None else 0}
        if s is None or c < 0:
            rows.append(row)
            continue
        row["obs_last_month"] = str(s.dropna().index.max())
        row["obs_cum_cm"] = float(s.at[t1_p]) if t1_p in s.index and pd.notna(s.at[t1_p]) else np.nan
        if sref_p in s.index and st1_p in s.index and pd.notna(s.at[st1_p]):
            row["obs_sub_cm"] = float(s.at[st1_p] - s.at[sref_p])
        else:
            row["obs_sub_cm"] = np.nan
        d, mean_k = model["dates"], model["mean"][k]

        def at(p, d=d, mean_k=mean_k):
            return float(mean_k[d.get_loc(p.to_timestamp("M"))])

        row["model_cum_cm"] = (at(t1_p) - at(ref_p)) * 100
        row["model_sub_cm"] = (at(st1_p) - at(sref_p)) * 100
        # month-to-month increments, observed vs model, over the determined months
        so = s.dropna()
        if len(so) >= 13:
            inc_o = so.diff().dropna()
            inc_m = pd.Series([at(q) for q in so.index], index=so.index).diff().dropna() * 100
            row["monthly_increment_corr"] = float(np.corrcoef(inc_o, inc_m)[0, 1])
            row["monthly_increment_rmse_cm"] = float(np.sqrt(((inc_o - inc_m) ** 2).mean()))
        yi0, yi1 = model["years"].index(ref_p.year), model["years"].index(t1_p.year)
        mem = (model["members_yr"][:, k, yi1] - model["members_yr"][:, k, yi0]) * 100
        row["model_cum_p10_cm"], row["model_cum_p90_cm"] = [float(v) for v in np.quantile(mem, [0.1, 0.9])]
        row["prior_rate_cm_yr"] = float(prior_rate.get(pile, np.nan))
        row["base_continue_cum_cm"] = row["prior_rate_cm_yr"] * years
        row["base_continue_sub_cm"] = row["prior_rate_cm_yr"] * (st1_p - sref_p).n / 12.0
        row["base_zero_cum_cm"] = 0.0
        rows.append(row)
    tab = pd.DataFrame(rows)
    scored = tab[(tab.cell >= 0) & tab.obs_cum_cm.notna()]
    scores = {"n_sites": int(len(scored)), "window": [str(ref_p + 1), str(t1_p)]}

    def block(sub: pd.DataFrame) -> dict:
        o = sub.obs_cum_cm.to_numpy(float)
        out = {n: r2_rmse(sub[f"{n}_cum_cm"].to_numpy(float), o)
               for n in ("model", "base_continue", "base_zero")}
        inside = (sub.obs_cum_cm >= sub.model_cum_p10_cm) & (sub.obs_cum_cm <= sub.model_cum_p90_cm)
        out["model"]["p10_p90_coverage"] = float(inside.mean()) if len(sub) else np.nan
        out["model"]["n_inside"] = int(inside.sum())
        if "monthly_increment_corr" in sub:
            out["model"]["monthly_increment_corr_median"] = float(sub.monthly_increment_corr.median())
        return out
    scores["cum"] = block(scored)
    # piles anchored at 200 m or deeper see most of the compacting column; the 70 m pile
    # at 秀潭 sees only its top (caveat iii), so the deep subset is reported as well
    deep = scored[scored.depth_m >= deep_min_m]
    scores["cum_deep"] = {"min_depth_m": deep_min_m, "n_sites": int(len(deep)), **block(deep)}
    sub = tab[(tab.cell >= 0) & tab.obs_sub_cm.notna()]
    scores["after_feeds"] = {"window": [str(sref_p + 1), str(st1_p)], "n_sites": int(len(sub))}

    def sub_block(sb: pd.DataFrame) -> dict:
        os_ = sb.obs_sub_cm.to_numpy(float)
        return {"model": r2_rmse(sb.model_sub_cm.to_numpy(float), os_),
                "base_continue": r2_rmse(sb.base_continue_sub_cm.to_numpy(float), os_),
                "base_zero": r2_rmse(np.zeros(len(sb)), os_)}
    if len(sub):
        scores["after_feeds"].update(sub_block(sub))
        sd = sub[sub.depth_m >= deep_min_m]
        scores["after_feeds"]["deep"] = {"n_sites": int(len(sd)), **sub_block(sd)}
    return tab, scores


def run(piles_csv: Path, sites_csv: Path, forward_npz: Path, members_npz: Path, out: Path,
        ref: str = "2023-12", t1: str = "2025-12", sub_ref: str = "2025-05",
        sub_t1: str = "2026-01") -> dict:
    piles = pd.read_csv(piles_csv)
    sites = pd.read_csv(sites_csv)
    # one row per printed pile: the depth printed in the reports selects the sensor
    printed = piles.groupby("pile")["depth_m"].agg(lambda s: int(s.dropna().mode().iloc[0]))
    sites = sites.merge(printed.rename("depth_m"), on=["pile", "depth_m"], how="inner")
    cum, resid = reconstruct_cumulative(piles, ref, sub_t1)
    model = model_at(forward_npz, members_npz, sites[["x_twd97", "y_twd97"]].to_numpy(float))
    first = piles[piles.month == piles.month.min()].set_index("pile")["year_cm"]
    tab, scores = score(cum, sites, model, first.to_dict(), ref, t1, sub_ref, sub_t1)
    out.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out / "sites.csv", index=False)
    resid.to_csv(out / "constraints.csv", index=False)
    series = cum.copy()
    series.index = [str(p) for p in series.index]
    for k, pile in enumerate(sites.pile):
        if model["cells"][k] >= 0:
            d = model["dates"]
            base = model["mean"][k][d.get_loc(pd.Period(ref, freq="M").to_timestamp("M"))]
            series[f"model_{pile}"] = [
                (model["mean"][k][d.get_loc(pd.Period(m, freq="M").to_timestamp("M"))] - base) * 100
                if pd.Period(m, freq="M").to_timestamp("M") in d else np.nan for m in series.index]
    series.to_csv(out / "series.csv", index_label="month")
    report = {
        "schema": 1, "task": "frozen-column-vs-printed-deep-piles",
        "prospective_validation": False, "independent_holdout": False,
        "causal_policy_validation": False,
        "observations": str(piles_csv), "observations_sha256": sha256(piles_csv),
        "n_verified_values": int(piles.verified.astype(bool).sum()), "n_values": int(len(piles)),
        "forward_archive": str(forward_npz), "forward_sha256": sha256(forward_npz),
        "members": str(members_npz), "members_sha256": sha256(members_npz),
        "column": model["rheology"], "reference_month": ref,
        "units": "cm cumulative compression, positive = shortening",
        "baselines": {"base_continue": "the pile's printed trailing-twelve-month compression in "
                                       "the first report continued at that rate",
                      "base_zero": "no displacement"},
        "printed_year_check": printed_year_check(piles, cum),
        "reconstruction": {"method": "weighted least squares on verified printed monthly "
                                     "(weight 1) and trailing-twelve-month (weight 0.5) values; "
                                     "months not connected to the reference stay undetermined",
                           "n_constraints": int(len(resid)),
                           "resid_rms_cm": float(np.sqrt((resid.resid_cm ** 2).mean())) if len(resid) else None,
                           "largest": resid.reindex(resid.resid_cm.abs().sort_values(ascending=False).index)
                                           .head(5).to_dict(orient="records") if len(resid) else []},
        "scores": scores, "independence": CAVEATS,
    }
    atomic_json(out / "report.json", _no_nan(report))
    return report


def _no_nan(x):
    """NaN -> None recursively (the report is written with allow_nan=False)."""
    if isinstance(x, dict):
        return {k: _no_nan(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_no_nan(v) for v in x]
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--piles", required=True, type=Path, help="twin.report_extract deep_piles.csv")
    ap.add_argument("--sites", required=True, type=Path, help="pile, depth_m, x_twd97, y_twd97")
    ap.add_argument("--forward", required=True, type=Path)
    ap.add_argument("--members", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--ref", default="2023-12")
    ap.add_argument("--end", default="2025-12")
    ap.add_argument("--sub-ref", default="2025-05")
    ap.add_argument("--sub-end", default="2026-01")
    args = ap.parse_args(argv)
    rep = run(args.piles, args.sites, args.forward, args.members, args.out, args.ref, args.end,
              args.sub_ref, args.sub_end)
    print(json.dumps({k: rep[k] for k in ("scores", "printed_year_check")}, indent=1,
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
