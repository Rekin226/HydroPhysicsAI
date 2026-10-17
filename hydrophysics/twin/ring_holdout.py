"""Continuation test of a frozen column on the 2023-2025 compaction rings and GNSS.

The leveling network the column was fitted on ends in 2021. The 14 multi-layer
compaction wells (rings) and the WRA GNSS stations run to March 2025, so they are the
only subsidence observations past the forecast origin. This module drives the frozen
flow model and column through January 2023 to March 2025 under three forcings (the
calibration's climatology; recorded pumping; recorded pumping and weather) and scores
each site's cumulative displacement and annual rate against the observations and
against two simple baselines (the site's own 2017-2022 rate continued; no change).

This is NOT a clean holdout: the rings through 2022 were the Stage-2 target and the
"independent check" used while the column configuration was chosen, the ring cache
already extended into 2025 while the model was selected, the rings measure their
anchor-to-surface interval while the column is total depth, GNSS carries tectonic
motion, and 2023-25 pumping uses the 2012-22 meter population under the frozen
conversion. The report carries these caveats verbatim.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .forward import build_model, compaction, load_columns, load_members, rollout
from .mechanism import band_compaction, ring_cells, ring_hiatus_table, ring_positions
from .release import atomic_json, sha256
from .scenario import climatology

ARMS = ("climatology", "observed_pumping", "observed_both")
MAX_REF_AGE_DAYS = 62   # a reference reading older than this before ``ref`` is not used
CAVEATS = [
    "(i) rings through 2022 were the Stage-2 target (calibrate_mlcw) and the independent "
    "check (calibrate_coupled rings_independent_r2) used in configuration, guard and tau "
    "ceiling selection",
    "(ii) the clean ring cache already extended to 2025-03 while the model was selected; "
    "docs/TWIN_RELEASE.md states these are not a fresh independent holdout",
    "(iii) rings measure their anchor-to-surface interval (0-290 m), the column is total "
    "depth, GNSS is the full column plus tectonic and reference-frame motion",
    "(iv) 2023-25 pumping uses the 2012-22 meter population and the frozen kWh conversion; "
    "weather is bias-aligned reanalysis",
    "(v) 27 months only; the flow model's head RMSE in this period is about 1.85 m",
    "(vi) sites with fewer than 12 observed months in the window are reported but excluded "
    "from the scores",
    "(vii) the rings have no survey in 2022 (December 2021 to February 2023) and change depth "
    "reference across that gap (every ring at a site jumps by a common -16 to +50 cm, and the "
    "monitored interval by up to 8 cm more than its rate implies); each ring site's "
    "cumulative therefore starts at its first 2023 survey, and the model and the baselines "
    "are differenced over the same months, about 2.1 of the 2.25 years",
]


# --------------------------------------------------------------------------------------
# scoring (pure numpy / pandas, tested synthetically)
# --------------------------------------------------------------------------------------
def window_span(s: pd.Series, ref: str, t0: str, t1: str,
                max_ref_age_days: float | None = None
                ) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """``(start, end)`` of the cumulative displacement :func:`window_stats` measures: the
    last observation on or before ``ref`` and the last one in ``[t0, t1]``. With
    ``max_ref_age_days``, a reference more than that many days before ``ref`` is replaced by
    the first observation in the window, so a survey hiatus ending inside it (the rings'
    2022 gap, across which they change depth reference) is never bridged. None when the
    window or the reference is unobserved (a record that starts inside the window stays
    unscored, as without the age limit)."""
    s = s.dropna()
    ref_end = pd.Timestamp(ref) + pd.offsets.MonthEnd(0)
    before = s.loc[:ref_end]
    win = s.loc[pd.Timestamp(t0): pd.Timestamp(t1) + pd.offsets.MonthEnd(0)]
    if win.empty or before.empty:
        return None
    if max_ref_age_days is None or (ref_end - before.index[-1]).days <= max_ref_age_days:
        return before.index[-1], win.index[-1]
    return win.index[0], win.index[-1]


def window_stats(s: pd.Series, ref: str, t0: str, t1: str, min_months: int = 12,
                 max_ref_age_days: float | None = None) -> tuple[float, float, int]:
    """``(cumulative, rate, n)``: displacement over :func:`window_span` (same units as
    ``s``), the OLS slope over the window (units/yr) and the number of observed months.
    NaN when the span is undefined or the window has fewer than ``min_months``."""
    s = s.dropna()
    win = s.loc[pd.Timestamp(t0): pd.Timestamp(t1) + pd.offsets.MonthEnd(0)]
    n = int(len(win))
    span = window_span(s, ref, t0, t1, max_ref_age_days)
    if span is None or n < min_months:
        return float("nan"), float("nan"), n
    years = (win.index - win.index[0]).days / 365.25
    slope = float(np.polyfit(years, win.to_numpy(dtype=float), 1)[0])
    return float(s.loc[span[1]] - s.loc[span[0]]), slope, n


def span_change(s: pd.Series, span: tuple[pd.Timestamp, pd.Timestamp] | None) -> float:
    """``s`` at the end of ``span`` minus ``s`` at its start, matched by calendar month
    (a model series differenced over an observed series' span); NaN without a span."""
    if span is None:
        return float("nan")
    by_month = pd.Series(s.to_numpy(dtype=float), index=s.index.to_period("M"))
    return float(by_month.loc[span[1].to_period("M")] - by_month.loc[span[0].to_period("M")])


def continued_rate(s: pd.Series, t0: str, t1: str, min_months: int = 24) -> float:
    """OLS slope (units/yr) of the observed series over ``[t0, t1]``; NaN below support."""
    s = s.dropna().loc[pd.Timestamp(t0): pd.Timestamp(t1) + pd.offsets.MonthEnd(0)]
    if len(s) < min_months:
        return float("nan")
    years = (s.index - s.index[0]).days / 365.25
    return float(np.polyfit(years, s.to_numpy(dtype=float), 1)[0])


def r2_rmse(pred: np.ndarray, obs: np.ndarray) -> dict:
    """R² (1 - SSE/SST), RMSE and bias over the sites where both are finite."""
    m = np.isfinite(pred) & np.isfinite(obs)
    if m.sum() < 3:
        return {"r2": float("nan"), "rmse": float("nan"), "bias": float("nan"), "n": int(m.sum())}
    p, o = pred[m], obs[m]
    sst = float(((o - o.mean()) ** 2).sum())
    sse = float(((p - o) ** 2).sum())
    return {"r2": 1.0 - sse / sst if sst > 0 else float("nan"),
            "rmse": float(np.sqrt(sse / m.sum())), "bias": float((p - o).mean()),
            "n": int(m.sum())}


# --------------------------------------------------------------------------------------
# observations
# --------------------------------------------------------------------------------------
def ring_totals(ddir: str) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    """{site: monthly total-column compaction (m, positive = shortening)} and site xy."""
    st = pd.read_csv(os.path.join(ddir, "mlcw_stations.csv"))
    pos = ring_positions(ddir)
    series, rows = {}, []
    for _, r in st.iterrows():
        name = r["sub_id"]
        if name not in pos:
            continue
        series[name] = band_compaction(pos[name])["total"]
        rows.append({"site": name, "x": float(r["x"]), "y": float(r["y"])})
    return series, pd.DataFrame(rows)


def gnss_vertical(ddir: str, step_m: float = 0.10) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    """{station: monthly median height change, sign flipped so sinking is positive (m)}
    and station xy. A station with a month-to-month jump above ``step_m`` is dropped
    (datum or antenna change)."""
    meta = pd.read_parquet(os.path.join(ddir, "ls_cache", "ls-wra-gnss-obs__meta.parquet"))
    series, rows = {}, []
    for _, r in meta.iterrows():
        sid = str(r["_station_id"])
        path = os.path.join(ddir, "ls_cache", f"ls-wra-gnss-obs__{sid}.parquet")
        if not os.path.isfile(path):
            continue
        df = pd.read_parquet(path)
        s = pd.to_numeric(df["value"], errors="coerce").dropna()
        s.index = pd.to_datetime(s.index)
        monthly = s.groupby(s.index.to_period("M")).median()
        monthly.index = monthly.index.to_timestamp("M")
        if monthly.empty or (monthly.diff().abs() > step_m).any():
            continue
        series[sid] = -(monthly - monthly.iloc[0])
        rows.append({"site": sid, "x": float(r["X_3826"]), "y": float(r["Y_3826"]),
                     "name": str(r.get("ObservatoryName", ""))})
    return series, pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# the model under three forcings
# --------------------------------------------------------------------------------------
def model_series(inp, members, forcing: dict, vep_json: str, ddir: str, cells: np.ndarray,
                 device, n_future: int, log=print) -> dict[str, np.ndarray]:
    """Per arm: ``(n_members, n_cells, T_hist + n_future)`` cumulative subsidence (m) of
    the frozen column on each member's own free-running hindcast continued by the arm's
    forcing from the record's end state (not a nudged state: the column was calibrated
    on free-running heads, and an observation update at the origin would put a head step
    into it)."""
    out = {arm: [] for arm in forcing}
    E_hist = inp.E_total[:, 1:].to(device)
    R_hist = inp.recharge_field[:, 1:].to(device)
    ground = inp.ground_elev.to(device)
    h0 = inp.initial_heads(0).to(device)
    for i, member in enumerate(members):
        t0 = time.perf_counter()
        model, scalars, zone = build_model(inp.grid, member, device)
        if "delay_Sd" in scalars or "aqt_Sa" in scalars:
            raise ValueError("slow-store state is not carried by this test")
        # zone: the column's map from the member's meta (six zones for a
        # --zone-county-split member, whose column must be split the same way)
        col = load_columns(vep_json, ddir, inp.hf, device, zone_of_cell=zone,
                           hpc0_fast_days=365.0,
                           county_sha256=member.meta.get("zone_county_sha256"))[0][1]
        hist = rollout(model, scalars, h0, E_hist, R_hist, ground, apex_from=h0)
        for arm, (E, R) in forcing.items():
            fut = rollout(model, scalars, hist[..., -1], E[:, :n_future].to(device),
                          R[:, :n_future].to(device), ground, month0=0, apex_from=h0)
            heads = torch.cat([hist, fut[..., 1:]], dim=-1)
            subs = compaction(col, heads)                       # (A, T) m
            out[arm].append(subs[cells.clip(min=0)])
        log(f"member {i + 1}/{len(members)} {member.label}: {time.perf_counter() - t0:.0f}s")
    return {arm: np.stack(v) for arm, v in out.items()}


def score_sites(obs: dict[str, pd.Series], xy: pd.DataFrame, cells: np.ndarray,
                sims: dict[str, np.ndarray], dates: pd.DatetimeIndex, archive: np.ndarray,
                ref: str, t0: str, t1: str, base_t0: str, base_t1: str,
                rates_only: bool = False, max_ref_age_days: float | None = MAX_REF_AGE_DAYS
                ) -> tuple[pd.DataFrame, dict]:
    """Per-site table and pooled scores for one observation family. Each site's observed
    cumulative runs over its own :func:`window_span` (``ref_month`` to ``end_month``), and
    the model, the archive and the continued-rate baseline are taken over the same months;
    rates are OLS slopes over the whole window, observed months only for the observations."""
    rows = []
    for k, (_, r) in enumerate(xy.iterrows()):
        name, c = r["site"], int(cells[k])
        s = obs[name]
        span = window_span(s, ref, t0, t1, max_ref_age_days)
        years = (span[1] - span[0]).days / 365.25 if span else float("nan")
        cum_o, rate_o, n = window_stats(s, ref, t0, t1, max_ref_age_days=max_ref_age_days)
        prior = continued_rate(s, base_t0, base_t1)
        row = {"site": name, "cell": c, "n_months": n,
               "ref_month": f"{span[0]:%Y-%m}" if span else "",
               "end_month": f"{span[1]:%Y-%m}" if span else "", "span_yr": years,
               "obs_cum_cm": cum_o * 100,
               "obs_rate_cm_yr": rate_o * 100, "prior_rate_cm_yr": prior * 100,
               "base_continue_cum_cm": prior * years * 100, "base_zero_cum_cm": 0.0,
               "base_zero_rate_cm_yr": 0.0}
        if c >= 0:
            a = pd.Series(archive[c], index=dates)
            rate_a = window_stats(a, ref, t0, t1, min_months=1)[1]
            row.update(archive_cum_cm=span_change(a, span) * 100,
                       archive_rate_cm_yr=rate_a * 100)
            for arm, sim in sims.items():
                series = [pd.Series(sim[m, k], index=dates) for m in range(sim.shape[0])]
                stats = np.array([(span_change(x, span),
                                   window_stats(x, ref, t0, t1, min_months=1)[1])
                                  for x in series])
                row[f"{arm}_cum_cm"] = float(np.mean(stats[:, 0])) * 100
                row[f"{arm}_rate_cm_yr"] = float(np.mean(stats[:, 1])) * 100
                lo, hi = np.quantile(stats[:, 0], [0.1, 0.9]) * 100
                row[f"{arm}_cum_p10_cm"], row[f"{arm}_cum_p90_cm"] = float(lo), float(hi)
                lo, hi = np.quantile(stats[:, 1], [0.1, 0.9]) * 100
                row[f"{arm}_rate_p10_cm_yr"], row[f"{arm}_rate_p90_cm_yr"] = float(lo), float(hi)
        rows.append(row)
    tab = pd.DataFrame(rows)
    scored = tab[(tab.n_months >= 12) & (tab.cell >= 0)]
    scores = {"n_sites": int(len(scored)), "sites_excluded": tab.site[~tab.index.isin(scored.index)].tolist()}
    for q, oc in (("rate", "obs_rate_cm_yr"),) + (() if rates_only else (("cum", "obs_cum_cm"),)):
        unit = "_cm_yr" if q == "rate" else "_cm"
        block = {}
        for name in list(sims) + ["archive", "base_continue", "base_zero"]:
            col = f"{name}_{q}{unit}" if name != "base_continue" or q == "cum" else "prior_rate_cm_yr"
            if col in scored:
                block[name] = r2_rmse(scored[col].to_numpy(dtype=float),
                                      scored[oc].to_numpy(dtype=float))
        # each block against its own member band (the rate block once used the cumulative's)
        for arm in sims:
            lo, hi = scored[f"{arm}_{q}_p10{unit}"], scored[f"{arm}_{q}_p90{unit}"]
            block[arm]["p10_p90_coverage"] = float(((scored[oc] >= lo) & (scored[oc] <= hi)).mean())
        scores[q] = block
    return tab, scores


def run(theta: list[str], inputs_pkl: str, forcing_npz: str, vep_json: str, forward_npz: str,
        ddir: str, out: Path, device: str = "cuda", t0: str = "2023-01-01",
        t1: str = "2025-03-31", max_ref_age_days: float | None = MAX_REF_AGE_DAYS,
        log=print) -> dict:
    torch.set_num_threads(8)
    device = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
    with open(inputs_pkl, "rb") as fh:
        inp = pickle.load(fh)
    members = load_members(theta)
    fz = np.load(forcing_npz)
    f_dates = pd.DatetimeIndex(pd.to_datetime(fz["dates"]))
    n_future = int(((f_dates >= pd.Timestamp(t0)) & (f_dates <= pd.Timestamp(t1))).sum())
    E_cli, cli_dates = climatology(inp.E_total.numpy(), inp.dates, n_future)
    R_cli, _ = climatology(inp.recharge_field.numpy(), inp.dates, n_future)
    E_act = torch.tensor(fz["energy"][:, :n_future], dtype=torch.float64)
    R_act = torch.tensor(fz["recharge"][:, :n_future], dtype=torch.float64)
    forcing = {"climatology": (torch.tensor(E_cli), torch.tensor(R_cli)),
               "observed_pumping": (E_act, torch.tensor(R_cli)),
               "observed_both": (E_act, R_act)}
    dates = pd.DatetimeIndex(list(inp.dates) + list(cli_dates)) + pd.offsets.MonthEnd(0)

    with np.load(forward_npz, allow_pickle=False) as z:
        npz = {k: z[k] for k in ("mask", "dx", "x0", "y0", "dates", "subs_mean_by_rheology")}
    archive = np.asarray(npz["subs_mean_by_rheology"][0, 0], dtype="float64")
    arch_dates = pd.DatetimeIndex(pd.to_datetime(npz["dates"])) + pd.offsets.MonthEnd(0)

    assert (arch_dates[:len(dates)] == dates).all()
    archive = archive[:, :len(dates)]          # the archive runs to 2032; score its 2023-25 only
    rings, ring_xy = ring_totals(ddir)
    gnss, gnss_xy = gnss_vertical(ddir)
    ring_c = ring_cells(npz, ring_xy[["x", "y"]].to_numpy())
    gnss_c = ring_cells(npz, gnss_xy[["x", "y"]].to_numpy()) if len(gnss_xy) else np.array([], int)
    keep = gnss_c >= 0
    gnss_xy, gnss_c = gnss_xy[keep].reset_index(drop=True), gnss_c[keep]
    cells = np.concatenate([ring_c, gnss_c])
    log(f"{len(ring_xy)} ring sites ({int((ring_c >= 0).sum())} on grid), "
        f"{len(gnss_xy)} GNSS stations on grid; {n_future} future months on {device}")
    sims = model_series(inp, members, forcing, vep_json, ddir, cells, device, n_future, log)
    nr = len(ring_xy)
    ref = str((pd.Timestamp(t0) - pd.offsets.MonthEnd(1)).date())
    ring_tab, ring_scores = score_sites(
        rings, ring_xy, ring_c, {a: s[:, :nr] for a, s in sims.items()}, dates, archive,
        ref, t0, t1, "2017-01-01", "2022-12-31", max_ref_age_days=max_ref_age_days)
    gnss_tab, gnss_scores = score_sites(
        gnss, gnss_xy, gnss_c, {a: s[:, nr:] for a, s in sims.items()}, dates, archive,
        ref, t0, t1, "2020-01-01", "2022-12-31", rates_only=True,
        max_ref_age_days=max_ref_age_days)
    # the rings' survey hiatuses and the depth-reference change across the 2022 one
    hiatuses = ring_hiatus_table(ring_positions(ddir))
    disc = hiatuses[hiatuses.discontinuity.astype(bool)
                    & (hiatuses.before <= pd.Timestamp(ref))
                    & (hiatuses.after >= pd.Timestamp(ref) - pd.Timedelta(days=366))]
    by_site = disc.drop_duplicates("site", keep="last").set_index("site")
    for col in ("common_offset_m", "unexplained_m"):
        ring_tab[f"hiatus_{col[:-2]}_cm"] = ring_tab.site.map(by_site[col] * 100)
    out.mkdir(parents=True, exist_ok=True)
    ring_tab.to_csv(out / "rings_sites.csv", index=False)
    gnss_tab.to_csv(out / "gnss_sites.csv", index=False)
    hiatuses.to_csv(out / "rings_hiatuses.csv", index=False)
    report = {
        "schema": 1, "task": "frozen-column-continuation-rings-gnss",
        "prospective_validation": False, "independent_holdout": False,
        "causal_policy_validation": False,
        "window": [t0, t1], "reference_month": ref,
        "reference_rule": (f"cumulative from the last reading on or before {ref}, or from the "
                           f"site's first reading in the window when that is more than "
                           f"{max_ref_age_days} days older; model, archive and baselines over "
                           "the same months (rings_sites.csv ref_month, end_month)"),
        "max_ref_age_days": max_ref_age_days,
        "ring_discontinuity": {
            "n_sites": int(disc.site.nunique()),
            "survey_gap": sorted({f"{a:%Y-%m}..{b:%Y-%m}" for a, b in
                                  zip(disc.before, disc.after, strict=True)}),
            "common_offset_cm": [round(float(v) * 100, 1) for v in disc.common_offset_m],
            "interval_unexplained_cm": [round(float(v) * 100, 1) for v in disc.unexplained_m],
            "sites_referenced_after_gap": ring_tab.site[ring_tab.ref_month > ref[:7]].tolist()},
        "members": len(members),
        "start_state": "free-running hindcast end state (no observation update)",
        "column": vep_json, "column_sha256": sha256(vep_json),
        "theta": theta, "inputs_pkl": inputs_pkl, "inputs_sha256": sha256(inputs_pkl),
        "forcing": forcing_npz, "forcing_sha256": sha256(forcing_npz),
        "forward_archive": forward_npz, "forward_sha256": sha256(forward_npz),
        "units": "cm (cumulative, positive = sinking) and cm/yr (OLS rate over the window)",
        "baselines": {"base_continue": "site's own 2017-22 (rings) / 2020-22 (GNSS) OLS rate continued",
                      "base_zero": "no displacement", "archive": "the forward archive's climatology column"},
        "rings": ring_scores, "gnss": gnss_scores, "independence": CAVEATS,
    }
    atomic_json(out / "report.json", report)
    return report


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--theta", action="append", required=True)
    ap.add_argument("--inputs-pkl", required=True)
    ap.add_argument("--forcing", required=True)
    ap.add_argument("--vep-json", required=True)
    ap.add_argument("--forward", required=True)
    ap.add_argument("--data", default=os.environ.get("HYDROMIND_GW_DATA"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-ref-age-days", type=float, default=MAX_REF_AGE_DAYS,
                    help="a reference reading older than this before the window is not used; "
                         "the site's first reading in the window is (default %(default)s)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    rep = run(args.theta, args.inputs_pkl, args.forcing, args.vep_json, args.forward,
              args.data, args.out, args.device, max_ref_age_days=args.max_ref_age_days)
    for fam in ("rings", "gnss"):
        print(f"== {fam}: {rep[fam]['n_sites']} sites scored; excluded {rep[fam]['sites_excluded']}")
        for q, block in rep[fam].items():
            if isinstance(block, dict):
                for name, m in block.items():
                    cov = f"  p10-p90 cov {m['p10_p90_coverage']:.2f}" if "p10_p90_coverage" in m else ""
                    print(f"  {q:4s} {name:18s} R2 {m['r2']:+.2f}  RMSE {m['rmse']:.2f}  bias {m['bias']:+.2f}{cov}")
    print(json.dumps({k: rep[k] for k in ("window", "members", "start_state")}))


if __name__ == "__main__":
    main()
