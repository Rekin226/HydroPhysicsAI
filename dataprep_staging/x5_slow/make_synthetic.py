"""I1.X5SC.gate0 synthetic leveling generator (CPU, under a minute per truth).

For each truth (N, A, V; e2common.SPEC["truths"]) it writes

    <out>/truth_<T>/data/            a data dir mirroring chou-shui-data/data by symlinks,
                                     except ls_cache/ls-wra-lsp-obs__choushui_panel.parquet,
                                     which is the synthetic panel (same parquet schema)
    <out>/truth_<T>/truth_column.json   the truth's zonal parameter sets
    <out>/truth_<T>/manifest.json       parameters, noise, hashes, diagnostics

and <out>/generator_manifest.json. The synthetic panel keeps the real benchmark ids,
coordinates and survey dates 2012-2021 (inventory columns only; elev_m is never read) and
replaces the elevation by BASE - truth compaction at the benchmark's cell and survey month -
noise. lon/lat/town/name are written as nulls (calibrate_coupled reads none of them).

The truth heads are dump_fit_heads.py's npz (calibrate_coupled's own nudged hindcast);
its provenance is checked against the spec. Self-check: every truth's compaction at the
benchmark cells through this module's driver (layer mean, compaction.column_compaction)
equals calibrate_coupled's own path (forward.column_from_params + calibrate_coupled._predict
on the layer heads).

Usage (repository root, frozen code):
    results/twin_runs/par32/py.sh dataprep_staging/x5_slow/make_synthetic.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2common as C  # noqa: E402

PANEL_NAME = os.path.basename(C.SPEC["panel"])
# 1e-4 cm: forward.column_from_params stores the parameters in float32 VEPColumn
# parameters (relative rounding ~1e-7); a wrong driver or mapping is off by cm
SELFCHECK_TOL_M = 1e-6


def noise_sd_m(column_json: dict) -> tuple[float, float]:
    """(residual sd s, per-survey elevation noise sd s/sqrt(2)) in metres, from the q0_nh
    column's in-sample leveling score (a model statistic)."""
    lev = column_json["leveling"]
    s_cm = math.sqrt(max(float(lev["rmse_cm"]) ** 2 - float(lev["bias_cm"]) ** 2, 0.0))
    return s_cm / 100.0, s_cm / 100.0 / math.sqrt(2.0)


def bisect(f, lo: float, hi: float, target: float, tol: float = 1e-6, it: int = 80) -> float:
    """x in [lo, hi] with f(x) = target for an increasing f."""
    flo, fhi = f(lo), f(hi)
    if not flo <= target <= fhi:
        raise ValueError(f"target {target} outside [{flo}, {fhi}] on [{lo}, {hi}]")
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        if f(mid) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def check_provenance(prov: dict) -> dict:
    """The truth heads must be calibrate_coupled's nudged hindcast of the declared theta
    and flags on clean_heads v1 (dump_fit_heads.py); returns the checked items."""
    theta = C.repo_path(C.SPEC["theta"])
    want = {"argv_flags": list(C.SPEC["calibrate_common"]),
            "theta": C.SPEC["theta"], "theta_sha256": C.file_sha256(theta),
            "wells_dir": C.SPEC["heads"]["wells_dir"]}
    argv = list(prov.get("argv") or [])
    got = {"argv_flags": argv[2:2 + len(C.SPEC["calibrate_common"])],
           "theta": prov.get("theta"), "theta_sha256": prov.get("theta_sha256"),
           "wells_dir": prov.get("wells_dir")}
    bad = [k for k in want if got[k] != want[k]]
    if argv[:2] != ["--theta", C.SPEC["theta"]]:
        bad.append("argv theta")
    if bad:
        raise SystemExit(f"truth heads provenance differs from the spec in {bad}")
    return want


def mirror_data_dir(src: Path, dst: Path) -> None:
    """``dst`` = symlinks to every entry of ``src`` except ls_cache, which is a real
    directory of symlinks to every entry except the panel (written separately)."""
    dst.mkdir(parents=True)
    for e in sorted(os.listdir(src)):
        if e == "ls_cache":
            continue
        os.symlink(src / e, dst / e)
    (dst / "ls_cache").mkdir()
    for e in sorted(os.listdir(src / "ls_cache")):
        if e == PANEL_NAME:
            continue
        os.symlink(src / "ls_cache" / e, dst / "ls_cache" / e)


def remove_synthetic_dir(d: Path) -> None:
    """Remove a dir written by ``mirror_data_dir`` without following any symlink."""
    if not d.exists():
        return
    lc = d / "ls_cache"
    if lc.is_dir() and not lc.is_symlink():
        for e in os.listdir(lc):
            p = lc / e
            if p.is_symlink() or p.name == PANEL_NAME:
                p.unlink()
            else:
                raise RuntimeError(f"unexpected entry {p}; refusing to delete")
        lc.rmdir()
    for e in os.listdir(d):
        p = d / e
        if p.is_symlink():
            p.unlink()
        else:
            raise RuntimeError(f"unexpected entry {p}; refusing to delete")
    d.rmdir()


def write_panel(path: Path, inv: pd.DataFrame, elev: np.ndarray, src_panel: Path) -> None:
    """The synthetic panel in the source panel's schema (field names, order and types)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pq.read_schema(src_panel).remove_metadata()
    cols = {}
    for f in schema:
        if f.name == "elev_m":
            cols[f.name] = pa.array(np.asarray(elev, dtype="float64"), type=f.type)
        elif f.name == "datetime":
            cols[f.name] = pa.array(inv["datetime"].to_numpy(dtype="datetime64[us]"),
                                    type=f.type)
        elif f.name in ("sid", "x_3826", "y_3826"):
            cols[f.name] = pa.array(inv[f.name].to_numpy(), type=f.type)
        else:
            cols[f.name] = pa.nulls(len(inv), type=f.type)
    tbl = pa.Table.from_pydict(cols, schema=schema)
    pq.write_table(tbl, path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--heads", default=str(C.repo_path(C.HEADS_NPZ)))
    ap.add_argument("--out", default=str(C.repo_path(C.SYNTH_DIR)))
    ap.add_argument("--force", action="store_true", help="replace existing synthetic dirs "
                    "(refused once the declaration is registered)")
    args = ap.parse_args(argv)
    out = Path(args.out)
    truths = list(C.SPEC["primary_truths"])
    if args.force and C.repo_path(C.DECL_PATH).exists() and \
            out.resolve() == C.repo_path(C.SYNTH_DIR).resolve():
        raise SystemExit(f"{C.DECL_ID} is registered: its synthetic panels are frozen")

    src_data = C.repo_path(C.SPEC["data_src"])
    src_panel = src_data / C.SPEC["panel"]
    prov = C.heads_provenance(args.heads)
    checked = check_provenance(prov)
    H = C.truth_heads(args.heads, C.SPEC["heads"]["scenario"], C.SPEC["heads"]["last_month"],
                      layers=True)
    h, dates, grid, lay = H["h"], H["dates"], H["grid"], H["layers"]
    if str(dates[-1].date()) != pd.Period(C.SPEC["heads"]["last_month"], freq="M") \
            .start_time.strftime("%Y-%m-%d"):
        raise SystemExit(f"the truth heads end {dates[-1].date()}, not at the spec's last "
                         f"month {C.SPEC['heads']['last_month']}")
    zone_cell = C.zone_map(grid, C.repo_path(C.SPEC["theta"]))
    base, colj = C.load_column_sets(C.repo_path(C.SPEC["q0_nh_column"]))
    if len(base) != 3:
        raise SystemExit(f"q0_nh column has {len(base)} zones, expected 3")
    s_m, sigma_e = noise_sd_m(colj)

    # inventory (sid, datetime, x, y only) -> cells, zones, months
    inv = C.panel_inventory(src_panel)
    if inv["datetime"].max() >= pd.Timestamp(C.SPEC["survey_window"][1]):
        raise SystemExit("inventory has a survey after the window")
    cell_of = C.benchmark_cells(inv, grid)
    inv["cell"] = inv["sid"].map(cell_of).astype("int64")
    inv["month"] = [C.nearest_month(dates, t) for t in inv["datetime"]]
    mod = pd.DatetimeIndex(dates).to_period("M")
    same = np.array([mod[m] == t.to_period("M") for m, t in
                     zip(inv["month"], inv["datetime"], strict=True)])
    on = inv["cell"].to_numpy() >= 0
    inv["zone"] = np.where(on, zone_cell[np.maximum(inv["cell"].to_numpy(), 0)], -1)
    n_surv = inv.groupby("sid").size()
    bench = inv.groupby("sid")[["cell", "zone"]].first()
    bench["n"] = n_surv
    calib = bench[(bench["cell"] >= 0) & (bench["zone"] == C.DISTAL) & (bench["n"] >= 5)]
    if len(calib) < 50:
        raise SystemExit(f"only {len(calib)} distal benchmarks for the truth calibration")

    # truth calibration on the distal benchmarks' cells (weighted per benchmark)
    ccells = calib["cell"].to_numpy()
    hc = h[ccells]
    zc = np.full(len(ccells), C.DISTAL)
    k16 = int(np.nonzero(mod == pd.Period("2016-01", freq="M"))[0][0])
    k17 = int(np.nonzero(mod == pd.Period("2017-01", freq="M"))[0][0])
    aa = np.full(len(ccells), k16)
    bb = np.full(len(ccells), k17)

    def bg16(sets):
        return float(C.background_cm(sets, zc, hc, k16, aa, bb, dates).mean())

    params = {
        "h_a0": bisect(lambda v: bg16(C.truth_sets(base, "A", h_a0=v)), 0.0, 30.0,
                       C.BG_TARGET_CM),
        "h_pc0": bisect(lambda v: bg16(C.truth_sets(base, "V", h_pc0=v)), 0.0,
                        float(h.max() - h.min()), C.BG_TARGET_CM)}

    # one noise draw for every truth (common random numbers)
    rng = np.random.default_rng(C.SPEC["noise"]["seed"])
    noise = rng.normal(0.0, sigma_e, size=len(inv))

    cells_all = np.unique(inv.loc[on, "cell"].to_numpy())
    zones_all = zone_cell[cells_all]
    pos = {int(c): i for i, c in enumerate(cells_all)}

    # self-check: this module's driver vs calibrate_coupled's own path, every truth
    selfcheck = {"what": "compaction at every on-grid benchmark cell, months <= "
                         f"{C.SPEC['heads']['last_month']}: e2common.column_series (layer "
                         "mean, compaction.column_compaction) vs forward.column_from_params "
                         "+ calibrate_coupled._predict on the layer heads",
                 "tol_m": SELFCHECK_TOL_M, "max_abs_diff_m": {}}
    series = {}
    for T in truths:
        sets = C.truth_sets(base, T, h_a0=params["h_a0"], h_pc0=params["h_pc0"])
        S = C.column_series(sets, zones_all, h[cells_all])            # (n_cells, T) m
        P = C.predict_like_calibrate(sets, zones_all, lay[:, cells_all, :].transpose(1, 0, 2))
        d = float(np.abs(S - P).max())
        selfcheck["max_abs_diff_m"][T] = d
        if not d <= SELFCHECK_TOL_M:
            raise SystemExit(f"self-check failed for truth {T}: max |diff| {d} m")
        series[T] = (sets, S)
    del lay

    out.mkdir(parents=True, exist_ok=True)
    top = {"spec_id": C.SPEC["id"], "spec_sha256": C.spec_sha256(),
           "heads_npz": str(Path(args.heads).resolve()),
           "heads_sha256": C.file_sha256(args.heads), "heads_provenance_checked": checked,
           "heads_code_tree_sha256": prov.get("code", {}).get("tree_sha256"),
           "heads_months": [str(dates[0].date()), str(dates[-1].date())],
           "q0_nh_column_sha256": C.file_sha256(C.repo_path(C.SPEC["q0_nh_column"])),
           "code_tree_sha256": C.code_state()["tree_sha256"],
           "noise": {"residual_sd_cm": s_m * 100.0, "per_survey_sd_cm": sigma_e * 100.0,
                     "seed": C.SPEC["noise"]["seed"]},
           "inventory": {"rows": int(len(inv)), "benchmarks": int(inv["sid"].nunique()),
                         "on_grid": int((bench["cell"] >= 0).sum()),
                         "distal_on_grid_ge5": int(len(calib)),
                         "first": str(inv["datetime"].min().date()),
                         "last": str(inv["datetime"].max().date()),
                         "month_of_date_equals_nearest_month": bool(same.all())},
           "solved": params, "selfcheck": selfcheck, "truths": {}}
    for T in truths:
        sets, S = series[T]
        sub = np.zeros(len(inv))
        sub[on] = S[[pos[int(c)] for c in inv.loc[on, "cell"]], inv.loc[on, "month"]]
        elev = float(C.SPEC["base_elev_m"]) - sub - noise
        tdir = out / f"truth_{T}"
        ddir = tdir / "data"
        if ddir.exists():
            if not args.force:
                raise SystemExit(f"{ddir} exists; pass --force to replace it")
            remove_synthetic_dir(ddir)
        tdir.mkdir(parents=True, exist_ok=True)
        mirror_data_dir(src_data, ddir)
        ppath = ddir / C.SPEC["panel"]
        write_panel(ppath, inv, elev, src_panel)
        with open(tdir / "truth_column.json", "w", encoding="utf-8") as fh:
            json.dump({"truth": T, "zonal": sets, "config": "zonal",
                       "creep": "aquitard" if T == "A" else "vep",
                       "note": "synthetic truth for I1.X5SC.gate0; not a fitted column"}, fh,
                      indent=1)
        diag = diagnostics(T, sets, ppath, h, dates, zone_cell, grid, k16, k17)
        man = {"truth": T, "description": C.SPEC["truths"][T], "data_dir": str(ddir.resolve()),
               "panel": str(ppath), "panel_sha256": C.file_sha256(ppath),
               "truth_column": str(tdir / "truth_column.json"),
               "distal_params": sets[C.DISTAL], "diagnostics": diag,
               "spec_sha256": C.spec_sha256(), "heads_sha256": top["heads_sha256"]}
        with open(tdir / "manifest.json", "w", encoding="utf-8") as fh:
            json.dump(man, fh, indent=1, default=str)
        top["truths"][T] = {k: man[k] for k in ("data_dir", "panel_sha256", "distal_params")}
        top["truths"][T]["diagnostics"] = diag
        print(f"truth {T}: {ppath} ({len(inv)} rows); distal bg 2016 "
              f"{diag['bg2016_distal_cm']:+.3f} cm/yr; screened-out benchmarks "
              f"{diag['screen']['dropped_by_max_rate']} (on grid "
              f"{diag['screen']['dropped_on_grid']})", flush=True)
    with open(out / "generator_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(top, fh, indent=1, default=str)
    print(json.dumps({k: top[k] for k in ("noise", "solved", "selfcheck")}, indent=1))
    return 0


def diagnostics(T, sets, ppath, h, dates, zone_cell, grid, k16, k17) -> dict:
    """Synthetic-world numbers only: what calibrate_coupled will see of the panel, the
    truth's distal background in 2016 and at each origin's pairs."""
    from hydrophysics.twin.leveling import load_panel, site_subsidence

    ddir = str(Path(ppath).parents[1])
    panel = load_panel(ddir)
    t0, t1 = "2012-01-01", "2022-12-01"           # calibrate_coupled's window
    kept = site_subsidence(panel, t0, t1, min_obs=5, max_rate=0.5)
    allk = site_subsidence(panel, t0, t1, min_obs=5, max_rate=None)
    first = panel.groupby("sid")[["x", "y"]].first()
    cell = {s: grid.cell_of(float(first.loc[s, "x"]), float(first.loc[s, "y"])) for s in allk}
    on = {s for s in allk if cell[s] >= 0}
    out = {"screen": {"ge5_surveys": len(allk), "ge5_on_grid": len(on),
                      "kept_after_max_rate": len(kept),
                      "kept_on_grid": len(on & set(kept)),
                      "dropped_by_max_rate": len(allk) - len(kept),
                      "dropped_on_grid": len(on - set(kept))}}
    distal = [s for s in on if zone_cell[cell[s]] == C.DISTAL]
    cells = np.array([cell[s] for s in distal])
    z = np.full(len(cells), C.DISTAL)
    out["bg2016_distal_cm"] = float(C.background_cm(
        sets, z, h[cells], k16, np.full(len(cells), k16), np.full(len(cells), k17),
        dates).mean())
    if T == "A":
        aq = C.aquitard_series(sets[C.DISTAL], h[cells])
        out["aquitard_term_rate_2016_truth_heads_cm"] = float(
            ((aq[:, k17] - aq[:, k16]) * 100.0 / C.years_between(dates, k16, k17)).mean())
    syn = panel.rename(columns={"x": "x_3826", "y": "y_3826"})
    mod = pd.DatetimeIndex(dates).to_period("M")
    out["origins"] = {}
    for o in C.SPEC["origins"]:
        pr = C.benchmark_pairs(syn, o)
        pr["cell"] = [grid.cell_of(x, y) for x, y in zip(pr["x"], pr["y"], strict=True)]
        pr = pr[pr["cell"] >= 0]
        pr = pr[zone_cell[pr["cell"].to_numpy()] == C.DISTAL]
        ka = np.array([int(np.nonzero(mod == t.to_period("M"))[0][0]) for t in pr["ta"]])
        kb = np.array([int(np.nonzero(mod == t.to_period("M"))[0][0]) for t in pr["tb"]])
        ko = int(np.nonzero(mod == pd.Period(o, freq="M"))[0][0])
        bg = C.background_cm(sets, np.full(len(pr), C.DISTAL), h[pr["cell"].to_numpy()],
                             ko, ka, kb, dates)
        out["origins"][o] = {"distal_pairs": int(len(pr)),
                             "truth_background_cm": float(bg.mean()) if len(pr) else None}
    return out


if __name__ == "__main__":
    raise SystemExit(main())
