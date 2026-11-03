"""N2CS one-county recorded-pumping multipliers (a NEW builder; build_r_both.py is frozen in
N2C's registered commands and is imported, never edited).

N2CS regresses e = obs - ZR on TWO contrasts of the same frozen R1c forward: c_CH (Changhua's
recorded anomaly only) and c_YL (Yunlin's only). Each arm needs a per-cell field in
build_r_both's layout (``month``, ``multiplier`` (n_months, n_active), ``n_active``,
``start``) that carries one county's r(t) and 1 everywhere else:

    changhua   r_CH on the 943 Changhua cells of zone_county_map/v2, 1 on every other cell
               (identical to N2C's declared secondary ``--scope changhua``)
    yunlin     1 on the Changhua cells, r_YL on every other cell (the complement)

Three modes, all writing NEW files only (refuse to overwrite) with a JSON record of every
input hash:

``x1``      the SCORING-TIME arms (2028 session only): r_county(t) by N2C's frozen rule
            (build_r_both.county_ratio + csv_round_trip, so each county's cells carry N2C's
            r bit for bit), months the union of both counties' recorded months from --first,
            a county without a month takes 1. Reads X1 (allowed use forcing_recorded_arm).
``unit``    the pre-gate files: the same layout and months 2023-01..2027-10 with every
            multiplier exactly 1 (no data read). Both must reproduce the run's baseline
            (ZR arm) within 1e-6 m.
``replays`` the OSSE forcing: seven calendar-aligned 58-month replays of the PRE-ORIGIN census
            (pump_electricity/preorigin/v3, opened through common.guard for
            design_time_evidence; rows in_grid / class all, dated <= 2022-12; x5_changhua_
            lever/replay.py's frozen ratio and replay rules, imported). Replay rpY maps
            2023-01 + j onto Y-01 + j (j = 0..57), so the N2CS window 2026-12..2027-10
            replays (Y+3)-12..(Y+4)-10. For each Y: rch{Y}.npz and ryl{Y}.npz, plus
            replays.json with the window anomalies a_CH, a_YL. The product forbids
            forcing_recorded_arm: the replays use twin.forward's recorded-pumping MECHANISM
            as a design device (synthetic future months, never scored against an
            observation, never a recorded arm of a registered test), N2C-reading precedent.

Usage (repository root):
  python -I dataprep_staging/n2cs/build_r_county.py unit --scope changhua --out <npz>
  python -I dataprep_staging/n2cs/build_r_county.py replays --out <dir>
  python -I dataprep_staging/n2cs/build_r_county.py x1 --x1 <county_monthly.parquet> \
      --first 2023-01 --scope changhua|yunlin --out <npz>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(HERE), str(REPO / "dataprep_staging")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import n2cs_const as K  # noqa: E402

SCOPES = ("changhua", "yunlin")


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def target_months(first: str = K.RECORDED_START, last: str = K.WINDOW[1]) -> list[str]:
    y0, m0 = int(first[:4]), int(first[5:7])
    y1, m1 = int(last[:4]), int(last[5:7])
    return [f"{k // 12}-{k % 12 + 1:02d}" for k in range(12 * y0 + m0 - 1, 12 * y1 + m1)]


def assemble(flags: np.ndarray, months: list[str], r_ch: np.ndarray, r_yl: np.ndarray,
             scope: str) -> dict:
    """The forward's per-cell arrays for one county's arm (build_r_both's layout)."""
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}")
    flags = np.asarray(flags, bool)
    r_ch, r_yl = np.asarray(r_ch, float), np.asarray(r_yl, float)
    if r_ch.shape != (len(months),) or r_yl.shape != (len(months),):
        raise ValueError("ratio series and months differ")
    on = r_ch if scope == "changhua" else r_yl
    sel = flags if scope == "changhua" else ~flags
    field = np.where(sel[None, :], on[:, None], 1.0).astype("float64")
    if not np.isfinite(field).all() or (field < 0).any():
        raise ValueError("non-finite or negative multiplier")
    return {"month": np.array(months), "multiplier": field,
            "n_active": np.int64(flags.size), "start": np.str_(months[0])}


def arm_problems(path: Path, flags: np.ndarray, scope: str) -> list[str]:
    """Lead ruling S7: the file on disk moves ONE county (multiplier exactly 1 on every cell of
    the other county), has the forward's layout and the grid of the county map."""
    flags = np.asarray(flags, bool)
    e = []
    with np.load(path, allow_pickle=False) as z:
        f = np.asarray(z["multiplier"], float)
        n = int(z["n_active"])
        months = [str(m) for m in z["month"]]
    if f.ndim != 2 or f.shape != (len(months), flags.size) or n != flags.size:
        return [f"{path}: field {f.shape} does not match {len(months)} months x {flags.size}"]
    other = ~flags if scope == "changhua" else flags
    if not np.array_equal(f[:, other], np.ones_like(f[:, other])):
        e.append(f"{path}: multiplier != 1 on cells outside {scope}")
    if not (np.isfinite(f).all() and (f >= 0).all()):
        e.append(f"{path}: non-finite or negative multiplier")
    return e


def load_flags(path: Path) -> tuple[np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as z:
        flags = np.asarray(z["changhua"]).astype(bool)
        n_active = int(z["n_active"])
        stored = str(z["sha256"])
    if stored != K.R1C_COUNTY_SHA or flags.shape != (n_active,):
        raise ValueError(f"{path}: not the county map R1c was calibrated on")
    return flags, {"path": K.COUNTY_MAP, "file_sha256": sha(path), "flags_sha256": stored,
                   "n_active": n_active, "n_changhua": int(flags.sum())}


def _write_once(out: Path, arrays: dict, rec: dict) -> dict:
    if out.suffix != ".npz":
        raise SystemExit("--out must end in .npz")
    if out.exists() or out.with_suffix(".json").exists():
        raise SystemExit(f"{out} (or its .json) exists; written once")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("xb") as fh:
        np.savez(fh, **arrays)
    rec = dict(rec, npz_sha256=sha(out), builder=f"{K.X}/build_r_county.py",
               builder_sha256=sha(Path(__file__)))
    with out.with_suffix(".json").open("x", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, indent=1) + "\n")
    return rec


def build_unit(flags: np.ndarray, scope: str) -> tuple[dict, dict]:
    months = target_months()
    ones = np.ones(len(months))
    arr = assemble(flags, months, ones, ones, scope)
    return arr, {"mode": "unit", "scope": scope, "first": months[0], "last": months[-1],
                 "n_months": len(months),
                 "rule": "every multiplier exactly 1 (pre-gate: the arm must reproduce the "
                         "run's baseline within 1e-6 m)"}


def build_x1(x1, first: str, flags: np.ndarray, scope: str, last: str | None = None
             ) -> tuple[dict, dict]:
    """Scoring-time arm from X1 by N2C's frozen per-county rule (imported)."""
    if sha(REPO / K.BUILD_R_BOTH) != K.BUILD_R_BOTH_SHA or sha(REPO / K.BUILD_R) != K.BUILD_R_SHA:
        raise SystemExit("build_r_both.py / build_r.py differ from N2C's frozen sha256")
    from n2_policy.build_r_both import county_ratio, csv_round_trip

    per = {c: csv_round_trip(county_ratio(x1, first, c, "value"))
           for c in ("yunlin", "changhua")}
    if last is not None:
        per = {c: r[r["month"] <= last].reset_index(drop=True) for c, r in per.items()}
    months = sorted(set(per["yunlin"]["month"]) | set(per["changhua"]["month"]))
    if not months:
        raise ValueError("no recorded month in range")
    idx = {c: r.set_index("month")["multiplier"].reindex(months).fillna(1.0).to_numpy()
           for c, r in per.items()}
    arr = assemble(flags, months, idx["changhua"], idx["yunlin"], scope)
    win = set(target_months(*K.WINDOW))
    rec = {"mode": "x1", "scope": scope, "first": months[0], "last": months[-1],
           "n_months": len(months),
           "rule": "r_county(t) = X1 in_grid_hybrid county value of month t / its 2012-2022 "
                   "mean for that month of year (N2's frozen rule, build_r_both.county_ratio); "
                   + ("Changhua cells r_CH, other cells 1" if scope == "changhua"
                      else "Changhua cells 1, other cells r_YL"),
           "months_recorded": {c: int(len(r)) for c, r in per.items()},
           "window_months_recorded": {c: int(len(win & set(r["month"]))) for c, r in
                                      per.items()},
           "flags": {c: {str(k): int(v) for k, v in r["flag"].value_counts().items()}
                     for c, r in per.items()}}
    return arr, rec


def read_v3(product_dir: Path):
    from common import guard

    p = guard.open_product(product_dir, K.V3_USE)
    df = p.read_table(K.V3_TABLE, columns=["county", "scope", "class", "date", "value"])
    return df, {"product": K.V3, "use": K.V3_USE, "table": K.V3_TABLE,
                "manifest_sha256": sha(Path(product_dir) / "MANIFEST.json"),
                "table_sha256": p.outputs[K.V3_TABLE]["sha256"], "guard_notes": p.notes,
                "rows_kept_by_guard": int(len(df))}


V3_LOOK_UNIT = "N2CS:v3_county_class_monthly_2012_2022"
V3_LOOK_SCOPE = "c0_pump_electricity_v3:2012-01..2022-12"


def log_v3_look() -> dict:
    """The design look on the pre-origin census, logged BEFORE the read (scoped members: an
    electricity object holding no registered well, benchmark or cell; never universal)."""
    from harness import prereg3

    return prereg3.record_look(
        [V3_LOOK_UNIT], V3_LOOK_SCOPE, K.WP, "design",
        members={V3_LOOK_UNIT: {
            "description": "pump_electricity/preorigin/v3 county_class_monthly.parquet, rows "
                           "scope in_grid, class all, counties changhua and yunlin, dated "
                           "2012-01..2022-12 (columns county, scope, class, date, value, "
                           "flag): the N2CS OSSE replays",
            "none": "electricity object (pre-origin census); contains no registered well, "
                    "leveling benchmark or model cell of the validation-unit registries"}},
        note="[N2CS-replays] design_time_evidence read of the pre-origin census by "
             "build_r_county.py replays, after N2CS.osse_design.v1; no head is read")


def filled_problems(RP, raw) -> list[str]:
    """Lead ruling S8: per replay and county, the share of the 11 window SOURCE months filled
    with r = 1 may not exceed MAX_FILLED_FRACTION."""
    out = []
    for y in K.REPLAY_YEARS:
        src = RP.replay(raw.fillna(1.0), y)
        w = src[(src["month"] >= K.WINDOW[0]) & (src["month"] <= K.WINDOW[1])]
        for c, f in RP.filled_months(raw, w["source"].tolist()).items():
            frac = f["n_filled"] / max(f["n_months"], 1)
            if frac > K.MAX_FILLED_FRACTION:
                out.append(f"rp{y} {c}: {f['n_filled']} of {f['n_months']} window source "
                           f"months filled (> {K.MAX_FILLED_FRACTION})")
    return out


def build_replays(out: Path, product_dir: Path, county_map: Path, *,
                  log_look: bool = True) -> dict:
    if sha(REPO / K.REPLAY) != K.REPLAY_SHA:              # checked BEFORE the import
        raise SystemExit(f"{K.REPLAY} differs from its bound sha {K.REPLAY_SHA[:8]}")
    sys.path.insert(0, str(REPO / "dataprep_staging" / "x5_changhua_lever"))
    import replay as RP

    if target_months() != RP.TARGET:
        raise SystemExit("replay.py's target months differ from N2CS's")
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty; replays are written once")
    flags, cmap = load_flags(county_map)
    look = log_v3_look() if log_look else None
    df, src = read_v3(product_dir)
    raw = RP.raw_ratios(df)
    bad = filled_problems(RP, raw)
    if bad:
        raise SystemExit("REFUSED (max filled fraction): " + "; ".join(bad))
    out.mkdir(parents=True, exist_ok=True)
    ratios = raw.fillna(1.0)
    rec = {"builder": f"{K.X}/build_r_county.py", "builder_sha256": sha(Path(__file__)),
           "rules_from": {"path": K.REPLAY, "sha256": K.REPLAY_SHA,
                          "functions": ["raw_ratios", "replay", "filled_months"]},
           "source": src, "county_map": cmap, "first": K.RECORDED_START,
           "n_months": len(RP.TARGET), "window": list(K.WINDOW),
           "filled_with_r1": RP.filled_months(raw), "max_filled_fraction":
           K.MAX_FILLED_FRACTION, "v3_look": (look or {}).get("time"), "replays": {}}
    for y in K.REPLAY_YEARS:
        rep = RP.replay(ratios, y)
        months = rep["month"].tolist()
        w = rep[(rep["month"] >= K.WINDOW[0]) & (rep["month"] <= K.WINDOW[1])]
        entry = {"source_first": rep["source"].iloc[0], "source_last": rep["source"].iloc[-1],
                 "window": {"months": w["month"].tolist(), "source": w["source"].tolist(),
                            "a_changhua": (w["r_changhua"] - 1.0).tolist(),
                            "a_yunlin": (w["r_yunlin"] - 1.0).tolist()}}
        for scope, tag in (("changhua", "rch"), ("yunlin", "ryl")):
            arr = assemble(flags, months, rep["r_changhua"].to_numpy(float),
                           rep["r_yunlin"].to_numpy(float), scope)
            p = out / f"{tag}{y}.npz"
            with p.open("xb") as fh:
                np.savez(fh, **arr)
            entry[tag] = {"file": p.name, "sha256": sha(p)}
        rec["replays"][f"rp{y}"] = entry
    with (out / "replays.json").open("x", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, indent=1) + "\n")
    return {"out": str(out), "replays_json_sha256": sha(out / "replays.json"),
            "files": {k: {t: v[t]["sha256"][:12] for t in ("rch", "ryl")}
                      for k, v in rec["replays"].items()}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("unit")
    u.add_argument("--scope", choices=SCOPES, required=True)
    u.add_argument("--out", required=True, type=Path)
    u.add_argument("--county-map", type=Path, default=REPO / K.COUNTY_MAP)
    r = sub.add_parser("replays")
    r.add_argument("--out", required=True, type=Path)
    r.add_argument("--product", type=Path, default=REPO / K.V3)
    r.add_argument("--county-map", type=Path, default=REPO / K.COUNTY_MAP)
    x = sub.add_parser("x1")
    x.add_argument("--x1", required=True, type=Path)
    x.add_argument("--first", default=K.RECORDED_START)
    x.add_argument("--last", default=None)
    x.add_argument("--scope", choices=SCOPES, required=True)
    x.add_argument("--out", required=True, type=Path)
    x.add_argument("--county-map", type=Path, default=REPO / K.COUNTY_MAP)
    a = ap.parse_args(argv)
    if a.cmd == "replays":
        print(json.dumps(build_replays(a.out, a.product, a.county_map), indent=1))
        return 0
    flags, cmap = load_flags(a.county_map)
    if a.cmd == "unit":
        arr, rec = build_unit(flags, a.scope)
    else:
        import pandas as pd

        arr, rec = build_x1(pd.read_parquet(a.x1), a.first, flags, a.scope, a.last)
        rec.update({"x1": str(a.x1), "x1_sha256": sha(a.x1),
                    "build_r_both_sha256": sha(REPO / K.BUILD_R_BOTH),
                    "build_r_sha256": sha(REPO / K.BUILD_R)})
    rec["county_map"] = cmap
    rec = _write_once(a.out, arr, rec)
    print(json.dumps({k: v for k, v in rec.items() if k != "flags"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
