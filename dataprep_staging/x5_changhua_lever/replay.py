"""Replayed pre-origin county pumping anomalies for R1c's recorded-pumping mechanism, and the
constant county cuts that give its steady-state lever per county.

Shared by X5.changhua_lever_model.v1 (rank 8: F, the gain of N2C's estimator) and
X5.n2c_reading.v1 (rank 10: the synthetic worlds' contrast). Model-only design inputs: the
outputs are multipliers on R1c's frozen climatology pumping; no head, leveling or pile value
is read, and no X1 (post-origin) electricity.

Source: ``model_ready/release_20261012b2/preorigin/electricity_census/v1``
(``county_class_monthly.parquet``), opened through ``common.guard`` for the use
``design_time_evidence`` (allowed by its MANIFEST; flags 0-6 kept by the guard, so flag-9
aggregates drop out), rows ``scope == "in_grid"``, ``class == "all"``, Changhua and Yunlin,
dated 2022-12 or earlier. The product's MANIFEST forbids ``forcing_recorded_arm``: the replay
uses twin.forward's recorded-pumping MECHANISM as a design device (synthetic future months,
never scored against an observation, never a recorded arm of any registered test); this is
disclosed in both declarations.

Ratio (N2C's frozen rule, per county, on this source): r_c(s) = value_c(s) / mean of value_c
over the 2012-2022 months with the same month of year. A source month without a kept row
takes r = 1 (N2's 'unrecorded' rule). replays.json records, per county, how many source
months were filled that way (over 2012-01..2022-12 and inside each replay's source block), so
a reader can see that the replays carry real anomalies rather than filled ones.

Replays: seven contiguous, calendar-aligned 58-month blocks. Replay ``rpY`` (Y = 2012..2018)
maps target month 2023-01 + j onto source month Y-01 + j (j = 0..57), so the N2C window
2026-11..2027-10 replays (Y+3)-11..(Y+4)-10. Written as N2C's per-cell field (build_r_both's
layout): r_CH on the 943 Changhua cells of zone_county_map/v2, r_YL on every other cell,
months 2023-01..2027-10, start 2023-01.

Constant cuts: ``ch30.csv`` and ``yl30.csv``, multiplier 0.7 from 2026-01 to 2032-12, applied
with the forward's county scope (``#changhua`` / ``#yunlin``).

Usage (repository root): python -I dataprep_staging/x5_changhua_lever/replay.py build --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

PRODUCT = "results/dataprep/model_ready/release_20261012b2/preorigin/electricity_census/v1"
TABLE = "county_class_monthly.parquet"
USE = "design_time_evidence"
SLICE_END = "2022-12"
CLIM_YEARS = (2012, 2022)
COUNTIES = ("changhua", "yunlin")
STARTS = (2012, 2013, 2014, 2015, 2016, 2017, 2018)
FIRST = "2023-01"
N_MONTHS = 58                                  # 2023-01 .. 2027-10
WINDOW = ("2026-11", "2027-10")
COUNTY_MAP = "results/dataprep/zone_county_map/v2/zone_county_map.npz"
R1C_COUNTY_SHA = "6ca7840401a5f77ace183acc605f38c936b79de94df54bec8b1134bbfd86529e"
CUT = 0.7
CUT_FIRST, CUT_LAST = "2026-01", "2032-12"


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def months(first: str, n: int) -> list[str]:
    return [str(p) for p in pd.period_range(first, periods=n, freq="M")]


TARGET = months(FIRST, N_MONTHS)


def county_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Month x county ratio to the 2012-2022 month-of-year mean (rows already flag-filtered
    by the guard), every month 2012-01..2022-12; a month without a kept row takes 1."""
    return raw_ratios(df).fillna(1.0)


def raw_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """:func:`county_ratios` before the fill (NaN where the source has no kept row)."""
    d = df[(df["scope"] == "in_grid") & (df["class"] == "all")
           & df["county"].isin(COUNTIES)].copy()
    d["p"] = pd.PeriodIndex(d["date"].astype(str), freq="M")
    d = d[d["p"] <= pd.Period(SLICE_END, freq="M")]
    if d.duplicated(["county", "p"]).any():
        raise ValueError("duplicate county-month rows")
    if not np.isfinite(d["value"].to_numpy(float)).all() or (d["value"] < 0).any():
        raise ValueError("non-finite or negative value")
    out = {}
    for c in COUNTIES:
        s = d[d["county"] == c].set_index("p")["value"].sort_index()
        ref = s[(s.index.year >= CLIM_YEARS[0]) & (s.index.year <= CLIM_YEARS[1])]
        clim = ref.groupby(ref.index.month).mean()
        if len(clim) != 12 or (clim <= 0).any():
            raise ValueError(f"{c}: the 2012-2022 climatology lacks a month or is not > 0")
        out[c] = s / s.index.month.map(clim).to_numpy()
    full = pd.period_range(f"{CLIM_YEARS[0]}-01", SLICE_END, freq="M")
    r = pd.DataFrame({c: out[c].reindex(full) for c in COUNTIES})
    r.index = r.index.astype(str)
    return r


def filled_months(raw: pd.DataFrame, src: list[str] | None = None) -> dict:
    """Per county: the source months (all, or those in ``src``) that take r = 1 by the fill."""
    sub = raw if src is None else raw.loc[src]
    out = {}
    for c in COUNTIES:
        m = [str(x) for x in sub.index[sub[c].isna()]]
        out[c] = {"n_filled": len(m), "n_months": int(len(sub)), "months": m}
    return out


def replay(ratios: pd.DataFrame, start_year: int) -> pd.DataFrame:
    """Target months 2023-01..2027-10 with the replayed source month and both ratios."""
    src = months(f"{start_year}-01", N_MONTHS)
    if src[-1] > SLICE_END or src[0] < f"{CLIM_YEARS[0]}-01":
        raise ValueError(f"replay {start_year}: source {src[0]}..{src[-1]} outside "
                         f"{CLIM_YEARS[0]}-01..{SLICE_END}")
    sub = ratios.loc[src]
    return pd.DataFrame({"month": TARGET, "source": src,
                         "r_changhua": sub["changhua"].to_numpy(float),
                         "r_yunlin": sub["yunlin"].to_numpy(float)})


def per_cell(rep: pd.DataFrame, changhua_flags: np.ndarray) -> dict:
    """The forward's per-cell npz arrays (build_r_both's layout)."""
    f = np.where(changhua_flags[None, :], rep["r_changhua"].to_numpy()[:, None],
                 rep["r_yunlin"].to_numpy()[:, None]).astype("float64")
    if not np.isfinite(f).all() or (f < 0).any():
        raise ValueError("non-finite or negative multiplier")
    return {"month": np.array(rep["month"].tolist()), "multiplier": f,
            "n_active": np.int64(changhua_flags.size), "start": np.str_(FIRST)}


def window_anomalies(rep: pd.DataFrame) -> dict:
    w = rep[(rep["month"] >= WINDOW[0]) & (rep["month"] <= WINDOW[1])]
    return {"months": w["month"].tolist(), "source": w["source"].tolist(),
            "a_changhua": (w["r_changhua"] - 1.0).tolist(),
            "a_yunlin": (w["r_yunlin"] - 1.0).tolist()}


def cut_csv() -> str:
    m = months(CUT_FIRST, (2032 - 2026 + 1) * 12)
    assert m[-1] == CUT_LAST
    return "month,multiplier\n" + "".join(f"{x},{CUT}\n" for x in m)


def load_flags(path: Path) -> tuple[np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as z:
        flags = np.asarray(z["changhua"]).astype(bool)
        n_active = int(z["n_active"])
        stored = str(z["sha256"])
    if stored != R1C_COUNTY_SHA or flags.shape != (n_active,):
        raise ValueError(f"{path}: not the county map R1c was calibrated on")
    return flags, {"path": COUNTY_MAP, "file_sha256": sha256(path), "flags_sha256": stored,
                   "n_active": n_active, "n_changhua": int(flags.sum())}


def read_source(product_dir: Path) -> tuple[pd.DataFrame, dict]:
    sys.path.insert(0, str(REPO / "dataprep_staging"))
    from common import guard

    p = guard.open_product(product_dir, USE)
    df = p.read_table(TABLE, columns=["county", "scope", "class", "date", "value"])
    rec = {"product": PRODUCT, "use": USE, "table": TABLE,
           "manifest_sha256": sha256(Path(product_dir) / "MANIFEST.json"),
           "table_sha256": p.outputs[TABLE]["sha256"], "guard_notes": p.notes,
           "rows_kept_by_guard": int(len(df))}
    return df, rec


def build(out: Path, product_dir: Path, county_map: Path) -> dict:
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty; replays are written once")
    out.mkdir(parents=True, exist_ok=True)
    flags, cmap = load_flags(county_map)
    df, src = read_source(product_dir)
    raw = raw_ratios(df)
    ratios = raw.fillna(1.0)
    rec = {"builder": "dataprep_staging/x5_changhua_lever/replay.py",
           "builder_sha256": sha256(Path(__file__)), "source": src, "county_map": cmap,
           "rule": "r_c(s) = in_grid all-class value / its 2012-2022 month-of-year mean; "
                   "missing source month -> 1", "first": FIRST, "n_months": N_MONTHS,
           "window": list(WINDOW), "filled_with_r1": filled_months(raw),
           "replays": {}, "cuts": {}}
    for y in STARTS:
        rep = replay(ratios, y)
        p = out / f"rp{y}.npz"
        np.savez(p, **per_cell(rep, flags))
        rec["replays"][f"rp{y}"] = {"file": p.name, "sha256": sha256(p),
                                    "source_first": rep["source"].iloc[0],
                                    "source_last": rep["source"].iloc[-1],
                                    "filled_with_r1": {
                                        c: {k: v for k, v in f.items() if k != "months"}
                                        for c, f in filled_months(
                                            raw, rep["source"].tolist()).items()},
                                    "window": window_anomalies(rep)}
    for name in ("ch30", "yl30"):
        p = out / f"{name}.csv"
        p.write_text(cut_csv(), encoding="utf-8")
        rec["cuts"][name] = {"file": p.name, "sha256": sha256(p), "multiplier": CUT,
                             "first": CUT_FIRST, "last": CUT_LAST,
                             "scope": "changhua" if name == "ch30" else "yunlin"}
    (out / "replays.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--product", default=PRODUCT)
    b.add_argument("--county-map", default=COUNTY_MAP)
    a = ap.parse_args(argv)

    def rp(s: str) -> Path:
        return Path(s) if Path(s).is_absolute() else REPO / s

    rec = build(rp(a.out), rp(a.product), rp(a.county_map))
    print(json.dumps({k: (v if k != "replays" else {r: x["sha256"] for r, x in v.items()})
                      for k, v in rec.items() if k not in ("source",)}, indent=1))
    print("replays.json sha256", sha256(rp(a.out) / "replays.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
