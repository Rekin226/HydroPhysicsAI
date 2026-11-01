"""Model side of X5.gnss_seasonal_column.v1 (round 7 rank 10), frozen and hashed BEFORE any
GNSS value is read, and the G0-OC synthetic check built on it.

    PY=~/miniforge3/envs/hydro/bin/python
    $PY dataprep_staging/sd_column/gsc_model_side.py build --out-dir <dir>   (lead; CPU)
    $PY dataprep_staging/sd_column/gsc_model_side.py check --out-dir <dir>   (byte-for-byte)
    $PY dataprep_staging/sd_column/gsc_model_side.py g0oc --out-dir <dir>    (lead; CPU)

Inputs (no observation): stations.csv of gnss_clean/preorigin/v4 read with its FIRST TEN
columns only (family, station, coordinates, twin_cell, role, note, first/last day, n_days:
metadata; rate_m_yr, annual_amp_m and the noise columns are never opened), the 1 km grid
and county/zone masks, and the SD.cut_response_spread.v1 forwards (main parameter set, free
hindcast 2012-01..2022-12, gain 0, no --data) of every column that passed SD's engineering
checks (SD's report.json must exist: SD's sealed-output policy forbids reading its forwards
before). The page column d290c592 is the primary arm; every other column is a report-only
arm, hashed at the same time. The storage-capped column SC (round 7 rank 17) is omitted.

Station rule (metadata only): twin_cell != -1; role not 'off-fan reference'; window = NGL
2012-01..2019-12 or WRA 2020-01..2022-12 intersected with the station's first/last day,
spanning >= 730 days; the station's coordinates fall in its twin_cell. Units (assign_units):

* NGL stations are assigned BY NAME, as the judge (rank 10) named them, never by the county
  mask: HUWE and DNAN (window clipped to 2014-08..2019-12) are the only NGL stations of the
  primary unit and form the co-primary stratum T1n (the least-seen unit); LUKN, ERLN, FUNY
  and PUS2 form the NGL part of the report-only Changhua arm (ERLN's cell is Yunlin by the
  mask; it was an S3 design unit, so it is neither primary nor 'least-seen'); every other
  on-grid NGL station (JSAN: apex, east of the plan box) is report-only. The mask county is
  recorded beside the name-based unit.
* WRA stations by the twin county mask at the cell: Yunlin stations are primary except ANES
  (seen by a planner; ruling R7.gnss_seasonal_S: report-only); Changhua stations form the
  WRA part of the Changhua arm (CHSG, seen, is listed as such and is report-only anyway).
* A primary candidate whose page-column annual amplitude over its window is below
  A_MIN_MOD_M (1 mm: its observed amplitude would be loading and noise only) or that has
  fewer than 2 model years is report-only.

Every unit window and data scope of the registration is derived from this frozen model side
(unit_scope), never typed by hand.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import gsc_estimator as E  # noqa: E402
import numpy as np  # noqa: E402
import sdlib as L  # noqa: E402

PRODUCT = "results/dataprep/gnss_clean/preorigin/v4"
STATIONS_CSV = f"{PRODUCT}/stations.csv"
META_COLUMNS = ["family", "station", "x_twd97", "y_twd97", "twin_cell", "role", "note",
                "first_day", "last_day", "n_days"]
WINDOWS = {"ngl": ("2012-01-01", "2019-12-31"), "wra": ("2020-01-01", "2022-12-31")}
MIN_SPAN_DAYS = 730
A_MIN_MOD_M = 0.001
SEEN_REPORT_ONLY = {"ANES": "seen by a planner (looks.jsonl 2026-11-01T11:14Z); ruling "
                            "R7.gnss_seasonal_S: report-only",
                    "CHSG": "seen by a planner (same look); ruling R7.gnss_seasonal_S: "
                            "report-only"}
REFERENCE_ROLE = "off-fan reference"
#: NGL stations named by the judge (rank 10): the co-primary stratum and the Changhua arm
NGL_STRATUM = ("DNAN", "HUWE")
NGL_STRATUM_WINDOW = ("2014-08-01", "2019-12-31")
NGL_CHANGHUA_ARM = ("ERLN", "FUNY", "LUKN", "PUS2")
NGL_OTHER_REASON = "NGL station not named in the judge's design (rank 10): report-only"
OUT_DIR = f"results/dataprep/reports/{L.GSC_ID}/model_side"
SD_REPORT = f"results/dataprep/reports/{L.SD_ID}/report.json"
PRIMARY_ARM = L.PAGE_COLUMN


def read_station_metadata(path: Path):
    """The first ten columns of stations.csv, checked by name (nothing else is parsed)."""
    import pandas as pd

    d = pd.read_csv(path, usecols=list(range(10)))
    if list(d.columns) != META_COLUMNS:
        raise ValueError(f"stations.csv first ten columns are {list(d.columns)}")
    return d


def months_between(first: str, last: str) -> np.ndarray:
    y0, m0 = int(first[:4]), int(first[5:7])
    y1, m1 = int(last[:4]), int(last[5:7])
    return np.arange(12 * y0 + m0 - 1, 12 * y1 + m1)


def station_rule(meta, grid, ch: np.ndarray, zone: np.ndarray) -> dict:
    """{station: record} for every on-grid, non-reference station with its window, county,
    zone and the rule's decision before the model amplitude (primary_candidate)."""
    import pandas as pd

    out = {}
    for _, r in meta.iterrows():
        rec = {"family": str(r.family), "twin_cell": int(r.twin_cell), "role": str(r.role)}
        if rec["twin_cell"] < 0 or rec["role"] == REFERENCE_ROLE:
            continue
        if rec["twin_cell"] >= len(ch):
            raise ValueError(f"{r.station}: twin_cell {rec['twin_cell']} is off the grid")
        w0, w1 = WINDOWS[rec["family"]]
        if rec["family"] == "ngl" and str(r.station) in NGL_STRATUM:
            w0, w1 = NGL_STRATUM_WINDOW
        a = max(pd.Timestamp(r.first_day), pd.Timestamp(w0))
        b = min(pd.Timestamp(r.last_day), pd.Timestamp(w1))
        span = int((b - a).days) + 1
        cell = grid.active_index(float(r.x_twd97), float(r.y_twd97))
        rec.update(window=[a.strftime("%Y-%m-%d"), b.strftime("%Y-%m-%d")], span_days=span,
                   cell_check=cell == rec["twin_cell"],
                   county="changhua" if bool(ch[rec["twin_cell"]]) else "yunlin",
                   zone=("proximal", "mid", "distal")[int(zone[rec["twin_cell"]])])
        ok = span >= MIN_SPAN_DAYS and rec["cell_check"]
        rec["eligible"] = ok
        rec["report_only_reason"] = (SEEN_REPORT_ONLY.get(str(r.station))
                                     or (None if ok else "window < 730 days or cell mismatch"))
        out[str(r.station)] = rec
    return out


def assign_units(st: dict, page: dict) -> dict:
    """Unit of every station record of station_rule (in place: unit, ngl_stratum,
    unit_basis, report_only_reason, page_amplitude_m) from the name lists, the county mask
    and the page arm's full-window amplitudes ``page`` ({station: {amplitude_m, n_years}}).
    Returns {unit: sorted station names}."""
    for s, rec in st.items():
        a = page.get(s, {}).get("amplitude_m")
        rec["page_amplitude_m"] = a
        big = (a is not None and a >= A_MIN_MOD_M
               and page.get(s, {}).get("n_years", 0) >= E.MIN_YEARS)
        if rec["family"] == "ngl":
            rec["unit_basis"] = "named (judge rank 10); mask county " + rec["county"]
            if s in NGL_STRATUM:
                cand, unit_else = True, "report_only"
            elif s in NGL_CHANGHUA_ARM:
                cand, unit_else = False, "changhua_arm"
            else:
                cand, unit_else = False, "report_only"
                rec["report_only_reason"] = rec["report_only_reason"] or NGL_OTHER_REASON
        else:
            rec["unit_basis"] = "county mask at the twin cell"
            cand = rec["county"] == "yunlin" and s not in SEEN_REPORT_ONLY
            unit_else = "changhua_arm" if rec["county"] == "changhua" else "report_only"
        prim = cand and rec["eligible"] and big
        rec["unit"] = "primary" if prim else unit_else
        if cand and not prim and rec["report_only_reason"] is None:
            rec["report_only_reason"] = "page amplitude below 1 mm or < 2 model years"
        rec["ngl_stratum"] = prim and s in NGL_STRATUM
    units = {"primary": sorted(s for s, r in st.items() if r["unit"] == "primary"),
             "ngl_stratum": sorted(s for s, r in st.items() if r["ngl_stratum"]),
             "changhua_arm": sorted(s for s, r in st.items() if r["unit"] == "changhua_arm"),
             "report_only": sorted(s for s, r in st.items() if r["unit"] == "report_only")}
    stray = set(units["ngl_stratum"]) - set(NGL_STRATUM)
    stray |= {s for s in units["primary"] if st[s]["family"] == "ngl"} - set(NGL_STRATUM)
    if stray:
        raise ValueError(f"NGL stations outside the named stratum in the primary: {stray}")
    return units


def unit_scope(st: dict, names) -> dict:
    """{'window': 'YYYY-MM..YYYY-MM', 'data_scope': 'gnss:...'} spanning the stations'
    windows (None when the unit is empty)."""
    if not names:
        return {"window": None, "data_scope": None}
    w0 = min(st[s]["window"][0] for s in names)[:7]
    w1 = max(st[s]["window"][1] for s in names)[:7]
    w = f"{w0}..{w1}"
    return {"window": w, "data_scope": f"gnss:{w}"}


def g0_strata(doc: dict) -> dict:
    """The strata G0-OC characterises beside the primary unit (T1n)."""
    return {"T1n": list(doc["units"]["ngl_stratum"])}


def model_series(subs: np.ndarray, cell: int, window) -> tuple[np.ndarray, np.ndarray]:
    """(month index m, monthly subsidence) of a column's free hindcast at a cell over the
    station window (baseline scenario; months <= 2022-12)."""
    m = months_between(*window)
    t = m - (12 * L.DATE0[0] + L.DATE0[1] - 1)
    if t.min() < 0 or t.max() > L.ORIGIN:
        raise ValueError(f"window {window} leaves the hindcast")
    return m, np.asarray(subs[0, cell, t], dtype="float64")


def build(repo: Path) -> tuple[dict, dict]:
    """(model_side document, {key: array} for model_series.npz). Deterministic."""
    rep = json.loads((repo / SD_REPORT).read_text())
    passed = sorted(c for c, v in rep["columns_checks"].items() if v.get("pass"))
    if PRIMARY_ARM not in passed:
        raise SystemExit("the page column did not pass SD's engineering checks")
    gm = L.grid_and_masks(repo)
    st = station_rule(read_station_metadata(repo / STATIONS_CSV), gm["grid"],
                      gm["changhua"], gm["zone"])
    att = {}
    for s in (repo / L.LOG / "attempts.jsonl").read_text().splitlines():
        if s.strip():
            r = json.loads(s)
            if r.get("event") == "forward" and r.get("rc") == 0:
                att[r["fwd"]] = r
    series, arms = {}, {}
    for c in passed:
        k = L.forward_of(c)
        p = repo / f"{L.out_stem(k)}.npz"
        if L.sha(p) != att[k]["npz_sha256"] or L.sha(p) != rep["forwards"][k]["npz_sha256"]:
            raise SystemExit(f"{p} is not the forward SD scored")
        z = L.load_npz(p)
        sub = L.column_subs(z, L.FORWARDS[k]["columns"].index(c))
        amps = {}
        for s, rec in st.items():
            m, y = model_series(sub, rec["twin_cell"], rec["window"])
            series[f"{c}|{s}|m"] = m.astype("int64")
            series[f"{c}|{s}|y"] = y
            co = E.year_coefficients(m, y)
            amps[s] = {"amplitude_m": E.amplitude(co) if co else None,
                       "phase_month": E.phase_months(co) if co else None,
                       "n_years": len(co)}
        arms[c] = {"tier": L.COLUMNS[c][2], "forward": k, "npz": str(p.relative_to(repo)),
                   "npz_sha256": L.sha(p), "column_sha256": L.sha(repo / L.column_path(c)),
                   "role": "primary" if c == PRIMARY_ARM else "report-only arm",
                   "full_window": amps}
    units = assign_units(st, arms[PRIMARY_ARM]["full_window"])
    doc = {"id": L.GSC_ID, "what": "model side, frozen before any GNSS read",
           "stations_csv": STATIONS_CSV, "stations_csv_sha256": L.sha(repo / STATIONS_CSV),
           "stations_csv_columns_read": META_COLUMNS,
           "sd_report": SD_REPORT, "sd_report_sha256": L.sha(repo / SD_REPORT),
           "f_subs_page": rep.get("f_subs_page"),
           "rule": {"windows": WINDOWS, "min_span_days": MIN_SPAN_DAYS,
                    "a_min_mod_m": A_MIN_MOD_M, "seen_report_only": SEEN_REPORT_ONLY,
                    "ngl_stratum": NGL_STRATUM, "ngl_stratum_window": NGL_STRATUM_WINDOW,
                    "ngl_changhua_arm": NGL_CHANGHUA_ARM, "ngl_other": NGL_OTHER_REASON,
                    "wra_units": "county mask at the twin cell",
                    "estimator": "gsc_estimator.year_coefficients / amplitude"},
           "stations": st, "arms": arms, "primary_arm": PRIMARY_ARM,
           "omitted_arms": {"SC": "round 7 rank 17 not finished at freeze; omitted"},
           "units": units,
           "unit_scopes": {k: unit_scope(st, v) for k, v in units.items()}}
    return doc, series


def series_bytes(series: dict) -> bytes:
    """Deterministic serialisation of the series (sorted keys, raw little-endian bytes)."""
    import hashlib

    h = hashlib.sha256()
    for k in sorted(series):
        a = np.ascontiguousarray(series[k])
        h.update(k.encode())
        h.update(str(a.dtype).encode())
        h.update(a.astype(a.dtype.newbyteorder("<")).tobytes())
    return h.hexdigest().encode()


def write(out_dir: Path, doc: dict, series: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = dict(doc, series_digest=series_bytes(series).decode())
    text = json.dumps(doc, indent=1, ensure_ascii=False, sort_keys=True, default=float) + "\n"
    with open(out_dir / "model_side.json", "x", encoding="utf-8") as fh:
        fh.write(text)
    with open(out_dir / "model_series.npz", "xb") as fh:
        np.savez(fh, **series)
    return {"model_side_json_sha256": L.sha(out_dir / "model_side.json"),
            "series_digest": doc["series_digest"]}


def load(out_dir: Path) -> tuple[dict, dict]:
    doc = json.loads((out_dir / "model_side.json").read_text())
    z = np.load(out_dir / "model_series.npz", allow_pickle=False)
    series = {k: z[k] for k in z.files}
    if series_bytes(series).decode() != doc["series_digest"]:
        raise ValueError("model_series.npz does not match model_side.json's digest")
    return doc, series


def g0_stations(doc: dict, series: dict) -> dict:
    """The primary unit's page-arm series, as gsc_synth.run expects."""
    return {s: {"family": doc["stations"][s]["family"],
                "m": series[f"{PRIMARY_ARM}|{s}|m"], "mod": series[f"{PRIMARY_ARM}|{s}|y"]}
            for s in doc["units"]["primary"]}


def main(argv=None) -> int:
    from common.paths import REPO

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("build", "check", "g0oc"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--n-rep", type=int, default=None)
    a = ap.parse_args(argv)
    out = Path(a.out_dir) if a.out_dir else REPO / OUT_DIR
    if a.cmd == "build":
        doc, series = build(REPO)
        print(json.dumps(write(out, doc, series)))
        return 0
    if a.cmd == "check":
        load(out)                                      # the series match the digest
        doc1, s1 = build(REPO)
        text = json.dumps(dict(doc1, series_digest=series_bytes(s1).decode()), indent=1,
                          ensure_ascii=False, sort_keys=True, default=float) + "\n"
        same = text.encode() == (out / "model_side.json").read_bytes()
        print(f"G-eng: model side reproduces byte for byte: {same}")
        return 0 if same else 3
    import gsc_synth as G

    doc, series = load(out)
    res = G.run(g0_stations(doc, series), n_rep=a.n_rep or G.N_REP, strata=g0_strata(doc))
    res["model_side_json_sha256"] = L.sha(out / "model_side.json")
    p = out.parent / "g0oc.json"
    with open(p, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, sort_keys=True, default=float)
        fh.write("\n")
    print(f"G0-OC pass (scored rule, both scales): {res['pass']}; marginal "
          f"{json.dumps(res['marginal'], sort_keys=True)}; wrote {p} (sha256 {L.sha(p)})")
    return 0 if res["pass"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
