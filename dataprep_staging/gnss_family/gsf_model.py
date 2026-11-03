"""Model side of X5.gnss_seasonal_family.v1: the frozen v1 model side (read, never rebuilt),
SD v2's cut30 values, the deduplicated member list and one model-only supplement (every arm's
free-hindcast subsidence at the TKJS(1)/NTUH(1) deep-pile cells over 2016-06..2019-12, from
the same sealed SD forwards v1 read). No observation is read here: the only product file
opened is deep_benchmarks/v2 mapping.csv, columns gsf_const.PILE_MAPPING_COLUMNS (metadata).

    PY=~/miniforge3/envs/hydro/bin/python
    $PY dataprep_staging/gnss_family/gsf_model.py supplement     (lead; CPU; writes once)
    $PY dataprep_staging/gnss_family/gsf_model.py engineering    (lead; writes once)
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(HERE), str(REPO / "dataprep_staging" / "sd_column"),
           str(REPO / "dataprep_staging")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gsf_const as K  # noqa: E402
import gsf_estimator as F  # noqa: E402
import numpy as np  # noqa: E402

#: bound code: this package plus the sd_column modules it imports (read-only)
CODE = (f"{K.X}/gsf_const.py", f"{K.X}/gsf_estimator.py", f"{K.X}/gsf_synth.py",
        f"{K.X}/gsf_model.py", f"{K.X}/gsf_gates.py", f"{K.X}/score_gsf.py",
        f"{K.X}/run_gsf.sh", f"{K.SDC}/gsc_estimator.py", f"{K.SDC}/gsc_model_side.py",
        f"{K.SDC}/sdlib.py", f"{K.SDC}/score_gsc.py")


def rp(s) -> Path:
    return Path(s) if Path(s).is_absolute() else REPO / s


def sha256(p) -> str:
    p = rp(p)
    if not p.exists():
        return "missing"
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def code_sha256() -> dict:
    return {p: sha256(p) for p in CODE}


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log_attempt(rec: dict, path: Path | None = None) -> None:
    p = path or rp(K.ATTEMPTS)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(rec, time=utc()), sort_keys=True) + "\n")


# --------------------------------------------------------------------------- v1 model side
def load_v1(model_dir: Path | None = None) -> tuple[dict, dict]:
    """(model_side doc, series) of the frozen v1 model side, sha256 and digest checked."""
    import gsc_model_side as M

    d = rp(model_dir or K.V1_MODEL_DIR)
    if sha256(d / "model_side.json") != K.V1_MODEL_SHA:
        raise SystemExit("ENGINEERING: v1 model_side.json is not 65b927cb")
    return M.load(d)


def primary_stations(ms: dict, series: dict, arms) -> dict:
    """{station: {zone, family, m, mod: {arm: y}}} of v1's primary unit (distal and mid)."""
    out = {}
    for s in ms["units"]["primary"]:
        rec = ms["stations"][s]
        if rec["zone"] not in K.ZONES:
            raise SystemExit(f"ENGINEERING: primary station {s} is in zone {rec['zone']}")
        m = np.asarray(series[f"{K.PAGE}|{s}|m"])
        mod = {}
        for a in arms:
            if not np.array_equal(np.asarray(series[f"{a}|{s}|m"]), m):
                raise SystemExit(f"ENGINEERING: {a}|{s} months differ from the page arm's")
            mod[a] = np.asarray(series[f"{a}|{s}|y"], float)
        out[s] = {"zone": rec["zone"], "family": rec["family"], "m": m, "mod": mod}
    return out


def cut30(sd_report: dict, arms) -> dict:
    out = {}
    for a in arms:
        out[a] = float(sd_report["per_column"][a]["summary"][K.CUT30_HORIZON][K.CUT30_GROUP]
                       [K.CUT30_FIELD])
    return out


def load_sd_report() -> dict:
    if sha256(K.SD_REPORT) != K.SD_REPORT_SHA:
        raise SystemExit("ENGINEERING: SD v2 report.json is not 5cfc9ea7")
    return json.loads(rp(K.SD_REPORT).read_text())


# --------------------------------------------------------------------------- pile supplement
def pile_mapping() -> dict:
    """{sensor: {grid_cell, depth_m, first_month, last_month}} for gsf_const.PILES, from the
    declared mapping.csv metadata columns only."""
    import pandas as pd

    d = pd.read_csv(rp(K.PILE_MAPPING), usecols=list(K.PILE_MAPPING_COLUMNS))
    out = {}
    for s in K.PILES:
        r = d[(d["sensor_id"] == s) & d["on_grid"].astype(str).str.lower().eq("true")]
        if len(r) != 1:
            raise SystemExit(f"ENGINEERING: {s} is not one on-grid mapping row")
        r = r.iloc[0]
        if float(r["depth_m"]) != K.PILES[s]["depth_m"]:
            raise SystemExit(f"ENGINEERING: {s} depth {r['depth_m']} != declared")
        out[s] = {"grid_cell": int(r["grid_cell"]), "depth_m": float(r["depth_m"]),
                  "first_month": str(r["first_month"]), "last_month": str(r["last_month"])}
    return out


def cell_check(pm: dict, ms: dict) -> dict:
    """Pile and GNSS cells against the declared ones (TKJS 541/541; NTUH pile 769, GNSS 771:
    the NTUH pair sits on two different cells, disclosed)."""
    out = {}
    for s, rec in K.PILES.items():
        g = int(ms["stations"][rec["gnss"]]["twin_cell"])
        out[s] = {"pile_cell": pm[s]["grid_cell"], "gnss_station": rec["gnss"], "gnss_cell": g,
                  "as_declared": pm[s]["grid_cell"] == rec["pile_cell"]
                  and g == rec["gnss_cell"], "same_cell": pm[s]["grid_cell"] == g}
    return out


def build_supplement(ms: dict) -> tuple[dict, dict]:
    """(doc, series): every v1 arm's baseline free-hindcast subsidence at the pile cells over
    PILE_WINDOW, from the forward npz v1 bound (sha256 checked). Deterministic."""
    import gsc_model_side as M
    import sdlib as L

    pm = pile_mapping()
    w = (f"{K.PILE_WINDOW[0]}-01", f"{K.PILE_WINDOW[1]}-28")
    series, forwards = {}, {}
    for a in sorted(ms["arms"]):
        arm = ms["arms"][a]
        p = rp(arm["npz"])
        if sha256(p) != arm["npz_sha256"]:
            raise SystemExit(f"ENGINEERING: {p} is not the forward v1 bound")
        k = arm["forward"]
        if k not in forwards:
            forwards[k] = L.load_npz(p)
        sub = L.column_subs(forwards[k], L.FORWARDS[k]["columns"].index(a))
        for s, rec in pm.items():
            m, y = M.model_series(sub, rec["grid_cell"], w)
            series[f"{a}|{s}|m"] = m.astype("int64")
            series[f"{a}|{s}|y"] = y
    doc = {"id": K.ID, "what": "pile-cell model supplement (model-only)",
           "v1_model_side_sha256": K.V1_MODEL_SHA, "piles": pm, "window": list(K.PILE_WINDOW),
           "arms": sorted(ms["arms"]), "pile_mapping_sha256": sha256(K.PILE_MAPPING),
           "pile_mapping_columns_read": list(K.PILE_MAPPING_COLUMNS),
           "forwards_sha256": {a: ms["arms"][a]["npz_sha256"] for a in sorted(ms["arms"])},
           "cells": cell_check(pm, ms)}
    return doc, series


def write_supplement(doc: dict, series: dict) -> dict:
    import gsc_model_side as M

    j, z = rp(K.PILE_MODEL_JSON), rp(K.PILE_MODEL_NPZ)
    j.parent.mkdir(parents=True, exist_ok=True)
    doc = dict(doc, series_digest=M.series_bytes(series).decode())
    with z.open("xb") as fh:
        np.savez(fh, **series)
    with j.open("x", encoding="utf-8") as fh:
        fh.write(json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    return {"pile_model_json_sha256": sha256(j), "series_digest": doc["series_digest"]}


def load_supplement() -> tuple[dict, dict]:
    import gsc_model_side as M

    doc = json.loads(rp(K.PILE_MODEL_JSON).read_text())
    z = np.load(rp(K.PILE_MODEL_NPZ), allow_pickle=False)
    series = {k: z[k] for k in z.files}
    if M.series_bytes(series).decode() != doc["series_digest"]:
        raise SystemExit("ENGINEERING: pile_model.npz does not match its digest")
    return doc, series


# --------------------------------------------------------------------------- engineering
def dedup_check(stations: dict, pile_series: dict | None) -> dict:
    """Every DEDUP pair byte-identical at every primary station (and pile cell)."""
    out = {}
    for dup, keep in K.DEDUP.items():
        same = all(np.array_equal(st["mod"][dup], st["mod"][keep]) for st in stations.values())
        if pile_series is not None:
            same = same and all(np.array_equal(pile_series[f"{dup}|{s}|y"],
                                               pile_series[f"{keep}|{s}|y"]) for s in K.PILES)
        out[dup] = {"kept": keep, "identical": bool(same)}
    return out


def engineering_record() -> dict:
    """The engineering record G-sep requires (model-only)."""
    ms, series = load_v1()
    sd = load_sd_report()
    arms = sorted(ms["arms"])
    missing = sorted((set(K.TIER_AB) | set(K.CH_TIER)) - set(arms))
    if missing:
        raise SystemExit(f"ENGINEERING: arms missing from the v1 model side: {missing}")
    if sorted(sd["tiers"]["A+B"]) != sorted(K.TIER_AB) or sorted(sd["tiers"]["CH"]) != \
            sorted(K.CH_TIER):
        raise SystemExit("ENGINEERING: SD v2 tiers differ from the declared members")
    st = primary_stations(ms, series, arms)
    sup_doc, sup = load_supplement()
    doc2, s2 = build_supplement(ms)
    import gsc_model_side as M

    rebuild = M.series_bytes(s2).decode() == sup_doc["series_digest"]
    dd = dedup_check(st, sup)
    c30 = cut30(sd, arms)
    cf = F.cluster_far(c30, list(K.SCREEN))
    zones = {z: sorted(s for s, v in st.items() if v["zone"] == z) for z in K.ZONES}
    ok = (rebuild and all(v["identical"] for v in dd.values())
          and all(v["as_declared"] for v in sup_doc["cells"].values())
          and len(zones["distal"]) >= 3
          and len(zones["mid"]) >= 3 and bool(cf["cluster"]) and bool(cf["far"]))
    return {"id": K.ID, "what": "engineering record (model-only): G-sep refuses without it",
            "pass": bool(ok), "v1_model_side_sha256": K.V1_MODEL_SHA,
            "v1_series_digest": ms["series_digest"], "sd_report_sha256": K.SD_REPORT_SHA,
            "pile_model_json_sha256": sha256(K.PILE_MODEL_JSON),
            "pile_series_digest": sup_doc["series_digest"], "pile_rebuild_identical": rebuild,
            "dedup": dd, "screen": list(K.SCREEN), "report_only_arms": list(K.REPORT_ONLY_ARMS),
            "cut30_cm": c30, "cluster_far": cf, "far_pairs": F.far_pairs(c30, list(K.SCREEN)),
            "zones": zones, "piles": sup_doc["piles"], "code_sha256": code_sha256()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("supplement", "engineering"))
    a = ap.parse_args(argv)
    try:
        return _main(a)
    except BaseException as e:                          # noqa: BLE001 - logged, re-raised
        log_attempt({"event": f"{a.cmd}_failed", "error": f"{type(e).__name__}: {e}"})
        raise


def _main(a) -> int:
    log_attempt({"event": f"{a.cmd}_start"})
    if a.cmd == "supplement":
        if rp(K.PILE_MODEL_JSON).exists():
            print(f"REFUSED: {K.PILE_MODEL_JSON} exists (written once)")
            return 1
        ms, _ = load_v1()
        doc, series = build_supplement(ms)
        r = write_supplement(doc, series)
        log_attempt({"event": "supplement", **r})
        print(json.dumps(r))
        return 0
    if rp(K.ENGINEERING).exists():
        print(f"REFUSED: {K.ENGINEERING} exists (written once)")
        return 1
    rec = engineering_record()
    with rp(K.ENGINEERING).open("x", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    log_attempt({"event": "engineering", "pass": rec["pass"],
                 "engineering_sha256": sha256(K.ENGINEERING)})
    print(f"engineering pass {rec['pass']}; {K.ENGINEERING} sha256 {sha256(K.ENGINEERING)}")
    return 0 if rec["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
