"""Scorer of X5.gnss_seasonal_column.v1 (round 7 rank 10): the GNSS seasonal hold-out of the
column's elastic response. Run ONCE by the lead, after registration; it is the only code
that opens GNSS daily values.

    PY=~/miniforge3/envs/hydro/bin/python
    $PY dataprep_staging/sd_column/score_gsc.py verify-registration
    $PY dataprep_staging/sd_column/score_gsc.py score --out <report.json>

Order inside ``score`` (each step must pass before the next; nothing GNSS-derived is read
before step 4):

1. the registration is intact (INDEX, ledger), the frozen code and every registered input
   hash as registered, the model side matches its digest and the registered G0-OC passed;
2. the product MANIFEST grants design_time_evidence (gate-exempt; guard.open_product);
3. prereg3.assert_can_score logs the scoring looks of T1 and T1n and refuses if any look
   blocks; a report-only look covers the arms' extra station-windows;
4. daily heights of the declared station windows, flags 0-1 only, minus the declared common
   mode (ngl_offfan primary), monthly medians of >= 10 days, negated to subsidence; the
   estimator on the observed months, obs and model together; report.json (mode 'x').
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
import gsc_model_side as M  # noqa: E402
import numpy as np  # noqa: E402
import sdlib as L  # noqa: E402

PREREG_ID = L.GSC_ID
WP = "X5"
T1 = "X5.gsc.T1_yunlin_pooled"
T1N = "X5.gsc.T1n_ngl_stratum"
UNIT_PRIMARY = "x5gsc.gnss.yunlin_primary"
UNIT_NGL = "x5gsc.gnss.yunlin_ngl_huwe_dnan"
UNIT_REPORT = "x5gsc.gnss.report_only_arms"
FROZEN = (f"{L.X}/sdlib.py", f"{L.X}/gsc_estimator.py", f"{L.X}/gsc_synth.py",
          f"{L.X}/gsc_model_side.py", f"{L.X}/score_gsc.py", f"{L.X}/run_gsc.sh")
REPORT = f"results/dataprep/reports/{PREREG_ID}/report.json"
USE = "design_time_evidence"
FLAGS = (0, 1)
MIN_DAYS = 10
REMOVAL = "ngl_offfan"
CHANGHUA_ARMS = ("page", "ccz", "q0c_csk1", "x5r1_cs")


class Engineering(Exception):
    def __init__(self, code: int, msg: str):
        super().__init__(msg)
        self.code = code


def station_ids(names) -> list[str]:
    """Member ids of GNSS stations in BOTH forms the registry already uses: the bare station
    name (e.g. the ANES/CHSG disclosure) and 'gnss:<station>' (earlier GNSS looks), so a
    membership intersection with any prior or later look sees them."""
    return sorted({x for s in names for x in (str(s), f"gnss:{s}")})


# --------------------------------------------------------------------------- observed side
def obs_monthly(daily, cm, stations: dict, removal: str | None = REMOVAL,
                min_days: int = MIN_DAYS) -> dict:
    """{station: (m, subsidence)} monthly medians of the declared window's daily heights,
    flags 0-1, minus the common mode ``removal`` (a column of the daily common-mode table;
    None = none; 'wra_hf' applies to WRA stations only, NGL keep ngl_offfan), negated."""
    import pandas as pd

    d = daily[daily["flag"].isin(FLAGS)].copy()
    d = d[d["station"].isin(list(stations))]
    c = cm.set_index("date")
    out = {}
    for s, g in d.groupby("station"):
        st = stations[s]
        w0, w1 = st["window"]
        g = g[(g["date"] >= w0) & (g["date"] <= w1)]
        v = g.set_index("date")["value"].astype(float)
        rem = removal
        if rem == "wra_hf" and st["family"] != "wra":
            rem = REMOVAL
        if rem is not None:
            v = (v - c[rem].reindex(v.index)).dropna()
        if v.empty:
            continue
        mon = pd.Series(v.values, index=[x[:7] for x in v.index])
        agg = mon.groupby(level=0).agg(["median", "count"])
        agg = agg[agg["count"] >= min_days]
        if agg.empty:
            continue
        m = np.array([12 * int(k[:4]) + int(k[5:7]) - 1 for k in agg.index], dtype="int64")
        out[s] = (m, -agg["median"].to_numpy(dtype=float))
    return out


def restrict(series: dict, arm: str, obs: dict, drop_year: int | None = None):
    """(obs, mod) dicts on the observed months (optionally without one calendar year)."""
    o2, m2 = {}, {}
    for s, (m, y) in obs.items():
        key = f"{arm}|{s}|m"
        if key not in series:
            continue
        mm, ym = series[key], series[f"{arm}|{s}|y"]
        keep = np.isin(m, mm)
        if drop_year is not None:
            keep &= (m // 12) != drop_year
        idx = np.searchsorted(mm, m[keep])
        o2[s] = (m[keep], y[keep])
        m2[s] = (mm[idx], ym[idx])
    return o2, m2


def stratum(doc: dict, series: dict, obs: dict, arm: str, names, B=E.B_SCORE, seed=E.SEED,
            drop_year=None) -> dict:
    sub = {s: obs[s] for s in names if s in obs}
    o, m = restrict(series, arm, sub, drop_year)
    pairs = E.station_pairs(o, m)
    fam = {s: doc["stations"][s]["family"] for s in pairs}
    res = E.score_stratum(pairs, fam, B=B, seed=seed)
    res["stations"] = sorted(pairs)
    res["stations_dropped"] = sorted(set(names) - set(pairs))
    if pairs:
        res["per_station"] = {s: {"A_obs_m": E.amplitude(list(p["obs"])),
                                  "A_mod_m": E.amplitude(list(p["mod"])),
                                  "years": p["years"]} for s, p in pairs.items()}
        po = {s: dict(zip(p["years"], map(tuple, p["obs"]), strict=True))
              for s, p in pairs.items()}
        pm = {s: dict(zip(p["years"], map(tuple, p["mod"]), strict=True))
              for s, p in pairs.items()}
        res["phase_lag_months_obs_minus_model"] = {
            s: ((E.phase_months(po[s]) - E.phase_months(pm[s]) + 6) % 12 - 6) for s in pairs}
        res["slope_form"] = E.slope_through_origin(o, m)
    return res


def page_mapping(t1: dict, t1n: dict) -> dict:
    v, vn = t1["verdict"], t1n["verdict"]
    clash = vn in E.OPPOSITE.get(v, frozenset())
    status = ("S (design-time, window flagged)" if v == "consistent" and not clash else "N")
    words = {
        "consistent": "C32 seasonal elastic subsidence at Yunlin GNSS consistent in size "
                      "(r_A {r:.2f}, 90 % CI {lo:.2f}-{hi:.2f}; design-time, window "
                      "flagged); C27 gains 'elastic mechanism consistent in size at the "
                      "seasonal scale (design-time)'; cut size untested",
        "larger": "C32 N: the model's seasonal elastic response is larger than observed at "
                  "Yunlin GNSS (r_A {r:.2f}, 90 % CI {lo:.2f}-{hi:.2f}; design-time); C27 "
                  "'no direction' -> 'short-term elastic size overstated; step C and GNSS "
                  "agree'; C03 note 'size likely overstated, no correction'; C26 gains the "
                  "coastal direction",
        "smaller": "C32 N: the model's seasonal elastic response is smaller than observed "
                   "(r_A {r:.2f}, 90 % CI {lo:.2f}-{hi:.2f}; design-time); C27 gains the "
                   "direction; no correction",
        "inconclusive": "C32 N, inconclusive (r_A {r:.2f}, 90 % CI {lo:.2f}-{hi:.2f}); C27 "
                        "unchanged with a power note"}
    r, (lo, hi) = t1.get("r_A", float("nan")), t1.get("ci90", [float("nan")] * 2)
    line = words[v].format(r=r, lo=lo, hi=hi)
    if clash:
        line += f"; the least-seen NGL stratum reads '{vn}', so the status stays N"
    return {"C32_status": status, "verdict": v, "ngl_stratum_verdict": vn,
            "strata_clash": clash, "line": line,
            "never": "no column switch, no subset range, no correction factor under any "
                     "outcome; C03, C09 and C31 do not move"}


def compute(doc: dict, series: dict, obs: dict, obs_alt: dict, B=E.B_SCORE) -> dict:
    """Every registered statistic from the observed monthly series (primary removal) and
    the alternative removals {'none': ..., 'wra_hf': ...} (report-only)."""
    U = doc["units"]
    prim, ngl = U["primary"], U["ngl_stratum"]
    t1 = stratum(doc, series, obs, M.PRIMARY_ARM, prim, B=B)
    t1n = stratum(doc, series, obs, M.PRIMARY_ARM, ngl, B=B)
    ro = {}
    for z in ("distal", "mid"):
        ro[f"zone_{z}"] = stratum(doc, series, obs, M.PRIMARY_ARM,
                                  [s for s in prim if doc["stations"][s]["zone"] == z], B=B)
    ro["wra_without_2021"] = stratum(doc, series, obs, M.PRIMARY_ARM, prim, B=B,
                                     drop_year=2021)
    for k, o in obs_alt.items():
        ro[f"removal_{k}"] = stratum(doc, series, o, M.PRIMARY_ARM, prim, B=B)
    full = {s: doc["arms"][M.PRIMARY_ARM]["full_window"][s]["amplitude_m"]
            for s in t1.get("stations", [])}
    if full and t1.get("per_station"):
        ro["full_window_A_mod"] = {
            "r_A": sum(v["A_obs_m"] for v in t1["per_station"].values()) / sum(full.values())}
    ro["arms"] = {a: stratum(doc, series, obs, a, prim, B=B)
                  for a in sorted(doc["arms"]) if a != M.PRIMARY_ARM}
    ch = U["changhua_arm"]
    ro["changhua_arm"] = {a: stratum(doc, series, obs, a, ch, B=B)
                          for a in CHANGHUA_ARMS if a in doc["arms"]}
    seen = {}
    for s in M.SEEN_REPORT_ONLY:
        if s in obs:
            o, m = restrict(series, M.PRIMARY_ARM, {s: obs[s]})
            if s in o:
                co, cm_ = E.year_coefficients(*o[s]), E.year_coefficients(*m[s])
                seen[s] = {"A_obs_m": E.amplitude(co) if co else None,
                           "A_mod_m": E.amplitude(cm_) if cm_ else None}
    ro["seen_stations_ANES_CHSG"] = seen
    return {"T1": t1, "T1n": t1n, "report_only": ro, "page_mapping": page_mapping(t1, t1n),
            "f_subs_page_declaration": doc.get("f_subs_page")}


# --------------------------------------------------------------------------- registry
def repo() -> Path:
    from common.paths import REPO

    return REPO


def verify_registration(prereg_dir=None, rows_dir=None) -> dict:
    from common.paths import LEDGER_ROWS, PREREG_DIR
    from harness import prereg3

    try:
        doc, h = prereg3.load_prereg(PREREG_ID, prereg_dir=prereg_dir or PREREG_DIR,
                                     rows_dir=rows_dir or LEDGER_ROWS)
    except prereg3.PreregistrationError as e:
        raise Engineering(1, f"not registered or not intact: {e}") from e
    r = repo()
    fz = doc.get("frozen_code", {})
    if sorted(fz) != sorted(FROZEN) or any(L.sha(r / p) != s for p, s in fz.items()):
        raise Engineering(1, "frozen code differs from the registration")
    bad = prereg3.changed_inputs(doc)
    if bad:
        raise Engineering(1, f"registered inputs changed: {bad}")
    if not doc.get("g0oc", {}).get("pass"):
        raise Engineering(1, "the registered G0-OC did not pass: the test is not run")
    return {"doc": doc, "sha256": h}


def read_observed(r: Path, ms: dict) -> tuple[dict, dict]:
    """Opens the product (design_time_evidence) and reads the declared station windows."""
    import pandas as pd
    from common import guard

    prod = guard.open_product(r / M.PRODUCT, USE)
    cols = ["family", "station", "date", "value", "flag"]
    daily = pd.concat([prod.read_table("ngl/daily.parquet", columns=cols),
                       prod.read_table("wra/daily.parquet", columns=cols)])
    cm = prod.read_table("common_mode.parquet", columns=["date", "ngl_offfan", "wra_hf"])
    st = {s: v for s, v in ms["stations"].items()}
    obs = obs_monthly(daily, cm, st, REMOVAL)
    alt = {"none": obs_monthly(daily, cm, st, None),
           "wra_hf": obs_monthly(daily, cm, st, "wra_hf")}
    return obs, alt


def score(out: Path, model_dir: Path, prereg_dir=None, rows_dir=None, log=True) -> int:
    from harness import prereg3

    r = repo()
    try:
        reg = verify_registration(prereg_dir, rows_dir)
        ms, series = M.load(model_dir)
        want = reg["doc"]["model_side"]
        if L.sha(model_dir / "model_side.json") != want["model_side_json_sha256"]:
            raise Engineering(1, "model_side.json differs from the registered one")
        from common import guard

        guard.open_product(r / M.PRODUCT, USE)                 # step 2: refuses if not
    except Engineering as e:
        print(f"ENGINEERING FAILURE (exit {e.code}): {e}")
        return e.code
    kw = {} if prereg_dir is None else {"prereg_dir": prereg_dir, "rows_dir": rows_dir}
    a = prereg3.assert_can_score(PREREG_ID, WP, log=log, **kw)
    ro = sorted(set(ms["units"]["changhua_arm"]) | set(ms["units"]["report_only"]))
    if log and ro:
        sc = M.unit_scope(ms["stations"], ro)
        prereg3.record_look(
            [UNIT_REPORT], sc["data_scope"], WP, "report_only",
            members={UNIT_REPORT: {"description": "report-only arms of " + PREREG_ID,
                                   "ids": station_ids(ro), "window": sc["window"]}},
            prereg_id=PREREG_ID, note="Changhua arm, ANES/CHSG and report-only stations; "
                                      "never gates", **kw)
    obs, alt = read_observed(r, ms)
    res = compute(ms, series, obs, alt)
    rep = {"id": PREREG_ID, "registration_sha256": reg["sha256"],
           "assert_can_score": {k: a[k] for k in ("sha256", "tests",
                                                   "post_registration_looks")},
           "label": "design-time (window flagged): the GNSS units were seen in product QC "
                    "and fusion; never compared with the twin before", **res}
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1, ensure_ascii=False, sort_keys=True, default=float)
        fh.write("\n")
    print(f"scored once: {out} (sha256 {L.sha(out)}); T1 {res['T1']['verdict']}, "
          f"T1n {res['T1n']['verdict']}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("verify-registration")
    s = sp.add_parser("score")
    s.add_argument("--out", default=None)
    s.add_argument("--model-dir", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "verify-registration":
        try:
            v = verify_registration()
        except Engineering as e:
            print(f"REFUSED: {e}")
            return e.code
        print(f"registered and intact: {PREREG_ID} sha256 {v['sha256']}")
        return 0
    r = repo()
    return score(Path(a.out) if a.out else r / REPORT,
                 Path(a.model_dir) if a.model_dir else r / M.OUT_DIR)


if __name__ == "__main__":
    raise SystemExit(main())
