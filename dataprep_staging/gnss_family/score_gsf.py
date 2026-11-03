"""Scorer of X5.gnss_seasonal_family.v1 (round 8 rank 11). Run ONCE by the lead, after
registration; it is the only code of this package that opens GNSS, pile or electricity
values. Nothing runs on import; the builder never ran ``score``.

    PY=~/miniforge3/envs/hydro/bin/python
    $PY dataprep_staging/gnss_family/score_gsf.py verify-registration
    $PY dataprep_staging/gnss_family/score_gsf.py score

Order inside ``score`` (each step must pass before the next; nothing observed is read before
step 4):

1. the registration is intact, the frozen code and every registered input hash as
   registered, the registered G-sep and G0 passed, the rulings file is the bound one;
2. every product opens for design_time_evidence (guard; gnss_clean/preorigin/v4 and
   deep_benchmarks/v2 failed their gates, so only the gate-exempt uses open);
3. prereg3.assert_can_score logs the scoring looks of every test (T1, T2, DIR, U2) and refuses
   on a blocking look; one report_only look covers ANES/CHSG;
4. GNSS daily heights of the declared windows through v1's obs_monthly unchanged (flags 0-1,
   ngl_offfan common mode, monthly medians of >= 10 days, negated), NO extra jump rule on the
   gated path (the qc_jumps-cleaned variant is report-only: its constants postdate a 2023-25
   look, U7); pile daily compaction (pre-origin variant, one feed per sensor, flags 0-2,
   2016-06..2019-12, spring months masked on pile and model alike); township-cell
   electricity (report-only F1); every registered statistic; report.json (mode 'x').
5. From the moment the scoring looks are logged, ANY failure writes the final report.json
   (status engineering_failure_after_look) and an attempts.jsonl row; there is no rerun.
"""
from __future__ import annotations

import argparse
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
import gsf_model as MD  # noqa: E402
import numpy as np  # noqa: E402

rp, sha256 = MD.rp, MD.sha256
FROZEN = MD.CODE


class Engineering(Exception):
    def __init__(self, code: int, msg: str):
        super().__init__(msg)
        self.code = code


# --------------------------------------------------------------------------- ledger texts
#: the screen's pass label names the deciding statistic (lead ruling on review B3)
LABEL = K.PASS_LABEL
#: DIR 'larger' carries R8.5's label with DIR's own caveat (no G0 of its own; v1's G0-OC failed)
DIR_LABEL = ("S (design-time, window flagged, outcome partly known; v1's directional arm, "
             "not covered by this G0; v1's G0-OC failed on its consistent path)")
T2_NOTE = (" Decided by T2, the frame-carrying statistic, whose outcome was close to known "
           "(N1's 6.1 mm InSAR median); T1, the frame-free statistic, could not exclude the "
           "far member at the expected half-width.")
NEVER = ("no column switch, no subset range replacing E24, no correction factor under any "
         "outcome; the page column and every page number stay; no cut number becomes S; the "
         "cut30 values of the members stay where SD v2 prints them, never beside this label")
OUTCOME_TEXT = {
    "a": ("E24 (beside, never replacing, the full family): the seasonal GNSS screen keeps "
          "{survivors} and excludes {excluded} among SD v2's leveling-equivalent columns "
          "(full family {family}): 'model-form spread among columns consistent with the GNSS "
          "seasonal amplitude' - " + LABEL + "." + T2_NOTE + " C11/C26/C29 notes name the "
          "surviving columns. " + NEVER),
    "b": ("Narrowing only: 'columns with the page column's coastal elastic response are "
          "larger than seasonal GNSS supports' - " + LABEL + "." + T2_NOTE + " Survivors "
          "{survivors}; excluded {excluded}; E24's full family {family} stays printed with "
          "the surviving members named BESIDE it. Notes: C03 'size likely overstated, no "
          "correction' (C03 stays W), C11, C26 and C29 'coastal elastic share likely "
          "overstated (design-time)'. " + NEVER),
    "c": ("Seasonal GNSS cannot narrow the column family: cluster and far members both "
          "survive ({survivors}; excluded {excluded})." + T2_NOTE + " E24 unchanged; README "
          "Limits line 'model-form spread not narrowed by seasonal GNSS'. Status N. " + NEVER),
    "d": ("No leveling-equivalent column reproduces the seasonal amplitude at Yunlin GNSS "
          "(every screen member excluded, by T1 or T2). E24's model-form spread is "
          "understated: README Limits and an E24 note; no column replaced. Status N. " + NEVER)}
DIR_TEXT = {
    "larger": ("C27 gains a decided direction: the page column's seasonal (elastic) "
               "subsidence response is larger than observed at Yunlin GNSS at both frame "
               "scales (r_A {r:.2f}, 90 % CI {lo:.2f}-{hi:.2f}) - " + DIR_LABEL + "; step C "
               "and GNSS agree in direction; C03 note 'size likely overstated, no correction'"),
    "smaller": ("C27 N note: the page column's seasonal elastic response reads smaller than "
                "observed at both frame scales (r_A {r:.2f}, 90 % CI {lo:.2f}-{hi:.2f}); "
                "status N; no correction"),
    "consistent": ("C27 N note: seasonal elastic size reads consistent within x1.5 at both "
                   "frame scales (r_A {r:.2f}, 90 % CI {lo:.2f}-{hi:.2f}); status N (no G0 "
                   "covers this arm; v1's G0-OC failed on this path); cut size untested"),
    "inconclusive": ("C27 unchanged (r_A {r:.2f}, 90 % CI {lo:.2f}-{hi:.2f}); power note: "
                     "v1's G0-OC gives P('larger' | r = 0.4) 0.57 and P('consistent' | r = 1) "
                     "0.11 under this arm's rule")}
DIR_CLASH = "; the least-seen NGL stratum (T1n) reads '{vn}', so C27 stays N"
U2_TEXT = ("U2 (named secondary, frame-free): GNSS frame over-read at {site} bounded one-sided "
           "by s <= {s_max:.2f} (design-time; non-spring pile months 2016-06..2019-12 against "
           "GNSS 2020-22 at the same site, stationarity assumed); reading: {reading}; it "
           "narrows no screen")


def outcome_line(reading: str, survivors, excluded, screen) -> str:
    """Member names only (review S5): no cut30 number next to the label."""
    return OUTCOME_TEXT[reading].format(
        survivors=", ".join(survivors) or "none", excluded=", ".join(excluded) or "none",
        family=", ".join(screen))


def dir_line(t1: dict, t1n: dict) -> dict:
    """C27 only (review S3): 'larger' carries DIR_LABEL; every other verdict is an N note."""
    import gsc_estimator as E

    v, vn = t1["verdict"], t1n["verdict"]
    clash = vn in E.OPPOSITE.get(v, frozenset())
    r, (lo, hi) = t1.get("r_A", float("nan")), t1.get("ci90", [float("nan")] * 2)
    line = DIR_TEXT[v].format(r=r, lo=lo, hi=hi)
    if clash:
        line += DIR_CLASH.format(vn=vn)
    status = DIR_LABEL if v == "larger" and not clash else "N"
    return {"verdict": v, "ngl_stratum_verdict": vn, "clash": clash, "C27_status": status,
            "line": line}


# --------------------------------------------------------------------------- observed
def obs_gnss(prod, ms: dict, names) -> tuple[dict, dict, dict]:
    """(gated obs, report-only qc_jumps-cleaned obs, qc record) for ``names``.

    Gated: v1's obs_monthly unchanged (flags 0-1, ngl_offfan, monthly medians of >= 10 days,
    negated); NO extra jump rule (review S1, U7). Report-only: the same after dropping WRA
    station-months with >= 3 consecutive-day |delta value_raw| > 3 cm."""
    import pandas as pd
    import score_gsc as S

    cols = ["family", "station", "date", "value_raw", "value", "flag"]
    daily = pd.concat([prod.read_table("ngl/daily.parquet", columns=cols),
                       prod.read_table("wra/daily.parquet", columns=cols)])
    daily = daily[daily["station"].isin(list(names))]
    cm = prod.read_table("common_mode.parquet", columns=["date", K.REMOVAL])
    st = {s: ms["stations"][s] for s in names}
    gated = S.obs_monthly(daily, cm, st, K.REMOVAL)
    qc = F.qc_jump_months(daily[daily["family"] == "wra"])
    cleaned = S.obs_monthly(F.drop_months(daily, qc), cm, st, K.REMOVAL)
    return gated, cleaned, {"qc_jump_months": sorted(f"{a}:{b}" for a, b in qc),
                            "n_qc_jump_months": len(qc), "role": "report-only (U7)"}


def obs_piles(prod) -> dict:
    df = prod.read_table("daily.parquet",
                         columns=["sensor_id", "feed", "date", "value", "flag", "variant"])
    out = {}
    for s in K.PILES:
        m, y, feed = F.pile_monthly(df, s)
        out[s] = {"m": m, "y": y, "feed": feed}
    return out


def elec_anomaly(prod, zone: np.ndarray, ch: np.ndarray, window: tuple[int, int]) -> dict:
    """{zone: {m: anomaly}} of Yunlin zone-total electricity (value_all), trend and calendar-
    month climatology removed over ``window`` (month indices), electricity side only."""
    z = np.load(prod.path(K.ELEC_FILE), allow_pickle=False)
    dates = [str(d) for d in z["dates"]]
    m = np.array([12 * int(d[:4]) + int(d[5:7]) - 1 for d in dates])
    v = np.asarray(z["value_all"], float)
    sel = (m >= window[0]) & (m <= window[1])
    out = {}
    for zi, name in ((2, "distal"), (1, "mid")):
        cells = (zone == zi) & ~ch
        tot = np.nansum(v[cells][:, sel], axis=0)
        out[name] = dict(zip(m[sel].tolist(), F.deseason(m[sel], tot).tolist(), strict=True))
    return out


# --------------------------------------------------------------------------- statistics
def on_model_months(obs: dict, series: dict, arms) -> tuple[dict, dict]:
    """(obs, {arm: mod}) restricted to the model months of each station."""
    o2, m2 = {}, {a: {} for a in arms}
    for s, (m, y) in obs.items():
        key = f"{K.PAGE}|{s}|m"
        if key not in series:
            continue
        mm = np.asarray(series[key])
        keep = np.isin(m, mm)
        idx = np.searchsorted(mm, m[keep])
        o2[s] = (m[keep], y[keep])
        for a in arms:
            m2[a][s] = (mm[idx], np.asarray(series[f"{a}|{s}|y"], float)[idx])
    return o2, m2


def screen_pairs(o: dict, mods: dict, arms, names, min_years: int = K.MIN_YEARS) -> dict:
    out = {}
    for s in names:
        if s not in o:
            continue
        yrs, Vo = F.year_vectors(*o[s])
        mod = {}
        for a in arms:
            ym, Vm = F.year_vectors(*mods[a][s])
            if ym != yrs:
                raise Engineering(2, f"{s}/{a}: model years differ from the observed years")
            mod[a] = Vm
        if len(yrs) >= min_years:
            out[s] = {"years": yrs, "obs": Vo, "mod": mod}
    return out


def screen_and_dir(ms: dict, series: dict, obs: dict, cluster_far: dict, seed: int,
                   stream: int, B: int) -> tuple[dict, dict, dict, dict, dict]:
    """(screen block, DIR block, pairs, o, mods) on one observed set."""
    import gsc_estimator as E

    arms = list(K.SCREEN) + list(K.REPORT_ONLY_ARMS)
    prim = ms["units"]["primary"]
    zone_of = {s: ms["stations"][s]["zone"] for s in ms["stations"]}
    o, mods = on_model_months(obs, series, arms)
    pairs = screen_pairs(o, mods, arms, prim)
    stk = F.stack(pairs, arms, zone_of)
    sc = F.screen(stk, np.random.default_rng(np.random.SeedSequence([seed, stream])), B)
    surv = [m for m in K.SCREEN if m in sc["survivors"]]
    excl = [m for m in K.SCREEN if m not in surv]
    rd = F.reading(surv, cluster_far["cluster"], cluster_far["far"])
    decided = {m: ("T1" if sc["members"][m]["T1"]["excluded"] else "")
               + ("T2" if sc["members"][m]["T2"]["excluded"] else "") for m in excl}
    scr = {"T1_obs": sc["T1_obs"], "T2_obs_m": sc["T2_obs_m"], "survivors": surv,
           "excluded": excl, "excluded_by": decided, "reading": rd,
           "members": {m: sc["members"][m] for m in K.SCREEN},
           "report_only_arms": {m: sc["members"][m] for m in K.REPORT_ONLY_ARMS},
           "line": outcome_line(rd, surv, excl, K.SCREEN),
           "decided_by_T2_disclosure": K.DECIDED_BY_T2,
           "n_stations": len(pairs), "stations": sorted(pairs),
           "stations_dropped": sorted(set(prim) - set(pairs))}
    # named secondary: v1's directional arm, estimator and criteria unchanged
    o1 = {s: o[s] for s in prim if s in o}
    p1 = E.station_pairs(o1, {s: mods[K.PAGE][s] for s in o1})
    fam = {s: ms["stations"][s]["family"] for s in p1}
    d1 = E.score_stratum(p1, fam, B=E.B_SCORE, seed=seed)
    pn = {s: p1[s] for s in K.NGL_STRATUM if s in p1}
    dn = E.score_stratum(pn, {s: fam[s] for s in pn}, B=E.B_SCORE, seed=seed)
    return scr, {"T1": d1, "T1n": dn, "mapping": dir_line(d1, dn)}, pairs, o, mods


def u2_sites(pairs: dict, pile_series: dict, piles: dict) -> dict:
    """{site: {"gnss", "pile", "gated", ...}}: each pile paired with its own GNSS station;
    spring months masked on the pile AND its model side; gated with >= MIN_YEARS_U2 years."""
    out = {}
    for sensor, rec in K.PILES.items():
        s = rec["gnss"]
        key = f"{K.PAGE}|{sensor}|m"
        if s not in pairs or key not in pile_series or sensor not in piles:
            continue
        pm = np.asarray(pile_series[key])
        py = np.asarray(pile_series[f"{K.PAGE}|{sensor}|y"], float)
        po = piles[sensor]
        keep = np.isin(po["m"], pm) & F.non_spring(po["m"])
        if not keep.sum():
            continue
        idx = np.searchsorted(pm, po["m"][keep])
        yrs, Vo = F.year_vectors(po["m"][keep], po["y"][keep])
        ym, Vm = F.year_vectors(pm[idx], py[idx])
        if not yrs or ym != yrs:
            continue
        out[s] = {"gnss": {"obs": pairs[s]["obs"], "mod": pairs[s]["mod"][K.PAGE]},
                  "pile": {"obs": Vo, "mod": Vm}, "gated": len(yrs) >= K.MIN_YEARS_U2,
                  "sensor": sensor, "years": yrs, "feed": po["feed"]}
    return out


def compute(ms: dict, series: dict, pile_series: dict, obs: dict, piles: dict,
            cluster_far: dict, seed: int, elec: dict | None, obs_qc: dict | None = None,
            B: int = K.B_SCORE) -> dict:
    scr, direc, pairs, o, mods = screen_and_dir(ms, series, obs, cluster_far, seed,
                                                K.STREAM_SCREEN, B)
    zone_of = {s: ms["stations"][s]["zone"] for s in ms["stations"]}
    # named secondary: U2 deep-pile frame bound, per site (page column)
    sites = u2_sites(pairs, pile_series, piles)
    u2 = F.u2_bound({k: {kk: v[kk] for kk in ("gnss", "pile", "gated")}
                     for k, v in sites.items()},
                    np.random.default_rng(np.random.SeedSequence([seed, K.STREAM_U2])), B)
    if u2.get("evaluable"):
        u2["line"] = U2_TEXT.format(site=u2["bound_site"], s_max=u2["s_max"],
                                    reading=u2["reading"])
    u2["site_meta"] = {k: {kk: v[kk] for kk in ("sensor", "years", "feed", "gated")}
                       for k, v in sites.items()}
    # report-only
    ro = {"phase_months_obs_minus_page": {
              s: ((F.phase_months(v["obs"]) - F.phase_months(v["mod"][K.PAGE]) + 6) % 12 - 6)
              for s, v in pairs.items()},
          "zone_amplitudes_m": {s: {"obs": F.amplitude(v["obs"]),
                                    "page": F.amplitude(v["mod"][K.PAGE]),
                                    "zone": zone_of[s]} for s, v in pairs.items()}}
    seen = {}
    for s in K.REPORT_STATIONS:
        if s in obs and f"{K.PAGE}|{s}|m" in series:
            oo, mm = on_model_months({s: obs[s]}, series, [K.PAGE])
            if s in oo:
                _, Vo = F.year_vectors(*oo[s])
                _, Vm = F.year_vectors(*mm[K.PAGE][s])
                seen[s] = {"A_obs_m": F.amplitude(Vo), "A_page_m": F.amplitude(Vm)}
    ro["ANES_CHSG"] = seen
    if obs_qc is not None:
        qs, qd, *_ = screen_and_dir(ms, series, obs_qc, cluster_far, seed,
                                    K.STREAM_SCREEN_QC, B)
        ro["qc_jumps_variant"] = {"screen": qs, "DIR": qd,
                                  "provenance": "U7: jump constants set after 2023-25 jump "
                                                "counts were seen (S3 repair round); "
                                                "report-only, never read"}
    if elec is not None:
        f1 = {}
        for a in K.SCREEN:
            for tag, drop in (("all", None), ("without_2021", K.F1_DROP_YEAR)):
                bo = F.f1_beta({s: o[s] for s in pairs}, elec, zone_of, drop)
                bm = F.f1_beta({s: mods[a][s] for s in pairs}, elec, zone_of, drop)
                f1[f"{a}|{tag}"] = {k: {"beta_obs": bo[k], "beta_mod": bm[k],
                                        "ratio": (bo[k] / bm[k] if bm[k] else float("nan"))}
                                    for k in bo}
        ro["F1_electricity_reduced_form"] = f1
    return {"T1_screen": scr, "DIR": direc, "U2": u2, "report_only": ro}


# --------------------------------------------------------------------------- registry
def verify_registration(prereg_dir=None, rows_dir=None) -> dict:
    from common.paths import LEDGER_ROWS, PREREG_DIR
    from harness import prereg3

    try:
        doc, h = prereg3.load_prereg(K.ID, prereg_dir=prereg_dir or PREREG_DIR,
                                     rows_dir=rows_dir or LEDGER_ROWS)
    except prereg3.PreregistrationError as e:
        raise Engineering(1, f"not registered or not intact: {e}") from e
    fz = doc.get("frozen_code", {})
    if sorted(fz) != sorted(FROZEN) or any(sha256(p) != s for p, s in fz.items()):
        raise Engineering(1, "frozen code differs from the registration")
    bad = prereg3.changed_inputs(doc)
    if bad:
        raise Engineering(1, f"registered inputs changed: {bad}")
    if not (doc.get("gates", {}).get("gsep", {}).get("pass") is True
            and doc.get("gates", {}).get("g0", {}).get("pass") is True):
        raise Engineering(1, "the registered G-sep or G0 did not pass: the test is not run")
    if sha256(K.RULINGS) != K.RULINGS_SHA:
        raise Engineering(1, "the rulings file changed")
    return {"doc": doc, "sha256": h}


def open_products():
    from common import guard

    return (guard.open_product(rp(K.GNSS_PRODUCT), K.USE),
            guard.open_product(rp(K.PILE_PRODUCT), K.USE, window_end="2022-12"),
            guard.open_product(rp(K.ELEC_PRODUCT), K.USE))


def _write_final(out: Path, rep: dict) -> str:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1, ensure_ascii=False, sort_keys=True, default=float)
        fh.write("\n")
    return sha256(out)


def after_look(reg: dict, ms: dict, series: dict, sup: dict, gnss, pile, elec,
               looks: dict) -> dict:
    """Everything after the scoring looks (observed values are read here)."""
    import sdlib as L

    doc = reg["doc"]
    names = sorted(set(ms["units"]["primary"]) | set(K.REPORT_STATIONS))
    obs, obs_qc, qc = obs_gnss(gnss, ms, names)
    piles = obs_piles(pile)
    gm = L.grid_and_masks(REPO)
    el = elec_anomaly(elec, gm["zone"], gm["changhua"], (12 * 2014 + 7, 12 * 2022 + 11))
    res = compute(ms, series, sup, obs, piles, doc["engineering"]["cluster_far"],
                  int(doc["seed"]), el, obs_qc=obs_qc)
    return {"id": K.ID, "status": "scored", "registration_sha256": reg["sha256"],
            "assert_can_score": looks, "label": LABEL, "qc_report_only": qc, **res}


def score(prereg_dir=None, rows_dir=None, log: bool = True) -> int:
    import traceback

    from harness import prereg3

    out = rp(K.REPORT)
    MD.log_attempt({"event": "score_start"})
    try:
        if out.exists():
            raise Engineering(1, f"{K.REPORT} exists: scored once")
        reg = verify_registration(prereg_dir, rows_dir)
        doc = reg["doc"]
        ms, series = MD.load_v1()
        sup_doc, sup = MD.load_supplement()
        if sup_doc["series_digest"] != doc["model_side"]["pile_series_digest"]:
            raise Engineering(1, "pile_model.npz is not the registered supplement")
        gnss, pile, elec = open_products()                     # step 2: refuses if not
    except (Engineering, SystemExit) as e:
        code = getattr(e, "code", 1) if isinstance(getattr(e, "code", 1), int) else 1
        MD.log_attempt({"event": "score_refused_before_look", "error": str(e)})
        print(f"ENGINEERING FAILURE before any look (exit {code}): {e}")
        return code
    kw = {} if prereg_dir is None else {"prereg_dir": prereg_dir, "rows_dir": rows_dir}
    a = prereg3.assert_can_score(K.ID, K.WP, log=log, **kw)
    looks = {k: a[k] for k in ("sha256", "tests", "post_registration_looks")}
    MD.log_attempt({"event": "score_looked", "assert_can_score_sha256": a["sha256"]})
    # from here on the looks are logged: any failure writes the FINAL report (review S4)
    try:
        if log:
            prereg3.record_look(
                [K.UNIT_REPORT], "gnss:2020-01..2022-12", K.WP, "report_only",
                members={K.UNIT_REPORT: {"description": "report-only ANES/CHSG rows of " + K.ID,
                                         "ids": sorted({x for s in K.REPORT_STATIONS
                                                        for x in (s, f"gnss:{s}")}),
                                         "window": "2020-01..2022-12"}},
                prereg_id=K.ID, note="seen stations; never gate", **kw)
        rep = after_look(reg, ms, series, sup, gnss, pile, elec, looks)
    except BaseException as e:                          # noqa: BLE001 - final either way
        rep = {"id": K.ID, "status": "engineering_failure_after_look",
               "registration_sha256": reg["sha256"], "assert_can_score": looks,
               "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc(),
               "final": "the looks are spent; no rerun; the test reads as not decided"}
        s = _write_final(out, rep)
        MD.log_attempt({"event": "score_failed_after_look", "report_sha256": s,
                        "error": rep["error"]})
        print(f"ENGINEERING FAILURE after the look: final report {out} (sha256 {s})")
        return 5
    s = _write_final(out, rep)
    MD.log_attempt({"event": "score_done", "report_sha256": s,
                    "reading": rep["T1_screen"]["reading"],
                    "dir": rep["DIR"]["mapping"]["verdict"]})
    print(f"scored once: {out} (sha256 {s}); reading {rep['T1_screen']['reading']}; DIR "
          f"{rep['DIR']['mapping']['verdict']}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("verify-registration")
    sp.add_parser("score")
    a = ap.parse_args(argv)
    if a.cmd == "verify-registration":
        try:
            v = verify_registration()
        except Engineering as e:
            print(f"REFUSED: {e}")
            return e.code
        print(f"registered and intact: {K.ID} sha256 {v['sha256']}")
        return 0
    return score()


if __name__ == "__main__":
    raise SystemExit(main())
