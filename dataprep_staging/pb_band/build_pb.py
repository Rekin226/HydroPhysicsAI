"""PB.decomposed_band.v1 band builder (round-8 rank 16, PB part; user ruling R8.13).

Refuses unless PB.decomposed_band.v1 is registered (INDEX sha, frozen code unchanged), E02d,
E02y, FV v2 and DR are registered at the sha256 the declaration binds, and every bound input
file has its declared sha256. Then, CPU only, model output and pre-origin tables only (no
observation is read):

1. identity check (engineering gate, 1e-9 m) of E02d's frozen artefact sigma_idio_A/B/C.csv
   against E02d's registered values: file sha256 (interval.artefact.files); per row
   sigma_idio = sqrt(sd_fit_rel^2 + sigma_ens_rel^2 + drift^2), half_width = z90 sigma_idio;
   sd_fit_rel and group equal DR's wells_e02d.csv; drift = sqrt(tau2) |t - anchor| with E02d's
   registered dr.decisions and lead definition; sigma_ens_rel = e02dlib.sigma_ens_rel of the
   registered 36 rerun members; predicted_m equals the registered centre (E02's predictions
   for A, the absolute registrations for B and C); 158 wells x the window's months; lead =
   months after 2022-12. Any failure: nothing is computed or written;
2. the per-well zone ribbon (pb_const docstring): per window, DR zone and month, the median
   over the zone's wells of half_width_m; zones with fewer than 5 wells are not drawn; the
   page draws window A only (B and C are computed in the output, ``drawn`` false);
3. the shared range per county on window A, the spread among model variants only: hull of
   {0, the members' county-median deviation from the member mean, the county-median
   departure of the FV v1 baseline forwards of FV v2's four truth families (fb_inputs.load_mu:
   FV's frozen departure against E02's registered forward)}; it excludes forcing-year
   variation and the between-zone part of the departure;
4. its leave-one-family-out coverage per county (design-time, model-only, window A only):
   for each family k, is k's county-wide departure inside the hull built without k, month by
   month; per family, pooled over the four families, and (report-only) leaving out each of
   FB's two families {R0,Q0} / {R1c,X5R1} whole.

U7 guard: :func:`u7_guard` refuses any coverage of the relative recipe (PB computes none: E02d
and E02y score it), and any coverage on a month before 2026-12; :func:`assert_no_observation_
inputs` refuses an input path of an observation product; :func:`no_station_ids` refuses an
output that holds a station id. Leave-one-forcing-year-out is dropped (never computed).

Writes once: <out>/pb_band.json (default results/twin/pb_band). Every attempt is appended to
results/twin_runs/pb_band/attempts.jsonl.

    python -I dataprep_staging/pb_band/build_pb.py hashes --twin-dir <results/twin>
    python -I dataprep_staging/pb_band/build_pb.py build --twin-dir <results/twin>
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STAGING = HERE.parent
for _p in (str(HERE), str(STAGING), str(STAGING / "e02d"), str(STAGING / "fb_family"),
           str(STAGING / "fv_decidability")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import e02dio as io  # noqa: E402  (sha256, registered, write_once, Refused)
import e02dlib as L  # noqa: E402  (E02d's frozen functions, used unchanged)
import pb_const as K  # noqa: E402

ART_COLS = ("station_id", "county", "layer", "group", "date", "lead_months", "predicted_m",
            "sd_fit_rel_m", "sigma_ens_rel_m", "drift_m", "sigma_idio_m", "half_width_m")
_ID_TOKEN = re.compile(r"(?<![0-9a-fA-F])[0-9]{8}(?![0-9a-fA-F])")


class U7Refused(io.Refused):
    """A coverage the U7 rule forbids (relative recipe, or a month before 2026-12)."""


# ----------------------------------------------------------------------------- months
def month_key(x) -> str:
    """'YYYY-MM' of a date-like value."""
    return pd.Timestamp(str(x)).strftime("%Y-%m")


def month_list(first: str, last: str) -> list[str]:
    return [m.strftime("%Y-%m") for m in pd.date_range(f"{first}-01", f"{last}-01", freq="MS")]


def lead_months(months) -> np.ndarray:
    o = pd.Timestamp(f"{K.ORIGIN}-01")
    t = pd.DatetimeIndex([pd.Timestamp(f"{month_key(m)}-01") for m in months])
    return ((t.year - o.year) * 12 + t.month - o.month).to_numpy(int)


# ----------------------------------------------------------------------------- guards
def u7_guard(months, recipe: str) -> list[str]:
    """The U7 guard of PB. ``recipe`` 'relative' (E02d's per-well recipe) is refused in every
    month: its 2023-26 coverage may never be computed before scoring (E02d sealed_until_
    scoring) and its window coverage belongs to E02d / E02y alone. ``recipe`` 'shared' (the
    shared range's LOFO coverage, model-only) is refused on any month before 2026-12. Returns
    the months as 'YYYY-MM'."""
    ms = [month_key(m) for m in months]
    if recipe == "relative":
        sealed = [m for m in ms if K.U7_SEALED[0] <= m <= K.U7_SEALED[1]]
        if sealed:
            raise U7Refused(f"U7: no 2023-26 coverage of the relative recipe may be computed "
                            f"(months {sealed[0]}..{sealed[-1]})")
        raise U7Refused("PB computes no coverage of the relative recipe; E02d and E02y score "
                        "it in their own sessions")
    if recipe != "shared":
        raise ValueError(f"unknown recipe {recipe!r}")
    if not ms:
        raise ValueError("no months")
    early = [m for m in ms if m < K.FIRST_COVERAGE_MONTH]
    if early:
        raise U7Refused(f"U7: PB computes no coverage on a month before "
                        f"{K.FIRST_COVERAGE_MONTH} ({early[0]}..{early[-1]})")
    return ms


def observation_inputs(paths) -> list[str]:
    return sorted(str(p) for p in paths if any(m in str(p) for m in K.OBSERVATION_MARKERS))


def assert_no_observation_inputs(paths) -> None:
    bad = observation_inputs(paths)
    if bad:
        raise U7Refused(f"PB reads no observation product: {bad}")


def no_station_ids(obj, ids=()) -> None:
    """Refuse an output that holds a station id (any of ``ids``, or any bare 8-digit token
    that is not part of a hex hash)."""
    text = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    hit = sorted({s for s in map(str, ids) if s and s in text})
    tok = _ID_TOKEN.findall(text)
    if hit or tok:
        raise io.EngineeringFailure(f"the output holds station ids ({len(hit)} known, "
                                    f"{len(tok)} 8-digit tokens)")


# ----------------------------------------------------------------------------- design
def design_spec() -> dict:
    """The design the declaration fixes (the builder checks the registered one equals it)."""
    return {
        "windows": {w: {"first": a, "last": b, "months": K.WINDOW_MONTHS_N[w],
                        "decided_by": K.WINDOW_TEST[w]} for w, (a, b) in K.WINDOWS.items()},
        "ribbon": {
            "source": "E02d's frozen sigma_idio_{A,B,C}.csv, column half_width_m "
                      f"(= z90 sigma_idio, z90 = {K.Z90!r})",
            "zone": "E02d's frozen 'group' column (DR.relative_drift.v1's zone grouping; "
                    "wells_e02d.csv)",
            "statistic": "per window, zone and month: the median over the zone's wells with a "
                         "finite half_width_m that month (numpy.median); drawn as +/- that "
                         "value around zero (the zone's common departure)",
            "drawn": "window A only (row C14a-r); windows B and C are computed in the output "
                     "with drawn = false and no page label",
            "min_zone_wells": K.MIN_ZONE_WELLS,
            "min_rule": "a zone with fewer than min_zone_wells distinct wells in a window is "
                        "not drawn for that window (counted, never named by well); a month "
                        "with fewer than min_zone_wells finite rows in a drawn zone is null",
            "station_ids": "none in the output (no_station_ids)"},
        "shared": {
            "window": K.SHARED_WINDOW,
            "member_part": f"a_m = members[m] - mean over the {K.N_MEMBERS} rerun members "
                           "(members_wells.npz); per county and month the median over the "
                           "county's wells of a_m; min and max over m",
            "family_part": "the FV v1 baseline forwards of FV v2's four truth families (R0, "
                           "Q0, R1c, X5R1): mu_k = fb_inputs.load_mu (osse_fv.departure of "
                           "the baseline scenario of FV v1's truth forward k against E02's "
                           "registered forward cf2c7154, datum-anchored on obs_mask "
                           "availability) on 2026-12..2027-10; per county and month the "
                           "median over the county's wells",
            "what": "the spread among model variants only (runs and recipe families); it "
                    "excludes forcing-year variation (FV v2's registered forcing-year line "
                    "stays beside it) and the between-zone part of the departure (a county "
                    "median, while E02d's relative operator removes zone medians)",
            "range": "lo = min(0, member min, min_k family), hi = max(0, member max, max_k "
                     "family), metres relative to E02's registered centre",
            "after_window": "not drawn after 2027-10 (the family forwards end there)"},
        "lofo": {
            "estimand": "per county: for each family k, the share of window-A months in which "
                        "k's county-wide departure lies inside [lo_-k, hi_-k], the hull built "
                        "without k (0, member part and the other three families; closed "
                        "interval, no tolerance); published per family and pooled over the "
                        "four families (inside / family-months)",
            "report_only": "leave one of FB's two families out whole ({R0,Q0} / {R1c,X5R1}): "
                           "each member family inside the hull of 0, the member part and the "
                           "other group",
            "kind": "design-time, model-only; no observation; no gate, no status mapping",
            "months": "window A only (2026-12..2027-10); never a month before 2026-12 (U7)"},
        "dropped": dict(K.DROPPED),
        "u7": {"sealed": f"{K.U7_SEALED[0]}..{K.U7_SEALED[1]}",
               "rule": "no coverage of the relative recipe in any month (2023-26 sealed by "
                       "E02d; window months scored by E02d/E02y); no coverage of any kind on "
                       "a month before 2026-12; no observation product among the inputs",
               "where": "build_pb.u7_guard, assert_no_observation_inputs and "
                        "tests/test_pb_band.py only (never hydrophysics/twin/coverage.py, "
                        "bands.py or harness/coverage3.py)"},
        "identity": {"tol_m": K.TOL_M,
                     "checks": ["file sha256 = E02d interval.artefact.files",
                                "158 wells x the window's months, lead = months after 2022-12",
                                "county from the station code; group and sd_fit_rel_m = "
                                "wells_e02d.csv",
                                "drift_m = sqrt(tau2) |year_frac - anchor| (E02d dr.decisions, "
                                "DR.relative_drift.v1's registered lead_definition)",
                                "summary.json decisions = E02d binds.dr.decisions and "
                                "lead_definition = DR.relative_drift.v1's (no fallback)",
                                "sigma_ens_rel_m = e02dlib.sigma_ens_rel(members)",
                                "sigma_idio_m = sqrt(sd^2 + ens^2 + drift^2)",
                                "half_width_m = z90 sigma_idio_m",
                                "predicted_m = the registered centre (CENTRE)"]},
        "rounding": f"metres to {K.ROUND} decimals in the output",
    }


# ----------------------------------------------------------------------------- inputs
def family_files() -> dict:
    import fb_inputs as FI

    return {FI.INPUT_FILES[k]: FI.UPSTREAM_SHA[FI.INPUT_FILES[k]] for k in K.FAMILY_INPUT_KEYS}


def bound_files() -> dict:
    """{repo-relative path: sha256} of every file PB reads, as upstream registrations recorded
    them (results/twin paths are read from --twin-dir)."""
    out = {K.SIGMA[w]: K.SIGMA_SHA[w] for w in K.WINDOWS}
    out.update({K.CENTRE[w]: K.CENTRE_SHA[w] for w in K.WINDOWS})
    out.update({K.SUMMARY: K.SUMMARY_SHA, K.WELLS_E02D: K.WELLS_E02D_SHA,
                K.MEMBERS_NPZ: K.MEMBERS_SHA})
    out.update(family_files())
    return out


def resolve(path: str, twin_dir) -> Path:
    if path.startswith(K.TWIN_PREFIX):
        return Path(twin_dir) / path[len(K.TWIN_PREFIX):]
    return io.repo_path(path)


def file_shas(paths, twin_dir) -> dict:
    """{path: sha256 or 'MISSING'} (hashes only; no value is read)."""
    out = {}
    for p in paths:
        q = resolve(p, twin_dir)
        out[p] = io.sha256(q) if q.is_file() else "MISSING"
    return out


def sha_problems(now: dict, want: dict) -> list[str]:
    return [f"{p}: {now.get(p)} != {h}" for p, h in sorted(want.items()) if now.get(p) != h]


def read_sigma(path) -> pd.DataFrame:
    a = pd.read_csv(path, dtype={"station_id": str})
    if tuple(a.columns) != ART_COLS:
        raise io.EngineeringFailure(f"{Path(path).name}: columns {list(a.columns)}")
    return a


# ----------------------------------------------------------------------------- identity
def _maxdiff(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or not (np.isfinite(a).all() and np.isfinite(b).all()):
        return float("inf")
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def identity_check(arts: dict, wells: pd.DataFrame, decisions: dict, lead_definition: str,
                   members: dict, centres: dict) -> dict:
    """E02d's frozen artefact against its registered values (1e-9 m). ``arts`` and ``centres``
    {window: frame}; ``wells`` DR's wells_e02d table; ``members`` {members, station_id,
    dates}. Raises EngineeringFailure on any failure; returns the per-window diffs."""
    w = wells[np.isfinite(wells["sd_fit_rel_m"].to_numpy(float))].reset_index(drop=True)
    ids = w["station_id"].astype(str).tolist()
    if len(ids) != K.N_WELLS or len(set(ids)) != K.N_WELLS:
        raise io.EngineeringFailure(f"wells_e02d: {len(ids)} wells with sd_fit_rel, not "
                                    f"{K.N_WELLS}")
    wi = w.set_index(w["station_id"].astype(str))
    mid = [str(s) for s in members["station_id"]]
    if sorted(mid) != sorted(ids):
        raise io.EngineeringFailure("members_wells.npz wells differ from wells_e02d.csv")
    M = np.asarray(members["members"], float)[:, [mid.index(s) for s in ids], :]
    if M.shape[0] != K.N_MEMBERS:
        raise io.EngineeringFailure(f"{M.shape[0]} members, not {K.N_MEMBERS}")
    mkeys = [month_key(d) for d in members["dates"]]
    groups = w["group"].astype(str).to_numpy()
    tau2 = np.array([float(decisions[c]["tau2_per_year2"])
                     if decisions.get(c, {}).get("include_drift") else 0.0
                     for c in w["county"]])
    anchor = L.lead_anchor(lead_definition, w["t_centre_year"].to_numpy(float))
    out, bad = {}, []
    for win, (first, last) in K.WINDOWS.items():
        a = arts[win].copy()
        a["m"] = [month_key(d) for d in a["date"]]
        months = month_list(first, last)
        r = {"rows": int(len(a))}
        if a.duplicated(["station_id", "m"]).any() or sorted(set(a["m"])) != months or \
                sorted(set(a["station_id"])) != sorted(ids) or len(a) != len(ids) * len(months):
            bad.append(f"{win}: not {len(ids)} wells x {first}..{last}")
            out[win] = r
            continue
        if not (lead_months(a["m"]) == a["lead_months"].to_numpy(int)).all():
            bad.append(f"{win}: lead_months is not months after {K.ORIGIN}")
        if (a["county"].astype(str) != a["station_id"].map(
                lambda s: K.COUNTY_CODE.get(s[:2], "other"))).any() or \
                (a["county"].astype(str) != a["station_id"].map(wi["county"])).any():
            bad.append(f"{win}: county differs from the station code / wells_e02d")
        if (a["group"].astype(str) != a["station_id"].map(wi["group"]).astype(str)).any():
            bad.append(f"{win}: group differs from wells_e02d.csv")
        piv = {c: a.pivot(index="station_id", columns="m", values=c).reindex(
            index=ids, columns=months).to_numpy(float)
            for c in ("sd_fit_rel_m", "sigma_ens_rel_m", "drift_m", "sigma_idio_m",
                      "half_width_m", "predicted_m")}
        sd_w = np.repeat(w["sd_fit_rel_m"].to_numpy(float)[:, None], len(months), axis=1)
        dt_y = L.year_frac(pd.DatetimeIndex([f"{m}-01" for m in months]))[None, :] - \
            anchor[:, None]
        ti = [mkeys.index(m) for m in months]
        ens = L.sigma_ens_rel(M[:, :, ti], groups)
        c = centres[win].copy()
        c["m"] = [month_key(d) for d in c["date"]]
        c = c[c["m"].isin(months)]
        if c.duplicated(["station_id", "m"]).any():
            bad.append(f"{win}: the centre file has duplicate station-months")
            cen = np.full_like(sd_w, np.nan)
        else:
            cen = c.pivot(index="station_id", columns="m", values="predicted_m").reindex(
                index=ids, columns=months).to_numpy(float)
        r.update({
            "sd_fit_rel_m": _maxdiff(piv["sd_fit_rel_m"], sd_w),
            "drift_m": _maxdiff(piv["drift_m"], np.sqrt(tau2[:, None] * dt_y ** 2)),
            "sigma_ens_rel_m": _maxdiff(piv["sigma_ens_rel_m"], ens),
            "sigma_idio_m": _maxdiff(piv["sigma_idio_m"], np.sqrt(
                piv["sd_fit_rel_m"] ** 2 + piv["sigma_ens_rel_m"] ** 2 + piv["drift_m"] ** 2)),
            "half_width_m": _maxdiff(piv["half_width_m"], K.Z90 * piv["sigma_idio_m"]),
            "predicted_m": _maxdiff(piv["predicted_m"], cen)})
        bad += [f"{win}: {k} differs by {v} m (> {K.TOL_M})" for k, v in r.items()
                if k != "rows" and not v <= K.TOL_M]
        out[win] = {k: (v if k == "rows" else float(f"{v:.3e}")) for k, v in r.items()}
    if bad:
        raise io.EngineeringFailure("identity check failed: " + "; ".join(bad))
    return {"ok": True, "tol_m": K.TOL_M, "windows": out}


# ----------------------------------------------------------------------------- ribbon
def _r(x):
    return None if x is None or not np.isfinite(x) else round(float(x), K.ROUND)


def ribbon(arts: dict, min_wells: int = K.MIN_ZONE_WELLS) -> dict:
    """{window: {months, label, decided_by, zones [{zone, county, n_wells, half_width_m}],
    not_drawn}}. No station id is carried."""
    out = {}
    for win, (first, last) in K.WINDOWS.items():
        a = arts[win].copy()
        a["m"] = [month_key(d) for d in a["date"]]
        months = month_list(first, last)
        zones, skipped = [], {"zones": 0, "wells": 0}
        for z in sorted(a["group"].astype(str).unique()):
            s = a[a["group"].astype(str) == z]
            n = int(s["station_id"].nunique())
            if n < min_wells:
                skipped["zones"] += 1
                skipped["wells"] += n
                continue
            hw = []
            for m in months:
                v = s.loc[s["m"] == m, "half_width_m"].to_numpy(float)
                v = v[np.isfinite(v)]
                hw.append(_r(np.median(v)) if len(v) >= min_wells else None)
            county = sorted(set(s["county"].astype(str)))
            zones.append({"zone": z, "county": county[0] if len(county) == 1 else "mixed",
                          "n_wells": n, "half_width_m": hw})
        drawn = win in K.DRAWN_RIBBON_WINDOWS
        out[win] = {"months": months, "drawn": drawn,
                    "label": dict(K.LABELS["ribbon_A"]) if drawn else None,
                    "decided_by": K.WINDOW_TEST[win], "zones": zones, "not_drawn": skipped}
    return out


# ----------------------------------------------------------------------------- shared
def member_county_dev(members: np.ndarray, counties, ti) -> dict:
    """{county: (M, T)}: per member, the median over the county's wells of the member's
    deviation from the member mean, on month indices ``ti``. ``members`` (M, W, T_all)."""
    X = np.asarray(members, float)[:, :, ti]
    a = X - X.mean(axis=0, keepdims=True)
    cs = np.asarray(counties)
    return {c: np.median(a[:, cs == c, :], axis=1) for c in K.COUNTIES}


def family_county_dep(mu: dict) -> dict:
    """{county: {family: (T,)}}: the median over the county's wells of mu_k (W, T)."""
    return {c: {k: np.median(np.asarray(mu[c][k], float), axis=0) for k in K.FAMILIES}
            for c in mu}


def hull(member_dev: np.ndarray, fam: dict, exclude=()) -> tuple[np.ndarray, np.ndarray]:
    """(lo, hi) per month: hull of 0, the member part and the families not in ``exclude``."""
    md = np.asarray(member_dev, float)
    parts = [np.zeros(md.shape[1]), md.min(axis=0), md.max(axis=0)]
    parts += [np.asarray(fam[k], float) for k in K.FAMILIES if k not in exclude]
    s = np.vstack(parts)
    return s.min(axis=0), s.max(axis=0)


def shared_range(member_dev: dict, fam: dict, months) -> dict:
    out = {}
    for c in K.COUNTIES:
        lo, hi = hull(member_dev[c], fam[c])
        md = np.asarray(member_dev[c], float)
        out[c] = {"months": list(months), "lo_m": [_r(x) for x in lo],
                  "hi_m": [_r(x) for x in hi],
                  "member_lo_m": [_r(x) for x in md.min(axis=0)],
                  "member_hi_m": [_r(x) for x in md.max(axis=0)],
                  "family_m": {k: [_r(x) for x in fam[c][k]] for k in K.FAMILIES}}
    return out


def _inside(x, lo, hi) -> np.ndarray:
    x = np.asarray(x, float)
    return (x >= lo) & (x <= hi)


def lofo(member_dev: dict, fam: dict, months) -> dict:
    """Leave-one-family-out coverage per county (design-time, model-only); window months
    only (U7 guard)."""
    ms = u7_guard(months, "shared")
    out = {}
    for c in K.COUNTIES:
        per = {}
        for k in K.FAMILIES:
            lo, hi = hull(member_dev[c], fam[c], exclude=(k,))
            h = _inside(fam[c][k], lo, hi)
            if len(h) != len(ms):
                raise io.EngineeringFailure(f"{c} {k}: {len(h)} months, not {len(ms)}")
            per[k] = {"inside": int(h.sum()), "n": int(len(h)),
                      "coverage": round(float(h.mean()), K.ROUND)}
        inside = sum(v["inside"] for v in per.values())
        n = sum(v["n"] for v in per.values())
        grp = {}
        for g, ks in K.FB_GROUPS.items():
            lo, hi = hull(member_dev[c], fam[c], exclude=ks)
            hs = [_inside(fam[c][k], lo, hi) for k in ks]
            gi, gn = int(sum(h.sum() for h in hs)), int(sum(len(h) for h in hs))
            grp[g] = {"inside": gi, "n": gn, "coverage": round(gi / gn, K.ROUND)}
        out[c] = {"per_family": per,
                  "pooled": {"inside": inside, "n": n, "coverage": round(inside / n, K.ROUND)},
                  "min_family": min(v["coverage"] for v in per.values()),
                  "fb_groups_report_only": grp}
    return {"months": ms, "kind": "design-time, model-only; no gate, no status mapping",
            "counties": out}


# ----------------------------------------------------------------------------- assemble
def assemble(arts: dict, wells: pd.DataFrame, decisions: dict, lead_definition: str,
             members: dict, centres: dict, mu: dict) -> dict:
    """Pure given the inputs: identity check, ribbon, shared range and LOFO (no IO)."""
    ident = identity_check(arts, wells, decisions, lead_definition, members, centres)
    w = wells[np.isfinite(wells["sd_fit_rel_m"].to_numpy(float))].reset_index(drop=True)
    ids = w["station_id"].astype(str).tolist()
    mid = [str(s) for s in members["station_id"]]
    M = np.asarray(members["members"], float)[:, [mid.index(s) for s in ids], :]
    mkeys = [month_key(d) for d in members["dates"]]
    months = month_list(*K.WINDOWS[K.SHARED_WINDOW])
    ti = [mkeys.index(m) for m in months]
    mdev = member_county_dev(M, w["county"].astype(str).to_numpy(), ti)
    fam = family_county_dep(mu)
    for c in K.COUNTIES:
        for k in K.FAMILIES:
            if np.asarray(fam[c][k]).shape != (len(months),):
                raise io.EngineeringFailure(f"{c} {k}: family part is not window A")
    out = {"id": K.PB_ID, "version": K.VERSION, "identity": ident,
           "ribbon": ribbon(arts),
           "shared": {"window": K.SHARED_WINDOW, "label": dict(K.LABELS["shared"]),
                      "what": dict(K.LABELS["shared_what"]),
                      "after_window": dict(K.LABELS["shared_end"]),
                      "counties": shared_range(mdev, fam, months)},
           "lofo": lofo(mdev, fam, months),
           "dropped": dict(K.DROPPED),
           "u7": {"relative_coverage_computed": False,
                  "coverage_months": months,
                  "observation_inputs": 0},
           "wells_by_county": {c: int((w["county"] == c).sum()) for c in K.COUNTIES}}
    no_station_ids(out, ids)
    return out


def summary_checks(summ: dict, e02d_doc: dict, dr_doc: dict) -> tuple[dict, str]:
    """(decisions, lead_definition) of E02d's artefact after the checks: summary.json's
    decisions equal E02d's registered binds.dr.decisions, and its lead_definition equals
    DR.relative_drift.v1's registered lead_definition (no fallback)."""
    dec = e02d_doc["binds"]["dr"]["decisions"]
    if summ.get("decisions") != dec:
        raise io.EngineeringFailure("summary.json decisions differ from E02d's registered dr")
    lead = dr_doc.get("lead_definition")
    if lead not in L.LEAD_DEFINITIONS or summ.get("lead_definition") != lead:
        raise io.EngineeringFailure(f"summary.json lead_definition "
                                    f"{summ.get('lead_definition')!r} != DR's registered "
                                    f"{lead!r}")
    return dec, lead


def _attempt(rec: dict) -> None:
    p = io.repo_path(K.ATTEMPTS)
    p.parent.mkdir(parents=True, exist_ok=True)
    rec = {"time": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), **rec}
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, sort_keys=True, default=str) + "\n")


def run(twin_dir, out_dir=None, prereg_dir=None) -> int:
    out_dir = Path(out_dir) if out_dir else Path(twin_dir) / K.OUT_DIR[len(K.TWIN_PREFIX):]
    doc, sha = io.registered(K.PB_ID, prereg_dir, frozen=K.FROZEN)
    if doc.get("design") != json.loads(json.dumps(design_spec())):
        raise io.Refused("the registered design differs from build_pb.design_spec()")
    for rid, want in doc["binds"]["registrations"].items():
        _, got = io.registered(rid, prereg_dir)
        if got != want:
            raise io.Refused(f"{rid}: registered sha256 {got} != bound {want}")
    want = doc["binds"]["files"]
    assert_no_observation_inputs(want)
    bad = sha_problems(file_shas(want, twin_dir), want)
    if bad:
        raise io.Refused(f"bound inputs differ: {bad}")
    target = out_dir / K.OUT_JSON
    if target.exists():
        raise io.Refused(f"{target} exists (the band is written once)")
    e02d, _ = io.registered(K.E02D_ID, prereg_dir)
    dr_doc, _ = io.registered(K.DR_ID, prereg_dir)
    summ = json.loads(resolve(K.SUMMARY, twin_dir).read_text(encoding="utf-8"))
    decisions, lead = summary_checks(summ, e02d, dr_doc)
    arts = {w: read_sigma(resolve(K.SIGMA[w], twin_dir)) for w in K.WINDOWS}
    centres = {w: pd.read_csv(resolve(K.CENTRE[w], twin_dir), dtype={"station_id": str},
                              usecols=["station_id", "date", "predicted_m"])
               for w in K.WINDOWS}
    wells = pd.read_csv(io.repo_path(K.WELLS_E02D), dtype={"station_id": str})
    with np.load(io.repo_path(K.MEMBERS_NPZ), allow_pickle=False) as z:
        members = {k: z[k] for k in ("members", "station_id", "dates")}
    import fb_inputs as FI

    w = wells[np.isfinite(wells["sd_fit_rel_m"].to_numpy(float))]
    by = {c: {"ids": w.loc[w["county"] == c, "station_id"].astype(str).tolist()}
          for c in K.COUNTIES}
    mu, _, _ = FI.load_mu(by)
    out = assemble(arts, wells, decisions, lead, members, centres, mu)
    out["declaration"] = {"id": K.PB_ID, "sha256": sha}
    out["inputs_sha256"] = dict(sorted(want.items()))
    out["fb_status"], out["exs_status"] = K.FB_STATUS, K.EXS_STATUS
    no_station_ids(out, w["station_id"].astype(str).tolist())
    osha = io.write_once(target, out)
    print(json.dumps({"written": str(target), "sha256": osha}, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hashes", help="sha256 of every bound input (no value is read)")
    h.add_argument("--twin-dir", type=Path, required=True)
    b = sub.add_parser("build")
    b.add_argument("--twin-dir", type=Path, required=True)
    b.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    if a.cmd == "hashes":
        want = bound_files()
        now = file_shas(want, a.twin_dir)
        bad = sha_problems(now, want)
        print(json.dumps({"files": now, "problems": bad or "none"}, indent=1))
        return 1 if bad else 0
    try:
        rc = run(a.twin_dir, a.out)
        _attempt({"cmd": "build", "outcome": "written"})
        return rc
    except io.Refused as exc:
        kind = "engineering_failure" if isinstance(exc, io.EngineeringFailure) else "refused"
        _attempt({"cmd": "build", "outcome": kind, "error": str(exc)})
        print(f"REFUSED ({kind}): {exc}")
        return 1
    except Exception as exc:  # noqa: BLE001 - every crash is logged before it propagates
        _attempt({"cmd": "build", "outcome": "crash", "error": repr(exc),
                  "trace": traceback.format_exc()[-2000:]})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
