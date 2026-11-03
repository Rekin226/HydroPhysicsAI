"""Constants and pure functions of round-8 rank 5 (uncertainty bands): U1.ext_transfer_power.v1
then E02x.ext_transfer_2026-12.v1, a departure-transfer relative band test at N3F's 66
never-calibrated external wells.

Nothing here opens a file. The scripts beside this module read the frozen model output, the
registrations and (for the lead, logged) the pre-origin heads; they are bound by sha256.

The transferred (departure) estimand. At an external well i of DR group g (DR's frozen
grouping of the 158 E02 wells: zone partition in both counties, groups merged by DR's rule):

    r_v[t] = observed - predicted at the 158 E02 wells (E02's frozen centre, its datum)
    D_g[t] = median{ r_v[t] : v an E02 well of group g reporting in month t }
             (defined only when >= MIN_DONORS donors report: E02d's min_others)
    e_i[t] = observed_i[t] - predicted_i[t] - D_g[t]

with predicted_i = N3F's registered centre (the frozen forecast cf2c7154 plus the well's own
pre-origin datum). The external well is never one of the donors, so this is E02d's
leave-one-out median with the external well left out by construction. The same operator
builds the in-sample spread (``sd_fit_rel``, pre-origin months), the relative member spread
and the score, so the band and the test measure the same quantity:

    sigma_idio_i[t] = sqrt(sd_fit_rel_i^2 + sigma_ens_rel_i[t]^2 + tau_c^2 dt_i[t]^2)

tau_c frozen from DR.relative_drift.v1's report (sha 32025941), entering where DR's 10 % rule
triggered (E02d's rule); dt from DR's registered lead definition.
"""
from __future__ import annotations

import datetime as dt
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent), str(HERE.parent / "e02d")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import e02dlib as D  # noqa: E402  (E02d's frozen library: imported, never edited)

UTC = dt.timezone.utc

# ----------------------------------------------------------------------------- identities
POWER_ID = "U1.ext_transfer_power.v1"
POWER_WP = "U1"
E02X_WP = "E02x"
U1D_ID = D.POWER_ID                      # U1.decomp_power.v1 (registered, sha 9467e8af)
DR_ID = D.DR_ID
DR_UNIT = D.DR_UNIT
DR_REPORT_SHA = "32025941"               # prefix of DR's report.json sha256 (the spec's tau)
E02C_ID = "E02c.prospective_2026-11.v1"
E02D_ID = D.E02D_ID
E02D_TEST = D.E02D_TEST
N3F_ID = "N3F.ext_level.prospective_2026-11.v1"
N3F_UNIT = "N3F:ext_heads_2026-11_2027-10"
N3F_DIR = "results/twin/prospective_2026-11_ext"
N3F_PRED_SHA_PREFIX = "f54e60fc"         # N3F's predictions.csv as FV.ext.v1 logged it
FV_EXT_ID = "FV.ext.v1"
FV_EXT_LOOK_UNIT = "FV:ext_preorigin_derived_N3Fwells"
GOV_ID = "GOV.heads_window_product.v1"
RULINGS = "results/dataprep/rulings/user_rulings_2026-11-03.json"
RULING_ID = "R8.3"
E02_DIR = D.E02_DIR
FORWARD = D.FORWARD
MODEL_SHA = D.MODEL_SHA
DR_DIR = f"results/dataprep/reports/{D.DR_ID}"

# ----------------------------------------------------------------------------- wells
COUNTY = D.COUNTY
GATED = ("Yunlin", "Changhua")           # gate candidates (Changhua only if its power passes)
REPORT_ONLY = ("Nantou",)                # user ruling R8.3: report-only, no pooling
EXPECTED_EXT = {"Yunlin": 37, "Changhua": 23, "Nantou": 6}   # N3F kept_by_county (66)
N_EXT = sum(EXPECTED_EXT.values())
N_GATED = EXPECTED_EXT["Yunlin"] + EXPECTED_EXT["Changhua"]   # 60: the unit's real count
N_DONORS = D.N_UNIT_WELLS                # 158
MIN_DONORS = D.MIN_OTHERS                # E02d min_others (5)
MIN_FIT_MONTHS_SD = D.MIN_FIT_MONTHS_SD  # 12

# ----------------------------------------------------------------------------- calendar
ORIGIN = D.ORIGIN
FIT = D.FIT
PRE_SCOPE = "heads:2012-01..2022-12"
PRE_UNIT = f"E02x:ext_preorigin_2012-01_2022-12_wells{N_EXT}"
WINDOWS = {
    "primary": {"first": "2026-12", "last": "2027-10", "months": 11,
                "deadline": dt.datetime(2026, 11, 30, 16, 0, tzinfo=UTC),
                "not_before": "2028-01-15", "no_later_than": "2028-01-31",
                "id": "E02x.ext_transfer_2026-12.v1",
                "role": "primary window"},
    "fallback": {"first": "2027-01", "last": "2027-10", "months": 10,
                 "deadline": dt.datetime(2026, 12, 31, 16, 0, tzinfo=UTC),
                 "not_before": "2028-01-15", "no_later_than": "2028-01-31",
                 "id": "E02x.ext_transfer_2027-01.v1",
                 "role": "pre-declared fallback (used only if the primary is not registered "
                         "before its deadline; its own power run)"},
}


def tag(win: str) -> str:
    return WINDOWS[win]["first"]


def prereg_id(win: str) -> str:
    return WINDOWS[win]["id"]


def test_id(win: str) -> str:
    return f"E02xP.ext_transfer_{tag(win)}"


def nantou_test_id(win: str) -> str:
    return f"E02xR.nantou_report_only_{tag(win)}"


def unit_id(win: str, n: int = N_GATED) -> str:
    w = WINDOWS[win]
    return f"E02x:ext_heads_{w['first']}_{w['last']}_wells{n}"


def nantou_unit_id(win: str, n: int = EXPECTED_EXT["Nantou"]) -> str:
    w = WINDOWS[win]
    return f"E02x:ext_heads_{w['first']}_{w['last']}_nantou{n}"


def scope(win: str) -> str:
    w = WINDOWS[win]
    return f"heads:{w['first']}..{w['last']}"


def months(win: str) -> pd.DatetimeIndex:
    w = WINDOWS[win]
    return D.month_index(w["first"], w["last"])


# ----------------------------------------------------------------------------- the rules
Z90 = D.Z90
TARGET, GATE_LEVEL = D.TARGET, D.GATE_LEVEL
POINT_WINDOW = D.POINT_WINDOW
MIN_MONTHS = D.MIN_MONTHS
MIN_WELLS_COUNTY = D.MIN_WELLS_COUNTY    # 20; for Changhua the 20-of-23 floor
MEMBER_TOL_M = D.MEMBER_MEAN_TOL_M       # 1e-3 m
FALSE_FAIL_MAX = 0.12                    # at a true 0.90 (stricter than U1.decomp's 0.20)
FF_AT = 0.90
POWER_MIN = 0.80
POWER_AT = 0.75
#: donor and external whole-window dropout: U1.decomp_power.v1's registered grid (lead, S2)
DROPOUT = (0.0, 0.2, 0.4)
DONOR_DROPOUT = DROPOUT
MIN_BREAKDOWN_WELLS = 5           # by-zone / by-group coverage suppressed below this
GROUPINGS = ("zone", "layer", "county")


def county_of(sid) -> str:
    return D.county_of(sid)


# ----------------------------------------------------------------------------- groups
def _parts(label: str) -> tuple[str, list[str]]:
    c, _, rest = str(label).partition(":")
    return c, (rest.split("+") if rest else [])


def map_to_group(county: str, raw: str, labels) -> str | None:
    """The merged partition label (``County:a+b``) of ``labels`` holding raw group ``raw``
    (a zone or a layer group); None if none does."""
    for lab in sorted(set(map(str, labels))):
        c, parts = _parts(lab)
        if c == county and raw in parts:
            return lab
    return None


def ext_groups(table: pd.DataFrame, donor_groups: dict, donor_subgroups: dict,
               grouping: dict) -> pd.DataFrame:
    """Assign each external well its DR group, sub-group and donor key (pure).

    ``table``: station_id, county, zone (prox/mid/dist), layer_group (L1/L2/L34).
    ``donor_groups`` / ``donor_subgroups``: {county: labels of DR's groups / sub-groups};
    ``grouping``: {county: DR's chosen grouping}. Gated counties take DR's group of their
    county (donors = the E02 wells of that group). Nantou (no DR county) is report-only: its
    donors are the E02 wells of the same raw column zone in either county (declared)."""
    t = table.copy()
    g, s, k, role = [], [], [], []
    for _, r in t.iterrows():
        c = str(r["county"])
        if c in GATED:
            how = grouping[c]
            if how == "county":
                gl = c
            else:
                raw = r["zone"] if how == "zone" else r["layer_group"]
                gl = map_to_group(c, raw, donor_groups[c])
            other_raw = r["layer_group"] if grouping[c] == "zone" else r["zone"]
            sl = map_to_group(c, other_raw, donor_subgroups[c])
            g.append(gl)
            s.append(sl)
            k.append(f"group:{gl}" if gl else None)
            role.append("gate_candidate")
        else:
            g.append(f"{c}:{r['zone']}")
            s.append(f"{c}:{r['layer_group']}")
            k.append(f"zone:{r['zone']}")
            role.append("report_only")
    t["group"], t["subgroup"], t["donor_key"], t["role"] = g, s, k, role
    return t


def donor_masks(ext: pd.DataFrame, donors: pd.DataFrame) -> dict[str, np.ndarray]:
    """{donor_key: bool mask over the donors' rows}: 'group:<label>' = DR group, 'zone:<z>' =
    the raw zone in either county (report-only Nantou)."""
    out = {}
    for key in sorted(set(ext["donor_key"].dropna())):
        kind, _, v = key.partition(":")
        col = "group" if kind == "group" else "zone"
        out[key] = (donors[col].astype(str) == v).to_numpy()
    return out


# ----------------------------------------------------------------------------- transfer
def departure(R_donor: np.ndarray, masks: dict, keys, min_donors: int = MIN_DONORS
              ) -> np.ndarray:
    """Median over each external well's donors (axis -2 of ``R_donor`` (..., Wd, T), NaN =
    missing) per month; NaN where fewer than ``min_donors`` donors report. Returns
    (..., We, T) in the order of ``keys`` (one donor key per external well)."""
    R = np.asarray(R_donor, dtype=float)
    keys = list(keys)
    out = np.full(R.shape[:-2] + (len(keys), R.shape[-1]), np.nan)
    cache = {}
    for j, k in enumerate(keys):
        if k is None or k not in masks:
            continue
        if k not in cache:
            sub = R[..., masks[k], :]
            n = np.isfinite(sub).sum(axis=-2)
            if sub.shape[-2] == 0:
                med = np.full(n.shape, np.nan)
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN months
                    med = np.nanmedian(sub, axis=-2)
            cache[k] = np.where(n >= max(1, int(min_donors)), med, np.nan)
        out[..., j, :] = cache[k]
    return out


def transfer_relative(R_ext: np.ndarray, R_donor: np.ndarray, masks: dict, keys,
                      min_donors: int = MIN_DONORS) -> np.ndarray:
    """e = R_ext - departure(R_donor) (NaN where either is undefined)."""
    return np.asarray(R_ext, dtype=float) - departure(R_donor, masks, keys, min_donors)


def sigma_ens_rel(M_ext: np.ndarray, M_donor: np.ndarray, masks: dict, keys,
                  min_donors: int = MIN_DONORS) -> np.ndarray:
    """Relative member spread at the external wells (We, T): root mean square over members m
    of a_m(i) - median over i's donors of a_m(v), a_m = member m minus the member mean
    (population form, as E02d). ``M_ext`` (M, We, T), ``M_donor`` (M, Wd, T)."""
    Me = np.asarray(M_ext, dtype=float)
    Md = np.asarray(M_donor, dtype=float)
    if Me.shape[0] != Md.shape[0]:
        raise ValueError("members differ between external wells and donors")
    ae = Me - Me.mean(axis=0, keepdims=True)
    ad = Md - Md.mean(axis=0, keepdims=True)
    rel = transfer_relative(ae, ad, masks, keys, min_donors)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.sqrt(np.nanmean(rel ** 2, axis=0))


def sigma_idio(sd_fit_rel, ens_rel, tau2, dt_years) -> np.ndarray:
    """E02d's formula, per well-month (We, T)."""
    return D.sigma_idio(sd_fit_rel, ens_rel, tau2, dt_years)


def transfer_hits(R_ext, R_donor, sigma, masks, keys, min_donors: int = MIN_DONORS,
                  z: float = Z90) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(hits, below, e): hit = |e| <= z sigma (NaN where e or sigma is undefined)."""
    e = transfer_relative(R_ext, R_donor, masks, keys, min_donors)
    ok = np.isfinite(e) & np.isfinite(sigma)
    hw = z * np.asarray(sigma, float)
    hits = np.where(ok, (np.abs(e) <= hw).astype(float), np.nan)
    below = np.where(ok, (e < -hw).astype(float), np.nan)
    return hits, below, e


def county_verdict(hits, mask, below=None) -> dict:
    """E02d's gate, exactly (e02dlib.county_verdict): wells with >= 6 scored months, >= 20
    such wells, coverage3 gate-3 upper90 with clusters = wells AND point coverage in
    [0.85, 0.95]."""
    return D.county_verdict(hits, mask, below)


def breakdown(hits: np.ndarray, labels, mask=None,
              min_wells: int = MIN_BREAKDOWN_WELLS) -> dict:
    """Report-only coverage by label (zone or group): wells, rows, share of hits; the
    coverage number is suppressed (None, labelled) below ``min_wells`` scored wells."""
    H = np.asarray(hits, dtype=float)
    lab = np.asarray(labels).astype(str)
    m = np.ones(len(lab), bool) if mask is None else np.asarray(mask, bool)
    out = {}
    for g in sorted(set(lab[m])):
        sel = m & (lab == g)
        ok = np.isfinite(H[sel])
        nw = int(ok.any(axis=1).sum())
        rec = {"wells": nw, "rows": int(ok.sum()), "coverage": None}
        if nw >= min_wells:
            rec["coverage"] = float(np.nanmean(H[sel]))
        else:
            rec["suppressed"] = f"< {min_wells} scored wells: coverage not reported"
        out[g] = rec
    return out


# ----------------------------------------------------------------------------- arithmetic
def p_survive(n: int, floor: int, dropout: float) -> float:
    """P(at least ``floor`` of ``n`` wells report) when each drops out for the whole window
    independently with probability ``dropout`` (binomial tail; constants only, no data)."""
    q = 1.0 - float(dropout)
    return float(sum(math.comb(n, k) * q ** k * (1.0 - q) ** (n - k)
                     for k in range(int(floor), n + 1)))


def arithmetic_bounds(counts: dict | None = None, dropout=DROPOUT,
                      floor: int = MIN_WELLS_COUNTY) -> dict:
    """Per gated county: P(evaluable) at each declared external dropout and the resulting
    upper bound on the power rule's minimum (a not-evaluable replicate is never a fail, so
    P(FAIL) <= P(>= floor wells report)). A county whose bound is below POWER_MIN cannot pass
    the power rule by construction: it is report-only by arithmetic (lead ruling, B1)."""
    counts = counts or {c: EXPECTED_EXT[c] for c in GATED}
    out = {}
    for c, n in counts.items():
        p = {f"{d:g}": p_survive(int(n), floor, d) for d in dropout}
        b = min(p.values())
        out[c] = {"wells": int(n), "floor": int(floor),
                  "p_at_least_floor_by_dropout": {k: round(v, 6) for k, v in p.items()},
                  "power_upper_bound": round(b, 6),
                  "bound": f"power <= {b:.3f} x the conditional power at dropout "
                           f"{max(dropout):g}",
                  "report_only_by_arithmetic": bool(b < POWER_MIN)}
    return out


def drift_tau2(decisions: dict, county: str) -> float:
    """tau_c^2 where DR's 10 % rule triggered (E02d's rule), else 0; Nantou (no DR county,
    report-only): the larger of the two counties' triggered tau^2 (declared, conservative)."""
    def one(c):
        d = decisions.get(c) or {}
        t = d.get("tau2_per_year2")
        if not d.get("include_drift") or t is None or not math.isfinite(float(t)):
            return 0.0
        return float(t)

    if county in GATED:
        return one(county)
    return max(one(c) for c in GATED)


def lead_months(ms: pd.DatetimeIndex) -> np.ndarray:
    return ((ms.year - 2022) * 12 + ms.month - 12).to_numpy(int)


def frame_to_arrays(frame: pd.DataFrame, wells, ms, cols) -> dict:
    return D.frame_to_arrays(frame, list(wells), ms, tuple(cols))
