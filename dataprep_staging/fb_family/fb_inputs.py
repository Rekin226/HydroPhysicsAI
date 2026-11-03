"""Inputs of FB.family_power.v1, read only by the registered run (never before registration):

* U1.decomp_power.v1's frozen well table (DR.relative_drift.v1/wells_e02d.csv, sha f3174c48):
  station_id, county, group, subgroup, sd_fit_rel_m ONLY (pre-origin derived; U1's own input).
  Identity gate: per county, the rows with a finite sd_fit_rel_m, in file order, carry exactly
  U1's registered groups, sub-groups and relative scale (sd / county median, 6 decimals).
* E02d's registered window-A artefact (results/twin/e02d_relative/sigma_idio_A.csv, sha
  b70da8b0, E02d.decomposed_prospective_2026-12.v1 inputs; lead ruling B3): sigma_idio_m per
  well-month in metres, the worlds' and the likelihood's scale. Gate (units and lead
  consistency with E02d): 158 wells x the 11 window months, every row; lead_months = months
  after the 2022-12 origin (48..58); sigma_idio_m = sqrt(sd_fit_rel_m^2 + sigma_ens_rel_m^2 +
  drift_m^2) (1e-9 m, e02dlib.sigma_idio); sd_fit_rel_m equals wells_e02d.csv's; group equals
  U1's.
* FV's inputs (model-only; no head value): E02c's wells_158.csv (cell, layer), FV v1's
  obs_mask.csv (availability 0/1), FV v1's truth forwards fv_truth_{R0,Q0,R1c,X5R1}.npz
  (baseline and the 58 impulse arms), FV v1's fv_zr0.npz (report-only offset), E02's
  registered forward datum_gate_innov.npz (cf2c7154) and FV v1's anomaly_pool.csv
  (pump_electricity preorigin v3, design_time_evidence). mu_k = osse_fv.departure(h_k, h_ZR,
  mask) and the unit responses osse_fv.basis(h_k), both on 2026-12..2027-10 (FV's frozen
  functions, imported, never copied).

``input_sha`` hashes files only (no value is read); ``upstream_pin_problems`` compares those
hashes with the sha256 the upstream registrations recorded.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STAGING = HERE.parent
for _p in (str(HERE), str(STAGING), str(STAGING / "e02d"), str(STAGING / "e02c"),
           str(STAGING / "fv_decidability")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fb_io as io  # noqa: E402
import fb_lib as L  # noqa: E402

WELLS_E02D = "results/dataprep/reports/DR.relative_drift.v1/wells_e02d.csv"
WELLS_158 = "results/dataprep/reports/E02c/baselines/wells_158.csv"
OBS_MASK = "results/dataprep/reports/FV.decidability.v1/inputs/obs_mask.csv"
ANOMALY_POOL = "results/dataprep/reports/FV.decidability.v1/inputs/anomaly_pool.csv"
SIGMA_IDIO_A = "results/twin/e02d_relative/sigma_idio_A.csv"
ZR_FORWARD = "results/twin_forward/datum_gate_innov.npz"
ZR0_FORWARD = "results/twin_forward/fv_zr0.npz"
ORIGIN = pd.Timestamp("2022-12-01")
SIGMA_TOL_M = 1e-9


def truth_forward(tag: str) -> str:
    return f"results/twin_forward/fv_truth_{tag}.npz"


INPUT_FILES = {"wells_e02d": WELLS_E02D, "wells_158": WELLS_158, "obs_mask": OBS_MASK,
               "anomaly_pool": ANOMALY_POOL, "sigma_idio_A": SIGMA_IDIO_A,
               "zr": ZR_FORWARD, "zr0": ZR0_FORWARD,
               **{f"truth_{t}": truth_forward(t) for t in L.TRUTHS}}
#: sha256 the upstream registrations recorded (U1 design, FV v1 inputs_frozen and inputs,
#: FV v2 truths[].v1_forward, E02c score_e02c.MODEL_SHA, E02d inputs); fv_zr0.npz is recorded
#: by no registration: its sha256 was taken by the FB builder on 2026-11-03 (hash only)
UPSTREAM_SHA = {
    WELLS_E02D: "f3174c481827f22efb4a6c7436359e765414bff07fa9d8bd06a65a73c51a92b8",
    WELLS_158: "8ea086330f8a9ecf5ff76313c9dc3e617e8e500775d70d7cc180f077340ac7d3",
    OBS_MASK: "605a35de5d07cab40cef3a1e86fae719502ff3afcd0f6ea36310ccf5425d0c89",
    ANOMALY_POOL: "acab736764fd635ecd21eb62f77d26dd3ea8132a184678e7ac0a0fd1fc94f290",
    SIGMA_IDIO_A: "b70da8b0f2d340d9609e7f7423ab8961596aa21292b1f578c3bfbf89acd51988",
    ZR_FORWARD: "cf2c7154160d9e4626d1acc5b5c3abd7866e09c3539a9249c657dd6b39007228",
    ZR0_FORWARD: "a1833d5f861817223ebafb5a651444239c8877d21a4d4752e91dbbbd356172cf",
    truth_forward("R0"): "2b1363bec1581e76bf98f422a30a023cbc8d21f4dabba82256b5670d6d019fcf",
    truth_forward("Q0"): "1a6a1ef7d1c9d1c6fff21f0a2e26bd2fb8ad36e073ec36f03e96f6bde99d74b1",
    truth_forward("R1c"): "faff857e7ffae99d9cbcf5ed440546482c9d9d1276cf6fad5871a5426b87fdf1",
    truth_forward("X5R1"): "9d8e6311d5e1ece3dc54a4c40719a13e18b1a440cfe87770707649276c4c9566"}
UPSTREAM_SOURCE = {ZR0_FORWARD: "FB builder hash 2026-11-03 (no registration records it)"}
WELL_COLS = ("station_id", "county", "group", "subgroup", "sd_fit_rel_m")
SIGMA_COLS = ("station_id", "county", "group", "date", "lead_months", "sd_fit_rel_m",
              "sigma_ens_rel_m", "drift_m", "sigma_idio_m")
COUNTY_CODE = {"07": "Changhua", "09": "Yunlin"}


def input_sha() -> dict:
    """{repo-relative path: sha256} of every input (hashes only; no value is read)."""
    return {p: io.sha256(io.repo_path(p)) for p in INPUT_FILES.values()}


def _registered_sha(doc: dict, path: str) -> str | None:
    for r in doc.get("inputs", []):
        if isinstance(r, dict) and str(r.get("path", "")).endswith(path):
            return r.get("sha256")
    return None


def upstream_pin_problems(shas: dict, u1_doc: dict | None = None,
                          fv2_doc: dict | None = None, e02d_doc: dict | None = None
                          ) -> list[str]:
    bad = [f"{p}: sha256 {shas.get(p)} != pinned {h}" for p, h in UPSTREAM_SHA.items()
           if shas.get(p) != h]
    if u1_doc is not None:
        h = u1_doc["design"].get("wells_e02d_sha256")
        if shas.get(WELLS_E02D) != h:
            bad.append(f"wells_e02d.csv sha256 differs from U1's design ({h})")
    if fv2_doc is not None:
        for t in fv2_doc.get("truths", []):
            if shas.get(truth_forward(t["tag"])) != t["v1_forward"]["sha256"]:
                bad.append(f"truth {t['tag']}: forward sha256 differs from FV v2's v1_forward")
    if e02d_doc is not None:
        h = _registered_sha(e02d_doc, SIGMA_IDIO_A)
        if shas.get(SIGMA_IDIO_A) != h:
            bad.append(f"sigma_idio_A.csv sha256 differs from E02d's registration ({h})")
    return bad


# ----------------------------------------------------------------------------- wells
def u1_identity(design: dict, wells: pd.DataFrame) -> dict:
    """Per county (ids, sd_fit_rel) in U1's order; refuses unless groups, sub-groups and the
    relative scale equal U1's registered design."""
    out = {}
    for c in L.COUNTIES:
        dc = design["counties"][c]
        w = wells[(wells["county"] == c) & np.isfinite(wells["sd_fit_rel_m"])]
        s = w["sd_fit_rel_m"].to_numpy(float)
        if len(w) != dc["wells"]:
            raise io.EngineeringFailure(f"{c}: {len(w)} wells, U1 has {dc['wells']}")
        if w["group"].astype(str).tolist() != dc["groups"]:
            raise io.EngineeringFailure(f"{c}: groups differ from U1's")
        if w["subgroup"].astype(str).tolist() != dc["subgroups"]:
            raise io.EngineeringFailure(f"{c}: sub-groups differ from U1's")
        rel = (s / np.median(s)).round(6)
        if np.max(np.abs(rel - np.asarray(dc["u1_scale"], float))) > 5e-7:
            raise io.EngineeringFailure(f"{c}: relative scale differs from U1's")
        out[c] = {"ids": w["station_id"].astype(str).tolist(), "sd_fit_rel": s,
                  "groups": dc["groups"]}
    return out


def read_wells(path=WELLS_E02D) -> pd.DataFrame:
    return pd.read_csv(io.repo_path(path), usecols=list(WELL_COLS),
                       dtype={"station_id": str})


def sigma_scale(art: pd.DataFrame, ident: dict) -> dict:
    """{county: sigma_idio (W, 11)} in U1's order, after the B3 units/lead gate."""
    a = art.copy()
    a["m"] = pd.to_datetime(a["date"].astype(str)).dt.strftime("%Y-%m")
    if a.duplicated(["station_id", "m"]).any():
        raise io.EngineeringFailure("sigma_idio_A: duplicate (well, month)")
    if sorted(set(a["m"])) != sorted(L.WINDOW_MONTHS):
        raise io.EngineeringFailure("sigma_idio_A: months are not 2026-12..2027-10")
    mt = pd.to_datetime(a["m"] + "-01")
    lead = (mt.dt.year - ORIGIN.year) * 12 + mt.dt.month - ORIGIN.month
    if not (lead.to_numpy(int) == a["lead_months"].to_numpy(int)).all():
        raise io.EngineeringFailure("sigma_idio_A: lead_months is not months after 2022-12")
    comb = np.sqrt(a["sd_fit_rel_m"] ** 2 + a["sigma_ens_rel_m"] ** 2 + a["drift_m"] ** 2)
    if float(np.max(np.abs(comb - a["sigma_idio_m"]))) > SIGMA_TOL_M:
        raise io.EngineeringFailure("sigma_idio_A: sigma_idio_m != e02dlib.sigma_idio of its "
                                    "terms")
    out = {}
    for c, v in ident.items():
        sub = a[a["station_id"].isin(v["ids"])]
        if len(sub) != len(v["ids"]) * L.T_WIN:
            raise io.EngineeringFailure(f"{c}: sigma_idio_A lacks a well-month")
        piv = sub.pivot(index="station_id", columns="m", values="sigma_idio_m").reindex(
            index=v["ids"], columns=L.WINDOW_MONTHS)
        sd = sub.groupby("station_id")["sd_fit_rel_m"].agg(["min", "max"]).reindex(v["ids"])
        if float(np.max(np.abs(sd["min"].to_numpy() - v["sd_fit_rel"]))) > SIGMA_TOL_M or \
                float(np.max(np.abs(sd["max"].to_numpy() - v["sd_fit_rel"]))) > SIGMA_TOL_M:
            raise io.EngineeringFailure(f"{c}: sigma_idio_A sd_fit_rel_m != wells_e02d.csv's")
        grp = sub.groupby("station_id")["group"].first().reindex(v["ids"]).astype(str)
        if grp.tolist() != list(v["groups"]):
            raise io.EngineeringFailure(f"{c}: sigma_idio_A groups differ from U1's")
        S = piv.to_numpy(float)
        if not np.isfinite(S).all() or (S <= 0).any():
            raise io.EngineeringFailure(f"{c}: sigma_idio_A is not positive and finite")
        out[c] = S
    return out


# ----------------------------------------------------------------------------- departures
def _check_meta(name: str, meta: dict, heads: np.ndarray, *, baseline_only: bool,
                impulses: bool = False) -> int:
    """Index of the baseline scenario after axis checks (engineering)."""
    import osse_fv as FV

    names = [str(x) for x in meta["scenario_names"]]
    if "baseline" not in names or (baseline_only and names != ["baseline"]):
        raise io.EngineeringFailure(f"{name}: scenarios {names[:3]}... lack 'baseline'")
    if impulses and names[1:1 + FV.HORIZON] != FV.impulse_names():
        raise io.EngineeringFailure(f"{name}: scenarios 1..58 are not FV's impulse arms")
    dates = pd.DatetimeIndex(pd.to_datetime(meta["dates"])).to_period("M").to_timestamp()
    if len(dates) < FV.T_RUN or dates[0] != pd.Timestamp(f"{FV.AXIS_START}-01"):
        raise io.EngineeringFailure(f"{name}: dates do not start {FV.AXIS_START} with >= "
                                    f"{FV.T_RUN} months")
    if dates[FV.WIN[0] + L.FV_WINDOW_SLICE[0]] != pd.Timestamp(f"{L.WINDOW[0]}-01") or \
            dates[FV.WIN[0] + L.FV_WINDOW_SLICE[1] - 1] != pd.Timestamp(f"{L.WINDOW[1]}-01"):
        raise io.EngineeringFailure(f"{name}: the window does not map to {L.WINDOW}")
    if not np.isfinite(heads[:, :, :FV.T_RUN]).all():
        raise io.EngineeringFailure(f"{name}: non-finite heads at a well")
    return names.index("baseline")


def window_departure(h_k: np.ndarray, h_ref: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """FV's departure (frozen osse_fv.departure) restricted to 2026-12..2027-10."""
    import osse_fv as FV

    d = FV.departure(h_k, h_ref, mask)
    return d[:, L.FV_WINDOW_SLICE[0]:L.FV_WINDOW_SLICE[1]]


def window_basis(heads: np.ndarray) -> np.ndarray:
    """FV's unit responses (frozen osse_fv.basis) on 2026-12..2027-10: (58, W, 11)."""
    import osse_fv as FV

    G = FV.basis(heads)
    return G[:, :, FV.WIN[0] + L.FV_WINDOW_SLICE[0]:FV.WIN[0] + L.FV_WINDOW_SLICE[1]]


def load_mu(by_county: dict) -> tuple[dict, dict, dict]:
    """({county: {truth: mu (W, T)}}, {county: {truth: G (58, W, T)}}, report-only
    provenance). Model-only reads."""
    import osse_fv as FV

    w158 = pd.read_csv(io.repo_path(WELLS_158), usecols=["station_id", "cell", "layer"],
                       dtype={"station_id": str}).set_index("station_id")
    ids = sorted(w158.index)
    w158 = w158.loc[ids]
    if (w158["cell"] < 0).any() or (w158["layer"] < 0).any():
        raise io.EngineeringFailure("a well has no forward cell or layer")
    pos = {s: i for i, s in enumerate(ids)}
    for c, v in by_county.items():
        miss = [s for s in v["ids"] if s not in pos]
        if miss:
            raise io.EngineeringFailure(f"{c}: {len(miss)} U1 wells absent from wells_158")
    mask = FV.read_mask(io.repo_path(OBS_MASK), ids)
    cells, layers = w158["cell"].to_numpy(int), w158["layer"].to_numpy(int)
    zmeta, zh = FV.well_heads(io.repo_path(ZR_FORWARD), cells, layers)
    h_zr = zh[_check_meta("zr", zmeta, zh, baseline_only=False)]
    del zh
    z0meta, z0h = FV.well_heads(io.repo_path(ZR0_FORWARD), cells, layers)
    h_zr0 = z0h[_check_meta("zr0", z0meta, z0h, baseline_only=True)]
    del z0h
    full, basis = {}, {}
    for t in L.TRUTHS:
        meta, h = FV.well_heads(io.repo_path(truth_forward(t)), cells, layers)
        if _check_meta(t, meta, h, baseline_only=False, impulses=True) != 0:
            raise io.EngineeringFailure(f"{t}: baseline is not scenario 0")
        full[t] = window_departure(h[0], h_zr, mask)
        basis[t] = window_basis(h)
        del h
    offset = window_departure(h_zr0, h_zr, mask)       # FV's ZR0 -> ZR part, report-only
    mu, G, prov = {}, {}, {"zr0_to_zr_offset_rms_m": {}, "report_only": True}
    for c, v in by_county.items():
        ix = [pos[s] for s in v["ids"]]
        mu[c] = {t: full[t][ix] for t in L.TRUTHS}
        G[c] = {t: basis[t][:, ix] for t in L.TRUTHS}
        prov["zr0_to_zr_offset_rms_m"][c] = float(np.sqrt(np.mean(offset[ix] ** 2)))
    return mu, G, prov


def load_pool() -> np.ndarray:
    import osse_fv as FV

    return FV.pool_matrix(pd.read_csv(io.repo_path(ANOMALY_POOL)))


def load_real(design: dict) -> tuple[dict, dict]:
    """The registered run's inputs ({"pool", county: {mu, scale, G, ids}}) and provenance.
    Refuses unless every input file has the design's sha256."""
    now = input_sha()
    if now != design["input_sha256"]:
        bad = sorted(k for k in set(now) | set(design["input_sha256"])
                     if now.get(k) != design["input_sha256"].get(k))
        raise io.EngineeringFailure(f"input sha256 differ from the design: {bad}")
    ident = u1_identity(design, read_wells())
    scale = sigma_scale(pd.read_csv(io.repo_path(SIGMA_IDIO_A), usecols=list(SIGMA_COLS),
                                    dtype={"station_id": str}), ident)
    mu, G, prov = load_mu(ident)
    out = {"pool": load_pool()}
    for c in L.COUNTIES:
        out[c] = {"mu": mu[c], "scale": scale[c], "G": G[c], "ids": ident[c]["ids"]}
    return out, prov


def county_ids(path=WELLS_158) -> dict:
    """{county: sorted station ids} from wells_158.csv's station_id column only (metadata;
    county from the station code prefix, as score_e02c.county_of); for the BF draft's unit
    membership. Refuses unless the counts are U1's (74, 84)."""
    s = pd.read_csv(io.repo_path(path), usecols=["station_id"],
                    dtype={"station_id": str})["station_id"].astype(str)
    out = {c: sorted(x for x in s if COUNTY_CODE.get(x[:2]) == c) for c in L.COUNTIES}
    for c in L.COUNTIES:
        if len(out[c]) != L.EXPECTED_WELLS[c]:
            raise io.EngineeringFailure(f"{c}: {len(out[c])} wells, expected "
                                        f"{L.EXPECTED_WELLS[c]}")
    return out
