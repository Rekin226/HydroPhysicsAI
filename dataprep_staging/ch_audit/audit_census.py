"""Census-only county audit (round 7 rank 8): bimonthly even split and duty-cap retention in
the Changhua forcing, with Yunlin beside it. No head, leveling or township value is read.

Inputs (allowed use model_design on both MANIFESTs):

* ``pump_electricity/model_census/v1`` (census.parquet, kwh.parquet): the forcing R1c and every
  Changhua rung was calibrated on. Retention is measured with ``pumping.clean_census``'s own
  semantics over 2012-01..2022-12 (the model's window), and the totals are checked against
  ``clean_census`` itself (an engineering check: a difference stops the run).
* ``pump_electricity/preorigin/v3`` (meters_<county>.parquet, pumps_<county>.npz): billing
  phase per meter (phase_code, else phase_detected where the code is 'unknown', C0's
  convention) and the cleaned per-pump monthly kWh. Even-split pairs are re-split by the
  county-class shape of the monthly-billed meters (``audit_lib.reshape_pairs``).

Both MANIFESTs must exist and allow model_design (fail closed). With ``--log-look`` the
model_design look is recorded BEFORE any product is read (an engineering rerun logs another
look; a crash leaves a look with no audit.json, the conservative side).

Outputs: ``<out>/audit.json`` (written once), holding the decisions D1 (the fixed Changhua
forcing version for every Changhua route), D1b (duty-cap member), D2 (2-month smoothing for
seasonal-amplitude gates) and the resolution of the '~8 % kept' memory figure. The decision
rules are fixed in ``audit_lib`` before any run. Log it as a model_design look
(``--log-look``, the lead; ``look-spec`` prints the call).

Usage (repository root, the lead; the default product paths need results/dataprep to resolve,
i.e. the main checkout or a frozen worktree whose results/ is the shared results):
    dataprep_staging/tools/cpu_slot.sh ~/miniforge3/envs/hydro/bin/python \\
        dataprep_staging/ch_audit/audit_census.py run --out <report dir> [--log-look]
    python dataprep_staging/ch_audit/audit_census.py look-spec
    python dataprep_staging/ch_audit/audit_census.py smoke --out <scratch dir>   (synthetic)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(REPO), str(REPO / "dataprep_staging"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import audit_lib as L  # noqa: E402

AUDIT_ID = "CH_audit.census_bookkeeping.v1"
X = "dataprep_staging/ch_audit"
MODEL_CENSUS = "results/dataprep/pump_electricity/model_census/v1"
PREORIGIN = "results/dataprep/pump_electricity/preorigin/v3"
COUNTIES = ("changhua", "yunlin")
T0, T1 = "2012-01-01", "2023-01-01"
WINDOW = ("2012-01", "2022-12")
YEARS = 11.0
CLASS_SERIES = ("all", "irrigation", "all_nodomestic")
REFERENCE_GWH_YR = {"preorigin_v3_card_changhua_in_grid": 187.36,
                    "model_census_v1_validation_added_changhua_plus_chiayi": 188.04}
LOOK_UNIT = "forcing.census.CH_audit"
LOOK_SCOPE = "pump_electricity:2012-01..2022-12"
CODE = (f"{X}/audit_census.py", f"{X}/audit_lib.py")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def look_spec() -> dict:
    """The record_look call (harness.prereg3): ``run --log-look`` makes it before reading; the
    lead may instead make it by hand, before the run."""
    return {"units": [LOOK_UNIT], "data_scope": LOOK_SCOPE, "wp": "X5",
            "purpose": "model_design",
            "members": {LOOK_UNIT: {
                "none": "census forcing only (pump electricity); no well, benchmark, cell head, "
                        "leveling or township value",
                "description": "round 7 rank 8 census audit: bimonthly even split and duty-cap "
                               "retention by county, model_census v1 and preorigin v3"}},
            "note": f"{AUDIT_ID}: forcing-only model_design look (judge rank 8: no registration "
                    "needed)"}


# ----------------------------------------------------------------------------- retention part
def retention_part(mc_dir: Path, grid) -> dict:
    from hydrophysics.twin.pumping import _cell_lookup, clean_census
    from hydrophysics.twin.scenario import purpose_class

    pumps = pd.read_parquet(mc_dir / "census.parquet")
    kwh = pd.read_parquet(mc_dir / "kwh.parquet", columns=["datetime", "electricity_kwh",
                                                            "pump"])
    pumps["sid"] = pumps["sid"].astype(str)
    cell = _cell_lookup(pumps, grid)
    pumps["cell"] = pumps["sid"].map(cell).fillna(-1).astype(np.int64)
    pumps["cls"] = pumps["PURPOSE"].map(purpose_class)
    pt = L.meter_table(pumps, kwh, T0, T1)
    # engineering check: clean_census itself (the model) gives the same totals and kept set
    kept_p, kept_k, rep = clean_census(pumps.drop(columns=["county", "cell", "cls"]), kwh,
                                       cap_duty=L.CAP_DUTY, t0=T0, t1=T1)
    mine_dedup = float(pt["e_dedup"].sum()) / 1e6
    mine_kept = float(pt.loc[~pt["dropped"], "e_dedup"].sum()) / 1e6
    model_kept_k = float(kept_k["electricity_kwh"].sum()) / 1e6
    chk = {"dedup_GWh_mine": mine_dedup, "dedup_GWh_clean_census":
           float(rep["kwh_dedup_GWh"].sum()),
           "kept_GWh_mine": mine_kept, "kept_GWh_clean_census": float(rep["kwh_kept_GWh"].sum()),
           "kept_GWh_clean_census_kwh_table": model_kept_k,
           "kept_set_equal": set(kept_p["sid"].astype(str)) == set(
               pt.loc[~pt["dropped"], "sid"])}
    tol = 1e-6 * max(1.0, mine_dedup)
    chk["passed"] = bool(chk["kept_set_equal"]
                         and abs(chk["dedup_GWh_mine"] - chk["dedup_GWh_clean_census"]) <= tol
                         and abs(chk["kept_GWh_mine"] - chk["kept_GWh_clean_census"]) <= tol
                         and abs(chk["kept_GWh_mine"] - model_kept_k) <= tol)
    if not chk["passed"]:
        raise SystemExit("ENGINEERING FAILURE: the retention table does not reproduce "
                         f"pumping.clean_census ({json.dumps(chk)}); nothing written")
    ret = L.retention(pt[pt["county"].isin(COUNTIES)], YEARS)
    return {"retention": ret, "model_check": chk, "memory_figure": L.memory_figure(ret)}


# ----------------------------------------------------------------------------- even-split part
def _entities(npz, meters: pd.DataFrame) -> dict:
    """In-grid (meter, class) entities: monthly kWh (NaN where any pump of the entity is NaN),
    billing phase and class."""
    val = np.asarray(npz["value"], dtype=np.float64)
    cell = np.asarray(npz["cell"], dtype=np.int64)
    midx = np.asarray(npz["meter_idx"], dtype=np.int64)
    pcls = np.asarray(npz["purpose_class"]).astype(str)
    keep = cell >= 0
    val, midx, pcls = val[keep], midx[keep], pcls[keep]
    classes = sorted(set(pcls))
    ccode = np.array([classes.index(c) for c in pcls], dtype=np.int64)
    key = midx * (len(classes) + 1) + ccode
    ukey, inv = np.unique(key, return_inverse=True)
    T = val.shape[1]
    E = np.zeros((len(ukey), T))
    anynan = np.zeros((len(ukey), T), dtype=bool)
    np.add.at(E, inv, np.nan_to_num(val, nan=0.0))
    np.logical_or.at(anynan, inv, ~np.isfinite(val))
    E_nan = np.where(anynan, np.nan, E)
    e_meter = ukey // (len(classes) + 1)
    e_cls = np.array(classes)[ukey % (len(classes) + 1)]
    mt = meters.set_index("meter_idx")
    code = pd.Series(e_meter).map(mt["phase_code"].astype(str)).fillna("unknown").to_numpy()
    det = pd.Series(e_meter).map(mt["phase_detected"].astype(str)).fillna("unknown").to_numpy()
    phase = L.resolve_phase(code, det)
    from_detected = ~np.isin(code.astype(str), L.PHASES) & (phase != "unknown")
    return {"E": E, "E_nan": E_nan, "phase": phase, "cls": e_cls, "classes": classes,
            "from_detected": from_detected}


def even_split_part(po_dir: Path) -> tuple[dict, dict]:
    seasonal: dict = {}
    extra: dict = {}
    for county in COUNTIES:
        npz = np.load(po_dir / f"pumps_{county}.npz", allow_pickle=False)
        meters = pd.read_parquet(po_dir / f"meters_{county}.parquet",
                                 columns=["meter_idx", "phase_code", "phase_detected"])
        dates = pd.to_datetime([str(d)[:7] for d in npz["dates"]])
        months = dates.month.to_numpy()
        years = dates.year.to_numpy()
        t_lo = int(np.flatnonzero(dates == pd.Timestamp(WINDOW[0]))[0])
        t_hi = int(np.flatnonzero(dates == pd.Timestamp(WINDOW[1]))[0]) + 1
        ent = _entities(npz, meters)
        E, E_nan, phase, ecls = ent["E"], ent["E_nan"], ent["phase"], ent["cls"]
        if "irrigation" not in ent["classes"]:
            raise SystemExit(f"ENGINEERING FAILURE: {county} purpose_class labels "
                             f"{ent['classes'][:8]} carry no 'irrigation'; nothing written")
        unk = float(E[phase == "unknown"].sum() / max(E.sum(), 1e-12))
        det_share = float(E[ent["from_detected"]].sum() / max(E.sum(), 1e-12))
        if unk > 0.01:
            raise SystemExit(f"ENGINEERING FAILURE: {county} {unk:.1%} of the energy has no "
                             "resolvable billing phase (neither phase_code nor phase_detected "
                             "is A, B or monthly, or the meter_idx join failed); nothing "
                             "written")
        monthly = phase == "monthly"
        # reference shapes: county-class monthly meters, else the county's all-class monthly
        cls_list = ent["classes"]
        ref_all = E[monthly].sum(axis=0)
        refs, ref_src = [ref_all], {}
        group = np.zeros(len(E), dtype=np.int64)
        for c in cls_list:
            m = monthly & (ecls == c)
            n_pos = int((E[m][:, t_lo:t_hi].sum(axis=1) > 0).sum())
            if n_pos >= L.N_MIN_REF:
                refs.append(E[m].sum(axis=0))
                group[ecls == c] = len(refs) - 1
                ref_src[c] = {"source": "class monthly meters", "n_meters": n_pos}
            else:
                ref_src[c] = {"source": "county all-class monthly meters (fallback)",
                              "n_meters_in_class": n_pos}
        R, counts = L.reshape_pairs(E_nan, phase, group, np.vstack(refs), months, t_lo, t_hi)
        R = np.where(np.isfinite(R), R, np.nan_to_num(E))
        share = L.even_split_share(E_nan, phase, months, t_lo, t_hi)
        sl = slice(t_lo, t_hi)
        mo, yr = months[sl], years[sl]
        sel = {"all": np.ones(len(E), bool), "irrigation": ecls == "irrigation",
               "all_nodomestic": ecls != "domestic"}
        seasonal[county] = {}
        for name in CLASS_SERIES:
            s = sel[name]
            e_s = float(E[s][:, sl].sum())
            bi_share = float(E[s & np.isin(phase, ["A", "B"])][:, sl].sum()) / max(e_s, 1e-12)
            ev_s = L.even_split_share(E_nan[s], phase[s], months, t_lo, t_hi)
            dl = E[s][:, sl].sum(axis=0)
            rs = R[s][:, sl].sum(axis=0)
            ref = E[s & monthly][:, sl].sum(axis=0)
            bi = s & np.isin(phase, ["A", "B"])
            seasonal[county][name] = {
                "delivered": L.series_metrics(dl, mo, yr),
                "resplit": L.series_metrics(rs, mo, yr),
                "monthly_meters_only": L.series_metrics(ref, mo, yr),
                "bimonthly_delivered": L.series_metrics(E[bi][:, sl].sum(axis=0), mo, yr),
                "bimonthly_resplit": L.series_metrics(R[bi][:, sl].sum(axis=0), mo, yr),
                "smooth2_delivered": L.series_metrics(L.smooth2(dl), mo, yr),
                "smooth2_resplit": L.series_metrics(L.smooth2(rs), mo, yr),
                "expected_amp_ratio_from_even_split": L.expected_amp_ratio(
                    bi_share * ev_s["energy_even_split_share"])}
        dl_all, rs_all = E[:, sl].sum(axis=0), R[:, sl].sum(axis=0)
        ys = np.unique(yr)
        a_d = np.array([dl_all[yr == y].sum() for y in ys])
        a_r = np.array([rs_all[yr == y].sum() for y in ys])
        e_tot = float(E[:, sl].sum())
        extra[county] = {
            "energy_share_by_phase": {p: float(E[phase == p][:, sl].sum() / e_tot)
                                      for p in ("A", "B", "monthly", "unknown")},
            "n_entities_by_phase": {p: int((phase == p).sum())
                                    for p in ("A", "B", "monthly", "unknown")},
            "energy_share_phase_from_detected": det_share,
            "energy_share_phase_unresolved": unk,
            **share, **counts, "reference_shapes": ref_src,
            "expected_amp_ratio_from_even_split": L.expected_amp_ratio(
                float(E[np.isin(phase, ["A", "B"])][:, sl].sum() / e_tot)
                * share["energy_even_split_share"]),
            "annual_sum_max_rel_change": float(np.max(np.abs(a_r - a_d) / a_d)),
            "in_grid_GWh_yr_delivered": e_tot / 1e6 / YEARS}
    return seasonal, extra


def bookkeeping_reading(seasonal: dict) -> dict:
    """Changhua / Yunlin forcing seasonal amplitude, delivered and re-split (the share of the
    Changhua response gap that bookkeeping could explain is the change of this ratio)."""
    out = {}
    for cls in CLASS_SERIES:
        r = {v: seasonal["changhua"][cls][v]["amp_rel"] / seasonal["yunlin"][cls][v]["amp_rel"]
             for v in ("delivered", "resplit")}
        r["change"] = r["resplit"] / r["delivered"] - 1.0
        out[cls] = r
    return out


def run(out_dir: Path, mc_dir: Path, po_dir: Path, grid, log_look: bool = False,
        prereg_dir=None) -> dict:
    out_dir = Path(out_dir)
    if "preregistration" in out_dir.resolve().parts:
        raise SystemExit("REFUSED: the audit never writes into the preregistration registry")
    target = out_dir / "audit.json"
    if target.exists():
        raise SystemExit(f"REFUSED: {target} exists; the audit is written once")
    files = [mc_dir / "census.parquet", mc_dir / "kwh.parquet", mc_dir / "MANIFEST.json",
             po_dir / "MANIFEST.json"] + [po_dir / f"{s}_{c}.{e}" for c in COUNTIES
                                          for s, e in (("pumps", "npz"), ("meters", "parquet"))]
    for mf in (mc_dir / "MANIFEST.json", po_dir / "MANIFEST.json"):
        if not mf.exists():
            raise SystemExit(f"REFUSED: {mf} missing; allowed uses cannot be checked")
        m = json.loads(mf.read_text())
        if "model_design" not in m.get("allowed_uses", []):
            raise SystemExit(f"REFUSED: {mf} does not allow model_design")
    if log_look:                      # the look is logged before any product is read
        from harness import prereg3
        kw = look_spec()
        if prereg_dir is not None:
            kw["prereg_dir"] = prereg_dir
            kw["rows_dir"] = Path(prereg_dir) / "_rows"
        prereg3.record_look(kw.pop("units"), kw.pop("data_scope"), kw.pop("wp"),
                            kw.pop("purpose"), **kw)
    ret = retention_part(mc_dir, grid)
    seasonal, extra = even_split_part(po_dir)
    annual = {c: extra[c]["annual_sum_max_rel_change"] for c in COUNTIES}
    dec = L.decide(seasonal, ret["retention"], annual)
    dec["forcing_parity_changhua"] = L.forcing_parity(ret["retention"], extra)
    ch_in = ret["retention"]["changhua"]["in_grid"]["all"]["kept_GWh_yr"]
    rep = {
        "id": AUDIT_ID, "kind": "census audit (forcing only, model_design look)",
        "window": list(WINDOW), "counties": list(COUNTIES),
        "inputs": {str(p): sha256(p) for p in files if p.exists()},
        "code_sha256": {c: sha256(REPO / c) for c in CODE},
        "decisions": dec,
        "memory_figure_8pct": ret["memory_figure"],
        "in_grid_kept_changhua_GWh_yr": ch_in,
        "in_grid_kept_references_GWh_yr": REFERENCE_GWH_YR,
        "retention": ret["retention"], "model_check": ret["model_check"],
        "even_split": extra, "seasonal": seasonal,
        "bookkeeping_changhua_over_yunlin_amp": bookkeeping_reading(seasonal),
        "rules": {"amp_tol": L.AMP_TOL, "phase_tol_days": L.PHASE_TOL_DAYS,
                  "saw_tol": L.SAW_TOL, "range_tol": L.RANGE_TOL,
                  "cap_near_tol": L.CAP_NEAR_TOL, "annual_tol": L.ANNUAL_TOL,
                  "parity_tol": L.PARITY_TOL,
                  "memory_band": list(L.MEMORY_BAND), "n_min_ref": L.N_MIN_REF,
                  "smooth2": "weights (1/4, 1/2, 1/4), zero phase",
                  "fixed_before_run": True},
        "notes": [
            "monthly values of bimonthly meters are even-split halves; the product allows "
            "neither scoring_target nor selection, so no monthly-disaggregation test scores "
            "against them; the re-split series is a forcing diagnostic, never a target",
            "the re-split assumes bimonthly meters share the within-pair shape of the same "
            "county's monthly-billed meters of the same class (a minority population)",
            "December 2022 of Dec-Jan-paired meters is the product's fill (flag 4); its pair "
            "lies outside the window and is not re-split"]}
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(target, "x") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False, sort_keys=True, default=float)
    return rep


# ----------------------------------------------------------------------------- synthetic
def synthetic_grid(nx: int = 6, ny: int = 6, dx: float = 1000.0) -> SimpleNamespace:
    mask = np.ones((ny, nx), dtype=bool)
    mask[0, 0] = False
    return SimpleNamespace(mask=mask, nx=nx, ny=ny, x0=0.0, y0=0.0, dx=dx,
                           n_active=int(mask.sum()))


def build_synthetic(d: Path, seed: int = 7, n_meters: int = 300) -> tuple[Path, Path]:
    """Tiny model_census-like and preorigin-like products with known structure: a seasonal
    true monthly draw, even-split bimonthly meters of both phases, monthly meters, two
    over-cap meters and one shared meter."""
    rng = np.random.default_rng(seed)
    mc, po = d / "model_census", d / "preorigin"
    mc.mkdir(parents=True)
    po.mkdir(parents=True)
    dates = pd.date_range("2007-01-01", "2022-12-01", freq="MS")
    T = len(dates)
    mo = dates.month.to_numpy()
    season = 1.0 + 0.35 * np.cos(2 * np.pi * (mo - 7.5) / 12.0)
    rows_p, rows_k = [], []
    for county in COUNTIES:
        pump_vals, pump_meter, pump_cell, pump_cls, uids, meters = [], [], [], [], [], []
        for m in range(n_meters):
            phase = ("monthly" if m % 6 == 0 else ("A" if m % 2 else "B"))
            cls = "irrigation" if m % 5 else "domestic"
            hp = 10.0
            base = rng.uniform(200, 800) * season * rng.lognormal(0, 0.05, T)
            if m in (1, 2):
                base = base * 40.0               # over the duty cap
            x = base.copy()
            if phase in ("A", "B"):
                st = L.pair_starts(phase, mo, 0, T)
                tot = x[st] + x[st + 1]
                x[st] = tot / 2
                x[st + 1] = tot / 2
            cell = -1 if m == 3 else int(rng.integers(1, 30))   # cell 0 is masked
            sid = f"{county}:{m}"
            npumps = 2 if m == 4 else 1          # a shared meter
            for j in range(npumps):
                rows_p.append({"sid": f"{sid}-{j}", "PUMP_HP": hp / npumps,
                               "電號1": f"{county}-M{m}",
                               "PURPOSE": "灌溉用水" if cls == "irrigation" else "家用",
                               "TWD97_X": (cell % 6) * 1000 + 500 if cell >= 0 else -9e5,
                               "TWD97_Y": (cell // 6) * 1000 + 500 if cell >= 0 else -9e5,
                               "county": county})
                sel = dates >= pd.Timestamp(T0)
                rows_k.append(pd.DataFrame({"datetime": dates[sel], "electricity_kwh": x[sel],
                                            "pump": f"{sid}-{j}"}))
            if m in (1, 2):
                continue                         # dropped by the cap: not in preorigin v3
            pump_vals.append(x)
            pump_meter.append(m)
            pump_cell.append(cell)
            pump_cls.append(cls)
            uids.append(sid)
            code = "unknown" if m % 25 == 7 else phase      # C0: code unknown, detected
            meters.append({"meter_idx": m, "phase_code": code, "phase_detected": phase})
        np.savez(po / f"pumps_{county}.npz", value=np.array(pump_vals),
                 meter_idx=np.array(pump_meter), cell=np.array(pump_cell),
                 purpose_class=np.array(pump_cls), uid=np.array(uids),
                 dates=np.array([d.strftime("%Y-%m") for d in dates]))
        pd.DataFrame(meters).to_parquet(po / f"meters_{county}.parquet", index=False)
    pd.DataFrame(rows_p).to_parquet(mc / "census.parquet", index=False)
    pd.concat(rows_k, ignore_index=True).to_parquet(mc / "kwh.parquet", index=False)
    man = {"allowed_uses": ["forcing_frozen", "model_design"]}
    for p in (mc, po):
        (p / "MANIFEST.json").write_text(json.dumps(man))
    return mc, po


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--model-census", default=MODEL_CENSUS)
    r.add_argument("--preorigin", default=PREORIGIN)
    r.add_argument("--log-look", action="store_true",
                   help="record the model_design look (the lead; harness.prereg3)")
    s = sub.add_parser("smoke")
    s.add_argument("--out", required=True)
    sub.add_parser("look-spec")
    a = ap.parse_args(argv)
    if a.cmd == "look-spec":
        print(json.dumps(look_spec(), indent=1, ensure_ascii=False))
        return 0
    if a.cmd == "smoke":
        mc, po = build_synthetic(Path(a.out) / "syn")
        rep = run(Path(a.out) / "audit", mc, po, synthetic_grid())
    else:
        from hydrophysics.twin.calibrate_flow import DEFAULT_PATHS
        from hydrophysics.twin.grid import build_grid
        grid = build_grid(str(REPO / DEFAULT_PATHS["polygon"]), dx=1000.0)
        rep = run(Path(a.out), REPO / a.model_census, REPO / a.preorigin, grid,
                  log_look=a.log_look)
    print(json.dumps({"decisions": {k: v["decision"] for k, v in rep["decisions"].items()
                                    if isinstance(v, dict) and "decision" in v},
                      "annual_sum_max_rel_change": rep["decisions"]["annual_sum_max_rel_change"],
                      "forcing_parity_changhua": rep["decisions"]["forcing_parity_changhua"],
                      "memory_figure_8pct": rep["memory_figure_8pct"]["reading"],
                      "in_grid_kept_changhua_GWh_yr": rep["in_grid_kept_changhua_GWh_yr"]},
                     indent=1, ensure_ascii=False, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
