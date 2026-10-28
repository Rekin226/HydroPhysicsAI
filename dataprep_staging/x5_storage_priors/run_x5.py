"""X5 route 3 on the real pre-2019 hourly heads (X5.storage_priors.v2_postfailure; runs once
after the declared synthetic recovery criteria pass and the look is logged).

Steps (each refuses to overwrite its output):

  select   wells read: AMP files of layer 2-4 screens (L1 v3 pre-origin station table,
           layer_flag 0); metadata only, no head value. Writes work/selection.csv.
  run      per well: cleaned hourly heads 2012-14 + 2016-18, earth-tide fit (Q1, O1, N2, M2)
           and the production barometric fit; writes work/well_estimates.csv.
  report   the declared gate (Spearman of Ss(M2) and Ss(BE) at inland deep wells), the coastal
           flags, Hsieh T, the AMP-C alternative at Tuku, the per-layer/zone prior table
           (results/dataprep/x5_storage_priors/v1, MANIFEST) and the report
           (results/dataprep/reports/X5.storage_priors.v1/report.json + reading.md).

Usage: python -I run_x5.py select|run|report [--workers 2]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import x5sp_const as K  # noqa: E402
import x5sp_lib as L  # noqa: E402
from common.hashing import sha256_file  # noqa: E402
from common.paths import REPO, rel_to_repo  # noqa: E402

SEL = K.SELECTION
EST = K.WORK / "well_estimates.csv"
OMEGA_M2 = 2 * np.pi / (12.4206012 * 3600.0)
TUKU_HEAD = "09050321"
TUKU_PILE = REPO / "chou-shui-data/data/ls_cache/ls-wra-dbm-obs__TKJS_282_29.parquet"
TUKU_PILE_DEPTH_M = 130.0


def zone_of(x_m: float) -> str:
    xk = x_m / 1000.0
    if xk <= K.ZONE_EDGES_KM[0]:
        return "distal"
    return "mid" if xk <= K.ZONE_EDGES_KM[1] else "proximal"


# ----------------------------------------------------------------------------- select
def select() -> pd.DataFrame:
    if SEL.exists():
        raise SystemExit(f"{SEL} exists")
    m = pd.read_parquet(K.META, columns=K.META_COLS)
    m["layer_code"] = pd.to_numeric(m["layer_code"], errors="coerce")
    m = m[(m["layer_flag"] == 0) & m["layer_code"].isin([2, 3, 4])]
    m = m[np.array([(K.WELLS_DIR / f"{s}.parquet").exists() for s in m["sid"]], bool)]
    m = m[m["x_twd97"].notna() & (m["x_twd97"] > 100000)].copy()
    m["zone"] = [zone_of(x) for x in m["x_twd97"]]
    m["inland"] = m["x_twd97"] > K.INLAND_X_M
    K.WORK.mkdir(parents=True, exist_ok=True)
    m.to_csv(SEL, index=False)
    print(f"{len(m)} wells -> {rel_to_repo(SEL)}")
    return m


# ----------------------------------------------------------------------------- run
_DAILY = None
_P = None


def _init():
    global _DAILY, _P
    _DAILY = pd.read_parquet(K.HEADS_DAILY, columns=["sid", "date", "flag"],
                             filters=[("date", ">=", pd.Timestamp(K.WINDOW[0])),
                                      ("date", "<=", pd.Timestamp(K.WINDOW[1]))])
    _P = L.load_pressure()


def one(row: dict) -> dict:
    r = {"sid": row["sid"]}
    try:
        h = L.load_hourly(row["sid"], _DAILY)
        r["n_hours"] = int(h.notna().sum())
        t = L.fit_tides(h, row["lat"], row["lon"])
        r["tide_status"] = t.get("status")
        for n in K.FIT_CONSTITUENTS:
            for k, v in t.get(n, {}).items():
                r[f"{n}_{k}"] = v
        cell = row["era5_cell"]
        if cell in _P.columns:
            b = L.fit_be(h, _P[cell])
            r.update({f"be_{k}": v for k, v in b.items()})
        else:
            r["be_status"] = "no_pressure_cell"
        r["error"] = ""
    except Exception as e:                       # engineering failure, logged per well
        r["error"] = f"{type(e).__name__}: {e}"
    return r


def run(workers: int) -> None:
    if EST.exists():
        raise SystemExit(f"{EST} exists; the real-data run is done once")
    sel = pd.read_csv(SEL, dtype={"sid": str})
    with ProcessPoolExecutor(workers, initializer=_init) as ex:
        rows = list(ex.map(one, sel.to_dict("records"), chunksize=4))
    df = sel.merge(pd.DataFrame(rows), on="sid", how="left")
    EST.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(EST, index=False)
    print(f"{len(df)} wells, errors: {(df['error'].fillna('') != '').sum()}")


# ----------------------------------------------------------------------------- derived
def derive(df: pd.DataFrame, n_por=K.N_POR, nu=K.NU) -> pd.DataFrame:
    d = df.copy()
    d["m2_det"] = (d["tide_status"] == "ok") & (d["M2_snr"] >= K.SNR_MIN)
    d["o1_det"] = (d["tide_status"] == "ok") & (d["O1_snr"] >= K.SNR_MIN)
    d["be_ok"] = (d["be_status"] == "accepted") & (d["be_A_K"] < 0) & d["be_BE_long"].between(
        K.BE_MIN, K.BE_MAX)
    d["ss_m2"] = np.where(d["m2_det"], L.c_nu(nu) / d["M2_gain"], np.nan)
    d["ss_o1"] = np.where(d["o1_det"], L.c_nu(nu) / d["O1_gain"], np.nan)
    d["ss_be"] = np.where(d["be_ok"], K.RHO_G * n_por * K.BETA_W / d["be_BE_long"], np.nan)
    ratio = d["M2_gain"] / d["O1_gain"]
    d["m2_o1_gain_ratio"] = np.where(d["o1_det"] & d["m2_det"], ratio, np.nan)
    d["ocean"] = np.select([d["o1_det"] & d["m2_det"] & (ratio > K.OCEAN_RATIO),
                            d["o1_det"] & d["m2_det"]], ["ocean", "earth"], "undetermined")
    d["ss_m2_over_be"] = d["ss_m2"] / d["ss_be"]
    use_m2 = d["m2_det"] & ((d["ocean"] == "earth")
                            | ((d["ocean"] == "undetermined") & d["inland"]))
    both = use_m2 & d["be_ok"]
    d["ss_source"] = np.select([both, use_m2, d["be_ok"]], ["M2+BE", "M2", "BE"], "none")
    d["ss_well"] = np.select([both, use_m2, d["be_ok"]],
                             [np.sqrt(d["ss_m2"] * d["ss_be"]), d["ss_m2"], d["ss_be"]],
                             np.nan)
    wt = K.RHO_G * n_por * K.BETA_W
    ssk = d["ss_well"] - wt
    d["ssk_floored"] = ssk <= 0.1 * d["ss_well"]
    d["ss_k"] = np.where(d["ssk_floored"], 0.1 * d["ss_well"], ssk)
    d["n_implied"] = (L.c_nu(nu) * d["be_BE_long"] / (d["M2_gain"] * K.RHO_G * K.BETA_W)
                      ).where(d["m2_det"] & d["be_ok"])
    return d


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def gate(d: pd.DataFrame) -> dict:
    rng = np.random.default_rng(K.BOOT_GROUP_SEED)

    def stat(mask, xa="ss_m2", xb="ss_be"):
        s = d[mask & d[xa].notna() & d[xb].notna()]
        n = len(s)
        if n < 3:
            return {"n": n, "spearman": None}
        a, b = s[xa].to_numpy(), s[xb].to_numpy()
        rho = spearman(a, b)
        bs = [spearman(a[i], b[i]) for i in (rng.integers(0, n, n) for _ in range(K.BOOT_GROUP))]
        perm = np.array([spearman(a, rng.permutation(b)) for _ in range(10000)])
        return {"n": n, "spearman": rho, "ci90": [float(np.nanquantile(bs, 0.05)),
                                                   float(np.nanquantile(bs, 0.95))],
                "p_one_sided_perm": float((np.sum(perm >= rho) + 1) / (len(perm) + 1)),
                "wells": sorted(s["sid"].tolist())}
    deep = d["layer_code"].isin(K.DEEP_LAYERS)
    prim = stat(deep & d["inland"])
    n, rho = prim["n"], prim["spearman"]
    if n < K.N_MIN or rho is None:
        verdict = "NOT_EVALUABLE"
    else:
        verdict = "PASS" if rho >= K.SPEARMAN_MIN else "FAIL"
    return {
        "primary": {**prim, "threshold": K.SPEARMAN_MIN, "n_min": K.N_MIN, "verdict": verdict,
                    "definition": "layer 3-4, layer_flag 0, x_twd97 > 182 km; M2 SNR >= 3; BE "
                                  "accepted, A_K < 0, 0.02 <= BE <= 1; Spearman(Ss_M2, Ss_BE)"},
        "secondary_reported_not_gating": {
            "inland_deep_earth_only": stat(deep & d["inland"] & (d["ocean"] != "ocean")),
            "inland_deep_O1_vs_BE": stat(deep & d["inland"], "ss_o1", "ss_be"),
            "inland_layers_2to4": stat(d["inland"]),
            "all_deep_incl_coastal": stat(deep),
            "coastal_deep": stat(deep & ~d["inland"]),
        }}


def ocean_summary(d: pd.DataFrame) -> dict:
    out = {}
    for z in ("distal", "mid", "proximal"):
        s = d[(d["zone"] == z) & d["layer_code"].isin([2, 3, 4])]
        both = s["ss_m2_over_be"].dropna()
        out[z] = {"n_wells": int(len(s)), "n_m2_detected": int(s["m2_det"].sum()),
                  "n_o1_detected": int(s["o1_det"].sum()),
                  "ocean_flag_counts": s["ocean"].value_counts().to_dict(),
                  "median_m2_o1_gain_ratio": _med(s["m2_o1_gain_ratio"]),
                  "median_ss_m2_over_ss_be": _med(both), "n_ratio": int(len(both))}
    return out


def _med(x) -> float | None:
    x = pd.Series(x).dropna()
    return float(np.median(x)) if len(x) else None


def hsieh(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in d[d["m2_det"] & d["inland"]].iterrows():
        S = (r["ss_well"] if np.isfinite(r["ss_well"]) else r["ss_m2"]) * K.SCREEN_B_M
        eta, se = r["M2_eta_deg"], r["M2_se_phase_deg"]
        if eta < -2 * se:
            kind, T = "estimate", L.hsieh_T(eta, S, OMEGA_M2)
        elif eta > 2 * se:
            kind, T = "phase_lead (leakage or ocean; Hsieh model does not apply)", np.nan
        else:
            kind, T = "lower_bound", L.hsieh_T(-2 * se, S, OMEGA_M2) if se > 0 else np.nan
        rows.append({"sid": r["sid"], "layer_code": r["layer_code"], "zone": r["zone"],
                     "eta_deg": eta, "se_phase_deg": se, "kind": kind, "T_m2s": T})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- priors
def group_rows(d: pd.DataFrame, keys: list[str], col: str) -> list[dict]:
    rng = np.random.default_rng(K.BOOT_GROUP_SEED)
    out = []
    for g, s in d[d[col].notna()].groupby(keys):
        g = g if isinstance(g, tuple) else (g,)
        x = np.log10(s[col].to_numpy())
        rec = dict(zip(keys, [str(v) if not isinstance(v, (int, np.integer)) else int(v)
                              for v in g], strict=True))
        rec.update({"quantity": col, "n_wells": int(len(x))})
        if len(x) < K.MIN_WELLS_PER_GROUP:
            rec["status"] = "not_estimated (fewer than 3 wells)"
            out.append(rec)
            continue
        bs = [np.median(x[rng.integers(0, len(x), len(x))]) for _ in range(K.BOOT_GROUP)]
        rec.update({"status": "estimated", "median_log10": float(np.median(x)),
                    "median": float(10 ** np.median(x)),
                    "ci90_median": [float(10 ** np.quantile(bs, 0.05)),
                                    float(10 ** np.quantile(bs, 0.95))],
                    "se_median_log10": float(np.std(bs, ddof=1)),
                    "sd_between_wells_log10": float(np.std(x, ddof=1)),
                    "sources": s["ss_source"].value_counts().to_dict()})
        out.append(rec)
    return out


def systematic(d: pd.DataFrame) -> dict:
    """log10 half-ranges of the group-free systematic factors: porosity (BE route), Poisson
    ratio (tide route) and the median method discrepancy at inland earth-tide wells."""
    lo, hi = K.N_POR_RANGE
    por = 0.5 * abs(np.log10(hi / lo)) * 0.5        # geometric-mean Ss carries half of it
    nlo, nhi = K.NU_RANGE
    nu = 0.5 * abs(np.log10(L.c_nu(nlo) / L.c_nu(nhi))) * 0.5
    s = d[(d["ss_source"] == "M2+BE") & d["inland"]]
    disc = (0.5 * abs(float(np.median(np.log10(s["ss_m2"] / s["ss_be"]))))
            if len(s) else 0.0)
    tot = float(np.sqrt(por ** 2 + nu ** 2 + disc ** 2))
    return {"porosity_half_range_log10": float(por), "poisson_half_range_log10": float(nu),
            "method_discrepancy_half_log10": float(disc), "n_method_wells": int(len(s)),
            "total_sys_log10": tot}


def amp_c(d: pd.DataFrame) -> dict:
    """Declared alternative A2 ('AMP-C'): 1-cpd gain of the 130 m Tuku pile against head
    09050321 in Mar-May 2017 and 2018 (5-day blocks, constant + trend + S1/S2 per block,
    complex least-squares transfer); compaction storage S = gain; compared with the well's
    Ss_k x 130 m. Design-time (the audit read Mar-May 2017-22 at this pair)."""
    try:
        daily = pd.read_parquet(K.HEADS_DAILY, columns=["sid", "date", "flag"],
                                filters=[("sid", "==", TUKU_HEAD)])
        h = L.load_hourly(TUKU_HEAD, daily)
        b = pd.read_parquet(TUKU_PILE)["value"]
        b.index = pd.DatetimeIndex(b.index).as_unit("ns")
        b = b.loc[K.WINDOW[0]:K.WINDOW[1]].resample("h").mean().reindex(h.index)
        H, B = [], []
        for yr in (2017, 2018):
            for s0 in pd.date_range(f"{yr}-03-01", f"{yr}-05-27", freq="5D"):
                sl = slice(s0, s0 + pd.Timedelta(hours=119))
                hh, bb = h.loc[sl], b.loc[sl]
                ok = hh.notna() & bb.notna()
                if ok.sum() < 96:
                    continue
                th = ((hh.index[ok] - s0).total_seconds() / 3600.0).to_numpy()
                Z = np.column_stack([np.ones_like(th), th, np.cos(2 * np.pi * th / 24),
                                     np.sin(2 * np.pi * th / 24), np.cos(4 * np.pi * th / 24),
                                     np.sin(4 * np.pi * th / 24)])
                ch = np.linalg.lstsq(Z, hh[ok].to_numpy(), rcond=None)[0]
                cb = np.linalg.lstsq(Z, bb[ok].to_numpy(), rcond=None)[0]
                H.append(ch[2] - 1j * ch[3])
                B.append(cb[2] - 1j * cb[3])
        H, B = np.array(H), np.array(B)
        if len(H) < 6:
            return {"status": "insufficient_blocks", "n_blocks": int(len(H))}
        G = np.sum(B * np.conj(H)) / np.sum(np.abs(H) ** 2)
        coh = np.abs(np.sum(B * np.conj(H))) ** 2 / (np.sum(np.abs(H) ** 2)
                                                     * np.sum(np.abs(B) ** 2))
        rng = np.random.default_rng(K.BOOT_GROUP_SEED)
        gb = []
        for _ in range(K.BOOT_GROUP):
            i = rng.integers(0, len(H), len(H))
            gb.append(abs(np.sum(B[i] * np.conj(H[i])) / np.sum(np.abs(H[i]) ** 2)))
        gain_mm_per_m = float(abs(G))           # vendor pile units are mm
        well = d[d["sid"] == TUKU_HEAD]
        ssk = float(well["ss_k"].iloc[0]) if len(well) else np.nan
        return {"status": "ok", "n_blocks": int(len(H)), "gain_mm_per_m": gain_mm_per_m,
                "gain_ci90": [float(np.quantile(gb, 0.05)), float(np.quantile(gb, 0.95))],
                "phase_deg": float(np.degrees(np.angle(G))), "coherence": float(coh),
                "S_pile_0_130m": gain_mm_per_m / 1000.0,
                "well_ss_k": ssk, "well_ss_k_x_130m": ssk * TUKU_PILE_DEPTH_M,
                "ratio_pile_over_well": (gain_mm_per_m / 1000.0) / (ssk * TUKU_PILE_DEPTH_M)
                if np.isfinite(ssk) and ssk > 0 else None,
                "units_assumption": "pile value in mm (vendor feed); gain in mm per m of head"}
    except Exception as e:
        return {"status": f"error {type(e).__name__}: {e}"}


def report() -> None:
    rdir = K.REPORT_DIR
    path = rdir / "report.json"
    if path.exists():
        raise SystemExit(f"{path} exists; scored once")
    syn = json.loads((K.SYN_DIR / "synthetic_check.json").read_text())
    raw = pd.read_csv(EST, dtype={"sid": str})
    d = derive(raw)
    g = gate(d)
    sens = {}
    for tag, n_por, nu in (("n0.2", 0.2, K.NU), ("n0.4", 0.4, K.NU), ("nu0.2", K.N_POR, 0.2),
                           ("nu0.3", K.N_POR, 0.3)):
        dd = derive(raw, n_por, nu)
        sens[tag] = {"median_ss_well_inland_deep": _med(
            dd.loc[dd["inland"] & dd["layer_code"].isin(K.DEEP_LAYERS), "ss_well"])}
    hs = hsieh(d)
    ac = amp_c(d)
    sysd = systematic(d)
    d.to_csv(K.WORK / "well_estimates_derived.csv", index=False)
    hs.to_csv(K.WORK / "hsieh_T.csv", index=False)
    # ---------------------------------------------------------------- prior table v1
    from common import manifest
    # v2 is a post-failure exploratory registration: its product never passes a gate
    gate_pass = False
    pv = manifest.new_version_dir("x5_storage_priors")     # creates v<N+1> exclusively
    groups = []
    for keys in (["layer_code", "zone"], ["layer_code", "CountyName"], ["layer_code"],
                 ["zone"]):
        for col in ("ss_well", "ss_k"):
            groups += [{"grouping": "+".join(keys), **r} for r in group_rows(d, keys, col)]
    pt = pd.DataFrame(groups)
    pt["sys_log10"] = sysd["total_sys_log10"]
    pt["prior_sd_log10"] = np.sqrt(pt["se_median_log10"] ** 2 + pt["sys_log10"] ** 2)
    pt.to_csv(pv / "ss_prior_table.csv", index=False)
    floors = {}
    for r in group_rows(d[d["layer_code"].isin([2, 3, 4])], ["zone"], "ss_k"):
        if r.get("status") == "estimated":
            floors[r["zone"]] = {"ske_floor": r["median"] * K.COLUMN_B_M,
                                 "ske_floor_ci90": [c * K.COLUMN_B_M for c in r["ci90_median"]],
                                 "n_wells": r["n_wells"]}
    col = {"rule": "Ske floor per zone = median aquifer Ss_k (layers 2-4) x 300 m; a LOWER "
                   "bound on the column's elastic storage (aquitards are not sampled)",
           "zones": floors,
           "global_floor_for_--ske-min": min((v["ske_floor"] for v in floors.values()),
                                             default=None)}
    (pv / "column_ske_floor.json").write_text(json.dumps(col, indent=1))
    wells_out = d[["sid", "layer_code", "zone", "CountyName", "inland", "m2_det", "o1_det",
                   "be_ok", "ocean", "ss_m2", "ss_o1", "ss_be", "ss_source", "ss_well",
                   "ss_k", "ssk_floored", "m2_o1_gain_ratio", "n_implied"]]
    wells_out.to_csv(pv / "ss_by_well.csv", index=False)
    card = (f"# x5_storage_priors {pv.name}\n\nSpecific storage Ss and aquifer skeletal Ss_k "
            f"per screen (layers 2-4) from 2012-14 + 2016-18 hourly heads: earth-tide M2 "
            f"strain response and barometric efficiency (Jacob), combined per well, summarised "
            f"per layer and zone with a bootstrap se and a systematic term. Registration "
            f"{K.PREREG_ID}: synthetic recovery criteria "
            f"{'PASS' if syn['result']['gating_pass'] else 'FAIL'}, cross-method Spearman at "
            f"inland deep wells {g['primary']['verdict']}. POST-FAILURE EXPLORATORY (the "
            f"synthetic check of {K.PREREG_ID_V1} failed); gate.passed is false, so the loader "
            f"refuses the 'prior' use without a ruling. Design-time data in any case: the "
            f"heads were inside the H1a barometric and P5 tidal fit windows, and the "
            f"2026-10-25 audit previewed the cross-method agreement on overlapping wells.\n")
    (pv / "PRODUCT_CARD.md").write_text(card)
    val = {"prereg_id": K.PREREG_ID, "synthetic_check": syn["result"],
           "gate_primary": {k: v for k, v in g["primary"].items() if k != "wells"},
           "systematic": sysd}
    (pv / "validation.json").write_text(json.dumps(val, indent=1, default=str))
    m = manifest.build_manifest(
        pv, product="x5_storage_priors", layer="L2",
        inputs=[rel_to_repo(K.META), rel_to_repo(K.PRESSURE), rel_to_repo(K.HEADS_DAILY),
                rel_to_repo(EST), rel_to_repo(K.SYN_DIR / "synthetic_check.json")],
        code=[rel_to_repo(K.CODE_DIR / f) for f in ("run_x5.py", "x5sp_lib.py",
                                                     "x5sp_const.py", "synthetic_check.py")],
        parameters={"window": K.WINDOW, "n_por": K.N_POR, "nu": K.NU, "snr_min": K.SNR_MIN,
                    "ocean_ratio": K.OCEAN_RATIO, "column_b_m": K.COLUMN_B_M},
        fit_window_end="2018-12", validation_report="validation.json",
        gate={"passed": gate_pass, "criteria": f"{K.PREREG_ID} (post-failure, exploratory; "
              "never validation; cannot promote): synthetic recovery criteria "
              f"{list(K.SYN_GATING)} and Spearman(Ss_M2, Ss_BE) >= 0.4 at inland deep wells "
              "(n >= 10); passed is false by construction (v1's synthetic check failed)",
              "synthetic_gating_pass": bool(syn["result"]["gating_pass"]),
              "spearman_verdict": g["primary"]["verdict"]},
        allowed_uses=["prior", "prior_width_input", "design_time_evidence",
                      "after_the_fact_analysis"],
        forbidden_uses=["scoring_target", "calibration_target", "validator", "baseline"],
        input_trees=[{"path": rel_to_repo(K.WELLS_DIR)}])
    manifest.write_manifest(pv, m)
    # ---------------------------------------------------------------- report
    rep = {
        "prereg_id": K.PREREG_ID, "v1": K.PREREG_ID_V1,
        "label": "post-failure, exploratory; never validation; cannot promote (design-time "
                 "data in any case)",
        "synthetic_check": syn["result"], "gate": g, "ocean_loading": ocean_summary(d),
        "counts": {"wells_read": int(len(d)), "tide_ok": int((d["tide_status"] == "ok").sum()),
                   "m2_detected": int(d["m2_det"].sum()), "o1_detected": int(d["o1_det"].sum()),
                   "be_ok": int(d["be_ok"].sum()), "errors": int((d["error"].fillna("") != "")
                                                                 .sum()),
                   "by_source": d["ss_source"].value_counts().to_dict()},
        "porosity_implied_inland": {"median": _med(d.loc[d["inland"], "n_implied"]),
                                    "n": int(d.loc[d["inland"], "n_implied"].notna().sum())},
        "sensitivity_median_ss_well_inland_deep": sens, "systematic": sysd,
        "hsieh": {"counts": hs["kind"].value_counts().to_dict() if len(hs) else {},
                  "median_T_estimates": _med(hs.loc[hs["kind"] == "estimate", "T_m2s"])
                  if len(hs) else None,
                  "median_T_lower_bound": _med(hs.loc[hs["kind"] == "lower_bound", "T_m2s"])
                  if len(hs) else None},
        "alternatives": {"A2_amp_c_tuku": ac,
                         "A1_earthquake_steps": "not run as an Ss estimator: see reading.md"},
        "column_ske_floor": col, "prior_product": {"dir": rel_to_repo(pv),
                                                  "gate_passed": gate_pass},
        "files": {rel_to_repo(p): sha256_file(p) for p in
                  (EST, K.WORK / "well_estimates_derived.csv", K.WORK / "hsieh_T.csv",
                   pv / "ss_prior_table.csv", pv / "column_ske_floor.json",
                   pv / "ss_by_well.csv")},
    }
    rdir.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as f:
        json.dump(rep, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({"gate": g["primary"], "syn": syn["result"]["gating_pass"]}, indent=1,
                     default=str))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", choices=("select", "run", "report"))
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args(argv)
    if a.step == "select":
        select()
    elif a.step == "run":
        run(a.workers)
    else:
        report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
