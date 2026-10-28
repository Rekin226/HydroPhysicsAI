"""X5 STEP 0 (exploratory gap decomposition) and estimators (a) and (b), as declared in
``X5.energy_volume.step0_ab.v1`` (run only after that declaration exists).

STEP 0 splits the twin-vs-WRA irrigation volume gap into (i) conversion (head, efficiency) and
(ii) cleaning coverage, and describes the energy that the cleaning drops. (a) is WRA
irrigation groundwater per census irrigation kWh (kept, dedup and raw bases); (b) is pump
physics per kept irrigation pump (rated HP, discharge-pipe diameter, design velocity; and
an efficiency-head band). Labels: STEP 0 exploratory, never validation; (a) and (b)
design-time, neither independent of electricity.

Writes results/dataprep/x5_energy_volume/step0_ab/ (tables) and, once,
results/dataprep/reports/X5.energy_volume.step0_ab.v1/report.json.

Usage: python -I step0_ab.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import rel_to_repo  # noqa: E402
from x5_energy_volume import census as C  # noqa: E402
from x5_energy_volume import wra  # noqa: E402
from x5_energy_volume import x5const as K  # noqa: E402

OUTD = K.OUT / "step0_ab"
JM = K.JANMAY
Y0, Y1 = K.REF_YEARS


def c_twin() -> float:
    return K.ETA_TWIN * K.M3_PER_KWH_PER_M / (K.LIFT_MEDIAN + K.HEAD_EXTRA_TWIN)


def moy_profile(E: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Month-of-year mean kWh (12,) of the selected pumps over 2012-01..2022-12."""
    t0, t1 = C.tidx("2012-01"), C.tidx("2022-12") + 1
    s = E[rows][:, t0:t1].sum(axis=0)
    return s.reshape(-1, 12).mean(axis=0)


def repeat_share(E_meter: np.ndarray) -> float:
    """Share of non-zero meter-months (2012-22) equal to the previous month (bimonthly
    billing split evenly)."""
    t0, t1 = C.tidx("2012-01"), C.tidx("2022-12") + 1
    X = E_meter[:, t0:t1]
    nz = X[:, 1:] > 0
    rep = np.isclose(X[:, 1:], X[:, :-1], rtol=0, atol=0.51) & nz
    return float(rep.sum() / max(nz.sum(), 1))


def q(s, qs=(0.1, 0.25, 0.5, 0.75, 0.9)) -> dict:
    s = pd.Series(s).dropna()
    return {f"q{int(x * 100)}": (float(s.quantile(x)) if len(s) else None) for x in qs}


def county(cn: str, wref: dict) -> dict:
    p = C.load_pumps(cn)
    L = C.load_kwh(cn, len(p))
    B = C.bases(p, L)
    volt = C.load_volt(cn)
    out: dict = {"n_pumps": int(len(p)), "n_meters": int(p["meter_idx"].nunique())}
    # ------------------------------------------------ class totals per basis
    tot = {}
    for b, E in B.items():
        cm = C.class_month(p, E)
        tot[b] = {"annual_2011_22": C.window_mean(cm, Y0, Y1),
                  "janmay_2011_22": C.window_mean(cm, Y0, Y1, JM),
                  "annual_2011_24": C.window_mean(cm, 2011, 2024),
                  "janmay_2011_24": C.window_mean(cm, 2011, 2024, JM)}
    out["kwh_GWh"] = {b: {w: {c: float(v[c]) / 1e6 for c in K.CLASSES} | {
        "all": float(v.sum()) / 1e6} for w, v in d.items()} for b, d in tot.items()}
    # ------------------------------------------------ (a)
    ref = wref[cn]
    a = {}
    for b in ("kept", "dedup", "raw"):
        a[b] = {
            "janmay_2011_22": ref["janmay_2011_22"] / float(tot[b]["janmay_2011_22"]["irrigation"]),
            "janmay_2011_24": ref["janmay_2011_24"] / float(tot[b]["janmay_2011_24"]["irrigation"]),
            "annual_2011_24": ref["annual_2011_24"] / float(tot[b]["annual_2011_24"]["irrigation"]),
        }
    a["primary"] = "kept.janmay_2011_22 (the scored test's reference)"
    a["u7_note"] = ("2011_24 variants contain 2023-24 (2 of 14 years); reported only, never "
                    "gating. janmay_2011_22 removes the printed 2023 and 2024 Jan-May values "
                    "from the printed 14-year Jan-May mean.")
    a["rounding_halfwidth_m3"] = ref["janmay_2011_22_rounding_halfwidth"]
    out["a_m3_per_kwh"] = a
    # ------------------------------------------------ STEP 0 decomposition (irrigation)
    ct = c_twin()
    dk = float(tot["dedup"]["janmay_2011_22"]["irrigation"])
    kk = float(tot["kept"]["janmay_2011_22"]["irrigation"])
    rk = float(tot["raw"]["janmay_2011_22"]["irrigation"])
    gap = ref["janmay_2011_22"] / (ct * kk)
    out["step0"] = {
        "c_twin_m3_per_kwh": ct,
        "twin_irrigation_volume_janmay_m3": ct * kk,
        "wra_irrigation_volume_janmay_2011_22_m3": ref["janmay_2011_22"],
        "gap_factor_wra_over_twin": gap,
        "if_dropped_energy_is_pumping": {
            "conversion_factor": ref["janmay_2011_22"] / (ct * dk), "coverage_factor": dk / kk,
            "log_share_conversion": float(np.log(ref["janmay_2011_22"] / (ct * dk)) / np.log(gap)),
            "log_share_coverage": float(np.log(dk / kk) / np.log(gap))},
        "if_dropped_energy_is_not_pumping": {"conversion_factor": gap, "coverage_factor": 1.0},
        "raw_over_dedup_irrigation": rk / dk,
        "twin_all_class_volume_annual_m3": ct * float(tot["kept"]["annual_2011_22"].sum()),
        "wra_total_groundwater_annual_2011_24_m3": wref["_cells"][cn]["annual"]["total"] * 1e8,
    }
    # ------------------------------------------------ the dropped energy, 2012-22
    m = p[p["meter_first_pump"]].set_index("meter_idx")
    first_rows = p.index[p["meter_first_pump"].to_numpy()].to_numpy()
    Em = B["raw"][first_rows]                      # meter series
    dropped_m = ~m["kept"].to_numpy(dtype=bool)
    kept_irr = (p["purpose_class"] == "irrigation").to_numpy() & p["kept"].to_numpy(dtype=bool)
    prof_k = moy_profile(B["kept"], np.where(kept_irr)[0])
    duty_k = m.loc[m["kept"] & (m["purpose_class"] == "irrigation"), "meter_duty_2012_2022"]
    duty_k_med = float(duty_k.median())
    t0, t1 = C.tidx("2012-01"), C.tidx("2022-12") + 1
    drop = {}
    for cls in K.CLASSES:
        selm = dropped_m & (m["purpose_class"].to_numpy() == cls)
        if not selm.any():
            continue
        rows_p = np.where((p["purpose_class"] == cls).to_numpy()
                          & ~p["kept"].to_numpy(dtype=bool))[0]
        e_yr = B["dedup"][rows_p][:, t0:t1].sum() / 11.0
        prof = moy_profile(B["dedup"], rows_p)
        mm = m[selm]
        mean_kwh = Em[selm][:, t0:t1].mean(axis=1)
        hp_impl = mean_kwh / (K.KW_PER_HP * 730.0 * duty_k_med)
        drop[cls] = {
            "n_meters": int(selm.sum()), "n_pumps": int(len(rows_p)),
            "dedup_GWh_per_yr_2012_22": float(e_yr) / 1e6,
            "share_of_class_dedup_energy": float(
                e_yr / max(B["dedup"][(p["purpose_class"] == cls).to_numpy()][:, t0:t1].sum()
                           / 11.0, 1e-9)),
            "duty": q(mm["meter_duty_2012_2022"]),
            "meter_hp_sum": q(mm["meter_hp_sum"]),
            "full_load_hours_per_month": q(mm["meter_duty_2012_2022"] * 730.0),
            "share_shared_meter": float((mm["meter_n_pumps"] > 1).mean()),
            "share_high_voltage": float((volt[first_rows[selm]] == "H").mean()),
            "share_in_irrigation_district": float(mm["irrigation_district"].notna().mean()),
            "bimonthly_repeat_share": repeat_share(Em[selm]),
            "moy_corr_with_kept_irrigation": float(np.corrcoef(prof, prof_k)[0, 1]),
            "peak_month": int(np.argmax(prof) + 1),
            "moy_cv": float(prof.std() / prof.mean()) if prof.mean() > 0 else None,
            "implied_hp_at_kept_irrigation_median_duty": q(hp_impl),
            "ratio_implied_to_recorded_hp": q(hp_impl / mm["meter_hp_sum"].replace(0, np.nan)),
        }
        # share of the class's dropped energy on meters above 3x / 10x the cap
        d = mm["meter_duty_2012_2022"].to_numpy()
        e = mean_kwh
        drop[cls]["energy_share_duty_1_to_3"] = float(e[(d > 1) & (d <= 3)].sum() / e.sum())
        drop[cls]["energy_share_duty_above_10"] = float(e[d > 10].sum() / e.sum())
    out["dropped"] = drop
    out["kept_irrigation"] = {
        "duty_median": duty_k_med, "duty": q(duty_k),
        "peak_month": int(np.argmax(prof_k) + 1),
        "moy_cv": float(prof_k.std() / prof_k.mean()),
        "bimonthly_repeat_share": repeat_share(
            Em[(m["kept"] & (m["purpose_class"] == "irrigation")).to_numpy()]),
        "share_high_voltage": float((volt[first_rows[(m["kept"] & (
            m["purpose_class"] == "irrigation")).to_numpy()]] == "H").mean())}
    # ------------------------------------------------ (b) pump physics, kept irrigation
    hp = pd.to_numeric(p["pump_hp"], errors="coerce")
    dia = p["tube_in"]
    ok = (kept_irr & (hp > K.B_HP_RANGE[0]).to_numpy() & (hp <= K.B_HP_RANGE[1]).to_numpy()
          & (dia >= K.B_DIA_RANGE[0]).to_numpy() & (dia <= K.B_DIA_RANGE[1]).to_numpy())
    w_jm = C.per_pump_window(B["kept"], Y0, Y1, JM)
    w_an = C.per_pump_window(B["kept"], Y0, Y1)
    area = np.pi * (dia.to_numpy() * 0.0254) ** 2 / 4.0
    pk = K.KW_PER_HP * hp.to_numpy()
    b = {"n_pumps_used": int(ok.sum()), "n_kept_irrigation": int(kept_irr.sum()),
         "energy_share_used_janmay": float(w_jm[ok].sum() / w_jm[kept_irr].sum())}
    for v in sorted({K.B_V_RANGE[0], K.B_V_PRIMARY, K.B_V_RANGE[1]}):
        cb = v * area * 3600.0 / np.where(pk > 0, pk, np.nan)
        b[f"v{v}"] = {
            "kwh_weighted_janmay_2011_22": float(np.nansum(cb[ok] * w_jm[ok]) / w_jm[ok].sum()),
            "kwh_weighted_annual_2011_22": float(np.nansum(cb[ok] * w_an[ok]) / w_an[ok].sum()),
            "pump_median": float(np.nanmedian(cb[ok])), "pump_q": q(cb[ok]),
            "implied_tdh_m_at_eta_0.5_median": float(np.nanmedian(
                0.5 * K.M3_PER_KWH_PER_M / cb[ok]))}
    b["primary"] = f"v{K.B_V_PRIMARY}.kwh_weighted_janmay_2011_22"
    b["b2_eta_tdh_band"] = {f"eta{e}_hdyn{h}": e * K.M3_PER_KWH_PER_M / (K.LIFT_MEDIAN + h)
                            for e in K.B2_ETA for h in K.B2_HDYN}
    out["b_m3_per_kwh"] = b
    out["hp_vs_dia_spearman"] = float(pd.Series(hp[ok]).corr(dia[ok], method="spearman"))
    return out


def main() -> int:
    rep = K.REPORT_DIR["step0_ab"] / "report.json"
    if rep.exists():
        raise SystemExit(f"{rep} exists; run once")
    cells = wra.parse()
    wref = wra.irrigation_reference(cells)
    wref["_cells"] = cells
    res = {"declaration": K.PREREG["step0_ab"],
           "labels": {"step0": "exploratory; never validation",
                      "a": "design-time; WRA volumes are kWh x pump-test ratios on the same "
                           "census, so (a) sets the conversion but cannot validate it",
                      "b": "design-time; not independent of electricity"},
           "wra_cells_1e8_m3": cells,
           "wra_irrigation_reference_m3": {c: wref[c] for c in K.COUNTIES},
           "counties": {}}
    for cn in K.COUNTIES:
        print("county", cn, flush=True)
        res["counties"][cn] = county(cn, wref)
    OUTD.mkdir(parents=True, exist_ok=True)
    rows = []
    for cn, d in res["counties"].items():
        for b_, w in d["kwh_GWh"].items():
            for win, v in w.items():
                for c, x in v.items():
                    rows.append({"county": cn, "basis": b_, "window": win, "class": c,
                                 "GWh_per_window_year": x})
    pd.DataFrame(rows).to_csv(OUTD / "kwh_by_basis_class.csv", index=False)
    res["outputs"] = [rel_to_repo(OUTD / "kwh_by_basis_class.csv")]
    rep.parent.mkdir(parents=True, exist_ok=True)
    with open(rep, "x", encoding="utf-8") as f:
        f.write(json.dumps(res, indent=1, ensure_ascii=False, default=float) + "\n")
    print(json.dumps({cn: {"a": d["a_m3_per_kwh"], "step0": d["step0"],
                           "b_primary": d["b_m3_per_kwh"][f"v{K.B_V_PRIMARY}"]}
                      for cn, d in res["counties"].items()}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
