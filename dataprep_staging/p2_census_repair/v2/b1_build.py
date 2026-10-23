"""P2a/P2b stage 1 (values only, before registration): billing-phase table, per-county
fan-wide profiles, production Chow-Lin split of every kept bimonthly irrigation meter, and the
P2b imputation of the open July-2026 phase-A month, for both variants.

No validation statistic is computed here (the sd model, its coverage and the skill tests come
from the registered test scripts; the sd is attached in the assembly step).

Writes to ``results/dataprep/yunlin_electricity/_work_p2abc_v2/`` (not a product):
  <variant>_profiles.npz            profile and population sums per profile county
  <variant>_<county>_meters.parquet phase table (meter_idx, code, detected phase, class,
                                    first pump in grid, Chow-Lin phase, rho, pairs re-split,
                                    profile member)
  <variant>_<county>_cl.npz         rows (meter positions), values (rows x T) after the split,
                                    resplit mask
  full_<county>_july.npz            open July-2026 phase-A meters and their imputations
  stage1.json                       counts, conservation checks, output hashes

Usage: python -I b1_build.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2lib as L  # noqa: E402
from common.hashing import sha256_file  # noqa: E402

JULY_YEARS = (2023, 2024, 2025)         # base of the open July-2026 imputation (P2b)


def pair_mean_july(M: np.ndarray, years) -> np.ndarray:
    """(n, len(years)) July value under the even split of each Jul-Aug bill (pair mean)."""
    return np.column_stack([(M[:, L.tidx(f"{y}-07")] + M[:, L.tidx(f"{y}-08")]) / 2
                            for y in years])


def build_variant(variant: str, l1: pd.DataFrame, out: dict) -> None:
    prod = L.open_c0(variant)
    v = L.VARIANTS[variant]
    cds = {c: L.CountyData(variant, c, prod=prod, l1=l1) for c in L.COUNTIES}
    profs = {}
    for c in ("yunlin", "changhua"):
        profs[c] = L.county_profile(cds[c])
    np.savez(L.WORK / f"{variant}_profiles.npz",
             **{f"{k}_{c}": np.asarray(val) for c, p in profs.items() for k, val in p.items()},
             dates=np.array([L.ym(i) for i in range(cds["yunlin"].T)]))
    info = {"profiles": {c: {"n_A": p["n_A"], "n_B": p["n_B"], "t_end": p["t_end"]}
                         for c, p in profs.items()}, "counties": {}}
    for c, cd in cds.items():
        t0 = time.time()
        prof = profs[L.PROFILE_OF[c]]["prof"]
        rows = L.chowlin_rows(cd)
        allrows, allvals, allmask = [], [], []
        rho = np.full(len(cd.meters), np.nan)
        cl_phase = np.full(len(cd.meters), "", dtype=object)
        n_rs = np.zeros(len(cd.meters), np.int64)
        cons = {"max_abs_pair_sum_change_kwh": 0.0, "max_rel_pair_sum_change": 0.0,
                "changed_outside_resplit": 0, "negative_values": 0, "windows": {}}
        for p_ in ("A", "B"):
            r = rows[p_]
            lo, hi = v["cl0"], v["win_hi"][p_]
            vals, mask, rr, op = L.apply_chowlin(cd.M, cd.F, cd.Mm, r, p_, prof, lo, hi)
            rho[r] = rr
            cl_phase[r] = p_
            n_rs[r] = mask.sum(1) // 2
            # conservation: every re-split pair keeps its sum; nothing else changes
            orig = cd.M[r].astype(np.float64)
            st = op.st
            S0 = np.nan_to_num(orig[:, st]) + np.nan_to_num(orig[:, st + 1])
            S1 = np.nan_to_num(vals[:, st]) + np.nan_to_num(vals[:, st + 1])
            d = np.abs(S1 - S0)
            cons["max_abs_pair_sum_change_kwh"] = max(cons["max_abs_pair_sum_change_kwh"],
                                                      float(d.max()) if d.size else 0.0)
            rel_ = d / np.maximum(np.abs(S0), 1.0)
            cons["max_rel_pair_sum_change"] = max(cons["max_rel_pair_sum_change"],
                                                  float(rel_.max()) if d.size else 0.0)
            same = (vals == orig) | (np.isnan(vals) & np.isnan(orig))
            cons["changed_outside_resplit"] += int((~same & ~mask).sum())
            cons["negative_values"] += int((vals[mask] < 0).sum())
            cons["windows"][p_] = {"lo": L.ym(lo), "hi_inclusive": L.ym(hi - 1),
                                   "n_pairs": int(op.K), "n_meters": int(len(r)),
                                   "last_pair": [L.ym(st[-1]), L.ym(st[-1] + 1)]}
            allrows.append(r)
            allvals.append(vals)
            allmask.append(mask)
        R = np.concatenate(allrows)
        V = np.concatenate(allvals)
        Mk = np.concatenate(allmask)
        np.savez(L.WORK / f"{variant}_{c}_cl.npz", rows=R, values=V, resplit=Mk)
        ph = cd.phase
        code, cls = cd.code, cd.mclass
        prof_member = np.where((cls == "irrigation") & cd.m_first_in_grid & (code == "M2")
                               & (ph == "A"), "A",
                               np.where((cls == "irrigation") & cd.m_first_in_grid
                                        & (code == "M1") & (ph == "B"), "B", ""))
        if c == "chiayi":
            prof_member[:] = ""
        tab = pd.DataFrame({
            "county": c, "meter_idx": cd.meters["meter_idx"].to_numpy(),
            "billing_code": code, "phase_code": cd.meters["phase_code"].astype(str).to_numpy(),
            "phase_detected": ph, "dec_jan_paired": cd.meters["dec_jan_paired"].to_numpy(),
            "class_first_pump": cls, "first_pump_in_grid": cd.m_first_in_grid,
            "chowlin_phase": cl_phase.astype(str), "rho": rho, "n_pairs_resplit": n_rs,
            "profile_member": prof_member, "profile_county": L.PROFILE_OF[c]})
        tab.to_parquet(L.WORK / f"{variant}_{c}_meters.parquet", index=False)
        # energy moved by the split (descriptive), months >= 2012-01
        w0 = L.tidx("2012-01")
        rec = np.nan_to_num(cd.M[R, w0:])
        new = np.nan_to_num(V[:, w0:])
        ci = {"n_meters": int(len(cd.meters)), "chowlin_meters": {p: int(len(x)) for p, x in
                                                                 rows.items()},
              "pairs_resplit": int(Mk.sum() // 2),
              "pairs_resplit_positive": int(((V > 0) & Mk).sum()),
              "rho_share": {str(x): float(np.mean(rho[R] == x)) for x in L.RHOS},
              "energy_rel_change_2012_on": float((new.sum() - rec.sum()) / rec.sum()),
              "mean_abs_change_over_energy_2012_on": float(np.abs(new - rec).sum() / rec.sum()),
              "conservation": cons, "pump_meter_check": cd.check,
              "seconds": round(time.time() - t0, 1)}
        if variant == "full":
            t26 = L.tidx("2026-07")
            openA = np.where(cd.Mm[:, t26] == 4)[0]           # C0 incomplete_pair_2026_07
            # recorded arm base: Jul-Aug pair means of the C0 record; Chow-Lin arm base: the
            # Chow-Lin July values (recorded values for meters outside the Chow-Lin set)
            base_rec = pair_mean_july(cd.M[openA], JULY_YEARS)
            Mcl = cd.M[openA].astype(np.float64).copy()
            pos = pd.Series(np.arange(len(R)), index=R)
            hit = pos.reindex(openA)
            ok = hit.notna().to_numpy()
            Mcl[ok] = V[hit[ok].astype(int).to_numpy()]
            base_cl = np.column_stack([Mcl[:, L.tidx(f"{y}-07")] for y in JULY_YEARS])
            imp_rec = np.mean(base_rec, axis=1)      # NaN if any base July is missing
            imp_cl = np.mean(base_cl, axis=1)
            np.savez(L.WORK / f"full_{c}_july.npz", rows=openA, imputed_recorded=imp_rec,
                     imputed_chowlin=imp_cl, base_recorded=base_rec, base_chowlin=base_cl,
                     years=np.array(JULY_YEARS))
            ci["july2026_open_phaseA"] = {
                "n_meters": int(len(openA)),
                "imputed_recorded": int(np.isfinite(imp_rec).sum()),
                "imputed_chowlin": int(np.isfinite(imp_cl).sum()),
                "left_missing": int((~np.isfinite(imp_rec)).sum())}
        info["counties"][c] = ci
        print(variant, c, {k: ci[k] for k in ("chowlin_meters", "pairs_resplit", "rho_share",
                                              "seconds")}, cons, flush=True)
    out[variant] = info


def main() -> None:
    t0 = time.time()
    L.WORK.mkdir(parents=True, exist_ok=True)
    l1 = L.l1_pumps()
    out = {"env": L.env_info()}
    for variant in ("preorigin", "full"):
        build_variant(variant, l1, out)
    out["outputs"] = {p.name: sha256_file(p) for p in sorted(L.WORK.iterdir())
                      if p.suffix in (".npz", ".parquet")}
    out["code"] = {p.name: sha256_file(p) for p in sorted(L.CODE_DIR.glob("*.py"))}
    out["seconds"] = round(time.time() - t0, 1)
    L.save_json(out, L.WORK / "stage1.json")
    print("seconds", out["seconds"])


if __name__ == "__main__":
    main()
