"""WP C0 repair round (resume 2026-10-14): sensitivity run for the residual post-origin
information left in the v3 pre-origin variant, declared as a manifest exemption (plan 2.3, U3).

The registered after-the-fact test R8 (C0.census_product.v3) found meters whose Dec-2022 and
Jan-2023 readings are equal non-zero halves (the signature of one December-2022/January-2023
bill read in 2023) but which the pre-origin Dec-Jan-paired rule does not label (their
2012-01..2022-11 history shows no Dec-Jan pairing): 1,544 Yunlin and 2,108 Changhua in-grid
meters, 0.68 % and 0.57 % of the in-grid December-2022 energy. Their 2022-12 value is kept as
read in ``preorigin/v3`` and enters the duty statistic (kept set) of ``preorigin/v3`` and
``census_L1/v3``. Selecting them needs the January-2023 reading, so they cannot be removed or
filled by a pre-origin rule; the v3 manifests declare them as an exemption, and this script is
the exemption's sensitivity run (after the fact, never fed back into any product):

* the same equal-halves selection as R8 (Dec-2022 and Jan-2023 finite, not both zero, equal
  within ``lib.PAIR_ATOL``, not Dec-Jan-paired), on every meter of the county and on the
  kept in-grid meters;
* the pre-origin duty statistic recomputed with their Dec-2022 value also left out (as v3 does
  for the paired meters): kept-meter flips, and the 2012-2022 in-grid kWh the flipped meters
  carry (kept -> dropped: their v3 pumps; dropped -> kept: their whole-meter L0 readings, an
  upper bound);
* the 2022-12 in-grid kWh the v3 pumps of untreated meters carry: share of the in-grid
  December-2022 total and of the 2012-2022 in-grid total.

The meter context is ``repair3_checks.ctx_independent`` (lib functions only, the registered
R1/R8 reconstruction). Output: reports/C0_evidence/repair3/residual_sensitivity.json.

``python -I repair3_residual.py log``  log the look first;  ``python -I repair3_residual.py run``
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib  # noqa: E402
import repair3_checks as rc  # noqa: E402  (read-only helpers; nothing is scored here)
from harness import prereg3  # noqa: E402

J = lib.tidx("2022-12")
W = slice(lib.tidx("2012-01"), J + 1)
OUT = lib.EVIDENCE / "repair3" / "residual_sensitivity.json"
UNIT = "C0r3:residual_unpaired_dec2022_sensitivity"


def log_look() -> None:
    prereg3.record_look(
        [UNIT], "tpc_census_kwh:2012-01..2023-01", lib.WP, "diagnose",
        members={UNIT: {"none": "C0 electricity-forcing construction object; contains no "
                                "registered well, benchmark or model cell of the "
                                "validation-unit registries",
                        "description": "after-the-fact sensitivity of the residual exemption: "
                                       "meters with Dec-2022 == Jan-2023 not labelled "
                                       "Dec-Jan-paired by the pre-origin rule"}},
        note="[C0-repair3-residual] sensitivity run of the residual post-origin information "
             "declared as an exemption of preorigin/v3 and census_L1/v3 (repair3_residual.py): "
             "uses January-2023 readings to select the meters, after the fact; counts, duty "
             "flips and kWh shares only; nothing feeds back into a product.")
    print("logged [C0-repair3-residual]")


def county(tag: str, l1: pd.DataFrame, pre: Path) -> dict:
    ctx = rc.ctx_independent(tag, l1)
    Lm, X, midx = ctx["Lm"], ctx["X"], ctx["midx"]
    paired, kept = ctx["paired"], ctx["kept"]
    # the meter structure ctx_independent used: whole county (Chiayi rebuilt from the census)
    sub = (lib.harmonise(tag, lib.read_census_l0(tag), lib.load_grid()) if tag == "chiayi"
           else l1[l1["county"] == tag])
    ig = sub.groupby("meter_idx")["in_grid"].any().reindex(midx).fillna(False) \
        .to_numpy(bool)
    hpp = pd.to_numeric(sub["pump_hp"], errors="coerce").fillna(0.0).clip(lower=0.0)
    hp_m = hpp.groupby(sub["meter_idx"].to_numpy()).sum().reindex(midx).to_numpy()
    d, j = Lm[:, J], Lm[:, J + 1]
    valid = np.isfinite(d) & np.isfinite(j) & ((d > 0) | (j > 0))
    eqh = valid & np.isclose(d, j, rtol=0, atol=lib.PAIR_ATOL)
    unt = eqh & ~paired
    duty_chk = lib.meter_duty(X, hp_m)          # must reproduce the context's duty
    fin = np.isfinite(duty_chk)
    same = bool(np.array_equal(fin, np.isfinite(ctx["duty"]))
                and (not fin.any() or np.max(np.abs(duty_chk[fin] - ctx["duty"][fin]))
                     <= 1e-12))
    X2 = X.copy()
    X2[unt, J] = np.nan
    duty2 = lib.meter_duty(X2, hp_m)
    kept2 = ~(duty2 > lib.CAP_DUTY)
    to_drop = kept & ~kept2
    to_keep = ~kept & kept2

    z = np.load(pre / rc.pumps_file(tag), allow_pickle=False)
    val, cell, pm = z["value"], z["cell"], z["meter_idx"]
    g = cell >= 0
    ing_dec = float(np.nansum(val[g, J]))
    ing_tot = float(np.nansum(val[g][:, W]))
    unt_set = set(midx[unt].tolist())
    on_unt = g & np.isin(pm, list(unt_set))
    e_unt = float(np.nansum(val[on_unt, J]))
    drop_set = list(set(midx[to_drop].tolist()))
    e_drop = float(np.nansum(val[g & np.isin(pm, drop_set)][:, W]))
    e_keep_upper = float(np.nansum(Lm[to_keep][:, W]))
    return {
        "meters": int(len(midx)), "meters_valid_dec_jan": int(valid.sum()),
        "meters_equal_halves": int(eqh.sum()), "untreated_meters_all": int(unt.sum()),
        "untreated_meters_kept_in_grid": int((unt & kept & ig).sum()),
        "duty_recomputation_equals_ctx": bool(same),
        "kept_flips_kept_to_dropped": int(to_drop.sum()),
        "kept_flips_dropped_to_kept": int(to_keep.sum()),
        "kept_flips_in_grid": int(((to_drop | to_keep) & ig).sum()),
        "in_grid_kWh_2012_2022_of_meters_kept_to_dropped": e_drop,
        "meter_kWh_2012_2022_of_meters_dropped_to_kept_upper": e_keep_upper,
        "dec2022_in_grid_kWh_v3": ing_dec,
        "dec2022_in_grid_kWh_on_untreated_meters": e_unt,
        "share_of_dec2022_in_grid": e_unt / ing_dec if ing_dec else None,
        "share_of_2012_2022_in_grid": e_unt / ing_tot if ing_tot else None,
        "in_grid_kWh_2012_2022_v3": ing_tot,
    }


def run() -> dict:
    t0 = time.time()
    pre = rc.vn()["preorigin"]
    l1 = rc.l1_v2()
    res = {"time_utc": prereg3._now(),
           "note": "after-the-fact sensitivity of the residual exemption (uses January-2023 "
                   "readings to select meters); not a registered test; never fed back",
           "selection": "Dec-2022 and Jan-2023 finite, not both 0, |Dec - Jan| <= "
                        f"{lib.PAIR_ATOL} kWh, meter not Dec-Jan-paired by the pre-origin rule",
           "products": {"preorigin": lib.rel(pre), "census_L1": lib.rel(rc.vn()["census_L1"])},
           "per_county": {t: county(t, l1, pre) for t in lib.COUNTIES}}
    res["seconds"] = round(time.time() - t0, 1)
    lib.save_json(res, OUT)
    return res


def main(argv: list[str]) -> int:
    if argv == ["log"]:
        log_look()
        return 0
    if argv == ["run"]:
        r = run()
        for t, v in r["per_county"].items():
            print(t, v)
        return 0
    raise SystemExit(__doc__)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
