"""X5 T1 predictions: anchored InSAR May-to-May increments (2020-05 -> 2021-05) at the
benchmarks surveyed in both months, written BEFORE the test is registered and hashed in it.

No leveling elevation is read (positions and survey presence only). Columns (mm, + = up):

* ``p2_offset_scaled``   PRIMARY: run P2, median-polish offset anchoring, scaled to 365 d
* ``p2_offset_unscaled`` declared alternative A1 (no scaling)
* ``p2_plane_scaled``    declared alternative A2 (offset + east/north plane per epoch)
* ``v3_offset_scaled``   declared alternative A3 (the full product F2 + second frame)
* ``p2_raw_scaled``      diagnostic: no GNSS anchoring (the v3 off-fan reference only)

    python -I predict_t1.py      (writes results/dataprep/x5_insar_anchor/t1_predictions/)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent), str(HERE.parent / "s4_insar")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import x5const as K  # noqa: E402
import x5lib as X  # noqa: E402
from common.paths import rel_to_repo  # noqa: E402

OUTD = K.OUT / "t1_predictions"


def anchored(ins: dict, st: pd.DataFrame, series: dict, bx, by, method: str):
    S = X.station_inputs(ins, st, series)
    corr = X.epoch_corrections(S["I"], S["G"], S["x"], S["y"], method)
    corr["dates"] = [str(d.date()) for d in ins["dates"]]
    Ib = X.point_matrix(ins, bx, by)
    return Ib, X.correction_at(corr, bx, by), corr, S


def main() -> int:
    OUTD.mkdir(parents=True, exist_ok=True)
    pred_path = OUTD / "predictions.csv"
    if pred_path.exists():
        raise SystemExit(f"{pred_path} exists; predictions are written once")
    bench = X.bench_geometry()
    bx, by = bench["x"].to_numpy(float), bench["y"].to_numpy(float)
    st = X.anchor_stations()
    series = X.gnss_daily(st, last_day="2022-12-31")
    out = bench[["sid", "x", "y", "remark"]].copy()
    out["county"] = X.county_of_points(bx, by)
    anchors = {}
    p2 = X.load_p2()
    Ib, cb, corr, S = anchored(p2, st, series, bx, by, "offset")
    for name, A, scaled in (("p2_offset_scaled", Ib + cb, True),
                            ("p2_offset_unscaled", Ib + cb, False),
                            ("p2_raw_scaled", Ib, True)):
        r = X.may_increment(A, p2["dates"], *K.T1_YEARS, scaled=scaled)
        out[name] = r["inc"]
        if name == "p2_offset_scaled":
            out["p2_span_days"] = r["span_days"]
            out["p2_n_epochs_2020"] = r["n0"]
            out["p2_n_epochs_2021"] = r["n1"]
    anchors["P2_offset"] = corr
    stations_used = {"P2": [n for n, c in zip(S["names"], np.isfinite(S["I"] - S["G"]).sum(0),
                                              strict=True) if c > 0]}
    _, cbp, corrp, _ = anchored(p2, st, series, bx, by, "plane")
    out["p2_plane_scaled"] = X.may_increment(Ib + cbp, p2["dates"], *K.T1_YEARS)["inc"]
    anchors["P2_plane"] = corrp
    del p2
    v3 = X.load_v3()
    Iv, cv, corrv, Sv = anchored(v3, st, series, bx, by, "offset")
    out["v3_offset_scaled"] = X.may_increment(Iv + cv, v3["dates"], *K.T1_YEARS)["inc"]
    anchors["v3_offset"] = corrv
    stations_used["v3"] = [n for n, c in zip(Sv["names"], np.isfinite(Sv["I"] - Sv["G"]).sum(0),
                                             strict=True) if c > 0]
    out.to_csv(pred_path, index=False)
    tabs = []                       # anchor table (epoch offsets, stations per epoch)
    for nm, c in anchors.items():
        tabs.append(pd.DataFrame({"series": nm, "date": c["dates"],
                                  "c_mm": c["c"], "a_mm_per_km": c["a"], "b_mm_per_km": c["b"],
                                  "n_stations": c["n"]}))
    pd.concat(tabs).to_csv(OUTD / "anchors.csv", index=False)
    summ = {"n_benchmarks": int(len(out)),
            "n_finite": {c: int(np.isfinite(out[c]).sum()) for c in out.columns
                         if c.startswith(("p2_", "v3_")) and "n_epochs" not in c
                         and "span" not in c},
            "stations_used": stations_used,
            "median_stations_per_epoch": {nm: float(np.median(c["n"])) for nm, c in
                                          anchors.items()},
            "note": "predictions only; no leveling elevation was read",
            "code": [rel_to_repo(HERE / f) for f in ("predict_t1.py", "x5lib.py",
                                                     "x5const.py")]}
    (OUTD / "summary.json").write_text(json.dumps(summ, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summ, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
