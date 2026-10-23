"""WP S3 fix round of the S3r repair (2026-10-18), diagnosis BEFORE any method change: does
the NGL detection common mode (leave-one-out median of the off-fan reference stations,
unchanged since v1) carry composition jumps like the WRA 5-nearest median?

    dataprep_staging/tools/cpu_slot.sh python -I -B \
        dataprep_staging/s3_gnss/v4/diag_cm_composition_ngl.py OUT_DIR

Same tabulation as ``diag_cm_composition.py`` (events: a reference station's gap > 10 d, or
its record start/end inside the window), for the NGL family of both variants, with the
unchanged v3 code. Writes ``events_ngl_<variant>.csv``, ``cm_only_steps_ngl_<variant>.csv``
and ``summary_ngl.json``. No injection, no A1/A2/A3 score.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import diag_cm_composition as DC  # noqa: E402  (puts the v3 code on the path)

M, P, RV = DC.M, DC.P, DC.RV


def run_ngl_keep(p, ref, window_end) -> dict:
    stations = P.ngl_policy(window_end)[0]
    ser = P.ngl_series(window_end, stations)[0]
    equip, quakes = P.ngl_metadata(list(ser), window_end)
    fam = P.Family("ngl", {})
    cm, frames = None, []
    for ipass in range(3):
        fam.results = {k: M.process_station(s, p, cm=None if cm is None else cm.get(k),
                                            equipment=equip.get(k), earthquakes=quakes.get(k))
                       for k, s in ser.items()}
        if ipass < 2:
            R = fam.resid_frame()
            frames.append(R)
            cm = P._loo_median(R, ref)
    return {"R2": frames[1], "ref": [r for r in ref if r in frames[1]], "stations": list(ser)}


def events(run: dict, variant: str) -> pd.DataFrame:
    R = run["R2"].sort_index()
    idx = pd.date_range(R.index.min(), R.index.max(), freq="D")
    R = R.reindex(idx)
    ref = run["ref"]
    ev_by_ref = {}
    for k in ref:
        av = R[k].notna()
        first, last = av[av].index.min(), av[av].index.max()
        g = DC.gaps_of(av.loc[first:last])
        ev = [(g1, f"{k} returns") for _, g1, _ in g] + [(g0, f"{k} leaves") for g0, _, _ in g]
        ev += [(first, f"{k} starts"), (last + pd.Timedelta(days=1), f"{k} ends")]
        ev_by_ref[k] = [(d, w) for d, w in ev if idx[0] < d <= idx[-1]]
    rows = []
    W = DC.W
    for s in run["stations"]:
        nb = [m for m in ref if m != s]
        cm_s = R[nb].median(axis=1)
        days: dict = {}
        for k in nb:
            for d, w in ev_by_ref[k]:
                days.setdefault(d, []).append(w)
        for d, what in sorted(days.items()):
            lo, hi = d - pd.Timedelta(days=W), d + pd.Timedelta(days=W - 1)
            before, after = R.loc[lo:d - pd.Timedelta(days=1), nb], R.loc[d:hi, nb]
            both = [k for k in nb if before[k].notna().sum() >= 3 and after[k].notna().sum() >= 3]
            j_all = float(cm_s.loc[d:hi].median() - cm_s.loc[lo:d - pd.Timedelta(days=1)].median())
            j_c = float("nan")
            if both:
                cs = R[both].median(axis=1)
                j_c = float(cs.loc[d:hi].median() - cs.loc[lo:d - pd.Timedelta(days=1)].median())
            rows.append({"variant": variant, "station": s, "date": d.date(),
                         "what": "; ".join(what),
                         "n_reporting_before": int((before.notna().sum() >= 3).sum()),
                         "n_reporting_after": int((after.notna().sum() >= 3).sum()),
                         "n_constant_subset": len(both), "cm_jump_mm": 1000 * j_all,
                         "constant_subset_jump_mm": 1000 * j_c,
                         "composition_part_mm": 1000 * (j_all - j_c)})
    return pd.DataFrame(rows)


def cm_only(variant: str, ev: pd.DataFrame) -> pd.DataFrame:
    st = pd.read_csv(DC.PRODUCT / variant / "v3" / "steps.csv")
    st = st[(st.family == "ngl") & (st.basis == "common_mode_only")]
    out = []
    for r in st.itertuples():
        d = pd.Timestamp(r.date)
        e = ev[ev.station == r.station]
        near = e[(pd.to_datetime(e.date) - d).abs().dt.days <= 5]
        near = near[near.composition_part_mm.notna()]
        b = near.loc[near.composition_part_mm.abs().idxmax()] if len(near) else None
        out.append({"variant": variant, "station": r.station, "date": r.date,
                    "size_mm": 1000 * r.size_m, "z_own_series": r.z_own_series,
                    "event": None if b is None else f"{b.date} {b.what}",
                    "composition_part_mm": None if b is None else b.composition_part_mm})
    return pd.DataFrame(out)


def main(out: str) -> int:
    od = Path(out)
    od.mkdir(parents=True, exist_ok=True)
    summ = {}
    for variant, end in (("preorigin", P.ORIGIN_END), ("full", None)):
        p, ref, _ = RV.load_tuned(str(DC.TUNE), variant)
        run = run_ngl_keep(p, ref, end)
        ev = events(run, variant)
        ev.to_csv(od / f"events_ngl_{variant}.csv", index=False)
        co = cm_only(variant, ev)
        co.to_csv(od / f"cm_only_steps_ngl_{variant}.csv", index=False)
        cp = pd.to_numeric(ev.composition_part_mm, errors="coerce").abs()
        summ[variant] = {"reference": run["ref"], "n_events": int(len(ev)),
                         "composition_part_abs_mm": {
                             "median": float(np.nanmedian(cp)) if len(cp) else None,
                             "p90": float(np.nanpercentile(cp, 90)) if cp.notna().any() else None,
                             "max": float(np.nanmax(cp)) if cp.notna().any() else None},
                         "n_events_ge_5mm": int((cp >= 5).sum()),
                         "n_events_ge_10mm": int((cp >= 10).sum()),
                         "n_cm_only_steps": int(len(co)),
                         "n_cm_only_near_event_ge_5mm": int(
                             (pd.to_numeric(co.composition_part_mm, errors="coerce").abs() >= 5)
                             .sum()) if len(co) else 0}
        print(variant, summ[variant])
    (od / "summary_ngl.json").write_text(json.dumps(summ, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
