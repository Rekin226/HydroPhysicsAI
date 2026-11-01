"""N3 mapping and model side, from metadata and frozen model output only (no observation).

Two files, hashed into N3.piles_prospective.v1 before registration:

* ``mapping.json`` - each pile's site (the WRA station list deep_pile_sites.csv: name, depth,
  TWD97 coordinates), its active cell in the page grid (twin.mechanism.ring_cells), and its
  head nest: the wells of the page model's 158-well list (stage3_wells.csv: id, model layer
  1-4, coordinates; the same wells as N2's and N2C's unit) within 1.5 km (Yunlin) or 3 km
  (溪州), each with its cell; plus the declared layer-fill rule (:func:`n3lib.layer_fill`).
* ``model_side.json`` - the frozen page model (datum_gate_innov.npz, ensemble mean,
  coupled_leveling column, baseline scenario) at those cells and wells: monthly column
  increments at the pile cells and the nest dD built with the same rule as the observations,
  for the window (2026-11..2028-11) and the pre-origin hindcast (2016-05..2022-12, rank 12);
  g_mod (12 and 24 months), the model-side informative gate, the ZR level predictions, and
  for 溪州 R1c's world (datum_gate_ccz.npz, report-only): its season coefficient, the
  rehearsal of informativeness on frozen ZR heads and the reset threshold.

Column pairing (fix A2; memory 'the column pairs with its head field'): both archives were
run with ``column_heads='free'``. After the 2022-12 origin, ``heads_mean`` holds the
DISPLAYED heads (the restarted trajectory, twin.forward: 'Displayed heads keep the
restart'), while the column is driven by the free projection from the hindcast's own end
state, which the archive does not save. So the window g_mod is defined as the column
increment per metre of DISPLAYED (restarted) ZR head change, not per metre of the heads that
drove it. The hindcast (to the origin month, gain 0) is paired: its slope
(``g_mod_hindcast_paired``, 2016-06..2022-12) is the declared report-only comparator.

Reads only metadata (station lists, coordinates, model layers) and model output. Usage:
    python -I dataprep_staging/n3_piles/build_mapping.py --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(HERE), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import n3lib as L  # noqa: E402

HINDCAST = ("2016-05", "2022-12")
HINDCAST_PAIRED = ("2016-06", "2022-12")       # = g0_n3.HINDCAST_YUNLIN (rank 12's months)
MODEL_MONTHS = ("2026-09", "2028-11")
COLUMN_DRIVER = {
    "free": "the window heads in heads_mean are the DISPLAYED (restarted) trajectory; the "
            "column after the origin is driven by the free projection from the hindcast's "
            "own end state (fix A2), which the archive does not save: g_mod is the column "
            "increment per metre of DISPLAYED (restarted) ZR head change, not per metre of "
            "its driver; the hindcast slope (gain 0, to the origin month) is paired and is "
            "the report-only comparator g_mod_hindcast_paired",
    "restart": "the column is driven by the displayed (restarted) heads: window g_mod is "
               "paired",
    None: "forward_options not recorded in the archive: pairing unknown (disclosed)"}


def _grid(npz) -> dict:
    return {k: npz[k] for k in ("mask", "dx", "x0", "y0")}


def cells_of(grid: dict, xy: np.ndarray) -> np.ndarray:
    from hydrophysics.twin.mechanism import ring_cells

    return ring_cells(grid, np.atleast_2d(np.asarray(xy, float)))


def build_mapping(sites: pd.DataFrame, wells: pd.DataFrame, grid: dict) -> dict:
    """Pile -> site, cell, nest wells (with layers and cells), layer fill. Metadata only."""
    wells = wells.copy()
    wells["sid"] = wells["sid"].astype(str)
    wcell = cells_of(grid, wells[["x", "y"]].to_numpy(float))
    out = {}
    for key, spec in L.PILES.items():
        rec = dict(spec)
        for tag in ("site_id", "fallback_site_id"):
            sid = spec.get(tag)
            if not sid:
                continue
            row = sites[sites["identifier"] == sid]
            if len(row) != 1:
                raise SystemExit(f"{sid}: expected one row in the station list")
            r = row.iloc[0]
            if str(r["pile"]) != spec["pile"]:
                raise SystemExit(f"{sid}: station list names {r['pile']}, not {spec['pile']}")
            c = int(cells_of(grid, [[r["x_twd97"], r["y_twd97"]]])[0])
            rec[tag.replace("site_id", "site")] = {
                "identifier": sid, "depth_m": int(r["depth_m"]), "x": float(r["x_twd97"]),
                "y": float(r["y_twd97"]), "cell": c}
        site = rec["site"]
        if site["cell"] < 0:
            raise SystemExit(f"{key}: pile off the active grid")
        d = np.hypot(wells["x"] - site["x"], wells["y"] - site["y"])
        sel = (d <= L.NEST_RADIUS_M[key]).to_numpy()
        nest = [{"sid": s, "layer": int(lay), "dist_m": round(float(dd), 1), "cell": int(c)}
                for s, lay, dd, c in zip(wells["sid"][sel], wells["layer"][sel], d[sel],
                                         wcell[sel], strict=True)]
        nest.sort(key=lambda w: (w["dist_m"], w["sid"]))
        if any(w["cell"] < 0 for w in nest):
            raise SystemExit(f"{key}: a nest well is off the grid")
        rec["nest"] = nest
        rec["nest_radius_m"] = L.NEST_RADIUS_M[key]
        rec["layer_fill"] = {str(k): v for k, v in
                             L.layer_fill({w["layer"] for w in nest}).items()}
        out[key] = rec
    return out


def month_labels(dates) -> list[str]:
    return [str(d)[:7] for d in dates]


def _series(values: np.ndarray, labels: list[str], span: tuple[str, str]) -> pd.Series:
    s = pd.Series(np.asarray(values, float), index=labels)
    return s.loc[span[0]:span[1]]


def model_world(npz_path: Path, mapping: dict, keys, span: tuple[str, str]) -> dict:
    """Column increments (cm) at the pile cells and nest dD (m) from model heads at the nest
    wells' cells and layers (and the cell's own layer-mean dD), months of ``span``."""
    with np.load(npz_path, allow_pickle=False) as z:
        labels = month_labels(z["dates"])
        rhe = [str(v) for v in z["rheology_labels"]]
        if rhe[0] != L.RHEOLOGY:
            raise SystemExit(f"{npz_path}: rheology axis 0 is {rhe[0]}, not {L.RHEOLOGY}")
        if str(z["scenario_names"][0]) != "baseline":
            raise SystemExit(f"{npz_path}: scenario 0 is not the baseline")
        subs = z["subs_mean_by_rheology"][0, 0]
        heads = z["heads_mean"][0]
        origin = int(z["origin"])
        opts = json.loads(str(z["forward_options"])) if "forward_options" in z else {}
        out = {"origin_month": labels[origin], "piles": {},
               "column_heads": opts.get("column_heads"), "gain": (
                   float(z["gain"]) if "gain" in z else None),
               "hindcast_gain": float(z["hindcast_gain"]) if "hindcast_gain" in z else None}
        for key in keys:
            m = mapping[key]
            c = m["site"]["cell"]
            S = _series(subs[c] * 100.0, labels, (L.prev_month(span[0]), span[1]))
            inc = S.diff().iloc[1:]
            hw = {w["sid"]: _series(heads[w["layer"] - 1, w["cell"]], labels,
                                    (L.prev_month(span[0]), span[1])) for w in m["nest"]}
            lay = {w["sid"]: w["layer"] for w in m["nest"]}
            ms = list(inc.index)
            dd = L.nest_dd(hw, lay, ms)
            hc = _series(heads[:, c].mean(axis=0), labels, (L.prev_month(span[0]), span[1]))
            dd_cell = -hc.diff().iloc[1:]
            out["piles"][key] = {"months": ms, "inc_cm": inc.round(6).tolist(),
                                 "dd_nest_m": dd.round(6).tolist(),
                                 "dd_cell_m": dd_cell.round(6).tolist(),
                                 "cum_cm": {k: float(S[k]) for k in (S.index[0], S.index[-1])}}
    return out


def _pooled(world: dict, keys, span: tuple[str, str]) -> dict:
    ys, xs, xc, gs = [], [], [], []
    for k in keys:
        p = world["piles"][k]
        sel = [i for i, m in enumerate(p["months"]) if span[0] <= m <= span[1]]
        ys += [p["inc_cm"][i] for i in sel]
        xs += [p["dd_nest_m"][i] for i in sel]
        xc += [p["dd_cell_m"][i] for i in sel]
        gs += [k] * len(sel)
    y, x, x2, g = map(np.asarray, (ys, xs, xc, gs))
    ok = np.isfinite(y) & np.isfinite(x)
    return {"g_mod": L.fe_slope(y[ok], x[ok], g[ok]), "n": int(ok.sum()),
            "g_mod_cell_report_only": L.fe_slope(y, x2, g)}


def model_side(mapping: dict, forward: Path, r1c_forward: Path | None) -> dict:
    span = (L.REF_MONTH, L.WINDOW24[1])
    world = model_world(forward, mapping, L.PRIMARY, span)
    hind = model_world(forward, mapping, L.PRIMARY, HINDCAST)
    w12 = _pooled(world, L.PRIMARY, L.WINDOW)
    w24 = _pooled(world, L.PRIMARY, L.WINDOW24)
    paired = _pooled(hind, L.PRIMARY, HINDCAST_PAIRED)
    ch = world.get("column_heads")
    pairing = {"column_heads": ch, "gain": world.get("gain"),
               "hindcast_gain": world.get("hindcast_gain"),
               "window_heads_drive_column": ch == "restart",
               "hindcast_paired": bool((world.get("hindcast_gain") or 0.0) == 0.0),
               "wording": COLUMN_DRIVER.get(ch, COLUMN_DRIVER[None])}
    gate, level = {}, {}
    for k in L.PRIMARY:
        p = world["piles"][k]
        sel = [i for i, m in enumerate(p["months"]) if L.WINDOW[0] <= m <= L.WINDOW[1]]
        dd = np.asarray([p["dd_nest_m"][i] for i in sel])
        inc = np.asarray([p["inc_cm"][i] for i in sel])
        rd, ri = float(np.sqrt(np.nanmean(dd ** 2))), float(np.sqrt(np.nanmean(inc ** 2)))
        gate[k] = {"rms_dd_m": rd, "rms_inc_cm": ri,
                   "pass": bool(rd >= L.MIN_RMS_DD_M and ri >= L.MIN_RMS_INC_CM)}
        inc_by = dict(zip(p["months"], p["inc_cm"], strict=True))
        z12 = float(sum(inc_by[m] for m in L.months(*L.WINDOW)))     # from the 2026-11 state
        z24 = float(sum(inc_by[m] for m in L.months(*L.WINDOW24)))
        level[k] = {"zr_12m_cm": z12, "zr_24m_cm": z24, "report_only_f08_12m_cm": 0.8 * z12,
                    "report_only_f08_24m_cm": 0.8 * z24}
    out = {"forward": {"path": L.FORWARD, "sha256": L.sha256(forward)},
           "column": {"path": L.COLUMN, "sha256": L.sha256(REPO / L.COLUMN)},
           "rheology": L.RHEOLOGY, "scenario": "baseline (ZR: climatology pumping after the "
                                               "2022-12 origin)",
           "world": world, "hindcast": hind,
           "g_mod_12": w12, "g_mod_24": w24,
           "g_mod_definition": "column increment (cm) per metre of DISPLAYED ZR nest head "
                               "change (n3lib.nest_dd on heads_mean), window months; see "
                               "column_pairing",
           "column_pairing": pairing,
           "g_mod_hindcast_paired": {**paired, "months": list(HINDCAST_PAIRED),
                                     "role": "report-only comparator (paired: the hindcast "
                                             "heads drove the column); never the decision's "
                                             "g_mod"},
           "informative_model_side": gate,
           "informative_model_side_pass": all(v["pass"] for v in gate.values()),
           "zr_level": level,
           "depth_convention": "the column (total depth, driven by the four-layer mean) is "
                               "compared unscaled with piles anchored at 300/400 m: every "
                               "model aquifer's median screen depth (mid zone 35.5, 119.8, "
                               "214, 289 m) lies above both anchors; compaction below the "
                               "anchor is in the column but not in the pile (biases M "
                               "high); a factor 0.8 variant is report-only"}
    if r1c_forward is not None:
        xw = model_world(r1c_forward, mapping, ("xizhou",), span)
        xh = model_world(r1c_forward, mapping, ("xizhou",), HINDCAST)
        p = xw["piles"]["xizhou"]
        inc = pd.Series(p["inc_cm"], index=p["months"])
        dd = pd.Series(p["dd_nest_m"], index=p["months"])
        win = inc.loc[L.WINDOW[0]:L.WINDOW[1]]
        thr = L.discontinuity_threshold(win.to_numpy())
        sc = L.season_coefficients(inc, dd, [2027], threshold_cm=thr)
        s0 = sc["seasons"][0]
        reb = s0.get("rebound_cm")
        ph = xh["piles"]["xizhou"]
        hinc = pd.Series(ph["inc_cm"], index=ph["months"])
        hsc = L.season_coefficients(hinc, pd.Series(ph["dd_nest_m"], index=ph["months"]),
                                    [2019, 2020, 2021, 2022],
                                    threshold_cm=L.discontinuity_threshold(hinc.to_numpy()))
        out["xizhou_r1c"] = {
            "forward": {"path": L.R1C_FORWARD, "sha256": L.sha256(r1c_forward)},
            "column": {"path": L.R1C_COLUMN, "sha256": L.sha256(REPO / L.R1C_COLUMN)
                       if (REPO / L.R1C_COLUMN).is_file() else None},
            "world": xw, "hindcast": xh, "season_2027_model": sc,
            "k_mod_2027": s0.get("k_cm_per_m"), "discontinuity_threshold_cm": thr,
            "k_mod_definition": "R1c's column rebound per metre of DISPLAYED (restarted) ZR "
                                "nest head rise, Jun-Nov 2027 (column_heads "
                                f"{xw.get('column_heads')!r}; see column_pairing)",
            "k_mod_hindcast_paired": {"seasons": hsc, "k": hsc["pooled"]["k"],
                                      "role": "report-only comparator (R1c's hindcast seasons "
                                              "2019-22, paired heads)"},
            "rehearsal": {"rebound_cm": reb, "rise_m": s0.get("rise_m"),
                          "heads": "displayed (restarted) ZR heads at the nest wells",
                          "informative": bool(reb is not None and abs(reb) >= 2 * L.PRINT_RES_CM
                                              and "k_cm_per_m" in s0),
                          "rule": "frozen ZR heads only (never X1 or PR): the model season "
                                  "rebound >= 0.2 cm (twice the print rounding) and the "
                                  "season rise >= 0.1 m; report-only, never gates"}}
    return out


def _sha_json(obj) -> str:
    import hashlib

    return hashlib.sha256((json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=1)
                           + "\n").encode()).hexdigest()


def run(out: Path, forward: Path | None = None, r1c_forward: Path | None = None,
        sites_csv: Path | None = None, wells_csv: Path | None = None,
        check_sha: bool = True) -> dict:
    forward = Path(forward or REPO / L.FORWARD)
    if check_sha and L.sha256(forward) != L.FORWARD_SHA:
        raise SystemExit(f"{forward} is not the registered forward (sha {L.FORWARD_SHA[:8]})")
    sites = pd.read_csv(sites_csv or REPO / L.PILE_SITES)
    wells = pd.read_csv(wells_csv or REPO / L.WELLS, dtype={"sid": str})
    with np.load(forward, allow_pickle=False) as z:
        grid = _grid(z)
    mapping = build_mapping(sites, wells, grid)
    r1c = Path(r1c_forward) if r1c_forward else REPO / L.R1C_FORWARD
    ms = model_side(mapping, forward, r1c if r1c.is_file() else None)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    srcs = {"sites": sites_csv or REPO / L.PILE_SITES, "wells": wells_csv or REPO / L.WELLS}
    mdoc = {"schema": 1, "what": "N3 pile/nest mapping (metadata only)",
            "sources": {k: {"path": str(Path(v).name), "sha256": L.sha256(v)}
                        for k, v in srcs.items()},
            "layer_rule": "per model layer: mean over the nest wells screened there of the "
                          "month-mean head change; an unscreened layer takes the screened "
                          "layer nearest in median screen depth (mid zone 35.5, 119.8, 214, "
                          "289 m; a tie takes both); dD = the column's equal layer weights "
                          "(twin.forward heads.mean(dim=0)) over the four filled layers",
            "excluded": L.EXCLUDED, "piles": mapping}
    (out / "mapping.json").write_text(json.dumps(mdoc, ensure_ascii=False, indent=1,
                                                 sort_keys=True) + "\n", encoding="utf-8")
    ms["mapping_sha256"] = L.sha256(out / "mapping.json")
    (out / "model_side.json").write_text(json.dumps(ms, ensure_ascii=False, indent=1,
                                                    sort_keys=True) + "\n", encoding="utf-8")
    return {"mapping": mdoc, "model_side": ms}


def summary(ms: dict) -> dict:
    s = {"g_mod_12": ms["g_mod_12"], "g_mod_24": ms["g_mod_24"],
         "g_mod_hindcast_paired": ms["g_mod_hindcast_paired"],
         "column_pairing": ms["column_pairing"],
         "informative_model_side": ms["informative_model_side"],
         "zr_level": ms["zr_level"]}
    if "xizhou_r1c" in ms:
        s["xizhou_r1c"] = {k: ms["xizhou_r1c"][k] for k in ("k_mod_2027",
                                                            "discontinuity_threshold_cm",
                                                            "rehearsal")}
    return s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    r = run(a.out)
    print(json.dumps(summary(r["model_side"]), indent=1, ensure_ascii=False))
    for n in ("mapping.json", "model_side.json"):
        print(n, L.sha256(a.out / n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
