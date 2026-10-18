"""WP S4a step 5: validate the fetched LiCSAR subsets and build the network/epoch inventory.

Runs in the ``licsbas`` env (GDAL). Read-only on L0. For one frame it checks every file the
fetch log says was written:

* the file opens as a georeferenced GeoTIFF with the expected data type (unw float32,
  cc uint8; metadata rasters float32 or uint8), its geotransform equals the remote file's
  geotransform shifted by the logged source window, and its size equals that window (the
  declared S4a-5 problem checks);
* the valid-pixel fraction over the whole window (unw valid = finite and != 0, cc valid =
  > 0); an empty window is a warning (it can be genuine portal content);
* index alignment of every pair: valid unw pixels over the sea of the frame landmask (0 when
  aligned) and the best integer shift of the pair's coherence against the coherence of the
  frame's first fetched pair ((0, 0) when aligned); misalignment is a warning.

No coherence statistic and no validity at the twin's cells is computed (r3): S4b fixes its
own per-zone coherence acceptance under its own pre-registration, so S4a does not look at
coherence or coverage of the fan before that.

It then summarises the network of fetched pairs (epochs, per-year counts, connected
components, gaps), the processing-batch grids (distinct geotransform origin and spacing), the
transfer per layout class, which off-fan GNSS stations of ``run_config.json`` (outside every
active cell of the twin grid, exact test in EPSG:3826) fall on valid unw pixels (pixel of
the station on the frame's reference grid, by index), and writes
``results/dataprep/insar_tools/s4a_run/validation_<frame>.json`` and ``pairs_<frame>.csv``.

Usage (licsbas env): python -I validate.py --frame F
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import network as nw  # noqa: E402
import numpy as np  # noqa: E402
from config import FRAMES, IFG_EXTS, RUN_CONFIG, RUN_DIR, TWIN_GRID  # noqa: E402

EXPECTED_DTYPE = {"unw": "Float32", "cc": "Byte"}
SHIFTS = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def last_records(log_path: Path) -> dict[str, dict]:
    """Last log record per output file (reruns append)."""
    out: dict[str, dict] = {}
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            r = json.loads(line)
            if r.get("status") == "ok" or r["file"] not in out:
                out[r["file"]] = r            # a later failure never hides an earlier success
    return out


def grid_id(native_gt) -> str:
    """Identifier of a portal batch's native grid: pixel spacing and origin."""
    g = native_gt
    return f"dx{g[1]:.10f}_x{g[0]:.7f}_y{g[3]:.7f}"


def pix(gt, lon, lat):
    c = np.floor((lon - gt[0]) / gt[1]).astype(int)
    r = np.floor((lat - gt[3]) / gt[5]).astype(int)
    return r, c


def read_array(gdal, path):
    """Band 1 of a raster as an array (the dataset is kept alive while the band is read)."""
    ds = gdal.Open(str(path))
    return ds.GetRasterBand(1).ReadAsArray()


def check_file(gdal, path: Path, rec: dict, kind: str) -> dict:
    ds = gdal.Open(str(path))
    b = ds.GetRasterBand(1)
    a = b.ReadAsArray()
    gt = ds.GetGeoTransform()
    out = {"dtype": gdal.GetDataTypeName(b.DataType), "shape": [ds.RasterXSize,
                                                                 ds.RasterYSize],
           "geotransform": list(gt), "nodata": b.GetNoDataValue()}
    probs = []
    g0 = rec["geotransform"]
    c0, r0, nc, nr = rec["src_window"]
    exp = [g0[0] + c0 * g0[1], g0[1], 0.0, g0[3] + r0 * g0[5], 0.0, g0[5]]
    if not np.allclose(gt, exp, rtol=0, atol=1e-9):
        probs.append(f"geotransform {gt} != expected {exp}")
    if [ds.RasterXSize, ds.RasterYSize] != [nc, nr]:
        probs.append(f"size {[ds.RasterXSize, ds.RasterYSize]} != window {[nc, nr]}")
    if kind in EXPECTED_DTYPE and out["dtype"] != EXPECTED_DTYPE[kind]:
        probs.append(f"dtype {out['dtype']} != {EXPECTED_DTYPE[kind]}")
    if out["dtype"] != rec.get("dtype"):
        probs.append(f"dtype {out['dtype']} != remote {rec.get('dtype')}")
    valid = (np.isfinite(a) & (a != 0)) if a.dtype.kind == "f" else (a > 0)
    out["valid_frac_window"] = float(valid.mean())
    # not one of the declared S4a-5 problem checks: an empty window is reported as a warning
    # (it can be genuine portal content, e.g. an epoch whose acquisition misses the fan)
    out["warnings"] = (["no valid pixel in the window"]
                       if out["valid_frac_window"] == 0.0 and kind in EXPECTED_DTYPE else [])
    out["problems"] = probs
    return out


def cc_best_shift(cc: np.ndarray, ref: np.ndarray) -> tuple[list[int] | None, float | None]:
    """Integer shift (dy, dx) in [-1, 1]^2 maximising the correlation of a pair's coherence with
    the frame's reference coherence raster (stable features: towns, roads, rivers, coast)."""
    best, rbest = None, -2.0
    r = ref[2:-2, 2:-2].astype(float)
    for dy, dx in SHIFTS:
        s = np.roll(np.roll(cc, dy, 0), dx, 1)[2:-2, 2:-2].astype(float)
        m = (s > 0) & (r > 0)
        if m.sum() < 1000:
            return None, None
        c = float(np.corrcoef(s[m], r[m])[0, 1])
        if c > rbest:
            best, rbest = [dy, dx], c
    return best, round(rbest, 4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True, choices=sorted(FRAMES))
    a = ap.parse_args()
    from osgeo import gdal
    gdal.UseExceptions()

    spec = FRAMES[a.frame]
    l0 = Path(spec["l0_dir"])
    cfg = json.loads(RUN_CONFIG.read_text())
    z = np.load(RUN_DIR / "twin_cells.npz")
    m = z["mask"].astype(bool)
    cells = {"lon": z["lon"][m], "lat": z["lat"][m]}
    recs = last_records(RUN_DIR / f"fetch_log_{a.frame}.jsonl")

    listed = nw.window_pairs(nw.parse_pairs(l0 / "listings/interferograms.html"),
                             spec["start"], spec["end"])
    unavailable = []
    for p in listed:
        have = nw.pair_products(l0 / "listings" / "ifg_dirs" / f"{p}.html", p, IFG_EXTS)
        if not all(have.values()):
            unavailable.append({"pair": p, **have})

    rows, problems, warnings, fetched = [], [], [], []
    # static references for the index-alignment checks (all on the frame index window): the
    # frame landmask (LiCSAR masks unw over the sea, so an aligned unw has no valid pixel where
    # landmask == 0) and the coherence of the frame's first fetched pair
    lm = read_array(gdal, l0 / "GEOC" / f"{a.frame}.geo.landmask.tif")
    sea = ~(np.isfinite(lm) & (lm > 0))
    cc_ref_pair = next((p for p in listed if (l0 / "GEOC" / p / f"{p}.geo.cc.tif").exists()),
                       None)
    cc_ref = read_array(gdal, l0 / "GEOC" / cc_ref_pair / f"{cc_ref_pair}.geo.cc.tif")
    by_class = defaultdict(lambda: {"n": 0, "remote": 0, "transferred": 0, "out": 0})
    for p in listed:
        row = {"frame": a.frame, "pair": p, "epoch1": p[:8], "epoch2": p[9:],
               "btemp_days": (dt.date(int(p[9:13]), int(p[13:15]), int(p[15:17]))
                              - dt.date(int(p[:4]), int(p[4:6]), int(p[6:8]))).days}
        ok = True
        for e in IFG_EXTS:
            f = l0 / "GEOC" / p / f"{p}.geo.{e}.tif"
            rec = recs.get(str(f))
            if not f.exists() or rec is None or rec.get("status") != "ok":
                ok = False
                row[f"{e}_status"] = "not_fetched" if rec is None else rec.get("status")
                continue
            chk = check_file(gdal, f, rec, e)
            if chk["problems"]:
                problems.append({"file": str(f), "problems": chk["problems"]})
            if chk["warnings"]:
                warnings.append({"file": str(f), "warnings": chk["warnings"]})
            cls = (f"{e}:{rec.get('compression') or 'NONE'}:"
                   f"{'tiled' if rec['native_block'][1] > 1 else 'strip'}")
            bc = by_class[cls]
            bc["n"] += 1
            bc["remote"] += rec["remote_bytes"]
            bc["transferred"] += rec.get("bytes_transferred") or 0
            bc["out"] += rec["out_bytes"]
            row.update({f"{e}_status": "ok", f"{e}_file": str(f.relative_to(l0)),
                        f"{e}_sha256": sha256(f), f"{e}_bytes": f.stat().st_size,
                        f"{e}_remote_bytes": rec["remote_bytes"],
                        f"{e}_transferred": rec.get("bytes_transferred"),
                        f"{e}_class": cls,
                        f"{e}_area_or_point": rec.get("area_or_point"),
                        f"{e}_remote_last_modified": rec.get("last_modified"),
                        f"{e}_valid_frac_window": round(chk["valid_frac_window"], 4)})
            if e == "unw":
                row["georef_variant"] = grid_id(rec["geotransform"])
                row["on_reference_grid"] = rec.get("window_rule") == "frame_index_window"
                row["gt_matches_reference"] = rec.get("geotransform_matches_reference")
                row["shape"] = "x".join(map(str, chk["shape"]))
        if ok:
            # index alignment. (1) unw against the static landmask: 0 valid unw pixels over the
            # sea when aligned (hundreds at a 1-px shift). (2) cc against the reference cc: best
            # integer shift (0, 0). (3) descriptive: valid unw where cc == 0, which is 0 for
            # most portal generations but a few hundred boundary pixels for the 2025-12/2026-01
            # deflate-strip unw generation, whose unw and cc are each aligned by (1) and (2)
            ua = read_array(gdal, l0 / row["unw_file"])
            ca = read_array(gdal, l0 / row["cc_file"])
            if ua.shape == ca.shape == sea.shape:
                uv = np.isfinite(ua) & (ua != 0)
                row["n_unw_valid_where_cc0"] = int((uv & (ca == 0)).sum())
                row["n_unw_valid_on_sea"] = int((uv & sea).sum())
                sh, rc = cc_best_shift(ca, cc_ref)
                row["cc_best_shift"] = "none" if sh is None else f"{sh[0]},{sh[1]}"
                row["cc_ref_corr"] = rc
                if row["n_unw_valid_on_sea"] > 0 or (sh is not None and sh != [0, 0]):
                    warnings.append({"file": row["unw_file"], "warnings": [
                        f"alignment: unw valid on sea {row['n_unw_valid_on_sea']}, cc best "
                        f"shift {sh}"]})
            else:
                row["n_unw_valid_where_cc0"] = None
                problems.append({"file": row["unw_file"],
                                 "problems": [f"unw shape {ua.shape} != cc shape {ca.shape} "
                                              f"or landmask shape {sea.shape}"]})
        row["fetched"] = ok
        if ok:
            fetched.append(p)
        rows.append(row)

    meta = {}
    for mname in ("E", "N", "U", "hgt", "landmask"):
        f = l0 / "GEOC" / f"{a.frame}.geo.{mname}.tif"
        rec = recs.get(str(f))
        if f.exists() and rec:
            chk = check_file(gdal, f, rec, mname)
            meta[mname] = {"file": str(f.relative_to(l0)), "sha256": sha256(f),
                           "grid": grid_id(rec["geotransform"]),
                           "remote_bytes": rec["remote_bytes"],
                           "transferred": rec.get("bytes_transferred"), **chk}
            if chk["problems"]:
                problems.append({"file": str(f), "problems": chk["problems"]})
        else:
            meta[mname] = {"status": "not_fetched"}

    # GNSS stations of the clip box on valid unw pixels, over a date-ordered sample of the
    # fetched pairs; S4a-9 uses the pre-origin pairs (epoch2 <= 20221231) only (U7 hygiene)
    stations = cfg.get("ngl_stations_in_box", [])
    lat0 = np.deg2rad(np.mean(cells["lat"]))
    tg = np.load(TWIN_GRID, allow_pickle=False)
    g_nx, g_ny, g_dx = int(tg["nx"]), int(tg["ny"]), float(tg["dx"])
    g_x0, g_y0 = float(tg["x0"]), float(tg["y0"])
    g_mask = np.asarray(tg["mask"]).astype(bool)
    import pyproj
    to_tm2 = pyproj.Transformer.from_crs(4326, 3826, always_xy=True)

    def dist_active_km(lon, lat):
        dx = (cells["lon"] - lon) * 111.32 * np.cos(lat0)
        dy = (cells["lat"] - lat) * 110.57
        return float(np.sqrt(dx * dx + dy * dy).min())

    def in_active_cell(lon, lat) -> bool:
        """Exact test on the twin grid (EPSG:3826; cell (i, j) spans x0 + j dx .. x0 +
        (j + 1) dx and y0 + i dx .. y0 + (i + 1) dx, the convention of twin_cells.py)."""
        x, y = to_tm2.transform(lon, lat)
        j, i = int(np.floor((x - g_x0) / g_dx)), int(np.floor((y - g_y0) / g_dx))
        return bool(0 <= i < g_ny and 0 <= j < g_nx and g_mask[i, j])

    ref = json.loads((RUN_DIR / f"reference_grid_{a.frame}.json").read_text())
    rg, (wc0, wr0) = ref["geotransform"], ref["window"][:2]
    ref_gt = [rg[0] + wc0 * rg[1], rg[1], 0.0, rg[3] + wr0 * rg[5], 0.0, rg[5]]

    def station_validity(sample):
        """Stations on valid unw pixels; the pixel is taken on the reference grid by index
        (all files of a frame share the reference index window; some unw geotransforms are
        stretched, see fetch_subsets.py)."""
        cnt = Counter()
        gt = ref_gt
        for p in sample:
            f = l0 / "GEOC" / p / f"{p}.geo.unw.tif"
            arr = read_array(gdal, f)
            for s_ in stations:
                r, c = pix(gt, np.array([s_["lon"]]), np.array([s_["lat"]]))
                r, c = int(r[0]), int(c[0])
                if 0 <= r < arr.shape[0] and 0 <= c < arr.shape[1] \
                        and np.isfinite(arr[r, c]) and arr[r, c] != 0:
                    cnt[s_["station"]] += 1
        return cnt, max(1, len(sample))

    pre = [p for p in fetched if p[9:] <= "20221231"]
    samp_pre = pre[:: max(1, len(pre) // 60)]
    samp_all = fetched[:: max(1, len(fetched) // 60)]
    c_pre, n_pre = station_validity(samp_pre)
    c_all, n_all = station_validity(samp_all)
    st_out = []
    for s_ in stations:
        d_km = dist_active_km(s_["lon"], s_["lat"])
        inside = in_active_cell(s_["lon"], s_["lat"])
        st_out.append({**s_, "dist_to_nearest_active_cell_centre_km": round(d_km, 2),
                       "in_active_cell": inside, "off_fan": not inside,
                       "valid_frac_preorigin_pairs": round(c_pre[s_["station"]] / n_pre, 3),
                       "valid_frac_all_pairs": round(c_all[s_["station"]] / n_all, 3)})
    st_summary = {"n_sampled_preorigin": len(samp_pre), "n_sampled_all": len(samp_all),
                  "n_off_fan_valid50_preorigin": sum(
                      1 for x in st_out if x["off_fan"] and x["valid_frac_preorigin_pairs"] >= 0.5),
                  "n_on_fan_valid50_preorigin": sum(
                      1 for x in st_out if not x["off_fan"]
                      and x["valid_frac_preorigin_pairs"] >= 0.5)}

    grids = Counter(r.get("georef_variant") for r in rows if r["fetched"])
    off_ref = [r["pair"] for r in rows if r["fetched"] and not r.get("on_reference_grid")]
    net = nw.network_stats(fetched)
    epochs = net.pop("epochs", [])
    per_epoch = Counter([p[:8] for p in fetched] + [p[9:] for p in fetched])
    unw_rows = [r for r in rows if r["fetched"]]
    empty = sorted({Path(w["file"]).parent.name for w in warnings
                    if "no valid pixel in the window" in w["warnings"]})
    usable = [p for p in fetched if p not in set(empty)]
    net_usable = nw.network_stats(usable)
    net_usable.pop("epochs", None)
    val = {
        "frame": a.frame, "window": [spec["start"], spec["end"]],
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "n_pairs_listed": len(listed), "n_pairs_unavailable": len(unavailable),
        "unavailable_pairs": unavailable, "n_pairs_fetched": len(fetched),
        "n_pairs_available_not_fetched": len(listed) - len(unavailable) - len(fetched),
        "network_fetched": net, "epochs_fetched": epochs,
        "pairs_empty_window": empty,
        "network_fetched_without_empty_pairs": {k: net_usable.get(k) for k in (
            "n_pairs", "n_epochs", "first_epoch", "last_epoch", "n_components", "components")},
        "pairs_per_epoch_min": min(per_epoch.values()) if per_epoch else 0,
        "epochs_2022": sum(1 for e in epochs if e.startswith("2022")),
        "georef_variants_unw": dict(grids),
        "n_pairs_off_reference_grid": len(off_ref), "pairs_off_reference_grid": off_ref,
        "reference_grid": ref, "reference_geotransform_of_window": ref_gt,
        "layout_classes": {k: dict(v) for k, v in by_class.items()},
        "bytes_transferred_ifg": sum(v["transferred"] for v in by_class.values()),
        "bytes_remote_full_ifg": sum(v["remote"] for v in by_class.values()),
        "bytes_stored_ifg": sum(v["out"] for v in by_class.values()),
        "metadata_rasters": meta,
        "metadata_georef_variant": meta.get("E", {}).get("grid"),
        "no_coherence_statistics": ("r3: S4a computes no coherence statistic and no validity "
                                    "at the twin's cells (S4b pre-registers its own)"),
        "bytes_transferred_meta": sum(v.get("transferred") or 0 for v in meta.values()),
        "valid_frac_window_unw_min": float(min(
            [r["unw_valid_frac_window"] for r in unw_rows])) if unw_rows else None,
        "alignment": {
            "cc_reference_pair": cc_ref_pair,
            "n_pairs_unw_valid_on_sea_gt0": sum(
                1 for r in unw_rows if (r.get("n_unw_valid_on_sea") or 0) > 0),
            "cc_best_shift_counts": dict(Counter(r.get("cc_best_shift") for r in unw_rows)),
            "n_pairs_unw_valid_where_cc0_gt0": sum(
                1 for r in unw_rows if (r.get("n_unw_valid_where_cc0") or 0) > 0),
            "max_unw_valid_where_cc0": max(
                [r.get("n_unw_valid_where_cc0") or 0 for r in unw_rows], default=None)},
        "gnss_stations_in_box": st_out,
        "gnss_summary": st_summary,
        "n_problems": len(problems), "problems": problems[:200],
        "n_warnings": len(warnings), "warnings": warnings[:200],
    }
    (RUN_DIR / f"validation_{a.frame}.json").write_text(json.dumps(val, indent=1) + "\n")
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in (
        "frame", "pair", "epoch1", "epoch2", "btemp_days", "fetched", "on_reference_grid",
        "shape"), k))
    with open(RUN_DIR / f"pairs_{a.frame}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({k: val[k] for k in ("frame", "n_pairs_listed", "n_pairs_unavailable",
                                          "n_pairs_fetched", "epochs_2022",
                                          "n_pairs_off_reference_grid",
                                          "bytes_transferred_ifg", "n_problems", "n_warnings",
                                          "pairs_empty_window", "alignment",
                                          "valid_frac_window_unw_min")}, indent=1))
    print("components", net.get("components"))
    print("without empty pairs", val["network_fetched_without_empty_pairs"])
    print("gnss", st_summary)
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
