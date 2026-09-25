"""Public line geometry for the decision page: the THSR centreline and the fan's rivers.

    python -m hydrophysics.twin.app.geo --out-dir hydrophysics/twin/app/geodata

Both come from OpenStreetMap through the Overpass API (no key; ODbL, attributed on the
page). They are public infrastructure and hydrography, not project data, so the traced
files are committed and the page builder never touches the network. ``data/`` is
gitignored repo-wide, which is why they live under ``app/geodata`` rather than
``twin/data``.

- **THSR**: every ``railway=rail + highspeed=yes`` way over the fan. OSM draws the line
  as several ways (viaduct, bridge, the two directions in places), so they are merged into
  one centreline: points are projected onto the corridor's principal axis, averaged in
  250 m bins across all ways, lightly smoothed and resampled every 250 m of chainage from
  the northern end. Stations: 彰化 Changhua and 雲林 Yunlin (OSM ``railway=station``).
- **Rivers**: the ways named 濁水溪 (Choushui), 新虎尾溪 (Xinhuwei) and 北港溪 (Beigang),
  simplified to 200 m. The flow model treats them as no-flow (STATE.md §2), which is what
  the page's river legend says.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

OVERPASS = "https://overpass-api.de/api/interpreter"
GEODATA = os.path.join(os.path.dirname(__file__), "geodata")
HSR_CSV = os.path.join(GEODATA, "thsr_choushui.csv")
HSR_STATIONS_CSV = os.path.join(GEODATA, "thsr_stations.csv")
RIVERS_CSV = os.path.join(GEODATA, "rivers_choushui.csv")
RIVERS = {"濁水溪": "Choushui", "新虎尾溪": "Xinhuwei", "北港溪": "Beigang"}
# the fan's lon/lat box, generous; points are clipped to the model grid by the builder
BBOX = (23.45, 120.10, 24.10, 120.75)
ATTRIB = "Rail and rivers: © OpenStreetMap contributors (ODbL)"


def trace_centreline(parts: list[np.ndarray], step: float = 250.0,
                     smooth: int = 3) -> np.ndarray:
    """Merge several polylines of one corridor into a centreline -> (N, 3) chainage, x, y.

    Points of every part are projected on the corridor's principal axis, binned at ``step``
    and averaged (so two parallel tracks give their middle), smoothed with a ``smooth``-bin
    moving average and resampled every ``step`` metres of arc length. Chainage starts at 0
    at the end with the largest y (north).
    """
    pts = np.concatenate([np.asarray(p, dtype="float64") for p in parts if len(p)])
    c = pts.mean(0)
    u, s, vt = np.linalg.svd(pts - c, full_matrices=False)
    ax = vt[0] if vt[0][1] < 0 else -vt[0]            # axis points north -> south
    t = (pts - c) @ ax
    edges = np.arange(t.min(), t.max() + step, step)
    b = np.clip(np.digitize(t, edges) - 1, 0, len(edges) - 1)
    keep = np.unique(b)
    cx = np.array([pts[b == k, 0].mean() for k in keep])
    cy = np.array([pts[b == k, 1].mean() for k in keep])
    if smooth > 1 and len(cx) > smooth:
        ker = np.ones(smooth) / smooth
        pad = smooth // 2
        cx = np.convolve(np.pad(cx, pad, mode="edge"), ker, mode="valid")
        cy = np.convolve(np.pad(cy, pad, mode="edge"), ker, mode="valid")
    seg = np.hypot(np.diff(cx), np.diff(cy))
    arc = np.r_[0.0, np.cumsum(seg)]
    ch = np.arange(0.0, arc[-1] + 1e-9, step)
    return np.column_stack([ch, np.interp(ch, arc, cx), np.interp(ch, arc, cy)])


def simplify(xy: np.ndarray, tol: float) -> np.ndarray:
    """Douglas-Peucker simplification of a polyline (N, 2) to tolerance ``tol`` metres."""
    xy = np.asarray(xy, dtype="float64")
    if len(xy) < 3:
        return xy
    keep = np.zeros(len(xy), dtype=bool)
    keep[[0, -1]] = True
    stack = [(0, len(xy) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = xy[i], xy[j]
        d = b - a
        n = np.hypot(*d)
        seg = xy[i + 1:j] - a
        if n == 0:
            dist = np.hypot(seg[:, 0], seg[:, 1])
        else:
            dist = np.abs(seg[:, 0] * d[1] - seg[:, 1] * d[0]) / n
        k = int(np.argmax(dist))
        if dist[k] > tol:
            m = i + 1 + k
            keep[m] = True
            stack += [(i, m), (m, j)]
    return xy[keep]


def load_hsr(path: str | None) -> pd.DataFrame | None:
    """The committed THSR centreline, or None when the file is absent."""
    if not path or not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    need = {"chainage_km", "x_twd97", "y_twd97"}
    if not need <= set(df.columns):
        raise ValueError(f"{path}: needs columns {sorted(need)}")
    return df


def load_rivers(path: str | None) -> pd.DataFrame | None:
    if not path or not os.path.exists(path):
        return None
    return pd.read_csv(path)


def _overpass(query: str) -> dict:
    import time

    import requests

    for attempt in range(4):                  # the public server sheds load with 429/504
        r = requests.post(OVERPASS, data={"data": query}, timeout=180,
                          headers={"User-Agent": "HydroPhysicsAI-twin/1.0"})
        if r.status_code in (429, 502, 503, 504) and attempt < 3:
            time.sleep(10 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("Overpass unavailable")


def main(argv=None) -> None:
    from pyproj import Transformer

    ap = argparse.ArgumentParser(description="trace THSR and rivers over the fan from OSM")
    ap.add_argument("--out-dir", default=GEODATA)
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    to_xy = Transformer.from_crs(4326, 3826, always_xy=True)
    s, w, n, e = BBOX

    def xy_of(geom):
        lon = np.array([g["lon"] for g in geom])
        lat = np.array([g["lat"] for g in geom])
        x, y = to_xy.transform(lon, lat)
        return np.column_stack([x, y])

    rail = _overpass(f'[out:json][timeout:90];way["railway"="rail"]["highspeed"="yes"]'
                     f'({s},{w},{n},{e});out geom;')
    parts = [xy_of(el["geometry"]) for el in rail["elements"]
             if el.get("tags", {}).get("tunnel") != "yes"]
    cl = trace_centreline(parts)
    pd.DataFrame({"chainage_km": np.round(cl[:, 0] / 1000.0, 3),
                  "x_twd97": np.round(cl[:, 1], 1), "y_twd97": np.round(cl[:, 2], 1)}
                 ).to_csv(os.path.join(args.out_dir, "thsr_choushui.csv"), index=False)
    st = _overpass(f'[out:json][timeout:60];node["railway"="station"]'
                   f'["operator"~"高速鐵路"]({s},{w},{n},{e});out body;')
    rows = []
    for el in st["elements"]:
        x, y = to_xy.transform(el["lon"], el["lat"])
        rows.append({"name_zh": el["tags"].get("name", ""),
                     "name_en": el["tags"].get("name:en", ""),
                     "x_twd97": round(x, 1), "y_twd97": round(y, 1)})
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "thsr_stations.csv"), index=False)

    names = "|".join(RIVERS)
    riv = _overpass(f'[out:json][timeout:90];way["waterway"="river"]["name"~"^({names})$"]'
                    f'({s},{w},{n},{e});out geom tags;')
    rows = []
    for part, el in enumerate(riv["elements"]):
        zh = el["tags"].get("name", "")
        if zh not in RIVERS:
            continue
        for x, y in simplify(xy_of(el["geometry"]), 200.0):
            rows.append({"river": RIVERS[zh], "name_zh": zh, "part": part,
                         "x_twd97": round(x, 1), "y_twd97": round(y, 1)})
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "rivers_choushui.csv"), index=False)
    print(f"THSR {cl[-1, 0] / 1000:.1f} km, {len(cl)} points; stations {len(st['elements'])}; "
          f"river vertices {len(rows)}")


if __name__ == "__main__":
    main()
