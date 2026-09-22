"""Fetch the real ground: Taiwanese orthophoto and a terrain model for the fan.

    python -m hydrophysics.twin.basemap --out results/twin/basemap.npz

The twin's block model stood on an inverse-distance interpolation of 344 well collar
elevations and wore no imagery at all, which is why it read as a diagram rather than as
the Choushui fan. This module pulls two public, unauthenticated sources and resamples both
onto the model's own TWD97 grid:

- **Orthophoto and topographic base**, from the National Land Surveying and Mapping
  Center's public tile service (``wmts.nlsc.gov.tw``, the service behind Taiwan e-Map).
  Layers used: ``PHOTO2`` (aerial imagery), ``EMAP`` (topographic map). The same service
  also serves ``LUIMAP`` (land use), ``TOWN``/``Village`` (administrative boundaries) and
  ``PHOTO2014``..``PHOTO2025`` (one mosaic per year), any of which ``--layers`` will take.
- **Terrain**, SRTM 1-arcsecond tiles from the AWS open terrain archive (no key). Raw
  big-endian int16 on a one-degree tile, so it needs no GeoTIFF reader; rasterio is not
  in this environment and the Copernicus COG is a tiled float-predictor TIFF.

Nothing here needs credentials, and both sources permit reuse with attribution; the
attribution strings travel in the output so the page can display them.
"""

from __future__ import annotations

import argparse
import io
import math
import os

import numpy as np
from pyproj import Transformer

NLSC = "https://wmts.nlsc.gov.tw/wmts/{layer}/default/GoogleMapsCompatible/{z}/{y}/{x}"
SRTM = "https://s3.amazonaws.com/elevation-tiles-prod/skadi/{lat3}/{name}.hgt.gz"
ATTRIB = {
    "imagery": "Orthophoto: National Land Surveying and Mapping Center, Taiwan (NLSC)",
    "basemap": "Topographic base: NLSC Taiwan e-Map",
    "terrain": "Terrain: SRTM 1 arc-second, NASA / USGS",
}


def _lonlat_to_tile(lon: float, lat: float, z: int) -> tuple[float, float]:
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat)))
         / math.pi) / 2.0 * n
    return x, y


def fetch_mosaic(lon0: float, lat0: float, lon1: float, lat1: float, layer: str,
                 z: int = 13, log=print):
    """Stitch the tile layer over a lon/lat box -> ``(image, (lon, lat) of its corners)``."""
    import requests
    from PIL import Image

    x0f, y0f = _lonlat_to_tile(lon0, lat1, z)
    x1f, y1f = _lonlat_to_tile(lon1, lat0, z)
    x0, y0, x1, y1 = int(x0f), int(y0f), int(x1f), int(y1f)
    nx, ny = x1 - x0 + 1, y1 - y0 + 1
    out = Image.new("RGB", (nx * 256, ny * 256))
    sess = requests.Session()
    got = 0
    for ty in range(y0, y1 + 1):
        for tx in range(x0, x1 + 1):
            url = NLSC.format(layer=layer, z=z, y=ty, x=tx)
            try:
                r = sess.get(url, timeout=30)
                if r.status_code == 200 and len(r.content) > 300:
                    out.paste(Image.open(io.BytesIO(r.content)).convert("RGB"),
                              ((tx - x0) * 256, (ty - y0) * 256))
                    got += 1
            except Exception:
                pass
        if (ty - y0 + 1) % 4 == 0:
            log(f"  {layer}: {got}/{nx*ny} tiles")
    n = 2 ** z

    def tile_to_lonlat(x, y):
        lon = x / n * 360.0 - 180.0
        lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
        return lon, lat

    west, north = tile_to_lonlat(x0, y0)
    east, south = tile_to_lonlat(x1 + 1, y1 + 1)
    log(f"  {layer}: {got}/{nx*ny} tiles, {out.size[0]}x{out.size[1]} px")
    return out, (west, south, east, north)


def resample_to_grid(img, box, grid, out_px: int = 1600, log=print):
    """Resample a Web Mercator mosaic onto the model grid's own extent (EPSG:3826)."""
    from PIL import Image

    west, south, east, north = box
    to_ll = Transformer.from_crs(3826, 4326, always_xy=True)
    W = out_px
    Hh = int(round(out_px * (grid.ny * grid.dx) / (grid.nx * grid.dx)))
    xs = grid.x0 + (np.arange(W) + 0.5) / W * grid.nx * grid.dx
    ys = grid.y0 + (np.arange(Hh) + 0.5) / Hh * grid.ny * grid.dx
    XX, YY = np.meshgrid(xs, ys)
    lon, lat = to_ll.transform(XX.ravel(), YY.ravel())
    # the mosaic is linear in Mercator y, not in latitude
    def merc_y(a):
        return np.log(np.tan(np.pi / 4 + np.radians(a) / 2))
    px = (lon - west) / (east - west) * img.size[0]
    py = (merc_y(north) - merc_y(lat)) / (merc_y(north) - merc_y(south)) * img.size[1]
    src = np.asarray(img)
    px = np.clip(px.astype("int64"), 0, img.size[0] - 1)
    py = np.clip(py.astype("int64"), 0, img.size[1] - 1)
    out = src[py, px].reshape(Hh, W, 3)
    log(f"  resampled to {W}x{Hh} on the model grid")
    return Image.fromarray(out[::-1])           # north up


# --------------------------------------------------------------------------------------
# Terrain: SRTM 1-arcsecond tiles from the AWS open terrain archive
# --------------------------------------------------------------------------------------
def fetch_dem(grid, log=print) -> np.ndarray:
    """SRTM 1-arcsecond elevation, bilinearly sampled at each active cell -> (A,) metres.

    ``.hgt.gz`` is 3601x3601 big-endian int16 on a one-degree tile, which needs no GeoTIFF
    reader at all. Copernicus 30 m would be marginally better but ships as a tiled,
    deflate-compressed, float-predictor COG, and rasterio is not in this environment.
    """
    import gzip

    import requests

    to_ll = Transformer.from_crs(3826, 4326, always_xy=True)
    cent = grid.centroids()
    lon, lat = to_ll.transform(cent[:, 0], cent[:, 1])
    out = np.full(len(lon), np.nan)
    need = sorted({(int(math.floor(b)), int(math.floor(a)))
                   for a, b in zip(lon, lat, strict=True)})
    for la, lo in need:
        name = f"N{la:02d}E{lo:03d}"
        url = SRTM.format(lat3=name[:3], name=name)
        try:
            raw = gzip.decompress(requests.get(url, timeout=300).content)
        except Exception as e:
            log(f"  {name}: unavailable ({type(e).__name__})")
            continue
        n = int(round((len(raw) / 2) ** 0.5))
        tile = np.frombuffer(raw, dtype=">i2").reshape(n, n).astype("float32")
        tile[tile < -1000] = np.nan
        sel = (np.floor(lat).astype(int) == la) & (np.floor(lon).astype(int) == lo)
        if not sel.any():
            continue
        # row 0 is the tile's northern edge; bilinear between the four neighbours
        fy = (la + 1 - lat[sel]) * (n - 1)
        fx = (lon[sel] - lo) * (n - 1)
        y0 = np.clip(np.floor(fy).astype(int), 0, n - 2)
        x0 = np.clip(np.floor(fx).astype(int), 0, n - 2)
        wy, wx = fy - y0, fx - x0
        v = ((1 - wy) * ((1 - wx) * tile[y0, x0] + wx * tile[y0, x0 + 1])
             + wy * ((1 - wx) * tile[y0 + 1, x0] + wx * tile[y0 + 1, x0 + 1]))
        out[sel] = v
        log(f"  {name}: {int(sel.sum())} cells")
    bad = ~np.isfinite(out)
    if bad.any():
        out[bad] = np.nanmedian(out)
        log(f"  {int(bad.sum())} cells had no elevation; filled with the median")
    return out


def main(argv=None) -> None:
    from .calibrate_flow import DEFAULT_PATHS
    from .grid import build_grid

    ap = argparse.ArgumentParser(description="fetch imagery and terrain for the twin")
    ap.add_argument("--polygon", default=DEFAULT_PATHS["polygon"])
    ap.add_argument("--dx", type=float, default=1000.0)
    ap.add_argument("--zoom", type=int, default=13)
    ap.add_argument("--px", type=int, default=1600)
    ap.add_argument("--layers", default="PHOTO2,EMAP")
    ap.add_argument("--no-dem", action="store_true")
    ap.add_argument("--out", default="results/twin/basemap.npz")
    args = ap.parse_args(argv)

    grid = build_grid(args.polygon, dx=args.dx)
    to_ll = Transformer.from_crs(3826, 4326, always_xy=True)
    lon0, lat0 = to_ll.transform(grid.x0, grid.y0)
    lon1, lat1 = to_ll.transform(grid.x0 + grid.nx * grid.dx, grid.y0 + grid.ny * grid.dx)
    print(f"fan extent lon {lon0:.4f}..{lon1:.4f} lat {lat0:.4f}..{lat1:.4f}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    store = {"attrib": np.array([ATTRIB[k] for k in ("imagery", "basemap", "terrain")])}
    for layer in [s.strip() for s in args.layers.split(",") if s.strip()]:
        img, box = fetch_mosaic(lon0, lat0, lon1, lat1, layer, z=args.zoom)
        out = resample_to_grid(img, box, grid, out_px=args.px)
        buf = io.BytesIO()
        out.save(buf, format="JPEG", quality=82, optimize=True)
        store[layer] = np.frombuffer(buf.getvalue(), dtype="uint8")
        p = args.out.replace(".npz", f"_{layer}.jpg")
        out.save(p, quality=82, optimize=True)
        print(f"  wrote {p} ({os.path.getsize(p)/1e6:.2f} MB)", flush=True)
    if not args.no_dem:
        print("terrain:", flush=True)
        dem = fetch_dem(grid)
        store["dem"] = dem.astype("float32")
        print(f"  elevation over the fan: {np.nanmin(dem):.1f} to {np.nanmax(dem):.1f} m, "
              f"median {np.nanmedian(dem):.1f} m", flush=True)
    np.savez_compressed(args.out, **store)
    print(f"wrote {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
