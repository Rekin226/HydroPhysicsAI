"""Minimal remote GeoTIFF header reader (first IFD) over HTTP range requests.

Used by ``estimate_sizes.py`` to price a clip before fetching: it reads the image size, the
tile (or strip) layout with every tile's byte offset and count, the compression and the
GeoTIFF pixel scale and tie point, without downloading pixel data. Nothing here decodes or
executes the downloaded bytes beyond parsing TIFF tag structures.
"""
from __future__ import annotations

import struct
import time

import requests

_T = {1: ("B", 1), 2: ("s", 1), 3: ("H", 2), 4: ("I", 4), 5: ("II", 8), 6: ("b", 1),
      7: ("B", 1), 8: ("h", 2), 9: ("i", 4), 10: ("ii", 8), 11: ("f", 4), 12: ("d", 8),
      16: ("Q", 8), 17: ("q", 8)}


class RangeReader:
    def __init__(self, session: requests.Session, url: str):
        self.s, self.url, self.nbytes, self.nreq, self.size = session, url, 0, 0, None

    def get(self, a: int, b: int) -> bytes:
        last = None
        for k in range(4):
            try:
                r = self.s.get(self.url, headers={"Range": f"bytes={a}-{b}"}, timeout=120)
                self.nreq += 1
                if r.status_code == 206:
                    self.nbytes += len(r.content)
                    cr = r.headers.get("Content-Range", "")
                    if "/" in cr and cr.rsplit("/", 1)[1].isdigit():
                        self.size = int(cr.rsplit("/", 1)[1])
                    return r.content
                if r.status_code == 200:      # server ignored the range
                    self.nbytes += len(r.content)
                    self.size = len(r.content)
                    return r.content[a:b + 1]
                last = f"HTTP {r.status_code}"
            except requests.RequestException as e:
                last = repr(e)
            time.sleep(2 * (k + 1))
        raise RuntimeError(f"range {a}-{b} failed for {self.url}: {last}")


def read_header(rr: RangeReader) -> dict:
    hdr = rr.get(0, 16383)
    bo = "<" if hdr[:2] == b"II" else ">"
    magic = struct.unpack(bo + "H", hdr[2:4])[0]
    if magic not in (42, 43):
        raise ValueError(f"not a TIFF ({magic}) at {rr.url}")
    big = magic == 43
    off = struct.unpack(bo + "Q", hdr[8:16])[0] if big else struct.unpack(bo + "I", hdr[4:8])[0]
    base, blk = (0, hdr) if off + 4096 < len(hdr) else (off, rr.get(off, off + 65535))

    def raw_at(p: int, n: int) -> bytes:
        if base <= p and p + n <= base + len(blk):
            return blk[p - base:p - base + n]
        return rr.get(p, p + n - 1)

    n = struct.unpack(bo + ("Q" if big else "H"), raw_at(off, 8 if big else 2))[0]
    es, hs, inl = (20, 8, 8) if big else (12, 2, 4)
    ents = raw_at(off + hs, es * n)
    tags = {}
    for j in range(n):
        e = ents[es * j: es * (j + 1)]
        if big:
            tag, typ, cnt = struct.unpack(bo + "HHQ", e[:12])
            val = e[12:20]
        else:
            tag, typ, cnt = struct.unpack(bo + "HHI", e[:8])
            val = e[8:12]
        if typ not in _T:
            continue
        fmt, sz = _T[typ]
        tot = sz * cnt
        if tot <= inl:
            raw = val[:tot]
        else:
            p = struct.unpack(bo + ("Q" if big else "I"), val)[0]
            raw = raw_at(p, tot)
        tags[tag] = raw.decode(errors="replace") if typ == 2 else \
            struct.unpack(bo + fmt * cnt, raw)
    W, H = tags[256][0], tags[257][0]
    if 322 in tags:
        tw, th, offs, cnts = tags[322][0], tags[323][0], tags[324], tags[325]
    else:
        tw, th, offs, cnts = W, tags[278][0], tags[273], tags[279]
    sx, sy = tags[33550][0], tags[33550][1]
    x0, y0 = tags[33922][3], tags[33922][4]
    return {"W": W, "H": H, "tw": tw, "th": th, "offs": offs, "cnts": cnts,
            "bits": tags[258][0], "comp": tags.get(259, (1,))[0],
            "pred": tags.get(317, (1,))[0], "sf": tags.get(339, (1,))[0],
            "sx": sx, "sy": sy, "x0": x0, "y0": y0, "tiled": 322 in tags,
            "nodata": tags.get(42113)}


def box_window(h: dict, box: dict) -> tuple[int, int, int, int]:
    """(col0, row0, ncols, nrows) of the box on the native grid, clipped to the image."""
    c0 = max(0, int((box["lon0"] - h["x0"]) / h["sx"]))
    c1 = min(h["W"], int(-(-(box["lon1"] - h["x0"]) // h["sx"])))
    r0 = max(0, int((h["y0"] - box["lat1"]) / h["sy"]))
    r1 = min(h["H"], int(-(-(h["y0"] - box["lat0"]) // h["sy"])))
    return c0, r0, max(0, c1 - c0), max(0, r1 - r0)


def window_tile_bytes(h: dict, win: tuple[int, int, int, int]) -> tuple[int, int]:
    """(bytes of the tiles/strips the window touches, number of tiles)."""
    c0, r0, nc, nr = win
    if nc == 0 or nr == 0:
        return 0, 0
    ntx = (h["W"] + h["tw"] - 1) // h["tw"]
    ks = [r * ntx + c for r in range(r0 // h["th"], (r0 + nr - 1) // h["th"] + 1)
          for c in range(c0 // h["tw"], (c0 + nc - 1) // h["tw"] + 1)]
    return sum(h["cnts"][k] for k in ks), len(ks)
