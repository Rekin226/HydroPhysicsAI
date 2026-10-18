"""WP S4a: interferogram-network and epoch inventory from the saved LiCSAR listings.

Library functions (pure; no network) used by ``fetch_subsets.py`` (which pairs to fetch),
``validate.py`` and ``build_inventory.py``:

* :func:`parse_pairs` / :func:`parse_epochs` read an Apache index saved by
  ``fetch_listings.py`` (hrefs ``yyyymmdd_yyyymmdd/`` and ``yyyymmdd/``);
* :func:`window_pairs` keeps pairs whose two epochs both fall in [start, end];
* :func:`network_stats` gives epochs, pairs, connected components (union-find), per-year
  epoch counts and largest epoch gaps, and the temporal-baseline histogram.
"""
from __future__ import annotations

import collections
import datetime as dt
import re
from pathlib import Path

_PAIR = re.compile(r'href="(\d{8})_(\d{8})/?"')
_EPOCH = re.compile(r'href="(\d{8})/?"')
_ROW_DATE = re.compile(r'href="(\d{8}_\d{8}|\d{8})/?".*?<td align="right">\s*'
                       r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2})')


def _d(s: str) -> dt.date:
    return dt.date(int(s[:4]), int(s[4:6]), int(s[6:8]))


def parse_pairs(html_path: str | Path) -> list[str]:
    html = Path(html_path).read_text(errors="replace")
    return sorted({f"{a}_{b}" for a, b in _PAIR.findall(html)})


def parse_epochs(html_path: str | Path) -> list[str]:
    html = Path(html_path).read_text(errors="replace")
    return sorted(set(_EPOCH.findall(html)))


def listing_mtimes(html_path: str | Path) -> dict[str, str]:
    """Directory modification time per entry (portal-side processing/upload time)."""
    html = Path(html_path).read_text(errors="replace")
    return {k: v for k, v in _ROW_DATE.findall(html)}


def window_pairs(pairs: list[str], start: str, end: str) -> list[str]:
    return [p for p in pairs if p[:8] >= start and p[9:] <= end]


def components(pairs: list[str]) -> list[list[str]]:
    nodes = sorted({p[:8] for p in pairs} | {p[9:] for p in pairs})
    par = {n: n for n in nodes}

    def find(x: str) -> str:
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    for p in pairs:
        par[find(p[:8])] = find(p[9:])
    comps: dict[str, list[str]] = {}
    for n in nodes:
        comps.setdefault(find(n), []).append(n)
    return sorted((sorted(v) for v in comps.values()), key=lambda v: (-len(v), v[0]))


def network_stats(pairs: list[str]) -> dict:
    if not pairs:
        return {"n_pairs": 0, "n_epochs": 0}
    eps = sorted({p[:8] for p in pairs} | {p[9:] for p in pairs})
    E = [_d(e) for e in eps]
    comps = components(pairs)
    per_year = {}
    for y in range(E[0].year, E[-1].year + 1):
        ey = [e for e in E if e.year == y]
        gaps = [(b - a).days for a, b in zip(ey[:-1], ey[1:], strict=True)]
        per_year[str(y)] = {"epochs": len(ey), "max_gap_days": max(gaps) if gaps else None}
    gaps_all = [((b - a).days, a.isoformat(), b.isoformat()) for a, b in zip(E[:-1], E[1:], strict=True)]
    big = sorted(gaps_all, reverse=True)[:5]
    bt = [(_d(p[9:]) - _d(p[:8])).days for p in pairs]
    hist = collections.Counter(bt)
    return {
        "n_pairs": len(pairs), "n_epochs": len(eps),
        "first_epoch": eps[0], "last_epoch": eps[-1],
        "n_components": len(comps),
        "components": [{"first": c[0], "last": c[-1], "n_epochs": len(c)} for c in comps],
        "epochs_per_year": per_year,
        "largest_epoch_gaps_days": [{"days": g, "from": a, "to": b} for g, a, b in big],
        "btemp_days_hist": {str(k): v for k, v in sorted(hist.items())},
        "btemp_days_median": sorted(bt)[len(bt) // 2],
        "epochs": eps,
    }


def epochs_without_pairs(epochs: list[str], pairs: list[str], start: str, end: str) -> list[str]:
    used = {p[:8] for p in pairs} | {p[9:] for p in pairs}
    return [e for e in epochs if start <= e <= end and e not in used]


_FILE_ROW = re.compile(r'href="([^"?/][^"]*)".*?<td align="right">\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2})'
                       r'\s*</td>\s*<td align="right">\s*([^<]*?)\s*</td>')


def parse_dir_files(html_path: str | Path) -> dict[str, dict]:
    """Files of one saved Apache index page: {name: {'mtime', 'size'}} (size as printed,
    e.g. '5.2M'; exact bytes come from the range responses at fetch time)."""
    html = Path(html_path).read_text(errors="replace")
    return {n: {"mtime": m, "size": s} for n, m, s in _FILE_ROW.findall(html)}


def pair_products(html_path: str | Path, pair: str, exts=("unw", "cc")) -> dict[str, bool]:
    """Which of ``<pair>.geo.<ext>.tif`` a saved pair-directory page lists."""
    files = parse_dir_files(html_path)
    return {e: f"{pair}.geo.{e}.tif" in files for e in exts}
