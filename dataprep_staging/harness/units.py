"""Pseudo-region units, temporal blocks and site-grouped folds (plan WP A1, 4.3, 4.5).

Two kinds of units, never mixed:

* **Design-time units** were already used by the 2026-10-12 pilots to make design choices
  (Yunlin easting terciles, north and south halves, the 2020-22 hold-out, the Bayesian-fusion
  pilot's 12 overlapping regions, the pilots' site folds, Changhua zones Z1-Z3). Results on
  them are design-time evidence, never validation of those choices (plan 4.1, 4.5).
* **Fresh units** are declared and hashed by WP A1 before any E1-E11 or P4.3 run
  (``results/dataprep/validation_units/v<N>/`` plus the pre-registration
  ``preregistration/A1-fresh-units-v<N>.json``): easting terciles with cut points rotated by
  half a tercile width, the two-tercile region, a 5-region disjoint k-means partition with
  seed 2027, the 2016-01..2018-12 temporal block, and fresh spatial-block folds.

The Changhua census obtained on 2026-10-12 superseded the P4 inference experiments (E1-E11,
P4.3); the fresh units now serve every downstream test that needs untouched validation units
(H2, H3, P3's out-of-sample test, P4m, S1, S5, I1).

Shaped stress regions: the river-adjacent north half (design-time) and a 10 km river band on
the Yunlin side that mirrors Changhua's Z2 (Changhua cells within 10 km of the county border,
outside the proximal zone x >= 205 km; this rule reproduces Z2's 300 cells and 24 wells).

A well belongs to a region when its model cell does (the pilots' ``R[obs_idx]``).
"""
from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

KMEANS_SEED = 2027
FOLD_SEED = 2027
PROXIMAL_X_KM = 205.0


def centroids(mask, x0: float, y0: float, dx: float) -> np.ndarray:
    """(A, 2) active-cell centres, row-major over the mask (``FanGrid.centroids``)."""
    rows, cols = np.nonzero(np.asarray(mask, bool))
    return np.column_stack([float(x0) + (cols + 0.5) * float(dx),
                            float(y0) + (rows + 0.5) * float(dx)])


def jaccard(a, b) -> float:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


# ----------------------------------------------------------------------------- design-time
def easting_terciles(x_km, sel) -> dict:
    """Design-time Yunlin easting terciles (cut points at the 1/3 and 2/3 quantiles of the
    selected cells' eastings; pilots ``d_pseudo_changhua.py``, ``p2_pseudo_region.py``)."""
    x = np.asarray(x_km, float)
    s = np.asarray(sel, bool)
    q1, q2 = np.quantile(x[s], [1 / 3, 2 / 3])
    return {"west": s & (x < q1), "mid": s & (x >= q1) & (x < q2), "east": s & (x >= q2),
            "_cuts_km": [float(q1), float(q2)]}


def halves(y_km, sel) -> dict:
    """Design-time north and south halves (northing median of the selected cells)."""
    y = np.asarray(y_km, float)
    s = np.asarray(sel, bool)
    md = np.median(y[s])
    return {"north": s & (y >= md), "south": s & (y < md), "_cut_km": float(md)}


def fusion_pilot_regions(x_km, y_km, sel) -> dict:
    """The Bayesian-fusion pilot's overlapping regions (``p1b_pseudo.py``): east and north
    terciles, east and north halves, k-means(3) partitions with seeds 1-3 on centroids in km,
    near-duplicates (Jaccard > 0.9) dropped in that order. Design-time units."""
    from sklearn.cluster import KMeans
    x, y, s = np.asarray(x_km, float), np.asarray(y_km, float), np.asarray(sel, bool)
    reg = {}
    for nm, v in (("east", x), ("north", y)):
        q1, q2 = np.quantile(v[s], [1 / 3, 2 / 3])
        reg[f"{nm}_t1"] = s & (v < q1)
        reg[f"{nm}_t2"] = s & (v >= q1) & (v < q2)
        reg[f"{nm}_t3"] = s & (v >= q2)
    for nm, v in (("east", x), ("north", y)):
        md = np.median(v[s])
        reg[f"{nm}_h1"] = s & (v < md)
        reg[f"{nm}_h2"] = s & (v >= md)
    for seed in (1, 2, 3):
        km = KMeans(3, n_init=10, random_state=seed).fit(np.c_[x[s], y[s]])
        for k in range(3):
            m = np.zeros(s.size, bool)
            m[np.flatnonzero(s)[km.labels_ == k]] = True
            reg[f"km{seed}_{k}"] = m
    keep = {}
    for nm, m in reg.items():
        if all(jaccard(m, m2) < 0.9 for m2 in keep.values()):
            keep[nm] = m
    return keep


def pilot_site_folds(well_xy, wsel, n: int = 5, seed: int = 0) -> list[list[int]]:
    """The pilots' site folds (design-time): wells grouped by coordinates rounded to 100 m,
    sites shuffled with ``default_rng(seed)`` and dealt round-robin into ``n`` folds."""
    wsel = np.asarray(wsel, bool)
    xy = np.round(np.asarray(well_xy, float)[wsel] / 100.0)
    sites: dict = {}
    for i, s in zip(np.flatnonzero(wsel), map(tuple, xy), strict=True):
        sites.setdefault(s, []).append(int(i))
    keys = list(sites)
    np.random.default_rng(seed).shuffle(keys)
    return [sum((sites[k] for k in keys[f::n]), []) for f in range(n)]


# ----------------------------------------------------------------------------- fresh
def rotated_terciles(x_km, sel) -> dict:
    """Fresh units: easting terciles with the cut points rotated by half a tercile width.

    The design-time cuts sit at the 1/3 and 2/3 quantiles of the selected cells' eastings;
    rotating them by half a tercile (1/6 in quantile) gives cuts at the 1/6, 1/2 and 5/6
    quantiles. Unit 1 = [q1/6, q1/2), unit 2 = [q1/2, q5/6), unit 3 = the wrap-around
    remainder (x >= q5/6 or x < q1/6: the far west and the far east strips, non-contiguous).
    Each fresh unit overlaps half of two design-time terciles."""
    x = np.asarray(x_km, float)
    s = np.asarray(sel, bool)
    a, b, c = np.quantile(x[s], [1 / 6, 1 / 2, 5 / 6])
    return {"rot1": s & (x >= a) & (x < b), "rot2": s & (x >= b) & (x < c),
            "rot3": s & ((x >= c) | (x < a)), "_cuts_km": [float(a), float(b), float(c)]}


def kmeans_partition(cen_km, sel, k: int = 5, seed: int = KMEANS_SEED) -> dict:
    """Fresh units: disjoint k-means partition of the selected cells (centroids in km;
    ``KMeans(k, n_init=10, random_state=seed)``). Regions are numbered by the easting of
    their centroid (west to east) so that the labels do not depend on k-means' internal
    order. Store the membership: a later scikit-learn may give a different partition."""
    from sklearn.cluster import KMeans
    c = np.asarray(cen_km, float)
    s = np.asarray(sel, bool)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(c[s])
    lab = km.labels_
    order = np.argsort(km.cluster_centers_[:, 0])
    rank = np.empty(k, int)
    rank[order] = np.arange(k)
    out = {}
    for j in range(k):
        m = np.zeros(s.size, bool)
        m[np.flatnonzero(s)[rank[lab] == j]] = True
        out[f"km{seed}_{j}"] = m
    return out


def distance_to_line_km(cell_xy, line) -> np.ndarray:
    """Distance (km) from each cell centre to a shapely (Multi)LineString in the same CRS."""
    from shapely.geometry import Point
    xy = np.asarray(cell_xy, float)
    return np.array([line.distance(Point(x, y)) for x, y in xy]) / 1000.0


def river_band(dist_km, sel, x_km, km: float = 10.0,
               proximal_x_km: float | None = PROXIMAL_X_KM):
    """Cells of ``sel`` within ``km`` of the county border, west of ``proximal_x_km``
    (the rule that reproduces Changhua's Z2: 300 cells, 24 wells)."""
    m = np.asarray(sel, bool) & (np.asarray(dist_km, float) <= km)
    if proximal_x_km is not None:
        m &= np.asarray(x_km, float) < proximal_x_km
    return m


def changhua_zones(dist_km, ch_sel, x_km, km: float = 10.0,
                   proximal_x_km: float = PROXIMAL_X_KM) -> dict:
    """Design-time Changhua zones of P4: Z1 proximal (x >= 205 km), Z2 within 10 km of the
    county border (outside Z1), Z3 the rest."""
    ch = np.asarray(ch_sel, bool)
    x = np.asarray(x_km, float)
    z1 = ch & (x >= proximal_x_km)
    z2 = river_band(dist_km, ch, x, km, proximal_x_km)
    return {"Z1": z1, "Z2": z2, "Z3": ch & ~z1 & ~z2}


# ----------------------------------------------------------------------------- folds
def well_sites(well_xy, radius_m: float = 100.0) -> np.ndarray:
    """Site id per well: single-linkage clusters of wells closer than ``radius_m`` (nested
    wells at one site stay together). Ids are numbered by first appearance."""
    xy = np.asarray(well_xy, float)
    if len(xy) == 1:
        return np.zeros(1, int)
    lab = fcluster(linkage(pdist(xy), method="single"), t=radius_m, criterion="distance")
    _, first = np.unique(lab, return_index=True)
    remap = {lab[i]: j for j, i in enumerate(sorted(first))}
    return np.array([remap[v] for v in lab])


def spatial_clusters(xy, diameter_km: float = 10.0) -> np.ndarray:
    """Complete-linkage clusters of points whose diameter is at most ``diameter_km``
    (deterministic; ids by first appearance)."""
    p = np.asarray(xy, float)
    if len(p) == 1:
        return np.zeros(1, int)
    lab = fcluster(linkage(pdist(p), method="complete"), t=diameter_km * 1000.0,
                   criterion="distance")
    _, first = np.unique(lab, return_index=True)
    remap = {lab[i]: j for j, i in enumerate(sorted(first))}
    return np.array([remap[v] for v in lab])


def spatial_block_folds(xy, ids=None, *, n_folds: int | None = 5, diameter_km: float = 10.0,
                        buffer_km: float = 5.0, seed: int = FOLD_SEED,
                        site_radius_m: float = 100.0) -> dict:
    """Site-grouped spatial-block folds with a buffer (plan 4.3).

    Points (wells or benchmarks) are grouped into sites (``site_radius_m``, nested wells
    together), sites into complete-linkage clusters of diameter <= ``diameter_km``. With an
    integer ``n_folds`` the clusters are shuffled with ``default_rng(seed)`` and each is dealt
    to the fold with the fewest points so far (ties: lowest fold); with ``n_folds=None`` every
    cluster is its own fold (leave one cluster out; used where a wide buffer would otherwise
    empty the training set). For test fold f the training set is every point outside f
    farther than ``buffer_km`` from all of f's points (heads: 10 km clusters, 5 km buffer;
    leveling: buffer >= 10 km)."""
    p = np.asarray(xy, float)
    n = len(p)
    ids = list(range(n)) if ids is None else list(ids)
    site = well_sites(p, site_radius_m)
    # cluster the site centres, then map back to points
    site_ids = np.unique(site)
    site_xy = np.array([p[site == s].mean(0) for s in site_ids])
    clus_of_site = spatial_clusters(site_xy, diameter_km)
    clus = clus_of_site[np.searchsorted(site_ids, site)]
    if n_folds is None:
        fold = clus.copy()
        n_folds = int(clus.max()) + 1
        assign = "leave_one_cluster_out"
    else:
        order = np.random.default_rng(seed).permutation(np.unique(clus))
        fold_of_clus, count = {}, np.zeros(n_folds, int)
        for c in order:
            f = int(np.argmin(count))
            fold_of_clus[int(c)] = f
            count[f] += int((clus == c).sum())
        fold = np.array([fold_of_clus[int(c)] for c in clus])
        assign = "balanced"
    d = np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(-1)) / 1000.0
    folds = []
    for f in range(n_folds):
        te = np.flatnonzero(fold == f)
        if te.size == 0:
            folds.append({"fold": f, "test": [], "train": [], "buffered_out": []})
            continue
        near = (d[:, te] <= buffer_km).any(1)
        tr = np.flatnonzero((fold != f) & ~near)
        out = np.flatnonzero((fold != f) & near)
        folds.append({"fold": f, "test": [ids[i] for i in te], "train": [ids[i] for i in tr],
                      "buffered_out": [ids[i] for i in out]})
    return {"fold_of": {ids[i]: int(fold[i]) for i in range(n)},
            "site_of": {ids[i]: int(site[i]) for i in range(n)},
            "cluster_of": {ids[i]: int(clus[i]) for i in range(n)},
            "folds": folds,
            "params": {"n_folds": n_folds, "assign": assign, "diameter_km": diameter_km,
                       "buffer_km": buffer_km,
                       "seed": seed if assign == "balanced" else None,
                       "site_radius_m": site_radius_m}}


def temporal_block(start: str, end: str) -> dict:
    """A temporal hold-out block (months inclusive)."""
    import pandas as pd
    months = pd.period_range(start, end, freq="M")
    return {"start": start, "end": end, "n_months": len(months),
            "months": [str(m) for m in months]}
