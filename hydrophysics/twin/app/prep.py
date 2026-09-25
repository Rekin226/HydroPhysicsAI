"""Data preparation for the twin's decision page. Every number the page shows starts here.

The page (``template.html``) does arithmetic only: superposing the response basis for the
current policy, averaging over cells, sampling the rail. Everything that needs judgement,
a percentile, a regression or a data file is computed in this module, so it can be tested
on CPU with synthetic arrays (``tests/test_viewer_app.py``).

Conventions: subsidence in cm (positive = sinking), heads in m, a "year-end" is December
of each year, and a policy response is always *policy minus baseline*.
"""

from __future__ import annotations

import base64
import gzip

import numpy as np
import pandas as pd

CLASSES = ("irrigation", "aquaculture", "livestock", "domestic", "industry", "other")
BASIS_KEYS = {"irr": "irrigation", "aqua": "aquaculture", "live": "livestock",
              "dom": "domestic", "ind": "industry", "oth": "other"}
STARTS = (2026, 2030)
DELTA_SNAPS_CM = (0.25, 0.5, 1.0, 2.0, 5.0)
DELTA_SNAPS_M = (0.5, 1.0, 2.0, 5.0)
Z10 = 1.2815515655446004            # standard-normal quantile of 0.90
Z25 = 0.6744897501960817            # ... of 0.75
# metres of vertical extent per scene unit, per unit of exaggeration: 1 unit = 1 km
VERT_SCALE = 1.0 / 1000.0

# Yunlin's 20 townships, public administrative names (romanisation as used by the county)
TOWNSHIP_EN = {
    "斗六市": "Douliu", "斗南鎮": "Dounan", "虎尾鎮": "Huwei", "西螺鎮": "Xiluo",
    "土庫鎮": "Tuku", "北港鎮": "Beigang", "古坑鄉": "Gukeng", "大埤鄉": "Dapi",
    "莿桐鄉": "Citong", "林內鄉": "Linnei", "二崙鄉": "Erlun", "崙背鄉": "Lunbei",
    "麥寮鄉": "Mailiao", "東勢鄉": "Dongshi", "褒忠鄉": "Baozhong", "台西鄉": "Taixi",
    "臺西鄉": "Taixi", "元長鄉": "Yuanchang", "四湖鄉": "Sihu", "口湖鄉": "Kouhu",
    "水林鄉": "Shuilin",
}
POOLED_ZH, POOLED_EN = "彰化縣 (未分鄉鎮)", "Changhua (not yet split by township)"

# Colour ramps, as control points in sRGB 0-255. The page interpolates linearly between
# them; the test checks the diverging endpoints stay distinct under deuteranopia.
PALETTES = {
    # blue = benefit (less sinking / higher head), orange = worse; ColorBrewer RdBu blue
    # arm with the PuOr orange arm, both colour-blind safe
    "diverging": [[33, 102, 172], [67, 147, 195], [146, 197, 222], [209, 229, 240],
                  [247, 247, 247], [254, 224, 182], [253, 184, 99], [224, 130, 20],
                  [179, 88, 6]],
    "cividis": [[0, 34, 78], [18, 54, 110], [59, 73, 108], [87, 92, 109], [112, 113, 115],
                [138, 135, 121], [166, 157, 117], [195, 181, 105], [225, 207, 85],
                [254, 232, 56]],
    "viridis": [[68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142], [38, 130, 142],
                [31, 158, 137], [53, 183, 121], [109, 205, 89], [180, 222, 44],
                [253, 231, 37]],
}


# --------------------------------------------------------------------------------------
# packing
# --------------------------------------------------------------------------------------
def quantise(a: np.ndarray, step: float, dtype: str = "int16") -> np.ndarray:
    """Round ``a`` to counts of ``step``; NaN becomes the dtype's minimum (a sentinel)."""
    info = np.iinfo(dtype)
    a = np.asarray(a, dtype="float64")
    q = np.round(np.where(np.isfinite(a), a, 0.0) / step)
    q = np.clip(q, info.min + 1, info.max)
    q = np.where(np.isfinite(a), q, info.min)
    return q.astype(dtype)


def pack(a: np.ndarray, step: float | None = None, dtype: str = "int16") -> dict:
    """Quantise, byte-shuffle and gzip an array for the page -> ``{t, s, sh, z}``.

    Byte shuffling (all low bytes, then all high bytes) roughly doubles what gzip gets
    out of smooth int16 fields. ``s`` is the value of one count (None for raw integers);
    the page decodes with ``DecompressionStream('gzip')``.
    """
    a = np.asarray(a)
    if step is not None:
        # never clip: widen the step until the largest value fits (the page reads ``s``)
        info = np.iinfo(dtype)
        big = float(np.nanmax(np.abs(a))) if a.size and np.isfinite(a).any() else 0.0
        limit = min(abs(info.min + 1), info.max) * 0.98
        if big / step > limit:
            step = big / limit
        q = quantise(a, step, dtype)
    else:
        q = np.ascontiguousarray(a.astype(dtype))
    raw = q.tobytes()
    width = q.dtype.itemsize
    if width > 1:
        raw = np.frombuffer(raw, dtype="uint8").reshape(-1, width).T.copy().tobytes()
    z = gzip.compress(raw, compresslevel=9, mtime=0)
    return {"t": {"int16": "i16", "uint8": "u8", "int32": "i32", "uint16": "u16"}[dtype],
            "s": step, "sh": list(q.shape), "z": base64.b64encode(z).decode("ascii")}


def unpack(p: dict) -> np.ndarray:
    """Inverse of :func:`pack` (used by the tests to read the page's payload back)."""
    dt = {"i16": "int16", "u8": "uint8", "i32": "int32", "u16": "uint16"}[p["t"]]
    raw = gzip.decompress(base64.b64decode(p["z"]))
    width = np.dtype(dt).itemsize
    if width > 1:
        raw = np.frombuffer(raw, dtype="uint8").reshape(width, -1).T.copy().tobytes()
    q = np.frombuffer(raw, dtype=dt).reshape(p["sh"])
    if p["s"] is None:
        return q
    out = q.astype("float64") * p["s"]
    if dt != "uint8":
        out[q == np.iinfo(dt).min] = np.nan
    return out


# --------------------------------------------------------------------------------------
# time axis and response basis
# --------------------------------------------------------------------------------------
def year_ends(dates) -> tuple[np.ndarray, list[int]]:
    """Indices of the December months and their years."""
    d = pd.to_datetime(pd.Index([str(x)[:10] for x in dates]))
    idx = np.nonzero(d.month == 12)[0]
    return idx, [int(y) for y in d.year[idx]]


def basis_rows(names: list[str]) -> dict[tuple[str, int], int]:
    """``{(class, start year): scenario index}`` from names like ``irr0_2026``."""
    out = {}
    for i, n in enumerate(names):
        if n == "baseline" or n.startswith("check") or "_" not in n:
            continue
        key, yr = n.rsplit("_", 1)
        cls = BASIS_KEYS.get(key.rstrip("0123456789"))
        if cls is not None and yr.isdigit():
            out[(cls, int(yr))] = i
    return out


def basis_deltas(subs_cm: np.ndarray, heads_m: np.ndarray, names: list[str],
                 ye: np.ndarray, starts=STARTS, classes=CLASSES) -> dict:
    """Per-lever responses at year-ends, ordered ``row = start_i * K + class_k``.

    Returns ``dsub`` (R, A, Y) cm, ``dheadL2`` (R, A, Y) m (layer index 1), ``dhead_end``
    (R, L, A) m at the last year-end, the row labels and which rows were present. A missing
    (class, start) row stays zero and is listed in ``missing``.
    """
    rows = basis_rows(names)
    b = names.index("baseline")
    K, A, Y = len(classes), subs_cm.shape[1], len(ye)
    L = heads_m.shape[1]
    R = len(starts) * K
    dsub = np.zeros((R, A, Y))
    dh2 = np.zeros((R, A, Y))
    dh_end = np.zeros((R, L, A))
    labels, missing = [], []
    for si, s in enumerate(starts):
        for k, c in enumerate(classes):
            r = si * K + k
            labels.append(f"{c}_{s}")
            i = rows.get((c, s))
            if i is None:
                missing.append(f"{c}_{s}")
                continue
            dsub[r] = subs_cm[i][:, ye] - subs_cm[b][:, ye]
            dh2[r] = heads_m[i, 1][:, ye] - heads_m[b, 1][:, ye]
            dh_end[r] = heads_m[i][:, :, ye[-1]] - heads_m[b][:, :, ye[-1]]
    return {"dsub": dsub, "dheadL2": dh2, "dhead_end": dh_end, "labels": labels,
            "missing": missing}


def superpose(rows: np.ndarray, factors, start_i: int, K: int = len(CLASSES)) -> np.ndarray:
    """The page's policy formula, in numpy: ``sum_k (1 - f_k) * rows[start_i * K + k]``."""
    f = np.asarray(factors, dtype="float64")
    out = np.zeros(rows.shape[1:])
    for k in range(K):
        if f[k] != 1.0:
            out = out + (1.0 - f[k]) * rows[start_i * K + k]
    return out


def parse_check(name: str) -> tuple[list[float], int] | None:
    """``"check_irr50_2026"`` -> (factors per class with irrigation at 0.5, 2026)."""
    if not name.startswith("check_") or "_" not in name[6:]:
        return None
    body, yr = name[6:].rsplit("_", 1)
    key = body.rstrip("0123456789")
    pct = body[len(key):]
    cls = BASIS_KEYS.get(key)
    if cls is None or not pct or not yr.isdigit():
        return None
    f = [1.0] * len(CLASSES)
    f[CLASSES.index(cls)] = int(pct) / 100.0
    return f, int(yr)


def _err(pred: np.ndarray, act: np.ndarray, y0: int) -> dict:
    a = act[:, -1] - act[:, y0]
    p = pred[:, -1] - pred[:, y0]
    err = np.abs(p - a)
    return {"fan_cm": float(abs(p.mean() - a.mean())),
            "p95_cell_cm": float(np.percentile(err, 95)),
            "rel": float(abs(p.mean() - a.mean()) / max(abs(a.mean()), 1e-9))}


def basis_error(dsub: np.ndarray, subs_cm_basis: np.ndarray, names: list[str],
                ye: np.ndarray, y0: int, solved: list[dict] | None = None,
                starts=STARTS) -> dict:
    """How far superposition is from solved runs, at the last year-end, fan mean and p95 cell.

    Every ``check_<lever><pct>_<year>`` run in the basis file (the same parameter set as
    the basis, so the gap is the non-linearity alone) is compared with the superposed
    ``pct`` factor; its key is the run's name and it carries ``"kind": "nonlinear"``. Every
    ``solved`` forward-run scenario (``{name, factors, start, _dsub}``, an ensemble mean)
    is compared the same way under its own name, with ``"kind": "ensemble"``: that gap
    adds the difference between the basis' single parameter set and the ensemble. ``y0``
    is the year-end index the forward change is measured from. Values in cm; ``rel`` as a
    fraction of the response. ``dsub`` must be the *uncalibrated* basis.
    """
    out = {}
    K = len(CLASSES)
    if "baseline" in names:
        b = subs_cm_basis[names.index("baseline")][:, ye]
        for i, n in enumerate(names):
            pc = parse_check(n)
            if pc is None or pc[1] not in starts:
                continue
            pred = superpose(dsub, pc[0], list(starts).index(pc[1]), K)
            out[n] = {"kind": "nonlinear", "factors": pc[0], "start": pc[1],
                      **_err(pred, subs_cm_basis[i][:, ye] - b, y0)}
    for x in solved or []:
        if x["start"] not in starts:
            continue
        pred = superpose(dsub, x["factors"], list(starts).index(x["start"]), K)
        out[x["name"]] = {"kind": "ensemble", "factors": list(x["factors"]),
                          "start": x["start"], **_err(pred, x["_dsub"], y0)}
    return out


def single_lever(factors) -> int | None:
    """The class index when exactly one factor is below 1, else None."""
    cut = [k for k, v in enumerate(factors) if v < 1.0 - 1e-9]
    return cut[0] if len(cut) == 1 else None


def calibrate_basis(dsub: np.ndarray, dhead: np.ndarray, solved: list[dict], y0: int,
                    starts=STARTS, K: int = len(CLASSES)) -> dict:
    """Per-class factors that rescale the basis to the ensemble's solved runs.

    The basis is one parameter set; the solved scenarios of the forward run are ensemble
    means. For every solved run that cuts a single class, the factor is the ratio of the
    two fan-mean responses at the horizon (forward change since year-end ``y0`` for
    subsidence, the change at the horizon for layer-2 head). A class with no such run gets
    the mean factor of the classes that have one, and is marked ``assumed``. Applying the
    factors makes the page's fast estimate agree with the solved run at the solved policy,
    so moving a slider onto a solved scenario does not make the numbers jump.
    Returns ``{"subs": [K], "head": [K], "source": [K]}``; factors are 1 without solved runs.
    """
    fs, fh, src = [None] * K, [None] * K, ["none"] * K
    for x in solved:
        k = single_lever(x["factors"])
        if k is None or x["start"] not in starts or fs[k] is not None:
            continue
        si = list(starts).index(x["start"])
        w = 1.0 - x["factors"][k]
        ps = w * (dsub[si * K + k][:, -1] - dsub[si * K + k][:, y0]).mean()
        ph = w * dhead[si * K + k][:, -1].mean()
        as_ = (x["_dsub"][:, -1] - x["_dsub"][:, y0]).mean()
        ah = x["_dh2"][:, -1].mean()
        if abs(ps) > 1e-9 and abs(ph) > 1e-9:
            fs[k], fh[k], src[k] = float(as_ / ps), float(ah / ph), f"solved:{x['name']}"
    known = [k for k in range(K) if fs[k] is not None]
    ms = float(np.mean([fs[k] for k in known])) if known else 1.0
    mh = float(np.mean([fh[k] for k in known])) if known else 1.0
    for k in range(K):
        if fs[k] is None:
            fs[k], fh[k] = ms, mh
            src[k] = "assumed" if known else "none"
    return {"subs": fs, "head": fh, "source": src}


def apply_calibration(rows: np.ndarray, factors, starts=STARTS,
                      K: int = len(CLASSES)) -> np.ndarray:
    """Scale basis rows ``(R, ...)`` ordered ``start_i * K + class_k`` by a per-class factor."""
    out = np.array(rows, dtype="float64", copy=True)
    for si in range(len(starts)):
        for k in range(K):
            out[si * K + k] *= factors[k]
    return out


def snap(value: float, allowed) -> float:
    """The allowed value nearest ``value`` (ties go to the larger)."""
    allowed = sorted(allowed)
    return float(min(allowed, key=lambda a: (abs(a - value), -a)))


def snap_up(value: float, allowed) -> float:
    """The smallest allowed value at or above ``value`` (the largest if none is)."""
    allowed = sorted(allowed)
    return float(next((a for a in allowed if a >= value - 1e-12), allowed[-1]))


def delta_scale(rows: np.ndarray, y0: int, allowed, cut: float = 0.3,
                K: int = len(CLASSES)) -> dict:
    """Fixed symmetric limit for a difference map, chosen once and independent of policy.

    The reference is a ``cut`` (30 %) reduction of the most responsive single lever with
    the first start year: the 98th percentile over cells of its absolute forward change,
    rounded *up* to an ``allowed`` value, so the reference policy itself never saturates.
    Retiring a lever outright then saturates the scale near the hot spots, which the
    legend marks with ≥ ends and a count of the cells beyond it, and a small policy still
    looks small.
    """
    best, lever = 0.0, None
    for k in range(K):
        r = rows[k]
        v = float(np.percentile(np.abs(cut * (r[:, -1] - r[:, y0])), 98))
        if v > best:
            best, lever = v, k
    return {"limit": snap_up(best, allowed) if best > 0 else float(min(allowed)),
            "p98": best, "lever": None if lever is None else CLASSES[lever]}


# --------------------------------------------------------------------------------------
# ensemble members (paired fan scalars from <forward>.members.csv)
# --------------------------------------------------------------------------------------
def pct(x, q=(10, 50, 90)) -> list[float]:
    return [float(v) for v in np.percentile(np.asarray(x, dtype="float64"), q)]


def paired_deltas(members: pd.DataFrame) -> dict:
    """Per scenario, each member's fan-scalar change against the *same member's* baseline.

    Pairing by (member, ic) removes the between-member spread both runs share. Returns
    ``{scenario: {n, dsubs (M,), dhead (M,), subs_p (p10,p50,p90), head_p, agree}}`` where
    ``agree`` counts members whose change in forward subsidence is negative (a benefit).
    ``base`` carries the baseline's own members (forward and end-of-run fan means).

    A parameter set run from several initial fields (``ic``) is one opinion, not several:
    ``n_sets`` and ``agree_sets`` count parameter sets (a set agrees when the mean of its
    runs is a benefit), and ``ic_spread`` is the largest difference between two runs of
    the same set, which shows how little the initial field adds.
    """
    key = ["member", "ic"] if "ic" in members.columns else ["member"]
    b = members[members.scenario == "baseline"].set_index(key)
    out = {"base": {"n": int(len(b)),
                    "fwd": [float(v) for v in b["subs_forward_cm"]],
                    "end": [float(v) for v in b["subs_end_cm"]],
                    "fwd_p": pct(b["subs_forward_cm"], (10, 25, 50, 75, 90)),
                    "end_p": pct(b["subs_end_cm"], (10, 25, 50, 75, 90)),
                    "head_p": pct(b["head_L2_change_m"], (10, 25, 50, 75, 90))}}
    for s in members.scenario.unique():
        if s == "baseline":
            continue
        x = members[members.scenario == s].set_index(key)
        common = x.index.intersection(b.index)
        dss = x.loc[common, "subs_forward_cm"] - b.loc[common, "subs_forward_cm"]
        ds = dss.to_numpy()
        dh = (x.loc[common, "head_L2_change_m"]
              - b.loc[common, "head_L2_change_m"]).to_numpy()
        if "ic" in key and len(common):
            by_set = dss.groupby(level="member")
            set_mean = by_set.mean()
            ic_spread = float((by_set.max() - by_set.min()).max())
        else:
            set_mean, ic_spread = dss, 0.0
        out[str(s)] = {"n": int(len(common)), "n_sets": int(len(set_mean)),
                       "agree_sets": int((set_mean < 0).sum()), "ic_spread": ic_spread,
                       "dsubs": [float(v) for v in ds],
                       "dhead": [float(v) for v in dh],
                       "subs_p": pct(ds), "subs_q": pct(ds, (10, 25, 50, 75, 90)),
                       "subs_mean": float(ds.mean()) if len(ds) else float("nan"),
                       "head_p": pct(dh),
                       "head_mean": float(dh.mean()) if len(dh) else float("nan"),
                       "agree": int((ds < 0).sum())}
    return out


# --------------------------------------------------------------------------------------
# per-member yearly fields (<forward>.members.npz, twin.forward --save-members yearly)
# --------------------------------------------------------------------------------------
AGREE_MIN = 0.8             # a cell's change counts as agreed when this share of sets agree


def load_member_fields(path: str) -> dict:
    """Read the ``--save-members yearly`` sidecar -> subsidence in cm, head in m.

    ``subs`` has shape (S, M, A, Y), where M runs over member x ic x rheology. ``headL2``
    has shape (S, Mh, A, Y). ``sets`` gives each subsidence member's parameter-set key
    (flow member, plus rheology when there is more than one). The initial fields of a set
    are one opinion, as in :func:`paired_deltas`."""
    z = np.load(path, allow_pickle=False)
    rheo = [str(v) for v in z["rheology"]]
    mem = [str(v) for v in z["member"]]
    multi = len(set(rheo)) > 1
    return {"subs": z["subs_members_yr"].astype("float32") * float(z["subs_scale_m"]) * 100.0,
            "headL2": z["headL2_members_yr"].astype("float32") * float(z["head_scale_m"]),
            "years": [int(y) for y in z["years"]],
            "scenario_names": [str(v) for v in z["scenario_names"]],
            "sets": [f"{m}|{r}" if multi else m for m, r in zip(mem, rheo, strict=True)],
            "member": mem, "ic": [int(v) for v in z["ic"]],
            "head_sets": [str(v) for v in z["head_member"]]}


def _set_means(x: np.ndarray, sets: list[str]) -> np.ndarray:
    """Average ``x`` (M, ...) over the members of each parameter set -> (n_sets, ...)."""
    keys = list(dict.fromkeys(sets))
    idx = np.array([keys.index(s) for s in sets])
    return np.stack([x[idx == k].mean(axis=0) for k in range(len(keys))])


def member_delta(mf: dict, s: int, y_ref: int, what: str = "subs") -> np.ndarray:
    """Paired per-member change of scenario ``s`` against the same member's baseline,
    measured from year-end ``y_ref`` -> (M, A, Y). This is the same quantity as the map's
    forward difference."""
    x = mf[what]
    d = x[s] - x[0]
    return d - d[..., y_ref:y_ref + 1]


def cell_agreement(delta: np.ndarray, sets: list[str]) -> np.ndarray:
    """Per cell and year-end, the share of parameter sets whose change has the sign of the
    ensemble-mean change -> (A, Y) in [0, 1]. A set that shows exactly zero change does
    not agree. So where nothing changes (before a policy starts), the share is 0, and the
    map's magnitude rule hatches those cells anyway."""
    sm = _set_means(np.asarray(delta, dtype="float64"), sets)          # (n_sets, A, Y)
    sign = np.sign(sm.mean(axis=0))
    return ((np.sign(sm) == sign[None]) & (sign[None] != 0)).mean(axis=0)


def fan_member_band(delta: np.ndarray, cells: np.ndarray | None = None,
                    q=(10, 25, 50, 75, 90)) -> np.ndarray:
    """Per-year percentiles over members of the average change over ``cells`` (all cells
    when ``None``) -> (len(q), Y). This is the real yearly band that replaces the
    constructed one, where the spread was known only at two dates."""
    d = np.asarray(delta, dtype="float64")
    avg = d.mean(axis=1) if cells is None else d[:, np.asarray(cells)].mean(axis=1)
    return np.percentile(avg, q, axis=0)


def township_agreement(delta: np.ndarray, sets: list[str], town_idx: np.ndarray,
                       n_towns: int, y: int = -1) -> list[dict]:
    """Per township at year-end ``y``, the number of parameter sets whose township-average
    change is a benefit (negative subsidence change) -> ``[{agree, n}]``. Cells without a
    township (index < 0 or >= n_towns) are ignored."""
    sm = _set_means(np.asarray(delta, dtype="float64")[..., y], sets)   # (n_sets, A)
    out = []
    for t in range(n_towns):
        sel = np.asarray(town_idx) == t
        if not sel.any():
            out.append({"agree": 0, "n": 0})
            continue
        v = sm[:, sel].mean(axis=1)
        out.append({"agree": int((v < 0).sum()), "n": int(len(v))})
    return out


def member_fan(x: np.ndarray) -> np.ndarray:
    """Fan average of per-member yearly fields ``(S, M, A, Y)`` -> ``(S, M, Y)``: the
    series the page plays one run at a time."""
    return np.asarray(x, dtype="float64").mean(axis=2)


def area_by_set(subs: np.ndarray, sets: list[str], thr: float) -> np.ndarray:
    """Per parameter set, the cells (1 km² each) whose rate over the last 12 months of a
    yearly field ``(M, A, Y)`` exceeds ``thr`` cm/yr -> (n_sets,) int. The same count as
    :func:`rate_area`, on each set's mean field."""
    s = _set_means(np.asarray(subs, dtype="float64")[..., -2:], sets)   # (n_sets, A, 2)
    return ((s[..., 1] - s[..., 0]) > thr).sum(axis=1)


def hsr_max_by_set(fwd: np.ndarray, sets: list[str], idx: np.ndarray, w: np.ndarray,
                   step_km: float, mixed: np.ndarray | None = None) -> np.ndarray:
    """Per parameter set, the largest angular distortion along the rail of a forward-change
    field ``(M, A)`` in cm, leaving out the ``mixed`` points (the zone lines) as the page's
    rail tile does -> (n_sets,)."""
    sm = _set_means(np.asarray(fwd, dtype="float64"), sets)             # (n_sets, A)
    keep = np.ones(len(idx), dtype=bool) if mixed is None else ~np.asarray(mixed, dtype=bool)
    out = []
    for f in sm:
        ad = angular_distortion(sample(f, idx, w), step_km)
        out.append(float(ad[keep].max()) if keep.any() else float(ad.max()))
    return np.array(out)


# --------------------------------------------------------------------------------------
# fields
# --------------------------------------------------------------------------------------
def model_artefacts(fan_monthly_cm: np.ndarray, origin: int, window: int = 6) -> dict:
    """Steps in the fan-mean subsidence series that are the model's, not the ground's.

    ``startup_cm``: change over the first three months, when the column settles onto the
    flow model's initial heads. ``restart_cm``: change over the ``window`` months after the
    forecast origin (where the run restarts from the observed head field) in excess of the
    median monthly change over the rest of the projection. Both are in the baseline and in
    every policy alike, so they cancel in a policy's difference but not in totals.
    """
    s = np.asarray(fan_monthly_cm, dtype="float64")
    d = np.diff(s)
    after = d[origin + window + 1:]
    med = float(np.median(after)) if len(after) else 0.0
    rest = float(s[min(origin + window, len(s) - 1)] - s[origin]) - window * med
    return {"startup_cm": float(s[min(3, len(s) - 1)] - s[0]), "restart_cm": rest,
            "window": window, "trend_cm_yr": 12.0 * med}


STARTUP_WINDOW = 6          # months the column takes to settle onto the first heads
RESTART_WINDOW = 12         # months the column takes to settle after the restart


def artefact_steps(s_cm: np.ndarray, origin: int, sel: np.ndarray,
                   startup: int = STARTUP_WINDOW, restart: int = RESTART_WINDOW) -> dict:
    """Per-cell start-up and restart steps in a monthly subsidence field, cells in ``sel``.

    ``s_cm`` is ``(A, T)``. The start-up step is the change over the first ``startup``
    months in excess of the cell's mean monthly change between month 12 and the origin;
    the restart step is the change over the ``restart`` months after the origin in excess
    of the mean monthly change after that. Means, not medians, because the monthly
    increments carry the seasonal cycle of the forcing. Cells outside ``sel`` get zero:
    elsewhere the excess is the size of that seasonal noise, not a step (the builder
    reports both). Returns ``startup``, ``restart`` (A,) in cm and the per-cell ``excess``
    series inside each window, which :func:`remove_artefacts` subtracts.
    """
    s = np.asarray(s_cm, dtype="float64")
    A, T = s.shape
    sel = np.asarray(sel, dtype=bool)
    t0 = min(12, max(origin - 1, 1))
    tr_pre = (s[:, origin] - s[:, t0]) / max(origin - t0, 1)
    e1 = min(origin + restart, T - 1)
    tr_post = (s[:, T - 1] - s[:, e1]) / max(T - 1 - e1, 1)
    w1 = min(startup, T - 1)
    ex1 = s[:, :w1 + 1] - s[:, :1] - np.arange(w1 + 1) * tr_pre[:, None]
    ex2 = s[:, origin:e1 + 1] - s[:, origin:origin + 1] - (np.arange(e1 + 1 - origin)
                                                          * tr_post[:, None])
    ex1[~sel] = 0.0
    ex2[~sel] = 0.0
    return {"startup": ex1[:, -1].copy(), "restart": ex2[:, -1].copy(),
            "ex_startup": ex1, "ex_restart": ex2, "origin": int(origin),
            "windows": [w1, e1 - origin]}


def remove_artefacts(s_cm: np.ndarray, steps: dict) -> np.ndarray:
    """``s_cm`` (A, T) with the steps of :func:`artefact_steps` taken out.

    Inside a window the cell's excess so far is subtracted; after it, the whole step. So a
    corrected cell follows its own trend through the window and every later value is
    shifted by the step. The policy differences are untouched: the steps are identical in
    the baseline and every scenario (both come before the first policy start).
    """
    s = np.array(s_cm, dtype="float64", copy=True)
    T = s.shape[1]
    ex1, ex2, o = steps["ex_startup"], steps["ex_restart"], steps["origin"]
    w1 = ex1.shape[1] - 1
    s[:, :w1 + 1] -= ex1
    s[:, w1 + 1:] -= steps["startup"][:, None]
    e1 = o + ex2.shape[1] - 1
    s[:, o:e1 + 1] -= ex2
    if e1 + 1 < T:
        s[:, e1 + 1:] -= steps["restart"][:, None]
    return s


def restart_bridge(s_cm: np.ndarray, ye: np.ndarray, y_obs: int) -> np.ndarray:
    """Per-cell excess of the first year after the restart over the years either side, cm.

    The restart from the observed heads at the origin makes many cells jump or rebound in
    the following year (outside the proximal zone mostly a rebound of a few cm). At
    year-end resolution the step is the restart year's change minus the mean of the
    previous and the next year's change; :func:`apply_bridge` takes it out.
    """
    s = np.asarray(s_cm, dtype="float64")
    if y_obs < 1 or y_obs + 2 >= len(ye):
        return np.zeros(s.shape[0])
    v = s[:, ye[y_obs - 1:y_obs + 3]]
    inc = np.diff(v, axis=1)
    return inc[:, 1] - 0.5 * (inc[:, 0] + inc[:, 2])


def apply_bridge(s_cm: np.ndarray, corr: np.ndarray, origin: int, window: int = RESTART_WINDOW
                 ) -> np.ndarray:
    """Subtract ``corr`` (A,) after the origin, ramped in over ``window`` months."""
    s = np.array(s_cm, dtype="float64", copy=True)
    T = s.shape[-1]
    ramp = np.clip((np.arange(T) - origin) / float(window), 0.0, 1.0)
    return s - corr[:, None] * ramp[None, :]


def boundary_contrast(field: np.ndarray, x_km: np.ndarray, at_km: float, dx_km: float = 1.0
                      ) -> dict:
    """Median of ``field`` in the first column of cells east and west of a zone line."""
    east = (x_km > at_km) & (x_km <= at_km + dx_km)
    west = (x_km <= at_km) & (x_km > at_km - dx_km)

    def med(m):
        return float(np.median(field[m])) if m.any() else float("nan")

    return {"km": float(at_km), "east": med(east), "west": med(west),
            "n_east": int(east.sum()), "n_west": int(west.sum())}


def rate_area(base_ye: np.ndarray, delta_ye: np.ndarray | None, thr: float) -> int:
    """Cells (1 km² each) whose rate over the final 12 months exceeds ``thr`` cm/yr."""
    s = base_ye if delta_ye is None else base_ye + delta_ye
    return int(((s[:, -1] - s[:, -2]) > thr).sum())


def abs_limits(field: np.ndarray, lo_q: float = 2.0, hi_q: float = 98.0) -> list[float]:
    """Colour limits for an absolute map: p2-p98, not the extremes."""
    v = np.asarray(field, dtype="float64")
    v = v[np.isfinite(v)]
    return [float(np.percentile(v, lo_q)), float(np.percentile(v, hi_q))]


def band_from_std(mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """(4, ...) p10, p25, p75, p90 under a normal approximation of the member spread."""
    return np.stack([mean - Z10 * std, mean - Z25 * std, mean + Z25 * std, mean + Z10 * std])


def active_index_grid(mask: np.ndarray) -> np.ndarray:
    """(ny, nx) -> active cell index, -1 outside the fan. Row 0 is the southern edge."""
    g = -np.ones(mask.shape, dtype="int64")
    r, c = np.nonzero(mask)
    g[r, c] = np.arange(len(r))
    return g


def bilinear_weights(x: np.ndarray, y: np.ndarray, x0: float, y0: float, dx: float,
                     mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bilinear weights on cell centres -> ``(idx (N,4), w (N,4), ok (N,))``.

    Inactive corners get zero weight and the rest are renormalised; a point with no active
    corner is not ``ok``.
    """
    g = active_index_grid(mask)
    ny, nx = mask.shape
    fx = (np.asarray(x) - x0) / dx - 0.5
    fy = (np.asarray(y) - y0) / dx - 0.5
    i0 = np.floor(fx).astype(int)
    j0 = np.floor(fy).astype(int)
    tx, ty = fx - i0, fy - j0
    idx = -np.ones((len(fx), 4), dtype="int64")
    w = np.zeros((len(fx), 4))
    for n, (di, dj, ww) in enumerate([(0, 0, (1 - tx) * (1 - ty)), (1, 0, tx * (1 - ty)),
                                      (0, 1, (1 - tx) * ty), (1, 1, tx * ty)]):
        ii, jj = i0 + di, j0 + dj
        inside = (ii >= 0) & (ii < nx) & (jj >= 0) & (jj < ny)
        cell = np.where(inside, g[np.clip(jj, 0, ny - 1), np.clip(ii, 0, nx - 1)], -1)
        idx[:, n] = cell
        w[:, n] = np.where(cell >= 0, ww, 0.0)
    tot = w.sum(1)
    ok = tot > 1e-9
    w[ok] /= tot[ok, None]
    return idx, w, ok


def sample(field: np.ndarray, idx: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Evaluate bilinear weights on a (A, ...) field -> (N, ...)."""
    f = np.asarray(field)
    safe = np.where(idx >= 0, idx, 0)
    return np.einsum("nk,nk...->n...", w, f[safe])


def angular_distortion(profile_cm: np.ndarray, step_km: float, smooth_km: float = 2.0,
                       window_km: float = 1.0) -> np.ndarray:
    """Differential settlement along a line: ``|Δs| / L`` over ``window_km``, unitless.

    The profile is first smoothed with a ``smooth_km`` moving average, because the 1 km
    model grid does not resolve anything shorter. Output has one value per point (the
    window centred on it, edges clamped).
    """
    s = np.asarray(profile_cm, dtype="float64") / 100.0           # metres
    n = len(s)
    k = max(1, int(round(smooth_km / step_km)))
    if k > 1 and n > k:
        pad = k // 2
        ker = np.ones(k) / k
        s = np.convolve(np.pad(s, (pad, k - 1 - pad), mode="edge"), ker, mode="valid")
    h = max(1, int(round(window_km / step_km / 2)))
    lo = np.clip(np.arange(n) - h, 0, n - 1)
    hi = np.clip(np.arange(n) + h, 0, n - 1)
    dist = np.maximum((hi - lo) * step_km * 1000.0, 1e-9)
    return np.abs(s[hi] - s[lo]) / dist


def zone_mixed(idx: np.ndarray, w: np.ndarray, zone: np.ndarray, step_km: float,
               reach_km: float = 1.5) -> np.ndarray:
    """Points of a sampled line whose value depends on cells of more than one fan zone.

    The column's parameters change abruptly at the zone lines, so the subsidence field
    steps there and any gradient measured across a line is the model's boundary, not the
    ground. A point's angular distortion (:func:`angular_distortion`) reads the bilinear
    stencils of every point within ``reach_km`` along the line (half the 2 km smoothing
    plus half the 1 km window); it is *mixed* when those stencils touch two zones.
    Returns a boolean (N,) array.
    """
    idx = np.asarray(idx)
    w = np.asarray(w)
    zone = np.asarray(zone)
    n = len(idx)
    zn = np.where((idx >= 0) & (w > 0), zone[np.where(idx >= 0, idx, 0)], -1)   # (N, 4)
    k = max(1, int(round(reach_km / max(step_km, 1e-9))))
    out = np.zeros(n, dtype=bool)
    for i in range(n):
        z = zn[max(0, i - k):min(n, i + k + 1)]
        out[i] = len(np.unique(z[z >= 0])) > 1
    return out


def fast_share(fan_delta: np.ndarray, dates, start_year: int, horizon: int = -1) -> dict:
    """How much of a policy's final fan-mean change arrives in its first year.

    ``fan_delta`` (T,) is the monthly fan mean of policy minus baseline (cm), ``dates``
    its month labels. Returns the share of the change at month ``horizon`` reached at
    December of the start year (``dec``), the largest share reached in any month of the
    first twelve (``peak``) and that month (``peak_month``). A share near one means the
    avoided subsidence is mostly a fast response to the higher heads, which in a
    visco-elastic column is largely elastic, so it would come back if pumping resumed.
    """
    d = np.asarray(fan_delta, dtype="float64")
    idx = pd.DatetimeIndex(pd.to_datetime([str(x)[:10] for x in dates]))
    end = d[horizon]
    first = np.nonzero(idx.year == start_year)[0]
    if abs(end) < 1e-12 or not len(first):
        return {"dec": None, "peak": None, "peak_month": None}
    sh = d[first] / end
    j = int(np.argmax(sh))
    return {"dec": float(sh[-1]), "peak": float(sh[j]), "peak_month": str(idx[first[j]])[:7]}


# --------------------------------------------------------------------------------------
# townships and leveling support
# --------------------------------------------------------------------------------------
def township_index(cell_town: pd.Series, A: int) -> tuple[np.ndarray, list[dict]]:
    """Per cell township index; 0 is the pooled unlabelled panel (Changhua half).

    Returns ``(town_idx (A,) uint8, towns)`` with ``towns[i] = {zh, en, pooled}``.
    """
    names = sorted(set(cell_town.dropna().astype(str)))
    towns = [{"zh": POOLED_ZH, "en": POOLED_EN, "pooled": True}]
    towns += [{"zh": n, "en": TOWNSHIP_EN.get(n, n), "pooled": False} for n in names]
    lookup = {n: i + 1 for i, n in enumerate(names)}
    idx = np.zeros(A, dtype="uint8")
    for c, n in cell_town.dropna().items():
        if 0 <= int(c) < A:
            idx[int(c)] = lookup[str(n)]
    return idx, towns


def leveling_support(obs: dict, xy: dict, cent: np.ndarray, dx: float,
                     hind_cm: np.ndarray, dates, years: list[int]) -> dict:
    """Per-benchmark hindcast comparison, like ``explorer3d.validate_against_leveling``.

    ``obs`` is ``leveling.site_subsidence`` output (m, re-zeroed at the first survey),
    ``hind_cm`` the baseline field over the record (A, T) in cm since Jan 2012. Both are
    re-zeroed at the site's first matched survey. Per site: cell, residual at the last
    survey (model minus observed, cm), mean bias, and the observed series placed on the
    model's own datum (``obs + model at the first survey``) at each calendar year in
    ``years`` (NaN without a survey that year), for plotting against the model curve.
    """
    idx = pd.DatetimeIndex(pd.to_datetime([str(d)[:10] for d in dates]))
    cells, res, bias, ser, xs, ys, pairs = [], [], [], [], [], [], []
    for sid, s in obs.items():
        if sid not in xy:
            continue
        d2 = ((cent - np.array(xy[sid])) ** 2).sum(1)
        c = int(np.argmin(d2))
        if np.sqrt(d2[c]) > dx:
            continue
        m = pd.Series(hind_cm[c], index=idx)
        al = m.reindex(s.index, method="nearest", tolerance=pd.Timedelta("45D"))
        ok = al.notna().to_numpy() & np.isfinite(s.to_numpy())
        if ok.sum() < 3:
            continue
        p = al.to_numpy()[ok]
        o = s.to_numpy()[ok] * 100.0
        t = s.index[ok]
        pz, oz = p - p[0], o - o[0]
        cells.append(c)
        res.append(float(pz[-1] - oz[-1]))
        bias.append(float((pz - oz).mean()))
        row = np.full(len(years), np.nan)
        for tt, v in zip(t, oz + p[0], strict=True):
            if tt.year in years:
                row[years.index(tt.year)] = v
        ser.append(row)
        xs.append(float(xy[sid][0]))
        ys.append(float(xy[sid][1]))
        pairs.append((pz, oz))
    return {"cell": np.array(cells, dtype="int64"), "resid": np.array(res),
            "bias": np.array(bias), "series": np.array(ser).reshape(len(cells), len(years)),
            "x": np.array(xs), "y": np.array(ys), "pairs": pairs}


def pooled_r2(pairs: list[tuple[np.ndarray, np.ndarray]]) -> float:
    if not pairs:
        return float("nan")
    p = np.concatenate([a for a, _ in pairs])
    o = np.concatenate([b for _, b in pairs])
    ss = float(((o - o.mean()) ** 2).sum())
    return 1.0 - float(((o - p) ** 2).sum()) / max(ss, 1e-12)


def township_skill(lev: dict, town_idx: np.ndarray, n_towns: int,
                   min_n: int = 5, max_bias_cm: float = 5.0) -> list[dict]:
    """Leveling support per township: n benchmarks, mean bias, pooled R² (n >= ``min_n``),
    and ``low`` when there are fewer than ``min_n`` benchmarks, ``|bias| > max_bias_cm``,
    or a negative R² (the hindcast does worse there than the benchmarks' own mean, which is
    what "poorly matched" means; it is how the 林內 Linnei apex hotspot is caught).
    """
    out = []
    site_town = town_idx[lev["cell"]] if len(lev["cell"]) else np.zeros(0, dtype=int)
    for t in range(n_towns):
        sel = np.nonzero(site_town == t)[0]
        n = int(len(sel))
        b = float(np.mean(lev["bias"][sel])) if n else float("nan")
        r2 = pooled_r2([lev["pairs"][i] for i in sel]) if n >= min_n else float("nan")
        low = (n < min_n or (np.isfinite(b) and abs(b) > max_bias_cm)
               or (np.isfinite(r2) and r2 < 0.0))
        out.append({"n": n, "bias": None if not np.isfinite(b) else round(b, 2),
                    "r2": None if not np.isfinite(r2) else round(r2, 3), "low": bool(low)})
    return out


# --------------------------------------------------------------------------------------
# colour check
# --------------------------------------------------------------------------------------
_DEUTAN = np.array([[0.367322, 0.860646, -0.227968],        # Machado et al. 2009, sev. 1
                    [0.280085, 0.672501, 0.047413],
                    [-0.011820, 0.042940, 0.968881]])


def _lin(c):
    c = np.asarray(c, dtype="float64") / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _lab(lin):
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]])
    xyz = lin @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], -1)


def deltaE_deutan(c1, c2) -> float:
    """CIE76 ΔE between two sRGB colours as a deuteranope sees them."""
    a = np.clip(_lin(c1) @ _DEUTAN.T, 0, 1)
    b = np.clip(_lin(c2) @ _DEUTAN.T, 0, 1)
    return float(np.linalg.norm(_lab(a) - _lab(b)))
