"""Shared, frozen helpers of FM.origin_memory.v1 (round-7 rank 2) and
FH.forcing_sensitivity.v1 (round-7 rank 7): the key-whitelisted reader of forward-run npz
files, the frozen well screens, the per-lead difference curves, the 'stays below' and 'first
exceeds' rules, the e-folding time and the registration checks.

Model output only. No observation array of any family is read here: every npz is opened
through :class:`WhitelistedNpz`, which refuses any key outside the declared whitelist (the
forward npz also holds ``hindcast_r2``, an in-sample score against observations, and ``gate``,
the theta's stored gate statistics; neither may ever be loaded). Both declarations list this
file in their ``frozen_code``, so any edit after registration refuses both scorers.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
X = "dataprep_staging/fm_origin_memory"

FLOOR_M = 0.05              # the 'stays below' floor (m), FS's lever floor
PCTL = 90.0                 # per-screen percentile (numpy.percentile, linear)
EFOLD_FIT_MIN_M = 1e-4      # leads with |D| below this are left out of the log-linear fit
N_ACTIVE = 2148             # active cells of the twin grid

# keys a scorer may load from a forward npz. Everything else (hindcast_r2: in-sample score
# against observations; gate: the theta's stored gate statistics; heads_std, subs_*:
# not needed) is refused by WhitelistedNpz
FM_KEYS = ("heads_mean", "dates", "scenario_names", "mask", "origin")
# FH also checks the recipe of its own new forwards (metadata only, no value derived from
# any observation)
FH_META_KEYS = ("horizon", "gain", "hindcast_gain", "n_members", "member_labels",
                "forward_options", "scenarios")
NEVER_KEYS = ("hindcast_r2", "gate")

PAGE_MEMBERS = ("stage3_datum_gate", "fold0", "fold1", "fold2", "fold3", "fold4",
                *(f"foldpost{i}" for i in range(12)))

# frozen well screens: N2's 158 registered stations (84 Yunlin, 74 Changhua) as cells and
# layers (0-based) of the twin grid; the same rows as dataprep_staging/x5_class_share
# (wells_frozen.json, station ids dropped here); SCREENS_SHA256 below; the same
# tuples as dataprep_staging/fs_start/score_fs.py (FS.start_invariance.v1)
CELLS = (
    2018, 2018, 1994, 1994, 1994, 2079, 2079, 2107, 2107, 2107, 2107, 2139, 2139, 2139, 1896,
    1896, 1896, 1896, 1954, 1934, 1954, 1934, 1954, 1954, 1749, 1749, 1749, 1723, 1723, 1740,
    1740, 1740, 1401, 1401, 1900, 1900, 1900, 1900, 1535, 1535, 1671, 1671, 1671, 1449, 1449,
    1545, 1545, 1529, 1529, 1522, 1522, 1522, 1893, 1893, 1633, 1633, 1633, 1305, 1305, 1407,
    1407, 1407, 1407, 1421, 1244, 1244, 1393, 1393, 1393, 1354, 1287, 1287, 1199, 1250, 686,
    686, 686, 686, 694, 694, 694, 378, 378, 814, 814, 632, 1133, 1133, 864, 864, 864, 537, 537,
    361, 541, 541, 541, 112, 112, 133, 514, 514, 514, 340, 329, 329, 329, 280, 280, 871, 871,
    1035, 1038, 990, 990, 1125, 1125, 1128, 967, 1064, 1056, 1056, 902, 902, 902, 895, 616,
    616, 653, 750, 750, 178, 178, 357, 357, 358, 201, 235, 235, 235, 255, 165, 165, 346, 346,
    346, 346, 353, 353, 353, 353, 353, 37, 37, 37, 37, 69, 8,
)
LAYERS = (
    1, 2, 0, 1, 2, 1, 2, 0, 1, 1, 2, 0, 1, 2, 0, 1, 1, 2, 0, 0, 1, 1, 1, 3, 0, 1, 1, 1, 1, 1,
    1, 2, 0, 2, 1, 1, 2, 3, 1, 1, 1, 1, 2, 0, 2, 1, 1, 0, 2, 0, 2, 2, 1, 1, 1, 1, 2, 1, 1, 1,
    1, 2, 2, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 1, 0, 1, 0, 3, 2, 0, 1, 0, 1,
    2, 0, 2, 1, 1, 1, 2, 1, 1, 3, 0, 1, 1, 0, 1, 1, 2, 0, 1, 0, 1, 0, 0, 0, 1, 2, 2, 1, 1, 0,
    1, 1, 1, 1, 2, 1, 1, 2, 3, 1, 2, 1, 1, 1, 1, 2, 3, 1, 1, 3, 0, 1, 1, 0, 1, 1, 2, 0, 1, 1,
    2, 3, 1, 1, 2, 3, 1, 1,
)
COUNTY = (
    "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCYYYYYY"
    "YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY"
)
SCREENS_SHA256 = "d66d3daaaacff49d1cedff68a47af82394d20046bf124c53049c3958ea693d1e"


class EngineeringError(RuntimeError):
    """A file, key or registration is not what the declaration names; nothing is scored."""


def _screens_sha() -> str:
    rows = [{"cell": c, "layer": k, "county": "changhua" if q == "C" else "yunlin"}
            for c, k, q in zip(CELLS, LAYERS, "".join(COUNTY), strict=True)]
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":"))
                          .encode()).hexdigest()


def masks() -> dict[str, np.ndarray]:
    """Boolean masks over the 158 screens: all, yunlin (the N2 unit's 84), changhua (74)."""
    c = np.array(list("".join(COUNTY))) == "C"
    return {"all": np.ones(len(CELLS), dtype=bool), "yunlin": ~c, "changhua": c}


def static_checks() -> None:
    if _screens_sha() != SCREENS_SHA256:
        raise SystemExit("frozen screen list was edited (screens sha mismatch)")
    if len(CELLS) != 158 or "".join(COUNTY).count("C") != 74:
        raise SystemExit("frozen screen list is not 158 screens / 74 Changhua")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rp(s: str | Path) -> Path:
    """A path argument, relative to the repository root unless absolute."""
    return Path(s) if Path(s).is_absolute() else REPO / s


def month_index(first: str, month: str) -> int:
    """0-based index of ``YYYY-MM`` ``month`` in a monthly series starting at ``first``."""
    y0, m0 = map(int, first.split("-"))
    y, m = map(int, month.split("-"))
    return (y - y0) * 12 + (m - m0)


def lead_month(origin_month: str, lead: int) -> str:
    """``YYYY-MM`` of lead ``lead`` (1 = the month after ``origin_month``)."""
    y, m = map(int, origin_month.split("-"))
    k = y * 12 + (m - 1) + lead
    return f"{k // 12:04d}-{k % 12 + 1:02d}"


# ----------------------------------------------------------------------------- the reader
class WhitelistedNpz:
    """A forward npz opened for the declared keys only. Listing member names is metadata;
    loading any key outside ``allowed`` raises (an explicit check, not an ``assert``, so it
    survives ``python -O``). Every key loaded is recorded in ``keys_read``."""

    def __init__(self, path: Path, allowed: tuple[str, ...]):
        bad = set(allowed) & set(NEVER_KEYS)
        if bad:
            raise EngineeringError(f"whitelist may never contain {sorted(bad)}")
        self.path = Path(path)
        self.allowed = frozenset(allowed)
        self.keys_read: list[str] = []
        self._z = np.load(self.path, allow_pickle=False)
        self.names = tuple(self._z.files)

    def __getitem__(self, key: str):
        if key not in self.allowed:
            raise EngineeringError(f"{self.path.name}: key {key!r} is not whitelisted "
                                   f"({sorted(self.allowed)})")
        if key not in self.names:
            raise EngineeringError(f"{self.path.name}: no key {key!r}")
        self.keys_read.append(key)
        return self._z[key]

    def close(self) -> None:
        self._z.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def load_run(path: Path, *, allowed: tuple[str, ...], names: tuple[str, ...],
             first: str, last: str, origin_month: str) -> dict:
    """Engineering checks on one forward npz (scenario names, monthly dates, origin, grid
    mask, heads_mean shape, finiteness), then the heads at the 158 frozen screens
    (``h``: scenario x screen x month, float64) and the fan layer-2 mean (``fanL2``:
    scenario x month). Only whitelisted keys are read."""
    e = []
    with WhitelistedNpz(path, allowed) as z:
        sn = tuple(str(s) for s in z["scenario_names"])
        if sn != names:
            e.append(f"scenario_names {sn} != {names}")
        dates = [str(s)[:7] for s in z["dates"]]
        n_t = month_index(first, last) + 1
        if len(dates) != n_t or dates[0] != first or dates[-1] != last:
            e.append(f"dates {dates[0] if dates else None}..{dates[-1] if dates else None} "
                     f"({len(dates)}), expected {first}..{last} ({n_t})")
        oi = month_index(first, origin_month)
        if int(z["origin"]) != oi:
            e.append(f"origin index {int(z['origin'])} != {oi} ({origin_month})")
        n_act = int(np.asarray(z["mask"]).astype(bool).sum())
        if n_act != N_ACTIVE:
            e.append(f"grid mask has {n_act} active cells, expected {N_ACTIVE}")
        hm = z["heads_mean"]
        if hm.shape != (len(names), 4, N_ACTIVE, n_t):
            e.append(f"heads_mean shape {hm.shape}")
        if e:
            raise EngineeringError(f"{path}: " + "; ".join(e))
        cells, layers = np.asarray(CELLS), np.asarray(LAYERS)
        h = np.asarray(hm[:, layers, cells, :], dtype=np.float64)
        fan = np.asarray(hm[:, 1, :, :], dtype=np.float64).mean(axis=1)
        if not (np.isfinite(h).all() and np.isfinite(fan).all()):
            raise EngineeringError(f"{path}: non-finite heads at the screens or in layer 2")
        meta = {k: z[k] for k in allowed if k not in FM_KEYS and k in z.names}
        keys = list(z.keys_read)
        full = hm
    return {"path": str(path), "sha256": sha256(path), "names": sn, "dates": dates,
            "origin": oi, "h": h, "fanL2": fan, "meta": meta, "keys_read": keys,
            "_heads": full}


def drop_full(d: dict) -> dict:
    """The run without its full head array (kept only for the E1 field comparison)."""
    return {k: v for k, v in d.items() if k != "_heads"}


# ----------------------------------------------------------------------------- curves
def diff(a: dict, b: dict, scen: str, i0: int, n_leads: int) -> np.ndarray:
    """(158, n_leads): a - b in scenario ``scen`` at leads 1..n_leads after index ``i0``
    (both runs indexed from 2012-01)."""
    sa, sb = a["names"].index(scen), b["names"].index(scen)
    return a["h"][sa, :, i0 + 1:i0 + 1 + n_leads] - b["h"][sb, :, i0 + 1:i0 + 1 + n_leads]


def fan_diff(a: dict, b: dict, scen: str, i0: int, n_leads: int) -> np.ndarray:
    sa, sb = a["names"].index(scen), b["names"].index(scen)
    return a["fanL2"][sa, i0 + 1:i0 + 1 + n_leads] - b["fanL2"][sb, i0 + 1:i0 + 1 + n_leads]


def p90_abs(d: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Per lead, the 90th percentile over screens of |d| (numpy.percentile, linear)."""
    x = np.abs(d if mask is None else d[mask])
    return np.percentile(x, PCTL, axis=0)


def unit_abs(d: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per lead, |mean over the masked screens of d| (the N2-unit aggregate when the mask
    is Yunlin's 84)."""
    return np.abs(d[mask].mean(axis=0))


def stays_below(curves: list[np.ndarray], floor: float = FLOOR_M) -> int | None:
    """The first lead L (1-based) such that every curve is strictly below ``floor`` at L and
    at every later lead of the curves; None if the last lead is not below."""
    below = np.all(np.stack([np.asarray(c) < floor for c in curves]), axis=0)
    if below.size == 0 or not below[-1]:
        return None
    k = len(below)
    while k > 0 and below[k - 1]:
        k -= 1
    return k + 1


def first_exceeds(f: np.ndarray, m: np.ndarray) -> int | None:
    """The first lead L (1-based) with f(L) > m(L) (strict); None if never."""
    idx = np.flatnonzero(np.asarray(f) > np.asarray(m))
    return int(idx[0]) + 1 if idx.size else None


def stays_above(f: np.ndarray, m: np.ndarray) -> int | None:
    """The first lead L after which f > m at every later lead; None if not at the last."""
    above = np.asarray(f) > np.asarray(m)
    if above.size == 0 or not above[-1]:
        return None
    k = len(above)
    while k > 0 and above[k - 1]:
        k -= 1
    return k + 1


def efold(a: np.ndarray) -> dict:
    """e-folding of a decaying magnitude curve ``a`` (leads 1..n), in months after lead 1:
    ``tau_1e``, where |a| first falls to |a(1)|/e (log-linear interpolation between the
    two leads around the crossing), and ``tau_fit``, -1/slope of the least-squares line of
    ln|a| on the lead over the leads with |a| >= EFOLD_FIT_MIN_M (None if fewer than 2
    such leads or a non-negative slope)."""
    a = np.abs(np.asarray(a, dtype=np.float64))
    out: dict = {"a1_m": float(a[0]) if a.size else None, "tau_1e_months": None,
                 "tau_fit_months": None, "fit_leads": 0}
    if a.size == 0 or a[0] <= 0:
        return out
    target = a[0] / math.e
    hit = np.flatnonzero(a <= target)
    if hit.size:
        i = int(hit[0])
        lo, hi = math.log(a[i - 1]), math.log(max(a[i], 1e-300))
        frac = (lo - math.log(target)) / (lo - hi) if lo != hi else 1.0
        out["tau_1e_months"] = float((i - 1) + frac)
    keep = np.flatnonzero(a >= EFOLD_FIT_MIN_M)
    out["fit_leads"] = int(keep.size)
    if keep.size >= 2:
        slope = float(np.polyfit(keep.astype(np.float64), np.log(a[keep]), 1)[0])
        out["fit_slope_per_month"] = slope
        if slope < 0:
            out["tau_fit_months"] = -1.0 / slope
    return out


def at_leads(curve: np.ndarray, leads=(1, 3, 6, 12, 24, 36, 48)) -> dict:
    return {str(k): float(curve[k - 1]) for k in leads if k <= len(curve)}


# ----------------------------------------------------------------------------- registration
def load_registered(prereg_id: str, prereg_dir: Path | None = None,
                    rows_dir: Path | None = None) -> tuple[dict, str]:
    """The registered declaration, verified against INDEX.jsonl and the ledger; every
    declared single-file input is re-hashed (prereg3.load_prereg)."""
    sys.path.insert(0, str(REPO / "dataprep_staging"))
    from harness import prereg3

    kw = {}
    if prereg_dir is not None:
        kw["prereg_dir"] = prereg_dir
    if rows_dir is not None:
        kw["rows_dir"] = rows_dir
    try:
        return prereg3.load_prereg(prereg_id, **kw)
    except prereg3.PreregistrationError as err:
        raise EngineeringError(f"registration: {err}") from err


def check_frozen(doc: dict, frozen: tuple[str, ...], repo: Path = REPO) -> dict:
    """sha256 of the frozen files against the declaration's ``frozen_code``."""
    want = doc.get("frozen_code") or {}
    got = {p: sha256(repo / p) if (repo / p).exists() else "missing" for p in frozen}
    bad = {p: {"registered": want.get(p), "now": got[p]} for p in frozen
           if want.get(p) != got[p]}
    if set(want) != set(frozen):
        bad["_list"] = {"registered": sorted(want), "now": sorted(frozen)}
    if bad:
        raise EngineeringError(f"frozen code differs from the registered declaration: {bad}")
    return got


def check_sha(path: Path, want: str, what: str) -> str:
    got = sha256(path)
    if got != want:
        raise EngineeringError(f"{what} {path} has sha256 {got[:16]}..., the declaration "
                               f"names {want[:16]}...")
    return got
